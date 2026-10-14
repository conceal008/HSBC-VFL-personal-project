"""Trusted VM provisioning under separate prep UIDs; existing training UIDs untouched."""

from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

MODEL_COMPONENTS = Path(__file__).resolve().parents[2] / "m5_modeling/components"
sys.path.insert(0, str(MODEL_COMPONENTS))
import functional_vm as vm  # noqa: E402
from portfolio_source import provision, sha, feature_spec  # type: ignore[import-not-found] # noqa: E402

NOTEBOOK = "S1.P7_portfolio_preparation.ipynb"
PREP_TIMEOUT_SECONDS = 1800
PROBE_PORT = 22
REPO_PARENT_DEPTH = 3


def guest_id(local):
    return local.parent.name + "_" + local.name


def stage(local, provisioned, dataset, repo):
    module = repo / "modules/m1_data_selection"
    cfg_path = module / "configs/portfolio_freeze.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    spec = cfg["datasets"][dataset]
    for party in vm.PARTIES:
        account = f"vflprep_{party}"
        root = f"/srv/preparation/{party}/{guest_id(local)}"
        vm.guest(party, f"sudo test ! -e {root}")
        # Separate preparer identity prevents the currently running model from reading new inputs.
        vm.guest(
            party,
            f"sudo useradd -m -s /usr/sbin/nologin {account} 2>/dev/null || id {account}; sudo install -d -m 755 /srv/preparation; sudo install -d -m 750 -o root -g {account} /srv/preparation/{party}; sudo install -d -m 755 {root} {root}/code; sudo install -d -m 550 -o root -g {account} {root}/input; sudo install -d -m 700 -o root -g root {root}/sealed; sudo install -d -m 700 -o {account} -g {account} {root}/data {root}/results {root}/logs",
        )
        uid = vm.guest(party, f"id -u {account}")
        vm.guest(
            party,
            f"sudo iptables -D OUTPUT -m owner --uid-owner {uid} -j REJECT --reject-with icmp-admin-prohibited 2>/dev/null || true; sudo iptables -I OUTPUT 1 -m owner --uid-owner {uid} -j REJECT --reject-with icmp-admin-prohibited; sudo ip6tables -C OUTPUT -m owner --uid-owner {uid} -j REJECT --reject-with icmp6-adm-prohibited 2>/dev/null || sudo ip6tables -I OUTPUT 1 -m owner --uid-owner {uid} -j REJECT --reject-with icmp6-adm-prohibited; sudo touch {root}/sealed/probe; sudo chmod 600 {root}/sealed/probe",
        )
        # Positive network probe is verified by trusted setup, before UID-enforced denial.
        vm.guest(
            party,
            f"/opt/secretflow/bin/python -c 'import socket;s=socket.create_connection((\"127.0.0.1\",{PROBE_PORT}),timeout=2);s.close()'",
        )
        own = local / party
        (own / "code").mkdir(parents=True)
        (own / "logs").mkdir()
        (own / "results").mkdir()
        code = [
            module / "components/portfolio_party.py",
            module / "components/external_party.py",
            cfg_path,
            module / "notebooks" / NOTEBOOK,
            *[
                MODEL_COMPONENTS / n
                for n in (
                    "functional_training.py",
                    "functional_metrics.py",
                    "functional_tls.py",
                )
            ],
        ]
        digests = {}
        for source in code:
            target = own / "code" / source.name
            target.write_bytes(source.read_bytes())
            digests[source.name] = sha(target)
            temporary = f"/tmp/{party}_{guest_id(local)}_{source.name}"
            vm.copy_to(target, party, temporary)
            vm.guest(
                party,
                f"sudo install -o root -g root -m 444 {temporary} {root}/code/{source.name}; sudo rm {temporary}",
            )
        inputs = {}
        for split in ("train", "validation"):
            source = provisioned / party / f"{split}.csv"
            inputs[source.name] = sha(source)
            temporary = f"/tmp/{party}_{guest_id(local)}_{source.name}"
            vm.copy_to(source, party, temporary)
            vm.guest(
                party,
                f"sudo install -o root -g {account} -m 440 {temporary} {root}/input/{source.name}; sudo rm {temporary}",
            )
        cold = provisioned / party / "confirmation_raw.csv"
        if cold.exists():
            # Trusted provisioning reads bytes only; the copy has no party-readable path.
            cold.chmod(vm.PRIVATE_FILE_MODE)
            try:
                temporary = f"/tmp/{party}_{guest_id(local)}_sealed"
                vm.copy_to(cold, party, temporary)
            finally:
                cold.chmod(0o000)
            vm.guest(
                party,
                f"sudo install -o root -g root -m 600 {temporary} {root}/sealed/confirmation_raw.csv; sudo rm {temporary}",
            )
        outcomes = [spec["label"]] if party == "alice" else []
        if party == "alice" and spec["treatment"]:
            outcomes.append(spec["treatment"])
        runtime = {
            "party": party,
            "account": account,
            "dataset": dataset,
            "run_id": guest_id(local),
            "root": root,
            "config": root + "/code/portfolio_freeze.yaml",
            "config_sha256": sha(cfg_path),
            "seeds": cfg["seeds"],
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "input_sha256": inputs,
            "code_sha256": digests,
            "feature_spec": feature_spec(spec, party),
            "encoding": {
                "seed": cfg["seeds"]["encoding"],
                **{
                    n: cfg[n]
                    for n in (
                        "hash_dimensions",
                        "wide_dimensions",
                        "wide_threshold",
                        "numeric_clip",
                    )
                },
            },
            "outcomes": outcomes,
            "network_probe_port": PROBE_PORT,
            "network_probe_verified": True,
            "notebook": NOTEBOOK,
        }
        path = own / "code/runtime.json"
        path.write_text(json.dumps(runtime, indent=2) + "\n")
        vm.copy_to(path, party, f"/tmp/{party}_{guest_id(local)}_runtime")
        vm.guest(
            party,
            f"sudo install -m 444 /tmp/{party}_{guest_id(local)}_runtime {root}/code/runtime.json; sudo rm /tmp/{party}_{guest_id(local)}_runtime",
        )
        (own / "logs/guest_root.json").write_text(
            json.dumps({"root": root, "account": account})
        )


def execute(local, party):
    runtime = json.loads((local / party / "code/runtime.json").read_text())
    root = runtime["root"]
    account = runtime["account"]
    code = "import os,sys;os.chdir(sys.argv[1]+'/..');sys.path.insert(0,sys.argv[1]);from portfolio_party import execute_preparation;execute_preparation(sys.argv[2])"
    env = os.environ.copy()
    env["LIMA_HOME"] = vm.LIMA_HOME_PATH
    args = [
        "limactl",
        "shell",
        vm.VM_NAMES[party],
        "sudo",
        "-u",
        account,
        "env",
        "-i",
        "PATH=/opt/secretflow/bin:/usr/bin:/bin",
        "OPENBLAS_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        f"IPYTHONDIR={root}/logs/ipython",
        "/opt/secretflow/bin/python",
        "-I",
        "-B",
        "-c",
        code,
        root + "/code",
        root + "/code/runtime.json",
    ]
    with (local / party / "logs/driver.log").open("x") as log:
        result = subprocess.run(
            args,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=PREP_TIMEOUT_SECONDS,
        )
    # Explicit whitelist: never copy cleaned data, source or preprocessing state to host.
    for sub, name in [
        ("results", "party.executed.ipynb"),
        ("results", "NOTEBOOK_COMPLETED.json"),
        ("results", "engineering.json"),
        ("logs", "isolation.json"),
    ]:
        remote = f"/tmp/{party}_{guest_id(local)}_{name}"
        available = vm.guest(
            party, f"sudo test -f {root}/{sub}/{name} && echo yes || echo no"
        )
        if available != "yes":
            continue
        try:
            vm.guest(
                party,
                f"sudo cp {root}/{sub}/{name} {remote};sudo chmod 600 {remote};sudo chown conceal {remote}",
            )
            vm.lima(
                "copy", f"{vm.VM_NAMES[party]}:{remote}", local / party / sub / name
            )
        finally:
            vm.guest(party, f"sudo rm -f {remote}")
    if result.returncode:
        raise RuntimeError("Owner preparation failed; retain private logs")
    receipt = json.loads((local / party / "results/engineering.json").read_text())
    if receipt != {
        "status": "prepared",
        "party": party,
        "train_only_fit": True,
        "sealed_confirmation_read": False,
        "input_unchanged": True,
        "isolation_checks_passed": True,
        "training_performed": False,
        "physical_isolation": False,
    }:
        raise ValueError("Unauthorized preparation receipt")
    return receipt


def run(workspace):
    workspace = Path(workspace).resolve()
    repo = Path(__file__).resolve().parents[REPO_PARENT_DEPTH]
    cfg_path = repo / "modules/m1_data_selection/configs/portfolio_freeze.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    run_id = "S1P7_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    local = workspace / "全数据集准备" / run_id
    local.mkdir(parents=True, exist_ok=False, mode=vm.PRIVATE_DIRECTORY_MODE)
    receipt = {"status": "running", "completed": [], "pid": os.getpid()}
    vm.save(local / "status.json", receipt)
    try:
        for dataset in cfg["datasets"]:
            dataset_root = local / dataset
            dataset_root.mkdir()
            provisioned = workspace / "全数据集投递" / run_id / dataset
            provision(workspace / "数据集", provisioned, dataset, cfg_path)
            stage(dataset_root, provisioned, dataset, repo)
            results = [execute(dataset_root, p) for p in vm.PARTIES]
            vm.save(
                dataset_root / "verification.json",
                {
                    "status": "prepared",
                    "owners": results,
                    "confirmation_unopened": True,
                    "training_performed": False,
                },
            )
            receipt["completed"].append(str(dataset_root))
            vm.save(local / "status.json", receipt)
        receipt["status"] = "passed"
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        vm.save(local / "status.json", receipt)
        vm.private_permissions(local)
    return local
