"""Owner-separated evaluation provider; exports aggregate reports only."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import yaml
import functional_vm as vm  # type: ignore[import-not-found]
import nonlinear_full_vm as provider  # type: ignore[import-not-found]

NOTEBOOK = "S7.P1_output_audit.ipynb"
CODE = (
    "functional_training.py",
    "functional_metrics.py",
    "functional_tls.py",
    "active_diagnostic.py",
    "nonlinear_full.py",
    "bounded_psi.py",
    "output_audit.py",
)
SPLITS = ("train", "validation")
CELLS = ("A0", "A1", "B0", "B1")
CODE_CELLS = 3


def mapping(source, party, seeds):
    if (
        not source.startswith(
            (
                "uci_bank_marketing_nonlinear_full_",
                "hillstrom_email_marketing_nonlinear_full_",
            )
        )
        or not source.replace("_", "").isalnum()
    ):
        raise ValueError("Invalid source run")
    result = {f"{s}_local.npz": f"data/{s}_local.npz" for s in SPLITS}
    if party == "bob":
        result["train.csv"] = "input/train.csv"
    else:
        result.update(
            {
                f"{c}_seed{seed}_{s}.npy": f"trainings/{c}_seed{seed}/results/{s}_predictions.npy"
                for c in CELLS
                for seed in seeds
                for s in SPLITS
            }
        )
    return result


def stage(workspace, source, output):
    repo = Path(__file__).resolve().parents[vm.REPO_PARENT_DEPTH]
    module = repo / "modules/m7_security"
    config = module / "configs/output_audit.yaml"
    cfg = yaml.safe_load(config.read_text())
    dataset = source.parent.name
    if dataset not in cfg["attribute_targets"]:
        raise ValueError("Undeclared data")
    run_id = (
        dataset
        + "_output_audit_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    target = output / dataset / run_id
    addresses = vm.setup_certificates(workspace)
    for party in vm.PARTIES:
        vm.restrict_network(party, addresses)
        own = target / party
        for sub in ("code", "data", "logs", "results"):
            (own / sub).mkdir(parents=True)
        root, old = f"/srv/vfl/{party}/{run_id}", f"/srv/vfl/{party}/{source.name}"
        vm.guest(
            party,
            f"sudo find /srv/vfl/{party} -mindepth 1 -maxdepth 1 -type d ! -name tls -exec chown root:root {{}} \\; -exec chmod 700 {{}} \\;",
        )
        vm.guest(
            party,
            f"sudo install -d -m 755 {root} {root}/code; sudo install -d -m 550 -o root -g {party} {root}/input; sudo install -d -m 700 -o {party} -g {party} {root}/data {root}/results {root}/logs {root}/trainings",
        )
        for name, path in mapping(source.name, party, cfg["model_seeds"]).items():
            vm.guest(
                party,
                f"sudo install -o root -g {party} -m 440 {old}/{path} {root}/input/{name}",
            )
        script = f"from pathlib import Path; import hashlib,json; p=Path('{root}/input'); (p/'manifest.json').write_text(json.dumps({{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir()}}))"
        vm.guest(party, "sudo /opt/secretflow/bin/python -c " + shlex.quote(script))
        vm.guest(
            party,
            f"sudo chown root:{party} {root}/input/manifest.json; sudo chmod 440 {root}/input/manifest.json",
        )
        sources = []
        for name in CODE:
            owner_module = (
                "m7_security"
                if name == "output_audit.py"
                else ("m3_alignment" if name == "bounded_psi.py" else "m5_modeling")
            )
            sources.append(repo / "modules" / owner_module / "components" / name)
        for path in [*sources, config, module / "notebooks" / NOTEBOOK]:
            dest = own / "code" / path.name
            dest.write_bytes(path.read_bytes())
            temp = f"/tmp/{run_id}_{party}_{path.name}"
            vm.copy_to(dest, party, temp)
            vm.guest(
                party,
                f"sudo install -o root -g root -m 444 {temp} {root}/code/{path.name}; sudo rm {temp}",
            )
        runtime = {
            "party": party,
            "run_id": run_id,
            "dataset": dataset,
            "config": f"{root}/code/output_audit.yaml",
            "config_sha256": provider.sha(config),
            "code_sha256": {n: provider.sha(own / "code" / n) for n in CODE},
            "notebook": NOTEBOOK,
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "source_run": source.name,
            "parties": {
                p: {
                    "address": f"lima-hsbc-{p}.internal:{vm.FED_PORT}",
                    "listen_addr": f"0.0.0.0:{vm.FED_PORT}",
                }
                for p in vm.PARTIES
            },
            "spu_addresses": {
                p: f"lima-hsbc-{p}.internal:{vm.SPU_PORT}" for p in vm.PARTIES
            },
        }
        vm.save(own / "code/runtime.json", runtime)
        temp = f"/tmp/{run_id}_{party}_runtime.json"
        vm.copy_to(own / "code/runtime.json", party, temp)
        vm.guest(
            party,
            f"sudo install -o root -g root -m 444 {temp} {root}/code/runtime.json; sudo rm {temp}",
        )
    vm.save(
        target / "declaration.json",
        {
            "step": "S7.P1",
            "source_run": source.name,
            "test_staged": False,
            "prediction_export": False,
            "run_id": run_id,
        },
    )
    vm.private_permissions(target)
    return target, run_id


def verify(target, run_id):
    for party in vm.PARTIES:
        root = f"/srv/vfl/{party}/{run_id}"
        checks = f"set -e; sudo -u {party} test ! -w {root}/input/train_local.npz; test $(sudo stat -c %a {root}/input) = 550; sudo -u {party} test ! -e {root}/input/test_local.npz; echo passed"
        if vm.guest(party, checks) != "passed":
            raise ValueError("Frozen capability denied")
        nb = json.loads((target / party / "results/party.executed.ipynb").read_text())
        cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        if [c["execution_count"] for c in cells] != list(
            range(1, CODE_CELLS + 1)
        ) or any(o["output_type"] == "error" for c in cells for o in c["outputs"]):
            raise ValueError("Notebook incomplete")
        if (
            not json.loads((target / party / "results/engineering.json").read_text())[
                "status"
            ]
            == "passed"
        ):
            raise ValueError("Owner incomplete")
    forbidden = ("*.npy", "*.npz", "model.share", "*.trace.log")
    if any(list(target.rglob(p)) for p in forbidden) or list(
        (target / "bob").rglob("private_*audit.json")
    ):
        raise ValueError("Forbidden export")
    bob = f"/srv/vfl/bob/{run_id}"
    if (
        vm.guest(
            "bob",
            f'set -e; test $(sudo find {bob}/input -name "*.npy" | wc -l) = 0; test $(sudo find {bob}/trainings -name "*predictions*" | wc -l) = 0; echo passed',
        )
        != "passed"
    ):
        raise ValueError("Bob score boundary failed")
    report = json.loads(
        (target / "alice/results/private_attribute_audit.json").read_text()
    )
    cfg = yaml.safe_load((target / "alice/code/output_audit.yaml").read_text())
    if len(report["details"]) != len(cfg["model_seeds"]) * len(("B0", "B1")) * (
        len(("exact", "rounded", "binary")) + 1
    ):
        raise ValueError("Incomplete probes")
    return {
        "status": "passed",
        "bob_scores_disclosed": False,
        "truth_exported": False,
        "test_read": False,
        "production_security": False,
    }


def execute(target, run_id):
    environment = os.environ.copy()
    environment["LIMA_HOME"] = vm.LIMA_HOME_PATH
    processes, logs = [], []
    success = False
    try:
        for party in vm.PARTIES:
            root = f"/srv/vfl/{party}/{run_id}"
            script = f"import os,sys; os.chdir('{root}'); sys.path.insert(0,'{root}/code'); import functional_training as ft; ft.execute_notebook('{root}/code/runtime.json')"
            log = (target / party / "logs/host_process.log").open("w")
            logs.append(log)
            command = [
                "limactl",
                "shell",
                vm.VM_NAMES[party],
                "sudo",
                "-u",
                party,
                "env",
                "-i",
                "PATH=/opt/secretflow/bin:/usr/bin:/bin",
                "OPENBLAS_NUM_THREADS=1",
                "OMP_NUM_THREADS=1",
                "PYTHONDONTWRITEBYTECODE=1",
                "/opt/secretflow/bin/python",
                "-c",
                script,
            ]
            processes.append(
                subprocess.Popen(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT
                )
            )
        started = time.monotonic()
        while any(p.poll() is None for p in processes):
            if (
                any(p.poll() not in (None, 0) for p in processes)
                or time.monotonic() - started > vm.ROUND_TIMEOUT_SECONDS
            ):
                raise RuntimeError("Owner exit or timeout")
            time.sleep(vm.CHECK_INTERVAL_SECONDS)
        if any(p.returncode for p in processes):
            raise RuntimeError("Owner exit")
        success = True
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for log in logs:
            log.close()
        for party in vm.PARTIES:
            if not success:
                vm.guest(
                    party, f"sudo pkill -u {party} -f {shlex.quote(run_id)} || true"
                )
            provider.export(target, run_id, party, failed=not success)
        vm.save(
            target / "status.json",
            {
                "status": "passed" if success else "failed",
                "exit_codes": [p.poll() for p in processes],
            },
        )
        vm.private_permissions(target)
    report = verify(target, run_id)
    vm.save(target / "verification.json", report)
    vm.private_permissions(target)
    return report


def queue(workspace, source_queue, output):
    source = json.loads(source_queue.read_text())
    if source["status"] != "passed" or len(source["completed"]) != len(vm.PARTIES):
        raise ValueError("Source queue incomplete")
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "completed": [], "current": None}
    vm.save(output / "status.json", state)
    try:
        for old in map(Path, source["completed"]):
            target, run_id = stage(workspace, old, output)
            state["current"] = str(target)
            vm.save(output / "status.json", state)
            execute(target, run_id)
            state["completed"].append(str(target))
        state.update(status="passed", current=None)
    except BaseException as exc:
        state.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        vm.save(output / "status.json", state)
        vm.private_permissions(output)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue(args.workspace, args.source_queue, args.output)
