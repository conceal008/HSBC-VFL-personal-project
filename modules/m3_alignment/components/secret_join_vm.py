"""Stage keys in owner VMs and export only a strict engineering allowlist."""

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
from bounded_psi import sha, TRIALS  # type: ignore[import-not-found]

NOTEBOOK = "S3.P3_secret_join.ipynb"
CODE = (
    "functional_training.py",
    "functional_metrics.py",
    "functional_tls.py",
    "bounded_psi.py",
    "secret_join.py",
    "secret_join_run.py",
)
EXPORT_NAMES = (
    "engineering.json",
    "timing.json",
    "party.executed.ipynb",
    "NOTEBOOK_COMPLETED.json",
)
CELL_COUNT = 3
INPUT_FILE_COUNT = 2


def validate_config(cfg):
    if (
        cfg["step_id"] != "S3.P3"
        or cfg["output"] != "secret_shares"
        or len(set(cfg["seeds"])) < len(TRIALS)
        or cfg["rows_per_owner"] < 1
        or not 0 < cfg["keep_probability"] < 1
    ):
        raise ValueError("Unapproved secret output protocol")


def stage(workspace, source, output):
    module = Path(__file__).resolve().parents[1]
    cfg_path = module / "configs/secret_join.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    validate_config(cfg)
    dataset = source.parent.name
    if dataset not in ("uci_bank_marketing", "hillstrom_email_marketing"):
        raise ValueError("Undeclared owner input")
    run_id = (
        dataset
        + "_secret_join_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    target = output / dataset / run_id
    target.mkdir(parents=True, exist_ok=False, mode=vm.PRIVATE_DIRECTORY_MODE)
    addresses = vm.setup_certificates(workspace)
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=module, text=True
    ).strip()
    for party in vm.PARTIES:
        own = target / party
        for sub in ("code", "results", "logs"):
            (own / sub).mkdir(parents=True)
        vm.restrict_network(party, addresses)
        root = f"/srv/vfl/{party}/{run_id}"
        old = f"/srv/vfl/{party}/{source.name}"
        vm.guest(
            party,
            f"sudo find /srv/vfl/{party} -mindepth 1 -maxdepth 1 -type d ! -name tls -exec chown root:root {{}} \\; -exec chmod 700 {{}} \\;",
        )
        vm.guest(
            party,
            f"sudo install -d -m 755 {root} {root}/code; sudo install -d -o root -g {party} -m 550 {root}/input; sudo install -d -o {party} -g {party} -m 700 {root}/results {root}/data {root}/logs {root}/trainings",
        )
        for name in ("keys.json", "manifest.json"):
            vm.guest(
                party,
                f"sudo install -o root -g {party} -m 440 {old}/input/{name} {root}/input/{name}",
            )
        check = f"set -e; sudo -u {party} test ! -r {old}/input/keys.json; sudo -u {party} test ! -w {root}/input/keys.json; sudo -u {party} test ! -e {root}/input/test.csv; test $(sudo find {root}/input -type f | wc -l) = {INPUT_FILE_COUNT}; echo passed"
        if vm.guest(party, check) != "passed":
            raise ValueError("Owner source/output capability failed")
        for name in CODE:
            source_file = (
                module.parent / "m5_modeling/components"
                if name.startswith("functional_")
                else module / "components"
            ) / name
            (own / "code" / name).write_bytes(source_file.read_bytes())
        for path in (cfg_path, module / "notebooks" / NOTEBOOK):
            (own / "code" / path.name).write_bytes(path.read_bytes())
        runtime = {
            "party": party,
            "run_id": run_id,
            "dataset": dataset,
            "config": f"{root}/code/{cfg_path.name}",
            "config_sha256": sha(cfg_path),
            "code_sha256": {n: sha(own / "code" / n) for n in CODE},
            "notebook": NOTEBOOK,
            "source_run": source.name,
            "git_sha": git_sha,
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
        for file in (own / "code").iterdir():
            temp = f"/tmp/{run_id}_{party}_{file.name}"
            vm.copy_to(file, party, temp)
            vm.guest(
                party,
                f"sudo install -o root -g root -m 444 {temp} {root}/code/{file.name}; sudo rm {temp}",
            )
    vm.private_permissions(target)
    return target


def export(target, party):
    root = f"/srv/vfl/{party}/{target.name}"
    for name in EXPORT_NAMES:
        temporary = f"/tmp/{target.name}_{party}_{name}"
        vm.guest(
            party,
            f"sudo install -o conceal -g conceal -m 600 {root}/results/{name} {temporary}",
        )
        vm.lima(
            "copy",
            f"{vm.VM_NAMES[party]}:{temporary}",
            target / party / "results" / name,
        )
        vm.guest(party, f"rm {temporary}")


def verify(target):
    for pattern in ("*.npy", "*.npz", "*.share", "keys.json", "construction.json"):
        if list(target.rglob(pattern)):
            raise ValueError("Forbidden key/vector/share export")
    for party in vm.PARTIES:
        own = target / party
        runtime = json.loads((own / "code/runtime.json").read_text())
        cfg = yaml.safe_load((own / "code/secret_join.yaml").read_text())
        validate_config(cfg)
        if sha(own / "code/secret_join.yaml") != runtime["config_sha256"] or any(
            sha(own / "code" / n) != h for n, h in runtime["code_sha256"].items()
        ):
            raise ValueError("Source/protocol fingerprint failed")
        report = json.loads((own / "results/engineering.json").read_text())
        expected = {
            "known_controls_passed": True,
            "secret_all_pairs_oracle_passed": True,
            "federation_anonymous_denied": True,
            "spu_anonymous_denied": True,
            "intersection_reconstructed": False,
            "membership_or_count_disclosed": False,
            "peer_indices_disclosed": False,
            "test_read": False,
            "native_rr22": False,
            "existing_training_consumes_this_psi": False,
        }
        if (
            report["status"] != "passed"
            or report["seeds"] != cfg["seeds"]
            or any(report.get(k) is not v for k, v in expected.items())
        ):
            raise ValueError("Secret output report incomplete")
        nb = json.loads((own / "results/party.executed.ipynb").read_text())
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        if [c["execution_count"] for c in code] != list(
            range(1, CELL_COUNT + 1)
        ) or any(o["output_type"] == "error" for c in code for o in c["outputs"]):
            raise ValueError("Notebook incomplete")
        root = f"/srv/vfl/{party}/{target.name}"
        count = len(cfg["seeds"])
        check = f"set -e; test $(sudo find {root}/trainings -name membership.share | wc -l) = {count}; sudo -u {party} test ! -w {root}/input/keys.json; sudo -u {party} test ! -e /srv/vfl/{'bob' if party == 'alice' else 'alice'}; echo passed"
        permission_code = (
            "from pathlib import Path;import os,stat,pwd;"
            + f"p=Path('{root}/trainings');uid=pwd.getpwnam('{party}').pw_uid;"
            + "files=list(p.rglob('membership.share'));"
            + f"assert len(files)=={count};"
            + "assert all(x.stat().st_uid==uid and stat.S_IMODE(x.stat().st_mode)==0o600 for x in files);print('passed')"
        )
        if (
            vm.guest(party, check) != "passed"
            or vm.guest(
                party,
                "sudo /opt/secretflow/bin/python -c " + shlex.quote(permission_code),
            )
            != "passed"
        ):
            raise ValueError("Owner shares/permissions failed")
    return {"verification": "passed", "shares_exported": False, "keys_exported": False}


def execute(target):
    processes, logs, errors = [], [], []
    success = False
    env = os.environ.copy()
    env["LIMA_HOME"] = vm.LIMA_HOME_PATH
    try:
        for party in vm.PARTIES:
            root = f"/srv/vfl/{party}/{target.name}"
            script = f"import os,sys;os.chdir('{root}');sys.path.insert(0,'{root}/code');from functional_training import execute_notebook;execute_notebook('{root}/code/runtime.json')"
            log = (target / party / "logs/driver.log").open("x")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [
                        "limactl",
                        "shell",
                        vm.VM_NAMES[party],
                        "sudo",
                        "-u",
                        party,
                        "env",
                        "-i",
                        "PATH=/opt/secretflow/bin:/usr/bin:/bin",
                        "OMP_NUM_THREADS=1",
                        "OPENBLAS_NUM_THREADS=1",
                        "PYTHONDONTWRITEBYTECODE=1",
                        f"IPYTHONDIR={root}/logs/ipython",
                        "/opt/secretflow/bin/python",
                        "-I",
                        "-B",
                        "-c",
                        script,
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        deadline = time.monotonic() + vm.ROUND_TIMEOUT_SECONDS
        while any(p.poll() is None for p in processes):
            if time.monotonic() > deadline or any(
                p.poll() not in (None, 0) for p in processes
            ):
                raise RuntimeError("Owner PSI failed/timed out")
            time.sleep(vm.CHECK_INTERVAL_SECONDS)
        if any(p.returncode for p in processes):
            raise RuntimeError("Owner exit failure")
        for party in vm.PARTIES:
            export(target, party)
        verify(target)
        success = True
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for log in logs:
            log.close()
        if not success:
            for party in vm.PARTIES:
                try:
                    vm.guest(
                        party,
                        f"sudo pkill -u {party} -f {shlex.quote(target.name)} || true",
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "party": party,
                            "operation": "stop",
                            "type": type(exc).__name__,
                        }
                    )
        vm.save(
            target / "status.json",
            {
                "status": "passed" if success else "failed",
                "exit_codes": [p.poll() for p in processes],
                "cleanup_errors": errors,
            },
        )
        vm.private_permissions(target)
    return target


def queue(workspace, output):
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/secret_join.yaml").read_text()
    )
    source = json.loads((workspace / cfg["source_queue"]).read_text())
    if source["status"] != "passed" or len(source["completed"]) != 2:
        raise ValueError("Source queue not verified")
    output.mkdir(parents=True, exist_ok=False, mode=vm.PRIVATE_DIRECTORY_MODE)
    state = {"status": "running", "completed": [], "current": None}
    try:
        for item in source["completed"]:
            target = stage(workspace, Path(item), output)
            state["current"] = str(target)
            vm.save(output / "status.json", state)
            execute(target)
            state["completed"].append(str(target))
        state.update(status="passed", current=None)
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        vm.save(output / "status.json", state)
        vm.private_permissions(output)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(queue(args.workspace, args.output))


def audit_secret_visibility(target):
    """Trusted owner-VM auditor loads each single share; returns only a pass bit.

    Historical rounds are revoked from training UIDs; never reopen their access.
    This checks metadata, not reconstruction, and exports no metadata or contents.
    """
    import re

    target = Path(target)
    if not re.fullmatch(
        r"(uci_bank_marketing|hillstrom_email_marketing)_secret_join_[0-9]{8}T[0-9]+Z",
        target.name,
    ):
        raise ValueError("Unexpected audit round identity")
    for party in vm.PARTIES:
        cfg = yaml.safe_load((target / party / "code/secret_join.yaml").read_text())
        root = f"/srv/vfl/{party}/{target.name}/trainings"
        code = (
            "import pickle;from pathlib import Path;import jax;import spu;"
            + f"files=list(Path('{root}').glob('join_seed*/results/membership.share'));assert len(files)=={len(cfg['seeds'])};"
            + "metas=[jax.tree_util.tree_leaves(pickle.load(f.open('rb'))['meta']) for f in files];"
            + "assert all(len(m)==4 and all(x.vtype==spu.Visibility.VIS_SECRET and x.protocol==spu.ProtocolKind.CHEETAH and x.field==spu.FieldType.FM64 for x in m) for m in metas);print('passed')"
        )
        result = vm.guest(
            party, "sudo /opt/secretflow/bin/python -c " + shlex.quote(code)
        )
        if not result.splitlines() or result.splitlines()[-1] != "passed":
            raise ValueError("Output metadata not verified secret")
    return {
        "all_four_outputs_VIS_SECRET": True,
        "CHEETAH_FM64_verified": True,
        "trusted_owner_vm_audit": True,
        "shares_or_metadata_exported": False,
    }
