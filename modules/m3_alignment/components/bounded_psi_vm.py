"""Trusted manager stages keys inside their owner VM; never exports PSI shares."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import yaml
import functional_vm as vm  # type: ignore[import-not-found]

NOTEBOOK = "S3.P2_bounded_psi.ipynb"
CODE = ("functional_training.py", "functional_metrics.py", "functional_tls.py", "bounded_psi.py")
CELL_COUNT = 3


def stage(workspace, source, local, module):
    from bounded_psi import sha  # type: ignore[import-not-found]
    m5 = module.parent / "m5_modeling"
    dataset = source.parent.name
    source_cfg = "uci_bank_preparation.yaml" if dataset == "uci_bank_marketing" else "hillstrom_preparation.yaml"
    data_cfg = yaml.safe_load((module.parent / "m1_data_selection/configs" / source_cfg).read_text())
    raw = workspace / "数据集" / data_cfg["source"]["relative_path"]
    with raw.open() as stream:
        rows = sum(1 for _ in csv.reader(stream, delimiter=data_cfg["source"]["separator"])) - 1
    run_id = dataset + "_bounded_psi_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = local / dataset / run_id
    addresses = vm.setup_certificates(workspace)
    cfg = module / "configs/bounded_psi.yaml"
    for party in vm.PARTIES:
        own = target / party
        for sub in ("code", "results", "logs"):
            (own / sub).mkdir(parents=True)
        vm.restrict_network(party, addresses)
        root = f"/srv/vfl/{party}/{run_id}"
        old = f"/srv/vfl/{party}/{source.name}"
        vm.guest(party, f"sudo install -d -m 755 {root} {root}/code; sudo install -d -m 550 -o root -g {party} {root}/input; sudo install -d -m 700 -o {party} -g {party} {root}/data {root}/results {root}/logs {root}/trainings")
        # Only the owner VM reads its old CSV. Do not copy features or outcomes.
        extraction = f"from pathlib import Path; import csv,json,hashlib; p=Path('{old}/input'); k=[r['record_id'] for s in ('train','validation') for r in csv.DictReader((p/(s+'.csv')).open())]; q=Path('{root}/input'); x=q/'keys.json'; x.write_text(json.dumps(k)); (q/'manifest.json').write_text(json.dumps({{'keys.json':hashlib.sha256(x.read_bytes()).hexdigest()}}))"
        vm.guest(party, "sudo /opt/secretflow/bin/python -c " + shlex.quote(extraction))
        vm.guest(party, f"sudo find {root}/input -type f -exec chown root:{party} {{}} \\; -exec chmod 440 {{}} \\;")
        sources = [*(m5 / "components" / n for n in CODE[:-1]), module / "components/bounded_psi.py", cfg, module / "notebooks" / NOTEBOOK]
        for path in sources:
            dest = own / "code" / path.name
            dest.write_bytes(path.read_bytes())
            temp = f"/tmp/{run_id}_{path.name}"
            vm.copy_to(dest, party, temp)
            vm.guest(party, f"sudo install -o root -g root -m 444 {temp} {root}/code/{path.name}; sudo rm {temp}")
        runtime = {"party": party, "run_id": run_id, "dataset": dataset, "config": f"{root}/code/{cfg.name}",
                   "config_sha256": sha(cfg), "code_sha256": {n: sha(own / "code" / n) for n in CODE},
                   "source_sha256": sha(raw), "key_namespace": dataset, "universe_rows": rows, "notebook": NOTEBOOK,
                   "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "parties": {p: {"address": f"lima-hsbc-{p}.internal:{vm.FED_PORT}", "listen_addr": f"0.0.0.0:{vm.FED_PORT}"} for p in vm.PARTIES},
                   "spu_addresses": {p: f"lima-hsbc-{p}.internal:{vm.SPU_PORT}" for p in vm.PARTIES}}
        vm.save(own / "code/runtime.json", runtime)
        temp = f"/tmp/{run_id}_runtime.json"
        vm.copy_to(own / "code/runtime.json", party, temp)
        vm.guest(party, f"sudo install -o root -g root -m 444 {temp} {root}/code/runtime.json; sudo rm {temp}")
    vm.private_permissions(target)
    return target, run_id


def execute(target, run_id):
    processes, logs = [], []
    environment = os.environ.copy()
    environment["LIMA_HOME"] = vm.LIMA_HOME_PATH
    try:
        for party in vm.PARTIES:
            root = f"/srv/vfl/{party}/{run_id}"
            script = f"import os,sys; os.chdir('{root}'); sys.path.insert(0,'{root}/code'); import functional_training as ft; ft.execute_notebook('{root}/code/runtime.json')"
            log = (target / party / "logs/driver.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(["limactl", "shell", vm.VM_NAMES[party], "sudo", "-u", party, "env", "-i", "PATH=/opt/secretflow/bin:/usr/bin:/bin", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "PYTHONDONTWRITEBYTECODE=1", "/opt/secretflow/bin/python", "-I", "-B", "-c", script], env=environment, stdout=log, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + vm.ROUND_TIMEOUT_SECONDS
        while any(p.poll() is None for p in processes):
            if time.monotonic() > deadline or any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("PSI participant failed/timed out; private diagnostics retained")
            time.sleep(vm.CHECK_INTERVAL_SECONDS)
        if any(p.returncode for p in processes):
            raise RuntimeError("PSI participant exit failure")
        for party in vm.PARTIES:
            root = f"/srv/vfl/{party}/{run_id}"
            for name in ("engineering.json", "private_timing.json", "party.executed.ipynb", "NOTEBOOK_COMPLETED.json"):
                temp = f"/tmp/{run_id}_{name}"
                vm.guest(party, f"sudo install -m 600 {root}/results/{name} {temp}; sudo chown conceal:conceal {temp}")
                vm.lima("copy", f"{vm.VM_NAMES[party]}:{temp}", target / party / "results" / name)
                vm.guest(party, f"rm {temp}")
            report = json.loads((target / party / "results/engineering.json").read_text())
            completed = json.loads((target / party / "results/NOTEBOOK_COMPLETED.json").read_text())
            if report["status"] != "passed" or completed["code_cells"] != CELL_COUNT:
                raise ValueError("PSI/Notebook incomplete")
            check = f"set -e; test $(sudo find {root}/trainings -name intersection.share | wc -l) = 5; sudo -u {party} test ! -w {root}/input/keys.json; sudo -u {party} test ! -e {root}/input/test.csv; sudo -u {party} test ! -e /srv/vfl/{'bob' if party == 'alice' else 'alice'}; echo passed"
            if vm.guest(party, check) != "passed":
                raise ValueError("PSI owner/output boundary failed")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for log in logs:
            log.close()
        vm.private_permissions(target)


def queue(workspace, source_queue, local):
    state = {"status": "running", "completed": [], "current": None}
    source = json.loads(source_queue.read_text())
    if source["status"] != "passed" or len(source["completed"]) != 2:
        raise ValueError("Source incomplete")
    local.mkdir(parents=True, exist_ok=False)
    try:
        for path in map(Path, source["completed"]):
            target, run_id = stage(workspace, path, local, Path(__file__).resolve().parents[1])
            state["current"] = str(target)
            vm.save(local / "status.json", state)
            execute(target, run_id)
            state["completed"].append(str(target))
        state.update(status="passed", current=None)
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        vm.save(local / "status.json", state)
        vm.private_permissions(local)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(queue(args.workspace, args.source_queue, args.output))
