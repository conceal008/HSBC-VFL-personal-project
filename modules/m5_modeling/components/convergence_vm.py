"""Trusted manager stages owner-only immutable inputs inside Alice VM; no matrix export."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import yaml
import functional_vm as vm  # type: ignore[import-not-found]
import convergence_audit as audit  # type: ignore[import-not-found]

CODE = ("functional_training.py", "functional_metrics.py", "functional_tls.py",
        "active_diagnostic.py", "nonlinear_full.py", "convergence_audit.py")
NOTEBOOK = "S5.E3_convergence_audit.ipynb"
NOTEBOOK_CELL_COUNT = 3


def source_paths(source_run, seeds):
    if not source_run.startswith(("uci_bank_marketing_nonlinear_full_", "hillstrom_email_marketing_nonlinear_full_")) or not source_run.replace("_", "").isalnum():
        raise ValueError("Invalid source round")
    mapping = {f"{s}_local.npz": f"data/{s}_local.npz" for s in audit.SPLITS}
    for cell in (*audit.CELLS, "B0", "B1"):
        for seed in seeds:
            for split in audit.SPLITS:
                mapping[f"{cell}_seed{seed}_prediction_{split}.npy"] = f"trainings/{cell}_seed{seed}/results/{split}_predictions.npy"
    return mapping


def stage(workspace, source, local, module):
    cfg_path = module / "configs/convergence_audit.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    audit.validate(cfg)
    dataset = source.parent.name
    if dataset not in cfg["datasets"]:
        raise ValueError("Undeclared dataset")
    run_id = dataset + "_convergence_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = local / dataset / run_id
    (target / "code").mkdir(parents=True)
    (target / "results").mkdir()
    (target / "logs").mkdir()
    root = f"/srv/vfl/alice/{run_id}"
    old = f"/srv/vfl/alice/{source.name}"
    mapping = source_paths(source.name, cfg["shared"]["model_seeds"])
    # Width was already declared public to both parties in S5.E2. No Bob values leave his VM.
    width_script = "import numpy as np; a=np.load(" + repr(f"/srv/vfl/bob/{source.name}/data/train_local.npz") + "); print(a['x'].shape[1])"
    width = int(vm.guest("bob", "sudo /opt/secretflow/bin/python -c " + shlex.quote(width_script)))
    vm.guest("alice", "sudo find /srv/vfl/alice -mindepth 1 -maxdepth 1 -type d ! -name tls -exec chown root:root {} \\; -exec chmod 700 {} \\;")
    vm.guest("alice", f"sudo install -d -m 755 {root} {root}/code; sudo install -d -m 550 -o root -g alice {root}/input; sudo install -d -m 700 -o alice -g alice {root}/data {root}/results {root}/logs {root}/trainings")
    # Copy as trusted manager directly guest→guest-directory. Host never receives these bytes.
    for name, path in mapping.items():
        vm.guest("alice", f"sudo install -o root -g alice -m 440 {old}/{path} {root}/input/{name}")
    manifest_script = f"from pathlib import Path; import hashlib,json; p=Path('{root}/input'); m={{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in p.iterdir()}}; (p/'manifest.json').write_text(json.dumps(m))"
    vm.guest("alice", "sudo /opt/secretflow/bin/python -c " + shlex.quote(manifest_script))
    vm.guest("alice", f"sudo chown root:alice {root}/input/manifest.json; sudo chmod 440 {root}/input/manifest.json")
    sources = [*(module / "components" / n for n in CODE), cfg_path, module / "notebooks" / NOTEBOOK]
    for path in sources:
        dest = target / "code" / path.name
        dest.write_bytes(path.read_bytes())
        temp = f"/tmp/{run_id}_{path.name}"
        vm.copy_to(dest, "alice", temp)
        vm.guest("alice", f"sudo install -o root -g root -m 444 {temp} {root}/code/{path.name}; sudo rm {temp}")
    runtime = {"party": "alice", "dataset": dataset, "run_id": run_id,
               "config": f"{root}/code/convergence_audit.yaml", "config_sha256": audit.sha(cfg_path),
               "code_sha256": {n: audit.sha(module / "components" / n) for n in CODE},
               "source_run": source.name, "bob_base_width": width, "notebook": NOTEBOOK,
               "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=module, text=True).strip()}
    vm.save(target / "code/runtime.json", runtime)
    temp = f"/tmp/{run_id}_runtime.json"
    vm.copy_to(target / "code/runtime.json", "alice", temp)
    vm.guest("alice", f"sudo install -o root -g root -m 444 {temp} {root}/code/runtime.json; sudo rm {temp}")
    vm.save(target / "declaration.json", {"step_id": "S5.E3", "source": str(source), "run_id": run_id,
            "test_staged": False, "no_bob_values_copied": True, "config_sha256": audit.sha(cfg_path)})
    vm.private_permissions(local)
    return target, runtime


def execute(target, runtime):
    root = f"/srv/vfl/alice/{runtime['run_id']}"
    script = f"import sys; sys.path.insert(0,'{root}/code'); import functional_training as ft; ft.execute_notebook('{root}/code/runtime.json')"
    vm.guest("alice", "sudo -u alice env -i PATH=/opt/secretflow/bin:/usr/bin:/bin OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /opt/secretflow/bin/python -c " + shlex.quote(script))
    for name in ("private_convergence.json", "engineering.json", "party.executed.ipynb", "NOTEBOOK_COMPLETED.json"):
        temp = f"/tmp/{runtime['run_id']}_{name}"
        vm.guest("alice", f"sudo install -m 600 {root}/results/{name} {temp}; sudo chown conceal:conceal {temp}")
        vm.lima("copy", f"hsbc-alice:{temp}", target / "results" / name)
        vm.guest("alice", f"rm {temp}")
    engineering = json.loads((target / "results/engineering.json").read_text())
    completed = json.loads((target / "results/NOTEBOOK_COMPLETED.json").read_text())
    if engineering["status"] != "passed" or completed["code_cells"] != NOTEBOOK_CELL_COUNT:
        raise ValueError("Engineering/Notebook incomplete")
    verify = f"set -e; test $(sudo stat -c %a {root}/input) = 550; sudo -u alice test ! -w {root}/input/train_local.npz; sudo -u alice test ! -e /srv/vfl/bob; test $(sudo find {root}/trainings -mindepth 1 -maxdepth 1 -type d | wc -l) = 10; sudo -u alice test ! -e {root}/input/test_local.npz; echo passed"
    if vm.guest("alice", verify) != "passed":
        raise ValueError("Owner isolation or fit directory verification failed")
    vm.private_permissions(target)
    return engineering


def queue(workspace, source_queue, local):
    module = Path(__file__).resolve().parents[1]
    source_status = json.loads(source_queue.read_text())
    if source_status["status"] != "passed" or len(source_status["completed"]) != 2:
        raise ValueError("Previous full round incomplete")
    local.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "completed": [], "current": None}
    vm.save(local / "status.json", state)
    try:
        for source in map(Path, source_status["completed"]):
            target, runtime = stage(workspace, source, local, module)
            state["current"] = str(target)
            vm.save(local / "status.json", state)
            execute(target, runtime)
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
    queue(args.workspace, args.source_queue, args.output)
