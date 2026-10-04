"""Stage a new diagnostic round; export neither own weights nor secret shares."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import math
from pathlib import Path
import shlex
import stat
import subprocess
import tarfile
import time

import yaml

import functional_vm as vm  # type: ignore[import-not-found]

NOTEBOOK_NAME = "S5.P3_active_diagnostic.ipynb"
CORE_NAMES = ("functional_training.py", "functional_metrics.py", "functional_tls.py")
MINIMUM_SEEDS = 5
NOTEBOOK_CELLS = 3


def verify_diagnostic(root):
    """Audit exported fingerprints, capability order, permissions and intervals."""
    root = Path(root)

    def require(ok, message):
        if not ok:
            raise ValueError(message)

    def read(path):
        return json.loads(path.read_text())

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    require(read(root / "status.json")["status"] == "passed", "Round failed")
    for path in [root, *root.rglob("*")]:
        require(not path.is_symlink(), "Export link forbidden")
        expected = vm.PRIVATE_DIRECTORY_MODE if path.is_dir() else vm.PRIVATE_FILE_MODE
        require(
            stat.S_IMODE(path.stat().st_mode) == expected, "Private permission failed"
        )
    for pattern in ["*.npz", "model.share", "own_weights.npy", "*.trace.log"]:
        require(not list(root.rglob(pattern)), "Forbidden export")
    cfg = yaml.safe_load((root / "alice/code/active_diagnostic.yaml").read_text())
    shared = cfg["shared"]
    seeds = shared["model_seeds"]
    runtime = read(root / "alice/code/runtime.json")
    if not runtime.get("diagnostic_is_smoke"):
        require(
            len(set(seeds)) == len(seeds) and len(seeds) >= MINIMUM_SEEDS,
            "Missing seeds",
        )
    fits = len(seeds) * len(shared["learning_rates"])
    counts = {}
    for party, multiplier in [("alice", 2), ("bob", 1)]:
        own = root / party
        own_runtime = read(own / "code/runtime.json")
        require(
            sha(Path(root) / party / "code/active_diagnostic.yaml")
            == own_runtime["config_sha256"],
            "Config fingerprint failed",
        )
        require(
            all(
                sha(own / "code" / n) == d
                for n, d in own_runtime["code_sha256"].items()
            ),
            "Code fingerprint failed",
        )
        require(all(read(own / "logs/isolation.json").values()), "Isolation failed")
        require(all(read(own / "logs/tls_checks.json").values()), "TLS failed")
        boundary = read(own / "logs/output_boundary.json")
        require(
            boundary["frozen_input_unchanged"]
            and boundary["bob_feature_input_to_training"] is False
            and boundary["selection_index_disclosed_to_bob"] is False,
            "Output boundary failed",
        )
        capability = read(own / "logs/test_capability.json")
        require(
            capability["all_validation_candidates_complete"]
            and capability["test_was_locked"],
            "Test unlocked early",
        )
        folders = list((own / "trainings").iterdir())
        require(len(folders) == fits * multiplier, "Training count failed")
        counts[party] = len(folders)
        for fit in folders:
            require(
                all(
                    (fit / sub).is_dir() for sub in ["code", "data", "results", "logs"]
                ),
                "Incomplete model directory",
            )
            require((fit / "code/environment.lock").is_file(), "Environment missing")
            require(
                all(
                    sha(fit / "code" / n) == d
                    for n, d in own_runtime["code_sha256"].items()
                ),
                "Model code snapshot changed",
            )
            refs = read(fit / "data/input_references.json")
            require(
                "test_local.npz" not in refs["derived_input"],
                "Training received test data",
            )
            if fit.name.startswith("active_secure"):
                progress = read(fit / "logs/progress.json")
                require(
                    progress["completed_epochs"]
                    == progress["planned_epochs"]
                    == shared["epochs"],
                    "Epochs missing",
                )
            if party == "alice":
                require(
                    read(fit / "results/status.json")["test_read"] is False,
                    "Candidate used test",
                )
        notebook = read(own / "results/party.executed.ipynb")
        code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        require(
            [c["execution_count"] for c in code] == list(range(1, NOTEBOOK_CELLS + 1)),
            "Notebook order failed",
        )
        require(
            not any(o["output_type"] == "error" for c in code for o in c["outputs"]),
            "Notebook error",
        )
        require(
            read(own / "results/NOTEBOOK_COMPLETED.json")["status"] == "passed",
            "Notebook incomplete",
        )
        if party == "bob":
            require(
                not list(own.rglob("*predictions.npy"))
                and not (own / "results/private_diagnostic.json").exists(),
                "Bob received scores",
            )
    selection = read(root / "alice/results/frozen_selection.json")
    require(
        selection["selection_split"] == "validation"
        and selection["test_parsed"] is False,
        "Invalid selection",
    )
    require(
        sha(root / "alice/results/frozen_selection.json")
        == read(root / "alice/logs/test_capability.json")["frozen_selection_sha256"],
        "Selection changed after unlock",
    )
    report = read(root / "alice/results/private_diagnostic.json")
    require(
        report["status"] == "completed"
        and {r["seed"] for r in report["seeds"]} == set(seeds),
        "Evaluation seeds failed",
    )
    require(len(report["numerical_diagnostics"]) == fits, "Numerical checks missing")
    for row in report["seeds"]:
        selected = selection["seeds"][str(row["seed"])]["name"]
        require(selected == row["selected_training"], "Selection differs")
        for metrics in [row["metrics"], *[c["metrics"] for c in row["comparisons"]]]:
            for v in metrics.values():
                require(
                    all(math.isfinite(v[k]) for k in ["estimate", "ci_low", "ci_high"])
                    and v["ci_low"] <= v["ci_high"]
                    and v["valid_replicates"] == shared["bootstrap_repeats"],
                    "Invalid interval",
                )
    return {
        "round": root.name,
        "verification": "passed",
        "counts": counts,
        "seeds": seeds,
        "evaluation_sha256": sha(root / "alice/results/private_diagnostic.json"),
        "active_reference_equivalent": report["active_reference_equivalent"],
        "scope": "known cohort diagnostic; not independent confirmation",
    }


def stage(workspace, origin, local, round_id, synthetic_only=False):
    declaration = json.loads((origin / "declaration.json").read_text())
    if json.loads((origin / "status.json").read_text())["status"] != "passed":
        raise ValueError("Origin not completed")
    repo = Path(__file__).resolve().parents[vm.REPO_PARENT_DEPTH]
    sources = repo / "modules/m5_modeling/components"
    request_path = repo / "modules/m5_modeling/configs/active_diagnostic.yaml"
    request = yaml.safe_load(request_path.read_text())
    base = repo / request["base_config"]
    if hashlib.sha256(base.read_bytes()).hexdigest() != request["expected_base_sha256"]:
        raise ValueError("Declared base configuration changed")
    for party in vm.PARTIES:
        for filename in CORE_NAMES:
            if (origin / party / "code" / filename).read_bytes() != (
                sources / filename
            ).read_bytes():
                raise ValueError("Consumed origin source differs")
    smoke = synthetic_only or declaration["prepared_experiment"] == "synthetic_only"
    dataset = declaration["dataset"]
    vm.stage_round(workspace, dataset, round_id, local, smoke)
    for party in vm.PARTIES:
        target = local / party / "code"
        cfg = yaml.safe_load((target / "local_functional.yaml").read_text())
        # Retain the frozen original training schedule, including its synthetic-only overrides.
        if not smoke:
            for key in (
                "model_seeds",
                "learning_rates",
                "epochs",
                "batch_size",
                "infeed_rows",
                "sigmoid",
                "l2_norm",
                "weight_bound",
                "training_boundary_limit",
            ):
                if cfg["shared"][key] != request[key]:
                    raise ValueError("Diagnostic changes original model parameters")
        if smoke:
            cfg["shared"]["top_k_fraction"] = request["smoke_top_k_fraction"]
        cfg["shared"]["bootstrap_seed"] = request["bootstrap_seed"]
        cfg["diagnostic"] = request
        cfg["step_id"] = request["step_id"]
        additions = [
            sources / "active_diagnostic.py",
            repo / "modules/m5_modeling/notebooks" / NOTEBOOK_NAME,
        ]
        for source in additions:
            snapshot = target / source.name
            snapshot.write_bytes(source.read_bytes())
        (target / "active_diagnostic.yaml").write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
        )
        guest_root = f"/srv/vfl/{party}/{round_id}"
        files = [
            "active_diagnostic.py",
            NOTEBOOK_NAME,
            "active_diagnostic.yaml",
        ]
        if party == "alice" and not smoke:
            evaluation = origin / "alice/results/private_evaluation.json"
            old = json.loads(evaluation.read_text())
            paths = [evaluation] + [
                origin
                / "alice/trainings"
                / row["selected_training"]
                / "results/test_predictions.npy"
                for row in old["metrics"]
            ]
            vm.save(
                target / "origin_output_manifest.json",
                {
                    str(p.relative_to(origin / "alice")): hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()
                    for p in paths
                },
            )
            files.append("origin_output_manifest.json")
        for filename in files:
            vm.copy_to(target / filename, party, f"/tmp/{party}_{filename}")
            vm.guest(
                party,
                f"sudo install -m 444 /tmp/{party}_{filename} {guest_root}/code/{filename}; sudo rm /tmp/{party}_{filename}",
            )
        runtime = json.loads((target / "runtime.json").read_text())
        runtime.update(
            origin_run_id=origin.name,
            diagnostic_is_smoke=smoke,
            notebook=NOTEBOOK_NAME,
            config=f"{guest_root}/code/active_diagnostic.yaml",
            config_sha256=hashlib.sha256(
                (target / "active_diagnostic.yaml").read_bytes()
            ).hexdigest(),
            code_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in target.iterdir()
                if p.is_file() and p.name != "runtime.json"
            },
        )
        vm.save(target / "runtime.json", runtime)
        vm.copy_to(target / "runtime.json", party, f"/tmp/{party}_runtime.json")
        vm.guest(
            party,
            f"sudo install -m 444 /tmp/{party}_runtime.json {guest_root}/code/runtime.json; sudo rm /tmp/{party}_runtime.json",
        )
    current = json.loads((local / "declaration.json").read_text())
    current.update(
        step_id=request["step_id"],
        origin_run_id=origin.name,
        origin_status_sha256=hashlib.sha256(
            (origin / "status.json").read_bytes()
        ).hexdigest(),
        purpose="active-only native versus MPC; frozen known-cohort diagnostic; no Bob feature input",
        test_capability="locked until all validation candidates and selections are frozen",
    )
    vm.save(local / "declaration.json", current)


def export_artifacts(local, round_id, party, failed=False):
    root = f"/srv/vfl/{party}/{round_id}"
    archive = f"/tmp/{party}_{round_id}{'_failed' if failed else ''}.tar"
    vm.guest(
        party,
        f"sudo tar --exclude='model.share' --exclude='*.npz' --exclude='own_weights.npy' --exclude='*.trace.log' -cf {shlex.quote(archive)} -C {shlex.quote(root)} code results logs trainings",
    )
    target = (
        local
        / party
        / ("failure_artifacts.tar" if failed else "authorized_artifacts.tar")
    )
    vm.lima("copy", f"{vm.VM_NAMES[party]}:{archive}", target)
    with tarfile.open(target) as bundle:
        bundle.extractall(local / party, filter="data")
    vm.guest(party, f"sudo rm {shlex.quote(archive)}")


def launch(workspace, origin, synthetic_only=False):
    workspace, origin = Path(workspace).resolve(), Path(origin).resolve()
    dataset = json.loads((origin / "declaration.json").read_text())["dataset"]
    round_id = f"{dataset}_active_diagnostic_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    local = workspace / "同优化器诊断" / dataset / round_id
    local.mkdir(parents=True, exist_ok=False, mode=vm.PRIVATE_DIRECTORY_MODE)
    processes, logs = [], []
    passed = False
    try:
        stage(workspace, origin, local, round_id, synthetic_only)
        environment = os.environ.copy()
        environment["LIMA_HOME"] = vm.LIMA_HOME_PATH
        for party in vm.PARTIES:
            root = f"/srv/vfl/{party}/{round_id}"
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
                f"IPYTHONDIR={root}/logs/ipython",
                "/opt/secretflow/bin/python",
                "-I",
                "-B",
                "-c",
                "import os,sys;os.chdir(sys.argv[1]+'/..');sys.path.insert(0,sys.argv[1]);from functional_training import execute_notebook;execute_notebook(sys.argv[2])",
                f"{root}/code",
                f"{root}/code/runtime.json",
            ]
            log = (local / party / "logs/driver.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT
                )
            )
        deadline = time.monotonic() + vm.ROUND_TIMEOUT_SECONDS
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("Participant diagnostic failed; preserve logs")
            if time.monotonic() > deadline:
                raise TimeoutError("Diagnostic timeout")
            time.sleep(vm.CHECK_INTERVAL_SECONDS)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("Participant failed")
        for party in vm.PARTIES:
            export_artifacts(local, round_id, party)
        passed = True
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for log in logs:
            log.close()
        if not passed:
            for party in vm.PARTIES:
                try:
                    vm.guest(
                        party,
                        f"sudo pkill -TERM -u {party} -f {shlex.quote(round_id)} || true",
                    )
                    export_artifacts(local, round_id, party, True)
                except (OSError, subprocess.SubprocessError, tarfile.TarError):
                    vm.save(
                        local / f"{party}_export_failed.json",
                        {"status": "not_collected"},
                    )
        vm.save(
            local / "status.json",
            {
                "status": "passed" if passed else "failed",
                "round_id": round_id,
                "exit_codes": [p.poll() for p in processes],
            },
        )
        (local / "实验日志.md").write_text(
            f"# 同优化器诊断：{round_id}\n\n来源 {origin.name}；状态 {'passed' if passed else 'failed'}。原件不改，实际新增训练和原生参照分别位于各方trainings。数据矩阵/份额/本方权重只留VM；测试只在选参冻结后解锁。数字、偏差与CI仅Alice私有。\n"
        )
        with (workspace / "实验日志/实验索引.md").open("a") as f:
            f.write(
                f"\n- `{round_id}`：S5.P3同优化器诊断 {'passed' if passed else 'failed'}；`{local}/实验日志.md`。\n"
            )
        vm.private_permissions(local)
    if not passed:
        raise RuntimeError(f"Diagnostic failed; {local}")
    return local


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--origin-round", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(launch(args.workspace, args.origin_round, args.smoke))
