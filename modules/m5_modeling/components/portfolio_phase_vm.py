"""Bounded native sessions and owner-only recovery/merge for the frozen portfolio."""

from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import yaml
import functional_vm as vm
import nonlinear_full_vm as export_vm
import portfolio_training_vm as original

MODULE = Path(__file__).resolve().parents[1]
CONTROL = yaml.safe_load((MODULE / "configs/portfolio_phase_control.yaml").read_text())
PHASE_NOTEBOOK = "S5.X2_portfolio_phase.ipynb"
MERGE_NOTEBOOK = "S5.X2_portfolio_merge.ipynb"
CONTROLLERS = ("portfolio_phase_control.py", "portfolio_phase_merge.py")
CELLS = ("A0", "B0")
LOCAL_CELLS = ("L0_LR_0", "L0_tree_0")


def bounded_lima(*args, capture=True):
    environment = dict(os.environ, LIMA_HOME=vm.LIMA_HOME_PATH)
    return subprocess.run(
        ["limactl", *map(str, args)],
        env=environment,
        check=True,
        text=True,
        capture_output=capture,
        timeout=CONTROL["command_timeout_seconds"],
    )


def health():
    environment = dict(os.environ, LIMA_HOME=vm.LIMA_HOME_PATH)
    for party in vm.PARTIES:
        subprocess.run(
            ["limactl", "shell", vm.VM_NAMES[party], "true"],
            env=environment,
            check=True,
            capture_output=True,
            timeout=CONTROL["management_timeout_seconds"],
        )


def read(path):
    return json.loads(Path(path).read_text())


def script(party, text):
    return vm.guest(party, "sudo /opt/secretflow/bin/python -c " + shlex.quote(text))


def write_guest(target, party, run_id, name, payload=None):
    own = target / party / "code" / name
    if payload is not None:
        vm.save(own, payload)
    temporary = f"/tmp/{party}_{run_id}_{name}"
    vm.copy_to(own, party, temporary)
    vm.guest(
        party,
        f"sudo install -o root -g root -m 444 {temporary} /srv/vfl/{party}/{run_id}/code/{name}; sudo rm {temporary}",
    )


def complete_source(source, dataset, seed, cfg):
    """Return complete bilateral cells and complete local block, never partial weights."""
    if source is None:
        return [], []
    source = Path(source)
    if source.parent.name != dataset or not re.fullmatch(
        dataset + r"_portfolio_training_[0-9]{8}T[0-9]+Z", source.name
    ):
        raise ValueError("Recovery source identity invalid")
    names = [f"{c}_seed{seed}" for c in CELLS]
    complete = True
    for party in vm.PARTIES:
        root = f"/srv/vfl/{party}/{source.name}"
        code = (
            f"from pathlib import Path;import json;p=Path({root!r});names={names!r};"
            + (
                "ok=all((p/'trainings'/n/'logs/progress.json').exists() and "
                "json.loads((p/'trainings'/n/'logs/progress.json').read_text()).get('completed_epochs')=="
                f"{cfg['shared']['epochs']} and (p/'trainings'/n/'results/model.share').exists() for n in names);"
            )
        )
        if party == "alice":
            code += "ok=ok and all((p/'trainings'/n/'results/status.json').exists() and json.loads((p/'trainings'/n/'results/status.json').read_text()).get('training')=='passed' for n in names);"
        complete &= (
            script(party, code + "print('complete' if ok else 'partial')") == "complete"
        )
    local = [f"{c}_seed{seed}" for c in LOCAL_CELLS]
    code = f"from pathlib import Path;import json;p=Path('/srv/vfl/alice/{source.name}');names={local!r};"
    code += "print('complete' if all((p/'trainings'/n/'results/status.json').exists() and json.loads((p/'trainings'/n/'results/status.json').read_text()).get('training')=='passed' for n in names) else 'partial')"
    return (names if complete else []), (
        local if script("alice", code) == "complete" else []
    )


def clone_owner(target, run_id, party, sources):
    """Copy and hash frozen owner assets entirely within that owner's VM."""
    root = f"/srv/vfl/{party}/{run_id}"
    runtime = read(target / party / "code/runtime.json")
    receipts, names = {}, []
    for source_run, fit_names in sources.items():
        previous = f"/srv/vfl/{party}/{source_run}"
        old = json.loads(vm.guest(party, f"sudo cat {previous}/code/runtime.json"))
        if (
            old["dataset"] != runtime["dataset"]
            or old["party"] != party
            or old["config_sha256"] != runtime["config_sha256"]
            or old["code_sha256"] != runtime["code_sha256"]
        ):
            raise ValueError("Recovery protocol or original math differs")
        for filename, digest in old["code_sha256"].items():
            if (
                vm.guest(party, f"sudo sha256sum {previous}/code/{filename}").split()[0]
                != digest
            ):
                raise ValueError("Recovery source code changed")
        for split in ("train", "validation"):
            if (
                vm.guest(party, f"sudo sha256sum {previous}/input/{split}.csv").split()[
                    0
                ]
                != vm.guest(party, f"sudo sha256sum {root}/input/{split}.csv").split()[
                    0
                ]
            ):
                raise ValueError("Recovery owner CSV differs")
        destination = (
            f"{root}/recovery_input/{source_run}"
            if len(sources) > 1
            else f"{root}/recovery_input"
        )
        vm.guest(party, f"sudo install -d -m 550 -o root -g {party} {destination}")
        for split in ("train", "validation"):
            vm.guest(
                party,
                f"sudo install -o root -g {party} -m 440 {previous}/data/{split}_local.npz {destination}/{split}_local.npz",
            )
        for name in fit_names:
            if name in names:
                raise ValueError("Duplicate merged fit")
            names.append(name)
            vm.guest(
                party,
                f"sudo cp -a {previous}/trainings/{name} {root}/trainings/{name}; sudo chown -R {party}:{party} {root}/trainings/{name}",
            )
        code = (
            "from pathlib import Path;import hashlib,json;"
            f"p=Path({root!r});old=Path({previous!r});names={fit_names!r};"
            "hashes={n:{str(x.relative_to(p/'trainings'/n)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (p/'trainings'/n).rglob('*') if x.is_file()} for n in names};"
            "assert all(hashlib.sha256((old/'trainings'/n/rel).read_bytes()).hexdigest()==value for n,items in hashes.items() for rel,value in items.items());print(json.dumps(hashes))"
        )
        receipts.update(json.loads(script(party, code)))
    manifest = {
        "party": party,
        "config_sha256": runtime["config_sha256"],
        "reused_fits": names,
        "fit_sha256": receipts,
        "source_runs": list(sources),
    }
    write_guest(target, party, run_id, "recovery_manifest.json", manifest)
    return manifest


def stage(workspace, dataset, run_id, target, seed=None, recovery=None, phases=None):
    original.stage(workspace, dataset, run_id, target)
    cfg = yaml.safe_load((MODULE / "configs/portfolio_training.yaml").read_text())
    secure_names, local_names = (
        complete_source(recovery, dataset, seed, cfg) if seed is not None else ([], [])
    )
    for party in vm.PARTIES:
        own = target / party / "code"
        for name in CONTROLLERS:
            (own / name).write_bytes((MODULE / "components" / name).read_bytes())
            write_guest(target, party, run_id, name)
        notebook = PHASE_NOTEBOOK if seed is not None else MERGE_NOTEBOOK
        (own / notebook).write_bytes((MODULE / "notebooks" / notebook).read_bytes())
        write_guest(target, party, run_id, notebook)
        if phases:
            sources = {
                Path(p).name: [
                    f"{c}_seed{s}"
                    for c in CELLS + (LOCAL_CELLS if party == "alice" else ())
                ]
                for s, p in phases.items()
            }
        elif recovery and (secure_names or (party == "alice" and local_names)):
            sources = {
                Path(recovery).name: secure_names
                + (local_names if party == "alice" else [])
            }
        else:
            sources = {}
        manifest = clone_owner(target, run_id, party, sources)
        runtime = read(own / "runtime.json")
        runtime.update(
            notebook=notebook,
            controller_sha256={n: original.sha(own / n) for n in CONTROLLERS},
            reused_completed_fits=secure_names,
            recovery_source_run=str(recovery) if recovery else None,
        )
        if seed is not None:
            if seed not in cfg["shared"]["model_seeds"]:
                raise ValueError("Unregistered selected seed")
            runtime["selected_model_seed"] = seed
        runtime["owner_reused_fits"] = manifest["reused_fits"]
        write_guest(target, party, run_id, "runtime.json", runtime)
    vm.save(
        target / "phase_declaration.json",
        {
            "selected_seed": seed,
            "phases": {str(s): str(p) for s, p in (phases or {}).items()},
            "recovery_source": str(recovery) if recovery else None,
            "secure_reused": secure_names,
            "local_reused": local_names,
            "new_training": seed is not None,
            "predictions_leave_owner_vm": False,
        },
    )
    vm.private_permissions(target)


def spawn(target, party, run_id):
    root = f"/srv/vfl/{party}/{run_id}"
    log = (target / party / "logs/host_process.log").open("x")
    env = dict(os.environ, LIMA_HOME=vm.LIMA_HOME_PATH)
    process = subprocess.Popen(
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
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return process, log


def wait_native(processes, target):
    deadline = time.monotonic() + CONTROL["session_deadline_seconds"]
    next_probe = time.monotonic()
    while any(p.poll() is None for p in processes):
        if any(p.poll() not in (None, 0) for p in processes):
            raise RuntimeError("Native participant failed")
        now = time.monotonic()
        if now > deadline:
            raise TimeoutError("Frozen session budget exceeded")
        if now >= next_probe:
            health()
            vm.save(
                target / "last_health.json",
                {
                    "both_management_endpoints_alive": True,
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            next_probe = now + CONTROL["health_interval_seconds"]
        time.sleep(CONTROL["poll_seconds"])
    if any(p.returncode != 0 for p in processes):
        raise RuntimeError("Native exit failure")


def verify(target, phases=None):
    target = Path(target)
    state = read(target / "status.json")
    if state["status"] != "passed" or any(x != 0 for x in state["exit_codes"]):
        raise ValueError("Native context incomplete")
    declaration = read(target / "declaration.json")
    phase = read(target / "phase_declaration.json")
    cfg = yaml.safe_load((target / "alice/code/nonlinear_full.yaml").read_text())
    original.validate_config(cfg, declaration["dataset"])
    seeds = cfg["shared"]["model_seeds"] if phases else [phase["selected_seed"]]
    if phases and (set(map(int, phases)) != set(seeds) or len(phases) != len(seeds)):
        raise ValueError("Missing five-seed phase lineage")
    for pattern in (
        "*.npy",
        "*.npz",
        "model.share",
        "owner_groups.json",
        "*.trace.log",
    ):
        if list(target.rglob(pattern)):
            raise ValueError("Forbidden private export")
    for party in vm.PARTIES:
        own = target / party
        runtime = read(own / "code/runtime.json")
        if (
            runtime["config_sha256"] != original.sha(own / "code/nonlinear_full.yaml")
            or runtime["config_sha256"] != declaration["config_sha256"]
        ):
            raise ValueError("Frozen config differs")
        for name, digest in dict(
            runtime["code_sha256"], **runtime["controller_sha256"]
        ).items():
            if original.sha(own / "code" / name) != digest:
                raise ValueError("Source differs")
        if not all(read(own / "logs/preparation_boundary.json").values()):
            raise ValueError("Preparation/sealed capability failure")
        # Bob merge performs no computation; its original five successful sessions
        # provide the native/TLS/partition receipts rather than a fictitious notebook.
        if not phases or party == "alice":
            notebook = read(own / "results/party.executed.ipynb")
            cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
            if [c["execution_count"] for c in cells] != list(
                range(1, original.CELL_COUNT + 1)
            ) or any(o["output_type"] == "error" for c in cells for o in c["outputs"]):
                raise ValueError("Notebook incomplete")
            if read(own / "results/NOTEBOOK_COMPLETED.json")[
                "status"
            ] != "passed" or not all(read(own / "logs/isolation.json").values()):
                raise ValueError("Owner completion/isolation failed")
            partition = read(own / "logs/full_partition.json")
            if (
                not partition["all_rows_retained"]
                or partition["source_rows"] != partition["used_rows"]
                or partition["test_included"]
            ):
                raise ValueError("Full development coverage failed")
        if not phases:
            if not all(read(own / "logs/tls_checks.json").values()):
                raise ValueError("Native TLS negative controls failed")
            boundary = read(own / "logs/output_boundary.json")
            if (
                boundary["old_test_read"]
                or boundary["scores_receiver"] != "alice"
                or not boundary["frozen_input_unchanged"]
                or boundary["model_count"] != len(CELLS)
            ):
                raise ValueError("Session output boundary invalid")
        expected = {
            f"{c}_seed{s}"
            for s in seeds
            for c in CELLS + (LOCAL_CELLS if party == "alice" else ())
        }
        if {p.name for p in (own / "trainings").iterdir()} != expected:
            raise ValueError("Fit inventory incomplete")
        for name in expected:
            folder = own / "trainings" / name
            if (
                not all(
                    (folder / s).is_dir() for s in ("code", "data", "results", "logs")
                )
                or not (folder / "code/environment.lock").exists()
            ):
                raise ValueError("Fit assets missing")
            if any(
                original.sha(folder / "code" / n) != d
                for n, d in runtime["code_sha256"].items()
            ):
                raise ValueError("Fit mathematical source differs")
            refs = read(folder / "data/input_references.json")
            if set(refs["frozen_input"]) != {"train.csv", "validation.csv"} or any(
                "test" in k for k in refs["derived_input"]
            ):
                raise ValueError("Test entered fit")
            if name.startswith(CELLS):
                progress = read(folder / "logs/progress.json")
                if (
                    progress["completed_epochs"] != progress["planned_epochs"]
                    or progress["planned_epochs"] != cfg["shared"]["epochs"]
                ):
                    raise ValueError("Fit unfinished")
            if (
                party == "alice"
                and read(folder / "results/status.json")["training"] != "passed"
            ):
                raise ValueError("Prediction boundary failed")
        if party == "bob" and list(own.rglob("*evaluation.json")):
            raise ValueError("Bob evaluation disclosure")
    if phases:
        for _, path in phases.items():
            verify(Path(path))
        report = read(target / "alice/results/private_feature_evaluation.json")
        if (
            report["model_seeds"] != seeds
            or report["independent_confirmation"]
            or not report["full_prepared_development"]
        ):
            raise ValueError("Aggregate scope invalid")
        for group in (report["models"], report["paired_comparisons"]):
            for metrics in group.values():
                for interval in metrics.values():
                    import math

                    if (
                        not all(
                            math.isfinite(interval[k])
                            for k in ("estimate", "ci_low", "ci_high")
                        )
                        or interval["ci_low"] > interval["ci_high"]
                        or interval["valid_replicates"]
                        != cfg["shared"]["bootstrap_repeats"]
                    ):
                        raise ValueError("Aggregate CI invalid")
        if not all(
            read(target / "alice/logs/phase_merge.json")[k]
            for k in (
                "owner_arrays_exact",
                "no_peer_plaintext_read",
                "five_seeds_complete",
            )
        ):
            raise ValueError("Owner merge lineage failed")
    return {
        "verification": "passed",
        "dataset": declaration["dataset"],
        "config_sha256": declaration["config_sha256"],
        "secure_fits": len(seeds) * len(CELLS),
        "alice_fits": len(seeds) * (len(CELLS) + len(LOCAL_CELLS)),
        "bob_fits": len(seeds) * len(CELLS),
        "full_train_validation": True,
        "independent_confirmation": False,
        "scope": "five_seed_aggregate" if phases else "single_seed_session",
    }


def launch(workspace, output, dataset, seed=None, recovery=None, phases=None):
    vm.lima = bounded_lima
    suffix = f"seed{seed}" if seed is not None else "aggregate"
    run_id = (
        dataset
        + "_portfolio_"
        + suffix
        + "_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    target = Path(output) / dataset / run_id
    target.mkdir(parents=True, exist_ok=False, mode=vm.PRIVATE_DIRECTORY_MODE)
    processes, logs, errors = [], [], []
    started, staged, success = time.monotonic(), False, False
    failure = None
    try:
        stage(Path(workspace), dataset, run_id, target, seed, recovery, phases)
        staged = True
        for party in vm.PARTIES if seed is not None else ("alice",):
            process, log = spawn(target, party, run_id)
            processes.append(process)
            logs.append(log)
        wait_native(processes, target)
        success = True
    except BaseException as exc:
        failure = exc
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=CONTROL["termination_grace_seconds"])
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for log in logs:
            log.close()
        for party in vm.PARTIES:
            try:
                if not success:
                    vm.guest(
                        party, f"sudo pkill -u {party} -f {shlex.quote(run_id)} || true"
                    )
                if staged:
                    export_vm.export(target, run_id, party, failed=not success)
            except Exception as exc:
                errors.append(
                    {
                        "party": party,
                        "type": type(exc).__name__,
                        "operation": "cleanup/export",
                    }
                )
        vm.save(
            target / "status.json",
            {
                "status": "passed" if success and not errors else "failed",
                "exit_codes": [p.returncode for p in processes],
                "cleanup_errors": errors,
                "error_type": type(failure).__name__ if failure else None,
                "wall_seconds": time.monotonic() - started,
            },
        )
        vm.private_permissions(target)
    if failure:
        raise failure
    if errors:
        raise RuntimeError("Owner export incomplete")
    try:
        audit = verify(target, phases)
    except Exception as exc:
        vm.save(
            target / "verification_failure.json",
            {"status": "failed", "type": type(exc).__name__, "reason": str(exc)},
        )
        raise
    vm.save(target / "verification.json", audit)
    (target / "实验日志.md").write_text(
        f"# {run_id}\n\n同一冻结开发分区/原优化数学，单seed会话或五seed本方合并；具体目的/源码/配置/恢复来源见declaration、phase_declaration与code。每模型独立trainings；清洗行数据/矩阵/逐行预测/份额留owner VM。封存未读，真实PSI与新增模型输出攻击未执行。\n"
    )
    vm.private_permissions(target)
    return target


def queue(workspace, output, recovery=None):
    cfg = yaml.safe_load((MODULE / "configs/portfolio_training.yaml").read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=vm.PRIVATE_DIRECTORY_MODE)
    state = {
        "status": "running",
        "pid": os.getpid(),
        "completed": [],
        "current_dataset": None,
        "current_seed": None,
        "phases": {},
    }
    vm.save(output / "status.json", state)
    try:
        for dataset in cfg["datasets"]:
            state.update(current_dataset=dataset, phases={})
            for seed in cfg["shared"]["model_seeds"]:
                state["current_seed"] = seed
                vm.save(output / "status.json", state)
                path = launch(
                    workspace,
                    output,
                    dataset,
                    seed,
                    recovery=(recovery or {}).get(dataset),
                )
                state["phases"][str(seed)] = str(path)
                vm.save(output / "status.json", state)
            state["current_seed"] = None
            vm.save(output / "status.json", state)
            path = launch(workspace, output, dataset, phases=state["phases"])
            state["completed"].append(str(path))
            vm.save(output / "status.json", state)
        state.update(status="passed", current_dataset=None)
    except BaseException as exc:
        state.update(
            status="failed", error_type=type(exc).__name__, error_message=str(exc)
        )
        raise
    finally:
        vm.save(output / "status.json", state)
        vm.private_permissions(output)
    return state
