"""Stage and audit a synthetic-only two-VM joint diagnostic."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import yaml
import functional_vm as vm  # type: ignore[import-not-found]
import active_vm as av  # type: ignore[import-not-found]

MINIMUM_SEEDS = 5
NOTEBOOK_CELL_COUNT = 3
NOTEBOOK = "S5.V1_joint_numerical.ipynb"
COMPONENTS = (
    "functional_training.py",
    "functional_metrics.py",
    "functional_tls.py",
    "functional_vm.py",
    "active_diagnostic.py",
    "joint_diagnostic.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(workspace, dataset, run_id, local):
    repo = Path(__file__).resolve().parents[vm.REPO_PARENT_DEPTH]
    module = repo / "modules/m5_modeling"
    config = module / "configs/joint_numerical.yaml"
    cfg = yaml.safe_load(config.read_text())
    if (
        cfg.get("synthetic_only") is not True
        or dataset not in cfg["diagnostic"]["datasets"]
    ):
        raise ValueError("Only declared synthetic tasks allowed")
    # Shared environment skeleton, then install this new run's frozen synthetic inputs.
    # No previously executed round or source archive is altered.
    vm.stage_round(workspace, dataset, run_id, local, smoke=True)
    for party in vm.PARTIES:
        code = local / party / "code"
        guest_root = f"/srv/vfl/{party}/{run_id}"
        for name in COMPONENTS:
            (code / name).write_bytes((module / "components" / name).read_bytes())
        (code / NOTEBOOK).write_bytes((module / "notebooks" / NOTEBOOK).read_bytes())
        (code / "joint_numerical.yaml").write_bytes(config.read_bytes())
        frames = vm.synthetic_inputs(cfg, dataset, party)
        manifest = {}
        for split, frame in frames.items():
            source = local / party / f"{split}_synthetic.csv"
            frame.to_csv(source, index=False)
            manifest[f"{split}.csv"] = sha(source)
            vm.copy_to(source, party, f"/tmp/{party}_{run_id}_{split}.csv")
            vm.guest(
                party,
                f"sudo install -o root -g {party} -m 440 /tmp/{party}_{run_id}_{split}.csv {guest_root}/input/{split}.csv",
            )
        source = local / party / "input_manifest.json"
        vm.save(source, manifest)
        vm.copy_to(source, party, f"/tmp/{party}_{run_id}_manifest.json")
        vm.guest(
            party,
            f"sudo install -o root -g {party} -m 440 /tmp/{party}_{run_id}_manifest.json {guest_root}/input/manifest.json",
        )
        runtime = json.loads((code / "runtime.json").read_text())
        runtime.update(
            synthetic_only=True,
            config=f"{guest_root}/code/joint_numerical.yaml",
            notebook=NOTEBOOK,
            config_sha256=sha(config),
            code_sha256={name: sha(code / name) for name in COMPONENTS},
        )
        vm.save(code / "runtime.json", runtime)
        for name in (*COMPONENTS, NOTEBOOK, "joint_numerical.yaml", "runtime.json"):
            vm.copy_to(code / name, party, f"/tmp/{party}_{run_id}_{name}")
            vm.guest(
                party,
                f"sudo install -o root -g root -m 444 /tmp/{party}_{run_id}_{name} {guest_root}/code/{name}",
            )
    declaration = json.loads((local / "declaration.json").read_text())
    declaration.update(
        step_id="S5.V1",
        synthetic_only=True,
        old_test_read=False,
        config_sha256=sha(config),
        scope=cfg["diagnostic"]["scope"],
    )
    vm.save(local / "declaration.json", declaration)
    vm.private_permissions(local)


def verify(root):
    root = Path(root)

    def require(ok, msg):
        if not ok:
            raise ValueError(msg)

    def read(path):
        return json.loads(path.read_text())

    require(read(root / "status.json")["status"] == "passed", "Round not passed")
    declaration = read(root / "declaration.json")
    require(
        declaration["synthetic_only"] is True and declaration["old_test_read"] is False,
        "Scope mismatch",
    )
    cfg = yaml.safe_load((root / "alice/code/joint_numerical.yaml").read_text())
    shared = cfg["shared"]
    require(len(set(shared["model_seeds"])) >= MINIMUM_SEEDS, "Too few seeds")
    fits = len(shared["model_seeds"]) * len(shared["learning_rates"])
    for pattern in ("*.npz", "model.share", "own_weights.npy", "*.trace.log"):
        require(not list(root.rglob(pattern)), "Forbidden export")
    for party in vm.PARTIES:
        own = root / party
        runtime = read(own / "code/runtime.json")
        require(
            sha(own / "code/joint_numerical.yaml")
            == runtime["config_sha256"]
            == declaration["config_sha256"],
            "Config mismatch",
        )
        require(
            all(sha(own / "code" / n) == d for n, d in runtime["code_sha256"].items()),
            "Source mismatch",
        )
        require(all(read(own / "logs/isolation.json").values()), "Isolation failed")
        require(all(read(own / "logs/tls_checks.json").values()), "TLS failed")
        require(
            read(own / "logs/output_boundary.json")["old_test_read"] is False,
            "Old test read",
        )
        notebook = read(own / "results/party.executed.ipynb")
        cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        require([c["execution_count"] for c in cells] == list(range(1, NOTEBOOK_CELL_COUNT + 1)), "Notebook order")
        require(
            not any(o["output_type"] == "error" for c in cells for o in c["outputs"]),
            "Notebook error",
        )
        require(
            read(own / "results/NOTEBOOK_COMPLETED.json")["status"] == "passed",
            "Notebook incomplete",
        )
        directories = list((own / "trainings").iterdir())
        require(len(directories) == fits, "Missing fits")
        for folder in directories:
            require(
                all((folder / s).is_dir() for s in ("code", "data", "results", "logs")),
                "Missing fit directories",
            )
            require(
                all(
                    sha(folder / "code" / n) == d
                    for n, d in runtime["code_sha256"].items()
                ),
                "Fit source mismatch",
            )
            require((folder / "code/environment.lock").is_file(), "Missing environment")
            progress = read(folder / "logs/progress.json")
            require(
                progress["completed_epochs"]
                == progress["planned_epochs"]
                == shared["epochs"],
                "Incomplete training",
            )
            require(
                "test_local.npz"
                not in read(folder / "data/input_references.json")["derived_input"],
                "Test entered fit",
            )
        if party == "bob":
            require(
                not list(own.rglob("*predictions.npy"))
                and not (own / "results/joint_math_report.json").exists(),
                "Bob disclosure",
            )
    report = read(root / "alice/results/joint_math_report.json")
    require(
        report["status"] == "passed" and len(report["rows"]) == fits,
        "Missing numerical checks",
    )
    for row in report["rows"]:
        require(
            row["passed"]
            and all(
                v["max_abs_error"] <= cfg["diagnostic"]["reference_tolerance"]
                for v in row["splits"].values()
            ),
            "Numerical failure",
        )
        equation = read(
            root / "alice/trainings" / row["name"] / "results/equation_check.json"
        )
        require(
            equation["passed"]
            and equation["max_abs_error"] <= cfg["diagnostic"]["equation_tolerance"],
            "Equation failure",
        )
    for summary in report["summary"]:
        low, high = summary["seed_mean_ci_95"]
        require(
            summary["seed_count"] == len(shared["model_seeds"])
            and 0 <= low <= high <= cfg["diagnostic"]["reference_tolerance"],
            "Interval invalid",
        )
    # Exported artifacts remain private even when the mathematical inputs are synthetic.
    import stat

    for path in (root, *root.rglob("*")):
        require(not path.is_symlink(), "Link forbidden")
        require(
            stat.S_IMODE(path.stat().st_mode)
            == (vm.PRIVATE_DIRECTORY_MODE if path.is_dir() else vm.PRIVATE_FILE_MODE),
            "Permission mismatch",
        )
    return {
        "verification": "passed",
        "round": root.name,
        "dataset": report["dataset"],
        "config_sha256": declaration["config_sha256"],
        "report_sha256": sha(root / "alice/results/joint_math_report.json"),
        "fits": fits,
        "summary": report["summary"],
    }


def launch(workspace, dataset):
    workspace = Path(workspace).resolve()
    run_id = f"{dataset}_joint_math_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    local = workspace / "联合数学核验" / dataset / run_id
    local.mkdir(parents=True, exist_ok=False, mode=vm.PRIVATE_DIRECTORY_MODE)
    processes = []
    logs = []
    passed = False
    try:
        stage(workspace, dataset, run_id, local)
        environment = os.environ.copy()
        environment["LIMA_HOME"] = vm.LIMA_HOME_PATH
        for party in vm.PARTIES:
            root = f"/srv/vfl/{party}/{run_id}"
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
                raise RuntimeError("Participant failed; inspect private logs")
            if time.monotonic() > deadline:
                raise TimeoutError("Round timeout")
            time.sleep(vm.CHECK_INTERVAL_SECONDS)
        if any(p.returncode for p in processes):
            raise RuntimeError("Participant exit failure")
        passed = True
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for log in logs:
            log.close()
        for party in vm.PARTIES:
            if not passed:
                vm.guest(
                    party, f"sudo pkill -u {party} -f {shlex.quote(run_id)} || true"
                )
            av.export_artifacts(local, run_id, party, failed=not passed)
        vm.save(
            local / "status.json",
            {
                "status": "passed" if passed else "failed",
                "synthetic_only": True,
                "exit_codes": [p.poll() for p in processes],
            },
        )
        (local / "实验日志.md").write_text(
            f"# S5.V1：{run_id}\n\n目的：完整联合数值核验。内容、输入/代码指纹见 declaration.json 和各方 code/；清洗/标准化数据留各自VM的 data/，每次拟合留 trainings/。结果见 Alice results/joint_math_report.json；失败见各方 logs/。纯合成，不评估业务增益。\n"
        )
        vm.private_permissions(local)
    audit = verify(local)
    vm.save(local / "verification.json", audit)
    vm.private_permissions(local)
    index = workspace / "实验日志/实验索引.md"
    with index.open("a") as handle:
        handle.write(
            f"\n- S5.V1 `{run_id}`：纯合成完整联合核验完成，独立验收通过；路径 `{local}`。\n"
        )
    print(
        json.dumps({"artifact": str(local), "verification": audit}, ensure_ascii=False),
        flush=True,
    )
    return local


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--dataset", choices=list(vm.PREPARED), required=True)
    args = parser.parse_args()
    launch(args.workspace, args.dataset)
