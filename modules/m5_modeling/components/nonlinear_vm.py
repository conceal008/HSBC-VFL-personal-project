"""Two-VM pilot provider. Do not stage test or export per-record predictions."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
import yaml
import functional_vm as vm  # type: ignore[import-not-found]

NOTEBOOK = "S5.E1_nonlinear_pilot.ipynb"
CODE = (
    "functional_training.py",
    "functional_metrics.py",
    "functional_tls.py",
    "active_diagnostic.py",
    "nonlinear_pilot.py",
)
MINIMUM_SEEDS = 5
LOCAL_REFERENCE_COUNT = 4
CELL_COUNT = 3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage(workspace, dataset, run_id, local):
    repo = Path(__file__).resolve().parents[vm.REPO_PARENT_DEPTH]
    module = repo / "modules/m5_modeling"
    config = module / "configs/nonlinear_pilot.yaml"
    cfg = yaml.safe_load(config.read_text())
    if (
        dataset not in cfg["experiment"]["features"]
        or len(cfg["shared"]["learning_rates"]) != 1
    ):
        raise ValueError("Undeclared pilot or schedule")
    addresses = vm.setup_certificates(workspace)
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    for party in vm.PARTIES:
        vm.restrict_network(party, addresses)
        root = f"/srv/vfl/{party}/{run_id}"
        own = local / party
        for sub in ("code", "data", "results", "logs"):
            (own / sub).mkdir(parents=True)
        # Revoke historical round entry permissions; preserve every frozen file.
        vm.guest(
            party,
            f"sudo find /srv/vfl/{party} -mindepth 1 -maxdepth 1 -type d ! -name tls -exec chown root:root {{}} \\; -exec chmod 700 {{}} \\;",
        )
        vm.guest(
            party,
            f"sudo install -d -m 755 {root} {root}/code; sudo install -d -m 550 -o root -g {party} {root}/input; sudo install -d -m 700 -o {party} -g {party} {root}/data {root}/results {root}/logs {root}/trainings",
        )
        for source in [
            *(module / "components" / n for n in CODE),
            config,
            module / "notebooks" / NOTEBOOK,
        ]:
            target = own / "code" / source.name
            target.write_bytes(source.read_bytes())
            vm.copy_to(target, party, f"/tmp/{party}_{run_id}_{source.name}")
            vm.guest(
                party,
                f"sudo install -o root -g root -m 444 /tmp/{party}_{run_id}_{source.name} {root}/code/{source.name}; sudo rm /tmp/{party}_{run_id}_{source.name}",
            )
        prepared = vm.PREPARED[dataset]
        source_dir = workspace / "联邦隔离产物" / prepared / party
        source_manifest = json.loads(
            (
                workspace / "联邦隔离结果" / prepared / party / "data_manifest.json"
            ).read_text()
        )
        manifest = {}
        for split in ("train", "validation"):
            source = source_dir / f"{split}.csv"
            digest = sha(source)
            if digest != source_manifest[source.name]:
                raise ValueError("Frozen input changed")
            manifest[source.name] = digest
            vm.copy_to(source, party, f"/tmp/{party}_{run_id}_{source.name}")
            vm.guest(
                party,
                f"sudo install -o root -g {party} -m 440 /tmp/{party}_{run_id}_{source.name} {root}/input/{source.name}; sudo rm /tmp/{party}_{run_id}_{source.name}",
            )
        manifest_path = own / "data/input_manifest.json"
        vm.save(manifest_path, manifest)
        vm.copy_to(manifest_path, party, f"/tmp/{party}_{run_id}_manifest.json")
        vm.guest(
            party,
            f"sudo install -o root -g {party} -m 440 /tmp/{party}_{run_id}_manifest.json {root}/input/manifest.json; sudo rm /tmp/{party}_{run_id}_manifest.json",
        )
        runtime = {
            "party": party,
            "run_id": run_id,
            "dataset": dataset,
            "config": f"{root}/code/nonlinear_pilot.yaml",
            "config_sha256": sha(config),
            "git_sha": git_sha,
            "code_sha256": {n: sha(own / "code" / n) for n in CODE},
            "notebook": NOTEBOOK,
            "is_smoke": False,
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
        vm.copy_to(
            own / "code/runtime.json", party, f"/tmp/{party}_{run_id}_runtime.json"
        )
        vm.guest(
            party,
            f"sudo install -m 444 /tmp/{party}_{run_id}_runtime.json {root}/code/runtime.json",
        )
    vm.save(
        local / "declaration.json",
        {
            "step_id": "S5.E1",
            "dataset": dataset,
            "round_id": run_id,
            "prepared_experiment": vm.PREPARED[dataset],
            "config_sha256": sha(config),
            "test_staged": False,
            "prediction_export": False,
            "scope": cfg["experiment"]["scope"],
        },
    )
    vm.private_permissions(local)


def export(local, run_id, party, failed=False):
    root = f"/srv/vfl/{party}/{run_id}"
    archive = f"/tmp/{party}_{run_id}_export.tar"
    vm.guest(
        party,
        f"sudo tar --exclude='model.share' --exclude='*.npz' --exclude='*.npy' --exclude='*.trace.log' -cf {shlex.quote(archive)} -C {shlex.quote(root)} code results logs trainings",
    )
    target = (
        local
        / party
        / ("failed_artifacts.tar" if failed else "authorized_artifacts.tar")
    )
    vm.lima("copy", f"{vm.VM_NAMES[party]}:{archive}", target)
    with tarfile.open(target) as bundle:
        bundle.extractall(local / party, filter="data")
    vm.guest(party, f"sudo rm {shlex.quote(archive)}")


def verify(root):
    import stat

    root = Path(root)

    def read(path):
        return json.loads(path.read_text())

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(read(root / "status.json")["status"] == "passed", "Round incomplete")
    cfg = yaml.safe_load((root / "alice/code/nonlinear_pilot.yaml").read_text())
    shared = cfg["shared"]
    seeds = shared["model_seeds"]
    cells = cfg["experiment"]["conditions"]
    require(
        len(set(seeds)) >= MINIMUM_SEEDS and len(shared["learning_rates"]) == 1,
        "Protocol changed",
    )
    declaration = read(root / "declaration.json")
    require(
        declaration["test_staged"] is False
        and declaration["prediction_export"] is False,
        "Capability invalid",
    )
    for pattern in ("*.npy", "*.npz", "model.share", "*.trace.log"):
        require(not list(root.rglob(pattern)), "Forbidden export")
    for party in vm.PARTIES:
        own = root / party
        runtime = read(own / "code/runtime.json")
        require(
            runtime["config_sha256"]
            == sha(own / "code/nonlinear_pilot.yaml")
            == declaration["config_sha256"],
            "Config mismatch",
        )
        require(
            all(
                sha(own / "code" / name) == digest
                for name, digest in runtime["code_sha256"].items()
            ),
            "Code mismatch",
        )
        require(
            all(read(own / "logs/isolation.json").values())
            and all(read(own / "logs/tls_checks.json").values()),
            "Isolation or TLS failed",
        )
        boundary = read(own / "logs/output_boundary.json")
        require(
            boundary["old_test_read"] is False
            and boundary["scores_receiver"] == "alice"
            and boundary["frozen_input_unchanged"],
            "Output boundary invalid",
        )
        require(
            all(
                v is (False if k == "selection_uses_labels" else True)
                for k, v in read(own / "logs/test_capability.json").items()
            ),
            "Test capability changed",
        )
        nb = read(own / "results/party.executed.ipynb")
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        require(
            [c["execution_count"] for c in code] == list(range(1, CELL_COUNT + 1))
            and not any(
                o["output_type"] == "error" for c in code for o in c["outputs"]
            ),
            "Notebook incomplete",
        )
        require(
            read(own / "results/NOTEBOOK_COMPLETED.json")["status"] == "passed",
            "Notebook completion missing",
        )
        folders = list((own / "trainings").iterdir())
        expected = len(seeds) * (
            len(cells) + (LOCAL_REFERENCE_COUNT if party == "alice" else 0)
        )  # 魔数豁免: 两表示乘两固定本地模型族
        require(len(folders) == expected, "Missing training folders")
        for folder in folders:
            require(
                all((folder / s).is_dir() for s in ("code", "data", "results", "logs"))
                and (folder / "code/environment.lock").is_file(),
                "Fit assets missing",
            )
            require(
                all(
                    sha(folder / "code" / name) == digest
                    for name, digest in runtime["code_sha256"].items()
                ),
                "Fit code changed",
            )
            refs = read(folder / "data/input_references.json")
            require(
                set(refs["frozen_input"]) == {"train.csv", "validation.csv"}
                and not any("test" in k for k in refs["derived_input"]),
                "Test entered fit",
            )
            if folder.name.startswith(tuple(cells)):
                progress = read(folder / "logs/progress.json")
                require(
                    progress["completed_epochs"]
                    == progress["planned_epochs"]
                    == shared["epochs"],
                    "Fit unfinished",
                )
        if party == "bob":
            require(
                not list(own.rglob("*evaluation.json")), "Bob evaluation disclosure"
            )
    report = read(root / "alice/results/private_feature_evaluation.json")
    require(
        report["status"] == "completed"
        and report["model_seeds"] == seeds
        and report["independent_confirmation"] is False,
        "Report scope invalid",
    )
    for group in [report["models"], report["paired_comparisons"]]:
        for metrics in group.values():
            for interval in metrics.values():
                require(
                    all(
                        __import__("math").isfinite(interval[k])
                        for k in ("estimate", "ci_low", "ci_high")
                    )
                    and interval["ci_low"] <= interval["ci_high"]
                    and interval["valid_replicates"] == shared["bootstrap_repeats"],
                    "Interval failed",
                )
    for path in (root, *root.rglob("*")):
        require(
            not path.is_symlink()
            and stat.S_IMODE(path.stat().st_mode)
            == (vm.PRIVATE_DIRECTORY_MODE if path.is_dir() else vm.PRIVATE_FILE_MODE),
            "Private permission failed",
        )
    return {
        "verification": "passed",
        "dataset": declaration["dataset"],
        "config_sha256": declaration["config_sha256"],
        "secure_fits": len(seeds) * len(cells),
        "alice_fits": len(seeds) * (len(cells) + LOCAL_REFERENCE_COUNT),
        "bob_fits": len(seeds) * len(cells),
        "independent_confirmation": False,
    }  # 魔数豁免: 两表示乘两固定本地模型族


def launch(workspace, dataset):
    workspace = Path(workspace).resolve()
    run_id = f"{dataset}_nonlinear_pilot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    local = workspace / "特征工程预试验" / dataset / run_id
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
                raise RuntimeError("Party failed; inspect private logs")
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
            export(local, run_id, party, failed=not passed)
        vm.save(
            local / "status.json",
            {
                "status": "passed" if passed else "failed",
                "exit_codes": [p.poll() for p in processes],
            },
        )
        (local / "实验日志.md").write_text(
            f"# S5.E1：{run_id}\n\n目的：C01单变量非线性四格开发预试验。固定记录选择和参数见declaration.json与code/；新增矩阵、抽样记录与分位点只在各VMdata/。每个拟合在trainings/独立目录；结果与CI见Alice results/private_feature_evaluation.json、特征工程结果分析.md。预测、矩阵与份额未导出，不代表独立确认或完整数据训练。\n"
        )
        vm.private_permissions(local)
    audit = verify(local)
    vm.save(local / "verification.json", audit)
    vm.private_permissions(local)
    with (workspace / "实验日志/实验索引.md").open("a") as f:
        f.write(
            f"\n- S5.E1 `{run_id}`：C01开发预试验完成、工程验收通过，结果仅私有目录 `{local}`。\n"
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
