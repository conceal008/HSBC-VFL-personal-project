"""Two-VM full-partition run provider. Do not stage test or export per-record predictions."""

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

NOTEBOOK = "S5.E5_joint_budget.ipynb"
CODE = (
    "functional_training.py",
    "functional_metrics.py",
    "functional_tls.py",
    "active_diagnostic.py",
    "nonlinear_full.py",
    "candidate_full.py",
    "joint_budget.py",
)
MINIMUM_SEEDS = 5
LOCAL_REFERENCE_COUNT = 8  # 原八只读参照
CELL_COUNT = 3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completed_recovery(dataset, cfg, module):
    """Reuse only bilateral final trajectories; never resume an unsaved epoch."""
    descriptor = os.environ.get("HSBC_E5_RECOVERY_PLAN")
    if not descriptor:
        return None
    plan = json.loads(Path(descriptor).read_text())
    if plan["dataset"] != dataset:
        return None
    source = plan["source_run"]
    allowed = {
        f"{cell}_seed{seed}"
        for cell in cfg["experiment"]["conditions"]
        for seed in cfg["shared"]["model_seeds"]
    }
    names = plan["completed_fits"]
    if (
        Path(source).name != source
        or not names
        or len(set(names)) != len(names)
        or not set(names) <= allowed
        or plan["original_exit_unknown"] is not True
    ):
        raise ValueError("Invalid recovery descriptor")
    owners = {}
    for party in vm.PARTIES:
        root = f"/srv/vfl/{party}/{source}"
        runtime = json.loads(vm.guest(party, f"sudo cat {root}/code/runtime.json"))
        source_config = yaml.safe_load(
            vm.guest(party, f"sudo cat {root}/code/joint_budget.yaml")
        )
        if source_config != cfg or runtime["dataset"] != dataset:
            raise ValueError("Recovery protocol differs")
        hashes = runtime["code_sha256"]
        inherited = (
            json.loads(vm.guest(party, f"sudo cat {root}/logs/recovery_lineage.json"))
            if runtime.get("reused_completed_fits")
            else None
        )
        if set(hashes) != set(CODE):
            raise ValueError("Recovery code manifest incomplete")
        for name, digest in hashes.items():
            actual = vm.guest(party, f"sudo sha256sum {root}/code/{name}").split()[0]
            if actual != digest or (
                name != "joint_budget.py"
                and sha(module / "components" / name) != digest
            ):
                raise ValueError("Recovery mathematical dependency changed")
        fit_hashes = {}
        for name in names:
            fit = f"{root}/trainings/{name}"
            expected = hashes
            if name in runtime.get("reused_completed_fits", []):
                expected = inherited.get("fit_code_sha256", {}).get(
                    name, inherited["source_code_sha256"]
                )
            if set(expected) != set(CODE) or any(
                digest != sha(module / "components" / code_name)
                for code_name, digest in expected.items()
                if code_name != "joint_budget.py"
            ):
                raise ValueError("Inherited mathematical dependency changed")
            progress = json.loads(vm.guest(party, f"sudo cat {fit}/logs/progress.json"))
            if (
                progress["completed_epochs"] != cfg["shared"]["epochs"]
                or progress["planned_epochs"] != cfg["shared"]["epochs"]
            ):
                raise ValueError("Recovery trajectory incomplete")
            vm.guest(party, f"sudo test -s {fit}/results/model.share")
            for code_name, digest in expected.items():
                if (
                    vm.guest(party, f"sudo sha256sum {fit}/code/{code_name}").split()[0]
                    != digest
                ):
                    raise ValueError("Recovery fit code differs")
            if party == "alice":
                for epoch in cfg["diagnostic"]["checkpoints"]:
                    snap = f"{fit}/checkpoints/e{epoch}/results"
                    diagnostic = json.loads(
                        vm.guest(party, f"sudo cat {snap}/diagnostic.json")
                    )
                    if diagnostic["epoch"] != epoch or (
                        epoch == cfg["diagnostic"]["checkpoints"][0]
                        and diagnostic["original_reference_passed"] is not True
                    ):
                        raise ValueError("Recovery checkpoint invalid")
                    for split in ("train", "validation"):
                        vm.guest(party, f"sudo test -s {snap}/{split}_predictions.npy")
            fit_hashes[name] = expected
        owners[party] = {
            "source_round": source,
            "source_code_sha256": hashes,
            "fit_code_sha256": fit_hashes,
            "source_config_sha256": runtime["config_sha256"],
            "reused_completed_fits": names,
            "original_exit_unknown": True,
            "not_retrained": True,
        }
    if any(
        owners["alice"][key] != owners["bob"][key]
        for key in ("source_code_sha256", "fit_code_sha256")
    ):
        raise ValueError("Recovery owners disagree on code")
    return owners


def stage(workspace, dataset, run_id, local):
    repo = Path(__file__).resolve().parents[vm.REPO_PARENT_DEPTH]
    module = repo / "modules/m5_modeling"
    config = module / "configs/joint_budget.yaml"
    cfg = yaml.safe_load(config.read_text())
    if (
        dataset not in cfg["experiment"]["features"]
        or len(cfg["shared"]["learning_rates"]) != 1
        or cfg["experiment"]["max_rows"] != {"train": None, "validation": None}
    ):
        raise ValueError("Undeclared full-partition run or schedule")
    recovery = completed_recovery(dataset, cfg, module)
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
        source_queue = json.loads(Path(os.environ["HSBC_E5_SOURCE_QUEUE"]).read_text())
        if source_queue["status"] != "passed":
            raise ValueError("Source reference queue incomplete")
        source_rounds = [
            Path(p).name
            for p in source_queue["completed"]
            if Path(p).parent.name == dataset
        ]
        if len(source_rounds) != 1:
            raise ValueError("Reference source ambiguous")
        source_root = f"/srv/vfl/{party}/{source_rounds[0]}"
        vm.guest(
            party, f"sudo install -d -m 550 -o root -g {party} {root}/input/references"
        )
        reference_files = {
            f"{split}_local.npz": f"data/{split}_local.npz"
            for split in ("train", "validation")
        }
        if party == "alice":
            for seed in cfg["shared"]["model_seeds"]:
                for cell in cfg["experiment"]["reference_cells"]:
                    for split in ("train", "validation"):
                        reference_files[f"{cell}_seed{seed}_{split}.npy"] = (
                            f"trainings/{cell}_seed{seed}/results/{split}_predictions.npy"
                        )
        for name, source_path in reference_files.items():
            result = vm.guest(
                party,
                f"sudo install -o root -g {party} -m 440 {source_root}/{source_path} {root}/input/references/{name};sudo sha256sum {root}/input/references/{name}",
            )
            manifest["references/" + name] = result.split()[0]
        for source_name in (
            "functional_training.py",
            "functional_metrics.py",
            "functional_tls.py",
            "active_diagnostic.py",
            "nonlinear_full.py",
        ):
            result = vm.guest(party, f"sudo sha256sum {source_root}/code/{source_name}")
            if result.split()[0] != sha(module / "components" / source_name):
                raise ValueError("Original reference implementation changed")
        source_config = yaml.safe_load(
            vm.guest(party, f"sudo cat {source_root}/code/nonlinear_full.yaml")
        )
        expected_shared = dict(cfg["shared"], epochs=source_config["shared"]["epochs"])
        if source_config["shared"] != expected_shared or any(
            source_config["experiment"][key] != cfg["experiment"][key]
            for key in (
                "order_seeds",
                "selection_seed",
                "quantiles",
                "features",
                "local_lr_c",
            )
        ):
            raise ValueError("Reference protocol differs")
        vm.save(
            own / "logs/reference_lineage.json",
            {
                "source_round": source_rounds[0],
                "source_step": "S5.E2",
                "original_code_matched": True,
                "shared_protocol_matched": True,
                "not_retrained": True,
            },
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
            "config": f"{root}/code/joint_budget.yaml",
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
        if recovery:
            lineage = recovery[party]
            old_root = f"/srv/vfl/{party}/{lineage['source_round']}"
            old_manifest = json.loads(
                vm.guest(party, f"sudo cat {old_root}/input/manifest.json")
            )
            if old_manifest != manifest:
                raise ValueError("Recovery frozen inputs differ")
            for name in lineage["reused_completed_fits"]:
                vm.guest(
                    party,
                    f"sudo test ! -e {root}/trainings/{name}; sudo cp -a {old_root}/trainings/{name} {root}/trainings/{name}; sudo chown -R {party}:{party} {root}/trainings/{name}",
                )
            runtime["reused_completed_fits"] = lineage["reused_completed_fits"]
            vm.save(own / "logs/recovery_lineage.json", lineage)
            vm.copy_to(
                own / "logs/recovery_lineage.json",
                party,
                f"/tmp/{party}_{run_id}_recovery.json",
            )
            vm.guest(
                party,
                f"sudo install -o {party} -g {party} -m 600 /tmp/{party}_{run_id}_recovery.json {root}/logs/recovery_lineage.json; sudo rm /tmp/{party}_{run_id}_recovery.json",
            )
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
            "step_id": "S5.E5",
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
    cfg = yaml.safe_load((root / "alice/code/joint_budget.yaml").read_text())
    shared = cfg["shared"]
    seeds = shared["model_seeds"]
    cells = cfg["experiment"]["conditions"]
    require(
        len(set(seeds)) >= MINIMUM_SEEDS and len(shared["learning_rates"]) == 1,
        "Protocol changed",
    )
    require(
        cfg["experiment"]["max_rows"] == {"train": None, "validation": None},
        "Row caps enabled",
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
            == sha(own / "code/joint_budget.yaml")
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
        partition = read(own / "logs/full_partition.json")
        require(
            partition["all_rows_retained"]
            and partition["source_rows"] == partition["used_rows"]
            and set(partition["used_rows"]) == {"train", "validation"}
            and not partition["test_included"],
            "Full partition coverage failed",
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
            reused = folder.name in runtime.get("reused_completed_fits", [])
            lineage = read(own / "logs/recovery_lineage.json") if reused else None
            expected_code = (
                lineage.get("fit_code_sha256", {}).get(
                    folder.name, lineage["source_code_sha256"]
                )
                if reused
                else runtime["code_sha256"]
            )
            require(
                all(
                    sha(folder / "code" / name) == digest
                    for name, digest in expected_code.items()
                ),
                "Fit code changed",
            )
            refs = read(folder / "data/input_references.json")
            require(
                {k for k in refs["frozen_input"] if not k.startswith("references/")}
                == {"train.csv", "validation.csv"}
                and not any("test" in k for k in refs["derived_input"]),
                "Test entered fit",
            )
            if any(folder.name.startswith(cell + "_seed") for cell in cells):
                progress = read(folder / "logs/progress.json")
                require(
                    progress["completed_epochs"]
                    == progress["planned_epochs"]
                    == shared["epochs"],
                    "Fit unfinished",
                )
        for cell in cells:
            for seed in seeds:
                trajectory = own / "trainings" / f"{cell}_seed{seed}"
                if party == "alice":
                    for epoch in cfg["diagnostic"]["checkpoints"]:
                        snap = trajectory / "checkpoints" / f"e{epoch}" / "results"
                        diagnostic = read(snap / "diagnostic.json")
                        require(
                            diagnostic["epoch"] == epoch
                            and isinstance(
                                diagnostic["full_unpadded_gradient_below_threshold"],
                                bool,
                            )
                            and diagnostic["gradient_values_disclosed"] is False
                            and diagnostic["convergence_proven"] is False
                            and (
                                epoch != cfg["diagnostic"]["checkpoints"][0]
                                or diagnostic["original_reference_passed"] is True
                            ),
                            "Checkpoint diagnostic invalid",
                        )
                        require(
                            read(snap / "status.json")["training"] == "passed"
                            and read(snap / "evaluation.json")["test_read"] is False,
                            "Checkpoint result missing",
                        )
        if party == "bob":
            require(
                not list(own.rglob("*evaluation.json")), "Bob evaluation disclosure"
            )
        require(
            read(own / "logs/reference_equivalence.json")["owner_matrices_exact"]
            and read(own / "logs/cross_control.json")["native_known_answer"],
            "Reference or mathematical control failed",
        )
    report = read(root / "alice/results/private_feature_evaluation.json")
    require(
        set(report["models"])
        == set(
            cfg["experiment"]["reference_cells"]
            + [f"{c}_e{e}" for c in cells for e in cfg["diagnostic"]["checkpoints"]]
        ),
        "Missing predeclared group",
    )
    require(
        report["status"] == "completed"
        and report["model_seeds"] == seeds
        and report["independent_confirmation"] is False
        and report["full_data_training"] is True,
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
        "full_train_validation": True,
        "independent_confirmation": False,
    }  # 魔数豁免: 两表示乘两固定本地模型族


def launch(workspace, dataset):
    workspace = Path(workspace).resolve()
    run_id = f"{dataset}_joint_budget_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    local = workspace / "联合优化诊断" / dataset / run_id
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
        cleanup_errors = []
        for party in vm.PARTIES:
            if not passed:
                try:
                    vm.guest(
                        party, f"sudo pkill -u {party} -f {shlex.quote(run_id)} || true"
                    )
                except Exception as error:
                    cleanup_errors.append(
                        {
                            "party": party,
                            "operation": "stop",
                            "error_type": type(error).__name__,
                        }
                    )
            try:
                export(local, run_id, party, failed=not passed)
            except Exception as error:
                cleanup_errors.append(
                    {
                        "party": party,
                        "operation": "export",
                        "error_type": type(error).__name__,
                    }
                )
        vm.save(
            local / "status.json",
            {
                "status": "passed" if passed and not cleanup_errors else "failed",
                "exit_codes": [p.poll() for p in processes],
                "cleanup_errors": cleanup_errors,
            },
        )
        (local / "实验日志.md").write_text(
            f"# S5.E5：{run_id}\n\n目的：实际联合与主动方四格48轮预算诊断。固定记录选择和参数见declaration.json与code/；新增矩阵、抽样记录与分位点只在各VMdata/。每个拟合在trainings/独立目录；结果与CI见Alice results/private_feature_evaluation.json、特征工程结果分析.md。预测、矩阵与份额未导出，全部冻结train/validation参与，test不投递；不代表独立确认。\n"
        )
        vm.private_permissions(local)
    audit = verify(local)
    vm.save(local / "verification.json", audit)
    vm.private_permissions(local)
    with (workspace / "实验日志/实验索引.md").open("a") as f:
        f.write(
            f"\n- S5.E5 `{run_id}`：实际联合预算诊断完成、工程验收通过，结果仅私有目录 `{local}`。\n"
        )
    print(
        json.dumps({"artifact": str(local), "verification": audit}, ensure_ascii=False),
        flush=True,
    )
    return local


def run_queue(workspace):
    """Serial complete-partition experiments with a durable host lifecycle receipt."""
    workspace = Path(workspace).resolve()
    queue = (
        workspace
        / "联合优化诊断"
        / ("queue_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    queue.mkdir(parents=True, exist_ok=False, mode=vm.PRIVATE_DIRECTORY_MODE)
    receipt = {
        "status": "running",
        "pid": os.getpid(),
        "completed": [],
        "current_dataset": None,
    }
    vm.save(queue / "status.json", receipt)
    vm.private_permissions(queue)
    print(
        json.dumps({"queue": str(queue), "pid": os.getpid()}, ensure_ascii=False),
        flush=True,
    )
    try:
        for dataset in vm.PREPARED:
            receipt["current_dataset"] = dataset
            vm.save(queue / "status.json", receipt)
            root = launch(workspace, dataset)
            receipt["completed"].append(str(root))
            vm.save(queue / "status.json", receipt)
        receipt.update(status="passed", current_dataset=None)
    except BaseException as error:
        receipt.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        vm.save(queue / "status.json", receipt)
        vm.private_permissions(queue)
    return queue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--dataset", choices=[*vm.PREPARED, "all"], required=True)
    args = parser.parse_args()
    if args.dataset == "all":
        run_queue(args.workspace)
    else:
        launch(args.workspace, args.dataset)
