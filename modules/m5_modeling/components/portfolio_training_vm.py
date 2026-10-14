"""Verified portfolio preparation to isolated two-cell training; owner cluster metadata."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import yaml
import functional_vm as vm  # type: ignore[import-not-found]
import nonlinear_full_vm as nf  # type: ignore[import-not-found]
import portfolio_groups as groups
import portfolio_network as network

NOTEBOOK = "S5.X2_portfolio_training.ipynb"
CODE = (
    "functional_training.py",
    "functional_metrics.py",
    "functional_tls.py",
    "active_diagnostic.py",
    "portfolio_training.py",
    "portfolio_math.py",
    "portfolio_resampling.py",
)
LOCAL_REFERENCE_COUNT = 2
MINIMUM_SEEDS = nf.MINIMUM_SEEDS
CELL_COUNT = nf.CELL_COUNT

sha = nf.sha
INPUT_FILE_COUNT = 3


def validate_config(cfg, dataset):
    if (
        cfg.get("step_id") != "S5.X2"
        or dataset not in cfg["datasets"]
        or (
            cfg["datasets"][dataset]["treatment"]
            and cfg["datasets"][dataset]["treatment_arms"] != [1]
        )
        or set(cfg["experiment"]["conditions"]) != {"A0", "B0"}
        or len(set(cfg["shared"]["model_seeds"])) < nf.MINIMUM_SEEDS
        or set(cfg["experiment"]["order_seeds"])
        != {str(s) for s in cfg["shared"]["model_seeds"]}
        or cfg["experiment"]["max_rows"] != {"train": None, "validation": None}
        or not re.fullmatch(
            r"S1P8R?_[0-9]{8}T[0-9]+Z", cfg["prepared_source_runs"][dataset]
        )
        or not re.fullmatch(r"[a-z][a-z0-9_]*", dataset)
    ):
        raise ValueError("Undeclared portfolio proxy protocol")


def validate_preparation(workspace, dataset, cfg):
    root = (
        Path(workspace)
        / "全数据集准备"
        / cfg["prepared_source_runs"][dataset]
        / dataset
    )
    for party in vm.PARTIES:
        runtime = json.loads((root / party / "code/runtime.json").read_text())
        engineering = json.loads(
            (root / party / "results/engineering.json").read_text()
        )
        if (
            runtime["party"] != party
            or runtime["dataset"] != dataset
            or runtime["config_sha256"] != cfg["source_config_sha256"]
            or engineering["status"] != "prepared"
            or not engineering["train_only_fit"]
            or engineering["sealed_confirmation_read"]
            or not engineering["input_unchanged"]
            or not engineering["isolation_checks_passed"]
        ):
            raise ValueError("Unverified preparation")


def stage(workspace, dataset, run_id, local):
    repo = Path(__file__).resolve().parents[vm.REPO_PARENT_DEPTH]
    module = repo / "modules/m5_modeling"
    config = module / "configs/portfolio_training.yaml"
    cfg = yaml.safe_load(config.read_text())
    if (
        dataset not in cfg["experiment"]["features"]
        or len(cfg["shared"]["learning_rates"]) != 1
        or cfg["experiment"]["max_rows"] != {"train": None, "validation": None}
    ):
        raise ValueError("Undeclared full-partition run or schedule")
    validate_config(cfg, dataset)
    validate_preparation(workspace, dataset, cfg)
    network_checks = network.preflight()
    network_checks.update(network.enforce_preparer_denial())
    vm.save(local / "network_preflight.json", network_checks)
    group_map = None
    group_receipt = None
    if cfg["datasets"][dataset]["resampling_unit"] == "cluster":
        group_map, group_receipt = groups.build(
            Path(workspace) / "数据集",
            Path(workspace)
            / "全数据集投递"
            / cfg["prepared_source_runs"][dataset]
            / dataset,
            dataset,
            repo / "modules/m1_data_selection/configs/portfolio_repair_freeze.yaml",
            Path(workspace) / "训练评估元数据" / run_id / "alice",
        )
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
            target = (
                own
                / "code"
                / ("nonlinear_full.yaml" if source == config else source.name)
            )
            target.write_bytes(source.read_bytes())
            vm.copy_to(target, party, f"/tmp/{party}_{run_id}_{target.name}")
            vm.guest(
                party,
                f"sudo install -o root -g root -m 444 /tmp/{party}_{run_id}_{target.name} {root}/code/{target.name}; sudo rm /tmp/{party}_{run_id}_{target.name}",
            )
        prepared = cfg["prepared_source_runs"][dataset] + "_" + dataset
        preparation = f"/srv/preparation/{party}/{prepared}"
        digest = vm.guest(
            party, f"sudo sha256sum {preparation}/code/portfolio_repair_freeze.yaml"
        ).split()[0]
        if digest != cfg["source_config_sha256"]:
            raise ValueError("Prepared source config changed")
        # Source CSV remains inside its owner VM; only digests reach the provider.
        script = (
            "from pathlib import Path;import hashlib,json;"
            f"p=Path('{preparation}/data');q=Path('{root}/input');"
            "m=json.loads((p/'manifest.json').read_text());"
            "names=('train.csv','validation.csv');"
            "assert set(m)==set(names);"
            "assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==m[n] for n in names);"
            "assert all(hashlib.sha256((q/n).read_bytes()).hexdigest()==m[n] for n in names);"
            "(q/'manifest.json').write_text(json.dumps(m));print('passed')"
        )
        for split in ("train", "validation"):
            vm.guest(
                party,
                f"sudo install -o root -g {party} -m 440 {preparation}/data/{split}.csv {root}/input/{split}.csv",
            )
        if (
            vm.guest(party, "sudo /opt/secretflow/bin/python -c " + shlex.quote(script))
            != "passed"
        ):
            raise ValueError("Prepared or copied input changed")
        vm.guest(
            party,
            f"sudo chown root:{party} {root}/input/manifest.json; sudo chmod 440 {root}/input/manifest.json",
        )
        input_count = INPUT_FILE_COUNT
        if party == "alice" and group_map is not None:
            temporary = f"/tmp/{party}_{run_id}_groups.json"
            vm.copy_to(group_map, party, temporary)
            vm.guest(
                party,
                f"sudo install -o root -g {party} -m 440 {temporary} {root}/input/owner_groups.json; sudo rm {temporary}",
            )
            if (
                vm.guest(
                    party, f"sudo sha256sum {root}/input/owner_groups.json"
                ).split()[0]
                != group_receipt["group_map_sha256"]
            ):
                raise ValueError("Owner group metadata changed")
            input_count += 1
        checks = (
            f"set -e; sudo test -f {preparation}/sealed/probe; sudo -u {party} test ! -r {preparation}/data/train.csv; "
            f"sudo -u {party} test ! -r {preparation}/sealed/probe; "
            f"sudo -u {party} test ! -w {root}/input/train.csv; "
            f"test $(sudo find {root}/input -type f | wc -l) = {input_count}; echo passed"
        )
        if vm.guest(party, checks) != "passed":
            raise ValueError("Preparation/sealed capability not denied")
        vm.save(
            own / "logs/preparation_boundary.json",
            {
                "owner_vm_copy_only": True,
                "preparation_read_denied": True,
                "sealed_read_denied": True,
                "frozen_inputs_readonly": True,
                "test_not_staged": True,
            },
        )
        runtime = {
            "party": party,
            "run_id": run_id,
            "dataset": dataset,
            "config": f"{root}/code/nonlinear_full.yaml",
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
        if party == "alice" and group_receipt is not None:
            runtime["group_map_sha256"] = group_receipt["group_map_sha256"]
            vm.save(own / "logs/group_boundary.json", group_receipt)
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
            "step_id": "S5.X2",
            "dataset": dataset,
            "round_id": run_id,
            "prepared_experiment": cfg["prepared_source_runs"][dataset] + "_" + dataset,
            "config_sha256": sha(config),
            "test_staged": False,
            "prediction_export": False,
            "scope": cfg["experiment"]["scope"],
        },
    )
    vm.private_permissions(local)


def verify(target):
    for party in vm.PARTIES:
        checks = json.loads(
            (target / party / "logs/preparation_boundary.json").read_text()
        )
        if not all(checks.values()):
            raise ValueError("Preparation capability failed")
    return verify_inventory(target)


def verify_inventory(root):
    import stat

    root = Path(root)

    def read(path):
        return json.loads(path.read_text())

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(read(root / "status.json")["status"] == "passed", "Round incomplete")
    cfg = yaml.safe_load((root / "alice/code/nonlinear_full.yaml").read_text())
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
    require(not list(root.rglob("owner_groups.json")), "Group identifiers exported")
    if cfg["datasets"][declaration["dataset"]]["resampling_unit"] == "cluster":
        group = read(root / "alice/logs/group_boundary.json")
        require(
            group["coverage_verified"]
            and group["source_unchanged"]
            and not group["outcomes_consulted"]
            and not group["sealed_opened"]
            and group["recipient"] == "alice",
            "Group metadata boundary failed",
        )
        require(
            not (root / "bob/logs/group_boundary.json").exists(),
            "Group metadata disclosed to Bob",
        )
    for party in vm.PARTIES:
        own = root / party
        runtime = read(own / "code/runtime.json")
        require(
            runtime["config_sha256"]
            == sha(own / "code/nonlinear_full.yaml")
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


def launch(workspace, output, dataset):
    cfg = yaml.safe_load(
        (
            Path(__file__).resolve().parents[vm.REPO_PARENT_DEPTH]
            / "modules/m5_modeling/configs/portfolio_training.yaml"
        ).read_text()
    )
    validate_config(cfg, dataset)
    run_id = (
        dataset
        + "_portfolio_training_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    target = output / dataset / run_id
    target.mkdir(parents=True, exist_ok=False, mode=vm.PRIVATE_DIRECTORY_MODE)
    processes, logs = [], []
    success, staged = False, False
    started = time.monotonic()
    try:
        stage(workspace, dataset, run_id, target)
        staged = True
        env = os.environ.copy()
        env["LIMA_HOME"] = vm.LIMA_HOME_PATH
        for party in vm.PARTIES:
            root = f"/srv/vfl/{party}/{run_id}"
            log = (target / party / "logs/host_process.log").open("x")
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
            )
        deadline = time.monotonic() + vm.ROUND_TIMEOUT_SECONDS
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("Participant failed")
            if time.monotonic() > deadline:
                raise TimeoutError("Fixed round budget exceeded")
            time.sleep(vm.CHECK_INTERVAL_SECONDS)
        if any(p.returncode for p in processes):
            raise RuntimeError("Participant exit failure")
        success = True
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for log in logs:
            log.close()
        errors = []
        for party in vm.PARTIES:
            if not success:
                try:
                    vm.guest(
                        party, f"sudo pkill -u {party} -f {shlex.quote(run_id)} || true"
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "party": party,
                            "operation": "stop",
                            "type": type(exc).__name__,
                        }
                    )
            if staged:
                try:
                    nf.export(target, run_id, party, failed=not success)
                except Exception as exc:
                    errors.append(
                        {
                            "party": party,
                            "operation": "export",
                            "type": type(exc).__name__,
                        }
                    )
        vm.save(
            target / "status.json",
            {
                "status": "passed" if success and not errors else "failed",
                "exit_codes": [p.poll() for p in processes],
                "cleanup_errors": errors,
                "wall_seconds": time.monotonic() - started,
            },
        )
        vm.private_permissions(target)
    if errors:
        raise RuntimeError("Incomplete owner export")
    try:
        audit = verify(target)
    except Exception as exc:
        state = json.loads((target / "status.json").read_text())
        state.update(
            status="failed", failure_phase="verification", error_type=type(exc).__name__
        )
        vm.save(target / "status.json", state)
        vm.private_permissions(target)
        raise
    vm.save(target / "verification.json", audit)
    (target / "实验日志.md").write_text(
        f"# S5.X2 {dataset}\n\n固定A0/B0两格五seed/12轮，目的、协议、来源和权限见declaration及code/。每拟合独立trainings/。清洗CSV和新增矩阵/分位点/预测/份额留本方VM，host仅授权审计与汇总；封存确认和原生PSI均未纳入本步。结果见alice/results/private_feature_evaluation.json；所有对照不利结果保留。\n"
    )
    vm.private_permissions(target)
    return target


def queue(workspace, output):
    cfg = yaml.safe_load(
        (
            Path(__file__).resolve().parents[vm.REPO_PARENT_DEPTH]
            / "modules/m5_modeling/configs/portfolio_training.yaml"
        ).read_text()
    )
    output.mkdir(parents=True, exist_ok=False, mode=vm.PRIVATE_DIRECTORY_MODE)
    state = {
        "status": "running",
        "pid": os.getpid(),
        "completed": [],
        "current_dataset": None,
    }
    vm.save(output / "status.json", state)
    try:
        for dataset in cfg["datasets"]:
            state["current_dataset"] = dataset
            vm.save(output / "status.json", state)
            target = launch(workspace, output, dataset)
            state["completed"].append(str(target))
        state.update(status="passed", current_dataset=None)
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue(args.workspace, args.output)
