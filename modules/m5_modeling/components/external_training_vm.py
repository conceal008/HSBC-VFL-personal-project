"""S1.P6 owner-VM-only transfer and fixed four-cell proxy training provider."""

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

NOTEBOOK = "S5.X1_external_training.ipynb"
CODE = nf.CODE
sha = nf.sha
INPUT_FILE_COUNT = 3


def validate_config(cfg, dataset):
    if (
        cfg.get("step_id") != "S5.X1"
        or dataset not in cfg["datasets"]
        or cfg["datasets"][dataset]["treatment"]
        or len(set(cfg["shared"]["model_seeds"])) < nf.MINIMUM_SEEDS
        or set(cfg["experiment"]["order_seeds"])
        != {str(s) for s in cfg["shared"]["model_seeds"]}
        or cfg["experiment"]["max_rows"] != {"train": None, "validation": None}
        or not re.fullmatch(r"S1P6_[0-9]{8}T[0-9]+Z", cfg["prepared_source_run"])
        or not re.fullmatch(r"[a-z][a-z0-9_]*", dataset)
    ):
        raise ValueError("Undeclared classification proxy protocol")


def validate_preparation(workspace, dataset, cfg):
    root = Path(workspace) / "新增代理准备" / cfg["prepared_source_run"] / dataset
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
    config = module / "configs/external_training.yaml"
    cfg = yaml.safe_load(config.read_text())
    if (
        dataset not in cfg["experiment"]["features"]
        or len(cfg["shared"]["learning_rates"]) != 1
        or cfg["experiment"]["max_rows"] != {"train": None, "validation": None}
    ):
        raise ValueError("Undeclared full-partition run or schedule")
    validate_config(cfg, dataset)
    validate_preparation(workspace, dataset, cfg)
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
        prepared = cfg["prepared_source_run"] + "_" + dataset
        preparation = f"/srv/preparation/{party}/{prepared}"
        digest = vm.guest(
            party, f"sudo sha256sum {preparation}/code/external_freeze.yaml"
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
        checks = (
            f"set -e; sudo test -f {preparation}/sealed/probe; sudo -u {party} test ! -r {preparation}/data/train.csv; "
            f"sudo -u {party} test ! -r {preparation}/sealed/probe; "
            f"sudo -u {party} test ! -w {root}/input/train.csv; "
            f"test $(sudo find {root}/input -type f | wc -l) = {INPUT_FILE_COUNT}; echo passed"
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
            "step_id": "S5.X1",
            "dataset": dataset,
            "round_id": run_id,
            "prepared_experiment": cfg["prepared_source_run"] + "_" + dataset,
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
    return nf.verify(target)


def launch(workspace, output, dataset, recovery_source=None):
    cfg = yaml.safe_load(
        (
            Path(__file__).resolve().parents[vm.REPO_PARENT_DEPTH]
            / "modules/m5_modeling/configs/external_training.yaml"
        ).read_text()
    )
    validate_config(cfg, dataset)
    run_id = (
        dataset
        + "_external_training_"
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
        if recovery_source is not None:
            attach_recovery(target, run_id, recovery_source)
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
        f"# S5.X1 {dataset}\n\n固定四格五seed/12轮，目的、协议、来源和权限见declaration及code/。每拟合独立trainings/。清洗CSV和新增矩阵/分位点/预测/份额留本方VM，host仅授权审计与汇总；封存确认和原生PSI均未纳入本步。结果见alice/results/private_feature_evaluation.json；所有对照不利结果保留。\n"
    )
    vm.private_permissions(target)
    return target


def attach_recovery(target, run_id, source):
    """Clone complete fits within their owner VM; never reuse partial weights."""
    source = Path(source)
    module = (
        Path(__file__).resolve().parents[vm.REPO_PARENT_DEPTH] / "modules/m5_modeling"
    )
    cfg = yaml.safe_load((module / "configs/external_training.yaml").read_text())
    dataset = json.loads((target / "declaration.json").read_text())["dataset"]
    if source.parent.name != dataset or not re.fullmatch(
        dataset + r"_external_training_[0-9]{8}T[0-9]+Z", source.name
    ):
        raise ValueError("Recovery source identity invalid")
    # Source seed is accepted only when all four secure fits are complete on both sides.
    eligible = []
    for seed in cfg["shared"]["model_seeds"]:
        complete = True
        for party in vm.PARTIES:
            root = f"/srv/vfl/{party}/{source.name}"
            code = (
                "import json;from pathlib import Path;"
                f"p=Path('{root}');names={[c + '_seed' + str(seed) for c in cfg['experiment']['conditions']]!r};"
                "ok=all((p/'trainings'/n/'logs/progress.json').exists() and "
                "json.loads((p/'trainings'/n/'logs/progress.json').read_text()).get('completed_epochs')=="
                f"{cfg['shared']['epochs']} and (p/'trainings'/n/'results/model.share').exists() for n in names);"
                + (
                    "ok=ok and all((p/'trainings'/n/'results/status.json').exists() and json.loads((p/'trainings'/n/'results/status.json').read_text()).get('training')=='passed' for n in names);"
                    if party == "alice"
                    else ""
                )
                + "print('complete' if ok else 'partial')"
            )
            complete &= (
                vm.guest(
                    party, "sudo /opt/secretflow/bin/python -c " + shlex.quote(code)
                )
                == "complete"
            )
        if complete:
            eligible.append(seed)
    names = [
        c + "_seed" + str(s) for s in eligible for c in cfg["experiment"]["conditions"]
    ]
    if not names:
        raise ValueError("No bilateral complete seed block to recover")
    for party in vm.PARTIES:
        own = target / party
        root = f"/srv/vfl/{party}/{run_id}"
        previous = f"/srv/vfl/{party}/{source.name}"
        runtime = json.loads((own / "code/runtime.json").read_text())
        old_runtime = json.loads(
            vm.guest(party, f"sudo cat {previous}/code/runtime.json")
        )
        if (
            old_runtime["dataset"] != dataset
            or old_runtime["party"] != party
            or old_runtime["config_sha256"] != runtime["config_sha256"]
            or old_runtime["code_sha256"] != runtime["code_sha256"]
        ):
            raise ValueError("Recovery protocol or original math differs")
        for filename, digest in old_runtime["code_sha256"].items():
            if (
                vm.guest(party, f"sudo sha256sum {previous}/code/{filename}").split()[0]
                != digest
            ):
                raise ValueError("Recovery source code changed")
        # This owner reads only its former inputs and fit assets.
        all_names = list(names)
        if party == "alice":
            for seed in cfg["shared"]["model_seeds"]:
                local = [
                    f"L0_{family}_{int(enhanced)}_seed{seed}"
                    for family in ("LR", "tree")
                    for enhanced in (False, True)
                ]
                check = (
                    "import json;from pathlib import Path;"
                    + f"p=Path('{previous}');names={local!r};"
                    + "print('complete' if all((p/'trainings'/n/'results/status.json').exists() and json.loads((p/'trainings'/n/'results/status.json').read_text()).get('training')=='passed' for n in names) else 'partial')"
                )
                if (
                    vm.guest(
                        party,
                        "sudo /opt/secretflow/bin/python -c " + shlex.quote(check),
                    )
                    == "complete"
                ):
                    all_names.extend(local)
        vm.guest(
            party, f"sudo install -d -o root -g {party} -m 550 {root}/recovery_input"
        )
        for split in ("train", "validation"):
            old_hash = vm.guest(
                party, f"sudo sha256sum {previous}/input/{split}.csv"
            ).split()[0]
            new_hash = vm.guest(
                party, f"sudo sha256sum {root}/input/{split}.csv"
            ).split()[0]
            if old_hash != new_hash:
                raise ValueError("Recovery input CSV differs")
            vm.guest(
                party,
                f"sudo install -o root -g {party} -m 440 {previous}/data/{split}_local.npz {root}/recovery_input/{split}_local.npz",
            )
        for name in all_names:
            vm.guest(
                party,
                f"sudo cp -a {previous}/trainings/{name} {root}/trainings/{name}; sudo chown -R {party}:{party} {root}/trainings/{name}",
            )
        receipt_code = (
            "import hashlib,json;from pathlib import Path;"
            f"p=Path('{root}');old=Path('{previous}');names={all_names!r};"
            "hashes={n:{str(x.relative_to(p/'trainings'/n)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (p/'trainings'/n).rglob('*') if x.is_file()} for n in names};"
            "assert all(hashlib.sha256((old/'trainings'/n/rel).read_bytes()).hexdigest()==value for n,items in hashes.items() for rel,value in items.items());"
            f"(p/'code/recovery_manifest.json').write_text(json.dumps({{'party':'{party}','source_run':'{source.name}','config_sha256':'{runtime['config_sha256']}','reused_fits':names,'fit_sha256':hashes}}));print('passed')"
        )
        if (
            vm.guest(
                party, "sudo /opt/secretflow/bin/python -c " + shlex.quote(receipt_code)
            )
            != "passed"
        ):
            raise ValueError("Recovered copy changed")
        helper = module / "components/external_training_recovery.py"
        (own / "code" / helper.name).write_bytes(helper.read_bytes())
        temporary = f"/tmp/{party}_{run_id}_recovery_helper.py"
        vm.copy_to(own / "code" / helper.name, party, temporary)
        vm.guest(
            party,
            f"sudo install -o root -g root -m 444 {temporary} {root}/code/{helper.name}; sudo rm {temporary}; sudo chown root:root {root}/code/recovery_manifest.json; sudo chmod 444 {root}/code/recovery_manifest.json",
        )
        runtime.update(
            reused_completed_fits=names,
            recovery_source_run=source.name,
            controller_sha256={helper.name: sha(helper)},
        )
        vm.save(own / "code/runtime.json", runtime)
        temporary = f"/tmp/{party}_{run_id}_recovery_runtime.json"
        vm.copy_to(own / "code/runtime.json", party, temporary)
        vm.guest(
            party,
            f"sudo install -o root -g root -m 444 {temporary} {root}/code/runtime.json; sudo rm {temporary}",
        )
    vm.save(
        target / "recovery_declaration.json",
        {
            "source_run": source.name,
            "bilateral_complete_seeds": eligible,
            "reused_secure_fits": names,
            "partial_fit_restart_from_initialization": True,
            "original_guest_exit_codes": "unknown",
        },
    )
    vm.private_permissions(target)


def queue(workspace, output):
    cfg = yaml.safe_load(
        (
            Path(__file__).resolve().parents[vm.REPO_PARENT_DEPTH]
            / "modules/m5_modeling/configs/external_training.yaml"
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
