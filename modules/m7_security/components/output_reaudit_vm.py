"""Owner-separated evaluation provider; exports aggregate reports only."""

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
import nonlinear_full_vm as provider  # type: ignore[import-not-found]

NOTEBOOK = "S7.P2_output_reaudit.ipynb"
CODE = (
    "functional_training.py",
    "functional_metrics.py",
    "functional_tls.py",
    "active_diagnostic.py",
    "nonlinear_full.py",
    "bounded_psi.py",
    "output_reaudit.py",
)
SPLITS = ("train", "validation")
CODE_CELLS = 3
SOURCE_CODE = {
    "E4": (*provider.CODE, "candidate_full.py"),
    "E5": (*provider.CODE, "candidate_full.py", "joint_budget.py"),
}
SOURCE_CONFIG = {"E4": "candidate_full.yaml", "E5": "joint_budget.yaml"}


def mapping(sources, party, cfg):
    if set(sources) != {"E4", "E5"} or any(
        Path(v).name != v or not v.replace("_", "").isalnum() for v in sources.values()
    ):
        raise ValueError("Invalid source runs")
    result = {
        f"{split}_local.npz": (sources["E4"], f"data/{split}_local.npz")
        for split in SPLITS
    }
    result.update(
        {
            f"check_{split}_local.npz": (sources["E5"], f"data/{split}_local.npz")
            for split in SPLITS
        }
    )
    if party == "bob":
        result["train.csv"] = (sources["E4"], "input/train.csv")
    else:
        for cell in cfg["output_cells"]:
            for seed in cfg["model_seeds"]:
                for split in SPLITS:
                    if cell.startswith("D_"):
                        trajectory = cell.removesuffix("_e48")
                        result[f"{cell}_seed{seed}_{split}.npy"] = (
                            sources["E5"],
                            f"trainings/{trajectory}_seed{seed}/checkpoints/e48/results/{split}_predictions.npy",
                        )
                    else:
                        result[f"{cell}_seed{seed}_{split}.npy"] = (
                            sources["E4"],
                            f"trainings/{cell}_seed{seed}/results/{split}_predictions.npy",
                        )
    return result


def stage(workspace, sources, output, dataset):
    repo = Path(__file__).resolve().parents[vm.REPO_PARENT_DEPTH]
    module = repo / "modules/m7_security"
    config = module / "configs/output_reaudit.yaml"
    cfg = yaml.safe_load(config.read_text())
    if dataset not in cfg["attribute_targets"]:
        raise ValueError("Undeclared data")
    mapping(sources, "alice", cfg)  # Validate identifiers before any shell action.
    run_id = (
        dataset
        + "_output_reaudit_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    target = output / dataset / run_id
    addresses = vm.setup_certificates(workspace)
    for party in vm.PARTIES:
        for kind, source_run in sources.items():
            expected = dataset + (
                "_candidate_full_" if kind == "E4" else "_joint_budget_"
            )
            if not source_run.startswith(expected):
                raise ValueError("Source type or dataset differs")
            source_root = f"/srv/vfl/{party}/{source_run}"
            source_runtime = json.loads(
                vm.guest(party, f"sudo cat {source_root}/code/runtime.json")
            )
            if source_runtime["party"] != party or source_runtime["dataset"] != dataset:
                raise ValueError("Source ownership differs")
            source_cfg = Path(source_runtime["config"]).name
            if source_cfg != SOURCE_CONFIG[kind] or set(
                source_runtime["code_sha256"]
            ) != set(SOURCE_CODE[kind]):
                raise ValueError("Source manifest schema differs")
            if (
                vm.guest(
                    party, f"sudo sha256sum {source_root}/code/{source_cfg}"
                ).split()[0]
                != source_runtime["config_sha256"]
            ):
                raise ValueError("Source config changed")
            for name, digest in source_runtime["code_sha256"].items():
                if (
                    vm.guest(
                        party, f"sudo sha256sum {source_root}/code/{name}"
                    ).split()[0]
                    != digest
                ):
                    raise ValueError("Source code changed")
        vm.restrict_network(party, addresses)
        own = target / party
        for sub in ("code", "data", "logs", "results"):
            (own / sub).mkdir(parents=True)
        root = f"/srv/vfl/{party}/{run_id}"
        vm.guest(
            party,
            f"sudo find /srv/vfl/{party} -mindepth 1 -maxdepth 1 -type d ! -name tls -exec chown root:root {{}} \\; -exec chmod 700 {{}} \\;",
        )
        vm.guest(
            party,
            f"sudo install -d -m 755 {root} {root}/code; sudo install -d -m 550 -o root -g {party} {root}/input; sudo install -d -m 700 -o {party} -g {party} {root}/data {root}/results {root}/logs {root}/trainings",
        )
        for name, (source_run, path) in mapping(sources, party, cfg).items():
            old = f"/srv/vfl/{party}/{source_run}"
            vm.guest(
                party,
                f"sudo install -o root -g {party} -m 440 {old}/{path} {root}/input/{name}",
            )
        compare = f"from pathlib import Path;import numpy as np;p=Path('{root}/input');pairs=[(p/(s+'_local.npz'),p/('check_'+s+'_local.npz')) for s in ('train','validation')];assert all(set(np.load(a).files)==set(np.load(b).files) and all(np.array_equal(np.load(a)[k],np.load(b)[k]) for k in np.load(a).files) for a,b in pairs);print('owner_matrices_exact')"
        if (
            vm.guest(
                party,
                f"sudo -u {party} /opt/secretflow/bin/python -c "
                + shlex.quote(compare),
            )
            != "owner_matrices_exact"
        ):
            raise ValueError("Source owner matrices differ")
        vm.guest(
            party,
            f"sudo rm {root}/input/check_train_local.npz {root}/input/check_validation_local.npz",
        )
        vm.save(
            own / "logs/source_lineage.json",
            {
                "source_runs": sources,
                "owner_matrices_exact": True,
                "source_queues_passed": True,
            },
        )
        script = f"from pathlib import Path; import hashlib,json; p=Path('{root}/input'); (p/'manifest.json').write_text(json.dumps({{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir()}}))"
        vm.guest(party, "sudo /opt/secretflow/bin/python -c " + shlex.quote(script))
        vm.guest(
            party,
            f"sudo chown root:{party} {root}/input/manifest.json; sudo chmod 440 {root}/input/manifest.json",
        )
        code_paths = []
        for name in CODE:
            owner_module = (
                "m7_security"
                if name == "output_reaudit.py"
                else ("m3_alignment" if name == "bounded_psi.py" else "m5_modeling")
            )
            code_paths.append(repo / "modules" / owner_module / "components" / name)
        for path in [*code_paths, config, module / "notebooks" / NOTEBOOK]:
            dest = own / "code" / path.name
            dest.write_bytes(path.read_bytes())
            temp = f"/tmp/{run_id}_{party}_{path.name}"
            vm.copy_to(dest, party, temp)
            vm.guest(
                party,
                f"sudo install -o root -g root -m 444 {temp} {root}/code/{path.name}; sudo rm {temp}",
            )
        runtime = {
            "party": party,
            "run_id": run_id,
            "dataset": dataset,
            "config": f"{root}/code/output_reaudit.yaml",
            "config_sha256": provider.sha(config),
            "code_sha256": {n: provider.sha(own / "code" / n) for n in CODE},
            "notebook": NOTEBOOK,
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "source_runs": sources,
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
        temp = f"/tmp/{run_id}_{party}_runtime.json"
        vm.copy_to(own / "code/runtime.json", party, temp)
        vm.guest(
            party,
            f"sudo install -o root -g root -m 444 {temp} {root}/code/runtime.json; sudo rm {temp}",
        )
    vm.save(
        target / "declaration.json",
        {
            "step": "S7.P2",
            "source_runs": sources,
            "test_staged": False,
            "prediction_export": False,
            "run_id": run_id,
            "config_sha256": provider.sha(config),
        },
    )
    vm.private_permissions(target)
    return target, run_id


def verify(target, run_id):
    if json.loads((target / "status.json").read_text())["status"] != "passed":
        raise ValueError("Owner execution failed")
    declaration = json.loads((target / "declaration.json").read_text())
    if declaration["test_staged"] or declaration["prediction_export"]:
        raise ValueError("Unregistered export capability")
    for party in vm.PARTIES:
        root = f"/srv/vfl/{party}/{run_id}"
        checks = f"set -e; sudo -u {party} test ! -w {root}/input/train_local.npz; test $(sudo stat -c %a {root}/input) = 550; sudo -u {party} test ! -e {root}/input/test_local.npz; echo passed"
        if vm.guest(party, checks) != "passed":
            raise ValueError("Frozen capability denied")
        nb = json.loads((target / party / "results/party.executed.ipynb").read_text())
        cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        if [c["execution_count"] for c in cells] != list(
            range(1, CODE_CELLS + 1)
        ) or any(o["output_type"] == "error" for c in cells for o in c["outputs"]):
            raise ValueError("Notebook incomplete")
        if (
            not json.loads((target / party / "results/engineering.json").read_text())[
                "status"
            ]
            == "passed"
        ):
            raise ValueError("Owner incomplete")
        own = target / party
        runtime = json.loads((own / "code/runtime.json").read_text())
        if (
            runtime["party"] != party
            or runtime["run_id"] != run_id
            or runtime["source_runs"] != declaration["source_runs"]
            or runtime["config_sha256"] != declaration["config_sha256"]
            or provider.sha(own / "code/output_reaudit.yaml")
            != runtime["config_sha256"]
            or set(runtime["code_sha256"]) != set(CODE)
            or any(
                provider.sha(own / "code" / n) != h
                for n, h in runtime["code_sha256"].items()
            )
        ):
            raise ValueError("Frozen execution provenance changed")
        if not all(
            json.loads((own / "logs/owner_isolation.json").read_text()).values()
        ):
            raise ValueError("Owner isolation failed")
        lineage = json.loads((own / "logs/source_lineage.json").read_text())
        if lineage["source_runs"] != declaration["source_runs"] or not (
            lineage["owner_matrices_exact"] and lineage["source_queues_passed"]
        ):
            raise ValueError("Source lineage incomplete")
    forbidden = ("*.npy", "*.npz", "model.share", "*.trace.log")
    if any(list(target.rglob(p)) for p in forbidden) or list(
        (target / "bob").rglob("private_*audit.json")
    ):
        raise ValueError("Forbidden export")
    bob = f"/srv/vfl/bob/{run_id}"
    if (
        vm.guest(
            "bob",
            f'set -e; test $(sudo find {bob}/input -name "*.npy" | wc -l) = 0; test $(sudo find {bob}/trainings -name "*predictions*" | wc -l) = 0; echo passed',
        )
        != "passed"
    ):
        raise ValueError("Bob score boundary failed")
    report = json.loads(
        (target / "alice/results/private_attribute_audit.json").read_text()
    )
    cfg = yaml.safe_load((target / "alice/code/output_reaudit.yaml").read_text())
    if len(report["details"]) != len(cfg["model_seeds"]) * len(
        cfg["attribute_cells"]
    ) * (len(("exact", "rounded", "binary")) + 1):
        raise ValueError("Incomplete probes")
    actual = {(v["cell"], v["seed"], v["view"]) for v in report["details"]}
    expected = {
        (c, s, v)
        for c in cfg["attribute_cells"]
        for s in cfg["model_seeds"]
        for v in ("baseline", "exact", "rounded", "binary")
    }
    if actual != expected:
        raise ValueError("Probe inventory differs")
    output = json.loads(
        (target / "alice/results/private_output_audit.json").read_text()
    )
    actual = {(v["cell"], v["seed"], v["view"]) for v in output["details"]}
    expected = {
        (c, s, v)
        for c in cfg["output_cells"]
        for s in cfg["model_seeds"]
        for v in ("exact", "rounded", "binary")
    }
    if actual != expected or len(output["details"]) != len(expected):
        raise ValueError("Output inventory differs")
    expected_summary = {
        c + "_" + v for c in cfg["output_cells"] for v in ("exact", "rounded", "binary")
    }
    if (
        set(output["utility"]) != expected_summary
        or set(output["aggregate"]) != expected_summary
    ):
        raise ValueError("Utility coverage differs")

    def finite_intervals(value):
        if isinstance(value, dict):
            if "estimate" in value and (
                not all(
                    __import__("math").isfinite(value[k])
                    for k in ("estimate", "ci_low", "ci_high")
                )
                or value["ci_low"] > value["ci_high"]
            ):
                raise ValueError("Invalid interval")
            for item in value.values():
                finite_intervals(item)
        elif isinstance(value, list):
            for item in value:
                finite_intervals(item)

    finite_intervals(report)
    finite_intervals(output)
    return {
        "status": "passed",
        "bob_scores_disclosed": False,
        "truth_exported": False,
        "test_read": False,
        "production_security": False,
    }


def execute(target, run_id):
    if (target / "status.json").exists() or any(
        (target / party / "logs/host_process.log").exists()
        or next((target / party / "results").iterdir(), None) is not None
        for party in vm.PARTIES
    ):
        raise FileExistsError("Used host audit directory; stage a new runtime")
    environment = os.environ.copy()
    environment["LIMA_HOME"] = vm.LIMA_HOME_PATH
    processes, logs = [], []
    success = False
    try:
        for party in vm.PARTIES:
            root = f"/srv/vfl/{party}/{run_id}"
            script = f"import os,sys; os.chdir('{root}'); sys.path.insert(0,'{root}/code'); from output_reaudit import execute_reaudit; execute_reaudit('{root}/code/runtime.json')"
            log = (target / party / "logs/host_process.log").open("w")
            logs.append(log)
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
                script,
            ]
            processes.append(
                subprocess.Popen(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT
                )
            )
        started = time.monotonic()
        while any(p.poll() is None for p in processes):
            if (
                any(p.poll() not in (None, 0) for p in processes)
                or time.monotonic() - started > vm.ROUND_TIMEOUT_SECONDS
            ):
                raise RuntimeError("Owner exit or timeout")
            time.sleep(vm.CHECK_INTERVAL_SECONDS)
        if any(p.returncode for p in processes):
            raise RuntimeError("Owner exit")
        success = True
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for log in logs:
            log.close()
        cleanup_errors = []
        for party in vm.PARTIES:
            if not success:
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
                provider.export(target, run_id, party, failed=not success)
            except Exception as error:
                cleanup_errors.append(
                    {
                        "party": party,
                        "operation": "export",
                        "error_type": type(error).__name__,
                    }
                )
        vm.save(
            target / "status.json",
            {
                "status": "passed" if success and not cleanup_errors else "failed",
                "exit_codes": [p.poll() for p in processes],
                "cleanup_errors": cleanup_errors,
            },
        )
        vm.private_permissions(target)
    if cleanup_errors:
        raise RuntimeError("Owner export incomplete; see private lifecycle receipt")
    try:
        report = verify(target, run_id)
    except Exception as error:
        receipt = json.loads((target / "status.json").read_text())
        receipt.update(
            status="failed",
            failure_phase="verification",
            error_type=type(error).__name__,
        )
        vm.save(target / "status.json", receipt)
        vm.private_permissions(target)
        raise
    vm.save(target / "verification.json", report)
    vm.private_permissions(target)
    return report


def read_sources(queue_path):
    state = json.loads(Path(queue_path).read_text())
    if state["status"] != "passed" or len(state["completed"]) != len(vm.PREPARED):
        raise ValueError("Source queue incomplete")
    result = {Path(p).parent.name: Path(p).name for p in state["completed"]}
    if set(result) != set(vm.PREPARED):
        raise ValueError("Source dataset coverage differs")
    return result


def queue(workspace, candidate_queue, diagnostic_queue, output):
    e4, e5 = read_sources(candidate_queue), read_sources(diagnostic_queue)
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "completed": [], "current": None}
    vm.save(output / "status.json", state)
    try:
        for dataset in vm.PREPARED:
            sources = {"E4": e4[dataset], "E5": e5[dataset]}
            target, run_id = stage(workspace, sources, output, dataset)
            state["current"] = str(target)
            vm.save(output / "status.json", state)
            execute(target, run_id)
            state["completed"].append(str(target))
        state.update(status="passed", current=None)
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
    parser.add_argument("--candidate-queue", type=Path, required=True)
    parser.add_argument("--diagnostic-queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue(args.workspace, args.candidate_queue, args.diagnostic_queue, args.output)
