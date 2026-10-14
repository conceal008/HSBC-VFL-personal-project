"""Resume unchanged owner preparation into new paths after a host VM failure."""

from __future__ import annotations
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import yaml
import portfolio_repair_vm as provider
from portfolio_source import sha

CELL_COUNT = 3


def audit(local, expected_config, repo):
    """Validate receipts and actual on-VM bytes without exporting owner data."""
    for party in provider.vm.PARTIES:
        own = local / party
        runtime = json.loads((own / "code/runtime.json").read_text())
        if runtime["config_sha256"] != expected_config or runtime["party"] != party:
            raise ValueError("Preparation origin differs")
        for name, digest in runtime["code_sha256"].items():
            if sha(own / "code" / name) != digest:
                raise ValueError("Actual source changed")
            if name == "portfolio_repair_party.py":
                current = repo / "modules/m1_data_selection/components" / name
                if ast.dump(ast.parse(current.read_text())) != ast.dump(
                    ast.parse((own / "code" / name).read_text())
                ):
                    raise ValueError("Executable owner source differs")
        receipt = json.loads((own / "results/engineering.json").read_text())
        if receipt != dict(
            status="prepared",
            party=party,
            train_only_fit=True,
            sealed_confirmation_read=False,
            input_unchanged=True,
            isolation_checks_passed=True,
            training_performed=False,
            physical_isolation=False,
        ):
            raise ValueError("Preparation receipt invalid")
        nb = json.loads((own / "results/party.executed.ipynb").read_text())
        cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        if [c["execution_count"] for c in cells] != list(
            range(1, CELL_COUNT + 1)
        ) or any(o["output_type"] == "error" for c in cells for o in c["outputs"]):
            raise ValueError("Owner notebook incomplete")
        if (
            json.loads((own / "results/NOTEBOOK_COMPLETED.json").read_text())["status"]
            != "passed"
        ):
            raise ValueError("Owner execution missing")
        script = (
            "from pathlib import Path;import json,hashlib;"
            f"p=Path({runtime['root']!r});r=json.loads((p/'code/runtime.json').read_text());"
            f"assert hashlib.sha256((p/'code/runtime.json').read_bytes()).hexdigest()=={sha(own / 'code/runtime.json')!r};"
            "h=lambda x:hashlib.sha256(x.read_bytes()).hexdigest();"
            "assert all(h(p/'code'/n)==v for n,v in r['code_sha256'].items());"
            "assert all(h(p/'input'/n)==v for n,v in r['input_sha256'].items());"
            "m=json.loads((p/'data/manifest.json').read_text());"
            "assert set(m)=={'train.csv','validation.csv'};"
            "assert all(h(p/'data'/n)==v for n,v in m.items());print('passed')"
        )
        if (
            provider.vm.guest(
                party, "sudo /opt/secretflow/bin/python -c " + shlex.quote(script)
            )
            != "passed"
        ):
            raise ValueError("Frozen owner artifacts differ")
        root, user = runtime["root"], runtime["account"]
        checks = f"set -e; sudo -u {user} test ! -r {root}/sealed/probe; sudo -u {party} test ! -r {root}/data/train.csv; sudo -u {user} test ! -w {root}/input/train.csv; echo passed"
        if provider.vm.guest(party, checks) != "passed":
            raise ValueError("Restored owner permissions differ")
    return dict(
        status="verified",
        data_exported=False,
        sealed_opened=False,
        actual_source_verified=True,
    )


def resume(workspace, previous):
    workspace, previous = Path(workspace), Path(previous)
    repo = Path(__file__).resolve().parents[provider.REPO_PARENT_DEPTH]
    config = repo / "modules/m1_data_selection/configs/portfolio_repair_freeze.yaml"
    cfg = yaml.safe_load(config.read_text())
    prior = json.loads((previous / "status.json").read_text())
    if prior["status"] != "failed" or prior.get("error_type") != "TimeoutExpired":
        raise ValueError("This recovery only accepts the recorded interrupted queue")
    run_id = "S1P8R_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = workspace / "全数据集准备" / run_id
    root.mkdir(mode=provider.vm.PRIVATE_DIRECTORY_MODE)
    state = dict(
        status="running",
        pid=os.getpid(),
        source_run=previous.name,
        prepared_runs={},
        reused=[],
        completed=[],
        current_dataset=None,
    )
    provider.vm.save(root / "status.json", state)
    try:
        for dataset in cfg["datasets"]:
            state["current_dataset"] = dataset
            provider.vm.save(root / "status.json", state)
            old = previous / dataset
            if str(old) in prior["completed"]:
                checks = audit(old, sha(config), repo)
                state["reused"].append(dataset)
                chosen = old
            else:
                chosen = root / dataset
                chosen.mkdir()
                provisioned = workspace / "全数据集投递" / run_id / dataset
                provider.provision(workspace / "数据集", provisioned, dataset, config)
                provider.stage(chosen, provisioned, dataset, repo)
                owners = [
                    provider.execute(chosen, party) for party in provider.vm.PARTIES
                ]
                provider.vm.save(
                    chosen / "verification.json",
                    dict(
                        status="prepared",
                        owners=owners,
                        confirmation_unopened=True,
                        training_performed=False,
                    ),
                )
                checks = audit(chosen, sha(config), repo)
            state["prepared_runs"][dataset] = chosen.parent.name
            state["completed"].append(str(chosen))
            provider.vm.save(
                root / (dataset + "_origin.json"),
                dict(source=str(chosen), verification=checks),
            )
            provider.vm.save(root / "status.json", state)
        state.update(status="passed", current_dataset=None)
    except BaseException as exc:
        state.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        provider.vm.save(root / "status.json", state)
        provider.vm.private_permissions(root)
    return root
