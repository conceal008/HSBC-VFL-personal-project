"""Audit contract fixtures are synthetic and do not stand in for live VM checks."""
from pathlib import Path
import json
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import functional_acceptance as audit  # noqa: E402


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture
def audit_fixture(tmp_path):
    root = tmp_path / "round"
    shared = {"model_seeds": [11, 22, 33, 44, 55], "learning_rates": [0.1],
              "l0_c_grid": [1], "epochs": 2, "bootstrap_repeats": 30}
    write(root / "status.json", {"status": "passed"})
    write(root / "declaration.json", {"dataset": "synthetic", "prepared_experiment": "prepared"})
    metric = {"estimate": 0.6, "ci_low": 0.5, "ci_high": 0.7, "valid_replicates": 30}
    evaluation = {"status": "completed", "selection": "validation only", "metrics": [],
                  "paired_differences": [], "output_attack_diagnostics": []}
    for seed in shared["model_seeds"]:
        for route in audit.ROUTES:
            evaluation["metrics"].append({"seed": seed, "route": route, "metrics": {"roc_auc": metric}})
            evaluation["output_attack_diagnostics"].append({"seed": seed, "route": route,
                "loss_attack_auc": 0.5, "ci_low": 0.4, "ci_high": 0.6})
        for route in audit.COMPARATORS:
            evaluation["paired_differences"].append({"seed": seed, "comparison": route, "metrics": {"roc_auc": metric}})
    for party in ["alice", "bob"]:
        p = root / party
        for name, payload in [
            ("logs/isolation.json", {"synthetic_permission_probe": True}),
            ("logs/tls_checks.json", {"synthetic_certificate_probe": True}),
            ("logs/output_boundary.json", {"frozen_input_unchanged": True,
                "joint_weights": "secret_shares_only", "bob_plaintext_output_check": "passed"}),
            ("results/NOTEBOOK_COMPLETED.json", {"status": "passed"}),
            ("results/party.executed.ipynb", {"cells": [
                {"cell_type": "code", "execution_count": i, "outputs": []} for i in [1, 2, 3]]})]:
            write(p / name, payload)
        (p / "code").mkdir()
        config = p / "code/local_functional.yaml"
        config.write_text(yaml.safe_dump({"shared": shared}))
        source = p / "code/source.py"
        source.write_text("# synthetic fixture\n")
        hashes = {f.name: audit.sha(f) for f in [config, source]}
        write(p / "code/runtime.json", {"code_sha256": hashes})
        original = tmp_path / "联邦隔离产物/prepared" / party / "train.csv"
        original.parent.mkdir(parents=True)
        original.write_text("synthetic fixture only\n")
        manifest = {"train.csv": audit.sha(original)}
        write(p / "data/manifest.json", manifest)
        names = [f"{route}_seed{seed}" for seed in shared["model_seeds"] for route in audit.SECURE_ROUTES]
        if party == "alice":
            names += [f"L0_{candidate}_seed{seed}" for seed in shared["model_seeds"] for candidate in [0, 1]]
        for name in names:
            f = p / "trainings" / name
            for directory in ["code", "data", "results", "logs"]:
                (f / directory).mkdir(parents=True)
            for asset in [source, config]:
                (f / "code" / asset.name).write_bytes(asset.read_bytes())
            (f / "code/environment.lock").write_text("synthetic dependency lock\n")
            write(f / "data/input_references.json", {"frozen_input": manifest, "derived_input": {"synthetic": "hash"}})
            if name.startswith(("L1_", "L3_")):
                write(f / "logs/progress.json", {"completed_epochs": 2, "planned_epochs": 2})
            if party == "alice":
                write(f / "results/status.json", {"training": "passed"})
                write(f / "results/validation_score.json", {"selection_split": "validation"})
                for split in ["train", "validation", "test"]:
                    (f / "results" / f"{split}_predictions.npy").write_bytes(b"synthetic placeholder, never loaded")
    write(root / "alice/results/private_evaluation.json", evaluation)
    return root, tmp_path


def test_completed_synthetic_artifact_contract(audit_fixture):
    root, workspace = audit_fixture
    result = audit.verify_round(root, workspace)
    assert result["checks"]["alice"]["training_directories"] == 20
    assert result["checks"]["bob"]["training_directories"] == 10
    assert result["evaluation"]["paired_comparisons"] == 10
    assert "roc_auc" not in json.dumps(result)


@pytest.mark.parametrize("mutation", ["incomplete", "bob_output", "input_changed", "shares_exported", "bad_ci", "source_changed"])
def test_audit_rejects_incomplete_or_forbidden_artifacts(audit_fixture, mutation):
    root, workspace = audit_fixture
    if mutation == "incomplete":
        write(root / "status.json", {"status": "failed"})
    elif mutation == "bob_output":
        (root / "bob/results/test_predictions.npy").write_bytes(b"synthetic")
    elif mutation == "input_changed":
        (workspace / "联邦隔离产物/prepared/alice/train.csv").write_text("modified synthetic input")
    elif mutation == "shares_exported":
        (root / "alice/results/model.share").write_bytes(b"synthetic")
    elif mutation == "source_changed":
        (root / "alice/code/source.py").write_text("# changed\n")
    else:
        p = root / "alice/results/private_evaluation.json"
        report = json.loads(p.read_text())
        report["metrics"][0]["metrics"]["roc_auc"]["ci_low"] = float("nan")
        write(p, report)
    with pytest.raises(ValueError):
        audit.verify_round(root, workspace)


@pytest.mark.parametrize('sigmoid_family', ['sr', 'df'])
def test_guarded_audit_rejects_training_saturation(audit_fixture, sigmoid_family):
    root, workspace = audit_fixture
    for party in ['alice', 'bob']:
        path = root / party / 'code/local_functional.yaml'
        cfg = yaml.safe_load(path.read_text())
        cfg['shared'].update(sigmoid=sigmoid_family, training_boundary_limit=.01)
        path.write_text(yaml.safe_dump(cfg))
        runtime = root / party / 'code/runtime.json'
        payload = json.loads(runtime.read_text())
        payload['code_sha256']['local_functional.yaml'] = audit.sha(path)
        write(runtime, payload)
        write(root / party / 'logs/tls_checks.json', {'protocol_trace_disabled': True})
        for fit in (root / party / 'trainings').iterdir():
            (fit / 'code/local_functional.yaml').write_bytes(path.read_bytes())
            if party == 'alice':
                write(fit / 'results/numerical_checks.json', {'probabilities_valid': True,
                    'training_boundary_fraction': 0., 'boundary_limit': .01})
    if sigmoid_family == 'df':
        from functional_vm import private_permissions
        private_permissions(root)
    assert audit.verify_round(root, workspace)['checks']['alice']['isolation'] == 'pass'
    if sigmoid_family == 'df':
        (root / 'status.json').chmod(0o644)
        with pytest.raises(ValueError, match='permissions'):
            audit.verify_round(root, workspace)
        (root / 'status.json').chmod(0o600)
    fit = next((root / 'alice/trainings').glob('L3_*'))
    write(fit / 'results/numerical_checks.json', {'probabilities_valid': True,
        'training_boundary_fraction': .2, 'boundary_limit': .01})
    with pytest.raises(ValueError):
        audit.verify_round(root, workspace)
