"""Disclosure-boundary and frozen-output tests with synthetic summary fixtures."""
from pathlib import Path
import json
import sys
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import result_archive as ra  # noqa: E402


def test_projection_excludes_hidden_record_metadata_and_gradient():
    payload, excluded = ra.approved_payload({"status": "passed", "models": {"A": {
        "roc_auc": {"estimate": 0.7, "ci_low": 0.6, "ci_high": 0.8},
        "predictions": [0.1, 0.2], "y": [0, 1], "credentials": "synthetic_fixture"}},
        "records": {"ids": ["synthetic_1"]}}, "feature")
    assert payload["models"]["A"]["roc_auc"]["estimate"] == 0.7
    assert "predictions" not in payload["models"]["A"]
    assert "y" not in payload["models"]["A"]
    assert "credentials" not in payload["models"]["A"]
    assert "/records" in excluded
    payload, omitted = ra.approved_payload({"curves": {"A_seed11": [{"epoch": 1,
        "full_unpadded_gradient_norm": 0.4, "validation": {"roc_auc": 0.7}}]}}, "convergence")
    assert "full_unpadded_gradient_norm" not in payload["curves"]["A_seed11"][0]
    assert omitted


def test_unreviewed_family_vector_and_nonfinite_fail_closed():
    for source, kind in [({"status": "passed"}, "unknown"),
                         ({"models": {"A": [0.1] * 10}}, "feature"),
                         ({"models": {"A": float("nan")}}, "feature"),
                         ({"records": {}}, "feature")]:
        with pytest.raises(ValueError):
            ra.approved_payload(source, kind)
    with pytest.raises(ValueError):
        ra.project(object(), [])


def test_archive_preserves_ci_and_original_failure_without_raw_io(tmp_path):
    workspace = tmp_path / "work"
    root = workspace / "functional/synthetic_bank/alice/results"
    root.mkdir(parents=True)
    source = root / "summary.json"
    source.write_text(json.dumps({"status": "completed", "models": {"A": {
        "roc_auc": {"estimate": 0.7, "ci_low": 0.6, "ci_high": 0.8}}},
        "model_seeds": [11, 22, 33, 44, 55], "records": {"ids": ["synthetic_1"]}}))
    code = root.parent / "code"
    code.mkdir()
    (code / "runtime.json").write_text(json.dumps({"party": "alice", "git_sha": "synthetic",
        "config": "/guest/config.yaml", "config_sha256": "synthetic"}))
    (code / "config.yaml").write_text(yaml.safe_dump({"shared": {"split_seed": 42},
        "experiment": {"order_seeds": {"11": 101}}}))
    (workspace / "functional/status.json").write_text('{"status":"failed","exit_codes":[1,0]}')
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"step_id": "S9.R1", "datasets": ["synthetic_bank"],
        "sources": [{"step": "S5.E2", "kind": "feature", "pattern": "**/summary.json",
                     "classification": "synthetic_fixture"}],
        "status_folders": ["functional"], "pending": ["not run"], "limitations": ["synthetic"]}))
    output = tmp_path / "results"
    index = ra.archive(workspace, cfg, output, tmp_path)
    assert index["execution_inventory"][0]["original_status"] == "failed"
    result = json.loads((output / "r000.json").read_text())
    assert result["results"]["models"]["A"]["roc_auc"]["ci_low"] == 0.6
    assert "/records" in result["omitted_fields"]
    assert result["config_provenance"]["seed_protocol"]["shared"]["split_seed"] == 42
    assert ra.display_archive(output)["report_count"] == 1
    frozen = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        ra.archive(workspace, cfg, output, tmp_path)
    assert frozen == {p.name: p.read_bytes() for p in output.iterdir()}


def test_empty_and_duplicated_source_do_not_create_outputs(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    cfg = tmp_path / "config.yaml"
    config = {"step_id": "S9.R1", "datasets": [], "sources": [
        {"step": "test", "kind": "timing", "pattern": "*.json", "classification": "synthetic"}],
        "status_folders": [], "pending": [], "limitations": []}
    cfg.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        ra.archive(workspace, cfg, tmp_path / "out", tmp_path)
    assert not (tmp_path / "out").exists()
    (workspace / "test.json").write_text('{"seconds":[1,2]}')
    config["sources"] *= 2
    cfg.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        ra.archive(workspace, cfg, tmp_path / "out", tmp_path)
    assert not (tmp_path / "out").exists()
    assert ra.config_provenance(workspace / "test.json", workspace)["config_path"] is None
