"""Synthetic-only contract checks for the external L0 runner."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler


SOURCE = Path(__file__).resolve().parents[1] / "components/external_l0.py"
SPEC = importlib.util.spec_from_file_location("external_l0", SOURCE)
assert SPEC and SPEC.loader
l0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(l0)

LAUNCH_SOURCE = SOURCE.with_name("launch_external_l0.py")
LAUNCH_SPEC = importlib.util.spec_from_file_location("launch_external_l0", LAUNCH_SOURCE)
assert LAUNCH_SPEC and LAUNCH_SPEC.loader
launcher = importlib.util.module_from_spec(LAUNCH_SPEC)
LAUNCH_SPEC.loader.exec_module(launcher)


def test_load_splits_rejects_cross_split_duplicate_and_post_outcome_feature(tmp_path):
    frames = {
        "train": pd.DataFrame({"record_id": ["a", "b"], "numeric__age": [1, 2], "label": [0, 1]}),
        "validation": pd.DataFrame({"record_id": ["c", "d"], "numeric__age": [3, 4], "label": [0, 1]}),
        "test": pd.DataFrame({"record_id": ["e", "f"], "numeric__age": [5, 6], "label": [0, 1]}),
    }
    for split, frame in frames.items():
        frame.to_csv(tmp_path / f"{split}.csv", index=False)
    manifest = {f"{split}.csv": l0.sha256(tmp_path / f"{split}.csv") for split in frames}
    runtime = {"data_dir": str(tmp_path), "accepted_manifest": manifest}
    spec = {"dataset": "uci_bank_marketing", "expected_feature_count": 1,
            "excluded_source_features": ["duration"]}
    loaded, features, _ = l0.load_splits(runtime, spec)
    assert features == ["numeric__age"]
    assert len(loaded) == 3

    frames["test"].loc[0, "record_id"] = "a"
    frames["test"].to_csv(tmp_path / "test.csv", index=False)
    runtime["accepted_manifest"]["test.csv"] = l0.sha256(tmp_path / "test.csv")
    with pytest.raises(ValueError, match="overlap"):
        l0.load_splits(runtime, spec)

    frames["test"].loc[0, "record_id"] = "e"
    for split, frame in frames.items():
        frame.insert(2, "numeric__duration", [1, 2])
        frame.to_csv(tmp_path / f"{split}.csv", index=False)
        runtime["accepted_manifest"][f"{split}.csv"] = l0.sha256(tmp_path / f"{split}.csv")
    spec["expected_feature_count"] = 2
    with pytest.raises(ValueError, match="excluded"):
        l0.load_splits(runtime, spec)


def test_alpha_selection_never_reads_test_frame(monkeypatch):
    train = pd.DataFrame({"numeric__x": [-2.0, -1.0, 1.0, 2.0],
                          "label": [0, 0, 1, 1]})
    valid = train.copy()
    frames = {"train": train, "validation": valid,
              "test": pd.DataFrame({"numeric__x": [float("nan")], "label": [1]})}
    called = []

    def fake_fit(x_train, y_train, x_eval, alpha, seed):
        called.append((len(x_train), len(x_eval), alpha, seed))
        return np.array([0.1, 0.2, 0.8, 0.9])

    monkeypatch.setattr(l0, "fit_score", fake_fit)
    spec = {"dataset": "uci_bank_marketing", "model_seeds": [11, 22],
            "alpha_grid": [0.0001, 0.001]}
    scaler = StandardScaler().fit(train[["numeric__x"]])
    assert l0.select_alpha(frames, ["numeric__x"], spec, scaler) == 0.0001
    assert len(called) == 4
    assert all(n_train == 4 and n_eval == 4 for n_train, n_eval, _, _ in called)


def test_centered_qini_positive_for_well_ranked_heterogeneous_effect():
    # First half benefits from treatment; second half does not.
    t = np.tile([0, 1], 100)
    y = np.zeros(len(t), dtype=int)
    y[:100] = t[:100]
    score = np.r_[np.ones(100), np.zeros(100)]
    assert l0.centered_qini(y, t, score) > 0
    assert np.isfinite(l0.uplift_at_k(y, t, score, 0.5))


@pytest.mark.parametrize("dataset", ["uci_bank_marketing", "hillstrom_email_marketing"])
def test_private_evaluation_uses_only_alice_synthetic_splits(tmp_path, monkeypatch, dataset):
    rng = np.random.default_rng(20260930)
    data = tmp_path / "alice_data"
    result = tmp_path / "alice_result"
    data.mkdir()
    result.mkdir()
    manifest = {}
    for split, n in (("train", 320), ("validation", 160), ("test", 160)):
        x = rng.normal(size=n)
        frame = pd.DataFrame({"record_id": [f"{split}-{i}" for i in range(n)],
                              "numeric__x": x, "numeric__z": rng.normal(size=n)})
        if dataset == "uci_bank_marketing":
            frame["label"] = (x + rng.normal(size=n) > 0).astype(int)
        else:
            frame["treatment"] = rng.choice([0, 1, 2], size=n)
            prob = 1 / (1 + np.exp(-(0.4 * x +
                            0.8 * (frame["treatment"].to_numpy() == 1) * (x > 0))))
            frame["label"] = rng.binomial(1, prob)
        path = data / f"{split}.csv"
        frame.to_csv(path, index=False)
        manifest[path.name] = l0.sha256(path)
    config = tmp_path / "config.yaml"
    config.write_text("synthetic: true\n")
    runtime = {"dataset": dataset, "prepared_experiment": "synthetic-test",
               "data_dir": str(data), "result_dir": str(result),
               "public_config_snapshot": str(config), "accepted_manifest": manifest,
               "code_sha256": "synthetic-code", "git_sha": "synthetic-git"}
    monkeypatch.setattr(l0, "check_isolation", lambda _: {"sandbox": True})
    spec = {"dataset": dataset, "expected_feature_count": 2,
            "excluded_source_features": ["duration", "visit"],
            "model_seeds": [11, 22], "split_seed": 20260927,
            "bootstrap_seed": 20260930, "bootstrap_repeats": 20,
            "alpha_grid": [0.0001, 0.001], "top_k_fraction": 0.1,
            "treatment_arms": [1, 2]}
    receipt = l0.evaluate(runtime, spec)
    metrics = json.loads((result / "l0_private_metrics.json").read_text())
    assert receipt["seeds_completed"] == 2
    assert len(metrics["metrics"]) == (2 if dataset == "uci_bank_marketing" else 4)
    assert (result / "结果分析.md").is_file()
    assert "record_id" not in (result / "结果分析.md").read_text()


def test_isolation_probe_fails_closed_outside_sandbox(tmp_path):
    own = tmp_path / "own"
    peer = tmp_path / "peer"
    own.mkdir()
    peer.mkdir()
    for path in (own / "train.csv", peer / "train.csv", peer / "input.csv"):
        path.write_text("present")
    runtime = {"data_dir": str(own),
               "probes": {"peer_input": str(peer / "input.csv"),
                          "peer_output": str(peer / "train.csv"),
                          "full_source": str(own / "train.csv"), "port": 7}}
    with pytest.raises(RuntimeError, match="Isolation preflight failed"):
        l0.check_isolation(runtime)


def test_launcher_registers_only_status_and_rejects_missing_inputs(tmp_path, monkeypatch):
    workspace = tmp_path / "8-9月"
    dataset = "uci_bank_marketing"
    prepared = dataset + "_synthetic"
    own = workspace / "联邦隔离产物" / prepared / "alice"
    peer = workspace / "联邦隔离产物" / prepared / "bob"
    own_input = workspace / "联邦隔离输入" / prepared / "alice"
    peer_input = workspace / "联邦隔离输入" / prepared / "bob"
    source_result = workspace / "联邦隔离结果" / prepared / "alice"
    log = workspace / "实验日志" / prepared
    source = workspace / "数据集" / "synthetic.csv"
    for folder in (own, peer, own_input, peer_input, source_result, log, source.parent):
        folder.mkdir(parents=True)
    for path in (own / "train.csv", peer / "train.csv", own_input / "input.csv",
                 peer_input / "input.csv", source):
        path.write_text("synthetic")
    (source_result / "data_manifest.json").write_text(json.dumps({"train.csv": "synthetic"}))
    (log / "declaration.json").write_text(json.dumps({"experiment_id": prepared,
        "data_source": {"relative_path": "synthetic.csv"}}))
    (workspace / "实验日志/实验索引.md").write_text("# synthetic index\n")
    runtime_python = workspace / "数据处理环境/v1/bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("synthetic runtime")
    monkeypatch.setattr(launcher.sys, "platform", "darwin")
    real_is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda path: True if str(path) == "/usr/bin/sandbox-exec"
                        else real_is_file(path))

    def fake_run(command, **kwargs):
        if command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, stdout="synthetic-head\n")
        private = Path(kwargs["cwd"])
        (private / "NOTEBOOK_COMPLETED.json").write_text("{}")
        (private / "l0_private_metrics.json").write_text("{}")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    output = launcher.launch(workspace, dataset, prepared)
    status = json.loads((output / "status.json").read_text())
    assert status["l0_status"] == "passed"
    assert status["secure_joint_training"] == "blocked"
    assert "peer" not in (workspace / "实验日志/实验索引.md").read_text()
    profiles = list((workspace / "联邦训练结果").glob("*/alice/sandbox.sb"))
    assert len(profiles) == 1
    assert str(own) in profiles[0].read_text()
    assert str(peer) not in profiles[0].read_text()

    (peer_input / "input.csv").unlink()
    with pytest.raises(FileNotFoundError, match="Missing accepted"):
        launcher.launch(workspace, dataset, prepared)
