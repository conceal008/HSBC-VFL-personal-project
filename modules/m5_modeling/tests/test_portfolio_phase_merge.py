"""Reject wrong recipient, incomplete phases and mismatched owner arrays."""

from pathlib import Path
import json
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import portfolio_phase_merge as m  # noqa: E402


def setup(tmp_path, monkeypatch):
    (tmp_path / "code").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "input").mkdir()
    (tmp_path / "input/manifest.json").write_text("{}")
    cfg = tmp_path / "code/config.yaml"
    cfg.write_text(
        "shared: {model_seeds: [11, 22, 33, 44, 55]}\nexperiment: {conditions: {A0: {}, B0: {}}}"
    )
    data = {"root": str(tmp_path), "party": "alice"}
    receipt = {
        "config_sha256": "control",
        "reused_fits": [
            f"{c}_seed{s}"
            for s in (11, 22, 33, 44, 55)
            for c in ("A0", "B0", "L0_LR_0", "L0_tree_0")
        ],
        "source_runs": ["phase1", "phase2"],
    }
    for split in m.nf.SPLITS:
        np.savez(tmp_path / "data" / (split + "_local.npz"), x=np.ones((3, 2)))
        for phase in receipt["source_runs"]:
            p = tmp_path / "recovery_input" / phase
            p.mkdir(parents=True, exist_ok=True)
            np.savez(p / (split + "_local.npz"), x=np.ones((3, 2)))
    monkeypatch.setattr(m.nf, "prepare", lambda *a: data)
    monkeypatch.setattr(m.control, "manifest", lambda *a: receipt)
    checked = []
    monkeypatch.setattr(m.control, "validate_recovered", lambda d, n: checked.append(n))
    monkeypatch.setattr(m.nf, "finalize", lambda d, r: {"records": r})
    monkeypatch.setattr(m.nf.ad, "manifest_check", lambda *a: True)
    runtime = {
        "party": "alice",
        "config": str(cfg),
        "config_sha256": "control",
        "code_sha256": {},
        "controller_sha256": {},
    }
    return runtime, receipt, checked


def test_merge_uses_exact_five_blocks_and_original_finalizer(tmp_path, monkeypatch):
    runtime, _, checked = setup(tmp_path, monkeypatch)
    result = m.merge(runtime)
    assert (
        set(result["records"]) == {"11", "22", "33", "44", "55"} and len(checked) == 20
    )
    receipt = json.loads((tmp_path / "logs/phase_merge.json").read_text())
    assert receipt["owner_arrays_exact"] and not receipt["new_training_performed"]


def test_merge_rejects_recipient_incomplete_and_arrays(tmp_path, monkeypatch):
    runtime, receipt, _ = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="recipient"):
        m.merge(dict(runtime, party="bob"))
    old = receipt["reused_fits"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        m.merge(runtime)
    receipt["reused_fits"].append(old)
    np.savez(tmp_path / "recovery_input/phase2/train_local.npz", x=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="arrays"):
        m.merge(runtime)


def test_merge_rejects_source_change_before_reading_rows(tmp_path, monkeypatch):
    runtime, _, _ = setup(tmp_path, monkeypatch)
    runtime["code_sha256"] = {"config.yaml": "wrong"}
    with pytest.raises(ValueError, match="source"):
        m.merge(runtime)
