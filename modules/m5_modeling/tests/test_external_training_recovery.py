"""Synthetic preservation checks for the recovery controller, not new training."""

from pathlib import Path
import ast
import hashlib
import inspect
import json
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import external_training_recovery as r  # noqa: E402


def test_skip_guard_is_the_only_changed_original_statement():
    original = ast.parse(inspect.getsource(r.nf.train_full))
    patched = r.patched_tree()
    guards = 0
    for node in ast.walk(patched):
        if (
            isinstance(node, ast.For)
            and ast.unparse(node.target) == "(cell, condition)"
        ):
            assert isinstance(node.body[0], ast.Assign) and isinstance(
                node.body[1], ast.If
            )
            assert isinstance(node.body[1].body[-1], ast.Continue)
            del node.body[:2]
            guards += 1
    assert guards == 1 and ast.dump(patched, include_attributes=False) == ast.dump(
        original, include_attributes=False
    )


def fixture(tmp_path):
    (tmp_path / "code").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "recovery_input").mkdir()
    data = {
        "root": str(tmp_path),
        "party": "alice",
        "train": {"x": np.ones((2, 1))},
        "validation": {"x": np.ones((1, 1))},
    }
    for split in ("train", "validation"):
        for sub in ("data", "recovery_input"):
            np.savez(tmp_path / sub / (split + "_local.npz"), x=data[split]["x"])
    names = {
        f"L0_{family}_{int(enhanced)}_seed11"
        for family in ("LR", "tree")
        for enhanced in (False, True)
    } | {"B0_seed11"}
    hashes = {}
    for name in names:
        folder = tmp_path / "trainings" / name / "results"
        folder.mkdir(parents=True)
        for split in ("train", "validation"):
            np.save(
                folder / (split + "_predictions.npy"),
                np.full((len(data[split]["x"]), 1), 0.5),
            )
        hashes[name] = {
            str(f.relative_to(folder.parent)): hashlib.sha256(
                f.read_bytes()
            ).hexdigest()
            for f in folder.iterdir()
        }
    receipt = {
        "party": "alice",
        "config_sha256": "control",
        "reused_fits": sorted(names),
        "fit_sha256": hashes,
    }
    (tmp_path / "code" / r.MANIFEST_NAME).write_text(json.dumps(receipt))
    return data, receipt


def test_prepare_requires_exact_owner_arrays_and_protocol(tmp_path, monkeypatch):
    data, receipt = fixture(tmp_path)
    monkeypatch.setattr(r.nf, "prepare", lambda *a: data)
    assert r.prepare_recovered("alice", {"config_sha256": "control"}, {}) is data
    np.savez(tmp_path / "recovery_input/train_local.npz", x=np.zeros((2, 1)))
    with pytest.raises(ValueError, match="arrays"):
        r.prepare_recovered("alice", {"config_sha256": "control"}, {})
    with pytest.raises(ValueError, match="ownership"):
        r.prepare_recovered("bob", {"config_sha256": "control"}, {})


def test_recovered_fit_cannot_change_or_be_unregistered(tmp_path):
    data, _ = fixture(tmp_path)
    assert r.validate_recovered(data, "B0_seed11")
    data["party"] = "bob"
    assert r.validate_recovered(data, "B0_seed11")
    data["party"] = "alice"
    with pytest.raises(ValueError, match="Undeclared"):
        r.validate_recovered(data, "partial_seed22")
    p = tmp_path / "trainings/B0_seed11/results/train_predictions.npy"
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        r.validate_recovered(data, "B0_seed11")


def test_probability_guard_even_if_bad_copy_has_matching_digest(tmp_path):
    data, receipt = fixture(tmp_path)
    p = tmp_path / "trainings/B0_seed11/results/train_predictions.npy"
    np.save(p, np.full((2, 1), 2.0))
    receipt["fit_sha256"]["B0_seed11"]["results/train_predictions.npy"] = (
        hashlib.sha256(p.read_bytes()).hexdigest()
    )
    (tmp_path / "code" / r.MANIFEST_NAME).write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="probability"):
        r.validate_recovered(data, "B0_seed11")


def test_local_reference_reuse_is_complete_block_only(tmp_path, monkeypatch):
    data, receipt = fixture(tmp_path)
    monkeypatch.setattr(r.nf, "local_references", lambda *a: {"fresh": True})
    assert len(r.local_references_recovered(data, 11, "control")) == 4
    assert r.local_references_recovered(data, 22, "control") == {"fresh": True}
    receipt["reused_fits"].remove("L0_LR_0_seed11")
    (tmp_path / "code" / r.MANIFEST_NAME).write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Partial"):
        r.local_references_recovered(data, 11, "control")


def test_recovery_entry_checks_source_and_controller_before_execution(
    tmp_path, monkeypatch
):
    helper = tmp_path / "helper.py"
    helper.write_text("control")
    runtime = {
        "config": str(tmp_path / "config.yaml"),
        "code_sha256": {
            "nonlinear_full.py": hashlib.sha256(
                Path(r.nf.__file__).read_bytes()
            ).hexdigest()
        },
        "controller_sha256": {
            "helper.py": hashlib.sha256(helper.read_bytes()).hexdigest()
        },
    }
    monkeypatch.setattr(
        r,
        "patched_tree",
        lambda: ast.parse("def train_full(runtime):\n return 'known-control'\n"),
    )
    assert r.train_recovered(runtime) == "known-control"
    helper.write_text("changed")
    with pytest.raises(ValueError, match="controller"):
        r.train_recovered(runtime)
    runtime["code_sha256"]["nonlinear_full.py"] = "bad"
    with pytest.raises(ValueError, match="math source"):
        r.train_recovered(runtime)
