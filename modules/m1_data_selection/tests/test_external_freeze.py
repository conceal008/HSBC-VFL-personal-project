"""Outcome-independent split, train-only transform and sealed-capability regressions."""

from pathlib import Path
import sys
import csv
import numpy as np
import pandas as pd
import pytest
import yaml

COMPONENTS = Path(__file__).resolve().parents[1] / "components"
sys.path.insert(0, str(COMPONENTS))
import external_freeze as ef  # noqa: E402
import external_party as ep  # noqa: E402


def test_provision_drops_identifiers_and_bob_outcomes_and_is_immutable(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "source.csv"
    with source.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID_code", "target", "a", "b"])
        writer.writerows([[f"PRIVATE_UNIT_ID_{i}", i % 2, i, -i] for i in range(200)])
    spec = {
        "source": "source.csv",
        "format": "csv",
        "columns": ["ID_code", "target", "a", "b"],
        "label": "target",
        "treatment": None,
        "sampling_fraction": 1.0,
        "parties": {"alice": {"features": ["a"]}, "bob": {"features": ["b"]}},
    }
    cfg = {
        "hash_bits": 64,
        "seeds": {"sampling": 11, "split": 22},
        "split_fractions": [0.6, 0.2, 0.2],
        "datasets": {"unit": spec},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    original = ef.sha(source)
    out = tmp_path / "new"
    receipt = ef.provision(raw, out, "unit", path)
    assert not receipt["confirmation_used"] and ef.sha(source) == original
    for p in ef.PARTIES:
        for split in ["train", "validation"]:
            content = (out / p / f"{split}.csv").read_text()
            assert "PRIVATE_UNIT_ID" not in content
            assert ("target" in content) == (p == "alice")
        assert (out / p / "confirmation_raw.csv").stat().st_mode & 0o777 == 0
    with pytest.raises(FileExistsError):
        ef.provision(raw, out, "unit", path)
    with pytest.raises(ValueError, match="pollute"):
        ef.provision(raw, raw / "new", "unit", path)
    # Split decisions accept no outcomes or raw identifiers.
    for i in range(100):
        assert ef.decision("unit", i, cfg, spec) == ef.decision("unit", i, cfg, spec)


def test_validation_never_fits_median_scale_or_categories():
    frames = {
        "train": pd.DataFrame(
            {
                "record_id": ["a", "b", "c"],
                "x": [0.0, 2.0, np.nan],
                "c": ["A", "B", "A"],
            }
        ),
        "validation": pd.DataFrame(
            {"record_id": ["v", "w"], "x": [1000.0, np.nan], "c": ["NEW", None]}
        ),
    }
    spec = {"features": ["x", "c"], "categorical": ["c"]}
    transformed, state = ep.encode(frames, spec)
    assert state["median"] == [1.0] and state["mean"] == [1.0]
    assert state["categorical"] == {"c": ["A", "B"]}
    assert np.array_equal(
        transformed["validation"].iloc[:, 2:].to_numpy(), np.zeros((2, 2))
    )
    with pytest.raises(ValueError, match="development"):
        ep.encode(dict(frames, confirmation_raw=frames["validation"]), spec)
    numeric = {s: f[["record_id", "x"]] for s, f in frames.items()}
    result, _ = ep.encode(numeric, {"features": ["x"], "categorical": []})
    assert result["train"].shape == (3, 2)


def test_fail_closed_permission_probe_requires_access_denied(tmp_path):
    assert ep.denied(tmp_path / "missing") is False
    file = tmp_path / "exists"
    file.write_text("unit synthetic probe")
    assert ep.denied(file) is False


def test_preparer_rejects_bob_outcomes_before_loading_data(tmp_path, monkeypatch):
    root = tmp_path
    (root / "code").mkdir()
    for sub in ("data", "results", "logs"):
        (root / sub).mkdir()
    (root / "code/external_freeze.yaml").write_text("unit")
    monkeypatch.setattr(ep, "isolation", lambda _: {"unit_double": True})
    runtime = {
        "root": str(root),
        "party": "bob",
        "dataset": "unit",
        "config_sha256": ep.sha(root / "code/external_freeze.yaml"),
        "code_sha256": {},
        "input_sha256": {},
        "feature_spec": {"features": ["x"], "categorical": []},
        "outcomes": ["label"],
    }
    with pytest.raises(ValueError, match="Bob outcome"):
        ep.prepare(runtime)
