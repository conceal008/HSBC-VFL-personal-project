"""Corruption is a rejected fit, never an inferred success receipt."""

from pathlib import Path
import json
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import portfolio_recovery_health as h  # noqa: E402


def fixture(tmp_path, party="alice"):
    (tmp_path / "data").mkdir()
    for split in ("train", "validation"):
        np.savez(
            tmp_path / "data" / (split + "_local.npz"),
            x=np.ones((3, 2)),
            y=np.ones(3),
            t=np.zeros(3),
        )
    for name in ("A0_seed11", "B0_seed11"):
        p = tmp_path / "trainings" / name
        for sub in ("code", "data", "results", "logs"):
            (p / sub).mkdir(parents=True)
        (p / "data/input_references.json").write_text(
            json.dumps(
                {
                    "frozen_input": {"train.csv": "a", "validation.csv": "b"},
                    "derived_input": {"train_local.npz": "c"},
                }
            )
        )
        (p / "logs/progress.json").write_text(
            json.dumps({"completed_epochs": 12, "planned_epochs": 12})
        )
        (p / "results/model.share").write_bytes(b"synthetic share")
        if party == "alice":
            (p / "results/status.json").write_text(
                json.dumps(
                    {
                        "training": "passed",
                        "probabilities_valid": True,
                        "test_read": False,
                    }
                )
            )
            for split in ("train", "validation"):
                np.save(
                    p / "results" / (split + "_predictions.npy"), np.full((3, 1), 0.5)
                )
    return ["A0_seed11", "B0_seed11"]


@pytest.mark.parametrize("party", ["alice", "bob"])
def test_valid_own_block_is_complete(tmp_path, party):
    names = fixture(tmp_path, party)
    assert h.inspect_block(tmp_path, names, party, 12, True) == {
        "complete": True,
        "reasons": [],
        "no_arrays_exported": True,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "empty_status",
        "partial_epoch",
        "invalid_probability",
        "truncated_npy",
        "missing_share",
        "test_ref",
    ],
)
def test_corrupt_or_partial_fit_is_not_reused(tmp_path, mutation):
    names = fixture(tmp_path)
    p = tmp_path / "trainings/B0_seed11"
    if mutation == "empty_status":
        (p / "results/status.json").write_text("")
    elif mutation == "partial_epoch":
        (p / "logs/progress.json").write_text(
            '{"completed_epochs":11,"planned_epochs":12}'
        )
    elif mutation == "invalid_probability":
        np.save(p / "results/train_predictions.npy", np.full((3, 1), 2))
    elif mutation == "truncated_npy":
        (p / "results/train_predictions.npy").write_bytes(b"incomplete")
    elif mutation == "missing_share":
        (p / "results/model.share").unlink()
    else:
        (p / "data/input_references.json").write_text(
            '{"frozen_input":{"train.csv":"a","test.csv":"b"},"derived_input":{}}'
        )
    result = h.inspect_block(tmp_path, names, "alice", 12, True)
    assert not result["complete"] and result["reasons"] and result["no_arrays_exported"]
    assert set(result) == {"complete", "reasons", "no_arrays_exported"}


def test_bob_cannot_have_prediction_even_with_complete_progress(tmp_path):
    names = fixture(tmp_path, "bob")
    np.save(
        tmp_path / "trainings/B0_seed11/results/train_predictions.npy",
        np.full((3, 1), 0.5),
    )
    assert not h.inspect_block(tmp_path, names, "bob", 12, True)["complete"]
