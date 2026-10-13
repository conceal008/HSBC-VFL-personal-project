"""Pairing and disclosure negative controls on synthetic timing."""

from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import secret_join_report as report  # noqa: E402


def fixture():
    cfg = {
        "seeds": [101, 102, 103, 104, 105],
        "bootstrap_seed": 1,
        "bootstrap_repeats": 30,
        "percentiles": [2.5, 97.5],
        "metric": "paired max",
        "interpretation": "conditional",
    }
    reports = {
        p: {
            "status": "passed",
            "seeds": cfg["seeds"],
            "intersection_reconstructed": False,
            "membership_or_count_disclosed": False,
            "peer_indices_disclosed": False,
            "test_read": False,
        }
        for p in report.PARTIES
    }
    timings = {p: {"seconds": [1 if p == "alice" else 2] * 5} for p in report.PARTIES}
    return reports, timings, cfg


def test_pair_max_not_two_independent_samples():
    reports, timings, cfg = fixture()
    out = report.summarize(reports, timings, cfg)
    assert out["paired_max_seconds"] == [2] * 5
    assert out["mean_seconds"] == {"estimate": 2.0, "ci_low": 2.0, "ci_high": 2.0}


@pytest.mark.parametrize(
    "failure",
    ["seed", "party", "status", "disclosure", "vector", "nonfinite", "negative"],
)
def test_invalid_reports_cannot_be_published(failure):
    reports, timings, cfg = fixture()
    if failure == "seed":
        cfg["seeds"] = [101]
    elif failure == "party":
        reports.pop("bob")
    elif failure == "status":
        reports["bob"]["status"] = "failed"
    elif failure == "disclosure":
        reports["bob"]["intersection_reconstructed"] = True
    elif failure == "vector":
        timings["bob"]["seconds"] = [2]
    elif failure == "nonfinite":
        timings["bob"]["seconds"][0] = float("nan")
    else:
        timings["bob"]["seconds"][0] = -1
    with pytest.raises(ValueError):
        report.summarize(reports, timings, cfg)
