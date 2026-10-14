"""Aggregate selector refuses unknown fields, missing seed coverage and row exports."""

from pathlib import Path
import sys
import json
import copy
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import portfolio_review as p  # noqa: E402


def fixture(tmp_path):
    models = {
        k: {
            "roc_auc": dict(estimate=0.6, ci_low=0.5, ci_high=0.7, valid_replicates=500)
        }
        for k in ["A0", "B0", "L0_LR_0", "L0_tree_0"]
    }
    report = dict(
        status="completed",
        independent_confirmation=False,
        raw_full_training=False,
        full_prepared_development=True,
        model_seeds=[11, 22, 33, 44, 55],
        models=models,
        paired_comparisons={"B0_minus_A0": copy.deepcopy(models["A0"])},
        seed_points={
            str(s): {k: {"roc_auc": 0.6} for k in models} for s in [11, 22, 33, 44, 55]
        },
        pvalues={"B0_minus_A0": 0.0625},
        holm_pvalues={"B0_minus_A0": 0.0625},
        prepared_source_run="frozen",
        resampling_unit="cluster",
        primary="roc_auc",
        records={"secret_row": "FORBIDDEN_IDENTIFIER"},
    )
    cfg = {
        "datasets": {"unit": {"resampling_unit": "cluster"}},
        "shared": {
            "model_seeds": report["model_seeds"],
            "bootstrap_repeats": 500,
            "split_seed": 12,
        },
        "experiment": {
            "conditions": {"A0": {}, "B0": {}},
            "comparisons": {"B0_minus_A0": {}},
            "order_seeds": {},
            "bootstrap_seed": 13,
        },
        "prepared_source_runs": {"unit": "frozen"},
    }
    runtime = dict(
        dataset="unit", config_sha256="configuration", git_sha="git", code_sha256={}
    )
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(report))
    return path, report, cfg, runtime


def test_selector_retains_all_metrics_without_private_records(tmp_path):
    path, report, cfg, runtime = fixture(tmp_path)
    result = p.review(path, runtime, cfg)
    assert (
        "FORBIDDEN_IDENTIFIER" not in json.dumps(result)
        and "records" not in result["results"]
    )
    assert result["results"]["models"] == report["models"]
    assert list(p.numbers({"a": [True, 0.6], "b": "private"})) == [("/a/1", 0.6)]


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "array",
        "metric",
        "interval",
        "nan",
        "seed",
        "model",
        "contrast",
        "p",
        "source",
        "scope",
    ],
)
def test_selector_fails_closed(tmp_path, mutation):
    path, report, cfg, runtime = fixture(tmp_path)
    if mutation == "unknown":
        report["identities"] = ["private"]
    elif mutation == "array":
        report["seed_points"]["11"]["A0"]["roc_auc"] = [0.6]
    elif mutation == "metric":
        report["models"]["A0"]["weights"] = report["models"]["A0"].pop("roc_auc")
    elif mutation == "interval":
        report["models"]["A0"]["roc_auc"]["valid_replicates"] = 499
    elif mutation == "nan":
        report["models"]["A0"]["roc_auc"]["estimate"] = float("nan")
    elif mutation == "seed":
        report["seed_points"].pop("11")
    elif mutation == "model":
        report["models"].pop("A0")
    elif mutation == "contrast":
        report["paired_comparisons"] = {}
    elif mutation == "p":
        report["pvalues"]["B0_minus_A0"] = 1.5
    elif mutation == "source":
        report["prepared_source_run"] = "other"
    else:
        report["raw_full_training"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        p.review(path, runtime, cfg)
