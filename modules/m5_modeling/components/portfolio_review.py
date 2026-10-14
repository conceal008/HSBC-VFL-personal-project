"""Fail-closed public research aggregate selector; no row or model state export."""

from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path

METRICS = {
    "roc_auc",
    "pr_auc",
    "recall_at_top_k",
    "log_loss",
    "brier",
    "arm1_centered_qini",
    "arm1_auuc",
    "arm1_uplift_at_top_k",
    "arm1_policy_value",
    "arm1_policy_gain_vs_no_contact",
    "mean_centered_qini",
}
FIELDS = {
    "status",
    "scope",
    "primary",
    "model_seeds",
    "models",
    "paired_comparisons",
    "seed_points",
    "pvalues",
    "holm_pvalues",
    "pvalue_scope",
    "bootstrap",
    "resampling_unit",
    "full_data_training",
    "full_prepared_development",
    "raw_full_training",
    "task_scope",
    "prepared_source_run",
    "independent_confirmation",
}
INTERVAL_KEYS = {"estimate", "ci_low", "ci_high", "valid_replicates"}
LOCAL_MODELS = {"L0_LR_0", "L0_tree_0"}


def finite(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Nonfinite/non-scalar public metric")


def metric_table(table, interval, expected_repeats):
    if not isinstance(table, dict) or not table or not set(table) <= METRICS:
        raise ValueError("Unregistered metric name")
    for value in table.values():
        if interval:
            if not isinstance(value, dict) or set(value) != INTERVAL_KEYS:
                raise ValueError("Unexpected interval fields")
            for scalar in value.values():
                finite(scalar)
            if (
                value["ci_low"] > value["ci_high"]
                or value["valid_replicates"] != expected_repeats
            ):
                raise ValueError("Incomplete interval")
        else:
            finite(value)


def review(source, runtime, cfg):
    report = json.loads(Path(source).read_text())
    if set(report) - FIELDS - {"records"}:
        raise ValueError("Unexpected result fields; review needed")
    expected = set(cfg["experiment"]["conditions"]) | LOCAL_MODELS
    if (
        report["status"] != "completed"
        or report["independent_confirmation"]
        or report["raw_full_training"]
        or not report["full_prepared_development"]
    ):
        raise ValueError("Result scope invalid")
    if (
        report["model_seeds"] != cfg["shared"]["model_seeds"]
        or set(report["models"]) != expected
    ):
        raise ValueError("Incomplete registered models")
    if set(report["paired_comparisons"]) != set(cfg["experiment"]["comparisons"]):
        raise ValueError("Incomplete paired comparisons")
    repeats = cfg["shared"]["bootstrap_repeats"]
    for group in [report["models"], report["paired_comparisons"]]:
        for table in group.values():
            metric_table(table, True, repeats)
    if set(report["seed_points"]) != {str(s) for s in cfg["shared"]["model_seeds"]}:
        raise ValueError("Incomplete seed summary")
    for seed in report["seed_points"].values():
        if set(seed) != expected:
            raise ValueError("Incomplete per-seed models")
        for table in seed.values():
            metric_table(table, False, repeats)
    for group in [report["pvalues"], report["holm_pvalues"]]:
        if set(group) != set(cfg["experiment"]["comparisons"]):
            raise ValueError("P-value contrast differs")
        for p in group.values():
            finite(p)
            if not 0 <= p <= 1:
                raise ValueError("Invalid P value")
    dataset = runtime["dataset"]
    if (
        dataset not in cfg["datasets"]
        or report["prepared_source_run"] != cfg["prepared_source_runs"][dataset]
        or report["resampling_unit"] != cfg["datasets"][dataset]["resampling_unit"]
    ):
        raise ValueError("Source or resampling protocol differs")
    if report["primary"] not in METRICS:
        raise ValueError("Primary metric unregistered")
    return {
        "step": "S5.X2",
        "dataset": dataset,
        "classification": "reviewed_public_research_aggregate",
        "source_file": str(source),
        "source_summary_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "config_provenance": {
            "config_sha256": runtime["config_sha256"],
            "runtime_git_sha": runtime["git_sha"],
            "code_sha256": runtime["code_sha256"],
            "seed_protocol": {
                "model_seeds": cfg["shared"]["model_seeds"],
                "order_seeds": cfg["experiment"]["order_seeds"],
                "source_split_seed": cfg["shared"]["split_seed"],
                "bootstrap_seed": cfg["experiment"]["bootstrap_seed"],
            },
        },
        "omitted_fields": ["/records"],
        "results": {k: copy.deepcopy(report[k]) for k in FIELDS if k in report},
    }


def numbers(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from numbers(child, prefix + "/" + str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from numbers(child, prefix + "/" + str(index))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        finite(value)
        yield prefix, value
