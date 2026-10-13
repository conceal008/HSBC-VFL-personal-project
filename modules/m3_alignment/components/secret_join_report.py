"""Summarize reviewed engineering timing without keys, shares or counts."""

from __future__ import annotations
import numpy as np

PARTIES = ("alice", "bob")
MINIMUM_SEEDS = 5


def summarize(reports, timings, cfg):
    seeds = cfg["seeds"]
    if (
        len(set(seeds)) < MINIMUM_SEEDS
        or set(reports) != set(PARTIES)
        or set(timings) != set(PARTIES)
    ):
        raise ValueError("Incomplete independent trial inventory")
    for party in PARTIES:
        report = reports[party]
        if (
            report["status"] != "passed"
            or report["seeds"] != seeds
            or any(
                report.get(k) is not False
                for k in (
                    "intersection_reconstructed",
                    "membership_or_count_disclosed",
                    "peer_indices_disclosed",
                    "test_read",
                )
            )
        ):
            raise ValueError("Unapproved disclosure/report")
    values = np.array([timings[p]["seconds"] for p in PARTIES], dtype=float)
    if (
        values.shape != (len(PARTIES), len(seeds))
        or not np.isfinite(values).all()
        or np.any(values <= 0)
    ):
        raise ValueError("Malformed/nonpositive timing vector")
    paired = np.max(values, axis=0)
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    draws = rng.choice(
        paired, size=(cfg["bootstrap_repeats"], len(seeds)), replace=True
    ).mean(axis=1)
    low, high = np.percentile(draws, cfg["percentiles"])
    return {
        "seeds": seeds,
        "participant_seconds": {p: values[i].tolist() for i, p in enumerate(PARTIES)},
        "paired_max_seconds": paired.tolist(),
        "mean_seconds": {
            "estimate": float(paired.mean()),
            "ci_low": float(low),
            "ci_high": float(high),
        },
        "metric": cfg["metric"],
        "ci_scope": cfg["interpretation"],
        "bootstrap_seed": cfg["bootstrap_seed"],
        "bootstrap_repeats": cfg["bootstrap_repeats"],
    }
