"""Alice-local evaluation; validation selection never receives test outcomes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from sklearn.metrics import average_precision_score, roc_auc_score


def qini(y, t, score):
    propensity = float(t.mean())
    if not 0 < propensity < 1:
        raise ValueError("Both randomized arms are required")
    gain = np.r_[0.0, np.cumsum((y * (t / propensity - (1 - t) / (1 - propensity)))[
        np.argsort(-score, kind="stable")]) / len(y)]
    fraction = np.linspace(0, 1, len(gain))
    return float(trapezoid(gain - fraction * gain[-1], fraction))


def policy_metrics(y, t, score, fraction):
    selected = np.zeros(len(y), dtype=bool)
    selected[np.argsort(-score, kind="stable")[:max(1, int(len(y) * fraction))]] = True
    propensity = float(t.mean())
    policy_value = np.mean(y * (selected * t / propensity
                                + (~selected) * (1 - t) / (1 - propensity)))
    control_value = np.mean(y * (1 - t) / (1 - propensity))
    groups = [y[selected & (t == arm)] for arm in (0, 1)]
    uplift = float(groups[1].mean() - groups[0].mean()) if all(map(len, groups)) else float("nan")
    return {"centered_qini": qini(y, t, score), "uplift_at_top_k": uplift,
            "policy_value": float(policy_value), "policy_gain_vs_no_contact": float(policy_value - control_value)}


def values(y, predictions, spec, shared, treatment=None):
    if not spec["treatment"]:
        score = predictions[:, 0]
        idx = np.argsort(-score, kind="stable")[:max(1, int(len(y) * shared["top_k_fraction"]))]
        return {"roc_auc": float(roc_auc_score(y, score)),
                "pr_auc": float(average_precision_score(y, score)),
                "recall_at_top_k": float(y[idx].sum() / max(1, y.sum()))}
    result = {}
    for arm in spec["treatment_arms"]:
        mask = np.isin(treatment, [0, arm])
        metrics = policy_metrics(y[mask], (treatment[mask] == arm).astype(int),
                                 (predictions[:, arm] - predictions[:, 0])[mask], shared["top_k_fraction"])
        result.update({f"arm{arm}_{name}": value for name, value in metrics.items()})
    return result


def validation_score(y, predictions, spec, shared, treatment=None):
    result = values(y, predictions, spec, shared, treatment)
    if not spec["treatment"]:
        return result["roc_auc"]
    return float(np.mean([result[f"arm{arm}_centered_qini"] for arm in spec["treatment_arms"]]))


def select_candidate(validation_scores):
    if not validation_scores or not np.isfinite(validation_scores).all():
        raise ValueError("Invalid validation scores")
    return int(np.argmax(validation_scores))


def bootstrap_metrics(y, predictions, spec, shared, seed, treatment=None, comparator=None):
    point = values(y, predictions, spec, shared, treatment)
    if comparator is not None:
        baseline = values(y, comparator, spec, shared, treatment)
        point = {key: value - baseline[key] for key, value in point.items()}
    draws: dict[str, list] = {key: [] for key in point}
    rng = np.random.default_rng(seed)
    for _ in range(shared["bootstrap_repeats"]):
        if treatment is None:
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) != 2:
                continue
            ts = None
        else:
            groups = [np.flatnonzero(treatment == arm) for arm in np.unique(treatment)]
            idx = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
            # Group concatenation must not make equal-score ranking depend on treatment.
            rng.shuffle(idx)
            ts = treatment[idx]
        current = values(y[idx], predictions[idx], spec, shared, ts)
        if comparator is not None:
            base = values(y[idx], comparator[idx], spec, shared, ts)
            current = {key: value - base[key] for key, value in current.items()}
        for key, value in current.items():
            if np.isfinite(value):
                draws[key].append(value)
    result = {}
    for key, value in point.items():
        if not draws[key]:
            raise ValueError("No valid bootstrap replicates")
        lo, hi = np.percentile(draws[key], shared["ci_percentiles"])
        result[key] = {"estimate": value, "ci_low": float(lo), "ci_high": float(hi),
                       "valid_replicates": len(draws[key])}
    return result


def membership_diagnostic(train_y, train_prob, test_y, test_prob, shared, seed):
    """A diagnostic of risk IF scores/labels were exposed to a black-box attacker."""
    rng = np.random.default_rng(seed)
    size = min(len(train_y), len(test_y), shared["membership_max_rows"])
    train_idx = rng.choice(len(train_y), size, replace=False)
    test_idx = rng.choice(len(test_y), size, replace=False)
    epsilon = shared["score_epsilon"]
    def confidence(labels, probs):
        p = np.clip(probs, epsilon, 1 - epsilon)
        return labels * np.log(p) + (1 - labels) * np.log(1 - p)
    score = np.r_[confidence(train_y[train_idx], train_prob[train_idx]),
                  confidence(test_y[test_idx], test_prob[test_idx])]
    membership = np.r_[np.ones(size), np.zeros(size)]
    auc = roc_auc_score(membership, score)
    draws = []
    for _ in range(shared["bootstrap_repeats"]):
        idx = np.r_[rng.choice(size, size, replace=True), size + rng.choice(size, size, replace=True)]
        draws.append(roc_auc_score(membership[idx], score[idx]))
    lo, hi = np.percentile(draws, shared["ci_percentiles"])
    return {"loss_attack_auc": float(auc), "ci_low": float(lo), "ci_high": float(hi),
            "scope": "Alice-private hypothetical score-and-label release; not proof against all attacks"}


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
