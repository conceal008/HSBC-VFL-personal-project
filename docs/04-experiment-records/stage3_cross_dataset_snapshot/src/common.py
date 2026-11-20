"""
第三阶段公共统计、等budget调参代理与可追溯IO工具模块。
严格遵守 AGENTS.md §4 统计严谨性规范：
- 6 seeds ([11, 22, 33, 44, 55, 66])，切分种子与模型种子分离记录
- 代码级冻结测试集（调参仅在 val/tune 上进行，test 仅在最终评测调用一次）
- 等调参网格（所有对比臂共享同一超参候选集）
- 两级配对 bootstrap（样本/实体簇 × 种子）95% CI + 双侧 p 值 + Holm-Bonferroni 校正
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

STAGE3_ROOT = "/Applications/总文件夹/汇丰/第三阶段"
CONFIG_PATH = os.path.join(STAGE3_ROOT, "configs", "stage3_config.json")


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def fast_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Fast exact ROC-AUC via C-level scipy.stats.rankdata (exact tie handling)."""
    y = np.asarray(y_true, dtype=np.int8)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(y_score, dtype=np.float64), method="average")
    sum_pos_ranks = float(ranks[y == 1].sum())
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int8)
    order = np.argsort(-np.asarray(y_score, dtype=np.float64))
    y_sorted = y[order]
    n_pos = max(int(y_sorted.sum()), 1)
    n_neg = max(len(y_sorted) - n_pos, 1)
    tpr = np.cumsum(y_sorted) / n_pos
    fpr = np.cumsum(1 - y_sorted) / n_neg
    return float(np.max(np.abs(tpr - fpr)))


def topk_lift(y_true: np.ndarray, y_score: np.ndarray, frac: float = 0.05) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    base = y.mean()
    if base <= 0:
        return float("nan")
    k = max(1, int(round(len(y) * frac)))
    idx = np.argpartition(-np.asarray(y_score), k - 1)[:k]
    return float(y[idx].mean() / base)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.int32)
    p = np.clip(np.asarray(y_score, dtype=np.float64), 1e-7, 1 - 1e-7)
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "top5_lift": topk_lift(y, p, 0.05),
        "top10_lift": topk_lift(y, p, 0.10),
    }


def paired_bootstrap_delta(
    y_true: np.ndarray,
    preds_base: np.ndarray,
    preds_new: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = fast_roc_auc,
    n_boot: int = 1000,
    seed: int = 20261119,
    cluster_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Two-level paired bootstrap across seeds and samples (or entity clusters).
    preds_base, preds_new: shape (n_seeds, n_samples)
    """
    y = np.asarray(y_true)
    pb = np.asarray(preds_base, dtype=np.float64)
    pn = np.asarray(preds_new, dtype=np.float64)
    assert pb.shape == pn.shape and pb.ndim == 2
    n_seeds, n_samples = pb.shape

    per_seed_base = [float(metric_fn(y, pb[s])) for s in range(n_seeds)]
    per_seed_new = [float(metric_fn(y, pn[s])) for s in range(n_seeds)]
    per_seed_delta = [per_seed_new[s] - per_seed_base[s] for s in range(n_seeds)]

    mean_base = float(np.mean(per_seed_base))
    mean_new = float(np.mean(per_seed_new))
    mean_delta = float(np.mean(per_seed_delta))

    rng = np.random.default_rng(seed)

    # Pre-index clusters if provided
    if cluster_ids is not None:
        cids = np.asarray(cluster_ids)
        uniq, inv = np.unique(cids, return_inverse=True)
        order = np.argsort(inv)
        counts = np.bincount(inv)
        starts = np.zeros(len(uniq), dtype=np.int64)
        starts[1:] = np.cumsum(counts[:-1])
        cluster_slices = [order[starts[i] : starts[i] + counts[i]] for i in range(len(uniq))]
        n_clusters = len(uniq)
    else:
        cluster_slices = None
        n_clusters = 0

    # Use seed-averaged score inside bootstrap when n_samples is large?
    # Better: draw one seed index + sample/cluster resample per bootstrap draw,
    # or draw n_seeds with replacement and compute exact mean delta.
    # For accuracy + speed: draw (n_seeds,) seed indices and evaluate mean delta across drawn seeds
    # (exact 2-level cluster/sample × seed bootstrap).
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        if cluster_slices is not None:
            c_idx = rng.integers(0, n_clusters, size=n_clusters)
            idx = np.concatenate([cluster_slices[c] for c in c_idx])
        else:
            idx = rng.integers(0, n_samples, size=n_samples)
        yb = y[idx]
        if yb.min() == yb.max():
            deltas[b] = mean_delta
            continue
        s_idx = rng.integers(0, n_seeds, size=n_seeds)
        # Compute unique sampled seeds weighted by multiplicity
        uniq_s, mult = np.unique(s_idx, return_counts=True)
        d_acc = 0.0
        for s, w in zip(uniq_s, mult):
            mb = metric_fn(yb, pb[s, idx])
            mn = metric_fn(yb, pn[s, idx])
            d_acc += w * (mn - mb)
        deltas[b] = d_acc / n_seeds

    ci_low, ci_high = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    # Two-sided bootstrap p-value with finite-sample +1 floor
    p_le = (int(np.sum(deltas <= 0.0)) + 1) / (n_boot + 1)
    p_ge = (int(np.sum(deltas >= 0.0)) + 1) / (n_boot + 1)
    p_two = float(min(1.0, 2.0 * min(p_le, p_ge)))

    return {
        "base_mean": mean_base,
        "base_std": float(np.std(per_seed_base, ddof=1)) if n_seeds > 1 else 0.0,
        "new_mean": mean_new,
        "new_std": float(np.std(per_seed_new, ddof=1)) if n_seeds > 1 else 0.0,
        "delta_mean": mean_delta,
        "delta_std": float(np.std(per_seed_delta, ddof=1)) if n_seeds > 1 else 0.0,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_raw": p_two,
        "per_seed_base": per_seed_base,
        "per_seed_new": per_seed_new,
        "per_seed_delta": per_seed_delta,
        "n_boot": n_boot,
        "clustered": cluster_ids is not None,
    }


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Step-down Holm-Bonferroni adjusted p-values."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        val = min(1.0, p * (m - rank))
        running_max = max(running_max, val)
        adj[orig_idx] = running_max
    return adj


def encode_tabular_train_val_test(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    df_te: pd.DataFrame,
    cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Leakage-free numeric + frequency/ordinal encoding fitted strictly on df_tr.
    Returns float32 arrays (X_tr, X_va, X_te) and feature names.
    """
    tr_list, va_list, te_list, names = [], [], [], []
    for c in cols:
        s_tr = df_tr[c]
        if pd.api.types.is_numeric_dtype(s_tr) or pd.api.types.is_bool_dtype(s_tr):
            tr_list.append(pd.to_numeric(s_tr, errors="coerce").to_numpy(dtype=np.float32))
            va_list.append(pd.to_numeric(df_va[c], errors="coerce").to_numpy(dtype=np.float32))
            te_list.append(pd.to_numeric(df_te[c], errors="coerce").to_numpy(dtype=np.float32))
            names.append(str(c))
        else:
            s_tr_str = s_tr.astype("string").fillna("__NA__")
            s_va_str = df_va[c].astype("string").fillna("__NA__")
            s_te_str = df_te[c].astype("string").fillna("__NA__")
            vc = s_tr_str.value_counts()
            # Ordinal code ordered by frequency (unseen -> -1) + frequency encoding
            cats = {k: float(i) for i, k in enumerate(vc.index)}
            freqs = (vc / max(len(s_tr_str), 1)).to_dict()
            tr_list.append(s_tr_str.map(cats).fillna(-1.0).to_numpy(dtype=np.float32))
            va_list.append(s_va_str.map(cats).fillna(-1.0).to_numpy(dtype=np.float32))
            te_list.append(s_te_str.map(cats).fillna(-1.0).to_numpy(dtype=np.float32))
            names.append(f"{c}__ord")

            tr_list.append(s_tr_str.map(freqs).fillna(0.0).to_numpy(dtype=np.float32))
            va_list.append(s_va_str.map(freqs).fillna(0.0).to_numpy(dtype=np.float32))
            te_list.append(s_te_str.map(freqs).fillna(0.0).to_numpy(dtype=np.float32))
            names.append(f"{c}__freq")

    X_tr = np.column_stack(tr_list).astype(np.float32) if tr_list else np.zeros((len(df_tr), 0), dtype=np.float32)
    X_va = np.column_stack(va_list).astype(np.float32) if va_list else np.zeros((len(df_va), 0), dtype=np.float32)
    X_te = np.column_stack(te_list).astype(np.float32) if te_list else np.zeros((len(df_te), 0), dtype=np.float32)
    return X_tr, X_va, X_te, names


DEFAULT_TUNE_GRID = [
    {"learning_rate": 0.05, "max_depth": 4, "l2_regularization": 5.0, "max_iter": 120},
    {"learning_rate": 0.05, "max_depth": 6, "l2_regularization": 5.0, "max_iter": 120},
    {"learning_rate": 0.10, "max_depth": 4, "l2_regularization": 1.0, "max_iter": 100},
    {"learning_rate": 0.10, "max_depth": 4, "l2_regularization": 10.0, "max_iter": 120},
    {"learning_rate": 0.10, "max_depth": 6, "l2_regularization": 5.0, "max_iter": 100},
    {"learning_rate": 0.05, "max_depth": 5, "l2_regularization": 10.0, "max_iter": 150},
]


def equal_tune_and_eval_hgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    seeds: Sequence[int] = (11, 22, 33, 44, 55, 66),
    grid: Sequence[Dict[str, Any]] = DEFAULT_TUNE_GRID,
    max_bins: int = 33,
    refit_train_val: bool = False,
    tune_seed: int = 11,
) -> Dict[str, Any]:
    """
    Strictly equal-tuned HistGradientBoostingClassifier (max_bins=33 matching SGB sketch_eps=0.03).
    - Hyperparameter selection ONLY sees (X_tr, y_tr) and (X_va, y_va).
    - X_te, y_te are strictly frozen and evaluated ONLY after best params are locked.
    - Returns per-seed predictions on X_te, summary metrics, best params, and wall-clock time.
    """
    t0 = time.perf_counter()
    best_cfg = None
    best_va_auc = -1.0
    tune_log = []

    for cfg in grid:
        clf = HistGradientBoostingClassifier(
            max_bins=max_bins,
            max_features=0.8,
            early_stopping=False,
            min_samples_leaf=20,
            max_leaf_nodes=None,
            random_state=tune_seed,
            **cfg,
        )
        clf.fit(X_tr, y_tr)
        p_va = clf.predict_proba(X_va)[:, 1]
        va_auc = float(roc_auc_score(y_va, p_va))
        tune_log.append({"cfg": dict(cfg), "val_auc": va_auc})
        if va_auc > best_va_auc:
            best_va_auc = va_auc
            best_cfg = dict(cfg)

    if refit_train_val:
        X_fit = np.vstack([X_tr, X_va])
        y_fit = np.concatenate([y_tr, y_va])
    else:
        X_fit = X_tr
        y_fit = y_tr

    preds_te = np.empty((len(seeds), len(y_te)), dtype=np.float64)
    preds_va = np.empty((len(seeds), len(y_va)), dtype=np.float64)
    per_seed_metrics = []
    n_fit = len(y_fit)
    sub_n = max(100, int(round(0.85 * n_fit)))

    for i, s in enumerate(seeds):
        rng_s = np.random.default_rng(int(s) * 1009 + 17)
        row_idx = rng_s.choice(n_fit, size=sub_n, replace=False)
        clf = HistGradientBoostingClassifier(
            max_bins=max_bins,
            max_features=0.8,
            early_stopping=False,
            min_samples_leaf=20,
            max_leaf_nodes=None,
            random_state=int(s),
            **best_cfg,
        )
        clf.fit(X_fit[row_idx], y_fit[row_idx])
        p_va = clf.predict_proba(X_va)[:, 1]
        p_te = clf.predict_proba(X_te)[:, 1]
        preds_va[i] = p_va
        preds_te[i] = p_te
        m = compute_metrics(y_te, p_te)
        m["seed"] = int(s)
        per_seed_metrics.append(m)

    elapsed = time.perf_counter() - t0
    summary = {}
    for k in ["auc", "pr_auc", "ks", "brier", "top5_lift", "top10_lift"]:
        vals = [m[k] for m in per_seed_metrics]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    # Also compute Bagging (mean probability across all 6 seeds)
    bag_pred = preds_te.mean(axis=0)
    bag_metrics = compute_metrics(y_te, bag_pred)

    return {
        "best_cfg": best_cfg,
        "best_val_auc": best_va_auc,
        "max_bins": max_bins,
        "refit_train_val": refit_train_val,
        "n_features": int(X_tr.shape[1]),
        "n_train": int(len(y_fit)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_te)),
        "seeds": [int(s) for s in seeds],
        "summary": summary,
        "bagging6_metrics": bag_metrics,
        "per_seed_metrics": per_seed_metrics,
        "elapsed_sec": float(elapsed),
        "preds_te": preds_te,  # caller strips before JSON dump
        "preds_va": preds_va,
    }


def strip_arrays(obj: Any) -> Any:
    """Recursively remove large numpy prediction arrays before JSON serialization."""
    if isinstance(obj, dict):
        return {
            k: strip_arrays(v)
            for k, v in obj.items()
            if k not in ("preds_te", "preds_va")
        }
    if isinstance(obj, list):
        return [strip_arrays(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(strip_arrays(data), f, ensure_ascii=False, indent=2)
