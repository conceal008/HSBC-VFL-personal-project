"""M6 · 四层指标：判别力 → 业务排序 → 增量 → 决策等价性。

框架的核心主张：**营销的正确目标是增量转化（uplift），不是响应率。**
响应率高的人可能本来就会买——把他们识别得更准不产生任何新增收益。
因此 AUC 在本项目里降为**诊断指标**，增益才是最终判据。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

BOOTSTRAP_DEFAULT = 1000
CI_LOW = 2.5
CI_HIGH = 97.5
PERCENT = 100.0


# ————————————————— 第一层：判别力 —————————————————

def auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def ks(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score)
    y_sorted = y[order]
    pos = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    neg = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(pos - neg)))


# ————————————————— 第二层：业务排序 —————————————————

def lift_at_k(y: np.ndarray, score: np.ndarray, k_frac: float) -> float:
    """Top-K 人群的响应率相对整体的倍数。营销只触达 Top-K，这才是业务口径。"""
    n_top = max(int(len(y) * k_frac), 1)
    top = np.argsort(-score)[:n_top]
    base = y.mean()
    return float(y[top].mean() / base) if base > 0 else float("nan")


def recall_at_k(y: np.ndarray, score: np.ndarray, k_frac: float) -> float:
    n_top = max(int(len(y) * k_frac), 1)
    top = np.argsort(-score)[:n_top]
    return float(y[top].sum() / max(y.sum(), 1))


# ————————————————— 第三层：增量（最终判据）—————————————————

def qini_curve(y: np.ndarray, treatment: np.ndarray,
               uplift_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Qini 曲线：按预测增益降序累积「处理组转化数 − 对照组转化数×比例修正」。

    曲线**含原点 (0, 0)**——一个人都不触达时增益必然为 0。
    缺了原点会让 AUUC 的积分从 x=1/n 起算，与随机基准的对角线不可比。
    """
    order = np.argsort(-uplift_score)
    y_o, t_o = y[order], treatment[order]
    n_t = np.cumsum(t_o)
    n_c = np.cumsum(1 - t_o)
    r_t = np.cumsum(y_o * t_o)
    r_c = np.cumsum(y_o * (1 - t_o))
    with np.errstate(divide="ignore", invalid="ignore"):
        qini = r_t - np.where(n_c > 0, r_c * n_t / np.maximum(n_c, 1), 0.0)
    frac = np.concatenate([[0.0], np.arange(1, len(y) + 1) / len(y)])
    return frac, np.concatenate([[0.0], qini])


def auuc(y: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray) -> float:
    """Qini 曲线下面积，按人数归一化。越大越好；随机排序时接近 0。"""
    frac, q = qini_curve(y, treatment, uplift_score)
    return float(np.trapezoid(q, frac) / max(len(y), 1))


def uplift_at_k(y: np.ndarray, treatment: np.ndarray,
                uplift_score: np.ndarray, k_frac: float) -> float:
    """Top-K 人群中处理组与对照组的转化率之差——即「因营销而改变行为」的比例。"""
    n_top = max(int(len(y) * k_frac), 1)
    top = np.argsort(-uplift_score)[:n_top]
    y_t, t_t = y[top], treatment[top]
    if t_t.sum() == 0 or (1 - t_t).sum() == 0:
        return float("nan")
    return float(y_t[t_t == 1].mean() - y_t[t_t == 0].mean())


# ————————————————— 第四层：决策等价性 —————————————————

def top_k_overlap(score_a: np.ndarray, score_b: np.ndarray, k_frac: float) -> float:
    """两套方案 Top-K 名单的重合度。

    框架的特色检验：**若重合度极高，AUC 提升 0.01 在业务上等于零**——
    两套方案推给的是几乎同一批人。这比任何统计量都更能说服业务方。
    """
    n_top = max(int(len(score_a) * k_frac), 1)
    set_a = set(np.argsort(-score_a)[:n_top].tolist())
    set_b = set(np.argsort(-score_b)[:n_top].tolist())
    return len(set_a & set_b) / n_top


# ————————————————— 置信区间 —————————————————

def bootstrap_auc_gap(y: np.ndarray, score_hi: np.ndarray, score_lo: np.ndarray,
                      n_boot: int, seed: int) -> Dict[str, float]:
    """AUC 差值的 bootstrap 置信区间。

    统计严谨性是不降级的红线：**任何「提升」都必须给 CI，且 CI 不含 0 才算数。**
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    gaps = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            gaps[i] = np.nan
            continue
        gaps[i] = roc_auc_score(y[idx], score_hi[idx]) - roc_auc_score(y[idx], score_lo[idx])
    gaps = gaps[~np.isnan(gaps)]
    lo, hi = np.percentile(gaps, [CI_LOW, CI_HIGH])
    return {"gap_mean": float(gaps.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "significant": bool(lo > 0 or hi < 0)}
