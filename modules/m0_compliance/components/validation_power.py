"""S0.1P · 离线排序比较的正例下界、小人群可检出效应与市场正例上限。

S0.1 的 N_eff ≥ 30,000 回答的是「随机对照增益检验要多少人」。本模块补三个
S0.1 没有回答、但恢复实验前必须先算清的量：

1. 离线比较两个模型的 ROC曲线下面积（AUROC）时，**决定精度的是正例数**，
   不是客户数。给定待检出的配对 AUROC 差，最少需要多少个正例。
2. 可用人群只有几千人时，双臂随机对照**最小能检出多大的效应**。
3. 用公开的跨境理财通新增投资者统计，给每个观察窗内的正例数**设上限**。

近似方法：
- 单个 AUROC 的标准误用 Hanley & McNeil（1982）近似；
- 两个模型在同一批测试客户上的 AUROC 估计是正相关的，配对差的标准误
  = 单个标准误 × sqrt(2 × (1 − 相关系数))，两个模型的 AUROC 取同值；
- 双臂检验沿用 S0.1 的 `required_n_per_arm`，口径完全一致。

这些都是**近似公式**，用于定量级与排优先级，不替代实验后基于真实分数的
配对 bootstrap 区间。
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

from scipy.stats import norm

from modules.m0_compliance.components.sample_size import required_n_per_arm

HANLEY_Q2_FACTOR = 2.0          # Hanley–McNeil：Q2 = 2A² / (1 + A)
MAX_POSITIVES = 100_000_000     # 二分搜索上界；超过即视为「不可检出」
MAX_LIFT_STEPS = 200            # 二分搜索迭代次数，足以收敛到 1e-12 量级
LIFT_EDGE = 1e-9                # 相对增益上界的收缩量，避免处理臂转化率恰为 1
MIN_EXPECTED_EVENTS = 5         # 对照臂期望转化数低于此值时，两比例检验的正态近似不可信
DAYS_PER_YEAR = 365
MONTHS_PER_YEAR = 12


def _z_sum(alpha: float, power: float) -> float:
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha 与 power 须在 (0, 1) 区间")
    return float(norm.ppf(1 - alpha / 2) + norm.ppf(power))


def auc_standard_error(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley–McNeil 近似：单个 AUROC 估计的标准误。"""
    if not 0 < auc < 1:
        raise ValueError("auc 须在 (0, 1) 区间")
    if n_pos < 1 or n_neg < 1:
        raise ValueError("正例数与负例数都须 ≥1")
    q1 = auc / (2 - auc)
    q2 = HANLEY_Q2_FACTOR * auc ** 2 / (1 + auc)
    variance = (auc * (1 - auc)
                + (n_pos - 1) * (q1 - auc ** 2)
                + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return math.sqrt(variance)


def negatives_for(n_pos: int, base_rate: float) -> int:
    """给定正例数与正例率，对应的负例数（四舍五入，至少 1）。"""
    if not 0 < base_rate < 1:
        raise ValueError("base_rate 须在 (0, 1) 区间")
    return max(1, round(n_pos * (1 - base_rate) / base_rate))


def paired_auc_mde(auc: float, n_pos: int, base_rate: float,
                   estimate_correlation: float, alpha: float, power: float) -> float:
    """给定正例数时，配对 AUROC 差的最小可察差异（绝对值）。

    estimate_correlation  两个模型 AUROC 估计之间的相关系数（同一批测试客户）
    """
    if not 0 <= estimate_correlation < 1:
        raise ValueError("estimate_correlation 须在 [0, 1) 区间")
    se = auc_standard_error(auc, n_pos, negatives_for(n_pos, base_rate))
    se_diff = se * math.sqrt(2 * (1 - estimate_correlation))
    return _z_sum(alpha, power) * se_diff


def positives_for_paired_auc_mde(auc: float, mde: float, base_rate: float,
                                 estimate_correlation: float,
                                 alpha: float, power: float) -> int:
    """检出配对 AUROC 差 ≥ mde 所需的最少正例数（二分搜索，MDE 随正例数单调递减）。"""
    if mde <= 0:
        raise ValueError("mde 须为正")

    def ok(n: int) -> bool:
        return paired_auc_mde(auc, n, base_rate, estimate_correlation, alpha, power) <= mde

    if not ok(MAX_POSITIVES):
        raise ValueError("在搜索上界内无法检出该差异")
    lo, hi = 1, MAX_POSITIVES
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def positives_table(auc: float, mdes: Iterable[float], base_rates: Iterable[float],
                    correlations: Iterable[float], alpha: float,
                    power: float) -> List[Dict[str, float]]:
    """(待检出配对差 × 正例率 × 估计相关系数) 网格上的正例下界。"""
    rows: List[Dict[str, float]] = []
    for base_rate in base_rates:
        for corr in correlations:
            for mde in mdes:
                n_pos = positives_for_paired_auc_mde(auc, mde, base_rate, corr, alpha, power)
                rows.append({
                    "auc": auc,
                    "paired_auc_mde": mde,
                    "base_rate": base_rate,
                    "estimate_correlation": corr,
                    "min_positives": n_pos,
                    "implied_customers": math.ceil(n_pos / base_rate),
                })
    return rows


def detectable_relative_lift(base_rate: float, n_total: int, alpha: float,
                             power: float, holdout_share: float) -> Optional[float]:
    """双臂随机对照、总人群 n_total 时可检出的最小相对增益。

    与 S0.1 同一检验口径（`required_n_per_arm`）。即使处理臂转化率接近 1
    也检不出时返回 None。
    """
    if not 0 < holdout_share < 1:
        raise ValueError("holdout_share 须在 (0, 1) 区间")
    arm = math.floor(n_total * min(holdout_share, 1 - holdout_share))
    if arm < 1:
        return None
    max_lift = (1 - base_rate) / base_rate          # 处理臂转化率不得达到 1

    def ok(lift: float) -> bool:
        return required_n_per_arm(base_rate, lift, alpha, power) <= arm

    hi = max_lift * (1 - LIFT_EDGE)
    if not ok(hi):
        return None
    lo = 0.0
    for _ in range(MAX_LIFT_STEPS):
        mid = (lo + hi) / 2
        if mid <= 0:
            break
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def small_population_table(base_rate: float, populations: Dict[str, int], alpha: float,
                           power: float, holdout_share: float) -> List[Dict[str, object]]:
    """各人群规模档位下双臂随机对照的可检出效应。"""
    rows: List[Dict[str, object]] = []
    smaller_share = min(holdout_share, 1 - holdout_share)
    for label, n_total in populations.items():
        lift = detectable_relative_lift(base_rate, n_total, alpha, power, holdout_share)
        expected_control_events = math.floor(n_total * smaller_share) * base_rate
        rows.append({
            "scenario": label,
            "n_total": n_total,
            "base_rate": base_rate,
            "expected_control_events": expected_control_events,
            "normal_approx_valid": expected_control_events >= MIN_EXPECTED_EVENTS,
            "min_relative_lift": lift,
            "min_absolute_lift": None if lift is None else base_rate * lift,
        })
    return rows


def market_positive_ceiling(new_investors: int, months_observed: int, window_days: int,
                            institution_shares: Iterable[float]) -> List[Dict[str, float]]:
    """由全市场新增投资者数推出一个观察窗内某机构正例数的上限。

    这是**上限**而不是估计：新增投资者是「首次参与」，不等于「首次成功申购」；
    机构份额是情景假设，不是事实。
    """
    if new_investors < 0 or months_observed < 1 or window_days < 1:
        raise ValueError("参数须为正")
    annual = new_investors / months_observed * MONTHS_PER_YEAR
    per_window = annual * window_days / DAYS_PER_YEAR
    rows: List[Dict[str, float]] = []
    for share in institution_shares:
        if not 0 < share <= 1:
            raise ValueError("机构份额须在 (0, 1] 区间")
        rows.append({
            "institution_share": share,
            "market_new_per_year": annual,
            "market_new_per_window": per_window,
            "institution_positive_ceiling_per_window": per_window * share,
        })
    return rows
