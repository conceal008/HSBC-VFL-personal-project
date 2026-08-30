"""S0.1 · 由统计可检验性反推业务最小可行规模。

本模块只做一件事：给定基线转化率与待检出的相对增益，算出双臂随机对照试验
每臂所需样本量。它回答的是「多小的人群规模会让整个方案无法被验证」，
而不是「多大的人群规模在业务上划算」——后者需要需求方输入。
"""
from __future__ import annotations

import math
from typing import Iterable, List, Dict

from scipy.stats import norm

RATE_ROUND_DIGITS = 6


def required_n_per_arm(base_rate: float, relative_lift: float,
                       alpha: float, power: float) -> int:
    """双侧两比例检验，每臂所需样本量（向上取整）。

    base_rate      对照臂转化率
    relative_lift  待检出的相对提升，处理臂转化率 = base_rate * (1 + relative_lift)
    alpha          双侧显著性水平
    power          检验功效
    """
    if not 0 < base_rate < 1:
        raise ValueError("base_rate 须在 (0, 1) 区间")
    if relative_lift <= 0:
        raise ValueError("relative_lift 须为正")

    treated_rate = base_rate * (1 + relative_lift)
    if treated_rate >= 1:
        raise ValueError("relative_lift 过大，处理臂转化率超出 (0, 1)")

    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    variance = base_rate * (1 - base_rate) + treated_rate * (1 - treated_rate)
    delta = treated_rate - base_rate
    return math.ceil((z_alpha + z_power) ** 2 * variance / delta ** 2)


def sample_size_table(base_rates: Iterable[float], relative_lifts: Iterable[float],
                      alpha: float, power: float,
                      holdout_share: float) -> List[Dict[str, float]]:
    """返回 (基线转化率 × 相对增益) 网格上的样本量需求。

    holdout_share 为对照臂占比；总可用人群 = 每臂样本量 / min(holdout_share, 1 - holdout_share)。
    """
    if not 0 < holdout_share < 1:
        raise ValueError("holdout_share 须在 (0, 1) 区间")

    smaller_arm_share = min(holdout_share, 1 - holdout_share)
    rows: List[Dict[str, float]] = []
    for base_rate in base_rates:
        for relative_lift in relative_lifts:
            per_arm = required_n_per_arm(base_rate, relative_lift, alpha, power)
            rows.append({
                "base_rate": base_rate,
                "relative_lift": relative_lift,
                "treated_rate": round(base_rate * (1 + relative_lift), RATE_ROUND_DIGITS),
                "n_per_arm": per_arm,
                "n_total": math.ceil(per_arm / smaller_arm_share),
            })
    return rows
