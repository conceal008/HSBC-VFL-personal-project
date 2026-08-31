"""M0 · 合规折损漏斗：由客户基数逐层折损推出可用样本 N_eff 的区间。

本模块只做算术，不做判断。每一层的折损率来自配置，且必须是显式假设——
本项目没有真实业务数据，任何"看起来合理"的点估计都会被误当成事实，
因此一律以低/中/高三档给出区间，让不确定性显式化。
"""
from __future__ import annotations

import math
from typing import Dict, List

SCENARIOS = ("low", "mid", "high")


def _validate_rate(name: str, values: Dict[str, float]) -> None:
    for scenario in SCENARIOS:
        if scenario not in values:
            raise ValueError("%s 缺少 %s 档取值" % (name, scenario))
        rate = values[scenario]
        if not 0 < rate <= 1:
            raise ValueError("%s 的 %s 档折损率 %r 不在 (0, 1] 区间" % (name, scenario, rate))
    if not values["low"] <= values["mid"] <= values["high"]:
        raise ValueError("%s 的三档取值须满足 low ≤ mid ≤ high" % name)


def funnel(base_population: Dict[str, float], stages: List[Dict]) -> List[Dict]:
    """逐层折损。

    base_population 与每个 stage 的 rates 都要给 low / mid / high 三档。
    返回每层折损后的存活人数（三档），最后一行即可用样本 N_eff 的区间。
    """
    for scenario in SCENARIOS:
        if scenario not in base_population:
            raise ValueError("base_population 缺少 %s 档取值" % scenario)

    remaining = {s: float(base_population[s]) for s in SCENARIOS}
    rows: List[Dict] = [{
        "stage": "客户基数",
        "rate_low": None, "rate_mid": None, "rate_high": None,
        "remaining_low": math.floor(remaining["low"]),
        "remaining_mid": math.floor(remaining["mid"]),
        "remaining_high": math.floor(remaining["high"]),
    }]

    for stage in stages:
        name = stage["name"]
        rates = stage["rates"]
        _validate_rate(name, rates)
        for scenario in SCENARIOS:
            remaining[scenario] *= rates[scenario]
        rows.append({
            "stage": name,
            "rate_low": rates["low"], "rate_mid": rates["mid"], "rate_high": rates["high"],
            "remaining_low": math.floor(remaining["low"]),
            "remaining_mid": math.floor(remaining["mid"]),
            "remaining_high": math.floor(remaining["high"]),
        })
    return rows


def attrition_rate(rows: List[Dict], scenario: str) -> float:
    """合规折损率 = 最终可用样本 / 客户基数。"""
    if scenario not in SCENARIOS:
        raise ValueError("scenario 须为 low / mid / high 之一")
    key = "remaining_%s" % scenario
    base = rows[0][key]
    if base <= 0:
        raise ValueError("客户基数须为正")
    return rows[-1][key] / base
