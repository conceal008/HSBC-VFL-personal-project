"""M6 · S6.P2 盈亏平衡与信息价值模型（框架 S6.8 的前置；约束与步骤复核 2026-10-25 第三节）。

把「值不值得做」写成三个选项的净值比较：不做、L1（本地模型 + 对方 k-匿名聚合统计）、L3（双向联邦）。
所有输入都是**显式假设的概率分布**，不含任何业务数据，也不把公开数据集上的命中率差换算成收益。
模型回答三件事：

1. 当前不确定性下，三个选项各自胜出的概率与期望净值；
2. L3 净值为 0 时所需的最低命中提升（盈亏平衡点）；
3. 每个参数的部分完全信息期望价值（EVPPI）：先把这个参数问清楚，决策的期望净值能提高多少。
   它给出需求方问题单与后续实验的优先级。

净值（项目期内，不贴现）：
    收益 = 有效年数 × 年联系人数 × 命中提升 × 因果占比 × 单客价值
    有效年数 = 项目期 − 上线前延迟
    成本 = 建设 + 合规 + 项目期 × 年运维
L3 的命中提升是相对本地单边模型的绝对值；L1 的命中提升 = L3 的命中提升 × L1 捕获比例。

L3 的出境机制按 S0.6R（modules/m0_compliance/components/export_mechanism.py）：
- 关键信息基础设施运营者 → 数据出境安全评估（第七条第（一）项）；
- 否则适用大湾区标准合同 → 大湾区标准合同（免数量门槛）；
- 否则训练人群（梯度，敏感个人信息）不满 1 万人 → 标准合同或认证（第八条）；
- 其余 → 数据出境安全评估（第七条第（二）项）。
⚠️ 技术侧自评，未经法务复核（W-001）。
安全评估可能不获通过：未通过时 L3 没有收益，损失合规成本与评估前已投入的准备成本
（建设成本 × 准备占比），不发生年运维。个人信息出境标准合同（含大湾区标准合同）不计不获通过的风险。
L1 只交换 k-匿名聚合统计，按不构成个人信息出境处理，只计内部合规评审成本；该前提由法务确认。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

MONTHS_PER_YEAR = 12
OPTIONS = ("不做", "L1", "L3")
HEAVY_LABEL = "数据出境安全评估"
CONTRACT_LABEL = "个人信息出境标准合同（含大湾区）"
DIST_KEYS = {"triangular": ("low", "mode", "high"), "uniform": ("low", "high"),
             "loguniform": ("low", "high"), "bernoulli": ("p",), "constant": ("value",)}
REQUIRED_PARAMETERS = (
    "horizon_years", "contacts_per_year", "delta_hit_l3", "causal_share", "value_per_customer",
    "l1_capture_ratio", "is_ciio", "gba_eligible", "training_cohort_small",
    "assessment_cost", "assessment_delay_months", "contract_cost", "contract_delay_months",
    "l3_build_cost", "l3_annual_ops", "l1_build_cost", "l1_annual_ops", "l1_compliance_cost",
    "l1_delay_months", "assessment_approved", "l3_prep_share",
)


def sample_parameter(spec: Dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """按配置抽样一个参数。支持 triangular / uniform / loguniform / bernoulli / constant。"""
    kind = spec.get("dist")
    if kind not in DIST_KEYS:
        raise ValueError("未知分布：%s" % kind)
    missing = [k for k in DIST_KEYS[kind] if k not in spec]
    if missing:
        raise ValueError("分布 %s 缺参数：%s" % (kind, "、".join(missing)))
    if n < 1:
        raise ValueError("抽样数须 ≥ 1")
    if kind == "triangular":
        return rng.triangular(spec["low"], spec["mode"], spec["high"], size=n)
    if kind == "uniform":
        return rng.uniform(spec["low"], spec["high"], size=n)
    if kind == "loguniform":
        if spec["low"] <= 0 or spec["high"] <= spec["low"]:
            raise ValueError("loguniform 须 0 < low < high")
        return np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"]), size=n))
    if kind == "bernoulli":
        if not 0 <= spec["p"] <= 1:
            raise ValueError("bernoulli 的 p 须在 [0, 1]")
        return (rng.random(n) < spec["p"]).astype(float)
    return np.full(n, float(spec["value"]))


def sample_parameters(specs: Dict[str, Dict], n: int,
                      rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """按配置顺序逐个抽样，参数之间相互独立。"""
    missing = [k for k in REQUIRED_PARAMETERS if k not in specs]
    if missing:
        raise ValueError("缺少参数：%s" % "、".join(missing))
    return {name: sample_parameter(spec, n, rng) for name, spec in specs.items()}


def point_parameters(specs: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    """每个参数取分布的中心值（三角取众数，均匀取中点，对数均匀取几何中点，伯努利取 p），用于确定性表。"""
    out = {}
    for name, spec in specs.items():
        kind = spec["dist"]
        if kind == "triangular":
            v = spec["mode"]
        elif kind == "uniform":
            v = (spec["low"] + spec["high"]) / 2
        elif kind == "loguniform":
            v = float(np.sqrt(spec["low"] * spec["high"]))
        elif kind == "bernoulli":
            v = spec["p"]
        else:
            v = spec["value"]
        out[name] = np.array([float(v)])
    return out


def l3_heavy_path(p: Dict[str, np.ndarray]) -> np.ndarray:
    """L3 是否须走数据出境安全评估（S0.6R 规则）。"""
    ciio = p["is_ciio"] > 0
    gba = p["gba_eligible"] > 0
    small = p["training_cohort_small"] > 0
    return ciio | (~gba & ~small)


def l3_compliance(p: Dict[str, np.ndarray]):
    """L3 的一次性合规成本与上线前延迟（月）。"""
    heavy = l3_heavy_path(p)
    cost = np.where(heavy, p["assessment_cost"], p["contract_cost"])
    delay = np.where(heavy, p["assessment_delay_months"], p["contract_delay_months"])
    return cost, delay


def _effective_years(horizon: np.ndarray, delay_months: np.ndarray) -> np.ndarray:
    return np.clip(horizon - delay_months / MONTHS_PER_YEAR, 0, None)


def npv_options(p: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """三个选项在每组参数抽样下的项目期净值。「不做」恒为 0（以现状为基准）。"""
    horizon = p["horizon_years"]
    per_year_l3 = (p["contacts_per_year"] * p["delta_hit_l3"] * p["causal_share"]
                   * p["value_per_customer"])
    comp_cost, comp_delay = l3_compliance(p)
    live = (_effective_years(horizon, comp_delay) * per_year_l3
            - p["l3_build_cost"] - comp_cost - horizon * p["l3_annual_ops"])
    blocked = l3_heavy_path(p) & (p["assessment_approved"] == 0)
    npv_l3 = np.where(blocked, -(p["l3_prep_share"] * p["l3_build_cost"] + comp_cost), live)
    npv_l1 = (_effective_years(horizon, p["l1_delay_months"]) * per_year_l3 * p["l1_capture_ratio"]
              - p["l1_build_cost"] - p["l1_compliance_cost"] - horizon * p["l1_annual_ops"])
    return {"不做": np.zeros_like(npv_l3), "L1": npv_l1, "L3": npv_l3}


def _stack(npvs: Dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([npvs[o] for o in OPTIONS])


def decision_summary(npvs: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """每个选项的期望净值、净值 > 0 的概率、胜出概率。"""
    stack = _stack(npvs)
    best = stack.argmax(axis=0)
    return [{"option": o,
             "expected_npv": float(npvs[o].mean()),
             "median_npv": float(np.median(npvs[o])),
             "prob_npv_positive": float((npvs[o] > 0).mean()),
             "prob_best": float((best == i).mean())}
            for i, o in enumerate(OPTIONS)]


def evpi(npvs: Dict[str, np.ndarray]) -> float:
    """完全信息期望价值：E[max_a NPV_a] − max_a E[NPV_a]。"""
    stack = _stack(npvs)
    return float(stack.max(axis=0).mean() - stack.mean(axis=1).max())


def _bin_labels(values: np.ndarray, n_bins: int) -> np.ndarray:
    unique = np.unique(values)
    if len(unique) <= n_bins:
        return np.searchsorted(unique, values)
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, values, side="right")


def evppi(npvs: Dict[str, np.ndarray], values: np.ndarray, n_bins: int) -> float:
    """单个参数的部分完全信息期望价值（按分位数分箱的非参数估计）。

    E_θ[max_a E(NPV_a | θ)] − max_a E[NPV_a]。取值不多于 n_bins 个的参数（伯努利）按取值分组。
    分箱估计在有限样本下偏正：与模型无关的参数也会得到小的正值，须用无关参数作噪声基线对照。
    """
    if n_bins < 2:
        raise ValueError("n_bins 须 ≥ 2")
    stack = _stack(npvs)
    values = np.asarray(values)
    if values.shape[0] != stack.shape[1]:
        raise ValueError("参数抽样数与净值抽样数不一致")
    labels = _bin_labels(values, n_bins)
    total = 0.0
    for b in np.unique(labels):
        mask = labels == b
        total += mask.mean() * stack[:, mask].mean(axis=1).max()
    return float(total - stack.mean(axis=1).max())


def evppi_table(npvs: Dict[str, np.ndarray], params: Dict[str, np.ndarray],
                names: Iterable[str], n_bins: int) -> List[Dict[str, Any]]:
    """按 EVPPI 从高到低排列的参数表。"""
    rows = [{"parameter": n, "evppi": evppi(npvs, params[n], n_bins)} for n in names]
    return sorted(rows, key=lambda r: r["evppi"], reverse=True)


def breakeven_delta_hit(p: Dict[str, np.ndarray]) -> np.ndarray:
    """L3 上线（合规获通过）时净值恰为 0 所需的命中提升（相对本地单边模型，绝对值）。

    延迟吃掉整个项目期时为 inf。不计安全评估不获通过的情形，那部分风险体现在 npv_options 里。
    """
    comp_cost, comp_delay = l3_compliance(p)
    years = _effective_years(p["horizon_years"], comp_delay)
    cost = p["l3_build_cost"] + comp_cost + p["horizon_years"] * p["l3_annual_ops"]
    unit = years * p["contacts_per_year"] * p["causal_share"] * p["value_per_customer"]
    out = np.full(np.shape(unit), np.inf)
    np.divide(cost, unit, out=out, where=unit > 0)
    return out


def breakeven_grid(specs: Dict[str, Dict], contacts: Iterable[float], values: Iterable[float],
                   heavy: bool) -> List[Dict[str, Any]]:
    """其余参数取中心值时，按年联系人数 × 单客价值列出 L3 的盈亏平衡命中提升。

    heavy=True 表示须申报数据出境安全评估，False 表示订立个人信息出境标准合同（含大湾区标准合同）。
    """
    rows = []
    for c in contacts:
        for v in values:
            p = point_parameters(specs)
            p["contacts_per_year"] = np.array([float(c)])
            p["value_per_customer"] = np.array([float(v)])
            p["is_ciio"] = np.array([1.0 if heavy else 0.0])
            p["gba_eligible"] = np.array([0.0])
            p["training_cohort_small"] = np.array([0.0 if heavy else 1.0])
            rows.append({"contacts_per_year": float(c), "value_per_customer": float(v),
                         "compliance_path": HEAVY_LABEL if heavy else CONTRACT_LABEL,
                         "breakeven_delta_hit": float(breakeven_delta_hit(p)[0])})
    return rows
