"""M6 · S6.P2 盈亏平衡与信息价值模型的单元测试。只检验实现忠实于定义；参数取值的含义由 notebook 说明。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m6_evaluation.components import breakeven as BE  # noqa: E402

CONFIG = ROOT / "modules/m6_evaluation/configs/s6_p2_breakeven.yaml"
SEED = 11
N = 20000
N_BINS = 20
TOL = 1e-6
REL_TOL = 0.02


def _specs():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["parameters"]


def _sample(n=N):
    return BE.sample_parameters(_specs(), n, np.random.default_rng(SEED))


def _fixed(**over):
    p = BE.point_parameters(_specs())
    for k, v in over.items():
        p[k] = np.array([float(v)])
    return p


def test_各分布抽样范围正确():
    rng = np.random.default_rng(SEED)
    lu = BE.sample_parameter({"dist": "loguniform", "low": 10, "high": 1000}, N, rng)
    assert lu.min() >= 10 and lu.max() <= 1000
    assert abs(np.median(np.log10(lu)) - 2) < 0.05          # 对数中点
    tri = BE.sample_parameter({"dist": "triangular", "low": 0, "mode": 1, "high": 2}, N, rng)
    assert tri.min() >= 0 and tri.max() <= 2
    b = BE.sample_parameter({"dist": "bernoulli", "p": 0.3}, N, rng)
    assert set(np.unique(b)) <= {0.0, 1.0} and abs(b.mean() - 0.3) < REL_TOL
    c = BE.sample_parameter({"dist": "constant", "value": 3}, 5, rng)
    assert np.all(c == 3)


def test_非法分布被拒绝():
    rng = np.random.default_rng(SEED)
    with pytest.raises(ValueError):
        BE.sample_parameter({"dist": "normal", "low": 0, "high": 1}, 10, rng)
    with pytest.raises(ValueError):
        BE.sample_parameter({"dist": "uniform", "low": 0}, 10, rng)
    with pytest.raises(ValueError):
        BE.sample_parameter({"dist": "loguniform", "low": 0, "high": 1}, 10, rng)
    with pytest.raises(ValueError):
        BE.sample_parameter({"dist": "bernoulli", "p": 1.5}, 10, rng)
    with pytest.raises(ValueError):
        BE.sample_parameters({"horizon_years": {"dist": "constant", "value": 3}}, 10, rng)


def test_合规路径按S0_6R规则():
    cases = [  # (关基, 大湾区, 训练人群不满1万) → 是否须申报数据出境安全评估
        ((1, 1, 1), True), ((1, 0, 0), True),
        ((0, 1, 0), False), ((0, 0, 1), False), ((0, 0, 0), True),
    ]
    for (ciio, gba, small), heavy in cases:
        p = _fixed(is_ciio=ciio, gba_eligible=gba, training_cohort_small=small)
        assert bool(BE.l3_heavy_path(p)[0]) is heavy


def test_净值公式手算核对():
    p = _fixed(is_ciio=0, gba_eligible=1, training_cohort_small=0, assessment_approved=1)
    years = p["horizon_years"][0] - p["contract_delay_months"][0] / BE.MONTHS_PER_YEAR
    per_year = (p["contacts_per_year"] * p["delta_hit_l3"] * p["causal_share"]
                * p["value_per_customer"])[0]
    expect = (years * per_year - p["l3_build_cost"][0] - p["contract_cost"][0]
              - p["horizon_years"][0] * p["l3_annual_ops"][0])
    assert abs(BE.npv_options(p)["L3"][0] - expect) < TOL


def test_安全评估未通过时只损失准备成本与合规成本():
    p = _fixed(is_ciio=1, assessment_approved=0)
    expect = -(p["l3_prep_share"][0] * p["l3_build_cost"][0] + p["assessment_cost"][0])
    assert abs(BE.npv_options(p)["L3"][0] - expect) < TOL
    p_ok = _fixed(is_ciio=0, gba_eligible=1, assessment_approved=0)   # 标准合同不受评估结果影响
    p_ok2 = _fixed(is_ciio=0, gba_eligible=1, assessment_approved=1)
    assert BE.npv_options(p_ok)["L3"][0] == BE.npv_options(p_ok2)["L3"][0]


def test_盈亏平衡点处L3净值为零():
    p = _fixed(is_ciio=0, gba_eligible=1, assessment_approved=1)
    p["delta_hit_l3"] = BE.breakeven_delta_hit(p)
    assert abs(BE.npv_options(p)["L3"][0]) < TOL * p["l3_build_cost"][0]


def test_延迟吃掉项目期时盈亏平衡为无穷():
    p = _fixed(is_ciio=1, assessment_delay_months=48)
    assert np.isinf(BE.breakeven_delta_hit(p)[0])


def test_EVPI非负且不小于EVPPI():
    p = _sample()
    npv = BE.npv_options(p)
    total = BE.evpi(npv)
    assert total >= 0
    for name in ("contacts_per_year", "delta_hit_l3", "is_ciio"):
        assert -TOL <= BE.evppi(npv, p[name], N_BINS) <= total + TOL


def test_确定选项时EVPI为零():
    npv = {"不做": np.zeros(N), "L1": np.full(N, -1.0), "L3": np.full(N, -2.0)}
    assert BE.evpi(npv) == pytest.approx(0.0)


def test_决定性参数的EVPPI等于EVPI():
    rng = np.random.default_rng(SEED)
    theta = rng.uniform(-1, 1, N)
    npv = {"不做": np.zeros(N), "L1": np.full(N, -1.0), "L3": theta}
    # theta 二值化后完全决定最优选项：分箱估计应接近 EVPI
    assert BE.evppi(npv, theta, N_BINS) == pytest.approx(BE.evpi(npv), rel=0.05)


def test_决策汇总概率之和为一():
    rows = BE.decision_summary(BE.npv_options(_sample()))
    assert [r["option"] for r in rows] == list(BE.OPTIONS)
    assert sum(r["prob_best"] for r in rows) == pytest.approx(1.0)


def test_盈亏平衡表随联系人数单调下降():
    rows = BE.breakeven_grid(_specs(), [5000, 50000], [12000], heavy=False)
    assert rows[0]["breakeven_delta_hit"] > rows[1]["breakeven_delta_hit"] > 0
    heavy = BE.breakeven_grid(_specs(), [5000], [12000], heavy=True)
    assert heavy[0]["breakeven_delta_hit"] > rows[0]["breakeven_delta_hit"]


def test_EVPPI输入长度不一致被拒绝():
    p = _sample()
    npv = BE.npv_options(p)
    with pytest.raises(ValueError):
        BE.evppi(npv, p["contacts_per_year"][:-1], N_BINS)
    with pytest.raises(ValueError):
        BE.evppi(npv, p["contacts_per_year"], 1)
