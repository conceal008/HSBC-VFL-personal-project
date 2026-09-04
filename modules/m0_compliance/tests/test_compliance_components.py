"""M0 · 合规计算组件的单元测试。

⚠️ 本项目不设合规角色，被测函数编码的法条解读**未经法务复核**（豁免 W-001）。
这些测试只检验「代码是否忠实实现了所写的解读」，**不检验解读本身是否正确**。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m0_compliance.components import export_mechanism as EM  # noqa: E402
from modules.m0_compliance.components import funnel as FN  # noqa: E402
from modules.m0_compliance.components import sample_size as SS  # noqa: E402

THRESHOLDS = {
    "sensitive_assessment_at_or_above": 10_000,
    "non_sensitive_exempt_below": 100_000,
    "non_sensitive_assessment_at_or_above": 1_000_000,
}
ALPHA = 0.05
POWER = 0.8


# ————————————————— 出境机制 —————————————————

def test_不提供个人信息时不适用():
    assert EM.required_mechanism(0, False, THRESHOLDS) == EM.NOT_APPLICABLE


def test_敏感信息按一万人分档():
    assert EM.required_mechanism(9_999, True, THRESHOLDS) == EM.STANDARD_CONTRACT
    assert EM.required_mechanism(10_000, True, THRESHOLDS) == EM.SECURITY_ASSESSMENT


def test_非敏感信息按十万与百万分档():
    assert EM.required_mechanism(99_999, False, THRESHOLDS) == EM.EXEMPT
    assert EM.required_mechanism(100_000, False, THRESHOLDS) != EM.EXEMPT
    assert EM.required_mechanism(1_000_000, False, THRESHOLDS) == EM.SECURITY_ASSESSMENT


def test_大湾区标准合同免除数量门槛():
    """这是 S0.7 推翻 S0.5 判断的依据：适用时不再按人数分档。"""
    for n in (1, 10_000, 5_000_000):
        assert EM.required_mechanism(n, True, THRESHOLDS,
                                     gba_eligible=True) == EM.GBA_STANDARD_CONTRACT


def test_敏感比非敏感要求更严():
    n = 50_000
    assert EM.required_mechanism(n, True, THRESHOLDS) == EM.SECURITY_ASSESSMENT
    assert EM.required_mechanism(n, False, THRESHOLDS) != EM.SECURITY_ASSESSMENT


# ————————————————— 样本量 —————————————————

def test_基线率越低所需样本越大():
    a = SS.required_n_per_arm(0.01, 0.2, ALPHA, POWER)
    b = SS.required_n_per_arm(0.10, 0.2, ALPHA, POWER)
    assert a > b


def test_要检出的提升越小所需样本越大():
    a = SS.required_n_per_arm(0.05, 0.1, ALPHA, POWER)
    b = SS.required_n_per_arm(0.05, 0.4, ALPHA, POWER)
    assert a > b


def test_功效越高所需样本越大():
    assert (SS.required_n_per_arm(0.05, 0.2, ALPHA, 0.9)
            > SS.required_n_per_arm(0.05, 0.2, ALPHA, POWER))


def test_非法参数被拒绝():
    with pytest.raises(ValueError):
        SS.required_n_per_arm(0.0, 0.2, ALPHA, POWER)
    with pytest.raises(ValueError):
        SS.required_n_per_arm(0.05, 0.0, ALPHA, POWER)


def test_样本量表覆盖全部组合():
    rows = SS.sample_size_table([0.02, 0.05], [0.1, 0.2], ALPHA, POWER, 0.5)
    assert len(rows) == 4
    with pytest.raises(ValueError):
        SS.sample_size_table([0.02], [0.1], ALPHA, POWER, 1.0)


# ————————————————— 折损漏斗 —————————————————

def _base():
    return {"low": 1_000_000.0, "mid": 1_000_000.0, "high": 1_000_000.0}


def _stages():
    return [{"name": "资格率", "rates": {"low": 0.05, "mid": 0.10, "high": 0.20}},
            {"name": "同意率", "rates": {"low": 0.30, "mid": 0.50, "high": 0.70}}]


def test_漏斗逐层单调递减():
    rows = FN.funnel(_base(), _stages())
    for key in ("remaining_low", "remaining_mid", "remaining_high"):
        vals = [r[key] for r in rows]
        assert all(a >= b for a, b in zip(vals, vals[1:])), f"{key} 未单调递减"


def test_乐观档存活数不低于悲观档():
    rows = FN.funnel(_base(), _stages())
    assert rows[-1]["remaining_high"] >= rows[-1]["remaining_low"]


def test_折损率等于末层比首层():
    rows = FN.funnel(_base(), _stages())
    assert abs(FN.attrition_rate(rows, "mid")
               - rows[-1]["remaining_mid"] / rows[0]["remaining_mid"]) < 1e-12


def test_缺档位或非法档位被拒绝():
    with pytest.raises(ValueError):
        FN.funnel({"low": 1.0, "mid": 1.0}, _stages())
    with pytest.raises(ValueError):
        FN.attrition_rate(FN.funnel(_base(), _stages()), "medium")
