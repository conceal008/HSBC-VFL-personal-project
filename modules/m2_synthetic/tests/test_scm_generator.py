"""M2 · 合成数据生成器的单元测试。

测试的重点不是「函数能跑」，而是**生成器的地面真值是否真的可控**——
若不可控，下游全部结论都失去检验依据。
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m2_synthetic.components.scm_generator import (  # noqa: E402
    generate, load_scenarios, theoretical_gain_signal, usable_mask)

SEED = 11
BASE_RATE_TOL = 0.02
ZERO_TOL = 1e-9


@pytest.fixture(scope="module")
def scenarios():
    raw = yaml.safe_load(open(ROOT / "modules/m2_synthetic/configs/scenarios.yaml",
                              encoding="utf-8"))
    return {s.name: s for s in load_scenarios(raw)}


def test_同种子完全可复现(scenarios):
    a = generate(scenarios["S1_基准"], SEED)
    b = generate(scenarios["S1_基准"], SEED)
    for key in ("x_a", "x_b", "y", "treatment", "in_overlap"):
        assert np.array_equal(a[key], b[key]), f"{key} 在同种子下不一致"


def test_不同种子产生不同数据(scenarios):
    a = generate(scenarios["S1_基准"], SEED)
    b = generate(scenarios["S1_基准"], SEED + 1)
    assert not np.array_equal(a["x_a"], b["x_a"])


def test_正样本率被校准到配置值(scenarios):
    """截距求解必须让实际正样本率落在配置的 base_rate 附近，否则场景不可控。"""
    cfg = scenarios["S1_基准"]
    d = generate(cfg, SEED)
    assert abs(d["y_control"].mean() - cfg.base_rate) < BASE_RATE_TOL


def test_零互补场景的理论增益为零(scenarios):
    """这是生成器实现正确的硬判据：λ=0 时被动方对信号无贡献。"""
    d = generate(scenarios["S2_零互补"], SEED)
    g = theoretical_gain_signal(d)
    m = usable_mask(d)
    from sklearn.metrics import roc_auc_score
    gain = (roc_auc_score(d["y_control"][m], g["with_b"][m])
            - roc_auc_score(d["y_control"][m], g["without_b"][m]))
    assert abs(gain) < ZERO_TOL, f"零互补场景理论增益应为 0，实际 {gain}"


def test_互补性越高理论增益越大(scenarios):
    from sklearn.metrics import roc_auc_score
    base = scenarios["S1_基准"]
    gains = []
    for lam in (0.0, 1.0, 2.0):
        d = generate(replace(base, complementarity=lam), SEED)
        g, m = theoretical_gain_signal(d), usable_mask(d)
        gains.append(roc_auc_score(d["y_control"][m], g["with_b"][m])
                     - roc_auc_score(d["y_control"][m], g["without_b"][m]))
    assert gains[0] < gains[1] < gains[2], f"增益未随互补性单调上升：{gains}"


def test_可用掩码是交集与同意的合取(scenarios):
    d = generate(scenarios["S1_基准"], SEED)
    m = usable_mask(d)
    assert np.array_equal(m, d["in_overlap"] & d["consent"])
    assert m.sum() <= d["in_overlap"].sum()


def test_重叠率接近配置值(scenarios):
    cfg = replace(scenarios["S1_基准"], overlap_rate=0.3)
    d = generate(cfg, SEED)
    assert abs(d["in_overlap"].mean() - cfg.overlap_rate) < BASE_RATE_TOL


def test_八个场景全部可生成(scenarios):
    assert len(scenarios) >= 6, "框架要求场景覆盖 ≥6"
    for name, cfg in scenarios.items():
        d = generate(cfg, SEED)
        assert usable_mask(d).sum() > 0, f"{name} 的可用样本为 0"


# ————————————————— 非线性信号形式（S2.3）—————————————————

def test_三种信号形式在零互补时都无贡献(scenarios):
    """λ=0 的硬验收判据必须对三种形式同时成立，否则新形式的实现有误。"""
    from sklearn.metrics import roc_auc_score
    base = scenarios["S1_基准"]
    for form in ("linear", "interaction", "threshold"):
        cfg = replace(base, signal_form=form, complementarity=0.0)
        d = generate(cfg, SEED)
        g, m = theoretical_gain_signal(d), usable_mask(d)
        gain = (roc_auc_score(d["y_control"][m], g["with_b"][m])
                - roc_auc_score(d["y_control"][m], g["without_b"][m]))
        assert abs(gain) < ZERO_TOL, f"{form} 在 λ=0 时理论增益应为 0，实际 {gain}"


def test_三种信号形式的增益都随互补性上升(scenarios):
    from sklearn.metrics import roc_auc_score
    base = scenarios["S1_基准"]
    for form in ("linear", "interaction", "threshold"):
        gains = []
        for lam in (0.0, 0.8, 2.0):
            cfg = replace(base, signal_form=form, complementarity=lam)
            d = generate(cfg, SEED)
            g, m = theoretical_gain_signal(d), usable_mask(d)
            gains.append(roc_auc_score(d["y_control"][m], g["with_b"][m])
                         - roc_auc_score(d["y_control"][m], g["without_b"][m]))
        assert gains[0] < gains[1] < gains[2], f"{form} 的增益未单调上升：{gains}"


def test_默认形式为线性且向后兼容(scenarios):
    """既有全部结论都建立在 linear 上，默认值一旦改变会静默污染历史结论。"""
    base = scenarios["S1_基准"]
    assert base.signal_form == "linear"
    a = generate(base, SEED)
    b = generate(replace(base, signal_form="linear"), SEED)
    assert np.array_equal(a["y_control"], b["y_control"])


def test_不同信号形式产生不同标签(scenarios):
    base = scenarios["S1_基准"]
    lin = generate(replace(base, signal_form="linear"), SEED)
    inter = generate(replace(base, signal_form="interaction"), SEED)
    step = generate(replace(base, signal_form="threshold"), SEED)
    assert not np.array_equal(lin["y_control"], inter["y_control"])
    assert not np.array_equal(lin["y_control"], step["y_control"])


def test_未知信号形式被拒绝(scenarios):
    with pytest.raises(ValueError):
        generate(replace(scenarios["S1_基准"], signal_form="不存在的形式"), SEED)


def test_非线性场景库可加载并与线性对照一致():
    """N6_线性_对照 与原场景库 S1_基准 参数相同，理论增益应一致。"""
    from sklearn.metrics import roc_auc_score
    raw = yaml.safe_load(open(ROOT / "modules/m2_synthetic/configs/scenarios_nonlinear.yaml",
                              encoding="utf-8"))
    nonlinear = {s.name: s for s in load_scenarios(raw)}
    assert len(nonlinear) >= 5
    old = yaml.safe_load(open(ROOT / "modules/m2_synthetic/configs/scenarios.yaml",
                              encoding="utf-8"))
    s1 = [s for s in load_scenarios(old) if s.name == "S1_基准"][0]

    def gain(cfg):
        d = generate(cfg, SEED)
        g, m = theoretical_gain_signal(d), usable_mask(d)
        return (roc_auc_score(d["y_control"][m], g["with_b"][m])
                - roc_auc_score(d["y_control"][m], g["without_b"][m]))

    assert abs(gain(nonlinear["N6_线性_对照"]) - gain(s1)) < 1e-12
