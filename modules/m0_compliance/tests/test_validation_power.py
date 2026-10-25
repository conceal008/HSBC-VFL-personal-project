"""M0 · S0.1P 统计功效组件的单元测试。

只检验「代码是否忠实实现了所写的近似公式」与单调性；
近似本身的准确度须在 M2 合成数据上用模拟复核（见 DR-MX-002 · D-2 的推翻条件）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m0_compliance.components import sample_size as SS  # noqa: E402
from modules.m0_compliance.components import validation_power as VP  # noqa: E402

ALPHA = 0.05
POWER = 0.8
AUC = 0.86
BASE = 0.03
CORR = 0.8
REFERENCE_POSITIVES = 300
REFERENCE_NEGATIVES = 8100
REFERENCE_SE = 0.013690      # Hanley–McNeil 手算值
SE_TOL = 1e-5
RELATIVE_TOL = 0.02
LOOSE, TIGHT = 0.02, 0.01
HIGH_CORR = 0.9
SPARSE_BASE = 0.01
N_EFF = 30_000
SMALL_N = 5_814
TINY_N = 3
S0_1_LIFT = 0.20
NEW_INVESTORS, MONTHS, WINDOW = 8_800, 8, 90
SHARES = [0.1, 0.2]
EXPECTED_ANNUAL = 13_200


def test_标准误与手算一致():
    se = VP.auc_standard_error(AUC, REFERENCE_POSITIVES, REFERENCE_NEGATIVES)
    assert abs(se - REFERENCE_SE) < SE_TOL


def test_正例越多标准误越小():
    small = VP.auc_standard_error(AUC, REFERENCE_POSITIVES, REFERENCE_NEGATIVES)
    large = VP.auc_standard_error(AUC, REFERENCE_POSITIVES * 2, REFERENCE_NEGATIVES * 2)
    assert large < small


def test_所需正例数满足且最小():
    n = VP.positives_for_paired_auc_mde(AUC, LOOSE, BASE, CORR, ALPHA, POWER)
    assert VP.paired_auc_mde(AUC, n, BASE, CORR, ALPHA, POWER) <= LOOSE
    assert VP.paired_auc_mde(AUC, n - 1, BASE, CORR, ALPHA, POWER) > LOOSE


def test_差异越小所需正例越多():
    loose = VP.positives_for_paired_auc_mde(AUC, LOOSE, BASE, CORR, ALPHA, POWER)
    tight = VP.positives_for_paired_auc_mde(AUC, TIGHT, BASE, CORR, ALPHA, POWER)
    assert tight > loose


def test_估计相关越高所需正例越少():
    low = VP.positives_for_paired_auc_mde(AUC, LOOSE, BASE, CORR, ALPHA, POWER)
    high = VP.positives_for_paired_auc_mde(AUC, LOOSE, BASE, HIGH_CORR, ALPHA, POWER)
    assert high < low


def test_正例下界几乎不随正例率变化():
    dense = VP.positives_for_paired_auc_mde(AUC, LOOSE, BASE, CORR, ALPHA, POWER)
    sparse = VP.positives_for_paired_auc_mde(AUC, LOOSE, SPARSE_BASE, CORR, ALPHA, POWER)
    assert abs(dense - sparse) / dense < RELATIVE_TOL   # 相差不到 2%：精度由正例数决定


def test_正例表覆盖全部组合且客户数由正例率换算():
    rows = VP.positives_table(AUC, [LOOSE, TIGHT], [BASE], [CORR], ALPHA, POWER)
    assert len(rows) == len([LOOSE, TIGHT])
    for r in rows:
        assert r["implied_customers"] == math.ceil(r["min_positives"] / BASE)


def test_小人群可检出效应与S0点1口径一致():
    lift = VP.detectable_relative_lift(BASE, N_EFF, ALPHA, POWER, 0.5)
    assert lift is not None and lift <= S0_1_LIFT
    per_arm = SS.required_n_per_arm(BASE, lift, ALPHA, POWER)
    assert per_arm <= N_EFF // 2


def test_人群越小可检出效应越大():
    big = VP.detectable_relative_lift(BASE, N_EFF, ALPHA, POWER, 0.5)
    small = VP.detectable_relative_lift(BASE, SMALL_N, ALPHA, POWER, 0.5)
    assert big is not None and small is not None and small > big


def test_极小人群标记正态近似失效():
    rows = VP.small_population_table(BASE, {"tiny": TINY_N, "big": N_EFF}, ALPHA, POWER, 0.5)
    by = {r["scenario"]: r for r in rows}
    assert by["tiny"]["normal_approx_valid"] is False
    assert by["big"]["normal_approx_valid"] is True


def test_对照臂为空时返回None():
    assert VP.detectable_relative_lift(BASE, 1, ALPHA, POWER, 0.5) is None


def test_市场上限按份额线性():
    rows = VP.market_positive_ceiling(NEW_INVESTORS, MONTHS, WINDOW, SHARES)
    assert rows[0]["market_new_per_year"] == EXPECTED_ANNUAL
    ratio = rows[1]["institution_positive_ceiling_per_window"] / rows[0]["institution_positive_ceiling_per_window"]
    assert abs(ratio - SHARES[1] / SHARES[0]) < SE_TOL


def test_非法参数被拒绝():
    with pytest.raises(ValueError):
        VP.auc_standard_error(1.0, REFERENCE_POSITIVES, REFERENCE_NEGATIVES)
    with pytest.raises(ValueError):
        VP.auc_standard_error(AUC, 0, REFERENCE_NEGATIVES)
    with pytest.raises(ValueError):
        VP.negatives_for(REFERENCE_POSITIVES, 0.0)
    with pytest.raises(ValueError):
        VP.paired_auc_mde(AUC, REFERENCE_POSITIVES, BASE, 1.0, ALPHA, POWER)
    with pytest.raises(ValueError):
        VP.positives_for_paired_auc_mde(AUC, 0.0, BASE, CORR, ALPHA, POWER)
    with pytest.raises(ValueError):
        VP.paired_auc_mde(AUC, REFERENCE_POSITIVES, BASE, CORR, 0.0, POWER)
    with pytest.raises(ValueError):
        VP.detectable_relative_lift(BASE, N_EFF, ALPHA, POWER, 1.0)
    with pytest.raises(ValueError):
        VP.market_positive_ceiling(NEW_INVESTORS, 0, WINDOW, SHARES)
    with pytest.raises(ValueError):
        VP.market_positive_ceiling(NEW_INVESTORS, MONTHS, WINDOW, [0.0])
