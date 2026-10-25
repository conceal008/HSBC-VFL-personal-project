"""M0 · S0.6R 关键信息基础设施运营者前提的单元测试。

⚠️ 被测函数编码的法条解读**未经法务复核**（W-001）。这些测试只检验
「代码是否忠实实现了所写的解读」，以及「默认参数下与 S0.6 已发布矩阵逐格一致」。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m0_compliance.components import export_mechanism as EM  # noqa: E402

S06_CONFIG = ROOT / "modules/m0_compliance/configs/s0_6_route_b.yaml"
S06_MATRIX = ROOT / "modules/m0_compliance/results/export_mechanism_matrix.csv"
THRESHOLDS = yaml.safe_load(S06_CONFIG.read_text(encoding="utf-8"))["thresholds"]
SMALL, LARGE = 5_000, 1_200_000


def test_默认参数与S0点6已发布矩阵逐格一致():
    with S06_MATRIX.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    for row in rows:
        sensitive = row["含敏感个人信息"] == "是"
        count = int(row["涉及人数"]) if row["cn→hk 提供"] != "无" else 0
        assert EM.required_mechanism(count, sensitive, THRESHOLDS) == row["所需机制"]


@pytest.mark.parametrize("sensitive", [True, False])
@pytest.mark.parametrize("gba", [True, False])
@pytest.mark.parametrize("count", [SMALL, LARGE])
def test_关基运营者一律安全评估(sensitive, gba, count):
    got = EM.required_mechanism(count, sensitive, THRESHOLDS, gba_eligible=gba, is_ciio=True)
    assert got == EM.SECURITY_ASSESSMENT


def test_关基运营者不提供个人信息时不适用():
    assert EM.required_mechanism(0, True, THRESHOLDS, is_ciio=True) == EM.NOT_APPLICABLE


def test_非关基小规模非敏感仍豁免():
    assert EM.required_mechanism(SMALL, False, THRESHOLDS, is_ciio=False) == EM.EXEMPT


def test_取最重机制():
    assert EM.strictest([EM.EXEMPT, EM.STANDARD_CONTRACT]) == EM.STANDARD_CONTRACT
    assert EM.strictest([EM.GBA_STANDARD_CONTRACT, EM.SECURITY_ASSESSMENT]) == EM.SECURITY_ASSESSMENT
    assert EM.strictest([]) == EM.NOT_APPLICABLE


def test_未知机制被拒绝():
    with pytest.raises(ValueError):
        EM.strictest(["随便写的机制"])
