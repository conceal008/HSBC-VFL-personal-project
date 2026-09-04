"""M9 · 证据链核验的单元测试。

本组件的价值全在「该拦时真的拦」。一个只会报 100% 的核验器比没有更糟——
它让人以为已经查过了。故每项都测两面：缺证据必判不可追溯、齐全必判通过。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m9_documentation.components import evidence_chain as EC  # noqa: E402

MAP_PATH = ROOT / "modules/m9_documentation/configs/evidence_map.yaml"
REAL_FILE = "modules/m9_documentation/DPIA.md"
FAKE_FILE = "modules/m9_documentation/不存在的文件.md"
FULL = 1.0


@pytest.fixture(scope="module")
def evidence_map():
    return yaml.safe_load(open(MAP_PATH, encoding="utf-8"))


def test_本仓库三项核验均为满分(evidence_map):
    """这是 M9 的放行判据：一致性 100% · 风险溯源 100% · 交付清单 10/10。"""
    assert EC.verify_claims(evidence_map["claims"])["rate"] == FULL
    assert EC.verify_risk_traceability(evidence_map["risks"])["rate"] == FULL
    d = EC.verify_deliverables(evidence_map["deliverables"])
    assert d["rate"] == FULL and d["total"] >= 10


def test_证据文件不存在则判不可追溯():
    r = EC.verify_claims([{"id": "X", "statement": "s", "nature": "实测",
                           "evidence": [FAKE_FILE]}])
    assert r["rate"] == 0.0
    assert r["rows"][0]["problems"][0]["status"] == EC.STATUS_MISSING


def test_没有证据的结论判不可追溯():
    r = EC.verify_claims([{"id": "X", "statement": "s", "nature": "实测", "evidence": []}])
    assert r["rate"] == 0.0, "空证据列表不得算作可追溯"


def test_性质未声明则判不可追溯():
    """性质（实测/推论/假设/未复核）必须显式——留空会让读者误以为是实测。"""
    r = EC.verify_claims([{"id": "X", "statement": "s", "nature": "",
                           "evidence": [REAL_FILE]}])
    assert r["rate"] == 0.0
    assert r["rows"][0]["nature_declared"] is False


def test_证据齐全且性质已声明则通过():
    r = EC.verify_claims([{"id": "X", "statement": "s", "nature": "合成数据实测",
                           "evidence": [REAL_FILE]}])
    assert r["rate"] == FULL


def test_风险无缓解措施则判未溯源():
    r = EC.verify_risk_traceability([{"id": "R", "risk": "x", "severity": "高",
                                      "mitigation": "", "evidence": [REAL_FILE]}])
    assert r["rate"] == 0.0, "只写风险不写缓解措施不算溯源"


def test_风险无实验依据则判未溯源():
    r = EC.verify_risk_traceability([{"id": "R", "risk": "x", "severity": "高",
                                      "mitigation": "有措施", "evidence": []}])
    assert r["rate"] == 0.0, "风险必须指向具体实验产出，不能只写「可能存在」"


def test_交付物缺失被识别():
    d = EC.verify_deliverables([{"name": "在", "path": REAL_FILE},
                                {"name": "不在", "path": FAKE_FILE}])
    assert d["present"] == 1 and d["missing"] == ["不在"]


def test_涉法结论必须标注未复核(evidence_map):
    """本项目不设合规角色，涉法结论一律不得声称已复核。"""
    legal = [c for c in evidence_map["claims"] if str(c["id"]).startswith("L")]
    assert legal, "证据映射中应存在涉法结论"
    for c in legal:
        assert "未复核" in c["nature"], f"{c['id']} 的性质未标注未复核"
