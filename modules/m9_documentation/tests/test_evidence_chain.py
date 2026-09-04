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


# ————————————————— 数字一致性核验（2026-09-04 审视后新增）—————————————————

def test_本仓库数字一致性满分(evidence_map):
    r = EC.verify_consistency(evidence_map["consistency_assertions"])
    assert r["rate"] == FULL, f"不一致项：{r['inconsistent_ids']}"


def test_陈旧数字被抓出(tmp_path):
    """真正的失误形态是「新值加上了、旧值忘删」——本检查抓的就是这个。"""
    doc = tmp_path / "d.md"
    doc.write_text("旧文里写着 17 组用例，另一处写 45 组用例。", encoding="utf-8")
    src = tmp_path / "s.sh"
    src.write_text("assert a\nassert b\n", encoding="utf-8")
    r = EC.verify_consistency([{
        "id": "T", "what": "用例数", "context_keyword": "组用例",
        "source": {"type": "count_matching", "path": str(src), "pattern": "^assert "},
        "must_appear_in": [str(doc)]}])
    assert r["rate"] == 0.0
    assert "17" in str(r["rows"][0]["problems"])


def test_真值缺失被抓出(tmp_path):
    doc = tmp_path / "d.md"
    doc.write_text("这里写着 99 组用例。", encoding="utf-8")
    src = tmp_path / "s.sh"
    src.write_text("assert a\n", encoding="utf-8")
    r = EC.verify_consistency([{
        "id": "T", "what": "用例数", "context_keyword": "组用例",
        "source": {"type": "count_matching", "path": str(src), "pattern": "^assert "},
        "must_appear_in": [str(doc)]}])
    assert r["rate"] == 0.0
    assert "未出现真值" in str(r["rows"][0]["problems"])


def test_正确陈述不被跳过规则误伤(tmp_path):
    """跳过规则只对与真值不符的数字生效——否则会把「写对了」判成「没写」。"""
    doc = tmp_path / "d.md"
    doc.write_text("共 2 组用例。", encoding="utf-8")
    src = tmp_path / "s.sh"
    src.write_text("assert a\nassert b\n", encoding="utf-8")
    r = EC.verify_consistency([{
        "id": "T", "what": "用例数", "context_keyword": "组用例",
        "source": {"type": "count_matching", "path": str(src), "pattern": "^assert "},
        "must_appear_in": [str(doc)]}])
    assert r["rate"] == FULL


def test_历史对照与显式豁免被放行(tmp_path):
    """同行并列真值的是对照；带 `数字豁免:` 标记的是历史记录。"""
    src = tmp_path / "s.sh"
    src.write_text("assert a\nassert b\n", encoding="utf-8")
    spec = {"id": "T", "what": "用例数", "context_keyword": "组用例",
            "source": {"type": "count_matching", "path": str(src), "pattern": "^assert "}}

    compare = tmp_path / "a.md"
    compare.write_text("此前写着 9 组用例，而实际是 2 组用例。", encoding="utf-8")
    assert EC.verify_consistency([dict(spec, must_appear_in=[str(compare)])])["rate"] == FULL

    waived = tmp_path / "b.md"
    waived.write_text("当时交付 9 组用例（数字豁免: 历史值）。\n共 2 组用例。",
                      encoding="utf-8")
    assert EC.verify_consistency([dict(spec, must_appear_in=[str(waived)])])["rate"] == FULL


def test_未知来源类型被拒绝():
    r = EC.verify_consistency([{"id": "T", "what": "x", "context_keyword": "个",
                                "source": {"type": "不存在的类型"},
                                "must_appear_in": []}])
    assert r["rate"] == 0.0
