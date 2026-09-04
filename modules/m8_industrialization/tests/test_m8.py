"""M8 · 部署护栏与故障注入的单元测试。

护栏的价值全在「该拒绝时真的拒绝」。一条只会放行的护栏比没有护栏更糟——
它给人以已经检查过的错觉。故这里每条规则都测两面：违规必拒、合规必放。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "platform" / "orchestration"))

import main_chain as MC  # noqa: E402
import pipeline as P  # noqa: E402

from modules.m8_industrialization.components import fault_injection as FI  # noqa: E402
from modules.m8_industrialization.components import guardrails as G  # noqa: E402

PROFILE_PATH = ROOT / "modules/m8_industrialization/configs/deployment_profile.yaml"
SMOKE_PATH = ROOT / "platform/configs/smoke.yaml"
FAULT_TOTAL = 7


@pytest.fixture(scope="module")
def profile():
    return yaml.safe_load(open(PROFILE_PATH, encoding="utf-8"))


@pytest.fixture(scope="module")
def smoke():
    return yaml.safe_load(open(SMOKE_PATH, encoding="utf-8"))


def _ok_config():
    return {"uplink_sigma": 0.1, "label_protection": "secure_aggregation",
            "gbdt_max_depth": 3, "k_anonymity": 10,
            "splitnn_mode": "frozen_pca", "route_selected_by": "exposure_and_compliance",
            "residual_plausibility_check": True, "list_stability_threshold": 0.95,
            "downlink_sigma": 1.0}


def test_合规配置放行(profile):
    r = G.check_deployment(_ok_config(), profile)
    assert r["verdict"] == G.VERDICT_PASS
    assert r["violations"] == []


def test_上行噪声不足被拒(profile):
    r = G.check_deployment(dict(_ok_config(), uplink_sigma=0.0), profile)
    assert r["verdict"] == G.VERDICT_BLOCK
    assert any(v["rule"].startswith("uplink_noise") for v in r["violations"])


def test_缺上行噪声字段也被拒(profile):
    cfg = _ok_config()
    del cfg["uplink_sigma"]
    assert G.check_deployment(cfg, profile)["verdict"] == G.VERDICT_BLOCK


def test_把高斯噪声当标签防护被拒(profile):
    r = G.check_deployment(dict(_ok_config(), label_protection="gaussian_noise"), profile)
    assert any("label_protection" in v["rule"] for v in r["violations"])


def test_树深超限被拒(profile):
    r = G.check_deployment(dict(_ok_config(), gbdt_max_depth=8), profile)
    assert any("gbdt_max_depth" in v["rule"] for v in r["violations"])


def test_k匿名过低被拒(profile):
    r = G.check_deployment(dict(_ok_config(), k_anonymity=3), profile)
    assert any("k_anonymity" in v["rule"] for v in r["violations"])


def test_随机冻结编码器被拒(profile):
    """它低于 L0 内地单方基线——不但没价值，还不如不做联邦。"""
    r = G.check_deployment(dict(_ok_config(), splitnn_mode="frozen_random"), profile)
    assert any("splitnn" in v["rule"] for v in r["violations"])


def test_以效果排序选路线被拒(profile):
    """分支卡 BC-M5-001 的 falsifier 已推翻该排序，不得再作选型依据。"""
    r = G.check_deployment(dict(_ok_config(), route_selected_by="effect_ranking"), profile)
    assert any("route_selection" in v["rule"] for v in r["violations"])


def test_未开合法性检查被拒(profile):
    """它是让上行加噪有意义的前提——缺了它，加多大噪声都能被放大幅度攻破。"""
    r = G.check_deployment(dict(_ok_config(), residual_plausibility_check=False), profile)
    assert any("residual_plausibility" in v["rule"] for v in r["violations"])


def test_名单稳定性阈值过松被拒(profile):
    """0.9 防不住定向操纵：45% 目标进入名单时重合度仍有 0.93。"""
    r = G.check_deployment(dict(_ok_config(), list_stability_threshold=0.9), profile)
    assert any("list_stability" in v["rule"] for v in r["violations"])


def test_下行噪声不足被拒(profile):
    """它防的不是标签（对标签无效），而是模型资产——无防护时名单可被完整复现。"""
    r = G.check_deployment(dict(_ok_config(), downlink_sigma=0.0), profile)
    assert any("model_asset_protection" in v["rule"] for v in r["violations"])


def test_每条违规都给出证据与整改方向(profile):
    r = G.check_deployment({"uplink_sigma": 0.0, "gbdt_max_depth": 9,
                            "k_anonymity": 2, "splitnn_mode": "frozen_random",
                            "label_protection": "gaussian_noise",
                            "route_selected_by": "effect_ranking",
                            "residual_plausibility_check": False,
                            "list_stability_threshold": 0.5,
                            "downlink_sigma": 0.0}, profile)
    assert len(r["violations"]) == r["checked_rules"], "六条规则应全部命中"
    for v in r["violations"]:
        assert v["evidence"] and v["remedy"], "拒绝必须附实测证据与整改方向"


def test_故障注入七项全通过(smoke):
    with tempfile.TemporaryDirectory() as d:
        cases = FI.run_all(P, MC, smoke, d)
    s = FI.summarize(cases)
    assert s["total"] == FAULT_TOTAL
    assert s["all_passed"], f"未通过：{s['failed_ids']}"


def test_故障注入编号与名称齐全(smoke):
    with tempfile.TemporaryDirectory() as d:
        cases = FI.run_all(P, MC, smoke, d)
    ids = [c["fault_id"] for c in cases]
    assert ids == ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]
    for c in cases:
        assert c["name"] and c["detail"]
