"""M5 · 实验编排的单元测试（同时充当主链路冒烟）。"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m2_synthetic.components.scm_generator import load_scenarios  # noqa: E402
from modules.m5_modeling.components.experiment import run_one  # noqa: E402

SEED = 11
SMALL_N = 4000


@pytest.fixture(scope="module")
def tiny():
    raw = yaml.safe_load(open(ROOT / "modules/m2_synthetic/configs/scenarios.yaml",
                              encoding="utf-8"))
    cfg = [s for s in load_scenarios(raw) if s.name == "S1_基准"][0]
    hp = yaml.safe_load(open(ROOT / "modules/m5_modeling/configs/experiment.yaml",
                             encoding="utf-8"))["hyperparams"]
    hp = dict(hp, n_rounds=10, flr_rounds=50, nn_epochs=2)
    return replace(cfg, n_party_a=SMALL_N), hp


def test_主链路产出全部五级基线(tiny):
    cfg, hp = tiny
    rows = run_one(cfg, SEED, hp, "random")
    levels = {r["level"] for r in rows}
    for prefix in ("L0_", "L1_", "L2_", "L3a_", "L3b_", "L3c_", "L4_"):
        assert any(lv.startswith(prefix) for lv in levels), f"缺少 {prefix} 级基线"


def test_同种子同划分结果可复现(tiny):
    cfg, hp = tiny
    a = {r["level"]: r.get("auc") for r in run_one(cfg, SEED, hp, "random")}
    b = {r["level"]: r.get("auc") for r in run_one(cfg, SEED, hp, "random")}
    for lv, v in a.items():
        if v is not None:
            assert abs(v - b[lv]) < 1e-12, f"{lv} 在同种子下不可复现"


def test_两种划分口径都能跑通(tiny):
    cfg, hp = tiny
    for split in ("random", "oot"):
        rows = run_one(cfg, SEED, hp, split)
        assert all(r["split"] == split for r in rows)
        assert len(rows) > 0


def test_auc落在合法区间且样本量一致(tiny):
    cfg, hp = tiny
    rows = run_one(cfg, SEED, hp, "random")
    aucs = [r["auc"] for r in rows if r.get("auc") is not None]
    assert aucs and all(0.0 <= a <= 1.0 for a in aucs)
    n_test = {r["n_test"] for r in rows if "n_test" in r}
    assert len(n_test) == 1, "同一次运行的测试集大小必须一致"


def test_产出增量与名单重合度记录(tiny):
    cfg, hp = tiny
    rows = run_one(cfg, SEED, hp, "random")
    assert any(r["level"].endswith("_UPLIFT") for r in rows)
    assert any(r["level"].startswith("OVERLAP_") for r in rows)


def test_通信成本被记账(tiny):
    """VFL 的成本必须被量化——只报精度不报通信量是不完整的。"""
    cfg, hp = tiny
    rows = {r["level"]: r for r in run_one(cfg, SEED, hp, "random")}
    assert rows["L3a_联邦LR"]["comm_floats"] > 0
    assert rows["L3b_纵向GBDT"]["comm_floats"] > 0
    assert np.isfinite(rows["L3a_联邦LR"]["comm_rounds"])
