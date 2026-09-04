"""M5 · 超参搜索的单元测试。

本组件唯一容易出错、且出错会静默污染结论的地方是**选参用了哪个划分**。
在测试集上选参会让搜索空间大的级别虚高更多——恰好是 L1，
污染方向有利于 L1，会毁掉 C1 的可信度。故这里专门锁住这条不变量。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m5_modeling.components import search as S  # noqa: E402

SEED = 11
N = 900
DIM = 4
TRAIN_FRAC = 0.6
VALID_FRAC = 0.2
FIXED = {"n_bins": 32, "min_gain": 0.0, "batch_size": 128}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(SEED)
    x_a = rng.normal(size=(N, DIM))
    x_b = rng.normal(size=(N, DIM))
    logit = x_a[:, 0] - x_b[:, 0] + 2.0 * x_b[:, 1]
    y = (rng.random(N) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return x_a, x_b, y


def test_网格展开覆盖全部组合():
    pts = list(S.grid_points({"a": [1, 2, 3], "b": [10, 20]}))
    assert len(pts) == 6
    assert len({tuple(sorted(p.items())) for p in pts}) == 6


def test_单值网格只出一个点():
    assert len(list(S.grid_points({"a": [1]}))) == 1


def test_三分划分互不相交且覆盖全体():
    tr, va, te = S.three_way_split(N, TRAIN_FRAC, VALID_FRAC, SEED)
    assert len(set(tr) & set(va)) == 0
    assert len(set(tr) & set(te)) == 0
    assert len(set(va) & set(te)) == 0
    assert len(tr) + len(va) + len(te) == N
    assert abs(len(tr) / N - TRAIN_FRAC) < 0.01


def test_三分划分同种子可复现():
    a = S.three_way_split(N, TRAIN_FRAC, VALID_FRAC, SEED)
    b = S.three_way_split(N, TRAIN_FRAC, VALID_FRAC, SEED)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_选参只看验证集(data):
    """核心不变量：返回的最优点必须是**验证集**上最好的，而不是测试集上最好的。

    构造一个网格，逐点算出验证与测试 AUC；若实现误用测试集选参，
    返回的 test_auc 就会等于全网格测试 AUC 的最大值——本测试正是要排除这一点。
    """
    x_a, x_b, y = data
    tr, va, te = S.three_way_split(N, TRAIN_FRAC, VALID_FRAC, SEED)
    grid = {"c_reg": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    best = S.search_level("L0_LR", grid, FIXED, x_a, x_b, y, tr, va, te, SEED)

    pairs = [S.evaluate_point("L0_LR", p, FIXED, x_a, x_b, y, tr, va, te, SEED)
             for p in S.grid_points(grid)]
    valid_aucs = [v for v, _ in pairs]
    best_by_valid = max(pairs, key=lambda pr: pr[0])

    assert best["valid_auc"] == pytest.approx(max(valid_aucs))
    assert best["test_auc"] == pytest.approx(best_by_valid[1])


def test_各级别都能被评估(data):
    x_a, x_b, y = data
    tr, va, te = S.three_way_split(N, TRAIN_FRAC, VALID_FRAC, SEED)
    points = {
        "L0_LR": {"c_reg": 1.0},
        "L1_LR": {"c_reg": 1.0, "n_segments": 16, "k_anonymity": 5},
        "L2_LR": {"c_reg": 1.0},
        "L4_LR": {"c_reg": 1.0},
        "L3a_联邦LR": {"flr_rounds": 50, "flr_lr": 0.5, "flr_l2": 1e-4},
        "L3b_纵向GBDT": {"n_rounds": 10, "max_depth": 2, "gbdt_lr": 0.1, "reg_lambda": 1.0},
        "L3c_SplitNN": {"dim_hidden": 8, "dim_embed": 2, "nn_epochs": 2, "nn_lr": 0.01},
    }
    for level, point in points.items():
        v, t = S.evaluate_point(level, point, FIXED, x_a, x_b, y, tr, va, te, SEED)
        assert 0.0 <= v <= 1.0 and 0.0 <= t <= 1.0, f"{level} 的 AUC 越界"


def test_未知级别被拒绝(data):
    x_a, x_b, y = data
    tr, va, te = S.three_way_split(N, TRAIN_FRAC, VALID_FRAC, SEED)
    with pytest.raises(ValueError):
        S.evaluate_point("L9_不存在", {}, FIXED, x_a, x_b, y, tr, va, te, SEED)


def test_报告的点数等于网格大小(data):
    x_a, x_b, y = data
    tr, va, te = S.three_way_split(N, TRAIN_FRAC, VALID_FRAC, SEED)
    grid = {"c_reg": [0.1, 1.0, 10.0]}
    assert S.search_level("L0_LR", grid, FIXED, x_a, x_b, y, tr, va, te,
                          SEED)["n_points"] == 3
