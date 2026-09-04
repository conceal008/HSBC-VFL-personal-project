"""M3 · 实体对齐仿真的单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m3_alignment.components.psi import (  # noqa: E402
    apply_misattribution, psi_cost, simulate_psi)

SEED = 11
N = 1000
RATE = 0.2


def _masks():
    rng = np.random.default_rng(SEED)
    in_overlap = rng.random(N) < 0.4
    mismatch = np.zeros(N, dtype=bool)
    idx = np.flatnonzero(in_overlap)[:50]
    mismatch[idx] = True
    return in_overlap, mismatch


def test_无错误时对齐完美():
    in_overlap, _ = _masks()
    r = simulate_psi(in_overlap, np.zeros(N, dtype=bool))
    assert r["precision"] == 1.0 and r["recall"] == 1.0
    assert r["false_positive"] == 0 and r["false_negative"] == 0


def test_漏配减少匹配量且降低召回():
    in_overlap, mismatch = _masks()
    r = simulate_psi(in_overlap, mismatch)
    assert r["n_matched"] < r["n_true_overlap"]
    assert r["recall"] < 1.0
    assert r["false_negative"] == int(mismatch.sum())


def test_匹配掩码不超出真实交集():
    in_overlap, mismatch = _masks()
    r = simulate_psi(in_overlap, mismatch)
    assert r["matched_mask"].sum() == r["n_matched"]


def test_错配不会把样本换回自己():
    """错位置换必须保证被选中的样本都拿到了别人的特征。"""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N, 5))
    y = apply_misattribution(x, RATE, SEED)
    changed = ~np.all(np.isclose(x, y), axis=1)
    assert changed.sum() > 0, "错配率 20% 却无任何样本被换"
    assert changed.sum() <= int(N * RATE)


def test_错配率为零时数据不变():
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N, 5))
    assert np.array_equal(apply_misattribution(x, 0.0, SEED), x)


def test_错配不改变数据的整体分布():
    """置换只是重排行，列的均值必须保持不变——否则说明实现引入了额外噪声。"""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N, 5))
    y = apply_misattribution(x, RATE, SEED)
    assert np.allclose(x.mean(axis=0), y.mean(axis=0), atol=0.05)


def test_成本随规模线性增长():
    small = psi_cost(1000, 1000)
    large = psi_cost(10000, 10000)
    assert large["comm_bytes"] == small["comm_bytes"] * 10
    assert large["exponentiations"] == small["exponentiations"] * 10
    assert small["rounds"] == large["rounds"]
