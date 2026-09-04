"""M6 · 评估指标的单元测试。判别力指标与 sklearn 对齐；增量指标验证其定义性质。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m6_evaluation.components import metrics as MT  # noqa: E402

SEED = 11
N = 2000
TOL = 1e-9
TOPK = 0.1


def _data():
    rng = np.random.default_rng(SEED)
    score = rng.normal(size=N)
    y = (rng.random(N) < 1.0 / (1.0 + np.exp(-score))).astype(int)
    return y, score


def test_auc_与_sklearn_一致():
    y, s = _data()
    assert abs(MT.auc(y, s) - roc_auc_score(y, s)) < TOL


def test_auc_对单调变换不变():
    """AUC 只依赖排序：任何单调变换都不应改变它。"""
    y, s = _data()
    assert abs(MT.auc(y, s) - MT.auc(y, s * 3.0 + 7.0)) < TOL


def test_随机打分的_auc_接近一半():
    rng = np.random.default_rng(SEED)
    y = (rng.random(N) < 0.5).astype(int)
    assert abs(MT.auc(y, rng.normal(size=N)) - 0.5) < 0.05


def test_ks_在零一之间且完美分离时为一():
    y, s = _data()
    assert 0.0 <= MT.ks(y, s) <= 1.0
    assert abs(MT.ks(y, y.astype(float)) - 1.0) < TOL


def test_lift_与_recall_的定义关系():
    """Recall@K = Lift@K × K，这是两者定义的直接推论。"""
    y, s = _data()
    lift, recall = MT.lift_at_k(y, s, TOPK), MT.recall_at_k(y, s, TOPK)
    assert abs(recall - lift * TOPK) < 0.01


def test_无区分力时_lift_接近一():
    rng = np.random.default_rng(SEED)
    y = (rng.random(N) < 0.3).astype(int)
    assert abs(MT.lift_at_k(y, rng.normal(size=N), TOPK) - 1.0) < 0.4


def test_topk重合度的边界():
    rng = np.random.default_rng(SEED)
    s = rng.normal(size=N)
    assert abs(MT.top_k_overlap(s, s, TOPK) - 1.0) < TOL
    assert MT.top_k_overlap(s, -s, TOPK) < 0.05


def test_增量指标能识别真实处理效应():
    """构造处理效应只在高分段为正的数据，uplift 打分必须能把它们排在前面。"""
    rng = np.random.default_rng(SEED)
    up = rng.normal(size=N)
    t = (rng.random(N) < 0.5).astype(int)
    y = ((rng.random(N) < 0.2 + 0.3 * t * (up > 1.0))).astype(int)
    assert MT.uplift_at_k(y, t, up, TOPK) > MT.uplift_at_k(y, t, -up, TOPK)


def test_auuc_对无效打分接近零():
    rng = np.random.default_rng(SEED)
    t = (rng.random(N) < 0.5).astype(int)
    y = (rng.random(N) < 0.3).astype(int)
    assert abs(MT.auuc(y, t, rng.normal(size=N))) < 0.02


def test_qini曲线长度与单调起点():
    y, s = _data()
    t = (np.arange(N) % 2).astype(int)
    x, g = MT.qini_curve(y, t, s)
    assert len(x) == len(g) and len(x) > 1
    assert abs(g[0]) < 1.0
