"""M7 · 隐私攻击的单元测试。

攻击代码的正确性尤其重要：**攻击写弱了会得出「防护有效」的假结论**，
而假结论会被写进方案。因此这里的测试重点是攻击强度的下界。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m5_modeling.components import models as M  # noqa: E402
from modules.m7_security.components import attacks as A  # noqa: E402

SEED = 11
N = 1200
DIM = 4
ROUNDS = 50
DELTA = 1e-5
PERFECT = 0.999


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(SEED)
    x_a = rng.normal(size=(N, DIM))
    x_b = rng.normal(size=(N, DIM))
    logit = x_a[:, 0] - x_b[:, 0] + 2.0 * x_b[:, 1]
    y = (rng.random(N) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return x_a, x_b, y


def test_无防护时标签完全泄露(data):
    """首轮残差 r = 0.5 − y 的符号与标签一一对应，泄露必须是满分。"""
    x_a, x_b, y = data
    fed = M.fit_federated_logistic(x_a, x_b, y, ROUNDS, 0.5, 1e-4, 0.0, SEED)
    out = A.label_inference_from_residuals(fed.residual_history, y)
    assert out["leak_auc_首轮"] > PERFECT


def test_泄露强度取方向无关的最大值(data):
    """残差与标签负相关，攻击者翻转符号即可——不取 max 会把完全泄露误报为 0。"""
    y = np.array([0, 0, 1, 1] * 10)
    anti = -y.astype(float)                       # 完全反向的打分
    out = A.label_inference_from_residuals(np.array([anti]), y)
    assert out["leak_auc_首轮"] > PERFECT


def test_噪声降低单轮泄露(data):
    x_a, x_b, y = data
    clean = M.fit_federated_logistic(x_a, x_b, y, ROUNDS, 0.5, 1e-4, 0.0, SEED)
    noisy = M.fit_federated_logistic(x_a, x_b, y, ROUNDS, 0.5, 1e-4, 3.0, SEED)
    c = A.label_inference_from_residuals(clean.residual_history, y)["leak_auc_首轮"]
    n = A.label_inference_from_residuals(noisy.residual_history, y)["leak_auc_首轮"]
    assert n < c


def test_跨轮平均攻击强于单轮攻击(data):
    """本项目的核心证伪：噪声轮间独立而标签恒定，平均即可消掉噪声。"""
    x_a, x_b, y = data
    fed = M.fit_federated_logistic(x_a, x_b, y, 200, 0.5, 1e-4, 3.0, SEED)
    single = A.label_inference_from_residuals(fed.residual_history, y)["leak_auc_首轮"]
    from sklearn.metrics import roc_auc_score
    a = roc_auc_score(y, fed.residual_history.mean(axis=0))
    averaged = max(a, 1.0 - a)
    assert averaged > single, "跨轮平均攻击未强于单轮攻击——证伪逻辑失效"


def test_嵌入反演的R2在合理区间(data):
    x_a, x_b, y = data
    sn = M.fit_splitnn(x_a, x_b, y, "bidirectional", 8, 4, 3, 256, 0.01, SEED)
    inv = A.embedding_inversion(sn.embedding_history, sn.embed_index, x_b, SEED, 0.5)
    assert -1.0 <= inv["inv_r2_mean"] <= 1.0
    assert inv["inv_r2_max"] >= inv["inv_r2_mean"]
    assert len(inv["inv_r2_per_dim"]) == DIM


def test_反演必须按索引对齐(data):
    """打乱索引后 R² 必须显著下降——若不下降，说明对齐根本没起作用。"""
    x_a, x_b, y = data
    sn = M.fit_splitnn(x_a, x_b, y, "bidirectional", 8, 4, 3, 256, 0.01, SEED)
    good = A.embedding_inversion(sn.embedding_history, sn.embed_index, x_b, SEED, 0.5)
    shuffled = np.random.default_rng(SEED).permutation(sn.embed_index)
    bad = A.embedding_inversion(sn.embedding_history, shuffled, x_b, SEED, 0.5)
    assert good["inv_r2_mean"] > bad["inv_r2_mean"]


def test_形态B不回传梯度故无梯度泄露(data):
    x_a, x_b, y = data
    fr = M.fit_splitnn(x_a, x_b, y, "frozen_random", 8, 4, 3, 256, 0.01, SEED)
    out = A.label_inference_from_gradients(fr.grad_history, fr.embed_index, y)
    assert abs(out["leak_auc_梯度方向"] - 0.5) < 1e-9


def test_单轮epsilon随噪声单调下降():
    eps = [A.gaussian_epsilon_per_round(s, DELTA) for s in (0.1, 1.0, 10.0)]
    assert eps[0] > eps[1] > eps[2] > 0
    assert A.gaussian_epsilon_per_round(0.0, DELTA) == float("inf")
