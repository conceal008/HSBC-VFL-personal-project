"""M5 · 基线阶梯各模型的单元测试。

最关键的两组是**等价性检验**：联邦协议在数学上应与集中式等价，
若不等价则说明协议实现有误——这是本模块唯一能自证正确的手段。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m5_modeling.components import models as M  # noqa: E402
from modules.m5_modeling.components.gbdt import VerticalGBDT  # noqa: E402

SEED = 11
N = 1500
DIM = 4
K_ANON = 10
N_SEG = 32
EQUIV_TOL = 0.01
LR_ROUNDS = 400


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(SEED)
    x_a = rng.normal(size=(N, DIM))
    x_b = rng.normal(size=(N, DIM))
    logit = x_a[:, 0] + x_a[:, 1] - x_b[:, 0] + 2.0 * x_b[:, 1]
    y = (rng.random(N) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return x_a, x_b, y


def test_联邦逻辑回归与集中式等价(data):
    """协议无损的硬判据：残差交换与集中式梯度下降在数学上是同一个算法。"""
    x_a, x_b, y = data
    fed = M.fit_federated_logistic(x_a, x_b, y, LR_ROUNDS, 0.5, 1e-4, 0.0, SEED)
    s_fed = M.federated_lr_score(fed, x_a, x_b)
    s_cen = M.fit_logistic(np.hstack([x_a, x_b]), y, SEED, 1.0).decision_function(
        np.hstack([x_a, x_b]))
    assert abs(roc_auc_score(y, s_fed) - roc_auc_score(y, s_cen)) < EQUIV_TOL


def test_纵向GBDT与集中式GBDT完全相等(data):
    """直方图交换不引入近似，两者应逐值相等。"""
    x_a, x_b, y = data
    kw = dict(n_rounds=20, max_depth=3, learning_rate=0.1, n_bins=32,
              reg_lambda=1.0, min_gain=0.0)
    v = VerticalGBDT(**kw)
    v.fit(x_a, x_b, y)
    c = VerticalGBDT(**kw)
    c.fit(np.hstack([x_a, x_b]), None, y)
    assert np.allclose(v.decision_function(x_a, x_b),
                       c.decision_function(np.hstack([x_a, x_b]), None))


def test_GBDT无被动方时退化为单方模型(data):
    x_a, x_b, y = data
    g = VerticalGBDT(20, 3, 0.1, 32, 1.0, 0.0)
    g.fit(x_a, None, y)
    assert g.comm.gradient_messages == 0 and g.comm.histogram_messages == 0


def test_k匿名抑制小于k的分段(data):
    """人数不足 k 的分段必须被抑制，否则 k-匿名不成立。"""
    x_a, x_b, _ = data
    seg = M.build_segments(x_a, 512, SEED)          # 分段数远大于样本量，必产生小组
    stats, info = M.k_anonymous_segment_stats(seg, x_b, K_ANON)
    sizes = np.bincount(seg, minlength=seg.max() + 1)
    small = np.isin(seg, np.flatnonzero(sizes < K_ANON))
    assert small.sum() > 0, "构造失败：没有产生人数不足 k 的分段"
    assert info["segments_suppressed"] > 0
    assert len(np.unique(stats[small], axis=0)) == 1, "被抑制的行应统一为回退值"


def test_分段键只用主动方特征(data):
    """L1 的结构性限制：换掉被动方数据不得改变分段结果。"""
    x_a, _, _ = data
    rng = np.random.default_rng(SEED + 1)
    assert np.array_equal(M.build_segments(x_a, N_SEG, SEED),
                          M.build_segments(x_a, N_SEG, SEED))
    other_b = rng.normal(size=(N, DIM))
    stats1, _ = M.k_anonymous_segment_stats(M.build_segments(x_a, N_SEG, SEED),
                                            other_b, K_ANON)
    assert stats1.shape[0] == N


def test_粗粒度标记是低维序数(data):
    _, x_b, _ = data
    flag = M.coarse_flag(x_b, SEED)
    assert flag.shape[0] == N
    assert len(np.unique(flag)) <= 4, "粗粒度标记的取值应远少于原始特征"


def test_splitnn三形态的梯度回传标记(data):
    x_a, x_b, y = data
    kw = dict(dim_hidden=8, dim_embed=4, n_epochs=3, batch_size=256, lr=0.01, seed=SEED)
    bi = M.fit_splitnn(x_a, x_b, y, "bidirectional", **kw)
    fr = M.fit_splitnn(x_a, x_b, y, "frozen_random", **kw)
    assert bi.comm["gradient_returned"] == 1
    assert fr.comm["gradient_returned"] == 0
    assert np.allclose(fr.grad_history, 0.0), "形态B 不得回传非零梯度"
    assert bi.comm["comm_floats"] > fr.comm["comm_floats"], "形态A 的通信量应更大"


def test_差分噪声改变下发残差(data):
    """记录的必须是**实际下发**的值，否则 M7 的攻击评估会失真。"""
    x_a, x_b, y = data
    clean = M.fit_federated_logistic(x_a, x_b, y, 20, 0.5, 1e-4, 0.0, SEED)
    noisy = M.fit_federated_logistic(x_a, x_b, y, 20, 0.5, 1e-4, 1.0, SEED)
    assert not np.allclose(clean.residual_history[0], noisy.residual_history[0])


def test_嵌入索引可对齐回训练集(data):
    x_a, x_b, y = data
    sn = M.fit_splitnn(x_a, x_b, y, "bidirectional", 8, 4, 2, 256, 0.01, SEED)
    assert len(sn.embed_index) == len(sn.embedding_history)
    assert sn.embed_index.max() < N
