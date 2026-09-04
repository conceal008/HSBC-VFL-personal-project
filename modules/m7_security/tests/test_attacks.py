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
from modules.m5_modeling.components.gbdt import VerticalGBDT  # noqa: E402
from modules.m7_security.components import attacks as A  # noqa: E402
from modules.m7_security.components import malicious as MAL  # noqa: E402

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


# ————————————————— A4 特征推断 —————————————————

def test_无防护时特征被精确恢复(data):
    """辅助样本数达到特征维数即可精确反解——这是本项目最严重的单项发现。"""
    x_a, x_b, y = data
    fed = M.fit_federated_logistic(x_a, x_b, y, 200, 0.5, 1e-4, 0.0, SEED)
    out = A.feature_inference_with_auxiliary(fed.partial_b_history, x_b, DIM + 2, SEED)
    assert out["feat_r2_mean"] > PERFECT
    assert out["feat_r2_min"] > PERFECT


def test_辅助样本少于特征维数则解不出(data):
    x_a, x_b, y = data
    fed = M.fit_federated_logistic(x_a, x_b, y, 200, 0.5, 1e-4, 0.0, SEED)
    few = A.feature_inference_with_auxiliary(fed.partial_b_history, x_b, DIM - 2, SEED)
    enough = A.feature_inference_with_auxiliary(fed.partial_b_history, x_b, DIM + 2, SEED)
    assert few["feat_r2_mean"] < enough["feat_r2_mean"]


def test_上行加噪显著削弱特征推断(data):
    """与标签推断相反：特征推断依赖病态线性系统的精确解，对噪声高度敏感。"""
    x_a, x_b, y = data
    clean = M.fit_federated_logistic(x_a, x_b, y, 200, 0.5, 1e-4, 0.0, SEED)
    noisy = M.fit_federated_logistic(x_a, x_b, y, 200, 0.5, 1e-4, 0.0, SEED,
                                     uplink_sigma=1.0)
    c = A.feature_inference_with_auxiliary(clean.partial_b_history, x_b, DIM + 2, SEED)
    n = A.feature_inference_with_auxiliary(noisy.partial_b_history, x_b, DIM + 2, SEED)
    assert n["feat_r2_mean"] < c["feat_r2_mean"] - 0.5


def test_上行噪声被记录进上传量(data):
    x_a, x_b, y = data
    clean = M.fit_federated_logistic(x_a, x_b, y, 20, 0.5, 1e-4, 0.0, SEED)
    noisy = M.fit_federated_logistic(x_a, x_b, y, 20, 0.5, 1e-4, 0.0, SEED,
                                     uplink_sigma=1.0)
    assert not np.allclose(clean.partial_b_history, noisy.partial_b_history)


def test_两个噪声旋钮作用于不同方向(data):
    """下行噪声防标签推断（A1），上行噪声防特征推断（A4），二者不可互相替代。

    两者的传导方式不对称，首轮即可分辨：
    - **下行噪声**加在残差上，首轮不影响上传量（此时 w_b 仍为零向量，上传量恒为 0），
      要到下一轮才经由 w_b 的更新间接传导过去。
    - **上行噪声**加在已算出的 x_b·w_b 之上，首轮就同时影响上传量**与**残差——
      因为加噪后的部分 logit 立刻进入主动方的 logit，从而改变预测与残差。

    这条不对称性有实际含义：上行噪声会直接扰动模型自身的训练信号，
    因此它的可用性代价必须单独测量，不能沿用下行噪声的结论。
    """
    x_a, x_b, y = data
    kw = dict(n_rounds=20, lr=0.5, l2=1e-4, seed=SEED)
    base = M.fit_federated_logistic(x_a, x_b, y, dp_sigma=0.0, **kw)
    down = M.fit_federated_logistic(x_a, x_b, y, dp_sigma=1.0, **kw)
    up = M.fit_federated_logistic(x_a, x_b, y, dp_sigma=0.0, uplink_sigma=1.0, **kw)

    # 首轮上传量：下行噪声尚未传导过来，上行噪声已直接体现
    assert np.allclose(down.partial_b_history[0], base.partial_b_history[0])
    assert not np.allclose(up.partial_b_history[0], base.partial_b_history[0])
    # 首轮残差：两者都会改变，但成因不同——下行是直接加噪，上行是经 logit 传导
    assert not np.allclose(down.residual_history[0], base.residual_history[0])
    assert not np.allclose(up.residual_history[0], base.residual_history[0])
    # 两个旋钮不是同一件事
    assert not np.allclose(down.partial_b_history, up.partial_b_history)


# ————————————————— A5 成员推断 —————————————————

def test_成员推断在无记忆时接近随机(data):
    x_a, _, y = data
    n = len(y)
    tr, te = np.arange(n // 2), np.arange(n // 2, n)
    mdl = M.fit_logistic(x_a[tr], y[tr], SEED, 1.0)
    out = A.membership_inference_from_loss(mdl.decision_function(x_a[tr]), y[tr],
                                           mdl.decision_function(x_a[te]), y[te])
    assert 0.5 <= out["membership_auc"] < 0.6


def test_LiRA随模型容量单调增强(data):
    """攻击有效性验证：容量越大记忆越多，攻击必须能反映出来——
    若不能，说明攻击太弱，不足以支持任何「安全」结论。"""
    x_a, x_b, y = data
    x = np.hstack([x_a, x_b])

    def lr_fit(x_t, y_t):
        return M.fit_logistic(x_t, y_t, SEED, 1.0).decision_function

    def gbdt_fit(x_t, y_t):
        g = VerticalGBDT(200, 10, 0.3, 32, 0.0, 0.0)
        g.fit(x_t, None, y_t)
        return lambda z: g.decision_function(z, None)

    low = A.membership_inference_lira(lr_fit, x, y, 8, SEED)["membership_auc_lira"]
    high = A.membership_inference_lira(gbdt_fit, x, y, 8, SEED)["membership_auc_lira"]
    assert high > low, "高容量模型的成员泄露未高于低容量模型——攻击可能失效"


# ————————————————— A6 / A7 恶意参与方（S7.4）—————————————————

def test_恶意探针在无防护时精确还原特征(data):
    """恶意攻击者不必等自然的梯度轨迹——它自己设计探针，条件数由它掌控。"""
    _, x_b, _ = data
    n = len(x_b)
    idx = np.random.default_rng(SEED).permutation(n)
    aux, tgt = idx[:DIM], idx[DIM:DIM + 20]
    out = MAL.probe_attack(x_b, aux, tgt, 0.5, 0.0, 1, SEED)
    assert out["feat_r2_mean"] > PERFECT
    assert out["plausible_amplitude"] is True


def test_放大探针幅度可攻破上行噪声(data):
    """噪声单独无效：信噪比与幅度成正比，攻击者放大幅度即可。"""
    _, x_b, _ = data
    n = len(x_b)
    idx = np.random.default_rng(SEED).permutation(n)
    aux, tgt = idx[:DIM], idx[DIM:DIM + 20]
    legal = MAL.probe_attack(x_b, aux, tgt, 0.5, 0.1, 1, SEED, amplitude=1.0)
    huge = MAL.probe_attack(x_b, aux, tgt, 0.5, 0.1, 1, SEED, amplitude=1e6)
    assert huge["feat_r2_mean"] > legal["feat_r2_mean"]
    assert huge["plausible_amplitude"] is False, "超大幅度必须被标为不合法"


def test_重复取平均按根号衰减噪声(data):
    """k 次重复等价于一次 σ/√k 的抽样——这是等价形式的正确性检验。"""
    _, x_b, _ = data
    n = len(x_b)
    idx = np.random.default_rng(SEED).permutation(n)
    aux, tgt = idx[:DIM], idx[DIM:DIM + 10]
    one = MAL.probe_attack(x_b, aux, tgt, 0.5, 0.1, 1, SEED)
    hundred = MAL.probe_attack(x_b, aux, tgt, 0.5, 0.1, 100, SEED)
    assert hundred["effective_sigma"] < one["effective_sigma"]
    assert abs(hundred["effective_sigma"] * 10 - one["effective_sigma"]) < 1e-12
    assert hundred["n_probes"] == one["n_probes"] * 100


def test_合法性检查识别朴素探针():
    """单位探针越界或过于稀疏，都会被识破。"""
    probe = np.zeros(100)
    probe[3] = 1000.0
    assert MAL.residual_plausibility_check(probe)["suspicious"] is True
    rng = np.random.default_rng(SEED)
    real = 1.0 / (1.0 + np.exp(-rng.normal(size=100))) - (rng.random(100) < 0.5)
    assert MAL.residual_plausibility_check(real)["suspicious"] is False


def test_伪装探针绕过合法性检查(data):
    """检查单独也无效：把探针叠在真实残差上、幅度压在 1 以内即可绕过。"""
    _, x_b, _ = data
    n = len(x_b)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    aux, tgt = idx[:DIM], idx[DIM:DIM + 10]
    p = 1.0 / (1.0 + np.exp(-rng.normal(size=n)))
    baseline = p - (rng.random(n) < p).astype(float)
    out = MAL.disguised_probe_attack(x_b, aux, tgt, 0.5, baseline, 1.0)
    assert out["flagged_by_check"] == 0, "伪装探针不应被合法性检查标记"
    assert out["feat_r2_mean"] > 0.5, "伪装探针在无噪声时仍应有效"


def test_定向抬分能把目标顶进名单():
    """A7 是完整性攻击：不偷数据，操纵决策。"""
    rng = np.random.default_rng(SEED)
    score = rng.normal(size=2000)
    boost = rng.choice(2000, size=40, replace=False)
    weak = MAL.targeted_boost_attack(score, boost, 0.0, 0.1)
    strong = MAL.targeted_boost_attack(score, boost, 5.0, 0.1)
    assert strong["attacked_in_topk"] > weak["attacked_in_topk"]
    assert strong["attacked_in_topk"] == strong["target_size"]
    assert weak["list_churn"] == 0.0, "零幅度不应改变名单"


def test_中等幅度抬分不触发宽松的稳定性阈值():
    """这正是把阈值由 0.9 收紧到 0.95 的理由。"""
    rng = np.random.default_rng(SEED)
    score = rng.normal(size=4000)
    boost = rng.choice(4000, size=80, replace=False)
    r = MAL.targeted_boost_attack(score, boost, 1.0, 0.1)
    overlap = 1.0 - r["list_churn"]
    assert r["attacked_in_topk"] > r["baseline_in_topk"], "中等幅度应能顶进目标"
    assert overlap > 0.9, "重合度仍高于旧阈值 0.9——旧阈值抓不到这种操纵"
