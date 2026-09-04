"""M5 · 五级基线阶梯 L0–L4 与三类纵向联邦算法。

阶梯的意义（框架 M5）：核心不是「用哪个模型」，而是「和谁比」。
**L1 是 VFL 的真正竞争对手**——绝大多数 VFL 研究只比 L0 与 L3，因而系统性高估 VFL。

| 级别 | 方案 | 跨境的是什么 |
|---|---|---|
| L0 | 内地单方模型 | 无 |
| L1 | 内地模型 + 香港 k-匿名聚合统计 | 分段统计量（k-匿名） |
| L2 | 内地模型 + 香港粗粒度标记 | 每客户 1 个序数标记 |
| L3a | 联邦逻辑回归 | 残差 ↔ 部分 logit |
| L3b | 纵向 GBDT（SecureBoost 式） | (g,h) ↔ 分桶直方图 |
| L3c | SplitNN | 表示 ↔ 表示梯度 |
| L4 | 集中式 | （不可行，仅作上界） |
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from .nn import Adam, Encoder, TopModel, sigmoid

LR_MAX_ITER = 2000
PROB_CLIP = 1e-6
TERTILE_LO = 1.0 / 3.0
TERTILE_HI = 2.0 / 3.0
FLOATS_PER_SEGMENT_STAT = 1


# ————————————————————————— L0 / L4 —————————————————————————

def fit_logistic(x: np.ndarray, y: np.ndarray, seed: int, c_reg: float) -> LogisticRegression:
    model = LogisticRegression(max_iter=LR_MAX_ITER, C=c_reg, random_state=seed)
    model.fit(x, y)
    return model


# ————————————————————————— L1：k-匿名聚合统计 —————————————————————————

def build_segments(x_a: np.ndarray, n_segments: int, seed: int) -> np.ndarray:
    """由内地侧特征划分客户分段。分段定义只用 A 的特征，因此 A 可独立计算。"""
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((x_a.shape[1], 2))
    coords = x_a @ proj
    side = int(np.sqrt(n_segments))
    seg = np.zeros(len(x_a), dtype=int)
    for dim in range(2):
        edges = np.quantile(coords[:, dim], np.linspace(0.0, 1.0, side + 1)[1:-1])
        seg = seg * side + np.searchsorted(edges, coords[:, dim])
    return seg


def k_anonymous_segment_stats(seg: np.ndarray, x_b: np.ndarray,
                              k: int) -> Tuple[np.ndarray, Dict[str, int]]:
    """香港侧按分段返回特征均值；成员数 < k 的分段被抑制（回落到全局均值）。

    这是 L1 的全部跨境内容：**分段统计量，不含任何个体记录**。
    """
    global_mean = x_b.mean(axis=0)
    out = np.tile(global_mean, (len(seg), 1))
    suppressed = 0
    released = 0
    for s in np.unique(seg):
        mask = seg == s
        if mask.sum() >= k:
            out[mask] = x_b[mask].mean(axis=0)
            released += 1
        else:
            suppressed += 1
    stats = {
        "segments_total": int(len(np.unique(seg))),
        "segments_released": released,
        "segments_suppressed": suppressed,
        "comm_floats": released * x_b.shape[1] * FLOATS_PER_SEGMENT_STAT,
    }
    return out, stats


# ————————————————————————— L2：粗粒度标记 —————————————————————————

def coarse_flag(x_b: np.ndarray, seed: int) -> np.ndarray:
    """香港侧对每个客户返回一个三档序数标记（低/中/高）。

    比 L1 泄露更多（逐客户），但仍远少于原始特征——用于定位「信息量来自哪里」。
    """
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(x_b.shape[1])
    score = x_b @ direction
    lo, hi = np.quantile(score, [TERTILE_LO, TERTILE_HI])
    return np.digitize(score, [lo, hi]).astype(float)[:, None]


# ————————————————————————— L3a：联邦逻辑回归 —————————————————————————

@dataclass
class FederatedLRResult:
    w_a: np.ndarray
    w_b: np.ndarray
    bias: float
    comm: Dict[str, int]
    residual_history: np.ndarray   # 每轮**实际下发给被动方**的残差，供 M7 标签推断攻击使用
    partial_b_history: np.ndarray  # 每轮**实际上传给主动方**的部分 logit，供 M7 特征推断攻击使用


def fit_federated_logistic(x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray,
                           n_rounds: int, lr: float, l2: float,
                           dp_sigma: float, seed: int,
                           uplink_sigma: float = 0.0) -> FederatedLRResult:
    """纵向联邦逻辑回归。

    协议：主动方算 logit = X_A·w_A + X_B·w_B + b（需要被动方送来部分 logit），
    再算残差 r = sigmoid(logit) − y 发给被动方；被动方用 r 更新自己的 w_B。
    **被动方全程看不到标签，主动方全程看不到 X_B。**

    两个噪声旋钮对应两个方向、两类攻击：
    - `dp_sigma`：主动方对**下发的残差**加噪，防的是被动方推断标签（攻击 A1）。
    - `uplink_sigma`：被动方对**上传的部分 logit**加噪，防的是主动方推断特征（攻击 A4）。
      两者是不同参与方的自我保护，不可互相替代。
    数学上等价于集中式逻辑回归（dp_sigma=0 时），因此可与 L4 对照检验实现是否正确。
    """
    rng = np.random.default_rng(seed)
    w_a = np.zeros(x_a.shape[1])
    w_b = np.zeros(x_b.shape[1])
    bias = 0.0
    n = len(y)
    residuals = np.zeros((n_rounds, n))
    partials = np.zeros((n_rounds, n))
    floats = 0

    for t in range(n_rounds):
        partial_b = x_b @ w_b                    # 被动方本地计算
        if uplink_sigma > 0:
            partial_b = partial_b + rng.normal(0.0, uplink_sigma, size=n)
        partials[t] = partial_b                  # 记录**实际上传**的值
        logit = x_a @ w_a + partial_b + bias
        r = sigmoid(logit) - y                   # 主动方本地计算
        r_sent = r + rng.normal(0.0, dp_sigma, size=n) if dp_sigma > 0 else r
        residuals[t] = r_sent                    # 记录**实际下发**的值，攻击只能看到它

        w_a -= lr * (x_a.T @ r / n + l2 * w_a)
        bias -= lr * r.mean()
        w_b -= lr * (x_b.T @ r_sent / n + l2 * w_b)   # 被动方本地更新
        floats += n * 2                          # 一来一回各 n 个浮点

    return FederatedLRResult(w_a, w_b, bias,
                             {"comm_rounds": n_rounds, "comm_floats": floats},
                             residuals, partials)


def federated_lr_score(res: FederatedLRResult, x_a: np.ndarray, x_b: np.ndarray) -> np.ndarray:
    return x_a @ res.w_a + x_b @ res.w_b + res.bias


# ————————————————————————— L3c：SplitNN —————————————————————————

@dataclass
class SplitNNResult:
    score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    comm: Dict[str, int]
    embedding_history: np.ndarray     # 被动方送出的表示（供 M7 反演攻击）
    grad_history: np.ndarray          # 回传给被动方的梯度（形态 B 下全为 0）
    embed_index: np.ndarray           # 上两者对应的训练集行号——攻击必须按它对齐
    encode_b: Callable[[np.ndarray], np.ndarray]   # 被动方最终编码器，攻击者可对任意样本求表示


def fit_splitnn(x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray,
                mode: str, dim_hidden: int, dim_embed: int,
                n_epochs: int, batch_size: int, lr: float,
                seed: int) -> SplitNNResult:
    """分割学习。三种形态：

    - `bidirectional`（**形态 A**）：被动方编码器随标签迭代训练，主动方**回传梯度**。
      合规上最重——梯度 cn→hk 属向境外提供个人信息。
    - `frozen_random`（**形态 B-随机**）：被动方编码器随机初始化后**冻结**，不回传梯度。
    - `frozen_pca`（**形态 B-自监督**）：被动方用自己的数据做 PCA 预训练后冻结，不回传梯度。
      对应框架里「用本地非对齐样本自监督预训练改善编码器表示」的思路。

    形态 B 是 M0 的 S0.6 提出、但当时无法回答的问题：
    **不回传梯度，表示质量会损失多少？** 本函数就是用来回答它的。
    """
    rng = np.random.default_rng(seed)
    enc_a = Encoder(x_a.shape[1], dim_hidden, dim_embed, rng)
    enc_b = Encoder(x_b.shape[1], dim_hidden, dim_embed, rng)

    if mode == "frozen_pca":
        # 用被动方自己的数据做主成分投影，作为「自监督预训练」的极简替身
        xb_centered = x_b - x_b.mean(axis=0)
        _, _, vt = np.linalg.svd(xb_centered, full_matrices=False)
        n_comp = min(dim_hidden, vt.shape[0])      # 主成分数不能超过特征维数
        w1 = np.zeros((x_b.shape[1], dim_hidden))
        w1[:, :n_comp] = vt[:n_comp].T
        enc_b.p["w1"] = w1
        enc_b.p["b1"] = np.zeros(dim_hidden)
        enc_b.p["w2"] = np.eye(dim_hidden, dim_embed)
        enc_b.p["b2"] = np.zeros(dim_embed)

    train_b = mode == "bidirectional"
    top = TopModel(dim_embed * 2, rng)
    opt_a = Adam(enc_a.p, lr)
    opt_top = Adam(top.p, lr)
    opt_b = Adam(enc_b.p, lr) if train_b else None

    n = len(y)
    n_batches = max(n // batch_size, 1)
    embed_log = np.zeros((0, dim_embed))
    grad_log = np.zeros((0, dim_embed))
    idx_log = np.zeros(0, dtype=int)
    floats = 0

    for epoch in range(n_epochs):
        order = rng.permutation(n)
        for bi in range(n_batches):
            idx = order[bi * batch_size:(bi + 1) * batch_size]
            if len(idx) == 0:
                continue
            h_a = enc_a.forward(x_a[idx])
            h_b = enc_b.forward(x_b[idx])          # 被动方 → 主动方：表示
            logit = top.forward(np.concatenate([h_a, h_b], axis=1))
            grad_logit = sigmoid(logit) - y[idx]

            g_top, g_h = top.backward(grad_logit)
            opt_top.step(top.p, g_top)
            opt_a.step(enc_a.p, enc_a.backward(g_h[:, :dim_embed]))

            g_h_b = g_h[:, dim_embed:]
            floats += len(idx) * dim_embed         # 表示上行
            if train_b:
                opt_b.step(enc_b.p, enc_b.backward(g_h_b))
                floats += len(idx) * dim_embed     # 梯度下行（形态 A 独有）

            if epoch == n_epochs - 1:
                embed_log = np.vstack([embed_log, h_b])
                grad_log = np.vstack([grad_log, g_h_b if train_b else np.zeros_like(g_h_b)])
                idx_log = np.concatenate([idx_log, idx])

    def score_fn(xa_new: np.ndarray, xb_new: np.ndarray) -> np.ndarray:
        h = np.concatenate([enc_a.forward(xa_new), enc_b.forward(xb_new)], axis=1)
        return top.forward(h)

    return SplitNNResult(score_fn,
                         {"comm_rounds": n_epochs * n_batches,
                          "comm_floats": floats,
                          "gradient_returned": int(train_b)},
                         embed_log, grad_log, idx_log, enc_b.forward)
