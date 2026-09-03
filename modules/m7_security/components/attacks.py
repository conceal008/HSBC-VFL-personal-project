"""M7 · 隐私攻击与防护评估。

三类攻击，对应三类协议暴露面：
- A1 标签推断：被动方从每轮回传的残差反推主动方的标签
- A2 嵌入反演：主动方从被动方上传的嵌入重建被动方原始特征
- A3 梯度反演：被动方从回传梯度反推标签（仅 SplitNN 形态A 存在此暴露面）

所有攻击都假设攻击者是**诚实但好奇**的协议内参与方，不假设外部窃听。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

RIDGE_EPS = 1e-6
HALF = 0.5
GAUSSIAN_MECH_CONST = 1.25   # 高斯机制 (ε,δ)-DP 标准式中的常数项


def label_inference_from_residuals(residual_history: np.ndarray,
                                   y_true: np.ndarray) -> Dict:
    """A1：被动方每轮都收到残差 r = y − p，用它直接排序即可推断标签。

    第 1 轮尤其致命：此时权重为 0、预测恒为 0.5，残差 r = y − 0.5，
    正负号与标签**一一对应**，泄露是完全的（LeakAUC = 1.0）。
    """
    def leak(r: np.ndarray) -> float:
        # 残差 r = p − y 与标签**负相关**，攻击者翻转符号即可；
        # 因此泄露强度是 max(auc, 1−auc)，而非原始 auc。
        a = roc_auc_score(y_true, r)
        return float(max(a, 1.0 - a))

    out = {}
    for tag, idx in [("首轮", 0), ("中轮", len(residual_history) // 2),
                     ("末轮", len(residual_history) - 1)]:
        out[f"leak_auc_{tag}"] = leak(residual_history[idx])
    out["leak_auc_最优轮"] = float(max(leak(r) for r in residual_history))
    return out


def gaussian_epsilon_per_round(sigma: float, delta: float) -> float:
    """单轮高斯机制的 ε（灵敏度取 1：翻转一个标签使该样本残差至多变动 1）。

    ⚠️ 这是**单轮**预算。本实现未做子采样放大与 RDP 会计，
    400 轮朴素合成后的总预算不具备可解释的隐私含义——见 M7 报告的降级说明。
    """
    if sigma <= 0:
        return float("inf")
    return float(np.sqrt(2.0 * np.log(GAUSSIAN_MECH_CONST / delta)) / sigma)


def embedding_inversion(embedding_history: np.ndarray, embed_index: np.ndarray,
                        x_b: np.ndarray, seed: int, train_frac: float) -> Dict:
    """A2：主动方拿到被动方嵌入 h_b，训练一个线性解码器重建 x_b。

    `embed_index` 是每条嵌入对应的训练集行号——嵌入按小批次乱序记录，
    不按行号对齐会得到毫无意义的 R²。报告逐维 R²，越高说明嵌入越接近可逆编码。
    """
    h = embedding_history
    x_b = x_b[embed_index]
    n = len(h)
    rng = np.random.default_rng(seed)
    o = rng.permutation(n)
    tr, te = o[:int(n * train_frac)], o[int(n * train_frac):]
    hb = np.hstack([h, np.ones((n, 1))])
    # 岭回归闭式解：攻击者用已知的少量样本对（辅助集）拟合解码器
    a = hb[tr].T @ hb[tr] + RIDGE_EPS * np.eye(hb.shape[1])
    w = np.linalg.solve(a, hb[tr].T @ x_b[tr])
    pred = hb[te] @ w
    ss_res = ((x_b[te] - pred) ** 2).sum(axis=0)
    ss_tot = ((x_b[te] - x_b[te].mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, RIDGE_EPS)
    return {"inv_r2_mean": float(r2.mean()), "inv_r2_max": float(r2.max()),
            "inv_r2_per_dim": [float(v) for v in r2]}


def label_inference_from_gradients(grad_history: np.ndarray,
                                   embed_index: np.ndarray,
                                   y_true: np.ndarray) -> Dict:
    """A3：SplitNN 形态A 中，被动方收到 ∂L/∂h_b。

    对二分类交叉熵，该梯度可写成 (p − y) · ∂score/∂h_b，标签只影响标量因子
    (p − y) 的**符号**。因此梯度向量在样本间的方向会按标签聚成两簇，
    攻击者对梯度做一维投影即可分离。此处用梯度与其均值方向的内积作为打分。
    """
    g = grad_history
    y_true = y_true[embed_index]
    direction = g.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm < RIDGE_EPS:
        return {"leak_auc_梯度方向": HALF}
    proj = g @ (direction / norm)
    auc = roc_auc_score(y_true, proj)
    return {"leak_auc_梯度方向": float(max(auc, 1.0 - auc))}
