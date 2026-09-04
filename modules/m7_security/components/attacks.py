"""M7 · 隐私攻击与防护评估。

三类攻击，对应三类协议暴露面：
- A1 标签推断：被动方从每轮回传的残差反推主动方的标签
- A2 嵌入反演：主动方从被动方上传的嵌入重建被动方原始特征
- A3 梯度反演：被动方从回传梯度反推标签（仅 SplitNN 形态A 存在此暴露面）
- A4 特征推断：主动方从上传的部分 logit 反推被动方原始特征（需少量辅助样本）
- A5 成员推断：从逐样本损失判断某条记录是否在训练集内

所有攻击都假设攻击者是**诚实但好奇**的协议内参与方，不假设外部窃听。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

RIDGE_EPS = 1e-6
HALF = 0.5
GAUSSIAN_MECH_CONST = 1.25   # 高斯机制 (ε,δ)-DP 标准式中的常数项
LOGIT_CLIP_ATTACK = 30.0     # 计算逐样本损失时的 logit 截断，防溢出
MIN_SHADOW_PER_SIDE = 3      # 逐样本校准所需的最少影子模型数（每侧）


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


def feature_inference_with_auxiliary(partial_b_history: np.ndarray,
                                     x_b: np.ndarray, n_aux: int,
                                     seed: int) -> Dict:
    """A4：主动方从每轮上传的部分 logit 反推被动方**原始特征**。

    威胁模型：主动方另有少量样本的 x_b 真值（辅助集）——
    这在现实中并不苛刻，例如双方共有的存量客户中，主动方已自行采集过对方口径的字段，
    或历史合作/公开渠道泄露过少量样本。

    攻击分两步，都是最小二乘：
    1. 用辅助集的 (x_b, partial_b) 解出每轮的权重 w_b（需 n_aux ≥ 特征维数）；
    2. 用解出的 w_b 反解其余样本的 x_b（需 w_b 轨迹张成整个特征空间）。

    ⚠️ 关键在于第 2 步的**条件数**：梯度下降的 w_b 轨迹虽然满秩，
    但奇异值跨度可达 5 个数量级，因此该攻击对上行噪声高度敏感——
    这与标签推断攻击（信号轮间恒定、可平均消噪）的性质完全相反。
    """
    n, dim = x_b.shape
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    aux, target = idx[:n_aux], idx[n_aux:]
    p = partial_b_history.T                       # n × T：每个样本在各轮的观测

    w_hat = np.linalg.lstsq(x_b[aux], p[aux], rcond=None)[0]        # dim × T
    x_hat = np.linalg.lstsq(w_hat.T, p[target].T, rcond=None)[0].T  # |target| × dim

    truth = x_b[target]
    ss_res = ((truth - x_hat) ** 2).sum(axis=0)
    ss_tot = ((truth - truth.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, RIDGE_EPS)
    cond = float(np.linalg.cond(w_hat.T))
    return {"feat_r2_mean": float(r2.mean()), "feat_r2_min": float(r2.min()),
            "feat_r2_per_dim": [float(v) for v in r2],
            "n_aux": int(n_aux), "dim_b": int(dim), "condition_number": cond}


def membership_inference_from_loss(score_train: np.ndarray, y_train: np.ndarray,
                                   score_test: np.ndarray,
                                   y_test: np.ndarray) -> Dict:
    """A5：从逐样本损失判断某条记录**是否在训练集里**。

    模型对训练样本的损失系统性地低于未见样本，攻击者据此排序即可。
    报告攻击 AUC：0.5 表示无法区分，越高表示成员关系泄露越严重。

    这类攻击在合规上尤其要紧：**「这个人是否在你的建模样本里」本身就是个人信息**，
    即便特征与标签都没泄露。
    """
    def nll(score: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-np.clip(score, -LOGIT_CLIP_ATTACK, LOGIT_CLIP_ATTACK)))
        p = np.clip(p, RIDGE_EPS, 1.0 - RIDGE_EPS)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    loss = np.concatenate([nll(score_train, y_train), nll(score_test, y_test)])
    member = np.concatenate([np.ones(len(y_train)), np.zeros(len(y_test))])
    auc = roc_auc_score(member, -loss)            # 损失越低越像成员
    return {"membership_auc": float(max(auc, 1.0 - auc)),
            "loss_gap": float(nll(score_test, y_test).mean()
                              - nll(score_train, y_train).mean())}


def membership_inference_lira(fit_score_fn, x: np.ndarray, y: np.ndarray,
                              n_shadow: int, seed: int) -> Dict:
    """A5 加强版：影子模型校准的成员推断（LiRA 思路的轻量实现）。

    朴素的损失阈值攻击把「样本难不难」和「样本在不在训练集里」混在一起，
    因此即便模型训练 AUC 打到 1.0 也只能勉强超过随机——**它弱到不足以支持任何结论**。

    LiRA 的做法是**逐样本校准**：对每个样本，分别统计「它在训练集内」与
    「它不在训练集内」时模型给出的置信度分布，再看目标模型的置信度更像哪一边。

    `fit_score_fn(x_train, y_train) -> callable(x) -> score`
    使得本攻击可施加于阶梯上的任意级别。
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    half = n // 2
    conf_in: List[List[float]] = [[] for _ in range(n)]
    conf_out: List[List[float]] = [[] for _ in range(n)]

    for k in range(n_shadow):
        idx = rng.permutation(n)
        tr, out = idx[:half], idx[half:]
        score = fit_score_fn(x[tr], y[tr])(x)
        p = 1.0 / (1.0 + np.exp(-np.clip(score, -LOGIT_CLIP_ATTACK, LOGIT_CLIP_ATTACK)))
        p_true = np.where(y == 1, p, 1.0 - p)
        p_true = np.clip(p_true, RIDGE_EPS, 1.0 - RIDGE_EPS)
        logit_conf = np.log(p_true / (1.0 - p_true))     # LiRA 的 logit 缩放
        for i in tr:
            conf_in[i].append(logit_conf[i])
        for i in out:
            conf_out[i].append(logit_conf[i])

    # 目标模型：用前一半样本训练，攻击者要判断每个样本是否属于这一半
    target_idx = rng.permutation(n)
    member = np.zeros(n)
    member[target_idx[:half]] = 1.0
    target_score = fit_score_fn(x[target_idx[:half]], y[target_idx[:half]])(x)
    p = 1.0 / (1.0 + np.exp(-np.clip(target_score, -LOGIT_CLIP_ATTACK, LOGIT_CLIP_ATTACK)))
    p_true = np.clip(np.where(y == 1, p, 1.0 - p), RIDGE_EPS, 1.0 - RIDGE_EPS)
    observed = np.log(p_true / (1.0 - p_true))

    stat = np.zeros(n)
    for i in range(n):
        if len(conf_out[i]) < MIN_SHADOW_PER_SIDE or len(conf_in[i]) < MIN_SHADOW_PER_SIDE:
            continue
        mu_out, sd_out = np.mean(conf_out[i]), np.std(conf_out[i]) + RIDGE_EPS
        stat[i] = (observed[i] - mu_out) / sd_out        # 比「非成员」分布高多少个标准差
    auc = roc_auc_score(member, stat)
    return {"membership_auc_lira": float(max(auc, 1.0 - auc)),
            "n_shadow": int(n_shadow)}
