# -*- coding: utf-8 -*-
"""M7 · 恶意（非半诚实）参与方的攻防。

此前全部攻防结论的前提都是**半诚实**：参与方严格执行协议，只是好奇。
本模块放开这一条——攻击者可以**偏离协议、发送构造的值**。

这不是理论上的严苛化。半诚实假设在真实场景中往往不成立：
协议由代码执行，而代码由某一方控制；对方无从验证收到的数字是不是真的按协议算出来的。

两类攻击对应两种目的：

- **A6 恶意主动方 · 探针攻击**（机密性）：不发真实残差，改发构造的探针向量，
  直接把被动方的原始特征"问"出来。
- **A7 恶意被动方 · 定向抬分**（完整性）：不发真实的部分 logit，改发构造值，
  把指定人群顶到名单前列。这不是偷数据，是**操纵决策**。
- **A8 多次建模的信息累积**（最现实的一类「合谋」）：周期性重训是协议要求的正常行为，
  而每次重训都给攻击者一批全新的独立观测。
- **A9 模型窃取**：被动方由已证的 A1 反解出主动方的逐样本打分，
  进而**独立复现整份名单**——资产转移，而不只是信息泄露。

以及一个防御维度：**可检测性**——诚实方能否看出对方偏离了协议。
"""
from __future__ import annotations

from typing import Dict

import numpy as np

RIDGE_EPS = 1e-9
RESIDUAL_LO = -1.0          # 合法残差 r = p − y 的取值下界
RESIDUAL_HI = 1.0           # 上界；越界即为协议偏离的直接证据
PLAUSIBLE_TOL = 1e-9
DENSE_RATIO_FLOOR = 0.5     # 真实残差的非零比例下限；低于此即形似探针
HALF = 0.5                  # 标签推断的方向判定阈值


def probe_attack(x_b: np.ndarray, aux_idx: np.ndarray, target_idx: np.ndarray,
                 lr: float, uplink_sigma: float, n_repeat: int,
                 seed: int, amplitude: float = RESIDUAL_HI) -> Dict:
    """A6：恶意主动方用单位探针直接读出内积，再借辅助集反解目标样本的特征。

    机制：被动方按协议更新 `w_b -= lr·(x_bᵀ·r)/n`。
    若主动方送的 r 是第 j 个样本上的单位向量，则更新量正比于 `x_b[j]`，
    下一轮上传的 `x_b·w_b` 之差直接给出 `x_b·x_b[j]`——即 Gram 矩阵的第 j 列。
    有了 d 个已知真值的辅助样本，每个目标只需 **1 次探针**即可精确反解。

    **与半诚实版本的两点关键差别**：
    1. 半诚实攻击只能利用梯度下降自然形成的 w_b 轨迹，那条轨迹病态
       （条件数约 5×10⁴）因而对噪声敏感；恶意攻击者**自己设计探针**，条件数由它掌控。
    2. 恶意攻击者可以**放大探针幅度**。单次探针的信号量级是 `lr·amplitude/n`，
       而噪声是固定的 σ——**信噪比因而与 amplitude 成正比**。
       并且它可以**重复同一探针取平均**，使噪声再按 1/√k 衰减。

    → **上行加噪能否防住恶意攻击者，完全取决于 amplitude 有没有被约束。**
      约束它的唯一手段是被动方对收到的残差做合法性检查
      （真实残差 `r = sigmoid(logit) − y` 必落在 [−1, 1]）。
      **没有这项检查，加多大的噪声都没有意义**——攻击者把幅度调大即可。
    """
    n = len(x_b)
    rng = np.random.default_rng(seed)
    cols: Dict[int, np.ndarray] = {}
    probes_used = 0

    # 重复 k 次取平均，等价于一次 σ/√k 的抽样——用等价形式而非暴力循环，
    # 既数学上精确，也让 k 取到 10⁶ 量级仍可计算。
    # 两次上传各带一次噪声，故差值的噪声标准差先乘 √2。
    effective_sigma = (uplink_sigma * np.sqrt(2.0) / np.sqrt(n_repeat)
                       if uplink_sigma > 0 else 0.0)
    for j in list(target_idx) + list(aux_idx):
        probe = np.zeros(n)
        probe[j] = amplitude
        delta_w = -lr * (x_b.T @ probe / n)       # 被动方照协议更新
        observed = x_b @ delta_w                  # 主动方观测上传量之差
        if effective_sigma > 0:
            observed = observed + rng.normal(0.0, effective_sigma, size=n)
        probes_used += n_repeat
        cols[j] = observed * (-n / (lr * amplitude))

    gram_target = np.column_stack([cols[j] for j in target_idx])
    x_hat = np.linalg.lstsq(x_b[aux_idx], gram_target[aux_idx, :], rcond=None)[0].T

    truth = x_b[target_idx]
    ss_res = ((truth - x_hat) ** 2).sum(axis=0)
    ss_tot = ((truth - truth.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, RIDGE_EPS)
    return {"feat_r2_mean": float(r2.mean()), "feat_r2_min": float(r2.min()),
            "n_probes": probes_used, "n_repeat": int(n_repeat),
            "amplitude": float(amplitude),
            "effective_sigma": float(effective_sigma),
            "plausible_amplitude": bool(abs(amplitude) <= RESIDUAL_HI),
            "n_aux": int(len(aux_idx)), "n_target": int(len(target_idx))}


def residual_plausibility_check(received: np.ndarray) -> Dict:
    """防御：被动方对收到的残差做合法性检查。

    真实残差 `r = sigmoid(logit) − y` 必然落在 [−1, 1]，
    且不可能是稀疏的单位向量——那不是任何真实标签配置能产生的形状。
    本检查给出两个信号：越界比例与稀疏度。

    ⚠️ 它拦得住**朴素**的探针（单位向量、越界值），但拦不住把探针伪装成
    合法残差形状的攻击者：把探针缩放到 [−1,1] 并叠加一个真实残差的基底即可。
    详见 `disguised_probe_attack`。**这条防御的价值是抬高门槛，不是封死。**
    """
    out_of_range = float(np.mean((received < RESIDUAL_LO - PLAUSIBLE_TOL)
                                 | (received > RESIDUAL_HI + PLAUSIBLE_TOL)))
    nonzero = float(np.mean(np.abs(received) > PLAUSIBLE_TOL))
    return {"out_of_range_ratio": out_of_range, "nonzero_ratio": nonzero,
            "suspicious": out_of_range > 0.0 or nonzero < DENSE_RATIO_FLOOR}


def disguised_probe_attack(x_b: np.ndarray, aux_idx: np.ndarray,
                           target_idx: np.ndarray, lr: float,
                           baseline_residual: np.ndarray,
                           amplitude: float) -> Dict:
    """伪装探针：把探针叠加在真实残差上并缩到合法幅度，绕过合法性检查。

    攻击者送 `r_disguised = r_true + amplitude·e_j`，只要幅度够小就落在 [−1,1] 内、
    且非零比例与真实残差一致——`residual_plausibility_check` 无法区分。
    代价是信号变弱，需要更多重复来把它从 r_true 的影响中分离出来。
    """
    n = len(x_b)
    cols: Dict[int, np.ndarray] = {}
    flagged = 0
    for j in list(target_idx) + list(aux_idx):
        probe = baseline_residual.copy()
        probe[j] = np.clip(probe[j] + amplitude, RESIDUAL_LO, RESIDUAL_HI)
        check = residual_plausibility_check(probe)
        flagged += int(check["suspicious"])
        # 攻击者对照送一次纯 baseline，两次之差即为探针的净效果
        delta = -lr * (x_b.T @ (probe - baseline_residual) / n)
        cols[j] = (x_b @ delta) * (-n / lr)
    gram = np.column_stack([cols[j] for j in target_idx])
    x_hat = np.linalg.lstsq(x_b[aux_idx], gram[aux_idx, :], rcond=None)[0].T
    truth = x_b[target_idx]
    ss_res = ((truth - x_hat) ** 2).sum(axis=0)
    ss_tot = ((truth - truth.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, RIDGE_EPS)
    return {"feat_r2_mean": float(r2.mean()), "flagged_by_check": int(flagged),
            "n_sent": int(len(target_idx) + len(aux_idx)), "amplitude": float(amplitude)}


def targeted_boost_attack(score_a: np.ndarray, boost_idx: np.ndarray,
                          amplitude: float, top_k: float) -> Dict:
    """A7：恶意被动方送构造的部分 logit，把指定人群顶进名单前列。

    这不是偷数据，是**操纵决策**——被动方可以让自己想推的客户进入营销名单。
    在纵向联邦里，被动方送的部分 logit 直接加进最终打分，主动方无从验证它是不是
    真的由 `x_b·w_b` 算出来的。
    """
    n = len(score_a)
    k = max(int(n * top_k), 1)
    baseline_top = set(np.argsort(-score_a)[:k].tolist())
    baseline_hits = len(baseline_top & set(boost_idx.tolist()))

    tampered = score_a.copy()
    tampered[boost_idx] += amplitude
    attacked_top = set(np.argsort(-tampered)[:k].tolist())
    attacked_hits = len(attacked_top & set(boost_idx.tolist()))

    return {"amplitude": float(amplitude), "top_k": float(top_k),
            "baseline_in_topk": baseline_hits, "attacked_in_topk": attacked_hits,
            "target_size": int(len(boost_idx)),
            "list_churn": 1.0 - len(baseline_top & attacked_top) / k}


def model_stealing_attack(residual_history: np.ndarray, partial_b: np.ndarray,
                          a_contrib_true: np.ndarray, y_true: np.ndarray,
                          top_k: float, use_round_average: bool) -> Dict:
    """A9：被动方窃取主动方的模型输出，进而独立复现整份名单。

    推导链条完全建立在已证的 A1 之上，不需要任何新假设：
      1. A1 已证被动方能从残差完全推断标签 y；
      2. 有了 y，由 `r = sigmoid(logit) − y` 反解出 logit；
      3. 被动方自己算得出 `partial_b = x_b·w_b`；
      4. 相减即得**主动方的全部贡献** `x_a·w_a + bias`——逐样本。

    后果不是「偷到几个参数」，而是**被动方可以不依赖主动方独立出名单**：
    它已经掌握了最终打分的每一项。这是商业价值的直接转移，
    也解释了为什么「原始特征没出域」并不等于「资产没被拿走」。

    `use_round_average=True` 时攻击者跨轮平均残差再推断标签——
    这正是 S7.2 用来击穿噪声防护的那一手。

    ⚠️ **阈值的选取会决定攻击强弱，不能想当然取 0。** 跨轮平均后的残差不再以 ±0.5
    为中心，用 0 切分会把一个排序完美（AUC=1.0）的信号切得很差。
    攻击者知道大致的正样本率（营销场景下这是公开常识），按该比例取分位数即可。
    最初版本正是因为固定阈值 0 而得出「噪声挡住了模型窃取」的错误结论。
    """
    if use_round_average:
        basis = residual_history.mean(axis=0)
    else:
        basis = residual_history[-1]
    # 残差与标签负相关：正样本的残差更小。按已知正样本率取分位切分。
    base_rate = float(np.mean(y_true))
    cutoff = np.quantile(basis, base_rate)
    y_hat = (basis <= cutoff).astype(int)
    label_acc = float(max((y_hat == y_true).mean(), (y_hat != y_true).mean()))
    if (y_hat == y_true).mean() < HALF:
        y_hat = 1 - y_hat                      # 方向无关：攻击者翻转符号即可

    p_hat = np.clip(residual_history[-1] + y_hat, RIDGE_EPS, 1.0 - RIDGE_EPS)
    logit_hat = np.log(p_hat / (1.0 - p_hat))
    a_hat = logit_hat - partial_b

    denom = np.std(a_contrib_true) * np.std(a_hat)
    corr = float(np.mean((a_hat - a_hat.mean()) * (a_contrib_true - a_contrib_true.mean()))
                 / denom) if denom > RIDGE_EPS else 0.0

    n = len(a_hat)
    k = max(int(n * top_k), 1)
    true_list = set(np.argsort(-(a_contrib_true + partial_b))[:k].tolist())
    stolen_list = set(np.argsort(-(a_hat + partial_b))[:k].tolist())
    return {"label_accuracy": label_acc, "score_correlation": corr,
            "topk_overlap": len(true_list & stolen_list) / k,
            "base_rate_used": base_rate,
            "round_averaged": bool(use_round_average)}


def multi_run_accumulation(x_b: np.ndarray, aux_idx: np.ndarray,
                           target_idx: np.ndarray, lr: float,
                           uplink_sigma: float, n_runs: int,
                           amplitude: float, seed: int) -> Dict:
    """A8：多次建模的信息累积——最现实的一类「合谋」。

    生产系统本来就要周期性重训。**每次重训都给攻击者一批全新的独立观测**，
    而防护噪声是每次独立抽的。攻击者把各次的观测合起来，噪声按 1/√N 衰减。

    这与 S7.4 的「重复探针」不同：重复探针在**一次**建模内做，
    会在流量上留下明显异常；而多次建模是**协议本身要求的正常行为**，
    攻击者什么额外动作都不用做——它只需要有耐心。

    → 防护强度必须按**模型生命周期内的累计重训次数**折算，而不是按单次。
    """
    per_run = []
    for run in range(n_runs):
        out = probe_attack(x_b, aux_idx, target_idx, lr, uplink_sigma, 1,
                           seed + run, amplitude=amplitude)
        per_run.append(out["feat_r2_mean"])
    combined = probe_attack(x_b, aux_idx, target_idx, lr, uplink_sigma, n_runs,
                            seed, amplitude=amplitude)
    return {"n_runs": int(n_runs),
            "single_run_r2": float(np.mean(per_run)),
            "accumulated_r2": float(combined["feat_r2_mean"]),
            "effective_sigma": float(combined["effective_sigma"]),
            "amplitude": float(amplitude)}
