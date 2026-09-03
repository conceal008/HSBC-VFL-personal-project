"""M3 · 实体对齐（PSI）仿真。

本模块**不实现**真实密码学 PSI，只仿真其外部可观测行为：
- 对齐结果的规模与偏差
- 匹配错误（错配 / 漏配）对下游模型的影响
- 通信与计算量的数量级

降级说明（W-002）：真实 PSI 需要 ECDH-PSI 或 OT 扩展等密码学原语，
本阶段用「带噪声的确定性匹配」替代，只保证**下游影响**这一层可信。
"""
from __future__ import annotations

from typing import Dict

import numpy as np

BYTES_PER_HASH = 32          # SHA-256 输出长度
ROUNDS_ECDH_PSI = 2          # ECDH-PSI 的典型交互轮数
BYTES_PER_MB = 1024 * 1024   # 通信量换算
EXPONENTIATIONS_PER_ITEM = 2 # 每条 ID 双方各做一次模幂


def simulate_psi(in_overlap: np.ndarray, mismatch: np.ndarray) -> Dict:
    """给定真实交集标记与错配标记，给出 PSI 的可观测输出。

    `matched` 是协议**认为**匹配上的集合；它与真实交集的差就是错配。
    """
    matched = in_overlap.copy()
    flip = mismatch & in_overlap                 # 真交集里被判为不匹配的
    false_pos = mismatch & (~in_overlap)         # 非交集里被误判为匹配的
    matched[flip] = False
    matched[false_pos] = True
    n_true = int(in_overlap.sum())
    n_matched = int(matched.sum())
    tp = int((matched & in_overlap).sum())
    return {
        "n_true_overlap": n_true,
        "n_matched": n_matched,
        "true_positive": tp,
        "false_positive": n_matched - tp,
        "false_negative": n_true - tp,
        "precision": tp / max(n_matched, 1),
        "recall": tp / max(n_true, 1),
        "matched_mask": matched,
    }


def psi_cost(n_a: int, n_b: int) -> Dict:
    """ECDH-PSI 的通信量与模幂次数的数量级估计（用于可行性判断，不是性能基准）。"""
    return {
        "comm_bytes": (n_a + n_b) * BYTES_PER_HASH * ROUNDS_ECDH_PSI,
        "comm_mb": (n_a + n_b) * BYTES_PER_HASH * ROUNDS_ECDH_PSI / BYTES_PER_MB,
        "exponentiations": EXPONENTIATIONS_PER_ITEM * (n_a + n_b),
        "rounds": ROUNDS_ECDH_PSI,
    }


def apply_misattribution(x_b: np.ndarray, rate: float, seed: int) -> np.ndarray:
    """错配：把一部分已匹配实体的被动方特征**张冠李戴**地互换。

    与漏配（少了人）不同，错配是「人还在、特征是别人的」，
    它把无关噪声直接注入模型，且在合规上属于**错误的个人信息关联**。
    实现为在选中子集内做一次错位置换（保证没有样本换回自己）。
    """
    x = x_b.copy()
    n = len(x)
    k = int(n * rate)
    if k < 2:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    x[idx] = x[np.roll(idx, 1)]                  # 错位一格：无人换回自己
    return x
