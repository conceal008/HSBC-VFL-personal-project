"""M5 · 极简 numpy 神经网络，用于 SplitNN（本机无 torch，见豁免 W-002）。

只实现够用的部分：单隐层 MLP + Adam。刻意保持简单，
因为本项目要验证的是「联邦结构」而非「网络结构」。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
HE_SCALE = 2.0
LOGIT_CLIP = 30.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -LOGIT_CLIP, LOGIT_CLIP)))


class Adam:
    """最小 Adam 实现。"""

    def __init__(self, params: Dict[str, np.ndarray], lr: float):
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = ADAM_BETA1 * self.m[k] + (1 - ADAM_BETA1) * g
            self.v[k] = ADAM_BETA2 * self.v[k] + (1 - ADAM_BETA2) * (g * g)
            m_hat = self.m[k] / (1 - ADAM_BETA1 ** self.t)
            v_hat = self.v[k] / (1 - ADAM_BETA2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)


class Encoder:
    """单隐层编码器：输入特征 → 表示向量。对应 VFL 中一方的本地模型。"""

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, rng: np.random.Generator):
        self.p = {
            "w1": rng.standard_normal((dim_in, dim_hidden)) * np.sqrt(HE_SCALE / dim_in),
            "b1": np.zeros(dim_hidden),
            "w2": rng.standard_normal((dim_hidden, dim_out)) * np.sqrt(HE_SCALE / dim_hidden),
            "b2": np.zeros(dim_out),
        }
        self._cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        a1 = np.tanh(x @ self.p["w1"] + self.p["b1"])
        out = a1 @ self.p["w2"] + self.p["b2"]
        self._cache = {"x": x, "a1": a1}
        return out

    def backward(self, grad_out: np.ndarray) -> Dict[str, np.ndarray]:
        """给定对输出（表示）的梯度，返回本地参数的梯度。

        注意：**返回值不含对输入 x 的梯度**——在 VFL 中，一方不需要也不应该
        把梯度继续传给数据源。这与「梯度回传给对方」是两件事。
        """
        x, a1 = self._cache["x"], self._cache["a1"]
        n = x.shape[0]
        g_w2 = a1.T @ grad_out / n
        g_b2 = grad_out.mean(axis=0)
        g_a1 = grad_out @ self.p["w2"].T
        g_z1 = g_a1 * (1 - a1 * a1)
        g_w1 = x.T @ g_z1 / n
        g_b1 = g_z1.mean(axis=0)
        return {"w1": g_w1, "b1": g_b1, "w2": g_w2, "b2": g_b2}


class TopModel:
    """顶层模型：拼接后的表示 → logit。持有标签的一方（主动方）拥有它。"""

    def __init__(self, dim_in: int, rng: np.random.Generator):
        self.p = {
            "w": rng.standard_normal((dim_in, 1)) * np.sqrt(HE_SCALE / dim_in),
            "b": np.zeros(1),
        }
        self._cache: Dict[str, np.ndarray] = {}

    def forward(self, h: np.ndarray) -> np.ndarray:
        self._cache = {"h": h}
        return (h @ self.p["w"] + self.p["b"]).ravel()

    def backward(self, grad_logit: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """返回 (顶层参数梯度, 对表示 h 的梯度)。

        对 h 的梯度就是 VFL 里**要不要回传给被动方**的那个东西——
        形态 A 回传，形态 B 不回传。
        """
        h = self._cache["h"]
        n = h.shape[0]
        g = grad_logit[:, None]
        return ({"w": h.T @ g / n, "b": np.array([g.mean()])}, g @ self.p["w"].T)
