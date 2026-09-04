"""M5 · 直方图式梯度提升树，支持「纵向联邦」记账（SecureBoost 式）。

SecureBoost 的核心机制：主动方算出每个样本的一阶/二阶梯度 (g, h) 后发给被动方；
被动方在**自己的特征**上做 (g, h) 的分桶直方图并回传增益；双方比较后选全局最优切分。
被动方始终看不到标签，主动方始终看不到对方特征值——只看到分桶统计。

本实现是**教学级最小版本**：直方图与增益按上述结构计算并记账，
但**不含任何密码学保护**（无同态加密）。见豁免 W-002。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

DENOM_EPS = 1e-12          # 分母下限，防 reg_lambda=0 时空桶除零
MIN_CHILD_WEIGHT = 1.0
GAIN_EPS = 1e-12
PROB_CLIP = 1e-6
LOGIT_CLIP = 30.0
HALF = 0.5


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -LOGIT_CLIP, LOGIT_CLIP)))


@dataclass
class Node:
    is_leaf: bool = True
    value: float = 0.0
    party: str = ""
    feature: int = -1
    threshold: float = 0.0
    left: Optional["Node"] = None
    right: Optional["Node"] = None


@dataclass
class CommLog:
    """跨方通信记账：VFL 的成本必须被量化，不能只报精度。"""
    rounds: int = 0
    gradient_messages: int = 0        # 主动方 → 被动方：(g, h) 向量
    histogram_messages: int = 0       # 被动方 → 主动方：分桶统计
    floats_sent: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "comm_rounds": self.rounds,
            "comm_grad_msgs": self.gradient_messages,
            "comm_hist_msgs": self.histogram_messages,
            "comm_floats": self.floats_sent,
        }


def _bin_features(x: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """等频分桶。返回 (桶索引矩阵, 每维的分位切点)。"""
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    cuts = np.quantile(x, quantiles, axis=0)
    binned = np.empty(x.shape, dtype=np.int16)
    for j in range(x.shape[1]):
        binned[:, j] = np.searchsorted(cuts[:, j], x[:, j], side="left")
    return binned, cuts


def _best_split_from_hist(g_hist: np.ndarray, h_hist: np.ndarray,
                          reg_lambda: float) -> Tuple[float, int, int]:
    """在一个方的直方图上找最优切分。返回 (增益, 特征号, 桶阈值)。"""
    g_total = g_hist.sum(axis=1, keepdims=True)
    h_total = h_hist.sum(axis=1, keepdims=True)
    g_left = np.cumsum(g_hist, axis=1)[:, :-1]
    h_left = np.cumsum(h_hist, axis=1)[:, :-1]
    g_right = g_total - g_left
    h_right = h_total - h_left

    valid = (h_left >= MIN_CHILD_WEIGHT) & (h_right >= MIN_CHILD_WEIGHT)
    # reg_lambda 可以为 0（无正则），此时空桶的 h 也为 0，分母会是 0。
    # 加 DENOM_EPS 兜底：它只在分母本就趋零时起作用，不改变正常分裂的增益。
    parent = (g_total ** 2) / (h_total + reg_lambda + DENOM_EPS)
    gain = ((g_left ** 2) / (h_left + reg_lambda + DENOM_EPS)
            + (g_right ** 2) / (h_right + reg_lambda + DENOM_EPS) - parent)
    gain = np.where(valid, gain, -np.inf)

    if not np.isfinite(gain).any():
        return -np.inf, -1, -1
    flat = int(np.argmax(gain))
    feat, cut = divmod(flat, gain.shape[1])
    return float(gain[feat, cut] * HALF), feat, cut


def _histogram(binned: np.ndarray, g: np.ndarray, h: np.ndarray,
               idx: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    n_feat = binned.shape[1]
    g_hist = np.zeros((n_feat, n_bins))
    h_hist = np.zeros((n_feat, n_bins))
    sub = binned[idx]
    for j in range(n_feat):
        g_hist[j] = np.bincount(sub[:, j], weights=g[idx], minlength=n_bins)
        h_hist[j] = np.bincount(sub[:, j], weights=h[idx], minlength=n_bins)
    return g_hist, h_hist


class VerticalGBDT:
    """纵向联邦 GBDT。party_b 为 None 时退化为单方 GBDT（用作 L0 基线）。"""

    def __init__(self, n_rounds: int, max_depth: int, learning_rate: float,
                 n_bins: int, reg_lambda: float, min_gain: float):
        self.n_rounds = n_rounds
        self.max_depth = max_depth
        self.lr = learning_rate
        self.n_bins = n_bins
        self.reg_lambda = reg_lambda
        self.min_gain = min_gain
        self.trees: List[Node] = []
        self.base_score = 0.0
        self.comm = CommLog()
        self._cuts: Dict[str, np.ndarray] = {}

    def _build(self, binned: Dict[str, np.ndarray], g: np.ndarray, h: np.ndarray,
               idx: np.ndarray, depth: int) -> Node:
        if depth >= self.max_depth or len(idx) <= 1:
            return Node(is_leaf=True,
                        value=float(-g[idx].sum() / (h[idx].sum() + self.reg_lambda)))

        best = (-np.inf, "", -1, -1)
        for party, bx in binned.items():
            gh, hh = _histogram(bx, g, h, idx, self.n_bins)
            if party == "b":
                # 记账：主动方发 (g,h)，被动方回直方图
                self.comm.gradient_messages += 1
                self.comm.histogram_messages += 1
                self.comm.floats_sent += int(len(idx) * 2 + gh.size * 2)
            gain, feat, cut = _best_split_from_hist(gh, hh, self.reg_lambda)
            if gain > best[0]:
                best = (gain, party, feat, cut)

        gain, party, feat, cut = best
        if gain < self.min_gain or feat < 0:
            return Node(is_leaf=True,
                        value=float(-g[idx].sum() / (h[idx].sum() + self.reg_lambda)))

        mask = binned[party][idx, feat] <= cut
        left_idx, right_idx = idx[mask], idx[~mask]
        if len(left_idx) == 0 or len(right_idx) == 0:
            return Node(is_leaf=True,
                        value=float(-g[idx].sum() / (h[idx].sum() + self.reg_lambda)))

        return Node(is_leaf=False, party=party, feature=feat, threshold=float(cut),
                    left=self._build(binned, g, h, left_idx, depth + 1),
                    right=self._build(binned, g, h, right_idx, depth + 1))

    def fit(self, x_a: np.ndarray, x_b: Optional[np.ndarray], y: np.ndarray) -> "VerticalGBDT":
        binned: Dict[str, np.ndarray] = {}
        binned["a"], self._cuts["a"] = _bin_features(x_a, self.n_bins)
        if x_b is not None:
            binned["b"], self._cuts["b"] = _bin_features(x_b, self.n_bins)

        p0 = float(np.clip(y.mean(), PROB_CLIP, 1 - PROB_CLIP))
        self.base_score = float(np.log(p0 / (1 - p0)))
        score = np.full(len(y), self.base_score)
        idx_all = np.arange(len(y))

        for _ in range(self.n_rounds):
            prob = _sigmoid(score)
            g = prob - y
            h = prob * (1 - prob)
            tree = self._build(binned, g, h, idx_all, 0)
            self.trees.append(tree)
            score = score + self.lr * self._predict_tree(tree, binned)
            self.comm.rounds += 1
        return self

    def _predict_tree(self, node: Node, binned: Dict[str, np.ndarray]) -> np.ndarray:
        n = next(iter(binned.values())).shape[0]
        out = np.zeros(n)
        stack = [(node, np.arange(n))]
        while stack:
            nd, idx = stack.pop()
            if nd.is_leaf:
                out[idx] = nd.value
                continue
            mask = binned[nd.party][idx, nd.feature] <= nd.threshold
            stack.append((nd.left, idx[mask]))
            stack.append((nd.right, idx[~mask]))
        return out

    def decision_function(self, x_a: np.ndarray, x_b: Optional[np.ndarray]) -> np.ndarray:
        binned = {"a": self._apply_cuts(x_a, self._cuts["a"])}
        if x_b is not None and "b" in self._cuts:
            binned["b"] = self._apply_cuts(x_b, self._cuts["b"])
        score = np.full(x_a.shape[0], self.base_score)
        for tree in self.trees:
            score = score + self.lr * self._predict_tree(tree, binned)
        return score

    def _apply_cuts(self, x: np.ndarray, cuts: np.ndarray) -> np.ndarray:
        out = np.empty(x.shape, dtype=np.int16)
        for j in range(x.shape[1]):
            out[:, j] = np.searchsorted(cuts[:, j], x[:, j], side="left")
        return out
