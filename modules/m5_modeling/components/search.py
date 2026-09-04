# -*- coding: utf-8 -*-
"""M5 · 各级别超参搜索。

回答一个针对 C1 的质疑：**「L1 只捕获 VFL 价值的 12.3%」会不会只是 L1 欠调参？**

方法上有一条硬要求：**选参只看验证集，测试集只用于最终报告**。
在测试集上选参会让所有级别一起虚高，且**搜索空间越大虚高越多**——
而本文刻意给 L1 最大的搜索空间（60 组，是 L0 的 12 倍），
若在测试集上选参，污染方向恰好**有利于 L1**，同样会毁掉结论。
"""
from __future__ import annotations

import itertools
from typing import Dict, Iterator, List, Tuple

import numpy as np

from ...m6_evaluation.components.metrics import auc
from . import models as M
from .gbdt import VerticalGBDT


def grid_points(grid: Dict[str, List]) -> Iterator[Dict]:
    """把 {参数名: 候选列表} 展开成逐个参数组合。"""
    keys = sorted(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def three_way_split(n: int, train_frac: float, valid_frac: float,
                    seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    a = int(n * train_frac)
    b = a + int(n * valid_frac)
    return order[:a], order[a:b], order[b:]


def _linear_scores(feats: np.ndarray, y: np.ndarray, tr: np.ndarray,
                   parts: List[np.ndarray], seed: int, c_reg: float) -> List[np.ndarray]:
    model = M.fit_logistic(feats[tr], y[tr], seed, c_reg)
    return [model.decision_function(feats[p]) for p in parts]


def evaluate_point(level: str, point: Dict, fixed: Dict, x_a: np.ndarray,
                   x_b: np.ndarray, y: np.ndarray, tr: np.ndarray,
                   va: np.ndarray, te: np.ndarray, seed: int) -> Tuple[float, float]:
    """返回该超参点在（验证集, 测试集）上的 AUC。选参只准用第一个值。"""
    if level == "L0_LR":
        s_va, s_te = _linear_scores(x_a, y, tr, [va, te], seed, point["c_reg"])
    elif level == "L1_LR":
        seg = M.build_segments(x_a, point["n_segments"], seed)
        stats, _ = M.k_anonymous_segment_stats(seg, x_b, point["k_anonymity"])
        feats = np.hstack([x_a, stats])
        s_va, s_te = _linear_scores(feats, y, tr, [va, te], seed, point["c_reg"])
    elif level == "L2_LR":
        feats = np.hstack([x_a, M.coarse_flag(x_b, seed)])
        s_va, s_te = _linear_scores(feats, y, tr, [va, te], seed, point["c_reg"])
    elif level == "L4_LR":
        feats = np.hstack([x_a, x_b])
        s_va, s_te = _linear_scores(feats, y, tr, [va, te], seed, point["c_reg"])
    elif level == "L3a_联邦LR":
        flr_res = M.fit_federated_logistic(x_a[tr], x_b[tr], y[tr], point["flr_rounds"],
                                           point["flr_lr"], point["flr_l2"], 0.0, seed)
        s_va = M.federated_lr_score(flr_res, x_a[va], x_b[va])
        s_te = M.federated_lr_score(flr_res, x_a[te], x_b[te])
    elif level == "L3b_纵向GBDT":
        model = VerticalGBDT(point["n_rounds"], point["max_depth"], point["gbdt_lr"],
                             fixed["n_bins"], point["reg_lambda"], fixed["min_gain"])
        model.fit(x_a[tr], x_b[tr], y[tr])
        s_va = model.decision_function(x_a[va], x_b[va])
        s_te = model.decision_function(x_a[te], x_b[te])
    elif level == "L3c_SplitNN":
        nn_res = M.fit_splitnn(x_a[tr], x_b[tr], y[tr], "bidirectional",
                               point["dim_hidden"], point["dim_embed"],
                               point["nn_epochs"], fixed["batch_size"],
                               point["nn_lr"], seed)
        s_va = nn_res.score_fn(x_a[va], x_b[va])
        s_te = nn_res.score_fn(x_a[te], x_b[te])
    else:
        raise ValueError("未知级别：%s" % level)
    return auc(y[va], s_va), auc(y[te], s_te)


def search_level(level: str, grid: Dict[str, List], fixed: Dict, x_a: np.ndarray,
                 x_b: np.ndarray, y: np.ndarray, tr: np.ndarray, va: np.ndarray,
                 te: np.ndarray, seed: int) -> Dict:
    """在验证集上选出最优超参点，返回它在测试集上的表现。"""
    best_valid = -np.inf
    best: Dict = {}
    n_points = 0
    for point in grid_points(grid):
        v, t = evaluate_point(level, point, fixed, x_a, x_b, y, tr, va, te, seed)
        n_points += 1
        if v > best_valid:
            best_valid = v
            best = {"valid_auc": float(v), "test_auc": float(t), "point": dict(point)}
    best.update({"level": level, "seed": seed, "n_points": n_points})
    return best
