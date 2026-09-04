"""M5 · 实验编排：在每个场景 × 每个种子上跑完整的五级基线阶梯。

划分口径同时做两种（框架 M6 要求）：
- D-1 随机划分：学术惯例，**会高估真实性能**
- D-2 时间外推（OOT）：金融业的真实口径
两者的差距本身就是要报告的结论。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ...m2_synthetic.components.scm_generator import GeneratorConfig, generate, usable_mask
from ...m6_evaluation.components import metrics as MT
from . import models as M
from .gbdt import VerticalGBDT

TRAIN_FRAC = 0.7
TOPK_FRACTIONS = (0.05, 0.10, 0.20)
PRIMARY_TOPK = 0.10


def _split(n: int, time_index: np.ndarray, mode: str,
           rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n_train = int(n * TRAIN_FRAC)
    if mode == "oot":
        order = np.argsort(time_index)          # 早期训练，晚期测试
    else:
        order = rng.permutation(n)
    return order[:n_train], order[n_train:]


def _uplift_scores(x_tr: np.ndarray, y_tr: np.ndarray, t_tr: np.ndarray,
                   x_te: np.ndarray, seed: int, c_reg: float) -> np.ndarray:
    """T-learner：处理组与对照组各训一个模型，增益 = 两者预测概率之差。"""
    treated, control = t_tr == 1, t_tr == 0
    if len(np.unique(y_tr[treated])) < 2 or len(np.unique(y_tr[control])) < 2:
        return np.zeros(len(x_te))
    m_t = M.fit_logistic(x_tr[treated], y_tr[treated], seed, c_reg)
    m_c = M.fit_logistic(x_tr[control], y_tr[control], seed, c_reg)
    return m_t.predict_proba(x_te)[:, 1] - m_c.predict_proba(x_te)[:, 1]


def run_one(cfg: GeneratorConfig, seed: int, hp: Dict, split_mode: str) -> List[Dict]:
    """跑一个 (场景, 种子, 划分方式) 组合，返回每个模型一行记录。"""
    data = generate(cfg, seed)
    mask = usable_mask(data)                     # 合规折损：交集 ∩ 同意
    x_a, x_b = data["x_a"][mask], data["x_b"][mask]
    y_obs, treat = data["y"][mask], data["treatment"][mask]
    y_ctrl = data["y_control"][mask]
    t_idx = data["time_index"][mask]
    n = len(y_ctrl)

    rng = np.random.default_rng(seed)
    tr, te = _split(n, t_idx, split_mode, rng)

    # —— 各级别可见的信息集 ——
    seg = M.build_segments(x_a, hp["n_segments"], seed)
    seg_stats, seg_info = M.k_anonymous_segment_stats(seg, x_b, hp["k_anonymity"])
    flag = M.coarse_flag(x_b, seed)
    sets = {
        "L0_内地单方": x_a,
        "L1_加k匿名统计": np.hstack([x_a, seg_stats]),
        "L2_加粗粒度标记": np.hstack([x_a, flag]),
        "L4_集中式": np.hstack([x_a, x_b]),
    }

    rows: List[Dict] = []
    scores: Dict[str, np.ndarray] = {}

    def record(level: str, algo: str, score: np.ndarray, comm: Dict) -> None:
        scores[level] = score
        row = {
            "scenario": cfg.name, "seed": seed, "split": split_mode, "level": level,
            "algo": algo, "n_usable": int(n), "n_test": int(len(te)),
            "auc": MT.auc(y_ctrl[te], score), "ks": MT.ks(y_ctrl[te], score),
        }
        for k in TOPK_FRACTIONS:
            row[f"lift@{int(k*PERCENT_INT)}"] = MT.lift_at_k(y_ctrl[te], score, k)
            row[f"recall@{int(k*PERCENT_INT)}"] = MT.recall_at_k(y_ctrl[te], score, k)
        row.update(comm)
        rows.append(row)

    # L0 / L1 / L2 / L4：线性与树两套
    for level, feats in sets.items():
        lin = M.fit_logistic(feats[tr], y_ctrl[tr], seed, hp["c_reg"])
        record(level + "_LR", "logistic", lin.decision_function(feats[te]),
               {"comm_floats": seg_info["comm_floats"] if level.startswith("L1") else 0})
    for level, xb_arg in [("L0_内地单方_GBDT", None), ("L4_集中式_GBDT", x_b)]:
        g = VerticalGBDT(hp["n_rounds"], hp["max_depth"], hp["gbdt_lr"],
                         hp["n_bins"], hp["reg_lambda"], hp["min_gain"])
        g.fit(x_a[tr], None if xb_arg is None else xb_arg[tr], y_ctrl[tr])
        record(level, "gbdt", g.decision_function(x_a[te], None if xb_arg is None else xb_arg[te]),
               g.comm.as_dict())

    # L3a 联邦逻辑回归
    flr = M.fit_federated_logistic(x_a[tr], x_b[tr], y_ctrl[tr], hp["flr_rounds"],
                                   hp["flr_lr"], hp["flr_l2"], 0.0, seed)
    record("L3a_联邦LR", "federated_logistic",
           M.federated_lr_score(flr, x_a[te], x_b[te]), flr.comm)

    # L3b 纵向 GBDT
    vg = VerticalGBDT(hp["n_rounds"], hp["max_depth"], hp["gbdt_lr"],
                      hp["n_bins"], hp["reg_lambda"], hp["min_gain"])
    vg.fit(x_a[tr], x_b[tr], y_ctrl[tr])
    record("L3b_纵向GBDT", "vertical_gbdt",
           vg.decision_function(x_a[te], x_b[te]), vg.comm.as_dict())

    # L3c SplitNN 三形态
    for mode, label in [("bidirectional", "L3c_SplitNN_形态A双向"),
                        ("frozen_pca", "L3c_SplitNN_形态B自监督"),
                        ("frozen_random", "L3c_SplitNN_形态B随机")]:
        sn = M.fit_splitnn(x_a[tr], x_b[tr], y_ctrl[tr], mode, hp["dim_hidden"],
                           hp["dim_embed"], hp["nn_epochs"], hp["batch_size"],
                           hp["nn_lr"], seed)
        record(label, "splitnn", sn.score_fn(x_a[te], x_b[te]), sn.comm)

    # —— 增量（uplift）：按信息集比较，而非按算法 ——
    for name, feats in sets.items():
        up = _uplift_scores(feats[tr], y_obs[tr], treat[tr], feats[te], seed, hp["c_reg"])
        rows.append({
            "scenario": cfg.name, "seed": seed, "split": split_mode,
            "level": name + "_UPLIFT", "algo": "t_learner", "n_usable": int(n),
            "n_test": int(len(te)),
            "auuc": MT.auuc(y_obs[te], treat[te], up),
            "uplift@10": MT.uplift_at_k(y_obs[te], treat[te], up, PRIMARY_TOPK),
        })

    # —— 决策等价性：L1 与各 L3 的 Top-K 名单重合度 ——
    for l3 in ["L3a_联邦LR", "L3b_纵向GBDT", "L3c_SplitNN_形态A双向"]:
        if "L1_加k匿名统计_LR" in scores and l3 in scores:
            rows.append({
                "scenario": cfg.name, "seed": seed, "split": split_mode,
                "level": f"OVERLAP_L1_vs_{l3}", "algo": "top_k_overlap",
                "topk_overlap@10": MT.top_k_overlap(scores["L1_加k匿名统计_LR"],
                                                    scores[l3], PRIMARY_TOPK),
            })
    return rows


PERCENT_INT = 100
