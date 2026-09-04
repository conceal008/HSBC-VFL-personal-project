# -*- coding: utf-8 -*-
"""platform · 主链路装配：M2 → M3 → M5 → M6 → M7。

每个模块贡献一个阶段，阶段之间只通过产物字典传递数据。
这样断点续跑才有意义——中断处的产物落盘后，续跑不需要重算上游。

⚠️ 导入方式说明：仓库目录 `platform/` 与 Python 标准库模块 `platform` 同名，
`import platform.orchestration.xxx` 不可行（stdlib 的 platform 是模块不是包）。
因此调用方需把本目录本身加入 sys.path 后直接 `import main_chain`。
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.m2_synthetic.components.scm_generator import (  # noqa: E402
    generate, load_scenarios)
from modules.m3_alignment.components.psi import simulate_psi  # noqa: E402
from modules.m5_modeling.components import models as M  # noqa: E402
from modules.m5_modeling.components.gbdt import VerticalGBDT  # noqa: E402
from modules.m6_evaluation.components import metrics as MT  # noqa: E402
from modules.m7_security.components import attacks as A  # noqa: E402

from pipeline import Stage  # noqa: E402

SCENARIO_CONFIG = "modules/m2_synthetic/configs/scenarios.yaml"
EXPERIMENT_CONFIG = "modules/m5_modeling/configs/experiment.yaml"
TRAIN_FRAC = 0.7
TOPK = 0.1


def _split(n: int, seed: int):
    order = np.random.default_rng(seed).permutation(n)
    cut = int(n * TRAIN_FRAC)
    return order[:cut], order[cut:]


def build_stages(smoke: Dict) -> List[Stage]:
    raw = yaml.safe_load(open(os.path.join(REPO_ROOT, SCENARIO_CONFIG), encoding="utf-8"))
    exp = yaml.safe_load(open(os.path.join(REPO_ROOT, EXPERIMENT_CONFIG), encoding="utf-8"))
    from dataclasses import replace
    cfg = [s for s in load_scenarios(raw) if s.name == smoke["scenario"]][0]
    cfg = replace(cfg, n_party_a=smoke["n_party_a"])
    hp = dict(exp["hyperparams"], **smoke["hyperparams"])
    seed = smoke["seeds"][0]

    def m2_generate(_ctx: Dict) -> Dict:
        d = generate(cfg, seed)
        return {"x_a_all": d["x_a"], "x_b_all": d["x_b"],
                "y_all": d["y_control"], "in_overlap": d["in_overlap"],
                "consent": d["consent"], "mismatch": d["mismatch"]}

    def m3_align(ctx: Dict) -> Dict:
        psi = simulate_psi(ctx["in_overlap"].astype(bool), ctx["mismatch"].astype(bool))
        keep = psi["matched_mask"] & ctx["consent"].astype(bool)
        return {"x_a": ctx["x_a_all"][keep], "x_b": ctx["x_b_all"][keep],
                "y": ctx["y_all"][keep],
                "psi_precision": psi["precision"], "psi_recall": psi["recall"],
                "n_usable": int(keep.sum())}

    def m5_model(ctx: Dict) -> Dict:
        x_a, x_b, y = ctx["x_a"], ctx["x_b"], ctx["y"]
        tr, te = _split(len(y), seed)
        seg = M.build_segments(x_a, smoke["n_segments"], seed)
        stats, _ = M.k_anonymous_segment_stats(seg, x_b, smoke["k_anonymity"])
        f_l1 = np.hstack([x_a, stats])
        s_l0 = M.fit_logistic(x_a[tr], y[tr], seed, hp["c_reg"]).decision_function(x_a[te])
        s_l1 = M.fit_logistic(f_l1[tr], y[tr], seed, hp["c_reg"]).decision_function(f_l1[te])
        flr = M.fit_federated_logistic(x_a[tr], x_b[tr], y[tr], hp["flr_rounds"],
                                       hp["flr_lr"], hp["flr_l2"], 0.0, seed)
        gb = VerticalGBDT(hp["n_rounds"], hp["max_depth"], hp["gbdt_lr"],
                          hp["n_bins"], hp["reg_lambda"], hp["min_gain"])
        gb.fit(x_a[tr], x_b[tr], y[tr])
        return {"y_test": y[te], "score_l0": s_l0, "score_l1": s_l1,
                "score_l3a": M.federated_lr_score(flr, x_a[te], x_b[te]),
                "score_l3b": gb.decision_function(x_a[te], x_b[te]),
                "residual_history": flr.residual_history, "y_train": y[tr],
                "comm_floats_l3a": flr.comm["comm_floats"],
                "comm_floats_l3b": gb.comm.as_dict()["comm_floats"]}

    def m6_evaluate(ctx: Dict) -> Dict:
        y = ctx["y_test"]
        out = {}
        for tag in ("l0", "l1", "l3a", "l3b"):
            out["auc_" + tag] = MT.auc(y, ctx["score_" + tag])
        out["gain_l3a_over_l1"] = out["auc_l3a"] - out["auc_l1"]
        out["topk_overlap_l1_l3a"] = MT.top_k_overlap(ctx["score_l1"], ctx["score_l3a"], TOPK)
        return out

    def m7_attack(ctx: Dict) -> Dict:
        leak = A.label_inference_from_residuals(ctx["residual_history"], ctx["y_train"])
        return {"leak_auc_首轮": leak["leak_auc_首轮"],
                "leak_auc_最优轮": leak["leak_auc_最优轮"],
                "eps_per_round": A.gaussian_epsilon_per_round(
                    smoke["attack"]["dp_sigma"], smoke["attack"]["dp_delta"])}

    return [Stage("m2_generate", m2_generate), Stage("m3_align", m3_align),
            Stage("m5_model", m5_model), Stage("m6_evaluate", m6_evaluate),
            Stage("m7_attack", m7_attack)]


SUMMARY_KEYS = ("n_usable", "psi_precision", "psi_recall",
                "auc_l0", "auc_l1", "auc_l3a", "auc_l3b",
                "gain_l3a_over_l1", "topk_overlap_l1_l3a",
                "comm_floats_l3a", "comm_floats_l3b",
                "leak_auc_首轮", "leak_auc_最优轮")


def summarize(ctx: Dict) -> Dict:
    """抽出用于比对的标量——断点续跑一致性就在这些数上逐位核对。"""
    return {k: (float(ctx[k]) if isinstance(ctx[k], (int, float, np.floating, np.integer))
                else ctx[k]) for k in SUMMARY_KEYS if k in ctx}
