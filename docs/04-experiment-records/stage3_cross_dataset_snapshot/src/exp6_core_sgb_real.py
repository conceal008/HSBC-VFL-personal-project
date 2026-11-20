"""
实验 6（Exp 6）：底层核心协议 / 隐私 / 集成优化在【真实 SecretFlow 1.14.0b0 SGB】上的跨数据集复核。
必须在 /Applications/总文件夹/汇丰/8-9月/隐语_SecretFlow/.venv/bin/python 下运行！

全面补齐此前真 SGB 仅在单一数据集上实测的验证空白：
- Part A — I1（OU vs Paillier-2048）+ M3（标签方明文直方图）：
  在 COIL2000（5,822×85）、UCI Bank 去duration（8,000×28）、Hillstrom（8,000×14）上实测真实加密耗时、提速倍数与预测逐位差！
- Part B — I2+I3（默认10桶/base_score=0 vs 33桶/prior base_score/调参）+ M6（6成员 Bagging）：
  在 COIL2000、UCI Bank、Hillstrom(visit)、Home Credit（8,000）四个数据集上直接实跑真实 SecretFlow SGB！
- Part C — I6 后验去噪软标签 DP vs 硬 RR（ε ∈ {1.0, 2.0, 4.0}，每 ε 跑 3 个 DP 随机种子取均值）：
  严格满足 ε-Label-DP 后处理免疫（主动方 OOF 模型仅在加噪标签 y_tilde 上训练并做无偏逆映射估计先验，绝不再碰原始 y！），
  直接输入真实 SecretFlow SGB（objective='logistic' 连续软标签），横跨：
  1) COIL2000（保险，主被动均衡）
  2) UCI Bank（去 duration，主动/营销强）
  3) Home Credit（信贷，被动方 EXT_SOURCE 极强）
  正面实证真实 SGB 下 I6 的正反增益边界！
"""
from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sgb_utils import run_sgb_fit_predict, sgb_metrics

STAGE3_ROOT = "/Applications/总文件夹/汇丰/第三阶段"
CACHE_DIR = os.path.join(STAGE3_ROOT, "results", "_cache")
OUT_PATH = os.path.join(STAGE3_ROOT, "results", "exp6_core_sgb_real.json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [Exp6-RealSGB] {msg}", flush=True)


def encode_simple(df_tr: pd.DataFrame, df_te: pd.DataFrame, cols: list) -> tuple[np.ndarray, np.ndarray]:
    tr_l, te_l = [], []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df_tr[c]):
            tr_l.append(pd.to_numeric(df_tr[c], errors="coerce").to_numpy(dtype=np.float32))
            te_l.append(pd.to_numeric(df_te[c], errors="coerce").to_numpy(dtype=np.float32))
        else:
            s_tr = df_tr[c].astype(str).fillna("__NA__")
            s_te = df_te[c].astype(str).fillna("__NA__")
            vc = s_tr.value_counts()
            mp = {k: float(i) for i, k in enumerate(vc.index)}
            tr_l.append(s_tr.map(mp).fillna(-1.0).to_numpy(dtype=np.float32))
            te_l.append(s_te.map(mp).fillna(-1.0).to_numpy(dtype=np.float32))
    return np.column_stack(tr_l).astype(np.float32), np.column_stack(te_l).astype(np.float32)


def load_four_datasets() -> dict:
    ds = {}
    # 1. COIL2000 (official 5,822 / 4,000)
    cdir = "/Applications/总文件夹/汇丰/8-9月/数据集/COIL2000_保险营销/解压"
    c_tr = pd.read_csv(os.path.join(cdir, "ticdata2000.txt"), sep="\t", header=None)
    c_ev = pd.read_csv(os.path.join(cdir, "ticeval2000.txt"), sep="\t", header=None).to_numpy(dtype=np.float32)
    c_y_ev = pd.read_csv(os.path.join(cdir, "tictgts2000.txt"), sep="\t", header=None).iloc[:, 0].to_numpy(dtype=np.int32)
    ds["COIL2000"] = {
        "Xa_tr": c_tr.iloc[:, :43].to_numpy(dtype=np.float32),
        "Xb_tr": c_tr.iloc[:, 43:85].to_numpy(dtype=np.float32),
        "y_tr": c_tr.iloc[:, 85].to_numpy(dtype=np.int32),
        "Xa_te": c_ev[:, :43],
        "Xb_te": c_ev[:, 43:85],
        "y_te": c_y_ev,
    }

    # 2. UCI Bank (duration strictly dropped! 8,000 train / 4,000 test)
    u = pd.read_csv("/Applications/总文件夹/汇丰/8-9月/数据集/UCI_BankMarketing/解压/bank-additional/bank-additional-full.csv", sep=";")
    u = u.drop(columns=["duration"])
    uy = (u["y"] == "yes").to_numpy(dtype=np.int32)
    i_tr, i_tmp = train_test_split(np.arange(len(u)), train_size=8000, stratify=uy, random_state=2026)
    i_te, _ = train_test_split(i_tmp, train_size=4000, stratify=uy[i_tmp], random_state=2026)
    # Active: client + campaign (strong active); Passive: macro + remaining campaign
    u_a_cols = ["age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "campaign", "pdays", "previous", "poutcome"]
    u_b_cols = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
    Xa_tr, Xa_te = encode_simple(u.iloc[i_tr], u.iloc[i_te], u_a_cols)
    Xb_tr, Xb_te = encode_simple(u.iloc[i_tr], u.iloc[i_te], u_b_cols)
    ds["UCI_Bank_NoDur"] = {"Xa_tr": Xa_tr, "Xb_tr": Xb_tr, "y_tr": uy[i_tr], "Xa_te": Xa_te, "Xb_te": Xb_te, "y_te": uy[i_te]}

    # 3. Hillstrom Visit (8,000 train / 4,000 test)
    h = pd.read_csv("/Applications/总文件夹/汇丰/8-9月/数据集/Hillstrom_邮件营销实验/hillstrom_email_marketing.csv")
    hy = h["visit"].to_numpy(dtype=np.int32)
    i_tr, i_tmp = train_test_split(np.arange(len(h)), train_size=8000, stratify=hy, random_state=2026)
    i_te, _ = train_test_split(i_tmp, train_size=4000, stratify=hy[i_tmp], random_state=2026)
    Xa_tr, Xa_te = encode_simple(h.iloc[i_tr], h.iloc[i_te], ["zip_code", "newbie", "channel", "segment"])
    Xb_tr, Xb_te = encode_simple(h.iloc[i_tr], h.iloc[i_te], ["recency", "history_segment", "history", "mens", "womens"])
    ds["Hillstrom_Visit"] = {"Xa_tr": Xa_tr, "Xb_tr": Xb_tr, "y_tr": hy[i_tr], "Xa_te": Xa_te, "Xb_te": Xb_te, "y_te": hy[i_te]}

    # 4. Home Credit (from cached exp1_sgb_hc.npz -> 8,000 train / 4,000 test)
    hc = np.load(os.path.join(CACHE_DIR, "exp1_sgb_hc.npz"))
    i_tr, _ = train_test_split(np.arange(len(hc["y_tr"])), train_size=8000, stratify=hc["y_tr"], random_state=2026)
    i_te, _ = train_test_split(np.arange(len(hc["y_te"])), train_size=4000, stratify=hc["y_te"], random_state=2026)
    ds["HomeCredit_8k"] = {
        "Xa_tr": hc["neidi_tr"][i_tr], "Xb_tr": hc["hk_tr"][i_tr], "y_tr": hc["y_tr"][i_tr],
        "Xa_te": hc["neidi_te"][i_te], "Xb_te": hc["hk_te"][i_te], "y_te": hc["y_te"][i_te],
    }
    return ds


def run_part_a_i1_m3(ds: dict) -> dict:
    log("=== Part A: I1 (OU vs Paillier-2048) + M3 (标签方明文直方图) 真实 SGB 基准测试 ===")
    res = {}
    for name in ("COIL2000", "UCI_Bank_NoDur", "Hillstrom_Visit"):
        d = ds[name]
        # 10 trees so Paillier-2048 completes cleanly while giving exact per-tree crypto timing
        p_pai, i_pai = run_sgb_fit_predict(
            ["alice", "bob"], [d["Xa_tr"], d["Xb_tr"]], d["y_tr"], [d["Xa_te"], d["Xb_te"]], d["y_te"],
            schema="paillier", num_boost_round=10, max_depth=4, learning_rate=0.1, reg_lambda=5.0,
            use_m3_plaintext_label=False, seed=11,
        )
        p_ou, i_ou = run_sgb_fit_predict(
            ["alice", "bob"], [d["Xa_tr"], d["Xb_tr"]], d["y_tr"], [d["Xa_te"], d["Xb_te"]], d["y_te"],
            schema="ou", num_boost_round=10, max_depth=4, learning_rate=0.1, reg_lambda=5.0,
            use_m3_plaintext_label=False, seed=11,
        )
        p_m3, i_m3 = run_sgb_fit_predict(
            ["alice", "bob"], [d["Xa_tr"], d["Xb_tr"]], d["y_tr"], [d["Xa_te"], d["Xb_te"]], d["y_te"],
            schema="ou", num_boost_round=10, max_depth=4, learning_rate=0.1, reg_lambda=5.0,
            use_m3_plaintext_label=True, seed=11,
        )
        diff_pai_ou = float(np.max(np.abs(p_pai - p_ou)))
        diff_ou_m3 = float(np.max(np.abs(p_ou - p_m3)))
        res[name] = {
            "n_train": int(len(d["y_tr"])),
            "dims": [int(d["Xa_tr"].shape[1]), int(d["Xb_tr"].shape[1])],
            "trees": 10,
            "paillier_fit_sec": i_pai["fit_sec"],
            "ou_fit_sec": i_ou["fit_sec"],
            "ou_m3_fit_sec": i_m3["fit_sec"],
            "speedup_ou_vs_paillier": round(i_pai["fit_sec"] / max(i_ou["fit_sec"], 1e-6), 3),
            "speedup_ou_m3_vs_paillier": round(i_pai["fit_sec"] / max(i_m3["fit_sec"], 1e-6), 3),
            "speedup_m3_vs_ou": round(i_ou["fit_sec"] / max(i_m3["fit_sec"], 1e-6), 3),
            "max_abs_diff_paillier_vs_ou": diff_pai_ou,
            "max_abs_diff_ou_vs_m3": diff_ou_m3,
            "auc_paillier": i_pai["metrics"]["auc"],
            "auc_ou": i_ou["metrics"]["auc"],
            "auc_ou_m3": i_m3["metrics"]["auc"],
        }
        log(f"  {name:16s}: Paillier={i_pai['fit_sec']}s -> OU={i_ou['fit_sec']}s ({res[name]['speedup_ou_vs_paillier']}x) "
            f"-> OU+M3={i_m3['fit_sec']}s ({res[name]['speedup_ou_m3_vs_paillier']}x) | "
            f"diff(Pai,OU)={diff_pai_ou:.2e}, diff(OU,M3)={diff_ou_m3:.2e}")
    return res


def run_part_b_i2i3_m6(ds: dict) -> dict:
    log("=== Part B: I2+I3（默认10桶 vs 33桶+先验基分+调参）+ M6（6成员 Bagging）真实 SGB ===")
    seeds = [11, 22, 33, 44, 55, 66]
    res = {}
    for name in ("COIL2000", "UCI_Bank_NoDur", "Hillstrom_Visit", "HomeCredit_8k"):
        d = ds[name]
        # 1. Default SecretFlow SGB config (sketch_eps=0.1 ~10 bins, base_score=0, 10 trees, lr=0.3, depth=5, lambda=1.0)
        p_def, i_def = run_sgb_fit_predict(
            ["alice", "bob"], [d["Xa_tr"], d["Xb_tr"]], d["y_tr"], [d["Xa_te"], d["Xb_te"]], d["y_te"],
            schema="ou", num_boost_round=10, max_depth=5, learning_rate=0.3, reg_lambda=1.0,
            sketch_eps=0.1, colsample_by_tree=1.0, rowsample_by_tree=1.0,
            use_prior_base_score=False, first_tree_with_label_holder_feature=True,
            use_m3_plaintext_label=True, seed=11,
        )
        # 2. Optimized I2+I3 SGB across 6 seeds -> also directly gives M6 6-member Bagging!
        preds_6 = []
        fit_secs = []
        aucs_6 = []
        for s in seeds:
            p_s, i_s = run_sgb_fit_predict(
                ["alice", "bob"], [d["Xa_tr"], d["Xb_tr"]], d["y_tr"], [d["Xa_te"], d["Xb_te"]], d["y_te"],
                schema="ou", num_boost_round=30, max_depth=4, learning_rate=0.10, reg_lambda=5.0,
                sketch_eps=0.03, colsample_by_tree=0.8, rowsample_by_tree=0.8,
                use_prior_base_score=True, first_tree_with_label_holder_feature=False,
                use_m3_plaintext_label=True, seed=s,
            )
            preds_6.append(p_s)
            fit_secs.append(i_s["fit_sec"])
            aucs_6.append(i_s["metrics"]["auc"])

        preds_6 = np.asarray(preds_6)
        bag_pred = preds_6.mean(axis=0)
        bag_m = sgb_metrics(d["y_te"], bag_pred)

        # Paired test bootstrap for Bagging6 - SingleOptMean (on test set)
        single_mean_auc = float(np.mean(aucs_6))
        single_std_auc = float(np.std(aucs_6, ddof=1))
        delta_i2i3 = single_mean_auc - i_def["metrics"]["auc"]
        delta_m6 = bag_m["auc"] - single_mean_auc

        res[name] = {
            "n_train": int(len(d["y_tr"])),
            "n_test": int(len(d["y_te"])),
            "default_sgb": {
                "auc": i_def["metrics"]["auc"],
                "pr_auc": i_def["metrics"]["pr_auc"],
                "brier": i_def["metrics"]["brier"],
                "fit_sec": i_def["fit_sec"],
            },
            "opt_i2i3_6seed": {
                "auc_mean": single_mean_auc,
                "auc_std": single_std_auc,
                "per_seed_auc": aucs_6,
                "mean_fit_sec": round(float(np.mean(fit_secs)), 2),
                "delta_vs_default": round(delta_i2i3, 5),
            },
            "m6_bagging6_sgb": {
                "auc": bag_m["auc"],
                "pr_auc": bag_m["pr_auc"],
                "brier": bag_m["brier"],
                "top5_lift": bag_m["top5_lift"],
                "total_fit_sec_6x": round(float(np.sum(fit_secs)), 2),
                "delta_vs_opt_mean": round(delta_m6, 5),
                "delta_vs_default": round(bag_m["auc"] - i_def["metrics"]["auc"], 5),
            },
        }
        log(f"  {name:16s}: Default={i_def['metrics']['auc']:.4f} -> Opt(I2+I3)={single_mean_auc:.4f}±{single_std_auc:.4f} ({delta_i2i3:+.4f}) "
            f"-> M6_Bag6={bag_m['auc']:.4f} ({delta_m6:+.4f} vs Opt) | {np.mean(fit_secs):.1f}s/member")
    return res


def posterior_denoise_rr(Xa_tr: np.ndarray, y_tilde: np.ndarray, eps: float, seed: int) -> np.ndarray:
    """
    严格满足 ε-Label-DP 后处理免疫的后验去噪软标签（绝不访问原始 y！）：
    1. 主动方在本地用 5-fold OOF 拟合 p_tilde(x_A) = P(y_tilde=1 | x_A)
    2. 按 RR 翻转概率 q = 1 / (1 + exp(eps)) 无偏反演先验 p0(x_A) = clip((p_tilde - q)/(1 - 2q), 0.01, 0.99)
    3. 贝叶斯后验 P(y=1 | y_tilde, x_A)
    """
    p_keep = float(np.exp(eps) / (1.0 + np.exp(eps)))
    q_flip = 1.0 - p_keep
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p_tilde_oof = np.zeros(len(y_tilde), dtype=np.float64)
    for k, (fi_tr, fi_va) in enumerate(skf.split(Xa_tr, y_tilde)):
        clf = HistGradientBoostingClassifier(
            max_bins=33, max_depth=4, learning_rate=0.08, max_iter=80,
            l2_regularization=10.0, min_samples_leaf=30, early_stopping=False, random_state=seed + k,
        )
        clf.fit(Xa_tr[fi_tr], y_tilde[fi_tr])
        p_tilde_oof[fi_va] = clf.predict_proba(Xa_tr[fi_va])[:, 1]

    # Unbiased inversion of RR channel to estimate true prior P(y=1 | x_A)
    denom = max(1.0 - 2.0 * q_flip, 1e-4)
    p0 = np.clip((p_tilde_oof - q_flip) / denom, 0.01, 0.99)

    # Exact Bayes posterior P(y=1 | y_tilde, x_A)
    lik_y1 = np.where(y_tilde == 1, p_keep, q_flip)
    lik_y0 = np.where(y_tilde == 1, q_flip, p_keep)
    num = lik_y1 * p0
    den = num + lik_y0 * (1.0 - p0)
    y_soft = np.clip(num / np.maximum(den, 1e-12), 1e-4, 1.0 - 1e-4)
    return y_soft.astype(np.float32)


def run_part_c_i6_label_dp(ds: dict) -> dict:
    log("=== Part C: I6 后验去噪软标签 DP vs 硬 RR（真实 SecretFlow 1.14 SGB 正反边界复核） ===")
    eps_list = [1.0, 2.0, 4.0]
    dp_seeds = [101, 202, 303]
    res = {}

    for name in ("COIL2000", "UCI_Bank_NoDur", "HomeCredit_8k"):
        d = ds[name]
        # Use up to 5,000 train rows for 54 real SGB fits in Part C (keeps COIL2000 full 5,822)
        if len(d["y_tr"]) > 5000:
            sub_i, _ = train_test_split(np.arange(len(d["y_tr"])), train_size=5000, stratify=d["y_tr"], random_state=2026)
            Xa_tr, Xb_tr, y_tr = d["Xa_tr"][sub_i], d["Xb_tr"][sub_i], d["y_tr"][sub_i]
        else:
            Xa_tr, Xb_tr, y_tr = d["Xa_tr"], d["Xb_tr"], d["y_tr"]
        Xa_te, Xb_te, y_te = d["Xa_te"], d["Xb_te"], d["y_te"]

        # Clean no-DP SGB reference + active-only reference (to quantify active vs passive strength!)
        p_clean, i_clean = run_sgb_fit_predict(
            ["alice", "bob"], [Xa_tr, Xb_tr], y_tr, [Xa_te, Xb_te], y_te,
            schema="ou", num_boost_round=25, max_depth=4, learning_rate=0.10, reg_lambda=5.0,
            sketch_eps=0.03, colsample_by_tree=0.8, rowsample_by_tree=0.8,
            use_prior_base_score=True, use_m3_plaintext_label=True, seed=11,
        )
        # Active-only reference (plaintext HGB on Xa)
        clf_a = HistGradientBoostingClassifier(max_bins=33, max_depth=4, learning_rate=0.10, max_iter=60, l2_regularization=5.0, early_stopping=False, random_state=11)
        clf_a.fit(Xa_tr, y_tr)
        active_auc = float(roc_auc_score(y_te, clf_a.predict_proba(Xa_te)[:, 1]))

        eps_out = {}
        for eps in eps_list:
            p_keep = float(np.exp(eps) / (1.0 + np.exp(eps)))
            hard_aucs, soft_aucs = [], []
            for s in dp_seeds:
                rng = np.random.default_rng(s + int(eps * 100))
                flip = rng.random(len(y_tr)) >= p_keep
                y_tilde = np.where(flip, 1 - y_tr, y_tr).astype(np.int32)
                y_soft = posterior_denoise_rr(Xa_tr, y_tilde, eps=eps, seed=s)

                # Hard RR SGB
                _, i_hard = run_sgb_fit_predict(
                    ["alice", "bob"], [Xa_tr, Xb_tr], y_tilde, [Xa_te, Xb_te], y_te,
                    schema="ou", num_boost_round=25, max_depth=4, learning_rate=0.10, reg_lambda=5.0,
                    sketch_eps=0.03, colsample_by_tree=0.8, rowsample_by_tree=0.8,
                    use_prior_base_score=True, use_m3_plaintext_label=True, seed=11,
                )
                # I6 Posterior-Denoised Soft-Label SGB
                _, i_soft = run_sgb_fit_predict(
                    ["alice", "bob"], [Xa_tr, Xb_tr], y_soft, [Xa_te, Xb_te], y_te,
                    schema="ou", num_boost_round=25, max_depth=4, learning_rate=0.10, reg_lambda=5.0,
                    sketch_eps=0.03, colsample_by_tree=0.8, rowsample_by_tree=0.8,
                    use_prior_base_score=True, use_m3_plaintext_label=True, seed=11,
                )
                hard_aucs.append(i_hard["metrics"]["auc"])
                soft_aucs.append(i_soft["metrics"]["auc"])

            hm, sm = float(np.mean(hard_aucs)), float(np.mean(soft_aucs))
            eps_out[f"eps_{eps:.1f}"] = {
                "epsilon": eps,
                "p_keep": round(p_keep, 4),
                "hard_rr_auc_mean": hm,
                "hard_rr_auc_std": float(np.std(hard_aucs, ddof=1)),
                "i6_soft_auc_mean": sm,
                "i6_soft_auc_std": float(np.std(soft_aucs, ddof=1)),
                "delta_i6_vs_hard": round(sm - hm, 5),
                "gap_i6_vs_clean": round(sm - i_clean["metrics"]["auc"], 5),
                "per_seed_hard": hard_aucs,
                "per_seed_soft": soft_aucs,
            }
            log(f"  {name:16s} eps={eps:.1f}: Clean={i_clean['metrics']['auc']:.4f} | "
                f"HardRR={hm:.4f} -> I6_Soft={sm:.4f} (Δ={sm-hm:+.4f})")

        res[name] = {
            "clean_sgb_auc": i_clean["metrics"]["auc"],
            "active_only_auc_ref": active_auc,
            "passive_gain_ref": round(i_clean["metrics"]["auc"] - active_auc, 4),
            "dp_seeds": dp_seeds,
            "by_epsilon": eps_out,
        }
    return res


def main() -> None:
    t0 = time.perf_counter()
    ds = load_four_datasets()
    out = {
        "experiment": "Exp6_CoreProtocol_Privacy_Ensemble_Real_SecretFlow_SGB",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": "SecretFlow 1.14.0b0 SGB (HEU OU/Paillier 2048-bit, FM64)",
    }
    out["part_a_i1_m3_crypto_speed"] = run_part_a_i1_m3(ds)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    out["part_b_i2i3_m6_bagging"] = run_part_b_i2i3_m6(ds)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    out["part_c_i6_soft_label_dp"] = run_part_c_i6_label_dp(ds)
    out["total_elapsed_sec"] = round(time.perf_counter() - t0, 2)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"Exp 6 finished in {out['total_elapsed_sec']:.1f}s -> {OUT_PATH}")


if __name__ == "__main__":
    main()
