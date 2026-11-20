"""
实验 3（Exp 3）：M1x 非对齐样本本地先验 + 联邦融合跨数据集验证（6-seed 等调参代理 + 配对 Bootstrap CI + Holm 校正）。
同时导出精简 .npz 供 exp3_sgb_confirm.py 在真实 SecretFlow 1.14 SGB 上复核！

验证场景：
1. IEEE-CIS 446,307 笔真实非对齐交易（占全量 75.6%，此前从未在联邦中利用！）：
   - 严格时间因果切分：对齐样本按 TransactionDT 切分 60% train / 15% val / 25% test；
   - 非对齐样本严格只取 TransactionDT <= train_end（零未来穿越！）；
   - 注意自然分布偏移：对齐交易欺诈率 7.85% vs 非对齐交易欺诈率 2.09%（3.75× 偏移），
     正面检验 M1x 在真实异质非对齐场景下是否增益、以及哪种融合机制（本地先验特征 vs Logit 校准融合）最抗偏移！
2. Home Credit 低重合率（20% / 50%）真实非对齐池（全量 307,511 客户，同分布对照）：
   - 冻结 60,000 测试客户，在 20% 重合（36k 对齐训练 + 199.5k 真实非对齐）与 50% 重合（90k 对齐训练 + 127.5k 真实非对齐）下，
     对比 L0_aligned / L0_plus_unaligned / F0_aligned / M1x_prior_feature / M1x_logit_fusion。
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    STAGE3_ROOT,
    compute_metrics,
    encode_tabular_train_val_test,
    equal_tune_and_eval_hgb,
    fast_roc_auc,
    holm_bonferroni,
    load_config,
    paired_bootstrap_delta,
    save_json,
)

CACHE_DIR = os.path.join(STAGE3_ROOT, "results", "_cache")
OUT_DIR = os.path.join(STAGE3_ROOT, "results")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [Exp3] {msg}", flush=True)


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def run_contrasts(y_te: np.ndarray, arm_preds: dict, pairs: list, cfg: dict, cluster_ids=None) -> list:
    rows = []
    for name, base_k, new_k in pairs:
        res = paired_bootstrap_delta(
            y_te,
            arm_preds[base_k],
            arm_preds[new_k],
            metric_fn=fast_roc_auc,
            n_boot=cfg["bootstrap_iters"],
            seed=cfg["bootstrap_seed"],
            cluster_ids=cluster_ids,
        )
        res["contrast"] = name
        res["base_arm"] = base_k
        res["new_arm"] = new_k
        rows.append(res)
    p_adj = holm_bonferroni([r["p_raw"] for r in rows])
    for r, pa in zip(rows, p_adj):
        r["p_holm"] = float(pa)
        r["significant_holm_05"] = bool(pa < cfg["alpha"])
    return rows


def build_local_priors(
    X_un: np.ndarray,
    y_un: np.ndarray,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    X_te: np.ndarray,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    严格无泄漏构造主动方本地先验特征（3 维）：
      1) logit_unaligned: 仅在真实非对齐样本 (X_un, y_un) 上拟合的本地先验 log-odds（对对齐训练集天然 0 过拟合！）
      2) logit_pool_oof:  在 (X_un + 4/5 X_tr) 上 5 折折外拟合的联合先验 log-odds（val/test 用全量 X_un+X_tr 拟合）
      3) prob_unaligned:  非对齐先验概率
    """
    prior_cfg = dict(max_bins=33, learning_rate=0.05, max_depth=6, l2_regularization=5.0, max_iter=150, min_samples_leaf=30, early_stopping=False, random_state=seed)

    # 1. Pure unaligned model -> zero leakage on X_tr
    m_un = HistGradientBoostingClassifier(**prior_cfg)
    m_un.fit(X_un, y_un)
    p_un_tr = m_un.predict_proba(X_tr)[:, 1]
    p_un_va = m_un.predict_proba(X_va)[:, 1]
    p_un_te = m_un.predict_proba(X_te)[:, 1]

    # 2. 5-fold OOF pooled (unaligned + aligned_tr) model
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p_pool_tr = np.zeros(len(y_tr), dtype=np.float64)
    for k, (fi_tr, fi_va) in enumerate(skf.split(X_tr, y_tr)):
        m_k = HistGradientBoostingClassifier(**{**prior_cfg, "random_state": seed + k + 1})
        m_k.fit(np.vstack([X_un, X_tr[fi_tr]]), np.concatenate([y_un, y_tr[fi_tr]]))
        p_pool_tr[fi_va] = m_k.predict_proba(X_tr[fi_va])[:, 1]

    m_full = HistGradientBoostingClassifier(**{**prior_cfg, "random_state": seed + 99})
    m_full.fit(np.vstack([X_un, X_tr]), np.concatenate([y_un, y_tr]))
    p_pool_va = m_full.predict_proba(X_va)[:, 1]
    p_pool_te = m_full.predict_proba(X_te)[:, 1]

    F_tr = np.column_stack([safe_logit(p_un_tr), safe_logit(p_pool_tr), p_un_tr]).astype(np.float32)
    F_va = np.column_stack([safe_logit(p_un_va), safe_logit(p_pool_va), p_un_va]).astype(np.float32)
    F_te = np.column_stack([safe_logit(p_un_te), safe_logit(p_pool_te), p_un_te]).astype(np.float32)
    return F_tr, F_va, F_te


def build_logit_fusion_arm(res_f0: dict, F_va: np.ndarray, y_va: np.ndarray, F_te: np.ndarray, y_te: np.ndarray, seeds: list) -> dict:
    """Calibrated logistic stacking combining F0 log-odds + unaligned local priors (fitted strictly on val)."""
    preds_te = np.empty_like(res_f0["preds_te"])
    per_seed = []
    for i, s in enumerate(seeds):
        X_meta_va = np.column_stack([safe_logit(res_f0["preds_va"][i]), F_va[:, 0], F_va[:, 1]])
        X_meta_te = np.column_stack([safe_logit(res_f0["preds_te"][i]), F_te[:, 0], F_te[:, 1]])
        lr = LogisticRegression(C=1.0, max_iter=500)
        lr.fit(X_meta_va, y_va)
        p_te = lr.predict_proba(X_meta_te)[:, 1]
        preds_te[i] = p_te
        m = compute_metrics(y_te, p_te)
        m["seed"] = int(s)
        per_seed.append(m)
    summary = {f"{k}_mean": float(np.mean([m[k] for m in per_seed])) for k in ["auc", "pr_auc", "ks", "brier", "top5_lift", "top10_lift"]}
    for k in ["auc", "pr_auc", "ks", "brier", "top5_lift", "top10_lift"]:
        summary[f"{k}_std"] = float(np.std([m[k] for m in per_seed], ddof=1))
    return {
        "method": "Logistic calibration fusion on val: logit(F0) + logit_unaligned + logit_pool_oof",
        "summary": summary,
        "bagging6_metrics": compute_metrics(y_te, preds_te.mean(axis=0)),
        "per_seed_metrics": per_seed,
        "preds_te": preds_te,
    }


def run_ieee_unaligned(cfg: dict) -> dict:
    log("=== 1/2 IEEE-CIS 446,307 笔真实非对齐交易 M1x 验证 ===")
    ieee_dir = cfg["datasets"]["ieee_cis_dir"]
    tx = pd.read_csv(os.path.join(ieee_dir, "train_transaction.csv"), low_memory=False)
    idn = pd.read_csv(os.path.join(ieee_dir, "train_identity.csv"), low_memory=False)
    id_set = set(idn["TransactionID"].values)

    tx = tx.sort_values("TransactionDT").reset_index(drop=True)
    aligned_mask = tx["TransactionID"].isin(id_set).values

    tx_al = tx.loc[aligned_mask].merge(idn, on="TransactionID", how="inner").sort_values("TransactionDT").reset_index(drop=True)
    tx_un = tx.loc[~aligned_mask].sort_values("TransactionDT").reset_index(drop=True)
    del tx, idn
    gc.collect()

    # Chronological split on aligned transactions: 60% tr / 15% va / 25% te
    n_al = len(tx_al)
    i_tr_end = int(0.60 * n_al)
    i_va_end = int(0.75 * n_al)
    t_tr_end = float(tx_al.loc[i_tr_end - 1, "TransactionDT"])

    # Strictly causal unaligned pool: ONLY unaligned tx with TransactionDT <= t_tr_end!
    tx_un_causal = tx_un.loc[tx_un["TransactionDT"] <= t_tr_end].reset_index(drop=True)
    log(f"  Aligned: tr={i_tr_end}, va={i_va_end-i_tr_end}, te={n_al-i_va_end} | "
        f"Causal Unaligned (<=T_tr_end): {len(tx_un_causal)} "
        f"(fraud rate: aligned_tr={tx_al.iloc[:i_tr_end]['isFraud'].mean():.4f} vs unaligned_tr={tx_un_causal['isFraud'].mean():.4f})")

    # Keep common transaction columns with <85% missing
    tx_cols = [
        c for c in tx_un_causal.columns
        if c not in ("TransactionID", "isFraud", "TransactionDT")
        and tx_al[c].isna().mean() < 0.85
    ]
    hk_cols = [c for c in tx_al.columns if c.startswith("id_") or c in ("DeviceType", "DeviceInfo")]

    d_tr, d_va, d_te = tx_al.iloc[:i_tr_end], tx_al.iloc[i_tr_end:i_va_end], tx_al.iloc[i_va_end:]
    y_tr = d_tr["isFraud"].to_numpy(dtype=np.int32)
    y_va = d_va["isFraud"].to_numpy(dtype=np.int32)
    y_te = d_te["isFraud"].to_numpy(dtype=np.int32)
    y_un = tx_un_causal["isFraud"].to_numpy(dtype=np.int32)
    card1_vals = d_te["card1"].to_numpy(dtype=np.float64)
    cids_te = np.where(np.isnan(card1_vals), -np.arange(1, len(d_te) + 1, dtype=np.float64), card1_vals)

    # Encode tx columns fitted on pooled (unaligned + aligned_tr) so categories seen in unaligned are properly coded!
    df_tx_pool = pd.concat([tx_un_causal[tx_cols], d_tr[tx_cols]], ignore_index=True)
    X_tx_pool, X_tx_va, X_tx_te, _ = encode_tabular_train_val_test(df_tx_pool, d_va, d_te, tx_cols)
    n_un = len(tx_un_causal)
    X_tx_un = X_tx_pool[:n_un]
    X_tx_tr = X_tx_pool[n_un:]
    X_hk_tr, X_hk_va, X_hk_te, _ = encode_tabular_train_val_test(d_tr, d_va, d_te, hk_cols)

    log("  Building leakage-free active-party local priors from real unaligned transactions ...")
    F_tr, F_va, F_te = build_local_priors(X_tx_un, y_un, X_tx_tr, y_tr, X_tx_va, X_tx_te, seed=cfg["split_seed"])

    grid = [
        {"learning_rate": 0.05, "max_depth": 6, "l2_regularization": 5.0, "max_iter": 120},
        {"learning_rate": 0.10, "max_depth": 6, "l2_regularization": 5.0, "max_iter": 100},
        {"learning_rate": 0.05, "max_depth": 4, "l2_regularization": 10.0, "max_iter": 150},
        {"learning_rate": 0.10, "max_depth": 4, "l2_regularization": 5.0, "max_iter": 120},
    ]

    arms_out, arm_preds = {}, {}

    # Arm 1: L0 aligned only
    arms_out["L0_aligned"] = equal_tune_and_eval_hgb(X_tx_tr, y_tr, X_tx_va, y_va, X_tx_te, y_te, seeds=cfg["seeds"], grid=grid, max_bins=33)
    arm_preds["L0_aligned"] = arms_out["L0_aligned"]["preds_te"]

    # Arm 2: L0 pooled (aligned + causal unaligned)
    arms_out["L0_plus_unaligned"] = equal_tune_and_eval_hgb(
        np.vstack([X_tx_un, X_tx_tr]), np.concatenate([y_un, y_tr]),
        X_tx_va, y_va, X_tx_te, y_te, seeds=cfg["seeds"], grid=grid, max_bins=33
    )
    arm_preds["L0_plus_unaligned"] = arms_out["L0_plus_unaligned"]["preds_te"]

    # Arm 3: F0 2-party VFL on aligned only
    arms_out["F0_aligned"] = equal_tune_and_eval_hgb(
        np.hstack([X_tx_tr, X_hk_tr]), y_tr,
        np.hstack([X_tx_va, X_hk_va]), y_va,
        np.hstack([X_tx_te, X_hk_te]), y_te, seeds=cfg["seeds"], grid=grid, max_bins=33
    )
    arm_preds["F0_aligned"] = arms_out["F0_aligned"]["preds_te"]

    # Arm 4: M1x prior feature (active appends leakage-free unaligned/OOF prior -> zero protocol change!)
    arms_out["M1x_prior_feature"] = equal_tune_and_eval_hgb(
        np.hstack([X_tx_tr, F_tr, X_hk_tr]), y_tr,
        np.hstack([X_tx_va, F_va, X_hk_va]), y_va,
        np.hstack([X_tx_te, F_te, X_hk_te]), y_te, seeds=cfg["seeds"], grid=grid, max_bins=33
    )
    arm_preds["M1x_prior_feature"] = arms_out["M1x_prior_feature"]["preds_te"]

    # Arm 5: M1x calibrated logit fusion on val (explicitly absorbs base-rate shift!)
    arms_out["M1x_logit_fusion"] = build_logit_fusion_arm(arms_out["F0_aligned"], F_va, y_va, F_te, y_te, cfg["seeds"])
    arm_preds["M1x_logit_fusion"] = arms_out["M1x_logit_fusion"]["preds_te"]

    for k, r in arms_out.items():
        log(f"  IEEE-M1x {k:18s}: AUC={r['summary']['auc_mean']:.4f}±{r['summary']['auc_std']:.4f} "
            f"PR={r['summary']['pr_auc_mean']:.4f} Bag6={r['bagging6_metrics']['auc']:.4f}")

    contrasts = run_contrasts(
        y_te,
        arm_preds,
        [
            ("L0_plus_unaligned - L0_aligned", "L0_aligned", "L0_plus_unaligned"),
            ("F0_aligned - L0_aligned", "L0_aligned", "F0_aligned"),
            ("F0_aligned - L0_plus_unaligned", "L0_plus_unaligned", "F0_aligned"),
            ("M1x_prior_feature - F0_aligned", "F0_aligned", "M1x_prior_feature"),
            ("M1x_logit_fusion - F0_aligned", "F0_aligned", "M1x_logit_fusion"),
            ("M1x_prior_feature - L0_plus_unaligned", "L0_plus_unaligned", "M1x_prior_feature"),
            ("M1x_logit_fusion - L0_plus_unaligned", "L0_plus_unaligned", "M1x_logit_fusion"),
        ],
        cfg,
        cluster_ids=cids_te,
    )
    for c in contrasts:
        log(f"  Contrast {c['contrast']:38s}: ΔAUC={c['delta_mean']:+.4f} "
            f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}] p_holm={c['p_holm']:.4f}")

    # Export 12k/6k slice for real SecretFlow SGB confirmation (top 60 tx cols + all hk + 3 prior cols)
    sgb_tr_i, _ = train_test_split(np.arange(len(y_tr)), train_size=12000, stratify=y_tr, random_state=cfg["split_seed"])
    sgb_te_i, _ = train_test_split(np.arange(len(y_te)), train_size=6000, stratify=y_te, random_state=cfg["split_seed"])
    tx_keep = min(60, X_tx_tr.shape[1])
    np.savez_compressed(
        os.path.join(CACHE_DIR, "exp3_sgb_ieee.npz"),
        tx_tr=X_tx_tr[sgb_tr_i, :tx_keep], tx_te=X_tx_te[sgb_te_i, :tx_keep],
        hk_tr=X_hk_tr[sgb_tr_i], hk_te=X_hk_te[sgb_te_i],
        prior_tr=F_tr[sgb_tr_i], prior_te=F_te[sgb_te_i],
        y_tr=y_tr[sgb_tr_i], y_te=y_te[sgb_te_i],
    )

    return {
        "dataset": "IEEE-CIS Real Unaligned Transactions (strictly causal <= T_tr_end)",
        "counts": {
            "aligned_train": int(len(y_tr)),
            "aligned_val": int(len(y_va)),
            "aligned_test": int(len(y_te)),
            "causal_unaligned_train": int(len(y_un)),
            "fraud_rate_aligned_tr": float(y_tr.mean()),
            "fraud_rate_unaligned_tr": float(y_un.mean()),
            "fraud_rate_test": float(y_te.mean()),
        },
        "arms": arms_out,
        "contrasts": contrasts,
    }


def run_home_credit_low_overlap(cfg: dict) -> dict:
    log("=== 2/2 Home Credit 低重合率（20% / 50%）真实非对齐池 M1x 验证 ===")
    df = pd.read_parquet(os.path.join(CACHE_DIR, "hc_master_features.parquet"))
    with open(os.path.join(CACHE_DIR, "hc_feature_groups.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    y_all = df["TARGET"].to_numpy(dtype=np.int32)
    idx_all = np.arange(len(df))
    # Freeze 60,000 test customers once for both overlap regimes
    idx_pool, idx_te = train_test_split(idx_all, test_size=60_000, stratify=y_all, random_state=cfg["split_seed"])

    out_regimes = {}
    for cov_name, overlap_frac in [("overlap_20pct", 0.20), ("overlap_50pct", 0.50)]:
        log(f"  --- Home Credit Regime: {cov_name} (simulated PSI overlap on full 247.5k pool) ---")
        idx_al, idx_un = train_test_split(
            idx_pool, train_size=overlap_frac, stratify=y_all[idx_pool], random_state=cfg["split_seed"] + int(overlap_frac * 100)
        )
        idx_tr, idx_va = train_test_split(idx_al, test_size=0.25, stratify=y_all[idx_al], random_state=cfg["split_seed"] + 7)

        d_un, d_tr, d_va, d_te = df.iloc[idx_un], df.iloc[idx_tr], df.iloc[idx_va], df.iloc[idx_te]
        y_un, y_tr, y_va, y_te = y_all[idx_un], y_all[idx_tr], y_all[idx_va], y_all[idx_te]

        # Encode active features fitted on pooled (unaligned + aligned_tr)
        d_n_pool = pd.concat([d_un[meta["neidi_cols"]], d_tr[meta["neidi_cols"]]], ignore_index=True)
        X_n_pool, X_n_va, X_n_te, _ = encode_tabular_train_val_test(d_n_pool, d_va, d_te, meta["neidi_cols"])
        n_un = len(d_un)
        X_n_un, X_n_tr = X_n_pool[:n_un], X_n_pool[n_un:]
        X_hk_tr, X_hk_va, X_hk_te, _ = encode_tabular_train_val_test(d_tr, d_va, d_te, meta["hk_cols"])

        F_tr, F_va, F_te = build_local_priors(X_n_un, y_un, X_n_tr, y_tr, X_n_va, X_n_te, seed=cfg["split_seed"])

        arms_out, arm_preds = {}, {}
        arms_out["L0_aligned"] = equal_tune_and_eval_hgb(X_n_tr, y_tr, X_n_va, y_va, X_n_te, y_te, seeds=cfg["seeds"], max_bins=33)
        arm_preds["L0_aligned"] = arms_out["L0_aligned"]["preds_te"]

        arms_out["L0_plus_unaligned"] = equal_tune_and_eval_hgb(
            np.vstack([X_n_un, X_n_tr]), np.concatenate([y_un, y_tr]),
            X_n_va, y_va, X_n_te, y_te, seeds=cfg["seeds"], max_bins=33
        )
        arm_preds["L0_plus_unaligned"] = arms_out["L0_plus_unaligned"]["preds_te"]

        arms_out["F0_aligned"] = equal_tune_and_eval_hgb(
            np.hstack([X_n_tr, X_hk_tr]), y_tr,
            np.hstack([X_n_va, X_hk_va]), y_va,
            np.hstack([X_n_te, X_hk_te]), y_te, seeds=cfg["seeds"], max_bins=33
        )
        arm_preds["F0_aligned"] = arms_out["F0_aligned"]["preds_te"]

        arms_out["M1x_prior_feature"] = equal_tune_and_eval_hgb(
            np.hstack([X_n_tr, F_tr, X_hk_tr]), y_tr,
            np.hstack([X_n_va, F_va, X_hk_va]), y_va,
            np.hstack([X_n_te, F_te, X_hk_te]), y_te, seeds=cfg["seeds"], max_bins=33
        )
        arm_preds["M1x_prior_feature"] = arms_out["M1x_prior_feature"]["preds_te"]

        arms_out["M1x_logit_fusion"] = build_logit_fusion_arm(arms_out["F0_aligned"], F_va, y_va, F_te, y_te, cfg["seeds"])
        arm_preds["M1x_logit_fusion"] = arms_out["M1x_logit_fusion"]["preds_te"]

        for k, r in arms_out.items():
            log(f"    HC-{cov_name} {k:18s}: AUC={r['summary']['auc_mean']:.4f}±{r['summary']['auc_std']:.4f}")

        contrasts = run_contrasts(
            y_te,
            arm_preds,
            [
                ("L0_plus_unaligned - L0_aligned", "L0_aligned", "L0_plus_unaligned"),
                ("F0_aligned - L0_aligned", "L0_aligned", "F0_aligned"),
                ("M1x_prior_feature - F0_aligned", "F0_aligned", "M1x_prior_feature"),
                ("M1x_logit_fusion - F0_aligned", "F0_aligned", "M1x_logit_fusion"),
            ],
            cfg,
        )
        for c in contrasts:
            log(f"    HC-{cov_name} {c['contrast']:32s}: ΔAUC={c['delta_mean']:+.4f} "
                f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}] p_holm={c['p_holm']:.4f}")

        if cov_name == "overlap_20pct":
            sgb_tr_i, _ = train_test_split(np.arange(len(y_tr)), train_size=12000, stratify=y_tr, random_state=cfg["split_seed"])
            sgb_te_i, _ = train_test_split(np.arange(len(y_te)), train_size=6000, stratify=y_te, random_state=cfg["split_seed"])
            np.savez_compressed(
                os.path.join(CACHE_DIR, "exp3_sgb_hc20.npz"),
                neidi_tr=X_n_tr[sgb_tr_i], neidi_te=X_n_te[sgb_te_i],
                hk_tr=X_hk_tr[sgb_tr_i], hk_te=X_hk_te[sgb_te_i],
                prior_tr=F_tr[sgb_tr_i], prior_te=F_te[sgb_te_i],
                y_tr=y_tr[sgb_tr_i], y_te=y_te[sgb_te_i],
            )

        out_regimes[cov_name] = {
            "note": f"【构造低重合条件】从全量 247,511 训练池按 {int(overlap_frac*100)}% 抽样交集，其余作为主动方真实非对齐客户",
            "counts": {"aligned_train": int(len(y_tr)), "aligned_val": int(len(y_va)), "unaligned_train": int(len(y_un)), "test": int(len(y_te))},
            "arms": arms_out,
            "contrasts": contrasts,
        }

    return out_regimes


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config()
    out = {
        "experiment": "Exp3_M1x_Unaligned_CrossDataset",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": cfg["seeds"],
        "split_seed": cfg["split_seed"],
        "engine": cfg["engines"]["proxy_model"],
        "ieee_cis_real_unaligned": run_ieee_unaligned(cfg),
        "home_credit_low_overlap": run_home_credit_low_overlap(cfg),
        "total_elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    save_json(out, os.path.join(OUT_DIR, "exp3_m1x_unaligned.json"))
    log(f"Exp 3 finished in {out['total_elapsed_sec']:.1f}s -> results/exp3_m1x_unaligned.json")


if __name__ == "__main__":
    main()
