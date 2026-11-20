"""
实验 5（Exp 5）：C2 种子客群扩展（Lookalike 相似度特征）跨 5 数据集首次系统实证。
来源：《后续计划_联邦×同业方法MVP_2026-11_v0.2.md》高优先级同业获客候选，此前零本地实证。

设计纪律：
- 严格只在主动方（标签方）本地特征上计算，零跨方原始特征交互、不改联邦协议。
- 严格防标签泄漏：训练集采用 5 折折外（OOF）构造，val/test 仅用全量训练集正负种子构造。
- 8 维 Lookalike 特征：
  1) cos_pos_centroid: 与正例种子质心的余弦相似度
  2) cos_neg_centroid: 与负例质心的余弦相似度
  3) cos_contrast: cos_pos - cos_neg
  4) dist_pos_centroid: 与正例种子质心的欧氏距离
  5) min_dist_pos_cluster: 与 5 个正例细分客群簇（MiniBatchKMeans）的最近欧氏距离
  6) max_cos_pos_cluster: 与 5 个正例细分客群簇的最大余弦相似度
  7) knn30_pos_rate_smooth: 30 近邻（余弦距离）贝叶斯平滑局部正例率
  8) knn10_pos_neg_dist_ratio: 到最近 10 个正例种子均距 ÷ 到最近 10 个负例均距
- 跨 5 个异构业务数据集：Home Credit（信贷）、Santander（银行交叉销售）、COIL2000（保险获客）、
  Orange Small Churn（电信流失，50k×230）、Hillstrom（邮件营销 visit，64k）。
- 6 seeds 等调参代理 + 配对 Bootstrap 95% CI + Holm-Bonferroni 校正。
"""
from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    STAGE3_ROOT,
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
    print(f"[{time.strftime('%H:%M:%S')}] [Exp5] {msg}", flush=True)


def _fit_transform_lookalike_block(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_eval: np.ndarray,
    seed: int = 2026,
    max_knn_ref: int = 15000,
) -> np.ndarray:
    """Fit lookalike prototypes & kNN strictly on (X_fit, y_fit) and score X_eval."""
    # Impute NaN with fold-train median and standardize
    med = np.nanmedian(X_fit, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xf = np.where(np.isnan(X_fit), med[None, :], X_fit).astype(np.float64)
    Xe = np.where(np.isnan(X_eval), med[None, :], X_eval).astype(np.float64)

    scaler = StandardScaler()
    Xf_s = np.clip(scaler.fit_transform(Xf), -6.0, 6.0)
    Xe_s = np.clip(scaler.transform(Xe), -6.0, 6.0)

    # Row L2-normalized copy for cosine
    nf = np.linalg.norm(Xf_s, axis=1, keepdims=True) + 1e-8
    ne = np.linalg.norm(Xe_s, axis=1, keepdims=True) + 1e-8
    Xf_u = Xf_s / nf
    Xe_u = Xe_s / ne

    pos_mask = y_fit == 1
    neg_mask = y_fit == 0
    prior = float(pos_mask.mean())

    # 1. Global positive/negative centroids
    mu_pos = Xf_s[pos_mask].mean(axis=0)
    mu_neg = Xf_s[neg_mask].mean(axis=0)
    mu_pos_u = mu_pos / (np.linalg.norm(mu_pos) + 1e-8)
    mu_neg_u = mu_neg / (np.linalg.norm(mu_neg) + 1e-8)

    cos_pos = Xe_u @ mu_pos_u
    cos_neg = Xe_u @ mu_neg_u
    cos_contrast = cos_pos - cos_neg
    dist_pos = np.linalg.norm(Xe_s - mu_pos[None, :], axis=1)

    # 2. Multi-prototype positive clusters (K=5)
    X_pos_s = Xf_s[pos_mask]
    k_cl = min(5, max(2, len(X_pos_s) // 20))
    km = MiniBatchKMeans(n_clusters=k_cl, random_state=seed, n_init=3, batch_size=512)
    km.fit(X_pos_s)
    centers = km.cluster_centers_
    centers_u = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
    # Pairwise distances & cosines to cluster centers
    dists_cl = np.sqrt(np.maximum(0.0, ((Xe_s[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)))
    min_dist_cl = dists_cl.min(axis=1)
    max_cos_cl = (Xe_u @ centers_u.T).max(axis=1)

    # 3. k-NN local density & pos/neg distance ratio (stratified subsample reference if huge)
    rng = np.random.default_rng(seed)
    if len(Xf_u) > max_knn_ref:
        ref_idx = rng.choice(len(Xf_u), size=max_knn_ref, replace=False)
        Xref_u, yref = Xf_u[ref_idx], y_fit[ref_idx]
    else:
        Xref_u, yref = Xf_u, y_fit

    nn_all = NearestNeighbors(n_neighbors=min(30, len(Xref_u)), metric="cosine", algorithm="brute", n_jobs=-1)
    nn_all.fit(Xref_u)
    _, nbr_idx = nn_all.kneighbors(Xe_u)
    # Smoothed local k-NN positive rate (m=10 prior pseudocount)
    knn_pos_rate = (yref[nbr_idx].sum(axis=1) + 10.0 * prior) / (nbr_idx.shape[1] + 10.0)

    # Distance ratio to top-10 positive seeds vs top-10 negative seeds
    Xref_pos = Xref_u[yref == 1]
    Xref_neg = Xref_u[yref == 0]
    k_pn = min(10, len(Xref_pos), len(Xref_neg))
    nn_p = NearestNeighbors(n_neighbors=k_pn, metric="cosine", algorithm="brute", n_jobs=-1).fit(Xref_pos)
    nn_n = NearestNeighbors(n_neighbors=k_pn, metric="cosine", algorithm="brute", n_jobs=-1).fit(Xref_neg)
    dp, _ = nn_p.kneighbors(Xe_u)
    dn, _ = nn_n.kneighbors(Xe_u)
    dist_ratio_pn = (dp.mean(axis=1) + 1e-4) / (dn.mean(axis=1) + 1e-4)

    return np.column_stack([
        cos_pos, cos_neg, cos_contrast, dist_pos,
        min_dist_cl, max_cos_cl, knn_pos_rate, dist_ratio_pn,
    ]).astype(np.float32)


def build_c2_oof_features(
    Xa_tr: np.ndarray,
    y_tr: np.ndarray,
    Xa_va: np.ndarray,
    Xa_te: np.ndarray,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strictly leakage-free 5-fold OOF Lookalike features on active party."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    C_tr = np.empty((len(y_tr), 8), dtype=np.float32)
    for k, (fi_tr, fi_va) in enumerate(skf.split(Xa_tr, y_tr)):
        C_tr[fi_va] = _fit_transform_lookalike_block(Xa_tr[fi_tr], y_tr[fi_tr], Xa_tr[fi_va], seed=seed + k + 1)
    C_va = _fit_transform_lookalike_block(Xa_tr, y_tr, Xa_va, seed=seed + 99)
    C_te = _fit_transform_lookalike_block(Xa_tr, y_tr, Xa_te, seed=seed + 99)
    return C_tr, C_va, C_te


def eval_dataset_c2(
    name: str,
    Xa_tr, Xa_va, Xa_te,
    Xb_tr, Xb_va, Xb_te,
    y_tr, y_va, y_te,
    cfg: dict,
    refit_train_val: bool = False,
) -> dict:
    log(f"=== C2 Lookalike on {name} (tr={len(y_tr)}, va={len(y_va)}, te={len(y_te)}, pos={y_te.mean():.4f}) ===")
    t0 = time.perf_counter()
    C_tr, C_va, C_te = build_c2_oof_features(Xa_tr, y_tr, Xa_va, Xa_te, seed=cfg["split_seed"])
    c2_sec = time.perf_counter() - t0
    log(f"  Built 8-dim OOF C2 features in {c2_sec:.1f}s")

    arm_specs = {
        "L0": (Xa_tr, Xa_va, Xa_te),
        "L0_plus_C2": (np.hstack([Xa_tr, C_tr]), np.hstack([Xa_va, C_va]), np.hstack([Xa_te, C_te])),
        "F0": (np.hstack([Xa_tr, Xb_tr]), np.hstack([Xa_va, Xb_va]), np.hstack([Xa_te, Xb_te])),
        "F0_plus_C2": (
            np.hstack([Xa_tr, C_tr, Xb_tr]),
            np.hstack([Xa_va, C_va, Xb_va]),
            np.hstack([Xa_te, C_te, Xb_te]),
        ),
    }

    arms_out, arm_preds = {}, {}
    for k, (Xtr, Xva, Xte) in arm_specs.items():
        res = equal_tune_and_eval_hgb(Xtr, y_tr, Xva, y_va, Xte, y_te, seeds=cfg["seeds"], max_bins=33, refit_train_val=refit_train_val)
        arms_out[k] = res
        arm_preds[k] = res["preds_te"]
        log(f"  {name:16s} {k:12s}: AUC={res['summary']['auc_mean']:.4f}±{res['summary']['auc_std']:.4f} "
            f"PR={res['summary']['pr_auc_mean']:.4f} Top5Lift={res['summary']['top5_lift_mean']:.3f}")

    pairs = [
        ("L0_plus_C2 - L0", "L0", "L0_plus_C2"),
        ("F0 - L0", "L0", "F0"),
        ("F0_plus_C2 - F0", "F0", "F0_plus_C2"),
    ]
    contrasts = []
    for cname, bk, nk in pairs:
        c = paired_bootstrap_delta(y_te, arm_preds[bk], arm_preds[nk], n_boot=cfg["bootstrap_iters"], seed=cfg["bootstrap_seed"])
        c["contrast"] = cname
        contrasts.append(c)
    p_adj = holm_bonferroni([c["p_raw"] for c in contrasts])
    for c, pa in zip(contrasts, p_adj):
        c["p_holm"] = float(pa)
        c["significant_holm_05"] = bool(pa < cfg["alpha"])
        log(f"  {name:16s} {c['contrast']:18s}: ΔAUC={c['delta_mean']:+.4f} "
            f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}] p_holm={c['p_holm']:.4f}")

    return {
        "dataset": name,
        "c2_feature_build_sec": round(c2_sec, 2),
        "active_dim": int(Xa_tr.shape[1]),
        "passive_dim": int(Xb_tr.shape[1]),
        "pos_rate_test": float(y_te.mean()),
        "arms": arms_out,
        "contrasts": contrasts,
    }


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config()
    results = {}

    # 1. Home Credit (60k cohort: 36k/12k/12k)
    hc = pd.read_parquet(os.path.join(CACHE_DIR, "hc_master_features.parquet"))
    with open(os.path.join(CACHE_DIR, "hc_feature_groups.json"), "r", encoding="utf-8") as f:
        hc_meta = json.load(f)
    si, _ = train_test_split(np.arange(len(hc)), train_size=60_000, stratify=hc["TARGET"].values, random_state=cfg["split_seed"])
    sub = hc.iloc[np.sort(si)].reset_index(drop=True)
    y = sub["TARGET"].to_numpy(dtype=np.int32)
    i_tr, i_tmp = train_test_split(np.arange(len(sub)), test_size=0.40, stratify=y, random_state=cfg["split_seed"])
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=cfg["split_seed"] + 1)
    Xa_tr, Xa_va, Xa_te, _ = encode_tabular_train_val_test(sub.iloc[i_tr], sub.iloc[i_va], sub.iloc[i_te], hc_meta["neidi_cols"])
    Xb_tr, Xb_va, Xb_te, _ = encode_tabular_train_val_test(sub.iloc[i_tr], sub.iloc[i_va], sub.iloc[i_te], hc_meta["hk_cols"])
    results["home_credit"] = eval_dataset_c2("HomeCredit_60k", Xa_tr, Xa_va, Xa_te, Xb_tr, Xb_va, Xb_te, y[i_tr], y[i_va], y[i_te], cfg)

    # 2. Santander Product (60k cohort: 36k/12k/12k)
    san = pd.read_parquet(os.path.join(CACHE_DIR, "santander_multimonth_60k.parquet"))
    with open(os.path.join(CACHE_DIR, "santander_feature_groups.json"), "r", encoding="utf-8") as f:
        san_meta = json.load(f)
    y = san["y_new_any"].to_numpy(dtype=np.int32)
    i_tr, i_tmp = train_test_split(np.arange(len(san)), test_size=0.40, stratify=y, random_state=cfg["split_seed"])
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=cfg["split_seed"] + 1)
    Xa_tr, Xa_va, Xa_te, _ = encode_tabular_train_val_test(san.iloc[i_tr], san.iloc[i_va], san.iloc[i_te], san_meta["neidi_cols"])
    Xb_tr, Xb_va, Xb_te, _ = encode_tabular_train_val_test(san.iloc[i_tr], san.iloc[i_va], san.iloc[i_te], san_meta["hk_cols"] + san_meta["third_cols"])
    results["santander"] = eval_dataset_c2("Santander_60k", Xa_tr, Xa_va, Xa_te, Xb_tr, Xb_va, Xb_te, y[i_tr], y[i_va], y[i_te], cfg)

    # 3. COIL2000 (official 5,822 / 4,000)
    cdir = cfg["datasets"]["coil2000_dir"]
    df_tr = pd.read_csv(os.path.join(cdir, "ticdata2000.txt"), sep="\t", header=None)
    X_ev = pd.read_csv(os.path.join(cdir, "ticeval2000.txt"), sep="\t", header=None).to_numpy(dtype=np.float32)
    y_ev = pd.read_csv(os.path.join(cdir, "tictgts2000.txt"), sep="\t", header=None).iloc[:, 0].to_numpy(dtype=np.int32)
    X_all, y_all = df_tr.iloc[:, :85].to_numpy(dtype=np.float32), df_tr.iloc[:, 85].to_numpy(dtype=np.int32)
    i_tr, i_va = train_test_split(np.arange(len(y_all)), test_size=0.20, stratify=y_all, random_state=cfg["split_seed"])
    results["coil2000"] = eval_dataset_c2(
        "COIL2000",
        X_all[i_tr, :43], X_all[i_va, :43], X_ev[:, :43],
        X_all[i_tr, 43:85], X_all[i_va, 43:85], X_ev[:, 43:85],
        y_all[i_tr], y_all[i_va], y_ev, cfg, refit_train_val=True,
    )

    # 4. Orange Small Churn (50,000 × 230, split numeric/cat cols into constructed 2-party: even vs odd valid cols)
    o_data = pd.read_csv(os.path.join(cfg["datasets"]["orange_small_dir"], "orange_small_train.data"), sep="\t", low_memory=False)
    o_y = pd.read_csv(os.path.join(cfg["datasets"]["orange_labels_dir"], "orange_small_train_churn.labels"), header=None).iloc[:, 0].to_numpy()
    y = (o_y == 1).astype(np.int32)
    # Drop all-NaN / constant columns
    valid_cols = [c for c in o_data.columns if o_data[c].notna().mean() > 0.05 and o_data[c].nunique(dropna=True) > 1]
    a_cols = valid_cols[::2]
    b_cols = valid_cols[1::2]
    i_tr, i_tmp = train_test_split(np.arange(len(o_data)), test_size=0.40, stratify=y, random_state=cfg["split_seed"])
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=cfg["split_seed"] + 1)
    Xa_tr, Xa_va, Xa_te, _ = encode_tabular_train_val_test(o_data.iloc[i_tr], o_data.iloc[i_va], o_data.iloc[i_te], a_cols)
    Xb_tr, Xb_va, Xb_te, _ = encode_tabular_train_val_test(o_data.iloc[i_tr], o_data.iloc[i_va], o_data.iloc[i_te], b_cols)
    results["orange_churn"] = eval_dataset_c2("Orange_Churn_50k", Xa_tr, Xa_va, Xa_te, Xb_tr, Xb_va, Xb_te, y[i_tr], y[i_va], y[i_te], cfg)

    # 5. Hillstrom Visit (64,000 rows)
    h = pd.read_csv(cfg["datasets"]["hillstrom_csv"])
    y = h["visit"].to_numpy(dtype=np.int32)
    i_tr, i_tmp = train_test_split(np.arange(len(h)), test_size=0.40, stratify=y, random_state=cfg["split_seed"])
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=cfg["split_seed"] + 1)
    Xa_tr, Xa_va, Xa_te, _ = encode_tabular_train_val_test(h.iloc[i_tr], h.iloc[i_va], h.iloc[i_te], ["zip_code", "newbie", "channel", "segment"])
    Xb_tr, Xb_va, Xb_te, _ = encode_tabular_train_val_test(h.iloc[i_tr], h.iloc[i_va], h.iloc[i_te], ["recency", "history_segment", "history", "mens", "womens"])
    results["hillstrom_visit"] = eval_dataset_c2("Hillstrom_Visit", Xa_tr, Xa_va, Xa_te, Xb_tr, Xb_va, Xb_te, y[i_tr], y[i_va], y[i_te], cfg)

    out = {
        "experiment": "Exp5_C2_SeedLookalike_CrossDataset",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": cfg["seeds"],
        "split_seed": cfg["split_seed"],
        "engine": cfg["engines"]["proxy_model"],
        "datasets": results,
        "total_elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    save_json(out, os.path.join(OUT_DIR, "exp5_c2_seed_lookalike.json"))
    log(f"Exp 5 finished in {out['total_elapsed_sec']:.1f}s -> results/exp5_c2_seed_lookalike.json")


if __name__ == "__main__":
    main()
