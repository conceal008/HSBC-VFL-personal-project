"""
实验 2（Exp 2）：X1 本地图特征与 X1+ 因果时序聚集/速度特征跨数据集验证（6-seed 等调参代理 + 配对 Bootstrap CI + Holm 校正）。
同时导出精简 .npz 供 exp2_sgb_confirm.py 在真实 SecretFlow 1.14 SGB 上复核！

验证场景：
1. Home Credit（100,000 客户，60k/20k/20k 冻结切分）：
   - 检验 X1+ 严格因果历史时序速度（近 90/180/365d 申请增速、拒贷恶化、还款延迟加速、近 6m 信用卡额度使用趋势）
     叠加在 F0(两方)、F3_pay_static(三方静态)、F4_static(四方静态) 上的净增量。
2. Santander 产品推荐（60,000 客户，2015-11..2016-04 严格历史 -> 2016-05 新购目标）：
   - 检验 X1+ 严格因果月度时序速度（各参与方仅在本地计算近 1/3/6m 活跃度/产品变动动量，零跨方明文交换）
     叠加在 L0、F0(两方静态)、F3(三方静态) 上的净增量。
3. IEEE-CIS 滚动三时间窗（Window 1 / 2 / 3，全交集 144,233 笔严格按 TransactionDT 排序）：
   - 正面关闭 Phase 2 v0.4 报告留下的「单测试窗 + 无真 SGB」局限：
     在 3 个连续不重叠未来测试窗上验证 L0 / F0 / F_x1_graph(因果图) / F_x1plus_vel(因果时间窗+uid) / F_x1_full 的跨期稳定性！
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
    print(f"[{time.strftime('%H:%M:%S')}] [Exp2] {msg}", flush=True)


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


def run_home_credit_x1plus(cfg: dict) -> dict:
    log("=== 1/3 Home Credit X1+ 因果历史时序速度 ===")
    df = pd.read_parquet(os.path.join(CACHE_DIR, "hc_master_features.parquet"))
    with open(os.path.join(CACHE_DIR, "hc_feature_groups.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    sub_idx, _ = train_test_split(
        np.arange(len(df)),
        train_size=100_000,
        stratify=df["TARGET"].values,
        random_state=cfg["split_seed"],
    )
    sub = df.iloc[np.sort(sub_idx)].reset_index(drop=True)
    y = sub["TARGET"].to_numpy(dtype=np.int32)
    idx_all = np.arange(len(sub))
    idx_tr, idx_tmp = train_test_split(idx_all, test_size=0.40, stratify=y, random_state=cfg["split_seed"])
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=cfg["split_seed"] + 1)

    df_tr, df_va, df_te = sub.iloc[idx_tr], sub.iloc[idx_va], sub.iloc[idx_te]
    y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]

    X_neidi_tr, X_neidi_va, X_neidi_te, _ = encode_tabular_train_val_test(df_tr, df_va, df_te, meta["neidi_cols"])
    X_hk_tr, X_hk_va, X_hk_te, _ = encode_tabular_train_val_test(df_tr, df_va, df_te, meta["hk_cols"])
    X_bur_tr, X_bur_va, X_bur_te, _ = encode_tabular_train_val_test(df_tr, df_va, df_te, meta["bureau_cols"])
    X_pay_tr, X_pay_va, X_pay_te, _ = encode_tabular_train_val_test(df_tr, df_va, df_te, meta["pay_cols"])
    X_vel_tr, X_vel_va, X_vel_te, nm_vel = encode_tabular_train_val_test(df_tr, df_va, df_te, meta["x1_velocity_cols"])

    arm_specs = {
        "F0_2party": (
            np.hstack([X_neidi_tr, X_hk_tr]),
            np.hstack([X_neidi_va, X_hk_va]),
            np.hstack([X_neidi_te, X_hk_te]),
        ),
        "F0_plus_x1v": (
            np.hstack([X_neidi_tr, X_hk_tr, X_vel_tr]),
            np.hstack([X_neidi_va, X_hk_va, X_vel_va]),
            np.hstack([X_neidi_te, X_hk_te, X_vel_te]),
        ),
        "F3_pay_static": (
            np.hstack([X_neidi_tr, X_hk_tr, X_pay_tr]),
            np.hstack([X_neidi_va, X_hk_va, X_pay_va]),
            np.hstack([X_neidi_te, X_hk_te, X_pay_te]),
        ),
        "F3_pay_plus_x1v": (
            np.hstack([X_neidi_tr, X_hk_tr, X_pay_tr, X_vel_tr]),
            np.hstack([X_neidi_va, X_hk_va, X_pay_va, X_vel_va]),
            np.hstack([X_neidi_te, X_hk_te, X_pay_te, X_vel_te]),
        ),
        "F4_static": (
            np.hstack([X_neidi_tr, X_hk_tr, X_bur_tr, X_pay_tr]),
            np.hstack([X_neidi_va, X_hk_va, X_bur_va, X_pay_va]),
            np.hstack([X_neidi_te, X_hk_te, X_bur_te, X_pay_te]),
        ),
        "F4_plus_x1v": (
            np.hstack([X_neidi_tr, X_hk_tr, X_bur_tr, X_pay_tr, X_vel_tr]),
            np.hstack([X_neidi_va, X_hk_va, X_bur_va, X_pay_va, X_vel_va]),
            np.hstack([X_neidi_te, X_hk_te, X_bur_te, X_pay_te, X_vel_te]),
        ),
    }

    arms_out, arm_preds = {}, {}
    for k, (Xtr, Xva, Xte) in arm_specs.items():
        res = equal_tune_and_eval_hgb(Xtr, y_tr, Xva, y_va, Xte, y_te, seeds=cfg["seeds"], max_bins=33)
        arms_out[k] = res
        arm_preds[k] = res["preds_te"]
        log(f"  HC-X1+ {k:16s}: AUC={res['summary']['auc_mean']:.4f}±{res['summary']['auc_std']:.4f} "
            f"PR={res['summary']['pr_auc_mean']:.4f} ({res['elapsed_sec']:.1f}s)")

    contrasts = run_contrasts(
        y_te,
        arm_preds,
        [
            ("F0_plus_x1v - F0_2party", "F0_2party", "F0_plus_x1v"),
            ("F3_pay_plus_x1v - F3_pay_static", "F3_pay_static", "F3_pay_plus_x1v"),
            ("F4_plus_x1v - F4_static", "F4_static", "F4_plus_x1v"),
            ("F4_plus_x1v - F0_2party", "F0_2party", "F4_plus_x1v"),
        ],
        cfg,
    )
    for c in contrasts:
        log(f"  Contrast {c['contrast']:32s}: ΔAUC={c['delta_mean']:+.4f} "
            f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}] p_holm={c['p_holm']:.4f}")

    # Export 12k/6k slice for real SGB confirmation
    sgb_tr_i, _ = train_test_split(np.arange(len(y_tr)), train_size=12000, stratify=y_tr, random_state=cfg["split_seed"])
    sgb_te_i, _ = train_test_split(np.arange(len(y_te)), train_size=6000, stratify=y_te, random_state=cfg["split_seed"])
    np.savez_compressed(
        os.path.join(CACHE_DIR, "exp2_sgb_hc.npz"),
        neidi_tr=X_neidi_tr[sgb_tr_i], neidi_te=X_neidi_te[sgb_te_i],
        hk_tr=X_hk_tr[sgb_tr_i], hk_te=X_hk_te[sgb_te_i],
        pay_tr=X_pay_tr[sgb_tr_i], pay_te=X_pay_te[sgb_te_i],
        vel_tr=X_vel_tr[sgb_tr_i], vel_te=X_vel_te[sgb_te_i],
        y_tr=y_tr[sgb_tr_i], y_te=y_te[sgb_te_i],
    )

    return {
        "dataset": "Home Credit Default Risk (100k cohort, 60k/20k/20k)",
        "x1_velocity_dim": len(nm_vel),
        "x1_velocity_cols": nm_vel,
        "arms": arms_out,
        "contrasts": contrasts,
    }


def run_santander_x1plus(cfg: dict) -> dict:
    log("=== 2/3 Santander 产品推荐 X1+ 因果月度时序速度 ===")
    df = pd.read_parquet(os.path.join(CACHE_DIR, "santander_multimonth_60k.parquet"))
    with open(os.path.join(CACHE_DIR, "santander_feature_groups.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    y = df["y_new_any"].to_numpy(dtype=np.int32)
    idx_all = np.arange(len(df))
    idx_tr, idx_tmp = train_test_split(idx_all, test_size=0.40, stratify=y, random_state=cfg["split_seed"])
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=cfg["split_seed"] + 1)

    df_tr, df_va, df_te = df.iloc[idx_tr], df.iloc[idx_va], df.iloc[idx_te]
    y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]

    X_n_tr, X_n_va, X_n_te, _ = encode_tabular_train_val_test(df_tr, df_va, df_te, meta["neidi_cols"])
    X_h_tr, X_h_va, X_h_te, _ = encode_tabular_train_val_test(df_tr, df_va, df_te, meta["hk_cols"])
    X_t_tr, X_t_va, X_t_te, _ = encode_tabular_train_val_test(df_tr, df_va, df_te, meta["third_cols"])
    V_n_tr, V_n_va, V_n_te, nm_vn = encode_tabular_train_val_test(df_tr, df_va, df_te, meta["x1v_neidi_cols"])
    V_h_tr, V_h_va, V_h_te, nm_vh = encode_tabular_train_val_test(df_tr, df_va, df_te, meta["x1v_hk_cols"])
    V_t_tr, V_t_va, V_t_te, nm_vt = encode_tabular_train_val_test(df_tr, df_va, df_te, meta["x1v_third_cols"])

    arm_specs = {
        "L0_neidi_static": (X_n_tr, X_n_va, X_n_te),
        "L0_plus_x1v_neidi": (
            np.hstack([X_n_tr, V_n_tr]),
            np.hstack([X_n_va, V_n_va]),
            np.hstack([X_n_te, V_n_te]),
        ),
        "F0_2party_static": (
            np.hstack([X_n_tr, X_h_tr]),
            np.hstack([X_n_va, X_h_va]),
            np.hstack([X_n_te, X_h_te]),
        ),
        "F0_plus_x1v_both": (
            np.hstack([X_n_tr, V_n_tr, X_h_tr, V_h_tr]),
            np.hstack([X_n_va, V_n_va, X_h_va, V_h_va]),
            np.hstack([X_n_te, V_n_te, X_h_te, V_h_te]),
        ),
        "F3_static": (
            np.hstack([X_n_tr, X_h_tr, X_t_tr]),
            np.hstack([X_n_va, X_h_va, X_t_va]),
            np.hstack([X_n_te, X_h_te, X_t_te]),
        ),
        "F3_plus_x1v_all": (
            np.hstack([X_n_tr, V_n_tr, X_h_tr, V_h_tr, X_t_tr, V_t_tr]),
            np.hstack([X_n_va, V_n_va, X_h_va, V_h_va, X_t_va, V_t_va]),
            np.hstack([X_n_te, V_n_te, X_h_te, V_h_te, X_t_te, V_t_te]),
        ),
    }

    arms_out, arm_preds = {}, {}
    for k, (Xtr, Xva, Xte) in arm_specs.items():
        res = equal_tune_and_eval_hgb(Xtr, y_tr, Xva, y_va, Xte, y_te, seeds=cfg["seeds"], max_bins=33)
        arms_out[k] = res
        arm_preds[k] = res["preds_te"]
        log(f"  SAN-X1+ {k:18s}: AUC={res['summary']['auc_mean']:.4f}±{res['summary']['auc_std']:.4f} "
            f"PR={res['summary']['pr_auc_mean']:.4f} ({res['elapsed_sec']:.1f}s)")

    contrasts = run_contrasts(
        y_te,
        arm_preds,
        [
            ("L0_x1v - L0_static", "L0_neidi_static", "L0_plus_x1v_neidi"),
            ("F0_x1v_both - F0_static", "F0_2party_static", "F0_plus_x1v_both"),
            ("F3_x1v_all - F3_static", "F3_static", "F3_plus_x1v_all"),
            ("F0_x1v_both - L0_x1v", "L0_plus_x1v_neidi", "F0_plus_x1v_both"),
            ("F3_x1v_all - F0_static", "F0_2party_static", "F3_plus_x1v_all"),
        ],
        cfg,
    )
    for c in contrasts:
        log(f"  Contrast {c['contrast']:28s}: ΔAUC={c['delta_mean']:+.4f} "
            f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}] p_holm={c['p_holm']:.4f}")

    # Export 10k/5k slice for real SGB confirmation
    sgb_tr_i, _ = train_test_split(np.arange(len(y_tr)), train_size=10000, stratify=y_tr, random_state=cfg["split_seed"])
    sgb_te_i, _ = train_test_split(np.arange(len(y_te)), train_size=5000, stratify=y_te, random_state=cfg["split_seed"])
    np.savez_compressed(
        os.path.join(CACHE_DIR, "exp2_sgb_santander.npz"),
        neidi_static_tr=X_n_tr[sgb_tr_i], neidi_static_te=X_n_te[sgb_te_i],
        neidi_x1v_tr=np.hstack([X_n_tr, V_n_tr])[sgb_tr_i], neidi_x1v_te=np.hstack([X_n_te, V_n_te])[sgb_te_i],
        hk_static_tr=X_h_tr[sgb_tr_i], hk_static_te=X_h_te[sgb_te_i],
        hk_x1v_tr=np.hstack([X_h_tr, V_h_tr])[sgb_tr_i], hk_x1v_te=np.hstack([X_h_te, V_h_te])[sgb_te_i],
        y_tr=y_tr[sgb_tr_i], y_te=y_te[sgb_te_i],
    )

    return {
        "dataset": "Santander Product Recommendation (60k cohort, 2015-11..2016-04 history -> 2016-05 target)",
        "x1v_dims": {"neidi": len(nm_vn), "hk_deposit": len(nm_vh), "third_cw": len(nm_vt)},
        "arms": arms_out,
        "contrasts": contrasts,
    }


def _entity_codes(s: pd.Series) -> np.ndarray:
    k, _ = pd.factorize(s.astype("object").where(s.notna(), None), use_na_sentinel=True)
    k = k.astype(np.float64)
    k[k < 0] = np.nan
    return k


def _causal_degree_features(df: pd.DataFrame, entity_col: str, partner_col: str, prefix: str) -> pd.DataFrame:
    """Strictly causal cumulative count & distinct-partner count prior to current row (time-sorted df)."""
    n = len(df)
    out = pd.DataFrame(index=df.index)
    e = df[entity_col]
    valid = e.notna()
    cnt = np.zeros(n, dtype=np.float32)
    cnt[valid] = df.loc[valid].groupby(entity_col).cumcount().to_numpy(dtype=np.float32)
    cnt[~valid] = np.nan
    out[f"{prefix}_prev_cnt"] = np.log1p(cnt)

    # Distinct partners seen strictly before current row
    p = df[partner_col]
    both = valid & p.notna()
    first_pair = np.zeros(n, dtype=np.float32)
    sub = df.loc[both, [entity_col, partner_col]]
    is_first = (~sub.duplicated([entity_col, partner_col])).to_numpy(dtype=np.float32)
    first_pair[both] = is_first
    tmp = pd.DataFrame({"e": e, "fp": first_pair})
    cum_uniq = np.full(n, np.nan, dtype=np.float32)
    cum_uniq[valid] = (tmp.loc[valid].groupby("e")["fp"].cumsum().to_numpy(dtype=np.float32) - first_pair[valid])
    out[f"{prefix}_prev_uniq_{partner_col}"] = cum_uniq
    return out


def _velocity_block(code: np.ndarray, t: np.ndarray, amt: np.ndarray, email: np.ndarray, tag: str) -> pd.DataFrame:
    """Strictly causal 1h/1d/7d velocity + amount ratio + prior distinct emails (rows already time-sorted)."""
    BIG = 1e8
    windows = {"1h": 3600.0, "1d": 86400.0, "7d": 7 * 86400.0}
    n = len(code)
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in ("cnt_1h", "cnt_1d", "cnt_7d", "amt_1d", "dt_prev", "amt_ratio", "n_email")}
    m = ~np.isnan(code)
    idx = np.flatnonzero(m)
    c = code[idx]
    order = idx[np.argsort(c, kind="stable")]
    cs, ts, am = code[order], t[order], amt[order]
    key = cs * BIG + ts
    k = np.arange(len(order))
    gstart = np.r_[0, np.flatnonzero(cs[1:] != cs[:-1]) + 1]
    gs = np.repeat(gstart, np.diff(np.r_[gstart, len(order)]))
    cum = np.r_[0.0, np.cumsum(am)]
    for name, w in windows.items():
        lo = np.searchsorted(key, cs * BIG + (ts - w), side="right")
        lo = np.maximum(lo, gs)
        out["cnt_" + name][order] = (k - lo).astype(np.float32)
        if name == "1d":
            out["amt_1d"][order] = (cum[k] - cum[lo]).astype(np.float32)
    has_prev = k > gs
    dt = np.full(len(order), np.nan, dtype=np.float64)
    dt[has_prev] = ts[has_prev] - ts[np.flatnonzero(has_prev) - 1]
    out["dt_prev"][order] = np.log1p(np.maximum(dt, 0.0)).astype(np.float32)
    prev_n = k - gs
    mean_prev = np.where(has_prev, (cum[k] - cum[gs]) / np.maximum(prev_n, 1), np.nan)
    out["amt_ratio"][order] = np.where(has_prev & (mean_prev > 0), am / np.where(mean_prev > 0, mean_prev, 1.0), np.nan).astype(np.float32)

    em = email[order]
    df_em = pd.DataFrame({"g": cs, "e": em})
    first = ((~df_em.duplicated(["g", "e"])) & df_em["e"].notna()).to_numpy(dtype=np.float32)
    out["n_email"][order] = (pd.Series(first).groupby(df_em["g"]).cumsum().to_numpy(dtype=np.float32) - first)
    return pd.DataFrame({f"vf_{q}__{tag}": v for q, v in out.items()})


def run_ieee_rolling_windows(cfg: dict) -> dict:
    log("=== 3/3 IEEE-CIS 滚动三时间窗复核 X1 因果图 + X1+ 速度 ===")
    ieee_dir = cfg["datasets"]["ieee_cis_dir"]
    tx = pd.read_csv(os.path.join(ieee_dir, "train_transaction.csv"), low_memory=False)
    idn = pd.read_csv(os.path.join(ieee_dir, "train_identity.csv"), low_memory=False)
    df = tx.merge(idn, on="TransactionID", how="inner").sort_values("TransactionDT").reset_index(drop=True)
    del tx, idn
    gc.collect()
    log(f"  IEEE-CIS aligned rows: {len(df)}, fraud rate: {df['isFraud'].mean():.4f}")

    # Compact, representative non-leaky base feature subsets for Mainland (tx) & HK (idn)
    # Keep all non-ID/non-label transaction cols except ultra-sparse V cols (>90% NaN) so 3 windows run fast & cleanly
    tx_base_cols = [
        c for c in df.columns
        if c not in ("TransactionID", "isFraud", "TransactionDT")
        and not c.startswith("id_") and c not in ("DeviceType", "DeviceInfo")
        and df[c].isna().mean() < 0.85
    ]
    hk_base_cols = [c for c in df.columns if c.startswith("id_") or c in ("DeviceType", "DeviceInfo")]

    # X1 causal graph features strictly inside each party (no cross-party raw data sharing!)
    # Active (tx): card1 <-> addr1, card1 <-> P_emaildomain, addr1 <-> card1
    # Passive (idn): DeviceInfo <-> id_30, DeviceInfo <-> id_31
    g_tx = pd.concat([
        _causal_degree_features(df, "card1", "addr1", "xg_card1"),
        _causal_degree_features(df, "card1", "P_emaildomain", "xg_card1em"),
        _causal_degree_features(df, "addr1", "card1", "xg_addr1"),
        _causal_degree_features(df, "P_emaildomain", "card1", "xg_email"),
    ], axis=1)
    g_hk = pd.concat([
        _causal_degree_features(df, "DeviceInfo", "id_31", "xh_dev"),
        _causal_degree_features(df, "id_31", "DeviceInfo", "xh_browser"),
    ], axis=1)

    # X1+ causal temporal velocity features strictly on active party (tx)
    t_arr = df["TransactionDT"].to_numpy(np.float64)
    amt_arr = df["TransactionAmt"].to_numpy(np.float64)
    em_code = _entity_codes(df["P_emaildomain"])
    day = np.floor(t_arr / 86400.0 - df["D1"].to_numpy(np.float64))
    ok_uid = df["card1"].notna() & df["addr1"].notna() & df["D1"].notna()
    uid_s = (df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_" + pd.Series(day).astype(str)).where(ok_uid, None)
    card_addr_s = (df["card1"].astype(str) + "_" + df["addr1"].astype(str)).where(df["card1"].notna() & df["addr1"].notna(), None)
    v_tx = pd.concat([
        _velocity_block(_entity_codes(df["card1"]), t_arr, amt_arr, em_code, "card1"),
        _velocity_block(_entity_codes(card_addr_s), t_arr, amt_arr, em_code, "card_addr"),
        _velocity_block(_entity_codes(uid_s), t_arr, amt_arr, em_code, "uid"),
    ], axis=1)

    keep_meta = [c for c in ("isFraud", "TransactionDT", "card1") if c not in tx_base_cols and c not in hk_base_cols]
    df_full = pd.concat([df[keep_meta + tx_base_cols + hk_base_cols], g_tx, g_hk, v_tx], axis=1)
    df_full = df_full.loc[:, ~df_full.columns.duplicated()]
    g_tx_cols = list(g_tx.columns)
    g_hk_cols = list(g_hk.columns)
    v_tx_cols = list(v_tx.columns)

    n = len(df_full)
    # 3 non-overlapping chronological rolling test windows
    windows = [
        ("W1_55_70", (0, int(0.45 * n)), (int(0.45 * n), int(0.55 * n)), (int(0.55 * n), int(0.70 * n))),
        ("W2_70_85", (0, int(0.60 * n)), (int(0.60 * n), int(0.70 * n)), (int(0.70 * n), int(0.85 * n))),
        ("W3_85_100", (0, int(0.75 * n)), (int(0.75 * n), int(0.85 * n)), (int(0.85 * n), n)),
    ]

    window_results = {}
    for wname, (tr0, tr1), (va0, va1), (te0, te1) in windows:
        log(f"  --- Rolling Window {wname}: tr=[{tr0}:{tr1}], va=[{va0}:{va1}], te=[{te0}:{te1}] ---")
        dtr, dva, dte = df_full.iloc[tr0:tr1], df_full.iloc[va0:va1], df_full.iloc[te0:te1]
        y_tr = dtr["isFraud"].to_numpy(dtype=np.int32)
        y_va = dva["isFraud"].to_numpy(dtype=np.int32)
        y_te = dte["isFraud"].to_numpy(dtype=np.int32)
        # Cluster IDs by card1 (fallback to unique negative row index if NaN) for entity-clustered bootstrap!
        c1_col = dte["card1"]
        if isinstance(c1_col, pd.DataFrame):
            c1_col = c1_col.iloc[:, 0]
        card1_vals = c1_col.to_numpy(dtype=np.float64)
        cids_te = np.where(np.isnan(card1_vals), -np.arange(1, len(dte) + 1, dtype=np.float64), card1_vals)

        X_tx_tr, X_tx_va, X_tx_te, _ = encode_tabular_train_val_test(dtr, dva, dte, tx_base_cols)
        X_hk_tr, X_hk_va, X_hk_te, _ = encode_tabular_train_val_test(dtr, dva, dte, hk_base_cols)
        G_tx_tr, G_tx_va, G_tx_te, _ = encode_tabular_train_val_test(dtr, dva, dte, g_tx_cols)
        G_hk_tr, G_hk_va, G_hk_te, _ = encode_tabular_train_val_test(dtr, dva, dte, g_hk_cols)
        V_tx_tr, V_tx_va, V_tx_te, _ = encode_tabular_train_val_test(dtr, dva, dte, v_tx_cols)

        arm_specs = {
            "L0_tx": (X_tx_tr, X_tx_va, X_tx_te),
            "F0_2party": (np.hstack([X_tx_tr, X_hk_tr]), np.hstack([X_tx_va, X_hk_va]), np.hstack([X_tx_te, X_hk_te])),
            "F_x1_graph": (
                np.hstack([X_tx_tr, G_tx_tr, X_hk_tr, G_hk_tr]),
                np.hstack([X_tx_va, G_tx_va, X_hk_va, G_hk_va]),
                np.hstack([X_tx_te, G_tx_te, X_hk_te, G_hk_te]),
            ),
            "F_x1plus_vel": (
                np.hstack([X_tx_tr, V_tx_tr, X_hk_tr]),
                np.hstack([X_tx_va, V_tx_va, X_hk_va]),
                np.hstack([X_tx_te, V_tx_te, X_hk_te]),
            ),
            "F_x1_full": (
                np.hstack([X_tx_tr, G_tx_tr, V_tx_tr, X_hk_tr, G_hk_tr]),
                np.hstack([X_tx_va, G_tx_va, V_tx_va, X_hk_va, G_hk_va]),
                np.hstack([X_tx_te, G_tx_te, V_tx_te, X_hk_te, G_hk_te]),
            ),
        }

        # Slightly tighter grid (4 configs) to keep 3 rolling windows × 5 arms × 6 seeds fast (~2 min total)
        ieee_grid = [
            {"learning_rate": 0.05, "max_depth": 6, "l2_regularization": 5.0, "max_iter": 120},
            {"learning_rate": 0.10, "max_depth": 6, "l2_regularization": 5.0, "max_iter": 100},
            {"learning_rate": 0.05, "max_depth": 4, "l2_regularization": 10.0, "max_iter": 150},
            {"learning_rate": 0.10, "max_depth": 4, "l2_regularization": 5.0, "max_iter": 120},
        ]

        arms_out, arm_preds = {}, {}
        for k, (Xtr, Xva, Xte) in arm_specs.items():
            res = equal_tune_and_eval_hgb(Xtr, y_tr, Xva, y_va, Xte, y_te, seeds=cfg["seeds"], grid=ieee_grid, max_bins=33)
            arms_out[k] = res
            arm_preds[k] = res["preds_te"]
            log(f"    {wname} {k:14s}: AUC={res['summary']['auc_mean']:.4f}±{res['summary']['auc_std']:.4f} "
                f"PR={res['summary']['pr_auc_mean']:.4f} ({res['elapsed_sec']:.1f}s)")

        contrasts = run_contrasts(
            y_te,
            arm_preds,
            [
                ("F0 - L0", "L0_tx", "F0_2party"),
                ("F_x1_graph - F0", "F0_2party", "F_x1_graph"),
                ("F_x1plus_vel - F0", "F0_2party", "F_x1plus_vel"),
                ("F_x1_full - F0", "F0_2party", "F_x1_full"),
                ("F_x1_full - F_x1_graph", "F_x1_graph", "F_x1_full"),
            ],
            cfg,
            cluster_ids=cids_te,
        )
        for c in contrasts:
            log(f"    {wname} {c['contrast']:24s}: ΔAUC={c['delta_mean']:+.4f} "
                f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}] p_holm={c['p_holm']:.4f}")

        window_results[wname] = {
            "days_test": [float(dte["TransactionDT"].min() / 86400.0), float(dte["TransactionDT"].max() / 86400.0)],
            "pos_rate_test": float(y_te.mean()),
            "n_train": int(len(y_tr)),
            "n_val": int(len(y_va)),
            "n_test": int(len(y_te)),
            "arms": arms_out,
            "contrasts": contrasts,
        }

        # For Window 3 (most recent future test window), export a compact 12k/6k slice for real SecretFlow SGB confirmation!
        if wname == "W3_85_100":
            # Take top 60 highest-variance/non-missing tx cols + all 40 hk cols so SGB runs cleanly in ~60s
            # Keep exact same base + graph + velocity split
            sgb_tr_i, _ = train_test_split(np.arange(len(y_tr)), train_size=12000, stratify=y_tr, random_state=cfg["split_seed"])
            sgb_te_i, _ = train_test_split(np.arange(len(y_te)), train_size=6000, stratify=y_te, random_state=cfg["split_seed"])
            # Keep first 60 core tx cols + all hk + all g_tx + all g_hk + all v_tx
            tx_keep = min(60, X_tx_tr.shape[1])
            np.savez_compressed(
                os.path.join(CACHE_DIR, "exp2_sgb_ieee_w3.npz"),
                tx_tr=X_tx_tr[sgb_tr_i, :tx_keep], tx_te=X_tx_te[sgb_te_i, :tx_keep],
                hk_tr=X_hk_tr[sgb_tr_i], hk_te=X_hk_te[sgb_te_i],
                gtx_tr=G_tx_tr[sgb_tr_i], gtx_te=G_tx_te[sgb_te_i],
                ghk_tr=G_hk_tr[sgb_tr_i], ghk_te=G_hk_te[sgb_te_i],
                vtx_tr=V_tx_tr[sgb_tr_i], vtx_te=V_tx_te[sgb_te_i],
                y_tr=y_tr[sgb_tr_i], y_te=y_te[sgb_te_i],
            )

    return {
        "dataset": "IEEE-CIS Fraud Detection (144,233 aligned rows, 3 chronological rolling test windows)",
        "bootstrap_note": "实体簇配对 bootstrap（按 card1 聚簇 × 6 seeds）",
        "windows": window_results,
    }


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config()
    out = {
        "experiment": "Exp2_X1_X1Plus_TemporalGraph_CrossDataset",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": cfg["seeds"],
        "split_seed": cfg["split_seed"],
        "engine": cfg["engines"]["proxy_model"],
        "home_credit_x1plus": run_home_credit_x1plus(cfg),
        "santander_x1plus": run_santander_x1plus(cfg),
        "ieee_rolling_x1plus": run_ieee_rolling_windows(cfg),
        "total_elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    save_json(out, os.path.join(OUT_DIR, "exp2_x1plus_temporal_graph.json"))
    log(f"Exp 2 finished in {out['total_elapsed_sec']:.1f}s -> results/exp2_x1plus_temporal_graph.json")


if __name__ == "__main__":
    main()
