"""
第三阶段数据预处理与缓存构建脚本（只读访问原始数据集，缓存写入 第三阶段/results/_cache/）。
1. Home Credit 全量 307,511 客户聚合：
   - 原始 application 表（分离内地特征 vs 香港 EXT_SOURCE）
   - 第三方征信表：bureau + bureau_balance
   - 新第四方消费/支付履约表：previous_application + installments_payments + credit_card_balance + POS_CASH_balance
   - X1+ 严格因果历史时序速度特征：近 90/180/365 天 vs 全历史申请增速、拒贷恶化、还款延迟加速与欠付趋势
2. Santander Product Recommendation 多月因果队列（60,000 客户，2015-11 至 2016-05）：
   - 2016-04 截面特征（内地人口渠道 + 构造香港存款账户 + 构造第三方信贷理财卡）
   - 2016-05 目标：是否新购任意此前未持有产品（new_any）
   - X1+ 严格因果月度时序速度特征（仅用 2015-11..2016-04，零未来信息）：近 1/3/6 月产品净增、增持频次、活跃度变化、新增加速度
"""
from __future__ import annotations

import gc
import json
import os
import time
import numpy as np
import pandas as pd

CACHE_DIR = "/Applications/总文件夹/汇丰/第三阶段/results/_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

HC_DIR = "/Applications/总文件夹/汇丰/8-9月/数据集/HomeCredit_DefaultRisk/解压"
SAN_CSV = "/Applications/总文件夹/汇丰/8-9月/数据集/santander-product-recommendation/解压/train_ver2.csv"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_home_credit_cache() -> None:
    out_path = os.path.join(CACHE_DIR, "hc_master_features.parquet")
    meta_path = os.path.join(CACHE_DIR, "hc_feature_groups.json")
    if os.path.exists(out_path) and os.path.exists(meta_path):
        log("Home Credit cache already exists, skipping.")
        return

    t0 = time.perf_counter()
    log("Loading Home Credit application_train.csv ...")
    app = pd.read_csv(os.path.join(HC_DIR, "application_train.csv"))
    ids = app["SK_ID_CURR"].values
    id_set = set(ids.tolist())

    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    neidi_raw_cols = [c for c in app.columns if c not in (["SK_ID_CURR", "TARGET"] + ext_cols)]

    # HK party engineered local stats strictly from EXT_SOURCE_1/2/3
    hk_df = pd.DataFrame({"SK_ID_CURR": ids})
    for c in ext_cols:
        hk_df[c] = app[c].astype(np.float32)
        hk_df[f"{c}_isna"] = app[c].isna().astype(np.float32)
    ext_mat = app[ext_cols].to_numpy(dtype=np.float64)
    with np.errstate(all="ignore"):
        hk_df["ext_mean"] = np.nanmean(ext_mat, axis=1).astype(np.float32)
        hk_df["ext_std"] = np.nanstd(ext_mat, axis=1).astype(np.float32)
        hk_df["ext_min"] = np.nanmin(ext_mat, axis=1).astype(np.float32)
        hk_df["ext_max"] = np.nanmax(ext_mat, axis=1).astype(np.float32)
        hk_df["ext_cnt"] = np.sum(~np.isnan(ext_mat), axis=1).astype(np.float32)
    hk_cols = [c for c in hk_df.columns if c != "SK_ID_CURR"]

    # --- Party 3: Bureau + Bureau Balance ---
    log("Aggregating bureau.csv & bureau_balance.csv ...")
    bb = pd.read_csv(os.path.join(HC_DIR, "bureau_balance.csv"))
    bb["is_dpd"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(np.float32)
    bb["is_severe_dpd"] = bb["STATUS"].isin(["3", "4", "5"]).astype(np.float32)
    bb_agg = bb.groupby("SK_ID_BUREAU", as_index=False).agg(
        bb_months=("MONTHS_BALANCE", "count"),
        bb_dpd_rate=("is_dpd", "mean"),
        bb_sev_dpd_sum=("is_severe_dpd", "sum"),
        bb_recent_m=("MONTHS_BALANCE", "max"),
    )
    del bb
    gc.collect()

    bur = pd.read_csv(os.path.join(HC_DIR, "bureau.csv"))
    bur = bur[bur["SK_ID_CURR"].isin(id_set)].merge(bb_agg, on="SK_ID_BUREAU", how="left")
    del bb_agg
    bur["is_active"] = (bur["CREDIT_ACTIVE"] == "Active").astype(np.float32)
    bur["is_closed"] = (bur["CREDIT_ACTIVE"] == "Closed").astype(np.float32)
    bur["debt_ratio"] = bur["AMT_CREDIT_SUM_DEBT"] / (bur["AMT_CREDIT_SUM"].abs() + 1.0)
    bur["has_overdue"] = ((bur["AMT_CREDIT_SUM_OVERDUE"].fillna(0) > 0) | (bur["CREDIT_DAY_OVERDUE"].fillna(0) > 0)).astype(np.float32)
    bur_agg = bur.groupby("SK_ID_CURR").agg(
        bur_cnt=("SK_ID_BUREAU", "count"),
        bur_active_cnt=("is_active", "sum"),
        bur_active_rate=("is_active", "mean"),
        bur_closed_cnt=("is_closed", "sum"),
        bur_days_credit_mean=("DAYS_CREDIT", "mean"),
        bur_days_credit_max=("DAYS_CREDIT", "max"),
        bur_days_credit_min=("DAYS_CREDIT", "min"),
        bur_enddate_max=("DAYS_CREDIT_ENDDATE", "max"),
        bur_overdue_max=("CREDIT_DAY_OVERDUE", "max"),
        bur_overdue_rate=("has_overdue", "mean"),
        bur_amt_sum=("AMT_CREDIT_SUM", "sum"),
        bur_amt_mean=("AMT_CREDIT_SUM", "mean"),
        bur_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
        bur_debt_ratio_mean=("debt_ratio", "mean"),
        bur_overdue_amt_sum=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        bur_prolong_sum=("CNT_CREDIT_PROLONG", "sum"),
        bur_bb_dpd_mean=("bb_dpd_rate", "mean"),
        bur_bb_sev_sum=("bb_sev_dpd_sum", "sum"),
    ).reset_index()
    bur_agg["bur_has_record"] = 1.0
    bureau_cols = [c for c in bur_agg.columns if c != "SK_ID_CURR"]
    del bur
    gc.collect()

    # --- Party 4 (NEW): previous_application + installments_payments + credit_card_balance + POS_CASH_balance ---
    # Also build X1+ strict causal temporal velocity features (recent 90d/180d/365d vs full history)!
    log("Aggregating previous_application.csv (base + X1+ velocity) ...")
    prev_use = [
        "SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS", "AMT_APPLICATION",
        "AMT_CREDIT", "AMT_ANNUITY", "AMT_DOWN_PAYMENT", "DAYS_DECISION",
        "CNT_PAYMENT", "RATE_DOWN_PAYMENT", "NAME_CONTRACT_TYPE",
    ]
    prev = pd.read_csv(os.path.join(HC_DIR, "previous_application.csv"), usecols=prev_use)
    prev = prev[prev["SK_ID_CURR"].isin(id_set)].copy()
    prev["is_approved"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(np.float32)
    prev["is_refused"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(np.float32)
    prev["is_canceled"] = (prev["NAME_CONTRACT_STATUS"] == "Canceled").astype(np.float32)
    prev["app_to_credit"] = prev["AMT_APPLICATION"] / (prev["AMT_CREDIT"].abs() + 1.0)
    prev["amt_diff"] = prev["AMT_APPLICATION"] - prev["AMT_CREDIT"]

    prev_agg = prev.groupby("SK_ID_CURR").agg(
        prev_cnt=("SK_ID_PREV", "count"),
        prev_appr_rate=("is_approved", "mean"),
        prev_ref_rate=("is_refused", "mean"),
        prev_ref_cnt=("is_refused", "sum"),
        prev_canc_rate=("is_canceled", "mean"),
        prev_amt_credit_mean=("AMT_CREDIT", "mean"),
        prev_amt_credit_max=("AMT_CREDIT", "max"),
        prev_annuity_mean=("AMT_ANNUITY", "mean"),
        prev_app_credit_ratio=("app_to_credit", "mean"),
        prev_amt_diff_mean=("amt_diff", "mean"),
        prev_down_rate_mean=("RATE_DOWN_PAYMENT", "mean"),
        prev_cnt_pay_mean=("CNT_PAYMENT", "mean"),
        prev_days_dec_max=("DAYS_DECISION", "max"),
        prev_days_dec_mean=("DAYS_DECISION", "mean"),
    ).reset_index()

    # X1+ temporal velocity from previous_application (strictly DAYS_DECISION <= 0, recent windows vs full history)
    prev["in_90d"] = (prev["DAYS_DECISION"] >= -90).astype(np.float32)
    prev["in_180d"] = (prev["DAYS_DECISION"] >= -180).astype(np.float32)
    prev["in_365d"] = (prev["DAYS_DECISION"] >= -365).astype(np.float32)
    prev["ref_180d"] = prev["is_refused"] * prev["in_180d"]
    prev["ref_365d"] = prev["is_refused"] * prev["in_365d"]
    prev["amt_180d"] = prev["AMT_APPLICATION"].fillna(0) * prev["in_180d"]
    prev["amt_365d"] = prev["AMT_APPLICATION"].fillna(0) * prev["in_365d"]

    prev_vel = prev.groupby("SK_ID_CURR").agg(
        x1v_prev_cnt_90d=("in_90d", "sum"),
        x1v_prev_cnt_180d=("in_180d", "sum"),
        x1v_prev_cnt_365d=("in_365d", "sum"),
        x1v_prev_ref_180d=("ref_180d", "sum"),
        x1v_prev_ref_365d=("ref_365d", "sum"),
        x1v_prev_amt_180d=("amt_180d", "sum"),
        x1v_prev_amt_365d=("amt_365d", "sum"),
        x1v_prev_days_span=("DAYS_DECISION", lambda s: float(s.max() - s.min())),
    ).reset_index()
    prev_vel = prev_vel.merge(prev_agg[["SK_ID_CURR", "prev_cnt", "prev_ref_rate", "prev_amt_credit_mean"]], on="SK_ID_CURR", how="left")
    prev_vel["x1v_prev_burst_180_ratio"] = prev_vel["x1v_prev_cnt_180d"] / (prev_vel["prev_cnt"] + 1.0)
    prev_vel["x1v_prev_ref_rate_180d"] = prev_vel["x1v_prev_ref_180d"] / (prev_vel["x1v_prev_cnt_180d"] + 0.5)
    prev_vel["x1v_prev_ref_trend"] = prev_vel["x1v_prev_ref_rate_180d"] - prev_vel["prev_ref_rate"]
    prev_vel["x1v_prev_amt_accel"] = (prev_vel["x1v_prev_amt_180d"] / (prev_vel["x1v_prev_cnt_180d"] + 1.0)) / (prev_vel["prev_amt_credit_mean"].abs() + 1.0)
    prev_vel.drop(columns=["prev_cnt", "prev_ref_rate", "prev_amt_credit_mean"], inplace=True)
    del prev
    gc.collect()

    log("Aggregating installments_payments.csv (base + X1+ velocity) ...")
    inst = pd.read_csv(os.path.join(HC_DIR, "installments_payments.csv"))
    inst = inst[inst["SK_ID_CURR"].isin(id_set)].copy()
    inst["dpd"] = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
    inst["dbd"] = (inst["DAYS_INSTALMENT"] - inst["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    inst["is_late"] = (inst["dpd"] > 0).astype(np.float32)
    inst["pay_diff"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
    inst["is_underpay"] = (inst["pay_diff"] > 1.0).astype(np.float32)
    inst["pay_ratio"] = inst["AMT_PAYMENT"] / (inst["AMT_INSTALMENT"].abs() + 1.0)

    inst_agg = inst.groupby("SK_ID_CURR").agg(
        inst_cnt=("SK_ID_PREV", "count"),
        inst_late_rate=("is_late", "mean"),
        inst_late_cnt=("is_late", "sum"),
        inst_dpd_mean=("dpd", "mean"),
        inst_dpd_max=("dpd", "max"),
        inst_dbd_mean=("dbd", "mean"),
        inst_underpay_rate=("is_underpay", "mean"),
        inst_pay_diff_mean=("pay_diff", "mean"),
        inst_pay_diff_sum=("pay_diff", "sum"),
        inst_pay_ratio_mean=("pay_ratio", "mean"),
        inst_amt_pay_sum=("AMT_PAYMENT", "sum"),
    ).reset_index()

    # X1+ temporal velocity from installments_payments (recent 180d & 365d vs full history)
    inst_180 = inst[inst["DAYS_INSTALMENT"] >= -180].groupby("SK_ID_CURR").agg(
        x1v_inst_cnt_180d=("SK_ID_PREV", "count"),
        x1v_inst_late_rate_180d=("is_late", "mean"),
        x1v_inst_dpd_mean_180d=("dpd", "mean"),
        x1v_inst_dpd_max_180d=("dpd", "max"),
        x1v_inst_underpay_180d=("is_underpay", "mean"),
        x1v_inst_pay_ratio_180d=("pay_ratio", "mean"),
    ).reset_index()
    inst_365 = inst[inst["DAYS_INSTALMENT"] >= -365].groupby("SK_ID_CURR").agg(
        x1v_inst_late_rate_365d=("is_late", "mean"),
        x1v_inst_dpd_mean_365d=("dpd", "mean"),
        x1v_inst_underpay_365d=("is_underpay", "mean"),
    ).reset_index()
    inst_vel = inst_agg[["SK_ID_CURR", "inst_late_rate", "inst_dpd_mean", "inst_underpay_rate"]].merge(
        inst_180, on="SK_ID_CURR", how="left"
    ).merge(inst_365, on="SK_ID_CURR", how="left")
    inst_vel["x1v_inst_late_trend_180"] = inst_vel["x1v_inst_late_rate_180d"] - inst_vel["inst_late_rate"]
    inst_vel["x1v_inst_dpd_trend_180"] = inst_vel["x1v_inst_dpd_mean_180d"] - inst_vel["inst_dpd_mean"]
    inst_vel["x1v_inst_underpay_trend_180"] = inst_vel["x1v_inst_underpay_180d"] - inst_vel["inst_underpay_rate"]
    inst_vel.drop(columns=["inst_late_rate", "inst_dpd_mean", "inst_underpay_rate"], inplace=True)
    del inst, inst_180, inst_365
    gc.collect()

    log("Aggregating credit_card_balance.csv ...")
    cc_use = [
        "SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_ATM_CURRENT", "AMT_DRAWINGS_CURRENT", "CNT_DRAWINGS_CURRENT",
        "AMT_PAYMENT_TOTAL_CURRENT", "SK_DPD", "SK_DPD_DEF",
    ]
    cc = pd.read_csv(os.path.join(HC_DIR, "credit_card_balance.csv"), usecols=cc_use)
    cc = cc[cc["SK_ID_CURR"].isin(id_set)].copy()
    cc["util"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"].abs() + 1.0)
    cc["atm_ratio"] = cc["AMT_DRAWINGS_ATM_CURRENT"] / (cc["AMT_DRAWINGS_CURRENT"].abs() + 1.0)
    cc["has_dpd"] = (cc["SK_DPD"] > 0).astype(np.float32)

    cc_agg = cc.groupby("SK_ID_CURR").agg(
        cc_months=("MONTHS_BALANCE", "count"),
        cc_util_mean=("util", "mean"),
        cc_util_max=("util", "max"),
        cc_bal_mean=("AMT_BALANCE", "mean"),
        cc_limit_mean=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        cc_draw_cnt_mean=("CNT_DRAWINGS_CURRENT", "mean"),
        cc_atm_ratio_mean=("atm_ratio", "mean"),
        cc_dpd_rate=("has_dpd", "mean"),
        cc_dpd_max=("SK_DPD", "max"),
    ).reset_index()

    cc_6m = cc[cc["MONTHS_BALANCE"] >= -6].groupby("SK_ID_CURR").agg(
        x1v_cc_util_6m=("util", "mean"),
        x1v_cc_draw_cnt_6m=("CNT_DRAWINGS_CURRENT", "mean"),
        x1v_cc_dpd_rate_6m=("has_dpd", "mean"),
    ).reset_index()
    cc_vel = cc_agg[["SK_ID_CURR", "cc_util_mean"]].merge(cc_6m, on="SK_ID_CURR", how="left")
    cc_vel["x1v_cc_util_trend_6m"] = cc_vel["x1v_cc_util_6m"] - cc_vel["cc_util_mean"]
    cc_vel.drop(columns=["cc_util_mean"], inplace=True)
    del cc, cc_6m
    gc.collect()

    log("Aggregating POS_CASH_balance.csv ...")
    pos = pd.read_csv(os.path.join(HC_DIR, "POS_CASH_balance.csv"))
    pos = pos[pos["SK_ID_CURR"].isin(id_set)].copy()
    pos["has_dpd"] = (pos["SK_DPD"] > 0).astype(np.float32)
    pos["has_dpd_def"] = (pos["SK_DPD_DEF"] > 0).astype(np.float32)
    pos["is_active"] = (pos["NAME_CONTRACT_STATUS"] == "Active").astype(np.float32)
    pos_agg = pos.groupby("SK_ID_CURR").agg(
        pos_months=("MONTHS_BALANCE", "count"),
        pos_active_rate=("is_active", "mean"),
        pos_future_mean=("CNT_INSTALMENT_FUTURE", "mean"),
        pos_dpd_rate=("has_dpd", "mean"),
        pos_dpd_max=("SK_DPD", "max"),
        pos_dpd_def_rate=("has_dpd_def", "mean"),
    ).reset_index()

    pos_6m = pos[pos["MONTHS_BALANCE"] >= -6].groupby("SK_ID_CURR").agg(
        x1v_pos_dpd_6m=("has_dpd", "mean"),
        x1v_pos_months_6m=("MONTHS_BALANCE", "count"),
    ).reset_index()
    del pos
    gc.collect()

    # Merge everything cleanly onto application_train order
    log("Merging all Home Credit feature blocks ...")
    pay_df = (
        app[["SK_ID_CURR"]]
        .merge(prev_agg, on="SK_ID_CURR", how="left")
        .merge(inst_agg, on="SK_ID_CURR", how="left")
        .merge(cc_agg, on="SK_ID_CURR", how="left")
        .merge(pos_agg, on="SK_ID_CURR", how="left")
    )
    pay_df["pay_has_record"] = (pay_df["prev_cnt"].notna() | pay_df["inst_cnt"].notna() | pay_df["cc_months"].notna() | pay_df["pos_months"].notna()).astype(np.float32)
    pay_cols = [c for c in pay_df.columns if c != "SK_ID_CURR"]

    vel_df = (
        app[["SK_ID_CURR"]]
        .merge(prev_vel, on="SK_ID_CURR", how="left")
        .merge(inst_vel, on="SK_ID_CURR", how="left")
        .merge(cc_vel, on="SK_ID_CURR", how="left")
        .merge(pos_6m, on="SK_ID_CURR", how="left")
    )
    vel_cols = [c for c in vel_df.columns if c != "SK_ID_CURR"]

    master = (
        app[["SK_ID_CURR", "TARGET"] + neidi_raw_cols]
        .merge(hk_df, on="SK_ID_CURR", how="left")
        .merge(bur_agg, on="SK_ID_CURR", how="left")
        .merge(pay_df, on="SK_ID_CURR", how="left")
        .merge(vel_df, on="SK_ID_CURR", how="left")
    )
    master["bur_has_record"] = master["bur_has_record"].fillna(0.0).astype(np.float32)

    # Downcast float64 -> float32 to save disk & speed up reads
    for c in master.columns:
        if c not in ("SK_ID_CURR", "TARGET") and master[c].dtype == np.float64:
            master[c] = master[c].astype(np.float32)

    master.to_parquet(out_path, index=False)
    meta = {
        "n_rows": int(len(master)),
        "pos_rate": float(master["TARGET"].mean()),
        "neidi_cols": neidi_raw_cols,
        "hk_cols": hk_cols,
        "bureau_cols": bureau_cols,
        "pay_cols": pay_cols,
        "x1_velocity_cols": vel_cols,
        "bureau_coverage": float((master["bur_has_record"] > 0).mean()),
        "pay_coverage": float((master["pay_has_record"] > 0).mean()),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"Saved Home Credit cache ({master.shape}) in {time.perf_counter()-t0:.1f}s | bureau_cov={meta['bureau_coverage']:.3f}, pay_cov={meta['pay_coverage']:.3f}")


def build_santander_cache() -> None:
    out_path = os.path.join(CACHE_DIR, "santander_multimonth_60k.parquet")
    meta_path = os.path.join(CACHE_DIR, "santander_feature_groups.json")
    if os.path.exists(out_path) and os.path.exists(meta_path):
        log("Santander cache already exists, skipping.")
        return

    t0 = time.perf_counter()
    log("Streaming Santander train_ver2.csv for 7-month causal cohort (2015-11..2016-05) ...")
    # Hash-sample ~7.5% of ncodpers deterministically -> ~68k customers in 2016-05, keep 60k present in both 2016-04 & 2016-05
    target_months = {"2015-11-28", "2015-12-28", "2016-01-28", "2016-02-28", "2016-03-28", "2016-04-28", "2016-05-28"}
    chunks = []
    for chunk in pd.read_csv(SAN_CSV, chunksize=500_000, low_memory=False):
        m = chunk["fecha_dato"].isin(target_months)
        if not m.any():
            continue
        sub = chunk.loc[m]
        # Deterministic customer hash filter (~8% of customers, independent of label)
        keep = ((sub["ncodpers"].astype(np.int64) * 2654435761) % 1000) < 85
        if keep.any():
            chunks.append(sub.loc[keep].copy())

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    log(f"Loaded {len(df)} filtered rows across {df['fecha_dato'].nunique()} months.")

    prod_cols = [c for c in df.columns if c.startswith("ind_") and c.endswith("_ult1")]
    for c in prod_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int8)

    # Clean numeric fields stored as strings
    for nc in ["age", "antiguedad", "renta", "ind_nuevo", "indrel", "ind_actividad_cliente", "cod_prov"]:
        df[nc] = pd.to_numeric(df[nc], errors="coerce").astype(np.float32)

    # Customers present in BOTH 2016-04-28 and 2016-05-28
    m4 = df[df["fecha_dato"] == "2016-04-28"].drop_duplicates("ncodpers").set_index("ncodpers")
    m5 = df[df["fecha_dato"] == "2016-05-28"].drop_duplicates("ncodpers").set_index("ncodpers")
    common_ids = np.sort(np.intersect1d(m4.index.values, m5.index.values))
    rng = np.random.default_rng(2026)
    if len(common_ids) > 60000:
        common_ids = np.sort(rng.choice(common_ids, size=60000, replace=False))
    log(f"Selected {len(common_ids)} customers present in both 2016-04 & 2016-05.")

    p4 = m4.loc[common_ids, prod_cols].to_numpy(dtype=np.int8)
    p5 = m5.loc[common_ids, prod_cols].to_numpy(dtype=np.int8)
    new_add = ((p5 == 1) & (p4 == 0)).astype(np.int8)
    y_new_any = (new_add.sum(axis=1) > 0).astype(np.int8)
    y_new_cnt = new_add.sum(axis=1).astype(np.int16)

    # Base features from 2016-04-28 (strictly before 2016-05-28)
    neidi_cols = [
        "ind_empleado", "pais_residencia", "sexo", "age", "ind_nuevo", "antiguedad",
        "indrel", "indrel_1mes", "tiprel_1mes", "indresi", "indext", "canal_entrada",
        "indfall", "cod_prov", "ind_actividad_cliente", "renta", "segmento",
    ]
    hk_deposit_prods = [
        "ind_ahor_fin_ult1", "ind_cco_fin_ult1", "ind_cno_fin_ult1", "ind_ctju_fin_ult1",
        "ind_ctma_fin_ult1", "ind_ctop_fin_ult1", "ind_ctpp_fin_ult1", "ind_deco_fin_ult1",
        "ind_deme_fin_ult1", "ind_dela_fin_ult1", "ind_ecue_fin_ult1", "ind_nomina_ult1",
        "ind_nom_pens_ult1", "ind_recibo_ult1",
    ]
    third_credit_wealth_prods = [c for c in prod_cols if c not in hk_deposit_prods]

    base = m4.loc[common_ids, neidi_cols + prod_cols].copy()
    base["hk_dep_total_m4"] = base[hk_deposit_prods].sum(axis=1).astype(np.float32)
    base["third_cw_total_m4"] = base[third_credit_wealth_prods].sum(axis=1).astype(np.float32)
    hk_cols = hk_deposit_prods + ["hk_dep_total_m4"]
    third_cols = third_credit_wealth_prods + ["third_cw_total_m4"]

    # X1+ Strictly Causal Multi-Month Temporal Velocity (2015-11..2016-04 ONLY, ZERO 2016-05 info!)
    hist_months = ["2015-11-28", "2015-12-28", "2016-01-28", "2016-02-28", "2016-03-28", "2016-04-28"]
    hist = df[df["fecha_dato"].isin(hist_months) & df["ncodpers"].isin(common_ids)].copy()
    hist["m_idx"] = hist["fecha_dato"].map({m: i for i, m in enumerate(hist_months)})
    hist = hist.sort_values(["ncodpers", "m_idx"])
    hist["dep_tot"] = hist[hk_deposit_prods].sum(axis=1).astype(np.float32)
    hist["cw_tot"] = hist[third_credit_wealth_prods].sum(axis=1).astype(np.float32)
    hist["all_tot"] = hist["dep_tot"] + hist["cw_tot"]

    # Compute month-over-month additions/drops strictly within 2015-11..2016-04
    # Pivot key totals by (ncodpers, m_idx)
    piv_dep = hist.pivot_table(index="ncodpers", columns="m_idx", values="dep_tot", aggfunc="last").reindex(common_ids)
    piv_cw = hist.pivot_table(index="ncodpers", columns="m_idx", values="cw_tot", aggfunc="last").reindex(common_ids)
    piv_act = hist.pivot_table(index="ncodpers", columns="m_idx", values="ind_actividad_cliente", aggfunc="last").reindex(common_ids)

    # Also exact count of individual product additions in (m2->m3), (m3->m4), (m4->m5=2016-04) strictly historical
    # m_idx: 0=2015-11, 1=2015-12, 2=2016-01, 3=2016-02, 4=2016-03, 5=2016-04
    def prod_add_drop(sub_cols, m_prev, m_curr):
        a = hist[hist["m_idx"] == m_prev].drop_duplicates("ncodpers").set_index("ncodpers")[sub_cols].reindex(common_ids)
        b = hist[hist["m_idx"] == m_curr].drop_duplicates("ncodpers").set_index("ncodpers")[sub_cols].reindex(common_ids)
        valid = a.notna().all(axis=1) & b.notna().all(axis=1)
        adds = ((b.fillna(0).values == 1) & (a.fillna(0).values == 0)).sum(axis=1).astype(np.float32)
        drops = ((b.fillna(0).values == 0) & (a.fillna(0).values == 1)).sum(axis=1).astype(np.float32)
        adds[~valid.values] = np.nan
        drops[~valid.values] = np.nan
        return adds, drops

    # Active-party (neidi) causal velocity features (activity & tenure dynamics)
    x1v_neidi = pd.DataFrame(index=common_ids)
    x1v_neidi["x1v_act_mean_6m"] = piv_act.mean(axis=1).astype(np.float32)
    x1v_neidi["x1v_act_std_6m"] = piv_act.std(axis=1).astype(np.float32)
    x1v_neidi["x1v_act_delta_1m"] = (piv_act[5] - piv_act[4]).astype(np.float32)
    x1v_neidi["x1v_act_delta_3m"] = (piv_act[5] - piv_act[2]).astype(np.float32)
    x1v_neidi["x1v_act_delta_5m"] = (piv_act[5] - piv_act[0]).astype(np.float32)
    x1v_neidi["x1v_months_seen_6m"] = piv_act.notna().sum(axis=1).astype(np.float32)

    # HK-deposit-party causal velocity features (strictly on hk_deposit_prods)
    x1v_hk = pd.DataFrame(index=common_ids)
    x1v_hk["x1v_dep_delta_1m"] = (piv_dep[5] - piv_dep[4]).astype(np.float32)
    x1v_hk["x1v_dep_delta_3m"] = (piv_dep[5] - piv_dep[2]).astype(np.float32)
    x1v_hk["x1v_dep_delta_5m"] = (piv_dep[5] - piv_dep[0]).astype(np.float32)
    x1v_hk["x1v_dep_mean_6m"] = piv_dep.mean(axis=1).astype(np.float32)
    x1v_hk["x1v_dep_max_6m"] = piv_dep.max(axis=1).astype(np.float32)
    x1v_hk["x1v_dep_std_6m"] = piv_dep.std(axis=1).astype(np.float32)
    dep_add_1m, dep_drop_1m = prod_add_drop(hk_deposit_prods, 4, 5)
    dep_add_2m, dep_drop_2m = prod_add_drop(hk_deposit_prods, 3, 4)
    dep_add_3m, dep_drop_3m = prod_add_drop(hk_deposit_prods, 2, 3)
    x1v_hk["x1v_dep_add_1m"] = dep_add_1m
    x1v_hk["x1v_dep_drop_1m"] = dep_drop_1m
    x1v_hk["x1v_dep_add_3m_sum"] = np.nansum(np.vstack([dep_add_1m, dep_add_2m, dep_add_3m]), axis=0).astype(np.float32)
    x1v_hk["x1v_dep_drop_3m_sum"] = np.nansum(np.vstack([dep_drop_1m, dep_drop_2m, dep_drop_3m]), axis=0).astype(np.float32)
    x1v_hk["x1v_dep_churn_events_3m"] = x1v_hk["x1v_dep_add_3m_sum"] + x1v_hk["x1v_dep_drop_3m_sum"]

    # Third-party causal velocity features (strictly on third_credit_wealth_prods)
    x1v_third = pd.DataFrame(index=common_ids)
    x1v_third["x1v_cw_delta_1m"] = (piv_cw[5] - piv_cw[4]).astype(np.float32)
    x1v_third["x1v_cw_delta_3m"] = (piv_cw[5] - piv_cw[2]).astype(np.float32)
    x1v_third["x1v_cw_delta_5m"] = (piv_cw[5] - piv_cw[0]).astype(np.float32)
    x1v_third["x1v_cw_mean_6m"] = piv_cw.mean(axis=1).astype(np.float32)
    cw_add_1m, cw_drop_1m = prod_add_drop(third_credit_wealth_prods, 4, 5)
    cw_add_2m, cw_drop_2m = prod_add_drop(third_credit_wealth_prods, 3, 4)
    cw_add_3m, cw_drop_3m = prod_add_drop(third_credit_wealth_prods, 2, 3)
    x1v_third["x1v_cw_add_1m"] = cw_add_1m
    x1v_third["x1v_cw_drop_1m"] = cw_drop_1m
    x1v_third["x1v_cw_add_3m_sum"] = np.nansum(np.vstack([cw_add_1m, cw_add_2m, cw_add_3m]), axis=0).astype(np.float32)

    out_df = pd.concat([base, x1v_neidi, x1v_hk, x1v_third], axis=1)
    out_df.insert(0, "ncodpers", common_ids)
    out_df.insert(1, "y_new_any", y_new_any)
    out_df.insert(2, "y_new_cnt", y_new_cnt)
    out_df.reset_index(drop=True, inplace=True)

    out_df.to_parquet(out_path, index=False)
    meta = {
        "n_rows": int(len(out_df)),
        "pos_rate": float(y_new_any.mean()),
        "target_month": "2016-05-28",
        "feature_month": "2016-04-28",
        "history_months": hist_months,
        "neidi_cols": neidi_cols,
        "hk_cols": hk_cols,
        "third_cols": third_cols,
        "x1v_neidi_cols": list(x1v_neidi.columns),
        "x1v_hk_cols": list(x1v_hk.columns),
        "x1v_third_cols": list(x1v_third.columns),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"Saved Santander 60k cache ({out_df.shape}) in {time.perf_counter()-t0:.1f}s | pos_rate={meta['pos_rate']:.4f}")


if __name__ == "__main__":
    build_home_credit_cache()
    build_santander_cache()
