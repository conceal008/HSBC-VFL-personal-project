"""
实验 1（Exp 1）：X2 多方 / 第三方参与方扩展跨数据集验证（6-seed 等调参代理 + 配对 Bootstrap CI + Holm 校正）。
同时导出精简 .npz 切分供 exp1_sgb_confirm.py 在真实 SecretFlow 1.14 SGB 上做 2方/3方/4方实跑复核。

验证场景：
1. Home Credit（100,000 分层抽样客户，60k/20k/20k 冻结切分）：
   - L0：内地申请表（不含 EXT_SOURCE）
   - F0：两方（内地 + 香港 EXT_SOURCE）
   - F3_bureau：三方（内地 + 香港 + 征信 bureau/bureau_balance，复现参考）
   - F3_pay：【新第三方】三方（内地 + 香港 + 支付/消费履约机构：previous_application + installments + credit_card + POS_CASH）
   - F4_all：【新四方】四方（内地 + 香港 + 征信 + 支付/消费履约机构）
   - F3_pay_cov70 / F3_pay_cov40：新第三方覆盖率降至 70% / 40% 时的敏感性
2. Santander 产品推荐【构造三方】（60,000 客户，2016-04 -> 2016-05 新增任意产品，36k/12k/12k）：
   - L0：内地（人口统计与渠道）
   - F0：两方（内地 + 构造香港存款/基础账户）
   - F3：【构造三方】（内地 + 构造香港存款账户 + 构造第三方信贷/理财/信用卡）
   - F_neidi_third：两方对照（内地 + 构造第三方）
3. UCI Bank Marketing【构造三方，严格剔除 duration】（全量 41,188 条，分层 60/20/20）：
   - L0：内地（客户基本画像）
   - F0：两方（内地 + 银行营销接触史，无 duration）
   - F3：【构造三方】（内地 + 营销接触史 + 第三方宏观经济景气指标）
   - 另附时间顺序切分对照（检验宏观变量跨期漂移边界）
"""
from __future__ import annotations

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
    print(f"[{time.strftime('%H:%M:%S')}] [Exp1] {msg}", flush=True)


def run_contrasts(y_te: np.ndarray, arm_preds: dict, pairs: list, cfg: dict) -> list:
    rows = []
    for name, base_k, new_k in pairs:
        res = paired_bootstrap_delta(
            y_te,
            arm_preds[base_k],
            arm_preds[new_k],
            metric_fn=fast_roc_auc,
            n_boot=cfg["bootstrap_iters"],
            seed=cfg["bootstrap_seed"],
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


def run_home_credit(cfg: dict) -> dict:
    log("=== 1/3 Home Credit 扩展新第三方（支付消费履约）与四方 VFL ===")
    df = pd.read_parquet(os.path.join(CACHE_DIR, "hc_master_features.parquet"))
    with open(os.path.join(CACHE_DIR, "hc_feature_groups.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Stratified sample of 100,000 customers (split_seed=2026) -> 60k train / 20k val / 20k test
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

    X_neidi_tr, X_neidi_va, X_neidi_te, nm_neidi = encode_tabular_train_val_test(df_tr, df_va, df_te, meta["neidi_cols"])
    X_hk_tr, X_hk_va, X_hk_te, nm_hk = encode_tabular_train_val_test(df_tr, df_va, df_te, meta["hk_cols"])
    X_bur_tr, X_bur_va, X_bur_te, nm_bur = encode_tabular_train_val_test(df_tr, df_va, df_te, meta["bureau_cols"])
    X_pay_tr, X_pay_va, X_pay_te, nm_pay = encode_tabular_train_val_test(df_tr, df_va, df_te, meta["pay_cols"])

    def mask_cov(X_tr, X_va, X_te, cov: float, seed: int):
        rng = np.random.default_rng(seed)
        out = []
        for X in (X_tr, X_va, X_te):
            Xc = X.copy()
            keep = rng.random(len(Xc)) < cov
            Xc[~keep] = np.nan
            out.append(Xc)
        return tuple(out)

    Xp70_tr, Xp70_va, Xp70_te = mask_cov(X_pay_tr, X_pay_va, X_pay_te, 0.70, 3070)
    Xp40_tr, Xp40_va, Xp40_te = mask_cov(X_pay_tr, X_pay_va, X_pay_te, 0.40, 3040)

    arm_specs = {
        "L0_neidi": (X_neidi_tr, X_neidi_va, X_neidi_te),
        "F0_2party": (
            np.hstack([X_neidi_tr, X_hk_tr]),
            np.hstack([X_neidi_va, X_hk_va]),
            np.hstack([X_neidi_te, X_hk_te]),
        ),
        "F3_bureau": (
            np.hstack([X_neidi_tr, X_hk_tr, X_bur_tr]),
            np.hstack([X_neidi_va, X_hk_va, X_bur_va]),
            np.hstack([X_neidi_te, X_hk_te, X_bur_te]),
        ),
        "F3_pay_new": (
            np.hstack([X_neidi_tr, X_hk_tr, X_pay_tr]),
            np.hstack([X_neidi_va, X_hk_va, X_pay_va]),
            np.hstack([X_neidi_te, X_hk_te, X_pay_te]),
        ),
        "F4_all_4party": (
            np.hstack([X_neidi_tr, X_hk_tr, X_bur_tr, X_pay_tr]),
            np.hstack([X_neidi_va, X_hk_va, X_bur_va, X_pay_va]),
            np.hstack([X_neidi_te, X_hk_te, X_bur_te, X_pay_te]),
        ),
        "F3_pay_cov70": (
            np.hstack([X_neidi_tr, X_hk_tr, Xp70_tr]),
            np.hstack([X_neidi_va, X_hk_va, Xp70_va]),
            np.hstack([X_neidi_te, X_hk_te, Xp70_te]),
        ),
        "F3_pay_cov40": (
            np.hstack([X_neidi_tr, X_hk_tr, Xp40_tr]),
            np.hstack([X_neidi_va, X_hk_va, Xp40_va]),
            np.hstack([X_neidi_te, X_hk_te, Xp40_te]),
        ),
    }

    arms_out, arm_preds = {}, {}
    for k, (Xtr, Xva, Xte) in arm_specs.items():
        res = equal_tune_and_eval_hgb(Xtr, y_tr, Xva, y_va, Xte, y_te, seeds=cfg["seeds"], max_bins=33)
        arms_out[k] = res
        arm_preds[k] = res["preds_te"]
        log(f"  HC {k:14s}: AUC={res['summary']['auc_mean']:.4f}±{res['summary']['auc_std']:.4f} "
            f"PR={res['summary']['pr_auc_mean']:.4f} Top5Lift={res['summary']['top5_lift_mean']:.3f} "
            f"Bag6={res['bagging6_metrics']['auc']:.4f} ({res['elapsed_sec']:.1f}s)")

    contrasts = run_contrasts(
        y_te,
        arm_preds,
        [
            ("F0_2party - L0_neidi", "L0_neidi", "F0_2party"),
            ("F3_bureau - F0_2party", "F0_2party", "F3_bureau"),
            ("F3_pay_new - F0_2party", "F0_2party", "F3_pay_new"),
            ("F4_all_4party - F0_2party", "F0_2party", "F4_all_4party"),
            ("F4_all_4party - F3_bureau", "F3_bureau", "F4_all_4party"),
            ("F4_all_4party - F3_pay_new", "F3_pay_new", "F4_all_4party"),
            ("F3_pay_cov70 - F0_2party", "F0_2party", "F3_pay_cov70"),
            ("F3_pay_cov40 - F0_2party", "F0_2party", "F3_pay_cov40"),
        ],
        cfg,
    )
    for c in contrasts:
        log(f"  Contrast {c['contrast']:28s}: ΔAUC={c['delta_mean']:+.4f} "
            f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}] p_holm={c['p_holm']:.4f}")

    # Save a deterministic 12k train / 6k test slice for real SecretFlow 1.14 SGB 2/3/4-party confirmation
    sgb_tr_i, _ = train_test_split(np.arange(len(y_tr)), train_size=12000, stratify=y_tr, random_state=cfg["split_seed"])
    sgb_te_i, _ = train_test_split(np.arange(len(y_te)), train_size=6000, stratify=y_te, random_state=cfg["split_seed"])
    np.savez_compressed(
        os.path.join(CACHE_DIR, "exp1_sgb_hc.npz"),
        neidi_tr=X_neidi_tr[sgb_tr_i], neidi_te=X_neidi_te[sgb_te_i],
        hk_tr=X_hk_tr[sgb_tr_i], hk_te=X_hk_te[sgb_te_i],
        bur_tr=X_bur_tr[sgb_tr_i], bur_te=X_bur_te[sgb_te_i],
        pay_tr=X_pay_tr[sgb_tr_i], pay_te=X_pay_te[sgb_te_i],
        y_tr=y_tr[sgb_tr_i], y_te=y_te[sgb_te_i],
    )

    return {
        "dataset": "Home Credit Default Risk (100k stratified cohort, 60k/20k/20k)",
        "split_note": "自然多表结构：内地(application去EXT) + 香港(EXT_SOURCE) + 征信(bureau/bb) + 新第四方支付履约(prev_app/installments/cc/pos)",
        "party_dims": {"neidi": len(nm_neidi), "hk": len(nm_hk), "bureau": len(nm_bur), "pay_new": len(nm_pay)},
        "natural_coverage": {"bureau": meta["bureau_coverage"], "pay_new": meta["pay_coverage"]},
        "pos_rate_test": float(y_te.mean()),
        "arms": arms_out,
        "contrasts": contrasts,
    }


def run_santander(cfg: dict) -> dict:
    log("=== 2/3 Santander 产品推荐【构造三方】 ===")
    df = pd.read_parquet(os.path.join(CACHE_DIR, "santander_multimonth_60k.parquet"))
    with open(os.path.join(CACHE_DIR, "santander_feature_groups.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    y = df["y_new_any"].to_numpy(dtype=np.int32)
    idx_all = np.arange(len(df))
    idx_tr, idx_tmp = train_test_split(idx_all, test_size=0.40, stratify=y, random_state=cfg["split_seed"])
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=cfg["split_seed"] + 1)

    df_tr, df_va, df_te = df.iloc[idx_tr], df.iloc[idx_va], df.iloc[idx_te]
    y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]

    X_neidi_tr, X_neidi_va, X_neidi_te, nm_neidi = encode_tabular_train_val_test(df_tr, df_va, df_te, meta["neidi_cols"])
    X_hk_tr, X_hk_va, X_hk_te, nm_hk = encode_tabular_train_val_test(df_tr, df_va, df_te, meta["hk_cols"])
    X_th_tr, X_th_va, X_th_te, nm_th = encode_tabular_train_val_test(df_tr, df_va, df_te, meta["third_cols"])

    arm_specs = {
        "L0_neidi": (X_neidi_tr, X_neidi_va, X_neidi_te),
        "F0_2party_deposit": (
            np.hstack([X_neidi_tr, X_hk_tr]),
            np.hstack([X_neidi_va, X_hk_va]),
            np.hstack([X_neidi_te, X_hk_te]),
        ),
        "F2_neidi_third": (
            np.hstack([X_neidi_tr, X_th_tr]),
            np.hstack([X_neidi_va, X_th_va]),
            np.hstack([X_neidi_te, X_th_te]),
        ),
        "F3_3party_constructed": (
            np.hstack([X_neidi_tr, X_hk_tr, X_th_tr]),
            np.hstack([X_neidi_va, X_hk_va, X_th_va]),
            np.hstack([X_neidi_te, X_hk_te, X_th_te]),
        ),
    }

    arms_out, arm_preds = {}, {}
    for k, (Xtr, Xva, Xte) in arm_specs.items():
        res = equal_tune_and_eval_hgb(Xtr, y_tr, Xva, y_va, Xte, y_te, seeds=cfg["seeds"], max_bins=33)
        arms_out[k] = res
        arm_preds[k] = res["preds_te"]
        log(f"  SAN {k:22s}: AUC={res['summary']['auc_mean']:.4f}±{res['summary']['auc_std']:.4f} "
            f"PR={res['summary']['pr_auc_mean']:.4f} Bag6={res['bagging6_metrics']['auc']:.4f} ({res['elapsed_sec']:.1f}s)")

    contrasts = run_contrasts(
        y_te,
        arm_preds,
        [
            ("F0_2party - L0_neidi", "L0_neidi", "F0_2party_deposit"),
            ("F2_neidi_third - L0_neidi", "L0_neidi", "F2_neidi_third"),
            ("F3_3party - F0_2party", "F0_2party_deposit", "F3_3party_constructed"),
            ("F3_3party - L0_neidi", "L0_neidi", "F3_3party_constructed"),
        ],
        cfg,
    )
    for c in contrasts:
        log(f"  Contrast {c['contrast']:28s}: ΔAUC={c['delta_mean']:+.4f} "
            f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}] p_holm={c['p_holm']:.4f}")

    # Export 10k/5k slice for real SecretFlow SGB confirmation
    sgb_tr_i, _ = train_test_split(np.arange(len(y_tr)), train_size=10000, stratify=y_tr, random_state=cfg["split_seed"])
    sgb_te_i, _ = train_test_split(np.arange(len(y_te)), train_size=5000, stratify=y_te, random_state=cfg["split_seed"])
    np.savez_compressed(
        os.path.join(CACHE_DIR, "exp1_sgb_santander.npz"),
        neidi_tr=X_neidi_tr[sgb_tr_i], neidi_te=X_neidi_te[sgb_te_i],
        hk_tr=X_hk_tr[sgb_tr_i], hk_te=X_hk_te[sgb_te_i],
        third_tr=X_th_tr[sgb_tr_i], third_te=X_th_te[sgb_te_i],
        y_tr=y_tr[sgb_tr_i], y_te=y_te[sgb_te_i],
    )

    return {
        "dataset": "Santander Product Recommendation (60k cohort, 2016-04->2016-05 new_any, 36k/12k/12k)",
        "split_note": "【构造三方】内地(人口统计与渠道) + 构造香港(14个基础/存款账户) + 构造第三方(10个信贷/理财/卡产品)",
        "party_dims": {"neidi": len(nm_neidi), "hk_deposit": len(nm_hk), "third_credit_wealth": len(nm_th)},
        "pos_rate_test": float(y_te.mean()),
        "arms": arms_out,
        "contrasts": contrasts,
    }


def run_uci_bank(cfg: dict) -> dict:
    log("=== 3/3 UCI Bank Marketing（严格剔除 duration）【构造三方】 ===")
    df = pd.read_csv(cfg["datasets"]["uci_bank_csv"], sep=";")
    assert "duration" in df.columns
    df = df.drop(columns=["duration"])  # 严格遵守 AGENTS.md §5 禁区
    y = (df["y"] == "yes").to_numpy(dtype=np.int32)

    neidi_cols = ["age", "job", "marital", "education", "default", "housing", "loan"]
    hk_cols = ["contact", "month", "day_of_week", "campaign", "pdays", "previous", "poutcome"]
    third_cols = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

    idx_all = np.arange(len(df))
    idx_tr, idx_tmp = train_test_split(idx_all, test_size=0.40, stratify=y, random_state=cfg["split_seed"])
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=cfg["split_seed"] + 1)

    df_tr, df_va, df_te = df.iloc[idx_tr], df.iloc[idx_va], df.iloc[idx_te]
    y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]

    X_neidi_tr, X_neidi_va, X_neidi_te, nm_neidi = encode_tabular_train_val_test(df_tr, df_va, df_te, neidi_cols)
    X_hk_tr, X_hk_va, X_hk_te, nm_hk = encode_tabular_train_val_test(df_tr, df_va, df_te, hk_cols)
    X_th_tr, X_th_va, X_th_te, nm_th = encode_tabular_train_val_test(df_tr, df_va, df_te, third_cols)

    arm_specs = {
        "L0_client": (X_neidi_tr, X_neidi_va, X_neidi_te),
        "F0_2party_campaign": (
            np.hstack([X_neidi_tr, X_hk_tr]),
            np.hstack([X_neidi_va, X_hk_va]),
            np.hstack([X_neidi_te, X_hk_te]),
        ),
        "F3_3party_macro": (
            np.hstack([X_neidi_tr, X_hk_tr, X_th_tr]),
            np.hstack([X_neidi_va, X_hk_va, X_th_va]),
            np.hstack([X_neidi_te, X_hk_te, X_th_te]),
        ),
    }

    arms_out, arm_preds = {}, {}
    for k, (Xtr, Xva, Xte) in arm_specs.items():
        res = equal_tune_and_eval_hgb(Xtr, y_tr, Xva, y_va, Xte, y_te, seeds=cfg["seeds"], max_bins=33)
        arms_out[k] = res
        arm_preds[k] = res["preds_te"]
        log(f"  UCI {k:20s}: AUC={res['summary']['auc_mean']:.4f}±{res['summary']['auc_std']:.4f} "
            f"PR={res['summary']['pr_auc_mean']:.4f} ({res['elapsed_sec']:.1f}s)")

    contrasts = run_contrasts(
        y_te,
        arm_preds,
        [
            ("F0_2party - L0_client", "L0_client", "F0_2party_campaign"),
            ("F3_3party_macro - F0_2party", "F0_2party_campaign", "F3_3party_macro"),
            ("F3_3party_macro - L0_client", "L0_client", "F3_3party_macro"),
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
        os.path.join(CACHE_DIR, "exp1_sgb_uci.npz"),
        neidi_tr=X_neidi_tr[sgb_tr_i], neidi_te=X_neidi_te[sgb_te_i],
        hk_tr=X_hk_tr[sgb_tr_i], hk_te=X_hk_te[sgb_te_i],
        third_tr=X_th_tr[sgb_tr_i], third_te=X_th_te[sgb_te_i],
        y_tr=y_tr[sgb_tr_i], y_te=y_te[sgb_te_i],
    )

    return {
        "dataset": "UCI Bank Marketing (41,188 rows, duration strictly dropped, stratified 60/20/20)",
        "split_note": "【构造三方】客户画像 + 银行营销史(去duration) + 外部宏观经济指标；分层切分反映横截面宏观叠加价值（跨年时间切分下宏观指标存在分布漂移，需另行考量）",
        "party_dims": {"client": len(nm_neidi), "campaign": len(nm_hk), "macro": len(nm_th)},
        "pos_rate_test": float(y_te.mean()),
        "arms": arms_out,
        "contrasts": contrasts,
    }


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config()
    out = {
        "experiment": "Exp1_X2_MultiParty_CrossDataset",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": cfg["seeds"],
        "split_seed": cfg["split_seed"],
        "engine": cfg["engines"]["proxy_model"],
        "home_credit": run_home_credit(cfg),
        "santander": run_santander(cfg),
        "uci_bank": run_uci_bank(cfg),
        "total_elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    save_json(out, os.path.join(OUT_DIR, "exp1_x2_multiparty.json"))
    log(f"Exp 1 finished in {out['total_elapsed_sec']:.1f}s -> results/exp1_x2_multiparty.json")


if __name__ == "__main__":
    main()
