"""
实验 4（Exp 4）：X3 价值 − 风险期望利润/价值排序跨数据集验证（营销场景全面补齐）。
此前 X3 仅在信贷（Home Credit）与反诈（IEEE-CIS）上验证；本实验在三个营销数据集上系统检验：
1. Hillstrom 邮件营销（64,000 样本，自带真实消费金额 spend（美元）+ 随机 A/B 邮件分组！）：
   - 构造两方：主动方（区域/新客/渠道/营销组）+ 被动方（历史消费额/近度/品类偏好）
   - 预测 visit 与 conversion（含 L0 / F0 / M6 Bagging）
   - 联合条件客单价回归 E[spend | conv=1]，对比「概率排序」vs「X3 期望消费额排序 P(conv)*E[spend|conv] - cost」
     在 Top 5% / 10% / 20% 下捕获的真实消费额（美元）与净收益，并用随机分组检验真实因果增收（Treated - Control）！
2. Santander 产品推荐（60,000 客户，真实年收入 renta + 分层产品价值【假设】）：
   - 对比 L0 / F0(两方) / F3(三方) 与「概率排序」vs「X3 期望价值排序」在 Top 5% / 10% / 20% 下的捕获价值与净收益。
3. COIL2000 保险营销（官方 5,822 训练 / 4,000 冻结测试，原表购买力档位 MKOOPKLA 构造佣金价值【假设】）：
   - 对比 L0 / F0 / M6 与「概率排序」vs「X3 期望价值排序」在 Top 5% / 10% / 20% 下的捕获价值与净收益。
"""
from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
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
    print(f"[{time.strftime('%H:%M:%S')}] [Exp4] {msg}", flush=True)


def topk_value_metrics(
    realized_value: np.ndarray,
    scores: np.ndarray,
    contact_cost: float,
    fracs=(0.05, 0.10, 0.20),
) -> dict:
    """Compute realized gross value, net profit, and capture share at Top-K%."""
    val = np.asarray(realized_value, dtype=np.float64)
    sc = np.asarray(scores, dtype=np.float64)
    n = len(val)
    total_val = max(float(val.sum()), 1e-12)
    order = np.argsort(-sc)
    out = {}
    for fr in fracs:
        k = max(1, int(round(n * fr)))
        sel = order[:k]
        gross = float(val[sel].sum())
        net = float(gross - k * contact_cost)
        out[f"top{int(fr*100)}"] = {
            "k": k,
            "gross_value": gross,
            "net_profit": net,
            "gross_per_targeted": gross / k,
            "net_per_targeted": net / k,
            "value_capture_share": gross / total_val,
        }
    return out


def paired_boot_topk_value(
    realized_value: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    contact_cost: float,
    frac: float = 0.10,
    n_boot: int = 1000,
    seed: int = 20261119,
) -> dict:
    """Two-level paired bootstrap (seeds × samples) for Top-K% net profit / gross value delta."""
    val = np.asarray(realized_value, dtype=np.float64)
    sa = np.asarray(scores_a, dtype=np.float64)
    sb = np.asarray(scores_b, dtype=np.float64)
    n_seeds, n = sa.shape
    k = max(1, int(round(n * frac)))

    def net_at_k(v, s):
        idx = np.argpartition(-s, k - 1)[:k]
        return float(v[idx].sum() - k * contact_cost)

    pa = [net_at_k(val, sa[i]) for i in range(n_seeds)]
    pb = [net_at_k(val, sb[i]) for i in range(n_seeds)]
    d0 = float(np.mean(pb) - np.mean(pa))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vb = val[idx]
        s_idx = rng.integers(0, n_seeds, size=n_seeds)
        uniq_s, mult = np.unique(s_idx, return_counts=True)
        acc = 0.0
        for s, w in zip(uniq_s, mult):
            acc += w * (net_at_k(vb, sb[s, idx]) - net_at_k(vb, sa[s, idx]))
        deltas[b] = acc / n_seeds

    p_le = (int(np.sum(deltas <= 0.0)) + 1) / (n_boot + 1)
    p_ge = (int(np.sum(deltas >= 0.0)) + 1) / (n_boot + 1)
    return {
        "frac": frac,
        "k": k,
        "base_net_mean": float(np.mean(pa)),
        "new_net_mean": float(np.mean(pb)),
        "delta_net_mean": d0,
        "delta_pct_over_gross_base": float(d0 / max(np.mean(pa) + k * contact_cost, 1e-9)),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "p_raw": float(min(1.0, 2.0 * min(p_le, p_ge))),
    }


def run_hillstrom(cfg: dict) -> dict:
    log("=== 1/3 Hillstrom 邮件营销（真实消费额 spend + 随机 A/B 检验） ===")
    df = pd.read_csv(cfg["datasets"]["hillstrom_csv"])
    cost = cfg["financial_assumptions"]["hillstrom"]["email_contact_cost_usd"]

    # Constructed 2-party vertical split:
    # Active (neidi_mkt): channel, zip_code, newbie, segment (email treatment arm)
    # Passive (hk_ecom): recency, history_segment, history, mens, womens
    a_cols = ["zip_code", "newbie", "channel", "segment"]
    b_cols = ["recency", "history_segment", "history", "mens", "womens"]

    y_conv = df["conversion"].to_numpy(dtype=np.int32)
    y_visit = df["visit"].to_numpy(dtype=np.int32)
    spend = df["spend"].to_numpy(dtype=np.float64)
    is_treated = (df["segment"] != "No E-Mail").to_numpy(dtype=np.int32)

    idx_all = np.arange(len(df))
    idx_tr, idx_tmp = train_test_split(idx_all, test_size=0.40, stratify=y_conv, random_state=cfg["split_seed"])
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=y_conv[idx_tmp], random_state=cfg["split_seed"] + 1)

    d_tr, d_va, d_te = df.iloc[idx_tr], df.iloc[idx_va], df.iloc[idx_te]
    Xa_tr, Xa_va, Xa_te, _ = encode_tabular_train_val_test(d_tr, d_va, d_te, a_cols)
    Xb_tr, Xb_va, Xb_te, _ = encode_tabular_train_val_test(d_tr, d_va, d_te, b_cols)
    Xab_tr, Xab_va, Xab_te = np.hstack([Xa_tr, Xb_tr]), np.hstack([Xa_va, Xb_va]), np.hstack([Xa_te, Xb_te])

    # 1) Visit task (high base rate 14.7% — also verifies I2+I3 & M6 on Hillstrom!)
    vis_L0 = equal_tune_and_eval_hgb(Xa_tr, y_visit[idx_tr], Xa_va, y_visit[idx_va], Xa_te, y_visit[idx_te], seeds=cfg["seeds"], max_bins=33)
    vis_F0 = equal_tune_and_eval_hgb(Xab_tr, y_visit[idx_tr], Xab_va, y_visit[idx_va], Xab_te, y_visit[idx_te], seeds=cfg["seeds"], max_bins=33)
    vis_delta = paired_bootstrap_delta(y_visit[idx_te], vis_L0["preds_te"], vis_F0["preds_te"], n_boot=cfg["bootstrap_iters"], seed=cfg["bootstrap_seed"])
    log(f"  Hillstrom VISIT: L0={vis_L0['summary']['auc_mean']:.4f} -> F0={vis_F0['summary']['auc_mean']:.4f} "
        f"(Δ={vis_delta['delta_mean']:+.4f}, Bag6={vis_F0['bagging6_metrics']['auc']:.4f})")

    # 2) Conversion task (0.90% pos)
    conv_L0 = equal_tune_and_eval_hgb(Xa_tr, y_conv[idx_tr], Xa_va, y_conv[idx_va], Xa_te, y_conv[idx_te], seeds=cfg["seeds"], max_bins=33)
    conv_F0 = equal_tune_and_eval_hgb(Xab_tr, y_conv[idx_tr], Xab_va, y_conv[idx_va], Xab_te, y_conv[idx_te], seeds=cfg["seeds"], max_bins=33)
    conv_delta = paired_bootstrap_delta(y_conv[idx_te], conv_L0["preds_te"], conv_F0["preds_te"], n_boot=cfg["bootstrap_iters"], seed=cfg["bootstrap_seed"])
    log(f"  Hillstrom CONV : L0={conv_L0['summary']['auc_mean']:.4f} -> F0={conv_F0['summary']['auc_mean']:.4f} "
        f"(Δ={conv_delta['delta_mean']:+.4f}, Bag6={conv_F0['bagging6_metrics']['auc']:.4f})")

    # 3) Conditional Spend Regressor E[spend | conv=1] fitted strictly on train+val positive converters
    # Use log1p(spend) regularized Ridge + HGB blend to avoid overfitting small converter sample (~460 rows)
    pos_tr = np.concatenate([idx_tr[y_conv[idx_tr] == 1], idx_va[y_conv[idx_va] == 1]])
    X_pos = np.vstack([Xab_tr[y_conv[idx_tr] == 1], Xab_va[y_conv[idx_va] == 1]])
    y_spend_pos = spend[pos_tr]
    X_pos_clean = np.nan_to_num(X_pos, nan=0.0)
    Xab_te_clean = np.nan_to_num(Xab_te, nan=0.0)
    mean_sp = float(y_spend_pos.mean())

    # Fit per-seed conditional spend predictor (blend of Ridge on log1p + HGBRegressor on spend)
    est_spend_te = np.empty((len(cfg["seeds"]), len(idx_te)), dtype=np.float64)
    for i, s in enumerate(cfg["seeds"]):
        reg_h = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=60, l2_regularization=10.0, min_samples_leaf=15, random_state=int(s))
        reg_h.fit(X_pos, np.log1p(y_spend_pos))
        p_h = np.expm1(np.clip(reg_h.predict(Xab_te), 0.0, 8.0))
        reg_r = Ridge(alpha=50.0)
        reg_r.fit(X_pos_clean, np.log1p(y_spend_pos))
        p_r = np.expm1(np.clip(reg_r.predict(Xab_te_clean), 0.0, 8.0))
        # Shrink 50% toward empirical conditional mean for variance stability
        est_spend_te[i] = np.clip(0.35 * p_h + 0.35 * p_r + 0.30 * mean_sp, 10.0, 1000.0)

    # Also build a direct E[spend] score using F0_prob * est_spend - cost vs pure F0_prob vs L0_prob
    ev_F0_te = conv_F0["preds_te"] * est_spend_te - cost
    spend_te = spend[idx_te]
    trt_te = is_treated[idx_te]

    # Summary Top-K realized spend (seed-averaged)
    val_summary = {}
    for name, sc in [("L0_prob", conv_L0["preds_te"]), ("F0_prob", conv_F0["preds_te"]), ("F0_X3_EV", ev_F0_te)]:
        per_s = [topk_value_metrics(spend_te, sc[i], cost) for i in range(len(cfg["seeds"]))]
        val_summary[name] = {
            tk: {
                m: float(np.mean([ps[tk][m] for ps in per_s]))
                for m in ("gross_value", "net_profit", "gross_per_targeted", "net_per_targeted", "value_capture_share")
            }
            for tk in ("top5", "top10", "top20")
        }

    # Bootstrap contrasts on Top 5%, 10%, 20% net profit (real USD spend!)
    val_contrasts = []
    for fr in (0.05, 0.10, 0.20):
        c1 = paired_boot_topk_value(spend_te, conv_L0["preds_te"], conv_F0["preds_te"], cost, frac=fr, n_boot=cfg["bootstrap_iters"], seed=cfg["bootstrap_seed"])
        c1["contrast"] = f"Top{int(fr*100)}%: F0_prob - L0_prob"
        c2 = paired_boot_topk_value(spend_te, conv_F0["preds_te"], ev_F0_te, cost, frac=fr, n_boot=cfg["bootstrap_iters"], seed=cfg["bootstrap_seed"])
        c2["contrast"] = f"Top{int(fr*100)}%: F0_X3_EV - F0_prob"
        c3 = paired_boot_topk_value(spend_te, conv_L0["preds_te"], ev_F0_te, cost, frac=fr, n_boot=cfg["bootstrap_iters"], seed=cfg["bootstrap_seed"])
        c3["contrast"] = f"Top{int(fr*100)}%: F0_X3_EV - L0_prob"
        val_contrasts.extend([c1, c2, c3])

    p_adj = holm_bonferroni([c["p_raw"] for c in val_contrasts])
    for c, pa in zip(val_contrasts, p_adj):
        c["p_holm"] = float(pa)
        c["significant_holm_05"] = bool(pa < cfg["alpha"])
        log(f"  Hillstrom {c['contrast']:30s}: ΔNet={c['delta_net_mean']:+.1f} USD "
            f"[{c['ci95_low']:+.1f}, {c['ci95_high']:+.1f}] p_holm={c['p_holm']:.4f}")

    # Bonus: Genuine causal incremental spend (Treated - Control per targeted customer) in Top 10% & 20%
    causal_check = {}
    for name, sc in [("L0_prob", conv_L0["preds_te"].mean(axis=0)), ("F0_prob", conv_F0["preds_te"].mean(axis=0)), ("F0_X3_EV", ev_F0_te.mean(axis=0))]:
        order = np.argsort(-sc)
        causal_check[name] = {}
        for fr in (0.10, 0.20):
            k = int(round(len(spend_te) * fr))
            sel = order[:k]
            sp_t = float(spend_te[sel][trt_te[sel] == 1].mean()) if np.any(trt_te[sel] == 1) else 0.0
            sp_c = float(spend_te[sel][trt_te[sel] == 0].mean()) if np.any(trt_te[sel] == 0) else 0.0
            causal_check[name][f"top{int(fr*100)}"] = {
                "treated_mean_spend_usd": round(sp_t, 4),
                "control_mean_spend_usd": round(sp_c, 4),
                "causal_ate_spend_usd": round(sp_t - sp_c, 4),
                "n_treated": int(np.sum(trt_te[sel] == 1)),
                "n_control": int(np.sum(trt_te[sel] == 0)),
            }

    return {
        "dataset": "Hillstrom Email Marketing (64,000 rows, real USD spend + randomized A/B treatment)",
        "assumption_tag": cfg["financial_assumptions"]["hillstrom"]["tag"],
        "contact_cost_usd": cost,
        "total_test_real_spend_usd": float(spend_te.sum()),
        "visit_auc": {"L0": vis_L0["summary"], "F0": vis_F0["summary"], "F0_bag6": vis_F0["bagging6_metrics"], "delta": vis_delta},
        "conversion_auc": {"L0": conv_L0["summary"], "F0": conv_F0["summary"], "F0_bag6": conv_F0["bagging6_metrics"], "delta": conv_delta},
        "topk_value_summary": val_summary,
        "value_contrasts": val_contrasts,
        "causal_ab_spend_check": causal_check,
    }


def run_santander_x3(cfg: dict) -> dict:
    log("=== 2/3 Santander 产品推荐 X3 期望价值排序（真实客户年收入 renta 加权【假设】） ===")
    df = pd.read_parquet(os.path.join(CACHE_DIR, "santander_multimonth_60k.parquet"))
    with open(os.path.join(CACHE_DIR, "santander_feature_groups.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    y = df["y_new_any"].to_numpy(dtype=np.int32)
    y_cnt = df["y_new_cnt"].to_numpy(dtype=np.float64)
    idx_all = np.arange(len(df))
    idx_tr, idx_tmp = train_test_split(idx_all, test_size=0.40, stratify=y, random_state=cfg["split_seed"])
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, stratify=y[idx_tmp], random_state=cfg["split_seed"] + 1)

    d_tr, d_va, d_te = df.iloc[idx_tr], df.iloc[idx_va], df.iloc[idx_te]
    y_tr, y_va, y_te = y[idx_tr], y[idx_va], y[idx_te]

    Xn_tr, Xn_va, Xn_te, _ = encode_tabular_train_val_test(d_tr, d_va, d_te, meta["neidi_cols"])
    Xh_tr, Xh_va, Xh_te, _ = encode_tabular_train_val_test(d_tr, d_va, d_te, meta["hk_cols"])
    Xt_tr, Xt_va, Xt_te, _ = encode_tabular_train_val_test(d_tr, d_va, d_te, meta["third_cols"])

    L0 = equal_tune_and_eval_hgb(Xn_tr, y_tr, Xn_va, y_va, Xn_te, y_te, seeds=cfg["seeds"], max_bins=33)
    F0 = equal_tune_and_eval_hgb(np.hstack([Xn_tr, Xh_tr]), y_tr, np.hstack([Xn_va, Xh_va]), y_va, np.hstack([Xn_te, Xh_te]), y_te, seeds=cfg["seeds"], max_bins=33)
    F3 = equal_tune_and_eval_hgb(np.hstack([Xn_tr, Xh_tr, Xt_tr]), y_tr, np.hstack([Xn_va, Xh_va, Xt_va]), y_va, np.hstack([Xn_te, Xh_te, Xt_te]), y_te, seeds=cfg["seeds"], max_bins=33)

    # Customer value based on real renta (training distribution empirical CDF -> 0.5x to 2.5x multiplier × 120 EUR base【假设】)
    base_val = cfg["financial_assumptions"]["santander"]["base_product_value_eur"]
    cost = cfg["financial_assumptions"]["santander"]["contact_cost_eur"]
    renta_tr = pd.to_numeric(d_tr["renta"], errors="coerce").to_numpy(dtype=np.float64)
    med_renta = float(np.nanmedian(renta_tr))
    sorted_tr_renta = np.sort(np.where(np.isnan(renta_tr), med_renta, renta_tr))

    renta_te = pd.to_numeric(d_te["renta"], errors="coerce").to_numpy(dtype=np.float64)
    renta_te_imp = np.where(np.isnan(renta_te), med_renta, renta_te)
    pct_te = np.searchsorted(sorted_tr_renta, renta_te_imp, side="right") / max(len(sorted_tr_renta), 1)
    # Potential customer value multiplier (0.5x to 2.5x) known at decision time from active party's renta!
    cust_val_te = base_val * (0.5 + 2.0 * pct_te)
    realized_val_te = cust_val_te * y_cnt[idx_te]

    ev_F0_te = F0["preds_te"] * cust_val_te[None, :] - cost
    ev_F3_te = F3["preds_te"] * cust_val_te[None, :] - cost

    val_summary = {}
    for name, sc in [("L0_prob", L0["preds_te"]), ("F0_prob", F0["preds_te"]), ("F0_X3_EV", ev_F0_te), ("F3_prob", F3["preds_te"]), ("F3_X3_EV", ev_F3_te)]:
        per_s = [topk_value_metrics(realized_val_te, sc[i], cost) for i in range(len(cfg["seeds"]))]
        val_summary[name] = {
            tk: {m: float(np.mean([ps[tk][m] for ps in per_s])) for m in ("gross_value", "net_profit", "value_capture_share")}
            for tk in ("top5", "top10", "top20")
        }

    val_contrasts = []
    for fr in (0.05, 0.10, 0.20):
        c1 = paired_boot_topk_value(realized_val_te, F0["preds_te"], ev_F0_te, cost, frac=fr, n_boot=cfg["bootstrap_iters"], seed=cfg["bootstrap_seed"])
        c1["contrast"] = f"Top{int(fr*100)}%: F0_X3_EV - F0_prob"
        c2 = paired_boot_topk_value(realized_val_te, F3["preds_te"], ev_F3_te, cost, frac=fr, n_boot=cfg["bootstrap_iters"], seed=cfg["bootstrap_seed"])
        c2["contrast"] = f"Top{int(fr*100)}%: F3_X3_EV - F3_prob"
        c3 = paired_boot_topk_value(realized_val_te, L0["preds_te"], ev_F3_te, cost, frac=fr, n_boot=cfg["bootstrap_iters"], seed=cfg["bootstrap_seed"])
        c3["contrast"] = f"Top{int(fr*100)}%: F3_X3_EV - L0_prob"
        val_contrasts.extend([c1, c2, c3])

    p_adj = holm_bonferroni([c["p_raw"] for c in val_contrasts])
    for c, pa in zip(val_contrasts, p_adj):
        c["p_holm"] = float(pa)
        c["significant_holm_05"] = bool(pa < cfg["alpha"])
        log(f"  Santander {c['contrast']:30s}: ΔNet={c['delta_net_mean']:+.1f} EUR "
            f"[{c['ci95_low']:+.1f}, {c['ci95_high']:+.1f}] p_holm={c['p_holm']:.4f}")

    return {
        "dataset": "Santander Product Recommendation (60k cohort, 12k frozen test)",
        "assumption_tag": cfg["financial_assumptions"]["santander"]["tag"],
        "base_product_value_eur": base_val,
        "contact_cost_eur": cost,
        "total_test_realized_value_eur": float(realized_val_te.sum()),
        "auc_summary": {"L0": L0["summary"]["auc_mean"], "F0": F0["summary"]["auc_mean"], "F3": F3["summary"]["auc_mean"]},
        "topk_value_summary": val_summary,
        "value_contrasts": val_contrasts,
    }


def run_coil2000_x3(cfg: dict) -> dict:
    log("=== 3/3 COIL2000 保险营销 X3 期望价值排序（购买力档位佣金价值【假设】） ===")
    cdir = cfg["datasets"]["coil2000_dir"]
    df_tr_full = pd.read_csv(os.path.join(cdir, "ticdata2000.txt"), sep="\t", header=None)
    X_ev_full = pd.read_csv(os.path.join(cdir, "ticeval2000.txt"), sep="\t", header=None).to_numpy(dtype=np.float32)
    y_te = pd.read_csv(os.path.join(cdir, "tictgts2000.txt"), sep="\t", header=None).iloc[:, 0].to_numpy(dtype=np.int32)

    X_tr_all = df_tr_full.iloc[:, :85].to_numpy(dtype=np.float32)
    y_tr_all = df_tr_full.iloc[:, 85].to_numpy(dtype=np.int32)

    # 80/20 tune split strictly inside official train, refit on full 5,822 train, eval on official 4,000 test
    i_tr, i_va = train_test_split(np.arange(len(y_tr_all)), test_size=0.20, stratify=y_tr_all, random_state=cfg["split_seed"])

    Xa_tr, Xa_va, Xa_te = X_tr_all[i_tr, :43], X_tr_all[i_va, :43], X_ev_full[:, :43]
    Xab_tr, Xab_va, Xab_te = X_tr_all[i_tr, :85], X_tr_all[i_va, :85], X_ev_full[:, :85]

    L0 = equal_tune_and_eval_hgb(Xa_tr, y_tr_all[i_tr], Xa_va, y_tr_all[i_va], Xa_te, y_te, seeds=cfg["seeds"], max_bins=33, refit_train_val=True)
    F0 = equal_tune_and_eval_hgb(Xab_tr, y_tr_all[i_tr], Xab_va, y_tr_all[i_va], Xab_te, y_te, seeds=cfg["seeds"], max_bins=33, refit_train_val=True)
    auc_delta = paired_bootstrap_delta(y_te, L0["preds_te"], F0["preds_te"], n_boot=cfg["bootstrap_iters"], seed=cfg["bootstrap_seed"])
    log(f"  COIL2000 AUC: L0={L0['summary']['auc_mean']:.4f} -> F0={F0['summary']['auc_mean']:.4f} "
        f"(Δ={auc_delta['delta_mean']:+.4f}, Bag6={F0['bagging6_metrics']['auc']:.4f})")

    # Value weighting【假设】: Purchasing power class MKOOPKLA (col 42, scale 1..8) -> commission value 150 * (0.5 + 0.2 * MKOOPKLA)
    base_val = cfg["financial_assumptions"]["coil2000"]["base_caravan_commission_eur"]
    cost = cfg["financial_assumptions"]["coil2000"]["contact_cost_eur"]
    pp_te = X_ev_full[:, 42].astype(np.float64)
    cust_val_te = base_val * (0.5 + 0.2 * pp_te)
    realized_val_te = cust_val_te * y_te

    ev_F0_te = F0["preds_te"] * cust_val_te[None, :] - cost
    val_summary = {}
    for name, sc in [("L0_prob", L0["preds_te"]), ("F0_prob", F0["preds_te"]), ("F0_X3_EV", ev_F0_te)]:
        per_s = [topk_value_metrics(realized_val_te, sc[i], cost) for i in range(len(cfg["seeds"]))]
        val_summary[name] = {
            tk: {m: float(np.mean([ps[tk][m] for ps in per_s])) for m in ("gross_value", "net_profit", "value_capture_share")}
            for tk in ("top5", "top10", "top20")
        }

    val_contrasts = []
    for fr in (0.05, 0.10, 0.20):
        c1 = paired_boot_topk_value(realized_val_te, L0["preds_te"], F0["preds_te"], cost, frac=fr, n_boot=cfg["bootstrap_iters"], seed=cfg["bootstrap_seed"])
        c1["contrast"] = f"Top{int(fr*100)}%: F0_prob - L0_prob"
        c2 = paired_boot_topk_value(realized_val_te, F0["preds_te"], ev_F0_te, cost, frac=fr, n_boot=cfg["bootstrap_iters"], seed=cfg["bootstrap_seed"])
        c2["contrast"] = f"Top{int(fr*100)}%: F0_X3_EV - F0_prob"
        c3 = paired_boot_topk_value(realized_val_te, L0["preds_te"], ev_F0_te, cost, frac=fr, n_boot=cfg["bootstrap_iters"], seed=cfg["bootstrap_seed"])
        c3["contrast"] = f"Top{int(fr*100)}%: F0_X3_EV - L0_prob"
        val_contrasts.extend([c1, c2, c3])

    p_adj = holm_bonferroni([c["p_raw"] for c in val_contrasts])
    for c, pa in zip(val_contrasts, p_adj):
        c["p_holm"] = float(pa)
        c["significant_holm_05"] = bool(pa < cfg["alpha"])
        log(f"  COIL2000 {c['contrast']:30s}: ΔNet={c['delta_net_mean']:+.1f} EUR "
            f"[{c['ci95_low']:+.1f}, {c['ci95_high']:+.1f}] p_holm={c['p_holm']:.4f}")

    return {
        "dataset": "COIL2000 Insurance Marketing (official 5,822 train / 4,000 frozen eval)",
        "assumption_tag": cfg["financial_assumptions"]["coil2000"]["tag"],
        "base_caravan_commission_eur": base_val,
        "contact_cost_eur": cost,
        "total_test_realized_value_eur": float(realized_val_te.sum()),
        "auc": {"L0": L0["summary"], "F0": F0["summary"], "F0_bag6": F0["bagging6_metrics"], "delta": auc_delta},
        "topk_value_summary": val_summary,
        "value_contrasts": val_contrasts,
    }


def main() -> None:
    t0 = time.perf_counter()
    cfg = load_config()
    out = {
        "experiment": "Exp4_X3_ExpectedValue_Marketing_CrossDataset",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": cfg["seeds"],
        "split_seed": cfg["split_seed"],
        "engine": cfg["engines"]["proxy_model"],
        "hillstrom": run_hillstrom(cfg),
        "santander": run_santander_x3(cfg),
        "coil2000": run_coil2000_x3(cfg),
        "total_elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    save_json(out, os.path.join(OUT_DIR, "exp4_x3_expected_value.json"))
    log(f"Exp 4 finished in {out['total_elapsed_sec']:.1f}s -> results/exp4_x3_expected_value.json")


if __name__ == "__main__":
    main()
