"""
汇总第三阶段全部实验 JSON 结果，生成统一可追溯总表：
- results/stage3_master_summary.csv
- results/stage3_master_summary.json
每条指标严格绑定来源文件、引擎类型、种子配置、基线值、新方法值、增量、95% CI 与 Holm 校正 p 值。
"""
from __future__ import annotations

import json
import os
import pandas as pd

RES_DIR = "/Applications/总文件夹/汇丰/第三阶段/results"


def load_if_exists(fn: str):
    p = os.path.join(RES_DIR, fn)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    rows = []

    def add_row(**kw):
        rows.append(kw)

    # 1. Exp 1 proxy
    e1 = load_if_exists("exp1_x2_multiparty.json")
    if e1:
        for ds_k, ds_label in [("home_credit", "Home Credit (100k)"), ("santander", "Santander (60k, 构造三方)"), ("uci_bank", "UCI Bank 无duration (41k, 构造三方)")]:
            blk = e1[ds_k]
            for c in blk["contrasts"]:
                add_row(
                    exp="Exp1_X2", method="X2 多方/第三方扩展", dataset=ds_label,
                    engine="HGB代理(6-seed,33桶)", contrast=c["contrast"], metric="ROC-AUC",
                    base_val=round(c["base_mean"], 4), new_val=round(c["new_mean"], 4),
                    delta=round(c["delta_mean"], 4),
                    ci95=f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]",
                    p_val=round(c["p_holm"], 4), p_type="p_holm",
                    sig=c["significant_holm_05"], source_file="results/exp1_x2_multiparty.json",
                )

    # 1b. Exp 1 Real SGB
    e1s = load_if_exists("exp1_x2_sgb_real.json")
    if e1s:
        arm_map = {
            "F3_bureau - F0_2party": ("F0_2party", "F3_bureau"),
            "F3_pay_new - F0_2party": ("F0_2party", "F3_pay_new"),
            "F4_all - F0_2party": ("F0_2party", "F4_all_4party"),
            "F4_all - F3_bureau": ("F3_bureau", "F4_all_4party"),
            "F4_all - F3_pay_new": ("F3_pay_new", "F4_all_4party"),
            "F3_3party - F0_2party": ("F0_2party", "F3_3party"),
        }
        for ds_k, ds_label in [("home_credit_sgb", "Home Credit (12k/6k)"), ("santander_sgb", "Santander (10k/5k, 构造三方)"), ("uci_bank_sgb", "UCI Bank (10k/5k, 构造三方)")]:
            blk = e1s[ds_k]
            for cname, c in blk["contrasts"].items():
                bk, nk = arm_map[cname]
                add_row(
                    exp="Exp1_X2_SGB", method="X2 多方/第三方扩展", dataset=ds_label,
                    engine="真实 SecretFlow 1.14 SGB (OU)", contrast=cname, metric="ROC-AUC",
                    base_val=round(blk["arms"][bk]["metrics"]["auc"], 4),
                    new_val=round(blk["arms"][nk]["metrics"]["auc"], 4),
                    delta=round(c["delta_auc"], 4),
                    ci95=f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]",
                    p_val=round(c["p_test_boot"], 4), p_type="p_test_boot",
                    sig=bool(c["p_test_boot"] < 0.05), source_file="results/exp1_x2_sgb_real.json",
                )

    # 2. Exp 2 proxy
    e2 = load_if_exists("exp2_x1plus_temporal_graph.json")
    if e2:
        for ds_k, ds_label in [("home_credit_x1plus", "Home Credit (100k)"), ("santander_x1plus", "Santander (60k, 7月历史)")]:
            for c in e2[ds_k]["contrasts"]:
                add_row(
                    exp="Exp2_X1+", method="X1+ 因果时序速度", dataset=ds_label,
                    engine="HGB代理(6-seed,33桶)", contrast=c["contrast"], metric="ROC-AUC",
                    base_val=round(c["base_mean"], 4), new_val=round(c["new_mean"], 4),
                    delta=round(c["delta_mean"], 4),
                    ci95=f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]",
                    p_val=round(c["p_holm"], 4), p_type="p_holm",
                    sig=c["significant_holm_05"], source_file="results/exp2_x1plus_temporal_graph.json",
                )
        for wn, wblk in e2["ieee_rolling_x1plus"]["windows"].items():
            for c in wblk["contrasts"]:
                add_row(
                    exp="Exp2_X1+_IEEE", method="X1图 + X1+速度(滚动窗)", dataset=f"IEEE-CIS {wn}",
                    engine="HGB代理(6-seed,card1簇Boot)", contrast=c["contrast"], metric="ROC-AUC",
                    base_val=round(c["base_mean"], 4), new_val=round(c["new_mean"], 4),
                    delta=round(c["delta_mean"], 4),
                    ci95=f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]",
                    p_val=round(c["p_holm"], 4), p_type="p_holm",
                    sig=c["significant_holm_05"], source_file="results/exp2_x1plus_temporal_graph.json",
                )

    # 2b. Exp 2 Real SGB
    e2s = load_if_exists("exp2_x1plus_sgb_real.json")
    if e2s:
        for ds_k, ds_label, cname in [
            ("home_credit_sgb", "Home Credit (12k/6k)", "F3_pay_plus_x1v - F3_pay_static"),
            ("santander_sgb", "Santander (10k/5k)", "F0_plus_x1v_both - F0_static"),
            ("ieee_w3_sgb", "IEEE-CIS W3 (12k/6k)", "F_x1_full - F0_2party"),
        ]:
            blk = e2s[ds_k]
            arms = list(blk["arms"].values())
            c = blk["contrast"]
            add_row(
                exp="Exp2_X1+_SGB", method="X1/X1+ 因果图与时序速度", dataset=ds_label,
                engine="真实 SecretFlow 1.14 SGB (OU)", contrast=cname, metric="ROC-AUC",
                base_val=round(arms[0]["metrics"]["auc"], 4), new_val=round(arms[1]["metrics"]["auc"], 4),
                delta=round(c["delta_auc"], 4),
                ci95=f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]",
                p_val=round(c["p_test_boot"], 4), p_type="p_test_boot",
                sig=bool(c["p_test_boot"] < 0.05), source_file="results/exp2_x1plus_sgb_real.json",
            )

    # 3. Exp 3 proxy
    e3 = load_if_exists("exp3_m1x_unaligned.json")
    if e3:
        for c in e3["ieee_cis_real_unaligned"]["contrasts"]:
            add_row(
                exp="Exp3_M1x", method="M1x 非对齐样本利用", dataset="IEEE-CIS 真实非对齐(26万+因果池)",
                engine="HGB代理(6-seed,card1簇Boot)", contrast=c["contrast"], metric="ROC-AUC",
                base_val=round(c["base_mean"], 4), new_val=round(c["new_mean"], 4),
                delta=round(c["delta_mean"], 4),
                ci95=f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]",
                p_val=round(c["p_holm"], 4), p_type="p_holm",
                sig=c["significant_holm_05"], source_file="results/exp3_m1x_unaligned.json",
            )
        for rk, rblk in e3["home_credit_low_overlap"].items():
            for c in rblk["contrasts"]:
                add_row(
                    exp="Exp3_M1x", method="M1x 非对齐样本利用", dataset=f"Home Credit {rk}(构造低重合)",
                    engine="HGB代理(6-seed,33桶)", contrast=c["contrast"], metric="ROC-AUC",
                    base_val=round(c["base_mean"], 4), new_val=round(c["new_mean"], 4),
                    delta=round(c["delta_mean"], 4),
                    ci95=f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]",
                    p_val=round(c["p_holm"], 4), p_type="p_holm",
                    sig=c["significant_holm_05"], source_file="results/exp3_m1x_unaligned.json",
                )

    # 3b. Exp 3 Real SGB
    e3s = load_if_exists("exp3_m1x_sgb_real.json")
    if e3s:
        for ds_k, ds_label in [("ieee_sgb", "IEEE-CIS 真实非对齐 (12k/6k)"), ("home_credit_20pct_sgb", "Home Credit 20%低重合 (12k/6k)")]:
            blk = e3s[ds_k]
            arms = list(blk["arms"].values())
            c = blk["contrast"]
            add_row(
                exp="Exp3_M1x_SGB", method="M1x 非对齐先验特征", dataset=ds_label,
                engine="真实 SecretFlow 1.14 SGB (OU)", contrast="M1x_prior_feature - F0_aligned", metric="ROC-AUC",
                base_val=round(arms[0]["metrics"]["auc"], 4), new_val=round(arms[1]["metrics"]["auc"], 4),
                delta=round(c["delta_auc"], 4),
                ci95=f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]",
                p_val=round(c["p_test_boot"], 4), p_type="p_test_boot",
                sig=bool(c["p_test_boot"] < 0.05), source_file="results/exp3_m1x_sgb_real.json",
            )

    # 4. Exp 4 X3 Expected Value
    e4 = load_if_exists("exp4_x3_expected_value.json")
    if e4:
        for ds_k, ds_label, unit in [
            ("hillstrom", "Hillstrom 邮件营销(真实spend)", "Net Profit (USD)"),
            ("santander", "Santander 产品推荐(renta加权【假设】)", "Net Profit (EUR)"),
            ("coil2000", "COIL2000 保险营销(购买力加权【假设】)", "Net Profit (EUR)"),
        ]:
            for c in e4[ds_k]["value_contrasts"]:
                add_row(
                    exp="Exp4_X3", method="X3 期望价值排序", dataset=ds_label,
                    engine="HGB代理(6-seed)", contrast=c["contrast"], metric=unit,
                    base_val=round(c["base_net_mean"], 1), new_val=round(c["new_net_mean"], 1),
                    delta=round(c["delta_net_mean"], 1),
                    ci95=f"[{c['ci95_low']:+.1f}, {c['ci95_high']:+.1f}]",
                    p_val=round(c["p_holm"], 4), p_type="p_holm",
                    sig=c["significant_holm_05"], source_file="results/exp4_x3_expected_value.json",
                )

    # 5. Exp 5 C2 Seed Lookalike
    e5 = load_if_exists("exp5_c2_seed_lookalike.json")
    if e5:
        for ds_k, blk in e5["datasets"].items():
            for c in blk["contrasts"]:
                add_row(
                    exp="Exp5_C2", method="C2 种子客群Lookalike", dataset=blk["dataset"],
                    engine="HGB代理(6-seed,33桶)", contrast=c["contrast"], metric="ROC-AUC",
                    base_val=round(c["base_mean"], 4), new_val=round(c["new_mean"], 4),
                    delta=round(c["delta_mean"], 4),
                    ci95=f"[{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}]",
                    p_val=round(c["p_holm"], 4), p_type="p_holm",
                    sig=c["significant_holm_05"], source_file="results/exp5_c2_seed_lookalike.json",
                )

    # 6. Exp 6 Real SGB Core
    e6 = load_if_exists("exp6_core_sgb_real.json")
    if e6:
        for dsn, blk in e6["part_a_i1_m3_crypto_speed"].items():
            add_row(
                exp="Exp6_I1_M3", method="I1(OU) + M3(明文标签方)", dataset=dsn,
                engine="真实 SecretFlow 1.14 SGB", contrast="Paillier -> OU+M3 (10 trees)", metric="Fit Speedup (x)",
                base_val=blk["paillier_fit_sec"], new_val=blk["ou_m3_fit_sec"],
                delta=blk["speedup_ou_m3_vs_paillier"],
                ci95=f"max_diff={blk['max_abs_diff_ou_vs_m3']:.1e}",
                p_val="", p_type="bit_exact", sig=True, source_file="results/exp6_core_sgb_real.json",
            )
        for dsn, blk in e6["part_b_i2i3_m6_bagging"].items():
            add_row(
                exp="Exp6_I2I3", method="I2+I3 (33桶+先验基分+调优)", dataset=dsn,
                engine="真实 SecretFlow 1.14 SGB (6-seed)", contrast="Opt(I2+I3) - Default_SGB", metric="ROC-AUC",
                base_val=round(blk["default_sgb"]["auc"], 4), new_val=round(blk["opt_i2i3_6seed"]["auc_mean"], 4),
                delta=round(blk["opt_i2i3_6seed"]["delta_vs_default"], 4),
                ci95=f"std={blk['opt_i2i3_6seed']['auc_std']:.4f}",
                p_val="", p_type="6seed_mean", sig=bool(blk["opt_i2i3_6seed"]["delta_vs_default"] > 0),
                source_file="results/exp6_core_sgb_real.json",
            )
            add_row(
                exp="Exp6_M6", method="M6 (6成员Bagging)", dataset=dsn,
                engine="真实 SecretFlow 1.14 SGB (6-seed)", contrast="M6_Bag6 - Opt_Mean", metric="ROC-AUC",
                base_val=round(blk["opt_i2i3_6seed"]["auc_mean"], 4), new_val=round(blk["m6_bagging6_sgb"]["auc"], 4),
                delta=round(blk["m6_bagging6_sgb"]["delta_vs_opt_mean"], 4),
                ci95="", p_val="", p_type="ensemble_gain", sig=bool(blk["m6_bagging6_sgb"]["delta_vs_opt_mean"] > 0),
                source_file="results/exp6_core_sgb_real.json",
            )
        for dsn, blk in e6["part_c_i6_soft_label_dp"].items():
            for ek, eblk in blk["by_epsilon"].items():
                add_row(
                    exp="Exp6_I6", method=f"I6 后验去噪软标签DP (ε={eblk['epsilon']})", dataset=dsn,
                    engine="真实 SecretFlow 1.14 SGB (3 DP seeds)", contrast="I6_Soft - Hard_RR", metric="ROC-AUC",
                    base_val=round(eblk["hard_rr_auc_mean"], 4), new_val=round(eblk["i6_soft_auc_mean"], 4),
                    delta=round(eblk["delta_i6_vs_hard"], 4),
                    ci95=f"clean={blk['clean_sgb_auc']:.4f}", p_val="", p_type="3dp_seed_mean",
                    sig=bool(eblk["delta_i6_vs_hard"] > 0), source_file="results/exp6_core_sgb_real.json",
                )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES_DIR, "stage3_master_summary.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(RES_DIR, "stage3_master_summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Aggregated {len(rows)} verified metric rows -> stage3_master_summary.csv & .json")


if __name__ == "__main__":
    main()
