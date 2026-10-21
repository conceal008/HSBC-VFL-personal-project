"""[切分] VertiBench 切分的 SGB 实跑复核：联邦 AUC、相对内地单边的增益（与代理 Ĝ 对照）、A1/A2 泄漏对照。

内地单边 / 集中式用与扫描相同的早停明文 XGB 重算（同一批客户级配对重采样）；业务切分用已有的 T2 优化版 SGB 作参照。

    python src/ext/vertibench_confirm.py --pairs vb_lo=vb_lo_sgb_opt vb_hi=vb_hi_sgb_opt business=business_sgb_ou_opt_m42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "ext"))
from vertibench_scan import PairedBoot, auc, fit_es, load  # noqa: E402

ATT = {"vb_lo": "attack_vb_lo.json", "vb_hi": "attack_vb_hi.json", "business": "attack_T2_opt.json"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="tag=run_id")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    X, idx, _, _ = load("business")
    Xv, idxv, _, _ = load("business_val")
    names = list(X.columns)
    tr = idx["is_test_time"].to_numpy() == 0
    te = ~tr
    va = idxv["is_test_time"].to_numpy() == 1
    y = idx["label_new_any"].to_numpy().astype(float)
    yv = idxv["label_new_any"].to_numpy().astype(float)
    Xn, Xvn = X.to_numpy(float), Xv.to_numpy(float)
    D = {"Xa": Xvn[~va], "ya": yv[~va], "Xv": Xvn[va], "yv": yv[va], "Xt": Xn[tr], "yt": y[tr], "Xe": Xn[te]}
    yte, ids = y[te], idx.loc[te, "ncodpers"].to_numpy()
    boot = PairedBoot(yte, ids, args.n_boot)
    s_c, _ = fit_es(np.arange(len(names)), D)
    b_c = boot.aucs(s_c)
    out ={"central_auc": auc(yte, s_c), "rows": []}
    for p in args.pairs:
        tag, run = p.split("=", 1)
        meta = json.loads((REPO_ROOT / f"data/processed/prepared/{tag}_columns.json").read_text())
        gcols = meta["guest_features"]
        gi = np.array([names.index(c) for c in gcols])
        s_g, _ = fit_es(gi, D)
        pred = pd.read_csv(REPO_ROOT / f"data/processed/predictions/{run}_predictions.csv")
        assert np.array_equal(pred["ncodpers"].to_numpy(), ids), f"{run} 行序与测试集不一致"
        s_f = pred["score"].to_numpy(float)
        b_g, b_f = boot.aucs(s_g), boot.aucs(s_f)
        ci = lambda d: [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        row = {"tag": tag, "run": run, "guest_auc": auc(yte, s_g), "fed_auc": auc(yte, s_f),
               "fed_auc_ci": ci(b_f), "fed_minus_guest": auc(yte, s_f) - auc(yte, s_g), "fed_minus_guest_ci": ci(b_f - b_g),
               "fed_minus_central": auc(yte, s_f) - out["central_auc"], "fed_minus_central_ci": ci(b_f - b_c),
               "proxy_gain": out["central_auc"] - auc(yte, s_g)}
        if "scan_row" in meta:
            row["host_share"] = meta["scan_row"]["host_share"]
            row["beta_hat"] = meta["scan_row"]["beta_hat"]
        a = REPO_ROOT / "data/processed/attack" / ATT.get(tag, f"attack_{tag}.json")
        if a.exists():
            A = json.loads(a.read_text())
            row["A1"] = {k: {"own_only": v["own_only"]["mean"], "own_plus_leaves": v["own_plus_leaves"]["mean"],
                             "leakage": v["leakage_auc"]} for k, v in A["A1_by_known_frac"].items()}
            for ph in ("A2_train", "A2_infer"):
                row[ph] = {k: A[ph][k] for k in ("host_split_nodes", "features_used", "features_total",
                                                 "mean_revealed_frac_all", "used_features_revealed_frac")}
            row["trees"] = A["trees"]
        out["rows"].append(row)
        print(f"[{tag}] 联邦 {row['fed_auc']:.4f}｜内地 {row['guest_auc']:.4f}｜联邦−内地 {row['fed_minus_guest']:+.4f} "
              f"[{row['fed_minus_guest_ci'][0]:+.4f}, {row['fed_minus_guest_ci'][1]:+.4f}]（代理 {row['proxy_gain']:+.4f}）"
              f"｜联邦−集中式 {row['fed_minus_central']:+.4f} [{row['fed_minus_central_ci'][0]:+.4f}, {row['fed_minus_central_ci'][1]:+.4f}]")
        if "A1" in row:
            print(f"       A1 泄漏（1% / 5% 已知）{row['A1']['0.01']['leakage']:+.3f} / {row['A1']['0.05']['leakage']:+.3f}"
                  f"（香港自有特征 {row['A1']['0.01']['own_only']:.3f}）｜A2 推理期：香港分裂节点 {row['A2_infer']['host_split_nodes']}，"
                  f"用到 {row['A2_infer']['features_used']}/{row['A2_infer']['features_total']} 列，用到的列还原 {row['A2_infer']['used_features_revealed_frac']:.0%}")
    (REPO_ROOT / "data/processed/vertibench_confirm.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
