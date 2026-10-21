"""[优化] 标签方本地特征筛选 —— 隐语 SGB 里标签方自己的分桶求和也走 HEU 编码大整数路径，
成本 ∝ 样本数 × 标签方列数（IEEE-CIS 内地 360 列时每棵树约 290 s）。筛选只用标签方自有特征 + 自有标签，不涉及对方数据。

做法：在调参 tag 的训练段上训练内地单边 XGB，按总增益（total_gain）排序取前 K 列；在验证段比较
内地单边验证 AUC，选「不低于全量 0.001 以内」的最小 K（判据只用标签方本地信息）；
集中式（内地前 K 列 + 香港全部列）验证 AUC 只作事后核对，不参与选择。
随后落一套只换内地列的新 tag（香港侧不变）。

    python src/ext/guest_feature_select.py --tag ieee --val-tag ieee_val --ks 30 50 80 120 --out-prefix ieee_k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "prepared"


def main(argv=None) -> int:
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ieee")
    ap.add_argument("--val-tag", default="ieee_val")
    ap.add_argument("--ks", type=int, nargs="+", default=[30, 50, 80, 120])
    ap.add_argument("--tol", type=float, default=0.001)
    ap.add_argument("--rounds", type=int, default=120)
    ap.add_argument("--out-prefix", default="ieee_k")
    args = ap.parse_args(argv)

    gv = pd.read_csv(DATA_DIR / f"{args.val_tag}_guest.csv")
    hv = pd.read_csv(DATA_DIR / f"{args.val_tag}_host.csv")
    iv = pd.read_csv(DATA_DIR / f"{args.val_tag}_index.csv")
    va = iv["is_test_time"].to_numpy() == 1
    y = iv["label_new_any"].to_numpy()
    gcols = list(gv.columns[1:])
    G, H = gv[gcols].to_numpy(float), hv.drop(columns="ncodpers").to_numpy(float)
    P = {"objective": "binary:logistic", "eta": 0.1, "max_depth": 6, "lambda": 10.0, "tree_method": "hist",
         "max_bin": 256, "nthread": 2, "seed": 42}

    def fit_auc(X):
        b = xgb.train(P, xgb.DMatrix(X[~va], y[~va]), args.rounds)
        return float(roc_auc_score(y[va], b.predict(xgb.DMatrix(X[va])))), b

    full_g, bg = fit_auc(G)
    full_c, _ = fit_auc(np.hstack([G, H]))
    gain = bg.get_score(importance_type="total_gain")
    order = sorted(range(len(gcols)), key=lambda j: -gain.get(f"f{j}", 0.0))
    print(f"全量 内地 {len(gcols)} 列：内地单边 {full_g:.4f}｜集中式 {full_c:.4f}｜有增益的列 {len(gain)}")
    res = {"full": {"k": len(gcols), "guest_val": full_g, "central_val": full_c}, "ks": []}
    for k in args.ks:
        sel = order[:k]
        ag, _ = fit_auc(G[:, sel])
        ac, _ = fit_auc(np.hstack([G[:, sel], H]))
        res["ks"].append({"k": k, "guest_val": ag, "central_val": ac, "central_gain_val": ac - ag})
        print(f"K={k:3d}：内地单边 {ag:.4f}（{ag - full_g:+.4f}）｜集中式 {ac:.4f}（{ac - full_c:+.4f}）｜验证增益 {ac - ag:+.4f}")
    ok = [r for r in res["ks"] if r["guest_val"] >= full_g - args.tol]
    best = min(ok, key=lambda r: r["k"]) if ok else res["full"]
    res["chosen_k"] = best["k"]
    chosen = [gcols[j] for j in order[: best["k"]]]
    res["chosen_cols"] = chosen
    print(f"选定 K = {best['k']}")
    tag_out = f"{args.out_prefix}{best['k']}"
    for src, dst in ((args.tag, tag_out), (args.val_tag, f"{tag_out}_val")):
        g = pd.read_csv(DATA_DIR / f"{src}_guest.csv")
        g[["ncodpers", *chosen]].to_csv(DATA_DIR / f"{dst}_guest.csv", index=False, float_format="%.6g")
        for part in ("host", "index"):
            pd.read_csv(DATA_DIR / f"{src}_{part}.csv").to_csv(DATA_DIR / f"{dst}_{part}.csv", index=False, float_format="%.6g")
        meta = json.loads((DATA_DIR / f"{src}_columns.json").read_text(encoding="utf-8"))
        meta.update({"split_tag": dst, "source_tag": src, "guest_features": chosen, "n_guest_features": len(chosen),
                     "guest_selection": f"标签方本地按内地单边 XGB total_gain 取前 {best['k']} 列（{args.val_tag} 训练段）"})
        (DATA_DIR / f"{dst}_columns.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"写出 {dst}")
    (REPO_ROOT / f"data/processed/guest_select_{args.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
