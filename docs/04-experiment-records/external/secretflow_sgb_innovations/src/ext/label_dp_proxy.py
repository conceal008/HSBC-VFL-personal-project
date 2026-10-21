"""[安全] 标签 DP 变体的明文代理筛选：集中式 XGB（与 SGB 默认同超参、近似同分桶）在各变体的私有化标签上训练，
看测试 AUC，先挑出值得花联邦算力的变体。SGB 与明文直方图 GBDT 在同分桶下等价，排序可参考。

    python src/ext/label_dp_proxy.py --eps 1 2 4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "prepared"
spec = importlib.util.spec_from_file_location("sgbopt", REPO_ROOT / "src" / "ext" / "train_federated_sgb_opt.py")
sgbopt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sgbopt)


def main(argv=None) -> int:
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="business")
    ap.add_argument("--eps", type=float, nargs="+", default=[1, 2, 4])
    ap.add_argument("--modes", nargs="+", default=["rr", "rr_soft", "rrprior", "rrprior_soft"])
    ap.add_argument("--max-bin", type=int, default=12, help="近似 SGB 默认 sketch_eps 0.1（约 10 桶）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--reg-lambda", type=float, default=0.1)
    ap.add_argument("--colsample", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--out", default="data/processed/attack/label_dp_proxy.json")
    args = ap.parse_args(argv)

    g = pd.read_csv(DATA_DIR / f"{args.tag}_guest.csv")
    h = pd.read_csv(DATA_DIR / f"{args.tag}_host.csv")
    idx = pd.read_csv(DATA_DIR / f"{args.tag}_index.csv")
    tr = idx["is_test_time"].to_numpy() == 0
    te = ~tr
    y = idx["label_new_any"].to_numpy().astype(float)
    G = g.drop(columns="ncodpers").to_numpy(float)
    X = np.hstack([G, h.drop(columns="ncodpers").to_numpy(float)])
    params = {"objective": "binary:logistic", "max_depth": args.depth, "eta": args.lr, "lambda": args.reg_lambda,
              "colsample_bytree": args.colsample, "tree_method": "hist", "max_bin": args.max_bin, "seed": args.seed, "nthread": 2}

    def fit_eval(lab, Xm=X):
        bst = xgb.train(params, xgb.DMatrix(Xm[tr], label=lab), num_boost_round=args.rounds)
        return float(roc_auc_score(y[te], bst.predict(xgb.DMatrix(Xm[te]))))

    res = {"true_labels_central": fit_eval(y[tr]), "true_labels_guest_only": fit_eval(y[tr], G)}
    print(f"真实标签：集中式 {res['true_labels_central']:.4f}｜内地单边 {res['true_labels_guest_only']:.4f}")
    for e in args.eps:
        for mode in args.modes:
            lab = sgbopt.label_dp(G[tr], y[tr].reshape(-1, 1), e, mode, 0.3, args.seed).ravel()
            auc = fit_eval(lab)
            res[f"eps{e:g}_{mode}"] = auc
            print(f"ε={e:g} {mode:13s} 测试 AUC {auc:.4f}")
    (REPO_ROOT / args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
