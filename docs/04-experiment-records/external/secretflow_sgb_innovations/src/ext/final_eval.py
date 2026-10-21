"""[优化] 测试集终评（只评一次）：调参后的明文三臂 + 多个联邦预测文件，客户级配对 bootstrap。

选配置规则（各臂相同，在看到测试集之前固定）：验证集最优 AUC 的 0.001 以内，取「轮数 × 深度」最小者
（SGB 逐层训练，每层一次全样本密文累加，成本 ∝ 轮数 × 深度）。
配对 CI 复用原版 train_local.cluster_bootstrap_ci（同一批重采样）。

    python src/ext/final_eval.py --fed orig=data/processed/predictions/business_sgb_federated_r100_m42_predictions.csv \
                                 --fed opt=data/processed/predictions/business_sgb_federated_ou_opt_r262_m42_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "cross_sell"))
from train_local import cluster_bootstrap_ci  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "processed" / "prepared"
OUT = REPO_ROOT / "data" / "processed"


def pick(rows: list[dict], tol: float = 0.001) -> dict:
    best = rows[0]["val_auc"]
    cand = [r for r in rows if r["val_auc"] >= best - tol]
    return min(cand, key=lambda r: r["rounds"] * r["max_depth"])


def main(argv=None) -> int:
    from xgboost import XGBClassifier

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="business")
    ap.add_argument("--tune", default="data/processed/tune_proxy_business_val.json")
    ap.add_argument("--fed", action="append", default=[], help="name=predictions.csv，可多次")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    tune = json.loads((REPO_ROOT / args.tune).read_text())
    g = pd.read_csv(DATA_DIR / f"{args.tag}_guest.csv")
    h = pd.read_csv(DATA_DIR / f"{args.tag}_host.csv")
    idx = pd.read_csv(DATA_DIR / f"{args.tag}_index.csv")
    tr = idx["is_test_time"].to_numpy() == 0
    te = ~tr
    y = idx["label_new_any"].to_numpy()
    Xs = {"guest": g.drop(columns="ncodpers").to_numpy(float), "host": h.drop(columns="ncodpers").to_numpy(float)}
    Xs["central"] = np.hstack([Xs["guest"], Xs["host"]])
    te_ids = idx.loc[te, "ncodpers"].to_numpy()

    scores, cfgs = {}, {}
    for arm, X in Xs.items():
        c = pick(tune[arm])
        cfgs[arm] = c
        m = XGBClassifier(n_estimators=c["rounds"], max_depth=c["max_depth"], learning_rate=c["learning_rate"],
                          reg_lambda=c["reg_lambda"], colsample_bytree=c["colsample_bytree"], tree_method="hist",
                          max_bin=256, random_state=args.seed, eval_metric="logloss").fit(X[tr], y[tr])
        scores[f"{arm}_tuned"] = m.predict_proba(X[te])[:, 1]
        print(f"[{arm}_tuned] 配置 {c}")
    for spec in args.fed:
        name, path = spec.split("=", 1)
        ref = pd.read_csv(REPO_ROOT / path)
        assert np.array_equal(ref["ncodpers"].to_numpy(), te_ids), f"{name} 行序与测试集不一致"
        scores[f"fed_{name}"] = ref["score"].to_numpy(float)
    res = cluster_bootstrap_ci(y[te], scores, te_ids, n_boot=args.n_boot, seed=args.seed)
    boot = res.pop("_boot")
    table = {n: {"auroc": a, "ci_lo": lo, "ci_hi": hi} for n, (a, lo, hi) in res.items()}
    for n, (a, lo, hi) in res.items():
        print(f"  {n:22s} AUROC {a:.4f} [{lo:.4f}, {hi:.4f}]")
    best_single = max(("guest_tuned", "host_tuned"), key=lambda k: res[k][0])
    pairs = [(f, best_single) for f in scores if f.startswith("fed_")]
    pairs += [(a, b) for a in scores if a.startswith("fed_") for b in scores if b.startswith("fed_") and a < b]
    pairs += [(f, "central_tuned") for f in scores if f.startswith("fed_")]
    for a, b in pairs:
        d = boot[a] - boot[b]
        table[f"{a}_minus_{b}"] = {"delta": res[a][0] - res[b][0], "ci_lo": float(np.percentile(d, 2.5)),
                                   "ci_hi": float(np.percentile(d, 97.5))}
        t = table[f"{a}_minus_{b}"]
        print(f"  {a} − {b}: {t['delta']:+.4f} [{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}]")
    (OUT / f"final_eval_{args.tag}.json").write_text(json.dumps({"configs": cfgs, "metrics": table}, indent=2),
                                                  encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
