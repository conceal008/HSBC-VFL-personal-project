"""[优化] 明文 XGBoost 代理在验证集 tag 上调参 —— 内地单边 / 香港单边 / 集中式各自独立调，保证对比公平。

为什么用代理：SGB 与明文直方图 GBDT 数学等价（分桶相同时），一次 SGB 约 12–16 分钟，
一次 XGB 约数秒；先用代理选配置，再用 SGB 确认。
为什么三臂都调：只调联邦而让单边保持默认超参，会把「调参收益」错记成「联邦增益」。

选择口径：验证 AUC（训练窗口末 3 个月，不碰测试集）；早停 20 轮，轮数上限 --max-rounds。
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "prepared"
OUT = REPO_ROOT / "data" / "processed"

GRID = {
    "learning_rate": [0.3, 0.1],
    "max_depth": [3, 4, 5, 6],
    "reg_lambda": [0.1, 1.0, 10.0],
    "colsample_bytree": [1.0, 0.7],
}


def main(argv=None) -> int:
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="business_val")
    ap.add_argument("--max-rounds", type=int, default=300)
    ap.add_argument("--max-bin", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    g = pd.read_csv(DATA_DIR / f"{args.tag}_guest.csv")
    h = pd.read_csv(DATA_DIR / f"{args.tag}_host.csv")
    idx = pd.read_csv(DATA_DIR / f"{args.tag}_index.csv")
    va = idx["is_test_time"].to_numpy() == 1
    y = idx["label_new_any"].to_numpy()
    Xs = {"guest": g.drop(columns="ncodpers").to_numpy(float), "host": h.drop(columns="ncodpers").to_numpy(float)}
    Xs["central"] = np.hstack([Xs["guest"], Xs["host"]])

    keys = list(GRID)
    res: dict[str, list] = {}
    t0 = time.perf_counter()
    for arm, X in Xs.items():
        rows = []
        for vals in itertools.product(*GRID.values()):
            p = dict(zip(keys, vals))
            m = XGBClassifier(n_estimators=args.max_rounds, early_stopping_rounds=20, eval_metric="auc",
                              tree_method="hist", max_bin=args.max_bin, random_state=args.seed, **p)
            m.fit(X[~va], y[~va], eval_set=[(X[va], y[va])], verbose=False)
            auc = float(roc_auc_score(y[va], m.predict_proba(X[va], iteration_range=(0, m.best_iteration + 1))[:, 1]))
            rows.append({**p, "rounds": int(m.best_iteration) + 1, "val_auc": auc})
        rows.sort(key=lambda r: -r["val_auc"])
        res[arm] = rows
        b = rows[0]
        print(f"[{arm:7s}] 最优 val AUC {b['val_auc']:.4f} | lr {b['learning_rate']} depth {b['max_depth']} "
              f"lambda {b['reg_lambda']} colsample {b['colsample_bytree']} rounds {b['rounds']}"
              f"（{len(rows)} 组，累计 {time.perf_counter() - t0:.0f}s）")
        for r in rows[1:4]:
            print(f"           次优 {r['val_auc']:.4f} | lr {r['learning_rate']} depth {r['max_depth']} "
                  f"lambda {r['reg_lambda']} colsample {r['colsample_bytree']} rounds {r['rounds']}")
    # 原始默认配置在验证集上的位置（100 轮 / depth5 / lr0.3 / lambda0.1 / colsample1）
    base = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.3, reg_lambda=0.1, tree_method="hist",
                         max_bin=args.max_bin, random_state=args.seed, eval_metric="logloss")
    for arm, X in Xs.items():
        base.fit(X[~va], y[~va])
        res[f"{arm}_default_val_auc"] = float(roc_auc_score(y[va], base.predict_proba(X[va])[:, 1]))
        print(f"[{arm:7s}] 原默认配置 val AUC {res[f'{arm}_default_val_auc']:.4f}")
    (OUT / f"tune_proxy_{args.tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
