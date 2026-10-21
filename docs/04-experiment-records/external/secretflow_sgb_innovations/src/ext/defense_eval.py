"""[安全] 防护的「效用代价」：各防护版联邦模型在测试集上的 AUC，与无防护基线 S0 做客户级配对 bootstrap。

同时给出相对「内地单边默认 XGB（原仓库口径：100 轮 / depth5 / lr0.3 / λ0.1）」的联邦增益，
看加了防护以后，联邦是否还值得做。

    python src/ext/defense_eval.py --run S0=business_sgb_S0_base --run S1=business_sgb_S1_ft0 ...
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
PRED = REPO_ROOT / "data" / "processed" / "predictions"
OUT = REPO_ROOT / "data" / "processed" / "attack"


def main(argv=None) -> int:
    from xgboost import XGBClassifier

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="name=run_id（第一个为基线）")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    g = pd.read_csv(DATA_DIR / "business_guest.csv")
    idx = pd.read_csv(DATA_DIR / "business_index.csv")
    tr = idx["is_test_time"].to_numpy() == 0
    te = ~tr
    y = idx["label_new_any"].to_numpy()
    te_ids = idx.loc[te, "ncodpers"].to_numpy()
    X = g.drop(columns="ncodpers").to_numpy(float)
    m = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.3, reg_lambda=0.1, random_state=args.seed,
                      objective="binary:logistic", eval_metric="logloss", tree_method="hist").fit(X[tr], y[tr])
    scores = {"guest_default": m.predict_proba(X[te])[:, 1]}
    names = []
    for spec in args.run:
        name, rid = spec.split("=", 1)
        ref = pd.read_csv(PRED / f"{rid}_predictions.csv")
        assert np.array_equal(ref["ncodpers"].to_numpy(), te_ids), f"{name} 行序与测试集不一致"
        scores[name] = ref["score"].to_numpy(float)
        names.append(name)
    res = cluster_bootstrap_ci(y[te], scores, te_ids, n_boot=args.n_boot, seed=args.seed)
    boot = res.pop("_boot")
    base = names[0]
    table = {}
    print(f"  {'方案':10s} {'AUROC':>7s}   95% CI            Δ vs {base}（配对）          联邦增益 vs 内地单边")
    for n in ["guest_default", *names]:
        a, lo, hi = res[n]
        row = {"auroc": a, "ci_lo": lo, "ci_hi": hi}
        line = f"  {n:10s} {a:.4f}  [{lo:.4f}, {hi:.4f}]"
        if n in names:
            for ref, key in ((base, "delta_vs_base"), ("guest_default", "gain_vs_guest")):
                d = boot[n] - boot[ref]
                row[key] = {"delta": a - res[ref][0], "ci_lo": float(np.percentile(d, 2.5)),
                            "ci_hi": float(np.percentile(d, 97.5))}
                line += f"  {row[key]['delta']:+.4f} [{row[key]['ci_lo']:+.4f}, {row[key]['ci_hi']:+.4f}]"
        table[n] = row
        print(line)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "defense_utility.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
