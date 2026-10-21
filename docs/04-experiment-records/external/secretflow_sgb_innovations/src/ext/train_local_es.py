"""[优化④] 明文三臂 + 训练集内部验证早停（同一规则作用于每一臂）。

问题（UCI 验证暴露）：原 train_local.py 固定 100 轮 / lr 0.3 / 不早停。在小数据或弱协作方特征下，
集中式会因过拟合**低于**单边（UCI：集中式 0.7947 < 内地单边 0.8066），使「max(两单边)」主判据失真——
测出来的不是「信息增量」而是「谁更不容易过拟合」。

修正口径（只动轮数选择，其余超参与原版同值）：
- 早停验证集只取自训练集、且模仿测试集切法：有时间维度→训练窗口末尾约 20% 行；否则按客户随机 20%（seed 42）；
- 每臂独立早停：XGBoost early_stopping_rounds=10，评估 AUC，上限 --max-rounds（默认 300）；
- 选好轮数后在全训练集重训（避免验证集样本浪费），再在测试集评估；
- CI / 配对差值完全复用原版 `train_local.cluster_bootstrap_ci`（同一批重采样）。
联邦侧对应：SGB 自带 enable_early_stop / validation_fraction / stopping_rounds（见 train_federated_sgb_opt.py 的 --early-stop）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "cross_sell"))
from train_local import auc, cluster_bootstrap_ci  # noqa: E402  原版口径，不另写

DATA_DIR = REPO_ROOT / "data" / "processed" / "prepared"
OUT_DIR = REPO_ROOT / "data" / "processed"
ID_COL, LABEL_COL = "ncodpers", "label_new_any"


def fit_es(Xtr, y, Xva, yva, Xall, yall, args):
    from xgboost import XGBClassifier
    kw = dict(max_depth=args.depth, learning_rate=args.lr, reg_lambda=args.reg_lambda,
              random_state=args.seed, objective="binary:logistic", tree_method="hist")
    m = XGBClassifier(n_estimators=args.max_rounds, eval_metric="auc", early_stopping_rounds=10, **kw)
    m.fit(Xtr, y, eval_set=[(Xva, yva)], verbose=False)
    best = int(m.best_iteration) + 1
    final = XGBClassifier(n_estimators=best, eval_metric="logloss", **kw).fit(Xall, yall)
    return final, best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="business")
    ap.add_argument("--max-rounds", type=int, default=300)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--reg-lambda", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--val-mode", default="auto", choices=["auto", "time", "cluster"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--compare-with", default=None)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    g = pd.read_csv(DATA_DIR / f"{args.tag}_guest.csv", low_memory=False)
    h = pd.read_csv(DATA_DIR / f"{args.tag}_host.csv", low_memory=False)
    idx = pd.read_csv(DATA_DIR / f"{args.tag}_index.csv", low_memory=False)
    tr = idx["is_test_time"].to_numpy() == 0
    te = ~tr
    # 早停验证集必须模仿测试集的切法：
    #   有时间维度（fecha_dato 多于 1 个值，如 Santander 时间外推）→ 训练窗口最后若干个月（约 val_frac 行）
    #   无时间维度（UCI / Home Credit：一行一客户、随机测试集）→ 训练集内按客户随机 val_frac
    months = idx.loc[tr, "fecha_dato"].astype(str)
    mode = args.val_mode if args.val_mode != "auto" else ("time" if months.nunique() > 1 else "cluster")
    if mode == "time":
        order = sorted(months.unique())
        cum, cut = 0, len(order)
        counts = months.value_counts()
        for m_ in reversed(order):
            if cum >= args.val_frac * len(months):
                break
            cum += int(counts[m_])
            cut -= 1
        va_months = set(order[cut:])
        va = tr & idx["fecha_dato"].astype(str).isin(va_months).to_numpy()
        print(f"  早停验证集（时间）：训练窗口末尾 {sorted(va_months)}，{int(va.sum()):,} 行")
    else:
        cust = idx.loc[tr, ID_COL].unique()
        va_cust = set(np.random.default_rng(args.seed).choice(cust, size=int(len(cust) * args.val_frac), replace=False))
        va = tr & idx[ID_COL].isin(va_cust).to_numpy()
        print(f"  早停验证集（客户随机）：{int(va.sum()):,} 行")
    fit = tr & ~va
    gc = [c for c in g.columns if c != ID_COL]
    hc = [c for c in h.columns if c != ID_COL]
    mats = {"guest": g[gc].to_numpy(np.float64), "host": h[hc].to_numpy(np.float64)}
    mats["central"] = np.hstack([mats["guest"], mats["host"]])
    y = idx[LABEL_COL].to_numpy()
    y_te, te_ids = y[te], idx.loc[te, ID_COL].to_numpy()

    scores, rounds = {}, {}
    for sc, X in mats.items():
        m, best = fit_es(X[fit], y[fit], X[va], y[va], X[tr], y[tr], args)
        scores[sc] = m.predict_proba(X[te])[:, 1]
        rounds[sc] = best
        print(f"  {sc:8s} {X.shape[1]:3d} 列 | 早停轮数 {best:3d} | AUROC = {auc(y_te, scores[sc]):.4f}")
    if args.compare_with:
        ref = pd.read_csv(REPO_ROOT / args.compare_with)
        assert np.array_equal(ref[ID_COL].to_numpy(), te_ids)
        scores["federated"] = ref["score"].to_numpy(np.float64)
    res = cluster_bootstrap_ci(y_te, scores, te_ids, n_boot=args.n_boot, seed=args.seed)
    boot = res.pop("_boot")
    table = {n: {"auroc": a, "ci_lo": lo, "ci_hi": hi} for n, (a, lo, hi) in res.items()}
    best_single = "guest" if res["guest"][0] >= res["host"][0] else "host"
    for arm in [a for a in ("central", "federated") if a in res]:
        d = boot[arm] - boot[best_single]
        table[f"gain_{arm}"] = {"delta": res[arm][0] - res[best_single][0],
                                "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5)),
                                "vs": best_single}
        print(f"  {arm} − max(单边={best_single}) = {table[f'gain_{arm}']['delta']:+.4f} "
              f"[{table[f'gain_{arm}']['ci_lo']:+.4f}, {table[f'gain_{arm}']['ci_hi']:+.4f}]")
    (OUT_DIR / f"train_local_es_{args.tag}_m{args.seed}.json").write_text(json.dumps(
        {"args": vars(args), "rounds": rounds, "metrics": table}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✅ {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
