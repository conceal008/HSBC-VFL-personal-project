"""[数据] Orange KDD Cup 2009（小版，5 万 × 230 匿名列，高缺失）→ 本流水线 tag 三件套。

口径（如实记录）：
- 官方测试集无标签 → 只用 5 万训练行，分层随机 80/20（seed 42）作训练 / 测试；验证 tag 在训练 80% 内再分层 80/20。
- 匿名列无业务语义 → 按原项目「随机切分」协议在**原始变量**层面切（random.Random(42).shuffle，一半变量给标签方），
  同一原始变量派生的列留在同一方；切分的香港重要性占比 / β̂ 由 vertibench 工具事后定位。
- 只用各方本地可得的信息编码：
  数值列：缺失率 > 95% 或常数的删；训练中位数填充；缺失率 ≥ 5% 加 `{col}__isna`（缺失本身有信息）。
  类别列（Var191–230）：标签方 → 5 折目标编码（平滑 m=100）+ log 频次；非标签方（无标签）→ log 频次 + 频次名次序数（取值 ≤ 30 时）。
- 每个标签一套 tag：orange_{churn,upsell}（+ _val）。

    python src/ext/make_orange_tags.py --src data/raw/orange
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "prepared"
NUM = [f"Var{i}" for i in range(1, 191)]
CAT = [f"Var{i}" for i in range(191, 231)]


def strat_split(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    te = np.zeros(len(y), bool)
    for v in np.unique(y):
        idx = np.flatnonzero(y == v)
        te[rng.choice(idx, size=int(round(len(idx) * frac)), replace=False)] = True
    return te


def target_encode(fit_x: pd.Series, fit_y: np.ndarray, app_x: pd.Series, prior: float, m=100.0, folds=5, seed=7):
    oof = np.empty(len(fit_x))
    part = np.random.default_rng(seed).integers(0, folds, len(fit_x))
    df = pd.DataFrame({"x": fit_x.to_numpy(), "y": fit_y})
    for k in range(folds):
        st = df[part != k].groupby("x")["y"].agg(["sum", "count"])
        enc = (st["sum"] + m * prior) / (st["count"] + m)
        oof[part == k] = df.loc[part == k, "x"].map(enc).fillna(prior).to_numpy()
    st = df.groupby("x")["y"].agg(["sum", "count"])
    enc = (st["sum"] + m * prior) / (st["count"] + m)
    return oof, app_x.map(enc).fillna(prior).to_numpy()


def encode(raw: pd.DataFrame, y: np.ndarray, fit: np.ndarray, guest_vars: set[str]):
    """fit = 拟合编码用的行（训练行）；返回 guest/host 两个 DataFrame（全部行）与列说明。"""
    g, h, meta = {}, {}, {}
    prior = float(y[fit].mean())
    for c in NUM:
        x = raw[c].astype(float)
        miss = float(x[fit].isna().mean())
        if miss > 0.95 or x[fit].nunique() <= 1:
            continue
        dst = g if c in guest_vars else h
        med = float(x[fit].median())
        dst[c] = x.fillna(med).to_numpy()
        if miss >= 0.05:
            dst[f"{c}__isna"] = x.isna().astype(float).to_numpy()
        meta[c] = {"type": "num", "missing": round(miss, 3), "side": "guest" if c in guest_vars else "host"}
    for c in CAT:
        x = raw[c].fillna("__NA__").astype(str)
        miss = float((x[fit] == "__NA__").mean())
        if miss > 0.95 or x[fit].nunique() <= 1:
            continue
        cnt = x[fit].value_counts()
        lf = np.log1p(x.map(cnt).fillna(0).to_numpy())
        if c in guest_vars:
            oof, app = target_encode(x[fit], y[fit], x[~fit], prior)
            te = np.empty(len(x))
            te[fit], te[~fit] = oof, app
            g[f"te_{c}"] = te
            g[f"lf_{c}"] = lf
        else:
            h[f"lf_{c}"] = lf
            if len(cnt) <= 30:
                rank = {v: i for i, v in enumerate(cnt.index)}
                h[f"ord_{c}"] = x.map(rank).fillna(len(cnt)).to_numpy().astype(float) / max(len(cnt) - 1, 1)
        meta[c] = {"type": "cat", "missing": round(miss, 3), "levels": int(len(cnt)), "side": "guest" if c in guest_vars else "host"}
    return pd.DataFrame(g), pd.DataFrame(h), meta


def write(tag, raw, y, train_mask, eval_mask, guest_vars, extra):
    rows = train_mask | eval_mask
    r = raw[rows].reset_index(drop=True)
    yy = y[rows]
    fit = train_mask[rows]
    G, H, meta = encode(r, yy, fit, guest_vars)
    ids = np.flatnonzero(rows)
    G.insert(0, "ncodpers", ids)
    H.insert(0, "ncodpers", ids)
    idx = pd.DataFrame({"ncodpers": ids, "label_new_any": yy.astype(int), "is_test_time": (~fit).astype(int), "is_test_cust": 0})
    G.to_csv(DATA_DIR / f"{tag}_guest.csv", index=False)
    H.to_csv(DATA_DIR / f"{tag}_host.csv", index=False)
    idx.to_csv(DATA_DIR / f"{tag}_index.csv", index=False)
    info = {"split_tag": tag, "source": "Orange KDD Cup 2009 small（训练 5 万行）", **extra,
            "guest_features": list(G.columns[1:]), "host_features": list(H.columns[1:]),
            "n_guest_features": G.shape[1] - 1, "n_host_features": H.shape[1] - 1,
            "train_rows": int(fit.sum()), "eval_rows": int((~fit).sum()),
            "train_pos_rate": float(yy[fit].mean()), "eval_pos_rate": float(yy[~fit].mean()), "variables": meta}
    (DATA_DIR / f"{tag}_columns.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{tag}] 训练 {int(fit.sum()):,}（正例 {int(yy[fit].sum())}）｜评估 {int((~fit).sum()):,}（正例 {int(yy[~fit].sum())}）"
          f"｜内地 {G.shape[1] - 1} 列 / 香港 {H.shape[1] - 1} 列")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/orange")
    ap.add_argument("--labels", nargs="+", default=["churn", "upselling"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    src = REPO_ROOT / args.src
    raw = pd.read_csv(src / "orange_small_train.data", sep="\t", low_memory=False)
    vars_all = NUM + CAT
    order = vars_all[:]
    random.Random(args.seed).shuffle(order)
    guest_vars = set(order[: len(order) // 2])
    for lab in args.labels:
        y = (pd.read_csv(src / f"orange_small_train_{lab}.labels", header=None)[0].to_numpy() == 1).astype(int)
        short = "upsell" if lab == "upselling" else lab
        te = strat_split(y, 0.2, args.seed)
        tr = ~te
        extra = {"label": lab, "split": f"原始变量随机对半（random.Random({args.seed})）", "test": "分层随机 20%（seed 42）"}
        write(f"orange_{short}", raw, y, tr, te, guest_vars, extra)
        va = np.zeros(len(y), bool)
        va[np.flatnonzero(tr)[strat_split(y[tr], 0.2, args.seed + 1)]] = True
        write(f"orange_{short}_val", raw, y, tr & ~va, va, guest_vars, {**extra, "test": "训练 80% 内再分层 20% 作验证"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
