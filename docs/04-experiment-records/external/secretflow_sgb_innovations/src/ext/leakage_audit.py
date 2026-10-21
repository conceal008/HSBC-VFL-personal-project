"""[优化②] 标签同源 / 泄漏审计 —— 在跑任何联邦实验之前执行。

动机（Santander 复现得出）：业务切分把 24 个「本月产品持有」放到香港侧，而标签公式是
「下月持有 且 本月未持有」—— 香港侧特征本身就是标签定义的一半。联邦增益几乎全部来自这些列
（原仓库 EXP-012：剥掉产品列后联邦·稳健 0.8584 < 内地单边 0.8667）。这类「同源列」会让
「联邦带来增益」的结论被高估，必须在实验前显式标出。

三道检查（只用训练集，不碰测试集）：
1. 单变量 AUC：每列单独对标签的 AUC（取 max(a, 1−a)）；≥ --strong（默认 0.70）标「强单列」
2. 血缘声明：--lineage 显式列出参与标签公式的源列（如 Santander 的 24 个产品列）→ 标「同源」
3. 侧别汇总：每侧的强单列 / 同源列数量与最大单变量 AUC；同源列落在协作方 → 给出 ⚠ 结论

用法：
    python src/ext/leakage_audit.py --tag business --lineage-regex '^ind_.*_ult1$'
    python src/ext/leakage_audit.py --tag uci_business
    python src/ext/leakage_audit.py --raw-csv raw/uci/bank-additional-full.csv --sep ';' --label-expr "y=='yes'" --drop y
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "prepared"
OUT_DIR = REPO_ROOT / "data" / "processed" / "audit_ext"


def uni_auc(x: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        x = np.where(np.isnan(x), np.nanmedian(x), x)
    if np.unique(x).size < 2:
        return 0.5
    a = roc_auc_score(y, x)
    return float(max(a, 1 - a))


def audit_frame(X: pd.DataFrame, y: np.ndarray, side: dict[str, str], strong: float, lineage: set[str]) -> dict:
    rows = []
    for c in X.columns:
        rows.append({"feature": c, "side": side.get(c, "-"), "uni_auc": uni_auc(X[c].to_numpy(), y),
                     "lineage": c in lineage})
    df = pd.DataFrame(rows).sort_values("uni_auc", ascending=False).reset_index(drop=True)
    df["strong"] = df["uni_auc"] >= strong
    summ = {}
    for s, g in df.groupby("side"):
        summ[s] = {"n": int(len(g)), "max_uni_auc": float(g.uni_auc.max()),
                   "n_strong": int(g.strong.sum()), "n_lineage": int(g.lineage.sum()),
                   "top": g.head(5)[["feature", "uni_auc"]].round(4).values.tolist()}
    warn = []
    if "host" in summ and summ["host"]["n_lineage"]:
        warn.append(f"协作方含 {summ['host']['n_lineage']} 个标签同源列 → 联邦增益会被高估，须报「剥掉同源列」的稳健臂")
    if "guest" in summ and summ["guest"]["n_lineage"]:
        warn.append(f"标签方含 {summ['guest']['n_lineage']} 个标签同源列 → 单边基线会被抬高")
    leak = df[df.uni_auc >= 0.95]
    if len(leak):
        warn.append(f"单变量 AUC ≥0.95 的列 {leak.feature.tolist()} → 疑似直接泄漏，应剔除")
    return {"table": df, "summary": summ, "warnings": warn}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="标签同源 / 泄漏审计")
    ap.add_argument("--tag", default=None, help="prepared/ 下的切分 tag")
    ap.add_argument("--raw-csv", default=None, help="或直接审计原始表（相对仓库上一级目录）")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--label-expr", default=None, help="原始表标签表达式，如 \"y=='yes'\"")
    ap.add_argument("--drop", default="", help="原始表中不参与审计的列（逗号分隔）")
    ap.add_argument("--lineage-regex", default=None, help="参与标签公式的源列（正则）")
    ap.add_argument("--strong", type=float, default=0.70)
    args = ap.parse_args(argv)

    if args.tag:
        g = pd.read_csv(DATA_DIR / f"{args.tag}_guest.csv")
        h = pd.read_csv(DATA_DIR / f"{args.tag}_host.csv")
        idx = pd.read_csv(DATA_DIR / f"{args.tag}_index.csv")
        tr = idx["is_test_time"].to_numpy() == 0
        X = pd.concat([g.drop(columns="ncodpers"), h.drop(columns="ncodpers")], axis=1).loc[tr]
        y = idx.loc[tr, "label_new_any"].to_numpy()
        side = {c: "guest" for c in g.columns} | {c: "host" for c in h.columns}
        name = args.tag
    else:
        raw = pd.read_csv(REPO_ROOT.parent / args.raw_csv, sep=args.sep, low_memory=False)
        y = raw.eval(args.label_expr).astype(int).to_numpy()
        X = raw.drop(columns=[c for c in args.drop.split(",") if c])
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]):   # 类别列：按类别正例率编码后再算 AUC（仅审计用）
                key = X[c].astype(str)
                X[c] = key.map(pd.Series(y, index=X.index).groupby(key).mean())
        side = {c: "raw" for c in X.columns}
        name = Path(args.raw_csv).stem
    lineage = {c for c in X.columns if args.lineage_regex and re.search(args.lineage_regex, c)}
    res = audit_frame(X, y, side, args.strong, lineage)

    print(f"[{name}] 训练行 {len(X):,} / 正例率 {y.mean():.4f} / 特征 {X.shape[1]}")
    for s, v in res["summary"].items():
        print(f"  {s:6s} {v['n']:3d} 列 | 最大单变量 AUC {v['max_uni_auc']:.4f} | 强单列(≥{args.strong}) {v['n_strong']}"
              f" | 同源列 {v['n_lineage']} | top: {v['top'][:3]}")
    for w in res["warnings"]:
        print(f"  ⚠ {w}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res["table"].to_csv(OUT_DIR / f"leakage_{name}.csv", index=False)
    (OUT_DIR / f"leakage_{name}.json").write_text(json.dumps(
        {"name": name, "summary": res["summary"], "warnings": res["warnings"]}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
