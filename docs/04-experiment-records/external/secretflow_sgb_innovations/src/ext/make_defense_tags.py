"""[安全] 香港侧数据级防护：在香港本地变换后再参与联邦（训练与推理都用变换后的特征）。

针对 A2（内地从香港分裂划分还原产品持有）：
- hostcnt：数据最小化 —— 24 个产品持有标记 → 1 列「持有产品数」，保留 ind_actividad_cliente。
  分裂只能揭示「产品数 ≥ k」，不再揭示具体持有哪一款产品（PIPL 最小必要原则的技术落地）。
- hostrr{ε}：本地随机响应 —— 每个产品标记以 p = 1/(1+e^ε) 独立翻转（逐比特 ε-LDP；24 列合计按组合定理为 24ε）。
  被还原的是加噪后的标记，真值只能以 1−p 的准确率被猜中。

    python src/ext/make_defense_tags.py --tag business --mode hostcnt
    python src/ext/make_defense_tags.py --tag business --mode hostrr --eps 2
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
PRODUCT = re.compile(r"^ind_.*_ult1$")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="business")
    ap.add_argument("--mode", required=True, choices=["hostcnt", "hostrr"])
    ap.add_argument("--eps", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    h = pd.read_csv(DATA_DIR / f"{args.tag}_host.csv")
    prods = [c for c in h.columns if PRODUCT.match(c)]
    if args.mode == "hostcnt":
        out_tag = f"{args.tag}_hostcnt"
        new = pd.DataFrame({"ncodpers": h["ncodpers"], "n_products": h[prods].sum(axis=1).astype(float)})
        for c in h.columns:
            if c not in prods and c != "ncodpers":
                new[c] = h[c]
        info = {"defense": "hostcnt", "products_aggregated": len(prods)}
    else:
        out_tag = f"{args.tag}_hostrr{args.eps:g}"
        p = 1.0 / (1.0 + np.exp(args.eps))
        rng = np.random.default_rng(args.seed)
        new = h.copy()
        flips = rng.random((len(h), len(prods))) < p
        new[prods] = np.where(flips, 1.0 - h[prods].to_numpy(), h[prods].to_numpy())
        info = {"defense": "hostrr", "eps_per_bit": args.eps, "flip_p": p, "cells_flipped": int(flips.sum())}
    new.to_csv(DATA_DIR / f"{out_tag}_host.csv", index=False)
    for suf in ("guest.csv", "index.csv"):
        (DATA_DIR / f"{out_tag}_{suf}").write_bytes((DATA_DIR / f"{args.tag}_{suf}").read_bytes())
    meta = json.loads((DATA_DIR / f"{args.tag}_columns.json").read_text(encoding="utf-8"))
    meta.update({"split_tag": out_tag, "derived_from": args.tag, "host_features": [c for c in new.columns if c != "ncodpers"],
                 "n_host_features": new.shape[1] - 1, **info})
    (DATA_DIR / f"{out_tag}_columns.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{out_tag}] 香港侧 {new.shape[1] - 1} 列 | {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
