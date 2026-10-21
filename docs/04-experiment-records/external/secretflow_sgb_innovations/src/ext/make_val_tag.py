"""[优化] 从某个 prepared tag 派生「验证集 tag」：只保留训练窗口，把窗口末尾若干个月标为 is_test_time=1。

用途：所有调参 / 开关选择都在这个 tag 上做，测试集（2016-02 之后）只在最终评估时碰一次。
产物与原 prepared/ 同格式，原版 train_local.py / train_federated_sgb.py 可直接读。

    python src/ext/make_val_tag.py --tag business --val-months 3      # → business_val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "prepared"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="business")
    ap.add_argument("--val-months", type=int, default=3)
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args(argv)
    out = args.out_tag or f"{args.tag}_val"

    g = pd.read_csv(DATA_DIR / f"{args.tag}_guest.csv", low_memory=False)
    h = pd.read_csv(DATA_DIR / f"{args.tag}_host.csv", low_memory=False)
    idx = pd.read_csv(DATA_DIR / f"{args.tag}_index.csv", low_memory=False)
    tr = (idx["is_test_time"] == 0).to_numpy()
    months = sorted(idx.loc[tr, "fecha_dato"].astype(str).unique())
    val_months = months[-args.val_months:]
    g, h, idx = g[tr].reset_index(drop=True), h[tr].reset_index(drop=True), idx[tr].reset_index(drop=True)
    is_val = idx["fecha_dato"].astype(str).isin(val_months)
    idx["is_test_time"] = is_val.astype(int)
    g.to_csv(DATA_DIR / f"{out}_guest.csv", index=False)
    h.to_csv(DATA_DIR / f"{out}_host.csv", index=False)
    idx.to_csv(DATA_DIR / f"{out}_index.csv", index=False)
    meta = json.loads((DATA_DIR / f"{args.tag}_columns.json").read_text(encoding="utf-8"))
    meta.update({"split_tag": out, "derived_from": args.tag, "val_months": val_months})
    (DATA_DIR / f"{out}_columns.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{out}] 训练 {int((~is_val).sum()):,} 行（{months[0]} ~ {months[-args.val_months - 1]}）"
          f"| 验证 {int(is_val.sum()):,} 行 / {int(idx.loc[is_val, 'label_new_any'].sum())} 正例（{val_months}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
