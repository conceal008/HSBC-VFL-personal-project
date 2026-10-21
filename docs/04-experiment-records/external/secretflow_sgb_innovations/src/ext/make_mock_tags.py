"""[数据] 把团队的 HSBC mock 文件（两侧分离：guest 含 fecha_dato 与标签，host 只有 ncodpers + 25 列）
转成本流水线的 tag 三件套：{tag}_guest.csv / {tag}_host.csv / {tag}_index.csv。
切分口径与原仓库一致：fecha_dato ≥ 2016-02-01 为时间外推测试集。不改动任何特征值。

    python src/ext/make_mock_tags.py --src <解压目录> --names balanced overlap20 overlap50 overlap80
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "prepared"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--names", nargs="+", default=["balanced", "overlap20", "overlap50", "overlap80"])
    ap.add_argument("--test-start", default="2016-02-01")
    args = ap.parse_args(argv)
    src = Path(args.src)
    bi = pd.read_csv(DATA_DIR / "business_index.csv")
    bg = pd.read_csv(DATA_DIR / "business_guest.csv")
    bh = pd.read_csv(DATA_DIR / "business_host.csv")
    key_b = bi["ncodpers"].astype(str) + "|" + bi["fecha_dato"].astype(str)
    for name in args.names:
        g = pd.read_csv(src / f"{name}_guest.csv")
        h = pd.read_csv(src / f"{name}_host.csv")
        assert np.array_equal(g["ncodpers"].to_numpy(), h["ncodpers"].to_numpy()), f"{name}：两侧行序不一致"
        idx = g[["ncodpers", "fecha_dato", "label_new_any", "label_n_new"]].copy()
        idx["is_test_time"] = (pd.to_datetime(idx["fecha_dato"]) >= pd.Timestamp(args.test_start)).astype(int)
        idx["is_test_cust"] = 0
        # mock 文件是编码前的原始值；取 business 母版（已编码）中同一「客户 × 月份」的行，按档位删列
        key = idx["ncodpers"].astype(str) + "|" + idx["fecha_dato"].astype(str)
        pos = pd.Series(np.arange(len(bi)), index=key_b).reindex(key).to_numpy()
        assert not np.isnan(pos).any(), f"{name}：有行不在 business 母版中"
        pos = pos.astype(int)
        raw_feats = [c for c in g.columns if c not in ("ncodpers", "fecha_dato", "label_new_any", "label_n_new")]
        drop = [c for c in bg.columns if c not in raw_feats and c not in ("ncodpers", "is_renta_missing")]
        gf = bg.iloc[pos].drop(columns=drop).reset_index(drop=True)
        hf = bh.iloc[pos].reset_index(drop=True)
        num = [c for c in raw_feats if pd.api.types.is_numeric_dtype(g[c]) and c in gf.columns]
        same_g = all(np.allclose(g[c].to_numpy(float), gf[c].to_numpy(float), equal_nan=True) for c in num
                     if not g[c].isna().any())
        same_h = all(np.allclose(h[c].to_numpy(float), hf[c].to_numpy(float), equal_nan=True) for c in h.columns if c != "ncodpers")
        same_y = np.array_equal(idx["label_new_any"].to_numpy(), bi["label_new_any"].to_numpy()[pos])
        h = hf
        tag = f"mock_{name}"
        gf.to_csv(DATA_DIR / f"{tag}_guest.csv", index=False)
        h.to_csv(DATA_DIR / f"{tag}_host.csv", index=False)
        idx.to_csv(DATA_DIR / f"{tag}_index.csv", index=False)
        te = idx["is_test_time"] == 1
        print(f"[{tag}] {len(g):,} 行 / {g['ncodpers'].nunique():,} 客户 | 内地 {gf.shape[1] - 1} 列 香港 {h.shape[1] - 1} 列 | "
              f"训练 {int((~te).sum()):,} 测试 {int(te.sum()):,}（{int(idx.loc[te, 'label_new_any'].sum())} 正例）| "
              f"删列 {drop} | 与母版逐值一致：内地数值列 {same_g}（{len(num)} 列）香港 {same_h} 标签 {same_y}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
