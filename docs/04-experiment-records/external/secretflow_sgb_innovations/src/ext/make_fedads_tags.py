"""[数据] FedAds（阿里妈妈原生纵向联邦广告转化数据）→ 本流水线 tag 三件套。

输入：本机按固定种子均匀抽样的原始行（train_s300k.csv / test_s200k.csv，值为 JSON 单元素数组的哈希 ID）。
特征归属沿用数据集官方划分：l_* 归标签方（内地 / guest），f_* 归非标签方（香港 / host）。
编码只用各方本地可得的信息：
- 标签方（有标签）：每个特征做 5 折交叉的目标编码（平滑 m=100，防止训练行看到自己的标签），
  高基数列（训练中取值 > 100 个）另加 log 频次编码；常数列删除。评估行用全部训练行拟合的编码。
- 非标签方（无标签）：3–5 个取值的类别做 one-hot。
产出：
- fedads：训练 = 抽样训练集，测试 = 抽样测试集（官方测试集）
- fedads_val：只在训练样本内随机切 80/20，供调参（编码只在 80% 上拟合）

    python src/ext/make_fedads_tags.py --src <fedads_sample 目录>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "prepared"
GUEST = [f"l_i_fea_{i}" for i in range(1, 11)] + [f"l_u_fea_{i}" for i in range(1, 7)] + ["l_c_fea"]
HOST = ["f_u_fea_1", "f_u_fea_2", "f_uc_fea_1", "f_uc_fea_2", "f_c"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    for c in df.columns[1:]:
        df[c] = df[c].str.slice(2, -2)            # '["123"]' → '123'（已确认全部为单元素）
    df["label"] = df["label"].astype(int)
    return df


def target_encode(fit: pd.DataFrame, apply: pd.DataFrame | None, col: str, prior: float, m: float = 100.0,
                  folds: int = 5, seed: int = 7):
    """返回 (fit 行的 OOF 编码, apply 行的编码)。"""
    oof = np.empty(len(fit))
    part = np.random.default_rng(seed).integers(0, folds, len(fit))
    for k in range(folds):
        tr = fit[part != k]
        st = tr.groupby(col)["label"].agg(["sum", "count"])
        enc = (st["sum"] + m * prior) / (st["count"] + m)
        oof[part == k] = fit.loc[part == k, col].map(enc).fillna(prior).to_numpy()
    st = fit.groupby(col)["label"].agg(["sum", "count"])
    enc = (st["sum"] + m * prior) / (st["count"] + m)
    app = apply[col].map(enc).fillna(prior).to_numpy() if apply is not None else None
    return oof, app


def build(fit: pd.DataFrame, ev: pd.DataFrame, tag: str) -> None:
    prior = fit["label"].mean()
    g_fit, g_ev = {}, {}
    for c in GUEST:
        if fit[c].nunique() <= 1:
            continue
        g_fit[f"te_{c}"], g_ev[f"te_{c}"] = target_encode(fit, ev, c, prior)
        if fit[c].nunique() > 100:
            cnt = fit[c].value_counts()
            g_fit[f"lf_{c}"] = np.log1p(fit[c].map(cnt).to_numpy())
            g_ev[f"lf_{c}"] = np.log1p(ev[c].map(cnt).fillna(0).to_numpy())
    h_fit, h_ev = {}, {}
    for c in HOST:
        for v in sorted(fit[c].unique(), key=lambda x: -int((fit[c] == x).sum())):
            name = f"{c}_v{len([k for k in h_fit if k.startswith(c + '_v')])}"
            h_fit[name] = (fit[c] == v).astype(float).to_numpy()
            h_ev[name] = (ev[c] == v).astype(float).to_numpy()
    ids = np.concatenate([fit["sample_id"].to_numpy(), ev["sample_id"].to_numpy()])
    guest = pd.DataFrame({"ncodpers": ids, **{k: np.concatenate([g_fit[k], g_ev[k]]) for k in g_fit}})
    host = pd.DataFrame({"ncodpers": ids, **{k: np.concatenate([h_fit[k], h_ev[k]]) for k in h_fit}})
    idx = pd.DataFrame({"ncodpers": ids, "label_new_any": np.concatenate([fit["label"], ev["label"]]),
                        "is_test_time": np.r_[np.zeros(len(fit), int), np.ones(len(ev), int)], "is_test_cust": 0})
    guest.to_csv(DATA_DIR / f"{tag}_guest.csv", index=False)
    host.to_csv(DATA_DIR / f"{tag}_host.csv", index=False)
    idx.to_csv(DATA_DIR / f"{tag}_index.csv", index=False)
    meta = {"split_tag": tag, "source": "FedAds (Tianchi 148347) 均匀抽样", "guest_features": list(g_fit),
            "host_features": list(h_fit), "train_rows": len(fit), "eval_rows": len(ev),
            "train_pos_rate": float(fit["label"].mean()), "eval_pos_rate": float(ev["label"].mean())}
    (DATA_DIR / f"{tag}_columns.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{tag}] 训练 {len(fit):,}（正例 {int(fit['label'].sum()):,}）｜评估 {len(ev):,}（正例 {int(ev['label'].sum()):,}）"
          f"｜内地 {len(g_fit)} 列 / 香港 {len(h_fit)} 列")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    args = ap.parse_args(argv)
    tr = load(Path(args.src) / "train_s300k.csv")
    te = load(Path(args.src) / "test_s200k.csv")
    build(tr, te, "fedads")
    rng = np.random.default_rng(20261022)
    va = rng.random(len(tr)) < 0.2
    build(tr[~va].reset_index(drop=True), tr[va].reset_index(drop=True), "fedads_val")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
