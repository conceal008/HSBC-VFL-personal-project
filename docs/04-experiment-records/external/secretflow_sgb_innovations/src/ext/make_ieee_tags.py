"""[数据] IEEE-CIS（官方全量，团队 vfl_official 的 PSI 交集）→ 本流水线 tag 三件套。

与团队 vfl_official 对齐的部分：
- 同一交集、同一时间顺序 60 / 20 / 20 切分（直接读其 processed/split_ids.csv）；测试集 = 最后 20%，与团队 LR 基线同一批行。
- 同一纵向切分：A 方（内地 / 标签方）= 交易表 394 列；B 方（香港）= 身份表 40 列（DeviceType、DeviceInfo、id_01–id_38）。
与团队不同的部分（面向 SGB 的树模型编码，只用各方本地信息）：
- 数值列：训练期缺失率 ≥ 95% 或常数的删；缺失填哨兵值 −999（树模型可单独切出缺失，避免为几百个 V 列各加缺失指示列）。
- 交易时间：TransactionDT 不直接入模（时间外推下单调时间特征会泄漏切分位置），只保留一天内小时 hour。
- 类别列：A 方有标签 → 5 折目标编码（m=100）+ log 频次；B 方无标签 → log 频次 + 频次名次序数（前 30 名，其余并为一档）。
- 主 tag `ieee`：训练 = 前 80%（train + validation），测试 = 最后 20%；调参 tag `ieee_val`：训练 = 前 60%，验证 = 中间 20%。

    python src/ext/make_ieee_tags.py --src data/raw/ieee
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "prepared"
SENT = -999.0


def target_encode(fx: pd.Series, fy: np.ndarray, ax: pd.Series, prior: float, m=100.0, folds=5, seed=7):
    oof = np.empty(len(fx))
    part = np.random.default_rng(seed).integers(0, folds, len(fx))
    df = pd.DataFrame({"x": fx.to_numpy(), "y": fy})
    for k in range(folds):
        st = df[part != k].groupby("x")["y"].agg(["sum", "count"])
        enc = (st["sum"] + m * prior) / (st["count"] + m)
        oof[part == k] = df.loc[part == k, "x"].map(enc).fillna(prior).to_numpy()
    st = df.groupby("x")["y"].agg(["sum", "count"])
    enc = (st["sum"] + m * prior) / (st["count"] + m)
    return oof, ax.map(enc).fillna(prior).to_numpy()


def encode_party(frame: pd.DataFrame, fit: np.ndarray, y: np.ndarray | None):
    out, meta = {}, {}
    for c in frame.columns:
        x = frame[c]
        miss = float(x[fit].isna().mean())
        if miss >= 0.95 or x[fit].nunique(dropna=True) <= 1:
            meta[c] = {"dropped": True, "missing": round(miss, 3)}
            continue
        if pd.api.types.is_numeric_dtype(x):
            out[c] = x.astype(float).fillna(SENT).to_numpy()
            meta[c] = {"type": "num", "missing": round(miss, 3)}
            continue
        s = x.astype("string").fillna("__NA__").astype(str)
        cnt = s[fit].value_counts()
        out[f"lf_{c}"] = np.log1p(s.map(cnt).fillna(0).to_numpy())
        if y is not None:
            te = np.empty(len(s))
            te[fit], te[~fit] = target_encode(s[fit], y[fit], s[~fit], float(y[fit].mean()))
            out[f"te_{c}"] = te
        else:
            rank = {v: i for i, v in enumerate(cnt.index[:30])}
            out[f"ord_{c}"] = s.map(rank).fillna(30).to_numpy().astype(float) / 30.0
        meta[c] = {"type": "cat", "missing": round(miss, 3), "levels": int(len(cnt))}
    return pd.DataFrame(out), meta


def build(tx, idn, y, ids, fit, rows, tag, desc):
    T = tx[rows].reset_index(drop=True)
    I = idn[rows].reset_index(drop=True)
    yy, f = y[rows], fit[rows]
    A = T.drop(columns=["TransactionID", "isFraud", "TransactionDT"]).copy()
    A.insert(0, "hour", ((tx.loc[rows, "TransactionDT"].to_numpy() / 3600.0) % 24.0))
    # 高基数数值编码的卡 / 地址字段补 log 频次（标签方本地统计）
    for c in ("card1", "card2", "addr1"):
        cnt = T.loc[f, c].value_counts()
        A[f"lf_{c}"] = np.log1p(T[c].map(cnt).fillna(0).to_numpy())
    G, gm = encode_party(A, f, yy)
    H, hm = encode_party(I.drop(columns=["TransactionID"]), f, None)
    rid = ids[rows]
    G.insert(0, "ncodpers", rid)
    H.insert(0, "ncodpers", rid)
    idx = pd.DataFrame({"ncodpers": rid, "label_new_any": yy.astype(int), "is_test_time": (~f).astype(int), "is_test_cust": 0})
    G.to_csv(DATA_DIR / f"{tag}_guest.csv", index=False, float_format="%.6g")
    H.to_csv(DATA_DIR / f"{tag}_host.csv", index=False, float_format="%.6g")
    idx.to_csv(DATA_DIR / f"{tag}_index.csv", index=False)
    meta = {"split_tag": tag, "source": "IEEE-CIS 官方全量，团队 vfl_official PSI 交集", "split": desc,
            "guest_features": list(G.columns[1:]), "host_features": list(H.columns[1:]),
            "n_guest_features": G.shape[1] - 1, "n_host_features": H.shape[1] - 1,
            "train_rows": int(f.sum()), "eval_rows": int((~f).sum()),
            "train_pos_rate": float(yy[f].mean()), "eval_pos_rate": float(yy[~f].mean()),
            "guest_raw": gm, "host_raw": hm}
    (DATA_DIR / f"{tag}_columns.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{tag}] 训练 {int(f.sum()):,}（正例 {int(yy[f].sum()):,}）｜评估 {int((~f).sum()):,}（正例 {int(yy[~f].sum()):,}）"
          f"｜内地 {G.shape[1] - 1} 列 / 香港 {H.shape[1] - 1} 列")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/ieee")
    args = ap.parse_args(argv)
    src = REPO_ROOT / args.src
    tx = pd.concat([pd.read_csv(src / f"ieee_tx_part{i}.csv.gz", low_memory=False) for i in range(4)], ignore_index=True)
    idn = pd.read_csv(src / "ieee_identity.csv.gz", low_memory=False)
    sp = pd.read_csv(src / "split_ids.csv")
    assert (tx["TransactionID"].to_numpy() == sp["TransactionID"].to_numpy()).all()
    assert (idn["TransactionID"].to_numpy() == sp["TransactionID"].to_numpy()).all()
    y = sp["isFraud"].to_numpy().astype(int)
    ids = sp["TransactionID"].to_numpy()
    s = sp["split"].to_numpy()
    allr = np.ones(len(sp), bool)
    build(tx, idn, y, ids, s != "test", allr, "ieee", "训练 = 前 80%（团队 train + validation），测试 = 最后 20%（团队 test）")
    build(tx, idn, y, ids, s == "train", s != "test", "ieee_val", "训练 = 前 60%，验证 = 中间 20%（团队 validation）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
