"""[扩展] 把其他公开数据集按「同一套纵向切分协议」落成与 prepared/ 同格式的矩阵。

目的：检验 federated_learning_new 的切分 + 评估方法在 Santander 以外是否成立。
产物格式与 `src/cross_sell/data/prepare_matrix.py` 完全一致，原版 `train_local.py` /
`train_federated_sgb.py` 不改一行即可直接读取：

    data/processed/prepared/{tag}_guest.csv   ncodpers + 内地（标签方）特征
    data/processed/prepared/{tag}_host.csv    ncodpers + 香港（协作方）特征
    data/processed/prepared/{tag}_index.csv   ncodpers / fecha_dato / label_new_any / label_n_new / is_test_time / is_test_cust
    data/processed/prepared/{tag}_columns.json

与 Santander 的口径差异（如实记录）：
- 两个新数据集都是「一行 = 一个客户」，没有月度快照 → 无法做时间外推；
  测试集 = 分层随机 20%（seed 42），仍写在 `is_test_time` 列里（原脚本只认这一列）。
  `fecha_dato` 写常量，名单层即「全部测试客户」。
- 编码规则沿用 prepare_matrix：连续量 训练集中位数填充 → 0.5%/99.5% 截尾 → 标准化；
  其余 紧凑序数编码、缺失成类、÷(类别数−1)。**一处泛化**：原版只给 renta 加缺失指示，
  这里对缺失率 ≥5% 的连续量统一加 `{col}__isna`，并跟随原列所在一侧（与 is_renta_missing 同规则）。
- 编码在**全表一次完成**，各切分只改变列归属 → 集中式在所有切分上逐位相同（与原项目自证口径一致）。

三类切分（与原方案同构）：
- business：按业务语义归属（见 SPECS）
- diagnostic：反事实——强信号全部放到标签方，协作方只留 1 列（单变量 AUC 最低的业务协作方列）
- random_seed{s}：列数与 business 相同的随机切分（Python random.Random(seed).shuffle，同原 split_fields）
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "prepared"
TEST_FRAC = 0.2
CONT_MIN_UNIQUE = 20      # 数值列唯一值 > 20 视为连续量
NA_FLAG_MIN_RATE = 0.05   # 连续量缺失率 ≥5% 加缺失指示列

SPECS = {
    # ---------------- UCI Bank Marketing（bank-additional-full，41,188 行）
    "uci": {
        "path": "raw/uci/bank-additional-full.csv",
        "read": {"sep": ";"},
        "label": lambda d: (d["y"] == "yes").astype(int),
        "drop": ["y", "duration"],   # duration = 通话时长，外呼结束才知道 → 公认标签泄漏（见 leakage_audit）
        # 内地（标签方 = 发起营销的银行）：营销触达记录 + 宏观指标（公开数据，随营销记录一起保存）
        "guest": ["contact", "month", "day_of_week", "campaign", "pdays", "previous", "poutcome",
                  "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"],
        # 香港（协作方 = 持有客户画像 / 信贷状态的机构）
        "host": ["age", "job", "marital", "education", "default", "housing", "loan"],
        "subsample": None,
    },
    # ---------------- Home Credit（application_train，307,511 行 → 抽 100,000 行与 Santander 同量级）
    "hc": {
        "path": "raw/hc/application_train.csv",
        "read": {},
        "label": lambda d: d["TARGET"].astype(int),
        "drop": ["TARGET", "SK_ID_CURR"],
        # 香港（协作方 = 外部数据方）：外部评分 + 征信查询次数 + 社交圈逾期 —— 都不是贷款机构自有数据
        "host": ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
                 "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
                 "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
                 "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
                 "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"],
        "guest": None,   # = 其余全部申请表字段（贷款机构自有）
        "subsample": 100_000,
    },
}


def _rank_encode(s: pd.Series) -> tuple[pd.Series, int, int]:
    numeric = pd.api.types.is_numeric_dtype(s)
    vals = sorted(s.dropna().unique()) if numeric else sorted({str(x) for x in s.dropna().unique()})
    code = {v: i for i, v in enumerate(vals)}
    n_na = int(s.isna().sum())
    out = s.map(code) if numeric else s.astype("object").map(lambda x: code[str(x)] if pd.notna(x) else np.nan)
    if n_na:
        out = out.fillna(len(vals))
    k = len(vals) + (1 if n_na else 0)
    out = out.astype(float)
    if k > 2:
        out = out / (k - 1)
    return out, k, n_na


def encode(df: pd.DataFrame, tr: np.ndarray) -> tuple[pd.DataFrame, dict[str, str], dict]:
    """返回 (编码后特征表, {派生列: 原列}, 处理记录)。"""
    out: dict[str, pd.Series] = {}
    parent: dict[str, str] = {}
    treat: dict[str, dict] = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > CONT_MIN_UNIQUE:
            na_rate = float(s.isna().mean())
            med = float(s[tr].median())
            v = s.fillna(med)
            lo, hi = (float(q) for q in v[tr].quantile([0.005, 0.995]))
            v = v.clip(lo, hi)
            mu, sd = float(v[tr].mean()), float(v[tr].std(ddof=0))
            out[c] = (v - mu) / sd if sd > 0 else v - mu
            parent[c] = c
            treat[c] = {"type": "continuous", "na_rate": na_rate, "median": med, "mean": mu, "std": sd}
            if na_rate >= NA_FLAG_MIN_RATE:
                out[f"{c}__isna"] = s.isna().astype(int)
                parent[f"{c}__isna"] = c
        else:
            out[c], k, n_na = _rank_encode(s)
            parent[c] = c
            treat[c] = {"type": "ordinal", "levels": k, "na": n_na}
    return pd.DataFrame(out), parent, treat


def univariate_auc(x: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if np.unique(x).size < 2:
        return 0.5
    a = roc_auc_score(y, x)
    return float(max(a, 1 - a))


def write_tag(tag, ids, X, cols_g, cols_h, y, test, meta):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = pd.concat([pd.Series(ids, name="ncodpers"), X[cols_g].reset_index(drop=True)], axis=1)
    h = pd.concat([pd.Series(ids, name="ncodpers"), X[cols_h].reset_index(drop=True)], axis=1)
    idx = pd.DataFrame({"ncodpers": ids, "fecha_dato": "2000-01-01", "label_new_any": y,
                        "label_n_new": y, "is_test_time": test.astype(int), "is_test_cust": test.astype(int)})
    g.to_csv(OUT_DIR / f"{tag}_guest.csv", index=False)
    h.to_csv(OUT_DIR / f"{tag}_host.csv", index=False)
    idx.to_csv(OUT_DIR / f"{tag}_index.csv", index=False)
    (OUT_DIR / f"{tag}_columns.json").write_text(json.dumps(
        {**meta, "split_tag": tag, "guest_features": cols_g, "host_features": cols_h,
         "n_guest_features": len(cols_g), "n_host_features": len(cols_h)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"  [{tag}] 内地 {len(cols_g)} 列 / 香港 {len(cols_h)} 列")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SPECS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random-seeds", default="42,43,44,45,46")
    ap.add_argument("--raw-root", default=str(REPO_ROOT.parent))
    args = ap.parse_args(argv)
    sp = SPECS[args.dataset]
    raw = pd.read_csv(Path(args.raw_root) / sp["path"], low_memory=False, **sp["read"])
    if sp["subsample"] and len(raw) > sp["subsample"]:
        raw = raw.sample(n=sp["subsample"], random_state=args.seed).reset_index(drop=True)
    y = sp["label"](raw).to_numpy()
    ids = np.arange(1, len(raw) + 1)
    # 分层随机测试集（无时间维度可用）
    rng = np.random.default_rng(args.seed)
    test = np.zeros(len(raw), dtype=bool)
    for cls in (0, 1):
        pos = np.where(y == cls)[0]
        test[rng.choice(pos, size=int(round(len(pos) * TEST_FRAC)), replace=False)] = True
    tr = ~test
    feats = raw.drop(columns=[c for c in sp["drop"] if c in raw.columns])
    host_raw = sp["host"]
    guest_raw = sp["guest"] or [c for c in feats.columns if c not in host_raw]
    assert not set(guest_raw) & set(host_raw) and set(guest_raw) | set(host_raw) == set(feats.columns), "切分不重不漏校验失败"
    X, parent, treat = encode(feats[guest_raw + host_raw], tr)
    side_of = {c: ("guest" if parent[c] in guest_raw else "host") for c in X.columns}
    G = [c for c in X.columns if side_of[c] == "guest"]
    H = [c for c in X.columns if side_of[c] == "host"]
    print(f"[{args.dataset}] {len(raw):,} 行 | 正例率 {y.mean():.4f} | 训练 {int(tr.sum()):,} / 测试 {int(test.sum()):,}"
          f" | 原始特征 {feats.shape[1]} → 编码后 {X.shape[1]}")
    meta = {"dataset": args.dataset, "rows": int(len(raw)), "positive_rate": float(y.mean()),
            "test_rule": f"stratified random {TEST_FRAC} seed {args.seed}", "treatment": treat}

    write_tag(f"{args.dataset}_business", ids, X, G, H, y, test, meta)

    # diagnostic：协作方只留「业务协作方原始列里单变量 AUC 最低的一列（含其派生列）」
    auc_raw = {c: max(univariate_auc(X.loc[tr, k].to_numpy(), y[tr]) for k in X.columns if parent[k] == c)
               for c in host_raw}
    keep = min(auc_raw, key=auc_raw.get)
    Hd = [c for c in X.columns if parent[c] == keep]
    Gd = [c for c in X.columns if c not in Hd]
    write_tag(f"{args.dataset}_diagnostic", ids, X, Gd, Hd, y, test, {**meta, "diagnostic_host_keep": keep})

    # random：按原始列随机分（派生列跟随原列），原始列数与 business 相同
    raw_cols = guest_raw + host_raw
    for s in [int(v) for v in args.random_seeds.split(",") if v]:
        sh = list(raw_cols)
        random.Random(s).shuffle(sh)
        gset = set(sh[: len(guest_raw)])
        Gr = [c for c in X.columns if parent[c] in gset]
        Hr = [c for c in X.columns if parent[c] not in gset]
        write_tag(f"{args.dataset}_random_seed{s}", ids, X, Gr, Hr, y, test, {**meta, "random_seed": s})
    (OUT_DIR / f"{args.dataset}_univariate_auc_host_raw.json").write_text(
        json.dumps(auc_raw, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
