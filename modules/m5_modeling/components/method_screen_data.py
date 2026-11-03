"""M5 · S5.M1 方法筛选的数据准备：把各公开数据集落成统一的两方矩阵（私有 npz，不入仓库）。

每个数据集两个视图：
- tune：fit = 训练（去掉验证），eval = 验证；只用于选配置 / 选方法。
- final：fit = 全部训练（含验证），eval = 测试；每个方法只评一次。
有标签依赖的编码（目标编码）在各视图的 fit 行上单独拟合（同原 SGB 流水线 *_val tag 的做法）。
ex_*：只有内地（标签方）特征的额外标注行（非共同客户），只进训练，不进评估。

角色切分沿用已有口径：mock / Orange / FedAds / UCI / HomeCredit 同 secretflow_sgb_innovations 的 make_*_tags 与
prepare_external；COIL2000 / Santander 交易同 M1 external_freeze（alice = 内地）；Santander G3 用 S5.G1 私有数组。
需要 pandas；只在调用时读取原始文件。
"""
from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CONT_MIN_UNIQUE = 20          # 数值列唯一值 > 20 视为连续量（prepare_external）
NA_FLAG_MIN_RATE = 0.05       # 连续量缺失率 ≥ 5% 加缺失指示列
DROP_MISSING_RATE = 0.95      # 训练期缺失率超过此值的列删除
CLIP_LOW = 0.005
CLIP_HIGH = 0.995
TE_SMOOTH = 100.0             # 目标编码平滑 m
TE_FOLDS = 5
TE_SEED = 7
HIGH_CARD = 100               # FedAds 标签方高基数列另加 log 频次
ORDINAL_MAX_LEVELS = 30       # 非标签方类别列频次名次序数（≤ 30 个取值，或前 30 名）
IEEE_SENTINEL = -999.0
SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24
FEDADS_SLICE = 2              # '["123"]' → '123'
COIL_FEATURES = 85
COIL_GUEST_FROM = 43          # 内地 = f44–f85（产品持有）

FEDADS_GUEST = (["l_i_fea_%d" % i for i in range(1, 11)] + ["l_u_fea_%d" % i for i in range(1, 7)] + ["l_c_fea"])
FEDADS_HOST = ["f_u_fea_1", "f_u_fea_2", "f_uc_fea_1", "f_uc_fea_2", "f_c"]
ORANGE_NUM_LAST = 190
ORANGE_CAT_LAST = 230
ORANGE_NUM = ["Var%d" % i for i in range(1, ORANGE_NUM_LAST + 1)]
ORANGE_CAT = ["Var%d" % i for i in range(ORANGE_NUM_LAST + 1, ORANGE_CAT_LAST + 1)]
UCI_GUEST = ["contact", "month", "day_of_week", "campaign", "pdays", "previous", "poutcome",
             "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
UCI_HOST = ["age", "job", "marital", "education", "default", "housing", "loan"]
HC_HOST = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
           "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT",
           "AMT_REQ_CREDIT_BUREAU_YEAR", "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
           "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE"]
SANTANDER_TX_VARS = 200


# ---------------------------------------------------------------- 通用编码（同 prepare_external.encode，不用标签）
def rank_encode(s, fit: np.ndarray):
    import pandas as pd
    numeric = pd.api.types.is_numeric_dtype(s)
    base = s[fit]
    vals = sorted(base.dropna().unique()) if numeric else sorted({str(x) for x in base.dropna().unique()})
    code = {v: i for i, v in enumerate(vals)}
    k = len(vals) + 1
    if numeric:
        out = s.map(code)
    else:
        out = s.astype("object").map(lambda x: code.get(str(x), np.nan) if pd.notna(x) else np.nan)
    out = out.fillna(len(vals)).astype(float)
    return out / (k - 1) if k > 2 else out


def generic_encode(df, fit: np.ndarray):
    import pandas as pd
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and s[fit].nunique(dropna=True) > CONT_MIN_UNIQUE:
            na = float(s[fit].isna().mean())
            v = s.fillna(float(s[fit].median()))
            lo, hi = (float(q) for q in v[fit].quantile([CLIP_LOW, CLIP_HIGH]))
            v = v.clip(lo, hi)
            mu, sd = float(v[fit].mean()), float(v[fit].std(ddof=0))
            out[c] = (v - mu) / sd if sd > 0 else v - mu
            if na >= NA_FLAG_MIN_RATE:
                out[c + "__isna"] = s.isna().astype(float)
        else:
            out[c] = rank_encode(s, fit)
    return pd.DataFrame(out)


def strat(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = np.zeros(len(y), bool)
    for v in np.unique(y):
        idx = np.flatnonzero(y == v)
        m[rng.choice(idx, size=int(round(len(idx) * frac)), replace=False)] = True
    return m


def target_encode(fx, fy: np.ndarray, apply: List, prior: float):
    """fit 行用 5 折折外编码，apply 各序列用全部 fit 行拟合的编码（平滑 m = TE_SMOOTH）。"""
    import pandas as pd
    oof = np.empty(len(fx))
    part = np.random.default_rng(TE_SEED).integers(0, TE_FOLDS, len(fx))
    df = pd.DataFrame({"x": fx.to_numpy(), "y": fy})
    for k in range(TE_FOLDS):
        st = df[part != k].groupby("x")["y"].agg(["sum", "count"])
        enc = (st["sum"] + TE_SMOOTH * prior) / (st["count"] + TE_SMOOTH)
        oof[part == k] = df.loc[part == k, "x"].map(enc).fillna(prior).to_numpy()
    st = df.groupby("x")["y"].agg(["sum", "count"])
    enc = (st["sum"] + TE_SMOOTH * prior) / (st["count"] + TE_SMOOTH)
    return oof, [a.map(enc).fillna(prior).to_numpy() for a in apply]


# ---------------------------------------------------------------- 视图打包
def summarize(views: Dict, meta: Dict, name: str) -> Dict:
    fin = views["final"]
    return {**meta, "name": name,
            "n_guest": int(fin["fit_xg"].shape[1]), "n_host": int(fin["fit_xh"].shape[1]),
            "final_fit_rows": int(len(fin["fit_y"])), "final_fit_pos": int(np.sum(fin["fit_y"])),
            "final_eval_rows": int(len(fin["ev_y"])), "final_eval_pos": int(np.sum(fin["ev_y"])),
            "tune_fit_rows": int(len(views["tune"]["fit_y"])), "tune_eval_rows": int(len(views["tune"]["ev_y"])),
            "tune_eval_pos": int(np.sum(views["tune"]["ev_y"])),
            "extra_rows": int(len(fin["ex_y"])) if fin.get("ex_y") is not None else 0}


def pack(views: Dict, meta: Dict, name: str, priv: Path) -> Dict:
    priv.mkdir(parents=True, exist_ok=True)
    arr: Dict[str, Any] = {"%s__%s" % (v, k): np.asarray(a) for v, d in views.items() for k, a in d.items()
                           if a is not None}
    np.savez_compressed(priv / (name + ".npz"), **arr)
    info = summarize(views, meta, name)
    (priv / (name + ".json")).write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


def view_generic(G, H, y, fit, ev, cl=None, EXG=None, exy=None, ex_fit=None):
    """G / H：原始特征（全部行）；fit / ev 行掩码；EXG：额外内地行原始特征（与 G 同列）。"""
    import pandas as pd
    rows = fit | ev
    n = int(rows.sum())
    if EXG is not None:
        allg = pd.concat([G[rows], EXG[ex_fit]], ignore_index=True)
        Ge = generic_encode(allg, np.r_[fit[rows], np.zeros(int(ex_fit.sum()), bool)])
        g_rows, g_ex = Ge.iloc[:n].reset_index(drop=True), Ge.iloc[n:].reset_index(drop=True)
    else:
        g_rows, g_ex = generic_encode(G[rows].reset_index(drop=True), fit[rows]), None
    h_rows = generic_encode(H[rows].reset_index(drop=True), fit[rows])
    f = fit[rows]
    out = {"fit_xg": g_rows[f].to_numpy(float), "fit_xh": h_rows[f].to_numpy(float), "fit_y": y[rows][f],
           "ev_xg": g_rows[~f].to_numpy(float), "ev_xh": h_rows[~f].to_numpy(float), "ev_y": y[rows][~f],
           "ev_cl": None if cl is None else cl[rows][~f]}
    if g_ex is not None:
        out["ex_xg"], out["ex_y"] = g_ex.to_numpy(float), exy[ex_fit]
    return out, list(g_rows.columns), list(h_rows.columns)


def simple_split(y: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """分层随机测试集；训练内再分层抽验证集（seed + 1）。返回 (测试掩码, 验证掩码)。"""
    seed = int(cfg["split_seed"])
    te = strat(y, float(cfg["test_frac"]), seed)
    va = np.zeros(len(y), bool)
    va[np.flatnonzero(~te)[strat(y[~te], float(cfg["val_frac"]), seed + 1)]] = True
    return te, va


def simple_views(G, H, y, cfg: Dict):
    te, va = simple_split(y, cfg)
    tune, _, _ = view_generic(G, H, y, ~te & ~va, va)
    final, gc, hc = view_generic(G, H, y, ~te, te)
    return {"tune": tune, "final": final}, gc, hc


# ---------------------------------------------------------------- 各数据集
def ds_santander_g3(raw: Path, priv: Path, cfg: Dict, g3_arrays: Path):
    d = np.load(g3_arrays)
    tr, va, te = "train", "inner_val", "dev_eval"
    tune = {"fit_xg": d[tr + "__xg"], "fit_xh": d[tr + "__xh"], "fit_y": d[tr + "__y"],
            "ev_xg": d[va + "__xg"], "ev_xh": d[va + "__xh"], "ev_y": d[va + "__y"], "ev_cl": d[va + "__cust"]}
    final = {"fit_xg": np.vstack([d[tr + "__xg"], d[va + "__xg"]]), "fit_xh": np.vstack([d[tr + "__xh"], d[va + "__xh"]]),
             "fit_y": np.r_[d[tr + "__y"], d[va + "__y"]],
             "ev_xg": d[te + "__xg"], "ev_xh": d[te + "__xh"], "ev_y": d[te + "__y"], "ev_cl": d[te + "__cust"]}
    return pack({"tune": tune, "final": final},
                {"source": "Santander 产品推荐，G3 构造两方切分（S5.G1 同一私有数组）", "split": "train / inner_val / dev_eval",
                 "eval_label": "开发评估集（dev_eval，S5.G1 起已打开）", "cluster": "客户"}, "santander_g3", priv)


def ds_mock(name: str, raw: Path, priv: Path, cfg: Dict):
    import pandas as pd
    src = raw / "数据集" / "HSBC_mock数据"
    g = pd.read_csv(src / (name + "_guest.csv"))
    h = pd.read_csv(src / (name + "_host.csv"))
    assert np.array_equal(g["ncodpers"].to_numpy(), h["ncodpers"].to_numpy())
    y = g["label_new_any"].to_numpy().astype(int)
    month = g["fecha_dato"].astype(str).to_numpy()
    test = (pd.to_datetime(g["fecha_dato"]) >= pd.Timestamp(cfg["mock_test_start"])).to_numpy()
    nv = int(cfg["mock_val_months"])
    months = sorted(set(month[~test]))
    val = np.isin(month, months[-nv:]) & ~test
    feats = [c for c in g.columns if c not in ("ncodpers", "fecha_dato", "label_new_any", "label_n_new")]
    G, H, cl = g[feats], h.drop(columns="ncodpers"), g["ncodpers"].to_numpy()
    EXG = exy = exm = None
    if name != "balanced":
        b = pd.read_csv(src / "balanced_guest.csv")
        b = b[~b["ncodpers"].isin(set(g["ncodpers"]))].reset_index(drop=True)
        EXG, exy = b.reindex(columns=feats), b["label_new_any"].to_numpy().astype(int)
        exm = b["fecha_dato"].astype(str).to_numpy()
    views: Dict = {}
    gc: List[str] = []
    hc: List[str] = []
    for v, (fit, ev, exmask) in {
        "tune": (~test & ~val, val, None if EXG is None else np.isin(exm, months[:-nv])),
        "final": (~test, test, None if EXG is None else np.isin(exm, months)),
    }.items():
        views[v], gc, hc = view_generic(G, H, y, fit, ev, cl, EXG, exy, exmask)
    return pack(views, {"source": "HSBC mock %s（Santander 派生）" % name,
                        "split": "时间外推：测试 ≥ %s；验证 = 训练窗口末 %d 个月" % (cfg["mock_test_start"], nv),
                        "cluster": "客户", "extra": None if EXG is None else "balanced 中不在本档的内地客户行（缺 tiprel_1mes）",
                        "guest_cols": gc, "host_cols": hc}, "mock_" + name, priv)


def fedads_load(path: Path):
    import pandas as pd
    df = pd.read_csv(path, dtype=str)
    for c in df.columns[1:]:
        df[c] = df[c].str.slice(FEDADS_SLICE, -FEDADS_SLICE)
    df["label"] = df["label"].astype(int)
    return df


def fedads_view(fit, ev, ex=None):
    prior = fit["label"].mean()
    nx = 0 if ex is None else len(ex)
    gf, ge, gx = {}, {}, {}
    for c in FEDADS_GUEST:
        if fit[c].nunique() <= 1:
            continue
        has_x = ex is not None and c in ex.columns
        oof, app = target_encode(fit[c], fit["label"].to_numpy(), [ev[c]] + ([ex[c]] if has_x else []), prior)
        gf["te_" + c], ge["te_" + c] = oof, app[0]
        gx["te_" + c] = app[1] if has_x else np.full(nx, np.nan)
        if fit[c].nunique() > HIGH_CARD:
            cnt = fit[c].value_counts()
            gf["lf_" + c] = np.log1p(fit[c].map(cnt).to_numpy())
            ge["lf_" + c] = np.log1p(ev[c].map(cnt).fillna(0).to_numpy())
            gx["lf_" + c] = np.log1p(ex[c].map(cnt).fillna(0).to_numpy()) if has_x else np.full(nx, np.nan)
    hf, he = {}, {}
    for c in FEDADS_HOST:
        for v in sorted(fit[c].unique(), key=lambda x: -int((fit[c] == x).sum())):
            nm = "%s_v%d" % (c, len([k for k in hf if k.startswith(c + "_v")]))
            hf[nm] = (fit[c] == v).astype(float).to_numpy()
            he[nm] = (ev[c] == v).astype(float).to_numpy()
    out = {"fit_xg": np.column_stack(list(gf.values())), "fit_xh": np.column_stack(list(hf.values())),
           "fit_y": fit["label"].to_numpy(), "ev_xg": np.column_stack(list(ge.values())),
           "ev_xh": np.column_stack(list(he.values())), "ev_y": ev["label"].to_numpy(), "ev_cl": None}
    if ex is not None:
        out["ex_xg"], out["ex_y"] = np.column_stack(list(gx.values())), ex["label"].to_numpy()
    return out, list(gf), list(hf)


def ds_fedads(raw: Path, priv: Path, cfg: Dict):
    src = raw / "federated_learning_new_复现与优化" / "fedads_sample"
    tr, te = fedads_load(src / "train_s300k.csv"), fedads_load(src / "test_s200k.csv")
    ex = fedads_load(src / cfg["fedads_extra_file"])
    va = np.random.default_rng(int(cfg["fedads_val_seed"])).random(len(tr)) < float(cfg["val_frac"])
    exva = np.random.default_rng(int(cfg["fedads_extra_val_seed"])).random(len(ex)) < float(cfg["val_frac"])
    tune, _, _ = fedads_view(tr[~va].reset_index(drop=True), tr[va].reset_index(drop=True),
                             ex[~exva].reset_index(drop=True))
    final, gc, hc = fedads_view(tr, te, ex)
    return pack({"tune": tune, "final": final},
                {"source": "FedAds（阿里妈妈）均匀抽样 30 万训练 / 20 万官方测试（同 INNOVATIONS）",
                 "split": "官方测试；验证 = 训练随机 20%", "cluster": "行",
                 "extra": "官方非对齐训练集伯努利抽样 3%（只有标签方 l_i / l_u 特征，缺 l_c_fea）",
                 "guest_cols": gc, "host_cols": hc}, "fedads", priv)


def orange_view(raw, y: np.ndarray, fit: np.ndarray, ev: np.ndarray, guest_vars: set):
    import pandas as pd
    rows = fit | ev
    r, yy, f = raw[rows].reset_index(drop=True), y[rows], fit[rows]
    g: Dict[str, np.ndarray] = {}
    h: Dict[str, np.ndarray] = {}
    prior = float(yy[f].mean())
    for c in ORANGE_NUM:
        x = r[c].astype(float)
        miss = float(x[f].isna().mean())
        if miss > DROP_MISSING_RATE or x[f].nunique() <= 1:
            continue
        dst = g if c in guest_vars else h
        dst[c] = x.fillna(float(x[f].median())).to_numpy()
        if miss >= NA_FLAG_MIN_RATE:
            dst[c + "__isna"] = x.isna().astype(float).to_numpy()
    for c in ORANGE_CAT:
        x = r[c].fillna("__NA__").astype(str)
        miss = float((x[f] == "__NA__").mean())
        if miss > DROP_MISSING_RATE or x[f].nunique() <= 1:
            continue
        cnt = x[f].value_counts()
        lf = np.log1p(x.map(cnt).fillna(0).to_numpy())
        if c in guest_vars:
            oof, (app,) = target_encode(x[f], yy[f], [x[~f]], prior)
            te = np.empty(len(x))
            te[f], te[~f] = oof, app
            g["te_" + c], g["lf_" + c] = te, lf
        else:
            h["lf_" + c] = lf
            if len(cnt) <= ORDINAL_MAX_LEVELS:
                rank = {v: i for i, v in enumerate(cnt.index)}
                h["ord_" + c] = x.map(rank).fillna(len(cnt)).to_numpy().astype(float) / max(len(cnt) - 1, 1)
    G, H = pd.DataFrame(g), pd.DataFrame(h)
    return ({"fit_xg": G[f].to_numpy(float), "fit_xh": H[f].to_numpy(float), "fit_y": yy[f],
             "ev_xg": G[~f].to_numpy(float), "ev_xh": H[~f].to_numpy(float), "ev_y": yy[~f], "ev_cl": None},
            list(G), list(H))


def ds_orange(label: str, raw: Path, priv: Path, cfg: Dict):
    import pandas as pd
    src = raw / "数据集" / "Orange_KDDCup2009"
    with zipfile.ZipFile(src / "orange_small_train.data.zip") as z:
        nm = [n for n in z.namelist() if n.endswith("orange_small_train.data")][0]
        with z.open(nm) as fh:
            data = pd.read_csv(fh, sep="\t", low_memory=False)
    seed = int(cfg["split_seed"])
    order = ORANGE_NUM + ORANGE_CAT
    random.Random(seed).shuffle(order)
    guest_vars = set(order[: len(order) // 2])
    y = (pd.read_csv(src / ("orange_small_train_%s.labels" % label), header=None)[0].to_numpy() == 1).astype(int)
    te, va = simple_split(y, cfg)
    tune, _, _ = orange_view(data, y, ~te & ~va, va, guest_vars)
    final, gc, hc = orange_view(data, y, ~te, te, guest_vars)
    short = {"upselling": "upsell"}.get(label, label)
    return pack({"tune": tune, "final": final},
                {"source": "Orange KDD Cup 2009 small · " + label,
                 "split": "原始变量随机对半（random.Random(%d)）；测试分层 20%%；验证 = 训练内分层 20%%" % seed,
                 "cluster": "行", "guest_cols": gc, "host_cols": hc}, "orange_" + short, priv)


def ds_uci(raw: Path, priv: Path, cfg: Dict):
    import pandas as pd
    with zipfile.ZipFile(raw / "数据集" / "UCI_BankMarketing" / "解压" / "bank-additional.zip") as z:
        nm = [n for n in z.namelist() if n.endswith("bank-additional-full.csv")][0]
        with z.open(nm) as fh:
            data = pd.read_csv(fh, sep=";")
    y = (data["y"] == "yes").astype(int).to_numpy()
    views, gc, hc = simple_views(data[UCI_GUEST], data[UCI_HOST], y, cfg)
    return pack(views, {"source": "UCI Bank Marketing（bank-additional-full，去 duration）",
                        "roles": "prepare_external · uci business", "split": "测试分层 20%；验证 = 训练内分层 20%",
                        "cluster": "行", "guest_cols": gc, "host_cols": hc}, "uci_bank", priv)


def ds_homecredit(raw: Path, priv: Path, cfg: Dict):
    import pandas as pd
    data = pd.read_csv(raw / "数据集" / "HomeCredit_DefaultRisk" / "解压" / "application_train.csv", low_memory=False)
    data = data.sample(n=int(cfg["homecredit_rows"]), random_state=int(cfg["split_seed"])).reset_index(drop=True)
    y = data["TARGET"].astype(int).to_numpy()
    feats = data.drop(columns=["TARGET", "SK_ID_CURR"])
    guest = [c for c in feats.columns if c not in HC_HOST]
    views, gc, hc = simple_views(feats[guest], feats[HC_HOST], y, cfg)
    return pack(views, {"source": "Home Credit application_train 抽 %d 行" % int(cfg["homecredit_rows"]),
                        "roles": "prepare_external · hc business（香港 = 外部评分 / 征信查询 / 社交圈）",
                        "split": "测试分层 20%；验证 = 训练内分层 20%", "cluster": "行",
                        "guest_cols": gc, "host_cols": hc}, "homecredit", priv)


def ds_coil(raw: Path, priv: Path, cfg: Dict):
    import pandas as pd
    src = raw / "数据集" / "COIL2000_保险营销" / "解压"
    cols = ["f%d" % i for i in range(1, COIL_FEATURES + 1)]
    tr = pd.read_csv(src / "ticdata2000.txt", sep=r"\s+", header=None, names=cols + ["label"])
    ev = pd.read_csv(src / "ticeval2000.txt", sep=r"\s+", header=None, names=cols)
    ev["label"] = pd.read_csv(src / "tictgts2000.txt", header=None)[0].to_numpy()
    data = pd.concat([tr, ev], ignore_index=True)
    y = data["label"].astype(int).to_numpy()
    te = np.r_[np.zeros(len(tr), bool), np.ones(len(ev), bool)]
    va = np.zeros(len(y), bool)
    va[np.flatnonzero(~te)[strat(y[~te], float(cfg["val_frac"]), int(cfg["split_seed"]) + 1)]] = True
    guest, host = cols[COIL_GUEST_FROM:], cols[:COIL_GUEST_FROM]
    G = data[guest].astype("category").apply(lambda s: s.cat.codes).astype(float)
    H = data[host].astype(float)
    tune, _, _ = view_generic(G, H, y, ~te & ~va, va)
    final, gc, hc = view_generic(G, H, y, ~te, te)
    return pack({"tune": tune, "final": final},
                {"source": "COIL2000 保险营销（官方训练 5,822 / 官方评估 4,000）", "split": "官方评估集作测试；验证 = 训练内分层 20%",
                 "roles": "M1 external_freeze：内地 f44–f85（产品），香港 f1–f43（社会人口）", "cluster": "行",
                 "guest_cols": gc, "host_cols": hc}, "coil2000", priv)


def ds_santander_tx(raw: Path, priv: Path, cfg: Dict):
    import pandas as pd
    data = pd.read_csv(raw / "数据集" / "Santander_CustomerTransaction" / "解压" / "train.csv")
    data = data.sample(frac=float(cfg["santander_tx_frac"]), random_state=int(cfg["split_seed"])).reset_index(drop=True)
    y = data["target"].astype(int).to_numpy()
    v = ["var_%d" % i for i in range(SANTANDER_TX_VARS)]
    views, gc, hc = simple_views(data[v[0::2]], data[v[1::2]], y, cfg)
    return pack(views, {"source": "Santander Customer Transaction 抽 %s" % cfg["santander_tx_frac"],
                        "roles": "M1 external_freeze：内地偶数列、香港奇数列", "split": "测试分层 20%；验证 = 训练内分层 20%",
                        "cluster": "行", "guest_cols": gc, "host_cols": hc}, "santander_tx", priv)


def ieee_encode(frame, fit: np.ndarray):
    import pandas as pd
    out = {}
    for c in frame.columns:
        s = frame[c]
        if s[fit].isna().mean() >= DROP_MISSING_RATE or s[fit].nunique() <= 1:
            continue
        if pd.api.types.is_numeric_dtype(s):
            out[c] = s.astype(float).fillna(IEEE_SENTINEL)
        else:
            x = s.fillna("__NA__").astype(str)
            cnt = x[fit].value_counts()
            out["lf_" + c] = np.log1p(x.map(cnt).fillna(0).astype(float))
            top = {v: i for i, v in enumerate(cnt.index[:ORDINAL_MAX_LEVELS])}
            out["ord_" + c] = x.map(top).fillna(len(top)).astype(float)
    return pd.DataFrame(out)


def ds_ieee(raw: Path, priv: Path, cfg: Dict):
    import pandas as pd
    src = raw / "federated_learning_new_复现与优化" / "ieee_sample"
    tx = pd.read_csv(src / "ieee_tx_part0.csv.gz", low_memory=False)
    idn = pd.read_csv(src / "ieee_identity.csv.gz", low_memory=False)
    df = tx.merge(idn, on="TransactionID", how="inner").sort_values("TransactionDT").reset_index(drop=True)
    y = df["isFraud"].astype(int).to_numpy()
    n = len(df)
    te = np.arange(n) >= int(n * float(cfg["ieee_test_start"]))
    va = (np.arange(n) >= int(n * float(cfg["ieee_val_start"]))) & ~te
    df["hour"] = (df["TransactionDT"] // SECONDS_PER_HOUR) % HOURS_PER_DAY
    host = [c for c in idn.columns if c != "TransactionID"]
    guest = [c for c in tx.columns if c not in ("TransactionID", "isFraud", "TransactionDT")] + ["hour"]
    views: Dict = {}
    gc: List[str] = []
    hc: List[str] = []
    for v, (fit, ev) in {"tune": (~te & ~va, va), "final": (~te, te)}.items():
        rows = fit | ev
        f = fit[rows]
        G = ieee_encode(df.loc[rows, guest].reset_index(drop=True), f)
        H = ieee_encode(df.loc[rows, host].reset_index(drop=True), f)
        views[v] = {"fit_xg": G[f].to_numpy(float), "fit_xh": H[f].to_numpy(float), "fit_y": y[rows][f],
                    "ev_xg": G[~f].to_numpy(float), "ev_xh": H[~f].to_numpy(float), "ev_y": y[rows][~f], "ev_cl": None}
        gc, hc = list(G), list(H)
    return pack(views, {"source": "IEEE-CIS 团队交集的第 1 块（ieee_tx_part0，全部有身份表）",
                        "split": "按 TransactionDT 时间顺序 60 / 20 / 20（与 INNOVATIONS 的全交集口径不同）",
                        "roles": "内地 = 交易表，香港 = 身份表", "cluster": "行", "guest_cols": gc, "host_cols": hc},
                "ieee_part0", priv)


def build(name: str, raw: Path, priv: Path, cfg: Dict, g3_arrays: Optional[Path] = None) -> Dict:
    """按名字准备一个数据集；返回元信息（行数、正例数、列数），矩阵写入 priv。"""
    table = {
        "mock_balanced": lambda: ds_mock("balanced", raw, priv, cfg),
        "mock_overlap80": lambda: ds_mock("overlap80", raw, priv, cfg),
        "mock_overlap50": lambda: ds_mock("overlap50", raw, priv, cfg),
        "mock_overlap20": lambda: ds_mock("overlap20", raw, priv, cfg),
        "fedads": lambda: ds_fedads(raw, priv, cfg),
        "orange_churn": lambda: ds_orange("churn", raw, priv, cfg),
        "orange_upsell": lambda: ds_orange("upselling", raw, priv, cfg),
        "orange_appetency": lambda: ds_orange("appetency", raw, priv, cfg),
        "uci_bank": lambda: ds_uci(raw, priv, cfg), "homecredit": lambda: ds_homecredit(raw, priv, cfg),
        "coil2000": lambda: ds_coil(raw, priv, cfg), "santander_tx": lambda: ds_santander_tx(raw, priv, cfg),
        "ieee_part0": lambda: ds_ieee(raw, priv, cfg),
    }
    if name == "santander_g3":
        assert g3_arrays is not None, "santander_g3 需要 S5.G1 私有数组路径"
        return ds_santander_g3(raw, priv, cfg, g3_arrays)
    return table[name]()
