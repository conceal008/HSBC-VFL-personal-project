"""M5 · S5.M5：Orange KDD Cup 2009 大版（15,000 列）的两方矩阵与 M3 两方特征预筛（DR-M5-014 · D-2 / D-3）。

- 读取 5 个训练分块（50,000 行）。测试集、验证集与两方划分同小版：method_screen_data.simple_split，
  原始变量用 random.Random(split_seed) 打乱后对半。
- 编码方式同小版 orange_view：数值列中位数填补、缺失率 ≥ 5% 加缺失指示；内地类别列折外目标编码 + log 频次，
  香港类别列 log 频次 + 频次名次序数（≤ 30 个取值）。
- 删列（缺失率 > 95% 或只有一个取值）、是否加缺失指示、是否加序数，都只在 tune 视图的 fit 行（训练去掉验证）上判定，
  列清单两个视图共用；填补值、频次与目标编码在各视图自己的 fit 行上拟合（同小版）。
- M3 预筛只在 tune 视图的 fit 行上做一次：
  - 内地（主动方）：只用内地特征拟合原版参数 XGBoost，按总增益排序；
  - 香港（被动方）：用「内地前 k 列 + 香港全部列」拟合同样的 XGBoost，按香港各列被选为分裂的次数排序。
    SGB 中被动方保存自己特征上的分裂，本来就知道自己哪些列被选中。
  增益或次数为 0 的列不入选；并列按列序。
- 主分析矩阵（每方前 k 列）按 method_screen_data.pack 的格式存私有目录，可直接交给 method_screen.Screen；
  敏感性矩阵（final 视图全部清洗后的列与两方排序）另存，在主分析选定的配置下评估不同 k。

只在调用时导入 pandas / xgboost。矩阵存 float32。
"""
from __future__ import annotations

import json
import random
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules.m5_modeling.components import method_screen as MS
from modules.m5_modeling.components import method_screen_cv as CV
from modules.m5_modeling.components import method_screen_data as MD

LARGE_NUM_LAST = 14740
LARGE_CAT_LAST = 15000
LARGE_NUM = ["Var%d" % i for i in range(1, LARGE_NUM_LAST + 1)]
LARGE_CAT = ["Var%d" % i for i in range(LARGE_NUM_LAST + 1, LARGE_CAT_LAST + 1)]
CHUNKS = 5
ORANGE_ROWS = 50000                 # 大版训练集行数（KDD Cup 2009 官方说明）
STAT_BLOCK = 512                    # 按列分块统计，避免一次复制整张训练矩阵
ROW_BLOCK = 4096                    # 敏感性组矩阵时的行块
NA_TOKEN = "__NA__"
LABELS = {"churn": "churn", "appetency": "appetency", "upselling": "upsell"}
ALL = "all"                         # 敏感性中「全部清洗后列」的记号


def name_of(label: str) -> str:
    return "orange_large_" + LABELS[label]


# ---------------------------------------------------------------- 读取
def _chunks(src: Path, chunks: int, prefix: str, names: List[str], usecols: Sequence[str], dtype: Dict):
    import pandas as pd
    for i in range(1, chunks + 1):
        stem = "%s.chunk%d" % (prefix, i)
        with zipfile.ZipFile(src / (stem + ".zip")) as z:
            member = [n for n in z.namelist() if n.endswith(stem)][0]
            with z.open(member) as fh:
                yield pd.read_csv(fh, sep="\t", header=0 if i == 1 else None, names=names, usecols=list(usecols),
                                  dtype=dtype, index_col=False)


def scan_numeric(src: Path, chunks: int = CHUNKS, num_vars: Sequence[str] = LARGE_NUM,
                 cat_vars: Sequence[str] = LARGE_CAT, prefix: str = "orange_large_train.data") -> List[str]:
    """第一遍：全部行上至少有一个非缺失值、且不是常数的数值变量（不看标签与切分）。
    被去掉的变量在任何行子集上也是全缺失或常数，按 D-2 的删列规则在 tune 视图 fit 行上同样会被删，结果不变；只为降低内存。"""
    names = list(num_vars) + list(cat_vars)
    cnt = np.zeros(len(num_vars), np.int64)
    lo = np.full(len(num_vars), np.inf)
    hi = np.full(len(num_vars), -np.inf)
    for df in _chunks(src, chunks, prefix, names, num_vars, {c: np.float32 for c in num_vars}):
        a = df[list(num_vars)].to_numpy(np.float32)
        cnt += (~np.isnan(a)).sum(axis=0)
        with warnings.catch_warnings():                # 全缺失列的 nanmin 告警
            warnings.simplefilter("ignore", RuntimeWarning)
            lo = np.fmin(lo, np.nanmin(a, axis=0))
            hi = np.fmax(hi, np.nanmax(a, axis=0))
        del df, a
    return [v for v, c, a, b in zip(num_vars, cnt, lo, hi) if c > 0 and b > a]


def read_raw(src: Path, chunks: int = CHUNKS, num_vars: Sequence[str] = LARGE_NUM,
             cat_vars: Sequence[str] = LARGE_CAT, prefix: str = "orange_large_train.data", max_rows: int = ORANGE_ROWS,
             all_num: Optional[Sequence[str]] = None, num_path: Optional[Path] = None):
    """读取各分块：返回 (数值矩阵 float32 [行, num_vars], 类别 DataFrame（category 类型）)。第 1 块含表头。
    all_num 为文件中的全部数值变量（默认 = num_vars）；num_vars 可以是其子集（scan_numeric 的结果）。
    数值矩阵预先按 max_rows 分配、逐块填入，避免拼接时内存翻倍；返回实际行数的切片。
    num_path 给定时数值矩阵写成按列存放的 .npy 并以只读映射返回（行数须恰为 max_rows）。"""
    import pandas as pd
    names = list(all_num if all_num is not None else num_vars) + list(cat_vars)
    dtype = {**{c: np.float32 for c in num_vars}, **{c: "category" for c in cat_vars}}
    mm = None
    if num_path is None:
        num = np.empty((max_rows, len(num_vars)), np.float32)
    else:                                   # 按列存放的磁盘映射：按列读取连续，且不占匿名内存
        mm = np.lib.format.open_memmap(num_path, mode="w+", dtype=np.float32, shape=(max_rows, len(num_vars)),
                                       fortran_order=True)
        num = mm
    parts_cat, start = [], 0
    for df in _chunks(src, chunks, prefix, names, list(num_vars) + list(cat_vars), dtype):
        end = start + len(df)
        assert end <= max_rows, "行数超过预分配的 %d" % max_rows
        num[start:end] = df[list(num_vars)].to_numpy(np.float32)
        parts_cat.append(df[list(cat_vars)].astype(object).reset_index(drop=True))
        start = end
        del df
    cat = pd.concat(parts_cat, ignore_index=True).astype("category")
    del parts_cat
    if mm is not None:
        assert start == max_rows, "磁盘映射要求行数恰为 %d" % max_rows
        mm.flush()
        del num, mm
        release_memory()
        return np.load(num_path, mmap_mode="r"), cat
    return num[:start], cat


def release_memory() -> None:
    """把已释放的堆内存交还操作系统（glibc malloc_trim；其他平台无操作）。"""
    import ctypes
    import gc
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def party_vars(seed: int, num_vars: Sequence[str] = LARGE_NUM, cat_vars: Sequence[str] = LARGE_CAT) -> set:
    order = list(num_vars) + list(cat_vars)
    random.Random(seed).shuffle(order)
    return set(order[: len(order) // 2])


# ---------------------------------------------------------------- 列清单（只看 tune 视图 fit 行）
def column_plan(num: np.ndarray, cat, fit: np.ndarray, guest: set,
                num_vars: Sequence[str] = LARGE_NUM) -> List[Dict]:
    """返回编码列清单：每项 {var, kind: num / cat, side: g / h, flag, ord}，顺序 = 数值列在前、类别列在后、各按原列序。"""
    plan: List[Dict] = []
    for j0 in range(0, num.shape[1], STAT_BLOCK):
        block = num[:, j0:j0 + STAT_BLOCK][fit]
        miss = np.isnan(block).mean(axis=0)
        with warnings.catch_warnings():                # 全缺失列的 nanmin 告警（该列随后被删）
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanmin(block, axis=0), np.nanmax(block, axis=0)
        for jj in range(block.shape[1]):
            if miss[jj] > MD.DROP_MISSING_RATE or not (hi[jj] > lo[jj]):
                continue
            v = num_vars[j0 + jj]
            plan.append({"var": v, "col": j0 + jj, "kind": "num", "side": "g" if v in guest else "h",
                         "flag": bool(miss[jj] >= MD.NA_FLAG_MIN_RATE), "ord": False})
    for c in cat.columns:
        x = cat[c][fit].astype(object).fillna(NA_TOKEN)
        miss = float((x == NA_TOKEN).mean())
        levels = int(x.nunique())
        if miss > MD.DROP_MISSING_RATE or levels <= 1:
            continue
        side = "g" if c in guest else "h"
        plan.append({"var": c, "col": None, "kind": "cat", "side": side, "flag": False,
                     "ord": side == "h" and levels <= MD.ORDINAL_MAX_LEVELS})
    return plan


def encoded_names(plan: List[Dict], side: str) -> List[str]:
    out = []
    for p in plan:
        if p["side"] != side:
            continue
        if p["kind"] == "num":
            out += [p["var"]] + ([p["var"] + "__isna"] if p["flag"] else [])
        elif side == "g":
            out += ["te_" + p["var"], "lf_" + p["var"]]
        else:
            out += ["lf_" + p["var"]] + (["ord_" + p["var"]] if p["ord"] else [])
    return out


# ---------------------------------------------------------------- 编码一个视图
def encode(num: np.ndarray, cat, y: np.ndarray, plan: List[Dict], fit: np.ndarray, ev: np.ndarray, side: str,
           keep: Optional[Sequence[str]] = None, lead: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
    """对 fit | ev 行（原顺序）编码一方的列；keep 给定时只算其中的列。返回 (float32 矩阵, 列名)。
    lead 给定时放在矩阵最前面（列名不含 lead）；矩阵预先分配、逐列填入，避免拼接时内存翻倍。"""
    rows = fit | ev
    f = fit[rows]
    yy = y[rows]
    prior = float(yy[f].mean())
    names = [n for n in encoded_names(plan, side) if keep is None or n in set(keep)]
    off = 0 if lead is None else lead.shape[1]
    out = np.empty((int(rows.sum()), off + len(names)), np.float32)
    if lead is not None:
        out[:, :off] = lead
    pos = {n: off + i for i, n in enumerate(names)}
    for p in plan:
        if p["side"] != side:
            continue
        v = p["var"]
        if p["kind"] == "num":
            x = num[rows, p["col"]]
            nan = np.isnan(x)
            if v in pos:
                med = np.float32(np.nanmedian(x[f])) if (~nan[f]).any() else np.float32(0)
                out[:, pos[v]] = np.where(nan, med, x)
            if v + "__isna" in pos:
                out[:, pos[v + "__isna"]] = nan
            continue
        mine = [n for n in (["te_" + v, "lf_" + v, "ord_" + v]) if n in pos]
        if not mine:
            continue
        x = cat[v][rows].astype(object).fillna(NA_TOKEN).astype(str).reset_index(drop=True)
        cnt = x[f].value_counts()
        if "lf_" + v in pos:
            out[:, pos["lf_" + v]] = np.log1p(x.map(cnt).fillna(0).to_numpy(float))
        if "te_" + v in pos:
            oof, (app,) = MD.target_encode(x[f], yy[f], [x[~f]], prior)
            col = out[:, pos["te_" + v]]
            col[f], col[~f] = oof, app
            out[:, pos["te_" + v]] = col
        if "ord_" + v in pos:
            rank = {lv: i for i, lv in enumerate(cnt.index)}
            out[:, pos["ord_" + v]] = x.map(rank).fillna(len(cnt)).to_numpy(float) / max(len(cnt) - 1, 1)
    return out, names


def view_of(G: np.ndarray, H: np.ndarray, y: np.ndarray, fit: np.ndarray, ev: np.ndarray) -> Dict:
    rows = fit | ev
    f = fit[rows]
    return {"fit_xg": G[f], "fit_xh": H[f], "fit_y": y[rows][f], "ev_xg": G[~f], "ev_xh": H[~f], "ev_y": y[rows][~f],
            "ev_cl": None}


# ---------------------------------------------------------------- M3 预筛
def importance_rank(P: MS.Params, X: np.ndarray, y: np.ndarray, kind: str, cols: Optional[Sequence[int]] = None
                    ) -> List[int]:
    """原版参数 XGBoost（seed = tune seed）在 X 上的重要性排序（kind：total_gain / weight）。
    只返回 cols（默认全部列）中重要性 > 0 的列号，按重要性降序、并列按列号升序。"""
    import xgboost as xgb
    d = MS.dmat(P, X, y, P.bins_default)
    cfg = P.default_cfg
    bst = xgb.train(MS.xgb_params(P, cfg, P.bins_default, MS.HALF, P.tune_seed), d, int(cfg["rounds"]))
    score = {int(k[1:]): float(v) for k, v in bst.get_score(importance_type=kind).items()}
    pool = range(X.shape[1]) if cols is None else cols
    hit = [j for j in pool if score.get(j, 0.0) > 0]
    return sorted(hit, key=lambda j: (-score[j], j))


def take(rank: List[int], k) -> Optional[List[int]]:
    """前 k 列（k = ALL 时为全部清洗后列，按原列序）；返回升序列号，保持编码列的原顺序。"""
    return sorted(rank[:k]) if k != ALL else None


# ---------------------------------------------------------------- 一个任务
def select_columns(num: np.ndarray, cat, y: np.ndarray, tfit: np.ndarray, guest: set, P: MS.Params, k_main: int,
                   num_vars: Sequence[str] = LARGE_NUM) -> Tuple[List[Dict], List[int], List[int]]:
    """列清单与两方排序：只读 tfit 行（tune 视图 fit 行 = 训练去掉验证）的特征与标签。
    内地：只用内地列，按总增益排序；香港：「内地前 k_main 列 + 香港全部列」上按香港各列被选为分裂的次数排序。"""
    plan = column_plan(num, cat, tfit, guest, num_vars)
    none = np.zeros_like(tfit)
    yf = y[tfit]
    G, _ = encode(num, cat, y, plan, tfit, none, "g")
    rank_g = importance_rank(P, G, yf, "total_gain")
    ctx = G[:, rank_g[:k_main]].copy()
    del G
    X, _ = encode(num, cat, y, plan, tfit, none, "h", lead=ctx)
    rank_h = [j - ctx.shape[1] for j in importance_rank(P, X, yf, "weight", cols=range(ctx.shape[1], X.shape[1]))]
    return plan, rank_g, rank_h


def read_labels(src: Path, label: str) -> np.ndarray:
    import pandas as pd
    return (pd.read_csv(src / ("orange_large_train_%s.labels" % label), header=None)[0].to_numpy() == 1).astype(int)


def build(label: str, num: np.ndarray, cat, y: np.ndarray, priv: Path, prep: Dict, P: MS.Params, k_main: int,
          num_vars: Sequence[str] = LARGE_NUM, cat_vars: Sequence[str] = LARGE_CAT,
          all_num: Optional[Sequence[str]] = None) -> Dict:
    """准备一个任务：主分析 npz / json（每方前 k_main 列）与敏感性 npz（final 视图全部列 + 两方排序）。"""
    assert len(y) == num.shape[0] == len(cat)
    te, va = MD.simple_split(y, prep)
    seed = int(prep["split_seed"])
    full = list(all_num if all_num is not None else num_vars)
    guest = party_vars(seed, full, cat_vars)
    tfit, tev, ffit, fev = ~te & ~va, va, ~te, te
    plan, rank_g, rank_h = select_columns(num, cat, y, tfit, guest, P, k_main, num_vars)
    gnames, hnames = encoded_names(plan, "g"), encoded_names(plan, "h")
    keep_g = [gnames[j] for j in sorted(rank_g[:k_main])]
    keep_h = [hnames[j] for j in sorted(rank_h[:k_main])]
    views = {}
    for vname, (fit, ev) in {"tune": (tfit, tev), "final": (ffit, fev)}.items():
        G, gc = encode(num, cat, y, plan, fit, ev, "g", keep_g)
        H, hc = encode(num, cat, y, plan, fit, ev, "h", keep_h)
        assert gc == keep_g and hc == keep_h
        views[vname] = view_of(G, H, y, fit, ev)
    name = name_of(label)
    meta = {"source": "Orange KDD Cup 2009 large · " + label,
            "split": "原始变量随机对半（random.Random(%d)）；测试分层 20%%；验证 = 训练内分层 20%%；"
                     "M3 预筛：每方前 %d 列（tune 视图 fit 行）" % (seed, k_main),
            "cluster": "行", "guest_cols": keep_g, "host_cols": keep_h,
            "prescreen": {"k": k_main, "vars_guest": len(guest), "vars_host": len(full) + len(cat_vars) - len(guest),
                          "vars_nonempty_num": len(num_vars),
                          "clean_guest": len(gnames), "clean_host": len(hnames),
                          "nonzero_guest": len(rank_g), "nonzero_host": len(rank_h)}}
    info = MD.pack(views, meta, name, priv)
    del views
    # 敏感性：final 视图全部清洗后列；两方分开编码、分开存（整块写出、不另行切片复制），降低内存峰值
    rows = ffit | fev
    for side, rank in (("g", rank_g), ("h", rank_h)):
        M, _ = encode(num, cat, y, plan, ffit, fev, side)
        np.save(priv / ("%s__sens_%s.npy" % (name, side)), M)
        np.savez(priv / ("%s__sens_%s.npz" % (name, side)), rank=np.asarray(rank, np.int64), fit=ffit[rows], y=y[rows])
        del M
        release_memory()
    return info


# ---------------------------------------------------------------- 敏感性（D-3）
def gather(parts: Sequence[Tuple[np.ndarray, Optional[List[int]]]], idx: np.ndarray) -> np.ndarray:
    """按行块把若干 (矩阵, 列号或 None = 全部列) 的 idx 行拼成一个 float32 矩阵，避免整块切片的中间副本。"""
    widths = [M.shape[1] if c is None else len(c) for M, c in parts]
    out = np.empty((len(idx), sum(widths)), np.float32)
    for i0 in range(0, len(idx), ROW_BLOCK):
        r = idx[i0:i0 + ROW_BLOCK]
        j0 = 0
        for (M, c), w in zip(parts, widths):
            out[i0:i0 + len(r), j0:j0 + w] = M[r] if c is None else M[np.ix_(r, np.asarray(c, np.int64))]
            j0 += w
    return out


def sensitivity(P: MS.Params, res: Dict, sens: Dict[str, np.ndarray], ks: Sequence, k_main: int) -> Dict:
    """主分析选定的优化版联邦配置（central）与内地单边配置（guest）下，不同 k 的测试 AUROC 与相对 k_main 的配对差。
    ks 按给定顺序评估；ALL 须放在最后：组好全部列的矩阵后即释放两方原矩阵。"""
    fit, y = sens.pop("fit"), sens.pop("y")
    fi, ei = np.flatnonzero(fit), np.flatnonzero(~fit)
    yf, ye = y[fi], y[ei]
    pf = MS.prior_fn(P)(yf)
    cc, cg = res["configs"]["central"], res["configs"]["guest"]
    rank_g, rank_h = list(sens["rank_g"]), list(sens["rank_h"])
    scores: Dict[str, List[np.ndarray]] = {}
    sizes = {}
    for k in ks:
        g, h = take(rank_g, k), take(rank_h, k)
        parts = [(sens["G"], g), (sens["H"], h)]
        Xf, Xe = gather(parts, fi), gather(parts, ei)
        ng = sens["G"].shape[1] if g is None else len(g)
        if k == ALL:
            del parts
            sens.pop("G"), sens.pop("H")
        sizes[str(k)] = {"guest": int(ng), "host": int(Xf.shape[1] - ng)}
        for key, (Xa, Xb, ck) in {"F_opt_p": (Xf, Xe, cc), "G_t": (Xf[:, :ng], Xe[:, :ng], cg)}.items():
            if MS.deterministic(ck):
                s0 = MS.train_pred(P, Xa, yf, Xb, ck, P.bins_fine, pf, P.seeds[0])
                scores["%s@%s" % (key, k)] = [s0] * len(P.seeds)
            else:
                scores["%s@%s" % (key, k)] = [MS.train_pred(P, Xa, yf, Xb, ck, P.bins_fine, pf, sd) for sd in P.seeds]
            MS.log("敏感性 k=%s · %s 测试 AUROC %.4f" % (k, key, np.mean([MS.auc(ye, s) for s in scores["%s@%s" % (key, k)]])))
        del Xf, Xe
    A = MS.two_level_bootstrap(P, ye, None, scores)
    mean = {key: float(np.mean([MS.auc(ye, s) for s in v])) for key, v in scores.items()}
    rows: List[Dict[str, Any]] = []
    for k in ks:
        for key in ("F_opt_p", "G_t"):
            a, b = "%s@%s" % (key, k), "%s@%s" % (key, k_main)
            row: Dict[str, Any] = {"k": str(k), "method": key, "auroc": mean[a], **sizes[str(k)]}
            row["vs_k_main"] = None if k == k_main else CV.paired(A, a, b, mean[a] - mean[b])
            rows.append(row)
        a, b = "F_opt_p@%s" % k, "G_t@%s" % k
        rows.append({"k": str(k), "method": "F_opt_p − G_t", "auroc": None, **sizes[str(k)],
                     "vs_k_main": None, "federal_gain": CV.paired(A, a, b, mean[a] - mean[b])})
    return {"rows": rows, "k_main": k_main}


def load_sens(priv: Path, name: str) -> Dict[str, np.ndarray]:
    """两方全部列矩阵以只读映射打开（不占匿名内存），排序、fit 掩码与标签读入内存。"""
    g, h = (np.load(priv / ("%s__sens_%s.npz" % (name, side))) for side in ("g", "h"))
    G, H = (np.load(priv / ("%s__sens_%s.npy" % (name, side)), mmap_mode="r") for side in ("g", "h"))
    return {"rank_g": g["rank"], "rank_h": h["rank"], "G": G, "H": H, "fit": g["fit"], "y": g["y"]}


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


# ---------------------------------------------------------------- 命令行
def ks_of(raw: Sequence) -> List:
    return [k if k == ALL else int(k) for k in raw]


def main(argv: Optional[List[str]] = None) -> int:
    """python -m modules.m5_modeling.components.method_screen_large --config <yaml> [--labels churn ...]

    环境变量：基础配置的 private_root_env / raw_root_env，与本步配置的 run_dir_env（每个任务的 S5.M1 JSON、
    敏感性 JSON 与交叉验证 JSON）。已完成的部分跳过，可续跑。
    """
    import argparse
    import os

    import yaml
    ap = argparse.ArgumentParser(description="S5.M5 Orange 大版 + M3 预筛")
    ap.add_argument("--config", default="modules/m5_modeling/configs/s5_m5_orange_large.yaml")
    ap.add_argument("--labels", nargs="*")
    a = ap.parse_args(argv)
    m5 = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(m5["base_config"]).read_text(encoding="utf-8"))
    m3 = yaml.safe_load(Path(m5["cv_config"]).read_text(encoding="utf-8"))
    P = MS.Params(cfg)
    priv, raw = Path(os.environ[cfg["private_root_env"]]), Path(os.environ[cfg["raw_root_env"]])
    out = Path(os.environ[m5["run_dir_env"]])
    out.mkdir(parents=True, exist_ok=True)
    src = raw / m5["raw_subdir"]
    labels = a.labels or list(m5["labels"])
    k_main = int(m5["k_main"])
    todo = [lb for lb in labels if not (priv / (name_of(lb) + ".npz")).exists()]
    if todo:
        import pandas as pd
        cache_num, cache_cat, cache_kept = (priv / ("orange_large__%s" % s) for s in ("num.npy", "cat.pkl", "kept.json"))
        if cache_num.exists() and cache_cat.exists() and cache_kept.exists():
            kept = json.loads(cache_kept.read_text(encoding="utf-8"))
            num, cat = np.load(cache_num, mmap_mode="r"), pd.read_pickle(cache_cat)
            MS.log("Orange 大版读取私有缓存（数值列按列映射）")
        else:
            kept = scan_numeric(src)
            MS.log("Orange 大版第一遍：%d / %d 个数值变量非空且非常数" % (len(kept), len(LARGE_NUM)))
            num, cat = read_raw(src, num_vars=kept, all_num=LARGE_NUM, num_path=cache_num)
            cat.to_pickle(cache_cat)
            cache_kept.write_text(json.dumps(kept), encoding="utf-8")
        MS.log("Orange 大版读取完成：%d 行、%d 个数值变量、%d 个类别变量" % (num.shape[0], num.shape[1], cat.shape[1]))
        for lb in todo:
            info = build(lb, num, cat, read_labels(src, lb), priv, cfg["prep"], P, k_main, num_vars=kept,
                         all_num=LARGE_NUM)
            MS.log("%s 数据已准备：%s" % (name_of(lb), info["prescreen"]))
            release_memory()
        del num, cat
        release_memory()
    for lb in labels:
        name = name_of(lb)
        if not (out / (name + ".json")).exists():
            MS.Screen(name, cfg, priv, out).run()
        res = json.loads((out / (name + ".json")).read_text(encoding="utf-8"))
        if not (out / (name + "__sens.json")).exists():
            write_json(out / (name + "__sens.json"), sensitivity(P, res, load_sens(priv, name), ks_of(m5["k_sens"]), k_main))
        if not (out / (name + "__cv.json")).exists():
            write_json(out / (name + "__cv.json"), CV.run(name, cfg, m3, priv, out, {}))
        MS.log("%s 完成" % name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
