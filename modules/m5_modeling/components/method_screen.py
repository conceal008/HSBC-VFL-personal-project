"""M5 · S5.M1 方法筛选：文献优化方法在明文直方图 GBDT 代理上的逐数据集评估与组合选择（DR-M5-012）。

代理依据：隐语 SGB 只加密梯度，分裂搜索与同分桶、同超参的明文直方图 GBDT 数学等价（INNOVATIONS 创新点 2 / 4，
代理与 SGB 实测差 ≤ 0.003）。所以先在代理上筛方法，再用真实 SGB 复核选出的组合（S5.M2）。

口径（DR-M5-012 · D-2 / D-4）：
- 配置、融合权重、蒸馏系数与组合只按 tune 视图的验证集选；final 视图的测试集每个方法评一次。
- 6 个 seed；AUROC 点估计 = 6 个 seed 的均值；配对差区间来自两层 bootstrap（测试客户簇 × seed）。
- 「联邦」臂用两方特征拼接训练（代理），「内地单边」只用标签方特征，同等调参。

需要 xgboost（2.1.x）与 scipy；导入本模块不依赖它们，调用时才导入。
"""
from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

PROB_CLIP = 1e-7                 # logit 计算的概率截断
PERCENTILE_LOW = 2.5
PERCENTILE_HIGH = 97.5
TOP_REPORT = 3                   # 结果 JSON 里保留的调参前几名
TIE_DECIMALS = 6                 # 选 w / α 时验证 AUC 的比较精度（并列取较大者）
LOG_DECIMALS = 4
PERCENT = 100
# 由主 seed 派生子随机流的乘子 / 偏移（与 S5.M1 运行时代码逐位一致）
BAG_SEED_MULT = 100
DP_SEED_MULT = 7
MISMATCH_SEED_MULT = 13
DP_SPLIT_OFFSET = 1
DROP_THRESHOLD = 0.5             # M5 剔除：置信分数低于此值的样本不进训练
HALF = 0.5                       # 隐语 SGB 默认初始分数 0 对应的概率


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def logit(p):
    p = np.clip(p, PROB_CLIP, 1 - PROB_CLIP)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Params:
    """把配置 screen 段展开为属性；seeds 来自配置顶层。"""

    def __init__(self, cfg: Dict):
        s = cfg["screen"]
        self.seeds: List[int] = list(cfg["seeds"])
        self.tune_seed: int = int(cfg["seed"])
        g = s["grid"]
        self.grid = [dict(zip(("lr", "depth", "lam", "col"), v))
                     for v in itertools.product(g["lr"], g["depth"], g["lam"], g["col"])]
        self.default_cfg = dict(s["default_cfg"])
        self.bins_default, self.bins_fine = int(s["bins_default"]), int(s["bins_fine"])
        self.max_rounds, self.early_stop, self.tol = int(s["max_rounds"]), int(s["early_stop"]), float(s["tol"])
        self.sub_grid, self.m4_top, self.folds = list(s["sub_grid"]), int(s["m4_top"]), int(s["folds"])
        self.fuse_w, self.student_alpha = list(s["fuse_w"]), list(s["student_alpha"])
        self.dp_eps, self.mismatch, self.conf_auc = list(s["dp_eps"]), list(s["mismatch"]), float(s["conf_auc"])
        self.bag, self.bag_sub = int(s["bag"]), float(s["bag_sub"])
        self.n_boot, self.boot_seed = int(s["n_boot"]), int(s["boot_seed"])
        self.nthread, self.eps_p = int(s["nthread"]), float(s["eps_p"])


# ---------------------------------------------------------------- 训练原语
def dmat(P: Params, X, y=None, bins=None, ref=None, margin=None, weight=None):
    import xgboost as xgb
    return xgb.QuantileDMatrix(X, label=y, max_bin=bins or P.bins_fine, ref=ref, base_margin=margin,
                               weight=weight, nthread=P.nthread)


def xgb_params(P: Params, cfg: Dict, bins: int, base: Optional[float], seed: int) -> Dict:
    p = {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist", "max_bin": bins,
         "eta": cfg["lr"], "max_depth": cfg["depth"], "reg_lambda": cfg["lam"], "colsample_bytree": cfg["col"],
         "subsample": cfg.get("sub", 1.0), "seed": seed, "nthread": P.nthread, "verbosity": 0}
    if base is not None:
        p["base_score"] = float(base)
    return p


def deterministic(cfg: Dict) -> bool:
    return cfg["col"] == 1.0 and cfg.get("sub", 1.0) == 1.0


def train_pred(P: Params, Xf, yf, Xe, cfg, bins, base, seed, mf=None, me=None, weight=None, margin_out=False):
    import xgboost as xgb
    dtr = dmat(P, Xf, yf, bins, margin=mf, weight=weight)
    dte = dmat(P, Xe, None, bins, ref=dtr, margin=me)
    bst = xgb.train(xgb_params(P, cfg, bins, base, seed), dtr, int(cfg["rounds"]))
    return bst.predict(dte, output_margin=margin_out)


def tune(P: Params, Xf, yf, Xe, ye, bins, base, mf=None, me=None, grid=None, sub=1.0) -> List[Dict]:
    import xgboost as xgb
    dtr = dmat(P, Xf, yf, bins, margin=mf)
    dev = dmat(P, Xe, ye, bins, ref=dtr, margin=me)
    rows = []
    for g in (grid if grid is not None else P.grid):
        cfg = {**g, "sub": g.get("sub", sub)}
        bst = xgb.train(xgb_params(P, cfg, bins, base, P.tune_seed), dtr, P.max_rounds, evals=[(dev, "v")],
                        early_stopping_rounds=P.early_stop, verbose_eval=False)
        rows.append({**cfg, "rounds": int(bst.best_iteration) + 1, "val_auc": float(bst.best_score)})
    rows.sort(key=lambda r: -r["val_auc"])
    return rows


def cost(cfg: Dict, bags: int = 1) -> float:
    """联邦计算量代理：轮数 × 深度 × 行采样 × 模型个数（SGB 每层一次全样本密文累加）。"""
    return cfg["rounds"] * cfg["depth"] * cfg.get("sub", 1.0) * bags


def pick(rows: List[Dict], tol: float) -> Dict:
    best = rows[0]["val_auc"]
    return min([r for r in rows if r["val_auc"] >= best - tol], key=cost)


def auc(y, s) -> float:
    """带并列处理的 AUROC（与 sklearn roc_auc_score 相同）。"""
    o = np.argsort(s, kind="mergesort")
    ss, yy = s[o], y[o].astype(float)
    g = np.r_[0, np.cumsum(ss[1:] != ss[:-1])]
    pos = np.bincount(g, weights=yy)
    neg = np.bincount(g, weights=1 - yy)
    return float(np.dot(pos, np.cumsum(neg) - neg + HALF * neg) / (pos.sum() * neg.sum()))


def prior_fn(P: Params) -> Callable:
    return lambda y: float(np.clip(np.mean(y), P.eps_p, 1 - P.eps_p))


def cross_fit(P: Params, Xf, yf, Xe, cfg, bins, seed, ex=None):
    """内地模型：fit 行的折外 logit 与 eval 行的 logit；ex = (X_extra, y_extra) 只进训练折。"""
    prior = prior_fn(P)
    part = np.random.default_rng(seed).integers(0, P.folds, len(yf))
    oof = np.empty(len(yf))
    for k in range(P.folds):
        tr = part != k
        X, y = Xf[tr], yf[tr]
        if ex is not None:
            X, y = np.vstack([X, ex[0]]), np.r_[y, ex[1]]
        oof[~tr] = train_pred(P, X, y, Xf[~tr], cfg, bins, prior(y), seed, margin_out=True)
    X, y = Xf, yf
    if ex is not None:
        X, y = np.vstack([X, ex[0]]), np.r_[y, ex[1]]
    return oof, train_pred(P, X, y, Xe, cfg, bins, prior(y), seed, margin_out=True)


def label_posterior(q, yn, rho, eps_p):
    """I6：q = 加噪标签上训练的内地模型输出；去偏得 p(x)，再按贝叶斯公式得后验软标签。"""
    p = np.clip((q - rho) / (1 - 2 * rho), eps_p, 1 - eps_p)
    post1 = p * (1 - rho) / (p * (1 - rho) + (1 - p) * rho)
    post0 = p * rho / (p * rho + (1 - p) * (1 - rho))
    return np.where(yn == 1, post1, post0)


def holm(pvals: List[float]) -> List[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out, run, m = [0.0] * len(pvals), 0.0, len(pvals)
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        out[i] = run
    return out


CONTRAST_PAIRS = [
    ("F_bins", "F_def", "I2 只细分桶"), ("F_tune", "F_def", "I2 只调参"), ("F_opt", "F_def", "I2 细分桶 + 调参"),
    ("F_opt_p", "F_opt", "I3 先验初始分数"), ("F_opt_p", "F_def", "I2 + I3 合计"),
    ("F_def", "G_t", "原版联邦 − 内地单边"), ("F_opt_p", "G_t", "优化版联邦 − 内地单边"),
    ("F_M4", "F_opt_p", "M4 行采样调参"), ("F_M1", "F_opt_p", "M1 逐样本先验"),
    ("F_M1x", "F_opt_p", "M1x 逐样本先验（含非共同客户）"), ("F_M1x", "G_x", "M1x − 内地单边（含非共同客户）"),
    ("G_x", "G_t", "内地单边加非共同客户"), ("F_M6", "F_opt_p", "M6 bagging"),
    ("M2_fuse", "F_opt_p", "M2 分数融合 − 优化版"), ("M2_fuse", "G_t", "M2 分数融合 − 内地单边"),
    ("M2_student", "G_t", "M2 蒸馏学生 − 内地单边"),
    ("M2_student_x", "G_x", "M2 蒸馏学生（含非共同客户）− 内地单边（含）"),
    ("COMBO", "F_opt_p", "验证集选定组合 − 优化版"), ("COMBO", "G_t", "验证集选定组合 − 内地单边"),
    ("COMBO", "F_def", "验证集选定组合 − 原版联邦")]


def contrast_pairs(P: Params) -> List[Tuple[str, str, str]]:
    pairs = list(CONTRAST_PAIRS)
    for e in P.dp_eps:
        t = "e%d" % int(e)
        pairs += [("DP_soft_" + t, "DP_hard_" + t, "I6 软 − 硬（ε=%d）" % int(e)),
                  ("DP_soft_" + t, "G_t", "I6 软标签联邦 − 内地单边（ε=%d）" % int(e)),
                  ("DP_hard_" + t, "G_t", "硬 RR 联邦 − 内地单边（ε=%d）" % int(e))]
    for r in P.mismatch:
        t = "r%d" % int(r * PERCENT)
        pct = int(r * PERCENT)
        pairs += [("M5_none_" + t, "F_opt_p", "错配 %d%% 的退化" % pct),
                  ("M5_weight_" + t, "M5_none_" + t, "M5 置信度加权（错配 %d%%）" % pct),
                  ("M5_drop_" + t, "M5_none_" + t, "M5 剔除低置信（错配 %d%%）" % pct)]
    return pairs


def two_level_bootstrap(P: Params, y, cl, scores: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    """每次重抽：测试客户簇有放回 + 各方法的 seed 有放回；返回各方法 n_boot 个 AUROC。"""
    y = y.astype(float)
    cl = np.arange(len(y)) if cl is None else cl
    _, ci = np.unique(cl, return_inverse=True)
    C = int(ci.max()) + 1
    keys = list(scores)
    prep = {}
    for k in keys:
        groups = []
        for s in scores[k]:
            o = np.argsort(s, kind="mergesort")
            ss = s[o]
            g = np.empty(len(s), np.int64)
            g[o] = np.r_[0, np.cumsum(ss[1:] != ss[:-1])]
            groups.append((g, int(g.max()) + 1))
        prep[k] = groups
    same = {k: all(x is scores[k][0] for x in scores[k]) for k in keys}
    rng = np.random.default_rng(P.boot_seed)
    A = {k: np.empty(P.n_boot) for k in keys}
    for b in range(P.n_boot):
        w = np.bincount(rng.integers(0, C, C), minlength=C)[ci].astype(float)
        wp, wn = w * y, w * (1 - y)
        tot = wp.sum() * wn.sum()
        cache: Dict[Tuple[str, int], float] = {}
        for k in keys:
            n = len(prep[k])
            draw = rng.integers(0, n, n)
            vals = []
            for j in draw:
                key = (k, 0) if same[k] else (k, int(j))
                if key not in cache:
                    g, G = prep[k][key[1]]
                    pos = np.bincount(g, weights=wp, minlength=G)
                    neg = np.bincount(g, weights=wn, minlength=G)
                    cache[key] = float(np.dot(pos, np.cumsum(neg) - neg + HALF * neg) / tot)
                vals.append(cache[key])
            A[k][b] = np.mean(vals)
    return A


# ---------------------------------------------------------------- 主流程
class Screen:
    def __init__(self, name: str, cfg: Dict, priv: Path, out: Path):
        self.name, self.P, self.out = name, Params(cfg), out
        d = np.load(priv / (name + ".npz"))
        self.meta = json.loads((priv / (name + ".json")).read_text(encoding="utf-8"))
        self.v: Dict[str, Dict] = {}
        for view in ("tune", "final"):
            g = {k.split("__", 1)[1]: d[k] for k in d.files if k.startswith(view + "__")}
            g["fit_xc"] = np.hstack([g["fit_xg"], g["fit_xh"]])
            g["ev_xc"] = np.hstack([g["ev_xg"], g["ev_xh"]])
            g["ex"] = (g["ex_xg"], g["ex_y"]) if "ex_xg" in g else None
            self.v[view] = g
        self.has_ex = self.v["final"]["ex"] is not None
        self.res: Dict = {"dataset": name, "meta": self.meta, "configs": {}, "val": {}, "test": {}, "timing": {}}
        self.scores: Dict[str, List[np.ndarray]] = {}
        self.costs: Dict[str, float] = {}

    def T(self, label, t0):
        self.res["timing"][label] = round(time.perf_counter() - t0, 1)
        log("%s · %s 用时 %ss" % (self.name, label, self.res["timing"][label]))

    def final_seeds(self, key, fn, cfg=None, det=None):
        """fn(seed) → 测试分数；确定性方法只拟合一次再复制。"""
        det = deterministic(cfg) if det is None and cfg is not None else bool(det)
        seeds = self.P.seeds
        self.scores[key] = [fn(seeds[0])] * len(seeds) if det else [fn(sd) for sd in seeds]
        y = self.v["final"]["ev_y"]
        a = [auc(y, s) for s in self.scores[key]]
        self.res["test"][key] = {"auroc": float(np.mean(a)), "per_seed": [float(x) for x in a],
                                 "seed_sd": float(np.std(a, ddof=1)) if not det else 0.0}
        log("%s · %-16s 测试 AUROC %.4f" % (self.name, key, np.mean(a)))

    # ---------- 1) 调参
    def tune_all(self):
        P, tv = self.P, self.v["tune"]
        prior = prior_fn(P)
        ytv, pt = tv["ev_y"], prior(tv["fit_y"])
        t0 = time.perf_counter()
        rows_g = tune(P, tv["fit_xg"], tv["fit_y"], tv["ev_xg"], ytv, P.bins_fine, pt)
        rows_c = tune(P, tv["fit_xc"], tv["fit_y"], tv["ev_xc"], ytv, P.bins_fine, pt)
        self.cg, self.cc = pick(rows_g, P.tol), pick(rows_c, P.tol)
        self.res["configs"].update({"guest": self.cg, "central": self.cc, "guest_top": rows_g[:TOP_REPORT],
                                    "central_top": rows_c[:TOP_REPORT]})
        self.T("tune_guest_central", t0)
        t0 = time.perf_counter()
        top = [{k: r[k] for k in ("lr", "depth", "lam", "col")} for r in rows_c[:P.m4_top]]
        rows_m4 = rows_c + [r for s in P.sub_grid
                            for r in tune(P, tv["fit_xc"], tv["fit_y"], tv["ev_xc"], ytv, P.bins_fine, pt,
                                          grid=top, sub=s)]
        rows_m4.sort(key=lambda r: -r["val_auc"])
        self.cm4 = pick(rows_m4, P.tol)
        self.res["configs"]["m4"] = self.cm4
        self.T("tune_m4", t0)
        self.cgx = None
        if self.has_ex:
            t0 = time.perf_counter()
            Xa, ya = np.vstack([tv["fit_xg"], tv["ex"][0]]), np.r_[tv["fit_y"], tv["ex"][1]]
            self.cgx = pick(tune(P, Xa, ya, tv["ev_xg"], ytv, P.bins_fine, prior(ya)), P.tol)
            self.res["configs"]["guest_extra"] = self.cgx
            self.T("tune_guest_extra", t0)
        t0 = time.perf_counter()
        mf, me = cross_fit(P, tv["fit_xg"], tv["fit_y"], tv["ev_xg"], self.cg, P.bins_fine, P.tune_seed)
        self.cm1 = pick(tune(P, tv["fit_xc"], tv["fit_y"], tv["ev_xc"], ytv, P.bins_fine, None, mf=mf, me=me), P.tol)
        self.res["configs"]["m1"] = self.cm1
        self.tune_margin = {"m1": (mf, me)}
        self.cm1x = None
        if self.has_ex:
            mfx, mex = cross_fit(P, tv["fit_xg"], tv["fit_y"], tv["ev_xg"], self.cgx, P.bins_fine, P.tune_seed,
                                 ex=tv["ex"])
            self.cm1x = pick(tune(P, tv["fit_xc"], tv["fit_y"], tv["ev_xc"], ytv, P.bins_fine, None,
                                  mf=mfx, me=mex), P.tol)
            self.res["configs"]["m1x"] = self.cm1x
            self.tune_margin["m1x"] = (mfx, mex)
        self.T("tune_m1", t0)

    # ---------- 2) 验证集上的选择
    def select(self):
        P, tv = self.P, self.v["tune"]
        prior = prior_fn(P)
        ytv, pt = tv["ev_y"], prior(tv["fit_y"])
        cc, cm4, cm1, cm1x = self.cc, self.cm4, self.cm1, self.cm1x
        t0 = time.perf_counter()

        def vpred(cfg, base, margins=None, seed=P.tune_seed):
            mf_, me_ = margins if margins else (None, None)
            return train_pred(P, tv["fit_xc"], tv["fit_y"], tv["ev_xc"], cfg, P.bins_fine, base, seed, mf=mf_, me=me_)

        def vbag(cfg, base, margins=None):
            c = {**cfg, "sub": P.bag_sub}
            return np.mean([vpred(c, base, margins, seed=P.tune_seed * BAG_SEED_MULT + j) for j in range(P.bag)],
                           axis=0)

        tm = self.tune_margin
        vs = {"P": vpred(cc, pt), "P_M4": vpred(cm4, pt), "M1": vpred(cm1, None, tm["m1"]),
              "P+M6": vbag(cc, pt), "M1+M6": vbag(cm1, None, tm["m1"]), "P_M4+M6": vbag(cm4, pt)}
        cand_cost = {"P": cost(cc), "P_M4": cost(cm4), "M1": cost(cm1),
                     "P+M6": cost({**cc, "sub": P.bag_sub}, P.bag), "M1+M6": cost({**cm1, "sub": P.bag_sub}, P.bag),
                     "P_M4+M6": cost({**cm4, "sub": P.bag_sub}, P.bag)}
        if self.has_ex:
            vs["M1x"] = vpred(cm1x, None, tm["m1x"])
            vs["M1x+M6"] = vbag(cm1x, None, tm["m1x"])
            cand_cost.update({"M1x": cost(cm1x), "M1x+M6": cost({**cm1x, "sub": P.bag_sub}, P.bag)})
        val = {k: auc(ytv, s) for k, s in vs.items()}
        best = max(val.values())
        self.combo = min([k for k in val if val[k] >= best - P.tol], key=lambda k: cand_cost[k])
        self.res["val"]["combo_candidates"] = {k: {"val_auc": val[k], "cost": cand_cost[k]} for k in val}
        self.res["val"]["combo_choice"] = self.combo
        vg = train_pred(P, tv["fit_xg"], tv["fit_y"], tv["ev_xg"], self.cg, P.bins_fine, pt, P.tune_seed)
        fa = {w: auc(ytv, w * logit(vs["P"]) + (1 - w) * logit(vg)) for w in P.fuse_w}
        self.w_best = max(P.fuse_w, key=lambda w: (round(fa[w], TIE_DECIMALS), w))
        self.res["val"]["fuse"] = {"w": self.w_best, "val_auc": fa[self.w_best], "val_auc_guest": auc(ytv, vg),
                                   "val_auc_fed": val["P"]}
        sa = self.student_select(self.cg)
        self.a_best = max(P.student_alpha, key=lambda a: (round(sa[a], TIE_DECIMALS), a))
        self.res["val"]["student"] = {"alpha": self.a_best, "val_auc": sa}
        if self.has_ex:
            sax = self.student_select(self.cgx, ex=tv["ex"])
            self.res["val"]["student_x"] = {"alpha": max(P.student_alpha, key=lambda a: (round(sax[a], TIE_DECIMALS), a)),
                                            "val_auc": sax}
        self.T("val_selection", t0)
        log("%s · 组合候选（验证 AUC）%s → 选 %s；融合 w=%s；蒸馏 α=%s" % (
            self.name, {k: round(v, LOG_DECIMALS) for k, v in val.items()}, self.combo, self.w_best, self.a_best))

    def student_select(self, cfg_s, ex=None):
        P, tv = self.P, self.v["tune"]
        prior = prior_fn(P)
        fy = tv["fit_y"]
        part = np.random.default_rng(P.tune_seed).integers(0, P.folds, len(fy))
        q = np.empty(len(fy))
        for k in range(P.folds):
            trm = part != k
            q[~trm] = train_pred(P, tv["fit_xc"][trm], fy[trm], tv["fit_xc"][~trm], self.cc, P.bins_fine,
                                 prior(fy[trm]), P.tune_seed)
        out = {}
        for a in P.student_alpha:
            t = a * fy + (1 - a) * q
            X, yy = tv["fit_xg"], t
            if ex is not None:
                X, yy = np.vstack([X, ex[0]]), np.r_[t, ex[1]]
            out[a] = auc(tv["ev_y"], train_pred(P, X, yy, tv["ev_xg"], cfg_s, P.bins_fine, prior(yy), P.tune_seed))
        return out

    # ---------- 3) 测试集
    def test_all(self):
        P, fv = self.P, self.v["final"]
        prior = prior_fn(P)
        Xf, yf, Xe = fv["fit_xc"], fv["fit_y"], fv["ev_xc"]
        Xgf, Xge = fv["fit_xg"], fv["ev_xg"]
        pf, dflt = prior(yf), P.default_cfg
        cg, cc, cm4, cm1, cm1x = self.cg, self.cc, self.cm4, self.cm1, self.cm1x
        bd, bf = P.bins_default, P.bins_fine
        t0 = time.perf_counter()
        self.final_seeds("G_def", lambda s: train_pred(P, Xgf, yf, Xge, dflt, bd, HALF, s), dflt)
        self.final_seeds("G_t", lambda s: train_pred(P, Xgf, yf, Xge, cg, bf, pf, s), cg)
        self.final_seeds("F_def", lambda s: train_pred(P, Xf, yf, Xe, dflt, bd, HALF, s), dflt)
        self.final_seeds("F_bins", lambda s: train_pred(P, Xf, yf, Xe, dflt, bf, HALF, s), dflt)
        self.final_seeds("F_tune", lambda s: train_pred(P, Xf, yf, Xe, cc, bd, HALF, s), cc)
        self.final_seeds("F_opt", lambda s: train_pred(P, Xf, yf, Xe, cc, bf, HALF, s), cc)
        self.final_seeds("F_opt_p", lambda s: train_pred(P, Xf, yf, Xe, cc, bf, pf, s), cc)
        self.final_seeds("F_M4", lambda s: train_pred(P, Xf, yf, Xe, cm4, bf, pf, s), cm4)
        self.costs.update({"G_def": 0, "G_t": 0, "F_def": cost(dflt), "F_bins": cost(dflt), "F_tune": cost(cc),
                           "F_opt": cost(cc), "F_opt_p": cost(cc), "F_M4": cost(cm4)})
        self.T("test_core", t0)

        t0 = time.perf_counter()
        margins: Dict[str, Dict[int, Tuple]] = {}

        def m1_scores(key, cfg_g, cfg_r, ex=None):
            def fn(s):
                mf_, me_ = cross_fit(P, Xgf, yf, Xge, cfg_g, bf, s, ex=ex)
                margins.setdefault(key, {})[s] = (mf_, me_)
                return train_pred(P, Xf, yf, Xe, cfg_r, bf, None, s, mf=mf_, me=me_)
            self.final_seeds(key, fn, det=False)
            self.costs[key] = cost(cfg_r)

        m1_scores("F_M1", cg, cm1)
        if self.has_ex:
            Xga, yga = np.vstack([Xgf, fv["ex"][0]]), np.r_[yf, fv["ex"][1]]
            self.final_seeds("G_x", lambda s: train_pred(P, Xga, yga, Xge, self.cgx, bf, prior(yga), s), self.cgx)
            self.costs["G_x"] = 0
            m1_scores("F_M1x", self.cgx, cm1x, ex=fv["ex"])
        self.T("test_m1", t0)

        t0 = time.perf_counter()

        def bag_fn(cfg, base, mkey=None):
            c = {**cfg, "sub": P.bag_sub}

            def fn(s):
                mf_, me_ = margins[mkey][s] if mkey else (None, None)
                return np.mean([train_pred(P, Xf, yf, Xe, c, bf, base, s * BAG_SEED_MULT + j, mf=mf_, me=me_)
                                for j in range(P.bag)], axis=0)
            return fn

        self.final_seeds("F_M6", bag_fn(cc, pf), det=False)
        self.costs["F_M6"] = cost({**cc, "sub": P.bag_sub}, P.bag)
        self.T("test_m6", t0)

        t0 = time.perf_counter()
        combo = self.combo
        combo_map = {"P": "F_opt_p", "P_M4": "F_M4", "M1": "F_M1", "M1x": "F_M1x", "P+M6": "F_M6"}
        if combo in combo_map:
            self.scores["COMBO"] = self.scores[combo_map[combo]]
            self.res["test"]["COMBO"] = {**self.res["test"][combo_map[combo]], "same_as": combo_map[combo]}
            self.costs["COMBO"] = self.costs[combo_map[combo]]
        else:
            base_cfg = {"M1+M6": cm1, "P_M4+M6": cm4, "M1x+M6": cm1x}[combo]
            mkey = {"M1+M6": "F_M1", "M1x+M6": "F_M1x"}.get(combo)
            self.final_seeds("COMBO", bag_fn(base_cfg, None if mkey else pf, mkey), det=False)
            self.costs["COMBO"] = cost({**base_cfg, "sub": P.bag_sub}, P.bag)
        self.res["test"]["COMBO"]["choice"] = combo
        self.T("test_combo", t0)

        t0 = time.perf_counter()
        yfv = fv["ev_y"]
        w = self.w_best
        self.scores["M2_fuse"] = [sigmoid(w * logit(a) + (1 - w) * logit(b))
                                  for a, b in zip(self.scores["F_opt_p"], self.scores["G_t"])]
        a_ = [auc(yfv, s) for s in self.scores["M2_fuse"]]
        self.res["test"]["M2_fuse"] = {"auroc": float(np.mean(a_)), "per_seed": [float(x) for x in a_], "w": w}
        self.costs["M2_fuse"] = cost(cc)

        def student_fn(alpha, cfg_s, ex=None):
            def fn(s):
                part = np.random.default_rng(s).integers(0, P.folds, len(yf))
                q = np.empty(len(yf))
                for k in range(P.folds):
                    trm = part != k
                    q[~trm] = train_pred(P, Xf[trm], yf[trm], Xf[~trm], cc, bf, prior(yf[trm]), s)
                t = alpha * yf + (1 - alpha) * q
                X, yy = Xgf, t
                if ex is not None:
                    X, yy = np.vstack([X, ex[0]]), np.r_[t, ex[1]]
                return train_pred(P, X, yy, Xge, cfg_s, bf, prior(yy), s)
            return fn

        self.final_seeds("M2_student", student_fn(self.a_best, cg), det=False)
        self.res["test"]["M2_student"]["alpha"] = self.a_best
        self.costs["M2_student"] = cost(cc) * P.folds
        if self.has_ex:
            ax = self.res["val"]["student_x"]["alpha"]
            self.final_seeds("M2_student_x", student_fn(ax, self.cgx, fv["ex"]), det=False)
            self.res["test"]["M2_student_x"]["alpha"] = ax
            self.costs["M2_student_x"] = cost(cc) * P.folds
        self.T("test_m2", t0)
        self.test_dp(Xf, yf, Xe, Xgf, cg, cc)
        self.test_mismatch(Xf, yf, Xe, cc, pf, fv["fit_xh"].shape[1])

    def test_dp(self, Xf, yf, Xe, Xgf, cg, cc):
        """I6：硬 RR 对照 vs 后验去噪软标签，配置同优化版。"""
        P, bf = self.P, self.P.bins_fine
        prior = prior_fn(P)
        t0 = time.perf_counter()
        for eps in P.dp_eps:
            rho = 1 / (1 + math.exp(eps))
            noisy: Dict[int, np.ndarray] = {}

            def yn_of(s, eps=eps, rho=rho, noisy=noisy):
                if s not in noisy:
                    flip = np.random.default_rng(s * DP_SEED_MULT + int(eps)).random(len(yf)) < rho
                    noisy[s] = np.where(flip, 1 - yf, yf)
                return noisy[s]

            def hard(s, yn_of=yn_of):
                yn = yn_of(s)
                return train_pred(P, Xf, yn, Xe, cc, bf, prior(yn), s)

            def soft(s, yn_of=yn_of, rho=rho):
                yn = yn_of(s)
                half = np.random.default_rng(s + DP_SPLIT_OFFSET).random(len(yn)) < HALF
                q = np.empty(len(yn))
                for h in (True, False):
                    m = half == h
                    q[~m] = train_pred(P, Xgf[m], yn[m], Xgf[~m], cg, bf, prior(yn[m]), s)
                ys = label_posterior(q, yn, rho, P.eps_p)
                return train_pred(P, Xf, ys, Xe, cc, bf, prior(ys), s)

            tag = "e%d" % int(eps)
            self.final_seeds("DP_hard_" + tag, hard, det=False)
            self.final_seeds("DP_soft_" + tag, soft, det=False)
            self.costs["DP_hard_" + tag] = self.costs["DP_soft_" + tag] = cost(cc)
        self.T("test_dp", t0)

    def test_mismatch(self, Xf, yf, Xe, cc, pf, nh):
        """M5：训练集注入香港行错位，模拟置信分数（区分错配的 AUC = conf_auc）；测试集不注入。"""
        from scipy.stats import norm
        P, bf = self.P, self.P.bins_fine
        prior = prior_fn(P)
        t0 = time.perf_counter()
        mu = norm.ppf(P.conf_auc) / math.sqrt(2)
        for r in P.mismatch:
            cache: Dict[int, Tuple] = {}

            def corrupt(s, r=r, cache=cache):
                if s not in cache:
                    rng = np.random.default_rng(s * MISMATCH_SEED_MULT + int(r * PERCENT))
                    idx = rng.choice(len(yf), int(round(r * len(yf))), replace=False)
                    perm = rng.permutation(idx)
                    Xm = Xf.copy()
                    Xm[idx, -nh:] = Xf[perm, -nh:]
                    mis = np.zeros(len(yf), bool)
                    mis[idx] = perm != idx
                    conf = sigmoid(rng.normal(np.where(mis, -mu, mu), 1.0))
                    cache[s] = (Xm, conf)
                return cache[s]

            def drop(s, corrupt=corrupt):
                Xm, conf = corrupt(s)
                k = conf >= DROP_THRESHOLD
                return train_pred(P, Xm[k], yf[k], Xe, cc, bf, prior(yf[k]), s)

            tag = "r%d" % int(r * PERCENT)
            self.final_seeds("M5_none_" + tag, lambda s, c=corrupt: train_pred(P, c(s)[0], yf, Xe, cc, bf, pf, s),
                             det=False)
            self.final_seeds("M5_weight_" + tag, lambda s, c=corrupt: train_pred(P, c(s)[0], yf, Xe, cc, bf, pf, s,
                                                                                weight=c(s)[1]), det=False)
            self.final_seeds("M5_drop_" + tag, drop, det=False)
            for k in ("none", "weight", "drop"):
                self.costs["M5_%s_%s" % (k, tag)] = cost(cc)
        self.T("test_m5", t0)

    # ---------- 4) 对比
    def contrasts(self):
        fv = self.v["final"]
        t0 = time.perf_counter()
        A = two_level_bootstrap(self.P, fv["ev_y"], fv.get("ev_cl"), self.scores)
        has = set(self.scores)
        out = []
        for a, b, label in contrast_pairs(self.P):
            if a not in has or b not in has:
                continue
            d = A[a] - A[b]
            est = self.res["test"][a]["auroc"] - self.res["test"][b]["auroc"]
            p = float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))
            out.append({"a": a, "b": b, "label": label, "diff": est, "ci_low": float(np.percentile(d, PERCENTILE_LOW)),
                        "ci_high": float(np.percentile(d, PERCENTILE_HIGH)), "p": p})
        for c, ph in zip(out, holm([c["p"] for c in out])):
            c["p_holm"] = ph
        for k in self.scores:
            self.res["test"][k]["ci_low"] = float(np.percentile(A[k], PERCENTILE_LOW))
            self.res["test"][k]["ci_high"] = float(np.percentile(A[k], PERCENTILE_HIGH))
        self.res["contrasts"] = out
        self.T("bootstrap", t0)

    def run(self) -> Dict:
        self.tune_all()
        self.select()
        self.test_all()
        self.contrasts()
        self.res["costs"] = self.costs
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / (self.name + ".json")).write_text(json.dumps(self.res, ensure_ascii=False, indent=1, default=float),
                                                      encoding="utf-8")
        log("%s · 完成" % self.name)
        return self.res


def main(argv: Optional[List[str]] = None) -> int:
    """命令行：按配置顺序准备（缺失时）并筛选各数据集，已完成的跳过。

    python -m modules.m5_modeling.components.method_screen --config <yaml> --out <目录> [--ds 名字 ...]
    私有目录取环境变量（配置 private_root_env / raw_root_env）；Santander G3 另需 --g3-arrays。
    """
    import argparse
    import os

    import yaml

    from modules.m5_modeling.components import method_screen_data as MD

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ds", nargs="*")
    ap.add_argument("--g3-arrays")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    priv = Path(os.environ[cfg["private_root_env"]])
    raw = Path(os.environ[cfg["raw_root_env"]])
    out = Path(a.out)
    for name in a.ds or cfg["datasets"]:
        if (out / (name + ".json")).exists():
            log("%s 已完成，跳过" % name)
            continue
        if not (priv / (name + ".npz")).exists():
            MD.build(name, raw, priv, cfg["prep"], Path(a.g3_arrays) if a.g3_arrays else None)
            log("%s 数据已准备" % name)
        t = time.perf_counter()
        Screen(name, cfg, priv, out).run()
        log("%s 总用时 %.0fs" % (name, time.perf_counter() - t))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
