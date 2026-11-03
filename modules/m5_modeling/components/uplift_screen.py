"""M5 · S5.M6：M8 增量（uplift）学习器——S / T / X / R 与响应模型，只用内地特征 vs 联邦特征（DR-M5-014 · D-4）。

- 数据与两方沿用 M1 已冻结的字段归属：Hillstrom（主动方 recency / history / mens / womens，被动方 zip_code / newbie / channel）、
  Criteo Uplift v2.1（主动方偶数列 f0–f10，被动方奇数列）。处理变量与结果只由主动方持有。
- 学习器（基模型 XGBoost，Künzel 等 2019；Nie & Wager 2021）：
  - S：结果模型以处理为额外特征，两种处理下的预测之差；
  - T：处理组、对照组各一个结果模型；
  - X：T 的两个模型 + 插补效应回归，按处理占比 e 加权：τ = e·τ0 + (1 − e)·τ1；
  - R：交叉拟合的结果均值 m(x) + 已知处理占比 e，伪结果 (y − m) / (t − e)、权重 (t − e)² 的回归；
  - 对照：响应模型，只用处理组拟合 P(结果 | x)，按「谁会买」排序。
- 指标：Qini 系数（Qini 曲线与随机线之间的面积，横轴为人群比例、纵轴为每位客户的增量结果）；前 30% 人群的增量（处理组与对照组结果率之差）。
- 选择只看 tune 视图（训练拟合、验证评估）；final 视图（训练 + 验证拟合）在测试集上评一次。
- 基模型不做行 / 列采样，交叉拟合折固定，拟合是确定的；bootstrap 只有测试行一层。

只在调用时导入 pandas / xgboost。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules.m5_modeling.components import method_screen as MS

LEARNERS = ("S", "T", "X", "R")
RESPONSE = "response"
FEATURE_SETS = ("main", "fed")           # 只用内地（主动方）特征 / 两方全部特征
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2
HILLSTROM_CONTROL = "No E-Mail"


# ---------------------------------------------------------------- 数据
def strat_split(strata: np.ndarray, fracs: Sequence[float], seed: int) -> np.ndarray:
    """每个层内随机排列后按比例切成训练 / 验证 / 测试；返回 0 / 1 / 2。"""
    rng = np.random.default_rng(seed)
    out = np.full(len(strata), SPLIT_TEST, np.int8)
    for s in np.unique(strata):
        idx = rng.permutation(np.flatnonzero(strata == s))
        n_tr = int(round(len(idx) * fracs[0]))
        n_va = int(round(len(idx) * fracs[1]))
        out[idx[:n_tr]] = SPLIT_TRAIN
        out[idx[n_tr:n_tr + n_va]] = SPLIT_VAL
    return out


def one_hot(frame, cols: Sequence[str]):
    import pandas as pd
    return pd.get_dummies(frame[list(cols)].astype(str), prefix=list(cols), dtype=float)


def load_hillstrom(path: Path, spec: Dict) -> Dict:
    import pandas as pd
    df = pd.read_csv(path)
    parts = {}
    for side in ("guest", "host"):
        s = spec[side]
        num = df[[c for c in s["features"] if c not in s.get("categorical", [])]].astype(float)
        cat = one_hot(df, s.get("categorical", [])) if s.get("categorical") else None
        parts[side] = num if cat is None else pd.concat([num, cat], axis=1)
    t = (df[spec["treatment"]] != HILLSTROM_CONTROL).to_numpy().astype(int)
    strata = df[spec["treatment"]].astype(str) + "|" + df[spec["strata_outcome"]].astype(str)
    split = strat_split(strata.to_numpy(), spec["fracs"], int(spec["split_seed"]))
    return {"xg": parts["guest"].to_numpy(float), "xh": parts["host"].to_numpy(float),
            "g_cols": list(parts["guest"].columns), "h_cols": list(parts["host"].columns), "t": t,
            "y": {o: df[o].to_numpy().astype(int) for o in spec["outcomes"]}, "split": split}


def load_criteo(path: Path, spec: Dict) -> Dict:
    import pandas as pd
    rng = np.random.default_rng(int(spec["sample_seed"]))
    keep = []
    cols = list(spec["guest"]["features"]) + list(spec["host"]["features"]) + [spec["treatment"]] + list(spec["outcomes"])
    for chunk in pd.read_csv(path, usecols=cols, chunksize=int(spec["read_chunk"])):
        keep.append(chunk[rng.random(len(chunk)) < float(spec["sample_frac"])])
    df = pd.concat(keep, ignore_index=True)
    t = df[spec["treatment"]].to_numpy().astype(int)
    strata = df[spec["treatment"]].astype(str) + "|" + df[spec["strata_outcome"]].astype(str)
    split = strat_split(strata.to_numpy(), spec["fracs"], int(spec["split_seed"]))
    return {"xg": df[list(spec["guest"]["features"])].to_numpy(float),
            "xh": df[list(spec["host"]["features"])].to_numpy(float),
            "g_cols": list(spec["guest"]["features"]), "h_cols": list(spec["host"]["features"]), "t": t,
            "y": {o: df[o].to_numpy().astype(int) for o in spec["outcomes"]}, "split": split}


def features(data: Dict, fs: str) -> np.ndarray:
    return data["xg"] if fs == "main" else np.hstack([data["xg"], data["xh"]])


# ---------------------------------------------------------------- 基模型
class Base:
    """固定配置的 XGBoost 分类 / 回归器（无行 / 列采样，拟合确定）。"""

    def __init__(self, cfg: Dict, bins: int, nthread: int, seed: int):
        self.cfg, self.bins, self.nthread, self.seed = cfg, bins, nthread, seed

    def params(self, objective: str) -> Dict:
        return {"objective": objective, "tree_method": "hist", "max_bin": self.bins, "eta": self.cfg["lr"],
                "max_depth": self.cfg["depth"], "reg_lambda": self.cfg["lam"], "subsample": 1.0,
                "colsample_bytree": 1.0, "seed": self.seed, "nthread": self.nthread, "verbosity": 0}

    def fit_predict(self, X, y, Xe_list: Sequence[np.ndarray], objective: str = "binary:logistic", weight=None
                    ) -> List[np.ndarray]:
        import xgboost as xgb
        d = xgb.QuantileDMatrix(X, label=y, weight=weight, max_bin=self.bins, nthread=self.nthread)
        bst = xgb.train(self.params(objective), d, int(self.cfg["rounds"]))
        return [bst.predict(xgb.QuantileDMatrix(Xe, max_bin=self.bins, ref=d, nthread=self.nthread)) for Xe in Xe_list]


def logloss(y, p) -> float:
    p = np.clip(p, MS.PROB_CLIP, 1 - MS.PROB_CLIP)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def pick_config(grid: List[Dict], make: Callable[[Dict], Base], Xf, tf, yf, Xe, te, ye) -> Tuple[Dict, List[Dict]]:
    """按 S 学习器结果模型（X + 处理）在验证集上的对数损失选配置。"""
    rows = []
    for cfg in grid:
        (p,) = make(cfg).fit_predict(np.column_stack([Xf, tf]), yf, [np.column_stack([Xe, te])])
        rows.append({**cfg, "val_logloss": logloss(ye, p)})
    rows.sort(key=lambda r: r["val_logloss"])
    best = {k: v for k, v in rows[0].items() if k != "val_logloss"}
    return best, rows


# ---------------------------------------------------------------- 学习器
def s_learner(b: Base, Xf, tf, yf, Xe) -> np.ndarray:
    one, zero = np.column_stack([Xe, np.ones(len(Xe))]), np.column_stack([Xe, np.zeros(len(Xe))])
    p1, p0 = b.fit_predict(np.column_stack([Xf, tf]), yf, [one, zero])
    return p1 - p0


def t_models(b: Base, Xf, tf, yf, targets: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    m1 = b.fit_predict(Xf[tf == 1], yf[tf == 1], targets)
    m0 = b.fit_predict(Xf[tf == 0], yf[tf == 0], targets)
    return m1, m0


def t_learner(b: Base, Xf, tf, yf, Xe) -> np.ndarray:
    (m1,), (m0,) = t_models(b, Xf, tf, yf, [Xe])
    return m1 - m0


def x_learner(b: Base, Xf, tf, yf, Xe, e: float) -> np.ndarray:
    X1, X0 = Xf[tf == 1], Xf[tf == 0]
    (m1_on0,), (m0_on1,) = b.fit_predict(X1, yf[tf == 1], [X0]), b.fit_predict(X0, yf[tf == 0], [X1])
    d1, d0 = yf[tf == 1] - m0_on1, m1_on0 - yf[tf == 0]
    (tau1,) = b.fit_predict(X1, d1, [Xe], "reg:squarederror")
    (tau0,) = b.fit_predict(X0, d0, [Xe], "reg:squarederror")
    return e * tau0 + (1 - e) * tau1


def r_learner(b: Base, Xf, tf, yf, Xe, e: float, folds: int, fold_seed: int) -> np.ndarray:
    part = np.random.default_rng(fold_seed).integers(0, folds, len(yf))
    m = np.empty(len(yf))
    for k in range(folds):
        (m[part == k],) = b.fit_predict(Xf[part != k], yf[part != k], [Xf[part == k]])
    r = tf - e
    (tau,) = b.fit_predict(Xf, (yf - m) / r, [Xe], "reg:squarederror", weight=r ** 2)
    return tau


def response(b: Base, Xf, tf, yf, Xe) -> np.ndarray:
    (p,) = b.fit_predict(Xf[tf == 1], yf[tf == 1], [Xe])
    return p


def scores_for(b: Base, Xf, tf, yf, Xe, folds: int, fold_seed: int) -> Dict[str, np.ndarray]:
    e = float(np.mean(tf))
    return {"S": s_learner(b, Xf, tf, yf, Xe), "T": t_learner(b, Xf, tf, yf, Xe),
            "X": x_learner(b, Xf, tf, yf, Xe, e), "R": r_learner(b, Xf, tf, yf, Xe, e, folds, fold_seed),
            RESPONSE: response(b, Xf, tf, yf, Xe)}


# ---------------------------------------------------------------- 指标
class Ranked:
    """按分数降序排好的测试行（并列合为一组），供带权 Qini 与前 q 人群增量反复计算。"""

    def __init__(self, score: np.ndarray, t: np.ndarray, y: np.ndarray):
        o = np.argsort(-score, kind="mergesort")
        s = score[o]
        self.order = o
        self.group = np.r_[0, np.cumsum(s[1:] != s[:-1])]
        self.G = int(self.group[-1]) + 1
        self.t, self.y = t[o].astype(float), y[o].astype(float)

    def curve(self, w: Optional[np.ndarray] = None):
        w = np.ones(len(self.t)) if w is None else w[self.order]
        agg = [np.bincount(self.group, weights=v, minlength=self.G)
               for v in (w * self.t, w * (1 - self.t), w * self.t * self.y, w * (1 - self.t) * self.y)]
        nt, nc, yt, yc = (np.r_[0.0, np.cumsum(a)] for a in agg)
        ratio = np.divide(nt, nc, out=np.zeros_like(nt), where=nc > 0)
        n = nt[-1] + nc[-1]
        return (nt + nc) / n, (yt - yc * ratio) / n, (nt, nc, yt, yc)

    def qini(self, w: Optional[np.ndarray] = None) -> float:
        x, q, _ = self.curve(w)
        area = float(np.sum((x[1:] - x[:-1]) * (q[1:] + q[:-1])) * MS.HALF)
        return area - MS.HALF * float(q[-1])

    def uplift_at(self, frac: float, w: Optional[np.ndarray] = None) -> float:
        x, _, (nt, nc, yt, yc) = self.curve(w)
        j = int(np.searchsorted(x, frac, side="left"))
        j = min(max(j, 1), len(x) - 1)
        if nt[j] == 0 or nc[j] == 0:
            return 0.0
        return float(yt[j] / nt[j] - yc[j] / nc[j])


def bootstrap(R: Dict[str, Ranked], n_rows: int, n_boot: int, seed: int, frac: float) -> Dict[str, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out = {k: {"qini": np.empty(n_boot), "uplift": np.empty(n_boot)} for k in R}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n_rows, n_rows), minlength=n_rows).astype(float)
        for k, r in R.items():
            out[k]["qini"][b], out[k]["uplift"][b] = r.qini(w), r.uplift_at(frac, w)
    return out


def paired(A: np.ndarray, B: np.ndarray, est: float) -> Dict:
    d = A - B
    return {"diff": est, "ci_low": float(np.percentile(d, MS.PERCENTILE_LOW)),
            "ci_high": float(np.percentile(d, MS.PERCENTILE_HIGH)),
            "p": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))}


def interval(A: np.ndarray, est: float) -> Dict:
    return {"est": est, "ci_low": float(np.percentile(A, MS.PERCENTILE_LOW)),
            "ci_high": float(np.percentile(A, MS.PERCENTILE_HIGH))}


# ---------------------------------------------------------------- 一个（数据集，结果）
def run_outcome(data: Dict, outcome: str, m6: Dict) -> Dict:
    sp, t, y = data["split"], data["t"], data["y"][outcome]
    tr, va, te = sp == SPLIT_TRAIN, sp == SPLIT_VAL, sp == SPLIT_TEST
    fin = tr | va
    bins, nthread, seed = int(m6["bins"]), int(m6["nthread"]), int(m6["seed"])
    folds, fseed, frac = int(m6["r_folds"]), int(m6["fold_seed"]), float(m6["top_frac"])

    def make(cfg):
        return Base(cfg, bins, nthread, seed)

    grid = [{"lr": float(m6["grid"]["lr"]), "depth": int(d), "lam": float(m6["grid"]["lam"]), "rounds": int(r)}
            for d in m6["grid"]["depth"] for r in m6["grid"]["rounds"]]
    res: Dict = {"outcome": outcome, "configs": {}, "val": {}, "choice": {}, "test": {}, "timing": {}}
    test_scores: Dict[str, np.ndarray] = {}
    for fs in FEATURE_SETS:
        X = features(data, fs)
        t0 = time.perf_counter()
        cfg, rows = pick_config(grid, make, X[tr], t[tr], y[tr], X[va], t[va], y[va])
        res["configs"][fs] = {"chosen": cfg, "grid": rows}
        b = make(cfg)
        val = scores_for(b, X[tr], t[tr], y[tr], X[va], folds, fseed)
        res["val"][fs] = {k: Ranked(s, t[va], y[va]).qini() for k, s in val.items()}
        pick = max(LEARNERS, key=lambda k: (res["val"][fs][k], -LEARNERS.index(k)))
        res["choice"][fs] = pick
        for k, s in scores_for(b, X[fin], t[fin], y[fin], X[te], folds, fseed).items():
            test_scores["%s:%s" % (fs, k)] = s
        res["timing"][fs] = round(time.perf_counter() - t0, 1)
        MS.log("%s · %s · %s：配置 %s，验证 Qini %s → %s，用时 %ss" % (
            data["name"], outcome, fs, cfg, {k: round(v, MS.LOG_DECIMALS + 1) for k, v in res["val"][fs].items()}, pick,
            res["timing"][fs]))
    R = {k: Ranked(s, t[te], y[te]) for k, s in test_scores.items()}
    B = bootstrap(R, int(te.sum()), int(m6["n_boot"]), int(m6["boot_seed"]), frac)
    point = {k: {"qini": r.qini(), "uplift": r.uplift_at(frac)} for k, r in R.items()}
    res["test"] = {k: {"qini": interval(B[k]["qini"], point[k]["qini"]),
                       "uplift_top": interval(B[k]["uplift"], point[k]["uplift"])} for k in R}
    sel = {fs: "%s:%s" % (fs, res["choice"][fs]) for fs in FEATURE_SETS}
    pairs = [(sel["fed"], sel["main"], "联邦选定 − 内地选定"),
             (sel["fed"], "fed:" + RESPONSE, "联邦选定 − 联邦响应模型"),
             (sel["main"], "main:" + RESPONSE, "内地选定 − 内地响应模型")]
    con = [{"a": a, "b": b_, "label": lab, **paired(B[a]["qini"], B[b_]["qini"], point[a]["qini"] - point[b_]["qini"])}
           for a, b_, lab in pairs]
    for c, ph in zip(con, MS.holm([c["p"] for c in con])):
        c["p_holm"] = ph
    res["contrasts"] = con
    res["counts"] = {name: {"rows": int(m.sum()), "treated": int(t[m].sum()), "events_treated": int(y[m & (t == 1)].sum()),
                            "events_control": int(y[m & (t == 0)].sum())}
                     for name, m in (("train", tr), ("val", va), ("test", te))}
    return res


def run_dataset(name: str, data: Dict, m6: Dict, outcomes: Sequence[str]) -> Dict:
    data = {**data, "name": name}
    return {"dataset": name, "guest_cols": data["g_cols"], "host_cols": data["h_cols"],
            "outcomes": {o: run_outcome(data, o, m6) for o in outcomes}}


LOADERS = {"hillstrom": load_hillstrom, "criteo": load_criteo}


def main(argv: Optional[List[str]] = None) -> int:
    """python -m modules.m5_modeling.components.uplift_screen --config <yaml> [--datasets hillstrom ...]

    环境变量：本步配置的 raw_root_env（原始数据根）与 run_dir_env（每个数据集一个只含聚合指标的 JSON）。已完成的跳过。
    """
    import argparse
    import os

    import yaml
    ap = argparse.ArgumentParser(description="S5.M6 增量学习器")
    ap.add_argument("--config", default="modules/m5_modeling/configs/s5_m6_uplift.yaml")
    ap.add_argument("--datasets", nargs="*")
    a = ap.parse_args(argv)
    m6 = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    raw, out = Path(os.environ[m6["raw_root_env"]]), Path(os.environ[m6["run_dir_env"]])
    out.mkdir(parents=True, exist_ok=True)
    for name in a.datasets or list(m6["datasets"]):
        if (out / (name + ".json")).exists():
            continue
        spec = m6["datasets"][name]
        t0 = time.perf_counter()
        data = LOADERS[name](raw / spec["path"], spec)
        MS.log("%s 读取完成：%d 行，处理占比 %.3f" % (name, len(data["t"]), float(np.mean(data["t"]))))
        r = run_dataset(name, data, m6, spec["outcomes"])
        r["timing_total"] = round(time.perf_counter() - t0, 1)
        (out / (name + ".json")).write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
