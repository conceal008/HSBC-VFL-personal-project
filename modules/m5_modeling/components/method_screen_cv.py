"""M5 · S5.M3：组合选择改为训练集交叉验证（DR-M5-013）。

S5.M1 用单一验证集选组合，分不开 0.005 量级的差距，且「并列取计算量最小」偏向 M1（DR-M5-012 复核）。本组件：
- 候选与 S5.M1 相同：P、P_M4、M1、P+M6、M1+M6、P_M4+M6，有非共同客户时加 M1x、M1x+M6；
- 每个候选的配置取自 S5.M1 结果 JSON（验证集上已选定），不再调参；
- 在 final 视图的训练行（= 全部训练集）上做交叉验证，测试集不参与选择；
- 折的划分按数据结构：行独立的任务分层 K 折；Santander G3 按客户分组 K 折；mock 四档按月份滚动外推
  （每折验证一段月份，训练只用更早的月份，非共同客户行同样只用更早的月份）；
- 规则 R1：交叉验证平均 AUROC 最高；规则 R2：与 R1 的逐折配对差不低于 −1 倍标准误的候选中取计算量最小；
- 选出的组合在测试集上评一次（6 个 seed），两层 bootstrap 给配对差区间，数据集内 Holm 校正。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules.m5_modeling.components import method_screen as MS

CANDS_BASE = ("P", "P_M4", "M1", "P+M6", "M1+M6", "P_M4+M6")
CANDS_EX = ("M1x", "M1x+M6")
CFG_KEY = {"P": "central", "P_M4": "m4", "M1": "m1", "M1x": "m1x"}
MARGIN_KIND = {"M1": "guest", "M1x": "guest_extra"}
BAG_TAG = "M6"
PERCENTILE_LOW = 2.5
PERCENTILE_HIGH = 97.5

Fold = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]      # (训练行, 验证行, 可用的非共同客户行掩码)


# ---------------------------------------------------------------- 候选
def candidates(has_ex: bool) -> List[str]:
    return list(CANDS_BASE) + (list(CANDS_EX) if has_ex else [])


def spec(res: Dict, name: str) -> Tuple[Dict, Optional[str], bool]:
    """候选名 → (联邦部分配置, 逐样本先验来源 guest / guest_extra / None, 是否 bagging)。"""
    base, _, tag = name.partition("+")
    return res["configs"][CFG_KEY[base]], MARGIN_KIND.get(base), tag == BAG_TAG


def cost_of(P: MS.Params, res: Dict, name: str) -> float:
    cfg, _, bag = spec(res, name)
    return MS.cost({**cfg, "sub": P.bag_sub}, P.bag) if bag else MS.cost(cfg)


def deterministic(res: Dict, name: str) -> bool:
    cfg, mkind, bag = spec(res, name)
    return not bag and mkind is None and MS.deterministic(cfg)


def predict(P: MS.Params, res: Dict, name: str, Xc_f, Xg_f, y_f, Xc_e, Xg_e, seed: int, ex=None,
            cache: Optional[Dict] = None) -> np.ndarray:
    """训练一个候选并对评估行打分；与 S5.M1 test_all 中同名方法的定义相同。cache 复用同一 seed 的交叉拟合 logit。"""
    cfg, mkind, bag = spec(res, name)
    base: Optional[float] = MS.prior_fn(P)(y_f)
    mf = me = None
    if mkind:
        key = (mkind, seed)
        if cache is not None and key in cache:
            mf, me = cache[key]
        else:
            gcfg = res["configs"]["guest" if mkind == "guest" else "guest_extra"]
            mf, me = MS.cross_fit(P, Xg_f, y_f, Xg_e, gcfg, P.bins_fine, seed,
                                  ex=ex if mkind == "guest_extra" else None)
            if cache is not None:
                cache[key] = (mf, me)
        base = None
    if bag:
        c = {**cfg, "sub": P.bag_sub}
        return np.mean([MS.train_pred(P, Xc_f, y_f, Xc_e, c, P.bins_fine, base, seed * MS.BAG_SEED_MULT + j,
                                      mf=mf, me=me) for j in range(P.bag)], axis=0)
    return MS.train_pred(P, Xc_f, y_f, Xc_e, cfg, P.bins_fine, base, seed, mf=mf, me=me)


# ---------------------------------------------------------------- 折
def folds_stratified(y: np.ndarray, k: int, seed: int) -> List[Fold]:
    rng = np.random.default_rng(seed)
    fid = np.empty(len(y), np.int64)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        fid[rng.permutation(idx)] = np.arange(len(idx)) % k
    return [(np.flatnonzero(fid != j), np.flatnonzero(fid == j), None) for j in range(k)]


def folds_grouped(groups: np.ndarray, k: int, seed: int) -> List[Fold]:
    u, inv = np.unique(groups, return_inverse=True)
    gf = np.random.default_rng(seed).permutation(len(u)) % k
    fid = gf[inv]
    return [(np.flatnonzero(fid != j), np.flatnonzero(fid == j), None) for j in range(k)]


def folds_time(month: np.ndarray, blocks: Sequence[Sequence[str]], ex_month: Optional[np.ndarray] = None) -> List[Fold]:
    """每个 block 是一段验证月份；训练只用早于该段首月的行（非共同客户行同理）。"""
    out: List[Fold] = []
    for b in blocks:
        start = min(b)
        va = np.isin(month, list(b))
        tr = month < start
        out.append((np.flatnonzero(tr), np.flatnonzero(va), None if ex_month is None else ex_month < start))
    return out


def build_groups(name: str, raw: Path, prep: Dict, g3_arrays: Optional[Path], fit_y: np.ndarray,
                 ex_y: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """从原始数据重建 final 视图训练行的客户号与月份。行顺序与 method_screen_data 相同，并用标签逐行核对。"""
    if name == "santander_g3":
        assert g3_arrays is not None, "santander_g3 需要 S5.G1 私有数组路径"
        d = np.load(g3_arrays)
        assert np.array_equal(np.r_[d["train__y"], d["inner_val__y"]], fit_y)
        return {"fit_cust": np.r_[d["train__cust"], d["inner_val__cust"]]}
    if not name.startswith("mock_"):
        return {}
    import pandas as pd
    tier = name.split("_", 1)[1]
    src = raw / "数据集" / "HSBC_mock数据"
    cols = ["ncodpers", "fecha_dato", "label_new_any"]
    g = pd.read_csv(src / (tier + "_guest.csv"), usecols=cols)
    test = (pd.to_datetime(g["fecha_dato"]) >= pd.Timestamp(prep["mock_test_start"])).to_numpy()
    month = np.asarray(g["fecha_dato"].astype(str), dtype=str)          # 定长字符串，np.load 无需 pickle
    assert np.array_equal(g["label_new_any"].to_numpy().astype(int)[~test], fit_y)
    out = {"fit_cust": g["ncodpers"].to_numpy()[~test], "fit_month": month[~test]}
    if ex_y is not None:
        months = sorted(set(month[~test]))
        b = pd.read_csv(src / "balanced_guest.csv", usecols=cols)
        b = b[~b["ncodpers"].isin(set(g["ncodpers"]))].reset_index(drop=True)
        exm = np.asarray(b["fecha_dato"].astype(str), dtype=str)
        keep = np.isin(exm, months)
        assert np.array_equal(b["label_new_any"].to_numpy().astype(int)[keep], ex_y)
        out["ex_month"] = exm[keep]
    return out


def fold_plan(name: str, scheme: str, sc: "MS.Screen", cvcfg: Dict, groups: Dict) -> List[Fold]:
    y = sc.v["final"]["fit_y"]
    if scheme == "row":
        return folds_stratified(y, int(cvcfg["k_row"]), int(cvcfg["fold_seed"]))
    if scheme == "group":
        return folds_grouped(groups["fit_cust"], int(cvcfg["k_group"]), int(cvcfg["fold_seed"]))
    if scheme == "time":
        return folds_time(groups["fit_month"], cvcfg["time_blocks"], groups.get("ex_month"))
    raise ValueError("未知的折方案：%s（%s）" % (scheme, name))


# ---------------------------------------------------------------- 交叉验证与选择
def cv_scores(P: MS.Params, res: Dict, sc: "MS.Screen", plan: List[Fold], cands: List[str]) -> Dict[str, List[float]]:
    fv = sc.v["final"]
    Xc, Xg, y, ex = fv["fit_xc"], fv["fit_xg"], fv["fit_y"], fv["ex"]
    out: Dict[str, List[float]] = {c: [] for c in cands}
    for tr, va, exm in plan:
        cache: Dict = {}
        ex_k = None if ex is None else ((ex[0], ex[1]) if exm is None else (ex[0][exm], ex[1][exm]))
        for c in cands:
            s = predict(P, res, c, Xc[tr], Xg[tr], y[tr], Xc[va], Xg[va], P.tune_seed, ex=ex_k, cache=cache)
            out[c].append(MS.auc(y[va], s))
    return out


def choose(cv: Dict[str, List[float]], costs: Dict[str, float], se_mult: float) -> Dict:
    """R1：平均 AUROC 最高；R2：配对差 ≥ −se_mult × 标准误的候选中计算量最小（并列时取 R1 顺序靠前者）。"""
    mean = {c: float(np.mean(v)) for c, v in cv.items()}
    r1 = max(cv, key=lambda c: mean[c])
    k = len(cv[r1])
    paired = {}
    for c in cv:
        d = np.asarray(cv[c]) - np.asarray(cv[r1])
        se = float(np.std(d, ddof=1) / np.sqrt(k)) if k > 1 else 0.0
        paired[c] = {"diff": float(np.mean(d)), "se": se, "eligible": bool(np.mean(d) >= -se_mult * se)}
    r2 = min([c for c in cv if paired[c]["eligible"]], key=lambda c: (costs[c], -mean[c]))
    return {"r1": r1, "r2": r2, "mean": mean, "paired_vs_r1": paired}


# ---------------------------------------------------------------- 测试集
def paired(A: Dict[str, np.ndarray], a: str, b: str, est: float) -> Dict:
    d = A[a] - A[b]
    return {"diff": est, "ci_low": float(np.percentile(d, PERCENTILE_LOW)),
            "ci_high": float(np.percentile(d, PERCENTILE_HIGH)),
            "p": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))}


def test_eval(P: MS.Params, res: Dict, sc: "MS.Screen", picks: Dict[str, str]) -> Dict:
    """picks：{"CV_R1": 候选名, "CV_R2": 候选名}；另算优化版 P 与内地单边 G_t。返回测试 AUROC 与配对差。"""
    fv = sc.v["final"]
    Xc, Xg, y, Xce, Xge, ye = fv["fit_xc"], fv["fit_xg"], fv["fit_y"], fv["ev_xc"], fv["ev_xg"], fv["ev_y"]
    pf = MS.prior_fn(P)(y)
    scores: Dict[str, List[np.ndarray]] = {}
    cache: Dict = {}
    done: Dict[str, List[np.ndarray]] = {}

    def run(name: str) -> List[np.ndarray]:
        if name not in done:
            if deterministic(res, name):
                s0 = predict(P, res, name, Xc, Xg, y, Xce, Xge, P.seeds[0], ex=fv["ex"], cache=cache)
                done[name] = [s0] * len(P.seeds)
            else:
                done[name] = [predict(P, res, name, Xc, Xg, y, Xce, Xge, sd, ex=fv["ex"], cache=cache)
                              for sd in P.seeds]
        return done[name]

    scores["P"] = run("P")
    for key, name in picks.items():
        scores[key] = run(name)
    cg = res["configs"]["guest"]
    if MS.deterministic(cg):
        g0 = MS.train_pred(P, Xg, y, Xge, cg, P.bins_fine, pf, P.seeds[0])
        scores["G_t"] = [g0] * len(P.seeds)
    else:
        scores["G_t"] = [MS.train_pred(P, Xg, y, Xge, cg, P.bins_fine, pf, sd) for sd in P.seeds]
    A = MS.two_level_bootstrap(P, ye, fv.get("ev_cl"), scores)
    au = {k: [MS.auc(ye, s) for s in v] for k, v in scores.items()}
    mean = {k: float(np.mean(v)) for k, v in au.items()}
    test = {k: {"auroc": mean[k], "per_seed": [float(x) for x in v]} for k, v in au.items()}
    pairs = [(k, "P") for k in picks] + [(k, "G_t") for k in picks]
    con = [{"a": a, "b": b, **paired(A, a, b, mean[a] - mean[b])} for a, b in pairs]
    for c, ph in zip(con, MS.holm([c["p"] for c in con])):
        c["p_holm"] = ph
    return {"test": test, "contrasts": con}


# ---------------------------------------------------------------- 主流程
def run(name: str, cfg: Dict, m3: Dict, priv: Path, run_dir: Path, groups: Dict) -> Dict:
    P = MS.Params(cfg)
    res = json.loads((run_dir / (name + ".json")).read_text(encoding="utf-8"))
    sc = MS.Screen(name, cfg, priv, run_dir / "unused")
    scheme = m3["fold_scheme"].get(name, "row")
    cands = candidates(sc.has_ex)
    t0 = time.perf_counter()
    plan = fold_plan(name, scheme, sc, m3["cv"], groups)
    cv = cv_scores(P, res, sc, plan, cands)
    costs = {c: cost_of(P, res, c) for c in cands}
    ch = choose(cv, costs, float(m3["cv"]["se_mult"]))
    t_cv = round(time.perf_counter() - t0, 1)
    MS.log("%s · 交叉验证（%s，%d 折）%s → R1 %s，R2 %s，用时 %ss" % (
        name, scheme, len(plan), {c: round(m, MS.LOG_DECIMALS) for c, m in ch["mean"].items()}, ch["r1"], ch["r2"], t_cv))
    t0 = time.perf_counter()
    ev = test_eval(P, res, sc, {"CV_R1": ch["r1"], "CV_R2": ch["r2"]})
    t_test = round(time.perf_counter() - t0, 1)
    s5m1 = next((c for c in res["contrasts"] if c["a"] == "COMBO" and c["b"] == "F_opt_p"), None)
    return {"dataset": name, "scheme": scheme, "folds": [{"train": int(len(tr)), "val": int(len(va)),
                                                          "val_pos": int(sc.v["final"]["fit_y"][va].sum())}
                                                         for tr, va, _ in plan],
            "cv": cv, "costs": costs, "choice": ch, **ev,
            "s5m1": {"choice": res["val"]["combo_choice"], "combo_minus_p": s5m1},
            "timing": {"cv": t_cv, "test": t_test}}


def main(argv: Optional[List[str]] = None) -> int:
    import os

    import yaml
    ap = argparse.ArgumentParser(description="S5.M3 交叉验证选组合")
    ap.add_argument("--config", default="modules/m5_modeling/configs/s5_m3_cv_combo.yaml")
    ap.add_argument("--datasets", nargs="*")
    a = ap.parse_args(argv)
    m3 = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(m3["base_config"]).read_text(encoding="utf-8"))
    priv, run_dir = Path(os.environ[cfg["private_root_env"]]), Path(os.environ[cfg["run_dir_env"]])
    out = Path(os.environ[m3["cv_dir_env"]])
    out.mkdir(parents=True, exist_ok=True)
    for name in a.datasets or m3["datasets"]:
        if (out / (name + ".json")).exists():
            continue
        gpath = out / (name + "__groups.npz")
        if m3["fold_scheme"].get(name, "row") != "row" and not gpath.exists():
            d = np.load(priv / (name + ".npz"))
            g3 = os.environ.get(m3["g3_arrays_env"])
            gr: Dict[str, Any] = build_groups(name, Path(os.environ[cfg["raw_root_env"]]), cfg["prep"],
                                              Path(g3) if g3 else None, d["final__fit_y"],
                                              d["final__ex_y"] if "final__ex_y" in d.files else None)
            np.savez(gpath, **gr)
        groups = dict(np.load(gpath)) if gpath.exists() else {}
        r = run(name, cfg, m3, priv, run_dir, groups)
        (out / (name + ".json")).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
