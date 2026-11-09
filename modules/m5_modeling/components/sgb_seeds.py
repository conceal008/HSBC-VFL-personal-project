"""M5 · S5.M12：列采样配置的 SGB 训练随机性（DR-M5-017 · D-4 ~ D-5、D-8）。

- 任务：优化版联邦配置列采样 < 1 的 9 个任务（按配置判定）；已有 seed 11 的 SGB 打分（S5.M9 / S5.M2 存档），
  再拟合方法筛选统一的其余 5 个 seed（Params.seeds[1:]），其余设置不变；
- 确定性核对：原版联邦（列、行采样都为 1）在指定任务上用另一个 seed 再拟合一次，测试打分与 seed 11 比较；
  逐位相同则原版联邦取单次拟合，不同则原版联邦也拟合 5 个新 seed（D-4 预先写定，D-8 执行）；
- 先拟合全部任务的优化版联邦新 seed，再拟合原版联邦新 seed；每个（任务，方法）第一个新 seed 前抽查前 2 棵原版与明文逐位相同
  并外推（sgb_i2i3_confirm.probe），超过上限的任务不做；每个 seed 分别存档，分段运行（sgb_i2i3_confirm.Segment）；
- 对比：I2 + I3（优化版联邦 − 原版联邦）用两层 bootstrap（测试客户簇 × seed，method_screen.two_level_bootstrap），
  另报告两个方法各自的 seed 标准差、seed 11 单次的同口径结论与两者是否一致。

需要 SecretFlow（secretflow-lite 1.13.0b0）；导入本模块不依赖它，调用时才导入。打分在模拟中直接取回，只用于离线评估。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from modules.m5_modeling.components import method_screen as MS
from modules.m5_modeling.components import method_screen_sgb as SG
from modules.m5_modeling.components import sgb_cost_screen as SC
from modules.m5_modeling.components import sgb_i2i3_confirm as IC

OPT, DEF = "F_opt_p", "F_def"


def seed_path(out: Path, name: str, method: str, seed: int, ext: str) -> Path:
    return out / ("%s__%s__s%d.%s" % (name, method, seed, ext))


def probe_path(out: Path, name: str, method: str) -> Path:
    return out / ("%s__%s__probe.json" % (name, method))


def stored(name: str, method: str, dirs: List[Path]) -> np.ndarray:
    """seed 11 的已存档打分：按顺序在 S5.M9、S5.M2 存档目录里找。"""
    for d in dirs:
        p = d / ("%s__%s.npy" % (name, method))
        if p.exists():
            return np.load(p).astype(float)
    raise FileNotFoundError("%s__%s.npy" % (name, method))


def fit_seed(P: MS.Params, args: tuple, seed: int) -> Dict[str, Any]:
    """一次明文直方图版本的 SGB，seed 为给定值。"""
    xg, xh, y, eg, eh, cfg, bins, base = args
    with SC.plaintext_label_holder(True):
        pred, secs = SG.run_sgb(P, xg, xh, y, eg, eh, cfg, bins, base, seed=seed)
    return {"pred": np.asarray(pred, np.float32), "seed": int(seed), "fit_seconds": round(float(secs), 1)}


def determinism(P: MS.Params, sc: "MS.Screen", res: Dict, ref: np.ndarray, seed: int) -> Dict[str, Any]:
    """D-4：原版联邦换一个 seed 再拟合，与 seed 11 的测试打分逐位比较。"""
    r = fit_seed(P, SC.task_args(P, sc, res, DEF), seed)
    a, b = np.asarray(ref, np.float32), r["pred"]
    return {"task": sc.name, "method": DEF, "seed": seed, "identical": bool(np.array_equal(a, b)),
            "max_abs_diff": float(np.max(np.abs(a.astype(float) - b.astype(float)))), "n_rows_differ": int(np.sum(a != b)),
            "fit_seconds": r["fit_seconds"]}


def methods_for(det: Dict) -> List[str]:
    """D-4 / D-8：优化版联邦总是拟合新 seed；原版联邦确定性核对未通过时也拟合新 seed。"""
    return [OPT] if det["identical"] else [OPT, DEF]


def status(out: Path, name: str, method: str, seeds: List[int]) -> Optional[str]:
    """不读数据即可判定的状态："skipped"（抽查判定不做）、"done"（全部新 seed 已存档）或 None。"""
    pj = probe_path(out, name, method)
    if pj.exists() and json.loads(pj.read_text(encoding="utf-8"))["verdict"] != "ok":
        return "skipped"
    if all(seed_path(out, name, method, sd, "json").exists() for sd in seeds):
        return "done"
    return None


def run_task(P: MS.Params, sc: "MS.Screen", res: Dict, out: Path, cap: float, seg: IC.Segment, min_probe: float,
             seeds: List[int], method: str) -> str:
    """一个（任务，方法）：抽查 → 逐个 seed 拟合并存档。返回 "done"、"skipped" 或 "paused"。"""
    name = sc.name
    pj = probe_path(out, name, method)
    if not pj.exists():
        if not seg.allows(min_probe):
            return "paused"
        pr = IC.probe(P, sc, res, method)
        pr["verdict"] = (IC.NOT_IDENTICAL if not pr["identical_first_trees"]
                         else IC.OVER_CAP if pr["projected_fit_seconds"] > cap else "ok")
        SC.write_json(pj, pr)
        MS.log("%s · %s 抽查：前 2 棵逐位一致 %s，外推 %.0f 秒（%s）" % (
            name, method, pr["identical_first_trees"], pr["projected_fit_seconds"], pr["verdict"]))
    pr = json.loads(pj.read_text(encoding="utf-8"))
    if pr["verdict"] != "ok":
        return "skipped"
    args = SC.task_args(P, sc, res, method)
    for sd in seeds:
        if seed_path(out, name, method, sd, "json").exists():
            continue
        if not seg.allows(pr["projected_fit_seconds"]):
            return "paused"
        r = fit_seed(P, args, sd)
        np.save(seed_path(out, name, method, sd, "npy"), r["pred"])
        seed_path(out, name, method, sd, "json").write_text(json.dumps({"seed": sd, "fit_seconds": r["fit_seconds"]}),
                                                            encoding="utf-8")
        MS.log("%s · %s seed %d 拟合完成，%.0f 秒" % (name, method, sd, r["fit_seconds"]))
    return "done"


def _sd(aucs: List[float]) -> float:
    return float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0


def contrast(P: MS.Params, y, cl, opts: List[np.ndarray], defs: List[np.ndarray]) -> Dict[str, Any]:
    """优化版联邦（opts 个 seed）− 原版联邦（defs 个 seed）：两层 bootstrap 的配对差、区间与 p；点估计为两边 seed 平均 AUROC 之差。"""
    A = MS.two_level_bootstrap(P, y, cl, {OPT: opts, DEF: defs})
    oa, da = [MS.auc(y, o) for o in opts], [MS.auc(y, d) for d in defs]
    d = A[OPT] - A[DEF]
    return {"diff": float(np.mean(oa) - np.mean(da)), "ci_low": float(np.percentile(d, SG.PERCENTILE_LOW)),
            "ci_high": float(np.percentile(d, SG.PERCENTILE_HIGH)),
            "p": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))), "opt_aucs": oa, "def_aucs": da,
            "opt_seed_sd": _sd(oa), "def_seed_sd": _sd(da)}


def scores(name: str, method: str, out: Path, dirs: List[Path], seeds: List[int]) -> List[np.ndarray]:
    """seed 11 的存档打分在前，其后是本步的新 seed。"""
    return [stored(name, method, dirs)] + [np.load(seed_path(out, name, method, sd, "npy")).astype(float) for sd in seeds]


def evaluate_task(P: MS.Params, sc: "MS.Screen", res: Dict, out: Path, dirs: List[Path], seeds: List[int],
                  methods: List[str]) -> Dict[str, Any]:
    """多 seed 与 seed 11 单次两种口径的 I2 + I3，及 S5.M1 / S5.M5 代理（6 seed）对照。"""
    fv = sc.v["final"]
    y, cl = fv["ev_y"], fv.get("ev_cl")
    sc_ = {m: scores(sc.name, m, out, dirs, seeds if m in methods else []) for m in (OPT, DEF)}
    six, one = contrast(P, y, cl, sc_[OPT], sc_[DEF]), contrast(P, y, cl, sc_[OPT][:1], sc_[DEF][:1])
    px = next((c for c in res["contrasts"] if c["a"] == OPT and c["b"] == DEF), None)
    fits = {m: round(sum(json.loads(seed_path(out, sc.name, m, sd, "json").read_text(encoding="utf-8"))["fit_seconds"]
                         for sd in seeds), 1) for m in methods}
    return {"task": sc.name, "multi_seed_methods": methods, "six_seed": six, "seed11": one, "proxy_6seed": px,
            "holds_six": six["ci_low"] > 0, "holds_seed11": one["ci_low"] > 0, "fit_seconds_new_seeds": fits}


def summarize(evals: Dict[str, Dict], det: Optional[Dict], skipped: Dict[str, str]) -> Dict[str, Any]:
    rows = [{"task": n, "diff_six": e["six_seed"]["diff"], "ci_low_six": e["six_seed"]["ci_low"],
             "ci_high_six": e["six_seed"]["ci_high"], "p_six": e["six_seed"]["p"], "opt_seed_sd": e["six_seed"]["opt_seed_sd"],
             "def_seed_sd": e["six_seed"]["def_seed_sd"], "diff_seed11": e["seed11"]["diff"],
             "ci_low_seed11": e["seed11"]["ci_low"], "holds_six": e["holds_six"], "holds_seed11": e["holds_seed11"],
             "same": e["holds_six"] == e["holds_seed11"]} for n, e in evals.items()]
    return {"tasks": rows, "n": len(rows), "holds_six": sum(r["holds_six"] for r in rows),
            "holds_seed11": sum(r["holds_seed11"] for r in rows), "changed": [r["task"] for r in rows if not r["same"]],
            "determinism": det, "skipped": skipped}


def main(argv: Optional[List[str]] = None) -> int:
    """python -m modules.m5_modeling.components.sgb_seeds [--config ...]"""
    import argparse
    import os

    import yaml
    ap = argparse.ArgumentParser(description="S5.M12 列采样配置的 SGB 训练随机性")
    ap.add_argument("--config", default="modules/m5_modeling/configs/s5_m12_sgb_seeds.yaml")
    a = ap.parse_args(argv)
    m12 = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(m12["base_config"]).read_text(encoding="utf-8"))
    P = MS.Params(cfg)
    priv, run_dir = Path(os.environ[cfg["private_root_env"]]), Path(os.environ[cfg["run_dir_env"]])
    m9_dir, sgb_dir, large_dir, out = (Path(os.environ[m12[k]]) for k in
                                       ("m9_dir_env", "sgb_dir_env", "large_dir_env", "out_dir_env"))
    dirs = [m9_dir, sgb_dir]
    out.mkdir(parents=True, exist_ok=True)
    comp = Path(__file__)
    fp = SC.fingerprint([comp, Path(a.config), Path(m12["decision_record"])] +
                        [comp.with_name(n) for n in ("sgb_i2i3_confirm.py", "sgb_cost_screen.py", "method_screen_sgb.py",
                                                     "method_screen.py")])
    with open(out / "fingerprint_launch.txt", "a", encoding="utf-8") as fh:
        fh.write("== %s ==\n%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), fp))
    seg = IC.Segment(m12["segment_budget_sec"], m12["segment_margin"])
    seeds = [int(s) for s in P.seeds[1:]]
    tasks = {t["name"]: t for t in m12["tasks"]}

    def screen(name):
        res_dir = large_dir if tasks.get(name, {}).get("large") else run_dir
        return (json.loads((res_dir / (name + ".json")).read_text(encoding="utf-8")),
                MS.Screen(name, cfg, priv, out / "unused"))

    dj = out / "determinism.json"
    if not dj.exists():
        name = m12["determinism_task"]
        if not seg.allows(m12["determinism_budget_sec"]):
            return 0
        res, sc = screen(name)
        SC.write_json(dj, determinism(P, sc, res, stored(name, DEF, dirs), seeds[0]))
        MS.log("%s · 原版联邦换 seed 逐位相同：%s" % (name, json.loads(dj.read_text(encoding="utf-8"))["identical"]))
        del sc
    det = json.loads(dj.read_text(encoding="utf-8"))
    methods = methods_for(det)
    skipped: Dict[str, str] = {}
    for method in methods:                       # D-8：先做全部任务的优化版联邦新 seed，再做原版联邦
        for name in tasks:
            if name in skipped or (out / ("%s__seeds_eval.json" % name)).exists():
                continue
            state = status(out, name, method, seeds)
            if state is None:
                res, sc = screen(name)
                state = run_task(P, sc, res, out, m12["time_cap_sec"], seg, m12["min_probe_budget_sec"], seeds, method)
                del sc
            if state == "paused":
                MS.log("本段预算用完（已用 %.0f 秒），下一段从 %s · %s 继续" % (seg.elapsed(), name, method))
                return 0
            if state == "skipped":
                skipped[name] = "%s：%s" % (method, json.loads(probe_path(out, name, method).read_text(encoding="utf-8"))["verdict"])
    evals: Dict[str, Dict] = {}
    for name in tasks:
        ej = out / ("%s__seeds_eval.json" % name)
        if name in skipped:
            continue
        if not ej.exists():
            res, sc = screen(name)
            SC.write_json(ej, evaluate_task(P, sc, res, out, dirs, seeds, methods))
            del sc
        evals[name] = json.loads(ej.read_text(encoding="utf-8"))
        ev = evals[name]
        MS.log("%s · 6 seed：I2 + I3 %+.4f [%+.4f, %+.4f]，seed 11 单次 %+.4f" % (
            name, ev["six_seed"]["diff"], ev["six_seed"]["ci_low"], ev["six_seed"]["ci_high"], ev["seed11"]["diff"]))
    SC.write_json(out / "summary.json", summarize(evals, det, skipped))
    MS.log("S5.M12 全部任务完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
