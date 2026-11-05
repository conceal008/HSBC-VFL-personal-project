"""M5 · S5.M9：I2 + I3 在其余 14 个任务上的真实 SGB 复核（DR-M5-016 · D-2 ~ D-5）。

- 每个任务：原版联邦 F_def 与优化版联邦 F_opt_p 各一次 SGB（seed 11），配置取自 S5.M1 / S5.M5 的结果 JSON；
  已有 S5.M2 存档打分的任务复用其优化版联邦（`reuse_opt`）。
- 新拟合用主动方明文直方图版本（S5.M7）。正式拟合前的抽查（`probe`）：
  前 2 棵原版与明文打分须逐位相同（D-3）；明文前 2 / 4 棵的边际每棵树秒数外推全部棵数（D-4），超过上限的不做。
- 分段运行：抽查、拟合、对比各自存档，会话中断或到达本段预算后下一段从停下处继续。
- 对比口径同 S5.M2 / S5.M8（`method_screen_sgb.evaluate`）；全部任务完成后另做跨任务 Holm（D-5）。

需要 SecretFlow（secretflow-lite 1.13.0b0）；导入本模块不依赖它，调用时才导入。打分在模拟中直接取回，只用于离线评估。
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from modules.m5_modeling.components import method_screen as MS
from modules.m5_modeling.components import method_screen_sgb as SG
from modules.m5_modeling.components import sgb_cost_screen as SC
from modules.m5_modeling.components import sgb_large_confirm as LC

METHODS = ("F_def", "F_opt_p")
PAIR = ("F_opt_p", "F_def")
PROBE_SMALL = 2
PROBE_LARGE = 4
NOT_IDENTICAL = "未做（明文不一致）"
OVER_CAP = "未做（外推超过上限）"


def probe(P: MS.Params, sc: "MS.Screen", res: Dict, method: str) -> Dict[str, Any]:
    """D-3 / D-4：前 2 棵原版与明文逐位比较；明文前 2 / 4 棵的边际每棵树秒数外推全部棵数。"""
    args = SC.task_args(P, sc, res, method)
    rounds = int(args[SC.ARG_CFG]["rounds"])
    small = SC.fit(P, args, plain=True, rounds=PROBE_SMALL)
    large = SC.fit(P, args, plain=True, rounds=PROBE_LARGE)
    base = SC.fit(P, args, plain=False, rounds=PROBE_SMALL)
    marg = (large["fit_seconds"] - small["fit_seconds"]) / (PROBE_LARGE - PROBE_SMALL)
    if rounds <= PROBE_LARGE:
        projected = large["fit_seconds"] * rounds / PROBE_LARGE
    else:
        projected = small["fit_seconds"] + max(marg, 0.0) * (rounds - PROBE_SMALL)
    return {"rounds": rounds, "depth": int(args[SC.ARG_CFG]["depth"]), "bins": args[SC.ARG_BINS],
            "fit_seconds_small": small["fit_seconds"], "fit_seconds_large": large["fit_seconds"],
            "fit_seconds_small_base": base["fit_seconds"], "marginal_sec_per_tree": marg,
            "projected_fit_seconds": projected, "identical_first_trees": bool(np.array_equal(small["pred"], base["pred"]))}


def plan_task(task: Dict) -> List[str]:
    """这个任务还要新拟合的方法（复用 S5.M2 存档的除外）。"""
    return [m for m in METHODS if not (m == "F_opt_p" and task.get("reuse_opt"))]


def status_of(name: str, method: str, out: Path) -> Optional[str]:
    """已存档的拟合 → "fitted"；已判定不做 → 原因；否则 None。"""
    if (out / ("%s__%s.json" % (name, method))).exists():
        return "fitted"
    pj = out / ("%s__%s__probe.json" % (name, method))
    if pj.exists():
        p = json.loads(pj.read_text(encoding="utf-8"))
        return p.get("verdict")
    return None


def skip_reasons(name: str, task: Dict, out: Path) -> Dict[str, str]:
    """已判定不做的方法及原因（D-3 / D-4）。"""
    st = {m: status_of(name, m, out) for m in plan_task(task)}
    return {m: str(r) for m, r in st.items() if r in (NOT_IDENTICAL, OVER_CAP)}


def scores(name: str, out: Path, sgb_dir: Path, task: Dict) -> Dict[str, np.ndarray]:
    """两个方法的 SGB 测试打分：新拟合在 out，复用的优化版联邦在 S5.M2 存档目录（复制一份到 out 并记录来源）。"""
    if task.get("reuse_opt") and not (out / ("%s__F_opt_p.npy" % name)).exists():
        shutil.copyfile(sgb_dir / ("%s__F_opt_p.npy" % name), out / ("%s__F_opt_p.npy" % name))
    return {m: np.load(out / ("%s__%s.npy" % (name, m))).astype(float) for m in METHODS}


def evaluate_task(P: MS.Params, sc: "MS.Screen", res: Dict, sgb: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """SGB 与同配置代理、SGB 上的 I2 + I3；只作描述：优化版联邦 SGB − 内地单边代理（6 seed）。"""
    proxy = {m: SG.proxy_pred(P, sc, res, m) for m in sgb}
    ev = SG.evaluate(P, sc, res, sgb, proxy, [PAIR])
    g_t = res["test"]["G_t"]["auroc"]
    ev["descriptive"] = {m: {"sgb_auroc": ev["methods"][m]["sgb_auroc"], "guest_only_proxy_6seed_auroc": g_t,
                             "sgb_minus_guest_only_proxy": ev["methods"][m]["sgb_auroc"] - g_t} for m in sgb}
    return ev


def summarize(evals: Dict[str, Dict], skipped: Dict[str, Dict[str, str]], tol: float) -> Dict[str, Any]:
    """跨任务汇总：I2 + I3 在 SGB 上成立的任务数、与代理结论一致的任务数、跨任务 Holm、超出容差的（任务，方法）。"""
    key = "%s − %s" % PAIR
    names = list(evals)
    holm = MS.holm([evals[n]["contrasts"][key]["sgb"]["p"] for n in names])
    rows, outside = [], []
    for n, h in zip(names, holm):
        c = evals[n]["contrasts"][key]
        sgb, px = c["sgb"], c["proxy_6seed"] or {}
        sgb_sig = sgb["ci_low"] > 0
        px_sig = bool(px) and px["ci_low"] > 0
        rows.append({"task": n, "sgb_diff": sgb["diff"], "sgb_ci_low": sgb["ci_low"], "sgb_ci_high": sgb["ci_high"],
                     "sgb_p": sgb["p"], "sgb_p_holm": h, "sgb_holds": sgb_sig, "proxy_diff": px.get("diff"),
                     "proxy_holds": px_sig, "agree": sgb_sig == px_sig})
        for m, v in evals[n]["methods"].items():
            if abs(v["sgb_minus_proxy"]["diff"]) > tol:
                outside.append({"task": n, "method": m, "sgb_minus_proxy": v["sgb_minus_proxy"]["diff"]})
    return {"tasks": rows, "sgb_holds": sum(r["sgb_holds"] for r in rows), "proxy_holds": sum(r["proxy_holds"] for r in rows),
            "agree": sum(r["agree"] for r in rows), "n": len(rows), "outside_tolerance": outside, "skipped": skipped}


class Segment:
    """一段运行的预算：只在「已用 + 外推 × 余量 ≤ 预算」时开始下一个拟合（D-4）。"""

    def __init__(self, budget: float, margin: float, clock: Callable[[], float] = time.perf_counter):
        self.budget, self.margin, self.clock, self.t0 = float(budget), float(margin), clock, clock()

    def elapsed(self) -> float:
        return self.clock() - self.t0

    def allows(self, projected: float) -> bool:
        return self.elapsed() + projected * self.margin <= self.budget


def run_task(P: MS.Params, sc: "MS.Screen", res: Dict, task: Dict, out: Path, cap: float, seg: Segment,
             min_probe: float) -> str:
    """一个任务：抽查 → 拟合（每个方法）→ 对比。返回 "done"、"skipped" 或 "paused"（本段预算不够，下一段继续）。"""
    name = sc.name
    for m in plan_task(task):
        st = status_of(name, m, out)
        if st == "fitted":
            continue
        if st in (NOT_IDENTICAL, OVER_CAP):
            return "skipped"
        pj = out / ("%s__%s__probe.json" % (name, m))
        if not pj.exists():
            if not seg.allows(min_probe):
                return "paused"
            pr = probe(P, sc, res, m)
            pr["verdict"] = (NOT_IDENTICAL if not pr["identical_first_trees"]
                             else OVER_CAP if pr["projected_fit_seconds"] > cap else "ok")
            SC.write_json(pj, pr)
            MS.log("%s · %s 抽查：前 2 棵逐位一致 %s，外推 %.0f 秒（%s）" % (
                name, m, pr["identical_first_trees"], pr["projected_fit_seconds"], pr["verdict"]))
            if pr["verdict"] != "ok":
                return "skipped"
        pr = json.loads(pj.read_text(encoding="utf-8"))
        if not seg.allows(pr["projected_fit_seconds"]):
            return "paused"
        LC.fit_one(P, sc, res, m, out)
        MS.log("%s · %s 拟合完成" % (name, m))
    return "done"


def main(argv: Optional[List[str]] = None) -> int:
    """python -m modules.m5_modeling.components.sgb_i2i3_confirm [--config ...]

    环境变量：基础配置的 private_root_env / run_dir_env，本步配置的 sgb_dir_env / large_dir_env / out_dir_env。
    """
    import argparse
    import os

    import yaml
    ap = argparse.ArgumentParser(description="S5.M9 I2 + I3 在其余任务上的真实 SGB 复核")
    ap.add_argument("--config", default="modules/m5_modeling/configs/s5_m9_sgb_i2i3.yaml")
    a = ap.parse_args(argv)
    m9 = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(m9["base_config"]).read_text(encoding="utf-8"))
    P = MS.Params(cfg)
    priv, run_dir = Path(os.environ[cfg["private_root_env"]]), Path(os.environ[cfg["run_dir_env"]])
    sgb_dir, large_dir, out = (Path(os.environ[m9[k]]) for k in ("sgb_dir_env", "large_dir_env", "out_dir_env"))
    out.mkdir(parents=True, exist_ok=True)
    comp = Path(__file__)
    fp = SC.fingerprint([comp, Path(a.config), Path(m9["decision_record"])] +
                        [comp.with_name(n) for n in ("sgb_large_confirm.py", "sgb_cost_screen.py",
                                                     "method_screen_sgb.py", "method_screen.py")])
    with open(out / "fingerprint_launch.txt", "a", encoding="utf-8") as fh:
        fh.write("== %s ==\n%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), fp))
    seg = Segment(m9["segment_budget_sec"], m9["segment_margin"])
    evals: Dict[str, Dict] = {}
    skipped: Dict[str, Dict[str, str]] = {}
    for task in m9["tasks"]:
        name = task["name"]
        ej = out / ("%s__eval.json" % name)
        if ej.exists():
            evals[name] = json.loads(ej.read_text(encoding="utf-8"))
            continue
        if skip_reasons(name, task, out):
            skipped[name] = skip_reasons(name, task, out)
            continue
        res_dir = large_dir if task.get("large") else run_dir
        res = json.loads((res_dir / (name + ".json")).read_text(encoding="utf-8"))
        sc = MS.Screen(name, cfg, priv, out / "unused")
        state = run_task(P, sc, res, task, out, m9["time_cap_sec"], seg, m9["min_probe_budget_sec"])
        if state == "paused":
            MS.log("本段预算用完（已用 %.0f 秒），下一段从 %s 继续" % (seg.elapsed(), name))
            return 0
        if state == "skipped":
            skipped[name] = skip_reasons(name, task, out)
            continue
        ev = evaluate_task(P, sc, res, scores(name, out, sgb_dir, task))
        ev["reuse_opt"] = bool(task.get("reuse_opt"))
        SC.write_json(ej, ev)
        evals[name] = ev
        MS.log("%s · 对比完成：I2 + I3 SGB %+.4f" % (name, ev["contrasts"]["%s − %s" % PAIR]["sgb"]["diff"]))
        del sc
    SC.write_json(out / "summary.json", summarize(evals, skipped, m9["proxy_tol"]))
    MS.log("S5.M9 全部任务完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
