"""M5 · S5.M11：M6 在其余 4 个代理显著任务上的真实 SGB 复核（DR-M5-017 · D-2 ~ D-3）。

- M6 = 6 个 SGB 成员的概率平均（优化版联邦配置、行采样 0.8、成员 seed 取 method_screen_sgb.bag_seeds，同 S5.M2 / S5.M4）；
- 用主动方明文直方图版本拟合；每个任务第一个成员前抽查前 2 棵原版与明文逐位相同，并外推单个成员的秒数（同 S5.M9）；
- 成员逐个存档，分段运行（sgb_i2i3_confirm.Segment），会话中断或到达本段预算后下一段从停下处继续；
- 对照：S5.M9 已拟合的优化版联邦 SGB 打分（同一输入、seed 11）；对比口径同 S5.M2（method_screen_sgb.evaluate）。

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

METHOD = "F_M6"
BASE = "F_opt_p"
PAIR = (METHOD, BASE)


def member_args(P: MS.Params, sc: "MS.Screen", res: Dict) -> tuple:
    """M6 成员的 SGB 输入（配置、分桶、初始分数取自结果 JSON）。"""
    fv = sc.v["final"]
    cfg, bins, b_sgb, _, mf, _ = SG.inputs(P, sc, res, METHOD)
    assert mf is None
    return (fv["fit_xg"], fv["fit_xh"], fv["fit_y"], fv["ev_xg"], fv["ev_xh"], cfg, bins, b_sgb)


def member_path(out: Path, name: str, j: int, ext: str) -> Path:
    return out / ("%s__%s_m%d.%s" % (name, METHOD, j, ext))


def probe(P: MS.Params, args: tuple) -> Dict[str, Any]:
    """前 2 棵原版与明文逐位比较；明文前 2 / 4 棵外推单个成员的秒数（seed 11 的抽查，同 S5.M9）。"""
    rounds = int(args[SC.ARG_CFG]["rounds"])
    small = SC.fit(P, args, plain=True, rounds=IC.PROBE_SMALL)
    large = SC.fit(P, args, plain=True, rounds=IC.PROBE_LARGE)
    base = SC.fit(P, args, plain=False, rounds=IC.PROBE_SMALL)
    marg = (large["fit_seconds"] - small["fit_seconds"]) / (IC.PROBE_LARGE - IC.PROBE_SMALL)
    projected = small["fit_seconds"] + max(marg, 0.0) * (rounds - IC.PROBE_SMALL)
    ok = bool(np.array_equal(small["pred"], base["pred"]))
    return {"rounds": rounds, "marginal_sec_per_tree": marg, "projected_member_seconds": projected,
            "identical_first_trees": ok}


def fit_member(P: MS.Params, args: tuple, seed: int) -> Dict[str, Any]:
    """一个成员：明文直方图版本的 SGB（行采样 0.8，seed 为成员 seed）。"""
    xg, xh, y, eg, eh, cfg, bins, base = args
    with SC.plaintext_label_holder(True):
        pred, secs = SG.run_sgb(P, xg, xh, y, eg, eh, cfg, bins, base, seed=seed)
    return {"pred": np.asarray(pred, np.float32), "seed": int(seed), "fit_seconds": round(float(secs), 1)}


def run_task(P: MS.Params, sc: "MS.Screen", res: Dict, out: Path, cap: float, seg: IC.Segment,
             min_probe: float) -> str:
    """一个任务：抽查 → 逐个成员拟合并存档。返回 "done"、"skipped" 或 "paused"。"""
    name = sc.name
    pj = out / ("%s__%s__probe.json" % (name, METHOD))
    args = member_args(P, sc, res)
    if not pj.exists():
        if not seg.allows(min_probe):
            return "paused"
        pr = probe(P, args)
        pr["verdict"] = (IC.NOT_IDENTICAL if not pr["identical_first_trees"]
                         else IC.OVER_CAP if pr["projected_member_seconds"] > cap else "ok")
        SC.write_json(pj, pr)
        MS.log("%s · M6 抽查：前 2 棵逐位一致 %s，单个成员外推 %.0f 秒（%s）" % (
            name, pr["identical_first_trees"], pr["projected_member_seconds"], pr["verdict"]))
    pr = json.loads(pj.read_text(encoding="utf-8"))
    if pr["verdict"] != "ok":
        return "skipped"
    for j, sd in enumerate(SG.bag_seeds(P)):
        if member_path(out, name, j, "json").exists():
            continue
        if not seg.allows(pr["projected_member_seconds"]):
            return "paused"
        r = fit_member(P, args, sd)
        np.save(member_path(out, name, j, "npy"), r["pred"])
        member_path(out, name, j, "json").write_text(json.dumps({"seed": r["seed"], "fit_seconds": r["fit_seconds"]}),
                                                     encoding="utf-8")
        MS.log("%s · M6 成员 %d（seed %d）拟合完成，%.0f 秒" % (name, j, sd, r["fit_seconds"]))
    return "done"


def members(P: MS.Params, name: str, out: Path) -> List[Dict[str, Any]]:
    """已存档的成员打分与拟合信息。"""
    return [{**json.loads(member_path(out, name, j, "json").read_text(encoding="utf-8")),
             "pred": np.load(member_path(out, name, j, "npy")).astype(float)} for j in range(P.bag)]


def evaluate_task(P: MS.Params, sc: "MS.Screen", res: Dict, out: Path, opt_dir: Path) -> Dict[str, Any]:
    """M6（成员平均）与优化版联邦（S5.M9 打分）：SGB 与同配置代理、SGB 上 M6 − 优化版联邦；另报告每个成员的 SGB 测试 AUROC。"""
    ms = members(P, sc.name, out)
    sgb = {METHOD: np.mean([m["pred"] for m in ms], axis=0),
           BASE: np.load(opt_dir / ("%s__%s.npy" % (sc.name, BASE))).astype(float)}
    proxy = {m: SG.proxy_pred(P, sc, res, m) for m in sgb}
    ev = SG.evaluate(P, sc, res, sgb, proxy, [PAIR])
    y = sc.v["final"]["ev_y"]
    ev["members"] = [{"seed": m["seed"], "fit_seconds": m["fit_seconds"], "sgb_auroc": MS.auc(y, m["pred"])} for m in ms]
    ev["fit_seconds_total"] = round(sum(m["fit_seconds"] for m in ms), 1)
    return ev


def summarize(evals: Dict[str, Dict], skipped: Dict[str, str], tol: float) -> Dict[str, Any]:
    """跨任务汇总：M6 在 SGB 上成立的任务数、与代理结论一致的任务数、M6 的 SGB 与代理之差超出容差的任务。"""
    key = "%s − %s" % PAIR
    rows = []
    for n, ev in evals.items():
        c = ev["contrasts"][key]
        sgb, px = c["sgb"], c["proxy_6seed"] or {}
        rows.append({"task": n, "sgb_diff": sgb["diff"], "sgb_ci_low": sgb["ci_low"], "sgb_ci_high": sgb["ci_high"],
                     "sgb_p": sgb["p"], "sgb_holds": sgb["ci_low"] > 0, "proxy_diff": px.get("diff"),
                     "proxy_holds": bool(px) and px["ci_low"] > 0,
                     "m6_sgb_minus_proxy": ev["methods"][METHOD]["sgb_minus_proxy"]["diff"]})
    for r in rows:
        r["agree"] = r["sgb_holds"] == r["proxy_holds"]
    return {"tasks": rows, "n": len(rows), "sgb_holds": sum(r["sgb_holds"] for r in rows),
            "agree": sum(r["agree"] for r in rows),
            "outside_tolerance": [r["task"] for r in rows if abs(r["m6_sgb_minus_proxy"]) > tol], "skipped": skipped}


def main(argv: Optional[List[str]] = None) -> int:
    """python -m modules.m5_modeling.components.sgb_m6_more [--config ...]"""
    import argparse
    import os

    import yaml
    ap = argparse.ArgumentParser(description="S5.M11 M6 在其余 4 个代理显著任务上的真实 SGB 复核")
    ap.add_argument("--config", default="modules/m5_modeling/configs/s5_m11_sgb_m6.yaml")
    a = ap.parse_args(argv)
    m11 = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(m11["base_config"]).read_text(encoding="utf-8"))
    P = MS.Params(cfg)
    priv, run_dir = Path(os.environ[cfg["private_root_env"]]), Path(os.environ[cfg["run_dir_env"]])
    opt_dir, out = (Path(os.environ[m11[k]]) for k in ("opt_dir_env", "out_dir_env"))
    out.mkdir(parents=True, exist_ok=True)
    comp = Path(__file__)
    fp = SC.fingerprint([comp, Path(a.config), Path(m11["decision_record"])] +
                        [comp.with_name(n) for n in ("sgb_i2i3_confirm.py", "sgb_cost_screen.py", "method_screen_sgb.py",
                                                     "method_screen.py")])
    with open(out / "fingerprint_launch.txt", "a", encoding="utf-8") as fh:
        fh.write("== %s ==\n%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), fp))
    seg = IC.Segment(m11["segment_budget_sec"], m11["segment_margin"])
    evals: Dict[str, Dict] = {}
    skipped: Dict[str, str] = {}
    for name in m11["tasks"]:
        ej = out / ("%s__%s__eval.json" % (name, METHOD))
        if ej.exists():
            evals[name] = json.loads(ej.read_text(encoding="utf-8"))
            continue
        res = json.loads((run_dir / (name + ".json")).read_text(encoding="utf-8"))
        sc = MS.Screen(name, cfg, priv, out / "unused")
        state = run_task(P, sc, res, out, m11["time_cap_sec"], seg, m11["min_probe_budget_sec"])
        if state == "paused":
            MS.log("本段预算用完（已用 %.0f 秒），下一段从 %s 继续" % (seg.elapsed(), name))
            return 0
        if state == "skipped":
            skipped[name] = json.loads((out / ("%s__%s__probe.json" % (name, METHOD))).read_text(encoding="utf-8"))["verdict"]
            continue
        ev = evaluate_task(P, sc, res, out, opt_dir)
        SC.write_json(ej, ev)
        evals[name] = ev
        MS.log("%s · M6 对比完成：SGB 上 M6 − 优化版 %+.4f" % (name, ev["contrasts"]["%s − %s" % PAIR]["sgb"]["diff"]))
        del sc
    SC.write_json(out / "summary.json", summarize(evals, skipped, m11["proxy_tol"]))
    MS.log("S5.M11 全部任务完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
