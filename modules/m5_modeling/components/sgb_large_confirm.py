"""M5 · S5.M8：Orange 大版流失的真实 SGB 复核（DR-M5-015 · D-6）。

- 前提：S5.M7 判定主动方明文直方图与原版逐位一致（DR-M5-015 · D-3），故用明文直方图版本拟合（结果相同、只是更快）。
- 输入：S5.M5 主分析的私有输入（每方前 300 个编码列，final 视图）与结果 JSON（选定配置、6 seed 代理结果）。
- 方法：原版联邦 F_def（隐语默认配置、10 桶、初始分数 0.5）与优化版联邦 F_opt_p（S5.M5 选定配置、33 桶、先验初始分数），
  各拟合一次（seed 11）；与同配置代理（seed 11）比较，SGB 内部对比 F_opt_p − F_def；口径同 S5.M2（`method_screen_sgb.evaluate`）。
- 每个方法的打分与拟合信息分别存档，会话中断后可续跑；`--method` 只拟合一个方法（一个后台段），不带时做对比。
- 启动时按 D-6 估算每个方法的拟合秒数并存档；估计超过上限的方法不拟合，如实记录。

需要 SecretFlow（secretflow-lite 1.13.0b0）；导入本模块不依赖它，调用时才导入。打分在模拟中直接取回，只用于离线评估。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from modules.m5_modeling.components import method_screen as MS
from modules.m5_modeling.components import method_screen_sgb as SG
from modules.m5_modeling.components import sgb_cost_screen as SC

METHODS = ("F_def", "F_opt_p")
PAIRS: Tuple[Tuple[str, str], ...] = (("F_opt_p", "F_def"),)


def estimate_seconds(P: MS.Params, res: Dict, method: str, sec_per_tree: float, depth_ref: int) -> float:
    """D-6 的时长估算：每棵树秒数（D-5，深度 depth_ref）× 层数比 × 棵数。"""
    cfg = SG.method_spec(P, res, method)[0]
    return float(sec_per_tree) * int(cfg["depth"]) / int(depth_ref) * int(cfg["rounds"])


def fit_one(P: MS.Params, sc: "MS.Screen", res: Dict, method: str, out: Path) -> Dict[str, Any]:
    """一个方法：明文直方图版本的 SGB 打分与同配置代理（已存档则直接读）。"""
    info = out / ("%s__%s.json" % (sc.name, method))
    if not info.exists():
        with SC.plaintext_label_holder(True):
            r = SG.confirm(P, sc, res, method, out)
        info.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return json.loads(info.read_text(encoding="utf-8"))


def run(P: MS.Params, sc: "MS.Screen", res: Dict, out: Path, methods: Sequence[str] = METHODS,
        pairs: Sequence[Tuple[str, str]] = PAIRS) -> Dict[str, Any]:
    """全部方法拟合后，SGB 与代理、SGB 内部对比（同一批 bootstrap）。"""
    fits = {m: fit_one(P, sc, res, m, out) for m in methods}
    sgb = {m: np.load(out / ("%s__%s.npy" % (sc.name, m))).astype(float) for m in methods}
    proxy = {m: SG.proxy_pred(P, sc, res, m) for m in methods}
    ev = SG.evaluate(P, sc, res, sgb, proxy, list(pairs))
    return {"task": sc.name, "fits": fits, **ev}


def plan(P: MS.Params, res: Dict, methods: Sequence[str], sec_per_tree: float, depth_ref: int,
         cap: float) -> Dict[str, Any]:
    """D-6：每个方法的估计秒数；不超过上限的才拟合。"""
    est = {m: round(estimate_seconds(P, res, m, sec_per_tree, depth_ref), 1) for m in methods}
    return {"estimate_seconds": est, "cap_seconds": float(cap), "todo": [m for m in methods if est[m] <= cap]}


def main(argv: Optional[List[str]] = None) -> int:
    """python -m modules.m5_modeling.components.sgb_large_confirm [--config ...] [--method F_opt_p|F_def]"""
    import argparse
    import os

    import yaml
    ap = argparse.ArgumentParser(description="S5.M8 Orange 大版流失的真实 SGB 复核")
    ap.add_argument("--config", default="modules/m5_modeling/configs/s5_m8_orange_large_sgb.yaml")
    ap.add_argument("--method", choices=METHODS, help="只拟合这一个方法（一个后台段），不做对比")
    a = ap.parse_args(argv)
    m8 = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(m8["base_config"]).read_text(encoding="utf-8"))
    P = MS.Params(cfg)
    priv = Path(os.environ[cfg["private_root_env"]])
    large_dir, out = (Path(os.environ[m8[k]]) for k in ("large_dir_env", "out_dir_env"))
    out.mkdir(parents=True, exist_ok=True)
    comp = Path(__file__)
    fp = SC.fingerprint([comp, Path(a.config), Path(m8["decision_record"]), comp.with_name("sgb_cost_screen.py"),
                         comp.with_name("method_screen_sgb.py"), comp.with_name("method_screen.py")])
    with open(out / "fingerprint_launch.txt", "a", encoding="utf-8") as fh:
        fh.write("== %s %s ==\n%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), a.method or "evaluate", fp))
    name = m8["task"]
    res = json.loads((large_dir / (name + ".json")).read_text(encoding="utf-8"))
    pl = plan(P, res, list(m8["methods"]), m8["sec_per_tree_ref"], m8["depth_ref"], m8["time_cap_sec"])
    SC.write_json(out / ("%s__plan.json" % name), pl)
    MS.log("%s · D-6 估算 %s，上限 %s 秒" % (name, pl["estimate_seconds"], pl["cap_seconds"]))
    sc = MS.Screen(name, cfg, priv, out / "unused")
    if a.method:
        assert a.method in pl["todo"], "估计超过上限，按 D-6 不拟合"
        fit_one(P, sc, res, a.method, out)
        MS.log("%s · %s 拟合完成" % (name, a.method))
        return 0
    pairs = [p for p in PAIRS if set(p) <= set(pl["todo"])]
    r = run(P, sc, res, out, tuple(pl["todo"]), pairs)
    SC.write_json(out / ("%s__summary.json" % name), {**r, "plan": pl})
    MS.log("%s · SGB 复核完成" % name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
