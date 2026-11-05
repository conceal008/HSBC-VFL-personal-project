"""M5 · S5.M7 / S5.M8：主动方明文直方图（候选文档 M3 后半）与预筛列数的真实 SGB 开销（DR-M5-015）。

- 隐语 1.13 `sgb_v` 默认把同一份梯度密文发给所有参与方，**包括主动方自己**，主动方特征上的分桶求和也走密文。
  `plaintext_label_holder()` 在实验代码里替换 `GradientEncryptor.cache_to_workers`：被动方仍收到同一份密文，
  主动方改用自己明文梯度的 HEU 明文对象（隐语 label_holder_feature_only 分支里主动方那一份的做法）。不改隐语源码。
- `exact_check`：完整拟合一次，与 S5.M2 存档的测试打分逐行比较（DR-M5-015 · D-3）。
- `timing`：预热后按「原版、明文、明文、原版」各拟合前 n 棵树，记录拟合秒数与两方分桶求和、取回秒数（D-4）。
- `k_cost`：Orange 大版上每方 k 列的单树开销（D-5）；`confirm_large`：Orange 大版流失的真实 SGB 复核（D-6）。

需要 SecretFlow（secretflow-lite 1.13.0b0）；导入本模块不依赖它，调用时才导入。打分在模拟中直接取回，只用于离线评估。
"""
from __future__ import annotations

import contextlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

from modules.m5_modeling.components import method_screen as MS
from modules.m5_modeling.components import method_screen_large as ML
from modules.m5_modeling.components import method_screen_sgb as SG

ORDER = ("base", "plain", "plain", "base")     # D-4：ABBA
WARMUP_TREES = 2
EXACT = "逐位一致"
EQUIVALENT = "数值等价"
NOT_EQUIVALENT = "不等价"
PRED_TOL = 1e-6          # D-3：「数值等价」允许的逐行最大差
ARG_CFG = 5              # SGB 输入元组 (xg, xh, y, eg, eh, cfg, bins, base_logit) 中配置的位置
ARG_BINS = 6             # 同上，分桶数的位置


# ---------------------------------------------------------------- 主动方明文直方图（D-2）
def plain_cache_to_workers(orig: Callable) -> Callable:
    """返回替换后的 cache_to_workers：被动方与原实现相同，主动方拿自己明文梯度的 HEU 明文对象。"""
    from secretflow.ml.boost.sgb_v.factory.components.gradient_encryptor import gradient_encryptor as GE

    def cache_to_workers(self, encrypted_gh, gh):
        if self.params.label_holder_feature_only:
            return orig(self, encrypted_gh, gh)
        cache = {w: encrypted_gh.to(self.heu, GE.move_config(w, self.gh_encoder))
                 for w in self.workers if w != self.label_holder}
        cache[self.label_holder] = gh.to(self.heu, GE.move_config(self.label_holder, self.gh_encoder))
        return cache
    return cache_to_workers


@contextlib.contextmanager
def plaintext_label_holder(enabled: bool = True) -> Iterator[None]:
    """enabled 为真时在上下文内替换 cache_to_workers，退出时恢复。"""
    from secretflow.ml.boost.sgb_v.factory.components.gradient_encryptor import gradient_encryptor as GE
    orig = GE.GradientEncryptor.cache_to_workers
    if enabled:
        GE.GradientEncryptor.cache_to_workers = plain_cache_to_workers(orig)
    try:
        yield
    finally:
        GE.GradientEncryptor.cache_to_workers = orig


@contextlib.contextmanager
def party_timer(acc: Dict[str, float]) -> Iterator[None]:
    """包一层计时：HEUObject 的分桶求和与取回（解密 / 解码到 PYU），按所在方累加秒数；不改计算。"""
    from secretflow.device.device.heu_object import HEUObject
    from secretflow.device.device.pyu import PYU
    bucket, move = HEUObject.batch_feature_wise_bucket_sum, HEUObject.to

    def timed_bucket(self, *a, **k):
        t = time.perf_counter()
        try:
            return bucket(self, *a, **k)
        finally:
            acc["bucket_sum@" + str(self.location)] += time.perf_counter() - t

    def timed_to(self, device, *a, **k):
        if not isinstance(device, PYU):
            return move(self, device, *a, **k)
        t = time.perf_counter()
        try:
            return move(self, device, *a, **k)
        finally:
            acc["to_pyu@" + str(self.location) + ("（明文）" if self.is_plain else "（密文）")] += time.perf_counter() - t

    HEUObject.batch_feature_wise_bucket_sum, HEUObject.to = timed_bucket, timed_to
    try:
        yield
    finally:
        HEUObject.batch_feature_wise_bucket_sum, HEUObject.to = bucket, move


def fit(P: MS.Params, args: Sequence, plain: bool, rounds: Optional[int] = None) -> Dict[str, Any]:
    """拟合一次 SGB（SG.run_sgb）。args = (xg, xh, y, eg, eh, cfg, bins, base_logit)；rounds 非空时只训前 rounds 棵。"""
    xg, xh, y, eg, eh, cfg, bins, base = args
    cfg = dict(cfg) if rounds is None else {**cfg, "rounds": int(rounds)}
    acc: Dict[str, float] = defaultdict(float)
    with plaintext_label_holder(plain), party_timer(acc):
        p, fit_s = SG.run_sgb(P, xg, xh, y, eg, eh, cfg, bins, base)
    return {"pred": p.astype(np.float32), "fit_seconds": fit_s, "trees": int(cfg["rounds"]),
            "parts": {k: round(v, 2) for k, v in sorted(acc.items())}}


def compare(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """逐行打分比较（float32 存档）与 D-3 判定。"""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    d = np.abs(a.astype(float) - b.astype(float))
    same_auc = MS.auc(y, a) == MS.auc(y, b)
    n_diff = int(np.sum(a != b))
    verdict = EXACT if n_diff == 0 else (EQUIVALENT if float(d.max()) <= PRED_TOL and same_auc else NOT_EQUIVALENT)
    return {"rows": int(len(a)), "rows_different": n_diff, "max_abs_diff": float(d.max()),
            "auroc_a": MS.auc(y, a), "auroc_b": MS.auc(y, b), "verdict": verdict}


def task_args(P: MS.Params, sc: "MS.Screen", res: Dict, method: str) -> tuple:
    fv = sc.v["final"]
    cfg, bins, b_sgb, _, mf, _ = SG.inputs(P, sc, res, method)
    assert mf is None, "S5.M7 只用不带逐样本先验的方法"
    return (fv["fit_xg"], fv["fit_xh"], fv["fit_y"], fv["ev_xg"], fv["ev_xh"], cfg, bins, b_sgb)


# ---------------------------------------------------------------- D-3 / D-4
def exact_check(P: MS.Params, sc: "MS.Screen", res: Dict, method: str, stored: np.ndarray) -> Dict[str, Any]:
    """明文直方图版本完整拟合一次，与 S5.M2 存档的测试打分逐行比较。"""
    args = task_args(P, sc, res, method)
    r = fit(P, args, plain=True)
    out = compare(stored, r["pred"], sc.v["final"]["ev_y"])
    out.update({"fit_seconds_plain": round(r["fit_seconds"], 1), "trees": r["trees"], "parts_plain": r["parts"]})
    return out


def timing(P: MS.Params, sc: "MS.Screen", res: Dict, method: str, max_trees: int) -> Dict[str, Any]:
    """预热 1 次（不记录），再按 ORDER 各拟合前 n 棵树；四次打分须两两相同。"""
    args = task_args(P, sc, res, method)
    n = min(int(max_trees), int(args[ARG_CFG]["rounds"]))
    fit(P, args, plain=False, rounds=min(WARMUP_TREES, n))
    runs = [dict(fit(P, args, plain=(kind == "plain"), rounds=n), kind=kind) for kind in ORDER]
    preds = [r.pop("pred") for r in runs]
    same = all(np.array_equal(preds[0], q) for q in preds[1:])
    return {"trees": n, "runs": runs, "preds_identical": bool(same), **summarize(runs, n, sc)}


def summarize(runs: List[Dict], n: int, sc: Optional["MS.Screen"] = None, ng: Optional[int] = None,
              nh: Optional[int] = None) -> Dict[str, Any]:
    """两种版本的平均每棵树秒数与节省比例（每次重复单独算一次比例，报告范围）。"""
    base = [r["fit_seconds"] / n for r in runs if r["kind"] == "base"]
    plain = [r["fit_seconds"] / n for r in runs if r["kind"] == "plain"]
    if sc is not None:
        ng, nh = sc.v["final"]["fit_xg"].shape[1], sc.v["final"]["fit_xh"].shape[1]
    assert ng is not None and nh is not None
    saving = [1.0 - p / b for b, p in zip(base, plain)] if len(base) == len(plain) else []   # ABBA：相邻的一对
    return {"sec_per_tree_base": float(np.mean(base)), "sec_per_tree_plain": float(np.mean(plain)),
            "saving": 1.0 - float(np.mean(plain)) / float(np.mean(base)),
            "saving_range": [float(min(saving)), float(max(saving))] if saving else None,
            "cols_guest": int(ng), "cols_host": int(nh), "guest_share": float(ng) / float(ng + nh)}


# ---------------------------------------------------------------- D-5 / D-6（Orange 大版）
def large_view(sens: Dict[str, np.ndarray], k) -> Dict[str, np.ndarray]:
    """S5.M5 敏感性矩阵（final 视图）上每方前 k 列的 fit / 测试矩阵。sens 来自 ML.load_sens，不修改。"""
    fit_mask, y = sens["fit"], sens["y"]
    fi, ei = np.flatnonzero(fit_mask), np.flatnonzero(~fit_mask)
    g, h = ML.take(list(sens["rank_g"]), k), ML.take(list(sens["rank_h"]), k)
    out = {}
    for tag, idx in (("fit", fi), ("ev", ei)):
        out[tag + "_xg"] = ML.gather([(sens["G"], g)], idx)
        out[tag + "_xh"] = ML.gather([(sens["H"], h)], idx)
    out["fit_y"], out["ev_y"] = y[fi], y[ei]
    return out


def large_args(P: MS.Params, v: Dict[str, np.ndarray], res: Dict, method: str) -> tuple:
    """Orange 大版：方法 → SGB 输入（F_def / F_opt_p，配置取自 S5.M5 结果）。"""
    cfg, bins, kind, mkind = SG.method_spec(P, res, method)
    assert mkind is None
    b_sgb, _ = SG.base_of(P, v["fit_y"], kind)
    return (v["fit_xg"], v["fit_xh"], v["fit_y"], v["ev_xg"], v["ev_xh"], cfg, bins, b_sgb)


def k_cost(P: MS.Params, sens: Dict[str, np.ndarray], res: Dict, ks: Sequence, trees: int) -> List[Dict[str, Any]]:
    """每方 k 列、优化版联邦配置的前 trees 棵树：原版与明文各一次，k 之间交替先后。"""
    rows = []
    for i, k in enumerate(ks):
        v = large_view(sens, k)
        args = large_args(P, v, res, "F_opt_p")
        fit(P, args, plain=False, rounds=min(WARMUP_TREES, trees))
        order = ("base", "plain") if i % 2 == 0 else ("plain", "base")
        runs = [dict(fit(P, args, plain=(kind == "plain"), rounds=trees), kind=kind) for kind in order]
        preds = [r.pop("pred") for r in runs]
        rows.append({"k": k, "trees": trees, "runs": runs, "preds_identical": bool(np.array_equal(*preds)),
                     **summarize(runs, trees, ng=v["fit_xg"].shape[1], nh=v["fit_xh"].shape[1])})
        del v, args
    return rows


def linear_extrapolate(cols: Sequence[float], secs: Sequence[float], target: float) -> Dict[str, float]:
    """每棵树秒数对两方总列数的最小二乘直线，外推到 target 列。"""
    slope, icpt = np.polyfit(np.asarray(cols, float), np.asarray(secs, float), 1)
    return {"slope_sec_per_col": float(slope), "intercept_sec": float(icpt), "target_cols": float(target),
            "sec_per_tree_extrapolated": float(slope * target + icpt)}


def confirm_large(P: MS.Params, v: Dict[str, np.ndarray], res: Dict, method: str, out: Path, name: str) -> Dict[str, Any]:
    """Orange 大版一个方法：明文直方图版本的真实 SGB（D-3 判定逐位一致后才用），打分存档，与同配置代理比较。"""
    args = large_args(P, v, res, method)
    r = fit(P, args, plain=True)
    np.save(out / ("%s__%s.npy" % (name, method)), r["pred"])
    cfg, bins = args[ARG_CFG], args[ARG_BINS]
    b_proxy_kind = SG.method_spec(P, res, method)[2]
    _, b_proxy = SG.base_of(P, v["fit_y"], b_proxy_kind)
    xf = np.hstack([v["fit_xg"], v["fit_xh"]])
    xe = np.hstack([v["ev_xg"], v["ev_xh"]])
    q = MS.train_pred(P, xf, v["fit_y"], xe, cfg, bins, b_proxy, P.tune_seed)
    ye = v["ev_y"]
    return {"sgb_auroc": MS.auc(ye, r["pred"]), "proxy_seed11_auroc": MS.auc(ye, q),
            "fit_seconds": round(r["fit_seconds"], 1), "trees": r["trees"], "depth": int(cfg["depth"]), "bins": bins,
            "parts": r["parts"], "max_abs_pred_diff_vs_proxy": float(np.max(np.abs(r["pred"].astype(float) - q)))}


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 运行入口
def fingerprint(paths: Sequence[Path]) -> str:
    import hashlib
    lines = ["%s sha256:%s" % (p, hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in paths]
    try:
        import secretflow
        lines.append("secretflow %s" % secretflow.__version__)
    except ImportError:                                   # pragma: no cover - 只在没有 SecretFlow 的环境
        lines.append("secretflow 未安装")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """python -m modules.m5_modeling.components.sgb_cost_screen --part timing|kcost|exact [--tasks ...]

    环境变量：基础配置的 private_root_env / run_dir_env，本步配置的 sgb_dir_env / large_dir_env / cost_dir_env。
    每个任务的结果写到 cost_dir 下单独的 JSON，已完成的跳过，可续跑。
    """
    import argparse
    import os

    import yaml
    ap = argparse.ArgumentParser(description="S5.M7 主动方明文直方图与 SGB 开销")
    ap.add_argument("--config", default="modules/m5_modeling/configs/s5_m7_sgb_cost.yaml")
    ap.add_argument("--part", required=True, choices=["timing", "kcost", "exact"])
    ap.add_argument("--tasks", nargs="*")
    a = ap.parse_args(argv)
    m7 = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(m7["base_config"]).read_text(encoding="utf-8"))
    P = MS.Params(cfg)
    priv, run_dir = Path(os.environ[cfg["private_root_env"]]), Path(os.environ[cfg["run_dir_env"]])
    sgb_dir, large_dir, out = (Path(os.environ[m7[k]]) for k in ("sgb_dir_env", "large_dir_env", "cost_dir_env"))
    out.mkdir(parents=True, exist_ok=True)
    comp = Path(__file__)
    fp = fingerprint([comp, Path(a.config), Path(m7["decision_record"]), comp.with_name("method_screen_sgb.py"),
                      comp.with_name("method_screen.py"), comp.with_name("method_screen_large.py")])
    with open(out / "fingerprint_launch.txt", "a", encoding="utf-8") as fh:
        fh.write("== %s · part %s ==\n%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), a.part, fp))
    method = m7["method"]
    if a.part == "kcost":
        name = m7["large_task"]
        path = out / ("kcost__%s.json" % name)
        if not path.exists():
            res = json.loads((large_dir / (name + ".json")).read_text(encoding="utf-8"))
            rows = k_cost(P, ML.load_sens(priv, name), res, [int(k) for k in m7["k_cost"]], int(m7["k_cost_trees"]))
            write_json(path, {"task": name, "rows": rows})
            MS.log("%s · 单树开销完成" % name)
        return 0
    tasks = a.tasks or list(m7["%s_tasks" % a.part])
    for name in tasks:
        path = out / ("%s__%s.json" % (a.part, name))
        if path.exists():
            continue
        res = json.loads((run_dir / (name + ".json")).read_text(encoding="utf-8"))
        sc = MS.Screen(name, cfg, priv, out / "unused")
        if a.part == "timing":
            info = timing(P, sc, res, method, int(m7["timing_trees"]))
            MS.log("%s · 计时：每棵树 原版 %.1fs / 明文 %.1fs，节省 %s，打分一致 %s" % (
                name, info["sec_per_tree_base"], info["sec_per_tree_plain"], format(info["saving"], ".0%"),
                info["preds_identical"]))
        else:
            info = exact_check(P, sc, res, method, np.load(sgb_dir / ("%s__%s.npy" % (name, method))))
            MS.log("%s · 逐位比较：%s（不同 %d / %d 行，最大差 %.2e）" % (
                name, info["verdict"], info["rows_different"], info["rows"], info["max_abs_diff"]))
        write_json(path, {"task": name, "method": method, **info})
        del sc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
