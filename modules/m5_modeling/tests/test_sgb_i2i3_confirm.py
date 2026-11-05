"""M5 · S5.M9 I2 + I3 真实 SGB 复核组件的单元测试。CI 没有 SecretFlow，用替身检验：
抽查（逐位比较与外推）、计划与状态、复用存档、对比与跨任务汇总、分段预算、单个任务的编排与命令行入口的续跑。"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m5_modeling.components import method_screen as MS  # noqa: E402
from modules.m5_modeling.components import sgb_cost_screen as SC  # noqa: E402
from modules.m5_modeling.components import sgb_i2i3_confirm as IC  # noqa: E402
from modules.m5_modeling.tests.test_method_screen import CFG  # noqa: E402

KEY = "F_opt_p − F_def"


def _args(rounds, depth=3):
    return (None,) * SC.ARG_CFG + ({"rounds": rounds, "depth": depth}, 33, 0.0)


def _fake_fit(sec_per_tree, base_pred=None):
    def fit(P_, args, plain, rounds=None):
        pred = np.arange(3, dtype=np.float32) if (plain or base_pred is None) else base_pred
        return {"pred": pred, "fit_seconds": 5.0 + sec_per_tree * rounds, "trees": rounds, "parts": {}}
    return fit


def test_抽查_逐位比较与外推(monkeypatch):
    P = MS.Params(CFG)
    monkeypatch.setattr(IC.SC, "task_args", lambda P_, sc, res, m: _args(100))
    monkeypatch.setattr(IC.SC, "fit", _fake_fit(10.0))
    r = IC.probe(P, None, {}, "F_def")
    assert r["identical_first_trees"] and abs(r["marginal_sec_per_tree"] - 10.0) < 1e-12
    assert abs(r["projected_fit_seconds"] - (25.0 + 10.0 * 98)) < 1e-9 and r["rounds"] == 100
    monkeypatch.setattr(IC.SC, "fit", _fake_fit(10.0, base_pred=np.array([0, 1, 3], np.float32)))
    assert not IC.probe(P, None, {}, "F_def")["identical_first_trees"]
    monkeypatch.setattr(IC.SC, "task_args", lambda P_, sc, res, m: _args(3))
    monkeypatch.setattr(IC.SC, "fit", _fake_fit(10.0))
    assert abs(IC.probe(P, None, {}, "F_def")["projected_fit_seconds"] - 45.0 * 3 / 4) < 1e-9   # 棵数不超过 4 时按比例


def test_计划_状态与复用存档(tmp_path):
    assert IC.plan_task({"name": "a"}) == ["F_def", "F_opt_p"]
    assert IC.plan_task({"name": "a", "reuse_opt": True}) == ["F_def"]
    assert IC.status_of("a", "F_def", tmp_path) is None
    (tmp_path / "a__F_def__probe.json").write_text(json.dumps({"verdict": IC.OVER_CAP}), encoding="utf-8")
    assert IC.status_of("a", "F_def", tmp_path) == IC.OVER_CAP
    (tmp_path / "a__F_def.json").write_text("{}", encoding="utf-8")
    assert IC.status_of("a", "F_def", tmp_path) == "fitted"
    sgb_dir = tmp_path / "sgb"
    sgb_dir.mkdir()
    np.save(sgb_dir / "a__F_opt_p.npy", np.array([0.5, 0.25], np.float32))
    np.save(tmp_path / "a__F_def.npy", np.array([0.1, 0.2], np.float32))
    s = IC.scores("a", tmp_path, sgb_dir, {"name": "a", "reuse_opt": True})
    assert np.array_equal(s["F_opt_p"], [0.5, 0.25]) and (tmp_path / "a__F_opt_p.npy").exists()


def _ev(diff, lo, p, px_lo, d_def=0.001, d_opt=0.002):
    m = {k: {"sgb_auroc": 0.8, "proxy_auroc": 0.8, "sgb_minus_proxy": {"diff": d, "ci_low": -1, "ci_high": 1, "p": 1}}
         for k, d in (("F_def", d_def), ("F_opt_p", d_opt))}
    return {"methods": m, "contrasts": {KEY: {"sgb": {"diff": diff, "ci_low": lo, "ci_high": diff + 0.01, "p": p},
                                              "proxy_6seed": {"diff": diff, "ci_low": px_lo, "ci_high": 1}}}}


def test_对比_描述与跨任务汇总(monkeypatch):
    P = MS.Params(CFG)
    seen = {}
    monkeypatch.setattr(IC.SG, "proxy_pred", lambda P_, sc, res, m: np.zeros(2))
    monkeypatch.setattr(IC.SG, "evaluate", lambda P_, sc, res, sgb, proxy, pairs: seen.update(pairs=pairs) or _ev(0.02, 0.01, 0.0, 0.01))
    ev = IC.evaluate_task(P, None, {"test": {"G_t": {"auroc": 0.75}}}, {"F_def": np.zeros(2), "F_opt_p": np.zeros(2)})
    assert seen["pairs"] == [IC.PAIR] and abs(ev["descriptive"]["F_opt_p"]["sgb_minus_guest_only_proxy"] - 0.05) < 1e-12
    s = IC.summarize({"a": _ev(0.02, 0.01, 0.01, 0.01), "b": _ev(0.004, -0.001, 0.2, 0.001, d_opt=0.006),
                      "c": _ev(0.0, -0.01, 0.9, -0.02)}, {"d": {"F_def": IC.OVER_CAP}}, 0.005)
    assert (s["n"], s["sgb_holds"], s["proxy_holds"], s["agree"]) == (3, 1, 2, 2)
    assert [r["sgb_p_holm"] for r in s["tasks"]] == pytest.approx([0.03, 0.4, 0.9])
    assert s["outside_tolerance"] == [{"task": "b", "method": "F_opt_p", "sgb_minus_proxy": 0.006}]
    assert s["skipped"] == {"d": {"F_def": IC.OVER_CAP}}


def test_分段预算():
    t = [0.0]
    seg = IC.Segment(100.0, 1.1, clock=lambda: t[0])
    assert seg.allows(90.0) and not seg.allows(91.0)
    t[0] = 50.0
    assert seg.elapsed() == 50.0 and seg.allows(45.0) and not seg.allows(46.0)


def test_单个任务_抽查拟合暂停与跳过(monkeypatch, tmp_path):
    P = MS.Params(CFG)
    t = [0.0]
    fitted = []
    proj = {"F_def": 50.0, "F_opt_p": 60.0}
    monkeypatch.setattr(IC, "probe", lambda P_, sc, res, m: {"identical_first_trees": True, "projected_fit_seconds": proj[m]})
    monkeypatch.setattr(IC.LC, "fit_one", lambda P_, sc, res, m, out: fitted.append(m) or t.__setitem__(0, t[0] + proj[m])
                        or (out / ("%s__%s.json" % (sc.name, m))).write_text("{}", encoding="utf-8"))
    sc = types.SimpleNamespace(name="a")
    seg = IC.Segment(100.0, 1.0, clock=lambda: t[0])
    assert IC.run_task(P, sc, {}, {"name": "a"}, tmp_path, 1000.0, seg, 10.0) == "paused"   # 拟合完 F_def 已用 50，再拟合 F_opt_p 需 60 > 剩余 50
    assert fitted == ["F_def"]
    seg2 = IC.Segment(100.0, 1.0, clock=lambda: t[0])
    assert IC.run_task(P, sc, {}, {"name": "a"}, tmp_path, 1000.0, seg2, 10.0) == "done" and fitted == ["F_def", "F_opt_p"]
    proj["F_def"] = 2000.0
    sc_b = types.SimpleNamespace(name="b")
    assert IC.run_task(P, sc_b, {}, {"name": "b"}, tmp_path, 1000.0, IC.Segment(1e9, 1.0), 10.0) == "skipped"
    assert json.loads((tmp_path / "b__F_def__probe.json").read_text(encoding="utf-8"))["verdict"] == IC.OVER_CAP
    monkeypatch.setattr(IC, "probe", lambda P_, sc, res, m: {"identical_first_trees": False, "projected_fit_seconds": 1.0})
    assert IC.run_task(P, types.SimpleNamespace(name="c"), {}, {"name": "c"}, tmp_path, 1000.0, IC.Segment(1e9, 1.0), 10.0) == "skipped"
    assert IC.status_of("c", "F_def", tmp_path) == IC.NOT_IDENTICAL
    assert IC.run_task(P, sc_b, {}, {"name": "b"}, tmp_path, 1000.0, IC.Segment(1e9, 1.0), 10.0) == "skipped"   # 已判定不做
    assert IC.run_task(P, types.SimpleNamespace(name="d"), {}, {"name": "d"}, tmp_path, 1000.0, IC.Segment(5.0, 1.0), 10.0) == "paused"


def test_命令行入口_分段续跑与汇总(monkeypatch, tmp_path):
    import yaml
    monkeypatch.chdir(ROOT)
    dirs = {k: tmp_path / k for k in ("priv", "run", "sgb", "large", "out")}
    for d in dirs.values():
        d.mkdir()
    m9 = yaml.safe_load((ROOT / "modules/m5_modeling/configs/s5_m9_sgb_i2i3.yaml").read_text(encoding="utf-8"))
    m9.update(sgb_dir_env="T_SGB", large_dir_env="T_LARGE", out_dir_env="T_OUT",
              tasks=[{"name": "a", "reuse_opt": True}, {"name": "b"}, {"name": "L", "large": True}])
    cfg_path = tmp_path / "m9.yaml"
    cfg_path.write_text(yaml.safe_dump(m9, allow_unicode=True), encoding="utf-8")
    for env, d in (("T_SGB", "sgb"), ("T_LARGE", "large"), ("T_OUT", "out"), (CFG["private_root_env"], "priv"),
                   (CFG["run_dir_env"], "run")):
        monkeypatch.setenv(env, str(dirs[d]))
    for d, n in (("run", "a"), ("run", "b"), ("large", "L")):
        (dirs[d] / (n + ".json")).write_text(json.dumps({"src": d}), encoding="utf-8")
    np.save(dirs["sgb"] / "a__F_opt_p.npy", np.zeros(2, np.float32))
    monkeypatch.setattr(IC.MS, "Screen", lambda name, cfg, priv, out: types.SimpleNamespace(name=name))
    calls, budget = [], {"left": 1}

    def fake_run_task(P_, sc, res, task, out, cap, seg, min_probe):
        calls.append((sc.name, res["src"]))
        if sc.name == "b" and budget["left"]:
            budget["left"] -= 1
            return "paused"
        if sc.name == "L":
            (out / "L__F_def__probe.json").write_text(json.dumps({"verdict": IC.OVER_CAP}), encoding="utf-8")
            return "skipped"
        for m in IC.plan_task(task):
            np.save(out / ("%s__%s.npy" % (sc.name, m)), np.zeros(2, np.float32))
            (out / ("%s__%s.json" % (sc.name, m))).write_text("{}", encoding="utf-8")
        return "done"

    monkeypatch.setattr(IC, "run_task", fake_run_task)
    monkeypatch.setattr(IC, "evaluate_task", lambda P_, sc, res, sgb: _ev(0.02, 0.01, 0.01, 0.01))
    assert IC.main(["--config", str(cfg_path)]) == 0
    assert calls == [("a", "run"), ("b", "run")] and not (dirs["out"] / "summary.json").exists()   # 第 1 段在 b 暂停
    assert IC.main(["--config", str(cfg_path)]) == 0
    assert calls[2:] == [("b", "run"), ("L", "large")]                                          # 已完成的 a 不再跑
    s = json.loads((dirs["out"] / "summary.json").read_text(encoding="utf-8"))
    assert s["n"] == 2 and s["skipped"] == {"L": {"F_def": IC.OVER_CAP}}
    assert json.loads((dirs["out"] / "a__eval.json").read_text(encoding="utf-8"))["reuse_opt"] is True
    assert (dirs["out"] / "fingerprint_launch.txt").read_text(encoding="utf-8").count("== ") == 2
    IC.main(["--config", str(cfg_path)])
    assert len(calls) == 4                                                                      # 第 3 段：全部已完成或已判定，不再拟合
