"""M5 · S5.M11 M6 真实 SGB 复核组件的单元测试。CI 没有 SecretFlow，用替身检验：
成员输入、抽查与外推、成员拟合在明文上下文内、逐个成员存档与分段暂停、成员平均与对比、跨任务汇总、命令行入口的续跑。"""
from __future__ import annotations

import contextlib
import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m5_modeling.components import method_screen as MS  # noqa: E402
from modules.m5_modeling.components import sgb_cost_screen as SC  # noqa: E402
from modules.m5_modeling.components import sgb_i2i3_confirm as IC  # noqa: E402
from modules.m5_modeling.components import sgb_m6_more as M6  # noqa: E402
from modules.m5_modeling.tests.test_method_screen import CFG  # noqa: E402

KEY = "F_M6 − F_opt_p"


def _sc(name="t"):
    fv = {k: k for k in ("fit_xg", "fit_xh", "fit_y", "ev_xg", "ev_xh")}
    fv["ev_y"] = np.array([0.0, 1.0, 0.0, 1.0])
    return types.SimpleNamespace(name=name, v={"final": fv})


def test_成员输入_抽查与外推(monkeypatch):
    P = MS.Params(CFG)
    monkeypatch.setattr(M6.SG, "inputs", lambda P_, sc, res, m: ({"rounds": 50, "sub": 0.8}, 33, 0.1, 0.2, None, None))
    args = M6.member_args(P, _sc(), {})
    assert args[:5] == ("fit_xg", "fit_xh", "fit_y", "ev_xg", "ev_xh") and args[SC.ARG_CFG]["sub"] == 0.8 and args[-1] == 0.1
    monkeypatch.setattr(M6.SC, "fit", lambda P_, a, plain, rounds=None: {"pred": np.arange(3, dtype=np.float32),
                                                                         "fit_seconds": 5.0 + 2.0 * rounds})
    pr = M6.probe(P, args)
    assert pr["identical_first_trees"] and abs(pr["projected_member_seconds"] - (9.0 + 2.0 * 48)) < 1e-9
    assert M6.member_path(Path("/x"), "t", 3, "npy").name == "t__F_M6_m3.npy"


def test_成员拟合在明文上下文内(monkeypatch):
    P = MS.Params(CFG)
    state = {"on": False}

    @contextlib.contextmanager
    def fake_ctx(enabled=True):
        state["on"] = enabled
        yield
        state["on"] = False

    seen = []
    monkeypatch.setattr(M6.SC, "plaintext_label_holder", fake_ctx)
    monkeypatch.setattr(M6.SG, "run_sgb", lambda P_, *a, seed=None, **k: seen.append((seed, state["on"])) or (np.array([0.25, 0.5]), 3.14))
    r = M6.fit_member(P, (None,) * 8, 1101)
    assert seen == [(1101, True)] and not state["on"] and r["pred"].dtype == np.float32 and r["fit_seconds"] == 3.1


def test_逐个成员存档_分段暂停与跳过(monkeypatch, tmp_path):
    P = MS.Params(CFG)
    t = [0.0]
    monkeypatch.setattr(M6, "member_args", lambda P_, sc, res: (None,) * 8)
    monkeypatch.setattr(M6, "probe", lambda P_, a: {"identical_first_trees": True, "projected_member_seconds": 10.0})
    fitted = []

    def fake_fit(P_, a, seed):
        fitted.append(seed)
        t[0] += 10.0
        return {"pred": np.full(4, seed, np.float32), "seed": seed, "fit_seconds": 10.0}

    monkeypatch.setattr(M6, "fit_member", fake_fit)
    seg = IC.Segment(35.0, 1.0, clock=lambda: t[0])
    assert M6.run_task(P, _sc(), {}, tmp_path, 1000.0, seg, 5.0) == "paused" and fitted == M6.SG.bag_seeds(P)[:3]
    seg2 = IC.Segment(1e9, 1.0, clock=lambda: t[0])
    assert M6.run_task(P, _sc(), {}, tmp_path, 1000.0, seg2, 5.0) == "done" and fitted == M6.SG.bag_seeds(P)
    ms = M6.members(P, "t", tmp_path)
    assert [m["seed"] for m in ms] == M6.SG.bag_seeds(P) and ms[0]["pred"].dtype == float
    monkeypatch.setattr(M6, "probe", lambda P_, a: {"identical_first_trees": True, "projected_member_seconds": 5000.0})
    assert M6.run_task(P, _sc("b"), {}, tmp_path, 1000.0, IC.Segment(1e9, 1.0), 5.0) == "skipped"
    assert M6.run_task(P, _sc("b"), {}, tmp_path, 1000.0, IC.Segment(1e9, 1.0), 5.0) == "skipped"   # 已判定不做
    monkeypatch.setattr(M6, "probe", lambda P_, a: {"identical_first_trees": False, "projected_member_seconds": 1.0})
    assert M6.run_task(P, _sc("c"), {}, tmp_path, 1000.0, IC.Segment(1e9, 1.0), 5.0) == "skipped"
    assert json.loads((tmp_path / "c__F_M6__probe.json").read_text(encoding="utf-8"))["verdict"] == IC.NOT_IDENTICAL
    assert M6.run_task(P, _sc("d"), {}, tmp_path, 1000.0, IC.Segment(1.0, 1.0), 5.0) == "paused"


def _ev(diff, lo, px_lo, d_m6=0.001):
    m = {k: {"sgb_auroc": 0.8, "proxy_auroc": 0.8, "sgb_minus_proxy": {"diff": d, "ci_low": -1, "ci_high": 1, "p": 1}}
         for k, d in (("F_M6", d_m6), ("F_opt_p", 0.0))}
    return {"methods": m, "contrasts": {KEY: {"sgb": {"diff": diff, "ci_low": lo, "ci_high": diff + 0.01, "p": 0.01},
                                              "proxy_6seed": {"diff": diff, "ci_low": px_lo, "ci_high": 1}}}}


def test_成员平均对比与汇总(monkeypatch, tmp_path):
    P = MS.Params(CFG)
    for j, sd in enumerate(M6.SG.bag_seeds(P)):
        np.save(M6.member_path(tmp_path, "t", j, "npy"), np.full(4, j, np.float32))
        M6.member_path(tmp_path, "t", j, "json").write_text(json.dumps({"seed": sd, "fit_seconds": 2.0}), encoding="utf-8")
    opt = tmp_path / "opt"
    opt.mkdir()
    np.save(opt / "t__F_opt_p.npy", np.array([0.1, 0.9, 0.2, 0.8], np.float32))
    seen = {}
    monkeypatch.setattr(M6.SG, "proxy_pred", lambda P_, sc, res, m: np.zeros(4))
    monkeypatch.setattr(M6.SG, "evaluate", lambda P_, sc, res, sgb, proxy, pairs: seen.update(
        m6=sgb["F_M6"].copy(), pairs=pairs) or _ev(0.004, 0.001, 0.002))
    ev = M6.evaluate_task(P, _sc(), {}, tmp_path, opt)
    assert np.allclose(seen["m6"], np.full(4, 2.5)) and seen["pairs"] == [M6.PAIR]
    assert len(ev["members"]) == P.bag and ev["fit_seconds_total"] == 2.0 * P.bag
    s = M6.summarize({"a": _ev(0.004, 0.001, 0.002), "b": _ev(0.002, -0.001, 0.001, d_m6=0.006)}, {"c": IC.OVER_CAP}, 0.005)
    assert (s["n"], s["sgb_holds"], s["agree"]) == (2, 1, 1) and s["outside_tolerance"] == ["b"]
    assert s["skipped"] == {"c": IC.OVER_CAP}


def test_命令行入口_分段续跑与汇总(monkeypatch, tmp_path):
    import yaml
    monkeypatch.chdir(ROOT)
    dirs = {k: tmp_path / k for k in ("priv", "run", "opt", "out")}
    for d in dirs.values():
        d.mkdir()
    m11 = yaml.safe_load((ROOT / "modules/m5_modeling/configs/s5_m11_sgb_m6.yaml").read_text(encoding="utf-8"))
    m11.update(opt_dir_env="T_OPT", out_dir_env="T_OUT", tasks=["a", "b", "c"])
    cfg_path = tmp_path / "m11.yaml"
    cfg_path.write_text(yaml.safe_dump(m11, allow_unicode=True), encoding="utf-8")
    for env, d in (("T_OPT", "opt"), ("T_OUT", "out"), (CFG["private_root_env"], "priv"), (CFG["run_dir_env"], "run")):
        monkeypatch.setenv(env, str(dirs[d]))
    for n in ("a", "b", "c"):
        (dirs["run"] / (n + ".json")).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(M6.MS, "Screen", lambda name, cfg, priv, out: _sc(name))
    calls, budget = [], {"left": 1}

    def fake_run_task(P_, sc, res, out, cap, seg, min_probe):
        calls.append(sc.name)
        if sc.name == "b" and budget["left"]:
            budget["left"] -= 1
            return "paused"
        if sc.name == "c":
            (out / "c__F_M6__probe.json").write_text(json.dumps({"verdict": IC.OVER_CAP}), encoding="utf-8")
            return "skipped"
        return "done"

    monkeypatch.setattr(M6, "run_task", fake_run_task)
    monkeypatch.setattr(M6, "evaluate_task", lambda P_, sc, res, out, opt: _ev(0.004, 0.001, 0.002))
    assert M6.main(["--config", str(cfg_path)]) == 0 and calls == ["a", "b"]
    assert not (dirs["out"] / "summary.json").exists()
    assert M6.main(["--config", str(cfg_path)]) == 0 and calls[2:] == ["b", "c"]
    s = json.loads((dirs["out"] / "summary.json").read_text(encoding="utf-8"))
    assert s["n"] == 2 and s["skipped"] == {"c": IC.OVER_CAP}
    assert (dirs["out"] / "fingerprint_launch.txt").read_text(encoding="utf-8").count("== ") == 2
