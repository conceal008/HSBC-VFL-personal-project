"""M5 · S5.M12 SGB 训练随机性组件的单元测试。CI 没有 SecretFlow，用替身检验：
存档查找顺序与路径、seed 拟合在明文上下文内、原版联邦确定性核对与方法选择、逐个 seed 存档与分段暂停、
两层 bootstrap 对比与汇总、命令行入口的两轮拟合（先优化版联邦、后原版联邦）与续跑。"""
from __future__ import annotations

import contextlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m5_modeling.components import method_screen as MS  # noqa: E402
from modules.m5_modeling.components import sgb_i2i3_confirm as IC  # noqa: E402
from modules.m5_modeling.components import sgb_seeds as SD  # noqa: E402
from modules.m5_modeling.tests.test_method_screen import CFG  # noqa: E402

Y = np.r_[np.ones(20), np.zeros(20)]


def _sc(name="t"):
    return types.SimpleNamespace(name=name, v={"final": {"ev_y": Y, "ev_cl": None}})


def test_存档查找顺序与路径(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    np.save(b / "t__F_def.npy", np.ones(3, np.float32))
    np.save(a / "t__F_opt_p.npy", np.zeros(3, np.float32))
    np.save(b / "t__F_opt_p.npy", np.ones(3, np.float32))
    assert SD.stored("t", "F_def", [a, b]).sum() == 3 and SD.stored("t", "F_opt_p", [a, b]).sum() == 0
    with pytest.raises(FileNotFoundError):
        SD.stored("u", "F_def", [a, b])
    assert SD.seed_path(Path("/x"), "t", "F_def", 22, "npy").name == "t__F_def__s22.npy"
    assert SD.probe_path(Path("/x"), "t", "F_opt_p").name == "t__F_opt_p__probe.json"


def test_seed拟合_确定性核对与方法选择(monkeypatch):
    P = MS.Params(CFG)
    state = {"on": False}

    @contextlib.contextmanager
    def fake_ctx(enabled=True):
        state["on"] = enabled
        yield
        state["on"] = False

    seen = []
    monkeypatch.setattr(SD.SC, "plaintext_label_holder", fake_ctx)
    monkeypatch.setattr(SD.SG, "run_sgb", lambda P_, *a, seed=None, **k: seen.append((seed, state["on"])) or (np.array([0.25, 0.5]), 2.0))
    monkeypatch.setattr(SD.SC, "task_args", lambda P_, sc, res, m: (None,) * 8)
    r = SD.determinism(P, _sc(), {}, np.array([0.25, 0.5]), 22)
    assert r["identical"] and r["max_abs_diff"] == 0.0 and r["n_rows_differ"] == 0
    assert seen == [(22, True)] and not state["on"]
    r = SD.determinism(P, _sc(), {}, np.array([0.25, 0.75]), 22)
    assert not r["identical"] and r["n_rows_differ"] == 1
    assert SD.methods_for({"identical": True}) == ["F_opt_p"] and SD.methods_for(r) == ["F_opt_p", "F_def"]


def test_逐个seed存档_分段暂停与跳过(monkeypatch, tmp_path):
    P = MS.Params(CFG)
    t = [0.0]
    seeds = [22, 33, 44]
    probed = []

    def ok_probe(P_, sc, res, m):
        probed.append(m)
        return {"identical_first_trees": True, "projected_fit_seconds": 10.0}

    monkeypatch.setattr(SD.IC, "probe", ok_probe)
    monkeypatch.setattr(SD.SC, "task_args", lambda P_, sc, res, m: (None,) * 8)
    fitted = []

    def fake_fit(P_, a, seed):
        fitted.append(seed)
        t[0] += 10.0
        return {"pred": np.full(4, seed, np.float32), "seed": seed, "fit_seconds": 10.0}

    monkeypatch.setattr(SD, "fit_seed", fake_fit)
    assert SD.status(tmp_path, "t", "F_def", seeds) is None
    assert SD.run_task(P, _sc(), {}, tmp_path, 1e3, IC.Segment(25.0, 1.0, clock=lambda: t[0]), 5.0, seeds, "F_def") == "paused"
    assert fitted == [22, 33] and probed == ["F_def"] and SD.status(tmp_path, "t", "F_def", seeds) is None
    assert SD.run_task(P, _sc(), {}, tmp_path, 1e3, IC.Segment(1e9, 1.0, clock=lambda: t[0]), 5.0, seeds, "F_def") == "done"
    assert fitted == seeds and probed == ["F_def"] and SD.status(tmp_path, "t", "F_def", seeds) == "done"
    assert SD.status(tmp_path, "t", "F_opt_p", seeds) is None              # 另一个方法独立存档
    monkeypatch.setattr(SD.IC, "probe", lambda P_, sc, res, m: {"identical_first_trees": True, "projected_fit_seconds": 5e3})
    assert SD.run_task(P, _sc("b"), {}, tmp_path, 1e3, IC.Segment(1e9, 1.0), 5.0, seeds, "F_opt_p") == "skipped"
    assert SD.status(tmp_path, "b", "F_opt_p", seeds) == "skipped"
    monkeypatch.setattr(SD.IC, "probe", lambda P_, sc, res, m: {"identical_first_trees": False, "projected_fit_seconds": 1.0})
    assert SD.run_task(P, _sc("c"), {}, tmp_path, 1e3, IC.Segment(1e9, 1.0), 5.0, seeds, "F_opt_p") == "skipped"
    assert SD.run_task(P, _sc("c"), {}, tmp_path, 1e3, IC.Segment(1e9, 1.0), 5.0, seeds, "F_opt_p") == "skipped"
    assert SD.run_task(P, _sc("d"), {}, tmp_path, 1e3, IC.Segment(1.0, 1.0), 5.0, seeds, "F_opt_p") == "paused"


def test_两层bootstrap对比与汇总(tmp_path):
    P = MS.Params(CFG)
    rng = np.random.default_rng(0)
    good = Y + rng.normal(0, 0.8, len(Y))
    weak = Y + rng.normal(0, 3.0, len(Y))
    c = SD.contrast(P, Y, None, [good, good + rng.normal(0, 0.1, len(Y))], [weak, weak + rng.normal(0, 1.5, len(Y))])
    assert c["diff"] > 0 and c["ci_low"] <= c["diff"] <= c["ci_high"] and c["opt_seed_sd"] > 0 and c["def_seed_sd"] > 0
    assert len(c["opt_aucs"]) == 2 and len(c["def_aucs"]) == 2
    one = SD.contrast(P, Y, None, [good], [weak])
    assert one["opt_seed_sd"] == 0.0 and one["def_seed_sd"] == 0.0
    assert abs(one["diff"] - (MS.auc(Y, good) - MS.auc(Y, weak))) < 1e-12
    for m, arr in (("F_opt_p", good), ("F_def", weak)):
        np.save(tmp_path / ("t__%s.npy" % m), arr.astype(np.float32))
        for sd in (22, 33):
            np.save(SD.seed_path(tmp_path, "t", m, sd, "npy"), (arr + sd * 1e-3).astype(np.float32))
            SD.seed_path(tmp_path, "t", m, sd, "json").write_text(json.dumps({"seed": sd, "fit_seconds": 3.0}), encoding="utf-8")
    res = {"contrasts": [{"a": "F_opt_p", "b": "F_def", "diff": 0.1, "ci_low": 0.05}]}
    ev = SD.evaluate_task(P, _sc(), res, tmp_path, [tmp_path], [22, 33], ["F_opt_p", "F_def"])
    assert len(ev["six_seed"]["opt_aucs"]) == 3 and len(ev["six_seed"]["def_aucs"]) == 3 and ev["proxy_6seed"]["diff"] == 0.1
    assert ev["fit_seconds_new_seeds"] == {"F_opt_p": 6.0, "F_def": 6.0} and len(ev["seed11"]["def_aucs"]) == 1
    only_opt = SD.evaluate_task(P, _sc(), res, tmp_path, [tmp_path], [22, 33], ["F_opt_p"])
    assert len(only_opt["six_seed"]["def_aucs"]) == 1 and only_opt["fit_seconds_new_seeds"] == {"F_opt_p": 6.0}
    e2 = {**ev, "holds_six": False}
    s = SD.summarize({"t": ev, "u": e2}, {"identical": False}, {"v": "F_opt_p：" + IC.OVER_CAP})
    assert s["n"] == 2 and s["changed"] == (["u"] if ev["holds_seed11"] else []) and s["determinism"] == {"identical": False}
    assert s["tasks"][0]["def_seed_sd"] == ev["six_seed"]["def_seed_sd"]


def test_命令行入口_确定性核对_两轮拟合_分段续跑与汇总(monkeypatch, tmp_path):
    import yaml
    monkeypatch.chdir(ROOT)
    dirs = {k: tmp_path / k for k in ("priv", "run", "m9", "sgb", "large", "out")}
    for d in dirs.values():
        d.mkdir()
    m12 = yaml.safe_load((ROOT / "modules/m5_modeling/configs/s5_m12_sgb_seeds.yaml").read_text(encoding="utf-8"))
    m12.update(m9_dir_env="T_M9", sgb_dir_env="T_SGB", large_dir_env="T_LARGE", out_dir_env="T_OUT",
               determinism_task="a", tasks=[{"name": "a"}, {"name": "L", "large": True}, {"name": "c"}])
    cfg_path = tmp_path / "m12.yaml"
    cfg_path.write_text(yaml.safe_dump(m12, allow_unicode=True), encoding="utf-8")
    for env, d in (("T_M9", "m9"), ("T_SGB", "sgb"), ("T_LARGE", "large"), ("T_OUT", "out"),
                   (CFG["private_root_env"], "priv"), (CFG["run_dir_env"], "run")):
        monkeypatch.setenv(env, str(dirs[d]))
    for d, n in (("run", "a"), ("large", "L"), ("run", "c")):
        (dirs[d] / (n + ".json")).write_text(json.dumps({"src": d}), encoding="utf-8")
    np.save(dirs["sgb"] / "a__F_def.npy", np.zeros(2, np.float32))
    monkeypatch.setattr(SD.MS, "Screen", lambda name, cfg, priv, out: _sc(name))
    monkeypatch.setattr(SD, "determinism", lambda P_, sc, res, ref, seed: {"task": sc.name, "seed": seed, "identical": False})
    calls, budget = [], {"left": 1}

    def fake_run_task(P_, sc, res, out, cap, seg, min_probe, seeds, method):
        calls.append((sc.name, res["src"], method, tuple(seeds)))
        if sc.name == "L" and method == "F_def" and budget["left"]:
            budget["left"] -= 1
            return "paused"
        if sc.name == "c":
            SD.SC.write_json(SD.probe_path(out, "c", method), {"verdict": IC.OVER_CAP})
            return "skipped"
        for sd in seeds:
            SD.seed_path(out, sc.name, method, sd, "json").write_text("{}", encoding="utf-8")
        return "done"

    evaluated = []
    monkeypatch.setattr(SD, "run_task", fake_run_task)
    monkeypatch.setattr(SD, "evaluate_task", lambda P_, sc, res, out, dirs_, seeds, methods: evaluated.append(
        (sc.name, tuple(methods))) or {
        "task": sc.name, "six_seed": {"diff": 0.01, "ci_low": 0.001, "ci_high": 0.02, "p": 0.01, "opt_seed_sd": 0.001,
                                      "def_seed_sd": 0.0005},
        "seed11": {"diff": 0.01, "ci_low": 0.002}, "holds_six": True, "holds_seed11": True})
    assert SD.main(["--config", str(cfg_path)]) == 0
    assert [c[:3] for c in calls] == [("a", "run", "F_opt_p"), ("L", "large", "F_opt_p"), ("c", "run", "F_opt_p"),
                                      ("a", "run", "F_def"), ("L", "large", "F_def")]
    assert calls[0][3] == (22, 33, 44, 55, 66) and not evaluated
    assert json.loads((dirs["out"] / "determinism.json").read_text(encoding="utf-8"))["seed"] == 22
    assert SD.main(["--config", str(cfg_path)]) == 0
    assert [c[:3] for c in calls[5:]] == [("L", "large", "F_def")]             # 已完成与已判定不做的不再读数据
    assert evaluated == [("a", ("F_opt_p", "F_def")), ("L", ("F_opt_p", "F_def"))]
    s = json.loads((dirs["out"] / "summary.json").read_text(encoding="utf-8"))
    assert s["n"] == 2 and s["skipped"] == {"c": "F_opt_p：" + IC.OVER_CAP} and not s["determinism"]["identical"]
    assert SD.main(["--config", str(cfg_path)]) == 0 and len(calls) == 6 and len(evaluated) == 2   # 全部存档：只重写汇总
    (dirs["out"] / "determinism.json").unlink()
    m12["determinism_budget_sec"] = 1e9
    m12["segment_budget_sec"] = 1.0
    cfg_path.write_text(yaml.safe_dump(m12, allow_unicode=True), encoding="utf-8")
    assert SD.main(["--config", str(cfg_path)]) == 0 and not (dirs["out"] / "determinism.json").exists()   # 预算不够：本段不做核对
