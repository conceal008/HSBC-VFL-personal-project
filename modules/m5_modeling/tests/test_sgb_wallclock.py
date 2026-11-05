"""M5 · S5.M10 两进程墙钟组件的单元测试。CI 没有 SecretFlow，用替身检验：
集群配置、按方拆输入、两进程启动与计时、预热 + ABBA 编排、汇总、命令行入口的续跑与私有输入清理。"""
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
from modules.m5_modeling.components import sgb_wallclock as WC  # noqa: E402
from modules.m5_modeling.tests.test_method_screen import CFG  # noqa: E402

RES = {"configs": {"central": {"lr": 0.1, "depth": 3, "lam": 10.0, "col": 1.0, "sub": 1.0, "rounds": 7}, "m1": {}}}


def _sc(name="t"):
    fv = {"fit_xg": np.zeros((6, 2)), "fit_xh": np.ones((6, 3)), "fit_y": np.r_[np.ones(2), np.zeros(4)],
          "ev_xg": np.zeros((4, 2)), "ev_xh": np.ones((4, 3))}
    return types.SimpleNamespace(name=name, v={"final": fv})


def test_集群配置与按方拆输入(tmp_path):
    c = WC.cluster_config("hk", [7001, 7002])
    assert c["self_party"] == "hk" and c["parties"]["neidi"]["address"].endswith(":7001")
    spec = WC.prepare(MS.Params(CFG), _sc(), RES, "F_opt_p", tmp_path)
    d = np.load(spec["data"])
    assert set(d.files) == {"xg", "xh", "y", "eg", "eh"} and d["xh"].shape == (6, 3)
    assert spec["cfg"]["rounds"] == 7 and spec["bins"] == MS.Params(CFG).bins_fine and spec["cols_host"] == 3
    assert abs(spec["base"] - float(MS.logit(np.array([MS.prior_fn(MS.Params(CFG))(d["y"])]))[0])) < 1e-12
    assert WC.save_pred(np.array([[0.5], [0.25]]), str(tmp_path / "p.npy")) == 1
    assert np.array_equal(np.load(tmp_path / "p.npy"), [0.5, 0.25])
    with pytest.raises(AssertionError):
        WC.prepare(MS.Params(CFG), _sc(), {"configs": {"central": {}, "m1": {"rounds": 3}}}, "F_M1", tmp_path)


def test_两进程启动与计时(tmp_path):
    cmds = []

    class FakeProc:
        def __init__(self, cmd, stdout=None, stderr=None):
            cmds.append(cmd)
            party, tag = cmd[cmd.index("--party") + 1], cmd[cmd.index("--tag") + 1]
            (tmp_path / ("%s__%s.json" % (tag, party))).write_text(json.dumps({"fit_seconds": 3.0}), encoding="utf-8")

        def wait(self):
            return 0

    tx = iter([100, 350])
    r = WC.launch(tmp_path / "s.json", True, 5, tmp_path, "x", [1, 2], popen=FakeProc, read_tx=lambda: next(tx))
    assert [c[c.index("--party") + 1] for c in cmds] == list(WC.PARTIES) and "--plain" in cmds[0]
    assert cmds[0][cmds[0].index("--plain") + 1] == "1" and r["loopback_tx_bytes"] == 250
    assert r["fit_seconds"] == {"neidi": 3.0, "hk": 3.0} and r["plain"] and r["trees"] == 5

    class Fail(FakeProc):
        def wait(self):
            return 1

    with pytest.raises(AssertionError):
        WC.launch(tmp_path / "s.json", False, 5, tmp_path, "y", [1, 2], popen=Fail, read_tx=lambda: 0)


def test_预热加ABBA与汇总(tmp_path):
    seen = []

    def fake_run(spec_path, plain, rounds, out, tag, ports):
        seen.append((plain, rounds))
        np.save(out / ("%s__pred.npy" % tag), np.arange(3.0))
        sec = (6.0 if plain else 10.0) * rounds
        return {"tag": tag, "plain": plain, "trees": rounds, "fit_seconds": {"neidi": sec, "hk": sec + 1.0},
                "launch_wall_seconds": sec + 2.0, "loopback_tx_bytes": 1000 if plain else 1010}

    spec = {"cfg": {"rounds": 7}, "cols_guest": 2, "cols_host": 3, "rows": 6}
    out = WC.timing(tmp_path / "s.json", spec, "t", 10, tmp_path, [1, 2], run=fake_run)
    assert seen == [(False, SC.WARMUP_TREES)] + [(k == "plain", 7) for k in SC.ORDER]
    assert out["trees"] == 7 and out["preds_identical"] and abs(out["saving"] - 0.4) < 1e-12
    assert out["saving_range"] == [pytest.approx(0.4)] * 2 and abs(out["hk_wall_per_tree_plain"] - (6.0 + 1.0 / 7)) < 1e-12
    assert out["loopback_tx_bytes_base"] == 1010 and out["loopback_tx_bytes_plain"] == 1000


def test_命令行入口_续跑与清理(monkeypatch, tmp_path):
    import yaml
    monkeypatch.chdir(ROOT)
    dirs = {k: tmp_path / k for k in ("priv", "run", "large", "out")}
    for d in dirs.values():
        d.mkdir()
    m10 = yaml.safe_load((ROOT / "modules/m5_modeling/configs/s5_m10_wallclock.yaml").read_text(encoding="utf-8"))
    m10.update(large_dir_env="T_LARGE", out_dir_env="T_OUT", tasks=[{"name": "a"}, {"name": "L", "large": True, "trees": 3}])
    cfg_path = tmp_path / "m10.yaml"
    cfg_path.write_text(yaml.safe_dump(m10, allow_unicode=True), encoding="utf-8")
    for env, d in (("T_LARGE", "large"), ("T_OUT", "out"), (CFG["private_root_env"], "priv"), (CFG["run_dir_env"], "run")):
        monkeypatch.setenv(env, str(dirs[d]))
    (dirs["run"] / "a.json").write_text(json.dumps(RES), encoding="utf-8")
    (dirs["large"] / "L.json").write_text(json.dumps(RES), encoding="utf-8")
    monkeypatch.setattr(WC.MS, "Screen", lambda name, cfg, priv, out: _sc(name))
    calls = []

    def fake_timing(spec_path, spec, name, max_trees, out, ports):
        calls.append((name, max_trees, Path(spec["data"]).exists()))
        return {"wall_per_tree_base": 2.0, "wall_per_tree_plain": 1.0, "saving": 0.5, "preds_identical": True}

    monkeypatch.setattr(WC, "timing", fake_timing)
    assert WC.main(["run", "--config", str(cfg_path)]) == 0
    assert calls == [("a", m10["timing_trees"], True), ("L", 3, True)]
    assert not (dirs["out"] / "a__inputs.npz").exists()                       # 私有输入副本用完即删
    assert json.loads((dirs["out"] / "timing__L.json").read_text(encoding="utf-8"))["saving"] == 0.5
    WC.main(["run", "--config", str(cfg_path)])
    assert len(calls) == 2                                                    # 已完成的任务跳过
    got = {}
    monkeypatch.setattr(WC, "run_party", lambda party, spec, plain, rounds, out, tag, ports: got.update(
        party=party, plain=plain, rounds=rounds, tag=tag, ports=ports, spec=spec))
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"k": 1}), encoding="utf-8")
    assert WC.main(["party", "--party", "hk", "--spec", str(spec_path), "--plain", "1", "--rounds", "4",
                    "--out", str(tmp_path), "--tag", "z", "--ports", "5,6"]) == 0
    assert got == {"party": "hk", "plain": True, "rounds": 4, "tag": "z", "ports": [5, 6], "spec": {"k": 1}}
