"""M5 · S5.M8 Orange 大版流失真实 SGB 复核组件的单元测试。CI 没有 SecretFlow，用替身检验：
拟合时处于主动方明文直方图上下文、退出后恢复；已存档的方法不重拟合；时长估算；对比只在 SGB 内部与同配置代理之间。"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from modules.m5_modeling.components import method_screen as MS  # noqa: E402
from modules.m5_modeling.components import sgb_large_confirm as LC  # noqa: E402
from modules.m5_modeling.tests.test_method_screen import CFG  # noqa: E402
from modules.m5_modeling.tests.test_sgb_cost_screen import GE_PATH, _GradientEncryptor  # noqa: E402


@pytest.fixture()
def fake_ge(monkeypatch):
    """替身：只需要 gradient_encryptor 模块（明文直方图上下文替换的对象）。"""
    ge = types.ModuleType(GE_PATH + ".gradient_encryptor")
    ge.GradientEncryptor = _GradientEncryptor
    ge.move_config = lambda pyu, enc: pyu
    pkg = types.ModuleType(GE_PATH)
    pkg.gradient_encryptor = ge
    monkeypatch.setitem(sys.modules, GE_PATH, pkg)
    monkeypatch.setitem(sys.modules, GE_PATH + ".gradient_encryptor", ge)
    return ge


RES = {"configs": {"central": {"lr": 0.1, "depth": 3, "lam": 10.0, "col": 1.0, "sub": 1.0, "rounds": 94}, "m1": {}},
       "contrasts": []}


def test_时长估算按层数与棵数():
    P = MS.Params(CFG)
    assert abs(LC.estimate_seconds(P, RES, "F_opt_p", 40.0, 3) - 40.0 * 94) < 1e-9
    d = P.default_cfg
    assert abs(LC.estimate_seconds(P, RES, "F_def", 40.0, 3) - 40.0 * int(d["depth"]) / 3 * int(d["rounds"])) < 1e-9



def test_估计超过上限的方法不拟合():
    P = MS.Params(CFG)
    sec = 40.0
    pl = LC.plan(P, RES, ["F_opt_p", "F_def"], sec, 3, sec * 94)
    assert pl["todo"] == ["F_opt_p"]                                        # 原版：深度 5、100 棵，估计超过上限
    assert pl["estimate_seconds"]["F_opt_p"] == round(sec * 94, 1) and pl["cap_seconds"] == sec * 94
    assert LC.plan(P, RES, ["F_opt_p"], sec, 3, sec * 94 - 1)["todo"] == []


def test_拟合在明文上下文内_已存档不重拟合(fake_ge, monkeypatch, tmp_path):
    P = MS.Params(CFG)
    orig = _GradientEncryptor.cache_to_workers
    calls = []
    y = np.r_[np.ones(3), np.zeros(3)]

    def fake_confirm(P_, sc, res, method, out):
        calls.append((method, _GradientEncryptor.cache_to_workers is not orig))
        np.save(out / ("%s__%s.npy" % (sc.name, method)), np.linspace(0.1, 0.9, 6).astype(np.float32))
        return {"sgb_auroc": 0.5, "fit_seconds": 1.0, "trees": 3}

    seen = {}

    def fake_eval(P_, sc, res, sgb, proxy, pairs):
        seen.update(sgb=sorted(sgb), proxy=sorted(proxy), pairs=pairs)
        return {"methods": {}, "contrasts": {}}

    monkeypatch.setattr(LC.SG, "confirm", fake_confirm)
    monkeypatch.setattr(LC.SG, "proxy_pred", lambda P_, sc, res, m: np.zeros(6))
    monkeypatch.setattr(LC.SG, "evaluate", fake_eval)
    sc = types.SimpleNamespace(name="t", v={"final": {"ev_y": y}})
    r = LC.run(P, sc, RES, tmp_path)
    assert calls == [("F_def", True), ("F_opt_p", True)]                 # 两个方法都在明文直方图上下文里拟合
    assert _GradientEncryptor.cache_to_workers is orig                     # 退出后恢复
    assert seen == {"sgb": ["F_def", "F_opt_p"], "proxy": ["F_def", "F_opt_p"], "pairs": [("F_opt_p", "F_def")]}
    assert json.loads((tmp_path / "t__F_opt_p.json").read_text())["trees"] == 3 and r["task"] == "t"
    calls.clear()
    LC.run(P, sc, RES, tmp_path)
    assert calls == []                                                      # 已存档：续跑时不重拟合


def test_命令行入口_单个方法与对比_超出上限不拟合(monkeypatch, tmp_path):
    import yaml
    monkeypatch.chdir(ROOT)
    large, out = tmp_path / "large", tmp_path / "out"
    large.mkdir()
    (large / "t.json").write_text(json.dumps(RES), encoding="utf-8")
    m8 = yaml.safe_load((ROOT / "modules/m5_modeling/configs/s5_m8_orange_large_sgb.yaml").read_text(encoding="utf-8"))
    m8.update(task="t", large_dir_env="T_LARGE", out_dir_env="T_OUT", time_cap_sec=1e9)
    cfg_path = tmp_path / "m8.yaml"
    cfg_path.write_text(yaml.safe_dump(m8, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("T_LARGE", str(large))
    monkeypatch.setenv("T_OUT", str(out))
    monkeypatch.setenv(CFG["private_root_env"], str(tmp_path / "priv"))
    monkeypatch.setattr(LC.MS, "Screen", lambda name, cfg, priv, o: types.SimpleNamespace(name=name))
    fitted, seen = [], {}
    monkeypatch.setattr(LC, "fit_one", lambda P, sc, res, m, o: fitted.append(m) or {})
    monkeypatch.setattr(LC, "run", lambda P, sc, res, o, methods, pairs: seen.update(methods=methods, pairs=pairs)
                        or {"task": sc.name})
    assert LC.main(["--config", str(cfg_path), "--method", "F_def"]) == 0 and fitted == ["F_def"]
    assert LC.main(["--config", str(cfg_path)]) == 0
    assert seen == {"methods": ("F_opt_p", "F_def"), "pairs": [("F_opt_p", "F_def")]}
    assert json.loads((out / "t__plan.json").read_text(encoding="utf-8"))["todo"] == ["F_opt_p", "F_def"]
    assert json.loads((out / "t__summary.json").read_text(encoding="utf-8"))["task"] == "t"
    assert "evaluate" in (out / "fingerprint_launch.txt").read_text(encoding="utf-8")
    m8["time_cap_sec"] = 1.0
    cfg_path.write_text(yaml.safe_dump(m8, allow_unicode=True), encoding="utf-8")
    with pytest.raises(AssertionError):
        LC.main(["--config", str(cfg_path), "--method", "F_def"])
    assert LC.main(["--config", str(cfg_path)]) == 0 and seen == {"methods": (), "pairs": []}
