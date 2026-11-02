"""M5 · S5.M2 真实 SGB 复核组件的单元测试。CI 没有 SecretFlow / xgboost，用替身检验编排、逐样本初始预测的注入与恢复、
M6 成员续跑与配对评估。"""
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
from modules.m5_modeling.components import method_screen_sgb as SG  # noqa: E402
from modules.m5_modeling.tests.test_method_screen import (  # noqa: E402
    CFG, fake_xgb, small_cfg, write_synthetic)

assert fake_xgb  # pytest 夹具经 import 注册


class _Obj:
    def __init__(self, v):
        self.v = v


class _PYU:
    def __init__(self, party):
        self.party = party

    def __call__(self, fn):
        return lambda *a, **k: _Obj(fn(*[x.v if isinstance(x, _Obj) else x for x in a], **k))


class _Fed:
    def __init__(self, partitions, partition_way=None):
        self.partitions = partitions


class _ModelBuilder:
    def __init__(self, label_holder):
        self.label_holder = label_holder

    def init_pred(self, sample_num, checkpoint_model=None, x=None):
        return self.label_holder(lambda: np.zeros((sample_num, 1)))()


class _Model:
    def __init__(self, w, holder):
        self.w, self.holder = w, holder

    def predict_with_trees(self, fed):
        xs = [p.v for p in fed.partitions.values()]
        return _Obj(np.hstack(xs) @ self.w)


class _Sgb:
    calls: list = []

    def __init__(self, heu=None):
        pass

    def train(self, params, dtrain, label):
        holder = list(label.partitions)[0]
        X = np.hstack([p.v for p in dtrain.partitions.values()])
        y = list(label.partitions.values())[0].v.reshape(-1)
        init = _ModelBuilder(holder).init_pred(len(y)).v.reshape(-1)
        _Sgb.calls.append({"params": params, "init": init})
        w = (X - X.mean(0)).T @ (y - y.mean()) / len(y) * params["learning_rate"]
        return _Model(w, holder)


@pytest.fixture()
def fake_sf(monkeypatch):
    mods = {}
    sf = types.ModuleType("secretflow")
    sf.init = lambda **k: None
    sf.shutdown = lambda: None
    sf.PYU = _PYU
    sf.HEU = lambda cfg, field: None
    sf.utils = types.SimpleNamespace(testing=types.SimpleNamespace(
        heu_config=lambda a, b: {"he_parameters": {"schema": "paillier"}}))
    mods["secretflow"] = sf
    nd = types.ModuleType("secretflow.data.ndarray")
    nd.FedNdarray, nd.PartitionWay = _Fed, types.SimpleNamespace(VERTICAL="v")
    mods["secretflow.data.ndarray"] = nd
    dev = types.ModuleType("secretflow.device")
    dev.reveal = lambda o: o.v
    mods["secretflow.device"] = dev
    sgbm = types.ModuleType("secretflow.ml.boost.sgb_v")
    sgbm.Sgb = _Sgb
    mods["secretflow.ml.boost.sgb_v"] = sgbm
    mb = types.ModuleType("secretflow.ml.boost.sgb_v.factory.components.model_builder.model_builder")
    mb.ModelBuilder = _ModelBuilder
    mods["secretflow.ml.boost.sgb_v.factory.components.model_builder.model_builder"] = mb
    mods["secretflow.ml.boost.sgb_v.factory.components.model_builder"] = types.ModuleType("mbpkg")
    mods["secretflow.ml.boost.sgb_v.factory.components.model_builder"].model_builder = mb
    spu = types.ModuleType("spu")
    spu.FieldType = types.SimpleNamespace(FM64="FM64")
    mods["spu"] = spu
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    _Sgb.calls = []
    return mods


def test_参数映射与初始分数():
    P = MS.Params(CFG)
    cfg = {"rounds": 5, "depth": 3, "lr": 0.1, "lam": 1.0, "col": 0.7, "sub": 0.8}
    p = SG.sgb_params(P, cfg, P.bins_fine, -2.0, 11)
    assert p["sketch_eps"] == SG.SKETCH_EPS_FINE and p["rowsample_by_tree"] == 0.8 and p["base_score"] == -2.0
    assert SG.sketch_eps(P, P.bins_default) == SG.SKETCH_EPS_DEFAULT
    y = np.r_[np.ones(10), np.zeros(90)]
    b_sgb, b_proxy = SG.base_of(P, y, "prior")
    assert abs(b_proxy - 0.1) < 1e-12 and abs(MS.sigmoid(b_sgb) - 0.1) < 1e-9
    assert SG.base_of(P, y, "half") == (0.0, 0.5) and SG.base_of(P, y, None) == (0.0, None)


def test_逐样本初始预测注入后恢复(fake_sf):
    P = MS.Params(CFG)
    rng = np.random.default_rng(0)
    xg, xh, eg, eh = rng.normal(size=(30, 2)), rng.normal(size=(30, 2)), rng.normal(size=(8, 2)), rng.normal(size=(8, 2))
    y = (rng.random(30) < 0.3).astype(float)
    cfg = {"rounds": 3, "depth": 2, "lr": 0.3, "lam": 1.0, "col": 1.0}
    mf, me = np.arange(30, dtype=float), np.full(8, 2.0)
    orig = _ModelBuilder.init_pred
    p, t = SG.run_sgb(P, xg, xh, y, eg, eh, cfg, P.bins_fine, 0.0, mf, me, seed=7)
    assert np.allclose(_Sgb.calls[-1]["init"], mf) and _ModelBuilder.init_pred is orig
    assert _Sgb.calls[-1]["params"]["seed"] == 7 and p.shape == (8,) and t >= 0
    p0, _ = SG.run_sgb(P, xg, xh, y, eg, eh, cfg, P.bins_fine, 0.0)
    assert np.allclose(_Sgb.calls[-1]["init"], 0) and np.all(p > p0)        # 推理时加回了本地 logit


def test_复核与评估编排(tmp_path, fake_xgb, fake_sf):
    write_synthetic(tmp_path, "syn", True)
    cfg = small_cfg()
    res = MS.Screen("syn", cfg, tmp_path, tmp_path / "o").run()
    P = MS.Params(cfg)
    sc = MS.Screen("syn", cfg, tmp_path, tmp_path / "o2")
    out = tmp_path / "sgb"
    out.mkdir()
    rows = {m: SG.confirm(P, sc, res, m, out) for m in ("F_def", "F_opt_p", "F_M1x", "F_M6")}
    assert all(0 <= r["sgb_auroc"] <= 1 for r in rows.values()) and rows["F_M6"]["trees"] == res["configs"]["central"]["rounds"]
    n_calls = len(_Sgb.calls)
    again = SG.confirm(P, sc, res, "F_M6", out)                                # 成员已存档，不再训练
    assert len(_Sgb.calls) == n_calls and again["sgb_auroc"] == rows["F_M6"]["sgb_auroc"]
    assert len(json.loads((out / "syn__F_M6_m0.json").read_text())) == 2
    sgb = {m: np.load(out / ("syn__%s.npy" % m)).astype(float) for m in rows}
    proxy = {m: v + 0.0 for m, v in sgb.items()}
    ev = SG.evaluate(P, sc, res, sgb, proxy, [("F_opt_p", "F_def"), ("F_M6", "F_opt_p"), ("F_M1", "F_opt_p")])
    assert set(ev["contrasts"]) == {"F_opt_p − F_def", "F_M6 − F_opt_p"}
    assert all(v["sgb_minus_proxy"]["diff"] == 0 for v in ev["methods"].values())
    assert ev["contrasts"]["F_opt_p − F_def"]["proxy_6seed"]["label"] == "I2 + I3 合计"
