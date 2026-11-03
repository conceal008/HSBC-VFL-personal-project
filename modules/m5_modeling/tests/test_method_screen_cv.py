"""M5 · S5.M3 交叉验证选组合组件的单元测试。只用程序生成的合成数据；CI 没有 xgboost，用 S5.M1 测试中的替身。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m5_modeling.components import method_screen as MS  # noqa: E402
from modules.m5_modeling.components import method_screen_cv as CV  # noqa: E402
from modules.m5_modeling.tests.test_method_screen import (  # noqa: E402
    fake_xgb, small_cfg, write_synthetic)

assert fake_xgb  # pytest 夹具经 import 注册
M3 = yaml.safe_load((ROOT / "modules/m5_modeling/configs/s5_m3_cv_combo.yaml").read_text(encoding="utf-8"))


def test_候选清单与计算量():
    P = MS.Params(small_cfg())
    res = {"configs": {"central": {"rounds": 10, "depth": 3, "col": 1.0}, "m4": {"rounds": 8, "depth": 3, "col": 1.0, "sub": 0.5},
                       "m1": {"rounds": 2, "depth": 3, "col": 0.7}, "m1x": {"rounds": 3, "depth": 3, "col": 1.0}}}
    assert CV.candidates(False) == list(CV.CANDS_BASE) and CV.candidates(True)[-2:] == list(CV.CANDS_EX)
    assert CV.spec(res, "M1x+M6")[1:] == ("guest_extra", True) and CV.spec(res, "P_M4")[1:] == (None, False)
    assert CV.cost_of(P, res, "P+M6") == MS.cost({"rounds": 10, "depth": 3, "sub": P.bag_sub}, P.bag)
    assert CV.deterministic(res, "P") and not CV.deterministic(res, "M1") and not CV.deterministic(res, "P+M6")


def test_三种折方案():
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.2).astype(int)
    k = M3["cv"]["k_row"]
    fs = CV.folds_stratified(y, k, M3["cv"]["fold_seed"])
    assert len(fs) == k and sorted(np.concatenate([va for _, va, _ in fs]).tolist()) == list(range(len(y)))
    assert max(abs(y[va].sum() - y.sum() / k) for _, va, _ in fs) <= 1
    g = rng.integers(0, 40, 200)
    for tr, va, _ in CV.folds_grouped(g, M3["cv"]["k_group"], M3["cv"]["fold_seed"]):
        assert not set(g[tr]) & set(g[va])
    months = np.array(["2015-%02d-28" % m for m in rng.integers(1, 13, 200)])
    exm = np.array(["2015-%02d-28" % m for m in rng.integers(1, 13, 50)])
    blocks = [["2015-07-28", "2015-08-28"], ["2015-11-28", "2015-12-28"]]
    for (tr, va, ex), b in zip(CV.folds_time(months, blocks, exm), blocks):
        assert (months[tr] < min(b)).all() and set(months[va]) <= set(b) and (exm[ex] < min(b)).all()


def test_选择规则():
    cv = {"P": [0.80, 0.81, 0.82], "M1": [0.801, 0.809, 0.821], "P+M6": [0.82, 0.83, 0.84]}
    costs = {"P": 30.0, "M1": 5.0, "P+M6": 150.0}
    ch = CV.choose(cv, costs, M3["cv"]["se_mult"])
    assert ch["r1"] == "P+M6" and ch["r2"] == "P+M6"                       # 差距远超标准误：R2 不退到便宜候选
    cv2 = {"P": [0.80, 0.82, 0.81], "M1": [0.805, 0.815, 0.812], "P+M6": [0.806, 0.816, 0.811]}
    ch2 = CV.choose(cv2, costs, M3["cv"]["se_mult"])
    assert ch2["r1"] == "P+M6" and ch2["r2"] == "M1" and ch2["paired_vs_r1"]["P+M6"]["diff"] == 0


def test_交叉验证与测试编排(tmp_path, fake_xgb):
    write_synthetic(tmp_path, "syn", True)
    cfg = small_cfg()
    res = MS.Screen("syn", cfg, tmp_path, tmp_path / "o").run()
    (tmp_path / "syn.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    out = CV.run("syn", cfg, M3, tmp_path, tmp_path, {})
    assert out["scheme"] == "row" and set(out["cv"]) == set(CV.candidates(True))
    assert all(len(v) == M3["cv"]["k_row"] for v in out["cv"].values())
    assert {"CV_R1", "CV_R2", "P", "G_t"} <= set(out["test"]) and len(out["contrasts"]) == 4
    assert all(c["ci_low"] <= c["ci_high"] and 0 <= c["p_holm"] <= 1 for c in out["contrasts"])
    assert out["s5m1"]["choice"] == res["val"]["combo_choice"]
    P = MS.Params(cfg)
    sc = MS.Screen("syn", cfg, tmp_path, tmp_path / "o2")
    fv = sc.v["final"]
    cache: dict = {}
    a = CV.predict(P, res, "M1+M6", fv["fit_xc"], fv["fit_xg"], fv["fit_y"], fv["ev_xc"], fv["ev_xg"], P.tune_seed, cache=cache)
    assert len(cache) == 1 and a.shape == fv["ev_y"].shape


def test_mock与G3分组重建(tmp_path):
    src = tmp_path / "raw" / "数据集" / "HSBC_mock数据"
    src.mkdir(parents=True)
    months = ["2015-12-28", "2016-01-28", "2016-02-28"]
    g = pd.DataFrame({"ncodpers": [1, 2, 1, 2, 1, 2], "fecha_dato": [m for m in months for _ in range(2)],
                      "label_new_any": [0, 1, 1, 0, 0, 1]})
    g.to_csv(src / "overlap80_guest.csv", index=False)
    b = pd.DataFrame({"ncodpers": [1, 7, 7, 8], "fecha_dato": ["2015-12-28", "2015-12-28", "2016-02-28", "2016-01-28"],
                      "label_new_any": [1, 0, 1, 1]})
    b.to_csv(src / "balanced_guest.csv", index=False)
    prep = {"mock_test_start": "2016-02-01"}
    out = CV.build_groups("mock_overlap80", tmp_path / "raw", prep, None, np.array([0, 1, 1, 0]), np.array([0, 1]))
    assert out["fit_month"].tolist() == months[:1] * 2 + months[1:2] * 2 and out["ex_month"].tolist() == ["2015-12-28", "2016-01-28"]
    np.savez(tmp_path / "groups.npz", **out)
    assert dict(np.load(tmp_path / "groups.npz"))["fit_month"].tolist() == out["fit_month"].tolist()   # 无需 pickle
    np.savez(tmp_path / "g3.npz", train__y=np.array([0, 1]), inner_val__y=np.array([1]),
             train__cust=np.array([5, 6]), inner_val__cust=np.array([9]))
    assert CV.build_groups("santander_g3", tmp_path, prep, tmp_path / "g3.npz", np.array([0, 1, 1]))["fit_cust"].tolist() == [5, 6, 9]
    assert CV.build_groups("uci_bank", tmp_path, prep, None, np.array([0])) == {}
