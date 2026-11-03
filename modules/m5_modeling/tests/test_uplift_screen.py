"""M5 · S5.M6 增量学习器组件的单元测试。只用程序生成的合成数据；CI 没有 xgboost，用线性最小二乘替身。"""
from __future__ import annotations

import gzip
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m5_modeling.components import uplift_screen as U  # noqa: E402

M6 = yaml.safe_load((ROOT / "modules/m5_modeling/configs/s5_m6_uplift.yaml").read_text(encoding="utf-8"))
RIDGE = 1e-3
P_CLIP = 1e-3


class _DM:
    def __init__(self, X, label=None, weight=None, max_bin=None, ref=None, nthread=None):
        self.X = np.asarray(X, float)
        self.y = None if label is None else np.asarray(label, float)
        self.w = None if weight is None else np.asarray(weight, float)


class _Booster:
    def __init__(self, coef, clip):
        self.coef, self.clip = coef, clip

    def predict(self, d):
        z = np.c_[np.ones(len(d.X)), d.X] @ self.coef
        return np.clip(z, P_CLIP, 1 - P_CLIP) if self.clip else z


def _train(params, d, num_boost_round):
    A = np.c_[np.ones(len(d.X)), d.X]
    w = np.ones(len(d.y)) if d.w is None else d.w
    coef = np.linalg.solve(A.T @ (A * w[:, None]) + RIDGE * np.eye(A.shape[1]), A.T @ (w * d.y))
    return _Booster(coef, params["objective"] == "binary:logistic")


@pytest.fixture()
def fake_xgb(monkeypatch):
    mod = types.ModuleType("xgboost")
    mod.QuantileDMatrix = _DM
    mod.train = _train
    monkeypatch.setitem(sys.modules, "xgboost", mod)
    return mod


def synth(n=3000, stream=0):
    rng = np.random.default_rng(stream)
    xg, xh = rng.normal(size=(n, 2)), rng.normal(size=(n, 2))
    t = (rng.random(n) < 0.6).astype(int)
    base = 0.2 + 0.1 * (xg[:, 1] > 0)
    lift = 0.25 * (xh[:, 0] > 0)                       # 增量只由香港特征决定
    y = (rng.random(n) < base + t * lift).astype(int)
    strata = np.char.add(t.astype(str), y.astype(str))
    return {"xg": xg, "xh": xh, "g_cols": ["g0", "g1"], "h_cols": ["h0", "h1"], "t": t,
            "y": {"visit": y, "conversion": y}, "split": U.strat_split(strata, [0.6, 0.2, 0.2], 1)}


def test_分层切分():
    s = np.array(["a"] * 100 + ["b"] * 50)
    sp = U.strat_split(s, [0.6, 0.2, 0.2], 3)
    assert np.array_equal(sp, U.strat_split(s, [0.6, 0.2, 0.2], 3))
    for lab, n in (("a", 100), ("b", 50)):
        m = s == lab
        assert np.sum(sp[m] == U.SPLIT_TRAIN) == round(0.6 * n) and np.sum(sp[m] == U.SPLIT_VAL) == round(0.2 * n)


def test_Qini与前q增量():
    rng = np.random.default_rng(2)
    n = 4000
    t = (rng.random(n) < 0.5).astype(int)
    good = rng.random(n) < 0.5
    y = (rng.random(n) < 0.1 + 0.3 * t * good).astype(int)
    perfect, noise, const = U.Ranked(good.astype(float), t, y), U.Ranked(rng.random(n), t, y), U.Ranked(np.zeros(n), t, y)
    assert perfect.qini() > 0.012 and abs(noise.qini()) < 0.006          # 理论值：0.15 × 0.5² / 2 = 0.01875
    assert abs(const.qini()) < 1e-12                          # 全部并列 = 随机线
    assert perfect.qini(np.ones(n)) == pytest.approx(perfect.qini())
    assert perfect.uplift_at(0.3) == pytest.approx(0.3, abs=0.06) and abs(perfect.uplift_at(1.0) - 0.15) < 0.05
    B = U.bootstrap({"p": perfect, "c": const}, n, 50, 7, 0.3)
    c = U.paired(B["p"]["qini"], B["c"]["qini"], perfect.qini())
    assert c["ci_low"] > 0 and c["p"] < 0.05
    iv = U.interval(B["p"]["uplift"], perfect.uplift_at(0.3))
    assert iv["ci_low"] <= iv["est"] <= iv["ci_high"]


def test_学习器找出增量(fake_xgb):
    d = synth()
    tr = d["split"] != U.SPLIT_TEST
    te = ~tr
    X = U.features(d, "fed")
    b = U.Base({"lr": 0.05, "depth": 3, "lam": 1.0, "rounds": 10}, 33, 1, 11)
    sc = U.scores_for(b, X[tr], d["t"][tr], d["y"]["visit"][tr], X[te], 3, 11)
    assert set(sc) == set(U.LEARNERS) | {U.RESPONSE}
    q = {k: U.Ranked(s, d["t"][te], d["y"]["visit"][te]).qini() for k, s in sc.items()}
    assert q["T"] > q[U.RESPONSE] and q["X"] > q[U.RESPONSE] and q["R"] > 0
    assert U.features(d, "main").shape[1] == 2 and X.shape[1] == 4


def test_单个结果编排(fake_xgb):
    d = {**synth(), "name": "syn"}
    m6 = {**M6, "n_boot": 40, "grid": {"lr": 0.05, "lam": 1.0, "depth": [3], "rounds": [10, 20]}}
    r = U.run_outcome(d, "visit", m6)
    assert set(r["choice"].values()) <= set(U.LEARNERS) and len(r["contrasts"]) == 3
    assert all(0 <= c["p_holm"] <= 1 and c["ci_low"] <= c["ci_high"] for c in r["contrasts"])
    assert len(r["configs"]["fed"]["grid"]) == 2 and r["counts"]["test"]["rows"] == int(np.sum(d["split"] == U.SPLIT_TEST))
    assert {k.split(":")[0] for k in r["test"]} == set(U.FEATURE_SETS)
    assert r["contrasts"][0]["a"].startswith("fed:") and r["contrasts"][0]["b"].startswith("main:")


def test_读取与命令行(tmp_path, fake_xgb, monkeypatch):
    rng = np.random.default_rng(4)
    n = 600
    seg = rng.choice(["No E-Mail", "Mens E-Mail", "Womens E-Mail"], n)
    hs = pd.DataFrame({"recency": rng.integers(1, 12, n), "history": rng.random(n) * 500, "mens": rng.integers(0, 2, n),
                       "womens": rng.integers(0, 2, n), "zip_code": rng.choice(["Urban", "Rural"], n),
                       "newbie": rng.integers(0, 2, n), "channel": rng.choice(["Web", "Phone"], n), "segment": seg,
                       "visit": rng.integers(0, 2, n), "conversion": (rng.random(n) < 0.1).astype(int)})
    hp = tmp_path / "raw" / M6["datasets"]["hillstrom"]["path"]
    hp.parent.mkdir(parents=True)
    hs.to_csv(hp, index=False)
    h = U.load_hillstrom(hp, M6["datasets"]["hillstrom"])
    assert h["t"].sum() == int(np.sum(seg != "No E-Mail")) and h["xg"].shape == (n, 4)
    assert h["h_cols"][:1] == ["newbie"] and h["xh"].shape[1] == 1 + 2 + 2
    cr = pd.DataFrame({**{"f%d" % i: rng.normal(size=5000) for i in range(12)}, "treatment": rng.integers(0, 2, 5000),
                       "conversion": rng.integers(0, 2, 5000), "visit": rng.integers(0, 2, 5000),
                       "exposure": rng.integers(0, 2, 5000)})
    cp = tmp_path / "raw" / M6["datasets"]["criteo"]["path"]
    cp.parent.mkdir(parents=True)
    with gzip.open(cp, "wt") as fh:
        cr.to_csv(fh, index=False)
    spec = {**M6["datasets"]["criteo"], "read_chunk": 1000, "sample_frac": 0.5}
    c = U.load_criteo(cp, spec)
    assert 2000 < len(c["t"]) < 3000 and c["g_cols"][1] == "f2" and c["xh"].shape[1] == 6
    assert np.array_equal(c["t"], U.load_criteo(cp, spec)["t"])     # 抽样确定
    cfg = {**M6, "n_boot": 20, "grid": {"lr": 0.05, "lam": 1.0, "depth": [3], "rounds": [10]},
           "datasets": {"hillstrom": {**M6["datasets"]["hillstrom"], "outcomes": ["visit"]}}}
    cpath = tmp_path / "m6.yaml"
    cpath.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv(M6["raw_root_env"], str(tmp_path / "raw"))
    monkeypatch.setenv(M6["run_dir_env"], str(tmp_path / "out"))
    assert U.main(["--config", str(cpath)]) == 0
    out = json.loads((tmp_path / "out" / "hillstrom.json").read_text(encoding="utf-8"))
    assert out["dataset"] == "hillstrom" and set(out["outcomes"]) == {"visit"}
    assert U.main(["--config", str(cpath)]) == 0                    # 已完成的跳过
