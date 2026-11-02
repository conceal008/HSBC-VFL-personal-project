"""M5 · S5.M1 方法筛选组件的单元测试。只用程序生成的合成数据；CI 没有 xgboost，用替身检验编排与统计。"""
from __future__ import annotations

import copy
import gzip
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m5_modeling.components import method_screen as MS  # noqa: E402
from modules.m5_modeling.components import method_screen_data as MD  # noqa: E402

CFG = yaml.safe_load((ROOT / "modules/m5_modeling/configs/s5_m1_method_screen.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- xgboost 替身：线性打分器
class _DM:
    def __init__(self, X, label=None, max_bin=None, ref=None, base_margin=None, weight=None, nthread=None):
        self.X = np.nan_to_num(np.asarray(X, float))
        self.y = None if label is None else np.asarray(label, float)
        self.margin = None if base_margin is None else np.asarray(base_margin, float)
        self.w = None if weight is None else np.asarray(weight, float)


class _Booster:
    def __init__(self, coef, base):
        self.coef, self.base = coef, base

    def predict(self, d, output_margin=False):
        z = d.X @ self.coef + (d.margin if d.margin is not None else self.base)
        return z if output_margin else 1 / (1 + np.exp(-z))


def _train(params, dtrain, num_boost_round, evals=(), early_stopping_rounds=None, verbose_eval=None):
    X, y = dtrain.X, dtrain.y
    w = np.ones(len(y)) if dtrain.w is None else dtrain.w
    sd = X.std(axis=0) + 1e-9
    coef = ((X - X.mean(axis=0)) / sd).T @ ((y - np.average(y, weights=w)) * w) / len(y) / sd
    coef = coef * params["eta"] * min(num_boost_round, 10)
    if params["colsample_bytree"] < 1 or params["subsample"] < 1:
        coef = coef * (1 + 0.01 * np.random.default_rng(params["seed"]).standard_normal(len(coef)))
    b = MS.logit(np.array([params.get("base_score", 0.5)]))[0]
    bst = _Booster(coef, b)
    if evals:
        d = evals[0][0]
        bst.best_iteration = min(num_boost_round, params["max_depth"] + 2) - 1
        bst.best_score = MS.auc(d.y, bst.predict(d))
    return bst


@pytest.fixture()
def fake_xgb(monkeypatch):
    mod = types.ModuleType("xgboost")
    mod.QuantileDMatrix = _DM
    mod.train = _train
    monkeypatch.setitem(sys.modules, "xgboost", mod)
    return mod


def small_cfg():
    cfg = copy.deepcopy(CFG)
    s = cfg["screen"]
    s["grid"] = {"lr": [0.3, 0.1], "depth": [3], "lam": [1.0], "col": [1.0, 0.7]}
    s.update({"m4_top": 2, "folds": 3, "bag": 2, "n_boot": 30, "dp_eps": [2.0], "mismatch": [0.1]})
    return cfg


def synth(n, rng, d_g=4, d_h=3):
    xg, xh = rng.normal(size=(n, d_g)), rng.normal(size=(n, d_h))
    z = xg[:, 0] + 1.5 * xh[:, 0] - 2.5
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return xg, xh, y


def write_synthetic(priv: Path, name: str, extra: bool):
    rng = np.random.default_rng(5)
    arr = {}
    for view, (nf, ne) in {"tune": (300, 200), "final": (400, 250)}.items():
        xg, xh, y = synth(nf, rng)
        eg, eh, ey = synth(ne, rng)
        arr.update({view + "__fit_xg": xg, view + "__fit_xh": xh, view + "__fit_y": y,
                    view + "__ev_xg": eg, view + "__ev_xh": eh, view + "__ev_y": ey,
                    view + "__ev_cl": np.arange(ne) // 2})
        if extra:
            x2, _, y2 = synth(150, rng)
            arr.update({view + "__ex_xg": x2, view + "__ex_y": y2})
    np.savez(priv / (name + ".npz"), **arr)
    (priv / (name + ".json")).write_text(json.dumps({"name": name}), encoding="utf-8")


# ---------------------------------------------------------------- 统计与选择原语
def test_参数展开与网格():
    P = MS.Params(CFG)
    assert len(P.grid) == 48 and P.seeds[0] == P.tune_seed and P.bins_fine > P.bins_default
    assert MS.deterministic({"col": 1.0}) and not MS.deterministic({"col": 0.7})


def test_AUROC含并列与sklearn一致():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.random(200), 1)
    assert abs(MS.auc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_选择规则取成本最小():
    rows = [{"val_auc": 0.80, "rounds": 100, "depth": 6}, {"val_auc": 0.7995, "rounds": 50, "depth": 3, "sub": 0.5},
            {"val_auc": 0.79, "rounds": 10, "depth": 2}]
    assert MS.pick(rows, 0.001)["rounds"] == 50
    assert MS.cost({"rounds": 10, "depth": 3, "sub": 0.5}, bags=2) == 30


def test_后验软标签与Holm():
    q = np.array([0.2, 0.2, 0.6, 0.6])
    yn = np.array([1, 0, 1, 0])
    post = MS.label_posterior(q, yn, 0.1, 1e-4)
    assert post[0] > post[1] and post[2] > post[3] and np.all((post > 0) & (post < 1))
    assert MS.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_两层bootstrap相同分数差为零():
    P = MS.Params(small_cfg())
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 120)
    s = rng.random(120)
    A = MS.two_level_bootstrap(P, y, None, {"a": [s] * 3, "b": [s, s + 0.0, s], "c": [rng.random(120)] * 3})
    assert np.allclose(A["a"], A["b"]) and A["a"].shape == (P.n_boot,)
    assert len(MS.contrast_pairs(P)) == len(MS.CONTRAST_PAIRS) + 3 + 3


# ---------------------------------------------------------------- 端到端编排（替身 xgboost）
@pytest.mark.parametrize("extra", [False, True])
def test_端到端筛选(tmp_path, fake_xgb, extra):
    write_synthetic(tmp_path, "syn", extra)
    out = tmp_path / "out"
    res = MS.Screen("syn", small_cfg(), tmp_path, out).run()
    saved = json.loads((out / "syn.json").read_text(encoding="utf-8"))
    assert saved["val"]["combo_choice"] == res["val"]["combo_choice"]
    keys = set(res["test"])
    assert {"G_t", "F_def", "F_opt_p", "F_M1", "F_M6", "COMBO", "M2_fuse", "M2_student", "DP_soft_e2",
            "M5_drop_r10"} <= keys
    assert ("F_M1x" in keys) == extra and ("M2_student_x" in keys) == extra
    labels = {c["label"] for c in res["contrasts"]}
    assert "I3 先验初始分数" in labels and all(0 <= c["p_holm"] <= 1 for c in res["contrasts"])
    assert len(res["test"]["F_def"]["per_seed"]) == len(CFG["seeds"]) and res["test"]["F_def"]["seed_sd"] == 0.0


def test_组合为bagging时单独评估(tmp_path, fake_xgb):
    write_synthetic(tmp_path, "syn", True)
    sc = MS.Screen("syn", small_cfg(), tmp_path, tmp_path / "o")
    sc.tune_all()
    sc.select()
    sc.combo = "M1x+M6"
    sc.test_all()
    assert sc.res["test"]["COMBO"]["choice"] == "M1x+M6" and "same_as" not in sc.res["test"]["COMBO"]
    assert sc.costs["COMBO"] == MS.cost({**sc.cm1x, "sub": sc.P.bag_sub}, sc.P.bag)


# ---------------------------------------------------------------- 数据准备
def prep_cfg():
    c = copy.deepcopy(CFG["prep"])
    c.update({"homecredit_rows": 60, "santander_tx_frac": 1.0})
    return c


def test_通用编码与分层抽样():
    df = pd.DataFrame({"num": np.r_[[np.nan] * 5, np.arange(35, dtype=float)],
                       "cat": ["a", "b", None, "c"] * 10, "small": [1, 2] * 20})
    fit = np.arange(40) < 30
    enc = MD.generic_encode(df, fit)
    assert {"num", "num__isna", "cat", "small"} == set(enc.columns)
    assert enc["cat"].max() == 1.0 and abs(enc.loc[fit, "num"].mean()) < 1e-9
    y = np.r_[np.ones(20, int), np.zeros(80, int)]
    m = MD.strat(y, 0.2, 1)
    assert m.sum() == 20 and y[m].sum() == 4
    te, va = MD.simple_split(y, prep_cfg())
    assert not (te & va).any() and te.sum() == 20


def test_目标编码折外():
    x = pd.Series(["a"] * 50 + ["b"] * 50)
    y = np.r_[np.ones(50), np.zeros(50)]
    oof, (app,) = MD.target_encode(x, y, [pd.Series(["a", "b", "z"])], 0.5)
    assert app[0] > app[1] and app[2] == 0.5 and oof[:50].mean() > oof[50:].mean()


def _mock_frames(n_cust, months, rng, with_tiprel):
    rows = []
    for c in range(n_cust):
        for m in months:
            r = {"ncodpers": 1000 + c, "fecha_dato": m, "sexo": rng.choice(["H", "V"]), "age": float(rng.integers(20, 80)),
                 "renta": rng.normal(1e5, 2e4) if rng.random() > 0.3 else np.nan, "segmento": rng.choice(["01", "02"]),
                 "label_new_any": int(rng.random() < 0.2), "label_n_new": 0}
            if with_tiprel:
                r["tiprel_1mes"] = rng.choice(["A", "I"])
            rows.append(r)
    g = pd.DataFrame(rows)
    h = pd.DataFrame({"ncodpers": g["ncodpers"], "ind_cco": rng.integers(0, 2, len(g)), "ind_ctma": rng.integers(0, 2, len(g))})
    return g, h


def test_mock数据含非共同客户(tmp_path):
    rng = np.random.default_rng(3)
    src = tmp_path / "raw" / "数据集" / "HSBC_mock数据"
    src.mkdir(parents=True)
    months = ["2015-10-28", "2015-11-28", "2015-12-28", "2016-01-28", "2016-02-28", "2016-03-28"]
    bg, _ = _mock_frames(12, months, rng, False)
    bg.to_csv(src / "balanced_guest.csv", index=False)
    g, h = _mock_frames(4, months, rng, True)
    g.to_csv(src / "overlap20_guest.csv", index=False)
    h.to_csv(src / "overlap20_host.csv", index=False)
    info = MD.build("mock_overlap20", tmp_path / "raw", tmp_path / "priv", prep_cfg())
    d = np.load(tmp_path / "priv" / "mock_overlap20.npz")
    assert info["extra_rows"] == len(d["final__ex_y"]) > len(d["tune__ex_y"]) > 0
    assert info["final_eval_rows"] == 4 * 2 and d["final__ev_cl"].shape == (8,)


def test_各公开数据集加载(tmp_path):
    rng = np.random.default_rng(4)
    raw, priv, cfg = tmp_path / "raw", tmp_path / "priv", prep_cfg()
    n = 60
    # UCI
    d = raw / "数据集" / "UCI_BankMarketing" / "解压"
    d.mkdir(parents=True)
    uci = pd.DataFrame({c: rng.normal(size=n) for c in MD.UCI_GUEST + MD.UCI_HOST})
    uci["job"] = rng.choice(["x", "y"], n)
    uci["y"] = np.where(rng.random(n) < 0.3, "yes", "no")
    with zipfile.ZipFile(d / "bank-additional.zip", "w") as z:
        z.writestr("bank-additional/bank-additional-full.csv", uci.to_csv(sep=";", index=False))
    assert MD.build("uci_bank", raw, priv, cfg)["n_host"] == len(MD.UCI_HOST)
    # COIL2000
    d = raw / "数据集" / "COIL2000_保险营销" / "解压"
    d.mkdir(parents=True)
    tr = rng.integers(0, 5, (n, MD.COIL_FEATURES + 1))
    tr[:, -1] = rng.random(n) < 0.3
    np.savetxt(d / "ticdata2000.txt", tr, fmt="%d")
    np.savetxt(d / "ticeval2000.txt", rng.integers(0, 5, (40, MD.COIL_FEATURES)), fmt="%d")
    np.savetxt(d / "tictgts2000.txt", (rng.random(40) < 0.3).astype(int), fmt="%d")
    assert MD.build("coil2000", raw, priv, cfg)["final_eval_rows"] == 40
    # Santander 交易
    d = raw / "数据集" / "Santander_CustomerTransaction" / "解压"
    d.mkdir(parents=True)
    st = pd.DataFrame(rng.normal(size=(n, MD.SANTANDER_TX_VARS)), columns=["var_%d" % i for i in range(MD.SANTANDER_TX_VARS)])
    st.insert(0, "target", (rng.random(n) < 0.3).astype(int))
    st.to_csv(d / "train.csv", index=False)
    assert MD.build("santander_tx", raw, priv, cfg)["n_guest"] == MD.SANTANDER_TX_VARS // 2
    # HomeCredit
    d = raw / "数据集" / "HomeCredit_DefaultRisk" / "解压"
    d.mkdir(parents=True)
    hc = pd.DataFrame({c: rng.normal(size=n) for c in MD.HC_HOST + ["AMT_CREDIT", "AMT_INCOME"]})
    hc["NAME_TYPE"] = rng.choice(["a", "b"], n)
    hc["TARGET"], hc["SK_ID_CURR"] = (rng.random(n) < 0.3).astype(int), np.arange(n)
    hc.to_csv(d / "application_train.csv", index=False)
    assert MD.build("homecredit", raw, priv, cfg)["n_host"] == len(MD.HC_HOST)


def test_Orange_IEEE_FedAds_G3加载(tmp_path):
    rng = np.random.default_rng(6)
    raw, priv, cfg = tmp_path / "raw", tmp_path / "priv", prep_cfg()
    n = 80
    d = raw / "数据集" / "Orange_KDDCup2009"
    d.mkdir(parents=True)
    org = pd.DataFrame({c: rng.normal(size=n) for c in MD.ORANGE_NUM})
    org.loc[rng.random(n) < 0.2, "Var1"] = np.nan
    for c in MD.ORANGE_CAT:
        org[c] = rng.choice(["p", "q", "r", None], n)
    with zipfile.ZipFile(d / "orange_small_train.data.zip", "w") as z:
        z.writestr("orange_small_train.data", org.to_csv(sep="\t", index=False))
    np.savetxt(d / "orange_small_train_churn.labels", np.where(rng.random(n) < 0.3, 1, -1), fmt="%d")
    info = MD.build("orange_churn", raw, priv, cfg)
    assert info["n_guest"] > 0 and info["n_host"] > 0
    # IEEE
    d = raw / "federated_learning_new_复现与优化" / "ieee_sample"
    d.mkdir(parents=True)
    tx = pd.DataFrame({"TransactionID": np.arange(n), "isFraud": (rng.random(n) < 0.3).astype(int),
                       "TransactionDT": rng.integers(0, 10 ** 6, n), "amt": rng.normal(size=n),
                       "card": rng.choice(["v", "m"], n)})
    idn = pd.DataFrame({"TransactionID": np.arange(n), "id_01": rng.normal(size=n), "Device": rng.choice(["a", "b"], n)})
    with gzip.open(d / "ieee_tx_part0.csv.gz", "wt") as f:
        tx.to_csv(f, index=False)
    with gzip.open(d / "ieee_identity.csv.gz", "wt") as f:
        idn.to_csv(f, index=False)
    assert MD.build("ieee_part0", raw, priv, cfg)["final_eval_rows"] == n - int(n * cfg["ieee_test_start"])
    # FedAds
    d = raw / "federated_learning_new_复现与优化" / "fedads_sample"
    d.mkdir(parents=True)

    def fed(rows, aligned):
        cols = MD.FEDADS_GUEST + (MD.FEDADS_HOST if aligned else [])
        if not aligned:
            cols = [c for c in cols if c != "l_c_fea"]
        df = pd.DataFrame({c: ['["%d"]' % v for v in rng.integers(0, 3, rows)] for c in cols})
        df.insert(0, "label", ['["%d"]' % v for v in (rng.random(rows) < 0.3).astype(int)])
        df.insert(0, "sample_id", np.arange(rows))
        return df
    fed(n, True).to_csv(d / "train_s300k.csv", index=False)
    fed(40, True).to_csv(d / "test_s200k.csv", index=False)
    fed(50, False).to_csv(d / cfg["fedads_extra_file"], index=False)
    info = MD.build("fedads", raw, priv, cfg)
    ex = np.load(priv / "fedads.npz")["final__ex_xg"]
    assert info["extra_rows"] == 50 and np.isnan(ex).any()
    # Santander G3 私有数组
    arr = {}
    for split, m in (("train", 30), ("inner_val", 20), ("dev_eval", 25)):
        arr.update({split + "__xg": rng.normal(size=(m, 3)), split + "__xh": rng.normal(size=(m, 2)),
                    split + "__y": rng.integers(0, 2, m), split + "__cust": np.arange(m)})
    np.savez(tmp_path / "g3.npz", **arr)
    info = MD.build("santander_g3", raw, priv, cfg, g3_arrays=tmp_path / "g3.npz")
    assert info["final_fit_rows"] == 50 and info["final_eval_rows"] == 25
