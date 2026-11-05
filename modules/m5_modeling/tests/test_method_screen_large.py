"""M5 · S5.M5 Orange 大版组件的单元测试。只用程序生成的合成数据；CI 没有 xgboost，用 S5.M1 测试中的替身（加重要性接口）。"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m5_modeling.components import method_screen as MS  # noqa: E402
from modules.m5_modeling.components import method_screen_large as ML  # noqa: E402
from modules.m5_modeling.tests import test_method_screen as T  # noqa: E402
from modules.m5_modeling.tests.test_method_screen import small_cfg  # noqa: E402

NUM = ["Var%d" % i for i in range(1, 9)]
CAT = ["Var%d" % i for i in range(9, 13)]
N = 400


def _train_with_score(params, dtrain, num_boost_round, evals=(), early_stopping_rounds=None, verbose_eval=None):
    bst = T._train(params, dtrain, num_boost_round, evals, early_stopping_rounds, verbose_eval)
    bst.get_score = lambda importance_type="weight": {"f%d" % j: abs(float(c)) for j, c in enumerate(bst.coef)
                                                      if abs(c) > 1e-6}
    return bst


@pytest.fixture()
def fake_xgb_imp(fake_xgb):
    fake_xgb.train = _train_with_score
    return fake_xgb


assert T.fake_xgb  # pytest 夹具经 import 注册
fake_xgb = T.fake_xgb


def synth_frame(seed: int = 3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=N) for c in NUM})
    df["Var2"] = np.nan                                   # 全缺失 → 删
    df["Var3"] = 7.0                                      # 常数 → 删
    df.loc[rng.random(N) < 0.2, "Var4"] = np.nan          # 缺失 20% → 加缺失指示
    for c in CAT:
        df[c] = rng.choice(["a", "b", "c"], size=N)
    df.loc[rng.random(N) < 0.1, "Var9"] = np.nan
    df["Var12"] = "only"                                  # 单一取值 → 删
    z = 1.5 * df["Var1"].fillna(0) + 1.5 * df["Var5"] - 1.0
    y = (rng.random(N) < 1 / (1 + np.exp(-z))).astype(int)
    return df, y


def write_chunks(src: Path, df: pd.DataFrame, chunks: int = 2, prefix: str = "orange_large_train.data"):
    src.mkdir(parents=True, exist_ok=True)
    for i, part in enumerate(np.array_split(np.arange(len(df)), chunks), start=1):
        buf = io.StringIO()
        df.iloc[part].to_csv(buf, sep="\t", index=False, header=(i == 1), na_rep="")
        stem = "%s.chunk%d" % (prefix, i)
        with zipfile.ZipFile(src / (stem + ".zip"), "w") as z:
            z.writestr(stem, buf.getvalue())


def test_读取分块与两方划分(tmp_path):
    df, _ = synth_frame()
    write_chunks(tmp_path, df)
    num, cat = ML.read_raw(tmp_path, chunks=2, num_vars=NUM, cat_vars=CAT, max_rows=N + 10)
    assert num.shape == (N, len(NUM)) and num.dtype == np.float32 and len(cat) == N
    assert np.isnan(num[:, 1]).all() and np.allclose(num[:, 0], df["Var1"].to_numpy(np.float32))
    assert cat["Var9"].isna().sum() == df["Var9"].isna().sum() and str(cat["Var10"].dtype) == "category"
    kept = ML.scan_numeric(tmp_path, chunks=2, num_vars=NUM, cat_vars=CAT)
    assert kept == [v for v in NUM if v not in ("Var2", "Var3")]           # 全缺失、常数被去掉
    sub, _ = ML.read_raw(tmp_path, chunks=2, num_vars=kept, cat_vars=CAT, max_rows=N, all_num=NUM)
    assert sub.shape == (N, len(kept)) and np.allclose(sub[:, kept.index("Var5")], df["Var5"].to_numpy(np.float32))
    mm, _ = ML.read_raw(tmp_path, chunks=2, num_vars=kept, cat_vars=CAT, max_rows=N, all_num=NUM,
                        num_path=tmp_path / "num.npy")
    assert isinstance(mm, np.memmap) and np.isfortran(mm) and np.array_equal(np.asarray(mm), sub, equal_nan=True)
    g = ML.party_vars(42, NUM, CAT)
    assert len(g) == (len(NUM) + len(CAT)) // 2 and g == ML.party_vars(42, NUM, CAT) and g != ML.party_vars(7, NUM, CAT)


def test_列清单只看fit行():
    df, _ = synth_frame()
    num, cat = df[NUM].to_numpy(np.float32), df[CAT]
    fit = np.arange(N) < 300
    guest = {"Var1", "Var4", "Var9", "Var10"}
    plan = ML.column_plan(num, cat, fit, guest, NUM)
    names = [p["var"] for p in plan]
    assert "Var2" not in names and "Var3" not in names and "Var12" not in names
    v4 = next(p for p in plan if p["var"] == "Var4")
    assert v4["flag"] and v4["side"] == "g"
    v11 = next(p for p in plan if p["var"] == "Var11")
    assert v11["side"] == "h" and v11["ord"]
    assert ML.encoded_names(plan, "g")[:3] == ["Var1", "Var4", "Var4__isna"]
    assert "te_Var9" in ML.encoded_names(plan, "g") and "ord_Var11" in ML.encoded_names(plan, "h")
    num2 = num.copy()
    num2[300:, 2] = np.arange(N - 300)                    # 只改非 fit 行 → 清单不变
    assert ML.column_plan(num2, cat, fit, guest, NUM) == plan


def test_编码与预筛并打包(tmp_path, fake_xgb_imp):
    df, y = synth_frame()
    num, cat = df[NUM].to_numpy(np.float32), df[CAT]
    cfg = small_cfg()
    P = MS.Params(cfg)
    info = ML.build("churn", num, cat, y, tmp_path, cfg["prep"], P, 2, NUM, CAT)
    kept = [v for v in NUM if v not in ("Var2", "Var3")]
    info_k = ML.build("churn", num[:, [NUM.index(v) for v in kept]], cat, y, tmp_path / "k", cfg["prep"], P, 2,
                      kept, CAT, all_num=NUM)
    assert info_k["guest_cols"] == info["guest_cols"] and info_k["host_cols"] == info["host_cols"]   # 预先去掉不改变结果
    name = ML.name_of("churn")
    pre = info["prescreen"]
    assert pre["k"] == 2 and len(info["guest_cols"]) <= 2 and len(info["host_cols"]) <= 2
    assert pre["clean_guest"] + pre["clean_host"] >= len(info["guest_cols"]) + len(info["host_cols"])
    sc = MS.Screen(name, cfg, tmp_path, tmp_path / "o")
    for v in ("tune", "final"):
        assert sc.v[v]["fit_xg"].shape[1] == len(info["guest_cols"]) and sc.v[v]["fit_xg"].dtype == np.float32
    assert abs(len(sc.v["final"]["ev_y"]) - N * cfg["prep"]["test_frac"]) <= 1
    sens = ML.load_sens(tmp_path, name)
    assert sens["G"].shape[1] == pre["clean_guest"] and len(sens["rank_g"]) == pre["nonzero_guest"]
    assert np.array_equal(sens["y"][~sens["fit"]], sc.v["final"]["ev_y"])
    g = sorted(sens["rank_g"][:2].tolist())
    assert np.array_equal(ML.gather([(sens["G"], g)], np.flatnonzero(sens["fit"])), sc.v["final"]["fit_xg"])


def test_预筛只读tune视图fit行(fake_xgb_imp):
    from modules.m5_modeling.components import method_screen_data as MD
    df, y = synth_frame()
    num, cat = df[NUM].to_numpy(np.float32), df[CAT]
    cfg = small_cfg()
    P = MS.Params(cfg)
    te, va = MD.simple_split(y, cfg["prep"])
    tfit = ~te & ~va
    guest = ML.party_vars(42, NUM, CAT)
    a = ML.select_columns(num, cat, y, tfit, guest, P, 2, NUM)
    y2, num2, cat2 = y.copy(), num.copy(), cat.copy()
    y2[~tfit] = 1 - y2[~tfit]                             # 改验证行与测试行的标签与特征
    num2[~tfit] = 0.0
    cat2.loc[~tfit, "Var10"] = "z"
    assert ML.select_columns(num2, cat2, y2, tfit, guest, P, 2, NUM) == a


def test_排序与取列(fake_xgb_imp):
    P = MS.Params(small_cfg())
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = (X[:, 3] + 0.1 * rng.normal(size=300) > 0).astype(int)
    r = ML.importance_rank(P, X, y, "total_gain")
    assert r[0] == 3 and len(set(r)) == len(r)
    sub = ML.importance_rank(P, X, y, "weight", cols=[0, 1])
    assert set(sub) <= {0, 1} and sub == [j for j in r if j in (0, 1)]
    assert ML.take([4, 0, 2], 2) == [0, 4] and ML.take([4, 0, 2], ML.ALL) is None


def test_敏感性(tmp_path, fake_xgb_imp):
    df, y = synth_frame()
    num, cat = df[NUM].to_numpy(np.float32), df[CAT]
    cfg = small_cfg()
    P = MS.Params(cfg)
    ML.build("appetency", num, cat, y, tmp_path, cfg["prep"], P, 2, NUM, CAT)
    res = {"configs": {"central": {"lr": 0.3, "depth": 3, "lam": 1.0, "col": 1.0, "rounds": 5},
                       "guest": {"lr": 0.3, "depth": 3, "lam": 1.0, "col": 0.7, "rounds": 5}}}
    out = ML.sensitivity(P, res, ML.load_sens(tmp_path, ML.name_of("appetency")), [1, 2, ML.ALL], 2)
    rows = out["rows"]
    assert {r["k"] for r in rows} == {"1", "2", ML.ALL} and out["k_main"] == 2
    main = [r for r in rows if r["k"] == "2" and r["method"] in ("F_opt_p", "G_t")]
    assert all(r["vs_k_main"] is None for r in main)
    other = [r for r in rows if r["k"] != "2" and r["method"] == "F_opt_p"]
    assert all(r["vs_k_main"]["ci_low"] <= r["vs_k_main"]["ci_high"] for r in other)
    gain = [r for r in rows if r["method"] == "F_opt_p − G_t"]
    assert len(gain) == 3 and all(0 <= r["federal_gain"]["p"] <= 1 for r in gain)
    ML.write_json(tmp_path / "s.json", out)
    assert json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))["k_main"] == 2
