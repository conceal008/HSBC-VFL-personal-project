"""M5 · G3 组件的单元测试：数据准备、各臂与 SGB 参数映射。只用程序生成的合成数据。"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m5_modeling.components import g3_arms as A  # noqa: E402
from modules.m5_modeling.components import g3_santander as G  # noqa: E402
from modules.m5_modeling.components import g3_sgb as SG  # noqa: E402

SALT = "G3-TEST"
N_BUCKETS = 100
K = 3
SEED = 5


def _raw_rows(customers=(11, 12, 13), months=("2015-01-28", "2015-02-28", "2015-03-28")):
    rows = []
    for c in customers:
        for i, m in enumerate(months):
            row = {G.DATE_COL: m, G.ID_COL: " %d" % c}
            for col in G.GUEST_NUMERIC:
                row[col] = " 30"
            for col in G.GUEST_CATEGORICAL:
                row[col] = "A" if c % 2 else "B"
            row["fecha_alta"] = "2014-01-01"
            for p in G.PRODUCTS:
                row[p] = "0"
            row[G.ACTIVITY] = "1" if i else "0"
            if c == 11 and i == 1:
                row[G.PRODUCTS[0]] = "1"          # 客户 11 在第二个月新增产品
            rows.append(row)
    return pd.DataFrame(rows)


def test_同一客户始终同一桶且组不重叠():
    assert G.customer_bucket(11, SALT, N_BUCKETS) == G.customer_bucket("11", SALT, N_BUCKETS)
    groups = {"a": {"buckets": [0, 10]}, "b": {"buckets": [10, 20]}}
    G.validate_groups(groups)
    assert G.group_of_bucket(5, groups) == "a" and G.group_of_bucket(15, groups) == "b"
    assert G.group_of_bucket(50, groups) == ""
    with pytest.raises(ValueError):
        G.validate_groups({"a": {"buckets": [0, 10]}, "b": {"buckets": [5, 20]}})
    with pytest.raises(ValueError):
        G.validate_groups({"a": {"buckets": [3, 3]}})


def test_配对标签按次月新增产品():
    pairs = G.build_split(_raw_rows(), ["2015-01", "2015-02"])
    assert len(pairs) == 6                                    # 3 客户 × 2 个有次月记录的特征月
    lab = pairs.set_index([G.ID_COL, G.DATE_COL])[G.LABEL]
    assert lab[(11, "2015-01-28")] == 1 and lab[(11, "2015-02-28")] == 0
    assert lab[(12, "2015-01-28")] == 0
    assert pairs[G.HK_LABEL].eq(1).all()                      # 次月活跃度为 1


def test_缺月不配对():
    rows = _raw_rows(months=("2015-01-28", "2015-03-28"))
    assert len(G.build_split(rows, ["2015-01"])) == 0


def test_编码列数与未见类别():
    pairs = G.build_split(_raw_rows(), ["2015-01", "2015-02"])
    enc = G.fit_encoder(pairs[pairs[G.ID_COL] != 12])
    xg, xh = G.encode(pairs, enc)
    assert xg.shape[1] == len(G.GUEST_FEATURES) and xh.shape[1] == len(G.HOST_FEATURES)
    unseen = pairs.copy()
    unseen["segmento"] = "Z"
    xg2, _ = G.encode(unseen, enc)
    assert (xg2[:, G.GUEST_FEATURES.index("segmento")] == G.MISSING).all()


def test_流式筛选只保留组内月份(tmp_path):
    rows = _raw_rows(customers=tuple(range(1, 40)))
    src = tmp_path / "raw.csv"
    rows.to_csv(src, index=False)
    groups = {"g": {"buckets": [0, N_BUCKETS // 2], "months": ["2015-02"]}}
    out = tmp_path / "sel.csv.gz"
    counts = G.select_rows(str(src), str(out), groups, SALT, N_BUCKETS, chunksize=17)
    with gzip.open(out, "rt") as fh:
        sel = pd.read_csv(fh, dtype=str)
    assert counts["g"] == len(sel) > 0
    assert sel[G.DATE_COL].str.startswith("2015-02").all()
    expected = sum(G.customer_bucket(c, SALT, N_BUCKETS) < N_BUCKETS // 2 for c in range(1, 40))
    assert len(sel) == expected


def test_评估桶数规则与月份列表():
    assert G.eval_buckets_needed(0.05, 100.0, 700, 1.0) == 140
    with pytest.raises(ValueError):
        G.eval_buckets_needed(0.0, 100.0, 700, 1.0)
    assert G.months_between("2015-11", "2016-02") == ["2015-11", "2015-12", "2016-01", "2016-02"]


def test_样本计数():
    pairs = G.build_split(_raw_rows(), ["2015-01", "2015-02"])
    rows = G.summary_counts([("all", pairs)])
    assert rows[0]["positives"] == 1 and rows[0]["customers"] == 3


def test_分段编号训练与评估一致():
    rng = np.random.default_rng(SEED)
    x = np.column_stack([rng.uniform(18, 80, 200), rng.integers(0, 3, 200)])
    cols = {"age": 0, "seg": 1}
    edges = A.segment_edges(x, cols, {"age": 4})
    s1 = A.segment_ids(x, cols, edges, {"seg": 3})
    s2 = A.segment_ids(x[:50], cols, edges, {"seg": 3})
    assert np.array_equal(s1[:50], s2)
    x_missing = x.copy()
    x_missing[:, 0] = -1
    assert len(np.unique(A.segment_ids(x_missing, cols, edges, {"seg": 3}))) <= 3


def test_k匿名抑制小分段():
    seg = np.array([0, 0, 0, 1, 1, 2])
    cust = np.array([1, 2, 3, 4, 4, 5])
    xh = np.arange(12, dtype=float).reshape(6, 2)
    table, gmean, info = A.k_anonymous_means(seg, cust, xh, K)
    assert set(table) == {0} and info["segments_suppressed"] == 2
    out = A.apply_segment_means(np.array([0, 1, 9]), table, gmean)
    assert np.allclose(out[0], xh[:3].mean(axis=0)) and np.allclose(out[1], gmean) and np.allclose(out[2], gmean)


def test_随机树嵌入k粗化():
    rng = np.random.default_rng(SEED)
    xh = rng.integers(0, 2, size=(300, 6)).astype(float)
    cust = np.arange(300)
    emb = A.tree_embedding(xh, 3, 2, SEED)
    codes = A.leaf_codes(emb, xh)
    allowed = A.k_coarsen(codes, cust, K)
    for t, keep in enumerate(allowed):
        for leaf in keep:
            assert (codes[:, t] == leaf).sum() >= K
    oh = A.one_hot_codes(codes, allowed)
    assert oh.shape[0] == 300 and oh.shape[1] == sum(len(k) for k in allowed)
    assert set(np.unique(oh)) <= {0.0, 1.0}


def test_打乱保留边际分布():
    x = np.arange(20, dtype=float).reshape(10, 2)
    s = A.shuffle_rows(x, SEED)
    assert np.array_equal(np.sort(s[:, 0]), x[:, 0]) and not np.array_equal(s, x)


def test_候选配置抽样确定且不越界():
    space = {"max_depth": [3, 4], "learning_rate": [0.1, 0.3], "num_boost_round": [10], "reg_lambda": [1.0]}
    a = A.sample_configs(space, 3, SEED)
    assert a == A.sample_configs(space, 3, SEED) and len(a) == 3
    with pytest.raises(ValueError):
        A.sample_configs(space, 5, SEED)


def test_调参选验证集最好的配置():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(400, 3))
    y = (x[:, 0] + rng.normal(scale=0.5, size=400) > 0).astype(int)
    configs = [{"max_depth": 2, "learning_rate": 0.3, "num_boost_round": 20, "reg_lambda": 1.0},
               {"max_depth": 1, "learning_rate": 0.01, "num_boost_round": 1, "reg_lambda": 1.0}]
    best, rows = A.tune(x[:300], y[:300], x[300:], y[300:], configs, 32, SEED, 1)
    assert best == configs[0] and len(rows) == 2


def test_SGB参数映射():
    p = SG.sgb_params({"num_boost_round": 10, "max_depth": 3, "learning_rate": 0.2, "reg_lambda": 1}, 0.03, 11)
    assert p["sketch_eps"] == 0.03 and p["objective"] == "logistic" and p["num_boost_round"] == 10


def _install_fake_secretflow(monkeypatch):
    """CI 环境没有 SecretFlow：用最小替身只检验 g3_sgb 的编排（设备、分区、参数、打分集与关闭），不检验密码学本身。"""
    import types

    calls = {"init": None, "params": None, "shutdown": 0}

    class PYU:
        def __init__(self, party):
            self.party = party

        def __call__(self, fn):
            return lambda: fn()

    class FedNdarray:
        def __init__(self, partitions, partition_way):
            self.parts = list(partitions.values())
            self.way = partition_way

    class Model:
        def predict(self, x):
            return np.hstack(x.parts).sum(axis=1)

    class Sgb:
        def __init__(self, heu):
            self.heu = heu

        def set_params(self, params):
            calls["params"] = params

        def fit(self, x, y):
            assert x.way == "vertical" and len(x.parts) == 2
            return Model()

    sf = types.ModuleType("secretflow")
    sf.init = lambda **kw: calls.__setitem__("init", kw)
    sf.shutdown = lambda: calls.__setitem__("shutdown", calls["shutdown"] + 1)
    sf.PYU = PYU
    sf.HEU = lambda cfg, field: ("heu", cfg["he_parameters"]["schema"], field)
    sf.utils = types.SimpleNamespace(testing=types.SimpleNamespace(
        heu_config=lambda holder, others: {"he_parameters": {"schema": "paillier"}, "holder": holder}))
    data = types.ModuleType("secretflow.data")
    ndarray = types.ModuleType("secretflow.data.ndarray")
    ndarray.FedNdarray = FedNdarray
    ndarray.PartitionWay = types.SimpleNamespace(VERTICAL="vertical")
    device = types.ModuleType("secretflow.device")
    device.reveal = lambda v: v
    ml = types.ModuleType("secretflow.ml")
    boost = types.ModuleType("secretflow.ml.boost")
    sgb_v = types.ModuleType("secretflow.ml.boost.sgb_v")
    sgb_v.Sgb = Sgb
    spu = types.ModuleType("spu")
    spu.FieldType = types.SimpleNamespace(FM64="FM64")
    for name, mod in {"secretflow": sf, "secretflow.data": data, "secretflow.data.ndarray": ndarray,
                      "secretflow.device": device, "secretflow.ml": ml, "secretflow.ml.boost": boost,
                      "secretflow.ml.boost.sgb_v": sgb_v, "spu": spu}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def test_SGB编排用替身检验(monkeypatch):
    calls = _install_fake_secretflow(monkeypatch)
    xg, xh = np.ones((4, 2)), np.ones((4, 3))
    params = SG.sgb_params({"num_boost_round": 2, "max_depth": 2, "learning_rate": 0.1, "reg_lambda": 1}, 0.05, 1)
    scores, timing = SG.fit_and_score(xg, xh, np.array([0, 1, 0, 1]), {"eval": (xg[:2], xh[:2])}, params, "ou")
    assert calls["init"]["parties"] == [SG.LABEL_PARTY, SG.PASSIVE_PARTY] and calls["init"]["debug_mode"] is True
    assert calls["params"] == params and calls["shutdown"] == 1
    assert np.allclose(scores["eval"], [5.0, 5.0]) and timing["fit_seconds"] >= 0
