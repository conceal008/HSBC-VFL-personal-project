"""M7 · B_T1 编码披露度量的单元测试。只用程序生成的合成数据。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m7_security.components import code_disclosure as D  # noqa: E402

K = 2
SEED = 7


def test_粗化把未允许叶子记为抑制():
    codes = np.array([[1, 5], [2, 5], [1, 6]])
    out = D.coarse_codes(codes, [{1}, {5, 6}])
    assert out.tolist() == [[1, 5], [D.SUPPRESSED, 5], [1, 6]]


def test_等价类按不同客户计():
    codes = np.array([[1, 1], [1, 1], [1, 1], [2, 2]])
    cust = np.array([10, 10, 11, 12])
    ids, table = D.tuple_ids(codes)
    assert len(table) == 2
    sizes = D.class_sizes(ids, cust)
    assert sizes.tolist() == [2, 2, 2, 1]
    u = D.uniqueness(codes, cust, K)
    assert u["distinct_tuples"] == 2 and u["customers"] == 3
    assert u["row_share_class_lt_k"] == 0.25 and u["row_share_unique"] == 0.25
    assert abs(u["customer_share_any_row_lt_k"] - 1 / 3) < 1e-12


def test_组合级抑制只保留足够大的组合():
    codes = np.array([[1, 1], [1, 1], [2, 2], [3, 3]])
    cust = np.array([1, 2, 3, 3])
    allowed = D.tuple_k_allowed(codes, cust, K)
    assert allowed == {(1, 1)}
    out = D.tuple_k_suppress(np.array([[1, 1], [2, 2]]), allowed)
    assert out.tolist() == [[1, 1], [D.SUPPRESSED, D.SUPPRESSED]]


def test_组合均值与缺失回退():
    ids = np.array([0, 0, 1])
    x = np.array([[1.0, D.MISSING], [0.0, D.MISSING], [1.0, 1.0]])
    means, gmean = D.region_means(ids, x)
    assert np.allclose(means[0], [0.5, 1.0])       # 组 0 第二列全缺失 → 全体均值 1.0
    assert np.allclose(gmean, [2 / 3, 1.0])
    scores = D.infer_scores(np.array([1, -1]), means, gmean)
    assert np.allclose(scores, [[1.0, 1.0], [2 / 3, 1.0]])


def test_查表未见组合记负一():
    table = np.array([[1, 1], [2, 2]])
    assert D.lookup_codes(np.array([[2, 2], [9, 9]]), table).tolist() == [1, -1]


def test_推断AUROC与单类字段():
    truth = np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, D.MISSING]])
    scores = np.array([[0.1, 0.5], [0.9, 0.5], [0.2, 0.5], [0.8, 0.5]])
    auc = D.attribute_auc(scores, truth)
    assert auc[0] == 1.0 and np.isnan(auc[1])


def test_确定推出比例与正确率():
    ids_tr = np.array([0, 0, 1, 1])
    x_tr = np.array([[1.0], [1.0], [0.0], [1.0]])     # 组 0 纯，组 1 不纯
    ids_ev = np.array([0, 1, -1])
    x_ev = np.array([[1.0], [0.0], [1.0]])
    share, correct, pos_found = D.certainty(ids_tr, x_tr, ids_ev, x_ev)
    assert abs(share[0] - 1 / 3) < 1e-12 and correct[0] == 1.0
    assert pos_found[0] == 0.5                          # 两个取值为 1 的评估行中，组 0 那行被确定推出


def test_端到端摘要与L1对照():
    rng = np.random.default_rng(SEED)
    x = rng.integers(0, 2, size=(400, 3)).astype(float)
    codes = x[:, :2].astype(int)                         # 编码完全由前两个字段决定
    out = D.disclosure_summary(codes[:300], x[:300], codes[300:], x[300:])
    assert out["auc"][0] == 1.0 and out["auc"][1] == 1.0
    assert abs(out["auc"][2] - 0.5) < 0.15              # 第三个字段与编码无关
    assert out["certain_share"][0] == 1.0 and out["certain_accuracy"][0] == 1.0
    assert out["positive_determined"][0] == 1.0
    assert out["unseen_tuple_share"][0] == 0.0
    seg_auc = D.segment_disclosure(np.full((100, 3), 0.5), x[300:])
    assert np.allclose(seg_auc, 0.5)


def test_seed维度汇总():
    m, sd = D.mean_over([np.array([1.0, np.nan]), np.array([3.0, np.nan])])
    assert m[0] == 2.0 and sd[0] == 1.0 and np.isnan(m[1])


def test_重建发布与独热和S5G1做法一致():
    from modules.m5_modeling.components import g3_arms as A
    rng = np.random.default_rng(SEED)
    xh = {"train": rng.integers(0, 2, size=(300, 5)).astype(float),
          "eval": rng.integers(0, 2, size=(50, 5)).astype(float)}
    cust = np.arange(300) // 3                           # 每位客户 3 行
    coarse = D.release_codes(xh, cust, 3, 2, K, SEED, tuple_k=False)
    emb = A.tree_embedding(xh["train"], 3, 2, SEED)
    raw = {n: A.leaf_codes(emb, v) for n, v in xh.items()}
    allowed = A.k_coarsen(raw["train"], cust, K)
    for n in xh:
        assert np.array_equal(D.one_hot_release(coarse[n], coarse["train"]), A.one_hot_codes(raw[n], allowed))
    tk = D.release_codes(xh, cust, 3, 2, K, SEED, tuple_k=True)
    sizes = D.class_sizes(D.tuple_ids(tk["train"])[0], cust)
    kept = (tk["train"] != D.SUPPRESSED).any(axis=1)
    assert (sizes[kept] >= K).all()
    assert D.one_hot_release(np.full((2, 2), D.SUPPRESSED), np.full((2, 2), D.SUPPRESSED)).shape == (2, 0)
