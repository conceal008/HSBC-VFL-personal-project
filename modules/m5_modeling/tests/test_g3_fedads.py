"""M5 · G3 FedAds 组件的单元测试。只用程序生成的合成数据。"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m5_modeling.components import g3_fedads as F  # noqa: E402
from modules.m5_modeling.components import g3_santander as G  # noqa: E402

SALT = "G3-FEDADS-TEST"
N_BUCKETS = 10
THRESHOLD = 3


def _cell(v):
    return '["%s"]' % v


def _write(path, n_rows, users, test=False):
    header = list(F.TRAIN_COLUMNS)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for i in range(n_rows):
            row = {F.ID_COL: str(i), F.LABEL_COL: _cell(int(i % 5 == 0))}
            for j, col in enumerate(F.LEADER_COLUMNS):
                row[col] = _cell(i % (j + 2))
            row[F.USER_COL] = _cell(users[i % len(users)])
            for j, col in enumerate(F.FOLLOWER_FEATURES):
                row[col] = _cell(i % (j + 2))
            fh.write(",".join(row[h] for h in header) + "\n")


def _groups():
    return {"train": {"buckets": [0, 6], "source": "train"},
            "inner_val": {"buckets": [6, 8], "source": "train"},
            "dev_eval": {"buckets": [8, 10], "source": "test"}}


def test_单元素列表外壳():
    assert F.unwrap('["-12"]') == "-12"
    with pytest.raises(ValueError):
        F.unwrap('["1","2"]')
    with pytest.raises(ValueError):
        F.unwrap("12")


def test_分桶与Santander同一公式():
    for u in ("-922337203", "42"):                   # 有符号用户号；位数短于银行卡号，避免门禁 1 误判
        assert F.user_bucket(u, SALT, N_BUCKETS) == G.customer_bucket(int(u), SALT, N_BUCKETS)


def test_组区间不得重叠():
    F.validate_groups(_groups())
    with pytest.raises(ValueError):
        F.validate_groups({"a": {"buckets": [0, 5], "source": "train"}, "b": {"buckets": [4, 8], "source": "test"}})
    with pytest.raises(ValueError):
        F.validate_groups({"a": {"buckets": [3, 3], "source": "train"}})
    assert F.group_for(9, _groups(), "train") == "" and F.group_for(9, _groups(), "test") == "dev_eval"


def test_筛选分段续写与解析(tmp_path):
    users = [str(-1000 - k) for k in range(40)]
    src = tmp_path / "train.csv"
    _write(src, 120, users)
    out = tmp_path / "sel.csv.gz"
    c1 = F.select_file(str(src), str(out), _groups(), SALT, N_BUCKETS, "train", start=0, limit=50)
    c2 = F.select_file(str(src), str(out), _groups(), SALT, N_BUCKETS, "train", start=50, append=True)
    whole = tmp_path / "whole.csv.gz"
    c3 = F.select_file(str(src), str(whole), _groups(), SALT, N_BUCKETS, "train")
    assert {k: c1[k] + c2[k] for k in c1} == c3 and "dev_eval" not in c3
    with gzip.open(out, "rt") as a, gzip.open(whole, "rt") as b:
        assert a.read() == b.read()
    df = F.parse_selected(str(out))
    assert set(df[F.GROUP_COL]) <= {"train", "inner_val"} and set(df[F.LABEL_COL]) <= {0, 1}
    for name, d in df.groupby(F.GROUP_COL):
        lo, hi = _groups()[name]["buckets"]
        assert all(lo <= F.user_bucket(u, SALT, N_BUCKETS) < hi for u in d[F.USER_COL])
    rows = F.summary_counts([("all", df)])
    assert rows[0]["rows"] == len(df) and rows[0]["positive_rows"] == int(df[F.LABEL_COL].sum())


def test_编码频数序号与未见值(tmp_path):
    users = [str(k) for k in range(30)]
    src = tmp_path / "t.csv"
    _write(src, 60, users)
    out = tmp_path / "s.csv.gz"
    F.select_file(str(src), str(out), {"train": {"buckets": [0, N_BUCKETS], "source": "train"}}, SALT, N_BUCKETS, "train")
    df = F.parse_selected(str(out))
    enc = F.fit_encoder(df, THRESHOLD)
    assert F.USER_COL not in enc and F.USER_COL not in F.LEADER_FEATURES
    assert enc["l_i_fea_10"][0] == "freq" and enc["l_i_fea_1"][0] == "ordinal"
    assert all(enc[c][0] == "ordinal" for c in F.FOLLOWER_FEATURES)
    xl, xf = F.encode(df, enc)
    assert xl.shape == (len(df), len(F.LEADER_FEATURES)) and xf.shape == (len(df), len(F.FOLLOWER_FEATURES))
    unseen = df.copy()
    unseen["l_i_fea_10"] = "999999"
    unseen["l_i_fea_1"] = "zzz"
    xl2, _ = F.encode(unseen, enc)
    assert (xl2[:, F.LEADER_FEATURES.index("l_i_fea_10")] == 0).all()
    assert (xl2[:, F.LEADER_FEATURES.index("l_i_fea_1")] == F.MISSING).all()
    oh = F.follower_one_hot(df, enc)
    assert oh.shape[1] == sum(len(enc[c][1]) for c in F.FOLLOWER_FEATURES)
    assert np.allclose(oh.reshape(len(df), -1).sum(axis=1), len(F.FOLLOWER_FEATURES))
