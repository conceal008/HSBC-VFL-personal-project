"""M5 · G3 五臂对比的 FedAds 部分：读取官方 CSV、按用户号加盐哈希分组、编码。

FedAds（官方纵向切分）每格是单元素 JSON 列表，例如 ["123"]。主动方（l_*，持转化标签）对应内地，
被动方（f_*，5 个低基数特征）对应香港。`l_u_fea_1` 近乎每行一值，作用户号：分组与客户簇重抽样都以它为单位。

筛选只用标准库，可在维护者本机直接运行（数据不出本机目录，只有筛选后的行转入会话）；编码部分按需导入 numpy / pandas。
"""
from __future__ import annotations

import gzip
import hashlib
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

ID_COL = "sample_id"
LABEL_COL = "label"
USER_COL = "l_u_fea_1"
GROUP_COL = "g3_group"
LEADER_COLUMNS: List[str] = (["l_i_fea_%d" % i for i in range(1, 11)]
                             + ["l_u_fea_%d" % i for i in range(1, 7)] + ["l_c_fea"])
# 用户号只作分组与客户簇：各组用户不相交，若作特征，评估组全为未见值，训练与评估分布不一致
LEADER_FEATURES: List[str] = [c for c in LEADER_COLUMNS if c != USER_COL]
FOLLOWER_FEATURES: List[str] = ["f_u_fea_1", "f_u_fea_2", "f_uc_fea_1", "f_uc_fea_2", "f_c"]
TRAIN_COLUMNS: List[str] = [ID_COL, LABEL_COL] + LEADER_COLUMNS + FOLLOWER_FEATURES
MISSING = -1.0
HEX_BASE = 16
CELL_PREFIX = '["'
CELL_SUFFIX = '"]'
ERROR_PREVIEW_CHARS = 40


# ————————————————— 筛选（标准库） —————————————————

def unwrap(cell: str) -> str:
    """'["x"]' → 'x'。多元素或格式不符时报错，避免静默截断。"""
    if not (cell.startswith(CELL_PREFIX) and cell.endswith(CELL_SUFFIX)) or '","' in cell:
        raise ValueError("不是单元素列表：%r" % cell[:ERROR_PREVIEW_CHARS])
    return cell[len(CELL_PREFIX):-len(CELL_SUFFIX)]


def user_bucket(user: str, salt: str, n_buckets: int) -> int:
    """与 g3_santander.customer_bucket 相同的加盐 SHA-256 取模，只是输入为用户号字符串（有符号 64 位整数）。"""
    digest = hashlib.sha256(("%s|%s" % (salt, int(user))).encode("utf-8")).hexdigest()
    return int(digest, HEX_BASE) % n_buckets


def group_for(bucket: int, groups: Dict[str, Dict], source: str) -> str:
    """桶号 → 组名，只看来源（train/test 文件）相符的组；组区间左闭右开且互不重叠。"""
    for name, spec in groups.items():
        lo, hi = spec["buckets"]
        if spec["source"] == source and lo <= bucket < hi:
            return name
    return ""


def validate_groups(groups: Dict[str, Dict]) -> None:
    """全部组（不分来源）的桶区间互不重叠：同一用户最多进入一个组，训练、内层验证、开发评估与封存段用户不相交。"""
    spans = sorted((spec["buckets"][0], spec["buckets"][1], name) for name, spec in groups.items())
    for lo, hi, name in spans:
        if not lo < hi:
            raise ValueError("组 %s 的桶区间为空" % name)
    for (_, hi1, a), (lo2, _, b) in zip(spans, spans[1:]):
        if lo2 < hi1:
            raise ValueError("组 %s 与 %s 的桶区间重叠" % (a, b))


def select_lines(lines: Iterable[str], header: List[str], groups: Dict[str, Dict], salt: str,
                 n_buckets: int, source: str) -> Iterator[Tuple[str, str]]:
    """逐行按用户号分组，产出（组名，原始行）。不解析标签。"""
    user_idx = header.index(USER_COL)
    for line in lines:
        cells = line.rstrip("\n").split(",")
        name = group_for(user_bucket(unwrap(cells[user_idx]), salt, n_buckets), groups, source)
        if name:
            yield name, line.rstrip("\n")


def select_file(src: str, out: str, groups: Dict[str, Dict], salt: str, n_buckets: int, source: str,
                start: int = 0, limit: Optional[int] = None, append: bool = False) -> Dict[str, int]:
    """从 src 的第 start 个数据行起（最多 limit 行）筛选，写入 out（gzip，首列为组名）。append 时不写表头。"""
    validate_groups(groups)
    counts = {name: 0 for name, spec in groups.items() if spec["source"] == source}
    with open(src, encoding="utf-8") as fh, gzip.open(out, "at" if append else "wt", encoding="utf-8") as fo:
        header = fh.readline().rstrip("\n").split(",")
        if not append:
            fo.write(",".join([GROUP_COL] + header) + "\n")
        for i, line in enumerate(fh):
            if i < start:
                continue
            if limit is not None and i >= start + limit:
                break
            for name, raw in select_lines([line], header, groups, salt, n_buckets, source):
                counts[name] += 1
                fo.write(name + "," + raw + "\n")
    return counts


# ————————————————— 解析与编码（numpy / pandas） —————————————————

def parse_selected(path: str):
    """读取筛选结果：每格去掉列表外壳，标签转 0/1；测试文件缺的列不补。"""
    import pandas as pd
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in df.columns:
        if col not in (GROUP_COL, ID_COL):
            df[col] = df[col].map(unwrap)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def fit_encoder(train, high_card_threshold: int) -> Dict[str, Tuple[str, Dict[str, float]]]:
    """只用训练组：主动方基数 > 阈值的列做频数编码（训练中出现次数），其余列与全部被动方列做序号编码。不用标签。"""
    enc: Dict[str, Tuple[str, Dict[str, float]]] = {}
    for col in LEADER_FEATURES + FOLLOWER_FEATURES:
        counts = train[col].value_counts()
        if col in LEADER_FEATURES and len(counts) > high_card_threshold:
            enc[col] = ("freq", {k: float(v) for k, v in counts.items()})
        else:
            enc[col] = ("ordinal", {k: float(i) for i, k in enumerate(sorted(counts.index))})
    return enc


def _encode_cols(df, cols: List[str], enc):
    import numpy as np
    out = np.empty((len(df), len(cols)))
    for j, col in enumerate(cols):
        kind, table = enc[col]
        unseen = 0.0 if kind == "freq" else MISSING
        out[:, j] = df[col].map(table).fillna(unseen).to_numpy(dtype=float)
    return out


def encode(df, enc):
    """返回（主动方矩阵，被动方矩阵）。频数编码中未见值为 0，序号编码中未见值为 MISSING。"""
    return _encode_cols(df, LEADER_FEATURES, enc), _encode_cols(df, FOLLOWER_FEATURES, enc)


def follower_one_hot(df, enc):
    """被动方各类别的指示列（类别取训练组出现过的），供 L1 返回分段内类别占比。"""
    import numpy as np
    cols = []
    for col in FOLLOWER_FEATURES:
        _, table = enc[col]
        for value in sorted(table, key=table.get):
            cols.append((df[col] == value).to_numpy(dtype=float))
    return np.column_stack(cols) if cols else np.zeros((len(df), 0))


def summary_counts(parts: List[Tuple[str, Any]]) -> List[Dict[str, object]]:
    """各组行数、用户数、正例行数与有正例的用户数。只供汇总输出。"""
    rows = []
    for name, df in parts:
        pos = df[df[LABEL_COL] == 1]
        rows.append({"group": name, "rows": int(len(df)), "users": int(df[USER_COL].nunique()),
                     "positive_rows": int(len(pos)), "positive_users": int(pos[USER_COL].nunique())})
    return rows
