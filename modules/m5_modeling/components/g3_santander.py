"""M5 · G3 五臂对比的 Santander 数据准备（S6.P3 预注册的主数据集）。

两段式，原始数据与派生行数据都只在维护者本机或会话临时目录，不进仓库：

1. ``select_rows``：在本机流式读原始 ``train_ver2.csv``，按客户号加盐哈希分桶，只保留各组需要的月份，写出压缩 CSV。
2. ``build_split`` / ``encode``：在分析环境里把「本月特征 → 次月标签」配对，做内地 19 列、香港 25 列的业务切分与编码。

业务切分沿用 10/21 汇报与 HSBC mock 的口径：内地（主动方，持标签）为客户属性，香港（被动方）为 24 个产品持有标记与活跃度。
主标签 ``label_new_any``：次月持有至少一个本月未持有的产品。香港自有标签 ``hk_next_active``（B_T2 用）：次月活跃度为 1。
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

ID_COL = "ncodpers"
DATE_COL = "fecha_dato"
PRODUCTS = (
    "ind_ahor_fin_ult1", "ind_aval_fin_ult1", "ind_cco_fin_ult1", "ind_cder_fin_ult1", "ind_cno_fin_ult1",
    "ind_ctju_fin_ult1", "ind_ctma_fin_ult1", "ind_ctop_fin_ult1", "ind_ctpp_fin_ult1", "ind_deco_fin_ult1",
    "ind_deme_fin_ult1", "ind_dela_fin_ult1", "ind_ecue_fin_ult1", "ind_fond_fin_ult1", "ind_hip_fin_ult1",
    "ind_plan_fin_ult1", "ind_pres_fin_ult1", "ind_reca_fin_ult1", "ind_tjcr_fin_ult1", "ind_valo_fin_ult1",
    "ind_viv_fin_ult1", "ind_nomina_ult1", "ind_nom_pens_ult1", "ind_recibo_ult1",
)
ACTIVITY = "ind_actividad_cliente"
GUEST_CATEGORICAL = ("ind_empleado", "pais_residencia", "sexo", "indrel_1mes", "tiprel_1mes", "indresi",
                     "indext", "canal_entrada", "indfall", "nomprov", "segmento")
GUEST_NUMERIC = ("age", "ind_nuevo", "antiguedad", "indrel", "tipodom", "cod_prov", "renta")
GUEST_DATE = ("fecha_alta",)
GUEST_FEATURES = GUEST_NUMERIC + GUEST_CATEGORICAL + ("tenure_days",)
HOST_FEATURES = PRODUCTS + (ACTIVITY,)
LABEL = "label_new_any"
HK_LABEL = "hk_next_active"
GROUP_COL = "g3_group"
MISSING = -1.0
ANTIGUEDAD_SENTINEL = -999999.0
HEX_BASE = 16
MONTHS_PER_YEAR = 12
YEAR_END = 4          # "2016-03-28"[0:4]
MONTH_START = 5       # "2016-03-28"[5:7]
YEAR_MONTH_LEN = 7    # "2016-03"


def customer_bucket(customer_id, salt: str, n_buckets: int) -> int:
    """客户号加盐 SHA-256 后取模分桶。同一客户在任何月份都落在同一桶，保证各组客户不相交。"""
    digest = hashlib.sha256(("%s|%s" % (salt, int(customer_id))).encode("utf-8")).hexdigest()
    return int(digest, HEX_BASE) % n_buckets


def group_of_bucket(bucket: int, groups: Dict[str, Dict]) -> str:
    """桶号 → 组名；不在任何组的桶返回空串。组的桶区间左闭右开且互不重叠。"""
    for name, spec in groups.items():
        lo, hi = spec["buckets"]
        if lo <= bucket < hi:
            return name
    return ""


def validate_groups(groups: Dict[str, Dict]) -> None:
    spans = sorted((spec["buckets"][0], spec["buckets"][1], name) for name, spec in groups.items())
    for (lo1, hi1, n1), (lo2, _, n2) in zip(spans, spans[1:]):
        if lo2 < hi1:
            raise ValueError("组 %s 与 %s 的桶区间重叠" % (n1, n2))
    for lo, hi, name in spans:
        if not lo < hi:
            raise ValueError("组 %s 的桶区间为空" % name)


def select_rows(csv_path: str, out_path: str, groups: Dict[str, Dict], salt: str, n_buckets: int,
                chunksize: int) -> Dict[str, int]:
    """流式读原始文件，只保留各组客户在该组月份内的行，加一列组名后写出（gzip CSV）。返回各组行数。"""
    validate_groups(groups)
    months = {name: set(spec["months"]) for name, spec in groups.items()}
    cache: Dict[int, str] = {}
    counts = {name: 0 for name in groups}
    first = True
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=str, keep_default_na=False):
        ids = pd.to_numeric(chunk[ID_COL].str.strip(), errors="coerce")
        chunk = chunk.loc[ids.notna()].copy()
        ids = ids.loc[ids.notna()].astype(np.int64)
        for cid in ids.unique():
            if cid not in cache:
                cache[cid] = group_of_bucket(customer_bucket(cid, salt, n_buckets), groups)
        grp = ids.map(cache)
        month = chunk[DATE_COL].str.slice(0, YEAR_MONTH_LEN)
        keep = np.zeros(len(chunk), dtype=bool)
        for name in groups:
            keep |= ((grp == name) & month.isin(months[name])).to_numpy()
        out = chunk.loc[keep].copy()
        out[GROUP_COL] = grp.loc[keep].to_numpy()
        for name in groups:
            counts[name] += int((out[GROUP_COL] == name).sum())
        out.to_csv(out_path, mode="w" if first else "a", header=first, index=False, compression="gzip")
        first = False
    return counts


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.strip().replace({"": None, "NA": None}), errors="coerce")


def _month_index(series: pd.Series) -> pd.Series:
    """'2016-03-28' → 2016×12+3，用于判断「次月」。"""
    s = series.astype(str)
    return (s.str.slice(0, YEAR_END).astype(int) * MONTHS_PER_YEAR
            + s.str.slice(MONTH_START, YEAR_MONTH_LEN).astype(int))


def build_split(rows: pd.DataFrame, feature_months: Sequence[str]) -> pd.DataFrame:
    """把同一客户「本月 → 次月」配对；只保留次月记录存在且恰好间隔一个月的行。

    返回列：客户号、特征月、内地原始列、香港原始列、主标签、香港自有标签。
    """
    df = rows.copy()
    df["_m"] = _month_index(df[DATE_COL])
    df[ID_COL] = _num(df[ID_COL]).astype(np.int64)
    for col in PRODUCTS + (ACTIVITY,):
        df[col] = _num(df[col]).fillna(0.0)
    nxt = df[[ID_COL, "_m"] + list(PRODUCTS) + [ACTIVITY]].copy()
    nxt["_m"] = nxt["_m"] - 1
    nxt = nxt.rename(columns={c: c + "__next" for c in list(PRODUCTS) + [ACTIVITY]})
    cur = df.loc[df[DATE_COL].astype(str).str.slice(0, YEAR_MONTH_LEN).isin(set(feature_months))]
    pair = cur.merge(nxt, on=[ID_COL, "_m"], how="inner")
    now = pair[list(PRODUCTS)].to_numpy()
    later = pair[[c + "__next" for c in PRODUCTS]].to_numpy()
    pair[LABEL] = ((later > 0) & (now <= 0)).any(axis=1).astype(int)
    pair[HK_LABEL] = (pair[ACTIVITY + "__next"] > 0).astype(int)
    keep = [ID_COL, DATE_COL] + list(GUEST_NUMERIC + GUEST_CATEGORICAL + GUEST_DATE) + list(HOST_FEATURES) + [LABEL, HK_LABEL]
    return pair[keep].sort_values([ID_COL, DATE_COL]).reset_index(drop=True)


def fit_encoder(train: pd.DataFrame) -> Dict[str, List[str]]:
    """类别列的取值表只从训练组学（学到的预处理状态，不入仓库）。"""
    return {c: sorted(set(train[c].astype(str).str.strip())) for c in GUEST_CATEGORICAL}


def encode(split: pd.DataFrame, encoder: Dict[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
    """返回（内地矩阵，香港矩阵），列顺序为 GUEST_FEATURES 与 HOST_FEATURES；缺失与未见类别记为 MISSING。"""
    g = pd.DataFrame(index=split.index)
    for c in GUEST_NUMERIC:
        v = _num(split[c])
        if c == "antiguedad":
            v = v.where(v != ANTIGUEDAD_SENTINEL)
        g[c] = v
    for c in GUEST_CATEGORICAL:
        lookup = {k: i for i, k in enumerate(encoder[c])}
        g[c] = split[c].astype(str).str.strip().map(lookup)
    alta = pd.to_datetime(split["fecha_alta"].astype(str).str.strip(), errors="coerce")
    dato = pd.to_datetime(split[DATE_COL].astype(str).str.strip(), errors="coerce")
    g["tenure_days"] = (dato - alta).dt.days
    xg = g[list(GUEST_FEATURES)].astype(float).fillna(MISSING).to_numpy()
    xh = split[list(HOST_FEATURES)].astype(float).fillna(MISSING).to_numpy()
    return xg, xh


def eval_buckets_needed(train_rate: float, customers_per_bucket: float, target_positives: int,
                        safety: float) -> int:
    """开发评估集与封存确认集各需要多少个桶：只用训练月份的正例率与评估月的客户数（不看评估月标签）。"""
    if not 0 < train_rate < 1 or customers_per_bucket <= 0:
        raise ValueError("正例率须在 (0,1)，每桶客户数须 > 0")
    return int(math.ceil(target_positives * safety / (train_rate * customers_per_bucket)))


def months_between(start: str, end: str) -> List[str]:
    """'2015-01' 到 '2016-02'（含两端）的月份列表。"""
    y0, m0 = int(start[:YEAR_END]), int(start[MONTH_START:YEAR_MONTH_LEN])
    y1, m1 = int(end[:YEAR_END]), int(end[MONTH_START:YEAR_MONTH_LEN])
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append("%04d-%02d" % (y, m))
        m += 1
        if m > MONTHS_PER_YEAR:
            y, m = y + 1, 1
    return out


def summary_counts(frames: Iterable[Tuple[str, pd.DataFrame]]) -> List[Dict]:
    """各组的行数、客户数与正例数（只用于核对样本量规则；客户数与正例数属公开竞赛数据的样本规模）。"""
    return [{"group": name, "rows": int(len(df)), "customers": int(df[ID_COL].nunique()),
             "positives": int(df[LABEL].sum()), "positive_customers": int(df.loc[df[LABEL] == 1, ID_COL].nunique())}
            for name, df in frames]
