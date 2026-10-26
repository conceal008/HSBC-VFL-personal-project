"""M7 · B_T1 叶子编码的披露度量：编码组合的等价类规模、按编码推断香港字段的能力、组合级 k-匿名抑制。

B_T1 由香港对每位对齐客户发出 k 粗化后的随机树叶子编码。内地本来就知道每行是哪位客户（对齐键），
所以要回答的不是「能否再识别」，而是：编码让内地对一位已识别客户的香港字段知道了多少。

两种度量：
- 等价类规模：同一编码组合下有多少位不同客户——小于 k 的组合本身就近乎指向个人；
- 字段推断：假设内地知道编码到特征区域的映射（最坏情形：知道树结构，或有少量已知香港字段的辅助客户），
  用训练发布中同一组合内的字段均值作为推断分数，在另一批客户上计算 AUROC 与「确定推出」比例。
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from modules.m5_modeling.components import g3_arms

SUPPRESSED = -1          # 被抑制的叶子（单棵树内客户数 < k，或组合级抑制）
MISSING = -1.0           # 与 g3_santander.MISSING 一致：香港字段缺失


# ————————————————— 编码与等价类 —————————————————

def coarse_codes(codes: np.ndarray, allowed: Sequence[Set[int]]) -> np.ndarray:
    """按每棵树允许发布的叶子集合粗化：不在集合内的叶子记为 SUPPRESSED。与 g3_arms.one_hot_codes 的发布内容等价。"""
    out = codes.astype(np.int64).copy()
    for t, keep in enumerate(allowed):
        out[~np.isin(out[:, t], sorted(keep)), t] = SUPPRESSED
    return out


def tuple_ids(codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """每行编码组合映射为整数编号。返回（编号，去重后的组合表）。"""
    table, inverse = np.unique(codes, axis=0, return_inverse=True)
    return inverse.reshape(-1).astype(np.int64), table


def class_sizes(ids: np.ndarray, customers: np.ndarray) -> np.ndarray:
    """每行所在等价类（同一组合）中不同客户的个数。"""
    pairs = np.unique(np.column_stack([ids, customers]), axis=0)
    per_class = np.bincount(pairs[:, 0], minlength=int(ids.max()) + 1 if len(ids) else 0)
    return per_class[ids]


def uniqueness(codes: np.ndarray, customers: np.ndarray, k: int) -> Dict[str, float]:
    """发布人群内编码组合的等价类统计：行与客户两个口径。"""
    ids, table = tuple_ids(codes)
    sizes = class_sizes(ids, customers)
    small = sizes < int(k)
    cust_small = np.unique(customers[small])
    n_cust = len(np.unique(customers))
    return {
        "rows": int(len(ids)),
        "customers": int(n_cust),
        "distinct_tuples": int(len(table)),
        "row_share_class_lt_k": float(small.mean()) if len(ids) else float("nan"),
        "row_share_unique": float((sizes == 1).mean()) if len(ids) else float("nan"),
        "customer_share_any_row_lt_k": float(len(cust_small) / n_cust) if n_cust else float("nan"),
        "median_class_size": float(np.median(sizes)) if len(ids) else float("nan"),
    }


# ————————————————— 组合级 k-匿名抑制 —————————————————

def tuple_k_allowed(codes_train: np.ndarray, customers_train: np.ndarray, k: int) -> Set[Tuple[int, ...]]:
    """训练发布中不同客户数 ≥ k 的编码组合。"""
    ids, table = tuple_ids(codes_train)
    sizes = class_sizes(ids, customers_train)
    keep_ids = np.unique(ids[sizes >= int(k)])
    return {tuple(int(v) for v in table[i]) for i in keep_ids}


def tuple_k_suppress(codes: np.ndarray, allowed: Set[Tuple[int, ...]]) -> np.ndarray:
    """组合不在允许集合内的行，所有树都记为 SUPPRESSED。"""
    out = codes.astype(np.int64).copy()
    keep = np.array([tuple(int(v) for v in row) in allowed for row in out], dtype=bool)
    out[~keep] = SUPPRESSED
    return out


# ————————————————— 按 S5.G1 的做法重建发布内容 —————————————————

def release_codes(x_host: Dict[str, np.ndarray], customers_train: np.ndarray, n_trees: int, depth: int, k: int,
                  seed: int, tuple_k: bool) -> Dict[str, np.ndarray]:
    """香港侧：训练期特征拟合随机树嵌入（g3_arms，与 S5.G1 相同），单棵树 k 粗化；tuple_k 时再做组合级抑制。
    x_host 的键为发布人群名，须含 "train"。返回各人群的粗化编码。"""
    emb = g3_arms.tree_embedding(x_host["train"], n_trees, depth, seed)
    raw = {n: g3_arms.leaf_codes(emb, v) for n, v in x_host.items()}
    allowed = g3_arms.k_coarsen(raw["train"], customers_train, k)
    coarse = {n: coarse_codes(c, allowed) for n, c in raw.items()}
    if tuple_k:
        keep = tuple_k_allowed(coarse["train"], customers_train, k)
        coarse = {n: tuple_k_suppress(c, keep) for n, c in coarse.items()}
    return coarse


def one_hot_release(coarse: np.ndarray, coarse_train: np.ndarray) -> np.ndarray:
    """内地收到的独热：每棵树按训练发布中出现过的未抑制叶子排序展开，抑制档全为 0。
    不做组合级抑制时与 g3_arms.one_hot_codes 的列与取值完全相同。"""
    cols = []
    for t in range(coarse_train.shape[1]):
        for leaf in np.unique(coarse_train[:, t]):
            if leaf != SUPPRESSED:
                cols.append((coarse[:, t] == leaf).astype(float))
    return np.column_stack(cols) if cols else np.zeros((len(coarse), 0))


# ————————————————— 字段推断 —————————————————

def _group_stats(ids: np.ndarray, x: np.ndarray, n_groups: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按组合编号向量化汇总：非缺失计数、和、最小值、最大值（各为 组数 × 字段数）。"""
    valid = x != MISSING
    cnt = np.zeros((n_groups, x.shape[1]))
    tot = np.zeros((n_groups, x.shape[1]))
    lo = np.full((n_groups, x.shape[1]), np.inf)
    hi = np.full((n_groups, x.shape[1]), -np.inf)
    np.add.at(cnt, ids, valid.astype(float))
    np.add.at(tot, ids, np.where(valid, x, 0.0))
    np.minimum.at(lo, ids, np.where(valid, x, np.inf))
    np.maximum.at(hi, ids, np.where(valid, x, -np.inf))
    return cnt, tot, lo, hi


def region_means(ids_train: np.ndarray, x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """每个编码组合内各字段的均值（缺失不计，组内全缺失时用全体均值）；以及全体均值作回退。返回（组数 × 字段数，字段数）。"""
    n_groups = int(ids_train.max()) + 1 if len(ids_train) else 0
    cnt, tot, _, _ = _group_stats(ids_train, x_train, n_groups)
    valid = x_train != MISSING
    global_mean = np.where(valid, x_train, 0.0).sum(axis=0) / np.maximum(valid.sum(axis=0), 1)
    means = np.where(cnt > 0, tot / np.maximum(cnt, 1), global_mean)
    return means, global_mean


def lookup_codes(codes_eval: np.ndarray, table_codes: np.ndarray) -> np.ndarray:
    """把评估行的编码组合映射到训练组合表的编号；训练中未出现的组合记 -1。"""
    index = {tuple(int(v) for v in row): i for i, row in enumerate(table_codes)}
    return np.array([index.get(tuple(int(v) for v in row), -1) for row in codes_eval], dtype=np.int64)


def infer_scores(ids_eval: np.ndarray, means: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """训练中未出现的组合（编号 -1）用全体均值。"""
    out = np.tile(fallback, (len(ids_eval), 1))
    seen = ids_eval >= 0
    out[seen] = means[ids_eval[seen]]
    return out


def attribute_auc(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """逐字段的推断 AUROC：真值为 0/1 的行，两类都出现时才计算，否则为 nan。"""
    out = np.full(truth.shape[1], np.nan)
    for j in range(truth.shape[1]):
        m = truth[:, j] != MISSING
        y = truth[m, j]
        if len(np.unique(y)) == 2:
            out[j] = roc_auc_score(y, scores[m, j])
    return out


def certainty(ids_train: np.ndarray, x_train: np.ndarray, ids_eval: np.ndarray,
              x_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐字段：评估行所在组合在训练发布中取值唯一（纯）的比例、这些行按该唯一值推断的正确率，
    以及取值为 1 的行（如持有某产品）中被确定推出为 1 的比例——后者扣除了「多数人不持有」带来的平凡确定性。"""
    n_groups = int(ids_train.max()) + 1 if len(ids_train) else 0
    cnt, _, lo, hi = _group_stats(ids_train, x_train, n_groups)
    pure = (cnt > 0) & (lo == hi)
    seen = ids_eval >= 0
    guess = np.full(x_eval.shape, np.nan)
    guess[seen] = np.where(pure[ids_eval[seen]], lo[ids_eval[seen]], np.nan)
    determined = ~np.isnan(guess) & (x_eval != MISSING)
    share = determined.mean(axis=0) if len(ids_eval) else np.full(x_eval.shape[1], np.nan)
    n_det = determined.sum(axis=0)
    correct = np.where(determined, guess == x_eval, False).sum(axis=0) / np.maximum(n_det, 1)
    positive = x_eval == 1.0
    n_pos = positive.sum(axis=0)
    pos_found = (determined & positive & (guess == 1.0)).sum(axis=0) / np.maximum(n_pos, 1)
    return share, np.where(n_det > 0, correct, np.nan), np.where(n_pos > 0, pos_found, np.nan)


def disclosure_summary(codes_train: np.ndarray, x_train: np.ndarray, codes_eval: np.ndarray,
                       x_eval: np.ndarray) -> Dict[str, np.ndarray]:
    """用训练发布学映射、在评估发布上度量：逐字段 AUROC、确定推出比例与其正确率。"""
    ids_tr, table_codes = tuple_ids(codes_train)
    means, gmean = region_means(ids_tr, x_train)
    ids_ev = lookup_codes(codes_eval, table_codes)
    scores = infer_scores(ids_ev, means, gmean)
    share, correct, pos_found = certainty(ids_tr, x_train, ids_ev, x_eval)
    return {"auc": attribute_auc(scores, x_eval), "certain_share": share, "certain_accuracy": correct,
            "positive_determined": pos_found,
            "unseen_tuple_share": np.array([float((ids_ev < 0).mean()) if len(ids_ev) else np.nan])}


def segment_disclosure(seg_eval_means: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """L1 对照：内地拿到的分段均值直接作推断分数时的逐字段 AUROC。"""
    return attribute_auc(seg_eval_means, x_eval)


def mean_over(values: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """seed 维度的均值与标准差（忽略 nan）。"""
    arr = np.vstack(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # 全为 nan 的字段（单类）返回 nan，属预期
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
