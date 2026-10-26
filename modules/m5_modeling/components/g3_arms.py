"""M5 · G3 五臂对比的各臂（S6.P3 预注册）：本地梯度提升树、L1 k-匿名分段统计、B_T1 随机树嵌入、B_T2 香港自有分数、阴性对照。

所有内地侧臂用同一模型族（XGBoost hist）与同一搜索空间；联邦臂 A 由 ``g3_sgb`` 用隐语 SGB 训练，
其超参在明文集中式代理上按同一搜索空间选定（SecureBoost 类协议在分桶与随机性相同时无损）。
各函数只做计算，参数一律由调用方从配置传入。
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.metrics import roc_auc_score

SEGMENT_SUPPRESSED = -1


# ————————————————— 本地梯度提升树与调参 —————————————————

def xgb_fit(x: np.ndarray, y: np.ndarray, config: Dict, max_bin: int, seed: int, n_jobs: int):
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=int(config["num_boost_round"]), max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]), reg_lambda=float(config["reg_lambda"]),
        tree_method="hist", max_bin=int(max_bin), subsample=1.0, colsample_bytree=1.0,
        objective="binary:logistic", eval_metric="auc", random_state=int(seed), n_jobs=int(n_jobs))
    model.fit(x, y)
    return model


def xgb_score(model, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def search_grid(space: Dict[str, Sequence]) -> List[Dict]:
    keys = sorted(space)
    return [dict(zip(keys, values)) for values in itertools.product(*(space[k] for k in keys))]


def sample_configs(space: Dict[str, Sequence], n: int, seed: int) -> List[Dict]:
    """从网格中不放回抽 n 组候选（同一 seed 对所有臂相同，保证各臂候选集一致）。"""
    grid = search_grid(space)
    if n > len(grid):
        raise ValueError("候选数 %d 超过网格大小 %d" % (n, len(grid)))
    pick = np.random.default_rng(seed).choice(len(grid), size=n, replace=False)
    return [grid[i] for i in sorted(pick)]


def tune(x_tr: np.ndarray, y_tr: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, configs: List[Dict],
         max_bin: int, seed: int, n_jobs: int) -> Tuple[Dict, List[Dict]]:
    """在内层验证集上按 AUROC 选配置。返回（最佳配置，每组候选的验证 AUROC 表）。"""
    rows = []
    for i, cfg in enumerate(configs):
        model = xgb_fit(x_tr, y_tr, cfg, max_bin, seed, n_jobs)
        rows.append(dict(cfg, candidate=i, val_auc=float(roc_auc_score(y_val, xgb_score(model, x_val)))))
    best = max(rows, key=lambda r: (r["val_auc"], -r["candidate"]))
    return {k: best[k] for k in configs[0]}, rows


# ————————————————— L1：内地分段 + 香港 k-匿名分段均值 —————————————————

def segment_edges(x_guest: np.ndarray, columns: Dict[str, int], quantiles: Dict[str, int]) -> Dict[str, np.ndarray]:
    """分段边界只从训练期内地数据学（学到的预处理状态，不入仓库）。"""
    edges = {}
    for name, n_q in quantiles.items():
        v = x_guest[:, columns[name]]
        v = v[v >= 0]
        edges[name] = np.unique(np.quantile(v, np.linspace(0, 1, n_q + 1)[1:-1]))
    return edges


def segment_ids(x_guest: np.ndarray, columns: Dict[str, int], edges: Dict[str, np.ndarray],
                categorical_sizes: Dict[str, int]) -> np.ndarray:
    """分段号 = 分位数分箱（缺失单独一档）与类别列的组合，只用内地特征。

    每一部分的进制固定（分箱数 + 1 档缺失；类别数 + 1 档缺失/未见），训练与评估的同一分段得到同一编号。
    """
    parts = []
    for name, e in edges.items():
        v = x_guest[:, columns[name]]
        parts.append((np.where(v >= 0, np.searchsorted(e, v, side="right") + 1, 0), len(e) + 2))
    for name, size in categorical_sizes.items():
        v = x_guest[:, columns[name]].astype(int)
        parts.append((np.where((v >= 0) & (v < size), v + 1, 0), size + 1))
    seg = np.zeros(len(x_guest), dtype=np.int64)
    for p, radix in parts:
        seg = seg * radix + p
    return seg


def k_anonymous_means(seg_train: np.ndarray, customers_train: np.ndarray, x_host_train: np.ndarray,
                      k: int) -> Tuple[Dict[int, np.ndarray], np.ndarray, Dict[str, int]]:
    """香港侧按分段返回特征均值；分段内不同客户数 < k 的分段被抑制，回落到全体均值。"""
    global_mean = x_host_train.mean(axis=0)
    table: Dict[int, np.ndarray] = {}
    suppressed = 0
    for s in np.unique(seg_train):
        mask = seg_train == s
        if len(np.unique(customers_train[mask])) >= k:
            table[int(s)] = x_host_train[mask].mean(axis=0)
        else:
            suppressed += 1
    info = {"segments_total": int(len(np.unique(seg_train))), "segments_released": len(table),
            "segments_suppressed": suppressed}
    return table, global_mean, info


def apply_segment_means(seg: np.ndarray, table: Dict[int, np.ndarray], global_mean: np.ndarray) -> np.ndarray:
    out = np.tile(global_mean, (len(seg), 1))
    for s, mean in table.items():
        out[seg == s] = mean
    return out


# ————————————————— B_T1：香港随机树嵌入，k 粗化后一次性发给内地 —————————————————

def tree_embedding(x_host_train: np.ndarray, n_trees: int, depth: int, seed: int) -> RandomTreesEmbedding:
    """无监督，不用任何标签。"""
    return RandomTreesEmbedding(n_estimators=int(n_trees), max_depth=int(depth), random_state=int(seed)).fit(x_host_train)


def leaf_codes(embedding: RandomTreesEmbedding, x_host: np.ndarray) -> np.ndarray:
    return np.column_stack([est.apply(x_host) for est in embedding.estimators_])


def k_coarsen(codes_train: np.ndarray, customers_train: np.ndarray, k: int) -> List[set]:
    """每棵树中不同客户数 ≥ k 的叶子才保留，其余叶子并入同一个「被抑制」档。返回每棵树允许发布的叶子集合。"""
    allowed = []
    for t in range(codes_train.shape[1]):
        keep = set()
        for leaf in np.unique(codes_train[:, t]):
            if len(np.unique(customers_train[codes_train[:, t] == leaf])) >= k:
                keep.add(int(leaf))
        allowed.append(keep)
    return allowed


def one_hot_codes(codes: np.ndarray, allowed: List[set]) -> np.ndarray:
    cols = []
    for t, keep in enumerate(allowed):
        for leaf in sorted(keep):
            cols.append((codes[:, t] == leaf).astype(float))
    return np.column_stack(cols) if cols else np.zeros((len(codes), 0))


# ————————————————— 阴性对照：对齐客户内打乱被动方特征行 —————————————————

def shuffle_rows(x_host: np.ndarray, seed: int) -> np.ndarray:
    """保留每列边际分布，切断与客户的对应关系。"""
    return x_host[np.random.default_rng(seed).permutation(len(x_host))]
