"""[攻击取证] 从导出的 SGB 模型结构重放「训练 / 推理时各方实际收到的样本划分」。

依据（secretflow 1.13.0b0 源码，sgb_v）：
- 标签方 → 协作方：每层把被选子节点的样本集合发给协作方（bucket_sum_calculator.py：
  `children_split_node_selects_bits.to(worker)`），协作方用父节点集合相减即得另一子节点
  → 协作方知道每个训练样本在每棵树（首树若开 first_tree_with_label_holder_feature 则除外）落在哪个叶子。
- 协作方 → 标签方：协作方特征上的每次分裂，左子样本集合发回标签方（split_tree_builder.py：
  `lchild_selects.append(selects.to(self.label_holder))`）→ 标签方知道协作方特征分裂的划分结果。
- 推理：每方把自己分裂树下各叶子的「可达样本」发给标签方（distributed_tree.py predict：
  `weight_selects.append(s.to(self.label_holder))`）。

本模块只做确定性路由；正确性由 `validate()` 与隐语原生 `SplitTree.predict_leaf_select` 逐样本比对保证。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def n_trees(m: dict) -> int:
    return int(m["model"]["common"]["tree_num"])


def owner_of(m: dict, t: int, node: int) -> tuple[str | None, int, float]:
    """节点 node 在第 t 棵树中由哪一方分裂：(方名, 特征下标, 阈值)；都没有则 (None, -1, inf)。"""
    for party, trees in m["model"]["split_trees"].items():
        st = trees[t]
        pos = st["split_indices"].index(node) if node in st["split_indices"] else -1
        if pos >= 0 and st["split_features"][pos] != -1:
            return party, int(st["split_features"][pos]), float(st["split_values"][pos])
    return None, -1, float("inf")


def route(m: dict, t: int, X: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """返回 (每个样本的叶子编号, {节点编号: 到达该节点的样本布尔向量})。左子 = x < 阈值（与隐语一致，见 validate）。"""
    n = next(iter(X.values())).shape[0]
    reach = {0: np.ones(n, dtype=bool)}
    leaf = np.full(n, -1, dtype=int)
    leaves = set(m["model"]["split_trees"][m["model"]["label_holder"]][t]["leaf_indices"])
    frontier = [0]
    while frontier:
        nxt = []
        for node in frontier:
            s = reach[node]
            if node in leaves:
                leaf[s] = node
                continue
            party, f, v = owner_of(m, t, node)
            go_left = np.ones(n, dtype=bool) if party is None else (X[party][:, f] < v)
            reach[2 * node + 1] = s & go_left
            reach[2 * node + 2] = s & ~go_left
            nxt += [2 * node + 1, 2 * node + 2]
        frontier = nxt
    return leaf, reach


def predict_raw(m: dict, X: dict[str, np.ndarray]) -> np.ndarray:
    """重放预测（base + Σ 叶子权重），用于与隐语预测逐位对账。"""
    lh = m["model"]["label_holder"]
    out = np.full(next(iter(X.values())).shape[0], float(m["model"]["common"]["base"]))
    for t in range(n_trees(m)):
        leaf, _ = route(m, t, X)
        idx = m["model"]["split_trees"][lh][t]["leaf_indices"]
        w = np.array([x[0] for x in m["model"]["leaf_weights"][t]])
        pos = {li: k for k, li in enumerate(idx)}
        out += w[[pos[l] for l in leaf]]
    return out


def host_split_nodes(m: dict, t: int, host: str) -> list[tuple[int, int, float]]:
    st = m["model"]["split_trees"][host][t]
    return [(int(n), int(f), float(v)) for n, f, v in zip(st["split_indices"], st["split_features"], st["split_values"])
            if f != -1]


def host_reach(m: dict, t: int, H: np.ndarray, host: str) -> dict[int, np.ndarray]:
    """推理时协作方视角的可达集合：协作方自己的分裂按 x<v 走，其余方的节点「两边都通」。
    标签方从协作方发来的叶子可达矩阵中，恰好能推出这些样本在协作方节点上的走向（已与隐语输出逐格核对）。"""
    n = H.shape[0]
    st = m["model"]["split_trees"][host][t]
    owned = {int(a): (int(f), float(v)) for a, f, v in zip(st["split_indices"], st["split_features"], st["split_values"]) if f != -1}
    leaves = set(st["leaf_indices"])
    reach = {0: np.ones(n, dtype=bool)}
    frontier = [0]
    while frontier:
        nxt = []
        for node in frontier:
            if node in leaves:
                continue
            s = reach[node]
            if node in owned:
                f, v = owned[node]
                left = H[:, f] < v
                reach[2 * node + 1], reach[2 * node + 2] = s & left, s & ~left
            else:
                reach[2 * node + 1], reach[2 * node + 2] = s.copy(), s.copy()
            nxt += [2 * node + 1, 2 * node + 2]
        frontier = nxt
    return reach
