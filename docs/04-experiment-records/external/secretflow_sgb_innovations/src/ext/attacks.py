"""[攻击] 半诚实参与方利用 SGB 协议中「合法收到的消息」做推断（secretflow 1.13 sgb_v）。

威胁模型：两方均半诚实、不合谋，严格按协议执行，只对自己收到的消息做离线分析。
消息内容按源码重放（见 sgb_replay.py 顶部），路由已与隐语原生 predict_leaf_select 逐样本核对一致。

A1 香港（协作方）→ 内地标签：
    香港在训练中收到每层节点的样本集合 → 知道每个训练样本在每棵树落在哪个叶子
    （首树开启 first_tree_with_label_holder_feature 时，首树不发给香港）。
    假设香港另有 --known-frac（默认 1%）训练样本的真实标签（如两地共同客户中已知的购买记录），
    用「自有特征 + 叶子 one-hot」训练逻辑回归推断其余样本标签；与「只用自有特征」的同法基线对比。
    泄漏量 = AUC(自有+叶子) − AUC(只用自有特征)，5 次随机已知集合取均值。
A2 内地（标签方）→ 香港特征：
    训练时：香港特征上每次分裂的左子样本集合发回内地 → 对该节点的样本，内地得知 x_f < v 是否成立。
    推理时：香港把自己分裂树下各叶子的可达样本发给内地 → 对每个被打分样本，凡是在「香港视角路径」上
           到达的香港分裂节点（内地节点两边都通，只被香港自己的上游分裂挡住），其结果都可推出
           （sgb_replay.host_reach，已与隐语原生 predict_leaf_select 输出逐格核对）。
    还原口径：对每个样本 × 香港特征维护取值区间，区间内只剩一个可能取值即「被确定还原」。
    「已知列名」为上界；「不知列名」用划分一致性聚类（重叠 ≥30 样本且 100% 一致则同列）给出匿名列数与纯度。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import sgb_replay as R

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "prepared"
OUT = REPO_ROOT / "data" / "processed" / "attack"


def load_tag(tag: str):
    g = pd.read_csv(DATA_DIR / f"{tag}_guest.csv")
    h = pd.read_csv(DATA_DIR / f"{tag}_host.csv")
    idx = pd.read_csv(DATA_DIR / f"{tag}_index.csv")
    return (g.drop(columns="ncodpers").to_numpy(float), h.drop(columns="ncodpers").to_numpy(float),
            idx, [c for c in h.columns if c != "ncodpers"])


# ----------------------------------------------------------------------------- A1
def attack_a1(m, X, y, host_sees_first_tree: bool, known_frac: float, reps: int = 5) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    T = R.n_trees(m)
    trees = list(range(T)) if host_sees_first_tree else list(range(1, T))
    cols = []
    for t in trees:
        leaf, _ = R.route(m, t, X)
        u, inv = np.unique(leaf, return_inverse=True)
        cols.append(sparse.csr_matrix((np.ones(len(leaf)), (np.arange(len(leaf)), inv)), shape=(len(leaf), len(u))))
    L = sparse.hstack(cols).tocsr()
    own = X["hk"]
    res = {"trees_visible": len(trees), "leaf_columns": int(L.shape[1]), "known_frac": known_frac}
    aucs = {"own_only": [], "own_plus_leaves": [], "leaves_only": []}
    n = len(y)
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        known = np.zeros(n, dtype=bool)
        known[rng.choice(n, size=int(n * known_frac), replace=False)] = True
        if y[known].sum() < 2:
            continue
        for name, F in (("own_only", sparse.csr_matrix(own)),
                        ("own_plus_leaves", sparse.hstack([sparse.csr_matrix(own), L]).tocsr()),
                        ("leaves_only", L)):
            clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000).fit(F[known], y[known])
            aucs[name].append(float(roc_auc_score(y[~known], clf.predict_proba(F[~known])[:, 1])))
    for k, v in aucs.items():
        res[k] = {"mean": float(np.mean(v)), "sd": float(np.std(v)), "runs": v}
    res["leakage_auc"] = res["own_plus_leaves"]["mean"] - res["own_only"]["mean"]
    return res


# ----------------------------------------------------------------------------- A2
def _determined(lo, hi, levels):
    """lo<=x<hi 区间内恰好剩一个可能取值 → 被确定还原。levels: 该特征的全部可能取值（排序）。"""
    cnt = np.zeros(lo.shape, dtype=int)
    for v in levels:
        cnt += (v >= lo) & (v < hi)
    return cnt == 1


def attack_a2(m, X, host: str, hcols: list[str], phase: str) -> dict:
    """phase = train：按完整路由到达该节点的训练样本；phase = infer：按香港视角可达的被打分样本。"""
    H = X[host]
    n, d = H.shape
    lo = np.full((n, d), -np.inf)
    hi = np.full((n, d), np.inf)
    T = R.n_trees(m)
    node_parts = []   # 不知列名聚类用：(feature, node_mask, left_mask)
    for t in range(T):
        hs = R.host_split_nodes(m, t, host)
        if not hs:
            continue
        reach = R.route(m, t, X)[1] if phase == "train" else R.host_reach(m, t, H, host)
        for node, f, v in hs:
            mask = reach.get(node, np.zeros(n, bool))
            if not mask.any():
                continue
            left = H[:, f] < v
            sel_l, sel_r = mask & left, mask & ~left
            hi[sel_l, f] = np.minimum(hi[sel_l, f], v)
            lo[sel_r, f] = np.maximum(lo[sel_r, f], v)
            node_parts.append((f, mask, left & mask))
    per_feat = {}
    for f, name in enumerate(hcols):
        levels = np.unique(H[:, f])
        touched = np.isfinite(lo[:, f]) | np.isfinite(hi[:, f])   # 至少被一次分裂约束过（常量列不算泄漏）
        det = _determined(lo[:, f], hi[:, f], levels) & touched
        per_feat[name] = {"levels": int(len(levels)), "used_in_model": bool(touched.any()),
                          "revealed_frac": float(det.mean())}
    used = [k for k, v in per_feat.items() if v["used_in_model"]]
    res = {"phase": phase, "n_samples": int(n), "host_split_nodes": len(node_parts),
           "features_used": len(used), "features_total": d,
           "mean_revealed_frac_all": float(np.mean([v["revealed_frac"] for v in per_feat.values()])),
           "cells_revealed_frac": float(np.mean([v["revealed_frac"] for v in per_feat.values()])),
           "used_features_revealed_frac": float(np.mean([per_feat[k]["revealed_frac"] for k in used])) if used else 0.0,
           "per_feature": per_feat}
    res["identity_free"] = cluster_partitions(node_parts) if node_parts else {}
    res["_bounds"] = (lo, hi)
    return res


def truth_eval(lo, hi, H, hcols, Ht, tcols) -> dict:
    """防护后（香港侧特征已变换）的还原结果对照「真实未变换的香港特征」。
    - 同列（随机响应 hostrr）：被确定还原的是加噪值；统计其中与真值一致的比例。
    - n_products（数据最小化 hostcnt）：只有「产品数被确定为 0」时才能推出每个产品标记 = 0（且必然正确）。
    """
    out = {}
    if list(hcols) == list(tcols):
        for f, name in enumerate(hcols):
            levels = np.unique(H[:, f])
            touched = np.isfinite(lo[:, f]) | np.isfinite(hi[:, f])
            det = _determined(lo[:, f], hi[:, f], levels) & touched
            rec = np.full(len(H), np.nan)
            for v in levels:
                rec[det & (v >= lo[:, f]) & (v < hi[:, f])] = v
            ok = det & (rec == Ht[:, f])
            pos = Ht[:, f] == 1
            claim1 = det & (rec == 1)
            out[name] = {"revealed_noisy_frac": float(det.mean()), "revealed_true_frac": float(ok.mean()),
                         "accuracy_among_revealed": float(ok.sum() / det.sum()) if det.any() else None,
                         "holders": int(pos.sum()),
                         "holder_recall": float((claim1 & pos).sum() / pos.sum()) if pos.any() else None,
                         "holder_precision": float((claim1 & pos).sum() / claim1.sum()) if claim1.any() else None}
        mode = "same_columns"
    elif "n_products" in hcols:
        f = list(hcols).index("n_products")
        levels = np.unique(H[:, f])
        touched = np.isfinite(lo[:, f]) | np.isfinite(hi[:, f])
        zero = touched & (hi[:, f] <= levels[levels > 0].min()) & (lo[:, f] <= 0)
        for name in tcols:
            if name in hcols:
                continue
            t = Ht[:, list(tcols).index(name)]
            out[name] = {"revealed_true_frac": float(zero.mean()), "accuracy_among_revealed": float((t[zero] == 0).mean()) if zero.any() else None,
                         "holders": int((t == 1).sum()), "holder_recall": 0.0 if (t == 1).any() else None, "holder_precision": None}
        mode = "count_zero_implies_all_zero"
    import re
    prod = {k: v for k, v in out.items() if re.match(r"^ind_.*_ult1$", k)}
    vals = [v["revealed_true_frac"] for v in prod.values()]
    hold = sum(v["holders"] for v in prod.values())
    rec1 = sum((v["holder_recall"] or 0) * v["holders"] for v in prod.values())
    prec_num = prec_den = 0.0
    for v in prod.values():
        if v["holder_precision"] is not None and v["holder_recall"]:
            tp = v["holder_recall"] * v["holders"]
            prec_num += tp
            prec_den += tp / v["holder_precision"]
    return {"mode": mode, "mean_true_revealed_frac": float(np.mean(vals)) if vals else 0.0,
            "holder_cells": int(hold), "holder_recall_pooled": float(rec1 / hold) if hold else 0.0,
            "holder_precision_pooled": float(prec_num / prec_den) if prec_den else None,
            "scope": "产品持有标记列（ind_*_ult1）", "per_feature": out}


def cluster_partitions(parts, min_overlap: int = 30, min_side: int = 5) -> dict:
    """不知列名：两个香港分裂节点在共同样本上划分 100% 一致（且重叠 ≥ min_overlap）→ 视为同一匿名列。"""
    K = len(parts)
    # 稠密 float32 BLAS（K≈千级 × n≈8 万）比稀疏乘快得多；计数 < 2^24，float32 精确
    A = np.vstack([p[1] for p in parts]).astype(np.float32)
    Lm = np.vstack([p[2] for p in parts]).astype(np.float32)
    ov = A @ A.T
    ll = Lm @ Lm.T
    Lm = A - Lm
    rr = Lm @ Lm.T
    agree = ll + rr
    informative = np.minimum(ll, rr) >= min_side   # 重叠里两侧都有样本，才算「划分一致」的证据
    del A, Lm
    parent = list(range(K))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(K):
        for j in np.where((ov[i] >= min_overlap) & (agree[i] == ov[i]) & informative[i])[0]:
            if j > i:
                parent[find(i)] = find(int(j))
    groups: dict[int, list[int]] = {}
    for i in range(K):
        groups.setdefault(find(i), []).append(i)
    feats = np.array([p[0] for p in parts])
    purity = [np.bincount(feats[g]).max() / len(g) for g in groups.values()]
    big = [g for g in groups.values() if len(g) >= 2]
    return {"split_nodes": K, "rule": f"overlap>={min_overlap}, 100% agree, both sides>={min_side}",
            "anonymous_columns": len(groups), "columns_with_2plus_nodes": len(big),
            "true_features_behind_nodes": int(len(np.unique(feats))),
            "node_weighted_purity": float(sum(p * len(g) for p, g in zip(purity, groups.values())) / K)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="train_federated_sgb_opt.py --dump-model 的产物")
    ap.add_argument("--tag", default="business")
    ap.add_argument("--host-sees-first-tree", type=int, default=0, choices=[0, 1],
                    help="首树是否发给香港：隐语默认 first_tree_with_label_holder_feature=True → 0")
    ap.add_argument("--known-frac", type=float, nargs="+", default=[0.01])
    ap.add_argument("--name", required=True)
    ap.add_argument("--truth-tag", default=None, help="防护实验：用该 tag 的香港原始特征作真值对照（如 business）")
    args = ap.parse_args(argv)

    m = R.load(args.model)
    G, H, idx, hcols = load_tag(args.tag)
    rows = np.array(m["train_rows"])
    te = np.where(idx["is_test_time"].to_numpy() == 1)[0]
    Xtr = {"neidi": G[rows], "hk": H[rows]}
    Xte = {"neidi": G[te], "hk": H[te]}
    y = idx["label_new_any"].to_numpy()[rows]
    # 重放预测与隐语预测对账（如有）
    out = {"name": args.name, "model": args.model, "tag": args.tag, "trees": R.n_trees(m)}
    a1_all = {}
    for kf in args.known_frac:
        a1 = attack_a1(m, Xtr, y, bool(args.host_sees_first_tree), kf)
        a1_all[f"{kf:g}"] = a1
        print(f"[{args.name}] A1 香港推断标签（已知 {kf:.0%} 标签，可见树 {a1['trees_visible']}）："
              f"只用自有特征 AUC {a1['own_only']['mean']:.4f} → +叶子 {a1['own_plus_leaves']['mean']:.4f}"
              f"（仅叶子 {a1['leaves_only']['mean']:.4f}）| 泄漏 +{a1['leakage_auc']:.4f}")
    a1 = a1_all[f"{args.known_frac[0]:g}"]
    a2tr = attack_a2(m, Xtr, "hk", hcols, "train")
    a2te = attack_a2(m, Xte, "hk", hcols, "infer")
    for a in (a2tr, a2te):
        top = sorted(a["per_feature"].items(), key=lambda kv: -kv[1]["revealed_frac"])[:5]
        print(f"[{args.name}] A2 内地还原香港特征（{a['phase']}，{a['n_samples']:,} 样本）：模型用到 "
              f"{a['features_used']}/{a['features_total']} 列；全部单元格被确定还原 {a['cells_revealed_frac']:.1%}"
              f"（被用到的列 {a['used_features_revealed_frac']:.1%}）；"
              f"不知列名聚出 {a['identity_free'].get('anonymous_columns')} 个匿名列（背后真实 "
              f"{a['identity_free'].get('true_features_behind_nodes')} 列，纯度 {a['identity_free'].get('node_weighted_purity', 0):.3f}）")
        print("          最多: " + "，".join(f"{k} {v['revealed_frac']:.1%}" for k, v in top))
    if args.truth_tag:
        _, Ht, _, tcols = load_tag(args.truth_tag)
        for a, rr in ((a2tr, rows), (a2te, te)):
            lo, hi = a["_bounds"]
            a["truth"] = truth_eval(lo, hi, (Xtr if a is a2tr else Xte)["hk"], hcols, Ht[rr], tcols)
            print(f"[{args.name}] A2 对照真值（{a['phase']}，{a['truth']['mode']}）：真实香港特征被确定还原 "
                  f"{a['truth']['mean_true_revealed_frac']:.1%}；「持有=1」的单元格被确定识别 {a['truth']['holder_recall_pooled']:.1%}"
                  f"（识别为持有者中真持有 {a['truth']['holder_precision_pooled'] if a['truth']['holder_precision_pooled'] is None else format(a['truth']['holder_precision_pooled'], '.1%')}）")
    for a in (a2tr, a2te):
        a.pop("_bounds", None)
    out.update({"A1": a1, "A1_by_known_frac": a1_all, "A2_train": a2tr, "A2_infer": a2te})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"attack_{args.name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
