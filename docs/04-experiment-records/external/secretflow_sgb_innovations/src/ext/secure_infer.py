"""[安全·协议层] SGB 模型 + SPU 安全推理：消除推理期 A2（香港叶子可达集合明文发给内地）。

隐语 SGB 原生推理（distributed_tree.py predict）：每方把「自己分裂下各叶子的可达样本」明文发给标签方，
标签方逐棵树求交得到叶子 → 标签方能还原香港特征（见 attacks.py A2_infer）。
本原型：训练不变（沿用已训练的 SGB 模型）；推理时两方各自在本地算叶子可达矩阵（n × 叶子数），
送进 SPU（秘密分享），在密态下逐元素相乘（求交）→ 乘叶子权重 → 累加；只把最终原始分数 reveal 给内地。
内地只得到每个客户的分数——这是它本来就要的输出，不再看到任何香港侧的划分。

    python src/ext/secure_infer.py --model data/processed/attack/model_S0_base.json \
        --ref data/processed/predictions/business_sgb_S0_base_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "prepared"


def leaf_select(tree: dict, X: np.ndarray, leaves: list[int]) -> np.ndarray:
    """单方视角的叶子可达矩阵：本方分裂按 x<v 走，其余方的节点两边都通（与 sgb_replay.host_reach 同逻辑）。"""
    n = X.shape[0]
    owned = {int(a): (int(f), float(v)) for a, f, v in zip(tree["split_indices"], tree["split_features"],
                                                          tree["split_values"]) if f != -1}
    leafset = set(leaves)
    reach = {0: np.ones(n, dtype=bool)}
    frontier = [0]
    while frontier:
        nxt = []
        for node in frontier:
            if node in leafset:
                continue
            s = reach[node]
            if node in owned:
                f, v = owned[node]
                left = X[:, f] < v
                reach[2 * node + 1], reach[2 * node + 2] = s & left, s & ~left
            else:
                reach[2 * node + 1], reach[2 * node + 2] = s, s
            nxt += [2 * node + 1, 2 * node + 2]
        frontier = nxt
    return np.stack([reach.get(l, np.zeros(n, bool)) for l in leaves], axis=1).astype(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="business")
    ap.add_argument("--ref", default=None, help="隐语原生推理的 predictions.csv，用于逐样本对账")
    ap.add_argument("--batch-trees", type=int, default=10)
    ap.add_argument("--out", default="data/processed/attack/secure_infer.json")
    args = ap.parse_args(argv)

    import jax.numpy as jnp
    import secretflow as sf
    from sklearn.metrics import roc_auc_score

    m = json.loads((REPO_ROOT / args.model).read_text(encoding="utf-8"))["model"]
    lh = m["label_holder"]
    host = [p for p in m["split_trees"] if p != lh][0]
    T = int(m["common"]["tree_num"])
    g = pd.read_csv(DATA_DIR / f"{args.tag}_guest.csv")
    h = pd.read_csv(DATA_DIR / f"{args.tag}_host.csv")
    idx = pd.read_csv(DATA_DIR / f"{args.tag}_index.csv")
    te = idx["is_test_time"].to_numpy() == 1
    G = g.drop(columns="ncodpers").to_numpy(float)[te]
    H = h.drop(columns="ncodpers").to_numpy(float)[te]
    y = idx["label_new_any"].to_numpy()[te]

    sf.init([lh, host], address="local", num_cpus=8, log_to_driver=False)
    p_lh, p_h = sf.PYU(lh), sf.PYU(host)
    cl = sf.utils.testing.cluster_def([lh, host])
    cl["runtime_config"] = {"protocol": "SEMI2K", "field": "FM64"}
    spu = sf.SPU(cl)

    # 各方只拿到自己那份分裂树；叶子权重只在内地
    x_lh = p_lh(lambda a: a)(G)
    x_h = p_h(lambda a: a)(H)
    trees_lh = p_lh(lambda t: t)(m["split_trees"][lh])
    trees_h = p_h(lambda t: t)(m["split_trees"][host])
    leaves = [m["split_trees"][lh][t]["leaf_indices"] for t in range(T)]   # 树结构（叶子编号）双方共知
    W = p_lh(lambda ws: [np.array([w[0] for w in wt], np.float32) for wt in ws])(m["leaf_weights"])

    t0 = time.perf_counter()
    acc = None
    for start in range(0, T, args.batch_trees):
        ts = list(range(start, min(T, start + args.batch_trees)))
        lv = [leaves[t] for t in ts]
        s_lh = p_lh(lambda tr, X, ts=ts, lv=lv: np.concatenate([leaf_select(tr[t], X, l) for t, l in zip(ts, lv)], 1))(trees_lh, x_lh)
        s_h = p_h(lambda tr, X, ts=ts, lv=lv: np.concatenate([leaf_select(tr[t], X, l) for t, l in zip(ts, lv)], 1))(trees_h, x_h)
        w = p_lh(lambda ws, ts=ts: np.concatenate([ws[t] for t in ts]))(W)
        part = spu(lambda a, b, ww: (a * b) @ ww)(s_lh.to(spu), s_h.to(spu), w.to(spu))
        acc = part if acc is None else spu(lambda u, v: u + v)(acc, part)
    raw = sf.reveal(acc.to(p_lh))          # 只 reveal 给内地
    secs = time.perf_counter() - t0
    score = 1.0 / (1.0 + np.exp(-(np.asarray(raw, float).ravel() + float(m["common"]["base"]))))
    auc = float(roc_auc_score(y, score))
    res = {"model": args.model, "trees": T, "test_rows": int(te.sum()), "seconds": secs, "auc": auc}
    if args.ref:
        ref = pd.read_csv(REPO_ROOT / args.ref)["score"].to_numpy(float)
        res.update({"max_abs_diff_vs_sgb": float(np.abs(ref - score).max()),
                    "auc_sgb_native": float(roc_auc_score(y, ref))})
    print(f"[安全推理] {T} 棵树 × {int(te.sum()):,} 个客户：SPU 推理 {secs:.1f}s｜AUC {auc:.4f}"
          + (f"｜隐语原生推理 AUC {res['auc_sgb_native']:.4f}，逐样本最大偏差 {res['max_abs_diff_vs_sgb']:.2e}" if args.ref else ""),
          flush=True)
    (REPO_ROOT / args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    sf.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
