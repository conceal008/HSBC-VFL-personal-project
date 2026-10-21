"""[安全·协议层] 隐语 SS-XGB（秘密分享 XGB，Fang 等 CIKM 2021）在 Santander business 切分上的成本测量。

与 SGB 的区别：梯度、节点样本分布、分裂增益都在秘密分享下计算，两方都看不到对方的样本划分
→ 按设计不存在 A1 / A2 依赖的「样本集合」消息。代价是 MPC 计算与通信。
必须用 ray 仿真模式 + 真正的 SPU（semi2k）测；debug 模式会用明文 PYU 模拟 SPU，耗时不可信。

做法：在训练集前 N 行上训练 T 棵树（超参与 SGB 默认一致），记录每棵树耗时；多个 N 拟合「每棵树耗时 ≈ a + b·N」，
外推到全量 78,267 行 × 100 棵树。测试集 AUC 只作可用性检查（N 小时偏低）。

    python src/ext/ss_xgb_cost.py --rows 2000 5000 10000 --trees 2
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="business")
    ap.add_argument("--rows", type=int, nargs="+", default=[2000, 5000, 10000])
    ap.add_argument("--trees", type=int, default=2)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--protocol", default="SEMI2K", choices=["SEMI2K", "CHEETAH", "REF2K"])
    ap.add_argument("--out", default="data/processed/attack/ss_xgb_cost.json")
    args = ap.parse_args(argv)

    import secretflow as sf
    import spu
    from secretflow.data import FedNdarray, PartitionWay
    from secretflow.ml.boost.ss_xgb_v import Xgb
    from sklearn.metrics import roc_auc_score

    g = pd.read_csv(DATA_DIR / f"{args.tag}_guest.csv")
    h = pd.read_csv(DATA_DIR / f"{args.tag}_host.csv")
    idx = pd.read_csv(DATA_DIR / f"{args.tag}_index.csv")
    tr = np.where(idx["is_test_time"].to_numpy() == 0)[0]
    te = np.where(idx["is_test_time"].to_numpy() == 1)[0]
    G = g.drop(columns="ncodpers").to_numpy(np.float32)
    H = h.drop(columns="ncodpers").to_numpy(np.float32)
    y = idx["label_new_any"].to_numpy(np.float32)

    sf.init(["neidi", "hk"], address="local", num_cpus=8, log_to_driver=False)   # 逻辑 CPU；=2 时两个 SPU actor 占满资源，PYU 任务无法调度（死锁）
    neidi, hk = sf.PYU("neidi"), sf.PYU("hk")
    cluster = sf.utils.testing.cluster_def(["neidi", "hk"])
    cluster["runtime_config"] = {"protocol": args.protocol, "field": "FM128"}   # 1.13 要求字符串（枚举无法 JSON 序列化）
    spu_dev = sf.SPU(cluster)

    def fed(a, b):
        return FedNdarray(partitions={neidi: neidi(lambda v: v)(a), hk: hk(lambda v: v)(b)},
                          partition_way=PartitionWay.VERTICAL)

    xte = fed(G[te], H[te])
    rows = []
    for n in args.rows:
        sel = tr[:n]
        xtr = fed(G[sel], H[sel])
        ytr = FedNdarray(partitions={neidi: neidi(lambda v: v)(y[sel].reshape(-1, 1))},
                         partition_way=PartitionWay.VERTICAL)
        params = {"num_boost_round": args.trees, "max_depth": args.max_depth, "learning_rate": 0.3,
                  "objective": "logistic", "reg_lambda": 0.1, "sketch_eps": 0.1, "seed": 42}
        stamps = []
        t0 = time.perf_counter()
        model = Xgb(spu_dev).train(params, xtr, ytr, dump_function=lambda *a, **k: stamps.append(time.perf_counter()))
        fit = time.perf_counter() - t0
        per_tree = np.diff([t0] + stamps).tolist() if stamps else [fit / args.trees]
        steady = float(np.mean(per_tree[1:])) if len(per_tree) > 1 else float(per_tree[0])
        print(f"[SS-XGB {args.protocol}] N={n:,} 训练完成 {fit:.1f}s｜逐棵 {[round(x, 1) for x in per_tree]}"
              f"（首棵含编译；稳态每棵 {steady:.1f}s）", flush=True)
        t1 = time.perf_counter()
        pred = np.asarray(sf.reveal(model.predict(xte, neidi).partitions[neidi])).ravel()
        pt = time.perf_counter() - t1
        auc = float(roc_auc_score(y[te], pred))
        rows.append({"rows": n, "trees": args.trees, "fit_seconds": fit, "tree_seconds": per_tree,
                     "per_tree_seconds": steady,
                     "predict_seconds": pt, "test_auc": auc})
        print(f"[SS-XGB {args.protocol}] N={n:,} 行 × {args.trees} 棵树（深度 {args.max_depth}）：训练 {fit:.1f}s"
              f"（稳态每棵 {steady:.1f}s）｜预测 {pt:.1f}s｜测试 AUC {auc:.4f}", flush=True)
        (REPO_ROOT / args.out).write_text(json.dumps({"protocol": args.protocol, "runs": rows}, indent=2), encoding="utf-8")
    res = {"protocol": args.protocol, "runs": rows}
    if len(rows) >= 2:
        n = np.array([r["rows"] for r in rows], float)
        t = np.array([r["per_tree_seconds"] for r in rows], float)
        b, a = np.polyfit(n, t, 1)
        full = a + b * len(tr)
        res.update({"fit_per_tree": {"a": float(a), "b": float(b)}, "extrapolated_per_tree_full": float(full),
                    "extrapolated_100_trees_hours": float(full * 100 / 3600)})
        print(f"外推：每棵树 ≈ {a:.1f} + {b * 1000:.2f}·(N/1000) 秒 → 全量 {len(tr):,} 行每棵 {full:.0f}s，"
              f"100 棵约 {full * 100 / 3600:.1f} 小时（单机 2 核仿真，不含跨境网络延迟）")
    (REPO_ROOT / args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    sf.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
