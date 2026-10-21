"""[优化版 · 基于原 src/cross_sell/train_federated_sgb.py，默认参数下行为与原版逐行一致]
新增开关（全部默认关闭/保持原值）：--he-schema（paillier|ou）、--enable-packbits、
--enable-quantization/--quantization-scale、--data-dir（读非 prepared/ 目录的同格式矩阵）。

联邦侧训练 · SecureBoost 冒烟（技术栈步骤 3 · TASK-016）—— SecretFlow，跑在 WSL 里。

**为什么有它**：纵向 LR 在本机学不到非标签方特征（EXP-009 / `diagnose_vertical_lr.py`），
而 SGB 走**同一套门禁**三项全过（`diagnose_sgb.py`：信息在协作方时 AUROC 1.0000、
协作方 gain 5301.74 vs 标签方噪声 14.87）→ 本机联邦性能线**改由 SGB 承载**。
本脚本是 SGB 在**真实业务主切分**上的第一次运行：量单次耗时 + 峰值内存。

**冒烟判据**（方案 §9）：① 能跑通 ② 单次 `fit` ≤ 30 min ③ 峰值内存 < 8GB。
冒烟直接用**目标配置与正式 run_id**（不另打 `smoke` 标记）：数据 / 列归属 / 超参都与正式跑一致。

口径（同 `train_federated.py`，两条链读同一份落盘矩阵）：
- 2 方：内地（guest，标签方，20 列）/ 香港（host，25 列）；`debug_mode=True` 单进程
- 加密：HEU / Paillier 2048，sk_keeper = **标签方（内地）**，evaluator = 香港
- 数据：`data/processed/prepared/{tag}_{guest,host,index}.csv`；切分用 `_index.csv` 的 `is_test_time`

**已核实的 SGB 本机 API（secretflow 1.13.0b0）**：
- `Sgb` = `SGBFactory`；训练参数走 `sgb.set_params({...})`（不是 `fit()` 入参）
- `_produce()` 里有 `assert self.heu is not None, "HEU must be set"` → **1.13 的 SGB 关不掉 HE**
  （`heu=None` 跑不了）→ 方案 §3.3「开/关 HE 对照靠 SGB」的设计需另想办法
- `model.feature_importance_device_wise(importance_type='gain')` 返回 `{device: {列下标: 重要性}}`
  → 可当「协作方到底有没有贡献」的直接判据
- 超参照用 `secretflow/ml/boost/sgb_v/core/params.py` 的 `get_classic_XGB_params()`：
  `num_boost_round=10 / learning_rate=0.3 / max_depth=5`（方案要求两端显式设同一组值，不指望这个函数）

跑法（WSL 内，仓库根目录）：

    export LD_LIBRARY_PATH=/home/joshua/.local/lib
    python3 src/cross_sell/train_federated_sgb.py                     # 默认 10 轮 / depth 5 / seed 42
    python3 src/cross_sell/train_federated_sgb.py --limit-rows 20000  # 降级杠杆：缩样本

降级杠杆（方案 §9 预定义的顺序）：缩样本 → 降树深 → 换镜像 / 更大内存机器。
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "prepared"
OUT_DIR = REPO_ROOT / "data" / "processed" / "predictions"

ID_COL = "ncodpers"
LABEL_COL = "label_new_any"

# 冒烟判据（方案 §9）
LIMIT_FIT_SECONDS = 30 * 60
LIMIT_PEAK_MB = 8 * 1024


def peak_mb() -> float:
    """进程峰值常驻内存（MB）。Linux: ru_maxrss 单位是 KB。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0



def _prior(Xtr, ytr, Xap, rho, seed):
    """标签方本地先验：只用自有特征 + 加噪标签训练，再按 RR 噪声率去偏得到 P(y=1|x)。"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=100, max_depth=5, learning_rate=0.1, random_state=seed).fit(Xtr, ytr)
    q = m.predict_proba(Xap)[:, 1]
    return np.clip((q - rho) / (1.0 - 2.0 * rho), 1e-6, 1 - 1e-6)


def _crossfit_prior(X, ynoisy, rho, seed, folds=2):
    p = np.empty(len(ynoisy))
    part = np.random.default_rng(seed + 13).integers(0, folds, len(ynoisy))
    for k in range(folds):
        p[part == k] = _prior(X[part != k], ynoisy[part != k], X[part == k], rho, seed)
    return p


def _posterior(p1, out, rho):
    """对称 RR 输出 out 下 y=1 的后验（DP 后处理，不额外消耗预算）。"""
    k = 1.0 - rho
    return np.where(out == 1, p1 * k / (p1 * k + (1 - p1) * rho), p1 * rho / (p1 * rho + (1 - p1) * k))


def label_dp(Xg: np.ndarray, y: np.ndarray, eps: float, mode: str, stage1: float, seed: int) -> np.ndarray:
    """[安全] 标签差分隐私（ε-LabelDP），四种变体：
    rr            对称随机响应（翻转概率 ρ = 1/(1+e^ε)）
    rr_soft       rr + 后验去噪：用标签方本地先验（只用自有特征与加噪标签，2 折交叉拟合）算 P(y=1|输出,x) 作软标签
    rrprior       RRWithPrior 两阶段（Ghazi 等 NeurIPS 2021 LP-2ST）：阶段 1 对称 RR；阶段 2 先验最大概率 ≥ e^ε/(1+e^ε)
                  时直接输出先验标签，否则对称 RR
    rrprior_soft  rrprior + 后验软标签（先验直接定的样本取先验概率本身）
    软标签是对 DP 输出的后处理，不改变 ε。每个真实标签只被随机化一次 → 整体 ε-LabelDP。
    """
    rng = np.random.default_rng(seed + 11)
    yt = y.ravel().astype(int)
    n = len(yt)
    rho = 1.0 / (1.0 + np.exp(eps))
    if mode in ("rr", "rr_soft"):
        out = np.where(rng.random(n) < rho, 1 - yt, yt)
        lab = out.astype(float)
        if mode == "rr_soft":
            lab = _posterior(_crossfit_prior(Xg, out, rho, seed), out, rho)
    else:
        s1 = rng.random(n) < stage1
        out = yt.copy()
        out[s1] = np.where(rng.random(int(s1.sum())) < rho, 1 - yt[s1], yt[s1])
        p1 = _prior(Xg[s1], out[s1], Xg[~s1], rho, seed)
        top = (p1 >= 0.5).astype(int)
        det = np.maximum(p1, 1 - p1) >= np.exp(eps) / (1.0 + np.exp(eps))
        y2 = yt[~s1]
        out[~s1] = np.where(det, top, np.where(rng.random(len(y2)) < rho, 1 - y2, y2))
        lab = out.astype(float)
        if mode == "rrprior_soft":
            lab[s1] = _posterior(_crossfit_prior(Xg[s1], out[s1], rho, seed), out[s1], rho)
            lab[~s1] = np.where(det, p1, _posterior(p1, out[~s1], rho))
        print(f"       [安全] 阶段 1 {int(s1.sum()):,} 行；阶段 2 中 {det.mean():.1%} 由先验直接定标签")
    print(f"       [安全] 标签 DP ε={eps}（{mode}）：训练标签均值 {lab.mean():.4f}（真实正例率 {yt.mean():.4f}），"
          f"硬输出与真值不同 {int((out != yt).sum()):,} 行")
    return lab.reshape(-1, 1)

def main(argv: list[str] | None = None) -> int:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")  # 让 SF / SGB 的逐树日志打出来

    ap = argparse.ArgumentParser(description="联邦侧训练 · SGB（步骤 3 冒烟 / 正式跑）")
    ap.add_argument("--tag", default="business")
    ap.add_argument("--role", default="federated", choices=["federated"])
    ap.add_argument("--num-boost-round", type=int, default=10, help="树数量（对齐 XGBoost 经典默认）")
    ap.add_argument("--max-depth", type=int, default=5, help="树深（对齐 XGBoost 经典默认）")
    ap.add_argument("--learning-rate", type=float, default=0.3)
    ap.add_argument("--reg-lambda", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-rows", type=int, default=0, help="只用前 N 行（降级用）；0 = 全量")
    ap.add_argument("--run-id", default=None, help="默认 {tag}_sgb_{role}_m{seed}")
    ap.add_argument("--audit-dir", default=None,
                    help="审计日志目录（方案 §3.1 写死项）：给出则把 SGB 的 audit_paths 指向它，"
                         "产出 `<dir>/{run_id}.tree_N`（HEU 加密审计）与 split_buckets pickle —— §3.3 证据②")
    ap.add_argument("--variant", default=None, choices=[None, "robust"],
                    help="robust = 「联邦·稳健」：host 侧剥掉全部产品列，只留 ind_actividad_cliente（归因工具）")
    ap.add_argument("--he-schema", default="paillier", choices=["paillier", "ou"],
                    help="[优化] HEU 同态方案：paillier（原版）/ ou（Okamoto-Uchiyama，加法同态、加解密更快）")
    ap.add_argument("--enable-packbits", action="store_true", help="[优化] SGB packbits 传输")
    ap.add_argument("--enable-quantization", action="store_true", help="[优化] g/h 量化（有损，须核对 AUC）")
    ap.add_argument("--quantization-scale", type=float, default=10000.0)
    ap.add_argument("--early-stop", action="store_true",
                    help="[优化④] SGB 内置早停：validation_fraction=0.2 / stopping_rounds=10 / eval_metric=roc_auc / save_best_model")
    # ---- [优化] 精度 / 效率开关（不给 = 隐语默认值，与原版一致）
    ap.add_argument("--sketch-eps", type=float, default=None, help="[优化] 分桶粒度，约 1/eps 个桶；隐语默认 0.1（≈10 桶）；1.13 实测下限 0.008（0.005 报错、0.004 段错误）")
    ap.add_argument("--enable-goss", action="store_true", help="[优化] GOSS 梯度单边采样（有损）")
    ap.add_argument("--top-rate", type=float, default=0.2)
    ap.add_argument("--bottom-rate", type=float, default=0.1)
    ap.add_argument("--rowsample", type=float, default=None)
    ap.add_argument("--colsample", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    # ---- [安全] 防护开关与攻击取证
    ap.add_argument("--first-tree-label-holder", type=int, default=None, choices=[0, 1],
                    help="[安全] 首棵树只用标签方特征（隐语默认 1）；0 = 关闭，用于对照")
    ap.add_argument("--label-rr-eps", type=float, default=None,
                    help="[安全] 标签方对**训练**标签做随机响应（标签差分隐私，ε）；测试标签不动")
    ap.add_argument("--label-dp", default="rr", choices=["rr", "rr_soft", "rrprior", "rrprior_soft"],
                    help="[安全] 与 --label-rr-eps 合用，见 label_dp()：rr 对称随机响应 / rr_soft 加后验去噪 / rrprior RRWithPrior 两阶段 / rrprior_soft 加后验软标签")
    ap.add_argument("--label-dp-stage1", type=float, default=0.3,
                    help="[安全] rrprior 第一阶段样本占比（该部分用对称 RR，并用来训练标签方本地先验模型）")
    ap.add_argument("--base-score", default=None,
                    help="[优化] 初始分数（logit 尺度）：数值，或 prior = 标签方用训练标签均值算 logit（隐语默认 0，即从 50% 起步，"
                         "正例极少时前几棵树被浪费）")
    ap.add_argument("--dump-model", default=None,
                    help="[攻击取证] 训练后把各方分裂树导出为 JSON（debug_mode 下 reveal），供 src/ext/attacks.py 重放各方所见")
    ap.add_argument("--data-dir", default=None, help="[扩展] 矩阵目录（默认 data/processed/prepared）")
    ap.add_argument("--bit-size", type=int, default=2048,
                    help="Paillier 密钥长度（§3.1 写死 2048；加密替代证据的 1024 臂用此覆盖）")
    args = ap.parse_args(argv)

    global DATA_DIR
    if args.data_dir:
        DATA_DIR = (REPO_ROOT / args.data_dir) if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    run_id = args.run_id or f"{args.tag}_sgb_{args.role}_m{args.seed}"
    t_start = time.perf_counter()

    g = pd.read_csv(DATA_DIR / f"{args.tag}_guest.csv", low_memory=False)
    h = pd.read_csv(DATA_DIR / f"{args.tag}_host.csv", low_memory=False)
    idx = pd.read_csv(DATA_DIR / f"{args.tag}_index.csv", low_memory=False)
    assert g[ID_COL].equals(h[ID_COL]) and g[ID_COL].equals(idx[ID_COL]), "三份产物行序不一致"

    tr = idx["is_test_time"].to_numpy() == 0
    te = ~tr
    if args.limit_rows:
        keep = np.zeros(len(g), dtype=bool)
        keep[np.where(tr)[0][: args.limit_rows]] = True
        keep[np.where(te)[0][: max(args.limit_rows // 3, 1)]] = True
        tr, te = keep & tr, keep & te

    gcols = [c for c in g.columns if c != ID_COL]
    hcols = [c for c in h.columns if c != ID_COL]
    if args.variant == "robust":   # 「联邦·稳健」：host 侧只留活跃度（§2.5 归因工具）
        keep = [c for c in hcols if c == "ind_actividad_cliente"]
        if not keep:
            raise ValueError("variant=robust 需要 host 侧存在 ind_actividad_cliente")
        dropped = [c for c in hcols if c not in keep]
        hcols = keep
        print(f"       variant=robust：host 侧剥掉 {len(dropped)} 个产品列，只留 {keep}")
    Xg = g.loc[tr, gcols].to_numpy(dtype=np.float64)
    Xh = h.loc[tr, hcols].to_numpy(dtype=np.float64)
    y = idx.loc[tr, LABEL_COL].to_numpy(dtype=np.float64).reshape(-1, 1)  # 列向量
    if args.label_rr_eps is not None and args.label_dp == "rr":   # [安全] 标签随机响应：以 1/(1+e^ε) 概率翻转训练标签
        flip_p = 1.0 / (1.0 + np.exp(args.label_rr_eps))
        flip = np.random.default_rng(args.seed + 7).random(y.shape[0]) < flip_p
        y = np.where(flip.reshape(-1, 1), 1.0 - y, y)
        print(f"       [安全] 标签随机响应 ε={args.label_rr_eps}：翻转概率 {flip_p:.4f}，实际翻转 {int(flip.sum()):,} 行")
    elif args.label_rr_eps is not None:
        y = label_dp(Xg, y, args.label_rr_eps, args.label_dp, args.label_dp_stage1, args.seed)
    Xg_te = g.loc[te, gcols].to_numpy(dtype=np.float64)
    Xh_te = h.loc[te, hcols].to_numpy(dtype=np.float64)
    y_te = idx.loc[te, LABEL_COL].to_numpy()

    print(f"[{run_id}] 训练 {Xg.shape[0]:,} 行（内地 {len(gcols)} 列 / 香港 {len(hcols)} 列，"
          f"合计 {len(gcols) + len(hcols)}）| 测试 {Xg_te.shape[0]:,} 行 / {int(y_te.sum())} 正例")
    print(f"       超参：num_boost_round={args.num_boost_round} max_depth={args.max_depth} "
          f"learning_rate={args.learning_rate} reg_lambda={args.reg_lambda} seed={args.seed}")
    print(f"       加载与切分耗时 {time.perf_counter() - t_start:.1f}s，峰值内存 {peak_mb():.0f} MB")

    import secretflow as sf
    import spu
    from secretflow.data.ndarray import FedNdarray, PartitionWay
    from secretflow.device import reveal
    from secretflow.ml.boost.sgb_v import Sgb
    from sklearn.metrics import roc_auc_score

    sf.init(parties=["neidi", "hk"], address="local", debug_mode=True)
    neidi, hk = sf.PYU("neidi"), sf.PYU("hk")
    print(f"       sf.init 完成（debug_mode=True，2 方），峰值内存 {peak_mb():.0f} MB")

    # HEU：私钥归标签方（内地），香港只持公钥做同态计算
    heu_cfg = sf.utils.testing.heu_config("neidi", ["hk"])
    if args.bit_size != 2048:      # 加密替代证据（§3.3 ①）：只改密钥强度，其余不动
        heu_cfg["he_parameters"]["key_pair"]["generate"]["bit_size"] = args.bit_size
        print(f"       加密替代臂：Paillier bit_size = {args.bit_size}（默认 2048）")
    if args.he_schema != "paillier":   # [优化] 只换同态方案，密钥长度 / 编码不动
        heu_cfg["he_parameters"]["schema"] = args.he_schema
        print(f"       [优化] HE schema = {args.he_schema}（bit_size {heu_cfg['he_parameters']['key_pair']['generate']['bit_size']}）")
    heu = sf.HEU(heu_cfg, spu.FieldType.FM64)

    x = FedNdarray(
        partitions={neidi: neidi(lambda: Xg)(), hk: hk(lambda: Xh)()},
        partition_way=PartitionWay.VERTICAL,
    )
    y_fed = FedNdarray(
        partitions={neidi: neidi(lambda: y)()},
        partition_way=PartitionWay.VERTICAL,
    )

    sgb = Sgb(heu=heu)
    # 审计日志（方案 §3.1 写死项 / §3.3 证据②）：SGB 的 audit_paths 按 party 名键，
    # 由 label_holder 侧写出 `<path>.tree_N`（HEU encrypt_with_audit 的产物）与
    # `<path>.split_buckets.tree_N.*.pickle` → 证明 Paillier 路径真被调用
    params = {
        "num_boost_round": args.num_boost_round,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "reg_lambda": args.reg_lambda,
        "objective": "logistic",
        "seed": args.seed,
    }
    for k, v in (("sketch_eps", args.sketch_eps), ("rowsample_by_tree", args.rowsample),
                 ("colsample_by_tree", args.colsample), ("gamma", args.gamma)):
        if v is not None:
            params[k] = v
    if args.base_score is not None:
        if args.base_score == "prior":
            pm = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
            params["base_score"] = float(np.clip(np.log(pm / (1 - pm)), -10, 10))
        else:
            params["base_score"] = float(args.base_score)
        print(f"       [优化] base_score = {params['base_score']:.4f}（{args.base_score}）")
    if args.enable_goss:
        params.update({"enable_goss": True, "top_rate": args.top_rate, "bottom_rate": args.bottom_rate})
    if args.first_tree_label_holder is not None:
        params["first_tree_with_label_holder_feature"] = bool(args.first_tree_label_holder)
    if args.early_stop:
        params.update({"enable_early_stop": True, "validation_fraction": 0.2, "stopping_rounds": 10,
                       "eval_metric": "roc_auc", "save_best_model": True, "stopping_tolerance": 0.0})
    if args.enable_packbits:
        params["enable_packbits"] = True
    if args.enable_quantization:
        params["enable_quantization"] = True
        params["quantization_scale"] = args.quantization_scale
    if args.audit_dir:
        audit_dir = Path(args.audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_prefix = audit_dir / run_id
        params["audit_paths"] = {"neidi": str(audit_prefix), "hk": str(audit_prefix)}
        print(f"       审计日志（§3.3 证据②）：{audit_prefix}.tree_N / .split_buckets.*")
    sgb.set_params(params)

    print(f"       开始 fit（{args.num_boost_round} 轮 / depth {args.max_depth}）…", flush=True)
    t0 = time.perf_counter()
    model = sgb.fit(x, y_fed)
    fit_s = time.perf_counter() - t0
    print(f"✅ fit 完成：耗时 {fit_s:.1f}s（{fit_s / 60:.2f} 分钟）｜峰值内存 {peak_mb():.0f} MB")

    if args.dump_model:   # [攻击取证] debug_mode 下各方对象都在本进程，reveal 后落盘（真实部署中各方只持有自己那份）
        md = reveal(model.to_dict()) if not isinstance(model.to_dict(), dict) else model.to_dict()
        def _plain(o):
            if hasattr(o, "data") and hasattr(o, "device"):
                o = reveal(o)
            if isinstance(o, dict):
                return {str(k): _plain(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_plain(v) for v in o]
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if o is None or isinstance(o, (bool, int, float, str)):
                return o
            return str(o)   # PYU 等设备句柄 → 方名字符串
        Path(args.dump_model).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump_model).write_text(json.dumps({"model": _plain(md), "gcols": gcols, "hcols": hcols,
                                                     "train_rows": np.where(tr)[0].tolist()}), encoding="utf-8")
        print(f"       [攻击取证] 模型分裂结构已导出：{args.dump_model}")

    # 判据判定
    print(f"       判据① 能跑通：✅（未抛异常、模型已产出）")
    print(f"       判据② 单次 fit ≤ {LIMIT_FIT_SECONDS / 60:.0f} min：{fit_s:.1f}s → "
          f"{'✅' if fit_s <= LIMIT_FIT_SECONDS else '❌ 超时'}")
    print(f"       判据③ 峰值内存 < {LIMIT_PEAK_MB / 1024:.0f} GB：{peak_mb():.0f} MB → "
          f"{'✅' if peak_mb() < LIMIT_PEAK_MB else '❌ 超内存'}")

    # 每方各自的特征重要性：直接看「协作方到底有没有贡献」（LR 就是死在这里）
    try:
        fi = model.feature_importance_device_wise()
        if not fi:
            print("       特征重要性：空（模型没有树）")
        else:
            for dev, d in fi.items():
                party = getattr(dev, "party", str(dev))
                d = {int(k): float(v) for k, v in dict(d).items()}
                cols = gcols if party == "neidi" else hcols
                named = {cols[k] if k < len(cols) else k: round(v, 2) for k, v in d.items()}
                print(f"       特征重要性 · {party}（合计 {sum(d.values()):.2f}）：{named}")
    except Exception as e:  # noqa: BLE001 - 诊断输出，失败不影响冒烟结论
        print(f"       （特征重要性取不到：{type(e).__name__}: {e}）")

    # 预测 + AUC
    x_te = FedNdarray(
        partitions={neidi: neidi(lambda: Xg_te)(), hk: hk(lambda: Xh_te)()},
        partition_way=PartitionWay.VERTICAL,
    )
    t1 = time.perf_counter()
    pred = reveal(model.predict(x_te)).ravel()
    auc = float(roc_auc_score(y_te, pred))
    print(f"✅ 预测完成：耗时 {time.perf_counter() - t1:.1f}s｜测试集 AUROC = {auc:.4f}")
    print(f"       预测分布：min={pred.min():.4f} max={pred.max():.4f} "
          f"mean={pred.mean():.4f} std={pred.std():.4f}")
    top = np.argsort(-pred)[: max(int(0.05 * len(pred)), 1)]
    lift = float(y_te[top].mean() / y_te.mean())
    print(f"       Top-5% 正例率 = {y_te[top].mean():.4f}（基准 {y_te.mean():.4f}，Lift {lift:.2f}×）")

    # 冒烟参照：同一份矩阵上的 sklearn 直方图梯度提升（**不是方案口径**——方案明文基线用 XGBoost，
    # 此数只为判断「这个矩阵上一个树模型大致能到哪」，避免把 SF 的 SGB 数当孤证）
    from sklearn.ensemble import HistGradientBoostingClassifier

    hgb = HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42)
    t2 = time.perf_counter()
    hgb.fit(np.hstack([Xg, Xh]), (y.ravel() >= 0.5).astype(int))   # 软标签（标签 DP 后处理）时取硬化标签，仅作参照
    hgb_auc = float(roc_auc_score(y_te, hgb.predict_proba(np.hstack([Xg_te, Xh_te]))[:, 1]))
    print(f"       冒烟参照（非方案口径）同矩阵 sklearn HistGB(100 轮, depth5) AUROC = {hgb_auc:.4f}"
          f"（耗时 {time.perf_counter() - t2:.0f}s）")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 逐行预测落盘（方案规定的产物格式：客户ID + 时间列 + y_true + score）
    # 用途：① §6.3 名单层 Lift 重算（快照月 / 客户级）② §6.6 配对差值 CI（同一批样本上配对 bootstrap）
    # 行序自校验：AUC 与 y_true 同批计算；行序若错位，AUC 会塌到 0.5 附近（本次 0.8750 ✓）
    pred_df = pd.DataFrame(
        {
            "ncodpers": idx.loc[te, ID_COL].to_numpy(),
            "fecha_dato": idx.loc[te, "fecha_dato"].to_numpy() if "fecha_dato" in idx.columns else np.full(int(te.sum()), ""),   # FedAds 等无月份列的数据集
            "y_true": y_te.astype(int),
            "score": pred,
        }
    )
    pred_path = OUT_DIR / f"{run_id}_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"✅ 逐行预测：data/processed/predictions/{pred_path.name}（{len(pred_df):,} 行）")

    summary = {
        "run_id": run_id, "tag": args.tag, "model": "sgb", "role": args.role,
        "num_boost_round": args.num_boost_round, "max_depth": args.max_depth,
        "learning_rate": args.learning_rate, "reg_lambda": args.reg_lambda, "seed": args.seed,
        "n_train": int(Xg.shape[0]), "n_test": int(Xg_te.shape[0]),
        "n_guest_cols": len(gcols), "n_host_cols": len(hcols),
        "limit_rows": args.limit_rows,
        "fit_seconds": fit_s, "total_seconds": time.perf_counter() - t_start,
        "peak_memory_mb": peak_mb(), "auroc": auc, "lift_top5": lift,
        # 痕迹补全（2026-09-28）：variant / bit_size 必须进记录，否则事后只能靠文件名反推
        "variant": args.variant, "bit_size": args.bit_size, "he_schema": args.he_schema,
        "enable_packbits": args.enable_packbits, "early_stop": args.early_stop,
        "sketch_eps": args.sketch_eps, "enable_goss": args.enable_goss, "first_tree_label_holder": args.first_tree_label_holder,
        "base_score": params.get("base_score"), "label_rr_eps": args.label_rr_eps, "label_dp": args.label_dp if args.label_rr_eps is not None else None, "effective_params": {k: v for k, v in params.items() if k != "audit_paths"}, "enable_quantization": args.enable_quantization, "audit_dir": args.audit_dir,
        "hgb_reference_auroc": hgb_auc,
        "predictions_file": str(pred_path.relative_to(REPO_ROOT)),
        "criteria": {
            "run_ok": True,
            "fit_le_30min": bool(fit_s <= LIMIT_FIT_SECONDS),
            "peak_lt_8gb": bool(peak_mb() < LIMIT_PEAK_MB),
        },
    }
    # 正式配置的首次运行也用**正式 run_id**，文件名**不带 `_smoke` 标记**（§9 纪律：
    # 同一配置不另打标记——否则报告主数会追着一个叫 smoke 的文件）
    (OUT_DIR / f"{run_id}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ 运行记录：data/processed/predictions/{run_id}.json")
    sf.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
