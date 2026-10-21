"""[切分] VertiBench 式切分扫描（Wu et al., ICLR 2024, arXiv 2307.02040）——联邦增益随「特征怎么分」如何变化。

VertiBench 用两个旋钮刻画纵向切分：
- 重要性切分 α：各方占比 r ~ Dir(α, α)，每列按 r 随机归属（先各发 1 列防空方）；E[方 i 的重要性] ∝ α_i，
  α 越小越失衡。重要性用 TreeSHAP 均值 |φ|（集中式模型、训练集抽样）归一化。
- 相关性切分 β：Pcor(Xi,Xj) = std(奇异值(Spearman 相关阵 C_ij), ddof=1)/√d，
  Icor = 跨方 Pcor − 方内 Pcor 均值（K=2 时）；先用 BRKGA 求 Icor 的最小/最大，再找 Icor 最接近
  (1−β)·min + β·max 的切分。β=0：方内强相关、跨方弱相关；β=1：反之。双方列数固定为业务切分的 20 / 25。
  另做一组「控重要性」的相关性切分：目标函数再加 |香港重要性占比 − 业务切分占比|，把重要性钉住只动相关性。

对每个切分在明文代理上测（SGB 与明文直方图 GBDT 同分桶等价，优化版 SGB 与集中式等效已在测试集确认）：
  内地单边 / 香港单边 AUC（各自早停定轮数），集中式 AUC 不随切分变；
  代理联邦增益 Ĝ = 集中式 − 内地单边（标签方自己的收益），以及 集中式 − max(两单边)（原项目判据）。
配对 CI：客户级 cluster bootstrap（向量化实现，同一批重采样）。
同时给出真实业务切分与 5 个随机切分在 (香港重要性占比, β) 平面上的位置（VertiBench 的 α̂、β̂ 估计）。

    python src/ext/vertibench_scan.py --alphas 0.1 0.3 1 3 10 100 --alpha-seeds 10 --betas 0 0.25 0.5 0.75 1 --beta-seeds 4
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
OUT = REPO_ROOT / "data" / "processed"
PARAMS = {"objective": "binary:logistic", "eta": 0.1, "max_depth": 6, "lambda": 10.0, "tree_method": "hist",
          "max_bin": 256, "nthread": 2, "seed": 42, "eval_metric": "auc"}


# ---------------------------------------------------------------- 数据
def load(tag: str):
    g = pd.read_csv(DATA_DIR / f"{tag}_guest.csv")
    h = pd.read_csv(DATA_DIR / f"{tag}_host.csv")
    idx = pd.read_csv(DATA_DIR / f"{tag}_index.csv")
    assert g["ncodpers"].equals(h["ncodpers"]) and g["ncodpers"].equals(idx["ncodpers"])
    X = pd.concat([g.drop(columns="ncodpers"), h.drop(columns="ncodpers")], axis=1)
    return X, idx, list(g.columns[1:]), list(h.columns[1:])


# ---------------------------------------------------------------- 配对 cluster bootstrap（向量化）
class PairedBoot:
    """每次重采样按客户多项式计数加权；所有分数共用同一批权重 → 配对差值。"""

    def __init__(self, y: np.ndarray, ids: np.ndarray, n_boot: int = 1000, seed: int = 42):
        uniq, inv = np.unique(ids, return_inverse=True)
        rng = np.random.default_rng(seed)
        cnt = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=n_boot).astype(np.float32)
        self.W = cnt[:, inv]
        self.y = y.astype(bool)

    def aucs(self, s: np.ndarray) -> np.ndarray:
        o = np.argsort(s, kind="mergesort")
        ss = s[o]
        starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]
        W = self.W[:, o]
        yo = self.y[o]
        P = np.add.reduceat(W * yo, starts, axis=1)
        N = np.add.reduceat(W * ~yo, starts, axis=1)
        below = np.cumsum(N, axis=1) - N
        return (P * (below + 0.5 * N)).sum(1) / (P.sum(1) * N.sum(1))


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


# ---------------------------------------------------------------- 模型
def fit_es(cols, D):
    """在验证 tag 上早停定轮数，再用全部训练行按该轮数重训，返回测试分数与轮数。"""
    import xgboost as xgb
    Xa, ya, Xv, yv, Xt, yt, Xe = D["Xa"], D["ya"], D["Xv"], D["yv"], D["Xt"], D["yt"], D["Xe"]
    b = xgb.train(PARAMS, xgb.DMatrix(Xa[:, cols], ya), 1000, evals=[(xgb.DMatrix(Xv[:, cols], yv), "va")],
                  early_stopping_rounds=50, verbose_eval=False)
    n = b.best_iteration + 1
    b2 = xgb.train(PARAMS, xgb.DMatrix(Xt[:, cols], yt), n)
    return b2.predict(xgb.DMatrix(Xe[:, cols])), n


def shap_share(D, n_rows: int = 20000, seed: int = 42):
    import xgboost as xgb
    cols = np.arange(D["Xt"].shape[1])
    _, n = fit_es(cols, D)
    b = xgb.train(PARAMS, xgb.DMatrix(D["Xt"], D["yt"]), n)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(D["Xt"]), size=min(n_rows, len(D["Xt"])), replace=False)
    phi = b.predict(xgb.DMatrix(D["Xt"][pick]), pred_contribs=True)[:, :-1]
    imp = np.abs(phi).mean(0)
    return imp / imp.sum()


# ---------------------------------------------------------------- 相关性度量
def spearman(X: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    R = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])])
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(R, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    return C


def pcor(C: np.ndarray) -> float:
    s = np.linalg.svd(C, compute_uv=False)
    return float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0


def icor(R: np.ndarray, gmask: np.ndarray) -> float:
    g, h = np.flatnonzero(gmask), np.flatnonzero(~gmask)
    return pcor(R[np.ix_(g, h)]) - 0.5 * (pcor(R[np.ix_(g, g)]) + pcor(R[np.ix_(h, h)]))


def brkga(fit, n: int, m_g: int, seed: int, pop: int = 80, gens: int = 200, elite: float = 0.2,
          mutant: float = 0.15, rho: float = 0.7):
    """Biased Random-Key GA（Gonçalves & Resende 2011）：随机键排序取前 m_g 列给内地，最小化 fit(mask)。"""
    rng = np.random.default_rng(seed)
    K = rng.random((pop, n))
    ne, nm = int(pop * elite), int(pop * mutant)

    def decode(k):
        m = np.zeros(n, bool)
        m[np.argsort(k)[:m_g]] = True
        return m

    best_f, best_m = np.inf, None
    for _ in range(gens):
        f = np.array([fit(decode(k)) for k in K])
        o = np.argsort(f)
        K, f = K[o], f[o]
        if f[0] < best_f:
            best_f, best_m = float(f[0]), decode(K[0])
        el = K[:ne]
        mu = rng.random((nm, n))
        nc = pop - ne - nm
        pe = el[rng.integers(0, ne, nc)]
        po = K[ne:][rng.integers(0, pop - ne, nc)]
        child = np.where(rng.random((nc, n)) < rho, pe, po)
        K = np.vstack([el, mu, child])
    return best_m, best_f


# ---------------------------------------------------------------- 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="business")
    ap.add_argument("--val-tag", default="business_val")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.3, 1, 3, 10, 100])
    ap.add_argument("--alpha-seeds", type=int, default=10)
    ap.add_argument("--betas", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1])
    ap.add_argument("--beta-seeds", type=int, default=4)
    ap.add_argument("--random-tags", default="random_seed42,random_seed43,random_seed44,random_seed45,random_seed46")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--fix-share", type=int, default=1, help="1=额外做「控香港重要性占比」的相关性切分")
    ap.add_argument("--out", default="data/processed/vertibench_scan.json")
    args = ap.parse_args(argv)
    t0 = time.time()

    X, idx, gcols0, hcols0 = load(args.tag)
    Xv_df, idxv, _, _ = load(args.val_tag)
    names = list(X.columns)
    assert list(Xv_df.columns) == names
    tr = idx["is_test_time"].to_numpy() == 0
    te = ~tr
    va = idxv["is_test_time"].to_numpy() == 1
    y = idx["label_new_any"].to_numpy().astype(float)
    yv_all = idxv["label_new_any"].to_numpy().astype(float)
    Xn, Xvn = X.to_numpy(float), Xv_df.to_numpy(float)
    D = {"Xa": Xvn[~va], "ya": yv_all[~va], "Xv": Xvn[va], "yv": yv_all[va],
         "Xt": Xn[tr], "yt": y[tr], "Xe": Xn[te]}
    yte, ids = y[te], idx.loc[te, "ncodpers"].to_numpy()
    boot = PairedBoot(yte, ids, args.n_boot)
    n = len(names)
    bmask = np.array([c in set(gcols0) for c in names])
    m_g = int(bmask.sum())

    share = shap_share(D)
    R = spearman(Xn[tr][np.random.default_rng(0).choice(int(tr.sum()), 30000, replace=False)])
    print(f"[{args.tag}] {n} 列（业务切分 内地 {m_g} / 香港 {n - m_g}）｜训练 {int(tr.sum()):,} 测试 {int(te.sum()):,}"
          f"｜{time.time() - t0:.0f}s")

    s_c, n_c = fit_es(np.arange(n), D)
    cent = {"auc": auc(yte, s_c), "rounds": n_c}
    b_c = boot.aucs(s_c)
    print(f"集中式 AUC {cent['auc']:.4f}（{n_c} 轮）")

    # Icor 的可达范围（固定 20/25 列）
    lo_m, lo_f = min((brkga(lambda m: icor(R, m), n, m_g, s) for s in (1, 2)), key=lambda t: t[1])
    hi_m, hi_f = min((brkga(lambda m: -icor(R, m), n, m_g, s) for s in (1, 2)), key=lambda t: t[1])
    imin, imax = lo_f, -hi_f
    print(f"Icor 范围 [{imin:.4f}, {imax:.4f}]（BRKGA）｜{time.time() - t0:.0f}s")

    cache: dict[tuple, dict] = {}

    def evaluate(gmask: np.ndarray, meta: dict) -> dict:
        gmask = gmask.copy()
        key = tuple(np.flatnonzero(gmask))
        if key not in cache:
            gi, hi = np.flatnonzero(gmask), np.flatnonzero(~gmask)
            sg, ng = fit_es(gi, D)
            sh, nh = fit_es(hi, D) if len(hi) else (np.full(len(yte), 0.5), 0)
            bg, bh = boot.aucs(sg), boot.aucs(sh)
            ag, ah = auc(yte, sg), auc(yte, sh)
            best_b, best_a = (bg, ag) if ag >= ah else (bh, ah)
            dg, db = b_c - bg, b_c - best_b
            ic = icor(R, gmask) if 1 < len(gi) < n - 1 else float("nan")
            cache[key] = {
                "n_guest": int(len(gi)), "guest_cols": [names[i] for i in gi],
                "host_share": float(share[hi].sum()), "icor": ic,
                "beta_hat": float(np.clip((ic - imin) / (imax - imin), 0, 1)) if ic == ic else float("nan"),
                "guest_auc": ag, "host_auc": ah, "guest_rounds": ng, "host_rounds": nh,
                "gain_vs_guest": cent["auc"] - ag, "gain_vs_guest_ci": [float(np.percentile(dg, 2.5)), float(np.percentile(dg, 97.5))],
                "gain_vs_best": cent["auc"] - best_a, "gain_vs_best_ci": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
            }
        r = {**meta, **cache[key]}
        rows.append(r)
        print(f"  {json.dumps(meta)[:60]:60s} 内地{r['n_guest']:2d}列 香港占比{r['host_share']:.2f} β̂{r['beta_hat']:.2f}"
              f" 内地{r['guest_auc']:.4f} 香港{r['host_auc']:.4f} Ĝ{r['gain_vs_guest']:+.4f}"
              f" [{r['gain_vs_guest_ci'][0]:+.4f},{r['gain_vs_guest_ci'][1]:+.4f}] ｜{time.time() - t0:.0f}s", flush=True)
        return r

    rows: list[dict] = []
    # 真实业务切分 + 随机切分（原项目对照）
    evaluate(bmask, {"kind": "business"})
    for t in [s for s in args.random_tags.split(",") if s]:
        cols = json.loads((DATA_DIR / f"{t}_columns.json").read_text(encoding="utf-8"))
        gset = set(pd.read_csv(DATA_DIR / f"{t}_guest.csv", nrows=1).columns[1:])
        evaluate(np.array([c in gset for c in names]), {"kind": "random", "tag": t, "n_guest_meta": cols.get("n_guest_features")})
    # 重要性切分
    for a in args.alphas:
        for s in range(args.alpha_seeds):
            rng = np.random.default_rng(1000 * s + 7)
            r = rng.dirichlet([a, a])
            m = rng.random(n) < r[0]
            first = rng.permutation(n)[:2]          # 各方至少 1 列（VertiBench 可选做法）
            m[first[0]], m[first[1]] = True, False
            evaluate(m, {"kind": "importance", "alpha": a, "seed": s, "r_guest": float(r[0])})
    # 相关性切分
    for b in args.betas:
        target = (1 - b) * imin + b * imax
        for s in range(args.beta_seeds):
            m, f = brkga(lambda mm: abs(icor(R, mm) - target), n, m_g, seed=100 + 10 * s + int(b * 4))
            evaluate(m, {"kind": "correlation", "beta": b, "seed": s, "icor_target": target})

    # 相关性切分（控重要性）：同时把香港重要性占比钉在业务切分的水平，单独看跨方相关性的作用
    if args.fix_share:
        s0 = float(share[~bmask].sum())
        span = imax - imin
        for b in args.betas:
            target = (1 - b) * imin + b * imax
            for s in range(args.beta_seeds):
                m, f = brkga(lambda mm: abs(icor(R, mm) - target) / span + abs(float(share[~mm].sum()) - s0), n, m_g,
                             seed=500 + 10 * s + int(b * 4))
                evaluate(m, {"kind": "correlation_fixshare", "beta": b, "seed": s, "icor_target": target, "share_target": s0})

    # α̂（VertiBench：对称 Dirichlet 同方差反推，K=2）
    def alpha_hat(sh):
        sig = ((sh - 0.5) ** 2 + ((1 - sh) - 0.5) ** 2) / 2
        return float((1 - 4 * sig) / (8 * sig)) if sig > 0 else float("inf")

    for r in rows:
        r["alpha_hat"] = alpha_hat(r["host_share"])
    res = {"tag": args.tag, "central": cent, "icor_min": imin, "icor_max": imax, "params": PARAMS,
           "feature_share": dict(zip(names, map(float, share))), "rows": rows,
           "elapsed_s": time.time() - t0}
    (REPO_ROOT / args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {args.out}｜合计 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
