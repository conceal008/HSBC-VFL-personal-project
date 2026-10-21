"""[切分] 汇总 vertibench_scan.json：各 α / β 档的代理增益、显著比例、与香港重要性占比 / β̂ 的相关。

    python src/ext/vertibench_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    path = REPO_ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/processed/vertibench_scan.json")
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d["rows"]
    out = {"central_auc": d["central"]["auc"], "icor_range": [d["icor_min"], d["icor_max"]]}
    print(f"集中式 {d['central']['auc']:.4f}｜Icor 范围 [{d['icor_min']:.4f}, {d['icor_max']:.4f}]")

    def grp(kind, key):
        res = []
        for v in sorted({r[key] for r in rows if r["kind"] == kind}):
            rs = [r for r in rows if r["kind"] == kind and r[key] == v]
            g = np.array([r["gain_vs_guest"] for r in rs])
            sig = np.mean([r["gain_vs_guest_ci"][0] > 0 for r in rs])
            res.append({key: v, "n": len(rs), "gain_mean": float(g.mean()), "gain_min": float(g.min()),
                        "gain_max": float(g.max()), "sig_frac": float(sig),
                        "host_share_mean": float(np.mean([r["host_share"] for r in rs])),
                        "beta_hat_mean": float(np.nanmean([r["beta_hat"] for r in rs])),
                        "guest_auc_mean": float(np.mean([r["guest_auc"] for r in rs])),
                        "n_guest_mean": float(np.mean([r["n_guest"] for r in rs]))})
            x = res[-1]
            print(f"  {kind:22s} {key}={v:<6g} n={x['n']:2d} Ĝ均值 {x['gain_mean']:+.4f} [{x['gain_min']:+.4f}, {x['gain_max']:+.4f}]"
                  f" 显著 {x['sig_frac']:.0%} 香港占比 {x['host_share_mean']:.2f} β̂ {x['beta_hat_mean']:.2f} 内地列 {x['n_guest_mean']:.1f}")
        return res

    out["importance"] = grp("importance", "alpha")
    out["correlation"] = grp("correlation", "beta")
    if any(r["kind"] == "correlation_fixshare" for r in rows):
        out["correlation_fixshare"] = grp("correlation_fixshare", "beta")
    for r in rows:
        if r["kind"] in ("business", "random"):
            print(f"  {r['kind']:10s} {r.get('tag', ''):14s} 香港占比 {r['host_share']:.2f} β̂ {r['beta_hat']:.2f} α̂ {r['alpha_hat']:.2f}"
                  f" 内地 {r['guest_auc']:.4f} Ĝ {r['gain_vs_guest']:+.4f} [{r['gain_vs_guest_ci'][0]:+.4f}, {r['gain_vs_guest_ci'][1]:+.4f}]")
    out["business"] = [r for r in rows if r["kind"] == "business"][0]
    out["random"] = [r for r in rows if r["kind"] == "random"]

    synth = [r for r in rows if r["kind"] in ("importance", "correlation", "correlation_fixshare")]
    hs = np.array([r["host_share"] for r in synth])
    g = np.array([r["gain_vs_guest"] for r in synth])
    bh = np.array([r["beta_hat"] for r in synth])
    ok = ~np.isnan(bh)
    out["spearman_share_gain"] = float(spearmanr(hs, g).correlation)
    out["spearman_beta_gain"] = float(spearmanr(bh[ok], g[ok]).correlation)
    # 控制占比后 β 的作用：Ĝ 对 [1, 占比, 占比², β̂] 回归
    A = np.column_stack([np.ones(ok.sum()), hs[ok], hs[ok] ** 2, bh[ok]])
    coef, *_ = np.linalg.lstsq(A, g[ok], rcond=None)
    out["ols_gain_on_share_share2_beta"] = coef.tolist()
    print(f"Spearman(香港占比, Ĝ) = {out['spearman_share_gain']:+.3f}｜Spearman(β̂, Ĝ) = {out['spearman_beta_gain']:+.3f}"
          f"｜OLS Ĝ ~ 占比 + 占比² + β̂：β̂ 系数 {coef[3]:+.4f}")
    fs = [r for r in rows if r["kind"] == "correlation_fixshare"]
    if fs:
        bb = np.array([r["beta_hat"] for r in fs])
        gg = np.array([r["gain_vs_guest"] for r in fs])
        out["fixshare_spearman_beta_gain"] = float(spearmanr(bb, gg).correlation)
        out["fixshare_share_range"] = [float(min(r["host_share"] for r in fs)), float(max(r["host_share"] for r in fs))]
        print(f"控占比组：香港占比 {out['fixshare_share_range'][0]:.2f}–{out['fixshare_share_range'][1]:.2f}，"
              f"Spearman(β̂, Ĝ) = {out['fixshare_spearman_beta_gain']:+.3f}")
    # 按香港占比分箱
    bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001]
    out["by_share"] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (hs >= lo) & (hs < hi)
        if m.any():
            sig = np.mean([r["gain_vs_guest_ci"][0] > 0 for r, k in zip(synth, m) if k])
            out["by_share"].append({"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()), "gain_mean": float(g[m].mean()),
                                    "gain_min": float(g[m].min()), "gain_max": float(g[m].max()), "sig_frac": float(sig)})
            print(f"  香港占比 [{lo:.1f},{min(hi, 1):.1f}) n={int(m.sum()):2d} Ĝ均值 {g[m].mean():+.4f}"
                  f" [{g[m].min():+.4f}, {g[m].max():+.4f}] 显著 {sig:.0%}")
    (REPO_ROOT / "data/processed/vertibench_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
