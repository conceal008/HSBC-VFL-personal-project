"""
实验 3 真实隐语 SGB 复核（Exp 3 SGB Confirm）：
在真实 SecretFlow 1.14.0b0 SGB (HEU OU + M3, 33桶, prior base_score) 上复核 M1x_prior_feature：
1. IEEE-CIS 真实非对齐交易先验 + 2方 SGB
2. Home Credit 20% 低重合池真实非对齐先验 + 2方 SGB
必须在 /Applications/总文件夹/汇丰/8-9月/隐语_SecretFlow/.venv/bin/python 下运行。
"""
from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sgb_utils import run_sgb_fit_predict

STAGE3_ROOT = "/Applications/总文件夹/汇丰/第三阶段"
CACHE_DIR = os.path.join(STAGE3_ROOT, "results", "_cache")
OUT_PATH = os.path.join(STAGE3_ROOT, "results", "exp3_m1x_sgb_real.json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [Exp3-SGB] {msg}", flush=True)


def paired_test_boot_single(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int = 1000, seed: int = 2026):
    rng = np.random.default_rng(seed)
    n = len(y)
    d0 = float(roc_auc_score(y, p_b) - roc_auc_score(y, p_a))
    ds = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        ds[b] = d0 if yb.min() == yb.max() else (roc_auc_score(yb, p_b[idx]) - roc_auc_score(yb, p_a[idx]))
    p_le = (int(np.sum(ds <= 0)) + 1) / (n_boot + 1)
    p_ge = (int(np.sum(ds >= 0)) + 1) / (n_boot + 1)
    return {
        "delta_auc": d0,
        "ci95_low": float(np.percentile(ds, 2.5)),
        "ci95_high": float(np.percentile(ds, 97.5)),
        "p_test_boot": float(min(1.0, 2.0 * min(p_le, p_ge))),
    }


def main() -> None:
    t0 = time.perf_counter()
    out = {
        "experiment": "Exp3_M1x_Real_SecretFlow_SGB_Confirm",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": "SecretFlow 1.14.0b0 SGB (HEU OU 2048-bit + M3, FM64, sketch_eps=0.03, prior base_score)",
    }

    log("=== 1/2 IEEE-CIS M1x Real SGB ===")
    ie = np.load(os.path.join(CACHE_DIR, "exp3_sgb_ieee.npz"))
    p_i0, i_i0 = run_sgb_fit_predict(
        ["neidi_tx", "hk_id"], [ie["tx_tr"], ie["hk_tr"]], ie["y_tr"],
        [ie["tx_te"], ie["hk_te"]], ie["y_te"],
        num_boost_round=40, max_depth=4, learning_rate=0.1, reg_lambda=5.0,
    )
    p_i1, i_i1 = run_sgb_fit_predict(
        ["neidi_tx", "hk_id"],
        [np.hstack([ie["tx_tr"], ie["prior_tr"]]), ie["hk_tr"]], ie["y_tr"],
        [np.hstack([ie["tx_te"], ie["prior_te"]]), ie["hk_te"]], ie["y_te"],
        num_boost_round=40, max_depth=4, learning_rate=0.1, reg_lambda=5.0,
    )
    log(f"  IEEE SGB F0={i_i0['metrics']['auc']:.4f} ({i_i0['fit_sec']}s) -> M1x_prior={i_i1['metrics']['auc']:.4f} ({i_i1['fit_sec']}s)")
    out["ieee_sgb"] = {
        "arms": {"F0_aligned": i_i0, "M1x_prior_feature": i_i1},
        "contrast": paired_test_boot_single(ie["y_te"], p_i0, p_i1),
    }

    log("=== 2/2 Home Credit 20% Overlap M1x Real SGB ===")
    hc = np.load(os.path.join(CACHE_DIR, "exp3_sgb_hc20.npz"))
    p_h0, i_h0 = run_sgb_fit_predict(
        ["neidi", "hk"], [hc["neidi_tr"], hc["hk_tr"]], hc["y_tr"],
        [hc["neidi_te"], hc["hk_te"]], hc["y_te"],
        num_boost_round=40, max_depth=4, learning_rate=0.1, reg_lambda=5.0,
    )
    p_h1, i_h1 = run_sgb_fit_predict(
        ["neidi", "hk"],
        [np.hstack([hc["neidi_tr"], hc["prior_tr"]]), hc["hk_tr"]], hc["y_tr"],
        [np.hstack([hc["neidi_te"], hc["prior_te"]]), hc["hk_te"]], hc["y_te"],
        num_boost_round=40, max_depth=4, learning_rate=0.1, reg_lambda=5.0,
    )
    log(f"  HC20 SGB F0={i_h0['metrics']['auc']:.4f} ({i_h0['fit_sec']}s) -> M1x_prior={i_h1['metrics']['auc']:.4f} ({i_h1['fit_sec']}s)")
    out["home_credit_20pct_sgb"] = {
        "arms": {"F0_aligned": i_h0, "M1x_prior_feature": i_h1},
        "contrast": paired_test_boot_single(hc["y_te"], p_h0, p_h1),
    }

    out["total_elapsed_sec"] = round(time.perf_counter() - t0, 2)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"Exp 3 SGB Confirm finished in {out['total_elapsed_sec']:.1f}s -> {OUT_PATH}")


if __name__ == "__main__":
    main()
