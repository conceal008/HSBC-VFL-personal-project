"""
实验 2 真实隐语 SGB 复核（Exp 2 SGB Confirm）：
在真实 SecretFlow 1.14.0b0 SGB (HEU OU + M3, 33桶, prior base_score) 上复核 X1 / X1+ 在三个数据集上的真实联邦增量：
1. Home Credit：F3_pay_static vs F3_pay_plus_x1v（三方真实 SGB）
2. Santander：F0_2party_static vs F0_plus_x1v_both（两方真实 SGB）
3. IEEE-CIS Window 3：F0_2party vs F_x1_full（两方真实 SGB — 正式关闭 Phase 2 v0.4 缺少真实 SGB 复核的局限！）
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
OUT_PATH = os.path.join(STAGE3_ROOT, "results", "exp2_x1plus_sgb_real.json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [Exp2-SGB] {msg}", flush=True)


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
        "experiment": "Exp2_X1Plus_Real_SecretFlow_SGB_Confirm",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": "SecretFlow 1.14.0b0 SGB (HEU OU 2048-bit + M3, FM64, sketch_eps=0.03, prior base_score)",
        "note": "单次拟合真实 SGB 复核（seed=11，40棵树，depth=4，lr=0.1，reg_lambda=5.0），区间为测试集配对抽样 95% CI",
    }

    # 1. Home Credit X1+ (3-party SGB)
    log("=== 1/3 Home Credit X1+ Real SGB ===")
    hc = np.load(os.path.join(CACHE_DIR, "exp2_sgb_hc.npz"))
    p_h0, i_h0 = run_sgb_fit_predict(
        ["neidi", "hk", "pay"],
        [hc["neidi_tr"], hc["hk_tr"], hc["pay_tr"]], hc["y_tr"],
        [hc["neidi_te"], hc["hk_te"], hc["pay_te"]], hc["y_te"],
        num_boost_round=40, max_depth=4, learning_rate=0.1, reg_lambda=5.0,
    )
    p_h1, i_h1 = run_sgb_fit_predict(
        ["neidi", "hk", "pay"],
        [hc["neidi_tr"], hc["hk_tr"], np.hstack([hc["pay_tr"], hc["vel_tr"]])], hc["y_tr"],
        [hc["neidi_te"], hc["hk_te"], np.hstack([hc["pay_te"], hc["vel_te"]])], hc["y_te"],
        num_boost_round=40, max_depth=4, learning_rate=0.1, reg_lambda=5.0,
    )
    log(f"  HC SGB static={i_h0['metrics']['auc']:.4f} ({i_h0['fit_sec']}s) -> +x1v={i_h1['metrics']['auc']:.4f} ({i_h1['fit_sec']}s)")
    out["home_credit_sgb"] = {
        "arms": {"F3_pay_static": i_h0, "F3_pay_plus_x1v": i_h1},
        "contrast": paired_test_boot_single(hc["y_te"], p_h0, p_h1),
    }

    # 2. Santander X1+ (2-party SGB)
    log("=== 2/3 Santander X1+ Real SGB ===")
    san = np.load(os.path.join(CACHE_DIR, "exp2_sgb_santander.npz"))
    p_s0, i_s0 = run_sgb_fit_predict(
        ["neidi", "hk_dep"],
        [san["neidi_static_tr"], san["hk_static_tr"]], san["y_tr"],
        [san["neidi_static_te"], san["hk_static_te"]], san["y_te"],
        num_boost_round=40, max_depth=4, learning_rate=0.1, reg_lambda=5.0,
    )
    p_s1, i_s1 = run_sgb_fit_predict(
        ["neidi", "hk_dep"],
        [san["neidi_x1v_tr"], san["hk_x1v_tr"]], san["y_tr"],
        [san["neidi_x1v_te"], san["hk_x1v_te"]], san["y_te"],
        num_boost_round=40, max_depth=4, learning_rate=0.1, reg_lambda=5.0,
    )
    log(f"  SAN SGB static={i_s0['metrics']['auc']:.4f} ({i_s0['fit_sec']}s) -> +x1v={i_s1['metrics']['auc']:.4f} ({i_s1['fit_sec']}s)")
    out["santander_sgb"] = {
        "arms": {"F0_static": i_s0, "F0_plus_x1v_both": i_s1},
        "contrast": paired_test_boot_single(san["y_te"], p_s0, p_s1),
    }

    # 3. IEEE-CIS Window 3 X1+ full (2-party SGB)
    log("=== 3/3 IEEE-CIS Window 3 X1+ Full Real SGB ===")
    ie = np.load(os.path.join(CACHE_DIR, "exp2_sgb_ieee_w3.npz"))
    p_i0, i_i0 = run_sgb_fit_predict(
        ["neidi_tx", "hk_id"],
        [ie["tx_tr"], ie["hk_tr"]], ie["y_tr"],
        [ie["tx_te"], ie["hk_te"]], ie["y_te"],
        num_boost_round=40, max_depth=4, learning_rate=0.1, reg_lambda=5.0,
    )
    p_i1, i_i1 = run_sgb_fit_predict(
        ["neidi_tx", "hk_id"],
        [np.hstack([ie["tx_tr"], ie["gtx_tr"], ie["vtx_tr"]]), np.hstack([ie["hk_tr"], ie["ghk_tr"]])], ie["y_tr"],
        [np.hstack([ie["tx_te"], ie["gtx_te"], ie["vtx_te"]]), np.hstack([ie["hk_te"], ie["ghk_te"]])], ie["y_te"],
        num_boost_round=40, max_depth=4, learning_rate=0.1, reg_lambda=5.0,
    )
    log(f"  IEEE-W3 SGB F0={i_i0['metrics']['auc']:.4f} ({i_i0['fit_sec']}s) -> F_x1_full={i_i1['metrics']['auc']:.4f} ({i_i1['fit_sec']}s)")
    out["ieee_w3_sgb"] = {
        "arms": {"F0_2party": i_i0, "F_x1_full": i_i1},
        "contrast": paired_test_boot_single(ie["y_te"], p_i0, p_i1),
    }

    out["total_elapsed_sec"] = round(time.perf_counter() - t0, 2)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"Exp 2 SGB Confirm finished in {out['total_elapsed_sec']:.1f}s -> {OUT_PATH}")


if __name__ == "__main__":
    main()
