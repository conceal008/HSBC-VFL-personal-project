"""
实验 1 真实隐语 SGB 复核（Exp 1 SGB Confirm）：
在真实 SecretFlow 1.14.0b0 SGB (HEU OU 2048-bit, FM64, 33桶 sketch_eps=0.03, base_score=prior) 上实跑：
1. Home Credit：2方(F0) vs 3方征信(F3_bureau) vs 新3方支付履约(F3_pay_new) vs 4方全量(F4_all)
   + 4方首次前2棵树「原版密文 vs M3标签方明文直方图」逐位一致性核对！
2. Santander【构造三方】：2方(F0) vs 3方(F3)
3. UCI Bank（去 duration）【构造三方】：2方(F0) vs 3方(F3)
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
OUT_PATH = os.path.join(STAGE3_ROOT, "results", "exp1_x2_sgb_real.json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [Exp1-SGB] {msg}", flush=True)


def paired_test_boot_single(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int = 1000, seed: int = 2026):
    rng = np.random.default_rng(seed)
    n = len(y)
    d0 = float(roc_auc_score(y, p_b) - roc_auc_score(y, p_a))
    ds = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            ds[b] = d0
        else:
            ds[b] = roc_auc_score(yb, p_b[idx]) - roc_auc_score(yb, p_a[idx])
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
        "experiment": "Exp1_X2_Real_SecretFlow_SGB_Confirm",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": "SecretFlow 1.14.0b0 SGB (HEU OU 2048-bit, FM64, sketch_eps=0.03, prior base_score)",
        "note": "单次拟合真实 SGB 复核（seed=11，40棵树，depth=4，lr=0.1，reg_lambda=5.0），区间为测试集配对抽样 95% CI（不含训练随机性）",
    }

    # --- 1. Home Credit 2/3/4-party ---
    log("=== Home Credit Real SGB (2-party / 3-party / 4-party) ===")
    hc = np.load(os.path.join(CACHE_DIR, "exp1_sgb_hc.npz"))
    y_tr, y_te = hc["y_tr"], hc["y_te"]

    # First: 4-party bit-exact check (first 2 trees: original encrypted vs M3 plaintext label holder)
    log("  4-party M3 bit-exact check (2 trees) ...")
    p_base2, info_base2 = run_sgb_fit_predict(
        ["neidi", "hk", "bureau", "pay"],
        [hc["neidi_tr"], hc["hk_tr"], hc["bur_tr"], hc["pay_tr"]],
        y_tr,
        [hc["neidi_te"], hc["hk_te"], hc["bur_te"], hc["pay_te"]],
        y_te,
        num_boost_round=2,
        max_depth=4,
        use_m3_plaintext_label=False,
    )
    p_m3_2, info_m3_2 = run_sgb_fit_predict(
        ["neidi", "hk", "bureau", "pay"],
        [hc["neidi_tr"], hc["hk_tr"], hc["bur_tr"], hc["pay_tr"]],
        y_tr,
        [hc["neidi_te"], hc["hk_te"], hc["bur_te"], hc["pay_te"]],
        y_te,
        num_boost_round=2,
        max_depth=4,
        use_m3_plaintext_label=True,
    )
    max_diff = float(np.max(np.abs(p_base2 - p_m3_2)))
    rows_diff = int(np.sum(p_base2 != p_m3_2))
    out["hc_4party_m3_check"] = {
        "trees": 2,
        "rows_different": rows_diff,
        "max_abs_diff": max_diff,
        "fit_sec_base": info_base2["fit_sec"],
        "fit_sec_m3": info_m3_2["fit_sec"],
        "speedup_ratio": round(info_base2["fit_sec"] / max(info_m3_2["fit_sec"], 1e-6), 3),
        "verdict": "逐位一致" if rows_diff == 0 else "存在微小浮点差",
    }
    log(f"  4-party M3 check: rows_diff={rows_diff}, max_diff={max_diff:.2e}, "
        f"base={info_base2['fit_sec']}s vs m3={info_m3_2['fit_sec']}s ({out['hc_4party_m3_check']['speedup_ratio']}x)")

    hc_arms = {
        "F0_2party": (["neidi", "hk"], [hc["neidi_tr"], hc["hk_tr"]], [hc["neidi_te"], hc["hk_te"]]),
        "F3_bureau": (["neidi", "hk", "bureau"], [hc["neidi_tr"], hc["hk_tr"], hc["bur_tr"]], [hc["neidi_te"], hc["hk_te"], hc["bur_te"]]),
        "F3_pay_new": (["neidi", "hk", "pay"], [hc["neidi_tr"], hc["hk_tr"], hc["pay_tr"]], [hc["neidi_te"], hc["hk_te"], hc["pay_te"]]),
        "F4_all_4party": (["neidi", "hk", "bureau", "pay"], [hc["neidi_tr"], hc["hk_tr"], hc["bur_tr"], hc["pay_tr"]], [hc["neidi_te"], hc["hk_te"], hc["bur_te"], hc["pay_te"]]),
    }
    hc_res, hc_preds = {}, {}
    for k, (pnames, xtr, xte) in hc_arms.items():
        p, info = run_sgb_fit_predict(
            pnames, xtr, y_tr, xte, y_te,
            num_boost_round=40, max_depth=4, learning_rate=0.1, reg_lambda=5.0,
            use_m3_plaintext_label=(rows_diff == 0),
        )
        hc_res[k] = info
        hc_preds[k] = p
        log(f"  HC SGB {k:14s}: AUC={info['metrics']['auc']:.4f} PR={info['metrics']['pr_auc']:.4f} fit={info['fit_sec']}s")

    out["home_credit_sgb"] = {
        "arms": hc_res,
        "contrasts": {
            "F3_bureau - F0_2party": paired_test_boot_single(y_te, hc_preds["F0_2party"], hc_preds["F3_bureau"]),
            "F3_pay_new - F0_2party": paired_test_boot_single(y_te, hc_preds["F0_2party"], hc_preds["F3_pay_new"]),
            "F4_all - F0_2party": paired_test_boot_single(y_te, hc_preds["F0_2party"], hc_preds["F4_all_4party"]),
            "F4_all - F3_bureau": paired_test_boot_single(y_te, hc_preds["F3_bureau"], hc_preds["F4_all_4party"]),
            "F4_all - F3_pay_new": paired_test_boot_single(y_te, hc_preds["F3_pay_new"], hc_preds["F4_all_4party"]),
        },
    }

    # --- 2. Santander 2-party vs 3-party ---
    log("=== Santander Real SGB (2-party vs 3-party) ===")
    san = np.load(os.path.join(CACHE_DIR, "exp1_sgb_santander.npz"))
    p_s2, i_s2 = run_sgb_fit_predict(
        ["neidi", "hk_dep"], [san["neidi_tr"], san["hk_tr"]], san["y_tr"],
        [san["neidi_te"], san["hk_te"]], san["y_te"],
        num_boost_round=40, max_depth=4, learning_rate=0.1, reg_lambda=5.0,
    )
    p_s3, i_s3 = run_sgb_fit_predict(
        ["neidi", "hk_dep", "third_cw"], [san["neidi_tr"], san["hk_tr"], san["third_tr"]], san["y_tr"],
        [san["neidi_te"], san["hk_te"], san["third_te"]], san["y_te"],
        num_boost_round=40, max_depth=4, learning_rate=0.1, reg_lambda=5.0,
    )
    log(f"  SAN SGB F0={i_s2['metrics']['auc']:.4f} ({i_s2['fit_sec']}s) -> F3={i_s3['metrics']['auc']:.4f} ({i_s3['fit_sec']}s)")
    out["santander_sgb"] = {
        "arms": {"F0_2party": i_s2, "F3_3party": i_s3},
        "contrasts": {"F3_3party - F0_2party": paired_test_boot_single(san["y_te"], p_s2, p_s3)},
    }

    # --- 3. UCI Bank 2-party vs 3-party ---
    log("=== UCI Bank Real SGB (2-party vs 3-party, no duration) ===")
    uci = np.load(os.path.join(CACHE_DIR, "exp1_sgb_uci.npz"))
    p_u2, i_u2 = run_sgb_fit_predict(
        ["client", "campaign"], [uci["neidi_tr"], uci["hk_tr"]], uci["y_tr"],
        [uci["neidi_te"], uci["hk_te"]], uci["y_te"],
        num_boost_round=40, max_depth=4, learning_rate=0.1, reg_lambda=5.0,
    )
    p_u3, i_u3 = run_sgb_fit_predict(
        ["client", "campaign", "macro"], [uci["neidi_tr"], uci["hk_tr"], uci["third_tr"]], uci["y_tr"],
        [uci["neidi_te"], uci["hk_te"], uci["third_te"]], uci["y_te"],
        num_boost_round=40, max_depth=4, learning_rate=0.1, reg_lambda=5.0,
    )
    log(f"  UCI SGB F0={i_u2['metrics']['auc']:.4f} ({i_u2['fit_sec']}s) -> F3={i_u3['metrics']['auc']:.4f} ({i_u3['fit_sec']}s)")
    out["uci_bank_sgb"] = {
        "arms": {"F0_2party": i_u2, "F3_3party": i_u3},
        "contrasts": {"F3_3party - F0_2party": paired_test_boot_single(uci["y_te"], p_u2, p_u3)},
    }

    out["total_elapsed_sec"] = round(time.perf_counter() - t0, 2)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"Exp 1 SGB Confirm finished in {out['total_elapsed_sec']:.1f}s -> {OUT_PATH}")


if __name__ == "__main__":
    main()
