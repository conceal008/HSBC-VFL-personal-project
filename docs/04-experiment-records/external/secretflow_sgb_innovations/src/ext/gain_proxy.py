"""[优化③] 明文代理增益估计 —— 用秒级明文 XGBoost 预判联邦增益，联邦 SGB 只做最后确认。

依据：SGB（SecureBoost 类）只加密梯度，分裂搜索与明文直方图 GBDT 数学等价（Santander 复现：
OU 与 Paillier 两版预测逐位相同；联邦 0.8991 vs 明文集中式 0.9058）。因此
    代理增益 Ĝ = AUC(集中式) − max(AUC(内地单边), AUC(香港单边))
是联邦主判据 Δ = AUC(联邦) − max(两单边) 的上界近似；配对 CI 直接复用原版 train_local.py 的
客户级 cluster bootstrap（同一批重采样）。

本脚本只做调度与汇总：逐 tag 调原版 train_local.py（不改其口径），再把 JSON 汇总成一张表，
并对 random_seed* 给出「随机切分下的增益分布」——业务切分的增益若落在随机分布之内，
说明它不是业务切分特有的。

用法：
    python src/ext/gain_proxy.py --tags uci_business,uci_diagnostic --random uci_random_seed{42..46}
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "data" / "processed"


def run_one(tag: str, n_boot: int, force: bool) -> dict:
    js = OUT / f"train_local_xgb_all_{tag}_m42.json"
    if force or not js.exists():
        cmd = [sys.executable, "src/cross_sell/train_local.py", "--model", "xgb", "--scope", "all",
               "--tag", tag, "--n-boot", str(n_boot)]
        p = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(f"{tag} 失败：{p.stderr[-800:]}")
    m = json.loads(js.read_text(encoding="utf-8"))["metrics"]
    g, h, c = m["guest"]["auroc"], m["host"]["auroc"], m["central"]["auroc"]
    best = "guest" if g >= h else "host"
    d = m[f"delta_{best}_minus_central"]   # 单边 − 集中式 → 取负即 集中式 − max(单边)
    return {"tag": tag, "guest": g, "host": h, "central": c, "best_single": best,
            "gain_proxy": -d["delta"], "gain_ci_lo": -d["ci_hi"], "gain_ci_hi": -d["ci_lo"],
            "guest_minus_host": m["delta_guest_minus_host"]["delta"], "n_boot": n_boot}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    ap.add_argument("--random", default="", help="随机切分 tag 列表（逗号分隔）")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-boot-random", type=int, default=300)
    ap.add_argument("--name", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    rows = [run_one(t, args.n_boot, args.force) for t in args.tags.split(",") if t]
    rnd = [run_one(t, args.n_boot_random, args.force) for t in args.random.split(",") if t]
    print(f"{'tag':28s} {'内地':>7s} {'香港':>7s} {'集中式':>7s}  代理增益 Ĝ [95% CI]")
    for r in rows + rnd:
        print(f"{r['tag']:28s} {r['guest']:.4f} {r['host']:.4f} {r['central']:.4f}  "
              f"{r['gain_proxy']:+.4f} [{r['gain_ci_lo']:+.4f}, {r['gain_ci_hi']:+.4f}]")
    summ = {"name": args.name, "rows": rows, "random": rnd}
    if rnd:
        gs = sorted(r["gain_proxy"] for r in rnd)
        summ["random_gain_min"], summ["random_gain_max"] = gs[0], gs[-1]
        summ["random_gain_mean"] = sum(gs) / len(gs)
        print(f"随机切分（{len(gs)} 个）代理增益：均值 {summ['random_gain_mean']:+.4f}，范围 [{gs[0]:+.4f}, {gs[-1]:+.4f}]")
    (OUT / f"gain_proxy_{args.name}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
