"""[切分] 把 vertibench_scan 里的某个切分落成 tag 三件套（列值与源 tag 逐位相同，只改列归属），供 SGB 联邦复核。

    python src/ext/make_split_tag.py --scan data/processed/vertibench_scan.json --pick kind=correlation,beta=0,seed=0 --out-tag vb_c0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "prepared"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="data/processed/vertibench_scan.json")
    ap.add_argument("--pick", required=True, help="逗号分隔的 key=value，匹配 scan rows 的唯一一行")
    ap.add_argument("--out-tag", required=True)
    args = ap.parse_args(argv)
    scan = json.loads((REPO_ROOT / args.scan).read_text(encoding="utf-8"))
    cond = dict(kv.split("=", 1) for kv in args.pick.split(","))
    hits = [r for r in scan["rows"] if all(str(r.get(k)) == v or (k in r and _num_eq(r[k], v)) for k, v in cond.items())]
    assert len(hits) == 1, f"匹配到 {len(hits)} 行"
    r = hits[0]
    src = scan["tag"]
    g = pd.read_csv(DATA_DIR / f"{src}_guest.csv")
    h = pd.read_csv(DATA_DIR / f"{src}_host.csv")
    idx = pd.read_csv(DATA_DIR / f"{src}_index.csv")
    allx = pd.concat([g, h.drop(columns="ncodpers")], axis=1)
    gc = r["guest_cols"]
    hc = [c for c in allx.columns[1:] if c not in set(gc)]
    allx[["ncodpers", *gc]].to_csv(DATA_DIR / f"{args.out_tag}_guest.csv", index=False)
    allx[["ncodpers", *hc]].to_csv(DATA_DIR / f"{args.out_tag}_host.csv", index=False)
    idx.to_csv(DATA_DIR / f"{args.out_tag}_index.csv", index=False)
    meta = {"split_tag": args.out_tag, "source_tag": src, "pick": cond, "guest_features": gc, "host_features": hc,
            "n_guest_features": len(gc), "n_host_features": len(hc), "scan_row": {k: v for k, v in r.items() if k != "guest_cols"}}
    (DATA_DIR / f"{args.out_tag}_columns.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{args.out_tag}] 内地 {len(gc)} 列 / 香港 {len(hc)} 列｜代理 Ĝ {r['gain_vs_guest']:+.4f}｜香港占比 {r['host_share']:.2f}｜β̂ {r['beta_hat']:.2f}")
    return 0


def _num_eq(a, b) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
