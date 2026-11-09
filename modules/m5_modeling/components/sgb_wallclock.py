"""M5 · S5.M10：主动方明文直方图在两进程生产模式下的墙钟（DR-M5-016 · D-7）。

- 两个本地进程各自一方（内地 = 主动方，香港 = 被动方），`sf.init(cluster_config, ray_mode=False)`，本机回环通信；
  每个进程只读自己那一方的特征（标签只在内地进程）。
- 明文直方图替换（S5.M7 的 `plaintext_label_holder`）在两个进程里同样开启，以保证两方执行同一程序；被动方收到的消息不变。
- 每个任务：优化版联邦配置（S5.M1 / S5.M5 选定）只训前 n 棵；先做一次 2 棵树的预热（不记录），再按原版、明文、明文、原版各一次；
  记录两方各自的训练墙钟、启动到结束的总墙钟、回环网卡发送字节数；四次的测试打分须两两逐位相同。
- 口径：容器 2 核、本机回环，不代表跨境网络或汇丰环境；只作开销的相对比较。

需要 SecretFlow（secretflow-lite 1.13.0b0）；导入本模块不依赖它，子进程里才导入。打分只由内地进程写到私有目录，只用于逐位比较。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from modules.m5_modeling.components import g3_ssxgb as G3
from modules.m5_modeling.components import method_screen as MS
from modules.m5_modeling.components import method_screen_sgb as SG
from modules.m5_modeling.components import sgb_cost_screen as SC

PARTIES = (SG.LABEL_PARTY, SG.PASSIVE_PARTY)
KINDS = ("base", "plain")
WARMUP_TREES = SC.WARMUP_TREES


def cluster_config(party: str, ports: Sequence[int]) -> Dict[str, Any]:
    return {"parties": {p: {"address": "%s:%d" % (G3.LOCALHOST, port)} for p, port in zip(PARTIES, ports)},
            "self_party": party}


def prepare(P: MS.Params, sc: "MS.Screen", res: Dict, method: str, out: Path) -> Dict[str, Any]:
    """把一个任务的 final 视图按方拆成私有输入文件，并写出 SGB 参数（配置、分桶、初始分数取自结果 JSON）。"""
    fv = sc.v["final"]
    cfg, bins, kind, mkind = SG.method_spec(P, res, method)
    assert mkind is None, "S5.M10 只用不带逐样本先验的方法"
    base, _ = SG.base_of(P, fv["fit_y"], kind)
    data = out / ("%s__inputs.npz" % sc.name)
    np.savez(data, xg=fv["fit_xg"], xh=fv["fit_xh"], y=fv["fit_y"], eg=fv["ev_xg"], eh=fv["ev_xh"])
    return {"data": str(data), "cfg": dict(cfg), "bins": int(bins), "base": float(base),
            "cols_guest": int(fv["fit_xg"].shape[1]), "cols_host": int(fv["fit_xh"].shape[1]), "rows": int(len(fv["fit_y"]))}


def save_pred(pred, path: str) -> int:
    """内地进程里把测试打分写到私有目录（只用于逐位比较）。"""
    np.save(path, np.asarray(pred, np.float64).reshape(-1))
    return 1


def run_party(party: str, spec: Dict[str, Any], plain: bool, rounds: int, out: Path, tag: str,
              ports: Sequence[int]) -> None:
    """一方进程：训练前 rounds 棵并对测试集打分；打分只由内地写出。"""
    import secretflow as sf
    import spu
    from secretflow.data.ndarray import FedNdarray, PartitionWay
    from secretflow.device import wait
    from secretflow.ml.boost.sgb_v import Sgb

    P = MS.Params(spec["base_config"])
    sf.init(address="local", cluster_config=cluster_config(party, ports), ray_mode=False, logging_level="warning")
    neidi, hk = sf.PYU(SG.LABEL_PARTY), sf.PYU(SG.PASSIVE_PARTY)
    heu_cfg = sf.utils.testing.heu_config(SG.LABEL_PARTY, [SG.PASSIVE_PARTY])
    heu_cfg["he_parameters"]["schema"] = SG.HE_SCHEMA
    heu = sf.HEU(heu_cfg, spu.FieldType.FM64)

    def load(key):
        return np.load(spec["data"])[key].astype(float)

    def fed(kg, kh):
        return FedNdarray(partitions={neidi: neidi(load)(kg), hk: hk(load)(kh)}, partition_way=PartitionWay.VERTICAL)

    x = fed("xg", "xh")
    y = FedNdarray(partitions={neidi: neidi(lambda: load("y").reshape(-1, 1))()}, partition_way=PartitionWay.VERTICAL)
    wait([x.partitions[neidi], x.partitions[hk], y.partitions[neidi]])
    cfg = {**spec["cfg"], "rounds": int(rounds)}
    with SC.plaintext_label_holder(plain):
        t0 = time.perf_counter()
        model = Sgb(heu=heu).train(SG.sgb_params(P, cfg, spec["bins"], spec["base"], P.tune_seed), x, y)
        fit_s = time.perf_counter() - t0
    pred = model.predict_with_trees(fed("eg", "eh"))
    path = str(out / ("%s__pred.npy" % tag))
    wait([neidi(save_pred)(pred, path)])
    (out / ("%s__%s.json" % (tag, party))).write_text(json.dumps({"fit_seconds": fit_s}), encoding="utf-8")
    sf.shutdown()


def launch(spec_path: Path, plain: bool, rounds: int, out: Path, tag: str, ports: Sequence[int],
           popen: Callable = subprocess.Popen, read_tx: Callable[[], int] = G3.read_loopback_tx) -> Dict[str, Any]:
    """起两个进程跑一次；返回两方训练墙钟、总墙钟与回环发送字节数。"""
    tx0, t0 = read_tx(), time.perf_counter()
    procs = []
    for p in PARTIES:
        log = open(out / ("%s__%s.log" % (tag, p)), "w", encoding="utf-8")
        cmd = [sys.executable, "-m", "modules.m5_modeling.components.sgb_wallclock", "party", "--party", p,
               "--spec", str(spec_path), "--plain", str(int(plain)), "--rounds", str(int(rounds)), "--out", str(out),
               "--tag", tag, "--ports", ",".join(map(str, ports))]
        procs.append((popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
    rc = [pr.wait() for pr, _ in procs]
    for _, log in procs:
        log.close()
    wall = time.perf_counter() - t0
    assert rc == [0, 0], "两方进程退出码 %s，见 %s__*.log" % (rc, tag)
    party_s = {p: json.loads((out / ("%s__%s.json" % (tag, p))).read_text(encoding="utf-8"))["fit_seconds"] for p in PARTIES}
    return {"tag": tag, "plain": bool(plain), "trees": int(rounds), "fit_seconds": party_s, "launch_wall_seconds": wall,
            "loopback_tx_bytes": read_tx() - tx0}


def timing(spec_path: Path, spec: Dict[str, Any], name: str, max_trees: int, out: Path, ports: Sequence[int],
           run: Callable = launch) -> Dict[str, Any]:
    """预热 1 次（原版 2 棵，不记录），再按 ORDER 各一次；四次打分须两两逐位相同。"""
    n = min(int(max_trees), int(spec["cfg"]["rounds"]))
    run(spec_path, False, min(WARMUP_TREES, n), out, "%s__warmup" % name, ports)
    runs = [run(spec_path, kind == "plain", n, out, "%s__%d_%s" % (name, i, kind), ports) for i, kind in enumerate(SC.ORDER)]
    preds = [np.load(out / ("%s__pred.npy" % r["tag"])) for r in runs]
    return {"task": name, "trees": n, "runs": runs, "preds_identical": all(np.array_equal(preds[0], p) for p in preds[1:]),
            **summarize(runs, n, spec)}


def summarize(runs: List[Dict], n: int, spec: Dict[str, Any]) -> Dict[str, Any]:
    """每棵树墙钟（以内地训练墙钟计；两方并行，两方的训练墙钟只差启动先后）、节省比例及两对 ABBA 的范围、回环字节。"""
    def per_tree(kind, key=SG.LABEL_PARTY):
        return [r["fit_seconds"][key] / n for r in runs if r["plain"] == (kind == "plain")]
    base, plain = per_tree("base"), per_tree("plain")
    pairs = [1.0 - p / b for b, p in zip(base, reversed(plain))]
    tx = {k: float(np.mean([r["loopback_tx_bytes"] for r in runs if r["plain"] == (k == "plain")])) for k in KINDS}
    return {"cols_guest": spec["cols_guest"], "cols_host": spec["cols_host"], "rows": spec["rows"],
            "wall_per_tree_base": float(np.mean(base)), "wall_per_tree_plain": float(np.mean(plain)),
            "saving": 1.0 - float(np.mean(plain)) / float(np.mean(base)), "saving_range": [min(pairs), max(pairs)],
            "hk_wall_per_tree_base": float(np.mean(per_tree("base", SG.PASSIVE_PARTY))),
            "hk_wall_per_tree_plain": float(np.mean(per_tree("plain", SG.PASSIVE_PARTY))),
            "loopback_tx_bytes_base": tx["base"], "loopback_tx_bytes_plain": tx["plain"]}


def main(argv: Optional[List[str]] = None) -> int:
    """python -m modules.m5_modeling.components.sgb_wallclock run [--config ...]；party 子命令由 launch 调用。"""
    import argparse
    import os

    import yaml
    ap = argparse.ArgumentParser(description="S5.M10 两进程生产模式下主动方明文直方图的墙钟")
    ap.add_argument("mode", choices=["run", "party"])
    ap.add_argument("--config", default="modules/m5_modeling/configs/s5_m10_wallclock.yaml")
    ap.add_argument("--party")
    ap.add_argument("--spec")
    ap.add_argument("--plain", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--tag")
    ap.add_argument("--ports")
    a = ap.parse_args(argv)
    if a.mode == "party":
        spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
        run_party(a.party, spec, bool(a.plain), a.rounds, Path(a.out), a.tag, [int(p) for p in a.ports.split(",")])
        return 0
    m10 = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(m10["base_config"]).read_text(encoding="utf-8"))
    P = MS.Params(cfg)
    priv, run_dir = Path(os.environ[cfg["private_root_env"]]), Path(os.environ[cfg["run_dir_env"]])
    large_dir, out = (Path(os.environ[m10[k]]) for k in ("large_dir_env", "out_dir_env"))
    out.mkdir(parents=True, exist_ok=True)
    comp = Path(__file__)
    fp = SC.fingerprint([comp, Path(a.config), Path(m10["decision_record"])] +
                        [comp.with_name(n) for n in ("sgb_cost_screen.py", "method_screen_sgb.py", "method_screen.py",
                                                     "g3_ssxgb.py")])
    with open(out / "fingerprint_launch.txt", "a", encoding="utf-8") as fh:
        fh.write("== %s ==\n%s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), fp))
    ports = [int(p) for p in m10["fed_ports"]]
    for task in m10["tasks"]:
        name = task["name"]
        path = out / ("timing__%s.json" % name)
        if path.exists():
            continue
        res = json.loads(((large_dir if task.get("large") else run_dir) / (name + ".json")).read_text(encoding="utf-8"))
        sc = MS.Screen(name, cfg, priv, out / "unused")
        spec = {**prepare(P, sc, res, m10["method"], out), "base_config": cfg}
        del sc
        spec_path = out / ("%s__spec.json" % name)
        spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        info = timing(spec_path, spec, name, int(task.get("trees", m10["timing_trees"])), out, ports)
        Path(spec["data"]).unlink()                                   # 私有输入副本用完即删
        SC.write_json(path, info)
        MS.log("%s · 两进程墙钟：每棵树 原版 %.1fs / 明文 %.1fs，节省 %s，打分一致 %s" % (
            name, info["wall_per_tree_base"], info["wall_per_tree_plain"], format(info["saving"], ".0%"),
            info["preds_identical"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
