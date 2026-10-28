"""M5 · G3 臂 A_secure：隐语 SS-XGB（训练全程秘密分享）两方训练，可分段续训。

- 两个本机进程各代表一方：内地（neidi，持标签）与香港（hk）。隐语生产模式（不用 Ray），两方经本机回环通信；
  SPU 为两方 SEMI2K、FM128。隐语 ``debug_mode`` 单进程无法建立 SPU 两方链路（建链时阻塞等待对方），
  所以不能沿用臂 A 的单进程模拟（DR-M5-010 · D-1）。
- 分段续训：每训完一棵树，各方把自己的叶子权重份额（SPU dump）与自己一侧的分裂信息写入各自目录，再写进度文件。
  续训时载入已训练的树，并按训练顺序逐棵回放到训练集预测份额上。隐语原生的 checkpoint 续训不做这一步，
  续训后的梯度会按初始分数计算，故此处自行回放（DR-M5-010 · D-2）。
- 分段边界由启动方按进度一次算定，两方拿到同一个 stop_at，保证在同一棵树后停下。
- 打分只揭示给内地（to_pyu=neidi），由内地写入私有目录。
- 需要 SecretFlow（secretflow-lite 1.13.0b0，Python 3.10）；导入本模块不依赖它，调用时才导入。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

LABEL_PARTY = "neidi"
PASSIVE_PARTY = "hk"
PARTIES = (LABEL_PARTY, PASSIVE_PARTY)
LOCALHOST = "127.0.0.1"
PROGRESS_FILE = "progress.json"
LOG_DIR = "logs"
LOOPBACK = "lo"
LO_TX_BYTES_FIELD = 8          # /proc/net/dev 每行冒号后第 9 个字段为发送字节
KB_PER_MB = 1024.0             # Linux 的 ru_maxrss 单位为 KB
SECONDS_PER_HOUR = 3600.0
MODULE = "modules.m5_modeling.components.g3_ssxgb"
ROOT_VALUE_DECIMALS = 6


# ————————————————— 纯函数：配置、路径、进度与成本 —————————————————

def cluster_config(party: str, fed_ports: Sequence[int]) -> Dict[str, Any]:
    """隐语生产模式的两方地址（本机回环）。"""
    return {"parties": {p: {"address": "%s:%d" % (LOCALHOST, port)} for p, port in zip(PARTIES, fed_ports)},
            "self_party": party}


def spu_cluster_def(spu_ports: Sequence[int], protocol: str, field: str) -> Dict[str, Any]:
    return {"nodes": [{"party": p, "address": "%s:%d" % (LOCALHOST, port)} for p, port in zip(PARTIES, spu_ports)],
            "runtime_config": {"protocol": protocol, "field": field}}


def share_path(root: str, party: str, index: int) -> Path:
    return Path(root) / party / ("weight_%03d.pkl" % index)


def tree_path(root: str, party: str, index: int) -> Path:
    return Path(root) / party / ("tree_%03d.pkl" % index)


def empty_progress() -> Dict[str, Any]:
    return {"trees_done": 0, "tree_seconds": [], "tree_lo_tx_bytes": [], "setup_seconds": None,
            "chunks": [], "predict": []}


def load_progress(root: str, party: str) -> Dict[str, Any]:
    path = Path(root) / party / PROGRESS_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else empty_progress()


def save_progress(root: str, party: str, progress: Dict[str, Any]) -> None:
    """先写临时文件再替换，进程中途被终止时不留半截进度。"""
    path = Path(root) / party / PROGRESS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def trees_done(root: str) -> int:
    """两方都已存下的树数（取较小者，另一方多存的一棵在续训时被覆盖）。"""
    return min(int(load_progress(root, p)["trees_done"]) for p in PARTIES)


def plan_stop(done: int, total: int, mean_tree_seconds: float, budget_seconds: float,
              overhead_seconds: float) -> int:
    """本段训到第几棵树为止：预算扣除续训开销后能容纳的整棵数，至少 1 棵，不超过总数。"""
    if mean_tree_seconds <= 0:
        raise ValueError("每棵树耗时须为正数")
    room = int((budget_seconds - overhead_seconds) // mean_tree_seconds)
    return min(total, done + max(room, 1))


def loopback_tx_bytes(proc_net_dev: str, iface: str = LOOPBACK) -> int:
    for line in proc_net_dev.splitlines():
        name, sep, rest = line.partition(":")
        if sep and name.strip() == iface:
            return int(rest.split()[LO_TX_BYTES_FIELD])
    raise ValueError("找不到网卡 %s" % iface)


def read_loopback_tx() -> int:
    return loopback_tx_bytes(Path("/proc/net/dev").read_text(encoding="utf-8"))


def cost_summary(progress: Dict[str, Any], total_trees: int, cap_hours: float) -> Dict[str, Any]:
    """拟合墙钟 = 首段全局准备 + 各棵树训练耗时；分段续训的进程启动、准备与回放另计为续训开销。"""
    tree_s = [float(s) for s in progress["tree_seconds"]]
    chunks = progress["chunks"]
    fit = float(progress["setup_seconds"] or 0.0) + sum(tree_s)
    resume = sum(float(c["setup_seconds"]) + float(c["replay_seconds"]) for c in chunks[1:])
    return {"trees": len(tree_s), "fit_seconds": fit, "mean_tree_seconds": fit / len(tree_s) if tree_s else None,
            "resume_overhead_seconds": resume, "chunks": len(chunks),
            "lo_tx_bytes": int(sum(progress["tree_lo_tx_bytes"])),
            "peak_rss_mb": max((float(c["peak_rss_mb"]) for c in chunks), default=None),
            "within_cap": len(tree_s) == total_trees and fit <= cap_hours * SECONDS_PER_HOUR}


def root_splits(root: str, n_trees: int) -> List[Optional[Tuple[str, int, float]]]:
    """每棵树根节点由哪一方、按哪个特征与分裂值切分；只用于小样本续训核对。"""
    out: List[Optional[Tuple[str, int, float]]] = []
    for i in range(n_trees):
        split = None
        for party in PARTIES:
            t = pickle.loads(tree_path(root, party, i).read_bytes())
            if t.split_features and t.split_features[0] >= 0:
                split = (party, int(t.split_features[0]), round(float(t.split_values[0]), ROOT_VALUE_DECIMALS))
        out.append(split)
    return out


# ————————————————— 各方进程（需要 SecretFlow） —————————————————

def _load_part(path: str, key: str, n_rows: Optional[int]):
    import numpy as np
    arr = np.load(path)[key]
    return (arr if n_rows is None else arr[:n_rows]).astype(float)


def _dump_obj(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(pickle.dumps(obj))


def _load_obj(path: str) -> Any:
    return pickle.loads(Path(path).read_bytes())


def _save_scores(pred: Any, path: str) -> None:
    import numpy as np
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(pred, dtype=float).reshape(-1))


def resumable_class(xgb_cls: type, model_cls: type, wait: Callable) -> type:
    """在隐语 Xgb 上加「载入已训练的树、回放预测、训到 stop_at」；单棵树的训练逻辑完全沿用隐语。"""

    class ResumableXgb(xgb_cls):  # type: ignore[valid-type, misc]
        def train_until(self, params: Dict, dtrain: Any, label: Any, model: Any, stop_at: int, on_tree: Callable,
                        on_tree_start: Callable = lambda: None, on_ready: Callable = lambda setup, replay: None,
                        replay: bool = True) -> Any:
            t0 = time.perf_counter()
            self._prepare(dict(params), dtrain, label)
            self._global_setup()
            self.setup_seconds = time.perf_counter() - t0
            self.model = model if model is not None else model_cls(self.spu[0], self.obj, self.base)
            t1 = time.perf_counter()
            done = list(zip(self.model.trees, self.model.weights)) if replay else []
            for tree, weight in done:
                self._update_pred(tree, weight)
            wait(self.pred)
            self.replay_seconds = time.perf_counter() - t1
            on_ready(self.setup_seconds, self.replay_seconds)
            while len(self.model.trees) < min(stop_at, self.trees):
                on_tree_start()
                t2 = time.perf_counter()
                self._tree_setup()
                tree, weight = self._train_tree()
                self.model.trees.append(tree)
                self.model.weights.append(weight)
                if len(self.model.trees) < self.trees:
                    self._update_pred(tree, weight)
                    wait(self.pred)
                else:
                    wait(list(tree.values()) + [weight])
                on_tree(len(self.model.trees) - 1, tree, weight, time.perf_counter() - t2)
            return self.model

    return ResumableXgb


def run_party(party: str, mode: str, arrays: str, prefix: str, ckpt: str, params: Dict, resume_from: int,
              stop_at: int, fed_ports: Sequence[int], spu_ports: Sequence[int], protocol: str, field: str,
              n_rows: Optional[int] = None, out: Optional[str] = None, replay: bool = True) -> None:
    """一方进程：mode=train 训练到 stop_at 并逐棵存档；mode=predict 载入全部树对 prefix 集打分，分数只给内地。

    replay=False 复现隐语原生续训（不回放已训练的树），只用于续训核对的对照。
    """
    import secretflow as sf
    from secretflow.data.ndarray import FedNdarray, PartitionWay
    from secretflow.device import wait
    from secretflow.ml.boost.ss_xgb_v import Xgb, XgbModel
    from secretflow.ml.boost.ss_xgb_v.core.node_split import RegType

    sf.init(address="local", cluster_config=cluster_config(party, fed_ports), ray_mode=False,
            logging_level="warning")
    dev = {p: sf.PYU(p) for p in PARTIES}
    spu_dev = sf.SPU(spu_cluster_def(spu_ports, protocol, field))
    x = FedNdarray(partitions={dev[LABEL_PARTY]: dev[LABEL_PARTY](_load_part)(arrays, prefix + "__xg", n_rows),
                               dev[PASSIVE_PARTY]: dev[PASSIVE_PARTY](_load_part)(arrays, prefix + "__xh", n_rows)},
                   partition_way=PartitionWay.VERTICAL)
    model = None
    if resume_from:
        model = XgbModel(spu_dev, RegType(params["objective"]), float(params.get("base_score", 0.0)))
        model.weights = [spu_dev.load([str(share_path(ckpt, p, i)) for p in PARTIES]) for i in range(resume_from)]
        model.trees = [{dev[p]: dev[p](_load_obj)(str(tree_path(ckpt, p, i))) for p in PARTIES}
                       for i in range(resume_from)]
    progress = load_progress(ckpt, party)
    if mode == "train":
        y = FedNdarray(partitions={dev[LABEL_PARTY]: dev[LABEL_PARTY](
            lambda: _load_part(arrays, prefix + "__y", n_rows).reshape(-1, 1))()},
            partition_way=PartitionWay.VERTICAL)
        progress["trees_done"] = resume_from
        for key in ("tree_seconds", "tree_lo_tx_bytes"):
            progress[key] = progress[key][:resume_from]
        start_tx = [0]

        def on_tree_start() -> None:
            start_tx[0] = read_loopback_tx()

        chunk: Dict[str, Any] = {"start": resume_from, "end": resume_from}

        def peak_mb() -> float:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / KB_PER_MB

        def on_ready(setup: float, replay_s: float) -> None:
            # 准备与回放一结束就记下本段，进程中途被终止时记录不丢
            if resume_from == 0:
                progress["setup_seconds"] = setup
            chunk.update(setup_seconds=setup, replay_seconds=replay_s, peak_rss_mb=peak_mb())
            progress["chunks"].append(chunk)
            save_progress(ckpt, party, progress)

        def on_tree(i: int, tree: Dict, weight: Any, seconds: float) -> None:
            tx = read_loopback_tx() - start_tx[0]
            spu_dev.dump(weight, [str(share_path(ckpt, p, i)) for p in PARTIES])
            wait([pyu(_dump_obj)(obj, str(tree_path(ckpt, pyu.party, i))) for pyu, obj in tree.items()])
            progress["tree_seconds"].append(seconds)
            progress["tree_lo_tx_bytes"].append(tx)
            progress["trees_done"] = i + 1
            chunk.update(end=i + 1, peak_rss_mb=peak_mb())
            save_progress(ckpt, party, progress)

        booster = resumable_class(Xgb, XgbModel, wait)(spu_dev)
        booster.train_until(params, x, y, model, stop_at, on_tree, on_tree_start, on_ready, replay)
    else:
        if model is None:
            raise ValueError("打分须先载入已训练的树（resume_from > 0）")
        t0 = time.perf_counter()
        pred = model.predict(x, to_pyu=dev[LABEL_PARTY])
        wait([dev[LABEL_PARTY](_save_scores)(pred.partitions[dev[LABEL_PARTY]], out)])
        progress["predict"].append({"prefix": prefix, "trees": resume_from, "seconds": time.perf_counter() - t0})
    save_progress(ckpt, party, progress)
    sf.shutdown()


# ————————————————— 启动两方进程 —————————————————

def party_command(party: str, mode: str, cfg: Dict[str, Any], resume_from: int, stop_at: int,
                  out: Optional[str] = None) -> List[str]:
    cmd = [cfg.get("python", sys.executable), "-m", MODULE, "--party", party, "--mode", mode,
           "--arrays", cfg["arrays"], "--prefix", cfg["prefix"], "--ckpt", cfg["ckpt"],
           "--params", json.dumps(cfg["params"]), "--resume-from", str(resume_from), "--stop-at", str(stop_at),
           "--fed-ports", ",".join(map(str, cfg["fed_ports"])), "--spu-ports", ",".join(map(str, cfg["spu_ports"])),
           "--protocol", cfg["protocol"], "--field", cfg["field"]]
    if cfg.get("n_rows") is not None:
        cmd += ["--n-rows", str(cfg["n_rows"])]
    if cfg.get("replay") is False:
        cmd += ["--no-replay"]
    return cmd + (["--out", out] if out else [])


def launch(mode: str, cfg: Dict[str, Any], resume_from: int, stop_at: int, out: Optional[str] = None,
           popen: Callable = subprocess.Popen) -> Dict[str, int]:
    """同时启动两方进程并等待结束；日志写入检查点目录的 logs/。返回各方退出码。"""
    log_dir = Path(cfg["ckpt"]) / LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = {}
    for party in reversed(PARTIES):
        log = open(log_dir / ("%s_%s_%03d.log" % (party, mode, resume_from)), "w", encoding="utf-8")
        procs[party] = (popen(party_command(party, mode, cfg, resume_from, stop_at, out), stdout=log,
                              stderr=subprocess.STDOUT, cwd=cfg["repo"]), log)
    codes = {}
    for party, (proc, log) in procs.items():
        codes[party] = proc.wait()
        log.close()
    return codes


def cap_room(progress: Dict[str, Any], done: int, mean_tree_seconds: float, cap_hours: float) -> int:
    """按预注册单次拟合上限，还能再开始几棵树（不开始一棵按均值会越过上限的树）。"""
    used = float(progress["setup_seconds"] or 0.0) + sum(float(s) for s in progress["tree_seconds"][:done])
    return max(int((cap_hours * SECONDS_PER_HOUR - used) // mean_tree_seconds), 0)


def run_chunk(cfg: Dict[str, Any], popen: Callable = subprocess.Popen) -> Dict[str, Any]:
    """训练一段：按两方进度、分段预算与单次拟合上限算出 stop_at，再启动两方；已到上限则不启动。"""
    done = trees_done(cfg["ckpt"])
    total = int(cfg["params"]["num_boost_round"])
    progress = load_progress(cfg["ckpt"], LABEL_PARTY)
    seen = progress["tree_seconds"][:done]
    mean = sum(seen) / len(seen) if seen else float(cfg["tree_seconds_estimate"])
    overhead = float(cfg["overhead_seconds_estimate"]) + done * float(cfg["replay_seconds_per_tree_estimate"])
    room = cap_room(progress, done, mean, float(cfg["cap_hours"]))
    stop = min(plan_stop(done, total, mean, float(cfg["chunk_budget_seconds"]), overhead), done + room)
    if stop <= done:
        return {"resume_from": done, "stop_at": done, "exit_codes": {}, "trees_done": done, "cap_reached": True}
    codes = launch("train", cfg, done, stop, popen=popen)
    return {"resume_from": done, "stop_at": stop, "exit_codes": codes, "trees_done": trees_done(cfg["ckpt"]),
            "cap_reached": False}


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("--party", "--mode", "--arrays", "--prefix", "--ckpt", "--params", "--fed-ports", "--spu-ports",
                 "--protocol", "--field"):
        ap.add_argument(name, required=True)
    ap.add_argument("--resume-from", type=int, default=0)
    ap.add_argument("--stop-at", type=int, required=True)
    ap.add_argument("--n-rows", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-replay", action="store_true")
    a = ap.parse_args(argv)
    ports = [[int(p) for p in s.split(",")] for s in (a.fed_ports, a.spu_ports)]
    run_party(a.party, a.mode, a.arrays, a.prefix, a.ckpt, json.loads(a.params), a.resume_from, a.stop_at,
              ports[0], ports[1], a.protocol, a.field, a.n_rows, a.out, not a.no_replay)


if __name__ == "__main__":
    main()
