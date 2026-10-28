"""M5 · G3 臂 A_secure（SS-XGB 分段续训）的单元测试。只用程序生成的合成数组；CI 没有 SecretFlow，用替身检验编排。"""
from __future__ import annotations

import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m5_modeling.components import g3_ssxgb as SX  # noqa: E402

PARAMS = {"num_boost_round": 3, "max_depth": 2, "learning_rate": 0.1, "reg_lambda": 1.0, "sketch_eps": 0.1,
          "objective": "logistic", "seed": 1}
PORTS = ([1, 2], [3, 4])


def test_两方地址与SPU配置():
    cc = SX.cluster_config("hk", PORTS[0])
    assert cc["self_party"] == "hk" and cc["parties"]["neidi"]["address"].endswith(":1")
    cd = SX.spu_cluster_def(PORTS[1], "SEMI2K", "FM128")
    assert [n["party"] for n in cd["nodes"]] == list(SX.PARTIES) and cd["runtime_config"]["field"] == "FM128"


def test_分段棵数():
    assert SX.plan_stop(0, 150, 140.0, 6600.0, 60.0) == 46
    assert SX.plan_stop(140, 150, 140.0, 6600.0, 60.0) == 150
    assert SX.plan_stop(5, 150, 140.0, 100.0, 60.0) == 6          # 预算不足一棵时也至少训一棵
    with pytest.raises(ValueError):
        SX.plan_stop(0, 10, 0.0, 100.0, 0.0)


def test_回环发送字节():
    text = ("Inter-| Receive | Transmit\n face |bytes packets\n"
            "    lo: 100 2 0 0 0 0 0 0 250 3 0 0 0 0 0 0\n  eth0: 1 1 0 0 0 0 0 0 9 1 0 0 0 0 0 0\n")
    assert SX.loopback_tx_bytes(text) == 250 and SX.loopback_tx_bytes(text, "eth0") == 9
    with pytest.raises(ValueError):
        SX.loopback_tx_bytes(text, "wlan0")


def test_进度存取与成本汇总(tmp_path):
    root = str(tmp_path)
    assert SX.trees_done(root) == 0
    p = SX.empty_progress()
    p.update(trees_done=3, tree_seconds=[10.0, 12.0, 11.0], tree_lo_tx_bytes=[5, 5, 6], setup_seconds=4.0,
             chunks=[{"setup_seconds": 4.0, "replay_seconds": 0.0, "peak_rss_mb": 100.0},
                     {"setup_seconds": 3.0, "replay_seconds": 2.0, "peak_rss_mb": 120.0}])
    SX.save_progress(root, "neidi", p)
    SX.save_progress(root, "hk", dict(p, trees_done=2))
    assert SX.trees_done(root) == 2 and SX.load_progress(root, "neidi") == p
    c = SX.cost_summary(p, 3, 1.0)
    assert c["fit_seconds"] == 37.0 and c["resume_overhead_seconds"] == 5.0 and c["lo_tx_bytes"] == 16
    assert c["peak_rss_mb"] == 120.0 and c["within_cap"] is True
    assert SX.cost_summary(p, 4, 1.0)["within_cap"] is False
    assert SX.cap_room(p, 3, 10.0, 0.01) == 0 and SX.cap_room(p, 3, 10.0, 1.0) == 356


def test_根节点分裂(tmp_path):
    for party, feats in (("neidi", [[2, -1], [-1]]), ("hk", [[-1, 0], [5]])):
        for i, f in enumerate(feats):
            path = SX.tree_path(str(tmp_path), party, i)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps(types.SimpleNamespace(split_features=f, split_values=[1.5] * len(f))))
    assert SX.root_splits(str(tmp_path), 2) == [("neidi", 2, 1.5), ("hk", 5, 1.5)]


class _Proc:
    def __init__(self, code):
        self.code = code

    def wait(self):
        return self.code


def _cfg(tmp_path, **kw):
    cfg = {"python": "py", "repo": str(tmp_path), "arrays": "a.npz", "prefix": "train", "ckpt": str(tmp_path / "ck"),
           "params": PARAMS, "fed_ports": PORTS[0], "spu_ports": PORTS[1], "protocol": "SEMI2K", "field": "FM128",
           "tree_seconds_estimate": 10.0, "overhead_seconds_estimate": 5.0, "replay_seconds_per_tree_estimate": 1.0,
           "chunk_budget_seconds": 27.0, "cap_hours": 1.0}
    cfg.update(kw)
    return cfg


def test_启动命令与分段(tmp_path):
    seen = []

    def popen(cmd, **kw):
        seen.append(cmd)
        return _Proc(0)

    cfg = _cfg(tmp_path, n_rows=50, replay=False)
    cmd = SX.party_command("neidi", "predict", cfg, 3, 3, out="o.npy")
    assert cmd[:3] == ["py", "-m", SX.MODULE] and "--no-replay" in cmd and cmd[-2:] == ["--out", "o.npy"]
    assert "--n-rows" in cmd and json.loads(cmd[cmd.index("--params") + 1]) == PARAMS
    r = SX.run_chunk(cfg, popen=popen)
    assert r["stop_at"] == 2 and r["exit_codes"] == {"hk": 0, "neidi": 0} and len(seen) == 2
    assert (tmp_path / "ck" / SX.LOG_DIR / "neidi_train_000.log").exists()
    p = SX.empty_progress()
    p.update(trees_done=2, tree_seconds=[1800.0, 1800.0], setup_seconds=1.0)
    for party in SX.PARTIES:
        SX.save_progress(cfg["ckpt"], party, p)
    assert SX.run_chunk(cfg, popen=popen)["cap_reached"] is True and len(seen) == 2


def test_命令行解析(monkeypatch):
    got = {}
    monkeypatch.setattr(SX, "run_party", lambda *a: got.setdefault("args", a))
    SX.main(["--party", "hk", "--mode", "train", "--arrays", "a", "--prefix", "train", "--ckpt", "c",
             "--params", json.dumps(PARAMS), "--fed-ports", "1,2", "--spu-ports", "3,4", "--protocol", "SEMI2K",
             "--field", "FM128", "--stop-at", "2", "--no-replay"])
    a = got["args"]
    assert a[0] == "hk" and a[5] == PARAMS and a[8] == [1, 2] and a[9] == [3, 4] and a[-1] is False


# ————————————————— 替身 SecretFlow：只检验编排（存档、续训回放、打分去向），不检验密码学 —————————————————

class _Obj:
    def __init__(self, v, device=None):
        self.v, self.device = v, device


class _PYU:
    def __init__(self, party):
        self.party = party

    def __call__(self, fn):
        def run(*args):
            return _Obj(fn(*[a.v if isinstance(a, _Obj) else a for a in args]), self)
        return run

    def __hash__(self):
        return hash(self.party)

    def __eq__(self, other):
        return isinstance(other, _PYU) and other.party == self.party


def _install_fake_secretflow(monkeypatch, calls):
    class SPU:
        def __init__(self, cdef):
            calls["spu"] = cdef

        def dump(self, obj, paths):
            for p in paths:
                Path(p).parent.mkdir(parents=True, exist_ok=True)
                Path(p).write_bytes(pickle.dumps(obj.v))

        def load(self, paths):
            return _Obj(pickle.loads(Path(paths[0]).read_bytes()), self)

    class FedNdarray:
        def __init__(self, partitions, partition_way):
            self.partitions = partitions

    class XgbModel:
        def __init__(self, spu, obj, base):
            self.spu, self.trees, self.weights = spu, [], []

        def predict(self, x, to_pyu):
            n = len(next(iter(x.partitions.values())).v)
            return FedNdarray({to_pyu: _Obj(np.full(n, sum(w.v for w in self.weights)), to_pyu)}, "v")

    class Xgb:
        def __init__(self, spu):
            self.spu = [spu]

        def _prepare(self, params, x, y):
            self.trees, self.obj, self.base, self.x = params.pop("num_boost_round"), "logistic", 0.0, x
            calls["y_rows"] = len(y.partitions[_PYU(SX.LABEL_PARTY)].v)

        def _global_setup(self):
            self.pred = [0.0]

        def _update_pred(self, tree, weight):
            calls["updates"].append(weight.v)

        def _tree_setup(self):
            pass

        def _train_tree(self):
            k = len(self.model.trees)
            tree = {d: _Obj(types.SimpleNamespace(split_features=[k], split_values=[1.0]), d) for d in self.x.partitions}
            return tree, _Obj(float(k + 1), self.spu[0])

    sf = types.ModuleType("secretflow")
    sf.init = lambda **kw: calls.__setitem__("init", kw)
    sf.shutdown = lambda: calls.__setitem__("shutdown", calls["shutdown"] + 1)
    sf.PYU, sf.SPU = _PYU, SPU
    ndarray = types.ModuleType("secretflow.data.ndarray")
    ndarray.FedNdarray, ndarray.PartitionWay = FedNdarray, types.SimpleNamespace(VERTICAL="vertical")
    device = types.ModuleType("secretflow.device")
    device.wait = lambda objs: None
    ss = types.ModuleType("secretflow.ml.boost.ss_xgb_v")
    ss.Xgb, ss.XgbModel = Xgb, XgbModel
    ns = types.ModuleType("secretflow.ml.boost.ss_xgb_v.core.node_split")
    ns.RegType = lambda v: v
    mods = {"secretflow": sf, "secretflow.data": types.ModuleType("secretflow.data"), "secretflow.data.ndarray": ndarray,
            "secretflow.device": device, "secretflow.ml": types.ModuleType("secretflow.ml"),
            "secretflow.ml.boost": types.ModuleType("secretflow.ml.boost"), "secretflow.ml.boost.ss_xgb_v": ss,
            "secretflow.ml.boost.ss_xgb_v.core": types.ModuleType("secretflow.ml.boost.ss_xgb_v.core"),
            "secretflow.ml.boost.ss_xgb_v.core.node_split": ns}
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)


def test_分段续训与打分用替身检验(tmp_path, monkeypatch):
    calls = {"shutdown": 0, "updates": []}
    _install_fake_secretflow(monkeypatch, calls)
    tx = iter(range(0, 1000, 7))
    monkeypatch.setattr(SX, "read_loopback_tx", lambda: next(tx))
    arrays = tmp_path / "a.npz"
    rng = np.random.default_rng(0)
    np.savez(arrays, train__xg=rng.random((6, 2)), train__xh=rng.random((6, 3)), train__y=np.array([0, 1] * 3),
             dev_eval__xg=rng.random((4, 2)), dev_eval__xh=rng.random((4, 3)))
    ck = str(tmp_path / "ck")
    args = (str(arrays), "train", ck, PARAMS)
    SX.run_party("neidi", "train", *args, 0, 2, *PORTS, "SEMI2K", "FM128", n_rows=5)
    p = SX.load_progress(ck, "neidi")
    assert p["trees_done"] == 2 and len(p["tree_seconds"]) == 2 and p["tree_lo_tx_bytes"] == [7, 7]
    assert calls["y_rows"] == 5 and calls["init"]["ray_mode"] is False and calls["updates"] == [1.0, 2.0]
    assert p["setup_seconds"] is not None and p["chunks"][0]["end"] == 2 and p["chunks"][0]["peak_rss_mb"] > 0
    assert SX.share_path(ck, "hk", 1).exists() and SX.tree_path(ck, "neidi", 1).exists()
    SX.run_party("neidi", "train", *args, 2, 9, *PORTS, "SEMI2K", "FM128")
    p = SX.load_progress(ck, "neidi")
    # 续训先按顺序回放已存的两棵，再训第三棵（最后一棵不再更新预测）
    assert calls["updates"] == [1.0, 2.0, 1.0, 2.0] and p["trees_done"] == 3 and [c["start"] for c in p["chunks"]] == [0, 2]
    assert SX.root_splits(ck, 3) == [("hk", 0, 1.0), ("hk", 1, 1.0), ("hk", 2, 1.0)]
    out = tmp_path / "pred.npy"
    SX.run_party("neidi", "predict", str(arrays), "dev_eval", ck, PARAMS, 3, 3, *PORTS, "SEMI2K", "FM128",
                 out=str(out))
    assert np.allclose(np.load(out), 6.0) and SX.load_progress(ck, "neidi")["predict"][0]["trees"] == 3
    with pytest.raises(ValueError):
        SX.run_party("neidi", "predict", str(arrays), "dev_eval", ck, PARAMS, 0, 0, *PORTS, "SEMI2K", "FM128")
    assert calls["shutdown"] == 3


def test_不回放对照(tmp_path, monkeypatch):
    calls = {"shutdown": 0, "updates": []}
    _install_fake_secretflow(monkeypatch, calls)
    monkeypatch.setattr(SX, "read_loopback_tx", lambda: 0)
    arrays = tmp_path / "a.npz"
    np.savez(arrays, train__xg=np.ones((4, 1)), train__xh=np.ones((4, 1)), train__y=np.array([0, 1, 0, 1]))
    ck = str(tmp_path / "ck")
    SX.run_party("hk", "train", str(arrays), "train", ck, PARAMS, 0, 1, *PORTS, "SEMI2K", "FM128")
    SX.run_party("hk", "train", str(arrays), "train", ck, PARAMS, 1, 3, *PORTS, "SEMI2K", "FM128", replay=False)
    assert calls["updates"] == [1.0, 2.0] and SX.load_progress(ck, "hk")["trees_done"] == 3   # 回放时会多出一次 1.0
