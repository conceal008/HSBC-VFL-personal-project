"""M5 · S5.M7 主动方明文直方图与 SGB 开销组件的单元测试。CI 没有 SecretFlow，用替身检验：
只替换主动方那一份、退出后恢复；计时包装不改返回值；逐位判定；ABBA 编排与汇总；Orange 大版按 k 取列。"""
from __future__ import annotations

import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m5_modeling.components import method_screen as MS  # noqa: E402
from modules.m5_modeling.components import method_screen_large as ML  # noqa: E402
from modules.m5_modeling.components import sgb_cost_screen as SC  # noqa: E402
from modules.m5_modeling.tests.test_method_screen import CFG  # noqa: E402

GE_PATH = "secretflow.ml.boost.sgb_v.factory.components.gradient_encryptor"
HEU_PATH = "secretflow.device.device.heu_object"
PYU_PATH = "secretflow.device.device.pyu"


class _Moved:
    def __init__(self, what, who):
        self.what, self.who = what, who


class _GH:
    def __init__(self, what):
        self.what = what

    def to(self, heu, cfg):
        return _Moved(self.what, cfg)


class _GradientEncryptor:
    def __init__(self, workers, holder, feature_only=False):
        self.workers, self.label_holder, self.heu, self.gh_encoder = workers, holder, "heu", "enc"
        self.params = types.SimpleNamespace(label_holder_feature_only=feature_only)

    def cache_to_workers(self, encrypted_gh, gh):          # 原实现：所有方都拿密文
        return {w: encrypted_gh.to(self.heu, w) for w in self.workers}


class _PYU:
    def __init__(self, party):
        self.party = party


class _HEUObject:
    def __init__(self, location, is_plain):
        self.location, self.is_plain = location, is_plain

    def batch_feature_wise_bucket_sum(self, *a, **k):
        return ("sum", self.location)

    def to(self, device, config=None):
        return ("moved", self.location, type(device).__name__)


@pytest.fixture()
def fake_sf_parts(monkeypatch):
    ge = types.ModuleType(GE_PATH + ".gradient_encryptor")
    ge.GradientEncryptor = _GradientEncryptor
    ge.move_config = lambda pyu, enc: pyu
    pkg = types.ModuleType(GE_PATH)
    pkg.gradient_encryptor = ge
    heu = types.ModuleType(HEU_PATH)
    heu.HEUObject = _HEUObject
    pyu = types.ModuleType(PYU_PATH)
    pyu.PYU = _PYU
    for name, mod in {GE_PATH: pkg, GE_PATH + ".gradient_encryptor": ge, HEU_PATH: heu, PYU_PATH: pyu}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return ge


def test_只把主动方那一份换成明文_退出后恢复(fake_sf_parts):
    orig = _GradientEncryptor.cache_to_workers
    enc = _GradientEncryptor(["neidi", "hk"], "neidi")
    with SC.plaintext_label_holder(True):
        cache = enc.cache_to_workers(_GH("cipher"), _GH("plain"))
        assert cache["hk"].what == "cipher" and cache["hk"].who == "hk"          # 被动方不变
        assert cache["neidi"].what == "plain" and cache["neidi"].who == "neidi"  # 主动方用明文
        fo = _GradientEncryptor(["neidi", "hk"], "neidi", feature_only=True)
        assert fo.cache_to_workers(_GH("cipher"), _GH("plain"))["neidi"].what == "cipher"   # 该模式沿用原实现
    assert _GradientEncryptor.cache_to_workers is orig
    with SC.plaintext_label_holder(False):
        assert _GradientEncryptor.cache_to_workers is orig
    with pytest.raises(RuntimeError), SC.plaintext_label_holder(True):
        raise RuntimeError("中途出错")
    assert _GradientEncryptor.cache_to_workers is orig


def test_计时包装不改返回值并按方累加(fake_sf_parts):
    acc = defaultdict(float)
    bucket, move = _HEUObject.batch_feature_wise_bucket_sum, _HEUObject.to
    with SC.party_timer(acc):
        a, b = _HEUObject("neidi", False), _HEUObject("hk", True)
        assert a.batch_feature_wise_bucket_sum() == ("sum", "neidi")
        assert a.to(_PYU("neidi")) == ("moved", "neidi", "_PYU")
        assert b.to(_PYU("neidi")) == ("moved", "hk", "_PYU")
        assert b.to("heu") == ("moved", "hk", "str")                         # 非 PYU 的移动不计时
    assert set(acc) == {"bucket_sum@neidi", "to_pyu@neidi（密文）", "to_pyu@hk（明文）"}
    assert _HEUObject.batch_feature_wise_bucket_sum is bucket and _HEUObject.to is move


def test_逐位判定():
    y = np.r_[np.ones(5), np.zeros(5)]
    a = np.linspace(0.1, 0.9, 10).astype(np.float32)
    assert SC.compare(a, a.copy(), y)["verdict"] == SC.EXACT
    b = a.copy()
    b[0] = np.nextafter(b[0], np.float32(1))                                  # 末位不同、排序不变
    r = SC.compare(a, b, y)
    assert r["verdict"] == SC.EQUIVALENT and r["rows_different"] == 1
    c = a[::-1].copy()
    assert SC.compare(a, c, y)["verdict"] == SC.NOT_EQUIVALENT


def test_ABBA_编排与汇总(monkeypatch):
    P = MS.Params(CFG)
    seen = []

    def fake_fit(P_, args, plain, rounds=None):
        seen.append(("plain" if plain else "base", rounds))
        sec = (6.0 if plain else 10.0) * rounds
        return {"pred": np.arange(4, dtype=np.float32), "fit_seconds": sec, "trees": rounds, "parts": {}}

    monkeypatch.setattr(SC, "fit", fake_fit)
    monkeypatch.setattr(SC, "task_args", lambda P_, sc, res, m: (None,) * SC.ARG_CFG + ({"rounds": 7}, 33, 0.0))
    sc = types.SimpleNamespace(v={"final": {"fit_xg": np.zeros((2, 3)), "fit_xh": np.zeros((2, 1))}})
    out = SC.timing(P, sc, {}, "F_opt_p", max_trees=10)
    assert seen == [("base", SC.WARMUP_TREES)] + [(k, 7) for k in SC.ORDER]   # 预热 + ABBA，棵数取 min(10, 7)
    assert out["trees"] == 7 and out["preds_identical"]
    assert abs(out["saving"] - 0.4) < 1e-12 and out["saving_range"] == [pytest.approx(0.4)] * 2
    assert out["cols_guest"] == 3 and abs(out["guest_share"] - 0.75) < 1e-12


def test_线性外推():
    r = SC.linear_extrapolate([200, 600, 2000], [5 + 0.1 * 200, 5 + 0.1 * 600, 5 + 0.1 * 2000], 13000)
    assert abs(r["slope_sec_per_col"] - 0.1) < 1e-9 and abs(r["sec_per_tree_extrapolated"] - 1305) < 1e-6


def test_大版按k取列与S5M5一致():
    rng = np.random.default_rng(0)
    G, H = rng.normal(size=(12, 6)).astype(np.float32), rng.normal(size=(12, 5)).astype(np.float32)
    fit = np.r_[np.ones(8, bool), np.zeros(4, bool)]
    y = (rng.random(12) < 0.5).astype(float)
    sens = {"G": G, "H": H, "fit": fit, "y": y, "rank_g": np.array([4, 0, 2, 5, 1, 3]), "rank_h": np.array([3, 1, 0, 4, 2])}
    v = SC.large_view(sens, 2)
    assert np.array_equal(v["fit_xg"], G[:8][:, [0, 4]]) and np.array_equal(v["ev_xh"], H[8:][:, [1, 3]])
    assert np.array_equal(v["fit_y"], y[:8]) and np.array_equal(v["ev_y"], y[8:])
    full = SC.large_view(sens, ML.ALL)
    assert full["fit_xg"].shape == (8, 6) and full["ev_xh"].shape == (4, 5)
    assert set(sens) == {"G", "H", "fit", "y", "rank_g", "rank_h"}            # 不修改传入的 sens


def _fake_fit_factory(seen, pred):
    def fake_fit(P_, args, plain, rounds=None):
        seen.append(("plain" if plain else "base", rounds, args[0].shape[1] if args[0] is not None else None))
        n = rounds or args[SC.ARG_CFG]["rounds"]
        return {"pred": pred(args), "fit_seconds": (6.0 if plain else 10.0) * n, "trees": n, "parts": {"x": 1.0}}
    return fake_fit


def test_完整拟合比较与大版单树开销编排(monkeypatch, tmp_path):
    P = MS.Params(CFG)
    seen = []
    ev = np.linspace(0.2, 0.8, 4).astype(np.float32)
    monkeypatch.setattr(SC, "fit", _fake_fit_factory(seen, lambda args: ev.copy()))
    monkeypatch.setattr(SC, "task_args", lambda P_, sc, res, m: (None,) * SC.ARG_CFG + ({"rounds": 3}, 33, 0.0))
    sc = types.SimpleNamespace(v={"final": {"ev_y": np.array([0.0, 1.0, 0.0, 1.0])}})
    r = SC.exact_check(P, sc, {}, "F_opt_p", ev.copy())
    assert r["verdict"] == SC.EXACT and seen == [("plain", None, None)] and r["trees"] == 3

    seen.clear()
    rng = np.random.default_rng(1)
    sens = {"G": rng.normal(size=(10, 5)).astype(np.float32), "H": rng.normal(size=(10, 4)).astype(np.float32),
            "fit": np.r_[np.ones(7, bool), np.zeros(3, bool)], "y": np.r_[np.ones(4), np.zeros(6)],
            "rank_g": np.array([0, 1, 2, 3, 4]), "rank_h": np.array([0, 1, 2, 3])}
    res = {"configs": {"central": {"lr": 0.1, "depth": 3, "lam": 1.0, "col": 1.0, "sub": 1.0, "rounds": 9}, "m1": {}}}
    monkeypatch.setattr(SC, "fit", _fake_fit_factory(seen, lambda args: np.zeros(3, np.float32)))
    rows = SC.k_cost(P, sens, res, [2, 4], trees=5)
    assert [s[0] for s in seen] == ["base", "base", "plain", "base", "plain", "base"]   # 每个 k 先预热，k 之间交替先后
    assert [s[2] for s in seen] == [2, 2, 2, 4, 4, 4] and all(r["preds_identical"] for r in rows)
    assert [r["cols_host"] for r in rows] == [2, 4] and abs(rows[0]["saving"] - 0.4) < 1e-12


def test_单次拟合_截棵数_上下文与分项(fake_sf_parts, monkeypatch):
    P = MS.Params(CFG)
    orig = _GradientEncryptor.cache_to_workers
    seen = []

    def fake_run_sgb(P_, xg, xh, y, eg, eh, cfg, bins, base):
        seen.append((cfg["rounds"], _GradientEncryptor.cache_to_workers is not orig))
        _HEUObject("neidi", False).batch_feature_wise_bucket_sum()          # 计时包装在拟合期间生效
        return np.array([0.25, 0.75]), 1.5

    monkeypatch.setattr(SC.SG, "run_sgb", fake_run_sgb)
    args = (None,) * SC.ARG_CFG + ({"rounds": 9, "depth": 3}, 33, 0.0)
    r = SC.fit(P, args, plain=True, rounds=4)
    assert r["trees"] == 4 and r["pred"].dtype == np.float32 and set(r["parts"]) == {"bucket_sum@neidi"}
    r = SC.fit(P, args, plain=False)
    assert r["trees"] == 9 and seen == [(4, True), (9, False)] and args[SC.ARG_CFG]["rounds"] == 9   # 不改传入的配置
    assert _GradientEncryptor.cache_to_workers is orig


def test_任务输入顺序_不接受逐样本先验(monkeypatch):
    P = MS.Params(CFG)
    fv = {k: k for k in ("fit_xg", "fit_xh", "fit_y", "ev_xg", "ev_xh")}
    sc = types.SimpleNamespace(v={"final": fv})
    monkeypatch.setattr(SC.SG, "inputs", lambda P_, sc_, res, m: ({"rounds": 5}, 33, 0.1, 0.2, None, None))
    assert SC.task_args(P, sc, {}, "F_opt_p") == ("fit_xg", "fit_xh", "fit_y", "ev_xg", "ev_xh", {"rounds": 5}, 33, 0.1)
    monkeypatch.setattr(SC.SG, "inputs", lambda P_, sc_, res, m: ({"rounds": 5}, 33, None, None, np.zeros(2), np.zeros(2)))
    with pytest.raises(AssertionError):
        SC.task_args(P, sc, {}, "F_M1")


def test_大版真实SGB复核入口_打分存档与代理比较(monkeypatch, tmp_path):
    P = MS.Params(CFG)
    v = {"fit_xg": np.zeros((4, 2)), "fit_xh": np.zeros((4, 1)), "fit_y": np.array([1.0, 0.0, 1.0, 0.0]),
         "ev_xg": np.zeros((4, 2)), "ev_xh": np.zeros((4, 1)), "ev_y": np.array([0.0, 1.0, 0.0, 1.0])}
    res = {"configs": {"central": {"lr": 0.1, "depth": 3, "lam": 1.0, "col": 1.0, "sub": 1.0, "rounds": 9}, "m1": {}}}
    pred = np.array([0.1, 0.9, 0.2, 0.8], np.float32)
    monkeypatch.setattr(SC, "fit", lambda P_, args, plain, rounds=None: {"pred": pred, "fit_seconds": 2.04, "trees": 9,
                                                                       "parts": {"x": 1.0}, "plain": plain})
    monkeypatch.setattr(SC.MS, "train_pred", lambda *a: np.array([0.2, 0.8, 0.3, 0.7]))
    r = SC.confirm_large(P, v, res, "F_opt_p", tmp_path, "t")
    assert np.array_equal(np.load(tmp_path / "t__F_opt_p.npy"), pred)
    assert r["sgb_auroc"] == 1.0 and r["proxy_seed11_auroc"] == 1.0 and r["trees"] == 9 and r["depth"] == 3
    assert abs(r["max_abs_pred_diff_vs_proxy"] - 0.1) < 1e-6 and r["fit_seconds"] == 2.0


def test_启动指纹与JSON(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")
    fp = SC.fingerprint([f])
    assert "sha256:2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881" in fp
    assert fp.splitlines()[-1].startswith("secretflow")
    SC.write_json(tmp_path / "b.json", {"判定": SC.EXACT})
    assert SC.EXACT in (tmp_path / "b.json").read_text(encoding="utf-8")


def test_命令行入口_三个部分与续跑(monkeypatch, tmp_path):
    import json

    import yaml
    monkeypatch.chdir(ROOT)
    dirs = {k: tmp_path / k for k in ("priv", "run", "sgb", "large", "cost")}
    for d in dirs.values():
        d.mkdir()
    m7 = yaml.safe_load((ROOT / "modules/m5_modeling/configs/s5_m7_sgb_cost.yaml").read_text(encoding="utf-8"))
    m7.update(sgb_dir_env="T_SGB", large_dir_env="T_LARGE", cost_dir_env="T_COST", timing_tasks=["a"], exact_tasks=["a"],
              large_task="L", k_cost=[2, 4], k_cost_trees=3)
    cfg_path = tmp_path / "m7.yaml"
    cfg_path.write_text(yaml.safe_dump(m7, allow_unicode=True), encoding="utf-8")
    for env, d in (("T_SGB", "sgb"), ("T_LARGE", "large"), ("T_COST", "cost"), (CFG["private_root_env"], "priv"),
                   (CFG["run_dir_env"], "run")):
        monkeypatch.setenv(env, str(dirs[d]))
    (dirs["run"] / "a.json").write_text("{}", encoding="utf-8")
    (dirs["large"] / "L.json").write_text("{}", encoding="utf-8")
    np.save(dirs["sgb"] / "a__F_opt_p.npy", np.zeros(3, np.float32))
    calls = []
    monkeypatch.setattr(SC.MS, "Screen", lambda name, cfg, priv, out: types.SimpleNamespace(name=name))
    monkeypatch.setattr(SC.ML, "load_sens", lambda priv, name: calls.append(("sens", name)) or {})
    monkeypatch.setattr(SC, "k_cost", lambda P_, sens, res, ks, trees: calls.append(("kcost", ks, trees)) or [{"k": 2}])
    monkeypatch.setattr(SC, "timing", lambda P_, sc, res, m, n: calls.append(("timing", sc.name, m, n)) or {
        "sec_per_tree_base": 2.0, "sec_per_tree_plain": 1.0, "saving": 0.5, "preds_identical": True})
    monkeypatch.setattr(SC, "exact_check", lambda P_, sc, res, m, stored: calls.append(("exact", sc.name, len(stored))) or {
        "verdict": SC.EXACT, "rows_different": 0, "rows": 3, "max_abs_diff": 0.0})
    for part in ("kcost", "timing", "exact"):
        assert SC.main(["--config", str(cfg_path), "--part", part]) == 0
    assert calls == [("sens", "L"), ("kcost", [2, 4], 3), ("timing", "a", "F_opt_p", 10), ("exact", "a", 3)]
    assert json.loads((dirs["cost"] / "exact__a.json").read_text(encoding="utf-8"))["method"] == "F_opt_p"
    assert json.loads((dirs["cost"] / "kcost__L.json").read_text(encoding="utf-8"))["rows"] == [{"k": 2}]
    assert (dirs["cost"] / "fingerprint_launch.txt").read_text(encoding="utf-8").count("== ") == 3
    calls.clear()
    for part in ("kcost", "timing", "exact"):
        SC.main(["--config", str(cfg_path), "--part", part])
    assert calls == []                                                       # 已完成的部分跳过
