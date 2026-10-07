"""Synthetic numerical, leakage-boundary and private manager acceptance tests."""
from pathlib import Path
import copy
import json
import sys
import numpy as np
import pytest
import yaml

COMPONENTS = Path(__file__).resolve().parents[1] / "components"
sys.path.insert(0, str(COMPONENTS))
import convergence_audit as ca  # noqa: E402
import convergence_vm as provider  # noqa: E402


def config():
    c = yaml.safe_load((COMPONENTS.parent / "configs/convergence_audit.yaml").read_text())
    c["shared"].update(batch_size=8, infeed_rows=16, bootstrap_repeats=4)
    return c


def reference_update(x, y, weights, shared):
    w = weights.copy()
    size = shared["batch_size"]
    for begin in range(0, len(x), size):
        xx, yy = x[begin:begin + size], y[begin:begin + size]
        regularized = np.r_[w[:-1, 0], 0][:, None]
        gradient = xx.T @ (ca.ad.numpy_df(xx @ w, "df") - yy) / size
        w -= shared["learning_rates"][0] * (gradient + shared["l2_norm"] * regularized / size)
    return w


def fixture_round(tmp_path, monkeypatch, uplift):
    cfg = config()
    dataset = "hillstrom_email_marketing" if uplift else "uci_bank_marketing"
    root = tmp_path / "alice" / "synthetic"
    for sub in ("code", "data", "input", "results", "logs", "trainings"):
        (root / sub).mkdir(parents=True)
    rng = np.random.default_rng(6)
    rows = {}
    for split in ca.SPLITS:
        n = 36
        x = rng.normal(size=(n, 2)).astype(np.float32)
        y = np.tile([0, 1], n // 2).astype(np.float32)
        t = np.tile([0, 1, 2], n // 3) if uplift else np.zeros(n, dtype=int)
        rows[split] = dict(x=x, enhanced_x=np.c_[x, x[:, 0] ** 2], y=y, t=t, ids=np.arange(n))
        np.savez(root / "input" / f"{split}_local.npz", **rows[split])
    for seed in cfg["shared"]["model_seeds"]:
        for cell in ca.CELLS:
            enhanced = cell == "A1"
            w = ca.nf.initial_weights(2, 2, int(enhanced), uplift, seed, cfg["shared"]["initialization_std"], False)
            def callback(epoch, weights):
                if epoch == 12:
                    for split in ca.SPLITS:
                        p = ca.predictions(rows[split], weights, enhanced, uplift, split)
                        for route in (cell, "B" + cell[-1]):
                            np.save(root / "input" / f"{route}_seed{seed}_prediction_{split}.npy", p)
            ca.trajectory(rows, cfg["shared"], cfg["audit"], cfg["experiment"]["order_seeds"][str(seed)], w,
                          uplift, enhanced, reference_update, callback)
    ca.save(root / "input/manifest.json", {p.name: ca.sha(p) for p in (root / "input").iterdir()})
    path = root / "code/convergence_audit.yaml"
    path.write_text(yaml.safe_dump(cfg))
    (root / "code/source.py").write_text("# synthetic\n")
    monkeypatch.setattr(ca.ft, "ROOT", tmp_path)
    runtime = dict(party="alice", dataset=dataset, run_id=root.name, config=str(path),
                   config_sha256=ca.sha(path), code_sha256={"source.py": ca.sha(root / "code/source.py")}, bob_base_width=2)
    return root, runtime


@pytest.mark.parametrize("uplift", [False, True])
def test_complete_synthetic_round_and_fail_closed(tmp_path, monkeypatch, uplift):
    root, runtime = fixture_round(tmp_path, monkeypatch, uplift)
    result = ca.run(runtime, reference_update)
    assert result["status"] == "passed" and result["maximum_reference_error"] == 0
    assert len(result["curves"]) == 10
    for curve in result["curves"].values():
        assert [point["epoch"] for point in curve] == list(ca.CHECKPOINTS)
    assert all(len(x["difference_ci"]) == 2 for x in result["paired"].values())
    assert all((p / "results/own_weights_epoch48.npy").exists() for p in (root / "trainings").iterdir())
    with pytest.raises(FileExistsError):
        ca.run(runtime, reference_update)
    changed = dict(runtime, party="bob")
    with pytest.raises(ValueError, match="Owner"):
        ca.run(changed, reference_update)
    changed = copy.deepcopy(runtime)
    changed["code_sha256"]["source.py"] = "wrong"
    with pytest.raises(ValueError, match="Code"):
        ca.run(changed, reference_update)
    (root / "input/train_local.npz").write_bytes(b"tamper")
    with pytest.raises(ValueError, match="Frozen"):
        ca.load_inputs(root)


def test_df_objective_gradient_matches_finite_differences():
    s = config()["shared"]
    x = np.array([[1.0, 0.2, 1], [-0.3, 1.0, 1], [0.1, -0.2, 1]])
    y = np.array([0, 1, 1])
    w = np.array([[0.2], [-0.1], [0.04]])
    epsilon = 1e-6
    numeric = []
    for k in range(len(w)):
        plus, minus = w.copy(), w.copy()
        plus[k] += epsilon
        minus[k] -= epsilon
        numeric.append((ca.diagnostics(x, y, plus, s)["df_surrogate_objective"]
                        - ca.diagnostics(x, y, minus, s)["df_surrogate_objective"]) / (2 * epsilon))
    assert np.linalg.norm(numeric) == pytest.approx(ca.diagnostics(x, y, w, s)["full_unpadded_gradient_norm"], abs=1e-8)
    assert ca.diagnostics(x, y, w, s)["effective_l2"] == s["l2_norm"] / s["batch_size"]


def test_contract_no_test_partial_protocol_or_tolerance_relaxation(tmp_path):
    ca.validate(config())
    for field, replacement in (("checkpoints", [12, 48]), ("prediction_tolerance", 1), ("endpoints", [12, 24])):
        cfg = config()
        cfg["audit"][field] = replacement
        with pytest.raises(ValueError):
            ca.validate(cfg)
    cfg = config()
    cfg["experiment"]["order_seeds"] = {}
    with pytest.raises(ValueError, match="order"):
        ca.validate(cfg)
    assert ca.assert_reference(np.zeros((2, 1)), np.zeros((2, 1)), 0.001) == 0
    with pytest.raises(ValueError, match="tolerance"):
        ca.assert_reference(np.zeros((2, 1)), np.ones((2, 1)), 0.001)
    with pytest.raises(ValueError, match="shape"):
        ca.assert_reference(np.zeros((2, 1)), np.zeros((1, 1)), 0.001)
    with pytest.raises(ValueError, match="Test"):
        ca.predictions({}, None, False, False, "test")
    (tmp_path / "input").mkdir()
    ca.save(tmp_path / "input/manifest.json", {"test.npz": "no"})
    with pytest.raises(ValueError, match="capability"):
        ca.load_inputs(tmp_path)


def test_projection_and_padding_not_silent_sample_drop():
    cfg = config()
    row = dict(x=np.ones((19, 2)), enhanced_x=np.ones((19, 3)), y=np.ones(19), t=np.zeros(19, dtype=int))
    calls, checkpoints = [], []
    def update(x, y, w, shared):
        calls.append((len(x), int(y.sum())))
        return w + 100
    ca.trajectory({"train": row}, cfg["shared"], cfg["audit"], 101, np.zeros((3, 1)),
                  False, False, update, lambda epoch, w: checkpoints.append((epoch, w.max())))
    assert calls[:2] == [(16, 16), (8, 3)]
    assert len(calls) == 96 and all(maximum == 10 for _, maximum in checkpoints)


def test_provider_staging_and_private_export(tmp_path, monkeypatch):
    from types import SimpleNamespace
    source = tmp_path / "uci_bank_marketing" / "uci_bank_marketing_nonlinear_full_20261006T064813414917Z"
    source.mkdir(parents=True)
    commands = []
    def guest(party, script):
        commands.append((party, script))
        if party == "bob":
            return "2"
        return "passed" if script.endswith("echo passed") else ""
    monkeypatch.setattr(provider.vm, "guest", guest)
    monkeypatch.setattr(provider.vm, "copy_to", lambda *args: None)
    monkeypatch.setattr(provider.subprocess, "check_output", lambda *a, **k: "synthetic\n")
    target, runtime = provider.stage(tmp_path, source, tmp_path / "new", COMPONENTS.parent)
    assert runtime["party"] == "alice"
    assert all("/srv/vfl/bob" not in s for p, s in commands if p == "alice")
    def fake_lima(*args):
        name = str(args[1]).split("_", 1)[-1]
        dest = Path(args[2])
        payload = {"status": "passed", "code_cells": 3}
        dest.write_text(json.dumps(payload))
        return SimpleNamespace(stdout=name)
    monkeypatch.setattr(provider.vm, "lima", fake_lima)
    assert provider.execute(target, runtime)["status"] == "passed"
    with pytest.raises(ValueError):
        provider.source_paths("../../bad", [11])


def test_queue_success_failure_preserve_state(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    ca.save(source, {"status": "passed", "completed": ["one", "two"]})
    def stage(w, s, local, m):
        target = local / s.name
        target.mkdir()
        return target, {}
    monkeypatch.setattr(provider, "stage", stage)
    monkeypatch.setattr(provider, "execute", lambda *a: None)
    result = provider.queue(tmp_path, source, tmp_path / "success")
    assert result["status"] == "passed" and len(result["completed"]) == 2
    def fail(*args):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(provider, "execute", fail)
    with pytest.raises(RuntimeError):
        provider.queue(tmp_path, source, tmp_path / "failed")
    assert json.loads((tmp_path / "failed/status.json").read_text())["status"] == "failed"
    ca.save(source, {"status": "running", "completed": []})
    with pytest.raises(ValueError):
        provider.queue(tmp_path, source, tmp_path / "invalid")
