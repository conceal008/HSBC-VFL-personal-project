"""Synthetic acceptance and routing tests; never claim mocks prove MPC security."""

from pathlib import Path
import contextlib
import json
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
for name in ("m7_security", "m5_modeling", "m3_alignment"):
    sys.path.insert(0, str(REPO / "modules" / name / "components"))
import output_audit as audit  # noqa: E402
import output_audit_vm as provider  # noqa: E402


def config():
    return yaml.safe_load(
        (REPO / "modules/m7_security/configs/output_audit.yaml").read_text()
    )


def test_controls_and_numerical_inverse(monkeypatch):
    cfg = config()
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(numpy=np))
    monkeypatch.setitem(sys.modules, "jax.numpy", np)
    x = np.asarray(cfg["synthetic_probe"]["x"], dtype=np.float32)
    y = np.asarray(cfg["synthetic_probe"]["y"], dtype=np.float32)
    z = audit.secret_probe(
        x, x, y, y, cfg["ridge"], cfg["inverse_steps"], cfg["inverse_tolerance"]
    )
    assert audit.synthetic_accept(
        z, audit.synthetic_expected(cfg), cfg["synthetic_tolerance"]
    )
    assert not audit.synthetic_accept(
        z + 1, audit.synthetic_expected(cfg), cfg["synthetic_tolerance"]
    )
    assert audit.view([0.127, 0.9], "rounded", cfg).tolist() == [0.13, 0.9]
    assert audit.view([0.1, 0.6], "binary", cfg).tolist() == [0, 1]
    with pytest.raises(ValueError):
        audit.view([], "bad", cfg)
    with pytest.raises(ValueError):
        audit.interval([1], cfg, 11)
    with pytest.raises(ValueError):
        audit.split_indices(1, 11, 1)
    labels = np.tile([0, 1], 10)
    c = np.arange(10)
    e = np.arange(10, 20)
    assert audit.attack_metrics(labels, labels, c, e, cfg)["auc"] == 1
    assert audit.attack_metrics(labels, -labels, c, e, cfg)["auc"] == 1
    assert audit.interval([1] * 5, cfg, 11)["ci_low"] == 1


def inputs(tmp_path, monkeypatch):
    cfg = config()
    cfg.update(maximum_rows=50, bootstrap_repeats=20, inverse_steps=16)
    monkeypatch.setattr(audit.ft, "ROOT", tmp_path)
    monkeypatch.setattr(audit.ft, "isolation_preflight", lambda p: {"unit_only": True})
    rng = np.random.default_rng(10)
    for party in provider.vm.PARTIES:
        root = tmp_path / party / "unit"
        for sub in ("input", "code", "data", "logs", "trainings", "results"):
            (root / sub).mkdir(parents=True)
        (root / "code/unit.py").write_text("unit")
        for split in audit.SPLITS:
            x = rng.normal(size=(60, 3)).astype(np.float32)
            row = {"x": x, "enhanced_x": x, "ids": np.arange(60)}
            if party == "alice":
                row.update(y=np.tile([0, 1], 30), t=np.zeros(60, dtype=int))
            np.savez(root / "input" / f"{split}_local.npz", **row)
            if party == "alice":
                for seed in cfg["model_seeds"]:
                    for cell in audit.CELLS:
                        p = (0.5 + 0.25 * np.tanh(x[:, 0]))[:, None]
                        np.save(root / "input" / f"{cell}_seed{seed}_{split}.npy", p)
        if party == "bob":
            pd.DataFrame(columns=["record_id", "numeric__previous", "f1", "f2"]).to_csv(
                root / "input/train.csv", index=False
            )
        manifest = {p.name: audit.sha(p) for p in (root / "input").iterdir()}
        audit.save(root / "input/manifest.json", manifest)
        cp = root / "code/config.yaml"
        cp.write_text(yaml.safe_dump(cfg))
    runtime = {
        "run_id": "unit",
        "party": "alice",
        "dataset": "uci_bank_marketing",
        "config": str(tmp_path / "alice/unit/code/config.yaml"),
        "config_sha256": audit.sha(tmp_path / "alice/unit/code/config.yaml"),
        "code_sha256": {},
    }
    return cfg, runtime


def test_owner_stress_and_private_targets(tmp_path, monkeypatch):
    cfg, runtime = inputs(tmp_path, monkeypatch)
    a = audit.owner_inputs("alice", runtime, cfg)
    b = audit.owner_inputs("bob", runtime, cfg)
    assert "predictions" not in b and b["target_index"] == 0
    assert audit.local_stress(a, cfg)
    assert len(list((tmp_path / "alice/unit/trainings").iterdir())) == 60
    x, xv = audit.probe_inputs(a, "B0", 11, 701, "exact", cfg)
    base, _ = audit.probe_inputs(a, "B0", 11, 701, "baseline", cfg)
    y, yv = audit.probe_inputs(b, "B0", 11, 701, "exact", cfg)
    assert base.shape[1] + 1 == x.shape[1] and len(y) == len(x) and len(yv) == len(xv)
    assert not set(audit.split_indices(60, 701, 50)[0]) & set(
        audit.split_indices(60, 701, 50)[1]
    )
    assert np.array_equal(
        a["train"]["y"][audit.matched_pools(a["train"], a["validation"], 701, 50)[0]],
        a["validation"]["y"][
            audit.matched_pools(a["train"], a["validation"], 701, 50)[1]
        ],
    )
    with pytest.raises(ValueError):
        audit.store_probe(a, "bad", [np.nan, 1, 1, 1], "B0", 11, "exact")
    (tmp_path / "alice/unit/input/test.csv").write_text("bad")
    m = json.loads((tmp_path / "alice/unit/input/manifest.json").read_text())
    m["test.csv"] = audit.sha(tmp_path / "alice/unit/input/test.csv")
    audit.save(tmp_path / "alice/unit/input/manifest.json", m)
    with pytest.raises(ValueError):
        audit.owner_inputs("alice", runtime, cfg)


def test_run_routes_only_aggregate_to_alice(tmp_path, monkeypatch):
    cfg, runtime = inputs(tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(numpy=np))
    monkeypatch.setitem(sys.modules, "jax.numpy", np)
    released = []

    class Value:
        def __init__(self, value, owner):
            self.value = value
            self.owner = owner

        def to(self, recipient):
            if self.owner == "secure":
                released.append(recipient.name)
            return Value(self.value, recipient.name)

    def unwrap(a):
        if isinstance(a, Value):
            return a.value
        if isinstance(a, list):
            return [unwrap(v) for v in a]
        return a

    class Device:
        def __init__(self, name):
            self.name = name

        def __call__(self, fn, **options):
            def call(*args, **kw):
                result = fn(*[unwrap(a) for a in args], **kw)
                if options.get("num_returns") == 2:
                    return tuple(Value(r, self.name) for r in result)
                return Value(result, self.name)

            return call

    sf = SimpleNamespace(PYU=Device, wait=lambda x: None, reveal=unwrap)

    @contextlib.contextmanager
    def session(*a):
        yield sf, Device("secure")

    monkeypatch.setattr(audit.bp, "session", session)
    result = audit.run(runtime)
    assert result["status"] == "passed" and set(released) == {"alice"}
    report = json.loads(
        (tmp_path / "alice/unit/results/private_attribute_audit.json").read_text()
    )
    assert len(report["details"]) == 40 and len(report["paired"]) == 6
    runtime["config_sha256"] = "bad"
    with pytest.raises(ValueError):
        audit.run(runtime)


def test_provider_mapping_and_stage(tmp_path, monkeypatch):
    cfg = config()
    source = tmp_path / "uci_bank_marketing" / "uci_bank_marketing_nonlinear_full_unit"
    with pytest.raises(ValueError):
        provider.mapping("../bad", "alice", cfg["model_seeds"])
    assert not any(
        n.endswith(".npy")
        for n in provider.mapping(source.name, "bob", cfg["model_seeds"])
    )
    assert len(provider.mapping(source.name, "alice", cfg["model_seeds"])) == 42
    commands = []
    vm = provider.vm
    monkeypatch.setattr(
        vm, "setup_certificates", lambda w: {"alice": "unit", "bob": "unit"}
    )
    monkeypatch.setattr(vm, "restrict_network", lambda *a: None)
    monkeypatch.setattr(vm, "guest", lambda p, c: commands.append((p, c)) or "passed")
    monkeypatch.setattr(vm, "copy_to", lambda *a: None)
    target, run = provider.stage(tmp_path, source, tmp_path / "output")
    assert not any("test.csv" in c for _, c in commands)
    assert all(
        "sudo install" in c for _, c in commands if "/input/" in c and "install" in c
    )
    report = {"details": [{}] * 40}
    for party in vm.PARTIES:
        audit.save(target / party / "results/engineering.json", {"status": "passed"})
        audit.save(
            target / party / "results/party.executed.ipynb",
            {
                "cells": [
                    {"cell_type": "code", "execution_count": i, "outputs": []}
                    for i in (1, 2, 3)
                ]
            },
        )
    audit.save(target / "alice/results/private_attribute_audit.json", report)
    assert provider.verify(target, run)["status"] == "passed"
    (target / "bob/results/forbidden.npy").write_bytes(b"forbidden")
    with pytest.raises(ValueError):
        provider.verify(target, run)


def test_provider_queue_and_failure(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    audit.save(source, {"status": "passed", "completed": ["a", "b"]})
    monkeypatch.setattr(provider, "stage", lambda w, s, o: (o / s, s.name))
    monkeypatch.setattr(provider, "execute", lambda *a: {"status": "unit"})
    assert provider.queue(tmp_path, source, tmp_path / "q")["status"] == "passed"

    def failed(*a):
        raise RuntimeError("unit")

    monkeypatch.setattr(provider, "execute", failed)
    with pytest.raises(RuntimeError):
        provider.queue(tmp_path, source, tmp_path / "fail")
    assert json.loads((tmp_path / "fail/status.json").read_text())["status"] == "failed"
    audit.save(source, {"status": "failed"})
    with pytest.raises(ValueError):
        provider.queue(tmp_path, source, tmp_path / "invalid")


def test_execute_exports_on_success_and_failure(tmp_path, monkeypatch):
    vm = provider.vm
    for party in vm.PARTIES:
        (tmp_path / party / "logs").mkdir(parents=True)
    exports = []
    monkeypatch.setattr(
        provider.provider,
        "export",
        lambda t, r, p, failed=False: exports.append((p, failed)),
    )
    monkeypatch.setattr(provider, "verify", lambda *a: {"status": "unit routing only"})
    monkeypatch.setattr(vm, "guest", lambda *a: "passed")
    for success in (True, False):

        class Process:
            def __init__(self, *a, **kw):
                self.returncode = 0 if success else 1

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

        monkeypatch.setattr(provider.subprocess, "Popen", Process)
        if success:
            assert provider.execute(tmp_path, "unit")["status"] == "unit routing only"
        else:
            with pytest.raises(RuntimeError):
                provider.execute(tmp_path, "unit")
    assert exports == [("alice", False), ("bob", False), ("alice", True), ("bob", True)]
