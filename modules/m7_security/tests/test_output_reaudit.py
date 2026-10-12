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
import output_reaudit as audit  # noqa: E402
import output_reaudit_vm as provider  # noqa: E402


def config():
    return yaml.safe_load(
        (REPO / "modules/m7_security/configs/output_reaudit.yaml").read_text()
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
                    for cell in cfg["output_cells"]:
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
    assert len(list((tmp_path / "alice/unit/trainings").iterdir())) == 330
    x, xv = audit.probe_inputs(a, "D_B0_e48", 11, 701, "exact", cfg)
    base, _ = audit.probe_inputs(a, "D_B0_e48", 11, 701, "baseline", cfg)
    y, yv = audit.probe_inputs(b, "D_B0_e48", 11, 701, "exact", cfg)
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
        audit.store_probe(a, "bad", [np.nan, 1, 1, 1], "D_B0_e48", 11, "exact")
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
    assert len(report["details"]) == 400 and len(report["paired"]) == 60
    runtime["config_sha256"] = "bad"
    with pytest.raises(ValueError):
        audit.run(runtime)


def test_constant_binary_output_is_not_useful_model_extraction(tmp_path, monkeypatch):
    cfg, runtime = inputs(tmp_path, monkeypatch)
    cfg["output_cells"] = ["D_A0_e48"]
    a = audit.owner_inputs("alice", runtime, cfg)
    for per_seed in a["predictions"].values():
        for split in audit.SPLITS:
            per_seed["D_A0_e48"][split][:] = 0.1
    assert audit.local_stress(a, cfg)
    report = json.loads(
        (Path(a["root"]) / "results/private_output_audit.json").read_text()
    )
    item = report["utility"]["D_A0_e48_binary"]
    assert item["constant_views"] == cfg["model_seeds"]
    assert item["unidentifiable_rankings"] == cfg["model_seeds"]
    assert item["roc_auc"]["estimate"] == 0.5


def test_owner_rejects_failed_actual_preflight(tmp_path, monkeypatch):
    cfg, runtime = inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(audit.ft, "isolation_preflight", lambda _: {"nonroot": False})
    with pytest.raises(ValueError, match="isolation"):
        audit.owner_inputs("alice", runtime, cfg)


def test_mapping_covers_all_declared_outputs_and_keeps_bob_scores_absent():
    cfg = config()
    sources = {
        "E4": "uci_bank_marketing_candidate_full_unit",
        "E5": "uci_bank_marketing_joint_budget_unit",
    }
    alice = provider.mapping(sources, "alice", cfg)
    bob = provider.mapping(sources, "bob", cfg)
    assert len([n for n in alice if n.endswith(".npy")]) == 22 * 5 * 2
    assert not any(n.endswith(".npy") for n in bob)
    assert alice["D_B0_e48_seed11_validation.npy"][1].endswith(
        "D_B0_seed11/checkpoints/e48/results/validation_predictions.npy"
    )
    assert alice["FXM0_seed11_validation.npy"][0] == sources["E4"]
    with pytest.raises(ValueError):
        provider.mapping(dict(sources, E5="../bad"), "alice", cfg)


def test_two_source_queues_must_complete_before_any_followup(tmp_path, monkeypatch):
    q4, q5 = tmp_path / "e4.json", tmp_path / "e5.json"
    for q in [q4, q5]:
        audit.save(
            q,
            {
                "status": "passed",
                "completed": [
                    str(tmp_path / d / (d + "_unit")) for d in provider.vm.PREPARED
                ],
            },
        )
    calls = []
    monkeypatch.setattr(provider, "stage", lambda w, s, o, d: (o / d, d))
    monkeypatch.setattr(provider, "execute", lambda t, r: calls.append(r))
    assert provider.queue(tmp_path, q4, q5, tmp_path / "out")["status"] == "passed"
    assert calls == list(provider.vm.PREPARED)
    audit.save(q5, {"status": "running", "completed": []})
    with pytest.raises(ValueError, match="incomplete"):
        provider.queue(tmp_path, q4, q5, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()


def test_stage_all_sources_and_complete_inventory_then_fail_closed(
    tmp_path, monkeypatch
):
    cfg = config()
    sources = {
        "E4": "uci_bank_marketing_candidate_full_unit",
        "E5": "uci_bank_marketing_joint_budget_unit",
    }
    vm = provider.vm
    commands = []
    monkeypatch.setattr(
        vm, "setup_certificates", lambda _: {"alice": "unit", "bob": "unit"}
    )
    monkeypatch.setattr(vm, "restrict_network", lambda *a: None)
    monkeypatch.setattr(vm, "copy_to", lambda *a: None)

    def guest(party, command):
        commands.append((party, command))
        if "sudo cat" in command:
            return json.dumps(
                {
                    "party": party,
                    "dataset": "uci_bank_marketing",
                    "config": "/unit/"
                    + provider.SOURCE_CONFIG[
                        "E4" if "_candidate_full_" in command else "E5"
                    ],
                    "config_sha256": "0" * 64,
                    "code_sha256": {
                        n: "0" * 64
                        for n in provider.SOURCE_CODE[
                            "E4" if "_candidate_full_" in command else "E5"
                        ]
                    },
                }
            )
        if "sha256sum" in command:
            return "0" * 64 + " unit"
        if "owner_matrices_exact" in command:
            return "owner_matrices_exact"
        return "passed"

    monkeypatch.setattr(vm, "guest", guest)
    target, run = provider.stage(
        tmp_path, sources, tmp_path / "out", "uci_bank_marketing"
    )
    for party in vm.PARTIES:
        runtime = json.loads((target / party / "code/runtime.json").read_text())
        assert runtime["source_runs"] == sources and runtime["party"] == party
        audit.save(
            target / party / "logs/owner_isolation.json", {"unit_preflight": True}
        )
        audit.save(target / party / "results/engineering.json", {"status": "passed"})
        audit.save(
            target / party / "results/party.executed.ipynb",
            {
                "cells": [
                    {"cell_type": "code", "execution_count": i, "outputs": []}
                    for i in [1, 2, 3]
                ]
            },
        )
    assert not any(
        ".npy" in command and "sudo install" in command
        for party, command in commands
        if party == "bob"
    )
    attributes = [
        {"cell": c, "seed": s, "view": v}
        for c in cfg["attribute_cells"]
        for s in cfg["model_seeds"]
        for v in ["baseline", "exact", "rounded", "binary"]
    ]
    outputs = [
        {"cell": c, "seed": s, "view": v}
        for c in cfg["output_cells"]
        for s in cfg["model_seeds"]
        for v in ["exact", "rounded", "binary"]
    ]
    audit.save(
        target / "alice/results/private_attribute_audit.json", {"details": attributes}
    )
    summaries = {
        c + "_" + v: {}
        for c in cfg["output_cells"]
        for v in ["exact", "rounded", "binary"]
    }
    output_report = {"details": outputs, "utility": summaries, "aggregate": summaries}
    audit.save(target / "alice/results/private_output_audit.json", output_report)
    audit.save(target / "status.json", {"status": "passed"})
    assert provider.verify(target, run)["status"] == "passed"
    output_report["utility"]["M0_exact"] = {
        "unit_interval": {"estimate": 0.0, "ci_low": 1.0, "ci_high": 0.0}
    }
    audit.save(target / "alice/results/private_output_audit.json", output_report)
    with pytest.raises(ValueError, match="interval"):
        provider.verify(target, run)
    output_report["utility"]["M0_exact"] = {}
    audit.save(target / "alice/results/private_output_audit.json", output_report)
    snapshot = (target / "alice/code/runtime.json").read_bytes()
    modified = json.loads(snapshot)
    modified["config_sha256"] = "changed"
    audit.save(target / "alice/code/runtime.json", modified)
    with pytest.raises(ValueError, match="provenance"):
        provider.verify(target, run)
    (target / "alice/code/runtime.json").write_bytes(snapshot)
    # A duplicated probe must not masquerade as full seed/group coverage.
    attributes[0] = attributes[1]
    audit.save(
        target / "alice/results/private_attribute_audit.json", {"details": attributes}
    )
    with pytest.raises(ValueError, match="inventory"):
        provider.verify(target, run)
    # Source code/config failure happens before any input capability is staged.
    monkeypatch.setattr(
        vm,
        "guest",
        lambda p, c: guest(p, c) if "sha256sum" not in c else "changed unit",
    )
    with pytest.raises(ValueError, match="changed"):
        provider.stage(tmp_path, sources, tmp_path / "bad", "uci_bank_marketing")


def test_hillstrom_utility_retains_three_potentials(tmp_path, monkeypatch):
    cfg, runtime = inputs(tmp_path, monkeypatch)
    runtime["dataset"] = "hillstrom_email_marketing"
    for party in provider.vm.PARTIES:
        root = tmp_path / party / "unit"
        if party == "bob":
            pd.DataFrame(columns=["record_id", "numeric__newbie", "f1", "f2"]).to_csv(
                root / "input/train.csv", index=False
            )
        for split in audit.SPLITS:
            path = root / "input" / f"{split}_local.npz"
            with np.load(path) as z:
                rows = {k: z[k] for k in z.files}
            if party == "alice":
                rows["t"] = np.arange(60) % 3
            np.savez(path, **rows)
            if party == "alice":
                for seed in cfg["model_seeds"]:
                    for cell in cfg["output_cells"]:
                        width = 1 if split == "train" else 3
                        prediction = np.tile(
                            (0.1 + 0.4 * (rows["y"] == 1))[:, None], (1, width)
                        )
                        np.save(
                            root / "input" / f"{cell}_seed{seed}_{split}.npy",
                            prediction,
                        )
        audit.save(
            root / "input/manifest.json",
            {
                p.name: audit.sha(p)
                for p in (root / "input").iterdir()
                if p.name != "manifest.json"
            },
        )
    data = audit.owner_inputs("alice", runtime, cfg)
    assert audit.local_stress(data, cfg)
    report = json.loads(
        (tmp_path / "alice/unit/results/private_output_audit.json").read_text()
    )
    assert len(report["utility"]) == 66
    assert all(
        "arm1_centered_qini" in item
        and "arm2_centered_qini" in item
        and "mean_centered_qini_minus_exact" in item
        for item in report["utility"].values()
    )


def test_owner_lifecycle_durable_even_if_vm_is_unreachable(tmp_path, monkeypatch):
    initial = tmp_path / "initial"
    for party in provider.vm.PARTIES:
        (initial / party / "logs").mkdir(parents=True)
        (initial / party / "results").mkdir()
    success = [True]
    unavailable = [False]

    class Process:
        def __init__(self, *a, **kw):
            self.returncode = 0 if success[0] else 1
            assert "env" in a[0] and "-i" in a[0]

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

    monkeypatch.setattr(provider.subprocess, "Popen", Process)

    def operation(*a, **kw):
        if unavailable[0]:
            raise ConnectionError("unit unreachable VM")
        return "passed"

    monkeypatch.setattr(provider.vm, "guest", operation)
    monkeypatch.setattr(provider.provider, "export", operation)
    monkeypatch.setattr(
        provider, "verify", lambda *a: {"status": "unit_only_not_security_proof"}
    )
    assert provider.execute(initial, "unit")["status"] == "unit_only_not_security_proof"

    frozen = {
        str(p.relative_to(initial)): p.read_bytes()
        for p in initial.rglob("*")
        if p.is_file()
    }
    with pytest.raises(FileExistsError):
        provider.execute(initial, "unit")
    assert frozen == {
        str(p.relative_to(initial)): p.read_bytes()
        for p in initial.rglob("*")
        if p.is_file()
    }

    def new_directory(name):
        folder = tmp_path / name
        for party in provider.vm.PARTIES:
            (folder / party / "logs").mkdir(parents=True)
            (folder / party / "results").mkdir()
        return folder

    invalid = new_directory("invalid")

    def reject(*a):
        raise ValueError("unit invalid verification")

    monkeypatch.setattr(provider, "verify", reject)
    with pytest.raises(ValueError, match="invalid verification"):
        provider.execute(invalid, "unit")
    bad = json.loads((invalid / "status.json").read_text())
    assert bad["status"] == "failed" and bad["failure_phase"] == "verification"
    offline = new_directory("offline")
    success[0] = False
    unavailable[0] = True
    with pytest.raises(RuntimeError, match="Owner exit"):
        provider.execute(offline, "unit")
    receipt = json.loads((offline / "status.json").read_text())
    assert receipt["status"] == "failed" and len(receipt["cleanup_errors"]) == 4
    no_export = new_directory("no_export")
    success[0] = True
    with pytest.raises(RuntimeError, match="export incomplete"):
        provider.execute(no_export, "unit")
    assert json.loads((no_export / "status.json").read_text())["status"] == "failed"


def test_source_identifier_rejection_precedes_vm_actions(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(provider.vm, "setup_certificates", lambda *a: calls.append(a))
    with pytest.raises(ValueError, match="Invalid source"):
        provider.stage(
            tmp_path,
            {"E4": "../escape", "E5": "unit"},
            tmp_path / "out",
            "uci_bank_marketing",
        )
    assert not calls


def test_whole_guest_notebook_guard_protects_existing_results(tmp_path, monkeypatch):
    root = tmp_path / "alice" / "unit"
    for name in ("code", "data", "results", "logs", "trainings"):
        (root / name).mkdir(parents=True)
    runtime = root / "code/runtime.json"
    runtime.write_text(json.dumps({"config": str(root / "code/config.yaml")}))
    calls = []
    monkeypatch.setattr(
        audit.ft, "execute_notebook", lambda p: calls.append(p) or {"unit": True}
    )
    monkeypatch.setattr(audit.ft, "ROOT", tmp_path)
    assert audit.execute_reaudit(runtime) == {"unit": True} and len(calls) == 1
    for name in ("data", "results", "logs", "trainings"):
        file = root / name / "frozen.txt"
        file.write_bytes(b"IMMUTABLE_UNIT_FIXTURE")
        before = {
            str(p.relative_to(root)): p.read_bytes()
            for p in root.rglob("*")
            if p.is_file()
        }
        with pytest.raises(FileExistsError):
            audit.execute_reaudit(runtime)
        assert before == {
            str(p.relative_to(root)): p.read_bytes()
            for p in root.rglob("*")
            if p.is_file()
        }
        file.unlink()
    assert len(calls) == 1
