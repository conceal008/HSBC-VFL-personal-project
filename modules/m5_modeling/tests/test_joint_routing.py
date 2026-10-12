"""Synthetic execution routing only, never a claim of native MPC or isolation."""

from pathlib import Path
import json
import sys
import numpy as np
import pytest
import yaml

COMPONENTS = Path(__file__).resolve().parents[1] / "components"
sys.path.insert(0, str(COMPONENTS))
import joint_budget as jb  # noqa: E402
import joint_budget_vm as jv  # noqa: E402


@pytest.mark.parametrize("uplift", [False, True])
@pytest.mark.parametrize("party", ["alice", "bob"])
def test_complete_full_routing_and_export_audit(tmp_path, monkeypatch, uplift, party):
    # These are routing doubles, never an MPC or isolation security proof.
    from types import ModuleType, SimpleNamespace
    import shutil
    from test_active_diagnostic import native_doubles
    import active_diagnostic as ad
    import functional_training as ft
    import functional_vm as vm

    native_doubles(monkeypatch)
    cfg = yaml.safe_load((COMPONENTS.parent / "configs/joint_budget.yaml").read_text())
    cfg["shared"].update(batch_size=8, infeed_rows=16, prediction_batch_size=8)
    cfg["smoke"]["split_rows"] = [17, 9, 7]
    cfg["smoke"]["features_per_party"] = 5
    cfg["shared"].update(top_k_fraction=1.0, bootstrap_repeats=20, tree_iterations=2)
    for spec in cfg["datasets"].values():
        spec.update(alice_features=5, bob_features=5)
    cfg["experiment"]["features"] = {
        "uci_bank_marketing": ["numeric__f0"],
        "hillstrom_email_marketing": ["numeric__f0", "numeric__f1"],
    }
    dataset = "hillstrom_email_marketing" if uplift else "uci_bank_marketing"
    for group in cfg["experiment"]["groups"].values():
        group.update(
            bob_keep=["numeric__f0", "numeric__f1"],
            alice_cross=["numeric__f0", "numeric__f1"],
            bob_cross=["numeric__f0", "numeric__f1"],
        )

    monkeypatch.setattr(ft, "ROOT", tmp_path)
    monkeypatch.setattr(ft, "isolation_preflight", lambda _: {"unit_preflight": True})

    def manifest(path):
        assert isinstance(path, Path), "Path contract, including final audit"
        return True

    monkeypatch.setattr(ad, "manifest_check", manifest)
    for owner in vm.PARTIES:
        root = tmp_path / owner / "new"
        for sub in ["input", "code", "data", "results", "logs", "trainings"]:
            (root / sub).mkdir(parents=True)
        for split, frame in vm.synthetic_inputs(cfg, dataset, owner).items():
            if split == "test":
                continue
            if owner == "alice":
                frame["label"] = np.arange(len(frame)) % 2
            frame.to_csv(root / "input" / f"{split}.csv", index=False)
        (root / "input/manifest.json").write_text("{}")
        (root / "code/config.yaml").write_text(yaml.safe_dump(cfg))
    runtime = {
        "party": party,
        "run_id": "new",
        "dataset": dataset,
        "config": str(tmp_path / party / "new/code/config.yaml"),
        "is_smoke": True,
        "synthetic_only": True,
        "parties": {"alice": {"address": "unit:1"}, "bob": {"address": "unit:2"}},
        "spu_addresses": {"alice": "unit:3", "bob": "unit:4"},
    }
    for owner in vm.PARTIES:
        own_data = jb.nf.prepare(owner, runtime, cfg)
        root = tmp_path / owner / "new"
        (root / "input/references").mkdir()
        for split in jb.SPLITS:
            np.savez(
                root / "input/references" / (split + "_local.npz"), **own_data[split]
            )
            if owner == "alice":
                for seed in cfg["shared"]["model_seeds"]:
                    for cell in cfg["experiment"]["reference_cells"]:
                        cols = 3 if uplift and split == "validation" else 1
                        prediction = np.tile(
                            (0.25 + 0.5 * own_data[split]["y"])[:, None], (1, cols)
                        )
                        np.save(
                            root
                            / "input/references"
                            / f"{cell}_seed{seed}_{split}.npy",
                            prediction,
                        )
    releases = []

    class Object:
        def __init__(self, value, owner):
            self.value, self.owner = value, owner

        def to(self, recipient):
            if self.owner == "secure" and recipient.name != "secure":
                releases.append(recipient.name)
            return Object(self.value, recipient.name)

    def unwrap(value):
        if isinstance(value, Object):
            return value.value
        if isinstance(value, dict):
            return {k: unwrap(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(unwrap(v) for v in value)
        return value

    class Plain:
        def __init__(self, name):
            self.name = name

        def __call__(self, fn, **opts):
            def call(*args, **kw):
                value = fn(*unwrap(args), **unwrap(kw))
                if opts.get("user_specified_num_returns") == 2:
                    return tuple(Object(v, self.name) for v in value)
                return Object(value, self.name)

            return call

    class Secure(Plain):
        def __init__(self, *a, **kw):
            self.name = "secure"

        def dump(self, value, paths):
            for path in paths:
                Path(path).write_text("UNIT_DOUBLE_NOT_SECRET_SHARE")

    sf = ModuleType("secretflow")
    sf.PYU = Plain
    sf.SPU = Secure
    sf.init = lambda **k: None
    sf.wait = lambda v: None
    sf.shutdown = lambda **k: None

    def reveal(obj):
        value = unwrap(obj)
        assert not isinstance(value, np.ndarray), "Driver must not reveal arrays"
        return value

    sf.reveal = reveal
    monkeypatch.setitem(sys.modules, "secretflow", sf)
    device = ModuleType("secretflow.device")
    device.SPUCompilerNumReturnsPolicy = SimpleNamespace(FROM_USER="USER")
    monkeypatch.setitem(sys.modules, device.__name__, device)
    spu = ModuleType("spu")
    spu.logging = SimpleNamespace(LogOptions=SimpleNamespace)
    monkeypatch.setitem(sys.modules, "spu", spu)
    for name in [
        "install_pinned_tls_adapter",
        "reject_anonymous_client",
        "reject_anonymous_spu",
    ]:
        monkeypatch.setattr(jb, name, lambda *a: True)
    original_checkpoint = jb.record_checkpoint

    def routing_checkpoint(
        data, name, epoch, predictions, cell, seed, tolerance, gradient_small
    ):
        # This fixture tests owner routing, not the numeric 12-epoch acceptance.
        # Independent rejection/numerical tests are in test_joint_budget.py.
        if epoch == 12:
            ref = Path(data["root"]) / "input/references"
            for split, value in predictions.items():
                np.save(
                    ref / f"{cell.removeprefix('D_')}_seed{seed}_{split}.npy", value
                )
        return original_checkpoint(
            data, name, epoch, predictions, cell, seed, tolerance, gradient_small
        )

    monkeypatch.setattr(jb, "record_checkpoint", routing_checkpoint)
    jb.train_full(runtime)
    assert releases and set(releases) == {"alice"}
    assert not list((tmp_path / "bob/new").rglob("*predictions.npy"))
    report = json.loads(
        (tmp_path / "alice/new/results/private_feature_evaluation.json").read_text()
    )
    assert len(report["models"]) == 20 and len(report["paired_comparisons"]) == len(
        cfg["experiment"]["comparisons"]
    )
    host = tmp_path / "host"
    host.mkdir()
    (host / "status.json").write_text('{"status":"passed"}')
    digest = jv.sha(tmp_path / "alice/new/code/config.yaml")
    (host / "declaration.json").write_text(
        json.dumps(
            {
                "test_staged": False,
                "prediction_export": False,
                "dataset": dataset,
                "config_sha256": digest,
            }
        )
    )
    for owner in vm.PARTIES:
        own = host / owner
        shutil.copytree(
            tmp_path / owner / "new",
            own,
            ignore=shutil.ignore_patterns("*.npz", "*.npy", "model.share"),
        )
        (own / "code/joint_budget.yaml").write_bytes(
            (own / "code/config.yaml").read_bytes()
        )
        (own / "code/runtime.json").write_text(
            json.dumps({"config_sha256": digest, "code_sha256": {}})
        )
        (own / "logs/output_boundary.json").write_text(
            '{"old_test_read":false,"scores_receiver":"alice","frozen_input_unchanged":true}'
        )
        (own / "logs/tls_checks.json").write_text('{"unit_stub_not_tls_proof":true}')
        (own / "logs/cross_control.json").write_text('{"native_known_answer":true}')
        (own / "results/party.executed.ipynb").write_text(
            json.dumps(
                {
                    "cells": [
                        {"cell_type": "code", "execution_count": i, "outputs": []}
                        for i in [1, 2, 3]
                    ]
                }
            )
        )
        (own / "results/NOTEBOOK_COMPLETED.json").write_text('{"status":"passed"}')
        for fit in (own / "trainings").iterdir():
            (fit / "code/environment.lock").write_text("unit fixture only")
            (fit / "data/input_references.json").write_text(
                json.dumps(
                    {
                        "frozen_input": {"train.csv": "unit", "validation.csv": "unit"},
                        "derived_input": {},
                    }
                )
            )
            (fit / "logs/progress.json").write_text(
                json.dumps(
                    {
                        "completed_epochs": cfg["shared"]["epochs"],
                        "planned_epochs": cfg["shared"]["epochs"],
                    }
                )
            )
    vm.private_permissions(host)
    assert jv.verify(host)["secure_fits"] == 20
    file = host / "alice/results/private_feature_evaluation.json"
    bad = json.loads(file.read_text())
    bad["models"]["D_A0_e48"][report["primary"]]["ci_low"] = float("nan")
    file.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Interval failed"):
        jv.verify(host)
