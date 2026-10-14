"""Synthetic control tests; actual data/MPC acceptance is a separate private audit."""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import pytest
import yaml

COMPONENTS = Path(__file__).resolve().parents[1] / "components"
sys.path.insert(0, str(COMPONENTS))
import portfolio_training as pilot  # noqa: E402
import portfolio_training_vm as provider  # noqa: E402


def test_sampling_does_not_depend_on_outcomes():
    frame = pd.DataFrame(
        {"record_id": [str(i) for i in range(20)], "label": np.arange(20) % 2}
    )
    selected = pilot.select_records(frame, 7, 11)["record_id"].tolist()
    frame["label"] = 1 - frame["label"]
    assert selected == pilot.select_records(frame, 7, 11)["record_id"].tolist()
    cfg = yaml.safe_load(
        (COMPONENTS.parent / "configs/portfolio_training.yaml").read_text()
    )
    shuffled = frame.sample(frac=1, random_state=cfg["experiment"]["selection_seed"])
    assert selected == pilot.select_records(shuffled, 7, 11)["record_id"].tolist()


def test_training_only_basis_constant_and_extreme():
    cfg = yaml.safe_load(
        (COMPONENTS.parent / "configs/portfolio_training.yaml").read_text()
    )
    train = pd.DataFrame({"age": [0.0, 1.0, 2.0, 3.0], "constant": np.ones(4)})
    frames = {
        "train": train,
        "validation": pd.DataFrame({"age": [999.0], "constant": [1.0]}),
    }
    extra, metadata = pilot.transform_basis(train, frames, ["age", "constant"], cfg)
    assert len(metadata) == 3 and all(m["feature"] == "age" for m in metadata)
    assert (
        np.isfinite(extra["validation"]).all()
        and np.abs(extra["validation"]).max() <= cfg["shared"]["clip_features"]
    )
    _, other = pilot.transform_basis(
        train,
        {
            "train": train,
            "validation": pd.DataFrame({"age": [-999.0], "constant": [1.0]}),
        },
        ["age"],
        cfg,
    )
    assert metadata == other


@pytest.mark.parametrize("uplift", [False, True])
def test_initializer_preserves_common_coordinates(uplift):
    import functional_training as ft
    import portfolio_math as ad

    base = pilot.initial_weights(2, 3, 0, uplift, 11, 0.01, True)
    enhanced = pilot.initial_weights(2, 3, 2, uplift, 11, 0.01, True)
    # Original A/B main coordinates and bias must stay identical.
    np.testing.assert_array_equal(
        enhanced[[0, 1, 4, 5, 6, -1]], base[[0, 1, 2, 3, 4, -1]]
    )
    np.testing.assert_array_equal(enhanced[2:4], np.zeros((2, 1)))
    indices, _ = ad.mapped_indices(4, 3, uplift)
    np.testing.assert_array_equal(
        pilot.initial_weights(2, 3, 2, uplift, 11, 0.01, False), enhanced[indices]
    )
    assert base.shape == ft.initial_weights(base.shape[0], 11, 0.01).shape


def test_test_capability_denied_and_statistics_controls():
    with pytest.raises(ValueError, match="denied"):
        pilot.feature_chunk({}, "test", 0, 1, None, False, True)
    assert pilot.exact_seed_p([1, 1, 1, 1, 1]) == 0.0625
    assert pilot.holm_adjust({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    assert pilot.exact_seed_p([0, 0, 0, 0, 0]) == 1.0


@pytest.mark.parametrize("uplift", [False, True])
@pytest.mark.parametrize("party", ["alice", "bob"])
def test_complete_full_routing_and_export_audit(tmp_path, monkeypatch, uplift, party):
    # These are routing doubles, never an MPC or isolation security proof.
    from types import ModuleType, SimpleNamespace
    import shutil
    import yaml
    from test_active_diagnostic import native_doubles
    import active_diagnostic as ad
    import functional_training as ft
    import functional_vm as vm

    native_doubles(monkeypatch)
    cfg = yaml.safe_load(
        (COMPONENTS.parent / "configs/portfolio_training.yaml").read_text()
    )
    cfg["shared"].update(
        batch_size=8, infeed_rows=16, epochs=2, prediction_batch_size=8
    )
    cfg["smoke"]["split_rows"] = [17, 9, 7]
    cfg["smoke"]["features_per_party"] = 5
    cfg["shared"].update(top_k_fraction=1.0, bootstrap_repeats=20, tree_iterations=2)
    for spec in cfg["datasets"].values():
        spec.update(alice_features=5, bob_features=5)
    cfg["experiment"]["features"] = {
        "homecredit": ["numeric__f0"],
        "criteo": ["numeric__f0", "numeric__f1"],
    }
    dataset = "criteo" if uplift else "homecredit"
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
        monkeypatch.setattr(pilot, name, lambda *a: True)
    pilot.train_full(runtime)
    assert releases and set(releases) == {"alice"}
    assert not list((tmp_path / "bob/new").rglob("*predictions.npy"))
    report = json.loads(
        (tmp_path / "alice/new/results/private_feature_evaluation.json").read_text()
    )
    assert len(report["models"]) == 4 and len(report["paired_comparisons"]) == 1
    host = tmp_path / "host"
    host.mkdir()
    (host / "status.json").write_text('{"status":"passed"}')
    digest = provider.sha(tmp_path / "alice/new/code/config.yaml")
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
        (own / "code/nonlinear_full.yaml").write_bytes(
            (own / "code/config.yaml").read_bytes()
        )
        (own / "code/runtime.json").write_text(
            json.dumps({"config_sha256": digest, "code_sha256": {}})
        )
        (own / "logs/output_boundary.json").write_text(
            '{"old_test_read":false,"scores_receiver":"alice","frozen_input_unchanged":true}'
        )
        (own / "logs/tls_checks.json").write_text('{"unit_stub_not_tls_proof":true}')
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
    assert provider.verify_inventory(host)["secure_fits"] == 10
    file = host / "alice/results/private_feature_evaluation.json"
    bad = json.loads(file.read_text())
    bad["models"]["A0"][report["primary"]]["ci_low"] = float("nan")
    file.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Interval failed"):
        provider.verify_inventory(host)


def test_full_partition_has_no_sampling_and_rejects_cap():
    frame = pd.DataFrame(
        {"record_id": [str(i) for i in range(19)], "label": np.arange(19) % 2}
    )
    cfg = yaml.safe_load(
        (COMPONENTS.parent / "configs/portfolio_training.yaml").read_text()
    )
    retained = pilot.select_records(frame, None, cfg["experiment"]["selection_seed"])
    assert len(retained) == len(frame) and set(retained["record_id"]) == set(
        frame["record_id"]
    )
    cfg["experiment"]["max_rows"]["train"] = 1
    with pytest.raises(ValueError, match="forbids row caps"):
        pilot.prepare("alice", {}, cfg)


def test_private_clusters_are_evaluation_only_and_order_aligned(tmp_path, monkeypatch):
    import hashlib
    import functional_training as ft
    import active_diagnostic as ad

    cfg = yaml.safe_load(
        (COMPONENTS.parent / "configs/portfolio_training.yaml").read_text()
    )
    dataset = "fedads"
    runtime = {"party": "alice", "run_id": "unit", "dataset": dataset}
    monkeypatch.setattr(ft, "ROOT", tmp_path)
    monkeypatch.setattr(ft, "isolation_preflight", lambda _: {"unit": True})
    monkeypatch.setattr(ad, "manifest_check", lambda _: True)
    root = tmp_path / "alice/unit"
    for sub in ["input", "data", "logs", "results"]:
        (root / sub).mkdir(parents=True)
    mapping = {}
    for split in ["train", "validation"]:
        keys = [split + str(i) for i in range(12)]
        pd.DataFrame(
            {"record_id": keys, "numeric__x": np.arange(12), "label": np.arange(12) % 2}
        ).to_csv(root / "input" / f"{split}.csv", index=False)
        mapping.update({key: "group" + str(i // 2) for i, key in enumerate(keys)})
    path = root / "input/owner_groups.json"
    path.write_text(json.dumps(mapping))
    runtime["group_map_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    data = pilot.prepare("alice", runtime, cfg)
    assert data["width"] == 1 and data["validation"]["x"].shape[1] == 1
    ordered = pilot.select_records(
        pd.DataFrame({"record_id": list(mapping)[12:]}),
        None,
        cfg["experiment"]["selection_seed"],
    )["record_id"].tolist()
    assert data["validation"]["clusters"].tolist() == [mapping[k] for k in ordered]
    with np.load(root / "data/validation_local.npz") as saved:
        assert "clusters" not in saved.files
    runtime["group_map_sha256"] = "changed"
    with pytest.raises(ValueError, match="metadata changed"):
        pilot.prepare("alice", runtime, cfg)


def test_uncentered_auuc_matches_independent_ipw_gain_integral():
    from scipy.integrate import trapezoid

    y = np.array([0, 1, 1, 0, 1, 0, 1, 0], dtype=float)
    t = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    p = np.c_[np.linspace(0.1, 0.5, len(y)), np.linspace(0.9, 0.2, len(y))]
    cfg = yaml.safe_load(
        (COMPONENTS.parent / "configs/portfolio_training.yaml").read_text()
    )
    spec = cfg["datasets"]["criteo"]
    shared = cfg["shared"]
    shared["top_k_fraction"] = 1.0
    metrics = pilot.metric_values(y, p, spec, shared, t)
    order = np.argsort(-(p[:, 1] - p[:, 0]), kind="stable")
    prop = t.mean()
    curve = np.r_[0, np.cumsum((y * (t / prop - (1 - t) / (1 - prop)))[order]) / len(y)]
    assert metrics["arm1_auuc"] == pytest.approx(
        trapezoid(curve, np.linspace(0, 1, len(curve)))
    )
    assert metrics["mean_centered_qini"] == metrics["arm1_centered_qini"]
