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
import nonlinear_pilot as pilot  # noqa: E402
import nonlinear_vm as provider  # noqa: E402


def test_sampling_does_not_depend_on_outcomes():
    frame = pd.DataFrame(
        {"record_id": [str(i) for i in range(20)], "label": np.arange(20) % 2}
    )
    selected = pilot.select_records(frame, 7, 11)["record_id"].tolist()
    frame["label"] = 1 - frame["label"]
    assert selected == pilot.select_records(frame, 7, 11)["record_id"].tolist()
    cfg = yaml.safe_load(
        (COMPONENTS.parent / "configs/nonlinear_pilot.yaml").read_text()
    )
    shuffled = frame.sample(frac=1, random_state=cfg["experiment"]["selection_seed"])
    assert selected == pilot.select_records(shuffled, 7, 11)["record_id"].tolist()


def test_training_only_basis_constant_and_extreme():
    cfg = yaml.safe_load(
        (COMPONENTS.parent / "configs/nonlinear_pilot.yaml").read_text()
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
    import active_diagnostic as ad

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
def test_complete_pilot_routing_and_export_audit(tmp_path, monkeypatch, uplift, party):
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
        (COMPONENTS.parent / "configs/nonlinear_pilot.yaml").read_text()
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
        "uci_bank_marketing": ["numeric__f0"],
        "hillstrom_email_marketing": ["numeric__f0", "numeric__f1"],
    }
    dataset = "hillstrom_email_marketing" if uplift else "uci_bank_marketing"
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
    pilot.train_pilot(runtime)
    assert releases and set(releases) == {"alice"}
    assert not list((tmp_path / "bob/new").rglob("*predictions.npy"))
    report = json.loads(
        (tmp_path / "alice/new/results/private_feature_evaluation.json").read_text()
    )
    assert len(report["models"]) == 8 and len(report["paired_comparisons"]) == 5
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
        (own / "code/nonlinear_pilot.yaml").write_bytes(
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
    assert provider.verify(host)["secure_fits"] == 20
    file = host / "alice/results/private_feature_evaluation.json"
    bad = json.loads(file.read_text())
    bad["models"]["A0"][report["primary"]]["ci_low"] = float("nan")
    file.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Interval failed"):
        provider.verify(host)


def test_stage_never_reads_test_and_rejects_changed_source(tmp_path, monkeypatch):
    vm = provider.vm
    copied, commands = [], []
    monkeypatch.setattr(
        vm, "setup_certificates", lambda _: {"alice": "unit-a", "bob": "unit-b"}
    )
    monkeypatch.setattr(vm, "restrict_network", lambda *a: None)
    monkeypatch.setattr(vm, "copy_to", lambda *a: copied.append(a))
    monkeypatch.setattr(vm, "guest", lambda *a: commands.append(a) or "")
    dataset = "uci_bank_marketing"
    prepared = vm.PREPARED[dataset]
    for party in vm.PARTIES:
        source = tmp_path / "联邦隔离产物" / prepared / party
        source.mkdir(parents=True)
        manifest = {}
        for split in ["train", "validation"]:
            path = source / f"{split}.csv"
            path.write_text("unit_fixture_only\n0\n")
            manifest[path.name] = provider.sha(path)
        result = tmp_path / "联邦隔离结果" / prepared / party
        result.mkdir(parents=True)
        (result / "data_manifest.json").write_text(json.dumps(manifest))
    root = tmp_path / "stage"
    provider.stage(tmp_path, dataset, "unit", root)
    assert all(Path(args[0]).name != "test.csv" for args in copied)
    assert any("chown root:root" in command for _, command in commands)
    for party in vm.PARTIES:
        runtime = json.loads((root / party / "code/runtime.json").read_text())
        assert runtime["party"] == party and runtime["is_smoke"] is False
        assert set(
            json.loads((root / party / "data/input_manifest.json").read_text())
        ) == {"train.csv", "validation.csv"}
    (tmp_path / "联邦隔离产物" / prepared / "alice/train.csv").write_text("changed")
    with pytest.raises(ValueError, match="Frozen input changed"):
        provider.stage(tmp_path, dataset, "bad", tmp_path / "bad")
    with pytest.raises(ValueError, match="Undeclared"):
        provider.stage(tmp_path, "unknown", "bad", tmp_path / "unknown")


def test_export_excludes_predictions_and_lifecycle(tmp_path, monkeypatch):
    import tarfile
    import io

    vm = provider.vm
    commands = []
    monkeypatch.setattr(vm, "guest", lambda *a: commands.append(a) or "")

    def lima(*args):
        assert args[0] == "copy"
        with tarfile.open(args[-1], "w") as archive:
            info = tarfile.TarInfo("results/authorized.json")
            content = b'{"unit":true}'
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))

    monkeypatch.setattr(vm, "lima", lima)
    (tmp_path / "alice").mkdir()
    provider.export(tmp_path, "unit", "alice")
    assert (tmp_path / "alice/results/authorized.json").is_file()
    command = commands[0][1]
    assert all(
        f"--exclude='{pattern}'" in command
        for pattern in ["model.share", "*.npy", "*.npz", "*.trace.log"]
    )

    def stage(workspace, dataset, run, local):
        for owner in vm.PARTIES:
            (local / owner / "logs").mkdir(parents=True)

    monkeypatch.setattr(provider, "stage", stage)
    exports = []
    monkeypatch.setattr(
        provider,
        "export",
        lambda local, run, party, failed=False: exports.append((party, failed)),
    )
    monkeypatch.setattr(
        provider, "verify", lambda _: {"verification": "unit lifecycle only"}
    )
    (tmp_path / "实验日志").mkdir()
    (tmp_path / "实验日志/实验索引.md").write_text("")
    for success in [True, False]:

        class Process:
            def __init__(self, *a, **kw):
                self.returncode = 0 if success else 1

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

        monkeypatch.setattr(provider.subprocess, "Popen", Process)
        if success:
            local = provider.launch(tmp_path, "uci_bank_marketing")
            assert json.loads((local / "status.json").read_text())["exit_codes"] == [
                0,
                0,
            ]
        else:
            with pytest.raises(RuntimeError, match="exit failure"):
                provider.launch(tmp_path, "uci_bank_marketing")
    assert len(exports) == 4 and exports[-1][1]
