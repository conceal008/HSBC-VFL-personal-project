"""Column contract, independent update, fail-closed output and scope controls."""

from pathlib import Path
import json
import sys
import numpy as np
import pytest

COMPONENTS = Path(__file__).resolve().parents[1] / "components"
sys.path.insert(0, str(COMPONENTS))
import joint_diagnostic as jd  # noqa: E402
import joint_vm as launcher  # noqa: E402


def test_real_reference_capability_denied():
    for runtime, cfg in [
        ({}, {}),
        ({"is_smoke": True}, {"synthetic_only": True}),
        ({"is_smoke": False, "synthetic_only": True}, {"synthetic_only": True}),
    ]:
        with pytest.raises(ValueError, match="synthetic-only"):
            jd.synthetic_guard(runtime, cfg)
    jd.synthetic_guard(
        {"is_smoke": True, "synthetic_only": True}, {"synthetic_only": True}
    )


def test_explicit_joint_column_contract_and_tail():
    a = np.array([[2.0], [3.0]], dtype=np.float32)
    b = np.array([[5.0], [7.0]], dtype=np.float32)
    t = np.array([1, 2])
    expected = np.array(
        [
            [2, 5, 1, 0, 2, 0, 5, 0, 1],
            [3, 7, 0, 1, 0, 3, 0, 7, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(jd.numpy_design(a, b, t, True, padding=1), expected)
    np.testing.assert_array_equal(
        jd.numpy_design(a, b, t, False), [[2, 5, 1], [3, 7, 1]]
    )
    forced = jd.numpy_design(a, b, t, True, potential=0)
    np.testing.assert_array_equal(forced[:, 2:8], np.zeros((2, 6)))


def test_equation_known_gradient_and_unpenalized_bias():
    x = np.array([[2.0, 1.0], [0.0, 1.0]])
    y = np.array([1.0, 0.0])
    w = np.zeros((2, 1))
    np.testing.assert_allclose(
        jd.equation_update(x, y, w, 0.1, 0.5, 2), [[0.05], [0.0]]
    )
    # Zero input rows give only L2 shrinkage, including a partial real last batch.
    np.testing.assert_allclose(
        jd.equation_update(np.zeros((2, 2)), y, np.array([[2.0], [3.0]]), 0.2, 0.4, 2),
        [[1.92], [3.0]],
    )
    np.testing.assert_array_equal(w, np.zeros((2, 1)))


def test_record_rejects_numerical_failure_and_keeps_evidence(tmp_path):
    root = tmp_path / "trainings/fit/results"
    root.mkdir(parents=True)
    data = {
        "root": str(tmp_path),
        "cfg": {"diagnostic": {"reference_tolerance": 0.001}},
    }
    for split in jd.SPLITS:
        np.save(root / f"{split}_reference_predictions.npy", np.array([[0.5], [0.5]]))
    with pytest.raises(ValueError, match="blocked"):
        jd.record(data, "fit", {s: np.array([[0.8], [0.5]]) for s in jd.SPLITS})
    assert json.loads((root / "numerical_check.json").read_text())["passed"] is False
    with pytest.raises(ValueError, match="shape"):
        jd.record(data, "fit", {s: np.array([[np.nan], [0.5]]) for s in jd.SPLITS})


def test_seed_intervals_are_complete_without_business_claim(tmp_path):
    names = []
    cfg = {
        "shared": {"learning_rates": [0.1]},
        "diagnostic": {
            "bootstrap_seed": 1,
            "bootstrap_repeats": 100,
            "scope": "synthetic test",
        },
    }
    (tmp_path / "results").mkdir()
    for seed in [11, 22, 33, 44, 55]:
        name = f"fit{seed}"
        names.append(name)
        folder = tmp_path / "trainings" / name
        (folder / "code").mkdir(parents=True)
        (folder / "results").mkdir()
        (folder / "code/training_config.json").write_text(
            json.dumps(
                {
                    "initialization_seed": seed,
                    "order_seed": seed + 100,
                    "learning_rate": 0.1,
                }
            )
        )
        (folder / "results/numerical_check.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "passed": True,
                    "splits": {s: {"max_abs_error": seed / 1000000} for s in jd.SPLITS},
                }
            )
        )
    jd.finalize(
        {"root": str(tmp_path), "cfg": cfg, "runtime": {"dataset": "synthetic"}}, names
    )
    report = json.loads((tmp_path / "results/joint_math_report.json").read_text())
    assert not report["business_gain_evaluated"]
    assert all(
        s["seed_count"] == 5
        and s["seed_mean_ci_95"][0] <= s["mean"] <= s["seed_mean_ci_95"][1]
        for s in report["summary"]
    )


def test_vm_audit_refuses_failed_round(tmp_path):
    (tmp_path / "status.json").write_text('{"status":"failed"}')
    with pytest.raises(ValueError, match="not passed"):
        launcher.verify(tmp_path)


def test_prepare_contract_owns_features_and_has_no_test_capability(
    tmp_path, monkeypatch
):
    import yaml
    import functional_training as ft
    import active_diagnostic as ad
    import functional_vm as vm

    cfg = yaml.safe_load(
        (COMPONENTS.parent / "configs/joint_numerical.yaml").read_text()
    )
    monkeypatch.setattr(ft, "ROOT", tmp_path)
    monkeypatch.setattr(
        ft, "isolation_preflight", lambda _: {"checked_by_integration": True}
    )
    monkeypatch.setattr(ad, "manifest_check", lambda _: True)
    for party in vm.PARTIES:
        root = tmp_path / party / "new"
        for sub in ("input", "data", "logs"):
            (root / sub).mkdir(parents=True)
        frames = vm.synthetic_inputs(cfg, "hillstrom_email_marketing", party)
        for split in jd.SPLITS:
            frames[split].to_csv(root / "input" / f"{split}.csv", index=False)
        runtime = {
            "run_id": "new",
            "dataset": "hillstrom_email_marketing",
            "is_smoke": True,
            "synthetic_only": True,
        }
        data = jd.prepare(party, runtime, cfg)
        assert data["width"] == 5 and "test" not in data
        assert ("y" in data["train"]) == (party == "alice")
        assert ad.shape(data)["train"] == [513, 5]
        with pytest.raises(ValueError, match="locked"):
            ad.chunk(data, "test", "x", 0, 1)


@pytest.mark.parametrize("uplift", [False, True])
@pytest.mark.parametrize("party", ["alice", "bob"])
def test_complete_joint_routing_and_export_audit(tmp_path, monkeypatch, uplift, party):
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
        (COMPONENTS.parent / "configs/joint_numerical.yaml").read_text()
    )
    cfg["shared"].update(
        batch_size=8, infeed_rows=16, epochs=2, prediction_batch_size=8
    )
    cfg["smoke"]["split_rows"] = [17, 9, 7]
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
        monkeypatch.setattr(jd, name, lambda *a: True)
    jd.train_joint(runtime)
    assert releases and set(releases) == {"alice"}
    assert not list((tmp_path / "bob/new").rglob("*predictions.npy"))
    report = json.loads(
        (tmp_path / "alice/new/results/joint_math_report.json").read_text()
    )
    assert len(report["rows"]) == 10 and all(row["passed"] for row in report["rows"])
    host = tmp_path / "host"
    host.mkdir()
    (host / "status.json").write_text('{"status":"passed"}')
    digest = launcher.sha(tmp_path / "alice/new/code/config.yaml")
    (host / "declaration.json").write_text(
        json.dumps(
            {"synthetic_only": True, "old_test_read": False, "config_sha256": digest}
        )
    )
    for owner in vm.PARTIES:
        own = host / owner
        shutil.copytree(
            tmp_path / owner / "new",
            own,
            ignore=shutil.ignore_patterns("*.npz", "model.share"),
        )
        (own / "code/joint_numerical.yaml").write_bytes(
            (own / "code/config.yaml").read_bytes()
        )
        (own / "code/runtime.json").write_text(
            json.dumps({"config_sha256": digest, "code_sha256": {}})
        )
        (own / "logs/output_boundary.json").write_text('{"old_test_read":false}')
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
            (fit / "logs/progress.json").write_text(
                json.dumps(
                    {
                        "completed_epochs": cfg["shared"]["epochs"],
                        "planned_epochs": cfg["shared"]["epochs"],
                    }
                )
            )
    vm.private_permissions(host)
    assert launcher.verify(host)["fits"] == 10
    file = host / "alice/results/joint_math_report.json"
    bad = json.loads(file.read_text())
    bad["rows"][0]["passed"] = False
    file.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Numerical failure"):
        launcher.verify(host)


def test_stage_contract_and_launcher_lifecycle(tmp_path, monkeypatch):
    import yaml
    import functional_vm as vm
    import active_vm as av

    copied = []
    commands = []

    def skeleton(workspace, dataset, run, local, smoke):
        assert smoke is True
        for owner in vm.PARTIES:
            (local / owner / "code").mkdir(parents=True)
            (local / owner / "logs").mkdir()
            (local / owner / "code/runtime.json").write_text(
                json.dumps({"is_smoke": True, "git_sha": "unit", "parties": {}})
            )
        (local / "declaration.json").write_text(
            '{"prepared_experiment":"synthetic_only"}'
        )

    monkeypatch.setattr(vm, "stage_round", skeleton)
    monkeypatch.setattr(vm, "copy_to", lambda *a: copied.append(a))
    monkeypatch.setattr(vm, "guest", lambda *a: commands.append(a) or "")
    root = tmp_path / "stage"
    root.mkdir()
    launcher.stage(tmp_path, "uci_bank_marketing", "new", root)
    assert copied and commands
    cfg = yaml.safe_load((root / "alice/code/joint_numerical.yaml").read_text())
    assert (
        cfg["synthetic_only"]
        and json.loads((root / "bob/code/runtime.json").read_text())["synthetic_only"]
    )
    with pytest.raises(ValueError, match="declared"):
        launcher.stage(tmp_path, "unregistered_task", "new", tmp_path / "bad")
    (tmp_path / "实验日志").mkdir()
    (tmp_path / "实验日志/实验索引.md").write_text("")
    monkeypatch.setattr(
        launcher,
        "stage",
        lambda workspace, dataset, run, local: skeleton(
            workspace, dataset, run, local, True
        ),
    )
    exports = []
    monkeypatch.setattr(
        av,
        "export_artifacts",
        lambda local, run, owner, failed=False: exports.append((owner, failed)),
    )
    monkeypatch.setattr(
        launcher, "verify", lambda _: {"verification": "unit lifecycle only"}
    )
    for success in [True, False]:

        class Process:
            def __init__(self, *a, **kw):
                self.returncode = 0 if success else 1

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

        monkeypatch.setattr(launcher.subprocess, "Popen", Process)
        if success:
            local = launcher.launch(tmp_path, "uci_bank_marketing")
            assert json.loads((local / "status.json").read_text())["status"] == "passed"
        else:
            with pytest.raises(RuntimeError, match="exit failure"):
                launcher.launch(tmp_path, "uci_bank_marketing")
    assert len(exports) == 4 and exports[-1][1]
