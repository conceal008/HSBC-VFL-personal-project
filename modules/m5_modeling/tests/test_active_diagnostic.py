"""Synthetic routing doubles; real MPC/numerical acceptance is separate."""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

COMPONENTS = Path(__file__).resolve().parents[1] / "components"
sys.path.insert(0, str(COMPONENTS))
import active_diagnostic as active  # noqa: E402
import active_vm as launcher  # noqa: E402
import functional_training as ft  # noqa: E402
import functional_vm as vm  # noqa: E402


def native_doubles(monkeypatch):
    jax = ModuleType("jax")
    jax.numpy, jax.jit = np, lambda fn: fn
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", np)

    class Sig(Enum):
        DF = "df"

    sig = ModuleType("secretflow.utils.sigmoid")
    sig.SigType = Sig
    sig.sigmoid = lambda x, _: 0.5 * x / (1 + np.abs(x)) + 0.5
    monkeypatch.setitem(sys.modules, sig.__name__, sig)
    linear = ModuleType("secretflow.ml.linear.linear_model")
    linear.RegType = SimpleNamespace(Logistic="LR")
    monkeypatch.setitem(sys.modules, linear.__name__, linear)
    model = ModuleType("secretflow.ml.linear.ss_sgd.model")
    model.Penalty = SimpleNamespace(L2="L2")
    model.Strategy = SimpleNamespace(NAIVE_SGD="SGD")

    def batch(
        x,
        y,
        w,
        lr,
        l2,
        sig_type=None,
        reg_type=None,
        penalty=None,
        total_batch=None,
        batch_size=None,
        strategy=None,
        dk_arr=None,
        enable_spu_cache=False,
    ):
        w = w.copy()
        for i in range(total_batch):
            xx = x[i * batch_size : (i + 1) * batch_size]
            yy = y[i * batch_size : (i + 1) * batch_size, None]
            prediction = sig.sigmoid(xx @ w, sig_type)
            regularization = w.copy()
            regularization[-1] = 0
            w -= lr * (
                (xx.T @ (prediction - yy)) / batch_size
                + l2 * regularization / batch_size
            )
        return w, None

    model._batch_update_w = batch
    monkeypatch.setitem(sys.modules, model.__name__, model)
    return batch


def make_fixture(tmp_path, monkeypatch, uplift):
    cfg = yaml.safe_load(
        (COMPONENTS.parent / "configs/local_functional.yaml").read_text()
    )
    cfg["shared"].update(
        model_seeds=[11, 22, 33, 44, 55],
        learning_rates=[0.1, 0.3],
        batch_size=12,
        infeed_rows=24,
        epochs=2,
        bootstrap_repeats=20,
        top_k_fraction=0.5,
        prediction_batch_size=24,
    )
    cfg["diagnostic"] = {"reference_tolerance": 0.001}
    monkeypatch.setattr(ft, "ROOT", tmp_path)
    monkeypatch.setattr(ft, "isolation_preflight", lambda _: {"nonroot": True})
    dataset = "hillstrom_email_marketing" if uplift else "uci_bank_marketing"
    for party in vm.PARTIES:
        root = tmp_path / party / "new"
        origin = tmp_path / party / "origin"
        for folder in ["input", "code", "data", "results", "logs", "trainings"]:
            (root / folder).mkdir(parents=True)
        (origin / "data").mkdir(parents=True)
        manifest = {}
        rng = np.random.default_rng(123)
        for split in ft.SPLITS:
            n = 60
            frame = pd.DataFrame(
                {
                    "record_id": [f"{split}{i}" for i in range(n)],
                    "numeric__a": rng.normal(size=n),
                    "numeric__b": rng.normal(size=n),
                }
            )
            if party == "alice":
                frame["label"] = np.tile([0, 1], n // 2)
                if uplift:
                    frame["treatment"] = np.tile([0, 1, 2], n // 3)
            path = root / "input" / f"{split}.csv"
            frame.to_csv(path, index=False)
            manifest[path.name] = ft.fingerprint(path)
        (root / "input/manifest.json").write_text(json.dumps(manifest))
        (root / "code/source.py").write_text("# synthetic")
    original_open = Path.open

    def readonly(path, mode="r", *args, **kwargs):
        if "input" in path.parts and mode == "r+b":
            raise PermissionError("immutable fixture")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", readonly)
    # Populate origin's immutable arrays using the existing preprocessing path.
    for party in vm.PARTIES:
        data = ft.prepare_local(party, "new", dataset, cfg)
        for split in ft.SPLITS:
            np.savez_compressed(
                tmp_path / party / "origin/data" / f"{split}_local.npz", **data[split]
            )
        for path in (tmp_path / party / "new/data").iterdir():
            path.unlink()
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    runtime = {
        "party": "alice",
        "run_id": "new",
        "origin_run_id": "origin",
        "dataset": dataset,
        "config": str(config),
        "parties": {p: {"address": f"{p}:1"} for p in vm.PARTIES},
        "spu_addresses": {p: f"{p}:2" for p in vm.PARTIES},
    }
    return cfg, runtime


@pytest.mark.parametrize("uplift", [False, True])
def test_initializer_mapping_and_secret_selection(monkeypatch, uplift):
    native_doubles(monkeypatch)
    indices, total = active.mapped_indices(2, 3, uplift)
    weights = active.mapped_weights(2, 3, uplift, 11, 0.01)
    np.testing.assert_equal(weights, ft.initial_weights(total, 11, 0.01)[indices])
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = np.array([0, 2])
    full = ft.build_design(
        x, np.zeros((2, 3)), np.zeros(2), None, t, "L3_secure", uplift, -1
    )
    reduced = active.active_design(x, t, uplift, -1, 1)
    np.testing.assert_equal(full[:, indices], reduced[:2])
    np.testing.assert_equal(reduced[-1], 0)
    np.testing.assert_equal(
        active.choose_secret_weights([weights, weights * 2], 1), weights * 2
    )


@pytest.mark.parametrize("uplift", [False, True])
@pytest.mark.parametrize("party", ["alice", "bob"])
def test_full_diagnostic_recipient_and_locked_test(
    tmp_path, monkeypatch, uplift, party
):
    native_doubles(monkeypatch)
    cfg, runtime = make_fixture(tmp_path, monkeypatch, uplift)
    runtime["party"] = party
    calls, releases = [], []

    class Object:
        def __init__(self, value, owner):
            self.value, self.owner = value, owner

        def to(self, recipient):
            if self.owner == "secure" and recipient.name != "secure":
                releases.append(recipient.name)
            return Object(self.value, recipient.name)

    def unwrap(v):
        if isinstance(v, Object):
            return v.value
        if isinstance(v, dict):
            return {k: unwrap(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return type(v)(unwrap(x) for x in v)
        return v

    class Plain:
        def __init__(self, name):
            self.name = name

        def __call__(self, fn):
            def execute(*args, **kwargs):
                values = unwrap(args)
                if fn is active.chunk:
                    calls.append((self.name, values[1], values[2]))
                    if self.name == "bob":
                        assert values[2] == "ids"
                if fn is active.prepare_active:
                    value = fn(self.name, *values[1:], **unwrap(kwargs))
                    assert "test" not in value
                    with pytest.raises(ValueError, match="locked"):
                        active.load_split(value, "test")
                    return Object(value, self.name)
                return Object(fn(*values, **unwrap(kwargs)), self.name)

            return execute

    class Secure(Plain):
        def __init__(self, *args, **kwargs):
            self.name = "secure"

        def __call__(self, fn, **opts):
            def execute(*args, **kwargs):
                value = fn(*unwrap(args), **unwrap(kwargs))
                if opts.get("user_specified_num_returns") == 2:
                    return tuple(Object(x, self.name) for x in value)
                return Object(value, self.name)

            return execute

        def dump(self, value, paths):
            for path in paths:
                Path(path).write_text("UNIT_ROUTING_DOUBLE_NOT_SECRET_SHARE")

    sf = ModuleType("secretflow")
    sf.PYU, sf.SPU = Plain, Secure
    sf.init = lambda **kw: None
    sf.wait = lambda value: None
    sf.shutdown = lambda **kw: None

    def reveal(obj):
        value = unwrap(obj)
        assert not isinstance(value, np.ndarray), "No arrays revealed to drivers"
        return value

    sf.reveal = reveal
    monkeypatch.setitem(sys.modules, "secretflow", sf)
    device = ModuleType("secretflow.device")
    device.SPUCompilerNumReturnsPolicy = SimpleNamespace(FROM_USER="USER")
    monkeypatch.setitem(sys.modules, device.__name__, device)
    spu = ModuleType("spu")
    spu.logging = SimpleNamespace(LogOptions=SimpleNamespace)
    monkeypatch.setitem(sys.modules, "spu", spu)
    monkeypatch.setattr(active, "install_pinned_tls_adapter", lambda: None)
    monkeypatch.setattr(active, "reject_anonymous_client", lambda *a: True)
    monkeypatch.setattr(active, "reject_anonymous_spu", lambda *a: True)
    origin = tmp_path / "alice/origin"
    (origin / "results").mkdir()
    metrics = []
    for seed in cfg["shared"]["model_seeds"]:
        for route in ["L0", "L1_secure", "L3_secure"]:
            name = f"{route}_{seed}"
            folder = origin / "trainings" / name / "results"
            folder.mkdir(parents=True)
            np.save(
                folder / "test_predictions.npy", np.full((60, 3 if uplift else 1), 0.5)
            )
            metrics.append({"seed": seed, "route": route, "selected_training": name})
    (origin / "results/private_evaluation.json").write_text(
        json.dumps({"metrics": metrics})
    )
    paths = [origin / "results/private_evaluation.json"] + list(
        (origin / "trainings").glob("*/results/test_predictions.npy")
    )
    (tmp_path / "alice/new/code/origin_output_manifest.json").write_text(
        json.dumps({str(p.relative_to(origin)): ft.fingerprint(p) for p in paths})
    )
    active.train_active(runtime)
    report = json.loads(
        (tmp_path / "alice/new/results/private_diagnostic.json").read_text()
    )
    assert report["active_reference_equivalent"]
    assert len(report["seeds"]) == 5 and all(
        len(s["comparisons"]) == 3 for s in report["seeds"]
    )
    assert releases and set(releases) == {"alice"}
    assert all(
        not list((tmp_path / "bob/new").rglob(pattern))
        for pattern in ["*predictions.npy", "own_weights.npy", "*evaluation.json"]
    )
    assert (
        json.loads((tmp_path / "alice/new/results/frozen_selection.json").read_text())[
            "test_parsed"
        ]
        is False
    )
    assert len(list((tmp_path / "alice/new/trainings").iterdir())) == 20
    assert len(list((tmp_path / "bob/new/trainings").iterdir())) == 10
    # Independently audit an exported synthetic fixture; unit markers are not MPC shares.
    import shutil

    host = tmp_path / "host"
    host.mkdir()
    for owner in vm.PARTIES:
        own = host / owner
        shutil.copytree(
            tmp_path / owner / "new",
            own,
            ignore=shutil.ignore_patterns("*.npz", "model.share", "own_weights.npy"),
        )
        (own / "logs/tls_checks.json").write_text(
            json.dumps(
                {
                    "federation_anonymous_denied": True,
                    "spu_anonymous_denied": True,
                    "protocol_trace_disabled": True,
                }
            )
        )
        (own / "logs/output_boundary.json").write_text(
            json.dumps(
                {
                    "frozen_input_unchanged": True,
                    "bob_feature_input_to_training": False,
                    "selection_index_disclosed_to_bob": False,
                }
            )
        )
        (own / "code/active_diagnostic.yaml").write_text(yaml.safe_dump(cfg))
        hashes = {
            p.name: ft.fingerprint(p) for p in (own / "code").iterdir() if p.is_file()
        }
        runtime_check = {
            "diagnostic_is_smoke": False,
            "config_sha256": ft.fingerprint(own / "code/active_diagnostic.yaml"),
            "code_sha256": hashes,
        }
        (own / "code/runtime.json").write_text(json.dumps(runtime_check))
        for fit in (own / "trainings").iterdir():
            if fit.name.startswith("active_secure"):
                (fit / "logs/progress.json").write_text(
                    json.dumps(
                        {
                            "completed_epochs": cfg["shared"]["epochs"],
                            "planned_epochs": cfg["shared"]["epochs"],
                        }
                    )
                )
            (fit / "code/environment.lock").write_text("synthetic fixture environment")
            for name in hashes:
                (fit / "code" / name).write_bytes((own / "code" / name).read_bytes())
        cells = [
            {"cell_type": "code", "execution_count": i, "outputs": []}
            for i in range(1, 4)
        ]
        (own / "results/party.executed.ipynb").write_text(json.dumps({"cells": cells}))
        (own / "results/NOTEBOOK_COMPLETED.json").write_text(
            json.dumps({"status": "passed"})
        )
        (own / "logs/test_capability.json").write_text(
            (tmp_path / owner / "new/logs/test_capability.json").read_text()
        )
    (host / "status.json").write_text(json.dumps({"status": "passed"}))
    vm.private_permissions(host)
    checked = launcher.verify_diagnostic(host)
    assert checked["counts"] == {"alice": 20, "bob": 10}
    file = host / "alice/results/private_diagnostic.json"
    file.chmod(0o644)
    with pytest.raises(ValueError, match="permission"):
        launcher.verify_diagnostic(host)
    file.chmod(0o600)
    broken = json.loads(file.read_text())
    broken["seeds"][0]["metrics"][next(iter(broken["seeds"][0]["metrics"]))][
        "ci_low"
    ] = float("nan")
    file.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="interval"):
        launcher.verify_diagnostic(host)
    da = active.prepare_active("alice", runtime, cfg)
    with pytest.raises(ValueError, match="not frozen"):
        active.unlock_test(da, False)
    with pytest.raises(FileExistsError):
        active.freeze_selection(
            da,
            {
                str(s): [
                    f"active_secure_seed{s}_candidate0",
                    f"active_secure_seed{s}_candidate1",
                ]
                for s in cfg["shared"]["model_seeds"]
            },
        )


@pytest.mark.parametrize("success", [True, False])
def test_launcher_keeps_new_round_and_filters_weights(tmp_path, monkeypatch, success):
    origin = tmp_path / "origin"
    origin.mkdir()
    (origin / "declaration.json").write_text(
        json.dumps({"dataset": "uci_bank_marketing"})
    )
    (tmp_path / "实验日志").mkdir()
    (tmp_path / "实验日志/实验索引.md").write_text("")
    staged = []

    def stage(workspace, origin, local, round_id, synthetic_only=False):
        staged.append(local)
        for party in vm.PARTIES:
            (local / party / "logs").mkdir(parents=True)

    monkeypatch.setattr(launcher, "stage", stage)

    class Process:
        def __init__(self, *a, **kw):
            self.returncode = 0 if success else 1

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    exports = []
    monkeypatch.setattr(
        launcher,
        "export_artifacts",
        lambda local, run, party, failed=False: exports.append((party, failed)),
    )
    monkeypatch.setattr(vm, "guest", lambda *a: "")
    if success:
        root = launcher.launch(tmp_path, origin)
    else:
        with pytest.raises(RuntimeError, match="failed"):
            launcher.launch(tmp_path, origin)
        root = staged[0]
    assert json.loads((root / "status.json").read_text())["status"] == (
        "passed" if success else "failed"
    )
    assert len(exports) == 2 and all(failed != success for _, failed in exports)
    assert root.stat().st_mode & 0o777 == 0o700


def test_export_filter_is_explicit(tmp_path, monkeypatch):
    import tarfile

    commands = []
    monkeypatch.setattr(vm, "guest", lambda p, cmd: commands.append(cmd) or "")

    def lima(*args):
        with tarfile.open(args[-1], "w"):
            pass

    monkeypatch.setattr(vm, "lima", lima)
    (tmp_path / "alice").mkdir()
    launcher.export_artifacts(tmp_path, "new", "alice")
    assert all(
        f"--exclude='{name}'" in commands[0]
        for name in ["model.share", "*.npz", "own_weights.npy", "*.trace.log"]
    )


def test_stage_configuration_and_source_contract(tmp_path, monkeypatch):
    repo = COMPONENTS.parents[2]
    origin = tmp_path / "origin"
    origin.mkdir()
    (origin / "status.json").write_text(json.dumps({"status": "passed"}))
    (origin / "declaration.json").write_text(
        json.dumps(
            {"dataset": "uci_bank_marketing", "prepared_experiment": "synthetic_only"}
        )
    )
    for party in vm.PARTIES:
        (origin / party / "code").mkdir(parents=True)
        for name in launcher.CORE_NAMES:
            (origin / party / "code" / name).write_bytes(
                (COMPONENTS / name).read_bytes()
            )

    def stage(workspace, dataset, run_id, local, smoke):
        for party in vm.PARTIES:
            target = local / party / "code"
            target.mkdir(parents=True)
            (target / "local_functional.yaml").write_bytes(
                (COMPONENTS.parent / "configs/local_functional.yaml").read_bytes()
            )
            (target / "runtime.json").write_text(json.dumps({"parties": {}}))
        (local / "declaration.json").write_text(
            json.dumps({"prepared_experiment": "synthetic_only"})
        )

    monkeypatch.setattr(vm, "stage_round", stage)
    monkeypatch.setattr(vm, "copy_to", lambda *a: None)
    monkeypatch.setattr(vm, "guest", lambda *a: "")
    local = tmp_path / "new"
    local.mkdir()
    launcher.stage(tmp_path, origin, local, "new")
    runtime = json.loads((local / "alice/code/runtime.json").read_text())
    assert runtime["notebook"] == launcher.NOTEBOOK_NAME
    assert "active_diagnostic.py" in runtime["code_sha256"]
    assert Path(runtime["config"]).name == "active_diagnostic.yaml"
    (origin / "status.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="not completed"):
        launcher.stage(tmp_path, origin, local, "new")
    (origin / "status.json").write_text(json.dumps({"status": "passed"}))
    (origin / "alice/code/functional_training.py").write_text("# differs")
    with pytest.raises(ValueError, match="source differs"):
        launcher.stage(tmp_path, origin, local, "new")
    assert repo.is_dir()


def test_cpu_bridge_keeps_native_code_and_global_state(monkeypatch):
    native = native_doubles(monkeypatch)
    old = dict(native.__globals__)
    clone = active.cpu_cache_hint_bridge(native)
    assert clone.__code__ is native.__code__
    assert native.__globals__["np"] is old["np"]
    assert clone.__globals__["jnp"] is np
    x = np.linspace(-4096, 4096, 1000, dtype=np.float32)
    assert np.isfinite(active.numpy_df(x, "df")).all()
    assert np.all(np.diff(active.numpy_df(x, "df")) >= 0)
    with pytest.raises(ValueError, match="only frozen DF"):
        active.numpy_df(x, "sr")
