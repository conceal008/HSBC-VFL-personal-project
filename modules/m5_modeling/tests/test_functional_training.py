"""Safety invariants and numerical transforms on synthetic fixtures only."""
from __future__ import annotations

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
import functional_metrics as metrics  # noqa: E402
import functional_training as training  # noqa: E402
import functional_tls as tls  # noqa: E402
import functional_vm as vm  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs/local_functional.yaml"


@pytest.fixture
def cfg():
    config = yaml.safe_load(CONFIG.read_text())
    config["shared"].update(config["smoke"]["shared_overrides"])
    for spec in config["datasets"].values():
        spec.update(alice_features=2, bob_features=2)
    return config


def test_bob_outcomes_and_cross_split_duplicates_rejected():
    frames = {name: pd.DataFrame({"record_id": [name], "numeric__x": [1.]}) for name in training.SPLITS}
    spec = {"prohibited_features": []}
    assert training.validate_frames(frames, "bob", spec) == ["numeric__x"]
    for frame in frames.values():
        frame["label"] = 1
    with pytest.raises(ValueError, match="Bob contains"):
        training.validate_frames(frames, "bob", spec)
    for frame in frames.values():
        del frame["label"]
    frames["test"]["record_id"] = "train"
    with pytest.raises(ValueError, match="Splits overlap"):
        training.validate_frames(frames, "bob", spec)


def test_future_fields_and_nonfinite_values_rejected():
    frames = {name: pd.DataFrame({"record_id": [name], "numeric__duration": [1.]}) for name in training.SPLITS}
    with pytest.raises(ValueError, match="Prohibited"):
        training.validate_frames(frames, "bob", {"prohibited_features": ["duration"]})
    frames["train"]["numeric__duration"] = np.inf
    with pytest.raises(ValueError, match="Nonfinite"):
        training.validate_frames(frames, "bob", {"prohibited_features": []})


def test_secure_transforms_mask_small_groups_and_keep_labels_out(monkeypatch):
    fake = ModuleType("jax")
    fake.numpy = np
    monkeypatch.setitem(sys.modules, "jax", fake)
    monkeypatch.setitem(sys.modules, "jax.numpy", np)
    groups = np.array([0, 0, 1])
    passive = np.array([[2.], [4.], [99.]])
    table = training.group_table(groups, passive, segments=2, minimum_group=2)
    np.testing.assert_allclose(table, [[3.], [0.]])
    design = training.build_design(np.ones((3, 1)), passive, groups, table, np.array([0, 1, 2]),
                                   route="L1_secure", is_uplift=True, potential=1)
    assert design.shape == (3, 9)
    np.testing.assert_allclose(design[:, 1], [3., 3., 0.])
    assert training.secure_equal(groups, groups)
    assert not training.secure_equal(groups, groups[::-1])
    partials = training.add_group_partials(training.group_partials(groups[:2], passive[:2], 2),
                                          training.group_partials(groups[2:], passive[2:], 2))
    np.testing.assert_allclose(training.finalize_groups(partials, 2), table)
    padded = training.chunk_design(np.ones((3, 1)), passive, groups, table, np.array([0, 1, 2]),
                                   "L1_secure", True, 1, padding=2)
    np.testing.assert_allclose(padded[:3], design)
    np.testing.assert_equal(padded[3:], 0)
    np.testing.assert_equal(training.pad_y(np.array([0, 1]), 2), [0, 1, 0, 0])


def test_tls_client_uses_channel_credentials_and_never_falls_back(tmp_path):
    files = {}
    for key in ("ca_cert", "key", "cert"):
        path = tmp_path / key
        path.write_text(f"synthetic_{key}")
        files[key] = str(path)
    captured = []
    grpc = SimpleNamespace(ssl_channel_credentials=lambda **args: ("CHANNEL", args),
                           secure_channel=lambda address, credentials, **args: captured.append((address, credentials)) or address)
    proxy = SimpleNamespace(_tls_config=files, _addresses={"alice": "alice:1", "bob": "bob:1"},
                            _grpc_options=[], _stubs={})
    tls.initialize_tls_channels(proxy, grpc, lambda channel: channel)
    assert len(captured) == 2 and captured[0][1][0] == "CHANNEL"
    assert captured[0][1][1]["private_key"] == b"synthetic_key"
    proxy._tls_config = None
    with pytest.raises(ValueError, match="requires all"):
        tls.initialize_tls_channels(proxy, grpc, lambda channel: channel)


def test_validation_choice_and_paired_bootstrap(cfg):
    assert metrics.select_candidate([0.3, 0.8]) == 1
    with pytest.raises(ValueError):
        metrics.select_candidate([np.nan])
    y = np.tile([0, 1], 30)
    prediction = np.linspace(0.1, 0.9, len(y))[:, None]
    result = metrics.bootstrap_metrics(y, prediction, {"treatment": False}, cfg["shared"],
                                       cfg["shared"]["bootstrap_seed"], comparator=prediction)
    assert all(value["estimate"] == value["ci_low"] == value["ci_high"] == 0 for value in result.values())


def test_uplift_policy_and_attack_diagnostic(cfg):
    y = np.tile([0, 1, 0, 1], 30)
    t = np.tile([0, 1, 1, 0], 30)
    score = np.linspace(-1, 1, len(y))
    result = metrics.policy_metrics(y, t, score, fraction=0.5)
    assert set(result) == {"centered_qini", "uplift_at_top_k", "policy_value", "policy_gain_vs_no_contact"}
    diag = metrics.membership_diagnostic(y, np.where(y, 0.9, 0.1), y, np.full(len(y), 0.5),
                                         cfg["shared"], cfg["shared"]["bootstrap_seed"])
    assert diag["loss_attack_auc"] == 1


def test_each_training_has_immutable_identity_and_code(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(training, "ROOT", tmp_path)
    root = tmp_path / "alice" / "round"
    for folder in ("input", "code", "data"):
        (root / folder).mkdir(parents=True)
    (root / "input/manifest.json").write_text('{}')
    (root / "code/entry.ipynb").write_text('synthetic')
    (root / "data/derived.json").write_text('{}')
    assert training.new_training("alice", "round", "L3_seed11_c1", cfg)
    target = root / "trainings/L3_seed11_c1"
    assert (target / "code/entry.ipynb").exists()
    assert json.loads((target / "data/input_references.json").read_text())["derived_input"]
    with pytest.raises(FileExistsError):
        training.new_training("alice", "round", "L3_seed11_c1", cfg)


def test_seed_permutation_is_local_and_reversible(cfg):
    seed = cfg["shared"]["model_seeds"][0]
    data = {"train": {"x": np.arange(12).reshape(6, 2)}}
    shuffled = training.get_array(data, "train", "x", seed)
    order = np.random.default_rng(seed).permutation(len(shuffled))
    recovered = training.pack_predictions([shuffled[:, 0]], seed, False, data, "train")
    np.testing.assert_equal(shuffled, data["train"]["x"][order])
    np.testing.assert_equal(recovered[:, 0], data["train"]["x"][:, 0])
    np.testing.assert_equal(training.get_chunk(data, "train", "x", 1, 3, seed), shuffled[1:3])
    recovered_chunks = training.pack_chunked_predictions([[shuffled[:3, 0], np.r_[shuffled[3:, 0], 999]]],
                                                         6, seed, False, data, "train")
    np.testing.assert_equal(recovered_chunks, recovered)


def test_invalid_dataset_never_stages_a_vm(tmp_path):
    with pytest.raises(ValueError, match="not registered"):
        vm.launch(tmp_path, "unregistered")


def make_local_fixture(tmp_path, monkeypatch, cfg, party="alice", uplift=False):
    """Actual synthetic CSV transforms, with simulated immutable filesystem access."""
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "isolation_preflight", lambda _: {"synthetic_fixture": True})
    root = tmp_path / party / "round"
    for subdir in ("input", "code", "data", "logs", "results"):
        (root / subdir).mkdir(parents=True)
    (root / "code/entry.py").write_text("# fixture source\n")
    manifest = {}
    rng = np.random.default_rng(17)
    for split in training.SPLITS:
        n = 90
        frame = pd.DataFrame({"record_id": [f"{split}{i}" for i in range(n)],
                              "numeric__x": rng.normal(size=n), "numeric__z": rng.normal(size=n)})
        if party == "alice":
            frame["label"] = np.tile([0, 1], n // 2)
            if uplift:
                frame["treatment"] = np.tile([0, 1, 2], n // 3)
        path = root / "input" / f"{split}.csv"
        frame.to_csv(path, index=False)
        manifest[path.name] = training.fingerprint(path)
    (root / "input/manifest.json").write_text(json.dumps(manifest))
    original_open = Path.open
    def immutable_open(path, mode="r", *args, **kwargs):
        if "input" in path.parts and mode == "r+b":
            raise PermissionError("fixture read-only")
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", immutable_open)
    dataset = "hillstrom_email_marketing" if uplift else "uci_bank_marketing"
    return training.prepare_local(party, "round", dataset, cfg)


@pytest.mark.parametrize("uplift", [False, True])
def test_actual_local_fits_selection_and_output_audit(tmp_path, monkeypatch, cfg, uplift):
    data = make_local_fixture(tmp_path, monkeypatch, cfg, uplift=uplift)
    records = training.fit_local_baselines(data, cfg)
    assert len(records) == 2
    first = Path(data["root"]) / "trainings" / records[0]["name"] / "results"
    for route in ("L1_secure", "L3_secure"):
        name = f"{route}_fixture"
        training.new_training("alice", "round", name, cfg)
        folder = Path(data["root"]) / "trainings" / name / "results"
        for source in first.iterdir():
            if source.is_file():
                folder.joinpath(source.name).write_bytes(source.read_bytes())
        records.append({"name": name, "seed": 11, "route": route})
    assert training.verify_local_outputs(data, records)
    status = training.finalize_alice(data, records)
    assert set(status) == {"status", "private_artifact"}
    report = json.loads(Path(status["private_artifact"]).read_text())
    assert len(report["metrics"]) == 3 and len(report["paired_differences"]) == 2
    assert all(r["seed"] == 11 for r in report["metrics"])
    for item in report["paired_differences"]:
        if item["comparison"] == "L3_secure_minus_L1_secure":
            assert all(m["estimate"] == 0 for m in item["metrics"].values())


def test_passive_data_and_output_freeze_fail_closed(tmp_path, monkeypatch, cfg):
    data = make_local_fixture(tmp_path, monkeypatch, cfg, party="bob")
    assert not ({"y", "t"} & set(data["train"]))
    assert training.verify_local_outputs(data, [])
    root = Path(data["root"])
    np.save(root / "results/test_predictions.npy", [0.5])
    with pytest.raises(ValueError, match="released model outputs"):
        training.verify_local_outputs(data, [])
    (root / "results/test_predictions.npy").unlink()
    data["train"]["y"] = [1]
    with pytest.raises(ValueError, match="obtained outcomes"):
        training.verify_local_outputs(data, [])
    del data["train"]["y"]
    (root / "input/train.csv").write_text("changed")
    with pytest.raises(ValueError, match="changed during"):
        training.verify_local_outputs(data, [])
    with pytest.raises(ValueError, match="fingerprint"):
        training.prepare_local("bob", "round", "uci_bank_marketing", cfg)


def test_isolation_negative_controls_require_permission_errors(monkeypatch):
    import errno
    monkeypatch.setattr(training.os, "geteuid", lambda: 1001)
    monkeypatch.setattr(training.subprocess, "run", lambda *args, **kw: SimpleNamespace(returncode=1))
    monkeypatch.setattr(training.subprocess, "check_output", lambda *args, **kw: "alice\n")
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **kw: "tmpfs" if str(path) == "/proc/mounts" else original(path, *a, **kw))
    def denied(*args, **kwargs):
        raise PermissionError(errno.EACCES, "fixture permission denial")
    monkeypatch.setattr(Path, "read_bytes", denied)
    monkeypatch.setattr(training.socket, "create_connection", denied)
    assert all(training.isolation_preflight("alice").values())
    monkeypatch.setattr(Path, "read_bytes", lambda *_: b"readable")
    with pytest.raises(RuntimeError, match="isolation"):
        training.isolation_preflight("alice")


def test_notebook_execution_keeps_outputs_private_and_failure_record(tmp_path, monkeypatch):
    import nbformat
    monkeypatch.setattr(training, "ROOT", tmp_path)
    root = tmp_path / "alice/round"
    (root / "code").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "logs").mkdir()
    runtime = root / "code/runtime.json"
    runtime.write_text(json.dumps({"party": "alice", "run_id": "round"}))
    source = root / "code/S5.P2_local_functional.ipynb"
    for succeeds in (True, False):
        nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("fixture"),
                  nbformat.v4.new_code_cell("print('private fixture')" if succeeds else "raise ValueError('fixture failure')")])
        nbformat.write(nb, source)
        if succeeds:
            training.execute_notebook(runtime)
        else:
            with pytest.raises(RuntimeError, match="Private Notebook failed"):
                training.execute_notebook(runtime)
        executed = nbformat.read(root / "results/party.executed.ipynb", as_version=4)
        assert executed.cells[1].outputs
        assert nbformat.read(source, as_version=4).cells[1].outputs == []


def test_invalid_or_degenerate_metrics_fail_instead_of_fabricated_ci(cfg):
    with pytest.raises(ValueError, match="Both randomized"):
        metrics.qini(np.ones(8), np.ones(8), np.zeros(8))
    with pytest.raises(ValueError):
        metrics.select_candidate([])
    shared = {**cfg["shared"], "bootstrap_repeats": 0}
    with pytest.raises(ValueError, match="No valid bootstrap"):
        metrics.bootstrap_metrics(np.array([0, 1]), np.array([[.2], [.8]]),
                                  {"treatment": False}, shared, 11)


def test_tls_version_pin_and_negative_grpc_paths(tmp_path, monkeypatch):
    import importlib.metadata
    ca = tmp_path / "ca"
    ca.write_text("synthetic")
    grpc = ModuleType("grpc")
    class Timeout(Exception):
        pass
    channel = SimpleNamespace(close=lambda: None)
    grpc.FutureTimeoutError = Timeout
    grpc.ssl_channel_credentials = lambda **_: "channel_credential"
    grpc.secure_channel = lambda *_: channel
    future = SimpleNamespace(result=lambda **_: None)
    grpc.channel_ready_future = lambda _: future
    monkeypatch.setitem(sys.modules, "grpc", grpc)
    assert not tls.reject_anonymous_client("fixture:1", ca, 1)
    def timeout(**kwargs):
        raise Timeout()
    future.result = timeout
    assert tls.reject_anonymous_client("fixture:1", ca, 1)
    module = ModuleType("secretflow.distributed.fed.proxy.grpc.grpc")
    class Proxy:
        pass
    module.GrpcProxy = Proxy
    module.fed_pb2_grpc = SimpleNamespace(SfFedProxyStub=lambda x: x)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "unexpected-version")
    with pytest.raises(RuntimeError, match="pinned"):
        tls.install_pinned_tls_adapter()
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "1.14.0b0")
    tls.install_pinned_tls_adapter()
    assert Proxy._init_channel is not None


@pytest.mark.parametrize("error", [None, "TLSV1_ALERT_UNKNOWN_CA", "unrelated error"])
def test_spu_negative_probe_only_accepts_tls_auth_rejection(monkeypatch, error):
    import socket
    import ssl
    from contextlib import nullcontext
    def wrap(*args, **kwargs):
        if error:
            raise ssl.SSLError(error)
        return nullcontext()
    context = SimpleNamespace(wrap_socket=wrap)
    monkeypatch.setattr(ssl, "create_default_context", lambda **_: context)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: nullcontext("socket"))
    assert tls.reject_anonymous_spu("fixture:1", "fixture_ca", 1) == bool(error and "ALERT" in error)


@pytest.mark.parametrize("party,uplift", [("alice", False), ("bob", True)])
def test_training_orchestration_recipient_contract(tmp_path, monkeypatch, cfg, party, uplift):
    """Plaintext unit double checks routing; actual MPC security is tested in VMs."""
    from enum import Enum
    monkeypatch.setattr(training, "ROOT", tmp_path)
    dataset = "hillstrom_email_marketing" if uplift else "uci_bank_marketing"
    local = {name: make_local_fixture(tmp_path, monkeypatch, cfg, party=name, uplift=uplift)
             for name in ("alice", "bob")}
    cfg["shared"]["batch_size"] = 30
    cfg["shared"]["infeed_rows"] = 60
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    released = []
    class Object:
        def __init__(self, value, owner):
            self.value, self.owner = value, owner
        def to(self, recipient):
            if isinstance(recipient, Plain) and not isinstance(recipient, Secure):
                released.append((self.owner, recipient.name))
            return Object(self.value, recipient.name)
    def unwrap(value):
        if isinstance(value, Object):
            return value.value
        if isinstance(value, dict):
            return {k: unwrap(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(unwrap(v) for v in value)
        return value
    class Plain:
        def __init__(self, name):
            self.name = name
        def __call__(self, fn):
            def invoke(*args, **kwargs):
                if fn is training.prepare_local:
                    return Object(local[self.name], self.name)
                return Object(fn(*unwrap(args), **unwrap(kwargs)), self.name)
            return invoke
    class Secure(Plain):
        def __init__(self, *args, **kwargs):
            self.name = "MPC_unit_double"
        def __call__(self, fn, **compile_options):
            invoke = super().__call__(fn)
            def wrapped(*args, **kwargs):
                result = invoke(*args, **kwargs)
                if compile_options.get("user_specified_num_returns") == 2:
                    return [Object(x, self.name) for x in result.value]
                return result
            return wrapped
        def dump(self, value, paths):
            # Only a routing marker; unit tests never pretend this is a secret share.
            for path in paths:
                Path(path).write_text("UNIT_DOUBLE_NOT_A_SECRET_SHARE")
    fake_sf = ModuleType("secretflow")
    fake_sf.PYU, fake_sf.SPU = Plain, Secure
    fake_sf.init = lambda **kw: None
    fake_sf.shutdown = lambda **kw: None
    fake_sf.wait = lambda _: None
    def reveal_public(value):
        result = unwrap(value)
        def audit(item):
            if isinstance(item, np.ndarray):
                raise AssertionError("Controller cannot reveal an array")
            if isinstance(item, dict):
                assert not {"y", "x", "t", "metrics", "predictions"} & set(item)
                for nested in item.values():
                    audit(nested)
            elif isinstance(item, (list, tuple)):
                for nested in item:
                    audit(nested)
        audit(result)
        return result
    fake_sf.reveal = reveal_public
    fake_spu = ModuleType("spu")
    fake_spu.logging = SimpleNamespace(LogOptions=SimpleNamespace)
    monkeypatch.setitem(sys.modules, "spu", fake_spu)
    monkeypatch.setitem(sys.modules, "secretflow", fake_sf)
    device = ModuleType("secretflow.device")
    device.SPUCompilerNumReturnsPolicy = SimpleNamespace(FROM_USER="USER")
    monkeypatch.setitem(sys.modules, device.__name__, device)
    class Sig(Enum):
        T3 = "t3"
        SR = "sr"
        DF = "df"
    sigmoid_module = ModuleType("secretflow.utils.sigmoid")
    sigmoid_module.SigType = Sig
    sigmoid_module.sigmoid = lambda x, _: 1 / (1 + np.exp(-x))
    monkeypatch.setitem(sys.modules, sigmoid_module.__name__, sigmoid_module)
    linear = ModuleType("secretflow.ml.linear.linear_model")
    linear.RegType = SimpleNamespace(Logistic="LR")
    monkeypatch.setitem(sys.modules, linear.__name__, linear)
    model = ModuleType("secretflow.ml.linear.ss_sgd.model")
    model.Penalty = SimpleNamespace(L2="L2")
    model.Strategy = SimpleNamespace(NAIVE_SGD="SGD")
    def batch(x, y, w, lr, regularization, **kwargs):
        p = 1 / (1 + np.exp(-(x @ w).reshape(-1)))
        return w - lr * ((x.T @ (p - y))[:, None] / len(y) + regularization * w), None
    model._batch_update_w = batch
    monkeypatch.setitem(sys.modules, model.__name__, model)
    jax = ModuleType("jax")
    jax.numpy = np
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", np)
    monkeypatch.setattr(training, "install_pinned_tls_adapter", lambda: None)
    monkeypatch.setattr(training, "reject_anonymous_client", lambda *a: True)
    monkeypatch.setattr(training, "reject_anonymous_spu", lambda *a: True)
    runtime = {"party": party, "run_id": "round", "dataset": dataset, "config": str(cfg_path),
               "parties": {p: {"address": f"{p}:1"} for p in local},
               "spu_addresses": {p: f"{p}:2" for p in local}}
    training.train_round(runtime)
    assert released and all(receiver == "alice" for _, receiver in released)
    assert not list((tmp_path / "bob").rglob("*predictions.npy"))
    assert (tmp_path / "alice/round/results/private_evaluation.json").exists()
    assert json.loads((tmp_path / party / "round/results/status.json").read_text())["status"] == "completed"
    assert training.secure_predict(np.array([[1., 1.]]), np.zeros((2, 1)), "t3")[0] == .5
    with pytest.raises(AssertionError, match="cannot reveal"):
        fake_sf.reveal(Object(np.array([0.4]), "MPC_unit_double"))


def test_vm_certificates_keep_private_keys_local(tmp_path, monkeypatch):
    commands, copies = [], []
    def run(command, **kwargs):
        commands.append(command)
        for flag in ("-keyout", "-out"):
            if flag in command:
                Path(command[command.index(flag) + 1]).write_text("synthetic_cert_fixture")
        return SimpleNamespace(stdout="", returncode=0)
    def lima(*args, **kwargs):
        copies.append(args)
        if args[0] == "copy" and "request.csr" in str(args[1]):
            Path(args[2]).write_text("synthetic_csr")
        return SimpleNamespace(stdout="", returncode=0)
    monkeypatch.setattr(vm.subprocess, "run", run)
    monkeypatch.setattr(vm, "lima", lima)
    monkeypatch.setattr(vm, "guest", lambda party, script: "192.168.104.10" if "getent" in script else "")
    assert set(vm.setup_certificates(tmp_path)) == {"alice", "bob"}
    vm.setup_certificates(tmp_path)
    assert sum("-keyout" in c for c in commands) == 1
    assert not any(":/srv/vfl/" in str(c[1]) and "key.pem" in str(c[1]) for c in copies)


@pytest.mark.parametrize("smoke", [False, True])
def test_vm_staging_and_network_restrictions_are_party_scoped(tmp_path, monkeypatch, smoke):
    commands, copies = [], []
    addresses = {"alice": "192.168.104.10", "bob": "192.168.104.11"}
    monkeypatch.setattr(vm, "setup_certificates", lambda _: addresses)
    monkeypatch.setattr(vm, "guest", lambda p, script: commands.append((p, script)) or ("1001" if "id -u" in script else ""))
    monkeypatch.setattr(vm, "copy_to", lambda src, party, target: copies.append((Path(src), party, target)))
    dataset = "uci_bank_marketing"
    if not smoke:
        prepared = vm.PREPARED[dataset]
        for party in vm.PARTIES:
            own = tmp_path / "联邦隔离产物" / prepared / party
            manifest_dir = tmp_path / "联邦隔离结果" / prepared / party
            own.mkdir(parents=True)
            manifest_dir.mkdir(parents=True)
            manifest = {}
            for split in training.SPLITS:
                path = own / f"{split}.csv"
                path.write_text(f"synthetic_only_{party}_{split}")
                manifest[path.name] = training.fingerprint(path)
            manifest["clean_features.csv"] = "unused_preparation_artifact_not_delivered"
            (manifest_dir / "data_manifest.json").write_text(json.dumps(manifest))
    local = tmp_path / "output"
    local.mkdir()
    vm.stage_round(tmp_path, dataset, "round", local, smoke)
    declaration = json.loads((local / "declaration.json").read_text())
    assert declaration["true_psi"] == "not_executed" and not declaration["raw_data_to_controller"]
    for party in vm.PARTIES:
        staged_manifest = json.loads((local / party / "data/manifest.json").read_text())
        assert set(staged_manifest) == {f"{split}.csv" for split in training.SPLITS}
    assert all(party in source.parts for source, party, target in copies if "联邦隔离产物" in source.parts)
    assert all("--dports 50051,50052" in script and "--uid-owner 1001" in script
               for _, script in commands if "iptables -N" in script)
    if smoke:
        alice = pd.read_csv(local / "alice/data/train.csv")
        bob = pd.read_csv(local / "bob/data/train.csv")
        assert "label" in alice and "label" not in bob
        assert alice["record_id"].equals(bob["record_id"])


@pytest.mark.parametrize("success", [False, True])
def test_vm_launch_preserves_failure_and_only_exports_authorized_artifacts(tmp_path, monkeypatch, success):
    import tarfile
    (tmp_path / "实验日志").mkdir()
    (tmp_path / "实验日志/实验索引.md").touch()
    staged = []
    def stage(workspace, dataset, round_id, local, smoke):
        staged.append(local)
        for party in vm.PARTIES:
            (local / party / "logs").mkdir(parents=True)
    monkeypatch.setattr(vm, "stage_round", stage)
    commands = []
    monkeypatch.setattr(vm, "guest", lambda p, script: commands.append(script) or "")
    class Process:
        returncode = 0 if success else 1
        def poll(self):
            return self.returncode
        def terminate(self):
            self.returncode = -1
    monkeypatch.setattr(vm.subprocess, "Popen", lambda *args, **kw: Process())
    def lima(*args, **kwargs):
        with tarfile.open(args[2], "w"):
            pass
        return SimpleNamespace(stdout="", returncode=0)
    monkeypatch.setattr(vm, "lima", lima)
    if success:
        local = vm.launch(tmp_path, "uci_bank_marketing")
        assert all("--exclude='model.share'" in c for c in commands if "sudo tar" in c)
    else:
        with pytest.raises(RuntimeError, match="Participant failed"):
            vm.launch(tmp_path, "uci_bank_marketing")
        local = staged[0]
    status = json.loads((local / "status.json").read_text())
    assert status["status"] == ("passed" if success else "failed")
    assert (local / "实验日志.md").exists()
    assert status["round_id"] in (tmp_path / "实验日志/实验索引.md").read_text()


def test_lima_wrapper_uses_dedicated_environment(monkeypatch):
    seen = []
    monkeypatch.setattr(vm.subprocess, "run", lambda *a, **kw: seen.append((a, kw)) or SimpleNamespace(stdout="done"))
    assert vm.lima("list").stdout == "done"
    assert seen[0][1]["env"]["LIMA_HOME"] == vm.LIMA_HOME_PATH
    assert seen[0][1]["check"] is True
    assert vm.guest("alice", "synthetic command") == "done"


def test_sr_projection_and_raw_probability_validation(monkeypatch):
    from enum import Enum
    jax = ModuleType('jax')
    jax.numpy = np
    monkeypatch.setitem(sys.modules, 'jax', jax)
    monkeypatch.setitem(sys.modules, 'jax.numpy', np)
    module = ModuleType('secretflow.utils.sigmoid')
    class Sig(Enum):
        SR = 'sr'
    module.SigType = Sig
    module.sigmoid = lambda x, _: .5 * x / np.sqrt(1 + x * x) + .5
    monkeypatch.setitem(sys.modules, module.__name__, module)
    weights = training.project_weights(np.array([[100.], [-100.], [1.]]), 10.)
    np.testing.assert_equal(weights[:, 0], [10., -10., 1.])
    p, valid = training.checked_predict(np.array([[4., -4., 1.], [-4., 4., 1.]]), weights, 'sr')
    assert bool(valid) and np.all((p > 0) & (p < 1)) and p[0] > p[1]
    module.sigmoid = lambda x, _: np.full_like(x, 1.1)
    p, valid = training.checked_predict(np.ones((1, 3)), weights, 'sr')
    assert not bool(valid) and p[0] == 1.1  # no clipping that could conceal a violation
    module.sigmoid = lambda x, _: np.full_like(x, np.nan)
    assert not bool(training.checked_predict(np.ones((1, 3)), weights, 'sr')[1])


def test_saturation_guard_rejects_before_validation_or_test_selection(tmp_path, cfg):
    data = {'root': str(tmp_path), 'shared': cfg['shared']}
    name = 'L3_secure_all_arms_seed11_candidate0'
    folder = tmp_path / 'trainings' / name / 'results'
    folder.mkdir(parents=True)
    with pytest.raises(ValueError, match='boundary limit'):
        training.store_predictions(data, name, np.zeros((100, 1)), np.ones((8, 1)) * .5, np.ones((8, 1)) * .5)
    assert not list(folder.glob('*predictions.npy'))
    assert not (folder / 'validation_score.json').exists()
    assert json.loads((folder / 'numerical_checks.json').read_text())['training_boundary_fraction'] == 1.


def test_wide_rare_outcome_sr_optimizer_has_finite_unclipped_scores():
    # Plain numerical reference only, not a cryptographic security test.
    rng = np.random.default_rng(20261001)
    x = np.clip(rng.normal(size=(512, 128)), -4, 4)
    x = np.c_[x, np.ones(len(x))]
    y = (rng.random(len(x)) < .02).astype(float)
    w = rng.normal(0, .01, x.shape[1])
    for _ in range(12):
        z = x @ w
        p = .5 * z / np.sqrt(1 + z * z) + .5
        w = np.clip(w - .3 * (x.T @ (p - y) / len(y)), -10, 10)
    z = x @ w
    p = .5 * z / np.sqrt(1 + z * z) + .5
    assert np.isfinite(p).all() and np.all((p > 0) & (p < 1))
    assert np.mean((p <= 1e-6) | (p >= 1 - 1e-6)) <= .01


@pytest.mark.parametrize('dataset', ['uci_bank_marketing', 'hillstrom_email_marketing'])
def test_synthetic_vm_inputs_match_task_and_never_give_bob_outcomes(cfg, dataset):
    alice = vm.synthetic_inputs(cfg, dataset, 'alice')
    bob = vm.synthetic_inputs(cfg, dataset, 'bob')
    for split in training.SPLITS:
        assert alice[split]['record_id'].equals(bob[split]['record_id'])
        assert 'label' in alice[split] and not {'label', 'treatment'} & set(bob[split])
        if cfg['datasets'][dataset]['treatment']:
            assert set(alice[split]['treatment']) == {0, 1, 2}
        else:
            assert 'treatment' not in alice[split]
    pd.testing.assert_frame_equal(alice['test'], vm.synthetic_inputs(cfg, dataset, 'alice')['test'])


def test_export_permissions_do_not_follow_links_or_change_frozen_source(tmp_path):
    import stat
    original = tmp_path / 'frozen'
    original.write_text('synthetic original only')
    original.chmod(0o640)
    private = tmp_path / 'private'
    (private / 'results').mkdir(parents=True)
    output = private / 'results/status.json'
    output.write_text('{"status":"synthetic"}')
    (private / 'source_link').symlink_to(original)
    vm.private_permissions(private)
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    assert stat.S_IMODE((private / 'results').stat().st_mode) == 0o700
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert stat.S_IMODE(original.stat().st_mode) == 0o640
    assert original.read_text() == 'synthetic original only'
