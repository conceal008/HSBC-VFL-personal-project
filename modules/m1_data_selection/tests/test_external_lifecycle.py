"""Synthetic owner capabilities, immutable inputs and VM delivery lifecycle."""

from pathlib import Path
import sys
import errno
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml

COMPONENTS = Path(__file__).resolve().parents[1] / "components"
sys.path.insert(0, str(COMPONENTS))
import external_party as ep  # noqa: E402
import external_freeze as ef  # noqa: E402
import external_vm as ev  # noqa: E402


def test_positive_network_control_and_fail_closed_isolation(tmp_path, monkeypatch):
    root = tmp_path
    runtime = dict(
        root=str(root),
        account="vflprep_alice",
        network_probe_port=22,
        network_probe_verified=True,
    )
    monkeypatch.setattr(ep.os, "geteuid", lambda: 1001)
    monkeypatch.setattr(
        ep.subprocess, "check_output", lambda *a, **k: "vflprep_alice\n"
    )
    monkeypatch.setattr(
        ep.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1)
    )
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *a, **k: "" if str(p) == "/proc/mounts" else original(p, *a, **k),
    )
    monkeypatch.setattr(ep, "denied", lambda *a: True)

    def denied_connection(*a, **k):
        raise OSError(errno.EHOSTUNREACH, "unit UID reject")

    monkeypatch.setattr(ep.socket, "create_connection", denied_connection)
    assert all(ep.isolation(runtime).values())
    with pytest.raises(ValueError, match="capability"):
        ep.isolation(dict(runtime, network_probe_verified=False))
    monkeypatch.setattr(
        ep.socket, "create_connection", lambda *a, **k: open(root / "probe", "w")
    )
    with pytest.raises(ValueError, match="capability"):
        ep.isolation(runtime)
    monkeypatch.setattr(ep.socket, "create_connection", denied_connection)
    monkeypatch.setattr(ep.os, "geteuid", lambda: 0)
    with pytest.raises(ValueError, match="capability"):
        ep.isolation(runtime)


def test_permission_probe_distinguishes_denial_from_absence(monkeypatch, tmp_path):
    def denied_open(*a, **k):
        raise PermissionError(errno.EACCES, "unit access denial")

    monkeypatch.setattr(Path, "open", denied_open)
    assert ep.denied(tmp_path / "exists")


def fixture_owner(root, party, dataset):
    for name in ["code", "input", "data", "logs", "results"]:
        (root / name).mkdir(parents=True)
    (root / "code/external_freeze.yaml").write_text("unit configuration")
    (root / "code/owner.py").write_text("unit source fingerprint")
    outcomes = (
        ["target", "treatment"]
        if party == "alice" and dataset == "criteo"
        else (["target"] if party == "alice" else [])
    )
    for split in ep.SPLITS:
        f = pd.DataFrame(
            {
                "record_id": [split + str(i) for i in range(6)],
                "x": np.arange(6, dtype=float),
            }
        )
        if party == "alice":
            f["target"] = (
                [0, 2, 0, 2, 0, 2] if dataset == "coil2000" else [0, 1, 0, 1, 0, 1]
            )
        if len(outcomes) > 1:
            f["treatment"] = [0, 0, 1, 1, 0, 1]
        f.to_csv(root / "input" / f"{split}.csv", index=False)
    return dict(
        root=str(root),
        party=party,
        dataset=dataset,
        config_sha256=ep.sha(root / "code/external_freeze.yaml"),
        code_sha256={"owner.py": ep.sha(root / "code/owner.py")},
        input_sha256={
            s + ".csv": ep.sha(root / "input" / f"{s}.csv") for s in ep.SPLITS
        },
        feature_spec={"features": ["x"], "categorical": []},
        outcomes=outcomes,
    )


@pytest.mark.parametrize(
    "party,dataset", [("alice", "coil2000"), ("alice", "criteo"), ("bob", "criteo")]
)
def test_owner_prepare_complete_and_immutable(tmp_path, monkeypatch, party, dataset):
    runtime = fixture_owner(tmp_path, party, dataset)
    monkeypatch.setattr(ep, "isolation", lambda _: {"unit_capability": True})
    before = dict(runtime["input_sha256"])
    receipt = ep.prepare(runtime)
    assert receipt["status"] == "prepared" and not receipt["sealed_confirmation_read"]
    assert not receipt["training_performed"] and receipt["input_unchanged"]
    for split in ep.SPLITS:
        f = pd.read_csv(tmp_path / "data" / f"{split}.csv")
        assert ("label" in f) == (party == "alice")
        assert ("treatment" in f) == (len(runtime["outcomes"]) > 1)
        assert ep.sha(tmp_path / "input" / f"{split}.csv") == before[split + ".csv"]
    state = json.loads((tmp_path / "data/preprocessing_state.json").read_text())
    assert state["fit_split"] == "train"
    snap = {
        str(p.relative_to(tmp_path)): ep.sha(p)
        for sub in ["data", "results", "logs"]
        for p in (tmp_path / sub).rglob("*")
        if p.is_file()
    }
    with pytest.raises(FileExistsError):
        ep.prepare(runtime)
    assert snap == {
        str(p.relative_to(tmp_path)): ep.sha(p)
        for sub in ["data", "results", "logs"]
        for p in (tmp_path / sub).rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize(
    "failure", ["config", "code", "input", "schema", "outcome", "treatment"]
)
def test_owner_refuses_tamper_or_insufficient_support(tmp_path, monkeypatch, failure):
    r = fixture_owner(tmp_path, "alice", "criteo")
    monkeypatch.setattr(ep, "isolation", lambda _: {"unit_capability": True})
    if failure == "config":
        r["config_sha256"] = "invalid"
    elif failure == "code":
        r["code_sha256"]["owner.py"] = "invalid"
    elif failure == "input":
        r["input_sha256"]["train.csv"] = "invalid"
    else:
        p = tmp_path / "input/train.csv"
        f = pd.read_csv(p)
        if failure == "schema":
            f["foreign_column"] = 1
        elif failure == "outcome":
            f["target"] = 1
        else:
            f["treatment"] = 1
        f.to_csv(p, index=False)
        r["input_sha256"]["train.csv"] = ep.sha(p)
    with pytest.raises(ValueError):
        ep.prepare(r)
    assert not (tmp_path / "results/engineering.json").exists()


def test_transform_rejects_duplicate_overlap_nonfinite():
    base = {
        "train": pd.DataFrame({"record_id": ["a", "b"], "x": [0.0, 1.0]}),
        "validation": pd.DataFrame({"record_id": ["v", "w"], "x": [2.0, 3.0]}),
    }
    spec = {"features": ["x"], "categorical": []}
    for which in ["duplicate", "overlap", "nonfinite"]:
        frames = {k: v.copy() for k, v in base.items()}
        if which == "duplicate":
            frames["train"]["record_id"] = ["a", "a"]
        elif which == "overlap":
            frames["validation"].loc[0, "record_id"] = "a"
        else:
            frames["validation"].loc[0, "x"] = np.inf
        with pytest.raises(ValueError):
            ep.encode(frames, spec)


def test_official_confirmation_is_byte_copied_and_unparsed(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "dev.txt").write_text("1 2 0\n3 4 1\n5 6 0\n7 8 1\n")
    (raw / "labels.txt").write_bytes(b"NOT_PARSEABLE_HOLDOUT_TARGETS\x00\xff")
    cfg = {
        "hash_bits": 64,
        "seeds": {"sampling": 11, "split": 22},
        "datasets": {
            "unit": {
                "source": "dev.txt",
                "format": "whitespace",
                "columns": ["a", "b", "y"],
                "label": "y",
                "treatment": None,
                "sampling_fraction": 1.0,
                "dev_train_fraction": 0.5,
                "official_confirmation": {"labels": "labels.txt"},
                "parties": {"alice": {"features": ["a"]}, "bob": {"features": ["b"]}},
            }
        },
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out"
    r = ef.provision(raw, out, "unit", p)
    assert not r["official_confirmation"]["labels"]["parsed"]
    f = out / "official_labels.sealed"
    assert f.stat().st_mode & 0o777 == 0
    f.chmod(0o600)
    assert f.read_bytes() == (raw / "labels.txt").read_bytes()
    cfg["datasets"]["unit"]["sampling_fraction"] = 0
    assert ef.decision("unit", 0, cfg, cfg["datasets"]["unit"]) == (None, None)
    with pytest.raises(ValueError):
        list(
            ef.source_rows(raw / "dev.txt", {"format": "whitespace", "columns": ["a"]})
        )
    with pytest.raises(ValueError):
        list(ef.source_rows(raw / "dev.txt", {"format": "csv", "columns": ["a"]}))


def test_vm_stage_routes_own_columns_seals_and_fingerprints(tmp_path, monkeypatch):
    copied = []
    commands = []
    vm = ev.vm
    monkeypatch.setattr(vm, "guest", lambda p, c: commands.append((p, c)) or "1001")
    monkeypatch.setattr(vm, "copy_to", lambda *a: copied.append(a))
    provisioned = tmp_path / "provisioned"
    local = tmp_path / "round" / "coil2000"
    local.mkdir(parents=True)
    for party in vm.PARTIES:
        own = provisioned / party
        own.mkdir(parents=True)
        for s in ["train", "validation"]:
            (own / f"{s}.csv").write_text("synthetic unit fixture")
        (own / "confirmation_raw.csv").write_text("sealed unit fixture")
        (own / "confirmation_raw.csv").chmod(0)
    ev.stage(local, provisioned, "coil2000", COMPONENTS.parents[2])
    for party in vm.PARTIES:
        r = json.loads((local / party / "code/runtime.json").read_text())
        assert r["account"] == "vflprep_" + party and r["network_probe_verified"]
        assert r["seeds"] and r["git_sha"] and r["root"].endswith("round_coil2000")
        assert set(r["input_sha256"]) == {"train.csv", "validation.csv"}
        assert bool(r["outcomes"]) == (party == "alice")
        assert (
            provisioned / party / "confirmation_raw.csv"
        ).stat().st_mode & 0o777 == 0
    assert any("iptables -I OUTPUT 1" in c for _, c in commands)
    assert not any(Path(a[0]).name == "test.csv" for a in copied)
    # Every data copy source belongs to that destination owner.
    for source, party, dest in copied:
        if Path(source).parent.name in vm.PARTIES:
            assert Path(source).parent.name == party


def test_vm_execute_whitelist_and_failure_journal(tmp_path, monkeypatch):
    vm = ev.vm
    own = tmp_path / "round/unit/alice"
    for sub in ["code", "logs", "results"]:
        (own / sub).mkdir(parents=True)
    r = {"root": "/srv/unit", "account": "vflprep_alice"}
    (own / "code/runtime.json").write_text(json.dumps(r))
    commands = []
    copied = []
    monkeypatch.setattr(
        vm,
        "guest",
        lambda p, c: commands.append(c) or ("yes" if "test -f" in c else ""),
    )
    valid = {
        "status": "prepared",
        "party": "alice",
        "train_only_fit": True,
        "sealed_confirmation_read": False,
        "input_unchanged": True,
        "isolation_checks_passed": True,
        "training_performed": False,
        "physical_isolation": False,
    }

    def copy(*args):
        target = Path(args[-1])
        copied.append(target.name)
        target.write_text(
            json.dumps(valid) if target.name == "engineering.json" else "{}"
        )

    monkeypatch.setattr(vm, "lima", copy)
    monkeypatch.setattr(
        ev.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0)
    )
    assert ev.execute(own.parent, "alice") == valid
    assert set(copied) == {
        "party.executed.ipynb",
        "NOTEBOOK_COMPLETED.json",
        "engineering.json",
        "isolation.json",
    }
    assert all("data/" not in c for c in commands)
    (own / "logs/driver.log").unlink()
    monkeypatch.setattr(
        ev.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1)
    )
    monkeypatch.setattr(vm, "guest", lambda *a: "no")
    with pytest.raises(RuntimeError):
        ev.execute(own.parent, "alice")
    assert (own / "logs/driver.log").exists()


def test_vm_queue_records_only_preparation_and_failure(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ev, "provision", lambda *a: calls.append(("provision", a[2])))
    monkeypatch.setattr(ev, "stage", lambda *a: calls.append(("stage", a[2])))
    monkeypatch.setattr(
        ev, "execute", lambda _, p: {"party": p, "training_performed": False}
    )
    r = ev.run(tmp_path)
    state = json.loads((r / "status.json").read_text())
    assert state["status"] == "passed" and len(state["completed"]) == 3
    for q in state["completed"]:
        assert not json.loads((Path(q) / "verification.json").read_text())[
            "training_performed"
        ]

    def failure(*a):
        raise RuntimeError("unit provisioning interruption")

    monkeypatch.setattr(ev, "provision", failure)
    with pytest.raises(RuntimeError):
        ev.run(tmp_path)
    states = [
        json.loads(p.read_text())
        for p in (tmp_path / "新增代理准备").glob("*/status.json")
    ]
    assert any(
        s["status"] == "failed" and s["error_type"] == "RuntimeError" for s in states
    )


def test_notebook_runner_checks_logs_before_executor_can_overwrite(
    tmp_path, monkeypatch
):
    from types import ModuleType

    root = tmp_path / "alice" / "unit"
    for sub in ["data", "results", "logs", "code"]:
        (root / sub).mkdir(parents=True)
    runtime = root / "code/runtime.json"
    runtime.write_text(json.dumps({"root": str(root)}))
    calls = []
    ft = ModuleType("functional_training")

    def executor(path):
        calls.append(path)
        (root / "logs/cell1.stdout.log").write_text("frozen synthetic log")
        (root / "results/party.executed.ipynb").write_text("frozen synthetic notebook")

    ft.execute_notebook = executor
    monkeypatch.setitem(sys.modules, "functional_training", ft)
    ep.execute_preparation(runtime)
    before = {str(p): ep.sha(p) for p in root.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        ep.execute_preparation(runtime)
    assert len(calls) == 1 and before == {
        str(p): ep.sha(p) for p in root.rglob("*") if p.is_file()
    }
    (root / "results/party.executed.ipynb").unlink()
    # Even logs alone must block, before creating any cell output.
    with pytest.raises(FileExistsError):
        ep.execute_preparation(runtime)
    assert len(calls) == 1
