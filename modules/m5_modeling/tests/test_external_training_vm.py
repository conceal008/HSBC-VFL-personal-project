"""Privacy boundaries and durable failure receipts using synthetic preparation."""

from pathlib import Path
import json
import sys
import pytest
import yaml

COMPONENTS = Path(__file__).resolve().parents[1] / "components"
sys.path.insert(0, str(COMPONENTS))
import external_training_vm as p  # noqa: E402


@pytest.fixture
def cfg():
    return yaml.safe_load(
        (COMPONENTS.parent / "configs/external_training.yaml").read_text()
    )


def prepared(tmp_path, cfg):
    for party in p.vm.PARTIES:
        root = (
            tmp_path / "新增代理准备" / cfg["prepared_source_run"] / "coil2000" / party
        )
        (root / "code").mkdir(parents=True)
        (root / "results").mkdir()
        p.vm.save(
            root / "code/runtime.json",
            {
                "party": party,
                "dataset": "coil2000",
                "config_sha256": cfg["source_config_sha256"],
            },
        )
        p.vm.save(
            root / "results/engineering.json",
            {
                "status": "prepared",
                "train_only_fit": True,
                "sealed_confirmation_read": False,
                "input_unchanged": True,
                "isolation_checks_passed": True,
            },
        )


@pytest.mark.parametrize(
    "mutation", ["treatment", "seeds", "source", "caps", "dataset"]
)
def test_protocol_rejects_unsafe_changes(cfg, mutation):
    dataset = "coil2000"
    if mutation == "treatment":
        cfg["datasets"][dataset]["treatment"] = True
    elif mutation == "seeds":
        cfg["shared"]["model_seeds"] = [11]
    elif mutation == "source":
        cfg["prepared_source_run"] = "../../sealed"
    elif mutation == "caps":
        cfg["experiment"]["max_rows"]["train"] = 20
    else:
        dataset = "../coil2000"
    with pytest.raises(ValueError):
        p.validate_config(cfg, dataset)


def test_source_accepts_only_frozen_train_only_preparation(tmp_path, cfg):
    prepared(tmp_path, cfg)
    p.validate_config(cfg, "coil2000")
    p.validate_preparation(tmp_path, "coil2000", cfg)
    root = (
        tmp_path
        / "新增代理准备"
        / cfg["prepared_source_run"]
        / "coil2000/bob/results/engineering.json"
    )
    state = json.loads(root.read_text())
    state["sealed_confirmation_read"] = True
    p.vm.save(root, state)
    with pytest.raises(ValueError):
        p.validate_preparation(tmp_path, "coil2000", cfg)


def mock_stage(monkeypatch, tmp_path, cfg, deny=False):
    prepared(tmp_path, cfg)
    monkeypatch.setattr(p.vm, "setup_certificates", lambda w: {})
    monkeypatch.setattr(p.vm, "restrict_network", lambda *a: None)
    copies = []
    commands = []
    monkeypatch.setattr(p.vm, "copy_to", lambda f, *a: copies.append(Path(f)))

    def guest(party, command):
        commands.append((party, command))
        if "sha256sum" in command:
            return cfg["source_config_sha256"] + "  source.yaml"
        return "denied" if deny and "sealed/probe" in command else "passed"

    monkeypatch.setattr(p.vm, "guest", guest)
    target = tmp_path / "staged"
    target.mkdir()
    return target, copies, commands


def test_stage_transfers_only_code_and_runtime_not_clean_rows(
    monkeypatch, tmp_path, cfg
):
    target, copies, commands = mock_stage(monkeypatch, tmp_path, cfg)
    p.stage(tmp_path, "coil2000", "coil2000_external_training_control", target)
    assert copies and not any(f.suffix in {".csv", ".npy", ".npz"} for f in copies)
    assert all("confirmation_raw" not in command for _, command in commands)
    assert any(
        "sha256" in command and "manifest.json" in command for _, command in commands
    )
    for party in p.vm.PARTIES:
        runtime = json.loads((target / party / "code/runtime.json").read_text())
        assert (
            p.sha(target / party / "code/nonlinear_full.yaml")
            == runtime["config_sha256"]
        )
        assert runtime["notebook"] == p.NOTEBOOK
        assert all(
            f"/srv/preparation/{party}/" in command
            for who, command in commands
            if who == party and "install -o root -g" in command and ".csv" in command
        )
    declaration = json.loads((target / "declaration.json").read_text())
    assert (
        declaration["test_staged"] is False
        and declaration["prediction_export"] is False
    )


def test_stage_stops_if_training_can_read_preparation(monkeypatch, tmp_path, cfg):
    target, _, _ = mock_stage(monkeypatch, tmp_path, cfg, deny=True)
    with pytest.raises(ValueError, match="denied"):
        p.stage(tmp_path, "coil2000", "coil2000_external_training_control", target)


def test_verification_rechecks_preparation_boundary(monkeypatch, tmp_path):
    for party in p.vm.PARTIES:
        (tmp_path / party / "logs").mkdir(parents=True)
        p.vm.save(
            tmp_path / party / "logs/preparation_boundary.json",
            {"sealed_read_denied": True},
        )
    monkeypatch.setattr(p.nf, "verify", lambda t: {"verification": "passed"})
    assert p.verify(tmp_path)["verification"] == "passed"
    p.vm.save(
        tmp_path / "bob/logs/preparation_boundary.json", {"sealed_read_denied": False}
    )
    with pytest.raises(ValueError):
        p.verify(tmp_path)


class Process:
    def __init__(self, code=0):
        self.returncode = code

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15


@pytest.mark.parametrize("failure", [None, "owner", "export", "verification", "stage"])
def test_launch_preserves_failure_and_both_owner_cleanup(
    monkeypatch, tmp_path, failure
):
    def stage(w, d, r, t):
        for party in p.vm.PARTIES:
            (t / party / "logs").mkdir(parents=True)
        if failure == "stage":
            raise ValueError("source failure")

    monkeypatch.setattr(p, "stage", stage)
    monkeypatch.setattr(
        p.subprocess, "Popen", lambda *a, **k: Process(1 if failure == "owner" else 0)
    )
    exports = []

    def export(t, r, party, failed):
        exports.append((party, failed))
        if failure == "export" and party == "alice":
            raise OSError("export failure")

    monkeypatch.setattr(p.nf, "export", export)
    monkeypatch.setattr(p.vm, "guest", lambda *a: "passed")

    def verify(t):
        if failure == "verification":
            raise ValueError("inventory failure")
        return {"verification": "passed"}

    monkeypatch.setattr(p, "verify", verify)
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            p.launch(tmp_path, tmp_path / "output", "coil2000")
    else:
        p.launch(tmp_path, tmp_path / "output", "coil2000")
    state = json.loads(next((tmp_path / "output").rglob("status.json")).read_text())
    assert state["status"] == ("failed" if failure else "passed")
    assert len(exports) == (0 if failure == "stage" else 2)
    if failure == "verification":
        assert state["failure_phase"] == "verification"
    if failure == "export":
        assert state["cleanup_errors"][0]["operation"] == "export"


@pytest.mark.parametrize("failed", [False, True])
def test_queue_durable_status_and_old_directory_refusal(monkeypatch, tmp_path, failed):
    def launch(w, o, d):
        if failed:
            raise ValueError("failed owner")
        return o / d / "run"

    monkeypatch.setattr(p, "launch", launch)
    output = tmp_path / "queue"
    if failed:
        with pytest.raises(ValueError):
            p.queue(tmp_path, output)
    else:
        p.queue(tmp_path, output)
    state = json.loads((output / "status.json").read_text())
    assert state["status"] == ("failed" if failed else "passed")
    assert len(state["completed"]) == (0 if failed else 2)
    before = (output / "status.json").read_bytes()
    with pytest.raises(FileExistsError):
        p.queue(tmp_path, output)
    assert (output / "status.json").read_bytes() == before


def test_recovery_attach_accepts_only_complete_bilateral_seeds(
    monkeypatch, tmp_path, cfg
):
    target = tmp_path / "target"
    target.mkdir()
    hashes = {n: p.sha(COMPONENTS / n) for n in p.CODE}
    runtime = {
        "dataset": "coil2000",
        "config_sha256": p.sha(COMPONENTS.parent / "configs/external_training.yaml"),
        "code_sha256": hashes,
    }
    p.vm.save(target / "declaration.json", {"dataset": "coil2000"})
    for party in p.vm.PARTIES:
        (target / party / "code").mkdir(parents=True)
        p.vm.save(target / party / "code/runtime.json", dict(runtime, party=party))
    copied = []
    monkeypatch.setattr(p.vm, "copy_to", lambda f, *a: copied.append(Path(f).name))

    def guest(party, command):
        if "cat " in command:
            return json.dumps(dict(runtime, party=party))
        if "sha256sum" in command:
            name = command.rsplit("/", 1)[-1]
            return hashes.get(name, "same-input") + "  source"
        if "progress.json" in command or "else" in command:
            return "complete" if "seed11" in command else "partial"
        return "passed"

    monkeypatch.setattr(p.vm, "guest", guest)
    source = tmp_path / "coil2000/coil2000_external_training_20261012T000000Z"
    p.attach_recovery(target, "control_new", source)
    receipt = json.loads((target / "recovery_declaration.json").read_text())
    assert (
        receipt["bilateral_complete_seeds"] == [11]
        and len(receipt["reused_secure_fits"]) == 4
    )
    assert "external_training_recovery.py" in copied and not any(
        x.endswith(".csv") for x in copied
    )
    assert receipt["partial_fit_restart_from_initialization"]
    with pytest.raises(ValueError, match="identity"):
        p.attach_recovery(target, "control_new", tmp_path / "../bad")
