"""Provider capability controls use no real data."""

from pathlib import Path
import sys
import json
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "m5_modeling/components"))
import secret_join_vm as p  # noqa: E402


def test_contract_refuses_plaintext_and_no_seeds():
    cfg = yaml.safe_load(
        (Path(p.__file__).parents[1] / "configs/secret_join.yaml").read_text()
    )
    p.validate_config(cfg)
    cfg["output"] = "plaintext"
    with pytest.raises(ValueError):
        p.validate_config(cfg)
    cfg["output"] = "secret_shares"
    cfg["seeds"] = [101]
    with pytest.raises(ValueError):
        p.validate_config(cfg)


def test_whitelist_export_never_copies_shares(monkeypatch, tmp_path):
    commands = []
    copies = []
    monkeypatch.setattr(p.vm, "guest", lambda party, cmd: commands.append(cmd))
    monkeypatch.setattr(p.vm, "lima", lambda *args: copies.append(args))
    p.export(tmp_path, "alice")
    assert len(copies) == len(p.EXPORT_NAMES)
    assert not any(
        ".share" in str(command) or "keys.json" in str(command) for command in commands
    )


def test_verify_refuses_key_or_share_export(tmp_path):
    (tmp_path / "membership.share").write_text("secret")
    with pytest.raises(ValueError, match="Forbidden"):
        p.verify(tmp_path)


def test_queue_failure_durable_and_old_directory_refused(monkeypatch, tmp_path):
    cfg = yaml.safe_load(
        (Path(p.__file__).parents[1] / "configs/secret_join.yaml").read_text()
    )
    source = tmp_path / cfg["source_queue"]
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"status": "passed", "completed": ["a/run", "b/run"]}))

    def fail(*args):
        raise ValueError("stage denied")

    monkeypatch.setattr(p, "stage", fail)
    output = tmp_path / "new"
    with pytest.raises(ValueError):
        p.queue(tmp_path, output)
    state = json.loads((output / "status.json").read_text())
    assert state["status"] == "failed"
    with pytest.raises(FileExistsError):
        p.queue(tmp_path, output)


def make_verified_fixture(tmp_path):
    module = Path(p.__file__).parents[1]
    cfg_path = module / "configs/secret_join.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    for party in p.vm.PARTIES:
        own = tmp_path / party
        (own / "code").mkdir(parents=True)
        (own / "results").mkdir()
        (own / "code/secret_join.yaml").write_bytes(cfg_path.read_bytes())
        (own / "code/helper.py").write_text("pass")
        p.vm.save(
            own / "code/runtime.json",
            {
                "config_sha256": p.sha(cfg_path),
                "code_sha256": {"helper.py": p.sha(own / "code/helper.py")},
            },
        )
        report = {
            key: True
            for key in (
                "known_controls_passed",
                "secret_all_pairs_oracle_passed",
                "federation_anonymous_denied",
                "spu_anonymous_denied",
            )
        }
        report.update(
            {
                key: False
                for key in (
                    "intersection_reconstructed",
                    "membership_or_count_disclosed",
                    "peer_indices_disclosed",
                    "test_read",
                    "native_rr22",
                    "existing_training_consumes_this_psi",
                )
            }
        )
        report.update(status="passed", seeds=cfg["seeds"])
        p.vm.save(own / "results/engineering.json", report)
        p.vm.save(
            own / "results/party.executed.ipynb",
            {
                "cells": [
                    {"cell_type": "code", "execution_count": i, "outputs": []}
                    for i in range(1, 4)
                ]
            },
        )


@pytest.mark.parametrize(
    "failure", [None, "code", "disclosure", "notebook", "permissions"]
)
def test_verify_rejects_source_or_disclosure_or_permissions(
    monkeypatch, tmp_path, failure
):
    make_verified_fixture(tmp_path)
    monkeypatch.setattr(
        p.vm, "guest", lambda *a: "failed" if failure == "permissions" else "passed"
    )
    if failure == "code":
        (tmp_path / "alice/code/helper.py").write_text("changed")
    if failure == "disclosure":
        f = tmp_path / "alice/results/engineering.json"
        report = json.loads(f.read_text())
        report["intersection_reconstructed"] = True
        p.vm.save(f, report)
    if failure == "notebook":
        p.vm.save(tmp_path / "alice/results/party.executed.ipynb", {"cells": []})
    if failure:
        with pytest.raises(ValueError):
            p.verify(tmp_path)
    else:
        assert p.verify(tmp_path)["shares_exported"] is False


def test_stage_code_only_and_source_restrictions(monkeypatch, tmp_path):
    copies = []
    commands = []
    monkeypatch.setattr(p.vm, "setup_certificates", lambda w: {})
    monkeypatch.setattr(p.vm, "restrict_network", lambda *a: None)
    monkeypatch.setattr(
        p.vm, "guest", lambda who, cmd: commands.append(cmd) or "passed"
    )
    monkeypatch.setattr(p.vm, "copy_to", lambda f, *a: copies.append(Path(f).name))
    source = tmp_path / "uci_bank_marketing/oldrun"
    target = p.stage(tmp_path, source, tmp_path / "new")
    assert target.exists() and copies
    assert not any(
        n in copies
        for n in ("keys.json", "manifest.json", "membership.share", "owner_keys.npy")
    )
    assert any("test ! -r" in c and "oldrun/input/keys.json" in c for c in commands)
    assert any("wc -l" in c for c in commands)
    with pytest.raises(ValueError):
        p.stage(tmp_path, tmp_path / "unapproved/run", tmp_path / "other")
    monkeypatch.setattr(p.vm, "guest", lambda *a: "denied")
    with pytest.raises(ValueError):
        p.stage(tmp_path, source, tmp_path / "denied")


class FinishedProcess:
    def __init__(self, failed=False):
        self.returncode = 1 if failed else 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15


@pytest.mark.parametrize("failure", [None, "owner", "export", "verify"])
def test_execution_failure_has_durable_owner_receipt(monkeypatch, tmp_path, failure):
    for party in p.vm.PARTIES:
        (tmp_path / party / "logs").mkdir(parents=True)
    monkeypatch.setattr(
        p.subprocess, "Popen", lambda *a, **k: FinishedProcess(failure == "owner")
    )
    monkeypatch.setattr(p.vm, "guest", lambda *a: "passed")

    def export(*a):
        if failure == "export":
            raise OSError("export denied")

    def verify(*a):
        if failure == "verify":
            raise ValueError("disclosure invalid")

    monkeypatch.setattr(p, "export", export)
    monkeypatch.setattr(p, "verify", verify)
    if failure:
        with pytest.raises((RuntimeError, OSError, ValueError)):
            p.execute(tmp_path)
    else:
        p.execute(tmp_path)
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["status"] == ("failed" if failure else "passed")


@pytest.mark.parametrize("denied", [False, True])
def test_visibility_audit_only_discloses_bool_and_refuses_failed_metadata(
    monkeypatch, tmp_path, denied
):
    target = tmp_path / "uci_bank_marketing_secret_join_20261013T000000Z"
    for party in p.vm.PARTIES:
        (target / party / "code").mkdir(parents=True)
        (target / party / "code/secret_join.yaml").write_text(
            "seeds: [101,102,103,104,105]"
        )
    commands = []

    def guest(party, command):
        commands.append(command)
        return "failed" if denied else "[public library info]\npassed"

    monkeypatch.setattr(p.vm, "guest", guest)
    if denied:
        with pytest.raises(ValueError):
            p.audit_secret_visibility(target)
    else:
        assert not p.audit_secret_visibility(target)["shares_or_metadata_exported"]
        assert len(commands) == 2 and all(
            "VIS_SECRET" in command and "CHEETAH" in command for command in commands
        )
    with pytest.raises(ValueError):
        p.audit_secret_visibility(tmp_path / "invalid-name")
