"""Recovery lifecycle negative controls; native validation remains provider.verify."""

from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import candidate_recovery as recovery  # noqa: E402


def test_recovery_requires_guest_receipts_and_preserves_original(tmp_path, monkeypatch):
    p = recovery.provider
    old = tmp_path / "safe_run"
    old.mkdir()
    (old / "declaration.json").write_text("{}")
    (old / "status.json").write_text('{"status":"running"}')
    for owner in p.vm.PARTIES:
        (old / owner / "logs").mkdir(parents=True)
        (old / owner / "logs/reference_lineage.json").write_text("{}")
    monkeypatch.setattr(
        p.vm, "guest", lambda *args: json.dumps({"status": "passed", "code_cells": 3})
    )

    def export(root, run, owner):
        (root / owner / "logs").mkdir()

    monkeypatch.setattr(p, "export", export)
    monkeypatch.setattr(
        p, "verify", lambda root: {"verification": "unit_lifecycle_not_security_proof"}
    )
    dest = recovery.recover(old, tmp_path / "recovered")
    assert (
        json.loads((dest / "status.json").read_text())["original_host_exit_status"]
        == "unknown"
    )
    assert (old / "status.json").read_text() == '{"status":"running"}'
    with pytest.raises(FileExistsError):
        recovery.recover(old, dest)
    monkeypatch.setattr(
        p.vm, "guest", lambda *args: '{"status":"failed","code_cells":1}'
    )
    with pytest.raises(ValueError, match="incomplete"):
        recovery.recover(old, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()
    with pytest.raises(ValueError, match="Unsafe"):
        recovery.recover(tmp_path / "unsafe;id", tmp_path / "refused")


def test_recovery_verification_failure_is_durable(tmp_path, monkeypatch):
    p = recovery.provider
    old = tmp_path / "safe_run"
    old.mkdir()
    (old / "declaration.json").write_text("{}")
    for owner in p.vm.PARTIES:
        (old / owner / "logs").mkdir(parents=True)
        (old / owner / "logs/reference_lineage.json").write_text("{}")
    monkeypatch.setattr(
        p.vm, "guest", lambda *args: '{"status":"passed","code_cells":3}'
    )
    monkeypatch.setattr(
        p, "export", lambda root, run, owner: (root / owner / "logs").mkdir()
    )

    def invalid(root):
        raise ValueError("fixture audit failed")

    monkeypatch.setattr(p, "verify", invalid)
    with pytest.raises(ValueError, match="audit failed"):
        recovery.recover(old, tmp_path / "failed")
    assert (
        json.loads((tmp_path / "failed/RECOVERY_FAILED.json").read_text())["status"]
        == "failed"
    )

    assert (
        json.loads((tmp_path / "failed/status.json").read_text())["status"] == "failed"
    )
