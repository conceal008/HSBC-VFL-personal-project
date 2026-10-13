"""Synthetic negative controls for recovery audit boundaries."""
from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import external_training_audit as audit  # noqa: E402


def fixture(tmp_path, monkeypatch, recovery=True):
    p = audit.provider
    monkeypatch.setattr(p, "verify", lambda t: {"verification": "passed"})
    p.vm.save(tmp_path / "status.json", {"exit_codes": [0, 0], "cleanup_errors": []})
    if recovery:
        p.vm.save(tmp_path / "recovery_declaration.json", {"source_run": "old"})
        for party in p.vm.PARTIES:
            own = tmp_path / party
            (own / "code").mkdir(parents=True)
            (own / "logs").mkdir()
            (own / "code/helper.py").write_text("pass")
            p.vm.save(own / "code/runtime.json", {
                "recovery_source_run": "old",
                "controller_sha256": {"helper.py": p.sha(own / "code/helper.py")},
            })
            p.vm.save(own / "logs/recovery_arrays.json", {"owner_arrays_exact": True})
    monkeypatch.setattr(p.vm, "guest", lambda *a: "passed")


@pytest.mark.parametrize("recovery", [False, True])
def test_verified_complete_lineage(tmp_path, monkeypatch, recovery):
    fixture(tmp_path, monkeypatch, recovery)
    assert audit.verify_lineage(tmp_path)["recovery_applied"] is recovery


@pytest.mark.parametrize("mutation", ["exit", "cleanup", "arrays", "source", "code", "guest"])
def test_refuses_invalid_recovery(tmp_path, monkeypatch, mutation):
    fixture(tmp_path, monkeypatch)
    if mutation in ("exit", "cleanup"):
        data = {"exit_codes": [0, 0], "cleanup_errors": []}
        data["exit_codes" if mutation == "exit" else "cleanup_errors"] = [None] if mutation == "exit" else ["failure"]
        audit.provider.vm.save(tmp_path / "status.json", data)
    elif mutation == "arrays":
        audit.provider.vm.save(tmp_path / "alice/logs/recovery_arrays.json", {"owner_arrays_exact": False})
    elif mutation == "source":
        path = tmp_path / "alice/code/runtime.json"
        data = json.loads(path.read_text())
        data["recovery_source_run"] = "different"
        audit.provider.vm.save(path, data)
    elif mutation == "code":
        (tmp_path / "alice/code/helper.py").write_text("changed")
    else:
        monkeypatch.setattr(audit.provider.vm, "guest", lambda *a: "failed")
    with pytest.raises(ValueError):
        audit.verify_lineage(tmp_path)


def test_owner_script_preserves_source_and_prints_no_data():
    script = audit.owner_script("/srv/vfl/alice/run")
    assert "write_text" not in script and "print('passed')" in script
    assert "fit_sha256" in script and "old/'trainings'" in script
