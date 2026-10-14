"""Interrupted preparation accepts only verified complete owner runs."""

from pathlib import Path
import json
import sys
import pytest

C = Path(__file__).resolve().parents[1] / "components"
sys.path.insert(0, str(C))
import portfolio_resume as p  # noqa: E402


def fixture(tmp_path):
    run = tmp_path / "old" / "homecredit"
    for party in p.provider.vm.PARTIES:
        own = run / party
        for sub in ["code", "results"]:
            (own / sub).mkdir(parents=True)
        code = own / "code/portfolio_repair_party.py"
        code.write_bytes((C / code.name).read_bytes())
        p.provider.vm.save(
            own / "code/runtime.json",
            dict(
                party=party,
                config_sha256="cfg",
                code_sha256={code.name: p.sha(code)},
                root="/srv/unit",
                account="vflprep_" + party,
            ),
        )
        p.provider.vm.save(
            own / "results/engineering.json",
            dict(
                status="prepared",
                party=party,
                train_only_fit=True,
                sealed_confirmation_read=False,
                input_unchanged=True,
                isolation_checks_passed=True,
                training_performed=False,
                physical_isolation=False,
            ),
        )
        p.provider.vm.save(
            own / "results/party.executed.ipynb",
            {
                "cells": [
                    dict(cell_type="code", execution_count=i, outputs=[])
                    for i in [1, 2, 3]
                ]
            },
        )
        p.provider.vm.save(
            own / "results/NOTEBOOK_COMPLETED.json", {"status": "passed"}
        )
    return run


def test_audit_refuses_changed_frozen_inputs_and_incomplete_receipts(
    tmp_path, monkeypatch
):
    run = fixture(tmp_path)
    monkeypatch.setattr(p.provider.vm, "guest", lambda *a: "passed")
    assert p.audit(run, "cfg", C.parents[2])["status"] == "verified"
    with pytest.raises(ValueError):
        p.audit(run, "wrong", C.parents[2])
    monkeypatch.setattr(p.provider.vm, "guest", lambda *a: "failed")
    with pytest.raises(ValueError):
        p.audit(run, "cfg", C.parents[2])
    monkeypatch.setattr(p.provider.vm, "guest", lambda *a: "passed")
    (run / "alice/code/portfolio_repair_party.py").write_text("pass")
    with pytest.raises(ValueError):
        p.audit(run, "cfg", C.parents[2])


@pytest.mark.parametrize("failure", [False, True])
def test_resume_preserves_old_status_and_routes_only_incomplete(
    tmp_path, monkeypatch, failure
):
    previous = tmp_path / "old"
    previous.mkdir()
    state = dict(
        status="failed",
        error_type="TimeoutExpired",
        completed=[str(previous / "homecredit")],
    )
    p.provider.vm.save(previous / "status.json", state)
    (tmp_path / "全数据集准备").mkdir()
    monkeypatch.setattr(p, "audit", lambda *a: {"status": "verified"})
    monkeypatch.setattr(p.provider, "provision", lambda *a: None)
    monkeypatch.setattr(p.provider, "stage", lambda *a: None)

    def execute(*a):
        if failure:
            raise RuntimeError("unit VM death")
        return {"status": "prepared"}

    monkeypatch.setattr(p.provider, "execute", execute)
    if failure:
        with pytest.raises(RuntimeError):
            p.resume(tmp_path, previous)
        output = next((tmp_path / "全数据集准备").iterdir())
    else:
        output = p.resume(tmp_path, previous)
    new = json.loads((output / "status.json").read_text())
    assert new["reused"] == ["homecredit"]
    assert new["status"] == ("failed" if failure else "passed")
    assert json.loads((previous / "status.json").read_text()) == state
    assert len(new["completed"]) == (1 if failure else 14)
