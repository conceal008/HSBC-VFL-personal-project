"""New-step metadata and corruption-aware bilateral recovery policy."""

from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import portfolio_resume_vm as r  # noqa: E402


def test_corrupt_partner_forces_fresh_secure_pair(monkeypatch):
    monkeypatch.setattr(r.phase.vm, "copy_to", lambda *a: None)

    def check(party, code):
        return json.dumps(
            {
                "complete": party == "alice",
                "reasons": [] if party == "alice" else ["JSONDecodeError"],
                "no_arrays_exported": True,
            }
        )

    monkeypatch.setattr(r.phase, "script", check)
    monkeypatch.setattr(r, "ELIGIBILITY_OUTPUT", None)
    pair, local = r.healthy_source(
        Path("homecredit/homecredit_portfolio_training_20261014T014214342444Z"),
        "homecredit",
        33,
        {"shared": {"epochs": 12}},
    )
    assert pair == [] and local == ["L0_LR_0_seed33", "L0_tree_0_seed33"]
    with pytest.raises(ValueError, match="identity"):
        r.healthy_source(Path("../sealed"), "homecredit", 11, {})


def test_new_runtime_does_not_relabel_math_template(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(r, "BASE_STAGE", lambda *a, **kw: calls.append("base"))
    permission_calls = []
    monkeypatch.setattr(r.phase.vm, "guest", lambda *a: permission_calls.append(a))
    for p in r.phase.vm.PARTIES:
        (tmp_path / p / "code").mkdir(parents=True)
        (tmp_path / p / "code/runtime.json").write_text('{"party":"' + p + '"}')

    def write(target, party, run_id, name, payload):
        calls.append(payload)

    monkeypatch.setattr(r.phase, "write_guest", write)
    r.stage(Path("workspace"), "homecredit", "new_run", tmp_path)
    assert calls[0] == "base" and all(
        x["execution_step_id"] == "S5.X3" and x["mathematical_template_step"] == "S5.X2"
        for x in calls[1:]
    )
    assert json.loads((tmp_path / "continuation_step.json").read_text())[
        "does_not_clear_previous_failure"
    ]
    assert len(permission_calls) == len(r.phase.vm.PARTIES)
    assert all("chmod 600" in c[1] and "chmod 700" in c[1] for c in permission_calls)
    assert all(x["owner_umask"] == r.CONTINUATION["owner_umask"] for x in calls[1:])


def test_share_permission_verification_is_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "BASE_VERIFY", lambda *a: {"verification": "passed"})
    monkeypatch.setattr(r.phase, "script", lambda *a: "passed")
    assert r.verify(tmp_path)["owner_model_share_permissions_verified"]
    monkeypatch.setattr(r.phase, "script", lambda *a: "denied")
    with pytest.raises(ValueError, match="permissions"):
        r.verify(tmp_path)


def test_x3_notebook_sets_configured_umask_before_training():
    import ast

    path = r.phase.MODULE / "notebooks/S5.X3_portfolio_phase.ipynb"
    nb = json.loads(path.read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    tree = ast.parse(cells[0])
    assert any(
        isinstance(n, ast.Call)
        and ast.unparse(n.func) == "os.umask"
        and ast.unparse(n.args[0]) == "runtime['owner_umask']"
        for n in ast.walk(tree)
    )
    assert "train_recovered(runtime)" in cells[1]


def test_queue_freezes_all_source_and_distinguishes_new_step(tmp_path, monkeypatch):
    for name in (
        "complete_source",
        "stage",
        "verify",
        "PHASE_NOTEBOOK",
        "MERGE_NOTEBOOK",
    ):
        monkeypatch.setattr(r.phase, name, getattr(r.phase, name))
    monkeypatch.setattr(r, "ELIGIBILITY_OUTPUT", None)
    monkeypatch.setattr(r, "SOURCE_GUARD", {})
    monkeypatch.setattr(r.phase, "queue", lambda *a: {"status": "unit queue only"})
    out = tmp_path / "new_queue"
    result = r.queue(tmp_path, out)
    assert result["status"] == "unit queue only"
    assert r.phase.PHASE_NOTEBOOK.startswith("S5.X3") and r.phase.verify is r.verify
    guard = json.loads(
        (tmp_path / "new_queue_eligibility/source_guard.json").read_text()
    )
    assert (
        str(r.CONTINUATION_CONFIG) in guard
        and str(r.phase.MODULE / "components/portfolio_training.py") in guard
    )
    r.SOURCE_GUARD = {str(r.CONTINUATION_CONFIG): "changed"}
    with pytest.raises(ValueError, match="source changed"):
        r.stage(tmp_path, "homecredit", "run", tmp_path)
