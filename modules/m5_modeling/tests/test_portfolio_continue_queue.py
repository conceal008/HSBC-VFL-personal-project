"""Recovery orchestration must preserve completed assets and record new failures."""

from pathlib import Path
import importlib.util
import json

import pytest
import yaml

SOURCE = Path(__file__).resolve().parents[1] / "experiments/portfolio_continue_queue.py"
SPEC = importlib.util.spec_from_file_location("portfolio_continue_entry", SOURCE)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


@pytest.fixture
def setup_queue(tmp_path, monkeypatch):
    phase = entry.provider.phase
    module = tmp_path / "module"
    (module / "configs").mkdir(parents=True)
    (module / "configs/portfolio_training.yaml").write_text(yaml.safe_dump({
        "datasets": {"done": {}, "next": {}}, "shared": {"model_seeds": [11, 22]}
    }))
    previous = tmp_path / "previous"
    previous.mkdir()
    done = previous / "done/aggregate"
    done.mkdir(parents=True)
    (done / "verification.json").write_text('{"verification":"passed"}')
    preserved = previous / "next/seed11"
    preserved.mkdir(parents=True)
    state = dict(status="failed", completed=[str(done)], current_dataset="next",
                 current_seed=22, phases={"11": str(preserved)})
    (previous / "status.json").write_text(json.dumps(state))
    guard = tmp_path / "guard.json"
    guard.write_text("{}")
    monkeypatch.setattr(phase, "MODULE", module)
    for name in ("stage", "verify", "complete_source", "PHASE_NOTEBOOK", "MERGE_NOTEBOOK"):
        monkeypatch.setattr(phase, name, getattr(phase, name))
    for name in ("SOURCE_GUARD", "ELIGIBILITY_OUTPUT"):
        monkeypatch.setattr(entry.provider, name, getattr(entry.provider, name))
    monkeypatch.setattr(phase.vm, "lima", phase.vm.lima)
    monkeypatch.setattr(phase.vm, "private_permissions", lambda *args: None)
    verified = []
    monkeypatch.setattr(entry.provider, "verify", lambda p: verified.append(p))
    return previous, tmp_path / "new", guard, done, preserved, verified


def test_continue_reuses_whole_assets_without_retraining_them(setup_queue, monkeypatch):
    previous, output, guard, done, preserved, verified = setup_queue
    original = (previous / "status.json").read_bytes()
    calls = []

    def launch(workspace, root, dataset, **kwargs):
        calls.append((dataset, kwargs))
        return root / dataset / ("new_seed" if "seed" in kwargs else "aggregate")

    monkeypatch.setattr(entry.provider.phase, "launch", launch)
    entry.run(previous.parent, previous, output, guard)
    result = json.loads((output / "status.json").read_text())
    assert result["status"] == "passed" and result["completed"][0] == str(done)
    assert calls[0] == ("next", {"seed": 22}) and len(calls) == 2
    assert calls[1][1]["phases"]["11"] == str(preserved)
    assert verified == [preserved]
    assert (previous / "status.json").read_bytes() == original


def test_new_failure_keeps_prior_completion_and_failure_evidence(setup_queue, monkeypatch):
    previous, output, guard, done, _, _ = setup_queue

    def failed(*args, **kwargs):
        raise RuntimeError("native test failure")

    monkeypatch.setattr(entry.provider.phase, "launch", failed)
    with pytest.raises(RuntimeError, match="native test failure"):
        entry.run(previous.parent, previous, output, guard)
    result = json.loads((output / "status.json").read_text())
    assert result["status"] == "failed" and result["completed"] == [str(done)]
    assert result["error_type"] == "RuntimeError" and result["current_seed"] == 22


def test_missing_original_source_rejected_before_output_creation(setup_queue):
    previous, output, guard, _, _, _ = setup_queue
    guard.write_text('{"missing_source.py":"bad"}')
    with pytest.raises(FileNotFoundError):
        entry.run(previous.parent, previous, output, guard)
    assert not output.exists()
