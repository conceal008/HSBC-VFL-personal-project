"""Resume a failed queue using unchanged published owner phase/merge operators.

Run with the project venv: python this_file WORKSPACE PRIOR_QUEUE NEW_QUEUE GUARD.
GUARD is the original JSON map of repository-relative source names to SHA256.
No private data or model weights are read by this host orchestration entry.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import yaml

REPOSITORY = Path(__file__).resolve().parents[3]  # 魔数豁免: experiments入口到仓库根的目录层级
sys.path.insert(0, str(REPOSITORY / "modules/m5_modeling/components"))
import portfolio_resume_vm as provider  # noqa: E402


def run(workspace, previous, output, guard):
    """Reuse only passed whole tasks/phases and start incomplete seeds afresh."""
    workspace, previous, output = map(Path, (workspace, previous, output))
    phase = provider.phase
    phase.vm.lima = phase.bounded_lima
    original = phase.read(previous / "status.json")
    if original["status"] != "failed":
        raise ValueError("Expected an explicitly failed prior queue")
    cfg = yaml.safe_load((phase.MODULE / "configs/portfolio_training.yaml").read_text())
    guard_map = phase.read(guard)
    provider.SOURCE_GUARD = {
        str(REPOSITORY / path): digest for path, digest in guard_map.items()
    }
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h
           for p, h in provider.SOURCE_GUARD.items()):
        raise ValueError("Original source guard differs")
    completed = list(original["completed"])
    expected = list(cfg["datasets"])
    if [Path(p).parent.name for p in completed] != expected[:len(completed)]:
        raise ValueError("Completed task prefix invalid")
    for path in completed:
        if phase.read(Path(path) / "verification.json")["verification"] != "passed":
            raise ValueError("Prior task verification absent")
    output.mkdir(mode=phase.vm.PRIVATE_DIRECTORY_MODE, exist_ok=False)
    provider.ELIGIBILITY_OUTPUT = output.parent / (output.name + "_eligibility")
    provider.ELIGIBILITY_OUTPUT.mkdir(mode=phase.vm.PRIVATE_DIRECTORY_MODE)
    phase.vm.save(provider.ELIGIBILITY_OUTPUT / "source_guard.json", provider.SOURCE_GUARD)
    phase.stage = provider.stage
    phase.verify = provider.verify
    phase.complete_source = provider.healthy_source
    phase.PHASE_NOTEBOOK = "S5.X3_portfolio_phase.ipynb"
    phase.MERGE_NOTEBOOK = "S5.X3_portfolio_merge.ipynb"
    state = dict(status="running", pid=os.getpid(), completed=completed,
                 prior_queue=str(previous), current_dataset=None, current_seed=None,
                 phases={}, correction=1)
    phase.vm.save(output / "status.json", state)
    try:
        for dataset in expected[len(completed):]:
            phases = dict(original["phases"]) if dataset == original["current_dataset"] else {}
            if not set(map(int, phases)).issubset(cfg["shared"]["model_seeds"]):
                raise ValueError("Unregistered prior seed")
            state.update(current_dataset=dataset, phases=phases)
            for seed in cfg["shared"]["model_seeds"]:
                if str(seed) in phases:
                    path = Path(phases[str(seed)])
                    if path.parent.name != dataset:
                        raise ValueError("Prior phase dataset differs")
                    provider.verify(path)
                    continue
                state["current_seed"] = seed
                phase.vm.save(output / "status.json", state)
                path = phase.launch(workspace, output, dataset, seed=seed)
                phases[str(seed)] = str(path)
                phase.vm.save(output / "status.json", state)
            state["current_seed"] = None
            phase.vm.save(output / "status.json", state)
            path = phase.launch(workspace, output, dataset, phases=phases)
            completed.append(str(path))
            phase.vm.save(output / "status.json", state)
        state.update(status="passed", current_dataset=None)
    except BaseException as exc:
        state.update(status="failed", error_type=type(exc).__name__, error_message=str(exc))
        raise
    finally:
        phase.vm.save(output / "status.json", state)
        phase.vm.private_permissions(output)
        phase.vm.private_permissions(provider.ELIGIBILITY_OUTPUT)


if __name__ == "__main__":
    run(*sys.argv[1:])
