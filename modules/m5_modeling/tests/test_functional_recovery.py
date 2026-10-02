"""Recovery refuses incomplete/changed fits; fixtures are wholly synthetic."""
import hashlib
import json
import shutil
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import functional_recovery as recovery  # noqa: E402


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture(params=["alice", "bob"])
def recovered_fixture(tmp_path, monkeypatch, request):
    party = request.param
    origin, root = [tmp_path / party / name for name in ["origin", "recovery"]]
    shared = {"model_seeds": [11, 22, 33, 44, 55], "learning_rates": [0.1], "l0_c_grid": [1], "epochs": 2}
    cfg = {"shared": shared, "protocol": "synthetic", "field": "synthetic", "datasets": {"synthetic": {"treatment": False}}}
    for p in [origin, root]:
        for name in ["input", "code", "data", "logs", "results", "trainings"]:
            (p / name).mkdir(parents=True)
        (p / "code/local_functional.yaml").write_text(yaml.safe_dump(cfg))
    manifest = {}
    values = {"x": np.ones((2, 1))}
    if party == "alice":
        values["y"] = np.array([0., 1.])
    for split in recovery.SPLITS:
        for p in [origin, root]:
            (p / "input" / f"{split}.csv").write_text("synthetic input only\n")
        manifest[f"{split}.csv"] = recovery.fingerprint(root / "input" / f"{split}.csv")
        np.savez_compressed(origin / "data" / f"{split}_local.npz", **values)
    write(origin / "input/manifest.json", {**manifest, "clean_features.csv": "unused preparation artifact"})
    write(root / "input/manifest.json", manifest)
    write(origin / "logs/tls_checks.json", {"synthetic_authenticated_training": True})
    routes = ["L0", "L1_secure", "L3_secure"] if party == "alice" else ["L1_secure", "L3_secure"]
    for route in routes:
        for seed in shared["model_seeds"]:
            count = 2 if route == "L0" else 1
            for candidate in range(count):
                f = root / "trainings" / f"{route}_seed{seed}_c{candidate}"
                write(f / "code/training_config.json", {"route": route, "seed": seed})
                source = f / "code/source.py"
                source.write_text("# frozen synthetic fit source\n")
                write(f / "code/runtime.json", {"code_sha256": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}})
                write(f / "data/input_references.json", {"derived_input": {
                    file.name: recovery.fingerprint(file) for file in (origin / "data").iterdir()}})
                write(f / "logs/progress.json", {"completed_epochs": 2})
                if party == "alice":
                    write(f / "results/status.json", {"training": "passed"})
                    for split in recovery.SPLITS:
                        (f / "results" / f"{split}_predictions.npy").write_bytes(b"synthetic placeholder, not loaded")
                shutil.copytree(f, origin / "trainings" / f.name)
    data = {"party": party, "root": str(root), "manifest": manifest, **{split: values for split in recovery.SPLITS}}
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    monkeypatch.setattr(recovery, "prepare_local", lambda *args: data)
    monkeypatch.setattr(recovery, "finalize_alice", lambda *args: {"status": "completed", "synthetic_evaluation": True})
    runtime = {"party": party, "run_id": root.name, "origin_run_id": origin.name,
               "dataset": "synthetic", "config": str(root / "code/local_functional.yaml")}
    return runtime, root, origin


def test_completed_fit_recovery_ignores_only_undelivered_preparation_artifact(recovered_fixture):
    runtime, root, origin = recovered_fixture
    before = recovery.fingerprint(origin / "input/manifest.json")
    result = recovery.recover_local(runtime)
    assert result["status"] == "completed"
    record = json.loads((root / "logs/recovery.json").read_text())
    assert not record["retraining"] and not record["retuning"]
    assert recovery.fingerprint(origin / "input/manifest.json") == before


@pytest.mark.parametrize("mutation", ["input", "arrays", "source", "schedule", "parameters", "authentication"])
def test_changed_or_incomplete_origin_cannot_be_recovered(recovered_fixture, mutation):
    runtime, root, origin = recovered_fixture
    if mutation == "input":
        (origin / "input/train.csv").write_text("changed")
    elif mutation == "arrays":
        np.savez_compressed(origin / "data/train_local.npz", x=np.zeros((2, 1)))
    elif mutation == "source":
        next((root / "trainings").glob("*/code/source.py")).write_text("# changed")
    elif mutation == "schedule":
        next((root / "trainings").glob("L3*/logs/progress.json")).write_text('{"completed_epochs":0}')
    elif mutation == "authentication":
        write(origin / "logs/tls_checks.json", {"synthetic_authenticated_training": False})
    else:
        cfg = yaml.safe_load((root / "code/local_functional.yaml").read_text())
        cfg["shared"]["epochs"] = 1
        (root / "code/local_functional.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        recovery.recover_local(runtime)
