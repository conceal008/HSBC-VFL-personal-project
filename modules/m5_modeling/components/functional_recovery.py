"""Party-local finalization recovery; preserves completed fits and the failed origin."""
from pathlib import Path
import json
import shutil

import numpy as np
import yaml

from functional_training import (ROOT, SPLITS, finalize_alice, fingerprint,  # type: ignore[import-not-found]
                                 prepare_local, verify_local_outputs)
from functional_metrics import save_json  # type: ignore[import-not-found]


def recover_local(runtime):
    party, run_id = runtime["party"], runtime["run_id"]
    root = ROOT / party / run_id
    original = ROOT / party / runtime["origin_run_id"]
    cfg = yaml.safe_load(Path(runtime["config"]).read_text())
    original_cfg = yaml.safe_load((original / "code/local_functional.yaml").read_text())
    if any(cfg[key] != original_cfg[key] for key in ("shared", "protocol", "field")):
        raise ValueError("Recovery must not change the frozen training or evaluation parameters")
    advisory_keys = {"alice_features", "bob_features"}
    spec = {key: value for key, value in cfg["datasets"][runtime["dataset"]].items() if key not in advisory_keys}
    original_spec = {key: value for key, value in original_cfg["datasets"][runtime["dataset"]].items() if key not in advisory_keys}
    if spec != original_spec or not all(json.loads((original / "logs/tls_checks.json").read_text()).values()):
        raise ValueError("Original specification or authentication evidence is invalid")
    original_manifest = json.loads((original / "input/manifest.json").read_text())
    manifest = json.loads((root / "input/manifest.json").read_text())
    for split in SPLITS:
        name = f"{split}.csv"
        if not fingerprint(original / "input" / name) == original_manifest[name] == manifest[name]:
            raise ValueError("Original consumed input changed; completed fits cannot be reused")
    data = prepare_local(party, run_id, runtime["dataset"], cfg)
    for split in SPLITS:
        with np.load(original / "data" / f"{split}_local.npz") as saved:
            if set(saved.files) != set(data[split]) or any(not np.array_equal(saved[key], data[split][key]) for key in saved.files):
                raise ValueError("Recovery preprocessing differs from the completed training")
    records = []
    for folder in sorted((root / "trainings").iterdir()):
        detail = json.loads((folder / "code/training_config.json").read_text())
        fit_runtime = json.loads((folder / "code/runtime.json").read_text())
        if any(fingerprint(folder / "code" / name) != digest for name, digest in fit_runtime["code_sha256"].items()):
            raise ValueError("Completed fit source changed")
        references = json.loads((folder / "data/input_references.json").read_text())
        if any(fingerprint(original / "data" / name) != digest for name, digest in references["derived_input"].items()):
            raise ValueError("Original processing artifact changed after fitting")
        route = "L0" if folder.name.startswith("L0_") else detail["route"]
        if route != "L0":
            progress = json.loads((folder / "logs/progress.json").read_text())
            if progress["completed_epochs"] != cfg["shared"]["epochs"]:
                raise ValueError("Incomplete secure training cannot be recovered as completed")
        if party == "alice":
            status = json.loads((folder / "results/status.json").read_text())
            if status["training"] != "passed" or any(not (folder / "results" / f"{split}_predictions.npy").exists() for split in SPLITS):
                raise ValueError("Completed prediction output missing")
            if any(fingerprint(folder / "results" / f"{split}_predictions.npy") !=
                   fingerprint(original / "trainings" / folder.name / "results" / f"{split}_predictions.npy") for split in SPLITS):
                raise ValueError("Recovery prediction differs from the completed fit")
        records.append({"name": folder.name, "route": route, "seed": detail["seed"]})
    shared = cfg["shared"]
    for route in (["L0", "L1_secure", "L3_secure"] if party == "alice" else ["L1_secure", "L3_secure"]):
        count = len(shared["l0_c_grid"]) + 1 if route == "L0" else len(shared["learning_rates"])
        for seed in shared["model_seeds"]:
            if sum(record["route"] == route and record["seed"] == seed for record in records) != count:
                raise ValueError("Completed candidate schedule is incomplete")
    verify_local_outputs(data, records)
    shutil.copyfile(original / "logs/tls_checks.json", root / "logs/tls_checks.json")
    save_json(root / "logs/recovery.json", {"origin_run_id": original.name,
        "consumed_input_unchanged": True, "preprocessing_arrays_equal": True,
        "completed_fit_sources_unchanged": True, "retraining": False, "retuning": False,
        "authentication_evidence": "original actual training TLS negative tests"})
    result = finalize_alice(data, records) if party == "alice" else {"status": "completed", "score_access": "not_granted"}
    save_json(root / "results/status.json", result)
    return result
