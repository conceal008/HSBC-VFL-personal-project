"""Single-owner VM preparation, with no capability to read sealed confirmation."""

from __future__ import annotations
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

SPLITS = ("train", "validation")
MINIMUM_OUTCOME_CLASSES = 2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)


def denied(path, mode="rb"):
    try:
        with Path(path).open(mode):
            pass
    except OSError as e:
        return e.errno in {errno.EACCES, errno.EPERM}
    return False


def isolation(runtime):
    root = Path(runtime["root"])
    account = runtime["account"]
    checks = {
        "nonroot": os.geteuid() != 0,
        "expected_account": subprocess.check_output(["id", "-un"], text=True).strip()
        == account,
        "no_sudo": subprocess.run(
            ["sudo", "-n", "true"], capture_output=True, check=False
        ).returncode
        != 0,
        "no_host_mount": "virtiofs" not in Path("/proc/mounts").read_text(),
        "sealed_read_denied": denied(root / "sealed/probe"),
        "sealed_write_denied": denied(root / "sealed/probe", "r+b"),
        "input_write_denied": denied(root / "input/train.csv", "r+b"),
        "foreign_training_denied": denied("/srv/forbidden/peer_sentinel"),
    }
    try:
        with socket.create_connection(
            ("127.0.0.1", runtime["network_probe_port"]), timeout=1
        ):
            pass
    except OSError as e:
        checks["network_denied"] = runtime["network_probe_verified"] and e.errno in {
            errno.EACCES,
            errno.EPERM,
            errno.EHOSTUNREACH,
        }
    else:
        checks["network_denied"] = False
    if not all(checks.values()):
        raise ValueError("Preparation capability check failed: " + json.dumps(checks))
    return checks


def encode(frames, spec):
    """Train-only medians/scale/vocabulary; JSON state usable at one future confirmation."""
    if set(frames) != set(SPLITS):
        raise ValueError("Only development frames allowed")
    features = spec["features"]
    cats = spec["categorical"]
    nums = [n for n in features if n not in cats]
    train = frames["train"]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(
        train[nums]
    )
    scaler = StandardScaler().fit(imputer.transform(train[nums]))
    states = {
        "fit_split": "train",
        "numeric": nums,
        "median": imputer.statistics_.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "categorical": {},
        "output_features": ["numeric__" + n for n in nums],
    }
    for c in cats:
        vocabulary = sorted(set(train[c].fillna("__MISSING__").astype(str)))
        states["categorical"][c] = vocabulary
        states["output_features"] += ["categorical__" + c + "_" + v for v in vocabulary]
    result = {}
    for split, frame in frames.items():
        if not frame["record_id"].is_unique:
            raise ValueError("Duplicate record key")
        values = [scaler.transform(imputer.transform(frame[nums]))]
        for c, vocabulary in states["categorical"].items():
            column = frame[c].fillna("__MISSING__").astype(str).to_numpy()
            values.append(
                np.stack([column == v for v in vocabulary], axis=1).astype(float)
            )
        matrix = np.column_stack(values)
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite owner features")
        result[split] = pd.DataFrame(matrix, columns=states["output_features"])
        result[split].insert(0, "record_id", frame["record_id"].to_numpy())
    if set(frames["train"]["record_id"]) & set(frames["validation"]["record_id"]):
        raise ValueError("Development overlap")
    return result, states


def prepare(runtime):
    root = Path(runtime["root"])
    # Fail before reading or writing anything in a previously used output directory.
    if (
        any((root / "data").iterdir())
        or any((root / "results").iterdir())
        or (root / "logs/isolation.json").exists()
    ):
        raise FileExistsError("Frozen preparation outputs exist; use a new runtime directory")
    checks = isolation(runtime)
    if sha(root / "code/external_freeze.yaml") != runtime["config_sha256"]:
        raise ValueError("Config changed")
    for name, digest in runtime["code_sha256"].items():
        if sha(root / "code" / name) != digest:
            raise ValueError("Code changed")
    inputs = runtime["input_sha256"]
    frames = {}
    spec = runtime["feature_spec"]
    outcomes = runtime["outcomes"]
    if runtime["party"] == "bob" and outcomes:
        raise ValueError("Bob outcome disclosure")
    for split in SPLITS:
        path = root / "input" / f"{split}.csv"
        if sha(path) != inputs[path.name]:
            raise ValueError("Input changed")
        frame = pd.read_csv(path)
        if list(frame.columns) != ["record_id"] + spec["features"] + outcomes:
            raise ValueError("Owner schema differs")
        frames[split] = frame
    clean, state = encode(frames, spec)
    for split, frame in clean.items():
        if runtime["party"] == "alice":
            y = pd.to_numeric(frames[split][outcomes[0]])
            if runtime["dataset"] == "coil2000":
                y = (y > 0).astype(int)
            if not set(y.unique()) == {0, 1}:
                raise ValueError(
                    "Outcome support insufficient; no resampling permitted"
                )
            frame["label"] = y.to_numpy()
            if len(outcomes) > 1:
                treatment = pd.to_numeric(frames[split][outcomes[1]])
                if not set(treatment.unique()) == {0, 1}:
                    raise ValueError("Treatment support insufficient")
                frame["treatment"] = treatment.to_numpy(dtype=int)
        frame.to_csv(root / "data" / f"{split}.csv", index=False)
    save(root / "data/preprocessing_state.json", state)
    save(
        root / "data/manifest.json",
        {p.name: sha(p) for p in (root / "data").glob("*.csv")},
    )
    save(root / "logs/isolation.json", checks)
    if any(sha(root / "input" / name) != digest for name, digest in inputs.items()):
        raise ValueError("Frozen input changed")
    receipt = {
        "status": "prepared",
        "party": runtime["party"],
        "train_only_fit": True,
        "sealed_confirmation_read": False,
        "input_unchanged": True,
        "isolation_checks_passed": True,
        "training_performed": False,
        "physical_isolation": False,
    }
    save(root / "results/engineering.json", receipt)
    return receipt


def execute_preparation(runtime_path):
    """Check the entire fresh Notebook directory before its executor can write logs."""
    runtime = json.loads(Path(runtime_path).read_text())
    root = Path(runtime["root"])
    if any(any((root / sub).iterdir()) for sub in ("data", "results", "logs")):
        raise FileExistsError("Frozen Notebook outputs exist; use a new runtime directory")
    import functional_training as ft  # type: ignore[import-not-found]

    ft.ROOT = root.parents[1]
    ft.execute_notebook(runtime_path)
