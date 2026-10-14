"""Inspect only an owner's recovery assets; corrupt assets are never completed fits."""

from __future__ import annotations
import json
from pathlib import Path
import zipfile
import numpy as np


def inspect_block(root, names, party, epochs, secure):
    root = Path(root)
    reasons = []
    try:
        arrays = {
            s: np.load(root / "data" / (s + "_local.npz"))
            for s in ("train", "validation")
        }
        for s, a in arrays.items():
            if not a.files or any(not np.isfinite(a[k]).all() for k in a.files):
                raise ValueError("Invalid owner matrix")
            if party == "alice" and not {"x", "y", "t"}.issubset(a.files):
                raise ValueError("Owner outcome matrix incomplete")
        for name in names:
            folder = root / "trainings" / name
            if not all(
                (folder / p).is_dir() for p in ("code", "data", "results", "logs")
            ):
                raise ValueError("Fit directory incomplete")
            refs = json.loads((folder / "data/input_references.json").read_text())
            if set(refs["frozen_input"]) != {"train.csv", "validation.csv"} or any(
                "test" in n for n in refs["derived_input"]
            ):
                raise ValueError("Recovery capability invalid")
            if secure:
                progress = json.loads((folder / "logs/progress.json").read_text())
                if (
                    progress["completed_epochs"] != epochs
                    or progress["planned_epochs"] != epochs
                    or (folder / "results/model.share").stat().st_size == 0
                ):
                    raise ValueError("Secure fit incomplete")
            if party == "alice":
                status = json.loads((folder / "results/status.json").read_text())
                if (
                    status.get("training") != "passed"
                    or not status.get("probabilities_valid")
                    or status.get("test_read") is not False
                ):
                    raise ValueError("Owner probability receipt incomplete")
                for split, a in arrays.items():
                    p = np.load(folder / "results" / (split + "_predictions.npy"))
                    if (
                        len(p) != len(a["x"])
                        or not np.isfinite(p).all()
                        or np.any((p < 0) | (p > 1))
                    ):
                        raise ValueError("Recovery prediction invalid")
            elif list(folder.rglob("*predictions.npy")):
                raise ValueError("Peer scores present")
    except (OSError, ValueError, KeyError, EOFError, TypeError, zipfile.BadZipFile) as exc:
        reasons.append(type(exc).__name__)
    return {"complete": not reasons, "reasons": reasons, "no_arrays_exported": True}
