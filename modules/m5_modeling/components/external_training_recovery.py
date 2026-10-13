"""Control-plane recovery only; retain the original nonlinear training math."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import nonlinear_full as nf  # type: ignore[import-not-found]

MANIFEST_NAME = "recovery_manifest.json"
FIRST_STATEMENT = 0


def manifest(root):
    return json.loads((Path(root) / "code" / MANIFEST_NAME).read_text())


def validate_recovered(data, name):
    root = Path(data["root"])
    receipt = manifest(root)
    if name not in receipt["reused_fits"]:
        raise ValueError("Undeclared recovered fit")
    folder = root / "trainings" / name
    for relative, digest in receipt["fit_sha256"][name].items():
        if hashlib.sha256((folder / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("Recovered fit changed")
    if data["party"] == "alice":
        for split in nf.SPLITS:
            prediction = np.load(folder / "results" / (split + "_predictions.npy"))
            if (
                len(prediction) != len(data[split]["x"])
                or not np.isfinite(prediction).all()
                or np.any((prediction < 0) | (prediction > 1))
            ):
                raise ValueError("Recovered probability/partition invalid")
    return True


def prepare_recovered(party, runtime, cfg):
    data = nf.prepare(party, runtime, cfg)
    root = Path(data["root"])
    receipt = manifest(root)
    if (
        receipt["party"] != party
        or receipt["config_sha256"] != runtime["config_sha256"]
    ):
        raise ValueError("Recovery ownership/protocol differs")
    for split in nf.SPLITS:
        current = np.load(root / "data" / (split + "_local.npz"))
        previous = np.load(root / "recovery_input" / (split + "_local.npz"))
        if set(current.files) != set(previous.files) or any(
            not np.array_equal(current[key], previous[key]) for key in current.files
        ):
            raise ValueError("Recovered input arrays differ")
    (root / "logs/recovery_arrays.json").write_text(
        json.dumps({"owner_arrays_exact": True, "no_peer_plaintext_read": True}) + "\n"
    )
    return data


def local_references_recovered(data, seed, run_id):
    receipt = manifest(data["root"])
    names = {
        f"L0_{family}_{int(enhanced)}": f"L0_{family}_{int(enhanced)}_seed{seed}"
        for family in ("LR", "tree")
        for enhanced in (False, True)
    }
    present = [name in receipt["reused_fits"] for name in names.values()]
    if any(present) and not all(present):
        raise ValueError("Partial local reference block not reusable")
    if all(present):
        for name in names.values():
            validate_recovered(data, name)
        return names
    return nf.local_references(data, seed, run_id)


def patched_tree():
    """Insert a skip guard immediately before the original design construction."""
    tree = ast.parse(inspect.getsource(nf.train_full))
    guards = 0
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.For)
            or ast.unparse(node.target) != "(cell, condition)"
        ):
            continue
        guard = ast.parse("""
name = f"{cell}_seed{seed}"
if name in runtime.get("reused_completed_fits", []):
    sf.wait([alice(validate_recovered)(da, name), bob(validate_recovered)(db, name)])
    records[str(seed)][cell] = name
    continue
""").body
        node.body[FIRST_STATEMENT:FIRST_STATEMENT] = guard
        guards += 1
    if guards != 1:
        raise ValueError("Unexpected original controller shape")
    return ast.fix_missing_locations(tree)


def train_recovered(runtime):
    root = Path(runtime["config"]).parent
    if (
        hashlib.sha256(Path(nf.__file__).read_bytes()).hexdigest()
        != runtime["code_sha256"]["nonlinear_full.py"]
    ):
        raise ValueError("Original math source changed")
    for filename, digest in runtime["controller_sha256"].items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != digest:
            raise ValueError("Recovery controller changed")
    namespace = dict(nf.__dict__)
    namespace.update(
        prepare=prepare_recovered,
        local_references=local_references_recovered,
        validate_recovered=validate_recovered,
    )
    exec(
        compile(
            patched_tree(),
            str(root / "external_training_recovery.py") + ":original-math-controller",
            "exec",
        ),
        namespace,
    )
    return namespace["train_full"](runtime)
