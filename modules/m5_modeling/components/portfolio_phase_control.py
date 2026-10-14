"""Control-plane recovery only; retain the original nonlinear training math."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import portfolio_training as nf  # type: ignore[import-not-found]

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
    for split in nf.SPLITS if receipt["reused_fits"] else ():
        current = np.load(root / "data" / (split + "_local.npz"))
        previous = np.load(root / "recovery_input" / (split + "_local.npz"))
        if set(current.files) != set(previous.files) or any(
            not np.array_equal(current[key], previous[key]) for key in current.files
        ):
            raise ValueError("Recovered input arrays differ")
    (root / "logs/recovery_arrays.json").write_text(
        json.dumps(
            {
                "owner_arrays_exact": bool(receipt["reused_fits"]),
                "comparison_required": bool(receipt["reused_fits"]),
                "no_peer_plaintext_read": True,
            }
        )
        + "\n"
    )
    return data


def local_references_recovered(data, seed, run_id):
    receipt = manifest(data["root"])
    names = {
        f"L0_{family}_{int(enhanced)}": f"L0_{family}_{int(enhanced)}_seed{seed}"
        for family in ("LR", "tree")
        for enhanced in (False,)
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
    seed_guards = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.For)
            and ast.unparse(node.target) == "seed"
            and ast.unparse(node.iter) == "shared['model_seeds']"
        ):
            guard = ast.parse(
                "if seed != runtime['selected_model_seed']: continue"
            ).body
            node.body[FIRST_STATEMENT:FIRST_STATEMENT] = guard
            seed_guards += 1
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
    if guards != 1 or seed_guards != 1:
        raise ValueError("Unexpected original controller shape")
    return ast.fix_missing_locations(tree)


def train_recovered(runtime):
    root = Path(runtime["config"]).parent
    if (
        hashlib.sha256(Path(nf.__file__).read_bytes()).hexdigest()
        != runtime["code_sha256"]["portfolio_training.py"]
    ):
        raise ValueError("Original math source changed")
    for filename, digest in runtime["controller_sha256"].items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != digest:
            raise ValueError("Recovery controller changed")
    if (
        runtime["selected_model_seed"]
        not in __import__("yaml").safe_load(Path(runtime["config"]).read_text())[
            "shared"
        ]["model_seeds"]
    ):
        raise ValueError("Unregistered phase seed")
    namespace = dict(nf.__dict__)
    namespace.update(
        prepare=prepare_phase,
        finalize=finalize_phase,
        local_references=local_references_recovered,
        validate_recovered=validate_recovered,
        save_json=save_phase_json,
    )
    exec(
        compile(
            patched_tree(),
            str(root / "portfolio_phase_control.py") + ":original-math-controller",
            "exec",
        ),
        namespace,
    )
    return namespace["train_full"](runtime)


def prepare_phase(party, runtime, cfg):
    data = prepare_recovered(party, runtime, cfg)
    data["phase_seed"] = runtime["selected_model_seed"]
    return data


def save_phase_json(path, payload):
    """Describe the selected session truthfully; numerical functions are untouched."""
    if Path(path).name == "output_boundary.json":
        payload = dict(
            payload, model_count=2, scope="single_seed_session"
        )  # 魔数豁免: 两格A0/B0
    return nf.save_json(path, payload)


def finalize_phase(data, records):
    seed = data["phase_seed"]
    if set(records) != {str(seed)} or set(records[str(seed)]) != {
        "A0",
        "B0",
        "L0_LR_0",
        "L0_tree_0",
    }:
        raise ValueError("Single phase fit block incomplete")
    root = Path(data["root"])
    (root / "results/phase_records.json").write_text(json.dumps(records))
    (root / "results/特征工程结果分析.md").write_text(
        "单seed物理会话完成，五seed汇总CI待合并；封存未读。\n"
    )
    return {
        "status": "phase_completed",
        "seed": seed,
        "independent_confirmation": False,
    }
