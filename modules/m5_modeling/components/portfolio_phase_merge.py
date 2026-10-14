"""Owner-only merge of five frozen phase predictions; no peer arrays or new fit."""

from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import yaml
import portfolio_training as nf
import portfolio_phase_control as control


def merge(runtime):
    if runtime["party"] != "alice":
        raise ValueError("Only the declared score recipient can evaluate")
    root = Path(runtime["config"]).parent.parent
    for name, digest in runtime["code_sha256"].items():
        if hashlib.sha256((root / "code" / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen mathematical source changed")
    for name, digest in runtime["controller_sha256"].items():
        if hashlib.sha256((root / "code" / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Merge control changed")
    cfg = yaml.safe_load(Path(runtime["config"]).read_text())
    data = nf.prepare("alice", runtime, cfg)
    receipt = control.manifest(root)
    seeds = cfg["shared"]["model_seeds"]
    expected = {
        f"{c}_seed{s}" for s in seeds for c in ("A0", "B0", "L0_LR_0", "L0_tree_0")
    }
    if (
        set(receipt["reused_fits"]) != expected
        or receipt["config_sha256"] != runtime["config_sha256"]
    ):
        raise ValueError("Five-seed input block incomplete")
    for phase in receipt["source_runs"]:
        for split in nf.SPLITS:
            old = np.load(root / "recovery_input" / phase / (split + "_local.npz"))
            new = np.load(root / "data" / (split + "_local.npz"))
            if set(old.files) != set(new.files) or any(
                not np.array_equal(old[k], new[k]) for k in new.files
            ):
                raise ValueError("Phase input arrays differ")
    for name in sorted(expected):
        control.validate_recovered(data, name)
    records = {
        str(s): {c: f"{c}_seed{s}" for c in ("A0", "B0", "L0_LR_0", "L0_tree_0")}
        for s in seeds
    }
    result = nf.finalize(data, records)
    nf.ad.manifest_check(root)
    nf.save_json(
        root / "logs/phase_merge.json",
        {
            "source_runs": receipt["source_runs"],
            "owner_arrays_exact": True,
            "no_peer_plaintext_read": True,
            "five_seeds_complete": True,
            "new_training_performed": False,
        },
    )
    nf.save_json(
        root / "logs/output_boundary.json",
        {
            "frozen_input_unchanged": True,
            "full_partition_training": True,
            "joint_weights": "secret shares only",
            "scores_receiver": "alice",
            "old_test_read": False,
            "model_count": len(seeds) * len(cfg["experiment"]["conditions"]),
            "scope": "five_phase_aggregate",
        },
    )
    return result
