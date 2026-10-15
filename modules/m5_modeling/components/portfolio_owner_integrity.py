"""Owner-local final inventory; return booleans, never row values or shares."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

from portfolio_recovery_health import inspect_block

SECURE_CELLS = ("A0", "B0")
LOCAL_CELLS = ("L0_LR_0", "L0_tree_0")
OWNER_FILE_MODE = 0o600
OWNER_DIRECTORY_MODE = 0o700


def inspect_owner(root, party, seeds, epochs, file_mode, directory_mode, aggregate=False):
    """Require explicit complete shares; Bob aggregate is checked through its phases."""
    root = Path(root)
    if party not in ("alice", "bob") or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid owner/seed inventory")
    if file_mode != OWNER_FILE_MODE or directory_mode != OWNER_DIRECTORY_MODE:
        raise ValueError("Owner permission contract cannot be weakened")
    secure = [f"{cell}_seed{seed}" for seed in seeds for cell in SECURE_CELLS]
    local = [f"{cell}_seed{seed}" for seed in seeds for cell in LOCAL_CELLS]
    expected = {root / "trainings" / name / "results/model.share" for name in secure}
    actual = set((root / "trainings").rglob("model.share"))
    if actual != expected or any(
        not f.is_file() or f.stat().st_size == 0
        or stat.S_IMODE(f.stat().st_mode) != file_mode for f in expected
    ):
        raise ValueError("Expected nonempty owner shares or file permissions invalid")
    if stat.S_IMODE((root / "trainings").stat().st_mode) != directory_mode:
        raise ValueError("Owner training directory permission invalid")
    matrices_checked = not aggregate or party == "alice"
    if matrices_checked:
        if not inspect_block(root, secure, party, epochs, True)["complete"]:
            raise ValueError("Secure owner block invalid")
        if party == "alice" and not inspect_block(root, local, party, epochs, False)["complete"]:
            raise ValueError("Local reference block invalid")
    manifest = json.loads((root / "code/recovery_manifest.json").read_text())
    permitted = set(secure + (local if party == "alice" else []))
    names = manifest["reused_fits"]
    if len(names) != len(set(names)) or not set(names) <= permitted or set(names) != set(manifest["fit_sha256"]):
        raise ValueError("Clone manifest inventory invalid")
    for name, items in manifest["fit_sha256"].items():
        for relative, digest in items.items():
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("Clone manifest path leaves owner fit")
            if hashlib.sha256((root / "trainings" / name / relative).read_bytes()).hexdigest() != digest:
                raise ValueError("Frozen clone differs from manifest")
    return {
        "expected_nonempty_owner_shares_600": True,
        "owner_block_checks_passed": True,
        "owner_matrices_checked_in_this_root": matrices_checked,
        "bob_aggregate_requires_verified_source_phases": aggregate and party == "bob",
        "frozen_copies_match_manifest": True,
        "trainings_directory_700": True,
        "row_values_exported": False,
    }
