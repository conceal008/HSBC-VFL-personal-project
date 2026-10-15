"""Explicit inventory must reject missing, empty, wrongly permissioned or changed assets."""

from pathlib import Path
import hashlib
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import portfolio_owner_integrity as audit  # noqa: E402

FILE_MODE = 0o600
DIRECTORY_MODE = 0o700
EPOCHS = 12
SEEDS = [11]


@pytest.fixture
def owner(tmp_path, monkeypatch):
    (tmp_path / "code").mkdir()
    (tmp_path / "trainings").mkdir(mode=DIRECTORY_MODE)
    for cell in audit.SECURE_CELLS:
        f = tmp_path / "trainings" / f"{cell}_seed11/results/model.share"
        f.parent.mkdir(parents=True)
        f.write_bytes(b"unit fixture, not a real model")
        f.chmod(FILE_MODE)
    (tmp_path / "code/recovery_manifest.json").write_text('{"reused_fits":[],"fit_sha256":{}}')
    monkeypatch.setattr(audit, "inspect_block", lambda *args: {"complete": True})
    return tmp_path


def check(root, party="alice", **kwargs):
    return audit.inspect_owner(root, party, SEEDS, EPOCHS, FILE_MODE, DIRECTORY_MODE, **kwargs)


def test_complete_owner_and_bob_aggregate_scope(owner, monkeypatch):
    assert check(owner)["owner_matrices_checked_in_this_root"]
    assert check(owner, "bob")["owner_matrices_checked_in_this_root"]
    monkeypatch.setattr(audit, "inspect_block", lambda *args: pytest.fail("Bob aggregate has no new matrix"))
    result = check(owner, "bob", aggregate=True)
    assert not result["owner_matrices_checked_in_this_root"]
    assert result["bob_aggregate_requires_verified_source_phases"]


@pytest.mark.parametrize("damage", ["missing", "empty", "mode", "extra", "directory"])
def test_absent_or_invalid_share_cannot_pass_vacuously(owner, damage):
    f = owner / "trainings/A0_seed11/results/model.share"
    if damage == "missing":
        f.unlink()
    elif damage == "empty":
        f.write_bytes(b"")
    elif damage == "mode":
        f.chmod(DIRECTORY_MODE)
    elif damage == "extra":
        (owner / "trainings/model.share").write_bytes(b"extra")
    else:
        (owner / "trainings").chmod(FILE_MODE)
    with pytest.raises(ValueError):
        check(owner)


@pytest.mark.parametrize("invalid_block", [0, 1])
def test_owner_arrays_or_local_reference_failure_is_rejected(owner, monkeypatch, invalid_block):
    calls = []

    def health(*args):
        calls.append(args)
        return {"complete": len(calls) - 1 != invalid_block}

    monkeypatch.setattr(audit, "inspect_block", health)
    with pytest.raises(ValueError, match="block invalid"):
        check(owner)


def test_manifest_file_change_and_path_escape_rejected(owner):
    f = owner / "trainings/A0_seed11/results/model.share"
    manifest = {"reused_fits": ["A0_seed11"], "fit_sha256": {
        "A0_seed11": {"results/model.share": hashlib.sha256(f.read_bytes()).hexdigest()}
    }}
    path = owner / "code/recovery_manifest.json"
    path.write_text(json.dumps(manifest))
    assert check(owner)["frozen_copies_match_manifest"]
    f.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        check(owner)
    manifest["fit_sha256"]["A0_seed11"] = {"../outside": "unused"}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="path"):
        check(owner)
    manifest["reused_fits"].append("A0_seed11")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inventory"):
        check(owner)


def test_invalid_owner_or_no_seeds_cannot_create_empty_success(owner):
    for party, seeds in [("peer", SEEDS), ("alice", []), ("bob", [11, 11])]:
        with pytest.raises(ValueError, match="inventory"):
            audit.inspect_owner(owner, party, seeds, EPOCHS, FILE_MODE, DIRECTORY_MODE)
    with pytest.raises(ValueError, match="cannot be weakened"):
        audit.inspect_owner(owner, "alice", SEEDS, EPOCHS, DIRECTORY_MODE, DIRECTORY_MODE)
