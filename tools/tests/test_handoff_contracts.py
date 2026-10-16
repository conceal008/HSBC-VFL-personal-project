"""Regression controls for evidence prefixes, authorized restoration and metadata loss."""

from pathlib import Path
import hashlib
import json
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handoff_contract import execution_mode, validate_archive, original_blob  # noqa: E402
from handoff_inventory import classify, rebuild  # noqa: E402


def draft(tmp_path):
    saved = tmp_path / "draft/worker.py.pending"
    saved.parent.mkdir()
    saved.write_bytes(b"# unvalidated draft\n")
    item = dict(
        original_path="worker.py",
        archived_path="draft/worker.py.pending",
        bytes=saved.stat().st_size,
        sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),
    )
    return dict(items=[item])


def authorization(tmp_path):
    path = tmp_path / "registry/decision_records/DR-TEST-RESUME.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            dict(
                dr_id=path.stem,
                status="active",
                authorized_by="project_owner",
                resume_research_experiments=True,
                promote_paused_assets=True,
                promoted_original_paths=["worker.py"],
                user_instruction="Synthetic test authorization only; execute nothing.",
            )
        )
    )
    return dict(
        status="resumed_by_user",
        resume_authorization_record=str(path.relative_to(tmp_path)),
    )


def test_pause_blocks_promotion_but_recorded_owner_resume_can_restore(tmp_path):
    inventory = draft(tmp_path)
    assert (
        validate_archive(tmp_path, inventory, {"status": "paused_by_user"})
        == "paused_by_user"
    )
    (tmp_path / "worker.py").write_text("# changed only in synthetic fixture")
    with pytest.raises(ValueError, match="promoted"):
        validate_archive(tmp_path, inventory, {"status": "paused_by_user"})
    assert (
        validate_archive(tmp_path, inventory, authorization(tmp_path))
        == "resumed_by_user"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing_record",
        "false_permission",
        "missing_user",
        "wrong_owner",
        "wrong_id",
        "path_escape",
        "unknown_state",
    ],
)
def test_resume_validation_cannot_be_enabled_by_unrecorded_flag(tmp_path, fault):
    state = authorization(tmp_path)
    file = tmp_path / state["resume_authorization_record"]
    value = yaml.safe_load(file.read_text())
    if fault == "missing_record":
        state.pop("resume_authorization_record")
    elif fault == "false_permission":
        value["resume_research_experiments"] = False
    elif fault == "missing_user":
        value["user_instruction"] = ""
    elif fault == "wrong_owner":
        value["authorized_by"] = "agent"
    elif fault == "wrong_id":
        value["dr_id"] = "DR-OTHER"
    elif fault == "path_escape":
        state["resume_authorization_record"] = "../outside.yaml"
    else:
        state["status"] = "automatic"
    file.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError):
        execution_mode(tmp_path, state)


@pytest.mark.parametrize("fault", ["bytes", "duplicate", "escape"])
def test_archive_immutability_still_applies_after_authorized_resume(tmp_path, fault):
    inventory = draft(tmp_path)
    state = authorization(tmp_path)
    if fault == "bytes":
        (tmp_path / "draft/worker.py.pending").write_text("# altered")
    elif fault == "duplicate":
        inventory["items"].append(dict(inventory["items"][0]))
    else:
        inventory["items"][0]["archived_path"] = "../secret.pending"
    with pytest.raises(ValueError):
        validate_archive(tmp_path, inventory, state)


def test_l0_failure_and_blocked_pipeline_capabilities_are_not_lost():
    item = classify(
        {
            "exit_code": 71,
            "l0_status": "failed",
            "psi": "blocked",
            "secure_joint_training": "blocked",
            "model_leakage_evaluation": "blocked",
            "separate_host": "not_verified",
        }
    )
    assert item["recorded_state"] == "failed" and item["state_source"] == "l0_status"
    assert item["exit_codes"] == [71] and item["exit_state"] == "known_nonzero"
    assert item["recorded_failure"] and len(item["capabilities"]) == 4


def test_fit_passed_and_unknown_process_exits_are_separate_facts():
    item = classify({"training": "passed", "test_read": False})
    assert item["recorded_state"] == "passed" and item["state_source"] == "training"
    assert item["exit_state"] == "not_recorded" and item["checks"]["test_read"] is False
    recovered = classify(
        {"status": "passed", "exit_codes": [None, None], "finalization_recovered": True}
    )
    assert (
        recovered["exit_state"] == "unknown_or_partial"
        and not recovered["recorded_failure"]
    )
    assert recovered["checks"]["finalization_recovered"] is True
    null_exit = classify({"status": "passed", "exit_codes": None, "recovered": True})
    assert (
        null_exit["exit_state"] == "unknown_or_partial"
        and not null_exit["recorded_failure"]
    )
    conflict = classify({"status": "passed", "exit_codes": [0, 71]})
    assert conflict["status_exit_conflict"] and conflict["recorded_failure"]


def test_historical_running_is_not_reported_as_live_activity():
    item = classify(
        {"driver_status": {"status": "running"}, "dataset_verified_complete": False}
    )
    assert item["historical_running"] and item["state_source"] == "driver_status.status"
    assert item["current_activity"] == "not_inferred_from_historical_file"
    assert item["checks"]["dataset_verified_complete"] is False
    assert classify({"preflight": True})["recorded_state"] == "preflight_only"


def test_rebuild_preserves_old_index_and_marks_invalid_records(tmp_path):
    source = tmp_path / "old.json"
    a = tmp_path / "a/status.json"
    b = tmp_path / "b/status.json"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_text('{"training":"passed"}')
    b.write_text("")
    source.write_text(
        json.dumps([{"path": str(a), "status": None}, {"path": str(b), "status": None}])
    )
    original = source.read_bytes()
    output = tmp_path / "v2.json"
    result = rebuild(source, output, "paused_by_user")
    assert source.read_bytes() == original and len(result["items"]) == 2
    assert result["items"][0]["recorded_state"] == "passed"
    assert result["items"][1]["recorded_state"] == "unreadable_or_invalid_preserved"
    with pytest.raises(FileExistsError):
        rebuild(source, output, "paused_by_user")
    with pytest.raises(ValueError):
        rebuild(source, tmp_path / "bad.json", "auto")
    source.write_text(json.dumps([{"path": str(a)}, {"path": str(a)}]))
    with pytest.raises(ValueError):
        rebuild(source, tmp_path / "duplicates.json", "paused_by_user")
    with pytest.raises(ValueError):
        original_blob(tmp_path, "--bad", "file.md")


def test_restored_assets_can_remain_when_owner_pauses_again(tmp_path):
    inventory = draft(tmp_path)
    state = authorization(tmp_path)
    (tmp_path / "worker.py").write_text("# valid restored synthetic code")
    assert validate_archive(tmp_path, inventory, state) == "resumed_by_user"
    paused_again = {
        "status": "paused_by_user",
        "archive_promotion_record": state["resume_authorization_record"],
    }
    assert validate_archive(tmp_path, inventory, paused_again) == "paused_by_user"
    record = tmp_path / state["resume_authorization_record"]
    payload = yaml.safe_load(record.read_text())
    payload["promoted_original_paths"] = []
    record.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="scope"):
        validate_archive(tmp_path, inventory, paused_again)


def test_promoted_components_require_registration_and_scoped_owner_record(tmp_path):
    inventory = draft(tmp_path)
    state = authorization(tmp_path)
    relative = "modules/example/components/worker.py"
    inventory["items"][0]["original_path"] = relative
    active = tmp_path / relative
    active.parent.mkdir(parents=True)
    active.write_text("# synthetic component")
    record = tmp_path / state["resume_authorization_record"]
    payload = yaml.safe_load(record.read_text())
    payload["promoted_original_paths"] = [relative]
    record.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="registered"):
        validate_archive(tmp_path, inventory, state)
    declaration = tmp_path / "registry/component_declarations/example.yaml"
    declaration.parent.mkdir()
    declaration.write_text(yaml.safe_dump({"path": relative}))
    assert validate_archive(tmp_path, inventory, state) == "resumed_by_user"
    inventory["items"][0]["original_path"] = "../outside.py"
    with pytest.raises(ValueError, match="relative"):
        validate_archive(tmp_path, inventory, state)
