"""Normalize historical execution metadata without reading model or row-data files."""

from __future__ import annotations

import json
from pathlib import Path

KNOWN_STATES = {
    "passed",
    "failed",
    "completed",
    "interrupted",
    "running",
    "blocked",
    "not_verified",
}
STATE_FIELDS = ("status", "training", "l0_status")
CAPABILITY_FIELDS = (
    "psi",
    "secure_joint_training",
    "model_leakage_evaluation",
    "separate_host",
)
BOOLEAN_FIELDS = (
    "test_read",
    "probabilities_valid",
    "native_equivalence",
    "dataset_verified_complete",
    "preflight",
    "recovered",
    "finalization_recovered",
    "retrained",
    "source_code_changed",
)


def classify(payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected a status object")
    states = {
        k: payload[k]
        for k in STATE_FIELDS
        if isinstance(payload.get(k), str) and payload[k] in KNOWN_STATES
    }
    driver = payload.get("driver_status")
    if (
        isinstance(driver, dict)
        and isinstance(driver.get("status"), str)
        and driver["status"] in KNOWN_STATES
    ):
        states["driver_status.status"] = driver["status"]
    candidates = [k for k in (*STATE_FIELDS, "driver_status.status") if k in states]
    field = candidates[0] if candidates else None
    codes = None
    if "exit_codes" in payload:
        codes = payload["exit_codes"]
        if codes is not None and (
            not isinstance(codes, list)
            or not all(v is None or type(v) is int for v in codes)
        ):
            raise ValueError("Invalid exit code list")
    elif "exit_code" in payload:
        code = payload["exit_code"]
        if code is not None and type(code) is not int:
            raise ValueError("Invalid exit code")
        codes = [code]
    nonzero = codes is not None and any(v is not None and v != 0 for v in codes)
    exit_state = (
        ("unknown_or_partial" if "exit_codes" in payload else "not_recorded")
        if codes is None
        else "known_nonzero"
        if nonzero
        else "empty"
        if not codes
        else "unknown_or_partial"
        if any(v is None for v in codes)
        else "all_reported_zero"
    )
    return {
        "recorded_state": states[field]
        if field
        else ("preflight_only" if payload.get("preflight") is True else "unclassified"),
        "state_source": field,
        "all_recorded_states": states,
        "exit_codes": codes,
        "exit_state": exit_state,
        "recorded_failure": nonzero
        or any(v in {"failed", "interrupted"} for v in states.values())
        or bool(payload.get("cleanup_errors")),
        "status_exit_conflict": bool(
            field and states[field] in {"passed", "completed"} and nonzero
        ),
        "historical_running": any(v == "running" for v in states.values()),
        "current_activity": "not_inferred_from_historical_file",
        "capabilities": {
            k: payload[k]
            for k in CAPABILITY_FIELDS
            if isinstance(payload.get(k), str)
            and payload[k] in {"blocked", "not_verified"}
        },
        "checks": {
            k: payload[k] for k in BOOLEAN_FIELDS if type(payload.get(k)) is bool
        },
        "score_access": payload.get("score_access")
        if payload.get("score_access") in {"not_granted", "alice_only"}
        else None,
        "cleanup_failed": bool(payload.get("cleanup_errors")),
    }


def rebuild(source_index, output, current_execution):
    """Write a new metadata-only version, refusing overwrite and non-status inputs."""
    source_index, output = Path(source_index), Path(output)
    if current_execution not in {"paused_by_user", "resumed_by_user"}:
        raise ValueError("Explicit current execution state is required")
    if output.exists():
        raise FileExistsError("Historical index is immutable; use a new version")
    source = json.loads(source_index.read_text())
    items = []
    seen = set()
    for old in source:
        file = Path(old["path"])
        if file.name != "status.json" or str(file) in seen:
            raise ValueError("Expected unique status.json references")
        seen.add(str(file))
        try:
            item = classify(json.loads(file.read_text()))
        except (OSError, ValueError, TypeError) as error:
            item = {
                "recorded_state": "unreadable_or_invalid_preserved",
                "error_type": type(error).__name__,
                "current_activity": "not_inferred_from_historical_file",
            }
        items.append(dict(path=str(file), **item))
    result = {
        "schema_version": 2,
        "source_index": str(source_index),
        "scope": "same frozen historical status-file set; no live activity inferred",
        "current_execution_status": current_execution,
        "items": items,
    }
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    output.chmod(0o600)
    return result
