"""Governance-only archive contracts; these helpers never start an experiment."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess

import yaml

PAUSED = "paused_by_user"
RESUMED = "resumed_by_user"
RECORD_DIRECTORY = "registry/decision_records"


def safe_path(repository, relative):
    repository = Path(repository).resolve()
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected a repository-relative path")
    resolved = (repository / path).resolve()
    if not resolved.is_relative_to(repository):
        raise ValueError("Path escapes repository")
    return repository / path


def owner_record(repository, reference):
    if (
        not isinstance(reference, str)
        or Path(reference).parent.as_posix() != RECORD_DIRECTORY
    ):
        raise ValueError("Resume requires a registered owner authorization record")
    file = safe_path(repository, reference)
    record = yaml.safe_load(file.read_text())
    if not isinstance(record, dict):
        raise ValueError("Owner authorization record must be an object")
    if (
        record.get("status") != "active"
        or record.get("authorized_by") != "project_owner"
        or not isinstance(record.get("user_instruction"), str)
        or not record["user_instruction"].strip()
        or record.get("dr_id") != file.stem
    ):
        raise ValueError("Record does not contain owner authorization")
    return record


def execution_mode(repository, current):
    """Validate existing authorization metadata; never grant or execute permission."""
    if current.get("status") == PAUSED:
        return PAUSED
    if current.get("status") != RESUMED:
        raise ValueError("Unknown research execution state")
    record = owner_record(repository, current.get("resume_authorization_record"))
    if record.get("resume_research_experiments") is not True:
        raise ValueError("Record does not authorize research resumption")
    return RESUMED


def validate_archive(repository, inventory, current):
    mode = execution_mode(repository, current)
    originals = set()
    archived = set()
    promoted = []
    for item in inventory["items"]:
        source, saved = item["original_path"], item["archived_path"]
        if source in originals or saved in archived:
            raise ValueError("Duplicate archive mapping")
        originals.add(source)
        archived.add(saved)
        active = safe_path(repository, source)
        path = safe_path(repository, saved)
        if (
            not path.name.endswith(".pending")
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ValueError("Paused asset is missing or executable by discovery")
        raw = path.read_bytes()
        if (
            len(raw) != item["bytes"]
            or hashlib.sha256(raw).hexdigest() != item["sha256"]
        ):
            raise ValueError("Archived bytes changed")
        if active.exists():
            promoted.append(source)
    if promoted:
        reference = current.get("archive_promotion_record") or current.get(
            "resume_authorization_record"
        )
        if not reference:
            raise ValueError("Draft promoted without an owner authorization record")
        record = owner_record(repository, reference)
        scopes = record.get("promoted_original_paths")
        if (
            record.get("promote_paused_assets") is not True
            or not isinstance(scopes, list)
            or not all(isinstance(p, str) for p in scopes)
        ):
            raise ValueError("Existing promoted drafts lack an explicit owner scope")
        for path in scopes:
            safe_path(repository, path)
        if not set(promoted) <= set(scopes):
            raise ValueError("Draft promoted outside the recorded owner scope")
        # Stopping experiments again must not require deleting valid components.
        for source in promoted:
            if "components" in Path(source).parts and Path(source).suffix == ".py":
                declarations = Path(repository) / "registry/component_declarations"
                declared = any(
                    (yaml.safe_load(p.read_text()) or {}).get("path") == source
                    for p in declarations.glob("*.yaml")
                )
                if not declared:
                    raise ValueError("Promoted component is not registered")
    return mode


def original_blob(repository, commit, relative):
    safe_path(repository, relative)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected a complete Git commit fingerprint")
    return subprocess.check_output(
        ["git", "show", commit + ":" + relative], cwd=repository
    )
