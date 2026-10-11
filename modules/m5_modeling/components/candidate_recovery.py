"""Recover completed authorized guest artifacts without rerunning a fit or inventing exits."""

from __future__ import annotations
import json
from pathlib import Path
import re
import shutil
import candidate_vm as provider  # type: ignore[import-not-found]


def recover(original, destination):
    """Both guest notebook receipts precede export; original host status stays untouched."""
    original, destination = Path(original), Path(destination)
    run_id = original.name
    if not re.fullmatch(r"[a-zA-Z0-9_]+", run_id):
        raise ValueError("Unsafe run identifier")
    if destination.exists():
        raise FileExistsError("Recovery requires a new immutable destination")
    for owner in provider.vm.PARTIES:
        receipt = json.loads(
            provider.vm.guest(
                owner,
                f"sudo cat /srv/vfl/{owner}/{run_id}/results/NOTEBOOK_COMPLETED.json",
            )
        )
        if receipt != {"status": "passed", "code_cells": provider.CELL_COUNT}:
            raise ValueError(
                "Guest notebook incomplete; recovery cannot substitute execution"
            )
    destination.mkdir(parents=True)
    try:
        shutil.copyfile(original / "declaration.json", destination / "declaration.json")
        for owner in provider.vm.PARTIES:
            (destination / owner).mkdir()
            provider.export(destination, run_id, owner)
            shutil.copyfile(
                original / owner / "logs/reference_lineage.json",
                destination / owner / "logs/reference_lineage.json",
            )
        provider.vm.save(
            destination / "status.json",
            {
                "status": "passed",
                "origin": "authorized_guest_recovery",
                "original_host_exit_status": "unknown",
                "retrained": False,
            },
        )
        provider.vm.private_permissions(destination)
        verification = provider.verify(destination)
        provider.vm.save(destination / "verification.json", verification)
    except BaseException as error:
        provider.vm.save(
            destination / "status.json",
            {
                "status": "failed",
                "origin": "authorized_guest_recovery",
                "original_host_exit_status": "unknown",
                "retrained": False,
            },
        )
        provider.vm.save(
            destination / "RECOVERY_FAILED.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "retrained": False,
            },
        )
        raise
    finally:
        provider.vm.private_permissions(destination)
    return destination
