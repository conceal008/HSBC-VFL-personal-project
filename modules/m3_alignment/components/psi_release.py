"""Fail-closed P4 release audit; it never opens identifiers or executes PSI."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_API = "secretflow_spu_psi_df_1_14"
ALLOWED_PARTIES = frozenset(("alice", "bob"))
STATUS_BLOCKED = "blocked"


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    config = yaml.safe_load(payload)
    if not isinstance(config, dict):
        raise ValueError("P4 config must be a mapping")
    for field in ("api", "protocol", "policy", "deployment", "identifiers"):
        if field not in config:
            raise ValueError(f"Missing P4 config field: {field}")
    return config, hashlib.sha256(payload).hexdigest()


def plaintext_recipients(config: dict[str, Any]) -> list[str]:
    """The pinned psi_df API returns aligned plaintext tables to receiver(s)."""
    if config["api"] != SUPPORTED_API:
        return []  # Unknown is not proof of secrecy; assess() fails closed.
    receiver = config["protocol"].get("receiver")
    if receiver not in ALLOWED_PARTIES:
        raise ValueError("PSI receiver must be one registered party")
    if config["protocol"].get("broadcast_result") is True:
        return sorted(ALLOWED_PARTIES)
    return [receiver]


def assess(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    """Assess declared readiness without treating config claims as attestations.

    A future secure-share adapter and physical-host/TLS verifiers must be coded and
    tested before the hard gates can turn true. A YAML edit cannot release P4.
    """
    recipients = plaintext_recipients(config)
    policy = config["policy"]
    deployment = config["deployment"]
    identifiers = config["identifiers"]
    api_known = config["api"] == SUPPORTED_API
    hosts = deployment.get("hosts") or []
    declared_distinct_hosts = (len(hosts) == len(ALLOWED_PARTIES)
                               and len(set(hosts)) == len(ALLOWED_PARTIES))
    channel_declared = bool(deployment.get("fed_mutual_tls")
                            and deployment.get("spu_mutual_tls"))
    checks = {
        "api_semantics_known": api_known,
        "intersection_policy_final": policy.get("s0_9_decision") == "approved",
        "no_plaintext_intersection": api_known and not recipients,
        "secret_shared_adapter_audited": False,
        "two_distinct_hosts_declared": declared_distinct_hosts,
        "two_distinct_hosts_attested": False,
        "mutual_tls_declared_for_both_channels": channel_declared,
        "mutual_tls_handshake_verified": False,
        "receiver_allowlist_matches_api": sorted(recipients)
            == sorted(policy.get("permitted_plaintext_recipients") or []),
        "stable_cross_party_identifiers": identifiers.get("kind") == "verified_customer_key",
    }
    protocol_gates = (
        "api_semantics_known", "intersection_policy_final", "no_plaintext_intersection",
        "secret_shared_adapter_audited", "two_distinct_hosts_declared",
        "two_distinct_hosts_attested", "mutual_tls_declared_for_both_channels",
        "mutual_tls_handshake_verified", "receiver_allowlist_matches_api",
    )
    blocked = [name for name in protocol_gates if not checks[name]]
    return {
        "status": STATUS_BLOCKED if blocked else "released",
        "config_sha256": config_sha256,
        "api": config["api"], "protocol": config["protocol"].get("name"),
        "actual_plaintext_recipients": recipients,
        "checks": checks, "protocol_blockers": blocked,
        "business_identity_limitation": not checks["stable_cross_party_identifiers"],
        "dataset_rows_read": False,
        "psi_executed": False,
        "secure_training_released": False,
    }
