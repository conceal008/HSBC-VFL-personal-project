"""The P4 audit must reject optimistic configuration without real proof."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "components/psi_release.py"
SPEC = importlib.util.spec_from_file_location("psi_release", SOURCE)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
CONFIG = SOURCE.parents[1] / "configs/psi_release.yaml"


def test_pinned_psi_df_plaintext_output_blocks_default_policy():
    config, fingerprint = audit.load_config(CONFIG)
    report = audit.assess(config, fingerprint)
    assert report["status"] == "blocked"
    assert report["actual_plaintext_recipients"] == ["alice"]
    assert not report["checks"]["no_plaintext_intersection"]
    assert not report["psi_executed"] and not report["dataset_rows_read"]
    assert report["business_identity_limitation"]


def test_broadcast_exposes_both_and_config_cannot_fake_release():
    config, fingerprint = audit.load_config(CONFIG)
    config["protocol"]["broadcast_result"] = True
    assert audit.plaintext_recipients(config) == ["alice", "bob"]
    config["protocol"]["broadcast_result"] = False
    config["policy"]["s0_9_decision"] = "approved"
    config["policy"]["permitted_plaintext_recipients"] = ["alice"]
    config["deployment"].update(hosts=["host-a", "host-b"],
                                  fed_mutual_tls=True, spu_mutual_tls=True)
    config["identifiers"]["kind"] = "verified_customer_key"
    report = audit.assess(config, fingerprint)
    assert report["status"] == "blocked"
    assert not report["checks"]["secret_shared_adapter_audited"]
    assert not report["checks"]["two_distinct_hosts_attested"]
    assert not report["checks"]["mutual_tls_handshake_verified"]


def test_unknown_api_and_invalid_receiver_fail_closed():
    config, fingerprint = audit.load_config(CONFIG)
    config["api"] = "unknown_adapter"
    report = audit.assess(config, fingerprint)
    assert report["status"] == "blocked"
    assert not report["checks"]["api_semantics_known"]
    config["api"] = audit.SUPPORTED_API
    config["protocol"]["receiver"] = "outsider"
    with pytest.raises(ValueError, match="receiver"):
        audit.assess(config, fingerprint)


def test_missing_contract_field_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("api: secretflow_spu_psi_df_1_14\n")
    with pytest.raises(ValueError, match="Missing P4"):
        audit.load_config(path)
