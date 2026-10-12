"""Registered fingerprints never exempt adjacent personal IDs or credentials."""
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci"))
from mask_declared_fingerprints import normalize, credential_keys  # noqa: E402

# Actual public protocol digest, assembled to avoid a raw false ID-shaped span.
DIGEST = "".join(("0176cae8", "e222c92c", "0f960999", "89246276",
                  "8256c747", "a979f85d", "65b5d8b7", "760afb4a"))


def test_only_registered_complete_digest_is_normalized():
    for field in ("config_hash", "source_config_sha256", "source_summary_sha256"):
        text = '"' + field + '": "' + DIGEST + '"'
        assert "DECLARED_SHA256" in normalize(text)
    for field in ("customer_sha256", "input_sha256", "identity_hash"):
        text = '"' + field + '": "' + DIGEST + '"'
        assert normalize(text) == text
    assert normalize('"config_hash": "' + DIGEST[:-1] + '"').endswith(DIGEST[:-1] + '"')
    assert normalize('"config_hash": "' + DIGEST + 'a"').endswith(DIGEST + 'a"')
    text = '{"config_hash":"' + DIGEST + '","card":"' + "1" * 18 + '"}'
    assert "1" * 18 in normalize(text)
    assert DIGEST in credential_keys(text)
    assert 'card:"' in credential_keys(text)


@pytest.mark.parametrize("mime", ["JSON data", "JSON text data"])
@pytest.mark.parametrize("leak", ["none", "card", "credential"])
def test_known_json_is_scanned_independently_of_file_mime(tmp_path, mime, leak):
    root = tmp_path / "repo"
    (root / "ci").mkdir(parents=True)
    for name in ("check_secrets.sh", "mask_declared_fingerprints.py"):
        (root / "ci" / name).write_bytes((ROOT / "ci" / name).read_bytes())
    fixture = '{"source_config_sha256":"' + DIGEST + '"'
    if leak == "card":
        fixture += ',"card":"' + "1" * 18 + '"'
    elif leak == "credential":
        fixture += ',"api_key":"' + "x" * 24 + '"'
    fixture += '}\n'
    (root / "summary.json").write_text(fixture)
    fake = tmp_path / "bin"
    fake.mkdir()
    tool = fake / "file"
    tool.write_text('#!/bin/sh\nprintf "%s\\n" "$1: ' + mime + '"\n')
    tool.chmod(0o755)
    import os
    environment = {**os.environ, "PATH": str(fake) + ":" + os.environ["PATH"]}
    result = subprocess.run(["bash", str(root / "ci/check_secrets.sh"), str(root)],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == (0 if leak == "none" else 1), result.stdout
    if leak == "card":
        assert "疑似银行卡号" in result.stdout
    if leak == "credential":
        assert "疑似凭证" in result.stdout
