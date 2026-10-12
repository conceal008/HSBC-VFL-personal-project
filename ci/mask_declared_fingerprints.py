"""Normalize only registered SHA-256 values for personal-identifier scanning.

Secrets are scanned before this normalization. No complete line/file is exempt.
Input/customer hashes deliberately do not belong to this small registry.
"""
from __future__ import annotations
from pathlib import Path
import re
import sys

FIELDS = (
    "config_hash", "config_sha256", "source_config_sha256", "code_sha256",
    "source_result_sha256", "source_summary_sha256",
)
QUOTE = r"(?:\\?[\"'])?"
PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])" + QUOTE + "(?:" + "|".join(FIELDS) + ")"
    + QUOTE + r"\s*:\s*" + QUOTE + r"(?:sha256:)?"
    + r"(?P<digest>[0-9a-fA-F]{64})(?![0-9a-zA-Z_])"
)
KEY_PATTERN = re.compile(
    r"(?:\\?[\"'])(?P<key>[A-Za-z_][A-Za-z0-9_-]*)(?:\\?[\"'])(?=\s*[:=])"
)


def normalize(text):
    def replace(match):
        begin, end = match.span("digest")
        return text[match.start():begin] + "DECLARED_SHA256" + text[end:match.end()]
    return PATTERN.sub(replace, text)


def credential_keys(text):
    """Unquote property names only; values and all digests remain untouched."""
    return KEY_PATTERN.sub(lambda match: match.group("key"), text)


if __name__ == "__main__":
    content = Path(sys.argv[1]).read_text(errors="replace")
    operation = credential_keys if "--credential-keys" in sys.argv[2:] else normalize
    sys.stdout.write(operation(content))
