"""Owner-local customer/event cluster bootstrap; identifiers never leave Alice."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

NEGATIVE_SUPPORT_BIT = 2


def load_groups(path, expected_sha, validation_ids):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("Owner cluster metadata changed")
    mapping = json.loads(payload)
    if not isinstance(mapping, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in mapping.items()
    ):
        raise ValueError("Invalid owner cluster metadata")
    if not set(validation_ids) <= set(mapping):
        raise ValueError("Owner validation cluster coverage missing")
    return np.array([mapping[k] for k in validation_ids])


def cluster_draw(clusters, y, rng):
    clusters, y = np.asarray(clusters), np.asarray(y)
    if clusters.ndim != 1 or len(clusters) != len(y) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Invalid cluster/bootstrap support")
    unique = np.unique(clusters)
    indices = {k: np.flatnonzero(clusters == k) for k in unique}
    strata = {}
    for k, idx in indices.items():
        pattern = int(np.any(y[idx] == 1)) + NEGATIVE_SUPPORT_BIT * int(
            np.any(y[idx] == 0)
        )
        strata.setdefault(pattern, []).append(k)
    chosen = np.concatenate(
        [rng.choice(group, len(group), replace=True) for group in strata.values()]
    )
    result = np.concatenate([indices[k] for k in chosen])
    rng.shuffle(result)
    return result
