"""Opaque SHA-256 keys joined with SPU sorting; outputs stay secret-shared."""

from __future__ import annotations

import hashlib
import numpy as np

HASH_WORDS = 8
WORD_BYTES = 4
HASH_BYTES = HASH_WORDS * WORD_BYTES


def encode_keys(keys):
    """Owner-only validation; no plaintext cross-party dictionary or lookup."""
    keys = list(keys)
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("Nonempty opaque string keys required")
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate owner keys")
    digests = [hashlib.sha256(key.encode("utf-8")).digest() for key in keys]
    if len(set(digests)) != len(digests):
        raise ValueError("Owner digest collision")
    return np.stack([np.frombuffer(value, dtype="<u4") for value in digests])


def secret_join(a, b):
    """Return original-order membership and peer row indices, all in MPC.

    The caller must supply both inputs with SECRET visibility and must not
    reveal this return value. Zero peer indices for unmatched rows are dummy
    values and may only be consumed alongside the corresponding secret mask.
    """
    import jax.numpy as jnp
    from jax import lax

    return join_kernel(a, b, jnp, lax.sort)


def join_kernel(a, b, array_api, sort):
    """Same expression graph with an injectable backend for synthetic tests."""
    jnp = array_api
    na, nb = a.shape[0], b.shape[0]
    keys = jnp.concatenate([a, b], axis=0)
    original = jnp.arange(na + nb, dtype=jnp.int32)
    operands = tuple(keys[:, column] for column in range(HASH_WORDS)) + (original,)
    sorted_fields = sort(operands, dimension=0, num_keys=HASH_WORDS)
    sorted_keys = jnp.stack(sorted_fields[:-1], axis=1)
    index = sorted_fields[-1]
    owners = index < na
    adjacent = jnp.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)
    adjacent = adjacent & (owners[1:] != owners[:-1])
    right = jnp.concatenate([adjacent, jnp.array([False])])
    left = jnp.concatenate([jnp.array([False]), adjacent])
    member = left | right
    next_index = jnp.concatenate([index[1:], jnp.array([0], dtype=jnp.int32)])
    previous_index = jnp.concatenate([jnp.array([0], dtype=jnp.int32), index[:-1]])
    peer = jnp.where(right, next_index, previous_index)
    peer = jnp.where(member, peer - jnp.where(owners, na, 0), 0)
    _, member, peer = sort((index, member, peer), dimension=0, num_keys=1)
    return member[:na], peer[:na], member[na:], peer[na:]


def reference_join(a, b):
    """Synthetic-only oracle. Never call this on both parties' real inputs."""
    left = {tuple(row): index for index, row in enumerate(a)}
    right = {tuple(row): index for index, row in enumerate(b)}
    if len(left) != len(a) or len(right) != len(b):
        raise ValueError("Duplicate oracle keys")
    ma = np.array([tuple(row) in right for row in a])
    mb = np.array([tuple(row) in left for row in b])
    pa = np.array([right.get(tuple(row), 0) for row in a], dtype=np.int32)
    pb = np.array([left.get(tuple(row), 0) for row in b], dtype=np.int32)
    return ma, pa, mb, pb
