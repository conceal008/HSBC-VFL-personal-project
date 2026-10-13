"""One treatment arm application contract; the pinned secure optimizer is unchanged."""

from __future__ import annotations

import numpy as np
import functional_training as ft

BINARY_COLUMN_FACTOR = 2


def local_design(x, treatment, is_uplift, potential=-1):
    if not is_uplift:
        return x
    arm = treatment if potential < 0 else np.full_like(treatment, potential)
    indicator = (arm[:, None] == 1).astype(x.dtype)
    return np.concatenate((x, indicator, x * indicator), axis=1)


def chunk_design(
    xa, xb, groups, table, treatment, route, is_uplift, potential, padding
):
    import jax.numpy as jnp

    if route != "L3_secure" or groups is not None or table is not None:
        raise ValueError("Portfolio only permits declared per-record secure design")
    x = jnp.concatenate((xa, xb), axis=1)
    if is_uplift:
        arm = treatment if potential < 0 else jnp.full_like(treatment, potential)
        indicator = (arm[:, None] == 1).astype(x.dtype)
        x = jnp.concatenate((x, indicator, x * indicator), axis=1)
    x = jnp.concatenate((x, jnp.ones((len(x), 1), dtype=x.dtype)), axis=1)
    return jnp.pad(x, ((0, padding), (0, 0)))


def mapped_indices(a, b, uplift):
    width = a + b
    indices = list(range(a))
    if uplift:
        indices += [width] + list(range(width + 1, width + 1 + a))
    indices += [BINARY_COLUMN_FACTOR * width + 1 if uplift else width]
    return indices, (
        BINARY_COLUMN_FACTOR * width + BINARY_COLUMN_FACTOR if uplift else width + 1
    )


def initial_weights(a, b, extra, uplift, seed, scale, include_bob):
    old_width, new_width = a + b, a + b + extra
    dimension = (
        BINARY_COLUMN_FACTOR * old_width + BINARY_COLUMN_FACTOR
        if uplift
        else old_width + 1
    )
    target_dimension = (
        BINARY_COLUMN_FACTOR * new_width + BINARY_COLUMN_FACTOR
        if uplift
        else new_width + 1
    )
    original = ft.initial_weights(dimension, seed, scale)
    target = np.zeros((target_dimension, 1), dtype=np.float32)
    old_main = list(range(old_width))
    new_main = list(range(a)) + list(range(a + extra, new_width))
    target[new_main] = original[old_main]
    if uplift:
        target[new_width] = original[old_width]
        target[np.array(new_main) + new_width + 1] = original[
            np.array(old_main) + old_width + 1
        ]
    target[-1] = original[-1]
    if include_bob:
        return target
    indices, _ = mapped_indices(a + extra, b, uplift)
    return target[indices]
