"""Owner-only opaque input, MPC join and secret correctness oracle."""

from __future__ import annotations
import json
from pathlib import Path
import secrets
import shutil
import time
import numpy as np
import yaml
import bounded_psi as bp  # type: ignore[import-not-found]
import secret_join as join  # type: ignore[import-not-found]

PRIVATE_ENTROPY_BITS = 128
PRIVATE_FILE_MODE = 0o600


def owner_input(party, runtime, cfg, seed):
    root = bp.ft.ROOT / party / runtime["run_id"]
    bp.ft.isolation_preflight(party)
    manifest = json.loads((root / "input/manifest.json").read_text())
    if (
        set(manifest) != {"keys.json"}
        or bp.sha(root / "input/keys.json") != manifest["keys.json"]
    ):
        raise ValueError("Frozen owner keys changed")
    keys = json.loads((root / "input/keys.json").read_text())
    join.encode_keys(keys)
    size = cfg["rows_per_owner"]
    if len(keys) < size:
        raise ValueError("Insufficient frozen keys; no adaptive resizing")
    folder = root / "trainings" / f"join_seed{seed}"
    folder.mkdir(exist_ok=False)
    for sub in ("data", "results", "logs"):
        (folder / sub).mkdir()
    shutil.copytree(root / "code", folder / "code")
    ordered = np.random.default_rng(seed).choice(len(keys), size=size, replace=False)
    entropy = secrets.randbits(PRIVATE_ENTROPY_BITS)
    rng = np.random.default_rng(entropy)
    chosen = [
        keys[i] if keep else "padding:" + party + ":" + secrets.token_hex()
        for i, keep in zip(ordered, rng.random(size) < cfg["keep_probability"])
    ]
    rng.shuffle(chosen)
    encoded = join.encode_keys(chosen)
    np.save(folder / "data/owner_keys.npy", encoded)
    bp.save(
        folder / "data/construction.json",
        {"private_entropy": str(entropy), "seed": seed, "input_manifest": manifest},
    )
    return encoded


def invariant(a, b, result):
    """Independent all-pairs equality check inside MPC; disclose only pass bit."""
    import jax.numpy as jnp

    ma, pa, mb, pb = result
    equal = jnp.all(a[:, None, :] == b[None, :, :], axis=-1)
    left = jnp.any(equal, axis=1)
    right = jnp.any(equal, axis=0)
    lp = jnp.where(left, jnp.argmax(equal, axis=1), 0)
    rp = jnp.where(right, jnp.argmax(equal, axis=0), 0)
    return (
        jnp.all(ma == left)
        & jnp.all(mb == right)
        & jnp.all(pa == lp)
        & jnp.all(pb == rp)
        & jnp.all(jnp.sum(equal, axis=1) <= 1)
        & jnp.all(jnp.sum(equal, axis=0) <= 1)
    )


def known_controls():
    import jax.numpy as jnp

    passed = jnp.array(True)
    for left, right in (
        (["a", "b"], ["b", "a"]),
        (["a", "b"], ["x", "z"]),
        (["a", "b", "c"], ["c", "x", "b"]),
        (["😀", "opaque key"], ["opaque key"]),
    ):
        a, b = join.encode_keys(left), join.encode_keys(right)
        result = join.secret_join(jnp.asarray(a), jnp.asarray(b))
        expected = join.reference_join(a, b)
        passed = passed & invariant(jnp.asarray(a), jnp.asarray(b), result)
        for actual, known in zip(result, expected):
            passed = passed & jnp.all(actual == jnp.asarray(known))
    return passed


def run(runtime):
    cfg = yaml.safe_load(Path(runtime["config"]).read_text())
    root = bp.ft.ROOT / runtime["party"] / runtime["run_id"]
    if cfg["output"] != "secret_shares" or len(set(cfg["seeds"])) < len(bp.TRIALS):
        raise ValueError("Unapproved output or seed protocol")
    if bp.sha(runtime["config"]) != runtime["config_sha256"] or any(
        bp.sha(root / "code" / name) != digest
        for name, digest in runtime["code_sha256"].items()
    ):
        raise ValueError("Code/protocol changed")
    seconds = []
    with bp.session(runtime, cfg) as (sf, secure):
        if not bool(sf.reveal(secure(known_controls)())):
            raise ValueError("Synthetic known answers failed")
        for seed in cfg["seeds"]:
            started = time.monotonic()
            a = sf.PYU("alice")(owner_input)("alice", runtime, cfg, seed).to(secure)
            b = sf.PYU("bob")(owner_input)("bob", runtime, cfg, seed).to(secure)
            result = secure(join.secret_join)(a, b)
            if not bool(sf.reveal(secure(invariant)(a, b, result))):
                raise ValueError("Secret correctness oracle failed")
            secure.dump(
                result,
                [
                    str(
                        bp.ft.ROOT
                        / party
                        / runtime["run_id"]
                        / "trainings"
                        / f"join_seed{seed}"
                        / "results/membership.share"
                    )
                    for party in bp.PARTIES
                ],
            )
            sf.wait(result)
            seconds.append(time.monotonic() - started)
    for seed in cfg["seeds"]:
        (root / "trainings" / f"join_seed{seed}" / "results/membership.share").chmod(
            PRIVATE_FILE_MODE
        )
    report = {
        "status": "passed",
        "kind": "generic_spu_secret_sort_join",
        "seeds": cfg["seeds"],
        "known_controls_passed": True,
        "secret_all_pairs_oracle_passed": True,
        "intersection_reconstructed": False,
        "membership_or_count_disclosed": False,
        "peer_indices_disclosed": False,
        "input_rows_padded_public": cfg["rows_per_owner"],
        "federation_anonymous_denied": True,
        "spu_anonymous_denied": True,
        "test_read": False,
        "native_rr22": False,
        "existing_training_consumes_this_psi": False,
    }
    bp.save(root / "results/engineering.json", report)
    bp.save(root / "results/timing.json", {"seconds": seconds})
    return report
