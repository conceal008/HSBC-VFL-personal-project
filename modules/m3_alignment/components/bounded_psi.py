"""Cryptographic PSI for a public finite universe; only shares leave each owner.

This is membership-vector MPC, not RR22 or arbitrary-identifier PSI. Public
universe metadata and one correctness bit are disclosed; memberships are not.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import secrets
import time
import numpy as np
import yaml
import functional_training as ft  # type: ignore[import-not-found]

PARTIES = ("alice", "bob")
TRIALS = (1, 2, 3, 4, 5)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def domain(namespace, rows):
    if (len(namespace) != 64 and namespace not in ("uci_bank_marketing", "hillstrom_email_marketing")) or rows < 1:  # 魔数豁免: SHA256的十六进制表示固定64字符
        raise ValueError("Invalid public universe")
    return {hashlib.sha256(f"{namespace}:{i}".encode()).hexdigest(): i for i in range(rows)}


def membership(keys, lookup, entropy, probability):
    if len(set(keys)) != len(keys) or any(k not in lookup for k in keys):
        raise ValueError("Duplicate/out-of-domain keys")
    if not 0 < probability < 1:
        raise ValueError("Invalid membership construction")
    chosen = np.random.default_rng(entropy).random(len(keys)) < probability
    vector = np.zeros(len(lookup), dtype=np.int64)
    vector[[lookup[k] for k, keep in zip(keys, chosen) if keep]] = 1
    return vector


def owner_input(party, runtime, cfg, trial):
    root = ft.ROOT / party / runtime["run_id"]
    ft.isolation_preflight(party)
    manifest = json.loads((root / "input/manifest.json").read_text())
    if set(manifest) != {"keys.json"} or sha(root / "input/keys.json") != manifest["keys.json"]:
        raise ValueError("Frozen key input mismatch")
    folder = root / "trainings" / f"psi_trial{trial}"
    folder.mkdir(exist_ok=False)
    for sub in ("code", "data", "results", "logs"):
        (folder / sub).mkdir()
    import shutil
    shutil.copytree(root / "code", folder / "code", dirs_exist_ok=True)
    entropy = secrets.randbits(128)  # 魔数豁免: 本方秘密抽样种子的熵位数，不能公开值
    vector = membership(json.loads((root / "input/keys.json").read_text()),
                        domain(runtime["key_namespace"], runtime["universe_rows"]),
                        entropy, cfg["keep_probability"])
    np.save(folder / "data/own_membership.npy", vector)
    save(folder / "data/construction.json", {"private_entropy": str(entropy),
         "trial": trial, "source_sha256": runtime["source_sha256"], "input_manifest": manifest})
    return vector


def intersection(a, b):
    import jax.numpy as jnp
    mask = a * b
    return jnp.concatenate([mask, jnp.sum(mask).reshape(1)])


def invariant(a, b, output):
    import jax.numpy as jnp
    mask, count = output[:-1], output[-1]
    return (jnp.all((a == 0) | (a == 1)) & jnp.all((b == 0) | (b == 1))
            & jnp.all(mask == ((a + b) == 2)) & (count == jnp.sum(mask))
            & (count <= jnp.sum(a)) & (count <= jnp.sum(b)))


def known_answer():
    import jax.numpy as jnp
    a, b = jnp.array([1, 0, 1, 0]), jnp.array([0, 1, 1, 0])
    return jnp.all(intersection(a, b) == jnp.array([0, 0, 1, 0, 1])) & invariant(a, b, intersection(a, b))


@contextmanager
def session(runtime, cfg):
    import secretflow as sf
    import spu
    party = runtime["party"]
    directory = ft.ROOT / party / "tls"
    tls = {"cert": str(directory / "cert.pem"), "key": str(directory / "key.pem"), "ca_cert": str(directory / "ca.pem")}
    ft.install_pinned_tls_adapter()
    sf.init(ray_mode=False, cluster_config={"self_party": party, "parties": runtime["parties"]},
            tls_config=tls, cross_silo_comm_backend="grpc", logging_level="warning",
            enable_waiting_for_other_parties_ready=True,
            cross_silo_comm_options={"timeout_in_ms": cfg["link_timeout_ms"]}, job_name=runtime["run_id"])
    passed = False
    try:
        if not ft.reject_anonymous_client(runtime["parties"][party]["address"], tls["ca_cert"], ft.TLS_NEGATIVE_TIMEOUT):
            raise ValueError("Anonymous federation accepted")
        nodes = []
        for p in PARTIES:
            d = ft.ROOT / p / "tls"
            opts = {"certificate_path": str(d / "cert.pem"), "private_key_path": str(d / "key.pem"),
                    "ca_file_path": str(d / "ca.pem"), "verify_depth": ft.CERT_VERIFY_DEPTH}
            nodes.append({"party": p, "address": runtime["spu_addresses"][p],
                          "listen_address": f"0.0.0.0:{ft.SPU_PORT}",
                          "tls_opts": {"server_ssl_opts": opts, "client_ssl_opts": opts}})
        log = spu.logging.LogOptions()
        log.system_log_path, log.trace_log_path, log.trace_content_length, log.enable_console_logger = "logs/spu.log", "", 0, False
        secure = sf.SPU({"nodes": nodes, "runtime_config": {"protocol": cfg["protocol"], "field": cfg["field"]}},
                        log_options=log, link_desc={"recv_timeout_ms": cfg["link_timeout_ms"], "http_timeout_ms": cfg["link_timeout_ms"]})
        sf.wait(secure(lambda: np.array(True))())
        if not ft.reject_anonymous_spu(runtime["spu_addresses"][party], tls["ca_cert"], ft.TLS_NEGATIVE_TIMEOUT):
            raise ValueError("Anonymous SPU accepted")
        yield sf, secure
        passed = True
    finally:
        sf.shutdown(barrier_on_shutdown=passed, on_error=not passed)


def run(runtime):
    cfg = yaml.safe_load(Path(runtime["config"]).read_text())
    if cfg["trials"] != list(TRIALS) or cfg["output"] != "secret_shares" or runtime["party"] not in PARTIES:
        raise ValueError("Unapproved protocol/output contract")
    root = ft.ROOT / runtime["party"] / runtime["run_id"]
    if sha(runtime["config"]) != runtime["config_sha256"] or any(sha(root / "code" / n) != h for n, h in runtime["code_sha256"].items()):
        raise ValueError("Configuration/code fingerprint mismatch")
    times = []
    with session(runtime, cfg) as (sf, secure):
        if not bool(sf.reveal(secure(known_answer)())):
            raise ValueError("Known answer failed")
        for trial in TRIALS:
            started = time.monotonic()
            a = sf.PYU("alice")(owner_input)("alice", runtime, cfg, trial).to(secure)
            b = sf.PYU("bob")(owner_input)("bob", runtime, cfg, trial).to(secure)
            result = secure(intersection)(a, b)
            if not bool(sf.reveal(secure(invariant)(a, b, result))):
                raise ValueError("Secret intersection invariant failed")
            secure.dump(result, [str(ft.ROOT / p / runtime["run_id"] / "trainings" / f"psi_trial{trial}" / "results/intersection.share") for p in PARTIES])
            sf.wait(result)
            times.append(time.monotonic() - started)
    report = {"status": "passed", "kind": "public_finite_universe_mpc_psi", "trials": list(TRIALS),
              "intersection_reconstructed": False, "test_read": False, "native_rr22": False,
              "known_answer_passed": True, "secret_invariants_passed": True,
              "federation_anonymous_denied": True, "spu_anonymous_denied": True,
              "existing_training_consumes_this_psi": False}
    save(root / "results/engineering.json", report)
    save(root / "results/private_timing.json", {"seconds": times, "universe_rows": runtime["universe_rows"]})
    return report
