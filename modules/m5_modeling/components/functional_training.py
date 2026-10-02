"""Two-party CHEETAH training; plaintext data stays inside its owner's VM.

The graph reuses SecretFlow's SS-SGD batch update; it does not implement crypto.
Only declared booleans, dimensions, fixed schedule and status reach both drivers.
Known synthetic precision probes are disclosed only in synthetic smoke mode.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from functional_metrics import (bootstrap_metrics, membership_diagnostic, save_json,  # type: ignore[import-not-found]
                                select_candidate, validation_score)
from functional_tls import install_pinned_tls_adapter, reject_anonymous_client, reject_anonymous_spu  # type: ignore[import-not-found]

NOTEBOOK_VERSION = 4
SPLITS = ("train", "validation", "test")
HASH_WORDS = 8
CERT_VERIFY_DEPTH = 1
TLS_NEGATIVE_TIMEOUT = 3
SSH_PORT = 22
FED_PORT = 50051
SPU_PORT = 50052
ROOT = Path("/srv/vfl")


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_frames(frames, party, spec):
    columns = list(frames["train"].columns)
    if any(list(frame.columns) != columns for frame in frames.values()):
        raise ValueError("Frozen schemas differ")
    outcomes = {"record_id", "label", "treatment"}
    features = [col for col in columns if col not in outcomes]
    if party == "bob" and ({"label", "treatment"} & set(columns)):
        raise ValueError("Bob contains forbidden outcomes")
    if party == "alice" and "label" not in columns:
        raise ValueError("Alice label missing")
    if not features:
        raise ValueError("No feature columns")
    expected = spec.get(f"{party}_features")
    if expected is not None and len(features) != expected:
        raise ValueError("Prepared feature dimension differs from frozen configuration")
    sets = []
    for frame in frames.values():
        if not frame["record_id"].is_unique or frame["record_id"].isna().any():
            raise ValueError("Invalid keys")
        sets.append(set(frame["record_id"]))
        if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite features")
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i + 1, len(sets))):
        raise ValueError("Splits overlap")
    for feature in features:
        if any(feature.endswith(f"__{name}") or feature.startswith(f"categorical__{name}_")
               for name in spec["prohibited_features"]):
            raise ValueError("Prohibited source feature")
    return features


def isolation_preflight(party):
    checks = {"nonroot": os.geteuid() != 0,
              "no_sudo": subprocess.run(["sudo", "-n", "true"], capture_output=True, check=False).returncode != 0,
              "no_host_mount": "virtiofs" not in Path("/proc/mounts").read_text(),
              "expected_account": subprocess.check_output(["id", "-un"], text=True).strip() == party}
    sentinel = Path("/srv/forbidden/peer_sentinel")
    try:
        sentinel.read_bytes()
    except OSError as error:
        checks["permission_denied_not_missing"] = error.errno in (errno.EACCES, errno.EPERM)
    else:
        checks["permission_denied_not_missing"] = False
    peer = "bob" if party == "alice" else "alice"
    try:
        with socket.create_connection((f"lima-hsbc-{peer}.internal", SSH_PORT), timeout=TLS_NEGATIVE_TIMEOUT):
            checks["peer_ssh_egress_denied"] = False
    except OSError as error:
        checks["peer_ssh_egress_denied"] = error.errno in (errno.EACCES, errno.EPERM, errno.ECONNREFUSED)
    if not all(checks.values()):
        raise RuntimeError("Participant isolation preflight failed")
    return checks


def prepare_local(party, run_id, dataset, cfg):
    checks = isolation_preflight(party)
    root = ROOT / party / run_id
    manifest = json.loads((root / "input/manifest.json").read_text())
    frames = {}
    for split in SPLITS:
        source = root / "input" / f"{split}.csv"
        if fingerprint(source) != manifest[f"{split}.csv"]:
            raise ValueError("Frozen input fingerprint mismatch")
        try:
            source.open("r+b").close()
        except PermissionError:
            pass
        else:
            raise RuntimeError("Input must be read-only")
        frames[split] = pd.read_csv(source)
    spec, shared = cfg["datasets"][dataset], cfg["shared"]
    features = validate_frames(frames, party, spec)
    scaler = StandardScaler().fit(frames["train"][features].to_numpy(dtype=float))
    quantiles = np.linspace(0, 1, shared["segments"] + 1)[1:-1]
    edges = np.quantile(frames["train"][features[0]], quantiles)
    data = {"party": party, "root": str(root), "features": features, "manifest": manifest,
            "spec": spec, "shared": shared}
    (root / "data").mkdir(exist_ok=True)
    for split, frame in frames.items():
        x = np.clip(scaler.transform(frame[features].to_numpy(dtype=float)),
                    -shared["clip_features"], shared["clip_features"]).astype(np.float32)
        ids = np.stack([np.frombuffer(hashlib.sha256(str(key).encode()).digest(), dtype="<u4")
                        for key in frame["record_id"]])
        assert ids.shape[1] == HASH_WORDS
        values = {"x": x, "ids": ids,
                  "groups": np.searchsorted(edges, frame[features[0]], side="right").astype(np.int32)}
        if party == "alice":
            values["y"] = frame["label"].to_numpy(dtype=np.float32)
            values["t"] = frame["treatment"].to_numpy(dtype=np.int32) if spec["treatment"] else np.zeros(len(frame), dtype=np.int32)
        data[split] = values
        np.savez_compressed(root / "data" / f"{split}_local.npz", **values)
    save_json(root / "data/preprocessing.json", {"features": features, "mean": scaler.mean_.tolist(),
               "scale": scaler.scale_.tolist(), "group_edges": edges.tolist(), "fit_split": "train"})
    save_json(root / "logs/isolation.json", checks)
    return data


def metadata(data):
    return {split: list(data[split]["x"].shape) for split in SPLITS}


def get_array(data, split, key, seed=None):
    value = data[split][key]
    if seed is not None:
        value = value[np.random.default_rng(seed).permutation(len(value))]
    return value


def get_chunk(data, split, key, begin, end, seed=None):
    return get_array(data, split, key, seed)[begin:end]


def secure_equal(a, b):
    import jax.numpy as jnp
    return jnp.all(a == b)


def group_table(groups, x, segments, minimum_group):
    return finalize_groups(group_partials(groups, x, segments), minimum_group)


def group_partials(groups, x, segments):
    import jax.numpy as jnp
    membership = (groups[:, None] == jnp.arange(segments)[None, :]).astype(x.dtype)
    count = jnp.sum(membership, axis=0)
    sums = membership.T @ x
    return count, sums


def add_group_partials(first, second):
    return first[0] + second[0], first[1] + second[1]


def finalize_groups(parts, minimum_group):
    import jax.numpy as jnp
    count, sums = parts
    mean = sums / jnp.maximum(count[:, None], 1)
    return jnp.where(count[:, None] >= minimum_group, mean, jnp.zeros_like(mean))


def build_design(xa, xb, groups, table, treatment, route, is_uplift, potential):
    import jax.numpy as jnp
    # A batched oblivious join is equivalent to secret-index gather but avoids
    # one OLE matrix multiplication per row in the pinned CHEETAH gather kernel.
    if route == "L1_secure":
        membership = (groups[:, None] == jnp.arange(table.shape[0])[None, :]).astype(xa.dtype)
        passive = membership @ table
    else:
        passive = xb
    x = jnp.concatenate([xa, passive], axis=1)
    if is_uplift:
        if potential >= 0:
            treatment = jnp.full_like(treatment, potential)
        indicator = (treatment[:, None] == jnp.array([1, 2])[None, :]).astype(x.dtype)
        interactions = (x[:, :, None] * indicator[:, None, :]).reshape((len(x), -1))
        x = jnp.concatenate([x, indicator, interactions], axis=1)
    return jnp.concatenate([x, jnp.ones((len(x), 1), dtype=x.dtype)], axis=1)


def initial_weights(n, seed, scale):
    return np.random.default_rng(seed).normal(0, scale, size=(n, 1)).astype(np.float32)


def chunk_design(xa, xb, groups, table, treatment, route, is_uplift, potential, padding):
    import jax.numpy as jnp
    design = build_design(xa, xb, groups, table, treatment, route, is_uplift, potential)
    # Public zero rows have no bias or feature contribution; include every real row.
    return jnp.pad(design, ((0, padding), (0, 0)))


def pad_y(y, padding):
    import jax.numpy as jnp
    return jnp.pad(y, (0, padding))


def secure_predict(x, weights, sig_type):
    from secretflow.utils.sigmoid import SigType, sigmoid
    return sigmoid((x @ weights).reshape(-1), SigType(sig_type))


def project_weights(weights, bound):
    """Project inside MPC; never release a coordinate or a clipping count."""
    import jax.numpy as jnp
    return jnp.clip(weights, -bound, bound)


def checked_predict(x, weights, sig_type):
    import jax.numpy as jnp
    probability = secure_predict(x, weights, sig_type)
    # SPU fixed-point values have no IEEE NaN/Inf, and is_finite has no lowering
    # in the pinned compiler. Check range in MPC; Alice checks finite before disk.
    valid = jnp.all((probability >= 0) & (probability <= 1))
    return probability, valid


def synthetic_precision_probe(logits):
    from secretflow.utils.sigmoid import SigType, sigmoid
    return {'sr': sigmoid(logits, SigType.SR), 'df': sigmoid(logits, SigType.DF)}


def configured_probe_inputs(logits):
    return np.asarray(logits, dtype=np.float32)


def new_training(party, run_id, name, cfg):
    root = ROOT / party / run_id
    target = root / "trainings" / name
    target.mkdir(parents=True, exist_ok=False)
    for subdir in ("code", "data", "results", "logs"):
        (target / subdir).mkdir()
    for source in (root / "code").iterdir():
        if source.is_file():
            shutil.copyfile(source, target / "code" / source.name)
    environment = Path("/opt/secretflow/environment.lock")
    if environment.exists():
        shutil.copyfile(environment, target / "code/environment.lock")
    save_json(target / "code/training_config.json", cfg)
    save_json(target / "data/input_references.json", {
        "directory": str(root / "data"), "frozen_input": json.loads((root / "input/manifest.json").read_text()),
        "derived_input": {p.name: fingerprint(p) for p in (root / "data").iterdir() if p.is_file()}})
    (target / "logs/实验日志.md").write_text(
        f"# 训练日志：{name}\n\n本轮目录：{run_id}。本次代码/配置见 code/，新增数据引用见 data/。"
        "参数选择只用验证集；test 不进入选择函数。对方数据不在本虚拟机。\n")
    return True


def local_design(x, t, is_uplift, potential=-1):
    if not is_uplift:
        return x
    if potential >= 0:
        t = np.full_like(t, potential)
    indicator = (t[:, None] == np.array([1, 2])[None, :]).astype(x.dtype)
    return np.c_[x, indicator, (x[:, :, None] * indicator[:, None, :]).reshape(len(x), -1)]


def fit_local_baselines(data, cfg):
    shared, spec = cfg["shared"], data["spec"]
    records = []
    candidates = [("L0_LR", c) for c in shared["l0_c_grid"]] + [("L0_tree", None)]
    for seed in shared["model_seeds"]:
        for index, (family, regularization) in enumerate(candidates):
            name = f"{family}_all_arms_seed{seed}_candidate{index}"
            new_training("alice", Path(data["root"]).name, name, {"shared": shared, "family": family, "C": regularization, "seed": seed})
            tr = data["train"]
            model = (LogisticRegression(C=regularization, max_iter=shared["l0_max_iter"], random_state=seed)
                     if family == "L0_LR" else HistGradientBoostingClassifier(
                         max_iter=shared["tree_iterations"], max_leaf_nodes=shared["tree_leaves"],
                         l2_regularization=shared["tree_l2"], random_state=seed))
            model.fit(local_design(tr["x"], tr["t"], spec["treatment"]), tr["y"])
            predictions = {}
            for split in SPLITS:
                row = data[split]
                potentials = [0] + spec.get("treatment_arms", []) if spec["treatment"] else [-1]
                predictions[split] = np.stack([model.predict_proba(local_design(
                    row["x"], row["t"], spec["treatment"], arm))[:, 1] for arm in potentials], axis=1)
            store_predictions(data, name, predictions["train"], predictions["validation"], predictions["test"])
            records.append({"name": name, "route": "L0", "seed": seed})
    return records


def store_predictions(data, name, train, validation, test):
    folder = Path(data["root"]) / "trainings" / name / "results"
    arrays = [np.asarray(p) for p in (train, validation, test)]
    if any(not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) for p in arrays):
        raise ValueError("Invalid probabilities before output storage")
    epsilon = data["shared"]["score_epsilon"]
    boundary = float(np.mean((arrays[0] <= epsilon) | (arrays[0] >= 1 - epsilon)))
    is_secure = name.startswith(("L1_", "L3_"))
    limit = data["shared"].get("training_boundary_limit", 1.0)
    save_json(folder / "numerical_checks.json", {"probabilities_valid": True,
        "training_boundary_fraction": boundary, "boundary_limit": limit,
        "criterion": "training only; predeclared engineering validity, not model selection"})
    if is_secure and boundary > limit:
        raise ValueError("Secure training scores exceed predeclared boundary limit")
    for split, predictions in zip(SPLITS, (train, validation, test)):
        np.save(folder / f"{split}_predictions.npy", np.asarray(predictions))
    v = data["validation"]
    score = validation_score(v["y"], np.asarray(validation), data["spec"], data["shared"],
                             v["t"] if data["spec"]["treatment"] else None)
    save_json(folder / "validation_score.json", {"score": score, "selection_split": "validation"})
    save_json(folder / "status.json", {"training": "passed", "receiver": "alice"})
    return True


def finalize_alice(data, records):
    root, shared, spec = Path(data["root"]), data["shared"], data["spec"]
    report = {"status": "completed", "selection": "validation only; strongest L0 selected within family grid",
              "round": root.name, "metrics": [], "paired_differences": [], "output_attack_diagnostics": []}
    for seed in shared["model_seeds"]:
        selected = {}
        for route in ("L0", "L1_secure", "L3_secure"):
            options = [r for r in records if r["seed"] == seed and r["route"] == route]
            validation_scores = [json.loads((root / "trainings" / r["name"] / "results/validation_score.json").read_text())["score"] for r in options]
            chosen = options[select_candidate(validation_scores)]
            folder = root / "trainings" / chosen["name"] / "results"
            selected[route] = np.load(folder / "test_predictions.npy")
            test, train = data["test"], data["train"]
            treatment = test["t"] if spec["treatment"] else None
            metric = bootstrap_metrics(test["y"], selected[route], spec, shared,
                                       shared["bootstrap_seed"] + seed, treatment)
            entry = {"seed": seed, "route": route, "selected_training": chosen["name"], "metrics": metric}
            report["metrics"].append(entry)
            save_json(folder / "selected_test_evaluation.json", entry)
            train_pred = np.load(folder / "train_predictions.npy")
            train_prob = train_pred[np.arange(len(train["y"])), train["t"] if spec["treatment"] else np.zeros(len(train["y"]), dtype=int)]
            test_prob = selected[route][np.arange(len(test["y"])), test["t"] if spec["treatment"] else np.zeros(len(test["y"]), dtype=int)]
            diagnostic = membership_diagnostic(train["y"], train_prob, test["y"], test_prob, shared, shared["bootstrap_seed"] + seed)
            report["output_attack_diagnostics"].append({"seed": seed, "route": route, **diagnostic})
        for comparator in ("L0", "L1_secure"):
            difference = bootstrap_metrics(test["y"], selected["L3_secure"], spec, shared,
                                           shared["bootstrap_seed"] + seed, treatment, selected[comparator])
            report["paired_differences"].append({"seed": seed, "comparison": f"L3_secure_minus_{comparator}", "metrics": difference})
    save_json(root / "results/private_evaluation.json", report)
    (root / "results/结果分析.md").write_text(
        "# 本轮私有训练结果\n\n指标、逐种子区间和配对差值见 private_evaluation.json。"
        "使用同一冻结测试切片，不代表五个人群；字段归属和预对齐是构造条件。"
        "L1 聚合表只在 MPC 内使用，其成本不代表原明文低成本路线。"
        "成员推断诊断是假设释放分数与标签后的风险评估；分数实际仅 Alice 可见。"
        "本轮不证明所有攻击安全、物理隔离或银行业务效果。\n")
    return {"status": "completed", "private_artifact": str(root / "results/private_evaluation.json")}



def verify_local_outputs(data, records):
    """Fail closed on modified frozen input or forbidden artifacts in Bob's VM."""
    root, party = Path(data["root"]), data["party"]
    for filename, digest in data["manifest"].items():
        if fingerprint(root / "input" / filename) != digest:
            raise ValueError("Frozen input changed during training")
    if party == "bob":
        if any({"y", "t"} & set(data[split]) for split in SPLITS):
            raise ValueError("Bob obtained outcomes")
        if list(root.rglob("*predictions.npy")) or (root / "results/private_evaluation.json").exists():
            raise ValueError("Bob obtained released model outputs")
    save_json(root / "logs/output_boundary.json", {"frozen_input_unchanged": True,
              "own_account": party, "bob_plaintext_output_check": "passed" if party == "bob" else "not_recipient",
              "joint_weights": "secret_shares_only", "completed_schedule": len(records)})
    return True


def train_round(runtime):
    import secretflow as sf
    import spu
    from secretflow.device import SPUCompilerNumReturnsPolicy
    from secretflow.ml.linear.linear_model import RegType
    from secretflow.ml.linear.ss_sgd.model import Penalty, Strategy, _batch_update_w
    from secretflow.utils.sigmoid import SigType

    cfg = yaml.safe_load(Path(runtime["config"]).read_text())
    shared, spec = cfg["shared"], cfg["datasets"][runtime["dataset"]]
    if shared["infeed_rows"] % shared["batch_size"]:
        raise ValueError("Infeed rows must be a batch-size multiple")
    party, run_id = runtime["party"], runtime["run_id"]
    cert_root = ROOT / party / "tls"
    tls = {"cert": str(cert_root / "cert.pem"), "key": str(cert_root / "key.pem"), "ca_cert": str(cert_root / "ca.pem")}
    install_pinned_tls_adapter()
    sf.init(ray_mode=False, cluster_config={"self_party": party, "parties": runtime["parties"]},
            tls_config=tls, cross_silo_comm_backend="grpc", logging_level="warning",
            cross_silo_comm_options={"timeout_in_ms": shared["link_timeout_ms"]},
            enable_waiting_for_other_parties_ready=True, job_name=run_id)
    passed = False
    try:
        if not reject_anonymous_client(runtime["parties"][party]["address"], tls["ca_cert"], TLS_NEGATIVE_TIMEOUT):
            raise RuntimeError("Federation accepted a client without certificate")
        alice, bob = sf.PYU("alice"), sf.PYU("bob")
        nodes = []
        for name in ("alice", "bob"):
            local_tls = ROOT / name / "tls"
            opts = {"certificate_path": str(local_tls / "cert.pem"), "private_key_path": str(local_tls / "key.pem"),
                    "ca_file_path": str(local_tls / "ca.pem"), "verify_depth": CERT_VERIFY_DEPTH}
            nodes.append({"party": name, "address": runtime["spu_addresses"][name],
                          "listen_address": f"0.0.0.0:{SPU_PORT}",
                          "tls_opts": {"server_ssl_opts": opts, "client_ssl_opts": opts}})
        log_options = spu.logging.LogOptions()
        log_options.system_log_path = "logs/spu.log"
        # Protocol traces can contain shares/intermediates. Never export or log them.
        log_options.trace_log_path = ""
        log_options.trace_content_length = 0
        log_options.enable_console_logger = False
        secure = sf.SPU({"nodes": nodes, "runtime_config": {"protocol": cfg["protocol"], "field": cfg["field"]}},
                        log_options=log_options,
                        link_desc={"recv_timeout_ms": shared["link_timeout_ms"], "http_timeout_ms": shared["link_timeout_ms"]})
        # SPU actors initialize asynchronously; await a data-free readiness task.
        sf.wait(secure(lambda: np.array(True))())
        if not reject_anonymous_spu(runtime["spu_addresses"][party], tls["ca_cert"], TLS_NEGATIVE_TIMEOUT):
            raise RuntimeError("SPU accepted a client without certificate")
        save_json(ROOT / party / run_id / "logs/tls_checks.json", {"federation_anonymous_denied": True,
                  "spu_anonymous_denied": True, "protocol_trace_disabled": True})
        if runtime.get("is_smoke"):
            # Known synthetic grid only. Sharing forces SECRET visibility so the
            # probe exercises MPC approximations, not public constant folding.
            grid = cfg['smoke']['precision_logits']
            result = sf.reveal(secure(synthetic_precision_probe)(alice(configured_probe_inputs)(grid).to(secure)))
            save_json(ROOT / party / run_id / 'logs/synthetic_precision_grid.json',
                      {'known_synthetic_logits': grid, 'results': {k: np.asarray(v).tolist() for k, v in result.items()}})
        da = alice(prepare_local)("alice", run_id, runtime["dataset"], cfg)
        db = bob(prepare_local)("bob", run_id, runtime["dataset"], cfg)
        shape_a, shape_b = sf.reveal([alice(metadata)(da), bob(metadata)(db)])
        for split in SPLITS:
            if shape_a[split][0] != shape_b[split][0]:
                raise ValueError("Participant row counts differ")
            aligned = secure(secure_equal)(alice(get_array)(da, split, "ids").to(secure),
                                            bob(get_array)(db, split, "ids").to(secure))
            if not bool(sf.reveal(aligned)):
                raise ValueError("Secure pre-alignment consistency failed")
        records = sf.reveal(alice(fit_local_baselines)(da, cfg))
        # records are only fixed schedule names/seeds, never scores or selected candidates.
        table_state = None
        rows = shape_a["train"][0]
        for begin in range(0, rows, shared["infeed_rows"]):
            end = min(rows, begin + shared["infeed_rows"])
            part = secure(group_partials, static_argnames="segments")(
                alice(get_chunk)(da, "train", "groups", begin, end).to(secure),
                bob(get_chunk)(db, "train", "x", begin, end).to(secure), segments=shared["segments"])
            table_state = part if table_state is None else secure(add_group_partials)(table_state, part)
            sf.wait(table_state)
        table = secure(finalize_groups, static_argnames="minimum_group")(
            table_state, minimum_group=shared["minimum_group"])
        for route in ("L1_secure", "L3_secure"):
            for seed in shared["model_seeds"]:
                designs, label_chunks, train_sizes = {}, [], []
                # Keep reusable small encrypted blocks, never construct a monolithic MPC design.
                for split in SPLITS:
                    total = shape_a[split][0]
                    block_size = shared["infeed_rows"] if split == "train" else shared["prediction_batch_size"]
                    order_seed = seed if split == "train" else None
                    potentials = [-1] if split == "train" else ([0] + spec.get("treatment_arms", []) if spec["treatment"] else [-1])
                    designs[split] = [[] for _ in potentials]
                    for begin in range(0, total, block_size):
                        end = min(total, begin + block_size)
                        padding = (-(end - begin)) % shared["batch_size"] if split == "train" else 0
                        a = alice(get_chunk)(da, split, "x", begin, end, order_seed).to(secure)
                        b = bob(get_chunk)(db, split, "x", begin, end, order_seed).to(secure)
                        groups = alice(get_chunk)(da, split, "groups", begin, end, order_seed).to(secure)
                        treatment = alice(get_chunk)(da, split, "t", begin, end, order_seed).to(secure)
                        for idx, potential in enumerate(potentials):
                            design = secure(chunk_design, static_argnames=("route", "is_uplift", "potential", "padding"))(
                                a, b, groups, table, treatment, route=route, is_uplift=spec["treatment"], potential=potential, padding=padding)
                            sf.wait(design)
                            designs[split][idx].append(design)
                        if split == "train":
                            y = alice(get_chunk)(da, split, "y", begin, end, seed).to(secure)
                            label_chunks.append(secure(pad_y, static_argnames="padding")(y, padding=padding))
                            train_sizes.append(end - begin + padding)
                columns = shape_a["train"][1] + shape_b["train"][1]
                if spec["treatment"]:
                    columns += len(spec["treatment_arms"]) + columns * len(spec["treatment_arms"])
                for candidate, lr in enumerate(shared["learning_rates"]):
                    name = f"{route}_all_arms_seed{seed}_candidate{candidate}"
                    detail = {"route": route, "seed": seed, "candidate": candidate, "learning_rate": lr, "shared": shared,
                              "arithmetic_field": cfg["field"], "last_batch": "zero padding; all real rows included"}
                    sf.wait([alice(new_training)("alice", run_id, name, detail), bob(new_training)("bob", run_id, name, detail)])
                    weights = secure(initial_weights, static_argnames=("n", "seed", "scale"))(
                        n=columns + 1, seed=seed, scale=shared["initialization_std"])
                    started = time.monotonic()
                    for epoch in range(shared["epochs"]):
                        for design, y, chunk_rows in zip(designs["train"][0], label_chunks, train_sizes):
                            weights, _ = secure(_batch_update_w,
                                static_argnames=("sig_type", "reg_type", "penalty", "total_batch", "batch_size", "strategy", "enable_spu_cache"),
                                num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER, user_specified_num_returns=2)(
                                    design, y, weights, lr, shared["l2_norm"], sig_type=SigType(shared["sigmoid"]),
                                    reg_type=RegType.Logistic, penalty=Penalty.L2, total_batch=chunk_rows // shared["batch_size"],
                                    batch_size=shared["batch_size"], strategy=Strategy.NAIVE_SGD, dk_arr=None, enable_spu_cache=False)
                            sf.wait(weights)
                            weights = secure(project_weights, static_argnames="bound")(
                                weights, bound=shared["weight_bound"])
                            sf.wait(weights)
                        save_json(ROOT / party / run_id / "trainings" / name / "logs/progress.json",
                                  {"completed_epochs": epoch + 1, "planned_epochs": shared["epochs"]})
                    secure.dump(weights, [str(ROOT / p / run_id / "trainings" / name / "results/model.share") for p in ("alice", "bob")])
                    predictions = {}
                    for split in SPLITS:
                        outputs = []
                        for chunks in designs[split]:
                            column = []
                            for design in chunks:
                                probability, valid = secure(checked_predict, static_argnames="sig_type",
                                    num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
                                    user_specified_num_returns=2)(design, weights, sig_type=shared["sigmoid"])
                                if not bool(sf.reveal(valid)):
                                    raise ValueError("Secure probability validity check failed")
                                column.append(probability.to(alice))
                            outputs.append(column)
                        predictions[split] = alice(pack_chunked_predictions)(outputs, shape_a[split][0],
                            seed if split == "train" else None, spec["treatment"], da, split)
                    sf.wait(alice(store_predictions)(da, name, predictions["train"], predictions["validation"], predictions["test"]))
                    records.append({"name": name, "route": route, "seed": seed})
                    print(json.dumps({"training": name, "status": "completed", "seconds": time.monotonic() - started}), flush=True)
        sf.wait([alice(verify_local_outputs)(da, records), bob(verify_local_outputs)(db, records)])
        result = sf.reveal(alice(finalize_alice)(da, records))
        save_json(ROOT / party / run_id / "results/status.json", result if party == "alice" else {"status": "completed", "score_access": "not_granted"})
        passed = True
    except Exception:
        import traceback
        (ROOT / party / run_id / "logs/failure_traceback.log").write_text(traceback.format_exc())
        raise
    finally:
        sf.shutdown(barrier_on_shutdown=passed, on_error=not passed)


def pack_chunked_predictions(outputs, rows, order_seed, is_uplift, data, split):
    columns = [np.concatenate([np.asarray(chunk).reshape(-1) for chunk in chunks])[:rows] for chunks in outputs]
    return pack_predictions(columns, order_seed, is_uplift, data, split)


def pack_predictions(outputs, order_seed, is_uplift, data, split):
    predictions = np.stack([np.asarray(item).reshape(-1) for item in outputs], axis=1)
    if order_seed is not None:
        inverse = np.argsort(np.random.default_rng(order_seed).permutation(len(predictions)))
        predictions = predictions[inverse]
        if is_uplift:
            # Training design is factual: retain only the observed probability in its arm slot.
            treatment = data[split]["t"]
            factual = np.repeat(predictions, 1 + int(treatment.max()), axis=1)
            predictions = factual
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite secure predictions")
    return predictions


def execute_notebook(runtime_path):
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    runtime = json.loads(Path(runtime_path).read_text())
    root = ROOT / runtime["party"] / runtime["run_id"]
    notebook = nbformat.read(root / "code" / runtime.get("notebook", "S5.P2_local_functional.ipynb"), as_version=NOTEBOOK_VERSION)
    shell = InteractiveShell.instance()
    shell.user_ns["RUNTIME_PATH"] = str(runtime_path)
    failure = None
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        class DurableOutput:
            def __init__(self, capture, durable):
                self.capture, self.durable = capture, durable
            def write(self, value):
                self.durable.write(value)
                self.durable.flush()
                return self.capture.write(value)
            def flush(self):
                self.durable.flush()
                self.capture.flush()
        # Native federation may fail via os._exit; retain diagnostics before a cell completes.
        with (root / "logs" / f"cell{count}.stdout.log").open("w") as out, (root / "logs" / f"cell{count}.stderr.log").open("w") as err:
            with capture_output() as captured:
                sys.stdout = DurableOutput(sys.stdout, out)
                sys.stderr = DurableOutput(sys.stderr, err)
                result = shell.run_cell(cell.source, store_history=False)
        cell.execution_count = count
        cell.outputs = []
        for channel in ("stdout", "stderr"):
            if getattr(captured, channel):
                cell.outputs.append(nbformat.v4.new_output("stream", name=channel, text=getattr(captured, channel)))
        for rich in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output("display_data", data=rich.data, metadata=rich.metadata))
        failure = result.error_before_exec or result.error_in_exec
        if failure:
            cell.outputs.append(nbformat.v4.new_output("error", ename=type(failure).__name__, evalue=str(failure), traceback=[]))
            break
    nbformat.write(notebook, root / "results/party.executed.ipynb")
    if failure:
        raise RuntimeError("Private Notebook failed; inspect own logs") from failure
    save_json(root / "results/NOTEBOOK_COMPLETED.json", {"status": "passed", "code_cells": count})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=Path)
    args = parser.parse_args()
    train_round(json.loads(args.runtime.read_text()))
