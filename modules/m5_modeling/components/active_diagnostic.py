"""Active-only native SS-SGD/MPC diagnostic; no passive features enter training."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import time
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

import functional_training as ft  # type: ignore[import-not-found]
from functional_metrics import (
    bootstrap_metrics,
    save_json,
    select_candidate,
    validation_score,
)  # type: ignore[import-not-found]
from functional_tls import (
    install_pinned_tls_adapter,
    reject_anonymous_client,
    reject_anonymous_spu,
)  # type: ignore[import-not-found]

PRETEST_SPLITS = ("train", "validation")
PARTIES = ("alice", "bob")
REFERENCE_PREFIX = "native_reference"
ACTIVE_PREFIX = "active_secure"
DF_SCALE = 0.5


def numpy_df(logits, sig_type):
    """Exact native DF expression on Alice's own NumPy arrays only."""
    if getattr(sig_type, "value", sig_type) != "df":
        raise ValueError("CPU reference supports only frozen DF")
    return DF_SCALE * (logits / (1 + np.abs(logits))) + DF_SCALE


def cpu_cache_hint_bridge(native):
    """Retain native update bytecode with NumPy tensors and identity cache hint.

    CPU JAX cannot lower the SPU cache hint; this environment also exits on the
    fused gradient. The native batch/update/L2 math remains the same bytecode.
    Replace this function's bindings only, never mutate the installed SPU/JAX.
    """
    bindings = dict(native.__globals__)
    bindings["jnp"] = np
    bindings["sigmoid"] = numpy_df
    bindings["spu"] = SimpleNamespace(
        experimental=SimpleNamespace(drop_cached_var=lambda value, *deps: value)
    )
    return FunctionType(
        native.__code__,
        bindings,
        native.__name__,
        native.__defaults__,
        native.__closure__,
    )


def mapped_indices(alice_width, bob_width, uplift):
    """Retain exactly Alice's coordinates and bias from the old joint initializer."""
    width = alice_width + bob_width
    indices = list(range(alice_width))
    if uplift:
        indices += [width, width + 1]
        indices += [
            width + 2 + i * 2 + arm for i in range(alice_width) for arm in range(2)
        ]
        width += 2 + width * 2
    return indices + [width], width + 1


def mapped_weights(alice_width, bob_width, uplift, seed, scale):
    indices, total = mapped_indices(alice_width, bob_width, uplift)
    return ft.initial_weights(total, seed, scale)[indices]


def active_design(x, treatment, uplift, potential, padding):
    import jax.numpy as jnp

    if uplift:
        if potential >= 0:
            treatment = jnp.full_like(treatment, potential)
        indicator = (treatment[:, None] == jnp.array([1, 2])[None, :]).astype(x.dtype)
        interaction = (x[:, :, None] * indicator[:, None, :]).reshape((len(x), -1))
        x = jnp.concatenate([x, indicator, interaction], axis=1)
    x = jnp.concatenate([x, jnp.ones((len(x), 1), dtype=x.dtype)], axis=1)
    return jnp.pad(x, ((0, padding), (0, 0)))


def choose_secret_weights(weights, index):
    import jax.numpy as jnp

    stacked = jnp.stack(weights)
    mask = (jnp.arange(len(weights)) == index).astype(stacked.dtype)
    return jnp.sum(stacked * mask[:, None, None], axis=0)


def manifest_check(root):
    manifest = json.loads((root / "input/manifest.json").read_text())
    for filename, digest in manifest.items():
        path = root / "input" / filename
        if ft.fingerprint(path) != digest:
            raise ValueError("Frozen input changed")
        try:
            path.open("r+b").close()
        except PermissionError:
            pass
        else:
            raise ValueError("Frozen input writable")
    return manifest


def load_split(data, split):
    """Test parsing has a separate capability, never passed to fitting/selection."""
    root = Path(data["root"])
    if split == "test" and not data.get("test_capability"):
        raise ValueError("Test capability locked")
    manifest_check(root)
    path = root / "input" / f"{split}.csv"
    frame = pd.read_csv(
        path, usecols=None if data["party"] == "alice" else ["record_id"]
    )
    ids = np.stack(
        [
            np.frombuffer(hashlib.sha256(str(k).encode()).digest(), dtype="<u4")
            for k in frame["record_id"]
        ]
    )
    result = {"ids": ids}
    if data["party"] == "alice":
        if list(frame.columns) != data["columns"]:
            raise ValueError("Frozen schema changed")
        x = data["scaler"].transform(frame[data["features"]].to_numpy(dtype=float))
        result["x"] = np.clip(
            x, -data["shared"]["clip_features"], data["shared"]["clip_features"]
        ).astype(np.float32)
        result["y"] = frame["label"].to_numpy(dtype=np.float32)
        result["t"] = (
            frame["treatment"].to_numpy(dtype=np.int32)
            if data["spec"]["treatment"]
            else np.zeros(len(frame), dtype=np.int32)
        )
        if not np.isfinite(result["x"]).all() or not set(np.unique(result["y"])) <= {
            0,
            1,
        }:
            raise ValueError("Invalid own training data")
        if not data.get("is_smoke"):
            origin = Path(data["origin_root"]) / "data" / f"{split}_local.npz"
            with np.load(origin) as frozen:
                if any(not np.array_equal(result[k], frozen[k]) for k in result):
                    raise ValueError("Active preprocessing differs from frozen origin")
    if split == "test":
        used = set(map(tuple, np.r_[data["train"]["ids"], data["validation"]["ids"]]))
        if used & set(map(tuple, ids)):
            raise ValueError("Test overlaps fitting split")
    np.savez_compressed(root / "data" / f"{split}_local.npz", **result)
    return result


def prepare_active(party, runtime, cfg):
    root = ft.ROOT / party / runtime["run_id"]
    checks = ft.isolation_preflight(party)
    manifest = manifest_check(root)
    shared, spec = cfg["shared"], cfg["datasets"][runtime["dataset"]]
    data = {
        "party": party,
        "root": str(root),
        "origin_root": str(ft.ROOT / party / runtime["origin_run_id"]),
        "manifest": manifest,
        "shared": shared,
        "spec": spec,
        "test_capability": False,
        "is_smoke": runtime.get("diagnostic_is_smoke", False),
    }
    header = pd.read_csv(root / "input/train.csv", nrows=0)
    features = [c for c in header if c not in {"record_id", "label", "treatment"}]
    data["width"] = len(features)
    if party == "alice":
        frames = {s: pd.read_csv(root / "input" / f"{s}.csv") for s in PRETEST_SPLITS}
        data["features"] = ft.validate_frames(frames, party, spec)
        data["columns"] = list(header.columns)
        data["scaler"] = StandardScaler().fit(
            frames["train"][features].to_numpy(dtype=float)
        )
        save_json(
            root / "data/preprocessing.json",
            {
                "features": features,
                "mean": data["scaler"].mean_.tolist(),
                "scale": data["scaler"].scale_.tolist(),
                "fit_split": "train",
                "test_not_parsed": True,
            },
        )
    for split in PRETEST_SPLITS:
        data[split] = load_split(data, split)
    if set(map(tuple, data["train"]["ids"])) & set(
        map(tuple, data["validation"]["ids"])
    ):
        raise ValueError("Validation overlaps train")
    save_json(root / "logs/isolation.json", checks)
    return data


def shape(data):
    return {s: [len(data[s]["ids"]), data["width"]] for s in PRETEST_SPLITS}


def chunk(data, split, key, begin, end, seed=None):
    if split not in PRETEST_SPLITS and not data.get("test_capability"):
        raise ValueError("Test capability locked")
    return ft.get_chunk(data, split, key, begin, end, seed)


def fit_reference(data, name, seed, lr, bob_width):
    """Native installed update on Alice fields only; no test object in this function."""
    from secretflow.ml.linear.linear_model import RegType
    from secretflow.ml.linear.ss_sgd.model import Penalty, Strategy, _batch_update_w
    from secretflow.utils.sigmoid import SigType

    shared, uplift = data["shared"], data["spec"]["treatment"]
    ref_name = name.replace(ACTIVE_PREFIX, REFERENCE_PREFIX)
    ft.new_training(
        "alice",
        Path(data["root"]).name,
        ref_name,
        {
            "seed": seed,
            "learning_rate": lr,
            "shared": shared,
            "purpose": "own-fields native arithmetic reference; not a joint cleartext model",
        },
    )
    weights = mapped_weights(
        data["width"], bob_width, uplift, seed, shared["initialization_std"]
    )
    tr = data["train"]
    order = np.random.default_rng(seed).permutation(len(tr["y"]))
    designs, labels = [], []
    for begin in range(0, len(order), shared["infeed_rows"]):
        idx = order[begin : begin + shared["infeed_rows"]]
        padding = (-len(idx)) % shared["batch_size"]
        designs.append(cpu_design(tr["x"][idx], tr["t"][idx], uplift, -1, padding))
        labels.append(np.pad(tr["y"][idx], (0, padding)))

    def update(x, y, w):
        updated, _ = _batch_update_w(
            x,
            y,
            w,
            lr,
            shared["l2_norm"],
            SigType(shared["sigmoid"]),
            RegType.Logistic,
            Penalty.L2,
            len(x) // shared["batch_size"],
            shared["batch_size"],
            Strategy.NAIVE_SGD,
            None,
            False,
        )
        return np.clip(updated, -shared["weight_bound"], shared["weight_bound"])

    installed_update = _batch_update_w
    _batch_update_w = cpu_cache_hint_bridge(installed_update)
    native = update
    for _ in range(shared["epochs"]):
        for x, y in zip(designs, labels):
            weights = native(x, y, weights)
    predictions = predict_local(data, weights, PRETEST_SPLITS)
    folder = Path(data["root"]) / "trainings" / ref_name / "results"
    for split, prediction in predictions.items():
        np.save(folder / f"{split}_predictions.npy", prediction)
    save_json(
        folder / "status.json",
        {
            "training": "passed",
            "test_read": False,
            "installed_update_sha256": ft.fingerprint(
                inspect.getsourcefile(installed_update)
            ),
            "cpu_bridge": "NumPy tensor bindings and DF expression; SPU drop-cache hint is identity; native update bytecode retained; global SPU/JAX unchanged",
        },
    )
    # Alice owns these weights; they are neither exported nor given to Bob.
    np.save(folder / "own_weights.npy", np.asarray(weights))
    return {"name": ref_name}


def cpu_design(x, treatment, uplift, potential, padding):
    design = ft.local_design(x, treatment, uplift, potential)
    design = np.c_[design, np.ones((len(x), 1), dtype=x.dtype)]
    return np.pad(design, ((0, padding), (0, 0)))


def predict_local(data, weights, splits):
    uplift = data["spec"]["treatment"]
    result = {}
    for split in splits:
        row = data[split]
        potentials = (
            [0] + data["spec"].get("treatment_arms", [])
            if uplift and split != "train"
            else [-1]
        )
        result[split] = np.stack(
            [
                np.asarray(
                    numpy_df(
                        (
                            cpu_design(row["x"], row["t"], uplift, p, 0) @ weights
                        ).reshape(-1),
                        data["shared"]["sigmoid"],
                    )
                )
                for p in potentials
            ],
            axis=1,
        )
    return result


def record_candidate(data, name, train, validation, reference, tolerance):
    folder = Path(data["root"]) / "trainings" / name / "results"
    ref = Path(data["root"]) / "trainings" / reference["name"] / "results"
    epsilon, limit = (
        data["shared"]["score_epsilon"],
        data["shared"]["training_boundary_limit"],
    )
    numerical = {
        "tolerance": tolerance,
        "criterion": "fixed before real run",
        "split_errors": {},
    }
    for split, prediction in [("train", train), ("validation", validation)]:
        p = np.asarray(prediction)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Invalid probability")
        if split == "train" and np.mean((p <= epsilon) | (p >= 1 - epsilon)) > limit:
            raise ValueError("Training boundary limit failed")
        # Factual training is one column in both native and MPC paths.
        native = np.load(ref / f"{split}_predictions.npy")
        if not np.isfinite(native).all() or np.any((native < 0) | (native > 1)):
            raise ValueError("Invalid native probability")
        error = np.abs(p - native)
        numerical["split_errors"][split] = {
            "max_abs_error": float(error.max()),
            "mean_abs_error": float(error.mean()),
        }
        np.save(folder / f"{split}_predictions.npy", p)
    numerical["passed"] = all(
        r["max_abs_error"] <= tolerance for r in numerical["split_errors"].values()
    )
    v = data["validation"]
    score = validation_score(
        v["y"],
        validation,
        data["spec"],
        data["shared"],
        v["t"] if data["spec"]["treatment"] else None,
    )
    save_json(
        folder / "validation_score.json",
        {"score": score, "selection_split": "validation"},
    )
    save_json(folder / "native_equivalence.json", numerical)
    save_json(
        folder / "status.json",
        {
            "training": "passed",
            "native_equivalence": numerical["passed"],
            "test_read": False,
        },
    )
    return True


def freeze_selection(data, names):
    root = Path(data["root"])
    result = {}
    for seed in data["shared"]["model_seeds"]:
        options = names[str(seed)]
        scores = [
            json.loads(
                (root / "trainings" / n / "results/validation_score.json").read_text()
            )["score"]
            for n in options
        ]
        index = select_candidate(scores)
        result[str(seed)] = {
            "index": index,
            "name": options[index],
            "validation_scores": scores,
        }
    with (root / "results/frozen_selection.json").open("x") as f:
        json.dump(
            {"seeds": result, "selection_split": "validation", "test_parsed": False},
            f,
            indent=2,
        )
    return result


def selection_ready(data):
    selection = json.loads(
        (Path(data["root"]) / "results/frozen_selection.json").read_text()
    )
    return (
        set(selection["seeds"]) == set(map(str, data["shared"]["model_seeds"]))
        and not selection["test_parsed"]
    )


def selection_index(selection, seed):
    return selection[str(seed)]["index"]


def unlock_test(data, allowed):
    if not allowed:
        raise ValueError("Selections not frozen")
    if data["party"] == "alice" and not selection_ready(data):
        raise ValueError("Selections not frozen")
    root = Path(data["root"])
    record = {
        "all_validation_candidates_complete": True,
        "test_was_locked": not data["test_capability"],
    }
    if data["party"] == "alice":
        record["frozen_selection_sha256"] = ft.fingerprint(
            root / "results/frozen_selection.json"
        )
    save_json(root / "logs/test_capability.json", record)
    data["test_capability"] = True
    data["test"] = load_split(data, "test")
    return data


def finish_seed(data, selection, seed, test_predictions, tolerance):
    root, origin = Path(data["root"]), Path(data["origin_root"])
    item = selection[str(seed)]
    test = data["test"]
    reference = (
        root
        / "trainings"
        / item["name"].replace(ACTIVE_PREFIX, REFERENCE_PREFIX)
        / "results"
    )
    native = predict_local(data, np.load(reference / "own_weights.npy"), ("test",))[
        "test"
    ]
    error = np.abs(np.asarray(test_predictions) - native)
    if not np.isfinite(test_predictions).all() or np.any(
        (test_predictions < 0) | (test_predictions > 1)
    ):
        raise ValueError("Invalid test probability")
    folder = root / "results/selections" / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    np.save(folder / "test_predictions.npy", test_predictions)
    np.save(folder / "native_test_predictions.npy", native)
    diagnostic = {
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "tolerance": tolerance,
        "passed": bool(error.max() <= tolerance),
    }
    if not data.get("is_smoke"):
        declared = json.loads((root / "code/origin_output_manifest.json").read_text())
        if any(
            ft.fingerprint(origin / path) != digest for path, digest in declared.items()
        ):
            raise ValueError("Frozen origin output changed")
    frozen = (
        json.loads((origin / "results/private_evaluation.json").read_text())
        if not data.get("is_smoke")
        else {"metrics": []}
    )
    shared, spec = data["shared"], data["spec"]
    treatment = test["t"] if spec["treatment"] else None
    bootstrap_seed = shared["bootstrap_seed"] + seed
    metrics = bootstrap_metrics(
        test["y"], test_predictions, spec, shared, bootstrap_seed, treatment
    )
    comparisons = []
    origin_hashes = (
        {"evaluation": ft.fingerprint(origin / "results/private_evaluation.json")}
        if not data.get("is_smoke")
        else {}
    )
    for route in () if data.get("is_smoke") else ("L0", "L1_secure", "L3_secure"):
        chosen = next(
            r for r in frozen["metrics"] if r["seed"] == seed and r["route"] == route
        )
        path = (
            origin
            / "trainings"
            / chosen["selected_training"]
            / "results/test_predictions.npy"
        )
        old = np.load(path)
        origin_hashes[route] = ft.fingerprint(path)
        difference = bootstrap_metrics(
            test["y"], old, spec, shared, bootstrap_seed, treatment, test_predictions
        )
        comparisons.append(
            {
                "comparison": f"{route}_minus_active_secure",
                "metrics": difference,
                "origin_training": chosen["selected_training"],
            }
        )
    report = {
        "seed": seed,
        "metrics": metrics,
        "comparisons": comparisons,
        "native_test_equivalence": diagnostic,
        "origin_hashes": origin_hashes,
        "selected_training": item["name"],
        "scope": "known cohort diagnostic; not independent confirmation",
    }
    save_json(folder / "evaluation.json", report)
    return True


def finalize_private(data, names, tolerance):
    root = Path(data["root"])
    rows = [
        json.loads(
            (root / "results/selections" / f"seed{s}/evaluation.json").read_text()
        )
        for s in data["shared"]["model_seeds"]
    ]
    diagnostics = [
        json.loads(
            (root / "trainings" / n / "results/native_equivalence.json").read_text()
        )
        for ns in names.values()
        for n in ns
    ]
    report = {
        "status": "completed",
        "round": root.name,
        "origin": data["origin_root"],
        "seeds": rows,
        "numerical_diagnostics": diagnostics,
        "reference_tolerance": tolerance,
        "active_reference_equivalent": all(d["passed"] for d in diagnostics)
        and all(r["native_test_equivalence"]["passed"] for r in rows),
        "limitations": "Known previously viewed cohort, mapped initializer, frozen parameters; not business confirmation. Local equivalence covers active-only, not the unseen joint plaintext design.",
    }
    save_json(root / "results/private_diagnostic.json", report)
    lines = [
        "# 同优化器主动方诊断",
        "",
        report["limitations"],
        "",
        "| seed | 指标 | 主动方估计 [95% CI] |",
        "|---|---|---|",
    ]
    for row in rows:
        for metric, v in row["metrics"].items():
            lines.append(
                f"| {row['seed']} | {metric} | {v['estimate']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}] |"
            )
    lines += ["", "| seed | 比较 | 指标 | 差值 [95% CI] |", "|---|---|---|---|"]
    for row in rows:
        for comparison in row["comparisons"]:
            for metric, v in comparison["metrics"].items():
                lines.append(
                    f"| {row['seed']} | {comparison['comparison']} | {metric} | {v['estimate']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}] |"
                )
    lines += [
        "",
        f"全部主动方原生/MPC数值对照通过：{report['active_reference_equivalent']}；容差{tolerance}。",
        "即使通过也不证明联合全特征数值无误或消除人群漂移。不要基于本次测试调参或翻转排序。",
    ]
    (root / "results/同优化器诊断报告.md").write_text("\n".join(lines) + "\n")
    manifest_check(root)
    return {"status": "completed", "recipient": "alice"}


def train_active(runtime):
    import secretflow as sf
    import spu
    from secretflow.device import SPUCompilerNumReturnsPolicy
    from secretflow.ml.linear.linear_model import RegType
    from secretflow.ml.linear.ss_sgd.model import Penalty, Strategy, _batch_update_w
    from secretflow.utils.sigmoid import SigType

    cfg = yaml.safe_load(Path(runtime["config"]).read_text())
    shared, spec = cfg["shared"], cfg["datasets"][runtime["dataset"]]
    party, run_id = runtime["party"], runtime["run_id"]
    root = ft.ROOT / party / run_id
    tls_root = ft.ROOT / party / "tls"
    tls = {
        "cert": str(tls_root / "cert.pem"),
        "key": str(tls_root / "key.pem"),
        "ca_cert": str(tls_root / "ca.pem"),
    }
    install_pinned_tls_adapter()
    sf.init(
        ray_mode=False,
        cluster_config={"self_party": party, "parties": runtime["parties"]},
        tls_config=tls,
        cross_silo_comm_backend="grpc",
        logging_level="warning",
        enable_waiting_for_other_parties_ready=True,
        cross_silo_comm_options={"timeout_in_ms": shared["link_timeout_ms"]},
        job_name=run_id,
    )
    passed = False
    try:
        if not reject_anonymous_client(
            runtime["parties"][party]["address"],
            tls["ca_cert"],
            ft.TLS_NEGATIVE_TIMEOUT,
        ):
            raise ValueError("Unauthenticated federation accepted")
        nodes = []
        for name in PARTIES:
            directory = ft.ROOT / name / "tls"
            opts = {
                "certificate_path": str(directory / "cert.pem"),
                "private_key_path": str(directory / "key.pem"),
                "ca_file_path": str(directory / "ca.pem"),
                "verify_depth": ft.CERT_VERIFY_DEPTH,
            }
            nodes.append(
                {
                    "party": name,
                    "address": runtime["spu_addresses"][name],
                    "listen_address": f"0.0.0.0:{ft.SPU_PORT}",
                    "tls_opts": {"server_ssl_opts": opts, "client_ssl_opts": opts},
                }
            )
        log = spu.logging.LogOptions()
        (
            log.system_log_path,
            log.trace_log_path,
            log.trace_content_length,
            log.enable_console_logger,
        ) = "logs/spu.log", "", 0, False
        secure = sf.SPU(
            {
                "nodes": nodes,
                "runtime_config": {"protocol": cfg["protocol"], "field": cfg["field"]},
            },
            log_options=log,
            link_desc={
                "recv_timeout_ms": shared["link_timeout_ms"],
                "http_timeout_ms": shared["link_timeout_ms"],
            },
        )
        sf.wait(secure(lambda: np.array(True))())
        if not reject_anonymous_spu(
            runtime["spu_addresses"][party], tls["ca_cert"], ft.TLS_NEGATIVE_TIMEOUT
        ):
            raise ValueError("Unauthenticated SPU accepted")
        save_json(
            root / "logs/tls_checks.json",
            {
                "federation_anonymous_denied": True,
                "spu_anonymous_denied": True,
                "protocol_trace_disabled": True,
            },
        )
        alice, bob = sf.PYU("alice"), sf.PYU("bob")
        da = alice(prepare_active)("alice", runtime, cfg)
        db = bob(prepare_active)("bob", runtime, cfg)
        a_shape, b_shape = sf.reveal([alice(shape)(da), bob(shape)(db)])
        for split in PRETEST_SPLITS:
            if a_shape[split][0] != b_shape[split][0]:
                raise ValueError("Rows differ")
            aligned = secure(ft.secure_equal)(
                alice(ft.get_array)(da, split, "ids").to(secure),
                bob(ft.get_array)(db, split, "ids").to(secure),
            )
            if not bool(sf.reveal(aligned)):
                raise ValueError("Prealignment differs")
        names, weights_by_seed = {}, {}
        for seed in shared["model_seeds"]:
            designs, labels, sizes = {}, [], []
            for split in PRETEST_SPLITS:
                potentials = (
                    [0] + spec.get("treatment_arms", [])
                    if spec["treatment"] and split != "train"
                    else [-1]
                )
                designs[split] = [[] for _ in potentials]
                total = a_shape[split][0]
                block = (
                    shared["infeed_rows"]
                    if split == "train"
                    else shared["prediction_batch_size"]
                )
                for begin in range(0, total, block):
                    end = min(total, begin + block)
                    padding = (
                        (-(end - begin)) % shared["batch_size"]
                        if split == "train"
                        else 0
                    )
                    order_seed = seed if split == "train" else None
                    xa = alice(chunk)(da, split, "x", begin, end, order_seed).to(secure)
                    t = alice(chunk)(da, split, "t", begin, end, order_seed).to(secure)
                    for idx, potential in enumerate(potentials):
                        design = secure(
                            active_design,
                            static_argnames=("uplift", "potential", "padding"),
                        )(
                            xa,
                            t,
                            uplift=spec["treatment"],
                            potential=potential,
                            padding=padding,
                        )
                        sf.wait(design)
                        designs[split][idx].append(design)
                    if split == "train":
                        y = alice(chunk)(da, split, "y", begin, end, seed).to(secure)
                        labels.append(
                            secure(ft.pad_y, static_argnames="padding")(
                                y, padding=padding
                            )
                        )
                        sizes.append(end - begin + padding)
            names[str(seed)], weights_by_seed[seed] = [], []
            for candidate, lr in enumerate(shared["learning_rates"]):
                name = f"{ACTIVE_PREFIX}_seed{seed}_candidate{candidate}"
                detail = {
                    "route": ACTIVE_PREFIX,
                    "seed": seed,
                    "candidate": candidate,
                    "learning_rate": lr,
                    "shared": shared,
                }
                sf.wait(
                    [
                        alice(ft.new_training)("alice", run_id, name, detail),
                        bob(ft.new_training)("bob", run_id, name, detail),
                    ]
                )
                ref = alice(fit_reference)(da, name, seed, lr, b_shape["train"][1])
                weights = secure(
                    mapped_weights,
                    static_argnames=(
                        "alice_width",
                        "bob_width",
                        "uplift",
                        "seed",
                        "scale",
                    ),
                )(
                    alice_width=a_shape["train"][1],
                    bob_width=b_shape["train"][1],
                    uplift=spec["treatment"],
                    seed=seed,
                    scale=shared["initialization_std"],
                )
                started = time.monotonic()
                for epoch in range(shared["epochs"]):
                    for x, y, size in zip(designs["train"][0], labels, sizes):
                        weights, _ = secure(
                            _batch_update_w,
                            static_argnames=(
                                "sig_type",
                                "reg_type",
                                "penalty",
                                "total_batch",
                                "batch_size",
                                "strategy",
                                "enable_spu_cache",
                            ),
                            num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
                            user_specified_num_returns=2,
                        )(
                            x,
                            y,
                            weights,
                            lr,
                            shared["l2_norm"],
                            sig_type=SigType(shared["sigmoid"]),
                            reg_type=RegType.Logistic,
                            penalty=Penalty.L2,
                            total_batch=size // shared["batch_size"],
                            batch_size=shared["batch_size"],
                            strategy=Strategy.NAIVE_SGD,
                            dk_arr=None,
                            enable_spu_cache=False,
                        )
                        sf.wait(weights)
                        weights = secure(ft.project_weights, static_argnames="bound")(
                            weights, bound=shared["weight_bound"]
                        )
                        sf.wait(weights)
                    save_json(
                        root / "trainings" / name / "logs/progress.json",
                        {
                            "completed_epochs": epoch + 1,
                            "planned_epochs": shared["epochs"],
                        },
                    )
                secure.dump(
                    weights,
                    [
                        str(
                            ft.ROOT
                            / p
                            / run_id
                            / "trainings"
                            / name
                            / "results/model.share"
                        )
                        for p in PARTIES
                    ],
                )
                prediction = {}
                for split in PRETEST_SPLITS:
                    outputs = []
                    for blocks in designs[split]:
                        values = []
                        for design in blocks:
                            probability, valid = secure(
                                ft.checked_predict,
                                static_argnames="sig_type",
                                num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
                                user_specified_num_returns=2,
                            )(design, weights, sig_type=shared["sigmoid"])
                            if not bool(sf.reveal(valid)):
                                raise ValueError("Probability range failed")
                            values.append(probability.to(alice))
                        outputs.append(values)
                    prediction[split] = alice(ft.pack_chunked_predictions)(
                        outputs,
                        a_shape[split][0],
                        seed if split == "train" else None,
                        False,
                        da,
                        split,
                    )
                sf.wait(
                    alice(record_candidate)(
                        da,
                        name,
                        prediction["train"],
                        prediction["validation"],
                        ref,
                        cfg["diagnostic"]["reference_tolerance"],
                    )
                )
                names[str(seed)].append(name)
                weights_by_seed[seed].append(weights)
                print(
                    json.dumps(
                        {
                            "training": name,
                            "status": "completed_pretest",
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
        selection = alice(freeze_selection)(da, names)
        sf.wait(selection)
        allowed = bool(sf.reveal(alice(selection_ready)(da)))
        da, db = alice(unlock_test)(da, allowed), bob(unlock_test)(db, allowed)
        if not bool(
            sf.reveal(
                secure(ft.secure_equal)(
                    alice(ft.get_array)(da, "test", "ids").to(secure),
                    bob(ft.get_array)(db, "test", "ids").to(secure),
                )
            )
        ):
            raise ValueError("Test alignment differs")
        # The only Bob data ever fed above was the prealignment key digest, not features.
        for seed in shared["model_seeds"]:
            index = alice(selection_index)(selection, seed).to(secure)
            weights = secure(choose_secret_weights)(weights_by_seed[seed], index)
            total = sf.reveal(alice(lambda d: len(d["test"]["ids"]))(da))
            outputs = []
            for potential in (
                [0] + spec.get("treatment_arms", []) if spec["treatment"] else [-1]
            ):
                values = []
                for begin in range(0, total, shared["prediction_batch_size"]):
                    end = min(total, begin + shared["prediction_batch_size"])
                    x, t = (
                        alice(chunk)(da, "test", "x", begin, end).to(secure),
                        alice(chunk)(da, "test", "t", begin, end).to(secure),
                    )
                    design = secure(
                        active_design,
                        static_argnames=("uplift", "potential", "padding"),
                    )(x, t, uplift=spec["treatment"], potential=potential, padding=0)
                    p, valid = secure(
                        ft.checked_predict,
                        static_argnames="sig_type",
                        num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
                        user_specified_num_returns=2,
                    )(design, weights, sig_type=shared["sigmoid"])
                    if not bool(sf.reveal(valid)):
                        raise ValueError("Test probability range failed")
                    values.append(p.to(alice))
                outputs.append(values)
            predictions = alice(ft.pack_chunked_predictions)(
                outputs, total, None, False, da, "test"
            )
            sf.wait(
                alice(finish_seed)(
                    da,
                    selection,
                    seed,
                    predictions,
                    cfg["diagnostic"]["reference_tolerance"],
                )
            )
        sf.wait(
            alice(finalize_private)(da, names, cfg["diagnostic"]["reference_tolerance"])
        )
        manifest_check(root)
        if party == "bob" and (
            list(root.rglob("*predictions.npy")) or list(root.rglob("own_weights.npy"))
        ):
            raise ValueError("Bob obtained plaintext output")
        save_json(
            root / "logs/output_boundary.json",
            {
                "frozen_input_unchanged": True,
                "bob_plaintext_output_check": "passed",
                "joint_weights": "secret_shares_only",
                "bob_feature_input_to_training": False,
                "selection_index_disclosed_to_bob": False,
            },
        )
        save_json(
            root / "results/status.json",
            {"status": "completed", "score_access": "alice_only"},
        )
        passed = True
    finally:
        sf.shutdown(barrier_on_shutdown=passed, on_error=not passed)
