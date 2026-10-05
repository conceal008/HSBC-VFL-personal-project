"""Synthetic-only complete joint numerical verification; never joins real rows."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
import functional_training as ft  # type: ignore[import-not-found]
import active_diagnostic as ad  # type: ignore[import-not-found]
import functional_vm as vm  # type: ignore[import-not-found]
from functional_tls import (
    install_pinned_tls_adapter,
    reject_anonymous_client,
    reject_anonymous_spu,
)  # type: ignore[import-not-found]
from functional_metrics import save_json  # type: ignore[import-not-found]

PARTIES = ("alice", "bob")
SPLITS = ("train", "validation")


def synthetic_guard(runtime, cfg):
    if not (
        runtime.get("is_smoke") is True
        and runtime.get("synthetic_only") is True
        and cfg.get("synthetic_only") is True
    ):
        raise ValueError("Joint plaintext reference is synthetic-only")


def numpy_design(xa, xb, treatment, uplift, potential=-1, padding=0):
    """Independent explicit column contract: A,B,arms,feature-major products,bias."""
    x = np.concatenate((xa, xb), axis=1)
    columns = [x]
    if uplift:
        arm = treatment if potential < 0 else np.full_like(treatment, potential)
        indicators = np.column_stack((arm == 1, arm == 2)).astype(x.dtype)
        columns.append(indicators)
        columns.append(
            np.column_stack(
                [
                    x[:, j] * indicators[:, k]
                    for j in range(x.shape[1])
                    for k in range(2)
                ]
            )
        )
    columns.append(np.ones((len(x), 1), dtype=x.dtype))
    return np.pad(np.concatenate(columns, axis=1), ((0, padding), (0, 0)))


def equation_update(x, y, w, lr, l2, batch_size):
    """Independent DF/L2/NAIVE_SGD equation; bias is unpenalized."""
    w = np.asarray(w).reshape(-1, 1).copy()
    for start in range(0, len(x), batch_size):
        block = x[start : start + batch_size]
        z = block @ w
        prob = 0.5 + 0.5 * z / (1 + np.abs(z))  # 魔数豁免: DF近似的固定数学系数，非超参数
        gradient = (
            block.T @ (prob - y[start : start + batch_size].reshape(-1, 1)) / batch_size
        )
        regularizer = w.copy()
        regularizer[-1] = 0
        w -= lr * gradient + lr * l2 * regularizer / batch_size
    return w


def prepare(party, runtime, cfg):
    synthetic_guard(runtime, cfg)
    root = ft.ROOT / party / runtime["run_id"]
    checks = ft.isolation_preflight(party)
    ad.manifest_check(root)
    frames = {split: pd.read_csv(root / "input" / f"{split}.csv") for split in SPLITS}
    spec, shared = cfg["datasets"][runtime["dataset"]], cfg["shared"]
    features = ft.validate_frames(frames, party, spec)
    scaler = StandardScaler().fit(frames["train"][features])
    data = dict(
        root=str(root),
        party=party,
        spec=spec,
        shared=shared,
        runtime=runtime,
        cfg=cfg,
        width=len(features),
    )
    for split, frame in frames.items():
        x = np.clip(
            scaler.transform(frame[features]),
            -shared["clip_features"],
            shared["clip_features"],
        ).astype(np.float32)
        import hashlib

        ids = np.stack(
            [
                np.frombuffer(hashlib.sha256(str(key).encode()).digest(), dtype="<u4")
                for key in frame["record_id"]
            ]
        )
        data[split] = {"x": x, "ids": ids}
        if party == "alice":
            data[split].update(
                y=frame["label"].to_numpy(dtype=np.float32),
                t=frame["treatment"].to_numpy(dtype=np.int32)
                if spec["treatment"]
                else np.zeros(len(frame), dtype=np.int32),
            )
        np.savez_compressed(root / "data" / f"{split}_local.npz", **data[split])
    save_json(
        root / "data/preprocessing.json",
        {
            "fit_split": "train",
            "features": features,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
    )
    save_json(root / "logs/isolation.json", checks)
    return data


def reference(data, name, init_seed, order_seed, lr):
    """Regenerate public synthetic inputs only. Never open peer data paths."""
    from secretflow.ml.linear.ss_sgd.model import _batch_update_w, Penalty, Strategy
    from secretflow.ml.linear.linear_model import RegType
    from secretflow.utils.sigmoid import SigType

    cfg, runtime = data["cfg"], data["runtime"]
    synthetic_guard(runtime, cfg)
    shared, spec = data["shared"], data["spec"]
    arrays = {}
    for party in PARTIES:
        frames = vm.synthetic_inputs(cfg, runtime["dataset"], party)
        features = [
            c
            for c in frames["train"].columns
            if c not in ("record_id", "label", "treatment")
        ]
        scaler = StandardScaler().fit(frames["train"][features])
        arrays[party] = {
            split: np.clip(
                scaler.transform(frames[split][features]),
                -shared["clip_features"],
                shared["clip_features"],
            ).astype(np.float32)
            for split in SPLITS
        }
    if not all(
        np.array_equal(arrays["alice"][split], data[split]["x"]) for split in SPLITS
    ):
        raise ValueError("Synthetic source differs from reconstruction")
    a, b = arrays["alice"], arrays["bob"]
    tr = data["train"]
    order = np.random.default_rng(order_seed).permutation(len(tr["x"]))
    designs, labels = [], []
    for begin in range(0, len(order), shared["infeed_rows"]):
        selected = order[begin : begin + shared["infeed_rows"]]
        padding = (-len(selected)) % shared["batch_size"]
        designs.append(
            numpy_design(
                a["train"][selected],
                b["train"][selected],
                tr["t"][selected],
                spec["treatment"],
                padding=padding,
            )
        )
        labels.append(np.pad(tr["y"][selected], (0, padding)))
    weights = ft.initial_weights(
        designs[0].shape[1], init_seed, shared["initialization_std"]
    )
    native = ad.cpu_cache_hint_bridge(_batch_update_w)
    max_error = 0.0
    for _ in range(shared["epochs"]):
        for x, y in zip(designs, labels):
            expected = equation_update(
                x, y, weights, lr, shared["l2_norm"], shared["batch_size"]
            )
            actual, _ = native(
                x,
                y,
                weights,
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
            max_error = max(max_error, float(np.max(np.abs(expected - actual))))
            weights = np.clip(expected, -shared["weight_bound"], shared["weight_bound"])
    if max_error > cfg["diagnostic"]["equation_tolerance"]:
        raise ValueError("Native update differs from independent equation")
    folder = Path(data["root"]) / "trainings" / name / "results"
    for split in SPLITS:
        row = data[split]
        potentials = [0, 1, 2] if spec["treatment"] and split != "train" else [-1]
        prediction = np.column_stack(
            [
                ad.numpy_df(
                    (
                        numpy_design(
                            a[split], b[split], row["t"], spec["treatment"], potential=p
                        )
                        @ weights
                    ).reshape(-1),
                    shared["sigmoid"],
                )
                for p in potentials
            ]
        )
        np.save(folder / f"{split}_reference_predictions.npy", prediction)
    save_json(
        folder / "equation_check.json",
        {
            "passed": True,
            "max_abs_error": max_error,
            "tolerance": cfg["diagnostic"]["equation_tolerance"],
            "reference": "independent equation and native bytecode; synthetic only",
        },
    )
    return True


def record(data, name, predictions):
    folder = Path(data["root"]) / "trainings" / name / "results"
    report = {
        "name": name,
        "passed": True,
        "splits": {},
        "tolerance": data["cfg"]["diagnostic"]["reference_tolerance"],
    }
    for split in SPLITS:
        p = np.asarray(predictions[split])
        expected = np.load(folder / f"{split}_reference_predictions.npy")
        if (
            not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1))
            or p.shape != expected.shape
        ):
            raise ValueError("Invalid probability or prediction shape")
        difference = np.abs(p - expected)
        report["splits"][split] = {
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
        }
        report["passed"] &= difference.max() <= report["tolerance"]
        np.save(folder / f"{split}_predictions.npy", p)
    report["passed"] = bool(report["passed"])
    save_json(folder / "numerical_check.json", report)
    if not report["passed"]:
        raise ValueError("Joint numerical tolerance failed; method experiments blocked")
    return True


def finalize(data, names):
    cfg = data["cfg"]
    root = Path(data["root"])
    rows = []
    summary = []
    for name in names:
        check = json.loads(
            (root / "trainings" / name / "results/numerical_check.json").read_text()
        )
        details = json.loads(
            (root / "trainings" / name / "code/training_config.json").read_text()
        )
        rows.append(
            {
                **check,
                "initialization_seed": details["initialization_seed"],
                "order_seed": details["order_seed"],
                "learning_rate": details["learning_rate"],
            }
        )
    rng = np.random.default_rng(cfg["diagnostic"]["bootstrap_seed"])
    for lr in cfg["shared"]["learning_rates"]:
        for split in SPLITS:
            errors = np.array(
                [
                    row["splits"][split]["max_abs_error"]
                    for row in rows
                    if row["learning_rate"] == lr
                ]
            )
            sampled = errors[
                rng.integers(
                    len(errors),
                    size=(cfg["diagnostic"]["bootstrap_repeats"], len(errors)),
                )
            ].mean(axis=1)
            low, high = np.percentile(sampled, [2.5, 97.5])  # 魔数豁免: 预声明95%区间百分位，不参与选参
            summary.append(
                {
                    "learning_rate": lr,
                    "split": split,
                    "maximum": float(errors.max()),
                    "mean": float(errors.mean()),
                    "seed_mean_ci_95": [float(low), float(high)],
                    "seed_count": len(errors),
                }
            )
    result = {
        "status": "passed",
        "synthetic_only": True,
        "dataset": data["runtime"]["dataset"],
        "rows": rows,
        "summary": summary,
        "scope": cfg["diagnostic"]["scope"],
        "business_gain_evaluated": False,
    }
    save_json(root / "results/joint_math_report.json", result)
    return {"status": "passed", "synthetic_only": True}


def train_joint(runtime):
    import secretflow as sf
    import spu
    from secretflow.device import SPUCompilerNumReturnsPolicy
    from secretflow.ml.linear.linear_model import RegType
    from secretflow.ml.linear.ss_sgd.model import Penalty, Strategy, _batch_update_w
    from secretflow.utils.sigmoid import SigType

    cfg = yaml.safe_load(Path(runtime["config"]).read_text())
    synthetic_guard(runtime, cfg)
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
        da = alice(prepare)("alice", runtime, cfg)
        db = bob(prepare)("bob", runtime, cfg)
        a_shape, b_shape = sf.reveal([alice(ad.shape)(da), bob(ad.shape)(db)])
        for split in SPLITS:
            if a_shape[split][0] != b_shape[split][0] or not bool(
                sf.reveal(
                    secure(ft.secure_equal)(
                        alice(ft.get_array)(da, split, "ids").to(secure),
                        bob(ft.get_array)(db, split, "ids").to(secure),
                    )
                )
            ):
                raise ValueError("Prealignment mismatch")
        names = []
        for seed in shared["model_seeds"]:
            order_seed = cfg["diagnostic"]["order_seeds"][str(seed)]
            designs, labels, sizes = {}, [], []
            for split in SPLITS:
                potentials = (
                    [0, 1, 2] if spec["treatment"] and split != "train" else [-1]
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
                    shuffle = order_seed if split == "train" else None
                    xa = alice(ad.chunk)(da, split, "x", begin, end, shuffle).to(secure)
                    xb = bob(ad.chunk)(db, split, "x", begin, end, shuffle).to(secure)
                    t = alice(ad.chunk)(da, split, "t", begin, end, shuffle).to(secure)
                    for i, p in enumerate(potentials):
                        x = secure(
                            ft.chunk_design,
                            static_argnames=(
                                "route",
                                "is_uplift",
                                "potential",
                                "padding",
                            ),
                        )(
                            xa,
                            xb,
                            None,
                            None,
                            t,
                            route="L3_secure",
                            is_uplift=spec["treatment"],
                            potential=p,
                            padding=padding,
                        )
                        sf.wait(x)
                        designs[split][i].append(x)
                    if split == "train":
                        y = alice(ad.chunk)(da, split, "y", begin, end, order_seed).to(
                            secure
                        )
                        labels.append(
                            secure(ft.pad_y, static_argnames="padding")(
                                y, padding=padding
                            )
                        )
                        sizes.append(end - begin + padding)
            for candidate, lr in enumerate(shared["learning_rates"]):
                name = f"joint_secure_seed{seed}_candidate{candidate}"
                names.append(name)
                details = {
                    "purpose": "pure synthetic complete joint numerical verification",
                    "initialization_seed": seed,
                    "order_seed": order_seed,
                    "data_seed": cfg["smoke"]["data_seed"],
                    "learning_rate": lr,
                    "shared": shared,
                }
                sf.wait(
                    [
                        alice(ft.new_training)("alice", run_id, name, details),
                        bob(ft.new_training)("bob", run_id, name, details),
                    ]
                )
                sf.wait(alice(reference)(da, name, seed, order_seed, lr))
                width = a_shape["train"][1] + b_shape["train"][1]
                dimension = width * 3 + 3 if spec["treatment"] else width + 1  # 魔数豁免: 两个处理组交互的固定列维度契约
                weights = secure(
                    ft.initial_weights, static_argnames=("n", "seed", "scale")
                )(n=dimension, seed=seed, scale=shared["initialization_std"])
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
                predictions = {}
                for split in SPLITS:
                    outputs = [[] for _ in designs[split]]
                    for i, chunks in enumerate(designs[split]):
                        for x in chunks:
                            score, valid = secure(
                                ft.checked_predict,
                                static_argnames="sig_type",
                                num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
                                user_specified_num_returns=2,
                            )(x, weights, sig_type=shared["sigmoid"])
                            if not bool(sf.reveal(valid)):
                                raise ValueError("MPC probability range failed")
                            outputs[i].append(score.to(alice))
                    predictions[split] = alice(ft.pack_chunked_predictions)(
                        outputs,
                        a_shape[split][0],
                        order_seed if split == "train" else None,
                        False,
                        da,
                        split,
                    )
                sf.wait(alice(record)(da, name, predictions))
                print(
                    json.dumps(
                        {
                            "completed_training": name,
                            "epoch_count": shared["epochs"],
                            "synthetic_only": True,
                        }
                    ),
                    flush=True,
                )
        sf.wait(alice(finalize)(da, names))
        sf.wait(
            [
                alice(ad.manifest_check)(ft.ROOT / "alice" / run_id),
                bob(ad.manifest_check)(ft.ROOT / "bob" / run_id),
            ]
        )
        save_json(
            root / "logs/output_boundary.json",
            {
                "frozen_input_unchanged": True,
                "synthetic_only": True,
                "joint_weights": "secret shares only",
                "scores_receiver": "alice",
                "old_test_read": False,
                "model_count": len(names),
            },
        )
        if party == "bob" and list(root.rglob("*predictions.npy")):
            raise ValueError("Bob received prediction")
        passed = True
    finally:
        sf.shutdown(barrier_on_shutdown=passed, on_error=not passed)
    return {"status": "passed", "synthetic_only": True, "receiver": party}
