"""Actual two-party native SS-SGD trajectories with frozen 12/24/48 endpoints.

Gradients, weights and their scalar norms never leave SPU. Alice receives only
already-authorized predictions/evaluation and predeclared diagnostic booleans.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import yaml
import candidate_full as cf  # type: ignore[import-not-found]
import nonlinear_full as nf  # type: ignore[import-not-found]
import functional_training as ft  # type: ignore[import-not-found]
import active_diagnostic as ad  # type: ignore[import-not-found]
from functional_metrics import save_json  # type: ignore[import-not-found]
from functional_tls import (
    install_pinned_tls_adapter,
    reject_anonymous_client,
    reject_anonymous_spu,
)  # type: ignore[import-not-found]

PARTIES = cf.PARTIES
SPLITS = cf.SPLITS
metadata = nf.metadata
MINIMUM_SEEDS = 5
CHECKPOINTS = (12, 24, 48)


def validate(cfg):
    shared, diag = cfg["shared"], cfg["diagnostic"]
    if (
        tuple(diag["checkpoints"]) != CHECKPOINTS
        or shared["epochs"] != max(CHECKPOINTS)
        or len(set(shared["model_seeds"])) < MINIMUM_SEEDS
        or shared["learning_rates"] != [diag["frozen_learning_rate"]]
        or set(cfg["experiment"]["conditions"]) != {"D_A0", "D_A1", "D_B0", "D_B1"}
        or cfg["experiment"]["max_rows"] != {"train": None, "validation": None}
        or set(cfg["experiment"]["order_seeds"]) != set(map(str, shared["model_seeds"]))
    ):
        raise ValueError("Unregistered diagnostic schedule")
    for cell, condition in cfg["experiment"]["conditions"].items():
        if condition != dict(
            enhanced=cell.endswith("1"),
            include_bob=cell.startswith("D_B"),
            mask=False,
            cross=False,
            negative=False,
        ):
            raise ValueError("Diagnostic factor changed")


def gradient_sum(x, y, w, valid_rows):
    from secretflow.utils.sigmoid import sigmoid, SigType

    xx, yy = x[:valid_rows], y[:valid_rows].reshape((-1, 1))
    return xx.T @ (sigmoid(xx @ w, SigType.DF) - yy)


def gradient_accept(gradient, w, total_rows, l2, batch_size, tolerance):
    import jax.numpy as jnp

    regularized = jnp.concatenate((w[:-1], jnp.zeros_like(w[-1:])), axis=0)
    g = gradient / total_rows + (l2 / batch_size) * regularized
    return jnp.sum(g * g) <= tolerance * tolerance


def record_checkpoint(
    data, name, epoch, predictions, cell, seed, tolerance, gradient_small
):
    root = Path(data["root"])
    snapshot = f"{name}/checkpoints/e{epoch}"
    folder = root / "trainings" / snapshot
    (folder / "results").mkdir(parents=True, exist_ok=False)
    nf.record(data, snapshot, predictions)
    reference_pass = None
    if epoch == CHECKPOINTS[0]:
        source_cell = cell.removeprefix("D_")
        for split in SPLITS:
            reference = np.load(
                root / "input/references" / f"{source_cell}_seed{seed}_{split}.npy"
            )
            p = np.asarray(predictions[split])
            if (
                p.shape != reference.shape
                or not np.isfinite(reference).all()
                or np.max(np.abs(p - reference)) > tolerance
            ):
                raise ValueError("12-epoch original MPC reference differs")
        reference_pass = True
    save_json(
        folder / "results/diagnostic.json",
        {
            "epoch": epoch,
            "original_reference_passed": reference_pass,
            "full_unpadded_gradient_below_threshold": bool(gradient_small),
            "gradient_values_disclosed": False,
            "convergence_proven": False,
        },
    )
    return snapshot


def checkpoint(
    secure,
    alice,
    bob,
    da,
    designs,
    labels,
    sizes,
    weights,
    a_shape,
    order_seed,
    name,
    cell,
    seed,
    epoch,
    cfg,
    records,
):
    import secretflow as sf
    from secretflow.device import SPUCompilerNumReturnsPolicy

    shared = cfg["shared"]
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
    total = a_shape["train"][0]
    gradient = secure(lambda w: np.zeros(w.shape, dtype=np.float32))(weights)
    for index, (x, y) in enumerate(zip(designs["train"][0], labels)):
        valid_rows = min(shared["infeed_rows"], total - index * shared["infeed_rows"])
        part = secure(gradient_sum, static_argnames="valid_rows")(
            x, y, weights, valid_rows=valid_rows
        )
        gradient = secure(lambda a, b: a + b)(gradient, part)
        sf.wait(gradient)
    small = secure(gradient_accept)(
        gradient,
        weights,
        total,
        shared["l2_norm"],
        shared["batch_size"],
        cfg["diagnostic"]["gradient_tolerance"],
    )
    snapshot = f"{name}/checkpoints/e{epoch}"
    sf.wait(
        alice(record_checkpoint)(
            da,
            name,
            epoch,
            predictions,
            cell,
            seed,
            shared["precision_tolerance"],
            small.to(alice),
        )
    )
    records[f"{cell}_e{epoch}"] = snapshot


def train_full(runtime):
    import secretflow as sf
    import spu
    from secretflow.device import SPUCompilerNumReturnsPolicy
    from secretflow.ml.linear.linear_model import RegType
    from secretflow.ml.linear.ss_sgd.model import Penalty, Strategy, _batch_update_w
    from secretflow.utils.sigmoid import SigType

    cfg = yaml.safe_load(Path(runtime["config"]).read_text())
    validate(cfg)
    shared, spec = cfg["shared"], cfg["datasets"][runtime["dataset"]]
    import time

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
        da = alice(cf.prepare)("alice", runtime, cfg)
        db = bob(cf.prepare)("bob", runtime, cfg)
        a_meta, b_meta = sf.reveal([alice(metadata)(da), bob(metadata)(db)])
        a_shape = {s: [a_meta["rows"][s], a_meta["base_width"]] for s in SPLITS}
        b_shape = {s: [b_meta["rows"][s], b_meta["base_width"]] for s in SPLITS}
        if not bool(
            sf.reveal(secure(cf.numerical_cross_accept)(shared["precision_tolerance"]))
        ):
            raise ValueError("Synthetic cross-term acceptance failed")
        save_json(root / "logs/cross_control.json", {"native_known_answer": True})
        for split in SPLITS:
            if a_shape[split][0] != b_shape[split][0]:
                raise ValueError("Prealignment dimensions differ")
            aligned = secure(lambda: np.array(True))()
            for begin in range(
                0, a_shape[split][0], cfg["experiment"]["alignment_chunk_rows"]
            ):
                end = min(
                    a_shape[split][0], begin + cfg["experiment"]["alignment_chunk_rows"]
                )
                checked = secure(ft.secure_equal)(
                    alice(ad.chunk)(da, split, "ids", begin, end, None).to(secure),
                    bob(ad.chunk)(db, split, "ids", begin, end, None).to(secure),
                )
                aligned = secure(lambda a, b: a & b)(aligned, checked)
                sf.wait(aligned)
            if not bool(sf.reveal(aligned)):
                raise ValueError("Prealignment mismatch")
        records = {}
        for seed in shared["model_seeds"]:
            order_seed = cfg["experiment"]["order_seeds"][str(seed)]
            own_records = alice(cf.local_references)(da, seed, run_id)
            records[str(seed)] = {
                cell: f"{cell}_seed{seed}"
                for cell in cfg["experiment"]["reference_cells"]
            }
            sf.wait(own_records)
            for cell, condition in cfg["experiment"]["conditions"].items():
                name = f"{cell}_seed{seed}"
                if name in runtime.get("reused_completed_fits", []):
                    for epoch in cfg["diagnostic"]["checkpoints"]:
                        records[str(seed)][f"{cell}_e{epoch}"] = (
                            f"{name}/checkpoints/e{epoch}"
                        )
                    continue
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
                        xa = alice(cf.feature_chunk)(
                            da,
                            split,
                            begin,
                            end,
                            shuffle,
                            condition["enhanced"],
                            True,
                            False,
                            condition["cross"],
                            False,
                        ).to(secure)
                        xb = bob(cf.feature_chunk)(
                            db,
                            split,
                            begin,
                            end,
                            shuffle,
                            False,
                            condition["include_bob"],
                            condition["mask"],
                            condition["cross"],
                            condition["negative"],
                        ).to(secure)
                        t = alice(ad.chunk)(da, split, "t", begin, end, shuffle).to(
                            secure
                        )
                        for i, p in enumerate(potentials):
                            x = secure(
                                cf.candidate_design,
                                static_argnames=(
                                    "route",
                                    "is_uplift",
                                    "potential",
                                    "padding",
                                    "cross_width",
                                    "cross_bound",
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
                                cross_width=len(
                                    cfg["experiment"]["groups"][runtime["dataset"]][
                                        "alice_cross"
                                    ]
                                )
                                if condition["cross"]
                                else 0,
                                cross_bound=cfg["experiment"]["cross_bound"],
                            )
                            sf.wait(x)
                            designs[split][i].append(x)
                        if split == "train":
                            y = alice(ad.chunk)(
                                da, split, "y", begin, end, order_seed
                            ).to(secure)
                            labels.append(
                                secure(ft.pad_y, static_argnames="padding")(
                                    y, padding=padding
                                )
                            )
                            sizes.append(end - begin + padding)
                for candidate, lr in enumerate(shared["learning_rates"]):
                    name = f"{cell}_seed{seed}"
                    details = {
                        "purpose": "actual MPC fixed-budget optimization diagnostic",
                        "cell": cell,
                        "initialization_seed": seed,
                        "order_seed": order_seed,
                        "selection_seed": cfg["experiment"]["selection_seed"],
                        "learning_rate": lr,
                        "shared": shared,
                        "condition": condition,
                        "group": cfg["experiment"]["groups"][runtime["dataset"]],
                        "negative_seeds": cfg["experiment"]["negative_seeds"],
                    }
                    sf.wait(
                        [
                            alice(ft.new_training)("alice", run_id, name, details),
                            bob(ft.new_training)("bob", run_id, name, details),
                        ]
                    )
                    weights = secure(
                        cf.initial_weights,
                        static_argnames=(
                            "a",
                            "b",
                            "extra",
                            "uplift",
                            "seed",
                            "scale",
                            "include_bob",
                            "cross_width",
                        ),
                    )(
                        a=a_meta["base_width"],
                        b=b_meta["base_width"],
                        extra=a_meta["extra_width"] if condition["enhanced"] else 0,
                        uplift=spec["treatment"],
                        seed=seed,
                        scale=shared["initialization_std"],
                        include_bob=condition["include_bob"],
                        cross_width=len(
                            cfg["experiment"]["groups"][runtime["dataset"]][
                                "alice_cross"
                            ]
                        )
                        if condition["cross"]
                        else 0,
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
                            weights = secure(
                                ft.project_weights, static_argnames="bound"
                            )(weights, bound=shared["weight_bound"])
                            sf.wait(weights)
                        if epoch + 1 in cfg["diagnostic"]["checkpoints"]:
                            checkpoint(
                                secure,
                                alice,
                                bob,
                                da,
                                designs,
                                labels,
                                sizes,
                                weights,
                                a_shape,
                                order_seed,
                                name,
                                cell,
                                seed,
                                epoch + 1,
                                cfg,
                                records[str(seed)],
                            )
                        save_json(
                            root / "trainings" / name / "logs/progress.json",
                            {
                                "completed_epochs": epoch + 1,
                                "planned_epochs": shared["epochs"],
                                "elapsed_training_seconds": time.monotonic() - started,
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
                    print(
                        json.dumps(
                            {
                                "completed_training": name,
                                "epoch_count": shared["epochs"],
                                "full_partition_training": True,
                            }
                        ),
                        flush=True,
                    )
        sf.wait(alice(nf.finalize)(da, records))
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
                "full_partition_training": True,
                "joint_weights": "secret shares only",
                "scores_receiver": "alice",
                "old_test_read": False,
                "model_count": len(shared["model_seeds"])
                * len(cfg["experiment"]["conditions"]),
            },
        )
        if party == "bob" and list(root.rglob("*predictions.npy")):
            raise ValueError("Bob received prediction")
        passed = True
    finally:
        sf.shutdown(barrier_on_shutdown=passed, on_error=not passed)
    return {"status": "passed", "full_partition_training": True, "receiver": party}
