"""Fixed paper-motivated factor adaptation on owner-isolated native MPC training."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import nonlinear_full as nf  # type: ignore[import-not-found]
import functional_training as ft  # type: ignore[import-not-found]
import active_diagnostic as ad  # type: ignore[import-not-found]
from functional_metrics import save_json  # type: ignore[import-not-found]
from functional_tls import (
    install_pinned_tls_adapter,
    reject_anonymous_client,
    reject_anonymous_spu,
)  # type: ignore[import-not-found]

PARTIES = nf.PARTIES
SPLITS = nf.SPLITS
UPLIFT_COLUMN_FACTOR = nf.UPLIFT_COLUMN_FACTOR
SYNTHETIC_XA = ((1.0, 2.0, 3.0), (2.0, 1.0, -2.0))
SYNTHETIC_XB = ((2.0, -1.0, 2.0), (1.0, 2.0, 3.0))
SYNTHETIC_EXPECTED = ((1.0, 2.0, -2.0, 4.0, 1.0), (2.0, 1.0, 2.0, -4.0, 1.0))
SYNTHETIC_CROSS_BOUND = 4
metadata = nf.metadata
record = nf.record


def prepare(party, runtime, cfg):
    data = nf.prepare(party, runtime, cfg)
    root = Path(data["root"])
    data["feature_names"] = ft.validate_frames(
        {s: pd.read_csv(root / "input" / (s + ".csv")) for s in SPLITS},
        party,
        data["spec"],
    )
    data["group"] = cfg["experiment"]["groups"][runtime["dataset"]]
    for split in SPLITS:
        with np.load(root / "input/references" / (split + "_local.npz")) as old:
            if set(old.files) != set(data[split]) or any(
                not np.array_equal(old[k], data[split][k]) for k in old.files
            ):
                raise ValueError("Reference matrix differs: refuse baseline reuse")
    save_json(
        root / "logs/reference_equivalence.json",
        {
            "owner_matrices_exact": True,
            "original_predictions_readonly": True,
            "not_retrained": True,
        },
    )
    return data


def feature_chunk(
    data,
    split,
    begin,
    end,
    seed,
    enhanced,
    include,
    mask=False,
    cross=False,
    negative=False,
):
    if split not in SPLITS:
        raise ValueError("Test capability denied")
    key = "enhanced_x" if enhanced else "x"
    x = ft.get_chunk(data, split, key, begin, end, seed).copy()
    names = data["feature_names"]
    if mask and data["party"] == "bob":
        keep = set(data["group"]["bob_keep"])
        x[:, [i for i, n in enumerate(names) if n not in keep]] = 0
    if cross:
        selected = [names.index(n) for n in data["group"][data["party"] + "_cross"]]
        own = data[split]["x"][:, selected]
        if negative and data["party"] == "bob":
            permutation = np.random.default_rng(
                data["cfg"]["experiment"]["negative_seeds"][split]
            ).permutation(len(own))
            own = own[permutation]
        if seed is not None:
            own = own[np.random.default_rng(seed).permutation(len(own))]
        x = np.c_[x, own[begin:end]]
    return x if include else np.zeros((len(x), 0), dtype=np.float32)


def candidate_design(
    xa,
    xb,
    groups,
    table,
    treatment,
    route,
    is_uplift,
    potential,
    padding,
    cross_width=0,
    cross_bound=1,
):
    import jax.numpy as jnp

    if cross_width:
        products = jnp.clip(
            xa[:, -cross_width:] * xb[:, -cross_width:], -cross_bound, cross_bound
        )
        xa = jnp.concatenate(
            (xa[:, :-cross_width], xb[:, :-cross_width], products), axis=1
        )
        xb = jnp.zeros((xa.shape[0], 0), dtype=xa.dtype)
    return ft.chunk_design(
        xa, xb, groups, table, treatment, route, is_uplift, potential, padding
    )


def initial_weights(a, b, extra, uplift, seed, scale, include_bob, cross_width=0):
    base = nf.initial_weights(a, b, extra, uplift, seed, scale, include_bob)
    if not cross_width:
        return base
    width = a + b + extra
    new_width = width + cross_width
    out = np.zeros(
        (
            new_width * UPLIFT_COLUMN_FACTOR + UPLIFT_COLUMN_FACTOR
            if uplift
            else new_width + 1,
            1,
        ),
        dtype=np.float32,
    )
    out[:width] = base[:width]
    if uplift:
        out[new_width : new_width + 2] = base[width : width + 2]
        out[new_width + 2 : new_width + 2 + width * 2] = base[
            width + 2 : width + 2 + width * 2
        ]
    out[-1] = base[-1]
    return out


def local_references(data, seed, run_id):
    names = {}
    root = Path(data["root"])
    for cell in data["cfg"]["experiment"]["reference_cells"]:
        name = f"{cell}_seed{seed}"
        names[cell] = name
        ft.new_training(
            "alice",
            run_id,
            name,
            {
                "purpose": "readonly S5.E2 prediction reuse, not retrained",
                "cell": cell,
                "seed": seed,
                "reference_step": "S5.E2",
            },
        )
        for split in SPLITS:
            source = root / "input/references" / f"{name}_{split}.npy"
            np.save(
                root / "trainings" / name / "results" / (split + "_predictions.npy"),
                np.load(source),
            )
        save_json(
            root / "trainings" / name / "results/reuse.json",
            {"readonly_reference": True, "not_retrained": True},
        )
    return names


def finalize(data, records):
    cfg = data["cfg"]
    root = Path(data["root"])
    for sid, rows in records.items():
        for cell in cfg["experiment"]["fusion_cells"]:
            name = f"F{cell}_seed{sid}"
            ft.new_training(
                "alice",
                root.name,
                name,
                {
                    "purpose": "fixed probability-space fusion, no fit, no distillation",
                    "sources": [cell, "L0_LR_" + cell[-1]],
                    "weight": cfg["experiment"]["fusion_weight"],
                    "seed": int(sid),
                },
            )
            for split in SPLITS:
                a = np.load(
                    root
                    / "trainings"
                    / rows[cell]
                    / "results"
                    / (split + "_predictions.npy")
                )
                b = np.load(
                    root
                    / "trainings"
                    / rows["L0_LR_" + cell[-1]]
                    / "results"
                    / (split + "_predictions.npy")
                )
                weight = cfg["experiment"]["fusion_weight"]
                np.save(
                    root
                    / "trainings"
                    / name
                    / "results"
                    / (split + "_predictions.npy"),
                    weight * a + (1 - weight) * b,
                )
            save_json(
                root / "trainings" / name / "results/fusion.json",
                {"derived_not_fit": True, "weight_fixed_before_run": True},
            )
            rows["F" + cell] = name
    result = nf.finalize(data, records)
    path = root / "results/特征工程结果分析.md"
    path.write_text(
        path.read_text().replace(
            "# C01 单变量非线性完整分区训练", "# C01/C02/C03/C08 多组固定因子与组合对照"
        )
        + "\n字段组保留是构造条件，跨方乘积仅在密态中计算，N为配对置乱负对照；F为固定概率融合，不是FedDF复现。原参照是只读预测复用，只有八新联合组进行了拟合。\n"
    )
    return result


def numerical_cross_accept(tolerance):
    import jax.numpy as jnp

    # Public synthetic known-answer, independently written expected feature/bias mapping.
    xa = jnp.array(SYNTHETIC_XA)
    xb = jnp.array(SYNTHETIC_XB)
    t = jnp.array([1, 2])
    expected = jnp.array(SYNTHETIC_EXPECTED)
    got = candidate_design(
        xa, xb, None, None, t, "L3_secure", False, -1, 0, 2, SYNTHETIC_CROSS_BOUND
    )
    return jnp.all(jnp.abs(got - expected) <= tolerance)


def train_full(runtime):
    import secretflow as sf
    import spu
    from secretflow.device import SPUCompilerNumReturnsPolicy
    from secretflow.ml.linear.linear_model import RegType
    from secretflow.ml.linear.ss_sgd.model import Penalty, Strategy, _batch_update_w
    from secretflow.utils.sigmoid import SigType

    cfg = yaml.safe_load(Path(runtime["config"]).read_text())
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
        da = alice(prepare)("alice", runtime, cfg)
        db = bob(prepare)("bob", runtime, cfg)
        a_meta, b_meta = sf.reveal([alice(metadata)(da), bob(metadata)(db)])
        a_shape = {s: [a_meta["rows"][s], a_meta["base_width"]] for s in SPLITS}
        b_shape = {s: [b_meta["rows"][s], b_meta["base_width"]] for s in SPLITS}
        if not bool(
            sf.reveal(secure(numerical_cross_accept)(shared["precision_tolerance"]))
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
            own_records = alice(local_references)(da, seed, run_id)
            records[str(seed)] = {
                cell: f"{cell}_seed{seed}"
                for cell in cfg["experiment"]["reference_cells"]
            }
            sf.wait(own_records)
            for cell, condition in cfg["experiment"]["conditions"].items():
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
                        xa = alice(feature_chunk)(
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
                        xb = bob(feature_chunk)(
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
                                candidate_design,
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
                    records[str(seed)][cell] = name
                    details = {
                        "purpose": "fixed C01/C02/C03 factor run",
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
                        initial_weights,
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
                                "full_partition_training": True,
                            }
                        ),
                        flush=True,
                    )
        sf.wait(alice(finalize)(da, records))
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
