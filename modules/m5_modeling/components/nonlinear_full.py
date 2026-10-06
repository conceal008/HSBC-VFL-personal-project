"""Owner-local nonlinear full-partition run; complete training stays in native secret devices."""

from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss
import functional_training as ft  # type: ignore[import-not-found]
import active_diagnostic as ad  # type: ignore[import-not-found]
import functional_metrics as fm  # type: ignore[import-not-found]
from functional_metrics import save_json  # type: ignore[import-not-found]
from functional_tls import (
    install_pinned_tls_adapter,
    reject_anonymous_client,
    reject_anonymous_spu,
)  # type: ignore[import-not-found]

PARTIES = ("alice", "bob")
UPLIFT_COLUMN_FACTOR = 3
SPLITS = ("train", "validation")


def select_records(frame, maximum, seed):
    """Public salt + own key; no label/treatment-dependent sampling."""
    rank = [
        hashlib.sha256(f"{seed}:{key}".encode()).digest() for key in frame["record_id"]
    ]
    positions = sorted(range(len(frame)), key=lambda i: rank[i])[:maximum]
    return frame.iloc[positions].reset_index(drop=True)


def transform_basis(train, frames, features, cfg):
    """Training quantiles and scale only; constants/duplicate knots give no columns."""
    basis = {split: [] for split in SPLITS}
    metadata = []
    for feature in features:
        if feature not in train:
            raise ValueError("Required owner feature missing")
        knots = np.unique(np.quantile(train[feature], cfg["experiment"]["quantiles"]))
        for knot in knots:
            values = {
                split: np.maximum(0, frame[feature].to_numpy(dtype=float) - knot)
                for split, frame in frames.items()
            }
            if values["train"].var() <= cfg["experiment"]["variance_floor"]:
                continue
            metadata.append({"feature": feature, "knot": float(knot)})
            for split in SPLITS:
                basis[split].append(values[split])
    if not metadata:
        return {
            split: np.zeros((len(frame), 0), dtype=np.float32)
            for split, frame in frames.items()
        }, metadata
    arrays = {split: np.column_stack(columns) for split, columns in basis.items()}
    scaler = StandardScaler().fit(arrays["train"])
    return {
        split: np.clip(
            scaler.transform(array),
            -cfg["shared"]["clip_features"],
            cfg["shared"]["clip_features"],
        ).astype(np.float32)
        for split, array in arrays.items()
    }, metadata


def prepare(party, runtime, cfg):
    if cfg["experiment"]["max_rows"] != {"train": None, "validation": None}:
        raise ValueError("Full training forbids row caps")
    root = ft.ROOT / party / runtime["run_id"]
    checks = ft.isolation_preflight(party)
    checks["no_test_input"] = not (root / "input/test.csv").exists()
    old = [
        p
        for p in (ft.ROOT / party).iterdir()
        if p.is_dir() and p.name not in ("tls", runtime["run_id"])
    ]
    import os

    checks["old_rounds_denied"] = all(not os.access(p, os.R_OK | os.X_OK) for p in old)
    if not all(checks.values()):
        raise ValueError("Test or history capability not locked")
    ad.manifest_check(root)
    frames = {split: pd.read_csv(root / "input" / f"{split}.csv") for split in SPLITS}
    spec = cfg["datasets"][runtime["dataset"]]
    features = ft.validate_frames(frames, party, spec)
    original_counts = {split: len(frame) for split, frame in frames.items()}
    frames = {
        split: select_records(
            frame,
            cfg["experiment"]["max_rows"][split],
            cfg["experiment"]["selection_seed"],
        )
        for split, frame in frames.items()
    }
    if any(len(frames[split]) != count for split, count in original_counts.items()):
        raise ValueError("Full partition was truncated")
    save_json(
        root / "logs/full_partition.json",
        {
            "all_rows_retained": True,
            "source_rows": original_counts,
            "used_rows": {s: len(f) for s, f in frames.items()},
            "test_included": False,
        },
    )
    scaler = StandardScaler().fit(frames["train"][features].to_numpy(dtype=float))
    own_features = (
        cfg["experiment"]["features"][runtime["dataset"]] if party == "alice" else []
    )
    additions, knots = transform_basis(frames["train"], frames, own_features, cfg)
    data = {
        "root": str(root),
        "party": party,
        "cfg": cfg,
        "spec": spec,
        "width": len(features),
        "extra_width": len(knots),
        "shared": cfg["shared"],
    }
    for split, frame in frames.items():
        x = np.clip(
            scaler.transform(frame[features].to_numpy(dtype=float)),
            -cfg["shared"]["clip_features"],
            cfg["shared"]["clip_features"],
        ).astype(np.float32)
        ids = np.stack(
            [
                np.frombuffer(hashlib.sha256(str(key).encode()).digest(), dtype="<u4")
                for key in frame["record_id"]
            ]
        )
        row = {"x": x, "enhanced_x": np.c_[x, additions[split]], "ids": ids}
        if party == "alice":
            row.update(
                y=frame["label"].to_numpy(dtype=np.float32),
                t=frame["treatment"].to_numpy(dtype=np.int32)
                if spec["treatment"]
                else np.zeros(len(frame), dtype=np.int32),
            )
        data[split] = row
        np.savez_compressed(root / "data" / f"{split}_local.npz", **row)
        frame.to_csv(root / "data" / f"{split}_selected.csv", index=False)
    if party == "alice" and any(
        len(np.unique(data[split]["y"])) != 2 for split in SPLITS
    ):
        raise ValueError(
            "Full training lacks label support; do not resample based on labels"
        )
    save_json(
        root / "data/preprocessing.json",
        {
            "new_fit_split": "full_train",
            "inherited_fit_pool": "M1 original train vocabulary and imputation",
            "knots": knots,
            "base_mean": scaler.mean_.tolist(),
            "base_scale": scaler.scale_.tolist(),
        },
    )
    save_json(root / "logs/isolation.json", checks)
    save_json(
        root / "logs/test_capability.json",
        {
            "old_test_not_staged": True,
            "history_directories_denied": True,
            "selection_uses_labels": False,
        },
    )
    return data


def metadata(data):
    return {
        "base_width": data["width"],
        "extra_width": data["extra_width"],
        "rows": {split: len(data[split]["x"]) for split in SPLITS},
    }


def feature_chunk(data, split, begin, end, seed, enhanced, include):
    if split not in SPLITS:
        raise ValueError("Test capability denied")
    key = "enhanced_x" if enhanced else "x"
    x = ft.get_chunk(data, split, key, begin, end, seed)
    return x if include else np.zeros((len(x), 0), dtype=np.float32)


def initial_weights(a, b, extra, uplift, seed, scale, include_bob):
    """Same base coordinates; new basis coefficients start at zero in all cells."""
    old_width = a + b
    new_width = old_width + extra
    old_dimension = (
        old_width * UPLIFT_COLUMN_FACTOR + UPLIFT_COLUMN_FACTOR
        if uplift
        else old_width + 1
    )  # 魔数豁免: 固定两处理组交互与偏置列契约
    new_dimension = (
        new_width * UPLIFT_COLUMN_FACTOR + UPLIFT_COLUMN_FACTOR
        if uplift
        else new_width + 1
    )  # 魔数豁免: 固定两处理组交互与偏置列契约
    base = ft.initial_weights(old_dimension, seed, scale)
    weights = np.zeros((new_dimension, 1), dtype=np.float32)
    old_main = list(range(old_width))
    new_main = list(range(a)) + list(range(a + extra, new_width))
    weights[new_main] = base[old_main]
    if uplift:
        weights[new_width : new_width + 2] = base[old_width : old_width + 2]
        for old, new in zip(old_main, new_main):
            weights[new_width + 2 + new * 2 : new_width + 2 + new * 2 + 2] = base[
                old_width + 2 + old * 2 : old_width + 2 + old * 2 + 2
            ]
    weights[-1] = base[-1]
    if include_bob:
        return weights
    indices, _ = ad.mapped_indices(a + extra, b, uplift)
    return weights[indices]


def record(data, name, predictions):
    folder = Path(data["root"]) / "trainings" / name / "results"
    for split, prediction in predictions.items():
        p = np.asarray(prediction)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Invalid probability")
        np.save(folder / f"{split}_predictions.npy", p)
    epsilon = data["shared"]["score_epsilon"]
    p = np.asarray(predictions["train"])
    boundary = float(np.mean((p <= epsilon) | (p >= 1 - epsilon)))
    if boundary > data["shared"]["training_boundary_limit"]:
        raise ValueError("Predeclared training boundary failed")
    save_json(
        folder / "status.json",
        {
            "training": "passed",
            "test_read": False,
            "probabilities_valid": True,
            "boundary_fraction": boundary,
        },
    )
    return True


def local_references(data, seed, run_id):
    cfg = data["cfg"]
    shared = cfg["shared"]
    names = {}
    for enhanced in (False, True):
        key = "enhanced_x" if enhanced else "x"
        for family in ("LR", "tree"):
            cell = f"L0_{family}_{int(enhanced)}"
            name = f"{cell}_seed{seed}"
            names[cell] = name
            ft.new_training(
                "alice",
                run_id,
                name,
                {
                    "purpose": "fixed local family reference",
                    "cell": cell,
                    "seed": seed,
                    "shared": shared,
                    "enhanced": enhanced,
                },
            )
            model = (
                LogisticRegression(
                    C=cfg["experiment"]["local_lr_c"],
                    max_iter=shared["l0_max_iter"],
                    random_state=seed,
                )
                if family == "LR"
                else HistGradientBoostingClassifier(
                    max_iter=shared["tree_iterations"],
                    max_leaf_nodes=shared["tree_leaves"],
                    l2_regularization=shared["tree_l2"],
                    random_state=seed,
                )
            )
            tr = data["train"]
            model.fit(
                ft.local_design(tr[key], tr["t"], data["spec"]["treatment"]), tr["y"]
            )
            predictions = {}
            for split in SPLITS:
                row = data[split]
                arms = (
                    [0, 1, 2]
                    if data["spec"]["treatment"] and split != "train"
                    else [-1]
                )
                predictions[split] = np.column_stack(
                    [
                        model.predict_proba(
                            ft.local_design(
                                row[key], row["t"], data["spec"]["treatment"], arm
                            )
                        )[:, 1]
                        for arm in arms
                    ]
                )
            record(data, name, predictions)
    return names


def metric_values(y, prediction, spec, shared, treatment):
    metrics = fm.values(
        y, prediction, spec, shared, treatment if spec["treatment"] else None
    )
    factual = prediction[
        np.arange(len(y)),
        treatment if spec["treatment"] else np.zeros(len(y), dtype=int),
    ]
    metrics.update(
        log_loss=float(log_loss(y, factual, labels=[0, 1])),
        brier=float(brier_score_loss(y, factual)),
    )
    if spec["treatment"]:
        metrics["mean_centered_qini"] = float(
            np.mean(
                [metrics[f"arm{arm}_centered_qini"] for arm in spec["treatment_arms"]]
            )
        )
    return metrics


def exact_seed_p(differences):
    """Two-sided sign-flip test of fitting randomness; symmetry is an assumption."""
    d = np.asarray(differences)
    observed = abs(d.mean())
    possible = [
        abs(np.mean(d * np.asarray(signs)))
        for signs in itertools.product((-1, 1), repeat=len(d))
    ]
    return float(np.mean(np.asarray(possible) >= observed))


def holm_adjust(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    result = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        running = max(running, (len(ordered) - rank) * pvalues[key])
        result[key] = min(1.0, running)
    return result


def finalize(data, records):
    root = Path(data["root"])
    cfg = data["cfg"]
    shared = data["shared"]
    spec = data["spec"]
    seeds = shared["model_seeds"]
    cells = list(records[str(seeds[0])])
    v = data["validation"]
    y = v["y"]
    t = v["t"]
    predictions = {
        str(seed): {
            cell: np.load(
                root
                / "trainings"
                / records[str(seed)][cell]
                / "results/validation_predictions.npy"
            )
            for cell in cells
        }
        for seed in seeds
    }
    points = {
        str(seed): {
            cell: metric_values(y, predictions[str(seed)][cell], spec, shared, t)
            for cell in cells
        }
        for seed in seeds
    }
    metric_names = list(points[str(seeds[0])][cells[0]])
    seed_draws = {
        str(seed): {cell: {metric: [] for metric in metric_names} for cell in cells}
        for seed in seeds
    }
    draws = {cell: {metric: [] for metric in metric_names} for cell in cells}
    contrasts = cfg["experiment"]["comparisons"]
    contrast_draws = {
        name: {metric: [] for metric in metric_names} for name in contrasts
    }
    strata = t * 2 + y.astype(int) if spec["treatment"] else y.astype(int)
    groups = [np.flatnonzero(strata == k) for k in np.unique(strata)]
    rng = np.random.default_rng(cfg["experiment"]["bootstrap_seed"])
    for _ in range(shared["bootstrap_repeats"]):
        idx = np.concatenate(
            [rng.choice(group, len(group), replace=True) for group in groups]
        )
        rng.shuffle(idx)
        selected = rng.integers(len(seeds), size=len(seeds))
        current = {}
        for seed in seeds:
            sid = str(seed)
            current[sid] = {}
            for cell in cells:
                metrics = metric_values(
                    y[idx], predictions[sid][cell][idx], spec, shared, t[idx]
                )
                current[sid][cell] = metrics
                for metric in metric_names:
                    seed_draws[sid][cell][metric].append(metrics[metric])
        averages = {
            cell: {
                metric: float(
                    np.mean([current[str(seeds[i])][cell][metric] for i in selected])
                )
                for metric in metric_names
            }
            for cell in cells
        }
        for cell in cells:
            for metric, value in averages[cell].items():
                draws[cell][metric].append(value)
        for name, coefficients in contrasts.items():
            for metric in metric_names:
                contrast_draws[name][metric].append(
                    sum(
                        coef * averages[cell][metric]
                        for cell, coef in coefficients.items()
                    )
                )

    def interval(estimate, values):
        values = np.asarray(values)
        if not np.isfinite(values).all() or not np.isfinite(estimate):
            raise ValueError("Invalid interval; no silent draw filtering")
        low, high = np.percentile(values, shared["ci_percentiles"])
        return {
            "estimate": float(estimate),
            "ci_low": float(low),
            "ci_high": float(high),
            "valid_replicates": len(values),
        }

    aggregate = {
        cell: {
            metric: interval(
                np.mean([points[str(seed)][cell][metric] for seed in seeds]),
                draws[cell][metric],
            )
            for metric in metric_names
        }
        for cell in cells
    }
    comparison = {
        name: {
            metric: interval(
                np.mean(
                    [
                        sum(
                            coef * points[str(seed)][cell][metric]
                            for cell, coef in coeff.items()
                        )
                        for seed in seeds
                    ]
                ),
                contrast_draws[name][metric],
            )
            for metric in metric_names
        }
        for name, coeff in contrasts.items()
    }
    primary = "mean_centered_qini" if spec["treatment"] else "roc_auc"
    pvalues = {
        name: exact_seed_p(
            [
                sum(
                    coef * points[str(seed)][cell][primary]
                    for cell, coef in coeff.items()
                )
                for seed in seeds
            ]
        )
        for name, coeff in contrasts.items()
    }
    corrected = holm_adjust(pvalues)
    for seed in seeds:
        for cell in cells:
            report = {
                metric: interval(
                    points[str(seed)][cell][metric], seed_draws[str(seed)][cell][metric]
                )
                for metric in metric_names
            }
            save_json(
                root
                / "trainings"
                / records[str(seed)][cell]
                / "results/evaluation.json",
                {
                    "split": "validation_development",
                    "metrics": report,
                    "primary": primary,
                    "test_read": False,
                },
            )
    report = {
        "status": "completed",
        "scope": cfg["experiment"]["scope"],
        "primary": primary,
        "model_seeds": seeds,
        "models": aggregate,
        "paired_comparisons": comparison,
        "seed_points": points,
        "pvalues": pvalues,
        "holm_pvalues": corrected,
        "pvalue_scope": "two-sided sign flips across five fitting seeds, assuming symmetry; not a population test; minimum resolution prevents p<0.05 with five seeds",
        "bootstrap": "same stratified row draws for all models plus seed resampling; strata label and treatment; conditional on the frozen training/validation partitions",
        "full_data_training": True,
        "independent_confirmation": False,
        "records": records,
    }
    save_json(root / "results/private_feature_evaluation.json", report)
    lines = [
        "# C01 单变量非线性完整分区训练",
        "",
        "完整冻结训练/验证分区对照，不能解释为独立确认或银行收益。",
        "",
        f"主指标：{primary}。参数未按结果重选。",
        "",
        "| 模型 | 均值 | 95% CI |",
        "|---|---|---|",
    ]
    for cell, metrics in aggregate.items():
        v = metrics[primary]
        lines.append(
            f"| {cell} | {v['estimate']:.3f} | [{v['ci_low']:.3f}, {v['ci_high']:.3f}] |"
        )
    lines += [
        "",
        "| 比较 | 差值 | 95% CI | Holm p（训练seed条件） |",
        "|---|---|---|---|",
    ]
    for name, metrics in comparison.items():
        v = metrics[primary]
        lines.append(
            f"| {name} | {v['estimate']:.3f} | [{v['ci_low']:.3f}, {v['ci_high']:.3f}] | {corrected[name]:.3f} |"
        )
    lines += [
        "",
        "CI包含验证行与训练随机性；五seed共享冻结完整分区，不是五个人群。p值是训练随机性符号翻转的条件检验，假定对称，不能替代业务人群检验。当前没有独立确认集或业务最小效应量，不给生产上线/营销真实收益结论。L0为固定参数家族参照，L1新响应头尚未跑。",
        "每个拟合的验证CI见对应results/evaluation.json；抽样/清洗、源码、配置和中间矩阵分别留本轮data/与trainings/。全部正负结果保留。",
    ]
    (root / "results/特征工程结果分析.md").write_text("\n".join(lines) + "\n")
    return {"status": "completed", "test_read": False}


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
        records = {}
        for seed in shared["model_seeds"]:
            order_seed = cfg["experiment"]["order_seeds"][str(seed)]
            own_records = alice(local_references)(da, seed, run_id)
            records[str(seed)] = {
                cell: f"L0_{family}_{int(enhanced)}_seed{seed}"
                for family in ("LR", "tree")
                for enhanced in (False, True)
                for cell in [f"L0_{family}_{int(enhanced)}"]
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
                            da, split, begin, end, shuffle, condition["enhanced"], True
                        ).to(secure)
                        xb = bob(feature_chunk)(
                            db,
                            split,
                            begin,
                            end,
                            shuffle,
                            False,
                            condition["include_bob"],
                        ).to(secure)
                        t = alice(ad.chunk)(da, split, "t", begin, end, shuffle).to(
                            secure
                        )
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
                        "purpose": "C01 owner-local nonlinear full-partition development run",
                        "cell": cell,
                        "initialization_seed": seed,
                        "order_seed": order_seed,
                        "selection_seed": cfg["experiment"]["selection_seed"],
                        "learning_rate": lr,
                        "shared": shared,
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
                        ),
                    )(
                        a=a_meta["base_width"],
                        b=b_meta["base_width"],
                        extra=a_meta["extra_width"] if condition["enhanced"] else 0,
                        uplift=spec["treatment"],
                        seed=seed,
                        scale=shared["initialization_std"],
                        include_bob=condition["include_bob"],
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
