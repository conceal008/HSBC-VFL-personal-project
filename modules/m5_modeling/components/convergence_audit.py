"""Owner-only DF convergence audit; never accepts passive features or test data."""
from __future__ import annotations

import hashlib
import inspect
import itertools
import json
from pathlib import Path
import time

import numpy as np
import yaml
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import active_diagnostic as ad  # type: ignore[import-not-found]
import functional_training as ft  # type: ignore[import-not-found]
import nonlinear_full as nf  # type: ignore[import-not-found]

SPLITS = ("train", "validation")
CELLS = ("A0", "A1")
ENDPOINTS = (12, 48)
CHECKPOINTS = (1, 3, 6, 12, 24, 48)
MINIMUM_SEEDS = 5
HALF = 0.5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def validate(cfg):
    shared, audit = cfg["shared"], cfg["audit"]
    if (len(shared["model_seeds"]) < MINIMUM_SEEDS
            or len(set(shared["model_seeds"])) != len(shared["model_seeds"])
            or tuple(audit["checkpoints"]) != CHECKPOINTS
            or tuple(audit["endpoints"]) != ENDPOINTS
            or shared["epochs"] != max(CHECKPOINTS)
            or shared["learning_rates"] != [audit["frozen_learning_rate"]]
            or audit["prediction_tolerance"] != shared["precision_tolerance"]
            or cfg["experiment"]["max_rows"] != {"train": None, "validation": None}):
        raise ValueError("Unregistered schedule or incomplete seed protocol")
    if set(cfg["experiment"]["order_seeds"]) != set(map(str, shared["model_seeds"])):
        raise ValueError("Missing separate order seed")


def load_inputs(root):
    manifest = json.loads((root / "input/manifest.json").read_text())
    if any("test" in name.lower() or name.startswith("/") or ".." in Path(name).parts
           for name in manifest):
        raise ValueError("Input capability denied")
    for name, digest in manifest.items():
        if sha(root / "input" / name) != digest:
            raise ValueError("Frozen input changed")
    result = {}
    for split in SPLITS:
        path = root / "input" / f"{split}_local.npz"
        if path.name not in manifest:
            raise ValueError("Unregistered owner matrix")
        with np.load(path, allow_pickle=False) as archive:
            row = {key: archive[key] for key in archive.files}
        if set(row) != {"x", "enhanced_x", "y", "t", "ids"}:
            raise ValueError("Only Alice's frozen matrix schema is allowed")
        if any(not np.isfinite(row[key]).all() for key in ("x", "enhanced_x", "y", "t")):
            raise ValueError("Nonfinite owner input")
        result[split] = row
        with (root / "data" / path.name).open("xb") as output:
            np.savez(output, **row)
    return result


def diagnostics(design, y, weights, shared):
    """DF primitive, not logistic likelihood; full unpadded gradient is descriptive."""
    z = (design @ weights).reshape(-1).astype(np.float64)
    yy = np.asarray(y).reshape(-1)
    lam = shared["l2_norm"] / shared["batch_size"]
    regularized = weights.copy()
    regularized[-1] = 0
    primitive = HALF * z + HALF * (np.abs(z) - np.log1p(np.abs(z)))
    objective = np.mean(primitive - yy * z) + HALF * lam * np.square(regularized).sum()
    gradient = design.T @ (ad.numpy_df(z, "df") - yy)[:, None] / len(yy) + lam * regularized
    probability = ad.numpy_df(z, "df")
    return {"df_surrogate_objective": float(objective),
            "full_unpadded_gradient_norm": float(np.linalg.norm(gradient)),
            "effective_l2": float(lam),
            "log_loss": float(log_loss(yy, probability, labels=[0, 1])),
            "brier": float(brier_score_loss(yy, probability)),
            "roc_auc": float(roc_auc_score(yy, probability))}


def predictions(row, weights, enhanced, uplift, split):
    if split not in SPLITS:
        raise ValueError("Test capability denied")
    x = row["enhanced_x" if enhanced else "x"]
    potentials = [0, 1, 2] if uplift and split == "validation" else [-1]
    return np.stack([ad.numpy_df((ad.cpu_design(x, row["t"], uplift, p, 0)
                                 @ weights).reshape(-1), "df") for p in potentials], axis=1)


def trajectory(rows, shared, audit, order_seed, weights, uplift, enhanced, update, callback):
    """Same fixed order, zero tail padding and projection per infeed as the secure run."""
    row = rows["train"]
    order = np.random.default_rng(order_seed).permutation(len(row["y"]))
    x = row["enhanced_x" if enhanced else "x"][order]
    y, t = row["y"][order], row["t"][order]
    blocks = []
    for begin in range(0, len(x), shared["infeed_rows"]):
        end = min(len(x), begin + shared["infeed_rows"])
        pad = (-(end - begin)) % shared["batch_size"]
        design = ad.cpu_design(x[begin:end], t[begin:end], uplift, -1, pad)
        target = np.pad(y[begin:end].reshape(-1, 1), ((0, pad), (0, 0)))
        blocks.append((design, target))
    for epoch in range(1, shared["epochs"] + 1):
        for design, target in blocks:
            weights = update(design, target, weights, shared)
            weights = np.clip(weights, -shared["weight_bound"], shared["weight_bound"])
        if epoch in audit["checkpoints"]:
            callback(epoch, weights)
    return weights


def paired_intervals(row, endpoint_predictions, cfg, spec):
    """Paired resampling of rows within outcome/treatment strata and fitting seeds."""
    shared, audit = cfg["shared"], cfg["audit"]
    metric = "mean_centered_qini" if spec["treatment"] else "roc_auc"
    y, t = row["y"], row["t"]
    strata = y.astype(int) * (1 + int(t.max())) + t.astype(int)
    groups = [np.flatnonzero(strata == key) for key in np.unique(strata)]
    seeds = list(map(str, shared["model_seeds"]))
    def value(p, index):
        return nf.metric_values(y[index], p[index], spec, shared, t[index])[metric]
    complete = np.arange(len(y))
    points, differences = {}, {}
    for cell in CELLS:
        differences[cell] = [value(endpoint_predictions[cell][s]["48"], complete)
                             - value(endpoint_predictions[cell][s]["12"], complete) for s in seeds]
        points[cell] = {str(e): [value(endpoint_predictions[cell][s][str(e)], complete)
                                 for s in seeds] for e in ENDPOINTS}
    rng = np.random.default_rng(audit["bootstrap_seed"])
    samples = {cell: {"12": [], "48": [], "difference": []} for cell in CELLS}
    for _ in range(shared["bootstrap_repeats"]):
        index = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        rng.shuffle(index)
        chosen = rng.integers(0, len(seeds), size=len(seeds))
        for cell in CELLS:
            values = {str(e): [value(endpoint_predictions[cell][s][str(e)], index)
                               for s in seeds] for e in ENDPOINTS}
            for key in ("12", "48"):
                samples[cell][key].append(float(np.asarray(values[key])[chosen].mean()))
            samples[cell]["difference"].append(float(
                (np.asarray(values["48"]) - np.asarray(values["12"]))[chosen].mean()))
    pvalues = {c: nf.exact_seed_p(differences[c]) for c in CELLS}
    corrected = nf.holm_adjust(pvalues)
    result = {}
    for cell in CELLS:
        result[cell] = {"metric": metric, "per_seed_difference": differences[cell],
                        "difference": float(np.mean(differences[cell])),
                        "difference_ci": np.percentile(samples[cell]["difference"], shared["ci_percentiles"]).tolist(),
                        "seed_sign_flip_p": pvalues[cell], "holm_p": corrected[cell]}
        for key in ("12", "48"):
            result[cell][key] = {"mean": float(np.mean(points[cell][key])),
                                "ci": np.percentile(samples[cell][key], shared["ci_percentiles"]).tolist()}
    return result


def assert_reference(actual, reference, tolerance):
    if actual.shape != reference.shape or not np.isfinite(reference).all():
        raise ValueError("Reference shape or finiteness failed")
    error = float(np.max(np.abs(actual - reference)))
    if error > tolerance:
        raise ValueError("Native/secure reference tolerance failed")
    return error


def run(runtime, update=None):
    cfg = yaml.safe_load(Path(runtime["config"]).read_text())
    validate(cfg)
    root = Path(runtime["config"]).parent.parent
    if runtime["party"] != "alice" or sha(runtime["config"]) != runtime["config_sha256"]:
        raise ValueError("Owner or config fingerprint failed")
    for name, digest in runtime["code_sha256"].items():
        if sha(root / "code" / name) != digest:
            raise ValueError("Code fingerprint failed")
    rows = load_inputs(root)
    shared, spec = cfg["shared"], cfg["datasets"][runtime["dataset"]]
    native_metadata = {"test_binding": update is not None}
    if update is None:
        from secretflow.ml.linear.ss_sgd.model import _batch_update_w, Penalty, Strategy
        from secretflow.ml.linear.linear_model import RegType
        from secretflow.utils.sigmoid import SigType
        native = ad.cpu_cache_hint_bridge(_batch_update_w)
        native_metadata.update(native_source_sha256=hashlib.sha256(inspect.getsource(_batch_update_w).encode()).hexdigest(),
                               native_bytecode_sha256=hashlib.sha256(_batch_update_w.__code__.co_code).hexdigest())
        def update(x, y, w, s):
            return native(x, y, w, s["learning_rates"][0], s["l2_norm"],
                          SigType(s["sigmoid"]), RegType.Logistic, Penalty.L2,
                          len(x) // s["batch_size"], s["batch_size"], Strategy.NAIVE_SGD,
                          None, False)[0]
    endpoint_predictions = {c: {} for c in CELLS}
    curves, reference_errors, existing_joint = {}, [], {}
    for seed, cell in itertools.product(shared["model_seeds"], CELLS):
        enhanced = cell == "A1"
        name = f"{cell}_seed{seed}_epochs48"
        extra = rows["train"]["enhanced_x"].shape[1] - rows["train"]["x"].shape[1] if enhanced else 0
        weights = nf.initial_weights(rows["train"]["x"].shape[1], runtime["bob_base_width"], extra,
                                     spec["treatment"], seed, shared["initialization_std"], False)
        folder = root / "trainings" / name
        ft.new_training("alice", root.name, name, {"purpose": "fixed-budget owner convergence diagnostic",
                        "initialization_seed": seed, "order_seed": cfg["experiment"]["order_seeds"][str(seed)],
                        "cell": cell, "shared": shared, "audit": cfg["audit"]})
        (folder / "logs/实验日志.md").write_text("# 固定训练预算诊断\n没有选参；完整冻结train拟合，validation记录全部检查点，test不读取。代码/配置在code，数据引用在data；本方模型与预测仅Alice。\n")
        curves[name] = []
        endpoint_predictions[cell][str(seed)] = {}
        started = time.monotonic()
        def callback(epoch, w):
            pred = {s: predictions(rows[s], w, enhanced, spec["treatment"], s) for s in SPLITS}
            design = ad.cpu_design(rows["train"]["enhanced_x" if enhanced else "x"],
                                   rows["train"]["t"], spec["treatment"], -1, 0)
            diagnostic = diagnostics(design, rows["train"]["y"], w, shared)
            diagnostic.update(epoch=epoch, validation=nf.metric_values(rows["validation"]["y"],
                              pred["validation"], spec, shared, rows["validation"]["t"]))
            curves[name].append(diagnostic)
            if epoch in ENDPOINTS:
                endpoint_predictions[cell][str(seed)][str(epoch)] = pred["validation"]
                for split in SPLITS:
                    np.save(folder / "results" / f"prediction_{split}_epoch{epoch}.npy", pred[split])
                np.save(folder / "results" / f"own_weights_epoch{epoch}.npy", w)
            if epoch == ENDPOINTS[0]:
                for split in SPLITS:
                    old = np.load(root / "input" / f"{cell}_seed{seed}_prediction_{split}.npy")
                    reference_errors.append(assert_reference(pred[split], old, cfg["audit"]["prediction_tolerance"]))
            save(folder / "results/curve.json", curves[name])
            save(folder / "logs/progress.json", {"completed_epochs": epoch, "planned_epochs": shared["epochs"],
                 "elapsed_seconds": time.monotonic() - started})
        trajectory(rows, shared, cfg["audit"], cfg["experiment"]["order_seeds"][str(seed)], weights,
                   spec["treatment"], enhanced, update, callback)
        print(json.dumps({"completed_training": name}), flush=True)
    for seed, cell in itertools.product(shared["model_seeds"], ("B0", "B1")):
        item = {}
        for split in SPLITS:
            pred = np.load(root / "input" / f"{cell}_seed{seed}_prediction_{split}.npy")
            row = rows[split]
            if split == "train":
                factual = pred.reshape(-1)
                item[split] = {"roc_auc": float(roc_auc_score(row["y"], factual)),
                               "log_loss": float(log_loss(row["y"], factual, labels=[0, 1])),
                               "brier": float(brier_score_loss(row["y"], factual))}
            else:
                item[split] = nf.metric_values(row["y"], pred, spec, shared, row["t"])
        existing_joint[f"{cell}_seed{seed}"] = item
    result = {"status": "passed", "curves": curves, "native_reference": native_metadata,
              "maximum_reference_error": max(reference_errors),
              "paired": paired_intervals(rows["validation"], endpoint_predictions, cfg, spec),
              "existing_joint_descriptive": existing_joint,
              "scope": "owner diagnostic; no new joint training; no independent confirmation"}
    save(root / "results/private_convergence.json", result)
    save(root / "results/engineering.json", {"status": "passed", "trajectories_complete": True,
         "reference_tolerance_passed": True, "test_staged": False, "new_joint_training": False,
         "prediction_export": False, "config_sha256": runtime["config_sha256"]})
    return result
