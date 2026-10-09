"""Actual authorized scores; owner-local stress tests and secret attribute probes."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
import functional_training as ft  # type: ignore[import-not-found]
import bounded_psi as bp  # type: ignore[import-not-found]

SPLITS = ("train", "validation")
CELLS = ("A0", "A1", "B0", "B1")
VIEWS = ("exact", "rounded", "binary")
PARTIES = ("alice", "bob")
HALF = 0.5
MINIMUM_SEEDS = 5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def view(scores, name, cfg):
    p = np.asarray(scores)
    if name == "exact":
        return p
    if name == "rounded":
        return np.round(p, cfg["round_decimals"])
    if name == "binary":
        return (p >= cfg["decision_threshold"]).astype(float)
    raise ValueError("Unregistered output view")


def interval(values, cfg, seed):
    a = np.asarray(values, dtype=float)
    if len(a) < MINIMUM_SEEDS or not np.isfinite(a).all():
        raise ValueError("Incomplete seed interval")
    rng = np.random.default_rng(seed)
    draws = rng.choice(a, (cfg["bootstrap_repeats"], len(a)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, cfg["ci_percentiles"])
    return {
        "estimate": float(a.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "scope": "attack/model seed conditional interval; five seeds share one data source",
    }


def split_indices(n, seed, maximum):
    idx = np.random.default_rng(seed).permutation(n)[:maximum]
    middle = len(idx) // 2
    if middle == 0:
        raise ValueError("Empty attack split")
    return idx[:middle], idx[middle:]  # 魔数豁免: 等分攻击辅助与评估分区


def loss_signal(y, p, epsilon):
    p = np.clip(p, epsilon, 1 - epsilon)
    return y * np.log(p) + (1 - y) * np.log(1 - p)


def attack_metrics(labels, signal, calibration, evaluation, cfg):
    y, s = np.asarray(labels), np.asarray(signal)
    direction = 1 if roc_auc_score(y[calibration], s[calibration]) >= HALF else -1
    candidates = np.quantile(direction * s[calibration], cfg["threshold_quantiles"])
    accuracies = [
        balanced_accuracy_score(y[calibration], direction * s[calibration] >= q)
        for q in candidates
    ]
    threshold = candidates[int(np.argmax(accuracies))]
    predicted = direction * s[evaluation]
    fpr, tpr, _ = roc_curve(y[evaluation], predicted)
    return {
        "auc": float(roc_auc_score(y[evaluation], predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y[evaluation], predicted >= threshold)
        ),
        "tpr_at_fixed_fpr": float(np.max(tpr[fpr <= cfg["maximum_fpr"]])),
        "direction_fit_only_on_calibration": True,
    }


def owner_inputs(party, runtime, cfg):
    root = ft.ROOT / party / runtime["run_id"]
    ft.isolation_preflight(party)
    manifest = json.loads((root / "input/manifest.json").read_text())
    if any("test" in n or sha(root / "input" / n) != h for n, h in manifest.items()):
        raise ValueError("Input capability/fingerprint failed")
    rows = {}
    for split in SPLITS:
        with np.load(root / "input" / f"{split}_local.npz", allow_pickle=False) as z:
            rows[split] = {k: z[k] for k in z.files}
        expected = (
            {"x", "enhanced_x", "ids", "y", "t"}
            if party == "alice"
            else {"x", "enhanced_x", "ids"}
        )
        if set(rows[split]) != expected:
            raise ValueError("Owner schema failed")
    if party == "bob":
        names = [
            n
            for n in pd.read_csv(root / "input/train.csv", nrows=0).columns
            if n != "record_id"
        ]
        target = cfg["attribute_targets"][runtime["dataset"]]
        rows["target_index"] = names.index(target)
    else:
        rows["predictions"] = {
            str(seed): {
                cell: {
                    s: np.load(
                        root / "input" / f"{cell}_seed{seed}_{s}.npy",
                        allow_pickle=False,
                    )
                    for s in SPLITS
                }
                for cell in CELLS
            }
            for seed in cfg["model_seeds"]
        }
    rows["root"] = str(root)
    return rows


def matched_pools(train, validation, seed, maximum):
    rng = np.random.default_rng(seed)
    ta = train["t"] * 2 + train["y"].astype(int)
    tb = validation["t"] * 2 + validation["y"].astype(int)
    aa, bb = [], []
    for label in np.union1d(ta, tb):
        a = rng.permutation(np.flatnonzero(ta == label))
        b = rng.permutation(np.flatnonzero(tb == label))
        n = min(len(a), len(b))
        aa.extend(a[:n])
        bb.extend(b[:n])
    positions = rng.permutation(len(aa))[:maximum]
    return np.asarray(aa)[positions], np.asarray(bb)[positions]


def local_stress(data, cfg):
    root = Path(data["root"])
    details = []
    for seed, attack_seed in zip(cfg["model_seeds"], cfg["attack_seeds"]):
        tr, va = data["train"], data["validation"]
        a, b = matched_pools(tr, va, attack_seed, cfg["maximum_rows"])
        count = len(a)
        y = np.r_[tr["y"][a], va["y"][b]]
        membership = np.r_[np.ones(count), np.zeros(count)]
        # Each class is split independently, preventing a missing membership class.
        cal, ev = split_indices(count, attack_seed, count)
        calibration, evaluation = np.r_[cal, cal + count], np.r_[ev, ev + count]
        x = np.c_[np.r_[tr["x"][a], va["x"][b]], y, np.r_[tr["t"][a], va["t"][b]]]
        feature_model = LogisticRegression(
            C=cfg["feature_c"], max_iter=cfg["max_iter"], random_state=attack_seed
        ).fit(x[calibration], membership[calibration])
        feature_signal = feature_model.predict_proba(x)[:, 1]
        feature_reference = attack_metrics(
            membership, feature_signal, calibration, evaluation, cfg
        )
        controls = {
            "positive": attack_metrics(
                membership, membership, calibration, evaluation, cfg
            ),
            "negative": attack_metrics(
                membership,
                np.random.default_rng(attack_seed).normal(size=len(membership)),
                calibration,
                evaluation,
                cfg,
            ),
        }
        if controls["positive"]["auc"] != 1:
            raise ValueError("Attacker positive control failed")
        for cell in CELLS:
            predicted = data["predictions"][str(seed)][cell]
            factual = (
                predicted["validation"][np.arange(len(va["y"])), va["t"]]
                if predicted["validation"].shape[1] > 1
                else predicted["validation"][:, 0]
            )
            scores = np.r_[predicted["train"][a, 0], factual[b]]
            for output_view in VIEWS:
                restricted = view(scores, output_view, cfg)
                signal = loss_signal(y, restricted, cfg["epsilon"])
                m = attack_metrics(membership, signal, calibration, evaluation, cfg)
                # A signal-only score model and feature+score model calibrate on disjoint records.
                combined = np.c_[x, signal, y]
                reference = LogisticRegression(
                    C=cfg["feature_c"],
                    max_iter=cfg["max_iter"],
                    random_state=attack_seed,
                ).fit(combined[calibration], membership[calibration])
                combined_metrics = attack_metrics(
                    membership,
                    reference.predict_proba(combined)[:, 1],
                    calibration,
                    evaluation,
                    cfg,
                )
                estimator = Ridge(alpha=cfg["ridge"]).fit(
                    tr["x"], view(predicted["train"], output_view, cfg)
                )
                surrogate = estimator.predict(va["x"]).reshape(-1, 1)
                target = view(factual[:, None], output_view, cfg)
                mse = float(np.mean((surrogate - target) ** 2))
                top = max(1, int(len(target) * cfg["top_fraction"]))
                overlap = (
                    len(
                        set(np.argsort(surrogate[:, 0])[-top:])
                        & set(np.argsort(target[:, 0])[-top:])
                    )
                    / top
                )
                name = f"{cell}_{output_view}_seed{seed}"
                ft.new_training(
                    "alice",
                    root.name,
                    name,
                    {
                        "purpose": "actual score stress, counterfactual external access",
                        "model_seed": seed,
                        "attack_seed": attack_seed,
                        "source_cell": cell,
                        "view": output_view,
                        "config": cfg,
                    },
                )
                item = {
                    "cell": cell,
                    "seed": seed,
                    "attack_seed": attack_seed,
                    "view": output_view,
                    "membership": m,
                    "feature_membership": feature_reference,
                    "feature_score_membership": combined_metrics,
                    "controls": controls,
                    "surrogate_mse": mse,
                    "top_overlap": overlap,
                }
                save(root / "trainings" / name / "results/attack.json", item)
                details.append(item)
    aggregates = {}
    for cell in CELLS:
        for output_view in VIEWS:
            selected = [
                d for d in details if d["cell"] == cell and d["view"] == output_view
            ]
            aggregates[f"{cell}_{output_view}"] = {
                metric: interval(
                    [d["membership"][metric] for d in selected],
                    cfg,
                    cfg["bootstrap_seed"],
                )
                for metric in ("auc", "balanced_accuracy", "tpr_at_fixed_fpr")
            }
            aggregates[f"{cell}_{output_view}"].update(
                {
                    "membership_minus_feature_auc": interval(
                        [
                            d["membership"]["auc"] - d["feature_membership"]["auc"]
                            for d in selected
                        ],
                        cfg,
                        cfg["bootstrap_seed"],
                    ),
                    "feature_score_minus_feature_auc": interval(
                        [
                            d["feature_score_membership"]["auc"]
                            - d["feature_membership"]["auc"]
                            for d in selected
                        ],
                        cfg,
                        cfg["bootstrap_seed"],
                    ),
                    "surrogate_mse": interval(
                        [d["surrogate_mse"] for d in selected],
                        cfg,
                        cfg["bootstrap_seed"],
                    ),
                    "top_overlap": interval(
                        [d["top_overlap"] for d in selected], cfg, cfg["bootstrap_seed"]
                    ),
                }
            )
    report = {
        "status": "completed",
        "details": details,
        "aggregate": aggregates,
        "bob_score_access": False,
        "threat_scope": "counterfactual score disclosure; Alice already knows own labels/membership; label/treatment-matched pools; feature/time drift can still confound attacks",
        "ci_scope": "conditional seed bootstrap; not independent population confirmation",
    }
    save(root / "results/private_output_audit.json", report)
    return True


def probe_inputs(data, cell, seed, attack_seed, output_view, cfg):
    row = data["validation"]
    cal, ev = split_indices(len(row["x"]), attack_seed, cfg["maximum_rows"])
    if "predictions" not in data:
        target = row["x"][:, data["target_index"]][:, None]
        return target[cal], target[ev]
    score = data["predictions"][str(seed)][cell]["validation"]
    if output_view != "baseline":
        score = view(score, output_view, cfg)
    base = np.c_[row["x"], np.ones(len(score))]
    augmented = (
        base
        if output_view == "baseline"
        else np.c_[row["x"], score, np.ones(len(score))]
    )
    return augmented[cal].astype(np.float32), augmented[ev].astype(np.float32)


def secret_probe(x, xv, y, yv, ridge, inverse_steps, inverse_tolerance):
    import jax.numpy as jnp

    g = (x.T @ x) / len(x) + ridge * jnp.eye(x.shape[1])
    inv = jnp.eye(x.shape[1]) / jnp.trace(g)
    identity = jnp.eye(x.shape[1])
    for _ in range(inverse_steps):
        inv = inv @ (2 * identity - g @ inv)
    w = inv @ ((x.T @ y) / len(x))
    mse = jnp.mean((xv @ w - yv) ** 2)
    reference = jnp.mean((yv - jnp.mean(y)) ** 2)
    return jnp.stack(
        [
            mse,
            reference,
            1 - mse / jnp.maximum(reference, np.finfo(np.float32).eps),
            (jnp.max(jnp.abs(g @ inv - identity)) <= inverse_tolerance).astype(x.dtype),
        ]
    )


def synthetic_expected(cfg):
    x = np.asarray(cfg["synthetic_probe"]["x"], dtype=np.float32)
    y = np.asarray(cfg["synthetic_probe"]["y"], dtype=np.float32)
    g = x.T @ x / len(x) + cfg["ridge"] * np.eye(x.shape[1])
    w = np.linalg.solve(g, x.T @ y / len(x))
    return float(np.mean((x @ w - y) ** 2))


def synthetic_accept(result, expected, tolerance):
    import jax.numpy as jnp

    return (jnp.abs(result[0] - expected) <= tolerance) & (result[-1] == 1)


def run(runtime):
    cfg = yaml.safe_load(Path(runtime["config"]).read_text())
    root = ft.ROOT / runtime["party"] / runtime["run_id"]
    if (
        len(cfg["model_seeds"]) < MINIMUM_SEEDS
        or sha(runtime["config"]) != runtime["config_sha256"]
        or any(sha(root / "code" / n) != h for n, h in runtime["code_sha256"].items())
    ):
        raise ValueError("Frozen protocol fingerprint failed")
    with bp.session(runtime, cfg) as (sf, secure):
        try:
            alice, bob = sf.PYU("alice"), sf.PYU("bob")
            da, db = (
                alice(owner_inputs)("alice", runtime, cfg),
                bob(owner_inputs)("bob", runtime, cfg),
            )
            for split in SPLITS:
                rows = int(sf.reveal(alice(lambda d, s: len(d[s]["ids"]))(da, split)))
                valid = secure(lambda: np.array(True))()
                for begin in range(0, rows, cfg["alignment_chunk_rows"]):
                    end = min(rows, begin + cfg["alignment_chunk_rows"])
                    ax = alice(ft.get_chunk)(da, split, "ids", begin, end, None).to(
                        secure
                    )
                    bx = bob(ft.get_chunk)(db, split, "ids", begin, end, None).to(
                        secure
                    )
                    chunk_ok = secure(ft.secure_equal)(ax, bx)
                    valid = secure(lambda a, b: a & b)(valid, chunk_ok)
                    sf.wait(valid)
                if not bool(sf.reveal(valid)):
                    raise ValueError("Attribute probe row mismatch")
            sample = cfg["synthetic_probe"]
            sx = alice(lambda a: np.asarray(a, dtype=np.float32))(sample["x"]).to(
                secure
            )
            sy = bob(lambda a: np.asarray(a, dtype=np.float32))(sample["y"]).to(secure)
            check = secure(secret_probe, static_argnames="inverse_steps")(
                sx,
                sx,
                sy,
                sy,
                ridge=cfg["ridge"],
                inverse_steps=cfg["inverse_steps"],
                inverse_tolerance=cfg["inverse_tolerance"],
            )
            if not bool(
                sf.reveal(
                    secure(synthetic_accept)(
                        check, synthetic_expected(cfg), cfg["synthetic_tolerance"]
                    )
                )
            ):
                raise ValueError("Synthetic MPC numerical control failed")
            sf.wait(alice(local_stress)(da, cfg))
            for seed, attack_seed in zip(cfg["model_seeds"], cfg["attack_seeds"]):
                for cell in ("B0", "B1"):
                    for output_view in ("baseline", "exact", "rounded", "binary"):
                        x, xv = alice(probe_inputs, num_returns=2)(
                            da, cell, seed, attack_seed, output_view, cfg
                        )
                        y, yv = bob(probe_inputs, num_returns=2)(
                            db, cell, seed, attack_seed, output_view, cfg
                        )
                        name = f"attribute_{cell}_{output_view}_seed{seed}"
                        sf.wait(
                            [
                                alice(ft.new_training)(
                                    "alice",
                                    runtime["run_id"],
                                    name,
                                    {
                                        "purpose": "auxiliary-learning oracle attribute stress",
                                        "config": cfg,
                                    },
                                ),
                                bob(ft.new_training)(
                                    "bob",
                                    runtime["run_id"],
                                    name,
                                    {"purpose": "secret truth only", "config": cfg},
                                ),
                            ]
                        )
                        result = secure(secret_probe, static_argnames="inverse_steps")(
                            x.to(secure),
                            xv.to(secure),
                            y.to(secure),
                            yv.to(secure),
                            ridge=cfg["ridge"],
                            inverse_steps=cfg["inverse_steps"],
                            inverse_tolerance=cfg["inverse_tolerance"],
                        ).to(alice)
                        sf.wait(
                            alice(store_probe)(
                                da, name, result, cell, seed, output_view
                            )
                        )
            sf.wait(alice(finish_probes)(da, cfg))
        except BaseException:
            import traceback

            (root / "logs/failure_traceback.log").write_text(traceback.format_exc())
            raise
    report = {
        "status": "passed",
        "actual_output_stress": True,
        "bob_scores_disclosed": False,
        "bob_attribute_truth_disclosed": False,
        "test_read": False,
        "production_security": False,
    }
    save(root / "results/engineering.json", report)
    return report


def store_probe(data, name, result, cell, seed, output_view):
    result = np.asarray(result)
    if not np.isfinite(result).all() or np.any(result[:2] < 0) or result[-1] != 1:
        raise ValueError("Secret numerical probe failed")
    item = {
        "cell": cell,
        "seed": seed,
        "view": output_view,
        "mse": float(result[0]),
        "reference_mse": float(result[1]),
        "relative_improvement": float(result[2]),
    }
    save(Path(data["root"]) / "trainings" / name / "results/attribute.json", item)
    return True


def finish_probes(data, cfg):
    root = Path(data["root"])
    details = [
        json.loads(p.read_text())
        for p in (root / "trainings").glob("attribute_*/results/attribute.json")
    ]
    summary = {}
    for cell in ("B0", "B1"):
        for output_view in VIEWS:
            diffs = []
            for seed in cfg["model_seeds"]:
                base = next(
                    d
                    for d in details
                    if d["cell"] == cell
                    and d["seed"] == seed
                    and d["view"] == "baseline"
                )
                current = next(
                    d
                    for d in details
                    if d["cell"] == cell
                    and d["seed"] == seed
                    and d["view"] == output_view
                )
                diffs.append(base["mse"] - current["mse"])
            summary[f"{cell}_{output_view}_mse_reduction"] = interval(
                diffs, cfg, cfg["bootstrap_seed"]
            )
    save(
        root / "results/private_attribute_audit.json",
        {
            "details": details,
            "paired": summary,
            "auxiliary_fit_and_eval_disjoint": True,
            "plaintext_truth_or_weights_disclosed": False,
            "threat_scope": "generous auxiliary-learning oracle in MPC; not current party Bob access or proof of no leakage",
        },
    )
    return True
