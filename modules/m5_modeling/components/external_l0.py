"""Alice-only external-data baselines; never imports or reads Bob's features."""
from __future__ import annotations

import errno
import hashlib
import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

HASH_CHUNK_BYTES = 1024 * 1024
MAX_ITER = 2000
FIT_TOL = 1e-4
BOOTSTRAP_MIN_VALID_FRACTION = 0.9
CI_PERCENTILES = (2.5, 97.5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def denied(path: Path, mode: str) -> bool:
    try:
        with path.open(mode):
            pass
    except OSError as error:
        return error.errno in (errno.EACCES, errno.EPERM)
    return False


def check_isolation(runtime: dict) -> dict:
    probes = runtime["probes"]
    checks = {
        "peer_input_read_denied": denied(Path(probes["peer_input"]), "rb"),
        "peer_output_read_denied": denied(Path(probes["peer_output"]), "rb"),
        "full_source_read_denied": denied(Path(probes["full_source"]), "rb"),
        "own_input_write_denied": denied(Path(runtime["data_dir"]) / "train.csv", "r+b"),
    }
    try:
        with socket.create_connection(("127.0.0.1", int(probes["port"])), timeout=1):
            pass
    except OSError as error:
        checks["network_denied"] = error.errno in (errno.EACCES, errno.EPERM)
    else:
        checks["network_denied"] = False
    if not all(checks.values()):
        raise RuntimeError(f"Isolation preflight failed: {checks}")
    return checks


def load_splits(runtime: dict, spec: dict) -> tuple[dict, list[str], dict]:
    root = Path(runtime["data_dir"])
    expected = runtime["accepted_manifest"]
    frames = {}
    fingerprints = {}
    for split in ("train", "validation", "test"):
        path = root / f"{split}.csv"
        actual = sha256(path)
        if actual != expected[path.name]:
            raise ValueError(f"Frozen input hash mismatch: {split}")
        fingerprints[split] = actual
        frame = pd.read_csv(path)
        if frame["record_id"].isna().any() or not frame["record_id"].is_unique:
            raise ValueError(f"Invalid record key: {split}")
        frames[split] = frame
    ids = [set(frame["record_id"]) for frame in frames.values()]
    if any(ids[i] & ids[j] for i in range(len(ids)) for j in range(i + 1, len(ids))):
        raise ValueError("Frozen splits overlap")
    columns = list(frames["train"].columns)
    if any(list(frame.columns) != columns for frame in frames.values()):
        raise ValueError("Train/validation/test schemas differ")
    outcomes = {"record_id", "label", "treatment"}
    features = [name for name in columns if name not in outcomes]
    if len(features) != int(spec["expected_feature_count"]):
        raise ValueError("Unexpected Alice feature count")
    if spec["dataset"] == "uci_bank_marketing" and "treatment" in columns:
        raise ValueError("UCI has no randomized treatment")
    if spec["dataset"] == "hillstrom_email_marketing" and "treatment" not in columns:
        raise ValueError("Hillstrom treatment missing")
    if not all(name in columns for name in ("record_id", "label")):
        raise ValueError("Alice label or record key missing")
    banned = set(spec["excluded_source_features"])
    if any(any(feature.endswith(f"__{name}") or feature.startswith(f"categorical__{name}_")
               for name in banned) for feature in features):
        raise ValueError("Post-outcome/source-excluded feature present")
    for frame in frames.values():
        if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite encoded feature")
        if set(frame["label"].unique()) - {0, 1}:
            raise ValueError("Nonbinary label")
    return frames, features, fingerprints


def fit_score(x_train: np.ndarray, y_train: np.ndarray,
              x_eval: np.ndarray, alpha: float, seed: int) -> np.ndarray:
    if len(np.unique(y_train)) != 2:
        raise ValueError("Training arm has one label class")
    model = SGDClassifier(loss="log_loss", alpha=alpha, random_state=seed,
                          max_iter=MAX_ITER, tol=FIT_TOL, average=True)
    model.fit(x_train, y_train)
    return model.predict_proba(x_eval)[:, 1]


def top_recall(y: np.ndarray, score: np.ndarray, fraction: float) -> float:
    n = max(1, int(len(y) * fraction))
    return float(y[np.argsort(-score)[:n]].sum() / max(y.sum(), 1))


def uplift_at_k(y: np.ndarray, t: np.ndarray, score: np.ndarray, fraction: float) -> float:
    n = max(1, int(len(y) * fraction))
    idx = np.argsort(-score)[:n]
    if len(np.unique(t[idx])) != 2:
        return float("nan")
    return float(y[idx][t[idx] == 1].mean() - y[idx][t[idx] == 0].mean())


def centered_qini(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    """Area of IPW gain curve above the random-ranking diagonal."""
    p = float(t.mean())
    if not 0 < p < 1:
        return float("nan")
    order = np.argsort(-score)
    contribution = y * (t / p - (1 - t) / (1 - p))
    gain = np.r_[0.0, np.cumsum(contribution[order]) / len(y)]
    fraction = np.linspace(0.0, 1.0, len(y) + 1)
    return float(np.trapezoid(gain - fraction * gain[-1], fraction))


def interval(y: np.ndarray, score: np.ndarray, metric, boot: int, seed: int,
             treatment: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(seed)
    if treatment is None:
        point = float(metric(y, score))
    else:
        point = float(metric(y, treatment, score))
    draws = []
    for _ in range(boot):
        if treatment is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            groups = [np.flatnonzero(treatment == arm) for arm in (0, 1)]
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        if len(np.unique(y[idx])) != 2:
            continue
        value = (metric(y[idx], score[idx]) if treatment is None
                 else metric(y[idx], treatment[idx], score[idx]))
        if np.isfinite(value):
            draws.append(value)
    if len(draws) < int(boot * BOOTSTRAP_MIN_VALID_FRACTION):
        raise ValueError("Insufficient valid bootstrap draws")
    low, high = np.percentile(draws, CI_PERCENTILES)
    return {"estimate": point, "ci_low": float(low), "ci_high": float(high),
            "bootstrap_valid": len(draws)}


def select_alpha(frames: dict, features: list[str], spec: dict, scaler: StandardScaler) -> float:
    train, valid = frames["train"], frames["validation"]
    xa = scaler.transform(train[features].to_numpy(dtype=float))
    xv = scaler.transform(valid[features].to_numpy(dtype=float))
    seeds = spec["model_seeds"]
    scores = []
    for alpha in spec["alpha_grid"]:
        vals = []
        if spec["dataset"] == "uci_bank_marketing":
            yv = valid["label"].to_numpy(dtype=int)
            for seed in seeds:
                prob = fit_score(xa, train["label"].to_numpy(dtype=int), xv, alpha, seed)
                vals.append(roc_auc_score(yv, prob))
        else:
            for arm in spec["treatment_arms"]:
                mask = valid["treatment"].isin([0, arm]).to_numpy()
                yv = valid.loc[mask, "label"].to_numpy(dtype=int)
                tv = (valid.loc[mask, "treatment"].to_numpy(dtype=int) == arm).astype(int)
                for seed in seeds:
                    treatment_probs = {}
                    for group in (0, arm):
                        tr = train["treatment"].to_numpy() == group
                        treatment_probs[group] = fit_score(xa[tr], train.loc[tr, "label"].to_numpy(dtype=int),
                                                           xv[mask], alpha, seed)
                    vals.append(centered_qini(yv, tv, treatment_probs[arm] - treatment_probs[0]))
        scores.append(float(np.mean(vals)))
    return float(spec["alpha_grid"][int(np.argmax(scores))])


def evaluate(runtime: dict, spec: dict) -> dict:
    checks = check_isolation(runtime)
    frames, features, fingerprints = load_splits(runtime, spec)
    train, test = frames["train"], frames["test"]
    scaler = StandardScaler().fit(train[features].to_numpy(dtype=float))
    x_train = scaler.transform(train[features].to_numpy(dtype=float))
    x_test = scaler.transform(test[features].to_numpy(dtype=float))
    alpha = select_alpha(frames, features, spec, scaler)
    output = {"dataset": spec["dataset"], "status": "l0_only", "selected_alpha": alpha,
              "model_seeds": spec["model_seeds"], "split_seed": spec["split_seed"],
              "bootstrap_seed": spec["bootstrap_seed"], "input_sha256": fingerprints,
              "prepared_experiment": runtime["prepared_experiment"],
              "config_sha256": sha256(Path(runtime["public_config_snapshot"])),
              "code_sha256": runtime["code_sha256"], "git_sha": runtime["git_sha"],
              "isolation_checks": checks, "metrics": []}
    for seed in spec["model_seeds"]:
        if spec["dataset"] == "uci_bank_marketing":
            y = test["label"].to_numpy(dtype=int)
            prob = fit_score(x_train, train["label"].to_numpy(dtype=int), x_test, alpha, seed)
            output["metrics"].append({
                "seed": seed,
                "roc_auc": interval(y, prob, roc_auc_score, spec["bootstrap_repeats"],
                                    spec["bootstrap_seed"] + seed),
                "pr_auc": interval(y, prob, average_precision_score, spec["bootstrap_repeats"],
                                   spec["bootstrap_seed"] + seed),
                "recall_at_top_k": interval(y, prob,
                    lambda a, b: top_recall(a, b, spec["top_k_fraction"]),
                    spec["bootstrap_repeats"], spec["bootstrap_seed"] + seed),
            })
        else:
            for arm in spec["treatment_arms"]:
                probabilities = {}
                for group in (0, arm):
                    mask = train["treatment"].to_numpy() == group
                    probabilities[group] = fit_score(x_train[mask],
                        train.loc[mask, "label"].to_numpy(dtype=int), x_test, alpha, seed)
                score = probabilities[arm] - probabilities[0]
                selected = test["treatment"].isin([0, arm]).to_numpy()
                y = test.loc[selected, "label"].to_numpy(dtype=int)
                t = (test.loc[selected, "treatment"].to_numpy(dtype=int) == arm).astype(int)
                output["metrics"].append({
                    "seed": seed, "treatment_arm": arm,
                    "centered_qini": interval(y, score[selected], centered_qini,
                        spec["bootstrap_repeats"], spec["bootstrap_seed"] + seed + arm, t),
                    "uplift_at_top_k": interval(y, score[selected],
                        lambda a, b, c: uplift_at_k(a, b, c, spec["top_k_fraction"]),
                        spec["bootstrap_repeats"], spec["bootstrap_seed"] + seed + arm, t),
                })
    path = Path(runtime["result_dir"]) / "l0_private_metrics.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    rows = ["# Alice-only L0 私有结果", "",
            f"数据集：{spec['dataset']}；预处理实验：{runtime['prepared_experiment']}。",
            f"配置 SHA-256：{output['config_sha256']}；代码 SHA-256：{output['code_sha256']}。",
            f"冻结切分种子：{spec['split_seed']}；模型种子：{spec['model_seeds']}；"
            f"bootstrap 种子：{spec['bootstrap_seed']}；验证集所选 alpha：{alpha}。", "",
            "以下区间是在**同一冻结测试切片**上重采样得到，不能代表跨机构或跨时间人群的不确定性。",
            "字段归属与共同客户交集均为构造条件；本步未做 PSI 或联合训练。", ""]
    if spec["dataset"] == "uci_bank_marketing":
        rows += ["| 模型种子 | ROC AUC（95% CI） | PR AUC（95% CI） | Top 10% 捕获率（95% CI） |",
                 "|---|---:|---:|---:|"]
        names: tuple[str, ...] = ("roc_auc", "pr_auc", "recall_at_top_k")
    else:
        rows += ["| 模型种子 | 邮件组 | 中心化 Qini（95% CI） | Top 10% 增量率（95% CI） |",
                 "|---|---:|---:|---:|"]
        names = ("centered_qini", "uplift_at_top_k")
    for item in output["metrics"]:
        cells = [str(item["seed"])]
        if "treatment_arm" in item:
            cells.append(str(item["treatment_arm"]))
        cells += [f"{item[name]['estimate']:.4f} [{item[name]['ci_low']:.4f}, "
                  f"{item[name]['ci_high']:.4f}]" for name in names]
        rows.append("| " + " | ".join(cells) + " |")
    rows += ["", "UCI 是响应预测诊断，不能解释为营销因果增量。"
             if spec["dataset"] == "uci_bank_marketing" else
             "Hillstrom 是随机邮件试验代理，但零售邮件效果不能外推到银行业务。",
             "模型种子共用同一测试切片，不能把五次结果当作五个独立业务样本。"]
    (Path(runtime["result_dir"]) / "结果分析.md").write_text("\n".join(rows) + "\n")
    return {"status": "l0_only", "seeds_completed": len(spec["model_seeds"]),
            "metrics_path": str(path)}
