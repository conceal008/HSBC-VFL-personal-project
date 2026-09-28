"""Read-only source adapters and append-only local VFL preparation artifacts."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

HASH_BLOCK = 1024 * 1024
RUN_TOKEN_LENGTH = 8
SPLITS = ("train", "validation", "test")
PARTIES = ("alice", "bob")
ID = "record_id"
LABEL = "label"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Exclusive creation: existing artifacts are never replaced."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, frame):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)


def inside(path, parent):
    return path == parent or parent in path.parents


def validate_layout(raw, derived, results, repo):
    paths = [Path(p).resolve() for p in (raw, derived, results, repo)]
    for i, first in enumerate(paths):
        for second in paths[i + 1:]:
            if inside(first, second) or inside(second, first):
                raise ValueError("原始数据、产物、结果及仓库目录必须分离且不能互相包含")
    return paths


def prepare_context(config_path, repo, roots=None, run_id=None):
    repo = Path(repo).resolve()
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = repo.parent / "8-9月"
    defaults = (base / "数据集", base / "数据处理产物", base / "数据处理运行结果")
    if roots is None:
        roots = tuple(Path(os.environ.get(key, str(default))) for key, default in zip(
            ("VFL_RAW_ROOT", "VFL_DERIVED_ROOT", "VFL_RESULTS_ROOT"), defaults))
    raw, derived, results, repo = validate_layout(*roots, repo)
    source = (raw / config["source"]["relative_path"]).resolve()
    if not inside(source, raw) or not source.is_file():
        raise ValueError("输入必须是原始目录内已存在的普通文件")
    dataset = config["dataset"]
    run_id = run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                        + "_" + uuid4().hex[:RUN_TOKEN_LENGTH])
    for name in (dataset, run_id):
        if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
            raise ValueError("dataset/run_id 必须是单个目录名")
    data_dir = derived / dataset / run_id
    result_dir = results / dataset / run_id
    # Resolve again to reject pre-existing symlinks in dataset/run paths.
    for target, root in ((data_dir, derived), (result_dir, results)):
        if not inside(target.resolve(), root) or target.exists():
            raise FileExistsError("运行目录已存在或离开指定根目录：" + str(target))
    data_dir.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)
    snapshot = result_dir / "source_snapshot"
    snapshot.mkdir()
    module = config_path.parent.parent
    for source_dir in (module / "components", module / "notebooks"):
        shutil.copytree(source_dir, snapshot / source_dir.name,
                        ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"))
    shutil.copyfile(config_path, snapshot / config_path.name)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True,
                         capture_output=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, text=True,
                           capture_output=True, check=False)
    fingerprint = {
        "dataset": dataset, "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path), "config_sha256": sha256(config_path),
        "seeds": config["seeds"], "git_sha": git.stdout.strip() or "unavailable",
        "git_dirty": bool(dirty.stdout), "python": sys.version,
        "executable": sys.executable, "platform": platform.platform(),
        "source": str(source), "source_sha256_before": sha256(source),
        "code_sha256": {str(p.relative_to(snapshot)): sha256(p)
                          for p in sorted(snapshot.rglob("*")) if p.is_file()},
        "dependencies": {d.metadata["Name"]: d.version
                          for d in importlib.metadata.distributions()},
        "data_dir": str(data_dir), "result_dir": str(result_dir),
    }
    write_json(result_dir / "fingerprint.json", fingerprint)
    return {"config": config, "source": source, "data_dir": data_dir,
            "result_dir": result_dir, "fingerprint": fingerprint}


def load_source(ctx):
    cfg = ctx["config"]
    frame = pd.read_csv(ctx["source"], sep=cfg["source"]["separator"])
    if frame.empty or list(frame.columns) != cfg["source"]["expected_columns"]:
        raise ValueError("数据为空或表头与版本契约不一致")
    key = ctx["fingerprint"]["source_sha256_before"]
    ids = [hashlib.sha256(f"{key}:{i}".encode()).hexdigest() for i in range(len(frame))]
    frame.insert(0, ID, ids)
    frame = frame.set_index(ID, drop=False)
    y = frame[cfg["label"]["column"]].map(cfg["label"]["mapping"])
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("标签有未定义取值或不同时含两个类别")
    outcomes = pd.DataFrame({ID: frame[ID], LABEL: y.astype(int)}, index=frame.index)
    if "treatment" in cfg:
        t = frame[cfg["treatment"]["column"]].map(cfg["treatment"]["mapping"])
        if t.isna().any():
            raise ValueError("处理组有未定义取值")
        outcomes["treatment"] = t.astype(int)
    selected = [f for party in cfg["parties"].values() for f in party["features"]]
    if len(selected) != len(set(selected)):
        raise ValueError("双方字段不能重复")
    forbidden = set(cfg["excluded_features"]) | {cfg["label"]["column"], ID, LABEL}
    if "treatment" in cfg:
        forbidden.add(cfg["treatment"]["column"])
    if set(selected) & forbidden:
        raise ValueError("特征含标签、处理变量、记录键或预测时点后的字段")
    if set(selected) | set(cfg["excluded_features"]) | {cfg["label"]["column"]} != set(frame.columns) - {ID}:
        raise ValueError("所有源字段必须有用途或排除理由")
    clean = frame[[ID] + selected].copy()
    for name, value in cfg.get("numeric_sentinels", {}).items():
        clean[name] = clean[name].replace(value, np.nan)
    for party in cfg["parties"].values():
        for name in party["categorical"]:
            clean[name] = clean[name].replace(cfg["missing_categories"], np.nan)
    return clean, outcomes


def split_records(frame, outcomes, cfg):
    ids = frame[ID].to_numpy()
    fractions = cfg["split"]["fractions"]
    if len(fractions) != len(SPLITS) or not np.isclose(sum(fractions), 1) or min(fractions) <= 0:
        raise ValueError("划分比例必须为三个正数，和为 1")
    if cfg["split"]["method"] == "source_order":
        train_end = int(len(ids) * fractions[0])
        val_end = int(len(ids) * sum(fractions[:2]))
        groups = (ids[:train_end], ids[train_end:val_end], ids[val_end:])
    elif cfg["split"]["method"] == "stratified":
        strata = outcomes[LABEL].astype(str)
        if "treatment" in outcomes:
            strata = strata + "_" + outcomes["treatment"].astype(str)
        train, other = train_test_split(ids, train_size=fractions[0],
                                       stratify=strata, random_state=cfg["seeds"]["split"])
        val, test = train_test_split(other, train_size=fractions[1] / sum(fractions[1:]),
                                    stratify=strata.loc[other], random_state=cfg["seeds"]["split"])
        groups = (train, val, test)
    else:
        raise ValueError("不支持的划分方法")
    split = pd.Series(index=frame.index, dtype="object", name="split")
    for name, group in zip(SPLITS, groups):
        if not len(group):
            raise ValueError("划分为空")
        split.loc[group] = name
    if split.isna().any() or not frame[ID].is_unique:
        raise ValueError("记录未完整划分或记录键不唯一")
    return split


def encode_party(frame, train_ids, spec):
    cats = spec["categorical"]
    nums = [f for f in spec["features"] if f not in cats]
    features = frame[spec["features"]].copy()
    # Integer-coded categories are categorical values, not numeric measurements.
    for name in cats:
        features[name] = features[name].map(lambda value: str(value) if pd.notna(value) else np.nan)
    encoder = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                              ("scale", StandardScaler())]), nums),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cats),
    ], sparse_threshold=0)
    encoder.fit(features.loc[train_ids])
    values = encoder.transform(features)
    if not np.isfinite(values).all():
        raise ValueError("编码后的特征含 NaN 或无穷值")
    names = list(encoder.get_feature_names_out())
    encoded = pd.DataFrame(values, columns=names, index=frame.index)
    encoded.insert(0, ID, frame[ID])
    num = encoder.named_transformers_["numeric"]
    cat = encoder.named_transformers_["categorical"]
    state = {"fit_split": "train", "input_features": spec["features"],
             "numeric_features": nums, "categorical_features": cats,
             "output_features": names, "median": num["impute"].statistics_.tolist(),
             "mean": num["scale"].mean_.tolist(), "scale": num["scale"].scale_.tolist(),
             "categories": [v.tolist() for v in cat["encode"].categories_],
             "unknown_categories": "all-zero one-hot", "missing_category": "__MISSING__"}
    return encoded, state


def overlap_inputs(frames, train_ids, cfg):
    ratio = cfg["overlap"]["shared_fraction_of_union"]
    if not 0 < ratio <= 1:
        raise ValueError("交集占并集比例必须在 (0, 1]")
    rng = np.random.default_rng(cfg["seeds"]["overlap"])
    ids = rng.permutation(train_ids)
    shared_end = int(len(ids) * ratio)
    left_end = shared_end + (len(ids) - shared_end) // 2
    shared = ids[:shared_end]
    alice = np.concatenate((shared, ids[shared_end:left_end]))
    bob = np.concatenate((shared, ids[left_end:]))
    tables = {"alice": frames["alice"].loc[rng.permutation(alice)],
              "bob": frames["bob"].loc[rng.permutation(bob)]}
    actual = set(tables["alice"][ID]) & set(tables["bob"][ID])
    if actual != set(shared):
        raise AssertionError("构造重叠不符合预期")
    return tables, {"meaning": "record-level synthetic overlap; no cryptographic PSI executed",
                    "ratio_denominator": "union of train records", "shared": len(shared),
                    "union": len(ids), "alice": len(alice), "bob": len(bob)}


def run_pipeline(ctx):
    cfg, data_dir, results = ctx["config"], ctx["data_dir"], ctx["result_dir"]
    try:
        frame, outcomes = load_source(ctx)
        split = split_records(frame, outcomes, cfg)
        write_csv(data_dir / "clean_features.csv", frame)
        write_csv(data_dir / "split_assignments.csv", pd.DataFrame({ID: frame[ID], "split": split}))
        train_ids = frame.loc[split == "train", ID].to_numpy()
        encoded, states, rows = {}, {}, []
        for party in PARTIES:
            encoded[party], states[party] = encode_party(frame, train_ids, cfg["parties"][party])
            party_dir = data_dir / party
            party_dir.mkdir()
            for name in SPLITS:
                ids = frame.loc[split == name, ID]
                table = encoded[party].loc[ids].copy()
                if party == "alice":
                    table = table.join(outcomes.drop(columns=[ID]))
                write_csv(party_dir / f"{name}.csv", table)
                rows.append({"party": party, "split": name, "rows": len(table),
                             "encoded_features": len(states[party]["output_features"]),
                             "has_label": LABEL in table.columns})
        overlap_dir = data_dir / "psi_inputs"
        overlap_dir.mkdir()
        tables, overlap = overlap_inputs(encoded, train_ids, cfg)
        for party, table in tables.items():
            if party == "alice":
                table = table.join(outcomes.drop(columns=[ID]))
            write_csv(overlap_dir / f"{party}.csv", table)
        write_json(results / "preprocessing_state.json", states)
        write_json(results / "overlap_audit.json", overlap)
        write_csv(results / "partition_summary.csv", pd.DataFrame(rows))
        quality = pd.DataFrame({"feature": frame.columns.drop(ID),
                                "missing_count": frame.drop(columns=[ID]).isna().sum().values})
        write_csv(results / "missingness.csv", quality)
        write_json(results / "data_contract.json", {
            "dataset": cfg["dataset"], "record_key": ID, "label": LABEL,
            "record_key_origin": "sha256(source_file_sha256:original_row_position)",
            "not_customer_identity": True, "prediction_time": cfg["prediction_time"],
            "parties": cfg["parties"], "excluded_features": cfg["excluded_features"],
            "feature_columns_for_training": {p: states[p]["output_features"] for p in PARTIES},
            "split": cfg["split"], "limits": cfg["limits"],
            "training_reader_rule": "select feature_columns_for_training explicitly; exclude record_id, label, treatment",
        })
        source_after = sha256(ctx["source"])
        if source_after != ctx["fingerprint"]["source_sha256_before"]:
            raise RuntimeError("原始输入哈希发生变化，本次运行不可用")
        checks = {"source_sha256_after": source_after, "raw_unchanged": True,
                  "split_disjoint": all(not set(frame.loc[split == a, ID]) & set(frame.loc[split == b, ID])
                                        for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]),
                  "finite_matrices": True, "preprocessing_fit_on_train_only": True,
                  "party_alignment": all(encoded["alice"].loc[split == name, ID].equals(
                      encoded["bob"].loc[split == name, ID]) for name in SPLITS),
                  "bob_has_no_outcomes": all(col not in encoded["bob"] for col in (LABEL, "treatment")),
                  "cryptographic_psi_executed": False, "federated_training_executed": False}
        if not all(checks[k] for k in ("split_disjoint", "party_alignment", "bob_has_no_outcomes")):
            raise AssertionError("划分/标签/对齐校验失败")
        write_json(results / "checks.json", checks)
        write_json(results / "data_manifest.json", {
            "config_sha256": ctx["fingerprint"]["config_sha256"],
            "files": {str(p.relative_to(data_dir)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                      for p in sorted(data_dir.rglob("*.csv"))}})
        report = (f"# {cfg['dataset']} 数据准备运行\n\n"
                  f"运行：`{ctx['fingerprint']['run_id']}`\n\n"
                  "原件哈希前后一致；训练集拟合的编码已应用于独立验证与测试划分。\n\n"
                  "## 输出\n\n" + pd.DataFrame(rows).to_string(index=False) + "\n\n"
                  "## 解释边界\n\n" + "\n".join("- " + item for item in cfg["limits"]) +
                  "\n\n处理完成不等于联邦训练完成；未产生模型效果结论。\n")
        with (results / "README.md").open("x", encoding="utf-8") as stream:
            stream.write(report)
        write_json(results / "PREPARATION_READY.json", {"run_id": ctx["fingerprint"]["run_id"], "checks": "passed"})
        return {"summary": pd.DataFrame(rows), "missingness": quality, "checks": checks,
                "overlap": overlap, "data_dir": data_dir, "result_dir": results}
    except Exception as exc:
        write_json(results / "FAILED.json", {"error_type": type(exc).__name__, "message": str(exc),
                                             "source_sha256_after": sha256(ctx["source"])})
        raise
