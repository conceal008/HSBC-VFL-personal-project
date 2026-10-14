"""Bounded single-owner encoding; fixed hashes and train-only fitted states."""

from __future__ import annotations

import ast
import csv
from functools import lru_cache
import hashlib
from pathlib import Path
import re
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from external_party import isolation, save, sha, SPLITS
from external_party import execute_preparation as execute_preparation

MISSING = {"", "NA", "NaN", "nan", "None", "null"}
HASH_BYTES = 8
SIGN_BYTE = 8
HASH_DIGEST_BYTES = 9
HASH_CACHE_SIZE = 100000


@lru_cache(maxsize=HASH_CACHE_SIZE)
def bucket(name, token, seed, width):
    digest = hashlib.blake2b(
        f"{seed}:{name}:{token}".encode(), digest_size=HASH_DIGEST_BYTES
    ).digest()
    return int.from_bytes(digest[:HASH_BYTES], "big") % width, (
        1.0 if digest[SIGN_BYTE] & 1 else -1.0
    )


def tokens(value, lists):
    if value.strip() in MISSING:
        return ["__MISSING__"]
    if lists and value.strip().startswith("["):
        try:
            result = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            result = re.split(r"[\s,]+", value.strip().strip("[]"))
        if not isinstance(result, (list, tuple)):
            raise ValueError("Invalid categorical token list")
        return [str(v) for v in result] or ["__EMPTY__"]
    return [value]


def numeric(value):
    if value.strip() in MISSING:
        return np.nan
    try:
        result = float(value)
    except ValueError:
        return None
    if not np.isfinite(result):
        raise ValueError("Nonfinite numeric feature")
    return result


def read_encoded(path, spec, outcomes, encoding):
    features = spec["features"]
    categorical = set(spec["categorical"])
    wide = len(features) > encoding["wide_threshold"]
    nums = [] if wide else [n for n in features if n not in categorical]
    positions = {n: i for i, n in enumerate(nums)}
    # All fields can have missing/non-numeric fallback tokens, so keep a fixed hash block.
    width = encoding["wide_dimensions"] if wide else encoding["hash_dimensions"]
    names = ["numeric__" + n for n in nums] + [
        "hashed__" + str(i) for i in range(width)
    ]
    ids, matrices, target = [], [], {n: [] for n in outcomes}
    field_hashes = (
        {n: bucket(n, "__NUMERIC__", encoding["seed"], width) for n in features}
        if wide
        else {}
    )
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["record_id", *features, *outcomes]:
            raise ValueError("Owner schema differs")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed owner input")
            ids.append(row["record_id"])
            vector = np.zeros(len(names), dtype=np.float32)
            for name in features:
                value = row[name]
                n = None if name in categorical else numeric(value)
                if name not in categorical and n is not None:
                    if wide:
                        if not np.isnan(n):
                            index, sign = field_hashes[name]
                            vector[index] += sign * np.sign(n) * np.log1p(abs(n))
                    else:
                        vector[positions[name]] = np.sign(n) * np.log1p(abs(n))
                        if encoding.get("numeric_transform") != "signed_log1p":
                            raise ValueError("Undeclared numeric transform")
                else:
                    if not wide and name in positions:
                        vector[positions[name]] = np.nan
                    for token in tokens(value, spec.get("token_lists", False)):
                        index, sign = bucket(name, token, encoding["seed"], width)
                        vector[len(nums) + index] += sign
            if wide:
                vector /= max(float(np.linalg.norm(vector)), 1.0)
            matrices.append(vector)
            for name in outcomes:
                value = row[name]
                if encoding.get("label_format") == "singleton_list":
                    singleton = ast.literal_eval(value)
                    if not isinstance(singleton, list) or len(singleton) != 1:
                        raise ValueError("Outcome list must have exactly one scalar")
                    value = singleton[0]
                y = float(value)
                if y not in {0.0, 1.0}:
                    raise ValueError("Nonbinary owner outcome")
                target[name].append(int(y))
    if not ids or len(set(ids)) != len(ids) or any(not k for k in ids):
        raise ValueError("Empty or duplicate owner record key")
    return ids, np.stack(matrices), target, names


def prepare(runtime):
    root = Path(runtime["root"])
    if (
        any(any((root / sub).iterdir()) for sub in ("data", "results"))
        or (root / "logs/isolation.json").exists()
    ):
        raise FileExistsError(
            "Frozen preparation outputs exist; use a new runtime directory"
        )
    checks = isolation(runtime)
    if sha(root / "code/portfolio_repair_freeze.yaml") != runtime["config_sha256"]:
        raise ValueError("Config changed")
    if any(sha(root / "code" / n) != h for n, h in runtime["code_sha256"].items()):
        raise ValueError("Code changed")
    if runtime["party"] == "bob" and runtime["outcomes"]:
        raise ValueError("Bob outcome disclosure")
    inputs = runtime["input_sha256"]
    if set(inputs) != {s + ".csv" for s in SPLITS}:
        raise ValueError("Sealed capability entered preparation")
    encoded = {}
    for split in SPLITS:
        p = root / "input" / f"{split}.csv"
        if sha(p) != inputs[p.name]:
            raise ValueError("Input changed")
        encoded[split] = read_encoded(
            p, runtime["feature_spec"], runtime["outcomes"], runtime["encoding"]
        )
    if set(encoded["train"][0]) & set(encoded["validation"][0]):
        raise ValueError("Development identities overlap")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(
        encoded["train"][1]
    )
    scaler = StandardScaler().fit(imputer.transform(encoded["train"][1]))
    for split, (ids, matrix, targets, names) in encoded.items():
        clean = np.clip(
            scaler.transform(imputer.transform(matrix)),
            -runtime["encoding"]["numeric_clip"],
            runtime["encoding"]["numeric_clip"],
        )
        if not np.isfinite(clean).all():
            raise ValueError("Nonfinite cleaned features")
        if runtime["party"] == "alice" and any(
            set(v) != {0, 1} for v in targets.values()
        ):
            raise ValueError("Outcome support insufficient; do not resample by outcome")
        with (root / "data" / f"{split}.csv").open("x", newline="") as f:
            outcomes = (
                (["label"] + (["treatment"] if len(targets) > 1 else []))
                if targets
                else []
            )
            writer = csv.writer(f)
            writer.writerow(["record_id", *names, *outcomes])
            writer.writerows(
                [key, *x.tolist(), *[targets[n][i] for n in runtime["outcomes"]]]
                for i, (key, x) in enumerate(zip(ids, clean))
            )
    save(
        root / "data/preprocessing_state.json",
        {
            "fit_split": "train",
            "encoding": runtime["encoding"],
            "output_features": encoded["train"][
                3  # 魔数豁免: 固定read_encoded四元组列名位置
            ],  # 魔数豁免: read_encoded固定四元组的列名位置
            "median": imputer.statistics_.tolist(),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "collision_free": False,
        },
    )
    save(
        root / "data/manifest.json",
        {p.name: sha(p) for p in (root / "data").glob("*.csv")},
    )
    save(root / "logs/isolation.json", checks)
    if any(sha(root / "input" / n) != h for n, h in inputs.items()):
        raise ValueError("Frozen input changed")
    receipt = {
        "status": "prepared",
        "party": runtime["party"],
        "train_only_fit": True,
        "sealed_confirmation_read": False,
        "input_unchanged": True,
        "isolation_checks_passed": True,
        "training_performed": False,
        "physical_isolation": False,
    }
    save(root / "results/engineering.json", receipt)
    return receipt
