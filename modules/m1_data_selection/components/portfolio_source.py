"""Read-only public fixture allocation, with time-safe panel outcomes and sealed splits.

The trusted fixture allocator has access to public source files. Training UIDs
never receive this capability; no joint cleaned table is written or exported.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import itertools
import json
from pathlib import Path
import yaml

from external_freeze import sha, PARTIES, PRIVATE_DIRECTORY_MODE, PRIVATE_FILE_MODE

HASH_BITS = 64
BYTES_PER_HASH = 8
MONTHS_PER_YEAR = 12
PRODUCT_CUSTOMER_FRACTION = 0.05
SPLITS = ("train", "validation", "confirmation_raw")


def feature_spec(spec, party):
    result = dict(spec["parties"][party])
    if "var_start" in result:
        result["features"] = [
            "Var" + str(i)
            for i in range(
                result.pop("var_start"), spec["width"] + 1, result.pop("var_step")
            )
        ]
        result["categorical"] = []
    result["hybrid_numeric"] = spec.get("hybrid_numeric", False)
    result["token_lists"] = spec.get("token_lists", False)
    return result


def rank(key, seed):
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{key}".encode()).digest()[:BYTES_PER_HASH], "big"
    )


def month_number(value):
    year, month = map(int, value[:7].split("-"))  # 魔数豁免: ISO YYYY-MM固定七字符
    if not 1 <= month <= MONTHS_PER_YEAR:
        raise ValueError("Invalid month")
    return year * MONTHS_PER_YEAR + month


def split_for(key, customer, target_month, cfg, spec):
    u = rank(customer if customer is not None else key, cfg["seeds"]["split"]) / (
        1 << HASH_BITS
    )
    a, b, _ = cfg["split_fractions"]
    group = "train" if u < a else ("validation" if u < a + b else "confirmation_raw")
    temporal = spec.get("temporal")
    if temporal is None:
        return group
    valid = (
        target_month <= temporal["train_last_target"]
        if group == "train"
        else target_month in temporal["validation_targets"]
        if group == "validation"
        else target_month in temporal["confirmation_targets"]
    )
    return group if valid else None


def checked_csv(path, columns):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != columns:
            raise ValueError("Declared source schema differs")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed source row")
            yield row


def product_state(row, products):
    values = []
    for name in products:
        v = row[name].strip()
        if v not in {"0", "1", "0.0", "1.0"}:
            return None
        values.append(int(float(v)))
    return values


def panel_pairs(rows, spec, cfg, mock=False):
    """Use feature month t, label at t+1. Never put target-month products in features."""
    previous = {}
    seen = set()
    products = spec.get(
        "products",
        [n for n in feature_spec(spec, "bob")["features"] if n.endswith("_ult1")],
    )
    if not products:
        raise ValueError("Panel product schema unavailable")
    for row in rows:
        customer, date = row[spec["key"]], row[spec["time"]]
        if (
            not mock
            and rank(customer, cfg["seeds"]["sampling"]) / (1 << HASH_BITS)
            >= spec.get("customer_sampling_fraction", PRODUCT_CUSTOMER_FRACTION)
        ):
            continue
        month = month_number(date)
        key = (customer, month)
        if key in seen:
            raise ValueError("Duplicate customer-month source")
        seen.add(key)
        state = product_state(row, products)
        old = previous.get(customer)
        if old is not None and month <= old[0]:
            raise ValueError("Panel is not chronologically ordered per customer")
        if (
            old is not None
            and month == old[0] + 1
            and old[2] is not None
            and state is not None
        ):
            y = int(any(new > before for before, new in zip(old[2], state)))
            if (
                mock
                and spec.get("mock_outcome_policy") != "derive_next_product_transition"
            ):
                raise ValueError("Unverified Mock outcomes cannot be reused")
            frozen = dict(old[1])
            frozen[spec["label"]] = str(y)
            yield f"{customer}:{old[0]}->{month}", customer, date[:7], frozen  # 魔数豁免: ISO月份前七字符
        previous[customer] = (month, row, state)


def rows(raw_root, spec, cfg):
    kind = spec["kind"]
    if kind == "csv":
        for index, row in enumerate(
            checked_csv(raw_root / spec["source"], spec["columns"])
        ):
            key = row[spec["key"]] if spec.get("key") else str(index)
            group = key if spec.get("split_group") == "source_sample_id" else None
            if spec.get("event_identity") == "source_key_plus_row":
                key = f"{key}:{index}"
            yield key, group, None, row
    elif kind == "orange":
        columns = ["Var" + str(i) for i in range(1, spec["width"] + 1)]
        with (raw_root / spec["labels"]).open() as labels:
            index = 0
            for chunk, source in enumerate(spec["sources"]):
                with (raw_root / source).open(newline="") as f:
                    reader = csv.reader(f, delimiter="\t")
                    if chunk == 0 and next(reader) != columns:
                        raise ValueError("First Orange chunk requires declared header")
                    for values in reader:
                        if len(values) != len(columns):
                            raise ValueError("Orange row width differs")
                        line = labels.readline().strip()
                        if line not in {"-1", "1"}:
                            raise ValueError("Orange outcome missing or invalid")
                        row = dict(zip(columns, values))
                        row[spec["label"]] = str(int(line == "1"))
                        yield str(index), None, None, row
                        index += 1
            if labels.read().strip():
                raise ValueError("Orange outcomes outnumber source rows")
    elif kind == "mock":
        a = checked_csv(raw_root / spec["guest"], spec["guest_columns"])
        b = checked_csv(raw_root / spec["host"], spec["host_columns"])

        def owner_pairs():
            for left, right in itertools.zip_longest(a, b):
                if (
                    left is None
                    or right is None
                    or left[spec["key"]] != right[spec["key"]]
                ):
                    raise ValueError("Mock owner row identities differ")
                if set(left) & set(right) != {spec["key"]}:
                    raise ValueError("Mock owner columns overlap")
                yield dict(left, **right)

        yield from panel_pairs(owner_pairs(), spec, cfg, mock=True)
    elif kind == "product":
        yield from panel_pairs(
            checked_csv(raw_root / spec["source"], spec["columns"]), spec, cfg
        )
    else:
        raise ValueError("Undeclared source format")


def input_files(spec):
    if spec["kind"] == "orange":
        return [*spec["sources"], spec["labels"]]
    if spec["kind"] == "mock":
        return [spec["guest"], spec["host"]]
    return [spec["source"]]


def provision(raw_root, output, dataset, config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    spec = cfg["datasets"][dataset]
    raw_root, output = Path(raw_root).resolve(), Path(output).resolve()
    if output == raw_root or raw_root in output.parents:
        raise ValueError("Output would pollute original source")
    output.mkdir(parents=True, exist_ok=False, mode=PRIVATE_DIRECTORY_MODE)
    source_paths = [(raw_root / name).resolve() for name in input_files(spec)]
    if not all(raw_root in p.parents for p in source_paths):
        raise ValueError("Source traversal denied")
    before = {str(p.relative_to(raw_root)): sha(p) for p in source_paths}
    # Reservoir contains ranks/keys only. No labels, features or population counts guide selection.
    heap = []
    limit = cfg["max_cohort_rows"]
    for key, customer, month, _ in rows(raw_root, spec, cfg):
        if split_for(key, customer, month, cfg, spec) is None:
            continue
        priority = rank(dataset + ":" + key, cfg["seeds"]["sampling"])
        entry = (-priority, key)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    selected = {key for _, key in heap}
    if len(selected) != len(heap):
        raise ValueError("Duplicate allocation identity")
    handles, writers, paths = [], {}, {}
    try:
        for party in PARTIES:
            own = output / party
            own.mkdir(mode=PRIVATE_DIRECTORY_MODE)
            outcomes = [spec["label"]] if party == "alice" else []
            if party == "alice" and spec["treatment"]:
                outcomes.append(spec["treatment"])
            fields = ["record_id", *feature_spec(spec, party)["features"], *outcomes]
            if len(fields) != len(set(fields)):
                raise ValueError("Owner feature/outcome schema overlaps")
            for split in SPLITS:
                path = own / f"{split}.csv"
                stream = path.open("x", newline="")
                handles.append(stream)
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writers[party, split], paths[party, split] = writer, path
        for key, customer, month, row in rows(raw_root, spec, cfg):
            if key not in selected:
                continue
            split = split_for(key, customer, month, cfg, spec)
            row["record_id"] = hashlib.sha256(
                (dataset + ":" + key).encode()
            ).hexdigest()
            for party in PARTIES:
                writer = writers[party, split]
                writer.writerow({k: row[k] for k in writer.fieldnames})
    finally:
        for stream in handles:
            stream.close()
    if any(sha(p) != before[str(p.relative_to(raw_root))] for p in source_paths):
        raise ValueError("Original source changed")
    manifest = {f"{p}/{s}.csv": sha(path) for (p, s), path in paths.items()}
    for (_, split), path in paths.items():
        path.chmod(0 if split == "confirmation_raw" else PRIVATE_FILE_MODE)
    receipt = {
        "dataset": dataset,
        "source_sha256": before,
        "config_sha256": sha(config_path),
        "manifest": manifest,
        "confirmation_used": False,
        "selection_uses_outcomes": False,
        "fit_performed": False,
        "time_safe": spec["kind"] in {"mock", "product"},
        "scope": spec["scope"],
    }
    with (output / "provisioning.json").open("x") as f:
        json.dump(receipt, f, indent=2)
    return receipt
