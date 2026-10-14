"""Trusted fixture metadata allocation for owner-only cluster evaluation.

Only source key/month positions are selected. No source labels or features are
consulted, no sealed file is opened, and the map is never public or sent to Bob.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import yaml

PREPARATION_COMPONENTS = (
    Path(__file__).resolve().parents[2] / "m1_data_selection/components"
)
sys.path.insert(0, str(PREPARATION_COMPONENTS))
from portfolio_source import month_number, rank, HASH_BITS, PRODUCT_CUSTOMER_FRACTION  # noqa: E402
from external_freeze import sha  # noqa: E402

PRIVATE_MODE = 0o600
PRIVATE_DIRECTORY_MODE = 0o700


def key_metadata(path, columns, key, time=None):
    positions = [columns.index(key)] + ([columns.index(time)] if time else [])
    with Path(path).open(newline="") as f:
        reader = csv.reader(f)
        if next(reader) != columns:
            raise ValueError("Metadata source schema differs")
        for row in reader:
            if len(row) != len(columns):
                raise ValueError("Metadata source row malformed")
            yield tuple(row[i] for i in positions)


def build(raw_root, provisioned, dataset, config_path, output):
    cfg = yaml.safe_load(Path(config_path).read_text())
    spec = cfg["datasets"][dataset]
    if (
        spec["kind"] not in {"mock", "product"}
        and spec.get("split_group") != "source_sample_id"
    ):
        raise ValueError("Cluster metadata is not declared for this task")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=PRIVATE_DIRECTORY_MODE)
    ids = []
    # Only first key field from already allocated development inputs, not sealed inputs.
    for split in ("train", "validation"):
        with (Path(provisioned) / "alice" / f"{split}.csv").open(newline="") as f:
            reader = csv.reader(f)
            if next(reader)[0] != "record_id":
                raise ValueError("Allocated identity schema differs")
            ids.extend(row[0] for row in reader)
    selected = set(ids)
    if len(selected) != len(ids):
        raise ValueError("Allocated identities overlap")
    source = Path(raw_root) / (
        spec["guest"] if spec["kind"] == "mock" else spec["source"]
    )
    columns = spec["guest_columns"] if spec["kind"] == "mock" else spec["columns"]
    before = sha(source)
    allocation = json.loads((Path(provisioned) / "provisioning.json").read_text())
    if allocation["source_sha256"].get(str(source.relative_to(raw_root))) != before:
        raise ValueError("Metadata original source changed")
    previous, mapping = {}, {}
    panel = spec["kind"] in {"mock", "product"}
    for index, meta in enumerate(
        key_metadata(source, columns, spec["key"], spec["time"] if panel else None)
    ):
        customer = meta[0]
        if panel:
            if spec["kind"] == "product" and rank(
                customer, cfg["seeds"]["sampling"]
            ) / (1 << HASH_BITS) >= spec.get(
                "customer_sampling_fraction", PRODUCT_CUSTOMER_FRACTION
            ):
                continue
            month = month_number(meta[1])
            old = previous.get(customer)
            previous[customer] = month
            if old is None or month != old + 1:
                continue
            key = f"{customer}:{old}->{month}"
        else:
            key = f"{customer}:{index}"
        record = hashlib.sha256((dataset + ":" + key).encode()).hexdigest()
        if record in selected:
            if record in mapping:
                raise ValueError("Metadata identity repeats")
            mapping[record] = hashlib.sha256(
                ("owner-cluster:" + dataset + ":" + customer).encode()
            ).hexdigest()
    if set(mapping) != selected or sha(source) != before:
        raise ValueError("Metadata allocation coverage/source mismatch")
    p = output / "owner_groups.json"
    with p.open("x") as f:
        json.dump(mapping, f)
    p.chmod(PRIVATE_MODE)
    receipt = {
        "coverage_verified": True,
        "source_unchanged": True,
        "outcomes_consulted": False,
        "sealed_opened": False,
        "recipient": "alice",
        "group_map_sha256": sha(p),
    }
    with (output / "verification.json").open("x") as f:
        json.dump(receipt, f)
    (output / "verification.json").chmod(PRIVATE_MODE)
    return p, receipt
