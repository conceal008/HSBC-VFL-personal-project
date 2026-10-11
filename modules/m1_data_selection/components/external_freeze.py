"""Trusted public-fixture provisioning; no statistics, fitting or label-dependent selection.

Sealed outcomes are copied only for provisioning. Neither party's preparation
code receives read permission to the sealed raw files. This is not an assertion
that a trusted administrator or a previous human has never read the public data.
"""

from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import shutil
import yaml

PARTIES = ("alice", "bob")
DEVELOPMENT = ("train", "validation")
CONFIRMATION = "confirmation_raw"
SHA_CHUNK_BYTES = 1024 * 1024
BITS_PER_BYTE = 8
PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(SHA_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def decision(dataset, index, cfg, spec):
    key = f"{dataset}:{index}"
    limit = 1 << cfg["hash_bits"]

    def uniform(salt):
        digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
        return int.from_bytes(digest[: cfg["hash_bits"] // BITS_PER_BYTE], "big") / limit

    if uniform(cfg["seeds"]["sampling"]) >= spec["sampling_fraction"]:
        return None, None
    u = uniform(cfg["seeds"]["split"])
    if "official_confirmation" in spec:
        split = "train" if u < spec["dev_train_fraction"] else "validation"
    else:
        train, val, _ = cfg["split_fractions"]
        split = (
            "train"
            if u < train
            else ("validation" if u < train + val else CONFIRMATION)
        )
    return hashlib.sha256(key.encode()).hexdigest(), split


def source_rows(path, spec):
    with path.open() as stream:
        if spec["format"] == "csv":
            reader = csv.DictReader(stream)
            if reader.fieldnames != spec["columns"]:
                raise ValueError("Source schema differs")
            yield from reader
        else:
            for line in stream:
                values = line.split()
                if len(values) != len(spec["columns"]):
                    raise ValueError("Source width differs")
                yield dict(zip(spec["columns"], values))


def provision(raw_root, output, dataset, config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    spec = cfg["datasets"][dataset]
    raw_root, output = Path(raw_root).resolve(), Path(output).resolve()
    if output == raw_root or raw_root in output.parents:
        raise ValueError("Output would pollute original source")
    output.mkdir(parents=True, exist_ok=False, mode=PRIVATE_DIRECTORY_MODE)
    source = (raw_root / spec["source"]).resolve()
    if raw_root not in source.parents:
        raise ValueError("Source traversal denied")
    before = sha(source)
    streams = []
    writers = {}
    files = {}
    try:
        for party in PARTIES:
            own = output / party
            own.mkdir(mode=PRIVATE_DIRECTORY_MODE)
            outcomes = [spec["label"]] if party == "alice" else []
            if party == "alice" and spec["treatment"]:
                outcomes.append(spec["treatment"])
            fields = ["record_id"] + spec["parties"][party]["features"] + outcomes
            splits = (
                DEVELOPMENT
                if "official_confirmation" in spec
                else (*DEVELOPMENT, CONFIRMATION)
            )
            for split in splits:
                path = own / f"{split}.csv"
                s = path.open("x", newline="")
                streams.append(s)
                writer = csv.DictWriter(s, fieldnames=fields)
                writer.writeheader()
                writers[party, split] = writer
                files[party, split] = path
        for index, row in enumerate(source_rows(source, spec)):
            key, split = decision(dataset, index, cfg, spec)
            if key is None:
                continue
            row["record_id"] = key
            for party in PARTIES:
                writer = writers[party, split]
                writer.writerow({k: row[k] for k in writer.fieldnames})
    finally:
        for stream in streams:
            stream.close()
    if sha(source) != before:
        raise ValueError("Original source changed")
    official = None
    if "official_confirmation" in spec:
        official = {}
        # Byte-for-byte copy and hash without parsing official holdout outcomes.
        for kind, rel in spec["official_confirmation"].items():
            original = (raw_root / rel).resolve()
            if raw_root not in original.parents:
                raise ValueError("Source traversal denied")
            destination = output / f"official_{kind}.sealed"
            shutil.copyfile(original, destination)
            official[kind] = {
                "file": destination.name,
                "sha256": sha(destination),
                "parsed": False,
            }
            destination.chmod(0o000)
    manifest = {f"{p}/{s}.csv": sha(f) for (p, s), f in files.items()}
    for (party, split), path in files.items():
        path.chmod(0o000 if split == CONFIRMATION else PRIVATE_FILE_MODE)
    record = {
        "dataset": dataset,
        "source_sha256": before,
        "config_sha256": sha(config_path),
        "manifest": manifest,
        "official_confirmation": official,
        "confirmation_used": False,
        "selection_uses_outcomes": False,
        "fit_performed": False,
    }
    (output / "provisioning.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
