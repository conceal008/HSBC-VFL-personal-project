"""Metadata allocation ignores source outcomes and bootstrap retains whole clusters."""

from pathlib import Path
import sys
import json
import csv
import hashlib
import yaml
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import portfolio_groups as g  # noqa: E402
import portfolio_resampling as res  # noqa: E402


def write(path, columns, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)


@pytest.mark.parametrize("kind", ["mock", "product", "csv"])
def test_key_projection_and_map_are_outcome_independent(tmp_path, monkeypatch, kind):
    raw = tmp_path / "raw"
    raw.mkdir()
    provisioned = tmp_path / "allocated"
    own = provisioned / "alice"
    own.mkdir(parents=True)
    spec = {
        "kind": kind,
        "key": "id",
        "time": "date",
        "columns": ["id", "date", "outcome"],
        "source": "source.csv",
        "guest": "source.csv",
        "guest_columns": ["id", "date", "outcome"],
        "split_group": "source_sample_id",
    }
    cfg = {"seeds": {"sampling": 11}, "datasets": {"unit": spec}}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    source = raw / "source.csv"
    write(
        source,
        spec["columns"],
        [
            ["u", "2015-12-01", "POISONED_UNKNOWN_LABEL"],
            ["u", "2016-01-01", "DO_NOT_CONSULT"],
        ],
    )
    monkeypatch.setattr(g, "rank", lambda *a: 0)
    keys = (
        ["u:1"]
        if kind == "csv"
        else [f"u:{g.month_number('2015-12-01')}->{g.month_number('2016-01-01')}"]
    )
    ids = [hashlib.sha256(("unit:" + k).encode()).hexdigest() for k in keys]
    write(own / "train.csv", ["record_id", "y"], [])
    write(own / "validation.csv", ["record_id", "y"], [[k, "UNKNOWN"] for k in ids])
    # A sealed file must never be opened by this metadata step.
    (own / "confirmation_raw.csv").write_text("INACCESSIBLE_SEALED")
    (own / "confirmation_raw.csv").chmod(0)
    (provisioned / "provisioning.json").write_text(
        json.dumps({"source_sha256": {"source.csv": g.sha(source)}})
    )
    target, receipt = g.build(raw, provisioned, "unit", p, tmp_path / "groups")
    assert (
        not receipt["sealed_opened"]
        and not receipt["outcomes_consulted"]
        and receipt["recipient"] == "alice"
    )
    assert (
        set(json.loads(target.read_text())) == set(ids)
        and target.stat().st_mode & 0o777 == 0o600
    )
    group = res.load_groups(target, receipt["group_map_sha256"], ids)
    assert len(group) == 1
    with pytest.raises(ValueError, match="changed"):
        res.load_groups(target, "invalid", ids)
    with pytest.raises(ValueError, match="coverage"):
        res.load_groups(target, receipt["group_map_sha256"], ["other"])
    source.write_text("altered")
    with pytest.raises(ValueError, match="changed"):
        g.build(raw, provisioned, "unit", p, tmp_path / "tamper")


def test_metadata_schema_and_undeclared_case_fail_closed(tmp_path):
    p = tmp_path / "source"
    p.write_text("id,date\nu\n")
    with pytest.raises(ValueError, match="malformed"):
        list(g.key_metadata(p, ["id", "date"], "id", "date"))
    with pytest.raises(ValueError, match="schema"):
        list(g.key_metadata(p, ["id"], "id"))
    cfg = tmp_path / "cfg"
    cfg.write_text(yaml.safe_dump({"datasets": {"unit": {"kind": "csv"}}}))
    with pytest.raises(ValueError, match="not declared"):
        g.build(tmp_path, tmp_path, "unit", cfg, tmp_path / "out")


def test_cluster_draw_preserves_whole_units_and_label_support():
    clusters = np.array(["a", "a", "b", "c", "c", "d", "e"])
    y = np.array([0, 1, 1, 0, 0, 0, 1])
    rng = np.random.default_rng(11)
    for _ in range(40):
        idx = res.cluster_draw(clusters, y, rng)
        assert set(y[idx]) == {0, 1}
        for key in np.unique(clusters):
            own = np.flatnonzero(clusters == key)
            counts = [np.sum(idx == i) for i in own]
            assert len(set(counts)) == 1
    with pytest.raises(ValueError, match="support"):
        res.cluster_draw(clusters, y[:-1], rng)
    with pytest.raises(ValueError, match="support"):
        res.cluster_draw(clusters, np.ones(len(y)), rng)
