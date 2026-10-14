"""Synthetic temporal/alignment/streaming controls, never the real sealed cohort."""

from pathlib import Path
import csv
import sys
import yaml
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import portfolio_source as ps  # noqa: E402


def write(path, columns, rows, delimiter=","):
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter=delimiter)
        if columns:
            w.writerow(columns)
        w.writerows(rows)


def config(spec):
    return {
        "seeds": {"sampling": 11, "split": 22},
        "max_cohort_rows": 30,
        "split_fractions": [0.6, 0.2, 0.2],
        "datasets": {"unit": spec},
    }


def test_reservoir_independent_of_labels_and_immutable_owners(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    columns = ["id", "a", "b", "y", "t"]
    source = raw / "s.csv"
    write(
        source,
        columns,
        [["PRIVATE_" + str(i), i, -i, i % 2, (i // 2) % 2] for i in range(200)],
    )
    s = {
        "kind": "csv",
        "source": "s.csv",
        "columns": columns,
        "label": "y",
        "treatment": "t",
        "key": "id",
        "scope": "synthetic unit",
        "parties": {
            "alice": {"features": ["a"], "categorical": []},
            "bob": {"features": ["b"], "categorical": []},
        },
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(config(s)))
    before = ps.sha(source)
    a = tmp_path / "new"
    receipt = ps.provision(raw, a, "unit", p)
    assert not receipt["selection_uses_outcomes"] and ps.sha(source) == before
    for party in ps.PARTIES:
        for split in ["train", "validation"]:
            text = (a / party / (split + ".csv")).read_text()
            assert "PRIVATE_" not in text
            assert (",y,t" in text) == (party == "alice")
        assert (a / party / "confirmation_raw.csv").stat().st_mode & 0o777 == 0
    write(
        source,
        columns,
        [["PRIVATE_" + str(i), i, -i, (i + 1) % 2, (i // 2) % 2] for i in range(200)],
    )
    b = tmp_path / "second"
    ps.provision(raw, b, "unit", p)
    assert (a / "bob/train.csv").read_bytes() == (b / "bob/train.csv").read_bytes()
    with pytest.raises(FileExistsError):
        ps.provision(raw, a, "unit", p)
    with pytest.raises(ValueError, match="pollute"):
        ps.provision(raw, raw / "new", "unit", p)
    s["source"] = "../cfg.yaml"
    p.write_text(yaml.safe_dump(config(s)))
    with pytest.raises(ValueError, match="traversal"):
        ps.provision(raw, tmp_path / "bad", "unit", p)


def test_orange_chunks_preserve_first_data_row_and_check_outcomes(tmp_path):
    s = {
        "kind": "orange",
        "width": 3,
        "sources": ["first", "next"],
        "labels": "labels",
        "label": "y",
    }
    write(tmp_path / "first", ["Var1", "Var2", "Var3"], [["", "2", "3"]], "\t")
    write(tmp_path / "next", None, [["4", "", "6"], ["7", "8", "9"]], "\t")
    (tmp_path / "labels").write_text("-1\n1\n-1\n")
    got = list(ps.rows(tmp_path, s, {}))
    assert len(got) == 3 and got[1][3]["Var1"] == "4" and got[0][3]["y"] == "0"
    assert ps.input_files(s) == ["first", "next", "labels"]
    (tmp_path / "labels").write_text("1\n")
    with pytest.raises(ValueError, match="outcome"):
        list(ps.rows(tmp_path, s, {}))
    (tmp_path / "labels").write_text("1\n1\n1\n1\n")
    with pytest.raises(ValueError, match="outnumber"):
        list(ps.rows(tmp_path, s, {}))
    (tmp_path / "first").write_text("badheader\n")
    with pytest.raises(ValueError, match="header"):
        list(ps.rows(tmp_path, s, {}))


def panel_spec():
    return {
        "key": "id",
        "time": "date",
        "label": "y",
        "products": ["p_ult1"],
        "mock_outcome_policy": "derive_next_product_transition",
        "parties": {"bob": {"features": ["p_ult1"], "categorical": []}},
    }


def test_panel_features_precede_outcome_and_original_label_is_not_ground_truth(
    monkeypatch,
):
    s = panel_spec()
    c = config(s)
    rows = [
        {
            "id": "u",
            "date": "2015-12-28",
            "p_ult1": "0",
            "label_new_any": "1",
            "a": "past",
        },
        {
            "id": "u",
            "date": "2016-01-28",
            "p_ult1": "1",
            "label_new_any": "0",
            "a": "future",
        },
    ]
    got = list(ps.panel_pairs(rows, s, c, mock=True))
    assert (
        got[0][3]["a"] == "past"
        and got[0][3]["p_ult1"] == "0"
        and got[0][3]["y"] == "1"
    )
    assert got[0][2] == "2016-01"
    s.pop("mock_outcome_policy")
    with pytest.raises(ValueError, match="Unverified"):
        list(ps.panel_pairs(rows, s, c, mock=True))
    s["mock_outcome_policy"] = "derive_next_product_transition"
    with pytest.raises(ValueError, match="Duplicate"):
        list(ps.panel_pairs([rows[0], rows[0]], s, c, mock=True))
    with pytest.raises(ValueError, match="chronologically"):
        list(ps.panel_pairs(rows[::-1], s, c, mock=True))
    missing = [dict(rows[0], p_ult1="NA"), rows[1]]
    assert list(ps.panel_pairs(missing, s, c, mock=True)) == []
    monkeypatch.setattr(ps, "rank", lambda *a: 0)
    assert list(ps.panel_pairs(rows, s, c))[0][3]["y"] == "1"
    monkeypatch.setattr(ps, "rank", lambda *a: 1 << ps.HASH_BITS)
    assert list(ps.panel_pairs(rows, s, c)) == []


def test_temporal_customer_groups_and_source_schema_fail_closed(tmp_path, monkeypatch):
    c = config({})
    s = {
        "temporal": {
            "train_last_target": "2015-12",
            "validation_targets": ["2016-01"],
            "confirmation_targets": ["2016-03"],
        }
    }
    for u, group, month in [
        (0.1, "train", "2015-12"),
        (0.7, "validation", "2016-01"),
        (0.9, "confirmation_raw", "2016-03"),
    ]:
        monkeypatch.setattr(ps, "rank", lambda *a, u=u: int(u * (1 << 64)))
        assert ps.split_for("record", "customer", month, c, s) == group
        assert ps.split_for("other_record", "customer", "2017-01", c, s) is None
    with pytest.raises(ValueError):
        ps.month_number("2016-13-01")
    assert ps.month_number("2016-01-01") - ps.month_number("2015-12-01") == 1
    p = tmp_path / "csv"
    p.write_text("a,b\n1\n")
    with pytest.raises(ValueError, match="Malformed"):
        list(ps.checked_csv(p, ["a", "b"]))
    with pytest.raises(ValueError, match="schema"):
        list(ps.checked_csv(p, ["a"]))
    with pytest.raises(ValueError, match="format"):
        list(ps.rows(tmp_path, {"kind": "unknown"}, c))
    assert ps.feature_spec(
        {"width": 5, "parties": {"alice": {"var_start": 1, "var_step": 2}}}, "alice"
    )["features"] == ["Var1", "Var3", "Var5"]


def test_mock_owner_guard_and_product_reader(tmp_path, monkeypatch):
    s = dict(
        panel_spec(),
        kind="mock",
        guest="g",
        host="h",
        guest_columns=["id", "date", "a"],
        host_columns=["id", "p_ult1"],
    )
    write(
        tmp_path / "g",
        s["guest_columns"],
        [["u", "2015-12-01", "old"], ["u", "2016-01-01", "new"]],
    )
    write(tmp_path / "h", s["host_columns"], [["u", 0], ["u", 1]])
    assert list(ps.rows(tmp_path, s, config(s)))[0][3]["a"] == "old"
    assert ps.input_files(s) == ["g", "h"]
    write(tmp_path / "h", s["host_columns"], [["wrong", 0]])
    with pytest.raises(ValueError, match="identities"):
        list(ps.rows(tmp_path, s, config(s)))
    s.update(kind="product", source="p", columns=["id", "date", "p_ult1"])
    write(
        tmp_path / "p", s["columns"], [["u", "2015-12-01", 0], ["u", "2016-01-01", 1]]
    )
    monkeypatch.setattr(ps, "rank", lambda *a: 0)
    assert list(ps.rows(tmp_path, s, config(s)))[0][3]["y"] == "1"
    assert ps.input_files(s) == ["p"]
