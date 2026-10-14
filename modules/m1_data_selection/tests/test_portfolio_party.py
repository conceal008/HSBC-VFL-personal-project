"""Check fixed hashing, token parsing, bounded width and train-only state."""

from pathlib import Path
import csv
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import portfolio_party as pp  # noqa: E402


def write(p, fields, rows):
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(rows)


def test_fixed_hybrid_and_wide_encoding_without_population_fit(tmp_path):
    encoding = {
        "seed": 13,
        "hash_dimensions": 8,
        "wide_dimensions": 4,
        "wide_threshold": 3,
    }
    spec = {"features": ["x", "c"], "categorical": ["c"], "token_lists": True}
    p = tmp_path / "input.csv"
    write(
        p,
        ["record_id", "x", "c", "y"],
        [["a", "2", "[123,456]", 0], ["b", "NA", "[]", 1]],
    )
    a = pp.read_encoded(p, spec, ["y"], encoding)
    assert a[1].shape == (2, 9) and a[1][0, 0] == 2 and np.isnan(a[1][1, 0])
    assert a[2] == {"y": [0, 1]} and a[3][0] == "numeric__x"
    b = pp.read_encoded(p, spec, ["y"], encoding)
    assert np.array_equal(a[1], b[1], equal_nan=True)
    # Non-numeric fallback remains stateless, unseen categories have a defined mapping.
    write(
        p,
        ["record_id", "x", "c", "y"],
        [["a", "date-token", "[1 2]", 0], ["b", "3", "NA", 1]],
    )
    assert pp.read_encoded(p, spec, ["y"], encoding)[1].shape == (2, 9)
    encoding["wide_threshold"] = 1
    matrix = pp.read_encoded(p, spec, ["y"], encoding)[1]
    assert (
        matrix.shape == (2, 4)
        and np.isfinite(matrix).all()
        and (np.linalg.norm(matrix, axis=1) <= 1.00001).all()
    )


@pytest.mark.parametrize(
    "failure", ["schema", "duplicate", "empty", "target", "nonfinite", "malformed"]
)
def test_owner_stream_fail_closed(tmp_path, failure):
    p = tmp_path / "input.csv"
    fields = ["record_id", "x", "y"]
    rows = [["a", "1", 0], ["b", "2", 1]]
    if failure == "schema":
        fields[1] = "foreign"
    elif failure == "duplicate":
        rows[1][0] = "a"
    elif failure == "empty":
        rows = []
    elif failure == "target":
        rows[0][-1] = 2
    elif failure == "nonfinite":
        rows[0][1] = "inf"
    else:
        rows[0] = ["a"]
    write(p, fields, rows)
    with pytest.raises(ValueError):
        pp.read_encoded(
            p,
            {"features": ["x"], "categorical": []},
            ["y"],
            {
                "seed": 1,
                "hash_dimensions": 4,
                "wide_dimensions": 4,
                "wide_threshold": 2,
            },
        )


def test_tokens_and_numbers():
    assert pp.tokens("[]", True) == ["__EMPTY__"]
    assert pp.tokens("NA", True) == ["__MISSING__"]
    assert pp.tokens("one", False) == ["one"]
    assert pp.numeric("date") is None and np.isnan(pp.numeric("NA"))
    assert pp.bucket("field", "token", 11, 8) == pp.bucket("field", "token", 11, 8)
