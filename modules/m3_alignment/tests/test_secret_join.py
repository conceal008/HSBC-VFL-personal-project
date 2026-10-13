"""Only synthetic keys; no dataset paths or joined real identities."""

from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import secret_join as s  # noqa: E402


@pytest.mark.parametrize("keys", [[], [""], [1], ["a", "a"]])
def test_owner_rejects_empty_nonstrings_and_duplicates(keys):
    with pytest.raises(ValueError):
        s.encode_keys(keys)


def test_full_256_bits_and_opaque_identity():
    a = s.encode_keys(["a", "a ", "甲"])
    assert a.shape == (3, s.HASH_WORDS) and a.dtype == np.uint32
    assert not np.array_equal(a[0], a[1])
    np.testing.assert_array_equal(a, s.encode_keys(["a", "a ", "甲"]))


@pytest.mark.parametrize(
    "left,right",
    [
        (["a"], ["a"]),
        (["a"], ["b"]),
        (["a", "c", "b"], ["b", "x", "c"]),
        (["a", "b", "c", "d"], ["c", "a"]),
        (["甲", "乙"], ["乙", "甲"]),
    ],
)
def test_secure_function_matches_synthetic_oracle(left, right):
    a, b = s.encode_keys(left), s.encode_keys(right)

    def sort(values, dimension, num_keys):
        assert dimension == 0
        order = np.lexsort(tuple(values[:num_keys])[::-1])
        return tuple(value[order] for value in values)

    actual = s.join_kernel(a, b, np, sort)
    expected = s.reference_join(a, b)
    for x, y in zip(actual, expected):
        np.testing.assert_array_equal(np.asarray(x), y)


def test_oracle_rejects_duplicate_encoded_keys():
    a = s.encode_keys(["a"])
    with pytest.raises(ValueError):
        s.reference_join(np.concatenate([a, a]), a)
