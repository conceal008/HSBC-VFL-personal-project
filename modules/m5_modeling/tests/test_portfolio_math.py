"""Column and common-initialization controls, separate from actual MPC acceptance."""

from pathlib import Path
import sys
from types import ModuleType
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import portfolio_math as pm  # noqa: E402


@pytest.mark.parametrize("uplift", [False, True])
def test_common_coordinates_and_zero_new_basis(uplift):
    base = pm.initial_weights(2, 3, 0, uplift, 11, 0.01, True)
    extra = pm.initial_weights(2, 3, 2, uplift, 11, 0.01, True)
    np.testing.assert_array_equal(extra[[0, 1, 4, 5, 6, -1]], base[[0, 1, 2, 3, 4, -1]])
    np.testing.assert_array_equal(extra[2:4], np.zeros((2, 1)))
    indices, dimension = pm.mapped_indices(4, 3, uplift)
    np.testing.assert_array_equal(
        pm.initial_weights(2, 3, 2, uplift, 11, 0.01, False), extra[indices]
    )
    assert dimension == extra.shape[0]
    if uplift:
        np.testing.assert_array_equal(extra[7], base[5])
        np.testing.assert_array_equal(
            extra[np.array([0, 1, 4, 5, 6]) + 8], base[np.arange(5) + 6]
        )


def test_binary_counterfactuals_and_padding(monkeypatch):
    jax = ModuleType("jax")
    jax.numpy = np
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", np)
    a = np.array([[2.0], [3.0]], dtype=np.float32)
    b = np.array([[5.0], [7.0]], dtype=np.float32)
    t = np.array([0, 1])
    expected = np.array([[2, 5, 0, 0, 0, 1], [3, 7, 1, 3, 7, 1], [0, 0, 0, 0, 0, 0]])
    got = pm.chunk_design(a, b, None, None, t, "L3_secure", True, -1, 1)
    np.testing.assert_array_equal(got, expected)
    for potential in [0, 1]:
        local = pm.local_design(np.c_[a, b], t, True, potential)
        secure = pm.chunk_design(a, b, None, None, t, "L3_secure", True, potential, 0)
        np.testing.assert_array_equal(secure[:, :-1], local)
        np.testing.assert_array_equal(local[:, 2], np.full(2, potential))
    np.testing.assert_array_equal(pm.local_design(a, t, False), a)
    assert pm.chunk_design(a, b, None, None, t, "L3_secure", False, -1, 0).shape == (
        2,
        3,
    )
    with pytest.raises(ValueError, match="permits"):
        pm.chunk_design(a, b, None, None, t, "L1_secure", True, -1, 0)
