"""Independent objective gradient and checkpoint fail-closed tests on synthetic data."""

from pathlib import Path
import copy
import sys
import numpy as np
import pytest
import yaml

COMPONENTS = Path(__file__).resolve().parents[1] / "components"
sys.path.insert(0, str(COMPONENTS))
import joint_budget as jb  # noqa: E402
from test_active_diagnostic import native_doubles  # noqa: E402


def config():
    return yaml.safe_load((COMPONENTS.parent / "configs/joint_budget.yaml").read_text())


def test_gradient_matches_independent_df_objective_and_excludes_tail(monkeypatch):
    native_doubles(monkeypatch)
    x = np.array([[1.0, 0.2, 1.0], [-0.3, 1.0, 1.0], [0.1, -0.2, 1.0], [0.0, 0.0, 0.0]])
    y = np.array([0.0, 1.0, 1.0, 0.0])
    w = np.array([[0.2], [-0.1], [0.04]])

    def objective(q):
        z = (x[:3] @ q).ravel()
        return np.sum(0.5 * z + 0.5 * (np.abs(z) - np.log1p(np.abs(z))) - y[:3] * z)

    epsilon = 1e-6
    numeric = []
    for k in range(len(w)):
        a, b = w.copy(), w.copy()
        a[k] += epsilon
        b[k] -= epsilon
        numeric.append((objective(a) - objective(b)) / (2 * epsilon))
    g = jb.gradient_sum(x, y, w, 3)
    assert g.shape == w.shape
    np.testing.assert_allclose(g.ravel(), numeric, atol=1e-8)
    x[-1] = 999
    y[-1] = 999
    np.testing.assert_allclose(jb.gradient_sum(x, y, w, 3), g)
    assert jb.gradient_accept(np.zeros_like(w), np.zeros_like(w), 3, 0.1, 512, 0.001)
    assert not jb.gradient_accept(np.ones_like(w), w, 3, 0.1, 512, 0.001)


def test_schedule_and_factor_cannot_change():
    cfg = config()
    jb.validate(cfg)
    for mutate in [
        lambda c: c["shared"].update(learning_rates=[0.3]),
        lambda c: c["diagnostic"].update(checkpoints=[12, 48]),
        lambda c: c["experiment"]["conditions"]["D_B0"].update(mask=True),
        lambda c: c["experiment"].update(max_rows={"train": 100, "validation": 100}),
    ]:
        bad = copy.deepcopy(cfg)
        mutate(bad)
        with pytest.raises(ValueError):
            jb.validate(bad)


def test_checkpoint_matches_original_or_stops_and_preserves_failure(tmp_path):
    root = tmp_path
    (root / "input/references").mkdir(parents=True)
    pred = {s: np.array([[0.2], [0.8]]) for s in jb.SPLITS}
    for s in jb.SPLITS:
        np.save(root / "input/references" / f"B0_seed11_{s}.npy", pred[s])
    data = {"root": str(root), "shared": config()["shared"]}
    name = jb.record_checkpoint(data, "D_B0_seed11", 12, pred, "D_B0", 11, 0.001, False)
    assert name == "D_B0_seed11/checkpoints/e12"
    with pytest.raises(FileExistsError):
        jb.record_checkpoint(data, "D_B0_seed11", 12, pred, "D_B0", 11, 0.001, False)
    bad = {s: 1 - p for s, p in pred.items()}
    with pytest.raises(ValueError, match="reference differs"):
        jb.record_checkpoint(
            data, "failed_D_B0_seed11", 12, bad, "D_B0", 11, 0.001, True
        )
    assert (
        root
        / "trainings/failed_D_B0_seed11/checkpoints/e12/results/train_predictions.npy"
    ).is_file()
