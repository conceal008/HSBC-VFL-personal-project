"""Synthetic owner-only controls and independent oracle negative cases."""

from pathlib import Path
import sys
import types
import json
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "m5_modeling/components"))
import secret_join_run as run  # noqa: E402


@pytest.fixture
def numpy_backend(monkeypatch):
    fake = types.ModuleType("jax")
    fake.numpy = np
    monkeypatch.setitem(sys.modules, "jax", fake)
    monkeypatch.setitem(sys.modules, "jax.numpy", np)
    monkeypatch.setattr(run.join, "secret_join", run.join.reference_join)


def test_known_controls_and_oracle_rejects_forged_peer(numpy_backend):
    assert run.known_controls()
    a = run.join.encode_keys(["a", "b"])
    b = run.join.encode_keys(["b", "x"])
    result = list(run.join.reference_join(a, b))
    assert run.invariant(a, b, result)
    result[1][1] = 1
    assert not run.invariant(a, b, result)
    a = np.concatenate([a, a[:1]], axis=0)
    assert not run.invariant(
        a, b, (np.ones(3, dtype=bool), np.zeros(3), np.ones(2, dtype=bool), np.zeros(2))
    )


def test_owner_padded_input_private_and_frozen(tmp_path, monkeypatch):
    root = tmp_path / "alice/run"
    inp = root / "input"
    inp.mkdir(parents=True)
    (root / "trainings").mkdir()
    (root / "code").mkdir()
    (root / "code/marker.py").write_text("pass")
    (inp / "keys.json").write_text(json.dumps(["a", "b", "c", "d"]))
    (inp / "manifest.json").write_text(
        json.dumps({"keys.json": run.bp.sha(inp / "keys.json")})
    )
    monkeypatch.setattr(run.bp.ft, "ROOT", tmp_path)
    monkeypatch.setattr(run.bp.ft, "isolation_preflight", lambda p: None)
    original = (inp / "keys.json").read_bytes()
    cfg = {"rows_per_owner": 3, "keep_probability": 0.8}
    runtime = {"run_id": "run"}
    data = run.owner_input("alice", runtime, cfg, 101)
    assert data.shape == (3, 8) and (inp / "keys.json").read_bytes() == original
    assert (root / "trainings/join_seed101/data/construction.json").exists()
    with pytest.raises(FileExistsError):
        run.owner_input("alice", runtime, cfg, 101)
    cfg["rows_per_owner"] = 5
    with pytest.raises(ValueError, match="Insufficient"):
        run.owner_input("alice", runtime, cfg, 102)
    (inp / "keys.json").write_text('["changed"]')
    with pytest.raises(ValueError, match="Frozen"):
        run.owner_input("alice", runtime, cfg, 102)


@pytest.mark.parametrize("failure", [None, "known", "oracle", "code", "output"])
def test_run_reveals_only_correctness_and_dumps_owner_shares(
    monkeypatch, tmp_path, failure
):
    from contextlib import contextmanager

    cfg = {
        "seeds": [101, 102, 103, 104, 105],
        "rows_per_owner": 64,
        "output": "plaintext" if failure == "output" else "secret_shares",
    }
    root = tmp_path / "alice/run"
    (root / "code").mkdir(parents=True)
    cfg_path = root / "code/protocol.json"
    cfg_path.write_text(json.dumps(cfg))
    (root / "code/helper.py").write_text("pass")
    runtime = {
        "party": "alice",
        "run_id": "run",
        "config": str(cfg_path),
        "config_sha256": run.bp.sha(cfg_path),
        "code_sha256": {
            "helper.py": "wrong"
            if failure == "code"
            else run.bp.sha(root / "code/helper.py")
        },
    }
    monkeypatch.setattr(run.bp.ft, "ROOT", tmp_path)
    disclosures = []
    dumped = []

    class Opaque:
        def to(self, *a):
            return self

    class Secure:
        def __call__(self, fn):
            if fn is run.known_controls:
                return lambda *a: failure != "known"
            if fn is run.invariant:
                return lambda *a: failure != "oracle"
            return lambda *a: Opaque()

        def dump(self, obj, paths):
            assert isinstance(obj, Opaque)
            for value in paths:
                path = Path(value)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic-share")
                dumped.append(path)

    class SF:
        @staticmethod
        def PYU(party):
            return lambda fn: lambda *a: Opaque()

        @staticmethod
        def reveal(value):
            assert isinstance(value, bool), "Intersection output must never be revealed"
            disclosures.append(value)
            return value

        @staticmethod
        def wait(value):
            pass

    @contextmanager
    def session(*a):
        yield SF, Secure()

    monkeypatch.setattr(run.bp, "session", session)
    (root / "results").mkdir()
    if failure:
        with pytest.raises(ValueError):
            run.run(runtime)
    else:
        report = run.run(runtime)
        assert (
            not report["membership_or_count_disclosed"]
            and len(disclosures) == 6
            and len(dumped) == 10
        )
        assert all(
            (p.stat().st_mode & 0o777) == 0o600 for p in dumped if "alice" in p.parts
        )
