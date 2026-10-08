"""Synthetic contract tests; mocked devices are never evidence of cryptographic PSI."""
import contextlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import yaml

COMP = Path(__file__).resolve().parents[1] / "components"
sys.path.insert(0, str(COMP.parent.parent / "m5_modeling/components"))
sys.path.insert(0, str(COMP))
import bounded_psi as bp  # noqa: E402
import bounded_psi_vm as provider  # noqa: E402


def jax_stub(monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(numpy=np))
    monkeypatch.setitem(sys.modules, "jax.numpy", np)


def test_public_domain_and_secret_mask_correctness(monkeypatch):
    jax_stub(monkeypatch)
    lookup = bp.domain("a" * 64, 8)
    keys = list(lookup)
    a = bp.membership(keys, lookup, 2, .8)
    b = bp.membership(keys, lookup, 3, .8)
    assert bp.known_answer() and bp.invariant(a, b, bp.intersection(a, b))
    assert set(a) <= {0, 1}
    assert not bp.invariant(a, b, np.zeros(9))
    for bad in ([keys[0], keys[0]], ["unknown"]):
        with pytest.raises(ValueError):
            bp.membership(bad, lookup, 2, .8)
    with pytest.raises(ValueError):
        bp.membership(keys, lookup, 2, 1)
    with pytest.raises(ValueError):
        bp.domain("short", 8)


class Object:
    def __init__(self, value):
        self.value = value
    def to(self, device):
        return self


class Device:
    def __call__(self, fn):
        return lambda *args: Object(fn(*(a.value if isinstance(a, Object) else a for a in args)))
    def dump(self, result, paths):
        for path in paths:
            Path(path).write_bytes(b"MOCK-SHARE-NOT-CRYPTO")


def fixture(tmp_path, monkeypatch):
    jax_stub(monkeypatch)
    monkeypatch.setattr(bp.ft, "ROOT", tmp_path)
    monkeypatch.setattr(bp.ft, "isolation_preflight", lambda p: {"synthetic": True})
    cfg = yaml.safe_load((COMP.parent / "configs/bounded_psi.yaml").read_text())
    for p in bp.PARTIES:
        root = tmp_path / p / "round"
        for s in ("code", "input", "trainings", "results"):
            (root / s).mkdir(parents=True)
        bp.save(root / "input/keys.json", list(bp.domain("a" * 64, 8)))
        bp.save(root / "input/manifest.json", {"keys.json": bp.sha(root / "input/keys.json")})
        (root / "code/config.yaml").write_text(yaml.safe_dump(cfg))
    root = tmp_path / "alice/round"
    runtime = dict(party="alice",run_id="round",source_sha256="a" * 64, key_namespace="a" * 64, universe_rows=8,
                   config=str(root / "code/config.yaml"),config_sha256=bp.sha(root / "code/config.yaml"),code_sha256={})
    sf = SimpleNamespace(PYU=lambda p: Device(),reveal=lambda o: o.value,wait=lambda o: None)
    @contextlib.contextmanager
    def session(r, c):
        yield sf, Device()
    monkeypatch.setattr(bp, "session", session)
    return root, runtime


def test_complete_synthetic_psi_and_fail_closed(tmp_path, monkeypatch):
    root, runtime = fixture(tmp_path, monkeypatch)
    result = bp.run(runtime)
    assert result["status"] == "passed" and not result["intersection_reconstructed"]
    assert len(list(tmp_path.rglob("intersection.share"))) == 10
    assert len(list(tmp_path.rglob("own_membership.npy"))) == 10
    assert not list(root.rglob("*test*"))
    with pytest.raises(FileExistsError):
        bp.run(runtime)
    with pytest.raises(ValueError):
        bp.run(dict(runtime,party="unapproved"))
    with pytest.raises(ValueError, match="fingerprint"):
        bp.run(dict(runtime,config_sha256="tamper"))
    bp.save(root / "input/manifest.json", {"test.json": "bad"})
    with pytest.raises(ValueError, match="Frozen"):
        bp.owner_input("alice",runtime,{},1)


def test_real_session_adapter_contract_and_failure_shutdown(tmp_path, monkeypatch):
    jax_stub(monkeypatch)
    calls = []
    sf = SimpleNamespace(init=lambda **kw: calls.append(kw),SPU=lambda *a,**kw: Device(),
                         wait=lambda o: None,shutdown=lambda **kw:calls.append(kw))
    monkeypatch.setitem(sys.modules,"secretflow",sf)
    monkeypatch.setitem(sys.modules,"spu",SimpleNamespace(logging=SimpleNamespace(LogOptions=SimpleNamespace)))
    monkeypatch.setattr(bp.ft,"install_pinned_tls_adapter",lambda:None)
    monkeypatch.setattr(bp.ft,"reject_anonymous_client",lambda *a:True)
    monkeypatch.setattr(bp.ft,"reject_anonymous_spu",lambda *a:True)
    runtime=dict(party="alice",run_id="round",parties={"alice":{"address":"synthetic"}},spu_addresses={p:p for p in bp.PARTIES})
    cfg=dict(link_timeout_ms=100,protocol="CHEETAH",field="FM64")
    with bp.session(runtime,cfg):
        pass
    assert calls[-1] == dict(barrier_on_shutdown=True,on_error=False)
    assert calls[0]["tls_config"] and not calls[0]["ray_mode"]
    monkeypatch.setattr(bp.ft,"reject_anonymous_spu",lambda *a:False)
    with pytest.raises(ValueError):
        with bp.session(runtime,cfg):
            pass
    assert calls[-1] == dict(barrier_on_shutdown=False,on_error=True)


def test_provider_inside_guest_key_extraction_and_exports(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    config = yaml.safe_load((COMP.parent.parent / "m1_data_selection/configs/uci_bank_preparation.yaml").read_text())
    raw = workspace / "数据集" / config["source"]["relative_path"]
    raw.parent.mkdir(parents=True)
    raw.write_text("synthetic;header\na;b\nc;d\n")
    commands=[]
    monkeypatch.setattr(provider.vm,"setup_certificates",lambda w:{})
    monkeypatch.setattr(provider.vm,"restrict_network",lambda *a:None)
    monkeypatch.setattr(provider.vm,"copy_to",lambda *a:None)
    monkeypatch.setattr(provider.vm,"guest",lambda p,s: commands.append((p,s)) or ("passed" if s.endswith("echo passed") else ""))
    monkeypatch.setattr(provider.subprocess,"check_output",lambda *a,**k:"synthetic\n")
    source=tmp_path / "uci_bank_marketing/round"
    target,run_id=provider.stage(workspace,source,tmp_path / "new",COMP.parent)
    assert all("'test'" not in s and "'label'" not in s for p,s in commands)
    class Process:
        returncode=0
        def __init__(self,*a,**k):
            pass
        def poll(self):
            return 0
    monkeypatch.setattr(provider.subprocess,"Popen",Process)
    def copy(*args):
        Path(args[-1]).write_text(json.dumps(dict(status="passed",code_cells=3)))
    monkeypatch.setattr(provider.vm,"lima",copy)
    provider.execute(target,run_id)
    assert not list(target.rglob("*.share")) and not list(target.rglob("*.npy"))


def test_provider_queue_error_and_success(tmp_path,monkeypatch):
    source=tmp_path / "source.json"
    bp.save(source,dict(status="passed",completed=["one","two"]))
    def stage(w,s,local,m):
        target=local/s
        target.mkdir()
        return target,"round"
    monkeypatch.setattr(provider,"stage",stage)
    monkeypatch.setattr(provider,"execute",lambda *a:None)
    assert provider.queue(tmp_path,source,tmp_path/"okay")["status"] == "passed"
    def fail(*a):
        raise ValueError("synthetic")
    monkeypatch.setattr(provider,"execute",fail)
    with pytest.raises(ValueError):
        provider.queue(tmp_path,source,tmp_path/"failure")
    assert json.loads((tmp_path/"failure/status.json").read_text())["status"] == "failed"
