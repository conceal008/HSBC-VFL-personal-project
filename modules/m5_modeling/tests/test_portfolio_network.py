"""Shared-router check ignores unrelated environments and fails before training."""

from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import portfolio_network as p  # noqa: E402


def test_router_inventory_uses_exact_project_root():
    text = "11 /opt/homebrew/bin/limactl usernet -p /project/_networks/user-v2/router.pid\n12 /opt/homebrew/bin/limactl usernet -p /other/router.pid\n13 shell ps /project/_networks/user-v2/\n"
    assert p.router_pids(text, Path("/project/_networks/user-v2")) == [11]


@pytest.mark.parametrize("count", [0, 1, 2])
def test_network_requires_one_project_router_and_both_positive_controls(
    monkeypatch, tmp_path, count
):
    monkeypatch.setattr(p.vm, "LIMA_HOME_PATH", str(tmp_path))
    root = tmp_path / "_networks/user-v2"
    text = "\n".join(
        f"{i + 1} /opt/homebrew/bin/limactl usernet -p {root}/router.pid"
        for i in range(count)
    )
    monkeypatch.setattr(p.subprocess, "check_output", lambda *a, **k: text)
    calls = []
    monkeypatch.setattr(
        p.vm, "guest", lambda party, code: calls.append(party) or "passed"
    )
    if count == 1:
        assert p.preflight()["single_shared_router"] and calls == ["alice", "bob"]
    else:
        with pytest.raises(RuntimeError):
            p.preflight()
        assert not calls


def test_positive_control_failure_does_not_relax_firewall(monkeypatch, tmp_path):
    monkeypatch.setattr(p.vm, "LIMA_HOME_PATH", str(tmp_path))
    monkeypatch.setattr(
        p.subprocess,
        "check_output",
        lambda *a,
        **k: f"1 /opt/homebrew/bin/limactl usernet -p {tmp_path}/_networks/user-v2/router.pid",
    )
    calls = []
    monkeypatch.setattr(p.vm, "guest", lambda _, code: calls.append(code) or "failed")
    with pytest.raises(RuntimeError):
        p.preflight()
    assert calls and all("iptables" not in c for c in calls)


def test_reboot_restores_prep_denial_without_touching_training_rules(monkeypatch):
    calls = []

    def guest(party, code):
        calls.append((party, code))
        return "1001" if code.startswith("id -u") else ""

    monkeypatch.setattr(p.vm, "guest", guest)
    assert p.enforce_preparer_denial()["preparation_uid_deny_all_restored"]
    rules = [c for _, c in calls if not c.startswith("id -u")]
    assert len(rules) == 2 and all(
        "iptables" in c and "ip6tables" in c and "--uid-owner 1001 -j REJECT" in c
        for c in rules
    )
    assert all("-F" not in c and "-j ACCEPT" not in c for c in rules)
    monkeypatch.setattr(p.vm, "guest", lambda *a: "0")
    with pytest.raises(RuntimeError):
        p.enforce_preparer_denial()
