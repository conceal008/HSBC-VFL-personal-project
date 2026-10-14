"""Trusted setup preflight for a single shared project router; no party data read."""

from __future__ import annotations
from pathlib import Path
import shlex
import subprocess
import functional_vm as vm

MANAGEMENT_PORT = 22
PROBE_TIMEOUT_SECONDS = 3


def router_pids(output, network_root):
    pids = []
    for line in output.splitlines():
        parts = line.strip().split(None, 1)
        if (
            len(parts) == 2
            and parts[1].startswith("/opt/homebrew/bin/limactl usernet ")
            and str(network_root) + "/" in parts[1]
        ):
            pids.append(int(parts[0]))
    return pids


def preflight():
    """Check management connectivity outside training UID, before native jobs start."""
    network_root = Path(vm.LIMA_HOME_PATH).resolve() / "_networks/user-v2"
    output = subprocess.check_output(["ps", "-axo", "pid,command"], text=True)
    pids = router_pids(output, network_root)
    if len(pids) != 1:
        raise RuntimeError("Project must have exactly one shared usernet router")
    for party in vm.PARTIES:
        peer = next(p for p in vm.PARTIES if p != party)
        code = (
            "import socket;"
            + f"s=socket.create_connection(('lima-hsbc-{peer}.internal',{MANAGEMENT_PORT}),timeout={PROBE_TIMEOUT_SECONDS});"
            + "s.close();print('passed')"
        )
        if (
            vm.guest(party, "sudo /opt/secretflow/bin/python -c " + shlex.quote(code))
            != "passed"
        ):
            raise RuntimeError("Management network positive control failed")
    return {
        "single_shared_router": True,
        "bidirectional_management_control": True,
        "training_firewall_relaxed": False,
        "mtls_relaxed": False,
    }


def enforce_preparer_denial():
    """Restore prep UID deny-all rules after VM reboot; training UID is separate."""
    for party in vm.PARTIES:
        account = f"vflprep_{party}"
        uid = vm.guest(party, f"id -u {account}")
        if not uid.isdigit() or int(uid) == 0:
            raise RuntimeError("Invalid prep identity")
        rule = f"-m owner --uid-owner {uid} -j REJECT"
        vm.guest(
            party,
            f"set -e; sudo iptables -C OUTPUT {rule} 2>/dev/null || sudo iptables -I OUTPUT 1 {rule}; sudo ip6tables -C OUTPUT {rule} 2>/dev/null || sudo ip6tables -I OUTPUT 1 {rule}",
        )
    return {"preparation_uid_deny_all_restored": True}
