"""Post-run lineage audit; disclose only booleans from each owner VM."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import external_training_vm as provider  # type: ignore[import-not-found]


def owner_script(root):
    """Verify retained files without returning predictions, shares or identifiers."""
    return f'''
import hashlib,json
from pathlib import Path
p=Path({root!r})
r=json.loads((p/'code/runtime.json').read_text())
m=json.loads((p/'code/recovery_manifest.json').read_text())
old=p.parent/r['recovery_source_run']
assert m['source_run']==r['recovery_source_run'] and m['party']==r['party']
assert m['config_sha256']==r['config_sha256']
assert set(r['reused_completed_fits']).issubset(m['reused_fits'])
assert all(hashlib.sha256((p/'code'/n).read_bytes()).hexdigest()==h for n,h in r['controller_sha256'].items())
assert all(hashlib.sha256((p/'trainings'/n/rel).read_bytes()).hexdigest()==h and hashlib.sha256((old/'trainings'/n/rel).read_bytes()).hexdigest()==h for n,files in m['fit_sha256'].items() for rel,h in files.items())
assert all(hashlib.sha256((p/'input'/n).read_bytes()).digest()==hashlib.sha256((old/'input'/n).read_bytes()).digest() for n in ('train.csv','validation.csv'))
print('passed')
'''


def verify_lineage(target):
    target = Path(target)
    base = provider.verify(target)
    state = json.loads((target / "status.json").read_text())
    if state["exit_codes"] != [0, 0] or state["cleanup_errors"]:
        raise ValueError("Owner exits/cleanup not verified")
    recovery = (target / "recovery_declaration.json").exists()
    if recovery:
        declaration = json.loads((target / "recovery_declaration.json").read_text())
        for party in provider.vm.PARTIES:
            own = target / party
            runtime = json.loads((own / "code/runtime.json").read_text())
            arrays = json.loads((own / "logs/recovery_arrays.json").read_text())
            if not all(arrays.values()) or not arrays.get("owner_arrays_exact"):
                raise ValueError("Recovery owner arrays differ")
            if runtime["recovery_source_run"] != declaration["source_run"] or any(
                provider.sha(own / "code" / name) != digest
                for name, digest in runtime["controller_sha256"].items()
            ):
                raise ValueError("Recovery controller/source differs")
            root = f"/srv/vfl/{party}/{target.name}"
            result = provider.vm.guest(
                party,
                "sudo /opt/secretflow/bin/python -c " + shlex.quote(owner_script(root)),
            )
            if result != "passed":
                raise ValueError("Owner-local retained lineage changed")
    return dict(
        base,
        owner_exits_verified=True,
        recovery_applied=recovery,
        recovery_lineage_verified=recovery,
        original_interrupted_exits="unknown" if recovery else "not_applicable",
        plaintext_peer_data_read=False,
    )
