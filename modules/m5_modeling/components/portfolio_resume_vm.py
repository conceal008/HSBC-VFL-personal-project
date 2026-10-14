"""Independent S5.X3 continuation; preserve X2 code/failures and frozen training math."""

from __future__ import annotations
import json
import hashlib
import yaml
from pathlib import Path
import re
import portfolio_phase_vm as phase

EXECUTION_STEP = "S5.X3"
BASE_STAGE = phase.stage
BASE_VERIFY = phase.verify
CONTINUATION_CONFIG = phase.MODULE / "configs/portfolio_continuation_control.yaml"
CONTINUATION = yaml.safe_load(CONTINUATION_CONFIG.read_text())
SOURCE_GUARD: dict[str, str] = {}


def healthy_source(source, dataset, seed, cfg):
    if source is None:
        return [], []
    source = Path(source)
    if source.parent.name != dataset or not re.fullmatch(
        dataset + r"_portfolio_training_[0-9]{8}T[0-9]+Z", source.name
    ):
        raise ValueError("Recovery source identity invalid")
    names = [f"{c}_seed{seed}" for c in phase.CELLS]
    secure = True
    checks = {}
    for party in phase.vm.PARTIES:
        helper = phase.MODULE / "components/portfolio_recovery_health.py"
        temporary = f"/tmp/{party}_recovery_health.py"
        phase.vm.copy_to(helper, party, temporary)
        # Trusted administrator inspects only this party's prior assets, exporting booleans.
        code = f"import runpy,json;m=runpy.run_path({temporary!r});print(json.dumps(m['inspect_block']('/srv/vfl/{party}/{source.name}',{names!r},{party!r},{cfg['shared']['epochs']},True)))"
        checks[party] = json.loads(phase.script(party, code))
        secure &= checks[party]["complete"]
    local = [f"{c}_seed{seed}" for c in phase.LOCAL_CELLS]
    code = f"import runpy,json;m=runpy.run_path('/tmp/alice_recovery_health.py');print(json.dumps(m['inspect_block']('/srv/vfl/alice/{source.name}',{local!r},'alice',{cfg['shared']['epochs']},False)))"
    local_check = json.loads(phase.script("alice", code))
    # Persist the eligibility diagnostics in the host's independent continuation log.
    if ELIGIBILITY_OUTPUT is not None:
        phase.vm.save(
            ELIGIBILITY_OUTPUT / f"{dataset}_seed{seed}_eligibility.json",
            {
                "source": str(source),
                "secure_checks": checks,
                "local_check": local_check,
                "secure_reused": bool(secure),
                "local_reused": bool(local_check["complete"]),
                "corruption_not_repaired": True,
            },
        )
    return (names if secure else []), (local if local_check["complete"] else [])


ELIGIBILITY_OUTPUT = None


def stage(*args, **kwargs):
    if any(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() != h
        for p, h in SOURCE_GUARD.items()
    ):
        raise ValueError("Frozen continuation source changed")
    BASE_STAGE(*args, **kwargs)
    target = Path(args[3])  # 魔数豁免: stage第四个参数是本轮输出目录的位置，不是实验超参
    run_id = args[2]
    for party in phase.vm.PARTIES:
        runtime = phase.read(target / party / "code/runtime.json")
        runtime["execution_step_id"] = EXECUTION_STEP
        runtime["mathematical_template_step"] = "S5.X2"
        runtime["owner_umask"] = CONTINUATION["owner_umask"]
        runtime["continuation_config_sha256"] = hashlib.sha256(
            CONTINUATION_CONFIG.read_bytes()
        ).hexdigest()
        root = f"/srv/vfl/{party}/{run_id}"
        # Only newly cloned copies are tightened; old frozen assets stay untouched.
        file_mode = format(CONTINUATION["copied_owner_file_mode"], "o")
        directory_mode = format(CONTINUATION["copied_owner_directory_mode"], "o")
        phase.vm.guest(
            party,
            f"sudo find {root}/trainings -type f -exec chmod {file_mode} {{}} \\;; sudo find {root}/trainings -type d -exec chmod {directory_mode} {{}} \\;",
        )
        phase.write_guest(target, party, run_id, "runtime.json", runtime)
    phase.vm.save(
        target / "continuation_step.json",
        {
            "step_id": EXECUTION_STEP,
            "mathematical_template_step": "S5.X2",
            "previous_step_status": "escalated",
            "does_not_clear_previous_failure": True,
        },
    )


def verify(target, phases=None):
    audit = BASE_VERIFY(target, phases)
    for party in phase.vm.PARTIES:
        run_id = Path(target).name
        code = (
            "from pathlib import Path;import stat;"
            + f"p=Path('/srv/vfl/{party}/{run_id}/trainings');"
            + f"assert all(stat.S_IMODE(f.stat().st_mode)=={CONTINUATION['copied_owner_file_mode']} for f in p.rglob('model.share'));print('passed')"
        )
        if phase.script(party, code) != "passed":
            raise ValueError("Owner model share permissions invalid")
    return dict(audit, owner_model_share_permissions_verified=True)


def queue(workspace, output, recovery=None):
    global ELIGIBILITY_OUTPUT, SOURCE_GUARD
    output = Path(output)
    ELIGIBILITY_OUTPUT = output.parent / (output.name + "_eligibility")
    ELIGIBILITY_OUTPUT.mkdir(mode=phase.vm.PRIVATE_DIRECTORY_MODE, exist_ok=False)
    files = [
        *(
            phase.MODULE / "components" / n
            for n in phase.original.CODE + phase.CONTROLLERS
        ),
        *(
            phase.MODULE / "components" / n
            for n in (
                "portfolio_training_vm.py",
                "portfolio_phase_vm.py",
                "portfolio_recovery_health.py",
                "portfolio_resume_vm.py",
                "portfolio_network.py",
                "portfolio_groups.py",
            )
        ),
        *(
            phase.MODULE / "configs" / n
            for n in ("portfolio_training.yaml", "portfolio_phase_control.yaml")
        ),
        *(
            phase.MODULE / "notebooks" / n
            for n in ("S5.X3_portfolio_phase.ipynb", "S5.X3_portfolio_merge.ipynb")
        ),
    ]
    files.append(CONTINUATION_CONFIG)
    SOURCE_GUARD = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    phase.vm.save(ELIGIBILITY_OUTPUT / "source_guard.json", SOURCE_GUARD)
    phase.complete_source = healthy_source
    phase.stage = stage
    phase.verify = verify
    phase.PHASE_NOTEBOOK = "S5.X3_portfolio_phase.ipynb"
    phase.MERGE_NOTEBOOK = "S5.X3_portfolio_merge.ipynb"
    try:
        return phase.queue(workspace, output, recovery)
    finally:
        phase.vm.private_permissions(ELIGIBILITY_OUTPUT)
