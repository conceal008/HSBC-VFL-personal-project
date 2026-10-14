"""Bounded native liveness and complete recovery blocks; no synthetic success claim."""

from pathlib import Path
import sys
import subprocess
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "components"))
import portfolio_phase_vm as p  # noqa: E402


def test_management_probe_has_timeout_and_both_parties(monkeypatch):
    calls = []
    monkeypatch.setattr(p.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    p.health()
    assert len(calls) == len(p.vm.PARTIES)
    assert all(
        c[1]["timeout"] == p.CONTROL["management_timeout_seconds"] and c[1]["check"]
        for c in calls
    )
    assert all(c[0][0][-1] == "true" for c in calls)


def test_failed_health_breaks_wait_without_waiting_native(monkeypatch, tmp_path):
    class Running:
        def poll(self):
            return None

    monkeypatch.setattr(
        p,
        "health",
        lambda: (_ for _ in ()).throw(subprocess.TimeoutExpired("management", 10)),
    )
    with pytest.raises(subprocess.TimeoutExpired):
        p.wait_native([Running()], tmp_path)


def test_native_failure_does_not_count_as_completed(monkeypatch, tmp_path):
    class Failed:
        returncode = 1

        def poll(self):
            return 1

    with pytest.raises(RuntimeError, match="exit"):
        p.wait_native([Failed()], tmp_path)


def test_reuse_needs_bilateral_pair_but_local_complete_block_is_independent(
    monkeypatch,
):
    cfg = yaml.safe_load((p.MODULE / "configs/portfolio_training.yaml").read_text())
    monkeypatch.setattr(
        p, "script", lambda party, code: "partial" if party == "bob" else "complete"
    )
    pair, local = p.complete_source(
        Path("homecredit/homecredit_portfolio_training_20261014T014214342444Z"),
        "homecredit",
        11,
        cfg,
    )
    assert not pair and len(local) == len(p.LOCAL_CELLS)
    monkeypatch.setattr(p, "script", lambda *a: "complete")
    pair, local = p.complete_source(
        Path("homecredit/homecredit_portfolio_training_20261014T014214342444Z"),
        "homecredit",
        11,
        cfg,
    )
    assert len(pair) == len(p.CELLS) and len(local) == len(p.LOCAL_CELLS)
    with pytest.raises(ValueError, match="identity"):
        p.complete_source(Path("../../sealed"), "homecredit", 11, cfg)


def test_no_recovery_source_means_no_claimed_reuse():
    assert p.complete_source(None, "homecredit", 11, {}) == ([], [])


def inventory(tmp_path, aggregate=False):
    cfg = yaml.safe_load((p.MODULE / "configs/portfolio_training.yaml").read_text())
    seeds = cfg["shared"]["model_seeds"] if aggregate else [11]
    target = tmp_path / "inventory"
    target.mkdir()
    p.vm.save(target / "status.json", {"status": "passed", "exit_codes": [0, 0]})
    digest = p.original.sha(p.MODULE / "configs/portfolio_training.yaml")
    p.vm.save(
        target / "declaration.json",
        {
            "dataset": "homecredit",
            "config_sha256": digest,
            "test_staged": False,
            "prediction_export": False,
        },
    )
    p.vm.save(target / "phase_declaration.json", {"selected_seed": 11})
    for party in p.vm.PARTIES:
        own = target / party
        for sub in ("code", "data", "logs", "results", "trainings"):
            (own / sub).mkdir(parents=True)
        (own / "code/nonlinear_full.yaml").write_bytes(
            (p.MODULE / "configs/portfolio_training.yaml").read_bytes()
        )
        hashes = {}
        for name in p.original.CODE + p.CONTROLLERS:
            (own / "code" / name).write_text("synthetic source only")
            hashes[name] = p.original.sha(own / "code" / name)
        p.vm.save(
            own / "code/runtime.json",
            {
                "config_sha256": digest,
                "code_sha256": {n: hashes[n] for n in p.original.CODE},
                "controller_sha256": {n: hashes[n] for n in p.CONTROLLERS},
            },
        )
        for name in ("preparation_boundary", "isolation", "tls_checks"):
            p.vm.save(own / "logs" / (name + ".json"), {"unit_fixture_only": True})
        p.vm.save(
            own / "logs/full_partition.json",
            {
                "all_rows_retained": True,
                "source_rows": {"train": 3, "validation": 2},
                "used_rows": {"train": 3, "validation": 2},
                "test_included": False,
            },
        )
        p.vm.save(
            own / "logs/output_boundary.json",
            {
                "old_test_read": False,
                "scores_receiver": "alice",
                "frozen_input_unchanged": True,
                "model_count": 2,
            },
        )
        p.vm.save(
            own / "results/party.executed.ipynb",
            {
                "cells": [
                    {"cell_type": "code", "execution_count": i, "outputs": []}
                    for i in [1, 2, 3]
                ]
            },
        )
        p.vm.save(own / "results/NOTEBOOK_COMPLETED.json", {"status": "passed"})
        for s in seeds:
            for cell in p.CELLS + (p.LOCAL_CELLS if party == "alice" else ()):
                fit = own / "trainings" / f"{cell}_seed{s}"
                for sub in ("code", "data", "logs", "results"):
                    (fit / sub).mkdir(parents=True)
                for name in p.original.CODE:
                    (fit / "code" / name).write_text("synthetic source only")
                (fit / "code/environment.lock").write_text("unit environment only")
                p.vm.save(
                    fit / "data/input_references.json",
                    {
                        "frozen_input": {"train.csv": "unit", "validation.csv": "unit"},
                        "derived_input": {},
                    },
                )
                p.vm.save(
                    fit / "logs/progress.json",
                    {
                        "completed_epochs": cfg["shared"]["epochs"],
                        "planned_epochs": cfg["shared"]["epochs"],
                    },
                )
                p.vm.save(fit / "results/status.json", {"training": "passed"})
    if aggregate:
        interval = {
            "estimate": 0.6,
            "ci_low": 0.5,
            "ci_high": 0.7,
            "valid_replicates": cfg["shared"]["bootstrap_repeats"],
        }
        p.vm.save(
            target / "alice/results/private_feature_evaluation.json",
            {
                "model_seeds": seeds,
                "independent_confirmation": False,
                "full_prepared_development": True,
                "models": {"A0": {"roc_auc": interval}},
                "paired_comparisons": {"B0_minus_A0": {"roc_auc": interval}},
            },
        )
        p.vm.save(
            target / "alice/logs/phase_merge.json",
            {
                "owner_arrays_exact": True,
                "no_peer_plaintext_read": True,
                "five_seeds_complete": True,
            },
        )
    p.vm.private_permissions(target)
    return target, cfg


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "status",
        "export",
        "config",
        "code",
        "preparation",
        "notebook",
        "isolation",
        "partition",
        "tls",
        "boundary",
        "count",
        "fit_code",
        "test_ref",
        "epochs",
        "prediction_status",
        "bob_evaluation",
    ],
)
def test_inventory_rejects_missing_or_changed_private_boundary(tmp_path, fault):
    target, _ = inventory(tmp_path)
    own = target / "alice"
    fit = own / "trainings/A0_seed11"
    if fault == "status":
        p.vm.save(target / "status.json", {"status": "failed", "exit_codes": [0, 0]})
    elif fault == "export":
        (own / "results/forbidden.npy").write_bytes(b"unit")
    elif fault == "config":
        (own / "code/nonlinear_full.yaml").write_text("step_id: bad")
    elif fault == "code":
        (own / "code/portfolio_training.py").write_text("changed")
    elif fault == "preparation":
        p.vm.save(own / "logs/preparation_boundary.json", {"sealed_denied": False})
    elif fault == "notebook":
        p.vm.save(
            own / "results/party.executed.ipynb",
            {"cells": [{"cell_type": "code", "execution_count": None, "outputs": []}]},
        )
    elif fault == "isolation":
        p.vm.save(own / "logs/isolation.json", {"nonroot": False})
    elif fault == "partition":
        p.vm.save(own / "logs/full_partition.json", {"all_rows_retained": False})
    elif fault == "tls":
        p.vm.save(own / "logs/tls_checks.json", {"anonymous_denied": False})
    elif fault == "boundary":
        p.vm.save(own / "logs/output_boundary.json", {"old_test_read": True})
    elif fault == "count":
        (own / "trainings/unexpected").mkdir()
    elif fault == "fit_code":
        (fit / "code/portfolio_training.py").write_text("changed")
    elif fault == "test_ref":
        p.vm.save(
            fit / "data/input_references.json",
            {"frozen_input": {"test.csv": "unit"}, "derived_input": {}},
        )
    elif fault == "epochs":
        p.vm.save(
            fit / "logs/progress.json", {"completed_epochs": 1, "planned_epochs": 12}
        )
    elif fault == "prediction_status":
        p.vm.save(fit / "results/status.json", {"training": "failed"})
    elif fault == "bob_evaluation":
        p.vm.save(target / "bob/results/private_evaluation.json", {"unit": "forbidden"})
    if fault:
        with pytest.raises((ValueError, KeyError)):
            p.verify(target)
    else:
        result = p.verify(target)
        assert (
            result["secure_fits"] == 2
            and result["alice_fits"] == 4
            and result["scope"] == "single_seed_session"
        )


@pytest.mark.parametrize("fault", [None, "lineage", "scope", "ci", "merge"])
def test_aggregate_requires_five_lineages_and_original_ci(tmp_path, monkeypatch, fault):
    import json

    target, cfg = inventory(tmp_path, True)
    phases = {str(s): Path(f"phase{s}") for s in cfg["shared"]["model_seeds"]}
    verify = p.verify
    checked = []
    monkeypatch.setattr(p, "verify", lambda t: checked.append(t))
    path = target / "alice/results/private_feature_evaluation.json"
    report = json.loads(path.read_text())
    if fault == "lineage":
        phases.pop("55")
    elif fault == "scope":
        report["independent_confirmation"] = True
    elif fault == "ci":
        report["models"]["A0"]["roc_auc"]["ci_low"] = float("nan")
    elif fault == "merge":
        p.vm.save(
            target / "alice/logs/phase_merge.json",
            {
                "owner_arrays_exact": False,
                "no_peer_plaintext_read": True,
                "five_seeds_complete": True,
            },
        )
    p.vm.save(path, report)
    if fault:
        with pytest.raises(ValueError):
            verify(target, phases)
    else:
        assert verify(target, phases)["secure_fits"] == 10 and len(checked) == 5


@pytest.mark.parametrize("recovery", [False, True])
def test_stage_and_clone_never_export_owner_arrays(tmp_path, monkeypatch, recovery):
    import json

    target = tmp_path / "target"
    target.mkdir()
    hashes = {n: p.original.sha(p.MODULE / "components" / n) for n in p.original.CODE}
    runtime = {
        "dataset": "homecredit",
        "config_sha256": p.original.sha(p.MODULE / "configs/portfolio_training.yaml"),
        "code_sha256": hashes,
    }

    def base(w, d, r, t):
        for party in p.vm.PARTIES:
            (t / party / "code").mkdir(parents=True)
            (t / party / "logs").mkdir()
            p.vm.save(t / party / "code/runtime.json", dict(runtime, party=party))

    monkeypatch.setattr(p.original, "stage", base)
    monkeypatch.setattr(
        p,
        "complete_source",
        lambda *a: (["A0_seed11", "B0_seed11"], ["L0_LR_0_seed11", "L0_tree_0_seed11"])
        if recovery
        else ([], []),
    )
    transfers = []
    commands = []
    monkeypatch.setattr(p.vm, "copy_to", lambda path, *a: transfers.append(Path(path)))

    def guest(party, command):
        commands.append((party, command))
        if "sudo cat" in command:
            return json.dumps(dict(runtime, party=party))
        if "sha256sum" in command:
            return hashes.get(command.rsplit("/", 1)[-1], "unit-input") + " source"
        return "passed"

    monkeypatch.setattr(p.vm, "guest", guest)
    monkeypatch.setattr(p, "script", lambda *a: json.dumps({}))
    p.stage(
        tmp_path,
        "homecredit",
        "new_run",
        target,
        11,
        Path("old_run") if recovery else None,
    )
    assert transfers and not any(
        x.suffix in {".csv", ".npy", ".npz"} for x in transfers
    )
    for party in p.vm.PARTIES:
        rt = p.read(target / party / "code/runtime.json")
        assert rt["selected_model_seed"] == 11 and rt["notebook"] == p.PHASE_NOTEBOOK
        for who, command in commands:
            if who == party and ("cp -a" in command or "install -o root -g" in command):
                assert "/srv/vfl/" + party + "/" in command
        receipt = p.read(target / party / "code/recovery_manifest.json")
        assert bool(receipt["reused_fits"]) == recovery


@pytest.mark.parametrize("fault", ["protocol", "source_code", "csv"])
def test_clone_rejects_changed_protocol_source_or_input(tmp_path, monkeypatch, fault):
    import json

    for party in p.vm.PARTIES:
        (tmp_path / party / "code").mkdir(parents=True)
    runtime = {
        "dataset": "homecredit",
        "party": "alice",
        "config_sha256": "cfg",
        "code_sha256": {"worker.py": "code"},
    }
    p.vm.save(tmp_path / "alice/code/runtime.json", runtime)
    calls = []

    def guest(party, command):
        calls.append(command)
        if "cat" in command:
            return json.dumps(
                dict(runtime, config_sha256="changed")
                if fault == "protocol"
                else runtime
            )
        if "worker.py" in command:
            return ("changed" if fault == "source_code" else "code") + " source"
        return ("old" if "old_run" in command else "new") + " source"

    monkeypatch.setattr(p.vm, "guest", guest)
    with pytest.raises(ValueError):
        p.clone_owner(tmp_path, "new_run", "alice", {"old_run": ["A0_seed11"]})
    assert not any("cp -a" in c for c in calls)


class NativeProcess:
    def __init__(self):
        self.returncode = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def wait(self, **kw):
        return self.returncode

    def kill(self):
        self.returncode = -9


@pytest.mark.parametrize("fault", [None, "stage", "native", "export", "verification"])
def test_launch_records_real_exit_failures_and_export_failures(
    tmp_path, monkeypatch, fault
):
    def stage(w, d, r, t, *a):
        for party in p.vm.PARTIES:
            (t / party / "logs").mkdir(parents=True)
        if fault == "stage":
            raise ValueError("stage rejected")

    monkeypatch.setattr(p, "stage", stage)

    def spawn(t, party, r):
        return NativeProcess(), (t / party / "logs/host_process.log").open("x")

    monkeypatch.setattr(p, "spawn", spawn)

    def wait(*a):
        if fault == "native":
            raise RuntimeError("native failed")

    monkeypatch.setattr(p, "wait_native", wait)
    monkeypatch.setattr(p.vm, "guest", lambda *a: "passed")
    exported = []

    def export(t, r, party, failed):
        exported.append((party, failed))
        if fault == "export":
            raise OSError("export rejected")

    monkeypatch.setattr(p.export_vm, "export", export)

    def verify(*a):
        if fault == "verification":
            raise ValueError("inventory rejected")
        return {"verification": "passed"}

    monkeypatch.setattr(p, "verify", verify)
    if fault:
        with pytest.raises((ValueError, RuntimeError)):
            p.launch(tmp_path, tmp_path / "queue", "homecredit", 11)
    else:
        assert p.launch(tmp_path, tmp_path / "queue", "homecredit", 11).exists()
    state = p.read(next((tmp_path / "queue").rglob("status.json")))
    assert state["error_type"] == (
        "ValueError"
        if fault == "stage"
        else "RuntimeError"
        if fault == "native"
        else None
    )
    assert len(exported) == (0 if fault == "stage" else 2)
    if fault == "verification":
        assert list((tmp_path / "queue").rglob("verification_failure.json"))


def test_spawn_retains_owner_identity_and_no_host_mount(tmp_path, monkeypatch):
    (tmp_path / "alice/logs").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(
        p.subprocess,
        "Popen",
        lambda command, **kw: calls.append(command) or NativeProcess(),
    )
    process, log = p.spawn(tmp_path, "alice", "new_run")
    log.close()
    command = calls[0]
    assert (
        command[command.index("-u") + 1] == "alice"
        and "-i" in command
        and "-I" in command
    )
    assert "/srv/vfl/bob/" not in " ".join(command) and process.returncode == 0


@pytest.mark.parametrize("failed", [False, True])
def test_queue_completes_all_phases_or_records_unfinished_task(
    tmp_path, monkeypatch, failed
):
    calls = []

    def launch(w, o, d, seed=None, recovery=None, phases=None):
        calls.append((d, seed, phases))
        if failed:
            raise ValueError("first phase rejected")
        return Path(o) / d / ("aggregate" if phases else str(seed))

    monkeypatch.setattr(p, "launch", launch)
    out = tmp_path / "queue"
    if failed:
        with pytest.raises(ValueError):
            p.queue(tmp_path, out)
    else:
        p.queue(tmp_path, out)
    state = p.read(out / "status.json")
    assert state["status"] == ("failed" if failed else "passed") and len(
        state["completed"]
    ) == (0 if failed else 14)
    if not failed:
        assert len(calls) == 84 and all(len(c[2]) == 5 for c in calls if c[2])
