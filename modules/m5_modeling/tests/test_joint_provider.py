"""Frozen-input routing and durable lifecycle tests on synthetic fixtures."""

from pathlib import Path
import json
import sys
import pytest
import yaml

COMPONENTS = Path(__file__).resolve().parents[1] / "components"
sys.path.insert(0, str(COMPONENTS))
import joint_budget_vm as jv  # noqa: E402


def test_stage_never_reads_test_and_rejects_changed_source(tmp_path, monkeypatch):
    vm = jv.vm
    copied, commands = [], []
    monkeypatch.setattr(
        vm, "setup_certificates", lambda _: {"alice": "unit-a", "bob": "unit-b"}
    )
    monkeypatch.setattr(vm, "restrict_network", lambda *a: None)
    monkeypatch.setattr(vm, "copy_to", lambda *a: copied.append(a))
    monkeypatch.setattr(vm, "guest", lambda *a: commands.append(a) or "")
    dataset = "uci_bank_marketing"
    prepared = vm.PREPARED[dataset]
    for party in vm.PARTIES:
        source = tmp_path / "联邦隔离产物" / prepared / party
        source.mkdir(parents=True)
        manifest = {}
        for split in ["train", "validation"]:
            path = source / f"{split}.csv"
            path.write_text("unit_fixture_only\n0\n")
            manifest[path.name] = jv.sha(path)
        result = tmp_path / "联邦隔离结果" / prepared / party
        result.mkdir(parents=True)
        (result / "data_manifest.json").write_text(json.dumps(manifest))
    queue = tmp_path / "source_queue.json"
    queue.write_text(
        json.dumps(
            {
                "status": "passed",
                "completed": [str(tmp_path / dataset / "source_round")],
            }
        )
    )
    monkeypatch.setenv("HSBC_E5_SOURCE_QUEUE", str(queue))
    source_cfg = yaml.safe_load(
        (COMPONENTS.parent / "configs/nonlinear_full.yaml").read_text()
    )

    def guest(owner, script):
        commands.append((owner, script))
        if "sudo cat" in script:
            return yaml.safe_dump(source_cfg)
        if "sha256sum" in script:
            for n in (
                "functional_training.py",
                "functional_metrics.py",
                "functional_tls.py",
                "active_diagnostic.py",
                "nonlinear_full.py",
            ):
                if script.endswith("/" + n):
                    return jv.sha(COMPONENTS / n) + " path"
            return "0" * 64 + " path"
        return ""

    monkeypatch.setattr(vm, "guest", guest)
    root = tmp_path / "stage"
    jv.stage(tmp_path, dataset, "unit", root)
    assert all(Path(args[0]).name != "test.csv" for args in copied)
    assert any("chown root:root" in command for _, command in commands)
    for party in vm.PARTIES:
        runtime = json.loads((root / party / "code/runtime.json").read_text())
        assert runtime["party"] == party and runtime["is_smoke"] is False
        manifest = json.loads((root / party / "data/input_manifest.json").read_text())
        assert {n for n in manifest if not n.startswith("references/")} == {
            "train.csv",
            "validation.csv",
        }
        assert "references/train_local.npz" in manifest and not any(
            "test" in n for n in manifest
        )
        assert (
            not any(".npy" in n for n in manifest)
            if party == "bob"
            else any(".npy" in n for n in manifest)
        )
    (tmp_path / "联邦隔离产物" / prepared / "alice/train.csv").write_text("changed")
    with pytest.raises(ValueError, match="Frozen input changed"):
        jv.stage(tmp_path, dataset, "bad", tmp_path / "bad")
    with pytest.raises(ValueError, match="Undeclared"):
        jv.stage(tmp_path, "unknown", "bad", tmp_path / "unknown")


def test_export_excludes_predictions_and_lifecycle(tmp_path, monkeypatch):
    import tarfile
    import io

    vm = jv.vm
    commands = []
    monkeypatch.setattr(vm, "guest", lambda *a: commands.append(a) or "")

    def lima(*args):
        assert args[0] == "copy"
        with tarfile.open(args[-1], "w") as archive:
            info = tarfile.TarInfo("results/authorized.json")
            content = b'{"unit":true}'
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))

    monkeypatch.setattr(vm, "lima", lima)
    (tmp_path / "alice").mkdir()
    jv.export(tmp_path, "unit", "alice")
    assert (tmp_path / "alice/results/authorized.json").is_file()
    command = commands[0][1]
    assert all(
        f"--exclude='{pattern}'" in command
        for pattern in ["model.share", "*.npy", "*.npz", "*.trace.log"]
    )

    def stage(workspace, dataset, run, local):
        for owner in vm.PARTIES:
            (local / owner / "logs").mkdir(parents=True)

    monkeypatch.setattr(jv, "stage", stage)
    exports = []
    monkeypatch.setattr(
        jv,
        "export",
        lambda local, run, party, failed=False: exports.append((party, failed)),
    )
    monkeypatch.setattr(jv, "verify", lambda _: {"verification": "unit lifecycle only"})
    (tmp_path / "实验日志").mkdir()
    (tmp_path / "实验日志/实验索引.md").write_text("")
    for success in [True, False]:

        class Process:
            def __init__(self, *a, **kw):
                self.returncode = 0 if success else 1

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

        monkeypatch.setattr(jv.subprocess, "Popen", Process)
        if success:
            local = jv.launch(tmp_path, "uci_bank_marketing")
            assert json.loads((local / "status.json").read_text())["exit_codes"] == [
                0,
                0,
            ]
        else:
            with pytest.raises(RuntimeError, match="exit failure"):
                jv.launch(tmp_path, "uci_bank_marketing")
    assert len(exports) == 4 and exports[-1][1]


def test_durable_queue_tracks_success_and_failure(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        jv,
        "launch",
        lambda workspace, dataset: calls.append(dataset) or workspace / dataset,
    )
    queue = jv.run_queue(tmp_path)
    receipt = json.loads((queue / "status.json").read_text())
    assert receipt["status"] == "passed" and len(receipt["completed"]) == 2
    assert calls == list(jv.vm.PREPARED)

    def failed(workspace, dataset):
        raise RuntimeError("unit lifecycle failure")

    monkeypatch.setattr(jv, "launch", failed)
    with pytest.raises(RuntimeError, match="unit lifecycle"):
        jv.run_queue(tmp_path)
    receipts = [
        json.loads((p / "status.json").read_text())
        for p in (tmp_path / "联合优化诊断").iterdir()
    ]
    assert any(
        d["status"] == "failed" and d["error_type"] == "RuntimeError" for d in receipts
    )


def test_transitive_recovery_uses_each_frozen_fit_manifest(tmp_path, monkeypatch):
    cfg = yaml.safe_load((COMPONENTS.parent / "configs/joint_budget.yaml").read_text())
    hashes = {n: jv.sha(COMPONENTS / n) for n in jv.CODE}
    inherited_hashes = dict(hashes, **{"joint_budget.py": "a" * 64})
    names = ["D_A0_seed11", "D_B0_seed22"]
    descriptor = tmp_path / "recovery.json"
    descriptor.write_text(
        json.dumps(
            {
                "dataset": "uci_bank_marketing",
                "source_run": "unit_recovery",
                "completed_fits": names,
                "original_exit_unknown": True,
            }
        )
    )
    monkeypatch.setenv("HSBC_E5_RECOVERY_PLAN", str(descriptor))
    runtime = {
        "dataset": "uci_bank_marketing",
        "code_sha256": hashes,
        "config_sha256": "unit",
        "reused_completed_fits": [names[0]],
    }
    lineage = {"source_code_sha256": inherited_hashes}
    completed = [48]
    changed = [False]

    def guest(owner, script):
        if script.endswith("/code/runtime.json"):
            return json.dumps(runtime)
        if script.endswith("/code/joint_budget.yaml"):
            return yaml.safe_dump(cfg)
        if script.endswith("/logs/recovery_lineage.json"):
            return json.dumps(lineage)
        if script.endswith("/logs/progress.json"):
            return json.dumps({"completed_epochs": completed[0], "planned_epochs": 48})
        if script.endswith("/diagnostic.json"):
            return json.dumps(
                {
                    "epoch": int(script.split("/checkpoints/e")[1].split("/")[0]),
                    "original_reference_passed": True,
                }
            )
        if "sha256sum" in script:
            name = Path(script.split()[-1]).name
            expected = (
                inherited_hashes if "/trainings/" + names[0] + "/" in script else hashes
            )
            return ("b" * 64 if changed[0] else expected[name]) + " unit"
        assert "test -s" in script
        return ""

    monkeypatch.setattr(jv.vm, "guest", guest)
    receipt = jv.completed_recovery("uci_bank_marketing", cfg, COMPONENTS.parent)
    assert receipt["alice"]["fit_code_sha256"][names[0]] == inherited_hashes
    assert receipt["bob"]["fit_code_sha256"][names[1]] == hashes
    assert (
        jv.completed_recovery("hillstrom_email_marketing", cfg, COMPONENTS.parent)
        is None
    )
    completed[0] = 23
    with pytest.raises(ValueError, match="incomplete"):
        jv.completed_recovery("uci_bank_marketing", cfg, COMPONENTS.parent)
    completed[0] = 48
    changed[0] = True
    with pytest.raises(ValueError, match="changed"):
        jv.completed_recovery("uci_bank_marketing", cfg, COMPONENTS.parent)


def test_unreachable_vm_cannot_mask_failure_or_drop_receipt(tmp_path, monkeypatch):
    def stage(workspace, dataset, run, local):
        for owner in jv.vm.PARTIES:
            (local / owner / "logs").mkdir(parents=True)

    monkeypatch.setattr(jv, "stage", stage)

    class Process:
        def __init__(self, *args, **kwargs):
            self.returncode = 1

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

    monkeypatch.setattr(jv.subprocess, "Popen", Process)

    def unavailable(*args, **kwargs):
        raise ConnectionError("unit VM unavailable")

    monkeypatch.setattr(jv.vm, "guest", unavailable)
    monkeypatch.setattr(jv, "export", unavailable)
    with pytest.raises(RuntimeError, match="Participant exit failure"):
        jv.launch(tmp_path, "uci_bank_marketing")
    root = next((tmp_path / "联合优化诊断/uci_bank_marketing").iterdir())
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "failed" and len(status["cleanup_errors"]) == 4
    assert all(r["error_type"] == "ConnectionError" for r in status["cleanup_errors"])
