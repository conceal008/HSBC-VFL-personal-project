"""Positive and negative controls for offline provisioning and party boundaries."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from modules.m1_data_selection.components import isolated_simulation as sim

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE / "components"))
spec = importlib.util.spec_from_file_location("isolated_party_for_tests", MODULE / "components/isolated_party.py")
assert spec and spec.loader
party = importlib.util.module_from_spec(spec)
spec.loader.exec_module(party)
ROWS = 120


def setup_inputs(tmp_path):
    repo = tmp_path / "repo"
    (repo / "modules/m1_data_selection/notebooks").mkdir(parents=True)
    shutil.copyfile(MODULE / "notebooks/S1.P2_party_preparation.ipynb",
                    repo / "modules/m1_data_selection/notebooks/S1.P2_party_preparation.ipynb")
    cfg = yaml.safe_load((MODULE / "configs/hillstrom_preparation.yaml").read_text())
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    source = repo.parent / "8-9月/数据集" / cfg["source"]["relative_path"]
    source.parent.mkdir(parents=True)
    frame = pd.DataFrame({col: [i % 2 for i in range(ROWS)] for col in cfg["source"]["expected_columns"]})
    frame["zip_code"] = "one"
    frame["channel"] = "two"
    frame["segment"] = [list(cfg["treatment"]["mapping"])[i % 3] for i in range(ROWS)]
    frame.to_csv(source,index=False)
    return repo, cfg, config, source


def test_distributor_uses_only_party_fields_and_label_independent_split(tmp_path):
    repo, cfg, config, source = setup_inputs(tmp_path)
    input_dirs = {p: tmp_path / p for p in sim.PARTIES}
    for p in input_dirs.values():
        p.mkdir()
    sim.distribute(cfg, source, input_dirs)
    a = pd.read_csv(input_dirs["alice"] / "input.csv")
    b = pd.read_csv(input_dirs["bob"] / "input.csv")
    assert a.record_id.equals(b.record_id) and a.split.equals(b.split)
    assert "conversion" in a and "segment" in a
    assert "conversion" not in b and "segment" not in b
    assert set(a.columns) & set(b.columns) == {"record_id", "split"}
    assert set(a.split) == {"train", "validation", "test"}


def test_missing_file_is_not_denied_access(tmp_path):
    assert not party.denied_open(tmp_path / "missing", "rb")
    path=tmp_path / "allowed"
    path.write_text("fixture")
    assert not party.denied_open(path, "rb")


def test_receipt_rejects_private_fields_and_fake_training():
    ok = {"party":"alice","status":"prepared","input_unchanged":True,"isolation_checks_passed":True,
          "artifacts":{f:"a"*64 for f in sim.SAFE_FILES},"training_status":"blocked","isolation_mode":"os_sandbox"}
    assert sim.safe_receipt(ok)==ok
    for key,value in [("rows",10),("training_status","complete"),("isolation_mode","separate_host"),
                      ("input_unchanged",False),("artifacts",{"arbitrary":"row values"})]:
        bad=copy.deepcopy(ok)
        bad[key]=value
        with pytest.raises(ValueError):
            sim.safe_receipt(bad)


@pytest.mark.skipif(sys.platform != "darwin", reason="native macOS enforcement tested only on macOS")
def test_real_os_isolation_and_raw_preservation(tmp_path):
    repo,cfg,config,source=setup_inputs(tmp_path)
    before=sim.sha256(source)
    journal=sim.run_isolated(config,repo)
    summary=json.loads((journal/'summary.json').read_text())
    assert summary['status']=='data_preparation_only'
    assert summary['physical_isolation_verified'] is False
    assert sim.sha256(source)==before
    for event in summary['events']:
        result=Path(event['private_result_uri'])
        checks=json.loads((result/'isolation_checks.json').read_text())
        assert all(checks.values())
        assert (result/'party.executed.ipynb').exists()
        assert (result/'NOTEBOOK_COMPLETED.json').exists()
        receipt=json.loads((journal/f"{event['party']}_receipt.json").read_text())
        assert 'rows' not in receipt and 'preprocessing_state' not in receipt
        # Removing sandbox must fail before data processing, rather than silently run.
        probe=subprocess.run([sys.executable,'-I','-c',
            'import sys,json;sys.path.insert(0,sys.argv[1]);from isolated_party import require_isolation;require_isolation(json.load(open(sys.argv[2])))',
            str(result/'code_snapshot'),str(result/'party_config.json')],capture_output=True,text=True)
        assert probe.returncode!=0 and '隔离检查失败' in probe.stderr


def test_outcome_changes_do_not_change_partition(tmp_path):
    repo, cfg, config, source = setup_inputs(tmp_path)
    before = None
    for iteration in range(2):
        paths = {p: tmp_path / str(iteration) / p for p in sim.PARTIES}
        for path in paths.values():
            path.mkdir(parents=True)
        sim.distribute(cfg, source, paths)
        current = pd.read_csv(paths['bob'] / 'input.csv')[['record_id', 'split']]
        if before is not None:
            pd.testing.assert_frame_equal(before, current)
        before = current
        frame = pd.read_csv(source)
        frame['conversion'] = 1 - frame['conversion']
        frame.to_csv(source, index=False)


def test_orchestration_and_private_notebooks_with_simulated_executor(tmp_path, monkeypatch):
    """Portable functional test; mocks are NOT evidence of kernel enforcement."""
    from types import SimpleNamespace
    import errno
    import nbformat

    repo, cfg, config, source = setup_inputs(tmp_path)
    monkeypatch.setattr(sim, 'sys', SimpleNamespace(platform='darwin', prefix=sys.prefix,
                                                    executable=sys.executable, version=sys.version))
    original_is_file = Path.is_file
    monkeypatch.setattr(Path, 'is_file', lambda p: True if str(p) == '/usr/bin/sandbox-exec' else original_is_file(p))
    monkeypatch.setitem(sys.modules, 'isolated_party', party)
    original_run = subprocess.run

    def fake_executor(command, **kwargs):
        if command[0] != '/usr/bin/sandbox-exec':
            return original_run(command, **kwargs)
        # Only inside the fake worker: simulate OS denial for portable business-logic coverage.
        with monkeypatch.context() as local:
            local.setattr(party, 'denied_open', lambda *args: True)
            def denied_connection(*args, **kwargs):
                raise PermissionError(errno.EPERM, 'synthetic test denial')
            local.setattr(party.socket, 'create_connection', denied_connection)
            party.execute_private_notebook(command[-2], command[-1])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sim.subprocess, 'run', fake_executor)
    journal = sim.run_isolated(config, repo)
    summary = json.loads((journal / 'summary.json').read_text())
    assert summary['training_status'] == 'blocked'
    for event in summary['events']:
        result = Path(event['private_result_uri'])
        own = json.loads((result / 'party_config.json').read_text())
        frame = pd.read_csv(Path(own['data_dir']) / 'train.csv')
        assert ('label' in frame) == (event['party'] == 'alice')
        nb = nbformat.read(result / 'party.executed.ipynb', as_version=4)
        assert all(cell.execution_count for cell in nb.cells if cell.cell_type == 'code')
        assert not any(out.output_type == 'error' for cell in nb.cells if cell.cell_type == 'code' for out in cell.outputs)
    assert (journal / '实验日志.md').exists()

    # A failed worker must create an auditable failed_closed experiment, with no fallback.
    def failed_executor(command, **kwargs):
        if command[0] != '/usr/bin/sandbox-exec':
            return original_run(command, **kwargs)
        return subprocess.CompletedProcess(command, 1)
    monkeypatch.setattr(sim.subprocess, 'run', failed_executor)
    with pytest.raises(RuntimeError, match='隔离运行失败'):
        sim.run_isolated(config, repo)
    failures = list((repo.parent / '8-9月/实验日志').glob('*/FAILED.json'))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text())['status'] == 'failed_closed'

    # Notebook cell failures preserve the failed notebook instead of claiming completion.
    failed_result = tmp_path / 'failed_notebook'
    failed_result.mkdir()
    failed_config = tmp_path / 'failed_config.json'
    failed_config.write_text(json.dumps({'result_dir': str(failed_result)}))
    failed_nb = tmp_path / 'failed.ipynb'
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("raise ValueError('fixture failure')")]), failed_nb)
    with pytest.raises(RuntimeError, match='Notebook 失败'):
        party.execute_private_notebook(failed_config, failed_nb)
    assert (failed_result / 'FAILED.json').exists()
    assert not (failed_result / 'NOTEBOOK_COMPLETED.json').exists()
