"""Synthetic fixtures check source protection, leakage prevention and reproducibility."""
import copy
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
import yaml

from modules.m1_data_selection.components import data_preparation as dp
from modules.m1_data_selection.components import execute_preparation as runner

MODULE = Path(__file__).resolve().parents[1]
ROWS = 180


def setup_case(tmp_path, name="uci_bank"):
    config = MODULE / "configs" / f"{name}_preparation.yaml"
    cfg = yaml.safe_load(config.read_text())
    roots = tuple(tmp_path / d for d in ("raw", "derived", "results"))
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    frame = pd.DataFrame({c: np.arange(ROWS) % 7 for c in cfg["source"]["expected_columns"]})
    for party in cfg["parties"].values():
        for col in party["categorical"]:
            frame[col] = np.where(np.arange(ROWS) % 2 == 0, "one", "two")
    if name == "hillstrom":
        frame["mens"] = np.arange(ROWS) % 2
        frame["womens"] = np.arange(ROWS) % 2
    frame[cfg["label"]["column"]] = np.resize(list(cfg["label"]["mapping"]), ROWS)
    if "treatment" in cfg:
        frame[cfg["treatment"]["column"]] = np.resize(list(cfg["treatment"]["mapping"]), ROWS)
    source = roots[0] / cfg["source"]["relative_path"]
    source.parent.mkdir(parents=True)
    frame.to_csv(source, sep=cfg["source"]["separator"], index=False)
    return config, cfg, roots, repo, source


@pytest.mark.parametrize("name", ["uci_bank", "hillstrom"])
def test_full_preparation_is_deterministic_and_read_only(tmp_path, name):
    config, cfg, roots, repo, source = setup_case(tmp_path, name)
    before = dp.sha256(source)
    a = dp.prepare_context(config, repo, roots, "first")
    result = dp.run_pipeline(a)
    b = dp.prepare_context(config, repo, roots, "second")
    dp.run_pipeline(b)
    assert dp.sha256(source) == before
    manifest_a = json.loads((a["result_dir"] / "data_manifest.json").read_text())
    manifest_b = json.loads((b["result_dir"] / "data_manifest.json").read_text())
    assert manifest_a == manifest_b
    assert result["checks"]["raw_unchanged"]
    assert not result["checks"]["cryptographic_psi_executed"]
    ids_by_split = []
    for split in dp.SPLITS:
        alice = pd.read_csv(a["data_dir"] / "alice" / f"{split}.csv")
        bob = pd.read_csv(a["data_dir"] / "bob" / f"{split}.csv")
        assert dp.LABEL in alice and dp.LABEL not in bob and "treatment" not in bob
        assert alice[dp.ID].equals(bob[dp.ID])
        assert np.isfinite(bob.drop(columns=[dp.ID]).to_numpy()).all()
        ids_by_split.append(set(alice[dp.ID]))
    assert not ids_by_split[0] & ids_by_split[1]
    assert not ids_by_split[0] & ids_by_split[2]
    assert not ids_by_split[1] & ids_by_split[2]
    assert len(set.union(*ids_by_split)) == ROWS
    with pytest.raises(FileExistsError):
        dp.prepare_context(config, repo, roots, "first")
    with pytest.raises(FileExistsError):
        dp.write_json(a["result_dir"] / "checks.json", {})
    with pytest.raises(FileExistsError):
        dp.write_csv(a["data_dir"] / "clean_features.csv", pd.DataFrame())


def test_train_only_fit_excludes_holdout_categories_and_extremes():
    frame = pd.DataFrame({dp.ID: ["a", "b", "c"], "amount": [0.0, 2.0, 10000.0],
                          "category": ["train", "train", "holdout"]}).set_index(dp.ID, drop=False)
    encoded, state = dp.encode_party(frame, ["a", "b"],
                                     {"features": ["amount", "category"], "categorical": ["category"]})
    assert state["mean"] == [1.0] and state["median"] == [1.0]
    assert state["categories"] == [["train"]]
    assert encoded.loc["c", "categorical__category_train"] == 0


def test_layout_rejects_raw_repo_overlap_and_symlink(tmp_path):
    raw, derived, results, repo = [tmp_path / name for name in ("raw", "derived", "results", "repo")]
    raw.mkdir()
    link = tmp_path / "link"
    link.symlink_to(raw, target_is_directory=True)
    for destination in (raw, raw / "child", link):
        with pytest.raises(ValueError):
            dp.validate_layout(raw, destination, results, repo)
    with pytest.raises(ValueError):
        dp.validate_layout(raw, derived, repo / "child", repo)


def test_bad_source_label_leakage_and_split_fail_closed(tmp_path):
    config, cfg, roots, repo, source = setup_case(tmp_path)
    ctx = dp.prepare_context(config, repo, roots)
    ctx["config"]["parties"]["alice"]["features"].append("duration")
    with pytest.raises(ValueError, match="特征含"):
        dp.run_pipeline(ctx)
    assert (ctx["result_dir"] / "FAILED.json").exists()
    assert not (ctx["result_dir"] / "PREPARATION_READY.json").exists()
    ctx2 = dp.prepare_context(config, repo, roots)
    frame, y = dp.load_source(ctx2)
    bad = copy.deepcopy(cfg)
    bad["split"]["fractions"] = [0.5, 0.5, 0.5]
    with pytest.raises(ValueError):
        dp.split_records(frame, y, bad)
    bad["split"]["fractions"] = cfg["split"]["fractions"]
    bad["split"]["method"] = "unsupported"
    with pytest.raises(ValueError):
        dp.split_records(frame, y, bad)
    source.write_text("bad_header\n1\n")
    with pytest.raises(ValueError, match="表头"):
        dp.load_source(ctx2)
    with pytest.raises(ValueError):
        dp.prepare_context(config, repo, roots, "../escape")


def test_source_change_detected_and_original_not_restored(tmp_path):
    config, cfg, roots, repo, source = setup_case(tmp_path)
    ctx = dp.prepare_context(config, repo, roots)
    with source.open("a") as stream:
        stream.write("\n")
    after_external_edit = dp.sha256(source)
    with pytest.raises(RuntimeError, match="哈希"):
        dp.run_pipeline(ctx)
    assert dp.sha256(source) == after_external_edit
    assert (ctx["result_dir"] / "FAILED.json").exists()


def test_dataset_subdirectory_symlink_cannot_escape(tmp_path):
    config, cfg, roots, repo, _ = setup_case(tmp_path)
    roots[1].mkdir()
    (roots[1] / cfg["dataset"]).symlink_to(roots[0], target_is_directory=True)
    with pytest.raises(FileExistsError):
        dp.prepare_context(config, repo, roots, "new")


@pytest.mark.parametrize("fail", [False, True])
def test_execute_preparation_keeps_source_clean_and_saves_failure(tmp_path, monkeypatch, fail):
    source = tmp_path / "S1.P1_test.ipynb"
    n = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('test')")])
    nbformat.write(n, source)
    before = dp.sha256(source)
    result_dir = tmp_path / "result"
    result_dir.mkdir()

    class FakeClient:
        def __init__(self, nb, **kwargs):
            self.nb = nb

        def execute(self):
            self.nb.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout",
                text="VFL_RUN_LOCATOR=" + json.dumps({"result_dir": str(result_dir)}) + "\n")]
            if fail:
                raise RuntimeError("constructed notebook failure")

    monkeypatch.setattr(runner, "NotebookClient", FakeClient)
    if fail:
        with pytest.raises(RuntimeError):
            runner.execute(source, tmp_path)
    else:
        assert runner.execute(source, tmp_path) == result_dir
    assert dp.sha256(source) == before
    assert (result_dir / "S1.P1_test.executed.ipynb").exists()
    assert (result_dir / "NOTEBOOK_COMPLETED.json").exists() != fail
