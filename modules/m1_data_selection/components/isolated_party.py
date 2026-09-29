"""Single-party preparation. Isolation checks run before any dataset read."""
from __future__ import annotations

import errno
import json
from pathlib import Path
import socket
import sys

import nbformat
import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

from data_preparation import encode_party, sha256, write_csv, write_json  # type: ignore[import-not-found]

DENIED_ERRNOS = {errno.EACCES, errno.EPERM}
SPLITS = ("train", "validation", "test")
NOTEBOOK_VERSION = 4
NETWORK_TIMEOUT = 1


def denied_open(path, mode):
    try:
        with Path(path).open(mode):
            pass
    except OSError as exc:
        return exc.errno in DENIED_ERRNOS
    return False


def require_isolation(config, record=True):
    """ENOENT and connection-refused are not evidence of access control."""
    probes = config["probes"]
    checks = {
        "peer_input_read_denied": denied_open(probes["peer_input"], "rb"),
        "peer_output_read_denied": denied_open(probes["peer_output"], "rb"),
        "peer_output_write_denied": denied_open(probes["peer_output"], "r+b"),
        "full_source_read_denied": denied_open(probes["full_source"], "rb"),
        "own_input_write_denied": denied_open(config["input_path"], "r+b"),
        "symlink_peer_read_denied": denied_open(probes["symlink_peer"], "rb"),
    }
    try:
        with socket.create_connection(("127.0.0.1", probes["port"]), timeout=NETWORK_TIMEOUT):
            pass
    except OSError as exc:
        checks["network_denied"] = exc.errno in DENIED_ERRNOS
    else:
        checks["network_denied"] = False
    if not all(checks.values()):
        raise RuntimeError("隔离检查失败，禁止读取数据：" + json.dumps(checks))
    if record:
        write_json(Path(config["result_dir"]) / "isolation_checks.json", checks)
    return checks


def process_own_data(config):
    require_isolation(config, record=False)
    source, target, result = (Path(config[k]) for k in ("input_path", "data_dir", "result_dir"))
    before = sha256(source)
    df = pd.read_csv(source).set_index("record_id", drop=False)
    spec = config["feature_spec"]
    required = ["record_id", "split"] + spec["features"] + config["outcome_columns"]
    if list(df.columns) != required or not df.index.is_unique:
        raise ValueError("本方输入表头或记录键不符合契约")
    if set(df["split"]) != set(SPLITS):
        raise ValueError("冻结划分必须含 train/validation/test")
    if config["party"] == "bob" and config["outcome_columns"]:
        raise ValueError("Bob 不得包含结果字段")
    clean = df[["record_id"] + spec["features"]].copy()
    for name, sentinel in config["numeric_sentinels"].items():
        clean[name] = clean[name].replace(sentinel, np.nan)
    for name in spec["categorical"]:
        clean[name] = clean[name].replace(config["missing_categories"], np.nan)
    encoded, state = encode_party(clean, df.index[df["split"] == "train"], spec)
    outcomes = pd.DataFrame(index=df.index)
    if config["party"] == "alice":
        for output, description in config["outcome_mapping"].items():
            # JSON mapping keys are strings; normalize only scalar value representation.
            values = df[description["column"]].astype(str).map(description["mapping"])
            if values.isna().any():
                raise ValueError("结果字段存在未定义值")
            outcomes[output] = values.astype(int)
    write_csv(target / "clean_features.csv", clean)
    write_csv(target / "split_assignments.csv", df[["record_id", "split"]])
    summary = []
    for split in SPLITS:
        part = encoded.loc[df["split"] == split].join(outcomes)
        write_csv(target / f"{split}.csv", part)
        summary.append({"split": split, "rows": len(part), "features": len(state["output_features"])})
    write_json(result / "preprocessing_state.json", state)
    write_json(result / "private_summary.json", summary)
    manifest = {p.name: sha256(p) for p in sorted(target.glob("*.csv"))}
    write_json(result / "data_manifest.json", manifest)
    if sha256(source) != before:
        raise RuntimeError("本方原始投递发生变化，拒绝放行")
    receipt = {"party": config["party"], "status": "prepared", "input_unchanged": True,
               "isolation_checks_passed": True, "artifacts": manifest,
               "training_status": "blocked", "isolation_mode": "os_sandbox"}
    write_json(result / "receipt.json", receipt)
    return summary


def execute_private_notebook(config_path, notebook_path):
    """In-process IPython executes real notebook cells with no network/Jupyter socket."""
    config = json.loads(Path(config_path).read_text())
    result = Path(config["result_dir"])
    nb = nbformat.read(notebook_path, as_version=NOTEBOOK_VERSION)
    shell = InteractiveShell.instance()
    shell.user_ns["CONFIG_PATH"] = str(config_path)
    count = 0
    failure = None
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        with capture_output() as captured:
            executed = shell.run_cell(cell.source, store_history=False)
        cell.execution_count = count
        cell.outputs = []
        for channel in ("stdout", "stderr"):
            value = getattr(captured, channel)
            if value:
                cell.outputs.append(nbformat.v4.new_output("stream", name=channel, text=value))
        for rich in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output("display_data", data=rich.data, metadata=rich.metadata))
        error = executed.error_before_exec or executed.error_in_exec
        if error:
            cell.outputs.append(nbformat.v4.new_output("error", ename=type(error).__name__, evalue=str(error), traceback=[]))
            failure = error
            break
    with (result / "party.executed.ipynb").open("x", encoding="utf-8") as stream:
        nbformat.write(nb, stream)
    if failure:
        write_json(result / "FAILED.json", {"reason": type(failure).__name__, "message": str(failure)})
        raise RuntimeError("本方 Notebook 失败，详见私有日志") from failure
    write_json(result / "NOTEBOOK_COMPLETED.json", {"cells": count, "status": "passed"})


if __name__ == "__main__":
    # -I omits the script directory; launcher adds only this trusted code snapshot.
    execute_private_notebook(sys.argv[1], sys.argv[2])
