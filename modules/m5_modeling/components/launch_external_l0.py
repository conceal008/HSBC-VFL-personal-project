"""Launch an Alice-only notebook inside a default-deny macOS sandbox.

The trusted local controller handles paths and hashes only. It never opens rows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO_PARENT_DEPTH = 3
NETWORK_DENIAL_PROBE_PORT = 7
PROCESS_TIMEOUT_SECONDS = 1800


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sb_quote(path: Path | str) -> str:
    return json.dumps(str(path), ensure_ascii=False)


def sandbox_profile(project: Path, own_data: Path, result: Path,
                    code: Path, runtime: Path) -> str:
    allowed = [Path(p) for p in ("/System", "/usr", "/Library", "/opt/homebrew", "/dev")]
    allowed += [own_data, result, code, runtime]
    reads = " ".join(f"(subpath {sb_quote(path)})" for path in allowed)
    exceptions = " ".join(f"(require-not (subpath {sb_quote(path)}))"
                          for path in (own_data, result, code, runtime))
    ancestors = {parent for path in (own_data, result, code, runtime)
                 for parent in path.parents if parent == project or project in parent.parents}
    exceptions += " " + " ".join(f"(require-not (literal {sb_quote(path)}))"
                                  for path in sorted(ancestors | {own_data, result, code, runtime}))
    return ("(version 1)\n(deny default)\n(allow process-exec)\n(allow sysctl-read)\n"
            "(allow file-read-metadata)\n"
            f"(allow file-read-data (literal \"/\") {reads})\n"
            f"(deny file-read* (require-all (subpath {sb_quote(project)}) {exceptions}))\n"
            f"(allow file-write* (subpath {sb_quote(result)}) (literal \"/dev/null\"))\n")


def launch(workspace: Path, dataset: str, prepared_id: str) -> Path:
    workspace = workspace.resolve()
    project = workspace.parent
    repo = Path(__file__).resolve().parents[REPO_PARENT_DEPTH]
    runtime_python = workspace / "数据处理环境/v1/bin/python"
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise RuntimeError("This step requires the tested macOS OS sandbox; no unsafe fallback")
    if not runtime_python.is_file():
        raise FileNotFoundError(runtime_python)
    if not prepared_id.startswith(dataset + "_"):
        raise ValueError("Prepared run and dataset mismatch")
    own_data = workspace / "联邦隔离产物" / prepared_id / "alice"
    peer_data = workspace / "联邦隔离产物" / prepared_id / "bob"
    own_input = workspace / "联邦隔离输入" / prepared_id / "alice" / "input.csv"
    peer_input = workspace / "联邦隔离输入" / prepared_id / "bob" / "input.csv"
    source_declaration = workspace / "实验日志" / prepared_id / "declaration.json"
    source_result = workspace / "联邦隔离结果" / prepared_id / "alice"
    if not all(path.is_file() for path in (own_input, peer_input, source_declaration,
                                          source_result / "data_manifest.json",
                                          own_data / "train.csv", peer_data / "train.csv")):
        raise FileNotFoundError("Missing accepted S1.P2 private input")
    declaration = json.loads(source_declaration.read_text())
    if declaration["experiment_id"] != prepared_id:
        raise ValueError("Accepted preparation declaration mismatch")
    source = workspace / "数据集" / declaration["data_source"]["relative_path"]
    if not source.is_file():
        raise FileNotFoundError("Full source probe must be an existing file")
    manifest = json.loads((source_result / "data_manifest.json").read_text())
    config = repo / "modules/m5_modeling/configs/external_l0.yaml"
    code_files = [repo / "modules/m5_modeling/components/external_l0.py",
                  repo / "modules/m1_data_selection/components/isolated_party.py",
                  repo / "modules/m1_data_selection/components/data_preparation.py"]
    notebook = repo / "modules/m5_modeling/notebooks/S5.P1_external_l0.ipynb"
    run_id = f"{dataset}_l0_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    result = workspace / "联邦训练结果" / run_id / "alice"
    code = result / "code_snapshot"
    log_dir = workspace / "实验日志" / run_id
    code.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)
    for path in code_files + [notebook, config]:
        shutil.copyfile(path, code / path.name)
    code_hash = hashlib.sha256((code / "external_l0.py").read_bytes()).hexdigest()
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                             capture_output=True, text=True).stdout.strip()
    runtime = {
        "dataset": dataset, "prepared_experiment": prepared_id,
        "data_dir": str(own_data), "result_dir": str(result),
        "public_config_snapshot": str(code / config.name),
        "accepted_manifest": manifest, "code_sha256": code_hash, "git_sha": git_sha,
        "probes": {"peer_input": str(peer_input), "peer_output": str(peer_data / "train.csv"),
                   "full_source": str(source), "port": NETWORK_DENIAL_PROBE_PORT},
    }
    save_json(result / "runtime.json", runtime)
    (result / "sandbox.sb").write_text(sandbox_profile(project, own_data, result, code,
                                                        runtime_python.parents[1]))
    declaration_out = {
        "experiment_id": run_id, "purpose": "Alice-only frozen-split L0 baseline",
        "prepared_experiment": prepared_id, "dataset": dataset,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "code_sha256": code_hash, "git_sha": git_sha,
        "allowed_disclosure": ["status", "private artifact path"],
        "isolation": "single-host OS sandbox; trusted administrator; no physical separation",
        "psi": "blocked", "secure_joint_training": "blocked", "model_leakage_evaluation": "blocked",
    }
    save_json(log_dir / "declaration.json", declaration_out)
    env = {key: os.environ[key] for key in ("PATH", "LANG") if key in os.environ}
    env.update(HOME=str(result), TMPDIR=str(result), IPYTHONDIR=str(result / "ipython"),
               PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    command = ["/usr/bin/sandbox-exec", "-f", str(result / "sandbox.sb"),
               str(runtime_python), "-I", "-B", "-c",
               "import sys; sys.path.insert(0,sys.argv[1]); from isolated_party import execute_private_notebook; execute_private_notebook(sys.argv[2],sys.argv[3])",
               str(code), str(result / "runtime.json"), str(code / notebook.name)]
    with (result / "process.log").open("w") as log:
        process = subprocess.run(command, cwd=result, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=PROCESS_TIMEOUT_SECONDS, check=False)
    passed = (process.returncode == 0 and (result / "NOTEBOOK_COMPLETED.json").is_file()
              and (result / "l0_private_metrics.json").is_file())
    status = {"experiment_id": run_id, "l0_status": "passed" if passed else "failed",
              "exit_code": process.returncode, "private_result_dir": str(result),
              "psi": "blocked", "secure_joint_training": "blocked",
              "model_leakage_evaluation": "blocked", "separate_host": "not_verified"}
    save_json(log_dir / "status.json", status)
    (log_dir / "实验日志.md").write_text(
        f"# 实验日志：{run_id}\n\n目的：在 S1.P2 冻结输入上运行 Alice-only L0。\n\n"
        f"内容：隔离前置检查 → 训练切片拟合 → 验证集选参 → 冻结测试集评估。"
        f"配置、代码、来源指纹见 declaration.json；私有指标和 Notebook 在 {result}。\n\n"
        f"结果：L0={status['l0_status']}；PSI、安全联合训练、模型泄漏评估均 blocked；"
        f"双物理主机未验证。错误详见 Alice 私有 process.log。\n")
    with (workspace / "实验日志/实验索引.md").open("a") as stream:
        stream.write(f"\n- `{run_id}`：L0={status['l0_status']}；PSI/L3/泄漏评估 blocked；"
                     f"[实验日志]({run_id}/实验日志.md)。\n")
    if not passed:
        raise RuntimeError(f"L0 failed; inspect private process.log: {result}")
    return log_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--dataset", required=True,
                        choices=["uci_bank_marketing", "hillstrom_email_marketing"])
    parser.add_argument("--prepared-id", required=True)
    args = parser.parse_args()
    print(launch(args.workspace, args.dataset, args.prepared_id))
