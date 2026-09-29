"""Trusted offline distributor and OS-enforced single-party launch, no training."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from uuid import uuid4

import yaml

from modules.m1_data_selection.components.data_preparation import inside, sha256, write_json

PARTIES = ("alice", "bob")
TOKEN_LENGTH = 8
HASH_DIGITS = 8
HEX_BASE = 16
HASH_SPACE = HEX_BASE ** HASH_DIGITS
NOTEBOOK_VERSION = 4
PROCESS_TIMEOUT = 300
SAFE_FILES = {"clean_features.csv", "split_assignments.csv", "train.csv", "validation.csv", "test.csv"}


def write_experiment_log(journal, status, events, error=None):
    """Human-readable experiment record without collecting private statistics."""
    lines = ["# 实验日志：" + journal.name, "", "## 目的", "",
             "从原件重新分发并验证两方独立预处理、内核访问限制和原件保护。", "",
             "## 内容与方法", "", "P0 登记 → P1 可信离线分发 → P2 权限拒绝检查 → P3 本方清洗 → P5 本方训练集拟合。",
             "配置、来源、预测时点和种子见 declaration.json。初始投递只供本方读取；双方无网络权限。", "",
             "## 中间清洗数据与结果", ""]
    for event in events:
        lines += [f"- {event['party']}：{event['status']}",
                  f"  - 私有中间数据：`{event['private_data_uri']}`",
                  f"  - 私有 Notebook/参数/校验：`{event['private_result_uri']}`",
                  f"  - 允许披露的产物哈希：`{event['party']}_receipt.json`"]
    lines += ["", "## 最终状态", "", f"`{status}`；联合训练 blocked；物理隔离未验证。",
              "运行的 OS 沙箱只保护本机参与方进程访问边界；实验管理员可信，尚未审计模型推断或恶意宿主攻击。"]
    if error:
        lines += ["", "失败原因：" + error, "未放宽边界；重试必须新建实验 ID。"]
    with (journal / "实验日志.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    with (journal.parent / "实验索引.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\n- [{journal.name}]({journal.name}/实验日志.md)：{status}；training=blocked\n")


def profile_text(project, own_input, own_data, own_result, code, runtime):
    def quote(value):
        return json.dumps(str(Path(value).resolve()), ensure_ascii=False)
    allowed = ["/System", "/usr", "/Library", "/opt/homebrew", "/dev", own_input, own_data, own_result, code, runtime]
    allowed_reads = " ".join(f"(subpath {quote(p)})" for p in allowed)
    except_project = " ".join(f"(require-not (subpath {quote(p)}))" for p in
                             (own_input, own_data, own_result, code, runtime))
    # realpath/stat must traverse parent directories; permit metadata traversal
    # of exact ancestors, while the global data-read rule still denies listing.
    ancestors = {parent for path in (own_input, own_data, own_result, code, runtime)
                 for parent in Path(path).resolve().parents if inside(parent, Path(project).resolve())}
    except_project += " " + " ".join(f"(require-not (literal {quote(p)}))" for p in sorted(ancestors))
    return f'''(version 1)
(deny default)
(allow process-exec)
(allow sysctl-read)
(allow file-read-metadata)
(allow file-read-data (literal "/") {allowed_reads})
(deny file-read* (require-all (subpath {quote(project)}) {except_project}))
(allow file-write* (subpath {quote(own_data)}) (subpath {quote(own_result)}) (literal "/dev/null"))
'''


def distribute(config, source, input_dirs):
    """Only public-source provisioning; no learned cleaning or label-driven split."""
    fingerprint = sha256(source)
    fields = {}
    outputs = {}
    streams = []
    try:
        for party in PARTIES:
            outcomes = [config["label"]["column"]] if party == "alice" else []
            if party == "alice" and "treatment" in config:
                outcomes.append(config["treatment"]["column"])
            fields[party] = ["record_id", "split"] + config["parties"][party]["features"] + outcomes
            stream = (input_dirs[party] / "input.csv").open("x", encoding="utf-8", newline="")
            streams.append(stream)
            outputs[party] = csv.DictWriter(stream, fieldnames=fields[party])
            outputs[party].writeheader()
        with source.open(encoding="utf-8", newline="") as stream:
            total = sum(1 for _ in csv.DictReader(stream, delimiter=config["source"]["separator"]))
        if not total:
            raise ValueError("原始文件无数据记录")
        # UCI uses source order. Hillstrom uses label-independent record hashing.
        with source.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=config["source"]["separator"])
            if reader.fieldnames != config["source"]["expected_columns"]:
                raise ValueError("原始版本表头不一致")
            for i, row in enumerate(reader):
                # A synthetic row key, not customer identity; outcomes must not affect splits.
                key = hashlib.sha256(f"{config['dataset']}:{i}".encode()).hexdigest()
                if config["split"]["method"] == "source_order":
                    value = i / total
                else:
                    hashed = hashlib.sha256(f"{config['seeds']['split']}:{key}".encode()).hexdigest()
                    value = int(hashed[:HASH_DIGITS], HEX_BASE) / HASH_SPACE
                fractions = config["split"]["fractions"]
                split = "train" if value < fractions[0] else (
                    "validation" if value < sum(fractions[:2]) else "test")
                row.update(record_id=key, split=split)
                for party in PARTIES:
                    outputs[party].writerow({k: row[k] for k in fields[party]})
    finally:
        for stream in streams:
            stream.close()
    if sha256(source) != fingerprint:
        raise RuntimeError("原件发生变化")
    return fingerprint


def safe_receipt(receipt):
    expected = {"party", "status", "input_unchanged", "isolation_checks_passed", "artifacts",
                "training_status", "isolation_mode"}
    if set(receipt) != expected or receipt["party"] not in PARTIES or receipt["status"] != "prepared":
        raise ValueError("回执包含未许可字段或状态")
    if receipt["input_unchanged"] is not True or receipt["isolation_checks_passed"] is not True:
        raise ValueError("回执校验失败")
    if receipt["training_status"] != "blocked" or receipt["isolation_mode"] != "os_sandbox":
        raise ValueError("本轮不得宣称训练或物理隔离已完成")
    if set(receipt["artifacts"]) != SAFE_FILES:
        raise ValueError("产物清单不符合白名单")
    for value in receipt["artifacts"].values():
        if not isinstance(value, str) or len(value) != hashlib.sha256().digest_size * 2:
            raise ValueError("校验和格式非法")
        int(value, HEX_BASE)
    return receipt


def run_isolated(config_path, repo):
    repo, config_path = Path(repo).resolve(), Path(config_path).resolve()
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise RuntimeError("本入口需要 macOS 内核沙箱；禁止无隔离回退")
    cfg = yaml.safe_load(config_path.read_text())
    base = repo.parent / "8-9月"
    experiment = cfg["dataset"] + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:TOKEN_LENGTH]
    roots = {key: base / name / experiment for key, name in
             (("input", "联邦隔离输入"), ("data", "联邦隔离产物"), ("result", "联邦隔离结果"), ("journal", "实验日志"))}
    raw_root = (base / "数据集").resolve()
    for root in roots.values():
        resolved = root.resolve()
        if inside(resolved, raw_root) or inside(resolved, repo) or not inside(resolved, base.resolve()):
            raise ValueError("输出目录越界或进入原件/仓库，禁止运行")
        root.mkdir(parents=True, exist_ok=False)
    dirs = {p: {key: roots[key] / p for key in ("input", "data", "result")} for p in PARTIES}
    for locations in dirs.values():
        for folder in locations.values():
            folder.mkdir()
    source = (raw_root / cfg["source"]["relative_path"]).resolve()
    if not inside(source, raw_root):
        raise ValueError("原始输入路径越界")
    journal = roots["journal"]
    write_json(journal / "declaration.json", {
        "experiment_id": experiment, "purpose": "重新验收两方独立预处理与内核访问限制",
        "contents": "P0/P1/P2/P3/P5-inputs；不执行 PSI、训练或效果评估",
        "supersedes": "S1.P1 的集中式数据准备，不用作严格隔离验收",
        "data_source": cfg["source"], "prediction_time": cfg["prediction_time"],
        "config_sha256": sha256(config_path), "seeds": cfg["seeds"],
        "split": "source order for UCI; outcome-independent hash for Hillstrom",
        "isolation": "os_sandbox; trusted host administrator; not separate physical hosts",
        "disclosure_allowlist": ["fixed receipt schema and artifact hashes"],
    })
    events = []
    try:
        fp = distribute(cfg, source, {p: dirs[p]["input"] for p in PARTIES})
        # Positive control: every forbidden file exists and is readable by trusted setup.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port)):
                pass
            for p in PARTIES:
                (dirs[p]["data"] / "probe_existing.txt").write_text("synthetic permission probe\n")
            for p in PARTIES:
                peer = "bob" if p == "alice" else "alice"
                result = dirs[p]["result"]
                code = result / "code_snapshot"
                code.mkdir()
                for filename in ("isolated_party.py", "data_preparation.py"):
                    shutil.copyfile(Path(__file__).parent / filename, code / filename)
                notebook = repo / "modules/m1_data_selection/notebooks/S1.P2_party_preparation.ipynb"
                shutil.copyfile(notebook, code / notebook.name)
                link = result / "peer_alias"
                link.symlink_to(dirs[peer]["input"] / "input.csv")
                for path in (dirs[peer]["input"] / "input.csv", dirs[peer]["data"] / "probe_existing.txt", source):
                    with path.open("rb") as stream:
                        stream.read(1)
                mapping = {"label": cfg["label"]} if p == "alice" else {}
                if p == "alice" and "treatment" in cfg:
                    mapping["treatment"] = cfg["treatment"]
                own = {"experiment_id": experiment, "party": p, "input_path": str(dirs[p]["input"] / "input.csv"),
                       "data_dir": str(dirs[p]["data"]), "result_dir": str(result),
                       "feature_spec": cfg["parties"][p], "outcome_columns": [v["column"] for v in mapping.values()],
                       "outcome_mapping": {k: {"column": v["column"], "mapping": {str(a): b for a, b in v["mapping"].items()}} for k, v in mapping.items()},
                       "numeric_sentinels": {k: v for k, v in cfg.get("numeric_sentinels", {}).items() if k in cfg["parties"][p]["features"]},
                       "missing_categories": cfg["missing_categories"], "seeds": cfg["seeds"],
                       "config_sha256": sha256(config_path), "source_sha256": fp,
                       "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True,
                                                 capture_output=True, check=False).stdout.strip() or "unavailable",
                       "code_sha256": {f.name: sha256(f) for f in code.iterdir() if f.is_file()},
                       "python_version": sys.version,
                       "probes": {"peer_input": str(dirs[peer]["input"] / "input.csv"),
                                  "peer_output": str(dirs[peer]["data"] / "probe_existing.txt"),
                                  "full_source": str(source), "symlink_peer": str(link), "port": port}}
                write_json(result / "party_config.json", own)
                profile = result / "sandbox.sb"
                profile.write_text(profile_text(repo.parent, dirs[p]["input"], dirs[p]["data"], result, code, Path(sys.prefix)))
                command = ["/usr/bin/sandbox-exec", "-f", str(profile), sys.executable, "-I", "-B", "-c",
                           "import sys; sys.path.insert(0,sys.argv[1]); from isolated_party import execute_private_notebook; execute_private_notebook(sys.argv[2],sys.argv[3])",
                           str(code), str(result / "party_config.json"), str(code / notebook.name)]
                env = {k: os.environ[k] for k in ("PATH", "LANG") if k in os.environ}
                env.update(HOME=str(result), IPYTHONDIR=str(result / "ipython"), TMPDIR=str(result),
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONNOUSERSITE="1")
                with (result / "process.log").open("x") as log:
                    process = subprocess.run(command, cwd=result, env=env, stdout=log, stderr=log,
                                             timeout=PROCESS_TIMEOUT, check=False, close_fds=True)
                if process.returncode != 0:
                    raise RuntimeError(f"{p} 隔离运行失败（退出码 {process.returncode}）；仅查看本方日志，禁止无隔离重试")
                receipt = safe_receipt(json.loads((result / "receipt.json").read_text()))
                write_json(journal / f"{p}_receipt.json", receipt)
                events.append({"party": p, "stage": "P2/P3/P5-inputs", "status": "passed",
                               "finished_at": datetime.now(timezone.utc).isoformat(),
                               "private_data_uri": str(dirs[p]["data"]), "private_result_uri": str(result)})
        if sha256(source) != fp:
            raise RuntimeError("原件哈希改变")
        write_json(journal / "summary.json", {"status": "data_preparation_only", "events": events,
                                               "raw_unchanged": True, "training_status": "blocked",
                                               "physical_isolation_verified": False})
        write_experiment_log(journal, "data_preparation_only", events)
    except Exception as exc:
        write_json(journal / "FAILED.json", {"error": str(exc), "completed_stages": events,
                                              "status": "failed_closed", "training_status": "blocked"})
        write_experiment_log(journal, "failed_closed", events, str(exc))
        raise
    return journal
