"""Archive reviewed aggregate reports, never dataset rows or model artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import yaml

FORBIDDEN = {
    "input_sha256", "origin_hashes", "records", "rows", "universe_rows",
    "row_counts", "train_rows", "validation_rows", "test_rows", "native_reference",
    "full_unpadded_gradient_norm", "weights", "model_weights", "predictions",
    "labels", "label", "record_id", "record_ids", "ids", "customer_id", "intersection_size",
    "intersection_count", "preprocessing_state", "credentials", "secret_key",
    "x", "y", "t", "features", "raw_values", "matrix", "gradient", "token", "password",
}
MAX_SCALAR_LIST = 5
KIND_KEYS = {
    "l0": ("dataset", "status", "selected_alpha", "model_seeds", "split_seed",
           "bootstrap_seed", "config_sha256", "code_sha256", "git_sha", "metrics"),
    "functional": ("status", "selection", "round", "metrics", "paired_differences",
                   "output_attack_diagnostics"),
    "diagnostic": ("status", "round", "seeds", "numerical_diagnostics",
                   "reference_tolerance", "active_reference_equivalent", "limitations"),
    "feature": ("status", "scope", "primary", "model_seeds", "models",
                "paired_comparisons", "seed_points", "pvalues", "holm_pvalues",
                "pvalue_scope", "bootstrap", "full_data_training", "independent_confirmation"),
    "convergence": ("status", "curves", "maximum_reference_error", "paired",
                    "existing_joint_descriptive", "scope"),
    "output": ("status", "details", "aggregate", "bob_score_access", "threat_scope", "ci_scope"),
    "attribute": ("details", "paired", "auxiliary_fit_and_eval_disjoint",
                  "plaintext_truth_or_weights_disclosed", "threat_scope"),
    "math": ("status", "synthetic_only", "dataset", "summary", "scope", "business_gain_evaluated"),
    "timing": ("seconds",),
}


def project(value, omitted, prefix=""):
    """Keep aggregate structures; reject unexpected long scalar vectors."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            path = prefix + "/" + key
            if key in FORBIDDEN:
                omitted.append(path)
            else:
                result[key] = project(item, omitted, path)
        return result
    if isinstance(value, list):
        if len(value) > MAX_SCALAR_LIST and any(not isinstance(x, dict) for x in value):
            raise ValueError("Unexpected scalar vector; needs separate disclosure review")
        return [project(x, omitted, prefix) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite metric cannot be silently repaired")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("Unsupported result value")


def approved_payload(source, kind):
    if kind not in KIND_KEYS:
        raise ValueError("Unreviewed report family")
    omitted = ["/" + k for k in source if k not in KIND_KEYS[kind]]
    payload = project({k: source[k] for k in KIND_KEYS[kind] if k in source}, omitted)
    if not payload:
        raise ValueError("No reviewed report fields")
    return payload, sorted(set(omitted))


def scalars(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from scalars(item, prefix + "/" + key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from scalars(item, prefix + "/" + str(index))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def config_provenance(path, workspace):
    root = path.parent
    if root.name == "results":
        root = root.parent
    for location in (root / "code", root):
        runtimes = list(location.glob("runtime.json"))
        if runtimes:
            runtime = json.loads(runtimes[0].read_text())
            config = runtime.get("config")
            receipt = {"config_path": config, "config_sha256": runtime.get("config_sha256"),
                       "runtime_git_sha": runtime.get("git_sha"),
                       "source_owner": runtime.get("party", "alice")}
            # Only protocol parameters, never runtime endpoints or learned state.
            if config:
                local = location / Path(config).name
                if local.is_file():
                    protocol = yaml.safe_load(local.read_text())
                    receipt["seed_protocol"] = {
                        "shared": {k: v for k, v in protocol.get("shared", {}).items()
                                   if "seed" in k or k in ("bootstrap_repeats",)},
                        "experiment": {k: v for k, v in protocol.get("experiment", {}).items() if "seed" in k},
                    }
            return receipt
    return {"config_path": None, "config_sha256": None,
            "note": "source report fields / linked original protocol provide provenance"}


def archive(workspace, config, output, repo):
    workspace, output, repo = Path(workspace).resolve(), Path(output), Path(repo)
    if output.exists():
        raise FileExistsError("Frozen archive exists; choose a new version")
    cfg = yaml.safe_load(Path(config).read_text())
    reports, inventory, metric_rows = [], [], []
    seen = set()
    # Build and validate everything before writing any archive file.
    for spec in cfg["sources"]:
        for source in sorted(workspace.glob(spec["pattern"])):
            if "trainings" in source.parts or not source.is_file():
                continue
            source = source.resolve()
            if not source.is_relative_to(workspace) or source in seen:
                raise ValueError("Escaped or duplicated result source")
            seen.add(source)
            raw = json.loads(source.read_text())
            payload, omitted = approved_payload(raw, spec["kind"])
            relative = str(source.relative_to(workspace))
            dataset = next((x for x in cfg["datasets"] if x in relative), "synthetic_or_engineering")
            key = f"r{len(reports):03d}"
            report = {"id": key, "step": spec["step"], "dataset": dataset,
                      "classification": "exploratory_or_smoke" if "smoke_" in relative else spec["classification"],
                      "source_file": str(source),
                      "source_summary_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "config_provenance": config_provenance(source, workspace),
                      "omitted_fields": omitted, "results": payload}
            reports.append(report)
            metric_rows.extend({"report": key, "step": spec["step"], "dataset": dataset,
                                "metric_path": path, "value": value}
                               for path, value in scalars(payload))
    if not reports:
        raise ValueError("No reports found; do not publish an empty archive")
    for folder in cfg["status_folders"]:
        for source in sorted((workspace / folder).glob("**/status.json")):
            if any(x in source.parts for x in ("data", "code", "input", "trainings")):
                continue
            raw = json.loads(source.read_text())
            inventory.append({"source_file": str(source), "original_status": raw.get("status"),
                              "exit_codes": raw.get("exit_codes"), "error_type": raw.get("error_type"),
                              "note": "frozen historical status; old running is not proof of a live process"})
    existing = [str(p.relative_to(repo)) for p in repo.glob("modules/*/results/**/*.csv")
                if "current_phase" not in str(p)]
    index = {"step": cfg["step_id"], "disclosure_rule": "DR-GOV-012 / governance 2.7",
             "reports": [{k: v for k, v in x.items() if k != "results"} for x in reports],
             "execution_inventory": inventory, "existing_repository_result_tables": sorted(existing),
             "pending_experiments": cfg["pending"], "limitations": cfg["limitations"]}
    output.mkdir(parents=True, exist_ok=False)
    for item in reports:
        (output / (item["id"] + ".json")).write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n")
    (output / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("report", "step", "dataset", "metric_path", "value"))
        writer.writeheader()
        writer.writerows(metric_rows)
    lines = ["# 当前阶段全部实验结果索引", "", "> 编写日期：2026-10-12 ｜ 版本：v1.0 ｜ 状态：已审查汇总归档",
             "> 数据来源：index.json逐文件来源与源码/配置指纹", "> 口径基准：DR-GOV-012；非业务或生产验收", "",
             "每份JSON保留全部所选汇总、每seed与原CI；metrics.csv是数值路径长表，value同时包括estimate/ci/p/seed等，不能混作同一种指标。未创建任何新CI或重选模型。历史running状态不证明任务仍在运行。", "",
             "|步骤|数据集|分类|结果JSON|原始评价摘要|", "|---|---|---|---|---|"]
    lines.extend(f"|{x['step']}|{x['dataset']}|{x['classification']}|[{x['id']}.json]({x['id']}.json)|`{x['source_file']}`|" for x in reports)
    lines.extend(["", "完整运行状态、原来已在仓库中的合成结果表链接和待执行项目见[index.json](index.json)。字段排除记录逐文件保留；不存在的数据/区间不补造。", ""] + cfg["limitations"])
    (output / "README.md").write_text("\n".join(lines) + "\n")
    return index


def display_archive(output):
    root = Path(output)
    index = json.loads((root / "index.json").read_text())
    return {"report_count": len(index["reports"]),
            "execution_receipts": len(index["execution_inventory"]),
            "disclosure_rule": index["disclosure_rule"],
            "pending": index["pending_experiments"],
            "all_results_table": (root / "README.md").read_text()}
