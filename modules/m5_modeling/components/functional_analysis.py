"""Alice-authorized post-training analysis; never changes selection or test metrics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import yaml

PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
NOTEBOOK_VERSION = 4
RATE_SCALE = 100.0
MINIMUM_CONFIRMATION_SEEDS = 5
ROUTES = ("L0", "L1_secure", "L3_secure")


def protect_new_artifacts(root):
    """Restrict only the new analysis tree; never change frozen inputs or results."""
    for directory, _, files in os.walk(root, followlinks=False):
        Path(directory).chmod(PRIVATE_DIRECTORY_MODE)
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(PRIVATE_FILE_MODE)


def load(path):
    return json.loads(Path(path).read_text())


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_diagnostics(prediction, fraction, probability=True):
    k = max(1, int(fraction * len(prediction)))
    cutoff = np.sort(prediction)[-k]
    return {"distinct_scores": int(len(np.unique(prediction))),
            "at_cutoff": int(np.sum(prediction == cutoff)),
            "probability_boundary_fraction": float(np.mean((prediction == 0) | (prediction == 1))) if probability else None}


def format_metric(value, scale=1):
    outside = not value["ci_low"] <= value["estimate"] <= value["ci_high"]
    text = (f'{value["estimate"] * scale:.2f} [{value["ci_low"] * scale:.2f}, {value["ci_high"] * scale:.2f}]'
            if scale != 1 else f'{value["estimate"]:.3f} [{value["ci_low"]:.3f}, {value["ci_high"]:.3f}]')
    return text + (" †" if outside else "")


def conditional_comparison(evaluation, seeds, metric, comparator):
    """Project confirmation rule; never feeds model or parameter selection."""
    rows = {(r['seed'], r['route']): r['metrics'][metric] for r in evaluation['metrics']}
    differences = {r['seed']: r['metrics'][metric] for r in evaluation['paired_differences']
                   if r['comparison'] == f'L3_secure_minus_{comparator}'}
    supported = []
    for seed in seeds:
        joint, reference, difference = rows[seed, 'L3_secure'], rows[seed, comparator], differences[seed]
        supported.append(difference['ci_low'] > 0 and joint['ci_low'] > reference['ci_high'])
    qualifies = len(set(seeds)) >= MINIMUM_CONFIRMATION_SEEDS and all(supported)
    return ('在此构造切片下达到项目的多 seed 配对正差和区间不重叠确认规则；仍无 p 值、独立人群或业务验证。'
            if qualifies else '未达到项目的多 seed 配对正差和区间不重叠确认规则，不能确认为稳健增益；这不等于证明效应为零。')


def write_analysis(context):
    root = Path(context["round_path"])
    if load(root / "status.json")["status"] != "passed":
        raise ValueError("Only completed and collected rounds may be analyzed")
    evaluation_path = root / "alice/results/private_evaluation.json"
    evaluation = load(evaluation_path)
    cfg = yaml.safe_load((root / "alice/code/local_functional.yaml").read_text())
    declaration = load(root / "declaration.json")
    spec = cfg["datasets"][declaration["dataset"]]
    output = Path(context["analysis_root"]) / datetime.now(timezone.utc).strftime("outputs_%Y%m%dT%H%M%S%fZ")
    output.mkdir(exist_ok=False, mode=PRIVATE_DIRECTORY_MODE)
    lines = ["# 本轮模型结果与输出风险分析", "",
             f"> 编写日期：{datetime.now(timezone.utc).date().isoformat()} ｜ 版本：v1.0 ｜ 状态：私有实验分析；不得提交 Public",
             f'> 数据来源：`{evaluation_path}`；SHA256 `{fingerprint(evaluation_path)}`',
             "> 口径基准：DR-M5-001；公开代理数据、构造的字段归属、冻结预对齐切片；可信宿主管理员", "",
             f'轮次 `{root.name}`；准备切片 `{declaration["prepared_experiment"]}`。',
             f'种子 `{cfg["shared"]["model_seeds"]}`；切分种子 `{cfg["shared"]["split_seed"]}`；只用验证集选参。',
             "模型/参数/Notebook/环境与本方数据指纹在每个选定模型的 trainings/ 子目录；全部候选及失败来源均保留。", ""]
    if declaration.get("recovery_mode"):
        lines += [f'本轮仅恢复收尾；实际模型训练来源 `{declaration["training_reused_from"]}`。原轮保留 failed；输入、处理数组、代码与完整候选已复核。未重新训练、未调参，不能把复制的目录重复计为新模型。', ""]
    primary = [f"arm{arm}_centered_qini" for arm in spec["treatment_arms"]] if spec["treatment"] else ["roc_auc"]
    metric_scale = RATE_SCALE if spec["treatment"] else 1
    lines += ["邮件指标表中数值及区间均乘100，便于查看率差；图与原始 JSON 保持原始单位。" if spec["treatment"] else "响应指标与原始 JSON 使用相同单位。", ""]
    for metric in primary:
        lines += [f"## {metric}（点估计与95%区间）", "", "| seed | L0 | 安全 L1 | 安全 L3 |", "|---|---|---|---|"]
        for seed in cfg["shared"]["model_seeds"]:
            row = {item["route"]: item for item in evaluation["metrics"] if item["seed"] == seed}
            lines.append("| " + " | ".join([str(seed)] + [format_metric(row[route]["metrics"][metric], metric_scale) for route in ROUTES]) + " |")
        lines += ["", "| seed | L3−L0 | L3−安全L1 |", "|---|---|---|"]
        for seed in cfg["shared"]["model_seeds"]:
            row = {item["comparison"]: item for item in evaluation["paired_differences"] if item["seed"] == seed}
            lines.append("| " + " | ".join([str(seed)] + [format_metric(row[f"L3_secure_minus_{route}"]["metrics"][metric], metric_scale) for route in ROUTES[:-1]]) + " |")
        lines.append("")
    lines += ["## 按项目标准判断", ""]
    for metric in primary:
        for comparator in ROUTES[:-1]:
            lines.append(f"- {metric}，L3 对 {comparator}：" + conditional_comparison(
                evaluation, cfg['shared']['model_seeds'], metric, comparator))
    lines += ["", "判断规则只用于冻结结果的解释，不用于新参数选择；L0 算法可能是确定性的，不同 seed 不保证产生不同模型。", ""]
    lines += ["## 全指标与选定模型", "", "| seed | route | metric | estimate [95% CI] | selected training |", "|---|---|---|---|---|"]
    diagnostics = []
    for row in evaluation["metrics"]:
        for name, value in row["metrics"].items():
            lines.append(f'| {row["seed"]} | {row["route"]} | {name} | {format_metric(value, metric_scale)} | `{row["selected_training"]}` |')
        prediction = np.load(root / "alice/trainings" / row["selected_training"] / "results/test_predictions.npy")
        scores = {f"arm{arm}_uplift": prediction[:, arm] - prediction[:, 0] for arm in spec["treatment_arms"]} if spec["treatment"] else {"response": prediction[:, 0]}
        for name, score in scores.items():
            diagnostics.append({"seed": row["seed"], "route": row["route"], "score": name,
                                **score_diagnostics(score, cfg["shared"]["top_k_fraction"], not spec["treatment"]),
                                "scope": "post-hoc score diagnostic only; not used for model choice"})
    (output / "posthoc_score_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    lines += ["", "## 输出攻击诊断", "", "| seed | route | 假设损失型成员攻击 AUC [95% CI] |", "|---|---|---|"]
    for row in evaluation["output_attack_diagnostics"]:
        lines.append(f'| {row["seed"]} | {row["route"]} | {row["loss_attack_auc"]:.3f} [{row["ci_low"]:.3f}, {row["ci_high"]:.3f}] |')
    lines += ["", "## 解释边界与后续实验", "",
        "这是在构造的字段切分、预对齐和单一冻结切片下的结果。五个模型种子不是五个人群；区间为300次自助法的切片内条件区间，不是跨种子均值的区间。没有报告 p 值或多重比较校正，不能使用统计显著性表述。",
        "UCI 是响应预测，冻结切分按原始顺序，未验证精确日期外推；不能称因果营销增量。Hillstrom arm1=Mens E-Mail，arm2=Womens E-Mail，分别与 arm0=No E-Mail 比较；策略价值是随机邮件试验的名单回测，不能外推银行收入。",
        f"L0 在逻辑回归和树模型间用验证集选定；安全 L1/L3 是固定轮数、L2正则的线性模型，采用运行配置中的 {cfg['shared'].get('sigmoid', '未声明')} 近似。DF 在本轮合成广域网格通过；旧 SR 实際定点范围检查失败，旧 T3 有硬截断。其优化器/模型容量不同，差异不能全部归于外部特征；这些分数未做概率校准。L1 聚合表始终在安全计算内，不代表原明文低成本方案。",
        "† 表示点估计落在分位数自助区间之外；该区间形式不保证覆盖点估计。TopK/策略指标对并列分数及切片顺序敏感，重采样还会改变并列项的排序。事后并列/截断诊断另存 JSON，不用于选参，也未修改已冻结指标。正式策略结论需在新声明下固定独立于标签的并列处理，并检验区间稳定性。",
        "成员诊断假设第三方同时取得分数与标签；当前 Bob 没有这些输出，Alice 是授权接收者。训练与测试的类比例/分布漂移也能形成损失差异，攻击 AUC 高于随机不等于已证明模型记忆泄漏。此测试不能替代属性推断、特征或模型反演测试。",
        "下一轮须先声明，再用同优化器的安全 active-only 消融隔离特征增量；检查近似函数的输入范围与校准；增加独立切分或时间窗口、更多自助重复与预先定义的检验。更强安全目标还须限制候选/训练分数及查询预算，评估聚合释放/差分隐私的效果损失。不得看本轮测试结果后回头改参。",
        "本轮账户/通道/输出检查通过，只能支持声明威胁模型下的功能验收。宿主管理员可信、半诚实不串通、真实客户 PSI 未执行；不声明绝对零泄漏或生产安全。", ""]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seeds = cfg["shared"]["model_seeds"]
    figure, axes = plt.subplots(len(primary), 1, figsize=(len(seeds) + len(ROUTES), len(ROUTES) * len(primary)), squeeze=False)
    for name, axis in zip(primary, axes[:, 0]):
        for offset, route in enumerate(ROUTES):
            points = [next(row["metrics"][name] for row in evaluation["metrics"] if row["seed"] == seed and row["route"] == route) for seed in seeds]
            position = np.arange(len(seeds)) * len(ROUTES) + offset
            line = axis.scatter(position, [v["estimate"] for v in points], label=route)
            axis.vlines(position, [v["ci_low"] for v in points], [v["ci_high"] for v in points], color=line.get_facecolor())
        axis.set_xticks(np.arange(len(seeds)) * len(ROUTES) + 1, [str(seed) for seed in seeds])
        axis.set_title(name)
        axis.set_xlabel("Model seed (same frozen test slice)")
        axis.legend()
    figure.tight_layout()
    figure_path = output / "各seed主指标与区间.png"
    figure.savefig(figure_path)
    plt.close(figure)
    lines += [f"![各 seed 主指标与区间]({figure_path})", ""]
    report = output / "模型结果与输出风险分析.md"
    report.write_text("\n".join(lines))
    protect_new_artifacts(output)
    return {"status": "completed", "report": str(report), "figure": str(figure_path), "score_diagnostics": str(output / "posthoc_score_diagnostics.json")}


def create_analysis_notebook(round_path):
    import nbformat
    from nbclient import NotebookClient
    root = Path(round_path).resolve()
    if load(root / "status.json")["status"] != "passed":
        raise ValueError("Incomplete round cannot produce a final analysis notebook")
    analysis = root / "alice/results" / datetime.now(timezone.utc).strftime("analysis_%Y%m%dT%H%M%S%fZ")
    code = analysis / "code"
    analysis.mkdir(exist_ok=False, mode=PRIVATE_DIRECTORY_MODE)
    code.mkdir(mode=PRIVATE_DIRECTORY_MODE)
    shutil.copyfile(__file__, code / Path(__file__).name)
    context = {"round_path": str(root), "analysis_root": str(analysis),
               "git_sha": load(root / "declaration.json")["git_sha"],
               "code_sha256": fingerprint(code / Path(__file__).name)}
    (code / "context.json").write_text(json.dumps(context, ensure_ascii=False, indent=2))
    template = Path(__file__).resolve().parents[1] / "notebooks/S5.P2_analysis.ipynb"
    notebook = nbformat.read(template, as_version=NOTEBOOK_VERSION)
    (analysis / "logs").mkdir(mode=PRIVATE_DIRECTORY_MODE)
    environment = os.environ.copy()
    environment.update(IPYTHONDIR=str(analysis / "logs/ipython"), MPLCONFIGDIR=str(analysis / "logs/matplotlib"))
    NotebookClient(notebook, kernel_name="python3", resources={"metadata": {"path": str(analysis)}}).execute(env=environment)
    destination = analysis / "结果分析.executed.ipynb"
    nbformat.write(notebook, destination)
    protect_new_artifacts(analysis)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("round", type=Path)
    args = parser.parse_args()
    print(create_analysis_notebook(args.round))
