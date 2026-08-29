#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门禁 2 · 步骤与元数据完整性（《项目维护约束 v2》8.2）

步进机制的强制力来源：没有它，Step-Score <10 的提交拦不住，自评就只是自觉。

校验对象：一个提交的 commit trailer + 它引用的 changelog 条目。
  · Change-Id  必须存在且对应 changelog/<id>.yaml 真实存在
  · Step-Id    必填（doc / chore 除外）
  · Step-Score 必须 ≥ 通过线（默认 10；模块 step_ledger 上调后取上调值）
  · Cross-Border 必填且为四枚举之一
  · Branch-Card 在 type=exp 时必填
  · 单步规模不作校验（2026-08-30 依 DR-GOV-003 取消上限）
  · changelog 的 step.declaration 非空；四维评分每项有 evidence；修正轮次 ≤2
  · verdict=committed 时四道硬门必须全 pass、无单项 ≤1、总分 = 四维之和
  · type=decision 必须有 DR 且 DR 含 falsifier
  · sensitive_review.triggered=true 时 conclusion 不得为空

用法：python3 ci/check_step_metadata.py [--rev <sha>] [仓库根]
退出码：0 通过；1 阻断；2 环境缺失
"""
import os
import re
import sys
import subprocess

try:
    import yaml
except ImportError:  # pragma: no cover
    print("需要 PyYAML：pip install pyyaml")
    sys.exit(2)

TYPES = ("feat", "fix", "exp", "refactor", "doc", "decision", "revert", "chore")
STEP_EXEMPT = ("doc", "chore")          # 这两类可简化 step 段，Step-Id 非必填
CROSS_BORDER_ENUM = ("none", "modifies_flow", "new_asset", "requires_m0_review")
DIMENSIONS = ("D1_contract", "D2_verifiability", "D3_reproducibility", "D4_information_gain")
DEFAULT_PASS = 10
MAX_REVISIONS = 2

BLOCKS = []


def block(item, msg):
    BLOCKS.append("[%s] %s" % (item, msg))


def git(*args):
    return subprocess.check_output(["git"] + list(args), text=True).strip()


def parse_trailers(message):
    trailers = {}
    for line in message.splitlines():
        m = re.match(r"^([A-Za-z][A-Za-z-]*):\s*(.+?)\s*$", line)
        if m:
            trailers[m.group(1)] = m.group(2)
    return trailers


def load_yaml(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def pass_threshold(module):
    """模块触发回溯抽查上调后通过线为 11，否则 10。"""
    status_path = "registry/module_status.yaml"
    if not (module and os.path.exists(status_path)):
        return DEFAULT_PASS
    entry = (load_yaml(status_path).get("modules") or {}).get(module) or {}
    ledger = os.path.join(entry.get("path", ""), "step_ledger.yaml")
    if entry.get("path") and os.path.exists(ledger):
        metrics = load_yaml(ledger).get("metrics") or {}
        try:
            return int(metrics.get("pass_threshold", DEFAULT_PASS))
        except (TypeError, ValueError):
            pass
    return DEFAULT_PASS


def check_step_section(entry, ctype, threshold, trailers):
    step = entry.get("step")
    if not isinstance(step, dict) or not step:
        if ctype in STEP_EXEMPT:
            return
        block("step", "changelog 无 step 段（仅 doc / chore 可省略）")
        return

    if not str(step.get("declaration") or "").strip():
        block("step.declaration", "步骤声明为空——未写声明就执行，违反禁止事项第 5 条")

    if not str(step.get("next_step") or "").strip():
        block("step.next_step", "未声明下一步（阶段五要求定稿时声明下一步）")

    revisions = step.get("revisions")
    if revisions is None:
        block("step.revisions", "缺 revisions 字段")
    else:
        try:
            if int(revisions) > MAX_REVISIONS:
                block("step.revisions", "修正轮次 %s > %d——两轮后仍不达标必须升级上报，不得强行提交"
                      % (revisions, MAX_REVISIONS))
        except (TypeError, ValueError):
            block("step.revisions", "revisions 不是整数：%r" % revisions)

    gates = step.get("gates") or {}
    assessment = step.get("assessment") or {}
    verdict = str(step.get("verdict") or "")

    if verdict == "committed":
        for name, value in sorted(gates.items()):
            if str(value) != "pass":
                block("step.gates", "%s=%s——硬门 fail 即本步作废重做，不得提交" % (name, value))
        if len(gates) < 4:
            block("step.gates", "四道硬门 G1–G4 未全部记录（当前 %d 项）" % len(gates))
    elif verdict:
        block("step.verdict", "verdict=%s 的步骤不应出现在提交里（只有 committed 可提交）" % verdict)
    else:
        block("step.verdict", "缺 verdict")

    evidence = assessment.get("evidence") or {}
    scores = []
    for dim in DIMENSIONS:
        key = dim.split("_")[0]
        if dim not in assessment:
            block("step.assessment", "缺维度 %s" % dim)
            continue
        try:
            score = int(assessment[dim])
        except (TypeError, ValueError):
            block("step.assessment", "%s 不是整数：%r" % (dim, assessment[dim]))
            continue
        scores.append(score)
        if not str(evidence.get(key) or "").strip():
            block("step.assessment", "%s 无证据引用——按 1.3 阶段三，无证据的评分自动降为 1 分" % dim)
        if verdict == "committed" and score <= 1:
            block("step.assessment", "%s=%d：有单项 ≤1 时判定为重做，不得提交" % (dim, score))

    total = assessment.get("total")
    if total is None:
        block("step.assessment", "缺 total")
    elif len(scores) == len(DIMENSIONS) and int(total) != sum(scores):
        block("step.assessment", "total=%s 与四维之和 %d 不符" % (total, sum(scores)))
    elif int(total) < threshold:
        block("step.assessment", "总分 %s < 通过线 %d" % (total, threshold))

    declared = trailers.get("Step-Score")
    if declared:
        want = "%s/12" % total
        if declared != want:
            block("Step-Score", "commit trailer 写 %s，changelog 写 %s——两者必须一致" % (declared, want))

    if trailers.get("Step-Id") and str(step.get("step_id") or "") != trailers["Step-Id"]:
        block("Step-Id", "commit trailer=%s 与 changelog step_id=%s 不一致"
              % (trailers["Step-Id"], step.get("step_id")))

    # 单步规模上限已于 2026-08-30 依 DR-GOV-003 取消：scope 段只作记录，不作放行判据。
    # 这里只提醒字段缺失，不阻断。
    if not (step.get("scope") or {}):
        print("提示：step.scope 未填写实际规模（不影响放行）")


def check_entry(entry, path, ctype, trailers):
    threshold = pass_threshold(entry.get("module"))

    impact = entry.get("cross_border_impact")
    if impact not in CROSS_BORDER_ENUM:
        block("cross_border_impact", "缺失或非法枚举：%r（应为 %s 之一）" % (impact, "/".join(CROSS_BORDER_ENUM)))
    if trailers.get("Cross-Border") and trailers["Cross-Border"] != str(impact):
        block("Cross-Border", "commit trailer=%s 与 changelog cross_border_impact=%s 不一致"
              % (trailers["Cross-Border"], impact))

    review = entry.get("sensitive_review") or {}
    if str(review.get("triggered")).lower() == "true" and not str(review.get("conclusion") or "").strip():
        block("sensitive_review", "triggered=true 但 conclusion 为空")

    links = entry.get("links") or {}
    if ctype == "exp":
        card = str(links.get("branch_card") or "")
        if not card or card == "none":
            block("branch_card", "type=exp 必须绑定分支卡（links.branch_card）")
    if ctype == "decision":
        dr = str(links.get("decision_record") or "")
        if not dr or dr == "none":
            block("decision_record", "type=decision 必须有 DR")
        else:
            dr_id = dr.split()[0].split("·")[0].strip()
            dr_path = "registry/decision_records/%s.yaml" % dr_id
            if not os.path.exists(dr_path):
                block("decision_record", "DR 文件不存在：%s" % dr_path)
            elif not str(load_yaml(dr_path).get("falsifier") or "").strip():
                block("decision_record", "%s 缺 falsifier——写不出则决策不成立（0.4 第五步）" % dr_id)

    check_step_section(entry, ctype, threshold, trailers)
    print("通过线：%d ｜ changelog：%s" % (threshold, path))


def main():
    argv = sys.argv[1:]
    rev = "HEAD"
    if "--rev" in argv:
        i = argv.index("--rev")
        rev = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    root = argv[0] if argv else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    print("== 门禁 2 · 步骤与元数据完整性 ==")
    message = git("log", "-1", "--format=%B", rev)
    subject = message.splitlines()[0] if message else ""
    print("提交：%s %s" % (git("log", "-1", "--format=%h", rev), subject))

    if len(git("log", "-1", "--format=%P", rev).split()) > 1:
        print("合并提交，跳过。")
        return 0

    trailers = parse_trailers(message)
    m = re.match(r"^(%s)\(([^)]+)\):" % "|".join(TYPES), subject)
    if not m:
        block("subject", "首行不符合 `<type>(<module>): <摘要>`，type 取 %s" % "|".join(TYPES))
        ctype = ""
    else:
        ctype = m.group(1)

    change_id = trailers.get("Change-Id")
    if not change_id:
        block("Change-Id", "缺失（禁止事项第 10 条）")
    else:
        path = "changelog/%s.yaml" % change_id
        if not os.path.exists(path):
            block("Change-Id", "对应文件不存在：%s" % path)
        else:
            check_entry(load_yaml(path), path, ctype, trailers)

    if not trailers.get("Step-Id") and ctype not in STEP_EXEMPT:
        block("Step-Id", "缺失（doc / chore 除外）")
    if "Cross-Border" not in trailers:
        block("Cross-Border", "trailer 缺失，必填四枚举之一")

    print()
    for line in BLOCKS:
        print("❌ BLOCK  %s" % line)
    print()
    print("== 结果：BLOCK=%d ==" % len(BLOCKS))
    if BLOCKS:
        print("门禁 2 未通过。Step-Score <10 或元数据不全的提交不允许存在（8.2）。")
        return 1
    print("门禁 2 通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
