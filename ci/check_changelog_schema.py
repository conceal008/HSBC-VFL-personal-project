#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门禁 2（下半）· changelog schema 校验（《项目维护约束 v2》4.2）

与 check_step_metadata.py 的分工：
  · check_step_metadata.py 深校验**一个**提交引用的那条 changelog（自洽性、分数、证据）
  · 本脚本结构校验**全部** changelog 条目（字段齐不齐、类型对不对、命名规不规范）
两者都通过，changelog 目录才是可被 CI 聚合成 CHANGELOG.md 的状态。

用法：python3 ci/check_changelog_schema.py [仓库根]
退出码：0 通过；1 阻断；2 环境缺失
"""
import io
import os
import re
import sys
import glob
import datetime

try:
    import yaml
except ImportError:  # pragma: no cover
    print("需要 PyYAML：pip install pyyaml")
    sys.exit(2)

TYPES = ("feat", "fix", "exp", "refactor", "doc", "decision", "revert", "chore")
STEP_EXEMPT = ("doc", "chore")
CROSS_BORDER_ENUM = ("none", "modifies_flow", "new_asset", "requires_m0_review")
ID_RE = re.compile(r"^CL-\d{8}-([A-Z]+\d*|MX|PLAT|GOV)-\d{3}$")
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")
REQUIRED = ("change_id", "timestamp", "agent", "module", "type", "title",
            "what", "why", "links", "cross_border_impact", "sensitive_review",
            "breaking_change", "reproducibility", "rollback", "verification")
TITLE_SOFT_LIMIT = 60
EMPTY_WORDS = ("优化", "改进", "完善", "调整了一下")

BLOCKS = []
WARNS = []


def block(cid, msg):
    BLOCKS.append("%s: %s" % (cid, msg))


def warn(cid, msg):
    WARNS.append("%s: %s" % (cid, msg))


def load_waivers():
    """已登记的豁免编号；未登记的 waiver_ref 一律阻断（《维护约束 v2》12.1.1）。"""
    path = "registry/waivers.yaml"
    if not os.path.exists(path):
        return set()
    try:
        data = yaml.safe_load(io.open(path, encoding="utf-8")) or {}
    except yaml.YAMLError:
        return set()
    return {str(w.get("waiver_id")) for w in (data.get("waivers") or [])
            if isinstance(w, dict) and w.get("waiver_id")}


def nonempty(value):
    return bool(str(value or "").strip())


def check_entry(path, entry):
    cid = os.path.basename(path)[:-5]

    if not isinstance(entry, dict):
        block(cid, "顶层不是映射")
        return

    for field in REQUIRED:
        if field not in entry:
            block(cid, "缺必填字段 %s" % field)

    declared = str(entry.get("change_id") or "")
    if declared != cid:
        block(cid, "change_id=%s 与文件名不一致" % declared)
    if not ID_RE.match(cid):
        block(cid, "文件名不符合 CL-<YYYYMMDD>-<模块>-<三位序号>")

    ts = entry.get("timestamp")
    # PyYAML 会把合法 ISO8601 直接解析成 datetime；字符串形式才需要正则校验
    if isinstance(ts, datetime.datetime):
        if ts.tzinfo is None:
            block(cid, "timestamp 缺时区")
    elif ts is not None and not TS_RE.match(str(ts)):
        block(cid, "timestamp 不是带时区的 ISO8601：%r" % ts)

    ctype = str(entry.get("type") or "")
    if ctype not in TYPES:
        block(cid, "type=%r 非法（应为 %s 之一）" % (ctype, "|".join(TYPES)))

    title = str(entry.get("title") or "")
    if not nonempty(title):
        block(cid, "title 为空")
    elif len(title) > TITLE_SOFT_LIMIT:
        warn(cid, "title %d 字，建议 ≤%d 字" % (len(title), TITLE_SOFT_LIMIT))

    for field in ("what", "why"):
        if not nonempty(entry.get(field)):
            block(cid, "%s 为空" % field)
    why = str(entry.get("why") or "").strip()
    if why and len(why) < 12 and any(w in why for w in EMPTY_WORDS):
        block(cid, "why 是无信息量表述（%r）——4.2 明确禁止" % why)

    impact = entry.get("cross_border_impact")
    if impact not in CROSS_BORDER_ENUM:
        block(cid, "cross_border_impact=%r 非法" % impact)
    elif impact != "none" and not nonempty(entry.get("cross_border_detail")):
        block(cid, "cross_border_impact=%s 时 cross_border_detail 不得为空" % impact)

    review = entry.get("sensitive_review")
    if not isinstance(review, dict):
        block(cid, "sensitive_review 缺失或不是映射")
    elif str(review.get("triggered")).lower() == "true" and not nonempty(review.get("conclusion")):
        block(cid, "sensitive_review.triggered=true 但 conclusion 为空")

    if entry.get("breaking_change") is True and not nonempty(entry.get("migration")):
        block(cid, "breaking_change=true 时 migration 不得为空")

    if not isinstance(entry.get("links"), dict):
        block(cid, "links 缺失或不是映射")

    repro = entry.get("reproducibility")
    if not isinstance(repro, dict):
        block(cid, "reproducibility 缺失或不是映射")
        if ctype == "exp":
            block(cid, "type=exp 必须记录 ≥5 个种子与 config 路径，但 reproducibility 段不存在")
        repro = {}
    if ctype == "exp" and isinstance(repro, dict) and repro:
        seeds = repro.get("seeds")
        if not (isinstance(seeds, list) and len(seeds) >= 5):
            block(cid, "type=exp 必须记录 ≥5 个种子（当前 %r）" % (seeds,))
        if not nonempty(repro.get("config")):
            block(cid, "type=exp 必须记录 config 路径")

    waiver = entry.get("waiver_ref")
    if waiver:
        known = load_waivers()
        if waiver not in known:
            block(cid, "waiver_ref=%s 未在 registry/waivers.yaml 登记——"
                       "未登记的绕行等同违规（12.1.1）" % waiver)

    step = entry.get("step")
    if ctype not in STEP_EXEMPT:
        if not isinstance(step, dict) or not step:
            block(cid, "type=%s 必须有 step 段" % ctype)
        else:
            for field in ("step_id", "declaration", "gates", "assessment", "verdict"):
                if not step.get(field):
                    block(cid, "step 段缺 %s" % field)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    print("== 门禁 2（下半）· changelog schema 校验 ==")

    paths = sorted(glob.glob("changelog/*.yaml"))
    print("条目数：%d" % len(paths))
    if not paths:
        print("changelog/ 为空，无对象可校验，通过。")
        return 0

    seen = {}
    for path in paths:
        try:
            with open(path, encoding="utf-8") as fh:
                entry = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            block(os.path.basename(path), "YAML 无法解析：%s" % exc)
            continue
        check_entry(path, entry)
        if isinstance(entry, dict):
            step = entry.get("step") or {}
            sid = step.get("step_id") if isinstance(step, dict) else None
            if sid:
                if sid in seen:
                    block(os.path.basename(path)[:-5],
                          "step_id=%s 与 %s 重复——一个步骤只能有一条 changelog" % (sid, seen[sid]))
                seen[sid] = os.path.basename(path)[:-5]

    print()
    for line in WARNS:
        print("⚠️  WARN   %s" % line)
    for line in BLOCKS:
        print("❌ BLOCK  %s" % line)
    print()
    print("== 结果：BLOCK=%d  WARN=%d ==" % (len(BLOCKS), len(WARNS)))
    if BLOCKS:
        print("门禁 2 未通过：changelog schema 校验失败（8.2）。")
        return 1
    print("changelog schema 校验通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
