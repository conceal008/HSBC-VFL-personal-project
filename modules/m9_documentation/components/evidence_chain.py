# -*- coding: utf-8 -*-
"""M9 · 证据链核验：每条对外结论都必须能追到具体的产出文件。

M9 的放行判据要求「一致性核验通过率 100%」与「DPIA 风险溯源率 100%」。
这两条只有做成**机器可验**才有意义——写成「已核验」四个字，谁也不知道核了什么。

本模块把「结论 → 证据文件」的映射变成可执行检查：
- 证据文件不存在 → 该条结论**不成立**，不得对外引用
- 证据文件为空 → 同上
- 结论声明的性质（实测 / 推论 / 假设 / 未复核）必须显式，不得留空
- **文档中声明的数字必须与仓库真值一致**（`verify_consistency`）——
  入口文档里的一个陈旧数字比缺一个证据文件更容易误导人：读者不会去核对，只会照着用

⚠️ 本模块核验的是**可追溯性**，不是**正确性**。
   证据文件存在不代表结论正确，只代表读者能查到它依据的是什么。
"""
from __future__ import annotations

import glob
import io
import os
import re
from typing import Dict, List

STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_EMPTY = "empty"
NATURE_REQUIRED = ("实测", "推论", "假设", "未复核")
MIN_FILE_BYTES = 1


def _check_file(path: str) -> str:
    if not os.path.exists(path):
        return STATUS_MISSING
    if os.path.getsize(path) < MIN_FILE_BYTES:
        return STATUS_EMPTY
    return STATUS_OK


def verify_claims(claims: List[Dict]) -> Dict:
    """逐条核验结论的证据可追溯性。

    每条 claim 需含：id · statement · nature · evidence（文件列表）。
    """
    rows: List[Dict] = []
    for claim in claims:
        missing = []
        for path in claim.get("evidence") or []:
            status = _check_file(path)
            if status != STATUS_OK:
                missing.append({"path": path, "status": status})
        nature = str(claim.get("nature") or "")
        nature_ok = any(n in nature for n in NATURE_REQUIRED)
        traceable = not missing and bool(claim.get("evidence")) and nature_ok
        rows.append({
            "id": claim.get("id"), "statement": claim.get("statement"),
            "nature": nature, "n_evidence": len(claim.get("evidence") or []),
            "traceable": traceable, "problems": missing,
            "nature_declared": nature_ok})
    total = len(rows)
    ok = sum(1 for r in rows if r["traceable"])
    return {"rows": rows, "total": total, "traceable": ok,
            "rate": (ok / total) if total else 1.0,
            "untraceable_ids": [r["id"] for r in rows if not r["traceable"]]}


def verify_risk_traceability(risks: List[Dict]) -> Dict:
    """DPIA 风险溯源：每条风险必须指向具体的 M7 实验产出，不能只写「可能存在」。"""
    rows: List[Dict] = []
    for risk in risks:
        refs = risk.get("evidence") or []
        missing = [{"path": p, "status": _check_file(p)}
                   for p in refs if _check_file(p) != STATUS_OK]
        has_measure = bool(str(risk.get("mitigation") or "").strip())
        traced = bool(refs) and not missing and has_measure
        rows.append({"id": risk.get("id"), "risk": risk.get("risk"),
                     "severity": risk.get("severity"), "traced": traced,
                     "has_mitigation": has_measure, "problems": missing})
    total = len(rows)
    ok = sum(1 for r in rows if r["traced"])
    return {"rows": rows, "total": total, "traced": ok,
            "rate": (ok / total) if total else 1.0,
            "untraced_ids": [r["id"] for r in rows if not r["traced"]]}


def verify_deliverables(deliverables: List[Dict]) -> Dict:
    """交付清单核验：逐项检查文件是否真的存在。"""
    rows = []
    for item in deliverables:
        path = item.get("path", "")
        status = _check_file(path) if path else STATUS_MISSING
        rows.append({"name": item.get("name"), "path": path,
                     "status": status, "note": item.get("note", "")})
    total = len(rows)
    ok = sum(1 for r in rows if r["status"] == STATUS_OK)
    return {"rows": rows, "total": total, "present": ok,
            "rate": (ok / total) if total else 1.0,
            "missing": [r["name"] for r in rows if r["status"] != STATUS_OK]}


# 只取**紧邻关键词左侧**的那个数字，中间允许 Markdown 强调符与空格。
# 用宽窗口取全部数字会把无关的数捞进来（如「七道门禁 + 45 组用例」里的 7），
# 从而把一个本该干净的检查变成噪声源——噪声大的检查最终会被人忽略。
ADJACENT_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[*_`]*\s*$")


def _count_matching(path: str, pattern: str) -> int:
    if not os.path.exists(path):
        return -1
    rx = re.compile(pattern)
    with io.open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if rx.search(line))


def _count_files(glob_pattern: str, exclude: str = "") -> int:
    paths = glob.glob(glob_pattern, recursive=True)
    if exclude:
        paths = [p for p in paths if exclude not in p]
    return len(paths)


def resolve_source(source: Dict) -> int:
    """算出某项事实的**真值**。真值只能从仓库本身算出，不能从文档里抄。"""
    kind = source.get("type")
    if kind == "count_matching":
        return _count_matching(source["path"], source["pattern"])
    if kind == "count_files":
        return _count_files(source["glob"], source.get("exclude", ""))
    raise ValueError("未知的事实来源类型：%s" % kind)


NUMBER_WAIVER_MARK = "数字豁免:"


def _line_containing(text: str, pos: int) -> str:
    lo = text.rfind("\n", 0, pos) + 1
    hi = text.find("\n", pos)
    return text[lo:hi if hi != -1 else len(text)]


def _is_historical_mention(line: str, truth: int) -> bool:
    """判断这处数字是**历史陈述**而非过时主张。两条规则，与术语检查同源：

    S1 同一行里也出现了真值 —— 那是对照或订正，例如
       「此前写着 17 组用例、而实际已是 45 组」。
       ⚠️ 本规则只在取到的数**与真值不符**时才适用——否则会把正确陈述也跳过。
    S2 显式标记 `数字豁免: <理由>` —— 用于 S1 覆盖不到的历史记录，
       例如「S-INIT.6 当时交付了 13 份组件声明」。

    ⚠️ 代价：把陈旧值和真值写在同一行即可绕过本检查。
       接受它的理由是——真正的失误形态是「新值加上了、旧值忘删」，
       那两个值几乎总在不同位置；而刻意同行并列的，通常正是对照。
    """
    if NUMBER_WAIVER_MARK in line:
        return True
    return str(truth) in line


def verify_consistency(assertions: List[Dict]) -> Dict:
    """核验文档中声明的数字与仓库真值是否一致。

    这补的是证据链此前的一个盲区：它只查「证据文件在不在」，
    不查「文档里写的数字对不对」。而**入口文档里的一个陈旧数字，
    比缺一个证据文件更容易误导人**——读者不会去核对，只会照着用。

    做法：在关键词附近抽取全部数字，要求真值在其中，且**没有其它数字**——
    后一条才是关键，它抓的是「新值加上了、旧值没删」这类最常见的失误。
    """
    rows: List[Dict] = []
    for item in assertions:
        try:
            truth = resolve_source(item["source"])
        except (ValueError, KeyError) as exc:
            rows.append({"id": item.get("id"), "what": item.get("what"),
                         "truth": None, "ok": False,
                         "problems": [{"path": "-", "found": [], "note": str(exc)}]})
            continue
        keyword = item["context_keyword"]
        problems = []
        for path in item.get("must_appear_in") or []:
            if not os.path.exists(path):
                problems.append({"path": path, "found": [], "note": "文件不存在"})
                continue
            text = io.open(path, encoding="utf-8", errors="ignore").read()
            found: List[str] = []
            for m in re.finditer(re.escape(keyword), text):
                left = ADJACENT_NUMBER_RE.search(text[:m.start()])
                if not left:
                    continue
                value = left.group(1)
                # 跳过规则只对**与真值不符**的数字生效：
                # 取到的就是真值时它是正确陈述，必须计入，否则本检查会把「写对了」也判成「没写」。
                if value != str(truth):
                    line = _line_containing(text, m.start())
                    if _is_historical_mention(line, truth):
                        continue
                found.append(value)
            uniq = sorted(set(found))
            if str(truth) not in uniq:
                problems.append({"path": path, "found": uniq, "note": "未出现真值"})
            elif len(uniq) > 1:
                stale = [v for v in uniq if v != str(truth)]
                problems.append({"path": path, "found": uniq,
                                 "note": "同时出现陈旧值 %s" % "、".join(stale)})
        rows.append({"id": item.get("id"), "what": item.get("what"),
                     "truth": truth, "ok": not problems, "problems": problems})
    total = len(rows)
    ok = sum(1 for r in rows if r["ok"])
    return {"rows": rows, "total": total, "consistent": ok,
            "rate": (ok / total) if total else 1.0,
            "inconsistent_ids": [r["id"] for r in rows if not r["ok"]]}
