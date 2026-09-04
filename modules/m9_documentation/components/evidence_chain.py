# -*- coding: utf-8 -*-
"""M9 · 证据链核验：每条对外结论都必须能追到具体的产出文件。

M9 的放行判据要求「一致性核验通过率 100%」与「DPIA 风险溯源率 100%」。
这两条只有做成**机器可验**才有意义——写成「已核验」四个字，谁也不知道核了什么。

本模块把「结论 → 证据文件」的映射变成可执行检查：
- 证据文件不存在 → 该条结论**不成立**，不得对外引用
- 证据文件为空 → 同上
- 结论声明的性质（实测 / 推论 / 假设 / 未复核）必须显式，不得留空

⚠️ 本模块核验的是**可追溯性**，不是**正确性**。
   证据文件存在不代表结论正确，只代表读者能查到它依据的是什么。
"""
from __future__ import annotations

import os
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
