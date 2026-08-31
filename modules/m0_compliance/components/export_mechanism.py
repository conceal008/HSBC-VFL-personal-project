"""M0 · 依人数与敏感度判定所需的个人信息出境机制。

规则来自《促进和规范数据跨境流动规定》第五、七、八条（逐字原文见
modules/m0_compliance/legal_references.md 一之二节）。本模块只做规则套用，不做定性——
"是否敏感个人信息"由 S0.3 / S0.4 的清单给出。
"""
from __future__ import annotations

from typing import Dict

EXEMPT = "豁免（无需申报/订立/认证）"
STANDARD_CONTRACT = "标准合同或个人信息保护认证"
SECURITY_ASSESSMENT = "数据出境安全评估"
NOT_APPLICABLE = "不适用（该方向不提供个人信息）"


def required_mechanism(person_count: int, is_sensitive: bool,
                       thresholds: Dict[str, int]) -> str:
    """返回向境外提供个人信息时所需的机制。

    person_count  自当年 1 月 1 日起累计向境外提供个人信息所涉人数
    is_sensitive  该批个人信息是否含敏感个人信息
    """
    if person_count <= 0:
        return NOT_APPLICABLE

    if is_sensitive:
        # 第七条：1 万人以上敏感个人信息 → 安全评估
        # 第八条：不满 1 万人敏感个人信息 → 标准合同或认证
        if person_count >= thresholds["sensitive_assessment_at_or_above"]:
            return SECURITY_ASSESSMENT
        return STANDARD_CONTRACT

    # 非敏感个人信息
    if person_count < thresholds["non_sensitive_exempt_below"]:
        return EXEMPT                                   # 第五条第（四）项
    if person_count >= thresholds["non_sensitive_assessment_at_or_above"]:
        return SECURITY_ASSESSMENT                      # 第七条
    return STANDARD_CONTRACT                            # 第八条
