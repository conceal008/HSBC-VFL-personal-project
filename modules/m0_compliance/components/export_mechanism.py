"""M0 · 依人数与敏感度判定所需的个人信息出境机制。

规则来自《促进和规范数据跨境流动规定》第五、七、八条（逐字原文见
modules/m0_compliance/legal_references.md 一之二节）。本模块只做规则套用，不做定性——
"是否敏感个人信息"由 S0.3 / S0.4 的清单给出。

2026-10-25（S0.6R，DR-MX-002 · D-3）：补入「关键信息基础设施运营者」前提。
第五条第（四）项、第七条第（二）项与第八条只适用于「关键信息基础设施运营者以外的数据处理者」；
关键信息基础设施运营者向境外提供个人信息，依第七条第（一）项一律申报数据出境安全评估。
默认 is_ciio=False，与 S0.5–S0.7 的既有结果完全一致。
"""
from __future__ import annotations

from typing import Dict, Iterable

EXEMPT = "豁免（无需申报/订立/认证）"
GBA_STANDARD_CONTRACT = "大湾区标准合同（免数量门槛）"
STANDARD_CONTRACT = "标准合同或个人信息保护认证"
SECURITY_ASSESSMENT = "数据出境安全评估"
NOT_APPLICABLE = "不适用（该方向不提供个人信息）"

# 从轻到重；同一批出境活动含多类资产时取最重者
STRICTNESS_ORDER = (NOT_APPLICABLE, EXEMPT, GBA_STANDARD_CONTRACT, STANDARD_CONTRACT,
                    SECURITY_ASSESSMENT)


def required_mechanism(person_count: int, is_sensitive: bool,
                       thresholds: Dict[str, int],
                       gba_eligible: bool = False,
                       is_ciio: bool = False) -> str:
    """返回向境外提供个人信息时所需的机制。

    person_count  自当年 1 月 1 日起累计向境外提供个人信息所涉人数
    is_sensitive  该批个人信息是否含敏感个人信息
    gba_eligible  是否适用《粤港澳大湾区（内地、香港）个人信息跨境流动标准合同》。
                  依香港数字政策办公室公布的便利措施，大湾区标准合同**免除数量门槛限制**，
                  因此适用时不再按 person_count 分档。
                  ⚠️ 适用性本身有前提（注册地、重要数据排除等），由调用方判定后传入。
    is_ciio       数据处理者是否为关键信息基础设施运营者。为 True 时依第七条第（一）项
                  一律返回数据出境安全评估。大湾区标准合同实施指引未规定关键信息基础设施
                  运营者能否适用，按 W-001「可以跨境一律取最严解释」，此时 gba_eligible 不再生效。
                  ⚠️ 是否为关键信息基础设施运营者由保护工作部门认定并通知运营者，技术侧无法判定。
    """
    if person_count <= 0:
        return NOT_APPLICABLE

    if is_ciio:
        return SECURITY_ASSESSMENT                      # 第七条第（一）项

    if gba_eligible:
        return GBA_STANDARD_CONTRACT

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


def strictest(mechanisms: Iterable[str]) -> str:
    """同一年度出境活动含多类资产（例如训练期梯度 + 试点期标识符）时，取最重的机制。"""
    items = list(mechanisms)
    if not items:
        return NOT_APPLICABLE
    unknown = [m for m in items if m not in STRICTNESS_ORDER]
    if unknown:
        raise ValueError("未知的出境机制：%s" % "、".join(unknown))
    return max(items, key=STRICTNESS_ORDER.index)
