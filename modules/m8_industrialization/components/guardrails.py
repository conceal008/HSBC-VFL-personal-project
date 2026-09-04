# -*- coding: utf-8 -*-
"""M8 · 部署护栏：把 M7 的攻防结论变成上线前不可绕过的检查。

写成代码而不是文档的理由：**写在文档里的防护基线，上线时没人会逐条核对。**
本模块让「这套配置能不能上线」成为一个可执行的判断，
并且每条拒绝理由都指回具体的实测证据文件。

⚠️ 本项目不设合规角色，阈值为技术侧自评，未经法务复核（豁免 W-001）。
⚠️ 阈值来自合成数据实验，接入真实数据后必须重新标定。
"""
from __future__ import annotations

from typing import Dict, List

VERDICT_PASS = "pass"
VERDICT_BLOCK = "block"
CHECKED_RULES = 9            # 本模块实现的护栏规则数，供调用方核对是否漏检


class GuardrailViolation(Dict):
    """一条护栏拒绝：包含违反项、实测依据与整改方向。"""


def _violation(rule: str, detail: str, evidence: str, remedy: str) -> Dict:
    return {"rule": rule, "detail": detail, "evidence": evidence, "remedy": remedy}


def check_deployment(config: Dict, profile: Dict) -> Dict:
    """核验一份部署配置是否满足防护基线。返回判定与逐条违规。

    `config` 是拟上线的运行配置；`profile` 是 deployment_profile.yaml 的内容。
    """
    violations: List[Dict] = []

    up = profile["uplink_noise"]
    sigma = config.get("uplink_sigma")
    if sigma is None or sigma < up["sigma_min"]:
        violations.append(_violation(
            "uplink_noise.sigma_min",
            "上行噪声 σ=%s 低于基线 %s——无防护时主动方 6 个辅助样本即可精确恢复被动方特征"
            % (sigma, up["sigma_min"]),
            up["evidence"],
            "把被动方上行噪声调至 ≥%s；实测该强度下可用性仅损失 0.0014" % up["sigma_min"]))

    rp = profile["residual_plausibility"]
    if rp["enabled"] and not config.get("residual_plausibility_check"):
        violations.append(_violation(
            "residual_plausibility.enabled",
            "未对收到的残差做合法性检查——**这会让上行加噪失去意义**："
            "恶意主动方放大探针幅度即可攻破（幅度 1000 时仅需 100 次重复）",
            rp["evidence"],
            "开启残差合法性检查（范围 [%s, %s]、非零比例 ≥%s）；"
            "它把攻击成本抬高约六个数量级"
            % (rp["range_min"], rp["range_max"], rp["min_nonzero_ratio"])))

    ls = profile["list_stability"]
    threshold = config.get("list_stability_threshold")
    if threshold is None or threshold < ls["min_topk_overlap_between_runs"]:
        violations.append(_violation(
            "list_stability.min_topk_overlap_between_runs",
            "名单稳定性告警阈值 %s 低于 %s——恶意被动方以中等幅度抬分时"
            "（45%% 目标客户进入名单）重合度仍有 0.93，不会触发告警"
            % (threshold, ls["min_topk_overlap_between_runs"]),
            ls["evidence"],
            "把阈值提到 ≥%s；注意这只是检测手段，不是防护"
            % ls["min_topk_overlap_between_runs"]))

    lp = profile["label_protection"]
    if not lp["gaussian_noise_allowed"] and config.get("label_protection") == "gaussian_noise":
        violations.append(_violation(
            "label_protection.gaussian_noise_allowed",
            "把高斯噪声当作标签防护——实测无效：跨轮平均即可消噪，泄露 AUC 回到 1.0000",
            lp["evidence"],
            "改用协议级手段（安全聚合 / 同态加密 / 秘密分享）；本阶段未实现，不得带此配置上线"))

    ma = profile["model_asset_protection"]
    down = config.get("downlink_sigma")
    if down is None or down < ma["downlink_sigma_min"]:
        violations.append(_violation(
            "model_asset_protection.downlink_sigma_min",
            "下行噪声 σ=%s 低于基线 %s——无防护时被动方可**独立复现整份名单**"
            "（Top-10%% 重合度 1.000）" % (down, ma["downlink_sigma_min"]),
            ma["evidence"],
            "把下行噪声调至 ≥%s。注意它防的不是标签（对标签无效），而是模型资产；"
            "实测可用性代价几乎为零" % ma["downlink_sigma_min"]))

    mc = profile["model_capacity"]
    depth = config.get("gbdt_max_depth")
    if depth is not None and depth > mc["gbdt_max_depth"]:
        violations.append(_violation(
            "model_capacity.gbdt_max_depth",
            "树深 %s 超过上限 %s——成员推断泄露随容量上升（深 3→6：0.5277→0.5453）"
            % (depth, mc["gbdt_max_depth"]),
            mc["evidence"],
            "把树深降到 ≤%s，或提供该深度下的成员推断实测结果" % mc["gbdt_max_depth"]))

    ka = profile["k_anonymity"]
    k = config.get("k_anonymity")
    if k is not None and k < ka["k_min"]:
        violations.append(_violation(
            "k_anonymity.k_min",
            "k=%s 低于下限 %s" % (k, ka["k_min"]),
            "registry/decision_records/DR-MX-001.yaml",
            "把 k 提到 ≥%s；实测 k 取 5/10/20 对效果影响 < 0.0001，取严无代价" % ka["k_min"]))

    sn = profile["splitnn"]
    if (sn["frozen_encoder_requires_pretraining"]
            and config.get("splitnn_mode") == "frozen_random"):
        violations.append(_violation(
            "splitnn.frozen_encoder_requires_pretraining",
            "随机冻结编码器（0.7076）低于 L0 内地单方基线（0.7127）——不如不做联邦",
            sn["evidence"],
            "改用自监督预训练的冻结编码器（frozen_pca，0.7213），或改用形态A"))

    rs = profile["route_selection"]
    if (not rs["effect_ranking_admissible"]
            and config.get("route_selected_by") == "effect_ranking"):
        violations.append(_violation(
            "route_selection.effect_ranking_admissible",
            "以合成数据上的效果排序选路线——该排序已被 falsifier 推翻（强交互下排序反转）",
            rs["evidence"],
            "改以暴露面与合规成本为依据；效果维度只能作条件性陈述"))

    return {"verdict": VERDICT_PASS if not violations else VERDICT_BLOCK,
            "violations": violations,
            "checked_rules": CHECKED_RULES}
