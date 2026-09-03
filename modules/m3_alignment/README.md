# M3 · 实体对齐与样本空间构造

> 状态：🟩 **初版完成（S3.1）** ｜ 步数预算 **9** ｜ 已用 **1** ｜ 认领人：无
> 最后更新：2026-09-03 ｜ 规范位置：`docs/00-framework/` §M3 · `docs/01-loops/` LOOP M3
> 仓库：https://github.com/conceal008/HSBC-VFL-personal-project

## 目标

在跨境标识体系不同（内地手机号/身份证/中文姓名 vs 香港 HKID/护照/英文姓名）导致精确匹配覆盖率必然很低的前提下，选型对齐协议并量化合规折损漏斗。

## 上游输入

M2：仿真台（正确性验收已通过）· M0：跨境资产清单中 PSI 交集的定性

## 输出契约（本模块必须产出）

`alignment_protocol.md` · `psi_components/` · `funnel_quantification.csv` · `alignment_error_sensitivity.png+csv` · `repeated_psi_defense.md` · `DR-M3-*.yaml`

**运行入口**：本模块凡产出处理或实验结果，一律以 `notebooks/<步骤号>_<简述>.ipynb` 为入口并**带输出提交**，使人打开仓库即可直接看到结果（《维护约束 v2》9.1）。逻辑放 `components/`，参数从 `configs/` 读；真实数据派生输出必须清除后再提交。

## 量化放行判据（不达标不放行）

- 对齐路线 ≥3 条，每路线 5/5 指标（召回·精确·通信量·耗时·泄露面）
- 假匹配敏感性 5 档（0/1/3/5/10%），能回答「假匹配 5% 时是否仍优于本地基线」
- 合规折损漏斗 5/5 层，每层含区间与依据；N_eff 已给出含区间
- 重复求交泄露量降低幅度已量化；与 M0 定性一致性 100%

## 当前结论

暂无。本模块尚未开始，`step_ledger.yaml` 中 `steps` 为空。

## 未决问题

- 交集是否以秘密分享形态保存（任何一方不获得明文名单）取决于 M0 的决策点 4

## 升级条件

N_eff 上界低于 S0.1 确定的业务最小可行规模 → 上报 M0 重审同意口径与人群定义。

## 下一步

**S3.1** —— 详见 `docs/01-loops/2_模块Loop执行规范_v2_步进量化版.md` 中 LOOP M3 的步骤分解表。
执行前必须先写步骤声明（will_produce / will_not_produce / success_criteria / risk），
并在 `registry/module_status.yaml` 认领本模块。

---

*本 README 必须保持当前——它是下一个 Agent 接手时最先读的文件。每步提交时同步更新「状态 / 已用步数 / 当前结论 / 未决问题 / 下一步」五处。*

## 本模块已完成（2026-09-03，初版）

| 产出 | 位置 |
|---|---|
| PSI 行为仿真 + 错配注入 | `components/psi.py` |
| 对齐质量 / 错配影响 / 成本量级 | `results/psi_quality.csv` · `misattribution_impact.csv` · `psi_cost.csv` |
| 运行入口（带输出） | `notebooks/S3.1_psi_alignment.ipynb` |

> **降级说明（W-002）**：本模块**不实现**密码学 PSI，只仿真其可观测行为。
> 进入 M8 工业化前必须换为真实实现或成熟库。

**结论**：漏配 10% 仅损失约 12% 增益，错配需达约 40% 才使增益折半。
**实体对齐质量不是本项目的主要风险**，可按常规工程标准投入。
