# M3 · 实体对齐与样本空间构造

> 状态：⬜ **未开始** ｜ 步数预算 **9** ｜ 已用 **0** ｜ 认领人：无
> 最后更新：2026-08-29 ｜ 规范位置：`docs/00-framework/` §M3 · `docs/01-loops/` LOOP M3
> 仓库：https://github.com/conceal008/HSBC-VFL-personal-project

## 目标

在跨境标识体系不同（内地手机号/身份证/中文姓名 vs 香港 HKID/护照/英文姓名）导致精确匹配覆盖率必然很低的前提下，选型对齐协议并量化合规折损漏斗。

## 上游输入

M2：仿真台（正确性验收已通过）· M0：跨境资产清单中 PSI 交集的定性

## 输出契约（本模块必须产出）

`alignment_protocol.md` · `psi_components/` · `funnel_quantification.csv` · `alignment_error_sensitivity.png+csv` · `repeated_psi_defense.md` · `DR-M3-*.yaml`

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
