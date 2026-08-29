# M9 · 合规证据链与交付物

> 状态：⬜ **未开始** ｜ 步数预算 **10** ｜ 已用 **0** ｜ 认领人：无
> 最后更新：2026-08-29 ｜ 规范位置：`docs/00-framework/` §M9 · `docs/01-loops/` LOOP M9
> 仓库：https://github.com/conceal008/HSBC-VFL-personal-project

## 目标

验收人不是技术方，而是法务与合规。第一步必须是一致性核验：从 M0 跨境对象清单出发，逐条追溯其在 M3–M8 的实际实现。

## 上游输入

M0–M8 全部产出

## 输出契约（本模块必须产出）

`DPIA.md` · `cross_border_filing_package/` · `consent_management_spec.md` · `data_cards/` · `model_card.md` · `cross_border_flow_registry.csv` · `audit_log_spec.md` · `residual_risk_signoff.yaml` · `DR-M9-*.yaml`

## 量化放行判据（不达标不放行）

- 交付清单 10/10 完成
- **一致性核验通过率 100%**
- DPIA 风险溯源率 100%（每条对应 M7 具体实验）
- 模型卡分群数 ≥3；跨境流动清单 4/4 字段（路径·频次·量级·依据）
- 残余风险接受方 100% 已指定；**法务追问次数 = 0**

## 当前结论

暂无。本模块尚未开始，`step_ledger.yaml` 中 `steps` 为空。

## 未决问题

- （暂无，随工作展开补充）

## 升级条件

一致性核验发现不符 → 不得通过修改文档对齐，必须回到对应模块修改实现。

## 下一步

**S9.1** —— 详见 `docs/01-loops/2_模块Loop执行规范_v2_步进量化版.md` 中 LOOP M9 的步骤分解表。
执行前必须先写步骤声明（will_produce / will_not_produce / success_criteria / risk），
并在 `registry/module_status.yaml` 认领本模块。

---

*本 README 必须保持当前——它是下一个 Agent 接手时最先读的文件。每步提交时同步更新「状态 / 已用步数 / 当前结论 / 未决问题 / 下一步」五处。*
