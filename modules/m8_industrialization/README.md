# M8 · 工业化鲁棒性与部署形态

> 状态：⬜ **未开始** ｜ 步数预算 **10** ｜ 已用 **0** ｜ 认领人：无
> 最后更新：2026-08-29 ｜ 规范位置：`docs/00-framework/` §M8 · `docs/01-loops/` LOOP M8
> 仓库：https://github.com/conceal008/HSBC-VFL-personal-project

## 目标

确立离线周期性联合打分 + 名单下发的生产形态，完成 7 项故障注入，并给出单方不可用降级路径与同意撤回的工程实现。

## 上游输入

M6–M7：已收敛的隐私-效用配置

## 输出契约（本模块必须产出）

`architecture_decision.md` · `deployment_topology.md` · `fault_injection_report.md` · `monitoring_config.yaml` · `consent_revocation_flow.md` · `runbook.md` · `DR-M8-*.yaml`

## 量化放行判据（不达标不放行）

- 架构路线 3 条 × 5 维评估
- **故障注入 7/7 = 100% 通过**（断点续跑·单方下线降级·网络抖动重试·schema 变更安全失败·数据延迟批次一致·重复执行幂等·同意撤回剔除）
- 降级 RTO 已测且 ≤约定值；**「产出错误名单」事件 = 0 次**
- 监控指标 ≥4 类；名单稳定性阈值已定义；审计日志 9/9 字段
- 撤回策略 3 种已评估、1 种已选定并测试；全链路可重跑且同配置输出一致

## 当前结论

暂无。本模块尚未开始，`step_ledger.yaml` 中 `steps` 为空。

## 未决问题

- 同意撤回策略三选一（等待自然重训 / 立即触发重训 / 机器遗忘）需在本模块给出推荐并说明理由

## 升级条件

故障注入中出现「产出错误名单」且无法修复 → 禁止上线，上报架构层重新设计。错误营销名单的业务与合规损害远大于不上线的机会成本。

## 下一步

**S8.1** —— 详见 `docs/01-loops/2_模块Loop执行规范_v2_步进量化版.md` 中 LOOP M8 的步骤分解表。
执行前必须先写步骤声明（will_produce / will_not_produce / success_criteria / risk），
并在 `registry/module_status.yaml` 认领本模块。

---

*本 README 必须保持当前——它是下一个 Agent 接手时最先读的文件。每步提交时同步更新「状态 / 已用步数 / 当前结论 / 未决问题 / 下一步」五处。*
