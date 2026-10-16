# 跨境共同客户联邦营销研究

> GitHub：[conceal008/HSBC-VFL-personal-project](https://github.com/conceal008/HSBC-VFL-personal-project) ｜ **Public / main**
> 当前状态（2026-10-15）：**所有实验已按用户要求暂停，双VM已正常停止。**
> Agent先读[AGENTS.md](AGENTS.md)，再读[暂停与接手入口](docs/03-handoff/START_HERE.md)。接手不自动授权恢复实验。

本项目研究内地与香港双方共同客户的联合营销：联邦模型相对强单方和低成本聚合统计方案，在哪些条件下有增量，以及如何约束中间信息和模型输出风险。当前证据来自公开研究代理和合成数据，尚不构成银行业务验证或生产准入。

## 当前进度

已完成14项本地代理任务、每任务5 seed、280个最终逻辑模型（270新拟合、10完整复用），另列4项历史任务参考。完整指标、每seed、置信区间和失败记录已公开审查。Amex继续暂缓。

新增S5.D2强单方端点诊断仅一次隔离预检失败，**0任务完成、没有新CI/效果结论**；修正一未测试/未运行，草稿已单独归档。新14任务输出重评、秘密PSI接入训练、分组正则/安全树、低成本L1比较和真正未查看人群确认待完成。

| 需要了解什么 | 入口 |
|---|---|
| 新Agent接手顺序与暂停边界 | [START_HERE](docs/03-handoff/START_HERE.md) |
| 已完成、有效结论、未完成和审计限制 | [当前项目状态](docs/03-handoff/CURRENT_STATUS.md) |
| 每轮目的、方法、代码、配置、结果和失败 | [实验总台账](docs/04-experiment-records/实验总台账.md) |
| 14＋4任务具体指标 | [完整结果报告](docs/04-experiment-records/reports/本地数据集联邦训练与结果汇总_v2.md) |
| 全部原值JSON/CSV与带输出Notebook | [冻结结果](modules/m5_modeling/results/portfolio_training_v2/README.md) / [Notebook](modules/m5_modeling/notebooks/S5.X3_portfolio_summary.ipynb) |
| 历史逐轮日志与本次归档日志 | [版本导航](docs/04-experiment-records/logbooks/README.md) / [v11](docs/04-experiment-records/logbooks/实验方法与运行日志_v11.md) |
| 本地资产、权限、双VM和私有恢复备份 | [文件与环境地图](docs/03-handoff/FILESYSTEM_MAP.md) |
| 暂停草稿的缺陷与恢复条件 | [S5.D2暂停归档](modules/m5_modeling/experiments/paused/S5.D2_20261015/README.md) |
| 用户恢复后应做的顺序 | [恢复计划](docs/03-handoff/RESTART_PLAN.md) |
| 决策、质量、阶段状态与交接 | [registry](registry/module_status.yaml) / [本次交接](docs/03-handoff/HO-20261015-Codex-M9-paused-handoff.md) |

## 目录职责

```text
docs/00-framework/            项目框架与量化目标
docs/01-loops/                各模块执行步骤
docs/02-governance/           数据、提交、日志与门禁规则
docs/03-handoff/              当前接手入口和历次交接
docs/04-experiment-records/   历史日志、完整报告、总台账、迁移映射
modules/m0_* … m9_*/          模块代码/配置/Notebook/冻结结果/测试
registry/                    决策、分支卡、组件声明、模块状态
changelog/                   一步一提交的独立变更记录
platform/                    共享平台层
ci/                          工程与证据链门禁
```

本机唯一代码来源是本仓库；`8-9月/项目交接中心/README.md`负责私有原始数据、实验目录、环境和备份的分类导航。原始/清洗行、标识/交集规模、逐行预测/标签、学习预处理状态、权重/份额和凭证不入Public。冻结results与源代码/配置原位保留，迁移仅针对日志/报告和未验收草稿，旧路径仍可导航。

## 结论的适用范围

联合信息收益随任务与字段构造变化，不保证优于强单方。5 seed条件符号翻转最小双侧p=0.0625，开发CI不等于独立人群增益。预对齐双VM模拟依赖可信宿主管理员、半诚实不合谋假设；秘密输出PSI原型未接训练，不能称生产安全或零泄漏。营销捕获率不能替代因果新增收益。

旧README和初始化预算/PR状态保留为[整理前历史快照](docs/04-experiment-records/history/README_整理前快照_20261015.md)，旧根目录实验日志保留兼容导航。正式业务、合规假设与失效自评继续以[目标要求](项目目标与要求.md)、各模块台账/audit_log及对外表述红线为准。

按[AGENTS.md](AGENTS.md)执行一步一提交和数据审查；本次结构整理完成工程门禁后正常推送main，不改Private、不删除历史、不强推。
