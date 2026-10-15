# 跨境共同客户联邦营销 · 纵向联邦学习研究仓库

> **GitHub：https://github.com/conceal008/HSBC-VFL-personal-project**
> 建立日期：2026-08-29 ｜ 当前阶段：十四本地代理任务五seed开发基线已完成，接续联合诊断、输出风险重评与安全协议集成
> Agent 与新成员入口：先读 [AGENTS.md](AGENTS.md)
> 2026-09-29 进度补记：M1 数据准备的 S1.P1–S1.P4 已推送到 [草稿 PR #1](https://github.com/conceal008/HSBC-VFL-personal-project/pull/1)，[远端门禁全部通过](https://github.com/conceal008/HSBC-VFL-personal-project/actions/runs/36535579716)；这些实现尚未合入 `main`，正式 S1.1 尚未完成。
> `main` 工程门禁补录：修复提交 `6e5d749` 的 [Actions 运行 36561258977](https://github.com/conceal008/HSBC-VFL-personal-project/actions/runs/36561258977) 八项作业全部通过；这与草稿 PR 的验收分别记录。

---

## 一、这个仓库回答什么问题

在内地与香港两个法人主体之间，针对**双方共同客户**联合建模以提升产品营销转化，全程满足两地法规约束。

核心问题不是"联邦模型 AUC 能做到多少"，而是三个更难的问题：

1. 联邦建模（L3）相对**低成本替代方案**（L1：本地模型 + 对方 k-匿名聚合统计）到底有没有增量？在什么条件下有？
2. 合规折损之后**还剩多少可用样本**？可用人群 = 可匹配 ∩ 交集 ∩ 内地单独同意 ∩ 香港直销同意 ∩ 质量达标。
3. 营销的正确目标是**增量转化（uplift）**，不是响应率。响应率提升可能只是把本来就会买的人识别得更准。

结论允许是负面的——"VFL 在本场景不划算"同样是有交付价值的结论，前提是论证扎实。

---

## 一之二、先读这一份

**[`实验方法与运行日志.md`](实验方法与运行日志_v10.md)** —— **2026-10-15当前实验入口**：全部具体指标/每seed/CI/失败记录与逐轮核心方法、源码、Notebook；后续先更新main再进入下一轮。

**[`项目框架与推进总览.md`](项目框架与推进总览.md)** —— ★ **当前统一入口**：以 8–9 月和根目录材料为基准，重整目标、规划方向、模块成熟度和下一轮任务。
**[`可行性论证全过程.md`](可行性论证全过程.md)** —— M0 九个步骤的完整论证链条，410 行，一次读完。
**[`项目进度总览.md`](项目进度总览.md)** —— 2026-09-04 历史快照；当前状态以《项目框架与推进总览》和模块台账为准。
**[`项目目标与要求.md`](项目目标与要求.md)** —— 目标 / 要求 / 现状与约束三部分，含**一处此前无人标记的口径冲突**。
**[`实验结果与结论.md`](实验结果与结论.md)** —— 初版实测结果与三个主结论，含适用边界与已知缺口。
**[`初版完成说明与降级清单.md`](初版完成说明与降级清单.md)** —— 降级了哪些标准、代偿是什么、哪些已恢复。
**[`modules/m9_documentation/对外表述红线.md`](modules/m9_documentation/对外表述红线.md)** —— 🔴 **对外材料发布前必读**；条目已随后续安全实验扩展。
**[`modules/m8_industrialization/上线准入清单.md`](modules/m8_industrialization/上线准入清单.md)** —— 上线前逐条勾选；阻塞状态须结合后续 M7 产物重新核对。

它回答：命题怎么定的 · 为什么可用样本只有几千人 · 为什么可验证性与轻量出境路径不可兼得 ·
哪些数字是法条事实、哪些是假设 · 论证可能错在哪 · 现在等谁拍板。

> 结论先行：**在当前命题与当前假设下，这件事无法被验证**——不是效果不好，也不是合规过不去，
> 而是可用人群太少，实验检不出目标效应。但这个结论建立在两个只有需求方能提供的数字上，
> **给出真实值就可能推翻它**。

## 二、目录导航

```
├── AGENTS.md                  ← Agent 入口，先读这个
├── docs/
│   ├── 00-framework/          方案流程框架 v2（做什么·量化放行判据）
│   ├── 01-loops/              模块 Loop 执行规范 v2（每步怎么做·自评打分）
│   ├── 02-governance/         项目维护约束 v2（怎么协作留痕·门禁·禁止事项）
│   └── 03-handoff/            会话交接记录 HO-*
├── registry/                  【核心】可机器核验的登记表
│   ├── cross_border_assets.yaml   跨境资产清单（M0 唯一权威，当前为空模板）
│   ├── glossary.yaml              权威术语表（全项目唯一术语来源，第 13 部分）
│   ├── module_status.yaml         十模块状态与认领
│   ├── decision_records/          DR-*.yaml，append-only
│   ├── branch_cards/              分支卡（含失败分支），append-only
│   └── component_declarations/    组件级跨境资产声明
├── modules/m0_compliance … m9_documentation/
│   每模块：README.md（当前状态）· step_ledger.yaml（步骤台账）· components/ configs/
│           notebooks/（处理与实验的运行入口，.ipynb 带输出提交）· experiments/ results/ tests/
├── platform/                  跨模块共享平台层（devices/dataframe/operators/components/orchestration/governance）
├── party_cn/ party_hk/        两侧分离，物理体现"互不可读对方数据"
├── changelog/                 变更条目，每条一文件；CHANGELOG.md 由 CI 聚合生成
└── ci/                        门禁脚本
```

## 三、模块与预算

> ⚠️ 下表是仓库初始化时的预算基线，保留用于追溯，**不是当前进度**。截至本次整理，模块台账共记录 26 条步骤；各模块成熟度见《项目框架与推进总览》§四。

| 模块 | 目录 | 步数预算 | 状态 |
|---|---|---|---|
| M0 业务问题定义与合规边界 | `modules/m0_compliance` | 10 | ⬜ 未开始 |
| M1 数据资产盘点与数据集选型 | `modules/m1_data_selection` | 9 | ⬜ 未开始 |
| M2 合成数据生成与仿真台 | `modules/m2_synthetic` | 9 | ⬜ 未开始 |
| M3 实体对齐与样本空间构造 | `modules/m3_alignment` | 9 | ⬜ 未开始 |
| M4 特征工程与跨域特征治理 | `modules/m4_features` | 10 | ⬜ 未开始 |
| M5 建模方案与基线阶梯 | `modules/m5_modeling` | 12 | ⬜ 未开始 |
| M6 效果验证与评估体系 | `modules/m6_evaluation` | 9 | ⬜ 未开始 |
| M7 安全攻防与隐私评测 | `modules/m7_security` | 15 | ⬜ 未开始 |
| M8 工业化鲁棒性与部署形态 | `modules/m8_industrialization` | 10 | ⬜ 未开始 |
| M9 合规证据链与交付物 | `modules/m9_documentation` | 10 | ⬜ 未开始 |
| **合计** | | **103** | 已用 0 步 |

推进顺序：M0 → M1 → M2 → M3 → M4 → M5（**先 L0/L1，后 L3**）→ M6 ⇄ M7 → M8 → M9。

---

## 四、当前状态与下一步

| 项 | 内容 |
|---|---|
| 已完成 | 仓库初始化（骨架 · 三份规范就位 · 登记表模板 · 十模块台账 · 数据合规门禁 · GitHub 接入） |
| 当前步骤 | S5.X3十四本地代理任务五seed完成，280最终逻辑模型（270新拟合、10完整复用）；全部指标、CI、代码和失败记录见[逐轮实验索引](实验方法与运行日志_v10.md)。新模型输出重评、秘密PSI接训练及独立人群确认待完成。 |
| **下一步** | 新增模型输出攻击需另声明；新独立人群来源待定，已见test不得重用。正式业务门禁不变 |
| 主要阻塞 | S0.1 的四要素需要业务输入：产品 P 的类别、Δ 长度、转化定义（点击/申请/开户/首笔交易）、业务最小可行规模。需求方尚未确认，若无输入则须以显式假设推进并在文档中标注 |
| 当前存放基线 | 2026-10-12按所有者明确要求保持 **Public**；经审查的公开研究数据集实验汇总指标、每seed结果、CI和日志可公开。旧Public全面禁止受限指标和短暂Private方案作为历史保留；原始/清洗行数据、逐行预测、身份/交集规模、学习预处理状态、份额及凭证仍不入库。见 `registry/decision_records/DR-GOV-012.yaml` 与维护约束§2.7 |

---

## 五、贡献与提交

一步一提交，提交前对照 [AGENTS.md](AGENTS.md) 第 3、4 节。`CHANGELOG.md` 由 CI 从 `changelog/*.yaml` 聚合生成，**禁止手工编辑**。

当前提交/推送检查统一通过 `bash ci/run_all_gates.sh` 执行数据、步骤、schema、合规一致性、代码质量、可复现性、冒烟、证据链与门禁回归；结果随每步骤changelog留痕。远端PR检查状态需另查，不能由本地通过代替。

当前实验状态（2026-10-12）：[S7.P2新增输出重评完整指标与分析](modules/m7_security/新增输出重评实验记录.md)及[带输出汇总Notebook](modules/m7_security/notebooks/S7.P2_output_reaudit_summary.ipynb)追加交付；旧阶段索引保持冻结。

当前实验状态（2026-10-13）：S5.X1两新增代理训练/来源核验完成，[全结果](modules/m5_modeling/results/external_training_v1/README.md)。S7.P2远端八项CI已成功；秘密输出PSI、正则/安全树及独立确认仍待执行。

当前对齐状态（2026-10-13）：[通用秘密求交](modules/m3_alignment/通用秘密输出求交实验记录.md)十轮实际完成、四输出秘密態；未接入旧训练。下一阶段按照用户最新要求覆盖本地其他数据集及HSBC_mock，Amex继续暂缓。
