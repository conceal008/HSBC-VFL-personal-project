# 跨境共同客户联邦营销 · 纵向联邦学习研究仓库

> **GitHub：https://github.com/conceal008/HSBC-VFL-personal-project**
> 建立日期：2026-08-29 ｜ 当前阶段：M0 未开始（仓库刚完成初始化）
> Agent 与新成员入口：先读 [AGENTS.md](AGENTS.md)

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

**[`可行性论证全过程.md`](可行性论证全过程.md)** —— M0 九个步骤的完整论证链条，410 行，一次读完。
**[`项目进度总览.md`](项目进度总览.md)** —— 当前进度快照：做到哪一步、产出了什么、卡在哪。
**[`项目目标与要求.md`](项目目标与要求.md)** —— 目标 / 要求 / 现状与约束三部分，含**一处此前无人标记的口径冲突**。

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
| 当前步骤 | 无（等待认领 M0） |
| **下一步** | **S0.1 写预测命题** → 产出 `modules/m0_compliance/problem_statement.md`（单句，含人群 / 时点 T / 窗口 Δ / 转化定义四要素 + 业务最小可行规模） |
| 主要阻塞 | S0.1 的四要素需要业务输入：产品 P 的类别、Δ 长度、转化定义（点击/申请/开户/首笔交易）、业务最小可行规模。需求方尚未确认，若无输入则须以显式假设推进并在文档中标注 |
| 合规基线 | 仓库为个人账户 Public 仓库，符合《维护约束 v2》1.1 修订后的个人仓库条款；代价是 **L-受限产物在本仓库一律按 L-禁止处理**（无个案放行通道）。判定、四项前提与推翻条件见 `registry/decision_records/DR-GOV-002.yaml` |

---

## 五、贡献与提交

一步一提交，提交前对照 [AGENTS.md](AGENTS.md) 第 3、4 节。`CHANGELOG.md` 由 CI 从 `changelog/*.yaml` 聚合生成，**禁止手工编辑**。

尚未实现的门禁（按《维护约束 v2》附录 B 的优先顺序）：`check_cross_border_consistency.py` → `check_step_metadata.py` → `check_step_scope.py` → `check_changelog_schema.py` → `check_reproducibility.py`。在其上线前，相应门禁靠人工执行，见 `ci/README.md`。
