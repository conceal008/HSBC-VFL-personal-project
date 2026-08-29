# AGENTS.md · 本仓库的 Agent 入口

> **动手改任何文件之前，必读本文件。读完本文件后必须再读 `docs/02-governance/3_项目维护约束_v2_步进量化版.md`。**

---

## 0. 仓库坐标（唯一远端）

| 项 | 值 |
|---|---|
| **GitHub 远端** | **https://github.com/conceal008/HSBC-VFL-personal-project** |
| clone（HTTPS） | `git clone https://github.com/conceal008/HSBC-VFL-personal-project.git` |
| 默认分支 | `main` |
| 可见性 | **Public**（⚠️ 偏离《维护约束 v2》1.1 的 Private 要求，见 `registry/decision_records/DR-GOV-001.yaml`） |
| 本地路径（当前维护者机器） | `/Applications/总文件夹/汇丰/HSBC-VFL-personal-project` |

**任何 Agent 完成工作后，必须把改动提交并推送到上述远端。** 未推送的产出按《维护约束 v2》原则一视为不存在。

> ⚠️ 因仓库为 Public：`registry/`、`modules/*/results/` 等一切内容都是世界可读。提交前逐条对照《维护约束 v2》第 2 部分的四级分级表，**L-禁止与未审查的 L-受限一律不得提交**。存疑时按 L-禁止处理。

---

## 1. 本项目是什么

跨境（内地 ↔ 香港）**共同客户联合营销**场景下的纵向联邦学习方案研究。目标不是"把 AUC 做高"，而是回答：**在两地法规约束下，联邦建模相对低成本替代方案（本地模型 + 对方 k-匿名聚合统计）到底有没有业务增量、在什么条件下有。**

方案按 M0–M9 十个模块推进，共 103 步预算。三份规范同时生效：

| 文件 | 定义什么 | 位置 |
|---|---|---|
| 方案流程框架 v2 | **做什么**（模块目标、量化放行判据） | `docs/00-framework/` |
| 模块 Loop 执行规范 v2 | **每一步怎么做**（步进式微循环、自评打分） | `docs/01-loops/` |
| 项目维护约束 v2 | **怎么协作与留痕**（提交、日志、门禁、禁止事项） | `docs/02-governance/` |

冲突时以《项目维护约束 v2》的**禁止性条款**优先。

---

## 2. 每次会话的标准动作

**开始时（顺序不可乱）**

```
1. 读 AGENTS.md（本文）
2. 读 docs/02-governance/3_项目维护约束_v2_步进量化版.md
3. 读 registry/module_status.yaml       → 确认目标模块没被别人认领
4. 读 modules/<目标模块>/README.md      → 当前状态·结论·未决问题
5. 读 docs/03-handoff/ 最近一份交接记录
6. 读 modules/<目标模块>/step_ledger.yaml → 当前是第几步
7. 认领模块：在 module_status.yaml 写 claimed_by / claimed_at / expires_at(24h) / current_step
```

**每一步（严格一步一提交）**

```
声明（will_produce / will_not_produce / success_criteria / risk）
  → 执行（超出 will_not_produce 边界立即停，记为下一步）
  → 自评（G1–G4 硬门 + D1–D4 打分，每项必须有证据引用）
  → 判定（≥10 通过；8–9 修正一轮；≤7 或任一项=0 重做；修正上限 2 轮）
  → 提交（changelog 条目 + step_ledger 更新 + 声明下一步）
  → 推送到 GitHub 远端
```

单步规模上限：**≤3 文件 / ≤300 行代码 / ≤1 路线 / ≤1 结论 / ≤15 次工具调用**。

**结束时**

```
1. 写 docs/03-handoff/HO-<YYYYMMDD>-<Agent>-<模块>.md（不写 = 会话不算结束）
2. 更新模块 README 与 step_ledger.yaml
3. 更新 registry/module_status.yaml（含未决问题）
4. 确认 changelog 条目已写、失败分支卡已归档
5. git push
```

---

## 3. 提交格式（CI 将校验五个 trailer）

```
<type>(<module>): <一行摘要>

<正文：改了什么、为什么>

Change-Id: CL-<YYYYMMDD>-<模块>-<三位序号>
Step-Id: S<模块号>.<步号>
Step-Score: <总分>/12
Branch-Card: <分支卡ID 或 none>
Cross-Border: none | modifies_flow | new_asset | requires_m0_review
```

`Step-Score < 10` 的提交不允许存在。type 取 `feat|fix|exp|refactor|doc|decision|revert|chore`。

---

## 4. 十条最容易违反的硬约束

1. 真实数据、真实标识、凭证密钥**绝不提交**；合成数据也不提交，只提交生成器 + 配置 + 种子。
2. 真实数据算出的分箱边界 / IV 表 / 分布统计 / 交集规模属 **L-受限**，审查后才可提交，结论写进 changelog 的 `sensitive_review`。
3. **一次提交只能跨一个步骤**，禁止合并两步。
4. 未写步骤声明不得开始执行；自评每一项必须能指到具体文件与行号。
5. 不得引入 `registry/cross_border_assets.yaml` 之外的跨境资产。
6. 发现"说的和做的不一致"时**必须改实现**，禁止改文档对齐。
7. 失败分支卡、已归档 DR **永不删除**；失败与负面结果必须记录。
8. 不在测试集上做分支选择；不事后修改指标定义。
9. 单种子、无置信区间的结果不得下结论（种子 ≥5，主指标 100% 带 CI）。
10. 知识只留在会话上下文而不写进仓库 = 没做。

---

## 5. 与历史项目目录的关系

本仓库是 2026-08 起的**新阶段唯一工作仓库**。历史材料留在维护者本机 `/Applications/总文件夹/汇丰/` 下（`6月前准备工作/`、`汇丰项目6-7月/`、`docs/`、`飞书群文件/`），**只读参考，不迁入本仓库**——其中包含真实数据集与从真实/半合成数据算出的产物，按 L-禁止 / L-受限处理。

历史事实与决策脉络见本机 `6月前准备工作/项目实时日志.md`（项目唯一总日志，历史动作仍需登记到该文件）。

**口径提示**：历史阶段的业务命题是"跨境新客户识别（NTB）"，本阶段 v2 框架的业务命题已改为"**共同客户联合营销**"（交集即目标人群）。引用历史结论前先确认命题是否仍适用。
