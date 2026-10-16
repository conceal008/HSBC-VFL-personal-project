# M9 · 合规证据链与交付物

> 状态：🟨 **S9.H2迁移漏洞修复；科研实验暂停** ｜ 步数预算 **10** ｜ 已用 **4** ｜ 认领人：无
> 最后更新：2026-10-16 ｜ 规范位置：`docs/00-framework/` §M9 · `docs/01-loops/` LOOP M9
> 仓库：https://github.com/conceal008/HSBC-VFL-personal-project

## 目标

验收人不是技术方，而是法务与合规。第一步必须是一致性核验：从 M0 跨境对象清单出发，逐条追溯其在 M3–M8 的实际实现。

## 上游输入

M0–M8 全部产出

## 输出契约（本模块必须产出）

`DPIA.md` · `cross_border_filing_package/` · `consent_management_spec.md` · `data_cards/` · `model_card.md` · `cross_border_flow_registry.csv` · `audit_log_spec.md` · `residual_risk_signoff.yaml` · `DR-M9-*.yaml`

**运行入口**：本模块凡产出处理或实验结果，一律以 `notebooks/<步骤号>_<简述>.ipynb` 为入口并**带输出提交**，使人打开仓库即可直接看到结果（《维护约束 v2》9.1）。逻辑放 `components/`，参数从 `configs/` 读；真实数据派生输出必须清除后再提交。

## 量化放行判据（不达标不放行）

- 交付清单 10/10 完成
- **一致性核验通过率 100%**
- DPIA 风险溯源率 100%（每条对应 M7 具体实验）
- 模型卡分群数 ≥3；跨境流动清单 4/4 字段（路径·频次·量级·依据）
- 残余风险接受方 100% 已指定；**法务追问次数 = 0**

## 当前结论

此句为初始化历史，已被S9.1、S9.R1和本次S9.H1取代。当前统一入口为[START_HERE](../../docs/03-handoff/START_HERE.md)；用户暂停全部实验，未验收D2草稿不计完成。

## 未决问题

- （暂无，随工作展开补充）

## 升级条件

一致性核验发现不符 → 不得通过修改文档对齐，必须回到对应模块修改实现。

## 下一步

当前：保持科研实验暂停，维护[交接与修复报告](../../docs/04-experiment-records/迁移漏洞复核与修复报告.md)。

> 以下S9.1是初始化步骤提示，现已完成，保留历史：

**S9.1** —— 详见 `docs/01-loops/2_模块Loop执行规范_v2_步进量化版.md` 中 LOOP M9 的步骤分解表。
执行前必须先写步骤声明（will_produce / will_not_produce / success_criteria / risk），
并在 `registry/module_status.yaml` 认领本模块。

---

*本 README 必须保持当前——它是下一个 Agent 接手时最先读的文件。每步提交时同步更新「状态 / 已用步数 / 当前结论 / 未决问题 / 下一步」五处。*

## S9.1 已完成（2026-09-04）

### 放行判据核对

| 判据 | 结果 |
|---|---|
| 交付清单 10/10 | ✅ **10/10**（机器核验） |
| 一致性核验通过率 100% | ✅ **13/13 = 100%**（机器核验） |
| DPIA 风险溯源率 100% | ✅ **7/7 = 100%**，每条指向具体实验产出 |
| 模型卡分群数 ≥3 | ✅ 4 群 |
| 跨境流动清单 4/4 字段 | ✅ 方向 · 路径 · 频次 · 量级 · 依据（5 项） |

### 为什么先做机器核验再写文档

M9 的验收人是法务与合规，不是技术方。写成「已核验」四个字，谁也不知道核了什么。
所以先把「结论 → 证据文件」的映射做成可执行检查，再写文档。

`ci/check_evidence_chain.py` 已接入 `run_all_gates.sh`，每次提交前执行。
它防的是一类很安静的事故：**有人删掉或重命名了一个结果文件，
引用它的结论就此失去依据，而七道门禁照样全绿。**

写本模块的过程中它就抓到过一次——证据映射引用了不存在的 `funnel_scenarios.csv`，
通过率掉到 92.3%。

### 交付物

| 文件 | 用途 |
|---|---|
| [`对外表述红线.md`](对外表述红线.md) | **9 条绝对不能说** · 5 条须带限定 · 3 条必须主动说 |
| [`DPIA.md`](DPIA.md) | 7 条风险，逐条溯源到 M7 实验 |
| [`model_card.md`](model_card.md) | 五级阶梯表现、失效条件、隐私属性 |
| [`results/cross_border_flow_registry.csv`](results/cross_border_flow_registry.csv) | 8 类跨境流动 |
| [`audit_log_spec.md`](audit_log_spec.md) | 九字段规格，含 `degraded` |
| [`consent_management_spec.md`](consent_management_spec.md) | 撤回策略三选一，选定 S2 |
| [`data_cards/`](data_cards/) | 合成数据卡 |
| [`residual_risk_signoff.yaml`](residual_risk_signoff.yaml) | **未签署，且无人可签** |

> **降级说明（W-001）**：本项目不设合规角色。`residual_risk_signoff.yaml` 的
> 签署位全部空缺，**保留该文件而不删除，是为了让「谁该签而没签」显式可见**——
> 删掉它会让缺口消失于无形。

当前具体结果：52份评价摘要、124份执行状态、35张既有表索引，见[完整阶段归档](results/current_phase_v1/README.md)和[带输出Notebook](notebooks/S9.R1_result_archive.ipynb)。按DR-GOV-012公开审查后研究汇总，不读取或提交行数据/预测/身份/份额；旧接受失效和未执行状态明确保留。独立新目录重建55文件字节一致，不生成新CI或结论。

## S9.H1 当前交接

用户要求暂停所有实验；停机与私有检查点完成。11份日志/报告正文迁移，13份D2草稿字节保留、明确缺陷和未执行状态；原results不动。见[当前进度](../../docs/03-handoff/CURRENT_STATUS.md)、[实验台账](../../docs/04-experiment-records/实验总台账.md)及[迁移清单](../../docs/04-experiment-records/migration_manifest.json)。此步为文档/资产整理，不新增研究指标，不追认旧法务/风险签署或D2通过。

## S9.H2 当前修复

旧证据行号兼容、未来授权恢复检查、状态元数据丢失、VM恢复依赖、私有入口权限和历史状态误导已分别修复，见[修复交接](../../docs/03-handoff/HO-20261016-Codex-M9-migration-repair.md)。没有恢复D2或其他科研实验；规范正文、13草稿与冻结结果保持不变。

S9.H2发布前工程验收：646测试、核心10768/11548=93.2%、53/53回归及M9四项100%；验收后只补状态回执，不改科研资产，实验保持暂停。main/远端CI另据实际发布回执核验。
