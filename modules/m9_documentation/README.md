# M9 · 合规证据链与交付物

> 状态：🟨 **文档与证据链级完成（S9.1）** ｜ 步数预算 **10** ｜ 已用 **1** ｜ 认领人：无
> 最后更新：2026-09-04 ｜ 规范位置：`docs/00-framework/` §M9 · `docs/01-loops/` LOOP M9
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
