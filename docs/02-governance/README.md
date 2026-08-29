# docs/02-governance · 治理规范

本目录存放《项目维护约束 v2（步进量化版）》——定义**怎么协作与留痕**。三份规范同时生效，冲突时以本文件目录下规范的**禁止性条款**优先。

| 文件 | 说明 |
|---|---|
| `3_项目维护约束_v2_步进量化版.md` | 与来源 `汇丰/8-9月/` 逐字一致。已按 DR-GOV-002 修订 1.1 可见性条款（v2.1），两份文件同步修改并经 diff 校验；其余条款一字未动 |

## 本仓库的落地补注（不修改规范正文）

规范正文按《维护约束 v2》第 10 部分与文末声明，修改需 DR + 合规 approve。因此以下落地信息写在本 README 而不是改正文：

| 规范条款 | 规范要求 | 本仓库实际 | 依据 |
|---|---|---|---|
| 1.1 仓库地址 | —（规范未指定） | **https://github.com/conceal008/HSBC-VFL-personal-project** | `registry/decision_records/DR-GOV-001.yaml` · repo_location |
| 1.1 可见性 | 组织/团队仓库 Private；**个人仓库可 Public**（四项前提缺一不可） | **Public** —— 个人账户仓库，四项前提当前全部成立，**合规** | DR-GOV-002（修订条款并作废 DR-GOV-001 · repo_visibility） |
| 1.1 分支保护 / 必需检查 / Secret push protection | main 禁止直推、CI 通过方可合并、开启 Secret 扫描 | **尚未开启**，需仓库所有者在 GitHub 设置中配置 | `registry/module_status.yaml` · governance.open_items |
| 附录 B CI 脚本 | 六个门禁脚本 | 仅 `check_secrets.sh`（门禁 1）已实现，其余人工执行 | DR-GOV-001 · init_scope · `ci/README.md` |

上表任一行状态改变时，必须同步更新本 README 与 `registry/module_status.yaml`；**不得反过来改规范正文使其"看起来一致"**（禁止事项第 16 条）。
