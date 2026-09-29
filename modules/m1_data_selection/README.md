# M1 · 数据资产盘点与数据集选型

> 状态：🟡 **隔离数据准备已重跑；本地测试与回归复核通过；完整训练未放行** ｜ 步数预算 **9** ｜ 已用 **4**（新增前置 S1.P1 / S1.P2 / S1.P3 / S1.P4）｜ 认领人：无
> 最后更新：2026-09-28 ｜ 规范位置：`docs/00-framework/` §M1 · `docs/01-loops/` LOOP M1
> 仓库：https://github.com/conceal008/HSBC-VFL-personal-project

## 目标

在不存在「真实两机构纵向切分 + 跨法域 + 营销响应标签 + 随机对照」四条件齐备的公开数据集这一前提下，确立「真实数据作外部效度锚点 + 合成数据作机制实验台」的双轨方案。

## 上游输入

M0：`problem_statement.md`（转化定义决定标签口径）

## 输出契约（本模块必须产出）

`dataset_selection_report.md` · `split_protocol/` · `split_fidelity_curve.png+csv` · `data_cards/*.yaml` · `DR-M1-*.yaml`

**运行入口**：本模块凡产出处理或实验结果，一律以 `notebooks/<步骤号>_<简述>.ipynb` 为入口并**带输出提交**，使人打开仓库即可直接看到结果（《维护约束 v2》9.1）。逻辑放 `components/`，参数从 `configs/` 读；真实数据派生输出必须清除后再提交。

## 量化放行判据（不达标不放行）

- 评估数据集 ≥4，每集 6/6 维度评估，每集「不能回答」条目 ≥3
- 四种切分协议 S1–S4 全部实现
- 增益衰减数据点 = 4 切分 × ≥5 种子，含 CI
- S1/S4 增益比值已计算并报告（模拟纵向切分高估效应的量化证据）
- 数据卡 6 字段完整度 100%

## 当前结论

S1.P4 修复第二次 Linux CI 的两项门禁回归缺陷。整改细节、失败历史与本机全门禁结果见 [S1.P4 报告](report/S1.P4_GitHub_CI回归整改.md) 和带输出 [Notebook](notebooks/S1.P4_ci_regression_followup.ipynb)；修复后 GitHub 全工作流状态以 PR checks 为准。


S1.P3 将本地工程门禁及首次 PR Linux 检查汇总到 [测试报告](report/S1.P3_工程测试报告.md) 和带输出的 [Notebook](notebooks/S1.P3_engineering_verification.ipynb)。首次远端 Q1 失败因 Ruff 未固定，已锁定 0.12.0；远端复核状态见 GitHub PR checks。报告不含数据处理统计或私有实验日志。


S1.P2 根据新增要求重新审计并从原件重跑 UCI/Hillstrom：各方在默认拒绝的操作系统沙箱内执行自己的 Notebook；中心仅收白名单回执。详见 [模块与隔离要求](联邦流程模块与隔离要求.md) 和 [旧流程审计](集中式准备_隔离审计.md)。旧 S1.P1 仅保留集中式调试历史，不作为严格隔离通过证据。物理隔离、真实 PSI 与安全联合训练尚未完成。

历史记录（已由 S1.P2 取代隔离验收）：

S1.P1 已建立 UCI/Hillstrom 的 Notebook 数据准备框架，输入只读，新数据与完整运行输出分别存放在仓库外。两份入口已从头执行，并验证相同配置重复处理生成相同数据文件。详见 [使用说明](数据准备框架_使用说明.md)。

此为新增工程前置步骤，原 S1.1–S1.9 尚未完成；未产生模型效果结论。

## 未决问题

- 已按 DR-MX-001 纳入增量转化评估；Hillstrom 输入已准备，增量模型与评估尚未运行。
- 候选评估矩阵、正式数据卡与 S1–S4 切分比较待完成。
- UCI 缺真实稳定客户 ID；构造的记录键不能验证客户身份匹配。

## 升级条件

全部候选集在语义契合维度评分均 ≤1（3 分制）→ 上报，建议合成数据升为主力。

## 下一步

**S1.1** —— 详见 `docs/01-loops/2_模块Loop执行规范_v2_步进量化版.md` 中 LOOP M1 的步骤分解表。
执行前必须先写步骤声明（will_produce / will_not_produce / success_criteria / risk），
并在 `registry/module_status.yaml` 认领本模块。

---

*本 README 必须保持当前——它是下一个 Agent 接手时最先读的文件。每步提交时同步更新「状态 / 已用步数 / 当前结论 / 未决问题 / 下一步」五处。*
