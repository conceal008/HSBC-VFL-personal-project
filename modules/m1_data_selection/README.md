# M1 · 数据资产盘点与数据集选型

> 状态：⬜ **未开始** ｜ 步数预算 **9** ｜ 已用 **0** ｜ 认领人：无
> 最后更新：2026-08-29 ｜ 规范位置：`docs/00-framework/` §M1 · `docs/01-loops/` LOOP M1
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

暂无。本模块尚未开始，`step_ledger.yaml` 中 `steps` 为空。

## 未决问题

- 是否纳入 uplift 评估？若纳入必须补 ≥1 个随机对照数据源（Criteo Uplift / Hillstrom）

## 升级条件

全部候选集在语义契合维度评分均 ≤1（3 分制）→ 上报，建议合成数据升为主力。

## 下一步

**S1.1** —— 详见 `docs/01-loops/2_模块Loop执行规范_v2_步进量化版.md` 中 LOOP M1 的步骤分解表。
执行前必须先写步骤声明（will_produce / will_not_produce / success_criteria / risk），
并在 `registry/module_status.yaml` 认领本模块。

---

*本 README 必须保持当前——它是下一个 Agent 接手时最先读的文件。每步提交时同步更新「状态 / 已用步数 / 当前结论 / 未决问题 / 下一步」五处。*
