# 会话交接记录 · M5 第三阶段跨数据集验证与汇报归档

> 交接日期：2026-11-20 ｜ 交接方：Antigravity ｜ 分支：`m5/stage3-cross-dataset`（基于 `origin/main` `75c3bb8`）
> 关联变更：`CL-20261120-M5-001` ｜ 关联决策：[DR-M5-018](../../registry/decision_records/DR-M5-018.yaml)

## 1. 本次归档内容

1. **汇总结果（128 条冻结指标）**：`modules/m5_modeling/results/stage3_cross_dataset_v1/`
   - `stage3_master_summary.csv` / `stage3_master_summary.json` 及 `exp1`–`exp6` 共 9 份代理与真实 SecretFlow 1.14.0b0 SGB 结果 JSON。
   - 不含任何原始数据、逐行打分、模型权重或 `_cache/` 派生缓存（符合 DR-GOV-012 公开研究汇总规范）。
2. **完整报告（2 份）**：`docs/04-experiment-records/reports/`
   - [第三阶段成果汇报_理论机制与跨数据集实证报告_v1.0.md](../04-experiment-records/reports/第三阶段成果汇报_理论机制与跨数据集实证报告_v1.0.md)（面向项目汇报：六大理论推导 + 128 条实证 + 四道准入门禁 + 10 页 PPT 提纲）。
   - [第三阶段_创新方法跨数据集验证报告_v1.0.md](../04-experiment-records/reports/第三阶段_创新方法跨数据集验证报告_v1.0.md)（按 `Exp1`–`Exp6` 逐项归档）。
3. **复现脚本快照**：`docs/04-experiment-records/stage3_cross_dataset_snapshot/`
   - 原样保留 13 个 `.py` 脚本、`run_all.sh` 与冻结配置 `stage3_config.json`（SHA-256 见快照 README），按 DR-M5-018 如实声明未按门禁 5/6 重构。

## 2. 核心实证结论速览

- **无条件必选（T1）**：`I1(OU)+M3(主动方明文直方图)` 在 3 个新数据集上提速 **4.74×–5.93×**（4 方 M3 提速 **1.662×**），全部 **`0.00e+00` 逐位一致**；`I2+I3`（`+0.23pp`–`+1.46pp`）与 `M6`（`+0.05pp`–`+0.37pp`）在 4/4 数据集真实 SGB 全正向。
- **主力涨点（T2）**：`X2` Home Credit 全新支付履约 4 表三方代理 `+2.18pp` / 真 SGB **`+1.79pp`**（为静态征信 `bureau` 的 5.6–6.6 倍），70% 覆盖仍 `+1.42pp`；`X1+` Santander 7 月因果动量代理 `+3.87pp` / 真 SGB **`+3.91pp`**（PR-AUC 翻倍），IEEE-CIS W3 紧致 60 列真 SGB **`+1.01pp`** 且提速 23%；`X3` Santander Top20% 净利 **`+3,744.5 EUR`**（`p_holm=0.018`），Hillstrom 随机 A/B 实证捕获价值 ≠ 因果增量。
- **条件启用（T3）**：`M1x` 在同分布 20% 低重合真 SGB 获 **`+1.48pp`**（`p=0.004`，+0.3s 零成本），但在 IEEE-CIS 跨渠道 3.75× 偏移池引发负迁移（`-0.54pp`），须用 `M1x_logit_fusion` 校准防护；`I6` 在主动方强（UCI $\varepsilon=2,4$）或 Home Credit $\varepsilon=4$（**`+0.76pp`**）正向，但在被动方主导且 $\varepsilon\le 2$ 时劣化（`-1.77pp` 至 `-6.66pp`）。
- **科学证伪**：纯主动方表格型 `C2` Lookalike 跨 5 数据集 0/5 显著正向（Santander 显著退化 `-0.61pp`）。

## 3. 未决事项与下一步建议

1. 如需将 `m5/stage3-cross-dataset` 与 `m5/method-screen-3`（PR #21）合并至 `main`，建议在 GitHub 上分别审阅合并（两分支均基于 `origin/main` `75c3bb8` 且文件名无冲突，DR 编号已顺延至 `DR-M5-018` 避免碰撞）。
2. 若后续决定将 `X1+`、`M1x_logit_fusion` 或 4 方 `M3` 正式合入 `modules/m5_modeling/components/`，须另立独立开发步骤按门禁 5/6 重构并补齐单测与组件声明。
3. 本机 `/Applications/总文件夹/汇丰/第三阶段/results/_cache/` 含公开数据集派生特征缓存，仅留本机，严禁入库。
