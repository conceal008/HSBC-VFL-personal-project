# 第三阶段跨数据集验证与真实 SecretFlow 1.14 SGB 复核结果（冻结）

> 编写日期：2026-11-20 ｜ 版本：v1.0 ｜ 状态：冻结（对应 [DR-M5-018](../../../../registry/decision_records/DR-M5-018.yaml)）
> 配套报告：[第三阶段成果汇报_理论机制与跨数据集实证报告_v1.0.md](../../../../docs/04-experiment-records/reports/第三阶段成果汇报_理论机制与跨数据集实证报告_v1.0.md) · [第三阶段_创新方法跨数据集验证报告_v1.0.md](../../../../docs/04-experiment-records/reports/第三阶段_创新方法跨数据集验证报告_v1.0.md)
> 复现脚本快照：[docs/04-experiment-records/stage3_cross_dataset_snapshot/](../../../../docs/04-experiment-records/stage3_cross_dataset_snapshot/README.md)

## 文件清单（共 11 份汇总结果，不含任何原始行数据、模型权重或 `_cache/` 派生缓存）

- `stage3_master_summary.csv` / `stage3_master_summary.json`：128 条跨数据集验证指标统一总表（每行带来源 JSON、95% CI 与 Holm 校正 p 值）。
- `exp1_x2_multiparty.json` / `exp1_x2_sgb_real.json`：X2 多方/高频第三方扩展（Home Credit 4 方 + 支付履约 4 表、Santander 模拟三方、UCI Bank 去 `duration` 模拟三方，6-seed 代理 + 真实 SecretFlow 1.14 SGB）。
- `exp2_x1plus_temporal_graph.json` / `exp2_x1plus_sgb_real.json`：X1/X1+ 本地因果图与时序速度（Home Credit、Santander 7 月队列、IEEE-CIS 滚动三窗，6-seed 代理 + 真实 SGB）。
- `exp3_m1x_unaligned.json` / `exp3_m1x_sgb_real.json`：M1x 非对齐样本先验融合（Home Credit 20%/50% 构造低重合池 vs IEEE-CIS 44.6 万真实非对齐交易，6-seed 代理 + 真实 SGB）。
- `exp4_x3_expected_value.json`：X3 期望净价值排序（Santander、COIL2000、Hillstrom 真实 `spend` + 随机 A/B 因果检验）。
- `exp5_c2_seed_lookalike.json`：C2 种子客群 Lookalike 相似度特征（5 个异构数据集首次实证，0/5 显著正向）。
- `exp6_core_sgb_real.json`：底层协议/集成/隐私核心优化（I1+M3 加密提速、I2+I3+M6 跨 4 数据集 6-seed、I6 后验去噪软标签 DP 跨 3 数据集 × 3 预算）全程真实 SecretFlow 1.14 SGB 实测。

## 口径与适用边界

1. 全部结果基于公开研究数据集的**模拟纵向切分**与构造低重合条件，营销单价与触达成本均已标注【假设】，仅作方法机制与边界验证，**不代表汇丰真实业务客群或投产收益**。
2. 真实 SecretFlow 1.14.0b0 SGB 运行于单机 `debug_mode=True`（真实 2048-bit PHEU 同态加密运算，不含跨机房网络延迟）；`Exp1/Exp2/Exp3` 真 SGB 为单次拟合（`seed=11`），其 95% CI 仅含测试集配对抽样不确定性。
3. 本目录所有 JSON/CSV 均已按 DR-GOV-012 审查：仅含公开数据集汇总指标，不含原始/清洗行数据、逐行预测分数、样本标识或其散列、模型权重或凭证。
