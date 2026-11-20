# 第三阶段实验复现脚本与配置快照（未重构存档）

> 编写日期：2026-11-20 ｜ 版本：v1.0 ｜ 决策依据：[DR-M5-018](../../../registry/decision_records/DR-M5-018.yaml)
> 冻结结果目录：[modules/m5_modeling/results/stage3_cross_dataset_v1/](../../../modules/m5_modeling/results/stage3_cross_dataset_v1/README.md)

## 重要声明（按 DR-M5-018 如实披露）

本目录保存的是第三阶段跨数据集验证（`Exp1`–`Exp6`）实际运行的**原样代码与配置快照**（与产出冻结结果 JSON/CSV 的代码逐字节一致），经所有者于 `2026-11-20` 确认归档于 `docs/04-experiment-records/` 下，**尚未按本仓库门禁 5（ruff/mypy/pytest ≥ 70% 行覆盖/组件声明）与门禁 6（R1 零硬编码魔数、R2 种子全从 YAML 配置读入）进行工程重构**。

- 存放于本目录的目的：确保仓库读者与后续 Agent 可 100% 追溯第三阶段全部 128 条汇总指标的计算细节与特征构造逻辑，同时不绕过、不弱化 `modules/` 下的既有门禁。
- 后续计划：若需将其中任意组件（如 `M1x` 校准融合、`X1+` 因果时序特征、4 方 `M3` 明文主动方直方图）纳入正式 `modules/m5_modeling/components/`，须另立步骤按门禁 5/6 重构、补齐单元测试与组件声明，并核对重构前后输出逐位一致。

## 目录内容

- `configs/stage3_config.json`：冻结实验配置（SHA-256: `feace06f06bca0830cb8394245b35a18a2e17c89127330def8bf544876e9b552`；其中的本地数据集绝对路径为维护者本机只读路径，公开数据集需自行下载后替换路径）。
- `src/common.py`：等调参代理（`HistGradientBoostingClassifier`, `max_bins=33`）、两层配对 bootstrap 与 Holm-Bonferroni 校正工具。
- `src/sgb_utils.py`：真实 SecretFlow 1.14.0b0 SGB（OU/Paillier-2048、2/3/4 方、运行时 M3 主动方明文直方图上下文管理器）。
- `src/prep_cache.py`：Home Credit 7 表聚合（含 4 张新支付履约表与因果历史速度）及 Santander 7 月因果队列缓存构建（缓存仅保留在本地，不入库）。
- `src/exp1_x2_multiparty.py` / `src/exp1_sgb_confirm.py`：Exp1（X2 多方/四方）。
- `src/exp2_x1plus_temporal_graph.py` / `src/exp2_sgb_confirm.py`：Exp2（X1/X1+ 因果图与时序速度 + IEEE-CIS 滚动三窗）。
- `src/exp3_m1x_unaligned.py` / `src/exp3_sgb_confirm.py`：Exp3（M1x 非对齐样本先验融合）。
- `src/exp4_x3_expected_value.py`：Exp4（X3 营销期望净价值排序 + Hillstrom 随机 A/B 检验）。
- `src/exp5_c2_seed_lookalike.py`：Exp5（C2 种子客群 Lookalike 跨 5 数据集实证）。
- `src/exp6_core_sgb_real.py`：Exp6（I1+M3 / I2+I3 / M6 / I6 真实 SGB 全量复核）。
- `src/aggregate_results.py` / `src/run_all.sh`：汇总 128 条指标总表与一键复现脚本。
