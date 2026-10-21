# 外部实验记录：隐语 SGB 纵向联邦的优化与攻防（secretflow_sgb_innovations）

> 编写日期：2026-10-21 ｜ 版本：v1.0 ｜ 状态：外部记录导入（S9.E1），不恢复本仓库任何暂停实验
> 数据来源：团队仓库 `SCUT-Elite-Camp/federated_learning_new` 原版流水线之上的扩展实验（云端容器运行，2026-10-19 → 10-21）
> 口径基准：DR-GOV-012（§2.7 公开研究数据集汇总指标）；结果为**单 seed（42）+ 测试集客户级 bootstrap CI**

**适用边界（先读）**

1. **全部结论来自公开研究数据集代理**（Santander、HSBC mock、FedAds、Orange KDD Cup 2009、IEEE-CIS），只验证机制，不承诺真实业务效果。
2. **本项目不设合规角色**，涉法判断未经法务复核。
3. **无防护的协议是双向泄露的**：本记录在隐语 SGB 上复现了两类半诚实攻击（被动方借叶子划分推断标签、主动方借分裂划分还原被动方特征）。
4. **单 seed，不构成本仓库结论**（AGENTS.md 硬约束 9 要求 ≥5 seed）：这里的 CI 只反映测试集抽样不确定性，不含训练随机性。本记录作为外部参考与后续正式实验的候选清单。
5. 本记录与 S5.X3 等已冻结结果相互独立，不修改任何 `modules/*/results`；科研实验暂停状态不变。

## 内容

| 路径 | 内容 |
|---|---|
| [INNOVATIONS.md](INNOVATIONS.md) | 7 个创新点的原理、实现路径、结果与局限，以及试过但不采用的方法 |
| `src/ext/` | 扩展代码（31 个脚本）；需放入 `federated_learning_new` 检出中运行，依赖其 `src/cross_sell/train_local.py` |
| `scripts/` | 实际运行用的串行队列脚本（每项幂等，可中断续跑；`S` 指工作根目录） |
| `tools/` | IEEE-CIS 交集抽取脚本、汇总表生成脚本 |
| `results/` | 汇总指标 JSON：调参三臂与联邦方案的测试 AUC 及配对 CI、攻击与防护汇总、切分扫描、每次联邦训练的配置与耗时 |

## 创新点一览

| # | 创新点 | 关键文件 / 参数 | 主要结果（单 seed） |
|---|---|---|---|
| 1 | OU 替换 Paillier | `train_federated_sgb_opt.py --he-schema ou` | 快 5.85 倍，预测逐位相同 |
| 2 | 细分桶 + 三臂同等调参 | `make_val_tag.py` → `tune_proxy.py` → `final_eval.py`；`--sketch-eps 0.03` | Santander 0.8991 → 0.9091，与集中式之差 CI 含 0 |
| 3 | 先验初始分数 | `--base-score prior` | FedAds +0.015；Santander 无变化 |
| 4 | 明文代理增益与切分预检（VertiBench 式） | `gain_proxy.py`、`vertibench_scan.py`、`vertibench_confirm.py` | 代理与 SGB 实测差 ≤ 0.003；增益随被动方重要性占比上升 |
| 5 | 攻击复现（A1 / A2） | `--dump-model` → `sgb_replay.py` → `attacks.py` | 推理期被用到的被动方离散列 100% 被确定还原；被动方推断标签 AUC ≈ 0.88 |
| 6 | 后验去噪标签差分隐私 | `--label-rr-eps ε --label-dp rr_soft` | 同 ε 比硬标签 +0.009 / +0.026 / +0.071（ε = 4 / 2 / 1） |
| 7 | SGB 训练 + SPU 安全推理 | `secure_infer.py` | AUC 不变，消除推理期特征还原 |

标签差分隐私只覆盖标签本身（随机响应每个标签只随机化一次）；叶子划分仍会反映模型学到的排序，不能据此说「标签受保护」。

## 公开审查（§2.7）

- 不含：原始 / 清洗行数据、逐行预测与标签、模型导出（树结构与阈值）、学习到的预处理参数、凭证。
- 攻击结果 JSON 已删除逐特征明细（含逐产品持有人数等细分小群体统计），只保留汇总指标。
- IEEE-CIS 的 PSI 交集行数不在公开仓库披露；其余数据集的样本量为公开数据集或抽样规模。
- 汇总 xlsx 未入库（二进制，内容与 JSON 重复，不便逐项审查）。

## 复现

```bash
# 在 federated_learning_new 检出根目录，把本目录 src/ext 复制为 src/ext 后：
python src/ext/make_val_tag.py --tag business --val-months 3
python src/ext/tune_proxy.py --tag business_val
python src/ext/train_federated_sgb_opt.py --tag business --he-schema ou --num-boost-round 134 --max-depth 6 \
       --reg-lambda 10 --sketch-eps 0.03 --run-id business_sgb_ou_opt_m42 --dump-model data/processed/attack/model_T2_opt.json
python src/ext/final_eval.py --tag business --fed opt=data/processed/predictions/business_sgb_ou_opt_m42_predictions.csv
(cd src/ext && python attacks.py --model ../../data/processed/attack/model_T2_opt.json --tag business \
       --host-sees-first-tree 0 --known-frac 0.01 0.05 --name T2_opt)
python src/ext/secure_infer.py --model data/processed/attack/model_T2_opt.json --tag business --batch-trees 3
python src/ext/vertibench_scan.py && python src/ext/vertibench_summary.py
```

环境：SGB 相关脚本用 SecretFlow 1.13.0b0（Python 3.10）；明文脚本用 xgboost、scikit-learn、pandas、scipy。
耗时在 2 核容器、隐语 debug 单进程模式下测得，没有网络开销，只用于相对比较。
