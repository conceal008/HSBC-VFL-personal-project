# M6 · 效果验证与评估体系

> 状态：🟨 **进行中（S6.1、S6.P1、S6.P2、S6.P3）** ｜ 步数预算 **9** ｜ 已用 **4** ｜ 认领人：Claude (claude-opus-5-5, Claude Code)
> 最后更新：2026-10-25 ｜ 规范位置：`docs/00-framework/` §M6 · `docs/01-loops/` LOOP M6
> 仓库：https://github.com/conceal008/HSBC-VFL-personal-project

## 目标

四层指标逐层递进（判别力 → 业务排序 → 增量 → 收益），三种划分全做（随机 / 时间外推 OOT / 跨群体）。营销的正确目标是增量转化，不是响应率。

## 上游输入

M5：基线阶梯全部结果

## 输出契约（本模块必须产出）

`evaluation_protocol.md`（冻结版）· `four_layer_results.csv` · `split_comparison.md` · `decision_equivalence.md` · `value_of_information_curve.png+csv` · `DR-M6-*.yaml`

**运行入口**：本模块凡产出处理或实验结果，一律以 `notebooks/<步骤号>_<简述>.ipynb` 为入口并**带输出提交**，使人打开仓库即可直接看到结果（《维护约束 v2》9.1）。逻辑放 `components/`，参数从 `configs/` 读；真实数据派生输出必须清除后再提交。

## 量化放行判据（不达标不放行）

- 指标 4/4 层；划分 3/3 种；种子 ≥5；主指标 CI 覆盖 100%
- AUC 报告位数 ≤3 位小数
- D-1 vs D-2 差距已量化；Top-K 重合度已给出数值
- 条件价值曲线 ≥4 横轴变量 × ≥5 点；盈亏平衡点已给坐标 + 现实可达性判断
- 「响应率 = 增量」表述 0 处

## 当前结论

- **评估口径 v1 已冻结**：[`evaluation_protocol.md`](evaluation_protocol.md)（S6.P1，2026-10-25）。新预注册的比较一律按它写；修改须另写 DR。
- S6.1 的四条结论见文末「本模块已完成」。
- S6.P1 校准的四条结论见文末「S6.P1」一节。
- **S6.P2：在需求方给出规模之前，最该先问的是年联系人数**，见文末「S6.P2」一节。

## 未决问题

- ~~业务最小可察差异 MDE 需在本模块冻结口径时定义~~ 已由 DR-MX-001 · D-4（相对增益 20%）与 evaluation_protocol v1 第二、四节定义
- D-1 原文要求在 M2 零效应合成设定上复核第一类错误率；S6.P1 在打分层面完成，M2 上的复核属于实验，待恢复后补
- Qini、AUUC 等增量指标的区间方法尚未校准（evaluation_protocol v1 第八节第 4 条）
- Top-K 的 K 取业务真实触达规模，需业务输入

## 升级条件

决策等价性检验显示 L1 与 L3 的 Top-K 重合度 ≥95% → 上报。这意味着 VFL 不产生业务差异，即使 AUC 有提升。

## 下一步

S6.2–S6.9 中需要实验结果的步骤，等所有者授权恢复实验（任务清单 T5）。G3 五臂对比已预注册（S6.P3，状态 proposed），授权后按其第九节冻结再执行。
需求方给出规模、成本或合规答复后，按新参数重跑 S6.P2。

> 步骤编号说明：Loop 规范中 S6.1 是「冻结评估口径」，本仓库的 S6.1 实际做的是指标库与条件价值曲线；冻结口径由 S6.P1 补做。

---

*本 README 必须保持当前——它是下一个 Agent 接手时最先读的文件。每步提交时同步更新「状态 / 已用步数 / 当前结论 / 未决问题 / 下一步」五处。*

## 本模块已完成（2026-09-03，初版）

| 产出 | 位置 |
|---|---|
| 指标库（AUC/KS/Lift/Recall/Qini/AUUC/Uplift@K/TopK重合/bootstrap） | `components/metrics.py` |
| 条件价值曲线图 | `results/conditional_value_curves.png` |
| 运行入口（带输出） | `notebooks/S6.1_conditional_value.ipynb` |

**核心结论**：

1. **互补性 λ 是主导因子**——增益从 λ=0 的 0.025 到 λ=2.0 的 0.209，跨度八倍。
2. **AUC 接近 ≠ 决策一致**——L1 与 L3a 的 Top-10% 名单仅重合 46%。
3. **盈亏平衡**——单人增益与可触达人数方向相反，总价值在中高重叠区最大；
   只看 AUC 增益会得出「重叠率越低越好」的错误结论。
4. **响应率 ≠ 增量**——B 侧数据使 Top-10% 增量近乎翻倍（+91%），而 AUUC 只涨 29%。

## S6.P1 · 配对比较判据校准与评估口径冻结（2026-10-25）

| 产出 | 位置 |
|---|---|
| 冻结的评估口径 v1 | [`evaluation_protocol.md`](evaluation_protocol.md) |
| 配对比较组件（两层重抽样、可检出差异、四类结论、合成校准） | `components/paired_eval.py` |
| 运行入口（带输出） | [`notebooks/S6.P1_criterion_calibration.ipynb`](notebooks/S6.P1_criterion_calibration.ipynb) |
| 结果表 | `results/s6p1_type1_error.csv` · `s6p1_power_at_predicted_mde.csv` · `s6p1_positives_simulated.csv` |

**结论**（纯合成打分，不训练模型）：

1. **新判据第一类错误率在名义水平附近**：零效应下 0.038–0.045，DR-MX-002 · D-1 推翻条件未触发。
2. **旧判据功效只有约 0.1**；5 个 seed 的符号翻转检验功效恒为 0；6 个 seed 的符号翻转在零效应下拒绝率约 0.8，只能回答「这一份测试集上」的条件问题。
3. **D-2 推翻条件触发**（441 个正例时功效 0.880，超出上限 0.001）。按预定后果用模拟法重算：检出配对差 0.02 约需 693 个正例，而不是 S0.1P 发布的 441 个——S0.1P 默认的两模型估计相关 0.8 偏乐观，合成设定中约 0.69。
4. **可检出差异取两法较大者**：150 个正例时重抽样标准误在备择下偏低约一成，正例较多时闭式近似偏保守。

## S6.P2 · 盈亏平衡与信息价值（2026-10-25）

| 产出 | 位置 |
|---|---|
| 净值、盈亏平衡与 EVPI/EVPPI 组件 | `components/breakeven.py` |
| 参数分布（19 项，逐项注明来源，全部为显式假设） | `configs/s6_p2_breakeven.yaml` |
| 运行入口（带输出） | [`notebooks/S6.P2_breakeven_value_of_information.ipynb`](notebooks/S6.P2_breakeven_value_of_information.ipynb) |
| 结果表 | `results/s6p2_decision_summary.csv` · `s6p2_evppi.csv` · `s6p2_evppi_by_contact_tier.csv` · `s6p2_breakeven_grid.csv` |

**结论**（只是「先问哪个数」的顺序与门槛形状，不是收益判断）：

1. **先问规模**：年联系人数的部分完全信息期望价值（EVPPI）最高，约占 EVPI 的 37%；其后是 L3 命中提升、L1 捕获比例、因果占比。
2. **年联系不足 2 万人时，任何单个参数都不改变「不做」的决策**；5 万人以上时 L3 的期望净值转正，出境机制（是否为关键信息基础设施运营者）才进入前列。
3. **合规参数在全样本上 EVPPI 为 0**：与「先问是否为关键信息基础设施运营者」的问题单顺序相反，问题单应改为先问共同客户数与可联系人数。
4. **盈亏平衡门槛**：年联系 2 万人、单客 1.4 万港元时，L3 须带来约 1.9 个百分点的命中提升（个人信息出境标准合同）或约 2.7 个百分点（数据出境安全评估）。

## S6.P3 · G3 五臂对比预注册（2026-10-25，proposed）

| 产出 | 位置 |
|---|---|
| 预注册声明 | [`S6.P3_G3五臂对比_预注册声明.md`](S6.P3_G3五臂对比_预注册声明.md) |
| 机器可读配置 | `configs/s6_p3_g3_preregistration.yaml` |

- **唯一主对比**：Santander 上双向联邦 A 相对 L1（k=10）的配对 AUROC 差，关心尺度 0.02。
- 九个臂：L0、L1（k = 5 / 10 / 20）、B_T1、B_T2（B_T3 即 L1_k10）、A、A_secure（SS-XGB，只在 Santander）、阴性对照 NC。
- Santander 重新按客户抽样，开发评估集期望 ≥ 700 个正例（S6.P1 检出 0.02 约需 693 个）；封存确认集为最新月份 × 未用客户，性质是「同源未用客户」，不是新的独立人群。
- 停止规则与结果到决策的映射在看结果之前写定。**不授权恢复实验**；冻结在 T5 授权之后。
