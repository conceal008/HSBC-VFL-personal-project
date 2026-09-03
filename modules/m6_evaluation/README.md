# M6 · 效果验证与评估体系

> 状态：🟩 **初版完成（S6.1）** ｜ 步数预算 **9** ｜ 已用 **1** ｜ 认领人：无
> 最后更新：2026-09-03 ｜ 规范位置：`docs/00-framework/` §M6 · `docs/01-loops/` LOOP M6
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

暂无。本模块尚未开始，`step_ledger.yaml` 中 `steps` 为空。

## 未决问题

- 业务最小可察差异 MDE 需在本模块冻结口径时定义（建议初值：Lift@K 相对提升 ≥3%，Uplift 相对提升 ≥5%），此前不得开始分支比较
- Top-K 的 K 取业务真实触达规模，需业务输入

## 升级条件

决策等价性检验显示 L1 与 L3 的 Top-K 重合度 ≥95% → 上报。这意味着 VFL 不产生业务差异，即使 AUC 有提升。

## 下一步

**S6.1** —— 详见 `docs/01-loops/2_模块Loop执行规范_v2_步进量化版.md` 中 LOOP M6 的步骤分解表。
执行前必须先写步骤声明（will_produce / will_not_produce / success_criteria / risk），
并在 `registry/module_status.yaml` 认领本模块。

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
