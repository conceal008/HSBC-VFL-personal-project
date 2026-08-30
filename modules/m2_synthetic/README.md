# M2 · 合成数据生成与仿真台

> 状态：⬜ **未开始** ｜ 步数预算 **9** ｜ 已用 **0** ｜ 认领人：无
> 最后更新：2026-08-29 ｜ 规范位置：`docs/00-framework/` §M2 · `docs/01-loops/` LOOP M2
> 仓库：https://github.com/conceal008/HSBC-VFL-personal-project

## 目标

构建参数可调、ground truth 可知的双方联合数据生成器，使「在什么条件下联邦有价值」成为可实验命题。主力用结构因果生成（SCM），SDV 类方法作保真度对照。

## 上游输入

M1：数据集选型结论与切分协议

## 输出契约（本模块必须产出）

`generator/` · `scenario_library.yaml` · `fidelity_report.md` · `correctness_validation.md` · `scope_disclaimer_template.md` · `DR-M2-*.yaml`

**运行入口**：本模块凡产出处理或实验结果，一律以 `notebooks/<步骤号>_<简述>.ipynb` 为入口并**带输出提交**，使人打开仓库即可直接看到结果（《维护约束 v2》9.1）。逻辑放 `components/`，参数从 `configs/` 读；真实数据派生输出必须清除后再提交。

## 量化放行判据（不达标不放行）

- 11 项可调参数覆盖 11/11
- 正确性验收：理论增益 vs 实测增益偏差 <5%，配置数 ≥3
- 保真度：KS ≤0.1 · 相关矩阵 Frobenius ≤0.15 · 下游排序 Spearman ≥0.8
- 标准场景 ≥6；同配置同种子输出哈希一致；无效改进清单 ≥1 条

## 当前结论

暂无。本模块尚未开始，`step_ledger.yaml` 中 `steps` 为空。

## 未决问题

- 双方联合生成器（两侧特征 + 共享潜变量 + 标签路径 + 标识符层匹配噪声统一定义）是文献空白，需自行设计

## 升级条件

正确性验收偏差 ≥5% 且 2 轮修正无法降低 → 禁止进入 M3。建立在错误实验台上的所有后续结论均无效，此条不可妥协。

## 下一步

**S2.1** —— 详见 `docs/01-loops/2_模块Loop执行规范_v2_步进量化版.md` 中 LOOP M2 的步骤分解表。
执行前必须先写步骤声明（will_produce / will_not_produce / success_criteria / risk），
并在 `registry/module_status.yaml` 认领本模块。

---

*本 README 必须保持当前——它是下一个 Agent 接手时最先读的文件。每步提交时同步更新「状态 / 已用步数 / 当前结论 / 未决问题 / 下一步」五处。*
