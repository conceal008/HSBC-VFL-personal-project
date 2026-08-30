# M7 · 安全攻防与隐私评测

> 状态：⬜ **未开始** ｜ 步数预算 **15** ｜ 已用 **0** ｜ 认领人：无
> 最后更新：2026-08-29 ｜ 规范位置：`docs/00-framework/` §M7 · `docs/01-loops/` LOOP M7
> 仓库：https://github.com/conceal008/HSBC-VFL-personal-project

## 目标

先形式化威胁模型（攻击者身份·能力·目标资产·攻击面），再做 ≥6 种攻击与 ≥4 条防御。第 5 条（PSI 重复求交差分）与第 6 条（分数/名单反推）是本场景特有、最可能形成差异化贡献的方向。

## 上游输入

M5：可攻击的模型产物 · M6：评估口径（防御代价用同一口径衡量）

## 输出契约（本模块必须产出）

`threat_model.md` · `attack_experiments/` · `attack_results.csv` · `privacy_utility_surface.png` · `cumulative_budget_accounting.md` · `residual_risk_register.yaml` · `DR-M7-*.yaml`

**运行入口**：本模块凡产出处理或实验结果，一律以 `notebooks/<步骤号>_<简述>.ipynb` 为入口并**带输出提交**，使人打开仓库即可直接看到结果（《维护约束 v2》9.1）。逻辑放 `components/`，参数从 `configs/` 读；真实数据派生输出必须清除后再提交。

## 量化放行判据（不达标不放行）

- 威胁模型 4/4 维；攻击实现 ≥6（①–⑥必做），每攻击 100% 有法律映射
- 防御路线 ≥4；ε 扫描点 ≥5；三维曲面（成功率 × ε × 业务指标）已产出
- **累积泄露预算核算 ≥12 轮**（生产按月重跑，一年 12 轮后 ε 还剩多少）
- M6–M7 迭代收敛（连续 2 轮配置不变）；残余风险接受方 100% 已指定

## 当前结论

暂无。本模块尚未开始，`step_ledger.yaml` 中 `steps` 为空。

## 未决问题

- 可接受攻击成功率阈值需在 S7.1 定义，须与 M0 的法律定性挂钩

## 升级条件

在满足 M6 业务效果要求的防御强度下，任一攻击成功率仍高于可接受阈值（阈值在 S7.1 定义）→ 上报 M0 重审信息流设计。

## 下一步

**S7.1** —— 详见 `docs/01-loops/2_模块Loop执行规范_v2_步进量化版.md` 中 LOOP M7 的步骤分解表。
执行前必须先写步骤声明（will_produce / will_not_produce / success_criteria / risk），
并在 `registry/module_status.yaml` 认领本模块。

---

*本 README 必须保持当前——它是下一个 Agent 接手时最先读的文件。每步提交时同步更新「状态 / 已用步数 / 当前结论 / 未决问题 / 下一步」五处。*
