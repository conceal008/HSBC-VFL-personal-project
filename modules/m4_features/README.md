# M4 · 特征工程与跨域特征治理

> 状态：⬜ **未开始** ｜ 步数预算 **10** ｜ 已用 **0** ｜ 认领人：无
> 最后更新：2026-08-29 ｜ 规范位置：`docs/00-framework/` §M4 · `docs/01-loops/` LOOP M4
> 仓库：https://github.com/conceal008/HSBC-VFL-personal-project

## 目标

在本地加工、时间口径统一、敏感度分级三条硬约束下产出特征字典与联邦特征筛选协议。跨境场景下双方快照冻结协议必须处理结算日与入账延迟差异。

## 上游输入

M3：样本空间与 N_eff · M0：可交换信息清单

## 输出契约（本模块必须产出）

`feature_dictionary_{cn,hk}.yaml` · `time_window_spec.md` · `sensitivity_classification.csv` · `federated_selection_protocol.md` · `leakage_detector/` · `DR-M4-*.yaml`

**运行入口**：本模块凡产出处理或实验结果，一律以 `notebooks/<步骤号>_<简述>.ipynb` 为入口并**带输出提交**，使人打开仓库即可直接看到结果（《维护约束 v2》9.1）。逻辑放 `components/`，参数从 `configs/` 读；真实数据派生输出必须清除后再提交。

## 量化放行判据（不达标不放行）

- 字典字段完整度 100%（语义·计算逻辑·时间口径·敏感级·是否入模·入模理由）
- 穿越检测覆盖 100% 入模特征，穿越违规数 = 0
- 快照冻结协议含结算日差异处理规则
- 筛选协议 3/3（F-1/F-2/F-3）× 3/3 对比维度
- 特征集交付 3 套（R-A/R-B/R-C）；训练-服务一致性差异率 0

## 当前结论

暂无。本模块尚未开始，`step_ledger.yaml` 中 `steps` 为空。

## 未决问题

- 隐性泄露边界待定：分箱边界、特征名、缺失率、IV 排名是否可交换，需逐项论证

## 升级条件

穿越检测违规数 >0 且本模块无法修复 → 上报 M1/M3 重审切分与时点定义。

## 下一步

**S4.1** —— 详见 `docs/01-loops/2_模块Loop执行规范_v2_步进量化版.md` 中 LOOP M4 的步骤分解表。
执行前必须先写步骤声明（will_produce / will_not_produce / success_criteria / risk），
并在 `registry/module_status.yaml` 认领本模块。

---

*本 README 必须保持当前——它是下一个 Agent 接手时最先读的文件。每步提交时同步更新「状态 / 已用步数 / 当前结论 / 未决问题 / 下一步」五处。*
