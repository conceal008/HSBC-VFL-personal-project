# 当前项目状态、有效结果与缺口

> 编写日期：2026-10-15 ｜ 版本：v1.0 ｜ 状态：用户暂停实验，归档交接
> 数据来源：模块台账、已发布S5.X3/S7.P2/S3.P3工件、S5.D2失败状态
> 口径基准：Public/main、DR-GOV-012、用户最新暂停指示

## 项目定位与证据层级

主线为内地与香港双方**共同客户联合营销**。当前完成的是公开研究代理和合成数据上的方法与工程验证，尚不是银行业务验证。业务产品、观察窗、转化定义、可用规模等仍有显式假设/待需求方确认项，M0升级和未签署风险接受没有被模型效果解除。

最新已发布实验基线为`9d222d869ec8748754ef9aa0788a2060625616ee`，远端[Actions 38696097711](https://github.com/conceal008/HSBC-VFL-personal-project/actions/runs/38696097711)成功。本次归档提交另有发布回执，不能用该旧CI证明新改动通过。

## 已完成

- **S5.X3：14项任务×5 seed，280最终逻辑模型＝270新拟合＋10经完整性核验复用**；4个历史任务另列。14任务不是14份独立数据集：Orange包含两宽度×三标签，HSBC_mock包含4个相关构造版本。
- 任务包括Home Credit、FedAds、Criteo、Orange small/large的churn/appetency/upselling、HSBC_mock balanced/overlap20/50/80、Santander产品。历史参考为UCI Bank、Hillstrom、COIL2000、Santander Transaction。**Amex继续暂缓**。
- 固定资源cohort上完整prepared开发分区；训练前标签无关上限25000，不称原始文件全量。全部每seed、95% CI、配对差、条件检验和正负结果在[结果表](../04-experiment-records/reports/本地数据集联邦训练与结果汇总_v2.md)与[机器汇总](../../modules/m5_modeling/results/portfolio_training_v2/all_local_summary.json)。3018数值路径包含统计元数据，不全是性能指标。
- 原两代理已完成单变量特征工程C01、多组组合/置乱/融合、固定预算主动方参照和48轮联合预算诊断。方法、负结果和适用边界见[实验台账](../04-experiment-records/实验总台账.md)。
- S7.P2：原两任务660输出场景、800密态属性拟合完成，见[输出重评记录](../../modules/m7_security/新增输出重评实验记录.md)；它**不覆盖后来14任务的新模型输出**。
- S3.P3通用秘密输出求交原型完成10轮，未接入训练gather；本轮训练仍为预对齐功能模拟，不能称已执行真实训练链路PSI。
- 双VM、独立非特权账号、只读输入、mTLS和CHEETAH密态训练已作工程验收；假设宿主管理员可信、半诚实不合谋，不能据此声称零泄漏或生产安全。

## 目前可说的结论

在构造的字段归属、cohort和固定开发划分下，Home Credit、四个Mock、Santander产品及部分Orange有正向联合信息收益。FedAds差值CI跨零且AUC区间覆盖0.5；Criteo联合中心化Qini点估计为负且差值CI跨零；Orange large churn无正向证据。具体数字一律引用[逐任务表](../04-experiment-records/reports/本地数据集联邦训练与结果汇总_v2.md)，不跨任务混合平均。

FedAds、Criteo、Orange large appetency/upselling的B0点估计低于至少一个单方LR/树，但强单方配对CI尚未交付，不能称已证实统计劣势。即使B0相对同优化器A0有增益，也未证明胜过低成本L1或强单方。Mock为公开Santander派生构造，非HSBC客户，版本名不是实际交集率、四版本不是独立人群。

五seed条件符号翻转检验最小双侧p=0.0625；不能宣称p<0.05显著业务增益。开发CI条件于固定人群/训练划分，不能替代独立人群确认。捕获率也不能替代营销因果收益。

## 暂停和未完成

- S5.D2：仅一次预检失败、0任务完成。修正一已声明但未执行；代码补丁后没有复测。草稿的新增firewall测试存在未定义`rt`的作用域错误，原样归档，不掩盖缺陷。
- 新14任务的输出风险重评、真正未查看人群确认、分组正则、安全树、正式L1聚合统计比较未完成。
- 秘密PSI与安全训练的端到端集成、攻击者实际可获信息下的更强推断/合谋/长期查询评估、工业多机与运维验证待做。
- 没有逐epoch联合loss/gradient/checkpoint的14任务工件，只能补端点诊断；不能凭最终预测补造收敛曲线。
- M5历史21/21、3次升级、14/21有修正、通过线11保持冻结；DR-M5-008仅另解锁D2一步，暂停/归档不算通过，不稀释原质量问题。第25消费步骤才进入下一强制抽查窗口。

## 模块台账快照

下表为本次整理后的已登记步骤数；不是业务完成百分比。M5 D2暂停尝试另记，M9包含本次S9.H1。

| 模块 | 已登记步骤 / 预算 | 当前入口 |
|---|---|---|
| M0 | 9 / 10 | [m0_compliance](../../modules/m0_compliance/README.md) / [台账](../../modules/m0_compliance/step_ledger.yaml) |
| M1 | 9 / 18 | [m1_data_selection](../../modules/m1_data_selection/README.md) / [台账](../../modules/m1_data_selection/step_ledger.yaml) |
| M2 | 3 / 9 | [m2_synthetic](../../modules/m2_synthetic/README.md) / [台账](../../modules/m2_synthetic/step_ledger.yaml) |
| M3 | 4 / 9 | [m3_alignment](../../modules/m3_alignment/README.md) / [台账](../../modules/m3_alignment/step_ledger.yaml) |
| M4 | 1 / 10 | [m4_features](../../modules/m4_features/README.md) / [台账](../../modules/m4_features/step_ledger.yaml) |
| M5 | 21 / 22 | [m5_modeling](../../modules/m5_modeling/README.md) / [台账](../../modules/m5_modeling/step_ledger.yaml) |
| M6 | 1 / 9 | [m6_evaluation](../../modules/m6_evaluation/README.md) / [台账](../../modules/m6_evaluation/step_ledger.yaml) |
| M7 | 7 / 15 | [m7_security](../../modules/m7_security/README.md) / [台账](../../modules/m7_security/step_ledger.yaml) |
| M8 | 1 / 10 | [m8_industrialization](../../modules/m8_industrialization/README.md) / [台账](../../modules/m8_industrialization/step_ledger.yaml) |
| M9 | 3 / 10 | [m9_documentation](../../modules/m9_documentation/README.md) / [台账](../../modules/m9_documentation/step_ledger.yaml) |

初版合成结论与后续实际代理结果须分层阅读。S5.4/S5.P2/S7.2等旧全部通过自评的失效范围，以对应audit_log与独立复评记录为准，原资产不删除、新结果不追认旧流程。M9的技术侧材料完整不等于法务审核或风险接受签署。
