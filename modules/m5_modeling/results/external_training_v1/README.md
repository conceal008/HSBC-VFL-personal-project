# S5.X1 额外数据集训练结果

> 编写日期：2026-10-13 ｜ 版本：v1.0 ｜ 状态：实际完成，经公开研究汇总审查
> 数据来源：两份JSON内source_file、源码/配置/seed及原摘要SHA256
> 口径基准：DR-GOV-012；固定开发分区，非独立确认

- [COIL2000全部指标](coil2000.json)
- [Santander全部指标](santander_transaction.json)
- [所有数值路径及原值](metrics.csv)：含estimate/CI/每seed/p/配置seed，不能混作同一种指标。
- [工程与恢复验收](engineering.json)
- [完整实验分析](../../额外代理训练实验记录.md)
- [带输出Notebook](../../notebooks/S5.X1_external_training_summary.ipynb)

只提交已审查标量汇总，排除records、身份、行数据/预测、交集规模、学习状态、模型份额和凭证。原始私有结果不覆盖，不生成新CI、不删负结果。每份JSON逐文件来源及原值可以复核；实际训练Notebook输出在本方私有环境，公开入口清空。清洗后的train/validation源与新matrix/分位点分别留本方input/data，模型输出在各训练results。
