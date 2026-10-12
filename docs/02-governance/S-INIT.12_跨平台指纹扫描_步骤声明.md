# S-INIT.12 跨平台指纹扫描

> 编写日期：2026-10-12 ｜ 版本：v1.0 ｜ 状态：执行前声明
> 数据来源：GitHub Actions 38217183847，原check_secrets.sh及真实配置文件指纹
> 口径基准：不改变冻结结果、不放宽客户标识或凭证拦截，Public汇总按DR-GOV-012

will_produce: 对JSON/Notebook等结构化文本统一扫描；只对明确列举的配置/代码/摘要SHA-256字段的完整64位十六进制值做标识扫描规范化，保留同一行其他内容和全部凭证扫描；新增跨MIME、真实标识和同一行凭证回归；修复远端误报并正常main发布。
will_not_produce: 不跳过整个JSON/结果文件，不改变标识/凭证正则，不豁免customer/input/identity哈希，不修改任何冻结指标、原散列或实验输入，不训练模型。
success_criteria: 原误报的配置指纹通过，真实标识/凭证同一行及JSON中仍阻断；Mac式JSON data与Linux式JSON text结果一致；完整门禁及远端CI复核。
risk: 这是已声明指纹字段的有限规范化，不能证明没有全部数据泄漏；贡献者将身份哈希冒写到配置字段仍需人工审查，禁止扩展为任意hash字段豁免。
