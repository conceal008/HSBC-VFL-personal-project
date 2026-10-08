# main 门禁兼容修复：步骤声明

> 编写日期：2026-09-29 ｜ 版本：v1.0 ｜ 状态：执行声明
> 数据来源：main 提交 `62536bc` 的 GitHub Actions run `36560490051`，及 PR #1 已验证的 CI 修复
> 口径基准：只修复工程门禁的环境与回归兼容性

- Step-Id：S-INIT.10；Change-Id：CL-20260929-GOV-002。
- will_produce：为 main 的 CI 安装 Jupyter Python 内核，锁定 Ruff/MyPy 到已提交环境锁版本，令类型检查只接收存在的源码目录，并使 PR 门禁回归检出真实 head 提交；记录首次失败与修复后结果。
- will_not_produce：合并 PR #1 的数据准备实现、修改联邦训练逻辑、上传数据或派生产物、抹除 main 首次远端失败。
- success_criteria：main 本机完整门禁通过，GitHub Actions 的八项作业全部通过；changelog 与交接可追溯两类原始失败。
- risk：Linux 与 macOS 的 Jupyter 环境差异仍会使单平台测试数量不同；本步只验证工程门禁，不证明物理隔离或安全联合训练。
