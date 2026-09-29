# 交接记录：main 远端门禁验收补录

> 会话日期：2026-09-29 ｜ Agent：Codex ｜ 范围：GOV
> Step-Id：S-GOV.13 ｜ Change-Id：CL-20260929-GOV-003

## 可复核结论

- main 的进度同步提交 `62536bc` 在 [run 36560490051](https://github.com/conceal008/HSBC-VFL-personal-project/actions/runs/36560490051) 失败：门禁 5 的 Q1 为未锁定 Ruff 引起的 ISC004/RUF100，Q3 为缺少 `python3` Jupyter 内核。后续门禁按依赖关系跳过。
- 独立整改提交 `6e5d7493ad8dc5b5523470532e49f8a6e2669994` 的 [run 36561258977](https://github.com/conceal008/HSBC-VFL-personal-project/actions/runs/36561258977) 为 completed/success。数据合规、步骤/schema、合规一致性、代码质量、可复现性、端到端、M9 证据链、门禁回归八项作业全部通过。
- 本机对应提交候选的完整门禁为 147 项测试通过、核心覆盖率 1297/1333（97.3%）、47/47 项回归通过。与远端结果分别记录。
- M1 工程前置 S1.P1–S1.P4 仍在草稿 PR #1，main 尚未合入其数据准备实现；正式 S1.1 和完整安全联合训练未验收。

## 本次 main 更新

只把已验证的远端结论补入 README、项目框架总览和治理状态，并新增本步骤声明及 changelog。本次文档候选的本机完整门禁通过：147 项测试、47/47 项回归；完整日志在仓库外 `8-9月/实验日志/验收_20260929-GOV13/gates.log`。此次文档补录将触发新 CI；新运行的实时状态仍以 GitHub Actions 页面为准。
