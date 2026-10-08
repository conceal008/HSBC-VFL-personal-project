# 交接记录：main 工程门禁兼容修复

> 会话日期：2026-09-29 ｜ Agent：Codex ｜ 范围：GOV / CI
> Step-Id：S-INIT.10 ｜ Change-Id：CL-20260929-GOV-002

## 触发与诊断

进度同步提交 `62536bc` 已进入 main，但 [Actions run 36560490051](https://github.com/conceal008/HSBC-VFL-personal-project/actions/runs/36560490051) 的门禁 5 失败，门禁 7、M9 与回归作业因此跳过。日志中的 Q1 是新版 Ruff 对旧源码新增 ISC004/RUF100 报错；Q3 为 `NoSuchKernel: No such kernel named python3`，147 项中 146 通过、1 失败。两项均由 main 的直接依赖未固定或缺失引起，与进度文档内容无关。

## 整改

- `requirements.txt` 添加 `ipykernel`，把 Ruff 和 MyPy 固定到已提交的 `environments/python.lock` 版本。
- MyPy 仅接收实际存在的 `modules/`、`platform/` 源码目录，确保精简回归夹具可用；没有任何源码目录时仍阻断。
- 回归作业在 PR 上检出实际 head SHA，避免合成 merge commit 干扰 HEAD 自检；工作流回归数量提示更正为 47 组。

## 验证与交接

本步骤只改变工程依赖与门禁运行方式，不修改数据处理或联邦训练逻辑。本机完整 `ci/run_all_gates.sh` 已通过：147 项测试、核心行覆盖率 1297/1333（97.3%）、47/47 项门禁回归通过；原始日志仅保存在本机仓库外 `8-9月/实验日志/验收_20260929-INIT10/gates.log`。推送后以本提交触发的 GitHub Actions 为最终远端结果。原失败 run 保留供追溯。M1 的 S1.P1–S1.P4 仍位于草稿 PR #1，正式 S1.1 尚未完成。
