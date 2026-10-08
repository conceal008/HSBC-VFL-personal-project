# 交接记录：main 中的 M1 分支进度

> 会话日期：2026-09-29 ｜ Agent：Codex ｜ 范围：GOV
> Step-Id：S-GOV.12 ｜ Change-Id：CL-20260929-GOV-001

## 已核对的事实

- 远端仓库为 `conceal008/HSBC-VFL-personal-project`，默认分支 `main`。
- M1 工程前置 S1.P1–S1.P4 位于 `m1/notebook-data-preparation-v1`；最新提交 `31028763c3363c1600adea11560f1fd9bb0c2d59`。PR #1 当前为 open/draft，目标分支为 main，未合并。
- GitHub Actions 运行 `36535579716` 为 completed/success；数据合规、步骤/schema、合规一致性、代码质量、可复现性、端到端、M9 证据链及门禁回归八项作业均通过。分支上的工程报告记录本机 164 项测试通过、核心行覆盖率 96.1%（1813/1887）、47/47 项回归通过；这些数字只属工程验证。
- main 上的 M1 正式 S1.1–S1.9 尚未提交。PR 中完成的是工程前置，不能据此宣称物理隔离、真实 PSI、安全训练或业务效果验证完成。

## 本次 main 变更

README 和《项目框架与推进总览》补充当前分支进度；`registry/module_status.yaml` 增加独立的同步记录，保留 main 原有模块状态和历史快照；新增执行声明和 changelog。没有复制 PR 的实现或数据产物。

## 验证

- 本机 `ci/run_all_gates.sh` 全部门禁通过：147 项测试、核心逻辑行覆盖 1297/1333（97.3%）、47/47 项门禁回归通过。
- 首次受限沙箱运行的 Jupyter 测试无法绑定 `127.0.0.1`，因此门禁 5 与其回归自检失败；在允许本机回环端口的环境重跑后通过。两次日志分别保留于本机仓库外 `8-9月/实验日志/验收_20260929-GOV12/`。
- 此次检查验证的是 main 的文档提交候选；PR #1 的独立分支已由远端运行 `36535579716` 通过全部八项作业。

## 交接与下一步

以 [PR #1](https://github.com/conceal008/HSBC-VFL-personal-project/pull/1) 及其 [CI 运行](https://github.com/conceal008/HSBC-VFL-personal-project/actions/runs/36535579716) 核对前置工程进度。后续若合入 PR，再将 main 的 M1 模块台账、当前状态与实际合入提交对齐；正式任务仍从 S1.1 开始。
