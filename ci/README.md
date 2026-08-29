# ci/ · 门禁脚本

《项目维护约束 v2》第 8 部分定义了 7 道门禁，按成本从低到高执行，前面失败不跑后面。

| 门禁 | 内容 | 脚本 | 状态 |
|---|---|---|---|
| 1 | 数据合规扫描（凭证·数据扩展名·体积·个人标识模式·.env） | `check_secrets.sh` | ✅ 已实现并在 CI 生效 |
| 2 | 步骤与元数据完整性（Change-Id / Step-Id / **Step-Score ≥10** / 声明 / 证据 / 修正轮次 / changelog schema） | `check_step_metadata.py`、`check_changelog_schema.py` | ⬜ 未实现 |
| 3 | 单步规模（≤3 文件 / ≤300 行 / ≤1 路线） | `check_step_scope.py` | ⬜ 未实现 |
| 4 | 合规一致性核验（六项，原则三的载体） | `check_cross_border_consistency.py` | ⬜ 未实现 |
| 5 | 代码质量（lint / 类型 / 单测覆盖 ≥70% / 组件 schema / 冒烟） | — | ⬜ 待第一份代码组件出现后接入 |
| 6 | 可复现性（硬编码魔数 = 0 / 种子被实际使用 / 环境锁一致） | `check_reproducibility.py` | ⬜ 未实现 |
| 7 | 端到端冒烟（合并到 main 时跑） | — | ⬜ 待主链路出现后接入 |

## 为什么门禁 1 先上线

它防的是真实数据与凭证入库——本项目**唯一不可逆**的错误（2.4：用新提交"覆盖"没有用，git 历史里它还在，且仓库为 Public）。
其余门禁校验的是元数据与一致性，在第一条业务提交出现之前无对象可校验。

## 实现顺序（附录 B）

`check_cross_border_consistency.py`（越早上线 M9 越省事）→ `check_step_metadata.py`（步进机制的强制力来源）→ 其余。

## 本地使用

```bash
bash ci/check_secrets.sh          # 提交前必跑；退出码 1 即阻断
```

脚本兼容 macOS 自带 bash 3.2 与 CI 的 bash 5。扫描对象是 git 跟踪的文件；不在 git 仓库时退回全目录扫描。

## 已知局限（如实记录，不要假装门禁比实际更强）

- 个人标识模式是启发式正则，只覆盖内地手机号 / 18 位身份证 / 16–19 位卡号 / 香港身份证号四种形态，**不能替代人工审查**。
- 不检测语义层面的 L-受限产物（例如一张只有比率没有标识的真实数据分布表），这类必须靠 PR 自查表与人工审查。
- 未接入 GitHub Secret 扫描的 push protection，该项需仓库所有者在 GitHub 仓库设置中开启。
