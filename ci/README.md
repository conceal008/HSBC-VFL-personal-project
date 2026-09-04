# ci/ · 门禁脚本

《项目维护约束 v2》第 8 部分定义了 7 道门禁，按成本从低到高执行，前面失败不跑后面。

| 门禁 | 内容 | 脚本 | 状态 |
|---|---|---|---|
| 1 | 数据合规扫描（凭证·数据扩展名·体积·个人标识模式·.env） | `check_secrets.sh` | ✅ 已实现并在 CI 生效 |
| 2 | 步骤与元数据完整性（Change-Id / Step-Id / **Step-Score ≥10** / 声明 / 证据 / 修正轮次） | `check_step_metadata.py` | ✅ 已实现并在 CI 生效 |
| 2 | changelog schema 校验（字段类型与必填、命名规范、step_id 唯一） | `check_changelog_schema.py` | ✅ 已实现并在 CI 生效 |
| — | **门禁回归测试**（验证门禁真的拦得住） | `tests/run_gate_tests.sh` | ✅ 17 组用例，每次 push 跑 |
| 3 | ~~单步规模~~ | ~~`check_step_scope.py`~~ | ⛔ **已取消**（2026-08-30，DR-GOV-003）：编号保留不复用，脚本不再实现 |
| 4 | 合规一致性核验（六项，原则三的载体） | `check_cross_border_consistency.py` | ✅ 已实现并在 CI 生效 |
| 5 | 代码质量（lint / 类型 / 单测覆盖 ≥70% / 组件 schema / 冒烟 / **术语禁用词**） | `check_code_quality.py` | ✅ 已实现并在 CI 生效 |
| 6 | 可复现性（魔数 = 0 / 种子来自配置 / 环境锁一致 / notebook N1·N2·N4·N5） | `check_reproducibility.py` | ✅ 已实现并在 CI 生效 |
| 7 | 端到端冒烟（主链路 / 时长 ≤5 分钟 / **断点续跑逐位一致**） | `check_e2e_smoke.py` | ✅ 已实现并在 CI 生效 |

## 门禁 4 依赖的两份契约文件

`check_cross_border_consistency.py` 除 `registry/cross_border_assets.yaml` 外，还读两份**尚未产出**的文件。
它们的路径与最小结构在此固定为契约，产出方按此写：

| 文件 | 谁产出 | 最小结构 |
|---|---|---|
| `registry/compliance_routes.yaml` | **M0 的 S0.9**（判决主线+回退线，DR-M0-001） | `routes: [{id: R-B, status: primary}, {id: R-C, status: fallback}, {id: R-A, status: rejected}]`；status 取 `primary` / `fallback` / `rejected` / `undecided` |
| `platform/governance/consent_batches.yaml` | **M8**（同意管理实现） | `batches: [{id: cn_marketing_consent_v2}]` |

**文件不存在 ≠ 放行**：只要有组件声明引用了 `legal_basis_ref` 或 `consent_dependency.requires_batch`，
而对应契约文件缺失，核验 3 / 核验 6 直接阻断并说明缺的是哪一步的产出。
没有任何声明时（当前状态）六项无核验对象，通过。

## 为什么门禁 1 先上线

它防的是真实数据与凭证入库——本项目**唯一不可逆**的错误（2.4：用新提交"覆盖"没有用，git 历史里它还在，且仓库为 Public）。
其余门禁校验的是元数据与一致性，在第一条业务提交出现之前无对象可校验。

## 实现顺序（附录 B）

`check_cross_border_consistency.py`（越早上线 M9 越省事）→ `check_step_metadata.py`（步进机制的强制力来源）→ `check_changelog_schema.py` → `check_reproducibility.py`（门禁 6）→ `check_code_quality.py`（门禁 5）→ `check_e2e_smoke.py`（门禁 7）。门禁 5 与 7 的触发条件是「第一份代码组件 / 主链路出现」，2026-09-03 初版满足后随即补齐。

**唯一正确的提交前入口是 `bash ci/run_all_gates.sh`**——不要手写 `cmd | tail -1 && git push` 这类管道链，管道的退出码是 tail 的（永远 0），本项目已因此两次把不通过门禁的提交推进 main。
门禁 3（单步规模）已取消，不再实现。

## 门禁回归测试

```bash
bash ci/tests/run_gate_tests.sh
```

12 组用例覆盖三个门禁的**放行**与**拦截**两侧：门禁 1（干净放行 / 拦数据·凭证·手机号·身份证·卡号）、
门禁 4（合规声明放行 / 六项逐项拦截 / 契约文件缺失不放行）、门禁 2（无 trailer / 元数据全面违规 /
DR 无 falsifier / 本仓库 HEAD 自检）、changelog schema（缺字段与非法枚举 / 文件名与 step_id 冲突 / 本仓库全绿）。

**夹具纪律**：凡含个人标识形态或凭证形态的样本一律在脚本里**运行时拼装**，绝不落盘提交——
否则门禁 1 扫本仓库时会命中夹具自身，把真仓库拦下来。已落盘的夹具（`ci/tests/fixtures/`）
只有 YAML 声明与 commit message，跨方通信样本存为 `sender.py.txt`，由测试脚本复制成 `.py` 到临时目录。

改任何门禁脚本后必须跑这个：门禁"不误报"很容易验证，"能拦住"却容易假通过。

## 本地使用

```bash
bash ci/check_secrets.sh                          # 门禁 1，提交前必跑
python3 ci/check_cross_border_consistency.py      # 门禁 4，改动声明或 M0 清单后必跑
python3 ci/check_step_metadata.py                 # 门禁 2，校验 HEAD；查别的提交加 --rev <sha>
python3 ci/check_changelog_schema.py              # 门禁 2 下半，校验全部 changelog 条目
python3 ci/check_reproducibility.py               # 门禁 6，改动代码/notebook/配置后必跑
python3 ci/check_code_quality.py                  # 门禁 5，改动组件或测试后必跑
python3 ci/check_e2e_smoke.py                     # 门禁 7，改动主链路或 platform/ 后必跑
bash ci/tests/run_gate_tests.sh                   # 门禁回归，改任何门禁脚本后必跑
```

两者退出码 1 即阻断（门禁 4 的退出码 2 表示缺 PyYAML）。

脚本兼容 macOS 自带 bash 3.2 与 CI 的 bash 5。扫描对象是 git 跟踪的文件；不在 git 仓库时退回全目录扫描。

## 已知局限（如实记录，不要假装门禁比实际更强）

- 个人标识模式是启发式正则，只覆盖内地手机号 / 18 位身份证 / 16–19 位卡号 / 香港身份证号四种形态，**不能替代人工审查**。
- 不检测语义层面的 L-受限产物（例如一张只有比率没有标识的真实数据分布表），这类必须靠 PR 自查表与人工审查。
- 与 GitHub 自带的 secret scanning + push protection **互补而非重复**：后者在服务端 push 时拦截**已知格式的凭证**（已确认为 enabled）；
  门禁 1 在 CI 中另查数据类文件扩展名、文件体积与个人标识形态（手机号 / 身份证 / 银行卡 / 香港身份证号），这些 GitHub 不查。

**门禁 2**

- `check_changelog_schema.py` 校验全部条目的结构，`check_step_metadata.py` 深校验 HEAD 那一条的自洽性；
  两者分工不同，都必须通过。
- 只校验**一个**提交（默认 HEAD，PR 上取 head sha），不回溯全历史；批量体检需自行循环 `--rev`。
- 通过线默认 10，若模块 `step_ledger.yaml` 的 `metrics.pass_threshold` 被回溯抽查上调为 11，自动取 11。
- 校验的是元数据**自洽**（trailer 与 changelog 一致、总分等于四维之和、每维有证据引用、硬门全 pass），
  **不能判断评分是否诚实**——3 分的证据写得对不对，只有回溯抽查与下游反证能发现（1.4）。
- `doc` / `chore` 可省略 step 段与 Step-Id，其余七类必填。合并提交自动跳过。

**门禁 6**

- 魔数判定用 AST，不是正则：`0 / 1 / -1 / 2`（及其浮点形式）不算魔数；
  模块级**全大写**变量的赋值不算魔数（那是具名常量）；确有必要的字面量可加
  `# 魔数豁免: <理由>` 放行——**豁免必须写理由**，空豁免等同违规，靠 PR 复核。
- 种子检查覆盖 `seed=` / `random_state=` / `random_seed=` 与
  `np.random.seed()` / `random.seed()` / `torch.manual_seed()` / `tf.random.set_seed()`
  的**数字字面量**形式；从配置读入的写法（`cfg["seed"]`）不受影响。
- notebook 无输出时只 WARN 不阻断——因为按 9.1，真实数据派生输出**本来就必须清除**，
  机器无法区分"清除了"与"根本没跑"。这一条只能靠 changelog 的 `sensitive_review` 人工留痕。
- N2（逻辑不放 notebook）以代码行数 >150 行 WARN 的方式近似，不阻断：行数是弱信号，
  真正的判断是"这段逻辑是否应该被单测覆盖"，机器判不了。

**门禁 4**

- 核验 4「无未声明流动」靠正则匹配跨方通信特征（socket / grpc / requests / httpx / urllib /
  `send_to_party` 等），**只能发现形似的调用**：换用未列入的库或自写封装即可绕过。
  模式表在脚本顶部 `COMM_PATTERNS`，新增通信方式时必须同步补充——这条纪律无法由脚本自我保证。
- 声明必须位于代码文件所在目录或其祖先目录（组件根 / 模块根），不接受子目录里的声明反向覆盖父目录。
- 核验 2 只比对 `category` 与 `is_personal_information` 两个字段；`allowed_path`、`protection_required`
  等字段的一致性尚未纳入，需要时在 `check_1_2_3_5_6` 里加字段名即可。
- 构造样本测试目前在本地临时目录执行（三组用例：全合规通过 / 六类违规全拦截 / 契约文件缺失不放行），
  **尚未作为回归夹具提交到仓库**，他人无法一键复跑。补夹具是 S-INIT.3 的附带项。
