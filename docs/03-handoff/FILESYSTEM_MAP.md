# 文件、权限与运行环境地图

> 编写日期：2026-10-15 ｜ 版本：v1.0 ｜ 状态：用户暂停实验，归档交接
> 数据来源：模块台账、已发布S5.X3/S7.P2/S3.P3工件、S5.D2失败状态
> 口径基准：Public/main、DR-GOV-012、用户最新暂停指示

## Public仓库的固定职责

| 位置 | 职责 | 维护原则 |
|---|---|---|
| `README.md`、本目录`START_HERE.md` | 唯一当前导航与接手入口 | 先更新状态，历史入口不再主导执行 |
| `docs/00-framework`、`01-loops`、`02-governance` | 目标、步骤、治理 | 不把旧副本当当前规范 |
| `docs/03-handoff` | 当前接手清单和历次HO记录 | 保留旧HO，当前入口明确取代旧“下一步” |
| `docs/04-experiment-records/logbooks` | v1–v10历史实验日志正文 | 只做路径整理；每版保留当时边界 |
| `docs/04-experiment-records/reports` | 14＋4任务完整结果总报告 | 公开指标仍指原results；不是新结果 |
| `modules/m*/components/configs/notebooks/tests` | 可用模块代码与运行契约 | 原路径保留，避免破坏源码/配置指纹 |
| `modules/m*/results` | 已审查冻结JSON/CSV/图表 | 不改写、不搬移；新结果另建版本 |
| `modules/m5_modeling/experiments/paused/S5.D2_20261015` | 未验收暂停实现 | `.pending`不可直接发现/执行；先读缺陷清单 |
| `registry`、`changelog` | 决策、分支、声明、步骤和发布 | 历史追加，不删除失效/失败记录 |

旧根目录日志/报告文件仍在，但已变为指向完整正文的导航。正文仅机械修正相对Markdown链接，全部文字与数值保留。迁移路径/前后指纹见[清单](../04-experiment-records/migration_manifest.json)。冻结结果与原训练代码一字未改。

## 本地私有资产

本机项目根为`/Applications/总文件夹/汇丰`，仓库为其下`HSBC-VFL-personal-project`，私有工作区为`8-9月`。新的**`8-9月/项目交接中心/README.md`**集中连接规范、原始数据/文献、准备/训练/评估的各阶段目录、暂停草稿、环境和恢复回执。

原始数据、清洗/切分行、owner映射、逐行分数/标签、模型份额、学习参数和VM私钥只在本地/各方VM，不上传Public。本地中心使用导航链接和元数据索引整合，保持冻结路径原位；“仓库产出副本”等旧副本只作历史，不是当前代码来源。项目唯一跨阶段总日志为`6月前准备工作/项目实时日志.md`。

## 环境与停机备份

- 宿主Python：`8-9月/数据处理环境/v1/bin/python`。项目CLI：`/opt/homebrew/bin/limactl`。
- `LIMA_HOME=/private/tmp/hsbc-vfl-lima`，VM名`hsbc-alice`、`hsbc-bob`，guest Python`/opt/secretflow/bin/python`。两盘逻辑容量各64GiB，当前均Stopped。
- guest各方训练资产：`/srv/vfl/alice`与`/srv/vfl/bob`。D2新能力目录`/srv/vfl_endpoint/alice`、账号`alice_endpoint`；不是旧源账号，首次预检失败资产仍保留。
- 为避免临时路径意外清理，本次在本地中心`40_环境与恢复/双VM停机检查点_20261015`创建完整停机APFS克隆，28项文件；原VM目录未改路径、未删除。备份内含敏感资产，整个目录仅本机700/文件600，**永不入库**。
- 小文件进行了SHA256逐项比对；两块大磁盘的检查是APFS克隆成功与逻辑长度一致，**没有执行全盘SHA256**，不能声称逐字全盘哈希验收。恢复前应先校验私有CHECKPOINT_MANIFEST，不覆盖原目录，先副本验证。

本次仅正常停止VM和备份，没有重建网络、改变原数据权限、清理旧router或启动新模型。恢复现场以本地暂停回执和原路径为准。

## 2026-10-16 S9.H2现行补充

前文“旧根仅导航”描述S9.H1时点，现已由兼容清单v2取代：11旧文件恢复原文前缀、保留原行号/锚点，尾部指向规范docs，不能在旧镜像独立追加新实验。规范11正文与旧清单未改。

本地状态以历史执行状态索引_v3.json和导航映射_v2.json为当前索引；旧版本保留。51个入口全部可达，1514状态的来源字段/退出未知/blocked得到保留，不推断当前进程。8-9月父目录现700；700/600原描述只针对中心/备份，不曾意味着所有源文件都该权限。原数据文件mode/字节未改。

LIMA_HOME实际指向8-9月/功能模拟环境/lima。28文件检查点不包含宿主Homebrew Lima/Python，备份SSH配置的IdentityFile/ControlPath引用原canonical路径。恢复前在副本处理路径并核对依赖；原配置不变，未做启动恢复或全盘哈希。可选socket_vmnet模板缺路径不等于当前user-v2网络损坏。
