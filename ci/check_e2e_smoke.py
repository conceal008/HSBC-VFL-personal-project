# -*- coding: utf-8 -*-
"""门禁 7 · 端到端冒烟（《项目维护约束 v2》第 8 部分 · 门禁 7）。

| 检查 | 门槛 |
|---|---|
| E1 模块主链路 | 跑通 |
| E2 运行时长 | ≤5 分钟（由 platform/configs/smoke.yaml 的 time_budget_seconds 给定） |
| E3 断点续跑 | 结果与不中断**逐位一致** |
| E4 单方不可用降级 | M8 之后启用；未到启用时点则报告并说明触发条件 |

E3 是本门禁真正的价值所在：它同时验证了两件事——
断点续跑机制本身可用，以及**主链路没有未固定的随机性**。
任何一处用了系统时间、未播种的随机数或依赖字典遍历顺序，两次结果就会分叉。

退出码：0 通过；1 阻断；2 依赖缺失。
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

SMOKE_CONFIG = "platform/configs/smoke.yaml"
ORCHESTRATION_DIR = os.path.join("platform", "orchestration")
FLOAT_TOL = 0.0                      # 逐位一致：不给容差
BLOCKS = []


def block(tag, msg):
    BLOCKS.append((tag, msg))
    print("❌ BLOCK  [%s] %s" % (tag, msg))


def _load_modules(root):
    # 仓库目录 platform/ 与标准库模块 platform 同名，无法按包路径导入
    # （stdlib 的 platform 是模块不是包），故把该目录本身加入 sys.path。
    path = os.path.join(root, ORCHESTRATION_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    import main_chain
    import pipeline
    return pipeline, main_chain


def check_e1_e2(pipeline, main_chain, smoke, workdir):
    stages = main_chain.build_stages(smoke)
    started = time.time()
    ctx = pipeline.run_pipeline(stages, smoke, os.path.join(workdir, "full"))
    elapsed = time.time() - started

    summary = main_chain.summarize(ctx)
    missing = [k for k in main_chain.SUMMARY_KEYS if k not in summary]
    if missing:
        block("E1", "主链路缺少产物：%s" % "、".join(missing))
    else:
        print("✅ E1 主链路跑通：%d 个阶段，%d 项产物"
              % (len(stages), len(summary)))

    budget = smoke["time_budget_seconds"]
    if elapsed > budget:
        block("E2", "主链路耗时 %.1fs > 预算 %ds——冒烟跑不完就没人会跑"
              % (elapsed, budget))
    else:
        print("✅ E2 运行时长：%.1fs ≤ %ds" % (elapsed, budget))
    return stages, summary


def check_e3(pipeline, main_chain, smoke, stages, baseline, workdir):
    """在每个阶段之后各中断一次，续跑后与不中断的结果逐位比对。"""
    for stage in stages[:-1]:
        d = os.path.join(workdir, "resume_" + stage.name)
        pipeline.run_pipeline(stages, smoke, d, stop_after=stage.name)
        ctx = pipeline.run_pipeline(stages, smoke, d)
        report = ctx["__report__"]
        if not report["stages_reused"]:
            block("E3", "在 %s 之后中断再续跑，没有复用任何阶段——"
                        "断点续跑机制未生效" % stage.name)
        resumed = main_chain.summarize(ctx)
        for key, want in baseline.items():
            got = resumed.get(key)
            if isinstance(want, float):
                if got is None or abs(got - want) > FLOAT_TOL:
                    block("E3", "在 %s 之后中断续跑，%s 与不中断不一致：%r vs %r——"
                                "主链路存在未固定的随机性"
                          % (stage.name, key, got, want))
            elif got != want:
                block("E3", "在 %s 之后中断续跑，%s 不一致：%r vs %r"
                      % (stage.name, key, got, want))
    if not BLOCKS:
        print("✅ E3 断点续跑：%d 个中断点，续跑结果与不中断逐位一致"
              % (len(stages) - 1))


def check_e4(pipeline, main_chain, smoke, workdir):
    """单方不可用降级：被动方下线时链路必须**降级出分并如实上报**。

    三条判据缺一不可：
    1. 不崩——降级路径本身要能跑通；
    2. 出分——降级后仍产出可用的名单，而不是空结果；
    3. **如实上报** `degraded=True` 并给出原因——静默降级比崩溃更危险，
       因为下游会把 L0 的名单当作联邦模型的名单来用。
    """
    if not pipeline.DEGRADATION_SUPPORTED:
        print("ℹ️  E4 单方不可用降级：**未启用**（框架规定 M8 之后启用）。"
              "触发条件——pipeline.py 的 DEGRADATION_SUPPORTED 置为 True 时，"
              "本项必须实现并通过。")
        return

    degraded_cfg = dict(smoke, party_b_available=False)
    try:
        ctx = pipeline.run_pipeline(main_chain.build_stages(degraded_cfg), degraded_cfg,
                                    os.path.join(workdir, "degraded"))
    except Exception as exc:                       # noqa: BLE001 —— 降级不该抛异常
        block("E4", "被动方不可用时链路抛出 %s：%s——降级路径必须能跑通"
              % (type(exc).__name__, exc))
        return

    report = ctx.get("__report__", {})
    summary = main_chain.summarize(ctx)
    if not report.get("degraded"):
        block("E4", "被动方不可用，但运行报告未标记 degraded——"
                    "**静默降级比崩溃更危险**：下游会把单方模型的名单当作联邦模型的名单")
    if not str(report.get("degradation_reason") or "").strip():
        block("E4", "标记了 degraded 但没有说明原因——运维看不出发生了什么")
    # 判据用主链路自己声明的 SUMMARY_KEYS，不写死某个字段名——
    # 写死字段名会让本检查只对当前这条链路有效，换条链路就形同虚设。
    missing = [k for k in main_chain.SUMMARY_KEYS if k not in summary]
    if missing:
        block("E4", "降级后缺少摘要项 %s——降级的目的是继续出名单，不是停摆"
              % "、".join(missing))

    normal = pipeline.run_pipeline(main_chain.build_stages(smoke), smoke,
                                   os.path.join(workdir, "normal_for_e4"))
    if normal.get("__report__", {}).get("degraded"):
        block("E4", "被动方可用时也被标记为 degraded——降级标记失去区分度")

    if not BLOCKS:
        print("✅ E4 单方不可用降级：降级出分且如实上报（原因：%s）"
              % report.get("degradation_reason"))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    print("== 门禁 7 · 端到端冒烟 ==")
    try:
        import yaml
    except ImportError:
        print("依赖缺失：PyYAML")
        return 2
    if not os.path.exists(SMOKE_CONFIG):
        block("E1", "%s 不存在——冒烟必须配置驱动，不得把规模写死在脚本里" % SMOKE_CONFIG)
        print("\n== 结果：BLOCK=%d ==" % len(BLOCKS))
        return 1

    smoke = yaml.safe_load(open(SMOKE_CONFIG, encoding="utf-8"))
    print("配置：%s ｜ 场景 %s ｜ 种子 %s"
          % (SMOKE_CONFIG, smoke["scenario"], smoke["seeds"]))
    print()

    pipeline, main_chain = _load_modules(root)
    workdir = tempfile.mkdtemp(prefix="gate7_")
    try:
        stages, summary = check_e1_e2(pipeline, main_chain, smoke, workdir)
        check_e3(pipeline, main_chain, smoke, stages, summary, workdir)
        check_e4(pipeline, main_chain, smoke, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("\n== 结果：BLOCK=%d ==" % len(BLOCKS))
    if BLOCKS:
        print("门禁 7 未通过：端到端冒烟不达标（第 8 部分 · 门禁 7）。")
        return 1
    print("门禁 7 通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
