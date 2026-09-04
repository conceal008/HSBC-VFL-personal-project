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


def check_e4(pipeline):
    if pipeline.DEGRADATION_SUPPORTED:
        block("E4", "已声明支持单方不可用降级，但本门禁尚未实现对应检查")
        return
    print("ℹ️  E4 单方不可用降级：**未启用**（框架规定 M8 之后启用）。"
          "触发条件——platform/orchestration/pipeline.py 的 "
          "DEGRADATION_SUPPORTED 置为 True 时，本项必须实现并通过。")


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
        check_e4(pipeline)
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
