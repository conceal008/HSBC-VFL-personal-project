# -*- coding: utf-8 -*-
"""M8 · 故障注入：框架要求的 7 项，逐项在真实主链路上跑。

M8 的放行判据写明「故障注入 7/7 = 100% 通过」，且列出了七类故障。
本模块不做仿真式的「假装失败」，而是**真的把故障注入到 platform 的主链路里**，
再核对链路的行为是否符合预期。

七项各自防的是不同的事故：

| 编号 | 故障 | 不做会怎样 |
|---|---|---|
| F1 | 断点续跑 | 长任务中断后只能从头跑，且没人知道结果是否一致 |
| F2 | 单方下线降级 | 对方系统一挂，整条链路停摆；或更糟——**静默降级**出错误名单 |
| F3 | 网络抖动重试 | 一次瞬时失败就要人工重跑 |
| F4 | schema 变更安全失败 | 对方悄悄改了字段，**照样出分**，名单全错而无人察觉 |
| F5 | 数据延迟批次一致 | 分批到达与一次到达结果不同，名单随到达顺序变化 |
| F6 | 重复执行幂等 | 重跑产生重复名单或覆盖错数据 |
| F7 | 同意撤回剔除 | 撤回同意的主体仍被使用——这是合规事故，不只是技术缺陷 |
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

RESULT_PASS = "pass"
RESULT_FAIL = "fail"
TRANSIENT_FAILURES = 2          # F3 注入的瞬时失败次数
LATE_BATCH_SPLIT = 0.5          # F5 把数据切成两批的比例
FLAKY_STAGE_INDEX = 2           # F3 注入瞬时故障的阶段序号（m5_model）


def _case(fault_id: str, name: str, passed: bool, detail: str) -> Dict:
    return {"fault_id": fault_id, "name": name,
            "result": RESULT_PASS if passed else RESULT_FAIL, "detail": detail}


def _same(a: Dict, b: Dict) -> bool:
    if set(a) != set(b):
        return False
    return all(np.allclose(a[k], b[k]) if isinstance(a[k], float) else a[k] == b[k]
               for k in a)


def run_all(pipeline, main_chain, smoke: Dict, workdir: str) -> List[Dict]:
    """跑完七项，返回逐项结果。任何一项 fail 即视为 M8 放行判据未达标。"""
    cases: List[Dict] = []
    stages = main_chain.build_stages(smoke)
    baseline_ctx = pipeline.run_pipeline(stages, smoke, os.path.join(workdir, "base"))
    baseline = main_chain.summarize(baseline_ctx)

    # —— F1 断点续跑 ——
    d = os.path.join(workdir, "f1")
    pipeline.run_pipeline(stages, smoke, d, stop_after="m3_align")
    resumed = main_chain.summarize(pipeline.run_pipeline(stages, smoke, d))
    cases.append(_case("F1", "断点续跑", _same(baseline, resumed),
                       "在 m3_align 之后中断再续跑，结果与不中断逐位比对"))

    # —— F2 单方下线降级 ——
    deg_cfg = dict(smoke, party_b_available=False)
    deg_ctx = pipeline.run_pipeline(main_chain.build_stages(deg_cfg), deg_cfg,
                                    os.path.join(workdir, "f2"))
    deg_report = deg_ctx.get("__report__", {})
    deg_summary = main_chain.summarize(deg_ctx)
    ok_f2 = (deg_report.get("degraded") is True
             and bool(str(deg_report.get("degradation_reason") or "").strip())
             and all(k in deg_summary for k in main_chain.SUMMARY_KEYS))
    cases.append(_case("F2", "单方下线降级", ok_f2,
                       "被动方不可用时须降级出分**并如实上报**——静默降级比崩溃更危险"))

    # —— F3 网络抖动重试 ——
    flaky_state = {"n": 0}
    original = stages[FLAKY_STAGE_INDEX].run

    def flaky(ctx: Dict) -> Dict:
        flaky_state["n"] += 1
        if flaky_state["n"] <= TRANSIENT_FAILURES:
            raise ConnectionError("模拟网络抖动")
        return original(ctx)

    flaky_stage = pipeline.Stage(stages[FLAKY_STAGE_INDEX].name, flaky)
    patched = (stages[:FLAKY_STAGE_INDEX] + [flaky_stage]
               + stages[FLAKY_STAGE_INDEX + 1:])
    retried = main_chain.summarize(pipeline.run_pipeline(
        patched, smoke, os.path.join(workdir, "f3"),
        max_retries=TRANSIENT_FAILURES + 1))
    cases.append(_case("F3", "网络抖动重试",
                       _same(baseline, retried) and flaky_state["n"] > TRANSIENT_FAILURES,
                       "注入 %d 次瞬时失败后重试成功，结果与无故障时一致" % TRANSIENT_FAILURES))

    # —— F4 schema 变更安全失败 ——
    bad_cfg = dict(smoke, expected_dim_b=smoke["expected_dim_b"] + 1)
    try:
        pipeline.run_pipeline(main_chain.build_stages(bad_cfg), bad_cfg,
                              os.path.join(workdir, "f4"))
        ok_f4, detail_f4 = False, "被动方特征维数不符却照常出分——名单会全错而无人察觉"
    except main_chain.SchemaMismatch:
        ok_f4, detail_f4 = True, "维数不符时抛 SchemaMismatch 并停止出分"
    cases.append(_case("F4", "schema 变更安全失败", ok_f4, detail_f4))

    # —— F5 数据延迟批次一致 ——
    # 真实场景：先到一批数据跑了一轮，剩下的晚到，再跑一轮。
    # 要验的是**晚到批次的那一轮，结果必须等于「所有数据一开始就在」的那一轮**——
    # 即先前那轮的缓存产物不得被错误复用。这依赖运行指纹随配置变化而变化。
    n_total = int(baseline["n_usable"])
    d5 = os.path.join(workdir, "f5")
    partial_cfg = dict(smoke, n_party_a=int(smoke["n_party_a"] * LATE_BATCH_SPLIT))
    partial_ctx = pipeline.run_pipeline(main_chain.build_stages(partial_cfg),
                                        partial_cfg, d5)          # 第一批先到
    full_ctx = pipeline.run_pipeline(stages, smoke, d5)           # 余下批次晚到，同一目录
    partial_n = int(partial_ctx["n_usable"])
    late = main_chain.summarize(full_ctx)
    reused_stale = full_ctx["__report__"]["stages_reused"]
    ok_f5 = (_same(baseline, late) and partial_n < n_total and not reused_stale)
    cases.append(_case("F5", "数据延迟批次一致", ok_f5,
                       "先到 %d 条跑一轮，余下晚到后再跑：结果须等于一次到齐（%d 条），"
                       "且不得复用前一轮的陈旧产物（实际复用 %d 个阶段，须为 0）"
                       % (partial_n, n_total, len(reused_stale))))

    # —— F6 重复执行幂等 ——
    d6 = os.path.join(workdir, "f6")
    first = main_chain.summarize(pipeline.run_pipeline(stages, smoke, d6))
    second_ctx = pipeline.run_pipeline(stages, smoke, d6)
    second = main_chain.summarize(second_ctx)
    reused = second_ctx["__report__"]["stages_reused"]
    cases.append(_case("F6", "重复执行幂等",
                       _same(first, second) and len(reused) == len(stages),
                       "第二次执行全部 %d 个阶段复用产物，结果一致" % len(reused)))

    # —— F7 同意撤回剔除 ——
    revoke_n = max(int(smoke["n_party_a"] * smoke["revoke_share"]), 1)
    rev_cfg = dict(smoke, revoked_indices=list(range(revoke_n)))
    rev_ctx = pipeline.run_pipeline(main_chain.build_stages(rev_cfg), rev_cfg,
                                    os.path.join(workdir, "f7"))
    kept = rev_ctx["kept_mask"].astype(bool)
    leaked = int(kept[:revoke_n].sum())
    shrunk = int(rev_ctx["n_usable"]) < n_total
    cases.append(_case("F7", "同意撤回剔除", leaked == 0 and shrunk,
                       "撤回 %d 个主体后，其中仍被使用的有 %d 个（须为 0），"
                       "可用样本由 %d 降至 %d"
                       % (revoke_n, leaked, n_total, int(rev_ctx["n_usable"]))))
    return cases


def summarize(cases: List[Dict]) -> Dict:
    passed = sum(1 for c in cases if c["result"] == RESULT_PASS)
    return {"total": len(cases), "passed": passed,
            "all_passed": passed == len(cases),
            "failed_ids": [c["fault_id"] for c in cases if c["result"] != RESULT_PASS]}
