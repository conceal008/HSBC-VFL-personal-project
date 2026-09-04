"""platform · 编排层的单元测试。

断点续跑的正确性是门禁 7 的立身之本：若续跑与不中断能得到不同结果，
门禁 7 的 E3 就失去意义。因此这里直接测「续跑是否复用、结果是否逐位一致」。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
# 仓库目录 platform/ 与标准库模块 platform 同名，无法按包路径导入，
# 故把编排目录本身加入 sys.path。
sys.path.insert(0, str(ROOT / "platform" / "orchestration"))
sys.path.insert(0, str(ROOT))

import main_chain as MC  # noqa: E402
import pipeline as P  # noqa: E402

SEED = 11
ARRAY_LEN = 5
DOUBLE = 2


def _stages(calls):
    def one(_ctx):
        calls.append("one")
        return {"a": np.arange(ARRAY_LEN)}

    def two(ctx):
        calls.append("two")
        return {"b": ctx["a"] * DOUBLE}

    return [P.Stage("one", one), P.Stage("two", two)]


def test_指纹只依赖配置内容而非键序():
    assert P.fingerprint({"a": 1, "b": 2}) == P.fingerprint({"b": 2, "a": 1})
    assert P.fingerprint({"a": 1}) != P.fingerprint({"a": 2})


def test_首次运行执行全部阶段():
    calls = []
    with tempfile.TemporaryDirectory() as d:
        ctx = P.run_pipeline(_stages(calls), {"seed": SEED}, d)
    assert calls == ["one", "two"]
    assert ctx["__report__"]["stages_reused"] == []


def test_续跑复用已完成阶段且结果一致():
    calls = []
    cfg = {"seed": SEED}
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        full = P.run_pipeline(_stages(calls), cfg, d1)
        P.run_pipeline(_stages(calls), cfg, d2, stop_after="one")
        resumed = P.run_pipeline(_stages(calls), cfg, d2)
    assert resumed["__report__"]["stages_reused"] == ["one"]
    assert resumed["__report__"]["stages_executed"] == ["two"]
    assert np.array_equal(full["b"], resumed["b"])


def test_配置改变则不复用旧产物():
    calls = []
    with tempfile.TemporaryDirectory() as d:
        P.run_pipeline(_stages(calls), {"seed": SEED}, d)
        ctx = P.run_pipeline(_stages(calls), {"seed": SEED + 1}, d)
    assert ctx["__report__"]["stages_reused"] == []


def test_运行清单可回答配置与指纹():
    with tempfile.TemporaryDirectory() as d:
        ctx = P.run_pipeline(_stages([]), {"seed": SEED}, d)
        path = os.path.join(d, "manifest.json")
        P.write_manifest(path, ctx["__report__"], {"seed": SEED})
        assert os.path.exists(path)
        import json
        m = json.load(open(path, encoding="utf-8"))
    assert m["config"]["seed"] == SEED
    assert m["report"]["fingerprint"]


def test_降级开关与实际能力一致():
    """开关不是标志位，是承诺：置 True 就必须真的能降级并如实上报。

    M8（S8.1）之前该开关为 False，门禁 7 的 E4 只作提示；
    实现后置 True，E4 随即变成必须通过的检查——这个联动是设计好的。
    这里守的是「开关为真则能力必须存在」，避免有人为了让 E4 跳过而随手改回 False，
    也避免有人为了显得完备而在没实现时置为 True。
    """
    if not P.DEGRADATION_SUPPORTED:
        return
    smoke = dict(_smoke_cfg(), party_b_available=False)
    with tempfile.TemporaryDirectory() as d:
        ctx = P.run_pipeline(MC.build_stages(smoke), smoke, d)
    report = ctx["__report__"]
    assert report["degraded"] is True, "声明支持降级，但被动方下线时未标记 degraded"
    assert report["degradation_reason"].strip(), "标记了降级却没说明原因"
    assert "auc_l0" in MC.summarize(ctx), "降级后应继续出分，而不是停摆"


# ————————————————— 主链路 —————————————————

def _smoke_cfg():
    return yaml.safe_load(open(ROOT / "platform/configs/smoke.yaml", encoding="utf-8"))


def test_主链路五个阶段齐备():
    stages = MC.build_stages(_smoke_cfg())
    assert [s.name for s in stages] == ["m2_generate", "m3_align", "m5_model",
                                        "m6_evaluate", "m7_attack"]


def test_主链路端到端产出全部摘要项():
    smoke = _smoke_cfg()
    with tempfile.TemporaryDirectory() as d:
        ctx = P.run_pipeline(MC.build_stages(smoke), smoke, d)
    summary = MC.summarize(ctx)
    for key in MC.SUMMARY_KEYS:
        assert key in summary, f"主链路缺少产物 {key}"
    assert 0.0 <= summary["auc_l3a"] <= 1.0


def test_主链路可复现():
    smoke = _smoke_cfg()
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        a = MC.summarize(P.run_pipeline(MC.build_stages(smoke), smoke, d1))
        b = MC.summarize(P.run_pipeline(MC.build_stages(smoke), smoke, d2))
    assert a == b, "同配置两次运行结果不同——主链路存在未固定的随机性"
