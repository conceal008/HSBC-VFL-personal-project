# -*- coding: utf-8 -*-
"""platform · 编排层：主链路、断点续跑与运行清单。

《项目维护约束 v2》给本目录的职责是「批次、重跑、断点续跑、单方不可用降级」。
本模块实现前三项；单方不可用降级在 M8 之后启用（见 `degradation_supported`）。

**断点续跑的设计要点**：每个阶段的产物落成一个 `.npz` / `.json`，
文件名由「阶段名 + 运行指纹」决定。重跑时若产物已存在则直接读取，
因此「一口气跑完」与「中断后续跑」必须得到**逐位相同**的结果——
这正是门禁 7 要验证的性质。若某阶段引入了未固定的随机性，两者就会不一致。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

FINGERPRINT_LEN = 12
JSON_INDENT = 2
DEGRADATION_SUPPORTED = False        # 单方不可用降级：M8 之后启用


def fingerprint(payload: Dict) -> str:
    """运行指纹：配置内容的稳定摘要，用于给产物命名并写进运行清单。"""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:FINGERPRINT_LEN]


@dataclass
class Stage:
    """一个可缓存的阶段。`run` 接收上游产物字典，返回本阶段产物字典。"""
    name: str
    run: Callable[[Dict], Dict]


@dataclass
class RunReport:
    fingerprint: str
    stages_executed: List[str] = field(default_factory=list)
    stages_reused: List[str] = field(default_factory=list)
    seconds: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return {"fingerprint": self.fingerprint,
                "stages_executed": self.stages_executed,
                "stages_reused": self.stages_reused,
                "seconds": self.seconds,
                "total_seconds": sum(self.seconds.values()),
                "degradation_supported": DEGRADATION_SUPPORTED}


def _artifact_path(checkpoint_dir: str, stage: str, fp: str) -> str:
    return os.path.join(checkpoint_dir, "%s.%s.npz" % (stage, fp))


def _save(path: str, payload: Dict) -> None:
    arrays, scalars = {}, {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        else:
            scalars[key] = value
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload_out = dict(arrays)
    payload_out["__scalars__"] = np.array(
        json.dumps(scalars, ensure_ascii=False, default=str))
    np.savez(path, **payload_out)


def _load(path: str) -> Dict:
    with np.load(path, allow_pickle=False) as data:
        out = {k: data[k] for k in data.files if k != "__scalars__"}
        if "__scalars__" in data.files:
            out.update(json.loads(str(data["__scalars__"])))
    return out


def run_pipeline(stages: List[Stage], config: Dict, checkpoint_dir: str,
                 stop_after: Optional[str] = None) -> Dict:
    """按顺序跑各阶段，产物落盘并在重跑时复用。

    `stop_after` 给定时在该阶段之后停下——用于模拟中断。
    再次调用（不带 stop_after）即为续跑：已完成的阶段直接读盘，不重算。
    """
    fp = fingerprint(config)
    report = RunReport(fingerprint=fp)
    ctx: Dict = {"config": config}

    for stage in stages:
        path = _artifact_path(checkpoint_dir, stage.name, fp)
        if os.path.exists(path):
            ctx.update(_load(path))
            report.stages_reused.append(stage.name)
        else:
            started = time.time()
            produced = stage.run(ctx)
            report.seconds[stage.name] = time.time() - started
            _save(path, produced)
            ctx.update(produced)
            report.stages_executed.append(stage.name)
        if stop_after and stage.name == stop_after:
            break

    ctx["__report__"] = report.as_dict()
    return ctx


def write_manifest(path: str, report: Dict, config: Dict) -> None:
    """运行清单：让任何人能回答「哪个 config、哪些种子、哪个代码版本」。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"report": report, "config": config}, fh,
                  ensure_ascii=False, indent=JSON_INDENT, default=str)
