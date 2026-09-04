"""按 gate 6 的 N1–N5 规范生成并**真实执行** notebook，把结果固化进 .ipynb。

规范约束（ci/check_reproducibility.py）：
- N1 文件名 S<步骤号>_<简述>.ipynb
- N4 首个代码 cell 打印 config / seed / git
- N5 execution_count 自上而下单调递增
- R1 cell 内不得出现裸数字字面量（全大写模块常量除外）
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_SEC = 900

HEADER = '''ROUND_DP = 4          # 表格展示精度（不影响任何计算结果）
import sys, subprocess, json
from pathlib import Path
ROOT = Path.cwd()
while not (ROOT / "registry").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, yaml
CONFIG_PATH = ROOT / "{cfg}"
config = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
seed = config.get("seeds", [config.get("seed")])[0]
git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True, cwd=ROOT).stdout.strip()
print("config:", CONFIG_PATH.relative_to(ROOT))
print("seed  :", seed, "| 全部种子:", config.get("seeds"))
print("git   :", git or "(未提交)")
print("numpy :", np.__version__, "| pandas:", pd.__version__)'''


def nb(cells):
    n = nbformat.v4.new_notebook()
    n.cells = [nbformat.v4.new_markdown_cell(c[1]) if c[0] == "md"
               else nbformat.v4.new_code_cell(c[1]) for c in cells]
    n.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                 "name": "python3"},
                  "language_info": {"name": "python"}}
    return n


def build(path: str, cells) -> None:
    out = ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    n = nb(cells)
    NotebookClient(n, timeout=TIMEOUT_SEC, kernel_name="python3",
                   resources={"metadata": {"path": str(out.parent)}}).execute()
    nbformat.write(n, out)
    print("已生成并执行:", path)
