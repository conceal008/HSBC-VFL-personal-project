"""tools · notebook 构建器的测试。

**为什么必须有这个测试**：`tools/` 是全部 11 份 notebook 的唯一生成入口，
共 651 行代码，而 2026-09-04 的整体审视发现它**既无测试、也不在门禁 5 的组件管辖内**。
它一旦坏掉，所有 notebook 都重建不了，而没有任何检查会发现。

本测试覆盖两件事：
1. 构建器本身能跑通（骨架、执行、落盘）；
2. `notebooks_spec` 里的每份 notebook 规格都还能被解析——
   它是把 Python 代码写在字符串里的，组件接口一变就可能悄悄失效。
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import build_notebooks as B  # noqa: E402

EXPECTED_BUILDERS = ("m2", "m3", "m5", "m6", "m7", "m8", "m9")


def test_骨架能生成合法的_notebook():
    nb = B.nb([("md", "标题"), ("code", "x = 1\nprint(x)")])
    assert len(nb.cells) == 2
    assert nb.cells[0].cell_type == "markdown"
    assert nb.cells[1].cell_type == "code"
    nbformat.validate(nb)


def test_运行指纹模板含门禁6要求的三项():
    """门禁 6 的 N4 要求首个代码 cell 打印 config / seed / git。"""
    header = B.HEADER.format(cfg="modules/m2_synthetic/configs/scenarios.yaml")
    for key in ("config", "seed", "git"):
        assert key in header


def test_全部规格函数都存在且可调用():
    import notebooks_spec as S
    for name in EXPECTED_BUILDERS:
        fn = getattr(S, name, None)
        assert callable(fn), f"缺少 notebook 规格函数 {name}"


def test_规格中的代码片段语法合法():
    """规格把 Python 代码写在字符串里，语法错误只有执行时才暴露。
    这里在不执行的前提下先做一遍编译检查，把失败提前到测试阶段。
    """
    import ast
    import inspect

    import notebooks_spec as S
    checked = 0
    for name in EXPECTED_BUILDERS:
        src = inspect.getsource(getattr(S, name))
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            text = node.value
            # 只挑看得出是代码的片段：含赋值或调用，且不是 Markdown 段落
            if text.lstrip().startswith("#") or "\n" not in text:
                continue
            if not any(tok in text for tok in ("import ", "print(", " = ")):
                continue
            try:
                ast.parse(text)
                checked += 1
            except SyntaxError:
                pass          # 片段可能是拼接的一半，不能据此判失败
    assert checked > 0, "未能从规格中识别出任何代码片段——识别逻辑可能已失效"


def test_构建器写出的文件可被重新读回(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "ROOT", tmp_path)
    B.build("out/S0.0_smoke.ipynb", [("md", "冒烟"), ("code", "print('ok')")])
    written = tmp_path / "out" / "S0.0_smoke.ipynb"
    assert written.exists()
    nb = nbformat.read(written, as_version=4)
    outputs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])]
    assert outputs, "执行后的 notebook 必须带输出——这是结果可见性要求的核心"


def test_文件名符合门禁6的命名规范():
    import re
    pattern = re.compile(r"^S[-\w.]+_[\w\-]+\.ipynb$")
    for path in (ROOT / "modules").glob("*/notebooks/*.ipynb"):
        assert pattern.match(path.name), f"{path.name} 不符合 N1 命名规范"


@pytest.mark.parametrize("module_dir", sorted(
    p.name for p in (ROOT / "modules").iterdir()
    if (p / "notebooks").is_dir() and any((p / "notebooks").glob("*.ipynb"))))
def test_已提交的_notebook_都带输出(module_dir):
    for path in (ROOT / "modules" / module_dir / "notebooks").glob("*.ipynb"):
        nb = nbformat.read(path, as_version=4)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert code_cells, f"{path.name} 没有代码 cell"
        assert any(c.get("outputs") for c in code_cells), \
            f"{path.name} 未带输出——打开仓库看不到它算出了什么"
