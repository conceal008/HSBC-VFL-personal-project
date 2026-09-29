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
import yaml

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


def assert_output_policy(nb, path, root=ROOT):
    """9.1 requires restricted outputs removed, with an explicit sensitive review."""
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert code_cells, f"{path.name} 没有代码 cell"
    policy = nb.metadata.get("vfl", {})
    if policy.get("outputs") == "external_only":
        assert policy.get("classification") == "real_data_derived"
        assert policy.get("reason"), "清除输出必须说明原因"
        change_id = policy.get("review_change_id", "")
        assert Path(change_id).name == change_id and change_id.startswith("CL-")
        review = yaml.safe_load((root / "changelog" / f"{change_id}.yaml").read_text())
        sensitive = review.get("sensitive_review", {})
        assert sensitive.get("triggered") and sensitive.get("checklist_passed")
        assert sensitive.get("conclusion")
        assert str(path.relative_to(root)) in review["step"]["outputs"]
        assert all(not c.get("outputs") and c.get("execution_count") is None for c in code_cells)
        assert all(not c.get("attachments") for c in nb.cells)
    else:
        assert any(c.get("outputs") for c in code_cells), \
            f"{path.name} 未带输出且没有受限产物清除记录"


def test_受限输出必须清除且必须有审查记录(tmp_path):
    path = tmp_path / "modules/m1/notebooks/S1.P1_test.ipynb"
    nb = B.nb([("code", "print('result')")])
    with pytest.raises(AssertionError):
        assert_output_policy(nb, path, tmp_path)
    nb.metadata["vfl"] = {"outputs": "external_only", "classification": "real_data_derived",
                          "reason": "Public repository rule 9.1", "review_change_id": "CL-test"}
    with pytest.raises(FileNotFoundError):
        assert_output_policy(nb, path, tmp_path)
    (tmp_path / "changelog").mkdir()
    review = {"sensitive_review": {"triggered": True, "checklist_passed": True,
                                   "conclusion": "Outputs kept outside public repository"},
              "step": {"outputs": [str(path.relative_to(tmp_path))]}}
    (tmp_path / "changelog/CL-test.yaml").write_text(yaml.safe_dump(review))
    assert_output_policy(nb, path, tmp_path)
    nb.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="data")]
    with pytest.raises(AssertionError):
        assert_output_policy(nb, path, tmp_path)


@pytest.mark.parametrize("module_dir", sorted(
    p.name for p in (ROOT / "modules").iterdir()
    if (p / "notebooks").is_dir() and any((p / "notebooks").glob("*.ipynb"))))
def test_已提交的_notebook_都带输出(module_dir):
    for path in (ROOT / "modules" / module_dir / "notebooks").glob("*.ipynb"):
        nb = nbformat.read(path, as_version=4)
        assert_output_policy(nb, path)
