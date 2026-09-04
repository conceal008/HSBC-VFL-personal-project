# -*- coding: utf-8 -*-
"""门禁 5 · 代码质量（《项目维护约束 v2》第 8 部分 · 门禁 5）。

六项检查，对应框架表格的五行加上术语规范：

  Q1 Lint / 格式化      0 error（ruff）
  Q2 类型检查           0 error（mypy）
  Q3 单元测试           通过率 100%，核心逻辑行覆盖 ≥70%
  Q4 组件声明           每个组件 100% 有声明
  Q5 组件冒烟测试       每个组件 ≥1 个测试
  Q6 术语规范           禁用词 0（registry/glossary.yaml 的 forbidden 词表）

**范围决策（须知其代价）**：Q1/Q2 只覆盖 `.py` 源码，**不覆盖 notebook**。
notebook 由门禁 6 的 N1–N5 单独约束。理由是 ruff 对 notebook 的跨 cell 分析会把
首个 cell 的导入判为未使用，逼着把 notebook 写成不自然的形态；而 notebook 的
真正风险（魔数、种子、指纹、执行顺序）门禁 6 已经覆盖。
**代价**：notebook 内的语法级坏味道（未使用变量、可疑比较）无人拦截。

退出码：0 通过；1 阻断；2 依赖缺失。
"""
from __future__ import annotations

import ast
import glob
import io
import os
import subprocess
import sys
import trace

COVERAGE_FLOOR = 0.70
# 平台层同样纳入管辖：platform/README 明写「每个组件必须有 schema 声明与冒烟测试」，
# 若只管 modules/，跨模块共享代码反而绕过了质量门槛，而它的影响面更大。
COMPONENT_GLOBS = ("modules/*/components/*.py", "platform/*/*.py")
TEST_GLOBS = ("modules/*/tests/test_*.py", "platform/tests/test_*.py",
              "tools/tests/test_*.py")
# tools/ 是全部 notebook 的唯一生成入口，2026-09-04 的审视发现它此前不在管辖内。
TEST_ROOTS = ("modules/", "platform/", "tools/")
DECLARATION_DIR = "registry/component_declarations"
LINT_DIRS = ("modules", "platform", "ci", "tools")
GLOSSARY = "registry/glossary.yaml"
TERM_WAIVER_MARK = "术语豁免:"
QUOTE_PAIRS = (("「", "」"), ("\u201c", "\u201d"), ('"', '"'), ("'", "'"))
DOC_GLOBS = ("*.md", "modules/*/*.md", "docs/**/*.md")
BLOCKS = []


def block(tag, msg):
    BLOCKS.append((tag, msg))
    print("❌ BLOCK  [%s] %s" % (tag, msg))


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def is_test_path(path):
    """测试文件不是组件——它们不需要组件声明，也不需要「自己的冒烟测试」。"""
    norm = os.sep + os.path.normpath(path).strip(os.sep)
    return (os.sep + "tests" + os.sep) in norm + os.sep


def source_files():
    out = []
    for d in LINT_DIRS:
        out += glob.glob(os.path.join(d, "**", "*.py"), recursive=True)
    return sorted(out)


# ————————————————————————— Q1 / Q2 —————————————————————————

def check_lint(files):
    code, out = run([sys.executable, "-m", "ruff", "check", "--output-format=concise"] + files)
    if code == 0:
        print("✅ Q1 lint：%d 个源文件 0 error" % len(files))
        return
    for line in out.splitlines():
        text = line.strip()
        # ruff 的汇总行（"Found N errors."／"[*] N fixable..."）不是错误本身
        if not text or text.startswith("Found") or text.startswith("[*]"):
            continue
        block("Q1", "lint：%s" % text)


def check_types():
    code, out = run([sys.executable, "-m", "mypy", "--ignore-missing-imports",
                     "--no-strict-optional", "--namespace-packages",
                     "--explicit-package-bases", "modules/", "platform/"])
    if code == 0:
        print("✅ Q2 类型检查：0 error")
        return
    for line in out.splitlines():
        if ": error:" in line:
            block("Q2", "类型检查：%s" % line.strip())


# ————————————————————————— Q3 —————————————————————————

def executable_lines(path):
    """可执行行 = AST 语句节点所在行；docstring 不计。"""
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt):
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                continue
            lines.add(node.lineno)
    return lines


def check_tests_and_coverage(components):
    import pytest

    targets = {os.path.abspath(p): p for p in components}
    # ⚠️ 不传 ignoredirs：stdlib trace 的 _Ignore 按**模块名**缓存判定，
    #    site-packages 里若有同名模块（如 models），我们的同名文件会被连坐漏计。
    tracer = trace.Trace(count=1, trace=0)
    roots = [r for r in TEST_ROOTS if os.path.isdir(r)]   # 只传存在的路径，否则 pytest 直接报错
    rc = tracer.runfunc(pytest.main, roots + ["-q", "--no-header",
                                              "-p", "no:cacheprovider"])
    if rc != 0:
        block("Q3", "单元测试未全部通过（pytest 退出码 %s）——通过率必须 100%%" % rc)
        return

    hit = {}
    for (fname, lineno), _ in tracer.results().counts.items():
        a = os.path.abspath(fname)
        if a in targets:
            hit.setdefault(targets[a], set()).add(lineno)

    total_e = total_h = 0
    for path in components:
        exe = executable_lines(path)
        cov = hit.get(path, set()) & exe
        total_e += len(exe)
        total_h += len(cov)
        pct = len(cov) / len(exe) if exe else 1.0
        if pct < COVERAGE_FLOOR:
            block("Q3", "%s 行覆盖 %.1f%% < %.0f%%——未被测试覆盖的逻辑等于未验证"
                  % (path, pct * 100, COVERAGE_FLOOR * 100))
    if total_e:
        print("✅ Q3 单元测试全部通过；核心逻辑行覆盖 %d/%d = %.1f%%"
              % (total_h, total_e, total_h / total_e * 100))


# ————————————————————————— Q4 / Q5 —————————————————————————

def load_declarations():
    import yaml
    decls = {}
    for path in sorted(glob.glob(os.path.join(DECLARATION_DIR, "*.yaml"))):
        try:
            d = yaml.safe_load(io.open(path, encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            block("Q4", "%s YAML 无法解析：%s" % (path, exc))
            continue
        if not d.get("path"):
            block("Q4", "%s 缺 path 字段——声明必须指明它描述的是哪个文件" % path)
            continue
        decls[str(d["path"])] = (path, d)
    return decls


def check_declarations(components, decls):
    missing = [p for p in components if p not in decls]
    for p in missing:
        block("Q4", "%s 没有组件声明——新增组件必须 100%% 有声明（门禁 5）" % p)
    for path, (dpath, d) in decls.items():
        if not os.path.exists(path):
            block("Q4", "%s 声明的 path=%s 不存在" % (dpath, path))
        for field in ("component", "module", "purpose", "cross_border_assets"):
            if field not in d:
                block("Q4", "%s 缺字段 %s" % (dpath, field))
    if not missing:
        print("✅ Q4 组件声明：%d 个组件全部有声明" % len(components))


def check_smoke(components, decls):
    before = len(BLOCKS)
    test_src = ""
    for t in [f for g in TEST_GLOBS for f in glob.glob(g)]:
        test_src += io.open(t, encoding="utf-8").read()
    for path in components:
        mod = os.path.splitext(os.path.basename(path))[0]
        declared = decls.get(path, (None, {}))[1].get("tests") or []
        referenced = mod in test_src
        if not (declared or referenced):
            block("Q5", "%s 没有任何冒烟测试——新增组件至少要有 1 个" % path)
        for t in declared:
            if not os.path.exists(t):
                block("Q5", "%s 声明的测试 %s 不存在" % (path, t))
    if len(BLOCKS) == before:
        print("✅ Q5 组件冒烟测试：%d 个组件各有 ≥1 个测试" % len(components))


# ————————————————————————— Q6 —————————————————————————

def _is_mention_not_use(line, word, correct):
    """判断禁用词是**被提及**而非**被使用**。两条跳过规则，各有代价：

    S1 同行出现了正确术语 —— 覆盖「改写规则本身」与「首次释义」两类：
       例「PIPL 写『去标识化』不写『假名化』」、「主动方（有标签方）」。
       **代价**：同一行里既正确用词又错误用词的混写不会被拦。
    S2 禁用词被引号包裹 —— 覆盖「引述他人表述并加以批评」：
       例 「不亲手跑一次梯度反演，就不会真的相信"数据不出域 ≠ 安全"」。
       **代价**：把误用套上引号即可绕过本检查。

    另有显式逃生口 `术语豁免: <理由>`，用于前两条覆盖不到的情形。
    """
    if TERM_WAIVER_MARK in line:
        return True
    if correct and correct in line:
        return True                                  # S1
    for left, right in QUOTE_PAIRS:
        i = line.find(left)
        while i != -1:
            j = line.find(right, i + len(left))
            if j == -1:
                break
            if word in line[i + len(left):j]:
                return True                          # S2
            i = line.find(left, j + len(right))
    return False


def check_terminology():
    import yaml
    if not os.path.exists(GLOSSARY):
        block("Q6", "%s 不存在——术语表是全项目唯一术语来源" % GLOSSARY)
        return
    terms = (yaml.safe_load(io.open(GLOSSARY, encoding="utf-8")) or {}).get("terms") or []
    banned = {}
    for t in terms:
        for word in t.get("forbidden") or []:
            banned[str(word)] = str(t.get("term", "?"))
    if not banned:
        print("✅ Q6 术语规范：词表未定义禁用词，无核验对象")
        return

    docs = []
    for pattern in DOC_GLOBS:
        docs += glob.glob(pattern, recursive=True)
    hits = 0
    for path in sorted(set(docs)):
        if os.path.abspath(path) == os.path.abspath(GLOSSARY):
            continue
        text = io.open(path, encoding="utf-8", errors="ignore").read()
        for line_no, line in enumerate(text.splitlines(), 1):
            for word, correct in banned.items():
                if word in line and not _is_mention_not_use(line, word, correct):
                    hits += 1
                    block("Q6", "%s:%d 使用禁用表述「%s」，应为「%s」"
                          % (path, line_no, word, correct))
    if hits == 0:
        print("✅ Q6 术语规范：%d 个禁用词在 %d 份文档中 0 命中" % (len(banned), len(docs)))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    print("== 门禁 5 · 代码质量 ==")
    try:
        import yaml  # noqa: F401
        import pytest  # noqa: F401
    except ImportError as exc:
        print("依赖缺失：%s（需要 PyYAML 与 pytest）" % exc)
        return 2

    components = sorted(c for g in COMPONENT_GLOBS for c in glob.glob(g)
                        if not is_test_path(c))
    files = source_files()
    print("源文件：%d ｜ 组件：%d" % (len(files), len(components)))
    print()

    check_lint(files)
    check_types()
    check_tests_and_coverage(components)
    decls = load_declarations()
    check_declarations(components, decls)
    check_smoke(components, decls)
    check_terminology()

    print("\n== 结果：BLOCK=%d ==" % len(BLOCKS))
    if BLOCKS:
        print("门禁 5 未通过：代码质量不达标（第 8 部分 · 门禁 5）。")
        return 1
    print("门禁 5 通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
