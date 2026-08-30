#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门禁 6 · 可复现性（《项目维护约束 v2》第 9 部分 + 9.1 Notebook 交付规范）

六项检查：
  R1 硬编码魔数 = 0        代码与 notebook 中不得出现未命名的数字字面量
  R2 种子来自配置          禁止 seed=<数字> / random_state=<数字> / np.random.seed(<数字>)
  R3 环境锁一致            requirements 中的包必须出现在 lock 文件中
  R4 notebook 命名规范      N1：<步骤号>_<简述>.ipynb
  R5 notebook 运行指纹      N4：首个代码 cell 必须打印 config / seed / git 版本
  R6 notebook 执行顺序      N5：execution_count 自上而下单调递增

用法：python3 ci/check_reproducibility.py [仓库根]
退出码：0 通过；1 阻断；2 环境缺失
"""
import ast
import io
import os
import re
import sys
import json
import glob

try:
    import yaml
except ImportError:  # pragma: no cover
    print("需要 PyYAML：pip install pyyaml")
    sys.exit(2)

CODE_ROOTS = ["modules", "platform", "party_cn", "party_hk"]
NOTEBOOK_GLOB = "modules/*/notebooks/*.ipynb"
CONFIG_GLOB = "modules/*/configs/*.yaml"
NOTEBOOK_NAME_RE = re.compile(r"^S[-\w.]+_[\w\-]+\.ipynb$")
FINGERPRINT_KEYS = ("config", "seed", "git")
# 不算魔数：布尔式取值、单位元、常见轴与维度
ALLOWED_NUMBERS = {0, 1, -1, 2, 0.0, 1.0, -1.0}
EXEMPT_COMMENT = "魔数豁免"
SEED_CALL_RE = re.compile(
    r"(?:\b(?:seed|random_state|random_seed)\s*=\s*(-?\d+(?:\.\d+)?)"
    r"|\b(?:np\.random\.seed|random\.seed|torch\.manual_seed|tf\.random\.set_seed)\s*\(\s*(-?\d+)\s*\))")
MAX_NOTEBOOK_CODE_LINES = 150
MAGIC_LINE_RE = re.compile(r"^(?:%{1,2}[A-Za-z]|[!?])")

BLOCKS = []
WARNS = []


def block(rule, msg):
    BLOCKS.append("[%s] %s" % (rule, msg))


def warn(rule, msg):
    WARNS.append("[%s] %s" % (rule, msg))


def exempt_lines(source):
    """带「# 魔数豁免: 理由」注释的行号集合。"""
    out = set()
    for i, line in enumerate(source.splitlines(), 1):
        if EXEMPT_COMMENT in line and "#" in line:
            out.add(i)
    return out


def scan_magic_numbers(source, origin, line_offset=0):
    """R1：AST 扫描数字字面量。命名常量（模块级全大写赋值）不算魔数。"""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        warn("R1", "%s 无法解析为 Python，跳过魔数扫描：%s" % (origin, exc))
        return
    skip = exempt_lines(source)

    named_const_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if names and all(n.isupper() for n in names) and node.value is not None:
                for sub in ast.walk(node.value):
                    named_const_nodes.add(id(sub))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
            continue
        if not isinstance(node.value, (int, float)):
            continue
        if node.value in ALLOWED_NUMBERS or id(node) in named_const_nodes:
            continue
        line = getattr(node, "lineno", 0)
        if line in skip:
            continue
        block("R1", "%s:%d 硬编码魔数 %r —— 参数必须来自 configs/，"
                    "或定义为模块级全大写常量，或加 `# %s: <理由>`"
              % (origin, line + line_offset, node.value, EXEMPT_COMMENT))


def scan_seeds(source, origin):
    """R2：种子必须来自配置，不得写死在代码或 notebook 里。"""
    for i, line in enumerate(source.splitlines(), 1):
        if EXEMPT_COMMENT in line:
            continue
        m = SEED_CALL_RE.search(line)
        if m:
            value = m.group(1) or m.group(2)
            block("R2", "%s:%d 种子写死为 %s —— 种子必须在配置中声明后读入"
                        "（第 9 部分第 2 条）" % (origin, i, value))


def iter_code_files():
    for root in CODE_ROOTS:
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if name.endswith(".py"):
                    yield os.path.join(dirpath, name)


def notebook_code_cells(nb):
    return [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]


def cell_source(cell):
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    # 去掉 IPython 魔法与 shell 命令，否则 ast 解析失败。
    # 只剥**行首无缩进**且形如 %name / %%name / !cmd / ?obj 的行——
    # 字符串格式化的续行（如 `      % (a, b)`）有缩进且 % 后不是字母，不能剥，
    # 否则整个 cell 语法被破坏、魔数扫描被静默跳过。
    return "\n".join(l for l in src.splitlines()
                     if not MAGIC_LINE_RE.match(l))


def check_notebook(path):
    name = os.path.basename(path)
    if not NOTEBOOK_NAME_RE.match(name):
        block("R4", "%s 命名不符合 N1：<步骤号>_<简述>.ipynb（例 S5.4_l0_vs_l1.ipynb）" % path)

    try:
        nb = json.load(io.open(path, encoding="utf-8"))
    except (ValueError, OSError) as exc:
        block("R4", "%s 不是合法的 notebook JSON：%s" % (path, exc))
        return

    cells = notebook_code_cells(nb)
    if not cells:
        block("R5", "%s 没有代码 cell" % path)
        return

    first = cell_source(cells[0]).lower()
    missing = [k for k in FINGERPRINT_KEYS if k not in first]
    if missing:
        block("R5", "%s 首个代码 cell 未打印运行指纹（N4），缺 %s —— "
                    "须输出 config 路径与 hash、seeds、git 版本" % (path, "/".join(missing)))

    counts = [c.get("execution_count") for c in cells]
    seen = [c for c in counts if isinstance(c, int)]
    if not seen:
        warn("R6", "%s 全部 cell 无 execution_count：若因 L-受限已清除输出，"
                   "须在 changelog 的 sensitive_review 记录原因（9.1）" % path)
    else:
        if seen != sorted(seen):
            block("R6", "%s 的 execution_count 非单调递增 %s —— "
                        "乱序执行才跑通的 notebook 不是可复现产物（N5）" % (path, seen))
        if len(seen) != len(cells):
            block("R6", "%s 有 %d 个代码 cell 未执行 —— 提交前须 Restart & Run All"
                  % (path, len(cells) - len(seen)))
        if not any(c.get("outputs") for c in cells):
            warn("R6", "%s 已执行但无任何输出：若因 L-受限已清除，"
                       "须在 changelog 的 sensitive_review 记录原因" % path)

    total = 0
    for idx, cell in enumerate(cells):
        src = cell_source(cell)
        total += len([l for l in src.splitlines() if l.strip()])
        scan_magic_numbers(src, "%s[cell %d]" % (path, idx + 1))
        scan_seeds(src, "%s[cell %d]" % (path, idx + 1))
    if total > MAX_NOTEBOOK_CODE_LINES:
        warn("R1", "%s 代码行数 %d > %d —— 疑似把逻辑写进了 notebook，"
                   "按 N2 应下沉到 components/" % (path, total, MAX_NOTEBOOK_CODE_LINES))


def check_env_lock():
    """R3：requirements 中的包必须出现在 lock 文件里。"""
    reqs = sorted(glob.glob("requirements*.txt") + glob.glob("environments/requirements*.txt"))
    locks = sorted(glob.glob("environments/*.lock") + glob.glob("*.lock"))
    if not reqs:
        return
    if not locks:
        block("R3", "存在 %s 但没有任何 lock 文件 —— 依赖版本未锁定，环境不可重建" % reqs[0])
        return
    locked = ""
    for lock in locks:
        locked += io.open(lock, encoding="utf-8").read().lower()
    for req in reqs:
        for line in io.open(req, encoding="utf-8"):
            pkg = re.split(r"[=<>!~\[; ]", line.strip(), 1)[0].strip().lower()
            if not pkg or pkg.startswith("#"):
                continue
            if pkg not in locked:
                block("R3", "%s 声明的 %s 未出现在 lock 文件中" % (req, pkg))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    print("== 门禁 6 · 可复现性 ==")

    code_files = sorted(iter_code_files())
    notebooks = sorted(glob.glob(NOTEBOOK_GLOB))
    configs = sorted(glob.glob(CONFIG_GLOB))
    print("代码文件：%d ｜ notebook：%d ｜ 配置：%d" % (len(code_files), len(notebooks), len(configs)))

    for path in code_files:
        source = io.open(path, encoding="utf-8").read()
        scan_magic_numbers(source, path)
        scan_seeds(source, path)

    for path in notebooks:
        check_notebook(path)

    check_env_lock()

    declared = False
    for path in configs:
        try:
            data = yaml.safe_load(io.open(path, encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            block("R2", "%s 无法解析：%s" % (path, exc))
            continue
        if isinstance(data, dict) and any(k in data for k in ("seed", "seeds", "random_state")):
            declared = True
    if configs and not declared:
        warn("R2", "modules/*/configs/ 下没有任何配置声明 seed/seeds —— "
                   "实验类步骤必须在配置中声明种子（第 9 部分第 2 条）")

    if not code_files and not notebooks:
        print("尚无代码与 notebook，六项检查无对象，通过。")

    print()
    for line in WARNS:
        print("⚠️  WARN   %s" % line)
    for line in BLOCKS:
        print("❌ BLOCK  %s" % line)
    print()
    print("== 结果：BLOCK=%d  WARN=%d ==" % (len(BLOCKS), len(WARNS)))
    if BLOCKS:
        print("门禁 6 未通过：可复现性不达标（第 9 部分）。")
        return 1
    print("门禁 6 通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
