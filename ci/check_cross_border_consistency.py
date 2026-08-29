#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门禁 4 · 合规一致性核验（《项目维护约束 v2》7.2 六项）

原则三的载体：说的和做的必须可机器核验。
每个涉及跨境信息流的组件必须声明它触碰了哪些跨境资产，本脚本自动比对 M0 清单，不一致即阻断合并。

六项核验：
  1 引用完整性  每个 m0_registry_ref 在 cross_border_assets.yaml 中存在且未作废
  2 定性一致性  声明的 asset 类型/定性与 M0 清单该条目一致
  3 路径合法性  legal_basis_ref 指向 M0 选定的主线或回退线，不得引用被否决路线
  4 无未声明流动  代码有跨方通信调用但组件无对应声明 → 阻断
  5 审计钩子存在  声明了跨境资产但未挂审计钩子 → 阻断
  6 同意依赖闭环  requires_batch 在同意管理配置中存在

用法：python3 ci/check_cross_border_consistency.py [仓库根]
退出码：0 通过；1 阻断；2 环境缺失（PyYAML）
"""
import os
import re
import sys
import glob

try:
    import yaml
except ImportError:  # pragma: no cover
    print("需要 PyYAML：pip install pyyaml")
    sys.exit(2)

REGISTRY = "registry/cross_border_assets.yaml"
ROUTES = "registry/compliance_routes.yaml"          # 契约：M0 的 S0.9 判决主线/回退线后产出
CONSENT = "platform/governance/consent_batches.yaml"  # 契约：同意管理配置（M8 落地）
DECL_GLOBS = [
    "modules/*/components/*/declaration.yaml",
    "modules/*/declarations.yaml",
    "platform/components/*/declaration.yaml",
    "registry/component_declarations/*.yaml",
    "party_cn/*/declaration.yaml",
    "party_hk/*/declaration.yaml",
]
CODE_ROOTS = ["modules", "platform", "party_cn", "party_hk"]
# 跨方通信的代码特征。宁可多报（需在声明里说明），不可漏报。
COMM_PATTERNS = [
    r"\bsocket\.(socket|create_connection)\b",
    r"\bgrpc\b",
    r"\brequests\.(get|post|put)\b",
    r"\bhttpx\.(get|post|AsyncClient)\b",
    r"\burllib\.request\b",
    r"\b(send|recv)_(to|from)_party\b",
    r"\bparty_(a|b|cn|hk)_(client|channel|endpoint|stub)\b",
    r"\bcross_border_(send|recv|call)\b",
]

BLOCKS = []
WARNS = []


def block(check, msg):
    BLOCKS.append("[核验 %s] %s" % (check, msg))


def warn(check, msg):
    WARNS.append("[核验 %s] %s" % (check, msg))


def load_yaml(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        try:
            return yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            block("0", "%s 无法解析：%s" % (path, exc))
            return {}


def collect_declarations():
    found = []
    for pattern in DECL_GLOBS:
        for path in sorted(glob.glob(pattern)):
            data = load_yaml(path)
            if isinstance(data, dict) and data:
                found.append((path, data))
    return found


def asset_index(registry):
    index = {}
    for item in (registry or {}).get("assets") or []:
        if isinstance(item, dict) and item.get("id"):
            index[str(item["id"])] = item
    return index


def route_status(routes):
    """返回 {route_id: status}。status 取 primary | fallback | rejected | 其他。"""
    table = {}
    for item in (routes or {}).get("routes") or []:
        if isinstance(item, dict) and item.get("id"):
            table[str(item["id"])] = str(item.get("status", "undecided"))
    return table


def declared_assets(decl):
    items = decl.get("cross_border_assets")
    return items if isinstance(items, list) else []


def check_1_2_3_5_6(declarations, assets, routes, routes_exists, consent, consent_exists):
    for path, decl in declarations:
        name = decl.get("component", path)
        items = declared_assets(decl)

        # ---- 核验 5：声明了跨境资产必须挂审计钩子 ----
        hooks = decl.get("audit_hooks")
        if items and not (isinstance(hooks, list) and hooks):
            block(5, "%s（%s）声明了 %d 项跨境资产但没有 audit_hooks" % (name, path, len(items)))

        for item in items:
            if not isinstance(item, dict):
                block(1, "%s（%s）的 cross_border_assets 条目不是映射：%r" % (name, path, item))
                continue
            label = item.get("asset", "<未命名资产>")
            ref = item.get("m0_registry_ref")

            # ---- 核验 1：引用完整性 ----
            if not ref:
                block(1, "%s · %s 缺 m0_registry_ref" % (name, label))
                continue
            ref = str(ref)
            entry = assets.get(ref)
            if entry is None:
                block(1, "%s · %s 的 m0_registry_ref=%s 在 %s 中不存在"
                      % (name, label, ref, REGISTRY))
                continue
            if str(entry.get("status", "active")) != "active":
                block(1, "%s · %s 引用了已作废条目 %s（status=%s）"
                      % (name, label, ref, entry.get("status")))

            # ---- 核验 2：定性一致性 ----
            for field in ("category", "is_personal_information"):
                if field in item and field in entry:
                    if str(item[field]) != str(entry[field]):
                        block(2, "%s · %s 的 %s=%r 与 M0 清单 %s 的 %r 不一致"
                              % (name, label, field, item[field], ref, entry[field]))

            # ---- 核验 3：路径合法性 ----
            basis = item.get("legal_basis_ref") or decl.get("legal_basis_ref")
            if basis:
                basis = str(basis)
                if not routes_exists:
                    block(3, "%s · %s 引用了 legal_basis_ref=%s，但 %s 不存在——"
                             "M0 的 S0.9 尚未判决主线/回退线，无法核验合规路径"
                          % (name, label, basis, ROUTES))
                else:
                    status = routes.get(basis)
                    if status is None:
                        block(3, "%s · %s 的 legal_basis_ref=%s 不在 %s 中"
                              % (name, label, basis, ROUTES))
                    elif status not in ("primary", "fallback"):
                        block(3, "%s · %s 引用了非选定路线 %s（status=%s）——"
                                 "只能引用主线或回退线" % (name, label, basis, status))
                    routes_ok = entry.get("applicable_routes")
                    if isinstance(routes_ok, list) and basis not in [str(x) for x in routes_ok]:
                        block(3, "%s · %s 的 legal_basis_ref=%s 不在 M0 清单 %s 的 applicable_routes 内"
                              % (name, label, basis, ref))

        # ---- 核验 6：同意依赖闭环 ----
        dependency = decl.get("consent_dependency") or {}
        batch = dependency.get("requires_batch") if isinstance(dependency, dict) else None
        if batch:
            batch = str(batch)
            if not consent_exists:
                block(6, "%s 依赖同意批次 %s，但 %s 不存在，无法核验闭环" % (name, batch, CONSENT))
            else:
                known = [str(b.get("id")) for b in (consent.get("batches") or [])
                         if isinstance(b, dict)]
                if batch not in known:
                    block(6, "%s 依赖的同意批次 %s 未在 %s 中登记" % (name, batch, CONSENT))


def check_4_undeclared_flows(declarations):
    """代码有跨方通信调用但所在组件无声明 → 阻断。"""
    declared_dirs = {os.path.dirname(p) for p, _ in declarations}
    for root in CODE_ROOTS:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fname)
                try:
                    text = open(path, encoding="utf-8").read()
                except (OSError, UnicodeDecodeError):
                    continue
                hits = [p for p in COMM_PATTERNS if re.search(p, text)]
                if not hits:
                    continue
                # 声明必须在该文件所在目录或其祖先目录（组件根 / 模块根），
                # 不接受"子目录里有别的声明"这种反向覆盖
                covered = any(dirpath == d or dirpath.startswith(d + os.sep)
                              for d in declared_dirs)
                if not covered:
                    block(4, "%s 含跨方通信特征 %s，但所在组件没有 declaration.yaml"
                          % (path, hits))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    print("== 门禁 4 · 合规一致性核验 ==")
    print("扫描根目录：%s" % root)

    registry = load_yaml(REGISTRY)
    if registry is None:
        block(1, "%s 不存在——它是 M0 唯一权威清单，缺失即无法核验任何跨境声明" % REGISTRY)
        registry = {}
    assets = asset_index(registry)

    routes_raw = load_yaml(ROUTES)
    routes_exists = routes_raw is not None
    routes = route_status(routes_raw)

    consent_raw = load_yaml(CONSENT)
    consent_exists = consent_raw is not None
    consent = consent_raw or {}

    declarations = collect_declarations()
    print("M0 清单条目：%d ｜ 组件声明：%d ｜ 路线判决：%s ｜ 同意配置：%s"
          % (len(assets), len(declarations),
             "已产出" if routes_exists else "未产出（M0/S0.9）",
             "已产出" if consent_exists else "未产出（M8）"))

    check_1_2_3_5_6(declarations, assets, routes, routes_exists, consent, consent_exists)
    check_4_undeclared_flows(declarations)

    if not declarations and not assets:
        print("当前无组件声明且 M0 清单为空模板：六项核验无对象，通过。")

    print()
    for line in WARNS:
        print("⚠️  WARN   %s" % line)
    for line in BLOCKS:
        print("❌ BLOCK  %s" % line)
    print()
    print("== 结果：BLOCK=%d  WARN=%d ==" % (len(BLOCKS), len(WARNS)))
    if BLOCKS:
        print("门禁 4 未通过。禁止通过修改文档消除不一致——必须回到实现或 M0 清单修正（禁止事项第 16 条）。")
        return 1
    print("门禁 4 通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
