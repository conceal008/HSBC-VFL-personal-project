# -*- coding: utf-8 -*-
"""M9 证据链核验（非编号门禁，随 run_all_gates.sh 一并执行）。

框架的七道门禁管的是「提交本身合不合规矩」；本检查管的是另一件事：
**已发布的结论是否还追得到证据。**

它防的是一类很安静的事故：有人删掉或重命名了一个结果文件，
文档里引用它的那条结论就此失去依据，而所有门禁照样全绿。

三项核验，任一不足 100% 即阻断：
  一致性核验   每条结论的证据文件都存在且非空，且性质已显式声明
  风险溯源     每条 DPIA 风险都指向具体实验产出，且有缓解措施
  交付清单     每项交付物都真实存在

退出码：0 通过；1 阻断；2 依赖缺失。
"""
from __future__ import annotations

import os
import sys

EVIDENCE_MAP = "modules/m9_documentation/configs/evidence_map.yaml"
FULL_RATE = 1.0
PERCENT = 100


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    print("== M9 · 证据链核验 ==")
    try:
        import yaml
    except ImportError:
        print("依赖缺失：PyYAML")
        return 2
    if not os.path.exists(EVIDENCE_MAP):
        print("❌ BLOCK  %s 不存在——没有映射就无从核验结论的可追溯性" % EVIDENCE_MAP)
        return 1

    sys.path.insert(0, root)
    from modules.m9_documentation.components import evidence_chain as ec

    data = yaml.safe_load(open(EVIDENCE_MAP, encoding="utf-8"))
    claims = ec.verify_claims(data.get("claims") or [])
    risks = ec.verify_risk_traceability(data.get("risks") or [])
    deliverables = ec.verify_deliverables(data.get("deliverables") or [])

    blocked = 0
    for label, res, ok_key, total_key, bad_key, hint in [
        ("一致性核验（结论可追溯）", claims, "traceable", "total", "untraceable_ids",
         "证据文件缺失或性质未声明的结论**不成立**，不得对外引用"),
        ("DPIA 风险溯源", risks, "traced", "total", "untraced_ids",
         "风险必须指向具体实验产出并给出缓解措施，不能只写「可能存在」"),
        ("交付清单", deliverables, "present", "total", "missing",
         "交付清单里列了但文件不在，等于没交付"),
    ]:
        rate = res["rate"]
        mark = "✅" if rate >= FULL_RATE else "❌"
        print("%s %s：%d/%d = %.1f%%"
              % (mark, label, res[ok_key], res[total_key], rate * PERCENT))
        if rate < FULL_RATE:
            blocked += 1
            print("   问题项：%s" % "、".join(str(x) for x in res[bad_key]))
            print("   %s" % hint)
            for row in res.get("rows", []):
                for problem in row.get("problems") or []:
                    print("     - %s：%s" % (problem["path"], problem["status"]))

    print()
    if blocked:
        print("证据链核验未通过：%d 项不足 100%%。" % blocked)
        return 1
    print("证据链核验通过：结论、风险、交付三项均 100%。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
