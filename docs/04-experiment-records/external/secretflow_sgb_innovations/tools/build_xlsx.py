"""防护代价 / 攻击效果 → xlsx（数据来自 defense_utility.json 与 attack_*.json）。"""
import json
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

A = Path(sys.argv[1])          # data/processed/attack
OUT = Path(sys.argv[2])

BLUE = Font(name="Arial", color="0000FF")
BLACK = Font(name="Arial")
BOLD = Font(name="Arial", bold=True)
TITLE = Font(name="Arial", bold=True, size=13)
NOTE = Font(name="Arial", italic=True, color="595959", size=9)
HEAD_FILL = PatternFill("solid", fgColor="E7ECF2")
thin = Side(style="thin", color="BFBFBF")
BOX = Border(top=thin, bottom=thin, left=thin, right=thin)
WRAP = Alignment(wrap_text=True, vertical="center")

util = json.loads((A / "defense_utility.json").read_text(encoding="utf-8"))
wb = Workbook()

# ---------------------------------------------------------------- 防护代价
ws = wb.active
ws.title = "防护代价"
ws["A1"] = "Santander 联邦 SGB：各防护方案的效用代价（测试集）"
ws["A1"].font = TITLE
ws["A2"] = ("测试集 21,735 行 / 645 正例；客户级配对 bootstrap 1000 次（seed 42）。"
            "蓝字 = 实验输出值（defense_utility.json），黑字 = 公式。所有联邦方案均为 OU 加密、100 轮、深度 5。")
ws["A2"].font = NOTE
heads = ["方案", "防护措施", "防护对象", "测试 AUC", "AUC 95% CI 下界", "AUC 95% CI 上界",
         "Δ vs S0（点估计）", "Δ vs S0 CI 下界", "Δ vs S0 CI 上界",
         "联邦增益 vs 内地单边", "增益 CI 下界", "增益 CI 上界", "增益结论"]
HR = 4
for j, h in enumerate(heads, 1):
    c = ws.cell(HR, j, h)
    c.font, c.fill, c.border, c.alignment = BOLD, HEAD_FILL, BOX, WRAP

rows = [
    ("内地单边", "不做联邦：内地只用自有 20 列（XGB 100 轮 / 深度 5 / lr 0.3 / λ 0.1）", "—", "guest_default"),
    ("S0", "无防护（隐语默认：首树只用标签方特征）", "—", "S0"),
    ("S1", "关闭首树保护（首树也用两方特征，香港可见）", "对照组", "S1"),
    ("S5", "香港数据最小化：24 个产品标记 → 1 列持有产品数", "香港特征（A2）", "S5"),
    ("S6", "香港产品标记本地随机响应，每位 ε = 2（翻转概率 11.9%）", "香港特征（A2）", "S6"),
    ("S4", "标签随机响应 ε = 4（翻转概率 1.8%）", "内地标签（A1）", "S4"),
    ("S3", "标签随机响应 ε = 2（翻转概率 11.9%）", "内地标签（A1）", "S3"),
    ("S2", "标签随机响应 ε = 1（翻转概率 26.9%）", "内地标签（A1）", "S2"),
    ("D4", "标签随机响应 ε = 4 + 后验去噪软标签（DP 后处理）", "内地标签（A1）", "D4"),
    ("D2", "标签随机响应 ε = 2 + 后验去噪软标签（DP 后处理）", "内地标签（A1）", "D2"),
    ("D1", "标签随机响应 ε = 1 + 后验去噪软标签（DP 后处理）", "内地标签（A1）", "D1"),
]
r0 = HR + 1
guest_r, s0_r = r0, r0 + 1
for i, (name, desc, target, key) in enumerate(rows):
    r = r0 + i
    u = util[key]
    ws.cell(r, 1, name).font = BOLD
    ws.cell(r, 2, desc).font = BLACK
    ws.cell(r, 3, target).font = BLACK
    for j, v in ((4, u["auroc"]), (5, u["ci_lo"]), (6, u["ci_hi"])):
        ws.cell(r, j, v).font = BLUE
    if key != "guest_default":
        ws.cell(r, 7, f"=D{r}-$D${s0_r}").font = BLACK
        ws.cell(r, 8, u["delta_vs_base"]["ci_lo"]).font = BLUE
        ws.cell(r, 9, u["delta_vs_base"]["ci_hi"]).font = BLUE
        ws.cell(r, 10, f"=D{r}-$D${guest_r}").font = BLACK
        ws.cell(r, 11, u["gain_vs_guest"]["ci_lo"]).font = BLUE
        ws.cell(r, 12, u["gain_vs_guest"]["ci_hi"]).font = BLUE
        ws.cell(r, 13, f'=IF(K{r}>0,"显著为正",IF(L{r}<0,"显著为负","不显著（CI 跨 0）"))').font = BLACK
    else:
        ws.cell(r, 13, "基线").font = BLACK
    for j in range(1, 14):
        cell = ws.cell(r, j)
        cell.border = BOX
        cell.alignment = WRAP
        if 4 <= j <= 12:
            cell.number_format = "0.0000;-0.0000;0.0000" if j in (4, 5, 6) else "+0.0000;-0.0000;0.0000"
last = r0 + len(rows) - 1
ws.cell(last + 2, 1, "增益 = 联邦方案 − 内地单边默认模型；「显著」= 配对 bootstrap 95% CI 不跨 0。").font = NOTE
widths = [10, 46, 16, 11, 12, 12, 13, 12, 12, 14, 12, 12, 16]
for j, w in enumerate(widths, 1):
    ws.column_dimensions[get_column_letter(j)].width = w
ws.row_dimensions[HR].height = 32
ws.freeze_panes = ws.cell(HR + 1, 2)

# ---------------------------------------------------------------- 攻击效果
wa = wb.create_sheet("攻击效果")
wa["A1"] = "各方案在两类模拟攻击下的泄漏量"
wa["A1"].font = TITLE
wa["A2"] = ("A1：香港用叶子成员 + 少量已知真实标签推断内地标签（逻辑回归，5 次随机已知集合均值，在 78,267 个训练样本上评估）。"
            "A2：内地从协议消息还原香港 24 个产品持有标记（与真实未加噪值对照）。蓝字 = 实验输出，黑字 = 公式。")
wa["A2"].font = NOTE
heads2 = ["方案", "说明",
          "A1 只用自有特征 AUC（已知 1%）", "A1 自有 + 叶子 AUC（已知 1%）", "A1 泄漏（已知 1%）",
          "A1 只用自有特征 AUC（已知 5%）", "A1 自有 + 叶子 AUC（已知 5%）", "A1 泄漏（已知 5%）",
          "A2 产品单元格被确定还原 · 训练期", "A2「持有」召回 · 训练期", "A2「持有」精确率 · 训练期",
          "A2 产品单元格被确定还原 · 推理期", "A2「持有」召回 · 推理期", "A2「持有」精确率 · 推理期"]
for j, h in enumerate(heads2, 1):
    c = wa.cell(HR, j, h)
    c.font, c.fill, c.border, c.alignment = BOLD, HEAD_FILL, BOX, WRAP
arows = [
    ("S0", "原版配置，无防护", "S0_base"),
    ("T2", "优化版（134 轮 / 深度 6 / 33 桶），无防护", "T2_opt"),
    ("S1", "关闭首树保护", "S1_ft0"),
    ("S5", "香港只报持有产品数", "S5_hostcnt"),
    ("S6", "香港产品标记随机响应 ε = 2", "S6_hostrr2"),
    ("S4", "标签随机响应 ε = 4", "S4_lrr4"),
    ("S3", "标签随机响应 ε = 2", "S3_lrr2"),
    ("S2", "标签随机响应 ε = 1", "S2_lrr1"),
    ("D4", "标签 RR ε = 4 + 后验去噪", "D4_rrsoft4"),
    ("D2", "标签 RR ε = 2 + 后验去噪", "D2_rrsoft2"),
    ("D1", "标签 RR ε = 1 + 后验去噪", "D1_rrsoft1"),
]
for i, (name, desc, key) in enumerate(arows):
    r = HR + 1 + i
    a = json.loads((A / f"attack_{key}.json").read_text(encoding="utf-8"))
    a1 = a["A1_by_known_frac"]
    wa.cell(r, 1, name).font = BOLD
    wa.cell(r, 2, desc).font = BLACK
    wa.cell(r, 3, a1["0.01"]["own_only"]["mean"]).font = BLUE
    wa.cell(r, 4, a1["0.01"]["own_plus_leaves"]["mean"]).font = BLUE
    wa.cell(r, 5, f"=D{r}-C{r}").font = BLACK
    wa.cell(r, 6, a1["0.05"]["own_only"]["mean"]).font = BLUE
    wa.cell(r, 7, a1["0.05"]["own_plus_leaves"]["mean"]).font = BLUE
    wa.cell(r, 8, f"=G{r}-F{r}").font = BLACK
    for col, phase in ((9, "A2_train"), (12, "A2_infer")):
        t = a[phase]["truth"]
        wa.cell(r, col, t["mean_true_revealed_frac"]).font = BLUE
        wa.cell(r, col + 1, t["holder_recall_pooled"]).font = BLUE
        p = t["holder_precision_pooled"]
        c = wa.cell(r, col + 2, p if p is not None else "—（无人被识别为持有者）")
        c.font = BLUE
    for j in range(1, 15):
        cell = wa.cell(r, j)
        cell.border = BOX
        cell.alignment = WRAP
        if j in (3, 4, 6, 7):
            cell.number_format = "0.0000"
        elif j in (5, 8):
            cell.number_format = "+0.0000;-0.0000;0.0000"
        elif j >= 9:
            cell.number_format = "0.0%"
widths2 = [8, 34, 13, 13, 11, 13, 13, 11, 14, 12, 12, 14, 12, 12]
for j, w in enumerate(widths2, 1):
    wa.column_dimensions[get_column_letter(j)].width = w
wa.row_dimensions[HR].height = 46
wa.freeze_panes = wa.cell(HR + 1, 3)

# ---------------------------------------------------------------- 协议层与可见树
wp = wb.create_sheet("协议层成本")
wp["A1"] = "协议层防护的成本（2 核单机仿真，不含跨境网络延迟）"
wp["A1"].font = TITLE
ss = json.loads((A / "ss_xgb_cost.json").read_text(encoding="utf-8"))
full = json.loads((A / "ss_xgb_full3.json").read_text(encoding="utf-8"))["runs"][0]
si0 = json.loads((A / "secure_infer.json").read_text(encoding="utf-8"))
si2 = json.loads((A / "secure_infer_T2.json").read_text(encoding="utf-8"))
hd = ["方案", "隐藏样本空间", "规模", "耗时（s）", "AUC", "备注"]
for j, h in enumerate(hd, 1):
    c = wp.cell(3, j, h); c.font, c.fill, c.border, c.alignment = BOLD, HEAD_FILL, BOX, WRAP
prow = [("SGB + Paillier（原版）训练", "否", "100 棵树", 5637.75, 0.8991, "business_sgb_federated_r100_m42.json"),
        ("SGB + OU（优化）训练", "否", "100 棵树", 963.26, 0.8991, "预测与 Paillier 逐位相同"),
        *[(f"SS-XGB（SEMI2K）训练", "是", f"{r['rows']:,} 行 × {r['trees']} 棵", r["fit_seconds"], r["test_auc"],
           "稳态每棵 %.1f s（首棵含编译）" % r["per_tree_seconds"]) for r in ss["runs"]],
        ("SS-XGB（SEMI2K）训练", "是", "78,267 行 × 3 棵", full["fit_seconds"], full["test_auc"],
         "稳态每棵 %.1f s；同配置明文 3 棵 0.8674；100 棵外推约 %.1f h" % (full["per_tree_seconds"], full["per_tree_seconds"] * 100 / 3600)),
        ("SGB 训练 + SPU 安全推理（原版模型）", "推理期是", "100 棵 × 21,735 客户", si0["seconds"], si0["auc"],
         "与原生推理最大偏差 %.1e；原生推理 1.2 s" % si0["max_abs_diff_vs_sgb"]),
        ("SGB 训练 + SPU 安全推理（优化版模型）", "推理期是", "134 棵 × 21,735 客户", si2["seconds"], si2["auc"],
         "与原生推理最大偏差 %.1e；原生推理 2.7 s；每批 3 棵树（10 棵会内存不足）" % si2["max_abs_diff_vs_sgb"])]
for i, row in enumerate(prow, 4):
    for j, v in enumerate(row, 1):
        c = wp.cell(i, j, v); c.font = BLUE if j in (4, 5) else BLACK; c.border = BOX; c.alignment = WRAP
        if j == 4: c.number_format = "#,##0.0"
        if j == 5: c.number_format = "0.0000"
for j, w in enumerate([34, 12, 22, 12, 9, 60], 1):
    wp.column_dimensions[get_column_letter(j)].width = w
vt = json.loads((A / "a1_vs_visible_trees.json").read_text(encoding="utf-8"))
r0v = 4 + len(prow) + 2
wp.cell(r0v, 1, "混合树模拟：香港可见树数量 vs 标签泄漏（S0 模型，已知 5% 标签）").font = BOLD
for j, h in enumerate(["可见树", "棵数", "A1 泄漏"], 1):
    c = wp.cell(r0v + 1, j, h); c.font, c.fill, c.border = BOLD, HEAD_FILL, BOX
for i, (k, v) in enumerate(vt.items(), r0v + 2):
    wp.cell(i, 1, k).border = BOX
    c = wp.cell(i, 2, v["trees"]); c.font = BLUE; c.border = BOX
    c = wp.cell(i, 3, v["leak_5pct"]); c.font = BLUE; c.border = BOX; c.number_format = "+0.0000;-0.0000"

# ---------------------------------------------------------------- HSBC mock
wm = wb.create_sheet("HSBC mock")
wm["A1"] = "HSBC mock 四档复核（默认配置；balanced 另含优化版）"
wm["A1"].font = TITLE
wm["A2"] = "原口径增益 = 联邦 − max(内地默认单边, 香港默认单边)，公式计算；CI 来自配对 bootstrap（蓝字）。团队公布值来自 Mock数据说明.md。"
wm["A2"].font = NOTE
PD = A.parent
hm = ["档位", "内地默认单边", "香港默认单边", "联邦（默认）", "原口径增益", "增益 CI 下界", "增益 CI 上界", "团队公布增益",
      "调参后最佳单边", "联邦（优化版）", "优化版 − 默认", "A1 泄漏（5% 已知）", "A2 推理期持有识别"]
for j, h in enumerate(hm, 1):
    c = wm.cell(4, j, h); c.font, c.fill, c.border, c.alignment = BOLD, HEAD_FILL, BOX, WRAP
published = {"balanced": 0.0672, "overlap80": 0.0332, "overlap50": 0.0337, "overlap20": 0.0323}
for i, t in enumerate(["balanced", "overlap80", "overlap50", "overlap20"], 5):
    tl = json.loads((PD / f"train_local_xgb_all_mock_{t}_m42.json").read_text(encoding="utf-8"))["metrics"]
    fe = json.loads((PD / f"final_eval_mock_{t}.json").read_text(encoding="utf-8"))["metrics"]
    at = json.loads((A / f"attack_M_{t}_def.json").read_text(encoding="utf-8"))
    best = "guest" if tl["guest"]["auroc"] >= tl["host"]["auroc"] else "host"
    dk = f"delta_{best}_minus_SF 联邦 SGB"
    vals = [t, tl["guest"]["auroc"], tl["host"]["auroc"], tl["SF 联邦 SGB"]["auroc"], f"=D{i}-MAX(B{i},C{i})",
            -tl[dk]["ci_hi"], -tl[dk]["ci_lo"], published[t],
            max(fe["guest_tuned"]["auroc"], fe["host_tuned"]["auroc"]),
            fe["fed_opt"]["auroc"] if "fed_opt" in fe else "未跑（CI 过宽）",
            f"=J{i}-D{i}" if "fed_opt" in fe else "—",
            at["A1_by_known_frac"]["0.05"]["leakage_auc"], at["A2_infer"]["truth"]["holder_recall_pooled"]]
    for j, v in enumerate(vals, 1):
        c = wm.cell(i, j, v); c.border = BOX; c.alignment = WRAP
        c.font = BLACK if (isinstance(v, str) and v.startswith("=")) or j == 1 else BLUE
        if j in (2, 3, 4, 9, 10): c.number_format = "0.0000"
        if j in (5, 6, 7, 8, 11, 12): c.number_format = "+0.0000;-0.0000;0.0000"
        if j == 13: c.number_format = "0.0%"
for j, w in enumerate([11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12], 1):
    wm.column_dimensions[get_column_letter(j)].width = w
wm.row_dimensions[4].height = 46

# ---------------------------------------------------------------- FedAds
wf = wb.create_sheet("FedAds")
wf["A1"] = "FedAds 复现（官方对齐训练集抽 30 万 / 测试集抽 20 万，正例约 0.6%）"
wf["A1"].font = TITLE
wf["A2"] = "相对差值 = 方案 − 参照（公式）；CI 来自客户级配对 bootstrap 1000 次（蓝字）。泄漏 = 香港另知 5% 标签时的 ΔAUC。"
wf["A2"].font = NOTE
PD = A.parent
fe = json.loads((PD / "final_eval_fedads.json").read_text(encoding="utf-8"))["metrics"]
PR = PD / "predictions"
hf = ["方案", "配置", "测试 AUC", "相对调参内地单边", "CI 下界", "CI 上界", "相对调参集中式", "CI 下界", "CI 上界", "训练耗时（s）", "A1 泄漏（5%）", "A2 推理期用到的香港列"]
for j, h in enumerate(hf, 1):
    c = wf.cell(4, j, h); c.font, c.fill, c.border, c.alignment = BOLD, HEAD_FILL, BOX, WRAP
base_rows = [("内地单边（调参）", "XGB lr0.1/深度3/λ1/27 轮", fe["guest_tuned"]["auroc"]),
             ("香港单边（调参）", "XGB lr0.3/深度3/λ0.1/4 轮", fe["host_tuned"]["auroc"]),
             ("集中式（调参）", "XGB lr0.3/深度3/λ10/列采样0.7/20 轮", fe["central_tuned"]["auroc"])]
for i, (n, cfg, a) in enumerate(base_rows, 5):
    for j, v in enumerate([n, cfg, a], 1):
        c = wf.cell(i, j, v); c.border = BOX; c.font = BLUE if j == 3 else BLACK
        if j == 3: c.number_format = "0.0000"
fed_rows = [("def", "联邦：原版默认", "SGB 100 轮/深度5/λ0.1/约10桶", "fedads_sgb_def", "F_def"),
            ("opt", "联邦：调参 + 33 桶", "SGB 20 轮/深度3/λ10/列采样0.7/33桶", "fedads_sgb_opt", "F_opt"),
            ("tuned10", "联邦：调参 + 约 10 桶", "同上，约10桶", "fedads_sgb_tuned10", None),
            ("optbase", "联邦：调参 + 33 桶 + 先验初始分数", "同上 + base_score=logit(先验)", "fedads_sgb_opt_base", "F_opt_base")]
for i, (k, n, cfg, rid, atk) in enumerate(fed_rows, 8):
    r = fe[f"fed_{k}"]
    g = fe[f"fed_{k}_minus_guest_tuned"]; ce = fe[f"fed_{k}_minus_central_tuned"]
    fit = json.loads((PR / f"{rid}.json").read_text(encoding="utf-8"))["fit_seconds"]
    leak = a2 = "—"
    if atk:
        at = json.loads((A / f"attack_{atk}.json").read_text(encoding="utf-8"))
        leak = at["A1_by_known_frac"]["0.05"]["leakage_auc"]
        a2 = f"{at['A2_infer']['features_used']}/{at['A2_infer']['features_total']}（被用到的列 100% 被确定还原）"
    vals = [n, cfg, r["auroc"], f"=C{i}-$C$5", g["ci_lo"], g["ci_hi"], f"=C{i}-$C$7", ce["ci_lo"], ce["ci_hi"], fit, leak, a2]
    for j, v in enumerate(vals, 1):
        c = wf.cell(i, j, v); c.border = BOX; c.alignment = WRAP
        c.font = BLACK if (isinstance(v, str) and v.startswith("=")) or j in (1, 2, 12) else BLUE
        if j == 3: c.number_format = "0.0000"
        if j in (4, 5, 6, 7, 8, 9, 11): c.number_format = "+0.0000;-0.0000;0.0000"
        if j == 10: c.number_format = "#,##0"
ld = json.loads((A / "label_dp_proxy_fedads.json").read_text(encoding="utf-8"))
r0 = 14
wf.cell(r0, 1, "标签 DP（明文代理，同调参集中式配置）").font = BOLD
for j, h in enumerate(["ε", "对称 RR", "后验去噪", "后验 − 对称"], 1):
    c = wf.cell(r0 + 1, j, h); c.font, c.fill, c.border = BOLD, HEAD_FILL, BOX
for i, e in enumerate(["2", "4", "8"], r0 + 2):
    for j, v in enumerate([e, ld[f"eps{e}_rr"], ld[f"eps{e}_rr_soft"], f"=C{i}-B{i}"], 1):
        c = wf.cell(i, j, v); c.border = BOX; c.font = BLACK if j in (1, 4) else BLUE
        if j in (2, 3): c.number_format = "0.0000"
        if j == 4: c.number_format = "+0.0000;-0.0000"
si = json.loads((A / "secure_infer_F.json").read_text(encoding="utf-8"))
wf.cell(r0 + 6, 1, f"SPU 安全推理（联邦：调参 + 33 桶 + 先验初始分数）：20 棵树 × 200,000 用户 {si['seconds']:.1f} s，"
                    f"AUC {si['auc']:.4f}（原生 {si['auc_sgb_native']:.4f}），逐样本最大偏差 {si['max_abs_diff_vs_sgb']:.1e}").font = BLACK
wf.cell(r0 + 7, 1, "先验初始分数在 Santander 上的复核：优化版 + 先验初始分数 0.9086，优化版 0.9091，差 −0.0005 [−0.0029, +0.0021]（无差异；Top-5% Lift 9.53× → 9.77×）").font = BLACK
for j, w in enumerate([30, 34, 9, 11, 9, 9, 11, 9, 9, 11, 11, 30], 1):
    wf.column_dimensions[get_column_letter(j)].width = w
wf.row_dimensions[4].height = 32

# ---------------------------------------------------------------- 说明
wn = wb.create_sheet("说明")
notes = [
    ("口径", "数据：Santander 产品推荐，原仓库 business 纵向切分（内地 20 列 + 标签；香港 25 列产品持有）；时间外推，训练 78,267 行，测试 21,735 行 / 645 正例。"),
    ("框架", "隐语 SecretFlow 1.13.0b0，SGB（sgb_v），OU 2048 位同态加密，debug 单进程，seed 42。"),
    ("CI", "客户级 cluster bootstrap 1000 次（seed 42），各方案在同一批重采样上配对，差值 CI 取 2.5% / 97.5% 分位。"),
    ("威胁模型", "两方半诚实、不合谋，只分析协议中合法收到的消息；消息按隐语源码重放，重放预测与隐语输出最大偏差 3.5e-7。"),
    ("A2 确定还原", "对每个「样本 × 香港特征」维护取值区间，区间内只剩一个可能取值即计为确定还原；召回 = 真实持有者中被确定识别为持有的比例，精确率 = 被识别为持有者中真实持有的比例。"),
    ("来源文件", "data/processed/attack/defense_utility.json；data/processed/attack/attack_*.json（生成于 2026-10-20）。"),
]
wn["A1"] = "说明"
wn["A1"].font = TITLE
for i, (k, v) in enumerate(notes, 3):
    wn.cell(i, 1, k).font = BOLD
    c = wn.cell(i, 2, v)
    c.font = BLACK
    c.alignment = Alignment(wrap_text=True, vertical="top")
wn.column_dimensions["A"].width = 14
wn.column_dimensions["B"].width = 110

wb.save(OUT)
print("saved", OUT)
