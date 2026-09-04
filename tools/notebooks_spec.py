"""五个模块 notebook 的内容定义。与 build_notebooks.py 配套。"""
from build_notebooks import HEADER, build

PLOT = '''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "PingFang SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
FIGSIZE_WIDE = (9, 4)
FIG_DPI = 150
GRID_W = 11
GRID_H = 7
FIGDIR = ROOT / "{mod}" / "results"'''


def m2():
    build("modules/m2_synthetic/notebooks/S2.1_scm_generation.ipynb", [
        ("md", "# S2.1 · 合成数据生成与正确性验收\n\n"
               "本步用结构因果模型（SCM）生成纵向联邦场景的双方数据。"
               "**关键在于地面真值已知**：互补性 λ、冗余度、重叠率都是我们设定的，"
               "因此可以检验模型学到的增益是否与理论增益一致。\n\n"
               "> ⚠️ 合成数据只能验证**机制**，不能替代真实业务数据做效果承诺。"),
        ("code", HEADER.format(cfg="modules/m2_synthetic/configs/scenarios.yaml")),
        ("code", "from modules.m2_synthetic.components.scm_generator import (\n"
                 "    load_scenarios, generate, usable_mask, theoretical_gain_signal)\n"
                 "scenarios = load_scenarios(config)\n"
                 "pd.DataFrame([vars(s) for s in scenarios]).set_index('name')"),
        ("md", "## 各场景的可用样本量与合规折损\n\n"
               "`usable = 交集 ∩ 同意`。这一步的折损是**合规成本的直接体现**。"),
        ("code", "rows = []\n"
                 "for s in scenarios:\n"
                 "    d = generate(s, seed)\n"
                 "    m = usable_mask(d)\n"
                 "    rows.append({'场景': s.name, '总样本': len(d['y']),\n"
                 "                 '交集内': int(d['in_overlap'].sum()),\n"
                 "                 '交集且同意': int(m.sum()),\n"
                 "                 '正样本率': float(d['y_control'][m].mean()),\n"
                 "                 '合规留存率': float(m.mean())})\n"
                 "pd.DataFrame(rows).set_index('场景').round(ROUND_DP)"),
        ("md", "## 正确性验收：理论增益 vs 实测增益\n\n"
               "`theoretical_gain_signal` 用**真实信号**构造两个 oracle 打分（含 B / 不含 B），"
               "其 AUC 差就是该场景下 B 侧数据的理论价值上界。"),
        ("code", "from sklearn.metrics import roc_auc_score\n"
                 "rows = []\n"
                 "for s in scenarios:\n"
                 "    for sd in config['seeds']:\n"
                 "        d = generate(s, sd); m = usable_mask(d); g = theoretical_gain_signal(d)\n"
                 "        y = d['y_control'][m]\n"
                 "        rows.append({'场景': s.name, '种子': sd,\n"
                 "                     '理论增益': roc_auc_score(y, g['with_b'][m])\n"
                 "                                 - roc_auc_score(y, g['without_b'][m])})\n"
                 "th = pd.DataFrame(rows).groupby('场景')['理论增益'].agg(['mean', 'std']).round(ROUND_DP)\n"
                 "th"),
        ("md", "**验收判据**：S2_零互补 的理论增益必须为 0——这是生成器实现正确的硬检验。"),
        ("code", "ZERO_TOL = 1e-9\n"
                 "v = float(th.loc['S2_零互补', 'mean'])\n"
                 "print('S2_零互补 理论增益 =', v)\n"
                 "print('验收:', '通过' if abs(v) < ZERO_TOL else '不通过')"),
        ("md", "## S2.3 · 第二种 SCM 结构：检验结论对生成假设的稳健性\n\n"
               "上面全部场景的信号都是潜变量的**线性组合**。这带来一个隐患："
               "「调参后 L3a 联邦LR > L3c SplitNN > L3b 纵向GBDT」这个排序，"
               "有多少是模型本身的差别，有多少只是**生成方式偏向线性模型**？\n\n"
               "分支卡 `BC-M5-001` 的 falsifier 正是为此写下的。这里加两种非线性信号形式：\n\n"
               "| 形式 | 信号 | 检验的机制 |\n|---|---|---|\n"
               "| `linear` | s_shared + s_a + λ·s_b | 可加贡献（原场景库） |\n"
               "| `interaction` | s_shared + s_a + λ·(s_a × s_b) | 被动方**只经交互**起作用 |\n"
               "| `threshold` | s_shared + s_a + λ·1[s_b > 0] | 被动方以阶跃方式起作用 |\n\n"
               "`interaction` 最有理论动机：同样的 s_b，在 s_a 高的人身上是正向、低的人身上是负向。"
               "线性模型无论怎么调参都拿不到它（需要显式交叉项），树模型可以逼近。"
               "**「双方特征的交互只有联合建模才能捕获」正是纵向联邦的核心论点之一。**"),
        ("code", "nl_raw = yaml.safe_load(open(\n"
                 "    ROOT / 'modules/m2_synthetic/configs/scenarios_nonlinear.yaml', encoding='utf-8'))\n"
                 "nl = load_scenarios(nl_raw)\n"
                 "rows = []\n"
                 "for c in nl:\n"
                 "    g = [roc_auc_score(generate(c, sd)['y_control'][usable_mask(generate(c, sd))],\n"
                 "                       theoretical_gain_signal(generate(c, sd))['with_b']"
                 "[usable_mask(generate(c, sd))])\n"
                 "         - roc_auc_score(generate(c, sd)['y_control'][usable_mask(generate(c, sd))],\n"
                 "                         theoretical_gain_signal(generate(c, sd))['without_b']"
                 "[usable_mask(generate(c, sd))])\n"
                 "         for sd in nl_raw['seeds']]\n"
                 "    rows.append({'场景': c.name, '信号形式': c.signal_form,\n"
                 "                 '互补性': c.complementarity, '理论增益': float(np.mean(g))})\n"
                 "pd.DataFrame(rows).set_index('场景').round(ROUND_DP)"),
        ("md", "**两项验收**：`N5_交互_零互补` 的理论增益必须为 0（λ=0 对三种形式同时成立）；"
               "`N6_线性_对照` 与原场景库 `S1_基准` 的理论增益必须一致（两份配置口径相同）。"),
        ("md", "### 调参后的模型排序：分支卡 falsifier 的判决"),
        ("code", "nlad = pd.read_csv(ROOT / 'modules/m5_modeling/results/nonlinear_ladder.csv')\n"
                 "COLS = ['N6_线性_对照', 'N1_交互_基准', 'N2_交互_高',\n"
                 "        'N3_阶跃_基准', 'N4_阶跃_高', 'N5_交互_零互补']\n"
                 "piv = nlad.pivot_table(index='level', columns='scenario', values='test_auc',\n"
                 "                       aggfunc='mean')\n"
                 "piv[COLS].round(ROUND_DP)"),
        ("code", "L3 = ['L3a_联邦LR', 'L3b_纵向GBDT', 'L3c_SplitNN']\n"
                 "for sc in COLS:\n"
                 "    order = piv.loc[L3, sc].sort_values(ascending=False)\n"
                 "    print(f\"{sc:14s} \" + ' > '.join(f'{k}({v:.4f})' for k, v in order.items()))"),
        ("code", "PCT = 100\n"
                 "print('C1 稳健性：L1 捕获比例')\n"
                 "for sc in COLS:\n"
                 "    l0, l1 = piv.loc['L0_LR', sc], piv.loc['L1_LR', sc]\n"
                 "    best3 = piv.loc[L3, sc].max()\n"
                 "    share = (l1 - l0) / (best3 - l0) * PCT if best3 > l0 else float('nan')\n"
                 "    print(f'  {sc:14s} L1−L0={l1-l0:+.4f}  最优L3−L0={best3-l0:+.4f}  "
                 "L1 捕获 {share:5.1f}%')"),
    ])


def m3():
    build("modules/m3_alignment/notebooks/S3.1_psi_alignment.ipynb", [
        ("md", "# S3.1 · 实体对齐（PSI）质量与下游影响\n\n"
               "本步**不实现**真实密码学 PSI，只仿真其可观测行为，"
               "回答一个工程问题：**对齐做得不好，会损失多少模型价值？**\n\n"
               "两类失败模式性质完全不同：\n"
               "- **漏配**：本该匹配上的人没匹配上 → 样本量减少\n"
               "- **错配**：把甲的特征接到乙身上 → 注入无关噪声，且属于错误的个人信息关联"),
        ("code", HEADER.format(cfg="modules/m2_synthetic/configs/scenarios.yaml")),
        ("code", "from dataclasses import replace\n"
                 "from modules.m2_synthetic.components.scm_generator import load_scenarios, generate, usable_mask\n"
                 "from modules.m3_alignment.components.psi import simulate_psi, psi_cost, apply_misattribution\n"
                 "from modules.m5_modeling.components import models as M\n"
                 "from sklearn.metrics import roc_auc_score\n"
                 "base = [s for s in load_scenarios(config) if s.name == 'S1_基准'][0]\n"
                 "pd.read_csv(ROOT / 'modules/m3_alignment/results/psi_quality.csv')"
                 ".groupby('match_error_rate')[['n_true_overlap','n_matched','false_negative','recall']]"
                 ".mean().round(ROUND_DP)"),
        ("md", "## 错配对 VFL 增益的影响"),
        ("code", "pd.read_csv(ROOT / 'modules/m3_alignment/results/misattribution_impact.csv')"
                 ".groupby('misattribution_rate')[['L0','L3a','增益']].mean().round(ROUND_DP)"),
        ("md", "## ECDH-PSI 的成本量级\n\n只用于判断可行性，不是性能基准。"),
        ("code", "pd.read_csv(ROOT / 'modules/m3_alignment/results/psi_cost.csv').round(2)"),
        ("md", "**结论**：漏配 10% 只损失约 12% 增益，错配 40% 才使增益折半。"
               "**实体对齐质量不是本项目的主要风险**——主要风险在互补性与合规折损。"),
    ])


def m5():
    build("modules/m5_modeling/notebooks/S5.1_baseline_ladder.ipynb", [
        ("md", "# S5.1 · 五级基线阶梯\n\n"
               "本项目的科学核心不是「联邦学习能不能跑通」，而是"
               "**联邦学习相对于合规成本低得多的替代方案，究竟多带来多少价值**。\n\n"
               "| 级别 | 含义 | 跨境暴露面 |\n|---|---|---|\n"
               "| L0 | 内地单方建模 | 无 |\n"
               "| L1 | 加对方 k-匿名聚合统计 | 聚合量，非个人信息 |\n"
               "| L2 | 加对方粗粒度标记 | 逐人低维标记 |\n"
               "| L3 | 纵向联邦（LR / GBDT / SplitNN） | 中间量逐人交换 |\n"
               "| L4 | 集中式（原始数据汇集） | 全量个人信息 |\n\n"
               "**L1 是 VFL 真正的竞争者**：它几乎没有合规成本。"
               "如果 L1 能拿到大部分价值，本项目的商业前提就不成立。"),
        ("code", HEADER.format(cfg="modules/m5_modeling/configs/experiment.yaml")),
        ("code", "import time\n"
                 "from modules.m2_synthetic.components.scm_generator import load_scenarios\n"
                 "from modules.m5_modeling.components.experiment import run_one\n"
                 "scen_cfg = yaml.safe_load(open(ROOT / 'modules/m2_synthetic/configs/scenarios.yaml', encoding='utf-8'))\n"
                 "scenarios = load_scenarios(scen_cfg)\n"
                 "hp, seeds = config['hyperparams'], config['seeds']\n"
                 "t0 = time.time(); rows = []\n"
                 "for c in scenarios:\n"
                 "    for sd in seeds:\n"
                 "        for sp in config['splits']:\n"
                 "            rows += run_one(c, sd, hp, sp)\n"
                 "df = pd.DataFrame(rows)\n"
                 "df.to_csv(ROOT / 'modules/m5_modeling/results/ladder_results_raw.csv', index=False)\n"
                 "print(f'网格完成 {time.time()-t0:.0f}s | {len(df)} 行 | "
                 "{df.scenario.nunique()} 场景 × {df.seed.nunique()} 种子 × {df.split.nunique()} 划分')"),
        ("md", "## 表1 · 阶梯总表（全场景合并，随机划分，均值 ± 95% CI）\n\n"
               "置信区间由**跨种子**变异给出——框架要求 ≥5 种子且主指标带区间，此项不降级。"),
        ("code", "CI_Z = 1.96\n"
                 "core = df[df.auc.notna()].copy()\n"
                 "def ci95(x):\n"
                 "    x = np.asarray(x, float); m = x.mean()\n"
                 "    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0\n"
                 "    return pd.Series({'AUC均值': m, 'CI下界': m - CI_Z*se, 'CI上界': m + CI_Z*se})\n"
                 "core[core.split == 'random'].groupby('level').auc.apply(ci95).unstack()"
                 ".sort_values('AUC均值', ascending=False).round(ROUND_DP)"),
        ("md", "## 表2 · 划分口径：D-1 随机 vs D-2 时间外推（OOT）\n\n"
               "金融业的真实口径是 OOT。差值为正说明随机划分**高估**了性能。"),
        ("code", "p = core.pivot_table(index='level', columns='split', values='auc', aggfunc='mean')\n"
                 "p['随机减OOT'] = p['random'] - p['oot']\n"
                 "p.sort_values('随机减OOT', ascending=False).round(ROUND_DP)"),
        ("md", "## 表3 · 逐场景：L3a 联邦LR 相对 L1 的净增益"),
        ("code", "w = core[core.split == 'random'].pivot_table(index=['scenario','seed'], columns='level', values='auc')\n"
                 "g = (w['L3a_联邦LR'] - w['L1_加k匿名统计_LR']).groupby('scenario').apply(ci95).unstack()\n"
                 "g['显著'] = np.where(g['CI下界'] > 0, '是', '否')\n"
                 "g.round(ROUND_DP)"),
        ("md", "## 表4 · 通信成本"),
        ("code", "c = core[core.split=='random'].groupby('level')[['comm_rounds','comm_floats']].mean().dropna(how='all')\n"
                 "c[c.comm_floats > 0].round(0)"),
        ("md", "## 表5 · 决策等价性：L1 与 L3 的 Top-10% 名单重合度\n\n"
               "AUC 接近**不等于**决策一致。真正影响业务的是名单换了多少人。"),
        ("code", "o = df[df['topk_overlap@10'].notna()]\n"
                 "o[o.split=='random'].groupby('level')['topk_overlap@10'].agg(['mean','std']).round(ROUND_DP)"),
        ("md", "## 表6 · 增量（uplift）评估\n\n"
               "**响应率 ≠ 增量**。营销真正要的是「因为营销才转化」的人。"),
        ("code", "u = df[df.auuc.notna()]\n"
                 "u[u.split=='random'].groupby('level')[['auuc','uplift@10']].mean().round(ROUND_DP)"),
        ("md", "## 表7 · 稳健性检验：调参能否救回 L1？\n\n"
               "上面的全部结论都用**同一组固定超参**。一个合理的质疑是："
               "「L1 只捕获 12.3%」会不会只是 L1 欠调参？\n\n"
               "依 DR-GOV-009 的红线「L1 必须与 L3 同等认真实现」，"
               "**L1 的搜索空间刻意给到最大**——60 组，是 L0 的 12 倍。"
               "若调参能翻转 C1，必须让它有机会翻转。\n\n"
               "方法上有一条硬要求：**选参只看验证集**（60/20/20 三分）。"
               "在测试集上选参会让搜索空间大的级别虚高更多，"
               "污染方向恰好**有利于 L1**，同样会毁掉结论。"),
        ("code", "hs = pd.read_csv(ROOT / 'modules/m5_modeling/results/hyperparam_search.csv')\n"
                 "t = hs.groupby('level').agg(网格点数=('n_points','first'),\n"
                 "                            验证AUC=('valid_auc','mean'),\n"
                 "                            测试AUC=('test_auc','mean'))\n"
                 "t['验证减测试'] = t['验证AUC'] - t['测试AUC']\n"
                 "t.sort_values('测试AUC', ascending=False).round(ROUND_DP)"),
        ("md", "「验证减测试」一列随网格增大而增大（L3b 的 81 组差 0.0412），"
               "说明选参乐观度被正确检出——这是方法本身的自检。"),
        ("md", "### 最强的一击：让 L1 直接在测试集上挑最优点\n\n"
               "下表的 oracle 上界**不是有效估计**（它用了测试集选参，等于作弊），"
               "但它给出「调参最多能帮 L1 到什么程度」的上限。"),
        ("code", "ob = pd.read_csv(ROOT / 'modules/m5_modeling/results/oracle_upper_bound.csv')\n"
                 "o = ob.groupby('level').oracle_test_auc.mean()\n"
                 "h = hs.groupby('level').test_auc.mean()\n"
                 "cmp2 = pd.DataFrame({'正规选参': h[o.index], 'oracle上界': o})\n"
                 "cmp2['乐观量'] = cmp2['oracle上界'] - cmp2['正规选参']\n"
                 "cmp2.round(ROUND_DP)"),
        ("code", "PCT = 100\n"
                 "for tag, col in [('正规选参', h), ('oracle上界（作弊）', o)]:\n"
                 "    gap_l1 = col['L1_LR'] - col['L0_LR']\n"
                 "    gap_l3 = col['L3a_联邦LR'] - col['L0_LR']\n"
                 "    print(f'{tag:18s} L1−L0={gap_l1:+.4f}  L3a−L0={gap_l3:+.4f}  '\n"
                 "          f'L1 捕获 {gap_l1/gap_l3*PCT:.1f}%')"),
        ("md", "**C1 站住了，而且更强了。**\n\n"
               "- 正规口径：L1 只捕获 **2.1%**\n"
               "- oracle 口径（让 L1 作弊）：也只有 **15.5%**\n"
               "- 此前 40 种子安慰剂对照给出 12.3%，正好落在两者之间\n\n"
               "三个独立口径互相印证：**调参救不回 L1**。"
               "L1 的瓶颈是结构性的——分段键只能用主动方特征，"
               "回传的统计量因而是主动方特征的函数。"),
    ])


def m6():
    build("modules/m6_evaluation/notebooks/S6.1_conditional_value.ipynb", [
        ("md", "# S6.1 · 条件价值曲线\n\n"
               "「联邦学习有没有用」是个坏问题——它**在什么条件下有用**才是可回答的。\n\n"
               "本步扫描四个结构参数，画出增益随条件变化的曲线，并给出**盈亏平衡点**："
               "增益低于多少时，项目不值得做。"),
        ("code", HEADER.format(cfg="modules/m5_modeling/configs/experiment.yaml")),
        ("code", PLOT.format(mod="modules/m6_evaluation") + "\n"
                 "sw = pd.read_csv(ROOT / 'modules/m5_modeling/results/sweep_results.csv')\n"
                 "L3, L1, L0 = 'L3a_联邦LR', 'L1_加k匿名统计_LR', 'L0_内地单方_LR'\n"
                 "def curve(param):\n"
                 "    d = sw[sw.sweep_param == param]\n"
                 "    t = d.pivot_table(index='sweep_value', columns='level', values='auc', aggfunc='mean')\n"
                 "    return pd.DataFrame({'L0': t[L0], 'L1': t[L1], 'L3a': t[L3],\n"
                 "                         'L3a减L0': t[L3]-t[L0], 'L3a减L1': t[L3]-t[L1]})\n"
                 "curve('complementarity').round(ROUND_DP)"),
        ("md", "**互补性 λ 是主导因子**：λ=0 时增益 0.025，λ=2.0 时增益 0.209——相差八倍。"),
        ("code", "PARAMS = ['complementarity', 'overlap_rate', 'redundancy', 'match_error_rate']\n"
                 "LABELS = ['互补性 λ', '重叠率 ρ', '冗余度', '匹配错误率']\n"
                 "N_COL = 2\n"
                 "fig, axes = plt.subplots(N_COL, N_COL, figsize=(GRID_W, GRID_H))\n"
                 "for ax, p, lab in zip(axes.ravel(), PARAMS, LABELS):\n"
                 "    c = curve(p)\n"
                 "    ax.plot(c.index, c['L3a减L0'], marker='o', label='L3a − L0')\n"
                 "    ax.plot(c.index, c['L3a减L1'], marker='s', label='L3a − L1')\n"
                 "    ax.axhline(0, color='gray', linewidth=1)\n"
                 "    ax.set_title(lab); ax.set_xlabel(lab); ax.set_ylabel('AUC 增益'); ax.legend()\n"
                 "fig.suptitle('条件价值曲线：纵向联邦的增益在什么条件下成立')\n"
                 "fig.tight_layout(); fig.savefig(FIGDIR / 'conditional_value_curves.png', dpi=FIG_DPI)\n"
                 "plt.close(fig); print('图已保存 conditional_value_curves.png')"),
        ("md", "## 关键对照：40 种子下的 L1 vs L3\n\n"
               "这是本项目**最核心的一个数字**。为压窄置信区间，此对照单独用 40 个种子。\n\n"
               "「安慰剂臂」把 B 侧换成同形状纯噪声——它分离出「分段本身带来的灵活性」，"
               "剩下的才是 B 侧数据的**净贡献**。"),
        ("code", "crit = pd.read_csv(ROOT / 'modules/m5_modeling/results/l1_vs_l3_critical.csv')\n"
                 "CI_Z = 1.96\n"
                 "def ci(x):\n"
                 "    m = x.mean(); se = x.std(ddof=1)/np.sqrt(len(x))\n"
                 "    return pd.Series({'均值': m, 'CI下界': m-CI_Z*se, 'CI上界': m+CI_Z*se})\n"
                 "cmp = pd.DataFrame({\n"
                 "    'L1减L0（表观增益）': ci(crit.L1_真实 - crit.L0),\n"
                 "    'L1减安慰剂（B的净贡献）': ci(crit.L1_真实 - crit.L1_安慰剂),\n"
                 "    'L3a减L0（VFL总增益）': ci(crit.L3a - crit.L0),\n"
                 "    'L3a减L1（VFL净增量）': ci(crit.L3a - crit.L1_真实)}).T\n"
                 "cmp['显著'] = np.where(cmp['CI下界'] > 0, '是', '否')\n"
                 "cmp.round(ROUND_DP)"),
        ("code", "share = (crit.L1_真实 - crit.L1_安慰剂).mean() / (crit.L3a - crit.L0).mean()\n"
                 "PCT = 100\n"
                 "print(f'L1 捕获了 VFL 价值的 {share*PCT:.1f}%')\n"
                 "print('→ L1 是真实的竞争者，但拿不到大头；VFL 的商业前提成立')"),
        ("md", "## 盈亏平衡：增益要多大才值得做\n\n"
               "决策价值 ≈ **增益 × 可触达人数 × 单人价值**。"
               "低重叠场景下单人增益虽高，但可触达人数极小，总价值反而低。"),
        ("code", "ov = curve('overlap_rate')\n"
                 "BASE_N = 1_000_000\n"
                 "ov['可触达人数'] = (ov.index * BASE_N).astype(int)\n"
                 "ov['相对总价值'] = ov['L3a减L1'] * ov['可触达人数']\n"
                 "ov[['L3a减L1', '可触达人数', '相对总价值']].round(ROUND_DP)"),
        ("md", "**结论**：单人增益与可触达人数方向相反，总价值在中高重叠区达到最大。"
               "只看 AUC 增益会得出「越低重叠越好」的错误结论。"),
    ])


def m7():
    build("modules/m7_security/notebooks/S7.1_privacy_attacks.ipynb", [
        ("md", "# S7.1 · 隐私攻击与防护评估\n\n"
               "攻击者设定为**诚实但好奇的协议内参与方**，不假设外部窃听。\n\n"
               "| 编号 | 攻击 | 谁攻击谁 | 暴露面 |\n|---|---|---|---|\n"
               "| A1 | 标签推断 | 被动方 → 主动方标签 | 每轮下发的残差 |\n"
               "| A2 | 嵌入反演 | 主动方 → 被动方特征 | 上传的嵌入 |\n"
               "| A3 | 梯度标签推断 | 被动方 → 主动方标签 | 回传的梯度（仅形态A） |"),
        ("code", HEADER.format(cfg="modules/m5_modeling/configs/experiment.yaml")),
        ("code", "atk = pd.read_csv(ROOT / 'modules/m7_security/results/attack_results.csv')\n"
                 "a1 = atk[atk.attack == 'A1_残差标签推断'].groupby('dp_sigma')[\n"
                 "    ['eps_per_round','utility_auc','leak_auc_首轮','leak_auc_最优轮']].mean()\n"
                 "a1.round(ROUND_DP)"),
        ("md", "首轮泄露 AUC = 1.0000：训练开始时权重为零、预测恒为 0.5，"
               "残差 `r = 0.5 − y` 的符号与标签**一一对应**。"
               "**不加防护的纵向联邦逻辑回归，标签是完全泄露的。**"),
        ("md", "## 证伪检验：跨轮平均攻击\n\n"
               "上表看起来 σ=1.0 就能把泄露压到 0.76 而几乎不损失可用性——这个结论**太好了**。\n\n"
               "但攻击者只用了单轮残差。噪声在轮间独立、标签恒定，"
               "**跨轮平均即可把噪声消掉**。这是必须自己打的证伪。"),
        ("code", "mr = pd.read_csv(ROOT / 'modules/m7_security/results/multiround_attack.csv')\n"
                 "mr.groupby('dp_sigma')[['可用性AUC','单轮攻击','跨轮平均攻击','前50轮平均']].mean().round(ROUND_DP)"),
        ("code", PLOT.format(mod="modules/m7_security") + "\n"
                 "m = mr.groupby('dp_sigma')[['可用性AUC','单轮攻击','跨轮平均攻击']].mean()\n"
                 "L0_REF = 0.7089\n"
                 "fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)\n"
                 "ax.plot(m.index, m['单轮攻击'], marker='o', label='泄露 AUC（单轮攻击）')\n"
                 "ax.plot(m.index, m['跨轮平均攻击'], marker='s', label='泄露 AUC（跨轮平均攻击）')\n"
                 "ax.plot(m.index, m['可用性AUC'], marker='^', label='模型可用性 AUC')\n"
                 "ax.axhline(L0_REF, color='red', linestyle='--', label='L0 内地单方基线')\n"
                 "ax.set_xscale('symlog'); ax.set_xlabel('高斯噪声 σ'); ax.set_ylabel('AUC')\n"
                 "ax.set_title('逐轮加噪防护：跨轮平均攻击下失效')\n"
                 "ax.legend(); fig.tight_layout()\n"
                 "fig.savefig(FIGDIR / 'dp_privacy_utility.png', dpi=FIG_DPI); plt.close(fig)\n"
                 "print('图已保存 dp_privacy_utility.png')"),
        ("md", "**结论（推翻了上一节的乐观读数）**：\n\n"
               "- σ=1.0 时跨轮平均攻击的泄露 AUC 回到 **1.0000**\n"
               "- 要把泄露压到 0.67，需要 σ=30，此时可用性降至 0.6352，"
               "**低于 L0 内地单方基线 0.7089**\n\n"
               "→ **逐轮加高斯噪声不是本协议的有效防护**。"
               "把泄露压下去所需的噪声，会先把联邦模型的价值清零。"
               "有效路径只能是**协议级**手段（安全聚合 / 同态加密 / 秘密分享）或改变暴露面。"),
        ("md", "## A2 嵌入反演 / A3 梯度标签推断"),
        ("code", "a2 = atk[atk.attack == 'A2_嵌入反演'].groupby('protocol')[\n"
                 "    ['utility_auc','inv_r2_mean','inv_r2_max','leak_auc_梯度方向']].mean()\n"
                 "a2.round(ROUND_DP)"),
        ("md", "## S7.4 · 恶意参与方：放开半诚实假设\n\n"
               "此前全部结论的前提是**半诚实**——参与方严格执行协议，只是好奇。"
               "但这个假设在真实场景中往往不成立：协议由代码执行，而代码由某一方控制，"
               "对方无从验证收到的数字是不是真按协议算的。\n\n"
               "**A6 恶意主动方**：不发真实残差，改发构造的探针向量。"
               "被动方按协议更新 `w_b -= lr·(x_bᵀ·r)/n`，若 r 是第 j 个样本上的单位向量，"
               "更新量正比于 `x_b[j]`——下一轮上传量之差直接给出 `x_b·x_b[j]`。"
               "有 d 个辅助样本，每个目标只需 **1 次探针**即可精确反解。"),
        ("code", "mp = pd.read_csv(ROOT / 'modules/m7_security/results/malicious_probe.csv')\n"
                 "t = mp.pivot_table(index='amplitude', columns='n_repeat',\n"
                 "                   values='feat_r2', aggfunc='mean')\n"
                 "t.columns = [f'重复×{c:,}' for c in t.columns]\n"
                 "t.round(ROUND_DP)"),
        ("md", "**关键发现：上行加噪与残差合法性检查必须同时开启，缺一不可。**\n\n"
               "| 防护组合 | 攻破 σ=0.1 所需成本 |\n|---|---|\n"
               "| 只加噪（幅度不受约束） | 幅度 1000 时**仅需 100 次重复** |\n"
               "| 只查合法性（无噪声） | 伪装探针 R²=0.859，**26 次全部未被标记** |\n"
               "| **两者同时** | 合法幅度下需约 **10⁸ 次重复**，实际不可行 |\n\n"
               "机制：单次探针的信号量级是 `lr·amplitude/n`，噪声是固定的 σ——"
               "**信噪比与幅度成正比**。约束幅度的唯一手段就是合法性检查"
               "（真实残差 `r = sigmoid(logit) − y` 必落在 [−1, 1] 且稠密）。\n\n"
               "→ 合法性检查把攻击成本抬高了**六个数量级**。"
               "它不是锦上添花，**它是让加噪防护有意义的前提**。"),
        ("md", "### A7 恶意被动方：定向抬分\n\n"
               "这不是偷数据，是**操纵决策**。被动方送的部分 logit 直接加进最终打分，"
               "主动方无从验证它是不是真由 `x_b·w_b` 算出的。"),
        ("code", "mb = pd.read_csv(ROOT / 'modules/m7_security/results/malicious_boost.csv')\n"
                 "g = mb.groupby('amplitude')[['baseline_in_topk', 'attacked_in_topk',\n"
                 "                             'target_size', 'list_churn']].mean()\n"
                 "g['目标进入率'] = g.attacked_in_topk / g.target_size\n"
                 "g['名单重合度'] = 1 - g.list_churn\n"
                 "g[['baseline_in_topk', 'attacked_in_topk', '目标进入率', '名单重合度']]"
                 ".round(ROUND_DP)"),
        ("md", "**旧的名单稳定性阈值 0.9 抓不到中等幅度的操纵**："
               "幅度 1.0 时 45% 的目标客户被顶进 Top-10% 名单，"
               "而名单重合度仍有 0.93 —— **高于 0.9，不会告警**。\n\n"
               "→ M8 的防护基线已据此收紧到 **0.95**。"
               "但要说清楚：这只是**检测**手段，不是防护——攻击者压低幅度仍可缓慢渗透。"
               "根治需要主动方能验证被动方送来的部分 logit 确由 `x_b·w_b` 算出"
               "（零知识证明类手段），本阶段未实现。"),
        ("md", "两个反直觉的结果：\n\n"
               "1. **形态B 挡住了梯度标签泄露**（A3 从 1.00 降到 0.50）——不回传梯度，暴露面直接消失。\n"
               "2. **形态B 并没有降低特征反演风险，反而更糟**（自监督形态 R²=0.667 > 形态A 的 0.488）。"
               "PCA 编码器是线性的、更容易求逆；随标签训练的编码器反而丢掉了更多与任务无关的信息。\n\n"
               "→ **「冻结编码器 = 更安全」是错的**。它换掉的是标签暴露面，不是特征暴露面。"),
        ("md", "## A4 特征推断：本项目最严重的单项发现\n\n"
               "威胁模型：主动方每轮收到被动方上传的部分 logit `x_b · w_b`，"
               "并另有**少量样本**的 x_b 真值（辅助集）。攻击分两步最小二乘：\n\n"
               "1. 用辅助集解出每轮的 w_b（需辅助样本数 ≥ 特征维数）；\n"
               "2. 用解出的 w_b 反解其余所有样本的 x_b。"),
        ("code", "fi = pd.read_csv(ROOT / 'modules/m7_security/results/feature_inference.csv')\n"
                 "fi[fi.uplink_sigma == 0].groupby('uplink_sigma')[\n"
                 "    ['可用性AUC','特征R²均值','特征R²最差维','条件数']].mean().round(ROUND_DP)"),
        ("md", "**无防护时特征被精确恢复（R² = 1.0000）**。加上 A1 的标签完全泄露，"
               "**无防护的纵向联邦逻辑回归是双向完全泄露的**——"
               "「原始数据不出本地」在这个协议下不构成任何实质保护。"),
        ("md", "### 上行加噪：与标签推断完全相反的结果\n\n"
               "被动方对**上传的部分 logit**加噪（`uplink_sigma`），"
               "这与 A1 的下行加噪是不同参与方的自我保护，不可互相替代。"),
        ("code", "ud = pd.read_csv(ROOT / 'modules/m7_security/results/uplink_defense_utility.csv')\n"
                 "fi2 = fi.groupby('uplink_sigma')[['特征R²均值']].mean()\n"
                 "ud2 = ud.groupby('uplink_sigma')[['训练与推理均加噪','L0']].mean()\n"
                 "ud2.join(fi2).round(ROUND_DP)"),
        ("md", "**结论与 A1 相反：上行加噪是有效防护，且代价极小。**\n\n"
               "- σ=0.1：特征 R² 从 1.0000 降到 0.196，可用性仅损失 0.0014\n"
               "- σ=1.0：R² 降到 0.028，但可用性损失 0.066（VFL 增益的 85%）\n\n"
               "机制解释：A1 的信号（标签）**轮间恒定**，跨轮平均即可消噪；"
               "A4 需要解一个**病态线性系统**（条件数约 4.8×10⁴），噪声被放大五个数量级。"
               "**同一种手段对两类攻击效果相反，因此防护方案必须逐攻击面评估，不能一刀切。**"),
        ("md", "### 证伪检验：更强的攻击能否绕过？"),
        ("code", "st = pd.read_csv(ROOT / 'modules/m7_security/results/feature_inference_stronger.csv')\n"
                 "st.groupby(['uplink_sigma','n_aux'])[['朴素最小二乘','加强版(最优岭系数)']]"
                 ".mean().round(ROUND_DP)"),
        ("md", "加强版攻击（岭正则化 + 更大辅助集）确实优于朴素最小二乘，"
               "但**没有推翻结论**：σ=1.0 时即使给攻击者 2000 个辅助样本（已知训练集大部分 x_b 真值，"
               "这是极其宽松的假设），R² 也只到 0.129。\n\n"
               "注意朴素最小二乘在噪声下**辅助样本越多反而越差**（σ=1.0、n_aux=2000 时 R² = −3.60）——"
               "只报朴素版本会在相反方向上误导读者，故两版并列报告。"),
        ("md", "## A5 成员推断：先验证攻击，再下结论\n\n"
               "「这个人是否在你的建模样本里」本身就是个人信息，即便特征与标签都没泄露。\n\n"
               "**但朴素的损失阈值攻击没有通过有效性验证**：在刻意制造记忆的模型上"
               "（训练 AUC = 1.0000、过拟合间隙 0.32），它也只能达到 0.52。"
               "攻击弱到这个程度，就**不能据此宣称「成员推断无威胁」**。"),
        ("code", "mc = pd.read_csv(ROOT / 'modules/m7_security/results/membership_capacity.csv')\n"
                 "g = mc.groupby(['depth','rounds'])[['训练AUC','测试AUC','membership_auc']].mean()\n"
                 "g['过拟合间隙'] = g['训练AUC'] - g['测试AUC']\n"
                 "g.round(ROUND_DP)"),
        ("md", "因此改用 **LiRA 式影子模型校准**：逐样本比较「在训练集内」与「不在训练集内」"
               "两种情形下的置信度分布。它随模型容量单调增强，**通过了有效性验证**。"),
        ("code", "lira = pd.read_csv(ROOT / 'modules/m7_security/results/membership_lira.csv')\n"
                 "lira.groupby('level').membership_auc_lira.agg(['mean','std']).round(ROUND_DP)"),
        ("md", "**结论：成员泄露由模型容量驱动，与是否联邦无关。**\n\n"
               "- L0 内地单方（0.5105）≈ L1（0.5105）≈ L4/L3 信息集（0.5103）——"
               "引入对方数据**不增加**成员泄露\n"
               "- 纵向 GBDT 深 3（0.528）→ 深 6（0.545）——**容量才是驱动因素**\n\n"
               "→ 工程含义：控制树深，而不是纠结要不要联邦。\n\n"
               "> **降级说明**：本实现用 32 个影子模型，LiRA 原文通常用 64–256；"
               "且本项目样本量约 2000、特征维数低，模型本身记忆有限。"
               "因此上述数值应读作**该设定下的下界**，不构成「成员推断不可行」的一般结论。"),
    ])


def m8():
    build("modules/m8_industrialization/notebooks/S8.1_guardrails_and_faults.ipynb", [
        ("md", "# S8.1 · 部署护栏与故障注入\n\n"
               "M8 的两件事：**上线前拦得住不合规配置**，**上线后扛得住故障**。\n\n"
               "路线取舍这一环已被 C7 改写——合成数据上的效果排序不可用于选型，"
               "取舍改以**暴露面与合规成本**为依据。因此本模块的重心是防护基线与鲁棒性，"
               "而不是「哪条路线效果好」。"),
        ("code", HEADER.format(cfg="modules/m8_industrialization/configs/deployment_profile.yaml")),
        ("md", "## 防护基线：每条都对应一个已实测的攻击\n\n"
               "写在文档里的基线，上线时没人会逐条核对。所以做成可执行检查。"),
        ("code", "RATIONALE_CHARS = 60\n"
                 "import tempfile\n"
                 "sys.path.insert(0, str(ROOT / 'platform' / 'orchestration'))\n"
                 "from modules.m8_industrialization.components import guardrails as G\n"
                 "from modules.m8_industrialization.components import fault_injection as FI\n"
                 "import pipeline as P, main_chain as MC\n"
                 "rows = []\n"
                 "BASELINE_KEYS = ['uplink_noise', 'label_protection', 'model_capacity',\n"
                 "                 'k_anonymity', 'splitnn', 'route_selection']\n"
                 "for key in BASELINE_KEYS:\n"
                 "    item = config[key]\n"
                 "    rows.append({'基线项': key,\n"
                 "                 '依据': str(item.get('evidence', '—')),\n"
                 "                 '理由': ' '.join(str(item.get('rationale', ''))"
                 ".split())[:RATIONALE_CHARS]})\n"
                 "pd.DataFrame(rows).set_index('基线项')"),
        ("md", "### 违规配置必须被拒绝\n\n"
               "一条只会放行的护栏比没有护栏更糟——它给人以已经检查过的错觉。"),
        ("code", "# 合规配置直接由基线派生——「合规」的定义就是基线本身，不该另写一套数字\n"
                 "good = {'uplink_sigma': config['uplink_noise']['sigma_min'],\n"
                 "        'label_protection': config['label_protection']['required_mechanism'],\n"
                 "        'gbdt_max_depth': config['model_capacity']['gbdt_max_depth'],\n"
                 "        'k_anonymity': config['k_anonymity']['k_min'],\n"
                 "        'splitnn_mode': 'frozen_pca',\n"
                 "        'route_selected_by': 'exposure_and_compliance'}\n"
                 "print('合规配置判定:', G.check_deployment(good, config)['verdict'])"),
        ("code", "# 违规配置同样由基线派生：逐项推到基线之外\n"
                 "OVER = 2\n"
                 "bad = dict(good,\n"
                 "           uplink_sigma=0.0,\n"
                 "           label_protection='gaussian_noise',\n"
                 "           gbdt_max_depth=config['model_capacity']['gbdt_max_depth'] * OVER,\n"
                 "           k_anonymity=config['k_anonymity']['k_min'] // OVER,\n"
                 "           splitnn_mode='frozen_random',\n"
                 "           route_selected_by='effect_ranking')\n"
                 "res = G.check_deployment(bad, config)\n"
                 "print('判定:', res['verdict'], '| 命中规则数:', len(res['violations']),\n"
                 "      '/ 共', res['checked_rules'])\n"
                 "pd.DataFrame(res['violations'])[['rule', 'detail']]"),
        ("md", "## 故障注入 7/7\n\n"
               "框架的放行判据要求七项全过。这里不做仿真式的「假装失败」，"
               "而是真的把故障注入 platform 的主链路。"),
        ("code", "smoke = yaml.safe_load(open(ROOT / 'platform/configs/smoke.yaml', encoding='utf-8'))\n"
                 "with tempfile.TemporaryDirectory() as d:\n"
                 "    cases = FI.run_all(P, MC, smoke, d)\n"
                 "fi = pd.DataFrame(cases).set_index('fault_id')\n"
                 "fi[['name', 'result', 'detail']]"),
        ("code", "s = FI.summarize(cases)\n"
                 "print(f\"故障注入 {s['passed']}/{s['total']}\", \n"
                 "      '——达标' if s['all_passed'] else f\"——未达标：{s['failed_ids']}\")"),
        ("md", "**七项各自防的是不同的事故**，其中两项值得单独说：\n\n"
               "- **F2 单方下线降级**：真正危险的不是崩溃，是**静默降级**——"
               "下游会把 L0 单方模型的名单当作联邦模型的名单来用。"
               "故判据要求降级后必须**如实上报** `degraded=True` 并给出原因。\n"
               "- **F4 schema 变更安全失败**：对方悄悄改了字段而链路照常出分，"
               "名单会全错而无人察觉。故要求抛异常停止出分，"
               "且**不得被重试掩盖**——重试确定性错误只是把故障拖长。"),
    ])


def m9():
    build("modules/m9_documentation/notebooks/S9.1_evidence_chain.ipynb", [
        ("md", "# S9.1 · 合规证据链核验\n\n"
               "M9 的验收人不是技术方，而是法务与合规。所以本模块的第一件事不是写文档，"
               "而是**让「每条结论追得到证据」成为可执行的检查**——"
               "写成「已核验」四个字，谁也不知道核了什么。\n\n"
               "> ⚠️ 核验的是**可追溯性**，不是**正确性**。"
               "证据文件存在不代表结论正确，只代表读者能查到它依据的是什么。"),
        ("code", HEADER.format(cfg="modules/m9_documentation/configs/evidence_map.yaml")),
        ("code", "from modules.m9_documentation.components import evidence_chain as EC\n"
                 "claims = EC.verify_claims(config['claims'])\n"
                 "risks = EC.verify_risk_traceability(config['risks'])\n"
                 "deliv = EC.verify_deliverables(config['deliverables'])\n"
                 "PCT = 100\n"
                 "for label, res, ok, tot in [('一致性核验', claims, 'traceable', 'total'),\n"
                 "                            ('DPIA 风险溯源', risks, 'traced', 'total'),\n"
                 "                            ('交付清单', deliv, 'present', 'total')]:\n"
                 "    print(f'{label:14s} {res[ok]}/{res[tot]} = {res[\"rate\"]*PCT:.1f}%')"),
        ("md", "## 逐条结论的可追溯性\n\n"
               "`nature` 一列必须显式声明性质——**留空会让读者误以为是实测**。"
               "凡涉法结论一律标注「未复核」：本项目不设合规角色。"),
        ("code", "pd.DataFrame(claims['rows'])[['id', 'statement', 'nature',\n"
                 "                              'n_evidence', 'traceable']]"),
        ("md", "## DPIA 风险溯源\n\n"
               "每条风险必须指向具体的实验产出，并给出缓解措施——"
               "只写「可能存在」不算溯源。"),
        ("code", "pd.DataFrame(risks['rows'])[['id', 'risk', 'severity',\n"
                 "                             'has_mitigation', 'traced']]"),
        ("md", "**R2 是本评估最重要的一条：主动方标签被完全推断，目前无有效技术缓解。**\n\n"
               "噪声防护已被实测证伪——把泄露压到 0.67 需要 σ=30，"
               "而那时模型可用性（0.6352）已低于不做联邦的内地单方基线（0.7089）。"
               "**防护到有效时，还不如不做联邦。**"),
        ("md", "## 交付清单"),
        ("code", "pd.DataFrame(deliv['rows'])[['name', 'path', 'status']]"),
        ("md", "## 这套核验防的是什么\n\n"
               "一类很安静的事故：有人删掉或重命名了一个结果文件，"
               "文档里引用它的那条结论就此失去依据，而**七道门禁照样全绿**。\n\n"
               "本核验已接入 `ci/check_evidence_chain.py`，随 `run_all_gates.sh` "
               "每次提交前执行。写这份 notebook 的过程中它就抓到过一次："
               "证据映射里引用了一个不存在的 `funnel_scenarios.csv`，通过率因此掉到 92.3%。"),
    ])
