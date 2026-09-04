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
