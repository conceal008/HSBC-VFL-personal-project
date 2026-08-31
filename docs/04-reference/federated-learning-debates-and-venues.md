# 联邦学习：争论、批评性文献与发表地图

> 整理日期：2026-09-01 ｜ 整理者：Claude (Opus 5)
> 定位：[`federated-learning-landscape.md`](federated-learning-landscape.md) 的**补充篇**。
> 前者回答「有什么」，本篇回答「**有什么争议、哪里被高估了、去哪里读和发**」。

## 可信度分档（同前篇）

🟢 本次检索已确认 ｜ 🟡 来自既有知识未逐条核对，引用前须点开 ｜ ⚪ 商业/行业信息可能过时

> **为什么单独做一篇「争论」**：前篇列的是方法与工具，读完容易产生「这个领域已经成熟」的错觉。
> 但联邦学习最值得知道的，恰恰是**它被高估的那几处**——尤其当你要向法务、评审或业务方
> 论证一个方案时，对方问的都是这些问题。

---

## 一、争论一：隐私宣称被高估了吗？

**结论：是，且被高估的方式很具体——「原始数据不出本地」不等于「不泄露个人信息」。**

| 文献 | 核心观点 | 链接 |
|---|---|---|
| 🟢 Breaking Secure Aggregation | **即使只拿到聚合梯度**，恶意服务器仍可推断标签。**cross-silo（参与方少）尤其危险** | [arXiv:2406.15731](https://arxiv.org/abs/2406.15731) |
| 🟢 SoK: Gradient Inversion Attacks in FL | 梯度反演攻击的系统化梳理 | [ACM DL](https://dl.acm.org/doi/10.5555/3766078.3766409) |
| 🟢 Gradient leakage attacks in FL | 综述：梯度泄露可达**像素级还原** | [AI Review 2023](https://link.springer.com/article/10.1007/s10462-023-10550-z) |
| 🟢 In the Pursuit of Privacy: Promises and Predicaments of FL in Healthcare | 医疗场景下 FL 隐私承诺与现实的落差 | [PMC8528445](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8528445/) |
| 🟢 Belt and Braces: When FL Meets DP | **CACM 正刊**讨论：FL 单独不够，必须叠加 DP 才有形式化保证 | [CACM](https://cacm.acm.org/research/belt-and-braces-when-federated-learning-meets-differential-privacy/) |
| 🟢 Emerging Paradigms for Securing FL Systems | 安全范式综述 | [arXiv:2509.21147](https://arxiv.org/pdf/2509.21147) |

**检索到的一句关键判断**（可直接引用的立场）：

> FL 能在不直接暴露原始数据的前提下协同学习，**但它对间接信息推断缺乏严格的隐私保证**。

**这条对任何 FL 方案的意义**：如果你的合规论证建立在「数据不出域所以安全」上，
它在技术上站不住。正确的表述是「原始数据不出本地，中间量的泄露风险另行量化与防护」。

> 本项目已把这条写进术语表：**禁用「数据不出域」，改用「原始数据不出本地」**
> （见 `registry/glossary.yaml`）——理由正是前者掩盖了中间量泄露。

---

## 二、争论二：FL 能自动满足 GDPR / PIPL 吗？

**结论：不能。这是法律界与技术界分歧最大的一处。**

| 文献 | 核心观点 | 链接 |
|---|---|---|
| 🟢 Privacy preservation in FL: An insightful survey **from the GDPR perspective** | 从 GDPR 视角系统审视 FL，指出多处不自动合规 | [Computers & Security](https://www.sciencedirect.com/science/article/pii/S0167404821002261) |
| 🟢 The potential of FL for public health: qualitative analysis of **GDPR compliance** | **「FL 系统并非天然符合 GDPR」**；且指出 FL 增加了「在源头检查数据」的难度 | [PMC11484284](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11484284/) |
| 🟢 FL, PETs, and Data Protection Laws in Medical Research: Scoping Review | 医学研究场景的法规范围综述 | [PMC10131784](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10131784/) |
| 🟢 Can Federated Learning Solve AI's Data Privacy Problem? **A Legal Analysis** | 法学期刊视角（Rutgers Law Record, 2025） | [lawrecord.com](https://lawrecord.com/2025/05/29/can-federated-learning-solve-ais-data-privacy-problem-a-legal-analysis/) |
| 🟢 FL's Consent Crisis | **同意机制**在 FL 下的困难：如何取得、如何撤回、撤回后模型怎么办 | [secureprivacy.ai](https://secureprivacy.ai/blog/consent-orchestration-federated-learning) |
| 🟢 FL for GDPR Compliance | 反向视角：用 FL 帮助达成 GDPR 合规 | [Springer](https://link.springer.com/chapter/10.1007/978-3-031-78925-0_38) |
| ⚪ FL & Global Data Sovereignty Compliance | 厂商视角（Duality），可作产业口径参考 | [dualitytech.com](https://dualitytech.com/blog/federated-learning-in-meeting-global-data-sovereignty-regulations/) |

**检索到的两句关键判断**：

> 「FL 系统并非天然符合 GDPR」——技术本身不自动带来法律合规。
>
> 「当 FL 跨越多个法域时，冲突的隐私法会造成不可能完成的合规局面」——
> 一个系统可能需要同时满足 GDPR 的明示同意与 CCPA 的选择退出标准。

**对跨境场景的直接意义**：这正是本项目 M0 反复遇到的问题——
内地 PIPL 与香港 PDPO 是**两套独立要求**，不能靠"我们用了联邦学习"一句话打发。
两地对同一概念还用不同法定术语（个人信息 / 个人资料），混用会让论证失效。

---

## 三、争论三：研究与落地之间有多大鸿沟？

**结论：非常大，而且有可量化的证据。**

| 文献 | 关键数据 | 链接 |
|---|---|---|
| 🟢 FL in healthcare: review of the **deployment gap** between simulated and real-world | **筛选 1,338 篇论文，772 篇入选，其中仅 25 篇（3.2%）是真实世界的 FL 部署** | [EPJ ST 2026](https://link.springer.com/article/10.1140/epjs/s11734-026-02475-9) |
| 🟢 A study on performance limitations in FL | 性能上限的系统研究 | [arXiv:2501.03477](https://arxiv.org/pdf/2501.03477) |
| 🟢 Exploring ML Models for FL: Approaches, Performance, and **Limitations** | 方法与局限的综述 | [arXiv:2311.10832](https://arxiv.org/pdf/2311.10832) |
| 🟢 Benchmarking FL in Edge Computing: Systematic Review | 评测设置不一致导致**难以公平比较** | [arXiv:2603.08735](https://arxiv.org/pdf/2603.08735) |
| 🟢 ATR-Bench: Adaptation, Trust, Reasoning | 新一代综合基准 | [arXiv:2505.16850](https://arxiv.org/pdf/2505.16850) |
| 🟢 From Data Heterogeneity to Convergence: 数据中心视角综述 | 从数据而非算法角度重审 FL | [arXiv:2606.10595](https://arxiv.org/pdf/2606.10595) |

**「3.2%」这个数字值得单独记住。** 它说明绝大多数 FL 论文停留在模拟环境，
真实部署的经验极其稀缺。**读一篇 FL 论文时先问：它是模拟还是真跑过？**

**检索到的两条方法学批评**：

> 许多 FL 研究缺乏透明的实验设置与开源代码，数据集与模型的不一致使公平比较困难，**损害可复现性**。
>
> 简化的数据切分往往只控制标签比例，**忽略了真实部署中耦合的异质性**——
> 这类差距会**高估准确率与稳定性**，并错误刻画收敛行为。

> **对本项目的直接印证**：这正是框架 v2 要求做 S1–S4 四种切分协议、
> 并计算「S1/S4 增益比值」作为**模拟纵向切分高估效应**量化证据的原因。
> 上面这条批评说明该设计不是过度谨慎，而是回应了领域内公认的问题。

---

## 四、争论四：贡献度量与激励机制真的可行吗？

**结论：理论成熟（Shapley 值），但实践脆弱。**

| 文献 | 观点 | 链接 |
|---|---|---|
| 🟢 **On the Fragility of Contribution Evaluation in FL** | **直接批评**：贡献度量本身是脆弱的 | [arXiv:2509.19921](https://arxiv.org/html/2509.19921) |
| 🟢 A Comprehensive Survey of Incentive Mechanism for FL | 激励机制综述：Stackelberg 博弈、拍卖、合约、Shapley、区块链、强化学习 | [arXiv:2106.15406](https://arxiv.org/pdf/2106.15406) |
| 🟢 Incentivizing Federated Learning | 激励设计 | [arXiv:2205.10951](https://arxiv.org/pdf/2205.10951) |
| 🟢 Incentive-Based FL: Architectural Elements and Future Directions | 架构视角 | [arXiv:2510.14208](https://arxiv.org/pdf/2510.14208) |
| 🟢 A Fairness-aware Incentive Scheme for FL | AAAI/ACM AIES | [ACM DL](https://dl.acm.org/doi/10.1145/3375627.3375840) |
| 🟢 A Comprehensive Study of Shapley Value in Data Analytics | Shapley 值在数据分析中的全面研究 | [arXiv:2412.01460](https://arxiv.org/pdf/2412.01460) |
| 🟢 Fairness-Aware FL with Trajectory Shapley Value | 沿优化轨迹评估贡献 | [arXiv:2605.30336](https://arxiv.org/html/2605.30336v1) |

**检索到的商业化困境描述**（很实在）：

> 数据方共享本地数据以构建能产生收益的联邦模型，**但若竞争对手加入同一联邦，参与方可能承担显著成本**；
> 且模型训练与商业化需要时间，**联邦积累到足够预算再支付参与方之间存在时间差**——
> 成本问题与「贡献与回报的暂时错配」尚未被充分解决。

---

## 五、争论五：效率代价被说清楚了吗？

**检索到的三条具体判断**：

> **MPC** 因大量消息交换而低效、不实用，并引入显著延迟，限制 FL 性能。
>
> **当前的同态加密方案缺乏实用 FL 所需的效率与可扩展性**，对资源受限的客户端尤甚。
>
> **差分隐私**在样本量小时会遭遇低信噪比问题。

以及一句概括性的：

> **FL 是一门权衡的艺术**——在模型精度、数据隐私、计算效率、通信成本与可扩展性之间。

**这对方案设计的意义**：任何声称「既保护隐私又不损失精度又不增加成本」的 FL 方案，
不是在某个维度上做了隐含妥协，就是没测过。**报告结果时把三个维度一起报**，
只报 AUC 的对比表是不完整的。

---

## 六、争论六：产业化的真实障碍

| 主题 | 检索到的判断 | 来源 |
|---|---|---|
| **互联互通** | 大规模商业化的主要挑战是**多个隐私计算平台之间互不互通**：厂商各自推出平台，技术路线、认证体系、算子算法不一致，缺乏统一标准，数据难以跨平台流动 | ⚪ 行业分析 |
| **数据要素市场** | 2024 年多项政策落地（公共数据资源、可信数据空间、企业数据资源），全国数据市场交易规模估计超 1,600 亿元、同比增长 30%+ | ⚪ [前瞻产业研究院](https://www.qianzhan.com/analyst/detail/220/250324-40a74d2b.html) |
| **成本下降** | 「安全计算平价时代」：数据安全成本从"高端奢侈品"变为"经济标配" | ⚪ [知乎行业展望](https://zhuanlan.zhihu.com/p/19535338799) |
| **联邦学习与数据交易** | FL 作为数据交易商业化的隐私计算技术路径 | ⚪ [赛迪网](https://www.ccidnet.com/2023/1227/10617686.shtml) |

> ⚠️ 本节全部为 ⚪ 档：行业分析与自媒体口径，**数字未经独立核实**，仅作产业背景参考，不得引用为事实。

---

## 七、来自生产一线的反思（本篇最值得读的一篇）

🟢 **[Federated Learning in Practice: Reflections and Projections](https://arxiv.org/abs/2410.08892)**（2024）

Google 等生产系统十年经验的反思，**不是论文综述而是从业者的自我审视**：

- FL 系统已扩展到**数百万设备**，并提供有意义的差分隐私保证
- Google **Gboard 的所有神经网络语言模型现已全部用 FL 训练，且自 2022 年起带形式化 DP 保证**
- **但仍存在关键挑战**：**服务器端 DP 保证的可验证性**、异构设备间的训练协调——这些限制了更广泛的采用
- 新趋势正在冲击传统 FL 框架：大型（多模态）模型，以及**训练、推理与个性化之间界限的模糊化**

相关的一手材料：

- 🟢 [Google Research Blog：Federated Learning（2017 首篇）](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html?m=1)
- 🟢 [Google Research：Advances in private training for production on-device language models](https://research.google/blog/advances-in-private-training-for-production-on-device-language-models/)
- 🟢 [英美联合博客系列：Privacy-Preserving Federated Learning](https://rtau.blog.gov.uk/2023/12/07/the-uk-us-blog-series-on-privacy-preserving-federated-learning-introduction/) —— **政府视角**（英国 RTAU + 美国 NIST），少见

---

## 八、补充论文与综述（按主题）

### 8.1 综合性综述

| 文献 | 链接 |
|---|---|
| 🟢 FL: A Survey of Core Challenges, Current Methods, and Opportunities（MDPI Computers 2026） | [MDPI](https://www.mdpi.com/2073-431X/15/3/155) |
| 🟢 Deep federated learning: systematic review of methods, applications, challenges（Frontiers 2025） | [Frontiers](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1617597/full) |
| 🟢 FL: Applications, challenges and future directions | [arXiv:2205.09513](https://arxiv.org/pdf/2205.09513) |
| 🟢 Systematic Literature Review on FL: From A **Model Quality** Perspective | [arXiv:2012.01973](https://arxiv.org/pdf/2012.01973) |
| 🟢 A Systematic Literature Review on **Client Selection** in FL | [arXiv:2306.04862](https://arxiv.org/pdf/2306.04862) |
| 🟢 FL: Overview, strategies, applications, tools and future directions | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2405844024141680) |
| 🟡 **FL: Challenges, Methods, and Future Directions**（Li Tian et al., IEEE Signal Processing Magazine 2020）——最常被引的入门综述之一 | 🟡 arXiv:1908.07873 |

### 8.2 纵向联邦补充

| 文献 | 链接 |
|---|---|
| 🟢 **Vertical Federated Learning: A Structured Literature Review** | [arXiv:2212.00622](https://arxiv.org/pdf/2212.00622) |
| 🟢 A Survey of Privacy Threats and Defense in VFL: Model Life Cycle Perspective | [arXiv:2402.03688](https://arxiv.org/pdf/2402.03688) |
| 🟢 Label Leakage in VFL: A Survey（IJCAI 2024） | [IJCAI](https://www.ijcai.org/proceedings/2024/0902.pdf) |

### 8.3 非 IID 与数据异质性

| 文献 | 链接 |
|---|---|
| 🟢 **A Thorough Assessment of the Non-IID Data Impact in FL** | [arXiv:2503.17070](https://arxiv.org/html/2503.17070v2) |
| 🟢 Non-IID data in FL: Survey with Taxonomy, Metrics, Methods, Frameworks | [arXiv:2411.12377](https://arxiv.org/html/2411.12377v2) |
| 🟢 FL on Non-IID Data Silos: An Experimental Study（ICDE 2022） | [arXiv:2102.02079](https://arxiv.org/pdf/2102.02079) |
| 🟢 Benchmarking Data Heterogeneity Evaluation Approaches for Personalized FL | [arXiv:2410.07286](https://arxiv.org/pdf/2410.07286) |

### 8.4 隐私-效用权衡

| 文献 | 链接 |
|---|---|
| 🟢 Mitigating Privacy-Utility Trade-off in Decentralized FL via **f-Differential Privacy** | [arXiv:2510.19934](https://arxiv.org/pdf/2510.19934) |
| 🟢 A Review of Privacy-preserving FL for the Internet-of-Things | [arXiv:2004.11794](https://arxiv.org/pdf/2004.11794) |
| 🟢 Trusted AI in Multi-agent Systems: Privacy and Security for Distributed Learning | [arXiv:2202.09027](https://arxiv.org/pdf/2202.09027) |
| 🟢 RFLPA: Robust FL against Poisoning with Secure Aggregation（NeurIPS 2024） | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/bcbdc25dc4f0be5ae8ac07232df6e33a-Paper-Conference.pdf) |

### 8.5 应用领域与新方向

| 文献 | 链接 |
|---|---|
| 🟢 Advancing oncology with FL（乳腺/肺/前列腺癌系统综述） | [medRxiv](https://www.medrxiv.org/content/10.1101/2024.08.08.24311681.full.pdf) |
| 🟢 Application of FL in manufacturing | [arXiv:2208.04664](https://arxiv.org/pdf/2208.04664) |
| 🟢 Survey of FL Models for **Spatial-Temporal Mobility** Applications | [arXiv:2305.05257](https://arxiv.org/pdf/2305.05257) |
| 🟢 Towards Privacy-Preserving Data-Driven **Education** | [arXiv:2503.13550](https://arxiv.org/pdf/2503.13550) |
| 🟢 **Quantum federated learning**: comprehensive literature review | [Springer QMI](https://link.springer.com/article/10.1007/s42484-025-00292-2) |
| 🟢 Secure Federated **Submodel** Learning（阿里，大规模推荐场景） | [arXiv:1911.02254](https://arxiv.org/pdf/1911.02254) |
| 🟢 Group Knowledge Transfer: FL of Large CNNs at the Edge | [arXiv:2007.14513](https://arxiv.org/pdf/2007.14513) |
| 🟢 FedSVD: Adaptive Orthogonalization for Private FL with LoRA | [arXiv:2505.12805](https://arxiv.org/pdf/2505.12805) |

---

## 九、发表地图：去哪读、去哪投

> 🟢 检索确认的一条重要事实：**目前没有专门的联邦学习会议或期刊**，
> 研究分散在通用 ML / AI / 系统 / 安全 / 通信各类场所——
> 有统计称 FL 论文覆盖 **42 个会议与 21 个期刊**。

### 9.1 会议（按方向）

| 方向 | 会议 |
|---|---|
| 机器学习理论与算法 | NeurIPS · ICML · ICLR · AISTATS |
| 人工智能 | AAAI · IJCAI |
| 数据挖掘 / 数据库 | KDD · ICDE · WWW · CIKM |
| 计算机视觉 | CVPR · ICCV · ECCV |
| **安全与隐私**（攻防类必看） | IEEE S&P · ACM CCS · USENIX Security · NDSS · **PETS** |
| **系统**（工程落地类） | MLSys · OSDI · SOSP · EuroSys · ATC |
| 多媒体 | ACM MM |

### 9.2 期刊

| 期刊 | 说明 |
|---|---|
| 🟡 ACM TIST（Transactions on Intelligent Systems and Technology） | 杨强团队的 FL 三分类综述发表处 |
| 🟢 ACM Computing Surveys | VFL 综述（武大 MARS 组，2025） |
| 🟡 IEEE TIFS（Information Forensics & Security） | 隐私攻防类的主要期刊 |
| 🟡 IEEE TKDE / TNNLS / TPAMI | 算法与表示学习 |
| 🟢 IEEE Internet of Things Journal · IEEE Access | 检索确认为 FL 的常见发表地（IoT/边缘方向尤多） |
| 🟢 Computers & Security | GDPR 视角综述发表处 |
| 🟢 Communications of the ACM | 面向广泛读者的立场文章（Belt and Braces） |
| 🟢 Nature/Springer 系（EPJ ST、Springer AI Review、Quantum Machine Intelligence） | 综述与跨学科 |

### 9.3 Workshop 与门户

| 资源 | 说明 | 链接 |
|---|---|---|
| 🟢 **The Federated Learning Portal** | 领域门户：会议、workshop、教程集散地 | [federated-learning.org](https://federated-learning.org/fl@fm-ijcai-2024/) |
| 🟢 FL@FM 系列 workshop | 依附于 IJCAI / NeurIPS 等，**联邦 + 基础模型**方向 | 同上 |
| 🟢 WikiCFP: Federated Learning | 征稿信息聚合 | [wikicfp](http://www.wikicfp.com/cfp/call?conference=federated+learning) |

### 9.4 论文追踪仓库

| 仓库 | 说明 | 链接 |
|---|---|---|
| 🟢 **federated-learning-updated-papers** | 顶会 FL 论文持续更新 | [github](https://github.com/mtuann/federated-learning-updated-papers) |
| 🟢 **awesome-federated-learning**（weimingwill） | 博客、视频、论文、软件全收；**单独维护了 conferences.md 与 journals.md** | [github](https://github.com/weimingwill/awesome-federated-learning) · [conferences.md](https://github.com/weimingwill/awesome-federated-learning/blob/master/conferences.md) · [journals.md](https://github.com/weimingwill/awesome-federated-learning/blob/master/journals.md) |

---

## 十、这些争论如何命中本项目

| 争论 | 对本项目的直接影响 | 我们已经做了什么 |
|---|---|---|
| **隐私宣称被高估**（cross-silo 聚合梯度仍泄露标签） | 本项目正是**两方 cross-silo**，是该攻击最有利的场景 | 术语表已禁用「数据不出域」；M7 须把该攻击纳入威胁模型 |
| **FL 不自动合规** | 直接对应 M0 的全部工作——PIPL 与 PDPO 是两套独立要求 | 八类跨境对象逐条定性、三条路线推演、23 个监管质询 |
| **跨法域冲突** | 内地「个人信息」/ 香港「个人资料」是两个法定概念 | 术语表按法域分列，禁止混用 |
| **3.2% 才是真实部署** | 提醒我们：读到的 VFL 效果数字大多来自模拟 | 框架已要求 S1–S4 切分协议与「模拟纵向切分高估效应」量化 |
| **评测不可复现、切分过于简化** | 印证 S1/S4 增益比值这一设计的必要性 | 已写入 M1 判据 |
| **贡献度量脆弱** | 若日后要做两方收益分配，不能直接照搬 Shapley | 尚未涉及，M7 的「贡献伪造/搭便车」攻击面已列入清单 |
| **MPC/HE 效率不足、DP 小样本信噪比低** | **与本项目的小样本困境叠加**：N_eff 只有几千人时，DP 几乎不可用 | 已在 `escalation_M0-001.md` 中体现为可验证性问题 |
| **FL 是权衡的艺术** | 支持框架的五级基线阶梯设计——L1 是真正的竞争对手 | M5 已写入，且 S0.4 发现 L1 的合规成本不比 L3 低一个数量级 |

**一句话**：本项目 M0 得出的「悲观」结论——三条路线可用样本均不达标、
可验证性与轻量出境路径不可兼得——**与领域内的批评性文献是一致的，不是我们做错了什么。**

---

## 十一、局限

1. **本篇偏批评视角**，是有意为之（前篇已覆盖「有什么」）。**不代表 FL 无价值**——
   Gboard 的生产实践证明它在 cross-device 场景确实成立。
2. 🟡 条目未逐条核对；⚪ 条目（尤其第六节的产业数字）**未经独立核实，不得引用为事实**。
3. 中文产业观察部分来源为行业媒体与自媒体，口径与统计方法未知。
4. 会议/期刊列表是**经验性的**，不是权威排名；投稿前请以各刊物当年的征稿范围为准。
5. 时效性：2026 年 9 月快照。
