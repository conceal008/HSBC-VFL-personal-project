# 联邦学习全景图：论文 · 算法 · 框架 · 平台 · 基准 · 标准

> 整理日期：2026-08-31 ｜ 整理者：Claude (Opus 5)
> 范围：**不限于本项目方向**，横向/纵向/迁移、隐私攻防、工程框架、大模型联邦、产业落地、标准法规全收
> 用途：技术选型的参考底稿与文献起点，不是选型结论

## 先看这一段：怎么读，以及可信度分档

这份文档有 200+ 个条目，**可信度不是均匀的**，使用前请按下面的分档判断：

| 标记 | 含义 |
|---|---|
| 🟢 | 本次检索**已确认**链接或出处（官方站点、检索结果中直接出现） |
| 🟡 | 来自既有知识，arXiv 编号与仓库路径**未逐条打开核对**——引用前请点开确认 |
| ⚪ | 商业产品或中文生态，**信息可能过时**，以官网为准 |

**arXiv 编号最容易记错。** 凡标 🟡 的编号，写进正式材料前务必点开核对——
这与本项目 `AGENTS.md` 铁律 1（数字必须可追溯）是同一条纪律。

---

## 一、起点：先读这三篇

| 文献 | 为什么是它 | 链接 |
|---|---|---|
| **Communication-Efficient Learning of Deep Networks from Decentralized Data**（McMahan et al., AISTATS 2017） | **FedAvg 原始论文**，联邦学习的起点。所有后续算法都在回答它留下的问题 | 🟡 arXiv:1602.05629 |
| **Advances and Open Problems in Federated Learning**（Kairouz et al., 2019/2021） | 58 位作者、100+ 页的领域地图。**想知道"这个方向有没有人做过"，先查它的目录** | 🟡 arXiv:1912.04977 |
| **Federated Machine Learning: Concept and Applications**（Yang Qiang et al., ACM TIST 2019） | 提出**横向/纵向/联邦迁移**三分类，中文语境下被引最多；纵向联邦这个词的来源 | 🟡 arXiv:1902.04885 |

补充两篇综述：

- 🟢 [A Survey on Federated Learning Systems: Vision, Hype and Reality](https://arxiv.org/pdf/1907.09693) — 从**系统**角度而非算法角度看 FL
- 🟢 [Non-IID data in FL: A Survey with Taxonomy, Metrics, Methods, Frameworks](https://arxiv.org/html/2411.12377v2)（2024） — 非独立同分布问题的全景

---

## 二、算法谱系

### 2.1 横向联邦：解决"客户端数据分布不同"

| 算法 | 核心想法 | 链接 |
|---|---|---|
| **FedAvg** | 本地多步 SGD 后平均参数。简单到不可思议，也是所有对比的基线 | 🟡 arXiv:1602.05629 |
| **FedProx** | 本地目标加 L2 近端项，限制本地模型偏离全局模型的距离 | 🟡 arXiv:1812.06127 |
| **SCAFFOLD** | 用**控制变量**（control variates）做方差缩减，对抗 client drift | 🟡 arXiv:1910.06378 |
| **FedNova** | 归一化本地更新步数，解决客户端本地 epoch 数不一致导致的目标漂移 | 🟡 arXiv:2007.07481 |
| **FedOpt / FedAdam / FedYogi** | 服务器端用自适应优化器（Adam/Yogi）而非简单平均 | 🟡 arXiv:2003.00295 |
| **MOON** | 把**对比学习**引入 FL：拉近本地表示与全局表示，推远与上一轮本地表示 | 🟡 arXiv:2103.16257 |
| **FedDyn** | 动态正则项，理论上使本地最优解与全局最优解一致 | 🟡 arXiv:2111.04263 |

🟢 实现参考：[Federated-Learning-Non-IID](https://github.com/meng1103/Federated-Learning-Non-IID)（PyTorch 实现 FedAvg/FedProx/MOON/SCAFFOLD/FedDyn）

### 2.2 个性化联邦：不追求一个全局模型

| 方法 | 核心想法 | 链接 |
|---|---|---|
| **Per-FedAvg** | 用 MAML 的思路，学一个"容易被微调"的初始模型 | 🟡 arXiv:2002.07948 |
| **pFedMe** | Moreau envelope 把个性化模型优化与全局模型学习解耦 | 🟡 arXiv:2006.08848 |
| **Ditto** | 多任务学习目标，个性化与鲁棒性同时获得 | 🟡 arXiv:2012.04221 |
| **FedBN** | 只有 BatchNorm 层保留在本地不聚合——极简但有效 | 🟡 arXiv:2102.07623 |
| **FedRep / LG-FedAvg** | 共享表示层 + 本地分类头 | 🟡 arXiv:2102.07078 |

🟢 基准：[pFL-Bench](https://arxiv.org/pdf/2206.03655)（个性化 FL 的综合评测）

### 2.3 纵向联邦（VFL）：特征分散在不同机构 —— **本项目主线**

| 方法 | 核心想法 | 链接 |
|---|---|---|
| **SecureBoost** | 纵向场景的无损 GBDT。同态加密保护梯度直方图，工业界最常用的 VFL 树模型 | 🟡 arXiv:1901.08755 |
| **SecureBoost+** | SecureBoost 的性能优化版，面向大规模 | 🟡 arXiv:2110.10927 |
| **SplitNN / Split Learning** | 神经网络按层切分到两方，交换中间层表示而非原始数据 | 🟡 arXiv:1812.00564 |
| **FedBCD** | 纵向场景的块坐标下降，减少通信轮次 | 🟡 arXiv:1912.11187 |
| **FDML** | 特征分布式的机器学习框架 | 🟡 arXiv:1806.06145 |

**必读综述与资源**

- 🟢 [Vertical Federated Learning for Effectiveness, Security, Applicability: A Survey](https://github.com/shentt67/VFL_Survey)（ACM Computing Surveys 2025，武汉大学 MARS 组）
- 🟢 [A Survey of Privacy Threats and Defense in VFL: From Model Life Cycle Perspective](https://arxiv.org/pdf/2402.03688)
- 🟢 [Label Leakage in Vertical Federated Learning: A Survey](https://www.ijcai.org/proceedings/2024/0902.pdf)（IJCAI 2024）— **标签泄露是 VFL 最核心的隐私问题**
- 🟢 [awesome-vertical-federated-learning](https://github.com/ngc436/awesome-vertical-federated-learning) — VFL 专门的资源列表
- 🟢 [Eliminating Label Leakage in Tree-Based VFL](https://arxiv.org/pdf/2307.10318)
- 🟢 [VFL Framework for Horizontally Partitioned Labels](https://arxiv.org/pdf/2106.10056)

### 2.4 无对齐样本的利用：自监督与半监督 VFL

现实中交集往往很小，交集外的样本能不能用？

- **FedHSSL** — 联邦混合自监督学习，用本地非对齐样本预训练编码器 🟡
- **Semi-VFL / VFL with unaligned data** — 半监督思路利用非对齐样本 🟡

> ⚠️ 概念提醒：用非对齐样本的**正确定位**是「改善本地表示质量」，不是「扩大训练集提 AUC」。
> 这两种表述的评估口径完全不同（见本项目 `docs/00-framework/` M5）。

### 2.5 联邦树模型

| 项目 | 说明 | 链接 |
|---|---|---|
| **FedTree** | 新加坡国立 Xtra 组，支持横向与纵向的联邦 GBDT | 🟢 [github.com/Xtra-Computing/FedTree](https://github.com/Xtra-Computing/FedTree) |
| **Federated XGBoost** | XGBoost 官方与社区的联邦扩展 | 🟡 |
| **SecureBoost（FATE 内置）** | 见 4.2 | 🟢 |

### 2.6 异步与系统调度

| 方法 | 解决什么 | 链接 |
|---|---|---|
| **FedAsync** | 异步聚合，避免慢客户端拖累整轮 | 🟡 arXiv:1903.03934 |
| **FedBuff** | 带缓冲的异步聚合，兼容安全聚合 | 🟡 arXiv:2106.06639 |
| **Oort** | 客户端选择：优先选"数据有用且算得快"的客户端（OSDI 2021） | 🟡 |
| **Papaya** | Meta 的生产级异步 FL 系统 | 🟡 arXiv:2111.04877 |

### 2.7 通信压缩

| 方法 | 手段 | 链接 |
|---|---|---|
| **Deep Gradient Compression (DGC)** | 梯度稀疏化，99.9% 稀疏度仍保精度 | 🟡 arXiv:1712.01887 |
| **FedPAQ** | 周期平均 + 量化 | 🟡 arXiv:1909.13014 |
| **QSGD / signSGD** | 梯度量化的经典基线 | 🟡 |

---

## 三、隐私与安全：这是 FL 最热也最容易做错的部分

### 3.1 隐私保护技术

| 技术 | 在 FL 中的角色 | 代价 |
|---|---|---|
| **差分隐私（DP）** | 加噪声换形式化保证。**报告结果必须给 ε、δ、邻接定义与组合方式**，只写"加了 DP"无意义 | 精度下降；多轮组合后预算耗尽 |
| **安全聚合（Secure Aggregation）** | 服务器只能看到聚合结果，看不到单个客户端更新（Bonawitz et al., CCS 2017） | 通信开销；掉线处理复杂 |
| **同态加密（HE）** | Paillier（加法同态）用于梯度聚合；CKKS（近似浮点）用于更复杂运算 | 计算开销大，Paillier 尤甚 |
| **秘密分享 / MPC** | 双方各持份额，任何一方看不到明文 | 通信轮次多 |
| **可信执行环境（TEE）** | SGX/TrustZone 里做明文计算 | 依赖硬件信任；侧信道风险 |

🟢 综述：[Privacy-Preserving Aggregation in Federated Learning: A Survey](https://arxiv.org/pdf/2203.17005)

### 3.2 攻击：知道它们能做到什么，才知道防护够不够

**梯度反演 / 数据重构**

- 🟡 **Deep Leakage from Gradients (DLG)** — arXiv:1906.08935，从梯度像素级还原训练图像
- 🟡 **iDLG** — arXiv:2001.02610，改进版，先恢复标签再恢复数据
- 🟡 **Inverting Gradients** — arXiv:2003.14053，证明大批量下依然可反演
- 🟢 [SoK: Gradient Inversion Attacks in Federated Learning](https://dl.acm.org/doi/10.5555/3766078.3766409)
- 🟢 [A Survey on Gradient Inversion: Attacks, Defenses and Future Directions](https://www.ijcai.org/proceedings/2022/0791.pdf)（IJCAI 2022）
- 🟢 [Gradient leakage attacks in federated learning](https://link.springer.com/article/10.1007/s10462-023-10550-z)（AI Review 2023）

**安全聚合也不是万能的**

- 🟢 [Breaking Secure Aggregation: Label Leakage from Aggregated Gradients](https://arxiv.org/abs/2406.15731) — **即使只拿到聚合梯度，恶意服务器仍可推断标签**。cross-silo（参与方少）场景尤其危险
- 🟢 [Strengthening Privacy in Robust FL through Secure Aggregation](https://www.ndss-symposium.org/wp-content/uploads/aiscc2024-12-paper.pdf)（NDSS 2024）

**VFL 特有攻击**

- 标签推断（从梯度符号/范数推断标签）— 见 3.1 的 IJCAI 2024 综述
- 特征重构（嵌入反演）
- 🟢 [Unsplit: model inversion, model stealing, label inference against split learning](https://arxiv.org/pdf/2410.09125) 相关方向
- 🟢 [Training on Fake Labels: Mitigating Label Leakage in Split Learning](https://arxiv.org/pdf/2410.09125)

**其他攻击面**

- **成员推断**（某人是否在训练集中）
- **属性推断**（推断未共享的敏感属性）
- **投毒与后门**：🟡 DBA (Distributed Backdoor Attack, ICLR 2020)；🟢 [When the Aggregator Cheats: Data-Free Backdoors in Federated LLM-based QA](https://arxiv.org/pdf/2606.27511)
- **搭便车（free-rider）**：不贡献数据却分享模型

### 3.3 防御与鲁棒聚合

| 方法 | 思路 | 链接 |
|---|---|---|
| **Krum / Multi-Krum** | 选与其他更新最接近的那个，排除离群 | 🟡 NeurIPS 2017 |
| **Trimmed Mean / Median** | 逐维去极值后平均 | 🟡 |
| **Bulyan** | 先 Krum 选一批，再 trimmed mean 聚合 | 🟡 |
| **FLTrust** | 服务器持一个干净根数据集，用余弦相似度识别恶意更新 | 🟡 arXiv:2012.13995 |
| **RFLPA** | 兼顾鲁棒性与隐私的聚合（NeurIPS 2024） | 🟢 [论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/bcbdc25dc4f0be5ae8ac07232df6e33a-Paper-Conference.pdf) |
| **梯度扰动 / Gradients Stand-in** | 用替身梯度防深度泄露 | 🟢 [arXiv:2410.08734](https://arxiv.org/pdf/2410.08734) |

🟢 政策与攻防全景：[The Federation Strikes Back: A Survey of FL Privacy Attacks, Defenses, Applications, and Policy Landscape](https://arxiv.org/pdf/2405.03636)

---

## 四、框架与平台

### 4.1 通用开源框架（横向为主）

| 框架 | 主导方 | 特点 | 链接 |
|---|---|---|---|
| **Flower** | Flower Labs / 剑桥 | **框架无关**（PyTorch/TF/JAX 都能接），样板代码少，Python 原生团队的默认选择 | 🟢 [flower.ai](https://flower.ai) · [github](https://github.com/adap/flower) |
| **NVIDIA FLARE** | NVIDIA | 企业级：内置安全聚合、管理控制台、审计追踪；2.6 版加了流式模型传输，带宽降 30–60% | 🟢 [github](https://github.com/NVIDIA/NVFlare) · [论文 arXiv:2210.13291](https://arxiv.org/pdf/2210.13291v2/1000) |
| **OpenFL** | Intel → Linux Foundation | 面向医疗，Intel 硬件优化 | 🟢 [github](https://github.com/securefederatedai/openfl) |
| **TensorFlow Federated (TFF)** | Google | 模拟研究强；绑定 TF 运行时 | 🟢 [tensorflow.org/federated](https://www.tensorflow.org/federated) |
| **PySyft** | OpenMined | **形式化隐私保证最强**（MPC + HE），代价是计算开销 | 🟢 [github](https://github.com/OpenMined/PySyft) |
| **FedML / TensorOpera** | FedML Inc. | 跨框架兼容好，从研究到生产一条链 | 🟢 [github](https://github.com/FedML-AI/FedML) |
| **FederatedScope** | 阿里达摩院 | 事件驱动架构，支持横向+纵向+图联邦 | 🟢 [github](https://github.com/alibaba/FederatedScope) |
| **FedLab** | — | 轻量研究框架 | 🟢 [github](https://github.com/SMILELab-FL/FedLab) |
| **FLUTE** | Microsoft | 大规模模拟 | 🟡 |
| **FLSim** | Meta | 移动端场景模拟 | 🟡 |
| **IBM FL** | IBM | 企业向 | 🟡 |
| **Substra** | Owkin / Linux Foundation | 医疗多中心，强调可追溯与审计 | 🟢 [github](https://github.com/Substra) |
| **pfl-research** | Apple | 私有联邦学习的**高速模拟**框架 | 🟢 [arXiv:2404.06430](https://arxiv.org/pdf/2404.06430) |
| **EasyFL** | — | 低门槛 | 🟡 |

> 🟢 **横向 vs 纵向支持**（本次检索确认）：PySyft、FATE、FedML、Flower、FedLearner、PaddleFL、FederatedScope **同时支持横向与纵向**；
> TFF、OpenFL、IBM FL、FLARE、FLSim、FLUTE、FedLab、EasyFL **只支持横向**。选型时这条最要紧。

🟢 框架对比研究：
[Comparative analysis of open-source FL frameworks](https://link.springer.com/article/10.1007/s13042-024-02234-z)（Springer 2024）·
[A Comprehensive Comparison of FL Frameworks](https://elib.dlr.de/215928/1/In%20Ver%C3%B6ffentlichung-%20A%20Comprehensive%20Comparison%20of%20Federated%20Learning%20Frameworks.pdf)（DLR）·
[UniFed: All-In-One FL Platform](https://arxiv.org/pdf/2207.10308)

### 4.2 中国生态：隐私计算平台

| 平台 | 主导方 | 特点 | 链接 |
|---|---|---|---|
| **FATE** | 微众银行 | **工业级 VFL 的事实标准**，SecureBoost / Hetero LR 的原产地；配套 KubeFATE 做 K8s 部署 | 🟢 [github](https://github.com/FederatedAI/FATE) |
| **SecretFlow（隐语）** | 蚂蚁集团 | **技术栈最全**：内置 MPC/TEE/HE 多种"隐私计算虚拟设备"，含 SPU（安全处理单元）、HEU（同态加密单元）、PSI 库；不是单一路线的框架 | 🟢 [github.com/secretflow](https://github.com/secretflow) · [gitee 镜像](https://gitee.com/secretflow/secretflow) |
| **PaddleFL** | 百度 | 基于 PaddlePaddle | 🟢 [github](https://github.com/PaddlePaddle/PaddleFL) |
| **Fedlearner** | 字节跳动 | 2019 起为广告投放场景做的神经网络 VFL，2020 开源；已落地电商、互金、教育 | 🟢 [github](https://github.com/bytedance/fedlearner) |
| **Rosetta** | 矩阵元 | 基于 TensorFlow 的隐私计算 | 🟡 |
| **Primihub** | 原语科技 | 开源隐私计算平台 | 🟡 |

⚪ **商业厂商**（信息以官网为准）：富数科技、华控清交、洞见科技、翼方健数、同盾科技、星环科技。

> ⚠️ **本项目的历史决议**：2026-07-29 会议明确**排除 FATE**（已停止更新），优先调研 **SecretFlow**。
> 见项目根目录 `AGENTS.md` §2.2。

### 4.3 密码学与隐私底座（不是 FL 框架，但 FL 依赖它们）

| 库 | 用途 | 链接 |
|---|---|---|
| **CrypTen** | Meta，PyTorch 上的 MPC | 🟢 [github](https://github.com/facebookresearch/CrypTen) |
| **TF-Encrypted** | TensorFlow 上的加密计算 | 🟢 [github](https://github.com/tf-encrypted/tf-encrypted) |
| **MP-SPDZ** | 通用 MPC 协议实现集合，学术界标准工具 | 🟢 [github](https://github.com/data61/MP-SPDZ) |
| **Microsoft SEAL** | 同态加密（BFV/CKKS） | 🟢 [github](https://github.com/microsoft/SEAL) |
| **OpenFHE** | 全同态加密开源库 | 🟢 [github](https://github.com/openfheorg/openfhe-development) |
| **TenSEAL** | SEAL 的 Python 张量封装 | 🟢 [github](https://github.com/OpenMined/TenSEAL) |
| **Concrete-ML** | Zama，TFHE 上的隐私 ML | 🟢 [github](https://github.com/zama-ai/concrete-ml) |
| **Opacus** | PyTorch 的差分隐私训练 | 🟢 [github](https://github.com/pytorch/opacus) |
| **TensorFlow Privacy** | TF 的 DP 训练 | 🟢 [github](https://github.com/tensorflow/privacy) |
| **PSI 库** | OpenMined PSI、蚂蚁 SecretFlow PSI | 🟡 |

---

## 五、基准与数据集

| 基准 | 覆盖 | 链接 |
|---|---|---|
| **LEAF** | 6 个 FL 任务（合成/CV/NLP），用户级切分的经典 | 🟢 [leaf.cmu.edu](https://leaf.cmu.edu) · 🟡 arXiv:1812.01097 |
| **FedScale** | **20 个联邦数据集** + 客户端系统行为数据，含系统性能评测 | 🟢 [github](https://github.com/SymbioticLab/FedScale) · [论文](https://www.mosharaf.com/wp-content/uploads/fedscale-icml22.pdf) |
| **NIID-Bench** | 6 种非 IID 切分策略下的算法对比（ICDE 2022） | 🟢 [github](https://github.com/Xtra-Computing/NIID-Bench) · [arXiv:2102.02079](https://arxiv.org/pdf/2102.02079) |
| **FL-bench** | 社区维护的算法评测集 | 🟢 [github](https://github.com/KarhouTam/FL-bench) |
| **pFL-Bench** | 个性化 FL 专门基准 | 🟢 [arXiv:2206.03655](https://arxiv.org/pdf/2206.03655) |
| **FLamby** | **医疗**跨中心真实数据基准 | 🟡 arXiv:2210.04620 |
| **VFLAIR** | **纵向联邦的攻防基准库**（ICLR 2024）——本项目最相关 | 🟢 [论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/916cb4e1aeafaa0757953c9bacd17337-Paper-Conference.pdf) |
| **VertiBench** | VFL 的特征分布多样性基准（ICLR 2024） | 🟡 |
| **FedLLM-Bench** | 联邦 LLM 的真实基准（NeurIPS 2024 D&B） | 🟢 [arXiv:2406.04845](https://arxiv.org/html/2406.04845v1) |
| **OARF** | 多任务真实场景基准 | 🟡 |
| **FLAIR** | 多标签图像分类（Apple） | 🟡 |

**营销/广告类公开数据（含随机对照，能算增益）**

- 🟢 **Criteo Uplift Prediction** — [ailab.criteo.com](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)，约 2500 万行，CC BY-NC-SA 4.0
- 🟡 **Hillstrom MineThatData** — 约 6.4 万行，邮件营销随机试验
- 🟡 **FedAds** — 阿里妈妈的 VFL 广告基准（SIGIR 2023），本项目历史阶段用过

---

## 六、联邦 + 大模型：目前最热的方向

| 项目 | 说明 | 链接 |
|---|---|---|
| **OpenFedLLM** | KDD 2024，联邦训练 LLM 的研究代码库，30+ 评测指标覆盖通用/医疗/金融/代码/数学 | 🟢 [github](https://github.com/rui-ye/OpenFedLLM) |
| **FATE-LLM** | 微众，工业级联邦大模型框架 | 🟢 [github](https://github.com/FederatedAI/FATE-LLM) |
| **FedLLM-Factory** | 统一库，实现 **15+ 种联邦微调方法**（LoRA 为主） | 🟢 [github](https://github.com/boyi-liu/FedLLM-Factory) |
| **Awesome-Federated-LLM-Learning** | 两个持续更新的论文集 | 🟢 [longtanle 版](https://github.com/longtanle/Awesome-Federated-LLM-Learning) · [Clin0212 版](https://github.com/Clin0212/Awesome-Federated-LLM-Learning) |

**代表论文**

- 🟢 [The Future of LLM Pre-training is Federated](https://arxiv.org/pdf/2405.10853) — 联邦预训练的可行性论证
- 🟢 [FLoRA: Federated Fine-Tuning with Heterogeneous Low-Rank Adaptations](https://arxiv.org/pdf/2409.05976)
- 🟢 [FedEx-LoRA: Exact Aggregation for Federated Fine-Tuning](https://arxiv.org/pdf/2410.09432)（ACL 2025）
- 🟡 **FedSA-LoRA**：LoRA 的选择性聚合（ICLR 2025）
- 🟢 [Memory-Efficient Federated Fine-Tuning via Layer Pruning](https://arxiv.org/pdf/2508.17209)
- 🟢 [Safe-FedLLM: Delving into the Safety of Federated LLMs](https://arxiv.org/pdf/2601.07177)
- 🟢 [FedMomentum: Preserving LoRA Training Momentum](https://arxiv.org/pdf/2603.08014)

> **为什么 LoRA 成了联邦微调的默认选择**：只传低秩矩阵而非全参数，通信量降几个数量级。
> 但**聚合语义有坑**——各客户端的 LoRA 矩阵直接平均在数学上不等于低秩更新的平均，
> FedEx-LoRA 与 FLoRA 就是在修这个问题。

---

## 七、产业落地案例

| 案例 | 场景 | 说明 |
|---|---|---|
| **Google Gboard** | 手机输入法下一词预测 | FL 的第一个大规模生产系统，cross-device 的范本 🟡 |
| **Apple** | Siri、QuickType | 与本地差分隐私结合；开源了 pfl-research 🟢 |
| **NVIDIA + 医疗机构** | 多中心医学影像 | FLARE 的主战场；COVID 预测多中心研究 🟡 |
| **Owkin / Substra** | 药物研发多中心 | 强调可追溯与合规审计 🟢 |
| **字节 Fedlearner** | 广告投放 | 神经网络 VFL，已落地电商/互金/教育 🟢 |
| **微众 FATE** | 银行风控、保险 | 中国金融业 VFL 落地最多的框架 🟢 |
| **蚂蚁 SecretFlow** | 金融风控、联合营销 | 多技术路线并存 🟢 |

---

## 八、标准、法规与治理

| 文件 | 内容 | 链接 |
|---|---|---|
| **IEEE 3652.1-2020** | **联邦机器学习的架构框架与应用指南**——目前唯一的国际标准。定义 FML 分类、应用场景、性能评估与监管要求 | 🟢 [IEEE SA](https://standards.ieee.org/ieee/3652.1/7453/) · [IEEE Xplore](https://ieeexplore.ieee.org/document/9382202/) · [标准导读](https://dl.acm.org/doi/10.1145/3511285.3511291) |
| **中国 PIPL / 数据出境三路径** | 见本项目 `modules/m0_compliance/legal_references.md`（逐字原文已整理） | 🟢 项目内 |
| **香港 PDPO 第 VIA 部** | 直接促销专章 | 🟢 项目内 |
| **粤港澳大湾区标准合同** | 免除跨境传输数量门槛 | 🟢 项目内 |

⚪ 中国信通院《隐私计算白皮书》系列、金标委相关标准——以官方发布为准。

---

## 九、Awesome 列表与学习路径

| 资源 | 说明 | 链接 |
|---|---|---|
| **GitHub Topic: federated-learning** | 按 star/fork 排序看生态全貌 | 🟢 [topics/federated-learning](https://github.com/topics/federated-learning) |
| **awesome-vertical-federated-learning** | VFL 专门 | 🟢 [github](https://github.com/ngc436/awesome-vertical-federated-learning) |
| **VFL_Survey**（武大 MARS） | 综述 + 论文列表 | 🟢 [github](https://github.com/shentt67/VFL_Survey) |
| **Awesome-Federated-LLM-Learning** | 联邦大模型 | 🟢 见第六节 |
| **Federated-Learning-Non-IID** | 五种主流算法的 PyTorch 实现 | 🟢 [github](https://github.com/meng1103/Federated-Learning-Non-IID) |

**建议的学习顺序**

1. 读 FedAvg 原文，跑通一个 MNIST 的 FedAvg（Flower 半小时能起）
2. 读 Kairouz 综述的目录，建立"这个领域有哪些问题"的地图
3. 按方向深入：横向选非 IID 那条线；纵向直接读武大 VFL 综述 + VFLAIR
4. **攻防必须做一遍**：不亲手跑一次梯度反演，就不会真的相信"数据不出域 ≠ 安全"
5. 工程上选一个框架吃透：研究选 Flower，纵向选 SecretFlow 或 FATE，企业部署看 FLARE

---

## 十、与本项目的关系

按相关性排序，**可以直接拿来用的**：

| 资源 | 用在哪 |
|---|---|
| **VFLAIR** | M7 攻防评测的现成基准，避免自造轮子 |
| **武大 VFL 综述 + IJCAI 2024 标签泄露综述** | M7 威胁模型的文献底稿 |
| **SecretFlow** | 平台选型的第一候选（7/29 决议排除 FATE） |
| **Criteo Uplift / Hillstrom** | M1 的随机对照数据源（已列入 `dataset_request.md`） |
| **NIID-Bench 的切分策略** | M1 的 S1–S4 切分协议可参考其实现 |
| **Breaking Secure Aggregation（2406.15731）** | **对本项目特别重要**：cross-silo（两方）场景下聚合梯度仍可反推标签——本项目正是两方 |
| **FedHSSL / Semi-VFL** | M5 非对齐样本利用的方法参考 |
| **IEEE 3652.1-2020** | M9 合规证据链可引用的国际标准 |

**明确不相关但值得知道的**：cross-device 的客户端选择（Oort）、移动端系统（FLSim/Papaya）——
本项目是 cross-silo 两方，这些不适用；但知道它们存在，能避免误用为参考。

---

## 十一、这份文档的局限

1. **🟡 标记的 arXiv 编号未逐条打开核对**，凭既有知识写出。**写进正式材料前必须点开确认**。
2. **覆盖有偏**：偏向本项目关心的 VFL、隐私攻防、金融场景；对图联邦、联邦强化学习、
   联邦推荐、边缘计算/无线通信侧的 FL 覆盖较浅。
3. **时效性**：2026 年 8 月的快照。FL 领域每年论文以千计，半年后这份列表会明显过时。
4. **没有做质量筛选**：列出不等于推荐。同一方向的多个方法之间孰优孰劣，本文档不作判断。
