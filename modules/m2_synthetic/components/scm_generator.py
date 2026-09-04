"""M2 · 双方联合结构因果生成器（SCM）。

为什么用 SCM 而不是「拟合真实分布再采样」：本项目必须知道
「标签信号有多少比例真实来自香港侧特征」，才能判断模型是否学到了该学的东西。
拟合类方法（SDV 等）只复制统计关系，给不出这个真值。

生成结构：
    共享潜变量 z_shared  ──┬──> 内地侧特征 X_A
                          └──> 香港侧特征 X_B
    内地私有潜变量 z_a    ────> X_A
    香港私有潜变量 z_b    ────> X_B

    logit = w_share·s_shared + w_a·s_a + λ·w_b·s_b + intercept
                                          ↑
                                    跨方互补度：λ 决定「只有香港侧才有的信号」占多少

    λ = 0 时，香港侧不提供任何独有信息 → VFL 的理论增益为 0。
    这是本生成器最重要的性质：它让「VFL 有没有价值」成为可实验命题。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

# —— 生成器内部常量（非可调参数，故为具名常量而非配置项）——
SIGNAL_LINEAR = "linear"            # s_shared + s_a + λ·s_b（可加，默认）
SIGNAL_INTERACTION = "interaction"  # s_shared + s_a + λ·(s_a × s_b)：B 只经交互起作用
SIGNAL_THRESHOLD = "threshold"      # s_shared + s_a + λ·1[s_b > 0]：B 以阶跃方式起作用
SIGNAL_FORMS = (SIGNAL_LINEAR, SIGNAL_INTERACTION, SIGNAL_THRESHOLD)
THRESHOLD_AT = 0.0                  # 阶跃形式的切点（标准化后取 0 即中位数附近）

DIM_SHARED_DEFAULT = 4        # 共享潜变量维数：双方都能观测到的结构
DIM_PRIVATE_A_DEFAULT = 6     # 主动方私有特征维数
DIM_PRIVATE_B_DEFAULT = 6     # 被动方私有特征维数
LOGIT_CLIP = 30.0          # 防止 sigmoid 溢出
STANDARDIZE_EPS = 1e-8     # 标准化除零保护
INTERCEPT_SEARCH_LO = -20.0
INTERCEPT_SEARCH_HI = 20.0
INTERCEPT_SEARCH_ITERS = 60
HALF = 0.5
TREATMENT_SHARE = 0.5      # 随机对照：处理组占比（双臂等分）


@dataclass
class GeneratorConfig:
    """11 项可调参数，对应《方案流程框架 v2》M2 的必备参数表。"""
    n_party_a: int                      # 内地侧客户数
    overlap_rate: float                 # ρ 重叠率：A 中同时是 B 客户的比例
    complementarity: float              # λ 跨方互补度 ★ 决定 VFL 是否有价值
    redundancy: float                   # 特征冗余度：共享潜变量的权重占比
    marginal_drift: float               # 边际漂移强度：B 侧特征分布的偏移
    size_asymmetry: float               # 样本量不对称比 N_B / N_A
    base_rate: float                    # 正样本率
    consent_rate: float                 # 同意率
    consent_selectivity: float          # 同意的选择性：>0 表示高分者更可能同意
    match_error_rate: float             # ID 假匹配 / 漏匹配率
    time_drift: float                   # 时间漂移强度（用于 OOT 划分）
    label_noise: float                  # 标签噪声率
    hte_strength: float                 # 异质处理效应强度（uplift 评估必需）
    signal_form: str = SIGNAL_LINEAR    # 信号形式：linear / interaction / threshold
    dim_shared: int = DIM_SHARED_DEFAULT
    dim_private_a: int = DIM_PRIVATE_A_DEFAULT
    dim_private_b: int = DIM_PRIVATE_B_DEFAULT
    name: str = "unnamed"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -LOGIT_CLIP, LOGIT_CLIP)))


def _standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / (x.std(axis=0) + STANDARDIZE_EPS)


def _b_contribution(cfg: "GeneratorConfig", s_a: np.ndarray,
                    s_b: np.ndarray) -> np.ndarray:
    """被动方对标签信号的贡献。三种形式检验的是**不同的机制**：

    - `linear`：可加贡献。被动方带来一份独立的信息，与主动方的信息不交互。
    - `interaction`：**只经交互起作用**。同样的 s_b，在 s_a 高的人身上是正向，
      在 s_a 低的人身上是负向。线性模型无论怎么调参都拿不到它（需要显式交叉项），
      树模型可以逼近。这是纵向联邦最有理论动机的一类价值来源。
    - `threshold`：阶跃贡献。被动方的信息只在越过某个门槛后才起作用，
      线性模型只能近似，树模型天然契合。

    λ=0 时三种形式都退化为「被动方无贡献」，零互补的硬验收判据因而对三者同时成立。
    """
    if cfg.signal_form == SIGNAL_LINEAR:
        return cfg.complementarity * s_b
    if cfg.signal_form == SIGNAL_INTERACTION:
        return cfg.complementarity * _standardize(s_a * s_b)
    if cfg.signal_form == SIGNAL_THRESHOLD:
        return cfg.complementarity * _standardize(
            (s_b > THRESHOLD_AT).astype(float))
    raise ValueError("未知的 signal_form：%s（可取 %s）"
                     % (cfg.signal_form, "/".join(SIGNAL_FORMS)))


def _solve_intercept(signal: np.ndarray, target_rate: float) -> float:
    """二分搜索截距，使 sigmoid(signal + b) 的均值等于目标正样本率。"""
    lo, hi = INTERCEPT_SEARCH_LO, INTERCEPT_SEARCH_HI
    for _ in range(INTERCEPT_SEARCH_ITERS):
        mid = (lo + hi) * HALF
        if _sigmoid(signal + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * HALF


def generate(cfg: GeneratorConfig, seed: int) -> Dict:
    """生成一份双方数据集。返回的字典含 ground truth，供正确性验收使用。"""
    rng = np.random.default_rng(seed)
    n = cfg.n_party_a

    # —— 潜变量 ——
    z_shared = rng.standard_normal((n, cfg.dim_shared))
    z_a = rng.standard_normal((n, cfg.dim_private_a))
    z_b = rng.standard_normal((n, cfg.dim_private_b))

    # —— 时间索引与漂移（用于 OOT 划分）——
    time_index = rng.uniform(0.0, 1.0, size=n)
    drift = cfg.time_drift * (time_index - HALF)[:, None]

    # —— 特征：冗余度决定共享潜变量在各方特征中的权重 ——
    load_a_shared = rng.standard_normal((cfg.dim_shared, cfg.dim_private_a)) * cfg.redundancy
    load_b_shared = rng.standard_normal((cfg.dim_shared, cfg.dim_private_b)) * cfg.redundancy
    x_a = z_a + z_shared @ load_a_shared + drift
    x_b = z_b + z_shared @ load_b_shared + cfg.marginal_drift + drift

    # —— 标签信号：三段可分离，使 λ 的含义精确 ——
    w_shared = rng.standard_normal(cfg.dim_shared)
    w_a = rng.standard_normal(cfg.dim_private_a)
    w_b = rng.standard_normal(cfg.dim_private_b)

    s_shared = _standardize(z_shared @ w_shared)
    s_a = _standardize(z_a @ w_a)
    s_b = _standardize(z_b @ w_b)

    b_term = _b_contribution(cfg, s_a, s_b)
    signal = s_shared + s_a + b_term
    intercept = _solve_intercept(signal, cfg.base_rate)
    true_logit = signal + intercept
    prob = _sigmoid(true_logit)

    # —— 标签噪声：以 label_noise 的概率翻转 ——
    y = (rng.uniform(size=n) < prob).astype(int)
    flip = rng.uniform(size=n) < cfg.label_noise
    y = np.where(flip, 1 - y, y)

    # —— 随机对照与异质处理效应（uplift 评估必需）——
    treatment = (rng.uniform(size=n) < TREATMENT_SHARE).astype(int)
    tau = cfg.hte_strength * _standardize(s_shared + b_term)
    prob_treated = _sigmoid(true_logit + tau)
    y_treated = (rng.uniform(size=n) < prob_treated).astype(int)
    y_observed = np.where(treatment == 1, y_treated, y)

    # —— 重叠：只有一部分 A 客户同时是 B 客户 ——
    in_overlap = rng.uniform(size=n) < cfg.overlap_rate

    # —— 同意：可带选择性（高分者更可能同意 → 选择偏差）——
    consent_score = _standardize(true_logit) * cfg.consent_selectivity + rng.standard_normal(n)
    consent_cut = np.quantile(consent_score, 1.0 - cfg.consent_rate)
    consent = consent_score >= consent_cut

    # —— ID 匹配噪声：一部分「匹配上」的其实是错配 ——
    mismatch = (rng.uniform(size=n) < cfg.match_error_rate) & in_overlap
    # 错配 = 香港侧特征被随机换成另一个人的
    perm = rng.permutation(n)
    x_b_observed = np.where(mismatch[:, None], x_b[perm], x_b)

    n_party_b = int(n * cfg.size_asymmetry)

    return {
        "config": cfg,
        "seed": seed,
        "x_a": x_a,
        "x_b": x_b_observed,
        "x_b_clean": x_b,
        "y": y_observed,
        "y_control": y,
        "treatment": treatment,
        "tau": tau,
        "time_index": time_index,
        "in_overlap": in_overlap,
        "consent": consent,
        "mismatch": mismatch,
        "true_logit": true_logit,
        "signal_shared": s_shared,
        "signal_a": s_a,
        "signal_b": s_b,
        "n_party_b": n_party_b,
    }


def usable_mask(data: Dict) -> np.ndarray:
    """可用样本 = 在交集内 ∧ 已同意。对应合规折损漏斗的后几层。"""
    return data["in_overlap"] & data["consent"]


def theoretical_gain_signal(data: Dict) -> Dict[str, np.ndarray]:
    """返回两种「上帝视角」的打分：含香港侧信号 vs 不含。

    两者的 AUC 之差就是本配置下 VFL 的**理论增益上界**，
    用于 M2 的正确性验收（理论增益 vs 实测增益偏差 <5%）。
    """
    cfg = data["config"]
    b_term = _b_contribution(cfg, data["signal_a"], data["signal_b"])
    with_b = data["signal_shared"] + data["signal_a"] + b_term
    without_b = data["signal_shared"] + data["signal_a"]
    return {"with_b": with_b, "without_b": without_b}


def load_scenarios(raw: Dict) -> List[GeneratorConfig]:
    """从配置字典构造场景列表：defaults 提供缺省值，scenarios 逐个覆盖。"""
    defaults = raw["defaults"]
    out: List[GeneratorConfig] = []
    for item in raw["scenarios"]:
        params = dict(defaults)
        params.update(item)
        out.append(GeneratorConfig(**params))
    return out
