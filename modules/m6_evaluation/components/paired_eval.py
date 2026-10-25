"""M6 · S6.P1 配对比较判据（DR-MX-002 · D-1、D-2 的实现）。

两个模型在同一批测试客户、同一组训练 seed 上比较时：

- **主判据**：配对 AUROC 差的置信区间。区间由两层重抽样给出——
  外层重抽训练 seed（反映训练随机性），内层按客户簇重抽测试样本（反映测试集抽样误差）。
  内层用 Poisson(1) 权重的 bootstrap（每个簇一个权重，簇内各行共享），
  这样同一组权重可以同时作用于两个模型、全部 seed，保证配对。
- **可检出规模**：两种算法。`closed_form` 用两模型 AUROC 重抽样复本之间的相关系数代入 S0.1P 的
  `paired_auc_mde`；`bootstrap` 直接用配对差的重抽样标准误 ×（z₁₋α/₂ + z_power）。
  S6.P1 校准显示：闭式近似在正例较多时偏保守，重抽样标准误在正例较少（约 150 个）时于备择下偏低约一成。
  正式报告用 `conservative`，即两者取大（evaluation_protocol v1 第四节）。
- **所需正例数**：`simulated_positives` 在备择设定下用模拟法求（DR-MX-002 · D-2 触发后的预定后果）。
- **旧判据**（≥5 seed 且两模型区间不重叠）一并输出，只作对照。
- **seed 级符号翻转检验**：精确枚举 2^S 种符号，S=5 时最小双侧 p 为 0.0625。

本模块只做统计计算，不接触任何真实数据；校准用的合成打分生成器也在这里，
notebook 只负责编排与展示。
"""
from __future__ import annotations

import itertools
import math
from statistics import NormalDist
from typing import Dict, Optional

import numpy as np

from modules.m0_compliance.components.validation_power import paired_auc_mde

HALF = 0.5
POISSON_RATE = 1.0
DEFAULT_CHUNK = 100
MAX_SIGN_FLIP_SEEDS = 16
CORRELATION_CAP = 0.99
P_FLOOR_TOL = 1e-12
BISECTION_STEPS = 60
OLD_RULE_MIN_SEEDS = 5          # 旧判据「跨 ≥5 seed 且区间不重叠」
HERMITE_NODES = 80              # Gauss–Hermite 积分节点数
INTERCEPT_BOUND = 30.0          # 截距二分搜索区间 [-30, 30]

MDE_CLOSED_FORM = "closed_form"
MDE_BOOTSTRAP = "bootstrap"
MDE_CONSERVATIVE = "conservative"

VERDICT_GAIN = "可分辨的提升"
VERDICT_LOSS = "可分辨的下降"
VERDICT_UNDERPOWERED = "未达可检出规模"
VERDICT_NO_DIFF = "在预声明尺度上无可分辨差异"


# ————————————————— 加权 AUROC —————————————————

def weighted_auc(y: np.ndarray, score: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """按行权重计算 AUROC，支持一次传入多组权重。

    y        (n,) 0/1 标签
    score    (n,) 打分
    weights  (B, n) 或 (n,)；返回 (B,) 或标量。并列打分按 1/2 计。
    """
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    if y.shape != score.shape or w.shape[1] != y.shape[0]:
        raise ValueError("y、score、weights 长度不一致")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    is_pos = y[order] == 1
    w = w[:, order]
    starts = np.flatnonzero(np.r_[True, sorted_score[1:] != sorted_score[:-1]])
    pos = np.where(is_pos, w, 0.0)
    neg = np.where(is_pos, 0.0, w)
    if len(starts) < len(sorted_score):                 # 有并列打分时按组合并
        pos = np.add.reduceat(pos, starts, axis=1)
        neg = np.add.reduceat(neg, starts, axis=1)
    neg_below = np.cumsum(neg, axis=1) - neg
    num = (pos * (neg_below + HALF * neg)).sum(axis=1)
    den = pos.sum(axis=1) * neg.sum(axis=1)
    out = np.full(den.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out if np.ndim(weights) == 2 else out[0]


def cluster_poisson_weights(clusters: np.ndarray, n_boot: int,
                            rng: np.random.Generator) -> np.ndarray:
    """每个簇抽一个 Poisson(1) 权重，簇内各行共享。返回 (n_boot, n)。"""
    _, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    draws = rng.poisson(POISSON_RATE, size=(n_boot, int(inverse.max()) + 1))
    return draws[:, inverse].astype(float)


# ————————————————— 配对比较 —————————————————

def sign_flip_p(seed_diffs: np.ndarray) -> float:
    """seed 级配对差的精确双侧符号翻转检验。"""
    d = np.asarray(seed_diffs, dtype=float)
    if d.ndim != 1 or not 1 <= len(d) <= MAX_SIGN_FLIP_SEEDS:
        raise ValueError("seed 数须在 1–16 之间")
    observed = abs(d.mean())
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(d))))
    stats = np.abs((signs * d).mean(axis=1))
    return float(np.mean(stats >= observed - P_FLOOR_TOL))


def min_sign_flip_p(n_seeds: int) -> float:
    """S 个 seed 时符号翻转检验能达到的最小双侧 p 值：2 / 2^S。"""
    if n_seeds < 1:
        raise ValueError("n_seeds 须 ≥1")
    return float(2.0 / 2 ** n_seeds)


def bootstrap_matrices(y: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
                       clusters: Optional[np.ndarray], n_boot: int,
                       rng: np.random.Generator, chunk: int = DEFAULT_CHUNK):
    """返回两模型每个 seed 在每组簇权重下的 AUROC：两个 (S, n_boot) 矩阵。

    同一组权重同时作用于两个模型和全部 seed，保证配对。
    """
    a = np.atleast_2d(np.asarray(scores_a, dtype=float))
    b = np.atleast_2d(np.asarray(scores_b, dtype=float))
    y = np.asarray(y)
    if a.shape != b.shape or a.shape[1] != len(y):
        raise ValueError("两模型打分形状须相同，且列数等于样本数")
    if n_boot < 1:
        raise ValueError("n_boot 须 ≥1")
    n_seeds, n = a.shape
    groups = np.arange(n) if clusters is None else np.asarray(clusters)
    parts_a, parts_b = [], []
    done = 0
    while done < n_boot:
        c = min(chunk, n_boot - done)
        w = cluster_poisson_weights(groups, c, rng)
        parts_a.append(np.stack([weighted_auc(y, a[s], w) for s in range(n_seeds)]))
        parts_b.append(np.stack([weighted_auc(y, b[s], w) for s in range(n_seeds)]))
        done += c
    return np.concatenate(parts_a, axis=1), np.concatenate(parts_b, axis=1)


def summarize(obs_a: np.ndarray, obs_b: np.ndarray, boot_a: np.ndarray, boot_b: np.ndarray,
              alpha: float, rng: np.random.Generator) -> Dict[str, float]:
    """由观察值与重抽样矩阵算出配对差、区间、p 值、估计相关、旧判据与符号翻转。

    外层对 seed 有放回重抽（每组簇权重各抽一次），与内层簇重抽组合成两层重抽样。
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha 须在 (0,1)")
    n_seeds, n_boot = boot_a.shape
    pick = rng.integers(0, n_seeds, size=(n_seeds, n_boot))
    cols = np.arange(n_boot)
    ma = boot_a[pick, cols].mean(axis=0)
    mb = boot_b[pick, cols].mean(axis=0)
    diff = (boot_a - boot_b)[pick, cols].mean(axis=0)
    keep = ~(np.isnan(diff) | np.isnan(ma) | np.isnan(mb))
    diff, ma, mb = diff[keep], ma[keep], mb[keep]
    lo_q, hi_q = alpha / 2, 1 - alpha / 2
    ci = np.quantile(diff, [lo_q, hi_q])
    ci_a, ci_b = np.quantile(ma, [lo_q, hi_q]), np.quantile(mb, [lo_q, hi_q])
    p = min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))
    corr = float(np.corrcoef(ma, mb)[0, 1]) if len(ma) > 2 else float("nan")
    seed_diffs = obs_a - obs_b
    return {
        "n_seeds": n_seeds,
        "n_valid_boot": int(keep.sum()),
        "auc_a": float(obs_a.mean()),
        "auc_b": float(obs_b.mean()),
        "diff": float(seed_diffs.mean()),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p_value": float(p),
        "estimate_correlation": corr,
        "se_diff": float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan"),
        "old_rule_pass": bool(n_seeds >= OLD_RULE_MIN_SEEDS
                              and (ci_a[0] > ci_b[1] or ci_b[0] > ci_a[1])),
        "sign_flip_p": sign_flip_p(seed_diffs) if n_seeds <= MAX_SIGN_FLIP_SEEDS else float("nan"),
    }


def observed_aucs(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """每个 seed 在原测试集上的 AUROC。"""
    s = np.atleast_2d(np.asarray(scores, dtype=float))
    ones = np.ones(s.shape[1])
    return np.array([weighted_auc(y, s[i], ones) for i in range(s.shape[0])])


def paired_bootstrap(y: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
                     clusters: Optional[np.ndarray], n_boot: int, alpha: float,
                     rng: np.random.Generator, chunk: int = DEFAULT_CHUNK) -> Dict[str, float]:
    """两层重抽样的配对 AUROC 差（A − B）。

    scores_a, scores_b  (S, n)：每行一个训练 seed，列为同一批测试样本
    clusters            (n,) 客户簇编号；None 表示每行一个簇
    """
    boot_a, boot_b = bootstrap_matrices(y, scores_a, scores_b, clusters, n_boot, rng, chunk)
    return summarize(observed_aucs(y, scores_a), observed_aucs(y, scores_b),
                     boot_a, boot_b, alpha, rng)


def detectable_gap(result: Dict[str, float], n_pos: int, base_rate: float,
                   alpha: float, power: float, method: str = MDE_CLOSED_FORM) -> float:
    """这批测试集上可检出的最小配对差。

    closed_form：重抽样估计的相关系数代入 S0.1P 的近似公式；
    bootstrap：配对差的重抽样标准误 ×（z₁₋α/₂ + z_power）；
    conservative：两者取大，正式报告用此法（evaluation_protocol v1）。
    """
    if method == MDE_CONSERVATIVE:
        return max(detectable_gap(result, n_pos, base_rate, alpha, power, MDE_BOOTSTRAP),
                   detectable_gap(result, n_pos, base_rate, alpha, power, MDE_CLOSED_FORM))
    if method == MDE_BOOTSTRAP:
        if not (0 < alpha < 1 and 0 < power < 1):
            raise ValueError("alpha 与 power 须在 (0,1)")
        z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
        return float(z * result["se_diff"])
    if method != MDE_CLOSED_FORM:
        raise ValueError("未知的可检出差异算法：%s" % method)
    corr = result["estimate_correlation"]
    corr = 0.0 if math.isnan(corr) else min(max(corr, 0.0), CORRELATION_CAP)
    auc_ref = min(max(result["auc_a"], result["auc_b"]), 1 - P_FLOOR_TOL)
    return paired_auc_mde(auc_ref, n_pos, base_rate, corr, alpha, power)


def verdict(result: Dict[str, float], mde: float, delta_of_interest: float) -> str:
    """按 evaluation_protocol v1 第三节给出四类结论之一。"""
    if result["ci_low"] > 0:
        return VERDICT_GAIN
    if result["ci_high"] < 0:
        return VERDICT_LOSS
    if mde > delta_of_interest:
        return VERDICT_UNDERPOWERED
    return VERDICT_NO_DIFF


# ————————————————— 校准用的合成打分 —————————————————

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def intercept_for_rate(base_rate: float, signal: float) -> float:
    """求截距 c，使 E[sigmoid(c + signal·z)] = base_rate（z ~ N(0,1)，Gauss–Hermite 积分）。"""
    if not 0 < base_rate < 1:
        raise ValueError("base_rate 须在 (0,1)")
    nodes, weights = np.polynomial.hermite_e.hermegauss(HERMITE_NODES)
    weights = weights / weights.sum()

    def mean_rate(c: float) -> float:
        return float(np.sum(weights * _sigmoid(c + signal * nodes)))

    lo, hi = -INTERCEPT_BOUND, INTERCEPT_BOUND
    for _ in range(BISECTION_STEPS):
        mid = (lo + hi) / 2
        if mean_rate(mid) < base_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def simulate_labels(n: int, base_rate: float, signal: float,
                    rng: np.random.Generator):
    """潜变量 z ~ N(0,1)，标签 y ~ Bernoulli(sigmoid(c + signal·z))，c 使总体正例率等于 base_rate。"""
    if not 0 < base_rate < 1 or n < 2:
        raise ValueError("base_rate 须在 (0,1)，n 须 ≥2")
    z = rng.standard_normal(n)
    intercept = intercept_for_rate(base_rate, signal)
    y = (rng.random(n) < _sigmoid(intercept + signal * z)).astype(int)
    return z, y


def simulate_scores(z: np.ndarray, sigma_sys: float, sigma_seed: float, n_seeds: int,
                    rng: np.random.Generator) -> np.ndarray:
    """模型打分 = z + 系统误差 + seed 误差。

    系统误差 N(0, sigma_sys²) 每个模型抽一次、各 seed 共享，代表模型本身的偏差；
    seed 误差 N(0, sigma_seed²) 每个 seed 各抽一次，代表训练随机性。
    两个模型的系统误差相互独立。总噪声越大 AUROC 越低。
    """
    systematic = sigma_sys * rng.standard_normal(len(z))
    return z[None, :] + systematic[None, :] + sigma_seed * rng.standard_normal((n_seeds, len(z)))


def total_sigma(sigma_sys: float, sigma_seed: float) -> float:
    """单个 seed 打分相对 z 的总噪声标准差。"""
    return math.sqrt(sigma_sys ** 2 + sigma_seed ** 2)


def systematic_sigma(sigma_total: float, sigma_seed: float) -> float:
    """由总噪声与 seed 噪声反推系统误差标准差。"""
    if sigma_total < sigma_seed:
        raise ValueError("总噪声不能小于 seed 噪声")
    return math.sqrt(sigma_total ** 2 - sigma_seed ** 2)


def population_auc(base_rate: float, signal: float, sigma: float, n_large: int,
                   rng: np.random.Generator) -> float:
    """大样本近似的单个 seed 总体 AUROC；sigma 为总噪声标准差。"""
    z, y = simulate_labels(n_large, base_rate, signal, rng)
    score = z + sigma * rng.standard_normal(n_large)
    return float(weighted_auc(y, score, np.ones(n_large)))


def sigma_for_gap(base_rate: float, signal: float, sigma_a: float, gap: float,
                  n_large: int, seed: int, sigma_max: float) -> float:
    """二分搜索模型 B 的总噪声，使总体 AUROC 比模型 A（总噪声 sigma_a）低 gap。

    每次都用同一随机数种子生成大样本，使比较只差在噪声大小上。
    """
    target = population_auc(base_rate, signal, sigma_a, n_large,
                            np.random.default_rng(seed)) - gap
    lo, hi = sigma_a, sigma_max
    for _ in range(BISECTION_STEPS):
        mid = (lo + hi) / 2
        if population_auc(base_rate, signal, mid, n_large, np.random.default_rng(seed)) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def calibration_run(n_pos: int, base_rate: float, signal: float, sigma_sys_a: float,
                    sigma_sys_b: float, sigma_seed: float, n_seeds: int, n_rep: int,
                    n_boot: int, alpha: float, power: float,
                    rng: np.random.Generator) -> Dict[str, float]:
    """重复 n_rep 次「抽测试集 → 两模型各 n_seeds 个 seed 打分 → 配对判据」，汇总拒绝率。

    同一组重抽样矩阵上，另用前 5 个 seed 计算旧判据与 5-seed 符号翻转，作对照。
    """
    if n_seeds < OLD_RULE_MIN_SEEDS:
        raise ValueError("n_seeds 须 ≥5，才能同时计算旧判据")
    n = int(round(n_pos / base_rate))
    new_reject = old_reject = flip_full = flip_five = 0
    corr, mde, mde_boot, se_boot, diffs, positives = [], [], [], [], [], []
    five = slice(0, OLD_RULE_MIN_SEEDS)
    for _ in range(n_rep):
        z, y = simulate_labels(n, base_rate, signal, rng)
        sa = simulate_scores(z, sigma_sys_a, sigma_seed, n_seeds, rng)
        sb = simulate_scores(z, sigma_sys_b, sigma_seed, n_seeds, rng)
        boot_a, boot_b = bootstrap_matrices(y, sa, sb, None, n_boot, rng)
        obs_a, obs_b = observed_aucs(y, sa), observed_aucs(y, sb)
        res = summarize(obs_a, obs_b, boot_a, boot_b, alpha, rng)
        res5 = summarize(obs_a[five], obs_b[five], boot_a[five], boot_b[five], alpha, rng)
        new_reject += int(res["ci_low"] > 0 or res["ci_high"] < 0)
        flip_full += int(res["sign_flip_p"] < alpha)
        old_reject += int(res5["old_rule_pass"])
        flip_five += int(res5["sign_flip_p"] < alpha)
        corr.append(res["estimate_correlation"])
        mde.append(detectable_gap(res, int(y.sum()), float(y.mean()), alpha, power))
        mde_boot.append(detectable_gap(res, int(y.sum()), float(y.mean()), alpha, power,
                                       MDE_BOOTSTRAP))
        se_boot.append(res["se_diff"])
        positives.append(int(y.sum()))
        diffs.append(res["diff"])
    return {
        "n_pos_target": n_pos,
        "n_rows": n,
        "mean_positives": float(np.mean(positives)),
        "n_seeds": n_seeds,
        "n_rep": n_rep,
        "mean_diff": float(np.mean(diffs)),
        "reject_rate_new_rule": new_reject / n_rep,
        "reject_rate_old_rule_5seed": old_reject / n_rep,
        "reject_rate_sign_flip_full": flip_full / n_rep,
        "reject_rate_sign_flip_5seed": flip_five / n_rep,
        "mean_estimate_correlation": float(np.nanmean(corr)),
        "mean_detectable_gap": float(np.mean(mde)),
        "mean_detectable_gap_bootstrap": float(np.mean(mde_boot)),
        "mean_bootstrap_se": float(np.mean(se_boot)),
        "sd_diff_across_reps": float(np.std(diffs, ddof=1)) if n_rep > 1 else float("nan"),
    }


def diff_sd_simulation(n_pos: int, base_rate: float, signal: float, sigma_sys_a: float,
                       sigma_sys_b: float, sigma_seed: float, n_seeds: int, n_rep: int,
                       rng: np.random.Generator) -> float:
    """配对差（各 seed 平均）跨重复的标准差：每次重抽测试集与两个模型的误差，不做重抽样，速度快。"""
    if n_rep < 2:
        raise ValueError("n_rep 须 ≥ 2")
    n = int(round(n_pos / base_rate))
    diffs = []
    for _ in range(n_rep):
        z, y = simulate_labels(n, base_rate, signal, rng)
        sa = simulate_scores(z, sigma_sys_a, sigma_seed, n_seeds, rng)
        sb = simulate_scores(z, sigma_sys_b, sigma_seed, n_seeds, rng)
        diffs.append(float((observed_aucs(y, sa) - observed_aucs(y, sb)).mean()))
    return float(np.std(diffs, ddof=1))


def simulated_positives(gap: float, n_pos_start: int, base_rate: float, signal: float,
                        sigma_total_a: float, sigma_seed: float, n_seeds: int, n_rep: int,
                        alpha: float, power: float, n_large: int, seed: int, sigma_max: float,
                        rng: np.random.Generator) -> Dict[str, float]:
    """模拟法求检出总体差距 gap 所需的测试集正例数（DR-MX-002 · D-2 触发后的预定后果）。

    在**备择**设定下（模型 B 的总体 AUROC 比 A 低 gap）估计配对差的标准差：
    先在 n_pos_start 处估一次，按标准差 ∝ 1/√正例数 推出所需正例，再在该处重估一次修正。
    零效应下的标准差偏小（两模型噪声相同、相关更高），用它规划会低估所需正例。
    """
    if gap <= 0:
        raise ValueError("gap 须 > 0")
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    sys_a = systematic_sigma(sigma_total_a, sigma_seed)
    sys_b = systematic_sigma(sigma_for_gap(base_rate, signal, sigma_total_a, gap, n_large, seed,
                                           sigma_max), sigma_seed)
    n_pos = n_pos_start
    for _ in range(2):  # 一次估计 + 一次在所需规模处的修正
        sd = diff_sd_simulation(n_pos, base_rate, signal, sys_a, sys_b, sigma_seed, n_seeds,
                                n_rep, rng)
        n_pos = max(1, int(math.ceil(n_pos * (z * sd / gap) ** 2)))
    return {"gap": gap, "required_positives": n_pos, "sd_at_required": sd}
