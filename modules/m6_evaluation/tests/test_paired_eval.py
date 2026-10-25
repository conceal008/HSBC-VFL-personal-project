"""M6 · S6.P1 配对比较判据的单元测试。只检验实现忠实于定义；判据的统计性质由 notebook 的模拟校准。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.m6_evaluation.components import paired_eval as PE  # noqa: E402

SEED = 7
N = 3000
BASE = 0.1
SIGNAL = 2.0
SIGMA = 0.5
SEED_SIGMA = 0.1
TOL = 1e-9
RATE_TOL = 0.005
ALPHA = 0.05
POWER = 0.8
N_BOOT = 200
N_SEEDS = 6
ROUND_DP = 1
REPEAT = 3


def _data(n=N, seeds=N_SEEDS, sigma_b=SIGMA):
    rng = np.random.default_rng(SEED)
    z, y = PE.simulate_labels(n, BASE, SIGNAL, rng)
    return (y, PE.simulate_scores(z, SIGMA, SEED_SIGMA, seeds, rng),
            PE.simulate_scores(z, sigma_b, SEED_SIGMA, seeds, rng), rng)


def test_加权AUROC等权时与sklearn一致():
    y, a, _, _ = _data()
    assert abs(PE.weighted_auc(y, a[0], np.ones(N)) - roc_auc_score(y, a[0])) < TOL


def test_加权AUROC并列打分与sklearn一致():
    y, a, _, _ = _data()
    s = np.round(a[0], ROUND_DP)
    assert abs(PE.weighted_auc(y, s, np.ones(N)) - roc_auc_score(y, s)) < TOL


def test_整数权重等价于重复样本():
    y, a, _, _ = _data()
    w = np.zeros(N)
    w[: N // 2] = REPEAT
    idx = np.repeat(np.arange(N // 2), REPEAT)
    assert abs(PE.weighted_auc(y, a[0], w) - roc_auc_score(y[idx], a[0][idx])) < TOL


def test_多组权重返回向量():
    y, a, _, rng = _data()
    w = PE.cluster_poisson_weights(np.arange(N), N_BOOT, rng)
    out = PE.weighted_auc(y, a[0], w)
    assert out.shape == (N_BOOT,)


def test_簇权重簇内共享():
    rng = np.random.default_rng(SEED)
    clusters = np.repeat(np.arange(N // REPEAT), REPEAT)
    w = PE.cluster_poisson_weights(clusters, N_BOOT, rng)
    assert w.shape == (N_BOOT, len(clusters))
    assert np.all(w[:, 0] == w[:, 1]) and np.all(w[:, 1] == w[:, 2])


def test_符号翻转最小p值():
    assert PE.min_sign_flip_p(5) == pytest.approx(0.0625)
    assert PE.min_sign_flip_p(6) == pytest.approx(0.03125)
    assert PE.sign_flip_p(np.ones(5)) == pytest.approx(0.0625)


def test_符号翻转无差异时p为1():
    assert PE.sign_flip_p(np.zeros(4)) == pytest.approx(1.0)


def test_配对重抽样结果字段与区间():
    y, a, b, rng = _data()
    res = PE.paired_bootstrap(y, a, b, None, N_BOOT, ALPHA, rng)
    assert res["ci_low"] <= res["diff"] <= res["ci_high"]
    assert 0 <= res["p_value"] <= 1
    assert res["n_seeds"] == N_SEEDS and res["n_valid_boot"] == N_BOOT


def test_明显更差的模型被判为可分辨下降():
    y, a, b, rng = _data(sigma_b=SIGMA * REPEAT)
    res = PE.paired_bootstrap(y, a, b, None, N_BOOT, ALPHA, rng)
    assert res["ci_low"] > 0
    assert PE.verdict(res, 0.01, 0.01) == PE.VERDICT_GAIN
    flipped = PE.paired_bootstrap(y, b, a, None, N_BOOT, ALPHA, rng)
    assert PE.verdict(flipped, 0.01, 0.01) == PE.VERDICT_LOSS


def test_区间含零时按可检出规模区分():
    res = {"ci_low": -0.01, "ci_high": 0.01}
    assert PE.verdict(res, 0.03, 0.01) == PE.VERDICT_UNDERPOWERED
    assert PE.verdict(res, 0.005, 0.01) == PE.VERDICT_NO_DIFF


def test_可检出差异为正():
    y, a, b, rng = _data()
    res = PE.paired_bootstrap(y, a, b, None, N_BOOT, ALPHA, rng)
    assert PE.detectable_gap(res, int(y.sum()), float(y.mean()), ALPHA, POWER) > 0


def test_截距使正例率达标():
    rng = np.random.default_rng(SEED)
    _, y = PE.simulate_labels(N * N_BOOT, BASE, SIGNAL, rng)
    assert abs(y.mean() - BASE) < RATE_TOL


def test_噪声越大总体AUROC越低():
    lo = PE.population_auc(BASE, SIGNAL, SIGMA, N * REPEAT, np.random.default_rng(SEED))
    hi = PE.population_auc(BASE, SIGNAL, SIGMA * REPEAT, N * REPEAT, np.random.default_rng(SEED))
    assert hi < lo


def test_系统误差与总噪声互逆():
    total = PE.total_sigma(SIGMA, SEED_SIGMA)
    assert abs(PE.systematic_sigma(total, SEED_SIGMA) - SIGMA) < TOL


def test_系统误差各seed共享():
    rng = np.random.default_rng(SEED)
    z = np.zeros(N)
    s = PE.simulate_scores(z, SIGMA, 0.0, N_SEEDS, rng)
    assert np.allclose(s[0], s[1])


def test_按目标差距求噪声():
    sigma_b = PE.sigma_for_gap(BASE, SIGNAL, SIGMA, 0.02, N * REPEAT, SEED, SIGMA * REPEAT * REPEAT)
    assert sigma_b > SIGMA


def test_校准运行字段齐全():
    rng = np.random.default_rng(SEED)
    out = PE.calibration_run(100, BASE, SIGNAL, SIGMA, SIGMA, SEED_SIGMA, N_SEEDS, 2, 50, ALPHA,
                             POWER, rng)
    for key in ("reject_rate_new_rule", "reject_rate_old_rule_5seed", "mean_detectable_gap",
                "mean_detectable_gap_bootstrap", "mean_bootstrap_se", "sd_diff_across_reps"):
        assert key in out


def test_非法输入被拒绝():
    y, a, b, rng = _data()
    with pytest.raises(ValueError):
        PE.weighted_auc(y[:-1], a[0], np.ones(N))
    with pytest.raises(ValueError):
        PE.paired_bootstrap(y, a, b[:, :-1], None, N_BOOT, ALPHA, rng)
    with pytest.raises(ValueError):
        PE.bootstrap_matrices(y, a, b, None, 0, rng)
    with pytest.raises(ValueError):
        PE.summarize(np.zeros(2), np.zeros(2), np.zeros((2, 2)), np.zeros((2, 2)), 1.0, rng)
    with pytest.raises(ValueError):
        PE.sign_flip_p(np.zeros(17))
    with pytest.raises(ValueError):
        PE.min_sign_flip_p(0)
    with pytest.raises(ValueError):
        PE.simulate_labels(1, BASE, SIGNAL, rng)
    with pytest.raises(ValueError):
        PE.intercept_for_rate(0.0, SIGNAL)
    with pytest.raises(ValueError):
        PE.calibration_run(100, BASE, SIGNAL, SIGMA, SIGMA, SEED_SIGMA, 4, 1, 10, ALPHA, POWER, rng)
    with pytest.raises(ValueError):
        PE.systematic_sigma(SEED_SIGMA, SIGMA)


def test_重抽样法可检出差异等于标准误乘z():
    from statistics import NormalDist
    y, a, b, rng = _data()
    res = PE.paired_bootstrap(y, a, b, None, N_BOOT, ALPHA, rng)
    z = NormalDist().inv_cdf(1 - ALPHA / 2) + NormalDist().inv_cdf(POWER)
    got = PE.detectable_gap(res, int(y.sum()), float(y.mean()), ALPHA, POWER, PE.MDE_BOOTSTRAP)
    assert res["se_diff"] > 0
    assert abs(got - z * res["se_diff"]) < TOL


def test_未知可检出差异算法被拒绝():
    y, a, b, rng = _data()
    res = PE.paired_bootstrap(y, a, b, None, N_BOOT, ALPHA, rng)
    with pytest.raises(ValueError):
        PE.detectable_gap(res, int(y.sum()), float(y.mean()), ALPHA, POWER, "其他")


def test_模拟法所需正例随差距缩小而增加():
    rng = np.random.default_rng(SEED)
    big = PE.simulated_positives(0.05, 100, BASE, SIGNAL, SIGMA, SEED_SIGMA, N_SEEDS, 20, ALPHA,
                                 POWER, N, SEED, SIGMA * REPEAT * REPEAT, rng)
    small = PE.simulated_positives(0.02, 100, BASE, SIGNAL, SIGMA, SEED_SIGMA, N_SEEDS, 20, ALPHA,
                                   POWER, N, SEED, SIGMA * REPEAT * REPEAT, rng)
    assert small["required_positives"] > big["required_positives"] > 0
    with pytest.raises(ValueError):
        PE.simulated_positives(0.0, 100, BASE, SIGNAL, SIGMA, SEED_SIGMA, N_SEEDS, 20, ALPHA,
                               POWER, N, SEED, SIGMA * REPEAT, rng)
    with pytest.raises(ValueError):
        PE.diff_sd_simulation(100, BASE, SIGNAL, SIGMA, SIGMA, SEED_SIGMA, N_SEEDS, 1, rng)


def test_保守法取两种算法较大者():
    y, a, b, rng = _data()
    res = PE.paired_bootstrap(y, a, b, None, N_BOOT, ALPHA, rng)
    args = (res, int(y.sum()), float(y.mean()), ALPHA, POWER)
    both = max(PE.detectable_gap(*args, PE.MDE_BOOTSTRAP), PE.detectable_gap(*args, PE.MDE_CLOSED_FORM))
    assert PE.detectable_gap(*args, PE.MDE_CONSERVATIVE) == pytest.approx(both)
