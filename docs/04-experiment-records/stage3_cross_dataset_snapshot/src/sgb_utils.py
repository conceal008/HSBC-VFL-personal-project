"""
真实 SecretFlow 1.14.0b0 SGB 封装模块（OU / Paillier、2/3/4 方、M3 明文标签方直方图、M6 Bagging、I6 软标签）。
必须在 /Applications/总文件夹/汇丰/8-9月/隐语_SecretFlow/.venv/bin/python 下运行。
"""
from __future__ import annotations

import contextlib
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@contextlib.contextmanager
def plaintext_label_holder_patch(enabled: bool = True):
    """
    运行时上下文管理器（不修改 .venv 源码）：
    M3 标签方明文直方图优化——被动方仍收到完全相同的密文梯度，仅标签方对其本地特征改用明文编码。
    """
    if not enabled:
        yield
        return
    from secretflow.ml.boost.sgb_v.factory.components.gradient_encryptor.gradient_encryptor import (
        GradientEncryptor,
        move_config,
    )

    orig_cache = GradientEncryptor.cache_to_workers

    def patched_cache(self, encrypted_gh, gh):
        if self.params.label_holder_feature_only:
            return orig_cache(self, encrypted_gh, gh)
        cache = {}
        for worker in self.workers:
            if worker == self.label_holder:
                cache[worker] = gh.to(self.heu, move_config(self.label_holder, self.gh_encoder))
            else:
                cache[worker] = encrypted_gh.to(self.heu, move_config(worker, self.gh_encoder))
        return cache

    GradientEncryptor.cache_to_workers = patched_cache
    try:
        yield
    finally:
        GradientEncryptor.cache_to_workers = orig_cache


NAN_SENTINEL = -9999.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def prior_logit(y: np.ndarray) -> float:
    p = float(np.clip(np.mean(y), 1e-5, 1.0 - 1e-5))
    return float(np.log(p / (1.0 - p)))


def sgb_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.int32)
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    order = np.argsort(-p)
    ys = y[order]
    n_pos = max(int(ys.sum()), 1)
    n_neg = max(len(ys) - n_pos, 1)
    ks = float(np.max(np.abs(np.cumsum(ys) / n_pos - np.cumsum(1 - ys) / n_neg)))
    k5 = max(1, int(round(len(y) * 0.05)))
    top5 = float(y[order[:k5]].mean() / max(y.mean(), 1e-12))
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": ks,
        "brier": float(brier_score_loss(y, p)),
        "top5_lift": top5,
    }


def run_sgb_fit_predict(
    party_names: Sequence[str],
    X_tr_parts: Sequence[np.ndarray],
    y_tr: np.ndarray,
    X_te_parts: Sequence[np.ndarray],
    y_te: Optional[np.ndarray] = None,
    schema: str = "ou",
    key_size: int = 2048,
    num_boost_round: int = 40,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    reg_lambda: float = 5.0,
    sketch_eps: float = 0.03,
    colsample_by_tree: float = 0.8,
    rowsample_by_tree: float = 0.8,
    use_prior_base_score: bool = True,
    base_score_override: Optional[float] = None,
    first_tree_with_label_holder_feature: bool = False,
    seed: int = 11,
    use_m3_plaintext_label: bool = True,
    objective: str = "logistic",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    在真实 SecretFlow 1.14 SGB (debug_mode 单进程模拟 + 真实 PHEU 密码学运算) 上训练并预测。
    party_names[0] 为标签方（sk_keeper），其余为被动方（evaluators）。
    """
    import secretflow as sf
    import spu
    from secretflow.data.ndarray import FedNdarray, PartitionWay
    from secretflow.ml.boost.sgb_v import Sgb

    assert len(party_names) == len(X_tr_parts) == len(X_te_parts)
    assert len(party_names) >= 2, "SGB vertical requires >= 2 parties"

    # SecretFlow SGB 拒绝 NaN：各参与方在本地将缺失值替换为哨兵值 -9999（落入最低分桶，等价“缺失单独成桶”），
    # 无跨方信息交换；正负无穷同样处理。
    X_tr_parts = [np.nan_to_num(np.asarray(a, dtype=np.float32), nan=NAN_SENTINEL, posinf=NAN_SENTINEL, neginf=NAN_SENTINEL) for a in X_tr_parts]
    X_te_parts = [np.nan_to_num(np.asarray(a, dtype=np.float32), nan=NAN_SENTINEL, posinf=NAN_SENTINEL, neginf=NAN_SENTINEL) for a in X_te_parts]

    y_tr_arr = np.asarray(y_tr, dtype=np.float32).reshape(-1, 1)
    if base_score_override is not None:
        base_score = float(base_score_override)
    elif use_prior_base_score and objective == "logistic":
        base_score = prior_logit(y_tr_arr)
    else:
        base_score = 0.0

    params = {
        "num_boost_round": int(num_boost_round),
        "max_depth": int(max_depth),
        "learning_rate": float(learning_rate),
        "objective": objective,
        "reg_lambda": float(reg_lambda),
        "gamma": 0.0,
        "sketch_eps": float(sketch_eps),
        "base_score": float(base_score),
        "colsample_by_tree": float(colsample_by_tree),
        "rowsample_by_tree": float(rowsample_by_tree),
        "first_tree_with_label_holder_feature": bool(first_tree_with_label_holder_feature),
        "fixed_point_parameter": 20,
        "batch_encoding_enabled": True,
        "seed": int(seed),
    }

    t_init0 = time.perf_counter()
    sf.init(parties=list(party_names), address="local", debug_mode=True)
    devs = [sf.PYU(n) for n in party_names]
    heu_cfg = {
        "sk_keeper": {"party": party_names[0]},
        "evaluators": [{"party": p} for p in party_names[1:]],
        "mode": "PHEU",
        "he_parameters": {
            "schema": schema,
            "key_pair": {"generate": {"bit_size": int(key_size)}},
        },
    }
    heu = sf.HEU(heu_cfg, spu.FieldType.FM64)
    init_sec = time.perf_counter() - t_init0

    try:
        X_tr_fed = FedNdarray(
            partitions={
                dv: dv(lambda a=np.asarray(arr, dtype=np.float32): a)()
                for dv, arr in zip(devs, X_tr_parts)
            },
            partition_way=PartitionWay.VERTICAL,
        )
        X_te_fed = FedNdarray(
            partitions={
                dv: dv(lambda a=np.asarray(arr, dtype=np.float32): a)()
                for dv, arr in zip(devs, X_te_parts)
            },
            partition_way=PartitionWay.VERTICAL,
        )
        y_tr_fed = FedNdarray(
            partitions={devs[0]: devs[0](lambda a=y_tr_arr: a)()},
            partition_way=PartitionWay.VERTICAL,
        )

        with plaintext_label_holder_patch(use_m3_plaintext_label):
            t_fit0 = time.perf_counter()
            model = Sgb(heu=heu).train(params, X_tr_fed, y_tr_fed)
            fit_sec = time.perf_counter() - t_fit0

        t_pred0 = time.perf_counter()
        probs = np.asarray(sf.reveal(model.predict(X_te_fed)), dtype=np.float64).reshape(-1)
        pred_sec = time.perf_counter() - t_pred0
    finally:
        sf.shutdown()

    info: Dict[str, Any] = {
        "parties": list(party_names),
        "party_dims": [int(a.shape[1]) for a in X_tr_parts],
        "n_train": int(len(y_tr_arr)),
        "n_test": int(X_te_parts[0].shape[0]),
        "schema": schema,
        "key_size": int(key_size),
        "use_m3_plaintext_label": bool(use_m3_plaintext_label),
        "params": params,
        "init_sec": round(init_sec, 3),
        "fit_sec": round(fit_sec, 3),
        "pred_sec": round(pred_sec, 3),
    }
    if y_te is not None:
        info["metrics"] = sgb_metrics(y_te, probs)
    return probs, info
