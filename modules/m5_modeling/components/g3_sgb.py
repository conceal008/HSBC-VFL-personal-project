"""M5 · G3 联邦臂：隐语 SGB（SecureBoost 类）单进程两方模拟训练与打分（S6.P3 预注册的臂 A 与阴性对照 NC）。

- 内地（neidi）为标签方，持有私钥；香港（hk）只持公钥做同态计算。同态方案 OU，密钥长度沿用隐语测试配置。
- 运行方式与 INNOVATIONS 的 SGB 路线相同：``debug_mode`` 单进程模拟两方，耗时只作臂间相对比较（DR-GOV-016 · D-3）。
- 打分在模拟中直接取回明文预测，只用于离线评估；真实部署时分数只交给内地。
- 需要 SecretFlow（secretflow-lite 1.13.0b0，Python 3.10）；本模块在导入时不依赖它，调用时才导入。
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np

LABEL_PARTY = "neidi"
PASSIVE_PARTY = "hk"


def sgb_params(config: Dict, sketch_eps: float, seed: int) -> Dict:
    """把与 XGBoost 共用的搜索空间映射为隐语 SGB 参数。"""
    return {
        "num_boost_round": int(config["num_boost_round"]),
        "max_depth": int(config["max_depth"]),
        "learning_rate": float(config["learning_rate"]),
        "reg_lambda": float(config["reg_lambda"]),
        "sketch_eps": float(sketch_eps),
        "objective": "logistic",
        "seed": int(seed),
    }


def fit_and_score(x_guest: np.ndarray, x_host: np.ndarray, y: np.ndarray,
                  scoring_sets: Dict[str, Tuple[np.ndarray, np.ndarray]], params: Dict,
                  he_schema: str) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """训练一次 SGB，并对每个打分集（内地矩阵，香港矩阵）返回预测概率。

    返回（各打分集的预测，计时）。scoring_sets 的行顺序在两方之间已对齐。
    """
    import secretflow as sf
    import spu
    from secretflow.data.ndarray import FedNdarray, PartitionWay
    from secretflow.device import reveal
    from secretflow.ml.boost.sgb_v import Sgb

    sf.init(parties=[LABEL_PARTY, PASSIVE_PARTY], address="local", debug_mode=True)
    neidi, hk = sf.PYU(LABEL_PARTY), sf.PYU(PASSIVE_PARTY)
    heu_cfg = sf.utils.testing.heu_config(LABEL_PARTY, [PASSIVE_PARTY])
    heu_cfg["he_parameters"]["schema"] = he_schema
    heu = sf.HEU(heu_cfg, spu.FieldType.FM64)

    def fed(xg: np.ndarray, xh: np.ndarray) -> FedNdarray:
        return FedNdarray(partitions={neidi: neidi(lambda: xg)(), hk: hk(lambda: xh)()},
                          partition_way=PartitionWay.VERTICAL)

    y_fed = FedNdarray(partitions={neidi: neidi(lambda: y.reshape(-1, 1).astype(float))()},
                       partition_way=PartitionWay.VERTICAL)
    sgb = Sgb(heu=heu)
    sgb.set_params(params)
    t0 = time.perf_counter()
    model = sgb.fit(fed(x_guest.astype(float), x_host.astype(float)), y_fed)
    fit_s = time.perf_counter() - t0
    scores = {}
    t1 = time.perf_counter()
    for name, (xg, xh) in scoring_sets.items():
        pred = reveal(model.predict(fed(xg.astype(float), xh.astype(float))))
        scores[name] = np.asarray(pred, dtype=float).reshape(-1)
    timing = {"fit_seconds": fit_s, "predict_seconds": time.perf_counter() - t1}
    sf.shutdown()
    return scores, timing
