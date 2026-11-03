"""M5 · S5.M2：用真实隐语 SGB（debug_mode 单进程两方模拟，OU）复核 S5.M1 代理选出的方法（DR-M5-012 · D-1）。

- 每个（数据集，方法）训练一次 SGB（seed 11），在 final 视图测试集上打分，与同配置代理（seed 11）比较。
- M1 / M1x：标签方本地把交叉拟合的 logit 作为每个样本的初始预测（替换隐语 init_pred 的常数），推理时在标签方加回；
  香港侧收到的消息形态不变（仍是梯度密文），协议不动。
- M6：6 个 SGB（行采样 0.8，成员 seed 为 tune_seed × 100 + j）概率平均；成员逐个存档，会话中断后可续跑。
- 区间只含测试集抽样（客户簇 bootstrap），不含训练随机性。

需要 SecretFlow（secretflow-lite 1.13.0b0）；导入本模块不依赖它，调用时才导入。打分在模拟中直接取回，只用于离线评估。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from modules.m5_modeling.components import method_screen as MS

LABEL_PARTY = "neidi"
PASSIVE_PARTY = "hk"
SKETCH_EPS_DEFAULT = 0.1          # 约 10 桶
SKETCH_EPS_FINE = 0.03            # 约 33 桶
HE_SCHEMA = "ou"
PERCENTILE_LOW = 2.5
PERCENTILE_HIGH = 97.5
HALF = 0.5


def sketch_eps(P: MS.Params, bins: int) -> float:
    return {P.bins_default: SKETCH_EPS_DEFAULT, P.bins_fine: SKETCH_EPS_FINE}[bins]


def sgb_params(P: MS.Params, cfg: Dict, bins: int, base_logit: float, seed: int) -> Dict:
    """S5.M1 配置 → 隐语 SGB 参数（base_score 在 logit 空间）。"""
    return {"num_boost_round": int(cfg["rounds"]), "max_depth": int(cfg["depth"]),
            "learning_rate": float(cfg["lr"]), "reg_lambda": float(cfg["lam"]),
            "colsample_by_tree": float(cfg["col"]), "rowsample_by_tree": float(cfg.get("sub", 1.0)),
            "sketch_eps": sketch_eps(P, bins), "objective": "logistic", "seed": int(seed),
            "base_score": float(base_logit)}


def method_spec(P: MS.Params, res: Dict, method: str) -> Tuple[Dict, int, Optional[str], Optional[str]]:
    """方法 → (配置, 分桶, 初始分数类型 half / prior / None, 逐样本先验来源 guest / guest_extra / None)。"""
    c = res["configs"]
    return {
        "F_def": (P.default_cfg, P.bins_default, "half", None),
        "F_opt_p": (c["central"], P.bins_fine, "prior", None),
        "F_M1": (c["m1"], P.bins_fine, None, "guest"),
        "F_M1x": (c.get("m1x"), P.bins_fine, None, "guest_extra"),
        "F_M6": ({**c["central"], "sub": P.bag_sub}, P.bins_fine, "prior", None),
    }[method]


def run_sgb(P: MS.Params, xg, xh, y, eg, eh, cfg, bins, base_logit, margin_fit=None, margin_ev=None, seed=None):
    """训练一次隐语 SGB 并对评估集打分；margin_fit 非空时替换标签方的初始预测。返回 (概率, 拟合秒数)。"""
    import secretflow as sf
    import spu
    from secretflow.data.ndarray import FedNdarray, PartitionWay
    from secretflow.device import reveal
    from secretflow.ml.boost.sgb_v import Sgb
    from secretflow.ml.boost.sgb_v.factory.components.model_builder import model_builder as MB

    sf.init(parties=[LABEL_PARTY, PASSIVE_PARTY], address="local", debug_mode=True)
    neidi, hk = sf.PYU(LABEL_PARTY), sf.PYU(PASSIVE_PARTY)
    heu_cfg = sf.utils.testing.heu_config(LABEL_PARTY, [PASSIVE_PARTY])
    heu_cfg["he_parameters"]["schema"] = HE_SCHEMA
    heu = sf.HEU(heu_cfg, spu.FieldType.FM64)

    def fed(a, b):
        return FedNdarray(partitions={neidi: neidi(lambda: a)(), hk: hk(lambda: b)()},
                          partition_way=PartitionWay.VERTICAL)

    orig = MB.ModelBuilder.init_pred
    if margin_fit is not None:
        m = np.asarray(margin_fit, float).reshape(-1, 1, order="F")

        def init_pred(self, sample_num, checkpoint_model=None, x=None):   # 标签方本地的逐样本初始预测
            assert checkpoint_model is None
            return self.label_holder(lambda: np.array(m, order="F"))()
        MB.ModelBuilder.init_pred = init_pred
    try:
        y_fed = FedNdarray(partitions={neidi: neidi(lambda: y.reshape(-1, 1).astype(float))()},
                           partition_way=PartitionWay.VERTICAL)
        t0 = time.perf_counter()
        model = Sgb(heu=heu).train(sgb_params(P, cfg, bins, base_logit, P.tune_seed if seed is None else seed),
                                   fed(xg.astype(float), xh.astype(float)), y_fed)
        fit_s = time.perf_counter() - t0
        raw = np.asarray(reveal(model.predict_with_trees(fed(eg.astype(float), eh.astype(float))))).reshape(-1)
    finally:
        MB.ModelBuilder.init_pred = orig
        sf.shutdown()
    if margin_ev is not None:
        raw = raw + np.asarray(margin_ev, float)       # 推理时标签方加回本地 logit
    return MS.sigmoid(raw), fit_s


def base_of(P: MS.Params, y, kind: Optional[str]) -> Tuple[float, Optional[float]]:
    """返回 (SGB logit 空间初始分数, 代理概率空间初始分数)。"""
    if kind == "prior":
        p = MS.prior_fn(P)(y)
        return float(MS.logit(np.array([p]))[0]), p
    return 0.0, (HALF if kind == "half" else None)


def inputs(P: MS.Params, sc: "MS.Screen", res: Dict, method: str):
    """一个方法的配置、分桶、两种初始分数与（M1 / M1x 的）交叉拟合 logit。"""
    fv = sc.v["final"]
    y = fv["fit_y"]
    cfg, bins, kind, mkind = method_spec(P, res, method)
    b_sgb, b_proxy = base_of(P, y, kind)
    mf = me = None
    if mkind:
        gcfg = res["configs"]["guest"] if mkind == "guest" else res["configs"]["guest_extra"]
        mf, me = MS.cross_fit(P, fv["fit_xg"], y, fv["ev_xg"], gcfg, P.bins_fine, P.tune_seed,
                              ex=fv["ex"] if mkind == "guest_extra" else None)
    return cfg, bins, b_sgb, b_proxy, mf, me


def bag_seeds(P: MS.Params) -> List[int]:
    return [P.tune_seed * MS.BAG_SEED_MULT + j for j in range(P.bag)]


def proxy_pred(P: MS.Params, sc: "MS.Screen", res: Dict, method: str, prepared=None) -> np.ndarray:
    """同配置代理（seed 11；M6 为同样 6 个 seed 的平均）。"""
    fv = sc.v["final"]
    cfg, bins, _, b_proxy, mf, me = prepared or inputs(P, sc, res, method)
    if method == "F_M6":
        return np.mean([MS.train_pred(P, fv["fit_xc"], fv["fit_y"], fv["ev_xc"], cfg, bins, b_proxy, sd)
                        for sd in bag_seeds(P)], axis=0)
    return MS.train_pred(P, fv["fit_xc"], fv["fit_y"], fv["ev_xc"], cfg, bins, b_proxy, P.tune_seed, mf=mf, me=me)


def confirm(P: MS.Params, sc: "MS.Screen", res: Dict, method: str, out: Path) -> Dict:
    """一个方法：SGB 打分（存 out/{ds}__{method}.npy）与同配置代理比较。M6 成员逐个存档，可续跑。"""
    fv = sc.v["final"]
    prepared = inputs(P, sc, res, method)
    cfg, bins, b_sgb, _, mf, me = prepared
    args = (fv["fit_xg"], fv["fit_xh"], fv["fit_y"], fv["ev_xg"], fv["ev_xh"], cfg, bins, b_sgb)
    if method == "F_M6":
        runs = []
        for j, sd in enumerate(bag_seeds(P)):
            mp, mj = out / ("%s__F_M6_m%d.npy" % (sc.name, j)), out / ("%s__F_M6_m%d.json" % (sc.name, j))
            if mp.exists() and mj.exists():
                runs.append((np.load(mp).astype(float), json.loads(mj.read_text())["fit_seconds"]))
                continue
            pj, tj = run_sgb(P, *args, seed=sd)
            np.save(mp, pj.astype(np.float32))
            mj.write_text(json.dumps({"seed": sd, "fit_seconds": round(tj, 1)}))
            runs.append((pj, tj))
        p, fit_s = np.mean([r[0] for r in runs], axis=0), sum(r[1] for r in runs)
    else:
        p, fit_s = run_sgb(P, *args, margin_fit=mf, margin_ev=me)
    q = proxy_pred(P, sc, res, method, prepared)
    np.save(out / ("%s__%s.npy" % (sc.name, method)), p.astype(np.float32))
    ye = fv["ev_y"]
    return {"sgb_auroc": MS.auc(ye, p), "proxy_seed11_auroc": MS.auc(ye, q), "proxy_6seed_auroc": res["test"][method]["auroc"],
            "fit_seconds": round(fit_s, 1), "trees": int(cfg["rounds"]), "depth": int(cfg["depth"]), "bins": bins,
            "max_abs_pred_diff_vs_proxy": float(np.max(np.abs(p - q)))}


def bootstrap(P: MS.Params, y, cl, scores: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """测试客户簇有放回重抽（单次拟合，不重抽 seed）。"""
    return MS.two_level_bootstrap(P, y, cl, {k: [v] for k, v in scores.items()})


def paired(A: Dict[str, np.ndarray], a: str, b: str, est: float) -> Dict:
    d = A[a] - A[b]
    return {"diff": est, "ci_low": float(np.percentile(d, PERCENTILE_LOW)),
            "ci_high": float(np.percentile(d, PERCENTILE_HIGH)),
            "p": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))}


def evaluate(P: MS.Params, sc: "MS.Screen", res: Dict, sgb: Dict[str, np.ndarray], proxy: Dict[str, np.ndarray],
             pairs: List[Tuple[str, str]]) -> Dict:
    """SGB 与代理逐方法对比、SGB 内部的方法对比（同一批 bootstrap）。"""
    fv = sc.v["final"]
    y = fv["ev_y"]
    scores = {**{"sgb_" + k: v for k, v in sgb.items()}, **{"proxy_" + k: v for k, v in proxy.items()}}
    A = bootstrap(P, y, fv.get("ev_cl"), scores)
    auc = {k: MS.auc(y, s) for k, s in scores.items()}
    methods = {m: {"sgb_auroc": auc["sgb_" + m], "proxy_auroc": auc["proxy_" + m],
                   "sgb_minus_proxy": paired(A, "sgb_" + m, "proxy_" + m, auc["sgb_" + m] - auc["proxy_" + m])}
               for m in sgb}
    con = {}
    for a, b in pairs:
        if a in sgb and b in sgb:
            con["%s − %s" % (a, b)] = {"sgb": paired(A, "sgb_" + a, "sgb_" + b, auc["sgb_" + a] - auc["sgb_" + b]),
                                       "proxy_6seed": next((c for c in res["contrasts"] if c["a"] == a and c["b"] == b),
                                                           None)}
    return {"methods": methods, "contrasts": con}
