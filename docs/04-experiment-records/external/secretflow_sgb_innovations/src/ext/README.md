# src/ext —— 切分方法跨数据集验证 + 框架优化

基于 `SCUT-Elite-Camp/federated_learning_new`（main，2026-10-19 读取）。**原版脚本一行未改**；
本目录只新增文件，产物格式与原 `data/processed/prepared/` 完全兼容。

| 文件 | 作用 | 对应优化 |
|---|---|---|
| `train_federated_sgb_opt.py` | 原 `train_federated_sgb.py` 的超集：默认参数下行为一致；新增 `--he-schema ou`、`--early-stop`、`--enable-packbits`、`--enable-quantization`、`--data-dir` | ① 加密提速 / ④ 早停 |
| `leakage_audit.py` | 实验前的标签同源 / 泄漏审计：单变量 AUC、血缘声明（`--lineage-regex`）、按侧汇总并给出警告 | ② |
| `gain_proxy.py` | 明文代理增益 Ĝ = 集中式 − max(两单边)，逐 tag 调原版 `train_local.py`，汇总业务 / 诊断 / 随机切分 | ③ |
| `train_local_es.py` | 明文三臂 + 早停（验证集模仿测试集切法：时间 / 客户），主判据稳健性检查 | ④ |
| `prepare_external.py` | 把 UCI Bank Marketing、Home Credit 按同一协议落成业务 / 诊断 / 随机切分矩阵 | 跨数据集验证 |

## 推荐流程（优化后）

```bash
# 0. 原版数据准备不变（prepare_santander → split_fields → prepare_matrix）或 prepare_external
# 1. 先审计：同源列 / 泄漏列在哪一侧
python src/ext/leakage_audit.py --tag business --lineage-regex '^ind_.*_ult1$'
# 2. 秒级明文代理：业务 vs 诊断 vs 随机切分的增益分布（决定值不值得跑联邦）
python src/ext/gain_proxy.py --name santander --tags business,diagnostic --random random_seed42,...
# 3. 稳健性：早停口径下主判据是否同向
python src/ext/train_local_es.py --tag business
# 4. 只对通过前三步的切分跑联邦确认（OU 比 Paillier 快约 6 倍，预测逐位相同）
python src/ext/train_federated_sgb_opt.py --tag business --num-boost-round 100 --seed 42 --he-schema ou --run-id business_sgb_federated_ou_r100_m42
```

## 运行环境（本次）
- 明文：Python 3.12，pandas 3.0.3，xgboost 3.4.1，scikit-learn 1.9.1
- 联邦：Python 3.10，secretflow 1.13.0b0（spu 0.9.4.dev20250618，sf-heu 0.6.0.dev20250514），debug_mode 单进程
- 机器：2 vCPU / 7 GB；耗时数字与原文（作者本机）不可直接比较，只比较同机内相对值

---

# 第二批：Santander 上的算法优化 + 安全攻防（2026-10-20）

| 文件 | 作用 |
|---|---|
| `train_federated_sgb_opt.py`（扩充） | 新增 `--sketch-eps`（分桶）、`--enable-goss/--top-rate/--bottom-rate`、`--rowsample/--colsample/--gamma`、`--first-tree-label-holder {0,1}`、`--label-rr-eps`（标签随机响应）、`--dump-model`（导出分裂结构供攻击取证）；默认行为仍与原版一致 |
| `make_val_tag.py` | 从训练窗口末 N 个月切出验证集 tag（`business_val`），所有选型只在它上面做 |
| `tune_proxy.py` | 明文 XGB 代理调参：内地单边 / 香港单边 / 集中式三臂同等调（48 组 × 3） |
| `final_eval.py` | 测试集终评（只评一次）：调参三臂 + 任意联邦预测文件，客户级配对 bootstrap |
| `sgb_replay.py` | 按隐语源码重放各方收到的样本划分（训练路由 / 推理期香港视角可达集合），已与隐语原生输出逐样本核对 |
| `attacks.py` | A1 香港推断标签（叶子成员 + 少量已知标签）；A2 内地还原香港特征（区间收缩，可对照真值、统计「持有=1」召回 / 精确率） |
| `make_defense_tags.py` | 香港侧数据级防护：`hostcnt`（24 个产品标记 → 持有产品数）、`hostrr --eps`（逐位本地随机响应） |
| `defense_eval.py` | 各防护版的效用代价：测试 AUC、相对无防护基线与内地单边的配对差值 |

```bash
# 优化
python src/ext/make_val_tag.py --tag business --val-months 3
python src/ext/tune_proxy.py --tag business_val
python src/ext/train_federated_sgb_opt.py --tag business --he-schema ou --num-boost-round 134 --max-depth 6 \
    --reg-lambda 10 --sketch-eps 0.03 --seed 42 --run-id business_sgb_ou_opt_m42 --dump-model data/processed/attack/model_T2_opt.json
python src/ext/final_eval.py --fed orig=data/processed/predictions/business_sgb_federated_r100_m42_predictions.csv \
                             --fed opt=data/processed/predictions/business_sgb_ou_opt_m42_predictions.csv
# 攻击（先用 --dump-model 训练）
cd src/ext && python attacks.py --model ../../data/processed/attack/model_S0_base.json --tag business \
    --truth-tag business --host-sees-first-tree 0 --known-frac 0.01 0.05 --name S0_base
# 防护
python src/ext/make_defense_tags.py --tag business --mode hostcnt
python src/ext/make_defense_tags.py --tag business --mode hostrr --eps 2
python src/ext/train_federated_sgb_opt.py --tag business --he-schema ou --num-boost-round 100 --label-rr-eps 2 ...
python src/ext/defense_eval.py --run S0=business_sgb_S0_base --run S3=business_sgb_S3_lrr2 ...
```
完整批量命令见 `run_q.sh`（2 核机器上串行比并行快；每项幂等，中断后重跑即续）。

---

# 第三批：标签 DP 改进、协议层防护、HSBC mock 复核（2026-10-20 晚）

| 文件 | 作用 |
|---|---|
| `train_federated_sgb_opt.py`（扩充） | `--label-dp {rr, rr_soft, rrprior, rrprior_soft}` 与 `--label-rr-eps` 合用：对称 RR / RR + 后验去噪软标签 / RRWithPrior 两阶段 / RRWithPrior + 软标签（软标签是 DP 后处理，ε 不变） |
| `label_dp_proxy.py` | 明文代理筛选四种标签 DP 变体（与联邦实测相差 < 0.006），先挑再跑联邦 |
| `ss_xgb_cost.py` | 隐语 SS-XGB（SEMI2K，ray 仿真 + 真实 SPU）成本测量，逐棵树计时、按样本量外推 |
| `secure_infer.py` | SGB 训练 + SPU 安全推理：两方叶子可达矩阵在密态下求交、乘权重累加，只把分数给内地；与原生推理逐样本对账 |
| `make_mock_tags.py` | 把团队 HSBC mock 文件（编码前原始值）按「客户 × 月份」映射回 business 已编码矩阵，生成 mock_* tag 并逐值核对 |

```bash
python src/ext/label_dp_proxy.py --eps 1 2 4
python src/ext/train_federated_sgb_opt.py --tag business --he-schema ou --num-boost-round 100 --label-rr-eps 2 --label-dp rr_soft --run-id business_sgb_D2_rrsoft2
python src/ext/ss_xgb_cost.py --rows 2000 5000 10000 --trees 3          # 注意 sf.init(num_cpus) 必须 > 2，否则两个 SPU actor 占满资源导致死锁
python src/ext/secure_infer.py --model data/processed/attack/model_T2_opt.json --batch-trees 3 \
    --ref data/processed/predictions/business_sgb_ou_opt_m42_predictions.csv  # 64 叶子时每批 3 棵，10 棵会内存不足
python src/ext/make_mock_tags.py --src <HSBC_mock 解压目录>
bash code/eval_mock.sh balanced overlap80 overlap50 overlap20
```

---

# 第四批：FedAds 复现 + 先验初始分数（2026-10-21 凌晨）

| 文件 / 参数 | 作用 |
|---|---|
| `make_fedads_tags.py` | FedAds 抽样行 → tag：主动方 l_* 做 5 折目标编码 + 高基数频次编码，被动方 f_* 做 one-hot；另切 fedads_val（训练样本内 80/20）供调参 |
| `train_federated_sgb_opt.py --base-score prior` | 由主动方用训练标签均值算 logit 作为 SGB 初始分数（隐语默认 0，即从 50% 起步）；FedAds +0.0150，Santander 无变化 |
| `label_dp_proxy.py --depth/--lr/--reg-lambda/--colsample/--rounds` | 代理筛选可指定超参 |
| 抽样脚本 `fedads_sample/sample_fedads.py`（本机） | 按行号固定种子均匀抽样：训练 300,000 / 2,598,552，测试 200,000 / 2,432,490 |

```bash
python src/ext/make_fedads_tags.py --src <fedads_sample 目录>
python src/ext/tune_proxy.py --tag fedads_val --max-rounds 300
python src/ext/train_federated_sgb_opt.py --tag fedads --he-schema ou --num-boost-round 20 --max-depth 3 --learning-rate 0.3 \
    --reg-lambda 10 --colsample 0.7 --sketch-eps 0.03 --base-score prior --run-id fedads_sgb_opt_base --seed 42
python src/ext/final_eval.py --tag fedads --tune data/processed/tune_proxy_fedads_val.json --fed optbase=data/processed/predictions/fedads_sgb_opt_base_predictions.csv
```
