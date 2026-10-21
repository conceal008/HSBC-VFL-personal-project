#!/bin/bash
# 串行队列（2 核机器上并行反而更慢）；每项幂等（日志含 EXIT 0 即跳过），被中断后重跑本脚本即可续跑
S=${S:-$(cd "$(dirname "$0")/.." && pwd)}  # 工作根目录：其下需有 repo/（federated_learning_new 检出 + 本目录 src/ext）与 env/
cd $S/repo
PY=$S/env/sf/bin/python
run(){ log=$1; shift; [ -f "$log" ] && grep -q "EXIT 0" "$log" && { echo "skip $log"; return; }; $PY src/ext/train_federated_sgb_opt.py --seed 42 --he-schema ou "$@" > "$log" 2>&1; echo "EXIT $?" >> "$log"; echo "done $log $(date +%T)"; }
A=data/processed/attack
run $S/sec_logs/S0_base.log    --tag business --run-id business_sgb_S0_base --num-boost-round 100 --dump-model $A/model_S0_base.json
run $S/opt_logs/V3b_goss_l10.log --tag business_val --run-id val_V3b_goss_l10 --num-boost-round 100 --enable-goss --top-rate 0.2 --bottom-rate 0.1 --reg-lambda 10
run $S/sec_logs/T2_opt.log     --tag business --run-id business_sgb_ou_opt_m42 --num-boost-round 134 --max-depth 6 --reg-lambda 10 --sketch-eps 0.03 --dump-model $A/model_T2_opt.json
run $S/sec_logs/S1_ft0.log     --tag business --run-id business_sgb_S1_ft0 --num-boost-round 100 --first-tree-label-holder 0 --dump-model $A/model_S1_ft0.json
run $S/sec_logs/S5_hostcnt.log --tag business_hostcnt --run-id business_hostcnt_sgb_S5 --num-boost-round 100 --dump-model $A/model_S5_hostcnt.json
run $S/sec_logs/S3_lrr2.log    --tag business --run-id business_sgb_S3_lrr2 --num-boost-round 100 --label-rr-eps 2 --dump-model $A/model_S3_lrr2.json
run $S/sec_logs/S6_hostrr2.log --tag business_hostrr2 --run-id business_hostrr2_sgb_S6 --num-boost-round 100 --dump-model $A/model_S6_hostrr2.json
run $S/sec_logs/S2_lrr1.log    --tag business --run-id business_sgb_S2_lrr1 --num-boost-round 100 --label-rr-eps 1 --dump-model $A/model_S2_lrr1.json
run $S/sec_logs/S4_lrr4.log    --tag business --run-id business_sgb_S4_lrr4 --num-boost-round 100 --label-rr-eps 4 --dump-model $A/model_S4_lrr4.json
run $S/sec_logs/T1_bins33.log  --tag business --run-id business_sgb_ou_bins33_m42 --num-boost-round 100 --sketch-eps 0.03
echo "QUEUE DONE $(date +%T)" >> $S/run_all.done
