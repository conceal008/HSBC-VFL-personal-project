#!/bin/bash
# 第三批：后验去噪标签 DP（rr_soft）+ HSBC mock 四档。每项幂等（日志含 EXIT 0 即跳过），中断后重跑即续。
S=${S:-$(cd "$(dirname "$0")/.." && pwd)}  # 工作根目录：其下需有 repo/（federated_learning_new 检出 + 本目录 src/ext）与 env/
cd $S/repo
PY=$S/env/sf/bin/python
L=$S/sec2_logs
A=data/processed/attack
run(){ log=$1; shift; [ -f "$log" ] && grep -q "EXIT 0" "$log" && { echo "skip $log"; return; }; $PY src/ext/train_federated_sgb_opt.py --seed 42 --he-schema ou "$@" > "$log" 2>&1; echo "EXIT $?" >> "$log"; echo "done $log $(date +%T)"; }
OPT="--num-boost-round 134 --max-depth 6 --reg-lambda 10 --sketch-eps 0.03"
run $L/D4_rrsoft4.log --tag business --run-id business_sgb_D4_rrsoft4 --num-boost-round 100 --label-rr-eps 4 --label-dp rr_soft --dump-model $A/model_D4_rrsoft4.json
run $L/D2_rrsoft2.log --tag business --run-id business_sgb_D2_rrsoft2 --num-boost-round 100 --label-rr-eps 2 --label-dp rr_soft --dump-model $A/model_D2_rrsoft2.json
run $L/D1_rrsoft1.log --tag business --run-id business_sgb_D1_rrsoft1 --num-boost-round 100 --label-rr-eps 1 --label-dp rr_soft --dump-model $A/model_D1_rrsoft1.json
for t in balanced overlap80 overlap50 overlap20; do
  run $L/M_${t}_def.log --tag mock_$t --run-id mock_${t}_sgb_def --num-boost-round 100 --dump-model $A/model_M_${t}_def.json
done
run $L/M_balanced_opt.log  --tag mock_balanced  --run-id mock_balanced_sgb_opt  $OPT --dump-model $A/model_M_balanced_opt.json
run $L/M_overlap20_opt.log --tag mock_overlap20 --run-id mock_overlap20_sgb_opt $OPT --dump-model $A/model_M_overlap20_opt.json
run $L/M_overlap50_opt.log --tag mock_overlap50 --run-id mock_overlap50_sgb_opt $OPT --dump-model $A/model_M_overlap50_opt.json
echo "QUEUE2 DONE $(date +%T)" >> $S/run_all.done
