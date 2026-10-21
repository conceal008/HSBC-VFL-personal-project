#!/bin/bash
# 第四批：FedAds（30 万训练 / 20 万测试抽样）。每项幂等，中断后重跑即续。
S=${S:-$(cd "$(dirname "$0")/.." && pwd)}  # 工作根目录：其下需有 repo/（federated_learning_new 检出 + 本目录 src/ext）与 env/
cd $S/repo
PY=$S/env/sf/bin/python
L=$S/fed_logs
A=data/processed/attack
run(){ log=$1; shift; [ -f "$log" ] && grep -q "EXIT 0" "$log" && { echo "skip $log"; return; }; $PY src/ext/train_federated_sgb_opt.py --seed 42 --he-schema ou "$@" > "$log" 2>&1; echo "EXIT $?" >> "$log"; echo "done $log $(date +%T)"; }
TUNED="--num-boost-round 20 --max-depth 3 --learning-rate 0.3 --reg-lambda 10 --colsample 0.7"
run $L/F_opt.log     --tag fedads --run-id fedads_sgb_opt     $TUNED --sketch-eps 0.03 --dump-model $A/model_F_opt.json
run $L/F_tuned10.log --tag fedads --run-id fedads_sgb_tuned10 $TUNED
run $L/F_def.log     --tag fedads --run-id fedads_sgb_def     --num-boost-round 100 --dump-model $A/model_F_def.json
echo "QUEUE3 DONE $(date +%T)" >> $S/run_all.done
