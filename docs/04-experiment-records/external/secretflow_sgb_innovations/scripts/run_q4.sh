#!/bin/bash
# 第四批续：FedAds 先验初始分数 → 原版默认配置。幂等。
S=${S:-$(cd "$(dirname "$0")/.." && pwd)}  # 工作根目录：其下需有 repo/（federated_learning_new 检出 + 本目录 src/ext）与 env/
cd $S/repo
PY=$S/env/sf/bin/python
L=$S/fed_logs
A=data/processed/attack
run(){ log=$1; shift; [ -f "$log" ] && grep -q "EXIT 0" "$log" && { echo "skip $log"; return; }; $PY src/ext/train_federated_sgb_opt.py --seed 42 --he-schema ou "$@" > "$log" 2>&1; echo "EXIT $?" >> "$log"; echo "done $log $(date +%T)"; }
TUNED="--num-boost-round 20 --max-depth 3 --learning-rate 0.3 --reg-lambda 10 --colsample 0.7"
run $L/F_opt_base.log --tag fedads --run-id fedads_sgb_opt_base $TUNED --sketch-eps 0.03 --base-score prior --dump-model $A/model_F_opt_base.json
run $L/F_def.log      --tag fedads --run-id fedads_sgb_def      --num-boost-round 100 --dump-model $A/model_F_def.json
echo "QUEUE4 DONE $(date +%T)" >> $S/run_all.done
