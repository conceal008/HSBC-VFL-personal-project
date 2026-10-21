#!/bin/bash
# 第五批：Santander 优化版 + 先验初始分数。幂等。
S=${S:-$(cd "$(dirname "$0")/.." && pwd)}  # 工作根目录：其下需有 repo/（federated_learning_new 检出 + 本目录 src/ext）与 env/
cd $S/repo
PY=$S/env/sf/bin/python
L=$S/sec_logs
run(){ log=$1; shift; [ -f "$log" ] && grep -q "EXIT 0" "$log" && { echo "skip $log"; return; }; $PY src/ext/train_federated_sgb_opt.py --seed 42 --he-schema ou "$@" > "$log" 2>&1; echo "EXIT $?" >> "$log"; echo "done $log $(date +%T)"; }
run $L/T4_opt_base.log --tag business --run-id business_sgb_ou_opt_base_m42 --num-boost-round 134 --max-depth 6 --reg-lambda 10 --sketch-eps 0.03 --base-score prior --dump-model data/processed/attack/model_T4_opt_base.json
echo "QUEUE5 DONE $(date +%T)" >> $S/run_all.done
