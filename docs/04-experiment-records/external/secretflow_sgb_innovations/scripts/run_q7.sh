#!/bin/bash
# Orange churn：优化版（集中式臂调参配置 + 细分桶 + 先验初始分数）与原版默认（100 轮 / 深 5 / lr 0.3）各一次，再跑 A1/A2
S=${S:-$(cd "$(dirname "$0")/.." && pwd)}  # 工作根目录：其下需有 repo/（federated_learning_new 检出 + 本目录 src/ext）与 env/
cd $S/repo
until grep -q "Q6 DONE" $S/run_q6.out 2>/dev/null; do sleep 60; done
PY=$S/env/sf/bin/python
L=$S/orange_logs
A=data/processed/attack
run(){ log=$1; shift; [ -f "$log" ] && grep -q "EXIT 0" "$log" && { echo "skip $log"; return; }; $PY src/ext/train_federated_sgb_opt.py --seed 42 --he-schema ou "$@" > "$log" 2>&1; echo "EXIT $?" >> "$log"; echo "done $log $(date +%T)"; }
att(){ name=$1; tag=$2; al=$L/ATT_$name.log; [ -f "$al" ] && grep -q "EXIT 0" "$al" && return; [ -f $A/model_$name.json ] || return
  (cd src/ext && $S/env/plain/bin/python attacks.py --model ../../data/processed/attack/model_$name.json --tag $tag \
     --host-sees-first-tree 0 --known-frac 0.01 0.05 --name $name > $al 2>&1; echo "EXIT $?" >> $al); echo "attack done $name $(date +%T)"; }
run $L/O_churn_opt.log --tag orange_churn --run-id orange_churn_sgb_opt --num-boost-round 71 --max-depth 3 --learning-rate 0.1 --reg-lambda 0.1 --colsample 0.7 --sketch-eps 0.03 --base-score prior --dump-model $A/model_O_churn_opt.json
att O_churn_opt orange_churn
run $L/O_churn_def.log --tag orange_churn --run-id orange_churn_sgb_def --num-boost-round 100 --dump-model $A/model_O_churn_def.json
att O_churn_def orange_churn
echo "Q7 DONE $(date +%T)"
