#!/bin/bash
# Orange upsell：优化版（集中式臂调参配置 23 轮 / 深 6 / λ10）与原版默认各一次，再跑 A1/A2
S=${S:-$(cd "$(dirname "$0")/.." && pwd)}  # 工作根目录：其下需有 repo/（federated_learning_new 检出 + 本目录 src/ext）与 env/
cd $S/repo
until grep -q "Q7 DONE" $S/run_q7.out 2>/dev/null; do sleep 60; done
PY=$S/env/sf/bin/python
L=$S/orange_logs
A=data/processed/attack
run(){ log=$1; shift; [ -f "$log" ] && grep -q "EXIT 0" "$log" && { echo "skip $log"; return; }; $PY src/ext/train_federated_sgb_opt.py --seed 42 --he-schema ou "$@" > "$log" 2>&1; echo "EXIT $?" >> "$log"; echo "done $log $(date +%T)"; }
att(){ name=$1; tag=$2; al=$L/ATT_$name.log; [ -f "$al" ] && grep -q "EXIT 0" "$al" && return; [ -f $A/model_$name.json ] || return
  (cd src/ext && $S/env/plain/bin/python attacks.py --model ../../data/processed/attack/model_$name.json --tag $tag \
     --host-sees-first-tree 0 --known-frac 0.01 0.05 --name $name > $al 2>&1; echo "EXIT $?" >> $al); echo "attack done $name $(date +%T)"; }
run $L/O_upsell_opt.log --tag orange_upsell --run-id orange_upsell_sgb_opt --num-boost-round 23 --max-depth 6 --learning-rate 0.1 --reg-lambda 10 --colsample 0.7 --sketch-eps 0.03 --base-score prior --dump-model $A/model_O_upsell_opt.json
att O_upsell_opt orange_upsell
run $L/O_upsell_def.log --tag orange_upsell --run-id orange_upsell_sgb_def --num-boost-round 100 --dump-model $A/model_O_upsell_def.json
att O_upsell_def orange_upsell
echo "Q8 DONE $(date +%T)"
