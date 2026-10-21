#!/bin/bash
# IEEE-CIS：优化版（集中式臂调参配置 112 轮 / 深 6 / λ10 + 33 桶 + 先验初始分数）→ A1/A2 → 原版默认（100 轮 / 深 5）→ A1/A2
S=${S:-$(cd "$(dirname "$0")/.." && pwd)}  # 工作根目录：其下需有 repo/（federated_learning_new 检出 + 本目录 src/ext）与 env/
cd $S/repo
until grep -q "Q7 DONE" $S/run_q7.out 2>/dev/null; do sleep 30; done
PY=$S/env/sf/bin/python
L=$S/ieee_logs
A=data/processed/attack
run(){ log=$1; shift; [ -f "$log" ] && grep -q "EXIT 0" "$log" && { echo "skip $log"; return; }; $PY src/ext/train_federated_sgb_opt.py --seed 42 --he-schema ou "$@" > "$log" 2>&1; echo "EXIT $?" >> "$log"; echo "done $log $(date +%T)"; }
att(){ name=$1; tag=$2; al=$L/ATT_$name.log; [ -f "$al" ] && grep -q "EXIT 0" "$al" && return; [ -f $A/model_$name.json ] || return
  (cd src/ext && $S/env/plain/bin/python attacks.py --model ../../data/processed/attack/model_$name.json --tag $tag \
     --host-sees-first-tree 0 --known-frac 0.01 0.05 --name $name > $al 2>&1; echo "EXIT $?" >> $al); echo "attack done $name $(date +%T)"; }
run $L/I_opt.log --tag ieee --run-id ieee_sgb_opt --num-boost-round 112 --max-depth 6 --learning-rate 0.1 --reg-lambda 10 --sketch-eps 0.03 --base-score prior --dump-model $A/model_I_opt.json
att I_opt ieee
run $L/I_def.log --tag ieee --run-id ieee_sgb_def --num-boost-round 100 --dump-model $A/model_I_def.json
att I_def ieee
echo "Q9 DONE $(date +%T)"
