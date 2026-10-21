#!/bin/bash
# VertiBench 切分复核：低 / 高香港重要性占比两个切分各跑一次优化版 SGB（同 T2 配置），再跑 A1/A2
S=${S:-$(cd "$(dirname "$0")/.." && pwd)}  # 工作根目录：其下需有 repo/（federated_learning_new 检出 + 本目录 src/ext）与 env/
cd $S/repo
PY=$S/env/sf/bin/python
mkdir -p $S/vb_logs
run(){ log=$1; shift; [ -f "$log" ] && grep -q "EXIT 0" "$log" && { echo "skip $log"; return; }; $PY src/ext/train_federated_sgb_opt.py --seed 42 --he-schema ou "$@" > "$log" 2>&1; echo "EXIT $?" >> "$log"; echo "done $log $(date +%T)"; }
A=data/processed/attack
for t in lo hi; do
  run $S/vb_logs/SGB_vb_$t.log --tag vb_$t --run-id vb_${t}_sgb_opt --num-boost-round 134 --max-depth 6 --reg-lambda 10 --sketch-eps 0.03 --dump-model $A/model_vb_$t.json
  al=$S/vb_logs/ATT_vb_$t.log
  if [ -f $A/model_vb_$t.json ] && ! grep -q "EXIT 0" $al 2>/dev/null; then
    (cd src/ext && $S/env/plain/bin/python attacks.py --model ../../data/processed/attack/model_vb_$t.json --tag vb_$t \
        --host-sees-first-tree 0 --known-frac 0.01 0.05 --name vb_$t > $al 2>&1; echo "EXIT $?" >> $al)
    echo "attack done $t $(date +%T)"
  fi
done
echo "Q6 DONE $(date +%T)"
