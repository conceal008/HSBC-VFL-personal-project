#!/bin/bash
# HSBC mock 四档：原仓库口径（默认 XGB 三臂 + 联邦配对）、调参三臂终评、攻击。只对已完成的联邦运行做评估。
S=${S:-$(cd "$(dirname "$0")/.." && pwd)}  # 工作根目录：其下需有 repo/（federated_learning_new 检出 + 本目录 src/ext）与 env/
cd $S/repo
PY=$S/env/plain/bin/python
P=data/processed/predictions
for t in ${@:-balanced overlap80 overlap50 overlap20}; do
  def=$P/mock_${t}_sgb_def_predictions.csv
  opt=$P/mock_${t}_sgb_opt_predictions.csv
  [ -f $def ] || { echo "[$t] 默认配置联邦还没跑完，跳过"; continue; }
  echo "================ mock_$t"
  $PY src/cross_sell/train_local.py --model xgb --scope all --tag mock_$t --compare-with $def 2>&1 | grep -E "AUROC|ΔAUC"
  feds="--fed def=$def"; [ -f $opt ] && feds="$feds --fed opt=$opt"
  $PY src/ext/final_eval.py --tag mock_$t $feds 2>&1 | grep -E "AUROC|−"
  (cd src/ext && $PY attacks.py --model ../../data/processed/attack/model_M_${t}_def.json --tag mock_$t --truth-tag mock_$t \
     --host-sees-first-tree 0 --known-frac 0.01 0.05 --name M_${t}_def 2>&1 | grep -E "A1|对照真值")
  if [ -f data/processed/attack/model_M_${t}_opt.json ]; then
    (cd src/ext && $PY attacks.py --model ../../data/processed/attack/model_M_${t}_opt.json --tag mock_$t --truth-tag mock_$t \
       --host-sees-first-tree 0 --known-frac 0.01 0.05 --name M_${t}_opt 2>&1 | grep -E "A1|对照真值")
  fi
done
