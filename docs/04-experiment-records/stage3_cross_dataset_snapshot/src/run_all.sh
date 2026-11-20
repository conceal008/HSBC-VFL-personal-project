#!/bin/zsh
# 第三阶段一键复现脚本（顺序执行；每个实验结果立即写入 results/，日志写入 logs/）
# 用法：zsh src/run_all.sh [proxy|sgb|all]
set -u
ROOT=/Applications/总文件夹/汇丰/第三阶段
PY=/opt/anaconda3/bin/python3
SFPY=/Applications/总文件夹/汇丰/8-9月/隐语_SecretFlow/.venv/bin/python
MODE=${1:-all}
cd $ROOT

run() {  # $1=python $2=script
  local name=$(basename $2 .py)
  if [[ -f "$ROOT/results/.done_$name" ]]; then echo "skip $name (done)"; return; fi
  echo "[$(date +%T)] START $name"
  $1 $ROOT/src/$2 > $ROOT/logs/$name.log 2>&1 && touch $ROOT/results/.done_$name
  echo "[$(date +%T)] END   $name (exit $?)"
}

if [[ $MODE == prep || $MODE == all ]]; then run $PY prep_cache.py; fi
if [[ $MODE == proxy || $MODE == all ]]; then
  run $PY exp1_x2_multiparty.py
  run $PY exp2_x1plus_temporal_graph.py
  run $PY exp3_m1x_unaligned.py
  run $PY exp4_x3_expected_value.py
  run $PY exp5_c2_seed_lookalike.py
fi
if [[ $MODE == sgb || $MODE == all ]]; then
  run $SFPY exp1_sgb_confirm.py
  run $SFPY exp2_sgb_confirm.py
  run $SFPY exp3_sgb_confirm.py
  run $SFPY exp6_core_sgb_real.py
fi
$PY $ROOT/src/aggregate_results.py
