#!/usr/bin/env bash
# 提交/推送前的唯一入口：按顺序跑全部已实现门禁 + 回归套件。
#
# 为什么需要它：把门禁写成 `bash ci/check_secrets.sh | tail -1 && ...` 这类形式时，
# 管道的退出码是 tail 的（永远 0），&& 链形同虚设——门禁失败了后续照样执行，
# 包括 git push。本项目已因此两次把不通过门禁的提交推进 main。
# 本脚本用显式退出码检查替代管道，并保留完整输出。
#
# 用法：bash ci/run_all_gates.sh && git push origin main
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FAILED=()

run_gate() {
  local name="$1"; shift
  echo "───────────── $name ─────────────"
  if "$@"; then
    echo "✅ $name"
  else
    echo "❌ $name（退出码 $?）"
    FAILED+=("$name")
  fi
  echo
}

run_gate "门禁 1 · 数据合规扫描"        bash ci/check_secrets.sh
run_gate "门禁 2 · 步骤与元数据"        python3 ci/check_step_metadata.py
run_gate "门禁 2 · changelog schema"    python3 ci/check_changelog_schema.py
run_gate "门禁 4 · 合规一致性核验"      python3 ci/check_cross_border_consistency.py
run_gate "门禁 5 · 代码质量"            python3 ci/check_code_quality.py
run_gate "门禁 6 · 可复现性"            python3 ci/check_reproducibility.py
run_gate "门禁 7 · 端到端冒烟"          python3 ci/check_e2e_smoke.py
run_gate "M9 · 证据链核验"              python3 ci/check_evidence_chain.py
run_gate "门禁回归测试"                 bash ci/tests/run_gate_tests.sh

echo "═════════════════════════════════"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "✅ 全部门禁通过，可以提交与推送。"
  exit 0
fi
echo "❌ 未通过：${FAILED[*]}"
echo "禁止推送。先停在原地诊断——不要绕过门禁手动 push。"
exit 1
