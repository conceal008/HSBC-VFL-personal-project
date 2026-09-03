#!/usr/bin/env bash
# 门禁回归测试 · 一条命令跑完三个已实现门禁的全部用例
#
# 为什么需要它：门禁脚本"不误报"很容易验证，"能拦住"却容易假通过。
# 每次改门禁脚本都必须跑这个，否则不知道拦截能力是不是被改没了。
#
# 用法：bash ci/tests/run_gate_tests.sh
# 退出码：0 全部用例通过；1 有用例失败
#
# 注意：凡含个人标识形态或凭证形态的样本一律**运行时拼装**，绝不落盘提交——
# 否则门禁 1 扫描本仓库时会命中夹具自身，把真仓库拦下来。
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIX="$ROOT/ci/tests/fixtures"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0

# assert <用例名> <期望退出码> <输出文件> [必须出现的子串...] [-- 必须不出现的子串...]
assert() {
  local name="$1" want="$2" out="$3"; shift 3
  local got=$RC ok=1
  [ "$got" = "$want" ] || { ok=0; echo "   退出码 期望 $want 实得 $got"; }
  local needle refute=0
  for needle in "$@"; do
    if [ "$needle" = "--" ]; then refute=1; continue; fi
    if [ "$refute" = 0 ]; then
      grep -qF -- "$needle" "$out" || { ok=0; echo "   缺少预期输出：$needle"; }
    else
      grep -qF -- "$needle" "$out" && { ok=0; echo "   出现了不该出现的输出：$needle"; }
    fi
  done
  if [ "$ok" = 1 ]; then
    echo "✅ PASS  $name"; PASS=$((PASS+1))
  else
    echo "❌ FAIL  $name"; sed 's/^/       /' "$out" | head -20; FAIL=$((FAIL+1))
  fi
}

run() { "$@" > "$TMP/out.txt" 2>&1; RC=$?; }

echo "================ 门禁 1 · 数据合规扫描 ================"
C1="$TMP/gate1"; mkdir -p "$C1/ci" "$C1/modules/m5_modeling/results"
cp "$ROOT/ci/check_secrets.sh" "$C1/ci/"
# 干净目录 → 期望通过
printf 'ok,auc\nl1,0.671\n' > "$C1/modules/m5_modeling/results/ladder.csv"
run bash "$C1/ci/check_secrets.sh" "$C1"
assert "门禁1-A 干净仓库放行" 0 "$TMP/out.txt" "门禁 1 通过"

# 违规目录 → 期望阻断。标识与凭证一律拼装，不写字面量。
PHONE="138$(printf '%08d' 12345678)"
IDNO="440101$(printf '%08d' 19900101)1234"
CARD="6222$(printf '%015d' 21234567890123)"
KEY="abcd1234efgh5678ijkl"
printf '客户号,手机\nC001,%s\n' "$PHONE" > "$C1/leak.csv"
printf '身份证 %s 卡号 %s\n' "$IDNO" "$CARD" > "$C1/notes.md"
printf 'api_key = "%s"\n' "$KEY" > "$C1/conf.py"
: > "$C1/features.parquet"
run bash "$C1/ci/check_secrets.sh" "$C1"
assert "门禁1-B 拦截数据/凭证/标识" 1 "$TMP/out.txt" \
  "疑似内地手机号" "疑似身份证号" "疑似银行卡号" "疑似凭证" "数据/模型产物入库" "CSV 出现在非白名单路径"

# 误报回归：URL 里的长数字串（如政府网站文章 ID）不得被当成卡号
C1B="$TMP/gate1b"; mkdir -p "$C1B/ci"
cp "$ROOT/ci/check_secrets.sh" "$C1B/ci/"
printf '来源：<https://www.example.gov.cn/2024-03/22/c_%s.htm>\n' "1712776611775634" > "$C1B/refs.md"
run bash "$C1B/ci/check_secrets.sh" "$C1B"
assert "门禁1-C URL 里的长数字不误报为卡号" 0 "$TMP/out.txt" "门禁 1 通过" -- "疑似银行卡号"

# 卡号形态样本必须运行时拼装——写成字面量会被门禁 1 扫到本脚本自身
CARD2="6222$(printf '%015d' 21234567890123)"
printf '卡号 %s\n' "$CARD2" > "$C1B/leak.md"
run bash "$C1B/ci/check_secrets.sh" "$C1B"
assert "门禁1-D 非 URL 上下文的卡号仍被拦截" 1 "$TMP/out.txt" "疑似银行卡号"

# 误报回归：浮点数小数部分的长数字串不得被当成卡号（实验结果 CSV 的常态）
C1D="$TMP/gate1d"; mkdir -p "$C1D/ci"
cp "$ROOT/ci/check_secrets.sh" "$C1D/ci/"
# 18 位小数样本同样运行时拼装——写成字面量会被门禁 1 扫到本脚本自身
FRAC1="0280822629$(printf '%08d' 67478067)"
FRAC2="0306962634$(printf '%08d' 82280717)"
printf 'auc,gap\n0.%s,0.%s\n' "$FRAC1" "$FRAC2" > "$C1D/results.md"
run bash "$C1D/ci/check_secrets.sh" "$C1D"
assert "门禁1-G 浮点数小数部分不误报为卡号" 0 "$TMP/out.txt" "门禁 1 通过" -- "疑似银行卡号"

# 边界回归：卡号**后**跟句点仍必须拦截——本次只收紧了前导边界，不得连带放宽后随边界
CARD3="6225$(printf '%012d' 880137634567)"
printf 'Card %s.\n' "$CARD3" > "$C1D/leak.md"
run bash "$C1D/ci/check_secrets.sh" "$C1D"
assert "门禁1-H 卡号后跟句点仍被拦截" 1 "$TMP/out.txt" "疑似银行卡号"

# 误报回归：URL 路径里的 11 位 ID（如知乎文章号）不得被当成手机号
C1C="$TMP/gate1c"; mkdir -p "$C1C/ci"
cp "$ROOT/ci/check_secrets.sh" "$C1C/ci/"
ZHIHU_ID="19$(printf '%09d' 535338799)"
printf '来源：<https://zhuanlan.zhihu.com/p/%s>\n' "$ZHIHU_ID" > "$C1C/refs.md"
run bash "$C1C/ci/check_secrets.sh" "$C1C"
assert "门禁1-E URL 里的 11 位 ID 不误报为手机号" 0 "$TMP/out.txt" "门禁 1 通过" -- "疑似内地手机号"

PHONE2="139$(printf '%08d' 12345678)"
printf '联系人 %s\n' "$PHONE2" > "$C1C/leak.md"
run bash "$C1C/ci/check_secrets.sh" "$C1C"
assert "门禁1-F 非 URL 上下文的手机号仍被拦截" 1 "$TMP/out.txt" "疑似内地手机号"

echo
echo "================ 门禁 4 · 合规一致性核验 ================"
C4="$TMP/gate4"
mkdir -p "$C4/registry" "$C4/platform/governance" "$C4/modules/m5_modeling/components/splitnn"
cp "$ROOT/ci/check_cross_border_consistency.py" "$C4/"
cp "$FIX/gate4/cross_border_assets.yaml" "$C4/registry/"
cp "$FIX/gate4/compliance_routes.yaml" "$C4/registry/"
cp "$FIX/gate4/consent_batches.yaml" "$C4/platform/governance/"
cp "$FIX/gate4/decl_good.yaml" "$C4/modules/m5_modeling/components/splitnn/declaration.yaml"
run python3 "$C4/check_cross_border_consistency.py" "$C4"
assert "门禁4-A 全合规声明放行" 0 "$TMP/out.txt" "门禁 4 通过"

mkdir -p "$C4/modules/m3_alignment/components/psi" "$C4/modules/m7_security/components/attack" "$C4/party_cn/bridge"
cp "$FIX/gate4/decl_bad_refs.yaml" "$C4/modules/m3_alignment/components/psi/declaration.yaml"
cp "$FIX/gate4/decl_bad_route.yaml" "$C4/modules/m7_security/components/attack/declaration.yaml"
cp "$FIX/gate4/sender.py.txt" "$C4/party_cn/bridge/sender.py"
run python3 "$C4/check_cross_border_consistency.py" "$C4"
assert "门禁4-B 六项核验逐项拦截" 1 "$TMP/out.txt" \
  "[核验 1]" "[核验 2]" "[核验 3]" "[核验 4]" "[核验 5]" "[核验 6]" \
  "CBA-999" "已作废条目 CBA-009" "非选定路线 R-A"

rm -f "$C4/registry/compliance_routes.yaml" "$C4/platform/governance/consent_batches.yaml"
rm -rf "$C4/modules/m3_alignment" "$C4/modules/m7_security" "$C4/party_cn"
run python3 "$C4/check_cross_border_consistency.py" "$C4"
assert "门禁4-C 契约文件缺失不放行" 1 "$TMP/out.txt" \
  "compliance_routes.yaml 不存在" "consent_batches.yaml 不存在"

echo
echo "================ 门禁 2 · 步骤与元数据完整性 ================"
C2="$TMP/gate2"; mkdir -p "$C2/ci" "$C2/changelog" "$C2/registry/decision_records"
cp "$ROOT/ci/check_step_metadata.py" "$C2/ci/"
(
  cd "$C2"
  git init -q -b main && git config user.email t@t && git config user.name t
  echo seed > seed.txt && git add -A && git commit -q -F "$FIX/gate2/msg_a.txt"
  git rev-parse HEAD > "$TMP/sha_a"
  cp "$FIX/gate2/CL-20260830-M5-001.yaml" changelog/
  git add -A && git commit -q -F "$FIX/gate2/msg_b.txt"
  git rev-parse HEAD > "$TMP/sha_b"
  cp "$FIX/gate2/CL-20260830-M0-002.yaml" changelog/
  cp "$FIX/gate2/DR-M0-001.yaml" registry/decision_records/
  git add -A && git commit -q -F "$FIX/gate2/msg_c.txt"
  git rev-parse HEAD > "$TMP/sha_c"
) > /dev/null

run python3 "$C2/ci/check_step_metadata.py" --rev "$(cat "$TMP/sha_a")" "$C2"
assert "门禁2-A 无 trailer 提交拦截" 1 "$TMP/out.txt" \
  "首行不符合" "[Change-Id] 缺失" "[Step-Id] 缺失" "[Cross-Border] trailer 缺失"

run python3 "$C2/ci/check_step_metadata.py" --rev "$(cat "$TMP/sha_b")" "$C2"
assert "门禁2-B 元数据全面违规拦截" 1 "$TMP/out.txt" \
  "步骤声明为空" "修正轮次 3 > 2" "硬门 fail" "无证据引用" "有单项 ≤1" \
  "与四维之和" "commit trailer 写 11/12" "非法枚举" "type=exp 必须绑定分支卡" "conclusion 为空"

run python3 "$C2/ci/check_step_metadata.py" --rev "$(cat "$TMP/sha_c")" "$C2"
assert "门禁2-C decision 的 DR 无 falsifier" 1 "$TMP/out.txt" "缺 falsifier"

run python3 "$ROOT/ci/check_step_metadata.py" "$ROOT"
assert "门禁2-D 本仓库 HEAD 自检通过" 0 "$TMP/out.txt" "门禁 2 通过"

echo
echo "================ changelog schema 校验 ================"
CS="$TMP/schema"; mkdir -p "$CS/ci" "$CS/changelog"
cp "$ROOT/ci/check_changelog_schema.py" "$CS/ci/"
cp "$FIX/gate2/CL-20260830-M5-001.yaml" "$CS/changelog/"
run python3 "$CS/ci/check_changelog_schema.py" "$CS"
assert "schema-A 缺字段与非法枚举拦截" 1 "$TMP/out.txt" \
  "缺必填字段 what" "缺必填字段 why" "cross_border_impact='maybe' 非法" "必须记录 ≥5 个种子与 config 路径"

cp "$FIX/gate2/CL-20260830-M5-001.yaml" "$CS/changelog/CL-20260830-M5-009.yaml"
run python3 "$CS/ci/check_changelog_schema.py" "$CS"
assert "schema-B 文件名与 change_id 不符 + step_id 重复" 1 "$TMP/out.txt" \
  "与文件名不一致" "重复"

printf 'change_id: CL-20260831-M5-002\nmodule: M5\ntype: doc\ntitle: 援引未登记豁免\nwaiver_ref: W-999\ncross_border_impact: none\nwhat: x\nwhy: y\nlinks: {}\nsensitive_review: {triggered: false}\nbreaking_change: false\nreproducibility: {}\nrollback: x\nverification: x\ntimestamp: 2026-08-31T12:00:00+08:00\nagent: t\n' \
  > "$CS/changelog/CL-20260831-M5-002.yaml"
mkdir -p "$CS/registry"
printf 'waivers:\n  - waiver_id: W-001\n' > "$CS/registry/waivers.yaml"
run python3 "$CS/ci/check_changelog_schema.py" "$CS"
assert "schema-D 未登记的 waiver_ref 被拦截" 1 "$TMP/out.txt" "W-999 未在 registry/waivers.yaml 登记"

run python3 "$ROOT/ci/check_changelog_schema.py" "$ROOT"
assert "schema-C 本仓库全部条目通过" 0 "$TMP/out.txt" "changelog schema 校验通过"

echo
echo "================ 门禁 6 · 可复现性 ================"
C6="$TMP/gate6"
mkdir -p "$C6/ci" "$C6/modules/m5_modeling/components" "$C6/modules/m5_modeling/notebooks" "$C6/modules/m5_modeling/configs"
cp "$ROOT/ci/check_reproducibility.py" "$C6/ci/"
cp "$FIX/gate6/good_component.py.txt" "$C6/modules/m5_modeling/components/ladder.py"
cp "$FIX/gate6/nb_good.ipynb.txt" "$C6/modules/m5_modeling/notebooks/S5.4_l0_vs_l1.ipynb"
printf 'seed: 42\nseeds: [11, 22, 33, 44, 55]\nsample_size: 5000\nthreshold: 0.73\n' \
  > "$C6/modules/m5_modeling/configs/l1_v1.yaml"
run python3 "$C6/ci/check_reproducibility.py" "$C6"
assert "门禁6-A 合规代码与 notebook 放行" 0 "$TMP/out.txt" "门禁 6 通过" -- "无法解析为 Python"

cp "$FIX/gate6/bad_component.py.txt" "$C6/modules/m5_modeling/components/bad.py"
cp "$FIX/gate6/nb_bad.ipynb.txt" "$C6/modules/m5_modeling/notebooks/l0_vs_l1.ipynb"
run python3 "$C6/ci/check_reproducibility.py" "$C6"
assert "门禁6-B 拦截魔数/写死种子/命名/指纹/乱序" 1 "$TMP/out.txt" \
  "[R1]" "[R2]" "[R4]" "[R5]" "[R6]" \
  "硬编码魔数 5000" "种子写死为 42" "命名不符合 N1" "未打印运行指纹" "非单调递增" "未执行"

rm -f "$C6/modules/m5_modeling/components/bad.py" "$C6/modules/m5_modeling/notebooks/l0_vs_l1.ipynb"
cp "$FIX/gate6/requirements.txt.txt" "$C6/requirements.txt"
run python3 "$C6/ci/check_reproducibility.py" "$C6"
assert "门禁6-C requirements 无 lock 文件即阻断" 1 "$TMP/out.txt" "没有任何 lock 文件"

mkdir -p "$C6/environments"
printf 'numpy==1.26.4\npandas==2.2.2\n' > "$C6/environments/m5.lock"
run python3 "$C6/ci/check_reproducibility.py" "$C6"
assert "门禁6-D lock 与 requirements 一致则放行" 0 "$TMP/out.txt" "门禁 6 通过"

run python3 "$ROOT/ci/check_reproducibility.py" "$ROOT"
assert "门禁6-E 本仓库自检通过" 0 "$TMP/out.txt" "门禁 6 通过"

echo
echo "================ 汇总 ================"
echo "PASS=$PASS  FAIL=$FAIL"
[ "$FAIL" = 0 ] || { echo "门禁回归测试未通过。"; exit 1; }
echo "门禁回归测试全部通过。"
