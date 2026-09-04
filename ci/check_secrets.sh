#!/usr/bin/env bash
# 门禁 1 · 数据合规扫描（《项目维护约束 v2》第 8 部分）
# 这是唯一必须在"第一次提交"就生效的门禁：它防的是真实数据与凭证入库，
# 而这是本项目唯一不可逆的错误（2.4：用新提交"覆盖"没有用，git 历史里它还在）。
#
# 用法：ci/check_secrets.sh [目标目录，默认为仓库根]
# 退出码：0 通过；1 阻断（BLOCK）。WARN 不影响退出码。
set -o pipefail   # 不用 set -u：需兼容 macOS 自带 bash 3.2 的空数组语义

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

BLOCK=0
WARN=0
SELF="ci/check_secrets.sh"

say_block() { echo "❌ BLOCK  $*"; BLOCK=$((BLOCK+1)); }
say_warn()  { echo "⚠️  WARN   $*"; WARN=$((WARN+1)); }

# 扫描对象：git 跟踪的文件（CI 场景）；不在 git 里时退回全目录
FILES=()
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  while IFS= read -r line; do FILES+=("$line"); done < <(git ls-files)
else
  while IFS= read -r line; do FILES+=("$line"); done < <(find . -type f -not -path "./.git/*" | sed 's|^\./||')
fi
if [ ${#FILES[@]} -eq 0 ]; then echo "无可扫描文件，跳过。"; exit 0; fi

echo "== 门禁 1 · 数据合规扫描 =="
echo "扫描根目录：$ROOT"
echo "文件数：${#FILES[@]}"
echo

# ---------- 1. 扩展名黑名单（真实/合成数据一律不入库）----------
# 白名单：modules/*/results/ 下的 csv 是"结果表"，允许；其余路径的数据类扩展名一律阻断。
for f in "${FILES[@]}"; do
  case "$f" in
    *.parquet|*.feather|*.pkl|*.npy|*.h5|*.csv.raw|*.pt|*.pth|*.ckpt|*.onnx)
      say_block "数据/模型产物入库：$f（第 2.1 条 L-禁止；合成数据也不提交，只提交生成器+配置+种子）" ;;
    *.csv)
      case "$f" in
        modules/*/results/*) : ;;   # 结果表白名单
        *) say_block "CSV 出现在非白名单路径：$f（只有 modules/*/results/ 下的结果表可提交）" ;;
      esac ;;
  esac
done

# ---------- 2. 凭证类文件 ----------
for f in "${FILES[@]}"; do
  b="$(basename "$f")"
  case "$b" in
    .env|.env.*) [ "$b" = ".env.example" ] || say_block "凭证文件入库：$f" ;;
    *.pem|*.key|id_rsa|id_ed25519) say_block "密钥文件入库：$f" ;;
    credentials*|secrets*) say_block "凭证文件入库：$f" ;;
  esac
done

# ---------- 3. 文件体积 ----------
for f in "${FILES[@]}"; do
  [ -f "$f" ] || continue
  sz=$(wc -c < "$f" | tr -d ' ')
  if   [ "$sz" -gt 52428800 ]; then say_block "文件 >50MB：$f（$((sz/1048576))MB）"
  elif [ "$sz" -gt 10485760 ]; then say_warn  "文件 >10MB：$f（$((sz/1048576))MB）"
  fi
done

# ---------- 4. 密钥/令牌模式 ----------
SECRET_PATTERNS=(
  'BEGIN (RSA|EC|OPENSSH|PGP|DSA) PRIVATE KEY'
  'AKIA[0-9A-Z]{16}'
  'gh[pousr]_[A-Za-z0-9]{30,}'
  'sk-[A-Za-z0-9]{20,}'
  'xox[baprs]-[A-Za-z0-9-]{10,}'
  '(api[_-]?key|secret|token|password|passwd)[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"']{12,}["'"'"']'
)
for f in "${FILES[@]}"; do
  [ -f "$f" ] || continue
  [ "$f" = "$SELF" ] && continue
  file "$f" | grep -qi "text" || continue
  for p in "${SECRET_PATTERNS[@]}"; do
    if grep -nEi "$p" "$f" >/dev/null 2>&1; then
      say_block "疑似凭证：$f（模式 /$p/）→ 命中即按 2.4 应急流程：先轮换凭证，再清历史"
    fi
  done
done

# ---------- 5. 个人标识模式 ----------
# 内地手机号 / 18 位身份证 / 银行卡号 / 香港身份证号
# 格式：模式@@标签@@跳过行的正则（可空）
# 「跳过行」是为了压制可判定的误报：URL 里的文章 ID、哈希值等长数字串不是卡号或手机号。
#
# ⚠️ 累积代价（第二次放宽时记录，2026-09-01）：
#   银行卡号与内地手机号两类模式现均跳过含 URL 的行。
#   这意味着**写在 URL 里的真实卡号或手机号会被漏掉**（例如 ?phone=138xxxxxxxx 形式的查询串）。
#   接受这个代价的理由：本仓库内容是文档与代码，真实标识若泄露几乎必然在数据文件中，
#   而数据类扩展名已被本脚本按路径整体阻断；凭证类另有 GitHub push protection 兜底。
#   **若日后仓库开始承载含 URL 查询串的日志或抓取结果，本条豁免必须重新评估。**
#   身份证号与香港身份证号两类模式**未放宽**，仍对 URL 行生效。
#
# ⚠️ 累积代价（第三次放宽时记录，2026-09-03）：
#   银行卡号模式的**前导**边界由 [^0-9] 收紧为 [^0-9.]，即紧跟在小数点后的长数字串不再判为卡号。
#   起因：实验结果 CSV 中的浮点数（如 0.028082262967478067）小数部分有 16–19 位连续数字，
#   逐个触发误报，而这类文件是本项目的核心产出，不可能不入库。
#   这意味着**紧接在小数点之后、中间无分隔的卡号会被漏掉**（例如「卡号.6225xxxxxxxxxxxx」）。
#   **后随**边界未动，仍为 [^0-9]，因此「卡号 6225xxxxxxxxxxxx。」这类正常写法照常拦截。
#   接受这个代价的理由：卡号紧跟小数点书写在真实文本中几乎不出现，
#   而浮点数产出是本仓库的常态；两者的误报/漏报比悬殊。
#   **若日后仓库开始承载可能以小数点为分隔符的标识串，本条豁免必须重新评估。**
#
# ⚠️ 累积代价（第四次放宽时记录，2026-09-03）：
#   内地手机号模式的**前导**边界同样由 [^0-9] 收紧为 [^0-9.]，原因与卡号那次相同——
#   实验结果 CSV 中的浮点数（如 175999.18587877366）小数部分含 11 位以 1 开头的数字串。
#   代价对称：紧接小数点之后、中间无分隔的手机号会被漏掉；后随边界未动。
#   **注意这已是第二次因实验结果 CSV 触发同类误报。** 若再出现第三次，
#   应当重新审视的不是边界规则，而是「长数字串标识模式」这条检查在数值产物上是否还适用——
#   届时的正确做法可能是按文件类型分流（数值型结果表只查凭证与真实 ID 字段名），
#   而不是继续逐个模式打补丁。
#   身份证号与香港身份证号两类模式**始终未放宽**。
ID_PATTERNS=(
  '(^|[^0-9.])1[3-9][0-9]{9}([^0-9]|$)@@内地手机号@@https?://'
  '(^|[^0-9A-Za-z])[1-9][0-9]{5}(19|20)[0-9]{2}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])[0-9]{3}[0-9Xx]([^0-9A-Za-z]|$)@@身份证号@@'
  '(^|[^0-9.])[0-9]{16,19}([^0-9]|$)@@银行卡号@@https?://'
  '(^|[^0-9A-Za-z])[A-Z]{1,2}[0-9]{6}\([0-9A]\)@@香港身份证号@@'
)
for f in "${FILES[@]}"; do
  [ -f "$f" ] || continue
  [ "$f" = "$SELF" ] && continue
  file "$f" | grep -qi "text" || continue
  for entry in "${ID_PATTERNS[@]}"; do
    p="${entry%%@@*}"
    rest="${entry#*@@}"
    label="${rest%%@@*}"
    skipre="${rest#*@@}"
    hit=$(grep -nE "$p" "$f" 2>/dev/null)
    if [ -n "$skipre" ] && [ -n "$hit" ]; then
      hit=$(printf '%s\n' "$hit" | grep -vE "$skipre" || true)
    fi
    hit=$(printf '%s' "$hit" | head -3)
    if [ -n "$hit" ]; then
      say_block "疑似$label：$f"
      echo "$hit" | sed 's/^/          /'
    fi
  done
done

echo
echo "== 结果：BLOCK=$BLOCK  WARN=$WARN =="
if [ "$BLOCK" -gt 0 ]; then
  echo "门禁 1 未通过。若已推送到远端，先按《项目维护约束 v2》2.4 轮换凭证并清理历史，再修复。"
  exit 1
fi
echo "门禁 1 通过。"
exit 0
