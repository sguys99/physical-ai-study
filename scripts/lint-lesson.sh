#!/usr/bin/env bash
# lint-lesson.sh — lesson.md 집필 규약 검사
#
# SSOT: docs/course-plan.md §3.1 frontmatter · §3.2 시각자료 · §3.7 문서 구조
#       §3.8 용어 도입 · §3.9 이해 사다리 · §3.10 문장부호 · §4 분량
# 근거: docs/course-plan.md §9.11 (2026-08-09 규약 개정)
#
# 사용법:
#   bash scripts/lint-lesson.sh course/w1-generative-core/01-physical-ai-landscape/lesson.md
#   bash scripts/lint-lesson.sh --all
#
# 종료 코드: FAIL이 하나라도 있으면 1, 아니면 0. WARN은 0으로 통과.

set -uo pipefail

PASS=0; FAIL=0; WARN=0

c_pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; PASS=$((PASS+1)); }
c_fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; FAIL=$((FAIL+1)); }
c_warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; WARN=$((WARN+1)); }
c_info() { printf '  ----  %s\n' "$1"; }

# ── 산문 문자 · 보호 구간 계산 ────────────────────────────────────
# 산문 = 총 문자 − 구조물(frontmatter + 코드펜스 + $$ 블록 + 표 행)
# 보호 구간 = 구조물 + 인라인 수식
#
# ⚠️ 알고리즘은 docs/course-plan.md §9.6의 계산 스크립트와 **글자 그대로 동일**해야 합니다.
#    §9.2·§9.4·§9.6·§9.7에 기록된 실측값과 §4 밴드, 그리고 독해 회귀식
#    (분 = 19.5 + 0.001186 × 산문)이 전부 이 정의 위에 서 있습니다.
#    헤딩을 빼거나 공백을 지우면 값이 ~25% 낮아져 과거 기록과 비교 불가가 됩니다.
prose_metrics() {
  python3 - "$1" <<'PY'
import re, sys
s = open(sys.argv[1], encoding='utf-8').read()
total = len(s)
fm = re.match(r'^---\n.*?\n---\n', s, re.S)
struct = len(fm.group(0)) if fm else 0
for m in re.finditer(r'```.*?```', s, re.S): struct += len(m.group(0))
body = re.sub(r'```.*?```', '', s, flags=re.S)
for m in re.finditer(r'\$\$.*?\$\$', body, re.S): struct += len(m.group(0))
body2 = re.sub(r'\$\$.*?\$\$', '', body, flags=re.S)
struct += sum(len(l) + 1 for l in body2.split('\n') if l.strip().startswith('|'))
inline = sum(len(m.group(0)) for m in re.finditer(r'(?<!\$)\$[^$\n]+\$(?!\$)', body2))
prose = total - struct
print(f"{prose} {total} {round(100*(struct+inline)/total)}")
PY
}

lint_one() {
  local f="$1"
  printf '\n\033[1m▐ %s\033[0m\n' "$f"

  if [[ ! -f "$f" ]]; then c_fail "파일이 없습니다"; return; fi

  local tier prose total protect
  tier=$(awk 'NR<=20 && /^tier:/ {print $2; exit}' "$f")
  read -r prose total protect < <(prose_metrics "$f")

  # ── A. frontmatter ──────────────────────────────────────────
  printf '\n  \033[1mA. frontmatter (§3.1)\033[0m\n'
  local missing=""
  for k in module week order title slug tier priority prereq tags est_reading_min updated sources_checked; do
    grep -qE "^${k}:" "$f" || missing="$missing $k"
  done
  [[ -z "$missing" ]] && c_pass "필수 12필드 완비" || c_fail "누락 필드:$missing"

  # ── B. 문서 구조 §3.7 ───────────────────────────────────────
  printf '\n  \033[1mB. 문서 구조 (§3.7)\033[0m\n'
  grep -qE '^## 0\..*이 모듈 지도' "$f" \
    && c_pass "§0 이 모듈 지도 존재" \
    || c_fail "§0 '이 모듈 지도' 없음 — 목차·오늘의 질문·모듈 지도가 들어가는 필수 섹션"
  grep -qE '^## .*한 장 정리' "$f" \
    && c_pass "「한 장 정리」 존재" \
    || c_fail "「한 장 정리」 없음 — 마무리 요약이 필수입니다"
  grep -qE '^## .*회사 스택 연결' "$f" \
    && c_pass "회사 스택 연결 존재" \
    || c_fail "회사 스택 연결 섹션 없음 (없으면 미완성)"
  grep -qE '^## .*(셀프 체크|퀴즈)' "$f" \
    && c_pass "셀프 체크 퀴즈 존재" || c_fail "셀프 체크 퀴즈 없음"
  grep -q '<details>' "$f" \
    && c_pass "퀴즈 정답 <details> 접힘" || c_fail "퀴즈 정답이 <details> 안에 없음"
  grep -qE '^## .*출처' "$f" && c_pass "출처 존재" || c_fail "출처 섹션 없음"

  # 대절별 📌 정리 박스 — 본문 절만 대상(부록성 절은 제외)
  local body_h2 pin_cnt
  body_h2=$(grep -cE '^## [0-9]+\.' "$f")
  pin_cnt=$(grep -c '📌 \*\*여기까지 정리\*\*' "$f")
  if [[ "$body_h2" -eq 0 ]]; then
    c_warn "번호 붙은 본문 대절이 없습니다 (§3.7 골격 확인)"
  elif [[ "$pin_cnt" -ge "$body_h2" ]]; then
    c_pass "📌 정리 박스 ${pin_cnt}개 / 본문 대절 ${body_h2}개"
  else
    c_fail "📌 정리 박스 부족: ${pin_cnt}개 / 본문 대절 ${body_h2}개 — 대절마다 3줄 요약"
  fi

  # ── C. 용어 도입 §3.8 ───────────────────────────────────────
  printf '\n  \033[1mC. 용어 도입 (§3.8)\033[0m\n'

  # 용어표: '| 용어 | ... | 비유 |' 헤더 행
  local termtbl
  termtbl=$(grep -cE '^\|[[:space:]]*용어[[:space:]]*\|' "$f")
  if [[ "$termtbl" -ge 1 ]]; then
    c_pass "절 시작 용어표 ${termtbl}개"
  else
    c_fail "절 시작 용어표 없음 — 대절마다 '| 용어 | 한 줄 정의 | 제어·LLM 비유 |'"
  fi

  # 약어 최초 전개 — 과거에 한 번도 전개되지 않은 것들
  local unexpanded=""
  for ab in MPC WBC VLA NFE RSSM DDS SLAM PPO IDM DoF BPE ELBO; do
    if grep -q "\b${ab}\b" "$f"; then
      # 같은 줄에 괄호 전개가 있는지 (영문 풀네임 또는 한글 설명)
      grep -qE "${ab}[[:space:]]*\(" "$f" || unexpanded="$unexpanded $ab"
    fi
  done
  [[ -z "$unexpanded" ]] \
    && c_pass "등장 약어가 모두 1회 이상 전개됨" \
    || c_fail "전개 없이 쓰인 약어:$unexpanded  → 최초 1회 'MPC(Model Predictive Control, 모델 예측 제어)'"

  # 한 줄 요약의 용어 밀도 — 영문 대문자 약어가 섞여 있으면 경고
  local summary_line
  summary_line=$(grep -m1 '^> \*\*한 줄 요약\*\*' "$f" || true)
  if [[ -n "$summary_line" ]]; then
    local jargon
    jargon=$(printf '%s' "$summary_line" | grep -oE '\b[A-Z]{2,}[0-9]*\b' | sort -u | tr '\n' ' ')
    [[ -z "$jargon" ]] \
      && c_pass "한 줄 요약에 미정의 약어 없음" \
      || c_warn "한 줄 요약에 약어 있음: ${jargon}— §3.8은 여기에 미정의 용어를 금지합니다"
  else
    c_warn "'> **한 줄 요약**' 줄을 찾지 못함"
  fi

  # ── D. 시각자료·표 §3.2 ────────────────────────────────────
  printf '\n  \033[1mD. 시각자료·표 (§3.2)\033[0m\n'
  local mermaid bullets tables
  mermaid=$(grep -c '^```mermaid' "$f")
  [[ "$mermaid" -ge 2 ]] \
    && c_pass "Mermaid ${mermaid}개 (모듈 지도는 §0·「한 장 정리」 2회)" \
    || c_fail "Mermaid ${mermaid}개 — 모듈 지도를 §0과 「한 장 정리」에 각각 두세요"

  # 표 블록 수 · 불릿 블록 수 (연속 행 묶음을 1블록으로)
  tables=$(awk '/^[[:space:]]*\|/{if(!in_t){n++;in_t=1}} !/^[[:space:]]*\|/{in_t=0} END{print n+0}' "$f")
  bullets=$(awk '/^[[:space:]]*[-*] /{if(!in_b){n++;in_b=1}} !/^[[:space:]]*[-*] /{in_b=0} END{print n+0}' "$f")
  if [[ "$tables" -le "$bullets" ]]; then
    c_pass "표 ${tables}블록 ≤ 불릿 ${bullets}블록"
  else
    c_fail "표 ${tables}블록 > 불릿 ${bullets}블록 — 나열·순차·근거는 불릿, 논증은 산문으로"
  fi

  # 표 행 수 상한 10
  local over_rows
  over_rows=$(awk '
    /^[[:space:]]*\|/ { rows++; if (rows > mx) mx = rows; next }
    { if (rows > 10) over++; rows = 0 }
    END { if (rows > 10) over++; print over+0 }
  ' "$f")
  [[ "$over_rows" -eq 0 ]] \
    && c_pass "모든 표가 10행 이하" \
    || c_fail "10행 초과 표 ${over_rows}개 — 축을 쪼개 나누거나 불릿으로"

  # 셀 120자 상한
  local over_cells
  over_cells=$(awk -F'|' '
    /^[[:space:]]*\|/ {
      for (i = 2; i < NF; i++) {
        cell = $i
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", cell)
        if (length(cell) > 120) n++
      }
    }
    END { print n+0 }
  ' "$f")
  [[ "$over_cells" -eq 0 ]] \
    && c_pass "모든 셀이 120자 이하" \
    || c_fail "120자 초과 셀 ${over_cells}개 — 표에 들어갈 내용이 아닙니다 (산문 + 소제목으로)"

  # ── E. 이해 사다리 §3.9 ────────────────────────────────────
  printf '\n  \033[1mE. 이해 사다리 (§3.9)\033[0m\n'
  local dodge
  dodge=$(grep -nE '(설명이 필요 없|이미 아는 것으로 전제|아는 것으로 보고 (넘어|건너)|배경이면 이 (식|부분)은|생략합니다\.$)' "$f" | head -5)
  if [[ -z "$dodge" ]]; then
    c_pass "설명 회피 문장 없음"
  else
    c_fail "설명 회피 문장 발견 — 생략할 거면 대신 링크를 주세요:"
    printf '        %s\n' "$dodge" | head -5
  fi

  # 선수 지식이 '없음'인데 제어/ML 개념을 앵커로 쓰는지
  if grep -qE '선수 지식.*없음' "$f" && grep -qE '(cascade|캐스케이드|MPC|상태공간|제어 배경)' "$f"; then
    c_fail "「선수 지식: 없음」인데 제어 개념을 가정하고 있습니다 — 무엇을 가정하는지 명시하세요"
  else
    c_pass "선수 지식 표기와 실제 가정이 모순되지 않음"
  fi

  # ── F. 분량 §4 ─────────────────────────────────────────────
  printf '\n  \033[1mF. 분량 (§4)\033[0m\n'
  local lo hi
  case "$tier" in
    # 2026-08-10: 세 밴드 각 +3,000. 고정 바닥(퀴즈·출처·팀 질문·실습 안내 +
    # §3.7 구조물)을 6,350자로 추정했으나 실측 셋이 8,348~8,711이었다. 근거는 §9.15.
    A) lo=15000; hi=21000 ;;
    B) lo=13000; hi=18000 ;;
    C) lo=12000; hi=17000 ;;
    *) lo=0; hi=0 ;;
  esac
  local est
  est=$(awk -v p="$prose" 'BEGIN{printf "%.0f", 19.5 + 0.001186*p}')
  if [[ "$hi" -eq 0 ]]; then
    c_warn "tier를 읽지 못해 분량 판정을 건너뜁니다 (읽은 값: '${tier:-없음}')"
    c_info "본문 산문 ${prose}자 · 총 ${total}자 · 독해 약 ${est}분"
  else
    if   [[ "$prose" -lt "$lo" ]]; then c_warn "본문 산문 ${prose}자 — Tier ${tier} 밴드(${lo}~${hi}) 미달, 독해 약 ${est}분"
    elif [[ "$prose" -gt "$hi" ]]; then c_fail "본문 산문 ${prose}자 — Tier ${tier} 밴드(${lo}~${hi}) 초과, 독해 약 ${est}분. 자르지 말고 deep-dive.md로 이관"
    else c_pass "본문 산문 ${prose}자 (Tier ${tier} 밴드 ${lo}~${hi}), 독해 약 ${est}분"
    fi
  fi
  c_info "총 ${total}자 · 보호 구간 ${protect}% (35% 이상이면 윤문은 산문 라인 판정 — course-plan §1)"

  # ── G. 문장부호 §3.10 ──────────────────────────────────────
  printf '\n  \033[1mG. 문장부호 (§3.10)\033[0m\n'
  # 산문 라인만 대상: frontmatter · 코드펜스 내부 · 표 행(| 시작) 제외
  local dash_n dot_n dash_hits dot_hits
  read -r dash_n dot_n < <(python3 - "$f" <<'PY'
import re, sys
s = open(sys.argv[1], encoding='utf-8').read()
s = re.sub(r'^---\n.*?\n---\n', '', s, flags=re.S)
s = re.sub(r'```.*?```', '', s, flags=re.S)
lines = [l for l in s.split('\n') if not l.strip().startswith('|')]
prose = '\n'.join(lines)
print(prose.count('—'), prose.count('·'))
PY
)
  if [[ "$dash_n" -lt 3 ]]; then
    c_pass "산문 줄표(—) ${dash_n}회"
  else
    c_fail "산문 줄표(—) ${dash_n}회. 쉼표, 괄호, 문장 분리로 바꾸세요 (헤딩은 한 구절로)"
    dash_hits=$(grep -n '—' "$f" | grep -v '^\s*[0-9]*:\s*|' | head -3)
    [[ -n "$dash_hits" ]] && printf '        %s\n' "$dash_hits"
  fi
  if [[ "$dot_n" -lt 3 ]]; then
    c_pass "산문 가운뎃점(·) ${dot_n}회"
  else
    c_fail "산문 가운뎃점(·) ${dot_n}회. 쉼표나 '와/과'로 바꾸세요"
    dot_hits=$(grep -n '·' "$f" | grep -v '^\s*[0-9]*:\s*|' | head -3)
    [[ -n "$dot_hits" ]] && printf '        %s\n' "$dot_hits"
  fi

  # deep-dive 링크가 있으면 파일이 실제로 있는지
  if grep -q 'deep-dive\.md' "$f"; then
    [[ -f "$(dirname "$f")/deep-dive.md" ]] \
      && c_pass "deep-dive.md 링크와 파일이 모두 존재" \
      || c_fail "deep-dive.md를 링크했으나 파일이 없습니다"
  fi
}

# ── 실행 ──────────────────────────────────────────────────────
if [[ $# -eq 0 ]]; then
  echo "사용법: bash scripts/lint-lesson.sh <lesson.md> [...]   |   --all" >&2
  exit 2
fi

if [[ "${1:-}" == "--all" ]]; then
  shopt -s nullglob
  for f in course/w*/*/lesson.md; do lint_one "$f"; done
else
  for f in "$@"; do lint_one "$f"; done
fi

printf '\n\033[1m─────────────────────────────────────────\033[0m\n'
printf '  \033[32mPASS %d\033[0m  ·  \033[31mFAIL %d\033[0m  ·  \033[33mWARN %d\033[0m\n\n' "$PASS" "$FAIL" "$WARN"
[[ "$FAIL" -eq 0 ]]
