#!/usr/bin/env bash
# lint-lesson.sh — lesson.md 집필 규약 검사
#
# SSOT: docs/course-plan.md §2.1 3층 문서 · §3.1 frontmatter · §3.2 시각자료
#       §3.7 문서 구조 · §3.8 용어 도입 · §3.9 학습자 전제와 비유
#       §3.10 문장부호 · §3.11 독자를 데려가는 장치 · §4 분량
# 근거: docs/course-plan.md §9.11 (2026-08-09 규약 개정)
#       docs/course-plan.md §9.19 (2026-08-25 규약 2차 개정)
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

# ── H2 대절 블록 분할 ────────────────────────────────────────────
# 각 H2 대절을 '시작줄<TAB>끝줄<TAB>제목' 한 줄로 출력합니다.
# 시작줄은 H2 헤딩 줄(1-indexed), 끝줄은 다음 H2 직전 줄 또는 파일 끝입니다.
#
# 코드펜스(```) 안의 '## '는 헤딩으로 세지 않습니다. 골격을 코드블록에 담은
# 문서(docs/course-plan.md §3.7)가 있고 lesson.md도 앞으로 그럴 수 있습니다.
#
# 이 헬퍼는 B군·C군·H군이 공유합니다.
h2_blocks() {
  python3 - "$1" <<'PY'
import sys
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
heads, fence = [], False
for i, l in enumerate(lines, 1):
    if l.lstrip().startswith('```'):
        fence = not fence
        continue
    if not fence and l.startswith('## '):
        heads.append((i, l[3:].strip()))
for k, (start, title) in enumerate(heads):
    end = heads[k + 1][0] - 1 if k + 1 < len(heads) else len(lines)
    print(f"{start}\t{end}\t{title}")
PY
}

# ── 「| 용어 |」 표 헤더 스캔 ──────────────────────────────────────
# `| 용어 |`로 시작하는 표 머리행을 '줄번호<TAB>열수'로 출력합니다.
# frontmatter와 코드펜스 안은 제외합니다.
#
# 2026-08-25 개정(§3.8)으로 **절 시작 용어표가 폐지**되고 절 끝 되짚기 표
# `| 용어 | 이 절에서 나온 뜻 |` 2열로 바뀌었습니다. 열 수가 이 둘을 가릅니다.
# 3열 이상이면 폐지된 `| 용어 | 한 줄 정의 | 제어·LLM 비유 |` 형태입니다.
term_table_rows() {
  python3 - "$1" <<'PY'
import re, sys
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
fence = fm = False
for i, l in enumerate(lines, 1):
    s = l.strip()
    if i == 1 and s == '---':
        fm = True; continue
    if fm:
        if s == '---': fm = False
        continue
    if s.startswith('```'):
        fence = not fence; continue
    if fence or not re.match(r'^\|\s*용어\s*\|', s):
        continue
    cells = re.split(r'(?<!\\)\|', s)
    if cells and not cells[0].strip():  cells = cells[1:]
    if cells and not cells[-1].strip(): cells = cells[:-1]
    print(f"{i}\t{len(cells)}")
PY
}

# ── 약어 첫 등장 · 첫 전개 위치 ──────────────────────────────────
# '약어<TAB>첫등장줄<TAB>전개줄<TAB>격차'를 격차 내림차순으로 출력합니다.
# 전개가 없으면 전개줄 0, 격차 -1입니다.
#
# §3.8 「역참조 금지(FAIL)」는 **첫 등장 자리에** 전개를 요구합니다. 파일 어딘가에
# 전개가 있기만 하면 되던 예전 검사는 W1-M1의 `FSQ`(첫 사용 L101, 전개 L566)를
# 통과시켰습니다. frontmatter의 `tags:`와 코드펜스 안 변수명은 오탐이라 제외합니다.
abbr_scan() {
  python3 - "$1" "$2" <<'PY'
import re, sys
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
skip = [True] * (len(lines) + 2)
fence = fm = False
for i, l in enumerate(lines, 1):
    s = l.strip()
    if i == 1 and s == '---':
        fm = True; continue
    if fm:
        if s == '---': fm = False
        continue
    if s.startswith('```'):
        fence = not fence; continue
    skip[i] = fence
rows = []
for ab in sys.argv[2].split():
    # 사용: 앞뒤가 영숫자가 아닌 자리 (한글 조사가 붙어도 잡힙니다)
    use = re.compile(r'(?<![A-Za-z0-9])' + re.escape(ab) + r'(?![A-Za-z0-9])')
    # 전개: 구 검사와 같은 판정식 '약어('
    exp = re.compile(re.escape(ab) + r'\s*\(')
    first = expand = 0
    for i, l in enumerate(lines, 1):
        if skip[i]:
            continue
        if not first and use.search(l):
            first = i
        if not expand and exp.search(l):
            expand = i
        if first and expand:
            break
    if first:
        # 전개가 첫 등장보다 앞서면(구 판정식의 경계 없는 매칭) 격차 0으로 봅니다
        gap = max(expand - first, 0) if expand else -1
        rows.append((ab, first, expand, gap))
rows.sort(key=lambda r: (-r[3], r[0]))
for r in rows:
    print("%s\t%d\t%d\t%d" % r)
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

  # 도입 산문 — 대절 제목 다음 첫 요소는 산문 (2026-08-25 신설, §3.7)
  # 대상은 번호 붙은 본문 대절만. 「흔한 오해」 「한 장 정리」 「출처」는 제외합니다
  local intro_n=0 intro_ok=0 intro_nf=0 intro_nw=0
  local intro_fail="" intro_warn="" intro_msg="" intro_line=""
  local hs he htitle hln hfirst hkind
  while IFS=$'\t' read -r hs he htitle; do
    [[ "$htitle" =~ ^[0-9]+\. ]] || continue
    intro_n=$((intro_n+1))
    hln=""; hfirst=""; hkind=""
    # 헤딩 다음 첫 비어있지 않은 줄 (줄 번호 + 내용, 들여쓰기는 벗겨짐)
    read -r hln hfirst < <(awk -v s="$hs" -v e="$he" 'NR>s && NR<=e && NF {print NR, $0; exit}' "$f")
    case "$hfirst" in
      '```mermaid'*) hkind="Mermaid" ;;
      '```'*)        hkind="코드펜스" ;;
      '|'*)          hkind="표" ;;
      '!['*)         hkind="이미지" ;;
    esac
    intro_msg="「${htitle}」 첫 요소가 ${hkind:-?}"
    [[ -n "$hln" ]] && intro_msg="${intro_msg} (L${hln})"
    if [[ -n "$hkind" ]]; then
      intro_nf=$((intro_nf+1))
      [[ "$intro_nf" -le 5 ]] && intro_fail+="L${hs}  ${intro_msg}"$'\n'
      continue
    fi
    case "$hfirst" in
      '- '*|'* '*) hkind="불릿" ;;
      '> '*)       hkind="인용" ;;
      '### '*)     hkind="소제목" ;;
      '')          hkind="내용 없음" ;;
    esac
    if [[ -n "$hkind" ]]; then
      intro_nw=$((intro_nw+1))
      intro_msg="「${htitle}」 첫 요소가 ${hkind}"
      [[ -n "$hln" ]] && intro_msg="${intro_msg} (L${hln})"
      [[ "$intro_nw" -le 5 ]] && intro_warn+="L${hs}  ${intro_msg}"$'\n'
    else
      intro_ok=$((intro_ok+1))
    fi
  done < <(h2_blocks "$f")

  if [[ "$intro_n" -gt 0 ]]; then
    if [[ "$intro_nf" -eq 0 && "$intro_nw" -eq 0 ]]; then
      c_pass "도입 산문 ${intro_ok}/${intro_n}개 대절 — 제목 다음 첫 요소가 모두 산문"
    fi
    if [[ "$intro_nf" -gt 0 ]]; then
      c_fail "도입 산문 없는 대절 ${intro_nf}/${intro_n}개 — §3.7: 대절 제목 다음 첫 요소는 산문 3~5문장입니다 (앞 절 요약 / 안 풀린 것 / 이 절이 답할 것)"
      while IFS= read -r intro_line; do
        [[ -n "$intro_line" ]] && printf '        %s\n' "$intro_line"
      done <<< "$intro_fail"
      [[ "$intro_nf" -gt 5 ]] && printf '        %s\n' "... 외 $((intro_nf-5))개"
    fi
    if [[ "$intro_nw" -gt 0 ]]; then
      c_warn "도입이 산문이 아닌 대절 ${intro_nw}/${intro_n}개 — §3.7: 불릿·인용·소제목보다 산문 3~5문장이 먼저입니다"
      while IFS= read -r intro_line; do
        [[ -n "$intro_line" ]] && printf '        %s\n' "$intro_line"
      done <<< "$intro_warn"
      [[ "$intro_nw" -gt 5 ]] && printf '        %s\n' "... 외 $((intro_nw-5))개"
    fi
  fi

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

  # 절 끝 되짚기 표 (2026-08-25 개정, §3.8)
  # 규격은 '| 용어 | 이 절에서 나온 뜻 |' **2열**, 위치는 그 절의 마지막(📌 직전)이고,
  # **그 절에서 처음 나온 용어가 있을 때만** 답니다. 절마다 강제하지 않습니다.
  # 폐지된 절 시작 용어표(3열)는 FAIL입니다. 비유 열이 비유를 문서 골격으로
  # 승격시킨 것이 2차 불만의 원인이었습니다(§9.19).
  local recap_n=0 wide_n=0 recap_lines="" wide_msg="" cline
  local tln tcols
  while IFS=$'\t' read -r tln tcols; do
    [[ -z "$tln" ]] && continue
    if [[ "$tcols" -ge 3 ]]; then
      wide_n=$((wide_n+1))
      [[ "$wide_n" -le 3 ]] && wide_msg+="L${tln}  ${tcols}열 용어표"$'\n'
    else
      recap_n=$((recap_n+1)); recap_lines="$recap_lines $tln"
    fi
  done < <(term_table_rows "$f")

  # 위치 판정 — 번호 붙은 본문 대절 안에서 되짚기 표가 📌보다 앞이어야 하고
  # 표와 📌 사이에 소제목이 끼면 절 끝이 아닙니다. 📌가 없는 절은 B군이 잡습니다
  local pos_nw=0 pos_msg="" pin sub
  while IFS=$'\t' read -r hs he htitle; do
    [[ "$htitle" =~ ^[0-9]+\. ]] || continue
    pin=$(awk -v s="$hs" -v e="$he" 'NR>=s && NR<=e && /📌 \*\*여기까지 정리\*\*/ {print NR; exit}' "$f")
    [[ -z "$pin" ]] && continue
    for tln in $recap_lines; do
      if [[ "$tln" -lt "$hs" || "$tln" -gt "$he" ]]; then continue; fi
      if [[ "$tln" -gt "$pin" ]]; then
        pos_nw=$((pos_nw+1))
        [[ "$pos_nw" -le 3 ]] && pos_msg+="L${tln}  「${htitle}」 되짚기 표가 📌(L${pin})보다 뒤"$'\n'
      else
        sub=$(awk -v s="$tln" -v e="$pin" 'NR>s && NR<e && /^### / {print NR; exit}' "$f")
        if [[ -n "$sub" ]]; then
          pos_nw=$((pos_nw+1))
          [[ "$pos_nw" -le 3 ]] && pos_msg+="L${tln}  「${htitle}」 되짚기 표와 📌 사이에 소제목(L${sub})"$'\n'
        fi
      fi
    done
  done < <(h2_blocks "$f")

  if [[ "$recap_n" -eq 0 ]]; then
    c_fail "절 끝 되짚기 표 없음 — §3.8: 절 끝에 '| 용어 | 이 절에서 나온 뜻 |' 2열로 그 절에서 처음 나온 용어를 정리하세요"
  elif [[ "$wide_n" -eq 0 && "$pos_nw" -eq 0 ]]; then
    c_pass "절 끝 되짚기 표 ${recap_n}개 (전부 2열, 절 끝)"
  fi
  if [[ "$wide_n" -gt 0 ]]; then
    c_fail "폐지된 절 시작 용어표 ${wide_n}개 — §3.8: '| 용어 | 한 줄 정의 | 제어·LLM 비유 |' 3열은 2026-08-25에 폐지됐습니다. 비유 열을 걷고 절 끝 2열 되짚기 표로 옮기세요"
    while IFS= read -r cline; do
      [[ -n "$cline" ]] && printf '        %s\n' "$cline"
    done <<< "$wide_msg"
    [[ "$wide_n" -gt 3 ]] && printf '        %s\n' "... 외 $((wide_n-3))개"
  fi
  if [[ "$pos_nw" -gt 0 ]]; then
    c_warn "절 끝이 아닌 되짚기 표 ${pos_nw}개 — §3.8: 되짚기 표는 그 절의 마지막, 📌 정리 박스 직전입니다"
    while IFS= read -r cline; do
      [[ -n "$cline" ]] && printf '        %s\n' "$cline"
    done <<< "$pos_msg"
    [[ "$pos_nw" -gt 3 ]] && printf '        %s\n' "... 외 $((pos_nw-3))개"
  fi

  # 약어 최초 전개 — §3.8 「역참조 금지(FAIL)」. **첫 등장 줄에** 전개가 있어야 합니다.
  # 파일 어딘가에 전개가 있으면 통과하던 구 검사는 W1-M1의 FSQ(첫 사용 L101,
  # 전개 L566, 465줄 격차)를 통과시켰습니다
  local abbr_list="MPC WBC VLA NFE RSSM DDS SLAM PPO IDM DoF BPE ELBO FSQ ACT DiT DCT IMU VQ-VAE"
  local ab_n=0 late_n=0 none_n=0 late_msg="" none_list=""
  local ab first expand gap
  while IFS=$'\t' read -r ab first expand gap; do
    [[ -z "$ab" ]] && continue
    ab_n=$((ab_n+1))
    if [[ "$gap" -lt 0 ]]; then
      none_n=$((none_n+1)); none_list="$none_list $ab"
    elif [[ "$gap" -gt 0 ]]; then
      late_n=$((late_n+1))
      [[ "$late_n" -le 5 ]] && late_msg+="${ab}(첫 등장 L${first} → 전개 L${expand}, ${gap}줄 격차)"$'\n'
    fi
  done < <(abbr_scan "$f" "$abbr_list")

  if [[ "$ab_n" -eq 0 ]]; then
    c_info "검사 목록의 약어가 본문에 없습니다"
  elif [[ "$late_n" -eq 0 && "$none_n" -eq 0 ]]; then
    c_pass "등장 약어 ${ab_n}개가 모두 첫 등장 줄에서 전개됨"
  fi
  if [[ "$late_n" -gt 0 ]]; then
    c_fail "역참조 약어 ${late_n}/${ab_n}개 — §3.8: 정의보다 먼저 쓰지 않습니다. 전개를 첫 등장 자리로 옮기거나 사용을 미루세요"
    while IFS= read -r cline; do
      [[ -n "$cline" ]] && printf '        %s\n' "$cline"
    done <<< "$late_msg"
    [[ "$late_n" -gt 5 ]] && printf '        %s\n' "... 외 $((late_n-5))개"
  fi
  if [[ "$none_n" -gt 0 ]]; then
    c_fail "전개 없이 쓰인 약어:${none_list}  → 최초 1회 'MPC(Model Predictive Control, 모델 예측 제어)'"
  fi

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
