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

# ── 그림 스캔 (캡션 · gantt · 이미지 참조) ────────────────────────
# 한 번의 훑기로 세 가지를 뽑습니다. 레코드는 종류로 시작합니다.
#   FIG<TAB>시작줄<TAB>끝줄<TAB>종류<TAB>캡션여부(1/0)
#   GANTT<TAB>줄번호
#   IMG<TAB>줄번호<TAB>경로            (로컬 상대경로만, http는 제외)
#
# 「읽는 법」 캡션(§3.2 그림 규약)은 그림이 끝난 **다음 3줄 안의 산문 줄**로 봅니다.
# 산문 줄은 비어있지 않고 표 행·헤딩·코드펜스·HTML 태그·이미지 단독 줄이 아닌 줄입니다.
#
# 판정 두 가지를 여기서 정해둡니다.
#  - **불릿과 인용은 캡션으로 인정합니다.** 「왼쪽 열은 X, 오른쪽은 Y」식 범례를 불릿으로
#    다는 것은 정당한 읽는 법이고, 인용 배지(⚠️ 미검증 등)도 대개 그림을 가리킵니다
#  - **단 📌 정리 박스는 인용이 이어지는 줄까지 통째로 인정하지 않습니다.** §3.7이 정의한
#    절 마무리 구조물이라 캡션 자리를 대신할 수 없습니다. 그림을 던지고 곧바로 절을 닫는 형태입니다
figure_scan() {
  python3 - "$1" <<'PY'
import re, sys
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')

PIN     = re.compile(r'^>\s*📌\s*\*\*여기까지 정리\*\*')
IMG     = re.compile(r'!\[[^\]]*\]\(\s*([^)\s]+)')
IMGONLY = re.compile(r'^(!\[[^\]]*\]\([^)]*\)\s*)+$')

# 📌 정리 박스는 머리줄뿐 아니라 인용이 이어지는 동안 전부 캡션이 아닙니다
pin = set()
i = 1
while i <= len(lines):
    if PIN.match(lines[i - 1].strip()):
        while i <= len(lines) and lines[i - 1].strip().startswith('>'):
            pin.add(i); i += 1
    else:
        i += 1

def is_prose(idx):
    s = lines[idx - 1].strip()
    if not s:                    return False   # 빈 줄
    if s.startswith('|'):        return False   # 표 행
    if s.startswith('#'):        return False   # 헤딩
    if s.startswith('```'):      return False   # 코드펜스
    if s.startswith('<'):        return False   # <details> 같은 HTML 구조물
    if IMGONLY.match(s):         return False   # 이미지만 있는 줄
    if idx in pin:               return False   # 절 마무리 📌 박스
    return True

def captioned(end):
    """end = 그림이 끝난 줄(1-indexed). 다음 3줄 안에 산문이 있으면 1"""
    for j in range(end + 1, min(end + 3, len(lines)) + 1):
        if is_prose(j):
            return 1
    return 0

out, fence, tag, start = [], False, '', 0
for i, l in enumerate(lines, 1):
    s = l.strip()
    if s.startswith('```'):
        if not fence:
            fence, tag, start = True, s[3:].strip().lower(), i
        else:
            fence = False
            if tag == 'mermaid':
                out.append("FIG\t%d\t%d\tMermaid\t%d" % (start, i, captioned(i)))
        continue
    if fence:
        if tag == 'mermaid' and re.match(r'^gantt\b', s):
            out.append("GANTT\t%d" % i)
        continue
    if not IMG.search(l):
        continue
    for m in IMG.finditer(l):
        p = m.group(1)
        if not re.match(r'^(https?:)?//|^data:', p):
            out.append("IMG\t%d\t%s" % (i, p))
    # 문장 안에 끼워 넣은 이미지는 그 줄에서 이미 설명되므로 캡션 검사 대상이 아닙니다
    rest = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', l).strip()
    if len(rest) < 2:
        out.append("FIG\t%d\t%d\t이미지\t%d" % (i, i, captioned(i)))
print('\n'.join(out))
PY
}

# ── ASCII 블록도 길이 ────────────────────────────────────────────
# '시작줄<TAB>끝줄<TAB>내용줄수'를 상한 초과분만 출력합니다.
#
# 대상은 **언어 태그가 없거나 text/plain인 코드펜스**입니다. python·bash·mermaid처럼
# 태그가 붙은 펜스는 코드나 다이어그램이지 ASCII 블록도가 아닙니다(§3.2).
# 근거는 W1-M1의 48줄 무호흡 블록도(L211-258)입니다.
ascii_blocks() {
  python3 - "$1" "$2" <<'PY'
import sys
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
limit = int(sys.argv[2])
fence, tag, start = False, '', 0
for i, l in enumerate(lines, 1):
    s = l.strip()
    if not s.startswith('```'):
        continue
    if not fence:
        fence, tag, start = True, s[3:].strip().lower(), i
    else:
        fence = False
        if tag in ('', 'text', 'plain') and i - start - 1 > limit:
            print("%d\t%d\t%d" % (start, i, i - start - 1))
PY
}

# ── 지정 구간의 첫 Mermaid 펜스 내용 ─────────────────────────────
# 줄별 앞뒤 공백을 벗기고 빈 줄을 버려 정규화합니다. Mermaid는 들여쓰기를
# 무시하므로 이렇게 같으면 렌더 결과가 완전히 같은 그림입니다.
# §3.2: 「한 장 정리」의 모듈 지도 재게시는 §0과 완전 동일 복제를 금지합니다.
first_mermaid_body() {
  python3 - "$1" "$2" "$3" <<'PY'
import sys
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
s, e = int(sys.argv[2]), min(int(sys.argv[3]), len(lines))
body, started = [], False
for i in range(s, e + 1):
    t = lines[i - 1].strip()
    if not started:
        if t == '```mermaid':
            started = True
        continue
    if t.startswith('```'):
        break
    if t:
        body.append(t)
print('\n'.join(body))
PY
}

# ── 설명 회피 · 「이미 안다」 문장 스캔 ───────────────────────────
# '줄번호<TAB>줄내용'을 출력합니다. **코드펜스 안은 제외**합니다. 골격 템플릿이나
# 예시 인용을 코드블록에 담은 문서에서 오탐이 나기 때문입니다(§3.9 예시 박스 자체가
# 코드펜스로 적혀 있습니다).
#
# §3.9 「수반 규칙」 두 항을 함께 봅니다.
#  - **"당신은 이미 안다" 류 문장 금지.** W1-M1 헤딩 「당신은 이미 이 구조를 알고 있다」(L85)가
#    대표 사례입니다. `이미 아실`, `익숙하실`, `아실 겁니다`도 같습니다
#  - **설명 회피 문장 금지.** 생략할 거면 대신 링크를 주세요
dodge_scan() {
  python3 - "$1" <<'PY'
import re, sys

# 설명 회피 (2026-08-09 이래의 기존 패턴, 하나도 빼지 않았습니다)
DODGE = [
    r'설명이 필요 없',
    r'이미 아는 것으로 전제',
    r'아는 것으로 보고 (넘어|건너)',
    r'배경이면 이 (식|부분)은',
    r'생략합니다\.$',
]
# "당신은 이미 안다" 류 (2026-08-25 §3.9 수반 규칙에서 추가)
ALREADY_KNOW = [
    r'당신은 이미',
    r'이미 .{0,10}알고 있',
    r'이미 아실',
    r'아실 겁니다',
    r'아실 것입니다',
    r'익숙하실',
    r'익숙하다면',
]
rx = re.compile('|'.join(DODGE + ALREADY_KNOW))

lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
fence = False
for i, l in enumerate(lines, 1):
    if l.strip().startswith('```'):
        fence = not fence
        continue
    if fence:
        continue
    if rx.search(l.rstrip()):
        print("%d\t%s" % (i, l.strip()))
PY
}

# ── 용어표 「비유」 열 스캔 ───────────────────────────────────────
# 표 **머리행**의 셀 중 「비유」를 포함하는 것이 있으면 '줄번호<TAB>머리행'을 출력합니다.
# 머리행 판정은 **다음 줄이 구분행**(`|---|---|`)인 것으로 합니다. 본문 데이터 셀에
# 「비유」라는 낱말이 들어간 것까지 잡으면 오탐이라, 머리행만 봅니다.
#
# §3.9: **용어표에 `제어·LLM 비유` 열을 두지 않습니다.** 이 열이 비유를 문서 아키텍처로
# 승격시켰고, W1-M1 실측으로 비유 지점 50개 중 36개가 이 열이었습니다.
# 2열 `| 용어 | 비유 |`도 같은 위반이라 열 수와 무관하게 잡습니다. C군의 「3열 이상
# 용어표」와 겹칠 수 있는데, 규약을 둘 어긴 것이니 그대로 둡니다.
analogy_col_rows() {
  python3 - "$1" <<'PY'
import re, sys
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
SEP = re.compile(r'^\|(\s*:?-+:?\s*\|)+$')
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
    if fence or not s.startswith('|'):
        continue
    nxt = lines[i].strip() if i < len(lines) else ''
    if not SEP.match(nxt):          # 다음 줄이 구분행이 아니면 머리행이 아닙니다
        continue
    cells = re.split(r'(?<!\\)\|', s)
    if cells and not cells[0].strip():  cells = cells[1:]
    if cells and not cells[-1].strip(): cells = cells[:-1]
    if any('비유' in c for c in cells):
        print("%d\t%s" % (i, s))
PY
}

# ── frontmatter prereq 항목 ──────────────────────────────────────
# prereq 값의 항목을 한 줄에 하나씩 출력합니다. 값이 비었으면 아무것도 출력하지 않습니다.
# 인라인 목록 `prereq: [W1-M1, W1-M5]`와 블록 목록(`prereq:` 다음 줄부터 `- W1-M1`)을
# 둘 다 읽습니다.
#
# §3.9: **`prereq`와 「선수 지식」이 단일 전제와 일치해야 합니다. 앞 모듈에서 배운 것만
# `prereq`에 올립니다.** 제어공학과 생성모델 내부 기제와 로봇 개념은 이제 **가르쳐야 할
# 대상**이지 선수 지식이 아닙니다.
prereq_items() {
  python3 - "$1" <<'PY'
import re, sys
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
if not lines or lines[0].strip() != '---':
    sys.exit(0)
fm = []
for l in lines[1:]:
    if l.strip() == '---':
        break
    fm.append(l)
items, i = [], 0
while i < len(fm):
    m = re.match(r'^prereq:\s*(.*)$', fm[i])
    if not m:
        i += 1; continue
    val = m.group(1).strip()
    if val.startswith('['):
        inner = val[1:val.rindex(']')] if ']' in val else val[1:]
        items = [x.strip() for x in inner.split(',')]
    elif val:
        items = [val]
    else:                            # 블록 목록
        j = i + 1
        while j < len(fm) and re.match(r'^\s*-\s+', fm[j]):
            items.append(re.sub(r'^\s*-\s+', '', fm[j]).strip())
            j += 1
    break
for it in items:
    it = it.strip().strip('\'"').strip()
    if it:
        print(it)
PY
}

# ── 초독 시간 추정 (§4 추정 기준 7종) ─────────────────────────────
# '총분 산문분 표분 그림분 수식분 퀴즈분 표행수 Mermaid수 ASCII수 SVG수 수식수'를
# 공백으로 이어 한 줄로 출력합니다. 인자는 파일과 prose_metrics()가 이미 계산한 산문 문자입니다.
#
# 2026-08-25에 `est_reading_min`이 **초독 시간**으로 재정의됐습니다(§4). 회귀식
# `분 = 19.5 + 0.001186 × 산문`은 산문만 세므로 표와 그림과 수식과 퀴즈를 읽는 시간이
# 빠져 있고, W1-M1에서 4.6배 불일치를 냈습니다(§9.19). 기준 7종은 §4가 정한 값입니다.
#   산문 550자/분 · 표 1행 25초 · Mermaid 1개 90초 · ASCII 블록 1개 120초
#   SVG 1장 45초 · display 수식 1개 90초 · 퀴즈 10문항 15분
#
# 산문 문자는 **prose_metrics()가 계산한 값을 그대로 받습니다.** 여기서 다시 세면
# §9.6의 정의와 갈라져 회귀식과 밴드가 서 있는 바닥이 무너집니다.
# 표 행은 구분행(`|---|---|`)을 빼고 셉니다. 구분행은 마크다운 문법이지 읽는 행이 아닙니다.
# 그림은 Mermaid와 ASCII 블록과 SVG 세 종을 합쳐 한 줄로 보고합니다.
# SVG는 원격 URL도 셉니다. 읽는 시간의 문제이지 파일 존재의 문제가 아니라서,
# 로컬 경로만 보는 figure_scan()과 여기서 갈립니다.
reading_estimate() {
  python3 - "$1" "$2" <<'PY'
import re, sys
path, prose = sys.argv[1], int(sys.argv[2])
src = open(path, encoding='utf-8').read()
lines = src.split('\n')

SEP = re.compile(r'^\|(\s*:?-+:?\s*\|)+$')
IMG = re.compile(r'!\[[^\]]*\]\(\s*([^)\s]+)')

rows = merm = asc = svg = 0
fence = fm = False
tag = ''
for i, l in enumerate(lines, 1):
    s = l.strip()
    if i == 1 and s == '---':
        fm = True; continue
    if fm:
        if s == '---': fm = False
        continue
    if s.startswith('```'):
        if not fence:
            fence, tag = True, s[3:].strip().lower()
        else:
            fence = False
            if tag == 'mermaid':                 merm += 1
            elif tag in ('', 'text', 'plain'):   asc += 1
        continue
    if fence:
        continue
    if s.startswith('|') and not SEP.match(s):
        rows += 1
    for m in IMG.finditer(l):
        p = m.group(1).split('#')[0].split('?')[0].lower()
        if p.endswith('.svg'):
            svg += 1

body = re.sub(r'```.*?```', '', src, flags=re.S)
disp = len(re.findall(r'\$\$.*?\$\$', body, re.S))
quiz = 15 if re.search(r'^## .*(셀프 체크|퀴즈)', src, re.M) else 0

m_prose = round(prose / 550.0)
m_table = round(rows * 25 / 60.0)
m_fig   = round((merm * 90 + asc * 120 + svg * 45) / 60.0)
m_math  = round(disp * 90 / 60.0)
# 합계는 반올림한 부분의 합입니다. 정보줄의 덧셈이 눈으로 맞아떨어져야 하기 때문입니다
tot = m_prose + m_table + m_fig + m_math + quiz
print("%d %d %d %d %d %d %d %d %d %d %d" % (
    tot, m_prose, m_table, m_fig, m_math, quiz, rows, merm, asc, svg, disp))
PY
}

# ── §0 「소요」의 이론 시간 ───────────────────────────────────────
# '줄번호<TAB>이론분'을 출력합니다. 「소요」는 있는데 「이론」 값을 읽지 못하면 이론분 -1,
# 「소요」 자체가 없으면 아무것도 출력하지 않습니다. 인자는 파일과 §0 구간(시작줄, 끝줄)입니다.
#
# **§0 안에서만, 그리고 `**소요**` 굵은 표기만** 찾습니다. 본문 다른 곳의 「소요 시간을
# 기록」(W1-M2 §0 완료 기준)과 「예상 소요: 약 4.4시간」(W2-M2 미검증 배지)이 오탐입니다.
#
# 표기 형식이 파일마다 다릅니다. 넷 다 받습니다.
#   `**소요**: 이론 3h / 실습 2h`      `**소요** 이론 2h, 실습 2~3h`
#   `**소요**는 이론 2h, 실습 2~3h입니다.`  `**소요** 이론 2~2.5h, 실습 1~2h`
# 「이론」 뒤의 값만 씁니다. h와 시간과 분을 받고 `2~2.5` 같은 범위는 중앙값으로 환산합니다.
sec0_duration() {
  python3 - "$1" "$2" "$3" <<'PY'
import re, sys
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
s, e = int(sys.argv[2]), min(int(sys.argv[3]), len(lines))

NUM  = r'\d+(?:\.\d+)?'
MARK = re.compile(r'\*\*\s*소요\s*\*\*')
VAL  = re.compile(r'이론[^0-9]{0,6}(' + NUM + r')\s*(?:[~∼\-]\s*(' + NUM + r'))?\s*(h|시간|분)')

found = 0
for i in range(s, e + 1):
    m = MARK.search(lines[i - 1])
    if not m:
        continue
    if not found:
        found = i
    v = VAL.search(lines[i - 1][m.end():])
    if not v:
        continue
    a = float(v.group(1))
    b = float(v.group(2)) if v.group(2) else a
    mid = (a + b) / 2.0
    print("%d\t%d" % (i, round(mid if v.group(3) == '분' else mid * 60)))
    sys.exit(0)
if found:
    print("%d\t-1" % found)
PY
}

# ── §0 목차 표 「읽는 시간」 열 합계 ──────────────────────────────
# '머리행줄번호<TAB>합계분<TAB>합산행수'를 출력합니다. 열이 없으면 아무것도 출력하지 않습니다.
# 인자는 파일과 §0 구간(시작줄, 끝줄)입니다.
#
# 대상은 `| 절 | 무엇을 | 끝나면 할 수 있는 것 | 읽는 시간 |` 머리행을 가진 표뿐입니다.
# §0에는 「선수 지식」 표도 있어서 아무 표나 세면 안 됩니다. 값은 `4분` 형태이고
# 범위(`3~4분`)는 중앙값으로 봅니다.
toc_reading_min() {
  python3 - "$1" "$2" "$3" <<'PY'
import re, sys
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
s, e = int(sys.argv[2]), min(int(sys.argv[3]), len(lines))

SEP = re.compile(r'^\|(\s*:?-+:?\s*\|)+$')
NUM = r'\d+(?:\.\d+)?'
VAL = re.compile(r'(' + NUM + r')\s*(?:[~∼\-]\s*(' + NUM + r'))?\s*분')

def cells(t):
    c = re.split(r'(?<!\\)\|', t)
    if c and not c[0].strip():  c = c[1:]
    if c and not c[-1].strip(): c = c[:-1]
    return [x.strip() for x in c]

for i in range(s, e + 1):
    t = lines[i - 1].strip()
    if not t.startswith('|'):
        continue
    head = cells(t)
    idx = next((k for k, c in enumerate(head) if '읽는 시간' in c), -1)
    if idx < 0:
        continue
    total, n, j = 0.0, 0, i + 1
    while j <= e and lines[j - 1].strip().startswith('|'):
        r = lines[j - 1].strip()
        if not SEP.match(r):
            rc = cells(r)
            if idx < len(rc):
                m = VAL.search(rc[idx])
                if m:
                    a = float(m.group(1))
                    b = float(m.group(2)) if m.group(2) else a
                    total += (a + b) / 2.0
                    n += 1
        j += 1
    print("%d\t%d\t%d" % (i, round(total), n))
    sys.exit(0)
PY
}

# ── 산문 라인의 줄표(—)·가운뎃점(·) 개수 ─────────────────────────
# '줄표수 가운뎃점수'를 한 줄로 출력합니다.
# 산문 라인 = frontmatter · 코드펜스 내부 · 표 행(| 시작)을 뺀 나머지입니다.
#
# G군(lesson.md)과 I군(eli5.md)이 공유합니다. §3.10은 두 층에 똑같이 적용되고,
# 규정이 하나면 판정도 하나여야 합니다.
# ⚠️ 2026-08-27 공용화 때 **G군의 판정과 출력이 한 글자도 달라지지 않음**을
#    7개 lesson의 G군 출력 diff로 확인했습니다. 여기를 고치면 두 군이 함께 움직입니다.
punct_counts() {
  python3 - "$1" <<'PY'
import re, sys
s = open(sys.argv[1], encoding='utf-8').read()
s = re.sub(r'^---\n.*?\n---\n', '', s, flags=re.S)
s = re.sub(r'```.*?```', '', s, flags=re.S)
lines = [l for l in s.split('\n') if not l.strip().startswith('|')]
prose = '\n'.join(lines)
print(prose.count('—'), prose.count('·'))
PY
}

# ── 워크드 예제 스캔 (H군) ───────────────────────────────────────
# 번호 붙은 본문 대절 중 **수식이나 계산이 나오는 절**만 골라
# '시작줄<TAB>제목<TAB>워크드예제줄(없으면 0)'을 출력합니다.
#
# §3.11: **수식이나 계산이 나오는 절마다 워크드 예제 최소 1개.** 실제 숫자를 대입해
# 손으로 따라가는 예제이고, 계산을 practice/*.py로 외주하지 않습니다. 실습 코드는
# **검산용**이지 최초 이해용이 아닙니다(W1-M1 실측 0개, §9.19).
#
# 판정 둘을 여기서 정해둡니다.
#  - **수식 절**은 `$$` display 블록이 있거나 인라인 수식이 **2개 이상**인 절입니다.
#    인라인 1개는 기호 하나를 스쳐 언급한 것일 수 있어 계산 절로 보지 않습니다
#  - **워크드 예제**는 lesson-template.md가 정한 `**숫자를 넣어봅니다.**`가 1순위이고,
#    문구가 조금 달라질 수 있어 굵은 리드 줄에 `숫자를 넣어`·`숫자를 대입`·`손으로 따라`가
#    들어간 것도 인정합니다
#
# 코드펜스 안은 수식으로도 예제로도 세지 않습니다. bash의 `$VAR`가 인라인 수식으로 잡힙니다.
worked_example_scan() {
  python3 - "$1" <<'PY'
import re, sys
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')

INL  = re.compile(r'(?<!\$)\$[^$\n]+\$(?!\$)')
LEAD = re.compile(r'\*\*[^*\n]*(?:숫자를 넣어|숫자를 대입|손으로 따라)[^*\n]*\*\*')

heads, skip, fence = [], [True] * (len(lines) + 2), False
for i, l in enumerate(lines, 1):
    if l.lstrip().startswith('```'):
        fence = not fence
        skip[i] = True
        continue
    skip[i] = fence
    if not fence and l.startswith('## '):
        heads.append((i, l[3:].strip()))

for k, (start, title) in enumerate(heads):
    end = heads[k + 1][0] - 1 if k + 1 < len(heads) else len(lines)
    if not re.match(r'^[0-9]+\.', title):
        continue
    disp = inl = ex = 0
    for i in range(start, end + 1):
        if skip[i]:
            continue
        l = lines[i - 1]
        disp += l.count('$$')
        inl += len(INL.findall(l))
        if not ex and LEAD.search(l):
            ex = i
    if disp or inl >= 2:
        print("%d\t%s\t%d" % (start, title, ex))
PY
}

# ── 퀴즈 3단 라벨 · 명령형 문항 스캔 (H군) ───────────────────────
# 퀴즈 대절 구간(시작줄, 끝줄) 안을 훑어 두 가지를 뽑습니다.
#   LABEL<TAB>라벨            찾은 3단 난이도 구간 라벨
#   IMP<TAB>줄번호<TAB>줄내용   명령형 종결이 있는 줄
#
# §3.11: 10문항을 전부 서술형 고난도로 내지 않습니다. 기억 확인 1~4, 적용 5~8, 종합 9~10이고
# 문항은 **평서형 질문**으로 씁니다(`계산하라`가 아니라 `계산해보세요`).
#
# 라벨은 `**기억 확인** (한 줄로 답이 나오는 것)` 형태를 기본으로 보되 소제목(`### 적용`)과
# 굵게 없는 줄머리도 받습니다. 낱말 경계를 요구하므로 문항 안의 `적용해보면`은 잡히지 않습니다.
# 명령형은 `하라`·`쓰라`·`시오`가 **종결 자리**에 왔을 때만 셉니다. `~하라고`, `~하라는`처럼
# 뒤에 글자가 이어지면 인용이나 연결이지 명령이 아닙니다.
quiz_scan() {
  python3 - "$1" "$2" "$3" <<'PY'
import re, sys
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
s, e = int(sys.argv[2]), min(int(sys.argv[3]), len(lines))

LABEL = re.compile(r'^(?:#{3,6}\s*)?\*{0,2}\s*(기억 확인|적용|종합)\b')
IMP   = re.compile(r'(?:하라|쓰라|시오)(?=[.!?)\]]|\s|$)')

found, fence = set(), False
for i in range(s, e + 1):
    t = lines[i - 1].strip()
    if t.startswith('```'):
        fence = not fence
        continue
    if fence:
        continue
    m = LABEL.match(t)
    if m:
        found.add(m.group(1))
    if IMP.search(t):
        print("IMP\t%d\t%s" % (i, t[:100]))
for w in ('기억 확인', '적용', '종합'):
    if w in found:
        print("LABEL\t%s" % w)
PY
}

# ── 합니다체가 아닌 종결 스캔 (H군) ──────────────────────────────
# '줄번호<TAB>종류<TAB>문맥'을 출력합니다. 종류는 `다` 또는 명사형 종결의 마지막 글자입니다.
#
# §3.11: **본문, `📌` 박스, 「한 장 정리」 표, 퀴즈를 전부 합니다체로 씁니다.** W1-M1은
# 본문이 합니다체인데 📌 박스 8개가 음슴체(`재발견이고 바뀌었다`)이고 퀴즈가 명령형이라
# 한 문서 안에서 세 목소리가 났습니다(§9.19).
#
# ⚠️ **한국어 종결 판정은 오탐이 나기 쉬워 범위를 좁게 잡았습니다.** 7개 lesson 실측으로
#    표본 30건과 명사형 전수 13건을 눈으로 확인해 오탐 0건이 되도록 다음을 정했습니다.
#  - **연결어미는 종결이 아닙니다.** `~다는 것`, `~다고`, `~다면`, `~다가`처럼 `다` 뒤에
#    글자가 이어지면 세지 않습니다. `다`가 문장부호나 줄 끝에 닿을 때만 종결로 봅니다
#  - **`니다`와 `시다`는 합니다체**라 제외합니다. `합니다`·`습니다`·`입니다`가 앞이고,
#    `~라고 합시다`·`~해봅시다`처럼 합니다체와 짝을 이루는 청유형이 뒤입니다.
#    §3.11의 워크드 예제 예시문 자체가 `~든다고 합시다.`로 끝납니다
#  - **헤딩은 대상이 아닙니다.** §3.11이 든 자리는 본문과 📌 박스와 표와 퀴즈이고, 절 제목은
#    한국어 문서에서 명사형이나 평서형으로 씁니다. 넣으면 7개 파일에서 22건이 전부 오탐이었습니다
#  - **표 셀은 대상입니다.** §3.11이 「한 장 정리」 표를 명시했습니다. 표 행은 셀 단위로 봅니다
#  - **명사형(`음`·`함`·`임`·`됨`·`옴`)은 흔한 명사를 걸러냅니다.** `묶음`, `프레임`, `없음`처럼
#    종결어미가 아닌 낱말이 표 셀에 자주 나옵니다. 12자 미만 짧은 셀은 문장이 아니라 라벨입니다.
#    거르지 않으면 이 갈래만 오탐률이 절반을 넘었습니다
#  - frontmatter, 코드펜스, `$$` 블록, 인라인 수식, 링크, 인용부호 안, 「출처」 절은 제외합니다
style_scan() {
  python3 - "$1" <<'PY'
import re, sys

INLINE_CODE = re.compile(r'`[^`\n]*`')
MDLINK      = re.compile(r'!?\[[^\]]*\]\([^)\n]*\)')
URL         = re.compile(r'https?://\S+')
HTML        = re.compile(r'<[^>\n]*>')
MATH        = re.compile(r'(?<!\$)\$[^$\n]+\$(?!\$)')
QUOTE       = re.compile(r'"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’'
                         r'|「[^」\n]*」|『[^』\n]*』'
                         r"|'[^'\n]*'")
SEP         = re.compile(r'^\|(\s*:?-+:?\s*\|)+$')

PLAIN = re.compile(r'(?<![니시])다(?=[.!?]|$)')
NOUN  = re.compile(r'([가-힣]+[음함임됨옴])(?=[.!?]|$)')

# 종결어미가 아니라 명사인 낱말 (실측 오탐 + 흔한 기술 용어)
STOP = set('''
묶음 모음 프레임 키프레임 처음 다음 마음 잡음 소음 게임 타임 이음 얼음 자음 모임
웃음 죽음 믿음 알림 구름 이름 저음 고음 화음 발음 녹음 방음 도움 싸움 물음 걸음 흐름
없음 있음 낮음 높음 짧음 적음 많음 작음 섞임 담음 쓰임
'''.split())

def clean(t):
    for rx in (INLINE_CODE, MDLINK, URL, HTML, MATH, QUOTE):
        t = rx.sub(' ', t)
    return re.sub(r'[*_~\s]+$', '', t.rstrip())

def segments(s):
    """표 행은 셀 단위로, 나머지는 줄 전체를 하나의 구간으로"""
    if s.startswith('|'):
        return [c.strip() for c in re.split(r'(?<!\\)\|', s) if c.strip()]
    return [s]

lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
fence = fm = disp = src = False
for i, l in enumerate(lines, 1):
    s = l.strip()
    if i == 1 and s == '---':
        fm = True; continue
    if fm:
        if s == '---': fm = False
        continue
    if s.startswith('```'):
        fence = not fence; continue
    if fence:
        continue
    if disp:
        if s.count('$$') % 2 == 1: disp = False
        continue
    if s.startswith('$$'):
        if s.count('$$') % 2 == 1: disp = True
        continue
    if s.startswith('#'):
        if s.startswith('## '): src = ('출처' in s)
        continue                       # 헤딩은 대상이 아닙니다
    if src or SEP.match(s):
        continue
    for seg in segments(s):
        c = clean(seg)
        for m in PLAIN.finditer(c):
            print("%d\t다\t%s" % (i, c[max(0, m.start() - 34):m.end() + 1]))
        for m in NOUN.finditer(c):
            w = m.group(1)
            if w in STOP or w[-2:] in STOP or len(c) < 12:
                continue
            print("%d\t%s\t%s" % (i, w[-1], c[max(0, m.start() - 34):m.end() + 1]))
PY
}

# ── eli5.md 단일 훑기 (I군 전용) ─────────────────────────────────
# 한 번의 훑기로 일곱 가지를 뽑습니다. 레코드는 종류로 시작합니다.
#   FIRST<TAB>줄번호<TAB>줄내용            첫 비어있지 않은 줄
#   MATH<TAB>줄번호<TAB>종류<TAB>줄내용     $$ 블록과 인라인 수식
#   ARXIV<TAB>줄번호<TAB>매치<TAB>줄내용    arXiv 문자열과 arXiv 번호(\d{4}\.\d{4,5})
#   TABLE<TAB>시작줄<TAB>끝줄<TAB>행수      표 블록 (구분행 |---|는 세지 않습니다)
#   BACKLINK<TAB>줄번호                    lesson.md와 § 절 참조가 함께 있는 줄
#   SECREF<TAB>줄번호<TAB>매치             「절 제목」이 뒤따르지 않는 §N
#   ABBR<TAB>약어                          서로 다른 영문 대문자 약어 (정렬)
#
# §2.1과 eli5-template.md가 정한 규격입니다. 수식 0, arXiv 인용 0, 영문 약어 최소,
# 표는 쓰지 않는 것이 기본, 단순화한 자리에는 「정확히는 lesson.md §N 「절 제목」에서」 링크.
#
# 판정을 여기서 정해둡니다.
#  - **코드펜스 안은 전부 제외합니다.** 수식도 arXiv 번호도 약어도 마찬가지입니다.
#    Mermaid 라벨과 코드 예시까지 잡으면 오탐이 규격보다 커집니다
#  - **약어는 한글 조사가 붙어도 잡습니다**(`MPC는`). abbr_scan()과 같은 경계식을 씁니다.
#    대신 링크 대상 `](...)`과 URL은 지웁니다. `../../img/DDPM-flow.svg`는 본문 약어가 아닙니다
#  - **표 행 수는 구분행을 뺍니다.** 구분행은 마크다운 문법이지 읽는 행이 아닙니다
#    (reading_estimate()와 같은 판정)
eli5_scan() {
  python3 - "$1" <<'PY'
import re, sys
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')

SEP  = re.compile(r'^\|(\s*:?-+:?\s*\|)+$')
INL  = re.compile(r'(?<!\$)\$[^$\n]+\$(?!\$)')
AXNM = re.compile(r'arxiv', re.I)
AXID = re.compile(r'(?<!\d)\d{4}\.\d{4,5}(?!\d)')
SECT = re.compile(r'§\s*\d+(?:\.\d+)*')
ABBR = re.compile(r'(?<![A-Za-z0-9])([A-Z]{2,}[0-9]*)(?![A-Za-z0-9])')
LINK = re.compile(r'\]\([^)]*\)|https?://\S+')

out = []
for i, l in enumerate(lines, 1):
    if l.strip():
        out.append("FIRST\t%d\t%s" % (i, l.strip()))
        break

abbrs = set()
fence = fm = disp = False
tbl_s = tbl_rows = 0
for i, l in enumerate(lines, 1):
    s = l.strip()
    if i == 1 and s == '---':
        fm = True; continue
    if fm:
        if s == '---': fm = False
        continue
    if s.startswith('```'):
        fence = not fence
        if tbl_s:
            out.append("TABLE\t%d\t%d\t%d" % (tbl_s, i - 1, tbl_rows)); tbl_s = tbl_rows = 0
        continue
    if fence:
        continue

    if s.startswith('|'):
        if not tbl_s:
            tbl_s, tbl_rows = i, 0
        if not SEP.match(s):
            tbl_rows += 1
    elif tbl_s:
        out.append("TABLE\t%d\t%d\t%d" % (tbl_s, i - 1, tbl_rows)); tbl_s = tbl_rows = 0

    if '$$' in l:
        # 여는 줄에서만 한 번 셉니다. 닫는 줄까지 세면 수식 하나가 두 곳으로 보입니다
        if l.count('$$') % 2 == 0 or not disp:
            out.append("MATH\t%d\t$$ 블록\t%s" % (i, s))
        if l.count('$$') % 2 == 1:
            disp = not disp
    elif not disp:
        m = INL.search(l)
        if m:
            out.append("MATH\t%d\t인라인 %s\t%s" % (i, m.group(0), s))

    m = AXNM.search(l) or AXID.search(l)
    if m:
        out.append("ARXIV\t%d\t%s\t%s" % (i, m.group(0), s))

    if 'lesson.md' in l and '§' in l:
        out.append("BACKLINK\t%d" % i)

    for m in SECT.finditer(l):
        rest = l[m.end():].lstrip()
        if rest.startswith('절'):
            rest = rest[1:].lstrip()
        if not rest.startswith('「'):
            out.append("SECREF\t%d\t%s" % (i, m.group(0)))

    for m in ABBR.finditer(LINK.sub(' ', l)):
        abbrs.add(m.group(1))

if tbl_s:
    out.append("TABLE\t%d\t%d\t%d" % (tbl_s, len(lines), tbl_rows))
for a in sorted(abbrs):
    out.append("ABBR\t%s" % a)
print('\n'.join(out))
PY
}

# ── 검사 상세를 최대 N줄까지 들여쓰기 출력 ───────────────────────
# 인자는 상한과 전체 건수입니다. 상한을 넘으면 '... 외 N개'를 덧붙입니다.
# I군이 씁니다. 기존 검사군은 각자의 출력 형식을 그대로 둡니다.
show_hits() {
  local n="$1" total="$2" line i=0
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    printf '        %s\n' "$line"
    i=$((i+1))
    [[ "$i" -ge "$n" ]] && break
  done
  [[ "$total" -gt "$n" ]] && printf '        %s\n' "... 외 $((total-n))개"
  return 0
}

lint_one() {
  local f="$1"
  printf '\n\033[1m▐ %s\033[0m\n' "$f"

  if [[ ! -f "$f" ]]; then c_fail "파일이 없습니다"; return; fi

  # 빈 파일 가드 — prose_metrics()가 0바이트 파일에서 ZeroDivisionError를 냅니다.
  # prose_metrics()는 §9.6 실측값·회귀식과의 비교 가능성 때문에 한 글자도 고치지 않으므로
  # 호출 쪽에서 막습니다. I군의 eli5.md 호출부에도 같은 가드가 있습니다
  if [[ ! -s "$f" ]]; then c_fail "파일이 비어 있습니다 (0바이트)"; return; fi

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

  # 초독 시간 (2026-08-25 신설, §4)
  # `est_reading_min`은 이제 **초독 시간**입니다. 산문에 표와 그림과 수식과 퀴즈 풀이를
  # 더한 값이고, 회귀식 값이 아닙니다. F군이 찍는 회귀식 값은 밴드 비교용 지표일 뿐입니다.
  # 검사는 셋입니다. 추정값과의 편차(WARN), §0 「소요」와의 정합(FAIL), 목차 합계(WARN).
  local rd_tot rd_pr rd_tb rd_fg rd_mt rd_qz rd_rows rd_mm rd_as rd_sv rd_ds
  read -r rd_tot rd_pr rd_tb rd_fg rd_mt rd_qz rd_rows rd_mm rd_as rd_sv rd_ds \
    < <(reading_estimate "$f" "$prose")
  : "${rd_tot:=0}" "${rd_pr:=0}" "${rd_tb:=0}" "${rd_fg:=0}" "${rd_mt:=0}" "${rd_qz:=0}"
  : "${rd_rows:=0}" "${rd_mm:=0}" "${rd_as:=0}" "${rd_sv:=0}" "${rd_ds:=0}"
  c_info "초독 시간 추정 ${rd_tot}분 (산문 ${rd_pr} + 표 ${rd_tb} + 그림 ${rd_fg} + 수식 ${rd_mt} + 퀴즈 ${rd_qz}) — §4 추정 기준 7종"
  c_info "추정 입력: 산문 ${prose}자 · 표 ${rd_rows}행 · Mermaid ${rd_mm} · ASCII ${rd_as} · SVG ${rd_sv} · display 수식 ${rd_ds}"

  local erm
  erm=$(awk 'NR<=20 && /^est_reading_min:/ {print $2; exit}' "$f" | tr -cd '0-9')

  if [[ -n "$erm" && "$rd_tot" -gt 0 ]]; then
    local rd_dev
    rd_dev=$(awk -v a="$erm" -v b="$rd_tot" 'BEGIN{printf "%.0f", 100*(a>b?a-b:b-a)/b}')
    if [[ "$rd_dev" -gt 30 ]]; then
      c_warn "est_reading_min ${erm}분이 초독 추정 ${rd_tot}분과 ${rd_dev}% 어긋납니다 — §4: 2026-08-25부터 이 값은 초독 시간(산문+표+그림+수식+퀴즈)입니다. 회귀식 값을 그대로 옮겨 적지 마세요"
    else
      c_pass "est_reading_min ${erm}분 ≈ 초독 추정 ${rd_tot}분 (편차 ${rd_dev}%)"
    fi
  fi

  # §0 「소요」·목차 합계와의 정합 — §0 구간을 먼저 잡습니다
  local s0_s="" s0_e=""
  local ahs ahe ahtitle
  while IFS=$'\t' read -r ahs ahe ahtitle; do
    case "$ahtitle" in
      0.*) [[ -z "$s0_s" ]] && { s0_s="$ahs"; s0_e="$ahe"; } ;;
    esac
  done < <(h2_blocks "$f")

  if [[ -n "$s0_s" ]]; then
    # 「소요」 대 est_reading_min (FAIL) — §4: 둘 중 하나가 학습자 기대를 배신합니다
    local dur_ln="" dur_min=""
    read -r dur_ln dur_min < <(sec0_duration "$f" "$s0_s" "$s0_e")
    if [[ -z "$dur_ln" ]]; then
      c_warn "§0에 「소요」 표기가 없습니다 — §4: est_reading_min과 §0 「소요」가 반드시 정합해야 합니다"
    elif [[ "$dur_min" -lt 0 ]]; then
      c_warn "§0 「소요」(L${dur_ln})에서 「이론」 시간을 읽지 못했습니다 — '**소요**: 이론 3h / 실습 2h' 형태로 쓰세요"
    elif [[ -z "$erm" ]]; then
      c_info "§0 「소요」 이론 ${dur_min}분 (L${dur_ln}) — est_reading_min이 없어 대조를 건너뜁니다"
    else
      local dur_tol dur_gap
      dur_tol=$(awk -v e="$erm" 'BEGIN{t=0.25*e; printf "%.0f", (t>15?t:15)}')
      dur_gap=$(( dur_min > erm ? dur_min - erm : erm - dur_min ))
      if [[ "$dur_gap" -gt "$dur_tol" ]]; then
        c_fail "§0 「소요」 이론 ${dur_min}분(L${dur_ln}) 대 est_reading_min ${erm}분 — 차이 ${dur_gap}분이 허용 ${dur_tol}분을 넘습니다. §4: 둘 중 하나가 학습자 기대를 배신합니다"
      else
        c_pass "§0 「소요」 이론 ${dur_min}분 ≈ est_reading_min ${erm}분 (차이 ${dur_gap}분 ≤ ${dur_tol}분)"
      fi
    fi

    # 목차 표 「읽는 시간」 합계 대 est_reading_min (WARN)
    local toc_ln="" toc_sum="" toc_n=""
    read -r toc_ln toc_sum toc_n < <(toc_reading_min "$f" "$s0_s" "$s0_e")
    if [[ -z "$toc_ln" ]]; then
      c_info "§0 목차 표에 「읽는 시간」 열이 없어 합계 검사를 건너뜁니다"
    elif [[ -z "$erm" || "$erm" -eq 0 ]]; then
      c_info "§0 목차 「읽는 시간」 합계 ${toc_sum}분 (${toc_n}행, L${toc_ln})"
    else
      local toc_dev
      toc_dev=$(awk -v a="$toc_sum" -v b="$erm" 'BEGIN{printf "%.0f", 100*(a>b?a-b:b-a)/b}')
      if [[ "$toc_dev" -gt 20 ]]; then
        c_warn "§0 목차 「읽는 시간」 합계 ${toc_sum}분(${toc_n}행, L${toc_ln})이 est_reading_min ${erm}분과 ${toc_dev}% 어긋납니다 — 절별 값과 총합 중 하나가 틀렸습니다"
      else
        c_pass "§0 목차 「읽는 시간」 합계 ${toc_sum}분 ≈ est_reading_min ${erm}분 (편차 ${toc_dev}%)"
      fi
    fi
  fi

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

  # 그림 「읽는 법」 캡션 (2026-08-25 신설, §3.2 그림 규약)
  # 근거: W1-M1의 41줄짜리 계보도(§6.1, L486-526)가 산문 해설 0자로 지나가고
  #       해설이 44줄 뒤에 나옵니다. 그림을 던지고 넘어가지 않게 하는 장치입니다
  local vis fig_n cap_bad cap_msg dline
  vis=$(figure_scan "$f")
  fig_n=$(printf '%s\n' "$vis" | awk -F'\t' '$1=="FIG"{n++} END{print n+0}')
  cap_bad=$(printf '%s\n' "$vis" | awk -F'\t' '$1=="FIG" && $5=="0"{n++} END{print n+0}')
  if [[ "$fig_n" -eq 0 ]]; then
    c_info "캡션을 검사할 그림(Mermaid·이미지)이 없습니다"
  elif [[ "$cap_bad" -eq 0 ]]; then
    c_pass "그림 ${fig_n}개 전부 다음 3줄 안에 캡션 있음"
  else
    c_warn "캡션 없는 그림 ${cap_bad}/${fig_n}개 — §3.2: 그림 바로 아래 1~3문장으로 어디를 보라고 지시하세요"
    cap_msg=$(printf '%s\n' "$vis" | awk -F'\t' '
      $1=="FIG" && $5=="0" {
        loc = ($2==$3) ? "L" $2 : "L" $2 "-" $3
        printf "%s  %s 뒤 3줄에 산문이 없습니다\n", loc, $4
      }' | head -5)
    while IFS= read -r dline; do
      [[ -n "$dline" ]] && printf '        %s\n' "$dline"
    done <<< "$cap_msg"
    [[ "$cap_bad" -gt 5 ]] && printf '        %s\n' "... 외 $((cap_bad-5))개"
  fi

  # ASCII 블록도 20줄 상한 (2026-08-25 신설, §3.2)
  local ascii_out ascii_n a1 a2 a3
  ascii_out=$(ascii_blocks "$f" 20)
  ascii_n=$(printf '%s\n' "$ascii_out" | awk 'NF{n++} END{print n+0}')
  if [[ "$ascii_n" -eq 0 ]]; then
    c_pass "ASCII 블록도가 모두 20줄 이하"
  else
    c_warn "20줄 초과 ASCII 블록도 ${ascii_n}개 — §3.2: 한 블록 20줄 상한. 조각내 산문 사이에 배치하거나 SVG로 옮기세요"
    while IFS=$'\t' read -r a1 a2 a3; do
      [[ -n "$a1" ]] && printf '        L%s-%s  내용 %s줄\n' "$a1" "$a2" "$a3"
    done < <(printf '%s\n' "$ascii_out" | head -5)
    [[ "$ascii_n" -gt 5 ]] && printf '        %s\n' "... 외 $((ascii_n-5))개"
  fi

  # Mermaid gantt 오용 (2026-08-25 신설, §3.2)
  # 근거: W1-M1의 청크 커버리지 다이어그램(L356-379)이 gantt이고 색 설명이 실제 렌더와 다릅니다
  local gantt_n gantt_at
  gantt_n=$(printf '%s\n' "$vis" | awk -F'\t' '$1=="GANTT"{n++} END{print n+0}')
  if [[ "$gantt_n" -eq 0 ]]; then
    c_pass "Mermaid gantt 오용 없음"
  else
    gantt_at=$(printf '%s\n' "$vis" | awk -F'\t' '$1=="GANTT"{printf " L%s", $2}')
    c_fail "Mermaid gantt ${gantt_n}개(${gantt_at} ) — §3.2: gantt는 프로젝트 일정용이라 색과 축이 의도대로 렌더되지 않습니다. 타이밍은 sequenceDiagram이나 SVG로 바꾸세요"
  fi

  # 모듈 지도 완전 동일 복제 (2026-08-25 신설, §3.2)
  local map0_s="" map0_e="" mapz_s="" mapz_e="" map0 mapz
  while IFS=$'\t' read -r hs he htitle; do
    case "$htitle" in
      0.*"이 모듈 지도"*) [[ -z "$map0_s" ]] && { map0_s="$hs"; map0_e="$he"; } ;;
    esac
    case "$htitle" in
      *"한 장 정리"*) [[ -z "$mapz_s" ]] && { mapz_s="$hs"; mapz_e="$he"; } ;;
    esac
  done < <(h2_blocks "$f")
  if [[ -n "$map0_s" && -n "$mapz_s" ]]; then
    map0=$(first_mermaid_body "$f" "$map0_s" "$map0_e")
    mapz=$(first_mermaid_body "$f" "$mapz_s" "$mapz_e")
    # 한쪽에 Mermaid가 없으면 위의 개수 검사가 이미 잡으므로 건너뜁니다
    if [[ -n "$map0" && -n "$mapz" ]]; then
      if [[ "$map0" == "$mapz" ]]; then
        c_warn "§0과 「한 장 정리」의 모듈 지도가 완전 동일 복제 — §3.2: 진행 표시나 절별 한 줄 요약 라벨을 더해 다른 정보를 주게 하세요"
      else
        c_pass "「한 장 정리」 모듈 지도가 §0과 다름"
      fi
    fi
  fi

  # 이미지 파일 존재 (§3.2: 파일은 img/{module-id}-{slug}.svg, 참조는 ../../img/)
  local img_n=0 img_bad=0 img_msg="" iln ipath icand
  while IFS=$'\t' read -r iln ipath; do
    [[ -z "$iln" ]] && continue
    img_n=$((img_n+1))
    if [[ "$ipath" == /* ]]; then icand=".${ipath}"; else icand="$(dirname "$f")/${ipath}"; fi
    if [[ ! -f "$icand" ]]; then
      img_bad=$((img_bad+1))
      [[ "$img_bad" -le 5 ]] && img_msg+="L${iln}  ${ipath}"$'\n'
    fi
  done < <(printf '%s\n' "$vis" | awk -F'\t' '$1=="IMG"{print $2"\t"$3}')
  if [[ "$img_n" -gt 0 ]]; then
    if [[ "$img_bad" -eq 0 ]]; then
      c_pass "이미지 참조 ${img_n}개가 모두 실재"
    else
      c_fail "없는 이미지 파일 ${img_bad}/${img_n}개 — §3.2: 파일은 img/{module-id}-{slug}.svg, 참조는 ../../img/"
      while IFS= read -r dline; do
        [[ -n "$dline" ]] && printf '        %s\n' "$dline"
      done <<< "$img_msg"
      [[ "$img_bad" -gt 5 ]] && printf '        %s\n' "... 외 $((img_bad-5))개"
    fi
  fi

  # ── E. 학습자 전제와 비유 §3.9 ─────────────────────────────
  # 「이해 사다리」는 2026-08-25에 폐기됐습니다(§9.19). 영역별 난이도 차등이
  # 2차 불만의 직접 원인이었고, §3.9는 「단일 전제」로 전면 재작성됐습니다.
  printf '\n  \033[1mE. 학습자 전제와 비유 (§3.9)\033[0m\n'

  # 설명 회피 · 「이미 안다」 문장 (2026-08-25 확장, §3.9 수반 규칙)
  local dodge_n=0 dodge_msg="" eline dln dtxt
  while IFS=$'\t' read -r dln dtxt; do
    [[ -z "$dln" ]] && continue
    dodge_n=$((dodge_n+1))
    [[ "$dodge_n" -le 5 ]] && dodge_msg+="L${dln}  ${dtxt}"$'\n'
  done < <(dodge_scan "$f")

  if [[ "$dodge_n" -eq 0 ]]; then
    c_pass "설명 회피·「이미 안다」 문장 없음"
  else
    c_fail "설명 회피·「이미 안다」 문장 ${dodge_n}건 — §3.9: 안다고 가정하는 것은 ML/DL 기본, LLM 기본, Agent·RAG 실무 셋뿐입니다. 경력을 근거로 설명을 건너뛸 수 없습니다. 생략할 거면 대신 링크를 주세요:"
    while IFS= read -r eline; do
      [[ -n "$eline" ]] && printf '        %s\n' "$eline"
    done <<< "$dodge_msg"
    [[ "$dodge_n" -gt 5 ]] && printf '        %s\n' "... 외 $((dodge_n-5))개"
  fi

  # 용어표 「비유」 열 (2026-08-25 신설, §3.9)
  local anlg_n=0 anlg_msg="" aln atxt
  while IFS=$'\t' read -r aln atxt; do
    [[ -z "$aln" ]] && continue
    anlg_n=$((anlg_n+1))
    [[ "$anlg_n" -le 3 ]] && anlg_msg+="L${aln}  ${atxt}"$'\n'
  done < <(analogy_col_rows "$f")

  if [[ "$anlg_n" -eq 0 ]]; then
    c_pass "용어표에 「비유」 열 없음"
  else
    c_fail "용어표 「비유」 열 ${anlg_n}개 — §3.9: 이 열이 비유를 문서 아키텍처로 승격시켰습니다(W1-M1 비유 지점 50개 중 36개). 열을 걷고 비유가 꼭 필요하면 절 끝 💡 <details> 박스 하나로 옮기세요:"
    while IFS= read -r eline; do
      [[ -n "$eline" ]] && printf '        %s\n' "$eline"
    done <<< "$anlg_msg"
    [[ "$anlg_n" -gt 3 ]] && printf '        %s\n' "... 외 $((anlg_n-3))개"
  fi

  # prereq 항목 (2026-08-25 신설, §3.9. 폐기된 「선수 지식 없음 + 제어 개념」 검사를 대체)
  # 새 §3.9에서 제어 개념은 **가르치는 대상**입니다. 선수 지식이 비어 있는데 본문에
  # 캐스케이드 제어가 나오는 것은 정상이고, 오히려 prereq에 올리는 쪽이 위반입니다.
  local pre_n=0 pre_bad=0 pre_list="" pitem
  while IFS= read -r pitem; do
    [[ -z "$pitem" ]] && continue
    pre_n=$((pre_n+1))
    if [[ ! "$pitem" =~ ^W[0-9]+-M[0-9]+$ ]]; then
      pre_bad=$((pre_bad+1)); pre_list="$pre_list ${pitem}"
    fi
  done < <(prereq_items "$f")

  if [[ "$pre_bad" -eq 0 ]]; then
    c_pass "prereq ${pre_n}개가 모두 모듈 ID"
  else
    c_fail "모듈 ID가 아닌 prereq ${pre_bad}/${pre_n}개:${pre_list}  → §3.9: prereq에는 앞 모듈에서 배운 것만 올립니다. 제어나 생성모델 내부 기제나 로봇 개념을 선수 지식으로 요구할 수 없습니다"
  fi

  # 비유 박스 절당 최대 1개 (2026-08-25 신설, §3.9)
  # 💡 <summary>를 단 <details>를 비유 박스로 봅니다. 여러 개가 필요하면
  # 그 절의 본문 설명이 부족하다는 뜻입니다.
  local box_all=0 box_nw=0 box_msg="" nbox
  box_all=$(awk '
    /^[[:space:]]*```/ { fence = !fence; next }
    fence { next }
    /<summary>/ && /💡/ { n++ }
    END { print n+0 }' "$f")
  while IFS=$'\t' read -r hs he htitle; do
    [[ "$htitle" =~ ^[0-9]+\. ]] || continue
    nbox=$(awk -v s="$hs" -v e="$he" '
      /^[[:space:]]*```/ { fence = !fence; next }
      fence { next }
      NR>=s && NR<=e && /<summary>/ && /💡/ { n++ }
      END { print n+0 }' "$f")
    if [[ "$nbox" -ge 2 ]]; then
      box_nw=$((box_nw+1))
      [[ "$box_nw" -le 3 ]] && box_msg+="L${hs}  「${htitle}」 비유 박스 ${nbox}개"$'\n'
    fi
  done < <(h2_blocks "$f")

  if [[ "$box_all" -eq 0 ]]; then
    c_info "💡 비유 박스가 없습니다 (§3.9: 본문은 비유 없이 완결되는 것이 기본)"
  elif [[ "$box_nw" -eq 0 ]]; then
    c_pass "비유 박스 ${box_all}개, 절당 1개 이하"
  else
    c_warn "비유 박스가 2개 이상인 대절 ${box_nw}개 — §3.9: 비유 박스는 절당 최대 1개입니다. 여러 개가 필요하면 그 절의 본문 설명이 부족하다는 뜻입니다"
    while IFS= read -r eline; do
      [[ -n "$eline" ]] && printf '        %s\n' "$eline"
    done <<< "$box_msg"
    [[ "$box_nw" -gt 3 ]] && printf '        %s\n' "... 외 $((box_nw-3))개"
  fi

  # ── F. 분량 §4 ─────────────────────────────────────────────
  printf '\n  \033[1mF. 분량 (§4)\033[0m\n'
  local lo hi
  # 📍 **밴드의 정본은 이 스크립트와 docs/course-plan.md §4 두 곳뿐입니다.**
  #    집필 도구(SKILL.md, lesson-template.md, authoring-checklist.md)는 숫자를 복사하지 않고
  #    여기를 가리킵니다. 숫자를 여러 곳에 두었다가 3중 불일치가 **두 번 재발**했습니다.
  #    §9.11이 한 번 해소했는데 2026-08-10 상향이 스크립트에만 반영돼 되살아났습니다(§9.19).
  #    밴드를 고칠 일이 생기면 여기와 §4 표를 **같은 커밋에서** 함께 고치세요.
  #
  # ⚠️ **이 밴드는 잠정입니다.** 2026-08-25 개정으로 문서가 3층(eli5.md, lesson.md,
  #    deep-dive.md)으로 갈리고(§2.1) 설명이 친절해지면서 산문 구성이 바뀝니다.
  #    §9.11이 W1-M1 파일럿으로 밴드를 교정한 것과 같은 절차로,
  #    **이번에도 W1-M1 시범 결과로 1회 교정**합니다(§4 2026-08-25 박스).
  case "$tier" in
    # 2026-08-10: 세 밴드 각 +3,000. 고정 바닥(퀴즈·출처·팀 질문·실습 안내 +
    # §3.7 구조물)을 6,350자로 추정했으나 실측 셋이 8,348~8,711이었다. 근거는 §9.15.
    A) lo=15000; hi=21000 ;;
    B) lo=13000; hi=18000 ;;
    C) lo=12000; hi=17000 ;;
    *) lo=0; hi=0 ;;
  esac
  # 회귀식 값은 **밴드 비교용 지표**이지 학습자가 볼 시간이 아닙니다. 2026-08-25에
  # `est_reading_min`이 초독 시간으로 재정의됐고(§4), 이 식은 **산문만** 세므로 표와
  # 그림과 수식과 퀴즈 풀이가 빠져 있습니다. 초독 시간은 A군이 따로 추정해 찍습니다.
  # 식 자체(19.5 + 0.001186 × 산문)는 그대로 둡니다. §9.7이 적합한 값이고
  # §9.2·§9.4·§9.6의 과거 실측값과의 비교 가능성이 이 식 위에 서 있습니다.
  local reg_min reg_note
  reg_min=$(awk -v p="$prose" 'BEGIN{printf "%.0f", 19.5 + 0.001186*p}')
  reg_note="회귀식 ${reg_min}분(산문만 세는 밴드 비교용 지표. 초독 시간은 frontmatter est_reading_min)"
  if [[ "$hi" -eq 0 ]]; then
    c_warn "tier를 읽지 못해 분량 판정을 건너뜁니다 (읽은 값: '${tier:-없음}')"
    c_info "본문 산문 ${prose}자 · 총 ${total}자 · ${reg_note}"
  else
    if   [[ "$prose" -lt "$lo" ]]; then c_warn "본문 산문 ${prose}자 — Tier ${tier} 밴드(${lo}~${hi}) 미달, ${reg_note}"
    elif [[ "$prose" -gt "$hi" ]]; then c_fail "본문 산문 ${prose}자 — Tier ${tier} 밴드(${lo}~${hi}) 초과, ${reg_note}. 자르지 말고 deep-dive.md로 이관"
    else c_pass "본문 산문 ${prose}자 (Tier ${tier} 밴드 ${lo}~${hi}), ${reg_note}"
    fi
  fi
  c_info "총 ${total}자 · 보호 구간 ${protect}% (35% 이상이면 윤문은 산문 라인 판정 — course-plan §1)"

  # ── G. 문장부호 §3.10 ──────────────────────────────────────
  printf '\n  \033[1mG. 문장부호 (§3.10)\033[0m\n'
  # 산문 라인만 대상: frontmatter · 코드펜스 내부 · 표 행(| 시작) 제외
  local dash_n dot_n dash_hits dot_hits
  read -r dash_n dot_n < <(punct_counts "$f")
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

  # ── H. 독자를 데려가는 장치 §3.11 ──────────────────────────
  # §3.11은 2026-08-25에 신설됐는데 집필 도구 3종(pai-agent.md, lesson-template.md,
  # authoring-checklist.md C6군)에만 반영되고 lint에는 빠져 있었습니다. 워크드 예제 0개와
  # 음슴체 혼용이 2차 피드백의 진단 항목이라(§9.19) 육안 검사에만 맡기지 않습니다.
  printf '\n  \033[1mH. 독자를 데려가는 장치 (§3.11)\033[0m\n'

  # 워크드 예제 — 수식이나 계산이 나오는 절마다 최소 1개 (FAIL)
  local we_n=0 we_bad=0 we_msg="" hline wln wtitle wex
  while IFS=$'\t' read -r wln wtitle wex; do
    [[ -z "$wln" ]] && continue
    we_n=$((we_n+1))
    if [[ "$wex" -eq 0 ]]; then
      we_bad=$((we_bad+1))
      [[ "$we_bad" -le 5 ]] && we_msg+="L${wln}  「${wtitle}」"$'\n'
    fi
  done < <(worked_example_scan "$f")

  if [[ "$we_n" -eq 0 ]]; then
    c_info "수식이나 계산이 나오는 대절이 없어 워크드 예제 검사를 건너뜁니다"
  elif [[ "$we_bad" -eq 0 ]]; then
    c_pass "워크드 예제가 수식 절 ${we_n}개에 모두 있음"
  else
    c_fail "워크드 예제 없는 수식 절 ${we_bad}/${we_n}개 — §3.11: 실제 숫자를 대입해 손으로 따라가는 예제를 답니다. 계산을 practice/*.py로 외주하지 마세요(실습 코드는 검산용입니다). 리드는 '**숫자를 넣어봅니다.**'"
    while IFS= read -r hline; do
      [[ -n "$hline" ]] && printf '        %s\n' "$hline"
    done <<< "$we_msg"
    [[ "$we_bad" -gt 5 ]] && printf '        %s\n' "... 외 $((we_bad-5))개"
  fi

  # 「여기서 많이 헷갈립니다」 박스 (WARN)
  # 절마다 강제하지 않습니다. 오해가 없는 절도 있어서, 문서 전체에 하나도 없을 때만 봅니다
  local cfz_n
  cfz_n=$(awk '
    /^[[:space:]]*```/ { fence = !fence; next }
    fence { next }
    /여기서 많이 헷갈립니다/ { n++ }
    END { print n+0 }' "$f")
  if [[ "$cfz_n" -gt 0 ]]; then
    c_pass "「여기서 많이 헷갈립니다」 박스 ${cfz_n}개"
  else
    c_warn "「여기서 많이 헷갈립니다」 박스가 하나도 없습니다 — §3.11: 「흔한 오해」는 문서 끝이라 정작 헷갈리는 자리에서 도움이 되지 않습니다. 오해가 생기기 쉬운 그 자리에 '> ⚠️ **여기서 많이 헷갈립니다**'를 다세요"
  fi

  # 퀴즈 대절과 「한 장 정리」 대절 구간을 한 번에 잡습니다
  local qz_s="" qz_e="" wrap_s="" wrap_e=""
  while IFS=$'\t' read -r hs he htitle; do
    case "$htitle" in
      *"셀프 체크"*|*"퀴즈"*) [[ -z "$qz_s" ]] && { qz_s="$hs"; qz_e="$he"; } ;;
    esac
    case "$htitle" in
      *"한 장 정리"*) [[ -z "$wrap_s" ]] && { wrap_s="$hs"; wrap_e="$he"; } ;;
    esac
  done < <(h2_blocks "$f")

  # 퀴즈 3단 난이도 (WARN) · 명령형 문항 (WARN)
  # 퀴즈 절 자체가 없으면 B군이 이미 FAIL을 내므로 여기서는 건너뜁니다
  if [[ -z "$qz_s" ]]; then
    c_info "셀프 체크 퀴즈 절이 없어 3단 난이도와 문항 형식 검사를 건너뜁니다 (B군이 이미 FAIL)"
  else
    local qz_out qz_missing="" qz_imp_n qz_imp_msg qlbl
    qz_out=$(quiz_scan "$f" "$qz_s" "$qz_e")
    for qlbl in "기억 확인" "적용" "종합"; do
      printf '%s\n' "$qz_out" \
        | awk -F'\t' -v w="$qlbl" '$1=="LABEL" && $2==w {f=1} END{exit !f}' \
        || qz_missing="${qz_missing} 「${qlbl}」"
    done
    if [[ -z "$qz_missing" ]]; then
      c_pass "퀴즈 3단 난이도 라벨 완비 (기억 확인, 적용, 종합)"
    else
      c_warn "퀴즈 난이도 구간 라벨 누락:${qz_missing} — §3.11: 10문항을 전부 서술형 고난도로 내지 않습니다. 기억 확인 1~4, 적용 5~8, 종합 9~10으로 나누세요"
    fi

    qz_imp_msg=$(printf '%s\n' "$qz_out" | awk -F'\t' '$1=="IMP"{printf "L%s  %s\n", $2, $3}')
    qz_imp_n=$(printf '%s\n' "$qz_imp_msg" | awk 'NF{n++} END{print n+0}')
    if [[ "$qz_imp_n" -eq 0 ]]; then
      c_pass "퀴즈 문항이 모두 평서형"
    else
      c_warn "명령형 퀴즈 문항 ${qz_imp_n}건 — §3.11: 문항은 평서형 질문으로 씁니다. '계산하라'가 아니라 '계산해보세요'"
      printf '%s\n' "$qz_imp_msg" | show_hits 3 "$qz_imp_n"
    fi
  fi

  # 자기 설명 프롬프트 (WARN) — 「한 장 정리」에 능동 회상 장치 1개
  if [[ -z "$wrap_s" ]]; then
    c_info "「한 장 정리」 절이 없어 자기 설명 프롬프트 검사를 건너뜁니다 (B군이 이미 FAIL)"
  else
    local recall_ln
    recall_ln=$(awk -v s="$wrap_s" -v e="$wrap_e" '
      NR>=s && NR<=e && /말해보/ && (/덮고/ || /보지 않고/) { print NR; exit }' "$f")
    if [[ -n "$recall_ln" ]]; then
      c_pass "자기 설명 프롬프트 있음 (L${recall_ln})"
    else
      c_warn "「한 장 정리」에 자기 설명 프롬프트가 없습니다 — §3.11: 능동 회상 장치를 1개 둡니다. '**덮고 말해보세요.** 이 그림을 보지 않고 ...를 순서대로 말해보세요'"
    fi
  fi

  # 문체 통일 (FAIL) — 본문·📌 박스·「한 장 정리」 표·퀴즈를 전부 합니다체로
  # 판정 등급 근거: 7개 lesson 실측 438건에서 표본 30건과 명사형 전수 13건을 눈으로 확인해
  # 오탐 0건이었습니다. 헤딩 제외와 명사 스톱리스트가 오탐 갈래 둘을 걷어낸 결과입니다
  local st_n st_msg
  st_msg=$(style_scan "$f" | awk -F'\t' '{printf "L%s  [%s] %s\n", $1, $2, $3}')
  st_n=$(printf '%s\n' "$st_msg" | awk 'NF{n++} END{print n+0}')
  if [[ "$st_n" -eq 0 ]]; then
    c_pass "문체가 합니다체로 통일됨"
  else
    c_fail "합니다체가 아닌 종결 ${st_n}곳 — §3.11: 본문, 📌 박스, 「한 장 정리」 표, 퀴즈를 전부 합니다체로 씁니다. W1-M1은 본문이 합니다체인데 📌 박스가 음슴체라 한 문서 안에서 세 목소리가 났습니다"
    printf '%s\n' "$st_msg" | show_hits 5 "$st_n"
  fi

  # ── I. 3층 문서 §2.1 ───────────────────────────────────────
  # 검사 대상은 lesson.md와 같은 디렉토리의 eli5.md입니다.
  #
  # 2026-08-25 개정으로 한 모듈의 읽을거리가 세 층이 됐습니다(§2.1).
  # eli5.md(먼저 읽는 그림 중심 입문) → lesson.md(본문) → deep-dive.md(심화)이고,
  # eli5.md는 §2 산출물 6번에서 **필수**로 규정됐습니다.
  #
  # **수식 0, arXiv 인용 0, 영문 약어 최소, frontmatter 없음.** 이 넷이 층을 가릅니다.
  # 산문 밴드 3,000~5,000자는 **lesson.md의 티어 밴드와 무관한 독립 밴드**라
  # 그 모듈이 Tier A든 C든 같은 숫자를 씁니다. 그래서 여기서는 tier를 보지 않습니다.
  printf '\n  \033[1mI. 3층 문서 (§2.1)\033[0m\n'
  local eli5 e_scan e_prose e_total e_protect
  eli5="$(dirname "$f")/eli5.md"

  if [[ ! -f "$eli5" ]]; then
    # TODO(Phase 8): eli5.md가 전 모듈로 확산되면 이 WARN을 FAIL로 올립니다.
    #   2026-08-27 현재 7개 모듈 전부 eli5.md가 없고, 생성은 Phase 5(W1-M1 시범)와
    #   Phase 8(확산)에서 일어납니다. 전환기 동안의 부재까지 FAIL로 잡으면
    #   lint가 상시 빨간불이 되어 신호가 죽습니다.
    #   확산이 끝나면 이 분기를 c_fail로 바꾸고 이 주석을 지우세요.
    c_warn "eli5.md 없음 — §2가 필수 산출물로 규정한 문서입니다(2026-08-25 신설). 지금은 확산 전 전환기라 WARN이고, Phase 8 확산이 끝나면 FAIL로 올립니다"
  else
    e_scan=$(eli5_scan "$eli5")
    # 빈 파일에는 prose_metrics()를 부르지 않습니다. 보호 구간 백분율이 0으로 나누기가 됩니다.
    # prose_metrics()는 §9.6 실측값과의 비교 가능성 때문에 한 글자도 고치지 않습니다
    e_prose=0; e_total=0; e_protect=0
    if [[ -s "$eli5" ]]; then
      read -r e_prose e_total e_protect < <(prose_metrics "$eli5")
      : "${e_prose:=0}" "${e_total:=0}" "${e_protect:=0}"
    fi

    # 1. 산문 3,000~5,000자 (§2.1). 미달과 초과 둘 다 FAIL입니다.
    #    미달은 대개 lesson.md의 요약을 짧게 쓴 것이고(그러면 층이 성립하지 않습니다),
    #    초과는 본문이 eli5로 새어 들어온 것입니다.
    if   [[ "$e_prose" -lt 3000 ]]; then
      c_fail "eli5 산문 ${e_prose}자 — §2.1 밴드(3,000~5,000) 미달. 그림과 이야기를 더 붙이세요. 요약을 짧게 쓴 것이면 읽기 부담만 늘어납니다"
    elif [[ "$e_prose" -gt 5000 ]]; then
      c_fail "eli5 산문 ${e_prose}자 — §2.1 밴드(3,000~5,000) 초과. 정밀한 설명은 lesson.md의 몫입니다"
    else
      c_pass "eli5 산문 ${e_prose}자 (§2.1 독립 밴드 3,000~5,000, 티어 무관)"
    fi

    # 2. frontmatter 없음 (§2.1). 첫 줄이 --- 이면 frontmatter입니다
    local e_l1
    e_l1=$(head -1 "$eli5" | tr -d '\r' | sed 's/[[:space:]]*$//')
    if [[ "$e_l1" == "---" ]]; then
      c_fail "eli5에 frontmatter가 있습니다 — §2.1: eli5.md는 frontmatter를 두지 않습니다. 첫 줄은 lesson.md 되돌이 링크입니다"
    else
      c_pass "eli5 frontmatter 없음"
    fi

    # 3. 첫 줄 되돌이 링크 (§2.1)
    #    문장 끝 표현은 조금 달라도 되게 세 조각의 포함 여부로 판정합니다
    local e_first_ln="" e_first_txt=""
    read -r e_first_ln e_first_txt \
      < <(printf '%s\n' "$e_scan" | awk -F'\t' '$1=="FIRST"{print $2"\t"$3; exit}')
    if [[ -z "$e_first_ln" ]]; then
      c_fail "eli5가 비어 있습니다"
    elif [[ "$e_first_txt" == *"본문은"* && "$e_first_txt" == *"lesson.md"* \
            && "$e_first_txt" == *"먼저 읽으세요"* ]]; then
      c_pass "eli5 첫 줄 되돌이 링크 있음 (L${e_first_ln})"
    else
      c_fail "eli5 첫 줄(L${e_first_ln})이 되돌이 링크가 아닙니다 — §2.1: '> 본문은 [lesson.md](lesson.md)입니다. 이 문서를 먼저 읽으세요.'"
      printf '        %s\n' "L${e_first_ln}  ${e_first_txt}"
    fi

    # 4. 수식 0 (§2.1). $$ 블록도 인라인도 쓰지 않습니다. 코드펜스 안은 제외합니다
    local e_math_n e_math_msg
    e_math_msg=$(printf '%s\n' "$e_scan" \
      | awk -F'\t' '$1=="MATH"{printf "L%s  %s  %s\n", $2, $3, substr($4,1,80)}')
    e_math_n=$(printf '%s\n' "$e_math_msg" | awk 'NF{n++} END{print n+0}')
    if [[ "$e_math_n" -eq 0 ]]; then
      c_pass "eli5 수식 0"
    else
      c_fail "eli5 수식 ${e_math_n}곳 — §2.1: eli5는 수식 0입니다. 수식은 lesson.md와 deep-dive.md의 몫입니다"
      printf '%s\n' "$e_math_msg" | show_hits 3 "$e_math_n"
    fi

    # 5. arXiv 인용 0 (§2.1). 논문 이름과 번호는 lesson.md의 「출처」가 담당합니다
    local e_ax_n e_ax_msg
    e_ax_msg=$(printf '%s\n' "$e_scan" \
      | awk -F'\t' '$1=="ARXIV"{printf "L%s  %s  %s\n", $2, $3, substr($4,1,80)}')
    e_ax_n=$(printf '%s\n' "$e_ax_msg" | awk 'NF{n++} END{print n+0}')
    if [[ "$e_ax_n" -eq 0 ]]; then
      c_pass "eli5 arXiv 인용 0"
    else
      c_fail "eli5 arXiv 인용 ${e_ax_n}곳 — §2.1: eli5는 arXiv 인용 0입니다. 논문 이름과 번호는 lesson.md의 「출처」가 담당합니다"
      printf '%s\n' "$e_ax_msg" | show_hits 3 "$e_ax_n"
    fi

    # 6. 골격 5절 (§2.1). 제목 문구가 조금 달라질 수 있어 핵심 구절로 느슨하게 봅니다
    local e_titles e_skel_missing="" e_skel_n=0 want
    e_titles=$(h2_blocks "$eli5" | cut -f3)
    for want in "이 모듈이 답하는 질문" "어떤 문제가 있었나" "그래서 무엇을 하나" \
                "오늘 만날 개념" "이제 lesson.md로"; do
      if printf '%s\n' "$e_titles" | grep -qF "$want"; then
        e_skel_n=$((e_skel_n+1))
      else
        e_skel_missing="${e_skel_missing} 「${want}」"
      fi
    done
    if [[ -z "$e_skel_missing" ]]; then
      c_pass "eli5 골격 5절 완비"
    else
      c_fail "eli5 골격 절 $((5-e_skel_n))개 누락:${e_skel_missing} — §2.1 골격"
    fi

    # 7. 줄표와 가운뎃점 (§3.10). G군과 **같은 산문 라인 판정**을 씁니다
    local e_dash="" e_dot="" e_punct_msg e_punct_n
    read -r e_dash e_dot < <(punct_counts "$eli5")
    : "${e_dash:=0}" "${e_dot:=0}"
    if [[ "$e_dash" -lt 3 && "$e_dot" -lt 3 ]]; then
      c_pass "eli5 산문 줄표(—) ${e_dash}회 · 가운뎃점(·) ${e_dot}회"
    else
      c_fail "eli5 산문 줄표(—) ${e_dash}회, 가운뎃점(·) ${e_dot}회 — §3.10은 eli5에도 그대로 적용됩니다. 부연은 쉼표나 괄호로, 나열은 쉼표로"
      e_punct_msg=$(grep -n '[—·]' "$eli5" | grep -v '^[0-9]*:[[:space:]]*|' || true)
      e_punct_n=$(printf '%s\n' "$e_punct_msg" | awk 'NF{n++} END{print n+0}')
      printf '%s\n' "$e_punct_msg" | show_hits 3 "$e_punct_n"
    fi

    # 8. 설명 회피·「이미 안다」 문장 (§3.9). 기존 dodge_scan()을 eli5에 대고 부릅니다
    local e_dodge_n e_dodge_msg
    e_dodge_msg=$(dodge_scan "$eli5" | awk -F'\t' '{printf "L%s  %s\n", $1, $2}')
    e_dodge_n=$(printf '%s\n' "$e_dodge_msg" | awk 'NF{n++} END{print n+0}')
    if [[ "$e_dodge_n" -eq 0 ]]; then
      c_pass "eli5에 설명 회피·「이미 안다」 문장 없음"
    else
      c_fail "eli5 설명 회피·「이미 안다」 문장 ${e_dodge_n}건 — §3.9: eli5는 아무 용어도 모르는 사람이 읽는 층입니다. 여기서 설명을 건너뛰면 갈 곳이 없습니다"
      printf '%s\n' "$e_dodge_msg" | show_hits 3 "$e_dodge_n"
    fi

    # 9. 표 (§2.1, WARN). eli5는 그림과 이야기의 층이고 표는 대조의 도구입니다.
    #    표를 쓰고 싶어지면 대개 요약을 하고 있다는 신호입니다
    local e_tbl_n e_tbl_max e_tbl_msg
    e_tbl_n=$(printf '%s\n' "$e_scan" | awk -F'\t' '$1=="TABLE"{n++} END{print n+0}')
    e_tbl_max=$(printf '%s\n' "$e_scan" \
      | awk -F'\t' '$1=="TABLE" && $4+0>mx{mx=$4+0} END{print mx+0}')
    if [[ "$e_tbl_n" -eq 0 ]]; then
      c_pass "eli5에 표 없음"
    elif [[ "$e_tbl_n" -eq 1 && "$e_tbl_max" -le 3 ]]; then
      c_pass "eli5 표 1개 (${e_tbl_max}행, 3행 이하)"
    else
      c_warn "eli5 표 ${e_tbl_n}개, 최대 ${e_tbl_max}행 — §2.1: 표를 쓰지 않는 것이 기본이고 꼭 필요하면 3행 이하로 하나만 둡니다. 그 내용은 대개 lesson.md의 비교표 자리입니다"
      e_tbl_msg=$(printf '%s\n' "$e_scan" \
        | awk -F'\t' '$1=="TABLE"{printf "L%s-%s  %s행\n", $2, $3, $4}')
      printf '%s\n' "$e_tbl_msg" | show_hits 3 "$e_tbl_n"
    fi

    # 10. 「정확히는 lesson.md §N 「절 제목」에서」 링크 (§2.1, WARN)
    #     lesson.md와 § 절 참조가 한 줄에 함께 있으면 되돌이 링크로 봅니다.
    #     첫 줄 안내 인용문에는 §가 없어 여기 세어지지 않습니다
    local e_back_n
    e_back_n=$(printf '%s\n' "$e_scan" | awk -F'\t' '$1=="BACKLINK"{n++} END{print n+0}')
    if [[ "$e_back_n" -gt 0 ]]; then
      c_pass "lesson.md 절 되돌이 링크 ${e_back_n}개"
    else
      c_warn "lesson.md 절 되돌이 링크가 하나도 없습니다 — §2.1: eli5는 정확도를 일부 포기하는 층입니다. 단순화한 자리마다 「정확히는 [lesson.md](lesson.md) §N 「절 제목」에서」를 다세요. 어디서 정확해지는지를 알려주지 않는 단순화는 그냥 틀린 설명입니다"
    fi

    # 11. 절 참조에 절 제목 병기 (§2.1, WARN)
    #     lesson.md는 재집필로 절 번호가 밀립니다. 제목을 같이 적어두면
    #     하류 참조가 조용히 깨지는 것을 막습니다
    local e_sec_n e_sec_msg
    e_sec_msg=$(printf '%s\n' "$e_scan" \
      | awk -F'\t' '$1=="SECREF"{printf "L%s  %s 뒤에 「절 제목」이 없습니다\n", $2, $3}')
    e_sec_n=$(printf '%s\n' "$e_sec_msg" | awk 'NF{n++} END{print n+0}')
    if [[ "$e_sec_n" -eq 0 ]]; then
      c_pass "eli5 절 참조에 절 제목 병기됨"
    else
      c_warn "절 제목 없는 절 참조 ${e_sec_n}개 — §2.1: '§3'이 아니라 '§3 「스택 5계층」'입니다. 번호가 밀려도 독자가 찾아갈 수 있게"
      printf '%s\n' "$e_sec_msg" | show_hits 3 "$e_sec_n"
    fi

    # 12. 그림 「읽는 법」 캡션 (§3.2, WARN). 기존 figure_scan()을 eli5에 대고 부릅니다
    local e_vis e_fig_n e_cap_bad e_cap_msg
    e_vis=$(figure_scan "$eli5")
    e_fig_n=$(printf '%s\n' "$e_vis" | awk -F'\t' '$1=="FIG"{n++} END{print n+0}')
    e_cap_bad=$(printf '%s\n' "$e_vis" | awk -F'\t' '$1=="FIG" && $5=="0"{n++} END{print n+0}')
    if [[ "$e_fig_n" -eq 0 ]]; then
      c_info "eli5에 캡션을 검사할 그림이 없습니다 (§2.1: eli5는 그림 위주의 층입니다)"
    elif [[ "$e_cap_bad" -eq 0 ]]; then
      c_pass "eli5 그림 ${e_fig_n}개 전부 다음 3줄 안에 「읽는 법」 캡션 있음"
    else
      c_warn "eli5 캡션 없는 그림 ${e_cap_bad}/${e_fig_n}개 — §3.2: 그림 바로 아래 1~3문장으로 어디를 먼저 보고 무엇을 따라가라고 지시하세요"
      e_cap_msg=$(printf '%s\n' "$e_vis" | awk -F'\t' '
        $1=="FIG" && $5=="0" {
          loc = ($2==$3) ? "L" $2 : "L" $2 "-" $3
          printf "%s  %s 뒤 3줄에 산문이 없습니다\n", loc, $4
        }')
      printf '%s\n' "$e_cap_msg" | show_hits 3 "$e_cap_bad"
    fi

    # 13. 영문 약어 (§2.1, WARN). 한국어로 풀어 쓰는 것이 1순위입니다
    local e_abbr_n e_abbr_list
    e_abbr_n=$(printf '%s\n' "$e_scan" | awk -F'\t' '$1=="ABBR"{n++} END{print n+0}')
    e_abbr_list=$(printf '%s\n' "$e_scan" | awk -F'\t' '$1=="ABBR"{printf " %s", $2}')
    if [[ "$e_abbr_n" -le 3 ]]; then
      c_pass "eli5 영문 약어 ${e_abbr_n}종${e_abbr_list}"
    else
      c_warn "eli5 영문 약어 ${e_abbr_n}종:${e_abbr_list} — §2.1: 영문 약어 최소가 규격입니다. 한국어로 풀어 쓰는 것이 1순위이고, 이름이 꼭 필요할 때만 한국어를 앞세우고 영문을 괄호로 답니다"
    fi
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
