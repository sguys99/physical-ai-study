---
name: humanize-korean
version: "1.6.0"
description: AI(ChatGPT·Claude·Gemini 등)가 쓴 한글 텍스트를 "사람이 쓴 글처럼" 윤문해주는 오케스트레이터 스킬. 번역투·영어 인용 과다·기계적 병렬·관용구·피동태 남용·접속사 남발·리듬 균일성·이모지/불릿 과다 등 10대 카테고리 40+ AI 티 패턴을 탐지·분류해 내용은 한 글자도 건드리지 않고 문체·리듬·표현만 자연스러운 한국어로 재작성한다. 트리거 — "AI 티 없애줘", "AI 같은 글 자연스럽게", "GPT/ChatGPT 문체", "AI 번역투 고쳐", "사람이 쓴 것처럼 윤문", "AI 윤문", "ChatGPT 티 제거", "한글 AI 탐지·윤문", "AI 글 사람처럼", "번역투 제거", "영어 인용 많은 글 윤문", "AI 글 티 안 나게", "휴머나이저", "humanize Korean", "AI detector bypass 한글". 후속 작업 — "특정 카테고리만 다시", "윤문 강도 조정", "장르 바꿔서", "이 문단만", "2차 윤문" 도 모두 이 스킬. 교육자료(`course/**/lesson.md`) 윤문은 `register: course` 프로파일로 호출 — strict 강제 + 산문 라인 기준 판정 + 문단 흐름 복원 + 문학체 어휘 드리프트 차단. 단순 맞춤법·오탈자 교정은 직접 처리, 번역은 번역 스킬, 내용 추가·삭제를 동반한 재작성은 별도 집필 스킬.
---

# Humanize Korean — AI 한글 티 제거 오케스트레이터 (v1.6)

> **v1.6 변경 고지 (2026-08-02) — `register: course` 프로파일 신설**
> 이 저장소의 교육자료(`course/**/lesson.md`) 윤문에서 두 가지 실패가 관찰됐습니다. ① 저윤문 — AI 티가 남았는데 전체 문자 변경률이 낮게 나와 등급 A로 통과(W1-M2: 1.48%). ② 과교정 — 번역투를 걷어내려다 잘 안 쓰는 문어체 어휘로 갈아탐. v1.6은 두 축을 함께 잡는 `register: course` 프로파일을 추가합니다. 그 외 동작은 v1.5 그대로입니다.
>
> - **신설**: `register: course` — strict 강제 + 산문 라인 기준 판정 + 문단 흐름 복원 + plain-language 가드.
> - **판정 지표 SSOT 이관**: `register: course`의 변경률 판정은 `docs/course-plan.md` **§1의 「변경률 판정 지표」 박스**를 따릅니다. 전체 문자 변경률은 기록만 하고 단독 판정 근거로 쓰지 않습니다. (v1.6 원문은 SSOT를 「§9.1(티어별 지표)」로 적었는데 §9.1은 W1-M2 윤문 실측 기록이고 규정이 아닙니다. 2026-09-01에 §1로 바로잡았습니다.)
> - 출처: `sguys99/ai-wiki`의 `register: wiki`(2026-07-10)를 이 저장소 문서 구조에 맞춰 이식.

> **v1.5 변경 고지 (2026-04-26) — v1.1 베이스라인 + Monolith Fast Path**
> v1.2(voice profile)·v1.3(candidate pool)·v1.4(역할별 모델 분산)는 모두 핫패스 비용을 잡지 못해 5,000자 입력에 25분이 걸렸습니다. v1.5는 **v1.1 단순 구조로 롤백한 뒤 단일 호출 monolith 에이전트만 추가**한 설계입니다.
>
> - **Fast 모드(디폴트)** — `humanize-monolith` 에이전트가 한 콜에서 탐지·윤문·자체검증 일괄 처리. 도구 호출 4~5회. 5,000자 이하 wall-clock 2~3분 목표.
> - **Strict 모드(`--strict`)** — v1.1 5인 파이프라인 그대로(detector·rewriter·auditor·reviewer + taxonomist 분류 자산 유지). 정밀 검증·장문(8,000자+) 처리·etc.
> - **삭제됨**: voice profile·candidate pool·promotion-checklist·sample-collection·권한 위계 §1~§6.
> - **유지됨**: 분류 체계 본진(C-9·C-10·D-7·H-3·I-3·I-4 등 v1.2~v1.3.1 신규 패턴)·rewriting-playbook·5인 에이전트 정의(strict 모드 백본).

## Phase 0: 컨텍스트 확인 및 모드 결정

작업 시작 시 가장 먼저 다음 한 줄을 사용자에게 출력한다.

```
humanize-korean v1.6 — {fast|strict} 모드{, register: course} / run_id: {YYYY-MM-DD-NNN}
```

### 모드 결정
- 사용자가 `--strict`·"정밀 모드"·"5인 파이프라인" 명시 → **strict**
- **`register: course` (또는 대상 파일이 `course/**/lesson.md`) → strict 강제** (길이 무관, 자동 승급 + 1줄 고지). 교육자료 산문은 얕은 Fast Path로는 티가 남아, 독립 탐지·자연성 리뷰가 있는 strict로만 처리한다. §register: course 참조.
- 입력 8,000자 초과 → **strict** (자동 승급 + 사용자에 1줄 고지)
- 그 외 모두 → **fast (디폴트)**

### register: course (교육자료 전용)

`register: course`는 이 저장소의 `course/**/lesson.md` 교육자료 산문에 특화된 프로파일이다. 다음 4가지를 한 묶음으로 적용한다:

1. **strict 강제** — 위 모드 결정대로 5인 파이프라인으로 라우팅.
2. **판정 지표를 산문 라인 기준으로 전환** — 아래 밴드 블록 참조. lesson.md는 문자의 40% 이상이 코드·표·수식(전량 보존 대상)이라 **전체 문자 변경률이 구조적으로 낮게 나온다.** 그 수치로 저윤문을 판정하면 오판이고, 재윤문을 지시하면 오히려 과윤문을 유발한다.
3. **문단 흐름 복원 허용** — 국소 span 편집을 넘어 문단 단위 재구조화 허용(문서 전체 재작성은 여전히 금지). `rewriting-playbook.md §1.Y` 참조.
4. **plain-language 가드** — 번역투를 피하려다 잘 안 쓰는 문어체·문학체 어휘로 갈아타지 않는다. `quick-rules.md`·`rewriting-playbook.md`의 "흔한 어휘 우선" 규칙 적용.

**변경률 판정 밴드** (SSOT는 `docs/course-plan.md` §1의 「변경률 판정 지표」 박스. 아래는 사본이며, 불일치 시 course-plan이 우선한다)

| 대상 | 주지표 | 목표 |
|---|---|---|
| **`course/**/lesson.md` 전부** (티어·보호 구간 비율과 무관) | 산문 라인 변경률 + 잔존 burden 감소율 | 라인 **20~50%** · burden **80%↓** 이상 |
| lesson이 아닌 짧은 산문 문서 (보호 구간 35% 미만) | 전체 문자 변경률 | **5~30%** |

> ⚠️ **2026-09-01 개정.** 이 표는 원래 「Tier A는 산문 라인, Tier B·C는 전체 문자」로 티어를 갈랐고 그 전에는 보호 구간 35%로 갈랐다. **둘 다 폐기됐다.** 티어는 애초에 원인이 아니었고(§9.6), 35% 문턱은 1차 규약 문서로 교정된 값인데 2차 규약이 구조물을 산문으로 옮기면서 그 비율을 구조적으로 끌어내려 W1-M1이 38%, W1-M2가 32%로 갈렸다. 같은 규약으로 쓴 두 문서가 지표를 달리 받는 것은 결함이다. **`course/**/lesson.md`는 무조건 산문 라인 판정이다.** 근거는 course-plan §1.

- **산문 = 코드펜스·표 행·LaTeX 블록·Mermaid·frontmatter를 제외한 라인.**
- 전체 문자 변경률은 **기록만** 하고 단독 판정 근거로 쓰지 않는다.
- **저윤문** — 주지표가 하한 미달이면서 S1이 잔존 → `rewrite_round_2`.
- **과윤문** — 주지표 상한 초과, 또는 문학체 드리프트 어휘 3개 이상 → `rollback_and_rewrite`.
- 상한을 올려도 안전한 이유는 strict의 `content-fidelity-auditor`가 의미 불변을 독립 감사하기 때문이다.

장르 힌트(칼럼·리포트 등)와 별개 축이다. register는 문체 register를, genre_hint는 글의 장르를 가리킨다. 오케스트레이터는 register=course일 때 하위 에이전트(`korean-style-rewriter`·`naturalness-reviewer`)에 `register: course`와 티어를 전달한다.

### run_id 결정
- 모든 경로는 **cwd 기준**. 새 폴더 생성도 cwd 기준 `_workspace/{YYYY-MM-DD-NNN}/`에 만든다.
- 기존 시퀀스 확인은 **`Glob` 도구**로 표지 파일을 매칭해 간접 조회.
  올바른 사용법: `Glob(pattern="_workspace/YYYY-MM-DD-*/01_input.txt")` → 결과에서 폴더명 추출 후 NNN 최댓값 + 1.
  주의: Glob은 디렉토리 자체는 매칭하지 못한다. 반드시 그 안의 표지 파일(`01_input.txt`)을 매칭할 것.
  `Bash ls`는 OS·셸 환경에 따라 경로 해석이 달라지므로 사용 금지.
- 당일 폴더가 없으면 NNN = 001. 있으면 마지막 NNN + 1.
- 부분 재실행 신호("이 카테고리만 다시"·"2차 윤문")일 경우 기존 run_id 재사용 + strict 모드로 자동 승급.

## Fast 모드 (디폴트)

### Phase 1: 입력 저장
1. cwd 기준 `_workspace/{run_id}/` 생성
2. 입력 텍스트를 `01_input.txt`에 저장
3. 첫 300자로 장르 자동 추정 (사용자 명시 시 우선)

### Phase 2: Monolith 호출
`humanize-monolith` 에이전트를 `Agent` 도구로 1회 호출.

입력:
```
input_path: <abs path>/_workspace/{run_id}/01_input.txt
quick_rules_path: ${CLAUDE_SKILL_DIR}/references/quick-rules.md
genre_hint: 칼럼 | 리포트 | 블로그 | 공적 | null
```

출력 (에이전트가 직접 작성):
- `_workspace/{run_id}/final.md` — 윤문본
- `_workspace/{run_id}/summary.md` — 메트릭·자체검증·하이라이트

monolith는 단일 호출 안에서 다음을 모두 수행 (자세히는 에이전트 정의 참조):
1. quick-rules 룰북 로드 → 메모리에서 패턴 탐지 + 윤문 + 자체검증 6항 점검
2. 변경률 50% 초과 시 자동 롤백
3. 자체검증 위반 시 1회 부분 재실행
4. final.md + summary.md 작성

### Phase 3: 결과 전달
사용자에게 다음 4개를 반환:
1. 한 줄 상태: `완료. 변경률 X% / 등급 Y / 자체검증 N/6 통과`
2. 윤문본 본문 (마크다운 블록)
3. summary.md의 핵심 표 (메트릭 + 카테고리 탐지 + 자체검증)
4. 등급 B 이하면 "정밀 검증이 필요하면 `--strict`로 5인 파이프라인" 안내

**디폴트 wall-clock 목표:** 5,000자 이하 2~3분, 8,000자 5~7분.

## Strict 모드 (`--strict` 또는 자동 승급)

v1.1 5인 파이프라인 그대로. 검증 분리·재윤문 루프가 의미 있을 때만 사용.

### Phase A: 탐지
`ai-tell-detector` 호출 → `02_detection.json`

### Phase B: 윤문 (최대 3회 루프)
`korean-style-rewriter` 호출 → `03_rewrite.md` + `03_rewrite_diff.json`

`register: course`면 호출 인자에 `register: course`와 `tier: A|B`를 함께 전달한다. Phase C의 `naturalness-reviewer`에도 동일하게 전달한다.

### Phase C: 병렬 검증 (에이전트 팀)
`TeamCreate`로 `humanize-review-team` 구성:
- `content-fidelity-auditor` → `04_fidelity_audit.json` (의미 동등성)
- `naturalness-reviewer` → `05_naturalness_review.json` (잔존·과윤문)

`TeamDelete` 후 종합 판정 매트릭스에 따라 분기:

| fidelity | naturalness | 종합 | 후속 |
|---|---|---|---|
| full_pass | accept / accept_with_note | **최종 승인** | Phase D |
| full_pass | rewrite_round_2 | **2차 윤문** | Phase B 재호출 (target finding) |
| full_pass | rollback_and_rewrite | **롤백 후 재윤문** | 윤문가에 edit 롤백 지시 |
| conditional_pass | - | **롤백된 edit만 재시도** | Phase B 재호출 |
| fail | - | **전면 재작업** | Phase B 전면 재호출 |

2차/3차 윤문 진입 시 `03_rewrite_v2.md`·`v3.md`로 버전 분리. **최대 3회 후 미해결이면 `hold_and_report`**로 사람 개입.

### Phase D: 최종 출력
1. `final.md`에 최종 윤문본 복사
2. `summary.md` 생성 (fast 모드와 동일 포맷)
3. 사용자에게 결과 + 등급 + 안내

## 부분 재실행 / 후속 명령

| 사용자 신호 | 처리 |
|---|---|
| "특정 카테고리만 다시" | strict 모드로 자동 전환, 해당 카테고리 finding만 Phase B 재실행 |
| "이 문단만" | strict 모드, 해당 문단만 입력으로 새 run_id 생성 |
| "2차 윤문"·"`/humanize-redo`" | 기존 run_id의 `final.md`를 새 입력으로 strict Phase B 재실행 |
| "윤문 강도 조정" | strict 모드, `min_severity` 옵션 변경 후 Phase A부터 재실행 |
| "장르 바꿔서" | `genre_hint` 변경 후 Phase A부터 재실행 |

## 옵션 (인자 끝에 자연어로)

- `register: course` — 교육자료 프로파일 (strict 강제 + 산문 라인 기준 판정 + 문단 흐름 복원 + plain-language 가드). §register: course 참조. `course/**/lesson.md`가 대상이면 자동 적용.
- `tier: A|B` — `register: course`가 하위 에이전트에 넘기는 참고값 (생략 시 `docs/course-plan.md` §4에서 해당 모듈 티어를 확인하고, 확인 불가면 A로 간주). **판정 지표를 가르지는 않는다** — lesson은 티어와 무관하게 산문 라인 판정이다. Tier C는 2026-09-01에 폐지됐고 남은 것은 A와 B뿐이다
- `장르: 칼럼|리포트|블로그|공적` — 장르 명시 (생략 시 자동 추정)
- `강도: 보수|기본|적극` — 윤문 강도 (기본값: 기본)
- `최소심각도: S1|S2|S3` — 탐지 임계값 (기본값: S2)
- `--strict` — 5인 파이프라인 강제 사용

## 데이터 흐름 요약

### Fast 모드 (디폴트)
```
01_input.txt
    ↓ [humanize-monolith — 단일 호출]
    ├ 메모리: quick-rules 로드 → 탐지 → 윤문 → 자체검증
    └→ final.md + summary.md
```

### Strict 모드
```
01_input.txt
    ↓ [ai-tell-detector]
02_detection.json
    ↓ [korean-style-rewriter]
03_rewrite.md + 03_rewrite_diff.json
    ↓ [병렬 팀]
    ├→ [content-fidelity-auditor] → 04_fidelity_audit.json
    └→ [naturalness-reviewer]      → 05_naturalness_review.json
    ↓ [오케스트레이터 종합]
    ├→ (재작업) Phase B로 복귀 (최대 3회)
    └→ (승인) final.md + summary.md
```

## 에이전트 호출 규칙

**모델:** 모두 `model: opus` 통일 (v1.1 베이스라인). 모델 다운그레이드는 v1.4에서 시도했으나 도구 호출 chain이 진짜 병목이라 효과 미미했음.

**에이전트 정의 위치:** 저장소 루트 `agents/`에 12종 정의(플러그인 컨벤션). Claude Code 탐색 경로:
1. 플러그인 설치 시 — `humanize-korean` 플러그인이 `agents/`를 번들로 제공(전역).
2. 스크립트 설치 시 — `install.sh`가 `agents/*.md`를 `~/.claude/agents/`에 심링크(전역).

필요 에이전트 6종:
- `humanize-monolith` (v1.5 신규, fast 전용)
- `ai-tell-detector` · `korean-style-rewriter` · `content-fidelity-auditor` · `naturalness-reviewer` (strict 5인 중 4명)
- `korean-ai-tell-taxonomist` (분류 체계 유지·확장 — 본 스킬 실행 중에는 호출 안 됨, 별도 명령으로만 트리거)

## 테스트 시나리오

### Fast 정상 흐름
- 입력: ChatGPT가 생성한 AI 칼럼 초안 (2,000~5,000자, 번역투·결말 공식·hype 어휘 풍부)
- 기대: monolith 1콜로 변경률 15~25%, 등급 A/B, wall-clock 2~3분, 자체검증 5~6/6

### Strict 정밀 검증 흐름
- 사용자 명시 `--strict` 또는 8,000자+ 입력
- 5인 파이프라인 끝까지 실행, 변경률 18~22%, 검증팀 full_pass

### register: course 흐름
- 입력: `course/**/lesson.md` 교육자료 산문 (코드·표·수식이 30%+ 섞인 문서. 2차 규약 재작성본은 보호 구간이 32~38%로 내려왔고 티어와 무관하게 산문 라인으로 판정한다)
- 기대: strict 자동 라우팅, **산문 라인 변경률 20~50% · 잔존 burden 80%↓**, S1 잔존 0, `naturalness-reviewer`가 희귀 어휘·문단 흐름 항목 통과, `content-fidelity-auditor` full_pass. 코드펜스·LaTeX·URL·표 행은 바이트 동일. plain-language 가드로 문학체 드리프트 0 시그널.
- **전체 문자 변경률이 5% 미만이어도 그 자체로는 저윤문 판정 근거가 아니다.** 주지표(산문 라인)를 보고 판정하고, 전체 문자율은 참고 기록으로만 남긴다.

### 엣지 케이스 — 이미 사람이 쓴 글
- monolith 자체 탐지에서 매치 거의 없음 → 변경률 5% 미만 + summary.md에 "윤문 불필요 가능성" 메모
- 사용자가 `--strict`로 강제 검증 가능

## 주의 사항

- **의미 불변이 최상위 불문율.** monolith·strict 모두에서 위반 즉시 롤백.
- **수치·고유명사·직접 인용은 탐지/윤문 대상 아님.** Do-NOT list 엄수.
- **장르 이탈 금지.** 칼럼이 에세이로, 에세이가 문학으로 옮겨가지 않는다.
- **register 보존.** 격식체 입력 → 격식체 출력. AI 티는 문법·수사이지 격식 자체가 아님.
- **변경률(일반) 30% 초과 → 경고, 50% 초과 → 강제 중단.** 단 `register: course`는 주지표가 다르다 — `course/**/lesson.md`는 **티어와 무관하게** 산문 라인 20~50%(하한 미달 시 저윤문, 상한 초과 시 과윤문)이다. 전체 문자 5~30%는 lesson이 아닌 짧은 산문 문서에만 쓴다. §register: course 참조.
- **문학체 드리프트 금지.** 번역투를 걷어내려다 잘 안 쓰는 문어체 어휘로 갈아타는 것은 과교정이다. 치환어는 "일반 기술 문서에서 쓰는 말인가"를 기준으로 고른다.
- **자동 로드 금지.** 프로젝트 CLAUDE.md 등 다른 파일을 자동 파싱해 옵션을 추론하지 않는다. (`register: course`의 티어 확인 목적으로 `docs/course-plan.md`를 읽는 것은 예외 — 사용자가 `tier:`를 생략했을 때만.)

## 참고 자료

- 슬림 룰북 (Fast 전용): [`references/quick-rules.md`](references/quick-rules.md) — S1·S2 핵심 패턴 + 자체검증 체크리스트
- 분류 체계 본진 (Strict 전용): [`references/ai-tell-taxonomy.md`](references/ai-tell-taxonomy.md) — 10대분류 × 40+ 패턴 전수
- 윤문 처방 (Strict 전용): [`references/rewriting-playbook.md`](references/rewriting-playbook.md) — 카테고리별 치환 레시피·장르별 허용 표
- 웹 서비스 스펙 (옵션): [`references/web-service-spec.md`](references/web-service-spec.md) — 웹 확장 시 로드
