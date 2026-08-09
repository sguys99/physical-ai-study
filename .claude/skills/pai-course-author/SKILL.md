---
name: pai-course-author
description: Physical AI 온보딩 4주 스터디의 모듈 교육자료를 집필하는 오케스트레이터 스킬. 마스터플랜(docs/physical-ai-4week-master-plan.md)의 모듈 섹션 + 회사 스택 + 갭 분석을 자동 로드해 lesson.md·practice/·labs/를 생성하고, /humanize-korean 윤문과 보존 검증까지 7단계로 진행한다. 트리거 — "/pai-course-author W1-M3", "W2-M2 교육자료 만들어줘", "이번 주 모듈 집필해줘", "W3 자료 5개 써줘", "lesson 다시 써줘", "practice만 보강", "윤문 다시". 마스터플랜 자체 수정이나 학습 실행 로그 기록은 이 스킬이 아니다.
---

# pai-course-author — Physical AI 모듈 교육자료 집필

> 집필 현황 SSOT는 `docs/course-plan.md`, 커리큘럼 SSOT는 `docs/physical-ai-4week-master-plan.md`입니다.
> 이 스킬은 두 문서를 연결해 모듈 하나를 산출물 3종으로 구현합니다.

## Phase 0 — 대상 모듈 확정

인자로 받은 모듈 ID(`W1-M3` 형식)를 아래 표로 폴더 경로에 매핑합니다. 인자가 없으면 `docs/course-plan.md`에서 **체크박스가 비어 있는 가장 앞선 모듈**을 제안하고 확인받습니다.

| 모듈 | 경로 | Tier | 우선순위 |
|---|---|---|---|
| W1-M1 | `course/w1-generative-core/01-physical-ai-landscape/` | A | P0 |
| W1-M2 | `course/w1-generative-core/02-simulator-bootcamp/` | A | P0 |
| W1-M3 | `course/w1-generative-core/03-diffusion-ddpm-dit/` | C | P0 |
| W1-M4 | `course/w1-generative-core/04-flow-matching/` | C | P0 |
| W1-M5 | `course/w1-generative-core/05-latent-discrete-fsq/` | B | P0 ★ |
| W2-M1 | `course/w2-policy-vla/01-imitation-learning-act/` | A | P0 |
| W2-M2 | `course/w2-policy-vla/02-diffusion-policy/` | A | P0 |
| W2-M3 | `course/w2-policy-vla/03-vla-lineage/` | B | P0 |
| W2-M4 | `course/w2-policy-vla/04-modern-vla-pi0-groot/` | B | P0 |
| W2-M5 | `course/w2-policy-vla/05-action-representation-fsq/` | B | P0 ★ |
| W3-M1 | `course/w3-wbc-rl/01-robot-rl-ppo-parallel/` | A | P0 |
| W3-M2 | `course/w3-wbc-rl/02-motion-imitation-amp/` | B | P0 |
| W3-M3 | `course/w3-wbc-rl/03-homie-deepdive/` | B | P0 ★ |
| W3-M4 | `course/w3-wbc-rl/04-sonic-deepdive/` | A | P0 ★★ |
| W3-M5 | `course/w3-wbc-rl/05-sim2real-deploy/` | A | P0 |
| W4-M1 | `course/w4-worldmodel-nav/01-world-model-lineage/` | C | P0 |
| W4-M2 | `course/w4-worldmodel-nav/02-wfm-wam/` | B | P1 |
| W4-M3 | `course/w4-worldmodel-nav/03-navigation-dualmap/` | A | P0 ★ |
| W4-M4 | `course/w4-worldmodel-nav/04-system-integration/` | B | P0 |
| W4-M5 | `course/capstone/` | 별도 | P0 |

시작 시 한 줄을 출력합니다:

```
pai-course-author — {모듈ID} / {경로} / Tier {A|B|C} / {신규|재집필|부분보강}
```

## Phase 1 — 컨텍스트 로드

다음을 **반드시** 읽고 시작합니다. 순서대로:

1. `docs/physical-ai-4week-master-plan.md` — **해당 모듈 섹션 + §2(회사 스택) + §3(갭 분석)**
2. `docs/course-plan.md` — §1 표준 절차, **§3.2 시각자료 · §3.7 문서 구조 · §3.8 용어 도입 · §3.9 이해 사다리**, §4 티어
3. `CLAUDE.md` — 작성 표준 (규약 본문은 course-plan에 있고 여기는 포인터)
4. `references/lesson-template.md` — 골격
5. 인접 모듈의 기존 `lesson.md`(있으면) — 선수 개념 중복을 피하고 링크로 연결하기 위해

> ⚠️ 플랜 없이 일반 지식으로 쓰면 회사 스택 연결이 빠져 가치가 절반이 됩니다. 이 단계를 건너뛰지 마세요.

## Phase 2 — 최신성 검증

**버전이 빠르게 바뀌는 항목만** 공식 문서로 확인합니다. 전면 조사 금지.

| 대상 | 확인할 것 | 출처 |
|---|---|---|
| LeRobot | 설치법, 데이터셋 포맷, 학습 CLI | github.com/huggingface/lerobot |
| mujoco_playground | 설치, G1 locomotion 태스크명 | github.com/google-deepmind/mujoco_playground |
| MuJoCo / MJX | 버전, EGL 렌더링 옵션 | mujoco.readthedocs.io |
| mujoco_menagerie | G1 모델 경로 | github.com/google-deepmind/mujoco_menagerie |
| unitree_mujoco / unitree_rl_gym / unitree_sdk2 | 배포 경로 | github.com/unitreerobotics |
| DualMap | 데모 실행법 | github.com/Eku127/DualMap |

확인한 URL과 날짜를 frontmatter `sources_checked`와 lesson 출처 섹션에 남깁니다. `requirements.txt`는 여기서 확인한 버전으로 고정합니다.

## Phase 3 — 초안 생성

`pai-agent` 에이전트를 **Agent 도구로 1회 호출**합니다. 전달할 것:

```
module_id: W1-M3
target_dir: course/w1-generative-core/03-diffusion-ddpm-dit/
tier: C                      # 본문 산문 문자: A=12,000~18,000 / B=10,000~15,000 / C=9,000~13,000 (초안은 중앙을 노릴 것 — 윤문이 2~5% 늘림)
master_plan_section: <해당 모듈 섹션 원문 그대로 붙여넣기>
company_stack: <마스터플랜 §2.2 표>
gap_analysis: <마스터플랜 §3 표>
latest_checks: <Phase 2에서 확인한 버전·URL·날짜>
adjacent_modules: <앞뒤 모듈의 slug와 이미 다룬 개념 목록>
```

**payload에 함께 강조해 전달할 것** (2026-08-09 신설 규약이라 놓치기 쉽습니다):

- 섹션 순서는 §3.7 고정. **`0. 이 모듈 지도`와 `한 장 정리`는 필수**
- 대절마다 **용어표 + `📌 여기까지 정리`** 3줄
- **약어 최초 1회 전개 · 역참조 금지 · 한 줄 요약과 §0에 미정의 용어 금지** (§3.8)
- **LLM 응용 실무는 있으나 생성모델 내부 기제는 갭** — 가정 금지 (§3.9)
- **표는 축 2개 이상인 비교만.** 10행·셀 120자 상한, 표 블록 ≤ 불릿 블록 (§3.2)
- 밴드 초과분은 자르지 말고 **`deep-dive.md`로 이관 후 본문에서 링크**

산출물: `lesson.md` + `practice/` + `labs/` (+ 필요시 `deep-dive.md`). **이 단계에서 윤문하지 않습니다.**

## Phase 4 — 정적·실행 검증

**먼저 린트를 돌리고, 남는 것만 `references/authoring-checklist.md`로 육안 확인**합니다.

```bash
bash scripts/lint-lesson.sh <target_dir>/lesson.md
```

린트가 검사하는 것: §0·「한 장 정리」 헤딩 존재 · 대절별 용어표와 `📌` 박스 · 표 행수·셀 길이 · 표:불릿 비 · 미전개 약어 · 산문 밴드 · 설명 회피 문장 · frontmatter 필드.

린트가 못 잡는 것(육안):

- 상대링크가 실존 파일을 가리킴 · 이미지 경로는 `../../img/`
- Mermaid 문법이 실제로 유효한지
- 절 안의 순서가 **개념 → 직관 → 수식 → 그림**인지 (블록도가 서사보다 먼저 오지 않았는가)
- 「선수 지식」과 `prereq`가 **실제로 가정하는 것**과 일치하는지
- 앵커로 쓴 제어 개념이 그 자리에서 한 줄 재진술됐는지
- CPU 실행 가능한 코드는 **직접 실행**(`python practice/NN_*.py --smoke`)
- GPU 필요분에 `> ⚠️ 미검증(GPU 필요)` 배지
- `jupytext --sync`로 `.py` ↔ `.ipynb` 왕복 확인, 노트북 출력 비움

실패 항목이 있으면 `pai-agent`를 해당 산출물만 대상으로 재호출합니다.

## Phase 5 — 윤문

`/humanize-korean`으로 `lesson.md` **본문만** 윤문합니다. 호출할 때 **`register: course`와 해당 모듈의 티어를 함께 전달합니다.**

```
/humanize-korean register: course tier: A   (대상: course/{주차}/{토픽}/lesson.md)
```

- `register: course`는 길이와 무관하게 **strict 5인 파이프라인을 강제**합니다. Tier A는 어차피 8,000자를 넘어 자동 승급되지만, Tier B·C 짧은 모듈도 얕은 Fast Path로는 티가 남으므로 명시 전달이 필요합니다. 시간이 더 걸리는 것이 정상입니다.
- `register: course`가 켜는 것: 산문 라인 기준 판정 · 문단 흐름 복원 허용 · plain-language 가드(문학체 어휘 드리프트 차단). 상세는 [`humanize-korean/SKILL.md`](../humanize-korean/SKILL.md) §register: course.
- 보존 대상(한 글자도 불변):
  **코드블록 · 셸 명령 · URL/arXiv ID · LaTeX 수식 · Mermaid 블록 · 수치·하이퍼파라미터 · frontmatter · 영문 약어 · 모듈 ID · 토픽 슬러그**

## Phase 6 — 보존 검증

윤문 전후를 diff로 대조합니다.

- 보존 대상이 한 글자라도 바뀌었으면 **해당 edit을 롤백**합니다
- **변경률 판정의 갈림선은 티어가 아니라 보호 구간 비율입니다** (SSOT: [`docs/course-plan.md`](../../../docs/course-plan.md) §1, 2026-08-05 개정)

  | 보호 구간 비율 | 주지표 | 정상 밴드 |
  |---|---|---|
  | **35% 이상** | 산문 라인 변경률(코드펜스·표 행·`$$` 제외) + 잔존 burden 감소율 | 라인 **20~50%** · burden **80%↓** 이상 |
  | 35% 미만 | 전체 문자 변경률 | **5~30%** |

  보호 구간 = 코드펜스 · Mermaid · LaTeX(블록+인라인) · 표 행 · frontmatter.
  집필된 lesson이 전부 35%를 넘으므로 **사실상 모든 lesson이 산문 라인 판정 대상**입니다.

  > 📏 §3.7 신설로 구조물(용어표·`📌` 박스·모듈 지도)이 늘면 보호 구간 비율이 더 올라갑니다. 35% 갈림선이 여전히 맞는지 파일럿에서 재측정하세요(§9.11).

- ⚠️ **전체 문자 변경률이 낮다는 이유만으로 재윤문을 지시하지 마세요.** lesson은 문자의 40% 이상이 코드·표·수식(전량 보존 대상)이라 전체 문자율이 구조적으로 낮게 나옵니다. W1-M2는 전체 1.48%였지만 S1 5/5·S2 41/43 해소에 산문 라인 34.2% 변경으로, 저윤문이 아니었습니다. 이 오판으로 재윤문을 돌리면 **과윤문을 유발합니다**(§9.1).
- **한글 수사 계수 대조를 포함합니다**(§9.3·§9.5) — `셋`·`다섯`·`세 줄` 같은 수사가 조용히 지워져 상호참조가 깨지는 사고가 두 번 있었습니다.
- **하류 검산기 재실행**(§9.9·§9.10) — `practice/`나 `labs/`가 lesson의 표·인용을 대조하고 있으면 윤문 후 다시 돌려 통과를 확인합니다.
- 주지표가 하한 미달이면서 S1이 남아 있을 때만 재윤문("윤문 다시")을 지시합니다
- 윤문본에 문학체 어휘 드리프트("포개다·결이 다르다·복리로·파고들다" 류)가 새로 생겼으면 **과교정**입니다. 롤백 후 plain-language 재작업을 요청합니다
- 정밀 검증이 필요하면 `content-fidelity-auditor` 에이전트를 씁니다

## Phase 7 — 마무리

1. `docs/course-plan.md`의 해당 모듈 체크박스를 `[x]`로 갱신 (5번 실행검증은 학습자 몫이므로 비워둠)
2. §0 대시보드의 완료 카운트와 "마지막 갱신" 날짜 갱신
3. lesson에서 나온 팀 질문을 `notes/questions-for-team.md`에 적립
4. **절 시작 용어표를 `notes/glossary.md`로 롤업** — 용어표는 glossary와 같은 `용어 | 정의 | 비유` 형식이므로 그대로 옮기고 `모듈` 열만 채웁니다. 이미 있는 용어는 중복 등재하지 말고, 관점이 다르면 기존 항목에 병기 (`action chunking`의 W1-M1↔W2-M1 선례)
5. 커밋: `:white_check_mark: W1-M3 diffusion-ddpm-dit 완료`

---

## 부분 재실행

| 요청 | 실행 구간 |
|---|---|
| "이 토픽 다시" | Phase 1부터 전체 |
| "practice만 보강" / "labs 보강" | Phase 3(해당 산출물만) → 4 → 7 |
| "윤문 다시" | Phase 5 → 6 |
| "최신 버전으로 갱신" | Phase 2 → 4 → 7 (`sources_checked` 갱신) |

## 주차 일괄 집필

"W2 5개 집필" 같은 요청은 모듈 단위로 **순차** 진행합니다. 앞 모듈의 내용을 뒤 모듈이 `adjacent_modules`로 참조해야 개념 중복과 누락을 피할 수 있기 때문입니다. 병렬 호출하지 마세요.

## 하지 않는 것

- 마스터플랜(`docs/physical-ai-4week-master-plan.md`) 수정 — SSOT입니다. 필요하면 제안하고 승인받습니다
- 학습 실행 로그(`docs/progress.md`) 작성 — 학습자가 실습 후 직접 씁니다
- awesome 리스트 통독 — 검색이 필요할 때만 색인으로
- 회사 내부 구현 추측 — `notes/questions-for-team.md`로
