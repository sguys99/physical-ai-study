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
| W1-M1 | `course/w1-generative-core/01-physical-ai-landscape/` | B | P0 |
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
2. `docs/course-plan.md` — §1 표준 절차, §3 공통 규약, §4 티어
3. `CLAUDE.md` — 작성 표준
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
tier: C                      # 분량: A=8~10장 / B=5~7장 / C=3~4장
master_plan_section: <해당 모듈 섹션 원문 그대로 붙여넣기>
company_stack: <마스터플랜 §2.2 표>
gap_analysis: <마스터플랜 §3 표>
latest_checks: <Phase 2에서 확인한 버전·URL·날짜>
adjacent_modules: <앞뒤 모듈의 slug와 이미 다룬 개념 목록>
```

산출물: `lesson.md` + `practice/` + `labs/`. **이 단계에서 윤문하지 않습니다.**

## Phase 4 — 정적·실행 검증

`references/authoring-checklist.md`를 항목별로 실행합니다. 요약:

- frontmatter 스키마 필드 완비 · `order`가 폴더 prefix와 일치
- 상대링크가 실존 파일을 가리킴 · 이미지 경로는 `../../img/`
- Mermaid 코드펜스 문법 유효
- CPU 실행 가능한 코드는 **직접 실행**(`python practice/NN_*.py --smoke`)
- GPU 필요분에 `> ⚠️ 미검증(GPU 필요)` 배지
- `jupytext --sync`로 `.py` ↔ `.ipynb` 왕복 확인, 노트북 출력 비움
- 시각자료 4종 이상 · 회사 스택 연결 섹션 존재 · 퀴즈 10문항 + `<details>` 정답

실패 항목이 있으면 `pai-agent`를 해당 산출물만 대상으로 재호출합니다.

## Phase 5 — 윤문

`/humanize-korean`으로 `lesson.md` **본문만** 윤문합니다.

- Tier A(8~10장)는 입력이 8,000자를 넘어 **strict 모드로 자동 승급**됩니다. 시간이 더 걸리는 것이 정상입니다.
- 보존 대상(한 글자도 불변):
  **코드블록 · 셸 명령 · URL/arXiv ID · LaTeX 수식 · Mermaid 블록 · 수치·하이퍼파라미터 · frontmatter · 영문 약어 · 모듈 ID · 토픽 슬러그**

## Phase 6 — 보존 검증

윤문 전후를 diff로 대조합니다.

- 보존 대상이 한 글자라도 바뀌었으면 **해당 edit을 롤백**합니다
- 변경률이 5% 미만(윤문 효과 없음) 또는 30% 초과(과윤문)면 경고하고 사용자 판단을 구합니다
- 정밀 검증이 필요하면 `content-fidelity-auditor` 에이전트를 씁니다

## Phase 7 — 마무리

1. `docs/course-plan.md`의 해당 모듈 체크박스를 `[x]`로 갱신 (5번 실행검증은 학습자 몫이므로 비워둠)
2. §0 대시보드의 완료 카운트와 "마지막 갱신" 날짜 갱신
3. lesson에서 나온 팀 질문을 `notes/questions-for-team.md`에, 새 용어를 `notes/glossary.md`에 적립
4. 커밋: `:white_check_mark: W1-M3 diffusion-ddpm-dit 완료`

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
