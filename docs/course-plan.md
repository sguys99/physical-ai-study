# Physical AI 온보딩 4주 — 교육자료 작성 계획

> **기준 문서**: [physical-ai-4week-master-plan.md](physical-ai-4week-master-plan.md) — 커리큘럼의 Single Source of Truth(SSOT)
> **이 문서의 역할**: 집필 현황 SSOT. 토픽 하나를 끝낼 때마다 해당 체크박스를 `[x]`로 갱신합니다.
> **학습 실행 로그**는 [progress.md](progress.md)에 따로 씁니다(무엇을 돌렸나 · 막힌 지점 · 소요시간 · GPU 비용).
> **스킬 연계**: 집필은 [`/pai-course-author`](../.claude/skills/pai-course-author/), 윤문은 [`/humanize-korean`](../.claude/skills/humanize-korean/).

---

## §0. 진행 현황 대시보드

**마지막 갱신**: 2026-08-01

| 트랙 | 완료 | 비고 |
|---|---|---|
| W1 모듈 (생성모델 코어) | 0 / 5 | — |
| W2 모듈 (정책·VLA) | 0 / 5 | — |
| W3 모듈 (WBC·RL) | 0 / 5 | — |
| W4 모듈 (월드모델·내비·통합) | 0 / 5 | — |
| 논문노트 3-pass | 0 / 6 | FSQ · SONIC · HOMIE · pi0 · DP · DualMap |
| 리포 투어 | 0 / 4 | OpenHomie · SONIC · unitree_rl_gym · DualMap |
| 용어집 (목표 100개) | 0 / 4주차 | 주차 종료 시 25개씩 |
| 캡스톤 | 0 / 1 | 스택 해설 문서 + 발표 |

> 모듈 하나의 완료 기준은 **산출물 5종이 전부 `[x]`** 인 상태입니다. 5번(실행검증)은 학습자가 클라우드 인스턴스에서 직접 확인한 뒤 체크합니다.

---

## §1. 토픽 1개 작성 표준 절차 (7단계)

절차는 여기서 한 번만 정의하고, 이후 토픽 목록은 체크박스로만 추적합니다.

| # | 단계 | 주체 | 하는 일 |
|---|---|---|---|
| 1 | **컨텍스트 로드** | 스킬 | `/pai-course-author W1-M3` 호출 → 마스터플랜의 해당 모듈 섹션 + §2(회사 스택) + §3(갭 분석) + CLAUDE.md 작성 표준을 자동 주입. **플랜 없이 일반 지식으로 쓰면 회사 스택 연결이 빠져 가치가 절반이 됩니다.** |
| 2 | **최신성 검증** | 스킬 | 버전이 빠르게 바뀌는 항목(LeRobot · mujoco_playground · MuJoCo · DualMap · Isaac Lab)만 공식 문서로 설치법·API 재확인. 출처 URL과 확인 날짜를 frontmatter `sources_checked`에 기록 |
| 3 | **초안 생성** | pai-agent | `lesson.md` + `practice/` + `labs/`를 §2 표준대로 작성. **이 단계에서 윤문하지 않습니다** |
| 4 | **정적·실행 검증** | pai-agent | frontmatter 스키마 · 상대링크 · Mermaid 파싱 확인. CPU에서 돌아가는 코드는 직접 실행. GPU가 필요한 항목은 `> ⚠️ 미검증(GPU 필요)` 배지를 달고 예상 출력만 명시 |
| 5 | **윤문** | /humanize-korean | `lesson.md` **본문만** 윤문 |
| 6 | **보존 검증** | 스킬 | 윤문 전후 diff 대조. 보존 대상이 바뀌었으면 롤백, 변경률이 5~30%를 벗어나면 경고 |
| 7 | **마무리** | — | 본 문서 체크박스 `[x]` 갱신 + 이모지 컨벤셔널 커밋 (예: `:white_check_mark: W1-M3 diffusion-ddpm-dit 완료`) |

> ⚠️ **윤문 철칙 — 아래는 한 글자도 바뀌면 안 됩니다.**
> 코드블록 · 셸 명령 · URL/arXiv ID · **LaTeX 수식** · **Mermaid 블록** · 수치·하이퍼파라미터 · frontmatter · 영문 약어 · 모듈 ID · 토픽 슬러그.
> `/humanize-korean`은 **문체·리듬·표현만** 바꿉니다. 윤문 뒤 실습 코드와 수식이 그대로인지 반드시 확인합니다.

> 📌 **Tier A lesson은 strict 경로**: `/humanize-korean`은 입력 8,000자를 넘으면 자동으로 strict(5인 파이프라인) 모드로 승급합니다. Tier A(A4 8~10장)는 여기 해당하므로 윤문에 fast보다 시간이 더 걸립니다.

> 💡 **후속 명령**: "이 토픽 다시" → 1단계부터 / "practice만 보강" → pai-agent 재호출(해당 산출물만) / "윤문 다시" → 5–6단계만.

---

## §2. 토픽별 산출물 5종 (모든 토픽 공통)

각 토픽은 `course/w<N>-<slug>/<NN>-<topic-slug>/` 아래에 다음을 갖습니다.

| # | 산출물 | 주체 | 내용 |
|---|---|---|---|
| 1 | `lesson.md` 초안 | pai-agent | 학습목표 3개+ · 완료기준 1줄 · 개념(제어·LLM 비유) · 최소 수식 유도 · 아키텍처 다이어그램 · **회사 스택 연결** · 흔한 오해 3가지 · 셀프체크 퀴즈 10문항 · 출처 · 다음 토픽 링크 |
| 2 | `practice/` | pai-agent | `NN_*.py`(jupytext percent) + 동기 생성 `.ipynb` + `requirements.txt`(버전 고정) + `README.md`(실행 순서) |
| 3 | `labs/` | pai-agent | 단계별 명령 + **성공 판정 기준**(어떤 출력이 나와야 정상인지) + 흔한 에러 3~5개와 대처법 |
| 4 | 윤문 | /humanize-korean | `lesson.md` 본문 윤문 + 보존 검증 |
| 5 | 실행검증 | **학습자** | 클라우드 인스턴스에서 `labs/` 완주 |

---

## §3. 공통 규약 (웹 배포 대비 — 지금부터 강제)

### 3.1 frontmatter 스키마 (`lesson.md` 필수)

```yaml
---
module: W1-M3                          # 마스터플랜 모듈 ID와 1:1
week: 1
order: 3                               # 폴더 숫자 prefix와 일치
title: "Diffusion 계보: DDPM → DiT"
slug: diffusion-ddpm-dit
tier: C                                # A(갭 집중) | B(혼합) | C(강점 압축)
priority: P0                           # P0 | P1 | P2
prereq: [W1-M1]
tags: [generative, diffusion, dit]
est_reading_min: 25
updated: 2026-08-01
sources_checked: 2026-08-01            # 최신성 검증 날짜
---
```

정적 사이트 빌드([P2], W4)는 이 frontmatter만 스캔하면 목차·순서·태그·선수 관계를 전부 재구성할 수 있습니다. 그래서 지금 지켜두면 나중에 재작업이 없습니다.

### 3.2 시각자료 — lesson 하나에 최소 4종

| 유형 | 도구 | 쓰는 곳 |
|---|---|---|
| **비교표** | Markdown 표 | 계보 비교, 스펙, 트레이드오프, 하이퍼파라미터, sim2real 요인×대응 — 비교는 무조건 표 |
| **흐름·계보 다이어그램** | Mermaid `flowchart` / `graph LR` | 파이프라인, 데이터 흐름, 논문 계보 트리 |
| **타이밍·주파수 다이어그램** | Mermaid `sequenceDiagram` | 제어 루프 주파수 예산, 지연과 chunking의 관계 |
| **아키텍처 블록도** | ASCII 코드블록 | **텐서 shape·차원 라벨이 필요한 곳.** Mermaid는 차원 표기가 지저분해지므로 여기만 ASCII |

- 4종 최소 구성: ① 계보·지형 ② 아키텍처 ③ **회사 스택 연결도** ④ 비교표
- Mermaid는 코드펜스로 씁니다(GitHub와 대부분의 정적 사이트 생성기가 그대로 렌더).
- 이미지 파일은 `img/{module-id}-{slug}.png`, 참조는 `../../img/`. 롤아웃 mp4·학습 커브 같은 대용량은 `artifacts/`(gitignore).

### 3.3 명명 규칙

- 주차 폴더 `w{N}-{slug}` / 토픽 폴더 `{NN}-{topic-slug}` — **숫자 prefix가 정렬 순서**
- 토픽 번호는 모듈 번호와 1:1 (`W2-M3` → `course/w2-policy-vla/03-vla-lineage/`)
- 전부 소문자 kebab-case, 공백 금지. 링크는 상대경로

### 3.4 practice 코드 규약

- **단일 소스**: `.py`가 원본(jupytext percent 포맷 `# %%` 셀 구분), `.ipynb`는 생성물.
  ```bash
  jupytext --set-formats py:percent,ipynb 01_fsq_tokenizer.py   # 최초 1회
  jupytext --sync 01_fsq_tokenizer.py                            # 이후 동기화
  ```
  둘 다 커밋하되 **노트북 출력은 비우고** 커밋합니다. 노트북은 커널 재시작 후 Run All로 완주되어야 합니다.
- **스모크 경로 우선**: 무거운 학습에는 `--smoke` 플래그(수백 스텝, 수 분)를 먼저 붙여 완주되게 하고 본 학습은 그 뒤에. GPU 시간 낭비와 입문자 좌절을 동시에 막는 장치입니다.
- **headless 고정**: `MUJOCO_GL=egl`, 뷰어를 띄우는 코드 금지. 결과는 `artifacts/{module-id}/`에 mp4/png로 저장.
- **읽기 쉬운 것 > 최적화**. 핵심 수식이 코드 어디에 대응하는지 `# eq.(3)` 형태로 주석.
- `requirements.txt`는 버전 고정. 30분 이상 걸릴 학습은 문서에 **예상 소요 시간·GPU 시간**을 명시.

### 3.5 미검증 배지

집필 환경은 CPU 로컬(WSL)입니다. GPU가 필요해 집필 시점에 실행할 수 없었던 절차에는 배지를 답니다.

```markdown
> ⚠️ **미검증(GPU 필요)** — 아래 절차는 집필 시점에 실행 검증되지 않았습니다.
> 예상 소요: A10G 기준 약 40분. 실행 후 결과를 `docs/progress.md`에 기록하고 이 배지를 제거하세요.
```

배지가 남아 있는 절차는 §2의 5번(실행검증) 체크박스와 짝을 이룹니다.

### 3.6 모르는 것은 모른다고 쓴다

회사 내부 구현(FSQ 모델의 실제 구조, SONIC 정책의 실제 입력 등)은 **추측하지 않습니다.** 대신 `notes/questions-for-team.md`에 질문으로 적립하고, lesson 본문에는 "팀 확인 필요" 표시만 남깁니다. 이 질문 목록이 캡스톤(W4-M5)의 검증 질문 리스트가 됩니다.

---

## §4. 분량 3단계 티어

총량은 비슷하게 유지하되 시간을 **갭 영역에 몰아줍니다.** 제어·ML·LLM은 기초를 생략하고 바로 심화로, 로봇 시뮬·SLAM은 기초부터.

| Tier | 분량 | 모듈 | 근거 |
|---|---|---|---|
| **A** 갭 집중 | A4 8~10장 | W1-M2, W2-M1, W2-M2, W3-M1, W3-M4, W3-M5, W4-M3 (7개) | 시뮬 실무 · 모방학습 파이프라인 · RL 학습 실무 · sim2real/미들웨어 · 3D 비전/SLAM. W3-M4(SONIC)는 회사 WBC 본체이자 "가장 정독할 논문"이라 예외적으로 A |
| **B** 혼합·회사스택 | A4 5~7장 | W1-M1, W1-M5, W2-M3, W2-M4, W2-M5, W3-M2, W3-M3, W4-M2, W4-M4 (9개) | 개념 자체는 익숙하나 **로봇 맥락 재해석 + 회사 스택 연결**이 본체 |
| **C** 강점 압축 | A4 3~4장 | W1-M3, W1-M4, W4-M1 (3개) | DDPM/FM/DiT/Dreamer는 "복습 + 로봇 관점 재해석". RSSM은 학습된 상태공간 모델이므로 제어 배경이면 즉시 이해됨 |
| — | 별도 형식 | W4-M5 캡스톤 (1개) | lesson이 아니라 산출물 템플릿 + 발표 골격 |

합계: A 7 + B 9 + C 3 + 캡스톤 1 = **20**

---

## §5. 모듈별 체크리스트

### W1 — Physical AI 지형도 + 생성모델 코어 `course/w1-generative-core/`

> 주간 목표: 분야 전체 지도를 그리고 회사 스택 위치를 안다 + 이후 모든 논문의 수학적 공통 기반을 손으로 구현한다.

- [ ] **01-physical-ai-landscape** `W1-M1` · Tier B · [P0] — 스택 5계층 · 주요 플레이어 지도 · 데이터 피라미드 / 산출물: 본인 언어의 스택 다이어그램
  - [x] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_
- [ ] **02-simulator-bootcamp** `W1-M2` · Tier A · [P0] — 클라우드 셋업 → MuJoCo 기초 → G1 로드·sin파 구동 → playground 스모크
  - [ ] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_
- [ ] **03-diffusion-ddpm-dit** `W1-M3` · Tier C · [P0] — DDPM forward/reverse · DiT(patchify, adaLN-zero) / **VLA 액션 헤드 대부분이 DiT 구조**
  - [ ] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_
- [ ] **04-flow-matching** `W1-M4` · Tier C · [P0] — CFM objective · Rectified Flow / 적은 스텝 샘플링 = 실시간 제어 가능
  - [ ] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_
- [ ] **05-latent-discrete-fsq** `W1-M5` · Tier B · [P0] ★ — LDM → VQ-VAE → **FSQ 직접 구현** / **이번 주 가장 중요한 실습, 끝까지 지킬 것**
  - [ ] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_

### W2 — 로봇 정책학습 & VLA & 액션 표현 `course/w2-policy-vla/`

> 주간 목표: 모방학습→VLA 계보를 꿰고 LeRobot 파이프라인을 1회 완주한다.

- [ ] **01-imitation-learning-act** `W2-M1` · Tier A · [P0] — BC와 compounding error · ACT의 action chunking · CVAE / HOMIE cockpit 연결
  - [ ] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_
- [ ] **02-diffusion-policy** `W2-M2` · Tier A · [P0] — multimodal 행동 분포 · receding horizon / **LeRobot PushT 학습→평가→롤아웃 완주, 끝까지 지킬 것**
  - [ ] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_
- [ ] **03-vla-lineage** `W2-M3` · Tier B · [P0] — RT-2 → OXE → OpenVLA → Octo / 이산 binning 토큰의 한계
  - [ ] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_
- [ ] **04-modern-vla-pi0-groot** `W2-M4` · Tier B · [P0] — pi0(VLM + FM action expert) · FAST · pi0.5 · GR00T N1/N1.5(System1/2, latent action)
  - [ ] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_
- [ ] **05-action-representation-fsq** `W2-M5` · Tier B · [P0] ★ — 액션 표현 스펙트럼 · LAPA · latent action / **종합 과제: "왜 FSQ 계층 구조인가" 1페이지 논증**
  - [ ] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_

### W3 — 휴머노이드 WBC & RL & Sim2Real `course/w3-wbc-rl/`

> 주간 목표: PPO→AMP→모션트래킹→SONIC 계보를 잇고, G1 보행 정책을 직접 학습시켜 sim2sim까지 완주한다. **제어 배경 최대 레버리지 구간.**

- [ ] **01-robot-rl-ppo-parallel** `W3-M1` · Tier A · [P0] — PPO 압축 · GPU 병렬 수천 환경 · **보상 설계 = 비용함수 설계** · 도메인 랜더마이제이션 / mujoco_playground G1 보행 학습
  - [ ] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_
- [ ] **02-motion-imitation-amp** `W3-M2` · Tier B · [P0] — AMP(스타일만 유도) vs 모션 트래킹(dense supervision) · AMASS 리타게팅 / 보상항 ablation
  - [ ] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_
- [ ] **03-homie-deepdive** `W3-M3` · Tier B · [P0] ★ — 상·하체 분리 설계 · upper-body pose curriculum / **OpenHomie 리포 투어**
  - [ ] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_
- [ ] **04-sonic-deepdive** `W3-M4` · Tier A · [P0] ★★ — 모션 트래킹 스케일링 · **universal token space** · kinematic planner / 산출물: HOMIE vs SONIC 비교표 + FSQ 접점 질문
  - [ ] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_
- [ ] **05-sim2real-deploy** `W3-M5` · Tier A · [P0] — sim2real 갭 5대 요인 · sim2sim 검증 문화 · ONNX→DDS 배포 경로 / **unitree_mujoco sim2sim 완주, 끝까지 지킬 것**
  - [ ] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_

### W4 — 월드모델·내비게이션·시스템 통합 `course/w4-worldmodel-nav/`

> 주간 목표: 월드모델 계보와 DualMap 내비 스택을 정리하고, 4주 학습을 회사 스택 아키텍처 문서로 통합한다.

- [ ] **01-world-model-lineage** `W4-M1` · Tier C · [P0] — World Models → DreamerV3(RSSM ≈ 학습된 상태공간 모델) → V-JEPA 2 · 활용 3축
  - [ ] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_
- [ ] **02-wfm-wam** `W4-M2` · Tier B · [P1] — Cosmos(WFM) · Genie · WAM 계열(액션을 latent frame으로 주입) / 서베이는 §3·§5만
  - [ ] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_
- [ ] **03-navigation-dualmap** `W4-M3` · Tier A · [P0] ★ — SLAM 초압축 → VLMaps·ConceptGraphs → **DualMap(concrete/abstract 이중 맵)** / DualMap 데모 실행 + 리포 투어
  - [ ] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_
- [ ] **04-system-integration** `W4-M4` · Tier B · [P0] — 주파수 예산(VLA 수 Hz / WBC 50~500Hz / 관절 kHz) · DDS·ROS2 · 온보드 vs 오프보드 · 실패 모드
  - [ ] lesson.md 초안 (pai-agent)
  - [ ] practice/ (.py + .ipynb)
  - [ ] labs/
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(학습자)_
- [ ] **05-capstone** `W4-M5` · 별도 형식 · [P0] — 스택 해설 문서 + 30분 발표 → `course/capstone/` / **끝까지 지킬 것**
  - [ ] lesson.md 초안 (캡스톤 가이드 + 발표 골격)
  - [ ] practice/ (다이어그램 소스 + 문서 템플릿)
  - [ ] labs/ (팀 검증 세션 진행 절차 + 질문 리스트)
  - [ ] 윤문 (/humanize-korean)
  - [ ] 실행검증 _(발표 리허설)_

---

## §6. 보조 트랙

### 6.1 논문 3-pass 정독 6편 → `notes/papers/`

템플릿(마스터플랜 §11.2): ①한 줄 요약 ②아키텍처 그림 ③우리 스택과의 연결 ④의문점. **④는 `notes/questions-for-team.md`에도 복사합니다.**

- [ ] **FSQ** (2309.15505) — W1-M5 / W2-M5에서 재독
- [ ] **Diffusion Policy** (2303.04137) — W2-M2
- [ ] **pi0** (2410.24164) — W2-M4
- [ ] **HOMIE** (2502.13013) — W3-M3
- [ ] **SONIC** (2511.07820) — W3-M4 · 이번 4주에서 가장 정독할 논문
- [ ] **DualMap** (README의 논문 링크) — W4-M3

나머지 논문은 2-pass까지만. 서베이 4편은 **지도이지 목적지가 아닙니다** — 필요한 섹션만 참조.

### 6.2 리포 투어 4개 → `repos/` 클론 + `notes/repo-tours/`

템플릿(마스터플랜 §11.3): 디렉토리 맵 → 진입점부터 콜스택 순 핵심 파일 5개 → 논문 수식↔코드 매핑 → 주요 하이퍼파라미터 10개 → 최소 실행 가이드 → 통합 시 손댈 인터페이스. **파일 경로:라인을 인용할 것.**

- [ ] **OpenHomie** — W3-M3 · 학습 config, 보상 함수, 배포 코드 구조
- [ ] **GR00T-WholeBodyControl** — W3-M4 · SONIC 공개 코드
- [ ] **unitree_rl_gym** — W3-M5 · train→play→sim2sim→real 흐름 (deploy 경로를 1페이지 문서로)
- [ ] **DualMap** — W4-M3 · 맵 자료구조와 질의 흐름

### 6.3 용어집 → `notes/glossary.md`

매일 5개씩, 주차 종료 시 25개 누적. 4주 뒤 100개 내외가 팀 대화 해독기가 됩니다. 용어는 마스터플랜 §3 용어 사전과 일치시킵니다.

- [ ] W1 종료 시 25개
- [ ] W2 종료 시 50개
- [ ] W3 종료 시 75개
- [ ] W4 종료 시 100개

### 6.4 팀 질문 리스트 → `notes/questions-for-team.md`

**온보딩의 핵심 산출물.** 모듈마다 최소 1개 적립. 특히 아래는 반드시 확인합니다.

- 팀 표준 학습 환경(시뮬레이터·클러스터·도커 이미지) — **W1 중에 확인.** Isaac 기반이면 그대로 물려받는 것이 최선
- 회사 파이프라인에서 SONIC 정책의 실제 입력은 무엇인가 (FSQ 토큰과의 접점)
- FSQ 모델의 실제 구조·레벨 설정·학습 데이터
- 마스터플랜 §2.3의 추정 파이프라인 다이어그램 검증

---

## §7. 진행 순서 — JIT 인터리빙 캘린더

집필은 **그 주가 시작되기 전에** 끝내고, 학습 중 나온 실측값(에러·소요시간·GPU 비용)을 다음 주 자료에 반영합니다.

| 시점 | 집필 | 학습 |
|---|---|---|
| **Week 0** (1~2일) | 파이프라인 셋업 + **W1 5개 집필** | 클라우드 인스턴스 확보, 부록 A 셋업 |
| **Week 1** | W1 피드백 패치 → **W2 5개 집필** | W1 학습 · 실행검증 |
| **Week 2** | **W3 5개 집필** | W2 학습 · 실행검증 (LeRobot 완주) |
| **Week 3** | **W4 5개 집필** | W3 학습 · 실행검증 (보행 학습 + sim2sim) |
| **Week 4** | 캡스톤 템플릿, [P2] `site/` 빌드 | W4 학습 + 캡스톤 |

- 집필은 각 주차 **직전 주말 또는 전날 3~5시간**에 배치합니다.
- 학습 중 발견한 오류는 즉시 패치하되 체크박스는 유지합니다(재집필이 아니므로).
- 🚧 **데드락 방지선**: 주차 시작 시점에 그 주 모듈이 **최소 3개** 확보돼 있지 않으면 학습을 시작하지 않습니다. 집필이 밀리면 학습도 멈춥니다.

---

## §8. 리스크와 컷 가이드

- **집필 지연** → 마스터플랜 §12 컷 순서 준용: ① W4-M2 서베이 정독 → 목차+§5만 ② W2-M3 OpenVLA 실행 → 코드 리딩 ③ W1-M3 DDPM 구현 → HF Annotated Diffusion 읽기 ④ Isaac Lab 전면 컷.
  **끝까지 지킬 것: FSQ 구현(W1-M5) · LeRobot 완주(W2-M2) · G1 보행 학습+sim2sim(W3) · 캡스톤 문서(W4-M5).**
- **시뮬 셋업 지연** → W1-M2에 1.5일까지 허용. 원칙: **환경이 안 돌아가는 상태로 다음 주로 넘어가지 않는다.** 셋업이 밀리면 그 주 [P2] 이론부터 컷.
- **집필 시점 GPU 미검증** → `> ⚠️ 미검증(GPU 필요)` 배지로 명시. 학습자가 실행검증 후 배지 제거.
- **자료 최신성** → LeRobot·mujoco_playground는 버전이 빠르게 움직입니다. 절차 2단계에서 공식 문서 확인 + `requirements.txt` 버전 고정 + frontmatter `sources_checked` 날짜 기록.
- **범위 폭주** → awesome 리스트 통독 금지. 마스터플랜의 핵심 논문 목록을 벗어나 자료를 무한 확장하지 않습니다.
- **회사 내부 구현 추측** → 금지. `notes/questions-for-team.md`로.

---

## §9. 진행 메모

- 토픽 집필은 [`/pai-course-author <모듈ID>`](../.claude/skills/pai-course-author/) 호출을 권장합니다(§1 표준 절차 참고). 작성·윤문 후 이 파일의 체크박스를 `[x]`로 갱신하고 커밋합니다.
- 모듈 ID·토픽 슬러그·논문 arXiv ID는 [physical-ai-4week-master-plan.md](physical-ai-4week-master-plan.md)와 일치시킵니다(SSOT). 임의 변경 금지.
- 마스터플랜 수정이 필요해 보이면 먼저 제안하고 승인받습니다.
- 정적 사이트 빌드(`site/`)는 [P2]이며 W4에 검토합니다. §3.1 frontmatter 규약을 지키면 스캔만으로 빌드됩니다.
