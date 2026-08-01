# Physical AI 4주 스터디 마스터 플랜 (v1.1)

> **대상**: Control Engineering 10년 + ML/DL/LLM/Agent 실무 경력, Physical AI 무경험 연구자·Application 엔지니어
> **목표**: 4주 안에 회사 기술 스택(Unitree G1, HOMIE, GEAR-SONIC, DualMap, FSQ 기반 계층 모델)의 논문·코드를 스스로 읽고, 팀 논의에 참여하며, 시뮬레이션 실습을 완주할 수 있는 "실무 기본 체력" 확보
> **작성일**: 2026-08-01

---

## 0. 이 문서 사용법

이 문서는 그 자체가 학습자료가 아니라 **학습자료 생성을 위한 마스터 플랜**입니다. 다음 흐름으로 사용하세요.

1. 각 모듈에는 고유 ID가 있습니다 (예: `W1-M2`). 상세 학습자료가 필요할 때, §11의 프롬프트 템플릿에 해당 모듈 섹션을 그대로 붙여넣어 Claude에게 요청하세요.
2. 코드 분석(OpenHomie, DualMap, unitree_rl_gym 등)은 Claude Code로 리포를 클론한 뒤 §11.3 템플릿으로 "코드베이스 투어"를 요청하세요.
3. 각 주차 마지막의 **체크포인트 질문**에 백지 상태로 답해보고, 막히면 해당 모듈만 다시 학습자료를 요청하세요.
4. 우선순위 태그: **[P0]** 반드시 / **[P1]** 가능하면 / **[P2]** 여유 시. 시간이 부족하면 §12의 컷 가이드를 따르세요.

---

## 1. 확인된 전제 (v1.1 반영 완료)

| 항목 | 확인된 조건 | 플랜 반영 |
|---|---|---|
| 학습 시간 | 풀타임 온보딩 (주 30시간 이상) ✅ | 주 5일 × 6~8시간 구조 유지 |
| 하드웨어 | 로컬 GPU 없음, **클라우드 GPU만 사용** | 실습 주 경로를 MuJoCo + mujoco_playground(전부 pip 설치, headless 친화)로 재설계. Isaac Lab은 [P1] 선택 과제로 강등. **부록 A(클라우드 가이드) 신설** |
| 시뮬레이터 경험 | **전무** | W1-M2를 "시뮬레이터 부트캠프"로 확장(필요시 1.5일). 원칙: 환경이 안 돌아가는 상태로 다음 주로 넘어가지 않는다 |
| 실기 로봇 | G1 실기 접근 불가 | 4주간 시뮬레이션 100%. 실기 배포 경로는 코드 리딩+문서화로 습득 |
| 언어 | 영어 논문 독해 가능 (가정 유지) | — |
| 담당 영역 | **전 영역 가능성 (VLA·WBC·내비·통합 모두)** | 풀스택 커버 유지가 정답. 통합 관점의 캡스톤 (A) "스택 해설 문서+발표"를 필수 산출물로 격상 |

---

## 2. 배경: Physical AI 지형도와 회사 스택 매핑

### 2.1 Physical AI 스택 5계층

Physical AI는 "인터넷 데이터로 학습한 지능을 물리 세계의 몸에 넣는 문제"이고, 현재 업계 표준 스택은 대략 5계층입니다.

```
[L5] 인지·매핑        : 카메라/LiDAR → 시맨틱 맵, 언어 grounding     ← DualMap
[L4] 상위 지능        : VLA / 월드모델 / 플래너 (수 Hz)              ← VLA, WFM/WAM, GR00T
[L3] 액션 인터페이스   : 연속 액션 or 이산 토큰 (chunking)            ← FSQ ★회사 핵심
[L2] 전신 제어 (WBC)  : RL 정책, 모션 트래킹 (50~500 Hz)             ← SONIC ★회사 사용
[L1] 하드웨어/미들웨어 : 관절 모터, DDS/SDK, 센서                     ← Unitree G1
      + 데이터 수집    : 텔레옵, 모캡, 휴먼 비디오                    ← HOMIE
```

핵심 통찰: **상위(L4)는 느리고 똑똑하게, 하위(L2)는 빠르고 반사적으로** 움직이며, 둘을 잇는 인터페이스(L3)가 시스템 설계의 승부처입니다. 회사가 FSQ 이산 토큰을 쓰는 것도, SONIC이 "universal token space"로 VLA·텔레옵·플래너를 하나의 정책에 연결하는 것도 모두 L3 설계 문제입니다.

### 2.2 회사 스택 한 줄 정리

| 기술 | 계층 | 정체 | 핵심 논문/코드 |
|---|---|---|---|
| **Unitree G1** | L1 | 23~43 DoF 휴머노이드. Livox Mid-360 LiDAR + 뎁스카메라, DDS 기반 SDK | support.unitree.com/home/en/G1_developer, github.com/unitreerobotics |
| **HOMIE (OpenHomie)** | L2+데이터 | 하체는 RL 보행 정책, 상체는 외골격 cockpit 텔레옵으로 분리한 loco-manipulation 시스템. 데이터 수집 파이프라인 역할 | arXiv:2502.13013, github.com/InternRobotics/OpenHomie |
| **GEAR-SONIC (SONIC)** | L2~L3 | NVIDIA GEAR의 WBC 파운데이션 정책. 모션 트래킹을 스케일업(42M 파라미터, 700시간 모캡, 21k GPU-hours)해서 하나의 통합 정책으로 VLA(GR00T N1.5)·VR 텔레옵·게임패드 kinematic planner(내비게이션·스타일 보행)를 모두 구동. "universal token space"가 핵심 | arXiv:2511.07820, github.com/NVlabs/GR00T-WholeBodyControl |
| **DualMap** | L5 | 온라인 open-vocabulary 시맨틱 매핑. 동적으로 변하는 환경에서 자연어 목표("빨간 컵 있는 곳으로")로 내비게이션 | github.com/Eku127/DualMap |
| **FSQ 기반 모델** | L3 | 상위 모델과 하위 제어기를 잇는 이산 액션 토큰화 (VQ-VAE의 개선판, codebook collapse 없음) | arXiv:2309.15505 |

### 2.3 추정 전체 파이프라인 (팀에 검증받을 것)

```
센서(RGB-D, LiDAR)
  → DualMap (시맨틱 맵 + 언어 목표 grounding)
  → VLA/상위 플래너 (목표 → 행동 의도)
  → FSQ 이산 액션 토큰
  → SONIC 스타일 WBC 정책 (모션 트래킹)
  → G1 관절 명령 (unitree_sdk2 / DDS)

[병렬] HOMIE 텔레옵 → 시연 데이터 수집 → VLA/정책 학습
[연구] World Model (WFM/WAM) → 데이터 증강, 정책 평가, 상상 기반 플래닝
```

**Week 4 캡스톤 과제 중 하나가 이 다이어그램을 팀원에게 검증받고 수정하는 것입니다.**

---

## 3. 학습 전략: 내 역량과의 갭 분석

| 보유 역량 | Physical AI에서의 레버리지 |
|---|---|
| Control 10년 | 동역학·상태공간·MPC 직관 → W3(WBC/RL/sim2real)를 남들보다 2배 빠르게. 보상 설계 = 비용함수 설계 |
| ML/DL | W1 생성모델 파트는 "복습 + 로봇 관점 재해석"으로 압축 가능 |
| LLM/Agent/RAG | VLA의 VLM 백본, 토크나이저, 시스템1/2 아키텍처가 즉시 익숙함. FSQ = "액션의 BPE"라는 비유가 통함 |

| 갭 (이번 4주의 실제 타깃) | 대응 |
|---|---|
| 로봇 시뮬레이터 실무 (MuJoCo 중심, Isaac은 개요만) | W1-M2 부트캠프 + W3 집중 실습 |
| RL 학습 실무 (PPO 병렬 학습 돌려보기) | W3 |
| 모방학습/VLA 학습 파이프라인 | W2 (LeRobot 완주) |
| Sim2Real, 로봇 미들웨어(DDS/ROS2) | W3 Day5, W4 Day4 |
| 3D 비전/SLAM/open-vocab 매핑 | W4 |

**용어 사전 미리보기** (첫 주에 자주 만날 약어):
VLA = Vision-Language-Action 모델 / WBC = Whole-Body Control / **WFM** = World Foundation Model (NVIDIA Cosmos처럼 범용 비디오 기반 월드모델) / **WAM** = World-Action Model (비디오 월드모델 백본에 액션을 latent frame으로 주입해 예측과 행동을 공동 생성하는 계열; Cosmos Policy, UWA, Fast-WAM 등) / IDM = Inverse Dynamics Model / OXE = Open X-Embodiment / teleop = 원격조작 / retargeting = 인간 모션→로봇 관절 매핑 / sim2sim = 학습 시뮬레이터(Isaac)→검증 시뮬레이터(MuJoCo) 이식.

---

## 4. 4주 로드맵 한눈에 보기

| 주차 | 주제 | 실습 산출물 | 회사 스택 연결 |
|---|---|---|---|
| **W1** | Physical AI 지형도 + 생성모델 코어 (Diffusion/FM/DiT/VQ-VAE/FSQ) | 2D toy Flow Matching 구현, FSQ 토크나이저 구현, 시뮬 환경 셋업 | FSQ ★ |
| **W2** | 로봇 정책학습 (ACT/DP) + VLA (pi0/OpenVLA/GR00T) + 액션 표현 | LeRobot으로 PushT에 Diffusion Policy 학습 완주 | VLA 백본, FSQ 계층 설계 |
| **W3** | 휴머노이드 WBC & RL (AMP→모션트래킹→SONIC, HOMIE) + Sim2Real | mujoco_playground로 G1 보행 정책 학습 → 표준 MuJoCo sim2sim | SONIC ★, HOMIE ★, G1 ★ |
| **W4** | 월드모델/WFM/WAM + 내비게이션(DualMap) + 시스템 통합 캡스톤 | DualMap 데모 실행, 회사 스택 아키텍처 문서 + 발표 | DualMap ★, 전체 통합 |

매일 리듬(권장): 오전 = 논문·개념 (Claude로 생성한 학습자료 + 원논문), 오후 = 실습/코드, 마감 30분 = 노트 정리 + 용어집 갱신.

---

## 5. Week 1 — Physical AI 지형도 + 생성모델 코어

**주간 목표**: (a) 분야 전체 지도를 그리고 회사 스택이 어디에 있는지 안다. (b) 이후 모든 논문의 수학적 공통 기반인 Diffusion/Flow Matching/이산 토큰화를 손으로 구현해본다.

### W1-M1. Physical AI 개요와 산업 지형 [P0] — Day 1

- 다룰 내용: Physical AI 정의와 "왜 지금인가"(파운데이션 모델 + 대규모 시뮬 + 저가 휴머노이드), 스택 5계층(§2.1), 주요 플레이어 지도 — NVIDIA(GR00T/Cosmos/Isaac), Physical Intelligence(pi0), Google DeepMind(Gemini Robotics/RT-2), Figure/Tesla/1X, 중국 생태계(Unitree, InternRobotics/Shanghai AI Lab, AgiBot). 데이터 피라미드(웹 비디오 → 휴먼 비디오 → 텔레옵 시연 → 실기 RL).
- 핵심 자료: 「A Tutorial on World Models and Physical AI」(arXiv:2606.12783 — 입문 튜토리얼), keon/awesome-physical-ai (지형 훑기용), GEAR-SONIC 프로젝트 페이지 데모 영상(nvlabs.github.io/GEAR-SONIC — 회사가 지향하는 최종 그림을 눈으로 확인).
- 산출물: 본인 언어로 그린 스택 다이어그램 1장 + "우리 회사 스택이 이 지도 어디에 있는가" 메모.

### W1-M2. 시뮬레이터 부트캠프 + 클라우드 실습 환경 구축 [P0] — Day 2 (시뮬 첫 경험이므로 필요시 Day 3 오전까지 허용)

- 다룰 내용: G1 스펙 훑기(DoF, 액추에이터, 센서, 온보드 컴퓨트), 관절공간 vs 태스크공간(제어 배경이면 30분 복습), 시뮬레이터 지형 — **MuJoCo(이번 4주의 주력: 가볍고 정확, pip 설치로 끝, headless 클라우드 친화)**, **mujoco_playground(MJX 기반 GPU 병렬 RL 스위트, G1 보행 태스크 내장 — W3의 주력, 태스크명은 리포 locomotion 목록에서 확인)**, Genesis(대안), Isaac Sim/Lab(업계 표준이지만 RTX 렌더링 요구로 클라우드 셋업 난도가 높음 → 이번 4주는 개념만, 실행은 [P1]).
- 실습 (시뮬 경험 전무 전제, 순서 엄수):
  1. 클라우드 인스턴스 셋업 — 부록 A 절차대로. `MUJOCO_GL=egl` headless 렌더링으로 mp4가 저장되는 것까지 확인.
  2. MuJoCo 기초: 공식 튜토리얼 노트북(mujoco.readthedocs.io) 1개 완주 — MJCF 구조(body/joint/actuator), step 루프, 렌더링.
  3. MuJoCo Menagerie에서 G1 모델 로드 → 관절 인덱스 확인, sin파 명령으로 팔다리를 흔들어 영상 저장 (첫 "내가 로봇을 움직였다" 경험).
  4. mujoco_playground 설치 → G1 locomotion 예제가 에러 없이 "돌아가는 것"까지만 확인 (본 학습은 W3에서).
- 핵심 자료: Unitree G1 developer docs, mujoco.readthedocs.io(공식 튜토리얼·Colab), github.com/google-deepmind/mujoco_menagerie, github.com/google-deepmind/mujoco_playground, github.com/unitreerobotics/unitree_mujoco.
- 보충: Modern Robotics(Lynch & Park) 2~3장 — 표기법 통일용 참조서. Underactuated Robotics(underactuated.mit.edu) — 제어 배경자에게 최고의 보행 동역학 교재.

### W1-M3. Diffusion 계보: DDPM → DiT [P0] — Day 3

- 다룰 내용: DDPM의 forward/reverse 과정과 노이즈 예측 손실(수식 유도 1회), 샘플러 직관, DiT 아키텍처(patchify, adaLN-zero, 왜 U-Net을 Transformer로 대체했는가). **로봇 관점 포인트: 현재 대부분의 VLA 액션 헤드가 DiT 구조.**
- 핵심 논문: DDPM(2006.11239), DiT(2212.09748).
- 보충: Lilian Weng "What are Diffusion Models?"(lilianweng.github.io), HuggingFace Annotated Diffusion(huggingface.co/blog/annotated-diffusion — 코드로 읽는 DDPM).
- 실습: MNIST 또는 2D toy 데이터에 미니 DDPM 학습 (노트북 1개, 2~3시간 컷).

### W1-M4. Flow Matching & Rectified Flow [P0] — Day 4

- 다룰 내용: Conditional Flow Matching objective 유도, 확률 경로와 벡터장, Rectified Flow(노이즈↔데이터 직선 연결)가 실전 표준인 이유, **왜 로봇 정책에서 Diffusion 대신 FM인가(적은 스텝 샘플링 = 실시간 제어 가능, 연속 액션에 자연스러움). pi0·GR00T·DiT4DiT의 objective가 전부 이것.**
- 핵심 논문: Flow Matching(2210.02747), Rectified Flow(2209.03003).
- 보충: MIT 6.S184 "Introduction to Flow Matching and Diffusion Models"(diffusion.csail.mit.edu) — 강의노트+실습 노트북이 최고 품질. 이 과목 노트북 하나를 그대로 실습으로 써도 됨.
- 실습: 2D toy 분포(two moons 등)에 Flow Matching 직접 구현, DDPM과 샘플링 스텝 수 비교.

### W1-M5. 잠재공간과 이산화: VAE/LDM → VQ-VAE → FSQ ★ [P0] — Day 5

- 다룰 내용: LDM(픽셀이 아니라 VAE 잠재공간에서 확산 — 비디오 월드모델 백본의 기반), VQ-VAE(codebook, straight-through estimator, codebook collapse 문제), **FSQ(각 차원을 고정 레벨로 반올림 — codebook·commitment loss 없이 collapse 원천 차단). 회사가 상위 모델↔하위 제어기를 FSQ 토큰으로 잇는 이유를 여기서 이해한다: 이산 토큰은 LLM식 autoregressive 예측과 궁합이 좋고, 인터페이스가 단순하며, 다양한 하위 정책을 갈아끼울 수 있다.**
- 핵심 논문: Latent Diffusion(2112.10752), VQ-VAE(1711.00937), FSQ(2309.15505).
- 실습: FSQ는 수십 줄이면 구현됨 — 직접 구현 후 이미지(또는 관절 궤적 시계열)를 토큰화/복원, VQ-VAE와 코드북 사용률 비교. **이번 주 가장 중요한 실습.**

### W1 체크포인트 (백지 답변)

1. DiT 블록 다이어그램을 그리고 conditioning이 들어가는 위치를 설명하라.
2. DDPM과 Flow Matching의 학습 objective 차이를 수식 수준으로 설명하라. 로봇 제어에서 FM이 선호되는 실용적 이유는?
3. FSQ가 VQ-VAE 대비 갖는 장점 2가지와, "상위 모델-하위 제어기 인터페이스"로 이산 토큰을 쓰는 시스템적 이점을 설명하라.

---

## 6. Week 2 — 로봇 정책학습 & VLA & 액션 표현

**주간 목표**: 모방학습→VLA로 이어지는 계보를 꿰고, LeRobot으로 학습 파이프라인을 1회 완주하며, 회사 FSQ 모델이 이 지형 어디에 있는지 파악한다.

### W2-M1. 모방학습 기초 + ACT [P0] — Day 1

- 다룰 내용: Behavior Cloning과 compounding error, **ACT — action chunking(한 번에 k스텝 예측)의 원전, CVAE 구조, 왜 chunking이 사실상 모든 후속 논문의 기본값이 됐는가**, ALOHA식 텔레옵 데이터 수집 → HOMIE cockpit과의 연결.
- 핵심 논문: ACT(2304.13705).
- 실습: LeRobot(github.com/huggingface/lerobot) 설치, 예제 데이터셋 시각화, ACT 학습 스크립트 실행 시작.

### W2-M2. Diffusion Policy [P0] — Day 2

- 다룰 내용: visuomotor 정책을 조건부 생성 문제로 재정의, multimodal 행동 분포(같은 상황에서 왼쪽/오른쪽 다 정답)를 회귀가 못 다루는 이유, receding horizon 실행. **거의 모든 논문의 baseline이므로 실험 설정(관측 horizon, action horizon 등)까지 체화할 것.**
- 핵심 논문: Diffusion Policy(2303.04137).
- 실습: **LeRobot으로 PushT 태스크에 Diffusion Policy 학습→평가→롤아웃 영상 확인 (이번 주 핵심 실습, 파이프라인 완주가 목표).** 클라우드에서는 뷰어 대신 평가 롤아웃을 mp4로 저장해 확인 (T4/A10G급이면 충분).

### W2-M3. VLA 계보: RT-2 → OXE → OpenVLA [P0] — Day 3

- 다룰 내용: RT-2(VLM에 액션 토큰 출력 추가 — VLA의 탄생), Open X-Embodiment(크로스 임바디먼트 데이터 통합), OpenVLA(7B 오픈소스, 이산 binning 토큰의 한계), Octo(제너럴리스트 정책).
- 핵심 논문: RT-2(2307.15818), OXE(2310.08864), OpenVLA(2406.09246).
- 보충: VLA 전체 지도가 필요하면 서베이 「An Anatomy of VLA Models」(2512.11362 — 팀 제공, 모듈→마일스톤→5대 챌린지 구조라 이 시점에 읽으면 잘 읽힘).
- 실습: OpenVLA 추론 데모 실행(GPU 여유 시), 또는 코드에서 액션 토크나이저 부분만 읽기.

### W2-M4. 현대 VLA 표준형: pi0 → pi0.5 → GR00T N1 [P0] — Day 4

- 다룰 내용: **pi0(VLM 백본 + Flow Matching action expert = 현재 표준형)**, pi0-FAST(DCT 기반 액션 토큰화로 autoregressive VLA 고속화), pi0.5(open-world 계층 추론), **GR00T N1/N1.5(System1/System2, latent action — SONIC과 직접 연결되는 NVIDIA 스택이므로 회사 관점에서 중요)**.
- 핵심 논문: pi0(2410.24164), FAST(2501.09747), GR00T N1(2503.14734).
- 보충: Physical Intelligence 블로그(physicalintelligence.company), github.com/NVIDIA/Isaac-GR00T.
- 실습: pi0 아키텍처 블록도 손으로 그리기, LeRobot의 pi0 구현 코드 리딩(Claude Code 활용 추천).

### W2-M5. 액션 표현 심화: LAPA·latent action·FSQ 계층 설계 ★ [P0] — Day 5

- 다룰 내용: 액션 표현 스펙트럼 정리 — 연속 회귀 / 이산 binning / FAST(DCT) / VQ·FSQ 토큰 / latent action. LAPA(액션 라벨 없는 휴먼 비디오에서 latent action 학습 — 데이터 병목의 돌파구), GR00T·DreamGen 계열의 latent action 활용. **종합 과제: "우리 회사는 왜 FSQ 기반 계층 구조를 선택했는가"를 1페이지로 논증(장점: 인터페이스 단순성·하위 정책 교체 가능·AR 모델 궁합 / 트레이드오프: 양자화 오차·고주파 제어 표현력). 이 문서를 팀원과의 첫 기술 토론 소재로 쓴다.**
- 핵심 논문: LAPA(2410.11758), FSQ 재독(2309.15505).
- 체크포인트 질문에 대비해 SONIC의 "universal token space" 부분(2511.07820 §해당)을 미리 훑어두면 W3가 편해짐.

### W2 체크포인트

1. action chunking이 필요한 이유를 compounding error와 지연(latency) 관점에서 설명하라.
2. pi0 아키텍처를 그리고, 이산 토큰 VLA(RT-2/OpenVLA) 대비 장단점을 말하라.
3. 이산 액션 토큰(FSQ) vs 연속 액션(FM head)의 트레이드오프를 3가지 축(정밀도, 추론속도, 시스템 결합성)에서 비교하라.
4. LeRobot 학습 파이프라인의 구성요소(데이터셋 포맷→학습→평가)를 설명할 수 있는가?

---

## 7. Week 3 — 휴머노이드 전신제어(WBC) & RL & Sim2Real ★제어 배경 최대 레버리지 구간

**주간 목표**: RL 기반 보행/전신제어의 계보(PPO→AMP→모션트래킹→SONIC)를 잇고, 직접 G1 보행 정책을 학습시켜 sim2sim까지 완주하며, 회사가 쓰는 SONIC·HOMIE를 깊게 읽는다.

### W3-M1. 로봇 RL 기초: PPO + 대량 병렬 시뮬 [P0] — Day 1

- 다룰 내용: PPO 핵심(제어 배경이면 반나절), 로봇 RL 성공 레시피 — GPU 병렬 수천 환경("Learning to Walk in Minutes"), 보상 설계(속도 추종 + 자세 + 에너지 + 접촉 페널티 = 사실상 비용함수 설계), 도메인 랜더마이제이션, actuator 모델.
- 핵심 논문: Learning to Walk in Minutes(2109.11978), 도메인 랜더마이제이션 원전(1703.06907).
- 보충: CS285(Berkeley Deep RL — PPO 강의만 선별 시청), Hwangbo et al. Science Robotics 2019(ANYmal — 학습 기반 보행의 고전), leggedrobotics/legged_gym + rsl_rl 코드.
- 실습: **mujoco_playground(MJX)로 G1 joystick/velocity-tracking 보행 정책 학습** — 단일 클라우드 GPU에서 수십 분~수 시간이면 학습되므로 시뮬 입문 + 클라우드 환경에 최적. 보상 커브·episode length를 W&B/TensorBoard로 원격 기록. [P1] 여유가 되고 회사 학습 인프라가 Isaac 기반으로 확인되면 Isaac Lab 경로(unitree_rl_gym)를 팀 표준 환경을 물려받아 병행 시도하되, 반나절 이상 막히면 즉시 playground로 복귀(§12).

### W3-M2. 모션 이미테이션: AMP → 모션 트래킹 [P0] — Day 2

- 다룰 내용: 보상만으로 배운 보행이 "로봇스러운" 이유 → 인간 모션 prior 주입의 두 갈래: **AMP(adversarial 판별기로 스타일만 유도)** vs **모션 트래킹(DeepMimic 계보 — 레퍼런스 모션을 직접 추종, dense supervision이라 스케일이 잘 됨)**. 모캡 데이터(AMASS 등)와 리타게팅(인간 골격→G1 관절) 파이프라인.
- 핵심 논문: AMP(2104.02180), DeepMimic(1804.02717).
- 실습: 학습 중인 보행 정책 커브 분석, 보상 항 하나 바꿔 재학습(ablation 감각 익히기).

### W3-M3. HOMIE 딥다이브 ★ [P0] — Day 3

- 다룰 내용: HOMIE의 분리 설계 — 하체는 RL 정책(상체 움직임 외란을 랜더마이제이션으로 흡수), 상체는 외골격 cockpit 텔레옵. upper-body pose curriculum, height tracking. **왜 이 구조가 데이터 수집에 유리한가 → 회사의 시연 데이터 파이프라인 관점에서 해석.**
- 핵심 자료: HOMIE(2502.13013), github.com/InternRobotics/OpenHomie.
- 실습: **Claude Code로 OpenHomie 리포 투어**(§11.3 템플릿) — 학습 config, 보상 함수, 배포 코드 구조 파악.

### W3-M4. SONIC / GEAR-SONIC 딥다이브 ★★ [P0] — Day 4

- 다룰 내용(회사 WBC 모델이므로 이번 4주에서 가장 정독할 논문): **모션 트래킹의 스케일링(1.2M→42M 파라미터, 700h 모캡, 21k GPU-hours)이 자연스럽고 강건한 범용 WBC를 만든다**는 주장, universal token space(로봇 모션·휴먼 모션·하이브리드를 하나의 잠재 표현으로 → VR 텔레옵·VLA·게임패드가 같은 정책 공유), real-time kinematic planner(트래킹→내비게이션 브리지: 스타일 보행, 스쿼트, 크롤링), GR00T N1.5 연결 데모(자율 loco-manipulation).
- 핵심 자료: SONIC(2511.07820), nvlabs.github.io/GEAR-SONIC, github.com/NVlabs/GR00T-WholeBodyControl (문서: nvlabs.github.io/GR00T-WholeBodyControl).
- 산출물: HOMIE vs SONIC 비교표(입력 인터페이스, 학습 데이터, 스케일, 배포 형태) + "우리 회사에서 SONIC 정책의 입력은 무엇이 되는가(FSQ 토큰과의 접점)" 질문 리스트 → 팀에 확인.

### W3-M5. Sim2Real & 배포 파이프라인 [P0] — Day 5

- 다룰 내용: sim2real 갭의 5대 요인(액추에이터 동특성, 지연, 접촉 모델, 센서 노이즈, 상태 추정), 대응책(도메인 랜더마이제이션, actuator net, 관측 이력 스택), **sim2sim 검증 문화(Isaac에서 학습 → MuJoCo에서 검증 → 실기)**, unitree_rl_gym의 deploy 경로(정책 ONNX → unitree_sdk2/DDS → 실기), 안전 절차(리모컨 킬스위치, 감쇠 모드).
- 핵심 자료: github.com/unitreerobotics/unitree_rl_gym (train→play→sim2sim→real 흐름이 문서화된 최적의 레퍼런스), unitree_sdk2 / unitree_sdk2_python.
- 실습: **학습한 정책을 학습 환경(MJX) 밖의 표준 MuJoCo(unitree_mujoco)로 이식해 동작을 검증(sim2sim)하고 롤아웃 mp4 저장 — 이번 주 최종 산출물.** unitree_rl_gym의 deploy 코드는 실행 대신 Claude Code로 리딩해 실기 배포 경로(정책 export → SDK/DDS 명령)를 1페이지 문서로 정리 (실기 접근 불가 조건의 보완책).

### W3 체크포인트

1. AMP와 모션 트래킹의 차이를 supervision 신호 관점에서 설명하고, SONIC이 트래킹을 선택한 이유(스케일링 관점)를 말하라.
2. HOMIE가 상·하체를 분리한 이유와 그 대가(전신 협조 동작의 한계)를 설명하라.
3. sim2real 갭 요인 5가지와 각각의 대표 대응책을 말하라.
4. 내가 학습시킨 보행 정책의 관측 벡터·액션 벡터·보상 항을 나열할 수 있는가?

---

## 8. Week 4 — 월드모델·WFM·WAM + 내비게이션 + 시스템 통합 캡스톤

**주간 목표**: 월드모델 계보와 최신 WFM/WAM 동향을 정리하고, DualMap 중심의 내비게이션 스택을 이해한 뒤, 4주 학습을 "회사 스택 아키텍처 문서 + 발표"로 통합한다.

### W4-M1. 월드모델 계보: World Models → Dreamer → V-JEPA 2 [P0] — Day 1

- 다룰 내용: 월드모델 = "행동 조건부 미래 예측기" 공통 정의 p(x_{t+1:t+H} | x_t, a, l), Ha & Schmidhuber(개념 원점), DreamerV3(RSSM — 잠재에서 상상해 정책 학습, 제어 배경이면 상태공간 모델로 즉시 이해됨), V-JEPA 2(픽셀 생성 없이 표현공간 예측 — 효율 노선), 세 가지 활용 축: ①정책 결합 ②학습된 시뮬레이터(RL/평가) ③데이터 생성.
- 핵심 논문: World Models(1803.10122), DreamerV3(2301.04104), V-JEPA 2(2506.09985).

### W4-M2. WFM/WAM: 비디오 백본 기반 월드모델과 액션 결합 [P1] — Day 2

- 다룰 내용: 비디오 생성 백본(LDM 기반)을 로봇 월드모델로 전용하는 흐름, **NVIDIA Cosmos(WFM의 대표)**, Genie 계열(latent action으로 상호작용), **WAM 계열 — 액션을 latent frame으로 주입해 비디오·액션을 공동 생성(Cosmos Policy, UWA, Fast-WAM 등; W1의 LDM이 왜 기반인지 여기서 회수됨)**, 월드모델을 시뮬레이터로 쓰는 RL/평가(WMPO, WorldEval 계열), 물리 일관성 한계와 벤치마크.
- 핵심 자료: **「World Model for Robot Learning: A Comprehensive Survey」(2605.00080 — 팀 제공. 이 분야 지도 그리기에 이 한 편이면 충분, 전부 읽지 말고 §3·§5 중심으로)**, Cosmos(2501.03575), Genie(2402.15391).
- 보충: 「A Comprehensive Survey on World Models for Embodied AI」(2510.16732 — 분류축이 다른 보조 서베이), 「Physics Cognition in Video Generation」(2503.21765 — 물리 일관성 관점 [P2]), Unitree 자체 WAM 플랫폼 UnifoLM-WMA-0(unitreerobotics GitHub — 회사 하드웨어 벤더의 월드모델이므로 존재만 알아둘 것), Li-Zn-H/AwesomeWorldModels.

### W4-M3. 내비게이션: SLAM 개요 → open-vocab 매핑 → DualMap ★ [P0] — Day 3

- 다룰 내용: 고전 스택 초압축(30분: SLAM→코스트맵→플래너; ORB-SLAM3, FAST-LIO2는 이름과 역할만), open-vocabulary 시맨틱 매핑 계보 — CLIP 특징을 3D에 사상(VLMaps), 장면그래프(ConceptGraphs), **DualMap(온라인·동적 환경 대응 dual map 구조 — concrete/abstract 맵, 자연어 질의 내비게이션)**, VLN(Vision-and-Language Navigation) 개요, legged robot 특유의 SLAM 이슈(진동, 시점 요동).
- 핵심 자료: github.com/Eku127/DualMap(논문 링크는 README), VLMaps(2210.05714), ConceptGraphs(2309.16650).
- 보충(팀 제공): VLN+파운데이션모델 서베이(2407.07035), VLN 종합 리뷰(Springer Discover Computing, s10791-026-09977-z), KwanWaiPang/Awesome-Legged-Robot-Localization-and-Mapping.
- 실습: **DualMap 데모 데이터셋 실행 + Claude Code로 리포 투어**(맵 자료구조와 질의 흐름 파악).

### W4-M4. 시스템 통합: 주파수 예산·미들웨어·안전 [P0] — Day 4

- 다룰 내용: §2.3 파이프라인을 실제 수치로 채우기 — VLA 수 Hz / WBC 50~500Hz / 관절 루프 kHz, 각 경계의 인터페이스(FSQ 토큰, 목표 자세, twist 명령), 지연과 chunking의 관계, DDS(unitree_sdk2)와 ROS2 개요, 온보드 vs 오프보드 컴퓨트 배치, 실패 모드와 안전 설계.
- 실습: 회사 스택 아키텍처 다이어그램 v2 작성(모듈별 입출력·주파수·실행 위치 명기).

### W4-M5. 캡스톤 + 회고 [P0] — Day 5

- 캡스톤(택1 또는 병행):
  - (A) **"우리 회사 Physical AI 스택 해설" 문서 + 30분 발표자료** — 팀원 검증용 질문 리스트 포함. 온보딩 스터디의 최종 산출물로 가장 추천.
  - (B) 시뮬 미니 파이프라인: 언어 명령 → 목표 좌표(간이 grounding) → 학습한 G1 보행 정책으로 이동 (W3 산출물 재사용).
- 회고: 체크포인트 전체 재답변 → 약한 모듈 식별 → §13의 다음 4주 계획 초안 작성.

### W4 체크포인트

1. 백지에 회사 전체 스택을 그리고 각 모듈의 입력·출력·주파수를 설명하라.
2. 월드모델의 3가지 활용 축(정책 결합/시뮬레이터/데이터 생성)을 대표 모델과 함께 설명하라.
3. DualMap이 "온라인·동적 환경"을 다루기 위해 도입한 구조를 설명하라.
4. 지금 팀 논문 리딩에 들어가면 못 알아들을 것 같은 주제는 무엇인가? (→ 다음 4주 계획의 입력)

---

## 9. 자료 라이브러리 (팀 제공 자료 재분류 + 보충)

### 9.1 팀 제공 자료 → 모듈 매핑

| 자료 | 정체 (조사 완료) | 배치 |
|---|---|---|
| arXiv:2605.00080 | World Model for Robot Learning 종합 서베이 (NTU 외, 2026-04) | W4-M2 핵심 |
| arXiv:2512.11362 | VLA Anatomy 서베이 (모듈→마일스톤→챌린지) | W2-M3 보충 |
| arXiv:2510.16732 | World Models for Embodied AI 서베이 (3축 분류) | W4-M2 보충 |
| arXiv:2503.21765 | 비디오 생성의 물리 인지 서베이 | W4-M2 [P2] |
| arXiv:2407.07035 | VLN × 파운데이션모델 서베이 | W4-M3 보충 |
| Springer s10791-026-09977-z | VLN 종합 리뷰 (Discover Computing 2026) | W4-M3 [P2] |
| DiT / FM / LDM / DDPM / RF | 생성모델 코어 | W1-M3~M5 |
| ACT / DP / pi0 | 정책 | W2-M1/M2/M4 |
| VQ-VAE / FSQ / LAPA | 액션 표현 | W1-M5, W2-M5 |
| World Models / DreamerV3 / V-JEPA 2 | 월드모델 | W4-M1 |
| AMP / SONIC | WBC | W3-M2/M4 |
| RT-2 / OXE / OpenVLA | VLA 계보 | W2-M3 |
| OpenHomie / GEAR-SONIC / DualMap / unitree 문서·GitHub | 회사 스택 | W3-M3/M4/M5, W4-M3 |
| awesome-legged-SLAM / AwesomeWorldModels / awesome-physical-ai ×2 | 색인용 (통독 금지, 검색용) | 수시 |
| arXiv:2606.12783 | World Models & Physical AI 튜토리얼 | W1-M1 |
| panaversity 코스 / DeepRobotics RL 가이드 / CVPR2026 워크숍 | 입문·행사 자료 — 참고 수준 [P2] | 수시 |

### 9.2 보충 자료 (조사 후 추가 — 왜 필요한지 포함)

**강의·교재**
- MIT 6.S184 (diffusion.csail.mit.edu): Flow Matching/Diffusion을 실습 노트북과 함께 — W1의 뼈대로 최적.
- Berkeley CS285: RL 기초 갭 메우기 (PPO 부분 선별).
- Underactuated Robotics (Tedrake, underactuated.mit.edu): 보행 동역학 — 제어 배경자용 심화 [P1].
- Modern Robotics (Lynch & Park): 기구학 표기 참조서.

**코드베이스·시뮬레이터 (실습 축)**
- huggingface/lerobot: 모방학습·VLA 학습 파이프라인 표준 — W2 실습 전체.
- isaac-sim/IsaacLab + unitreerobotics/unitree_rl_gym: G1 RL 학습→배포 표준 경로 — W3.
- google-deepmind/mujoco_menagerie, unitreerobotics/unitree_mujoco: sim2sim 검증.
- google-deepmind/mujoco_playground, Genesis-Embodied-AI/Genesis: GPU 여건이 안 될 때의 대체 경로.
- NVlabs/GR00T-WholeBodyControl: SONIC 공개 코드 — W3-M4.
- NVIDIA/Isaac-GR00T: GR00T N1 코드 — W2-M4.
- unitreerobotics/unitree_sdk2(_python): 실기 배포 인터페이스 — W3-M5.

**논문 (계보 완성용)**
- 정책/VLA: Octo(2405.12213), FAST(2501.09747), pi0.5(2504.16054), GR00T N1(2503.14734).
- WBC/RL: Learning to Walk in Minutes(2109.11978), DeepMimic(1804.02717), Hwangbo 2019 Science Robotics, 도메인 랜더마이제이션(1703.06907), HOMIE(2502.13013).
- 월드모델: Cosmos(2501.03575), Genie(2402.15391).
- 내비게이션: VLMaps(2210.05714), ConceptGraphs(2309.16650), CLIP(2103.00020), ORB-SLAM3(2007.11898), FAST-LIO2(2107.06829).
- 벤치마크(존재만 알기): LIBERO, Open X-Embodiment 데이터셋.

---

## 10. 학습 운영 가이드

- **논문 3-pass**: 1차(제목·초록·그림, 10분) → 2차(구조·수식 개요, 30분) → 3차(재현 가능 수준 정독, 핵심 논문만). 이 플랜에서 3-pass 대상은 FSQ, pi0, Diffusion Policy, SONIC, HOMIE, DualMap 6편이면 충분.
- **노트 체계**: 논문 1편 = ①한 줄 요약 ②아키텍처 그림 ③우리 스택과의 연결 ④의문점(팀 질문거리). 의문점 목록이 곧 온보딩 질문 리스트가 됨.
- **용어집**: 매일 5개씩 추가. 4주 뒤 100개 내외의 개인 용어집이 팀 대화 해독기가 됨.
- **Claude 활용 패턴**: 오전 학습자료 생성(§11.1) → 논문 막히면 딥다이브(§11.2) → 오후 코드는 Claude Code 투어(§11.3) → 저녁 셀프 퀴즈("오늘 모듈에서 소크라테스식 질문 10개 내줘").
- **주의**: awesome 리스트 통독 금지(무한 토끼굴). 이 플랜의 핵심 논문만 따라가고, 리스트는 검색이 필요할 때만.

---

## 11. Claude / Claude Code 프롬프트 템플릿

### 11.1 모듈 상세 학습자료 생성 (Claude)

```
당신은 Physical AI 온보딩 교육 콘텐츠 작성자입니다.

[내 배경] control engineering 10년, ML/DL/LLM/agent 실무 경험. 로봇 시뮬레이션·RL 실무·Physical AI는 처음.
[회사 스택] Unitree G1, HOMIE, GEAR-SONIC(SONIC WBC), DualMap, FSQ 기반 상위-하위 계층 모델.

아래 모듈의 상세 학습자료를 작성해주세요.
--- (마스터 플랜에서 해당 모듈 섹션 전체를 붙여넣기) ---

요구사항:
1. 개념 설명: 제어공학·LLM 개념에 빗댄 비유를 적극 사용 (예: FSQ ≈ 액션의 토크나이저).
2. 핵심 수식: 유도 과정 포함하되 최소한으로.
3. 아키텍처: 텍스트 다이어그램으로 입출력·차원 명시.
4. 회사 스택 연결: 이 개념이 우리 파이프라인 어디에 쓰이는지.
5. 흔한 오해 3가지와 교정.
6. 셀프 체크 퀴즈 10문항 (답은 접어서 뒤에).
분량: A4 4~6장. 웹 검색으로 최신 정보를 보강하고 출처를 남겨주세요.
```

### 11.2 논문 딥다이브 (Claude)

```
다음 논문을 3-pass 방식으로 딥다이브해주세요: [arXiv 링크]
[내 배경·회사 스택은 11.1과 동일]

산출물:
1. 한 문단 요약 + 이 논문이 계보에서 차지하는 위치 (선행 연구 → 이 논문 → 후속).
2. 방법 상세: 아키텍처 다이어그램(텍스트), 학습 objective 수식과 직관.
3. 실험에서 진짜 중요한 표/그림 2개와 해석.
4. 한계와 후속 연구가 공격한 지점.
5. 우리 회사 스택 관점의 시사점 + 팀에 물어볼 질문 3개.
6. 나를 검증할 질문 5개 (내가 답하면 채점해줄 것).
```

### 11.3 코드베이스 투어 (Claude Code)

```
# 리포 클론 후 Claude Code에서:
이 리포([OpenHomie | DualMap | unitree_rl_gym | GR00T-WholeBodyControl])의 코드베이스 투어를 해줘.

1. 디렉토리 맵: 각 최상위 디렉토리의 역할 한 줄씩.
2. 실행 진입점부터 콜스택 순서로 핵심 파일 5개 워크스루 (파일 경로:라인 인용).
3. 논문 [arXiv ID]의 핵심 수식/알고리즘이 코드 어디에 구현돼 있는지 매핑.
4. 설정 파일(config)에서 실무적으로 중요한 하이퍼파라미터 10개와 의미.
5. 내 환경([GPU/OS 명시])에서 데모를 돌리기 위한 최소 실행 가이드 작성 후, 가능하면 직접 실행해서 검증.
6. 이 코드를 우리 스택에 통합한다면 손대야 할 인터페이스 지점.
```

### 11.4 실습 랩 가이드 생성 (Claude / Claude Code)

```
아래 실습의 단계별 랩 가이드를 만들어주세요.
[실습명] 예: LeRobot으로 PushT에 Diffusion Policy 학습
[환경] Ubuntu 22.04, RTX ____, CUDA ____, 시간 예산 __시간

포함할 것:
1. 사전 준비 체크리스트 (설치 명령 포함, 버전 고정).
2. 단계별 절차 — 각 단계마다 "성공 판정 기준"(어떤 출력이 나와야 정상인지) 명시.
3. 결과 해석 가이드: 어떤 지표/커브를 보고 무엇을 판단하는가.
4. 흔한 에러 5개와 트러블슈팅.
5. 심화 변형 과제 2개 (예: horizon 바꿔 ablation).
```

---

## 12. 리스크와 우선순위

- **시뮬 경험 전무 + 클라우드 전용이 최대 리스크 조합** → 대응: 주력 실습 스택을 전부 "pip 설치 + headless"로 통일(MuJoCo, mujoco_playground, LeRobot). Isaac Lab은 주 경로에서 제외하고 [P1] 선택 과제로. 단, **회사 학습 인프라가 Isaac 기반일 가능성이 높으니 첫 주에 팀 표준 학습 환경(시뮬레이터·클러스터·도커 이미지)을 반드시 확인**하고, 있으면 그것을 그대로 물려받는 것이 최선.
- **클라우드 특유의 리스크**: 인스턴스 비용 폭주(스팟 활용, 학습 없는 시간 정지 습관), 체크포인트·데이터 유실(퍼시스턴트 볼륨 필수), 렌더링 이슈(`MUJOCO_GL=egl`, 뷰어 대신 mp4 저장). 상세는 부록 A.
- **시뮬 입문 리스크**: W1-M2에 1.5일까지 허용. 원칙 — "환경이 안 돌아가는 상태로 다음 주로 넘어가지 않는다". 셋업이 밀리면 그 주의 [P2] 이론부터 컷.
- **시간 부족 시 컷 순서**: ① W4-M2의 서베이 정독 → 목차+§5만 (WAM 개념만 확보) ② W2-M3의 OpenVLA 실행 → 코드 리딩으로 대체 ③ W1-M3의 DDPM 구현 → HF Annotated Diffusion 읽기로 대체 ④ Isaac Lab 시도는 전면 컷. **끝까지 지킬 것: FSQ 구현(W1-M5), LeRobot 완주(W2-M2), G1 보행 학습(playground)+sim2sim(W3), 캡스톤 문서(W4-M5).**
- **범위 통제**: 이 플랜은 "깊이 6편(회사 스택 핵심) + 넓이 계보"입니다. 서베이 4편을 전부 정독하려는 유혹을 경계하세요 — 서베이는 지도이지 목적지가 아닙니다. 담당 영역이 전 스택으로 확인된 만큼, 특정 트랙 심화보다 캡스톤(A)의 통합 관점을 우선하세요.

---

## 13. 4주 이후 (다음 스텝 미리보기)

W4 회고 결과에 따라 다음 중 하나로 심화하는 것을 권합니다. ① VLA 트랙: LeRobot으로 자체 데이터 파인튜닝 + FSQ 토크나이저를 회사 모델에 맞게 실험 ② WBC 트랙: SONIC 코드로 커스텀 모션 트래킹 학습 + 실기 배포 절차 습득 ③ 내비 트랙: DualMap을 G1 센서 스트림에 연결하는 통합 실험 ④ 월드모델 트랙: 회사 데이터로 정책 평가용 월드모델 실험. 선택되면 동일한 형식의 4주 심화 플랜을 다시 요청하세요.

---

## 부록 A. 클라우드 GPU 실습 환경 가이드

- **인스턴스 선택**: W1~W2(생성모델 toy, LeRobot PushT)는 T4/A10G/L4 1장이면 충분. W3(playground RL)는 A10G/L4/L40S급 권장. A100/H100은 이 플랜에서 불필요. Isaac Lab을 [P1]으로 시도할 경우에만 RT 코어가 있는 GPU(L4/L40S/A10G 계열)가 필요 — A100/V100은 Isaac Sim 렌더링을 지원하지 않음.
- **제공자**: 회사 클라우드 계정이 있으면 그것 우선(AWS g5/g6, GCP L4 등). 개인 실험은 시간당 과금형(RunPod, Lambda, Vast 등)이 편리. 노트북형(Colab Pro 등)은 W1 toy 실습과 MuJoCo 튜토리얼까지는 충분.
- **셋업 원칙 4가지**: ① 코드·데이터·체크포인트는 퍼시스턴트 볼륨에 (인스턴스는 소모품으로 취급) ② Docker 이미지 또는 버전 고정 conda/uv 환경으로 재현성 확보 ③ `MUJOCO_GL=egl`로 headless 렌더링, 결과는 뷰어 대신 mp4/이미지로 저장 ④ 학습 로그는 W&B/TensorBoard로 원격 확인, 장시간 작업은 tmux 세션 유지.
- **비용 통제**: 스팟/프리엠티블 + 잦은 체크포인트 저장 조합, 학습이 없는 시간에는 인스턴스 정지를 습관화. 주간 예산 상한을 정해두기.
- 사용할 제공자가 정해지면 이 부록과 §11.4 템플릿을 함께 Claude에게 주고 "내 제공자 기준 상세 셋업 가이드"를 생성 요청하세요.

---

*v1.1 — 2026-08-01. 확인된 조건(풀타임 온보딩 / 클라우드 GPU 전용 / 시뮬레이터 경험 전무 / 전 영역 담당 가능성)을 반영해 실습 주 경로를 MuJoCo + mujoco_playground 중심으로 재설계하고, W1-M2 시뮬레이터 부트캠프와 부록 A를 신설함.*