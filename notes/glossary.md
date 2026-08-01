# 용어집

> 매일 5개씩, 주차 종료 시 25개 누적. 4주 뒤 100개 내외가 목표입니다.
> 용어는 [마스터플랜 §3](../docs/physical-ai-4week-master-plan.md)의 용어 사전과 일치시킵니다.
> 형식: **약어/용어** — 한 줄 정의 · (있으면) 제어·ML·LLM 대응 비유 · 처음 만난 모듈

---

## 미리 채워둔 기본 약어 (마스터플랜 §3)

| 용어 | 정의 | 비유 / 메모 | 모듈 |
|---|---|---|---|
| **VLA** | Vision-Language-Action 모델. 시각·언어 입력에서 로봇 액션을 직접 출력 | VLM에 액션 디코더를 붙인 것 | W2-M3 |
| **WBC** | Whole-Body Control. 전신 관절을 협조 제어 | 다관절 시스템의 통합 제어기 | W3 |
| **WFM** | World Foundation Model. NVIDIA Cosmos처럼 범용 비디오 기반 월드모델 | 학습된 플랜트 모델 | W4-M2 |
| **WAM** | World-Action Model. 비디오 월드모델 백본에 액션을 latent frame으로 주입해 예측과 행동을 공동 생성 | 모델 + 정책 동시 학습 | W4-M2 |
| **IDM** | Inverse Dynamics Model. 연속 관측에서 그 사이의 액션을 역추정 | 역동역학 | W2-M5 |
| **OXE** | Open X-Embodiment. 여러 로봇 플랫폼의 시연 데이터를 통합한 데이터셋 | — | W2-M3 |
| **teleop** | 원격조작. 사람이 로봇을 직접 조종해 시연 데이터를 만드는 것 | — | W2-M1 |
| **retargeting** | 인간 모션 → 로봇 관절 매핑 | 좌표계·링크 길이 변환 | W3-M2 |
| **sim2sim** | 학습 시뮬레이터(Isaac/MJX) → 검증 시뮬레이터(표준 MuJoCo) 이식 검증 | 플랜트 모델 교차 검증 | W3-M5 |
| **FSQ** | Finite Scalar Quantization. 각 차원을 고정 레벨로 반올림하는 이산화 — codebook 없이 collapse 원천 차단 | **액션의 토크나이저** | W1-M5 |

---

## W1

_(집필·학습 진행하며 추가)_

## W2

## W3

## W4
