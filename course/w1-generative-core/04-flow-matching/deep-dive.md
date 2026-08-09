# W1-M4 심화 (계보 연대기, 증명 전문, 변환 대수, 인용 주의)

> 이 문서는 [lesson.md](lesson.md)에서 덜어낸 심화 내용입니다. 본문을 먼저 읽으세요.

본문에는 **결과와 직관과 그림과 회사 스택 연결**만 남기고, 유도의 중간 단계와 계보 세부, 트레이드오프 심층, 1차 출처 검증 메모를 여기로 옮겼습니다. 내용은 삭제하지 않았습니다. 읽는 순서만 바뀌었습니다.

| § | 무엇이 있나 | 본문에서 오는 링크 |
|---|---|---|
| §1 | 계보 세 갈래의 연대기 | lesson §2 |
| §2 | CFM 정리 증명 전문 | lesson §4.2 |
| §3 | 가우시안 경로의 아핀 구조 | lesson §5.1 |
| §4 | VP와 RF의 변환 대수 | lesson §5.4 |
| §5 | reflow의 이론적 뒷받침과 스텝 축소 트레이드오프 | lesson §6.3 · §6.4 |
| §6 | 1차 출처 검증 메모와 인용 주의 | lesson 「출처」 |

---

## 1. 계보 세 갈래의 연대기

본문 §2의 Mermaid는 압축한 그림입니다. 각 노드가 무엇을 바꿨는지 한 줄씩 붙입니다.

**시뮬레이션을 없애는 줄기**

- **Neural ODE**(Chen et al., NeurIPS 2018)가 잔차망을 연속시간 미분방정식으로 다시 읽었습니다. 깊이가 이산 층 수가 아니라 적분 구간이 됩니다
- **CNF(연속 정규화 흐름)**가 그 위에 가역 흐름과 최대가능도 학습을 올렸습니다. 개념은 정확했지만 매 그래디언트 스텝마다 ODE를 풀어야 해 스케일이 나지 않았습니다
- **Score SDE**(arXiv:2011.13456)가 확산을 연속시간 확률미분방정식으로 놓고 같은 주변분포를 갖는 probability flow ODE를 유도했습니다. 확산 쪽에서 ODE로 넘어오는 다리가 여기입니다(W1-M3 §5.1)
- **Flow Matching**(arXiv:2210.02747, Lipman, Chen, Ben-Hamu, Nickel, Le, ICLR 2023)이 학습을 회귀로 바꿔 시뮬레이션을 없앴습니다. 학습 비용이 확산 모델과 같아지면서 백본과 데이터와 인프라가 그대로 재사용됩니다
- **Conditional FM**이 같은 논문에서 계산 불가능한 주변 목표를 조건부 목표로 갈아끼웁니다. 본문 §4의 정리입니다

**직선으로 펴는 줄기**

- **Rectified Flow**(arXiv:2209.03003, Liu, Gong, Liu, "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow")가 직선 보간을 스케줄로 고정하고 reflow 절차를 붙였습니다
- **Stochastic Interpolants**는 두 편으로 나뉩니다. arXiv:2209.15571이 Albergo와 Vanden-Eijnden 2인의 ICLR 2023 판이고, arXiv:2303.08797이 Albergo, Boffi, Vanden-Eijnden 3인의 JMLR 본편입니다. 저자 수와 다루는 범위가 달라 본문에서도 구분해 인용했습니다
- **InstaFlow**(arXiv:2309.06380, ICLR 2024)가 reflow 뒤에 증류를 붙여 1스텝을 노립니다
- **SD3**(arXiv:2403.03206)가 Rectified Flow를 MMDiT 백본과 결합하고 $t$ 샘플링을 logit-normal로 바꿨습니다. 이미지 쪽 표준이 여기서 정해집니다

**스텝 수 자체를 학습 대상으로 올리는 줄기**

- **Shortcut Models**(arXiv:2410.12557, ICLR 2025)는 스텝 크기를 조건으로 받는 단일 모델을 self-consistency로 학습해 증류를 없앱니다
- **MeanFlow**(arXiv:2505.13447, NeurIPS 2025 oral)는 순간속도 대신 구간 평균속도를 학습해 from scratch로 1-NFE를 냅니다. ImageNet 256에서 1-NFE FID 3.43입니다
- **AdaFlow**(arXiv:2402.04292, NeurIPS 2024)는 분산에 따라 스텝 수를 가변으로 잡습니다. 단봉이면 자동으로 1스텝에 수렴한다는 관찰이 로봇 쪽에서 특히 흥미롭습니다

**로봇으로 내려오는 가지**

- **pi0**(arXiv:2410.24164)가 PaliGemma 3B 백본에 300M 액션 전문가를 붙였습니다. 청크 $H=50$, 추론 10 integration steps
- **GR00T N1**(arXiv:2503.14734)의 System 1이 Flow Matching 기반 DiT 액션 헤드이고 $K=4$입니다
- **FlowPolicy**(arXiv:2412.04987, AAAI 2025)는 Consistency FM 기반 3D 정책으로 추론 7배 가속을 보고합니다

---

## 2. CFM 정리 증명 전문

본문 §4.2는 두 항이 통과한다는 결과만 적었습니다. 전개를 그대로 옮깁니다.

제곱을 전개하면 이렇습니다.

$$
\bigl\|v_\theta-u\bigr\|^2=\bigl\|v_\theta\bigr\|^2-2\bigl\langle v_\theta,\,u\bigr\rangle+\bigl\|u\bigr\|^2
$$

세 번째 항은 $\mathcal{L}_{\text{FM}}$ 쪽에서는 $\|u_t(x)\|^2$, $\mathcal{L}_{\text{CFM}}$ 쪽에서는 $\|u_t(x|x_1)\|^2$이고 둘 다 $\theta$와 무관하므로 상수 $C$로 흡수됩니다. 남은 둘을 봅니다.

**이차항.** $p_t$의 정의를 그대로 대입하고 $v_\theta$가 $x_1$에 무관하다는 것만 쓰면 됩니다.

$$
\int\|v_\theta(x,t)\|^2p_t(x)\,dx
=\iint\|v_\theta(x,t)\|^2\,p_t(x\mid x_1)q(x_1)\,dx_1dx
=\mathbb{E}_{q(x_1)p_t(x\mid x_1)}\|v_\theta\|^2
$$

**교차항.** 여기가 심장입니다. 본문 §3.2의 박스에 $p_t(x)$를 곱한 형태

$$
p_t(x)\,u_t(x)=\int u_t(x\mid x_1)\,p_t(x\mid x_1)\,q(x_1)\,dx_1
$$

를 그대로 넣고, 내적이 두 번째 인자에 선형이므로 적분을 밖으로 뺍니다.

$$
\int\bigl\langle v_\theta,\,u_t(x)\bigr\rangle p_t(x)\,dx
=\int\Bigl\langle v_\theta,\,\int u_t(x\mid x_1)p_t(x\mid x_1)q(x_1)dx_1\Bigr\rangle dx
=\mathbb{E}_{q(x_1)p_t(x\mid x_1)}\bigl\langle v_\theta,\,u_t(x\mid x_1)\bigr\rangle
$$

양쪽이 같습니다. $\blacksquare$

**성립의 열쇠는 $u_t$의 정의 그 자체입니다.** 사후평균으로 정의했기 때문에 $p_tu_t$가 조건부의 적분으로 인수분해되고, 그 덕에 교차항이 통과합니다. 다른 정의였다면 정리가 무너집니다.

---

## 3. 가우시안 경로의 아핀 구조

본문 §5.1은 코드에 들어가는 형태만 남겼습니다.

$$
u_t\bigl(x_t\mid x_1,x_0\bigr)=\dot\alpha_tx_1+\dot\sigma_tx_0
$$

$x_0=(x_t-\alpha_tx_1)/\sigma_t$를 대입해 $x_t$만으로 다시 쓰면 이렇게 됩니다.

$$
u_t(x\mid x_1)=\frac{\dot\sigma_t}{\sigma_t}\,x+\Bigl(\dot\alpha_t-\frac{\dot\sigma_t}{\sigma_t}\alpha_t\Bigr)x_1
$$

**제어 대응**은 상태 $x$에 대한 아핀 벡터장입니다. $\dot\sigma_t/\sigma_t$가 시스템 행렬(스칼라배)이고 나머지가 $x_1$을 입력으로 받는 입력 행렬인 시변 1차 선형계라 해가 닫힙니다. 조건부 경로에서 모든 것이 닫힌 형태로 나오는 이유가 이 구조입니다.

---

## 4. VP와 RF의 변환 대수

본문 §5.4는 "스케일 재조정과 시간 재매개화로 서로 옮겨진다"는 결론과 $(\sigma,\alpha)$ 평면의 도형만 남겼습니다. 변환식은 이렇습니다.

$$
\underbrace{x^{\text{VP}}_{s}}_{\text{DDPM 계열}}=\frac{x^{\text{RF}}_{t(s)}}{\sqrt{t(s)^2+\bigl(1-t(s)\bigr)^2}},
\qquad
t(s)\ \text{는}\ \frac{\alpha^{\text{VP}}_s}{\sigma^{\text{VP}}_s}=\frac{t(s)}{1-t(s)}\ \text{의 해}
$$

확인은 한 줄입니다. $(t,1-t)$를 그 $\ell_2$ 노름으로 나누면 $\alpha^2+\sigma^2=1$이 되고 이것이 정확히 VP 조건입니다. 비율 $\alpha/\sigma$는 나눗셈에 불변이라 SNR이 보존됩니다. 반대로 VP 경로를 $\alpha+\sigma$로 나누면 선형 경로가 됩니다.

정당화의 뿌리는 자유도 세기입니다. $x_t=\alpha_tx_1+\sigma_tx_0$ 꼴의 가우시안 경로는 각 시각에서 스케일과 SNR 두 값으로 완전히 결정됩니다. 그런데 스케일은 입력을 상수배 한 것과 같아 신경망이 흡수하는 자유도이므로, 본질적 자유도는 SNR 궤적 하나뿐입니다.

---

## 5. reflow의 이론적 뒷받침과 스텝 축소 트레이드오프

### 5.1 왜 reflow가 수렴하는가

본문 §6.3은 절차와 성과만 남겼습니다. 논문의 이론적 뒷받침은 **rectify 연산이 볼록 수송비용을 증가시키지 않는다**는 것입니다. 즉 $k$-rectified를 반복해도 어떤 볼록 비용 기준에서 나빠지지 않으므로 "재계획의 재계획"이 수렴합니다.

**제어 대응**을 좀 더 붙이면 이렇습니다. 초기 계획이 장애물을 피하느라 꺾여 있을 때, 그 계획이 만든 시작점과 끝점 쌍만 남기고 다시 최단 경로를 풀면 궤적이 매끄러워집니다. 이때 끝점 배정이 바뀌지 않으므로 계획의 "무엇을 하려 했는가"는 보존되고 "어떻게 가는가"만 펴집니다. reflow가 분포를 보존하면서 직선성만 올리는 것과 같은 구조입니다.

### 5.2 어느 기법을 언제 쓰는가

본문 §6.4의 표를 지불하는 대가 기준으로 다시 읽습니다.

- **샘플러 교체와 경로 설계**는 품질 손실 없이 얻는 이득입니다. 전자는 학습을 건드리지 않고, 후자는 처음부터 그렇게 학습하면 됩니다
- **reflow**는 샘플 생성과 재학습 비용을 지불하고 큰 NFE의 품질을 내줍니다. 로봇에서 큰 NFE를 쓸 일이 없다면 손해가 아닐 수 있습니다
- **증류**는 교사 모델을 따로 유지해야 하고, 학생이 교사의 편향을 상속합니다
- **self-consistency와 평균속도 학습**은 증류 없이 적은 스텝을 노리지만 학습 자체가 더 까다롭습니다
- **적응형 solver**는 NFE를 데이터가 정하게 맡깁니다. 최악값 예산을 잡아야 하므로 실시간 제어에서는 상한을 따로 걸어야 합니다

다봉성이 중요한 조작 태스크에서 1-step 증류가 모드를 뭉갤 수 있다는 점(W1-M3 §7.3)은 이 표의 어느 줄을 고르든 함께 재야 하는 축입니다.

---

## 6. 1차 출처 검증 메모와 인용 주의

### 6.1 미검증으로 남긴 것 셋

- **Rectified Flow의 $k$별 1-step FID 표**(6.18 / 12.21 / 8.15 등)는 2차 출처 간 값이 충돌해 **정성 서술로만 처리**했습니다. 쓸 수 있는 수치는 "CIFAR-10 1-step FID 4.85, recall 0.51" 하나뿐입니다
- **GR00T N1.5 이후는 arXiv 논문이 없습니다.** NVIDIA 모델 카드와 `github.com/NVIDIA/Isaac-GR00T`만 있으므로 N1.5를 논문 인용처럼 쓰지 마세요. 본문이 인용한 K=4는 N1(arXiv:2503.14734) 기준입니다
- **Neural ODE**(Chen et al., NeurIPS 2018)와 **CNF/FFJORD**는 arXiv ID를 재확인하지 않아 본문에 ID를 표기하지 않았습니다

### 6.2 명칭 인용 주의

**pi0의 액션 전문가는 DiT가 아닙니다.** PaliGemma와 가중치를 나눠 갖는 mixture-of-experts형 별도 트랜스포머이고 조건 주입도 adaLN이 아닙니다. DiT가 정확한 쪽은 GR00T N1의 System 1입니다. "요즘 VLA는 다 Flow Matching에 DiT"라는 요약은 **목적함수 계보와 블록 계보를 섞은 것**이라 분리해 읽어야 정확합니다(W1-M3 §7.4와 [용어집](../../../notes/glossary.md)의 구분).

**pi0의 세 숫자도 축이 다릅니다.** $H=50$은 청크 길이, 최대 50 Hz는 하위 소비 주파수 $f_2$, 10 integration steps가 NFE입니다. 본문 §8.3이 이 셋을 갈라 놓았습니다.

### 6.3 회사 스택 부분의 지위

본문 §9의 계층 배치는 [마스터플랜](../../../docs/physical-ai-4week-master-plan.md) §2.1~2.3의 **추정** 기반이고 검증은 W4-M5 캡스톤입니다. §9.4의 7개 항목은 확인되지 않았고 추측하지 않았습니다. 질문은 `notes/questions-for-team.md`의 M4-1부터 M4-6까지로 적립돼 있습니다.
