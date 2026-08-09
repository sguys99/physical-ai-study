# W1-M3 심화 (전개 전문, 대조표, 인용 주의)

> 이 문서는 [lesson.md](lesson.md)에서 덜어낸 심화 내용입니다. 본문을 먼저 읽으세요.

본문에는 **결과와 직관과 그림과 회사 스택 연결**만 남기고, 유도의 중간 단계와 계보 세부, 축별 대조표, 1차 출처 검증 메모를 여기로 옮겼습니다. 내용은 삭제하지 않았습니다. 읽는 순서만 바뀌었습니다.

| § | 무엇이 있나 | 본문에서 오는 링크 |
|---|---|---|
| §1 | 계보 두 줄기의 연대기 | lesson §2 |
| §2 | 닫힌 형태 유도 전문 | lesson §3.2 |
| §3 | ELBO 항별 전개와 KL 붕괴 | lesson §4.2 |
| §4 | $\mu$에서 $\epsilon$으로의 대수 | lesson §4.3 |
| §5 | 샘플러 일반형과 DDIM 재정식화 | lesson §5.1 |
| §6 | $\epsilon_\theta$와 score의 관계 | lesson §4.4 |
| §7 | U-Net을 왜 버렸나, 축별 전문 | lesson §6.4 |
| §8 | conditioning과 액션 헤드 계보의 인용 주의 | lesson §6.2 · §7.4 · 출처 |

---

## 1. 계보 두 줄기의 연대기

본문 §2의 Mermaid는 두 줄기로 압축한 그림입니다. 각 노드가 무엇을 바꿨는지 한 줄씩 붙입니다.

**왼쪽 줄기, 샘플링 비용을 깎는 역사**

- **Sohl-Dickstein 2015**(arXiv:1503.03585)가 비평형 열역학의 확산 과정으로 생성 모델을 세웠습니다. 학습은 됐지만 샘플 품질이 실용권에 못 미쳤습니다
- **Song & Ermon 2019**(arXiv:1907.05600)는 같은 대상을 score matching과 Langevin 동역학으로 접근했습니다. 본문 §4.4의 관계식이 두 계보가 같은 것을 학습했음을 보입니다
- **DDPM 2020**(arXiv:2006.11239)이 목적함수를 $\mathcal{L}_{\text{simple}}$ 한 줄로 정리해 학습을 안정시켰습니다. 대신 샘플링이 250~1000 스텝입니다
- **DDIM 2020**(arXiv:2010.02502)이 forward를 비마르코프 족으로 재정식화해 격자를 건너뛸 수 있게 했습니다
- **Score SDE 2021**(arXiv:2011.13456)이 확산을 연속시간 SDE로 놓고 같은 주변분포를 갖는 probability flow ODE를 유도했습니다. 고차 solver의 문턱이 여기서 열립니다
- **Flow Matching**(arXiv:2210.02747)과 **Rectified Flow**(arXiv:2209.03003)가 경로 자체를 설계 대상으로 바꿉니다. W1-M4 담당입니다

**오른쪽 줄기, 백본을 갈아치우는 역사**

- **improved DDPM**(arXiv:2102.09672)이 분산을 학습 대상으로 되돌리고 cosine 스케줄을 도입했습니다. 본문 §4.2에서 포기한 자유도를 되찾는 작업입니다
- **ADM**(arXiv:2105.05233)이 U-Net을 개량하고 classifier guidance를 붙였습니다
- **classifier-free guidance**(arXiv:2207.12598)가 별도 분류기 없이 조건 강도를 조절하게 했습니다
- **LDM**(arXiv:2112.10752)이 픽셀이 아니라 VAE 잠재 위에서 확산합니다. 왜 잠재인지는 W1-M5 담당입니다
- **DiT**(arXiv:2212.09748)가 U-Net을 Transformer로 교체합니다. 본문 §6이 여기입니다

두 줄기는 로봇 액션 헤드에서 다시 만납니다. Diffusion Policy(arXiv:2303.04137)는 왼쪽 줄기의 DDPM/DDIM을 그대로 쓰고, GR00T N1(arXiv:2503.14734)과 pi0(arXiv:2410.24164)는 오른쪽 줄기의 블록에 Flow Matching 목적함수를 얹었습니다.

---

## 2. 닫힌 형태 유도 전문

본문 §3.2는 결과만 실었습니다. 두 스텝을 실제로 펼치면 이렇습니다.

$$
x_t = \sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-1}\right) + \sqrt{1-\alpha_t}\,\epsilon_t
= \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} + \underbrace{\sqrt{\alpha_t(1-\alpha_{t-1})}\,\epsilon_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_t}_{\text{독립 가우시안 두 개의 합}}
$$

독립 가우시안의 합은 분산이 더해지므로

$$
\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) = \alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t = 1 - \alpha_t\alpha_{t-1}
$$

두 항이 하나의 $\mathcal{N}(0,(1-\alpha_t\alpha_{t-1})I)$로 합쳐집니다. 귀납으로 $t$까지 밀면 본문 §3.2의 boxed 결과가 나옵니다.

**제어 대응을 한 번 더.** 공분산 전파 재귀 $P_t = A_tP_{t-1}A_t^\top + Q_t$를 손으로 풀면 위 식이 나옵니다. $A_t$가 $I$의 스칼라배이고 $Q_t$가 등방이라 재귀가 대수적으로 닫힙니다. 일반적인 $A,Q$라면 수치로 돌려야 합니다. $\sqrt{\bar\alpha_t}$가 신호 전달 이득, $1-\bar\alpha_t$가 누적 잡음 분산이고 $\mathrm{SNR}(t)=\bar\alpha_t/(1-\bar\alpha_t)$입니다.

---

## 3. ELBO 항별 전개와 KL 붕괴

본문 §4.2는 세 덩어리와 결과만 보입니다. 각 항이 왜 그렇게 되는지 붙입니다.

- **$L_T = D_{\mathrm{KL}}(q(x_T|x_0)\,\|\,p(x_T))$에는 파라미터가 없습니다.** forward가 고정 상수 스케줄이고 prior $p(x_T)=\mathcal{N}(0,I)$도 고정이기 때문입니다. 본문 §3.3의 표에서 $\bar\alpha_{1000}=4.04\times10^{-5}$라 이 항은 0에 가깝지만 정확히 0은 아닙니다. W1-M4가 이 틈을 다시 짚습니다
- **$L_0 = -\log p_\theta(x_0|x_1)$**은 마지막 한 스텝의 재구성 항입니다. 이산 픽셀값을 다루는 별도의 디코더 항으로 처리됩니다
- **$L_{t-1}$**이 학습의 본체입니다

DDPM은 $p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t,t),\ \sigma_t^2 I)$로 공분산을 **고정**합니다. 두 가우시안의 KL은 일반적으로 평균 차이 항과 공분산 항의 합인데, 공분산이 같으면 공분산 항이 사라지고 평균 차이의 제곱항만 남습니다.

$$
D_{\mathrm{KL}}\!\left(\mathcal{N}(\tilde\mu,\sigma^2I)\,\|\,\mathcal{N}(\mu_\theta,\sigma^2I)\right) = \frac{1}{2\sigma^2}\|\tilde\mu-\mu_\theta\|^2
$$

**포기한 것은 분산 모델링의 자유도입니다.** improved DDPM(arXiv:2102.09672)이 나중에 이를 학습 대상으로 되돌립니다.

---

## 4. $\mu$에서 $\epsilon$으로의 대수

본문 §4.3은 결과 두 식만 실었습니다. 중간 단계입니다.

닫힌 형태를 뒤집으면 $x_0 = \frac{1}{\sqrt{\bar\alpha_t}}(x_t-\sqrt{1-\bar\alpha_t}\,\epsilon)$입니다. 이것을 $\tilde\mu_t$에 대입해 정리하면 $x_0$이 사라집니다.

$$
\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\right)
$$

모델도 같은 꼴로 잡습니다.

$$
\mu_\theta(x_t,t) := \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\right)
$$

$x_t$는 모델이 이미 갖고 있는 입력이라 차이에서 그대로 빠지고, 남는 것은 $\epsilon$과 $\epsilon_\theta$의 차이뿐입니다. §3의 KL 결과에 대입하면 본문 §4.3의 $\lambda_t$ 형태가 나옵니다.

**제어 대응 전문.** 좌표 변환입니다. 추정 대상을 상태($\mu$)에서 잡음($\epsilon$)으로 바꿨을 뿐 정보량은 같습니다. 다만 새 좌표에서는 타깃 스케일이 $t$와 무관하게 $\mathcal{N}(0,I)$로 정규화되어 신경망이 모든 $t$에서 같은 스케일의 타깃을 봅니다. 플랜트 이득을 정규화한 오차 좌표로 옮겨 잡는 것과 같은 이유입니다.

---

## 5. 샘플러 일반형과 DDIM 재정식화

본문 §5.1은 두 갱신식을 나란히 놓았습니다. 사실 둘은 **같은 식의 $\eta$ 양 끝**입니다. DDIM 논문의 일반형이 이렇습니다.

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t,t) + \sigma_t z,
\qquad
\sigma_t = \eta\sqrt{\tilde\beta_t}
$$

- $\eta=1$이면 ancestral(DDPM 원본)과 같아집니다
- $\eta=0$이면 확률 항이 사라져 결정론적 DDIM이 됩니다
- 실습 [`practice/02_samplers_compare.py`](practice/02_samplers_compare.py)가 이 하나의 함수로 두 샘플러를 모두 구현합니다

**비마르코프 재정식화가 왜 필요한가.** DDPM의 forward는 마르코프 사슬이라 $q(x_{t-1}|x_t,x_0)$이 인접 스텝에만 정의됩니다. DDIM은 $q(x_t|x_0)$의 주변분포만 같게 유지하는 **비마르코프 forward 족**을 새로 정의합니다. 주변분포가 같으므로 이미 학습된 $\epsilon_\theta$를 그대로 쓸 수 있고, 인접 제약이 없으므로 부분수열 $\{\tau_1<\dots<\tau_S\}$ 위에서 바로 샘플링할 수 있습니다. **재학습 없이 스텝을 줄이는 근거가 여기입니다.**

**제어 대응 전문.** 스텝 수는 이산화 격자 수이고 NFE와 같습니다. 확률적 시스템은 노이즈 항 때문에 격자를 마음대로 늘릴 수 없지만, 결정론적 ODE로 바꿔놓으면 절단오차만 관리하면 되고 고차 적분기(DPM-Solver 계열)를 붙일 수 있습니다.

---

## 6. $\epsilon_\theta$와 score의 관계

본문 §4.4는 결과만 한 줄로 실었습니다. 유도입니다.

$q(x_t|x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$가 가우시안이므로 로그밀도의 기울기가 닫힌 형태로 나옵니다.

$$
\nabla_{x_t}\log q(x_t\mid x_0) = -\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t} = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}
\qquad\Longrightarrow\qquad
\epsilon_\theta \approx -\sqrt{1-\bar\alpha_t}\,s_\theta
$$

**DDPM 계열과 score 기반 계열은 같은 대상을 다른 이름으로 학습했습니다.** 그래서 본문 §2에서 두 줄기가 Score SDE에서 합류합니다. $s_\theta$를 "가장 그럴듯한 쪽으로 가는 벡터장"으로 읽으면 샘플링은 그것을 적분하는 일이 되고, 이 관점이 본문 §5의 ODE 시각으로 이어집니다.

---

## 7. U-Net을 왜 버렸나, 축별 전문

본문 §6.4는 세 축의 표와 로봇 쪽 결론만 실었습니다. 나머지 두 축을 포함한 전체입니다.

| 축 | U-Net (ADM, LDM) | DiT |
|---|---|---|
| 귀납 편향 | 다중해상도 conv + skip. 공간 국소성 내장 | 없음. 위치 인코딩으로만 전달 |
| 스케일링 | 채널·해상도·블록 수를 손으로 조합 | depth·width·토큰 수 세 노브, Gflops로 예산 관리 |
| 조건 주입 | 시간 임베딩 + attention 블록 삽입 | adaLN-Zero 기본, 이종 토큰은 cross-attn |
| 연산량·성능 | LDM-4 103.6 / ADM 1120 / ADM-U 742 Gflops | DiT-XL/2 **118.6 Gflops · 675M** · FID **2.27**(cfg 1.50) / 3.22(cfg 1.25) / 9.62(무보정) |
| 로봇 쪽 실익 | 이종 조건 토큰 붙이기가 번거롭다 | 멀티뷰 이미지·언어·proprioception을 토큰으로 그냥 붙인다 |

**patchify와 Gflops의 근거.** 본문 §6.1이 "품질을 정하는 것은 Gflops"라고 단언하는 근거는 논문의 스케일링 실험입니다. $p$를 8에서 2로 줄이면 파라미터는 오히려 약간 줄지만 토큰 수가 제곱으로 늘어 Gflops만 커지는데 FID는 개선됩니다. 그리고 Gflops가 비슷한 서로 다른 구성(DiT-S/2와 DiT-B/4)이 비슷한 FID를 얻습니다. 두 관찰이 함께여야 "파라미터가 아니라 Gflops"라는 주장이 섭니다.

마지막 줄이 로봇에서 결정적입니다. 이미지 생성은 조건이 클래스 라벨 하나지만 액션 헤드는 카메라 $V$대의 패치 토큰, 언어 토큰 $L$개, proprioception을 동시에 받습니다. **길이가 가변인 이종 토큰 집합에 조건을 거는 일은 Transformer가 원래 하던 일이고 conv U-Net에는 남의 일입니다.** FID 소수점이 아니라 이 인터페이스 유연성이 액션 헤드가 DiT로 수렴한 실제 이유입니다.

---

## 8. conditioning과 액션 헤드 계보의 인용 주의

### 8.1 adaLN-Zero 원문 인용

본문 §6.3의 "블록이 항등함수"라는 주장의 1차 출처는 DiT 논문 §3의 이 문장입니다.

> we initialize the MLP to output the zero-vector for all $\alpha$; this initializes the full DiT block as the identity function

실습 [`practice/03_dit_action_head.py`](practice/03_dit_action_head.py)의 `verify_identity()`가 `assert torch.allclose(block(h,c), h, atol=0)`로 이 문장을 코드로 확인합니다.

### 8.2 conditioning 4방식 표의 조건

본문 §6.2 표의 수치는 전부 **DiT-XL/2, FID-50K, 400K 학습 스텝, guidance 없음** 조건입니다. 같은 논문의 최종 결과 FID 2.27은 7M 스텝에 cfg 1.50을 건 값이라 서로 다른 조건입니다. 두 숫자를 같은 축에 놓고 비교하면 안 됩니다.

cross-attention의 "약 15% 오버헤드"는 Gflops 119.37 대비 137.62를 논문이 서술한 표현입니다.

**LLM 어휘로 옮기면.** adaLN은 본체를 건드리지 않고 정규화 계수만 조건에 따라 바꾸는 **정규화 계수 경로**이고, in-context는 조건을 프롬프트 토큰으로 붙이는 **토큰 경로**입니다. 표의 FID가 말하는 것은 정규화 계수 경로가 프롬프트 경로를 크게 이겼다는 사실입니다. 조건이 시퀀스 전체에 균일하게 작용해야 할 때는 토큰보다 정규화에 태우는 편이 낫고, 반대로 조건이 위치마다 다르게 걸려야 하면 cross-attention이 필요합니다. 액션 헤드가 스텝 $t$는 adaLN으로, 관측은 cross-attention으로 나눠 받는 이유가 이것입니다(본문 §7.1).

### 8.3 VLA 액션 헤드 계보를 읽는 주의

본문 §7.4 표가 마스터플랜의 "대부분의 VLA 액션 헤드가 DiT 구조"에 붙이는 단서를 풀어 씁니다.

- **GR00T N1**(arXiv:2503.14734)은 액션 모듈이 Diffusion Transformer이고 스텝 조건 주입이 adaptive layer normalization입니다. "DiT 구조"가 정확한 쪽입니다
- **GR00T N1.5**는 같은 DiT 구조에 VLM만 Eagle 2.5로 교체했습니다. 근거는 논문이 아니라 NVIDIA 모델카드와 `github.com/NVIDIA/Isaac-GR00T`라 2차 출처입니다
- **pi0**(arXiv:2410.24164)의 액션 전문가는 **DiT가 아닙니다.** PaliGemma와 구조를 공유하는 300M Gemma 블록이고 스텝 조건도 AdaLN이 아니라 노이즈 액션과 $\phi(\tau)$를 MLP로 접어 넣는 방식입니다. 다만 같은 논문이 DiT 블록 베이스라인도 함께 보고하며 그쪽은 AdaLN-Zero입니다
- **Diffusion Policy**(arXiv:2303.04137)는 1D conv U-Net 또는 Transformer이고 목적함수가 DDPM/DDIM 계열입니다. W2-M2가 본체입니다

**objective(대부분 Flow Matching)와 블록 계보(DiT 또는 언어모델 블록)를 분리해 읽어야** 요약이 정확해집니다. 두 축을 뭉치면 pi0가 반례가 됩니다.
