# W2-M1 랩 — LeRobot 파이프라인을 처음 돌리고, lesson이 "미확인"으로 남긴 값을 자기 손으로 닫기

> **모듈**: W2-M1 · [`../lesson.md`](../lesson.md)
> **8단계 · 약 2시간 5분** (CPU만 있으면 약 2.5시간) · lesson 정독 2~2.5h는 별도
> **로컬 RTX 3080으로 완결. 클라우드 불필요. G1 실기 불필요. HF 계정 불필요.**
> 이 랩의 제출물은 [`worksheet.md`](worksheet.md)입니다. 시작 전에 사본을 하나 만들어 두세요.

[W1-M2 랩](../../../w1-generative-core/02-simulator-bootcamp/labs/README.md)이 **MuJoCo 환경을 만드는 랩**이었다면, 이 랩은 **로봇 학습 파이프라인을 처음 돌리는 랩**입니다. 데이터셋을 받고, 정책을 만들고, 손실이 내려가는 것까지 봅니다. 시뮬레이터도 모방학습 파이프라인도 처음이라는 전제로 씁니다.

> 🔴 **이 랩의 무게중심 — "설치됐다"를 확인하는 랩이 아닙니다**
>
> lesson [§10 신뢰도 각주](../lesson.md)에는 원래 **미확인 4건**이 있었고, [§3.6](../lesson.md)의 $\tau_{\text{infer}}$ 칸은 비어 있었습니다. **그 넷은 바로 이 랩의 절차로 닫혔고, 지금은 lesson §10에 ✅로 적혀 있습니다.**
>
> **그래서 이 랩이 하는 일은 "닫힌 것을 자기 손으로 다시 확인하고, 자기 기기의 숫자로 표를 다시 채우는 것"입니다.** 남이 잰 값을 읽는 것과 자기가 재는 것은 다릅니다 — 특히 $\tau_{\text{infer}}$는 **기기마다 자릿수가 달라져 결론까지 바뀝니다**(Step 6).
>
> | lesson §10의 항목 | 상태 | 어느 Step이 재확인하는가 |
> |---|---|---|
> | ACT/ALOHA 데이터의 기록 FPS = **50 Hz** | ✅ 닫힘 | **Step 3** |
> | PushT `repo_id` = **`lerobot/pusht`** | ✅ 닫힘 | **Step 3** |
> | 앙상블 오설정 시 **`NotImplementedError`** | ✅ 닫힘 | **Step 2** |
> | 하이퍼파라미터 21개가 설치본과 일치 | ✅ 닫힘 | **Step 2** |
> | **파라미터 수 — HF 문서 80M vs 실측 51.6M** | 🔴 **원인 미확인** | **Step 6** (여기서도 안 닫힙니다) |
> | **Jetson급 온보드 지연** | 📎 **미측정** | (측정 불가 — 하드웨어 없음) |
> | $\tau_{\text{infer}}$ — **자기 기기 값** | — | **Step 6 → Step 8.A** |
>
> 전체 2시간 중 **실제로 명령이 도는 시간은 다 합쳐 3분**입니다(CPU면 14분). 나머지는 **읽고 쓰고 계산하는 시간**입니다.
> **① Step 8.A — lesson §3.6 표를 자기 기기의 τ_infer로 다시 계산하고, "지연은 제약이 아니다"가 몇 배 여유인지 자기 숫자로 쓰는 것.**
> **② Step 8.B — lesson 상단의 완료 기준 (a)(b)(c)를 한 문단으로 쓰는 것. 이것이 이 모듈의 최종 제출물입니다.**
> **Step 1~6에 40분, Step 7~8에 85분**을 씁니다. 비율이 뒤집히면 이 모듈을 잘못 하고 있는 것입니다.

---

## 0. 사전 준비 체크리스트

### 0.1 이 랩에 필요한 것 / 필요 없는 것

| 항목 | 필요 여부 | 비고 |
|---|---|---|
| **Python 3.12** | **필요** | LeRobot이 3.12를 요구합니다(lesson §8.2). 3.13 이상에서는 설치가 막힙니다 |
| **리포 루트의 `.venv-lerobot/`** ★ | **필요** | 🔴 **W1에서 쓰던 `.venv`와 별개입니다.** 섞으면 안 됩니다 — 0.2와 에러 표 **E7** |
| **로컬 GPU (RTX 3080급)** | **강권, 필수 아님** | Step 1~4·6은 CPU로도 그대로 돕니다. **Step 5 학습 스모크만 31초 → 약 10분**이 됩니다(0.3). **100k 본 학습만 GPU 전용**이고 이 모듈에서는 하지 않습니다 |
| **디스크 여유 — 최소 8 GB** ★ | **필요** | 실측 내역이 아래 표에 있습니다. **큰 데이터셋(1.7 GB)까지 받으면 12 GB.** 에러 표 **E6** |
| **인터넷** | **필요** | pip · HF Hub 데이터셋 · ResNet18 사전학습 가중치 |
| **HF(Hugging Face) 계정·토큰** | **불필요** | 공개 데이터셋은 인증 없이 받아집니다(1.74 GB를 40.6초에). 경고만 뜹니다 — **E4**. 학습은 `--policy.push_to_hub=false`로 — **E3** |
| **`ffmpeg`** | **불필요** | 없어도 됩니다. `torchcodec`이 로드에 실패하고 `pyav`로 폴백하는데 **모든 절차가 그 상태에서 정상 동작**했습니다 — **E1** |
| 디스플레이 / X11 / 뷰어 | **불필요이자 금지** | `lerobot-dataset-viz`는 **반드시 `--save 1`** 로 씁니다(Step 4). 뷰어를 띄우지 않습니다 |
| Unitree G1 실기 | 불필요 | 이 4주 내내 불필요 |
| 클라우드 인스턴스 | 불필요 | 30분 넘는 절차가 하나도 없습니다 |
| `gym-aloha` · `gym-pusht` | **불필요** | 시뮬 롤아웃·평가는 **W2-M2의 몫**입니다(lesson §8.1) |
| **excalidraw (웹)** | **필요** | Step 7에서 씁니다. 설치 없이 [excalidraw.com](https://excalidraw.com)에서 파일을 엽니다 |
| **선수 모듈 W2-M1 practice** | **필수** | `01`~`04`를 먼저 돌려야 Step 8이 성립합니다 — 0.2 ③ |
| 선수 모듈 W1-M1 | **권장** | Step 8의 200 ms 문턱이 [W1-M1 §3.3](../../../w1-generative-core/01-physical-ai-landscape/lesson.md), 부등식이 §4.1 |
| 선수 모듈 W1-M2 | 권장 | 헤드리스 규약과 tmux 사용법이 [그 랩](../../../w1-generative-core/02-simulator-bootcamp/labs/README.md)에 있습니다 |

**디스크 내역 — 집필 환경 실측**

| 무엇 | 크기 | 언제 생기나 | 지워도 되나 |
|---|---|---|---|
| `.venv-lerobot/` | **5.8 GB** | Step 1 (1회) | ❌ 이 랩 내내 씁니다 |
| `~/.cache/huggingface/lerobot/` | **70 MB** (ALOHA sim만) ~ **1.7 GB** (3종 전부) | Step 3 | ✅ 다시 받으면 됩니다 |
| `~/.cache/torch/` | **45 MB** | Step 5 (ResNet18 사전학습 가중치) | ✅ |
| `outputs/train/<job>/checkpoints/NNNNNN/` | **592 MB / 개** | Step 5 | ✅ **다 보면 지우세요** — **E6** |

> 🔴 **가장 잘 놀라는 것이 마지막 행입니다.** 200스텝 스모크와 260스텝 재개만 했는데 `outputs/`가 **1.2 GB**가 됩니다(체크포인트 2개). `--save_freq` 기본값 20,000으로 100k 스텝을 돌리면 5개 ≈ **3.0 GB**입니다.

### 0.2 선수 확인 — 세 가지

**① ★ 인터프리터. 이 모듈만 별도 가상환경을 씁니다.**

> 🔴 **W1의 `.venv`(torch 2.13)와 W2의 `.venv-lerobot`(torch 2.11)은 다른 환경이고, 섞으면 안 됩니다.**
> LeRobot 0.6.1이 자기 torch 버전을 끌고 오기 때문입니다. 하나의 venv에 밀어 넣으면 W1 실습이 깨집니다.
> **W1 practice를 돌릴 때는 `.venv`, 이 랩에서는 `.venv-lerobot`** — 매 명령의 인터프리터를 명시적으로 씁니다.

아직 없다면 만듭니다. **실측 42초**입니다.

```bash
cd /path/to/physical-ai-study
uv venv --python 3.12 .venv-lerobot
uv pip install --python .venv-lerobot/bin/python "lerobot[training,viz]"
```

`uv`가 없으면 `python3.12 -m venv .venv-lerobot && .venv-lerobot/bin/pip install "lerobot[training,viz]"` 로도 됩니다(더 느립니다).

| extra | 들어오는 것 | 왜 필요한가 |
|---|---|---|
| `training` | `dataset` + `accelerate` + `wandb` | Step 5의 `lerobot-train` |
| `viz` | `rerun-sdk` | Step 4의 `lerobot-dataset-viz` |

> 📌 **이 문서는 `source activate`를 쓰지 않습니다.** `.venv-lerobot/bin/python` 처럼 **경로를 직접 씁니다.** 활성화한 셸을 잊고 다른 창에서 명령을 치는 것이 이 랩에서 가장 흔한 사고이고, 경로를 박아두면 그 사고가 원천 차단됩니다.

**② 이 저장소의 `.gitignore` 확인.** 이 랩은 대용량 산출물을 만듭니다.

```bash
grep -nE "^outputs/|rrd|venv-lerobot" .gitignore
```

```
225:artifacts/**/*.rrd
232:outputs/
250:.venv-lerobot/
```

세 줄이 다 있어야 합니다 — **`.rrd`(Step 4의 25 MB)** · **`outputs/`(Step 5의 체크포인트 1.2 GB)** · **`.venv-lerobot/`(5.8 GB)**. 줄 번호는 달라도 됩니다.

**③ W2-M1 practice 완주 여부.** Step 8이 practice의 계산 결과를 이어받습니다.

```bash
ls artifacts/W2-M1/0*_*.csv
```

```
artifacts/W2-M1/01_latency_budget.csv  artifacts/W2-M1/03_compounding_error.csv
artifacts/W2-M1/02_ensemble_weights.csv  artifacts/W2-M1/04_act_cvae_log.csv
```

없으면 먼저 돌리세요. **의존성이 없어 어느 환경에서든 즉시 돕니다**([practice README §2.1](../practice/README.md)).

```bash
cd course/w2-policy-vla/01-imitation-learning-act/practice
python3 01_latency_budget.py --sweep && python3 02_ensemble_weights.py && python3 03_compounding_error.py
```

> 🔴 **practice `01`이 계산한 지연 예산 표는 lesson의 30/50 Hz 병기를 그대로 옮긴 것입니다.** 이 랩 Step 3이 **fps를 확정**하고, Step 6이 **τ_infer를 측정**합니다. Step 8.A에서 그 두 값으로 표를 다시 채우는 것이 이 모듈의 결론입니다.

### 0.3 실행 시간 — 집필 환경 실측

명령이 실제로 도는 시간입니다. **합계가 GPU 약 3분 / CPU 약 14분**입니다.

| Step | 명령 | GPU | CPU | 비고 |
|---|---|---|---|---|
| 1 | `uv pip install "lerobot[training,viz]"` | **42초** | 동일 | 1회만. 이미 있으면 0초 |
| 1 | 임포트·CLI 확인 | **약 3초** | 동일 | |
| 2 | `verify_act_install.py --quick` | **약 7초** | 동일 | 대부분 torch 임포트입니다. 첫 실행은 **약 11초** |
| 3 | `verify_act_install.py --quick --with-datasets` | **약 10초**(캐시 있을 때) | 동일 | **첫 다운로드 40.6초**(1.74 GB) |
| 4 | `lerobot-dataset-viz --save 1` | **8.8초** | 동일 | 산출 `.rrd` **25.0 MB** |
| 5 | `lerobot-train --steps=200` | **31.3초** | **약 9.7분** | ResNet18 가중치 44.7 MB 다운로드 포함 |
| 5 | 재개 `--resume=true --steps=260` | **약 15초** | 약 3분 | |
| 6 | `verify_act_install.py --compare-cpu` | **약 12초** | 약 30초 | CPU 측정이 포함돼 GPU 기기에서도 CPU 시간이 듭니다 |
| — | **합계** | **약 3분** | **약 14분** | 명령 실행만. 읽고 쓰는 시간은 0.4 |
| — | *(참고) 100k 스텝 본 학습* | *약 2.1시간* | *약 80시간* | 🔴 **이 모듈에서는 하지 않습니다**(lesson §8.1) |

> ⚠️ **CPU 열의 학습 시간은 20스텝 실측(68초)에서 환산한 값**입니다. 200스텝을 CPU로 끝까지 돌린 것은 아닙니다.
> 📌 **"GPU가 없어서 못 한다"가 성립하지 않는 랩입니다.** 유일하게 느려지는 것이 Step 5이고 그것도 10분입니다. 오히려 CPU가 있으면 **Step 6의 GPU/CPU 대비표가 더 풍부해집니다**(6.4).

### 0.4 시간 예산

| Step | 내용 | 명령 실행 | 예상 소요 | 누적 | 성격 |
|---|---|---|---|---|---|
| 1 | 환경 관문 — `.venv-lerobot` | 42초 | **5분** | 5m | 붙여넣기 |
| 2 | **ACT 기본값 21개 자동 대조 ★** | 7초 | **8분** | 13m | ★ 출력 읽기 |
| 3 | **데이터셋 3종 메타 → fps 확정 ★★** | 10~45초 | **7분** | 20m | ★★ **§3.6 두 열의 정체가 갈리는 자리** |
| 4 | `dataset-viz` headless | 8.8초 | **5분** | 25m | 붙여넣기 |
| 5 | **학습 스모크 200스텝 + 재개 ★** | 31초 / 10분 | **10분** | 35m | ★ 손실 곡선 + KL 손계산 |
| 6 | **추론 경로 실측 ★★** | 12초 | **5분** | **40m** | ★★ **τ_infer가 여기서 나옵니다** |
| 7 | **★ 백지 ACT 아키텍처 (excalidraw)** | — | **30분** | 1h 10m | ★ **코드 없음. 전부 손** |
| 8 | **★★ 제출물 두 개 (§3.6 재계산 · 완료 기준)** | — | **55분** | **약 2시간 5분** | ★★ **코드 없음. 전부 글** |

**Step 1~6 합계가 40분, Step 7~8이 85분입니다.** CPU라면 Step 5가 10분 늘어 전체 약 2.5시간입니다.

> Step 8에서 55분을 넘겼다면 **잘 하고 있다는 뜻**입니다. 반대로 Step 7·8을 합쳐 40분에 끝냈다면 lesson을 열어놓고 베낀 것은 아닌지 확인하세요.

### 0.5 검증 환경

이 문서의 출력 예시는 아래 환경에서 **직접 실행해 붙인 것**이고, **미검증 배지는 두 곳뿐입니다**(0.6).

| 항목 | 값 |
|---|---|
| 날짜 | **2026-08-08 ~ 09** |
| OS | **Ubuntu on WSL2** (Linux 6.6.87.2-microsoft-standard-WSL2) |
| Python | **3.12.12** (uv 관리) |
| 가상환경 | **리포 루트 `.venv-lerobot/`** — 5.8 GB |
| `lerobot` | **0.6.1** |
| `torch` | **2.11.0+cu130** · `torch.cuda.is_available() == True` |
| `torchvision` / `torchcodec` / `av` | 0.26.0 / 0.11.1(**로드 실패 → pyav 폴백**) / 15.1.0 |
| `rerun-sdk` | 0.33.1 |
| GPU | **RTX 3080 12GB** — 이 랩에서는 선택 사항 |
| `ffmpeg` | **설치돼 있지 않음** (그래서 E1이 납니다. 그래도 전 절차 정상) |

> ⚠️ **벽시계(ms)는 실행마다 크게 흔들립니다. 이 랩에서 가장 조심할 지점입니다.**
> τ_infer는 집필 중 같은 GPU·같은 스크립트로 **6.5 / 8.1 / 11.9 / 13.0 / 22.9 / 66.0 ms** 가 나왔습니다. CPU는 **92 ~ 189 ms** 입니다. 원인 둘입니다 — ① **유휴 GPU의 SM 클럭이 0 MHz까지 내려갔다가 올라오는 시간**(그래서 스크립트가 워밍업 12회 뒤 **중앙값**을 씁니다) ② **같은 기기에서 다른 작업이 돌 때**(66 ms가 나온 실행은 load average 12였습니다).
> 그래서 스크립트가 **측정 시점의 load average를 함께 찍습니다.** 부하가 높으면 조용해진 뒤 다시 재세요.
> 🔴 **판정은 절대값이 아니라 배율과 부등호로 합니다.** 예외가 셋 — **하이퍼파라미터 21개(정확히 일치)** · **z=0 결정론성(정확히 0)** · **청크 shape `(1, 100, 14)`** 는 값이 맞아야 합니다.

### 0.6 미검증으로 남는 것 — 두 가지뿐

> ⚠️ **미검증(sudo 필요)** — E1의 근본 처방인 `sudo apt install ffmpeg`는 집필 시점에 **실행 검증되지 않았습니다.**
> 권한이 필요해서입니다. **처방을 안 써도 이 랩은 전부 돕니다** — pyav 폴백으로 Step 3·4·5가 전부 정상 완주했습니다.
> 실행했다면 결과를 [`../../../../docs/progress.md`](../../../../docs/progress.md)에 기록하고 이 배지를 제거하세요.

> ⚠️ **미측정(하드웨어 없음)** — **Jetson급 온보드 SoC의 추론 지연은 측정하지 않았습니다.**
> Step 6의 CPU 열은 **데스크톱 x86 CPU**이고, 온보드 SoC와는 다른 이야기입니다. lesson §7 세 번째 오해가 "온보드 추론이면 33 ms 안에 끝나야 한다"고 조건부로 남긴 자리는 **이 랩에서도 열려 있습니다**(팀 질문 `M1-6`).

그 외 **설치 · 데이터셋 · 시각화 · 학습 스모크 · 재개 · 추론 실측은 전부 실행 검증됐습니다.** 배지가 없습니다.

### 0.7 통과 기준 — 여기까지 되면 W2-M2로 넘어가도 된다

**이 7개가 전부 `[x]`면 이 모듈은 끝난 것입니다.**

- [ ] **G1** — `verify_act_install.py --quick`이 **`21/21 PASS`** 와 **`FAIL 0`** 를 찍었다 *(Step 2)*
- [ ] **G2 ★★** — `lerobot/aloha_sim_transfer_cube_human`의 **`fps=50`** 을 자기 눈으로 확인했고, lesson §3.6 표의 **@50 Hz 열과 @30 Hz 열이 각각 무엇인지** 말할 수 있다 *(Step 3)*
- [ ] **G3** — `lerobot-dataset-viz --save 1`이 **`.rrd` 25 MB**를 만들었고, 뷰어를 한 번도 띄우지 않았다 *(Step 4)*
- [ ] **G4 ★** — `lerobot-train --steps=200`이 **loss 24.4 → 3.8**로 내려가는 것을 봤고, **KL 기여율을 손으로 계산**했다(96.8% → 86.0%) *(Step 5)*
- [ ] **G5 ★★** — **τ_infer**를 측정했고 `select_action` 1회차와 2회차의 **배율(한 자릿수 이상)** 이 왜 그런지 설명할 수 있다 *(Step 6)*
- [ ] **G6 ★** — excalidraw 워크시트를 **lesson §4.1·§4.2를 닫아둔 채** 채웠고, 청크 shape이 **`(1, 100, 14)`** 임을 대조로 확인했다 *(Step 7)*
- [ ] **G7 ★★** — worksheet ⑥에 **§3.6 표를 자기 숫자로 다시 채웠고**, ⑦에 **완료 기준 (a)(b)(c)를 한 문단**으로 썼다 *(Step 8)*

> 🔴 **G2와 G7이 이 모듈의 중심입니다.**
> G2 없이 G7을 하면 lesson의 30 Hz 가정을 그대로 베끼게 되고, G7 없이 G2만 하면 "fps가 50이더라"로 끝납니다.

### 0.8 "붙여넣으면 되는 것" vs "판단이 필요한 것"

| | 그냥 붙여넣으세요 | 판단이 필요합니다 |
|---|---|---|
| **Step 1** | `uv venv` · `uv pip install` | (없음) |
| **Step 2** | `verify_act_install.py --quick` | **PASS/WARN/FAIL 셋을 어떻게 가를 것인가** — 설치 문제인가 문서 문제인가 |
| **Step 3** | `--with-datasets` | ★★ **§3.6 표의 두 열이 각각 무엇인가.** 기록 FPS와 배포 $f_2$는 같은 양인가 |
| **Step 4** | `--save 1 --output-dir ...` | (없음 — 뜨면 통과) |
| **Step 5** | `lerobot-train ... --steps=200` | ★ **KL 기여율이 96.8%인 것을 "KL이 지배한다"로 읽을 것인가** |
| **Step 6** | `--compare-cpu` | ★★ **앙상블이 GPU에서 통과한 것을 "앙상블은 싸다"로 읽을 것인가** |
| **Step 7·8** | (없음) | **전부** |

**판단 칸에서 막히면 정상입니다.** 붙여넣기 칸에서 막히면 [§흔한 에러](#흔한-에러와-대처--7개)를 보세요.

### 0.9 설치·인자 상세는 practice README에

지연 예산 계산기의 인자, 앙상블 가중 분석, compounding error 토이 시뮬은 **[`../practice/README.md`](../practice/README.md)에 이미 정리돼 있습니다.** 이 랩은 그걸 반복하지 않고 **"LeRobot을 어떤 순서로 돌리고, 어떤 출력이 나와야 정상이며, 그것으로 무엇을 계산해야 하는지"** 만 다룹니다. 두 문서를 나란히 열어두세요.

---

## Step 1 — 환경 관문: `.venv-lerobot` (5분)

검증하는 lesson 절: [§8.2 환경과 커맨드](../lesson.md).

### 1.1 명령

```bash
cd /path/to/physical-ai-study

# ① 설치 (이미 있으면 건너뛰세요) — 실측 42초
uv venv --python 3.12 .venv-lerobot
uv pip install --python .venv-lerobot/bin/python "lerobot[training,viz]"

# ② 임포트와 버전
.venv-lerobot/bin/python -c "import lerobot, torch, sys; print(sys.version.split()[0], lerobot.__version__ if hasattr(lerobot,'__version__') else '', torch.__version__, torch.cuda.is_available())"

# ③ CLI가 깔렸는가
ls .venv-lerobot/bin/ | grep '^lerobot-'

# ④ ★ W1 환경과 섞이지 않았는지 — 이쪽에는 lerobot이 없어야 정상입니다
.venv/bin/python -c "import lerobot" 2>&1 | tail -1
```

### 1.2 성공 판정 기준

**②**

```
3.12.12  2.11.0+cu130 True
```

**③ CLI 18개가 나옵니다.**

```
lerobot-annotate            lerobot-info
lerobot-calibrate           lerobot-record
lerobot-dataset-viz         lerobot-replay
lerobot-edit-dataset        lerobot-rollout
lerobot-eval                lerobot-setup-can
lerobot-find-cameras        lerobot-setup-motors
lerobot-find-joint-limits   lerobot-teleoperate
lerobot-find-port           lerobot-train
lerobot-imgtransform-viz    lerobot-train-tokenizer
```

*(실제로는 한 줄에 하나씩 18줄이 나옵니다. 지면 때문에 두 열로 접었습니다.)*

**④**

```
ModuleNotFoundError: No module named 'lerobot'
```

| 판정 항목 | 기대 | 아니면 |
|---|---|---|
| Python | **3.12.x** | 3.13 이상이면 LeRobot 설치가 막힙니다. `--python 3.12`를 다시 주세요 |
| `torch` | **2.11.0+cu130** (버전은 달라도 됨) | lesson §8.2는 "PyTorch 2.10 이상"이 요구조건이라고 적었습니다 |
| `cuda.is_available()` | **`True` 또는 `False` 모두 정상** | `False`면 0.3의 CPU 열을 보세요. 이 랩은 완주됩니다 |
| CLI 개수 | **18개** | `lerobot-train`·`lerobot-dataset-viz` 둘만 있으면 충분합니다 |
| **④의 `ModuleNotFoundError`** ★ | **이게 정상입니다** | 여기서 lerobot이 임포트되면 **W1 환경에 섞어 넣은 것**입니다 — **E7** |

### 1.3 여기서 볼 것 — CLI 목록이 이미 계보를 말합니다

```bash
.venv-lerobot/bin/lerobot-train --help 2>&1 | grep -A3 "policy.type"
```

`--policy.type`의 선택지에 **`act` · `diffusion` · `pi0` · `pi0_fast` · `pi05` · `groot` · `smolvla` · `vqbet` …** 가 함께 들어 있습니다.

> 🔴 **W2 전체가 이 한 줄 아래에 있습니다.** `act`(이 모듈) · `diffusion`(W2-M2) · `pi0`/`pi05`/`groot`(W2-M4)가 **같은 CLI·같은 데이터 포맷**을 씁니다. lesson §1.3 계보 그림이 커맨드 한 줄로 보이는 자리입니다.
> 목록에 **`lerobot-rollout --strategy.type` 의 `dagger`** 도 있습니다 — lesson §2.3이 "DAgger가 완전히 사라진 것은 아니다"라고 쓴 근거입니다.
> `lerobot-train-tokenizer`라는 CLI도 있습니다. **액션 토크나이저(W2-M5 / FSQ)와 접점일 수 있으나 이번에 내용을 확인하지 않았습니다.** 추측하지 말고 worksheet ⑧의 확인 항목으로 남기세요.

**worksheet ① 에 기록**: 설치 소요 시간, torch 버전, `cuda.is_available()`, CLI 개수.

---

## Step 2 — ACT 기본값 21개를 자동 대조 ★ (8분)

검증하는 lesson 절: [§6.4 하이퍼파라미터 표](../lesson.md) · [§2.5 · §2.6 docstring 인용](../lesson.md) · [§7 첫 오해](../lesson.md) · [§10 신뢰도 각주](../lesson.md).

lesson §6.4의 표 21행은 "LeRobot 문서를 읽고 옮겨 적은 값"입니다. **지금 손에 깔린 0.6.1이 정말 그런지**를 코드가 대조합니다.

### 2.1 명령

```bash
cd /path/to/physical-ai-study
.venv-lerobot/bin/python course/w2-policy-vla/01-imitation-learning-act/labs/verify_act_install.py --quick
echo "exit=$?"
```

> ⏱️ **예상 소요: 약 7초**(첫 실행은 torch 임포트 때문에 약 11초). `--quick`은 모델을 만들지 않아 GPU를 거의 안 씁니다.

### 2.2 성공 판정 기준

**① 맨 앞에 lerobot 경고 세 줄이 뜹니다. 정상입니다.**

```
WARNING:lerobot.configs.policies:Device 'None' is not available. Switching to 'cuda'.
```

`ACTConfig()`를 device 지정 없이 만들 때 lerobot이 내는 안내입니다. 대조에는 영향이 없습니다.

**② 환경 표.**

```
=== [1] 환경 — 무엇이 깔려 있는가 ===

  항목         값                       비고
  -----------  -----------------------  ----------------------------------------
  python       3.12.12                  /.../physical-ai-study/.venv-lerobot/bin/python
  lerobot      0.6.1                    lesson §8.2 기준
  torch        2.11.0+cu130             cuda_available=True
  torchvision  0.26.0                   ResNet-18 사전학습 가중치의 출처
  torchcodec   0.11.1                   로드 실패해도 무해 — 에러 표 E1
  av (pyav)    15.1.0                   torchcodec 실패 시의 폴백 디코더
  rerun-sdk    0.33.1                   viz extra. Step 4에서 씁니다
  numpy        2.2.6
  GPU          NVIDIA GeForce RTX 3080  cuda 13.0
```

**③ ★ 하이퍼파라미터 21개 — 이 블록이 Step 2의 본체입니다.**

```
=== [2] lesson §6.4 하이퍼파라미터 21개 ↔ ACTConfig() 기본값 ===

  필드                                lesson §6.4      설치본  대조
  ----------------------------------  -----------  ----------  ----
  chunk_size                                  100         100  PASS
  n_action_steps                              100         100  PASS
  n_obs_steps                                   1           1  PASS
  vision_backbone                      'resnet18'  'resnet18'  PASS
  dim_model                                   512         512  PASS
  n_heads                                       8           8  PASS
  n_encoder_layers                              4           4  PASS
  n_decoder_layers                              1           1  PASS
  n_vae_encoder_layers                          4           4  PASS
  latent_dim                                   32          32  PASS
  use_vae                                    True        True  PASS
  kl_weight                                    10          10  PASS
  dim_feedforward                            3200        3200  PASS
  dropout                                     0.1         0.1  PASS
  pre_norm                                  False       False  PASS
  feedforward_activation                   'relu'      'relu'  PASS
  replace_final_stride_with_dilation        False       False  PASS
  optimizer_lr                              1e-05       1e-05  PASS
  optimizer_lr_backbone                     1e-05       1e-05  PASS
  optimizer_weight_decay                   0.0001      0.0001  PASS
  temporal_ensemble_coeff                    None        None  PASS

  대조 결과: 21/21 PASS  ← lesson §6.4 표와 완전 일치
```

**④ ★ 영문 축자 인용 3건 — `../lesson.md`를 읽어 글자 단위로 대조합니다.**

```
=== [3] lesson의 영문 축자 인용 3건 ↔ 설치본 소스 원문 ===

  대조 기준: 설치본 `configuration_act.py` (정본) ↔ `lesson.md`

  #   설치본 위치               설치본 원문  lesson 표기  대조
  --  ------------------------  -----------  -----------  ----
  Q1  configuration_act.py:47   있음         일치         PASS
  Q2  configuration_act.py:75   있음         일치         PASS
  Q3  configuration_act.py:108  있음         일치         PASS

  → 세 인용문이 설치본 원문과 **글자 단위로 같습니다.** lesson을 읽을 때
     영문 블록을 그대로 믿어도 된다는 뜻입니다.
```

**⑤ 예외 클래스 — lesson §10 각주 1건이 여기서 닫힙니다.**

```
=== [4] temporal ensembling 오설정 — 예외 클래스 (lesson §10 각주) ===

  ACTConfig(temporal_ensemble_coeff=0.01, n_action_steps=100)  ← §2.6이 금지한 조합

    예외 클래스 : NotImplementedError   [PASS] (기대: NotImplementedError)
    메시지      : `n_action_steps` must be 1 when using temporal ensembling. This is because the policy needs to be queried every step to compute the ensembled action.
    n_action_steps=1 로 바꾸면 : 통과   [PASS]

  🔴 docstring 문장([3] Q2)과 예외 메시지는 **다른 문장**입니다. 둘을 섞지 마세요.
```

**⑥ 종합.**

```
================================================================================================
  종합 — PASS 28 · FAIL 0 · WARN 0 · INFO 0  (총 28건)
================================================================================================

  [저장] /.../artifacts/W2-M1/labs/verify_act_install.csv
exit=0
```

| 판정 항목 | 기대 | 판정 방법 |
|---|---|---|
| **21개 대조** | **`21/21 PASS`** | ★ **G1.** 하나라도 FAIL이면 lerobot 버전이 다릅니다 |
| 인용 Q1 (`chunk size size`) | **PASS** | 상류 오타가 **그대로 존재**합니다. lesson §2.5의 주석이 옳았습니다 |
| 인용 Q2 (앙상블 docstring) | **PASS** | |
| 인용 Q3 (`n_decoder_layers` 주석) | **PASS** | ★ 2.3에 사연이 있습니다 |
| 예외 클래스 | **`NotImplementedError`** | lesson §10 각주가 닫힌 근거 |
| **종합** | **FAIL 0 · WARN 0** | |
| **exit code** | **0** | FAIL이 있으면 1이 됩니다 |

### 2.3 ★ 여기서 판정할 것 — 이 스크립트가 실제로 잡아낸 것

> 🔴 **"전부 PASS니 볼 것 없다"로 넘기면 이 Step의 절반을 버립니다.**

이 스크립트는 두 종류를 구분합니다.

| | 무엇인가 | exit code | 무엇을 해야 하나 |
|---|---|---|---|
| **FAIL** | **설치본이 lesson과 다르다** | **1** | 환경을 의심하세요. 버전이 어긋났을 가능성이 큽니다 |
| **WARN** | **설치본은 정상인데 lesson 쪽 표기가 어긋난다** | 0 | **문서를 고칠 대상**입니다. worksheet ⑨에 적으세요 |

**Q3가 지금 PASS인 데는 사연이 있습니다.** 집필 검증 중 이 스크립트를 처음 돌렸을 때 Q3가 **WARN**이었습니다.

| | 문장 |
|---|---|
| 설치본 원문(정본, `configuration_act.py:108-109`) | `...there is a bug in the code` **`that means only`** `the first layer is used.` |
| 당시 lesson §7의 인용 | `...there is a bug in the code` **`that only`** `the first layer is used.` |

**`means` 한 단어가 빠져 있었고, 그래서 축자 인용이 아니었습니다.** lesson을 그 자리에서 고쳤고, 그래서 지금 돌리면 세 건 다 PASS입니다.

> **이 사연이 왜 이 랩에 실려 있는가.** lesson §7의 첫 오해가 정확히 *"논문에 적힌 하이퍼파라미터가 곧 돌아간 코드다"* 를 교정하면서 **"리포 투어 규약이 파일 경로:라인을 인용할 것을 요구하는 이유의 실물 사례"** 라고 씁니다. **그 문장을 담은 인용문 자체가 한 단어 어긋나 있었습니다.**
> 그리고 그것을 잡아낸 것은 문서를 다시 읽은 것이 아니라 **설치본과 기계적으로 대조한 것**입니다. 사람이 읽으면 `that only`도 그냥 읽힙니다.
> 🔴 **그래서 이 스크립트는 lesson의 문구를 상수로 베껴 두지 않고 `../lesson.md`를 매번 읽습니다.** 베껴 두면 lesson이 개정될 때 스크립트만 조용히 낡습니다 — 집필 중 실제로 한 번 그렇게 어긋났습니다.

**worksheet ③에 답할 것**: **"`that means only` → `that only`로 한 단어가 빠졌을 때, lesson §7의 주장 자체는 바뀌는가?"** 한 줄로 판정하세요. 그리고 **"바뀌지 않는데도 고쳐야 하는 이유"** 를 한 줄 더.

---

## Step 3 — 데이터셋 3종 메타 ★★ (7분)

검증하는 lesson 절: [§10 신뢰도 각주 2건](../lesson.md) · [§3.6 시간 환산](../lesson.md) · [§8.2 공개 데이터셋](../lesson.md).

**이 Step이 이 랩의 첫 번째 발견입니다.**

### 3.1 명령

```bash
cd /path/to/physical-ai-study
.venv-lerobot/bin/python course/w2-policy-vla/01-imitation-learning-act/labs/verify_act_install.py --quick --with-datasets
```

> ⏱️ **예상 소요: 첫 실행 40.6초**(`aloha_mobile_cabinet` 1.74 GB 다운로드) **· 이후 약 10초.**
> 🔴 **1.74 GB를 받고 싶지 않다면** 스크립트 상단의 `DATASETS` 리스트에서 그 줄을 지우세요. 다만 **카메라 3대 구성**이 그 데이터셋에만 있어 §4.1 블록도의 "토큰 수는 카메라 수에 비례" 대목을 눈으로 볼 기회를 잃습니다.

### 3.2 성공 판정 기준

**① 다운로드 진행 표시와 경고. 정상입니다.**

```
Fetching 4 files: 100%|██████████| 4/4 [00:00<00:00, 1684.80it/s]
WARNING:lerobot.datasets.utils:Unknown fields in DatasetInfo: ['files_size_in_mb']. These will be ignored.
```

HF 미인증 경고(`You are sending unauthenticated requests to the HF Hub...`)도 함께 뜰 수 있습니다 — **E4**. 공개 데이터셋은 그대로 받아집니다.

**② ★★ 메타 표 — 여기가 발견입니다.**

```
=== [7] 데이터셋 메타 — lesson §10 각주 'ACT/ALOHA 기록 FPS 미확인'을 닫는다 ===

  repo_id                                fps  에피소드   프레임  총 시간  구성
  -------------------------------------  ---  --------  -------  -------  ------------
  lerobot/aloha_sim_transfer_cube_human   50        50   20,000    6.7분  1대 · D_a=14
  lerobot/aloha_mobile_cabinet            50        85  127,500   42.5분  3대 · D_a=14
  lerobot/pusht                           10       206   25,650   42.8분  1대 · D_a=2

  🔴 **fps 열을 자기 눈으로 확인하는 것이 이 절의 목적입니다.** ...
```

| 판정 항목 | 기대 | 무엇을 뜻하는가 |
|---|---|---|
| **ALOHA 2종의 `fps`** ★★ | **둘 다 50** | **G2.** 3.3에서 판정 |
| **PushT `repo_id`** | **`lerobot/pusht` 존재** | lesson §10에 ✅로 적힌 항목. W2-M2가 쓸 데이터셋입니다 |
| PushT `fps` | **10** | ALOHA와 **5배 다릅니다.** 같은 `chunk_size=100`이 전혀 다른 시간이 됩니다 |
| `aloha_sim_transfer_cube_human` 에피소드 | **50** · 총 **6.7분** | ★ 3.4에서 판정 |
| `aloha_mobile_cabinet` 카메라 | **3대** (`cam_high`·`cam_left_wrist`·`cam_right_wrist`) | §4.1의 "이미지 토큰이 압도적으로 많다" |
| `D_action` | ALOHA **14** · PushT **2** | 양팔 7 DoF × 2 = 14. §4.1 출력 Linear의 차원 |

**③ 로컬 캐시 위치도 확인해두세요.**

```bash
du -sh ~/.cache/huggingface/lerobot/hub/datasets--lerobot--*
```

```
1.7G    /home/.../.cache/huggingface/lerobot/hub/datasets--lerobot--aloha_mobile_cabinet
67M     /home/.../.cache/huggingface/lerobot/hub/datasets--lerobot--aloha_sim_transfer_cube_human
7.5M    /home/.../.cache/huggingface/lerobot/hub/datasets--lerobot--pusht
```

> 데이터셋 메타가 보고하는 크기는 **7.7 MB / 69.9 MB / 1,741.6 MB** 이고, 디스크 실측은 위와 같습니다.
> **세 자릿수 차이가 나는 이유는 카메라 수 × 해상도 × 프레임 수**입니다 — PushT는 96×96 한 대, `aloha_mobile_cabinet`은 480×640 세 대입니다.
> 🔴 **캐시 경로가 `~/.cache/huggingface/lerobot/hub/` 아래**라는 것도 기억해두세요. lesson §8.2가 적어둔 `~/.cache/huggingface/lerobot/{repo-id}`와 실제 레이아웃이 다릅니다(HF Hub 표준 캐시 규약을 따릅니다). 지울 때 이 경로를 씁니다.

### 3.3 ★★ 여기서 판정할 것 — §3.6 표의 두 열은 서로 다른 로봇의 이야기다

> 🔴 **이 랩에서 가장 중요한 한 줄입니다. worksheet ⑤에 지금 적으세요.**

lesson [§3.6](../lesson.md)의 표에는 열이 둘 있습니다 — **@30 Hz**와 **@50 Hz**. 처음 읽으면 "확인이 안 돼서 둘 다 적어뒀나 보다"로 읽힙니다. **틀린 독해입니다.** lesson 본문이 이렇게 적어뒀습니다.

> **ACT/ALOHA 공개 데이터의 기록 FPS는 50 Hz로 확인됐습니다**(2026-08-09 실측). 그래도 두 열을 병기하는 이유가 있습니다 — 50 Hz 열은 *ALOHA 자신*의 개방루프 창이고, 30 Hz 열은 *우리 로봇의 $f_2$를 30 Hz로 가정*했을 때입니다.

**방금 확인한 `fps=50`이 그 문장의 앞부분입니다.** 그러니 두 열의 정체를 이렇게 갈라 두세요.

| | **@50 Hz 열** | **@30 Hz 열** |
|---|---|---|
| 어디서 온 값인가 | **데이터셋 메타의 `fps`** — 방금 봤습니다 | **가정**. 우리 로봇의 소비 주파수 $f_2$ |
| 무엇을 말하는가 | **ALOHA 자신**이 청크 100개를 실행하는 데 걸리는 시간 = **2,000 ms** | **우리 로봇**에 같은 설정을 얹었을 때 = **3,333 ms** |
| 200 ms 문턱과 맞대는가 | 아니오 (ALOHA에는 낙상 모드가 없음) | **예 — 16.7배 밖** |
| 확정 상태 | ✅ **실측** | 📎 **가정** — 실제 값은 팀 질문 `M3-5` |

> 🔴 **여기가 이 Step의 핵심입니다.** **기록 FPS(데이터가 만들어진 주파수)와 배포 $f_2$(정책 출력이 소비되는 주파수)는 다른 양입니다.** 같을 수도 있고 다를 수도 있고, 다르면 액션을 리샘플링하거나 청크 길이를 환산해야 합니다.
> 그것을 묻는 것이 팀 질문 [`W2M1-2`](../lesson.md)입니다 — *"시연 기록 FPS와 배포 시 하위 소비 주파수 $f_2$가 같은가?"* **방금 왼쪽 항의 값을 손에 넣었고, 오른쪽 항은 여전히 모릅니다.**

> 📌 **정확히 무엇이 확정됐는지 선을 지키세요.**
> 확정된 것은 **"LeRobot이 배포하는 ALOHA 데이터셋 2종의 `fps`가 50이다"** 입니다.
> **ACT 논문 원 수집 장비의 기록 주파수까지 같은지는 논문을 봐야 알 수 있고, 이번에 보지 않았습니다.** 학습 파이프라인이 실제로 쓰는 값이 50이라는 것까지가 이 랩의 사실입니다.

### 3.4 여기서 볼 것 — "적은 데이터"의 기준점

`aloha_sim_transfer_cube_human`이 **50 에피소드 · 20,000 프레임 · 총 6.7분**입니다.

lesson [§8.2·§10](../lesson.md)이 인용한 ACT 초록의 **"10분 분량 시연 · 50 시연"** 과 **자릿수가 같습니다.**

> 🔴 **로봇 학습에서 "적은 데이터"가 어느 정도인지의 기준점입니다.** LLM 파인튜닝 감각으로 오면 "6.7분짜리 데이터로 뭘 하나" 싶은데, 이 분야에서는 그것이 **정상적인 시작점**입니다. 사람이 로봇을 직접 몰아야 하므로 데이터가 시간에 선형으로 비쌉니다(lesson §2.1·§5.2).
> 그래서 팀 질문 [`W2M1-1`](../lesson.md)("확보된 에피소드 수와 시간 규모는?")이 **W2 실습의 방향을 가르는 질문**입니다 — 자릿수가 같으면 재현이고, 다르면 다른 이야기가 됩니다.

**worksheet ⑤에 기록**: 세 데이터셋의 fps·에피소드·총 시간, 그리고 3.3의 표를 자기 숫자로.

---

## Step 4 — 데이터셋 시각화, 뷰어 없이 (5분)

검증하는 lesson 절: [§8.2 `lerobot-dataset-viz`](../lesson.md) · [§4.1 관측 3종](../lesson.md).

### 4.1 명령

> 🔴 **`--save 1`을 반드시 주세요.** 이것이 뷰어 spawn을 끕니다. 빼면 GUI를 띄우려 하고 헤드리스에서는 실패합니다. 이 저장소의 "뷰어를 띄우는 코드 금지" 규약과 정확히 맞물리는 플래그입니다.

```bash
cd /path/to/physical-ai-study
.venv-lerobot/bin/lerobot-dataset-viz \
  --repo-id lerobot/aloha_sim_transfer_cube_human \
  --episode-index 0 \
  --save 1 \
  --output-dir artifacts/W2-M1/labs
```

> ⏱️ **예상 소요: 8.8초.** `--repo-id`와 `--episode-index`는 **둘 다 필수 인자**입니다 — **E5**.

### 4.2 성공 판정 기준

**① 붉은 traceback 수십 줄이 먼저 쏟아집니다. 🔴 이것이 정상입니다.**

```
OSError: libavutil.so.56: cannot open shared object file: No such file or directory
...
[end of libtorchcodec loading traceback].
'torchcodec' is installed but cannot be loaded (see the error above). Falling back to 'pyav' as a default decoder.
```

> 🔴 **입문자가 이 랩에서 가장 크게 오해하는 지점입니다.** 화면이 빨갛게 물들어 "설치 실패"로 읽히지만, **마지막 줄을 보세요** — `Falling back to 'pyav' as a default decoder.` 폴백이 성공했다는 뜻이고, 그 뒤 모든 절차가 정상 동작합니다. 상세는 **E1**.

**② 산출물이 생깁니다.**

```bash
ls -la artifacts/W2-M1/labs/
```

```
-rw-r--r-- 1 ... 24965119 ... lerobot_aloha_sim_transfer_cube_human_episode_0.rrd
```

| 판정 항목 | 기대 | 아니면 |
|---|---|---|
| 종료 코드 | **0** | argparse 에러면 **E5** — 인자 두 개 다 줬는지 |
| `.rrd` 파일 | **약 25.0 MB** | 0바이트면 `--save 1`이 안 먹은 것 |
| traceback | **떠도 정상** | 마지막 줄에 `Falling back to 'pyav'`가 있어야 합니다 — **E1** |
| 뷰어 창 | **뜨면 안 됩니다** | `--save 1`을 빠뜨렸습니다 |

### 4.3 `.rrd`를 실제로 보려면

**이 랩의 통과 기준(G3)에는 파일을 여는 것이 들어 있지 않습니다.** 25 MB를 내려받는 데 시간이 들고, 이 모듈의 무게중심은 Step 7·8이기 때문입니다. 다만 **한 번은 보는 것을 권합니다** — 데이터셋이 무엇인지 손에 잡히는 유일한 시각 자료입니다.

```bash
# (a) 로컬 PC에서 파일을 가져와 rerun으로 열기
scp <user>@<host>:/path/to/artifacts/W2-M1/labs/*.rrd .
pip install rerun-sdk==0.33.1 && rerun lerobot_aloha_sim_transfer_cube_human_episode_0.rrd

# (b) 원격에서 스트리밍 — 포트를 열어야 합니다
.venv-lerobot/bin/lerobot-dataset-viz --repo-id lerobot/aloha_sim_transfer_cube_human \
  --episode-index 0 --mode distant
# 로컬에서:  rerun rerun+http://<IP>:<GRPC_PORT>/proxy
```

> 📌 **`.rrd`는 대용량이라 `.gitignore`의 `artifacts/**/*.rrd`로 제외돼 있습니다.** 커밋되지 않습니다.
> 📌 **파일을 열면 볼 것**: 위쪽에 카메라 프레임(480×640), 아래쪽에 `observation.state`와 `action` 14채널의 시계열. **두 곡선이 거의 겹칩니다** — 텔레옵에서 리더 팔의 명령이 곧 팔로워의 다음 상태이기 때문이고, lesson §2.1의 "액션 라벨이 공짜로 생긴다"가 그림으로 보이는 자리입니다.

---

## Step 5 — ACT 학습 스모크 200스텝 ★ (10분)

검증하는 lesson 절: [§3.4 CVAE loss](../lesson.md) · [§8.1 범위 경계](../lesson.md) · [§8.2 학습 커맨드](../lesson.md).

> 🔴 **이 Step의 목표는 "학습이 시작되는 것"이지 "학습이 끝나는 것"이 아닙니다.** lesson §8.1이 그 경계를 못박았습니다 — 완주·평가·롤아웃은 **W2-M2**(집필 예정)의 몫이고, 여기서 완주하려 들면 다음 모듈의 무게중심을 먼저 소모합니다.

### 5.1 명령

```bash
cd /path/to/physical-ai-study
.venv-lerobot/bin/lerobot-train \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/act_w2m1_smoke \
  --job_name=act_w2m1_smoke \
  --batch_size=8 --steps=200 --log_freq=25 --save_freq=200 \
  --wandb.enable=false
```

> ⏱️ **예상 소요: GPU 31.3초 / CPU 약 9.7분.** CPU라면 `--policy.device=cpu`로 바꾸세요.
> 🔴 **`--policy.push_to_hub=false`가 없으면** lesson §8.2 커맨드처럼 `--policy.repo_id=<HF 네임스페이스>/...`를 요구합니다 — **E3**. HF 계정이 없어도 되게 하는 플래그입니다.
> 🔴 **`--save_freq`를 총 스텝 수 이상으로 주세요.** 체크포인트 하나가 **592 MB**입니다 — **E6**.
> 📌 CPU라면 [W1-M2 랩 §1.5](../../../w1-generative-core/02-simulator-bootcamp/labs/README.md)의 tmux를 쓰세요. 10분짜리라 필수는 아닙니다.

### 5.2 성공 판정 기준

**① 모델이 만들어지고 파라미터 수가 찍힙니다.**

```
INFO ... Creating policy
INFO ... Output dir: outputs/train/act_w2m1_smoke
INFO ... Effective batch size: 8 x 1 = 8
INFO ... num_learnable_params=51613582 (52M)
```

ResNet18 사전학습 가중치 **44.7 MB**를 처음 한 번 내려받습니다.

**② ★ 손실 로그 8줄. 이 블록이 Step 5의 본체입니다.**

```
step:25 smpl:200 ep:0 epch:0.01 loss:24.430 grdn:387.896 lr:1.0e-05 updt_s:0.074 data_s:0.002
step:50 ... loss:7.772  grdn:174.753 ...
step:75 ... loss:5.739  grdn:142.276 ...
step:100 ... loss:4.950 grdn:122.973 ...
step:125 ... loss:4.493 grdn:117.926 ...
step:150 ... loss:4.229 grdn:107.448 ...
step:175 ... loss:3.919 grdn:105.914 ...
step:200 ... loss:3.756 grdn:101.213 ...
```

집필 검증 실측(`l1_loss`·`kld_loss`는 로그에 함께 찍힙니다):

| step | `loss` | `grdn` | `l1_loss` | `kld_loss` |
|---|---|---|---|---|
| 25 | **24.430** | 387.896 | 0.784 | 2.365 |
| 50 | 7.772 | 174.753 | 0.669 | 0.710 |
| 100 | 4.950 | 122.973 | 0.606 | 0.434 |
| 150 | 4.229 | 107.448 | 0.590 | 0.364 |
| **200** | **3.756** | 101.213 | **0.528** | **0.323** |

기타 로그 값: `updt_s ≈ 0.074` · `data_s ≈ 0.002` · `smp/s ≈ 105` · **`mem_gb = 2.10`** · `epch = 0.08` · **첫 스텝만 6.75초**(워밍업), 이후 **약 13 step/s**.

**③ 체크포인트가 저장되고 정상 종료합니다.**

```bash
ls -la outputs/train/act_w2m1_smoke/checkpoints/
du -sh outputs/train/act_w2m1_smoke/checkpoints/000200
```

```
drwxr-xr-x ... 000200
lrwxrwxrwx ... last -> 000200
592M    outputs/train/act_w2m1_smoke/checkpoints/000200
```

| 판정 항목 | 기대 | 판정 방법 |
|---|---|---|
| `num_learnable_params` | **51613582 (52M)** | ★ **Step 6 [5]와 정확히 같은 값이고 lesson 본문의 "약 52M"과 일치합니다.** HF 문서의 80M과는 다릅니다 — 6.3에서 판정 |
| **loss 방향** | **24.4 → 3.8로 단조 감소** | ★ **G4.** 절대값은 흔들려도 됩니다. **내려가는 것**이 판정 기준입니다 |
| `grdn` (grad norm) | **388 → 101** | 함께 내려가야 정상. 발산하면 학습률·배치를 의심 |
| `mem_gb` | **약 2.10** | batch 8 · 1카메라 기준. 12 GB GPU에 넉넉합니다 |
| `updt_s` | **약 0.074**(GPU) / **약 2.86**(CPU) | 약 39배 차이 |
| 체크포인트 | **592 MB** | ★ **E6.** `model.safetensors`만 206.8 MB |
| 종료 코드 | **0** | |

### 5.3 ★ 여기서 계산할 것 — KL 기여율 (손으로)

> 🔴 **이 계산이 Step 5의 진짜 산출물입니다. 로그를 보고 손으로 하세요.**

lesson [§3.4](../lesson.md)가 이렇게 씁니다.

$$\mathcal{L}_{\text{ACT}} = \underbrace{\|\hat a - a\|_1}_{\texttt{l1\_loss}} + \beta \cdot \underbrace{D_{\mathrm{KL}}}_{\texttt{kld\_loss}}, \qquad \beta = \texttt{kl\_weight} = 10.0$$

그리고 **"읽는 순서 주의"** 로 못박습니다 — *"`kl_weight=10.0`이라는 숫자만 보고 'KL이 10배 중요하다'로 읽으면 안 됩니다. 두 항의 실제 크기를 학습 로그에서 따로 봐야 합니다."*

**이제 그 로그가 손에 있습니다. 채우세요.**

| step | `l1_loss` | `kld_loss` | $\beta \times$ `kld_loss` | 합 | **KL 기여율** |
|---|---|---|---|---|---|
| 25 | 0.784 | 2.365 | $10 \times 2.365 = 23.65$ | 24.43 | **96.8 %** |
| 200 | 0.528 | 0.323 | $10 \times 0.323 = 3.23$ | 3.76 | **86.0 %** |

두 가지를 확인하세요.

1. **$\ell_1 + 10 \times$ KL 이 `loss` 열과 맞습니까?** $0.784 + 23.65 = 24.43$ ✓ · $0.528 + 3.23 = 3.76$ ✓ → **lesson §3.4의 loss 조립식이 실물에서 그대로 성립합니다.**
2. **KL 기여율이 96.8% → 86.0%로 내려갑니다.** [practice `04`](../practice/README.md)가 토이 CVAE에서 본 곡선(step 1의 **99.1%** → step 3000의 **7.0%**)의 **앞부분 그대로**입니다.

> 🔴 **200 스텝은 너무 일러서 아직 역전 전입니다.** practice `04`는 3,000 스텝에서 7.0%까지 내려갔습니다. **즉 "KL이 loss를 지배한다"는 관찰은 학습 초기의 현상이고, 수렴하면 $\ell_1$이 지배합니다.** 여기서 "$\beta=10$이 너무 크다"고 결론 내리면 틀립니다 — 두 항의 스케일이 학습 중 크게 바뀌기 때문에 **가중치만 보고는 판단할 수 없다**는 것이 lesson §3.4의 요지이고, 그것이 실물 LeRobot ACT에서 재현된 것입니다.

**worksheet ⑤에 기록**: 위 표를 자기 로그로 다시 채우고, **"몇 스텝쯤에서 역전될 것 같은가"** 를 practice `04` 곡선에서 외삽해 적으세요.

### 5.4 재개가 되는지 확인 (2분)

lesson §8.2의 재개 커맨드가 그대로 도는지 봅니다.

```bash
.venv-lerobot/bin/lerobot-train \
  --config_path=outputs/train/act_w2m1_smoke/checkpoints/last/pretrained_model/train_config.json \
  --resume=true --steps=260
```

```bash
ls outputs/train/act_w2m1_smoke/checkpoints/
```

```
000200  000260  last
```

| 판정 항목 | 기대 | 아니면 |
|---|---|---|
| 종료 코드 | **0** | |
| `checkpoints/000260` | **생깁니다** | |
| `last` | **`-> 000260`** 심볼릭 링크 | 용량 0입니다 |

> 🔴 **`--steps`를 늘려 주지 않으면 즉시 끝납니다.** 이미 도달한 스텝이라 할 일이 없기 때문입니다. 재개 커맨드는 "이어서 더 돌린다"이지 "다시 돌린다"가 아닙니다.
> 🔴 **여기서 디스크가 1.2 GB가 됩니다.** 체크포인트 2개 × 592 MB — **E6**. 확인이 끝났으면 지우세요.

```bash
du -sh outputs/train/act_w2m1_smoke     # 1.2G
# rm -rf outputs/train/act_w2m1_smoke   # 다 봤으면
```

### 5.5 여기서 멈추는 이유

**13 step/s에서 100k 스텝을 외삽하면 약 2.1시간**(RTX 3080 · batch 8 · 1카메라)입니다. lesson §8.2가 인용한 "단일 GPU 수 시간"과 정합합니다. CPU라면 **약 80시간**이라 비현실적입니다.

> 🔴 **하지 마세요.** lesson §8.1의 경계 표가 "학습 100k 스텝 완주"를 **이 모듈에서 하지 않는 것**으로 지정했습니다. 완주·평가·롤아웃은 W2-M2이고, 그쪽이 PushT로 끝까지 갑니다.
> **"파이프라인이 돌기 시작하는 것까지 확인하고 멈추는 것이 설계된 종료 지점입니다."**

---

## Step 6 — 추론 경로 실측 ★★ (5분)

검증하는 lesson 절: [§3.6 지연 예산](../lesson.md) · [§2.5 두 숫자의 구분](../lesson.md) · [§2.7 $z=0$](../lesson.md) · [§4.1 블록도](../lesson.md) · [§7 세 번째 오해](../lesson.md) · [§8.2·§10 파라미터 수](../lesson.md).

**Step 8.A의 재료가 전부 여기서 나옵니다.**

### 6.1 명령

```bash
cd /path/to/physical-ai-study
.venv-lerobot/bin/python course/w2-policy-vla/01-imitation-learning-act/labs/verify_act_install.py --compare-cpu
```

> ⏱️ **예상 소요: 약 12초**(GPU 기기). CPU 단독 기기면 `--compare-cpu`가 자동으로 생략되고 약 30초입니다.

### 6.2 성공 판정 기준

**① ★ 파라미터 수 — 카메라를 늘려도 늘지 않습니다.**

```
=== [5] 파라미터 수 — 카메라를 늘리면 몇 개가 늘어나는가 ===

  카메라 수  학습가능 파라미터  1대 대비 증분      ≈
  ---------  -----------------  -------------  -----
          1         51,613,582             +0  51.6M
          2         51,613,582             +0  51.6M
          3         51,613,582             +0  51.6M
          4         51,613,582             +0  51.6M

    use_vae=True   : 51,613,582
    use_vae=False  : 34,232,142
    차이 (VAE posterior 인코더) : 17,381,440  ≈ 17.4M   ← 추론에서는 안 쓰입니다 (§2.7·§4.2)

    실측                 : 51.6M (51,613,582)
    lesson 본문 주장     : 약 52M   [PASS]  (출처: lesson.md)
    HF 문서 주장         : 약 80M   ❌ 실측과 불일치
```

**② ★ 추론 실측 5종.**

```
=== [6] 추론 경로 실측 — lesson §3.6의 τ_infer 자리를 채운다 ===

  (a) predict_action_chunk(obs).shape = (1, 100, 14)   [PASS] 기대 (1, 100, 14) = [B, chunk_size, D_action]

  (b) τ_infer (청크 1회 생성, 20회 중앙값) = **12.8 ms**
      (측정 시점 load average 1.7 / 5.9 · CPU 20코어)

      lesson §3.6 표의 예산에 이 값을 넣으면:
  설정                       n_act     f2  예산[ms]  τ_infer 대비  들어가나
  -------------------------  -----  -----  --------  ------------  --------
  LeRobot 기본값               100  50 Hz     2,000  155.8배 여유  ✅
  LeRobot 기본값               100  30 Hz     3,333  259.6배 여유  ✅
  docstring 예시                50  50 Hz     1,000   77.9배 여유  ✅
  짧은 receding horizon         10  50 Hz       200   15.6배 여유  ✅
  temporal ensembling(강제)      1  50 Hz        20    1.6배 여유  ✅
  temporal ensembling(강제)      1  30 Hz        33    2.6배 여유  ✅

      → **6행 전부 여유입니다.** lesson §3.6 첫째 결론('지연은 제약이 아니다')이 자기 기기 숫자로 확정되는 자리입니다.
         worksheet ⑥ 6.1로 옮기세요.

  (c) select_action 1~6회차 [ms] : 10.04 / 1.07 / 0.33 / 0.29 / 0.27 / 0.25
      1회차 ÷ 이후 평균 = **23배**  ← 1회차만 추론이고 이후는 큐에서 꺼내기만 합니다 (§2.5)

  (d) z=0 결정론성 — 같은 관측 2회의 최대차 = **0.000e+00**   [PASS] 기대 정확히 0

  (e) 추론 VRAM peak = 0.26 GB

  (f) 앙상블 모드(temporal_ensemble_coeff=0.01, n_action_steps=1) 스텝당 = **11.7 ms**
      30 Hz 예산 33 ms 대비 ✅ 충족 (2.9배 여유)
      50 Hz 예산 20 ms 대비 ✅ 충족 (1.7배 여유)
```

**③ ★★ GPU / CPU 대비 — 이 랩의 두 번째 발견입니다.**

```
=== [6-b] 같은 정책을 CPU로도 — 하드웨어가 실행 모드의 가용 범위를 정한다 ===

  τ_infer  GPU   12.8 ms   ·   CPU  132.5 ms   → **10.3배**

  실행 모드                     f2  예산[ms]  GPU            CPU
  -------------------------  -----  --------  -------------  -------------
  기본값 n_action_steps=100  30 Hz     3,333  ✅ 260배 여유  ✅ 25배 여유
  기본값 n_action_steps=100  50 Hz     2,000  ✅ 156배 여유  ✅ 15배 여유
  앙상블 n_action_steps=1    30 Hz        33  ✅ 3배 여유    ❌ 4.0배 부족
  앙상블 n_action_steps=1    50 Hz        20  ✅ 2배 여유    ❌ 6.6배 부족
```

**④ 종합.**

```
  종합 — PASS 37 · FAIL 0 · WARN 0 · INFO 6  (총 43건)

  [저장] /.../artifacts/W2-M1/labs/verify_act_install.csv
```

| 판정 항목 | 기대 | 판정 방법 |
|---|---|---|
| **청크 shape** | **정확히 `(1, 100, 14)`** | ★ **Step 7의 정답 대조 기준입니다.** `[B, chunk_size, D_action]` |
| 카메라 수별 증분 | **전부 `+0`** | 결정론적. 6.3에서 판정 |
| `use_vae` 차이 | **17.4M** | 추론에서 안 쓰이는 파라미터(§4.2) |
| **τ_infer** | **한 자릿수~십수 ms**(GPU) / **100 ms 언저리**(CPU) | ★ **G5.** 6.2 아래 주의를 먼저 읽으세요 |
| **1회차/이후 배율** | **20배 이상** | ★ 6.4에서 판정 |
| **z=0 최대차** | **정확히 `0.000e+00`** | ★ 흔들리면 안 되는 값입니다 |
| 앙상블 스텝당 | **τ_infer와 거의 같음** | 앙상블 자체의 계산은 무시할 만합니다. 비싼 것은 **매 스텝 추론** |
| CPU/GPU 배율 | **10~15배** | |
| **종합** | **FAIL 0 · WARN 0** | INFO는 측정값이라 판정 대상이 아닙니다 |

> ⚠️ **τ_infer의 절대값을 이 문서와 맞추려 하지 마세요.** 같은 GPU·같은 스크립트로 집필 중 **6.5 / 8.1 / 10.1 / 11.2 / 12.8 / 16.3 / 22.9 / 66.0 ms** 가 나왔습니다. 유휴 GPU의 클럭 램프업과 다른 작업의 부하 때문이고, 66 ms가 나온 실행은 load average가 12였습니다.
> **판정은 세 가지 부등호로 합니다** — ① 기본 설정 예산(2,000/3,333 ms)에 **두 자릿수 배 이상 여유**가 있는가 ② 앙상블 예산(20/33 ms)에서 **GPU와 CPU가 갈리는가** ③ `select_action` 1회차가 이후보다 **한 자릿수 이상 비싼가**. 셋 다 맞으면 정상입니다.
> 별도 실측(도시에)에서는 τ_infer **7.7 ms** · CPU **112.1 ms**였습니다. 자기 값이 그 근방이면 좋고, 아니어도 위 부등호만 맞으면 넘어가세요.

### 6.3 ★ 여기서 판정할 것 ① — 51.6M · 52M · 80M 세 숫자

> 🔴 **이 자리에서 결론을 만들어내지 마세요.** 이 랩에서 **닫히지 않는 유일한 항목**입니다.

세 숫자가 있습니다.

| 값 | 출처 | 실측과의 관계 |
|---|---|---|
| **51,613,582 (51.6M)** | **이 설치본 실측** | — |
| **약 52M** | **lesson 본문** (§3.6·§7·§8.2) | ✅ **일치** |
| **약 80M** | **HF `docs/lerobot/act` 문서** | ❌ **불일치** |

lesson [§10](../lesson.md)이 이 불일치를 이미 🔴 항목으로 기록해두고 **"왜 다른지는 확인하지 못했습니다 — 원 ACT 구현을 읽어야 알 수 있고 이번에 읽지 않았습니다"** 로 닫았습니다. **이 랩도 그 이상 가지 않습니다.**

**확인된 사실은 둘뿐입니다.**

1. **LeRobot 0.6.1의 `ACTPolicy`가 만드는 학습가능 파라미터는 51.6M이고, 카메라 수와 무관합니다.**
2. **무관한 이유는 백본이 공유되기 때문입니다** — `modeling_act.py:334`에서 `self.backbone`을 **하나만** 만들고, `modeling_act.py:475`에서 카메라별로 **반복 호출**합니다.

```bash
sed -n '334p;474,476p' .venv-lerobot/lib/python3.12/site-packages/lerobot/policies/act/modeling_act.py
```

**원 ACT 구현이 카메라마다 백본을 두는지, 문서의 수치가 다른 구성 기준인지는 원 저장소를 직접 봐야 알 수 있고 이번에 보지 않았습니다.** 추측으로 단정하지 말고 **확인 대상**으로 남기세요.

> 📌 **다만 lesson의 지연 결론은 흔들리지 않습니다.** 파라미터가 52M이든 80M이든 τ_infer를 **직접 재보면** 되고, 재보니 예산의 100분의 1 이하입니다. **어긋난 것은 인용 수치이지 논증이 아닙니다.** 이 구분을 worksheet ④에 명시적으로 쓰세요 — **"수치가 틀렸다"와 "결론이 틀렸다"를 가르는 연습**입니다.

### 6.4 ★ 여기서 판정할 것 ② — 큐 소비가 한 자릿수 이상 싸다는 것의 뜻

`select_action` 1회차가 10.04 ms, 2~6회차가 0.25~1.07 ms입니다(집필 실측 배율 **23~57배**).

lesson [§2.5](../lesson.md)가 `chunk_size`와 `n_action_steps`를 **두 개의 별개 필드**로 구분하라고 못박은 이유가 여기 있습니다.

```
1회차:  predict_action_chunk()  →  액션 100개를 큐에 채움   ← 약 10 ms (모델 forward)
2회차~: 큐에서 popleft()                                    ← 약 0.2 ms (메모리 읽기)
```

> 🔴 **`n_action_steps`는 "몇 번 비싼 일을 하는가"를 정하는 노브입니다.** 100이면 100스텝에 한 번, 1이면 매 스텝. 그 사이의 스텝들은 사실상 공짜입니다.
> **그래서 `chunk_size`를 늘리는 것과 `n_action_steps`를 늘리는 것은 비용 구조가 다릅니다** — 전자는 한 번의 forward를 무겁게 하고(디코더 쿼리 수), 후자는 forward 횟수를 줄입니다. [practice `03`](../practice/README.md)이 $k$와 $n$을 따로 스윕한 이유입니다.

### 6.5 ★★ 여기서 판정할 것 ③ — 앙상블이 통과한 것을 어떻게 읽는가

> 🔴 **이 랩에서 결론을 잘못 내리기 가장 쉬운 곳입니다.**

lesson [§7 세 번째 오해](../lesson.md)가 이렇게 씁니다.

> 즉 **추론 호출이 청크당 1회에서 스텝당 1회로 100배**가 되고, 30 Hz면 허용 지연 예산이 3,333 ms에서 **33 ms**로 조여집니다. 온보드 추론이면 이 33 ms 안에 약 52M forward가 끝나야 합니다. **여기서 하드웨어가 답을 가릅니다** — RTX 3080은 앙상블 모드에서 스텝당 **8.1 ms**로 4배 여유를 두고 통과하지만, 같은 코드가 **CPU에서는 112 ms로 초과**합니다.

**lesson이 이미 결론을 적어뒀습니다. 그러니 여기서 할 일은 그 문장을 자기 기기에서 재현하는 것이고, 재현되면 두 열의 부등호가 갈려야 합니다.** 집필 검증 실측은 GPU 앙상블 스텝당 11.7 ms(33 ms 예산에 2.9배 여유), **CPU 132.5 ms(4.0배 부족)** 였습니다.

| | GPU (RTX 3080) | CPU (데스크톱 x86) | Jetson급 온보드 |
|---|---|---|---|
| τ_infer | 12.8 ms | 132.5 ms | **미측정** |
| 기본값 예산 2,000 ms @50 Hz | ✅ 156배 여유 | ✅ 15배 여유 | ? |
| **앙상블 예산 20 ms @50 Hz** | ✅ 2배 여유 | ❌ **6.6배 부족** | ? |
| **앙상블 예산 33 ms @30 Hz** | ✅ 3배 여유 | ❌ **4.0배 부족** | ? |

> 🔴 **같은 코드·같은 가중치인데 실행 모드의 가용 범위가 하드웨어로 갈립니다.**
> lesson [§5.3](../lesson.md)이 L3 설계 변수로 셋을 들었습니다 — `chunk_size` · `n_action_steps` · 앙상블 여부. **여기에 연산 하드웨어가 네 번째 변수로 결합합니다.** CPU에서도 ACT 기본 설정은 50 Hz로 돌릴 수 있지만 temporal ensembling은 원리적으로 불가능합니다.
> **그리고 lesson §7이 "온보드 추론이면"이라고 조건부로 남긴 자리는 이 랩에서도 열려 있습니다.** 데스크톱 x86 CPU와 Jetson급 SoC는 다른 이야기이고, **후자는 측정하지 않았습니다.** 팀 질문 `M1-6`(온보드 추론 여부)과 곱해서 읽어야 하는 항목입니다.

**worksheet ⑥에 답할 것**: "우리 로봇이 온보드 추론이라면 temporal ensembling을 켤 수 있는가?" — **답을 쓰지 말고, 답하려면 무엇을 알아야 하는지를 쓰세요.**

---

## Step 7 — ★ 백지 ACT 아키텍처 (excalidraw) (30분)

> 🔴 **여기서부터 코드가 없습니다.** lesson [학습 목표 4번](../lesson.md)이 지정한 산출물 그 자체입니다 — *"ACT 아키텍처를 백지에 그리면서 입출력 텐서 shape을 채울 수 있다 — 이미지 `[B, n_cam, 3, H, W]`부터 액션 청크 `[B, 100, D_action]`까지."*

### 7.1 규칙 — 먼저 lesson을 닫으세요

```bash
ls course/w2-policy-vla/01-imitation-learning-act/labs/act_architecture_worksheet.excalidraw
```

[excalidraw.com](https://excalidraw.com)에서 이 파일을 엽니다(설치 불필요, 좌상단 메뉴 → Open).

**시작하기 전에 반드시:**

1. **[`../lesson.md`](../lesson.md)의 §4.1·§4.2 블록도를 닫습니다.** 보면서 채우면 베끼기이지 백지 연습이 아닙니다.
2. Step 6의 출력도 닫습니다. 채운 뒤에 대조합니다.
3. 타이머를 **20분** 맞춥니다(워크시트 자체가 25~30분을 권합니다). **막힌 칸은 물음표로 남기고 넘어가세요** — 그 물음표가 무엇을 모르는지 알려줍니다.
4. 남은 10분은 **대조 + 틀린 칸 메모**입니다. 🔴 **정답만 고쳐 적지 말고 "왜 틀렸는지"를 한 줄씩** 적으세요.

### 7.2 워크시트의 구조 — 세 구획

**73개 요소의 껍데기(박스·화살표·shape 칸)는 그려져 있고 내용이 빈칸입니다.** 세 구획을 **다 채워야 한 문항**입니다.

| 구획 | 무엇을 채우나 | 대조 위치 |
|---|---|---|
| **A · 학습 경로** | 입력 3종(★ **학습 시에만 존재하는 것**에 표시) · ① VAE 인코더 · ② 메인 인코더 · ③ 디코더 · 출력 Linear · 손실 조립 | lesson §4.1 블록도 |
| **B · 추론 경로** | **통째로 빠지는 스택 하나에 ✗** · `z = torch.____(...)` · **왜 하필 0인가** · 결론(결정론성) | lesson §4.2 + §2.7의 `z` 고정 코드 |
| **C · 세 숫자와 청크 큐** | `chunk_size` / `n_action_steps` / `temporal_ensemble_coeff` · **큐 100칸에 실행분 ■ / 폐기분 □ 직접 칠하기** · 실행 정책 3갈래 · **지연 예산 부등식의 어느 자리에 들어가나** | lesson §6.4 표 · §2.5 · §2.6 |

**마지막에 "이 그림에 없는 것" 칸이 있습니다** — 언어 입력 유무, `n_obs_steps=1`의 함의 등. **거기가 이 워크시트에서 가장 값진 칸입니다.**

### 7.3 ★ 채점 체크리스트 (worksheet ⑤에 그대로 있습니다)

채운 뒤 [`../lesson.md`](../lesson.md) §4.1·§4.2·§6.4와 대조하세요.

**구획 A — shape 6개가 핵심입니다**

- [ ] **출력 청크 `[B, 100, D_action]`** — Step 6 (a)의 실측 `(1, 100, 14)`와 일치 ★
- [ ] VAE 인코더 입력 시퀀스 = **`[B, 102, 512]`** — `[CLS]` 1 + 상태 1 + 액션 100
- [ ] latent = **`[B, 32]`** (`latent_dim=32`) · mu와 log σ²로 split되기 전은 `[B, 64]`
- [ ] 이미지 토큰 = **카메라 2대 480×640에서 600개** — $2 \times (480/32) \times (640/32) = 2 \times 15 \times 20$
- [ ] 메인 인코더 입력 = **`[B, 602, 512]`** — 이미지 600 + 상태 1 + latent 1
- [ ] 디코더 쿼리 = **`[100, 512]`**, 개수가 곧 **`chunk_size`**

**구획 A — 구조: 왜 그런가**

- [ ] 스택이 **셋**이고 층수가 **4 / 4 / 1** — 디코더가 1층인 이유는 **원 구현의 버그를 재현성 때문에 따라간 것**(§7 첫 오해)
- [ ] **자기회귀가 아니다** — 한 번의 forward로 100스텝이 **병렬**로 나옵니다
- [ ] **관측은 단일 프레임**(`n_obs_steps=1`, 다른 값은 검증에서 거부) → **입력은 마르코프, 출력은 비마르코프**
- [ ] **언어 입력이 없다** → 태스크당 정책 하나. L5가 "해당 없음"인 이유
- [ ] 어텐션 비용의 거의 전부가 **이미지 토큰** → 지연을 줄일 첫 손잡이는 **카메라 수와 해상도**

**구획 B — 추론 경로**

- [ ] **VAE 인코더 스택이 통째로 빠진다** — 정답 액션이 없으므로
- [ ] **$z = 0$** (`torch.zeros`) — prior 평균으로 고정
- [ ] 결론 ① **배포 모델은 결정론적** (Step 6 (d)의 `0.000e+00`이 이것)
- [ ] 결론 ② **$z$는 실행 시 모드를 고르는 장치가 아니다** — 학습 시 시연자 변동을 흡수하는 정규화 장치
- [ ] 안 쓰이는 파라미터가 **17.4M** (Step 6 [5])

**구획 C — 세 숫자와 큐**

- [ ] 기본값에서 큐 100칸이 **전부 ■**(실행), 폐기 0개 · 새 관측 반영 주기 = **100 / $f_2$**
- [ ] receding horizon $k$면 **앞 $k$칸만 ■**, 나머지 $100-k$칸은 **□(폐기)** — 예측했지만 버립니다
- [ ] 앙상블은 `n_action_steps=1` **강제**이고, 겹치는 예측을 **버리지 않고 지수 가중 평균**합니다
- [ ] 지연 예산 부등식 $H_{\text{chunk}} \ge f_2(T_{\text{replan}}+\tau_{\text{infer}}+\tau_{\text{comm}})$의 좌변에 들어가는 것은 **`n_action_steps`** — `chunk_size`가 아닙니다
- [ ] 손실 조립 = `loss = l1_loss + mean_kld * kl_weight`

### 7.4 검산 — 자기가 쓴 것만으로

lesson을 열기 전에 셋을 계산해 보세요.

1. **카메라를 2대에서 4대로 늘리면 메인 인코더 입력 토큰이 몇 개가 되나?** → $4 \times 300 + 2 = 1202$
2. **그때 파라미터 수는 몇 개 늘어나나?** → **0개** (Step 6 [5]) — 백본이 공유되므로
3. **1과 2가 동시에 성립하는 것이 이상하지 않은가?** → 이상하지 않습니다. **파라미터는 안 늘고 연산과 어텐션 길이는 늘어납니다.** 여기서 헷갈리면 §4.1의 "이미지 토큰이 압도적으로 많다"를 잘못 읽은 것입니다.

> 🔴 3번이 이 Step에서 가장 교육적인 자리입니다. **"모델 크기"와 "추론 비용"이 같은 축이 아니라는 것**을 실측 두 개로 확인하는 것이고, 팀 질문 [`W2M1-4`](../lesson.md)(카메라 구성)가 왜 지연 질문인지의 근거입니다.

---

## Step 8 — ★★ 이 모듈의 제출물 두 개 (55분)

> 🔴 **이 랩의 본체입니다.** Step 1~7은 전부 이 둘의 재료였습니다.

**배분: 8.A 28분 · 8.B 22분 · 기록 5분.**

### 8.A ★★ lesson §3.6 표를 자기 숫자로 다시 계산 (28분)

lesson [§3.6](../lesson.md)의 표에는 **$\tau_{\text{infer}}$ 열이 없습니다.** 본문이 "터무니없이 넉넉하다"고 서술하고 **"한 자릿수~십수 ms"라는 범위**를 한 줄 덧붙였을 뿐입니다(절대값이 흔들려 범위로 적혀 있습니다 — 6.2 아래 주의). **자기 기기 값으로 열을 직접 세우는 것이 이 문항입니다.**

**worksheet ⑥ 6.1의 표를 채우세요.** 뼈대는 이렇습니다.

$$T_{\text{replan}} + \tau_{\text{infer}} + \tau_{\text{comm}} \;\le\; \frac{\texttt{n\_action\_steps}}{f_2}$$

**표를 두 번 채웁니다 — 두 열이 서로 다른 로봇의 이야기이기 때문입니다(3.3).**

| 설정 | `n_act` | 예산 @**50 Hz**(ALOHA 실측) | 예산 @**30 Hz**(우리 로봇 가정) | **τ_infer 대비 여유** (@30 Hz) | **200 ms 문턱 대비** (@30 Hz) |
|---|---|---|---|---|---|
| LeRobot 기본값 | 100 | ______ ms | ______ ms | ______배 | ______배 밖 |
| docstring 예시 | 50 | ______ ms | ______ ms | ______배 | ______배 밖 |
| 짧은 receding horizon | 10 | ______ ms | ______ ms | ______배 | ______ |
| temporal ensembling(강제) | 1 | ______ ms | ______ ms | ______배 | 안전 |

**내 $\tau_{\text{infer}}$ = ______ ms** (Step 6 (b)) · **CPU 값 = ______ ms** (Step 6 [6-b])

**그리고 아래 다섯 문장을 자기 숫자로 완성하세요.** 🔴 **숫자가 안 붙은 문장은 lesson 요약을 옮겨 적은 것입니다.**

| # | 문장 | 자기 숫자 | 출처 |
|---|---|---|---|
| 1 | **"지연은 제약이 아니다"** — 기본 설정의 예산은 τ_infer의 ______배다 | | Step 6 (b) |
| 2 | **"그러므로 `chunk_size=100`은 지연 논거로 정해진 값이 아니다"** — 예산이 ______ ms인데 필요한 것은 ______ ms다 | | Step 6 (b) |
| 3 | **"최악 반응 지연이 200 ms 문턱의 ______배 밖이다"** — 그리고 이 계산에 쓴 $f_2$는 **실측인가 가정인가** | | Step 3 · 3.3 |
| 4 | **"앙상블은 예산을 ______분의 1로 조인다"** — 그리고 그 예산에 **내 GPU는 ______, 내 CPU는 ______** | | Step 6 (f)·[6-b] |
| 5 | **"청크 100개 중 실제로 모델을 부르는 것은 ______번이고 나머지 ______번은 ______배 싸다"** | | Step 6 (c) |

> 📌 **검산 — practice `01`이 자동으로 해줍니다.** 손으로 채운 뒤 대조하세요.
> ```bash
> cd course/w2-policy-vla/01-imitation-learning-act/practice
> python3 01_latency_budget.py --sweep            # 30 Hz — lesson 본문 기준
> python3 01_latency_budget.py --f2 50 --sweep    # 50 Hz — ALOHA 실측 기준
> ```
> `--sweep`이 200 ms 문턱을 넘지 않는 `n_action_steps` 상한을 찾아줍니다. **30 Hz에서 6이었던 값이 50 Hz에서는 몇이 됩니까?** 이 한 숫자가 "전신 균형 태스크라면 청크를 몇 스텝까지 실행해도 되는가"의 답이고, **그때 추론 빈도가 몇 Hz가 되는지**가 다음 설계 질문입니다.

**worksheet ⑥ 6.3에 답할 것**: **"lesson §3.6의 세 결론(지연은 제약이 아니다 / 최악 반응 지연 3.3초 / 앙상블이 예산을 100분의 1로) 중, 내 실측으로 값이 바뀐 것은 몇 개이고 결론 자체가 뒤집힌 것은 몇 개입니까?"**

> 🔴 **힌트 — 세 번째가 위험합니다.** GPU에서는 앙상블이 통과하지만 CPU에서는 안 됩니다(6.5). **"결론이 뒤집혔다"고 쓰기 전에, 뒤집힌 것이 lesson의 주장인지 아니면 하드웨어라는 조건이 추가된 것인지 구분하세요.** lesson §7은 애초에 "온보드 추론이면"이라는 조건을 달고 있습니다.

### 8.B ★★ 완료 기준 — 한 문단 (22분)

> 🔴 **이것이 이 모듈의 최종 제출물입니다.** lesson 상단의 **완료 기준**을 그대로 옮깁니다.

> **완료 기준**: LeRobot ACT 기본값 `chunk_size=100`, `n_action_steps=100`, `temporal_ensemble_coeff=None`을 놓고 **(a)** 이 설정이 청크를 끝까지 개방루프로 실행한다는 뜻임을 지적하고 **(b)** 30 Hz 기준 3.33초의 최악 반응 지연을 계산하고 **(c)** 그 값이 [W1-M1 §3.3](../../../w1-generative-core/01-physical-ai-landscape/lesson.md)의 "자세가 무너지기까지 200 ms" 문턱과 왜 충돌하지 않는지(ALOHA는 고정 베이스라 낙상 모드가 없다)를 **한 문단으로 쓸 수 있다.**

**worksheet ⑦에 씁니다. lesson과 worksheet의 앞 칸들을 전부 닫고 한 번에 쓰세요.**

세 조각이 각각 무엇을 요구하는지만 갈라둡니다.

| 조각 | 무엇을 써야 하나 | 어느 Step의 재료 |
|---|---|---|
| **(a) 개방루프 완주** | 두 값이 **같다**는 사실에서 **"100스텝 동안 새 관측을 전혀 보지 않는다"** 를 끌어낼 것. **receding horizon이 기본값이 아니라 옵션**이라는 것까지 | Step 2(대조표) · Step 6 (c)(큐 소비) |
| **(b) 3.33초 계산** | $100/30 = 3.333$초 = 3,333 ms. 🔴 **그리고 그 30 Hz가 "우리 로봇의 $f_2$ 가정"이라는 것을 밝힐 것** — ALOHA 자신은 50 Hz라 2,000 ms입니다(Step 3). 두 값이 왜 다른지가 팀 질문 `W2M1-2` | Step 3 · Step 8.A |
| **(c) 충돌하지 않는 이유** | **ALOHA는 고정 베이스라 낙상 모드가 없다.** 3.3초 낡은 명령의 최악은 잡기 실패이고 재시도 가능. 200 ms는 **이족 균형 문제의 시간상수**라 이 태스크에 애초에 존재하지 않는다 | lesson §3.6·§6.2 |

**🔴 한 문단을 쓴 뒤, 아래 두 줄을 반드시 덧붙이세요.** 이것이 이 답을 lesson 요약과 가르는 지점입니다.

1. **"그러므로 이 하이퍼파라미터를 우리 로봇으로 그대로 옮기면 안 된다"** — 그 이유를 lesson §6.2 표의 세 행(고정 베이스 조작 / 이동 중 조작 / 전신 균형)으로.
2. **"그리고 옮길지 판단하려면 무엇을 알아야 하는가"** — 팀 질문 [`W2M1-5`](../lesson.md)(가장 빠른 실패 모드의 시간상수)와 [`W2M1-2`](../lesson.md)(기록 FPS와 배포 $f_2$의 일치 여부)를 지목.

**자가 채점 (worksheet ⑦ 하단에 그대로 있습니다)**

- [ ] (a)(b)(c)가 **한 문단 안에** 들어 있다 (세 개의 분리된 답이 아니라)
- [ ] (b)에 **30 Hz 값과 50 Hz 값이 둘 다** 있고, 어느 쪽이 **실측**이고 어느 쪽이 **가정**인지 밝혔다
- [ ] (c)가 **"ACT가 잘못 설계됐다"가 아니라 "태스크가 다르다"** 로 되어 있다
- [ ] 마지막 두 줄(이전 금지 · 무엇을 알아야 하나)이 붙어 있다
- [ ] **자기 실측 숫자가 최소 3개** 들어 있다 (fps · τ_infer · 문턱 대비 배수)

### 8.C 마무리 기록 (5분)

**① [`../../../../docs/progress.md`](../../../../docs/progress.md)에 실행 로그.** 랩을 완주한 날 반드시 남기세요 — 여기 쌓인 실측값이 다음 주 자료의 입력이 됩니다.

```markdown
## 2026-__-__ (W2-M1 / imitation-learning-act)

**환경**: 로컬 / WSL2 / RTX 3080 12GB (또는 CPU) / `.venv-lerobot` · lerobot 0.6.1 · torch 2.11.0+cu130
**돌린 것**:
- 설치 `uv pip install "lerobot[training,viz]"` → __초
- `verify_act_install.py --quick` → 21/21 PASS · FAIL __ · WARN __ · __초
- `--with-datasets` → fps ____ / ____ / ____ · 다운로드 __초
- `lerobot-dataset-viz --save 1` → .rrd __ MB · __초
- `lerobot-train --steps=200` → loss ____ → ____ · KL 기여율 ____% → ____% · __초
- 재개 `--resume=true --steps=260` → exit __ · 체크포인트 __개 · 디스크 __ GB
- `verify_act_install.py --compare-cpu` → τ_infer GPU ____ ms / CPU ____ ms · 큐 배율 ____배
- Step 7 백지 아키텍처 → 맞힌 칸 __ / 틀린 칸 __ / 물음표 __
- Step 8.A §3.6 재계산 → 200 ms 문턱 대비 ____배 · 8.B 완료 기준 문단 작성

**막힌 지점**: 증상 → 원인 → 해결 (labs 반영 여부)
**소요**: Step 1~6 __분 / Step 7~8 __분
**GPU 비용**: 로컬이므로 $0
**미검증 배지 해제**: ffmpeg 처방을 실행했다면 기록 / Jetson 측정은 여전히 미측정
**다음 액션**:
```

**② [`../../../../notes/glossary.md`](../../../../notes/glossary.md)에 용어 5개.** 후보: **behavior cloning · compounding error · covariate shift · DAgger · action chunking · `chunk_size` vs `n_action_steps` · temporal ensembling · CVAE latent · `kl_weight`($\beta$) · LeRobotDataset · receding horizon**. 용어마다 **제어·ML·LLM 비유**를 한 칸 붙이는 것이 이 저장소 규약입니다.

**③ [`../../../../notes/questions-for-team.md`](../../../../notes/questions-for-team.md)에 질문.** lesson §11의 `W2M1-1`~`W2M1-6` 중 **본인이 실제로 물어볼 것 둘**을 고르고, Step 3·6·8에서 새로 생긴 질문을 추가합니다.

> **Step 6을 하고 나면 `W2M1-4`(카메라 구성)를 자기 언어로 던질 수 있습니다.** 카메라를 늘려도 파라미터가 안 느는데 토큰은 선형으로 는다는 것을 직접 봤기 때문입니다.

**④ [`../../../../docs/course-plan.md`](../../../../docs/course-plan.md) 체크박스.** W2-M1 산출물 중 **5번(실행검증)** 을 갱신합니다.

---

## 결과 해석 가이드 — 어떤 출력을 보고 무엇을 판단하는가

| 보는 것 | 어디서 | 무엇을 판단하는가 |
|---|---|---|
| `.venv/bin/python -c "import lerobot"` → **`ModuleNotFoundError`** | Step 1 ④ | ★ **환경이 안 섞였다는 증거.** 임포트되면 **E7** |
| `--policy.type` 선택지에 `act`·`diffusion`·`pi0`·`groot` | Step 1.3 | W2 전체가 같은 CLI 아래에 있다 — lesson §1.3 계보 |
| **`21/21 PASS`** | `verify [2]` | ★ **G1.** 설치본이 lesson §6.4와 같다 |
| Q1이 **PASS** (`chunk size size` 오타 존재) | `verify [3]` | 상류 오타가 실재한다. lesson §2.5의 주석이 옳았다 |
| Q3가 **PASS** (`that means only`) | `verify [3]` | ★ 원래 WARN이었고 lesson을 고쳐 닫았습니다 — 2.3의 사연 |
| `NotImplementedError` | `verify [4]` | lesson §10 각주 3번이 닫힌다 |
| **docstring 문장 ≠ 예외 메시지** | `verify [3]` vs `[4]` | 두 문장을 섞어 인용하면 안 된다 |
| **`fps = 50`** (ALOHA 2종) | `verify [7]` | ★★ **G2.** @50 Hz 열은 ALOHA 자신, @30 Hz 열은 우리 로봇 $f_2$ 가정 — 3.3 |
| `lerobot/pusht` 존재 · `fps=10` | `verify [7]` | lesson §10 각주 2번이 닫힌다. W2-M2가 쓸 데이터셋 |
| **50 에피소드 · 6.7분** | `verify [7]` | ★ 로봇 학습에서 "적은 데이터"의 기준점 (§8.2 초록 인용과 자릿수 동일) |
| `D_action` = **14 / 2** | `verify [7]` | §4.1 출력 Linear의 차원. 데이터셋이 정한다 |
| **`Falling back to 'pyav'`** | Step 4·5 | ★ **정상.** 그 위 붉은 traceback은 무해 — **E1** |
| `.rrd` **25 MB** · 뷰어 안 뜸 | Step 4 | ★ **G3.** `--save 1`이 먹었다 |
| `num_learnable_params=51613582 (52M)` | Step 5 ① | ★ **verify [5]와 같은 값.** HF 문서의 "약 80M"과 다름 — 6.3 |
| **loss 24.4 → 3.8** | Step 5 ② | ★ **G4.** 절대값이 아니라 **방향**이 판정 기준 |
| `grdn` 388 → 101 | Step 5 ② | 함께 내려가야 정상 |
| `mem_gb = 2.10` | Step 5 ② | batch 8 · 1카메라. 12 GB에 넉넉 |
| **KL 기여율 96.8% → 86.0%** | Step 5.3 (손계산) | ★ **G4.** practice `04` 곡선(99.1%→7.0%)의 앞부분 |
| `updt_s` 0.074 vs 2.86 | Step 5 / CPU | 약 39배. 100k 외삽이 2.1h vs 80h |
| 체크포인트 **592 MB/개** | Step 5.4 | ★ **E6.** `--save_freq`를 크게 |
| **청크 shape `(1, 100, 14)`** | `verify [6](a)` | ★ **Step 7의 정답 대조 기준.** `[B, chunk_size, D_action]` |
| **카메라 1~4대 증분 `+0`** | `verify [5]` | ★ 백본 공유 — `modeling_act.py:334`·`475` |
| `use_vae` 차이 **17.4M** | `verify [5]` | 추론에서 안 쓰이는 파라미터 (§4.2) |
| **τ_infer + load average** | `verify [6](b)` | ★ **G5.** §3.6에 없던 열이 채워지는 자리. **절대값이 아니라 배율로 판정** |
| **큐 소비 20배 이상** | `verify [6](c)` | ★ `chunk_size`와 `n_action_steps`가 별개 필드인 물리적 이유 |
| **z=0 최대차 `0.000e+00`** | `verify [6](d)` | ★ 정확히 0이어야 함. §2.7 결정론성 |
| 앙상블 스텝당 ≈ τ_infer | `verify [6](f)` | 비싼 것은 앙상블 계산이 아니라 **매 스텝 추론** |
| **앙상블 GPU ✅ / CPU ❌** | `verify [6-b]` | ★★ **하드웨어가 L3의 네 번째 변수** — 6.5에서 판정 |
| **WARN 0 · FAIL 0** | `verify` 종합 | WARN이 뜨면 설치가 아니라 **문서 수정 대상** — 2.3 |

---

## 흔한 에러와 대처 — 7개

**E1이 이 랩의 대표 에러입니다.** 실패가 아닌데 실패처럼 보입니다.

### E1. ★ 붉은 traceback 수십 줄 — `torchcodec` 로드 실패

```
OSError: libavutil.so.56: cannot open shared object file: No such file or directory
  ...
OSError: Could not load this library: .../torchcodec/libtorchcodec_core4.so
[end of libtorchcodec loading traceback].
'torchcodec' is installed but cannot be loaded (see the error above). Falling back to 'pyav' as a default decoder.
```

| | |
|---|---|
| **언제 뜨나** | 실제로 **프레임을 디코딩하는 쪽** — `lerobot-dataset-viz`(Step 4) · `lerobot-train`(Step 5) · `LeRobotDataset` 생성. **메타데이터만 읽는 Step 3에서는 안 뜹니다** |
| **원인** | 시스템에 **`ffmpeg`이 설치돼 있지 않습니다**(`which ffmpeg` → not found · `ldconfig -p \| grep libavutil` → 없음). `torchcodec`은 `libavutil.so.56~.60`(FFmpeg 4~8) 중 하나를 찾다가 전부 실패합니다 |
| **영향** | **없습니다.** `pyav` 15.1.0으로 폴백해 데이터셋 로드·시각화·학습이 **전부 정상 동작**했습니다. 이 문서의 Step 3·4·5 출력이 **모두 이 폴백 상태에서 나온 것**입니다 |
| **판정** | 🔴 **마지막 줄만 보세요.** `Falling back to 'pyav' as a default decoder.` 가 있으면 정상입니다 |
| **대처** | **그대로 진행하세요.** 없애고 싶으면 아래 |

```bash
sudo apt install -y ffmpeg
which ffmpeg && ldconfig -p | grep libavutil
```

> ⚠️ **미검증(sudo 필요)** — 위 처방은 집필 시점에 **실행 검증되지 않았습니다.** 권한 때문입니다.
> **처방을 안 써도 이 랩은 전부 돕니다.** 실행했다면 결과를 [`../../../../docs/progress.md`](../../../../docs/progress.md)에 기록하고 이 배지를 제거하세요.

> 📌 **왜 이것을 랩 문서 첫 에러로 두는가.** 시뮬·로봇 스택은 **"에러처럼 보이지만 폴백이 성공한 로그"** 가 흔합니다. W1-M2 랩의 E1(종료 시 EGL traceback)도 같은 종류였습니다. **traceback의 길이가 아니라 마지막 줄을 읽는 습관**이 이 분야의 기본기입니다.

### E2. `NotImplementedError: n_action_steps must be 1 when using temporal ensembling`

```python
ACTConfig(temporal_ensemble_coeff=0.01, n_action_steps=100)
# NotImplementedError: `n_action_steps` must be 1 when using temporal ensembling.
# This is because the policy needs to be queried every step to compute the ensembled action.
```

| | |
|---|---|
| **원인** | **에러가 아니라 설계입니다.** 앙상블을 만들려면 매 스텝 추론해야 하므로 논리적으로 강제되는 제약입니다(lesson §2.6·§7 세 번째 오해) |
| **대처** | `n_action_steps=1`로 함께 바꾸세요. 그러면 **추론 호출이 100배**가 되고 예산이 33 ms(@30 Hz)로 조여집니다 — Step 6.5 |
| **주의** | 이 **예외 메시지**와 docstring 문장은 **다른 문장**입니다. 인용할 때 섞지 마세요 |

### E3. `lerobot-train`이 `--policy.repo_id`를 요구한다

| | |
|---|---|
| **원인** | 학습 결과를 HF Hub에 올리는 것이 기본 동작입니다. lesson §8.2 커맨드는 `--policy.repo_id=<HF 네임스페이스>/act_w2m1`를 주는 형태입니다 |
| **대처** | HF 계정 없이 돌리려면 **`--policy.push_to_hub=false`** 를 주세요. 이 랩의 Step 5 커맨드가 그렇게 돼 있습니다 |

### E4. `Warning: You are sending unauthenticated requests to the HF Hub.`

```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

| | |
|---|---|
| **원인** | HF 토큰이 없습니다 |
| **영향** | **없습니다.** 공개 데이터셋은 인증 없이 받아집니다 — 실측으로 **1.74 GB를 40.6초**에 받았습니다. 속도 제한만 걸립니다 |
| **대처** | 무시하세요. 자주 받을 것 같으면 `export HF_TOKEN=...` |

### E5. `lerobot-dataset-viz: error: the following arguments are required: --episode-index`

| | |
|---|---|
| **원인** | `--repo-id`와 `--episode-index`가 **둘 다 필수**입니다 |
| **대처** | 두 인자를 다 주고, **`--save 1 --output-dir artifacts/W2-M1/labs`도 함께** 주세요 |

### E6. ★ 스모크만 돌렸는데 디스크가 GB 단위로 준다

```bash
du -sh outputs/train/act_w2m1_smoke
# 1.2G     ← 200스텝 + 260스텝 재개만 했는데
```

| | |
|---|---|
| **원인** | **체크포인트 하나가 592 MB**입니다. 내역: `model.safetensors` **206.8 MB**(52M × fp32) + 옵티마이저 상태 등. `--save_freq` 마다 하나씩 쌓입니다 |
| **기본값** | `--save_freq=20000` · `--steps=100000` → 체크포인트 **5개 ≈ 3.0 GB** |
| **증상** | Step 5.4의 재개까지 하면 체크포인트가 2개가 되어 **1.2 GB** |
| **대처** | 스모크에서는 **`--save_freq`를 총 스텝 수 이상**으로 주어 마지막 하나만 남기세요. 다 봤으면 지웁니다 |

```bash
rm -rf outputs/train/act_w2m1_smoke
```

> 📌 `outputs/`는 이 저장소 `.gitignore` 232행에 등재돼 있어 **커밋되지는 않습니다. 디스크만 먹습니다.**
> 📌 `checkpoints/last`는 **심볼릭 링크라 용량 0**입니다. 두 번 세지 마세요.

### E7. ★ `ModuleNotFoundError: No module named 'lerobot'` — 인터프리터를 잘못 골랐다

| | |
|---|---|
| **원인** | W1용 `.venv`나 시스템 `python3`으로 돌렸습니다. LeRobot은 **`.venv-lerobot`에만** 있습니다 |
| **판정** | `verify_act_install.py`가 `[1] 환경` 표에 인터프리터 경로를 찍고, `.venv-lerobot`이 아니면 **WARN**을 냅니다 |
| **대처** | 모든 명령에 **`.venv-lerobot/bin/python`** 또는 **`.venv-lerobot/bin/lerobot-*`** 경로를 직접 쓰세요 |

> 🔴 **반대 방향이 더 위험합니다.** "귀찮으니 `.venv`에 lerobot을 설치하자"고 하면 **torch가 2.13 → 2.11로 내려가면서 W1 practice가 깨집니다.** 두 환경을 분리한 것이 그 이유입니다 — 0.2 ①.

### 에러는 아니지만 자주 놀라는 것 셋

| 증상 | 판정 |
|---|---|
| **`WARNING: Device 'None' is not available. Switching to 'cuda'.` 가 세 번 뜬다** | **정상입니다.** `verify`가 device 지정 없이 `ACTConfig()`를 만들 때 lerobot이 내는 안내입니다. 대조 결과에 영향 없음 |
| **`WARNING:lerobot.datasets.utils:Unknown fields in DatasetInfo: ['files_size_in_mb']`** | **정상입니다.** 데이터셋 메타에 이 lerobot 버전이 모르는 필드가 있다는 안내이고, 무시됩니다 |
| **τ_infer가 실행마다 6.5 ~ 66 ms로 흔들린다** | **정상입니다.** batch 1짜리 벤치는 커널 실행 오버헤드가 지배하고, 유휴 GPU의 클럭 램프업과 다른 작업의 부하가 그대로 들어옵니다. 스크립트가 함께 찍는 **load average**를 보세요. **판정은 절대값이 아니라 배율로** — 0.5와 6.2 아래의 주의 |

---

## 심화 — 여유가 있으면 (각 10분)

**변형 A — `n_action_steps`를 바꿔 큐 소비 패턴이 어떻게 달라지나**

```bash
# verify 스크립트를 고치지 않고, 파이썬 한 줄로 확인합니다
.venv-lerobot/bin/python - <<'EOF'
import torch, time, sys
sys.path.insert(0, "course/w2-policy-vla/01-imitation-learning-act/labs")
from verify_act_install import build_policy
for n_act in (1, 10, 50, 100):
    p = build_policy(1, True, "cuda", n_action_steps=n_act); p.eval(); p.reset()
    obs = {"observation.images.cam0": torch.rand(1,3,480,640, device="cuda"),
           "observation.state": torch.zeros(1,14, device="cuda")}
    with torch.no_grad():
        for _ in range(3): p.select_action(obs)
        p.reset(); torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(100): p.select_action(obs)
        torch.cuda.synchronize()
    print(f"n_action_steps={n_act:3d}  100스텝 총 {(time.perf_counter()-t0)*1e3:7.1f} ms")
EOF
```

**100스텝을 실행하는 총 시간이 `n_action_steps`에 어떻게 의존합니까?** 추론 호출 횟수가 $\lceil 100/n \rceil$이므로 **총 시간 ≈ 호출 횟수 × τ_infer + 100 × 큐 비용**이어야 합니다. 실측이 그 식과 맞습니까? **맞는다면 lesson §2.5의 두 필드 구분이 비용 모델로 확인된 것**입니다.

**변형 B — 카메라 3대 데이터셋으로 학습 스모크를 다시**

```bash
.venv-lerobot/bin/lerobot-train \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/train/act_w2m1_3cam --job_name=act_w2m1_3cam \
  --batch_size=8 --steps=200 --log_freq=25 --save_freq=1000 --wandb.enable=false
```

**`num_learnable_params`가 1카메라 때와 같습니까?**(같아야 합니다 — Step 6 [5]) **그런데 `mem_gb`와 `updt_s`는 어떻게 됩니까?** 이 두 줄의 대비가 §4.1의 "파라미터는 안 늘고 어텐션 길이가 는다"의 실물입니다. 🔴 **`--save_freq`를 크게 주세요** — E6.

---

## 다음

- **제출물**: [`worksheet.md`](worksheet.md) 전부 기입 — 특히 **⑥ §3.6 재계산**과 **⑦ 완료 기준 문단**
- **실행 기록**: [`../../../../docs/progress.md`](../../../../docs/progress.md) (8.C)
- **용어·질문**: [`../../../../notes/glossary.md`](../../../../notes/glossary.md) · [`../../../../notes/questions-for-team.md`](../../../../notes/questions-for-team.md)
- **이전 토픽**: [W1-M1 Physical AI 지형도](../../../w1-generative-core/01-physical-ai-landscape/lesson.md) — §4의 지연 예산 부등식에 이 랩이 실제 숫자를 넣었습니다
- **다음 토픽**: **W2-M2 Diffusion Policy** *(집필 예정)* — 여기서 만든 파이프라인으로 **PushT 학습 → 평가 → 롤아웃을 완주**합니다. `lerobot/pusht`를 Step 3에서 이미 확인해뒀습니다. 그리고 [practice `04`](../practice/README.md)가 남긴 "배포 경로가 점추정이라 다봉성이 남는다"를 **추론 자체가 샘플링인 모델**로 받습니다
