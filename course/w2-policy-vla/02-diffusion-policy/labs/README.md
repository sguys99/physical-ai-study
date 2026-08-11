# W2-M2 랩. LeRobot 파이프라인을 끝까지 통과시키기

> **모듈**: W2-M2, [`../lesson.md`](../lesson.md)
> **8단계, 약 2시간 20분.** lesson 정독 2h와 [`../practice/`](../practice/) 1h는 별도
> **로컬 RTX 3080 12GB로 완결. 클라우드 불필요. G1 실기 불필요. HF 계정 불필요.**
> **디스크 여유 8 GB 이상이 필요합니다.** 체크포인트 하나가 3.0 GB입니다(0.1).
> 이 랩의 제출물은 [`worksheet.md`](worksheet.md)입니다. 시작 전에 사본을 하나 만들어 두세요.

[W2-M1 랩](../../01-imitation-learning-act/labs/README.md)이 **로봇 학습 파이프라인을 처음 돌리는 랩**이었다면, 이 랩은 **그 파이프라인을 끝까지 통과시키는 랩**입니다. 거기서는 손실이 내려가는 것을 보고 멈췄습니다. 여기서는 학습한 정책을 실제 환경에 넣어 굴리고, 롤아웃 mp4가 디스크에 떨어지는 것까지 봅니다.

마스터플랜이 이 모듈에 건 요구가 하나뿐입니다. **PushT에 Diffusion Policy를 학습하고 평가하고 롤아웃 영상을 확인하는 것**이고, 컷 가이드의 "끝까지 지킬 것" 네 항목 중 하나가 이 완주입니다. 학습만 하고 끝내면 이 랩을 한 것이 아닙니다.

> 🔴 **이 랩의 무게중심. 파이프라인 완주는 절반이고 나머지 절반은 산수입니다**
>
> 학습 명령을 붙여넣어 손실이 내려가는 것을 보는 데는 12분이면 됩니다. 그 12분이 이 랩의 목적이라면 랩을 만들 이유가 없습니다. 이 랩이 실제로 요구하는 것은 셋입니다.
>
> | | 무엇을 | 어느 Step |
> |---|---|---|
> | ① | **라이브러리 기본값으로 돌린 것이 왜 논문 재현이 아닌지**를 설정 7항목으로 지목한다 | Step 4 |
> | ② | **자기 기기의 추론 지연**을 재서 lesson §4.2의 실행 방식 비교표 4행을 자기 숫자로 다시 쓴다 | Step 7 |
> | ③ | 그럼에도 태스크가 도는 이유를 **receding horizon**으로 설명한다 | Step 8 |
>
> lesson §4.2가 박아둔 값은 집필자의 RTX 3080에서 나온 것입니다. **남이 잰 값을 읽는 것과 자기가 재는 것은 다릅니다.** 특히 이 표는 한 칸이 바뀌면 "예산 안에 드는가" 열의 판정까지 뒤집힙니다. 논문 레시피 행의 여유가 95 ms뿐이라, 자기 기기가 15% 느리면 그 행은 **초과로 넘어갑니다.**
>
> 전체 2시간 20분 중 **명령이 실제로 도는 시간은 다 합쳐 약 6분**입니다. 나머지는 읽고 쓰고 계산하는 시간입니다. **Step 1~6에 79분, Step 7~8에 60분**을 씁니다. 비율이 뒤집히면 이 모듈을 잘못 하고 있는 것입니다.
>
> W2-M1 랩보다 명령 시간이 두 배입니다. 학습을 두 번 돌리고 평가를 한 번 돌리기 때문입니다. 그래도 여전히 전체의 5%입니다.

---

## 0. 사전 준비

### 0.1 이 랩에 필요한 것과 필요 없는 것

| 항목 | 필요 여부 | 비고 |
|---|---|---|
| **리포 루트의 `.venv-lerobot/`** ★ | **필요** | 🔴 W2-M1에서 만든 그 환경입니다. **새로 만들지 않습니다.** W1용 `.venv`와 섞으면 안 됩니다 (에러 표 **E7**) |
| **`pusht` · `diffusion` · `evaluation` extras** ★ | **필요** | Step 1에서 얹습니다. `diffusion`이 없으면 정책 생성 시점에 죽습니다 (**E3**) |
| **로컬 GPU (RTX 3080급)** | **강권** | Step 3~5가 CPU에서는 실용적이지 않습니다. 0.3의 CPU 열을 보세요 |
| **디스크 여유 8 GB 이상** ★ | **필요** | 🔴 **체크포인트 하나가 3.0 GB**입니다. 아래 내역 표 (**E6**) |
| **인터넷** | **필요** | pip · HF Hub 데이터셋 · ResNet-18 사전학습 가중치 |
| **HF(Hugging Face) 계정과 토큰** | **불필요** | 공개 데이터셋은 인증 없이 받아집니다. 경고만 뜹니다 (**E5**). 학습은 `--policy.push_to_hub=false`로 (**E8**) |
| **`ffmpeg`** | **불필요** | 없어도 됩니다. `torchcodec`이 로드에 실패하고 `pyav`로 폴백하는데 **학습도 평가도 mp4 저장도 그 상태에서 정상 동작**했습니다 (**E4**) |
| 디스플레이 / X11 / 뷰어 | **불필요이자 금지** | 롤아웃은 mp4 파일로 떨어집니다. 창을 띄우지 않습니다 |
| 클라우드 인스턴스 | 불필요 | 필수 절차에 30분 넘는 것이 없습니다. 선택 항목인 본 학습만 약 4.4시간입니다 |
| Unitree G1 실기 | 불필요 | 이 4주 내내 불필요 |
| **선수 모듈 W2-M1 랩** | **필수** | `.venv-lerobot`이 거기서 만들어집니다. ACT 대비 숫자도 거기 있습니다 |
| **선수 모듈 W2-M2 practice** | **필수** | Step 7이 `practice/04`를 씁니다. `01`~`03`도 먼저 돌리세요 |
| 선수 모듈 W1-M1 | **권장** | Step 7의 200 ms 문턱이 [W1-M1 §3.3](../../../w1-generative-core/01-physical-ai-landscape/lesson.md), 부등식이 §4.1 |

**디스크 내역입니다. 이 랩에서 가장 자주 사고가 나는 자리라 먼저 못박습니다.**

| 무엇 | 크기 | 언제 생기나 | 지워도 되나 |
|---|---|---|---|
| `.venv-lerobot/` + 이번 extras | **약 6 GB** | Step 1 (1회) | ❌ 이 랩 내내 씁니다 |
| `~/.cache/huggingface/lerobot/hub/datasets--lerobot--pusht/` | **7.5 MB** | Step 2 | ✅ 다시 받으면 됩니다 |
| `~/.cache/torch/` (ResNet-18 가중치) | **45 MB** | Step 3 | ✅ |
| **`outputs/train/<job>/checkpoints/NNNNNN/`** ★ | **3.0 GB / 개** | Step 3 | ✅ **Step 5까지 보고 나면 지우세요** (**E6**) |
| `outputs/eval/<dir>/` (mp4 2개 + json) | **280 KB** | Step 5 | ✅ 가볍습니다 |

> 🔴 **체크포인트 3.0 GB의 내역이 중요합니다.** `pretrained_model/model.safetensors`가 **1,003 MB**이고 `training_state/optimizer_state.safetensors`가 **2.0 GB**입니다. Adam이 파라미터마다 1차 2차 모멘트 둘을 들고 있어 옵티마이저 상태가 모델의 두 배입니다. W2-M1의 ACT는 같은 구조로 592 MB였고(198 MB + 394 MB), 이 모듈은 **5.1배**입니다.
>
> **lesson §5.3이 적은 "1,003 MB"는 `model.safetensors` 한 파일**이고, 학습이 실제로 쓰는 디스크는 그 세 배입니다. 두 숫자를 섞지 마세요. 선택 항목인 본 학습(10만 스텝, `save_freq` 기본 20,000)은 체크포인트가 5개 쌓여 **약 15 GB**가 됩니다.

### 0.2 선수 확인, 세 가지

**① `.venv-lerobot`이 그대로 있는가.** W2-M1에서 만든 것을 재사용합니다. 없으면 그 랩의 Step 1로 돌아가세요.

```bash
ls -d .venv-lerobot && .venv-lerobot/bin/python -c "import lerobot, torch; print(lerobot.__version__, torch.__version__, torch.cuda.is_available())"
# 0.6.1 2.11.0+cu130 True
```

**② W1용 `.venv`와 섞지 않았는가.** 이쪽에는 lerobot이 없어야 정상입니다.

```bash
.venv/bin/python -c "import lerobot" 2>&1 | tail -1
# ModuleNotFoundError: No module named 'lerobot'   ← 이게 정상입니다
```

**③ practice를 먼저 돌렸는가.** Step 7이 `practice/04`의 출력을 씁니다. `01`~`03`은 설치 없이 도니까 먼저 끝내두세요.

```bash
cd course/w2-policy-vla/02-diffusion-policy/practice
python3 01_mean_collapse.py && python3 02_receding_horizon.py && python3 03_nfe_budget.py
cd -
```

### 0.3 실행 시간, 집필 환경 실측

| Step | 명령 | GPU 실측 | 비고 |
|---|---|---|---|
| 1 | `uv pip install "lerobot[pusht,diffusion,evaluation]"` | **미측정** | 1회만. 집필 시점에 이미 설치돼 있어 재설치 시간을 재지 못했습니다. W2-M1의 `[training,viz]`가 42초였습니다 |
| 1 | `verify_dp_install.py --quick` | **6.5초** | 대부분 torch 임포트입니다 |
| 2 | `verify_dp_install.py --with-dataset` | **7.6초** | 캐시가 있을 때. 첫 다운로드는 7.5 MB라 금방입니다 |
| 3 | `lerobot-train --steps=300` (기본값) | **학습 루프 64초** | 기동 약 12초와 3 GB 저장이 더 붙습니다. **벽시계로 1분 30초를 잡으세요** |
| 4 | `lerobot-train --steps=20` (논문 레시피) | **27.7초** | 체크포인트를 안 만듭니다(`--save_checkpoint=false`) |
| 5 | `lerobot-eval` 2 에피소드 | **`eval_s` 13.1초** | 모델 로드와 환경 생성이 더 붙습니다. 벽시계 1분을 잡으세요 |
| 6 | 사전학습 체크포인트 시도 | **수십 초** | 네트워크. 두 번 다 실패로 끝나는 것이 정상입니다 |
| 7 | `practice/04 --device cuda` | **31초** | `--smoke`면 12.5초 |
| — | **합계** | **약 6분** | 명령 실행만. 읽고 쓰는 시간은 0.4 |
| — | *(선택) 10만 스텝 본 학습* | *약 4.4시간* | 🔴 **필수가 아닙니다.** 아래 배지를 보세요 |

**300스텝 실측의 상세입니다.** 정상 속도가 초당 6.3~7.0 스텝, `smp/s` 370~424, `mem_gb` 4.97~5.08이었습니다. 손실은 이렇게 내려갔습니다.

| step | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|
| `loss` | 0.883 | 0.204 | 0.108 | 0.080 | 0.074 | **0.064** |

> ⚠️ **미검증(GPU 필요).** **10만 스텝 본 학습은 집필 시점에 실행되지 않았습니다.**
> 예상 소요는 RTX 3080 12GB 기준 **약 4.4시간**이고 300스텝 실측(64초)에서 외삽한 값입니다. 디스크는 `save_freq` 기본 20,000에서 체크포인트 5개, **약 15 GB**입니다.
> **이 랩의 완주 조건이 아닙니다.** 돌린다면 tmux 안에서 돌리고, 결과를 [`../../../../docs/progress.md`](../../../../docs/progress.md)에 기록한 뒤 이 배지를 제거하세요.
>
> ```bash
> tmux new -s dp
> .venv-lerobot/bin/lerobot-train --dataset.repo_id=lerobot/pusht --policy.type=diffusion \
>   --policy.horizon=16 --policy.n_action_steps=8 --policy.crop_shape="[84,84]" \
>   --policy.use_group_norm=true --policy.pretrained_backbone_weights=null \
>   --policy.use_separate_rgb_encoder_per_camera=false \
>   --policy.device=cuda --policy.push_to_hub=false \
>   --batch_size=64 --steps=100000 --save_freq=50000 --wandb.enable=false \
>   --output_dir=outputs/train/w2m2_dp_full --job_name=dp_pusht_full
> # Ctrl-b d 로 빠져나오고, tmux attach -t dp 로 돌아옵니다
> ```
>
> `--save_freq=50000`으로 줄여 잡은 것은 디스크 때문입니다. 기본값 20,000이면 15 GB, 50,000이면 6 GB입니다.

### 0.4 시간 예산

| Step | 내용 | 명령 실행 | 예상 소요 | 누적 | 성격 |
|---|---|---|---|---|---|
| 1 | extras 설치와 **설치본 41항목 자동 대조** ★ | 1분 | **10분** | 10m | ★ 출력 읽기 |
| 2 | **PushT 데이터셋을 자기 손으로 세기** | 8초 | **7분** | 17m | 픽셀 좌표 확인 |
| 3 | 기본값 학습 스모크 300스텝 | 1분 30초 | **12분** | 29m | 손실 곡선 읽기 |
| 4 | **논문 레시피로 다시 돌리기 ★** | 28초 | **15분** | 44m | ★ **7항목 대조** |
| 5 | **평가와 롤아웃 mp4 ★★** | 1분 | **20분** | **64m** | ★★ **E1을 넘고 영상을 본다** |
| 6 | **사전학습 체크포인트가 막히는 자리 ★** | 1분 | **15분** | 79m | ★ **E2 판단** |
| 7 | **★★ 자기 기기 지연으로 §4.2 표 다시 쓰기** | 31초 | **25분** | 1h 44m | ★★ **코드 한 줄, 나머지는 산수** |
| 8 | **★★ 완료 기준 한 문단 (최종 제출물)** | — | **35분** | **약 2시간 20분** | ★★ **코드 없음. 전부 글** |

### 0.5 검증 환경

이 문서의 모든 실측값은 아래 환경에서 나왔습니다.

| 항목 | 값 |
|---|---|
| 날짜 | **2026-08-11** |
| OS | **Ubuntu on WSL2** (Linux 6.6.87.2-microsoft-standard-WSL2) |
| Python | **3.12.12** (uv 관리) |
| 가상환경 | **리포 루트 `.venv-lerobot/`** |
| `lerobot` | **0.6.1** |
| `torch` | **2.11.0+cu130** · `torch.cuda.is_available() == True` |
| `diffusers` / `gym-pusht` / `pymunk` / `gymnasium` | 0.39.0 / 0.1.6 / 6.11.1 / 1.3.0 |
| `numpy` / `torchvision` / `scipy` / `shapely` | 2.2.6 / 0.26.0 / 1.18.0 / 2.1.2 |
| GPU | **RTX 3080 12GB** |
| `ffmpeg` | **설치돼 있지 않음** (그래서 E4가 납니다. 그래도 전 절차 정상) |

> ⚠️ **벽시계(ms)는 실행마다 크게 흔들립니다.** Step 7에서 가장 조심할 지점이고, [`../practice/README.md`](../practice/README.md) §4.4가 같은 기기에서 두 번 돌린 값이 DDIM 16에서 106.4 ms와 135.6 ms로 갈린 기록을 남겨뒀습니다. **판정은 절대값이 아니라 자릿수와 순서로** 합니다.

### 0.6 미검증으로 남는 것, 두 가지

> ⚠️ **미검증(GPU 필요)** 10만 스텝 본 학습과 그 결과의 성공률입니다. 0.3의 배지를 보세요. **이 랩의 완주 조건이 아닙니다.** 300스텝 스모크로도 파이프라인은 전부 통과합니다. 다만 성공률은 0.0이 나오고, 그것이 정상입니다(Step 5).

> ⚠️ **미측정(재설치 안 함)** Step 1의 extras 설치 시간입니다. 집필 시점에 이미 깔려 있어 재설치를 하지 않았습니다. 패키지 목록과 버전은 전부 확인했습니다(0.5).

### 0.7 통과 기준. 여기까지 되면 W2-M3으로 넘어가도 된다

- [ ] `verify_dp_install.py --with-dataset`이 **FAIL 0**으로 끝난다 (Step 1, 2)
- [ ] 학습 스모크 300스텝의 손실이 **단조에 가깝게 내려간다** (Step 3)
- [ ] 논문 레시피 학습 로그에서 **정책 설정 6항목이 전부 바뀐 것**을 눈으로 확인했다 (Step 4)
- [ ] **`eval_info.json`과 롤아웃 mp4 2개가 디스크에 있다** (Step 5) ← **이것이 완주의 정의입니다**
- [ ] 사전학습 체크포인트가 왜 정공법이 아닌지 **자기 문장으로** 말할 수 있다 (Step 6)
- [ ] `worksheet.md` ⑥의 **§4.2 표 4행이 자기 기기 숫자로** 채워졌다 (Step 7)
- [ ] `worksheet.md` ⑦의 **완료 기준 한 문단**을 (a)(b)(c) 세 조각으로 썼다 (Step 8)

### 0.8 붙여넣으면 되는 것과 판단이 필요한 것

| | 그냥 붙여넣으세요 | 판단이 필요합니다 |
|---|---|---|
| **Step 1** | `uv pip install` · `verify_dp_install.py` | **FAIL과 WARN을 어떻게 가를 것인가.** 설치 문제인가 문서 문제인가 |
| **Step 2** | `--with-dataset` | `observation.state`의 대역이 왜 0~512인가 |
| **Step 3** | `lerobot-train --steps=300` | 손실이 0.064까지 내려간 것을 **"잘 배웠다"로 읽을 것인가** |
| **Step 4** | 6개 오버라이드가 붙은 학습 명령 | ★ **7항목 중 하나는 로그에 안 나옵니다. 어느 것이고 왜인가** |
| **Step 5** | `--eval.use_async_envs=false` | ★★ **`pc_success 0.0`을 어떻게 읽을 것인가.** 실패인가 정상인가 |
| **Step 6** | 두 개의 실패하는 명령 | ★ **막힌 것을 확인한 뒤 무엇을 정공법으로 삼을 것인가** |
| **Step 7·8** | `practice/04` 한 줄 | **나머지 전부** |

**판단 칸에서 막히면 정상입니다.** 붙여넣기 칸에서 막히면 [흔한 에러](#흔한-에러와-대처-7개)를 보세요.

### 0.9 practice와 이 랩의 분담

겹치지 않게 잘라뒀습니다. 같은 표를 두 번 확인하는 것이 아닙니다.

| | [`../practice/`](../practice/) | 이 랩 |
|---|---|---|
| 무엇과 대조하나 | lesson의 숫자를 **산술로** 재계산 | lesson의 숫자를 **설치본 실물과** 대조 |
| 무거운 것 | 없음. `01`~`03`은 표준 라이브러리만 | 데이터셋 · 학습 · 평가 · mp4 |
| §4.2 표 | `02`가 lesson 값으로 4행을 재계산 | Step 7이 **자기 기기 값으로** 다시 씀 |
| `drop_n_last_frames` | `02`가 31과 7을 **산술로** 재현 | `verify` [3]이 **설치본 기본값과 소스 주석**을 확인 |
| 파라미터 수 | `04`가 자기 기기에서 셈 | `verify` [5]가 lesson 값과 대조 |

---

## Step 1. extras 설치와 설치본 대조 (10분)

### 1.1 명령

```bash
# ① extras 얹기. 🔴 새 venv가 아니라 W2-M1의 .venv-lerobot에 얹습니다
uv pip install --python .venv-lerobot/bin/python "lerobot[pusht,diffusion,evaluation]"

# ② 무엇이 들어왔는지
.venv-lerobot/bin/python -c "import diffusers, gym_pusht, pymunk; print('ok')"

# ③ 설치본 자동 대조. 여기가 이 Step의 본체입니다
.venv-lerobot/bin/python course/w2-policy-vla/02-diffusion-policy/labs/verify_dp_install.py --quick
```

`--quick`은 모델을 만들지 않아 6.5초에 끝납니다. Step 2에서 전체를 돌립니다.

### 1.2 성공 판정 기준

```
================================================================================================
  W2-M2 labs. LeRobot Diffusion Policy 설치본 자동 대조기  (--quick)
  기대값 출처: /..../course/w2-policy-vla/02-diffusion-policy/lesson.md
================================================================================================

=== [1] 환경. 무엇이 깔려 있는가 ===

  항목         값                       비고
  -----------  -----------------------  --------------------------------------------
  python       3.12.12                  /..../.venv-lerobot/bin/python
  lerobot      0.6.1                    0.6.1에서 집필하고 검증
  torch        2.11.0+cu130             cuda_available=True
  diffusers    0.39.0                   🔴 없으면 정책 생성이 죽습니다. 에러 표 E3
  gym-pusht    0.1.6                    평가 환경. 에러 표 E1
  pymunk       6.11.1                   PushT의 2D 물리
  ...
  GPU          NVIDIA GeForce RTX 3080  cuda 13.0

=== [2] lesson §6.1 설정 드리프트 7항목 ↔ 설치본 기본값 ===

  필드                                   lesson 기본값                            설치본  대조  논문 레시피  드리프트
  -----------------------------------  ---------------  --------------------------------  ----  -----------  --------
  horizon                                           64                                64  PASS           16  다름
  n_action_steps                                    32                                32  PASS            8  다름
  crop_shape                                      None                              None  PASS   '[84, 84]'  다름
  use_group_norm                                 False                             False  PASS         True  다름
  pretrained_backbone_weights          ImageNet 가중치  'ResNet18_Weights.IMAGENET1K_V1'  PASS         None  다름
  use_separate_rgb_encoder_per_camera             True                              True  PASS        False  다름
  batch_size                                         8                                 8  PASS           64  다름

  대조 결과: 14/14 PASS   (7항목 × 2. 기본값 일치 + 논문 레시피와의 드리프트)
  기대값 출처: lesson.md §6.1
```

이어서 `[3]` `drop_n_last_frames`, `[4]` 제약, `[6]` `PushtEnv` 필드가 나오고 마지막에 종합이 찍힙니다.

```
================================================================================================
  종합. PASS 33 / FAIL 0 / WARN 0 / INFO 0  (총 33건)
================================================================================================

  [저장] /..../artifacts/W2-M2/labs/verify_dp_install.csv
```

| 판정 항목 | 기대 | 아니면 |
|---|---|---|
| **`[1]` diffusers** | **설치됨** | FAIL이면 `diffusion` extra가 빠졌습니다 (**E3**) |
| **`[1]` gym-pusht** | **설치됨** | FAIL이면 `pusht` extra가 빠졌습니다 |
| `[1]` 인터프리터 | `.venv-lerobot` 경로 | WARN이면 파이썬을 잘못 골랐습니다 (**E7**) |
| **`[2]` 드리프트 14항목** | **14/14 PASS** | ★ 하나라도 FAIL이면 lerobot 버전이 다릅니다 |
| `[3]` 소스 주석 | `configuration_diffusion.py:118` | 줄 번호는 달라도 됩니다. 찾기만 하면 PASS |
| `[4]` `horizon=9` 시도 | **`ValueError`** | 예외가 안 나면 제약이 사라진 것입니다 |
| **`--quick` 종합** | **PASS 33 / FAIL 0** | |
| **exit code** | **0** | FAIL이 있으면 1이 됩니다 |

### 1.3 여기서 판정할 것. FAIL과 WARN은 서로 다른 신호다

W2-M1의 `verify_act_install.py`와 같은 규약입니다.

| | 무엇인가 | exit code | 무엇을 해야 하나 |
|---|---|---|---|
| **FAIL** | **설치본이 lesson과 다르다** | **1** | 환경을 의심하세요. 버전이 어긋났을 가능성이 큽니다 |
| **WARN** | **설치본은 정상인데 lesson 쪽 표기나 실행 환경이 어긋난다** | 0 | **문서나 실행 방식을 고칠 대상**입니다. worksheet ②에 적으세요 |
| **INFO** | 판정 대상이 아닌 측정값 | 0 | 기록만 |

**이 스크립트가 lesson.md를 읽는다는 점이 중요합니다.** 기대값을 파일 안에 상수로 베껴두면 lesson이 개정될 때 스크립트만 낡아서 **정상인 문서를 계속 잡습니다.** W2-M1 랩에서 실제로 그 사고가 있었습니다(course-plan §9.9). 그래서 `[2]`는 lesson §6.1 표를 파싱하고, `[3]`은 산문의 `$16 - 8 - 2 + 1 = 7$`을 찾아 재계산하고, `[5]`는 §5.3의 파라미터 수를 읽습니다. **첫 줄의 "기대값 출처"가 `lesson.md` 경로면 정상**이고, "폴백 상수"라고 나오면 스크립트가 lesson을 못 찾은 것이라 대조의 의미가 절반으로 줄어듭니다.

### 1.4 여기서 볼 것. 드리프트 열이 전부 "다름"이라는 사실

`[2]` 출력의 마지막 열이 일곱 줄 전부 **"다름"** 입니다. 이것이 lesson §6.1의 주장이 설치본에서 그대로 재현됐다는 뜻입니다. **오버라이드 없이 `lerobot-train --policy.type=diffusion`을 돌리면 논문 재현이 아니라 다른 모델을 학습하는 것**입니다. Step 3과 Step 4가 그 두 모델을 각각 한 번씩 돌려 로그로 확인합니다.

`batch_size` 한 줄만 성격이 다릅니다. 나머지 여섯은 `DiffusionConfig`(정책 설정)에 있고 이것만 `TrainPipelineConfig`(학습 설정)에 있습니다. **파일이 달라서 놓치기 쉬운 자리**이고, Step 4에서 다시 걸립니다.

---

## Step 2. PushT 데이터셋을 자기 손으로 세기 (7분)

### 2.1 명령

```bash
.venv-lerobot/bin/python course/w2-policy-vla/02-diffusion-policy/labs/verify_dp_install.py --with-dataset
```

`--quick`을 뺐으므로 `[5]` 파라미터 수까지 세고, `--with-dataset`으로 `[7]`이 붙습니다. 전체 7.6초입니다.

### 2.2 성공 판정 기준

```
=== [5] 파라미터 수와 num_inference_steps 실물 해소 ===

  항목                        lesson.md §5.3       설치본  대조
  --------------------------  --------------  -----------  ----
  총 학습가능 파라미터           262,709,026  262,709,026  PASS
  그중 U-Net                     251,511,938  251,511,938  PASS
  나머지(시각 인코더와 부속)      11,197,088   11,197,088  PASS

  U-Net 비중 : 95.7%   ← 무게의 거의 전부가 몸통에 있습니다

  num_inference_steps 실물 : 100   [PASS]   (설정값 None → 해소)

=== [7] `lerobot/pusht` 메타. 자기 손으로 다시 세어본다 ===

  항목         lesson.md 「실습으로 가기」  설치본  대조
  -----------  ---------------------------  ------  ----
  에피소드                             206     206  PASS
  프레임                            25,650  25,650  PASS
  fps                                   10      10  PASS
  관측 해상도                        96×96   96×96  PASS

  평균 에피소드 : 124.5 프레임 = 12.5초
  카메라        : ['observation.image']
  action shape  : (2,)   state shape : (2,)

  observation.state 범위 : min [13.5, 32.9]  max [496.1, 511.0]

================================================================================================
  종합. PASS 41 / FAIL 0 / WARN 0 / INFO 0  (총 41건)
================================================================================================
```

| 판정 항목 | 기대 | 무엇을 뜻하는가 |
|---|---|---|
| **`[5]` 총 파라미터** | **262,709,026** | ★ Step 3의 학습 로그 `num_learnable_params`와 **정확히 같은 값**이어야 합니다 |
| `[5]` U-Net 비중 | **95.7%** | 무게가 전부 몸통입니다. 시각 인코더는 11.2M뿐 |
| **`[5]` `num_inference_steps` 실물** | **100** | ★ 설정값은 `None`인데 100으로 해소됩니다. **정책 한 번에 forward 100회** |
| `[7]` 에피소드 / 프레임 | **206 / 25,650** | |
| `[7]` fps | **10** | ALOHA의 50 Hz와 **5배** 다릅니다 |
| **`[7]` 관측 해상도** | **96×96** | ★ 2.4에서 판정 |
| **종합** | **PASS 41 / FAIL 0 / WARN 0** | |

### 2.3 여기서 볼 것. `observation.state`가 픽셀 좌표라는 증거

출력 마지막의 한 줄이 이 Step에서 눈으로 확인할 것입니다.

```
observation.state 범위 : min [13.5, 32.9]  max [496.1, 511.0]
```

관절각이면 이 대역이 나올 수 없습니다. 라디안이면 $\pm\pi$ 안이고 도 단위여도 360을 넘지 않습니다. **0에서 512 사이라는 것은 이 값이 좌표라는 뜻**이고, PushT의 시뮬 캔버스가 512×512입니다. 관측 이미지만 96×96으로 줄여서 주고 상태 벡터는 원래 좌표계 그대로 옵니다.

lesson §2.1이 "화면 픽셀 좌표계 위의 밀대 위치"라고 적은 것이 이 값입니다. 액션도 같은 좌표계의 2차원 목표 위치라 **위치 명령**이고, lesson §4.3이 "속도 명령보다 위치 명령에서 일관되게 더 좋았다"고 인용한 논문 §IV-B의 그 축입니다. 팀 질문 `W2M2-2`가 우리 하위 제어기는 어느 쪽인지 묻는 이유이기도 합니다.

### 2.4 여기서 판정할 것. 96과 384는 다른 숫자다

`[6]`의 출력을 다시 보세요.

```
gym_kwargs가 실제로 넘기는 것 : ['max_episode_steps', 'obs_type', 'render_mode', 'visualization_height', 'visualization_width']
observation_height / width   : 384 / 384
→ 두 필드는 gym_kwargs에 없습니다. [PASS]
```

`PushtEnv` 설정에는 `observation_height = observation_width = 384`가 있습니다. 데이터셋이 96×96인데 평가 환경이 384를 준다면 **학습과 평가의 해상도가 어긋나 성능이 무너질 상황**입니다. 그런데 `gym_kwargs`가 그 둘을 넘기지 않습니다. 넘어가는 것은 `visualization_*` 쪽이고, 그것은 **롤아웃 mp4의 해상도**입니다.

Step 5에서 이 두 숫자를 나란히 보게 됩니다. **관측은 96×96으로 들어가고 영상은 384×384로 저장됩니다.** 같은 실행에서 서로 다른 두 숫자가 나오는 것이 정상입니다.

| 숫자 | 어디에 쓰이나 | 어디서 오나 |
|---|---|---|
| **96×96** | 정책이 실제로 보는 관측 | 데이터셋 `observation.image` · `gym_pusht`의 기본 관측 |
| **384×384** | 롤아웃 mp4의 화면 | `PushtEnv.visualization_width/height` |
| **512×512** | 상태와 액션의 좌표계 | PushT 시뮬 캔버스. 2.3의 그 대역 |

---

## Step 3. 기본값으로 학습 스모크 300스텝 (12분)

### 3.1 명령

```bash
.venv-lerobot/bin/lerobot-train \
  --dataset.repo_id=lerobot/pusht --policy.type=diffusion \
  --policy.device=cuda --policy.push_to_hub=false \
  --steps=300 --batch_size=64 --save_freq=300 --log_freq=50 --wandb.enable=false \
  --output_dir=outputs/train/w2m2_dp_smoke --job_name=dp_pusht_smoke
```

세 인자에 이유가 있습니다.

- **`--policy.push_to_hub=false`** 없으면 `--policy.repo_id`를 요구합니다 (**E8**).
- **`--save_freq=300`** 총 스텝 수와 같게 줘서 체크포인트를 **마지막 하나만** 남깁니다. 하나가 3.0 GB입니다 (**E6**).
- **`--batch_size=64`** 논문이 이미지 실험에 쓴 값입니다. 기본값은 8이고, 이 한 줄이 §6.1 일곱 항목 중 하나를 미리 갚습니다.

🔴 **그래서 이 실행은 "완전한 기본값"이 아닙니다.** 정책 설정 여섯 항목은 전부 기본값이고 `batch_size`만 논문 값입니다. Step 4에서 나머지 여섯을 갚습니다.

### 3.2 성공 판정 기준

```
INFO ... num_learnable_params=262709026 (263M)
INFO ... step:50  ... loss:0.883 ... smp/s:...  mem_gb:5.08
INFO ... step:100 ... loss:0.204 ...
INFO ... step:150 ... loss:0.108 ...
INFO ... step:200 ... loss:0.080 ...
INFO ... step:250 ... loss:0.074 ...
INFO ... step:300 ... loss:0.064 ... mem_gb:4.97
INFO ... Checkpoint policy after step 300
```

`...`로 줄인 자리에는 `smpl`(누적 샘플 수), `ep`, `epch`(에폭 진행), `grdn`(그래디언트 노름), `updt_s`, `data_s`가 함께 찍힙니다. **집필 시점에 값을 기록해둔 것은 위에 적은 것뿐**이라 나머지는 생략했습니다. `epch`는 누적 샘플을 데이터셋 프레임 수로 나눈 값이고, 300스텝 × 배치 64 = 19,200을 25,650으로 나누면 **0.75 근처**가 나와야 맞습니다. Step 5에서 다시 씁니다.

| 판정 항목 | 기대 | 판정 방법 |
|---|---|---|
| `num_learnable_params` | **262709026 (263M)** | ★ **Step 2 `[5]`와 정확히 같은 값**입니다 |
| **`loss` 방향** | **0.883에서 0.064로 감소** | ★ 절대값은 흔들려도 됩니다. **내려가는 것**이 판정 기준입니다 |
| 속도 | **6.3~7.0 step/s** · `smp/s` 370~424 | 한참 느리면 GPU를 안 쓰고 있는지 보세요 |
| `mem_gb` | **약 5.0** | batch 64 · 카메라 1대 기준. 12 GB에 넉넉합니다 |
| 체크포인트 | **`checkpoints/000300`이 3.0 GB** | ★ **E6.** `model.safetensors`만 1,003 MB |
| 붉은 traceback | **떠도 정상** | 마지막 줄에 `Falling back to 'pyav'`가 있으면 됩니다 (**E4**) |
| 종료 코드 | **0** | |

```bash
du -sh outputs/train/w2m2_dp_smoke/checkpoints/000300
# 3.0G
du -h outputs/train/w2m2_dp_smoke/checkpoints/000300/pretrained_model/model.safetensors
# 1003M
du -h outputs/train/w2m2_dp_smoke/checkpoints/000300/training_state/optimizer_state.safetensors
# 2.0G
```

### 3.3 여기서 판정할 것. 손실 0.064를 "잘 배웠다"로 읽을 것인가

읽으면 안 됩니다. 이 손실은 **노이즈 예측 오차**입니다. lesson §3.2의 학습 목표를 다시 보세요. 신경망이 맞히는 대상은 액션이 아니라 주입된 노이즈 $\epsilon$이고, 손실은 그 제곱오차입니다.

$$
\mathcal{L}(\theta) \;=\; \mathbb{E}_{k,\;\epsilon}\Bigl\|\, \epsilon - \epsilon_\theta(\mathbf{O}_t,\; \mathbf{A}_t^{0} + \sigma_k \epsilon,\; k) \Bigr\|_2^2
$$

여기서 세 가지가 따라옵니다.

- **손실이 낮다고 태스크를 잘한다는 보장이 없습니다.** 노이즈가 큰 스텝($k$가 큰 쪽)은 맞히기 쉬워서 손실이 금방 내려갑니다. 300스텝에서 0.064까지 떨어진 것은 대부분 그쪽입니다.
- **그래서 체크포인트 선택을 손실로 할 수 없습니다.** lesson §2.2가 짚은 것과 결이 같습니다. 거기서는 암묵 정책 계열이 손실과 성능의 상관이 없어 체크포인트를 고를 근거가 사라진다고 했는데, 확산 정책은 그 문제가 없다고 적었습니다. **없다는 것은 상관이 강하다는 뜻이 아니라 진동하지 않는다는 뜻**입니다. 실제 선택 기준은 환경 평가 성공률이고, 그래서 `lerobot-train`에 `env_eval_freq`(기본 20,000)가 있습니다.
- **Step 5의 `pc_success 0.0`이 이 지점과 이어집니다.** 손실은 14배 내려갔는데 성공률은 0입니다. 모순이 아니라 두 지표가 다른 것을 재고 있는 것입니다.

팀 질문 `W2M2-5`가 "체크포인트 선택 기준과 평가 프로토콜은 무엇인가"를 묻는 이유가 여기 있습니다.

### 3.4 설정이 실제로 무엇으로 저장됐는지 보기

체크포인트에 `config.json`이 함께 저장됩니다. Step 4의 대조 기준이 되므로 지금 한 번 찍어두세요.

```bash
.venv-lerobot/bin/python -c "
import json,sys
d=json.load(open(sys.argv[1]))
for k in ('horizon','n_action_steps','crop_shape','use_group_norm','pretrained_backbone_weights','use_separate_rgb_encoder_per_camera','drop_n_last_frames'):
    print(f'{k:38s} {d.get(k)!r}')
" outputs/train/w2m2_dp_smoke/checkpoints/000300/pretrained_model/config.json
```

```
horizon                                64
n_action_steps                         32
crop_shape                             None
use_group_norm                         False
pretrained_backbone_weights            'ResNet18_Weights.IMAGENET1K_V1'
use_separate_rgb_encoder_per_camera    True
drop_n_last_frames                     7
```

여섯 줄이 전부 §6.1 표의 **왼쪽 열**입니다. 일곱 번째 줄 `drop_n_last_frames`가 7인데, 이 설정에 공식을 넣으면 $64 - 32 - 2 + 1 = 31$입니다. **lesson §6.2가 적은 어긋남을 자기 체크포인트에서 확인한 것**이고, `verify` `[3]`이 방금 산술로 확인해준 것과 같은 자리입니다.

---

## Step 4. 논문 레시피로 다시 돌리기 ★ (15분)

### 4.1 명령

```bash
SCR=outputs/train/w2m2_dp_paper
.venv-lerobot/bin/lerobot-train \
  --dataset.repo_id=lerobot/pusht --policy.type=diffusion \
  --policy.horizon=16 --policy.n_action_steps=8 --policy.crop_shape="[84,84]" \
  --policy.use_group_norm=true --policy.pretrained_backbone_weights=null \
  --policy.use_separate_rgb_encoder_per_camera=false \
  --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=64 --steps=20 --log_freq=10 --save_checkpoint=false --wandb.enable=false \
  --output_dir=$SCR --job_name=dp_pusht_paper 2>&1 | tee /tmp/dp_paper.log
```

🔴 **`--policy.crop_shape="[84,84]"`의 따옴표를 생략하면 안 됩니다.** 셸이 대괄호를 파일명 확장으로 먹습니다 (**E9**).

🔴 **`--save_checkpoint=false`가 이 명령의 핵심 편의입니다.** 여기서 확인할 것은 설정이 반영되는가 하나뿐이라 3.0 GB를 쓸 이유가 없습니다. 이 플래그를 주면 출력 디렉터리 자체가 만들어지지 않습니다.

`--steps=20`인 이유도 같습니다. **설정 반영은 로그 첫머리에서 결정되므로 20스텝이면 충분합니다.** 27.7초에 끝납니다.

### 4.2 성공 판정 기준

```
INFO ... {'batch_size': 64,
 ...
 'policy': {'beta_end': 0.02,
            ...
            'crop_shape': [84, 84],
            'horizon': 16,
            'n_action_steps': 8,
            'pretrained_backbone_weights': None,
            'use_group_norm': True,
            'use_separate_rgb_encoder_per_camera': False,
            ...
INFO ... num_learnable_params=262709026 (263M)
INFO ... step:10 smpl:640 ep:5  epch:0.02 loss:1.130 grdn:12.527 updt_s:0.730 mem_gb:5.54
INFO ... step:20 smpl:1K  ep:10 epch:0.05 loss:1.032 grdn:9.972  updt_s:0.134 mem_gb:4.97
```

여섯 줄을 눈으로 세는 대신 이렇게 뽑으면 됩니다.

```bash
grep -E "'(horizon|n_action_steps|crop_shape|use_group_norm|pretrained_backbone_weights|use_separate_rgb_encoder_per_camera)':" /tmp/dp_paper.log
# 정확히 6줄이 나와야 합니다

grep -oE "\{'batch_size': [0-9]+" /tmp/dp_paper.log | head -1
# {'batch_size': 64
```

| 판정 항목 | 기대 | 판정 방법 |
|---|---|---|
| **`horizon`** | **16** | ★ 기본값 64에서 바뀜 |
| **`n_action_steps`** | **8** | ★ 기본값 32에서 바뀜 |
| **`crop_shape`** | **`[84, 84]`** | ★ 기본값 `None`에서 바뀜. 따옴표를 뺐으면 여기가 틀립니다 |
| **`use_group_norm`** | **`True`** | ★ 기본값 `False`에서 바뀜 |
| **`pretrained_backbone_weights`** | **`None`** | ★ 기본값 ImageNet 가중치에서 바뀜 |
| **`use_separate_rgb_encoder_per_camera`** | **`False`** | ★ 기본값 `True`에서 바뀜 |
| **`batch_size`** | **64** | ★ 4.3에서 판정 |
| `num_learnable_params` | **262709026** | ★ **Step 3과 같습니다.** 4.4에서 판정 |
| `loss` | 1.130에서 1.032 | 20스텝이라 방향만 봅니다 |
| 체크포인트 | **생기지 않습니다** | `--save_checkpoint=false`가 걸렸다는 증거. `ls $SCR`가 "No such file" |
| 완주 | 로그 끝에 `Training: 100%\|...\| 20/20` | 🔴 `tee`로 파이프했으므로 `$?`는 `tee`의 것입니다. **로그 마지막 줄로 판정하세요** |

### 4.3 ★ 여기서 판정할 것 ①. 일곱 항목 중 하나는 로그의 다른 자리에 있다

여섯 개는 `'policy':` 블록 안에 들여쓰기된 채로 나오고, `batch_size`는 **로그 맨 앞 최상위 딕셔너리**에 있습니다. 정책 설정이 아니라 학습 설정이기 때문입니다.

```
INFO ... {'batch_size': 64,          ← 최상위. TrainPipelineConfig
 'policy': {'beta_end': 0.02,        ← 여기부터가 DiffusionConfig
            'horizon': 16,
```

같은 로그에 `'eval': {'batch_size': 14,`도 나오는데 **이것은 또 다른 `batch_size`** 입니다. 평가 시 병렬 환경 개수이고 CPU 코어 수에서 자동으로 정해집니다. Step 5에서 다시 만납니다.

**세 개의 `batch_size`가 한 로그에 있습니다.** 학습 배치(64), 평가 환경 개수(14 자동), 그리고 §6.1 표가 말하는 기본값(8)입니다. worksheet ④에 셋을 구분해 적으세요. 이것이 lesson §6.1이 "파일이 달라 놓치기 쉬운 자리"라고 적은 것의 실물입니다.

### 4.4 ★ 여기서 판정할 것 ②. 설정을 넷 배로 줄였는데 파라미터가 그대로다

`horizon`이 64에서 16으로, `n_action_steps`가 32에서 8로 줄었습니다. 그런데 `num_learnable_params`는 **262709026으로 Step 3과 한 자리도 다르지 않습니다.**

이유는 lesson §5.2와 §5.3에 있습니다. 몸통이 **시간축 1차원 합성곱**이라 시퀀스 길이가 채널 수를 바꾸지 않고, 시각 인코더의 공간 소프트맥스 출력도 해상도에 무관합니다. `crop_shape`를 84로 준 것도 파라미터를 바꾸지 않습니다.

**여기서 따라오는 결론이 이 랩에서 가장 실무적인 것 중 하나입니다.**

- 파라미터 수가 같다고 **같은 모델이 아닙니다.** 두 실행은 예측 구간도 실행 구간도 정규화도 백본 초기화도 다릅니다.
- 그러므로 **모델 크기를 비교 근거로 쓸 수 없습니다.** "263M짜리 Diffusion Policy 기준으로 몇 % 개선"이라는 문장은 어느 설정인지 특정하지 못합니다.
- lesson §6.1의 결론이 이것입니다. **비교 기준으로 이 모델을 인용할 때는 라이브러리 버전과 오버라이드한 설정을 함께 적어야 그 비교가 재현 가능해집니다.**

### 4.5 여기서 볼 것. 손실 절대값을 두 실행 사이에서 비교하지 말 것

Step 3의 20스텝 근처 손실과 Step 4의 20스텝 손실을 나란히 놓고 싶어집니다. 놓지 마세요. 두 실행은 **정규화도 백본 초기화도 자르기도 다릅니다.** 특히 `pretrained_backbone_weights=null`이라 논문 레시피 쪽은 시각 인코더가 무작위에서 시작합니다. 초기 손실이 더 높게 나오는 것이 당연하고, 그 차이는 학습의 우열이 아닙니다.

**비교 가능한 것은 방향뿐입니다.** 둘 다 내려가면 둘 다 정상입니다.

---

## Step 5. 평가와 롤아웃 mp4 ★★ (20분)

**이 Step이 완주의 정의입니다.** 여기까지 오면 마스터플랜이 이 모듈에 건 요구를 채운 것입니다.

### 5.1 먼저 한 번 실패해 보세요

기본값 그대로 돌리면 죽습니다. **일부러 한 번 보고 넘어갑니다.** 이 에러가 이 랩에서 가장 오해를 많이 사는 자리라서입니다.

```bash
.venv-lerobot/bin/lerobot-eval \
  --policy.path=outputs/train/w2m2_dp_smoke/checkpoints/last/pretrained_model \
  --env.type=pusht --eval.n_episodes=2 --policy.device=cuda \
  --output_dir=outputs/eval/w2m2_fail_demo
```

```
gymnasium.error.NamespaceNotFound: Namespace gym_pusht not found. Have you installed the proper package for gym_pusht?
...
ConnectionResetError: [Errno 104] Connection reset by peer
```

**설치가 잘못된 것이 아닙니다.** 바로 확인할 수 있습니다.

```bash
.venv-lerobot/bin/python -c "import gym_pusht; print('부모 프로세스에서는 멀쩡합니다')"
# 부모 프로세스에서는 멀쩡합니다
```

원인은 **평가용 병렬 환경의 워커 프로세스**입니다. `EvalConfig.use_async_envs`가 기본 `True`이고 환경이 2개 이상이면 `AsyncVectorEnv`가 선택되고(`envs/configs.py:93`), 그 생성자에 `context="forkserver"`가 붙습니다(`envs/configs.py:112`). **forkserver 워커는 부모의 `import gym_pusht`를 상속하지 않습니다.** 워커 쪽 gym 레지스트리에 `gym_pusht/PushT-v0`가 없어 이름을 못 찾고, 그 뒤 부모가 워커와의 연결을 잃으면서 `ConnectionResetError`가 따라 붙습니다. 자세한 것은 에러 표 **E1**입니다.

에피소드 수만 주고 배치를 안 줘도 이 경로로 갑니다. `EvalConfig.batch_size`의 기본값이 0이고 그러면 CPU 코어 수로 자동 결정되는데, `n_episodes`를 넘지 않게 잘립니다. `--eval.n_episodes=2`면 배치가 2가 되어 **1보다 커집니다.**

### 5.2 명령

```bash
.venv-lerobot/bin/lerobot-eval \
  --policy.path=outputs/train/w2m2_dp_smoke/checkpoints/last/pretrained_model \
  --env.type=pusht --eval.batch_size=2 --eval.n_episodes=2 \
  --eval.use_async_envs=false --policy.device=cuda \
  --output_dir=outputs/eval/w2m2_smoke_own
```

**`--eval.use_async_envs=false` 한 줄이 5.1의 에러를 넘습니다.** `--eval.batch_size=1`도 우회가 되는데, 배치가 1이면 라이브러리가 알아서 `SyncVectorEnv`로 강등하기 때문입니다. 다만 그러면 에피소드를 하나씩 굴려 느려집니다.

### 5.3 성공 판정 기준

```
INFO ... {'avg_max_reward': 0.008472222570179341,
 'avg_sum_reward': 0.24730972038236465,
 'eval_ep_s': 6.538316607475281,
 'eval_s': 13.076633214950562,
 'n_episodes': 2,
 'pc_success': 0.0, ...}
```

```bash
ls -la outputs/eval/w2m2_smoke_own/videos/pusht_0/
# eval_episode_0.mp4   145167
# eval_episode_1.mp4   121290

.venv-lerobot/bin/python -c "
import av,sys
s=av.open(sys.argv[1]).streams.video[0]
print(f'{s.codec_context.width}x{s.codec_context.height}  {s.frames} frames  {float(s.average_rate):.0f} fps  {s.frames/float(s.average_rate):.0f} s')
" outputs/eval/w2m2_smoke_own/videos/pusht_0/eval_episode_0.mp4
# 384x384  300 frames  10 fps  30 s
```

| 판정 항목 | 기대 | 아니면 |
|---|---|---|
| **mp4 2개** | **`videos/pusht_0/eval_episode_{0,1}.mp4`** | ★ **이것이 완주의 증거입니다.** 없으면 롤아웃이 저장되지 않은 것 |
| **mp4 속성** | **384×384 · 300 프레임 · 10 fps · 30초** | ★ Step 2.4의 두 숫자가 여기서 만납니다 |
| `eval_info.json` | `per_task` · `per_group` · `overall` 3단 | 집계가 세 층입니다 |
| **`pc_success`** | **0.0** | ★ **이것이 정상입니다.** 5.4에서 판정 |
| `avg_max_reward` | **약 0.0085** | 스모크 체크포인트라 바닥입니다 |
| `eval_s` / `eval_ep_s` | **13.1 / 6.5초** | 에피소드당 6.5초, 최대 300스텝 |
| 뷰어 창 | **뜨면 안 됩니다** | 이 파이프라인은 파일로만 저장합니다 |
| 종료 코드 | **0** | `NamespaceNotFound`면 **E1** |

### 5.4 ★★ 여기서 판정할 것. `pc_success 0.0`은 실패인가

**실패가 아닙니다. 300스텝짜리 체크포인트에서 이것 말고 다른 값이 나오면 그쪽을 의심해야 합니다.**

논문 레시피의 본 학습은 10만 스텝이고 약 4.4시간입니다. 300스텝은 그 **0.3%** 입니다. 데이터셋이 25,650 프레임인데 배치 64로 300스텝이면 19,200 샘플, 곧 **전체를 한 바퀴도 못 돈 상태**입니다(Step 3 로그의 `epch` 열이 0.75 근처면 이 계산이 맞은 것입니다). 이 상태에서 T자 블록을 목표 자세로 미는 데 성공한다면 그것이 이상합니다.

**그런데 파이프라인은 전부 통과했습니다.** 이 구분이 이 Step의 판정입니다.

| 무엇을 확인했나 | 이번 실행으로 확인됐나 |
|---|---|
| 체크포인트가 로드되는가 | ✅ |
| 정책이 환경에서 액션을 내는가 | ✅ |
| 300스텝 에피소드가 끝까지 굴러가는가 | ✅ |
| 보상과 성공 판정이 집계되는가 | ✅ `eval_info.json` 3단 |
| 롤아웃이 mp4로 저장되는가 | ✅ 2개 |
| **정책이 태스크를 푸는가** | ❌ **이건 학습량의 문제이고 파이프라인의 문제가 아닙니다** |

Step 3.3에서 "손실이 14배 내려갔는데 성공률이 0"이라고 예고한 자리가 여기입니다. **두 지표는 다른 것을 잽니다.** 손실은 노이즈 예측 오차이고 성공률은 태스크 달성입니다. 체크포인트를 손실로 고를 수 없다는 것이 이 대비의 실물 증거입니다.

### 5.5 ★★ 롤아웃 영상을 실제로 보세요. 이 Step에서 제일 중요합니다

숫자만 보고 넘어가면 이 랩을 절반만 한 것입니다. **mp4를 열어서 정책이 무엇을 하고 있었는지 한 문단으로 적는 것**이 worksheet ⑤의 항목입니다.

헤드리스 환경이라 창을 띄우지 않습니다. 파일을 로컬로 가져가 보세요.

```bash
# WSL2라면 윈도우 쪽으로 복사해서 열면 됩니다
cp outputs/eval/w2m2_smoke_own/videos/pusht_0/eval_episode_0.mp4 /mnt/c/Users/$USER/Desktop/

# 원격 서버라면
# scp user@host:~/project/.../eval_episode_0.mp4 .
```

**볼 때 이 셋을 보세요.**

- **밀대가 움직이기는 하는가.** 완전히 굳어 있다면 액션이 안 나오는 것이라 파이프라인 문제입니다. 움직이는데 엉뚱하면 학습량 문제입니다.
- **움직임이 매끄러운가 끊기는가.** 32스텝마다 한 번씩 재계획하므로 **3.2초에 한 번** 새 궤적이 시작됩니다. 그 경계에서 방향이 튀는지 보세요. lesson §4.3이 말한 "실행 구간을 길게 잡으면 반응이 느려진다"의 눈에 보이는 형태입니다.
- **T자 블록에 닿기는 하는가.** `avg_max_reward`가 0.0085라는 것은 목표와의 겹침이 거의 0이라는 뜻입니다. 영상에서 그것이 어떤 모습인지 확인하세요.

30초짜리 두 편입니다. 두 편 다 보세요. **같은 정책이 서로 다른 초기 상태에서 무엇을 하는지 보는 것**이 lesson §3.3의 "매번 다시 뽑는 초기값"과 이어집니다.

---

## Step 6. 사전학습 체크포인트가 막히는 자리 ★ (15분)

### 6.1 왜 이 Step이 있는가

Step 5의 성공률이 0.0으로 나오면 자연스럽게 드는 생각이 있습니다. **"제대로 학습된 것을 받아서 돌려보면 되지 않나."** 허브에 `lerobot/diffusion_pusht`가 있고, lesson §6.1이 그것을 논문 레시피의 근거로 인용하기까지 했습니다.

**막혀 있습니다.** 그래서 이 Step은 성공하지 않습니다. **두 번 실패하는 것을 확인하고 왜 직접 학습본이 정공법인지 결론을 내리는 Step**입니다. 학습자가 여기서 한 시간을 태우는 것을 막으려고 넣었습니다.

### 6.2 명령 ①. 그냥 평가해 보기

```bash
.venv-lerobot/bin/lerobot-eval \
  --policy.path=lerobot/diffusion_pusht \
  --env.type=pusht --eval.batch_size=1 --eval.n_episodes=1 \
  --eval.use_async_envs=false --policy.device=cuda \
  --output_dir=outputs/eval/w2m2_pretrained_try
```

```
lerobot.processor.pipeline.ProcessorMigrationError: Model 'lerobot/diffusion_pusht' requires migration to processor format.
Run: python src/lerobot/processor/migrate_policy_normalization.py --pretrained-path lerobot/diffusion_pusht
Original error: Config file 'policy_preprocessor.json' not found on the Hugging Face Hub
```

lerobot 0.6.1은 정책 옆에 **전처리기와 후처리기 설정 파일**을 요구합니다. 정규화 통계를 정책 가중치에서 분리해 파이프라인으로 옮긴 구조 변경이고, 허브의 그 체크포인트는 변경 이전 포맷이라 파일이 없습니다.

### 6.3 명령 ②. 에러가 시키는 대로 해 보기

에러 메시지의 명령을 그대로 붙여넣으면 **파일을 못 찾습니다.** `src/lerobot/...`는 소스 트리 경로이고 pip로 설치한 환경에는 그런 디렉터리가 없습니다. 설치본에서 경로를 뽑아야 합니다.

```bash
MIG=$(.venv-lerobot/bin/python -c "import lerobot.processor.migrate_policy_normalization as m; print(m.__file__)")
echo $MIG
# /..../.venv-lerobot/lib/python3.12/site-packages/lerobot/processor/migrate_policy_normalization.py

.venv-lerobot/bin/python "$MIG" \
  --pretrained-path lerobot/diffusion_pusht \
  --output-dir outputs/migrated/diffusion_pusht
```

```
...
Exception: Couldn't encode 84
```

**중간까지는 갑니다.** `config.json`과 processor 4종이 만들어집니다. 그런데 `model.safetensors`가 나오기 전에 멈춥니다.

```bash
ls outputs/migrated/diffusion_pusht
# config.json  policy_postprocessor.json  policy_preprocessor.json  ... (safetensors 없음)
```

원인은 설정 직렬화입니다. `crop_shape`가 `[84, 84]`인데 직렬화 라이브러리가 그 안의 정수 84를 인코딩하지 못하고 예외를 냅니다. **논문 레시피의 그 자르기 설정이 마이그레이션을 막는 것**이라 아이러니한 자리입니다.

### 6.4 ★ 여기서 판정할 것. 그래서 무엇이 정공법인가

| 경로 | 상태 | 왜 |
|---|---|---|
| 허브 사전학습본을 바로 평가 | ❌ 막힘 | 전처리기 파일이 없어 `ProcessorMigrationError` |
| 동봉 마이그레이션 스크립트로 변환 | ❌ 막힘 | `crop_shape`의 84 인코딩에서 중단, 가중치가 안 나옴 |
| **직접 학습한 체크포인트를 평가** | ✅ **된다** | ★ 처음부터 processor 포맷으로 저장됩니다. Step 5가 그 증거 |

**결론은 하나입니다. 이 랩의 정공법은 자기가 학습한 체크포인트를 평가하는 것이고, Step 3에서 Step 5로 가는 경로가 그것입니다.**

여기서 얻어야 할 것이 명령 하나가 아닙니다. **저장 포맷이 바뀌면 남이 만든 체크포인트는 그냥 못 씁니다.** 회사 스택에서 이 문제가 어디에 걸리는지 생각해 보세요. GEAR-SONIC이든 HOMIE든 외부 체크포인트를 받아 쓰는 순간 같은 종류의 벽을 만납니다. 팀 질문 `W2M2-3`이 "베이스라인 재현 레시피가 팀에 고정돼 있는가"를 묻는 이유이고, worksheet ⑧에 이 관점의 질문을 하나 추가하세요.

### 6.5 결과 정리와 디스크 회수

Step 5까지 확인했으면 학습 산출물을 지워도 됩니다. 3.0 GB가 돌아옵니다.

```bash
du -sh outputs/train/w2m2_dp_smoke outputs/eval/*
# 🔴 mp4와 eval_info.json은 남기고 싶으면 먼저 다른 곳으로 옮기세요
mkdir -p artifacts/W2-M2/labs
cp -r outputs/eval/w2m2_smoke_own artifacts/W2-M2/labs/
rm -rf outputs/train/w2m2_dp_smoke outputs/migrated
```

> 📌 `outputs/`는 이 저장소 `.gitignore`에 등재돼 있어 **커밋되지는 않습니다. 디스크만 먹습니다.**
> 📌 `artifacts/`의 대용량도 gitignore 대상입니다. mp4 두 편은 합쳐 266 KB라 부담이 없습니다.

---

## Step 7. 자기 기기 지연으로 §4.2 표 다시 쓰기 ★★ (25분)

**여기부터 코드가 거의 없습니다.** 명령 한 줄을 돌리고 나머지는 전부 계산과 글입니다.

### 7.1 명령

```bash
cd course/w2-policy-vla/02-diffusion-policy/practice
source ../../../../.venv-lerobot/bin/activate
python 04_dp_latency_bench.py --device cuda
deactivate; cd -
```

31초입니다. 흔들림이 크면 `--reps 30`으로 다시 재세요.

### 7.2 성공 판정 기준

```
  설정                          NFE  lesson  내 중앙값  내 최소  내 최대    비율  0.3~3.0배
  DDIM 1, 바닥                    1     9.4       10.3      9.2     13.1  1.10배  PASS
  DDIM 10, 논문 §III-D           10    68.0       68.8     66.9     72.5  1.01배  PASS
  DDIM 16, 논문 부록 실기 설정   16   107.1      106.4    104.9    110.2  0.99배  PASS
  DDPM 100, 라이브러리 기본값   100   665.2      672.3    661.7    818.2  1.01배  PASS
  DDPM 100, 논문 PushT 설정     100   705.2      690.6    682.0    700.0  0.98배  PASS

  NFE 1 (10.3 ms)  <  NFE 10 (68.8 ms)  <  NFE 16 (106.4 ms)  <  NFE 100 (681.5 ms)
  단조 증가인가   [PASS]

  대조 결과: 19/19 PASS
```

| 판정 항목 | 기대 | 판정 방법 |
|---|---|---|
| **단조 증가** | **PASS** | ★ **진짜 판정 항목입니다.** NFE가 오르면 지연도 올라야 합니다 |
| 비율 밴드 | **0.3~3.0배** | 절대값은 기기마다 다릅니다. 밴드 안이면 됩니다 |
| 액션 청크 모양 | 논문 설정에서 `(1, 8, 2)` | $T_a=8$이 실제로 걸렸다는 증거 |
| 파라미터 수 | **262,709,026** | Step 2 `[5]`, Step 3 로그와 세 번째로 같은 값 |
| 종합 | **19/19 PASS** | `--smoke`면 13/13 |

> ⚠️ **절대값을 이 문서와 맞추려 하지 마세요.** 같은 기기에서 두 번 돌린 것만으로 DDIM 16이 106.4 ms와 135.6 ms로 갈렸습니다([`../practice/README.md`](../practice/README.md) §3.4). 자기 값을 기록할 때는 **한 번의 숫자가 아니라 몇 번 돌린 범위**로 적으세요.

### 7.3 ★★ 여기서 계산할 것. lesson §4.2 표 4행을 자기 숫자로

**이것이 worksheet ⑥이고 이 랩의 두 번째 제출물입니다.**

lesson §4.2의 표는 이렇게 생겼습니다.

| 실행 방식 | $T_a$ | 재계획 주기 @10 Hz | 추론 실측 | 예산 안에 드는가 | 최악 반응 지연 |
|---|---|---|---|---|---|
| 스텝마다 재계획 | 1 | 100 ms | 665 ms (DDPM 100) | 아니오, 6.65배 초과 | 100 ms |
| 논문 레시피 | 8 | 800 ms | 705 ms (DDPM 100) | 예, 여유 95 ms | 800 ms |
| 논문 실기 설정 | 8 | 800 ms | 107 ms (DDIM 16) | 예, 여유 693 ms | 800 ms |
| 라이브러리 기본값 | 32 | 3,200 ms | 665 ms (DDPM 100) | 예, 여유 2,535 ms | 3,200 ms |

**세 열을 자기 값으로 다시 채웁니다.** 나머지 세 열은 설정에서 나오는 값이라 바뀌지 않습니다.

$$
T_{\text{replan}} = \frac{T_a}{f_2}, \qquad
\text{여유} = \frac{T_a}{f_2} - \tau_{\text{infer}}, \qquad
\text{최악 반응 지연} = \frac{T_a}{f_2}
$$

$f_2 = 10$ Hz이므로 $T_a$ 스텝이 곧 $100 T_a$ ms입니다. 통신 지연 $\tau_{\text{comm}}$은 시뮬이라 0으로 둡니다.

**두 번째 행을 특히 조심해서 계산하세요.** 집필 기기에서 여유가 95 ms뿐이었습니다. 자기 기기의 `DDPM 100, 논문 PushT 설정` 중앙값이 **800 ms를 넘으면 그 행의 판정이 "초과"로 뒤집힙니다.** 그러면 lesson이 "아슬아슬하다"고 적은 것이 자기 기기에서는 "성립하지 않는다"가 됩니다. 어느 쪽이든 **worksheet에 자기 판정을 쓰세요.** 문서를 베끼면 안 됩니다.

### 7.4 여기서 볼 것. 계층 판정이 자기 숫자에서도 유지되는가

lesson §7.1이 세 문장으로 자리를 갈랐습니다. 자기 값으로 다시 확인하세요.

| 계층 | 스텝 예산 | NFE 100 | NFE 1 | lesson의 판정 |
|---|---|---|---|---|
| L2 전신 제어 500 Hz | 2 ms | ❌ | ❌ | 산술적으로 불가능 |
| L2 전신 제어 50 Hz | 20 ms | ❌ | ⚠️ 들어감 | 불가능 (실제 쓰는 설정 기준) |
| PushT 10 Hz, 스텝마다 | 100 ms | ❌ | ✅ | 6.65배 초과 |
| L4 상위 지능 1 Hz | 1,000 ms | ✅ | ✅ | 가능 |

⚠️ 표시한 칸이 결을 요구하는 자리입니다. **NFE 1이면 L2 하단에 들어갑니다.** `practice/03`이 이미 이것을 결로 남겨뒀습니다(그 README §4.3의 `[5]`). lesson §7.1의 "L2 불가"는 **실제로 쓰는 설정인 NFE 100에 대한 판정**으로 읽어야 맞고, NFE 1 단독은 대역의 느린 끝에서만 성립하며 품질까지 보존한다는 뜻이 아닙니다(lesson §5.3 단서).

**worksheet ⑥에 이 결을 한 문장으로 적으세요.** "L2 불가"를 조건 없이 외워두면 다음 모듈에서 Flow Matching이 NFE를 4~10으로 내렸을 때 판정을 다시 못 합니다.

---

## Step 8. 완료 기준 한 문단 ★★ (35분)

**코드가 없습니다. 전부 글입니다. 이 랩의 최종 제출물입니다.**

### 8.1 무엇을 쓰는가

lesson이 요구하는 완료 기준을 이 랩의 형태로 옮기면 이렇습니다.

> 라이브러리 기본값으로 PushT를 돌린 결과를 놓고
> **(a)** 이것이 논문 재현이 아닌 이유를 설정 7항목으로 지목하고
> **(b)** 자기 기기의 추론 지연으로 10 Hz 예산 대비 몇 배인지 계산하고
> **(c)** 그럼에도 태스크가 도는 이유를 receding horizon으로 설명하는
> **한 문단**

### 8.2 재료는 이미 다 있습니다

| 조각 | 어디서 나왔나 | worksheet 어디 |
|---|---|---|
| (a) 7항목 | Step 1 `verify [2]` · Step 3 `config.json` · Step 4 학습 로그 | ② · ④ |
| (b) 배수 | Step 7 `practice/04`의 자기 중앙값 ÷ 100 ms | ⑥ |
| (c) receding horizon | Step 5 롤아웃 영상 · lesson §4.1과 §4.2 | ⑤ · ⑥ |

### 8.3 쓸 때의 규칙 세 가지

- 🔴 **숫자가 안 붙으면 lesson 요약을 옮겨 적은 것입니다.** (b)에는 자기 기기의 ms가, (a)에는 일곱 개의 필드명이 들어가야 합니다.
- 🔴 **(c)가 제일 어렵습니다.** "receding horizon을 쓰니까 된다"는 설명이 아닙니다. **실행 구간 길이가 재계획 주기를 사고, 그 대가가 최악 반응 지연이라는 거래 구조**를 써야 합니다. 기본값에서 그 대가가 3,200 ms이고 [W1-M1](../../../w1-generative-core/01-physical-ai-landscape/lesson.md) §3.3의 200 ms 문턱의 16배라는 것까지 오면 완성입니다.
- 🔴 **평면 밀기와 전신 로봇의 차이를 한 줄로 덧붙이세요.** PushT에는 낙상 모드가 없어 3.2초를 감당하지만 G1은 아닙니다. 이것이 lesson §7.2의 두 번째 항목입니다.

### 8.4 자가 채점

쓴 문단을 놓고 아래를 확인하세요. 셋 다 되면 이 모듈을 소화한 것입니다.

- [ ] 일곱 항목을 **필드명으로** 댈 수 있다. "설정이 다르다"가 아니라 `horizon`, `n_action_steps`, `crop_shape`, `use_group_norm`, `pretrained_backbone_weights`, `use_separate_rgb_encoder_per_camera`, `batch_size`
- [ ] 배수 계산에 **자기 기기의 숫자**가 들어 있다
- [ ] "그럼에도 도는 이유"에 **최악 반응 지연이라는 대가**가 함께 적혀 있다

### 8.5 마무리 세 곳

- **`worksheet.md`** 전부 기입. 특히 ⑥ 표 재계산과 ⑦ 완료 기준 문단
- **[`../../../../docs/progress.md`](../../../../docs/progress.md)** 돌린 것, 막힌 지점, 소요 시간, 디스크 사용량. 자기 기기의 `practice/04` 값도 여기에
- **[`../../../../notes/questions-for-team.md`](../../../../notes/questions-for-team.md)** Step 6에서 새로 생긴 질문. 실습을 하고 나서야 생긴 질문이 제일 값집니다

---

## 흔한 에러와 대처, 7개

**E1과 E2가 이 랩의 대표 에러입니다.** 둘 다 Step 본문에서 정면으로 다뤘고, 여기는 요약과 판정 기준입니다.

### E1. ★★ `NamespaceNotFound: Namespace gym_pusht not found`

```
gymnasium.error.NamespaceNotFound: Namespace gym_pusht not found. Have you installed the proper package for gym_pusht?
...
ConnectionResetError: [Errno 104] Connection reset by peer
```

| | |
|---|---|
| **언제 뜨나** | `lerobot-eval`을 **기본 설정으로** 돌릴 때. Step 5.1 |
| **원인** | `EvalConfig.use_async_envs`가 기본 `True`이고 환경이 2개 이상이면 `AsyncVectorEnv`(`envs/configs.py:93`)가 `context="forkserver"`(`:112`)로 뜹니다. **forkserver 워커는 부모의 `import gym_pusht`를 상속하지 않습니다** |
| **오해 지점** | 🔴 **설치 문제로 보입니다.** 실제로는 부모 프로세스에서 `import gym_pusht`가 멀쩡합니다. 여기서 재설치를 반복하며 시간을 태우기 딱 좋습니다 |
| **판정** | `.venv-lerobot/bin/python -c "import gym_pusht"`가 성공하면 설치는 정상입니다 |
| **대처** | **`--eval.use_async_envs=false`** (확인함). `--eval.batch_size=1`도 Sync로 강등돼 우회됩니다 |
| **대가** | 환경을 순차로 굴리므로 에피소드가 많으면 느려집니다. 2에피소드에 13.1초였습니다 |

### E2. ★★ 사전학습 체크포인트가 두 단계 모두에서 막힌다

```
lerobot.processor.pipeline.ProcessorMigrationError: Model 'lerobot/diffusion_pusht' requires migration to processor format.
Run: python src/lerobot/processor/migrate_policy_normalization.py --pretrained-path lerobot/diffusion_pusht
Original error: Config file 'policy_preprocessor.json' not found on the Hugging Face Hub
```

| | |
|---|---|
| **언제 뜨나** | `--policy.path=lerobot/diffusion_pusht`로 평가할 때. Step 6.2 |
| **원인** | 0.6.1은 정규화를 정책에서 분리해 processor 파일로 두는데, 허브의 그 체크포인트는 그 이전 포맷입니다 |
| **함정 ①** | 🔴 **에러가 시키는 경로가 pip 설치본에 없습니다.** `src/lerobot/...`는 소스 트리 경로입니다. Step 6.3의 `$MIG` 방식으로 뽑으세요 |
| **함정 ②** | 🔴 **마이그레이션도 끝까지 안 갑니다.** `Exception: Couldn't encode 84`로 중단되어 `model.safetensors`가 안 나옵니다. `crop_shape`의 84를 직렬화하지 못합니다 |
| **대처** | **직접 학습한 체크포인트를 쓰세요.** 처음부터 processor 포맷이라 문제가 없습니다(확인함). Step 3에서 Step 5로 가는 경로가 그것입니다 |
| **뜻** | "사전학습본으로 빠르게 성공을 맛본다"는 흔한 경로가 이 버전에서는 막혀 있습니다 |

### E3. `require_package("diffusers", extra="diffusion")` 에서 죽는다

| | |
|---|---|
| **언제 뜨나** | `--policy.type=diffusion`으로 정책을 만드는 순간. Step 3 |
| **원인** | `diffusion` extra가 안 들어왔습니다. `pusht`만 깔면 이 에러를 만납니다 |
| **판정** | `verify_dp_install.py`의 `[1] diffusers` 행이 FAIL이면 이것입니다 |
| **대처** | `uv pip install --python .venv-lerobot/bin/python "lerobot[pusht,diffusion,evaluation]"` |

### E4. 붉은 traceback 수십 줄, `torchcodec` 로드 실패

```
Could not load libtorchcodec. Likely causes: 1. FFmpeg is not properly installed ...
'torchcodec' is installed but cannot be loaded. Falling back to 'pyav' as a default decoder.
```

| | |
|---|---|
| **언제 뜨나** | 프레임을 디코딩하는 쪽. `lerobot-train`(Step 3, 4)과 `lerobot-eval`(Step 5). **메타데이터만 읽는 Step 2에서는 안 뜹니다** |
| **원인** | 시스템에 `ffmpeg`이 없습니다 |
| **영향** | **없습니다.** 이 문서의 Step 3, 4, 5 출력이 **전부 이 폴백 상태에서 나온 것**입니다. mp4 저장까지 정상이었습니다 |
| **판정** | 🔴 **마지막 줄만 보세요.** `Falling back to 'pyav' as a default decoder.`가 있으면 정상입니다 |
| **대처** | 그대로 진행하세요 |

> ⚠️ **미검증(sudo 필요)** 근본 처방인 `sudo apt install -y ffmpeg`은 집필 시점에 실행 검증되지 않았습니다. 권한 때문입니다. **처방을 안 써도 이 랩은 전부 돕니다.**

> 📌 **traceback의 길이가 아니라 마지막 줄을 읽는 습관**이 이 분야의 기본기입니다. W1-M2 랩의 E1(종료 시 EGL traceback), W2-M1 랩의 E1이 전부 같은 종류였습니다.

### E5. `Warning: You are sending unauthenticated requests to the HF Hub.`

| | |
|---|---|
| **원인** | HF 토큰이 없습니다 |
| **영향** | **없습니다.** `lerobot/pusht`는 7.5 MB라 속도 제한이 문제가 되지 않습니다 |
| **대처** | 무시하세요. 자주 받을 것 같으면 `export HF_TOKEN=...` |

### E6. ★ 스모크만 돌렸는데 디스크가 3 GB씩 준다

```bash
du -sh outputs/train/w2m2_dp_smoke
# 3.0G     ← 300스텝만 돌렸는데
```

| | |
|---|---|
| **원인** | **체크포인트 하나가 3.0 GB**입니다. `model.safetensors` **1,003 MB** + `optimizer_state.safetensors` **2.0 GB**. Adam이 파라미터마다 모멘트 둘을 들고 있어 옵티마이저 상태가 모델의 두 배입니다 |
| **비교** | W2-M1의 ACT는 같은 구조로 **592 MB**였습니다(198 + 394). **5.1배**이고 파라미터 비율과 같습니다 |
| **기본값** | `--save_freq=20000` · `--steps=100000` → 체크포인트 **5개, 약 15 GB** |
| **대처** | 스모크에서는 **`--save_freq`를 총 스텝 수 이상**으로. 설정만 볼 때는 **`--save_checkpoint=false`** 로 아예 안 만듭니다(Step 4) |
| **회수** | `rm -rf outputs/train/w2m2_dp_smoke` 로 3.0 GB가 돌아옵니다 |

> 📌 `checkpoints/last`는 **심볼릭 링크라 용량 0**입니다. 두 번 세지 마세요.
> 📌 lesson §5.3의 "1,003 MB"는 **`model.safetensors` 한 파일**입니다. 학습이 실제로 쓰는 디스크는 그 세 배입니다.

### E7. ★ `ModuleNotFoundError: No module named 'lerobot'`

| | |
|---|---|
| **원인** | W1용 `.venv`나 시스템 `python3`으로 돌렸습니다. LeRobot은 **`.venv-lerobot`에만** 있습니다 |
| **판정** | `verify_dp_install.py`가 `[1]` 표에 인터프리터 경로를 찍고, `.venv-lerobot`이 아니면 **WARN**을 냅니다 |
| **대처** | 모든 명령에 **`.venv-lerobot/bin/python`** 또는 **`.venv-lerobot/bin/lerobot-*`** 경로를 직접 쓰세요 |

> 🔴 **반대 방향이 더 위험합니다.** "귀찮으니 `.venv`에 lerobot을 설치하자"고 하면 **torch 버전이 내려가면서 W1 practice가 깨집니다.** 두 환경을 분리한 것이 그 이유입니다.

### 에러는 아니지만 자주 놀라는 것 셋

| 증상 | 판정 |
|---|---|
| **E8. `lerobot-train`이 `--policy.repo_id`를 요구한다** | 학습 결과를 허브에 올리는 것이 기본 동작입니다. **`--policy.push_to_hub=false`** 를 주세요. 이 랩의 모든 학습 명령이 그렇게 돼 있습니다 |
| **E9. `--policy.crop_shape=[84,84]`가 이상하게 파싱된다** | **따옴표를 빠뜨렸습니다.** 셸이 대괄호를 파일명 확장으로 먹습니다. `--policy.crop_shape="[84,84]"` 로 주세요 |
| **`WARNING: Device 'None' is not available. Switching to 'cuda'.` 가 여러 번 뜬다** | **정상입니다.** `verify`가 device 지정 없이 `DiffusionConfig()`를 만들 때 lerobot이 내는 안내입니다. 대조 결과에 영향 없음 |

---

## 심화, 여유가 있으면 (각 15분)

**변형 A. `n_action_steps`를 바꿔 재계획 주기를 눈으로 보기**

Step 5의 평가를 `n_action_steps`만 바꿔 다시 돌립니다. 체크포인트의 `config.json`을 고치는 대신 평가 시 오버라이드가 가능한지부터 확인하세요.

```bash
# 학습을 다시 하지 않고 T_a만 바꿔 평가할 수 있는가?
# 🔴 그럴 수 없다면 그 사실 자체가 답입니다. worksheet ⑧에 왜 그런지 적으세요.
.venv-lerobot/bin/lerobot-eval --help 2>&1 | grep -E "policy\.(n_action_steps|horizon)"
```

> ⚠️ **미검증(GPU 필요).** 이 변형은 집필 시점에 실행되지 않았습니다. 오버라이드가 되는지부터가 확인 대상입니다.

**변형 B. NFE를 줄여 같은 체크포인트를 다시 평가**

lesson §6.3의 두 숫자를 자기 체크포인트에서 확인합니다. `num_inference_steps`를 100에서 16으로, 샘플러를 DDIM으로 바꾸면 **같은 가중치로 6배 빨라집니다.** 성공률이 얼마나 떨어지는지가 관심사입니다.

> ⚠️ **미검증(GPU 필요).** 집필 시점에 실행되지 않았습니다. 300스텝 체크포인트는 원래 성공률이 0이라 **의미 있는 비교가 되려면 본 학습 체크포인트가 필요합니다.** 0.3의 4.4시간 배지를 먼저 보세요.

**두 변형 모두 답보다 질문이 값집니다.** 되는지 안 되는지를 확인하고, 안 되면 왜 안 되는지를 `worksheet.md` ⑧에 적으세요.

---

## 다음

- **제출물**: [`worksheet.md`](worksheet.md) 전부 기입. 특히 **⑥ §4.2 재계산**과 **⑦ 완료 기준 문단**
- **실행 기록**: [`../../../../docs/progress.md`](../../../../docs/progress.md). 자기 기기의 지연값과 디스크 사용량
- **용어와 질문**: [`../../../../notes/glossary.md`](../../../../notes/glossary.md)와 [`../../../../notes/questions-for-team.md`](../../../../notes/questions-for-team.md)
- **이전 토픽**: [W2-M1 모방학습 기초와 ACT](../../01-imitation-learning-act/lesson.md). 거기서 만든 파이프라인을 여기서 끝까지 통과시켰습니다
- **다음 토픽**: **W2-M3 VLA 계보** *(집필 예정)*. 여기까지는 태스크마다 정책을 따로 학습했습니다. 다음은 언어를 조건으로 받는 거대 백본이 그 자리를 대신하는 계보이고, Step 7이 낸 "반복 횟수가 지연을 지배한다"가 그쪽에서는 **토큰 개수**로 모양을 바꿉니다
