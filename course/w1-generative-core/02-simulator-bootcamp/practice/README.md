# W1-M2 practice — 시뮬레이터 부트캠프

[`../lesson.md`](../lesson.md)의 §3(MuJoCo 계산 모델) · §4.1(MJCF→mp4 파이프라인) · §5(G1 모델 읽기) ·
§6(headless 렌더링) · §7(mujoco_playground)을 **직접 돌려보는** 실습입니다.

§4.1의 ASCII 파이프라인이 이 네 스크립트의 설계도입니다.

```
MJCF(XML) → mjModel → mjData → mj_step 루프 → Renderer → mp4/png
   01          01·02     01·02      01·03        01·03      01·03
                                                  playground 스모크 → 04
```

> **뷰어는 쓰지 않습니다.** 클라우드 GPU 인스턴스에는 디스플레이가 없습니다.
> 모든 스크립트가 `import mujoco` **이전에** `MUJOCO_GL=egl`를 설정하고, 결과는 전부 파일로 저장합니다.
> `mujoco.viewer`는 어느 스크립트에도 없습니다. matplotlib도 `Agg` 백엔드로 강제돼 있어 `plt.show()`가 없습니다.

---

## 1. 사전 준비

### 1.1 패키지

```bash
cd course/w1-generative-core/02-simulator-bootcamp/practice

python3 -m venv .venv          # 또는  uv venv .venv   (Python 3.12 이상 필요)
source .venv/bin/activate
pip install -r requirements.txt
```

버전은 집필 환경(Python 3.12.12, 2026-08-01)에서 실제로 설치·실행 검증한 값으로 고정돼 있습니다.

> ⚠️ **이름 함정**: PyPI 배포명은 `playground`, import 이름은 `mujoco_playground`입니다.
> `pip install mujoco_playground`는 **존재하지 않는 패키지**입니다 (lesson §7.1).

### 1.2 G1 로봇 모델 — pip이 아니라 git clone

`02`·`03`은 mujoco_menagerie의 G1 MJCF가 필요합니다. **약 2.3 GB**입니다.

```bash
# 리포 루트에서
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git repos/mujoco_menagerie
```

`repos/`는 gitignore 대상이라 커밋되지 않습니다.

**모델 경로 규약** — 스크립트는 이 순서로 찾습니다.

| 우선순위 | 방법 |
|---|---|
| 1 | `--menagerie /path/to/mujoco_menagerie` 인자 |
| 2 | 환경변수 `MENAGERIE_PATH` |
| 3 | (기본) 리포 루트 기준 `repos/mujoco_menagerie` |

```bash
export MENAGERIE_PATH=/mnt/persistent/mujoco_menagerie   # 퍼시스턴트 볼륨에 두면 인스턴스를 갈아도 유지됩니다
```

`unitree_g1/` 디렉토리가 없으면 스크립트가 클론 명령을 안내하고 종료합니다.

### 1.3 렌더 백엔드

```bash
export MUJOCO_GL=egl    # 클라우드 GPU 인스턴스의 기본값
```

스크립트 안에서도 `os.environ.setdefault("MUJOCO_GL", "egl")`가 걸려 있으므로 명령줄에서 안 줘도 됩니다.
단, **노트북에서는 첫 셀이 실행되기 전에 `mujoco`가 import된 커널이면 백엔드가 이미 결정돼 있습니다** — 커널을 재시작하세요.

---

## 2. 실행 순서

| # | 스크립트 | 이 스크립트가 답하는 질문 | GPU | 소요 (CPU) |
|---|---|---|---|---|
| 1 | `01_mujoco_basics.py` | MJCF 문자열 하나로 물리·렌더·mp4 경로가 전부 도는가? `timestep`을 바꾸면 정확도와 속도가 얼마나 거래되는가? | 불필요 | **1분 이내** |
| 2 | `02_g1_inspect.py` | G1의 `nq/nv/nu`·관절 인덱스·`kp`/`kv`·센서·킨매틱 트리는 실제로 어떤 값인가? (자료를 믿지 말고 뽑아본다) | 불필요 | **수 초** |
| 3 | `03_g1_sin_wave.py` | `ctrl`에 sin파를 넣으면 로봇이 움직이는가? PD 서보가 명령을 얼마나 따라가는가? 다리를 흔들면 어떻게 되는가? | 불필요 | **20~30초** (smoke 5초) |
| 4 | `04_playground_smoke.py` | W3에서 쓸 `G1JoystickFlatTerrain`이 에러 없이 1스텝 도는가? 지금 CPU인가 GPU인가? | 스모크는 불필요 / **본 학습은 필수(W3-M1)** | **1~2분** (첫 실행은 더) |

```bash
# 스모크 먼저 — 경로·렌더·설치가 맞는지 수 초 안에 확인
MUJOCO_GL=egl python 01_mujoco_basics.py   --smoke
MUJOCO_GL=egl python 02_g1_inspect.py      --smoke
MUJOCO_GL=egl python 03_g1_sin_wave.py     --smoke
MUJOCO_GL=egl python 04_playground_smoke.py --no-step   # reset/step 없이 스펙만

# 전체
MUJOCO_GL=egl python 01_mujoco_basics.py
MUJOCO_GL=egl python 02_g1_inspect.py --csv
MUJOCO_GL=egl python 03_g1_sin_wave.py --joints arms
MUJOCO_GL=egl python 03_g1_sin_wave.py --joints legs    # ← 넘어집니다. 그게 핵심입니다
MUJOCO_GL=egl python 04_playground_smoke.py --steps 3
```

모든 산출물은 리포 루트의 **`artifacts/W1-M2/`** 에 저장됩니다(스크립트가 위치에서 자동으로 경로를 찾습니다).

| 파일 | 만든 스크립트 | 내용 |
|---|---|---|
| `01_timestep_study.png` | 01 | 궤적 · 에너지 오차 · dt-오차-비용 트레이드오프 3-panel |
| `01_pendulum_t0.png` / `01_pendulum.mp4` | 01 | 렌더 경로 확인용 |
| `joints_g1.csv` | 02 (`--csv`) | 관절 30행 × 14컬럼 (인덱스·게인·체인·부모 body) |
| `03_tracking_{joints}.png` | 03 | 목표 vs 실제 · 추종 오차 · 골반 높이 · 관성-지연 4-panel |
| `g1_sin_{joints}.mp4` | 03 | 롤아웃 영상 (**gitignore 대상** — 커밋되지 않는 것이 정상) |

### 노트북으로 돌리려면

`.py`가 원본이고 `.ipynb`는 jupytext로 동기 생성한 사본입니다. 둘 중 아무거나 편집해도 됩니다.

```bash
jupytext --sync 01_mujoco_basics.py   # .py <-> .ipynb 양방향 동기화
jupyter lab                            # 커널 재시작 후 Run All로 완주됩니다
```

노트북에서는 `argparse`가 인자를 받지 않고 **기본값(전체 모드)** 으로 돕니다.
`--smoke`로 돌리고 싶으면 마지막 셀을 `main(["--smoke"])`로 고치세요.

---

## 3. 실측으로 확인된 함정 세 가지

시뮬 첫 경험이면 이 셋에서 시간을 태웁니다. 미리 적어둡니다.

### ① 종료할 때 뜨는 EGL traceback은 **무해합니다**

```
Exception ignored in: <function Renderer.__del__ at 0x...>
Traceback (most recent call last):
  ...
OpenGL.raw.EGL._errors.EGLError: <exception str() failed>
```

`Exception ignored in:` 으로 시작하면 **파이썬이 이미 그 예외를 무시했다**는 뜻입니다.
프레임은 정상적으로 저장돼 있습니다 — `[저장] ...` 줄이 찍혔는지만 확인하면 됩니다.
이 저장소의 스크립트는 전부 `renderer.close()`를 명시 호출해 이 경고를 줄여놨지만,
인터프리터 종료 시점에 따라 여전히 나올 수 있습니다. (lesson §6.2)

### ② `MUJOCO_GL`은 `import mujoco` **이전에** 설정돼야 합니다

렌더 백엔드가 import 시점에 결정됩니다. 나중에 `os.environ`을 바꿔도 소용없습니다.

```python
import os
os.environ["MUJOCO_GL"] = "egl"   # ← 반드시 먼저
import mujoco                      # ← 그 다음
```

노트북이면 **첫 셀**에서, 그리고 이미 mujoco를 import한 커널이라면 **커널 재시작 후**에 해야 합니다.

### ③ JAX 첫 호출은 몇십 초 멈춰 있습니다 (04)

`jax.jit(env.reset)`의 첫 호출은 추적 + 컴파일이라 CPU에서 **수십 초**가 걸립니다.
멈춘 게 아닙니다. LLM 서빙에서 첫 요청이 느린 것(웜업)과 같은 구조입니다.

`04`는 그 대기 전에 안내 문구를 먼저 출력합니다. 실행 중
`Module ... load on device 'cpu' took ... (cached)` 같은 줄이 쏟아지는 것도 MuJoCo Warp 로그이고 정상입니다.

> 같은 머신에서 두 번째로 돌리면 커널 캐시(`~/.cache/warp/`, `~/.cache/jax/`)가 따뜻해서
> 첫 호출도 빨라집니다. 그래서 "첫 호출 / 두 번째 호출" 비가 1에 가깝게 나올 수 있습니다 — 정상입니다.

### (보너스) `g1.xml`이 아니라 `scene.xml`

`g1.xml`에는 바닥이 없어 로봇이 무한히 낙하합니다. `02`·`03`은 `scene.xml`을 로드합니다 (lesson §5.1).

---

## 4. 결과를 어떻게 읽는가

### `01` — timestep 트레이드오프

- **(a) 패널은 궤적 자체가 아니라 "가장 작은 dt 대비 각도 편차"입니다.** 궤적을 그대로 겹쳐 그리면
  전부 포개져서 아무것도 안 보입니다(편차가 60° 진폭에 대해 수 도 수준). 편차로 그리면
  **오차가 진폭이 아니라 위상으로 누적된다**는 것이 보입니다 — 봉투가 시간에 따라 커지는 모양입니다.
  `dt=0.05`면 6초 만에 8°까지 벌어집니다.
- **(b) 패널의 에너지 오차는 전부 수치 오차입니다.** 감쇠가 0인 진자는 물리적으로 에너지가 보존되므로,
  관측되는 드리프트는 적분 오차뿐입니다. 참값을 모를 때 오차를 재는 고전적인 수법입니다.
- **(c) 패널의 log-log 기울기가 수렴 차수**입니다. 집필 환경 실측으로 `Euler` ≈ 0.9(1차),
  `RK4` ≈ 3.4. RK4는 오차가 훨씬 작지만 **스텝당 벽시계 비용이 약 2배**입니다 — 표의 "벽시계" 열로 확인하세요.
- 세로 점선 두 개가 **G1 classic(0.002)과 MJX(0.004)** 의 위치입니다. MJX가 어느 방향으로 이동했는지 보세요.
  `timestep`을 줄이는 것이 항상 정답이 아니라는 것이 lesson §9의 두 번째 오해입니다.

### `02` — 인덱스와 게인

- **세 주소 체계 정렬 검증이 통과**했다면 이 모델은 "자유 관절 1개 + hinge 나열" 구조이고
  `ctrl id + 7 = qpos adr` 산술이 성립합니다. `scene.xml`은 29개 전부 통과합니다.
  **그런데 `scene_with_hands.xml`은 손가락 4개에서 이 규칙이 깨집니다** —
  `<actuator>` 선언 순서와 관절 트리 순서가 다르기 때문입니다(§6의 명령으로 직접 확인).
  회사 MJCF도 다를 수 있습니다. 손으로 세지 말고 `mj_name2id` + `actuator_trnid`로 조회하세요.
- **`M_eff` 최대/최소 비가 89배**입니다(고관절 0.925 vs 손목 0.010 kg·m²). 그래서 `dampratio="1"`이
  관절마다 다른 `kv`를 만들어냅니다. 모든 관절에 같은 `kv`를 주면 어떤 관절은 과감쇠가 됩니다.
- **`forcerange = [0, 0]` = 토크 제한 없음.** 실기 무릎은 90 N·m(기본)/120 N·m(EDU)에서 포화합니다.
  sim2real 갭의 첫 번째 씨앗입니다 (lesson §5.6).
- classic vs MJX 게인 비교표에서 **kp 500 → 20~75, kv 관절별 → 2 고정**을 확인하세요.
  같은 로봇의 MJCF가 두 벌 존재한다는 것 자체가 sim2sim의 축소판입니다.

### `03` — 추종 오차 (이 실습의 핵심)

- **(a) 점선(목표각)과 실선(실제각)이 거의 겹칩니다.** 당연합니다 — 명령이 0.5 Hz인데 서보 대역폭
  $\omega_n = \sqrt{k_p/M_{eff}}$ 은 9~35 Hz입니다. 대역폭보다 한참 낮은 명령이라 잘 따라갑니다.
- **(b) 추종 오차는 mrad 단위**입니다. 유효관성이 큰 어깨(수십 mrad)와 작은 손목(수 mrad)의 차이를 보세요.
- **(d)가 가장 볼 만합니다.** 유효관성 vs 위상 지연이 eq.(4)의 2차계 예측선을 대체로 따라갑니다.
  정확히 맞지는 않습니다 — eq.(4)는 관절 하나를 떼어낸 1자유도 근사이고 실제 $M(q)$는 자세에 따라 변하며
  관절끼리 커플링돼 있으니까요. **경향이 맞는지**를 보는 것이 목적입니다.
- **명령 주파수를 올려보세요.** `--freq 5` 로 대역폭에 가까이 가면 진폭이 줄고 지연이 커집니다.
  익숙한 보드 선도 그대로입니다.
- **(c) `--joints legs`면 1초 안에 넘어집니다.** PD 서보는 자기 관절각만 지킬 뿐 "넘어지지 않는다"는
  목적을 모릅니다. 전신의 균형을 목적함수에 넣어 푸는 것이 WBC이고, 그걸 학습으로 얻는 것이 W3-M1입니다.
  **이 그림 한 장이 "왜 RL 보행 정책이 필요한가"의 답입니다.**

### `04` — W3 예고

- **`jax.devices()`가 `[CpuDevice(id=0)]`면 CPU입니다.** 이 스모크는 통과하지만 본 학습은 불가능합니다.
- `n_substeps = ctrl_dt/sim_dt = 0.02/0.002 = 10`. 정책 50 Hz / 물리 500 Hz.
  **정책이 명령을 갱신하지 않는 10 스텝 동안에도 물리는 계속 돕니다** — W1-M1의 action chunking과 같은 구조입니다.
- **보상항 24개**를 LQR 비용함수 $J = \sum (x-x_{ref})'Q(x-x_{ref}) + u'Ru$ 와 나란히 놓고 보세요.
  가중치가 0인 항이 여럿이라는 것도 눈여겨볼 만합니다 — 튜닝의 흔적입니다.
- `feet_air_time` · `feet_phase`처럼 **접촉 이벤트에 걸린 항은 미분 불가능**합니다.
  그게 MPC 대신 RL을 쓰는 진짜 이유입니다 (lesson §7.5).

---

## 5. 학습자가 직접 채우는 것

### `g1_kinematic_tree_worksheet.excalidraw` — 이 모듈의 진짜 산출물

**빈칸 워크시트**입니다. pelvis(root)에서 뻗어나가는 다섯 체인(왼다리 6 · 오른다리 6 · 허리 3→몸통 ·
왼팔 7 · 오른팔 7)의 노드가 전부 비어 있습니다.

1. `python 02_g1_inspect.py --csv` 를 돌려 `artifacts/W1-M2/joints_g1.csv` 를 만듭니다
2. [excalidraw.com](https://excalidraw.com)에서 `File > Open`으로 이 파일을 엽니다
   (VS Code면 `Excalidraw` 확장으로 바로 열립니다)
3. 각 노드에 **관절 이름 · `jnt` id · `qpos` adr · `ctrl` id** 를 손으로 적습니다
4. 상단의 "확인할 규칙"·"차원 셈" 상자와 하단의 확인 문항까지 채웁니다
5. 다 채운 뒤 lesson §5.4 표와 대조합니다

**정답을 미리 채워두지 않았습니다.** 손으로 한 번 적어보는 것이 목적입니다 —
인덱스를 잘못 짚어 팔 대신 다리가 움직이는 버그는 W3에서 반드시 한 번 만나게 되고,
그때 이 워크시트를 채워본 30분이 몇 시간을 아껴줍니다.

모르는 칸은 물음표로 남기고, 그 물음표를 [`../../../../notes/questions-for-team.md`](../../../../notes/questions-for-team.md)로 옮기세요.
특히 **회사 G1이 23 / 29 / 43 DoF 중 어느 구성인지**(lesson §5.3)는 이 워크시트 전체를 다시 그려야 하는 질문입니다.

### `joints_g1.csv` — 모델이 바뀌면 다시 뽑는 파일

menagerie는 갱신되고 회사 MJCF는 또 다를 수 있습니다.
`--menagerie` / `--variant`를 바꿔 다시 돌리면 CSV가 갱신됩니다. 코드를 고칠 일은 없습니다.

컬럼: `jnt_id, name, type, qpos_adr, dof_adr, ctrl_id, range_lo, range_hi, kp, kv, m_eff, chain, body, parent_body`

---

## 6. 심화 — 인자를 바꿔보기

```bash
# 01: G1 MJX가 쓰는 적분기로 바꿔서 비교
python 01_mujoco_basics.py --integrator implicitfast

# 02: 손 포함 43 DoF 모델을 해부 (lesson §12 퀴즈 2번의 답을 직접 확인)
#     ← 여기서 "ctrl id + 7 = qpos adr" 규칙이 손가락 4개에서 깨지는 것을 보게 됩니다.
#        MJCF의 <actuator> 선언 순서와 관절 트리 순서가 달라서입니다. 산술 규칙을 믿으면 안 되는 실물 증거.
python 02_g1_inspect.py --variant scene_with_hands.xml --csv

# 02: MJX 모델의 키프레임 이름이 다른 것 확인 ('stand'가 아니라 'home'/'knees_bent')
python 02_g1_inspect.py --variant scene_mjx.xml

# 03: 명령 주파수를 서보 대역폭 쪽으로 — 진폭 감쇠와 위상 지연이 커진다
python 03_g1_sin_wave.py --freq 5 --seconds 3

# 03: 진폭을 관절 한계 밖으로 — 클립이 몇 번 걸리는지
python 03_g1_sin_wave.py --amp 2.0

# 03: 29개 전부 흔들기 (그래프만, 빠름)
python 03_g1_sin_wave.py --joints all --no-video

# 04: 거친 지형 태스크의 스펙은 얼마나 다른가
python 04_playground_smoke.py --env G1JoystickRoughTerrain --no-step
```

---

## 7. 다음

- 랩 가이드: [`../labs/`](../labs/) — 클라우드 인스턴스 생성부터 첫 mp4까지, **단계별 명령 + 성공 판정 기준**.
  설치가 막히거나 위 함정 셋으로 해결이 안 되면 여기가 본체입니다.
- 실행 기록: 돌린 것 · 막힌 지점 · 소요 시간 · GPU 비용을 [`../../../../docs/progress.md`](../../../../docs/progress.md)에 남기세요.
  여기 쌓인 실측값이 W3 자료의 입력이 됩니다.
- 다음 토픽: Diffusion 계보 DDPM → DiT *(예정: `../../03-diffusion-ddpm-dit/`)*
