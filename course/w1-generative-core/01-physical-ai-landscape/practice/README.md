# W1-M1 practice — Physical AI 개요와 산업 지형

[`../lesson.md`](../lesson.md)의 §2.3(5계층 스택) · §2.4(대역폭 분리) · §2.5(데이터 피라미드) · §3(지연 예산) · §4(플레이어 계보)를
**직접 돌려보고 그림으로 뽑는** 실습입니다.

> **GPU·시뮬레이터 불필요.** MuJoCo도 쓰지 않으므로 `MUJOCO_GL=egl` 전제는 이 모듈에 해당하지 않습니다.
> 대신 **matplotlib은 `Agg` 백엔드로 강제**되어 있습니다(`matplotlib.use("Agg")`가 import 최상단).
> 창을 띄우지 않고 결과를 전부 파일로 저장하므로 헤드리스 클라우드 인스턴스에서 그대로 돌아갑니다.
> `plt.show()`는 어느 스크립트에도 없습니다.

전체 소요: **CPU에서 1분 미만.** 세 스크립트를 다 돌려도 수 초입니다.

---

## 1. 설치

```bash
cd course/w1-generative-core/01-physical-ai-landscape/practice

python3 -m venv .venv          # 또는  uv venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

버전은 집필 환경(Python 3.12.3)에서 실제로 설치·실행 검증한 값으로 고정돼 있습니다.

---

## 2. 실행 순서

| # | 스크립트 | 무엇을 하는가 | 출력 |
|---|---|---|---|
| 1 | `01_stack_frequency_budget.py` | **이 모듈의 메인 실습.** 5계층 주파수 예산 표 · 대역폭 분리비 · lesson §3 부등식(최소 청크 길이) · 명령 공백 시뮬레이션 · 지연 안전 vs 반응성 트레이드오프 | `01_frequency_budget.png` |
| 2 | `02_data_pyramid_plot.py` | 데이터 피라미드 4층과 5계층 스택을 문서용 그림으로 렌더 (스택은 **로그 주파수 축**) | `02_data_pyramid.png`, `02_stack_layers.png` |
| 3 | `03_player_landscape.py` | `players.csv`(lesson §4 표)를 읽어 조직×베팅 계층 매트릭스 + 발표 타임라인 렌더 | `03_player_landscape.png` |

```bash
# 스모크 (수 초, 가장 먼저 이걸로 완주 확인)
python 01_stack_frequency_budget.py --smoke
python 02_data_pyramid_plot.py      --smoke
python 03_player_landscape.py       --smoke

# 전체
python 01_stack_frequency_budget.py
python 02_data_pyramid_plot.py
python 03_player_landscape.py
```

모든 그림은 리포 루트의 **`artifacts/W1-M1/`** 에 저장됩니다(스크립트 위치에서 자동으로 경로를 찾습니다).

### 노트북으로 돌리려면

`.py`가 원본이고 `.ipynb`는 jupytext로 동기 생성한 사본입니다. 둘 중 아무거나 편집해도 됩니다.

```bash
jupytext --sync 01_stack_frequency_budget.py    # .py <-> .ipynb 양방향 동기화
jupyter lab                                     # 커널 재시작 후 Run All로 완주됩니다
```

노트북에서는 `argparse`가 인자를 받지 않고 기본값(전체 모드)으로 돕니다.

---

## 3. 자주 쓰는 인자

`01_stack_frequency_budget.py`는 lesson §3의 수치를 직접 바꿔볼 수 있습니다.

```bash
# 청크를 8스텝으로 줄이면 명령 공백이 얼마나 생기는가
python 01_stack_frequency_budget.py --h-chunk 8

# 온보드 추론이 느려서 300ms 걸린다면?
python 01_stack_frequency_budget.py --tau-infer 0.300

# WBC를 200Hz로 올리면 필요한 청크 길이는?
python 01_stack_frequency_budget.py --f2 200
```

세 스크립트 모두 `--ascii-labels`를 받습니다. 한글 폰트가 있어도 **영문 라벨로 강제 렌더**해 비교할 수 있습니다
(`--ascii-labels`로 뽑은 그림은 `*_ascii.png`로 따로 저장됩니다).

### 한글 폰트가 없을 때

실행 환경에 한글 폰트가 없으면 matplotlib 라벨이 두부(□)로 깨집니다.
각 스크립트의 `setup_korean_font()`가 **폰트 이름이 아니라 실제 한글 글리프 보유 여부**로 폰트를 고르고,
찾지 못하면 경고 한 줄을 찍고 **영문 라벨로 자동 폴백**합니다. 그림 파일명은 그대로 유지됩니다.

한글로 뽑고 싶으면:

```bash
sudo apt install -y fonts-nanum
rm -rf ~/.cache/matplotlib      # 폰트 캐시 삭제 후 재실행
```

---

## 4. 학습자가 직접 고치는 것

### `players.csv` — 이 파일은 갱신되라고 있는 파일입니다

lesson §4 표를 CSV로 옮긴 것입니다. 이 분야는 몇 주 단위로 새 모델이 나오므로,
**새 발표가 나오면 CSV에 한 줄만 추가하고 `03_player_landscape.py`를 다시 돌리면** 그림이 갱신됩니다.
코드를 고칠 일은 없습니다.

컬럼 규약은 `03_player_landscape.py` 상단 마크다운 셀에 있습니다. 두 가지만 지키세요.

- `date`는 **아는 만큼만** 씁니다(`2026` / `2026-04` / `2026-04-16`). 모르면 빈칸.
  연도만 아는 항목은 타임라인에 그 해 전체를 덮는 막대로 그려집니다 — 없는 정밀도를 지어내지 않기 위한 장치입니다.
- **1차 출처로 확인되지 않은 수치는 넣지 마세요.** lesson §4가 의도적으로 배제한 수치
  (Figure 생산 대수 · Optimus DoF · 1X 선주문 수량)는 CSV에도 없습니다. `date_confidence`를 `unknown`으로 두면 됩니다.

### `my_stack_template.excalidraw` — 이 모듈의 진짜 산출물

같은 폴더의 `my_stack_template.excalidraw`는 **"본인 언어로 그린 스택 다이어그램"** 을 채워 넣는 편집 가능한 스켈레톤입니다.
[excalidraw.com](https://excalidraw.com)에서 `File > Open`으로 이 파일을 열어 직접 채우세요
(VS Code를 쓴다면 `Excalidraw` 확장으로 바로 열립니다).

lesson의 그림을 그대로 베끼지 말고, **모르는 칸은 물음표로 남겨두세요.** 그 물음표가 `notes/questions-for-team.md`에 들어갈 팀 질문 리스트가 됩니다.

---

## 5. 결과를 어떻게 읽는가

- **`01`의 대역폭 분리비 표** — L1/L2와 L2/L4는 캐스케이드 제어의 경험칙(inner를 outer의 5~10배)을 넉넉히 만족하지만
  **L4/L5는 대표 비율이 약 2배**에 그칩니다. 이 두 계층은 대역폭 분리라기보다 기능 분리(인지 vs 행동 의도)라는 뜻입니다.
- **`01`의 명령 공백 타임라인** — `H_required`보다 1스텝만 부족해도 공백이 생깁니다. `#`이 명령, `.`이 공백입니다.
- **`01`의 트레이드오프 표** — 청크를 늘리면 공백은 0으로 가지만 **최악 반응 지연이 선형으로 커집니다**($H_{chunk}/f_2$).
  MPC에서 예측 지평 $N$을 늘릴 때와 같은 구조의 트레이드오프입니다.
- **`02_stack_layers.png`** — 로그 축에서 막대 사이가 얼마나 벌어져 있는지가 곧 대역폭 분리입니다. **가장 넓은 경계는 L2↔L4(×50)** 이고 L1↔L2(×8.9)가 그 다음입니다. 가장 넓은 그 틈에 L3(액션 인터페이스)가 놓여 있습니다.
- **`03`의 매트릭스** — **L3 열이 비어 있다는 것**이 이 그림의 핵심입니다. 공개 논문 수준에서 액션 인터페이스를
  독립 계층으로 내세운 조직이 드물고, lesson §2.3의 "승부처는 L3"와 맞물립니다.

---

## 6. 다음

- 랩 가이드: [`../labs/`](../labs/)
- 다음 토픽: [시뮬레이터 부트캠프](../../02-simulator-bootcamp/lesson.md)
