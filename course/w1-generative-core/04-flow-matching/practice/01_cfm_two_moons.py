# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # W1-M4 실습 1 — CFM을 직접 구현하고, "개별 target은 전부 틀렸지만 평균이 정답"을 눈으로 본다
#
# lesson.md `§4`(CFM 정리) · `§5.2`(직선 경로) · `§6.1~6.2`(왜 스텝이 줄고, 왜 1스텝은 안 되는가)의
# 실행판입니다. **W1-M3의 `01_ddpm_toy.py`와 데이터·모델·학습 설정을 전부 같게 맞췄습니다** —
# 바뀐 것은 objective 하나뿐이고, 그래야 `02`에서 두 곡선을 같은 축에 겹쳐 그릴 수 있습니다.
#
# 확인할 것:
#
# 1. **§5.2의 손실이 정말 한 줄인가** — `t~U(0,1)` / `x_t=(1-t)x₀+t·x₁` / `MSE(v, x₁−x₀)`.
#    W1-M3의 `Schedule` 데이터클래스가 사전계산해 들고 다니던 **7,000개짜리 버퍼가 0개**가 되는 것을
#    숫자로 찍습니다.
# 2. **§4.3 "개별 target은 전부 틀렸지만 그 평균이 정답"의 실증** ★ — 격자점 $x^\star$ 하나를 골라
#    그 점을 지나는 여러 $(x_0,x_1)$ 쌍의 $x_1-x_0$을 모아 뿌리고, 학습된 $v_\theta(x^\star,t)$가
#    그 **평균**에 얼마나 가까운지 잽니다. **이 모듈에서 가장 교육적인 그림입니다.**
#    덤으로 주변 속도장의 **닫힌 형태**(경험분포에 대한 사후평균)를 직접 계산해 학습된 장과 대조합니다.
# 3. **§6.2 보간선 교차** — 실제 배치에서 같은 $(x,t)$ 근방을 지나면서 방향이 반대인 보간선 쌍을
#    찾아 그립니다. 조건부 경로가 직선인데도 주변 궤적이 휘는 이유가 이 그림 한 장입니다.
# 4. **NFE를 바꿔가며 Euler 적분** — 품질 곡선. $\mathcal{E}$(에너지 거리)와 sliced $W_1$은
#    **W1-M3 `02_samplers_compare.py`의 구현을 그대로 import해서** 씁니다(같은 자로 재야 하므로).
#
# 출력(`artifacts/W1-M4/`):
#
# | 파일 | 내용 |
# |---|---|
# | `01_targets.png` | ★ 개별 target 산점도 + 평균 + $v_\theta$ / 닫힌 형태 vs 학습된 벡터장 |
# | `01_crossing.png` | 보간선 교차(§6.2) + 학습된 ODE 궤적이 휘는 모습 |
# | `01_samples_nfe.png` | NFE별 Euler 샘플 격자 |
# | `01_loss_quality.png` | 학습 곡선 + $t$ 구간별 손실 + 품질 vs NFE |
# | `01_nfe.csv` | NFE별 energy·sW1·초 |
# | `01_model.pt` | 체크포인트 — **`02_rectified_flow_reflow.py`가 읽습니다** (gitignore) |
#
# **GPU 불필요.** 2차원 데이터에 작은 MLP라 CPU에서도 완주합니다(실측은 `README.md` §2.1).
# `--smoke`는 수십 초.

# %%
from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import inspect
import sys
import time
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless 고정 — 뷰어를 띄우지 않는다

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

MODULE_ID = "W1-M4"

# ── W1-M3와 "완전히 같게" 맞춘 설정 ─────────────────────────────────────────────
# 이 값들이 하나라도 다르면 02의 겹쳐 그리기가 성립하지 않습니다. 근거는 각각
# 03-diffusion-ddpm-dit/practice/01_ddpm_toy.py 의 parse_args() 기본값입니다.
DATA_KIND = "moons"     # make_moons(noise=0.06)
N_DATA = 16384          # 학습 데이터 수
N_SAMPLE = 4096         # 평가 시 뽑는 샘플 수 (= W1-M3 02의 --n-sample)
N_PROJ = 256            # sliced W1의 투영 방향 수 (= W1-M3 02의 --n-proj)
HIDDEN, LAYERS, TEMB = 256, 4, 128
STEPS, BATCH, LR, EMA_DECAY = 12000, 512, 2e-3, 0.999

# W1-M3 「실습으로 가기」와 그 `02`가 쓴 것과 같은 NFE 집합. 여기서 1000은 "격자 전체"가 아니라 그냥 큰 값입니다
# (FM에는 학습 격자라는 것이 없습니다 — §5.3 '스케줄 의존성' 행).
NFE_GRID = [1, 2, 4, 10, 20, 50, 250, 1000]
NFE_GRID_SMOKE = [1, 4, 20, 250]

# 연속 t ∈ [0,1]을 sinusoidal 임베딩의 주파수 대역에 올려놓는 스케일. 아래 VelocityMLP 참고.
T_SCALE = 1000.0


# %% [markdown]
# ## 0. W1-M3의 코드를 그대로 가져온다
#
# 이 모듈이 새로 짜는 것은 **objective와 샘플러뿐**입니다. 데이터 생성·정규화·백본 MLP·EMA·
# 품질 지표·폰트 유틸은 W1-M3 practice에 이미 있고, 그것을 **그대로** 씁니다.
# 파일명이 숫자로 시작해 `import`가 안 되므로 `importlib`로 경로 로드합니다(W1-M3와 같은 규약).
#
# > W1-M3의 `02_samplers_compare.py`를 로드하면 그 안에서 `01_ddpm_toy.py`도 함께 올라옵니다.
# > 그래서 한 번의 로드로 `M1`(스케줄·MLP·EMA·데이터)과 `M2`(에너지 거리·sliced $W_1$)를 둘 다 얻습니다.

# %%
_ROOT_MARKERS = ("course", "docs", "CLAUDE.md")
_W1M3 = ("course", "w1-generative-core", "03-diffusion-ddpm-dit", "practice")


def find_repo_root() -> Path:
    """리포 루트를 찾는다 (스크립트/노트북 양쪽에서 동작)."""
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd().resolve()
    for cand in (start, *start.parents):
        if all((cand / m).exists() for m in _ROOT_MARKERS):
            return cand
    return start


def here() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def w1m3_dir() -> Path:
    return find_repo_root().joinpath(*_W1M3)


def artifacts_dir() -> Path:
    out = find_repo_root() / "artifacts" / MODULE_ID
    out.mkdir(parents=True, exist_ok=True)
    return out


THIS_FILE = "01_cfm_two_moons.py"


def source_of(fn, fname: str = THIS_FILE) -> str:
    """함수의 소스를 문자열로 얻는다.

    `inspect.getsource`는 **노트북/`exec` 환경에서 실패**합니다(파일에 대응하는 코드 객체가
    없기 때문). 그때는 같은 폴더의 원본 `.py`를 열어 `ast`로 해당 함수만 잘라 옵니다.
    이 스크립트는 자기 소스를 화면에 찍는 셀이 있어서(§1) 이 폴백이 없으면 노트북에서 죽습니다.
    """
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        p = here() / fname
        if not p.is_file():
            return ""
        src = p.read_text()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == fn.__name__:
                return "\n".join(src.splitlines()[node.lineno - 1:node.end_lineno])
        return ""


def load_sibling(fname: str, modname: str, folder: Path | None = None):
    """숫자로 시작하는 파일을 모듈로 로드한다 (W1-M3 03_dit_action_head.py와 같은 규약)."""
    cands = [folder] if folder is not None else [here(), w1m3_dir()]
    for cand in cands:
        p = cand / fname
        if p.is_file():
            spec = importlib.util.spec_from_file_location(modname, p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[modname] = mod  # dataclass가 __module__을 되찾을 수 있게
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit(f"[에러] {fname} 를 찾지 못했습니다 (찾아본 곳: {[str(c) for c in cands]})")


M2 = load_sibling("02_samplers_compare.py", "m3_samplers", w1m3_dir())
M1 = M2.M1

# 재사용하는 것 — 새로 짜지 않습니다
EpsMLP, EMA = M1.EpsMLP, M1.EMA
make_data, normalize = M1.make_data, M1.normalize
pick_device, timestep_embedding = M1.pick_device, M1.timestep_embedding
make_schedule = M1.make_schedule          # §5.3 대조용으로만 씁니다 (FM은 안 씁니다)
energy_distance, sliced_w1 = M2.energy_distance, M2.sliced_w1

USE_KOREAN = False


def setup_korean_font(force_ascii: bool = False) -> bool:
    global USE_KOREAN
    USE_KOREAN = M1.setup_korean_font(force_ascii)
    return USE_KOREAN


def lab(ko: str, en: str) -> str:
    """그림 라벨 전용 번역 헬퍼 (W1-M3와 같은 역할)."""
    return ko if USE_KOREAN else en


def _dwidth(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, width: int, align: str = "left") -> str:
    gap = max(0, width - _dwidth(s))
    if align == "right":
        return " " * gap + s
    if align == "center":
        left = gap // 2
        return " " * left + s + " " * (gap - left)
    return s + " " * gap


def print_table(headers: list[str], rows: list[list[str]], aligns: list[str] | None = None) -> None:
    aligns = aligns or ["left"] * len(headers)
    widths = [max(_dwidth(h), *(_dwidth(r[i]) for r in rows)) if rows else _dwidth(h)
              for i, h in enumerate(headers)]
    print("  " + " | ".join(_pad(h, w, "center") for h, w in zip(headers, widths)))
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        print("  " + " | ".join(_pad(c, w, a) for c, w, a in zip(r, widths, aligns)))


# %% [markdown]
# ## 1. 손실 — lesson §5.2가 정말 한 줄인가
#
# $$\mathcal{L}_{\text{RF}}(\theta)=\mathbb{E}_{t\sim\mathcal{U}[0,1],\,x_0\sim\mathcal{N}(0,I),\,x_1\sim q}
#   \bigl\|v_\theta\bigl((1-t)x_0+tx_1,\ t\bigr)-(x_1-x_0)\bigr\|^2$$
#
# 아래 함수의 **본문 세 줄**이 그 식 전부입니다. 대조군은 W1-M3 `01_ddpm_toy.py`의
# `Schedule` 데이터클래스 + `make_schedule()` + `q_sample()`이고, 그쪽에는
# $\beta_t,\alpha_t,\bar\alpha_t,\sqrt{\bar\alpha_t},\dots$ 일곱 개의 길이-$T$ 버퍼가 있습니다.
# **여기에는 그것이 없습니다.** 계수가 $t$와 $1-t$ 그 자체이기 때문입니다(§5.3 '스케줄 의존성' 행).
#
# > **시간 규약 주의.** W1-M3는 데이터가 $t{=}0$, 노이즈가 $t{=}T$였습니다. FM은 **반대로**
# > 노이즈가 $t{=}0$, 데이터가 $t{=}1$입니다(§5.3 첫 행). 부호를 한 번 헷갈리면 모델이
# > 정확히 반대 방향으로 학습되고, 샘플이 노이즈로 수렴합니다.

# %%
def cfm_loss(model: nn.Module, x1: torch.Tensor,
             gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """lesson §5.2의 직선 경로 CFM 손실. x1 [B,D] = 데이터(t=1 쪽).

    반환: (샘플별 손실 [B], 뽑힌 t [B])
    """
    dev = x1.device
    t = torch.rand(x1.shape[0], generator=gen).to(dev)              # eq. §5.2  t ~ U(0,1)
    x0 = torch.randn(x1.shape, generator=gen).to(dev)               # eq. §5.2  x₀ ~ N(0,I)
    x_t = (1.0 - t[:, None]) * x0 + t[:, None] * x1                 # eq. §5.2  x_t = (1−t)x₀ + t·x₁
    target = x1 - x0                                                # eq. §5.2  u = x₁ − x₀  (t 무관!)
    per = ((model(x_t, t) - target) ** 2).mean(dim=1)
    return per, t


def schedule_footprint() -> tuple[int, int]:
    """W1-M3의 스케줄이 들고 다니던 사전계산 버퍼 원소 수 vs FM의 그것."""
    sch = make_schedule(1000)
    n = sum(int(getattr(sch, f).numel()) for f in
            ("beta", "alpha", "abar", "abar_prev", "sqrt_abar",
             "sqrt_one_minus_abar", "beta_tilde"))
    return n, 0


# %% [markdown]
# ## 2. 모델 — 백본은 W1-M3의 것을 그대로
#
# lesson §7.1이 "백본·옵티마이저·EMA·조건 경로는 한 줄도 안 바뀐다"고 못박은 그대로입니다.
# `EpsMLP`를 **수정 없이** 재사용하고, 감싸는 껍데기 하나만 새로 씁니다.
#
# 껍데기가 하는 일은 **시간 인자의 단위 변환 하나**입니다. `timestep_embedding`의 주파수가
# $t\in\{1,\dots,1000\}$을 전제로 설계돼 있어서 $t\in[0,1]$을 그대로 넣으면 모든 주파수 성분이
# 거의 상수가 되고 시간 조건이 사라집니다. $\times 1000$이 그 대역 맞춤입니다.
#
# > **이것이 lesson §7.1이 세지 않은 "네 번째 줄"입니다.** 학습 루프에서는 안 보이지만
# > (모델 안에 숨겼으므로) 없으면 학습이 안 됩니다. 실무에서 FM으로 갈아탈 때
# > 첫 번째로 밟는 지뢰라서 여기 적어둡니다.

# %%
class VelocityMLP(nn.Module):
    """v_θ(x, t), t ∈ [0,1] 연속. 본체는 W1-M3의 EpsMLP 그대로."""

    def __init__(self, data_dim: int = 2, hidden: int = HIDDEN, n_layers: int = LAYERS,
                 temb_dim: int = TEMB, t_scale: float = T_SCALE):
        super().__init__()
        self.net = EpsMLP(data_dim, hidden, n_layers, temb_dim)   # ← 재사용. 한 줄도 안 고쳤습니다
        self.t_scale = t_scale

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(x, t * self.t_scale)      # 시간 규약만 맞춘다 (위 markdown 참고)


# %% [markdown]
# ## 3. 학습 루프
#
# W1-M3 `train()`과 **줄 수까지 거의 같습니다.** 옵티마이저·LR 스케줄·EMA·로깅이 전부 동일하고,
# 안쪽에서 `q_sample` + `L_simple` 자리에 `cfm_loss`가 들어간 것뿐입니다.

# %%
def train_cfm(model: nn.Module, x1_all: torch.Tensor, *, steps: int, batch: int, lr: float,
              ema_decay: float, log_every: int, seed: int) -> tuple[EMA, list[float], np.ndarray]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.05, 1.0 - s / steps))
    ema = EMA(model, ema_decay)     # ← W1-M3의 EMA 그대로 (워밍업 포함)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = x1_all.shape[0]
    losses: list[float] = []
    bucket_sum, bucket_cnt = np.zeros(10), np.zeros(10)
    run = 0.0
    for s in range(1, steps + 1):
        idx = torch.randint(0, n, (batch,), generator=g).to(x1_all.device)
        per, t = cfm_loss(model, x1_all[idx], g)
        loss = per.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        ema.update(model)
        run += loss.item()
        if s % log_every == 0:
            losses.append(run / log_every)
            run = 0.0
        if s > steps * 0.8:      # 후반부만 집계 (t 구간별 난이도 진단용)
            b = (t.detach() * 10).long().clamp(0, 9).cpu().numpy()
            np.add.at(bucket_sum, b, per.detach().cpu().numpy())
            np.add.at(bucket_cnt, b, 1.0)
    return ema, losses, bucket_sum / np.maximum(bucket_cnt, 1.0)


# %% [markdown]
# ## 4. 샘플링 — 명시적 Euler, NFE = K
#
# lesson §7.1 아래쪽 블록도 그대로입니다. 스케줄 계수 세 개도, 노이즈 재주입도 없습니다.
#
# ```
#  x ← x₀ ~ N(0, I);   h ← 1/K
#  for k = 0 … K-1:  x ← x + h · v_θ(x, k·h)
# ```
#
# `trace=True`면 모든 중간 상태를 쌓아 돌려줍니다(§6.2 궤적 그림과 직선성 지표용).

# %%
@torch.no_grad()
def euler_sample(model: nn.Module, n: int, device: torch.device, *, nfe: int, seed: int = 0,
                 dim: int = 2, trace: bool = False, x0: torch.Tensor | None = None):
    """NFE = nfe 회의 forward로 x₀ → x₁ 을 적분한다."""
    if x0 is None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        x0 = torch.randn(n, dim, generator=g).to(device)
    x = x0.clone()
    h = 1.0 / nfe
    xs = [x.clone()] if trace else None
    for k in range(nfe):
        t = torch.full((x.shape[0],), k * h, device=device)
        x = x + h * model(x, t)                    # eq. §7.1 샘플링:  x ← x + h·v_θ(x, k·h)
        if trace:
            xs.append(x.clone())
    return (x, torch.stack(xs, 0)) if trace else x


@torch.no_grad()
def straightness(model: nn.Module, n: int, device: torch.device, *, k_fine: int = 100,
                 seed: int = 0, dim: int = 2) -> float:
    r"""직선성 지표  S(Z) = ∫₀¹ E‖(Z₁−Z₀) − Ż_t‖² dt      (eq. §6.2)

    Euler로 궤적을 풀면서 각 스텝의 속도 Ż_t = v_θ(Z_t, t)를 기록하고, 끝점 Z₁까지 나온 뒤
    현(Z₁−Z₀)과의 차이를 t에 대해 평균합니다. **S=0이면 완전한 직선이고 1스텝이 정확합니다.**
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    z0 = torch.randn(n, dim, generator=g).to(device)
    x, h, vs = z0.clone(), 1.0 / k_fine, []
    for k in range(k_fine):
        t = torch.full((n,), k * h, device=device)
        v = model(x, t)
        vs.append(v)
        x = x + h * v
    chord = (x - z0)[None]                 # (Z₁ − Z₀)  [1, n, dim]
    V = torch.stack(vs, 0)                 # Ż_t        [K, n, dim]
    return float(((chord - V) ** 2).sum(-1).mean())


# %% [markdown]
# ## 5. ★ §4.3 — 개별 target은 전부 틀렸지만 평균이 정답
#
# 이 절이 이 스크립트의 핵심입니다. lesson §4.3의 문장을 두 방향에서 확인합니다.
#
# **(A) 경험적으로.** 점 $x^\star$와 시각 $t^\star$를 고르고, $(x_0,x_1)$ 쌍을 대량으로 뽑아
# $x_t$가 $x^\star$ 근처에 오는 것만 골라냅니다. 그 쌍들의 target $x_1-x_0$을 전부 뿌리면
# **사방으로 흩어집니다.** 그런데 그 평균이 학습된 $v_\theta(x^\star,t^\star)$와 거의 같습니다.
#
# **(B) 닫힌 형태로.** 데이터를 경험분포 $q=\frac1N\sum_i\delta_{x_1^{(i)}}$로 두면 §3.2의 적분이
# 유한합이 되어 **그냥 계산됩니다.** 조건부 경로가 $p_t(x|x_1)=\mathcal{N}(tx_1,(1-t)^2I)$이므로
#
# $$w_i \;\propto\; \exp\Bigl(-\frac{\|x-t\,x_1^{(i)}\|^2}{2(1-t)^2}\Bigr),
#   \qquad
#   u_t(x)=\sum_i w_i\,\frac{x_1^{(i)}-x}{1-t}$$
#
# 뒤쪽 분수가 나오는 이유: $x_t=x$와 $x_1$이 정해지면 $x_0=(x-tx_1)/(1-t)$가 **결정**되므로
# 그 쌍의 target은 $x_1-x_0=(x_1-x)/(1-t)$입니다. 즉 (A)의 산점도와 (B)의 가중평균은 같은 것을
# 다른 순서로 계산한 것이고, 둘이 맞아떨어지면 §4의 정리가 이 데이터에서 성립한 것입니다.
#
# > $t\to1$에서 $1/(1-t)$가 폭발합니다. 유한 데이터에서 주변 속도장이 실제로 발산하는 것이라
# > 버그가 아닙니다(그래서 아래에서 $t^\star$를 1에서 떨어뜨려 고릅니다).

# %%
@torch.no_grad()
def exact_marginal_velocity(x: torch.Tensor, t: float, data: torch.Tensor) -> torch.Tensor:
    """경험분포에 대한 주변 속도장의 닫힌 형태.  x [M,D], data [N,D] → [M,D]   (eq. §3.2 박스)"""
    s = max(1.0 - t, 1e-6)
    # log w_i = −‖x − t·x₁⁽ⁱ⁾‖² / (2s²)   → softmax 로 정규화 (수치 안정성 위해 logsumexp 사용)
    logw = -torch.cdist(x, t * data) ** 2 / (2.0 * s * s)     # [M, N]
    w = torch.softmax(logw, dim=1)
    # u_t(x) = Σ_i w_i (x₁⁽ⁱ⁾ − x) / (1−t)
    return (w @ data - x) / s


@torch.no_grad()
def targets_through_point(x_star: torch.Tensor, t: float, data: torch.Tensor, *,
                          n_pairs: int, keep: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """x_t 가 x_star 근처를 지나는 (x₀,x₁) 쌍을 골라 그 target 들을 모은다.

    반환: (targets [keep, D], 그 쌍의 x₁ [keep, D])
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = data.device
    idx = torch.randint(0, data.shape[0], (n_pairs,), generator=g).to(dev)
    x1 = data[idx]
    x0 = torch.randn(x1.shape, generator=g).to(dev)
    x_t = (1.0 - t) * x0 + t * x1                       # eq. §5.2
    d = (x_t - x_star[None]).norm(dim=1)
    sel = d.argsort()[:keep]                            # x_star 에 가장 가까운 것부터 keep 개
    return (x1[sel] - x0[sel]), x1[sel]


# %% [markdown]
# ## 6. §6.2 — 보간선은 교차한다
#
# 서로 다른 $(x_0,x_1)$ 쌍의 직선이 같은 $(x,t)$ 근방을 지나면서 **방향이 반대**인 경우를 찾습니다.
# 그 지점에서 주변 속도장은 두 방향의 (사후) 평균이라 어느 직선도 아닌 제3의 방향이 되고,
# 그래서 실제 ODE 해가 휩니다. lesson §6.2의 ASCII 그림을 데이터에서 실제로 찾아내는 것입니다.

# %%
@torch.no_grad()
def find_crossing(data: torch.Tensor, *, n_pairs: int, seed: int,
                  ts: np.ndarray) -> dict:
    """가장 뚜렷한 교차(가까운 x_t · 반대 방향 target) 한 쌍을 찾는다."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = data.device
    idx = torch.randint(0, data.shape[0], (n_pairs,), generator=g).to(dev)
    x1 = data[idx]
    x0 = torch.randn(x1.shape, generator=g).to(dev)
    u = x1 - x0                                          # target [M,D]
    un = u / u.norm(dim=1, keepdim=True).clamp_min(1e-9)
    cos = un @ un.T                                      # 방향 유사도 [M,M]
    best = None
    for t in ts:
        x_t = (1.0 - float(t)) * x0 + float(t) * x1
        D = torch.cdist(x_t, x_t)
        D.fill_diagonal_(float("inf"))
        D = D.masked_fill(cos > -0.5, float("inf"))      # 방향이 충분히 어긋난 쌍만 (cos < −0.5)
        v, j = D.min(dim=1)
        i = int(v.argmin())
        if not torch.isfinite(v[i]):
            continue
        cand = dict(t=float(t), dist=float(v[i]), i=i, j=int(j[i]))
        if best is None or cand["dist"] < best["dist"]:
            best = cand
    if best is None:
        return {}
    i, j, t = best["i"], best["j"], best["t"]
    best.update(x0i=x0[i], x1i=x1[i], x0j=x0[j], x1j=x1[j],
                xt=(1.0 - t) * x0[i] + t * x1[i],
                ui=u[i], uj=u[j], umean=0.5 * (u[i] + u[j]),
                cos=float(cos[i, j]))
    return best


# %% [markdown]
# ## 7. W1-M3와 같은 분포를 쓰고 있는지 확인
#
# `02`가 W1-M3의 곡선과 **같은 축에 겹쳐 그리는** 것이 이 모듈의 결론입니다. 그 전제가
# "두 모듈이 같은 데이터·같은 지표를 쓴다"이므로, 여기서 **bit 단위로** 대조해 둡니다.
# W1-M3 `01_ddpm_toy.py`가 체크포인트에 `data_ref = xn[:8192]`를 넣어둔 것을 그대로 읽습니다.

# %%
def verify_same_distribution(xn: np.ndarray) -> tuple[bool, str]:
    ck = find_repo_root() / "artifacts" / "W1-M3" / "01_model.pt"
    if not ck.is_file():
        return False, f"W1-M3 체크포인트가 없어 대조하지 못했습니다 ({ck})"
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    ref = np.asarray(blob["data_ref"])
    mine = xn[:ref.shape[0]]
    if ref.shape != mine.shape:
        return False, f"shape 불일치: W1-M3 {ref.shape} vs 여기 {mine.shape}"
    same = bool(np.array_equal(ref, mine))
    diff = float(np.abs(ref - mine).max())
    return same, (f"W1-M3 01_model.pt 의 data_ref와 **bit 단위로 동일** (max|Δ|={diff:g})"
                  if same else f"⚠️ 값이 다릅니다 (max|Δ|={diff:g}) — 02의 겹쳐 그리기를 신뢰하지 마세요")


# %% [markdown]
# ## 8. 그림

# %%
def _arrow(ax, origin, vec, color, label=None, lw=2.2, alpha=1.0):
    ax.annotate("", xy=(origin[0] + vec[0], origin[1] + vec[1]), xytext=(origin[0], origin[1]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                                shrinkA=0, shrinkB=0))
    if label:
        ax.plot([], [], color=color, lw=lw, label=label)


def plot_targets(x_star: np.ndarray, t_star: float, tg: np.ndarray, v_learn: np.ndarray,
                 v_exact: np.ndarray, x1_sel: np.ndarray, data: np.ndarray,
                 grid: dict, path: Path) -> Path:
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # (a) 속도 공간 — 개별 target 산점도 + 평균 + v_θ
    a = ax[0]
    a.scatter(tg[:, 0], tg[:, 1], s=8, alpha=0.30, color="#adb5bd",
              label=lab(f"개별 target $x_1-x_0$  ({len(tg)}개)", f"per-pair target $x_1-x_0$  (n={len(tg)})"))
    m = tg.mean(0)
    a.scatter([m[0]], [m[1]], s=170, marker="*", color="#111", zorder=6,
              label=lab("그 평균 (몬테카를로)", "their mean (Monte Carlo)"))
    a.scatter([v_learn[0]], [v_learn[1]], s=110, marker="X", color="#c92a2a", zorder=6,
              label=lab(r"학습된 $v_\theta(x^\star,t^\star)$", r"learned $v_\theta(x^\star,t^\star)$"))
    a.scatter([v_exact[0]], [v_exact[1]], s=110, marker="P", color="#1971c2", zorder=6,
              label=lab(r"닫힌 형태 $u_t(x^\star)$", r"closed form $u_t(x^\star)$"))
    a.axhline(0, color="#dee2e6", lw=1), a.axvline(0, color="#dee2e6", lw=1)
    a.set_xlabel("$u_x$"), a.set_ylabel("$u_y$")
    a.set_title(lab("(a) 개별 target은 사방으로 흩어진다 — 그런데 평균이 정답 (§4.3)",
                    "(a) per-pair targets scatter — yet the mean is the answer (§4.3)"), fontsize=11)
    # 세 마커가 겹쳐서 구분이 안 될 정도인 것이 요점이므로 숫자를 함께 적어 둡니다.
    sig, err = float(tg.std(0).mean() * np.sqrt(2)), float(np.linalg.norm(v_learn - v_exact))
    a.text(0.03, 0.03,
           lab(f"개별 target 산포 σ = {sig:.2f}\n"
               fr"$\|v_\theta - u_t\|$ = {err:.3f}   (σ의 {err / max(sig, 1e-9) * 100:.1f}%)",
               f"target spread σ = {sig:.2f}\n"
               fr"$\|v_\theta - u_t\|$ = {err:.3f}   ({err / max(sig, 1e-9) * 100:.1f}% of σ)"),
           transform=a.transAxes, fontsize=9.5, va="bottom", ha="left",
           bbox=dict(boxstyle="round,pad=0.4", fc="#fff9db", ec="#f59f00", alpha=0.9))
    a.legend(fontsize=8.5, loc="upper right"), a.grid(alpha=0.3), a.set_aspect("equal")

    # (b) 데이터 공간 — x_star 를 지나는 보간선들
    a = ax[1]
    a.scatter(data[:, 0], data[:, 1], s=2, alpha=0.10, color="#adb5bd")
    x0_sel = (x_star[None] - t_star * x1_sel) / (1.0 - t_star)   # x₀ = (x − t·x₁)/(1−t)
    for k in range(min(40, len(x1_sel))):
        a.plot([x0_sel[k, 0], x1_sel[k, 0]], [x0_sel[k, 1], x1_sel[k, 1]],
               color="#4dabf7", lw=0.7, alpha=0.45)
    a.scatter(x1_sel[:, 0], x1_sel[:, 1], s=14, color="#1971c2", alpha=0.7,
              label=lab("이 쌍들의 $x_1$", "$x_1$ of these pairs"))
    a.scatter([x_star[0]], [x_star[1]], s=140, marker="*", color="#c92a2a", zorder=6,
              label=lab(rf"$x^\star$  ($t^\star$={t_star:.2f})", rf"$x^\star$  ($t^\star$={t_star:.2f})"))
    _arrow(a, x_star, 0.35 * v_learn / (np.linalg.norm(v_learn) + 1e-9), "#c92a2a",
           lab(r"$v_\theta$ 방향", r"$v_\theta$ direction"))
    a.set_title(lab(r"(b) $x^\star$ 를 지나는 보간선들 — 도착지가 제각각",
                    r"(b) interpolants through $x^\star$ — all heading elsewhere"), fontsize=11)
    a.set_xlim(-3.5, 3.5), a.set_ylim(-3.5, 3.5), a.set_aspect("equal")
    a.legend(fontsize=8.5), a.grid(alpha=0.25)

    # (c) 벡터장 전체 — 닫힌 형태 vs 학습된 것
    a = ax[2]
    gx, gy, ue, ul = grid["gx"], grid["gy"], grid["exact"], grid["learn"]
    err = np.linalg.norm(ue - ul, axis=-1)
    sc = a.scatter(gx.ravel(), gy.ravel(), c=err.ravel(), s=52, cmap="magma_r",
                   vmin=0, vmax=max(err.max(), 1e-6))
    a.quiver(gx, gy, ue[..., 0], ue[..., 1], color="#1971c2", alpha=0.85,
             scale=28, width=0.004)
    a.quiver(gx, gy, ul[..., 0], ul[..., 1], color="#c92a2a", alpha=0.85,
             scale=28, width=0.004)
    a.plot([], [], color="#1971c2", lw=2, label=lab("닫힌 형태 $u_t$", "closed form $u_t$"))
    a.plot([], [], color="#c92a2a", lw=2, label=lab(r"학습된 $v_\theta$", r"learned $v_\theta$"))
    fig.colorbar(sc, ax=a, label=lab(r"$\|u_t-v_\theta\|$", r"$\|u_t-v_\theta\|$"))
    a.set_title(lab(rf"(c) $t$={grid['t']:.2f} 에서 벡터장 대조 — 겹칠수록 §4 정리가 성립",
                    rf"(c) vector fields at $t$={grid['t']:.2f} — overlap = theorem holds"), fontsize=11)
    a.set_aspect("equal"), a.legend(fontsize=8.5), a.grid(alpha=0.25)

    fig.suptitle(lab("W1-M4 · CFM 정리의 실증 — 개별 target은 전부 틀렸지만 그 평균이 주변 속도장이다",
                     "W1-M4 · CFM theorem in action — every per-pair target is wrong, their mean is not"),
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_crossing(cr: dict, traj: np.ndarray, v_at_cross: np.ndarray, data: np.ndarray,
                  s_value: float, path: Path) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # (a) 교차하는 두 보간선
    a = ax[0]
    a.scatter(data[:, 0], data[:, 1], s=2, alpha=0.10, color="#adb5bd")
    xt = cr["xt"]
    # 화면 크기를 먼저 정하고 화살표 길이를 그 비율로 잡습니다 — 각도가 보여야 하는 그림이라
    # 화살표가 눈에 띄어야 합니다(길이 자체는 정규화했으므로 의미 없습니다).
    pts = np.stack([cr["x0i"], cr["x1i"], cr["x0j"], cr["x1j"], xt])
    ctr = pts.mean(0)
    rad = float(np.abs(pts - ctr).max()) * 1.35
    L = 0.30 * rad

    def _unit(v):
        return L * np.asarray(v) / (np.linalg.norm(v) + 1e-9)

    for tag, c, nm in (("i", "#1971c2", "i"), ("j", "#f08c00", "j")):
        p0, p1 = cr[f"x0{tag}"], cr[f"x1{tag}"]
        a.plot([p0[0], p1[0]], [p0[1], p1[1]], color=c, lw=1.8, alpha=0.85)
        a.scatter([p0[0]], [p0[1]], color=c, marker="o", s=55)   # ● = x₀ (노이즈 쪽)
        a.scatter([p1[0]], [p1[1]], color=c, marker="s", s=55)   # ■ = x₁ (데이터 쪽)
        _arrow(a, xt, _unit(cr[f"u{tag}"]), c,
               lab(f"보간선 {nm} 과 그 target 방향", f"interpolant {nm} and its target"), lw=3.0)
    _arrow(a, xt, _unit(v_at_cross), "#c92a2a", lab(r"$v_\theta$ = 제3의 방향",
                                                    r"$v_\theta$ = a third direction"), lw=3.6)
    a.scatter([xt[0]], [xt[1]], s=190, marker="X", color="#c92a2a", zorder=7,
              label=lab(f"교차 근방  t={cr['t']:.2f}", f"near-crossing  t={cr['t']:.2f}"))
    a.set_title(lab(f"(a) 같은 (x,t)를 지나는 두 보간선 · 방향 cos={cr['cos']:.2f}  (§6.2)\n"
                    "● = $x_0$, ■ = $x_1$",
                    f"(a) two interpolants through one (x,t) · cos={cr['cos']:.2f}  (§6.2)\n"
                    "● = $x_0$, ■ = $x_1$"), fontsize=11)
    a.set_xlim(ctr[0] - rad, ctr[0] + rad), a.set_ylim(ctr[1] - rad, ctr[1] + rad)
    a.set_aspect("equal"), a.legend(fontsize=8.5, loc="best"), a.grid(alpha=0.25)

    # (b) 학습된 ODE 궤적은 휜다
    a = ax[1]
    a.scatter(data[:, 0], data[:, 1], s=2, alpha=0.10, color="#adb5bd")
    for k in range(traj.shape[1]):
        a.plot(traj[:, k, 0], traj[:, k, 1], color="#1971c2", lw=1.4, alpha=0.85)
        a.plot([traj[0, k, 0], traj[-1, k, 0]], [traj[0, k, 1], traj[-1, k, 1]],
               color="#868e96", lw=0.9, ls="--", alpha=0.9)
    a.plot([], [], color="#1971c2", lw=1.6, label=lab("학습된 ODE 궤적", "learned ODE trajectory"))
    a.plot([], [], color="#868e96", lw=1.2, ls="--", label=lab("그 시작점–끝점을 잇는 직선",
                                                               "straight chord of the same pair"))
    a.scatter(traj[0, :, 0], traj[0, :, 1], color="#111", s=26, zorder=5,
              label=lab("$x_0$ (노이즈)", "$x_0$ (noise)"))
    a.set_title(lab(f"(b) 조건부 경로는 직선인데 궤적은 휜다 — $S(Z)$={s_value:.3f}",
                    f"(b) conditional paths are straight, trajectories are not — $S(Z)$={s_value:.3f}"),
                fontsize=11)
    a.set_aspect("equal"), a.legend(fontsize=8.5, loc="best"), a.grid(alpha=0.25)

    fig.suptitle(lab("W1-M4 · 1스텝이 왜 그냥 되지 않는가 (§6.2) — 02에서 reflow로 이 휨을 폅니다",
                     "W1-M4 · why one Euler step is not free (§6.2) — 02 straightens this with reflow"),
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_samples_nfe(samples: dict[int, np.ndarray], real: np.ndarray, nfes: list[int],
                     path: Path) -> Path:
    fig, axes = plt.subplots(1, len(nfes), figsize=(2.05 * len(nfes), 2.55),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, nfe in zip(axes, nfes):
        ax.scatter(real[:, 0], real[:, 1], s=1.5, alpha=0.10, color="#adb5bd")
        p = samples[nfe]
        ax.scatter(p[:, 0], p[:, 1], s=2.0, alpha=0.40, color="#2f9e44")
        ax.set_xlim(-3, 3), ax.set_ylim(-3, 3)
        ax.set_xticks([]), ax.set_yticks([]), ax.set_aspect("equal")
        ax.set_title(f"NFE={nfe}", fontsize=11)
    axes[0].set_ylabel(lab("FM · Euler", "FM · Euler"), fontsize=11)
    fig.suptitle(lab("W1-M4 · 직선 경로로 학습한 모델을 Euler로 적분 (회색 = 원본 분포)",
                     "W1-M4 · Euler integration of a straight-path model (grey = data)"), fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_loss_quality(losses: list[float], bucket: np.ndarray, res: list[dict], floor: float,
                      log_every: int, path: Path) -> Path:
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))

    x = np.arange(1, len(losses) + 1) * log_every
    ax[0].plot(x, losses, lw=1.6, color="#2f9e44")
    ax[0].set_xlabel(lab("학습 스텝", "training step"))
    ax[0].set_ylabel(r"$\mathcal{L}_{\rm RF}$")
    ax[0].set_title(lab("(a) 학습 곡선 (§5.2)", "(a) training loss (§5.2)"))
    ax[0].grid(alpha=0.3)

    centers = np.arange(10) * 0.1 + 0.05
    ax[1].bar(centers, bucket, width=0.085, color="#495057")
    ax[1].set_xlabel(lab("$t$ 구간 (폭 0.1) — 0=노이즈, 1=데이터", "$t$ bucket (width 0.1) — 0=noise, 1=data"))
    ax[1].set_ylabel(lab(r"평균 $\|v_\theta-u\|^2$", r"mean $\|v_\theta-u\|^2$"))
    ax[1].set_title(lab("(b) $t$ 구간별 손실 — 어디가 어려운가",
                        "(b) loss by $t$ bucket — where is it hard"))
    ax[1].grid(alpha=0.3, axis="y")

    nfe = [d["nfe"] for d in res]
    ax[2].loglog(nfe, [d["energy"] for d in res], "o-", color="#2f9e44", lw=2,
                 label=lab("FM (1-rectified)", "FM (1-rectified)"))
    ax[2].axhline(floor, color="#868e96", ls="--", lw=1.2,
                  label=lab(f"지표 노이즈 바닥 {floor:.5f}", f"metric noise floor {floor:.5f}"))
    ax[2].set_xlabel("NFE"), ax[2].set_ylabel(lab("에너지 거리", "energy distance"))
    ax[2].set_title(lab("(c) 품질 vs NFE — 낮을수록 좋다", "(c) quality vs NFE — lower is better"))
    ax[2].legend(), ax[2].grid(alpha=0.3, which="both")

    fig.suptitle(lab("W1-M4 · 직선 경로 CFM — 학습과 NFE 응답",
                     "W1-M4 · straight-path CFM — training and NFE response"), fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# %% [markdown]
# ## 9. main

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W1-M4 실습 1: two moons에 CFM 직접 구현 (lesson §4·§5.2·§6)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--n-data", type=int, default=N_DATA)
    p.add_argument("--steps", type=int, default=STEPS, help="학습 스텝 수 (W1-M3 01과 동일)")
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--hidden", type=int, default=HIDDEN)
    p.add_argument("--layers", type=int, default=LAYERS)
    p.add_argument("--temb", type=int, default=TEMB)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--ema", type=float, default=EMA_DECAY)
    p.add_argument("--n-sample", type=int, default=N_SAMPLE)
    p.add_argument("--n-proj", type=int, default=N_PROJ)
    p.add_argument("--t-star", type=float, default=0.5, help="§4.3 실증에서 볼 시각 t*")
    p.add_argument("--n-pairs", type=int, default=200000, help="§4.3에서 뽑을 (x₀,x₁) 쌍 수")
    p.add_argument("--keep", type=int, default=400, help="§4.3에서 x* 근방으로 채택할 쌍 수")
    p.add_argument("--k-fine", type=int, default=100, help="직선성 S(Z) 계산용 Euler 스텝 수")
    p.add_argument("--smoke", action="store_true", help="수십 초에 완주하는 축소 경로")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--ascii-labels", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    nfes = NFE_GRID_SMOKE if args.smoke else NFE_GRID
    if args.smoke:
        # n_data 는 줄이지 않습니다 — 줄이면 W1-M3와 **다른 표본**이 되어 §7의 bit 대조가 깨집니다.
        args.steps, args.n_sample = 800, 1024
        args.n_pairs, args.keep, args.k_fine = 40000, 200, 30
        print("[smoke] 축소 경로로 실행합니다 (800 스텝). "
              "샘플 품질은 학습이 덜 된 값이라 결론에 쓰지 마세요.")
    setup_korean_font(args.ascii_labels)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = pick_device(args.device)
    out_dir = artifacts_dir()
    t_start = time.perf_counter()

    print("=" * 92)
    print(f"  {MODULE_ID} 실습 1 — two moons에 Conditional Flow Matching (직선 경로)")
    print("=" * 92)

    # --- [1] 손실이 정말 한 줄인가 (lesson §5.2·§5.3) -------------------------
    print("\n=== [1] lesson §5.2 — 손실 본문과 사라진 스케줄 테이블 ===")
    body = [ln for ln in source_of(cfm_loss).splitlines() if "eq. §5.2" in ln]
    for ln in body:
        print("   " + ln.strip())
    if not body:
        print("   (소스를 읽지 못했습니다 — cfm_loss() 정의를 직접 보세요)")
    n_ddpm_buf, n_fm_buf = schedule_footprint()
    print_table(["항목", "W1-M3 (DDPM)", "W1-M4 (직선 FM)"],
                [["시간 샘플링", "torch.randint(1, T+1)  이산", "torch.rand()  연속 U(0,1)"],
                 ["보간 계수", "√ᾱ_t, √(1−ᾱ_t)  ← 테이블 인덱싱", "t, 1−t  ← 그 자체"],
                 ["target", "ε (노이즈)", "x₁ − x₀ (속도, t 무관)"],
                 ["사전계산 버퍼 원소 수", f"{n_ddpm_buf:,}개 (길이 1000 × 7종)", f"{n_fm_buf}개"],
                 ["학습되는 스케줄 파라미터", "0개 (설계 상수)", "0개 (애초에 없음)"]],
                ["left", "left", "left"])
    print("  → lesson §5.3 표의 '스케줄 의존성' 행이 이것입니다. "
          "Schedule 데이터클래스가 통째로 사라집니다.")

    # --- [2] 데이터 — W1-M3와 같은 것인지 확인 --------------------------------
    print(f"\n=== [2] 데이터 — {DATA_KIND} n={args.n_data} (W1-M3 01과 같은 호출) ===")
    raw = make_data(DATA_KIND, args.n_data, args.seed)
    xn, mu, sd = normalize(raw)
    print(f"  정규화 후 평균 {xn.mean(0).round(3)}  표준편차 {xn.std(0).round(3)}")
    same, msg = verify_same_distribution(xn)
    print(f"  {'✅' if same else 'ⓘ'} {msg}")
    if not same:
        print("     (W1-M3 01_ddpm_toy.py 를 먼저 돌리면 대조됩니다. 겹쳐 그리기는 02의 몫입니다.)")
    x1_all = torch.as_tensor(xn, device=dev)
    ref = x1_all[:8192]
    ref_eval = ref[:args.n_sample]
    half = min(args.n_sample, ref.shape[0] // 2)
    floor = energy_distance(ref[:half], ref[half:2 * half])
    print(f"  지표 노이즈 바닥(같은 분포 두 표본, n={half}): energy={floor:.5f}"
          "   ← 이보다 작은 차이는 표본 노이즈입니다")

    # --- [3] 학습 -----------------------------------------------------------
    model = VelocityMLP(2, args.hidden, args.layers, args.temb).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\n=== [3] 학습 — L_RF, {args.steps} 스텝 batch {args.batch} ===")
    print(f"  v_θ 파라미터 {n_par:,}개 "
          f"(W1-M3 ε_θ와 **같은 백본** EpsMLP: hidden {args.hidden} × {args.layers}층)")
    log_every = max(1, args.steps // 120)
    t0 = time.perf_counter()
    ema, losses, bucket = train_cfm(model, x1_all, steps=args.steps, batch=args.batch, lr=args.lr,
                                    ema_decay=args.ema, log_every=log_every, seed=args.seed)
    train_sec = time.perf_counter() - t0
    print(f"  학습 {train_sec:.1f}s   최종 손실 {losses[-1]:.4f} (초기 {losses[0]:.4f})")
    ema.copy_to(model)
    model.eval()
    print("  t 구간별 평균 손실: "
          + "  ".join(f"{i / 10:.1f}-{(i + 1) / 10:.1f}: {v:.3f}" for i, v in enumerate(bucket)))
    hard = int(np.argmax(bucket))
    print(f"  ← 가장 어려운 구간은 t={hard / 10:.1f}~{(hard + 1) / 10:.1f} ({bucket[hard]:.3f}), "
          f"양 끝은 {bucket[0]:.3f} / {bucket[-1]:.3f}로 낮습니다. **뒤집힌 U자입니다.**")
    print("     양 끝에서 낮은 이유가 명확합니다 — t=0이면 x_t=x₀이라 target x₁−x₀의 불확실성이")
    print("     x₁ 쪽뿐이고(≈Var(x₁)), t=1이면 x_t=x₁이라 x₀ 쪽뿐입니다(≈Var(x₀)=1). 가운데서는")
    print("     둘 다 모르므로 잔차가 커집니다. W1-M3의 L_simple이 '작은 t가 어렵다'는 단조 형태였던")
    print("     것과 대비되고, 이 모양이 곧 §5.4가 말한 **암묵적 t-가중**의 실체입니다.")

    # --- [4] ★ §4.3 실증 -----------------------------------------------------
    print(f"\n=== [4] ★ lesson §4.3 — 개별 target은 전부 틀렸지만 평균이 정답 ===")
    rows, keep_pack = [], None
    for t_star in ([args.t_star] if args.smoke else [0.3, args.t_star, 0.7]):
        # x* 는 "그 시각에 실제로 확률질량이 있는 곳"이어야 의미가 있으므로 p_t에서 하나 뽑습니다.
        gsel = torch.Generator(device="cpu").manual_seed(args.seed + 11)
        k = int(torch.randint(0, x1_all.shape[0], (1,), generator=gsel))
        x_star = (1.0 - t_star) * torch.randn(2, generator=gsel).to(dev) + t_star * x1_all[k]
        tg, x1_sel = targets_through_point(x_star, t_star, x1_all, n_pairs=args.n_pairs,
                                           keep=args.keep, seed=args.seed + 5)
        mean_t = tg.mean(0)
        with torch.no_grad():
            v_learn = model(x_star[None], torch.full((1,), t_star, device=dev))[0]
        v_exact = exact_marginal_velocity(x_star[None], t_star, x1_all)[0]
        spread = float(tg.std(dim=0).norm())               # 개별 target이 흩어진 정도 σ
        se = spread / np.sqrt(len(tg))                     # 그 평균 자체의 표준오차 σ/√K
        e_mc = float((v_learn - mean_t).norm())            # 학습된 것 vs 몬테카를로 평균
        e_ex = float((v_learn - v_exact).norm())           # 학습된 것 vs 닫힌 형태
        rows.append([f"{t_star:.2f}", f"{spread:.3f}", f"{se:.3f}",
                     f"{e_mc:.3f}", f"{e_ex:.3f}", f"{e_ex / max(spread, 1e-9):.4f}"])
        if abs(t_star - args.t_star) < 1e-9:
            keep_pack = dict(t=t_star, x_star=x_star, tg=tg, x1_sel=x1_sel,
                             v_learn=v_learn, v_exact=v_exact)
    print_table(["t*", "개별 target 산포 σ", "평균의 표준오차 σ/√K",
                 "‖v_θ−평균‖", "‖v_θ−닫힌형태‖", "오차/산포"], rows, ["right"] * 6)
    print("  → 세 열을 나란히 읽는 것이 요점입니다. **개별 target은 σ만큼 흩어져 있는데**")
    print("     학습된 v_θ와 닫힌 형태의 차이는 그 수백분의 일입니다(마지막 열).")
    print("     ‖v_θ−평균‖이 그보다 큰 것은 v_θ가 틀려서가 아니라 **몬테카를로 평균 쪽에**")
    print("     표준오차 σ/√K 만큼의 잡음이 있기 때문입니다 — 두 열의 크기가 비슷하면 정상입니다.")
    print("     ℓ₂ 회귀의 최소해가 조건부 기댓값이라는 것 하나로 나오는 결과입니다(§4.3).")

    # 벡터장 전체 대조 (그림 (c)용)
    gx, gy = np.meshgrid(np.linspace(-2.6, 2.6, 17), np.linspace(-2.6, 2.6, 17))
    gp = torch.as_tensor(np.stack([gx.ravel(), gy.ravel()], 1), dtype=torch.float32, device=dev)
    t_grid = args.t_star
    with torch.no_grad():
        ul = model(gp, torch.full((gp.shape[0],), t_grid, device=dev)).cpu().numpy()
    ue = exact_marginal_velocity(gp, t_grid, x1_all).cpu().numpy()
    rel = float(np.linalg.norm(ue - ul) / max(np.linalg.norm(ue), 1e-9))
    print(f"  격자 {gp.shape[0]}점에서 벡터장 상대오차 ‖u−v_θ‖/‖u‖ = {rel:.3f}  (t={t_grid:.2f})")
    grid = dict(gx=gx, gy=gy, exact=ue.reshape(*gx.shape, 2), learn=ul.reshape(*gx.shape, 2),
                t=t_grid)

    # --- [5] §6.2 보간선 교차 + 궤적 -----------------------------------------
    print(f"\n=== [5] lesson §6.2 — 보간선 교차와 휘는 궤적 ===")
    ts = np.linspace(0.15, 0.85, 8 if not args.smoke else 4)
    cr = find_crossing(x1_all, n_pairs=2048 if not args.smoke else 512, seed=args.seed + 3, ts=ts)
    if cr:
        with torch.no_grad():
            v_cross = model(cr["xt"][None], torch.full((1,), cr["t"], device=dev))[0]
        ang = lambda a, b: float(np.degrees(np.arccos(np.clip(   # noqa: E731
            float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-9)), -1, 1))))
        print(f"  가장 뚜렷한 교차: t={cr['t']:.2f}, 두 x_t 사이 거리 {cr['dist']:.4f}, "
              f"두 target의 방향 cos={cr['cos']:.2f}")
        print(f"  그 점에서 v_θ는 두 보간선 방향과 각각 {ang(v_cross, cr['ui']):.0f}° / "
              f"{ang(v_cross, cr['uj']):.0f}° 벌어져 있습니다 — **어느 쪽 직선도 아닙니다.**")
    else:
        print("  ⚠️ 조건에 맞는 교차 쌍을 못 찾았습니다 (--n-pairs를 늘려보세요)")
    s_val = straightness(model, min(2048, args.n_sample), dev, k_fine=args.k_fine, seed=args.seed + 2)
    print(f"  직선성 지표 S(Z) = {s_val:.4f}   (eq. §6.2, K={args.k_fine} Euler로 근사)")
    print("  ← 0이면 완전한 직선이고 1스텝이 정확합니다. 0이 아닌 이유가 위의 교차입니다.")
    print("     이 값을 02가 reflow 전후로 비교합니다.")
    _, traj = euler_sample(model, 14, dev, nfe=args.k_fine, seed=args.seed + 4, trace=True)

    # --- [6] NFE 스윕 --------------------------------------------------------
    print(f"\n=== [6] NFE 스윕 — Euler 적분, 샘플 {args.n_sample}개 ===")
    euler_sample(model, 64, dev, nfe=4, seed=0)   # 워밍업 (첫 호출의 커널 로딩을 측정에서 제외)
    results, samples = [], {}
    for nfe in nfes:
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = euler_sample(model, args.n_sample, dev, nfe=nfe, seed=args.seed + 1)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        e = energy_distance(x, ref_eval)
        w = sliced_w1(x, ref_eval, args.n_proj, args.seed)
        results.append(dict(method="fm", nfe=nfe, energy=e, sw1=w, sec=sec))
        samples[nfe] = x.cpu().numpy()
        print(f"    NFE={nfe:4d}  energy={e:.5f}  sW1={w:.4f}  {sec * 1e3:8.1f} ms")
    best = min(results, key=lambda d: d["energy"])
    ok = [d for d in results if d["energy"] <= floor * 5]
    print(f"\n  최저 에너지 거리: NFE={best['nfe']}  ({best['energy']:.5f}), "
          f"노이즈 바닥 {floor:.5f}")
    print(f"  노이즈 바닥 5배({floor * 5:.5f}) 안에 드는 최소 NFE: "
          + (f"**{min(ok, key=lambda d: d['nfe'])['nfe']}**" if ok else "없음"))

    # NFE=1이 왜 실패하는가 — 값이 아니라 **구조**를 보여줘야 합니다 (lesson §6.1 vs §6.2)
    s1 = samples[nfes[0]]
    print(f"\n  ⚠️ NFE={nfes[0]} 샘플의 표준편차 {s1.std(0).round(3)} vs 데이터 "
          f"{ref_eval.cpu().numpy().std(0).round(3)}")
    print("     거의 한 점으로 뭉칩니다. 우연이 아니라 **정확히 예측되는 결과**입니다 — 독립 짝짓기에서")
    print("     u₀(x)=E[x₁−x₀|x₀=x]=E[x₁]−x 이므로 x + 1·u₀(x) = E[x₁], 즉 시작점과 무관하게")
    print("     데이터 평균 한 점으로 갑니다. lesson §6.1이 '직선이면 1스텝이 정확'이라고 한 것은")
    print("     **조건부 경로**의 이야기이고, 학습된 주변 속도장은 §6.2대로 휘어 있습니다.")
    print("     이 한 점을 분포로 되돌리는 것이 02의 reflow입니다.")
    print("  ⓘ NFE=1·2 행을 W1-M3 02의 같은 행과 비교해 보세요 — 겹쳐 그리기는 02가 합니다.")

    # --- [7] 산출물 ----------------------------------------------------------
    if not args.no_plot:
        if keep_pack is not None:
            p = plot_targets(keep_pack["x_star"].cpu().numpy(), keep_pack["t"],
                             keep_pack["tg"].cpu().numpy(), keep_pack["v_learn"].cpu().numpy(),
                             keep_pack["v_exact"].cpu().numpy(), keep_pack["x1_sel"].cpu().numpy(),
                             xn[:4096], grid, out_dir / "01_targets.png")
            print(f"\n  [저장] {p}   ← ★ 이 모듈에서 가장 교육적인 그림")
        if cr:
            crn = {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in cr.items()}
            p = plot_crossing(crn, traj.cpu().numpy(), v_cross.cpu().numpy(), xn[:4096],
                              s_val, out_dir / "01_crossing.png")
            print(f"  [저장] {p}")
        p = plot_samples_nfe(samples, ref_eval.cpu().numpy(), nfes, out_dir / "01_samples_nfe.png")
        print(f"  [저장] {p}")
        p = plot_loss_quality(losses, bucket, results, floor, log_every,
                              out_dir / "01_loss_quality.png")
        print(f"  [저장] {p}")

    csv_path = out_dir / "01_nfe.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "nfe", "energy", "sw1", "sec"])
        w.writeheader()
        w.writerows(results)
    print(f"  [저장] {csv_path}")

    ck = out_dir / ("01_model_smoke.pt" if args.smoke else "01_model.pt")
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "cfg": dict(data_dim=2, hidden=args.hidden, n_layers=args.layers,
                    temb_dim=args.temb, t_scale=T_SCALE),
        "data": dict(kind=DATA_KIND, n=args.n_data, mu=mu, sd=sd, seed=args.seed),
        "data_ref": xn[:8192],           # 02가 품질 지표의 기준 분포로 씁니다 (W1-M3와 같은 규약)
        "straightness": s_val,
        "smoke": args.smoke,
        "train_steps": args.steps,
    }, ck)
    print(f"  [저장] {ck}  ({ck.stat().st_size / 1e6:.2f} MB)"
          "   ← 02_rectified_flow_reflow.py 가 이 파일을 읽습니다")

    print(f"\n총 소요 {time.perf_counter() - t_start:.1f}s")
    print("다음: python 02_rectified_flow_reflow.py   "
          "(reflow로 궤적을 펴고 W1-M3의 DDPM/DDIM 곡선과 같은 축에 겹칩니다)")


if __name__ == "__main__":
    import sys

    # 노트북(ipykernel)에서는 argparse가 jupyter의 -f 인자를 먹지 않도록 빈 리스트를 넘긴다.
    # → 전부 기본값으로 실행됩니다. --smoke로 돌리려면 이 셀을 main(["--smoke"])로 고치세요.
    main(None if "ipykernel" not in sys.modules else [])
