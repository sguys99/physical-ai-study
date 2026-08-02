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
# # W1-M3 실습 1 — 스케줄과 손실을 숫자로 확인하고, 2D toy에 미니 DDPM
#
# lesson.md `§3`(forward) · `§4`(reverse와 손실)의 실행판입니다. 세 가지를 합니다.
#
# **(A) `§3.3` 표를 코드가 다시 계산합니다.** $T=1000$, $\beta$ 선형 $10^{-4}\to0.02$에서
# $t\in\{1,100,250,500,1000\}$의 $\beta_t,\bar\alpha_t,\sqrt{\bar\alpha_t},\sqrt{1-\bar\alpha_t},\mathrm{SNR}$.
# lesson에 인쇄된 값과 **`assert`로 대조**합니다 — 어긋나면 코드가 아니라 표를 의심해야 하는 지점입니다.
#
# **(B) `§4.4`의 $\lambda_t$ 표**도 같은 방식으로 대조합니다($\sigma_t^2=\beta_t$ 가정).
# $\lambda_1/\lambda_{1000}\approx49$ — ELBO가 작은 $t$에 몇 배를 걸어주고 있었는지가 이 숫자입니다.
#
# **(C) 2D toy 분포(`make_moons`)에 미니 DDPM**을 학습합니다. $\epsilon_\theta(x_t,t)$는 작은 MLP,
# 시간 임베딩은 sinusoidal. 학습이 끝나면 ancestral로 뽑아 원본과 겹쳐 봅니다.
#
# 출력(`artifacts/W1-M3/`):
#
# | 파일 | 내용 |
# |---|---|
# | `01_schedule.png` | $\bar\alpha_t$·SNR 곡선 + $\lambda_t$ (3-panel) |
# | `01_forward_diffusion.png` | $t$를 키우며 데이터가 $\mathcal{N}(0,I)$로 무너지는 격자 |
# | `01_loss.png` | 학습 곡선 + **$t$ 구간별 평균 손실** |
# | `01_samples.png` | ancestral 샘플 vs 원본 |
# | `01_model.pt` | 체크포인트 — **`02_samplers_compare.py`가 읽습니다** (gitignore) |
#
# **GPU 불필요.** 이 모듈은 데이터가 2차원이고 모델이 작아 CPU에서 완주합니다
# (실측은 `README.md` §2.1). `--smoke`는 수십 초.

# %%
from __future__ import annotations

import argparse
import math
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless 고정 — 뷰어를 띄우지 않는다

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402
from sklearn.datasets import make_moons, make_swiss_roll  # noqa: E402

MODULE_ID = "W1-M3"

# ── lesson §3.3 표에 인쇄된 값 (ground truth) ────────────────────────────────
# 코드가 이 값을 재현하지 못하면 둘 중 하나가 틀린 것입니다. 조용히 넘기지 않습니다.
# 열 순서: (t, β_t, ᾱ_t, √ᾱ_t, √(1-ᾱ_t), SNR)
LESSON_3_3 = [
    (1, 0.00010, 0.99990, 0.99995, 0.0100, 9999.0),
    (100, 0.00207, 0.89702, 0.94711, 0.3209, 8.71),
    (250, 0.00506, 0.52409, 0.72394, 0.6899, 1.10),
    (500, 0.01004, 0.07859, 0.28033, 0.9599, 0.0853),
    (1000, 0.02000, 4.04e-5, 0.00635, 1.0000, 4.04e-5),
]
# ── lesson §4.4 λ_t 표 (σ_t² = β_t 가정) ────────────────────────────────────
LESSON_4_4 = [(1, 0.500), (2, 0.273), (10, 0.0737), (100, 0.0101), (500, 0.00550), (1000, 0.0102)]
LESSON_4_4_RATIO = 49.0  # λ_1 / λ_1000


# %% [markdown]
# ## 0. 경로 · 폰트 · 표 유틸 (W1-M2·M5 practice와 동일 규약)
#
# > `t`는 이 모듈에서 diffusion 스텝이므로, W1-M5의 번역 헬퍼 `t(ko, en)`은
# > 여기서 **`lab(ko, en)`** 으로 이름만 바꿨습니다. 역할은 같습니다.

# %%
_ROOT_MARKERS = ("course", "docs", "CLAUDE.md")


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


def artifacts_dir() -> Path:
    out = find_repo_root() / "artifacts" / MODULE_ID
    out.mkdir(parents=True, exist_ok=True)
    return out


_KO_FONT_PREFERENCE = (
    "Pretendard", "NanumGothic", "Nanum Gothic", "Malgun Gothic", "NanumBarunGothic",
    "Noto Sans KR", "Noto Sans CJK KR", "Noto Sans CJK JP", "Source Han Sans KR",
    "AppleGothic", "Spoqa Han Sans Neo", "UnDotum", "Baekmuk Gulim",
)
_PROBE_CHARS = "한글노이즈스텝"

USE_KOREAN = False


def _has_hangul(font_path: str) -> bool:
    try:
        face = FT2Font(font_path)
        return all(face.get_char_index(ord(c)) != 0 for c in _PROBE_CHARS)
    except Exception:
        return False


def setup_korean_font(force_ascii: bool = False) -> bool:
    global USE_KOREAN
    plt.rcParams["axes.unicode_minus"] = False
    if force_ascii:
        USE_KOREAN = False
        print("[font] --ascii-labels 지정 → 영문 라벨로 렌더합니다.")
        return False
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in _KO_FONT_PREFERENCE:
        if name not in installed:
            continue
        path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
        if _has_hangul(path):
            plt.rcParams["font.family"] = name
            USE_KOREAN = True
            print(f"[font] 한글 폰트 사용: {name}")
            return True
    USE_KOREAN = False
    print("[font] 경고: 한글 글리프 폰트를 찾지 못해 영문 라벨로 폴백합니다."
          " (해결: apt install fonts-nanum 후 matplotlib 캐시 삭제)")
    return False


def lab(ko: str, en: str) -> str:
    """그림 라벨 전용 번역 헬퍼 (W1-M5의 t(ko,en)와 같은 역할)."""
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
    """한글 폭을 고려한 간단한 표 출력."""
    aligns = aligns or ["left"] * len(headers)
    widths = [max(_dwidth(h), *(_dwidth(r[i]) for r in rows)) if rows else _dwidth(h)
              for i, h in enumerate(headers)]
    print("  " + " | ".join(_pad(h, w, "center") for h, w in zip(headers, widths)))
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        print("  " + " | ".join(_pad(c, w, a) for c, w, a in zip(r, widths, aligns)))


def pick_device(choice: str) -> torch.device:
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA를 못 찾아 CPU로 폴백합니다.")
        choice = "cpu"
    dev = torch.device(choice)
    if dev.type == "cuda":
        print(f"[device] {torch.cuda.get_device_name(0)}  "
              f"capability {torch.cuda.get_device_capability(0)}  torch {torch.__version__}")
    else:
        print(f"[device] CPU  torch {torch.__version__}  "
              "(이 모듈은 데이터가 2차원이라 GPU 이득이 거의 없습니다)")
    return dev


# %% [markdown]
# ## 1. 스케줄 — 설계 상수이지 학습 대상이 아니다
#
# lesson §3.1이 못박은 그대로입니다. $\beta_t$ 스케줄 전체가 **설계 상수**라 학습되는 파라미터가 하나도
# 없습니다. 제어로 읽으면 $x_t = A_t x_{t-1} + w_t$에서 $A_t=\sqrt{\alpha_t}I$, $\mathrm{Cov}(w_t)=\beta_t I$
# 인 시변 시스템의 계수를 손으로 적어놓은 것입니다.
#
# **인덱스 규약**: 배열 인덱스 `i`가 $t=i+1$입니다. 논문의 $t$는 1부터, 파이썬은 0부터라
# 여기서 한 번 어긋나면 표가 통째로 밀립니다.

# %%
@dataclass
class Schedule:
    """DDPM 선형 β 스케줄. 텐서는 전부 [T], 인덱스 i ↔ 스텝 t=i+1."""
    T: int
    beta: torch.Tensor
    alpha: torch.Tensor
    abar: torch.Tensor          # ᾱ_t = Π α_s
    abar_prev: torch.Tensor     # ᾱ_{t-1}, ᾱ_0 := 1
    sqrt_abar: torch.Tensor     # √ᾱ_t          ← 신호 전달 이득
    sqrt_one_minus_abar: torch.Tensor  # √(1-ᾱ_t) ← 누적 잡음 표준편차
    beta_tilde: torch.Tensor    # β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t)·β_t   (lesson §4.1)

    def to(self, device: torch.device) -> "Schedule":
        return Schedule(self.T, *(getattr(self, f).to(device) for f in
                                  ("beta", "alpha", "abar", "abar_prev", "sqrt_abar",
                                   "sqrt_one_minus_abar", "beta_tilde")))

    def snr(self) -> torch.Tensor:
        # lesson §3.2: SNR(t) = ᾱ_t / (1-ᾱ_t)
        return self.abar / (1.0 - self.abar)

    def lambda_t(self) -> torch.Tensor:
        # lesson §4.4: λ_t = β_t² / (2σ_t²α_t(1-ᾱ_t)),  σ_t² = β_t 로 두면 β_t/(2α_t(1-ᾱ_t))
        return self.beta / (2.0 * self.alpha * (1.0 - self.abar))


def make_schedule(T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02,
                  dtype: torch.dtype = torch.float64) -> Schedule:
    beta = torch.linspace(beta_start, beta_end, T, dtype=dtype)
    alpha = 1.0 - beta
    abar = torch.cumprod(alpha, dim=0)
    abar_prev = torch.cat([torch.ones(1, dtype=dtype), abar[:-1]])
    beta_tilde = (1.0 - abar_prev) / (1.0 - abar) * beta
    return Schedule(T, beta, alpha, abar, abar_prev,
                    abar.sqrt(), (1.0 - abar).sqrt(), beta_tilde)


# %% [markdown]
# ### 1.1 lesson §3.3 표 재현 — 대조까지
#
# `assert`가 통과해야 정상입니다. 실패하면 **표가 틀렸을 가능성**을 먼저 의심하세요
# (코드는 정의 그대로 4줄이고, 표는 사람이 옮겨 적은 것입니다).

# %%
def check_schedule_table(sch: Schedule, tol_digits: int = 3) -> list[list[str]]:
    """lesson §3.3의 5행을 재계산하고 인쇄값과 대조한다."""
    rows: list[list[str]] = []
    for t, b_ref, ab_ref, sq_ref, sq1_ref, snr_ref in LESSON_3_3:
        i = t - 1  # 인덱스 규약: i = t-1
        got = (float(sch.beta[i]), float(sch.abar[i]), float(sch.sqrt_abar[i]),
               float(sch.sqrt_one_minus_abar[i]), float(sch.snr()[i]))
        ref = (b_ref, ab_ref, sq_ref, sq1_ref, snr_ref)
        ok = all(abs(g - r) <= max(abs(r), 1e-12) * 10 ** (-tol_digits + 1) + 1e-9
                 for g, r in zip(got, ref))
        rows.append([str(t), f"{got[0]:.5f}", f"{got[1]:.5f}", f"{got[2]:.5f}",
                     f"{got[3]:.4f}", f"{got[4]:.4g}", "일치" if ok else "★불일치"])
        assert ok, (f"lesson §3.3 t={t} 불일치: 계산 {got} vs 표 {ref}\n"
                    "  → 코드가 아니라 lesson 표를 확인하세요.")
    return rows


def check_lambda_table(sch: Schedule, tol_digits: int = 3) -> list[list[str]]:
    """lesson §4.4의 λ_t 6행 + λ_1/λ_1000 비를 대조한다."""
    lam = sch.lambda_t()
    rows: list[list[str]] = []
    for t, ref in LESSON_4_4:
        got = float(lam[t - 1])
        ok = abs(got - ref) <= abs(ref) * 10 ** (-tol_digits + 1)
        rows.append([str(t), f"{got:.6g}", f"{ref:g}", "일치" if ok else "★불일치"])
        assert ok, f"lesson §4.4 t={t} 불일치: 계산 {got:.6g} vs 표 {ref}"
    ratio = float(lam[0] / lam[-1])
    assert abs(ratio - LESSON_4_4_RATIO) < 1.0, f"λ_1/λ_1000 = {ratio:.2f}, 표는 ≈{LESSON_4_4_RATIO}"
    return rows


# %% [markdown]
# ## 2. Forward — 닫힌 형태 한 줄
#
# lesson §3.2가 유도한 결과가 코드로는 정확히 한 줄입니다. **이 한 줄이 diffusion을 학습 가능하게
# 만들었습니다** — 없으면 $x_t$를 얻으려고 체인을 $t$번 굴려야 하므로 $t\sim\mathcal{U}\{1..T\}$
# 배치 학습이 불가능합니다.

# %%
def q_sample(x0: torch.Tensor, t: torch.Tensor, sch: Schedule,
             eps: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """x_0과 스텝 t에서 x_t를 한 번에 만든다. t는 1-based 정수 텐서 [B]."""
    if eps is None:
        eps = torch.randn_like(x0)
    i = t - 1
    # lesson §3.2 닫힌 형태:  x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    x_t = sch.sqrt_abar[i][:, None] * x0 + sch.sqrt_one_minus_abar[i][:, None] * eps
    return x_t, eps


# %% [markdown]
# ## 3. 데이터 — 2D toy
#
# 이미지가 아니라 **2차원 점 구름**을 씁니다. 분포 전체를 산점도 한 장으로 볼 수 있어
# "샘플러가 분포를 맞혔는가"를 FID 같은 대리지표 없이 눈으로 판정할 수 있기 때문입니다.
# `--data swiss`를 주면 swiss roll의 2D 투영을 씁니다.

# %%
def make_data(kind: str, n: int, seed: int) -> np.ndarray:
    if kind == "moons":
        x, _ = make_moons(n_samples=n, noise=0.06, random_state=seed)
    elif kind == "swiss":
        x3, _ = make_swiss_roll(n_samples=n, noise=0.35, random_state=seed)
        x = np.stack([x3[:, 0], x3[:, 2]], axis=1)  # 2D 투영
    else:
        raise ValueError(kind)
    x = x.astype(np.float32)
    return x


def normalize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DDPM은 데이터가 대략 N(0,1) 스케일이라고 가정한다 (prior가 N(0,I)이므로)."""
    mu, sd = x.mean(0), x.std(0)
    return (x - mu) / sd, mu, sd


# %% [markdown]
# ## 4. 모델 — $\epsilon_\theta(x_t, t)$
#
# 시간 임베딩은 sinusoidal(Transformer의 위치 인코딩과 같은 형태)이고, 본체는 잔차 MLP입니다.
# 시간 조건은 각 층에 **더해서** 넣습니다 — DiT의 adaLN이 하는 일(§6.2)의 최소판이라고 보면 됩니다.
# 여기서는 scale 없이 shift만 씁니다.

# %%
def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """sinusoidal 시간 임베딩. t: [B] (1-based 정수) → [B, dim]"""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class EpsMLP(nn.Module):
    """ε_θ(x_t, t) — 2D toy용 잔차 MLP."""

    def __init__(self, data_dim: int = 2, hidden: int = 256, n_layers: int = 4, temb_dim: int = 128):
        super().__init__()
        self.temb_dim = temb_dim
        self.temb = nn.Sequential(nn.Linear(temb_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.inp = nn.Linear(data_dim, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(n_layers))
        self.shift = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(n_layers))
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, data_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        c = self.temb(timestep_embedding(t, self.temb_dim))
        h = self.inp(x)
        for blk, sh in zip(self.blocks, self.shift):
            h = h + blk(h + sh(c))  # 시간 조건 주입 = shift만 쓰는 FiLM (DiT adaLN의 축소판)
        return self.out(h)


class EMA:
    """가중치 지수이동평균. DDPM 원논문도 씁니다 — 2D toy에서도 샘플 품질 차이가 큽니다.

    **워밍업이 필수입니다.** decay=0.999를 처음부터 쓰면 시정수가 1,000 스텝이라
    `--smoke`(수백 스텝)에서는 EMA 가중치가 사실상 **초기값**에 눌립니다. 초기값은
    (03의 zero-init 헤드처럼) 아무것도 예측하지 못하는 상태라 샘플이 발산합니다.
    그래서 초반에는 decay를 (1+n)/(10+n)로 낮춰 씁니다 — 흔히 쓰는 처방입니다.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.n = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))  # 워밍업
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


# %% [markdown]
# ## 5. 학습 — $\mathcal{L}_{\text{simple}}$
#
# lesson §4.4의 박스가 그대로 세 줄입니다. `t`를 균등하게 뽑는 것이 **$\lambda_t$를 버렸다**는 말의
# 실체입니다 — ELBO대로라면 $t=1$ 쪽에 약 50배 가중치를 걸어야 합니다(§1.1의 λ 표).

# %%
def train(model: nn.Module, x0_all: torch.Tensor, sch: Schedule, *, steps: int, batch: int,
          lr: float, ema_decay: float, log_every: int, seed: int) -> tuple[EMA, list[float], np.ndarray]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.05, 1.0 - s / steps))
    ema = EMA(model, ema_decay)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = x0_all.shape[0]
    losses: list[float] = []
    # t 구간별 손실 누적 (10구간) — λ_t 논의를 눈으로 확인하기 위한 진단
    bucket_sum = np.zeros(10)
    bucket_cnt = np.zeros(10)
    run = 0.0
    for s in range(1, steps + 1):
        idx = torch.randint(0, n, (batch,), generator=g).to(x0_all.device)
        x0 = x0_all[idx]
        t = torch.randint(1, sch.T + 1, (batch,), generator=g).to(x0_all.device)  # t ~ U{1..T}
        x_t, eps = q_sample(x0, t, sch)
        eps_hat = model(x_t, t)
        per = ((eps - eps_hat) ** 2).mean(dim=1)
        loss = per.mean()  # lesson §4.4  L_simple = E_{t,x0,ε} ‖ε − ε_θ(x_t, t)‖²
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        ema.update(model)
        run += loss.item()
        if s % log_every == 0:
            losses.append(run / log_every)
            run = 0.0
        if s > steps * 0.8:  # 학습 후반부만 집계 (초반 과도구간 제외)
            b = ((t - 1).float() / sch.T * 10).long().clamp(0, 9).cpu().numpy()
            pv = per.detach().cpu().numpy()
            np.add.at(bucket_sum, b, pv)
            np.add.at(bucket_cnt, b, 1.0)
    return ema, losses, bucket_sum / np.maximum(bucket_cnt, 1.0)


# %% [markdown]
# ## 6. 샘플링 — ancestral
#
# lesson §4.3의 $\mu_\theta$ 재파라미터화를 그대로 옮깁니다. 여기서는 **전체 격자 $T=1000$** 만 씁니다.
# 격자를 건너뛰는 이야기(DDIM · NFE)는 `02_samplers_compare.py`의 몫입니다.

# %%
@torch.no_grad()
def ancestral_sample(model: nn.Module, n: int, sch: Schedule, device: torch.device,
                     seed: int = 0, dim: int = 2) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, dim, generator=g).to(device)
    for t in range(sch.T, 0, -1):
        i = t - 1
        tt = torch.full((n,), t, dtype=torch.long, device=device)
        eps = model(x, tt)
        # lesson §4.3:  μ_θ = (1/√α_t)·(x_t − β_t/√(1-ᾱ_t)·ε_θ)
        mean = (x - sch.beta[i] / sch.sqrt_one_minus_abar[i] * eps) / sch.alpha[i].sqrt()
        if t > 1:
            # σ_t² = β̃_t (DDPM 논문의 두 선택지 중 하한. σ_t²=β_t로 둬도 결과는 비슷합니다)
            z = torch.randn(x.shape, generator=g).to(device)
            x = mean + sch.beta_tilde[i].sqrt() * z
        else:
            x = mean
    return x


# %% [markdown]
# ## 7. 그림

# %%
def plot_schedule(sch: Schedule, path: Path) -> Path:
    t = np.arange(1, sch.T + 1)
    abar = sch.abar.numpy()
    snr = sch.snr().numpy()
    lam = sch.lambda_t().numpy()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].plot(t, abar, label=r"$\bar\alpha_t$", lw=2)
    ax[0].plot(t, np.sqrt(abar), label=r"$\sqrt{\bar\alpha_t}$", lw=2, ls="--")
    ax[0].plot(t, np.sqrt(1 - abar), label=r"$\sqrt{1-\bar\alpha_t}$", lw=2, ls=":")
    ax[0].set_xlabel("t"), ax[0].set_ylabel(lab("값", "value"))
    ax[0].set_title(lab("(a) 신호 이득과 잡음 표준편차 (§3.2)",
                        "(a) signal gain vs noise std (§3.2)"))
    ax[0].legend(), ax[0].grid(alpha=0.3)

    t_cross = int(np.argmin(np.abs(snr - 1.0))) + 1
    ax[1].semilogy(t, snr, lw=2, color="#c92a2a")
    ax[1].axhline(1.0, color="k", ls="--", lw=1)
    ax[1].axvline(t_cross, color="#868e96", ls=":", lw=1.5)
    ax[1].annotate(lab(f"SNR=1 은 t≈{t_cross}", f"SNR=1 at t≈{t_cross}"),
                   xy=(t_cross, 1.0), xytext=(t_cross + 0.09 * sch.T, 30),
                   arrowprops=dict(arrowstyle="->", color="#868e96"), fontsize=10)
    ax[1].set_xlabel("t"), ax[1].set_ylabel("SNR")
    ax[1].set_title(lab(r"(b) SNR$(t)=\bar\alpha_t/(1-\bar\alpha_t)$ (§3.3)",
                        r"(b) SNR$(t)=\bar\alpha_t/(1-\bar\alpha_t)$ (§3.3)"))
    ax[1].grid(alpha=0.3, which="both")

    ax[2].semilogy(t, lam, lw=2, color="#1971c2")
    ax[2].scatter([1, sch.T], [lam[0], lam[-1]], color="#c92a2a", zorder=5)
    ax[2].annotate(lab(f"$\\lambda_1/\\lambda_{{{sch.T}}}$ ≈ {lam[0] / lam[-1]:.0f}배",
                       f"$\\lambda_1/\\lambda_{{{sch.T}}}$ ≈ {lam[0] / lam[-1]:.0f}x"),
                   xy=(1, lam[0]), xytext=(0.22 * sch.T, lam.max() * 0.5),
                   fontsize=11, color="#c92a2a")
    ax[2].set_xlabel("t"), ax[2].set_ylabel(r"$\lambda_t$")
    ax[2].set_title(lab(r"(c) 버려진 ELBO 가중치 $\lambda_t$ (§4.4)",
                        r"(c) discarded ELBO weight $\lambda_t$ (§4.4)"))
    ax[2].grid(alpha=0.3, which="both")

    fig.suptitle(lab(f"W1-M3 · DDPM 선형 스케줄 T={sch.T}, β: 1e-4 → 0.02",
                     f"W1-M3 · DDPM linear schedule T={sch.T}, beta: 1e-4 -> 0.02"), fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_forward(x0: torch.Tensor, sch: Schedule, path: Path, seed: int = 0) -> Path:
    # T=1000이면 lesson §3.3 표와 같은 t를 쓴다. --T를 바꾸면 같은 비율로 스케일.
    steps = [0, 1, 100, 250, 500, 1000] if sch.T == 1000 else \
        [0, 1] + sorted({max(1, round(sch.T * f)) for f in (0.1, 0.25, 0.5, 1.0)})
    n = min(2000, x0.shape[0])
    torch.manual_seed(seed)
    xs = x0[:n]
    fig, axes = plt.subplots(1, len(steps), figsize=(3.0 * len(steps), 3.2), sharex=True, sharey=True)
    for ax, t in zip(axes, steps):
        if t == 0:
            pts = xs
            sub = lab("원본 $x_0$", "data $x_0$")
        else:
            tt = torch.full((n,), t, dtype=torch.long)
            pts, _ = q_sample(xs, tt, sch)
            i = t - 1
            sub = (f"$\\sqrt{{\\bar\\alpha}}$={float(sch.sqrt_abar[i]):.3f}  "
                   f"SNR={float(sch.snr()[i]):.3g}")
        p = pts.numpy()
        ax.scatter(p[:, 0], p[:, 1], s=3, alpha=0.35, color="#1971c2")
        ax.set_title(f"t = {t}\n{sub}", fontsize=10)
        ax.set_xlim(-4, 4), ax.set_ylim(-4, 4)
        ax.set_aspect("equal"), ax.grid(alpha=0.2)
    fig.suptitle(lab("forward 확산 — 알려진 선형 시스템이 데이터를 $\\mathcal{N}(0,I)$로 민다 (§3)",
                     "forward diffusion — a known linear system pushes data to $\\mathcal{N}(0,I)$ (§3)"),
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_loss(losses: list[float], bucket: np.ndarray, log_every: int, path: Path) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(1, len(losses) + 1) * log_every
    ax[0].plot(x, losses, lw=1.6, color="#1971c2")
    ax[0].set_xlabel(lab("학습 스텝", "training step"))
    ax[0].set_ylabel(r"$\mathcal{L}_{\rm simple}$")
    ax[0].set_title(lab("(a) 학습 곡선", "(a) training loss"))
    ax[0].grid(alpha=0.3)

    centers = np.arange(10) * 100 + 50
    ax[1].bar(centers, bucket, width=85, color="#495057")
    ax[1].set_xlabel(lab("t 구간 (100 단위)", "t bucket (width 100)"))
    ax[1].set_ylabel(lab("평균 $\\|\\epsilon-\\epsilon_\\theta\\|^2$", "mean $\\|\\epsilon-\\epsilon_\\theta\\|^2$"))
    ax[1].set_title(lab("(b) t 구간별 손실 — 어디가 어려운가",
                        "(b) loss by t bucket — where is it hard"))
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_samples(real: np.ndarray, fake: np.ndarray, path: Path, nfe: int) -> Path:
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4))
    ax[0].scatter(real[:, 0], real[:, 1], s=4, alpha=0.35, color="#1971c2")
    ax[0].set_title(lab(f"(a) 원본 데이터  n={len(real)}", f"(a) data  n={len(real)}"))
    ax[1].scatter(fake[:, 0], fake[:, 1], s=4, alpha=0.35, color="#c92a2a")
    ax[1].set_title(lab(f"(b) ancestral 샘플  NFE={nfe}", f"(b) ancestral samples  NFE={nfe}"))
    ax[2].scatter(real[:, 0], real[:, 1], s=4, alpha=0.25, color="#1971c2",
                  label=lab("원본", "data"))
    ax[2].scatter(fake[:, 0], fake[:, 1], s=4, alpha=0.25, color="#c92a2a",
                  label=lab("생성", "samples"))
    ax[2].legend(markerscale=3)
    ax[2].set_title(lab("(c) 겹쳐 보기", "(c) overlay"))
    for a in ax:
        a.set_xlim(-3, 3), a.set_ylim(-3, 3), a.set_aspect("equal"), a.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# %% [markdown]
# ## 8. main

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W1-M3 실습 1: DDPM 스케줄 검산 + 2D toy 미니 DDPM (lesson §3·§4)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--data", default="moons", choices=["moons", "swiss"])
    p.add_argument("--n-data", type=int, default=16384)
    p.add_argument("--T", type=int, default=1000, help="확산 스텝 수 (기본 1000 = DDPM 원설정)")
    p.add_argument("--steps", type=int, default=12000, help="학습 스텝 수")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--temb", type=int, default=128, help="sinusoidal 시간 임베딩 차원")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--n-sample", type=int, default=4096, help="학습 후 뽑을 샘플 수")
    p.add_argument("--smoke", action="store_true", help="수십 초에 완주하는 축소 경로 (경로 확인용)")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--ascii-labels", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.smoke:
        args.steps, args.n_data, args.n_sample = 600, 4096, 512
        print("[smoke] 축소 경로로 실행합니다 (600 스텝). "
              "샘플 품질은 학습이 덜 된 값이라 결론에 쓰지 마세요.")
    setup_korean_font(args.ascii_labels)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = pick_device(args.device)
    out_dir = artifacts_dir()
    t_start = time.perf_counter()

    print("=" * 84)
    print(f"  {MODULE_ID} 실습 1 — DDPM 스케줄 검산 + 2D toy 미니 DDPM")
    print("=" * 84)

    # --- [1] 스케줄 검산 (lesson §3.3) -------------------------------------
    sch64 = make_schedule(args.T, dtype=torch.float64)
    print(f"\n=== [1] lesson §3.3 표 재현 — T={args.T}, β 선형 1e-4 → 0.02 ===")
    if args.T == 1000:
        rows = check_schedule_table(sch64)
        print_table(["t", "β_t", "ᾱ_t", "√ᾱ_t", "√(1-ᾱ_t)", "SNR", "lesson 표"],
                    rows, ["right"] * 6 + ["center"])
        snr = sch64.snr().numpy()
        t_cross = int(np.argmin(np.abs(snr - 1.0))) + 1
        print(f"  SNR이 1을 지나는 지점: t = {t_cross}   "
              f"(ᾱ={float(sch64.abar[t_cross - 1]):.5f})")
        print("  ← lesson §3.3은 't≈250'이라 씁니다. 표의 t=250 행이 SNR 1.10으로 이미 1 근처이고,"
              f" 정확한 교차점은 {t_cross}입니다.")
        print(f"  t=1000에서 신호 감쇠 배율: 1/√ᾱ = {1 / float(sch64.sqrt_abar[-1]):.1f}배"
              "   (lesson 본문은 '158배'로 반올림)")
        print("  ✅ lesson §3.3 표 5행 전부 일치 (assert 통과)")
    else:
        print(f"  [건너뜀] --T {args.T} 이므로 lesson 표(T=1000)와 대조하지 않습니다.")

    # --- [2] λ_t 검산 (lesson §4.4) ----------------------------------------
    print(f"\n=== [2] lesson §4.4 λ_t 표 재현 — σ_t² = β_t 가정 ===")
    if args.T == 1000:
        rows = check_lambda_table(sch64)
        print_table(["t", "λ_t (계산)", "λ_t (lesson)", "대조"], rows,
                    ["right", "right", "right", "center"])
        lam = sch64.lambda_t()
        print(f"  λ_1/λ_1000 = {float(lam[0] / lam[-1]):.2f}   ← lesson §4.4의 '≈49'")
        print("  ELBO는 작은 t(마지막 디테일 복원)에 약 49배를 걸고 있었습니다.")
        print("  L_simple은 그 가중치를 버리므로 → 큰 t가 상대적으로 승격됩니다.")
        print("  ✅ lesson §4.4 표 6행 + 비율 전부 일치 (assert 통과)")
    else:
        print(f"  [건너뜀] --T {args.T}")

    sch = sch64.to(dev)
    sch = Schedule(sch.T, *(getattr(sch, f).float() for f in
                            ("beta", "alpha", "abar", "abar_prev", "sqrt_abar",
                             "sqrt_one_minus_abar", "beta_tilde")))

    # --- [3] 데이터 ---------------------------------------------------------
    print(f"\n=== [3] 데이터 — {args.data} n={args.n_data} ===")
    raw = make_data(args.data, args.n_data, args.seed)
    xn, mu, sd = normalize(raw)
    print(f"  정규화 전 평균 {raw.mean(0).round(3)}  표준편차 {raw.std(0).round(3)}")
    print(f"  정규화 후 평균 {xn.mean(0).round(3)}  표준편차 {xn.std(0).round(3)}"
          "   ← prior가 N(0,I)이므로 데이터도 같은 스케일에 둡니다")
    x0_all = torch.as_tensor(xn, device=dev)

    if not args.no_plot:
        p = plot_schedule(sch64, out_dir / "01_schedule.png")
        print(f"  [저장] {p}")
        p = plot_forward(torch.as_tensor(xn), sch64.to(torch.device("cpu")),
                         out_dir / "01_forward_diffusion.png", args.seed)
        print(f"  [저장] {p}   ← t를 키우면 데이터가 가우시안으로 무너지는 격자")

    # --- [4] 학습 -----------------------------------------------------------
    model = EpsMLP(2, args.hidden, args.layers, args.temb).to(dev)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"\n=== [4] 학습 — L_simple, {args.steps} 스텝 batch {args.batch} ===")
    print(f"  ε_θ 파라미터 {n_param:,}개 (hidden {args.hidden} × {args.layers}층, "
          f"시간 임베딩 {args.temb}차원)")
    log_every = max(1, args.steps // 120)
    t0 = time.perf_counter()
    ema, losses, bucket = train(model, x0_all, sch, steps=args.steps, batch=args.batch,
                                lr=args.lr, ema_decay=args.ema, log_every=log_every,
                                seed=args.seed)
    train_sec = time.perf_counter() - t0
    print(f"  학습 {train_sec:.1f}s   최종 손실 {losses[-1]:.4f} (초기 {losses[0]:.4f})")
    ema.copy_to(model)  # 샘플링은 EMA 가중치로
    model.eval()
    print("  t 구간별 평균 손실: "
          + "  ".join(f"{i * 100}-{(i + 1) * 100}: {v:.3f}" for i, v in enumerate(bucket)))
    print("  ← 작은 t가 더 어렵습니다(거의 깨끗한 x_t에서 ε을 맞히는 문제). "
          "ELBO가 λ_t로 그쪽을 밀어주던 이유입니다.")

    # --- [5] 샘플링 ---------------------------------------------------------
    print(f"\n=== [5] ancestral 샘플링 — NFE = T = {args.T} ===")
    t0 = time.perf_counter()
    fake = ancestral_sample(model, args.n_sample, sch, dev, seed=args.seed + 1).cpu().numpy()
    sample_sec = time.perf_counter() - t0
    print(f"  {args.n_sample}개 생성 {sample_sec:.1f}s  "
          f"({sample_sec / args.T * 1000:.3f} ms/NFE, 배치 전체 기준)")
    print(f"  생성 샘플 평균 {fake.mean(0).round(3)}  표준편차 {fake.std(0).round(3)}"
          f"   (데이터: {xn.mean(0).round(3)} / {xn.std(0).round(3)})")

    if not args.no_plot:
        p = plot_loss(losses, bucket, log_every, out_dir / "01_loss.png")
        print(f"  [저장] {p}")
        p = plot_samples(xn[:args.n_sample], fake, out_dir / "01_samples.png", args.T)
        print(f"  [저장] {p}   ← (c)에서 두 구름이 겹치면 학습이 된 것입니다")

    # --- [6] 체크포인트 -----------------------------------------------------
    ck = out_dir / ("01_model_smoke.pt" if args.smoke else "01_model.pt")
    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "cfg": dict(data_dim=2, hidden=args.hidden, n_layers=args.layers, temb_dim=args.temb),
        "schedule": dict(T=args.T, beta_start=1e-4, beta_end=0.02),
        "data": dict(kind=args.data, mu=mu, sd=sd, seed=args.seed),
        "data_ref": xn[:8192],  # 02가 품질 지표의 기준 분포로 씁니다
        "smoke": args.smoke,
        "train_steps": args.steps,
    }, ck)
    print(f"\n  [저장] {ck}  ({ck.stat().st_size / 1e6:.2f} MB)"
          "   ← 02_samplers_compare.py 가 이 파일을 읽습니다")

    print(f"\n총 소요 {time.perf_counter() - t_start:.1f}s")
    print("다음: python 02_samplers_compare.py   (같은 모델로 ancestral vs DDIM을 NFE 축에서 가릅니다)")


if __name__ == "__main__":
    import sys

    # 노트북(ipykernel)에서는 argparse가 jupyter의 -f 인자를 먹지 않도록 빈 리스트를 넘긴다.
    # → 전부 기본값으로 실행됩니다. --smoke로 돌리려면 이 셀을 main(["--smoke"])로 고치세요.
    main(None if "ipykernel" not in sys.modules else [])
