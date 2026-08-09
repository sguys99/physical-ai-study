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
# # W1-M5 실습 2 — 코드북 사용률 정면 비교: FSQ vs VQ-VAE (G1 관절 궤적)
#
# **이 모듈의 핵심 실습입니다.** lesson 「실습으로 가기」의 문장 — *"이 모듈의 진짜 산출물은 노트북 하나가 아니라
# '코드북 사용률 곡선을 내 눈으로 봤다'는 상태"* — 를 만드는 스크립트입니다.
#
# 같은 인코더·디코더에 **양자화기만 갈아끼워** 학습합니다. 공정 비교가 이 실습의 생명입니다.
#
# ```
# x ∈ R^[16, 29] ─ Enc ─ z ─┬─ FSQ: round(f(z))          ─ Dec ─ x̂
#                            └─ VQ : argmin_j ||z − C_j|| ─ Dec ─ x̂
#                               ↑ 여기만 다르다 (lesson §6 블록도)
# ```
#
# 측정할 것:
#
# | 지표 | 정의 | 왜 |
# |---|---|---|
# | **코드북 사용률** | 검증셋을 인코딩할 때 **최소 1회 이상 쓰인** 코드워드의 비율 (원논문 정의 그대로) | 이 모듈의 대표 숫자 |
# | perplexity | $\exp(H)$, $H$ = 코드 사용 분포의 엔트로피 | "고르게 쓰는가"까지 본다 |
# | 재구성 MSE | 정규화 공간 + **라디안·도 환산** | 사용률만 높고 복원이 나쁘면 의미 없다 (lesson 「흔한 오해」 2번) |
# | 파라미터 수 | 전체 / 양자화기 | VQ의 코드북이 FSQ엔 없다 (lesson §5.4) |
#
# 그리고 **`--sweep`으로 코드북 크기 $2^8 / 2^{10} / 2^{12}$** 를 훑어 lesson §5.5의 공정성 주장
# — *"FSQ의 우위는 $2^{10}$을 넘어서면서 나타난다"* — 을 직접 확인합니다(원문 Fig 3의 축소 재현).
#
# 출력(`artifacts/W1-M5/`):
# `02_curves.png` · `02_usage.png` · `02_code_hist.png` · `02_results.csv` ·
# `02_data_samples.png` · `fsq_g1_*.pt`(체크포인트, gitignore)
#
# **GPU 권장**(RTX 3080에서 기본 설정 약 2분, `--sweep` 약 6분). `--smoke`는 1분 이내.
# `--device cpu`로도 돌지만 수 배 느립니다.

# %%
from __future__ import annotations

import os

# ⚠️ `import mujoco` 보다 먼저 (W1-M2 lesson §6.2). 이 스크립트도 G1 포즈를 몇 장 렌더한다.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import csv  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Sequence  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless 고정 — 뷰어 금지

import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402

MODULE_ID = "W1-M5"

# W1-M2 ground-truth §3: G1JoystickFlatTerrain의 정책 주기 ctrl_dt=0.02 → 50 Hz.
# 데이터도 같은 레이트로 만든다(상위 액션 시퀀스와 같은 시간 격자에 두기 위해).
FS_HZ = 50.0
N_JOINTS = 29  # W1-M2 실측 nu=29 (다리 12 + 허리 3 + 팔 14)

# lesson §5.3 = 원논문 Table 1. --sweep이 쓰는 세 행.
TABLE1_SWEEP: list[tuple[int, list[int]]] = [
    (2**8, [8, 6, 5]),
    (2**10, [8, 5, 5, 5]),
    (2**12, [7, 5, 5, 5, 5]),
]


# %% [markdown]
# ## 0. 경로 · 폰트 · 표 유틸 (W1-M2 practice와 동일 규약)
#
# **모델 경로 규약** — W1-M2와 완전히 같은 3단입니다.
#
# | 우선순위 | 방법 |
# |---|---|
# | 1 | `--menagerie /path/to/mujoco_menagerie` |
# | 2 | 환경변수 `MENAGERIE_PATH` |
# | 3 | (기본) 리포 루트 기준 `repos/mujoco_menagerie` |

# %%
_ROOT_MARKERS = ("course", "docs", "CLAUDE.md")


def find_repo_root() -> Path:
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


def resolve_menagerie(arg: str | None = None) -> Path:
    """--menagerie > MENAGERIE_PATH > <repo>/repos/mujoco_menagerie"""
    cand = arg or os.environ.get("MENAGERIE_PATH") or str(find_repo_root() / "repos" / "mujoco_menagerie")
    p = Path(cand).expanduser().resolve()
    if not (p / "unitree_g1").is_dir():
        raise SystemExit(
            f"[에러] mujoco_menagerie를 찾지 못했습니다: {p}\n"
            "  해결: git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git "
            "repos/mujoco_menagerie\n"
            "  또는: --menagerie <경로> / 환경변수 MENAGERIE_PATH 지정"
        )
    return p


_KO_FONT_PREFERENCE = (
    "Pretendard", "NanumGothic", "Nanum Gothic", "Malgun Gothic", "NanumBarunGothic",
    "Noto Sans KR", "Noto Sans CJK KR", "Noto Sans CJK JP", "Source Han Sans KR",
    "AppleGothic", "Spoqa Han Sans Neo", "UnDotum", "Baekmuk Gulim",
)
_PROBE_CHARS = "한글사용률코드"

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


def t(ko: str, en: str) -> str:
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
    widths = [
        max(_dwidth(headers[i]), *(_dwidth(r[i]) for r in rows)) if rows else _dwidth(headers[i])
        for i in range(len(headers))
    ]
    print(" | ".join(_pad(h, w, "center") for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(_pad(c, w, a) for c, w, a in zip(r, widths, aligns)))


# %% [markdown]
# ## 1. 양자화기 — 여기만 다르다
#
# `FSQ`는 `01_fsq_minimal.py`와 **한 글자도 다르지 않습니다**(스크립트를 독립 실행 가능하게 두려고
# 옮겨 적은 것뿐입니다). 두 양자화기가 같은 인터페이스를 갖도록 맞춰둡니다:
#
# ```
# forward(z: [B, d]) -> (zq: [B, d],  aux_loss: scalar,  idx: [B] int64)
# ```
#
# **`aux_loss`가 인터페이스에 들어가 있다는 것 자체가 이 실습의 논지입니다.** FSQ는 항상 0을 돌려주고,
# VQ는 codebook + β·commitment를 돌려줍니다 (lesson §4의 손실 3항).

# %%
class FSQ(nn.Module):
    """Finite Scalar Quantization — arXiv:2309.15505 §3.1 + 부록 A.1. 학습 파라미터 0개."""

    def __init__(self, levels: Sequence[int], eps: float = 1e-3):
        super().__init__()
        lv = torch.tensor(list(levels), dtype=torch.float32)
        basis = torch.cat([torch.ones(1), torch.cumprod(lv[:-1], dim=0)])  # eq.(3)
        self.register_buffer("levels", lv)
        self.register_buffer("basis", basis)
        self.eps = eps
        self.d = len(levels)
        self.codebook_size = int(np.prod(np.asarray(levels, dtype=np.int64)))

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """eq.(1)의 f: z_i ↦ ⌊L_i/2⌋·tanh(z_i) (짝수 L이면 반 칸 비대칭 보정)."""
        half_l = (self.levels - 1) * (1 - self.eps) / 2
        offset = torch.where(self.levels % 2 == 0,
                             torch.full_like(self.levels, 0.5), torch.zeros_like(self.levels))
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    @staticmethod
    def round_ste(x: torch.Tensor) -> torch.Tensor:
        """eq.(2) x + sg(round(x) - x)."""
        return x + (torch.round(x) - x).detach()

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """eq.(1) ẑ = round(f(z)) → [-1, 1] 재정규화."""
        return self.round_ste(self.bound(z)) / (self.levels // 2)

    def _digits(self, zhat: torch.Tensor) -> torch.Tensor:
        hw = self.levels // 2
        return zhat * hw + hw

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """eq.(3) 코드 → 토큰 ID (mixed-radix 전단사)."""
        return torch.round((self._digits(zhat) * self.basis).sum(-1)).long()

    def indices_to_codes(self, idx: torch.Tensor) -> torch.Tensor:
        hw = self.levels // 2
        digits = torch.floor(idx.unsqueeze(-1).float() / self.basis) % self.levels
        return (digits - hw) / hw

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zq = self.quantize(z)
        aux = torch.zeros((), device=z.device, dtype=z.dtype)  # ← 보조 손실이 없다 (lesson §5.4)
        return zq, aux, self.codes_to_indices(zq)


# %%
class VQ(nn.Module):
    """VQ-VAE 양자화기 — arXiv:1711.00937. 학습되는 코드북 `C ∈ R^[K, d]`.

    기본값은 **트릭을 전부 끈 바닐라 구성**입니다. 그것이 FSQ 논문의 논지이기 때문입니다.
    `--vq-init data` / `--vq-ema` 로 켜면 사용률이 어떻게 달라지는지 직접 비교할 수 있습니다.
    (집필 환경 실측으로는 이 장난감 설정에서 네 조합 모두 18~21%대였습니다 — README §5 참조.
     "트릭을 넣으면 해결된다"가 자동으로 성립하지 않는다는 관측입니다.)
    """

    def __init__(self, codebook_size: int, dim: int, beta: float = 0.25,
                 ema: bool = False, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.d = dim
        self.beta = beta
        self.ema = ema
        self.decay = decay
        self.eps = eps
        self.embed = nn.Embedding(codebook_size, dim)
        # 정본 참조구현(VQ-VAE PyTorch 포팅)의 초기화. K가 커지면 코드워드가 원점 근처
        # 반지름 1/K 공에 몰린다 — collapse의 씨앗이 여기서부터 뿌려진다.
        self.embed.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self._data_init_done = True
        if ema:
            self.embed.weight.requires_grad_(False)
            self.register_buffer("ema_count", torch.zeros(codebook_size))
            self.register_buffer("ema_sum", self.embed.weight.data.clone())

    @torch.no_grad()
    def enable_data_init(self) -> None:
        """첫 배치의 인코더 출력으로 코드북을 초기화하도록 예약한다 (`--vq-init data`)."""
        self._data_init_done = False

    @torch.no_grad()
    def _maybe_data_init(self, z: torch.Tensor) -> None:
        if self._data_init_done:
            return
        n = z.shape[0]
        pick = torch.randint(0, n, (self.codebook_size,), device=z.device)
        self.embed.weight.data.copy_(z[pick])
        if self.ema:
            self.ema_sum.copy_(self.embed.weight.data)
            self.ema_count.fill_(1.0)
        self._data_init_done = True

    def forward(self, z_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lead = z_in.shape[:-1]
        z = z_in.reshape(-1, self.d)
        self._maybe_data_init(z)
        C = self.embed.weight  # [K, d]
        # eq.(4) k = argmin_j ||z - C_j||²  — 전개해서 행렬곱으로
        d2 = (z.pow(2).sum(1, keepdim=True) - 2 * z @ C.t() + C.pow(2).sum(1)[None, :])
        idx = d2.argmin(dim=1)
        zq = self.embed(idx)

        if self.ema:
            if self.training:
                self._ema_update(z.detach(), idx)
            aux = self.beta * F.mse_loss(z, zq.detach())  # commitment만 남는다
        else:
            # eq.(5) L = ||x-x̂||² + ||sg[z] - C_k||² + β||z - sg[C_k]||²
            #            (재구성)      (codebook)         (commitment)
            aux = F.mse_loss(zq, z.detach()) + self.beta * F.mse_loss(z, zq.detach())

        zq = z + (zq - z).detach()  # STE — VQ도 FSQ와 똑같이 STE를 쓴다 (lesson §5.4 표)
        return zq.reshape(*lead, self.d), aux, idx.reshape(*lead)

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, idx: torch.Tensor) -> None:
        onehot = F.one_hot(idx, self.codebook_size).to(z.dtype)  # [B, K]
        self.ema_count.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
        self.ema_sum.mul_(self.decay).add_(onehot.t() @ z, alpha=1 - self.decay)
        n = self.ema_count.sum()
        cnt = (self.ema_count + self.eps) / (n + self.codebook_size * self.eps) * n
        self.embed.weight.data.copy_(self.ema_sum / cnt.unsqueeze(1))


# %% [markdown]
# ## 2. 오토인코더 — 인코더·디코더는 양쪽이 **완전히 동일**
#
# ```
# x [B, 16, 29] ─flatten→ [B, 464] ─ Linear(464,512) SiLU Linear(512,512) SiLU ─ Linear(512, d) ─ z [B, d]
#                                                    ↑ 이 몸통이 FSQ/VQ 공통
# z ─ 양자화기 ─ zq [B, d] ─ Linear(d,512) SiLU Linear(512,512) SiLU Linear(512,464) ─ x̂ [B, 16, 29]
# ```
#
# 마지막 투영만 잠재 차원 $d$에 맞춰 크기가 달라집니다. **FSQ는 $d=4$, VQ는 $d=64$**(기본값)이고,
# 이 차이가 lesson §5.4의 "차원의 역전"입니다 — FSQ는 $d<10$, VQ는 훨씬 큰 $d$가 보통.
# 그래서 파라미터 수도 함께 보고합니다.
#
# ### `--tokens-per-chunk` — lesson §7.4의 미확인 항목을 손잡이로
#
# 기본값은 **청크당 토큰 1개**(lesson §6 블록도 그대로)입니다. `--tokens-per-chunk N`을 주면
# 인코더가 $z \in \mathbb{R}^{N \times d}$를 내고 각각을 독립 양자화해 **청크당 토큰 N개**가 됩니다.
# lesson §7.4의 "시간축을 어떻게 다루는가 — 프레임당 1토큰인가, 청크당 N토큰인가"를 **직접 만져보는**
# 손잡이입니다. N을 키우면 복원이 좋아지고 압축률이 떨어집니다. 그 트레이드오프 곡선이
# 곧 토크나이저 설계의 실체입니다.

# %%
class TrajAE(nn.Module):
    """궤적 청크 오토인코더. 양자화기만 갈아끼운다."""

    def __init__(self, in_dim: int, hidden: int, latent_dim: int, quantizer: nn.Module,
                 n_tok: int = 1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim * n_tok),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim * n_tok, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, in_dim),
        )
        self.quantizer = quantizer
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.n_tok = n_tok

    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, T, D] -> 토큰 ID [B, n_tok]. 03에서 쓰는 진입점."""
        z = self.enc(x.reshape(x.shape[0], -1)).reshape(x.shape[0], self.n_tok, self.latent_dim)
        return self.quantizer(z)[2]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x [B, T, D] -> (x̂ [B, T, D], aux_loss, idx [B, n_tok])."""
        b, tt, dd = x.shape
        z = self.enc(x.reshape(b, -1)).reshape(b, self.n_tok, self.latent_dim)
        zq, aux, idx = self.quantizer(z)
        xh = self.dec(zq.reshape(b, -1)).reshape(b, tt, dd)
        return xh, aux, idx

    def param_counts(self) -> dict[str, int]:
        quant = sum(p.numel() for p in self.quantizer.parameters())
        quant += sum(b.numel() for b in self.quantizer.buffers() if b.dtype.is_floating_point) \
            if isinstance(self.quantizer, VQ) and self.quantizer.ema else 0
        total = sum(p.numel() for p in self.parameters())
        return dict(total=total, quantizer=quant, backbone=total - quant)


# %% [markdown]
# ## 3. 데이터 — G1 관절 궤적을 MuJoCo에서 직접 만든다
#
# 별도 다운로드 없이 재현 가능해야 하므로 궤적을 **생성**합니다. 회사가 FSQ로 토큰화하는 대상은
# 이미지가 아니라 **액션**이므로(lesson §7), 데이터도 액션이어야 합니다.
#
# ### 정규화 규약 — 관절별 `jnt_range`를 $[-1, 1]$로
#
# $$
# x_j = 2\,\frac{q_j - \text{lo}_j}{\text{hi}_j - \text{lo}_j} - 1,
# \qquad q_j = \text{lo}_j + \frac{x_j + 1}{2}(\text{hi}_j - \text{lo}_j)
# \tag{6}
# $$
#
# `lo`/`hi`는 G1 `scene.xml`의 `actuator_ctrlrange`입니다. W1-M2 lesson §5.5에서 확인했듯
# `<position ... inheritrange="1"/>` 이므로 `ctrlrange = jnt_range`이고, 따라서 **$x \in [-1,1]$이면
# 항상 관절 한계 안**입니다. 관절마다 range 폭이 다르므로(0.52 rad ~ 5.5 rad) 같은 정규화 오차가
# 라디안으로는 10배까지 차이납니다 — 03에서 관절별로 환산해 볼 지점입니다.
#
# ### 궤적 3종 (다양성 확보용)
#
# | 종류 | 생성 방식 | 무엇을 흉내내나 |
# |---|---|---|
# | `sine` | 3-성분 sin 합, 0.15~1.2 Hz | 주기적 동작 (W1-M2 `03_g1_sin_wave.py`의 확장) |
# | `waypoint` | 랜덤 목표 3~6개를 smoothstep 보간 | 자세 전환 시퀀스 |
# | `reach` | `stand`에서 랜덤 포즈로 갔다 오기 (minimum-jerk) | 도달-복귀 동작 |
#
# 셋을 무작위로 섞습니다.
#
# ### 관절 시너지 — 왜 완전 무작위 29차원이 아닌가
#
# 실제 휴머노이드 모션은 **관절이 서로 강하게 상관돼 저차원 구조**를 갖습니다(모캡 데이터에 PCA를
# 걸면 소수의 성분이 분산 대부분을 설명한다는 것은 잘 알려진 사실입니다). 반면 29개 관절을 서로
# **독립적으로** 흔든 궤적에는 그런 구조가 없어서 **어떤 토크나이저로도 압축되지 않습니다** —
# 토큰 하나(10 bits)로 담을 상관구조가 애초에 없기 때문입니다.
#
# 그래서 궤적을 $n_{syn}$차원 **시너지 계수 공간**에서 만든 뒤 고정된 기저 $B \in \mathbb{R}^{n_{syn} \times 29}$로
# 관절 공간에 올립니다.
#
# $$x(t) = \mathrm{clip}\big(x^{stand} + c(t)\,B,\; -1,\; 1\big), \qquad c(t) \in \mathbb{R}^{n_{syn}} \tag{7}$$
#
# $B$는 **학습셋과 검증셋이 같은 것을 씁니다**(같은 다양체 위의 데이터여야 하므로). `--synergies 0`을
# 주면 $B = I_{29}$가 되어 상관 없는 데이터가 되고, 그때 재구성이 얼마나 무너지는지 직접 볼 수 있습니다.
# **이 대비 자체가 lesson §6의 "무엇을 버려도 되는지를 결정하는 것이 토크나이저 설계"의 실습판입니다.**
#
# > ⚠️ **이 데이터는 텔레옵 시연도 모캡 리타게팅도 아닌 장난감입니다.**
# > 회사 토크나이저의 학습 데이터가 무엇인지는 lesson §7.4대로 미확인입니다.

# %%
@dataclass
class G1Spec:
    lo: np.ndarray          # [29] 관절 하한 [rad]
    hi: np.ndarray          # [29] 관절 상한 [rad]
    x_stand: np.ndarray     # [29] stand 키프레임의 정규화 좌표
    names: list[str] = field(default_factory=list)
    model_path: str = ""


def load_g1_spec(menagerie: Path, variant: str = "scene.xml") -> tuple[mujoco.MjModel, G1Spec]:
    """G1 MJCF에서 관절 한계와 stand 자세를 읽는다 (W1-M2 §5.4~5.5와 같은 방식)."""
    path = menagerie / "unitree_g1" / variant
    if not path.is_file():
        raise SystemExit(f"[에러] 파일이 없습니다: {path}")
    m = mujoco.MjModel.from_xml_path(str(path))
    assert m.nu == N_JOINTS, f"nu={m.nu}, 기대 {N_JOINTS} (W1-M2 실측)"
    lo = m.actuator_ctrlrange[:, 0].copy()
    hi = m.actuator_ctrlrange[:, 1].copy()
    assert np.all(hi > lo), "ctrlrange가 비어 있는 액추에이터가 있습니다"
    q_stand = m.key_qpos[0][7:7 + m.nu].copy() if m.nkey > 0 else np.zeros(m.nu)
    x_stand = normalize(q_stand, lo, hi)  # eq.(6)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}" for i in range(m.nu)]
    return m, G1Spec(lo=lo, hi=hi, x_stand=x_stand, names=names, model_path=str(path))


def normalize(q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """eq.(6) 라디안 → [-1, 1]."""
    return 2.0 * (q - lo) / (hi - lo) - 1.0


def denormalize(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """eq.(6)의 역 — [-1, 1] → 라디안."""
    return lo + (x + 1.0) * 0.5 * (hi - lo)


def _smoothstep(u: np.ndarray) -> np.ndarray:
    return u * u * (3.0 - 2.0 * u)


def _min_jerk(u: np.ndarray) -> np.ndarray:
    return 10 * u**3 - 15 * u**4 + 6 * u**5


SYNERGY_SEED = 12345  # 학습/검증이 **같은 다양체**를 공유해야 하므로 시드를 고정한다


def synergy_basis(n_syn: int, D: int) -> np.ndarray:
    """eq.(7)의 고정 기저 B [n_syn, D]. n_syn=0 이면 항등(= 관절이 서로 독립)."""
    if n_syn <= 0:
        return np.eye(D, dtype=np.float64)
    rng = np.random.default_rng(SYNERGY_SEED)
    B = rng.normal(size=(n_syn, D))
    return B / np.linalg.norm(B, axis=1, keepdims=True)


def make_trajectories(n_traj: int, T_traj: int, spec: G1Spec, seed: int,
                      n_syn: int = 6) -> np.ndarray:
    """정규화 공간 [-1,1]의 궤적 [n_traj, T_traj, 29]를 생성한다 (eq.(7))."""
    rng = np.random.default_rng(seed)
    D = len(spec.x_stand)
    B = synergy_basis(n_syn, D)
    K = B.shape[0]                       # 계수 공간 차원
    tt = np.arange(T_traj) / FS_HZ
    u = np.linspace(0.0, 1.0, T_traj)
    out = np.empty((n_traj, T_traj, D), dtype=np.float32)

    for n in range(n_traj):
        kind = rng.integers(0, 3)
        # 성분 일부만 움직이게 해서 "전부 다 흔들리는" 단조로운 데이터가 되지 않게 한다
        active = rng.random(K) < rng.uniform(0.35, 1.0)
        if not active.any():
            active[rng.integers(0, K)] = True
        c = np.zeros((T_traj, K))

        if kind == 0:  # sine — 3성분 합
            for _ in range(3):
                amp = rng.uniform(0.2, 1.0, size=K) * active
                freq = rng.uniform(0.15, 1.2, size=K)
                pha = rng.uniform(0.0, 2 * np.pi, size=K)
                c += amp * np.sin(2 * np.pi * freq * tt[:, None] + pha)
        elif kind == 1:  # waypoint — smoothstep 보간
            w = int(rng.integers(3, 7))
            key = rng.uniform(-1.0, 1.0, size=(w, K)) * active
            key[0] = 0.0
            seg = np.linspace(0, w - 1, T_traj)
            i0 = np.clip(np.floor(seg).astype(int), 0, w - 2)
            frac = _smoothstep(seg - i0)[:, None]
            c = (1 - frac) * key[i0] + frac * key[i0 + 1]
        else:  # reach & return — minimum-jerk 프로파일로 갔다 오기
            target = rng.uniform(-1.0, 1.0, size=K) * active
            s = _min_jerk(1.0 - np.abs(1 - 2 * u))  # 0 → 1 → 0 을 매끈하게
            c = s[:, None] * target

        delta = c @ B                                     # eq.(7) 계수 → 관절 공간
        peak = np.max(np.abs(delta))
        if peak > 1e-8:                                   # 궤적마다 진폭을 다르게
            delta *= rng.uniform(0.15, 0.9) / peak
        out[n] = np.clip(spec.x_stand + delta, -1.0, 1.0).astype(np.float32)  # 관절 한계 클립
    return out


def chunk_index(n_traj: int, T_traj: int, chunk: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """(traj_id, start) 쌍을 만든다. 청크 자체를 복사하지 않아 메모리를 아낀다."""
    starts = np.arange(0, T_traj - chunk + 1, stride)
    tid = np.repeat(np.arange(n_traj), len(starts))
    st = np.tile(starts, n_traj)
    return tid, st


def gather_chunks(X: torch.Tensor, tid: torch.Tensor, st: torch.Tensor, chunk: int) -> torch.Tensor:
    """X [N, T, D] 에서 청크 [B, chunk, D] 를 뽑는다."""
    pos = st.unsqueeze(1) + torch.arange(chunk, device=X.device).unsqueeze(0)  # [B, chunk]
    return X[tid.unsqueeze(1), pos]


# %% [markdown]
# ## 4. 학습 · 평가
#
# ### 사용률의 정의와 **상한**
#
# 원논문 정의를 그대로 씁니다: *"The fraction of the codewords that are used at least once when
# encoding the validation set."* 청크 하나 → 토큰 하나이므로 **검증셋 샘플 수 $N_{val}$ 이
# $|\mathcal{C}|$ 보다 충분히 크지 않으면 사용률은 구조적으로 100%에 도달할 수 없습니다.**
#
# $$\text{usage} \le \min\!\left(1,\; \frac{N_{val}}{|\mathcal{C}|}\right)$$
#
# 이미지 토크나이저는 이미지 한 장이 토큰 수백 개를 만들어 이 문제가 없습니다. 액션 토크나이저는
# 다릅니다 — **평가 설계에서 먼저 걸리는 함정**이라 스크립트가 이 상한을 매번 찍어줍니다.

# %%
@dataclass
class RunResult:
    tag: str
    method: str                 # "FSQ" | "VQ"
    codebook_size: int
    latent_dim: int
    n_tok: int
    levels: list[int] | None
    usage: float                # 검증셋 사용률 [0,1]
    usage_ceiling: float
    n_used: int
    perplexity: float
    mse_norm: float
    mse_ratio: float            # mse_norm / (평균 예측 베이스라인 MSE)
    rmse_rad_mean: float
    rmse_rad_max: float
    params_total: int
    params_quant: int
    train_sec: float
    hist_epoch: list[int] = field(default_factory=list)
    hist_val_mse: list[float] = field(default_factory=list)
    hist_usage: list[float] = field(default_factory=list)
    counts: np.ndarray | None = None   # [|C|] 검증셋 코드 사용 횟수


@torch.no_grad()
def evaluate(model: TrajAE, Xv: torch.Tensor, tid: torch.Tensor, st: torch.Tensor,
             chunk: int, cb_size: int, batch: int, half_range: torch.Tensor) -> dict:
    model.eval()
    counts = torch.zeros(cb_size, dtype=torch.long, device=Xv.device)
    se_sum = torch.zeros(Xv.shape[-1], device=Xv.device)  # 관절별 제곱오차 합
    n_elem = 0
    for i in range(0, tid.numel(), batch):
        x = gather_chunks(Xv, tid[i:i + batch], st[i:i + batch], chunk)
        xh, _, idx = model(x)
        counts += torch.bincount(idx.reshape(-1), minlength=cb_size)
        se_sum += ((x - xh) ** 2).sum(dim=(0, 1))
        n_elem += x.shape[0] * x.shape[1]
    mse_per_joint = se_sum / n_elem                       # 정규화 공간
    rmse_rad = mse_per_joint.sqrt() * half_range          # eq.(6): 정규화 1 = half_range rad
    p = counts.float() / counts.sum()
    nz = p[p > 0]
    perplexity = float(torch.exp(-(nz * nz.log()).sum()))
    n_used = int((counts > 0).sum())
    model.train()
    return dict(
        counts=counts.cpu().numpy(),
        n_used=n_used,
        usage=n_used / cb_size,
        perplexity=perplexity,
        mse_norm=float(mse_per_joint.mean()),
        rmse_rad=rmse_rad.cpu().numpy(),
    )


@torch.no_grad()
def mean_baseline_mse(Xt: torch.Tensor, Xv: torch.Tensor, chunk: int) -> float:
    """'학습셋 평균 청크를 그냥 내놓는' 무정보 베이스라인. 토큰이 실제로 얼마나 버는지의 기준선."""
    tid, st = (torch.as_tensor(a, device=Xt.device) for a in chunk_index(Xt.shape[0], Xt.shape[1], chunk, chunk))
    mu = gather_chunks(Xt, tid, st, chunk).mean(0, keepdim=True)
    tid, st = (torch.as_tensor(a, device=Xv.device) for a in chunk_index(Xv.shape[0], Xv.shape[1], chunk, chunk))
    return float(((gather_chunks(Xv, tid, st, chunk) - mu) ** 2).mean())


def train_one(
    method: str, Xt: torch.Tensor, Xv: torch.Tensor, spec: G1Spec, args: argparse.Namespace,
    levels: list[int] | None = None, codebook_size: int | None = None, tag: str = "",
    baseline_mse: float = 1.0,
) -> tuple[RunResult, TrajAE]:
    """양자화기 하나를 학습하고 지표를 돌려준다. 인코더·디코더 구성은 method와 무관하게 동일."""
    dev = Xt.device
    torch.manual_seed(args.seed)  # 두 방법이 **같은 초기화**에서 출발하도록 고정
    chunk = args.chunk
    in_dim = chunk * N_JOINTS

    if method == "FSQ":
        assert levels is not None
        quant: nn.Module = FSQ(levels)
        latent_dim, cb = quant.d, quant.codebook_size
    else:
        assert codebook_size is not None
        quant = VQ(codebook_size, args.vq_dim, beta=args.beta, ema=args.vq_ema)
        if args.vq_init == "data":
            quant.enable_data_init()
        latent_dim, cb = args.vq_dim, codebook_size

    n_tok = args.tokens_per_chunk
    model = TrajAE(in_dim, args.hidden, latent_dim, quant, n_tok=n_tok).to(dev)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    tid_t, st_t = (torch.as_tensor(a, device=dev) for a in chunk_index(Xt.shape[0], Xt.shape[1], chunk, args.stride))
    tid_v, st_v = (torch.as_tensor(a, device=dev) for a in chunk_index(Xv.shape[0], Xv.shape[1], chunk, chunk))
    half_range = torch.as_tensor((spec.hi - spec.lo) / 2.0, dtype=torch.float32, device=dev)

    n_chunks = tid_t.numel()
    n_val_tokens = tid_v.numel() * n_tok       # 사용률 상한 = min(1, N_val_tokens/|C|)
    steps_per_epoch = max(1, n_chunks // args.batch)
    res = RunResult(tag=tag or method, method=method, codebook_size=cb, latent_dim=latent_dim,
                    n_tok=n_tok, levels=levels, usage=0.0,
                    usage_ceiling=min(1.0, n_val_tokens / cb), n_used=0,
                    perplexity=0.0, mse_norm=0.0, mse_ratio=0.0,
                    rmse_rad_mean=0.0, rmse_rad_max=0.0,
                    params_total=0, params_quant=0, train_sec=0.0)

    print(f"\n  ── {res.tag}: |C|={cb:,}  d={latent_dim}  청크당 토큰={n_tok}  "
          f"train chunks={n_chunks:,}  val 토큰={n_val_tokens:,}  "
          f"사용률 상한={res.usage_ceiling * 100:.1f}%")

    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(n_chunks, device=dev)
        run_rec = run_aux = 0.0
        for s in range(steps_per_epoch):
            sel = perm[s * args.batch:(s + 1) * args.batch]
            x = gather_chunks(Xt, tid_t[sel], st_t[sel], chunk)
            xh, aux, _ = model(x)
            rec = F.mse_loss(xh, x)          # eq.(5)의 재구성 항 — FSQ는 이것뿐이다
            loss = rec + aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_rec += float(rec.detach())
            run_aux += float(aux.detach())
        if ep % args.eval_every == 0 or ep == args.epochs:
            ev = evaluate(model, Xv, tid_v, st_v, chunk, cb, args.eval_batch, half_range)
            res.hist_epoch.append(ep)
            res.hist_val_mse.append(ev["mse_norm"])
            res.hist_usage.append(ev["usage"])
            print(f"     ep {ep:3d}/{args.epochs}  train_rec={run_rec / steps_per_epoch:.5f}  "
                  f"aux={run_aux / steps_per_epoch:.5f}  val_mse={ev['mse_norm']:.5f}  "
                  f"(베이스라인의 {ev['mse_norm'] / baseline_mse * 100:4.1f}%)  "
                  f"usage={ev['usage'] * 100:5.1f}%  ppl={ev['perplexity']:.1f}")
    res.train_sec = time.perf_counter() - t0

    ev = evaluate(model, Xv, tid_v, st_v, chunk, cb, args.eval_batch, half_range)
    pc = model.param_counts()
    res.counts = ev["counts"]
    res.n_used = ev["n_used"]
    res.usage = ev["usage"]
    res.perplexity = ev["perplexity"]
    res.mse_norm = ev["mse_norm"]
    res.mse_ratio = ev["mse_norm"] / baseline_mse
    res.rmse_rad_mean = float(np.mean(ev["rmse_rad"]))
    res.rmse_rad_max = float(np.max(ev["rmse_rad"]))
    res.params_total = pc["total"]
    res.params_quant = pc["quantizer"]
    return res, model


# %% [markdown]
# ## 5. 그림
#
# - `02_curves.png` — 학습 곡선. **왼쪽은 복원, 오른쪽은 사용률.** 둘을 같이 봐야 합니다
#   (lesson 「흔한 오해」 2번: 사용률은 필요조건이지 충분조건이 아님).
# - `02_code_hist.png` — **코드 사용 히스토그램.** 내림차순으로 정렬한 사용 횟수입니다.
#   곡선이 바닥(0)에 닿는 지점부터 오른쪽이 전부 **죽은 코드**입니다. VQ에서 이 구간이 보이면
#   그것이 lesson §4.1의 codebook collapse를 눈으로 본 것입니다.
# - `02_usage.png` — 코드북 크기 스윕. 원문 Fig 3의 축소 재현.

# %%
def plot_curves(results: list[RunResult], out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for r in results:
        style = "-" if r.method == "FSQ" else "--"
        color = "#1f77b4" if r.method == "FSQ" else "#d62728"
        axes[0].plot(r.hist_epoch, r.hist_val_mse, style, color=color, marker="o", ms=3, label=r.tag)
        axes[1].plot(r.hist_epoch, [u * 100 for u in r.hist_usage], style, color=color,
                     marker="o", ms=3, label=r.tag)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel(t("검증 재구성 MSE (정규화 공간)", "val recon MSE (normalized)"))
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[0].set_title(t("(a) 복원 — 낮을수록 좋다", "(a) reconstruction (lower better)"))

    axes[1].axhline(100, color="#999999", lw=0.8, ls=":")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel(t("코드북 사용률 [%]", "codebook usage [%]"))
    axes[1].set_ylim(-3, 105)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    axes[1].set_title(t("(b) 코드북 사용률 — 이 모듈의 대표 지표",
                        "(b) codebook usage (the headline metric)"))
    fig.suptitle(t("같은 인코더·디코더, 양자화기만 교체", "same encoder/decoder, quantizer swapped"))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_code_hist(results: list[RunResult], out: Path) -> Path:
    groups = [("FSQ", "#1f77b4"), ("VQ", "#d62728")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, (meth, color) in zip(axes, groups):
        sub = sorted([r for r in results if r.method == meth], key=lambda r: r.codebook_size)
        for k, r in enumerate(sub):
            c = np.sort(r.counts)[::-1].astype(float)
            dead = float((c == 0).mean())
            alpha = 0.45 + 0.55 * (k + 1) / max(1, len(sub))
            ax.plot(np.arange(len(c)) / len(c) * 100, np.maximum(c, 0.4),
                    color=color, alpha=alpha, lw=1.8,
                    label=f"|C|={r.codebook_size:,}  " + t("죽은 코드", "dead") + f" {dead * 100:.1f}%")
        ax.set_yscale("log")
        ax.set_xlabel(t("코드워드 순위 [%] (사용 횟수 내림차순)", "codeword rank [%] (desc by count)"))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title(t(f"{meth} — 검증셋 코드 사용 횟수", f"{meth} — val code usage counts"))
    axes[0].set_ylabel(t("사용 횟수 (0은 바닥선에 눕혀 표시)", "count (0 shown at floor)"))
    fig.suptitle(t("바닥선에 눕은 구간이 '죽은 코드'다 (lesson §4.1 codebook collapse)",
                   "the floor segment is dead codes (codebook collapse)"))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_sweep(results: list[RunResult], out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    for meth, color, mk in (("FSQ", "#1f77b4", "o"), ("VQ", "#d62728", "s")):
        sub = sorted([r for r in results if r.method == meth], key=lambda r: r.codebook_size)
        if not sub:
            continue
        xs = [r.codebook_size for r in sub]
        axes[0].plot(xs, [r.usage * 100 for r in sub], marker=mk, color=color, label=meth)
        axes[1].plot(xs, [r.mse_norm for r in sub], marker=mk, color=color, label=meth)
        axes[2].plot(xs, [r.params_quant / 1e3 for r in sub], marker=mk, color=color, label=meth)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel(t("코드북 크기 |C|", "codebook size |C|"))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    axes[0].axhline(100, color="#999999", lw=0.8, ls=":")
    axes[0].set_ylim(-3, 105)
    axes[0].set_ylabel(t("사용률 [%]", "usage [%]"))
    axes[0].set_title(t("(a) 코드북 사용률 — 원문 Fig 3의 축소 재현",
                        "(a) codebook usage (mini repro of Fig 3)"))
    axes[1].set_yscale("log")
    axes[1].set_ylabel(t("검증 재구성 MSE", "val recon MSE"))
    axes[1].set_title(t("(b) 복원 — 작은 코드북에서는 VQ가 유리할 수 있다",
                        "(b) reconstruction"))
    axes[2].set_ylim(bottom=0)
    axes[2].set_ylabel(t("양자화기 파라미터 [천 개]", "quantizer parameters [k]"))
    axes[2].set_title(t("(c) FSQ는 0개 — 코드북이 없다", "(c) FSQ has none"))
    fig.suptitle(t("코드북 크기 스윕 — lesson §5.5의 공정성 주장 확인",
                   "codebook size sweep"))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def render_data_samples(m: mujoco.MjModel, X: np.ndarray, spec: G1Spec, out: Path,
                        n: int = 6, height: int = 320, width: int = 400) -> Path:
    """생성한 궤적이 진짜 G1 포즈인지 눈으로 확인 (물리 없이 mj_forward + 오프스크린 렌더)."""
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 2.4, -8.0, 135.0
    cam.lookat[:] = [0.0, 0.0, 0.75]
    rng = np.random.default_rng(0)
    tiles = []
    try:
        for _ in range(n):
            i, k = int(rng.integers(0, X.shape[0])), int(rng.integers(0, X.shape[1]))
            q = denormalize(X[i, k].astype(np.float64), spec.lo, spec.hi)  # eq.(6)
            mujoco.mj_resetDataKeyframe(m, d, 0) if m.nkey > 0 else mujoco.mj_resetData(m, d)
            d.qpos[7:7 + m.nu] = q
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=cam)
            tiles.append(renderer.render().copy())
    finally:
        renderer.close()
    grid = np.concatenate([np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:6], axis=1)], axis=0)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.imshow(grid)
    ax.axis("off")
    ax.set_title(t("생성한 궤적에서 뽑은 G1 포즈 6장 (mj_forward, 물리 없음)",
                   "6 poses sampled from the generated trajectories (mj_forward)"))
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# %% [markdown]
# ## 6. 실행

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W1-M5 실습 2: FSQ vs VQ 코드북 사용률 비교 (lesson §4·§5)")
    p.add_argument("--menagerie", default=None, help="mujoco_menagerie 경로")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--chunk", type=int, default=16, help="청크 길이 T (기본 16 = lesson §6 블록도)")
    p.add_argument("--stride", type=int, default=8, help="학습 청크 stride (기본 8, 검증은 항상 chunk)")
    p.add_argument("--traj-len", type=int, default=200, help="궤적 1개 길이 [프레임] (기본 200 = 4초 @50Hz)")
    p.add_argument("--n-train-traj", type=int, default=3000)
    p.add_argument("--n-val-traj", type=int, default=2400,
                   help="검증 궤적 수. 사용률 상한 = N_val_tokens/|C| 이므로 넉넉해야 합니다")
    p.add_argument("--synergies", type=int, default=6,
                   help="관절 시너지 차원 n_syn (eq.(7)). 0이면 29관절 독립 = 압축 불가능한 데이터")
    p.add_argument("--tokens-per-chunk", type=int, default=1,
                   help="청크당 토큰 수 (기본 1 = lesson §6 블록도). 키우면 복원↑ 압축률↓")
    p.add_argument("--levels", default="8,5,5,5", help="FSQ 레벨 (기본 Table 1의 2^10 행)")
    p.add_argument("--codebook", type=int, default=1024, help="VQ 코드북 크기 (기본 1024 = 2^10)")
    p.add_argument("--vq-dim", type=int, default=64, help="VQ 잠재 차원 d (기본 64)")
    p.add_argument("--beta", type=float, default=0.25, help="commitment 계수 β (VQ-VAE 계열 관행값)")
    p.add_argument("--vq-ema", action="store_true", help="코드북 EMA 업데이트를 켠다 (트릭 ①)")
    p.add_argument("--vq-init", default="uniform", choices=["uniform", "data"],
                   help="코드북 초기화. uniform=정본 참조구현, data=첫 배치 인코더 출력 (트릭 ②)")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--eval-batch", type=int, default=4096)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sweep", action="store_true",
                   help="코드북 크기 2^8/2^10/2^12를 훑는다 (원문 Fig 3 축소 재현)")
    p.add_argument("--no-render", action="store_true", help="데이터 샘플 렌더를 건너뛴다")
    p.add_argument("--smoke", action="store_true", help="수 분 내 완주하는 축소 경로 (경로 확인용)")
    p.add_argument("--ascii-labels", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


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
        print(f"[device] CPU  torch {torch.__version__}  (GPU보다 수 배 느립니다)")
    return dev


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.smoke:
        # 결과의 정확도가 아니라 **경로**를 확인하는 모드. 사용률 숫자는 참고용입니다.
        args.n_train_traj, args.n_val_traj = 400, 600
        args.epochs, args.eval_every = 6, 2
        args.sweep = False
        print("[smoke] 축소 경로로 실행합니다 (궤적 400/600, 6 epoch). "
              "사용률 숫자는 학습이 덜 된 값이라 결론에 쓰지 마세요.")
    setup_korean_font(args.ascii_labels)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = pick_device(args.device)
    out_dir = artifacts_dir()

    print("=" * 82)
    print(f"  {MODULE_ID} 실습 2 — FSQ vs VQ 코드북 사용률 (G1 관절 궤적 {N_JOINTS}차원)")
    print("=" * 82)

    # --- [1] 데이터 ---------------------------------------------------------
    menagerie = resolve_menagerie(args.menagerie)
    m, spec = load_g1_spec(menagerie)
    print(f"\n=== [1] 데이터 — G1 궤적 생성 ===")
    print(f"  모델: {spec.model_path}  (nu={m.nu})")
    print(f"  관절 range 폭: 최소 {np.min(spec.hi - spec.lo):.3f} rad "
          f"({spec.names[int(np.argmin(spec.hi - spec.lo))]}) / "
          f"최대 {np.max(spec.hi - spec.lo):.3f} rad "
          f"({spec.names[int(np.argmax(spec.hi - spec.lo))]})  ← 같은 정규화 오차라도 "
          f"라디안으로는 {np.max(spec.hi - spec.lo) / np.min(spec.hi - spec.lo):.1f}배 차이")
    t0 = time.perf_counter()
    Xtr = make_trajectories(args.n_train_traj, args.traj_len, spec, seed=args.seed, n_syn=args.synergies)
    Xva = make_trajectories(args.n_val_traj, args.traj_len, spec, seed=args.seed + 10_000, n_syn=args.synergies)
    gen_sec = time.perf_counter() - t0
    print(f"  train {Xtr.shape}  val {Xva.shape}   생성 {gen_sec:.1f}s  "
          f"({Xtr.nbytes / 1e6:.0f} MB + {Xva.nbytes / 1e6:.0f} MB)")
    print(f"  시너지 차원 n_syn={args.synergies}"
          + ("  (0 = 29관절 독립 — 압축할 상관구조가 없는 대조군)" if args.synergies <= 0
             else f"  → 포즈가 {args.synergies}차원 다양체 위에 있다 (eq.(7))"))
    assert np.all(np.abs(Xtr) <= 1.0 + 1e-6), "정규화 공간을 벗어난 샘플이 있습니다"
    Xt = torch.as_tensor(Xtr, device=dev)
    Xv = torch.as_tensor(Xva, device=dev)
    base_mse = mean_baseline_mse(Xt, Xv, args.chunk)
    print(f"  무정보 베이스라인(학습셋 평균 청크) val MSE = {base_mse:.5f}"
          "   ← 토큰이 이보다 얼마나 낮추는지가 '토큰이 버는 정보량'입니다")

    if not args.no_render:
        p = render_data_samples(m, Xva, spec, out_dir / "02_data_samples.png")
        print(f"  [저장] {p}   ← 생성 데이터가 진짜 G1 포즈인지 눈으로 확인")

    # --- [2] 실행할 구성 -----------------------------------------------------
    levels = [int(v) for v in args.levels.split(",")]
    configs: list[dict] = [
        dict(method="FSQ", levels=levels, tag=f"FSQ L={levels}"),
        dict(method="VQ", codebook_size=args.codebook, tag=f"VQ |C|={args.codebook}"),
    ]
    if args.sweep:
        for _, lv in TABLE1_SWEEP:
            if lv != levels:
                configs.append(dict(method="FSQ", levels=lv, tag=f"FSQ L={lv}"))
        for k in (2**8, 2**10, 2**12):
            if k != args.codebook:
                configs.append(dict(method="VQ", codebook_size=k, tag=f"VQ |C|={k}"))

    print(f"\n=== [2] 학습 — 구성 {len(configs)}개 "
          f"(chunk={args.chunk}, hidden={args.hidden}, epochs={args.epochs}, batch={args.batch}) ===")
    print(f"  VQ 설정: d={args.vq_dim}, β={args.beta}, init={args.vq_init}, EMA={args.vq_ema}")
    if args.vq_init == "uniform" and not args.vq_ema:
        print("  → **트릭 없는 바닐라 VQ**입니다(정본 참조구현 그대로). `--vq-init data` · `--vq-ema` 를")
        print("     켜면 사용률이 어떻게 달라지는지 직접 비교해 보세요 — README §5의 실측표와 대조할 수 있습니다.")

    results: list[RunResult] = []
    models: dict[str, TrajAE] = {}
    for cfg in configs:
        r, model = train_one(cfg["method"], Xt, Xv, spec, args,
                             levels=cfg.get("levels"), codebook_size=cfg.get("codebook_size"),
                             tag=cfg["tag"], baseline_mse=base_mse)
        results.append(r)
        models[r.tag] = model

    # --- [3] 결과 표 ---------------------------------------------------------
    print("\n=== [3] 결과 ===")
    rows = []
    for r in results:
        rows.append([
            r.tag, f"{r.codebook_size:,}", f"{r.latent_dim}",
            f"{r.n_used:,}", f"{r.usage * 100:.1f}%", f"{r.usage_ceiling * 100:.0f}%",
            f"{r.perplexity:,.0f}", f"{r.mse_norm:.5f}", f"{r.mse_ratio * 100:.0f}%",
            f"{r.rmse_rad_mean:.4f}", f"{np.degrees(r.rmse_rad_mean):.2f}",
            f"{r.params_quant:,}", f"{r.params_total:,}", f"{r.train_sec:.0f}s",
        ])
    print_table(
        ["구성", "|C|", "d", "쓰인 코드", "사용률", "상한", "ppl", "val MSE", "vs 기준선",
         "RMSE[rad]", "RMSE[deg]", "양자화기 param", "전체 param", "학습"],
        rows, ["left"] + ["right"] * 13,
    )

    csv_path = out_dir / "02_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "method", "codebook_size", "latent_dim", "tokens_per_chunk", "levels",
                    "n_used", "usage", "usage_ceiling", "perplexity", "val_mse_norm",
                    "baseline_mse_norm", "mse_ratio_vs_baseline",
                    "rmse_rad_mean", "rmse_deg_mean", "rmse_rad_max",
                    "params_quantizer", "params_total", "train_sec"])
        for r in results:
            w.writerow([r.tag, r.method, r.codebook_size, r.latent_dim, r.n_tok,
                        "|".join(map(str, r.levels)) if r.levels else "",
                        r.n_used, f"{r.usage:.4f}", f"{r.usage_ceiling:.4f}",
                        f"{r.perplexity:.2f}", f"{r.mse_norm:.6f}", f"{base_mse:.6f}",
                        f"{r.mse_ratio:.4f}",
                        f"{r.rmse_rad_mean:.5f}", f"{np.degrees(r.rmse_rad_mean):.4f}",
                        f"{r.rmse_rad_max:.5f}", r.params_quant, r.params_total,
                        f"{r.train_sec:.1f}"])
    print(f"  [저장] {csv_path}")

    # --- [4] 그림 -----------------------------------------------------------
    print("\n=== [4] 그림 저장 ===")
    print(f"  [저장] {plot_curves(results[:2], out_dir / '02_curves.png')}")
    print(f"  [저장] {plot_code_hist(results, out_dir / '02_code_hist.png')}")
    if args.sweep:
        print(f"  [저장] {plot_sweep(results, out_dir / '02_usage.png')}")
    else:
        print("  [skip] --sweep 없이는 02_usage.png를 만들지 않습니다 (구성이 2개뿐)")

    # --- [5] 체크포인트 (03이 읽는다) ----------------------------------------
    # 스모크·변형 실행이 본 학습 체크포인트를 덮어쓰지 않도록 파일명에 구성을 박는다
    suffix = "_smoke" if args.smoke else ""
    if args.tokens_per_chunk != 1:
        suffix += f"_n{args.tokens_per_chunk}"
    if args.synergies != 6:
        suffix += f"_syn{args.synergies}"
    ck = out_dir / f"fsq_g1_L{'-'.join(map(str, levels))}_T{args.chunk}{suffix}.pt"
    fsq_res = next(r for r in results if r.method == "FSQ" and r.levels == levels)
    torch.save(dict(
        state_dict=models[fsq_res.tag].state_dict(),
        levels=levels, chunk=args.chunk, hidden=args.hidden, n_joints=N_JOINTS,
        n_tok=args.tokens_per_chunk, synergies=args.synergies,
        lo=spec.lo, hi=spec.hi, names=spec.names,
        usage=fsq_res.usage, mse_norm=fsq_res.mse_norm, fs_hz=FS_HZ,
    ), ck)
    print(f"  [저장] {ck}   (gitignore 대상 — 03_g1_action_tokenizer.py가 읽습니다)")

    # --- [6] 해석 가이드 ------------------------------------------------------
    fsq_all = [r for r in results if r.method == "FSQ"]
    vq_all = [r for r in results if r.method == "VQ"]
    print("\n" + "=" * 82)
    print("  읽는 법")
    print("=" * 82)
    if fsq_all and vq_all:
        f10 = min(fsq_all, key=lambda r: abs(r.codebook_size - 1024))
        v10 = min(vq_all, key=lambda r: abs(r.codebook_size - 1024))
        print(f"  · |C|≈2^10 에서 사용률: FSQ {f10.usage * 100:.1f}%  vs  VQ {v10.usage * 100:.1f}%")
        print(f"    (원논문 MaskGIT ImageNet 256 실험은 FSQ 100% vs VQ 81% — lesson §5.5)")
        print(f"  · 양자화기 파라미터: FSQ {f10.params_quant:,}개  vs  VQ {v10.params_quant:,}개")
    if args.sweep:
        print("  · 02_usage.png (a)에서 |C|가 커질 때 두 곡선이 어떻게 갈리는지 보세요.")
        print("    lesson §5.5: FSQ의 우위는 2^10을 **넘어서면서** 나타나고, 그 이하에서는 VQ가 낫습니다.")
    print("  · 02_code_hist.png 오른쪽(VQ)에서 곡선이 바닥에 눕는 구간 = 죽은 코드 = collapse.")
    print("  · 사용률만 보고 결론내지 마세요 — val MSE를 같이 봐야 합니다 (lesson 「흔한 오해」 2번).")
    print("  · 다음 → 03_g1_action_tokenizer.py 에서 이 FSQ 모델로 궤적을 토큰 열로 바꿉니다.")
    print("=" * 82)


# %%
if __name__ == "__main__":
    import sys

    main(None if "ipykernel" not in sys.modules else [])
