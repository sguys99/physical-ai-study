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
# # W1-M5 실습 1 — FSQ를 40줄로: `ẑ = round(f(z))`
#
# 마스터플랜이 **"FSQ는 수십 줄이면 구현된다"** 고 못박은 지점을 그대로 확인하는 스크립트입니다.
# lesson.md `§5.1`(수식) · `§5.3`(Table 1) · `§6`(ASCII 블록도)의 실행판입니다.
#
# 확인할 것:
#
# 1. **바운딩 함수** $f: z_i \mapsto \lfloor L_i/2 \rfloor \tanh(z_i)$ 와 **짝수 $L$ 보정**
#    $$\hat z = \mathrm{round}(f(z)) \tag{1}$$
# 2. **STE** — `round_ste: x ↦ x + sg(round(x) - x)`. 순전파는 반올림, 역전파는 항등
#    $$\texttt{round\_ste}(x) = x + \mathrm{sg}\big(\mathrm{round}(x) - x\big) \tag{2}$$
# 3. **전단사 열거(mixed-radix)** — 이것이 "LLM 토큰으로 그대로 쓸 수 있다"의 실체
#    $$\mathrm{idx} = \sum_{i=1}^{d} \hat z^{\,\text{digit}}_i \cdot b_i,
#      \qquad b = [1,\; L_1,\; L_1 L_2,\; \dots] \tag{3}$$
# 4. **자체 검증 assert 5종** — 왕복 · 인덱스 범위 · 유일성 · 그래디언트 · 채널별 레벨 수
# 5. **Table 1의 5개 구성** $|\mathcal{C}| = \prod_i L_i$ 를 직접 계산해 240 / 1000 / 4375 / 15360 / 64000 확인
# 6. **격자 시각화** — $d=2, L=[5,5]$ 의 25개 코드와 $\mathbb{R}^2$ 분할, $d=3, L=3$ 이면 27개(원문 Fig 1)
#
# 출력:
# - `artifacts/W1-M5/01_fsq_grid.png` — 격자·분할·열거 순서 3-panel
# - `artifacts/W1-M5/01_fsq_bound_round.png` — $f(z)$ 와 `round_ste(f(z))`, 그리고 그래디언트 (원문 Fig 2 오른쪽)
#
# **GPU 불필요. CPU에서 수 초.** 인자 없이 그냥 돌리면 됩니다.

# %%
from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless 고정 — 뷰어를 띄우지 않는다

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402

MODULE_ID = "W1-M5"

# lesson §5.3 = 원논문 Table 1. 이 표는 하드코딩하고 곱은 코드가 계산한다.
TABLE1: list[tuple[int, list[int]]] = [
    (2**8, [8, 6, 5]),
    (2**10, [8, 5, 5, 5]),
    (2**12, [7, 5, 5, 5, 5]),
    (2**14, [8, 8, 8, 6, 5]),
    (2**16, [8, 8, 8, 5, 5, 5]),
]
# ground-truth §1.2 검산값 — 여기가 어긋나면 구현이 아니라 표가 틀린 것이다.
TABLE1_EXPECTED_SIZES = [240, 1000, 4375, 15360, 64000]


# %% [markdown]
# ## 0. 경로 · 폰트 · 표 유틸 (W1-M2 practice와 동일 규약)

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
_PROBE_CHARS = "한글격자토큰"

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
    """한글 폭을 고려한 간단한 표 출력."""
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
# ## 1. FSQ — 이게 전부입니다
#
# 아래 클래스가 **FSQ의 전량**입니다. 학습되는 텐서가 하나도 없다는 점을 먼저 확인하세요
# (`register_buffer`만 있고 `nn.Parameter`가 없습니다). lesson §5.4 표의 마지막 행
# "Parameters: 코드북 → **없음**"이 코드 수준에서 이렇게 나타납니다.
#
# ### 짝수 $L$ 보정 — 왜 필요한가
#
# 양자화는 **정수로** 반올림합니다. $L$이 홀수면 $\lfloor L/2 \rfloor$ 폭의 대칭 구간
# $[-\lfloor L/2\rfloor, \lfloor L/2\rfloor]$ 안에 정수가 정확히 $L$개 들어갑니다
# ($L=5$ → $\{-2,-1,0,1,2\}$). **짝수면 그렇지 않습니다** — 대칭 구간에 정수를 담으면
# 개수가 홀수가 되어버립니다. 그래서 원논문은 격자를 반 칸 밀어 비대칭으로 만듭니다
# ($L=4$ → $\{-2,-1,0,1\}$). 그 반 칸이 아래 `offset`이고, `shift`는 $z=0$이 격자 중앙 근처로
# 오도록 되돌리는 보정항입니다.
#
# > 📌 `shift = tan(offset/half_l)`은 **원논문 부록 A.1 의사코드 그대로**입니다.
# > $z=0 \mapsto 0$ 을 정확히 만족시키는 값은 $\mathrm{arctanh}(\texttt{offset}/\texttt{half\_l})$ 이고,
# > 두 함수는 3차항까지 일치합니다($\tan x \approx x + x^3/3 \approx \mathrm{arctanh}\,x$).
# > 게다가 $L=2$면 `offset/half_l > 1`이 되어 `arctanh`가 발산합니다. 논문이 `tan`을 쓴 이유로
# > 보이며, 실습도 논문 쪽을 따릅니다.

# %%
class FSQ(nn.Module):
    """Finite Scalar Quantization — arXiv:2309.15505 §3.1 + 부록 A.1.

    학습 파라미터 0개. 격자는 하이퍼파라미터 `levels`로 고정돼 있습니다.
    """

    def __init__(self, levels: Sequence[int], eps: float = 1e-3):
        super().__init__()
        lv = torch.tensor(list(levels), dtype=torch.float32)
        basis = torch.cat([torch.ones(1), torch.cumprod(lv[:-1], dim=0)])  # eq.(3) b = [1, L1, L1L2, ...]
        self.register_buffer("levels", lv)          # L = [L_1, ..., L_d]  (학습 대상 아님)
        self.register_buffer("basis", basis)        # mixed-radix 자릿수 가중치
        self.eps = eps
        self.d = len(levels)
        self.codebook_size = int(np.prod(np.asarray(levels, dtype=np.int64)))  # |C| = prod L_i

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """eq.(1)의 f. tanh로 유계화 + 격자 폭으로 스케일. 짝수 L이면 반 칸 비대칭 보정."""
        half_l = (self.levels - 1) * (1 - self.eps) / 2               # ≈ ⌊L/2⌋ (홀수 L에서 정확히)
        offset = torch.where(self.levels % 2 == 0,
                             torch.full_like(self.levels, 0.5),
                             torch.zeros_like(self.levels))           # 짝수 L의 반 칸
        shift = torch.tan(offset / half_l)                            # 원문 부록 A.1 그대로
        return torch.tanh(z + shift) * half_l - offset

    @staticmethod
    def round_ste(x: torch.Tensor) -> torch.Tensor:
        """eq.(2) x + sg(round(x) - x). 순전파는 round(x), 역전파는 항등(그래디언트 1)."""
        return x + (torch.round(x) - x).detach()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """eq.(1) ẑ = round(f(z)) 를 [-1, 1]로 재정규화해 반환 (디코더가 먹기 좋게)."""
        q = self.round_ste(self.bound(z))                             # eq.(1)+(2)
        return q / (self.levels // 2)                                 # 정규화된 코드 zhat

    def _digits(self, zhat: torch.Tensor) -> torch.Tensor:
        """정규화 코드 [-1,1] → 자릿수 {0, ..., L_i-1}."""
        hw = self.levels // 2
        return zhat * hw + hw

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """eq.(3) 코드 → 토큰 ID (mixed-radix 전단사)."""
        return torch.round((self._digits(zhat) * self.basis).sum(-1)).long()

    def indices_to_codes(self, idx: torch.Tensor) -> torch.Tensor:
        """eq.(3)의 역 — 토큰 ID → 코드. 이 왕복이 성립해야 LLM 토큰으로 쓸 수 있다."""
        hw = self.levels // 2
        digits = torch.floor(idx.unsqueeze(-1).float() / self.basis) % self.levels
        return (digits - hw) / hw

    @property
    def codebook(self) -> torch.Tensor:
        """implied codebook 전체 [|C|, d]. 저장된 텐서가 아니라 **그때그때 열거**한 것."""
        return self.indices_to_codes(torch.arange(self.codebook_size, device=self.levels.device))

    def extra_repr(self) -> str:
        return f"levels={[int(v) for v in self.levels]}, d={self.d}, |C|={self.codebook_size}"


# %% [markdown]
# ## 2. 자체 검증 — assert 5종
#
# 구현이 맞는지 남에게 묻지 않고 코드가 스스로 증명하게 합니다.
#
# | # | 검증 | 왜 중요한가 |
# |---|---|---|
# | 1 | `indices_to_codes(codes_to_indices(ẑ)) == ẑ` | **왕복**이 깨지면 토큰이 액션을 잃는다 |
# | 2 | `0 ≤ idx < ∏L_i` | 상위 LM의 vocabulary 크기가 곧 이 범위다 |
# | 3 | implied codebook의 원소가 전부 **유일** | 전단사(bijection)의 정의 |
# | 4 | `z.grad`가 유한하고 0이 아님 | **STE**가 없으면 인코더로 신호가 안 흐른다 |
# | 5 | 채널 $i$가 정확히 $L_i$개 값을 취함 | 격자가 설계대로 만들어졌는가 |

# %%
def verify_fsq(levels: Sequence[int], n: int = 4096, seed: int = 0, verbose: bool = True) -> dict:
    """FSQ 구현 자체 검증. 실패하면 AssertionError로 즉시 멈춘다."""
    torch.manual_seed(seed)
    q = FSQ(levels)
    lv = [int(v) for v in levels]

    # --- [4] STE 그래디언트: z에 requires_grad를 걸고 흘려본다 ---
    z = (torch.randn(n, q.d) * 2.0).requires_grad_(True)
    zhat = q(z)
    zhat.sum().backward()
    assert z.grad is not None, "STE가 끊겼습니다 — z.grad가 None"
    assert torch.isfinite(z.grad).all(), "그래디언트에 NaN/Inf"
    assert (z.grad.abs() > 0).any(), "그래디언트가 전부 0 — round의 미분이 그대로 흘렀습니다"

    # 해석해와 대조: d(zhat)/dz = f'(z) / ⌊L/2⌋ = half_l·sech²(z+shift) / (L//2)
    half_l = (q.levels - 1) * (1 - q.eps) / 2
    offset = torch.where(q.levels % 2 == 0, torch.full_like(q.levels, 0.5), torch.zeros_like(q.levels))
    shift = torch.tan(offset / half_l)
    grad_ref = half_l * (1 - torch.tanh(z.detach() + shift) ** 2) / (q.levels // 2)
    ste_err = float((z.grad - grad_ref).abs().max())
    assert ste_err < 1e-5, f"STE 그래디언트가 해석해와 다릅니다 (max err {ste_err})"

    with torch.no_grad():
        zhat = q(z.detach())
        idx = q.codes_to_indices(zhat)

        # --- [2] 인덱스 범위 ---
        assert int(idx.min()) >= 0 and int(idx.max()) < q.codebook_size, \
            f"토큰 ID가 [0, {q.codebook_size}) 밖입니다"

        # --- [1] 왕복 ---
        back = q.indices_to_codes(idx)
        rt_err = float((back - zhat).abs().max())
        assert rt_err < 1e-5, f"왕복 실패 (max err {rt_err})"

        # --- [3] implied codebook 유일성 ---
        cb = q.codebook
        assert cb.shape == (q.codebook_size, q.d)
        uniq = torch.unique(cb, dim=0)
        assert uniq.shape[0] == q.codebook_size, "코드워드가 중복됩니다 — 전단사 아님"
        assert torch.equal(q.codes_to_indices(cb), torch.arange(q.codebook_size)), \
            "열거 순서가 인덱스와 일치하지 않습니다"

        # --- [5] 채널별 레벨 수 ---
        n_lv = [int(torch.unique(zhat[:, i]).numel()) for i in range(q.d)]
        assert n_lv == lv, f"채널별 레벨 수가 {n_lv}, 기대는 {lv}"

    if verbose:
        print(f"  [OK] L={lv}  d={q.d}  |C|={q.codebook_size:,}  "
              f"왕복오차={rt_err:.2e}  STE오차={ste_err:.2e}  채널별 레벨={n_lv}")
    return dict(levels=lv, d=q.d, size=q.codebook_size, rt_err=rt_err, ste_err=ste_err)


# %% [markdown]
# ## 3. Table 1 검산 — $|\mathcal{C}| = \prod_i L_i$
#
# lesson §5.3의 핵심: **곱이 target과 정확히 같지 않습니다.** 코드북 크기가 2의 거듭제곱이어야 할
# 이유가 없기 때문입니다. 아래 표의 "비율" 열이 1.000이 아닌 것을 눈으로 확인하세요.
#
# 그리고 `L_i ≥ 5` 휴리스틱(원문 §3.2)도 자동으로 점검합니다. Table 1의 다섯 구성이
# 큰 코드북에 도달하는 방식이 **레벨을 키우는 것이 아니라 채널을 늘리는 것**이라는 점도 봐두세요
# ($2^{16}$ 행이 `[8,8,8,5,5,5]`로 $d=6$입니다 — lesson 「흔한 오해」 4번).

# %%
def check_table1() -> None:
    print("\n=== [2] 원논문 Table 1 검산 (lesson §5.3) ===")
    rows = []
    for (target, lv), expect in zip(TABLE1, TABLE1_EXPECTED_SIZES):
        size = int(np.prod(np.asarray(lv, dtype=np.int64)))
        assert size == expect, f"|C| 계산이 ground-truth와 다릅니다: {size} != {expect}"
        assert all(v >= 5 for v in lv), f"L_i >= 5 휴리스틱 위반: {lv}"
        rows.append([
            f"2^{int(np.log2(target))}", str(lv), f"{len(lv)}", f"{size:,}",
            f"{size / target:.3f}", f"{np.log2(size):.2f}",
        ])
    print_table(
        ["target |C|", "L (Table 1)", "d", "prod L_i", "prod/target", "bits"],
        rows, ["center", "left", "right", "right", "right", "right"],
    )
    print("  → 240 / 1,000 / 4,375 / 15,360 / 64,000. 논문 표현은 \"approximately match\"입니다.")
    print("  → 채널 수 d가 3 → 6으로 늘어날 뿐, L_i는 5~8에 머뭅니다 (L_i >= 5 휴리스틱).")


# %% [markdown]
# ## 4. 원문 Fig 1 예시 — $d=3$, $L=3$ 이면 코드가 27개
#
# 논문이 첫 페이지에 든 예시입니다: $\mathcal{C} = \{(-1,-1,-1), (-1,-1,0), \dots, (1,1,1)\}$, $|\mathcal{C}| = L^d = 27$.
#
# > 📌 **열거 *순서*는 논문 서술과 다릅니다.** 우리 `basis = [1, 3, 9]`는 **첫 채널이 가장 빨리 도는**
# > mixed-radix이고, 논문 본문의 나열은 마지막 채널이 빨리 도는 순서로 적혀 있습니다.
# > 집합 $\mathcal{C}$ 자체는 같고, 전단사이기만 하면 어느 순서든 무방합니다 —
# > **단, 상위 모델과 하위 디코더가 같은 순서를 써야 합니다.** 인터페이스 계약의 일부입니다.

# %%
def demo_fig1() -> None:
    print("\n=== [3] 원문 Fig 1 예시: d=3, L=3 ===")
    q = FSQ([3, 3, 3])
    cb = q.codebook
    assert q.codebook_size == 27 == 3**3
    lex = cb[np.lexsort([cb[:, i].numpy() for i in range(q.d - 1, -1, -1)])]

    def fmt(row) -> str:
        return "(" + ",".join(f"{int(v):>2d}" for v in row) + ")"

    print(f"  |C| = L^d = 3^3 = {q.codebook_size}")
    print(f"  사전순 정렬: {', '.join(fmt(r) for r in lex[:3])}, ... , {fmt(lex[-1])}")
    print(f"  basis(우리 열거 순서) = {[int(b) for b in q.basis]}  ← 첫 채널이 가장 빨리 돈다")
    print(f"  idx 0..3 의 코드     = {[[int(v) for v in c] for c in cb[:4]]}")


# %% [markdown]
# ## 5. 격자 시각화 — $d=2$, $L=[5,5]$
#
# 3-panel:
#
# - **(a)** $\mathbb{R}^2$의 $z$ 를 토큰 ID로 색칠. `tanh` 때문에 **바깥 셀이 무한히 넓습니다** —
#   포화 구간의 모든 $z$가 같은 토큰으로 갑니다. 이것이 유계화의 대가입니다.
# - **(b)** $f(z)$ 로 옮긴 뒤 어디로 반올림되는지 화살표. 격자점은 25개이고 **고정**돼 있습니다.
# - **(c)** 코드 격자에 토큰 ID를 적어둔 것 — eq.(3) mixed-radix 열거 순서가 눈에 보입니다.
#
# VQ 그림과 비교해 보세요(lesson §6 아래쪽 블록도). VQ면 (c)의 점들이 **학습으로 움직이고**,
# 안 뽑힌 점은 그 자리에 얼어붙습니다. FSQ의 (c)는 처음부터 끝까지 이 모양 그대로입니다.

# %%
def plot_grid_2d(levels: Sequence[int], out: Path, res: int = 401, lim: float = 3.0) -> Path:
    q = FSQ(levels)
    assert q.d == 2, "이 그림은 d=2 전용입니다"

    g = torch.linspace(-lim, lim, res)
    zz = torch.stack(torch.meshgrid(g, g, indexing="ij"), dim=-1).reshape(-1, 2)
    with torch.no_grad():
        idx = q.codes_to_indices(q(zz)).reshape(res, res).numpy()
        fz = q.bound(zz)
        cb = q.codebook  # [25, 2] 정규화 코드
        cb_int = (q._digits(cb) - q.levels // 2).numpy()  # 정수 격자 좌표
        cb_idx = q.codes_to_indices(cb).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    ax = axes[0]
    ax.imshow(idx.T, origin="lower", extent=(-lim, lim, -lim, lim), cmap="tab20", interpolation="nearest")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title(t("(a) z-공간 분할 — 색 = 토큰 ID\n바깥 셀이 넓다 (tanh 포화)",
                   "(a) partition of z-space (color = token id)\nouter cells are wide (tanh saturation)"))

    ax = axes[1]
    sel = torch.randperm(zz.shape[0])[:220]
    with torch.no_grad():
        src = fz[sel].numpy()
        dst = torch.round(fz[sel]).numpy()
    ax.quiver(src[:, 0], src[:, 1], dst[:, 0] - src[:, 0], dst[:, 1] - src[:, 1],
              angles="xy", scale_units="xy", scale=1, width=0.004, color="#888888", alpha=0.8)
    ax.scatter(src[:, 0], src[:, 1], s=6, c="#1f77b4", label="$f(z)$")
    ax.scatter(cb_int[:, 0], cb_int[:, 1], s=90, marker="s", facecolors="none",
               edgecolors="#d62728", linewidths=1.6, label=t("격자점 (고정)", "grid points (fixed)"))
    ax.set_xlabel("$f(z)_1$")
    ax.set_ylabel("$f(z)_2$")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(t("(b) round(f(z)) — 어디로 스냅되는가", "(b) round(f(z)): where samples snap"))
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.scatter(cb_int[:, 0], cb_int[:, 1], s=60, c="#d62728")
    for (a, b), i in zip(cb_int, cb_idx):
        ax.annotate(str(int(i)), (a, b), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel(t("채널 1 (레벨 $L_1$=%d)" % levels[0], "channel 1 ($L_1$=%d)" % levels[0]))
    ax.set_ylabel(t("채널 2 (레벨 $L_2$=%d)" % levels[1], "channel 2 ($L_2$=%d)" % levels[1]))
    ax.set_title(t(f"(c) implied codebook — |C| = {q.codebook_size}\n숫자 = eq.(3) 토큰 ID",
                   f"(c) implied codebook |C| = {q.codebook_size}\nnumbers = eq.(3) token id"))
    ax.grid(alpha=0.25)

    fig.suptitle(t(f"FSQ 격자  L={list(levels)}  (학습되는 파라미터 0개)",
                   f"FSQ lattice  L={list(levels)}  (zero learned parameters)"))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# %% [markdown]
# ## 6. 원문 Fig 2 오른쪽 — 단일 채널 $L=5$
#
# 왼쪽 패널이 논문 Fig 2 오른쪽 그림입니다. $f(z)$는 매끈하고, `round_ste(f(z))`는 계단입니다.
#
# **오른쪽 패널이 이 실습에서 더 중요합니다.** `round`의 진짜 미분은 거의 모든 곳에서 **0**이고
# (빨간 바닥선), 그래서 학습 신호가 인코더에 도달하지 못합니다. STE는 그 자리에 $f'(z)$를
# 대신 꽂아 넣습니다(파란 곡선). 제어로 옮기면 **비선형 요소(포화·백래시)를 루프 해석을 위해
# 이득 1로 국소 선형화**하는 것과 같은 트릭입니다 (lesson §4).

# %%
def plot_bound_round(level: int, out: Path, lim: float = 4.0, n: int = 1201) -> Path:
    q = FSQ([level])
    z = torch.linspace(-lim, lim, n).unsqueeze(-1).requires_grad_(True)
    fz = q.bound(z)
    qz = q.round_ste(fz)
    qz.sum().backward()
    g_ste = z.grad.squeeze(-1).numpy().copy()

    zz = z.detach().squeeze(-1).numpy()
    fzz = fz.detach().squeeze(-1).numpy()
    qzz = qz.detach().squeeze(-1).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    ax.plot(zz, fzz, lw=2, label="$f(z) = \\lfloor L/2 \\rfloor \\tanh(z)$")
    ax.step(zz, qzz, where="mid", lw=2, color="#d62728", label="$\\mathrm{round}(f(z))$")
    for lv in np.unique(np.round(qzz)):
        ax.axhline(lv, color="#bbbbbb", lw=0.7, ls=":")
    ax.set_xlabel("$z$")
    ax.set_ylabel("$\\hat z$")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_title(t(f"(a) L={level} 단일 채널 — 원문 Fig 2 오른쪽",
                   f"(a) single channel, L={level} (paper Fig 2 right)"))

    ax = axes[1]
    ax.plot(zz, g_ste, lw=2, label=t("STE 그래디언트 $= f'(z)$", "STE gradient $= f'(z)$"))
    ax.plot(zz, np.zeros_like(zz), lw=2, color="#d62728",
            label=t("round의 실제 미분 (거의 모든 곳 0)", "true d(round)/dz (0 a.e.)"))
    ax.set_xlabel("$z$")
    ax.set_ylabel("$d\\hat z / dz$")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_title(t("(b) STE가 없으면 학습 신호가 끊긴다",
                   "(b) without STE the learning signal dies"))

    fig.suptitle(t("바운딩 + 반올림 + STE", "bounding + rounding + STE"))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# %% [markdown]
# ## 7. 실행

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W1-M5 실습 1: FSQ 최소 구현 + 자체 검증 (lesson §5)")
    p.add_argument("--levels", default="8,5,5,5",
                   help="추가로 검증할 레벨 구성 (기본 8,5,5,5 = Table 1의 2^10 행)")
    p.add_argument("--samples", type=int, default=4096, help="검증에 쓸 랜덤 z 개수 (기본 4096)")
    p.add_argument("--no-plot", action="store_true", help="그림 저장을 건너뛴다")
    p.add_argument("--ascii-labels", action="store_true", help="한글 폰트가 있어도 영문 라벨로 렌더")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_korean_font(args.ascii_labels)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = artifacts_dir()

    print("=" * 78)
    print(f"  {MODULE_ID} 실습 1 — FSQ 최소 구현  (torch {torch.__version__}, CPU로 충분)")
    print("=" * 78)

    # --- [1] 자체 검증 -------------------------------------------------------
    print("\n=== [1] 자체 검증 — 왕복 · 인덱스 범위 · 유일성 · STE · 채널별 레벨 ===")
    to_check: list[list[int]] = [lv for _, lv in TABLE1]
    extra = [int(v) for v in args.levels.split(",")]
    if extra not in to_check:
        to_check.append(extra)
    to_check += [[3, 3, 3], [4, 4], [8, 8]]  # 짝수 L 경로도 반드시 밟는다
    for lv in to_check:
        verify_fsq(lv, n=args.samples, seed=args.seed)
    print("  → 5종 assert 전부 통과. 짝수 L([4,4] · [8,8])에서도 격자가 정확히 L개입니다.")

    # --- [2] Table 1 --------------------------------------------------------
    check_table1()

    # --- [3] Fig 1 예시 -----------------------------------------------------
    demo_fig1()

    # --- [4] 파라미터 수 -----------------------------------------------------
    print("\n=== [4] FSQ의 학습 파라미터 수 (lesson §5.4 · 퀴즈 8번) ===")
    q = FSQ(extra)
    n_param = sum(p.numel() for p in q.parameters())
    n_buf = sum(b.numel() for b in q.buffers())
    print(f"  FSQ({q.extra_repr()})")
    print(f"    nn.Parameter : {n_param}개   ← 코드북이 없다는 것의 코드 수준 증거")
    print(f"    buffer       : {n_buf}개 (levels {q.d} + basis {q.d}, 학습 대상 아님)")
    print(f"  같은 |C|={q.codebook_size:,}를 VQ로 하면 코드북 = |C| x d_vq 파라미터가 필요합니다.")
    print("  (lesson 퀴즈 8번: |C|=2^12, d=512 → 2,097,152 ≈ 2M)")

    # --- [5] 그림 -----------------------------------------------------------
    if args.no_plot:
        print("\n[skip] --no-plot 지정 → 그림 없음")
    else:
        print("\n=== [5] 그림 저장 ===")
        p1 = plot_grid_2d([5, 5], out_dir / "01_fsq_grid.png")
        print(f"  [저장] {p1}")
        p2 = plot_bound_round(5, out_dir / "01_fsq_bound_round.png")
        print(f"  [저장] {p2}")

    print("\n" + "=" * 78)
    print("  요점: 위 FSQ 클래스에 nn.Parameter가 0개다. 코드북이 없으니 죽을 코드도 없다.")
    print("        다음 → 02_fsq_vs_vq_g1.py 에서 이것을 VQ와 정면 비교합니다.")
    print("=" * 78)


# %%
if __name__ == "__main__":
    import sys

    main(None if "ipykernel" not in sys.modules else [])
