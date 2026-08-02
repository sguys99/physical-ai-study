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
# # W1-M3 실습 3 — DiT 블록을 손으로 구현하고 액션 청크에 붙인다
#
# lesson.md `§6.3`(블록도) · `§7.2`(액션 청크판)의 실행판입니다. **W1 체크포인트 1번 문항**
# ("DiT 블록 다이어그램을 그리고 conditioning이 들어가는 위치를 설명하라")의 코드 대응이고,
# 백지 워크시트는 [`dit_block_worksheet.excalidraw`](dit_block_worksheet.excalidraw)입니다.
#
# 확인할 것:
#
# 1. **§6.3 블록도를 그대로** — `LayerNorm(elementwise_affine=False)` → $\odot(1+\gamma_1)\oplus\beta_1$
#    → MHSA → $\odot\alpha_1$ → residual, 그리고 같은 패턴의 MLP 가지.
#    조건 경로는 `SiLU → Linear(d → 6d)`이고 **이 Linear를 0으로 초기화**합니다.
# 2. **필수 검증 ①** — $\alpha=0$ 초기화에서 `block(h, c)`가 `h`와 **정확히 같은지** `assert`.
#    lesson §6.3 마지막 줄("$\alpha_1=\alpha_2=0 \Rightarrow$ 블록이 항등함수")을 코드가 증명합니다.
# 3. **필수 검증 ②** — 조건 MLP를 랜덤 초기화로 바꾸면 항등이 깨지는 것도 함께 봅니다(대조군).
# 4. **액션 청크판**(§7.2) — `[B,32,29]` → 시간축 patchify $p_t=2$ → `[B,16,d]` → N블록 →
#    unpatchify → `[B,32,29]`. shape이 끝까지 통하는지 `assert`.
# 5. **$D=29$의 근거** — `mujoco_menagerie`의 G1을 로드해 `model.nu`를 직접 읽습니다(W1-M2 실측값).
# 6. **작은 학습 데모** — G1 관절 한계(`jnt_range`) 안의 sin파 궤적에 이 헤드를 조건부 디노이징으로
#    짧게 학습시켜, 노이즈에서 그럴듯한 궤적이 나오는지 봅니다.
# 7. **파라미터 수와 1회 forward 시간(ms)** — lesson §8.3의 "3 ms/NFE 가정"이 어느 규모에서
#    타당한지 **자기 기기로** 확인하는 것이 요점입니다.
#
# 출력(`artifacts/W1-M3/`): `03_action_chunk.png` · `03_forward_bench.csv`
#
# > ⚠️ **여기 나오는 헤드 구조는 lesson §6.3·§7.2의 교육용 재현이지 회사 구현이 아닙니다.**
# > 회사 L4에 액션 헤드가 있는지, DiT 계열인지, 조건을 어떻게 주입하는지는 전부 **미확인**입니다
# > (lesson §8.4 ①②⑤ · §10의 팀 질문 1·3번).
#
# **GPU 불필요.** `--smoke`는 1분 이내.

# %%
from __future__ import annotations

import os

# ⚠️ `import mujoco` 보다 먼저 (W1-M2 lesson §6.2). 이 스크립트는 렌더는 안 하지만 규약을 지킵니다.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import csv  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless 고정 — 뷰어를 띄우지 않는다

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

MODULE_ID = "W1-M3"

# lesson §7.1·§7.2의 액션 청크 텐서 규격
HORIZON = 32       # H — 청크 길이 (W1-M1 §4 부등식)
PATCH_T = 2        # p_t — 시간축 patchify
N_JOINTS_FALLBACK = 29  # D — W1-M2 실측 nu=29 (menagerie가 없을 때의 폴백)
FS_HZ = 50.0       # 액션 소비 주파수 (W1-M2 ground-truth: ctrl_dt=0.02)


# %% [markdown]
# ## 0. 01·02의 코드 재사용 + 경로 규약
#
# `Schedule` · `timestep_embedding` · DDIM 샘플러는 앞 두 스크립트의 것을 그대로 씁니다.
#
# **모델 경로 규약** — W1-M2·M5와 완전히 같은 3단입니다. 다만 이 스크립트는 menagerie가 없어도
# **에러로 죽지 않고** $D=29$ 상수로 진행합니다(DiT 블록 검증에는 로봇 모델이 필요 없기 때문).
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


def here() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def artifacts_dir() -> Path:
    out = find_repo_root() / "artifacts" / MODULE_ID
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_sibling(fname: str, modname: str):
    for cand in (here(), find_repo_root() / "course" / "w1-generative-core"
                 / "03-diffusion-ddpm-dit" / "practice"):
        p = cand / fname
        if p.is_file():
            spec = importlib.util.spec_from_file_location(modname, p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[modname] = mod
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit(f"[에러] {fname} 를 찾지 못했습니다 (같은 폴더에 있어야 합니다)")


M2 = load_sibling("02_samplers_compare.py", "samplers_compare")
M1 = M2.M1
Schedule, make_schedule = M1.Schedule, M1.make_schedule
timestep_embedding, q_sample = M1.timestep_embedding, M1.q_sample
pick_device, EMA = M1.pick_device, M1.EMA
make_grid = M2.make_grid

USE_KOREAN = False


def setup_korean_font(force_ascii: bool = False) -> bool:
    global USE_KOREAN
    USE_KOREAN = M1.setup_korean_font(force_ascii)
    return USE_KOREAN


def lab(ko: str, en: str) -> str:
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
# ## 1. DiT 블록 — lesson §6.3을 그대로
#
# ```
#  c ∈ R^[B, d] ── SiLU → Linear(d → 6d)  ★ 0으로 초기화
#                        │ chunk 6
#             γ₁ β₁ α₁ γ₂ β₂ α₂   각각 [B, 1, d]  ← T축으로 broadcast
#
#  h [B,T,d] ──┬─ LN(affine 없음) → ⊙(1+γ₁) ⊕ β₁ → MHSA → ⊙α₁ ─⊕→
#              └───────────────────────────────────────────────┘
#              ┬─ LN(affine 없음) → ⊙(1+γ₂) ⊕ β₂ → MLP(4d) → ⊙α₂ ─⊕→ [B,T,d]
#              └───────────────────────────────────────────────┘
# ```
#
# **LLM 대응**: adaLN은 FiLM 계열의 조건부 정규화입니다. 본체 가중치는 그대로 두고 정규화 계수만
# 조건에 따라 바꾸는 개입이라, 조건을 프롬프트 토큰으로 붙이는 in-context 방식(§6.2 FID 35.24)과
# 달리 시퀀스 전체에 균일하게 작용합니다.
#
# **제어 대응**: $\alpha$를 0에서 출발시키는 것은 **소프트 스타트**입니다. 다단 캐스케이드를 전 이득으로
# 한꺼번에 닫으면 초기 과도응답이 발산할 수 있어 안쪽부터 순차적으로 닫는데, adaLN-Zero는 그 순서를
# 옵티마이저에 위임합니다.

# %%
def modulate(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """lesson §6.3:  ⊙(1+γ) ⊕ β.  x [B,T,d], gamma/beta [B,1,d]"""
    return x * (1.0 + gamma) + beta


class DiTBlock(nn.Module):
    """lesson §6.3 블록도 1:1 구현.

    zero_init=True 가 adaLN-**Zero**, False 가 그냥 adaLN(§6.2 표의 3행 vs 4행).
    cross_dim을 주면 §7.2 블록도의 cross-attention 가지가 추가로 붙습니다
    (조건 게이트는 §6.3의 6d 경로를 건드리지 않도록 **별도 3d 경로**로 뒀습니다).
    """

    def __init__(self, d: int, n_heads: int, mlp_ratio: float = 4.0,
                 zero_init: bool = True, cross_dim: int | None = None):
        super().__init__()
        self.d = d
        # LayerNorm의 학습 스케일·시프트를 제거한다 — 그 자리를 γ, β가 대신하기 때문 (§6.2 adaLN 행)
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        hidden = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden, d))
        # lesson §6.3 조건 경로: SiLU → Linear(d → 6d)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))

        self.cross = None
        if cross_dim is not None:
            self.norm_x = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
            self.cross = nn.MultiheadAttention(d, n_heads, batch_first=True, kdim=cross_dim,
                                               vdim=cross_dim)
            self.ada_x = nn.Sequential(nn.SiLU(), nn.Linear(d, 3 * d))

        self.zero_init = zero_init
        self.reset_conditioning(zero_init)

    def reset_conditioning(self, zero_init: bool) -> None:
        """조건 MLP만 다시 초기화한다. 검증 ②(대조군)에서 이 스위치를 씁니다."""
        self.zero_init = zero_init
        lin = self.ada[1]
        if zero_init:
            # ★ adaLN-Zero의 Zero. α가 0으로 나오므로 잔차 가지가 닫히고 블록 = 항등함수
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        else:
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.normal_(lin.bias, std=0.02)
        if self.cross is not None:
            lx = self.ada_x[1]
            if zero_init:
                nn.init.zeros_(lx.weight), nn.init.zeros_(lx.bias)
            else:
                nn.init.normal_(lx.weight, std=0.02), nn.init.normal_(lx.bias, std=0.02)

    def forward(self, h: torch.Tensor, c: torch.Tensor,
                obs: torch.Tensor | None = None) -> torch.Tensor:
        """h [B,T,d], c [B,d], obs [B,N_obs,cross_dim] → [B,T,d]"""
        # chunk 6 → 각각 [B,d] → [B,1,d]로 펴서 T축 broadcast (§6.3)
        g1, b1, a1, g2, b2, a2 = (v.unsqueeze(1) for v in self.ada(c).chunk(6, dim=-1))

        x = modulate(self.norm1(h), g1, b1)
        x, _ = self.attn(x, x, x, need_weights=False)
        h = h + a1 * x                      # ⊙α₁ 뒤 residual — α₁=0이면 이 가지가 닫힌다

        if self.cross is not None and obs is not None:
            g3, b3, a3 = (v.unsqueeze(1) for v in self.ada_x(c).chunk(3, dim=-1))
            q = modulate(self.norm_x(h), g3, b3)
            x, _ = self.cross(q, obs, obs, need_weights=False)
            h = h + a3 * x

        x = modulate(self.norm2(h), g2, b2)
        h = h + a2 * self.mlp(x)            # ⊙α₂ 뒤 residual
        return h


# %% [markdown]
# ### 1.1 필수 검증 ① — $\alpha=0$이면 블록이 항등함수
#
# DiT 원문 §3: *"we initialize the MLP to output the zero-vector for all $\alpha$;
# this initializes the full DiT block as the identity function."*
# 이 문장을 `assert`로 만듭니다. **정확히** 같아야 합니다 — `0 × (무엇이든) = 0`이므로
# `h_out = h_in + 0 + 0 = h_in`이고 부동소수점 오차조차 없습니다.

# %%
def verify_identity(d: int, n_heads: int, B: int, T: int, device: torch.device,
                    seed: int = 0) -> dict:
    torch.manual_seed(seed)
    h = torch.randn(B, T, d, device=device)
    c = torch.randn(B, d, device=device)

    blk = DiTBlock(d, n_heads, zero_init=True).to(device).eval()
    with torch.no_grad():
        out0 = blk(h, c)
    max_dev0 = float((out0 - h).abs().max())
    exact = bool(torch.equal(out0, h))
    assert torch.allclose(out0, h, atol=0.0, rtol=0.0), \
        f"adaLN-Zero 초기화인데 블록이 항등이 아닙니다 (max |out-h| = {max_dev0:g})"

    # 검증 ② — 조건 MLP를 랜덤 초기화로 바꾸면 항등이 깨진다 (대조군)
    blk.reset_conditioning(zero_init=False)
    with torch.no_grad():
        out1 = blk(h, c)
    max_dev1 = float((out1 - h).abs().max())
    assert max_dev1 > 1e-4, \
        f"랜덤 초기화인데도 블록이 항등입니다 (max |out-h| = {max_dev1:g}) — 대조군이 성립 안 함"

    # α만 0이고 γ, β는 살아 있으면? → 여전히 항등 (α가 잔차 게이트이므로)
    blk.reset_conditioning(zero_init=False)
    with torch.no_grad():
        w, b = blk.ada[1].weight, blk.ada[1].bias
        w[2 * d:3 * d].zero_(), b[2 * d:3 * d].zero_()   # α₁ = 0
        w[5 * d:6 * d].zero_(), b[5 * d:6 * d].zero_()   # α₂ = 0
        out2 = blk(h, c)
    max_dev2 = float((out2 - h).abs().max())
    assert torch.allclose(out2, h, atol=0.0, rtol=0.0), \
        "α만 0으로 만들었는데 항등이 아닙니다 — α가 잔차 게이트가 아니라는 뜻입니다"

    return dict(exact=exact, zero=max_dev0, random=max_dev1, alpha_only=max_dev2)


# %% [markdown]
# ## 2. 액션 헤드 — lesson §7.2
#
# 블록 **내부는 그대로 두고** 입력·조건·출력 경로만 바꿉니다. 이것이 §7.1 표의 요점입니다.
#
# ```
#  a_t [B, H=32, D=29] ─ reshape ─→ [B, 16, p_t·D=58] ─ Linear ─→ [B, 16, d]
#                                                        + 1D 시간 위치 인코딩
#  c = Emb(diffusion step t) + Emb(state q_0)  ∈ R^[B, d]
#  → N × DiTBlock → LN → ⊙(1+γ) ⊕ β → Linear(d → p_t·D) → reshape → [B, 32, 29]
# ```
#
# `Linear`가 하는 일이 §6.1의 patchify와 같습니다. 이미지는 2D 공간 패치, 액션은 **시간축 패치**라
# 토큰 수가 $H/p_t$로 줄어듭니다(§7.1 표: 256 → 16).

# %%
class ActionDiT(nn.Module):
    """lesson §7.2 블록도. ε_θ(a_t, t, state) → ε̂ [B, H, D]"""

    def __init__(self, horizon: int = HORIZON, action_dim: int = 29, patch: int = PATCH_T,
                 d: int = 128, depth: int = 4, n_heads: int = 4, state_dim: int | None = None,
                 cross_dim: int | None = None, temb_dim: int = 128):
        super().__init__()
        assert horizon % patch == 0, f"H={horizon}가 p_t={patch}로 나누어떨어져야 합니다"
        self.horizon, self.action_dim, self.patch, self.d = horizon, action_dim, patch, d
        self.n_tokens = horizon // patch
        self.temb_dim = temb_dim

        # 시간축 patchify: [B, H, D] → [B, H/p, p·D] → Linear → [B, H/p, d]   (lesson §7.2)
        self.patch_embed = nn.Linear(patch * action_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, d))
        nn.init.normal_(self.pos, std=0.02)

        # 조건 c — diffusion 스텝 t (+ 선택적으로 proprioception state)
        self.t_embed = nn.Sequential(nn.Linear(temb_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.s_embed = nn.Linear(state_dim, d) if state_dim else None

        self.blocks = nn.ModuleList(DiTBlock(d, n_heads, zero_init=True, cross_dim=cross_dim)
                                    for _ in range(depth))
        # 최종 레이어도 adaLN + zero-init (DiT 원문의 final layer)
        self.norm_f = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.head = nn.Linear(d, patch * action_dim)
        nn.init.zeros_(self.ada_f[1].weight), nn.init.zeros_(self.ada_f[1].bias)
        nn.init.zeros_(self.head.weight), nn.init.zeros_(self.head.bias)

    def forward(self, a: torch.Tensor, t: torch.Tensor,
                state: torch.Tensor | None = None,
                obs: torch.Tensor | None = None) -> torch.Tensor:
        B, H, D = a.shape
        assert (H, D) == (self.horizon, self.action_dim), \
            f"입력 {tuple(a.shape)}가 규격 [B,{self.horizon},{self.action_dim}]와 다릅니다"

        h = self.patch_embed(a.reshape(B, self.n_tokens, self.patch * D)) + self.pos
        assert h.shape == (B, self.n_tokens, self.d)

        c = self.t_embed(timestep_embedding(t, self.temb_dim))
        if self.s_embed is not None and state is not None:
            c = c + self.s_embed(state)     # 조건을 **더한다** — §6.3의 c = Emb(t)+Emb(y)와 같은 형태

        for blk in self.blocks:
            h = blk(h, c, obs)

        g, b = (v.unsqueeze(1) for v in self.ada_f(c).chunk(2, dim=-1))
        h = modulate(self.norm_f(h), g, b)
        out = self.head(h).reshape(B, H, D)    # unpatchify
        assert out.shape == a.shape
        return out


# %% [markdown]
# ## 3. G1에서 $D$를 읽어온다 — $D=29$의 근거
#
# lesson §7.2가 "$D = 29$ ← W1-M2 실측 `nu=29`"라고 적은 그 값입니다. 하드코딩하지 않고 매번
# 모델에서 읽습니다. menagerie가 없으면 **경고만 하고 29로 진행**합니다 — DiT 블록 검증 자체에는
# 로봇 모델이 필요 없기 때문입니다.

# %%
def resolve_menagerie(arg: str | None = None) -> Path | None:
    """--menagerie > MENAGERIE_PATH > <repo>/repos/mujoco_menagerie. 없으면 None."""
    cand = arg or os.environ.get("MENAGERIE_PATH") \
        or str(find_repo_root() / "repos" / "mujoco_menagerie")
    p = Path(cand).expanduser().resolve()
    return p if (p / "unitree_g1").is_dir() else None


def load_g1(menagerie: Path | None) -> dict:
    """G1의 nu · ctrlrange · 관절 이름을 읽는다. 실패하면 D=29 폴백 스펙."""
    if menagerie is None:
        print("  ⚠️ mujoco_menagerie를 찾지 못했습니다 → D=29 상수로 진행합니다.")
        print("     (받으려면: git clone --depth 1 "
              "https://github.com/google-deepmind/mujoco_menagerie.git repos/mujoco_menagerie)")
        n = N_JOINTS_FALLBACK
        return dict(nu=n, lo=-np.ones(n, np.float32), hi=np.ones(n, np.float32),
                    names=[f"joint_{i}" for i in range(n)], source="폴백 상수", real=False)
    import mujoco  # menagerie가 있을 때만 임포트
    xml = menagerie / "unitree_g1" / "scene.xml"
    if not xml.is_file():
        xml = menagerie / "unitree_g1" / "g1.xml"
    m = mujoco.MjModel.from_xml_path(str(xml))
    rng = m.actuator_ctrlrange.copy()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}"
             for i in range(m.nu)]
    return dict(nu=int(m.nu), lo=rng[:, 0].astype(np.float32), hi=rng[:, 1].astype(np.float32),
                names=names, source=str(xml), real=True)


def make_chunks(n: int, horizon: int, D: int, seed: int) -> np.ndarray:
    """관절 한계 안(정규화 좌표 [-1,1])의 sin파 궤적 청크를 만든다. → [n, horizon, D]

    x_j(τ) = clip( c_j + A_j·sin(2π f_j τ/f_s + φ_j),  −1, 1 )
    정규화 규약은 W1-M5 eq.(6)과 같습니다: x = 2(q−lo)/(hi−lo) − 1 ∈ [−1,1] ⟺ 관절 한계 안.
    """
    rng = np.random.default_rng(seed)
    tau = np.arange(horizon, dtype=np.float32) / FS_HZ            # [H]
    center = rng.uniform(-0.35, 0.35, (n, 1, D)).astype(np.float32)
    amp = rng.uniform(0.05, 0.45, (n, 1, D)).astype(np.float32)
    freq = rng.uniform(0.2, 1.5, (n, 1, D)).astype(np.float32)     # 0.2~1.5 Hz
    phase = rng.uniform(0, 2 * np.pi, (n, 1, D)).astype(np.float32)
    x = center + amp * np.sin(2 * np.pi * freq * tau[None, :, None] + phase)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def to_deg(x_norm: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """정규화 좌표 → 라디안 → 도. W1-M5 eq.(6)의 역변환."""
    q = lo + (x_norm + 1.0) / 2.0 * (hi - lo)
    return np.rad2deg(q)


# %% [markdown]
# ## 4. 학습 — 조건부 디노이징
#
# 손실은 01과 같은 $\mathcal{L}_{\text{simple}}$입니다. 달라지는 것은 **조건에 시작 포즈가 들어간다**는
# 것뿐입니다. 이러면 "같은 시작 포즈에서 이어질 수 있는 궤적"이 여러 개이므로 §7.3의 다봉성이
# 그대로 재현됩니다 — 회귀 헤드였다면 그 모드들의 평균(= 거의 정지한 궤적)을 냈을 자리입니다.

# %%
def train_head(model: ActionDiT, data: torch.Tensor, sch: Schedule, *, steps: int, batch: int,
               lr: float, ema_decay: float, log_every: int, seed: int) -> tuple[EMA, list[float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    lrs = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.05, 1.0 - s / steps))
    ema = EMA(model, ema_decay)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = data.shape[0]
    losses, run = [], 0.0
    for s in range(1, steps + 1):
        idx = torch.randint(0, n, (batch,), generator=g).to(data.device)
        a0 = data[idx]                       # [B, H, D]
        state = a0[:, 0, :]                  # 조건: 시작 포즈 (proprioception 대용)
        t = torch.randint(1, sch.T + 1, (batch,), generator=g).to(data.device)
        # q_sample은 [B, ...] 어느 shape이든 되게 짜여 있지 않으므로 평탄화해서 쓴다
        flat = a0.reshape(a0.shape[0], -1)
        x_t, eps = q_sample(flat, t, sch)    # lesson §3.2 닫힌 형태
        eps_hat = model(x_t.view_as(a0), t, state)
        loss = ((eps.view_as(a0) - eps_hat) ** 2).mean()   # lesson §4.4 L_simple
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lrs.step()
        ema.update(model)
        run += loss.item()
        if s % log_every == 0:
            losses.append(run / log_every)
            run = 0.0
    return ema, losses


@torch.no_grad()
def sample_chunks(model: ActionDiT, state: torch.Tensor, sch: Schedule, nfe: int,
                  seed: int = 0, clip_x0: bool = True) -> torch.Tensor:
    """DDIM(η=0)으로 액션 청크를 생성한다. 02에서 적은 NFE에 유리하다고 확인한 그 샘플러.

    `clip_x0` — $\\hat x_0$을 $[-1,1]$로 자릅니다. 이미지 diffusion 구현의 `clip_denoised`와
    같은 장치인데, **로봇 쪽에서는 근거가 더 강합니다**: 정규화 좌표의 $\\pm1$이 곧 관절 한계라
    "데이터가 이 밖에 있을 수 없다"가 물리적 사실입니다(W1-M5 eq.(6)). $t=T$ 근처에서
    $\\sqrt{\\bar\\alpha_T}\\approx0.0064$로 나누느라 예측 오차가 157배로 증폭되는 것을 막습니다.
    """
    B = state.shape[0]
    H, D = model.horizon, model.action_dim
    dev = state.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(B, H, D, generator=g).to(dev)
    tau = make_grid(sch.T, nfe)
    for k in range(len(tau) - 1, -1, -1):
        t = tau[k]
        t_prev = tau[k - 1] if k > 0 else 0
        ab_t = sch.abar[t - 1]
        ab_prev = sch.abar[t_prev - 1] if t_prev > 0 else torch.ones((), device=dev)
        tt = torch.full((B,), t, dtype=torch.long, device=dev)
        eps = model(x, tt, state)
        # lesson §5.1 DDIM: x̂_0 = (x_t − √(1-ᾱ_t)ε_θ)/√ᾱ_t,  x_{τ'} = √ᾱ_{τ'} x̂_0 + √(1-ᾱ_{τ'}) ε_θ
        x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        if clip_x0:
            x0 = x0.clamp(-1.0, 1.0)  # 관절 한계 = 알려진 데이터 상한 (위 docstring)
            eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()  # 자른 x̂_0과 일관되게 ε도 갱신
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).clamp_min(0).sqrt() * eps
    return x


# %% [markdown]
# ## 5. forward 시간 벤치 — lesson §8.3의 "3 ms/NFE"는 어느 규모의 값인가
#
# §8.3 표는 3 ms/NFE와 10 ms/NFE를 **가정값**으로 놓고 자리별 NFE 상한을 계산합니다.
# 그 가정이 어느 모델 크기에 해당하는지를 자기 기기에서 직접 재는 것이 이 절의 목적입니다.
# 배치 1, 토큰 16개(H=32, p_t=2) 기준입니다.

# %%
@torch.no_grad()
def bench_forward(cfgs: list[tuple[int, int, int]], D: int, device: torch.device,
                  reps: int = 20, horizon: int = HORIZON, patch: int = PATCH_T) -> list[dict]:
    out = []
    for d, depth, heads in cfgs:
        m = ActionDiT(horizon, D, patch, d=d, depth=depth, n_heads=heads,
                      state_dim=D).to(device).eval()
        n_par = sum(p.numel() for p in m.parameters())
        a = torch.randn(1, horizon, D, device=device)
        t = torch.full((1,), 500, dtype=torch.long, device=device)
        s = torch.randn(1, D, device=device)
        for _ in range(3):  # 워밍업
            m(a, t, s)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            m(a, t, s)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / reps * 1e3
        out.append(dict(d=d, depth=depth, heads=heads, params=n_par, ms=ms))
        del m
    return out


# %% [markdown]
# ## 6. 그림

# %%
def plot_chunks(real: np.ndarray, gen: np.ndarray, state: np.ndarray, losses: list[float],
                spec: dict, joints: list[int], path: Path, nfe: int, log_every: int) -> Path:
    """real [n_real, H, D], gen [n_gen, H, D] — 같은 시작 포즈 조건에서 생성한 것."""
    lo, hi, names = spec["lo"], spec["hi"], spec["names"]
    tau = np.arange(real.shape[1]) / FS_HZ
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))

    for k, j in enumerate(joints):
        a = ax[0, k]
        a.plot(tau, to_deg(real[0, :, j], lo[j], hi[j]), color="#1971c2", lw=2.5,
               label=lab("원본 청크", "real chunk"))
        for i in range(gen.shape[0]):
            a.plot(tau, to_deg(gen[i, :, j], lo[j], hi[j]), color="#c92a2a", lw=1.2,
                   ls="--", alpha=0.8, label=lab("생성", "generated") if i == 0 else None)
        a.scatter([0], [to_deg(state[j], lo[j], hi[j])], color="k", zorder=5, s=45,
                  label=lab("조건 = 시작 포즈", "conditioned start pose"))
        a.axhline(np.rad2deg(lo[j]), color="#adb5bd", ls=":", lw=1)
        a.axhline(np.rad2deg(hi[j]), color="#adb5bd", ls=":", lw=1)
        a.set_title(f"{names[j]}", fontsize=10)
        a.set_xlabel(lab("시간 [s]", "time [s]")), a.set_ylabel(lab("관절각 [deg]", "joint angle [deg]"))
        a.grid(alpha=0.3)
        if k == 0:
            a.legend(fontsize=8)

    x = np.arange(1, len(losses) + 1) * log_every
    ax[1, 0].plot(x, losses, color="#1971c2", lw=1.6)
    ax[1, 0].set_xlabel(lab("학습 스텝", "training step"))
    ax[1, 0].set_ylabel(r"$\mathcal{L}_{\rm simple}$")
    ax[1, 0].set_title(lab("(d) 학습 곡선", "(d) training loss")), ax[1, 0].grid(alpha=0.3)

    # 프레임 간 변화량 분포 — 생성 궤적이 원본만큼 매끄러운가
    dr = np.abs(np.diff(real, axis=1)).ravel()
    dg = np.abs(np.diff(gen, axis=1)).ravel()
    bins = np.linspace(0, max(dr.max(), dg.max()), 60)
    ax[1, 1].hist(dr, bins=bins, alpha=0.6, color="#1971c2", density=True,
                  label=lab("원본", "real"))
    ax[1, 1].hist(dg, bins=bins, alpha=0.6, color="#c92a2a", density=True,
                  label=lab("생성", "generated"))
    ax[1, 1].set_xlabel(lab("프레임 간 변화량 |Δx| (정규화 좌표)", "|Δx| per frame (normalized)"))
    ax[1, 1].set_ylabel(lab("밀도", "density"))
    ax[1, 1].set_title(lab("(e) 매끄러움 — 겹칠수록 좋다", "(e) smoothness — overlap is good"))
    ax[1, 1].legend(), ax[1, 1].grid(alpha=0.3)

    # 값 분포 — 관절 한계 [-1,1] 밖으로 나가는가
    ax[1, 2].hist(real.ravel(), bins=60, alpha=0.6, color="#1971c2", density=True,
                  label=lab("원본", "real"))
    ax[1, 2].hist(gen.ravel(), bins=60, alpha=0.6, color="#c92a2a", density=True,
                  label=lab("생성", "generated"))
    for v in (-1, 1):
        ax[1, 2].axvline(v, color="k", ls="--", lw=1.2)
    ax[1, 2].set_xlabel(lab("정규화 좌표 x (±1 = 관절 한계)", "normalized x (±1 = joint limit)"))
    ax[1, 2].set_ylabel(lab("밀도", "density"))
    ax[1, 2].set_title(lab("(f) 관절 한계 준수", "(f) staying inside joint limits"))
    ax[1, 2].legend(), ax[1, 2].grid(alpha=0.3)

    fig.suptitle(lab(f"W1-M3 · DiT 액션 헤드 [B,{real.shape[1]},{real.shape[2]}] · "
                     f"DDIM NFE={nfe} · (a)~(c)는 같은 시작 포즈 조건",
                     f"W1-M3 · DiT action head [B,{real.shape[1]},{real.shape[2]}] · "
                     f"DDIM NFE={nfe} · (a)-(c) share one conditioning pose"), fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# %% [markdown]
# ## 7. main

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W1-M3 실습 3: DiT 블록 구현 + 액션 청크 헤드 (lesson §6·§7)")
    p.add_argument("--menagerie", default=None, help="mujoco_menagerie 경로")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--horizon", type=int, default=HORIZON, help="청크 길이 H (기본 32)")
    p.add_argument("--patch", type=int, default=PATCH_T, help="시간축 patchify p_t (기본 2)")
    p.add_argument("--dim", type=int, default=128, help="hidden 차원 d")
    p.add_argument("--depth", type=int, default=4, help="DiT 블록 수 N")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--n-train", type=int, default=8192, help="학습 청크 수")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--nfe", type=int, default=20, help="생성 시 DDIM NFE (기본 20 = 02의 관찰)")
    p.add_argument("--n-gen", type=int, default=4, help="같은 조건에서 뽑을 샘플 수 (다봉성 확인용)")
    p.add_argument("--no-clip-x0", action="store_true",
                   help="x̂_0 클리핑을 끈다 (관절 한계를 안다는 사전지식을 빼고 보기)")
    p.add_argument("--bench-reps", type=int, default=20)
    p.add_argument("--no-bench-xl", action="store_true", help="DiT-XL 급 벤치 행을 건너뛴다")
    p.add_argument("--no-train", action="store_true", help="블록 검증과 벤치만 하고 학습은 건너뛴다")
    p.add_argument("--smoke", action="store_true", help="1분 이내에 완주하는 축소 경로")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--ascii-labels", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.smoke:
        args.steps, args.n_train, args.bench_reps = 300, 2048, 5
        args.no_bench_xl = True
        print("[smoke] 축소 경로로 실행합니다 (300 스텝, DiT-XL 벤치 생략). "
              "생성 궤적 품질은 학습이 덜 된 값입니다.")
    setup_korean_font(args.ascii_labels)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = pick_device(args.device)
    out_dir = artifacts_dir()
    t_start = time.perf_counter()

    print("=" * 88)
    print(f"  {MODULE_ID} 실습 3 — DiT 블록(§6.3) → 액션 청크 헤드(§7.2)")
    print("=" * 88)

    # --- [1] adaLN-Zero 항등함수 검증 ---------------------------------------
    print("\n=== [1] 필수 검증 — α=0이면 블록이 항등함수인가 (lesson §6.3) ===")
    n_tok = args.horizon // args.patch
    v = verify_identity(args.dim, args.heads, B=4, T=n_tok, device=dev, seed=args.seed)
    print_table(["초기화", "max |block(h,c) − h|", "판정"],
                [["adaLN-Zero (α MLP = 0)", f"{v['zero']:.3e}",
                  "✅ 항등 (bit 단위 일치)" if v["exact"] else "✅ 항등 (allclose)"],
                 ["랜덤 초기화 (대조군)", f"{v['random']:.3e}", "❌ 항등 아님 — 정상"],
                 ["랜덤이지만 α만 0", f"{v['alpha_only']:.3e}", "✅ 항등 — α가 잔차 게이트"]],
                ["left", "right", "left"])
    print("  → 세 줄이 함께 말하는 것: **항등을 만드는 것은 γ·β가 아니라 α 하나**입니다.")
    print("     lesson §6.2 표에서 adaLN(25.21) → adaLN-Zero(19.47)의 차이가 이 α 하나입니다.")
    print("     28개 블록을 통과해도 초기 신호가 변형되지 않으므로 그래디언트가 폭주·소멸하지 않습니다.")

    # --- [2] G1에서 D 읽기 --------------------------------------------------
    print("\n=== [2] D = ? — G1 모델에서 직접 읽는다 (lesson §7.2의 'D=29 ← W1-M2 실측') ===")
    spec = load_g1(resolve_menagerie(args.menagerie))
    D = spec["nu"]
    print(f"  모델: {spec['source']}")
    print(f"  model.nu = {D}   " + ("✅ lesson §7.2의 D=29와 일치" if D == 29
                                    else f"⚠️ lesson은 29를 씁니다 (읽은 값 {D})"))
    if spec["real"]:
        w = spec["hi"] - spec["lo"]
        print(f"  ctrlrange 폭: 최소 {w.min():.3f} rad ({spec['names'][int(w.argmin())]}) / "
              f"최대 {w.max():.3f} rad ({spec['names'][int(w.argmax())]})")

    # --- [3] shape 관통 확인 (lesson §7.2) -----------------------------------
    print(f"\n=== [3] shape 관통 — [B,{args.horizon},{D}] → [B,{n_tok},d] → [B,{args.horizon},{D}] ===")
    model = ActionDiT(args.horizon, D, args.patch, d=args.dim, depth=args.depth,
                      n_heads=args.heads, state_dim=D).to(dev)
    B = 4
    a = torch.randn(B, args.horizon, D, device=dev)
    t = torch.randint(1, 1001, (B,), device=dev)
    st = torch.randn(B, D, device=dev)
    out = model(a, t, st)
    assert out.shape == (B, args.horizon, D), f"출력 shape {tuple(out.shape)}"
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  입력 a_t   [B={B}, H={args.horizon}, D={D}]      = {B * args.horizon * D:,} 원소")
    print(f"  patchify   [B={B}, {n_tok}, {args.patch}·D={args.patch * D}] "
          f"→ Linear → [B={B}, {n_tok}, d={args.dim}]")
    print(f"  N={args.depth} 블록 통과 후 unpatchify → 출력 {tuple(out.shape)}   ✅ assert 통과")
    print(f"  파라미터 {n_par:,}개  (토큰 {n_tok}개 — lesson §7.1 표의 '이미지 256 → 액션 8~32')")

    # 초기 상태에서 헤드가 0을 내놓는지 (final layer도 zero-init이므로)
    print(f"  초기 출력 norm = {out.abs().max().item():.3e}  "
          "← 최종 Linear도 zero-init이라 학습 시작 시 ε̂ = 0입니다")

    # §7.2 블록도의 cross-attention 가지도 shape이 통하는지 (관측 토큰 조건)
    V, N_img, L, obs_dim = 2, 64, 16, 512   # 카메라 2대 × 패치 64 + 언어 토큰 16 (예시 수치)
    xmodel = ActionDiT(args.horizon, D, args.patch, d=args.dim, depth=2, n_heads=args.heads,
                       state_dim=D, cross_dim=obs_dim).to(dev)
    obs = torch.randn(B, V * N_img + L, obs_dim, device=dev)
    xout = xmodel(a, t, st, obs)
    assert xout.shape == (B, args.horizon, D)
    print(f"  cross-attention 가지: 관측 토큰 [B,{V * N_img + L},{obs_dim}] "
          f"(카메라 {V}대×{N_img} + 언어 {L}) → 출력 {tuple(xout.shape)}   ✅ assert 통과")
    print("  ⓘ 이 가변 길이 이종 토큰을 그냥 붙일 수 있다는 것이 lesson §6.4 마지막 줄"
          "('conv U-Net에는 남의 일')의 실체입니다.")
    del xmodel

    # --- [4] forward 시간 벤치 ----------------------------------------------
    print(f"\n=== [4] 1회 forward 시간 — lesson §8.3의 '3 ms/NFE 가정'과 대조 ===")
    cfgs = [(args.dim, args.depth, args.heads), (256, 6, 8), (512, 12, 8)]
    if not args.no_bench_xl:
        cfgs.append((1152, 28, 16))  # DiT-XL 구성 (lesson §6.3: d=1152, 28블록, 16헤드)
    bench = bench_forward(cfgs, D, dev, args.bench_reps, args.horizon, args.patch)
    rows = []
    for b in bench:
        note = ""
        if (b["d"], b["depth"]) == (args.dim, args.depth):
            note = "← 이 실습"
        elif b["d"] == 1152:
            note = "← DiT-XL 구성 (§6.3)"
        rows.append([f"d={b['d']}, {b['depth']}블록, {b['heads']}헤드",
                     f"{b['params'] / 1e6:.1f}M", f"{b['ms']:.2f}",
                     f"{3.0 / b['ms']:.1f}" if b["ms"] > 0 else "—", note])
    print_table(["구성", "파라미터", "1 forward [ms]", "3 ms 예산 내 NFE", ""],
                rows, ["left", "right", "right", "right", "left"])
    print(f"  측정: 배치 1, 토큰 {n_tok}개, {dev.type}, {args.bench_reps}회 평균")
    print("  → **'3 ms/NFE'는 모델 크기·하드웨어·토큰 수가 정해져야 나오는 숫자입니다.** 위 표의 어느")
    print("     행이 우리 헤드인지가 §8.3 표의 NFE 상한을 통째로 바꿉니다. 실측값은 §10 팀 질문 2번.")
    if dev.type == "cpu":
        print("  ⚠️ CPU 측정값입니다. 실제 배치는 GPU이므로 이 숫자를 그대로 예산에 넣지 마세요.")
    with open(out_dir / "03_forward_bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["d", "depth", "heads", "params", "ms"])
        w.writeheader(), w.writerows(bench)
    print(f"  [저장] {out_dir / '03_forward_bench.csv'}")

    if args.no_train:
        print(f"\n총 소요 {time.perf_counter() - t_start:.1f}s  (--no-train)")
        return

    # --- [5] 학습 데모 -------------------------------------------------------
    print(f"\n=== [5] 학습 데모 — G1 관절 한계 안의 sin파 궤적 {args.n_train:,}개 ===")
    data_np = make_chunks(args.n_train, args.horizon, D, args.seed)
    assert np.all(np.abs(data_np) <= 1.0 + 1e-6), "정규화 좌표를 벗어난 샘플이 있습니다"
    print(f"  데이터 {data_np.shape}  (50 Hz 기준 청크 하나 = {args.horizon / FS_HZ:.2f}초)")
    print(f"  값 범위 [{data_np.min():.3f}, {data_np.max():.3f}] ⊂ [-1,1] = 관절 한계 안")
    data = torch.as_tensor(data_np, device=dev)

    sch = make_schedule(1000).to(dev)
    sch = Schedule(sch.T, *(getattr(sch, f).float() for f in
                            ("beta", "alpha", "abar", "abar_prev", "sqrt_abar",
                             "sqrt_one_minus_abar", "beta_tilde")))
    log_every = max(1, args.steps // 100)
    t0 = time.perf_counter()
    ema, losses = train_head(model, data, sch, steps=args.steps, batch=args.batch, lr=args.lr,
                             ema_decay=args.ema, log_every=log_every, seed=args.seed)
    train_sec = time.perf_counter() - t0
    print(f"  학습 {train_sec:.1f}s ({args.steps} 스텝)  "
          f"손실 {losses[0]:.4f} → {losses[-1]:.4f}")
    ema.copy_to(model)
    model.eval()

    # --- [6] 생성 -----------------------------------------------------------
    print(f"\n=== [6] 생성 — DDIM NFE={args.nfe}, 같은 시작 포즈에서 {args.n_gen}개 ===")
    ref = data[:1]                                  # 원본 청크 1개
    state = ref[:, 0, :].repeat(args.n_gen, 1)      # 그 시작 포즈를 조건으로 n_gen번
    print(f"  x̂_0 클리핑 [-1,1]: {'끔 (--no-clip-x0)' if args.no_clip_x0 else '켬'}"
          "   ← 정규화 좌표의 ±1이 곧 관절 한계라 물리적으로 정당한 사전지식입니다")
    t0 = time.perf_counter()
    gen = sample_chunks(model, state, sch, args.nfe, seed=args.seed + 7,
                        clip_x0=not args.no_clip_x0)
    gen_sec = time.perf_counter() - t0
    gen_np = gen.cpu().numpy()
    ref_np = ref.cpu().numpy()
    print(f"  생성 {gen_sec * 1e3:.0f} ms ({gen_sec / args.nfe * 1e3:.2f} ms/NFE, "
          f"배치 {args.n_gen}개)")

    lo, hi = spec["lo"], spec["hi"]
    start_err = np.abs(gen_np[:, 0, :] - ref_np[0, 0, :][None])
    start_err_deg = np.abs(to_deg(gen_np[:, 0, :], lo, hi) - to_deg(ref_np[0, 0, :], lo, hi)[None])
    viol = float((np.abs(gen_np) > 1.0).mean() * 100)
    d_real = float(np.abs(np.diff(data_np[:256], axis=1)).mean())
    d_gen = float(np.abs(np.diff(gen_np, axis=1)).mean())
    print_table(["지표", "값", "읽는 법"],
                [["시작 포즈 정합 (정규화)", f"{start_err.mean():.4f}",
                  "조건이 실제로 먹혔는가 (0에 가까울수록 좋다)"],
                 ["시작 포즈 정합 (도)", f"{start_err_deg.mean():.2f}°", "라디안 환산"],
                 ["관절 한계 |x|>1 비율", f"{viol:.2f}%",
                  "학습 데이터는 0% — 생성이 한계를 넘으면 클리핑이 필요하다"],
                 ["프레임간 |Δx| 평균 (원본)", f"{d_real:.4f}", "매끄러움 기준값"],
                 ["프레임간 |Δx| 평균 (생성)", f"{d_gen:.4f}",
                  "원본보다 크면 떨리는 궤적 = 하위 제어기에 나쁘다"]],
                ["left", "right", "left"])
    print("  ⓘ 같은 시작 포즈에서 뽑은 청크들이 서로 다른 것이 정상입니다 — lesson §7.3의 다봉성입니다.")
    print("     회귀 헤드였다면 이 모드들의 평균(거의 정지한 궤적)을 냈을 자리입니다.")

    if not args.no_plot:
        rng = np.random.default_rng(args.seed)
        joints = sorted(rng.choice(D, size=3, replace=False).tolist())
        p = plot_chunks(ref_np, gen_np, ref_np[0, 0, :], losses, spec, joints,
                        out_dir / "03_action_chunk.png", args.nfe, log_every)
        print(f"\n  [저장] {p}")
        print("     (a)~(c): 파란 실선이 원본, 빨간 점선이 같은 시작 포즈에서 생성한 것,"
              " 점선 회색이 관절 한계")

    print(f"\n총 소요 {time.perf_counter() - t_start:.1f}s")
    print("다음: 백지에 DiT 블록을 직접 그리세요 → dit_block_worksheet.excalidraw "
          "(정답 대조: lesson.md §6.3)")


if __name__ == "__main__":
    import sys

    # 노트북(ipykernel)에서는 argparse가 jupyter의 -f 인자를 먹지 않도록 빈 리스트를 넘긴다.
    # → 전부 기본값으로 실행됩니다. --smoke로 돌리려면 이 셀을 main(["--smoke"])로 고치세요.
    main(None if "ipykernel" not in sys.modules else [])
