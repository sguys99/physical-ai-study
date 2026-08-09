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
# # W1-M4 실습 3 — W1-M3의 액션 헤드를 **그대로 가져와** objective만 FM으로 바꾼다
#
# lesson.md `§5.3`(마지막 행 "코드 변경량: 세 줄") · `§7.1`(학습 루프 블록도) · `§7.2`(액션 헤드판) ·
# `§8.2`(자리별 NFE 상한)의 실행판입니다.
#
# **DiT 블록을 다시 짜지 않습니다.** W1-M3
# [`03_dit_action_head.py`](../../03-diffusion-ddpm-dit/practice/03_dit_action_head.py)를
# 모듈로 로드해 `DiTBlock` · `ActionDiT` · G1 로딩 · 청크 생성 · forward 벤치 · 그림을
# 전부 재사용하고, **바뀌는 것은 학습 스텝 함수 하나**입니다. 그 함수 두 개를 `difflib`로
# 비교해 **정말 세 줄인지 스크립트가 직접 세어서 출력**합니다. lesson §5.3 마지막 행의 검산입니다.
#
# 확인할 것:
#
# 1. **"세 줄만 바뀐다"의 기계적 검증** — `ddpm_train_step` vs `fm_train_step`의 diff 줄 수.
# 2. **재사용 목록과 새로 쓴 줄 수** — 스크립트 끝에서 `inspect.getsource`로 세어 출력합니다.
# 3. **청크 규격은 lesson §7.2에 맞춰 `[B, 50, 29]`** — $H{=}50$은 pi0 실수, $D{=}29$는 W1-M2 실측 `nu`.
#    (W1-M3는 $H{=}32$였으므로 `--horizon` 기본값만 다릅니다.)
# 4. **같은 데이터·같은 백본·같은 스텝 수로 DDPM 헤드와 FM 헤드를 나란히 학습**하고
#    NFE별 재구성 품질과 wall-clock을 표로. lesson §8.2의 "자리별 NFE 상한"에
#    **자기 기기 실측 ms/NFE**를 대입한 표까지 만들어 CSV로 저장합니다.
#
# 출력(`artifacts/W1-M4/`): `03_fm_vs_ddpm.png` · `03_fm_chunks.png` ·
# `03_fm_vs_ddpm.csv` · `03_forward_bench.csv` · `03_nfe_budget.csv`
#
# > ⚠️ **여기 나오는 헤드 구조는 lesson §7.2의 교육용 재현이지 회사 구현이 아닙니다.**
# > 회사 L4에 액션 헤드가 있는지, objective가 FM인지, 경로 스케줄과 $t$ 분포가 무엇인지,
# > VLM이 적분 루프 밖인지는 전부 **미확인**입니다(lesson §9.4 · 「팀에 물어볼 것」의 M4-1~M4-5).
#
# **GPU 불필요.** `--smoke`는 1~2분 이내.

# %%
from __future__ import annotations

import os

# ⚠️ `import mujoco` 보다 먼저 (W1-M2 lesson §6.2). 렌더는 하지 않지만 규약을 지킵니다.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import ast  # noqa: E402
import csv  # noqa: E402
import difflib  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import sys  # noqa: E402
import textwrap  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless 고정 — 뷰어를 띄우지 않는다

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

MODULE_ID = "W1-M4"

# lesson §7.2의 액션 청크 텐서 규격 — W1-M3(H=32)와 다른 것은 H 하나뿐입니다.
HORIZON = 50        # H — pi0 실수 (arXiv:2410.24164)
PATCH_T = 2         # p_t — 시간축 patchify → 토큰 25개 (lesson §7.2 블록도의 [B,25,d])
T_SCALE = 1000.0    # 연속 t를 sinusoidal 임베딩 대역에 올리는 스케일 (01과 같은 이유)

NFE_GRID = [1, 2, 4, 10, 20, 50]
NFE_GRID_SMOKE = [1, 4, 20]
REF_NFE_FM, REF_NFE_DDPM = 200, 1000   # "적분 오차"의 기준이 되는 충분히 큰 NFE

# lesson §8.2 표의 자리들. (이름, 주기 예산 ms, VLM이 루프 밖에서 먹는 ms)
BUDGET_SLOTS = [("L2 안에 직접 · 500 Hz", 2.0, 0.0),
                ("L2 안에 직접 · 50 Hz", 20.0, 0.0),
                ("L4 청크 생성 · 10 Hz", 100.0, 60.0),
                ("L4 청크 생성 · 5 Hz", 200.0, 60.0)]


# %% [markdown]
# ## 0. W1-M3의 액션 헤드를 통째로 로드
#
# `03_dit_action_head.py`를 로드하면 그 안에서 `02_samplers_compare.py` → `01_ddpm_toy.py`까지
# 딸려 옵니다. 즉 **한 줄로 W1-M3 practice 세 파일 전부**를 확보합니다.

# %%
_ROOT_MARKERS = ("course", "docs", "CLAUDE.md")
_W1M3 = ("course", "w1-generative-core", "03-diffusion-ddpm-dit", "practice")


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


def w1m3_dir() -> Path:
    return find_repo_root().joinpath(*_W1M3)


def artifacts_dir() -> Path:
    out = find_repo_root() / "artifacts" / MODULE_ID
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_sibling(fname: str, modname: str, folder: Path):
    p = folder / fname
    if not p.is_file():
        raise SystemExit(f"[에러] {p} 를 찾지 못했습니다")
    spec = importlib.util.spec_from_file_location(modname, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


M3 = load_sibling("03_dit_action_head.py", "m3_action_head", w1m3_dir())
M2, M1 = M3.M2, M3.M1

# ── 재사용 목록 (새로 짜지 않은 것) ──────────────────────────────────────────
REUSED = {
    "DiTBlock (§6.3 블록)": M3.DiTBlock,
    "ActionDiT (§7.2 헤드)": M3.ActionDiT,
    "modulate": M3.modulate,
    "verify_identity (adaLN-Zero 항등 assert)": M3.verify_identity,
    "resolve_menagerie": M3.resolve_menagerie,
    "load_g1 (nu=29 읽기)": M3.load_g1,
    "make_chunks (sin파 궤적)": M3.make_chunks,
    "to_deg": M3.to_deg,
    "sample_chunks (DDIM 샘플러)": M3.sample_chunks,
    "bench_forward (1회 forward ms)": M3.bench_forward,
    "plot_chunks (궤적 그림)": M3.plot_chunks,
    "Schedule / make_schedule": M1.make_schedule,
    "timestep_embedding": M1.timestep_embedding,
    "EMA": M1.EMA,
    "energy_distance": M2.energy_distance,
    "sliced_w1": M2.sliced_w1,
}

DiTBlock, ActionDiT = M3.DiTBlock, M3.ActionDiT
make_chunks, to_deg, load_g1 = M3.make_chunks, M3.to_deg, M3.load_g1
Schedule, make_schedule = M1.Schedule, M1.make_schedule
EMA, pick_device = M1.EMA, M1.pick_device
energy_distance, sliced_w1 = M2.energy_distance, M2.sliced_w1
print_table = M3.print_table
FS_HZ = M3.FS_HZ


def setup_korean_font(force_ascii: bool = False) -> bool:
    """M3 쪽 플래그까지 세팅해야 재사용한 plot_chunks 의 라벨이 한글로 나옵니다."""
    return M3.setup_korean_font(force_ascii)


def lab(ko: str, en: str) -> str:
    return M3.lab(ko, en)


# %% [markdown]
# ## 1. 헤드 래퍼 — 시간 규약만 다르고 본체는 같은 객체
#
# 두 objective가 **완전히 같은 `ActionDiT`**를 씁니다. 파라미터 수도, 1회 forward 시간도 같습니다.
# 다른 것은 `t`를 어떤 단위로 넣느냐 하나입니다.
#
# | | DDPM 헤드 | FM 헤드 |
# |---|---|---|
# | `t` 정의역 | 정수 $\{1,\dots,1000\}$ | 실수 $[0,1]$ |
# | `t_scale` | 1.0 | **1000.0** |
# | 본체 | `ActionDiT` | **같은 `ActionDiT`** |
#
# 래퍼로 단위를 흡수했기 때문에 학습 스텝의 마지막 줄 `model(x_t, t, state)`가 두 쪽에서
# **글자 하나 안 다릅니다.** 그 덕에 아래 diff가 정확히 세 줄이 됩니다.

# %%
class ActionHead(nn.Module):
    """W1-M3 ActionDiT 를 감싸고 시간 인자 규약만 맞춘다. 파라미터는 전부 ActionDiT 것."""

    def __init__(self, *, t_scale: float, **kw):
        super().__init__()
        self.net = ActionDiT(**kw)          # ← W1-M3의 클래스. 한 줄도 안 고쳤습니다
        self.t_scale = t_scale
        self.horizon, self.action_dim = self.net.horizon, self.net.action_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor, state: torch.Tensor | None = None,
                obs: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(x, t.float() * self.t_scale, state, obs)


# %% [markdown]
# ## 2. ★ 학습 스텝 — 여기가 전부 바뀌는 곳
#
# 두 함수를 **일부러 같은 서식·같은 시그니처**로 썼습니다. `difflib`로 비교해 몇 줄이 다른지
# 스크립트가 직접 세어 출력합니다(lesson §5.3 마지막 행의 검산).
#
# `fm_train_step`이 `sch`를 인자로 받으면서 **한 번도 쓰지 않는 것**도 그대로 두었습니다 —
# §5.3 표의 "스케줄 의존성: 없음" 행이 시각적으로 드러나는 자리입니다.

# %%
def _b(v: torch.Tensor) -> torch.Tensor:
    """[B] → [B,1,1] 브로드캐스트. 두 스텝 함수가 공유합니다."""
    return v[:, None, None]


def ddpm_train_step(model: nn.Module, a1: torch.Tensor, state: torch.Tensor,
                    sch, g: torch.Generator) -> torch.Tensor:
    """W1-M3 §4.4  L_simple = E‖ε − ε_θ(x_t, t)‖²    (a1 = 데이터 청크 [B,H,D])"""
    B, dev = a1.shape[0], a1.device
    t = torch.randint(1, sch.T + 1, (B,), generator=g).to(dev)                        # ① 시간
    x0 = torch.randn(a1.shape, generator=g).to(dev)
    x_t = _b(sch.sqrt_abar[t - 1]) * a1 + _b(sch.sqrt_one_minus_abar[t - 1]) * x0     # ② 보간
    target = x0                                                                       # ③ target
    return ((model(x_t, t, state) - target) ** 2).mean()


def fm_train_step(model: nn.Module, a1: torch.Tensor, state: torch.Tensor,
                  sch, g: torch.Generator) -> torch.Tensor:
    """lesson §5.2  L_RF = E‖v_θ(x_t, t) − (x₁ − x₀)‖²   (sch 는 받기만 하고 쓰지 않습니다)"""
    B, dev = a1.shape[0], a1.device
    t = torch.rand(B, generator=g).to(dev)                                            # ① 시간 ← 변경
    x0 = torch.randn(a1.shape, generator=g).to(dev)
    x_t = (1.0 - _b(t)) * x0 + _b(t) * a1                                             # ② 보간 ← 변경
    target = a1 - x0                                                                  # ③ target ← 변경
    return ((model(x_t, t, state) - target) ** 2).mean()


THIS_FILE = "03_fm_action_head.py"


def source_of(fn, fname: str = THIS_FILE) -> str:
    """함수/클래스의 소스를 문자열로 얻는다.

    `inspect.getsource`는 **노트북/`exec` 환경에서 실패**합니다(파일에 대응하는 코드 객체가
    없기 때문). 그때는 같은 폴더의 원본 `.py`를 열어 `ast`로 해당 정의만 잘라 옵니다.
    W1-M3에서 import한 것들은 실제 파일에서 로드됐으므로 첫 경로로 그냥 성공합니다.
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


def function_body(fn) -> list[str]:
    """함수의 **본문만** 뽑는다 (def 줄·docstring 제외, 주석은 코드로 취급해 남김).

    시그니처가 여러 줄일 수 있어 문자열 매칭 대신 `ast`로 본문 첫 문장의 줄 번호를 찾습니다.
    """
    src = textwrap.dedent(source_of(fn))
    if not src.strip():
        return []
    stmts = ast.parse(src).body[0].body
    first = stmts[0]
    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str) and len(stmts) > 1):
        first = stmts[1]                              # docstring 건너뛰기
    lines = src.splitlines()
    return [ln.rstrip() for ln in lines[first.lineno - 1:] if ln.strip()]


def objective_diff() -> tuple[list[str], int]:
    """두 스텝 함수의 본문 diff. 반환: (diff 줄, 바뀐 줄 수)"""
    a, b = function_body(ddpm_train_step), function_body(fm_train_step)
    diff = list(difflib.unified_diff(a, b, "ddpm_train_step", "fm_train_step", lineterm="", n=0))
    changed = sum(1 for ln in diff if ln.startswith("+") and not ln.startswith("+++"))
    return diff, changed


# %% [markdown]
# ## 3. 학습 루프 하나로 두 objective — 나머지가 안 바뀐다는 증거
#
# W1-M3 `train_head()`에서 안쪽 5줄만 `step_fn`으로 빼낸 것입니다. 옵티마이저(AdamW) ·
# LR 스케줄 · EMA · 로깅이 전부 그대로이고, 두 objective가 **이 함수 하나를 공유**합니다.

# %%
def train_head(model: nn.Module, data: torch.Tensor, sch, step_fn, *, steps: int, batch: int,
               lr: float, ema_decay: float, log_every: int, seed: int) -> tuple[EMA, list[float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    lrs = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.05, 1.0 - s / steps))
    ema = EMA(model, ema_decay)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = data.shape[0]
    losses, run = [], 0.0
    for s in range(1, steps + 1):
        idx = torch.randint(0, n, (batch,), generator=g).to(data.device)
        a1 = data[idx]                      # [B, H, D]
        state = a1[:, 0, :]                 # 조건: 시작 포즈 (proprioception 대용, W1-M3와 동일)
        loss = step_fn(model, a1, state, sch, g)     # ← 여기 하나만 갈아끼웁니다
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


# %% [markdown]
# ## 4. FM 샘플러 — Euler + 관절 한계 클리핑
#
# lesson §7.1 아래쪽 블록도 그대로입니다. 클리핑은 W1-M3 `sample_chunks`의 `clip_x0`와
# **정확히 같은 자리**인데, 직선 경로에서는 $\hat x_1$이 현재 속도로 $t=1$까지 외삽한 값입니다.
#
# $$\hat x_1 = x_t + (1-t)\,v_\theta(x_t,t) \;\xrightarrow{\text{clip}\ [-1,1]}\;
#   v \leftarrow \frac{\hat x_1 - x_t}{1-t}$$
#
# 정규화 좌표의 $\pm1$이 곧 관절 한계라 **"데이터가 이 밖에 있을 수 없다"가 물리적 사실**입니다
# (W1-M5 eq.(6) · W1-M3 03의 같은 논거).

# %%
@torch.no_grad()
def euler_sample_chunks(model: nn.Module, state: torch.Tensor, nfe: int, *, seed: int = 0,
                        clip_x1: bool = True, x0: torch.Tensor | None = None) -> torch.Tensor:
    B, dev = state.shape[0], state.device
    H, D = model.horizon, model.action_dim
    if x0 is None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        x0 = torch.randn(B, H, D, generator=g).to(dev)
    x, h = x0.clone(), 1.0 / nfe
    for k in range(nfe):
        t = k * h
        v = model(x, torch.full((B,), t, device=dev), state)
        if clip_x1:
            x1h = (x + (1.0 - t) * v).clamp(-1.0, 1.0)   # x̂₁ 을 관절 한계 안으로
            v = (x1h - x) / max(1.0 - t, 1e-6)           # 자른 x̂₁과 일관되게 v 갱신
        x = x + h * v                                     # eq. §7.1  x ← x + h·v_θ(x, k·h)
    return x


# %% [markdown]
# ## 5. 평가 — 세 가지를 함께 본다
#
# | 지표 | 무엇을 재는가 |
# |---|---|
# | **energy distance** | 생성한 청크 **분포**가 데이터 분포와 같은가 (청크를 $H\!\cdot\!D$ 차원 벡터로 펴서) |
# | **적분 오차** | 같은 조건·같은 초기 노이즈에서 큰 NFE 결과와 얼마나 벌어지는가 → **NFE를 줄여 잃는 것** |
# | **시작 포즈 정합** | 조건이 실제로 먹혔는가 (도 단위) |
#
# 조건은 **held-out 실제 청크들의 시작 포즈**를 그대로 씁니다. 하나의 포즈에서 여러 개를 뽑으면
# 다봉성은 보이지만 분포 비교가 안 되기 때문입니다(그 그림은 §7에서 따로 그립니다).

# %%
@torch.no_grad()
def eval_chunks(gen: torch.Tensor, real: torch.Tensor, ref_gen: torch.Tensor,
                lo: np.ndarray, hi: np.ndarray, n_proj: int, seed: int) -> dict:
    B = gen.shape[0]
    e = energy_distance(gen.reshape(B, -1), real.reshape(B, -1))
    w = sliced_w1(gen.reshape(B, -1), real.reshape(B, -1), n_proj, seed)
    dev_ref = float((gen - ref_gen).pow(2).sum(-1).sqrt().mean())      # 적분 오차 (정규화 좌표)
    g0, r0 = gen[:, 0, :].cpu().numpy(), real[:, 0, :].cpu().numpy()
    start_deg = float(np.abs(to_deg(g0, lo, hi) - to_deg(r0, lo, hi)).mean())
    viol = float((gen.abs() > 1.0).float().mean() * 100)
    smooth = float(gen.diff(dim=1).abs().mean())
    return dict(energy=e, sw1=w, self_dev=dev_ref, start_deg=start_deg,
                viol_pct=viol, smooth=smooth)


# %% [markdown]
# ## 6. 그림

# %%
def plot_compare(res: list[dict], path: Path, ms_per_nfe: float) -> Path:
    C = {"ddpm": "#1971c2", "fm": "#2f9e44"}
    N = {"ddpm": lab("DDPM 헤드 + DDIM", "DDPM head + DDIM"),
         "fm": lab("FM 헤드 + Euler", "FM head + Euler")}
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.5))
    for obj in ("ddpm", "fm"):
        r = sorted([d for d in res if d["objective"] == obj], key=lambda d: d["nfe"])
        n = [d["nfe"] for d in r]
        ax[0].loglog(n, [d["energy"] for d in r], "o-", color=C[obj], lw=2, label=N[obj])
        ax[1].loglog(n, [d["self_dev"] for d in r], "o-", color=C[obj], lw=2, label=N[obj])
        ax[2].loglog([d["nfe"] * ms_per_nfe for d in r], [d["energy"] for d in r], "o-",
                     color=C[obj], lw=2, label=N[obj])
    ax[0].set_xlabel("NFE"), ax[0].set_ylabel(lab("에너지 거리 (청크 분포)",
                                                  "energy distance (chunk dist.)"))
    ax[0].set_title(lab("(a) 청크 분포 품질 vs NFE", "(a) chunk distribution quality vs NFE"))
    ax[1].set_xlabel("NFE")
    ax[1].set_ylabel(lab("큰 NFE 기준 대비 편차", "deviation from large-NFE reference"))
    ax[1].set_title(lab("(b) 적분 오차 — NFE를 줄여 잃는 것",
                        "(b) integration error — what fewer steps cost"))
    ax[2].set_xlabel(lab(f"헤드 지연 [ms]  (실측 {ms_per_nfe:.2f} ms/NFE, 배치 1)",
                         f"head latency [ms]  (measured {ms_per_nfe:.2f} ms/NFE, batch 1)"))
    ax[2].set_ylabel(lab("에너지 거리", "energy distance"))
    ax[2].axvline(140.0, color="#c92a2a", ls="--", lw=1.4)
    ax[2].annotate(lab("L4 5 Hz 자리 예산 140 ms", "L4 5 Hz budget 140 ms"),
                   xy=(140, ax[2].get_ylim()[1]), fontsize=9, color="#c92a2a",
                   rotation=90, va="top", ha="right")
    ax[2].set_title(lab("(c) 품질 vs 지연 — 로봇이 실제로 쓰는 축 (§8.2)",
                        "(c) quality vs latency — the axis robots actually spend (§8.2)"))
    for a in ax:
        a.legend(fontsize=9), a.grid(alpha=0.3, which="both")
    fig.suptitle(lab(f"W1-M4 · 같은 ActionDiT · 같은 데이터 · 같은 학습 스텝 — objective만 교체 "
                     f"[B,{HORIZON},29]",
                     f"W1-M4 · same ActionDiT, same data, same steps — objective swapped only "
                     f"[B,{HORIZON},29]"), fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# %% [markdown]
# ## 7. main

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W1-M4 실습 3: W1-M3 액션 헤드 재사용 + objective만 FM으로 (lesson §5.3·§7·§8)")
    p.add_argument("--menagerie", default=None, help="mujoco_menagerie 경로")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--horizon", type=int, default=HORIZON, help="청크 길이 H (기본 50 = pi0)")
    p.add_argument("--patch", type=int, default=PATCH_T)
    p.add_argument("--dim", type=int, default=128, help="hidden 차원 d")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--n-train", type=int, default=8192)
    p.add_argument("--n-eval", type=int, default=512, help="평가용 held-out 청크 수")
    p.add_argument("--steps", type=int, default=3000, help="objective마다 같은 스텝 수로 학습")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--n-proj", type=int, default=256)
    p.add_argument("--n-gen", type=int, default=4, help="다봉성 그림에서 같은 조건으로 뽑을 수")
    p.add_argument("--gen-nfe", type=int, default=10, help="다봉성 그림에 쓸 NFE (pi0의 10 steps)")
    p.add_argument("--no-clip", action="store_true", help="관절 한계 클리핑을 끈다")
    p.add_argument("--bench-reps", type=int, default=20)
    p.add_argument("--no-bench-xl", action="store_true", help="DiT-XL 급 벤치 행을 건너뛴다")
    p.add_argument("--smoke", action="store_true", help="1~2분 이내에 완주하는 축소 경로")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--ascii-labels", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    nfes = NFE_GRID_SMOKE if args.smoke else NFE_GRID
    ref_fm, ref_ddpm = REF_NFE_FM, REF_NFE_DDPM
    if args.smoke:
        args.steps, args.n_train, args.n_eval, args.bench_reps = 400, 2048, 256, 5
        args.no_bench_xl = True
        ref_fm, ref_ddpm = 50, 200
        print("[smoke] 축소 경로로 실행합니다 (400 스텝, DiT-XL 벤치 생략). "
              "생성 품질은 학습이 덜 된 값입니다.")
    setup_korean_font(args.ascii_labels)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = pick_device(args.device)
    out_dir = artifacts_dir()
    t_start = time.perf_counter()

    print("=" * 96)
    print(f"  {MODULE_ID} 실습 3 — W1-M3의 DiT 액션 헤드 재사용 · objective만 FM으로 교체")
    print("=" * 96)

    # --- [1] ★ 정말 세 줄인가 (lesson §5.3) ---------------------------------
    print("\n=== [1] ★ lesson §5.3 '코드 변경량: 세 줄' 검산 ===")
    diff, changed = objective_diff()
    for ln in diff:
        print("   " + ln)
    print(f"\n  → 바뀐 줄 **{changed}줄**"
          + ("  ✅ lesson §5.3의 '세 줄'과 일치" if changed == 3
             else f"  ⚠️ lesson은 세 줄이라고 적었습니다 (실측 {changed}줄)"))
    print("     ① 시간 샘플링(이산 → 연속) ② 보간식 ③ target(ε → x₁−x₀)")
    print("  ⓘ 정직하게 덧붙일 것이 하나 있습니다 — **연속 t의 임베딩 스케일(×1000)**입니다.")
    print("     위 diff에 안 나오는 이유는 ActionHead 래퍼 안에 넣었기 때문이고, 없으면 학습이")
    print("     안 됩니다. lesson §7.1이 세지 않은 '네 번째 줄'이며 실무에서 첫 지뢰입니다.")

    # --- [2] adaLN-Zero 항등 검증 (W1-M3의 assert를 그대로 다시) --------------
    print("\n=== [2] 재사용 확인 — W1-M3의 항등함수 assert 를 그대로 다시 돌린다 ===")
    n_tok = args.horizon // args.patch
    v = M3.verify_identity(args.dim, args.heads, B=4, T=n_tok, device=dev, seed=args.seed)
    print(f"  adaLN-Zero: max|block(h,c)−h| = {v['zero']:.3e}  "
          f"(랜덤 초기화 대조군 {v['random']:.3e})   ✅ 통과")
    print("  → objective를 바꿔도 **블록은 그대로**입니다. 이 assert가 그 사실의 증거입니다.")

    # --- [3] G1에서 D 읽기 ---------------------------------------------------
    print("\n=== [3] D = ? — G1 모델에서 직접 (lesson §7.2의 'D=29 ← W1-M2 실측') ===")
    spec = load_g1(M3.resolve_menagerie(args.menagerie))
    D = spec["nu"]
    lo, hi = spec["lo"], spec["hi"]
    print(f"  모델: {spec['source']}")
    print(f"  model.nu = {D}   " + ("✅ lesson §7.2의 D=29와 일치" if D == 29
                                    else f"⚠️ lesson은 29를 씁니다 (읽은 값 {D})"))
    print(f"  청크 규격 [B, H={args.horizon}, D={D}] → patchify p_t={args.patch} → 토큰 {n_tok}개"
          f"   (lesson §7.2 블록도의 [B,25,d])")
    print(f"  50 Hz 기준 청크 하나 = {args.horizon / FS_HZ:.2f}초 "
          f"— pi0의 H=50·f₂=50 Hz면 정확히 1초입니다(lesson §8.3)")

    # --- [4] 데이터 ----------------------------------------------------------
    print(f"\n=== [4] 데이터 — G1 관절 한계 안의 sin파 궤적 (W1-M3 make_chunks 재사용) ===")
    data_np = make_chunks(args.n_train + args.n_eval, args.horizon, D, args.seed)
    assert np.all(np.abs(data_np) <= 1.0 + 1e-6), "정규화 좌표를 벗어난 샘플이 있습니다"
    data = torch.as_tensor(data_np[:args.n_train], device=dev)
    real_eval = torch.as_tensor(data_np[args.n_train:], device=dev)     # held-out
    state_eval = real_eval[:, 0, :].contiguous()
    print(f"  학습 {tuple(data.shape)} · 평가(held-out) {tuple(real_eval.shape)}")
    smooth_real = float(real_eval.diff(dim=1).abs().mean())
    print(f"  원본 프레임간 |Δx| 평균 {smooth_real:.4f}  ← 생성이 이보다 크면 떨리는 궤적")

    sch = make_schedule(1000).to(dev)
    sch = Schedule(sch.T, *(getattr(sch, f).float() for f in
                            ("beta", "alpha", "abar", "abar_prev", "sqrt_abar",
                             "sqrt_one_minus_abar", "beta_tilde")))

    # --- [5] 두 헤드를 나란히 학습 -------------------------------------------
    print(f"\n=== [5] 학습 — 같은 백본·같은 데이터·같은 {args.steps} 스텝, objective만 다르게 ===")
    heads, losses_all = {}, {}
    kw = dict(horizon=args.horizon, action_dim=D, patch=args.patch, d=args.dim,
              depth=args.depth, n_heads=args.heads, state_dim=D)
    for obj, step_fn, tsc in (("ddpm", ddpm_train_step, 1.0), ("fm", fm_train_step, T_SCALE)):
        torch.manual_seed(args.seed)                 # 같은 초기 가중치에서 출발
        m = ActionHead(t_scale=tsc, **kw).to(dev)
        n_par = sum(p.numel() for p in m.parameters())
        t0 = time.perf_counter()
        ema, losses = train_head(m, data, sch, step_fn, steps=args.steps, batch=args.batch,
                                 lr=args.lr, ema_decay=args.ema,
                                 log_every=max(1, args.steps // 100), seed=args.seed)
        ema.copy_to(m)
        m.eval()
        heads[obj], losses_all[obj] = m, losses
        print(f"  {obj.upper():5s} 파라미터 {n_par:,}개 · 학습 {time.perf_counter() - t0:.1f}s · "
              f"손실 {losses[0]:.4f} → {losses[-1]:.4f}")
    print("  ⓘ 두 손실의 절대값을 비교하면 안 됩니다 — 재는 대상이 다릅니다(ε 예측 vs 속도 예측).")
    print("     비교는 아래 [7]의 생성 품질로 합니다.")

    # --- [6] forward 시간 벤치 (W1-M3 bench_forward 재사용) -------------------
    print(f"\n=== [6] 1회 forward 시간 — 두 objective가 **완전히 같은 값**을 씁니다 ===")
    cfgs = [(args.dim, args.depth, args.heads), (256, 6, 8), (512, 12, 8)]
    if not args.no_bench_xl:
        cfgs.append((1152, 28, 16))          # DiT-XL 구성 (673.7M · fp32 가중치만 2.7 GB)
    if dev.type == "cuda":
        # 방금 학습을 끝낸 직후라 캐싱 할당자가 블록을 잡고 있습니다. DiT-XL 행이 2.7 GB를
        # 새로 요구하므로 비워주지 않으면 첫 측정에 할당 비용이 섞여 값이 크게 뜁니다.
        torch.cuda.empty_cache()
    bench = M3.bench_forward(cfgs, D, dev, args.bench_reps, args.horizon, args.patch)
    rows = [[f"d={b['d']}, {b['depth']}블록, {b['heads']}헤드", f"{b['params'] / 1e6:.1f}M",
             f"{b['ms']:.2f}", f"{200 - 60:.0f}", f"{int((200 - 60) / b['ms'])}",
             "← 이 실습" if (b["d"], b["depth"]) == (args.dim, args.depth)
             else ("← DiT-XL 구성" if b["d"] == 1152 else "")]
            for b in bench]
    print_table(["구성", "파라미터", "1 forward [ms]", "L4 5Hz 헤드 예산 [ms]", "들어가는 NFE", ""],
                rows, ["left", "right", "right", "right", "right", "left"])
    ms_per_nfe = float(next(b["ms"] for b in bench
                            if (b["d"], b["depth"]) == (args.dim, args.depth)))
    print(f"  측정: 배치 1, 토큰 {n_tok}개, {dev.type}, {args.bench_reps}회 평균")
    print("  → **objective는 이 표를 바꾸지 못합니다.** FM이 바꾸는 것은 '몇 번 부르는가'뿐입니다"
          " (lesson 「흔한 오해」 3번).")
    print("  ⚠️ 배치 1·토큰 수십 개는 커널 실행 오버헤드가 연산량을 지배하는 영역이라 "
          "실행마다 최대 2배까지 흔들립니다.")
    print("     **판정은 절대값이 아니라 세로 방향 배율로 하세요**(모델이 커질 때 몇 배가 되는가). "
          "W1-M3 03과 같은 주의사항입니다.")
    if dev.type == "cpu":
        print("  ⚠️ CPU 측정값입니다. 실제 배치는 GPU이므로 이 숫자를 그대로 예산에 넣지 마세요.")
    with open(out_dir / "03_forward_bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["d", "depth", "heads", "params", "ms"])
        w.writeheader(), w.writerows(bench)
    print(f"  [저장] {out_dir / '03_forward_bench.csv'}")

    # --- [7] NFE 스윕 --------------------------------------------------------
    print(f"\n=== [7] NFE 스윕 — held-out {args.n_eval}개 시작 포즈로 조건부 생성 ===")
    clip = not args.no_clip
    ref = {
        "fm": euler_sample_chunks(heads["fm"], state_eval, ref_fm, seed=args.seed + 7,
                                  clip_x1=clip),
        "ddpm": M3.sample_chunks(heads["ddpm"], state_eval, sch, ref_ddpm, seed=args.seed + 7,
                                 clip_x0=clip),
    }
    print(f"  적분 오차의 기준: FM NFE={ref_fm} · DDPM NFE={ref_ddpm} (같은 seed)")
    res = []
    for obj in ("ddpm", "fm"):
        for nfe in nfes:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            x = (euler_sample_chunks(heads["fm"], state_eval, nfe, seed=args.seed + 7, clip_x1=clip)
                 if obj == "fm" else
                 M3.sample_chunks(heads["ddpm"], state_eval, sch, nfe, seed=args.seed + 7,
                                  clip_x0=clip))
            if dev.type == "cuda":
                torch.cuda.synchronize()
            sec = time.perf_counter() - t0
            met = eval_chunks(x, real_eval, ref[obj], lo, hi, args.n_proj, args.seed)
            res.append(dict(objective=obj, nfe=nfe, sec_batch=sec,
                            ms_per_nfe_batch1=ms_per_nfe, **met))
    rows = []
    for nfe in nfes:
        a = next(d for d in res if d["objective"] == "ddpm" and d["nfe"] == nfe)
        b = next(d for d in res if d["objective"] == "fm" and d["nfe"] == nfe)
        rows.append([str(nfe), f"{a['energy']:.4f}", f"{b['energy']:.4f}",
                     f"{a['energy'] / max(b['energy'], 1e-12):.2f}배",
                     f"{a['self_dev']:.4f}", f"{b['self_dev']:.4f}",
                     f"{a['start_deg']:.2f}°", f"{b['start_deg']:.2f}°",
                     f"{nfe * ms_per_nfe:.1f}"])
    print_table(["NFE", "energy DDPM", "energy FM", "DDPM/FM",
                 "적분오차 DDPM", "적분오차 FM", "시작포즈 DDPM", "시작포즈 FM", "헤드 [ms]"],
                rows, ["right"] * 9)
    print(f"  ⓘ '헤드 [ms]' = NFE × {ms_per_nfe:.2f} ms(배치 1 실측). "
          "표의 sec_batch 열은 배치 전체라 다른 값입니다.")
    print(f"  ⓘ 관절 한계 클리핑: {'켬' if clip else '끔 (--no-clip)'} — "
          "FM 쪽은 x̂₁=x_t+(1−t)·v 를 자릅니다(§4의 수식).")
    for obj in ("ddpm", "fm"):
        r = [d for d in res if d["objective"] == obj]
        best = min(r, key=lambda d: d["energy"])
        print(f"  {obj.upper():5s} 최저 energy NFE={best['nfe']} ({best['energy']:.4f}), "
              f"|x|>1 비율 {best['viol_pct']:.2f}%, 프레임간 |Δx| {best['smooth']:.4f} "
              f"(원본 {smooth_real:.4f})")
    # "같은 품질을 몇 NFE로 내는가" — 지연 예산으로 바로 번역되는 숫자입니다.
    print("\n  같은 품질에 필요한 NFE (기준 = DDPM이 가장 큰 NFE에서 낸 energy):")
    thr = next(d for d in res if d["objective"] == "ddpm" and d["nfe"] == nfes[-1])["energy"]
    cells = {}
    for obj in ("ddpm", "fm"):
        ok = [d for d in res if d["objective"] == obj and d["energy"] <= thr]
        cells[obj] = min(ok, key=lambda d: d["nfe"]) if ok else None
    a, b = cells["ddpm"], cells["fm"]
    print(f"    기준 energy ≤ {thr:.4f}:  DDPM NFE={a['nfe'] if a else '미달'}"
          f" ({a['nfe'] * ms_per_nfe:.0f} ms)  ·  FM NFE={b['nfe'] if b else '미달'}"
          + (f" ({b['nfe'] * ms_per_nfe:.0f} ms)" if b else "")
          + (f"   → **{a['nfe'] / b['nfe']:.1f}배 적은 NFE**" if a and b else ""))
    print("     이 배율이 lesson §8.1 표의 '헤드 시간' 열에 그대로 들어가는 숫자입니다.")

    # --- [8] lesson §8.2 표에 자기 기기 실측 ms 넣기 --------------------------
    print(f"\n=== [8] lesson §8.2 자리별 NFE 상한 — 실측 {ms_per_nfe:.2f} ms/NFE 대입 ===")
    budget_rows, budget_csv = [], []
    for name, period, vlm in BUDGET_SLOTS:
        head_ms = period - vlm
        n_meas = int(head_ms // ms_per_nfe)
        budget_rows.append([name, f"{period:.0f}", f"{head_ms:.0f}",
                            f"{int(head_ms // 3.0)}", f"{int(head_ms // 13.2)}", f"**{n_meas}**",
                            "✓" if n_meas >= 10 else "✗", "✓" if n_meas >= 4 else "✗"])
        budget_csv.append(dict(slot=name, period_ms=period, vlm_ms=vlm, head_budget_ms=head_ms,
                               ms_per_nfe=ms_per_nfe, nfe_max=n_meas,
                               pi0_k10_fits=int(n_meas >= 10), groot_k4_fits=int(n_meas >= 4)))
    print_table(["자리", "주기 예산", "헤드 예산", "3 ms 가정", "13.2 ms(§8.2)",
                 f"{ms_per_nfe:.2f} ms 실측", "pi0 K=10", "GR00T K=4"],
                budget_rows, ["left"] + ["right"] * 6 + ["center"])
    print("  → 3 ms·13.2 ms 열은 lesson §8.2가 인쇄한 값이고, 마지막 세 열이 이 기기의 실측입니다.")
    print(f"     이 헤드는 {sum(p.numel() for p in heads['fm'].parameters()) / 1e6:.2f}M이라 "
          "pi0의 300M 액션 전문가보다 훨씬 작습니다 — 실측 상한을 그대로 예산에 넣지 마세요.")
    with open(out_dir / "03_nfe_budget.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(budget_csv[0].keys()))
        w.writeheader(), w.writerows(budget_csv)
    print(f"  [저장] {out_dir / '03_nfe_budget.csv'}")

    # --- [9] 산출물 ----------------------------------------------------------
    with open(out_dir / "03_fm_vs_ddpm.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["objective", "nfe", "energy", "sw1", "self_dev",
                                          "start_deg", "viol_pct", "smooth", "sec_batch",
                                          "ms_per_nfe_batch1"])
        w.writeheader(), w.writerows(res)
    print(f"  [저장] {out_dir / '03_fm_vs_ddpm.csv'}")

    if not args.no_plot:
        p = plot_compare(res, out_dir / "03_fm_vs_ddpm.png", ms_per_nfe)
        print(f"  [저장] {p}")
        # 같은 시작 포즈에서 여러 개 — 다봉성 그림. W1-M3의 plot_chunks 를 그대로 씁니다.
        st = real_eval[:1, 0, :].repeat(args.n_gen, 1)
        gen = euler_sample_chunks(heads["fm"], st, args.gen_nfe, seed=args.seed + 9, clip_x1=clip)
        rng = np.random.default_rng(args.seed)
        joints = sorted(rng.choice(D, size=3, replace=False).tolist())
        p = M3.plot_chunks(real_eval[:1].cpu().numpy(), gen.cpu().numpy(),
                           real_eval[0, 0, :].cpu().numpy(), losses_all["fm"], spec, joints,
                           out_dir / "03_fm_chunks.png", args.gen_nfe,
                           max(1, args.steps // 100))
        print(f"  [저장] {p}   ← W1-M3 plot_chunks() 재사용. NFE={args.gen_nfe}"
              " (pi0의 10 steps)로 뽑았습니다")
        print("     ⓘ 그림 제목이 'W1-M3 · DDIM'으로 찍히는 것은 **재사용한 함수가 자기 제목을**")
        print("        **박기 때문**입니다. 실제 내용은 FM 헤드를 Euler로 뽑은 것입니다. 고치려면")
        print("        plot_chunks()를 복사해 와야 하고, 그러면 [10]의 재사용 회계에서 63줄이")
        print("        새 코드로 옮겨갑니다 — 재사용의 소소한 대가라 그대로 두었습니다.")

    # --- [10] 재사용 회계 ----------------------------------------------------
    print("\n=== [10] 재사용 회계 — 무엇을 가져왔고 무엇을 새로 썼는가 ===")

    def nlines(obj) -> int:
        src = source_of(obj)
        return len(src.splitlines()) if src.strip() else 0

    reused_rows, reused_total = [], 0
    for name, obj in REUSED.items():
        n = nlines(obj)
        reused_total += n
        reused_rows.append([name, f"{n}줄", obj.__module__])
    print_table(["재사용한 것 (W1-M3에서)", "줄 수", "출처 모듈"], reused_rows,
                ["left", "right", "left"])
    new_objs = {"ActionHead (시간 규약 래퍼)": ActionHead,
                "ddpm_train_step (대조군)": ddpm_train_step,
                "fm_train_step (★ 이 모듈의 본체)": fm_train_step,
                "train_head (step_fn만 갈아끼우는 공용 루프)": train_head,
                "euler_sample_chunks (FM 샘플러)": euler_sample_chunks,
                "eval_chunks (평가 지표)": eval_chunks}
    new_rows, new_total = [], 0
    for name, obj in new_objs.items():
        n = nlines(obj)
        new_total += n
        new_rows.append([name, f"{n}줄"])
    print_table(["새로 쓴 것", "줄 수"], new_rows, ["left", "right"])
    print(f"\n  재사용 {reused_total}줄 · 새로 쓴 핵심 코드 {new_total}줄 "
          f"(비율 {new_total / max(reused_total, 1) * 100:.0f}%)")
    print(f"  그중 **objective 자체는 {nlines(fm_train_step)}줄이고 diff는 {changed}줄**입니다.")
    print("  → lesson §5.3의 '백본·옵티마이저·조건 주입 경로는 한 줄도 안 바뀐다'가 이 회계입니다.")

    print(f"\n총 소요 {time.perf_counter() - t_start:.1f}s")
    print("다음: labs/README.md — 이 숫자들로 lesson §8.1·§8.2 표를 자기 손으로 다시 계산하세요")


if __name__ == "__main__":
    import sys

    # 노트북(ipykernel)에서는 argparse가 jupyter의 -f 인자를 먹지 않도록 빈 리스트를 넘긴다.
    main(None if "ipykernel" not in sys.modules else [])
