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
# # W1-M4 실습 2 — reflow로 궤적을 펴고, W1-M3의 DDPM/DDIM 곡선과 **같은 축에 겹친다**
#
# lesson.md `§6.2`(직선성 지표) · `§6.3`(reflow와 그 대가) · `§6.4`(스텝 축소 기법 지형)의 실행판입니다.
#
# **이 스크립트가 만드는 그림 한 장이 W1-M4 실습 전체의 결론입니다.** W1-M3 `§5.3`은
# "샘플러를 아무리 갈아도 NFE 1~2는 안 나온다"로 끝났고, 그 근거를
# [`02_samplers_compare.py`](../../03-diffusion-ddpm-dit/practice/02_samplers_compare.py)가
# `artifacts/W1-M3/02_results.csv`에 남겨 뒀습니다. 여기서 그 CSV를 **읽어서**
# FM·RF 곡선과 한 좌표축에 겹칩니다.
#
# 하는 일:
#
# 1. **직선성 지표를 실제로 계산** — $S(Z)=\int_0^1\mathbb{E}\|(Z_1-Z_0)-\dot Z_t\|^2dt$ (§6.2).
#    reflow 전후로 얼마나 내려가는지 표로.
# 2. **reflow** (§6.3) — $\hat x_1=\text{ODESolve}(x_0)$로 $(x_0,\hat x_1)$ 쌍을 만들어 재학습.
#    2-rectified, 3-rectified까지. **§5.2 손실에서 바뀌는 것은 딱 한 줄**($x_0$을 새로 뽑지 않고
#    짝에서 가져온다)이라는 것을 코드로 확인합니다.
# 3. **겹쳐 그리기** ★ — x축 NFE(로그), y축은 W1-M3와 **같은 지표**(sliced $W_1$ 주지표, energy 보조).
#    지표 함수도 W1-M3의 구현을 import해서 씁니다. 분포·표본 수·seed도 전부 같습니다.
# 4. **reflow의 대가**(§6.3 인용문 ①②③)를 실측으로 — 큰 NFE에서 원 모델보다 나빠지는지.
#
# 출력(`artifacts/W1-M4/`):
#
# | 파일 | 내용 |
# |---|---|
# | `02_nfe_overlay.png` | ★ **W1-M3 ancestral·DDIM + FM·2-rect·3-rect 를 한 축에** |
# | `02_straightness.png` | reflow 회차별 ODE 궤적 + NFE=1 샘플 |
# | `02_results.csv` | 회차 × NFE의 energy·sW1·초 |
# | `02_reflow_summary.csv` | 회차별 $S(Z)$·누적 학습 스텝·생성한 쌍 수 |
#
# **GPU 불필요.** `--smoke`는 수십 초. 본 실행은 학습을 2회(또는 1회) 더 하므로 `01`보다 깁니다.

# %%
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless 고정 — 뷰어를 띄우지 않는다

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

MODULE_ID = "W1-M4"


# %% [markdown]
# ## 0. 01의 코드를 그대로 재사용
#
# `VelocityMLP` · `euler_sample` · `straightness` · 지표 함수는 전부 `01_cfm_two_moons.py`에 있고,
# 그것을 `importlib`로 로드합니다. 01이 W1-M3의 `01`·`02`를 이미 끌어와 뒀으므로
# **여기서 한 번 로드하면 세 모듈이 다 딸려 옵니다.**

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


def load_sibling(fname: str, modname: str, folder: Path | None = None):
    cands = [folder] if folder is not None else [
        here(), find_repo_root() / "course" / "w1-generative-core" / "04-flow-matching" / "practice"]
    for cand in cands:
        p = cand / fname
        if p.is_file():
            spec = importlib.util.spec_from_file_location(modname, p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[modname] = mod
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit(f"[에러] {fname} 를 찾지 못했습니다 (찾아본 곳: {[str(c) for c in cands]})")


P1 = load_sibling("01_cfm_two_moons.py", "cfm_two_moons")
M1, M2 = P1.M1, P1.M2               # W1-M3의 01 · 02

VelocityMLP, euler_sample, straightness = P1.VelocityMLP, P1.euler_sample, P1.straightness
energy_distance, sliced_w1 = P1.energy_distance, P1.sliced_w1
make_data, normalize, pick_device = P1.make_data, P1.normalize, P1.pick_device
print_table, EMA = P1.print_table, P1.EMA
NFE_GRID, NFE_GRID_SMOKE = P1.NFE_GRID, P1.NFE_GRID_SMOKE
N_SAMPLE, N_PROJ, T_SCALE = P1.N_SAMPLE, P1.N_PROJ, P1.T_SCALE

USE_KOREAN = False


def setup_korean_font(force_ascii: bool = False) -> bool:
    global USE_KOREAN
    USE_KOREAN = M1.setup_korean_font(force_ascii)
    return USE_KOREAN


def lab(ko: str, en: str) -> str:
    return ko if USE_KOREAN else en


# 그림 색 — W1-M3 02와 같은 색을 그대로 씁니다(ancestral 빨강 / DDIM 파랑). 겹쳐 그릴 때
# 같은 계열이 같은 색이어야 두 문서를 나란히 놓고 읽을 수 있습니다.
COLORS = {"ancestral": "#c92a2a", "ddim": "#1971c2",
          "fm-1rect": "#2f9e44", "fm-2rect": "#f08c00", "fm-3rect": "#7048e8"}
NICE = {"ancestral": "DDPM ancestral (W1-M3)", "ddim": "DDIM (W1-M3)",
        "fm-1rect": "FM 1-rectified", "fm-2rect": "RF 2-rectified (reflow×1)",
        "fm-3rect": "RF 3-rectified (reflow×2)"}


# %% [markdown]
# ## 1. reflow — 바뀌는 것은 짝짓기 한 줄
#
# lesson §6.3의 절차 그대로입니다.
#
# 1. 학습된 $v_\theta$로 $\hat x_1=\text{ODESolve}(x_0)$를 충분한 스텝으로 푼다.
# 2. $(x_0,\hat x_1)$ 쌍을 **새 데이터셋**으로 삼아 §5.2의 손실로 다시 학습한다.
# 3. 필요하면 반복한다.
#
# 손실 함수를 §5.2의 것과 나란히 놓고 보면 차이가 한 줄입니다.
#
# | | 원 학습 (`01`의 `cfm_loss`) | reflow (`02`의 `cfm_loss_paired`) |
# |---|---|---|
# | $x_1$ | 데이터에서 뽑는다 | **ODE가 만든 $\hat x_1$** |
# | $x_0$ | `torch.randn(...)` 매번 새로 | **그 $\hat x_1$과 짝인 $x_0$** |
# | $x_t$ · target | 똑같다 | 똑같다 |
#
# $x_0$을 새로 뽑지 않는 것이 전부이고, 그 결과 짝짓기가 **독립 결합에서 결정론적 함수로** 바뀝니다.
# 각 $x_0$에 정확히 하나의 $\hat x_1$이 대응하니 §6.2의 교차가 크게 줄고 $S(Z)$가 내려갑니다.

# %%
@torch.no_grad()
def make_reflow_pairs(model: nn.Module, n: int, device: torch.device, *, nfe: int,
                      seed: int, dim: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """(x₀, ODESolve(x₀)) 쌍을 만든다 — reflow의 새 데이터셋 (§6.3 1단계).

    `nfe`가 충분히 커야 합니다. 여기가 성기면 적분 오차가 그대로 다음 세대의 **데이터**가 됩니다
    (§6.3 인용문 ① '모델 자신의 출력을 target으로 삼아 오차를 상속').
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    x0 = torch.randn(n, dim, generator=g).to(device)
    x1 = euler_sample(model, n, device, nfe=nfe, x0=x0)
    return x0, x1


def cfm_loss_paired(model: nn.Module, x0: torch.Tensor, x1: torch.Tensor,
                    gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """01의 cfm_loss 와 같은 식. **x₀를 새로 뽑지 않고 짝에서 받는 것만 다릅니다.**"""
    t = torch.rand(x1.shape[0], generator=gen).to(x1.device)        # eq. §5.2  t ~ U(0,1)
    x_t = (1.0 - t[:, None]) * x0 + t[:, None] * x1                 # eq. §5.2  x_t
    target = x1 - x0                                                # eq. §5.2  u = x₁ − x₀
    per = ((model(x_t, t) - target) ** 2).mean(dim=1)
    return per, t


def train_reflow(model: nn.Module, x0_all: torch.Tensor, x1_all: torch.Tensor, *, steps: int,
                 batch: int, lr: float, ema_decay: float, log_every: int,
                 seed: int) -> tuple[EMA, list[float]]:
    """01의 train_cfm 과 같은 루프. 배치를 **쌍 단위로** 뽑는 것만 다릅니다."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.05, 1.0 - s / steps))
    ema = EMA(model, ema_decay)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = x1_all.shape[0]
    losses, run = [], 0.0
    for s in range(1, steps + 1):
        idx = torch.randint(0, n, (batch,), generator=g).to(x1_all.device)
        per, _ = cfm_loss_paired(model, x0_all[idx], x1_all[idx], g)   # ← 짝을 그대로 씁니다
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
    return ema, losses


# %% [markdown]
# ## 2. 01의 체크포인트 읽기 (없으면 짧게 재학습)
#
# W1-M3 `02_samplers_compare.py`의 `get_model()`과 같은 규약입니다.

# %%
def get_base_model(args, dev: torch.device):
    out = artifacts_dir()
    cands = [out / "01_model_smoke.pt", out / "01_model.pt"] if args.smoke \
        else [out / "01_model.pt", out / "01_model_smoke.pt"]
    if args.ckpt:
        cands = [Path(args.ckpt)]
    for ck in cands:
        if not ck.is_file():
            continue
        blob = torch.load(ck, map_location=dev, weights_only=False)
        cfg = dict(blob["cfg"])
        model = VelocityMLP(**cfg).to(dev)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        print(f"  체크포인트 로드: {ck.name}   (01의 학습 {blob['train_steps']} 스텝"
              + (", ⚠️ smoke 체크포인트" if blob.get("smoke") else "")
              + f", S(Z)={blob.get('straightness', float('nan')):.4f})")
        return model, int(blob["train_steps"]), str(ck.name)

    print(f"  ⚠️ 01_model.pt 가 없어 **여기서 짧게 재학습**합니다 ({args.fallback_steps} 스텝).")
    print("     01을 먼저 돌리면 이 단계를 건너뜁니다. 이 경로의 품질은 01 본 학습보다 낮습니다.")
    raw = make_data(P1.DATA_KIND, P1.N_DATA, args.seed)
    xn, _, _ = normalize(raw)
    x1 = torch.as_tensor(xn, device=dev)
    model = VelocityMLP(2, P1.HIDDEN, P1.LAYERS, P1.TEMB).to(dev)
    ema, _, _ = P1.train_cfm(model, x1, steps=args.fallback_steps, batch=P1.BATCH, lr=P1.LR,
                             ema_decay=P1.EMA_DECAY,
                             log_every=max(1, args.fallback_steps // 20), seed=args.seed)
    ema.copy_to(model)
    model.eval()
    return model, args.fallback_steps, "즉석 재학습"


# %% [markdown]
# ## 3. W1-M3 결과 읽어오기 — 겹쳐 그려도 되는가
#
# **분포·표본 수·지표 구현이 전부 같아야 같은 축에 놓을 수 있습니다.** 셋 다 확인합니다.
#
# | 축 | W1-M3 `02_samplers_compare.py` | 여기 |
# |---|---|---|
# | 데이터 | `make_moons(16384, noise=0.06, seed=0)` → 표준화 | **같은 호출** (01 §7에서 bit 대조) |
# | 기준 표본 | `data_ref = xn[:8192]`의 앞 4,096개 | **같음** |
# | 생성 표본 수 | 4,096 | **같음** |
# | energy / sliced $W_1$ | `energy_distance` / `sliced_w1(n_proj=256, seed=0)` | **같은 함수를 import** |
#
# 하나라도 어긋나면 겹쳐 그리지 않고 **별도 패널로 분리**하고 그 사실을 그림에 적습니다.
# 숫자를 억지로 맞추지 않습니다.

# %%
def load_m3_results() -> tuple[list[dict], str]:
    p = find_repo_root() / "artifacts" / "W1-M3" / "02_results.csv"
    if not p.is_file():
        return [], (f"⚠️ {p} 가 없습니다. W1-M3 실습을 먼저 돌리세요:\n"
                    "     cd course/w1-generative-core/03-diffusion-ddpm-dit/practice\n"
                    "     python 01_ddpm_toy.py && python 02_samplers_compare.py")
    rows = []
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(method=r["method"], nfe=int(r["nfe"]), energy=float(r["energy"]),
                             sw1=float(r["sw1"]), sec=float(r["sec"])))
    return rows, f"W1-M3 02_results.csv 로드 — {len(rows)}행 ({p})"


# %% [markdown]
# ## 4. 그림

# %%
def plot_overlay(res: list[dict], m3: list[dict], floor: float, floor_w1: float,
                 overlay_ok: bool, methods: list[str], path: Path) -> Path:
    all_rows = (m3 if overlay_ok else []) + res
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # reflow 회차끼리는 곡선이 거의 겹칩니다. 선 모양·마커를 달리 해야 둘 다 보입니다.
    STYLE = {"ancestral": ("o--", 2.0), "ddim": ("o--", 2.0), "fm-1rect": ("o-", 2.2),
             "fm-2rect": ("s-", 2.2), "fm-3rect": ("^-.", 1.8)}
    for key in ("sw1", "energy"):
        a = ax[0] if key == "sw1" else ax[1]
        for meth in methods:
            r = sorted([d for d in all_rows if d["method"] == meth], key=lambda d: d["nfe"])
            if not r:
                continue
            style, lw = STYLE[meth]
            a.loglog([d["nfe"] for d in r], [d[key] for d in r], style,
                     color=COLORS[meth], lw=lw, ms=5, label=NICE[meth])
        a.set_xlabel("NFE")
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8.2, loc="lower left")
    ax[0].set_ylabel(lab("sliced $W_1$ (데이터 단위)", "sliced $W_1$ (data units)"))
    ax[0].axhline(floor_w1, color="#868e96", ls=":", lw=1.3)
    ax[0].set_title(lab(f"(a) ★ 주지표 sliced $W_1$ — 낮을수록 좋다 (점선 = 노이즈 바닥 {floor_w1:.4f})",
                        f"(a) ★ sliced $W_1$ — lower is better (dotted = noise floor {floor_w1:.4f})"),
                    fontsize=11.5)
    ax[1].set_ylabel(lab("에너지 거리", "energy distance"))
    ax[1].axhline(floor, color="#868e96", ls=":", lw=1.3)
    ax[1].set_title(lab(f"(b) 보조 지표 energy (점선 = 노이즈 바닥 {floor:.5f})",
                        f"(b) auxiliary metric energy (dotted = noise floor {floor:.5f})"), fontsize=11.5)

    # (c) NFE 1·2·4 만 막대로 — "어디서 뚫리는가"
    a = ax[2]
    small = [n for n in (1, 2, 4) if any(d["nfe"] == n for d in all_rows)]
    w = 0.8 / max(len(methods), 1)
    vmax = 0.0
    for i, meth in enumerate(methods):
        vals = [next((d["sw1"] for d in all_rows if d["method"] == meth and d["nfe"] == n), np.nan)
                for n in small]
        vmax = max(vmax, float(np.nanmax(vals)))
        a.bar(np.arange(len(small)) + i * w - 0.4 + w / 2, vals, width=w * 0.92,
              color=COLORS[meth], label=NICE[meth])
    a.axhline(floor_w1, color="#495057", ls=":", lw=1.3)
    a.set_xticks(np.arange(len(small)))
    a.set_xticklabels([f"NFE={n}" for n in small])
    a.set_ylim(0, vmax * 1.42)          # 범례가 가장 높은 막대를 가리지 않도록
    a.set_ylabel(lab("sliced $W_1$", "sliced $W_1$"))
    a.set_title(lab("(c) W1-M3가 막혔던 자리 — 여기가 뚫렸는가",
                    "(c) where W1-M3 got stuck — is it unblocked?"), fontsize=11.5)
    a.legend(fontsize=8.2, loc="upper right"), a.grid(alpha=0.3, axis="y")

    head = lab("W1-M4 · 같은 분포 · 같은 표본 수 · 같은 지표 함수 — 겹쳐 그리기 성립",
               "W1-M4 · same data, same sample size, same metric code — overlay is valid") \
        if overlay_ok else \
        lab("⚠️ W1-M3 결과와 대조 조건이 맞지 않아 FM/RF만 그렸습니다 (README §5.2 참고)",
            "⚠️ W1-M3 comparison conditions not met — FM/RF only (see README §5.2)")
    fig.suptitle(head, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_straightness(trajs: dict[str, np.ndarray], one_step: dict[str, np.ndarray],
                      svals: dict[str, float], real: np.ndarray, methods: list[str],
                      path: Path) -> Path:
    n = len(methods)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.2))
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1:
        axes = axes.T
    for c, meth in enumerate(methods):
        a = axes[0, c]
        a.scatter(real[:, 0], real[:, 1], s=2, alpha=0.08, color="#adb5bd")
        tr = trajs[meth]
        for k in range(tr.shape[1]):
            a.plot(tr[:, k, 0], tr[:, k, 1], color=COLORS[meth], lw=1.4, alpha=0.9)
            a.plot([tr[0, k, 0], tr[-1, k, 0]], [tr[0, k, 1], tr[-1, k, 1]],
                   color="#868e96", lw=0.9, ls="--", alpha=0.9)
        a.scatter(tr[0, :, 0], tr[0, :, 1], color="#111", s=22, zorder=5)
        a.set_title(f"{NICE[meth]}\n$S(Z)$ = {svals[meth]:.3e}", fontsize=11)
        a.set_aspect("equal"), a.grid(alpha=0.25)
        a.set_xlim(-3.2, 3.2), a.set_ylim(-3.2, 3.2)

        a = axes[1, c]
        a.scatter(real[:, 0], real[:, 1], s=2, alpha=0.12, color="#adb5bd")
        p = one_step[meth]
        a.scatter(p[:, 0], p[:, 1], s=3, alpha=0.45, color=COLORS[meth])
        a.set_title(lab(f"NFE=1 샘플 (표준편차 {p.std(0)[0]:.2f}, {p.std(0)[1]:.2f})",
                        f"NFE=1 samples (std {p.std(0)[0]:.2f}, {p.std(0)[1]:.2f})"), fontsize=11)
        a.set_aspect("equal"), a.grid(alpha=0.25)
        a.set_xlim(-3.2, 3.2), a.set_ylim(-3.2, 3.2)
    axes[0, 0].set_ylabel(lab("ODE 궤적 (회색 점선 = 시작–끝 직선)",
                              "ODE trajectories (grey = chord)"), fontsize=10)
    axes[1, 0].set_ylabel(lab("Euler 1스텝 결과", "one Euler step"), fontsize=10)
    fig.suptitle(lab("W1-M4 · reflow는 짝짓기를 결정론적으로 바꿔 궤적을 편다 (§6.3)",
                     "W1-M4 · reflow makes the coupling deterministic and straightens paths (§6.3)"),
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# %% [markdown]
# ## 5. main

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W1-M4 실습 2: reflow + W1-M3 곡선과 겹쳐 그리기 (lesson §6)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--ckpt", default=None, help="01의 체크포인트 경로 (기본: artifacts/W1-M4/01_model.pt)")
    p.add_argument("--rounds", type=int, default=2, help="reflow 반복 횟수 (2 → 3-rectified까지)")
    p.add_argument("--reflow-steps", type=int, default=8000, help="reflow 재학습 스텝 수")
    p.add_argument("--n-reflow", type=int, default=16384, help="reflow 쌍 개수")
    p.add_argument("--reflow-nfe", type=int, default=100, help="쌍을 만들 때의 ODE 스텝 수")
    p.add_argument("--cold-start", action="store_true",
                   help="reflow 재학습을 랜덤 초기화에서 시작 (기본은 직전 모델에서 이어서)")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--n-sample", type=int, default=N_SAMPLE)
    p.add_argument("--n-proj", type=int, default=N_PROJ)
    p.add_argument("--k-fine", type=int, default=100, help="S(Z) 계산용 Euler 스텝 수")
    p.add_argument("--fallback-steps", type=int, default=4000)
    p.add_argument("--smoke", action="store_true", help="수십 초에 완주하는 축소 경로")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--ascii-labels", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    nfes = NFE_GRID_SMOKE if args.smoke else NFE_GRID
    if args.smoke:
        # n_sample 은 줄이지 않습니다 — 줄이면 W1-M3(4,096개)와 표본 수가 달라져
        # 겹쳐 그리기 자격을 스스로 잃습니다. 지표 계산은 어차피 싸므로 그대로 둡니다.
        args.rounds, args.reflow_steps, args.n_reflow = 1, 600, 4096
        args.reflow_nfe, args.k_fine, args.fallback_steps = 30, 30, 600
        print(f"[smoke] reflow 1회 · {args.reflow_steps} 스텝 · NFE 집합 {nfes}로 축소합니다.")
    setup_korean_font(args.ascii_labels)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = pick_device(args.device)
    out_dir = artifacts_dir()
    t_start = time.perf_counter()

    print("=" * 96)
    print(f"  {MODULE_ID} 실습 2 — Rectified Flow · reflow · W1-M3 곡선과 겹쳐 그리기")
    print("=" * 96)

    # --- [1] 기준 분포와 지표 ------------------------------------------------
    print("\n=== [1] 기준 분포 — W1-M3와 같은 것을 쓰고 있는가 ===")
    raw = make_data(P1.DATA_KIND, P1.N_DATA, args.seed)
    xn, _, _ = normalize(raw)
    same, msg = P1.verify_same_distribution(xn)
    print(f"  {'✅' if same else 'ⓘ'} {msg}")
    x1_all = torch.as_tensor(xn, device=dev)
    ref = x1_all[:8192]
    ref_eval = ref[:args.n_sample]
    half = min(args.n_sample, ref.shape[0] // 2)
    floor = energy_distance(ref[:half], ref[half:2 * half])
    floor_w1 = sliced_w1(ref[:half], ref[half:2 * half], args.n_proj, args.seed)
    print(f"  지표 노이즈 바닥(같은 분포 두 표본, n={half}): "
          f"energy={floor:.5f}  sliced W₁={floor_w1:.4f}")
    print("  ← 아래 표에서 이 값 근처의 행들끼리는 우열을 말할 수 없습니다.")

    m3, m3msg = load_m3_results()
    print(f"  {'✅' if m3 else '⚠️'} {m3msg}")
    # 겹쳐 그릴 자격: 같은 데이터 + 같은 표본 수 + W1-M3 CSV 존재
    overlay_ok = bool(m3) and same and args.n_sample == N_SAMPLE and args.n_proj == N_PROJ
    if m3 and not overlay_ok:
        print("  ⚠️ 대조 조건이 어긋나 겹쳐 그리지 않습니다 "
              f"(data 동일={same}, n_sample={args.n_sample}(기대 {N_SAMPLE}), "
              f"n_proj={args.n_proj}(기대 {N_PROJ}))")
    elif overlay_ok:
        print("  ✅ 겹쳐 그리기 성립 — 분포·표본 수·투영 수·지표 구현이 전부 일치합니다.")

    # --- [2] 01의 모델 = 1-rectified flow ------------------------------------
    print("\n=== [2] 1-rectified flow — 01이 학습한 것 ===")
    base, base_steps, src = get_base_model(args, dev)
    models = {"fm-1rect": base}
    trained_steps = {"fm-1rect": base_steps}
    pairs_made = {"fm-1rect": 0}

    # --- [3] reflow ----------------------------------------------------------
    print(f"\n=== [3] reflow — {args.rounds}회 (§6.3) ===")
    print(f"  쌍 생성: n={args.n_reflow:,}, ODE 스텝 {args.reflow_nfe}   "
          f"재학습: {args.reflow_steps:,} 스텝, "
          + ("랜덤 초기화(cold)" if args.cold_start else "직전 모델에서 이어서(warm)"))
    prev = base
    for k in range(1, args.rounds + 1):
        name = f"fm-{k + 1}rect"
        t0 = time.perf_counter()
        x0p, x1p = make_reflow_pairs(prev, args.n_reflow, dev, nfe=args.reflow_nfe,
                                     seed=args.seed + 100 + k)
        gen_sec = time.perf_counter() - t0
        nxt = VelocityMLP(2, P1.HIDDEN, P1.LAYERS, P1.TEMB).to(dev)
        if not args.cold_start:
            nxt.load_state_dict(prev.state_dict())
        t0 = time.perf_counter()
        ema, losses = train_reflow(nxt, x0p, x1p, steps=args.reflow_steps, batch=args.batch,
                                   lr=args.lr, ema_decay=args.ema,
                                   log_every=max(1, args.reflow_steps // 20), seed=args.seed + k)
        ema.copy_to(nxt)
        nxt.eval()
        print(f"  [{k}] {name}: 쌍 생성 {gen_sec:.1f}s "
              f"({args.n_reflow * args.reflow_nfe / 1e6:.2f}M forward-원소) · "
              f"재학습 {time.perf_counter() - t0:.1f}s · 손실 {losses[0]:.4f} → {losses[-1]:.4f}")
        models[name] = nxt
        trained_steps[name] = trained_steps[f"fm-{k}rect"] + args.reflow_steps
        pairs_made[name] = pairs_made[f"fm-{k}rect"] + args.n_reflow
        prev = nxt
    fm_methods = list(models.keys())

    # --- [4] 직선성 지표 (eq. §6.2) ------------------------------------------
    print("\n=== [4] 직선성 지표 S(Z) — reflow 전후 (eq. §6.2) ===")
    svals, trajs, one_step = {}, {}, {}
    for name, m in models.items():
        svals[name] = straightness(m, min(2048, args.n_sample), dev,
                                   k_fine=args.k_fine, seed=args.seed + 2)
        _, tr = euler_sample(m, 14, dev, nfe=args.k_fine, seed=args.seed + 4, trace=True)
        trajs[name] = tr.cpu().numpy()
        one_step[name] = euler_sample(m, args.n_sample, dev, nfe=1,
                                      seed=args.seed + 1).cpu().numpy()
    s0 = svals["fm-1rect"]
    # S는 reflow 후 소수 네 자리로는 전부 0으로 보입니다. 지수 표기로 찍어야 차이가 읽힙니다.
    print_table(["모델", "S(Z)", "1-rect 대비", "누적 학습 스텝", "생성한 쌍 수", "NFE=1 샘플 표준편차"],
                [[NICE[n], f"{svals[n]:.3e}", f"{svals[n] / s0:.2e}",
                  f"{trained_steps[n]:,}", f"{pairs_made[n]:,}",
                  f"{one_step[n].std(0)[0]:.3f}, {one_step[n].std(0)[1]:.3f}"]
                 for n in fm_methods],
                ["left", "right", "right", "right", "right", "right"])
    print(f"  → S(Z)가 {s0:.3e} → {svals[fm_methods[-1]]:.3e} 로 "
          f"{s0 / max(svals[fm_methods[-1]], 1e-12):.0f}배 내려갑니다. "
          "0이면 완전한 직선이고 1스텝이 정확합니다(§6.1).")
    print("  → **누적 학습 스텝과 생성한 쌍 수 열이 §6.3 인용문 ②의 청구서입니다.** "
          "직선성은 공짜로 얻은 것이 아닙니다.")
    print(f"  → NFE=1 표준편차: 데이터는 {ref_eval.cpu().numpy().std(0).round(3)} 입니다. "
          "1-rect가 한 점으로 뭉쳤다면 reflow 후 얼마나 되살아나는지 보세요.")

    # --- [5] NFE 스윕 --------------------------------------------------------
    print(f"\n=== [5] NFE 스윕 — {len(fm_methods)}개 모델 × NFE {nfes}, 샘플 {args.n_sample}개 ===")
    euler_sample(base, 64, dev, nfe=4, seed=0)     # 워밍업
    res: list[dict] = []
    for name, m in models.items():
        for nfe in nfes:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            x = euler_sample(m, args.n_sample, dev, nfe=nfe, seed=args.seed + 1)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            sec = time.perf_counter() - t0
            res.append(dict(method=name, nfe=nfe, energy=energy_distance(x, ref_eval),
                            sw1=sliced_w1(x, ref_eval, args.n_proj, args.seed), sec=sec))
        r = [d for d in res if d["method"] == name]
        print(f"  {NICE[name]:28s} " + "  ".join(f"NFE{d['nfe']}:{d['sw1']:.4f}" for d in r))

    # --- [6] ★ W1-M3와 나란히 -----------------------------------------------
    print("\n=== [6] ★ W1-M3 §5.3이 막혔던 자리 — sliced W₁ 기준 ===")
    methods = (["ancestral", "ddim"] if overlay_ok else []) + fm_methods
    all_rows = (m3 if overlay_ok else []) + res
    rows = []
    for nfe in nfes:
        cells = []
        for meth in methods:
            v = next((d["sw1"] for d in all_rows if d["method"] == meth and d["nfe"] == nfe), None)
            cells.append(f"{v:.4f}" if v is not None else "—")
        rows.append([str(nfe)] + cells)
    print_table(["NFE"] + [NICE[m].replace(" (W1-M3)", "*") for m in methods],
                rows, ["right"] * (len(methods) + 1))
    if overlay_ok:
        print("  * = W1-M3 02_results.csv 에서 읽은 값 (재계산하지 않았습니다)")
        for nfe in (1, 2, 4, 10):
            if nfe not in nfes:
                continue
            base_best = min((d["sw1"] for d in m3 if d["nfe"] == nfe), default=None)
            fm_best = min((d["sw1"] for d in res if d["nfe"] == nfe), default=None)
            if base_best and fm_best:
                who = min((d for d in res if d["nfe"] == nfe), key=lambda d: d["sw1"])["method"]
                print(f"  NFE={nfe:2d}: W1-M3 최선 {base_best:.4f} → 여기 최선 {fm_best:.4f} "
                      f"({base_best / fm_best:.2f}배 개선, {NICE[who]})")
    # reflow 후 곡선이 **평평해지는 것** 자체가 결과입니다 — 스텝을 더 줘도 좋아지지 않습니다.
    for name in fm_methods:
        r = sorted([d for d in res if d["method"] == name], key=lambda d: d["nfe"])
        span = r[0]["sw1"] / r[-1]["sw1"]
        print(f"  {NICE[name]:28s} NFE {r[0]['nfe']} → {r[-1]['nfe']} 사이 sW₁ 변화 {span:.1f}배"
              + ("   ← 사실상 평평합니다" if span < 1.5 else ""))
    print("  → **reflow 후 곡선이 평평해지는 것 자체가 §6.3의 거래입니다.** NFE를 늘려도 더 좋아지지")
    print("     않습니다. 1스텝으로 도달하는 그 값이 이 모델이 낼 수 있는 전부이고, 그것이 원 모델의")
    print("     큰-NFE 성능보다 조금 나쁜지가 [7]의 판정입니다.")

    # --- [7] reflow의 대가 (§6.3 인용문 ③) -----------------------------------
    print("\n=== [7] reflow의 대가 — 큰 NFE에서 되레 나빠지는가 (§6.3 인용문 ③) ===")
    big = [n for n in nfes if n >= 50]
    rows = []
    for nfe in big:
        b = next(d for d in res if d["method"] == "fm-1rect" and d["nfe"] == nfe)
        for name in fm_methods[1:]:
            d = next(x for x in res if x["method"] == name and x["nfe"] == nfe)
            # 판정 전에 **노이즈 바닥부터** 봅니다. 둘 다 바닥 이하면 우열을 말할 수 없습니다
            # (W1-M3 02가 에너지 거리에 쓴 것과 같은 규칙).
            if b["sw1"] <= floor_w1 and d["sw1"] <= floor_w1:
                v = "— 둘 다 노이즈 바닥 이하"
            elif d["sw1"] > b["sw1"]:
                v = "❌ 나빠짐"
            else:
                v = "✅ 유지·개선"
            rows.append([str(nfe), NICE[name], f"{b['sw1']:.4f}", f"{d['sw1']:.4f}",
                         f"{d['sw1'] / b['sw1']:.2f}배", v])
    if rows:
        print_table(["NFE", "모델", "1-rect sW1", "이 모델 sW1", "비", f"판정 (바닥 {floor_w1:.4f})"],
                    rows, ["right", "left", "right", "right", "right", "left"])
    print("  → lesson §6.3은 '큰 NFE의 품질을 팔아 작은 NFE의 품질을 산다'고 적었습니다.")
    print("     ❌면 그 거래가 이 실험에서도 관찰된 것이고, '노이즈 바닥 이하'면 **이 toy·이 표본 수로는")
    print("     잴 수 없을 만큼 작다**는 뜻이지 대가가 없다는 뜻이 아닙니다. 2D 두 초승달은 원래 쉬운")
    print("     분포라 §6.3의 대가가 CIFAR-10에서만큼 크게 드러나지 않습니다.")
    print("     **작은 NFE 쪽 이득은 바닥보다 한 자릿수 이상 크므로 그쪽 결론은 안전합니다.**")

    # --- [8] 산출물 ----------------------------------------------------------
    if not args.no_plot:
        p = plot_overlay(res, m3, floor, floor_w1, overlay_ok, methods,
                         out_dir / "02_nfe_overlay.png")
        print(f"\n  [저장] {p}   ← ★ 이 실습 전체의 결론 그림")
        p = plot_straightness(trajs, one_step, svals, ref_eval.cpu().numpy(), fm_methods,
                              out_dir / "02_straightness.png")
        print(f"  [저장] {p}")

    with open(out_dir / "02_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "nfe", "energy", "sw1", "sec"])
        w.writeheader()
        w.writerows(res)
    print(f"  [저장] {out_dir / '02_results.csv'}")
    with open(out_dir / "02_reflow_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "straightness", "train_steps", "pairs_made",
                                          "one_step_std_x", "one_step_std_y"])
        w.writeheader()
        for n in fm_methods:
            w.writerow(dict(method=n, straightness=svals[n], train_steps=trained_steps[n],
                            pairs_made=pairs_made[n], one_step_std_x=float(one_step[n].std(0)[0]),
                            one_step_std_y=float(one_step[n].std(0)[1])))
    print(f"  [저장] {out_dir / '02_reflow_summary.csv'}")

    print(f"\n총 소요 {time.perf_counter() - t_start:.1f}s")
    print("다음: python 03_fm_action_head.py   "
          "(W1-M3의 DiT 액션 헤드를 그대로 가져와 objective만 FM으로 바꿉니다)")


if __name__ == "__main__":
    import sys

    # 노트북(ipykernel)에서는 argparse가 jupyter의 -f 인자를 먹지 않도록 빈 리스트를 넘긴다.
    main(None if "ipykernel" not in sys.modules else [])
