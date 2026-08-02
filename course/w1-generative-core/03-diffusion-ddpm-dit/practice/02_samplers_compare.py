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
# # W1-M3 실습 2 — ancestral vs DDIM: NFE를 줄일 때 두 샘플러가 갈리는 지점을 눈으로 본다
#
# lesson.md `§5`(샘플러 — 스텝 수가 곧 제어 지연)의 실행판입니다.
#
# `01_ddpm_toy.py`가 학습한 **같은 모델**을 그대로 씁니다. 학습을 다시 하지 않는 것이 요점입니다 —
# DDIM은 "학습된 모델을 그대로 쓰면서 forward를 비마르코프로 재정식화"한 것이므로(lesson §5.1),
# 샘플러 교체에 재학습이 필요 없다는 사실 자체가 확인 대상입니다.
#
# NFE $\in\{1,2,4,10,20,50,250,1000\}$에서 두 샘플러를 돌리고 **품질과 wall-clock을 한 표에**
# 놓습니다. lesson §5.2가 "NFE는 품질 노브가 아니라 지연 예산 항목"이라고 못박은 문장의 근거입니다.
#
# **품질 지표** — 2D이므로 FID 같은 대리지표가 필요 없습니다. 분포 사이 거리를 직접 잽니다.
#
# $$\mathcal{E}(X,Y) = \frac{2}{nm}\sum_{i,j}\|x_i-y_j\| - \frac{1}{n^2}\sum_{i,j}\|x_i-x_j\|
#   - \frac{1}{m^2}\sum_{i,j}\|y_i-y_j\| \;\ge 0,\quad =0 \iff X\overset{d}{=}Y$$
#
# 에너지 거리(energy distance)와 **sliced Wasserstein-1**을 둘 다 직접 구현합니다(scipy 의존 없음).
# 전자는 분포 동일성의 척도, 후자는 **데이터 단위**로 읽히는 값이라 같이 봅니다.
#
# 출력(`artifacts/W1-M3/`): `02_samplers_grid.png` · `02_quality_vs_nfe.png` · `02_results.csv`
#
# **GPU 불필요.** `--smoke`는 NFE 집합을 줄여 수십 초에 끝납니다.

# %%
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless 고정 — 뷰어를 띄우지 않는다

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402

MODULE_ID = "W1-M3"

# lesson §13이 명시한 NFE 집합. 1000 = 학습 격자 전체.
NFE_GRID = [1, 2, 4, 10, 20, 50, 250, 1000]
NFE_GRID_SMOKE = [1, 4, 20, 250]


# %% [markdown]
# ## 0. 01의 코드를 그대로 재사용
#
# `Schedule` · `EpsMLP` · `q_sample` · 데이터 생성은 `01_ddpm_toy.py`에 있는 것을 **그대로** 씁니다.
# 파일명이 숫자로 시작해 `import`가 안 되므로 `importlib`로 경로 로드합니다(W1-M5 practice와 같은 규약).

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


def load_m1():
    """01_ddpm_toy.py 를 모듈로 로드한다."""
    for cand in (here(), find_repo_root() / "course" / "w1-generative-core"
                 / "03-diffusion-ddpm-dit" / "practice"):
        p = cand / "01_ddpm_toy.py"
        if p.is_file():
            spec = importlib.util.spec_from_file_location("ddpm_toy", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["ddpm_toy"] = mod  # dataclass가 __module__을 되찾을 수 있게
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("[에러] 01_ddpm_toy.py 를 찾지 못했습니다 (같은 폴더에 있어야 합니다)")


M1 = load_m1()
Schedule, EpsMLP, EMA = M1.Schedule, M1.EpsMLP, M1.EMA
make_schedule, make_data, normalize = M1.make_schedule, M1.make_data, M1.normalize
pick_device = M1.pick_device


# %% [markdown]
# ## 1. 폰트·표 유틸 (01과 동일)

# %%
_KO_FONT_PREFERENCE = M1._KO_FONT_PREFERENCE
USE_KOREAN = False


def setup_korean_font(force_ascii: bool = False) -> bool:
    """01의 것을 그대로 호출하고 이 모듈의 플래그에도 반영한다."""
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
# ## 2. 두 샘플러 — 한 함수로
#
# lesson §5.1의 두 갱신식은 사실 **같은 식의 $\eta$ 양 끝**입니다. DDIM 논문의 일반형이 이렇습니다.
#
# $$\hat x_0 = \frac{x_\tau - \sqrt{1-\bar\alpha_\tau}\,\epsilon_\theta}{\sqrt{\bar\alpha_\tau}},\qquad
# x_{\tau'} = \sqrt{\bar\alpha_{\tau'}}\hat x_0 + \sqrt{1-\bar\alpha_{\tau'}-\sigma^2}\,\epsilon_\theta + \sigma z$$
#
# $$\sigma = \eta\sqrt{\frac{1-\bar\alpha_{\tau'}}{1-\bar\alpha_\tau}}\sqrt{1-\frac{\bar\alpha_\tau}{\bar\alpha_{\tau'}}}$$
#
# - $\eta=0$ → **DDIM**. 확률항이 없으니 probability flow ODE의 Euler 적분이고, 같은 노이즈면 같은 결과.
# - $\eta=1$ → **ancestral**. 격자가 전체($\tau'=\tau-1$)면 $\sigma^2=\tilde\beta_t$가 되어
#   DDPM 원본 갱신식과 정확히 같아집니다. 이것이 "$\eta=1$이 ancestral"의 의미입니다.
#
# **NFE를 줄인다 = 격자 $\{\tau_1<\dots<\tau_S\}$를 성기게 잡는다**는 뜻이고, 두 샘플러의 차이는
# 격자를 건너뛸 때 무엇이 깨지는가입니다(lesson §5.1 비교표).

# %%
def make_grid(T: int, nfe: int) -> list[int]:
    """{1..T}에서 균등 간격 부분수열 τ를 고른다.

    항상 **τ_S = T** 입니다 — 샘플링은 순수 노이즈 x_T에서 출발하므로 가장 큰 격자점이
    T가 아니면 "x_T를 x_τ로 착각하고" 시작하게 됩니다. NFE=1이면 τ=[T] 한 점이고,
    이때는 x_T에서 x̂_0을 한 번에 찍는 셈입니다.
    """
    if nfe >= T:
        return list(range(1, T + 1))
    # 균등 간격 (improved-DDPM의 respacing과 같은 방식). T에서 1 쪽으로 내려가며 잡는다.
    return sorted({int(round(v)) for v in np.linspace(T, 1, nfe)})


@torch.no_grad()
def sample(model, n: int, sch: Schedule, tau: list[int], eta: float,
           device: torch.device, seed: int = 0, dim: int = 2) -> torch.Tensor:
    """일반화 샘플러. eta=0 → DDIM, eta=1 → ancestral. NFE = len(tau)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, dim, generator=g).to(device)
    for k in range(len(tau) - 1, -1, -1):
        t = tau[k]
        t_prev = tau[k - 1] if k > 0 else 0            # τ' = 0 이면 ᾱ_0 = 1 (= x_0 자체)
        ab_t = sch.abar[t - 1]
        ab_prev = sch.abar[t_prev - 1] if t_prev > 0 else torch.ones((), device=device)
        tt = torch.full((n,), t, dtype=torch.long, device=device)
        eps = model(x, tt)
        # lesson §5.1 DDIM:  x̂_0 = (x_t − √(1-ᾱ_t)·ε_θ) / √ᾱ_t
        x0_hat = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        sigma = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
        # 방향항. σ=0 이면 lesson §5.1의 DDIM 갱신식 그대로
        dir_t = (1 - ab_prev - sigma ** 2).clamp_min(0).sqrt() * eps
        x = ab_prev.sqrt() * x0_hat + dir_t
        if eta > 0 and t_prev > 0:
            x = x + sigma * torch.randn(x.shape, generator=g).to(device)
    return x


# %% [markdown]
# ## 3. 품질 지표 — numpy/torch로 직접
#
# **에너지 거리**는 분포가 같을 때만 0이 되는 진짜 거리(정확히는 준거리)입니다. $O(n^2)$이지만
# $n$이 수천이면 순식간입니다. **sliced Wasserstein-1**은 랜덤 방향에 투영한 1D Wasserstein의
# 평균이고, 값이 **데이터 단위**(정규화 좌표)라 "얼마나 틀렸나"를 감으로 읽을 수 있습니다.

# %%
def energy_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """E(X,Y) = 2·E‖x−y‖ − E‖x−x'‖ − E‖y−y'‖ ≥ 0, 0 ⟺ 같은 분포"""
    xy = torch.cdist(x, y).mean()
    xx = torch.cdist(x, x).mean()
    yy = torch.cdist(y, y).mean()
    return float(2 * xy - xx - yy)


def sliced_w1(x: torch.Tensor, y: torch.Tensor, n_proj: int = 256, seed: int = 0) -> float:
    """랜덤 방향에 투영한 1D Wasserstein-1의 평균. 두 표본 크기가 같다고 가정."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = x.shape[1]
    theta = torch.randn(d, n_proj, generator=g).to(x.device)
    theta = theta / theta.norm(dim=0, keepdim=True)
    px, _ = (x @ theta).sort(dim=0)   # [n, n_proj]
    py, _ = (y @ theta).sort(dim=0)
    m = min(px.shape[0], py.shape[0])
    return float((px[:m] - py[:m]).abs().mean())


# %% [markdown]
# ## 4. 모델 확보 — 01의 체크포인트를 읽거나, 없으면 짧게 재학습

# %%
def get_model(args, dev: torch.device):
    """01_model.pt 를 읽는다. 없으면 같은 구성으로 짧게 학습하고 그 사실을 알린다."""
    out = artifacts_dir()
    cands = [out / "01_model_smoke.pt", out / "01_model.pt"] if args.smoke \
        else [out / "01_model.pt", out / "01_model_smoke.pt"]
    if args.ckpt:
        cands = [Path(args.ckpt)]

    for ck in cands:
        if not ck.is_file():
            continue
        blob = torch.load(ck, map_location=dev, weights_only=False)
        cfg, sc = blob["cfg"], blob["schedule"]
        model = EpsMLP(**cfg).to(dev)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        sch = make_schedule(sc["T"], sc["beta_start"], sc["beta_end"]).to(dev)
        sch = Schedule(sch.T, *(getattr(sch, f).float() for f in
                                ("beta", "alpha", "abar", "abar_prev", "sqrt_abar",
                                 "sqrt_one_minus_abar", "beta_tilde")))
        ref = torch.as_tensor(blob["data_ref"], device=dev)
        print(f"  체크포인트 로드: {ck.name}   (01의 학습 {blob['train_steps']} 스텝"
              + (", ⚠️ smoke 체크포인트" if blob.get("smoke") else "") + ")")
        return model, sch, ref, str(ck.name)

    # --- 폴백: 없으면 여기서 짧게 학습 -------------------------------------
    print("  ⚠️ 01_model.pt 가 없어 **여기서 짧게 재학습**합니다 "
          f"({args.fallback_steps} 스텝). 01을 먼저 돌리면 이 단계를 건너뜁니다.")
    print("     이 경로의 샘플 품질은 01 본 학습(12,000 스텝)보다 낮습니다 — 표를 읽을 때 감안하세요.")
    raw = make_data("moons", 16384, args.seed)
    xn, _, _ = normalize(raw)
    x0 = torch.as_tensor(xn, device=dev)
    sch = make_schedule(1000).to(dev)
    sch = Schedule(sch.T, *(getattr(sch, f).float() for f in
                            ("beta", "alpha", "abar", "abar_prev", "sqrt_abar",
                             "sqrt_one_minus_abar", "beta_tilde")))
    model = EpsMLP(2, 256, 4, 128).to(dev)
    t0 = time.perf_counter()
    ema, _, _ = M1.train(model, x0, sch, steps=args.fallback_steps, batch=512, lr=2e-3,
                         ema_decay=0.999, log_every=max(1, args.fallback_steps // 20),
                         seed=args.seed)
    ema.copy_to(model)
    model.eval()
    print(f"     재학습 {time.perf_counter() - t0:.1f}s")
    return model, sch, x0[:8192], "즉석 재학습"


# %% [markdown]
# ## 5. 그림

# %%
def plot_grid(samples: dict[tuple[str, int], np.ndarray], real: np.ndarray,
              nfes: list[int], path: Path) -> Path:
    methods = ["ancestral", "ddim"]
    fig, axes = plt.subplots(len(methods), len(nfes),
                             figsize=(2.05 * len(nfes), 2.25 * len(methods)),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for r, meth in enumerate(methods):
        for c, nfe in enumerate(nfes):
            ax = axes[r, c]
            ax.scatter(real[:, 0], real[:, 1], s=1.5, alpha=0.10, color="#adb5bd")
            p = samples[(meth, nfe)]
            ax.scatter(p[:, 0], p[:, 1], s=2.0, alpha=0.40,
                       color="#c92a2a" if meth == "ancestral" else "#1971c2")
            ax.set_xlim(-3, 3), ax.set_ylim(-3, 3)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(f"NFE={nfe}", fontsize=11)
            if c == 0:
                ax.set_ylabel(lab("ancestral (η=1)", "ancestral (eta=1)") if meth == "ancestral"
                              else lab("DDIM (η=0)", "DDIM (eta=0)"), fontsize=11)
    fig.suptitle(lab("W1-M3 · NFE를 줄이면 두 샘플러가 어디서 갈리는가 (회색 = 원본 분포)",
                     "W1-M3 · where the two samplers diverge as NFE drops (grey = data)"),
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_quality(res: list[dict], path: Path, ms_per_nfe: float) -> Path:
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    for meth, color in (("ancestral", "#c92a2a"), ("ddim", "#1971c2")):
        r = [d for d in res if d["method"] == meth]
        nfe = [d["nfe"] for d in r]
        ax[0].loglog(nfe, [d["energy"] for d in r], "o-", color=color, label=meth, lw=2)
        ax[1].loglog(nfe, [d["sw1"] for d in r], "o-", color=color, label=meth, lw=2)
        ax[2].loglog([d["sec"] * 1e3 for d in r], [d["energy"] for d in r], "o-",
                     color=color, label=meth, lw=2)
    ax[0].set_xlabel("NFE"), ax[0].set_ylabel(lab("에너지 거리", "energy distance"))
    ax[0].set_title(lab("(a) 품질 vs NFE — 낮을수록 좋다", "(a) quality vs NFE — lower is better"))
    ax[1].set_xlabel("NFE"), ax[1].set_ylabel("sliced $W_1$")
    ax[1].set_title(lab("(b) sliced Wasserstein-1 (데이터 단위)",
                        "(b) sliced Wasserstein-1 (data units)"))
    ax[2].set_xlabel(lab("샘플링 wall-clock [ms]", "sampling wall-clock [ms]"))
    ax[2].set_ylabel(lab("에너지 거리", "energy distance"))
    ax[2].set_title(lab("(c) 품질 vs 시간 — 지연 예산이 x축",
                        "(c) quality vs time — latency budget on x"))
    for a in ax:
        a.legend(), a.grid(alpha=0.3, which="both")
    fig.suptitle(lab(f"lesson §5.2의 가정은 1 NFE = 3 ms · 이 toy 모델 실측은 "
                     f"{ms_per_nfe:.3f} ms/NFE (배치 전체)",
                     f"lesson §5.2 assumes 3 ms/NFE · this toy model measures "
                     f"{ms_per_nfe:.3f} ms/NFE (whole batch)"), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# %% [markdown]
# ## 6. main

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W1-M3 실습 2: ancestral vs DDIM을 NFE 축에서 비교 (lesson §5)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--ckpt", default=None, help="체크포인트 경로 (기본: artifacts/W1-M3/01_model.pt)")
    p.add_argument("--n-sample", type=int, default=4096, help="구성마다 뽑을 샘플 수")
    p.add_argument("--n-proj", type=int, default=256, help="sliced W1의 투영 방향 수")
    p.add_argument("--fallback-steps", type=int, default=4000,
                   help="체크포인트가 없을 때 즉석 재학습 스텝 수")
    p.add_argument("--repeat", type=int, default=1, help="wall-clock 측정 반복 횟수 (중앙값)")
    p.add_argument("--smoke", action="store_true", help="NFE 집합을 줄여 수십 초에 완주")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--ascii-labels", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    nfes = NFE_GRID_SMOKE if args.smoke else NFE_GRID
    if args.smoke:
        args.n_sample, args.fallback_steps = 1024, 600
        print(f"[smoke] NFE 집합을 {nfes}로 줄이고 샘플 {args.n_sample}개로 실행합니다.")
    setup_korean_font(args.ascii_labels)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = pick_device(args.device)
    out_dir = artifacts_dir()
    t_start = time.perf_counter()

    print("=" * 92)
    print(f"  {MODULE_ID} 실습 2 — ancestral vs DDIM · NFE를 줄일 때 갈리는 지점")
    print("=" * 92)

    # --- [1] 모델 -----------------------------------------------------------
    print("\n=== [1] 모델 — 01이 학습한 것을 그대로 (재학습 없음) ===")
    model, sch, ref, src = get_model(args, dev)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  ε_θ 파라미터 {n_param:,}개   T={sch.T}   출처={src}")
    print("  ⓘ DDIM은 학습된 모델을 그대로 씁니다 — 샘플러를 바꾸는 데 재학습이 필요 없습니다"
          " (lesson §5.1).")
    ref_eval = ref[:args.n_sample]
    # 지표의 노이즈 바닥 — 같은 분포에서 뽑은 두 표본 사이의 에너지 거리.
    # "0이어야 할 값"이 표본 크기 때문에 얼마나 뜨는지를 먼저 재 둡니다. 이게 없으면
    # 표의 소수점 넷째 자리 차이를 실제 차이로 착각합니다.
    half = min(args.n_sample, ref.shape[0] // 2)
    floor = energy_distance(ref[:half], ref[half:2 * half])
    print(f"  지표 노이즈 바닥(같은 분포 두 표본, n={half}): energy={floor:.5f}")

    # --- [2] 스윕 -----------------------------------------------------------
    print(f"\n=== [2] 스윕 — NFE {nfes} × {{ancestral, DDIM}} ===")
    print(f"  구성마다 {args.n_sample}개 생성, wall-clock {args.repeat}회 측정(중앙값)")
    results: list[dict] = []
    samples: dict[tuple[str, int], np.ndarray] = {}
    # 워밍업 — 첫 호출은 커널 로딩/할당 때문에 느립니다. 이걸 안 하면 NFE=1 행의 wall-clock이
    # NFE=2보다 크게 나오는 이상한 표가 됩니다(측정 대상이 아닌 비용이 섞이는 것).
    sample(model, 64, sch, make_grid(sch.T, 4), 0.0, dev, seed=0)
    for meth, eta in (("ancestral", 1.0), ("ddim", 0.0)):
        for nfe in nfes:
            tau = make_grid(sch.T, nfe)
            secs = []
            for r in range(args.repeat):
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                x = sample(model, args.n_sample, sch, tau, eta, dev, seed=args.seed + 1)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                secs.append(time.perf_counter() - t0)
            sec = float(np.median(secs))
            e = energy_distance(x, ref_eval)
            w = sliced_w1(x, ref_eval, args.n_proj, args.seed)
            results.append(dict(method=meth, nfe=len(tau), sec=sec, energy=e, sw1=w))
            samples[(meth, nfe)] = x.cpu().numpy()
            print(f"    {meth:9s} NFE={len(tau):4d}  energy={e:.5f}  sW1={w:.4f}  {sec * 1e3:8.1f} ms")

    # --- [3] 표 -------------------------------------------------------------
    print(f"\n=== [3] NFE ↔ 품질 ↔ 시간 (샘플 {args.n_sample}개 기준) ===")
    rows = []
    for nfe in nfes:
        a = next(d for d in results if d["method"] == "ancestral" and d["nfe"] == min(nfe, sch.T))
        b = next(d for d in results if d["method"] == "ddim" and d["nfe"] == min(nfe, sch.T))
        ratio = a["energy"] / b["energy"] if b["energy"] > 0 else float("nan")
        rows.append([str(a["nfe"]),
                     f"{a['energy']:.5f}", f"{b['energy']:.5f}", f"{ratio:.2f}배",
                     f"{a['sw1']:.4f}", f"{b['sw1']:.4f}",
                     f"{a['sec'] * 1e3:.1f}", f"{b['sec'] * 1e3:.1f}"])
    print_table(["NFE", "energy anc", "energy DDIM", "anc/DDIM",
                 "sW1 anc", "sW1 DDIM", "ms anc", "ms DDIM"], rows, ["right"] * 8)

    best = min(results, key=lambda d: d["energy"])
    print(f"\n  최저 에너지 거리: {best['method']} NFE={best['nfe']}  ({best['energy']:.5f})")
    print(f"  지표의 노이즈 바닥: {floor:.5f}"
          "   ← **같은 분포**에서 뽑은 두 표본 사이의 에너지 거리")
    print("     이보다 작은 차이는 표본 노이즈입니다. 표에서 이 값 근처 행끼리는 우열을 말하지 마세요.")

    # 어느 NFE에서 누가 이기는가 — 노이즈 바닥을 넘는 차이만 인정한다
    print("\n  두 샘플러의 차이 (노이즈 바닥을 넘는 것만):")
    verdicts = []
    for nfe in nfes:
        n_eff = min(nfe, sch.T)
        a = next(d for d in results if d["method"] == "ancestral" and d["nfe"] == n_eff)
        b = next(d for d in results if d["method"] == "ddim" and d["nfe"] == n_eff)
        gap = a["energy"] - b["energy"]
        if gap == 0.0:
            v = "완전히 동일 — 같은 갱신식이 됩니다"
        elif abs(gap) < floor:
            v = "차이 없음 (노이즈 바닥 이하)"
        else:
            v = f"**DDIM 우세** ({a['energy'] / b['energy']:.2f}배)" if gap > 0 \
                else f"**ancestral 우세** ({b['energy'] / a['energy']:.2f}배)"
        verdicts.append((n_eff, v))
        print(f"    NFE={n_eff:4d}: {v}")
    print("  ⓘ NFE=1은 두 샘플러가 **정확히 같습니다** — 격자가 τ=[T] 한 점이라 그 스텝이 곧 마지막")
    print("     스텝이고, 마지막 스텝에는 어차피 노이즈를 주입하지 않기 때문입니다(x_0을 내놓아야 하므로).")
    print("     NFE가 작을수록 DDIM이 벌어지는 것이 lesson §5.1 비교표의 '스텝 부분집합 사용' 행입니다.")

    # "품질 기준을 어디에 두느냐"에 따라 필요한 NFE가 달라진다 — 그 의존성 자체를 보여준다.
    # 임계값 하나를 고르면 원하는 결론이 나오게 만들 수 있으므로 여러 개를 함께 찍습니다.
    print("\n  품질 기준별로 필요한 최소 NFE (기준 = 노이즈 바닥의 배수):")
    thr_rows = []
    for k in (3, 5, 10, 30):
        thr = floor * k
        cell = {}
        for meth in ("ancestral", "ddim"):
            r = sorted([d for d in results if d["method"] == meth], key=lambda d: d["nfe"])
            ok = [d for d in r if d["energy"] <= thr]
            cell[meth] = min(ok, key=lambda d: d["nfe"]) if ok else None
        a, b = cell["ancestral"], cell["ddim"]
        gain = f"{a['nfe'] / b['nfe']:.1f}배" if (a and b) else "—"
        thr_rows.append([f"{k}×", f"{thr:.5f}",
                         f"{a['nfe']}" if a else "미달",
                         f"{b['nfe']}" if b else "미달",
                         f"{a['sec'] * 1e3:.0f}" if a else "—",
                         f"{b['sec'] * 1e3:.0f}" if b else "—", gain])
    print_table(["기준", "energy 임계", "NFE anc", "NFE DDIM", "ms anc", "ms DDIM", "NFE 절감"],
                thr_rows, ["right"] * 7)
    print("  → 'DDIM이 몇 배 싸다'는 **품질 기준을 어디 두느냐에 달려 있습니다.** 기준을 느슨하게 할수록")
    print("     격차가 벌어지고, 아주 빡빡하게 잡으면 둘 다 큰 NFE를 요구합니다.")
    print("     lesson §5.2 표의 '헤드 시간' 행을 실제로 줄이려면 이 표의 어느 줄에 앉을지 먼저 정해야 합니다.")

    # 01의 3 ms/NFE 가정과 비교 (lesson §5.2·§8.3)
    per = [d["sec"] / d["nfe"] * 1e3 for d in results if d["nfe"] >= 50]
    ms_per_nfe = float(np.median(per))
    ms_per_nfe_1 = ms_per_nfe / args.n_sample * 1000  # 샘플 1개 기준 [μs]
    print(f"\n  실측 {ms_per_nfe:.3f} ms/NFE (배치 {args.n_sample}개 전체) "
          f"= 샘플 1개당 {ms_per_nfe_1:.2f} μs/NFE")
    print("  lesson §5.2·§8.3은 1 NFE = 3 ms를 가정합니다. 그 3 ms는 VLM 조건을 받는 "
          "액션 헤드(수억 파라미터, cross-attention 포함)의 값이고,")
    print(f"  여기 모델은 {n_param / 1e6:.2f}M 파라미터에 2차원 데이터라 비교 대상이 아닙니다. "
          "**NFE가 시간에 선형**이라는 관계만 같습니다.")

    # --- [4] 산출물 ---------------------------------------------------------
    if not args.no_plot:
        p = plot_grid(samples, ref_eval.cpu().numpy(), nfes, out_dir / "02_samplers_grid.png")
        print(f"\n  [저장] {p}")
        p = plot_quality(results, out_dir / "02_quality_vs_nfe.png", ms_per_nfe)
        print(f"  [저장] {p}")
    csv_path = out_dir / "02_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "nfe", "energy", "sw1", "sec"])
        w.writeheader()
        w.writerows(results)
    print(f"  [저장] {csv_path}")

    print(f"\n총 소요 {time.perf_counter() - t_start:.1f}s")
    print("다음: python 03_dit_action_head.py   (DiT 블록을 손으로 구현하고 액션 청크에 붙입니다)")


if __name__ == "__main__":
    import sys

    # 노트북(ipykernel)에서는 argparse가 jupyter의 -f 인자를 먹지 않도록 빈 리스트를 넘긴다.
    # → 전부 기본값으로 실행됩니다. --smoke로 돌리려면 이 셀을 main(["--smoke"])로 고치세요.
    main(None if "ipykernel" not in sys.modules else [])
