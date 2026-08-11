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
# # W2-M1 실습 3, compounding error를 상한이 아니라 실측으로
#
# [`../lesson.md`](../lesson.md) `§3.2`(왜 $T^2$인가)와 `§3.3`(chunking이 상한을 어떻게 바꾸는가)의
# 주장을 **토이 시뮬레이션으로 실증**합니다. **이 스크립트가 practice의 무게중심입니다.**
#
# ---
#
# ## ⚠️ 먼저 정직성 단서, 이것이 무엇이 아닌가
#
# **이 시뮬레이션은 실제 로봇도 실제 정책도 아닙니다. 가정된 오차 모델 위의 확률 과정입니다.**
#
# 아래 곡선의 **최적 $k$ 값 자체는 오차 모델의 산물**입니다. 파라미터를 바꾸면 최적점이 움직입니다.
# 실측하려면 실제 정책을 여러 `chunk_size`로 학습해 성공률을 재야 하고, 그건 W2-M2 이후의 일입니다.
#
# 이 스크립트가 보여주는 것은 특정 $k$가 아니라 **구조적 사실 하나**입니다.
# > 이득이 $1/k$이고 대가가 두 가지($\epsilon_k$ 증가 + 개방루프 동안의 무보정)라면,
# > **누적 비용 곡선은 반드시 U자가 되고 최적 $k$는 중간에 있다.**
#
# lesson §3.3이 "최적 $k$가 중간에 있습니다"라고 한 문장을 눈으로 보는 것이 목적입니다.
#
# ---
#
# ## 모델은 lesson의 세 문장을 그대로 코드로
#
# **① 관(tube)과 그 밖** (§2.2). 전문가 시연은 상태공간에 관을 만듭니다. 관 안에서는 정책이
# 학습 데이터를 갖고 있어 정상 추종하고, 관 밖에서는 **복구 데이터가 없어 되돌리는 항이 아예 없습니다.**
# 코드에서는 `in_dist` 플래그가 이 경계이고, 이탈은 **흡수 상태**입니다.
#
# **② 결정 하나 = 실수할 기회 하나** (§3.2). lesson의 $\epsilon$은 오차 크기가 아니라
# **0/1 실수 지시자의 기대값**입니다($\epsilon = \mathbb{E}[\mathbb{1}\{\pi_\theta(o)\neq\pi^*(o)\}]$).
# 그래서 코드도 매 결정마다 확률 $\epsilon_k$로 베르누이 시행을 합니다.
# 결정 횟수가 $T \to T/k$로 줄면 시행 횟수가 그만큼 줄어듭니다. 이것이 $1/k$ 이득의 전부입니다.
#
# **③ 개방루프 동안 외란은 보정되지 않는다** (§3.3 덧붙임). 상한식에는 이 항이 없습니다.
# 코드에서는 매 스텝 외란 $w_t$가 들어오는데 보정은 **결정 시점에만** 걸립니다.
# 청크가 길수록 보정 사이에 외란이 오래 쌓입니다.
#
# ## 비교 대상 3종 (lesson §3.6 표의 행과 1:1)
#
# | 모드 | $k$ (예측) | $n$ (실행) | lesson 대응 |
# |---|---|---|---|
# | ① 단일 스텝 BC | 1 | 1 | §6.1 왼쪽 열 |
# | ② 청크 개방루프 완주 | 100 | 100 | §3.6 1행, **LeRobot 기본값** |
# | ③ receding horizon | 100 | 10 또는 50 | §3.6 2행과 3행 |
#
# 출력:
# - stdout에 3종 비교표와 $k$ 스윕 ASCII U자 곡선
# - `artifacts/W2-M1/03_compounding_error.csv`
#
# **의존성 0. 표준 라이브러리만 씁니다.** 기본 실행 30초 이내, `--smoke` 는 수 초.

# %%
from __future__ import annotations

import argparse
import csv
import math
import random
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

MODULE_ID = "W2-M1"


# %% [markdown]
# ## 0. 경로와 표 유틸 (01, 02와 동일 규약)

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


def disp_width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def pad(s: str, width: int, align: str = "l") -> str:
    gap = max(0, width - disp_width(s))
    return (" " * gap + s) if align == "r" else (s + " " * gap)


def render_table(headers: list[str], rows: list[list[str]], aligns: str | None = None) -> str:
    aligns = aligns or "l" * len(headers)
    widths = [disp_width(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], disp_width(cell))
    head = "  " + "  ".join(pad(h, widths[i], aligns[i]) for i, h in enumerate(headers))
    sep = "  " + "  ".join("-" * w for w in widths)
    body = ["  " + "  ".join(pad(c, widths[i], aligns[i]) for i, c in enumerate(r)) for r in rows]
    return "\n".join([head, sep, *body])


# %% [markdown]
# ## 1. 오차 모델은 $\epsilon_k$를 어떤 형태로 잡았고 왜 그런가
#
# $$\epsilon_k = \epsilon_1\bigl(1 + c\,k^{\,p}\bigr), \qquad p = 0.5\ (\text{기본}),\quad c = \texttt{--eps-growth}$$
#
# **왜 $\sqrt{k}$인가.** 청크 $k$개를 "충분히 맞히지 못할" 확률을 생각합니다.
# 청크 안 $k$개 예측의 오차가 **완전 독립**이면 실수 확률이 $k$에 비례하고(합집합 상한),
# **완전 종속**이면 $k$에 무관합니다. 실제 청크는 한 번의 forward가 만든 **상관된 궤적 조각**이라
# 그 중간이고, 상관 있는 합의 표준편차가 $\sqrt{k}$로 자라는 것이 가장 흔한 중간 모델입니다.
#
# **그리고 $\sqrt{k}$는 임계 지수입니다.** §3.3의 상한이 $\epsilon_k T^2/(2k)$이므로 지배항은 $\epsilon_k/k$인데,
# $\epsilon_k \propto k^p$를 넣으면 $k^{p-1}$입니다. $p<1$이면 이 항 **단독으로는 계속 감소**합니다.
# 즉 **상한식만으로는 U자가 나오지 않습니다.** U자를 만드는 나머지 힘이
# lesson §3.3 덧붙임이 지목한 **반응성 비용**(개방루프 동안 외란 무보정)이고, 이 시뮬은 그것을 물리로 갖고 있습니다.
# `--eps-power`로 $p$를 바꿔 이 구조를 직접 만져볼 수 있습니다.

# %%
@dataclass
class ErrorModel:
    """관(tube) 위의 토이 오차 모델. 전부 정규화 좌표(관 반경 r = 1)."""

    T: int = 400              # 에피소드 길이 [steps]
    tube_r: float = 1.0       # 관 반경. 이탈 거리가 이 값을 넘으면 스텝당 비용이 1로 포화
    eps1: float = 0.0015      # 단일 스텝 결정의 실수 확률 epsilon_1  # eq.(§3.2)
    eps_growth: float = 0.12  # c — 청크가 길수록 회귀가 어렵다        # eq.(§3.3)
    eps_power: float = 0.5    # p — 위 markdown 참조
    kappa: float = 0.5        # 관 안에서 결정 하나가 되돌리는 이탈의 비율 (0이면 되돌림 없음)
    sigma_w: float = 0.035    # 스텝당 외란 표준편차 (개방루프 동안 보정되지 않는다)
    fail_kick: float = 1.2    # 실수 시 관 밖으로 밀려나는 거리 (tube_r 단위)
    fail_drift: float = 0.02  # 이탈 후 스텝당 발산 (복구 피드백이 없으므로 멀어지기만)

    def eps_k(self, k: int) -> float:
        """청크 단위 실수 확률.  # eq.(§3.3)"""
        return min(1.0, self.eps1 * (1.0 + self.eps_growth * (k ** self.eps_power)))

    def bound(self, k: int) -> float:
        """lesson §3.3의 상한 eps_k * T^2 / (2k). 실측과 나란히 보기 위한 참고값."""
        return self.eps_k(k) * self.T ** 2 / (2.0 * k)


# %% [markdown]
# ## 2. 롤아웃 하나
#
# 상태는 기준 궤적으로부터의 **이탈** $d_t$ 하나입니다(관 좌표계).
# 원래 문제는 $s_{t+1} = s_t + a_t + w_t$의 추종이고, 전문가 액션을 뺀 오차 좌표로 옮기면 아래가 됩니다.
#
# ```
# 결정 j (시각 t0, 관측 d_obs):
#     eps_k 로 베르누이 시행 → 실수하면 관 밖으로 (흡수)          # eq.(§3.2)
#     보정량  = -kappa * d_obs   (관 안에서만. 관 밖이면 0)        # §2.2 복구 데이터 없음
#     n 스텝에 나눠 실행하고, 매 스텝 외란 w_t 가 들어온다         # §3.3 덧붙임: 반응성 비용
#     비용   += min(|d_t| / r, 1)                                  # eq.(§3.2) 스텝당 비용 상한 1
# ```

# %%
def rollout(model: ErrorModel, chunk_k: int, n_action_steps: int,
            rng: random.Random) -> tuple[float, bool, float]:
    """에피소드 하나. (누적 비용, 이탈 여부, 평균 |이탈|)을 돌려준다."""
    gauss = rng.gauss          # 루프 안에서 조회 비용을 줄인다 (읽기 쉬움 > 최적화지만 이건 한 줄)
    d = 0.0
    in_dist = True
    cost = 0.0
    abs_sum = 0.0
    t = 0
    eps_k = model.eps_k(chunk_k)
    r = model.tube_r

    while t < model.T:
        n_exec = min(n_action_steps, model.T - t)
        d_obs = d

        # --- 결정 하나 = 회귀 한 번 = 실수할 기회 하나 -----------------  # eq.(§3.2)
        if in_dist and rng.random() < eps_k:
            in_dist = False
            d += model.fail_kick * r * (1.0 if rng.random() < 0.5 else -1.0)

        # --- 보정: 관 안에서만 유효 (§2.2, 관 밖에는 복구 시연이 없다) ---
        corr_per_step = (-model.kappa * d_obs / n_exec) if in_dist else 0.0

        for _ in range(n_exec):
            d += corr_per_step + gauss(0.0, model.sigma_w)   # 외란은 매 스텝, 보정은 결정 시점에만
            if not in_dist:
                d += model.fail_drift * (1.0 if d >= 0.0 else -1.0)
            a = abs(d)
            cost += a / r if a < r else 1.0                  # eq.(§3.2) 스텝당 비용 상한 1
            abs_sum += a
            t += 1

    return cost, (not in_dist), abs_sum / model.T


def evaluate(model: ErrorModel, chunk_k: int, n_action_steps: int,
             trials: int, seed: int) -> dict[str, float]:
    """같은 설정을 trials 번 반복해 평균낸다.

    평균|이탈|은 **이탈하지 않은 에피소드만** 평균냅니다. 이탈 후에는 발산이 지배해서
    섞으면 '추종이 얼마나 매끄러운가'를 못 읽습니다 — 이탈률 열이 이미 그 정보를 갖고 있습니다.
    """
    rng = random.Random(seed)
    costs: list[float] = []
    fails = 0
    devs_ok: list[float] = []
    for _ in range(trials):
        c, failed, dev = rollout(model, chunk_k, n_action_steps, rng)
        costs.append(c)
        if failed:
            fails += 1
        else:
            devs_ok.append(dev)
    mean = math.fsum(costs) / trials
    var = math.fsum((c - mean) ** 2 for c in costs) / max(1, trials - 1)
    std = math.sqrt(var)
    return {
        "chunk_k": chunk_k,
        "n_action_steps": n_action_steps,
        "decisions": math.ceil(model.T / n_action_steps),
        "eps_k": model.eps_k(chunk_k),
        "mean_cost": mean,
        "std_cost": std,
        "sem_cost": std / math.sqrt(trials),
        "cost_per_step": mean / model.T,
        "fail_rate": fails / trials,
        "mean_abs_dev_ok": (math.fsum(devs_ok) / len(devs_ok)) if devs_ok else float("nan"),
        "bound_s33": model.bound(chunk_k),
    }


def classify_shape(sweep: list[dict[str, float]]) -> tuple[str, int]:
    """곡선 모양을 판정한다. 최소점이 양 끝과 통계적으로 구분되는지로 본다.

    단순히 argmin 이 내부에 있는지로 보면 동점(예: 외란 0 → 비용 0)에서 오판합니다.
    양 끝 비용이 최소점보다 표준오차 2배를 넘게 큰지를 봅니다.
    """
    costs = [r["mean_cost"] for r in sweep]
    sems = [r["sem_cost"] for r in sweep]
    i = min(range(len(costs)), key=lambda j: costs[j])
    tol_lo = 2.0 * max(sems[0], sems[i]) + 1e-9
    tol_hi = 2.0 * max(sems[-1], sems[i]) + 1e-9
    higher_at_start = costs[0] - costs[i] > tol_lo
    higher_at_end = costs[-1] - costs[i] > tol_hi
    if higher_at_start and higher_at_end:
        return "U자", i
    if higher_at_start:
        return "단조 감소", i
    if higher_at_end:
        return "단조 증가", i
    return "평탄", i


# %% [markdown]
# ## 3. ASCII 곡선
#
# matplotlib을 쓰지 않습니다. 가로 막대로 $k$ vs 비용을 그리고 최소점을 `<<<` 로 표시합니다.

# %%
def ascii_curve(results: list[dict[str, float]], key: str = "mean_cost",
                width: int = 52, label: str = "") -> str:
    vals = [r[key] for r in results]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    best = min(range(len(results)), key=lambda i: vals[i])
    lines = [f"  {label}  (막대 = 누적 비용. 짧을수록 좋다)",
             f"  최소 {lo:,.1f}  ~  최대 {hi:,.1f}"]
    for i, r in enumerate(results):
        v = vals[i]
        filled = round((v - lo) / span * (width - 4)) + 4
        mark = "  <<< 최소" if i == best else ""
        lines.append(
            f"  k={int(r['chunk_k']):>3} n={int(r['n_action_steps']):>3} "
            f"|{'#' * filled:<{width}}| {v:>8,.1f}{mark}"
        )
    return "\n".join(lines)


# %% [markdown]
# ## 4. 메인

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W2-M1 실습 3: compounding error 토이 시뮬 — lesson §3.2·§3.3")
    p.add_argument("--horizon", type=int, default=400, help="에피소드 길이 T (기본 400)")
    p.add_argument("--trials", type=int, default=1000,
                   help="설정당 반복 횟수 (기본 1000). 이탈은 드문 사건이라 표본이 적으면 곡선이 울퉁불퉁해진다")
    p.add_argument("--seed", type=int, default=0, help="난수 시드 (기본 0)")
    p.add_argument("--eps1", type=float, default=0.0015,
                   help="단일 스텝 결정의 실수 확률 epsilon_1 (기본 0.0015)")
    p.add_argument("--eps-growth", type=float, default=0.12,
                   help="eps_k = eps1 (1 + c k^p) 의 c (기본 0.12). 0이면 eps_k 가 k에 무관")
    p.add_argument("--eps-power", type=float, default=0.5,
                   help="위 식의 p (기본 0.5 = sqrt(k)). 임계 지수는 1")
    p.add_argument("--kappa", type=float, default=0.5,
                   help="결정 하나가 관 안에서 되돌리는 이탈의 비율 (기본 0.5, 0이면 되돌림 없음)")
    p.add_argument("--sigma-w", type=float, default=0.035,
                   help="스텝당 외란 표준편차 (기본 0.035). 개방루프 동안 보정되지 않는다")
    p.add_argument("--smoke", action="store_true",
                   help="수 초에 끝나는 축소 실행 (trials 50 · k 격자 축소)")
    p.add_argument("--no-csv", action="store_true", help="CSV 저장을 건너뛴다")
    return p.parse_args(argv)


K_GRID_FULL = [1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
K_GRID_COARSE = [1, 2, 5, 10, 20, 50, 100]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = artifacts_dir()
    trials = 50 if args.smoke else args.trials
    t_start = time.perf_counter()

    model = ErrorModel(
        T=args.horizon, eps1=args.eps1, eps_growth=args.eps_growth,
        eps_power=args.eps_power, kappa=args.kappa, sigma_w=args.sigma_w,
    )

    print("=" * 104)
    print(f"  {MODULE_ID} 실습 3 — compounding error 토이 시뮬  (표준 라이브러리만 · 의존성 0)")
    print("=" * 104)
    if args.smoke:
        print("  [--smoke] 축소 실행입니다. 곡선 모양만 확인하고 숫자를 결론에 쓰지 마세요.")
    print(f"\n  T={model.T} · trials={trials} · seed={args.seed}")
    print(f"  eps_k = {model.eps1:g} * (1 + {model.eps_growth:g} * k^{model.eps_power:g})"
          f"   |  kappa={model.kappa:g} · sigma_w={model.sigma_w:g}")
    print("\n  ⚠️  이것은 실제 로봇이 아니라 **가정된 오차 모델 위의 시뮬레이션**입니다.")
    print("      최적 k 값 자체는 모델의 산물입니다. 보여주는 것은 U자가 나타난다는 구조적 사실입니다.")

    # --- [1] 비교 대상 3종 ---------------------------------------------------
    print("\n=== [1] 비교 3종 — lesson §3.6 표의 행과 1:1 ===\n")
    configs = [
        ("① 단일 스텝 BC", 1, 1),
        ("② 청크 개방루프 완주 (LeRobot 기본값)", 100, 100),
        ("③ receding horizon (docstring 예시)", 100, 50),
        ("③ receding horizon (짧게)", 100, 10),
    ]
    cmp_rows = []
    cmp_res = []
    for label, k, n in configs:
        r = evaluate(model, k, n, trials, args.seed)
        cmp_res.append((label, r))
        dev = r["mean_abs_dev_ok"]
        cmp_rows.append([
            label, str(k), str(n), str(int(r["decisions"])),
            f"{r['eps_k']:.5f}",
            f"{r['mean_cost']:,.1f}", f"±{r['sem_cost']:,.1f}",
            f"{r['fail_rate'] * 100:.0f}%",
            "-" if dev != dev else f"{dev:.3f}",
            f"{r['bound_s33']:,.0f}",
        ])
    print(render_table(
        ["설정", "k", "n", "결정수", "eps_k", "평균 비용", "±표준오차",
         "이탈률", "평균|이탈| (비이탈)", "§3.3 상한"],
        cmp_rows, aligns="lrrrrrrrrr"))
    print("\n  읽는 법:")
    print("   · '평균|이탈|' 은 **이탈하지 않은 에피소드만** 평균낸 값입니다 — 추종이 얼마나 매끄러운가.")
    print("     ②가 가장 큽니다. 100스텝 동안 외란이 한 번도 보정되지 않기 때문입니다(반응성 비용).")
    print("   · '이탈률' 은 반대 방향입니다. 결정 횟수가 많을수록(=n이 작을수록) 실수 기회가 많습니다.")
    print("     **receding horizon 이 공짜가 아니라는 것**이 이 표의 요점입니다(lesson §2.5).")
    print("   · '§3.3 상한' 열은 eps_k T^2/(2k) 입니다. k가 클 때 **상한이 실측보다 작습니다.**")
    print("     상한이 틀린 게 아니라 **다른 것을 세고 있습니다** — 이탈 비용만 세고 반응성 비용을")
    print("     세지 않습니다. lesson §3.3 덧붙임이 지목한 바로 그 누락입니다.")

    # --- [2] k 스윕, 이 스크립트의 본체 -------------------------------------
    print("\n=== [2] k 스윕 (모드 ②: n = k, 청크를 끝까지 실행) — 최적 k 는 중간에 있는가 ===\n")
    ks = K_GRID_COARSE if args.smoke else K_GRID_FULL
    sweep_open = [evaluate(model, k, k, trials, args.seed) for k in ks]
    print(ascii_curve(sweep_open, label="모드 ② 개방루프 완주"))
    shape_open, i_best = classify_shape(sweep_open)
    best_open = sweep_open[i_best]
    k_star = int(best_open["chunk_k"])
    print(f"\n  최적 k* = {k_star}  (비용 {best_open['mean_cost']:,.1f} ± {best_open['sem_cost']:,.1f})")
    print(f"  k=1 대비 {sweep_open[0]['mean_cost'] / best_open['mean_cost']:.2f}배 개선 · "
          f"k={ks[-1]} 대비 {sweep_open[-1]['mean_cost'] / best_open['mean_cost']:.2f}배 개선")
    print(f"  곡선 모양 판정(양 끝이 최소점보다 표준오차 2배 넘게 큰가): **{shape_open}**")
    interior = shape_open == "U자"

    # --- [3] 같은 스윕을 receding horizon 으로 -------------------------------
    print("\n=== [3] 같은 k 를 receding horizon 으로 실행하면 (n = max(1, k/4)) ===\n")
    sweep_rh = [evaluate(model, k, max(1, k // 4), trials, args.seed) for k in ks]
    print(ascii_curve(sweep_rh, label="모드 ③ receding horizon"))
    shape_rh, i_rh = classify_shape(sweep_rh)
    best_rh = sweep_rh[i_rh]
    print(f"\n  최적 (k={int(best_rh['chunk_k'])}, n={int(best_rh['n_action_steps'])}) "
          f"비용 {best_rh['mean_cost']:,.1f}  vs  모드 ② 최적 k={k_star} 비용 "
          f"{best_open['mean_cost']:,.1f}   (모양: {shape_rh})")
    print("  → 최소점의 위치가 모드 ②와 다릅니다. **k 와 n 은 따로 튜닝해야 하는 두 노브**이고,")
    print("     그래서 LeRobot 설정에 필드가 둘인 것입니다(lesson §2.5).")
    print("     k 가 작은 구간에서 모드 ③이 모드 ②보다 나쁜 것도 눈여겨보세요 — n=1 로 눌린 채")
    print("     eps_k 만 커지는 구간이라, 재계획이 항상 이득은 아니라는 뜻입니다.")

    # --- [4] 두 힘 분해 ------------------------------------------------------
    print("\n=== [4] U자는 어디서 오는가 — 힘을 하나씩 꺼본다 ===\n")
    ks_v = K_GRID_COARSE
    variants = [
        ("기본 (두 힘 다 켬)", model),
        ("eps_k 증가를 끔 (--eps-growth 0)",
         ErrorModel(T=model.T, eps1=model.eps1, eps_growth=0.0, eps_power=model.eps_power,
                    kappa=model.kappa, sigma_w=model.sigma_w)),
        ("외란을 끔 (--sigma-w 0)",
         ErrorModel(T=model.T, eps1=model.eps1, eps_growth=model.eps_growth,
                    eps_power=model.eps_power, kappa=model.kappa, sigma_w=0.0)),
    ]
    var_rows = []
    for name, mdl in variants:
        sw = [evaluate(mdl, k, k, trials, args.seed) for k in ks_v]
        shape, ib = classify_shape(sw)
        var_rows.append([name, str(ks_v[ib]), f"{sw[ib]['mean_cost']:,.1f}",
                         f"{sw[0]['mean_cost']:,.1f}", f"{sw[-1]['mean_cost']:,.1f}", shape])
    print(render_table(["변형", "최적 k", "최소 비용", "k=1 비용", f"k={ks_v[-1]} 비용", "곡선 모양"],
                       var_rows, aligns="lrrrrl"))
    print(f"\n  (격자 {ks_v} · trials={trials})")
    print("  → 외란을 끄면(반응성 비용 제거) 곡선이 단조 감소로 바뀝니다 — 청크는 길수록 좋아집니다.")
    print("     **lesson §3.3 상한식만으로는 U자가 나오지 않는다**는 뜻이고, 그래서 §3.3이 덧붙임에서")
    print("     '상한식만 보면 반응성 비용이 안 보인다'고 따로 짚은 것입니다.")
    print("     eps_k 증가를 끄면 최소점이 오른쪽으로 밀립니다 — 청크 회귀 난이도가 최적 k 의 상한을 정합니다.")

    # --- [5] CSV -------------------------------------------------------------
    if not args.no_csv:
        path = out_dir / "03_compounding_error.csv"
        with path.open("w", newline="", encoding="utf-8") as fp:
            wr = csv.writer(fp)
            cols = ["mode", "chunk_k", "n_action_steps", "decisions", "eps_k",
                    "mean_cost", "std_cost", "sem_cost", "cost_per_step", "fail_rate",
                    "mean_abs_dev_ok", "bound_s33"]
            wr.writerow(cols)
            for label, r in cmp_res:
                wr.writerow(["compare:" + label] + [f"{r[c]:.6g}" for c in cols[1:]])
            for r in sweep_open:
                wr.writerow(["sweep_open_loop"] + [f"{r[c]:.6g}" for c in cols[1:]])
            for r in sweep_rh:
                wr.writerow(["sweep_receding_horizon"] + [f"{r[c]:.6g}" for c in cols[1:]])
        print(f"\n[저장] {path}")

    dt = time.perf_counter() - t_start
    print(f"\n  소요 {dt:.1f}초 (T={model.T} · trials={trials} · k 격자 {len(ks)}개 · 변형 3종)")

    print("\n" + "=" * 104)
    print("  요점: 이득은 1/k, 대가는 eps_k 증가와 개방루프 무보정. 그래서 최적 k 는 중간에 있다.")
    print("        **이 곡선의 최적 k 값은 오차 모델의 산물입니다.** 실측하려면 실제 정책을 학습해야 합니다.")
    print("        구조적 사실은 하나 — U자가 나타난다는 것, 그리고 k 와 n 이 다른 노브라는 것.")
    print("        다음 → 04_act_cvae_minimal.py (torch 필요)")
    print("=" * 104)
    return 0 if interior else 1


# %%
if __name__ == "__main__":
    import sys

    raise SystemExit(main(None if "ipykernel" not in sys.modules else []))
