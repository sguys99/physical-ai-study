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
# # W2-M2 실습 1. 평균의 함정을 수치로 확인한다 (lesson §2.1, §2.2)
#
# [`../lesson.md`](../lesson.md) `§2.1`이 수식 한 줄로 적은 사실을 격자 탐색으로 다시 확인합니다.
#
# $$\arg\min_{f}\; \mathbb{E}_{(o,a)\sim\mathcal{D}}\bigl\|a - f(o)\bigr\|_2^2 \;=\; \mathbb{E}[a \mid o] \tag{§2.1}$$
#
# 이 스크립트의 성격은 계산기가 아니라 **검산기**입니다. 닫힌형이 말하는 답(표본평균)과
# 격자 탐색이 찾아낸 답을 대조해 **PASS/FAIL을 찍습니다.** 둘이 갈리면 여기서 걸립니다.
#
# ## 확인할 것 넷
#
# 1. **제곱오차의 최소해가 표본평균이다.** 항등식 $\mathrm{MSE}(c) = \mathrm{Var} + (\bar{a}-c)^2$을
#    격자 위의 모든 점에서 검사합니다. 이 항등식이 성립하면 최소점이 $c=\bar{a}$인 것이 곧바로 따라옵니다.
# 2. **그 답이 어느 모드와도 닮지 않는다.** 좌우 대칭인 두 갈래 시연에서 최소해는 두 모드 사이의
#    골짜기에 놓입니다. 표본이 거의 없는 자리입니다.
# 3. **혼합비를 바꾸면 최소해가 움직인다.** 0.5, 0.7, 0.9로 바꿔가며 최소해가 어디로 가는지 봅니다.
#    **균형 혼합에서 가장 나쁜 답이 나옵니다.**
# 4. **손실을 $\ell_1$으로 바꿔도 함정은 남는다.** 절대편차의 최소해는 중앙값이라
#    불균형 혼합에서는 한쪽 모드에 붙지만, 대칭 혼합에서는 중앙값도 가운데입니다.
#    lesson §2.2의 "문제는 손실 함수의 종류가 아니라 출력이 값 하나라는 사실"이 여기서 숫자로 나옵니다.
#
# 출력:
# - stdout: 격자 탐색 대 닫힌형 대조표, 혼합비 스윕표, 히스토그램 ASCII 막대
# - `artifacts/W2-M2/01_mean_collapse.csv`
#
# **의존성 0. 표준 라이브러리만 씁니다.** `python3 01_mean_collapse.py` 로 즉시 돌아갑니다.

# %%
from __future__ import annotations

import argparse
import csv
import random
import re
import statistics
import unicodedata
from pathlib import Path

MODULE_ID = "W2-M2"

# lesson §2.1의 PushT 예시를 1차원으로 줄인 것.
# 블록 왼쪽을 밀어 시계 방향(-1)으로 돌리거나 오른쪽을 밀어 반시계 방향(+1)으로 돌린다.
MODE_LEFT = -1.0
MODE_RIGHT = +1.0

# 최소해가 어느 한쪽 모드에 "붙었다"고 볼 거리 기준. 모드 잡음 표준편차의 몇 배 이내인가.
# 1.5 sigma 안이면 그 모드의 표본 분포 안에 들어가 있다는 뜻이다.
SNAP_SIGMA = 1.5


# %% [markdown]
# ## 0. 경로와 표 유틸 (W2-M1 practice와 동일 규약)

# %%
_ROOT_MARKERS = ("course", "docs", "CLAUDE.md")


def find_repo_root() -> Path:
    """리포 루트를 찾는다. 스크립트와 노트북 양쪽에서 동작한다."""
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


def lesson_path() -> Path:
    return here().parent / "lesson.md"


def disp_width(s: str) -> int:
    """한글은 2칸을 먹는다. 표 정렬용."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def pad(s: str, width: int, align: str = "l") -> str:
    gap = max(0, width - disp_width(s))
    if align == "r":
        return " " * gap + s
    if align == "c":
        left = gap // 2
        return " " * left + s + " " * (gap - left)
    return s + " " * gap


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
# ## 1. 데이터. 같은 관측에서 갈린 시연
#
# lesson §2.1의 상황을 1차원으로 줄입니다. 관측은 하나로 고정하고, 액션만 왼쪽 $-1$과 오른쪽 $+1$로
# 갈립니다. 각 모드에는 사람 손의 떨림에 해당하는 가우시안 잡음을 얹습니다.
#
# `p_left`가 왼쪽 모드를 고를 확률입니다. $p_{\text{left}}=0.5$가 lesson이 말하는 좌우 대칭이고,
# 0.7과 0.9는 한쪽으로 기운 시연 집합입니다.

# %%
def make_demos(n: int, p_left: float, sigma: float, rng: random.Random) -> list[float]:
    """같은 관측에서 두 갈래로 갈린 시연 액션을 만든다. 각 모드에 가우시안 잡음."""
    out: list[float] = []
    for _ in range(n):
        center = MODE_LEFT if rng.random() < p_left else MODE_RIGHT
        out.append(center + rng.gauss(0.0, sigma))
    return out


def true_mixture_mean(p_left: float) -> float:
    """모집단 기댓값. 표본평균이 수렴할 값.  # eq.(§2.1) 우변"""
    return p_left * MODE_LEFT + (1.0 - p_left) * MODE_RIGHT


# %% [markdown]
# ## 2. 두 손실의 최소해. 격자 탐색과 닫힌형
#
# 상수 예측 $c$ 하나만 낼 수 있는 회귀를 생각합니다. lesson §2.1이 말하는 "출력이 값 하나"의
# 가장 단순한 형태입니다. 두 손실을 각각 격자로 훑습니다.
#
# $$\mathrm{MSE}(c) = \frac{1}{N}\sum_i (a_i - c)^2, \qquad
#   \mathrm{MAD}(c) = \frac{1}{N}\sum_i |a_i - c|$$
#
# 닫힌형은 각각 표본평균과 표본중앙값입니다. 격자가 그 답을 찾아내는지가 이 절의 검사 항목입니다.

# %%
def mse_at(demos: list[float], c: float) -> float:
    """제곱오차 평균.  # eq.(§2.1) 좌변"""
    return sum((a - c) ** 2 for a in demos) / len(demos)


def mad_at(demos: list[float], c: float) -> float:
    """절대편차 평균. lesson §2.1의 'l1이면 해가 조건부 중앙값으로 바뀔 뿐'."""
    return sum(abs(a - c) for a in demos) / len(demos)


def grid_argmin(fn, lo: float, hi: float, step: float) -> tuple[float, float]:
    """[lo, hi]를 step 간격으로 훑어 최소점을 찾는다. (argmin, min값)"""
    n = int(round((hi - lo) / step))
    best_c, best_v = lo, fn(lo)
    for i in range(1, n + 1):
        c = lo + i * step
        v = fn(c)
        if v < best_v:
            best_c, best_v = c, v
    return best_c, best_v


def mad_minimizer_interval(demos: list[float]) -> tuple[float, float]:
    """절대편차의 최소해 구간. 표본 수가 짝수면 가운데 두 순서통계량 사이가 전부 최소다.

    MAD(c)는 조각별 선형이라 최소점이 한 점이 아니라 구간이 될 수 있다.
    격자 탐색 결과를 '표본중앙값과 같은가'로 검사하면 이 평평한 구간 때문에 헛되이 어긋난다.
    """
    s = sorted(demos)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2], s[n // 2]
    return s[n // 2 - 1], s[n // 2]


def mse_identity_residual(demos: list[float], c: float) -> float:
    """항등식 MSE(c) = Var + (mean - c)^2 의 잔차. 0이어야 한다.

    이 항등식이 성립하면 최소점이 c = mean인 것이 곧바로 따라온다.
    격자 탐색이 아니라 이 잔차가 §2.1 수식의 진짜 증명이다.
    """
    mean = statistics.fmean(demos)
    var = sum((a - mean) ** 2 for a in demos) / len(demos)
    return mse_at(demos, c) - (var + (mean - c) ** 2)


# %% [markdown]
# ## 3. 최소해가 골짜기에 놓이는가
#
# 최소해가 표본평균이라는 것만으로는 lesson의 논지가 완성되지 않습니다. 그 값이 **시연이 거의 없는
# 자리**여야 "어느 시연과도 닮지 않은 답"이 됩니다. 두 가지로 잽니다.
#
# - **최근접 모드까지의 거리**를 모드 잡음 $\sigma$의 배수로. 3배를 넘으면 어느 모드에도 속하지 않는 값입니다.
# - **히스토그램 골짜기 깊이**. 최소해가 놓인 구간의 표본 수를 최빈 구간의 표본 수로 나눈 값입니다.
#   0에 가까울수록 그 자리에 시연이 없다는 뜻입니다.

# %%
def dist_to_nearest_mode(c: float, sigma: float) -> tuple[float, float]:
    """최근접 모드 중심까지의 거리와 그 거리의 sigma 배수."""
    d = min(abs(c - MODE_LEFT), abs(c - MODE_RIGHT))
    return d, d / sigma


def histogram(demos: list[float], lo: float, hi: float, bins: int) -> tuple[list[int], float]:
    counts = [0] * bins
    width = (hi - lo) / bins
    for a in demos:
        idx = int((a - lo) / width)
        if 0 <= idx < bins:
            counts[idx] += 1
    return counts, width


def valley_depth(demos: list[float], c: float, lo: float, hi: float, bins: int) -> tuple[float, int, int]:
    """최소해가 놓인 구간의 표본 수 / 최빈 구간의 표본 수. (비율, 그 구간 표본 수, 최빈 구간 표본 수)"""
    counts, width = histogram(demos, lo, hi, bins)
    idx = int((c - lo) / width)
    idx = max(0, min(bins - 1, idx))
    peak = max(counts) if counts else 0
    return (counts[idx] / peak if peak else float("nan")), counts[idx], peak


def ascii_hist(demos: list[float], lo: float, hi: float, bins: int, marks: dict[str, float],
               width: int = 46) -> str:
    """히스토그램을 ASCII 막대로. marks에 표시할 위치를 넣으면 해당 줄에 화살표를 단다."""
    counts, bin_w = histogram(demos, lo, hi, bins)
    peak = max(counts) or 1
    mark_idx: dict[int, list[str]] = {}
    for name, pos in marks.items():
        idx = max(0, min(bins - 1, int((pos - lo) / bin_w)))
        mark_idx.setdefault(idx, []).append(name)
    lines = []
    for i, cnt in enumerate(counts):
        center = lo + (i + 0.5) * bin_w
        bar = "#" * int(round(cnt / peak * width))
        tag = ("  <== " + ", ".join(mark_idx[i])) if i in mark_idx else ""
        lines.append(f"  {center:+5.2f} |{bar:<{width}} {cnt:5d}{tag}")
    return "\n".join(lines)


# %% [markdown]
# ## 4. lesson 참조값 읽기
#
# lesson §1이 인용한 W2-M1 실습 실측값(조건부 오토인코더 44.4%, 순수 $\ell_1$ 회귀 44.7%)을
# `../lesson.md`에서 직접 읽어 옵니다. 이 스크립트의 토이 데이터와 그쪽 실험은 데이터가 다르므로
# **판정하지 않고 참조로만 나란히 찍습니다.**

# %%
def read_lesson_amplitude_claims() -> tuple[float, float] | None:
    """lesson §1에서 W2-M1 진폭 보존율 두 값을 읽는다. 실패하면 None."""
    p = lesson_path()
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8")
    m = re.search(r"조건부 오토인코더\s*([\d.]+)%,\s*순수.*?회귀\s*([\d.]+)%", text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


# %% [markdown]
# ## 5. 혼합비 스윕
#
# lesson §2.1이 "두 모드가 좌우 대칭이면 무게중심은 정확히 가운데"라고 한 자리에서 대칭을 깨봅니다.
# 최소해가 어느 쪽으로 얼마나 움직이는지, 그리고 그때 답이 얼마나 덜 나빠지는지를 봅니다.

# %%
def sweep_ratios(ratios: list[float], n: int, sigma: float, grid_step: float,
                 seed: int) -> list[dict]:
    rows: list[dict] = []
    for p_left in ratios:
        rng = random.Random(seed + int(p_left * 1000))
        demos = make_demos(n, p_left, sigma, rng)

        mean = statistics.fmean(demos)
        med = statistics.median(demos)
        c_mse, v_mse = grid_argmin(lambda c: mse_at(demos, c), -2.0, 2.0, grid_step)
        c_mad, v_mad = grid_argmin(lambda c: mad_at(demos, c), -2.0, 2.0, grid_step)

        d_mse, s_mse = dist_to_nearest_mode(c_mse, sigma)
        d_mad, s_mad = dist_to_nearest_mode(c_mad, sigma)
        depth, cnt_here, cnt_peak = valley_depth(demos, c_mse, -2.0, 2.0, 40)
        depth_l1, _, _ = valley_depth(demos, c_mad, -2.0, 2.0, 40)

        rows.append({
            "p_left": p_left,
            "n": n,
            "sigma": sigma,
            "pop_mean": true_mixture_mean(p_left),
            "sample_mean": mean,
            "sample_median": med,
            "grid_mse_argmin": c_mse,
            "grid_mse_min": v_mse,
            "grid_mad_argmin": c_mad,
            "grid_mad_min": v_mad,
            "mse_gap_to_mean": abs(c_mse - mean),
            "mad_gap_to_median": abs(c_mad - med),
            "mse_dist_sigma": s_mse,
            "mad_dist_sigma": s_mad,
            "mse_snapped_to_mode": s_mse <= SNAP_SIGMA,
            "mad_snapped_to_mode": s_mad <= SNAP_SIGMA,
            "valley_depth": depth,
            "valley_depth_l1": depth_l1,
            "valley_count": cnt_here,
            "peak_count": cnt_peak,
            "amplitude_keep": abs(c_mse) / 1.0,  # 참 모드 진폭 1.0 대비 예측 진폭
        })
    return rows


def median_stability(n_seeds: int, n: int, sigma: float, seed: int) -> tuple[list[float], list[float]]:
    """대칭 혼합에서 표본평균과 표본중앙값이 시드에 따라 얼마나 흔들리는지 본다.

    대칭 이봉 표본의 중앙값은 골짜기 안 어디에서든 잡힐 수 있다. 표본 하나가 넘어가면
    이쪽 모드 꼬리에서 저쪽 모드 꼬리로 건너뛰기 때문이다. 평균에는 이 불안정성이 없다.
    """
    means, meds = [], []
    for s in range(n_seeds):
        rng = random.Random(seed + 7919 * (s + 1))
        d = make_demos(n, 0.5, sigma, rng)
        means.append(statistics.fmean(d))
        meds.append(statistics.median(d))
    return means, meds


# %% [markdown]
# ## 6. CSV 저장

# %%
def write_csv(path: Path, rows: list[dict]) -> Path:
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    return path


# %% [markdown]
# ## 7. 메인

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W2-M2 실습 1: 평균의 함정 검산기 (lesson §2.1, §2.2)")
    p.add_argument("--seed", type=int, default=20260811, help="난수 시드 (기본 20260811)")
    p.add_argument("--n", type=int, default=4000, help="시연 표본 수 (기본 4000)")
    p.add_argument("--sigma", type=float, default=0.15, help="각 모드의 가우시안 잡음 표준편차 (기본 0.15)")
    p.add_argument("--grid-step", type=float, default=0.001, help="격자 탐색 간격 (기본 0.001)")
    p.add_argument("--ratios", type=float, nargs="+", default=[0.5, 0.7, 0.9],
                   help="왼쪽 모드 선택 확률 목록 (기본 0.5 0.7 0.9)")
    p.add_argument("--no-csv", action="store_true", help="CSV 저장을 건너뛴다")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    n_pass = n_check = 0

    print("=" * 100)
    print(f"  {MODULE_ID} 실습 1. 평균의 함정을 수치로 확인한다  (표준 라이브러리만, 의존성 0)")
    print(f"  seed={args.seed}, n={args.n}, sigma={args.sigma}, 격자 간격={args.grid_step}")
    print("=" * 100)

    # --- [1] 균형 혼합에서의 기본 관찰 -------------------------------------
    rng = random.Random(args.seed)
    demos = make_demos(args.n, 0.5, args.sigma, rng)
    mean = statistics.fmean(demos)
    med = statistics.median(demos)
    c_mse, v_mse = grid_argmin(lambda c: mse_at(demos, c), -2.0, 2.0, args.grid_step)
    c_mad, v_mad = grid_argmin(lambda c: mad_at(demos, c), -2.0, 2.0, args.grid_step)

    print("\n=== [1] 좌우 대칭 시연에서 두 손실의 최소해 ===\n")
    print(f"  시연 {args.n}개, 왼쪽 모드 {MODE_LEFT:+.0f} 와 오른쪽 모드 {MODE_RIGHT:+.0f} 가 5대 5")
    print(f"  실제로 왼쪽에 배정된 표본 비율: {sum(1 for a in demos if a < 0) / len(demos):.3f}")
    print()
    tol = args.grid_step * 1.5  # 격자 간격의 절반 + 누적 부동소수 오차
    lo_med, hi_med = mad_minimizer_interval(demos)
    ok_l2 = abs(c_mse - mean) <= tol
    ok_l1 = (lo_med - tol) <= c_mad <= (hi_med + tol)
    print(render_table(
        ["손실", "격자 최소해", "격자 최소값", "닫힌형이 말하는 답", "닫힌형 값", "대조"],
        [
            ["제곱오차 (l2)", f"{c_mse:+.4f}", f"{v_mse:.5f}",
             f"표본평균 {mean:+.4f}", f"{mse_at(demos, mean):.5f}",
             "PASS" if ok_l2 else "FAIL"],
            ["절대편차 (l1)", f"{c_mad:+.4f}", f"{v_mad:.5f}",
             f"중앙 구간 [{lo_med:+.4f}, {hi_med:+.4f}]", f"{mad_at(demos, med):.5f}",
             "PASS" if ok_l1 else "FAIL"],
        ],
        aligns="lrrlrl"))
    n_check += 2
    n_pass += int(ok_l2) + int(ok_l1)
    print("\n  l1 쪽 닫힌형이 점이 아니라 구간인 이유: MAD(c)는 조각별 선형이라 최소가 평평합니다.")
    print(f"  표본 수가 짝수면 가운데 두 순서통계량 사이가 전부 최소해입니다 (폭 {hi_med - lo_med:.4f}).")

    # 항등식 잔차. 격자 탐색보다 이쪽이 §2.1 수식의 진짜 증명이다.
    resid = max(abs(mse_identity_residual(demos, c)) for c in
                [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, mean])
    ok_id = resid < 1e-9
    n_check += 1
    n_pass += int(ok_id)
    print(f"\n  항등식 MSE(c) = Var + (mean - c)^2 의 최대 잔차: {resid:.3e}"
          f"   [{'PASS' if ok_id else 'FAIL'}]")
    print("  이 항등식이 0이면 최소점이 c = 표본평균인 것이 곧바로 따라옵니다. lesson §2.1 수식이 그 진술입니다.")

    # --- [2] 그 답이 어디에 놓이는가 ----------------------------------------
    print("\n=== [2] 최소해가 놓인 자리. 시연이 있는 곳인가 ===\n")
    d_mse, s_mse = dist_to_nearest_mode(c_mse, args.sigma)
    depth, cnt_here, cnt_peak = valley_depth(demos, c_mse, -2.0, 2.0, 40)
    print(ascii_hist(demos, -1.6, 1.6, 32,
                     {"제곱오차 최소해": c_mse, "왼쪽 모드": MODE_LEFT, "오른쪽 모드": MODE_RIGHT}))
    print(f"\n  최소해 {c_mse:+.4f} 에서 최근접 모드까지 거리: {d_mse:.4f}  = 모드 잡음 sigma의 {s_mse:.1f}배")
    print(f"  최소해가 놓인 구간의 표본 수 {cnt_here}개, 최빈 구간 {cnt_peak}개, 골짜기 깊이 {depth:.4f}")
    ok_valley = s_mse > SNAP_SIGMA and depth < 0.05
    n_check += 1
    n_pass += int(ok_valley)
    print(f"  판정: 최소해가 두 모드 사이 골짜기에 놓였는가  [{'PASS' if ok_valley else 'FAIL'}]")
    print("  lesson §2.1: '가운데는 블록 정중앙이고 거기를 밀면 회전이 걸리지 않습니다.'")

    # --- [3] 혼합비 스윕 -----------------------------------------------------
    print("\n=== [3] 혼합비를 바꾸면 최소해가 어디로 가는가 ===\n")
    rows = sweep_ratios(args.ratios, args.n, args.sigma, args.grid_step, args.seed)
    table_rows = []
    for r in rows:
        table_rows.append([
            f"{r['p_left']:.2f}",
            f"{r['pop_mean']:+.3f}",
            f"{r['grid_mse_argmin']:+.4f}",
            f"{r['mse_dist_sigma']:.1f}s",
            "붙음" if r["mse_snapped_to_mode"] else "골짜기",
            f"{r['grid_mad_argmin']:+.4f}",
            f"{r['mad_dist_sigma']:.1f}s",
            "붙음" if r["mad_snapped_to_mode"] else "골짜기",
        ])
    print(render_table(
        ["p_left", "모집단 평균", "l2 최소해", "모드까지", "판정",
         "l1 최소해", "모드까지", "판정"],
        table_rows, aligns="rrrrlrrl"))
    print("\n  '모드까지'는 최근접 모드 중심까지의 거리를 모드 잡음 sigma의 배수로 적은 것입니다.")
    print(f"  '판정'은 그 거리가 {SNAP_SIGMA}sigma 이내면 그 모드에 붙었다고 본 것입니다.")

    # 균형 혼합에서 최소해가 가장 나쁘다. 최근접 모드까지 거리가 최대여야 한다.
    balanced = min(rows, key=lambda r: abs(r["p_left"] - 0.5))
    worst = max(rows, key=lambda r: r["mse_dist_sigma"])
    ok_worst = balanced is worst
    n_check += 1
    n_pass += int(ok_worst)
    print(f"\n  최근접 모드에서 가장 멀리 떨어진 최소해가 나온 혼합비: p_left={worst['p_left']:.2f}"
          f"   [{'PASS' if ok_worst else 'FAIL'}] 균형 혼합에서 가장 나쁜 답")

    # l1은 불균형에서 한쪽 모드로 붙는다. 이것이 §2.2의 논지다.
    unbal = [r for r in rows if abs(r["p_left"] - 0.5) > 0.05]
    ok_l1_snap = all(r["mad_snapped_to_mode"] for r in unbal) if unbal else True
    n_check += 1
    n_pass += int(ok_l1_snap)
    print(f"  불균형 혼합에서 l1 최소해가 한쪽 모드에 붙었는가"
          f"   [{'PASS' if ok_l1_snap else 'FAIL'}]")
    bal_l1 = not balanced["mad_snapped_to_mode"]
    n_check += 1
    n_pass += int(bal_l1)
    print(f"  대칭 혼합에서는 l1 최소해도 어느 모드에도 붙지 못했는가"
          f"   (최근접 모드까지 {balanced['mad_dist_sigma']:.1f}sigma,"
          f" 골짜기 깊이 {balanced['valley_depth_l1']:.4f})"
          f"   [{'PASS' if bal_l1 else 'FAIL'}]")
    print("  → lesson §2.2: 문제는 손실 함수의 종류가 아니라 출력이 값 하나라는 사실입니다.")
    print("     l1으로 바꾸면 불균형일 때 한쪽 모드로 붙지만, 그것은 '다수결로 한 모드를 버리는 것'이지")
    print("     여러 모드를 표현한 것이 아닙니다. 대칭이면 그 도피처마저 없습니다.")

    # --- [3b] 대칭 혼합에서 중앙값은 불안정하다 ------------------------------
    print("\n=== [3b] 대칭 혼합에서 중앙값이 흔들리는 폭 ===\n")
    means9, meds9 = median_stability(9, args.n, args.sigma, args.seed)
    spread_mean = max(means9) - min(means9)
    spread_med = max(meds9) - min(meds9)
    print(render_table(
        ["통계량", "시드 9개 최소", "최대", "폭", "폭 / sigma"],
        [
            ["표본평균", f"{min(means9):+.4f}", f"{max(means9):+.4f}",
             f"{spread_mean:.4f}", f"{spread_mean / args.sigma:.2f}"],
            ["표본중앙값", f"{min(meds9):+.4f}", f"{max(meds9):+.4f}",
             f"{spread_med:.4f}", f"{spread_med / args.sigma:.2f}"],
        ],
        aligns="lrrrr"))
    ok_unstable = spread_med > 5.0 * spread_mean
    n_check += 1
    n_pass += int(ok_unstable)
    print(f"\n  중앙값의 흔들림이 평균의 {spread_med / spread_mean:.1f}배"
          f"   [{'PASS' if ok_unstable else 'FAIL'}] 5배 초과를 불안정으로 본다")
    print("  lesson §2.1은 모집단 기준으로 '대칭이면 중앙값도 가운데'라고 적었습니다. 대칭 이봉 모집단의")
    print("  중앙값은 정확히 0입니다. 그런데 유한 표본에서는 표본 하나가 넘어갈 때마다 중앙값이")
    print("  이쪽 모드 꼬리에서 저쪽 모드 꼬리로 건너뜁니다. 골짜기를 벗어나지는 않으니 결론은 같고,")
    print("  덧붙는 사실이 하나 생깁니다. l1 회귀의 답은 대칭 이봉에서 재현조차 되지 않습니다.")

    # --- [4] 진폭 보존율. W2-M1 실측과 나란히 --------------------------------
    print("\n=== [4] 진폭 보존율. 앞 모듈 실측과 나란히 놓기 ===\n")
    keep = balanced["amplitude_keep"]
    print(f"  참 모드 진폭 1.0 대비 이 토이 회귀의 예측 진폭: {abs(balanced['grid_mse_argmin']):.4f}"
          f"  = 보존율 {keep * 100:.1f}%")
    claims = read_lesson_amplitude_claims()
    if claims:
        print(f"  lesson §1이 인용한 W2-M1 실습 실측: 조건부 오토인코더 {claims[0]:.1f}%, "
              f"순수 l1 회귀 {claims[1]:.1f}%")
        print("  두 실험은 데이터도 모델도 다르므로 판정하지 않고 참조로만 둡니다.")
        print("  같은 것은 방향뿐입니다. 배포 경로가 값 하나면 진폭이 깎입니다.")
    else:
        print("  (lesson.md를 찾지 못해 W2-M1 실측 인용을 건너뜁니다)")

    # --- [5] CSV -------------------------------------------------------------
    if not args.no_csv:
        path = write_csv(artifacts_dir() / "01_mean_collapse.csv", rows)
        print(f"\n[저장] {path}  ({len(rows)}행)")

    print("\n" + "=" * 100)
    print(f"  대조 결과: {n_pass}/{n_check} PASS")
    print("  요점: 제곱오차의 최소해는 표본평균이고, 대칭 이봉 시연에서 그 자리에는 시연이 없다.")
    print("        l1으로 바꾸면 중앙값이 되어 불균형일 때 한쪽 모드로 붙지만 여전히 값 하나다.")
    print("        추론 자체를 샘플링으로 바꾸는 것이 lesson §3의 처방이다.")
    print("        다음 → 02_receding_horizon.py")
    print("=" * 100)
    return 0 if n_pass == n_check else 1


# %%
if __name__ == "__main__":
    import sys

    raise SystemExit(main(None if "ipykernel" not in sys.modules else []))
