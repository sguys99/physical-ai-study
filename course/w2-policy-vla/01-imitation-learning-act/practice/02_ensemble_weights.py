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
# # W2-M1 실습 2, temporal ensembling 가중과 $n_{\text{eff}}$ 검산
#
# [`../lesson.md`](../lesson.md) `§3.5`(지수 가중의 정의)와 `§6.4`(계수별 표 5행)를 재현합니다.
#
# **이 스크립트가 특히 중요한 이유**: lesson 집필 중 실제로 오류가 잡힌 지점입니다
# (`docs/course-plan.md` §9.7의 "검증이 잡아낸 것" 표 첫 행). §3.5가 **가중 총합 $S$의 닫힌형**만
# 주는데 §6.4 표의 마지막 열은 실제로 **$S \div \max_i w_i$** 였습니다.
# $m>0$이면 최대 가중이 $w_0=1$이라 두 값이 우연히 같아 문제가 숨어 있었고,
# $m<0$에서 갈립니다. $m=-0.01$에서 $S=171.0$인데 $n_{\text{eff}}=63.5$입니다.
# **이 스크립트는 그 구분을 매번 눈앞에 찍습니다.**
#
# 확인할 것:
#
# 1. **가중 규약.** $w_i = e^{-mi}$이고 **$w_0$가 가장 오래된 예측**
#    (LeRobot `ACTTemporalEnsembler` docstring: *"where w₀ is the oldest action"*)
#    $$a_t^{\text{exec}} = \frac{\sum_{i=0}^{n-1} w_i \hat a_t^{(i)}}{\sum_{i=0}^{n-1} w_i},
#      \qquad w_i = \exp(-m i) \tag{§3.5}$$
# 2. **닫힌형과 직접 합산의 일치**
#    $$S = \sum_{i=0}^{n-1} e^{-mi} = \frac{1 - e^{-mn}}{1 - e^{-m}} \tag{§3.5}$$
# 3. **유효 기여 수**는 $S$가 아니다
#    $$n_{\text{eff}} = \frac{S}{\max_i w_i} \tag{§3.5}$$
# 4. **§6.4 표 5행**($m = -0.1, -0.01, 0, 0.01, 0.1$, $n=100$)을 재현해 자동 대조
#
# 출력:
# - stdout에 §6.4 표 재현과 PASS/FAIL과 ASCII 가중 분포
# - `artifacts/W2-M1/02_ensemble_weights.csv`
#
# **의존성 0. 표준 라이브러리만 씁니다.**

# %%
from __future__ import annotations

import argparse
import csv
import math
import unicodedata
from pathlib import Path

MODULE_ID = "W2-M1"

# lesson §6.4 표의 ground truth (n = 100).
#   (m, w0, w99, 비 w0/w99, n_eff, 성격)
LESSON_S64_ROWS: list[tuple[float, float, float, float, float, str]] = [
    (-0.1, 1.0, 1.99e4, 5.0e-5, 10.5, "최신 예측 단독에 근접 → 순수 재계획과 거의 같음"),
    (-0.01, 1.0, 2.69, 0.372, 63.5, "최신 쪽으로 살짝 기울인 평균"),
    (0.0, 1.0, 1.0, 1.0, 100.0, "균등 평균"),
    (0.01, 1.0, 0.372, 2.69, 63.5, "권장값 · 과거 쪽으로 살짝 기울인 거의 균등 평균"),
    (0.1, 1.0, 5.0e-5, 1.99e4, 10.5, "최초 예측 단독에 근접 → 새 관측이 거의 반영 안 됨"),
]

# lesson §3.5 본문이 명시한 갈림 사례: m=-0.01, n=100 에서 S=171.0 인데 n_eff=63.5
LESSON_S_AT_M_NEG001 = 171.0

REL_TOL = 0.01  # lesson 표기가 3자리 유효숫자라 1% 상대오차면 충분하다


# %% [markdown]
# ## 0. 경로와 표 유틸 (01과 동일 규약)

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
    if align == "r":
        return " " * gap + s
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
# ## 1. 가중과 총합과 유효 기여 수
#
# 세 함수가 전부입니다. 어려운 것은 산술이 아니라 **$w_0$가 어느 쪽인가**와
# **$S$와 $n_{\text{eff}}$가 다른 양이라는 것**입니다.

# %%
def weights(m: float, n: int) -> list[float]:
    """w_i = exp(-m i), i=0 이 가장 오래된 예측.  # eq.(§3.5)

    LeRobot ACTTemporalEnsembler docstring 규약: "where w_0 is the oldest action".
    따라서 m > 0 이면 오래된 예측에 큰 가중이 붙습니다 — 직관과 반대라 자주 틀리는 자리입니다.
    """
    return [math.exp(-m * i) for i in range(n)]


def weight_sum_direct(m: float, n: int) -> float:
    """직접 합산."""
    return math.fsum(weights(m, n))


def weight_sum_closed(m: float, n: int) -> float:
    """등비급수 닫힌형 S = (1 - e^{-mn}) / (1 - e^{-m}).  # eq.(§3.5)

    m = 0 은 공비가 1이라 0/0 이 됩니다. 극한값 n 을 씁니다.
    """
    if abs(m) < 1e-15:
        return float(n)
    return (1.0 - math.exp(-m * n)) / (1.0 - math.exp(-m))


def n_effective(m: float, n: int) -> float:
    """유효 기여 수 n_eff = S / max_i w_i.  # eq.(§3.5)

    "실질적으로 몇 개의 예측이 기여하는가". S 자체가 아닙니다 —
    m > 0 이면 max w = w_0 = 1 이라 우연히 같아지지만 m < 0 이면 갈립니다.
    """
    w = weights(m, n)
    return math.fsum(w) / max(w)


def ratio_oldest_newest(m: float, n: int) -> float:
    """w_0 / w_{n-1} = exp(m(n-1)). '지수 계수는 m x n 으로 읽어야 한다'의 근거."""
    w = weights(m, n)
    return w[0] / w[-1]


# %% [markdown]
# ## 2. lesson §6.4 표 재현 + 자동 대조

# %%
def close_enough(got: float, want: float, rel: float = REL_TOL) -> bool:
    if want == 0.0:
        return abs(got) < rel
    return abs(got - want) / abs(want) <= rel


def verify_lesson_table(n: int = 100) -> tuple[list[list[str]], int, int, list[str]]:
    rows: list[list[str]] = []
    notes: list[str] = []
    n_pass = n_check = 0

    for m, w0_x, wlast_x, ratio_x, neff_x, character in LESSON_S64_ROWS:
        w = weights(m, n)
        got_w0, got_wlast = w[0], w[-1]
        got_ratio = ratio_oldest_newest(m, n)
        got_neff = n_effective(m, n)
        got_S = weight_sum_direct(m, n)

        checks = [
            close_enough(got_w0, w0_x),
            close_enough(got_wlast, wlast_x),
            close_enough(got_ratio, ratio_x),
            close_enough(got_neff, neff_x),
        ]
        n_check += len(checks)
        n_pass += sum(checks)

        # m < 0 이면 S 와 n_eff 가 갈린다. 이 표의 존재 이유
        split = "" if m >= 0 else f"S={got_S:,.1f} ≠ n_eff"
        rows.append([
            f"{m:+.2f}" if m else " 0.00",
            f"{got_w0:.3g}",
            f"{got_wlast:.3g}",
            f"{got_ratio:.3g}",
            f"{got_S:,.1f}",
            f"{got_neff:.1f}",
            "PASS" if all(checks) else "FAIL",
            character,
        ])
        if split:
            notes.append(f"  m={m:+g}: {split}  (총합 {got_S:,.1f} vs 유효 기여 수 {got_neff:.1f})")
    return rows, n_pass, n_check, notes


# %% [markdown]
# ## 3. 닫힌형 vs 직접 합산
#
# 두 경로가 같은 값을 내는지 확인합니다. 부동소수 오차만 허용합니다.
# $m=0$은 닫힌형이 $0/0$이므로 극한값 $n$으로 분기합니다. 코드에서 자주 터지는 자리입니다.

# %%
def verify_closed_form(m_list: list[float], n_list: list[int]) -> tuple[list[list[str]], int, int]:
    rows: list[list[str]] = []
    n_pass = n_check = 0
    for m in m_list:
        for n in n_list:
            direct = weight_sum_direct(m, n)
            closed = weight_sum_closed(m, n)
            rel = abs(direct - closed) / max(abs(direct), 1e-300)
            ok = rel < 1e-9
            n_check += 1
            n_pass += int(ok)
            rows.append([
                f"{m:+g}", str(n), f"{direct:,.6g}", f"{closed:,.6g}",
                f"{rel:.2e}", "PASS" if ok else "FAIL",
            ])
    return rows, n_pass, n_check


# %% [markdown]
# ## 4. ASCII 가중 분포
#
# matplotlib을 쓰지 않습니다(의존성 0 유지). 대신 터미널에 막대를 그립니다.
# **왼쪽이 $i=0$ = 가장 오래된 예측**이라는 것을 축 라벨에 박아둡니다.

# %%
def ascii_bars(m: float, n: int, n_bins: int = 40, width: int = 56) -> str:
    """가중 분포를 ASCII 막대로. 예측 n개를 n_bins개 구간으로 묶어 평균 가중을 그린다."""
    w = weights(m, n)
    wmax = max(w)
    n_bins = min(n_bins, n)
    lines: list[str] = []
    for b in range(n_bins):
        lo = b * n // n_bins
        hi = max(lo + 1, (b + 1) * n // n_bins)
        seg = w[lo:hi]
        avg = math.fsum(seg) / len(seg)
        frac = avg / wmax
        bar = "#" * max(0 if frac < 1e-4 else 1, round(frac * width))
        lines.append(f"  i={lo:>4}..{hi - 1:<4} |{bar:<{width}}| {avg:.4g}")
    head = (f"  m = {m:+g},  n = {n}   (막대 길이 = 가중 / 최대가중)\n"
            f"  위 = i 작음 = 가장 오래된 예측  |  아래 = i 큼 = 가장 최신 예측")
    return head + "\n" + "\n".join(lines)


# %% [markdown]
# ## 5. CSV 저장

# %%
def write_csv(path: Path, m_list: list[float], n_list: list[int]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fp:
        wr = csv.writer(fp)
        wr.writerow([
            "m", "n", "w_oldest", "w_newest", "ratio_oldest_over_newest",
            "S_direct", "S_closed_form", "n_eff", "S_equals_n_eff",
        ])
        for m in m_list:
            for n in n_list:
                w = weights(m, n)
                direct = weight_sum_direct(m, n)
                closed = weight_sum_closed(m, n)
                neff = n_effective(m, n)
                wr.writerow([
                    f"{m:g}", n, f"{w[0]:.6g}", f"{w[-1]:.6g}",
                    f"{ratio_oldest_newest(m, n):.6g}",
                    f"{direct:.6g}", f"{closed:.6g}", f"{neff:.6g}",
                    int(abs(direct - neff) / max(direct, 1e-300) < 1e-9),
                ])
    return path


# %% [markdown]
# ## 6. 메인

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W2-M1 실습 2: temporal ensembling 가중과 n_eff 검산 — lesson §3.5·§6.4")
    p.add_argument("--m", type=float, default=0.01,
                   help="temporal_ensemble_coeff (기본 0.01 = LeRobot 권장값)")
    p.add_argument("--n", type=int, default=100,
                   help="쌓인 예측 수 (기본 100 = chunk_size 상한)")
    p.add_argument("--bins", type=int, default=20, help="ASCII 막대 구간 수 (기본 20)")
    p.add_argument("--no-csv", action="store_true", help="CSV 저장을 건너뛴다")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = artifacts_dir()

    print("=" * 100)
    print(f"  {MODULE_ID} 실습 2 — temporal ensembling 가중  (표준 라이브러리만 · 의존성 0)")
    print("=" * 100)

    # --- [1] 규약 확인 -------------------------------------------------------
    print("\n=== [1] 가중 규약 — w_0 는 '가장 오래된' 예측이다 ===\n")
    w_demo = weights(0.01, 5)
    print("  m = 0.01, n = 5 일 때 w = [" + ", ".join(f"{v:.4f}" for v in w_demo) + "]")
    print("  → i가 커질수록 가중이 작아집니다. i=0 이 최고령이므로 **m>0 은 과거를 더 믿는 설정**입니다.")
    print("     LeRobot ACTTemporalEnsembler docstring: \"where w_0 is the oldest action\"")

    # --- [2] 닫힌형 대조 -----------------------------------------------------
    print("\n=== [2] 닫힌형 S = (1 - e^{-mn}) / (1 - e^{-m}) vs 직접 합산 ===\n")
    cf_rows, cf_pass, cf_check = verify_closed_form(
        sorted({-0.1, -0.01, 0.0, 0.01, 0.1, args.m}), sorted({10, args.n}))
    print(render_table(["m", "n", "직접 합산", "닫힌형", "상대오차", "대조"],
                       cf_rows, aligns="rrrrrl"))
    print(f"\n  {cf_pass}/{cf_check} PASS  "
          f"(m=0 은 닫힌형이 0/0 이라 극한값 n 으로 분기 — 구현에서 자주 터지는 자리)")

    # --- [3] §6.4 표 재현 ----------------------------------------------------
    print(f"\n=== [3] lesson §6.4 표 재현 (n = {args.n}) — 문서의 숫자를 코드가 검산한다 ===\n")
    rows, n_pass, n_check, notes = verify_lesson_table(args.n)
    print(render_table(
        ["m", "w_0(최고령)", "w_last(최신)", "비 w0/wlast", "총합 S", "n_eff", "대조", "성격"],
        rows, aligns="rrrrrrll"))
    print(f"\n  대조 결과: {n_pass}/{n_check} PASS"
          f"{'  ← lesson §6.4와 일치 (상대오차 1% 이내)' if n_pass == n_check else '  ← ❌ 불일치'}")

    # --- [4] S 와 n_eff 가 갈리는 지점 ---------------------------------------
    print("\n=== [4] S 와 n_eff 는 다른 양이다 — m < 0 에서 갈린다 ===\n")
    for note in notes:
        print(note)
    s_neg = weight_sum_direct(-0.01, 100)
    neff_neg = n_effective(-0.01, 100)
    ok_split = close_enough(s_neg, LESSON_S_AT_M_NEG001) and close_enough(neff_neg, 63.5)
    print(f"\n  lesson §3.5 본문: \"m=-0.01 에서 S=171.0 인데 n_eff=63.5\"")
    print(f"  계산값          : S={s_neg:,.1f} · n_eff={neff_neg:.1f}   "
          f"[{'PASS' if ok_split else 'FAIL'}]")
    print("  → m > 0 이면 max w = w_0 = 1 이라 S 와 n_eff 가 우연히 같습니다.")
    print("     m < 0 이면 최대가 최신 쪽(w_{n-1})으로 옮겨가 둘이 갈립니다.")
    print("     §6.4 열 이름이 '유효 기여 수 n_eff' 인 이유이고, 실제로 집필 중 잡힌 오류 지점입니다.")
    n_check += 1
    n_pass += int(ok_split)

    # --- [5] m x n 으로 읽기 -------------------------------------------------
    print("\n=== [5] 계수는 m 이 아니라 m x n 으로 읽어야 한다 ===\n")
    mn_rows = []
    for m in (0.01, 0.1):
        for n in (10, 50, 100):
            mn_rows.append([f"{m:g}", str(n), f"{m * n:g}",
                            f"{ratio_oldest_newest(m, n):,.3g}", f"{n_effective(m, n):.1f}"])
    print(render_table(["m", "n", "m x n", "가중비 w0/wlast", "n_eff"], mn_rows, aligns="rrrrr"))
    print("\n  → 같은 m=0.1 이라도 n=10 이면 가중비 2.5배(거의 균등)인데 n=100 이면 2만 배입니다.")
    print("     n 은 chunk_size 까지 자라므로 '0.1은 작은 값'이라는 직관이 자릿수를 틀립니다(lesson §6.4).")

    # --- [6] ASCII 가중 분포 -------------------------------------------------
    print(f"\n=== [6] 가중 분포 — m = {args.m:g}, n = {args.n} ===\n")
    print(ascii_bars(args.m, args.n, n_bins=args.bins))

    # --- [7] CSV -------------------------------------------------------------
    if not args.no_csv:
        m_list = sorted({-0.1, -0.01, 0.0, 0.01, 0.1, args.m})
        n_list = sorted({10, 25, 50, 100, args.n})
        path = write_csv(out_dir / "02_ensemble_weights.csv", m_list, n_list)
        print(f"\n[저장] {path}  ({len(m_list) * len(n_list)}행)")

    print("\n" + "=" * 100)
    print("  요점: 권장값 0.01 은 감쇠가 아니라 '거의 균등 평균'이다(가중비 2.69, n_eff 63.5).")
    print("        0.1 은 작아 보이지만 최초 예측 단독에 가깝다 — 새 관측이 거의 반영되지 않는다.")
    print("        그리고 이 기능을 켜는 순간 n_action_steps=1 이 강제되어 지연 예산이 100분의 1이 된다(실습 01).")
    print("        다음 → 03_compounding_error.py")
    print("=" * 100)
    return 0 if n_pass == n_check else 1


# %%
if __name__ == "__main__":
    import sys

    raise SystemExit(main(None if "ipykernel" not in sys.modules else []))
