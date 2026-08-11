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
# # W2-M1 실습 1, 지연 예산 계산기가 lesson §3.6 표를 검산한다
#
# [`../lesson.md`](../lesson.md) `§3.6`의 표를 **재현**하고, 임의 파라미터로 **다시 계산**합니다.
#
# 이 스크립트의 성격은 계산기가 아니라 **검산기**입니다. lesson 본문에 박혀 있는 숫자
# (3,333 ms, 1,667 ms, 333 ms, 33 ms, 0.3 Hz, 30 Hz 같은 값)를 코드가 다시 계산해
# **자동 대조하고 PASS/FAIL을 찍습니다.** 문서의 숫자와 코드의 숫자가 갈리면 여기서 걸립니다.
#
# 확인할 것:
#
# 1. **허용 지연 예산.** 좌변에 들어가는 것은 `chunk_size`가 아니라 `n_action_steps`다
#    $$T_{\text{replan}} + \tau_{\text{infer}} + \tau_{\text{comm}} \;\le\; \frac{\texttt{n\_action\_steps}}{f_2} \tag{§3.6}$$
# 2. **최악 반응 지연**과 **추론 호출 빈도**
#    $$\tau_{\text{react}}^{\max} = \frac{\texttt{n\_action\_steps}}{f_2}, \qquad
#      f_{\text{infer}} = \frac{f_2}{\texttt{n\_action\_steps}} \tag{§3.6}$$
# 3. **200 ms 문턱**([W1-M1 §3.3](../../../w1-generative-core/01-physical-ai-landscape/lesson.md),
#    "발이 미끄러져 자세가 무너지기까지 200 ms")과의 대조. **각 설정이 전신 균형 태스크에서 안전한가.**
# 4. `--sweep`은 `n_action_steps`를 1~100으로 훑어 200 ms 문턱을 넘는 **경계값**을 찾는다
#
# 출력:
# - stdout에 §3.6 표 재현(30 Hz와 50 Hz 병기)과 PASS/FAIL
# - `artifacts/W2-M1/01_latency_budget.csv`
#
# **의존성 0. 표준 라이브러리만 씁니다.** `python3 01_latency_budget.py` 로 즉시 돌아갑니다.

# %%
from __future__ import annotations

import argparse
import csv
import unicodedata
from pathlib import Path

MODULE_ID = "W2-M1"

# W1-M1 §3.3의 "발이 미끄러져 자세가 무너지기까지 200 ms". 전신 균형 태스크의 개방루프 창 상한.
BALANCE_THRESHOLD_MS = 200.0

# lesson §3.6 표의 ground truth. 여기가 어긋나면 구현이 아니라 문서(또는 이 상수)가 틀린 것이다.
#   (라벨, chunk_size, n_action_steps, 예산@30Hz[ms], 예산@50Hz[ms], 추론빈도@30Hz[Hz], 최악반응@30Hz[ms])
LESSON_S36_ROWS: list[tuple[str, int, int, int, int, float, int]] = [
    ("LeRobot 기본값", 100, 100, 3333, 2000, 0.3, 3333),
    ("docstring 예시 receding horizon", 100, 50, 1667, 1000, 0.6, 1667),
    ("짧은 receding horizon", 100, 10, 333, 200, 3.0, 333),
    ("temporal ensembling(강제)", 100, 1, 33, 20, 30.0, 33),
]

# lesson §3.6 둘째 문단과 퀴즈 8번의 "200 ms 문턱과 비교하면 약 16배 밖"
LESSON_BALANCE_RATIO_FLOOR = 16.0


# %% [markdown]
# ## 0. 경로와 표 유틸 (W1-M5 practice와 동일 규약)

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
    line = "  " + "  ".join(pad(h, widths[i], aligns[i]) for i, h in enumerate(headers))
    sep = "  " + "  ".join("-" * w for w in widths)
    body = [
        "  " + "  ".join(pad(c, widths[i], aligns[i]) for i, c in enumerate(r))
        for r in rows
    ]
    return "\n".join([line, sep, *body])


# %% [markdown]
# ## 1. 핵심 산술은 세 줄이면 끝난다
#
# lesson §3.6이 강조하는 것은 이 식이 어렵다는 게 아니라 **좌변에 무엇이 들어가느냐**입니다.
# `chunk_size`가 아니라 `n_action_steps`입니다. 예측만 하고 버리는 뒷부분은 지연을 버텨주지 않으니까요.
#
# `허용 지연 예산`과 `최악 반응 지연`은 **같은 식**인데 뜻이 다릅니다.
# 앞은 "버퍼에 쌓인 명령이 몇 ms를 버티나"(공급), 뒤는 "새 관측이 명령에 반영되기까지 몇 ms"(신선도)입니다.
# 청크를 끝까지 실행하는 설정에서는 두 값이 같은 숫자가 됩니다.

# %%
def latency_budget_ms(n_action_steps: int, f2_hz: float) -> float:
    """허용 지연 예산 [ms] = n_action_steps / f2.  # eq.(§3.6)"""
    return n_action_steps / f2_hz * 1000.0


def worst_reaction_ms(n_action_steps: int, f2_hz: float) -> float:
    """최악 반응 지연 [ms]. 실행 길이만큼 새 관측을 못 본다 → 예산과 같은 식.  # eq.(§3.6)"""
    return n_action_steps / f2_hz * 1000.0


def inference_rate_hz(n_action_steps: int, f2_hz: float) -> float:
    """추론 호출 빈도 [Hz] = f2 / n_action_steps.  # eq.(§3.6)"""
    return f2_hz / n_action_steps


def balance_verdict(worst_ms: float, threshold_ms: float = BALANCE_THRESHOLD_MS) -> tuple[bool, float]:
    """전신 균형 태스크(W1-M1 §3.3의 200 ms 문턱)에서 안전한가. (안전여부, 문턱 대비 배수)"""
    return worst_ms <= threshold_ms, worst_ms / threshold_ms


# %% [markdown]
# ## 2. lesson §3.6 표 재현 + 자동 대조
#
# 4행을 30 Hz와 50 Hz 양쪽으로 계산하고, `LESSON_S36_ROWS`에 박아둔 lesson의 값과 대조합니다.
# **반올림 규칙은 lesson 표기와 같은 정수 ms 반올림**입니다.

# %%
def verify_lesson_table(threshold_ms: float = BALANCE_THRESHOLD_MS) -> tuple[list[list[str]], int, int]:
    """§3.6 표를 재계산해 lesson 값과 대조한다. (표 행, 통과 수, 검사 수)"""
    rows: list[list[str]] = []
    n_pass = n_check = 0

    for label, chunk, n_act, b30, b50, rate30, react30 in LESSON_S36_ROWS:
        got_b30 = latency_budget_ms(n_act, 30.0)
        got_b50 = latency_budget_ms(n_act, 50.0)
        got_rate30 = inference_rate_hz(n_act, 30.0)
        got_react30 = worst_reaction_ms(n_act, 30.0)

        checks = [
            (round(got_b30) == b30),
            (round(got_b50) == b50),
            (abs(got_rate30 - rate30) < 1e-9),
            (round(got_react30) == react30),
        ]
        n_check += len(checks)
        n_pass += sum(checks)
        ok = all(checks)

        safe, ratio = balance_verdict(got_react30, threshold_ms)
        verdict = "안전" if safe else f"초과 {ratio:.1f}배"

        rows.append([
            label,
            str(chunk),
            str(n_act),
            f"{got_b30:,.0f}",
            f"{got_b50:,.0f}",
            f"{got_rate30:g}",
            f"{got_react30:,.0f}",
            verdict,
            "PASS" if ok else "FAIL",
        ])
    return rows, n_pass, n_check


# %% [markdown]
# ## 3. 임의 파라미터 재계산
#
# `--f2`와 `--chunk-size`와 `--n-action-steps`로 아무 설정이나 넣어봅니다.
# `n_action_steps ≤ chunk_size`는 LeRobot 설정 검증이 강제하는 유일한 제약입니다(lesson §2.5).

# %%
def describe_config(chunk_size: int, n_action_steps: int, f2_hz: float,
                    threshold_ms: float = BALANCE_THRESHOLD_MS) -> list[str]:
    lines: list[str] = []
    if n_action_steps > chunk_size:
        lines.append(
            f"  ⚠️  n_action_steps({n_action_steps}) > chunk_size({chunk_size}) — "
            "LeRobot 설정 검증에서 거부되는 조합입니다(lesson §2.5)."
        )
    if n_action_steps != 1:
        lines.append(
            f"  ℹ️  temporal ensembling을 켜려면 n_action_steps=1이어야 합니다"
            f"(현재 {n_action_steps}). lesson §2.6."
        )

    budget = latency_budget_ms(n_action_steps, f2_hz)
    react = worst_reaction_ms(n_action_steps, f2_hz)
    rate = inference_rate_hz(n_action_steps, f2_hz)
    safe, ratio = balance_verdict(react, threshold_ms)
    discarded = chunk_size - n_action_steps

    lines += [
        f"  f2 = {f2_hz:g} Hz · chunk_size = {chunk_size} · n_action_steps = {n_action_steps}",
        f"    예측하고 버리는 스텝  : {discarded}개 "
        f"({'개방루프 완주' if discarded == 0 else 'receding horizon'})",
        f"    허용 지연 예산        : {budget:,.1f} ms   (T_replan + tau_infer + tau_comm 의 합이 이 안이어야)",
        f"    최악 반응 지연        : {react:,.1f} ms   (새 관측이 명령에 반영되기까지)",
        f"    추론 호출 빈도        : {rate:g} Hz     "
        f"({1000.0 / rate:,.1f} ms 마다 1회)" if rate > 0 else "",
        f"    200 ms 문턱 대비      : {'안전' if safe else f'초과 {ratio:.2f}배'}  "
        f"(전신 균형 태스크 기준 · W1-M1 §3.3)",
    ]
    return [ln for ln in lines if ln]


# %% [markdown]
# ## 4. 문턱 경계 스윕
#
# `n_action_steps`를 1부터 `chunk_size`까지 훑어 **200 ms 문턱을 만족하는 최대값**을 찾습니다.
# 30 Hz에서 답은 손으로도 나옵니다. $n/30 \le 0.2\,\mathrm{s} \Rightarrow n \le 6$.
# 코드가 그 6을 찾아내는지 보는 것이 이 절의 목적입니다.

# %%
def sweep_threshold(chunk_size: int, f2_list: list[float],
                    threshold_ms: float = BALANCE_THRESHOLD_MS) -> list[tuple[float, int | None, float]]:
    """각 f2에 대해 문턱을 만족하는 최대 n_action_steps를 찾는다."""
    out: list[tuple[float, int | None, float]] = []
    for f2 in f2_list:
        best: int | None = None
        for n in range(1, chunk_size + 1):
            if worst_reaction_ms(n, f2) <= threshold_ms + 1e-9:
                best = n
            else:
                break
        margin = worst_reaction_ms(best, f2) if best else float("nan")
        out.append((f2, best, margin))
    return out


# %% [markdown]
# ## 5. CSV 저장

# %%
def write_csv(path: Path, chunk_size: int, n_list: list[int], f2_list: list[float],
              threshold_ms: float = BALANCE_THRESHOLD_MS) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow([
            "chunk_size", "n_action_steps", "f2_hz",
            "latency_budget_ms", "worst_reaction_ms", "inference_rate_hz",
            "balance_threshold_ms", "safe_for_whole_body_balance", "ratio_vs_threshold",
        ])
        for f2 in f2_list:
            for n in n_list:
                react = worst_reaction_ms(n, f2)
                safe, ratio = balance_verdict(react, threshold_ms)
                w.writerow([
                    chunk_size, n, f"{f2:g}",
                    f"{latency_budget_ms(n, f2):.3f}", f"{react:.3f}",
                    f"{inference_rate_hz(n, f2):.6g}",
                    f"{threshold_ms:g}", int(safe), f"{ratio:.4f}",
                ])
    return path


# %% [markdown]
# ## 6. 메인

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W2-M1 실습 1: 지연 예산 계산기 — lesson §3.6 표 검산")
    p.add_argument("--f2", type=float, default=30.0,
                   help="하위 소비 주파수 f2 [Hz] (기본 30)")
    p.add_argument("--chunk-size", type=int, default=100,
                   help="예측하는 스텝 수 (기본 100 = LeRobot 기본값)")
    p.add_argument("--n-action-steps", type=int, default=100,
                   help="실행하는 스텝 수 (기본 100 = 개방루프 완주)")
    p.add_argument("--threshold-ms", type=float, default=BALANCE_THRESHOLD_MS,
                   help="전신 균형 문턱 [ms] (기본 200 = W1-M1 §3.3)")
    p.add_argument("--sweep", action="store_true",
                   help="n_action_steps 1~chunk_size 스윕으로 문턱 경계값을 찾는다")
    p.add_argument("--no-csv", action="store_true", help="CSV 저장을 건너뛴다")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = artifacts_dir()

    print("=" * 96)
    print(f"  {MODULE_ID} 실습 1 — 지연 예산 계산기  (표준 라이브러리만 · 의존성 0)")
    print("=" * 96)

    # --- [1] lesson §3.6 표 재현 + 대조 -------------------------------------
    print("\n=== [1] lesson §3.6 표 재현 — 문서의 숫자를 코드가 검산한다 ===\n")
    rows, n_pass, n_check = verify_lesson_table(args.threshold_ms)
    print(render_table(
        ["설정", "chunk", "n_act", "예산@30Hz", "@50Hz", "추론@30Hz", "최악반응@30Hz",
         f"{args.threshold_ms:g}ms 문턱", "대조"],
        rows,
        aligns="lrrrrrrll",
    ))
    print(f"\n  단위: 예산·반응 [ms] · 추론 [Hz]")
    print(f"  대조 결과: {n_pass}/{n_check} PASS"
          f"{'  ← lesson §3.6과 완전 일치' if n_pass == n_check else '  ← ❌ 불일치 발생'}")

    # 200 ms 문턱 대비 배수. lesson이 '약 16배 밖'이라고 쓴 값
    react_default = worst_reaction_ms(100, 30.0)
    ratio = react_default / args.threshold_ms
    ok_ratio = LESSON_BALANCE_RATIO_FLOOR <= ratio < LESSON_BALANCE_RATIO_FLOOR + 1
    print(f"\n  기본값 최악 반응 지연 {react_default:,.1f} ms ÷ 문턱 {args.threshold_ms:g} ms "
          f"= {ratio:.2f}배   [{'PASS' if ok_ratio else 'FAIL'}] lesson '약 16배 밖'")
    print("  → ALOHA식 고정 베이스에는 낙상 모드가 없어 문제가 되지 않습니다(lesson §3.6·§6.2).")
    print("     같은 숫자가 휴머노이드 전신 균형에서는 넘어짐입니다.")

    # --- [2] 임의 설정 재계산 ------------------------------------------------
    print("\n=== [2] 지정한 설정으로 다시 계산 ===\n")
    for line in describe_config(args.chunk_size, args.n_action_steps, args.f2, args.threshold_ms):
        print(line)

    # --- [3] 문턱 경계 스윕 --------------------------------------------------
    if args.sweep:
        print("\n=== [3] --sweep : 200 ms 문턱을 만족하는 최대 n_action_steps ===\n")
        sw = sweep_threshold(args.chunk_size, [20.0, 30.0, 50.0, 100.0, 200.0], args.threshold_ms)
        sw_rows = [
            [f"{f2:g}", str(best) if best else "없음",
             f"{margin:,.1f}" if best else "-",
             f"{worst_reaction_ms(best + 1, f2):,.1f}" if best and best < args.chunk_size else "-"]
            for f2, best, margin in sw
        ]
        print(render_table(
            ["f2 [Hz]", "최대 n_act", "그때 반응지연[ms]", "n_act+1이면[ms]"],
            sw_rows, aligns="rrrr"))
        b30 = dict((f2, best) for f2, best, _ in sw).get(30.0)
        print(f"\n  30 Hz 경계값 = {b30}   (손계산: n/30 ≤ 0.2 s ⟹ n ≤ 6)"
              f"   [{'PASS' if b30 == 6 else 'FAIL'}]")
        print("  → 전신 균형이 걸린 태스크라면 청크를 예측은 100개 하더라도 "
              "실행은 6개까지만 하고 재계획해야 한다는 뜻입니다.")
        print("     그러면 추론 호출이 "
              f"{inference_rate_hz(6, 30.0):.1f} Hz가 되고, 그 비용을 감당할 수 있는지가 "
              "다음 설계 질문입니다(팀 질문 M4-5).")

    # --- [4] CSV ------------------------------------------------------------
    if not args.no_csv:
        n_list = sorted({1, 2, 5, 6, 10, 20, 25, 50, 75, 100, args.n_action_steps})
        n_list = [n for n in n_list if 1 <= n <= max(args.chunk_size, args.n_action_steps)]
        f2_list = sorted({20.0, 30.0, 50.0, 100.0, args.f2})
        path = write_csv(out_dir / "01_latency_budget.csv",
                         args.chunk_size, n_list, f2_list, args.threshold_ms)
        print(f"\n[저장] {path}  ({len(n_list) * len(f2_list)}행)")

    print("\n" + "=" * 96)
    print("  요점: 부등식의 좌변은 chunk_size가 아니라 n_action_steps다.")
    print("        기본값(100/100)의 3,333 ms는 지연이 제약이 아니라는 뜻이고,")
    print("        따라서 chunk_size=100은 지연 논거로 정해진 값이 아니다(lesson §1과 「흔한 오해」).")
    print("        다음 → 02_ensemble_weights.py")
    print("=" * 96)
    return 0 if n_pass == n_check else 1


# %%
if __name__ == "__main__":
    import sys

    raise SystemExit(main(None if "ipykernel" not in sys.modules else []))
