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
# # W2-M2 실습 3. 함수 평가 횟수를 지연 예산에 넣는다 (lesson §4.2, §5.3, §7.1)
#
# [`../lesson.md`](../lesson.md) §4.2의 지연 실측표 5행을 **직접 읽어** 두 가지를 합니다.
#
# 1. **NFE당 비용으로 나눠 봅니다.** 지연이 함수 평가 횟수에 정확히 비례한다면 ms/NFE가 일정해야
#    합니다. 일정하지 않다면 어딘가에 **횟수와 무관한 고정 비용**이 있다는 뜻입니다.
#    NFE 1과 NFE 100 두 점으로 직선을 세우고 나머지 점의 잔차를 봅니다.
#    $$\tau(\text{NFE}) \;\approx\; \tau_{\text{fixed}} \;+\; \tau_{\text{step}} \times \text{NFE} \tag{§5.3}$$
#    이 분해가 lesson §5.3의 **"느리게 만드는 것은 모델 크기가 아니라 반복 횟수"** 를 정량화합니다.
#    기울기가 반복 비용이고 절편이 관측 인코딩처럼 한 번만 드는 비용입니다.
# 2. **계층별 스텝 예산에 넣어 봅니다.** L2 전신 제어(50~500 Hz), L4 상위 지능(수 Hz),
#    PushT(10 Hz)에 다섯 설정을 각각 넣어 들어가는지 아닌지 판정하고 lesson §7.1의 서술과 대조합니다.
#    마지막으로 재계획 주기를 $T_a$만큼 늘렸을 때 각 설정이 몇 배 여유가 되는지 계산합니다.
#
# **주의 하나.** 이 스크립트는 lesson에 적힌 실측값을 재료로 산술만 합니다. 지연을 **직접 재는 것**은
# `04_dp_latency_bench.py`의 몫입니다. 남이 잰 값을 읽는 것과 자기가 재는 것은 다릅니다.
#
# 출력:
# - stdout: NFE당 비용표, 직선 적합과 잔차, 계층 판정표, 재계획 여유표
# - `artifacts/W2-M2/03_nfe_budget.csv`
#
# **의존성 0. 표준 라이브러리만 씁니다.**

# %%
from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path

MODULE_ID = "W2-M2"

# lesson §7.1이 명시하지 않은 것. "수 Hz"의 구체적 값이 없어 이 대역을 스스로 정해 넣는다.
L4_PROBE_HZ = [1.0, 2.0, 5.0, 10.0]

# 재계획 주기를 만드는 실행 구간 후보. lesson §4.2 표의 T_a 값들이다.
T_A_CANDIDATES = [1, 8, 32]


# %% [markdown]
# ## 0. 경로와 표 유틸

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


def disp_width(s: str) -> int:
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
# ## 1. lesson.md 파서
#
# 02와 같은 규약입니다. 표는 헤더 키워드로 찾고, 못 찾으면 조용히 넘어가지 않고 FAIL로 보고합니다.

# %%
class LessonMissing(Exception):
    """lesson.md에서 필요한 값을 찾지 못했다."""


def lesson_path() -> Path:
    return here().parent / "lesson.md"


def read_lesson() -> str:
    p = lesson_path()
    if not p.exists():
        raise LessonMissing(f"lesson.md를 찾지 못했습니다: {p}")
    return p.read_text(encoding="utf-8")


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def find_table(text: str, header_keys: list[str]) -> list[list[str]]:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cells = split_row(line)
        if not all(any(k in c for c in cells) for k in header_keys):
            continue
        if i + 1 >= len(lines) or set(lines[i + 1].replace("|", "").replace(" ", "")) - {"-", ":"}:
            continue
        body: list[list[str]] = []
        for ln in lines[i + 2:]:
            if not ln.startswith("|"):
                break
            body.append(split_row(ln))
        return body
    raise LessonMissing(f"헤더 {header_keys} 를 가진 표를 찾지 못했습니다")


def num(cell: str) -> float:
    m = re.search(r"-?[\d,]*\.?\d+", cell.replace("**", ""))
    if not m:
        raise LessonMissing(f"숫자를 찾지 못했습니다: {cell!r}")
    return float(m.group(0).replace(",", ""))


def parse_latency_table(text: str) -> list[dict]:
    """§4.2 지연 실측표 5행을 읽는다."""
    body = find_table(text, ["설정", "NFE", "중앙값", "최소", "최대"])
    out: list[dict] = []
    for cells in body:
        out.append({
            "label": cells[0],
            "nfe": int(num(cells[1])),
            "median_ms": num(cells[2]),
            "min_ms": num(cells[3]),
            "max_ms": num(cells[4]),
        })
    if len(out) < 3:
        raise LessonMissing("지연 실측표 행이 3개 미만입니다")
    # 넓은 표에서 긴 설정 이름 대신 쓸 번호. NFE 오름차순으로 매긴다.
    for i, r in enumerate(sorted(out, key=lambda x: (x["nfe"], x["median_ms"])), 1):
        r["idx"] = i
    return out


def parse_l2_band(text: str) -> tuple[float, float, float, float]:
    """§7.1의 '전신 제어는 50~500 Hz라 스텝 예산이 2~20 ms' 를 읽는다."""
    m = re.search(r"전신 제어는\s*(\d+)~(\d+)\s*Hz라 스텝 예산이\s*(\d+)~(\d+)\s*ms", text)
    if not m:
        raise LessonMissing("§7.1의 L2 대역 서술을 찾지 못했습니다")
    hz_lo, hz_hi, ms_lo, ms_hi = (float(x) for x in m.groups())
    return hz_lo, hz_hi, ms_lo, ms_hi


def parse_l2_quote(text: str) -> tuple[float, float]:
    """§7.1이 §4.2에서 끌어다 쓴 두 숫자. '한 번만 해도 9.4 ms이고 100회면 665 ms'."""
    m = re.search(r"한 번만 해도\s*([\d.]+)\s*ms이고\s*(\d+)회면\s*([\d,]+)\s*ms", text)
    if not m:
        raise LessonMissing("§7.1의 인용 숫자를 찾지 못했습니다")
    return float(m.group(1)), float(m.group(3).replace(",", ""))


def parse_pusht_hz(text: str) -> float:
    m = re.search(r"PushT는\s*([\d.]+)\s*Hz라", text)
    if not m:
        raise LessonMissing("§4.2의 PushT 제어 주파수를 찾지 못했습니다")
    return float(m.group(1))


# %% [markdown]
# ## 2. 고정 비용과 반복 비용으로 나누기
#
# 지연이 함수 평가 횟수에 **정확히** 비례한다면 ms/NFE가 다섯 행에서 같아야 합니다. 그렇지 않다면
# 그 차이가 곧 고정 비용의 흔적입니다. 관측 이미지를 ResNet에 한 번 통과시키는 비용, 스케줄러를
# 준비하는 비용, 커널을 띄우는 비용은 반복 횟수와 무관하게 한 번만 듭니다.
#
# 두 점으로 직선을 세웁니다. NFE가 가장 작은 행과 라이브러리 기본값 행입니다. 두 행은 같은 구조에서
# 잰 값이라 구조 차이가 섞이지 않습니다. 나머지 행은 그 직선에 대한 **잔차**로 읽습니다.

# %%
def fit_affine(p_lo: tuple[float, float], p_hi: tuple[float, float]) -> tuple[float, float]:
    """두 점 (NFE, ms)를 지나는 직선. (고정 비용 ms, NFE당 비용 ms)  # eq.(§5.3)"""
    (n1, t1), (n2, t2) = p_lo, p_hi
    if n1 == n2:
        raise ValueError("두 점의 NFE가 같습니다")
    slope = (t2 - t1) / (n2 - n1)
    intercept = t1 - slope * n1
    return intercept, slope


def predict_ms(intercept: float, slope: float, nfe: int) -> float:
    return intercept + slope * nfe


def max_nfe_within(budget_ms: float, intercept: float, slope: float) -> int:
    """예산 안에 들어가는 최대 함수 평가 횟수. 고정 비용도 못 내면 0."""
    if budget_ms <= intercept:
        return 0
    return int((budget_ms - intercept) // slope)


# %% [markdown]
# ## 3. 계층별 판정
#
# lesson §7.1이 자리를 가르는 근거가 이 산수입니다. 스텝 예산은 주파수의 역수이고, 그 안에
# 추론이 들어가지 않으면 그 계층 안에는 못 들어갑니다.

# %%
def step_budget_ms(hz: float) -> float:
    """제어 주파수 하나가 주는 스텝 예산 [ms]."""
    return 1000.0 / hz


def fits(latency_ms: float, budget_ms: float) -> bool:
    return latency_ms <= budget_ms


def ascii_latency_plot(rows: list[dict], intercept: float, slope: float, width: int = 48) -> str:
    """NFE 대 지연을 막대로. 옆에 적합값과 잔차를 함께 적는다."""
    peak = max(r["median_ms"] for r in rows) or 1.0
    lines = []
    for r in sorted(rows, key=lambda x: (x["nfe"], x["median_ms"])):
        bar = "#" * max(1, int(round(r["median_ms"] / peak * width)))
        pred = predict_ms(intercept, slope, r["nfe"])
        lines.append(
            f"  [{r['idx']}] NFE {r['nfe']:>3d} |{bar:<{width}} {r['median_ms']:>7.1f} ms"
            f"   적합 {pred:>7.1f}   잔차 {r['median_ms'] - pred:>+7.1f}")
    return "\n".join(lines)


# %% [markdown]
# ## 4. CSV 저장

# %%
def write_csv(path: Path, records: list[dict]) -> Path:
    cols = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    return path


# %% [markdown]
# ## 5. 메인

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W2-M2 실습 3: NFE 예산 검산기 (lesson §4.2, §5.3, §7.1)")
    p.add_argument("--extra-hz", type=float, nargs="+", default=None,
                   help="추가로 판정해볼 제어 주파수 목록 [Hz]")
    p.add_argument("--no-csv", action="store_true", help="CSV 저장을 건너뛴다")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    n_pass = n_check = 0
    notes: list[list[str]] = []

    print("=" * 104)
    print(f"  {MODULE_ID} 실습 3. 함수 평가 횟수를 지연 예산에 넣는다  (표준 라이브러리만, 의존성 0)")
    print("=" * 104)

    try:
        text = read_lesson()
        rows = parse_latency_table(text)
        hz_lo, hz_hi, ms_lo, ms_hi = parse_l2_band(text)
        quote_1, quote_100 = parse_l2_quote(text)
        pusht_hz = parse_pusht_hz(text)
    except LessonMissing as exc:
        print(f"\n  [FAIL] {exc}")
        print("  이 스크립트는 lesson.md를 읽어 검산합니다. practice/ 폴더 안에서 실행하세요.")
        return 1

    print(f"\n  lesson §4.2에서 읽은 지연 실측 {len(rows)}행, PushT {pusht_hz:g} Hz")
    print(f"  lesson §7.1에서 읽은 L2 대역 {hz_lo:g}~{hz_hi:g} Hz, 스텝 예산 {ms_lo:g}~{ms_hi:g} ms")

    # lesson 내부 정합성. §7.1이 인용한 대역이 주파수의 역수와 맞는가.
    ok_band = (abs(step_budget_ms(hz_hi) - ms_lo) < 0.05 and abs(step_budget_ms(hz_lo) - ms_hi) < 0.05)
    n_check += 1
    n_pass += int(ok_band)
    print(f"  대역 정합성: 1000/{hz_hi:g} = {step_budget_ms(hz_hi):g} ms, "
          f"1000/{hz_lo:g} = {step_budget_ms(hz_lo):g} ms   [{'PASS' if ok_band else 'FAIL'}]")

    # --- [1] NFE당 비용 -------------------------------------------------------
    print("\n=== [1] NFE당 비용. 지연은 함수 평가 횟수에 비례하는가 ===\n")
    per_nfe = [(r, r["median_ms"] / r["nfe"])
               for r in sorted(rows, key=lambda x: (x["nfe"], x["median_ms"]))]
    print(render_table(
        ["#", "설정", "NFE", "중앙값 [ms]", "최소", "최대", "ms/NFE"],
        [[str(r["idx"]), r["label"], str(r["nfe"]), f"{r['median_ms']:.1f}", f"{r['min_ms']:.1f}",
          f"{r['max_ms']:.1f}", f"{v:.3f}"] for r, v in per_nfe],
        aligns="rlrrrrr"))
    print("\n  아래 표들은 폭을 줄이려고 설정 이름 대신 이 # 번호를 씁니다.")
    lo = min(v for _, v in per_nfe)
    hi = max(v for _, v in per_nfe)
    print(f"\n  ms/NFE 범위 {lo:.3f} ~ {hi:.3f}, 최대가 최소의 {hi / lo:.2f}배입니다.")
    print("  일정하지 않습니다. 곧 지연이 NFE에 정비례하지 않고 어딘가 고정 비용이 섞여 있습니다.")
    print("  가장 큰 ms/NFE가 NFE가 가장 작은 행에서 나오는 것이 그 증거입니다.")
    ok_shape = max(per_nfe, key=lambda x: x[1])[0]["nfe"] == min(r["nfe"] for r in rows)
    n_check += 1
    n_pass += int(ok_shape)
    print(f"  판정: 최대 ms/NFE가 최소 NFE 행에서 나왔는가   [{'PASS' if ok_shape else 'FAIL'}]")

    # --- [2] 고정 비용과 반복 비용 -------------------------------------------
    print("\n=== [2] 두 점으로 직선을 세우고 나머지를 잔차로 읽는다 ===\n")
    row_lo = min(rows, key=lambda r: r["nfe"])
    anchors_hi = [r for r in rows if "기본값" in r["label"]]
    row_hi = anchors_hi[0] if anchors_hi else max(rows, key=lambda r: r["nfe"])
    intercept, slope = fit_affine((row_lo["nfe"], row_lo["median_ms"]),
                                  (row_hi["nfe"], row_hi["median_ms"]))
    print(f"  기준점 두 개: NFE {row_lo['nfe']} ({row_lo['label']}) 와 "
          f"NFE {row_hi['nfe']} ({row_hi['label']})")
    print(f"  적합 결과: tau(NFE) = {intercept:.2f} ms + {slope:.3f} ms x NFE\n")
    print(ascii_latency_plot(rows, intercept, slope))

    resid_rows = []
    for r in sorted(rows, key=lambda x: (x["nfe"], x["median_ms"])):
        pred = predict_ms(intercept, slope, r["nfe"])
        rel = (r["median_ms"] - pred) / pred * 100.0
        is_anchor = r is row_lo or r is row_hi
        resid_rows.append([str(r["idx"]), r["label"], str(r["nfe"]), f"{r['median_ms']:.1f}",
                           f"{pred:.1f}", f"{r['median_ms'] - pred:+.1f}", f"{rel:+.1f}%",
                           "기준점" if is_anchor else ""])
    print()
    print(render_table(["#", "설정", "NFE", "실측", "적합", "잔차 [ms]", "상대 잔차", ""],
                       resid_rows, aligns="rlrrrrrl"))

    free_rows = [r for r in rows if r is not row_lo and r is not row_hi]
    same_arch = [r for r in free_rows if "논문 PushT 설정" not in r["label"]]
    worst_rel = max(abs(r["median_ms"] - predict_ms(intercept, slope, r["nfe"]))
                    / predict_ms(intercept, slope, r["nfe"]) * 100.0
                    for r in same_arch) if same_arch else 0.0
    ok_fit = worst_rel < 5.0
    n_check += 1
    n_pass += int(ok_fit)
    print(f"\n  기준점을 뺀 같은 구조 행들의 최대 상대 잔차 {worst_rel:.1f}%"
          f"   [{'PASS' if ok_fit else 'FAIL'}] 5% 미만을 직선으로 본다")
    print("  구조가 다른 행(논문 PushT 설정)은 기준선에서 벗어나는 것이 정상입니다.")
    print("  이미지 자르기, 그룹 정규화, 카메라별 인코더 설정이 전부 달라 같은 직선 위에 있을 이유가 없습니다.")

    print(f"\n  고정 비용 {intercept:.2f} ms, 반복 1회 비용 {slope:.3f} ms")
    share_rows = []
    for nfe in sorted({r["nfe"] for r in rows}):
        pred = predict_ms(intercept, slope, nfe)
        share_rows.append([str(nfe), f"{intercept:.2f}", f"{slope * nfe:.1f}",
                           f"{intercept / pred * 100:.1f}%", f"{slope * nfe / pred * 100:.1f}%"])
    print()
    print(render_table(["NFE", "고정 [ms]", "반복 [ms]", "고정 비중", "반복 비중"],
                       share_rows, aligns="rrrrr"))
    top = max(rows, key=lambda r: r["nfe"])
    top_share = slope * top["nfe"] / predict_ms(intercept, slope, top["nfe"]) * 100
    ok_dominant = top_share > 95.0
    n_check += 1
    n_pass += int(ok_dominant)
    print(f"\n  NFE {top['nfe']}에서 반복 비중이 {top_share:.1f}%   "
          f"[{'PASS' if ok_dominant else 'FAIL'}] 95% 초과면 반복이 지배한다고 본다")
    print("  lesson §5.3: 이 정책을 느리게 만드는 것은 모델 크기가 아니라 반복 횟수입니다.")
    print("  파라미터가 앞 모듈의 5.1배인 것은 반복 1회 비용에만 들어갑니다. 그 1회 비용을 100번 무는 것이")
    print("  진짜 원인이고, 고정 비용은 전체의 1%도 안 됩니다.")

    # --- [3] 계층 판정 --------------------------------------------------------
    print("\n=== [3] 계층별 스텝 예산에 넣어 본다 ===\n")
    l2_probes: list[tuple[str, float]] = [
        (f"{hz_lo:g} Hz", hz_lo), ("100 Hz", 100.0), (f"{hz_hi:g} Hz", hz_hi),
    ]
    l4_probes: list[tuple[str, float]] = [(f"{h:g} Hz", h) for h in L4_PROBE_HZ]
    if args.extra_hz:
        l4_probes += [(f"{h:g} Hz", h) for h in args.extra_hz]
    probes = [(f"L2 {n}", h) for n, h in l2_probes] + [(f"L4 {n}", h) for n, h in l4_probes]

    records: list[dict] = []
    by_idx = {r["idx"]: r for r in rows}
    for i in sorted(by_idx):
        r = by_idx[i]
        records.append({"idx": i, "label": r["label"], "nfe": r["nfe"],
                        "median_ms": r["median_ms"]})

    def grid(title: str, subset: list[tuple[str, float]]) -> str:
        header = ["#", "NFE", "지연 [ms]"] + [f"{n} / {step_budget_ms(h):g}ms" for n, h in subset]
        body = []
        for rec in records:
            cells = [str(rec["idx"]), str(rec["nfe"]), f"{rec['median_ms']:.1f}"]
            for _, h in subset:
                ok = fits(rec["median_ms"], step_budget_ms(h))
                cells.append("들어감" if ok else "초과")
                rec[f"fits_{h:g}hz"] = int(ok)
            body.append(cells)
        return title + "\n" + render_table(header, body, aligns="rrr" + "l" * len(subset))

    print(grid(f"  L2 전신 제어 {hz_lo:g}~{hz_hi:g} Hz\n", l2_probes))
    print()
    print(grid("  L4 상위 지능. lesson은 '수 Hz'라고만 적었으므로 대역은 이 스크립트가 정한 것입니다\n",
               l4_probes))

    # lesson §7.1의 세 주장 대조
    print("\n  lesson §7.1의 주장과 대조합니다.\n")
    policy_rows = [r for r in rows if r["nfe"] == max(x["nfe"] for x in rows)]
    l2_budgets = [step_budget_ms(h) for _, h in probes if "L2" in _]
    claim_l2 = all(not fits(r["median_ms"], b) for r in policy_rows for b in l2_budgets)
    n_check += 1
    n_pass += int(claim_l2)
    print(f"  ① 'L2 안에는 들어갈 수 없습니다' (NFE {policy_rows[0]['nfe']} 기준)"
          f"   [{'PASS' if claim_l2 else 'FAIL'}]")

    row1 = min(rows, key=lambda r: r["nfe"])
    row100 = row_hi
    ok_quote = (abs(row1["median_ms"] - quote_1) < 0.05
                and abs(round(row100["median_ms"]) - quote_100) < 1.0)
    n_check += 1
    n_pass += int(ok_quote)
    print(f"  ② §7.1이 인용한 {quote_1:g} ms와 {quote_100:g} ms가 §4.2 표와 같은가"
          f"   (표: {row1['median_ms']:g}, {row100['median_ms']:g})"
          f"   [{'PASS' if ok_quote else 'FAIL'}]")

    l4_ok = [name for name, hz in probes if "L4" in name
             and fits(row100["median_ms"], step_budget_ms(hz))]
    claim_l4 = len(l4_ok) > 0
    n_check += 1
    n_pass += int(claim_l4)
    print(f"  ③ 'L4 자리는 가능합니다'   [{'PASS' if claim_l4 else 'FAIL'}]"
          f"   {row100['median_ms']:g} ms가 들어가는 L4 주파수: {', '.join(l4_ok) or '없음'}")

    # 결이 하나 있다. NFE 1은 L2 하단에는 들어간다.
    if fits(row1["median_ms"], step_budget_ms(hz_lo)):
        notes.append([
            f"NFE {row1['nfe']}의 {row1['median_ms']:g} ms는 L2 하단 {hz_lo:g} Hz의 "
            f"{step_budget_ms(hz_lo):g} ms 예산에는 들어갑니다"
            f"(예산의 {row1['median_ms'] / step_budget_ms(hz_lo) * 100:.0f}%).",
            f"§7.1의 'L2 불가'는 실제로 쓰는 설정인 NFE {row100['nfe']}에 대한 판정으로 읽어야 맞고,",
            "NFE 1 단독은 대역의 느린 끝에서만 성립합니다. 100 Hz 위로는 그마저 초과합니다.",
            "다만 NFE 1은 지연만 해결하지 품질까지 보존하는 설정이 아닙니다(lesson §5.3 단서).",
        ])

    print("\n  예산 안에 들어가는 최대 함수 평가 횟수를 적합 직선으로 역산하면 이렇습니다.\n")
    inv_rows = []
    for name, hz in probes + [(f"PushT {pusht_hz:g} Hz", pusht_hz)]:
        b = step_budget_ms(hz)
        inv_rows.append([name, f"{b:g}", str(max_nfe_within(b, intercept, slope))])
    print(render_table(["계층", "스텝 예산 [ms]", "최대 NFE"], inv_rows, aligns="lrr"))
    print("\n  고정 비용조차 못 내는 자리에서는 0이 나옵니다. 스텝 수를 줄이는 것으로는 닿지 못한다는 뜻이고,")
    print("  그때는 구조를 바꾸거나 계층을 바꿔야 합니다.")

    # --- [4] 재계획 주기로 사는 여유 -----------------------------------------
    print(f"\n=== [4] 재계획 주기를 T_a로 늘렸을 때의 여유 (@ {pusht_hz:g} Hz) ===\n")
    head = ["#", "설정", "NFE", "지연 [ms]"] + [f"T_a={t} ({t / pusht_hz * 1000:,.0f}ms)"
                                                for t in T_A_CANDIDATES]
    hr_rows = []
    for rec in records:
        cells = [str(rec["idx"]), rec["label"], str(rec["nfe"]), f"{rec['median_ms']:.1f}"]
        for t in T_A_CANDIDATES:
            b = t / pusht_hz * 1000.0
            ratio = b / rec["median_ms"]
            cells.append(f"{ratio:.2f}배" if ratio >= 1.0 else f"{ratio:.2f}배 (초과)")
            rec[f"headroom_ta{t}"] = ratio
        hr_rows.append(cells)
    print(render_table(head, hr_rows, aligns="rlrr" + "r" * len(T_A_CANDIDATES)))
    print("\n  '몇 배'는 재계획 주기를 추론 지연으로 나눈 값입니다. 1보다 작으면 명령이 끊깁니다.")
    print("  lesson §4.2가 말한 대로 실행 구간이 곧 지연을 사는 돈입니다. 대가는 최악 반응 지연이고")
    print("  그 값이 재계획 주기와 같은 숫자입니다.")

    # --- [5] 메모와 CSV ------------------------------------------------------
    if notes:
        print("\n=== [5] 판정하지 않고 남기는 결 ===\n")
        for i, nt in enumerate(notes, 1):
            for j, line in enumerate(nt):
                print(f"  ({i}) {line}" if j == 0 else f"      {line}")

    if not args.no_csv:
        path = write_csv(artifacts_dir() / "03_nfe_budget.csv", records)
        print(f"\n[저장] {path}  ({len(records)}행)")

    print("\n" + "=" * 104)
    print(f"  대조 결과: {n_pass}/{n_check} PASS")
    print(f"  요점: 지연 = 고정 {intercept:.1f} ms + 반복 {slope:.2f} ms x NFE 로 갈린다.")
    print("        고정 비용은 미미하고 반복이 지배하므로, 줄일 것은 모델이 아니라 스텝 수다.")
    print("        스텝 수를 줄일 수 없으면 실행 구간을 늘려 재계획 주기를 사야 하고 반응 지연을 지불한다.")
    print("        직접 재보는 것은 → 04_dp_latency_bench.py")
    print("=" * 104)
    return 0 if n_pass == n_check else 1


# %%
if __name__ == "__main__":
    import sys

    raise SystemExit(main(None if "ipykernel" not in sys.modules else []))
