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
# # W2-M2 실습 2. 세 숫자로 실행 방식을 계산한다 (lesson §4.1, §4.2, §6.2)
#
# 이 스크립트는 [`../lesson.md`](../lesson.md)를 **직접 읽습니다.** 표의 숫자를 상수로 베껴두면
# 문서를 고칠 때 스크립트만 낡아 정상인 문서를 계속 FAIL로 잡습니다(course-plan §9.9의 기록).
# 그래서 여기서는 **lesson이 피검사 대상이고 산술이 기준점**입니다.
#
# ## 검산 대상 넷
#
# 1. **§4.2 실행 방식 비교표 4행.** $T_a$와 10 Hz만 주면 재계획 주기와 최악 반응 지연이 정해집니다.
#    $$T_{\text{replan}} = \tau_{\text{react}}^{\max} = \frac{T_a}{f} \tag{§4.2}$$
#    여유 또는 초과 배수까지 다시 계산해 표의 값과 대조합니다.
# 2. **설정 제약 두 개.** `n_action_steps <= horizon - n_obs_steps + 1`(§4.1 `select_action` docstring)과
#    `horizon % 2**len(down_dims) == 0`(§5.2). 라이브러리 기본값 $(2, 64, 32)$와 논문 레시피 $(2, 16, 8)$
#    양쪽에 넣습니다.
# 3. **`drop_n_last_frames` 공식.** $T_p - T_a - T_o + 1$에 두 설정을 대입해 **31과 7**을 재현하고
#    §6.2가 기록한 어긋남을 확인합니다.
# 4. **실행 구간 슬라이스.** `generate_actions`가 `actions[:, start:end]`에서 쓰는
#    $\text{start} = T_o - 1$, $\text{end} = \text{start} + T_a$가 lesson의 `actions[:, 1:33]`,
#    `actions[:, 1:9]`와 맞는지.
#
# 마지막에 두 설정의 타임라인을 **ASCII로 나란히** 그립니다. `select_action` docstring의
# o/h/a 범례를 그대로 따릅니다.
#
# 출력:
# - stdout: 대조표, 제약 검사, 타임라인 2종
# - `artifacts/W2-M2/02_receding_horizon.csv`
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

# 하드코딩한 값이 없습니다. 제어 주파수도, 세 숫자도, U-Net 다운샘플 단계 수도 전부
# lesson.md에서 읽어 옵니다. 상수로 베껴두면 문서가 바뀔 때 스크립트만 낡습니다.


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
# 표는 헤더 셀의 키워드로 찾습니다. 절 번호가 바뀌어도 표가 남아 있으면 계속 찾아냅니다.
# 파싱에 실패하면 **조용히 넘어가지 않고 FAIL로 보고**합니다. 검산기가 검사를 건너뛰면
# 검산기가 아니게 되기 때문입니다.

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
    """헤더 셀에 header_keys가 모두 들어 있는 markdown 표의 본문 행을 돌려준다."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cells = split_row(line)
        if not all(any(k in c for c in cells) for k in header_keys):
            continue
        if i + 1 >= len(lines) or set(lines[i + 1].replace("|", "").replace(" ", "")) - {"-", ":"}:
            continue  # 다음 줄이 구분선이 아니면 표 헤더가 아니다
        body: list[list[str]] = []
        for ln in lines[i + 2:]:
            if not ln.startswith("|"):
                break
            body.append(split_row(ln))
        return body
    raise LessonMissing(f"헤더 {header_keys} 를 가진 표를 찾지 못했습니다")


def num(cell: str) -> float:
    """셀에서 첫 숫자를 뽑는다. 천 단위 쉼표와 굵게 표시를 걷어낸다."""
    m = re.search(r"-?[\d,]*\.?\d+", cell.replace("**", ""))
    if not m:
        raise LessonMissing(f"숫자를 찾지 못했습니다: {cell!r}")
    return float(m.group(0).replace(",", ""))


def parse_control_hz(text: str) -> float:
    m = re.search(r"PushT는\s*([\d.]+)\s*Hz라 한 스텝이\s*([\d,]+)\s*ms", text)
    if not m:
        raise LessonMissing("§4.2의 제어 주파수 문장을 찾지 못했습니다")
    hz, step_ms = float(m.group(1)), float(m.group(2).replace(",", ""))
    if abs(1000.0 / hz - step_ms) > 1e-6:
        raise LessonMissing(f"§4.2 문장 자체가 어긋납니다: {hz} Hz 인데 한 스텝 {step_ms} ms")
    return hz


def parse_horizons(text: str) -> list[tuple[int, int, int]]:
    """§4.1의 '$T_o=2$, $T_p=64$, $T_a=32$' 두 벌을 (To, Tp, Ta)로 읽는다."""
    hits = re.findall(r"\$T_o=(\d+)\$,\s*\$T_p=(\d+)\$,\s*\$T_a=(\d+)\$", text)
    if len(hits) < 2:
        raise LessonMissing("§4.1의 세 숫자 두 벌을 찾지 못했습니다")
    return [(int(a), int(b), int(c)) for a, b, c in hits[:2]]


def parse_down_stages(text: str) -> int:
    m = re.search(r"다운샘플 단계가 (\d+)개라 (\d+)의 배수", text)
    if not m:
        raise LessonMissing("§5.2의 다운샘플 단계 서술을 찾지 못했습니다")
    stages, multiple = int(m.group(1)), int(m.group(2))
    if 2 ** stages != multiple:
        raise LessonMissing(f"§5.2 서술이 어긋납니다: {stages}단인데 {multiple}의 배수라고 적혀 있습니다")
    return stages


def parse_drop_claims(text: str) -> list[tuple[int, int, int, int]]:
    """§6.2의 'Tp - Ta - To + 1 = 값' 대입 결과를 순서대로, 중복 없이 읽는다."""
    hits = re.findall(r"\$(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*\+\s*1\s*=\s*(\d+)\$", text)
    seen, out = set(), []
    for h in hits:
        t = tuple(int(x) for x in h)
        if t not in seen:
            seen.add(t)
            out.append(t)
    if len(out) < 2:
        raise LessonMissing("§6.2의 대입 결과 두 벌을 찾지 못했습니다")
    return out


def parse_slices(text: str) -> list[tuple[int, int]]:
    """'actions[:, 1:33]' 같은 슬라이스 표기를 순서대로, 중복 없이 읽는다."""
    hits = re.findall(r"actions\[:,\s*(\d+):(\d+)\]", text)
    seen, out = set(), []
    for a, b in hits:
        t = (int(a), int(b))
        if t not in seen:
            seen.add(t)
            out.append(t)
    if len(out) < 2:
        raise LessonMissing("실행 구간 슬라이스 두 벌을 찾지 못했습니다")
    return out


def parse_default_drop_value(text: str) -> int:
    m = re.search(r"값은 여전히 (\d+)입니다", text)
    if not m:
        raise LessonMissing("§6.2의 '값은 여전히 N입니다' 서술을 찾지 못했습니다")
    return int(m.group(1))


# %% [markdown]
# ## 2. 핵심 산술. 세 줄이면 끝난다
#
# lesson §4.2가 강조하는 것은 이 식이 어렵다는 것이 아니라 **분자에 무엇이 들어가느냐**입니다.
# 예측 구간 $T_p$가 아니라 실행 구간 $T_a$입니다. 만들어놓고 버리는 뒷부분은 시간을 벌어주지 않습니다.
#
# 여기서도 재계획 주기와 최악 반응 지연이 **같은 식인데 뜻이 다릅니다.** 앞은 "다음 추론까지 몇 ms를
# 버티나"(공급), 뒤는 "새 관측이 명령에 반영되기까지 몇 ms"(신선도)입니다.

# %%
def replan_period_ms(t_a: int, hz: float) -> float:
    """재계획 주기 [ms] = T_a / f.  # eq.(§4.2)"""
    return t_a / hz * 1000.0


def worst_reaction_ms(t_a: int, hz: float) -> float:
    """최악 반응 지연 [ms]. 실행 구간 내내 새 관측을 안 본다.  # eq.(§4.2)"""
    return t_a / hz * 1000.0


def budget_verdict(infer_ms: float, replan_ms: float) -> tuple[bool, float, float]:
    """예산 안에 드는가. (들어감, 여유 ms, 초과 배수)"""
    return infer_ms <= replan_ms, replan_ms - infer_ms, infer_ms / replan_ms


def slice_bounds(t_o: int, t_a: int) -> tuple[int, int]:
    """generate_actions의 actions[:, start:end].  # eq.(§4.1)

    start = n_obs_steps - 1 이라 궤적의 맨 앞 T_o-1 칸은 과거 시점이라 버려진다.
    """
    start = t_o - 1
    return start, start + t_a


def drop_n_last_frames(t_p: int, t_a: int, t_o: int) -> int:
    """DiffusionConfig 주석의 공식.  # eq.(§6.2)"""
    return t_p - t_a - t_o + 1


def check_constraints(t_o: int, t_p: int, t_a: int, down_stages: int) -> list[tuple[str, bool, str]]:
    """설정 제약 두 개를 검사한다. (이름, 통과, 설명)"""
    lim = t_p - t_o + 1
    factor = 2 ** down_stages
    return [
        ("n_action_steps <= horizon - n_obs_steps + 1",
         t_a <= lim,
         f"{t_a} <= {t_p} - {t_o} + 1 = {lim}"),
        (f"horizon % {factor} == 0",
         t_p % factor == 0,
         f"{t_p} % {factor} = {t_p % factor}  (down_dims {down_stages}단)"),
    ]


# %% [markdown]
# ## 3. ASCII 타임라인
#
# `select_action` docstring의 범례를 그대로 씁니다.
#
# - `o` 관측이 쓰이는 칸
# - `h` 액션이 생성되는 칸
# - `a` 액션이 실제로 실행되는 칸
#
# 기준 시각 $n$이 현재 관측 시각이고, 궤적의 왼쪽 끝은 $n - T_o + 1$입니다. 2회차는 $T_a$만큼
# 오른쪽으로 밀려 시작합니다. **1회차 실행이 끝나는 칸과 2회차 실행이 시작되는 칸이 맞닿는 것**이
# 재계획 주기가 $T_a$라는 말의 그림입니다.

# %%
def timeline_rows(t_o: int, t_p: int, t_a: int, rounds: int = 2) -> list[tuple[str, str]]:
    """(라벨, 문자열) 목록. 열 하나가 한 스텝이다."""
    total = (rounds - 1) * t_a + t_p
    out: list[tuple[str, str]] = []
    for r in range(rounds):
        base = r * t_a  # r회차 궤적의 왼쪽 끝 열
        obs = ["."] * total
        gen = ["."] * total
        exe = ["."] * total
        for i in range(t_p):
            gen[base + i] = "h"
        for i in range(t_o):
            obs[base + i] = "o"
        start, end = slice_bounds(t_o, t_a)
        for i in range(start, end):
            exe[base + i] = "a"
        out.append((f"{r + 1}회차 관측 o", "".join(obs)))
        out.append((f"{r + 1}회차 생성 h", "".join(gen)))
        out.append((f"{r + 1}회차 실행 a", "".join(exe)))
    return out


def render_timeline(name: str, t_o: int, t_p: int, t_a: int, hz: float) -> str:
    rows = timeline_rows(t_o, t_p, t_a)
    total = len(rows[0][1])
    ruler = ["."] * total
    for r in range(2):
        ruler[r * t_a + t_o - 1] = "^"  # 각 회차의 기준 시각 n
    label_w = max(disp_width(lbl) for lbl, _ in rows)
    lines = [
        f"  [{name}]  T_o={t_o}, T_p={t_p}, T_a={t_a}  @ {hz:g} Hz",
        f"  {pad('', label_w)}  {'':<0}열 하나 = 한 스텝 = {1000.0 / hz:g} ms",
    ]
    for lbl, s in rows:
        lines.append(f"  {pad(lbl, label_w)}  {s}")
    lines.append(f"  {pad('기준 시각 n', label_w)}  {''.join(ruler)}")
    lines.append(
        f"  {pad('', label_w)}  재계획 주기 = T_a = {t_a}스텝 = {replan_period_ms(t_a, hz):,.0f} ms, "
        f"최악 반응 지연도 같은 값")
    start, end = slice_bounds(t_o, t_a)
    lines.append(
        f"  {pad('', label_w)}  슬라이스 actions[:, {start}:{end}], "
        f"앞의 {start}칸은 과거 시점이라 버립니다")
    return "\n".join(lines)


# %% [markdown]
# ## 4. §4.2 표 대조

# %%
def verify_exec_table(text: str, hz: float) -> tuple[list[list[str]], int, int, list[dict]]:
    body = find_table(text, ["실행 방식", "재계획 주기", "최악 반응 지연"])
    rows: list[list[str]] = []
    records: list[dict] = []
    n_pass = n_check = 0

    for cells in body:
        label = cells[0]
        t_a = int(num(cells[1]))
        lesson_replan = num(cells[2])
        infer = num(cells[3])
        sampler = re.search(r"\(([^)]+)\)", cells[3])
        verdict_cell = cells[4]
        lesson_worst = num(cells[5])

        got_replan = replan_period_ms(t_a, hz)
        got_worst = worst_reaction_ms(t_a, hz)
        fits, margin, ratio = budget_verdict(infer, got_replan)

        lesson_fits = verdict_cell.startswith("예")
        m_margin = re.search(r"여유\s*([\d,]+)\s*ms", verdict_cell)
        m_ratio = re.search(r"([\d.]+)배 초과", verdict_cell)

        checks = [
            ("재계획 주기", abs(got_replan - lesson_replan) < 0.5),
            ("최악 반응 지연", abs(got_worst - lesson_worst) < 0.5),
            ("예산 판정", fits == lesson_fits),
        ]
        if m_margin:
            checks.append(("여유", abs(margin - float(m_margin.group(1).replace(",", ""))) < 0.5))
        elif m_ratio:
            checks.append(("초과 배수", abs(ratio - float(m_ratio.group(1))) < 0.005))
        else:
            checks.append(("판정 근거 파싱", False))

        n_check += len(checks)
        n_pass += sum(1 for _, ok in checks if ok)

        rows.append([
            label,
            str(t_a),
            f"{got_replan:,.0f}",
            f"{infer:,.0f}",
            sampler.group(1) if sampler else "",
            ("여유 " + f"{margin:,.0f} ms") if fits else (f"{ratio:.2f}배 초과"),
            f"{got_worst:,.0f}",
            "PASS" if all(ok for _, ok in checks) else
            "FAIL:" + ",".join(n for n, ok in checks if not ok),
        ])
        records.append({
            "label": label, "T_a": t_a, "replan_ms": got_replan, "infer_ms": infer,
            "fits": int(fits), "margin_ms": margin, "over_ratio": ratio,
            "worst_reaction_ms": got_worst,
        })
    return rows, n_pass, n_check, records


# %% [markdown]
# ## 5. CSV 저장

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
# ## 6. 메인

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W2-M2 실습 2: receding horizon 산술 검산기 (lesson §4.1, §4.2, §6.2)")
    p.add_argument("--hz", type=float, default=None,
                   help="타임라인을 그릴 제어 주파수 [Hz]. §4.2 표 대조는 언제나 lesson의 값으로 한다")
    p.add_argument("--config", type=int, nargs=3, metavar=("T_O", "T_P", "T_A"), default=None,
                   help="임의 설정 하나를 추가로 검사한다 (예: --config 2 32 16)")
    p.add_argument("--no-csv", action="store_true", help="CSV 저장을 건너뛴다")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    n_pass = n_check = 0

    print("=" * 100)
    print(f"  {MODULE_ID} 실습 2. 세 숫자로 실행 방식을 계산한다  (표준 라이브러리만, 의존성 0)")
    print("=" * 100)

    try:
        text = read_lesson()
    except LessonMissing as exc:
        print(f"\n  [FAIL] {exc}")
        print("  이 스크립트는 lesson.md를 읽어 검산합니다. practice/ 폴더 안에서 실행하세요.")
        return 1

    # §4.2 표는 lesson이 10 Hz를 전제로 쓴 표다. --hz로 대조 기준을 바꾸면 검산이 성립하지 않으므로
    # 표 대조는 언제나 lesson의 값으로 하고, --hz는 타임라인을 다시 그리는 데만 쓴다.
    lesson_hz = parse_control_hz(text)
    hz = args.hz if args.hz else lesson_hz
    down_stages = parse_down_stages(text)
    horizons = parse_horizons(text)
    print(f"\n  lesson.md에서 읽은 값: 제어 주파수 {lesson_hz:g} Hz, "
          f"U-Net 다운샘플 {down_stages}단(horizon은 {2 ** down_stages}의 배수)")
    print(f"  설정 두 벌: 라이브러리 기본값 {horizons[0]}, 논문 레시피 {horizons[1]}  (T_o, T_p, T_a)")
    if hz != lesson_hz:
        print(f"  [알림] --hz {hz:g}는 타임라인에만 적용됩니다. §4.2 표 대조는 {lesson_hz:g} Hz로 합니다.")

    # --- [1] §4.2 실행 방식 비교표 -------------------------------------------
    print("\n=== [1] lesson §4.2 실행 방식 비교표를 다시 계산한다 ===\n")
    rows, p1, c1, records = verify_exec_table(text, lesson_hz)
    n_pass += p1
    n_check += c1
    print(render_table(
        ["실행 방식", "T_a", "재계획 주기", "추론", "샘플러", "예산 판정", "최악 반응", "대조"],
        rows, aligns="lrrrlllr"))
    print(f"\n  단위: 주기와 지연 [ms].  대조 결과: {p1}/{c1} PASS"
          f"{'  lesson §4.2와 일치' if p1 == c1 else '  불일치 발생'}")
    print("  분자에 들어가는 것은 예측 구간 T_p가 아니라 실행 구간 T_a입니다.")
    print("  만들어놓고 버리는 뒷부분은 시간을 벌어주지 않습니다.")

    # --- [2] 설정 제약 --------------------------------------------------------
    print("\n=== [2] 설정 제약 두 개 ===\n")
    names = ["라이브러리 기본값", "논문 PushT 레시피"]
    cfgs = list(horizons)
    if args.config:
        cfgs.append(tuple(args.config))
        names.append("사용자 지정")
    con_rows = []
    for name, (t_o, t_p, t_a) in zip(names, cfgs):
        for cname, ok, detail in check_constraints(t_o, t_p, t_a, down_stages):
            con_rows.append([name, f"({t_o}, {t_p}, {t_a})", cname, detail,
                             "PASS" if ok else "FAIL"])
            n_check += 1
            n_pass += int(ok)
    print(render_table(["설정", "(T_o,T_p,T_a)", "제약", "대입", "판정"], con_rows))

    # --- [3] 슬라이스 --------------------------------------------------------
    print("\n=== [3] 실행 구간 슬라이스 actions[:, start:end] ===\n")
    lesson_slices = parse_slices(text)
    sl_rows = []
    for i, (name, (t_o, t_p, t_a)) in enumerate(zip(names[:2], cfgs[:2])):
        got = slice_bounds(t_o, t_a)
        ref = lesson_slices[i] if i < len(lesson_slices) else None
        ok = ref == got
        n_check += 1
        n_pass += int(ok)
        sl_rows.append([name, f"start = T_o - 1 = {got[0]}", f"end = start + T_a = {got[1]}",
                        f"actions[:, {got[0]}:{got[1]}]",
                        f"actions[:, {ref[0]}:{ref[1]}]" if ref else "없음",
                        "PASS" if ok else "FAIL"])
    print(render_table(["설정", "start", "end", "계산값", "lesson 표기", "대조"], sl_rows))
    print(f"\n  앞의 {cfgs[0][0] - 1}칸이 버려집니다. 궤적의 왼쪽 끝은 현재가 아니라 과거 시각이기 때문입니다.")

    # --- [4] drop_n_last_frames ----------------------------------------------
    print("\n=== [4] drop_n_last_frames = T_p - T_a - T_o + 1 ===\n")
    claims = parse_drop_claims(text)
    default_value = parse_default_drop_value(text)
    dr_rows = []
    for (t_p, t_a, t_o, claimed) in claims:
        got = drop_n_last_frames(t_p, t_a, t_o)
        ok = got == claimed
        n_check += 1
        n_pass += int(ok)
        dr_rows.append([f"({t_p}, {t_a}, {t_o})",
                        f"{t_p} - {t_a} - {t_o} + 1",
                        str(got), str(claimed),
                        "일치" if got == default_value else f"어긋남 (기본값 {default_value})",
                        "PASS" if ok else "FAIL"])
    print(render_table(["(T_p, T_a, T_o)", "대입", "공식값", "lesson 값", "실제 기본값과", "대조"],
                       dr_rows, aligns="lllrll"))
    mism = [r for r in claims if drop_n_last_frames(*r[:3]) != default_value]
    print(f"\n  라이브러리 기본값은 {default_value}입니다. 공식값이 그와 다른 설정이 {len(mism)}개입니다.")
    print("  lesson §6.2가 기록한 어긋남이 이것이고, 원인은 확인되지 않았습니다.")
    print("  설치본에서 직접 확인하는 것은 labs/의 몫입니다. 여기서는 산술만 재현합니다.")

    # --- [5] 타임라인 --------------------------------------------------------
    print("\n=== [5] 타임라인. 관측 구간, 생성 구간, 실행 구간이 어떻게 겹치는가 ===\n")
    print("  범례: o 관측이 쓰이는 칸, h 액션이 생성되는 칸, a 실제로 실행되는 칸, ^ 기준 시각 n")
    print("  (select_action docstring의 범례를 그대로 따릅니다)\n")
    for name, (t_o, t_p, t_a) in zip(names, cfgs):
        print(render_timeline(name, t_o, t_p, t_a, hz))
        print()
    print("  두 그림의 차이가 이 모듈의 요점입니다. 기본값은 한 번 만든 궤적으로 3.2초를 버팁니다.")
    print("  그동안 새 관측은 명령에 반영되지 않습니다. 그것이 최악 반응 지연입니다.")

    # --- [6] CSV -------------------------------------------------------------
    if not args.no_csv:
        path = write_csv(artifacts_dir() / "02_receding_horizon.csv", records)
        print(f"\n[저장] {path}  ({len(records)}행)")

    print("\n" + "=" * 100)
    print(f"  대조 결과: {n_pass}/{n_check} PASS")
    print("  요점: 재계획 주기와 최악 반응 지연은 같은 식이고 분자는 T_a다.")
    print("        스텝마다 재계획하면 예산이 100 ms인데 기본 설정 추론이 665 ms라 성립하지 않는다.")
    print("        receding horizon은 편의 기능이 아니라 이 모델을 실시간 루프에 얹는 전제 조건이다.")
    print("        다음 → 03_nfe_budget.py")
    print("=" * 100)
    return 0 if n_pass == n_check else 1


# %%
if __name__ == "__main__":
    import sys

    raise SystemExit(main(None if "ipykernel" not in sys.modules else []))
