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
# # W1-M1 실습 3 — 플레이어 지도 데이터화
#
# lesson.md `§6.2`(주요 플레이어 계보 표)를 `players.csv`로 옮기고, pandas로 읽어
# **① 조직 × 베팅 계층 매트릭스**와 **② 최신 모델 발표 타임라인**을 렌더합니다.
#
# > **이 실습의 목적은 "학습자가 직접 고치는 것"입니다.**
# > 이 분야는 몇 주 단위로 새 모델이 나옵니다. 새 발표가 나오면
# > `players.csv`에 **한 줄만 추가**하고 이 스크립트를 다시 돌리면 그림이 갱신됩니다.
# > 코드를 고칠 일은 없습니다.
#
# ## CSV 스키마
#
# | 컬럼 | 의미 |
# |---|---|
# | `org` | 조직명 |
# | `lineage` | 계보 (화살표는 ASCII `->` 로 씀 — CSV 인코딩 사고 방지) |
# | `latest` | 2026-08 기준 최신 모델. 없으면 빈칸 |
# | `date` | 발표 시점. `YYYY` / `YYYY-MM` / `YYYY-MM-DD` 중 **아는 만큼만**. 모르면 빈칸 |
# | `date_confidence` | `confirmed`(1차 출처 확인) / `estimated`(시점 추정) / `unknown`(미확인) |
# | `bet_layer` | 베팅 영역. `L1`~`L5` + 비계층 영역 `SIM`(시뮬 인프라) `DATA`(데이터·월드모델). `;`로 구분 |
# | `note` | lesson §6.2 표의 '특기' 열 |
#
# **날짜 정밀도를 그림에 그대로 반영합니다** — `2026`만 아는 항목은 2026년 전체를 덮는 가로 막대로,
# `2026-07-31`처럼 확정된 항목은 점으로 찍습니다. 없는 정밀도를 지어내지 않기 위한 장치입니다.
#
# > lesson §6.2이 "1차 출처 미확인"으로 배제한 정량 수치(Figure 생산 대수, Optimus DoF, 1X 선주문 수량)는
# > **CSV에 넣지 않았습니다.** 세 조직은 정성 서술만 있는 행으로 남아 있어 타임라인에는 나오지 않습니다.
#
# > **그림 라벨로 쓰이는 `org` · `lineage` · `latest`는 가급적 ASCII로 쓰세요.**
# > 한글 폰트가 없는 환경에서 영문 폴백을 해도 이 세 컬럼은 CSV 값이 그대로 나가기 때문에
# > 한글이 들어 있으면 그 부분만 두부(□)로 깨집니다. 한글 설명은 `note`에 넣으면 됩니다.
# > (깨지면 `save_and_check()`가 "렌더에 빠진 글리프" 경고를 찍어 알려줍니다.)
#
# GPU 불필요. 전체 실행 수 초.

# %%
from __future__ import annotations

import argparse
import sys
import unicodedata
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless 고정 — plt.show() 금지

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402

MODULE_ID = "W1-M1"
CSV_NAME = "players.csv"


# %% [markdown]
# ## 0. 경로·폰트 유틸 (01/02번 스크립트와 동일)

# %%
_ROOT_MARKERS = ("course", "docs", "CLAUDE.md")


def script_dir() -> Path:
    """이 파일이 있는 practice/ 디렉토리 (노트북에서는 cwd)."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def find_repo_root() -> Path:
    """리포 루트. 스크립트로 실행하면 practice/의 parents[4]가 루트다."""
    start = script_dir()
    for cand in (start, *start.parents):
        if all((cand / m).exists() for m in _ROOT_MARKERS):
            return cand
    return start


def artifacts_dir() -> Path:
    out = find_repo_root() / "artifacts" / MODULE_ID
    out.mkdir(parents=True, exist_ok=True)
    return out


def csv_path() -> Path:
    """players.csv 위치. 노트북에서 cwd가 다를 수 있으므로 몇 군데를 훑는다."""
    here = script_dir()
    candidates = [
        here / CSV_NAME,
        find_repo_root() / "course/w1-generative-core/01-physical-ai-landscape/practice" / CSV_NAME,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"{CSV_NAME}을 찾지 못했습니다. 확인한 경로: {[str(c) for c in candidates]}")


# %%
_KO_FONT_PREFERENCE = (
    "Pretendard", "NanumGothic", "Nanum Gothic", "Malgun Gothic", "NanumBarunGothic",
    "Noto Sans KR", "Noto Sans CJK KR", "Noto Sans CJK JP", "Source Han Sans KR",
    "AppleGothic", "Spoqa Han Sans Neo", "UnDotum", "Baekmuk Gulim",
)
_PROBE_CHARS = "한글계층베팅"

USE_KOREAN = False


def _has_hangul(font_path: str) -> bool:
    try:
        face = FT2Font(font_path)
        return all(face.get_char_index(ord(c)) != 0 for c in _PROBE_CHARS)
    except Exception:
        return False


def setup_korean_font(force_ascii: bool = False) -> bool:
    """한글 폰트를 찾아 기본 폰트로 설정. 실패하면 영문 라벨로 폴백."""
    global USE_KOREAN
    plt.rcParams["axes.unicode_minus"] = False
    if force_ascii:
        USE_KOREAN = False
        print("[font] --ascii-labels 지정 → 영문 라벨로 렌더합니다.")
        return False

    installed = {f.name for f in fm.fontManager.ttflist}
    for name in _KO_FONT_PREFERENCE:
        if name not in installed:
            continue
        path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
        if _has_hangul(path):
            plt.rcParams["font.family"] = name
            USE_KOREAN = True
            print(f"[font] 한글 폰트 사용: {name}")
            return True

    USE_KOREAN = False
    print(
        "[font] 경고: 한글 글리프를 가진 폰트를 찾지 못해 그림 라벨을 영문으로 폴백합니다."
        " (해결: apt install fonts-nanum 후 ~/.cache/matplotlib 삭제)"
    )
    return False


def t(ko: str, en: str) -> str:
    return ko if USE_KOREAN else en


def save_and_check(fig, out_path: Path, dpi: int = 140, rect: tuple | None = None) -> Path:
    """그림을 저장하면서 '글리프 없음(두부현상)' 경고를 눈에 띄게 잡아낸다."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.tight_layout(rect=rect) if rect else fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
    missing = {str(w.message) for w in caught if "missing from font" in str(w.message)}
    for msg in sorted(missing):
        print(f"[font] 경고: 렌더에 빠진 글리프가 있습니다 → {msg}")
    plt.close(fig)
    return out_path


# %% [markdown]
# ## 1. CSV 로드와 파싱
#
# 베팅 영역 축은 **L1~L5(스택 계층) + SIM·DATA(비계층 영역)** 입니다.
# lesson §6.2의 "주 베팅 계층" 열에 '시뮬 인프라', '데이터·월드모델'처럼 계층이 아닌 항목이 있어
# 정보를 버리지 않으려고 두 칸을 더 뒀습니다.

# %%
# 아래(하드웨어) → 위(인지) 순. 그림에서 L1이 아래로 가도록 이 순서를 쓴다.
LAYER_AXIS = ["L1", "L2", "L3", "L4", "L5", "SIM", "DATA"]
LAYER_LABEL_KO = {
    "L1": "L1 하드웨어", "L2": "L2 전신 제어", "L3": "L3 액션 인터페이스",
    "L4": "L4 상위 지능", "L5": "L5 인지·매핑",
    "SIM": "SIM 시뮬 인프라", "DATA": "DATA 데이터·월드모델",
}
LAYER_LABEL_EN = {
    "L1": "L1 Hardware", "L2": "L2 WBC", "L3": "L3 Action interface",
    "L4": "L4 High-level", "L5": "L5 Perception",
    "SIM": "SIM Simulation infra", "DATA": "DATA Data / World model",
}


def load_players(path: Path | None = None) -> pd.DataFrame:
    """players.csv를 읽어 파생 컬럼을 붙인다."""
    path = path or csv_path()
    df = pd.read_csv(path, dtype=str).fillna("")
    df["bets"] = df["bet_layer"].apply(
        lambda s: [x.strip() for x in s.split(";") if x.strip()]
    )
    df[["date_start", "date_end", "date_prec"]] = df["date"].apply(
        lambda s: pd.Series(parse_partial_date(s))
    )
    df["disp"] = display_names(df)
    return df


def display_names(df: pd.DataFrame) -> list[str]:
    """축 라벨용 이름. 같은 조직이 여러 줄이면 계보 앞머리를 붙여 구분한다.

    (NVIDIA는 GR00T / Cosmos / Isaac 세 줄이라 그냥 'NVIDIA'로는 구분이 안 된다.)
    """
    dup = df["org"].duplicated(keep=False)
    out = []
    for org, lineage, is_dup in zip(df["org"], df["lineage"], dup):
        if is_dup:
            head = lineage.split("->")[0].strip().split("/")[0].strip()
            out.append(f"{org} · {head}")
        else:
            out.append(org)
    return out


def parse_partial_date(s: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str]:
    """'2026' / '2026-04' / '2026-04-16' / '' 를 (구간 시작, 구간 끝, 정밀도)로.

    없는 정밀도를 만들어내지 않기 위해, 아는 만큼만 구간으로 표현한다.
    """
    s = (s or "").strip()
    if not s:
        return None, None, "none"
    parts = s.split("-")
    if len(parts) == 1:  # 연도만
        start = pd.Timestamp(int(parts[0]), 1, 1)
        return start, start + pd.DateOffset(years=1), "year"
    if len(parts) == 2:  # 연-월
        start = pd.Timestamp(int(parts[0]), int(parts[1]), 1)
        return start, start + pd.DateOffset(months=1), "month"
    start = pd.Timestamp(int(parts[0]), int(parts[1]), int(parts[2]))
    return start, start + pd.Timedelta(days=1), "day"


# %%
def print_players_table(df: pd.DataFrame) -> None:
    """정렬된 플레이어 표를 stdout에 출력."""
    print("\n=== [1] 플레이어 지도 (lesson §6.2) ===")
    view = df.copy()
    view["date_sort"] = view["date_start"].fillna(pd.Timestamp("2100-01-01"))
    view = view.sort_values(["date_sort", "org"], kind="stable")
    cols = ["org", "lineage", "latest", "date", "date_confidence", "bet_layer"]
    show = view[cols].rename(columns={
        "org": "조직", "lineage": "계보", "latest": "최신",
        "date": "시점", "date_confidence": "확실성", "bet_layer": "베팅 영역",
    })
    # 한글은 표시 폭이 2칸이라 east_asian_width를 켜야 열이 맞는다
    with pd.option_context("display.max_colwidth", 46, "display.width", 220,
                           "display.unicode.east_asian_width", True):
        print(show.to_string(index=False))
    print(
        f"\n  총 {len(df)}개 조직 · 시점이 확인된 항목 {int(df['date_start'].notna().sum())}개"
        f" (confirmed {int((df['date_confidence'] == 'confirmed').sum())} /"
        f" estimated {int((df['date_confidence'] == 'estimated').sum())} /"
        f" unknown {int((df['date_confidence'] == 'unknown').sum())})"
    )


def print_layer_summary(df: pd.DataFrame) -> None:
    """베팅 영역별로 어느 조직이 몰려 있는지."""
    print("\n=== [2] 베팅 영역별 조직 ===")
    for layer in LAYER_AXIS:
        orgs = df.loc[df["bets"].apply(lambda b, l=layer: l in b), "org"].tolist()
        label = LAYER_LABEL_KO[layer]
        pad = " " * max(0, 24 - sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in label))
        print(f"  {label}{pad}({len(orgs)}) : {', '.join(orgs) if orgs else '-'}")
    print("  * L3(액션 인터페이스)에 공개적으로 베팅한 조직이 비어 있다는 것 자체가 lesson §3.4의")
    print("    '승부처는 L3' 주장과 맞물린다. 공개 논문 수준에서 L3를 독립 계층으로 내세운 곳이 드물다.")


# %% [markdown]
# ## 2. 그림 ① 조직 × 베팅 영역 매트릭스

# %%
def plot_bet_matrix(df: pd.DataFrame, ax) -> None:
    """조직(행) × 베팅 영역(열) 히트맵.

    조직명을 y축에 두면 라벨을 회전시키지 않아도 되고, (b) 타임라인과 축이 맞는다.
    """
    orgs = df["disp"].tolist()
    mat = np.zeros((len(orgs), len(LAYER_AXIS)))
    for i, bets in enumerate(df["bets"]):
        for b in bets:
            if b in LAYER_AXIS:
                mat[i, LAYER_AXIS.index(b)] = 1.0

    ax.imshow(mat, cmap="Blues", vmin=0, vmax=1.6, aspect="auto", origin="upper")
    for i in range(len(orgs)):
        for j in range(len(LAYER_AXIS)):
            if mat[i, j] > 0:
                ax.text(j, i, "O", ha="center", va="center", fontsize=11,
                        fontweight="bold", color="#123a5c")

    labels = LAYER_LABEL_KO if USE_KOREAN else LAYER_LABEL_EN
    ax.set_xticks(range(len(LAYER_AXIS)))
    ax.set_xticklabels([labels[k] for k in LAYER_AXIS], fontsize=9.5)
    ax.set_yticks(range(len(orgs)))
    ax.set_yticklabels(orgs, fontsize=9.5)
    ax.set_xticks(np.arange(-0.5, len(LAYER_AXIS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(orgs), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.6)
    ax.tick_params(which="minor", length=0)
    # L1~L5(스택 계층)와 SIM/DATA(비계층 영역)의 경계
    ax.axvline(4.5, color="#c0392b", lw=1.4, ls="--")
    ax.set_title(
        t("(a) 누가 어느 계층에 베팅했는가   (점선 오른쪽 = 스택 계층이 아닌 영역)",
          "(a) Who bets on which layer   (right of dashed line = non-stack areas)"),
        fontsize=11,
    )


# %% [markdown]
# ## 3. 그림 ② 최신 모델 발표 타임라인
#
# 날짜 정밀도를 그대로 표현합니다.
#
# | 표기 | 뜻 |
# |---|---|
# | 채운 점 + 짧은 막대 | `confirmed` — 1차 출처로 확인된 날짜 |
# | 빈 점 + 구간 막대 | `estimated` — 연/월까지만 아는 항목. 막대가 그 구간 전체 |
# | (표시 없음) | `unknown` — 시점 미확인. 타임라인에서 제외하고 그림 아래 목록에만 남긴다 |

# %%
def plot_timeline(df: pd.DataFrame, ax) -> None:
    """최신 모델 발표 시점 타임라인 (날짜 정밀도를 막대 길이로 표현)."""
    dated = df[df["date_start"].notna()].copy()
    dated = dated.sort_values("date_start", kind="stable").reset_index(drop=True)

    for y, row in dated.iterrows():
        confirmed = row["date_confidence"] == "confirmed"
        color = "#1f4e79" if confirmed else "#8aa4bf"
        start, end = row["date_start"], row["date_end"]
        ax.plot([start, end], [y, y], lw=6, color=color, alpha=0.55, solid_capstyle="butt")
        ax.plot(
            [start], [y], marker="o", ms=9, color=color,
            markerfacecolor=color if confirmed else "white", markeredgewidth=1.8,
        )
        label = row["latest"] or row["lineage"]
        ax.text(end + pd.Timedelta(days=18), y, f"{label}  ({row['date']})",
                va="center", ha="left", fontsize=9, color="#1f2d3d")

    ax.set_yticks(range(len(dated)))
    ax.set_yticklabels(dated["disp"], fontsize=9.5)
    ax.set_ylim(-0.8, len(dated) - 0.2)
    ax.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2027-10-01"))
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel(t("발표 시점", "announcement date"), fontsize=10)
    ax.set_title(
        t("(b) 최신 모델 발표 타임라인   (채운 점 = 확인된 날짜, 빈 점 = 구간만 아는 추정)",
          "(b) Latest-model timeline   (filled = confirmed date, hollow = period estimate)"),
        fontsize=11,
    )

    undated = df[df["date_start"].isna()]["disp"].tolist()
    if undated:
        ax.text(
            0.01, -0.20,
            t(f"시점 미확인(1차 출처 없음): {', '.join(undated)}",
              f"undated (no primary source): {', '.join(undated)}"),
            transform=ax.transAxes, fontsize=9, color="#8a6d3b",
        )


# %%
def plot_landscape(df: pd.DataFrame, out_path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.5, 10.0), gridspec_kw={"height_ratios": [1.05, 0.95]}
    )
    plot_bet_matrix(df, ax1)
    plot_timeline(df, ax2)
    fig.suptitle(
        t("W1-M1 · 플레이어 지형 (lesson §6.2, players.csv에서 생성)",
          "W1-M1 · Player landscape (lesson §6.2, generated from players.csv)"),
        fontsize=14,
    )
    return save_and_check(fig, out_path, rect=(0, 0.02, 1, 0.97))


# %% [markdown]
# ## 4. 실행

# %%
def _in_notebook() -> bool:
    """노트북/커널 안에서 도는가.

    커널의 sys.argv(`-f .../kernel-xxxx.json`)를 argparse에 그대로 넘기면 SystemExit가 난다.
    IPython 유무와 argv[0] 두 가지로 판정한다.
    """
    try:
        from IPython import get_ipython  # type: ignore
        if get_ipython() is not None:
            return True
    except Exception:
        pass
    argv0 = Path(sys.argv[0]).name.lower() if sys.argv else ""
    return argv0 == "" or "ipykernel" in argv0 or "jupyter" in argv0 or "colab" in argv0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W1-M1: 플레이어 지도 데이터화 (lesson §6.2)")
    p.add_argument("--smoke", action="store_true", help="표만 출력하고 그림은 건너뛴다")
    p.add_argument("--csv", type=str, default=None, help="players.csv 경로 (기본: 스크립트와 같은 폴더)")
    p.add_argument("--ascii-labels", action="store_true", help="한글 폰트가 있어도 영문 라벨로 렌더")
    if argv is None:
        argv = [] if _in_notebook() else sys.argv[1:]
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_korean_font(force_ascii=args.ascii_labels)

    print("=" * 78)
    print(f" W1-M1 실습 3 — 플레이어 지도 {'(smoke)' if args.smoke else ''}")
    print("=" * 78)

    df = load_players(Path(args.csv) if args.csv else None)
    print_players_table(df)
    print_layer_summary(df)

    if args.smoke:
        print("\n[smoke] 그림 렌더는 건너뜁니다. 전체 실행은 --smoke 없이.")
        return

    suffix = "_ascii" if args.ascii_labels else ""
    out = plot_landscape(df, artifacts_dir() / f"03_player_landscape{suffix}.png")
    print(f"\n[저장] {out}")
    print("\n[갱신 방법] 새 모델이 나오면 players.csv에 한 줄 추가하고 이 스크립트를 다시 돌리세요.")
    print("            1차 출처로 확인되지 않은 수치는 넣지 말고 date_confidence를 unknown으로 두세요.")


# %%
if __name__ == "__main__":
    main()
