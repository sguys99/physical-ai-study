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
# # W1-M1 실습 2 — 데이터 피라미드 + 5계층 스택 렌더
#
# lesson.md `§5.1`(데이터 피라미드)와 `§3.2`(5계층 블록도)을 matplotlib 그림으로 만듭니다.
# 문서용 그림 소스이므로, 수치가 갱신되면 이 파일의 상수만 고치면 됩니다.
#
# 산출물:
#
# | 파일 | 내용 |
# |---|---|
# | `artifacts/W1-M1/02_data_pyramid.png` | 데이터 피라미드 4층 + 양↔라벨 정확도 트레이드오프 축 |
# | `artifacts/W1-M1/02_stack_layers.png` | 5계층 스택. **주파수를 로그 스케일 축**으로 그려 대역폭 분리를 눈으로 확인 |
#
# GPU 불필요. 전체 실행 수 초.
#
# > **한글 폰트 주의** — 실행 환경에 한글 폰트가 없으면 라벨이 두부(□)로 깨집니다.
# > 아래 `setup_korean_font()`가 폰트를 탐색하고, 못 찾으면 **영문 라벨로 자동 폴백**합니다.
# > `--ascii-labels`로 강제 폴백해 두 경로를 모두 확인할 수 있습니다.

# %%
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless 고정 — plt.show() 금지

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

MODULE_ID = "W1-M1"


# %% [markdown]
# ## 0. 경로·폰트 유틸 (01번 스크립트와 동일)

# %%
_ROOT_MARKERS = ("course", "docs", "CLAUDE.md")


def find_repo_root() -> Path:
    """리포 루트 디렉토리를 찾는다 (스크립트/노트북 양쪽에서 동작).

    스크립트로 실행하면 이 파일은
    course/w1-generative-core/01-physical-ai-landscape/practice/ 아래에 있으므로
    Path(__file__).resolve().parents[4]가 리포 루트다.
    """
    try:
        start = Path(__file__).resolve().parent
    except NameError:  # 노트북에는 __file__이 없다
        start = Path.cwd().resolve()
    for cand in (start, *start.parents):
        if all((cand / m).exists() for m in _ROOT_MARKERS):
            return cand
    return start


def artifacts_dir() -> Path:
    out = find_repo_root() / "artifacts" / MODULE_ID
    out.mkdir(parents=True, exist_ok=True)
    return out


# %%
# 폰트 이름이 아니라 "실제 한글 글리프 보유 여부"로 판정한다.
# (예: 'Noto Sans Gothic'은 이름에 Gothic이 있지만 고대 고트 문자용이라 한글이 없다.)
_KO_FONT_PREFERENCE = (
    "Pretendard", "NanumGothic", "Nanum Gothic", "Malgun Gothic", "NanumBarunGothic",
    "Noto Sans KR", "Noto Sans CJK KR", "Noto Sans CJK JP", "Source Han Sans KR",
    "AppleGothic", "Spoqa Han Sans Neo", "UnDotum", "Baekmuk Gulim",
)
_PROBE_CHARS = "한글피라미드계층"

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
# ## 1. 데이터 피라미드 — lesson §5.1
#
# 위로 갈수록 데이터는 넘치는데 액션 $a_t$ 가 없고, 아래로 갈수록 $a_t$ 는 정확한데 데이터가 마릅니다.
# 정량 수치는 lesson §5.1 표에서 그대로 가져왔습니다.

# %%
@dataclass
class PyramidTier:
    """데이터 피라미드 한 층 (lesson §5.1)."""

    idx: int              # 1(최상단) ~ 4(최하단)
    title_ko: str
    title_en: str
    half_top: float       # 사다리꼴 윗변 반폭
    half_bottom: float    # 사다리꼴 아랫변 반폭
    color: str
    detail_ko: list[str] = field(default_factory=list)
    detail_en: list[str] = field(default_factory=list)


PYRAMID: list[PyramidTier] = [
    PyramidTier(
        1, "웹 비디오 (YouTube 등)", "Web video (YouTube etc.)",
        0.34, 0.28, "#dbe7f3",
        ["수십억 클립 · 액션 라벨 0", "임바디먼트 무관"],
        ["billions of clips, zero action labels", "embodiment-agnostic"],
    ),
    PyramidTier(
        2, "휴먼 비디오 · 모캡", "Human video / MoCap",
        0.28, 0.21, "#b7cfe6",
        ["수백~수천 시간 · 인간 골격 라벨",
         "SONIC 학습 데이터: 700시간 = 1억+ 프레임",
         "(arXiv:2511.07820)"],
        ["hundreds~thousands of hours, human skeleton labels",
         "SONIC training data: 700 h = 100M+ frames",
         "(arXiv:2511.07820)"],
    ),
    PyramidTier(
        3, "텔레옵 시연 (로봇 실기)", "Teleop demonstrations (real robot)",
        0.21, 0.13, "#7fa9d0",
        ["10^4 ~ 10^6 궤적 · 액션 라벨 정확",
         "Open X-Embodiment: 100만+ 궤적 / 22종 로봇 (arXiv:2310.08864)",
         "AgiBot World Beta: 1,003,672 궤적 (arXiv:2503.06669)"],
        ["10^4 ~ 10^6 trajectories, exact action labels",
         "Open X-Embodiment: 1M+ trajectories / 22 robots (arXiv:2310.08864)",
         "AgiBot World Beta: 1,003,672 trajectories (arXiv:2503.06669)"],
    ),
    PyramidTier(
        4, "실기 RL / 온라인 상호작용", "Real-robot RL / online interaction",
        0.13, 0.045, "#3d6f9e",
        ["가장 적음 · 보상까지 있음", "임바디먼트 종속"],
        ["smallest, includes reward signal", "embodiment-specific"],
    ),
]

_PYRAMID_CX = 0.30       # 피라미드 중심 x
_PYRAMID_TEXT_X = 0.72   # 오른쪽 설명 블록 시작 x


# %%
def plot_data_pyramid(out_path: Path) -> Path:
    """lesson §5.1 데이터 피라미드를 렌더한다."""
    n = len(PYRAMID)
    fig, ax = plt.subplots(figsize=(12.5, 7.6))

    for i, tier in enumerate(PYRAMID):
        y_top = n - i          # 4,3,2,1
        y_bot = y_top - 1.0
        cx = _PYRAMID_CX
        poly = Polygon(
            [
                (cx - tier.half_top, y_top),
                (cx + tier.half_top, y_top),
                (cx + tier.half_bottom, y_bot),
                (cx - tier.half_bottom, y_bot),
            ],
            closed=True,
            facecolor=tier.color,
            edgecolor="white",
            lw=2.0,
        )
        ax.add_patch(poly)

        y_mid = (y_top + y_bot) / 2.0
        # 사다리꼴 안에는 층 번호만 (아래층은 폭이 좁아 긴 텍스트가 안 들어간다)
        ax.text(
            cx, y_mid, str(tier.idx),
            ha="center", va="center", fontsize=20, fontweight="bold",
            color="white" if i >= 2 else "#1f4e79",
        )

        # 오른쪽 설명 블록 + 연결선
        half_mid = (tier.half_top + tier.half_bottom) / 2.0
        ax.plot(
            [cx + half_mid, _PYRAMID_TEXT_X - 0.03], [y_mid, y_mid],
            color="#9aa5b1", lw=1.0, ls=":",
        )
        title = t(tier.title_ko, tier.title_en)
        details = tier.detail_ko if USE_KOREAN else tier.detail_en
        ax.text(
            _PYRAMID_TEXT_X, y_mid + 0.20,
            f"{tier.idx}. {title}",
            ha="left", va="center", fontsize=12, fontweight="bold", color="#1f2d3d",
        )
        for j, line in enumerate(details):
            ax.text(
                _PYRAMID_TEXT_X, y_mid - 0.06 - 0.19 * j,
                line, ha="left", va="center", fontsize=9, color="#42536b",
            )

    # 좌우 트레이드오프 축
    ax.annotate(
        "", xy=(-0.26, n - 0.12), xytext=(-0.26, 0.12),
        arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#2e7d32"),
    )
    ax.text(
        -0.33, n / 2.0, t("데이터 양이 많아진다", "more data"),
        rotation=90, ha="center", va="center", fontsize=11, color="#2e7d32", fontweight="bold",
    )
    ax.annotate(
        "", xy=(1.62, 0.12), xytext=(1.62, n - 0.12),
        arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#c0392b"),
    )
    ax.text(
        1.69, n / 2.0,
        t("액션 라벨이 정확해진다 · 임바디먼트 종속", "action labels get exact / embodiment-specific"),
        rotation=270, ha="center", va="center", fontsize=11, color="#c0392b", fontweight="bold",
    )

    ax.set_xlim(-0.45, 1.80)
    ax.set_ylim(-0.55, n + 0.55)
    ax.axis("off")
    ax.set_title(
        t("W1-M1 · 데이터 피라미드 (lesson §5.1)", "W1-M1 · Data pyramid (lesson §5.1)"),
        fontsize=14, pad=14,
    )
    ax.text(
        -0.45, -0.42,
        t("병목은 '데이터 부족'이 아니라 '라벨 없는 상단을 하단으로 끌어내리는 법'이다"
          "  →  IDM · latent action · world model (W2-M5, W4-M1/M2)",
          "The bottleneck is not data volume but pulling unlabeled upper tiers down"
          "  ->  IDM / latent action / world model (W2-M5, W4-M1/M2)"),
        ha="left", va="center", fontsize=10, color="#5a6675", style="italic",
    )
    return save_and_check(fig, out_path)


# %% [markdown]
# ## 2. 5계층 스택 — lesson §3.2 / §3.3
#
# 주파수를 **로그 스케일 x축**에 올리면 lesson §3.3의
# $\omega_{L1} \gg \omega_{L2} \gg \omega_{L4} \gg \omega_{L5}$ 가 막대 사이 거리로 그대로 보입니다.
# L3만 고정 주파수가 없는 **이벤트 구동**(L4 호출당 1회) 계층이라 빗금으로 표시했습니다.

# %%
@dataclass
class StackLayer:
    """스택 한 계층 (lesson §3.2)."""

    tag: str
    name_ko: str
    name_en: str
    f_lo: float | None
    f_hi: float | None
    company_ko: str
    company_en: str
    color: str


STACK: list[StackLayer] = [  # 아래(빠름) → 위(느림) 순서로 y축에 쌓는다
    StackLayer("L1", "하드웨어·미들웨어", "Hardware / Middleware", 1000.0, 2000.0,
               "Unitree G1 · unitree_sdk2 / DDS", "Unitree G1 · unitree_sdk2 / DDS", "#1f4e79"),
    StackLayer("L2", "전신 제어 WBC", "Whole-Body Control", 50.0, 500.0,
               "GEAR-SONIC / HOMIE", "GEAR-SONIC / HOMIE", "#2e6da4"),
    StackLayer("L3", "액션 인터페이스", "Action Interface", None, None,
               "FSQ 기반 계층 모델", "FSQ-based hierarchical model", "#7fa9d0"),
    StackLayer("L4", "상위 지능 (VLA/월드모델)", "High-level (VLA / World Model)", 1.0, 10.0,
               "VLA / World Model", "VLA / World Model", "#b7cfe6"),
    StackLayer("L5", "인지·매핑", "Perception / Mapping", 0.5, 5.0,
               "DualMap", "DualMap", "#dbe7f3"),
]

# ytick 부제로 쓸 대역폭 분리비 (기하평균 기준, lesson §3.3 · 01번 스크립트와 동일한 계산)
_RATIO_NOTE_KO = {
    "L1": "기준",
    "L2": "L1 대비 ÷8.9",
    "L3": "이벤트 구동",
    "L4": "L2 대비 ÷50",
    "L5": "L4 대비 ÷2.0",
}
_RATIO_NOTE_EN = {
    "L1": "reference",
    "L2": "/8.9 vs L1",
    "L3": "event-driven",
    "L4": "/50 vs L2",
    "L5": "/2.0 vs L4",
}


# %%
def plot_stack_layers(out_path: Path) -> Path:
    """lesson §3.2의 5계층 스택을 로그 주파수 축 위에 렌더한다."""
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_xscale("log")

    yticks, ylabels = [], []
    for y, lyr in enumerate(STACK):
        note = (_RATIO_NOTE_KO if USE_KOREAN else _RATIO_NOTE_EN)[lyr.tag]
        yticks.append(y)
        ylabels.append(f"{lyr.tag} · {t(lyr.name_ko, lyr.name_en)}\n({note})")

        if lyr.f_lo is None:  # L3 — 고정 주파수 없음. L4 구간에 빗금으로 표시
            lo, hi = 1.0, 10.0
            ax.barh(y, hi - lo, left=lo, height=0.55,
                    facecolor="none", edgecolor=lyr.color, hatch="///", lw=1.6)
            label = t("고정 주파수 없음 · L4 호출당 1회", "no fixed rate · once per L4 call")
        else:
            lo, hi = lyr.f_lo, lyr.f_hi
            ax.barh(y, hi - lo, left=lo, height=0.55, color=lyr.color, edgecolor="#1f2d3d", lw=0.8)
            ax.plot([np.sqrt(lo * hi)], [y], marker="|", ms=16, color="#c0392b", lw=2)
            label = f"{lo:,.0f}–{hi:,.0f} Hz" if lo >= 1 else f"{lo:g}–{hi:g} Hz"

        ax.text(hi * 1.35, y + 0.14, label, va="center", ha="left", fontsize=10, color="#1f2d3d")
        ax.text(hi * 1.35, y - 0.20, t(lyr.company_ko, lyr.company_en),
                va="center", ha="left", fontsize=9.5, color="#c0392b", fontweight="bold")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_ylim(-0.7, len(STACK) - 0.3)
    ax.set_xlim(0.3, 1e5)
    ax.set_xlabel(t("동작 주파수 [Hz] — 로그 스케일", "operating frequency [Hz] — log scale"), fontsize=11)
    ax.grid(axis="x", which="major", alpha=0.35)
    ax.grid(axis="x", which="minor", alpha=0.12)
    ax.set_title(
        t("W1-M1 · 5계층 스택과 대역폭 분리 (lesson §3.2 / §3.3)",
          "W1-M1 · Five-layer stack and bandwidth separation (lesson §3.2 / §3.3)"),
        fontsize=14, pad=12,
    )
    ax.text(
        0.32, -0.62,
        t("빨간 눈금 = 기하평균 대표 주파수.  위로 갈수록 느리고 똑똑해진다."
          "  계층은 설계 취향이 아니라 연산 예산이 강제한 결과다.",
          "Red tick = geometric-mean nominal rate. Upper layers are slower and smarter;"
          " the hierarchy is forced by the compute budget."),
        fontsize=9.5, color="#5a6675", style="italic", va="center",
    )
    return save_and_check(fig, out_path)


# %% [markdown]
# ## 3. 실행

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
    p = argparse.ArgumentParser(description="W1-M1: 데이터 피라미드 + 5계층 스택 렌더 (lesson §3.2/§5.1)")
    p.add_argument("--smoke", action="store_true",
                   help="다른 실습과 인터페이스를 맞추기 위한 플래그. 이 스크립트는 원래 수 초라 결과가 동일하다")
    p.add_argument("--ascii-labels", action="store_true", help="한글 폰트가 있어도 영문 라벨로 렌더")
    if argv is None:
        argv = [] if _in_notebook() else sys.argv[1:]
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_korean_font(force_ascii=args.ascii_labels)

    print("=" * 78)
    print(f" W1-M1 실습 2 — 데이터 피라미드 + 5계층 스택 {'(smoke)' if args.smoke else ''}")
    print("=" * 78)

    out_dir = artifacts_dir()
    # 자동 폴백(폰트 없음)일 때도 파일명은 그대로 유지한다.
    # --ascii-labels로 "일부러" 영문 렌더를 뽑을 때만 별도 파일로 저장해 비교할 수 있게 한다.
    suffix = "_ascii" if args.ascii_labels else ""

    p1 = plot_data_pyramid(out_dir / f"02_data_pyramid{suffix}.png")
    print(f"[저장] {p1}")
    p2 = plot_stack_layers(out_dir / f"02_stack_layers{suffix}.png")
    print(f"[저장] {p2}")

    print("\n피라미드 층 요약 (lesson §5.1):")
    for tier in PYRAMID:
        print(f"  {tier.idx}. {tier.title_ko}")
        for line in tier.detail_ko:
            print(f"       - {line}")


# %%
if __name__ == "__main__":
    main()
