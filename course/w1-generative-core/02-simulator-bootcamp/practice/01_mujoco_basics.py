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
# # W1-M2 실습 1 — MuJoCo 최소 단위: MJCF 문자열에서 mp4까지
#
# lesson.md `§3`(계산 모델) · `§4.1`(파이프라인 블록도) · `§6`(headless 렌더링)을
# **가장 작은 모델 하나로** 전 구간 통과시키는 스크립트입니다.
#
# G1(29 DoF)은 다음 스크립트(`02`, `03`)에서 다룹니다. 여기서는 관절 1개짜리 진자로
# 파이프라인만 확인합니다. 큰 모델에서 문제가 생겼을 때 "물리가 문제인가 / 렌더가 문제인가 /
# 파일 경로가 문제인가"를 가르는 기준선이 이 스크립트입니다.
#
# 이 실습에서 확인할 것:
#
# 1. **인라인 MJCF 문자열** → `MjModel.from_xml_string` → `MjData` → `mj_step` → 렌더 → png/mp4
#    (파일 의존 없음. menagerie를 아직 클론하지 않았어도 돕니다)
# 2. `mjModel`(상수)과 `mjData`(상태)에 실제로 무엇이 들어 있는지 필드 단위로 출력 — lesson §3.1
# 3. 운동방정식 $M(q)\ddot q + c(q,\dot q) = \tau_{act} + J_c^\top f_c$ 를 **코드에서 잔차로 검증** — lesson §3.3
# 4. **`timestep` 비교 실험** — 같은 진자를 `dt`를 바꿔 적분해 궤적·에너지 오차·계산 시간을 한 장에.
#    lesson §3.5의 "정확도 ↔ 속도 트레이드오프"의 실증이고, MJX가 `timestep`을 0.002 → 0.004로
#    **올린** 이유를 숫자로 보게 됩니다.
# 5. headless 렌더 — `MUJOCO_GL=egl`, 뷰어 없이 png 1장 + 짧은 mp4
#
# **GPU 불필요. CPU에서 1분 이내.** 결과는 리포 루트의 `artifacts/W1-M2/`에 저장됩니다.

# %%
from __future__ import annotations

import os

# ⚠️ 반드시 `import mujoco` 보다 먼저. MuJoCo의 렌더 백엔드는 import 시점에 결정된다 (lesson §6.2).
#    노트북에서도 이 셀이 첫 셀이어야 하고, 이미 mujoco를 import한 커널이면 재시작해야 한다.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless 고정 — plt.show() 금지, 결과는 전부 파일로

import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402

MODULE_ID = "W1-M2"


# %% [markdown]
# ## 0. 경로 · 폰트 · 표 유틸
#
# 출력은 리포 루트의 `artifacts/W1-M2/`에 저장합니다.
# 이 파일은 `course/w1-generative-core/02-simulator-bootcamp/practice/` 아래에 있으므로
# 스크립트로 실행하면 `parents[4]`가 리포 루트입니다. 노트북에는 `__file__`이 없으므로
# cwd에서 위로 올라가며 마커 디렉토리를 찾습니다. (W1-M1과 동일한 규약)

# %%
_ROOT_MARKERS = ("course", "docs", "CLAUDE.md")


def find_repo_root() -> Path:
    """리포 루트 디렉토리를 찾는다 (스크립트/노트북 양쪽에서 동작)."""
    try:
        start = Path(__file__).resolve().parent  # .../practice
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
# 한글 폰트 탐색 (W1-M1과 동일). "이름"이 아니라 "실제 한글 글리프 보유 여부"로 판정하고,
# 없으면 영문 라벨로 폴백한다.
_KO_FONT_PREFERENCE = (
    "Pretendard",
    "NanumGothic",
    "Nanum Gothic",
    "Malgun Gothic",
    "NanumBarunGothic",
    "Noto Sans KR",
    "Noto Sans CJK KR",
    "Noto Sans CJK JP",
    "Source Han Sans KR",
    "AppleGothic",
    "Spoqa Han Sans Neo",
    "UnDotum",
    "Baekmuk Gulim",
)
_PROBE_CHARS = "한글적분오차"

USE_KOREAN = False  # setup_korean_font()가 갱신


def _has_hangul(font_path: str) -> bool:
    try:
        face = FT2Font(font_path)
        return all(face.get_char_index(ord(c)) != 0 for c in _PROBE_CHARS)
    except Exception:
        return False


def setup_korean_font(force_ascii: bool = False) -> bool:
    """한글 폰트를 찾아 matplotlib 기본 폰트로 설정. 실패하면 영문 폴백."""
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
        " (해결: apt install fonts-nanum 후 matplotlib 폰트 캐시 삭제)"
    )
    return False


def t(ko: str, en: str) -> str:
    """폰트 상황에 따라 한글/영문 라벨을 고른다."""
    return ko if USE_KOREAN else en


# %%
def _dwidth(s: str) -> int:
    """터미널 표시 폭(한글은 2칸)."""
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
    """한글 폭을 고려한 간단한 표 출력."""
    aligns = aligns or ["left"] * len(headers)
    widths = [
        max(_dwidth(headers[i]), *(_dwidth(r[i]) for r in rows)) if rows else _dwidth(headers[i])
        for i in range(len(headers))
    ]
    line = "-+-".join("-" * w for w in widths)
    print(" | ".join(_pad(h, w, "center") for h, w in zip(headers, widths)))
    print(line)
    for r in rows:
        print(" | ".join(_pad(c, w, a) for c, w, a in zip(r, widths, aligns)))


# %%
def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> Path:
    """프레임 리스트를 mp4로 저장.

    imageio(+imageio-ffmpeg)를 우선 쓰고, 없으면 mediapy로 폴백한다.
    imageio-ffmpeg는 ffmpeg 바이너리를 동봉하므로 시스템에 ffmpeg가 없어도 된다.
    프레임 크기는 16의 배수로 잡는 것이 안전하다(H.264 매크로블록).
    """
    arr = np.asarray(frames, dtype=np.uint8)  # (T, H, W, 3)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, list(arr), fps=fps, macro_block_size=None)
    except Exception as e:  # pragma: no cover - 환경 의존
        print(f"[mp4] imageio 실패({e}) → mediapy로 폴백")
        import mediapy

        mediapy.write_video(str(path), arr, fps=fps)
    return path


# %% [markdown]
# ## 1. 인라인 MJCF — 파일 없이 모델 만들기
#
# MJCF는 XML입니다. `<option>`(적분 설정) · `<worldbody>`(body 트리) · `<actuator>` · `<sensor>` ·
# `<keyframe>` 이 큰 뼈대이고, G1의 `scene.xml`도 규모만 다르지 구조는 같습니다 (lesson §4.1).
#
# 여기서는 **관절 1개짜리 진자**를 씁니다. 상태가 $(\theta, \dot\theta)$ 2차원이라
# 손으로 검산할 수 있고, 접촉이 없어서 에너지가 보존되므로 **적분 오차를 에너지로 측정**할 수 있습니다.
#
# - `<flag energy="enable"/>` → `data.energy = [위치에너지, 운동에너지]`가 채워집니다.
# - `<flag contact="disable"/>` → 접촉 계산을 꺼서 진자가 바닥을 만나도 통과합니다.
#   (이 실습에서는 순수 적분 오차만 보고 싶으므로 접촉이라는 오차원을 제거)
# - `gravity="0 0 -9.81"`, 로드 길이 0.5 m, 로드 0.5 kg + 끝단 추 1.0 kg.

# %%
PENDULUM_XML = """
<mujoco model="w1m2_pendulum">
  <option timestep="{timestep}" integrator="{integrator}" gravity="0 0 -9.81">
    <flag energy="enable" contact="disable"/>
  </option>

  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>

  <worldbody>
    <light name="top" pos="0 0 2.5" dir="0 0 -1" diffuse=".8 .8 .8"/>
    <geom name="floor" type="plane" size="2 2 .05" rgba=".85 .87 .9 1"/>
    <body name="pivot" pos="0 0 1.0">
      <!-- y축 둘레로 도는 hinge 1개. 이 관절 하나가 nq=1, nv=1을 만든다. -->
      <joint name="hinge" type="hinge" axis="0 1 0" damping="{damping}"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02"
            rgba=".18 .40 .70 1" mass="0.5"/>
      <geom name="bob" type="sphere" pos="0 0 -0.5" size="0.055"
            rgba=".80 .30 .20 1" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""


def build_pendulum(
    timestep: float = 0.002, integrator: str = "Euler", damping: float = 0.0
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """인라인 MJCF 문자열로 모델과 데이터를 만든다.

    lesson §4.1 ①→②→③ 구간. `from_xml_string`은 파일 대신 문자열을 컴파일한다는 것만 다르고
    결과(`mjModel`)는 `from_xml_path`와 동일하다.
    """
    xml = PENDULUM_XML.format(timestep=timestep, integrator=integrator, damping=damping)
    m = mujoco.MjModel.from_xml_string(xml)  # ← 컴파일 1회. 여기서 nq/nv/nu가 확정된다
    d = mujoco.MjData(m)  # ← 상태. 환경 개수만큼 만들 수 있다
    return m, d


# %% [markdown]
# ## 2. `mjModel` / `mjData` 안을 들여다보기 — lesson §3.1
#
# 두 객체의 역할 분담을 **직접 출력해서** 확인합니다.
#
# | | `mjModel` | `mjData` |
# |---|---|---|
# | 성격 | 상수(컴파일 결과) | 상태 + 매 스텝 재계산되는 중간량 |
# | 개수 | **1개** | **N개**(환경 수만큼) |
#
# 이 분리가 MJX/GPU 병렬(모델 1개 + 데이터 4096개)의 전제입니다.

# %%
def print_model_summary(m: mujoco.MjModel, title: str = "mjModel") -> None:
    """mjModel의 대표 상수 필드를 출력한다 (lesson §3.1 · §4.1 ②)."""
    print(f"\n=== [{title}] 상수 — 컴파일 결과, 매 스텝 변하지 않음 ===")
    rows = [
        ["nq  (일반화 좌표 수)", str(m.nq), "qpos의 길이"],
        ["nv  (자유도 수)", str(m.nv), "qvel/qacc/힘 벡터의 길이"],
        ["nu  (액추에이터 수)", str(m.nu), "ctrl의 길이"],
        ["nbody", str(m.nbody), "world 포함"],
        ["njnt", str(m.njnt), "관절 수"],
        ["ngeom", str(m.ngeom), "충돌·시각 지오메트리"],
        ["nkey", str(m.nkey), "키프레임 수"],
        ["opt.timestep", f"{m.opt.timestep:g}", "적분 주기 dt [s]"],
        ["opt.integrator", str(mujoco.mjtIntegrator(m.opt.integrator).name), "적분 스킴"],
        ["opt.iterations", str(m.opt.iterations), "제약 solver 반복 횟수"],
        ["sum(body_mass)", f"{m.body_mass.sum():.4f} kg", "world body 제외하면 실제 질량"],
    ]
    print_table(["필드", "값", "의미"], rows, aligns=["left", "right", "left"])

    jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    gnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)]
    print(f"  관절: {jnames}")
    print(f"  지오메트리: {gnames}")


def print_data_summary(m: mujoco.MjModel, d: mujoco.MjData, title: str = "mjData") -> None:
    """mjData의 대표 상태·중간량 필드를 출력한다 (lesson §3.1 · §4.1 ③)."""
    print(f"\n=== [{title}] 상태 — 매 mj_step마다 갱신됨 ===")
    rows = [
        ["time", f"{d.time:.4f}", "[s]"],
        ["qpos", np.array2string(d.qpos, precision=4), f"shape {d.qpos.shape} = (nq,)"],
        ["qvel", np.array2string(d.qvel, precision=4), f"shape {d.qvel.shape} = (nv,)"],
        ["qacc", np.array2string(d.qacc, precision=4), f"shape {d.qacc.shape} = (nv,)"],
        ["ctrl", np.array2string(d.ctrl, precision=4), f"shape {d.ctrl.shape} = (nu,)"],
        ["qfrc_bias", np.array2string(d.qfrc_bias, precision=4), "코리올리+원심+중력 c(q,qdot)"],
        ["qfrc_actuator", np.array2string(d.qfrc_actuator, precision=4), "ctrl -> 일반화력"],
        ["qfrc_constraint", np.array2string(d.qfrc_constraint, precision=4), "제약력 J^T f_c"],
        ["energy", np.array2string(d.energy, precision=6), "[위치, 운동] (flag energy=enable)"],
        ["ncon", str(d.ncon), "이번 스텝의 접촉점 개수 (가변!)"],
    ]
    print_table(["필드", "값", "의미"], rows, aligns=["left", "left", "left"])


# %% [markdown]
# ## 3. 운동방정식을 잔차로 확인 — lesson §3.3 eq.(1)
#
# $$
# M(q)\,\ddot q + c(q,\dot q) \;=\; \tau_{\text{act}} + \tau_{\text{passive}} + J_c^\top f_c
# \tag{1}
# $$
#
# MuJoCo 필드로 옮기면 이렇습니다.
#
# ```
# qM @ qacc  +  qfrc_bias  ==  qfrc_actuator + qfrc_passive + qfrc_applied + qfrc_constraint
# ```
#
# `qM`은 희소 저장이므로 `mj_fullM`으로 조밀 행렬을 꺼내 씁니다.
# 이 잔차가 0에 가깝다는 것을 **한 번 눈으로 확인해두면** 이후 어떤 필드가 무엇인지 헷갈리지 않습니다.

# %%
def full_mass_matrix(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    """희소 저장된 질량행렬을 (nv, nv) 조밀 행렬로 펼친다.

    ⚠️ 버전 주의: 희소 질량행렬 필드 이름이 mujoco 3.11에서 `d.qM` -> `d.M`으로 바뀌었고,
    `mj_fullM`의 시그니처도 `(m, dst, qM)` -> `(m, d, dst)`로 바뀌었습니다.
    lesson §3.3 · §4.1의 표기(`qM`)는 그 이전 버전 기준이므로, 옛 문서·예제 코드를 볼 때
    이름 차이를 감안하세요. 아래는 두 버전 모두에서 동작합니다.
    """
    M = np.zeros((m.nv, m.nv))
    try:  # mujoco >= 3.11
        mujoco.mj_fullM(m, d, M)
    except TypeError:  # mujoco < 3.11
        mujoco.mj_fullM(m, M, d.qM)  # type: ignore[attr-defined]
    return M


def eom_residual(m: mujoco.MjModel, d: mujoco.MjData) -> float:
    """lesson §3.3 eq.(1)의 잔차 노름을 계산한다.

    mj_forward 직후(또는 mj_step 직후)의 mjData에서 각 항을 꺼내 등식을 확인한다.
    """
    M = full_mass_matrix(m, d)  # eq.(1)의 M(q)
    lhs = M @ d.qacc + d.qfrc_bias  # eq.(1) 좌변
    rhs = d.qfrc_actuator + d.qfrc_passive + d.qfrc_applied + d.qfrc_constraint  # eq.(1) 우변
    return float(np.linalg.norm(lhs - rhs))


# %% [markdown]
# ## 4. 스텝 루프 — lesson §4.1의 스텝 루프 블록
#
# `mj_step` 한 번이 하는 일은 lesson §3.2의 다섯 단계입니다.
# 여기서는 그 결과인 궤적·에너지만 기록합니다.
#
# **에너지가 적분 오차의 척도인 이유**: 감쇠가 없는 진자는 물리적으로 총에너지가 보존됩니다.
# 따라서 시뮬레이션에서 관측되는 $|E(t) - E(0)| / |E(0)|$ 는 **전부 수치 오차**입니다.
# 참값을 모르는 상태에서 오차를 재는 고전적인 수법이고, 제어에서 리아푸노프 함수의 수치적 드리프트를
# 보는 것과 같은 발상입니다.

# %%
def rollout(
    timestep: float,
    integrator: str,
    duration: float,
    theta0_deg: float = 60.0,
    damping: float = 0.0,
) -> dict:
    """진자를 duration 초 적분하고 궤적·에너지·벽시계 시간을 기록한다."""
    m, d = build_pendulum(timestep=timestep, integrator=integrator, damping=damping)
    d.qpos[0] = np.deg2rad(theta0_deg)
    mujoco.mj_forward(m, d)  # 상태를 넣은 뒤 파생량(에너지 포함)을 한 번 채운다
    e0 = float(d.energy.sum())

    n_steps = int(round(duration / timestep))
    ts = np.empty(n_steps)
    qs = np.empty(n_steps)
    es = np.empty(n_steps)

    wall0 = time.perf_counter()
    for i in range(n_steps):
        mujoco.mj_step(m, d)  # lesson §3.2의 5단계가 여기서 한 번 돈다
        ts[i] = d.time
        qs[i] = d.qpos[0]
        es[i] = d.energy.sum()
    wall = time.perf_counter() - wall0

    return {
        "timestep": timestep,
        "integrator": integrator,
        "n_steps": n_steps,
        "wall": wall,
        "t": ts,
        "theta": qs,
        "energy": es,
        "e0": e0,
        "rel_err": np.abs(es - e0) / abs(e0),  # 상대 에너지 오차 = 적분 오차의 대리 지표
        "model": m,
        "data": d,
    }


# %% [markdown]
# ## 5. `timestep` 비교 실험 — lesson §3.5
#
# 같은 진자를 `dt`만 바꿔 적분하면 무슨 일이 생기는가.
#
# lesson §3.5의 표에서 classic G1은 `timestep=0.002`, MJX G1은 `timestep=0.004`입니다.
# **MJX는 정확도를 의도적으로 낮췄습니다.** 그 거래의 크기를 여기서 직접 잽니다.
#
# 두 적분기를 같이 돌립니다.
#
# - **`Euler`** — MuJoCo 기본(semi-implicit). 스텝당 비용이 싸고 오차는 대략 $O(\Delta t)$
# - **`RK4`** — 스텝당 파생량 계산이 4회라 비싸지만 오차 차수가 훨씬 높음
#
# G1이 쓰는 `implicitfast`는 속도 의존 항(감쇠·마찰)을 암묵적으로 처리하는 변형이라
# 이 진자(감쇠 0)에서는 `Euler`와 사실상 같은 결과가 나옵니다. `--integrator`로 바꿔볼 수 있습니다.

# %%
def timestep_sweep(
    timesteps: list[float], integrators: list[str], duration: float, theta0_deg: float
) -> dict[str, list[dict]]:
    """(적분기 × timestep) 격자를 전부 굴려 결과를 모은다."""
    out: dict[str, list[dict]] = {}
    for integ in integrators:
        runs = []
        for dt in timesteps:
            r = rollout(dt, integ, duration, theta0_deg)
            r.pop("model")  # 표/그림에는 불필요. 메모리만 차지
            r.pop("data")
            runs.append(r)
        out[integ] = runs
    return out


def print_sweep_table(sweep: dict[str, list[dict]], duration: float) -> None:
    """timestep 스윕 결과를 표로. lesson §3.5의 트레이드오프를 숫자로 본다."""
    print(f"\n=== [4] timestep 비교 (진자 {duration:g}s 적분, 감쇠 0 → 에너지 보존이 참값) ===")
    rows = []
    for integ, runs in sweep.items():
        ref_theta = runs[0]["theta"][-1]  # 가장 작은 dt를 그 적분기의 기준으로
        for r in runs:
            rows.append([
                integ,
                f"{r['timestep']:g}",
                f"{r['n_steps']:,}",
                f"{r['wall'] * 1e3:.1f}",
                f"{r['rel_err'][-1]:.2e}",
                f"{np.rad2deg(r['theta'][-1] - ref_theta):+.3f}",
            ])
    print_table(
        ["적분기", "dt [s]", "스텝 수", "벽시계 [ms]", "|dE|/E0 (끝)", "각도 편차 [deg]"],
        rows,
        aligns=["left", "right", "right", "right", "right", "right"],
    )
    print("  읽는 법:")
    print("   - dt를 10배 키우면 스텝 수와 벽시계 시간은 1/10로 줄지만 에너지 오차는 커진다.")
    print("   - Euler 계열은 오차가 대략 dt에 비례(1차), RK4는 훨씬 가파르게 줄어드는 대신 스텝당 비용이 약 2배.")
    print("   - '각도 편차'는 같은 적분기의 최소 dt 결과 대비. 오차가 크기가 아니라 위상으로 나타나는 것에 주목.")
    print("   - lesson §3.5: MJX G1은 timestep 0.002 -> 0.004, iterations 100 -> 5.")
    print("     정확도를 깎아 처리량을 산 것이고, 그 거래의 크기가 위 표의 규모다.")


def empirical_order(runs: list[dict]) -> float:
    """log(dt) vs log(최종 에너지 오차)의 기울기 = 경험적 수렴 차수."""
    dts = np.array([r["timestep"] for r in runs])
    errs = np.array([max(r["rel_err"][-1], 1e-16) for r in runs])
    return float(np.polyfit(np.log(dts), np.log(errs), 1)[0])


# %%
def plot_timestep_study(sweep: dict[str, list[dict]], out_path: Path, duration: float) -> Path:
    """3-panel: (a) 궤적 (b) 에너지 오차 시계열 (c) dt-오차-비용 트레이드오프."""
    base_integ = list(sweep.keys())[0]
    runs = sweep[base_integ]
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(runs)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 4.9))

    # --- (a) 기준(가장 작은 dt) 대비 각도 편차 ---
    # 궤적을 그대로 겹쳐 그리면 전부 포개져서 아무것도 안 보인다(편차가 수 도 수준).
    # 기준 대비 편차로 그려야 "오차가 위상으로 누적된다"가 보인다.
    ref = runs[0]
    for r, c in zip(runs[1:], colors[1:]):
        theta_ref = np.interp(r["t"], ref["t"], ref["theta"])  # 시간 격자가 달라 보간
        ax1.plot(r["t"], np.rad2deg(r["theta"] - theta_ref), lw=1.6, color=c,
                 label=f"dt={r['timestep']:g}", alpha=0.95)
    ax1.axhline(0, color="k", lw=0.9, alpha=0.5)
    ax1.set_xlabel(t("시간 [s]", "time [s]"))
    ax1.set_ylabel(t(f"기준(dt={ref['timestep']:g}) 대비 각도 편차 [deg]",
                     f"deviation from dt={ref['timestep']:g} [deg]"))
    ax1.set_title(t(f"(a) 오차는 크기가 아니라 위상으로 쌓인다 — {base_integ}",
                    f"(a) Error accumulates as phase, not amplitude — {base_integ}"), fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="upper left")

    # --- (b) 에너지 상대 오차 (감쇠 0이므로 전부 수치 오차) ---
    for r, c in zip(runs, colors):
        ax2.semilogy(r["t"], np.maximum(r["rel_err"], 1e-16), lw=1.6, color=c,
                     label=f"dt={r['timestep']:g}")
    ax2.set_xlabel(t("시간 [s]", "time [s]"))
    ax2.set_ylabel(t("상대 에너지 오차  $|E(t)-E_0|/|E_0|$", "relative energy error  $|E(t)-E_0|/|E_0|$"))
    ax2.set_title(t("(b) 보존돼야 할 에너지가 새는 양 = 적분 오차",
                    "(b) Energy drift = integration error"), fontsize=11)
    ax2.set_ylim(bottom=1e-13)  # |E-E0|가 0을 스쳐 지나가며 생기는 바닥 스파이크를 잘라낸다
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8, loc="lower right")

    # --- (c) dt vs 오차 vs 비용 ---
    markers = {"Euler": "o", "implicitfast": "s", "RK4": "^", "implicit": "D"}
    for integ, rs in sweep.items():
        dts = np.array([r["timestep"] for r in rs])
        errs = np.array([max(r["rel_err"][-1], 1e-16) for r in rs])
        slope = empirical_order(rs)
        ax3.loglog(dts, errs, marker=markers.get(integ, "o"), lw=1.8,
                   label=f"{integ} (기울기 {slope:.1f})" if USE_KOREAN else f"{integ} (slope {slope:.1f})")
    ax3.set_xlabel(t("적분 주기  dt [s]", "timestep  dt [s]"))
    ax3.set_ylabel(t("최종 상대 에너지 오차", "final relative energy error"))
    ax3.set_title(t("(c) 정확도 ↔ 속도 트레이드오프 (lesson §3.5)",
                    "(c) Accuracy vs speed trade-off (lesson 3.5)"), fontsize=11)
    ax3.grid(alpha=0.3, which="both")

    # G1 classic / MJX의 timestep을 세로선으로 표시 — 어느 영역에서 노는지 감을 잡는다
    for dt_mark, label, color in ((0.002, "G1 classic 0.002", "#1f4e79"),
                                  (0.004, "G1 MJX 0.004", "#c0392b")):
        ax3.axvline(dt_mark, ls="--", lw=1.2, color=color, alpha=0.75)
        ax3.text(dt_mark, ax3.get_ylim()[1], f" {label}", rotation=90,
                 va="top", ha="left", fontsize=8, color=color)
    ax3.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        t(f"W1-M2 · MuJoCo 적분 기초 — 진자 {duration:g}s (lesson §3.2 · §3.5)",
          f"W1-M2 · MuJoCo integration basics — pendulum {duration:g}s (lesson 3.2 / 3.5)"),
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# %% [markdown]
# ## 6. headless 렌더 — lesson §6
#
# **뷰어는 쓰지 않습니다.** 클라우드 인스턴스에는 디스플레이가 없습니다.
# `mujoco.Renderer`로 오프스크린 렌더 → numpy 배열 → 파일 저장이 이 저장소의 유일한 관찰 경로입니다.
#
# 세 가지만 기억하세요.
#
# 1. `Renderer(m, height, width)` — **(높이, 너비) 순서.** `(240, 320)`을 넣으면 `(240, 320, 3)`이 나옵니다.
# 2. `r.update_scene(d)` 로 mjData를 mjvScene에 반영한 뒤 `r.render()`.
# 3. 다 쓰면 `r.close()`. 안 부르면 종료 시 `Renderer.__del__`에서 EGL traceback이 뜰 수 있습니다
#    (`Exception ignored in:` 접두사 — **무해합니다.** lesson §6.2)

# %%
def render_frames(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    duration: float,
    fps: int,
    height: int,
    width: int,
) -> tuple[list[np.ndarray], float]:
    """duration 초를 적분하며 1/fps 간격으로 프레임을 모은다.

    물리 스텝(dt=opt.timestep)과 렌더 주기(1/fps)는 별개다.
    dt=0.002, fps=25면 20 물리 스텝마다 1프레임. lesson §4.1의 'N 스텝마다 1프레임'.
    """
    frames: list[np.ndarray] = []
    r = mujoco.Renderer(m, height, width)  # ← (height, width) 순서
    wall0 = time.perf_counter()
    try:
        while d.time < duration:
            mujoco.mj_step(m, d)
            if len(frames) < d.time * fps:
                r.update_scene(d)
                frames.append(r.render().copy())  # (H, W, 3) uint8
    finally:
        r.close()  # ← 명시 호출. 종료 시 EGL traceback을 예방한다
    return frames, time.perf_counter() - wall0


# %% [markdown]
# ## 7. 실행

# %%
def _in_notebook() -> bool:
    """노트북/커널 안에서 도는가 (argparse가 커널 인자를 먹지 않도록)."""
    try:
        from IPython import get_ipython  # type: ignore

        if get_ipython() is not None:
            return True
    except Exception:
        pass
    argv0 = Path(sys.argv[0]).name.lower() if sys.argv else ""
    return argv0 == "" or "ipykernel" in argv0 or "jupyter" in argv0 or "colab" in argv0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W1-M2 실습 1: MuJoCo 최소 단위 — MJCF 문자열에서 mp4까지 (lesson §3 · §4.1 · §6)"
    )
    p.add_argument("--smoke", action="store_true", help="짧게 (적분 3s, mp4 1s) — 수 초 안에 완주")
    p.add_argument("--duration", type=float, default=6.0, help="timestep 실험의 적분 길이 [s] (기본 6)")
    p.add_argument("--theta0", type=float, default=60.0, help="진자 초기 각도 [deg] (기본 60)")
    p.add_argument("--integrator", default="Euler",
                   choices=["Euler", "implicitfast", "implicit", "RK4"],
                   help="(a)(b) 패널에 쓸 기준 적분기 (기본 Euler)")
    p.add_argument("--video-seconds", type=float, default=3.0, help="mp4 길이 [s] (기본 3)")
    p.add_argument("--fps", type=int, default=25, help="mp4 fps (기본 25)")
    p.add_argument("--height", type=int, default=240, help="렌더 높이 [px] (기본 240)")
    p.add_argument("--width", type=int, default=320, help="렌더 너비 [px] (기본 320)")
    p.add_argument("--ascii-labels", action="store_true", help="한글 폰트가 있어도 영문 라벨로 렌더")
    if argv is None:
        argv = [] if _in_notebook() else sys.argv[1:]
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    duration = 3.0 if args.smoke else args.duration
    video_seconds = 1.0 if args.smoke else args.video_seconds
    fps = 20 if args.smoke else args.fps
    suffix = "_ascii" if args.ascii_labels else ""

    setup_korean_font(force_ascii=args.ascii_labels)
    out_dir = artifacts_dir()

    print("=" * 84)
    print(f" W1-M2 실습 1 — MuJoCo 최소 단위 {'(smoke)' if args.smoke else ''}")
    print(f" MUJOCO_GL={os.environ.get('MUJOCO_GL')}  ·  mujoco {mujoco.__version__}")
    print("=" * 84)

    # ---- [1][2] 모델·데이터 들여다보기 ----
    m, d = build_pendulum(timestep=0.002, integrator=args.integrator)
    d.qpos[0] = np.deg2rad(args.theta0)
    mujoco.mj_forward(m, d)

    print_model_summary(m)
    print_data_summary(m, d, "mjData @ t=0 (mj_forward 직후)")

    # ---- [3] eq.(1) 잔차 확인 ----
    print("\n=== [3] 운동방정식 잔차 (lesson §3.3 eq.(1)) ===")
    res0 = eom_residual(m, d)
    for _ in range(100):
        mujoco.mj_step(m, d)
    res1 = eom_residual(m, d)
    print(f"  ||M*qacc + qfrc_bias - (qfrc_actuator + qfrc_passive + qfrc_applied + qfrc_constraint)||")
    print(f"    t=0.000 : {res0:.3e}")
    print(f"    t={d.time:.3f} : {res1:.3e}")
    assert res0 < 1e-9 and res1 < 1e-9, "eq.(1)이 성립하지 않는다 — 필드 해석이 잘못됐다"
    print("  -> 0에 가까우면 eq.(1)의 각 항을 올바른 필드에 대응시킨 것이다. (assert 통과)")

    # ---- [4] timestep 비교 ----
    timesteps = [0.0005, 0.002, 0.005, 0.02] if args.smoke else [0.0002, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    integrators = [args.integrator, "RK4"] if args.integrator != "RK4" else ["Euler", "RK4"]
    sweep = timestep_sweep(timesteps, integrators, duration, args.theta0)
    print_sweep_table(sweep, duration)
    for integ, runs in sweep.items():
        print(f"  경험적 수렴 차수({integ}) = log-log 기울기 {empirical_order(runs):.2f}")

    fig_path = plot_timestep_study(sweep, out_dir / f"01_timestep_study{suffix}.png", duration)
    print(f"\n[저장] {fig_path}")

    # ---- [5] 렌더 ----
    print("\n=== [5] headless 렌더 (lesson §6) ===")
    m, d = build_pendulum(timestep=0.002, integrator=args.integrator)
    d.qpos[0] = np.deg2rad(args.theta0)
    mujoco.mj_forward(m, d)

    r = mujoco.Renderer(m, args.height, args.width)
    r.update_scene(d)
    px = r.render()
    r.close()
    print(f"  Renderer(m, {args.height}, {args.width}) -> render() shape={px.shape} dtype={px.dtype}")
    assert px.shape == (args.height, args.width, 3), "Renderer 인자는 (height, width) 순서다"

    png_path = out_dir / "01_pendulum_t0.png"
    plt.imsave(png_path, px)
    print(f"[저장] {png_path}")

    frames, wall = render_frames(m, d, video_seconds, fps, args.height, args.width)
    mp4_path = write_mp4(out_dir / "01_pendulum.mp4", frames, fps)
    print(f"  프레임 {len(frames)}장 ({args.height}x{args.width}) 렌더에 {wall:.2f}s "
          f"= 프레임당 {wall / max(len(frames), 1) * 1e3:.0f} ms")
    print(f"[저장] {mp4_path}")

    print("\n" + "-" * 84)
    print(" 다음: 02_g1_inspect.py — 같은 파이프라인을 G1 29 DoF 모델에 적용한다")
    print(" 종료 직후 'Exception ignored in: <function Renderer.__del__>' + EGLError가 보여도")
    print(" 무해합니다 (lesson §6.2). 위에 [저장] 줄이 찍혔다면 파일은 정상입니다.")
    print("-" * 84)


# %%
if __name__ == "__main__":
    main()
