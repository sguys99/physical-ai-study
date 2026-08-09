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
# # W1-M2 실습 3 — "내가 로봇을 움직였다": G1에 sin파 명령 넣기
#
# 이 모듈의 하이라이트입니다. `stand` 키프레임에서 시작해 지정한 관절군에 sin파 **목표 관절각**을
# 흘려 넣고, 결과를 mp4로 저장합니다.
#
# lesson.md `§5.4`(관절 인덱스) · `§5.5`(position 액추에이터) · `§6`(headless 렌더)의 실행판입니다.
#
# 확인할 것:
#
# 1. `data.ctrl`은 **토크가 아니라 목표 관절각[rad]** 이다 — 코드에서 수식으로 검증
#    $$f = k_p(\texttt{ctrl} - q) - k_v \dot q \tag{1}$$
# 2. 명령은 반드시 `jnt_range`(= `ctrlrange`, `inheritrange="1"`)로 클립한다
# 3. **추종 오차와 위상 지연** — kp=500 임계감쇠 위치 서보가 sin 명령을 얼마나 따라가는가.
#    관절마다 유효관성이 달라 서보 대역폭이 다르고, 그래서 같은 명령에도 지연이 다르다
# 4. **다리를 흔들면 로봇이 넘어진다.** 자유 관절이라 아무도 부양해주지 않는다 —
#    이것이 W3에서 RL 보행 정책이 필요한 이유의 물리적 실체
#
# 출력:
# - `artifacts/W1-M2/g1_sin_{joints}.mp4` — 롤아웃 영상
# - `artifacts/W1-M2/03_tracking_{joints}.png` — 목표 vs 실제 추종 그래프 (교육의 핵심)
#
# **GPU 불필요.** 기본 설정(6초, 30fps, 480x640)에서 CPU로 20~30초.
# `--smoke`를 붙이면 수 초 안에 완주합니다. **먼저 `--smoke`로 경로를 확인하세요.**

# %%
from __future__ import annotations

import os

# ⚠️ 반드시 `import mujoco` 보다 먼저 (lesson §6.2).
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless 고정

import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402

MODULE_ID = "W1-M2"
DAMPRATIO = 1.0  # lesson §5.5 — <position kp="500" dampratio="1"/>
FALL_HEIGHT = 0.40  # 골반 높이가 이 아래로 내려가면 "넘어졌다"고 본다 [m] (stand는 0.79)


# %% [markdown]
# ## 0. 경로 · 폰트 · 표 유틸 (01·02와 동일 규약)

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


def resolve_menagerie(arg: str | None = None) -> Path:
    """--menagerie > MENAGERIE_PATH > <repo>/repos/mujoco_menagerie"""
    cand = arg or os.environ.get("MENAGERIE_PATH") or str(find_repo_root() / "repos" / "mujoco_menagerie")
    p = Path(cand).expanduser().resolve()
    if not (p / "unitree_g1").is_dir():
        raise SystemExit(
            f"[에러] mujoco_menagerie를 찾지 못했습니다: {p}\n"
            "  해결: git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git "
            "repos/mujoco_menagerie\n"
            "  또는: --menagerie <경로> / 환경변수 MENAGERIE_PATH 지정"
        )
    return p


# %%
_KO_FONT_PREFERENCE = (
    "Pretendard", "NanumGothic", "Nanum Gothic", "Malgun Gothic", "NanumBarunGothic",
    "Noto Sans KR", "Noto Sans CJK KR", "Noto Sans CJK JP", "Source Han Sans KR",
    "AppleGothic", "Spoqa Han Sans Neo", "UnDotum", "Baekmuk Gulim",
)
_PROBE_CHARS = "한글관절추종"

USE_KOREAN = False


def _has_hangul(font_path: str) -> bool:
    try:
        face = FT2Font(font_path)
        return all(face.get_char_index(ord(c)) != 0 for c in _PROBE_CHARS)
    except Exception:
        return False


def setup_korean_font(force_ascii: bool = False) -> bool:
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
    print("[font] 경고: 한글 글리프 폰트를 찾지 못해 영문 라벨로 폴백합니다."
          " (해결: apt install fonts-nanum 후 matplotlib 캐시 삭제)")
    return False


def t(ko: str, en: str) -> str:
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
    widths = [
        max(_dwidth(headers[i]), *(_dwidth(r[i]) for r in rows)) if rows else _dwidth(headers[i])
        for i in range(len(headers))
    ]
    print(" | ".join(_pad(h, w, "center") for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(_pad(c, w, a) for c, w, a in zip(r, widths, aligns)))


def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> Path:
    """imageio(+imageio-ffmpeg) 우선, 실패하면 mediapy 폴백."""
    arr = np.asarray(frames, dtype=np.uint8)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, list(arr), fps=fps, macro_block_size=None)
    except Exception as e:  # pragma: no cover
        print(f"[mp4] imageio 실패({e}) → mediapy로 폴백")
        import mediapy

        mediapy.write_video(str(path), arr, fps=fps)
    return path


# %% [markdown]
# ## 1. 관절군 선택 — lesson §5.4의 29 DoF 구성
#
# ```
# 다리 12 (6×2) + 허리 3 + 팔 14 (7×2) = 29
# ```
#
# **`ctrl` 인덱스는 관절 이름으로 찾습니다.** 손으로 세지 않습니다 (lesson §5.4).
# HOMIE가 "하체 RL + 상체 텔레옵"으로 상·하체를 나누는 것도 결국 이 인덱스 분할입니다(lesson 「회사 스택 연결」).

# %%
GROUP_KEYS = {
    "legs": ("hip", "knee", "ankle"),
    "waist": ("waist",),
    "arms": ("shoulder", "elbow", "wrist"),
}


def actuator_names(m: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}" for i in range(m.nu)]


def select_ctrl_ids(m: mujoco.MjModel, group: str) -> list[int]:
    """관절군 -> ctrl 인덱스 리스트. 이름 매칭이므로 모델이 바뀌어도 깨지지 않는다."""
    names = actuator_names(m)
    if group == "all":
        return list(range(m.nu))
    keys = GROUP_KEYS[group]
    ids = [i for i, n in enumerate(names) if any(k in n for k in keys)]
    if not ids:
        raise SystemExit(f"[에러] '{group}' 관절군에 해당하는 액추에이터가 없습니다: {names}")
    return ids


def actuator_gains(m: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(kp, kv, M_eff). lesson §5.5 eq.(2): M_eff = (kv/2ζ)²/kp"""
    kp = m.actuator_gainprm[:, 0].copy()  # gainprm[0] = kp
    kv = -m.actuator_biasprm[:, 2].copy()  # biasprm = [0, -kp, -kv]
    m_eff = np.where(kp > 0, (kv / (2 * DAMPRATIO)) ** 2 / np.maximum(kp, 1e-12), np.nan)
    return kp, kv, m_eff


# %% [markdown]
# ## 2. sin파 명령 — `ctrl`은 목표 관절각이다
#
# $$
# \texttt{ctrl}_i(t) = \operatorname{clip}\Big(q^{stand}_i + A\sin(2\pi f t),\; \text{lo}_i,\; \text{hi}_i\Big)
# \tag{2}
# $$
#
# 그러면 액추에이터가 내는 힘은 lesson §5.5의
# $f = k_p(\texttt{ctrl} - q) - k_v\dot q$ 입니다.
#
# **클립을 왜 하는가**: `ctrlrange` 밖의 명령을 주면 MuJoCo가 조용히 `ctrlrange`로 잘라냅니다
# (`ctrllimited`가 켜져 있을 때). 즉 로봇이 부서지지는 않지만 **내가 준 명령과 실제 적용된 명령이
# 달라집니다.** 그 상태로 추종 오차를 보면 "서보가 나쁘다"는 잘못된 결론에 도달합니다.
# 스스로 클립하고 **몇 개가 잘렸는지 세는 것**이 안전한 습관입니다.
# (RL 정책에서 `action_scale`을 두고 기본 자세 대비 오프셋을 제한하는 것도 같은 이유입니다.)

# %%
def sine_targets(
    q_stand: np.ndarray,
    ctrl_ids: list[int],
    t_now: float,
    amp: float,
    freq: float,
) -> np.ndarray:
    """eq.(2)의 클립 전 목표각 벡터 [nu]."""
    tgt = q_stand.copy()
    tgt[ctrl_ids] = q_stand[ctrl_ids] + amp * np.sin(2.0 * np.pi * freq * t_now)  # eq.(2)
    return tgt


def verify_ctrl_is_position(
    m: mujoco.MjModel, d: mujoco.MjData, q_pre: np.ndarray, dq_pre: np.ndarray
) -> float:
    """eq.(1) f = kp(ctrl - q) - kv*qdot 를 actuator_force와 대조해 잔차를 돌려준다.

    이 잔차가 0이면 'ctrl은 토크가 아니라 목표 관절각'이 코드 수준에서 확인된 것이다.

    ⚠️ 타이밍 주의: `mj_step` 직후의 `d.actuator_force`는 **그 스텝의 시작 상태**로 계산된 값이고
    `d.qpos`/`d.qvel`은 **적분이 끝난 뒤**의 값입니다. 그래서 스텝 직후의 qpos로 대조하면
    한 스텝만큼 어긋나 잔차가 남습니다. 스텝 전에 찍어둔 (q_pre, dq_pre)로 비교해야 정확히 0이 됩니다.
    시뮬레이터에서 "언제의 값인가"를 따지는 습관은 지연 모델링·sim2real에서 그대로 쓰입니다.
    """
    kp, kv, _ = actuator_gains(m)
    f_pred = kp * (d.ctrl - q_pre) - kv * dq_pre  # eq.(1)
    return float(np.max(np.abs(f_pred - d.actuator_force)))


# %% [markdown]
# ## 3. 롤아웃 + 렌더
#
# 물리 스텝(`dt = 0.002 s`)과 렌더 주기(`1/fps`)는 별개입니다. 30fps면 약 17 물리 스텝마다 1프레임.
# 카메라는 골반을 따라다니도록 매 프레임 `lookat`을 갱신합니다 (**뷰어가 아니라 오프스크린 카메라**).

# %%
def rollout(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    ctrl_ids: list[int],
    q_stand: np.ndarray,
    seconds: float,
    amp: float,
    freq: float,
    fps: int,
    height: int,
    width: int,
    render: bool = True,
) -> dict:
    """sin파 명령을 넣으며 시뮬레이션하고 궤적과 프레임을 기록한다."""
    lo = m.actuator_ctrlrange[:, 0].copy()
    hi = m.actuator_ctrlrange[:, 1].copy()

    n_steps = int(round(seconds / m.opt.timestep))
    ts = np.empty(n_steps)
    cmd = np.empty((n_steps, m.nu))  # 클립 후 실제 적용된 목표각
    raw = np.empty((n_steps, m.nu))  # 클립 전 명령
    act = np.empty((n_steps, m.nu))  # 실제 관절각 qpos[7:]
    pelvis_z = np.empty(n_steps)
    n_clipped = 0

    frames: list[np.ndarray] = []
    renderer = mujoco.Renderer(m, height, width) if render else None
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 3.0
    cam.elevation = -12.0
    cam.azimuth = 135.0

    res_max = 0.0
    wall0 = time.perf_counter()
    try:
        for i in range(n_steps):
            tgt_raw = sine_targets(q_stand, ctrl_ids, d.time, amp, freq)  # eq.(2)
            tgt = np.clip(tgt_raw, lo, hi)  # ctrlrange = jnt_range (inheritrange="1")
            n_clipped += int(np.count_nonzero(~np.isclose(tgt, tgt_raw)))
            d.ctrl[:] = tgt

            # eq.(1) 검증용으로 스텝 '전' 상태를 찍어둔다 (docstring의 타이밍 주의 참조)
            q_pre = d.qpos[7:7 + m.nu].copy()
            dq_pre = d.qvel[6:6 + m.nu].copy()

            mujoco.mj_step(m, d)

            ts[i] = d.time
            raw[i] = tgt_raw
            cmd[i] = tgt
            act[i] = d.qpos[7:7 + m.nu]
            pelvis_z[i] = d.qpos[2]
            res_max = max(res_max, verify_ctrl_is_position(m, d, q_pre, dq_pre))  # eq.(1) 검증

            if renderer is not None and len(frames) < d.time * fps:
                cam.lookat[:] = d.xpos[1]  # body 1 = pelvis. 로봇을 계속 화면 중앙에
                renderer.update_scene(d, camera=cam)
                frames.append(renderer.render().copy())
    finally:
        if renderer is not None:
            renderer.close()  # ← 명시 호출 (lesson §6.2의 EGL traceback 예방)
    wall = time.perf_counter() - wall0

    fallen = np.flatnonzero(pelvis_z < FALL_HEIGHT)
    return dict(
        t=ts, cmd=cmd, raw=raw, act=act, pelvis_z=pelvis_z, frames=frames,
        n_clipped=n_clipped, wall=wall, eq1_residual=res_max,
        fall_time=float(ts[fallen[0]]) if fallen.size else None,
        n_steps=n_steps,
    )


# %% [markdown]
# ## 4. 추종 오차와 위상 지연 — **여기가 이 실습의 핵심**
#
# kp=500 · 임계감쇠($\zeta=1$) 위치 서보는 2차 시스템입니다.
#
# $$
# \frac{q}{q_{des}}(s) = \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2},
# \qquad \omega_n = \sqrt{\frac{k_p}{M_{\text{eff}}}}
# \tag{3}
# $$
#
# 명령 주파수 $\omega = 2\pi f$ 에서의 위상 지연은
#
# $$
# \phi = -\arctan\!\frac{2\zeta\,\omega\,\omega_n}{\omega_n^2 - \omega^2},
# \qquad \text{시간 지연} = -\phi/\omega
# \tag{4}
# $$
#
# 실측 위상은 실제 궤적을 $a\sin\omega t + b\cos\omega t + c$ 로 최소자승 적합해서 뽑습니다.
#
# **예측과 실측이 정확히 맞지는 않습니다.** eq.(3)은 관절 하나를 떼어낸 1자유도 근사이고,
# 실제 $M(q)$는 자세에 따라 변하며 관절끼리 커플링돼 있고 중력 항도 있습니다.
# 그래도 **"유효관성이 큰 관절일수록 지연이 크다"는 경향**은 그대로 나옵니다 —
# 그 경향을 확인하는 것이 목적입니다.

# %%
def fit_sinusoid(ts: np.ndarray, y: np.ndarray, freq: float) -> tuple[float, float]:
    """y ≈ a sin(wt) + b cos(wt) + c 를 최소자승 적합해 (진폭, 위상[rad])을 돌려준다."""
    w = 2 * np.pi * freq
    A = np.stack([np.sin(w * ts), np.cos(w * ts), np.ones_like(ts)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef[0], coef[1]
    return float(np.hypot(a, b)), float(np.arctan2(b, a))


def predicted_phase(freq: float, kp: float, m_eff: float, zeta: float = DAMPRATIO) -> float:
    """eq.(4) 2차 시스템 위상 지연 [rad] (음수)."""
    w = 2 * np.pi * freq
    wn = np.sqrt(kp / m_eff)  # eq.(3)
    return float(-np.arctan2(2 * zeta * w * wn, wn**2 - w**2))


def tracking_report(
    m: mujoco.MjModel,
    res: dict,
    ctrl_ids: list[int],
    freq: float,
    amp: float,
    settle_periods: float = 0.5,
) -> list[dict]:
    """구동된 관절별 추종 오차·위상 지연을 계산한다."""
    kp, kv, m_eff = actuator_gains(m)
    names = actuator_names(m)
    ts, cmd, act = res["t"], res["cmd"], res["act"]

    # 초기 과도구간 제외. 넘어졌으면 넘어지기 전까지만 본다.
    t_start = settle_periods / max(freq, 1e-9)
    t_end = res["fall_time"] if res["fall_time"] is not None else ts[-1]
    mask = (ts >= t_start) & (ts <= t_end)
    if mask.sum() < 20:  # smoke처럼 너무 짧으면 전 구간 사용
        mask = np.ones_like(ts, dtype=bool)
    # 적합 구간이 몇 주기인가. 2주기 미만이면 위상 추정이 신뢰할 수 없다.
    n_periods = float((ts[mask][-1] - ts[mask][0]) * freq)

    out = []
    for i in ctrl_ids:
        err = act[mask, i] - cmd[mask, i]
        amp_meas, phase_meas = fit_sinusoid(ts[mask], act[mask, i], freq)
        _, phase_cmd = fit_sinusoid(ts[mask], cmd[mask, i], freq)
        dphi = np.arctan2(np.sin(phase_meas - phase_cmd), np.cos(phase_meas - phase_cmd))
        out.append(dict(
            ctrl_id=i,
            name=names[i],
            kp=float(kp[i]),
            kv=float(kv[i]),
            m_eff=float(m_eff[i]),
            wn_hz=float(np.sqrt(kp[i] / m_eff[i]) / (2 * np.pi)),
            rms_err=float(np.sqrt(np.mean(err**2))),
            max_err=float(np.max(np.abs(err))),
            amp_ratio=float(amp_meas / amp) if amp > 0 else np.nan,
            lag_meas_ms=float(-dphi / (2 * np.pi * freq) * 1e3),
            lag_pred_ms=float(-predicted_phase(freq, kp[i], m_eff[i]) / (2 * np.pi * freq) * 1e3),
            n_periods=n_periods,
        ))
    return out


def print_tracking_table(rep: list[dict], freq: float, max_rows: int = 8) -> None:
    print(f"\n=== [3] 추종 오차와 위상 지연 (명령 {freq:g} Hz) ===")
    rep_sorted = sorted(rep, key=lambda r: -r["m_eff"])
    show = rep_sorted if len(rep_sorted) <= max_rows else rep_sorted[:max_rows // 2] + rep_sorted[-max_rows // 2:]
    rows = [[
        r["name"],
        f"{r['m_eff']:.4f}",
        f"{r['wn_hz']:.1f}",
        f"{r['rms_err'] * 1e3:.2f}",
        f"{r['max_err'] * 1e3:.2f}",
        f"{r['lag_meas_ms']:+.1f}",
        f"{r['lag_pred_ms']:+.1f}",
    ] for r in show]
    print_table(
        ["관절", "M_eff [kg·m²]", "w_n [Hz]", "RMS오차 [mrad]", "최대오차 [mrad]",
         "지연 실측 [ms]", "지연 eq.(4) [ms]"],
        rows,
        aligns=["left"] + ["right"] * 6,
    )
    if len(rep_sorted) > max_rows:
        print(f"  (유효관성 상위 {max_rows // 2}개 + 하위 {max_rows // 2}개만 표시. 전체 {len(rep_sorted)}개)")
    n_periods = rep[0].get("n_periods", 0.0) if rep else 0.0
    if n_periods < 2.0:
        print(f"  ⚠️ 위상 적합 구간이 {n_periods:.1f}주기뿐입니다(2주기 미만).")
        print("     '지연 실측' 값은 신뢰할 수 없습니다 — --smoke를 빼거나 --seconds를 늘리세요.")
    print("  읽는 법:")
    print("   - w_n = sqrt(kp/M_eff)가 서보 대역폭. 명령 주파수가 w_n보다 훨씬 낮으면 오차가 거의 없다.")
    print("   - --freq 를 올려 명령을 w_n에 가깝게 가져가면 진폭이 줄고 지연이 커진다. 직접 해볼 것.")
    print("   - eq.(4) 예측은 1자유도 근사다. 실제로는 M(q)가 자세에 따라 변하고 관절끼리 커플링된다.")
    print("     값이 정확히 맞지 않는 것이 정상이고, '관성이 크면 지연이 크다'는 경향을 보는 것이 목적.")


# %% [markdown]
# ## 5. 그림 — 목표 vs 실제

# %%
def plot_tracking(
    res: dict, rep: list[dict], freq: float, group: str, out_path: Path
) -> Path:
    """2x2: (a) 목표 vs 실제 (b) 추종 오차 (c) 골반 높이 (d) 관성-지연 관계."""
    ts = res["t"]
    # 유효관성 최대/중앙/최소 세 관절을 대표로
    rep_sorted = sorted(rep, key=lambda r: r["m_eff"])
    picks = [rep_sorted[-1], rep_sorted[len(rep_sorted) // 2], rep_sorted[0]]
    colors = ["#c0392b", "#1f4e79", "#27ae60"]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.4))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # --- (a) 목표 vs 실제 ---
    for r, c in zip(picks, colors):
        i = r["ctrl_id"]
        ax_a.plot(ts, np.rad2deg(res["cmd"][:, i]), ls="--", lw=1.4, color=c, alpha=0.8)
        ax_a.plot(ts, np.rad2deg(res["act"][:, i]), lw=1.9, color=c,
                  label=f"{r['name']}  (M_eff={r['m_eff']:.3f})")
    ax_a.set_xlabel(t("시간 [s]", "time [s]"))
    ax_a.set_ylabel(t("관절각 [deg]", "joint angle [deg]"))
    ax_a.set_title(t("(a) 점선 = ctrl(목표각), 실선 = qpos(실제각)",
                     "(a) dashed = ctrl (target), solid = qpos (actual)"), fontsize=11)
    ax_a.grid(alpha=0.3)
    ax_a.legend(fontsize=8)

    # --- (b) 추종 오차 ---
    for r, c in zip(picks, colors):
        i = r["ctrl_id"]
        ax_b.plot(ts, (res["act"][:, i] - res["cmd"][:, i]) * 1e3, lw=1.5, color=c,
                  label=f"{r['name']}  RMS={r['rms_err'] * 1e3:.2f} mrad")
    ax_b.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax_b.set_xlabel(t("시간 [s]", "time [s]"))
    ax_b.set_ylabel(t("추종 오차  q - ctrl  [mrad]", "tracking error  q - ctrl  [mrad]"))
    ax_b.set_title(t("(b) 추종 오차 — kp=500 위치 서보가 얼마나 따라가는가",
                     "(b) Tracking error — how well the kp=500 servo follows"), fontsize=11)
    ax_b.grid(alpha=0.3)
    ax_b.legend(fontsize=8)

    # --- (c) 골반 높이 = 넘어졌는가 ---
    ax_c.plot(ts, res["pelvis_z"], lw=2.0, color="#1f4e79")
    ax_c.axhline(0.79, ls=":", lw=1.2, color="#7f8c8d")
    ax_c.text(ts[0], 0.79, t(" stand 키프레임 0.79 m", " stand keyframe 0.79 m"),
              va="bottom", fontsize=8, color="#7f8c8d")
    ax_c.axhline(FALL_HEIGHT, ls="--", lw=1.4, color="#c0392b")
    ax_c.text(ts[0], FALL_HEIGHT, t(" 넘어짐 판정선", " fall threshold"),
              va="bottom", fontsize=8, color="#c0392b")
    if res["fall_time"] is not None:
        ax_c.axvline(res["fall_time"], ls="-", lw=1.4, color="#c0392b", alpha=0.6)
        # stand 기준선(0.79) 텍스트와 겹치지 않도록 축 중간 높이에 붙인다
        ax_c.text(res["fall_time"], 0.55,
                  t(f" t={res['fall_time']:.2f}s 넘어짐", f" fell at t={res['fall_time']:.2f}s"),
                  va="center", fontsize=10, color="#c0392b")
    ax_c.set_ylim(0, max(0.95, float(res["pelvis_z"].max()) * 1.2))  # 0.79 기준선 라벨 자리 확보
    ax_c.set_xlabel(t("시간 [s]", "time [s]"))
    ax_c.set_ylabel(t("골반 높이  qpos[2]  [m]", "pelvis height  qpos[2]  [m]"))
    ax_c.set_title(t("(c) 균형 정책이 없으면 어떻게 되는가",
                     "(c) What happens without a balance policy"), fontsize=11)
    ax_c.grid(alpha=0.3)

    # --- (d) 유효관성 vs 위상 지연 ---
    meffs = np.array([r["m_eff"] for r in rep])
    lag_m = np.array([r["lag_meas_ms"] for r in rep])
    lag_p = np.array([r["lag_pred_ms"] for r in rep])
    order = np.argsort(meffs)
    ax_d.semilogx(meffs[order], lag_p[order], "-", lw=1.8, color="#7f8c8d",
                  label=t("eq.(4) 2차계 예측", "eq.(4) 2nd-order prediction"))
    ax_d.semilogx(meffs, lag_m, "o", ms=6, color="#c0392b",
                  label=t("실측 (sin 적합)", "measured (sine fit)"))
    ax_d.set_xlabel(t("유효관성  $M_{eff}=(k_v/2\\zeta)^2/k_p$  [kg·m²]",
                      "effective inertia  $M_{eff}=(k_v/2\\zeta)^2/k_p$  [kg·m²]"))
    ax_d.set_ylabel(t("위상 지연 [ms]", "phase lag [ms]"))
    ax_d.set_title(t(f"(d) 관성이 크면 지연이 크다 (명령 {freq:g} Hz)",
                     f"(d) Larger inertia, larger lag (command {freq:g} Hz)"), fontsize=11)
    ax_d.grid(alpha=0.3, which="both")
    ax_d.legend(fontsize=8)

    fig.suptitle(
        t(f"W1-M2 · G1 sin파 구동 — 관절군 '{group}' (lesson §5.5)",
          f"W1-M2 · G1 sine drive — joint group '{group}' (lesson 5.5)"),
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# %% [markdown]
# ## 6. 실행

# %%
def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        if get_ipython() is not None:
            return True
    except Exception:
        pass
    argv0 = Path(sys.argv[0]).name.lower() if sys.argv else ""
    return argv0 == "" or "ipykernel" in argv0 or "jupyter" in argv0 or "colab" in argv0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W1-M2 실습 3: G1에 sin파 목표각 넣기 (lesson §5.4 · §5.5 · §6)")
    p.add_argument("--menagerie", default=None, help="mujoco_menagerie 경로")
    p.add_argument("--joints", default="arms", choices=["arms", "legs", "waist", "all"],
                   help="구동할 관절군 (기본 arms)")
    p.add_argument("--freq", type=float, default=0.5, help="sin 주파수 [Hz] (기본 0.5)")
    p.add_argument("--amp", type=float, default=0.2, help="sin 진폭 [rad] (기본 0.2 ≈ 11.5°)")
    p.add_argument("--seconds", type=float, default=6.0, help="시뮬 길이 [s] (기본 6 = 0.5Hz 명령의 3주기)")
    p.add_argument("--fps", type=int, default=30, help="mp4 fps (기본 30)")
    p.add_argument("--height", type=int, default=480, help="렌더 높이 [px] (기본 480)")
    p.add_argument("--width", type=int, default=640, help="렌더 너비 [px] (기본 640)")
    p.add_argument("--no-video", action="store_true", help="렌더를 건너뛰고 그래프만 (가장 빠름)")
    p.add_argument("--smoke", action="store_true",
                   help="1.5초 · 15fps · 320x240으로 짧게 — 먼저 이걸로 경로 확인")
    p.add_argument("--ascii-labels", action="store_true", help="한글 폰트가 있어도 영문 라벨로 렌더")
    if argv is None:
        argv = [] if _in_notebook() else sys.argv[1:]
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seconds = 1.5 if args.smoke else args.seconds
    fps = 15 if args.smoke else args.fps
    height = 240 if args.smoke else args.height
    width = 320 if args.smoke else args.width
    suffix = "_ascii" if args.ascii_labels else ""

    setup_korean_font(force_ascii=args.ascii_labels)
    menagerie = resolve_menagerie(args.menagerie)
    scene = menagerie / "unitree_g1" / "scene.xml"  # g1.xml이 아니라 scene.xml (lesson §5.1)

    print("=" * 92)
    print(f" W1-M2 실습 3 — G1 sin파 구동 {'(smoke)' if args.smoke else ''}")
    print(f" scene: {scene}")
    print(f" MUJOCO_GL={os.environ.get('MUJOCO_GL')}  ·  mujoco {mujoco.__version__}")
    print("=" * 92)

    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)

    # 키프레임은 인덱스가 아니라 이름으로 (lesson §3.5의 함정)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
    assert kid >= 0, "'stand' 키프레임이 없습니다 — scene_mjx.xml을 로드한 것은 아닌지 확인"
    mujoco.mj_resetDataKeyframe(m, d, kid)
    mujoco.mj_forward(m, d)

    q_stand = d.qpos[7:7 + m.nu].copy()  # 목표각의 기준점 = 서 있는 자세
    d.ctrl[:] = q_stand  # 나머지 관절은 그 자리를 지키게 (안 주면 ctrl=0 -> 전 관절 0도로 접힌다)

    ctrl_ids = select_ctrl_ids(m, args.joints)
    names = actuator_names(m)

    print(f"\n=== [1] 설정 ===")
    print(f"  키프레임 'stand' (id={kid}) → 골반 높이 {d.qpos[2]:.3f} m")
    print(f"  구동 관절군 '{args.joints}': {len(ctrl_ids)}개 / 전체 {m.nu}개")
    print(f"    ctrl id {ctrl_ids[0]}~{ctrl_ids[-1]} : {names[ctrl_ids[0]]} ... {names[ctrl_ids[-1]]}")
    print(f"    (lesson §5.4: ctrl id i ↔ qpos adr i+7 ↔ qvel adr i+6 ↔ jnt id i+1)")
    print(f"  명령: ctrl_i(t) = q_stand_i + {args.amp:g}*sin(2π*{args.freq:g}*t), ctrlrange로 클립  [eq.(2)]")
    print(f"  ⚠️ ctrl은 토크가 아니라 목표 관절각[rad]입니다 (lesson §5.5).")
    print(f"  시뮬 {seconds:g}s @ dt={m.opt.timestep:g}s = {int(seconds / m.opt.timestep):,} 스텝"
          f"{'' if args.no_video else f', 렌더 {fps}fps {height}x{width}'}")

    # ---- 롤아웃 ----
    print("\n=== [2] 롤아웃 ===")
    res = rollout(m, d, ctrl_ids, q_stand, seconds, args.amp, args.freq,
                  fps, height, width, render=not args.no_video)
    print(f"  {res['n_steps']:,} 스텝 + 프레임 {len(res['frames'])}장 → {res['wall']:.1f}s "
          f"(실시간 대비 {seconds / max(res['wall'], 1e-9):.2f}x)")
    print(f"  eq.(1) 잔차 max|kp(ctrl-q) - kv*qdot - actuator_force| = {res['eq1_residual']:.3e}")
    assert res["eq1_residual"] < 1e-6, "eq.(1)이 안 맞는다 — position 액추에이터가 아닐 수 있다"
    print("  -> 0이면 'ctrl은 목표 관절각'이 코드 수준에서 확인된 것 (assert 통과).")
    print(f"  ctrlrange 클립 발생: {res['n_clipped']:,}회"
          + ("  (--amp를 키우면 늘어납니다)" if res["n_clipped"] == 0 else "  ← 명령이 관절 한계를 넘었습니다"))

    # ---- 추종 분석 ----
    rep = tracking_report(m, res, ctrl_ids, args.freq, args.amp)
    print_tracking_table(rep, args.freq)
    if res["fall_time"] is not None:
        print(f"  ⚠️ t={res['fall_time']:.2f}s 에 넘어졌습니다. 넘어진 뒤에도 관절은 목표각을 계속 추종하지만,")
        print("     자세와 접촉이 완전히 달라져 M(q)가 변하므로 위 추종 지표는 '서 있을 때의 값'이 아닙니다.")

    # ---- 넘어졌는가 ----
    print("\n=== [4] 균형 ===")
    print(f"  골반 높이: 시작 0.790 m → 끝 {res['pelvis_z'][-1]:.3f} m (최저 {res['pelvis_z'].min():.3f} m)")
    if res["fall_time"] is not None:
        print(f"  ❗ t={res['fall_time']:.2f}s 에 넘어졌습니다 (골반 < {FALL_HEIGHT} m).")
        print("     당연한 결과입니다. floating base 로봇을 붙잡아주는 것은 아무것도 없고,")
        print("     각 관절은 자기 목표각만 지킬 뿐 '넘어지지 않는다'는 목적을 모릅니다.")
        print("     전신의 균형을 목적함수에 넣어 푸는 것이 WBC이고, 그것을 학습으로 얻는 것이 W3-M1입니다.")
        print("     (회사 스택으로는 L2의 GEAR-SONIC / HOMIE 계층)")
    else:
        print("  ✅ 넘어지지 않았습니다. 다리를 흔들면 어떻게 되는지 꼭 해보세요:")
        print("     python 03_g1_sin_wave.py --joints legs")
        print("     PD 서보는 관절각만 지킬 뿐 균형을 모릅니다. 그것이 W3에서 RL 보행 정책이 필요한 이유입니다.")

    # ---- 저장 ----
    out_dir = artifacts_dir()
    fig_path = plot_tracking(res, rep, args.freq, args.joints,
                             out_dir / f"03_tracking_{args.joints}{suffix}.png")
    print(f"\n[저장] {fig_path}")
    if res["frames"]:
        mp4 = write_mp4(out_dir / f"g1_sin_{args.joints}.mp4", res["frames"], fps)
        print(f"[저장] {mp4}   ({len(res['frames'])}프레임 @ {fps}fps)")
        print("  mp4는 gitignore 대상입니다. 인스턴스에서 내려받거나 W&B에 올려서 보세요.")
    else:
        print("[skip] --no-video 지정 → mp4 없음")

    print("\n" + "-" * 92)
    print(" 다음: 04_playground_smoke.py — W3에서 쓸 RL 환경이 도는지만 확인")
    print(" 종료 시 EGL traceback('Exception ignored in:')은 무해합니다 (lesson §6.2).")
    print("-" * 92)


# %%
if __name__ == "__main__":
    main()
