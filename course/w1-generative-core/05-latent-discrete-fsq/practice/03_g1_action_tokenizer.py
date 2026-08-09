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
# # W1-M5 실습 3 — 액션 토크나이저로 쓰기: G1 궤적 → 정수 열 → 궤적
#
# **"이게 왜 회사 스택인가"를 체감하는 스크립트입니다.**
#
# lesson §7.1 그림에서 L3가 하는 일이 정확히 이것입니다 — 액션 의도를 **정수 하나**로 바꿔
# 하위 WBC에 넘기고, 하위가 그것을 연속 궤적으로 되돌립니다. 여기서는 상위 모델 대신 우리가
# 직접 궤적을 넣어 인코더·디코더만 돌립니다.
#
# 확인할 것:
#
# 1. **토큰 열** — 궤적이 `[413, 87, 902, ...]` 같은 정수 열이 됩니다. **이게 LLM에 그대로 들어갈
#    수 있는 형태**라는 것이 요점입니다(lesson §1.1 이유 ①·②).
# 2. **복원 오차를 라디안·도 단위로, 관절별로** — "이 정밀도로 로봇을 제어할 수 있는가"를
#    직접 판단하게. 관절마다 `jnt_range` 폭이 11배까지 다르므로 같은 정규화 오차가 다른 각도가 됩니다.
# 3. **압축률** — lesson §6과 퀴즈 9번의 계산을 코드가 재현합니다.
#    $$\frac{T \cdot D \cdot 32}{n_{tok}\log_2 |\mathcal{C}|}
#      = \frac{16 \cdot 29 \cdot 32}{\log_2 1000} = \frac{14{,}848}{9.97} \approx 1{,}489 \tag{8}$$
# 4. **원본 vs 복원 오버레이 그래프**와, 복원 궤적을 MuJoCo G1에 다시 넣은 **mp4**.
#
# 입력: `02_fsq_vs_vq_g1.py`가 저장한 체크포인트. 없으면 **자동으로 짧게 재학습**합니다(GPU 약 15초).
#
# 출력(`artifacts/W1-M5/`):
# `03_tokens.txt` · `03_overlay.png` · `03_joint_error.png` · `03_token_recon.mp4`(gitignore)
#
# > ⚠️ **이 토크나이저는 장난감입니다.** 학습 데이터도 우리가 만든 합성 궤적이고, 구조도
# > MLP 오토인코더입니다. **회사 FSQ 모델이 무엇을 토큰화하는지는 lesson §7.4대로 미확인**입니다.

# %%
from __future__ import annotations

import os

# ⚠️ `import mujoco` 보다 먼저 (W1-M2 lesson §6.2). 뷰어는 쓰지 않는다.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless 고정

import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402

MODULE_ID = "W1-M5"
FLOAT_BITS = 32  # 원본을 float32로 셌을 때 (lesson §6 · 퀴즈 9번)


# %% [markdown]
# ## 0. 02의 코드를 그대로 재사용
#
# `FSQ` · `TrajAE` · 데이터 생성 · 정규화(eq.(6))는 `02_fsq_vs_vq_g1.py`에 있는 것을 **그대로**
# 불러 씁니다. 파일명이 숫자로 시작해 `import`가 안 되므로 `importlib`로 경로 로드합니다.
# 같은 폴더에서 실행하면 자동으로 찾습니다.

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


def load_m2():
    """02_fsq_vs_vq_g1.py 를 모듈로 로드한다."""
    for cand in (here(), find_repo_root() / "course" / "w1-generative-core"
                 / "05-latent-discrete-fsq" / "practice"):
        p = cand / "02_fsq_vs_vq_g1.py"
        if p.is_file():
            spec = importlib.util.spec_from_file_location("fsq_vs_vq_g1", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["fsq_vs_vq_g1"] = mod   # dataclass가 __module__을 되찾을 수 있게
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("[에러] 02_fsq_vs_vq_g1.py 를 찾지 못했습니다 (같은 폴더에 있어야 합니다)")


M2 = load_m2()
FSQ, TrajAE = M2.FSQ, M2.TrajAE
normalize, denormalize = M2.normalize, M2.denormalize
N_JOINTS, FS_HZ = M2.N_JOINTS, M2.FS_HZ


# %%
_KO_FONT_PREFERENCE = (
    "Pretendard", "NanumGothic", "Nanum Gothic", "Malgun Gothic", "NanumBarunGothic",
    "Noto Sans KR", "Noto Sans CJK KR", "Noto Sans CJK JP", "Source Han Sans KR",
    "AppleGothic", "Spoqa Han Sans Neo", "UnDotum", "Baekmuk Gulim",
)
_PROBE_CHARS = "한글토큰복원"

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
    """imageio(+imageio-ffmpeg) 우선, 실패하면 mediapy 폴백 (W1-M2와 동일)."""
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
# ## 1. 모델 확보 — 체크포인트를 읽거나, 없으면 짧게 학습
#
# `02`를 먼저 돌렸으면 체크포인트를 그대로 씁니다. 없으면 같은 구성으로 짧게 학습합니다
# (GPU 약 15초). **02와 03의 결과 숫자를 비교하려면 반드시 `02`를 먼저 돌리세요.**

# %%
def build_model(levels: list[int], chunk: int, hidden: int, n_tok: int, dev: torch.device) -> TrajAE:
    quant = FSQ(levels)
    return TrajAE(chunk * N_JOINTS, hidden, quant.d, quant, n_tok=n_tok).to(dev)


def quick_train(model: TrajAE, Xt: torch.Tensor, chunk: int, epochs: int, batch: int,
                lr: float, seed: int) -> float:
    """02와 같은 손실(재구성 한 항)로 짧게 학습한다. FSQ는 보조 손실이 없다."""
    torch.manual_seed(seed)
    dev = Xt.device
    tid, st = (torch.as_tensor(a, device=dev)
               for a in M2.chunk_index(Xt.shape[0], Xt.shape[1], chunk, chunk // 2))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = tid.numel()
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        steps = max(1, n // batch)
        for s in range(steps):
            sel = perm[s * batch:(s + 1) * batch]
            x = M2.gather_chunks(Xt, tid[sel], st[sel], chunk)
            xh, aux, _ = model(x)
            loss = F.mse_loss(xh, x) + aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        if ep % max(1, epochs // 4) == 0:
            print(f"     ep {ep:3d}/{epochs}  train_loss={tot / steps:.5f}")
    return time.perf_counter() - t0


def get_model(args, spec, dev: torch.device) -> tuple[TrajAE, dict]:
    levels = [int(v) for v in args.levels.split(",")]
    ck_path = Path(args.ckpt) if args.ckpt else \
        artifacts_dir() / f"fsq_g1_L{'-'.join(map(str, levels))}_T{args.chunk}.pt"

    if ck_path.is_file() and not args.retrain:
        blob = torch.load(ck_path, map_location=dev, weights_only=False)
        levels = list(blob["levels"])
        model = build_model(levels, blob["chunk"], blob["hidden"], blob.get("n_tok", 1), dev)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        print(f"  체크포인트 로드: {ck_path}")
        print(f"    levels={levels}  chunk={blob['chunk']}  청크당 토큰={blob.get('n_tok', 1)}  "
              f"02에서의 사용률={blob.get('usage', float('nan')) * 100:.1f}%")
        return model, dict(levels=levels, chunk=blob["chunk"], hidden=blob["hidden"],
                           n_tok=blob.get("n_tok", 1), synergies=blob.get("synergies", 6),
                           source=str(ck_path))

    print(f"  체크포인트가 없어 짧게 학습합니다 ({args.train_epochs} epoch). "
          f"02를 먼저 돌리면 이 단계를 건너뜁니다.")
    n_traj = 400 if args.smoke else 2000
    Xtr = M2.make_trajectories(n_traj, args.traj_len, spec, seed=args.seed, n_syn=args.synergies)
    Xt = torch.as_tensor(Xtr, device=dev)
    model = build_model(levels, args.chunk, args.hidden, args.tokens_per_chunk, dev)
    sec = quick_train(model, Xt, args.chunk, args.train_epochs, args.batch, args.lr, args.seed)
    model.eval()
    print(f"    학습 {sec:.0f}s")
    return model, dict(levels=levels, chunk=args.chunk, hidden=args.hidden,
                       n_tok=args.tokens_per_chunk, synergies=args.synergies, source="즉석 학습")


# %% [markdown]
# ## 2. 토큰화 — 궤적을 정수 열로
#
# ```
# q [T_traj, 29] rad ─eq.(6)→ x [T_traj, 29] ∈[-1,1]
#                              ↓ 청크 분할 (겹치지 않게)
#                             x [n_chunk, 16, 29]
#                              ↓ Enc → z → round(f(z)) → eq.(3)
#                          tokens [n_chunk, n_tok]  ← **이 정수 열이 인터페이스다**
#                              ↓ eq.(3)의 역 → Dec
#                          x̂ [n_chunk, 16, 29] ─eq.(6)⁻¹→ q̂ [T_traj, 29] rad
# ```
#
# **토큰 → 궤적 방향은 인코더를 쓰지 않습니다.** `indices_to_codes(idx)` 로 코드를 복원한 뒤
# 디코더만 통과시킵니다. 상위 LM이 토큰을 뱉었을 때 하위가 하는 일이 정확히 이 경로입니다.

# %%
@torch.no_grad()
def tokenize(model: TrajAE, x: torch.Tensor, chunk: int) -> torch.Tensor:
    """x [T_traj, 29] (정규화) -> 토큰 [n_chunk, n_tok]."""
    n = x.shape[0] // chunk
    xc = x[: n * chunk].reshape(n, chunk, -1)
    return model.encode_indices(xc)


@torch.no_grad()
def detokenize(model: TrajAE, tokens: torch.Tensor, chunk: int) -> torch.Tensor:
    """토큰 [n_chunk, n_tok] -> x̂ [n_chunk*chunk, 29]. **인코더를 쓰지 않는 경로.**"""
    zq = model.quantizer.indices_to_codes(tokens)              # eq.(3)의 역
    xh = model.dec(zq.reshape(tokens.shape[0], -1))
    return xh.reshape(-1, N_JOINTS)


# %% [markdown]
# ## 3. 압축률 — lesson §6 · 퀴즈 9번과 같은 계산
#
# $$
# \text{원본} = T \cdot D \cdot 32 \text{ bits}, \qquad
# \text{토큰} = n_{tok} \cdot \log_2 |\mathcal{C}| \text{ bits}
# \tag{8}
# $$
#
# 기본 설정($T{=}16$, $D{=}29$, $|\mathcal{C}|{=}1000$, $n_{tok}{=}1$)이면
# $14{,}848 / 9.97 \approx 1{,}489$배입니다. lesson 퀴즈 9번의 답과 같은 숫자가 나와야 합니다.
#
# 실제 저장은 비트를 쪼개 쓰지 않으므로 $\lceil 9.97 \rceil = 10$ bits로 잡는 것이 현실적입니다 —
# 두 값을 모두 출력합니다.

# %%
def compression_report(chunk: int, n_joints: int, cb_size: int, n_tok: int) -> dict:
    raw_bits = chunk * n_joints * FLOAT_BITS                       # eq.(8) 분자
    bits_ideal = n_tok * math.log2(cb_size)                        # eq.(8) 분모 (이론)
    bits_stored = n_tok * math.ceil(math.log2(cb_size))            # 실제 저장
    return dict(
        raw_bits=raw_bits, bits_ideal=bits_ideal, bits_stored=bits_stored,
        ratio_ideal=raw_bits / bits_ideal, ratio_stored=raw_bits / bits_stored,
        bits_per_second_ideal=bits_ideal * FS_HZ / chunk,
    )


# %% [markdown]
# ## 4. 그림 · 영상
#
# `03_token_recon.mp4`는 **왼쪽 원본 / 오른쪽 복원**을 나란히 붙인 영상입니다.
#
# > 📌 **물리를 끄고 관절각만 재생합니다**(`mj_forward`). 복원된 29관절 명령을 물리와 함께
# > PD 서보에 그대로 먹이면 **로봇은 넘어집니다** — W1-M2 `03_g1_sin_wave.py --joints legs`에서 본
# > 그대로입니다. 그건 토크나이저 탓이 아니라 **균형을 지킬 WBC가 없기 때문**이고(lesson 「흔한 오해」 3번),
# > 그 WBC를 학습으로 얻는 것이 W3-M1입니다. `--physics`를 주면 그 장면도 직접 볼 수 있습니다.

# %%
def plot_overlay(q_ref: np.ndarray, q_hat: np.ndarray, names: list[str], sel: list[int],
                 tokens: np.ndarray, chunk: int, out: Path) -> Path:
    tt = np.arange(q_ref.shape[0]) / FS_HZ
    n = len(sel)
    fig, axes = plt.subplots(n, 1, figsize=(11.5, 2.1 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, j in zip(axes, sel):
        ax.plot(tt, np.degrees(q_ref[:, j]), lw=1.8, color="#333333",
                label=t("원본", "original"))
        ax.plot(tt, np.degrees(q_hat[:, j]), lw=1.8, color="#d62728", ls="--",
                label=t("토큰 복원", "token recon"))
        for b in range(0, q_ref.shape[0] + 1, chunk):   # 청크 경계
            ax.axvline(b / FS_HZ, color="#cccccc", lw=0.6, ls=":")
        ax.set_ylabel(t("각도 [deg]", "angle [deg]"))
        err = np.degrees(np.sqrt(np.mean((q_ref[:, j] - q_hat[:, j]) ** 2)))
        ax.set_title(f"{names[j]}   RMSE {err:.2f}°", fontsize=9, loc="left")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2, loc="upper right")
    axes[-1].set_xlabel(t("시간 [s]  (점선 = 청크 경계, 청크마다 토큰 1개)",
                          "time [s]  (dotted = chunk boundary)"))
    head = ", ".join(str(int(v)) for v in tokens.reshape(-1)[:12])
    fig.suptitle(t(f"원본 vs 토큰 복원 — 토큰 열 [{head}, ...]",
                   f"original vs token reconstruction — tokens [{head}, ...]"))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_joint_error(rmse_deg: np.ndarray, names: list[str], span_deg: np.ndarray, out: Path) -> Path:
    order = np.argsort(-rmse_deg)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))
    y = np.arange(len(order))
    axes[0].barh(y, rmse_deg[order], color="#d62728")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([names[i].replace("_joint", "") for i in order], fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel(t("복원 RMSE [deg]", "recon RMSE [deg]"))
    axes[0].grid(alpha=0.25, axis="x")
    axes[0].set_title(t("(a) 관절별 복원 오차 — 큰 순서",
                        "(a) per-joint reconstruction error"))

    axes[1].scatter(span_deg, rmse_deg, s=40, color="#1f77b4")
    for i in order[:5]:
        axes[1].annotate(names[i].replace("_joint", ""), (span_deg[i], rmse_deg[i]),
                         textcoords="offset points", xytext=(5, 3), fontsize=7)
    axes[1].set_xlabel(t("관절 가동범위 hi-lo [deg]", "joint range hi-lo [deg]"))
    axes[1].set_ylabel(t("복원 RMSE [deg]", "recon RMSE [deg]"))
    axes[1].grid(alpha=0.25)
    axes[1].set_title(t("(b) 오차는 가동범위에 비례한다 — 정규화 공간에서 일하기 때문",
                        "(b) error scales with joint range"))
    fig.suptitle(t("같은 정규화 오차라도 관절마다 각도가 다르다 (eq.(6))",
                   "same normalized error maps to different angles"))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def render_side_by_side(m: mujoco.MjModel, q_ref: np.ndarray, q_hat: np.ndarray,
                        out: Path, fps: int, height: int, width: int,
                        physics: bool = False) -> Path:
    """왼쪽 원본 / 오른쪽 복원. 기본은 mj_forward 재생(물리 없음). 뷰어는 쓰지 않는다."""
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance, cam.elevation, cam.azimuth = 2.4, -8.0, 135.0
    cam.lookat[:] = [0.0, 0.0, 0.75]
    step = max(1, int(round(FS_HZ / fps)))
    frames: list[np.ndarray] = []
    sep = np.full((height, 4, 3), 255, dtype=np.uint8)

    def playback_kinematic(q_seq: np.ndarray) -> list[np.ndarray]:
        outs = []
        for k in range(0, q_seq.shape[0], step):
            mujoco.mj_resetDataKeyframe(m, d, 0) if m.nkey > 0 else mujoco.mj_resetData(m, d)
            d.qpos[7:7 + m.nu] = q_seq[k]
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera=cam)
            outs.append(renderer.render().copy())
        return outs

    def playback_physics(q_seq: np.ndarray) -> list[np.ndarray]:
        """PD 서보에 목표각을 먹이고 실제로 물리를 돈다. 다리 명령이 크면 넘어진다."""
        mujoco.mj_resetDataKeyframe(m, d, 0) if m.nkey > 0 else mujoco.mj_resetData(m, d)
        n_sub = max(1, int(round((1.0 / FS_HZ) / m.opt.timestep)))
        outs = []
        for k in range(q_seq.shape[0]):
            d.ctrl[:] = np.clip(q_seq[k], m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
            for _ in range(n_sub):
                mujoco.mj_step(m, d)
            if k % step == 0:
                renderer.update_scene(d, camera=cam)
                outs.append(renderer.render().copy())
        return outs

    play = playback_physics if physics else playback_kinematic
    try:
        left = play(q_ref)
        right = play(q_hat)
    finally:
        renderer.close()
    for a, b in zip(left, right):
        frames.append(np.concatenate([a, sep, b], axis=1))
    return write_mp4(out, frames, fps)


# %% [markdown]
# ## 5. 실행

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W1-M5 실습 3: FSQ 액션 토크나이저 (lesson §6·§7)")
    p.add_argument("--menagerie", default=None, help="mujoco_menagerie 경로")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--ckpt", default=None, help="체크포인트 경로 (기본: artifacts/W1-M5/fsq_g1_*.pt)")
    p.add_argument("--retrain", action="store_true", help="체크포인트가 있어도 새로 학습")
    p.add_argument("--levels", default="8,5,5,5")
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--tokens-per-chunk", type=int, default=1)
    p.add_argument("--synergies", type=int, default=6)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--traj-len", type=int, default=200)
    p.add_argument("--demo-len", type=int, default=480,
                   help="시연 궤적 길이 [프레임] (chunk의 배수 권장, 기본 480 = 9.6초 @50Hz)")
    p.add_argument("--demo-seed", type=int, default=789,
                   help="시연 궤적 시드. 바꾸면 다른 동작이 나옵니다 — 좌우대칭 동작은 "
                        "토큰 열도 회문(palindrome)이 됩니다")
    p.add_argument("--train-epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--show-tokens", type=int, default=40, help="stdout에 찍을 토큰 개수")
    p.add_argument("--joints", default="left_shoulder_pitch,left_elbow,right_knee,waist_yaw",
                   help="오버레이에 그릴 관절 (이름 일부, 쉼표 구분)")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--height", type=int, default=400)
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--physics", action="store_true",
                   help="mj_forward 재생 대신 PD 서보 + 물리로 돌린다 (넘어지는 것이 정상)")
    p.add_argument("--no-video", action="store_true", help="mp4를 만들지 않는다 (가장 빠름)")
    p.add_argument("--smoke", action="store_true", help="축소 경로 (경로 확인용)")
    p.add_argument("--ascii-labels", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.smoke:
        args.train_epochs, args.demo_len, args.fps = 4, 64, 12
        print("[smoke] 축소 경로로 실행합니다 (학습 4 epoch, 시연 64프레임).")
    setup_korean_font(args.ascii_labels)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = M2.pick_device(args.device)
    out_dir = artifacts_dir()

    print("=" * 82)
    print(f"  {MODULE_ID} 실습 3 — G1 액션 토크나이저: 궤적 → 정수 열 → 궤적")
    print("=" * 82)

    # --- [1] 모델 ------------------------------------------------------------
    menagerie = M2.resolve_menagerie(args.menagerie)
    m, spec = M2.load_g1_spec(menagerie)
    print("\n=== [1] 모델 확보 ===")
    model, cfg = get_model(args, spec, dev)
    chunk, n_tok = cfg["chunk"], cfg["n_tok"]
    cb_size = model.quantizer.codebook_size
    print(f"  |C| = ∏L_i = {cb_size:,}   d = {model.quantizer.d}   "
          f"양자화기 학습 파라미터 {sum(p.numel() for p in model.quantizer.parameters())}개   "
          f"출처: {cfg['source']}")

    # --- [2] 시연 궤적 → 토큰 -------------------------------------------------
    print("\n=== [2] 토큰화 ===")
    demo_len = (args.demo_len // chunk) * chunk
    Xd = M2.make_trajectories(1, demo_len, spec, seed=args.demo_seed, n_syn=cfg["synergies"])[0]
    x = torch.as_tensor(Xd, device=dev)
    tokens = tokenize(model, x, chunk)                       # [n_chunk, n_tok]
    xh = detokenize(model, tokens, chunk)                    # 인코더를 쓰지 않는 경로

    q_ref = denormalize(Xd.astype(np.float64), spec.lo, spec.hi)                    # eq.(6)
    q_hat = denormalize(xh.cpu().numpy().astype(np.float64), spec.lo, spec.hi)
    q_ref = q_ref[: q_hat.shape[0]]

    tok_flat = tokens.reshape(-1).cpu().numpy()
    print(f"  궤적 {demo_len}프레임 ({demo_len / FS_HZ:.2f}s @ {FS_HZ:.0f}Hz) → "
          f"청크 {tokens.shape[0]}개 × 토큰 {n_tok}개 = **정수 {tok_flat.size}개**")
    shown = tok_flat[: args.show_tokens]
    print("\n  토큰 열 (이게 LLM vocabulary에 그대로 들어갈 수 있는 형태입니다):")
    for i in range(0, len(shown), 10):
        print("    " + ", ".join(f"{int(v):4d}" for v in shown[i:i + 10]))
    if tok_flat.size > len(shown):
        print(f"    ... (총 {tok_flat.size}개)")
    print(f"  토큰 ID 범위: [{int(tok_flat.min())}, {int(tok_flat.max())}]  ⊂  [0, {cb_size - 1}]")
    print(f"  서로 다른 토큰 {len(np.unique(tok_flat))}종 / 이 궤적에서")
    if np.array_equal(tok_flat, tok_flat[::-1]):
        print("  📌 토큰 열이 회문(palindrome)입니다 — 시연 궤적이 '갔다 돌아오는' 대칭 동작이라서입니다.")
        print("     토큰 열이 동작의 구조를 그대로 반영한다는 증거입니다. --demo-seed 를 바꿔보세요.")
    print("  📌 인접 청크가 같은 토큰을 반복하는 구간이 보입니다. 동작이 느릴 때 나타나며,")
    print("     상위 모델이 '토큰을 유지'하는 것과 하위가 '마지막 토큰을 홀드'하는 것이")
    print("     같은 결과를 낸다는 뜻입니다 (lesson §7.2의 sequenceDiagram).")

    tok_path = out_dir / "03_tokens.txt"
    with open(tok_path, "w", encoding="utf-8") as f:
        f.write(f"# W1-M5 03_g1_action_tokenizer.py — G1 궤적 {demo_len}프레임의 FSQ 토큰\n")
        f.write(f"# levels={cfg['levels']}  |C|={cb_size}  chunk={chunk}  tokens_per_chunk={n_tok}\n")
        f.write(" ".join(str(int(v)) for v in tok_flat) + "\n")
    print(f"  [저장] {tok_path}")

    # --- [3] 왕복 검증 --------------------------------------------------------
    tok2 = tokenize(model, torch.as_tensor(
        normalize(q_hat, spec.lo, spec.hi).astype(np.float32), device=dev), chunk)
    same = int((tok2.reshape(-1) == tokens.reshape(-1)).sum())
    print(f"\n  [참고] 복원 궤적을 다시 토큰화하면 {same}/{tok_flat.size}개가 같은 토큰입니다.")
    print("         (오토인코더가 멱등일 이유는 없습니다 — 100%가 아니어도 정상입니다)")

    # --- [4] 복원 오차 --------------------------------------------------------
    print("\n=== [3] 복원 오차 — 이 정밀도로 로봇을 제어할 수 있는가? ===")
    err = q_ref - q_hat
    rmse_rad = np.sqrt((err ** 2).mean(axis=0))
    rmse_deg = np.degrees(rmse_rad)
    maxe_deg = np.degrees(np.abs(err).max(axis=0))
    span_deg = np.degrees(spec.hi - spec.lo)
    order = np.argsort(-rmse_deg)
    rows = [[spec.names[i].replace("_joint", ""), f"{span_deg[i]:.0f}",
             f"{rmse_rad[i]:.4f}", f"{rmse_deg[i]:.2f}", f"{maxe_deg[i]:.2f}",
             f"{rmse_deg[i] / span_deg[i] * 100:.1f}%"] for i in order[:8]]
    rows.append(["… (하위 21개 생략)", "", "", "", "", ""])
    for i in order[-3:]:
        rows.append([spec.names[i].replace("_joint", ""), f"{span_deg[i]:.0f}",
                     f"{rmse_rad[i]:.4f}", f"{rmse_deg[i]:.2f}", f"{maxe_deg[i]:.2f}",
                     f"{rmse_deg[i] / span_deg[i] * 100:.1f}%"])
    print_table(["관절", "range[deg]", "RMSE[rad]", "RMSE[deg]", "max err[deg]", "RMSE/range"],
                rows, ["left", "right", "right", "right", "right", "right"])
    print(f"  전체 평균 RMSE = {rmse_rad.mean():.4f} rad = {rmse_deg.mean():.2f}°   "
          f"최대 관절 = {rmse_deg.max():.2f}° ({spec.names[int(order[0])]})")
    print("  → 판단은 학습자 몫입니다. 위치 제어 로봇의 반복 정밀도가 보통 0.1° 수준이라는 것과")
    print("    비교하면, **이 토큰 하나로는 관절 명령을 직접 만들 수 없다**는 결론이 나옵니다.")
    print("    lesson 「흔한 오해」 3번가 말하는 것이 이것입니다 — 토큰은 **의도의 해상도**이지")
    print("    관절 명령의 해상도가 아닙니다. 관절 명령은 하위 WBC가 50~500 Hz로 만듭니다.")

    # --- [5] 압축률 ----------------------------------------------------------
    print("\n=== [4] 압축률 (lesson §6 · 퀴즈 9번) ===")
    c = compression_report(chunk, N_JOINTS, cb_size, n_tok)
    print_table(
        ["항목", "값"],
        [["원본 (float32)", f"{chunk} x {N_JOINTS} x {FLOAT_BITS} = {c['raw_bits']:,} bits"],
         ["토큰 (이론)", f"{n_tok} x log2({cb_size:,}) = {c['bits_ideal']:.2f} bits"],
         ["토큰 (실제 저장)", f"{n_tok} x {math.ceil(math.log2(cb_size))} = {c['bits_stored']} bits"],
         ["압축률 (이론)", f"{c['ratio_ideal']:.1f} 배"],
         ["압축률 (저장 기준)", f"{c['ratio_stored']:.1f} 배"],
         ["상위→하위 대역폭", f"{c['bits_per_second_ideal']:.1f} bits/s "
                              f"({FS_HZ / chunk:.2f} Hz x {c['bits_ideal']:.2f} bits)"]],
        ["left", "left"],
    )
    if chunk == 16 and cb_size == 1000 and n_tok == 1:
        assert abs(c["raw_bits"] - 14848) < 1e-9 and abs(c["ratio_ideal"] - 1489.9) < 0.5, \
            "lesson 퀴즈 9번의 계산과 어긋납니다"
        print("  ✓ lesson 퀴즈 9번의 14,848 bits / 9.97 bits / 약 1,489배와 일치합니다.")
    print("  → 대역폭이 사실상 공짜가 되는 대신 그만큼 버렸습니다. 무엇을 버려도 되는지를")
    print("    정하는 것이 토크나이저 설계이고, 그래서 학습 데이터 분포가 결정적입니다.")

    # --- [6] 그림·영상 --------------------------------------------------------
    print("\n=== [5] 그림 · 영상 ===")
    keys = [k.strip() for k in args.joints.split(",") if k.strip()]
    sel = []
    for k in keys:
        hit = [i for i, nm in enumerate(spec.names) if k in nm]
        if hit:
            sel.append(hit[0])
    sel = sel or list(order[:3])
    print(f"  [저장] {plot_overlay(q_ref, q_hat, spec.names, sel, tok_flat, chunk, out_dir / '03_overlay.png')}")
    print(f"  [저장] {plot_joint_error(rmse_deg, spec.names, span_deg, out_dir / '03_joint_error.png')}")

    if args.no_video:
        print("  [skip] --no-video 지정 → mp4 없음")
    else:
        mode = "physics" if args.physics else "kinematic"
        mp4 = render_side_by_side(m, q_ref, q_hat, out_dir / f"03_token_recon_{mode}.mp4",
                                  args.fps, args.height, args.width, physics=args.physics)
        print(f"  [저장] {mp4}   (왼쪽 원본 / 오른쪽 토큰 복원, mode={mode})")
        print("         mp4는 gitignore 대상입니다.")
        if not args.physics:
            print("         --physics 를 주면 PD 서보 + 물리로 돌립니다. **넘어지는 것이 정상**이며,")
            print("         그것이 'WBC 없이 관절 명령만으로는 안 된다'의 실물 증거입니다 (W1-M2 §03).")

    # --- [7] 한계 ------------------------------------------------------------
    print("\n" + "=" * 82)
    print("  한계 — 정직하게")
    print("=" * 82)
    print("  · 이 토크나이저는 **장난감**입니다. 학습 데이터가 합성 궤적이고 구조도 MLP AE입니다.")
    print("  · 회사 FSQ 모델이 무엇을 토큰화하는지(관절각? 속도? latent motion?), 레벨 L과 d가")
    print("    무엇인지, 토큰 레이트가 얼마인지는 **집필 시점에 확인되지 않았습니다** (lesson §7.4).")
    print("    notes/questions-for-team.md 의 W1-M5 질문 6개가 그 목록입니다.")
    print("  · 여기서 본 복원 오차를 회사 시스템의 성능으로 읽으면 안 됩니다.")
    print("  · 다음 → W2-M5에서 이 토큰을 액션 표현 스펙트럼 전체 위에 놓고 비교합니다.")
    print("=" * 82)


# %%
if __name__ == "__main__":
    main(None if "ipykernel" not in sys.modules else [])
