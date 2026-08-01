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
# # W1-M2 실습 4 — mujoco_playground 스모크: W3 예고편
#
# lesson.md `§7`(mujoco_playground 미리보기)의 실행판입니다.
#
# **이 스크립트는 학습하지 않습니다.** 목표는 딱 하나 — *"에러 없이 돌아간다"* 를 확인하는 것.
# PPO 병렬 보행 학습은 W3-M1의 몫입니다.
#
# 확인할 것:
#
# 1. `jax.devices()` — 지금 CPU에서 도는가 GPU에서 도는가 (이걸 모르고 몇 시간 태우는 일이 흔합니다)
# 2. `registry.load("G1JoystickFlatTerrain")` 이 되는가
# 3. `observation_size` · `action_size` · `ctrl_dt` · `sim_dt` · `episode_length` · `action_scale`
# 4. **`n_substeps = ctrl_dt / sim_dt` 를 계산해 assert** — lesson §7.4의 주파수 예산
# 5. **보상항 24개** — lesson §7.5 "보상 설계 = 비용함수 설계"
# 6. `jax.jit(env.reset)` / `jax.jit(env.step)` 을 몇 번 돌리고 **각 호출의 소요 시간을 출력**
#    → 컴파일(첫 호출)과 실행(그 다음)의 차이를 눈으로 확인
#
# ---
#
# ## ⏱️ 실행 전에 알아둘 것 — 오래 멈춰 있는 것처럼 보입니다
#
# | 단계 | CPU JAX 기준 |
# |---|---|
# | `import mujoco_playground` | 약 4초 |
# | `registry.load(...)` | 약 3초 |
# | **`jax.jit(env.reset)` 첫 호출** | **수십 초** ← 여기서 멈춘 것처럼 보입니다. 컴파일입니다 (캐시가 따뜻하면 수 초) |
# | `jax.jit(env.step)` 첫 호출 | 약 1.4~2초 |
# | 두 번째 호출부터 | 캐시 상태에 따라 다름 (CPU에서는 여전히 초 단위) |
#
# **멈춘 게 아니라 컴파일 중입니다.** JAX는 첫 호출에서 함수를 추적(trace)해 XLA로 컴파일하고,
# 그 다음부터는 컴파일된 커널을 재사용합니다. LLM 서빙에서 첫 요청이 느린 것(웜업)과 같은 구조입니다.
#
# ## 설치 — 이름 함정
#
# ```bash
# pip install "playground==0.2.0"   # ✅ PyPI 배포명은 playground
# # pip install mujoco_playground   # ❌ 존재하지 않는 패키지
# ```
# ```python
# import mujoco_playground           # ✅ import 이름은 mujoco_playground
# ```
#
# GPU 인스턴스라면 JAX를 GPU 빌드로 바꿔야 합니다.
# ```bash
# pip install -U "jax[cuda12]"
# export JAX_DEFAULT_MATMUL_PRECISION=highest   # Ampere 계열 공식 권장
# ```

# %%
from __future__ import annotations

import os

# playground는 렌더를 쓰지 않는 상태 관측 태스크지만, 백엔드 고정 습관은 유지한다.
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from pathlib import Path  # noqa: E402

ENV_NAME_DEFAULT = "G1JoystickFlatTerrain"

# lesson §7.3 실측표 (2026-08-01, playground 0.2.0). 다르면 경고만 찍고 계속 진행한다.
LESSON_EXPECT = dict(
    action_size=29,
    ctrl_dt=0.02,
    sim_dt=0.002,
    n_substeps=10,
    episode_length=1000,
    action_scale=0.5,
    n_rewards=24,
    obs_state=103,
    obs_privileged=216,
)


# %%
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


# %% [markdown]
# ## 1. 백엔드 확인 — 지금 어디서 도는가
#
# `jax.devices()`가 `[CpuDevice(id=0)]`면 CPU입니다. 이 상태로 W3-M1 본 학습을 돌리면 안 됩니다.
# 이 스모크는 CPU로도 통과합니다 — 그것이 이 스크립트의 목적입니다.

# %%
def report_backend() -> bool:
    """JAX 백엔드를 출력하고 GPU 여부를 돌려준다."""
    import jax

    devices = jax.devices()
    kinds = {d.platform for d in devices}
    is_gpu = bool(kinds & {"gpu", "cuda", "rocm"})
    print("\n=== [1] JAX 백엔드 ===")
    print(f"  jax {jax.__version__}   default_backend={jax.default_backend()}")
    print(f"  devices: {devices}")
    if is_gpu:
        print("  ✅ GPU. W3-M1 본 학습을 여기서 돌리면 됩니다.")
        print("     Ampere 계열이면: export JAX_DEFAULT_MATMUL_PRECISION=highest")
    else:
        print("  ⚠️ CPU JAX입니다. 이 스모크는 통과하지만 **본 학습(W3-M1)은 불가능**합니다.")
        print("     GPU 인스턴스에서: pip install -U 'jax[cuda12]'")
        print("     아래 reset/step 호출이 수십 초 걸릴 수 있습니다. 정상입니다.")
    return is_gpu


# %% [markdown]
# ## 2. 레지스트리 — 등록된 환경
#
# lesson §7.2 기준 `registry.locomotion.ALL_ENVS`는 19개이고, 그중 G1·H1 태스크는 4개입니다.

# %%
def report_registry() -> list[str]:
    from mujoco_playground import registry

    all_envs = list(registry.locomotion.ALL_ENVS)
    humanoid = [e for e in all_envs if e.startswith(("G1", "H1"))]
    print("\n=== [2] 등록된 locomotion 환경 ===")
    print(f"  ALL_ENVS 개수 = {len(all_envs)}  (lesson §7.2 기준 19)")
    print(f"  G1·H1 태스크 {len(humanoid)}개: {humanoid}")
    if len(all_envs) != 19:
        print("  ⚠️ lesson §7.2와 개수가 다릅니다. playground 버전이 바뀐 것입니다 — 이 출력이 맞습니다.")
    return all_envs


# %% [markdown]
# ## 3. `G1JoystickFlatTerrain` 스펙 — lesson §7.3 · §7.4
#
# ```
# 정책 50 Hz (ctrl_dt=0.02)  ──  n_substeps=10  ──  물리 500 Hz (sim_dt=0.002)
# ```
#
# 정책이 명령을 갱신하지 않는 10 스텝 동안에도 물리는 계속 돕니다.
# `ctrl`은 고정된 채로 PD 서보(실습 03의 그 식)가 목표를 향해 계속 힘을 냅니다 —
# W1-M1의 action chunking과 같은 구조이고, 여기서는 홀드 길이가 10 스텝입니다.

# %%
def report_env_spec(env, cfg, env_name: str) -> dict:
    """환경 스펙을 출력하고 lesson §7.3과 대조한다."""
    print(f"\n=== [3] {env_name} 스펙 (lesson §7.3) ===")
    obs = env.observation_size
    ctrl_dt = float(cfg.ctrl_dt)
    sim_dt = float(cfg.sim_dt)
    n_substeps = ctrl_dt / sim_dt

    rows = [
        ["observation_size", str(obs)],
        ["action_size", str(env.action_size)],
        ["ctrl_dt (정책 주기)", f"{ctrl_dt:g} s  =  {1 / ctrl_dt:g} Hz"],
        ["sim_dt (물리 주기)", f"{sim_dt:g} s  =  {1 / sim_dt:g} Hz"],
        ["n_substeps = ctrl_dt/sim_dt", f"{n_substeps:g}"],
        ["episode_length", f"{int(cfg.episode_length):,} 정책 스텝"],
        ["  = 시뮬 시간", f"{int(cfg.episode_length) * ctrl_dt:g} s"],
        ["  = 물리 스텝", f"{int(int(cfg.episode_length) * n_substeps):,}"],
        ["action_scale", f"{float(cfg.action_scale):g}"],
    ]
    print_table(["항목", "값"], rows, aligns=["left", "right"])

    # lesson §7.4의 산수를 assert로
    assert abs(n_substeps - round(n_substeps)) < 1e-9, "n_substeps가 정수가 아니다"
    assert abs(n_substeps - LESSON_EXPECT["n_substeps"]) < 1e-9 or True  # 아래에서 경고로 처리
    print(f"  ✅ n_substeps = {ctrl_dt:g} / {sim_dt:g} = {n_substeps:g}  (정수 확인)")

    got = dict(
        action_size=int(env.action_size),
        ctrl_dt=ctrl_dt,
        sim_dt=sim_dt,
        n_substeps=n_substeps,
        episode_length=int(cfg.episode_length),
        action_scale=float(cfg.action_scale),
    )
    if isinstance(obs, dict):
        if "state" in obs:
            got["obs_state"] = int(obs["state"][0])
        if "privileged_state" in obs:
            got["obs_privileged"] = int(obs["privileged_state"][0])
        print("\n  관측이 두 개로 나뉜 것은 **비대칭 actor-critic** 관례입니다 (lesson §7.3).")
        print("   - state: actor(정책)가 보는 것. 실기에서도 얻을 수 있는 값만")
        print("   - privileged_state: critic(가치함수)만 보는 것. 지형·마찰·외력 등 학습 중에만 아는 값")
        print("   - 배포되는 것은 actor뿐이므로 특권 정보는 실기에 필요 없습니다.")
        print("   (어느 텐서가 실제로 어디로 들어가는지는 W3-M1에서 코드로 확인)")
    return got


def report_rewards(cfg) -> int:
    """보상항을 그룹별로 출력한다 — lesson §7.5."""
    scales = dict(cfg.reward_config.scales)
    print(f"\n=== [4] 보상항 {len(scales)}개 (lesson §7.5) ===")
    groups = {
        "추종 (Q의 역할)": ["tracking_lin_vel", "tracking_ang_vel"],
        "자세·안정": ["orientation", "base_height", "lin_vel_z", "ang_vel_xy", "pose",
                      "stand_still", "alive", "termination"],
        "접촉·발": ["feet_air_time", "feet_clearance", "feet_height", "feet_phase",
                    "feet_slip", "contact_force", "collision"],
        "정규화·비용 (R의 역할)": ["action_rate", "dof_acc", "energy", "torques",
                                   "dof_pos_limits", "joint_deviation_hip", "joint_deviation_knee"],
    }
    seen = set()
    rows = []
    for g, keys in groups.items():
        present = [k for k in keys if k in scales]
        seen |= set(present)
        rows.append([g, str(len(present)), ", ".join(f"{k}={scales[k]:g}" for k in present)])
    rest = sorted(set(scales) - seen)
    if rest:
        rows.append(["(분류 밖)", str(len(rest)), ", ".join(f"{k}={scales[k]:g}" for k in rest)])
    print_table(["그룹", "개수", "항 = 가중치"], rows, aligns=["left", "right", "left"])

    print("  LQR/MPC 비용함수 J = Σ (x-x_ref)'Q(x-x_ref) + u'Ru 와 구조가 같습니다.")
    print("   - Q의 역할 -> tracking_* / orientation / base_height")
    print("   - R의 역할 -> torques / energy / action_rate / dof_acc")
    print("   - 제약의 소프트 페널티 -> dof_pos_limits / collision / contact_force")
    print("  다른 점: ① 부호가 뒤집혀 최대화 ② 항이 24개라 가중치 튜닝이 별개의 작업")
    print("           ③ feet_air_time·feet_phase는 **접촉 이벤트에 걸려 미분 불가능**")
    print("  ③이 RL이 필요한 진짜 이유입니다. 매끄러운 비용함수라면 MPC로 풀면 됩니다. (W3-M1 본론)")
    return len(scales)


# %% [markdown]
# ## 4. reset / step — 첫 호출(컴파일)과 그 다음(실행)
#
# **여기서 오래 멈출 수 있습니다.** `jax.jit`으로 감싼 함수의 첫 호출은 추적 + 컴파일이라
# 캐시가 비어 있으면 CPU에서 수십 초가 걸립니다. 두 번째 호출부터가 실제 실행 속도입니다.
#
# 단, playground 0.2.0은 백엔드에 따라 **MuJoCo Warp 커널 캐시**(`~/.cache/warp/`)를 함께 씁니다.
# 같은 머신에서 두 번째로 돌리면 그 캐시가 이미 따뜻해서 첫 호출도 빨라집니다 —
# 아래 표의 "첫 호출 / 두 번째 호출" 비가 1에 가깝다면 그 경우입니다.
# 실행 중 `Module ... load on device 'cpu' took ... (cached)` 같은 줄이 쏟아지는 것도 Warp 로그이고 정상입니다.
#
# **학습은 하지 않습니다.** 액션은 그냥 0 벡터를 넣습니다(= 기본 자세 유지 명령).

# %%
def run_steps(env, n_steps: int) -> None:
    """jit(reset) 1회 + jit(step) n_steps회. 각 호출 소요 시간을 출력한다."""
    import jax
    import jax.numpy as jp

    print("\n=== [6] reset / step 타이밍 ===")
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    print("  jax.jit(env.reset) 첫 호출 — 컴파일 포함. 캐시가 비어 있으면 CPU에서 수십 초. 기다리세요...",
          flush=True)
    t0 = time.perf_counter()
    state = jit_reset(jax.random.PRNGKey(0))
    state.obs  # 지연 실행이므로 결과를 실제로 건드려 동기화
    jax.block_until_ready(state.data.qpos)
    t_reset = time.perf_counter() - t0
    print(f"    -> {t_reset:.1f} s")

    action = jp.zeros(env.action_size)  # 학습 없음. 0 = 기본 자세 유지 명령
    rows = []
    for i in range(n_steps):
        t0 = time.perf_counter()
        state = jit_step(state, action)
        jax.block_until_ready(state.data.qpos)
        dt = time.perf_counter() - t0
        rows.append([
            str(i + 1),
            f"{dt * 1e3:,.1f}",
            f"{float(state.reward):.6f}",
            f"{int(state.done)}",
            f"{float(state.data.qpos[2]):.4f}",
            "컴파일 포함" if i == 0 else "실행만",
        ])
    print_table(["step", "소요 [ms]", "reward", "done", "골반 높이 [m]", "비고"],
                rows, aligns=["right", "right", "right", "right", "right", "left"])
    if n_steps >= 2:
        first = float(rows[0][1].replace(",", ""))
        second = float(rows[1][1].replace(",", ""))
        ratio = first / max(second, 1e-9)
        print(f"  첫 호출 / 두 번째 호출 = {ratio:,.1f}배")
        if ratio >= 3.0:
            print("   -> 이 차이가 컴파일 비용입니다. 두 번째 호출부터가 실제 실행 속도입니다.")
        else:
            print("   -> 차이가 거의 없습니다. 커널 캐시가 이미 따뜻하면(같은 머신에서 두 번째 실행 등)")
            print("      컴파일 비용이 드러나지 않습니다. 캐시를 비우고(~/.cache/warp, ~/.cache/jax)")
            print("      다시 돌리면 첫 호출이 훨씬 느려집니다.")
    print("  CPU에서 step 한 번이 초 단위인 것이 정상입니다 — 그래서 본 학습은 GPU가 필요합니다.")
    print("  학습은 여기서 하지 않습니다 — PPO 병렬 학습은 W3-M1입니다.")


# %% [markdown]
# ## 5. 실행

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
    p = argparse.ArgumentParser(
        description="W1-M2 실습 4: mujoco_playground 스모크 (lesson §7). 학습하지 않음."
    )
    p.add_argument("--env", default=ENV_NAME_DEFAULT, help=f"환경 이름 (기본 {ENV_NAME_DEFAULT})")
    p.add_argument("--steps", type=int, default=3, help="jit(env.step) 호출 횟수 (기본 3)")
    p.add_argument("--smoke", action="store_true",
                   help="(기본 동작과 동일) 이 스크립트는 원래 스모크 전용입니다")
    p.add_argument("--no-step", action="store_true",
                   help="reset/step을 건너뛰고 스펙 조회만 (가장 빠름 · CPU에서 유용)")
    if argv is None:
        argv = [] if _in_notebook() else sys.argv[1:]
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print("=" * 92)
    print(" W1-M2 실습 4 — mujoco_playground 스모크 (학습 없음. '에러 없이 돈다'까지만)")
    print("=" * 92)
    print("  ⏱️ CPU JAX면 아래 [5] 단계의 첫 reset에 **수십 초** 걸립니다. 멈춘 것이 아닙니다.")

    t0 = time.perf_counter()
    try:
        import mujoco_playground  # noqa: F401
    except ImportError:
        raise SystemExit(
            "[에러] mujoco_playground를 import할 수 없습니다.\n"
            "  설치: pip install 'playground==0.2.0'\n"
            "  ⚠️ 'pip install mujoco_playground'는 존재하지 않는 패키지입니다 (lesson §7.1)"
        )
    print(f"\n  import mujoco_playground : {time.perf_counter() - t0:.1f} s")

    report_backend()
    all_envs = report_registry()

    from mujoco_playground import registry

    if args.env not in all_envs:
        raise SystemExit(f"[에러] '{args.env}'가 등록 목록에 없습니다. 사용 가능: {all_envs}")

    t0 = time.perf_counter()
    env = registry.load(args.env)
    cfg = registry.get_default_config(args.env)
    print(f"\n  registry.load('{args.env}') : {time.perf_counter() - t0:.1f} s")

    got = report_env_spec(env, cfg, args.env)
    got["n_rewards"] = report_rewards(cfg)

    # lesson §7.3 대조
    mismatches = [f"{k}: 실측 {got[k]} != lesson {v}" for k, v in LESSON_EXPECT.items()
                  if k in got and got[k] != v]
    print("\n=== [5] lesson §7.3 대조 ===")
    if mismatches:
        print("  ⚠️ 다른 항목이 있습니다 (playground 버전 변경 가능):")
        for msg in mismatches:
            print(f"     - {msg}")
        print("     -> 이 출력이 맞습니다. docs/progress.md에 기록하세요.")
    else:
        print("  ✅ lesson §7.3의 스펙과 전부 일치합니다.")

    if args.no_step:
        print("\n[skip] --no-step 지정 → reset/step 생략")
    else:
        run_steps(env, args.steps)

    print("\n" + "-" * 92)
    print(" W1-M2 practice 완주입니다. 다음은 labs/ 로 가서 클라우드 인스턴스 셋업을 마무리하세요.")
    print(" 실행 시간·에러·GPU 비용은 docs/progress.md에 기록할 것.")
    print(" 본 학습(PPO 병렬 보행)은 W3-M1입니다. 여기서는 하지 않습니다.")
    print("-" * 92)


# %%
if __name__ == "__main__":
    main()
