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
# # W2-M2 실습 4. 지연을 자기 기기에서 직접 잰다 (lesson §4.2, §5.3)
#
# 앞의 셋은 [`../lesson.md`](../lesson.md)에 적힌 실측값을 재료로 산술을 했습니다. 이 스크립트는
# **그 표를 자기 기기에서 다시 만듭니다.** 남이 잰 값을 읽는 것과 자기가 재는 것은 다릅니다.
#
# ## 재는 것
#
# 다섯 설정에서 `DiffusionModel.generate_actions` 한 번의 벽시계 시간입니다.
#
# 1. **DDPM 100, 라이브러리 기본값** `DiffusionConfig()` 그대로
# 2. **DDPM 100, 논문 PushT 설정** `horizon=16, n_action_steps=8, crop_shape=(84,84),
#    use_group_norm=True, pretrained_backbone_weights=None, use_separate_rgb_encoder_per_camera=False`
# 3. **DDIM 16** 논문 부록의 실기 설정
# 4. **DDIM 10** 논문 §III-D 본문
# 5. **DDIM 1** 바닥값. 함수 평가를 한 번만 한다
#
# 파라미터 수도 세어 lesson §5.3의 **262,709,026개**, U-Net **251,511,938개**와 대조합니다.
#
# ## 판정 기준을 절대값으로 잡지 않는 이유
#
# 기기가 다르면 절대값은 당연히 다릅니다. GPU 클럭이 유휴 상태에서 내려갔다 올라오는 시간,
# 같은 기기에서 도는 다른 작업, 드라이버 버전이 전부 벽시계에 섞입니다(W2-M1 labs가 같은 스크립트로
# 6.5 ms에서 66.0 ms까지 본 기록이 있습니다). 그래서 여기서는 **동일성이 아니라 두 가지**를 봅니다.
#
# - **자릿수**: 자기 측정이 lesson 값의 0.3배에서 3배 사이인가
# - **순서**: NFE 오름차순으로 지연도 오름차순인가. **이쪽이 진짜 판정 항목입니다.**
#   순서가 깨지면 측정이 잘못됐거나 설정이 반영되지 않은 것입니다.
#
# ## 실행
#
# ```bash
# python 04_dp_latency_bench.py --device cuda            # 본 측정, 반복 10회
# python 04_dp_latency_bench.py --device cpu --smoke     # 반복 3회, NFE 100 건너뜀
# ```
#
# `--smoke`는 반복을 3회로 낮추고 **NFE 100짜리 두 설정을 건너뜁니다.** CPU에서 그 둘은 한 번에
# 몇 분씩 걸립니다.
#
# 출력:
# - stdout: 파라미터 대조, 측정표, lesson §4.2 표와의 나란한 비교
# - `artifacts/W2-M2/04_dp_latency_bench.csv`
#
# **이 스크립트만 `torch`와 `lerobot`이 필요합니다.** 없으면 크래시하지 않고 설치 안내 후 정상 종료합니다.

# %%
from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import time
import unicodedata
from pathlib import Path

MODULE_ID = "W2-M2"

DEPS_OK = True
DEPS_ERR = ""
try:
    import torch

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
except ImportError as exc:  # 크래시하지 않는다. 안내 후 정상 종료.
    DEPS_OK = False
    DEPS_ERR = str(exc)

INSTALL_HELP = """
  torch 또는 lerobot이 없습니다. 이 스크립트(04)만 이 둘이 필요하고 01, 02, 03은 의존성이 없습니다.

  설치 (집필 환경에서 확인된 조합):
    python3.12 -m venv .venv-lerobot && source .venv-lerobot/bin/activate
    uv pip install "lerobot[pusht,diffusion,evaluation]"

  diffusion extra가 빠지면 정책을 만드는 순간 diffusers 부재로 죽습니다(lesson 「실습으로 가기」).

  설치 없이도 이 모듈의 산술은 전부 확인할 수 있습니다:
    python3 01_mean_collapse.py
    python3 02_receding_horizon.py
    python3 03_nfe_budget.py
"""

# 측정 설정. 워밍업을 먼저 돌려야 첫 호출의 커널 컴파일과 메모리 할당이 측정에 섞이지 않는다.
N_WARMUP = 2

# 자릿수 판정 밴드. 기기가 달라도 이 안에는 들어와야 같은 현상을 보고 있다고 본다.
RATIO_LO, RATIO_HI = 0.3, 3.0


# %% [markdown]
# ## 0. 경로와 표 유틸 (01~03과 동일 규약)

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
# ## 1. lesson.md 참조값
#
# 02, 03과 같은 규약입니다. 표를 상수로 베끼지 않고 매번 읽습니다.

# %%
def lesson_path() -> Path:
    return here().parent / "lesson.md"


def read_lesson_text() -> str | None:
    p = lesson_path()
    return p.read_text(encoding="utf-8") if p.exists() else None


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def find_table(text: str, header_keys: list[str]) -> list[list[str]] | None:
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cells = split_row(line)
        if not all(any(k in c for c in cells) for k in header_keys):
            continue
        if i + 1 >= len(lines) or set(lines[i + 1].replace("|", "").replace(" ", "")) - {"-", ":"}:
            continue
        body = []
        for ln in lines[i + 2:]:
            if not ln.startswith("|"):
                break
            body.append(split_row(ln))
        return body
    return None


def num(cell: str) -> float | None:
    m = re.search(r"-?[\d,]*\.?\d+", cell.replace("**", ""))
    return float(m.group(0).replace(",", "")) if m else None


def lesson_latency_by_nfe(text: str) -> dict[tuple[int, str], float]:
    """§4.2 지연표를 (NFE, 라벨 키워드) -> 중앙값 ms 로 읽는다."""
    body = find_table(text, ["설정", "NFE", "중앙값", "최소", "최대"]) or []
    out: dict[tuple[int, str], float] = {}
    for cells in body:
        nfe = num(cells[1])
        med = num(cells[2])
        if nfe is None or med is None:
            continue
        key = "논문" if "논문 PushT 설정" in cells[0] else "기본"
        out[(int(nfe), key)] = med
    return out


def lesson_param_counts(text: str) -> list[int]:
    """§5.3의 파라미터 수 세 개를 순서대로 읽는다. 총, U-Net, 나머지."""
    return [int(x.replace(",", "")) for x in re.findall(r"\*\*([\d,]{9,})개", text)]


# %% [markdown]
# ## 2. 측정 대상 다섯 설정
#
# 관측은 lesson §5.2 블록도의 입력과 같습니다. 이미지 `(B, T_o, 카메라, 3, 96, 96)`과
# 상태 `(B, T_o, 2)`이고 배치는 1, 카메라는 1대입니다. 논문 설정에서 `crop_shape=(84,84)`가
# 붙어도 **입력은 그대로 96x96**이고 자르기는 정책 안에서 일어납니다.

# %%
def input_features():
    return {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
    }


def output_features():
    return {"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}


def build_settings() -> list[dict]:
    """(이름, NFE, lesson 표 키, config 오버라이드) 다섯 벌."""
    paper = dict(
        horizon=16,
        n_action_steps=8,
        crop_shape=(84, 84),
        use_group_norm=True,
        pretrained_backbone_weights=None,
        use_separate_rgb_encoder_per_camera=False,
    )
    ddim = dict(noise_scheduler_type="DDIM")
    return [
        {"name": "DDPM 100, 라이브러리 기본값", "nfe": 100, "key": "기본", "over": {}},
        {"name": "DDPM 100, 논문 PushT 설정", "nfe": 100, "key": "논문", "over": dict(paper)},
        {"name": "DDIM 16, 논문 부록 실기 설정", "nfe": 16, "key": "기본",
         "over": dict(ddim, num_inference_steps=16)},
        {"name": "DDIM 10, 논문 §III-D", "nfe": 10, "key": "기본",
         "over": dict(ddim, num_inference_steps=10)},
        {"name": "DDIM 1, 바닥", "nfe": 1, "key": "기본",
         "over": dict(ddim, num_inference_steps=1)},
    ]


def make_policy(device: str, over: dict):
    cfg = DiffusionConfig(
        input_features=input_features(),
        output_features=output_features(),
        device=device,
        **over,
    )
    policy = DiffusionPolicy(cfg)
    policy.eval()
    policy.to(device)
    return cfg, policy


def make_batch(cfg, device: str) -> dict:
    """generate_actions가 기대하는 형태. observation.images는 카메라 축이 하나 더 있다."""
    return {
        "observation.state": torch.randn(1, cfg.n_obs_steps, 2, device=device),
        "observation.images": torch.rand(1, cfg.n_obs_steps, 1, 3, 96, 96, device=device),
    }


# %% [markdown]
# ## 3. 측정
#
# 워밍업 2회 뒤 반복 측정의 **중앙값**을 씁니다. 평균이 아니라 중앙값인 이유는 벽시계 측정에
# 가끔 섞이는 큰 이상치 하나가 평균을 통째로 끌고 가기 때문입니다.
#
# GPU에서는 커널이 비동기로 큐에 쌓이므로 **매 측정마다 `torch.cuda.synchronize()`** 를 걸어야
# 실제로 끝난 시각을 잽니다. 이것을 빼면 말도 안 되게 빠른 숫자가 나옵니다.

# %%
def measure(policy, batch, device: str, reps: int) -> dict:
    """generate_actions 1회의 벽시계 시간 [ms]. 워밍업 후 반복의 중앙값."""
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    with torch.no_grad():
        for _ in range(N_WARMUP):
            policy.diffusion.generate_actions(batch)
        sync()

        samples: list[float] = []
        for _ in range(reps):
            sync()
            t0 = time.perf_counter()
            actions = policy.diffusion.generate_actions(batch)  # eq.(§3.2) 역방향 반복 전체
            sync()
            samples.append((time.perf_counter() - t0) * 1000.0)

    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "action_shape": tuple(actions.shape),
        "samples": samples,
    }


def count_params(policy) -> tuple[int, int]:
    total = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    unet = sum(p.numel() for p in policy.diffusion.unet.parameters() if p.requires_grad)
    return total, unet


# %% [markdown]
# ## 4. 환경 기록
#
# 벽시계 값만 남기고 측정 조건을 안 남기면 나중에 그 숫자를 해석할 수 없습니다.
# 기기 이름과 부하를 함께 찍습니다.

# %%
def env_lines(device: str) -> list[str]:
    lines = [f"  torch {torch.__version__}, device {device}"]
    if device == "cuda" and torch.cuda.is_available():
        lines.append(f"  GPU {torch.cuda.get_device_name(0)}, "
                     f"메모리 {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GiB")
    try:
        la = os.getloadavg()
        lines.append(f"  load average {la[0]:.2f}, {la[1]:.2f}, {la[2]:.2f}"
                     + ("   부하가 높습니다. 측정값이 흔들립니다." if la[0] > 4 else ""))
    except (OSError, AttributeError):
        pass
    return lines


def pick_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("  [알림] cuda를 요청했지만 사용할 수 없습니다. cpu로 내려갑니다.")
        return "cpu"
    if requested == "mps" and not (hasattr(torch.backends, "mps")
                                   and torch.backends.mps.is_available()):
        print("  [알림] mps를 요청했지만 사용할 수 없습니다. cpu로 내려갑니다.")
        return "cpu"
    return requested


# %% [markdown]
# ## 5. CSV 저장

# %%
def write_csv(path: Path, records: list[dict]) -> Path:
    cols = ["name", "nfe", "median_ms", "min_ms", "max_ms", "lesson_ms", "ratio",
            "params", "unet_params", "action_shape", "device", "reps"]
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items() if k in cols})
    return path


# %% [markdown]
# ## 6. 메인

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W2-M2 실습 4: Diffusion Policy 추론 지연 직접 측정 (lesson §4.2, §5.3)")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"],
                   help="측정할 장치 (기본 cuda, 없으면 cpu로 내려간다)")
    p.add_argument("--reps", type=int, default=10, help="반복 측정 횟수 (기본 10)")
    p.add_argument("--smoke", action="store_true",
                   help="반복을 3회로 낮추고 NFE 100짜리 두 설정을 건너뛴다")
    p.add_argument("--no-csv", action="store_true", help="CSV 저장을 건너뛴다")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("=" * 104)
    print(f"  {MODULE_ID} 실습 4. 지연을 자기 기기에서 직접 잰다")
    print("=" * 104)

    if not DEPS_OK:
        print(f"\n  의존성이 없습니다: {DEPS_ERR}")
        print(INSTALL_HELP)
        return 0

    device = pick_device(args.device)
    reps = 3 if args.smoke else args.reps
    settings = build_settings()
    if args.smoke:
        settings = [s for s in settings if s["nfe"] < 100]
        print("\n  [--smoke] 반복 3회, NFE 100짜리 두 설정을 건너뜁니다.")

    for line in env_lines(device):
        print(line)

    text = read_lesson_text()
    lesson_lat = lesson_latency_by_nfe(text) if text else {}
    lesson_params = lesson_param_counts(text) if text else []
    if not lesson_lat:
        print("  [알림] lesson.md의 §4.2 표를 읽지 못했습니다. 대조 없이 측정만 합니다.")

    n_pass = n_check = 0
    records: list[dict] = []

    # --- [1] 설정별 측정 ------------------------------------------------------
    print(f"\n=== [1] 설정 {len(settings)}개 측정 (워밍업 {N_WARMUP}회 후 {reps}회 반복의 중앙값) ===\n")
    for s in settings:
        cfg, policy = make_policy(device, s["over"])
        total, unet = count_params(policy)
        batch = make_batch(cfg, device)
        eff_nfe = cfg.num_inference_steps or cfg.num_train_timesteps
        res = measure(policy, batch, device, reps)

        lesson_ms = lesson_lat.get((s["nfe"], s["key"]))
        ratio = (res["median_ms"] / lesson_ms) if lesson_ms else float("nan")
        records.append({
            "name": s["name"], "nfe": eff_nfe, "median_ms": res["median_ms"],
            "min_ms": res["min_ms"], "max_ms": res["max_ms"],
            "lesson_ms": lesson_ms if lesson_ms else "",
            "ratio": ratio, "params": total, "unet_params": unet,
            "action_shape": "x".join(str(x) for x in res["action_shape"]),
            "device": device, "reps": reps,
        })

        # 설정이 실제로 반영됐는지. NFE와 액션 청크 모양을 함께 확인한다.
        ok_nfe = eff_nfe == s["nfe"]
        ok_shape = res["action_shape"][1] == cfg.n_action_steps
        n_check += 2
        n_pass += int(ok_nfe) + int(ok_shape)
        print(f"  {s['name']}")
        print(f"    NFE {eff_nfe} (기대 {s['nfe']})  [{'PASS' if ok_nfe else 'FAIL'}], "
              f"액션 청크 {res['action_shape']} (T_a={cfg.n_action_steps})  "
              f"[{'PASS' if ok_shape else 'FAIL'}]")
        print(f"    중앙값 {res['median_ms']:.1f} ms   최소 {res['min_ms']:.1f}   "
              f"최대 {res['max_ms']:.1f}   파라미터 {total:,}")

        del policy
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- [2] 파라미터 대조 ----------------------------------------------------
    print("\n=== [2] 파라미터 수를 lesson §5.3과 대조 ===\n")
    totals = {r["params"] for r in records}
    unets = {r["unet_params"] for r in records}
    print(f"  측정된 총 파라미터 집합: {', '.join(f'{t:,}' for t in sorted(totals))}")
    print(f"  측정된 U-Net 파라미터 집합: {', '.join(f'{t:,}' for t in sorted(unets))}")
    ok_same = len(totals) == 1
    n_check += 1
    n_pass += int(ok_same)
    print(f"  설정이 달라도 총 개수가 같은가   [{'PASS' if ok_same else 'FAIL'}]")
    print("  lesson §5.3: horizon을 16에서 64로 바꿔도 총 개수가 변하지 않습니다.")
    print("  시간축 합성곱이라 시퀀스 길이가 채널 수를 바꾸지 않고, 공간 소프트맥스 출력도 해상도에 무관합니다.")

    if len(lesson_params) >= 2:
        want_total, want_unet = lesson_params[0], lesson_params[1]
        ok_t = totals == {want_total}
        ok_u = unets == {want_unet}
        n_check += 2
        n_pass += int(ok_t) + int(ok_u)
        print(f"\n  lesson 총 {want_total:,}개   [{'PASS' if ok_t else 'FAIL'}]")
        print(f"  lesson U-Net {want_unet:,}개   [{'PASS' if ok_u else 'FAIL'}]"
              f"   전체의 {want_unet / want_total * 100:.1f}%")
    else:
        print("\n  [알림] lesson.md에서 파라미터 수를 읽지 못해 대조를 건너뜁니다.")

    # --- [3] lesson §4.2 표와 나란히 -----------------------------------------
    print("\n=== [3] lesson §4.2 표와 자기 측정을 나란히 ===\n")
    rows = []
    for r in sorted(records, key=lambda x: (x["nfe"], x["median_ms"])):
        lm = r["lesson_ms"]
        if lm:
            band = RATIO_LO <= r["ratio"] <= RATIO_HI
            rows.append([r["name"], str(r["nfe"]), f"{lm:.1f}", f"{r['median_ms']:.1f}",
                         f"{r['min_ms']:.1f}", f"{r['max_ms']:.1f}",
                         f"{r['ratio']:.2f}배", "PASS" if band else "FAIL"])
            n_check += 1
            n_pass += int(band)
        else:
            rows.append([r["name"], str(r["nfe"]), "없음", f"{r['median_ms']:.1f}",
                         f"{r['min_ms']:.1f}", f"{r['max_ms']:.1f}", "", ""])
    print(render_table(
        ["설정", "NFE", "lesson", "내 중앙값", "내 최소", "내 최대", "비율", f"{RATIO_LO}~{RATIO_HI}배"],
        rows, aligns="lrrrrrrl"))

    # 진짜 판정 항목. NFE가 오르면 지연도 올라야 한다.
    print("\n  순서 판정. NFE가 오르면 지연도 올라야 합니다.\n")
    by_nfe: dict[int, float] = {}
    for r in records:
        by_nfe.setdefault(r["nfe"], []).append(r["median_ms"])
    seq = [(n, statistics.median(v)) for n, v in sorted(by_nfe.items())]
    mono = all(seq[i][1] < seq[i + 1][1] for i in range(len(seq) - 1))
    n_check += 1
    n_pass += int(mono)
    print("  " + "  <  ".join(f"NFE {n} ({v:.1f} ms)" for n, v in seq))
    print(f"  단조 증가인가   [{'PASS' if mono else 'FAIL'}]")
    print("  이 순서가 깨지면 설정이 반영되지 않았거나 측정에 다른 부하가 섞인 것입니다.")
    print("  절대값이 lesson과 달라도 됩니다. 기기와 클럭 상태가 다르면 당연히 다릅니다.")

    if len(seq) >= 2:
        (n1, t1), (n2, t2) = seq[0], seq[-1]
        slope = (t2 - t1) / (n2 - n1)
        intercept = t1 - slope * n1
        print(f"\n  이 기기의 분해: tau(NFE) = {intercept:.2f} ms + {slope:.3f} ms x NFE")
        print("  03_nfe_budget.py가 lesson 값으로 낸 분해와 기울기를 비교해 보세요.")
        print("  기울기가 곧 U-Net forward 1회 비용이고, 그것이 이 정책이 느린 유일한 이유입니다.")

    # --- [4] CSV -------------------------------------------------------------
    if not args.no_csv:
        path = write_csv(artifacts_dir() / "04_dp_latency_bench.csv", records)
        print(f"\n[저장] {path}  ({len(records)}행)")

    print("\n" + "=" * 104)
    print(f"  대조 결과: {n_pass}/{n_check} PASS")
    print("  요점: 자기 기기의 숫자를 가져야 lesson의 표를 판단할 수 있다.")
    print("        절대값은 기기마다 다르고, 판정해야 할 것은 자릿수와 NFE 순서다.")
    print("        여기서 나온 값을 docs/progress.md에 남기면 다음 모듈의 입력이 된다.")
    print("=" * 104)
    return 0 if n_pass == n_check else 1


# %%
if __name__ == "__main__":
    import sys

    raise SystemExit(main(None if "ipykernel" not in sys.modules else ["--smoke"]))
