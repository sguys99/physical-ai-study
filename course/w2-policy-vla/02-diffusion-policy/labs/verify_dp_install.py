#!/usr/bin/env python3
"""W2-M2 labs. LeRobot Diffusion Policy 설치본 자동 대조기.

`../lesson.md`가 문서에 박아둔 값들이 **실제로 설치된 LeRobot에서도 그런가**를 한 줄로 확인합니다.
lesson을 읽으며 외운 숫자와 손에 깔린 패키지가 갈리면 여기서 걸립니다.

    .venv-lerobot/bin/python labs/verify_dp_install.py                 # [1]~[6]
    .venv-lerobot/bin/python labs/verify_dp_install.py --quick         # [1]~[4]와 [6]. 모델 생성 없음
    .venv-lerobot/bin/python labs/verify_dp_install.py --with-dataset  # [7]까지 (네트워크 사용)

무엇을 대조하는가:

  [1] 환경 리포트. python, lerobot, torch, cuda, diffusers, gym-pusht, pymunk
  [2] lesson §6.1 설정 드리프트 표 7항목 ↔ `DiffusionConfig()`와 `TrainPipelineConfig` 기본값
  [3] lesson §6.2 `drop_n_last_frames` 산수 ↔ 설치본 기본값과 소스 주석 원문
  [4] 제약 검증. n_action_steps 상한, horizon 배수, num_inference_steps 해소
  [5] 파라미터 수. 총계와 U-Net 몫  (모델을 만듭니다. `--quick`이면 건너뜁니다)
  [6] `PushtEnv` 필드. fps, episode_length, gym_id, gym_kwargs가 넘기지 않는 것
  [7] `lerobot/pusht` 메타. 에피소드, 프레임, fps, 관측 해상도. `--with-dataset`을 줄 때만

판정 규약은 W2-M1 `verify_act_install.py`, practice `01`~`04`와 같습니다.

  * **FAIL**. 설치본이 lesson과 어긋남. 하나라도 있으면 **exit code 1**.
  * **WARN**. 설치본은 정상인데 lesson 쪽 표기나 실행 환경이 어긋남. exit code는 0입니다.
    화면과 CSV에 남고 `worksheet.md`에 기록해 자료 수정의 입력으로 씁니다.
  * **INFO**. 판정 대상이 아닌 측정값과 참고값.

🔴 **lesson의 값을 이 파일에 상수로 베껴 두지 않습니다.** `../lesson.md`를 읽어 기대값을 뽑습니다.
   상수로 베끼면 lesson이 개정될 때 스크립트만 조용히 낡아 정상인 문서를 계속 잡습니다
   (course-plan §9.9에 그 사고 기록이 있습니다). lesson을 못 찾을 때만 폴백 상수를 씁니다.

결과는 `artifacts/W2-M2/labs/verify_dp_install.csv`에 저장됩니다.
뷰어를 띄우지 않습니다. 그림도 만들지 않습니다. 이 스크립트의 산출물은 표와 CSV입니다.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.metadata as md
import platform
import re
import sys
import unicodedata
from pathlib import Path

MODULE_ID = "W2-M2"

# ── lesson.md를 못 찾았을 때만 쓰는 폴백 ──────────────────────────────────────
# 이 값들은 "기대값의 출처"가 아니라 **비상용**입니다. 정상 경로에서는 lesson에서 읽습니다.
FALLBACK = {
    "drift": [
        # (필드, 라이브러리 기본값, 논문 레시피)
        ("horizon", 64, 16),
        ("n_action_steps", 32, 8),
        ("crop_shape", None, (84, 84)),
        ("use_group_norm", False, True),
        ("pretrained_backbone_weights", "<imagenet>", None),
        ("use_separate_rgb_encoder_per_camera", True, False),
        ("batch_size", 8, 64),
    ],
    "drop_cases": [(16, 8, 2, 7), (64, 32, 2, 31)],
    "params_total": 262_709_026,
    "params_unet": 251_511_938,
    "pusht_hz": 10,
    "dataset": (206, 25_650, 10, 96, 96),
}

# 설치본 소스에서 찾을 문장. 이것이 정본입니다.
SRC_COMMENT_NEEDLE = "horizon - n_action_steps - n_obs_steps + 1"
SRC_DOCSTRING_NEEDLE = "n_action_steps <= horizon - n_obs_steps + 1"

IMAGENET = "<imagenet>"  # "ImageNet 사전학습 가중치" 셀을 나타내는 표식


# ══════════════════════════════════════════════════════════════════════════════
# 공용 도구
# ══════════════════════════════════════════════════════════════════════════════
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
    out = find_repo_root() / "artifacts" / MODULE_ID / "labs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def disp_width(s: str) -> int:
    """한글은 2칸을 먹는다. 표 정렬용."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def pad(s: str, width: int, align: str = "l") -> str:
    gap = max(0, width - disp_width(s))
    return " " * gap + s if align == "r" else s + " " * gap


def render_table(header: list[str], rows: list[list[str]], aligns: str = "") -> str:
    aligns = (aligns + "l" * len(header))[: len(header)]
    widths = [
        max([disp_width(header[i])] + [disp_width(r[i]) for r in rows])
        for i in range(len(header))
    ]
    out = ["  " + "  ".join(pad(header[i], widths[i], aligns[i]) for i in range(len(header)))]
    out.append("  " + "  ".join("-" * widths[i] for i in range(len(header))))
    for r in rows:
        out.append("  " + "  ".join(pad(r[i], widths[i], aligns[i]) for i in range(len(header))))
    return "\n".join(out)


def lesson_path() -> Path:
    """`labs/` 옆의 `../lesson.md`."""
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd().resolve()
    return here.parent / "lesson.md"


def load_lesson() -> tuple[str | None, str]:
    p = lesson_path()
    if not p.exists():
        return None, f"찾지 못함 ({p})"
    return p.read_text(encoding="utf-8"), str(p)


def fmt(v: object) -> str:
    if v is IMAGENET or v == IMAGENET:
        return "ImageNet 가중치"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, str):
        return f"'{v}'"
    if isinstance(v, tuple):
        return "[" + ", ".join(str(x) for x in v) + "]"
    return str(v)


# ══════════════════════════════════════════════════════════════════════════════
# 결과 적재
# ══════════════════════════════════════════════════════════════════════════════
class Ledger:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def add(self, section: str, item: str, expected: object, actual: object,
            verdict: str, note: str = "") -> str:
        self.rows.append({
            "section": section, "item": item,
            "expected": str(expected), "actual": str(actual),
            "verdict": verdict, "note": note,
        })
        return verdict

    def count(self, verdict: str, section: str | None = None) -> int:
        return sum(1 for r in self.rows
                   if r["verdict"] == verdict and (section is None or r["section"] == section))

    def total(self, section: str | None = None) -> int:
        return sum(1 for r in self.rows if section is None or r["section"] == section)

    def save(self, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["section", "item", "expected", "actual",
                                              "verdict", "note"])
            w.writeheader()
            w.writerows(self.rows)


# ══════════════════════════════════════════════════════════════════════════════
# lesson 파서. 기대값은 전부 여기서 나옵니다
# ══════════════════════════════════════════════════════════════════════════════
def _clean_cell(s: str) -> str:
    return s.replace("**", "").replace("`", "").strip()


def _coerce(field: str, raw: str) -> object:
    """lesson 표의 셀 문자열을 파이썬 값으로. 못 읽으면 원문 그대로 돌려준다."""
    t = _clean_cell(raw)
    head = t.split(",")[0].strip()  # "None, 자르지 않음" → "None"
    low = head.lower()
    if field in ("horizon", "n_action_steps", "batch_size"):
        m = re.search(r"-?\d+", head)
        return int(m.group()) if m else t
    if low in ("none", "null"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    m = re.match(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", head)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    if "imagenet" in t.lower():
        return IMAGENET
    return t


def parse_drift_table(text: str) -> list[tuple[str, object, object]] | None:
    """lesson §6.1의 설정 드리프트 표 + 산문의 batch_size 한 줄."""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines[:-1]):
        # 🔴 헤더 행만 잡습니다. `라이브러리 기본값`은 §4.2 지연표의 **데이터 셀**에도
        #    나오므로 부분 문자열만 보고 잡으면 엉뚱한 표를 읽습니다(집필 중 실제로 그랬습니다).
        if not ln.startswith("|"):
            continue
        if not re.match(r"^\|[\s\-:|]+\|$", lines[i + 1]):
            continue                               # 다음 줄이 구분선이어야 헤더입니다
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] == "항목" and "라이브러리 기본값" in cells[1]:
            start = i
            break
    if start is None:
        return None

    out: list[tuple[str, object, object]] = []
    for ln in lines[start + 2:]:                    # +2 는 헤더 구분선 건너뛰기
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < 3:
            break
        names = [_clean_cell(x) for x in cells[0].split("/")]
        defs = cells[1].split("/")
        pap = cells[2].split("/")
        if len(names) == 2 and len(defs) == 2 and len(pap) == 2:
            for j, nm in enumerate(names):
                out.append((nm, _coerce(nm, defs[j]), _coerce(nm, pap[j])))
        else:
            nm = names[0]
            out.append((nm, _coerce(nm, cells[1]), _coerce(nm, cells[2])))

    # 일곱 번째 항목은 표가 아니라 산문에 있습니다. 파일이 달라 놓치기 쉬운 자리라 lesson도 그렇게 적었습니다.
    m = re.search(r"`batch_size`\s*기본값이\s*\*\*(\d+)\*\*[^*]*\*\*(\d+)\*\*", text)
    if m:
        out.append(("batch_size", int(m.group(1)), int(m.group(2))))
    return out or None


def parse_drop_cases(text: str) -> list[tuple[int, int, int, int]] | None:
    """lesson §6.2가 산문에 적어둔 `$16 - 8 - 2 + 1 = 7$` 꼴을 전부 찾는다."""
    found = re.findall(r"\$(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*\+\s*1\s*=\s*(\d+)\$", text)
    # 퀴즈 정답 절이 같은 계산을 한 번 더 적어두므로 중복을 걷어냅니다.
    seen: list[tuple[int, int, int, int]] = []
    for g in found:
        t = tuple(int(x) for x in g)
        if t not in seen:
            seen.append(t)                          # type: ignore[arg-type]
    return seen or None


def parse_param_counts(text: str) -> tuple[int | None, int | None]:
    m_tot = re.search(r"총\s*\*\*([\d,]+)개\*\*", text)
    m_unet = re.search(r"U-Net이\s*\*\*([\d,]+)개", text)
    tot = int(m_tot.group(1).replace(",", "")) if m_tot else None
    unet = int(m_unet.group(1).replace(",", "")) if m_unet else None
    return tot, unet


def parse_pusht_hz(text: str) -> int | None:
    m = re.search(r"PushT는\s*(\d+)\s*Hz", text)
    return int(m.group(1)) if m else None


def parse_dataset_facts(text: str) -> tuple[int, int, int, int, int] | None:
    m = re.search(
        r"([\d,]+)\s*에피소드,\s*([\d,]+)\s*프레임,\s*(\d+)\s*fps,\s*(\d+)×(\d+)", text)
    if not m:
        return None
    g = [int(x.replace(",", "")) for x in m.groups()]
    return (g[0], g[1], g[2], g[3], g[4])


# ══════════════════════════════════════════════════════════════════════════════
# [1] 환경
# ══════════════════════════════════════════════════════════════════════════════
def section_env(led: Ledger) -> None:
    print("\n=== [1] 환경. 무엇이 깔려 있는가 ===\n")

    def ver(pkg: str) -> str:
        try:
            return md.version(pkg)
        except Exception:
            return "(없음)"

    import torch

    rows = [
        ["python", platform.python_version(), sys.executable],
        ["lerobot", ver("lerobot"), "0.6.1에서 집필하고 검증"],
        ["torch", torch.__version__, f"cuda_available={torch.cuda.is_available()}"],
        ["diffusers", ver("diffusers"), "🔴 없으면 정책 생성이 죽습니다. 에러 표 E3"],
        ["gym-pusht", ver("gym-pusht"), "평가 환경. 에러 표 E1"],
        ["pymunk", ver("pymunk"), "PushT의 2D 물리"],
        ["gymnasium", ver("gymnasium"), ""],
        ["torchvision", ver("torchvision"), "ResNet-18 사전학습 가중치의 출처"],
        ["numpy", ver("numpy"), ""],
    ]
    if torch.cuda.is_available():
        rows.append(["GPU", torch.cuda.get_device_name(0), f"cuda {torch.version.cuda}"])
    else:
        rows.append(["GPU", "(없음. CPU로 진행)", "[5]가 느려질 뿐 완주는 됩니다"])
    print(render_table(["항목", "값", "비고"], rows))

    in_venv = ".venv-lerobot" in sys.executable
    led.add("[1] 환경", "인터프리터가 .venv-lerobot", "True", str(in_venv),
            "PASS" if in_venv else "WARN",
            "" if in_venv else "W1용 .venv와 섞으면 안 됩니다. 에러 표 E7")
    if not in_venv:
        print("\n  ⚠️ 이 인터프리터는 `.venv-lerobot`이 아닙니다. 랩 README §0.2를 보세요.")

    has_diffusers = ver("diffusers") != "(없음)"
    led.add("[1] 환경", "diffusers 설치", "설치됨", ver("diffusers"),
            "PASS" if has_diffusers else "FAIL",
            "" if has_diffusers else 'uv pip install "lerobot[diffusion]"')
    has_pusht = ver("gym-pusht") != "(없음)"
    led.add("[1] 환경", "gym-pusht 설치", "설치됨", ver("gym-pusht"),
            "PASS" if has_pusht else "FAIL",
            "" if has_pusht else 'uv pip install "lerobot[pusht]"')


# ══════════════════════════════════════════════════════════════════════════════
# [2] lesson §6.1 설정 드리프트 7항목
# ══════════════════════════════════════════════════════════════════════════════
def section_drift(led: Ledger, lesson_text: str | None) -> None:
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    print("\n=== [2] lesson §6.1 설정 드리프트 7항목 ↔ 설치본 기본값 ===\n")

    drift = parse_drift_table(lesson_text) if lesson_text else None
    if drift is None:
        drift = list(FALLBACK["drift"])            # type: ignore[arg-type]
        src = "폴백 상수"
        print("  ⚠️ lesson §6.1 표를 읽지 못해 폴백 상수로 대조합니다.\n")
    else:
        src = "lesson.md §6.1"

    cfg = DiffusionConfig()
    train_defaults = {f.name: f.default for f in dataclasses.fields(TrainPipelineConfig)}

    rows: list[list[str]] = []
    for field, want_default, want_paper in drift:
        if field == "batch_size":
            got = train_defaults.get(field, "(필드 없음)")
            where = "TrainPipelineConfig"
        else:
            got = getattr(cfg, field, "(필드 없음)")
            where = "DiffusionConfig"

        if want_default is IMAGENET or want_default == IMAGENET:
            ok = isinstance(got, str) and "IMAGENET" in got.upper()
        elif isinstance(want_default, tuple):
            ok = tuple(got) == want_default if got is not None else False
        else:
            ok = got == want_default and type(got) is type(want_default)
        drifted = fmt(want_default) != fmt(want_paper)

        rows.append([field, fmt(want_default), fmt(got), "PASS" if ok else "FAIL",
                     fmt(want_paper), "다름" if drifted else "같음"])
        led.add("[2] 드리프트", f"{field} 기본값", fmt(want_default), fmt(got),
                "PASS" if ok else "FAIL", f"{where}, 출처 {src}")
        led.add("[2] 드리프트", f"{field} 논문 레시피와 다른가", "다름",
                "다름" if drifted else "같음", "PASS" if drifted else "FAIL",
                f"lesson이 적은 논문 레시피 값 {fmt(want_paper)}")

    print(render_table(["필드", "lesson 기본값", "설치본", "대조", "논문 레시피", "드리프트"],
                       rows, aligns="lrrlrl"))

    n_pass = led.count("PASS", "[2] 드리프트")
    n_all = led.total("[2] 드리프트")
    print(f"\n  대조 결과: {n_pass}/{n_all} PASS   (7항목 × 2. 기본값 일치 + 논문 레시피와의 드리프트)")
    print(f"  기대값 출처: {src}")
    print("  → 일곱 항목이 전부 '다름'이면 lesson §6.1의 주장이 설치본에서 재현된 것입니다.")
    print("     곧 오버라이드 없이 돌린 학습은 논문 재현이 아니라 다른 모델 학습입니다.")


# ══════════════════════════════════════════════════════════════════════════════
# [3] lesson §6.2 drop_n_last_frames
# ══════════════════════════════════════════════════════════════════════════════
def section_drop_frames(led: Ledger, lesson_text: str | None) -> None:
    import lerobot.policies.diffusion.configuration_diffusion as cfg_mod
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    print("\n=== [3] lesson §6.2 `drop_n_last_frames` 산수 ↔ 설치본 ===\n")

    cfg = DiffusionConfig()
    actual = cfg.drop_n_last_frames

    # ① 공식이 설치본 주석에 그대로 있는가. lesson이 수식으로 옮겨 적은 그 문장입니다.
    src = Path(cfg_mod.__file__).read_text(encoding="utf-8")
    has_comment = SRC_COMMENT_NEEDLE in src
    line_no = next((i + 1 for i, ln in enumerate(src.splitlines())
                    if SRC_COMMENT_NEEDLE in ln), None)
    print(f"  소스 주석 원문 : `# {SRC_COMMENT_NEEDLE}`")
    print(f"  위치           : configuration_diffusion.py:{line_no}   "
          f"[{'PASS' if has_comment else 'FAIL'}]")
    led.add("[3] drop_n_last", "소스 주석의 공식", SRC_COMMENT_NEEDLE,
            f"line {line_no}" if has_comment else "없음",
            "PASS" if has_comment else "FAIL", "lesson §6.2가 수식으로 옮긴 그 문장")

    # ② lesson이 산문에 적은 계산 두 건을 그대로 재현한다
    cases = parse_drop_cases(lesson_text) if lesson_text else None
    src_label = "lesson.md §6.2"
    if cases is None:
        cases = list(FALLBACK["drop_cases"])        # type: ignore[arg-type]
        src_label = "폴백 상수"
        print("\n  ⚠️ lesson §6.2의 계산식을 읽지 못해 폴백 상수로 대조합니다.")

    rows: list[list[str]] = []
    for tp, ta, to, want in cases:
        got_formula = tp - ta - to + 1
        arith_ok = got_formula == want
        matches_default = got_formula == actual
        label = "논문 레시피" if (tp, ta) == (16, 8) else "라이브러리 기본값"
        rows.append([label, f"({tp}, {ta}, {to})", str(want), str(got_formula),
                     "PASS" if arith_ok else "FAIL", str(actual),
                     "일치" if matches_default else "어긋남"])
        led.add("[3] drop_n_last", f"{label} 공식 계산", str(want), str(got_formula),
                "PASS" if arith_ok else "FAIL", f"출처 {src_label}")

    print("\n" + render_table(
        ["설정", "(T_p, T_a, T_o)", "lesson", "재계산", "대조", "설치본 기본값", "관계"],
        rows, aligns="llrrlrl"))

    # ③ lesson이 "어긋남"이라고 적은 것이 실제로 어긋나는가
    mismatch = [r for r in rows if r[-1] == "어긋남"]
    ok = len(mismatch) == 1 and mismatch[0][0] == "라이브러리 기본값"
    led.add("[3] drop_n_last", "기본값이 자기 주석과 어긋난다", "어긋남 1건(기본값)",
            f"어긋남 {len(mismatch)}건", "PASS" if ok else "FAIL",
            "lesson §6.2가 기록한 불일치의 재현. 원인은 확인되지 않았습니다")

    print(f"\n  설치본 기본값 : drop_n_last_frames = {actual}")
    print("  → 논문 레시피 (16, 8, 2)에서는 공식이 7이라 기본값과 맞습니다.")
    print("     라이브러리 기본값 (64, 32, 2)에서는 공식이 31인데 값은 7 그대로입니다.")
    print("     🔴 **원인은 확인되지 않았습니다.** lesson §6.2도 그렇게 적었습니다. 추측하지 마세요.")
    print("     학습 샘플 구성에 실제로 어떤 영향이 가는지는 worksheet ⑤의 관찰 항목입니다.")


# ══════════════════════════════════════════════════════════════════════════════
# [4] 제약 검증
# ══════════════════════════════════════════════════════════════════════════════
def section_constraints(led: Ledger) -> None:
    import lerobot.policies.diffusion.modeling_diffusion as mod
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    print("\n=== [4] 제약 검증. 세 숫자가 아무 값이나 되지 않는다 ===\n")

    cfg = DiffusionConfig()
    configs = [
        ("라이브러리 기본값", cfg.horizon, cfg.n_action_steps, cfg.n_obs_steps),
        ("논문 PushT 레시피", 16, 8, 2),
    ]

    # ① n_action_steps <= horizon - n_obs_steps + 1   (설치본 docstring이 명시하는 요구)
    rows: list[list[str]] = []
    for label, h, a, o in configs:
        bound = h - o + 1
        ok = a <= bound
        rows.append([label, str(h), str(a), str(o), f"{a} <= {bound}",
                     "PASS" if ok else "FAIL"])
        led.add("[4] 제약", f"{label} n_action_steps 상한", f"<= {bound}", str(a),
                "PASS" if ok else "FAIL", "modeling_diffusion.py select_action docstring")
    print(render_table(["설정", "horizon", "n_action", "n_obs", "n_action <= h-o+1", "대조"],
                       rows, aligns="lrrrrl"))

    src = Path(mod.__file__).read_text(encoding="utf-8")
    has_doc = SRC_DOCSTRING_NEEDLE in src
    led.add("[4] 제약", "docstring이 상한을 명시", SRC_DOCSTRING_NEEDLE,
            "있음" if has_doc else "없음", "PASS" if has_doc else "FAIL",
            "설치본이 근거. 이 제약은 예외로 강제되지 않고 문서로만 있습니다")
    print(f"\n  docstring 원문 : {SRC_DOCSTRING_NEEDLE}   "
          f"[{'PASS' if has_doc else 'FAIL'}]  (modeling_diffusion.py select_action)")
    print("  ⓘ 이 상한은 **예외로 강제되지 않습니다.** 어겨도 조용히 지나가므로 손으로 확인해야 합니다.")

    # ② horizon % 2**len(down_dims) == 0   (이쪽은 실제로 예외가 납니다)
    factor = 2 ** len(cfg.down_dims)
    rows = []
    for label, h, _a, _o in configs:
        ok = h % factor == 0
        rows.append([label, str(h), str(factor), f"{h} % {factor} = {h % factor}",
                     "PASS" if ok else "FAIL"])
        led.add("[4] 제약", f"{label} horizon 배수", f"{factor}의 배수", str(h),
                "PASS" if ok else "FAIL", f"down_dims={cfg.down_dims}")
    print("\n" + render_table(["설정", "horizon", "다운샘플 배수", "나머지", "대조"],
                              rows, aligns="lrrrl"))

    bad_h = factor + 1 if factor > 1 else 3          # 어떤 factor에서도 배수가 아닌 값
    try:
        DiffusionConfig(horizon=bad_h)
        raised, detail = False, "예외 없음"
    except ValueError as e:
        raised, detail = True, f"ValueError: {str(e)[:60]}"
    except Exception as e:                            # noqa: BLE001
        raised, detail = False, f"{type(e).__name__}"
    led.add("[4] 제약", f"horizon={bad_h}이면 예외", "ValueError", detail,
            "PASS" if raised else "FAIL", "배수 제약은 __post_init__이 실제로 막습니다")
    print(f"\n  horizon={bad_h} 시도 → {detail}   [{'PASS' if raised else 'FAIL'}]")

    # ③ num_inference_steps = None 이 무엇으로 해소되는가
    print(f"\n  num_inference_steps 기본값 : {cfg.num_inference_steps}"
          f"   num_train_timesteps : {cfg.num_train_timesteps}")
    ok = cfg.num_inference_steps is None and cfg.num_train_timesteps == 100
    led.add("[4] 제약", "num_inference_steps 기본값이 None", "None",
            str(cfg.num_inference_steps), "PASS" if cfg.num_inference_steps is None else "FAIL",
            "None이면 학습 스텝 수로 해소됩니다")
    led.add("[4] 제약", "해소 목표값 num_train_timesteps", "100",
            str(cfg.num_train_timesteps), "PASS" if cfg.num_train_timesteps == 100 else "FAIL",
            "곧 정책 한 번 호출에 몸통 forward 100회")
    print(f"  → 해소 목표는 {cfg.num_train_timesteps}입니다. "
          f"{'실물 해소는 [5]에서 확인합니다.' if ok else ''}")


# ══════════════════════════════════════════════════════════════════════════════
# [5] 파라미터 수와 스텝 해소 (모델을 만듭니다)
# ══════════════════════════════════════════════════════════════════════════════
def section_params(led: Ledger, lesson_text: str | None, device: str) -> None:
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    print("\n=== [5] 파라미터 수와 num_inference_steps 실물 해소 ===\n")
    print(f"  모델을 만듭니다({device}). ResNet-18 가중치가 캐시에 없으면 처음 한 번 받습니다.\n")

    inp = {
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
    }
    out = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
    cfg = DiffusionConfig(input_features=inp, output_features=out, device=device)
    policy = DiffusionPolicy(cfg)
    policy.eval()

    total = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    unet = sum(p.numel() for p in policy.diffusion.unet.parameters() if p.requires_grad)
    rest = total - unet
    resolved = policy.diffusion.num_inference_steps

    want_total, want_unet = (parse_param_counts(lesson_text) if lesson_text else (None, None))
    src_label = "lesson.md §5.3"
    if want_total is None or want_unet is None:
        want_total = want_total or FALLBACK["params_total"]   # type: ignore[assignment]
        want_unet = want_unet or FALLBACK["params_unet"]      # type: ignore[assignment]
        src_label = "폴백 상수"

    rows = [
        ["총 학습가능 파라미터", f"{want_total:,}", f"{total:,}",
         "PASS" if total == want_total else "FAIL"],
        ["그중 U-Net", f"{want_unet:,}", f"{unet:,}",
         "PASS" if unet == want_unet else "FAIL"],
        ["나머지(시각 인코더와 부속)", f"{want_total - want_unet:,}", f"{rest:,}",
         "PASS" if rest == want_total - want_unet else "FAIL"],
    ]
    print(render_table(["항목", src_label, "설치본", "대조"], rows, aligns="lrrl"))
    led.add("[5] 파라미터", "총 학습가능 파라미터", f"{want_total:,}", f"{total:,}",
            "PASS" if total == want_total else "FAIL", f"출처 {src_label}")
    led.add("[5] 파라미터", "U-Net 파라미터", f"{want_unet:,}", f"{unet:,}",
            "PASS" if unet == want_unet else "FAIL", f"출처 {src_label}")
    print(f"\n  U-Net 비중 : {unet / total * 100:.1f}%"
          "   ← 무게의 거의 전부가 몸통에 있습니다")

    led.add("[5] 파라미터", "num_inference_steps 실물 해소", str(cfg.num_train_timesteps),
            str(resolved), "PASS" if resolved == cfg.num_train_timesteps else "FAIL",
            "[4] ③의 실물 확인. 정책 한 번 호출에 몸통 forward 이 횟수")
    print(f"\n  num_inference_steps 실물 : {resolved}   "
          f"[{'PASS' if resolved == cfg.num_train_timesteps else 'FAIL'}]"
          f"   (설정값 {cfg.num_inference_steps} → 해소)")
    print("  → 이 횟수가 lesson §4.2의 665 ms를 만드는 원인입니다. 크기가 아니라 반복입니다.")

    del policy


# ══════════════════════════════════════════════════════════════════════════════
# [6] PushtEnv 필드
# ══════════════════════════════════════════════════════════════════════════════
def section_env_config(led: Ledger, lesson_text: str | None) -> None:
    from lerobot.envs.configs import PushtEnv

    print("\n=== [6] `PushtEnv` 필드. 무엇이 환경으로 넘어가고 무엇이 안 넘어가는가 ===\n")

    env = PushtEnv()
    kwargs = env.gym_kwargs

    want_hz = (parse_pusht_hz(lesson_text) if lesson_text else None) or FALLBACK["pusht_hz"]
    src_label = "lesson.md §4.2" if lesson_text and parse_pusht_hz(lesson_text) else "폴백 상수"

    rows = [
        ["fps", str(want_hz), str(env.fps), "PASS" if env.fps == want_hz else "FAIL",
         f"출처 {src_label}. 한 스텝 {1000 / env.fps:.0f} ms"],
        ["episode_length", "300", str(env.episode_length),
         "PASS" if env.episode_length == 300 else "FAIL", "롤아웃 mp4의 프레임 수가 됩니다"],
        ["gym_id", "gym_pusht/PushT-v0", env.gym_id,
         "PASS" if env.gym_id == "gym_pusht/PushT-v0" else "FAIL",
         "이 이름을 못 찾는 것이 에러 표 E1"],
    ]
    print(render_table(["필드", "기대", "설치본", "대조", "비고"], rows, aligns="lrrll"))
    for r in rows:
        led.add("[6] 환경설정", r[0], r[1], r[2], r[3], r[4])

    print(f"\n  gym_kwargs가 실제로 넘기는 것 : {sorted(kwargs)}")
    print(f"  observation_height / width   : {env.observation_height} / {env.observation_width}")
    leaked = [k for k in ("observation_height", "observation_width") if k in kwargs]
    ok = not leaked
    led.add("[6] 환경설정", "observation_height/width가 gym_kwargs에 없다", "없음",
            "없음" if ok else f"있음 {leaked}", "PASS" if ok else "FAIL",
            "lesson §6.2. 384는 롤아웃 mp4 해상도일 뿐 관측 해상도가 아닙니다")
    print(f"  → 두 필드는 gym_kwargs에 {'없습니다' if ok else '있습니다'}. "
          f"[{'PASS' if ok else 'FAIL'}]")
    print("     그래서 384는 환경에 전달되지 않고, 관측은 데이터셋과 같은 96×96으로 들어옵니다.")
    print("     384는 `visualization_*`가 만드는 **롤아웃 mp4의 해상도**입니다. 두 숫자를 섞지 마세요.")


# ══════════════════════════════════════════════════════════════════════════════
# [7] 데이터셋 메타 (--with-dataset)
# ══════════════════════════════════════════════════════════════════════════════
def section_dataset(led: Ledger, lesson_text: str | None) -> None:
    print("\n=== [7] `lerobot/pusht` 메타. 자기 손으로 다시 세어본다 ===\n")
    print("  ⓘ 네트워크를 씁니다. 메타데이터만 읽으므로 붉은 torchcodec traceback은 안 뜹니다.")
    print("     그 traceback은 실제 프레임을 디코딩하는 쪽에서 뜨고 무해합니다. 에러 표 E4.\n")

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    want = (parse_dataset_facts(lesson_text) if lesson_text else None)
    src_label = "lesson.md 「실습으로 가기」"
    if want is None:
        want = FALLBACK["dataset"]                  # type: ignore[assignment]
        src_label = "폴백 상수"
    w_ep, w_fr, w_fps, w_h, w_w = want              # type: ignore[misc]

    try:
        meta = LeRobotDatasetMetadata("lerobot/pusht")
    except Exception as e:                           # noqa: BLE001
        led.add("[7] 데이터셋", "lerobot/pusht 메타 로드", "성공",
                f"{type(e).__name__}", "FAIL", str(e)[:80])
        print(f"  ❌ 로드 실패: {type(e).__name__}: {str(e)[:120]}")
        return

    img_shape = meta.features["observation.image"]["shape"]
    rows = [
        ["에피소드", f"{w_ep:,}", f"{meta.total_episodes:,}",
         "PASS" if meta.total_episodes == w_ep else "FAIL"],
        ["프레임", f"{w_fr:,}", f"{meta.total_frames:,}",
         "PASS" if meta.total_frames == w_fr else "FAIL"],
        ["fps", str(w_fps), str(meta.fps), "PASS" if meta.fps == w_fps else "FAIL"],
        ["관측 해상도", f"{w_h}×{w_w}", f"{img_shape[0]}×{img_shape[1]}",
         "PASS" if (img_shape[0], img_shape[1]) == (w_h, w_w) else "FAIL"],
    ]
    print(render_table(["항목", src_label, "설치본", "대조"], rows, aligns="lrrl"))
    for r in rows:
        led.add("[7] 데이터셋", r[0], r[1], r[2], r[3], f"출처 {src_label}")

    print(f"\n  평균 에피소드 : {meta.total_frames / meta.total_episodes:.1f} 프레임"
          f" = {meta.total_frames / meta.total_episodes / meta.fps:.1f}초")
    print(f"  카메라        : {meta.camera_keys}")
    print(f"  action shape  : {tuple(meta.features['action']['shape'])}"
          f"   state shape : {tuple(meta.features['observation.state']['shape'])}")

    st = meta.stats.get("observation.state", {})
    if "min" in st and "max" in st:
        lo = [round(float(x), 1) for x in st["min"]]
        hi = [round(float(x), 1) for x in st["max"]]
        print(f"\n  observation.state 범위 : min {lo}  max {hi}")
        print("  → 🔴 이 값이 관절각이 아니라 **픽셀 좌표**라는 증거입니다. 96도 아니고 384도 아닌")
        print("     0~512 대역인데, 시뮬 캔버스가 512×512이고 관측 이미지만 96×96으로 줄여 주기")
        print("     때문입니다. lesson §2.1이 '화면 픽셀 좌표계 위의 밀대 위치'라고 적은 그 값입니다.")
        led.add("[7] 데이터셋", "observation.state가 픽셀 좌표 대역", "최대 100 초과",
                f"max {hi}", "PASS" if max(hi) > 100 else "FAIL",
                "관절각이면 이 대역이 나오지 않습니다")


# ══════════════════════════════════════════════════════════════════════════════
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W2-M2 labs. LeRobot Diffusion Policy 설치본 자동 대조기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--quick", action="store_true",
                   help="[5]를 건너뛴다. 모델을 만들지 않아 수 초에 끝납니다")
    p.add_argument("--with-dataset", action="store_true",
                   help="[7] 데이터셋 메타까지. 네트워크를 씁니다")
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"],
                   help="[5]에서 쓸 장치 (기본 cpu. 파라미터를 세는 데는 GPU가 필요 없습니다)")
    p.add_argument("--no-csv", action="store_true", help="CSV 저장을 건너뛴다")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    led = Ledger()

    lesson_text, lesson_where = load_lesson()

    print("=" * 96)
    print(f"  {MODULE_ID} labs. LeRobot Diffusion Policy 설치본 자동 대조기"
          f"{'  (--quick)' if args.quick else ''}")
    print(f"  기대값 출처: {lesson_where}")
    print("=" * 96)
    if lesson_text is None:
        print("\n  ⚠️ `../lesson.md`를 찾지 못했습니다. 폴백 상수로 대조하며, 이 스크립트는")
        print("     lesson 옆(`labs/`)에서 실행하는 것을 전제로 합니다.")
        led.add("[0] 전제", "lesson.md 위치", "labs/ 옆", "없음", "WARN", lesson_where)

    section_env(led)
    section_drift(led, lesson_text)
    section_drop_frames(led, lesson_text)
    section_constraints(led)

    if not args.quick:
        import torch
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        section_params(led, lesson_text, device)
    else:
        print("\n  ⓘ --quick 이므로 [5] 파라미터 수를 건너뜁니다.")

    section_env_config(led, lesson_text)

    if args.with_dataset:
        section_dataset(led, lesson_text)
    else:
        print("\n  ⓘ [7] 데이터셋 메타는 `--with-dataset`을 줄 때만 돕니다(네트워크 사용).")

    n_fail, n_warn = led.count("FAIL"), led.count("WARN")
    n_pass, n_info = led.count("PASS"), led.count("INFO")
    print("\n" + "=" * 96)
    print(f"  종합. PASS {n_pass} / FAIL {n_fail} / WARN {n_warn} / INFO {n_info}"
          f"  (총 {led.total()}건)")
    print("=" * 96)
    if n_fail:
        print("\n  ❌ FAIL 항목:")
        for r in led.rows:
            if r["verdict"] == "FAIL":
                print(f"     - [{r['section']}] {r['item']}: "
                      f"기대 {r['expected']} / 실제 {r['actual']}")
    if n_warn:
        print("\n  ⚠️ WARN 항목 (설치는 정상. worksheet ②에 기록):")
        for r in led.rows:
            if r["verdict"] == "WARN":
                print(f"     - [{r['section']}] {r['item']}: "
                      f"기대 {r['expected']} / 실제 {r['actual']}")

    if not args.no_csv:
        out = artifacts_dir() / "verify_dp_install.csv"
        led.save(out)
        print(f"\n  [저장] {out}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
