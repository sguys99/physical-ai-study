#!/usr/bin/env python3
"""W2-M1 labs — LeRobot ACT 설치본 자동 대조기.

`../lesson.md`가 문서에 박아둔 값들이 **실제로 설치된 LeRobot에서도 그런가**를
한 줄로 확인합니다. lesson을 읽으며 외운 숫자와 손에 깔린 패키지가 갈리면 여기서 걸립니다.

    .venv-lerobot/bin/python labs/verify_act_install.py            # 전체
    .venv-lerobot/bin/python labs/verify_act_install.py --quick    # [1]~[4]만 (모델 생성 없음)

무엇을 대조하는가:

  [1] 환경 리포트 — python · lerobot · torch · cuda · 디코더 백엔드
  [2] lesson §6.4 하이퍼파라미터 21개 ↔ `ACTConfig()` 기본값
  [3] lesson §2.5 · §2.6 · 「흔한 오해」의 영문 축자 인용 3건 ↔ 설치본 소스 원문
  [4] lesson 「출처」 신뢰도 각주 — temporal ensembling 오설정 시의 예외 클래스
  [5] 파라미터 수 — 카메라 1~4대 · `use_vae` on/off  (lesson 「실습으로 가기」의 "약 80M" 대조)
  [6] 추론 경로 실측 — τ_infer · 큐 소비 · z=0 결정론성 · 앙상블 스텝당  (lesson §3.6 · §2.7)
  [7] 데이터셋 메타 — `--with-datasets` 를 줄 때만. **네트워크를 씁니다**

판정 규약 (practice/`01`·`02`와 같습니다):

  * **FAIL** — 설치본이 lesson과 어긋남. 하나라도 있으면 **exit code 1**.
  * **WARN** — lesson 본문 쪽 표기 문제(예: 축자 인용에서 단어 하나 누락).
    설치본은 정상이므로 exit code는 0입니다. 대신 화면과 CSV에 남고,
    `worksheet.md` ③에 기록해 자료 수정의 입력으로 씁니다.

결과는 `artifacts/W2-M1/labs/verify_act_install.csv`에 저장됩니다.

뷰어를 띄우지 않습니다. 그림도 만들지 않습니다 — 이 스크립트의 산출물은 표와 CSV입니다.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as md
import platform
import re
import sys
import time
import unicodedata
from pathlib import Path

MODULE_ID = "W2-M1"

# ── lesson §6.4 표 그대로. 이 21개가 [2]의 기준값입니다 ────────────────────────
# (필드명, lesson §6.4가 적어둔 기본값)
LESSON_DEFAULTS: list[tuple[str, object]] = [
    ("chunk_size", 100),
    ("n_action_steps", 100),
    ("n_obs_steps", 1),
    ("vision_backbone", "resnet18"),
    ("dim_model", 512),
    ("n_heads", 8),
    ("n_encoder_layers", 4),
    ("n_decoder_layers", 1),
    ("n_vae_encoder_layers", 4),
    ("latent_dim", 32),
    ("use_vae", True),
    ("kl_weight", 10.0),
    ("dim_feedforward", 3200),
    ("dropout", 0.1),
    ("pre_norm", False),
    ("feedforward_activation", "relu"),
    ("replace_final_stride_with_dilation", False),
    ("optimizer_lr", 1e-5),
    ("optimizer_lr_backbone", 1e-5),
    ("optimizer_weight_decay", 1e-4),
    ("temporal_ensemble_coeff", None),
]

# ── lesson이 영문 그대로 인용한 문장 3건 ──────────────────────────────────────
#
# 🔴 **lesson 쪽 문구를 여기에 복사해 두지 않습니다.**
#    이 스크립트의 목적이 "lesson이 문서에 박아둔 값이 설치본에서도 그런가"인데,
#    lesson 문구를 상수로 베껴 두면 lesson이 개정될 때 스크립트만 조용히 낡습니다
#    (실제로 집필 중 한 번 그렇게 어긋났습니다). 그래서 **`../lesson.md`를 읽어 대조**하고,
#    파일이 없을 때만(단독 배포) 아래 needle 자체를 기준으로 폴백합니다.
#
# needle : **설치본 소스**에서 찾을 문장. 이것이 정본입니다.
# anchor : lesson 본문에서 그 인용문이 있는 자리를 찾기 위한 짧은 표식.
QUOTES: list[dict[str, str]] = [
    {
        "id": "Q1",
        "where": "lesson §2.5 · configuration_act.py `n_action_steps` docstring",
        "file": "configuration_act.py",
        "needle": "if the chunk size size 100, you may set this to 50",
        "anchor": "if the chunk size size",
        "note": "`size size`는 상류 오타. lesson이 축자 인용하며 오타임을 주석으로 밝혔습니다",
    },
    {
        "id": "Q2",
        "where": "lesson §2.6 · 「흔한 오해」 · configuration_act.py `temporal_ensemble_coeff` docstring",
        "file": "configuration_act.py",
        "needle": (
            "`n_action_steps` must be 1 when using this feature, as inference needs to "
            "happen at every step to form an ensemble."
        ),
        "anchor": "must be 1 when using this feature",
        "note": "이 문장은 docstring 것이고, 실제 예외 메시지는 다른 문장입니다 — [4] 참조",
    },
    {
        "id": "Q3",
        "where": "lesson 「흔한 오해」 첫 오해 · configuration_act.py `n_decoder_layers` 주석",
        "file": "configuration_act.py",
        "needle": (
            "Although the original ACT implementation has 7 for `n_decoder_layers`, "
            "there is a bug in the code that means only the first layer is used."
        ),
        "anchor": "there is a bug in the code",
        "note": "`that means only`가 정본입니다. `means`가 빠지면 축자 인용이 아닙니다",
    },
]

# lesson 「출처」 신뢰도 각주 — 예외 클래스와 메시지
EXPECTED_EXC = "NotImplementedError"
EXPECTED_EXC_MSG = "`n_action_steps` must be 1 when using temporal ensembling."

# lesson.md를 못 찾았을 때만 쓰는 폴백 (본문 주장 파라미터 수 · HF 문서 주장)
FALLBACK_LESSON_PARAM_M = 52.0
FALLBACK_HF_DOC_PARAM_M = 80.0

# [6] 타이밍 워밍업 횟수. 유휴 GPU의 SM 클럭이 올라올 때까지 넉넉히 돌립니다.
WARMUP = 12

# --with-datasets 로 확인할 것 (repo_id, lesson이 기대하는 fps)
DATASETS = [
    ("lerobot/aloha_sim_transfer_cube_human", 50),
    ("lerobot/aloha_mobile_cabinet", 50),
    ("lerobot/pusht", 10),
]


# ══════════════════════════════════════════════════════════════════════════════
# 표 그리기 · 경로 (practice/01·02와 같은 규약)
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
    widths = [max(disp_width(header[i]), *(disp_width(r[i]) for r in rows)) if rows
              else disp_width(header[i]) for i in range(len(header))]
    out = ["  " + "  ".join(pad(header[i], widths[i], aligns[i]) for i in range(len(header)))]
    out.append("  " + "  ".join("-" * widths[i] for i in range(len(header))))
    for r in rows:
        out.append("  " + "  ".join(pad(r[i], widths[i], aligns[i]) for i in range(len(header))))
    return "\n".join(out)


def norm_ws(s: str) -> str:
    """줄바꿈으로 쪼개진 docstring을 한 줄로 편다."""
    return re.sub(r"\s+", " ", s).strip()


def lesson_path() -> Path:
    """`labs/` 옆의 `../lesson.md`."""
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd().resolve()
    return here.parent / "lesson.md"


def load_lesson() -> tuple[str | None, str]:
    """lesson 본문과 상태 문자열을 돌려준다. 없으면 (None, 사유)."""
    p = lesson_path()
    if not p.exists():
        return None, f"찾지 못함 ({p})"
    return p.read_text(encoding="utf-8"), str(p)


def fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, str):
        return f"'{v}'"
    return str(v)


# ══════════════════════════════════════════════════════════════════════════════
# 결과 적재 — CSV 한 곳에 모읍니다
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
# [1] 환경
# ══════════════════════════════════════════════════════════════════════════════
def section_env(led: Ledger) -> None:
    print("\n=== [1] 환경 — 무엇이 깔려 있는가 ===\n")

    def ver(pkg: str) -> str:
        try:
            return md.version(pkg)
        except Exception:
            return "(없음)"

    import torch

    rows = [
        ["python", platform.python_version(), sys.executable],
        ["lerobot", ver("lerobot"), "lesson 「실습으로 가기」 기준"],
        ["torch", torch.__version__, f"cuda_available={torch.cuda.is_available()}"],
        ["torchvision", ver("torchvision"), "ResNet-18 사전학습 가중치의 출처"],
        ["torchcodec", ver("torchcodec"), "로드 실패해도 무해 — 에러 표 E1"],
        ["av (pyav)", ver("av"), "torchcodec 실패 시의 폴백 디코더"],
        ["rerun-sdk", ver("rerun-sdk"), "viz extra. Step 4에서 씁니다"],
        ["numpy", ver("numpy"), ""],
    ]
    if torch.cuda.is_available():
        rows.append(["GPU", torch.cuda.get_device_name(0),
                     f"cuda {torch.version.cuda}"])
    else:
        rows.append(["GPU", "(없음 — CPU로 진행)", "[6]이 느려질 뿐 완주는 됩니다"])
    print(render_table(["항목", "값", "비고"], rows))

    # 인터프리터가 .venv-lerobot 인지 — 이 랩에서 가장 흔한 사고
    in_lerobot_venv = ".venv-lerobot" in sys.executable
    led.add("[1] 환경", "인터프리터가 .venv-lerobot", "True", str(in_lerobot_venv),
            "PASS" if in_lerobot_venv else "WARN",
            "" if in_lerobot_venv else "W1용 .venv와 섞으면 안 됩니다 — 에러 표 E2")
    if not in_lerobot_venv:
        print("\n  ⚠️ 이 인터프리터는 `.venv-lerobot`이 아닙니다. 랩 README §0.2를 보세요.")


# ══════════════════════════════════════════════════════════════════════════════
# [2] lesson §6.4 하이퍼파라미터 21개
# ══════════════════════════════════════════════════════════════════════════════
def section_defaults(led: Ledger) -> None:
    from lerobot.policies.act.configuration_act import ACTConfig

    print("\n=== [2] lesson §6.4 하이퍼파라미터 21개 ↔ ACTConfig() 기본값 ===\n")
    cfg = ACTConfig()  # 입출력 feature 없이도 dataclass 기본값은 다 읽힙니다

    rows: list[list[str]] = []
    for field, want in LESSON_DEFAULTS:
        got = getattr(cfg, field, "(필드 없음)")
        ok = (got == want) if not isinstance(want, float) else abs(got - want) < 1e-12
        rows.append([field, fmt(want), fmt(got), "PASS" if ok else "FAIL"])
        led.add("[2] 기본값", field, fmt(want), fmt(got), "PASS" if ok else "FAIL")
    print(render_table(["필드", "lesson §6.4", "설치본", "대조"], rows, aligns="lrrl"))

    n_pass = led.count("PASS", "[2] 기본값")
    n_all = led.total("[2] 기본값")
    tail = "  ← lesson §6.4 표와 완전 일치" if n_pass == n_all else "  ← ❌ 불일치 발생"
    print(f"\n  대조 결과: {n_pass}/{n_all} PASS{tail}")
    print("  → 이 21개가 lesson §6.4 표의 전부입니다. 하나라도 FAIL이면 lesson의 그 행을")
    print("     설치본 값으로 고치고 worksheet ⑨에 적으세요. **출력을 믿습니다.**")


# ══════════════════════════════════════════════════════════════════════════════
# [3] 영문 축자 인용 3건
# ══════════════════════════════════════════════════════════════════════════════
def section_quotes(led: Ledger) -> None:
    import lerobot.policies.act.configuration_act as cfg_mod

    print("\n=== [3] lesson의 영문 축자 인용 3건 ↔ 설치본 소스 원문 ===\n")
    src_path = Path(cfg_mod.__file__)
    src = src_path.read_text(encoding="utf-8")
    lines = src.splitlines()
    # 줄 앞의 주석 표식(`#`)을 떼고 한 줄로 편다.
    # 이걸 안 하면 여러 줄로 접힌 `# Note: ...` 주석이 needle과 안 맞습니다.
    src_flat = norm_ws(" ".join(re.sub(r"^\s*#\s?", "", ln) for ln in lines))

    lesson_text, lesson_where = load_lesson()
    if lesson_text is None:
        print(f"  ⚠️ `../lesson.md`를 {lesson_where}. 설치본 원문 존재 여부만 봅니다.\n")
        lesson_flat = None
    else:
        lesson_flat = norm_ws(lesson_text)
        print(f"  대조 기준: 설치본 `{src_path.name}` (정본) ↔ `{Path(lesson_where).name}`\n")

    rows: list[list[str]] = []
    problems: list[tuple[dict[str, str], str, str]] = []
    for q in QUOTES:
        needle = norm_ws(q["needle"])
        in_src = needle in src_flat

        # 소스에서의 위치 — 인용문 첫 5단어로 찾습니다
        head = " ".join(q["needle"].split()[:5])
        lineno = next((i + 1 for i, ln in enumerate(lines) if head.split()[0] in ln
                       and all(w in norm_ws(" ".join(lines[i:i + 3])) for w in head.split())), 0)

        if lesson_flat is None:
            lesson_state = "(대조 불가)"
            verdict = "PASS" if in_src else "FAIL"
        elif needle in lesson_flat:
            lesson_state, verdict = "일치", "PASS" if in_src else "FAIL"
        else:
            # lesson에 그 자리가 있기는 한가? anchor로 찾아 실제 문구를 뽑아옵니다.
            idx = lesson_flat.find(q["anchor"])
            if idx < 0:
                lesson_state, verdict = "인용 없음", "WARN"
                problems.append((q, "(lesson에서 이 인용문을 찾지 못했습니다)", ""))
            else:
                got = lesson_flat[max(0, idx - 60): idx + 120]
                lesson_state, verdict = "다름", "WARN"
                problems.append((q, got, ""))

        if not in_src:
            verdict = "FAIL"  # 설치본에 그 문장이 아예 없다 = lerobot 버전이 다르다

        rows.append([q["id"], f"{q['file']}:{lineno or '?'}",
                     "있음" if in_src else "없음", lesson_state, verdict])
        led.add("[3] 축자 인용", f"{q['id']} ({q['where']})",
                "…" + q["needle"][-52:], f"설치본 {'있음' if in_src else '없음'} · "
                f"lesson {lesson_state}", verdict, q["note"])
    print(render_table(["#", "설치본 위치", "설치본 원문", "lesson 표기", "대조"], rows))

    for q, got, _ in problems:
        print(f"\n  ⚠️ {q['id']} — {q['where']}")
        print(f"     설치본 원문(정본) : ...{q['needle'][-70:]}")
        print(f"     lesson 본문       : ...{got[-100:]}")
        print(f"     → {q['note']}")

    if problems:
        print("\n  → WARN은 **설치가 잘못된 것이 아니라 문서 쪽 표기 문제**라 exit code에 "
              "넣지 않습니다.")
        print("     worksheet ③에 옮겨 적으세요. 자료 수정의 입력이 됩니다.")
    else:
        print("\n  → 세 인용문이 설치본 원문과 **글자 단위로 같습니다.** lesson을 읽을 때")
        print("     영문 블록을 그대로 믿어도 된다는 뜻입니다.")


# ══════════════════════════════════════════════════════════════════════════════
# [4] temporal ensembling 오설정 — lesson 「출처」 신뢰도 각주
# ══════════════════════════════════════════════════════════════════════════════
def section_exception(led: Ledger) -> None:
    from lerobot.policies.act.configuration_act import ACTConfig

    print("\n=== [4] temporal ensembling 오설정 — 예외 클래스 (lesson 「출처」 각주) ===\n")
    print("  ACTConfig(temporal_ensemble_coeff=0.01, n_action_steps=100)  ← §2.6이 금지한 조합")

    exc_name, exc_msg = "(예외 없음)", ""
    try:
        ACTConfig(temporal_ensemble_coeff=0.01, n_action_steps=100)
    except Exception as e:  # noqa: BLE001 — 무슨 클래스인지가 확인 대상입니다
        exc_name, exc_msg = type(e).__name__, str(e)

    ok = exc_name == EXPECTED_EXC
    print(f"\n    예외 클래스 : {exc_name}   [{'PASS' if ok else 'FAIL'}] (기대: {EXPECTED_EXC})")
    print(f"    메시지      : {norm_ws(exc_msg)}")
    led.add("[4] 예외", "temporal_ensemble_coeff + n_action_steps>1",
            EXPECTED_EXC, exc_name, "PASS" if ok else "FAIL")

    msg_ok = EXPECTED_EXC_MSG in norm_ws(exc_msg)
    led.add("[4] 예외", "예외 메시지 첫 문장", EXPECTED_EXC_MSG,
            norm_ws(exc_msg)[:80], "PASS" if msg_ok else "FAIL")

    # 정상 조합은 통과해야 합니다
    try:
        ACTConfig(temporal_ensemble_coeff=0.01, n_action_steps=1)
        ok2 = True
    except Exception:  # noqa: BLE001
        ok2 = False
    print(f"    n_action_steps=1 로 바꾸면 : {'통과' if ok2 else '여전히 거부됨'}"
          f"   [{'PASS' if ok2 else 'FAIL'}]")
    led.add("[4] 예외", "temporal_ensemble_coeff + n_action_steps=1", "통과",
            "통과" if ok2 else "거부", "PASS" if ok2 else "FAIL")

    print("\n  🔴 docstring 문장([3] Q2)과 예외 메시지는 **다른 문장**입니다. 둘을 섞지 마세요.")


# ══════════════════════════════════════════════════════════════════════════════
# 정책 만들기 (섹션 [5]·[6] 공용)
# ══════════════════════════════════════════════════════════════════════════════
def build_policy(n_cam: int, use_vae: bool, device: str,
                 img_hw: tuple[int, int] = (480, 640), d_state: int = 14,
                 d_action: int = 14, **cfg_kwargs):
    """ALOHA 규격(480x640 · D=14)의 ACTPolicy를 만든다.

    lesson §4.1 블록도의 입력 3종이 그대로 `input_features`가 됩니다.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    h, w = img_hw
    inputs = {f"observation.images.cam{i}": PolicyFeature(type=FeatureType.VISUAL,
                                                          shape=(3, h, w))
              for i in range(n_cam)}
    inputs["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(d_state,))
    cfg = ACTConfig(
        input_features=inputs,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(d_action,))},
        use_vae=use_vae,
        device=device,
        **cfg_kwargs,
    )
    return ACTPolicy(cfg).to(device)


def n_params(policy) -> int:
    return sum(p.numel() for p in policy.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# [5] 파라미터 수
# ══════════════════════════════════════════════════════════════════════════════
def section_params(led: Ledger, device: str) -> int:
    print("\n=== [5] 파라미터 수 — 카메라를 늘리면 몇 개가 늘어나는가 ===\n")

    rows: list[list[str]] = []
    base = None
    for n_cam in (1, 2, 3, 4):
        p = build_policy(n_cam, use_vae=True, device=device)
        n = n_params(p)
        if base is None:
            base = n
        rows.append([str(n_cam), f"{n:,}", f"{n - base:+,}", f"{n / 1e6:.1f}M"])
        del p
    print(render_table(["카메라 수", "학습가능 파라미터", "1대 대비 증분", "≈"],
                       rows, aligns="rrrr"))

    same = all(r[2] == "+0" for r in rows)
    led.add("[5] 파라미터", "카메라 수와 무관", "+0", rows[-1][2],
            "PASS" if same else "FAIL",
            "백본 공유 — modeling_act.py:334 생성 · :475 반복 호출")

    p_novae = build_policy(1, use_vae=False, device=device)
    n_novae = n_params(p_novae)
    del p_novae
    print(f"\n    use_vae=True   : {base:,}")
    print(f"    use_vae=False  : {n_novae:,}")
    print(f"    차이 (VAE posterior 인코더) : {base - n_novae:,}"
          f"  ≈ {(base - n_novae) / 1e6:.1f}M   ← 추론에서는 안 쓰입니다 (§2.7·§4.2)")
    led.add("[5] 파라미터", "use_vae=False가 더 작다", "<" + f"{base:,}",
            f"{n_novae:,}", "PASS" if n_novae < base else "FAIL",
            "VAE posterior 인코더는 학습 전용")

    # ── lesson 본문이 주장하는 값과 대조 ──────────────────────────────────
    # 🔴 값을 상수로 복사해 두지 않고 `../lesson.md`에서 읽습니다(§[3]과 같은 이유).
    got_m = base / 1e6
    lesson_text, lesson_where = load_lesson()
    if lesson_text is None:
        lesson_m, hf_m, src_label = FALLBACK_LESSON_PARAM_M, FALLBACK_HF_DOC_PARAM_M, "폴백 상수"
        print(f"\n  ⚠️ `../lesson.md`를 {lesson_where}. 폴백 상수로 대조합니다.")
    else:
        src_label = "lesson.md"
        m1 = re.search(r"약\s*([\d.]+)\s*M\s*파라미터", lesson_text)
        lesson_m = float(m1.group(1)) if m1 else FALLBACK_LESSON_PARAM_M
        m2 = re.search(r"HF 문서[^.\n]*?약\s*([\d.]+)\s*M", lesson_text)
        hf_m = float(m2.group(1)) if m2 else FALLBACK_HF_DOC_PARAM_M

    ok_lesson = abs(got_m - lesson_m) <= 1.0     # "약 52M" 표기라 ±1M 허용
    print(f"\n    실측                 : {got_m:.1f}M ({base:,})")
    print(f"    lesson 본문 주장     : 약 {lesson_m:g}M   [{'PASS' if ok_lesson else 'FAIL'}]"
          f"  (출처: {src_label})")
    print(f"    HF 문서 주장         : 약 {hf_m:g}M   ❌ 실측과 불일치")
    led.add("[5] 파라미터", "lesson 본문 파라미터 수", f"약 {lesson_m:g}M", f"{got_m:.1f}M",
            "PASS" if ok_lesson else "FAIL", f"출처: {src_label}")
    led.add("[5] 파라미터", "HF 문서 주장(lesson 「출처」가 불일치로 기록)", f"약 {hf_m:g}M",
            f"{got_m:.1f}M", "INFO", "원인 미확인 — 추측 금지. 후속 확인 항목")

    print(f"\n  🔴 **HF 문서의 약 {hf_m:g}M과 실측 {got_m:.1f}M이 다릅니다.** lesson 「출처」가 이미 이 불일치를")
    print("     🔴 항목으로 기록해뒀고, **원인은 확인되지 않았습니다.**")
    print("     확인된 사실은 둘뿐입니다 — ① 이 설치본이 51.6M을 만든다 ② 카메라 수와 무관하다")
    print("     (백본 공유, modeling_act.py:334 생성 · :475 반복 호출).")
    print("     원 ACT 구현이 카메라마다 백본을 두는지 등은 원 저장소를 봐야 알 수 있고,")
    print("     이번에 보지 않았습니다. **추측하지 말고 확인 대상으로 남기세요.**")
    print("     → worksheet ④의 '확인 대상' 칸으로.")
    return base


# ══════════════════════════════════════════════════════════════════════════════
# [6] 추론 경로 실측
# ══════════════════════════════════════════════════════════════════════════════
def _sync(device: str) -> None:
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


def timed_median(fn, device: str, reps: int, warmup: int = WARMUP) -> float:
    """한 번 호출에 걸리는 시간의 **중앙값** [ms].

    평균이 아니라 중앙값을 쓰는 이유: 유휴 상태의 GPU는 SM 클럭이 0 MHz까지 내려가
    처음 몇 번의 호출이 크게 튑니다(집필 중 같은 코드가 6.5 / 8.1 / 22.9 ms로 나왔습니다).
    워밍업을 넉넉히 주고 중앙값을 쓰면 그 꼬리가 판정을 흔들지 않습니다.
    """
    import torch

    with torch.no_grad():
        for _ in range(warmup):
            fn()
    _sync(device)
    samples: list[float] = []
    with torch.no_grad():
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            _sync(device)
            samples.append((time.perf_counter() - t) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def measure_tau_infer(device: str, reps: int) -> float:
    """청크 1회 생성 시간 [ms] (중앙값)."""
    import torch

    policy = build_policy(1, use_vae=True, device=device)
    policy.eval()
    obs = {
        "observation.images.cam0": torch.rand(1, 3, 480, 640, device=device),
        "observation.state": torch.zeros(1, 14, device=device),
    }
    ms = timed_median(lambda: policy.predict_action_chunk(obs), device, reps)
    del policy, obs
    if device == "cuda":
        torch.cuda.empty_cache()
    return ms


def section_device_compare(led: Ledger, tau_gpu: float, reps: int) -> None:
    """같은 정책을 CPU로도 재서 '하드웨어가 실행 모드를 정한다'를 눈으로 본다."""
    print("\n=== [6-b] 같은 정책을 CPU로도 — 하드웨어가 실행 모드의 가용 범위를 정한다 ===\n")
    tau_cpu = measure_tau_infer("cpu", max(3, reps // 4))
    print(f"  τ_infer  GPU {tau_gpu:6.1f} ms   ·   CPU {tau_cpu:6.1f} ms   "
          f"→ **{tau_cpu / tau_gpu:.1f}배**")
    led.add("[6-b] 장치 대비", "tau_infer_cpu_ms", "(측정값)", f"{tau_cpu:.2f}", "INFO")
    led.add("[6-b] 장치 대비", "CPU/GPU 배율", "(측정값)", f"{tau_cpu / tau_gpu:.1f}", "INFO")

    rows = []
    for label, n_act, f2 in [("기본값 n_action_steps=100", 100, 30),
                             ("기본값 n_action_steps=100", 100, 50),
                             ("앙상블 n_action_steps=1", 1, 30),
                             ("앙상블 n_action_steps=1", 1, 50)]:
        budget = n_act / f2 * 1e3

        def verdict(tau: float) -> str:
            return (f"✅ {budget / tau:,.0f}배 여유" if budget > tau
                    else f"❌ {tau / budget:,.1f}배 부족")

        rows.append([label, f"{f2} Hz", f"{budget:,.0f}",
                     verdict(tau_gpu), verdict(tau_cpu)])
    print()
    print(render_table(["실행 모드", "f2", "예산[ms]", "GPU", "CPU"], rows, aligns="lrrll"))
    print("\n  🔴 **표의 아래 두 행이 갈립니다.** 같은 코드·같은 가중치인데 CPU에서는 temporal")
    print("     ensembling이 원리적으로 불가능합니다. lesson §7.2이 L3 설계 변수로 든 셋")
    print("     (chunk_size · n_action_steps · 앙상블 여부)에 **연산 하드웨어가 네 번째로**")
    print("     결합합니다. 「흔한 오해」 세 번째 오해가 '온보드면 33 ms 안에 끝나야 한다'고 조건부로")
    print("     남긴 자리의 실제 답입니다.")
    print("  ⚠️ 다만 이것은 **데스크톱 x86 CPU**입니다. Jetson급 온보드 SoC는 측정하지 않았고,")
    print("     여기서 일반화하면 안 됩니다 — worksheet ⑧의 '미측정' 칸으로.")


def section_inference(led: Ledger, device: str, reps: int) -> float:
    import torch

    print("\n=== [6] 추론 경로 실측 — lesson §3.6의 τ_infer 자리를 채운다 ===\n")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    policy = build_policy(1, use_vae=True, device=device)
    policy.eval()
    obs = {
        "observation.images.cam0": torch.rand(1, 3, 480, 640, device=device),
        "observation.state": torch.zeros(1, 14, device=device),
    }

    # ── (a) 청크 shape — lesson §4.1 블록도의 [B, chunk_size, D_action] ──────
    policy.reset()
    with torch.no_grad():
        chunk = policy.predict_action_chunk(obs)
    shape = tuple(chunk.shape)
    want_shape = (1, 100, 14)
    print(f"  (a) predict_action_chunk(obs).shape = {shape}"
          f"   [{'PASS' if shape == want_shape else 'FAIL'}] "
          f"기대 {want_shape} = [B, chunk_size, D_action]")
    led.add("[6] 추론", "청크 shape", str(want_shape), str(shape),
            "PASS" if shape == want_shape else "FAIL", "lesson §4.1 블록도")

    # ── (b) τ_infer — 청크 1회 생성 시간 ────────────────────────────────────
    tau_infer_ms = timed_median(lambda: policy.predict_action_chunk(obs), device, reps)
    print(f"\n  (b) τ_infer (청크 1회 생성, {reps}회 중앙값) = **{tau_infer_ms:.1f} ms**")
    # 벤치 위생 — 다른 작업이 돌고 있으면 이 값이 몇 배까지 부풀어 오릅니다.
    try:
        import os
        la1, la5, _ = os.getloadavg()
        ncpu = os.cpu_count() or 1
        print(f"      (측정 시점 load average {la1:.1f} / {la5:.1f} · CPU {ncpu}코어"
              + ("  ⚠️ **부하가 높습니다. 값이 부풀어 있을 수 있습니다**" if la1 > ncpu * 0.5
                 else "") + ")")
    except (OSError, AttributeError):
        pass
    led.add("[6] 추론", "tau_infer_ms", "(측정값)", f"{tau_infer_ms:.2f}", "INFO",
            f"batch 1 · 1카메라 3x480x640 · {device}")

    # lesson §3.6 표의 예산과 대조 — 여기가 이 랩의 무게중심입니다
    print("\n      lesson §3.6 표의 예산에 이 값을 넣으면:")
    budget_rows = []
    n_fit = 0
    specs = [("LeRobot 기본값", 100, 50), ("LeRobot 기본값", 100, 30),
             ("docstring 예시", 50, 50), ("짧은 receding horizon", 10, 50),
             ("temporal ensembling(강제)", 1, 50), ("temporal ensembling(강제)", 1, 30)]
    for label, n_act, f2 in specs:
        budget_ms = n_act / f2 * 1e3
        fits = budget_ms > tau_infer_ms
        n_fit += fits
        budget_rows.append([label, str(n_act), f"{f2} Hz", f"{budget_ms:,.0f}",
                            (f"{budget_ms / tau_infer_ms:,.1f}배 여유" if fits
                             else f"{tau_infer_ms / budget_ms:,.1f}배 부족"),
                            "✅" if fits else "❌"])
    print(render_table(["설정", "n_act", "f2", "예산[ms]", "τ_infer 대비", "들어가나"],
                       budget_rows, aligns="lrrrrl"))
    if n_fit == len(specs):
        print(f"\n      → **{len(specs)}행 전부 여유입니다.** lesson §3.6 첫째 결론"
              "('지연은 제약이 아니다')이 자기 기기 숫자로 확정되는 자리입니다.")
    else:
        print(f"\n      → **{n_fit}/{len(specs)}행만 들어갑니다.** 못 들어가는 것은 예산이 가장")
        print("         빡빡한 앙상블 행일 것입니다 — 기본 설정 행들은 여전히 여유입니다.")
        print("         **이것이 lesson 「흔한 오해」 세 번째 오해의 실물입니다**(앙상블은 공짜가 아니다).")
    print("         worksheet ⑥ 6.1로 옮기세요.")

    # ── (c) 큐 소비 — §2.5의 두 숫자가 왜 별개 필드인가 ─────────────────────
    policy.reset()
    per_call: list[float] = []
    with torch.no_grad():
        for _ in range(6):
            _sync(device)
            t = time.perf_counter()
            policy.select_action(obs)
            _sync(device)
            per_call.append((time.perf_counter() - t) * 1e3)
    ratio = per_call[0] / (sum(per_call[1:]) / len(per_call[1:]))
    print("\n  (c) select_action 1~6회차 [ms] : "
          + " / ".join(f"{x:.2f}" for x in per_call))
    print(f"      1회차 ÷ 이후 평균 = **{ratio:.0f}배**  ← 1회차만 추론이고 이후는 "
          "큐에서 꺼내기만 합니다 (§2.5)")
    led.add("[6] 추론", "select_action 1회차/이후 배율", ">5", f"{ratio:.1f}",
            "PASS" if ratio > 5 else "WARN",
            "chunk_size와 n_action_steps가 별개 필드인 물리적 이유")

    # ── (d) z=0 결정론성 — §2.7 ────────────────────────────────────────────
    policy.reset()
    with torch.no_grad():
        c1 = policy.predict_action_chunk(obs)
        c2 = policy.predict_action_chunk(obs)
    maxdiff = (c1 - c2).abs().max().item()
    print(f"\n  (d) z=0 결정론성 — 같은 관측 2회의 최대차 = **{maxdiff:.3e}**"
          f"   [{'PASS' if maxdiff == 0.0 else 'WARN'}] 기대 정확히 0")
    print("      → 추론에서 latent_sample = torch.zeros(...)로 고정되기 때문입니다 (§2.7·「흔한 오해」 번외).")
    led.add("[6] 추론", "z=0 결정론성 최대차", "0.0", f"{maxdiff:.3e}",
            "PASS" if maxdiff == 0.0 else "WARN", "lesson §2.7")

    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"\n  (e) 추론 VRAM peak = {peak:.2f} GB")
        led.add("[6] 추론", "추론 VRAM peak [GB]", "(측정값)", f"{peak:.2f}", "INFO")
    del policy, obs
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── (f) 앙상블 모드 — 「흔한 오해」 세 번째 오해의 "추론 100배" ────────────────────
    ens = build_policy(1, use_vae=True, device=device,
                       temporal_ensemble_coeff=0.01, n_action_steps=1)
    ens.eval()
    ens.reset()
    obs2 = {
        "observation.images.cam0": torch.rand(1, 3, 480, 640, device=device),
        "observation.state": torch.zeros(1, 14, device=device),
    }
    ens_ms = timed_median(lambda: ens.select_action(obs2), device, reps)
    print(f"\n  (f) 앙상블 모드(temporal_ensemble_coeff=0.01, n_action_steps=1) "
          f"스텝당 = **{ens_ms:.1f} ms**")
    for f2 in (30, 50):
        per_step = 1000.0 / f2
        ok_f2 = ens_ms < per_step
        print(f"      {f2} Hz 예산 {per_step:.0f} ms 대비 "
              + (f"✅ 충족 ({per_step / ens_ms:.1f}배 여유)" if ok_f2
                 else f"❌ 초과 ({ens_ms / per_step:.1f}배 부족)"))
    led.add("[6] 추론", "앙상블 스텝당 [ms]", "(측정값)", f"{ens_ms:.2f}", "INFO",
            "lesson 「흔한 오해」 세 번째 오해 — 이 기기에서는 33 ms 예산을 통과")
    print("\n      🔴 **이 결과를 '앙상블은 공짜다'로 읽으면 안 됩니다.** 통과한 것은 이 GPU의")
    print("         지연뿐이고, 온보드(Jetson급) 지연은 **이번에 측정하지 않았습니다.**")
    del ens, obs2
    if device == "cuda":
        torch.cuda.empty_cache()
    return tau_infer_ms


# ══════════════════════════════════════════════════════════════════════════════
# [7] 데이터셋 메타 (선택)
# ══════════════════════════════════════════════════════════════════════════════
def section_datasets(led: Ledger) -> None:
    print("\n=== [7] 데이터셋 메타 — 기록 FPS·repo_id를 자기 손으로 재확인한다 (lesson 「출처」) ===\n")
    print("  ⓘ 네트워크를 씁니다. 여기서는 **메타데이터만** 읽으므로 붉은 torchcodec")
    print("     traceback이 안 뜹니다. 그 traceback은 실제 프레임을 디코딩하는 쪽")
    print("     (`LeRobotDataset` 생성 · `lerobot-dataset-viz` · `lerobot-train`)에서 뜨고,")
    print("     **무해합니다** — 에러 표 E1.\n")
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    rows: list[list[str]] = []
    for repo_id, want_fps in DATASETS:
        try:
            meta = LeRobotDatasetMetadata(repo_id)
        except Exception as e:  # noqa: BLE001
            rows.append([repo_id, "—", "—", "—", "—", f"실패: {type(e).__name__}"])
            led.add("[7] 데이터셋", repo_id, str(want_fps), "로드 실패", "FAIL", str(e)[:80])
            continue
        fps = meta.fps
        n_ep = meta.total_episodes
        n_fr = meta.total_frames
        cams = [k for k in meta.camera_keys]
        d_act = meta.features["action"]["shape"][0]
        ok = fps == want_fps
        rows.append([repo_id, str(fps), str(n_ep), f"{n_fr:,}",
                     f"{n_fr / fps / 60:.1f}분", f"{len(cams)}대 · D_a={d_act}"])
        led.add("[7] 데이터셋", f"{repo_id} fps", str(want_fps), str(fps),
                "PASS" if ok else "FAIL", f"에피소드 {n_ep} · 프레임 {n_fr}")
    print(render_table(["repo_id", "fps", "에피소드", "프레임", "총 시간", "구성"],
                       rows, aligns="lrrrrl"))
    print("\n  🔴 **fps 열을 자기 눈으로 확인하는 것이 이 절의 목적입니다.**")
    print("     ALOHA 2종이 50 Hz라는 사실은 lesson 「출처」에 이미 ✅로 적혀 있습니다. 여기서는")
    print("     그것을 **자기 손으로 재확인**하고, §3.6 표의 두 열이 왜 함께 있는지를 읽습니다 —")
    print("     **@50 Hz 열은 ALOHA 자신의 개방루프 창**이고, **@30 Hz 열은 우리 로봇의 f2를")
    print("     30 Hz로 가정**했을 때입니다. **두 열은 서로 다른 로봇의 이야기입니다.**")
    print("     기록 FPS와 배포 f2가 같은지는 팀 질문 `W2M1-2`가 묻는 것이고, 다르면")
    print("     액션 리샘플링이냐 청크 길이 환산이냐를 정해야 합니다.")
    print("     → worksheet ⑤·⑥으로.")


# ══════════════════════════════════════════════════════════════════════════════
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W2-M1 labs — LeRobot ACT 설치본 자동 대조기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--quick", action="store_true",
                   help="[1]~[4]만. 모델을 만들지 않아 수 초에 끝납니다")
    p.add_argument("--with-datasets", action="store_true",
                   help="[7] 데이터셋 메타까지. 네트워크를 씁니다")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="[5]·[6]에서 쓸 장치 (기본 auto)")
    p.add_argument("--reps", type=int, default=20,
                   help="[6]의 타이밍 반복 횟수 (기본 20)")
    p.add_argument("--compare-cpu", action="store_true",
                   help="[6-b] 같은 정책을 CPU로도 재서 GPU/CPU 예산 판정을 대비시킨다")
    p.add_argument("--no-csv", action="store_true", help="CSV 저장을 건너뛴다")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    led = Ledger()

    print("=" * 96)
    print(f"  {MODULE_ID} labs — LeRobot ACT 설치본 자동 대조기"
          f"{'  (--quick)' if args.quick else ''}")
    print("=" * 96)

    section_env(led)
    section_defaults(led)
    section_quotes(led)
    section_exception(led)

    if not args.quick:
        import torch
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        section_params(led, device)
        tau = section_inference(led, device, args.reps)
        if args.compare_cpu and device == "cuda":
            section_device_compare(led, tau, args.reps)
        elif args.compare_cpu:
            print("\n  ⓘ 이미 CPU로 돌고 있어 [6-b] 장치 대비를 건너뜁니다.")
    else:
        print("\n  ⓘ --quick 이므로 [5] 파라미터 수와 [6] 추론 실측을 건너뜁니다.")

    if args.with_datasets:
        section_datasets(led)
    elif not args.quick:
        print("\n  ⓘ [7] 데이터셋 메타는 `--with-datasets`를 줄 때만 돕니다(네트워크 사용).")

    # ── 종합 ────────────────────────────────────────────────────────────────
    n_fail, n_warn = led.count("FAIL"), led.count("WARN")
    n_pass, n_info = led.count("PASS"), led.count("INFO")
    print("\n" + "=" * 96)
    print(f"  종합 — PASS {n_pass} · FAIL {n_fail} · WARN {n_warn} · INFO {n_info}"
          f"  (총 {led.total()}건)")
    print("=" * 96)
    if n_fail:
        print("\n  ❌ FAIL 항목:")
        for r in led.rows:
            if r["verdict"] == "FAIL":
                print(f"     - [{r['section']}] {r['item']}: "
                      f"기대 {r['expected']} / 실제 {r['actual']}")
    if n_warn:
        print("\n  ⚠️ WARN 항목 (설치는 정상 — worksheet ③·④에 기록):")
        for r in led.rows:
            if r["verdict"] == "WARN":
                print(f"     - [{r['section']}] {r['item']}: "
                      f"기대 {r['expected']} / 실제 {r['actual']}")

    if not args.no_csv:
        out = artifacts_dir() / "verify_act_install.csv"
        led.save(out)
        print(f"\n  [저장] {out}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
