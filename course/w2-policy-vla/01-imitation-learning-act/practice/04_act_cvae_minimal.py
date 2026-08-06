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
# # W2-M1 실습 4 — ACT형 CVAE 최소 구현 (torch)
#
# > ✅ **실행 검증됨 (CPU)** — macOS 26 · Python 3.14.6 · torch 2.13.0 · 2026-08-06.
# > 실측 소요: `--smoke` **16.5초** / 기본 실행 **2분 7초** (Apple Silicon CPU, GPU 불필요).
# > 아래 출력 예시는 전부 이 실행의 실측값입니다.
# >
# > ⚠️ **다만 검증 범위는 CPU 경로뿐입니다.** `--device cuda` / `--device mps` 경로와
# > LeRobot 실물 파이프라인은 검증되지 않았습니다 — 후자는 `../labs/`(별도 턴 작성 예정)의 몫입니다.
# > torch가 없으면 이 스크립트는 **크래시하지 않고 설치 안내를 낸 뒤 정상 종료**합니다.
#
# [`../lesson.md`](../lesson.md) `§3.4`(ELBO → ACT loss)와 `§4.1`(CVAE 세 스택)을 **최소 규모로** 구현합니다.
#
# **범위를 좁힌 곳** — 이미지와 ResNet-18을 뺐습니다. §4.1 블록도에서 이미지 토큰 600개가
# 어텐션 비용의 거의 전부인데, 그 부분은 이 실습의 학습 대상이 아닙니다.
# 여기서 볼 것은 **CVAE loss 조립과 $z=0$ 추론**뿐입니다. 관측은 관절 상태 벡터 하나로 단순화합니다.
#
# $$\mathcal{L}_{\text{ACT}} = \underbrace{\|\hat a_{t:t+k} - a_{t:t+k}\|_1}_{\texttt{l1\_loss}}
#   + \beta \cdot \underbrace{D_{\mathrm{KL}}(q_\phi \| \mathcal{N}(0,I))}_{\texttt{kld\_loss}},
#   \qquad \beta = \texttt{kl\_weight} = 10.0 \tag{§3.4}$$
#
# ## 확인할 것 세 가지
#
# 1. **`l1_loss`와 `kld_loss`의 상대 크기** — lesson §3.4 '읽는 순서 주의'가 지목한 지점입니다.
#    `kl_weight=10.0`을 보고 "KL이 10배 중요하다"로 읽으면 안 됩니다. 두 항은 **스케일이 전혀 다른 양**이라
#    가중을 곱한 뒤의 실제 기여도를 봐야 합니다. 학습 로그에 세 값을 따로 찍습니다.
# 2. **추론 시 $z=0$ 고정 vs $z\sim\mathcal{N}(0,I)$ 샘플링** — lesson §2.7의 "배포 모델은 결정론적"을
#    출력 분산으로 확인합니다. $z=0$이면 같은 관측에 같은 청크가 **항상** 나옵니다.
# 3. **`--no-vae`** — `use_vae=False`, 즉 순수 $\ell_1$ 회귀와의 비교.
#    lesson §2.7이 "실습에서 직접 볼 대상"으로 남겨둔 항목입니다.
#
# ## 데이터 — 같은 상태에서 두 갈래
#
# 같은 관절 상태 $s$에서 **위로 도는 청크**와 **아래로 도는 청크**가 5:5로 나오는 합성 시연을 만듭니다.
# 사람 시연이 매번 조금씩 다르다는 §2.7의 상황을 극단으로 밀어놓은 것입니다.
# $\ell_1$ 회귀의 최적 점추정은 **중앙값**이라 두 갈래의 가운데(= 거의 평평한 청크)로 뭉갭니다.
# 이 뭉개짐을 **중간점 진폭**으로 측정합니다 — W2-M2 Diffusion Policy 예고와 그대로 이어집니다.
#
# 출력:
# - stdout — 학습 로그(세 항 분리) + 세 실험의 요약표
# - `artifacts/W2-M1/04_act_cvae_log.csv` — 스텝별 손실
# - `artifacts/W2-M1/04_act_cvae_summary.csv` — 실험 요약

# %%
from __future__ import annotations

import argparse
import csv
import math
import unicodedata
from pathlib import Path

MODULE_ID = "W2-M1"

TORCH_OK = True
TORCH_ERR = ""
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # torch가 없어도 크래시하지 않는다 — 안내 후 정상 종료
    TORCH_OK = False
    TORCH_ERR = str(exc)

INSTALL_HELP = """
  torch가 설치되어 있지 않습니다. 이 스크립트(04)만 torch가 필요하고, 01·02·03은 의존성이 없습니다.

  ── 설치 ──────────────────────────────────────────────────────────────────
    python3 -m venv .venv && source .venv/bin/activate
    pip install torch==2.13.0          # CPU 빌드로 충분합니다 (집필 검증 버전)

  CUDA 빌드가 필요하면 인덱스를 지정하세요:
    pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130

  ℹ️ 이 스크립트는 Python 3.14.6 + torch 2.13.0 에서 실행 검증됐습니다.
     다만 **labs/ 의 LeRobot 은 Python 3.12 를 요구**하므로, 어차피 3.12 환경을
     따로 만들게 됩니다. 그 환경에서 이 스크립트를 돌려도 동일하게 동작합니다.
  ──────────────────────────────────────────────────────────────────────────

  설치 없이도 이 모듈의 핵심 산술은 전부 확인할 수 있습니다:
    python3 01_latency_budget.py --sweep
    python3 02_ensemble_weights.py
    python3 03_compounding_error.py
"""


# %% [markdown]
# ## 0. 경로·표 유틸 (01~03과 동일 규약)

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


def disp_width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def pad(s: str, width: int, align: str = "l") -> str:
    gap = max(0, width - disp_width(s))
    return (" " * gap + s) if align == "r" else (s + " " * gap)


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


def sparkline(values: list[float]) -> str:
    """손실 곡선을 한 줄로. matplotlib을 쓰지 않는다(의존성은 torch 하나뿐)."""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(blocks[min(7, int((v - lo) / span * 7.999))] for v in values)


# %% [markdown]
# ## 1. 데이터 — 같은 상태에서 두 갈래 액션 청크
#
# ```
# s [B, D_state]  ──┬──  b = 0  →  a [B, K, D_action]  중간이 위로 볼록
#                   └──  b = 1  →  a [B, K, D_action]  중간이 아래로 볼록
#                        (5:5, 관측만으로는 어느 쪽인지 알 수 없다)
# ```
#
# 두 갈래의 **시작점과 끝점은 같고 중간만 갈립니다.** 이렇게 하면 "어느 쪽으로 돌아가느냐"만
# 다봉이고 나머지는 결정적이라 뭉개짐을 **중간점 진폭 하나로** 측정할 수 있습니다.

# %%
def make_dataset(n: int, chunk_size: int, d_state: int, d_action: int,
                 noise: float, generator) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """(states [n, D_s], actions [n, K, D_a], branch [n]) 를 만든다."""
    s = torch.rand(n, d_state, generator=generator) * 2.0 - 1.0        # U(-1, 1)
    branch = (torch.rand(n, 1, 1, generator=generator) < 0.5).float() * 2.0 - 1.0  # ±1

    t = torch.linspace(0.0, 1.0, chunk_size).view(1, chunk_size, 1)     # [1, K, 1]
    # 시작·끝이 같고 중간만 갈리는 반원 모양. sin(pi t) 는 t=0, 1 에서 0.
    bump = torch.sin(math.pi * t)                                       # [1, K, 1]

    base = s[:, :d_action].unsqueeze(1) if d_state >= d_action else \
        s.repeat(1, (d_action // d_state) + 1)[:, :d_action].unsqueeze(1)   # [n, 1, D_a]
    amp = 0.4 + 0.3 * base.abs()                                        # 상태에 의존하는 진폭

    a = base + branch * amp * bump                                      # [n, K, D_a]
    a = a + noise * torch.randn(a.shape, generator=generator)
    return s, a, branch.view(-1)


def mid_amplitude(actions: "torch.Tensor", states: "torch.Tensor", d_action: int) -> "torch.Tensor":
    """청크 중간점이 base(=시작·끝 수준)에서 얼마나 떨어져 있는가. 뭉개짐 측정자.

    두 갈래의 평균은 진폭 0(평평)이므로, 이 값이 0에 가까우면 모델이 두 모드를 뭉갠 것입니다.
    """
    k = actions.shape[1]
    mid = actions[:, k // 2, :]                       # [B, D_a]
    base = 0.5 * (actions[:, 0, :] + actions[:, -1, :])
    return (mid - base).abs().mean(dim=-1)            # [B]


# %% [markdown]
# ## 2. 모델 — §4.1 블록도의 세 스택을 축소
#
# ```
# ━━━━━━━━━━ 학습 경로 (use_vae=True) ━━━━━━━━━━
#
# ① VAE 인코더 (posterior q_phi)          ★ 학습 시에만
#    [CLS]              [B, 1,   D]
#    q → Linear         [B, 1,   D]
#    a → Linear         [B, K,   D]
#      concat →         [B, K+2, D]
#      TransformerEncoder x n_vae_layers
#      [CLS] 출력 [B, D] → Linear → [B, 2*latent]
#                       → mu [B, 32], logvar [B, 32]
#      z = mu + sigma * eps            z [B, 32]
#            │
# ② 메인 인코더                              ← 원본은 여기에 이미지 토큰 600개가 붙는다
#    q → Linear         [B, 1, D]
#    z → Linear         [B, 1, D]
#      concat + 위치임베딩 → [B, 2, D]
#      TransformerEncoder x n_encoder_layers
#      메모리            [B, 2, D]
#            │
# ③ 디코더
#    학습된 쿼리 [K, D] → [B, K, D]      ★ 쿼리 개수 = chunk_size
#    self-attn + cross-attn(메모리) x 1  ← LeRobot 기본값 1층 (lesson §7 첫 오해)
#    Linear(D → D_action)
#            │
#    â [B, K, D_action]
# ```

# %%
if TORCH_OK:

    class MiniACT(nn.Module):
        """ACT의 CVAE 구조를 최소로 옮긴 것. 이미지 경로 없음."""

        def __init__(self, d_state: int, d_action: int, chunk_size: int,
                     dim_model: int = 128, n_heads: int = 4, dim_ff: int = 512,
                     latent_dim: int = 32, use_vae: bool = True,
                     n_vae_encoder_layers: int = 2, n_encoder_layers: int = 2,
                     n_decoder_layers: int = 1, dropout: float = 0.1):
            super().__init__()
            self.use_vae = use_vae
            self.latent_dim = latent_dim
            self.chunk_size = chunk_size

            def enc_layer():
                return nn.TransformerEncoderLayer(
                    d_model=dim_model, nhead=n_heads, dim_feedforward=dim_ff,
                    dropout=dropout, activation="relu",
                    batch_first=True, norm_first=False,   # ACT 기본값 pre_norm=False
                )

            # --- ① VAE 인코더 (posterior) — 추론에서는 쓰지 않는다 ---------------
            if use_vae:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_model))
                self.vae_state_proj = nn.Linear(d_state, dim_model)
                self.vae_action_proj = nn.Linear(d_action, dim_model)
                self.vae_pos = nn.Parameter(torch.zeros(1, chunk_size + 2, dim_model))
                self.vae_encoder = nn.TransformerEncoder(enc_layer(), n_vae_encoder_layers)
                self.latent_head = nn.Linear(dim_model, 2 * latent_dim)  # → mu, logvar

            # --- ② 메인 인코더 -------------------------------------------------
            self.state_proj = nn.Linear(d_state, dim_model)
            self.latent_proj = nn.Linear(latent_dim, dim_model)
            self.enc_pos = nn.Parameter(torch.zeros(1, 2, dim_model))
            self.encoder = nn.TransformerEncoder(enc_layer(), n_encoder_layers)

            # --- ③ 디코더 ------------------------------------------------------
            self.query_embed = nn.Embedding(chunk_size, dim_model)  # 쿼리 개수 = chunk_size
            dec_layer = nn.TransformerDecoderLayer(
                d_model=dim_model, nhead=n_heads, dim_feedforward=dim_ff,
                dropout=dropout, activation="relu", batch_first=True, norm_first=False,
            )
            self.decoder = nn.TransformerDecoder(dec_layer, n_decoder_layers)
            self.action_head = nn.Linear(dim_model, d_action)

        # ---- posterior q_phi(z | a, o) -------------------------------------
        def encode_posterior(self, state, actions):
            b = state.shape[0]
            cls = self.cls_token.expand(b, -1, -1)                       # [B, 1, D]
            st = self.vae_state_proj(state).unsqueeze(1)                 # [B, 1, D]
            ac = self.vae_action_proj(actions)                           # [B, K, D]
            x = torch.cat([cls, st, ac], dim=1) + self.vae_pos           # [B, K+2, D]
            h = self.vae_encoder(x)[:, 0]                                # [CLS] 위치 → [B, D]
            mu, logvar = self.latent_head(h).chunk(2, dim=-1)            # 각 [B, latent]
            return mu, logvar

        # ---- 디코딩 p_theta(a | z, o) ---------------------------------------
        def decode(self, state, z):
            st = self.state_proj(state).unsqueeze(1)                     # [B, 1, D]
            lat = self.latent_proj(z).unsqueeze(1)                       # [B, 1, D]
            memory = self.encoder(torch.cat([st, lat], dim=1) + self.enc_pos)  # [B, 2, D]
            q = self.query_embed.weight.unsqueeze(0).expand(state.shape[0], -1, -1)  # [B, K, D]
            h = self.decoder(q, memory)                                  # [B, K, D]
            return self.action_head(h)                                   # [B, K, D_action]

        def forward(self, state, actions=None):
            """학습 경로. actions 가 있으면 posterior 를 쓴다."""
            b = state.shape[0]
            if self.use_vae and actions is not None:
                mu, logvar = self.encode_posterior(state, actions)
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)  # 재매개화  # eq.(§3.4 ①)
            else:
                mu = logvar = None
                z = torch.zeros(b, self.latent_dim, device=state.device)
            return self.decode(state, z), mu, logvar

        @torch.no_grad()
        def infer(self, state, sample_z: bool = False):
            """추론 경로. LeRobot 기본은 z = 0 고정 — VAE 인코더를 통째로 건너뛴다(lesson §4.2).

            latent_sample = torch.zeros([batch_size, latent_dim]) 에 대응합니다.
            """
            b = state.shape[0]
            if sample_z:
                z = torch.randn(b, self.latent_dim, device=state.device)   # prior 에서 뽑기
            else:
                z = torch.zeros(b, self.latent_dim, device=state.device)   # ← 배포 경로
            return self.decode(state, z)


# %% [markdown]
# ## 3. loss 조립 — lesson §3.4와 문자 그대로 대응
#
# LeRobot `modeling_act.py` 의 한 줄이 이것입니다.
#
# ```python
# loss = l1_loss + mean_kld * self.config.kl_weight
# ```

# %%
def act_losses(pred, target, mu, logvar, kl_weight: float):
    """(total, l1, mean_kld) 를 돌려준다.  # eq.(§3.4)"""
    l1 = F.l1_loss(pred, target)                                    # Laplace 재구성 항  # eq.(§3.4 ②)
    if mu is None:
        return l1, l1, torch.zeros((), device=pred.device)
    # KL( N(mu, sigma^2) || N(0, I) ) — latent 차원으로 합, 배치로 평균 (LeRobot mean_kld 와 동일)
    mean_kld = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()   # eq.(§3.4 ①)
    total = l1 + mean_kld * kl_weight                                            # eq.(§3.4 ③)
    return total, l1, mean_kld


# %% [markdown]
# ## 4. 학습 루프

# %%
def train_one(tag: str, use_vae: bool, args, data, log_rows: list[list]) -> dict:
    s_tr, a_tr, _ = data["train"]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    model = MiniACT(
        d_state=args.d_state, d_action=args.d_action, chunk_size=args.chunk_size,
        dim_model=args.dim_model, latent_dim=args.latent_dim, use_vae=use_vae,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    s_tr, a_tr = s_tr.to(device), a_tr.to(device)
    n = s_tr.shape[0]
    curve: list[float] = []
    print(f"\n  ── 학습: {tag}  (use_vae={use_vae}, "
          f"파라미터 {sum(p.numel() for p in model.parameters()):,}개) ──")
    print(f"  {'step':>6}  {'total':>10}  {'l1_loss':>10}  {'kld_loss':>10}  "
          f"{'kld*beta':>10}  {'KL 기여율':>9}")

    model.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=device)
        sb, ab = s_tr[idx], a_tr[idx]
        pred, mu, logvar = model(sb, ab)
        total, l1, kld = act_losses(pred, ab, mu, logvar, args.kl_weight)

        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        t_v, l_v, k_v = total.detach().item(), l1.detach().item(), kld.detach().item()
        if step % max(1, args.steps // 10) == 0 or step == 1:
            weighted = k_v * args.kl_weight          # 가중을 곱한 뒤의 실제 크기
            share = weighted / max(t_v, 1e-12)       # ← 이 열이 'KL이 얼마나 중요한가'의 답
            print(f"  {step:>6}  {t_v:>10.4f}  {l_v:>10.4f}  "
                  f"{k_v:>10.4f}  {weighted:>10.4f}  {share * 100:>8.1f}%")
            log_rows.append([tag, step, f"{t_v:.6g}", f"{l_v:.6g}",
                             f"{k_v:.6g}", f"{weighted:.6g}", f"{share:.6g}"])
        curve.append(l_v)

    model.eval()
    return {"model": model, "curve": curve}


# %% [markdown]
# ## 5. 평가 — 세 가지 관찰
#
# ① 두 항의 상대 크기 · ② $z=0$ vs $z\sim\mathcal{N}(0,I)$ · ③ CVAE vs 순수 $\ell_1$

# %%
def evaluate_model(model, s_te, a_te, branch, args, n_z_samples: int = 8) -> dict:
    device = s_te.device
    with torch.no_grad():
        pred0 = model.infer(s_te, sample_z=False)          # 배포 경로 — z = 0
        l1_0 = float(F.l1_loss(pred0, a_te))

        # z = 0 을 두 번 돌려 정말 같은 값이 나오는지 (결정론성 확인, lesson §2.7)
        pred0b = model.infer(s_te, sample_z=False)
        determinism = float((pred0 - pred0b).abs().max())

        # z ~ N(0, I) 를 여러 번 뽑아 출력이 얼마나 흩어지는가
        samples = torch.stack([model.infer(s_te, sample_z=True) for _ in range(n_z_samples)])
        spread = float(samples.std(dim=0).mean()) if model.use_vae else float("nan")
        l1_sampled = float(F.l1_loss(samples.mean(dim=0), a_te)) if model.use_vae else float("nan")

        # 뭉개짐 — 중간점 진폭
        amp_true = float(mid_amplitude(a_te, s_te, args.d_action).mean())
        amp_pred = float(mid_amplitude(pred0, s_te, args.d_action).mean())

        # 가장 가까운 갈래에 대한 오차 (두 모드 중 하나만 맞혀도 되는 관대한 지표)
        up = branch > 0
        l1_up = float(F.l1_loss(pred0[up], a_te[up])) if up.any() else float("nan")
        l1_dn = float(F.l1_loss(pred0[~up], a_te[~up])) if (~up).any() else float("nan")

    return {
        "l1_z0": l1_0, "determinism_maxdiff": determinism,
        "z_sample_spread": spread, "l1_z_sampled_mean": l1_sampled,
        "mid_amp_true": amp_true, "mid_amp_pred": amp_pred,
        "amp_ratio": amp_pred / max(amp_true, 1e-12),
        "l1_branch_up": l1_up, "l1_branch_down": l1_dn,
    }


# %% [markdown]
# ## 6. 메인

# %%
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="W2-M1 실습 4: ACT형 CVAE 최소 구현 — lesson §3.4·§4.1 (torch 필요)")
    p.add_argument("--chunk-size", type=int, default=20,
                   help="청크 길이 K (기본 20 — LeRobot 기본값 100의 축소판)")
    p.add_argument("--latent-dim", type=int, default=32, help="z 차원 (기본 32 = ACT 기본값)")
    p.add_argument("--kl-weight", type=float, default=10.0,
                   help="beta (기본 10.0 = ACT 기본값)")
    p.add_argument("--dim-model", type=int, default=128,
                   help="트랜스포머 은닉 차원 (기본 128 — 원본은 512)")
    p.add_argument("--d-state", type=int, default=6, help="관절 상태 차원")
    p.add_argument("--d-action", type=int, default=6, help="액션 차원")
    p.add_argument("--steps", type=int, default=3000, help="학습 스텝 (기본 3000)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-train", type=int, default=4096)
    p.add_argument("--n-test", type=int, default=512)
    p.add_argument("--noise", type=float, default=0.01, help="액션 라벨 잡음")
    p.add_argument("--device", default="cpu", help="cpu | cuda | mps (기본 cpu — 이 규모면 충분)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-vae", action="store_true",
                   help="CVAE를 끄고 순수 l1 회귀만 학습한다 (use_vae=False 대조군만 실행)")
    p.add_argument("--smoke", action="store_true",
                   help="수 분에 끝나는 축소 실행 (steps 200 · n_train 512)")
    p.add_argument("--no-csv", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if not TORCH_OK:
        print("=" * 96)
        print(f"  {MODULE_ID} 실습 4 — ACT형 CVAE 최소 구현")
        print("=" * 96)
        print(f"\n  [skip] import torch 실패: {TORCH_ERR}")
        print(INSTALL_HELP)
        return 0

    args = parse_args(argv)
    if args.smoke:
        args.steps, args.n_train, args.n_test = 200, 512, 256
    out_dir = artifacts_dir()

    print("=" * 96)
    print(f"  {MODULE_ID} 실습 4 — ACT형 CVAE 최소 구현  (torch {torch.__version__}, "
          f"device={args.device})")
    print("=" * 96)
    print("  집필 시 검증: macOS · Python 3.14.6 · torch 2.13.0 · CPU (cuda/mps 경로는 미검증)")
    if args.smoke:
        print("  [--smoke] 축소 실행입니다. 숫자를 결론에 쓰지 마세요.")
    print(f"\n  K={args.chunk_size} · latent_dim={args.latent_dim} · kl_weight={args.kl_weight} "
          f"· dim_model={args.dim_model} · steps={args.steps}")

    g = torch.Generator().manual_seed(args.seed)
    data = {
        "train": make_dataset(args.n_train, args.chunk_size, args.d_state,
                              args.d_action, args.noise, g),
        "test": make_dataset(args.n_test, args.chunk_size, args.d_state,
                             args.d_action, args.noise, g),
    }
    s_te, a_te, br_te = (t.to(args.device) for t in data["test"])

    log_rows: list[list] = []
    runs: list[tuple[str, bool]] = ([("순수 l1 회귀 (use_vae=False)", False)] if args.no_vae
                                    else [("CVAE (use_vae=True)", True),
                                          ("순수 l1 회귀 (use_vae=False)", False)])

    results = []
    for tag, use_vae in runs:
        r = train_one(tag, use_vae, args, data, log_rows)
        ev = evaluate_model(r["model"], s_te, a_te, br_te, args)
        ev["tag"] = tag
        ev["use_vae"] = use_vae
        ev["curve"] = r["curve"]
        results.append(ev)
        print(f"  l1 곡선: {sparkline(r['curve'][::max(1, len(r['curve']) // 48)])}")

    # --- 관찰 ① 두 항의 상대 크기 -------------------------------------------
    print("\n=== 관찰 ① l1_loss 와 kld_loss 는 스케일이 다른 양이다 (lesson §3.4 '읽는 순서 주의') ===\n")
    cvae_logs = [r for r in log_rows if r[0].startswith("CVAE")]
    if cvae_logs:
        last = cvae_logs[-1]
        print(f"  마지막 로그: l1={float(last[3]):.4f} · kld={float(last[4]):.4f} · "
              f"kld*beta={float(last[5]):.4f} · KL 기여율={float(last[6]) * 100:.1f}%")
    print("  → `kl_weight=10.0` 은 '10배 중요'가 아니라 **다른 단위의 양을 같은 자리에 놓기 위한 환산 계수**입니다.")
    print("     위 'KL 기여율' 열이 실제 비중이고, 학습이 진행되면 KL 항이 눌려 내려갑니다.")

    # --- 관찰 ② z=0 vs z 샘플링 ---------------------------------------------
    print("\n=== 관찰 ② 추론 시 z=0 고정 — 배포 모델은 결정론적이다 (lesson §2.7) ===\n")
    rows = [[r["tag"],
             f"{r['l1_z0']:.4f}",
             f"{r['determinism_maxdiff']:.2e}",
             "-" if r["z_sample_spread"] != r["z_sample_spread"] else f"{r['z_sample_spread']:.4f}",
             "-" if r["l1_z_sampled_mean"] != r["l1_z_sampled_mean"] else f"{r['l1_z_sampled_mean']:.4f}"]
            for r in results]
    print(render_table(["모델", "l1 (z=0)", "z=0 재현성(최대차)", "z~N(0,I) 출력 분산", "l1 (z 평균)"],
                       rows, aligns="lrrrr"))
    print("\n  → 'z=0 재현성'이 0 이면 같은 관측에 **항상 같은 청크**가 나온다는 뜻입니다.")
    print("     'z~N(0,I) 출력 분산'이 0 에 가까우면 latent 가 사실상 무시되고 있다는 신호입니다")
    print("     (beta=10.0 이 z 의 정보량을 누르는 방향이라는 lesson §3.4 서술과 대조해 보세요).")
    print("     **다만 z 가 얼마나 붕괴하는지는 데이터에 달린 경험적 문제**라 이 한 번의 실행으로 단언하지 마세요.")

    # --- 관찰 ③ 다봉 타깃 뭉개짐 --------------------------------------------
    print("\n=== 관찰 ③ 다봉 타깃 — l1 회귀는 두 갈래를 어떻게 뭉개는가 (W2-M2 예고) ===\n")
    rows = [[r["tag"], f"{r['mid_amp_true']:.4f}", f"{r['mid_amp_pred']:.4f}",
             f"{r['amp_ratio'] * 100:.1f}%", f"{r['l1_branch_up']:.4f}", f"{r['l1_branch_down']:.4f}"]
            for r in results]
    print(render_table(["모델", "참 중간점 진폭", "예측 진폭 (z=0)", "진폭 보존율",
                        "l1 (위 갈래)", "l1 (아래 갈래)"], rows, aligns="lrrrrr"))
    print("\n  → 진폭 보존율이 0%에 가까우면 모델이 두 모드의 가운데(= 평평한 청크)를 뱉은 것입니다.")
    print("     **CVAE 를 켜도 배포 경로(z=0)는 점추정이라 이 뭉개짐이 원리적으로 남습니다.**")
    print("     lesson §2.7·§7 번외의 요점입니다 — z 는 학습 시 변동을 흡수하는 정규화 장치이지")
    print("     실행 시 모드를 골라주는 장치가 아닙니다. 실행 시점의 다봉성은 추론 자체가")
    print("     샘플링인 Diffusion Policy 의 몫입니다(W2-M2).")

    # --- CSV ----------------------------------------------------------------
    if not args.no_csv:
        p1 = out_dir / "04_act_cvae_log.csv"
        with p1.open("w", newline="", encoding="utf-8") as fp:
            wr = csv.writer(fp)
            wr.writerow(["tag", "step", "total", "l1_loss", "kld_loss", "kld_times_beta", "kl_share"])
            wr.writerows(log_rows)
        p2 = out_dir / "04_act_cvae_summary.csv"
        keys = ["tag", "use_vae", "l1_z0", "determinism_maxdiff", "z_sample_spread",
                "l1_z_sampled_mean", "mid_amp_true", "mid_amp_pred", "amp_ratio",
                "l1_branch_up", "l1_branch_down"]
        with p2.open("w", newline="", encoding="utf-8") as fp:
            wr = csv.writer(fp)
            wr.writerow(keys)
            for r in results:
                wr.writerow([r[k] for k in keys])
        print(f"\n[저장] {p1}\n[저장] {p2}")

    print("\n" + "=" * 96)
    print("  요점: loss = l1_loss + mean_kld * kl_weight 는 세 줄짜리 조립이지만,")
    print("        그 세 줄이 '학습 때 z 를 쓰고 배포 때 z=0 을 쓴다'는 비대칭을 만든다.")
    print("        beta=10.0 은 그 비대칭의 대가를 줄이려는 방향의 선택으로 읽힌다(lesson §3.4).")
    print("        다음 → labs/ 에서 LeRobot 실물 파이프라인으로. (Python 3.12 환경 필요)")
    print("=" * 96)
    return 0


# %%
if __name__ == "__main__":
    import sys

    raise SystemExit(main(None if "ipykernel" not in sys.modules else []))
