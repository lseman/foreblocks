import math

# ============================================================
# RL-ish online learner with advantage weighting + EMA teacher
# ============================================================
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# small helpers
# ------------------------------
def _to_tensor(x):
    if torch.is_tensor(x):
        return x.float().unsqueeze(0) if x.dim() == 1 else x.float()
    x = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(x).unsqueeze(0) if x.ndim == 1 else torch.from_numpy(x)

def _softplus(x, beta=1.0):
    return math.log1p(math.exp(beta * x)) / beta

class _EMA:
    def __init__(self, beta=0.99):
        self.beta = beta
        self.value = None
    def update(self, x):
        x = float(x)
        self.value = x if self.value is None else self.beta * self.value + (1 - self.beta) * x
        return self.value
    def get(self, default=0.0):
        return self.value if self.value is not None else default

# ============================================================
# Radius policy (distributional, risk-aware, advantage-weighted)
# ============================================================
class RadiusPolicy(nn.Module):
    """
    Predicts a Gaussian over Δlog r, then squashes with tanh into [-Δmax, Δmax].
    new_radius = radius * exp(Δ)
    """
    def __init__(self, d_in: int, hidden: int = 64, delta_max: float = 0.15, dropout_p: float = 0.05):
        super().__init__()
        self.delta_max = float(delta_max)

        def block(cin, cout):
            return nn.Sequential(
                nn.LayerNorm(cin),
                nn.Linear(cin, cout),
                nn.SiLU(),
                nn.Dropout(dropout_p),
            )

        self.backbone = nn.Sequential(
            block(d_in, hidden),
            block(hidden, hidden),
        )
        self.head_mu = nn.Linear(hidden, 1)
        self.head_logstd = nn.Linear(hidden, 1)

        # init: small std, small outputs
        nn.init.zeros_(self.head_mu.weight); nn.init.zeros_(self.head_mu.bias)
        nn.init.constant_(self.head_logstd.weight, -2.0); nn.init.constant_(self.head_logstd.bias, -2.0)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.head_mu(h)        # (B,1)
        log_std = torch.clamp(self.head_logstd(h), -4.0, 1.0)  # cap std
        return mu, log_std

    @torch.no_grad()
    def step(self, feats, radius: float, min_r: float, max_r: float,
             temperature: float = 1.0, risk_kappa: float = 0.5):
        """
        Inference rule:
          Δ_raw ~ N(mu, (temp*std)^2); use risk-aware mean: mu - κ·std
          Δ = tanh(Δ_raw) * Δmax, then clamp by available headroom.
        """
        x = feats if torch.is_tensor(feats) else _to_tensor(feats)
        mu, log_std = self.forward(x)
        std = torch.exp(log_std).clamp_min(1e-4)
        mu = mu.squeeze(-1)
        std = std.squeeze(-1)

        # risk-aware deterministic delta (no sampling)
        delta_raw = (mu - risk_kappa * std).item() / max(1e-6, float(temperature))
        delta = math.tanh(delta_raw) * self.delta_max

        # headroom clamp: do not overshoot bounds in one step
        max_up = math.log(max_r / max(radius, 1e-12))
        max_dn = math.log(max(radius, 1e-12) / max(min_r, 1e-12))
        delta = float(np.clip(delta, -max_dn, max_up))

        new_r = float(radius * math.exp(delta))
        new_r = float(max(min(new_r, max_r), min_r))
        return new_r, float(delta)

# ============================================================
# Region attention with learned query + temperature + masking
# ============================================================
class RegionAttention(nn.Module):
    """
    Learned query attends over region features.
    scores(feats) -> softmax with temperature τ.
    """
    def __init__(self, d_in: int, d_model: int = 64, n_heads: int = 4, dropout_p: float = 0.05, tau: float = 1.0):
        super().__init__()
        self.tau = float(tau)
        self.inp = nn.Linear(d_in, d_model)
        self.q_token = nn.Parameter(torch.zeros(1, 1, d_model))  # learned query
        nn.init.normal_(self.q_token, std=0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout_p)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout_p),
            nn.Linear(d_model, 1),
        )

    @torch.no_grad()
    def scores(self, feats_batch: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """
        feats_batch: (R, d_in)
        mask: optional boolean mask (R,) where False means 'exclude'
        """
        X = _to_tensor(feats_batch).unsqueeze(0)                # (1,R,d_in)
        H = torch.silu(self.inp(X))                             # (1,R,D)
        Q = self.q_token.expand(1, 1, H.size(-1))               # (1,1,D)
        # attend query over regions
        H2, _ = self.attn(Q, H, H, need_weights=False)          # (1,1,D)
        s = self.mlp(H2).squeeze(0).squeeze(-1)                 # (1,) → scalar score per query (we want per-region weights)
        # use keys' values for per-region scores instead:
        # project region embeddings to logits
        logits = self.mlp(torch.cat([H], dim=1)).squeeze(-1).squeeze(0)  # (R,)

        if mask is not None:
            m = torch.as_tensor(mask, dtype=torch.bool)
            logits[~m] = -1e9

        w = torch.softmax(logits / max(1e-6, self.tau), dim=0).cpu().numpy()
        # sanitize
        if not np.all(np.isfinite(w)) or w.sum() <= 0:
            w = np.ones(len(feats_batch), dtype=np.float64)
        return w / (w.sum() + 1e-12)


@dataclass
class RLConfig:
    buffer_cap: int = 2048
    lr: float = 3e-4
    train_every: int = 1
    steps_per_train: int = 64
    batch: int = 128
    delta_max: float = 0.15
    temperature: float = 1.0
    risk_kappa: float = 0.5
    ema_tau: float = 0.995

class RadiusLearnerAgent:
    """
    Online learner for Δ prediction with advantage-weighted regression.
    Stores (features, delta_taken, reward) and regresses μ toward good deltas.
    """
    def __init__(self, d_in: int, cfg: RLConfig | None = None, device: str = "cpu"):
        self.cfg = cfg or RLConfig()
        self.device = device
        self.policy = RadiusPolicy(d_in, delta_max=self.cfg.delta_max).to(device)
        # EMA teacher for stability
        self.teacher = RadiusPolicy(d_in, delta_max=self.cfg.delta_max).to(device)
        self.teacher.load_state_dict(self.policy.state_dict())
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        self.buf: list[tuple[np.ndarray, float, float]] = []
        self._ticks = 0
        self._baseline = _EMA(beta=0.99)

    def push(self, feats: np.ndarray, delta: float, reward: float):
        if not np.all(np.isfinite(feats)):  # keep buffer clean
            return
        self.buf.append((feats.astype(np.float32), float(delta), float(reward)))
        if len(self.buf) > self.cfg.buffer_cap:
            self.buf.pop(0)
        self._baseline.update(reward)

    def _advantage(self, r: float) -> float:
        b = self._baseline.get(0.0)
        return float(r - b)

    def maybe_train(self):
        self._ticks += 1
        if self._ticks % self.cfg.train_every != 0:
            return
        if len(self.buf) < max(64, self.cfg.batch):
            return

        import random
        for _ in range(self.cfg.steps_per_train):
            B = random.sample(self.buf, self.cfg.batch)
            X = torch.stack([_to_tensor(b[0]).squeeze(0) for b in B]).to(self.device)      # (B,d)
            y_delta = torch.tensor([b[1] for b in B], dtype=torch.float32, device=self.device).unsqueeze(1)  # (B,1)
            adv = torch.tensor([self._advantage(b[2]) for b in B], dtype=torch.float32, device=self.device).unsqueeze(1)

            mu, log_std = self.policy(X)     # (B,1), (B,1)
            std = torch.exp(log_std).clamp_min(1e-4)

            # project target deltas back through tanh^-1 to raw space (numerically safe)
            eps = 1e-6
            y_clip = torch.clamp(y_delta / self.policy.delta_max, -1 + eps, 1 - eps)
            y_raw = 0.5 * torch.log((1 + y_clip) / (1 - y_clip))   # atanh

            # advantage weighting (positive advantages emphasized via softplus)
            w = F.softplus(adv)  # >=0

            # negative log-likelihood of target under N(mu, std)
            nll = 0.5 * ((y_raw - mu) / std).pow(2) + log_std
            loss = (w * nll).mean()

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()

            # update EMA teacher
            with torch.no_grad():
                for t, s in zip(self.teacher.parameters(), self.policy.parameters()):
                    t.mul_(self.cfg.ema_tau).add_((1.0 - self.cfg.ema_tau) * s)

    @torch.no_grad()
    def step(self, feats, radius: float, min_r: float, max_r: float):
        # delegate to policy with configured risk/temperature
        return self.policy.step(
            feats, radius, min_r, max_r,
            temperature=self.cfg.temperature,
            risk_kappa=self.cfg.risk_kappa,
        )

# ------------------------------------------------------------
# optional meta: Reptile on teacher (kept compatible)
# ------------------------------------------------------------
class ReptileMetaLearner:
    def __init__(self, policy: RadiusPolicy, lr_outer: float = 1e-2):
        self.master = policy
        self.lr_outer = lr_outer

    @torch.no_grad()
    def step(self, task_policy: RadiusPolicy):
        for p_master, p_task in zip(self.master.parameters(), task_policy.parameters()):
            p_master.add_(self.lr_outer * (p_task - p_master))
