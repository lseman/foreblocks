from dataclasses import dataclass

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Budget Scheduler
# ──────────────────────────────────────────────────────────────────────────────
class BudgetScheduler:
    def __init__(
        self,
        b_start: float = 1.0,
        b_end: float = 0.8,
        total_steps: int | None = None,
    ):
        self.b_start = float(b_start)
        self.b_end = float(b_end)
        self.total_steps = total_steps
        self._step = 0

    def set_total_steps(self, total_steps: int):
        self.total_steps = int(total_steps)

    def step(self):
        self._step += 1

    def get_budget(self, current_step: int | None = None) -> float:
        s = self._step if current_step is None else int(current_step)
        if self.total_steps is None or self.total_steps <= 0:
            return self.b_end
        if s >= self.total_steps:
            return self.b_end
        alpha = s / float(self.total_steps)
        return (1.0 - alpha) * self.b_start + alpha * self.b_end


# ──────────────────────────────────────────────────────────────────────────────
# GateSkip diagnostics
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class GateStats:
    gate_mean: torch.Tensor
    gate_active_mean: torch.Tensor
    token_keep_ratio: torch.Tensor
    token_skip_ratio: torch.Tensor
    budget_error: torch.Tensor
    smoothness: torch.Tensor


# ──────────────────────────────────────────────────────────────────────────────
# GateSkip primitives
# ──────────────────────────────────────────────────────────────────────────────
class ResidualGate(nn.Module):
    """
    Token-conditioned residual gate with scalar or per-feature gating.

    Gate:
        g = σ(W2 φ(W1 Norm(x)))

    Residual:
        h = h_prev + g ⊙ o

    Default output is token-wise [B,T,1], which is the safest form of gating.
    """

    def __init__(
        self,
        d_model: int,
        gate_dim: int = 1,
        hidden_dim: int | None = None,
        norm_type: str = "layernorm",
        layer_norm_eps: float = 1e-5,
        init_bias: float = 2.0,
        init_std: float = 0.02,
    ):
        super().__init__()
        if gate_dim not in (1, d_model):
            raise ValueError("ResidualGate gate_dim must be 1 or equal to d_model")
        if norm_type not in ("layernorm", "identity"):
            raise ValueError("norm_type must be 'layernorm' or 'identity'")

        self.d_model = int(d_model)
        self.gate_dim = int(gate_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else int(d_model)

        if norm_type == "layernorm":
            self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        else:
            self.norm = nn.Identity()

        self.fc1 = nn.Linear(d_model, self.hidden_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, self.gate_dim, bias=True)

        nn.init.normal_(self.fc1.weight, mean=0.0, std=init_std)
        nn.init.constant_(self.fc1.bias, 0.0)

        # Safer than 5.0: starts mostly open, but not fully saturated.
        nn.init.normal_(self.fc2.weight, mean=0.0, std=init_std)
        nn.init.constant_(self.fc2.bias, float(init_bias))

        self.register_buffer("_last_gate_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer(
            "_last_gate_active_mean", torch.tensor(0.0), persistent=False
        )

    def gate_scores(
        self,
        h_prev: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            g:    [B,T,gate_dim]
            gbar: [B,T] token-level scalar importance
        """
        x = self.norm(h_prev)
        x = self.act(self.fc1(x))
        g = torch.sigmoid(self.fc2(x))

        if self.gate_dim == 1:
            gbar = g.squeeze(-1)
        else:
            gbar = g.mean(dim=-1)

        with torch.no_grad():
            self._last_gate_mean.copy_(g.mean().detach())
            if active_mask is None:
                self._last_gate_active_mean.copy_(gbar.mean().detach())
            else:
                am = active_mask.to(device=gbar.device, dtype=torch.bool)
                denom = am.sum().clamp_min(1)
                active_mean = gbar.masked_select(am).sum() / denom
                self._last_gate_active_mean.copy_(active_mean.detach())

        return g, gbar

    def forward(
        self,
        h_prev: torch.Tensor,
        o: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g, gbar = self.gate_scores(h_prev, active_mask=active_mask)
        h_gated = h_prev + g * o
        return h_gated, g, gbar


def _normalize_active_mask(
    active_mask: torch.Tensor | None,
    ref: torch.Tensor,
) -> torch.Tensor:
    B, T = ref.shape[:2]
    device = ref.device
    if active_mask is None:
        return torch.ones(B, T, dtype=torch.bool, device=device)
    active_mask = active_mask.to(device=device, dtype=torch.bool)
    if active_mask.shape != (B, T):
        raise ValueError(
            f"active_mask shape {tuple(active_mask.shape)} must be {(B, T)}"
        )
    return active_mask


@torch.no_grad()
def _exact_topk_keep_mask(
    gbar: torch.Tensor,
    keep: float,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Returns a per-row boolean keep mask using exact top-k over active positions.

    Args:
        gbar:        [B,T] token scores (higher = keep)
        keep:        fraction in [0,1]
        active_mask: [B,T] bool; only True positions participate in budget
    Returns:
        keep_mask:   [B,T] bool, True where token is kept through the expensive path
    """
    B, T = gbar.shape
    device = gbar.device

    keep = float(keep)
    if keep < 0.0 or keep > 1.0:
        raise ValueError(f"keep must be in [0,1], got {keep}")

    active_mask = _normalize_active_mask(active_mask, gbar)
    keep_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    if keep >= 1.0:
        return active_mask
    if keep <= 0.0:
        return keep_mask

    neg_inf = torch.finfo(gbar.dtype).min
    masked_scores = torch.where(active_mask, gbar, torch.full_like(gbar, neg_inf))

    for b in range(B):
        active = active_mask[b]
        n_active = int(active.sum().item())
        if n_active == 0:
            continue

        k = int(round(keep * n_active))
        k = max(0, min(k, n_active))

        if k == 0:
            continue
        if k == n_active:
            keep_mask[b, active] = True
            continue

        scores_b = masked_scores[b]
        _, idx = torch.topk(scores_b, k=k, dim=0, largest=True, sorted=False)
        row_keep = torch.zeros(T, dtype=torch.bool, device=device)
        row_keep[idx] = True
        row_keep &= active
        keep_mask[b] = row_keep

    return keep_mask


def _compute_gate_smoothness(
    gbar: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Mean absolute temporal variation over adjacent active positions.
    """
    B, T = gbar.shape
    if T <= 1:
        return gbar.new_zeros(())

    active_mask = _normalize_active_mask(active_mask, gbar)
    pair_mask = active_mask[:, 1:] & active_mask[:, :-1]

    diffs = (gbar[:, 1:] - gbar[:, :-1]).abs()
    denom = pair_mask.sum().clamp_min(1)
    return diffs.masked_select(pair_mask).sum() / denom


def _compute_gate_stats(
    gbar: torch.Tensor,
    keep_mask: torch.Tensor | None,
    budget: float | None,
    active_mask: torch.Tensor | None = None,
) -> GateStats:
    active_mask = _normalize_active_mask(active_mask, gbar)

    denom_active = active_mask.sum().clamp_min(1)
    gate_mean = gbar.mean()
    gate_active_mean = gbar.masked_select(active_mask).sum() / denom_active

    if keep_mask is None:
        token_keep_ratio = gate_active_mean.new_tensor(1.0)
        token_skip_ratio = gate_active_mean.new_tensor(0.0)
    else:
        keep_active = keep_mask & active_mask
        keep_count = keep_active.sum()
        token_keep_ratio = keep_count.to(gbar.dtype) / denom_active.to(gbar.dtype)
        token_skip_ratio = 1.0 - token_keep_ratio

    if budget is None:
        budget_error = gate_active_mean.new_zeros(())
    else:
        budget_error = token_keep_ratio - float(budget)

    smoothness = _compute_gate_smoothness(gbar, active_mask=active_mask)

    return GateStats(
        gate_mean=gate_mean,
        gate_active_mean=gate_active_mean,
        token_keep_ratio=token_keep_ratio,
        token_skip_ratio=token_skip_ratio,
        budget_error=budget_error,
        smoothness=smoothness,
    )


def gateskip_apply(
    enabled: bool,
    h_prev: torch.Tensor,
    o: torch.Tensor,
    gate: ResidualGate,
    budget: float | None,
    aux_terms: list[torch.Tensor],
    lambda_s: float,
    active_mask: torch.Tensor | None = None,
    lambda_budget: float = 0.0,
    lambda_smooth: float = 0.0,
    warmup_soft_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, GateStats]:
    """
    Returns:
        h_out,
        skip_mask or None,
        stats

    Behavior:
      - disabled: plain residual add
      - enabled + budget is None: soft gate only
      - enabled + budget in [0,1]: soft gate + exact top-k keep
      - warmup_soft_only=True: ignore hard skipping even if budget is set

    Regularization hooks appended to aux_terms:
      - lambda_s      * mean(g)                 (sparsity / compute pressure)
      - lambda_budget * (realized_keep-budget)^2
      - lambda_smooth * temporal_gate_smoothness
    """
    active_mask = _normalize_active_mask(active_mask, h_prev)

    if not enabled:
        h_out = h_prev + o
        stats = _compute_gate_stats(
            gbar=torch.ones(
                h_prev.shape[0],
                h_prev.shape[1],
                device=h_prev.device,
                dtype=h_prev.dtype,
            ),
            keep_mask=active_mask,
            budget=1.0,
            active_mask=active_mask,
        )
        return h_out, None, stats

    h_gated, g, gbar = gate(h_prev, o, active_mask=active_mask)

    # Inactive positions are always copied through.
    h_gated = torch.where(active_mask.unsqueeze(-1), h_gated, h_prev)

    keep_mask: torch.Tensor | None = None
    if (budget is not None) and (not warmup_soft_only):
        budget = float(budget)
        if budget >= 1.0:
            keep_mask = active_mask
        elif budget <= 0.0:
            keep_mask = torch.zeros_like(active_mask)
        else:
            keep_mask = _exact_topk_keep_mask(
                gbar, keep=budget, active_mask=active_mask
            )

    if keep_mask is None:
        h_out = h_gated
        skip_mask = ~active_mask
    else:
        h_out = torch.where(keep_mask.unsqueeze(-1), h_gated, h_prev)
        skip_mask = ~keep_mask

    stats = _compute_gate_stats(
        gbar=gbar,
        keep_mask=keep_mask,
        budget=budget,
        active_mask=active_mask,
    )

    if lambda_s > 0.0:
        aux_terms.append(lambda_s * stats.gate_active_mean)

    if lambda_budget > 0.0 and budget is not None:
        aux_terms.append(lambda_budget * (stats.budget_error**2))

    if lambda_smooth > 0.0:
        aux_terms.append(lambda_smooth * stats.smoothness)

    return h_out, skip_mask


def apply_skip_to_kv(
    updated: dict[str, torch.Tensor] | None,
    skip_mask: torch.Tensor,
    prev_layer_state: dict[str, dict[str, torch.Tensor]] | None,
    attn_type: str,
) -> dict[str, torch.Tensor] | None:
    """
    For skipped tokens, copy KV from previous layer cache.

    Shapes:
        k,v:      [B, nh, T, hd]
        skip_mask [B, T] bool  (True = skip / copy-through)

    Note:
        This is a cache-level optimization. It should be validated per attention
        variant, since some attention mechanisms may tolerate stale KV better
        than others.
    """
    if updated is None:
        return updated
    if prev_layer_state is None:
        return updated
    if attn_type not in prev_layer_state:
        return updated

    prev_kv = prev_layer_state[attn_type]
    skip_expanded = skip_mask.unsqueeze(1).unsqueeze(-1)  # [B,1,T,1]

    for key in ("k", "v"):
        if key not in updated or key not in prev_kv:
            continue

        prev_tensor = prev_kv[key]
        cur_tensor = updated[key]

        if prev_tensor.device != cur_tensor.device:
            prev_tensor = prev_tensor.to(cur_tensor.device)
        if prev_tensor.dtype != cur_tensor.dtype:
            prev_tensor = prev_tensor.to(cur_tensor.dtype)

        updated[key] = torch.where(skip_expanded, prev_tensor, cur_tensor)

    return updated
