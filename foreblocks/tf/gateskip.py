from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Budget Scheduler
# ──────────────────────────────────────────────────────────────────────────────
class BudgetScheduler:
    def __init__(self, b_start: float = 1.0, b_end: float = 0.8, total_steps: Optional[int] = None):
        self.b_start, self.b_end = float(b_start), float(b_end)
        self.total_steps = total_steps
        self._step = 0

    def set_total_steps(self, total_steps: int):
        self.total_steps = int(total_steps)

    def step(self):
        self._step += 1

    def get_budget(self, current_step: Optional[int] = None) -> float:
        s = self._step if current_step is None else int(current_step)
        if self.total_steps is None or s >= self.total_steps:
            return self.b_end
        decay = (self.b_start - self.b_end) * (s / self.total_steps)
        return max(self.b_end, self.b_start - decay)

# ──────────────────────────────────────────────────────────────────────────────
# GateSkip primitives
# ──────────────────────────────────────────────────────────────────────────────
class ResidualGate(nn.Module):
    """Vector gate g(h_prev)=σ(W h_prev + b) applied to the module output o."""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.proj.bias, 5.0)
        self.register_buffer("_last_gbar", torch.tensor(0.0), persistent=False)

    def gate_scores(self, h_prev: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.proj(h_prev))          # [B,T,H]
        gbar = g.mean(dim=-1)                         # [B,T]
        self._last_gbar = gbar.mean()                 # for quick monitoring
        return g, gbar

    def forward(self, h_prev: torch.Tensor, o: torch.Tensor):
        g, gbar = self.gate_scores(h_prev)
        h_gated = h_prev + g * o
        return h_gated, g, gbar
@torch.no_grad()
def _quantile_threshold(gbar: torch.Tensor, keep) -> torch.Tensor:
    """
    Compute skip mask using a per-row threshold so that ~keep*T tokens are kept.
    Args:
        gbar: (B, T) token-level gate scores
        keep: float in [0,1] or 0D tensor; fraction of tokens to KEEP
    Returns:
        skip_mask: (B, T) bool (True = skip/copy-through)
    """
    B, T = gbar.shape
    device = gbar.device

    # Normalize keep to a Python float
    if isinstance(keep, torch.Tensor):
        keep = float(keep.item())
    else:
        keep = float(keep)

    if keep >= 1.0:
        return torch.zeros(B, T, dtype=torch.bool, device=device)
    if keep <= 0.0:
        return torch.ones(B, T, dtype=torch.bool, device=device)

    # How many to keep/skip per row
    k_keep = int(round(keep * T))
    k_keep = max(0, min(T, k_keep))
    skip_quota = T - k_keep

    if skip_quota <= 0:
        return torch.zeros(B, T, dtype=torch.bool, device=device)
    if skip_quota >= T:
        return torch.ones(B, T, dtype=torch.bool, device=device)

    # Find the threshold τ as the max among the smallest `skip_quota` elements per row.
    # Use topk on -gbar to select `skip_quota` smallest values efficiently.
    # topk returns values sorted in descending order; take the LAST to get the largest
    # among the selected smallest => τ.
    vals, _ = torch.topk(-gbar, k=skip_quota, dim=1, largest=True, sorted=True)  # shape (B, skip_quota)
    tau = -vals[:, -1]  # (B,)

    # Skip if gbar <= τ
    skip_mask = gbar <= tau.unsqueeze(1)
    return skip_mask

def gateskip_apply(
    enabled: bool,
    h_prev: torch.Tensor,
    o: torch.Tensor,
    gate: ResidualGate,
    budget: Optional[float],
    aux_l2_terms: List[torch.Tensor],
    lambda_s: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns (h_out, skip_mask or None).
    - If disabled: plain residual add.
    - If enabled & budget is None: gate only (no skipping).
    - If enabled & budget in (0,1]: gate + per-sample quantile skipping.
    """
    if not enabled:
        return h_prev + o, None

    h_gated, g, gbar = gate(h_prev, o)

    if lambda_s > 0:
        # defer scaling to the outer loss to keep this pure
        aux_l2_terms.append((g * g).mean())

    if budget is None or budget >= 1.0:
        return h_gated, None
    if budget <= 0.0:
        return h_prev, torch.ones_like(gbar, dtype=torch.bool)

    skip_mask = _quantile_threshold(gbar, float(budget))
    h_out = torch.where(skip_mask.unsqueeze(-1), h_prev, h_gated)
    return h_out, skip_mask

def apply_skip_to_kv(
    updated: Optional[Dict[str, torch.Tensor]],
    skip_mask: torch.Tensor,
    prev_layer_state: Optional[Dict[str, Dict[str, torch.Tensor]]],
    attn_type: str,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    For skipped tokens, copy KV from previous layer's cache.
    Shapes: k,v: [B, nh, T, hd]. skip_mask: [B,T] (bool).
    """
    if updated is None or prev_layer_state is None or attn_type not in prev_layer_state:
        return updated
    prev_kv = prev_layer_state[attn_type]
    skip_expanded = skip_mask.unsqueeze(1).unsqueeze(-1)  # [B,1,T,1]
    for key in ("k", "v"):
        if key in updated and key in prev_kv:
            # Ensure devices match
            if updated[key].device != prev_kv[key].device:
                prev_kv[key] = prev_kv[key].to(updated[key].device)
            updated[key] = torch.where(skip_expanded, prev_kv[key], updated[key])
    return updated
