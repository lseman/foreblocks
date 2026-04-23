import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Mixture-of-Depths (MoD)
# =============================================================================
class MoDRouter(nn.Module):
    """
    MoD router that emits a scalar score per token.

    mode:
      - "token": scores [B,T,1]
      - "seq":   scores [B,1,1] (legacy, not MoD-correct)
    """

    def __init__(
        self,
        d_model: int,
        mode: str = "token",  # "token" | "seq"
        hidden: int = 0,  # 0 => linear
        use_norm: bool = False,
        init_bias: float = 0.0,
    ):
        super().__init__()
        if mode not in ("token", "seq"):
            raise ValueError(f"MoDRouter.mode must be 'token' or 'seq', got {mode}")
        self.mode = mode
        self.norm = nn.LayerNorm(d_model) if use_norm else nn.Identity()

        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.net = nn.Linear(d_model, 1)

        if isinstance(self.net, nn.Linear) and self.net.bias is not None:
            nn.init.constant_(self.net.bias, init_bias)
        elif isinstance(self.net, nn.Sequential):
            last = self.net[-1]
            if isinstance(last, nn.Linear) and last.bias is not None:
                nn.init.constant_(last.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,D]
        returns scores:
          token => [B,T,1]
          seq   => [B,1,1]
        """
        x = self.norm(x)
        if self.mode == "seq":
            x = x.mean(dim=1, keepdim=True)  # [B,1,D]
        return self.net(x)

    @staticmethod
    def predictor_keep_mask(
        logits: torch.Tensor,
        active_mask: torch.Tensor | None = None,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        scores = logits.squeeze(-1) if logits.dim() == 3 else logits
        if active_mask is None:
            active_mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            active_mask = active_mask.to(device=scores.device, dtype=torch.bool)
        return active_mask & (scores > threshold)


def _normalize_active_mask(
    scores: torch.Tensor,
    active_mask: torch.Tensor | None,
) -> torch.Tensor:
    if active_mask is None:
        return torch.ones_like(scores, dtype=torch.bool)
    active_mask = active_mask.to(device=scores.device, dtype=torch.bool)
    if active_mask.shape != scores.shape:
        raise ValueError(
            f"active_mask shape {tuple(active_mask.shape)} must match scores "
            f"{tuple(scores.shape)}"
        )
    return active_mask


def mod_topk_mask(
    logits: torch.Tensor,
    keep_rate: float,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Expert-choice top-k routing used by Mixture-of-Depths.

    Returns a boolean keep mask [B, T] with exactly ceil(keep_rate * active_tokens)
    routed positions per sample (clamped by the number of active tokens).
    """
    scores = logits.squeeze(-1) if logits.dim() == 3 else logits
    active_mask = _normalize_active_mask(scores, active_mask)

    keep_rate = float(keep_rate)
    if keep_rate <= 0.0:
        return torch.zeros_like(scores, dtype=torch.bool)
    if keep_rate >= 1.0:
        return active_mask

    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    for b in range(scores.size(0)):
        active_idx = torch.nonzero(active_mask[b], as_tuple=False).flatten()
        n_active = int(active_idx.numel())
        if n_active == 0:
            continue
        k = int(math.ceil(keep_rate * n_active))
        k = max(0, min(n_active, k))
        if k == 0:
            continue
        chosen = torch.topk(scores[b, active_idx], k=k, dim=0, sorted=False).indices
        keep_mask[b, active_idx[chosen]] = True
    return keep_mask


def mod_capacity(
    keep_mask: torch.Tensor,
) -> int:
    """Return the fixed routed capacity for the current batch."""
    if keep_mask.dim() != 2:
        raise ValueError(f"keep_mask must be [B,T], got {tuple(keep_mask.shape)}")
    if keep_mask.numel() == 0:
        return 0
    return int(keep_mask.sum(dim=1).max().item())


def mod_routed_indices(
    keep_mask: torch.Tensor,
    capacity: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a keep mask [B,T] into sorted routed indices [B,C] and slot mask [B,C].
    """
    if keep_mask.dim() != 2:
        raise ValueError(f"keep_mask must be [B,T], got {tuple(keep_mask.shape)}")
    B, T = keep_mask.shape
    C = mod_capacity(keep_mask) if capacity is None else int(capacity)
    if C < 0:
        raise ValueError(f"capacity must be >= 0, got {C}")
    indices = torch.zeros(B, C, dtype=torch.long, device=keep_mask.device)
    slot_mask = torch.zeros(B, C, dtype=torch.bool, device=keep_mask.device)
    if C == 0:
        return indices, slot_mask

    for b in range(B):
        idx = torch.nonzero(keep_mask[b], as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        idx = torch.sort(idx).values
        n = min(C, int(idx.numel()))
        indices[b, :n] = idx[:n]
        slot_mask[b, :n] = True
        if n < C:
            indices[b, n:] = idx[0]
    return indices, slot_mask


def mod_router_aux_loss(
    logits: torch.Tensor,
    keep_mask: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    BCE auxiliary used to train the causal router predictor from non-causal top-k
    selections, following the paper's sampling workaround.
    """
    scores = logits.squeeze(-1) if logits.dim() == 3 else logits
    active_mask = _normalize_active_mask(scores, active_mask)
    targets = keep_mask.to(dtype=scores.dtype)
    if active_mask.any():
        return F.binary_cross_entropy_with_logits(
            scores[active_mask], targets[active_mask]
        )
    return scores.new_zeros(())


@dataclass
class MoDBudgetScheduler:
    """
    Provides a target keep-rate per layer (0..1), optionally annealed.

    Typical use:
      start_keep=1.0, end_keep=0.7, warmup then decay.

    layer_profile:
      - "flat": same keep-rate for all layers
      - "deeper_more": keep deeper layers a bit more
      - "deeper_less": keep deeper layers a bit less
    """

    num_layers: int
    start_keep: float = 1.0
    end_keep: float = 0.85
    warmup_steps: int = 0
    total_steps: int = 50_000
    layer_profile: str = "flat"

    _step: int = 0

    def step(self) -> None:
        self._step += 1

    def _progress(self) -> float:
        if self.total_steps <= 0:
            return 1.0
        s = max(0, self._step - self.warmup_steps)
        return min(1.0, s / max(1, self.total_steps))

    def get_keep_rate(self, layer_idx: int) -> float:
        p = self._progress()
        base = (1.0 - p) * self.start_keep + p * self.end_keep
        base = float(max(0.0, min(1.0, base)))

        if self.layer_profile == "flat":
            return base

        depth = (layer_idx + 1) / max(1, self.num_layers)
        if self.layer_profile == "deeper_more":
            scale = 0.8 + 0.4 * depth  # [0.8, 1.2]
        elif self.layer_profile == "deeper_less":
            scale = 1.2 - 0.4 * depth  # [1.2, 0.8]
        else:
            raise ValueError(f"Unknown layer_profile: {self.layer_profile}")

        return float(max(0.0, min(1.0, base * scale)))
