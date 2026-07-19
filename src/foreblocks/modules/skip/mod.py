"""foreblocks.modules.skip.mod.

Mixture-of-Depths (MoD) — per-token router for dynamic FLOP allocation.

Implements the MoD architecture: a learned per-token router dynamically allocates
compute across transformer layers, routing only the highest-scoring keep_rate
fraction of tokens through each block's attention and MLP while the rest route
around via residual. Features expert-choice top-k selection, budget scheduling
with annealing, and auxiliary BCE loss for causal predictor training.

Original paper:
    Ritter, S., Richards, B., Lillicrap, T., Humphreys, P. C., & Santoro, A.
    (2024).
    "Mixture-of-Depths: Dynamically allocating compute in transformer-based
    language models."
    arXiv:2404.02258 [[arXiv]](https://arxiv.org/abs/2404.02258)

Core API:
- MoDRouter: per-token or per-sequence router for dynamic FLOP allocation
- mod_topk_mask: expert-choice top-k routing mask (MoD §3.3)
- mod_routed_indices: convert keep mask to sorted routed indices
- mod_router_aux_loss: BCE auxiliary loss for causal router training (MoD §3.5)
- MoDBudgetScheduler: budget annealing with layer profiles
- LayerDropoutSchedule: per-layer dropout with depth profiles

"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Mixture-of-Depths (MoD)
# =============================================================================
class MoDRouter(nn.Module):
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
    """Expert-choice top-k routing (MoD).  [Ritter et al., 2024, §3.3]

    Returns a boolean keep mask ``[B,T]`` with exactly
    ``ceil(keep_rate * n_active)`` routed positions per sample, selected by
    the highest router scores.  Unlike fixed-capacity MoE dispatching this
    does not drop tokens — it simply routes fewer to deeper layers.

    Parameters
    ----------
    logits : torch.Tensor
        Router scores ``[B,T,1]`` or ``[B,T]``.
    keep_rate : float
        Fraction of active tokens to route forward (0..1).
    active_mask : torch.Tensor | None
        Optional ``[B,T]`` bool mask; only ``True`` positions participate.

    Returns
    -------
    torch.Tensor
        Boolean keep mask ``[B,T]``, ``True`` for tokens routed to this layer.
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
    """Return the fixed routed capacity for the current batch.

    Equal to ``max_b ceil(keep_rate * n_active_b)``.
    """
    if keep_mask.dim() != 2:
        raise ValueError(f"keep_mask must be [B,T], got {tuple(keep_mask.shape)}")
    if keep_mask.numel() == 0:
        return 0
    return int(keep_mask.sum(dim=1).max().item())


def mod_routed_indices(
    keep_mask: torch.Tensor,
    capacity: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a keep mask ``[B,T]`` into sorted routed indices ``[B,C]`` and slot mask ``[B,C]``.

    Useful for gathering routed token embeddings before passing them through
    the deeper layer.

    Parameters
    ----------
    keep_mask : torch.Tensor
        Boolean keep mask ``[B,T]``.
    capacity : int | None
        Fixed capacity.  If ``None``, uses ``mod_capacity(keep_mask)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(indices, slot_mask)`` where ``indices[b,c]`` holds the original
        token position and ``slot_mask[b,c]`` indicates a valid slot.
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
    """BCE auxiliary loss for training the MoD causal router [Ritter et al., 2024, §3.5].

    The predictor is a causal (autoregressive) MLP, but the top-k mask is
    computed non-causally (over the whole sequence).  This auxiliary loss
    trains the causal predictor to match the non-causal expert-choice mask.

    Parameters
    ----------
    logits : torch.Tensor
        Raw predictor scores ``[B,T,1]`` or ``[B,T]``.
    keep_mask : torch.Tensor
        Target boolean mask ``[B,T]`` (from non-causal top-k).
    active_mask : torch.Tensor | None
        Optional active token mask.

    Returns
    -------
    torch.Tensor
        Scalar BCE loss over active positions.
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


@dataclass
class LayerDropoutSchedule:
    num_layers: int
    base_dropout: float
    max_dropout: float | None = None
    profile: str = "flat"

    def get_dropout(self, layer_idx: int) -> float:
        """Get the dropout rate for a specific layer.

        Parameters
        ----------
        layer_idx : int
            Layer index (0-indexed, 0 is the first/shallowest layer).

        Returns
        -------
        float
            Dropout rate (clamped to [0, 1]).
        """
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(
                f"layer_idx={layer_idx} outside [0, {self.num_layers})"
            )
        if self.max_dropout is None or self.profile == "flat":
            return float(max(0.0, min(1.0, self.base_dropout)))

        depth = layer_idx / max(1, self.num_layers - 1)

        if self.profile not in {"deeper_more", "deeper_less"}:
            raise ValueError(f"Unknown profile: {self.profile}")
        # Both profiles interpolate from the configured shallow-layer value to
        # the configured deepest-layer value. The profile name documents the
        # intended ordering; validate it to catch inverted configurations.
        if self.profile == "deeper_more" and self.max_dropout < self.base_dropout:
            raise ValueError("deeper_more requires max_dropout >= base_dropout")
        if self.profile == "deeper_less" and self.max_dropout > self.base_dropout:
            raise ValueError("deeper_less requires max_dropout <= base_dropout")
        p = (1.0 - depth) * self.base_dropout + depth * self.max_dropout

        return float(max(0.0, min(1.0, p)))
