import torch
import torch.nn as nn
from typing import Tuple
from dataclasses import dataclass

# =============================================================================
# Mixture-of-Depths (MoD) / Dynamic Layer Skipping
# =============================================================================
class LayerGate(nn.Module):
    """
    Predicts keep probability logits for a layer.

    mode:
      - "token": logits [B,T,1]
      - "seq":   logits [B,1,1] (whole sample / sequence)
    """

    def __init__(
        self,
        d_model: int,
        mode: str = "token",  # "token" | "seq"
        hidden: int = 0,      # 0 => linear
        use_norm: bool = True,
        init_bias: float = 2.0,  # positive => mostly keep at init
    ):
        super().__init__()
        if mode not in ("token", "seq"):
            raise ValueError(f"LayerGate.mode must be 'token' or 'seq', got {mode}")
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

        # bias to keep
        if isinstance(self.net, nn.Linear) and self.net.bias is not None:
            nn.init.constant_(self.net.bias, init_bias)
        elif isinstance(self.net, nn.Sequential):
            last = self.net[-1]
            if isinstance(last, nn.Linear) and last.bias is not None:
                nn.init.constant_(last.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,D]
        returns logits:
          token => [B,T,1]
          seq   => [B,1,1]
        """
        x = self.norm(x)
        if self.mode == "seq":
            x = x.mean(dim=1, keepdim=True)  # [B,1,D]
        return self.net(x)

    @staticmethod
    def sample_straight_through(
        logits: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = True,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          keep_prob: sigmoid(logits)
          keep_mask: straight-through (hard) or soft mask
        """
        keep_prob = torch.sigmoid(logits)

        if not hard:
            return keep_prob, keep_prob

        # Gumbel-Sigmoid straight-through
        u = torch.rand_like(logits).clamp_(eps, 1.0 - eps)
        g = torch.log(u) - torch.log(1.0 - u)
        y = torch.sigmoid((logits + g) / max(temperature, eps))
        y_hard = (y > 0.5).to(y.dtype)
        y_st = y_hard.detach() - y.detach() + y
        return keep_prob, y_st


@dataclass
class LayerBudgetScheduler:
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

