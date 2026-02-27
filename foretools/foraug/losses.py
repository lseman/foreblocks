"""
Composite loss function for AutoDA-Timeseries (Section 3.5.2).

L_composite = sum_{z=1,2,3} [ 1/(2*w_z^2) * L_z + ln(1 + w_z^2) ]

where:
  L1 = task-specific loss (MSE, CE, etc.)
  L2 = intra-layer diversity loss (Shannon entropy of probabilities)
  L3 = inter-layer diversity loss (KL divergence between layers)
  w_z = learnable weights for loss balancing
"""

import torch
import torch.nn as nn
from typing import List


class CompositeLoss(nn.Module):
    """Composite loss with learnable uncertainty-based weighting.

    Implements Eq. 8 from the paper, combining task loss with diversity
    regularization terms.

    Args:
        init_weights: Initial values for learnable weights (w1, w2, w3).
        eps: Small constant for numerical stability.
    """

    def __init__(self, init_weights: tuple = (1.0, 1.0, 1.0), eps: float = 1e-10):
        super().__init__()
        self.eps = eps

        # Learnable weights w_z (we store w_z directly, use w_z^2 in formula)
        self.log_w = nn.ParameterList([
            nn.Parameter(torch.tensor(float(w)).log())
            for w in init_weights
        ])

    def forward(
        self,
        task_loss: torch.Tensor,
        all_probs: List[torch.Tensor],
    ) -> dict:
        """Compute composite loss.

        Args:
            task_loss: Scalar task-specific loss (L1).
            all_probs: List of K probability tensors, each (B, n).

        Returns:
            Dictionary with total loss and individual components.
        """
        # L2: Intra-layer diversity loss (Eq. 9-10)
        # Maximize entropy within each layer -> minimize negative entropy
        intra_diversity = self._intra_layer_diversity(all_probs)

        # L3: Inter-layer diversity loss (Eq. 11)
        # Encourage different distributions across layers
        inter_diversity = self._inter_layer_diversity(all_probs)

        # Compute weighted composite loss (Eq. 8)
        losses = [task_loss, -intra_diversity, -inter_diversity]
        total = torch.tensor(0.0, device=task_loss.device)

        loss_details = {}
        for z, (loss_z, log_w_z) in enumerate(zip(losses, self.log_w)):
            w_z_sq = log_w_z.exp() ** 2
            weighted = 0.5 / w_z_sq * loss_z + torch.log(1.0 + w_z_sq)
            total = total + weighted
            loss_details[f"L{z+1}"] = loss_z.item()
            loss_details[f"w{z+1}"] = w_z_sq.sqrt().item()

        loss_details["total"] = total.item()
        loss_details["intra_entropy"] = intra_diversity.item()
        loss_details["inter_kl"] = inter_diversity.item()

        return total, loss_details

    def _intra_layer_diversity(self, all_probs: List[torch.Tensor]) -> torch.Tensor:
        """Compute intra-layer diversity as average Shannon entropy (Eq. 9-10).

        H(p^(k)_i) = -sum_j p^(k)_{i,j} * log(p^(k)_{i,j} + eps)
        L2 = sum_k E_i[H(p^(k)_i)]
        """
        total_entropy = torch.tensor(0.0, device=all_probs[0].device)
        for prob in all_probs:
            # prob: (B, n)
            entropy = -(prob * torch.log(prob + self.eps)).sum(dim=-1)  # (B,)
            total_entropy = total_entropy + entropy.mean()
        return total_entropy

    def _inter_layer_diversity(self, all_probs: List[torch.Tensor]) -> torch.Tensor:
        """Compute inter-layer diversity as KL divergence between consecutive layers (Eq. 11).

        L3 = sum_{k=2}^{K} E_i[KL(p^(k-1)_i || p^(k)_i)]
        """
        if len(all_probs) < 2:
            return torch.tensor(0.0, device=all_probs[0].device)

        total_kl = torch.tensor(0.0, device=all_probs[0].device)
        for k in range(1, len(all_probs)):
            p_prev = all_probs[k - 1] + self.eps
            p_curr = all_probs[k] + self.eps
            kl = (p_prev * (p_prev.log() - p_curr.log())).sum(dim=-1)  # (B,)
            total_kl = total_kl + kl.mean()
        return total_kl
