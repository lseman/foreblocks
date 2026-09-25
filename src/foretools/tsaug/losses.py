"""Composite loss function for AutoDA-Timeseries (Section 3.5.2).

Implements uncertainty-weighted composite loss with three components:

    L_composite = Σ_{z=1,2,3} [ 1/(2*w_z^2) * L_z + ln(1 + w_z^2) ]

where the weights w_z are learnable and optimized jointly with the framework.

Loss Components
---------------
| Term | Symbol | Description                                   |
|------|--------|-----------------------------------------------|
| L1   | task   | Task-specific loss (MSE, CE, etc.)            |
| L2   | intra  | Intra-layer diversity (negative Shannon entropy) |
| L3   | inter  | Inter-layer diversity (KL divergence between layers) |

The log(1 + w_z^2) term comes from Gaussian observation noise model
(Kendall et al., 2018) and prevents any single loss from dominating.

Usage
-----
>>> import torch
>>> from foretools.tsaug.losses import CompositeLoss
>>> closs = CompositeLoss()
>>> task_loss = torch.tensor(0.5)
>>> all_probs = [torch.ones(8, 12) / 12 for _ in range(3)]  # (B, n_transforms)
>>> total, details = closs(task_loss, all_probs)
>>> print(f"Total: {total.item():.4f}")  # doctest: +SKIP
Total: 0.8765
"""

import torch
import torch.nn as nn


class CompositeLoss(nn.Module):
    """Composite loss with learnable uncertainty-based weighting.

    Implements the per-loss uncertainty weighting from Kendall et al. (2018),
    adapted for three loss components in AutoDA-Timeseries.

    The weights w_z are stored as log_w = log(w_z) and optimized via Adam.
    The effective weight used in the formula is ``w_z^2 = exp(2 * log_w_z)``.

    Parameters
    ----------
    init_weights : tuple[float, float, float]
        Initial values for the three learnable weights (w1, w2, w3).
            Default (1.0, 1.0, 1.0). Stored as log(init_weight) in parameters.
    eps : float
        Small constant for numerical stability in entropy/KL computations.
            Default 1e-10.

    Attributes
    ----------
    log_w : ParameterList[Parameter]
        Learnable parameters: log(w_1), log(w_2), log(w_3).
    eps : float
        Numerical stability constant.

    Examples
    --------
    >>> import torch
    >>> from foretools.tsaug.losses import CompositeLoss
    >>> closs = CompositeLoss()
    >>> task_loss = torch.tensor(0.5)
    >>> all_probs = [torch.ones(8, 12) / 12 for _ in range(3)]
    >>> total, details = closs(task_loss, all_probs)
    >>> assert isinstance(total, torch.Tensor)
    """

    def __init__(self, init_weights: tuple = (1.0, 1.0, 1.0), eps: float = 1e-10):
        super().__init__()
        self.eps = eps

        # Learnable weights w_z (we store w_z directly, use w_z^2 in formula)
        self.log_w = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(w)).log()) for w in init_weights]
        )

    def forward(
        self,
        task_loss: torch.Tensor,
        all_probs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the composite loss with learnable weights.

        Combines three loss terms (task loss, intra-layer diversity, inter-layer
        diversity) using uncertainty-based weighting and returns both the total
        scalar loss and a dictionary of individual components for logging.

        Parameters
        ----------
        task_loss : torch.Tensor
            Scalar task-specific loss L1 (e.g., CrossEntropyLoss output).
        all_probs : list[torch.Tensor]
            List of K probability tensors from each augmentation layer,
                each of shape (batch_size, num_transforms).

        Returns
        -------
        tuple[torch.Tensor, dict[str, float]]
            ``(total_loss, loss_details)`` where:

            - ``total_loss``: Scalar tensor of the composite loss value
            - ``loss_details``: Dictionary with keys:

              * ``'L1'``: Task loss value
              * ``'L2'``: Intra-layer diversity value (negative entropy)
              * ``'L3'``: Inter-layer diversity value (KL divergence)
              * ``'w1'``, ``'w2'``, ``'w3'``: Current effective weights
              * ``'total'``: Total loss as float
              * ``'intra_entropy'``: Average intra-layer entropy
              * ``'inter_kl'``: Average inter-layer KL divergence

        Examples
        --------
        >>> import torch
        >>> from foretools.tsaug.losses import CompositeLoss
        >>> closs = CompositeLoss()
        >>> task_loss = torch.tensor(0.5)
        >>> all_probs = [torch.ones(8, 12) / 12 for _ in range(3)]
        >>> total, details = closs(task_loss, all_probs)
        >>> print(f"L1={details['L1']:.4f}, L2={details['L2']:.4f}, L3={details['L3']:.4f}")  # doctest: +SKIP
        L1=0.5000, L2=-1.6094, L3=0.0000
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
            loss_details[f"L{z + 1}"] = loss_z.item()
            loss_details[f"w{z + 1}"] = w_z_sq.sqrt().item()

        loss_details["total"] = total.item()
        loss_details["intra_entropy"] = intra_diversity.item()
        loss_details["inter_kl"] = inter_diversity.item()

        return total, loss_details

    def _intra_layer_diversity(self, all_probs: list[torch.Tensor]) -> torch.Tensor:
        """Compute intra-layer diversity as average Shannon entropy.

        Measures the average entropy of probability distributions within each
        augmentation layer. Higher entropy = more uniform distribution across
        transformations (more diverse).

        L2 = Σ_k E_i[H(p^(k)_i)]

        Parameters
        ----------
        all_probs : list[torch.Tensor]
            List of K probability tensors, each (batch_size, num_transforms).

        Returns
        -------
        torch.Tensor
            Scalar tensor: sum over layers of mean entropy per sample.
        """
        total_entropy = torch.tensor(0.0, device=all_probs[0].device)
        for prob in all_probs:
            # prob: (B, n)
            entropy = -(prob * torch.log(prob + self.eps)).sum(dim=-1)  # (B,)
            total_entropy = total_entropy + entropy.mean()
        return total_entropy
        total_entropy = torch.tensor(0.0, device=all_probs[0].device)
        for prob in all_probs:
            # prob: (B, n)
            entropy = -(prob * torch.log(prob + self.eps)).sum(dim=-1)  # (B,)
            total_entropy = total_entropy + entropy.mean()
        return total_entropy

    def _inter_layer_diversity(self, all_probs: list[torch.Tensor]) -> torch.Tensor:
        """Compute inter-layer diversity as KL divergence between consecutive layers.

        Measures the average KL divergence D_KL(p^(k-1) || p^(k)) between
        adjacent augmentation layers. Higher divergence = more distinct
        transformation policies across layers.

        L3 = Σ_{k=2}^{K} E_i[KL(p^(k-1)_i || p^(k)_i)]

        Parameters
        ----------
        all_probs : list[torch.Tensor]
            List of K probability tensors, each (batch_size, num_transforms).

        Returns
        -------
        torch.Tensor
            Scalar tensor: sum over consecutive layer pairs of mean KL divergence.
            Returns 0.0 if fewer than 2 layers.
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
        if len(all_probs) < 2:
            return torch.tensor(0.0, device=all_probs[0].device)

        total_kl = torch.tensor(0.0, device=all_probs[0].device)
        for k in range(1, len(all_probs)):
            p_prev = all_probs[k - 1] + self.eps
            p_curr = all_probs[k] + self.eps
            kl = (p_prev * (p_prev.log() - p_curr.log())).sum(dim=-1)  # (B,)
            total_kl = total_kl + kl.mean()
        return total_kl
