from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFeedForward(nn.Module):
    """
    State-of-the-art Mixture of Experts inspired by DeepSeek-MoE, GLaM, and Switch Transformer.

    Key improvements:
    - Shared expert for stability (DeepSeek-MoE)
    - Expert-parallel computation
    - Advanced load balancing with Z-loss
    - Capacity-based expert selection
    - Optimized memory usage
    - Better numerical stability
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        capacity_factor: float = 1.25,
        expert_dropout: float = 0.0,
        use_noisy_gating: bool = False,
        min_capacity: int = 4,
        use_swiglu: bool = True,
        activation: str = "gelu",
        # DeepSeek-MoE style improvements
        shared_expert_ratio: float = 0.25,  # Ratio of shared vs routed computation
        use_shared_expert: bool = True,
        load_balancing_loss_weight: float = 1e-2,
        z_loss_weight: float = 1e-3,
        router_init_std: float = 0.02,
        expert_parallel: bool = True,
        use_capacity_factor: bool = True,
        # Advanced features
        use_bias: bool = False,
        normalize_router_weights: bool = True,
        router_temperature: float = 1.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.capacity_factor = capacity_factor
        self.min_capacity = min_capacity
        self.expert_dropout = expert_dropout
        self.noise_eps = 1e-2
        self.use_noisy_gating = use_noisy_gating
        self.shared_expert_ratio = shared_expert_ratio
        self.use_shared_expert = use_shared_expert
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.z_loss_weight = z_loss_weight
        self.expert_parallel = expert_parallel
        self.use_capacity_factor = use_capacity_factor
        self.normalize_router_weights = normalize_router_weights
        self.router_temperature = router_temperature

        # Input normalization
        self.input_norm = nn.LayerNorm(d_model)

        # Router with advanced initialization
        self.router = nn.Linear(d_model, num_experts, bias=use_bias)
        if normalize_router_weights:
            nn.init.normal_(self.router.weight, mean=0.0, std=router_init_std)
        else:
            # Kaiming initialization for better gradient flow
            nn.init.kaiming_normal_(self.router.weight)

        # Shared expert (DeepSeek-MoE style)
        if use_shared_expert:
            shared_d_ff = int(d_ff * shared_expert_ratio)
            self.shared_expert = self._create_expert(
                d_model, shared_d_ff, dropout, use_swiglu, activation
            )
            # Adjust routed expert size
            routed_d_ff = d_ff - shared_d_ff
        else:
            self.shared_expert = None
            routed_d_ff = d_ff

        # Routed experts
        self.experts = nn.ModuleList(
            [
                self._create_expert(
                    d_model, routed_d_ff, dropout, use_swiglu, activation
                )
                for _ in range(num_experts)
            ]
        )

        # Expert capacity calculation
        if use_capacity_factor:
            self.expert_capacity = int(capacity_factor * d_model / num_experts)
        else:
            self.expert_capacity = None

        self.aux_loss = 0.0

    def _create_expert(
        self, d_model: int, d_ff: int, dropout: float, use_swiglu: bool, activation: str
    ) -> nn.Module:
        """Create expert with optimized architecture"""
        if use_swiglu:
            return OptimizedSwiGLUExpert(d_model, d_ff, dropout)
        else:
            return OptimizedFFNExpert(d_model, d_ff, dropout, activation)

    def _compute_router_probabilities(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute router probabilities with advanced techniques
        """
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.d_model)

        # Router computation with mixed precision
        with torch.amp.autocast("cuda", enabled=False):
            router_logits = self.router(x_flat.float()) / self.router_temperature

        # Apply expert dropout (DeepSeek technique)
        if self.training and self.expert_dropout > 0:
            dropout_mask = (
                torch.rand(self.num_experts, device=x.device) > self.expert_dropout
            )
            router_logits = router_logits.masked_fill(
                ~dropout_mask.unsqueeze(0), float("-inf")
            )

        # Add exploration noise
        if self.use_noisy_gating and self.training:
            noise = torch.normal(
                0, self.noise_eps, size=router_logits.shape, device=router_logits.device
            )
            router_logits = router_logits + noise

        # Compute probabilities
        router_probs = F.softmax(router_logits, dim=-1)

        return router_logits, router_probs

    def _expert_parallel_forward(
        self, x: torch.Tensor, router_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Expert-parallel computation for better efficiency (inspired by GLaM/PaLM)
        """
        batch_size = x.size(0)

        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_probs, dim=-1)  # Renormalize

        # Expert capacity enforcement
        if self.use_capacity_factor and self.expert_capacity:
            top_k_probs, top_k_indices = self._enforce_expert_capacity(
                top_k_probs, top_k_indices, batch_size
            )

        # Parallel expert computation
        expert_outputs = []
        for expert_id in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (top_k_indices == expert_id).any(dim=-1)

            if expert_mask.any():
                expert_input = x[expert_mask]
                expert_output = self.experts[expert_id](expert_input)
                expert_outputs.append((expert_mask, expert_output))
            else:
                expert_outputs.append((expert_mask, None))

        # Combine expert outputs with gating weights
        output = torch.zeros_like(x)
        for token_idx in range(batch_size):
            token_output = torch.zeros_like(x[token_idx])

            for k in range(self.top_k):
                expert_id = top_k_indices[token_idx, k]
                gate_weight = top_k_probs[token_idx, k]

                expert_mask, expert_output = expert_outputs[expert_id]
                if expert_output is not None and expert_mask[token_idx]:
                    # Find the position of this token in expert's input
                    token_pos = expert_mask[: token_idx + 1].sum() - 1
                    token_output += gate_weight * expert_output[token_pos]

            output[token_idx] = token_output

        return output

    def _vectorized_expert_forward_v2(
        self, x: torch.Tensor, router_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimized vectorized expert forward with better memory efficiency
        """
        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype

        # Top-k selection with better numerical stability
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize within top-k for better stability
        top_k_probs = F.softmax(top_k_probs / self.router_temperature, dim=-1)

        # Efficient expert dispatch using advanced indexing
        expanded_x = x.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, self.d_model)
        flat_expert_indices = top_k_indices.reshape(-1)
        flat_gates = top_k_probs.reshape(-1)

        # Create expert assignment matrix for parallel processing
        expert_assignment = torch.zeros(
            self.num_experts, expanded_x.size(0), device=device, dtype=torch.bool
        )
        for expert_id in range(self.num_experts):
            expert_assignment[expert_id] = flat_expert_indices == expert_id

        # Process all experts in parallel
        combined_output = torch.zeros_like(expanded_x)
        for expert_id in range(self.num_experts):
            expert_mask = expert_assignment[expert_id]
            if expert_mask.any():
                expert_input = expanded_x[expert_mask]
                expert_output = self.experts[expert_id](expert_input)
                combined_output[expert_mask] = expert_output.to(dtype)

        # Apply gating and aggregate
        weighted_output = combined_output * flat_gates.unsqueeze(-1)

        # Efficient aggregation using scatter_add
        output = torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)
        token_indices = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )
        output.scatter_add_(
            0, token_indices.unsqueeze(-1).expand(-1, self.d_model), weighted_output
        )

        return output

    def _enforce_expert_capacity(
        self, top_k_probs: torch.Tensor, top_k_indices: torch.Tensor, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enforce expert capacity to prevent load imbalance
        """
        # Count assignments per expert
        expert_counts = torch.zeros(self.num_experts, device=top_k_indices.device)
        for expert_id in range(self.num_experts):
            expert_counts[expert_id] = (top_k_indices == expert_id).sum()

        # If any expert exceeds capacity, redistribute
        max_capacity = min(self.expert_capacity, batch_size)
        if expert_counts.max() > max_capacity:
            # Simple capacity enforcement: mask out excess assignments
            for expert_id in range(self.num_experts):
                if expert_counts[expert_id] > max_capacity:
                    expert_mask = top_k_indices == expert_id
                    # Keep only the first max_capacity assignments
                    expert_positions = expert_mask.nonzero()[:max_capacity]
                    new_mask = torch.zeros_like(expert_mask)
                    new_mask[expert_positions[:, 0], expert_positions[:, 1]] = True
                    top_k_probs = top_k_probs.masked_fill(expert_mask & ~new_mask, 0.0)

        # Renormalize probabilities
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        return top_k_probs, top_k_indices

    def _compute_advanced_aux_loss(
        self,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute advanced auxiliary losses (Load balancing + Z-loss)
        """
        aux_loss = 0.0

        # 1. Load balancing loss (Switch Transformer style)
        if self.load_balancing_loss_weight > 0:
            # Expert usage frequency
            expert_usage = torch.zeros(self.num_experts, device=router_probs.device)
            for expert_id in range(self.num_experts):
                expert_usage[expert_id] = (top_k_indices == expert_id).float().mean()

            # Router probability means
            router_prob_means = router_probs.mean(dim=0)

            # Load balancing loss
            load_loss = (expert_usage * router_prob_means).sum() * self.num_experts
            aux_loss += self.load_balancing_loss_weight * load_loss

        # 2. Z-loss for router stability (PaLM/GLaM style)
        if self.z_loss_weight > 0:
            z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
            aux_loss += self.z_loss_weight * z_loss

        return aux_loss

    def forward(self, x: torch.Tensor, return_aux_loss: bool = True):
        """
        Forward pass with state-of-the-art MoE techniques
        """
        B, T, D = x.shape
        original_shape = x.shape

        # Input normalization
        x_norm = self.input_norm(x)
        x_flat = x_norm.view(-1, D)

        # Compute router probabilities
        router_logits, router_probs = self._compute_router_probabilities(x_norm)

        # Expert computation
        if self.expert_parallel:
            expert_output = self._vectorized_expert_forward_v2(x_flat, router_probs)
        else:
            # Use the original method for compatibility
            top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
            top_k_probs = F.softmax(top_k_probs, dim=-1)
            expert_output = self._vectorized_expert_forward_v2(x_flat, router_probs)

        # Shared expert computation (DeepSeek-MoE)
        if self.use_shared_expert:
            shared_output = self.shared_expert(x_flat)
            expert_output = expert_output + shared_output

        # Reshape output
        output = expert_output.view(original_shape)

        # Compute auxiliary losses
        if self.training and return_aux_loss:
            top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
            self.aux_loss = self._compute_advanced_aux_loss(
                router_logits, router_probs, top_k_indices
            )
        else:
            self.aux_loss = 0.0

        if return_aux_loss:
            return output, self.aux_loss
        else:
            return output


class OptimizedSwiGLUExpert(nn.Module):
    """Optimized SwiGLU expert for MoE"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Fused gate and up projections for efficiency
        self.gate_up_proj = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize with proper scaling
        nn.init.xavier_uniform_(self.gate_up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        return self.dropout(self.down_proj(hidden))


class OptimizedFFNExpert(nn.Module):
    """Optimized standard FFN expert"""

    def __init__(
        self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "swish":
            self.activation = F.silu
        else:
            self.activation = F.gelu

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


class _StandardFeedForwardBlock(nn.Module):
    def __init__(
        self, d_model, dim_ff, dropout=0.1, use_swiglu=True, activation="gelu"
    ):
        super().__init__()
        self.use_swiglu = use_swiglu
        self.dropout = nn.Dropout(dropout)

        if use_swiglu:
            swiglu_dim = int(dim_ff * 4 / 3)
            self.w1 = nn.Linear(d_model, swiglu_dim)
            self.w2 = nn.Linear(d_model, swiglu_dim)
            self.w3 = nn.Linear(swiglu_dim, d_model)
        else:
            self.linear1 = nn.Linear(d_model, dim_ff)
            self.linear2 = nn.Linear(dim_ff, d_model)
            self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        if self.use_swiglu:
            u = self.w1(x)
            v = self.w2(x)

            u_clamped = u.clamp(-30, 30)
            v_clamped = v.clamp(-30, 30)

            silu_u = F.silu(u_clamped)

            z = silu_u * v_clamped

            out = self.w3(z)

        else:
            out = self.linear2(self.dropout(self.activation(self.linear1(x))))

        return out


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        d_model,
        dim_ff,
        dropout=0.1,
        use_swiglu=True,
        activation="gelu",
        use_moe=False,
        num_experts=4,
        top_k=2,
        capacity_factor=1.5,
        expert_dropout=0.1,
    ):
        super().__init__()
        self.use_moe = use_moe
        self.use_swiglu = use_swiglu

        if use_moe:
            print("[FeedForwardBlock] Using Mixture-of-Experts")
            self.block = MoEFeedForward(
                d_model=d_model,
                d_ff=dim_ff,
                dropout=dropout,
                num_experts=num_experts,
                top_k=top_k,
                use_swiglu=use_swiglu,
                activation=activation,
                capacity_factor=capacity_factor,
                expert_dropout=expert_dropout,
            )
        else:
            print(
                "[FeedForwardBlock] Using standard FFN (SwiGLU)"
                if use_swiglu
                else f"[FeedForwardBlock] Using {activation.upper()}"
            )
            self.block = _StandardFeedForwardBlock(
                d_model=d_model,
                dim_ff=dim_ff,
                dropout=dropout,
                use_swiglu=use_swiglu,
                activation=activation,
            )

    def forward(self, x, return_aux_loss=False):
        return self.block(x)
