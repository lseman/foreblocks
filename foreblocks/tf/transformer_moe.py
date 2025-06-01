import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

################################################################################
# Mixture of Experts (MoE) implementation using Triton for optimized performance
################################################################################


@triton.jit
def swiglu_kernel(
    x_ptr,
    gate_up_weight_ptr,
    down_weight_ptr,
    out_ptr,
    N,
    D_MODEL,
    D_FF,
    stride_x,
    stride_out,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < N
    mask_k = offs_k < D_MODEL

    x_ptrs = x_ptr + offs_m[:, None] * stride_x + offs_k[None, :]
    x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

    acc_gate = tl.zeros((BLOCK_M, D_FF), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, D_FF), dtype=tl.float32)

    for k in range(0, D_MODEL, BLOCK_K):
        offs_k_inner = k + tl.arange(0, BLOCK_K)
        mask_k_inner = offs_k_inner < D_MODEL

        x_chunk = tl.load(
            x_ptr + offs_m[:, None] * stride_x + offs_k_inner[None, :],
            mask=mask_m[:, None] & mask_k_inner[None, :],
            other=0.0,
        )

        gate_w = tl.load(
            gate_up_weight_ptr + offs_k_inner[:, None] + tl.arange(0, D_FF)[None, :],
            mask=mask_k_inner[:, None],
            other=0.0,
        )
        up_w = tl.load(
            gate_up_weight_ptr
            + (D_FF + offs_k_inner[:, None])
            + tl.arange(0, D_FF)[None, :],
            mask=mask_k_inner[:, None],
            other=0.0,
        )

        acc_gate += tl.dot(x_chunk, gate_w)
        acc_up += tl.dot(x_chunk, up_w)

    gate_silu = acc_gate / (1.0 + tl.exp(-acc_gate))
    hidden = gate_silu * acc_up

    acc_out = tl.zeros((BLOCK_M, D_MODEL), dtype=tl.float32)
    for k in range(0, D_FF, BLOCK_K):
        offs_k_inner = k + tl.arange(0, BLOCK_K)
        mask_k_inner = offs_k_inner < D_FF

        hidden_chunk = hidden[:, offs_k_inner]
        down_w = tl.load(
            down_weight_ptr + offs_k_inner[:, None] + tl.arange(0, D_MODEL)[None, :],
            mask=mask_k_inner[:, None],
            other=0.0,
        )

        acc_out += tl.dot(hidden_chunk, down_w)

    out_ptrs = out_ptr + offs_m[:, None] * stride_out + tl.arange(0, D_MODEL)[None, :]
    tl.store(out_ptrs, acc_out, mask=mask_m[:, None])


def triton_swiglu_forward(x, gate_up_weight, down_weight):
    N, D_MODEL = x.shape
    D_FF = gate_up_weight.size(1) // 2

    out = torch.empty((N, D_MODEL), device=x.device, dtype=x.dtype)

    BLOCK_M = 32
    BLOCK_K = min(128, triton.next_power_of_2(D_MODEL))
    grid = (triton.cdiv(N, BLOCK_M),)

    swiglu_kernel[grid](
        x,
        gate_up_weight,
        down_weight,
        out,
        N,
        D_MODEL,
        D_FF,
        x.stride(0),
        out.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )
    return out


class TritonMoEDispatcher:
    def __init__(self, d_model: int, top_k: int, block_d: int = 64):
        self.d_model = d_model
        self.top_k = top_k
        self.block_d = triton.next_power_of_2(block_d)
        self._buffers = {}

    @staticmethod
    def compute_expert_row_indices(top_k_indices, expert_counts):
        N, K = top_k_indices.shape
        device = top_k_indices.device

        expert_offsets = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.long), expert_counts.cumsum(0)]
        )

        flat_indices = top_k_indices.view(-1)
        sort_indices = torch.argsort(flat_indices, stable=True)

        row_indices = torch.empty(N * K, device=device, dtype=torch.long)

        offset = 0
        for expert_id in range(expert_counts.size(0)):
            count = expert_counts[expert_id].item()
            if count > 0:
                row_indices[sort_indices[offset : offset + count]] = torch.arange(
                    expert_offsets[expert_id],
                    expert_offsets[expert_id + 1],
                    device=device,
                    dtype=torch.long,
                )
                offset += count

        return row_indices.view(N, K)

    @staticmethod
    @triton.jit
    def moe_dispatch_kernel(
        x_ptr,
        gates_ptr,
        indices_ptr,
        expert_indices_ptr,
        expert_input_ptr,
        N,
        D,
        K,
        stride_x,
        stride_gates,
        stride_indices,
        stride_expert_indices,
        stride_output,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_d = tl.arange(0, BLOCK_D)
        mask = offs_d < D

        x_ptrs = x_ptr + pid * stride_x + offs_d
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        for k in range(K):
            gate = tl.load(gates_ptr + pid * stride_gates + k)
            expert_row = tl.load(expert_indices_ptr + pid * stride_expert_indices + k)

            weighted_x = x * gate
            out_ptrs = expert_input_ptr + expert_row * stride_output + offs_d
            tl.store(out_ptrs, weighted_x, mask=mask)

    def dispatch(self, x, top_k_probs, top_k_indices):
        N, D = x.shape
        K = self.top_k

        flat_indices = top_k_indices.view(-1)
        expert_counts = torch.bincount(flat_indices, minlength=top_k_probs.size(-1))
        expert_row_indices = self.compute_expert_row_indices(
            top_k_indices, expert_counts
        )

        total_rows = expert_counts.sum().item()
        key = (total_rows, D, x.device, x.dtype)

        if key not in self._buffers or self._buffers[key].size(0) < total_rows:
            self._buffers[key] = torch.empty(
                (total_rows, D), device=x.device, dtype=x.dtype
            )

        expert_input = self._buffers[key][:total_rows]

        if TRITON_AVAILABLE and x.is_cuda and N >= 32:
            self.moe_dispatch_kernel[(N,)](
                x_ptr=x,
                gates_ptr=top_k_probs,
                indices_ptr=top_k_indices,
                expert_indices_ptr=expert_row_indices,
                expert_input_ptr=expert_input,
                N=N,
                D=D,
                K=K,
                stride_x=x.stride(0),
                stride_gates=top_k_probs.stride(0),
                stride_indices=top_k_indices.stride(0),
                stride_expert_indices=expert_row_indices.stride(0),
                stride_output=expert_input.stride(0),
                BLOCK_D=self.block_d,
            )
        else:
            for i in range(N):
                for k in range(K):
                    gate = top_k_probs[i, k]
                    expert_row = expert_row_indices[i, k]
                    expert_input[expert_row] = x[i] * gate

        return expert_input, expert_row_indices


class MoEFeedForward(nn.Module):
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
        shared_expert_ratio: float = 0.25,
        use_shared_expert: bool = True,
        load_balancing_loss_weight: float = 1e-2,
        z_loss_weight: float = 1e-3,
        router_init_std: float = 0.02,
        expert_parallel: bool = True,
        use_capacity_factor: bool = True,
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
        self.router_temperature = router_temperature

        self._needs_load_balancing = load_balancing_loss_weight > 0
        self._needs_z_loss = z_loss_weight > 0
        self._needs_expert_dropout = expert_dropout > 0
        self._needs_capacity_enforcement = use_capacity_factor and capacity_factor > 0
        self._router_temp_not_one = abs(router_temperature - 1.0) > 1e-6

        self.input_norm = nn.LayerNorm(d_model)
        self.router = nn.Linear(d_model, num_experts, bias=use_bias)

        if normalize_router_weights:
            nn.init.normal_(self.router.weight, mean=0.0, std=router_init_std)
        else:
            nn.init.kaiming_normal_(self.router.weight, mode="fan_in")

        if use_shared_expert:
            shared_d_ff = int(d_ff * shared_expert_ratio)
            self.shared_expert = self._create_expert(
                d_model, shared_d_ff, dropout, use_swiglu, activation
            )
            routed_d_ff = d_ff - shared_d_ff
        else:
            self.shared_expert = None
            routed_d_ff = d_ff

        self.experts = nn.ModuleList(
            [
                self._create_expert(
                    d_model, routed_d_ff, dropout, use_swiglu, activation
                )
                for _ in range(num_experts)
            ]
        )

        if self._needs_capacity_enforcement:
            self.expert_capacity = max(
                min_capacity, int(capacity_factor * d_model / num_experts)
            )
        else:
            self.expert_capacity = None

        self.aux_loss = 0.0
        self.dispatcher = TritonMoEDispatcher(d_model=d_model, top_k=top_k)

    def _create_expert(
        self, d_model: int, d_ff: int, dropout: float, use_swiglu: bool, activation: str
    ) -> nn.Module:
        if use_swiglu:
            return MoE_SwiGLUExpert(d_model, d_ff, dropout)
        else:
            return MoE_FFNExpert(d_model, d_ff, dropout, activation)

    def _compute_router_probabilities(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.d_model)

        router_logits = self.router(x_flat)

        if self._router_temp_not_one:
            router_logits = router_logits / self.router_temperature

        if self.training and self._needs_expert_dropout:
            dropout_mask = (
                torch.rand(self.num_experts, device=x.device) > self.expert_dropout
            )
            router_logits = router_logits.masked_fill(
                ~dropout_mask.unsqueeze(0), float("-inf")
            )

        if self.use_noisy_gating and self.training:
            noise = torch.randn_like(router_logits) * self.noise_eps
            router_logits = router_logits + noise

        router_probs = F.softmax(router_logits, dim=-1)
        return router_logits, router_probs

    def _optimized_expert_forward(
        self, x: torch.Tensor, router_probs: torch.Tensor
    ) -> torch.Tensor:
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        expert_input_buf, expert_row_indices = self.dispatcher.dispatch(
            x, top_k_probs, top_k_indices
        )
        expert_counts = torch.bincount(
            top_k_indices.view(-1), minlength=self.num_experts
        )

        expert_output_buf = torch.zeros_like(expert_input_buf)
        offset = 0

        for expert_id, count in enumerate(expert_counts.tolist()):
            if count == 0:
                continue
            end_offset = offset + count
            expert_input = expert_input_buf[offset:end_offset]
            expert_output_buf[offset:end_offset] = self.experts[expert_id](expert_input)
            offset = end_offset

        flat_output = torch.zeros_like(x)
        expert_row_flat = expert_row_indices.view(-1)
        token_indices = (
            torch.arange(x.size(0), device=x.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )

        flat_output.index_add_(0, token_indices, expert_output_buf[expert_row_flat])
        return flat_output

    def _compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training or (
            not self._needs_load_balancing and not self._needs_z_loss
        ):
            return torch.tensor(0.0, device=router_logits.device)

        aux_loss = 0.0

        if self._needs_load_balancing:
            num_tokens = router_probs.size(0)
            flat_indices = top_k_indices.view(-1)
            usage_counts = torch.bincount(flat_indices, minlength=self.num_experts)
            expert_usage = usage_counts.float() / (num_tokens * self.top_k)
            router_prob_means = router_probs.mean(dim=0)
            load_loss = (expert_usage * router_prob_means).sum() * self.num_experts
            aux_loss += self.load_balancing_loss_weight * load_loss

        if self._needs_z_loss:
            z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
            aux_loss += self.z_loss_weight * z_loss

        return aux_loss

    def forward(self, x: torch.Tensor, return_aux_loss: bool = True):
        B, T, D = x.shape
        original_shape = x.shape

        x_norm = self.input_norm(x)
        x_flat = x_norm.view(-1, D)

        router_logits, router_probs = self._compute_router_probabilities(x_norm)
        expert_output = self._optimized_expert_forward(x_flat, router_probs)

        if self.use_shared_expert:
            shared_output = self.shared_expert(x_flat)
            expert_output = expert_output + shared_output

        output = expert_output.view(original_shape)

        if return_aux_loss and self.training:
            top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
            self.aux_loss = self._compute_aux_loss(
                router_logits, router_probs, top_k_indices
            )
        else:
            self.aux_loss = 0.0

        if return_aux_loss:
            return output, self.aux_loss
        else:
            return output


class MoE_SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.dropout_p = dropout
        self._needs_dropout = dropout > 0.0
        self.d_model = d_model
        self.d_ff = d_ff

        self.gate_up_proj = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        nn.init.xavier_uniform_(self.gate_up_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.down_proj.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape

        if (
            TRITON_AVAILABLE
            and x.is_cuda
            and B >= 64
            and D >= 512
            and not (self.training and self._needs_dropout)
        ):
            try:
                return triton_swiglu_forward(
                    x, self.gate_up_proj.weight, self.down_proj.weight
                )
            except Exception:
                pass

        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = torch.clamp(gate, -10, 10)
        hidden = F.silu(gate) * up

        if self.training and self._needs_dropout:
            hidden = F.dropout(hidden, p=self.dropout_p, training=True, inplace=True)

        return self.down_proj(hidden)


class MoE_FFNExpert(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"
    ):
        super().__init__()

        self.dropout_p = dropout
        self._needs_dropout = dropout > 0.0
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)

        activation = activation.lower()
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        elif activation in ("swish", "silu"):
            self.activation = F.silu
        else:
            self.activation = F.gelu

        nn.init.xavier_uniform_(self.linear1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.linear2.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)

        if self.training and self._needs_dropout:
            x = F.dropout(x, p=self.dropout_p, training=True, inplace=True)

        return self.linear2(x)


############################################################
# End of MoE
############################################################


class _StandardFeedForwardBlock(nn.Module):
    """
    Optimized standard feedforward block with performance improvements:
    - Cached activation functions
    - Inline dropout for SwiGLU
    - Optimized SwiGLU implementation
    - Better numerical stability
    """

    def __init__(
        self, d_model, dim_ff, dropout=0.1, use_swiglu=True, activation="gelu"
    ):
        super().__init__()
        self.use_swiglu = use_swiglu
        self.dropout_p = dropout

        # Cache whether we need dropout for faster runtime checks
        self._needs_dropout = dropout > 0.0

        if use_swiglu:
            # Optimized SwiGLU with better dimension calculation
            swiglu_dim = int(dim_ff * 4 / 3)
            self.w1 = nn.Linear(
                d_model, swiglu_dim, bias=False
            )  # Often works better without bias
            self.w2 = nn.Linear(d_model, swiglu_dim, bias=False)
            self.w3 = nn.Linear(swiglu_dim, d_model)

            # No separate dropout layer for SwiGLU (applied inline)
            self.dropout = None

        else:
            self.linear1 = nn.Linear(d_model, dim_ff)
            self.linear2 = nn.Linear(dim_ff, d_model)

            # Cache activation function for faster dispatch
            if activation == "relu":
                self.activation = F.relu
            elif activation == "gelu":
                self.activation = F.gelu
            elif activation == "swish" or activation == "silu":
                self.activation = F.silu
            else:
                self.activation = F.gelu  # Default fallback

            # Keep dropout layer for standard FFN
            self.dropout = nn.Dropout(dropout) if self._needs_dropout else None

    def forward(self, x):
        if self.use_swiglu:
            # Optimized SwiGLU implementation
            # Compute both projections
            u = self.w1(x)
            v = self.w2(x)

            # More stable clamping (avoid extreme values)
            u = u.clamp(-20, 20)  # Reduced range for better stability
            v = v.clamp(-20, 20)

            # Fused SiLU + multiply (more efficient than separate operations)
            z = F.silu(u) * v

            # Apply inline dropout for SwiGLU
            if self.training and self._needs_dropout:
                z = F.dropout(z, p=self.dropout_p, training=True)

            # Output projection
            return self.w3(z)

        else:
            # Standard feedforward with cached activation
            x = self.linear1(x)
            x = self.activation(x)

            # Apply dropout if needed
            if self.dropout is not None:
                x = self.dropout(x)

            return self.linear2(x)


class FeedForwardBlock(nn.Module):
    """
    Optimized feedforward block wrapper with performance improvements:
    - Cached configuration checks
    - Better MoE integration
    - Simplified forward logic
    """

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

        # Cache configuration for faster runtime dispatch
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

            # Cache whether MoE block supports aux loss
            self._supports_aux_loss = (
                hasattr(self.block, "forward")
                and "return_aux_loss" in self.block.forward.__code__.co_varnames
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
            self._supports_aux_loss = False

    def forward(self, x, return_aux_loss=False):
        """Optimized forward with optional auxiliary loss handling"""
        if self.use_moe:
            if return_aux_loss:
                # Case 1: MoE returns (output, aux_loss) directly
                try:
                    return self.block(x, return_aux_loss=True)
                except TypeError:
                    # Case 2: fallback, MoE returns output only, aux_loss separately
                    output = self.block(x)
                    if hasattr(self.block, "aux_loss"):
                        aux_loss = self.block.aux_loss()
                    else:
                        aux_loss = 0.0
                    return output, aux_loss
            else:
                return self.block(x)
        else:
            output = self.block(x)
            return (output, 0.0) if return_aux_loss else output
