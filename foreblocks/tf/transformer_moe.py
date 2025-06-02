import math
from typing import Dict, Optional, Tuple

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
    BLOCK_FF: tl.constexpr,
):
    """Optimized SwiGLU kernel with better memory coalescing and vectorization"""
    pid_m = tl.program_id(0)
    pid_ff = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ff = pid_ff * BLOCK_FF + tl.arange(0, BLOCK_FF)

    mask_m = offs_m < N
    mask_ff = offs_ff < D_FF

    # Initialize accumulators for gate and up projections
    acc_gate = tl.zeros((BLOCK_M, BLOCK_FF), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_FF), dtype=tl.float32)

    # Process input in chunks for better memory bandwidth utilization
    for k_start in range(0, D_MODEL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < D_MODEL

        # Load input chunk with proper masking
        x_ptrs = x_ptr + offs_m[:, None] * stride_x + offs_k[None, :]
        x_chunk = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load gate weights (first half of gate_up_weight)
        gate_w_ptrs = gate_up_weight_ptr + offs_k[:, None] * D_FF + offs_ff[None, :]
        gate_w = tl.load(
            gate_w_ptrs, mask=mask_k[:, None] & mask_ff[None, :], other=0.0
        )

        # Load up weights (second half of gate_up_weight)
        up_w_ptrs = (
            gate_up_weight_ptr + (D_FF + offs_k[:, None]) * D_FF + offs_ff[None, :]
        )
        up_w = tl.load(up_w_ptrs, mask=mask_k[:, None] & mask_ff[None, :], other=0.0)

        # Accumulate matrix multiplications
        acc_gate += tl.dot(x_chunk, gate_w, allow_tf32=True)
        acc_up += tl.dot(x_chunk, up_w, allow_tf32=True)

    # Apply SiLU activation to gate: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    gate_silu = acc_gate * tl.sigmoid(acc_gate)
    hidden = gate_silu * acc_up

    # Down projection - simplified single-pass approach
    acc_out = tl.zeros((BLOCK_M, D_MODEL), dtype=tl.float32)

    for d_start in range(0, D_MODEL, BLOCK_K):
        offs_d = d_start + tl.arange(0, BLOCK_K)
        mask_d = offs_d < D_MODEL

        # Load portion of down weight matrix
        down_w_ptrs = down_weight_ptr + offs_ff[:, None] * D_MODEL + offs_d[None, :]
        down_w = tl.load(
            down_w_ptrs, mask=mask_ff[:, None] & mask_d[None, :], other=0.0
        )

        # Compute this portion of output
        out_chunk = tl.dot(hidden, down_w, allow_tf32=True)

        # Accumulate to output buffer
        if d_start == 0:
            acc_out = tl.zeros((BLOCK_M, len(offs_d)), dtype=tl.float32)

        # Add to appropriate slice of accumulator
        for i in range(len(offs_d)):
            if mask_d[i]:
                acc_out[:, i] = out_chunk[:, i]

    # Store final output
    out_ptrs = out_ptr + offs_m[:, None] * stride_out + tl.arange(0, D_MODEL)[None, :]
    final_mask = mask_m[:, None] & (tl.arange(0, D_MODEL)[None, :] < D_MODEL)
    tl.store(out_ptrs, acc_out, mask=final_mask)


def triton_swiglu_forward(x, gate_up_weight, down_weight):
    """Optimized SwiGLU forward with better kernel launch configuration"""
    N, D_MODEL = x.shape
    D_FF = gate_up_weight.size(1) // 2

    out = torch.empty((N, D_MODEL), device=x.device, dtype=x.dtype)

    # Optimized block sizes based on problem dimensions and GPU architecture
    BLOCK_M = min(64, triton.next_power_of_2(N))
    BLOCK_K = min(128, triton.next_power_of_2(D_MODEL))
    BLOCK_FF = min(128, triton.next_power_of_2(D_FF))

    # 2D grid for better parallelization
    grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(D_FF, BLOCK_FF))

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
        BLOCK_FF=BLOCK_FF,
    )
    return out


@triton.jit
def moe_dispatch_kernel(
    x_ptr,
    top_k_probs_ptr,
    top_k_indices_ptr,
    expert_row_indices_ptr,
    expert_input_ptr,
    N,
    D,
    stride_x_n,
    stride_x_d,
    stride_probs_n,
    stride_probs_k,
    stride_indices_n,
    stride_indices_k,
    stride_expert_indices_n,
    stride_expert_indices_k,
    stride_expert_input_n,
    stride_expert_input_d,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Optimized MoE dispatch kernel"""
    pid = tl.program_id(0)

    # Each program handles one token
    token_id = pid
    if token_id >= N:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Load input for this token
    x_ptrs = x_ptr + token_id * stride_x_n + offs_d * stride_x_d
    x_vec = tl.load(x_ptrs, mask=mask_d, other=0.0)

    # Process each expert assignment for this token
    for k in tl.static_range(K):
        # Load probability for this token and expert k
        prob_ptr = top_k_probs_ptr + token_id * stride_probs_n + k * stride_probs_k
        prob = tl.load(prob_ptr)

        # Load expert row index
        expert_row_ptr = (
            expert_row_indices_ptr
            + token_id * stride_expert_indices_n
            + k * stride_expert_indices_k
        )
        expert_row = tl.load(expert_row_ptr)

        # Apply gating and store to expert buffer
        weighted_x = x_vec * prob

        # Store to expert buffer
        expert_ptrs = (
            expert_input_ptr
            + expert_row * stride_expert_input_n
            + offs_d * stride_expert_input_d
        )
        tl.store(expert_ptrs, weighted_x, mask=mask_d)


class TritonMoEDispatcher:
    def __init__(self, d_model: int, top_k: int, block_d: int = 64):
        self.d_model = d_model
        self.top_k = top_k
        self.block_d = triton.next_power_of_2(block_d)
        self._buffers = {}
        # Enhanced buffer management with memory pooling
        self._buffers: Dict[Tuple, torch.Tensor] = {}
        self._max_cached_size = 8  # Maximum number of cached buffers per size
        self._buffer_usage_count: Dict[Tuple, int] = {}

        # Pre-allocated workspace for expert row indices computation
        self._expert_offsets_cache: Optional[torch.Tensor] = None
        self._max_experts_cached = 0

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

    def dispatch(
        self, x: torch.Tensor, top_k_probs: torch.Tensor, top_k_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized dispatch with adaptive kernel selection"""
        N, D = x.shape
        K = self.top_k

        # Compute expert assignments
        flat_indices = top_k_indices.view(-1)
        expert_counts = torch.bincount(flat_indices, minlength=top_k_probs.size(-1))
        expert_row_indices = self.compute_expert_row_indices(
            top_k_indices, expert_counts
        )

        total_rows = expert_counts.sum().item()
        expert_input = self._get_or_create_buffer(total_rows, D, x.device, x.dtype)

        # Adaptive kernel selection based on problem size and hardware
        use_triton = (
            TRITON_AVAILABLE and x.is_cuda and N >= 32 and D >= 64 and total_rows > 0
        )

        if use_triton:
            # Use optimized Triton kernel for large problems
            BLOCK_D = min(self.block_d, triton.next_power_of_2(D))

            grid = (N,)  # One program per token

            moe_dispatch_kernel[grid](
                x_ptr=x,
                top_k_probs_ptr=top_k_probs,
                top_k_indices_ptr=top_k_indices,
                expert_row_indices_ptr=expert_row_indices,
                expert_input_ptr=expert_input,
                N=N,
                D=D,
                stride_x_n=x.stride(0),
                stride_x_d=x.stride(1),
                stride_probs_n=top_k_probs.stride(0),
                stride_probs_k=top_k_probs.stride(1),
                stride_indices_n=top_k_indices.stride(0),
                stride_indices_k=top_k_indices.stride(1),
                stride_expert_indices_n=expert_row_indices.stride(0),
                stride_expert_indices_k=expert_row_indices.stride(1),
                stride_expert_input_n=expert_input.stride(0),
                stride_expert_input_d=expert_input.stride(1),
                K=K,  # Now passed as constexpr
                BLOCK_D=BLOCK_D,
            )
        else:
            # Optimized PyTorch fallback with vectorization
            self._dispatch_pytorch_optimized(
                x, top_k_probs, expert_row_indices, expert_input, N, K
            )

        return expert_input[:total_rows], expert_row_indices

    def _dispatch_pytorch_optimized(
        self,
        x: torch.Tensor,
        top_k_probs: torch.Tensor,
        expert_row_indices: torch.Tensor,
        expert_input: torch.Tensor,
        N: int,
        K: int,
    ) -> None:
        """Optimized PyTorch fallback using vectorized operations"""
        # Vectorized approach - process all assignments at once
        token_indices = (
            torch.arange(N, device=x.device).unsqueeze(1).expand(-1, K).reshape(-1)
        )
        k_indices = (
            torch.arange(K, device=x.device).unsqueeze(0).expand(N, -1).reshape(-1)
        )
        expert_rows = expert_row_indices.view(-1)
        probs = top_k_probs.view(-1)

        # Filter out any invalid expert rows (shouldn't happen but safety check)
        valid_mask = expert_rows < expert_input.size(0)
        if not valid_mask.all():
            token_indices = token_indices[valid_mask]
            expert_rows = expert_rows[valid_mask]
            probs = probs[valid_mask]

        # Vectorized assignment using advanced indexing
        expert_input[expert_rows] = x[token_indices] * probs.unsqueeze(-1)

    def _get_or_create_buffer(
        self, total_rows: int, D: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Optimized buffer management with pooling"""
        key = (total_rows, D, device, dtype)

        if key in self._buffers:
            self._buffer_usage_count[key] = self._buffer_usage_count.get(key, 0) + 1
            return self._buffers[key]

        # Clean up old buffers if we have too many
        if len(self._buffers) >= self._max_cached_size:
            # Remove least recently used buffer
            lru_key = min(
                self._buffer_usage_count.keys(), key=self._buffer_usage_count.get
            )
            del self._buffers[lru_key]
            del self._buffer_usage_count[lru_key]

        # Create new buffer
        buffer = torch.empty((total_rows, D), device=device, dtype=dtype)
        self._buffers[key] = buffer
        self._buffer_usage_count[key] = 1

        return buffer

    def cleanup_buffers(self) -> None:
        """Clean up cached buffers to free memory"""
        self._buffers.clear()
        self._buffer_usage_count.clear()
        self._expert_offsets_cache = None
        self._max_experts_cached = 0


class LearnedRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        dropout: float = 0.0,
        jitter: float = 0.01,
        use_bias: bool = False,
        use_switch_gating: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.jitter = jitter
        self.use_switch_gating = use_switch_gating

        self.router = nn.Linear(d_model, num_experts, bias=use_bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        if use_bias:
            nn.init.constant_(self.router.bias, 0.0)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        logits = self.router(x)

        if self.training and self.jitter > 0:
            if self.use_switch_gating:
                noise = torch.randn_like(logits) * self.jitter
                logits = logits + noise
            else:
                logits = logits + torch.randn_like(logits) * self.jitter

        probs = F.softmax(logits, dim=-1)
        aux = {"router_entropy": -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()}
        return logits, probs, aux


class HashRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, num_hashes: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.num_hashes = num_hashes
        self.hash_weights = nn.Parameter(
            torch.randn(num_hashes, d_model) * 0.02, requires_grad=False
        )

    def forward(self, x):
        hash_values = torch.matmul(x, self.hash_weights.t())
        hash_indices = torch.argmax(hash_values, dim=-1) % self.num_experts
        probs = torch.zeros(x.size(0), self.num_experts, device=x.device)
        probs.scatter_(1, hash_indices.unsqueeze(1), 1.0)
        logits = torch.log(probs + 1e-8)
        return logits, probs, {}


class RandomRouter(nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, x):
        probs = torch.rand(x.size(0), self.num_experts, device=x.device)
        probs = F.softmax(probs, dim=-1)
        logits = torch.log(probs + 1e-8)
        return logits, probs, {}


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
        min_capacity: int = 4,
        use_swiglu: bool = True,
        activation: str = "gelu",
        shared_expert_ratio: float = 0.25,
        use_shared_expert: bool = True,
        load_balancing_loss_weight: float = 1e-2,
        z_loss_weight: float = 1e-3,
        router_init_std: float = 0.02,
        use_capacity_factor: bool = True,
        use_bias: bool = False,
        normalize_router_weights: bool = True,
        router_temperature: float = 1.0,
        router_type: str = "learned",
        # State-of-the-art optimizations
        use_switch_gating: bool = True,
        use_expert_choice: bool = False,
        expert_choice_k: int = 4,
        use_gradient_checkpointing: bool = False,
        use_mixed_precision: bool = True,
        use_adaptive_capacity: bool = True,
        use_router_z_loss: bool = True,
        use_cosine_router: bool = False,
        expert_dropout_strategy: str = "random",
        use_auxiliary_loss_scheduling: bool = True,
        warmup_aux_loss_factor: float = 0.1,
        use_expert_regularization: bool = True,
        expert_diversity_weight: float = 1e-4,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.capacity_factor = capacity_factor
        self.min_capacity = min_capacity
        self.expert_dropout = expert_dropout
        self.shared_expert_ratio = shared_expert_ratio
        self.use_shared_expert = use_shared_expert
        self.load_balancing_loss_weight = load_balancing_loss_weight
        self.z_loss_weight = z_loss_weight
        self.use_capacity_factor = use_capacity_factor
        self.router_temperature = router_temperature

        # State-of-the-art parameters
        self.use_switch_gating = use_switch_gating
        self.use_expert_choice = use_expert_choice
        self.expert_choice_k = expert_choice_k
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_mixed_precision = use_mixed_precision
        self.use_adaptive_capacity = use_adaptive_capacity
        self.use_router_z_loss = use_router_z_loss
        self.use_cosine_router = use_cosine_router
        self.expert_dropout_strategy = expert_dropout_strategy
        self.use_auxiliary_loss_scheduling = use_auxiliary_loss_scheduling
        self.warmup_aux_loss_factor = warmup_aux_loss_factor
        self.use_expert_regularization = use_expert_regularization
        self.expert_diversity_weight = expert_diversity_weight

        # Training step counter for scheduling
        self.register_buffer("training_step", torch.tensor(0))

        # Pre-compute boolean flags
        self._needs_load_balancing = load_balancing_loss_weight > 0
        self._needs_z_loss = z_loss_weight > 0
        self._needs_expert_dropout = expert_dropout > 0
        self._needs_capacity_enforcement = use_capacity_factor and capacity_factor > 0
        self._router_temp_not_one = abs(router_temperature - 1.0) > 1e-6

        # Router initialization
        if router_type == "learned":
            self.router = LearnedRouter(
                d_model,
                num_experts,
                use_bias=use_bias,
                use_switch_gating=use_switch_gating,
            )
        elif router_type == "hash":
            self.router = HashRouter(d_model, num_experts)
        elif router_type == "random":
            self.router = RandomRouter(num_experts)
        else:
            self.router = nn.Linear(d_model, num_experts, bias=use_bias)
            if normalize_router_weights:
                nn.init.normal_(self.router.weight, mean=0.0, std=router_init_std)
            else:
                nn.init.kaiming_normal_(self.router.weight, mode="fan_in")

        # Cosine router enhancement
        if self.use_cosine_router and not isinstance(
            self.router, (HashRouter, RandomRouter)
        ):
            self.router_scale = nn.Parameter(torch.tensor(1.0))

        # Normalization layers
        self.input_norm = nn.LayerNorm(d_model)
        self.router_norm = nn.LayerNorm(d_model) if router_type == "learned" else None

        # Expert creation
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

        # Capacity management
        if self._needs_capacity_enforcement:
            if self.use_adaptive_capacity:
                self.register_buffer("capacity_history", torch.zeros(100))
                self.register_buffer("capacity_idx", torch.tensor(0))
            self.expert_capacity = max(
                min_capacity, int(capacity_factor * d_model / num_experts)
            )
        else:
            self.expert_capacity = None

        self.aux_loss = 0.0
        self.dispatcher = TritonMoEDispatcher(d_model=d_model, top_k=top_k)

        # Buffers and cache
        self.register_buffer("_eps", torch.tensor(1e-8))
        self._last_batch_size = 0
        self._token_indices_cache = None
        self.register_buffer("expert_utilization", torch.zeros(num_experts))
        self.register_buffer("utilization_momentum", torch.tensor(0.9))

    def _create_expert(
        self, d_model: int, d_ff: int, dropout: float, use_swiglu: bool, activation: str
    ) -> nn.Module:
        if use_swiglu:
            return MoE_SwiGLUExpert(d_model, d_ff, dropout)
        else:
            return MoE_FFNExpert(d_model, d_ff, dropout, activation)

    def _compute_router_probs(self, x):
        """Enhanced router computation"""
        x_router = self.router_norm(x) if self.router_norm is not None else x

        if isinstance(self.router, (LearnedRouter, HashRouter, RandomRouter)):
            return self.router(x_router)
        else:
            logits = self.router(x_router)

            # Cosine similarity routing
            if self.use_cosine_router:
                x_norm = F.normalize(x_router, dim=-1)
                w_norm = F.normalize(self.router.weight, dim=-1)
                logits = F.linear(x_norm, w_norm) * self.router_scale

            if self._router_temp_not_one:
                logits = logits / self.router_temperature

            # Router z-loss
            router_aux = {}
            if self.use_router_z_loss and self.training:
                z_loss = torch.logsumexp(logits, dim=-1).square().mean()
                router_aux["z_loss"] = z_loss

            probs = F.softmax(logits, dim=-1)
            return logits, probs, router_aux

    def _expert_choice_routing(
        self, x: torch.Tensor, router_probs: torch.Tensor
    ) -> torch.Tensor:
        """Expert Choice routing"""
        flat_probs = router_probs.view(-1, self.num_experts)
        output = torch.zeros_like(x.view(-1, self.d_model))

        for expert_id in range(self.num_experts):
            expert_probs = flat_probs[:, expert_id]
            top_tokens = torch.topk(
                expert_probs, min(self.expert_choice_k, len(expert_probs))
            )[1]

            if len(top_tokens) > 0:
                expert_input = x.view(-1, self.d_model)[top_tokens]
                expert_output = self.experts[expert_id](expert_input)
                output[top_tokens] += expert_output

        return output

    def _adaptive_expert_dropout(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Adaptive expert dropout"""
        if not self._needs_expert_dropout or not self.training:
            return router_probs

        if self.expert_dropout_strategy == "balanced":
            expert_usage = router_probs.mean(dim=0)
            avg_usage = expert_usage.mean()
            dropout_mask = (expert_usage > 1.5 * avg_usage) & (
                torch.rand_like(expert_usage) < self.expert_dropout
            )
            router_probs = router_probs * (~dropout_mask).float()
        elif self.expert_dropout_strategy == "adaptive":
            confidence = router_probs.max(dim=-1)[0]
            dropout_prob = self.expert_dropout * (1 - confidence).unsqueeze(-1)
            dropout_mask = torch.bernoulli(dropout_prob).bool()
            router_probs = router_probs * (~dropout_mask).float()
        else:  # random
            if torch.rand(1).item() < self.expert_dropout:
                expert_to_drop = torch.randint(0, self.num_experts, (1,)).item()
                router_probs[:, expert_to_drop] = 0

        return router_probs / (router_probs.sum(dim=-1, keepdim=True) + self._eps)

    def _update_adaptive_capacity(self, actual_usage: torch.Tensor):
        """Update capacity based on usage"""
        if not self.use_adaptive_capacity:
            return

        avg_usage = actual_usage.float().mean()
        idx = self.capacity_idx % 100
        self.capacity_history[idx] = avg_usage
        self.capacity_idx += 1

        if self.capacity_idx > 10:
            recent_avg = self.capacity_history[: min(100, self.capacity_idx)].mean()
            if recent_avg > 0.8 * self.expert_capacity:
                self.expert_capacity = int(self.expert_capacity * 1.1)
            elif recent_avg < 0.4 * self.expert_capacity:
                self.expert_capacity = max(
                    self.min_capacity, int(self.expert_capacity * 0.9)
                )

    def _get_token_indices(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Cache token indices"""
        if (
            self._token_indices_cache is None
            or self._last_batch_size != batch_size
            or self._token_indices_cache.device != device
        ):

            self._token_indices_cache = (
                torch.arange(batch_size, device=device)
                .unsqueeze(1)
                .expand(-1, self.top_k)
                .reshape(-1)
            )
            self._last_batch_size = batch_size

        return self._token_indices_cache

    def _optimized_expert_forward(
        self, x: torch.Tensor, router_probs: torch.Tensor
    ) -> torch.Tensor:
        """Optimized expert forward pass"""
        router_probs = self._adaptive_expert_dropout(router_probs)

        if self.use_expert_choice:
            return self._expert_choice_routing(x, router_probs)

        # Token choice routing
        if self.use_switch_gating and self.top_k == 1:
            top_expert_idx = torch.argmax(router_probs, dim=-1)
            top_expert_probs = router_probs.gather(1, top_expert_idx.unsqueeze(1))
            top_k_indices = top_expert_idx.unsqueeze(1)
            top_k_probs = top_expert_probs
        else:
            top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)

        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + self._eps)

        # Update utilization tracking
        if self.training:
            expert_counts = torch.bincount(
                top_k_indices.view(-1), minlength=self.num_experts
            ).float()
            self.expert_utilization = (
                self.utilization_momentum * self.expert_utilization
                + (1 - self.utilization_momentum) * expert_counts / expert_counts.sum()
            )
            self._update_adaptive_capacity(expert_counts)

        # Dispatch and process
        with torch.amp.autocast("cuda", enabled=self.use_mixed_precision):
            expert_input_buf, expert_row_indices = self.dispatcher.dispatch(
                x, top_k_probs, top_k_indices
            )
            expert_counts = torch.bincount(
                top_k_indices.view(-1), minlength=self.num_experts
            )
            expert_output_buf = torch.empty_like(expert_input_buf)

            if expert_input_buf.size(0) > 0:
                if self.use_gradient_checkpointing and self.training:
                    self._process_experts_with_checkpointing(
                        expert_input_buf, expert_output_buf, expert_counts
                    )
                else:
                    self._process_experts_vectorized(
                        expert_input_buf, expert_output_buf, expert_counts
                    )

        # Combine outputs
        flat_output = torch.zeros_like(x)
        if expert_input_buf.size(0) > 0:
            expert_row_flat = expert_row_indices.view(-1)
            token_indices = self._get_token_indices(x.size(0), x.device)
            flat_output.index_add_(0, token_indices, expert_output_buf[expert_row_flat])

        return flat_output

    def _process_experts_with_checkpointing(
        self,
        expert_input_buf: torch.Tensor,
        expert_output_buf: torch.Tensor,
        expert_counts: torch.Tensor,
    ):
        """Expert processing with gradient checkpointing"""
        offset = 0
        for expert_id, count in enumerate(expert_counts):
            if count > 0:
                end_offset = offset + count
                expert_input = expert_input_buf[offset:end_offset]
                expert_output_buf[offset:end_offset] = (
                    torch.utils.checkpoint.checkpoint(
                        self.experts[expert_id], expert_input, use_reentrant=False
                    )
                )
                offset = end_offset

    def _process_experts_vectorized(
        self,
        expert_input_buf: torch.Tensor,
        expert_output_buf: torch.Tensor,
        expert_counts: torch.Tensor,
    ):
        """Vectorized expert processing"""
        offset = 0
        for expert_id, count in enumerate(expert_counts):
            if count > 0:
                end_offset = offset + count
                expert_input = expert_input_buf[offset:end_offset]
                expert_output_buf[offset:end_offset] = self.experts[expert_id](
                    expert_input
                )
                offset = end_offset

    def _compute_aux_loss_optimized(
        self,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
        router_aux: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Optimized auxiliary loss computation"""
        if not self.training:
            return torch.tensor(0.0, device=router_logits.device, requires_grad=False)

        aux_loss = torch.tensor(0.0, device=router_logits.device)

        # Auxiliary loss scheduling
        if self.use_auxiliary_loss_scheduling:
            warmup_factor = min(1.0, float(self.training_step) / 1000.0)
            aux_weight = (
                self.warmup_aux_loss_factor
                + (1.0 - self.warmup_aux_loss_factor) * warmup_factor
            )
        else:
            aux_weight = 1.0

        if self._needs_load_balancing:
            num_tokens = router_probs.size(0)
            flat_indices = top_k_indices.view(-1)
            usage_counts = torch.bincount(
                flat_indices, minlength=self.num_experts
            ).float()
            expert_usage = usage_counts / (num_tokens * self.top_k + self._eps)
            router_prob_means = router_probs.mean(dim=0)

            load_loss = torch.dot(expert_usage, router_prob_means) * self.num_experts

            if self.use_expert_regularization:
                prob_std = router_probs.std(dim=0).mean()
                diversity_loss = -prob_std
                load_loss += self.expert_diversity_weight * diversity_loss

            aux_loss += aux_weight * self.load_balancing_loss_weight * load_loss

        if self._needs_z_loss:
            z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()
            aux_loss += aux_weight * self.z_loss_weight * z_loss

        if "z_loss" in router_aux:
            aux_loss += aux_weight * self.z_loss_weight * router_aux["z_loss"]

        return aux_loss

    def forward(self, x: torch.Tensor, return_aux_loss: bool = True) -> torch.Tensor:
        """Enhanced forward pass"""
        if self.training:
            self.training_step += 1

        original_shape = x.shape
        x_norm = self.input_norm(x)
        x_flat = x_norm.view(-1, self.d_model)

        router_logits, router_probs, router_aux = self._compute_router_probs(x_flat)
        expert_output = self._optimized_expert_forward(x_flat, router_probs)

        if self.use_shared_expert:
            if self.use_gradient_checkpointing and self.training:
                shared_output = torch.utils.checkpoint.checkpoint(
                    self.shared_expert, x_flat, use_reentrant=False
                )
            else:
                shared_output = self.shared_expert(x_flat)
            expert_output = expert_output + shared_output

        output = expert_output.view(original_shape)

        if return_aux_loss and self.training:
            top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
            self.aux_loss = self._compute_aux_loss_optimized(
                router_logits, router_probs, top_k_indices, router_aux
            )
        else:
            self.aux_loss = 0.0

        return (output, self.aux_loss) if return_aux_loss else output

    def reset_cache(self):
        """Reset internal caches"""
        self._token_indices_cache = None
        self._last_batch_size = 0
        if hasattr(self.dispatcher, "cleanup_buffers"):
            self.dispatcher.cleanup_buffers()
        self.expert_utilization.zero_()


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
