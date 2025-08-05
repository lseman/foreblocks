import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

################################################################################
# Mixture of Experts (MoE) implementation using Triton for optimized performance
################################################################################

if TRITON_AVAILABLE:

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
            up_w = tl.load(
                up_w_ptrs, mask=mask_k[:, None] & mask_ff[None, :], other=0.0
            )

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
        out_ptrs = (
            out_ptr + offs_m[:, None] * stride_out + tl.arange(0, D_MODEL)[None, :]
        )
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
        expert_row_indices_ptr,
        expert_input_ptr,
        N,
        D,
        stride_x_n,
        stride_x_d,
        stride_probs_n,
        stride_probs_k,
        stride_expert_indices_n,
        stride_expert_indices_k,
        stride_expert_input_n,
        stride_expert_input_d,
        K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """SOTA parallel MoE dispatch kernel with (token_id, k) unrolled in grid"""

        pid = tl.program_id(0)
        token_id = pid // K
        k = pid % K

        if token_id >= N:
            return

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        # Load x[token_id, :]
        x_ptrs = x_ptr + token_id * stride_x_n + offs_d * stride_x_d
        x_vec = tl.load(
            x_ptrs, mask=mask_d, other=tl.zeros([BLOCK_D], dtype=tl.float32)
        )

        # Load prob[token_id, k]
        prob_ptr = top_k_probs_ptr + token_id * stride_probs_n + k * stride_probs_k
        prob = tl.load(prob_ptr)
        prob = tl.maximum(prob, 1e-6)  # clamp for numerical stability

        # Load expert_row_indices[token_id, k]
        expert_row_ptr = (
            expert_row_indices_ptr
            + token_id * stride_expert_indices_n
            + k * stride_expert_indices_k
        )
        expert_row = tl.load(expert_row_ptr)

        # Compute weighted input
        weighted_x = x_vec * prob

        # Store into expert_input[expert_row, :]
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
        self,
        x: torch.Tensor,
        top_k_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """SOTA parallel dispatch with enhanced Triton grid and safety checks"""
        N, D = x.shape
        K = self.top_k

        flat_indices = top_k_indices.view(-1)
        expert_counts = torch.bincount(flat_indices, minlength=top_k_probs.size(-1))
        expert_row_indices = self.compute_expert_row_indices(
            top_k_indices, expert_counts
        )

        total_rows = expert_counts.sum().item()
        expert_input = self._get_or_create_buffer(total_rows, D, x.device, x.dtype)

        BLOCK_D = min(self.block_d, triton.next_power_of_2(D))

        # Grid is now (N * K,) to parallelize over token-expert pairs
        grid = (N * K,)

        moe_dispatch_kernel[grid](
            x_ptr=x,
            top_k_probs_ptr=top_k_probs,
            expert_row_indices_ptr=expert_row_indices,
            expert_input_ptr=expert_input,
            N=N,
            D=D,
            stride_x_n=x.stride(0),
            stride_x_d=x.stride(1),
            stride_probs_n=top_k_probs.stride(0),
            stride_probs_k=top_k_probs.stride(1),
            stride_expert_indices_n=expert_row_indices.stride(0),
            stride_expert_indices_k=expert_row_indices.stride(1),
            stride_expert_input_n=expert_input.stride(0),
            stride_expert_input_d=expert_input.stride(1),
            K=K,
            BLOCK_D=BLOCK_D,
        )

        return expert_input[:total_rows], expert_row_indices

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



class NoisyTopKRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        input_dropout: float = 0.0,
        jitter: float = 0.01,
        use_bias: bool = False,
        use_switch_gating: bool = True,
        use_fused_softmax: bool = True,
        use_inplace_ops: bool = True,
        clamp_range: Tuple[float, float] = (-1e4, 1e4),
        eps: float = 1e-8,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.jitter = jitter
        self.use_switch_gating = use_switch_gating
        self.use_fused_softmax = use_fused_softmax
        self.use_inplace_ops = use_inplace_ops
        self.eps = eps

        self.router = nn.Linear(d_model, num_experts, bias=use_bias)
        self._init_router()

        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else None
        self.clamp_min = clamp_range[0]
        self.clamp_max = clamp_range[1]

        self.register_buffer("_eps_tensor", torch.tensor(eps))
        self._noise_cache = {}

    def _init_router(self):
        with torch.no_grad():
            std = 0.02
            bound = std * math.sqrt(3)
            self.router.weight.uniform_(-bound, bound)
            if self.router.bias is not None:
                self.router.bias.zero_()

    def _get_noise(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (shape, device, dtype)
        if key not in self._noise_cache or self._noise_cache[key].shape != shape:
            self._noise_cache[key] = torch.empty(shape, device=device, dtype=dtype)
        return self._noise_cache[key].normal_(0, 1)

    def _compute_probs_and_entropy(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_fused_softmax:
            max_logits = logits.max(dim=-1, keepdim=True).values
            shifted_logits = logits - max_logits

            exp_logits = torch.exp(shifted_logits)
            probs = exp_logits / (exp_logits.sum(dim=-1, keepdim=True) + self.eps)

            probs = torch.clamp(probs, self.eps, 1.0)
            log_probs = torch.log(probs)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
        else:
            probs = F.softmax(logits, dim=-1)
            probs = torch.clamp(probs, self.eps, 1.0)
            entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()

        return probs, entropy

    def forward(self, x: torch.Tensor, return_raw_logits: bool = False) -> Tuple:
        if x.numel() == 0:
            dummy = torch.zeros((*x.shape[:-1], self.num_experts), device=x.device, dtype=x.dtype)
            return dummy, dummy, {"router_entropy": torch.tensor(0.0, device=x.device, dtype=x.dtype)}

        if self.training and self.input_dropout is not None:
            x = self.input_dropout(x)

        raw_logits = self.router(x)

        if self.training and self.jitter > 0:
            noise = self._get_noise(raw_logits.shape, raw_logits.device, raw_logits.dtype)
            jittered = raw_logits + (noise * self.jitter).detach()
        else:
            jittered = raw_logits

        logits = torch.clamp(jittered, self.clamp_min, self.clamp_max) if not self.use_inplace_ops else jittered.clamp_(self.clamp_min, self.clamp_max)

        probs, entropy = self._compute_probs_and_entropy(logits)

        if return_raw_logits:
            return raw_logits, logits, probs, {"router_entropy": entropy}
        else:
            return logits, probs, {"router_entropy": entropy}

    def reset_cache(self):
        self._noise_cache.clear()

    def get_cache_info(self) -> Dict:
        return {
            "num_cached_tensors": len(self._noise_cache),
            "cache_memory_mb": sum(
                tensor.numel() * tensor.element_size() for tensor in self._noise_cache.values()
            ) / (1024 ** 2),
        }


class HashRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, num_hashes: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.num_hashes = num_hashes
        self.hash_weights = nn.Parameter(
            torch.randn(num_hashes, d_model) * 0.02, requires_grad=False
        )

    def forward(self, x):
        if x.numel() == 0:
            return (
                torch.zeros(x.size(0), self.num_experts, device=x.device),
                torch.zeros(x.size(0), self.num_experts, device=x.device),
                {},
            )

        hash_values = torch.matmul(x, self.hash_weights.t())
        hash_indices = torch.argmax(hash_values, dim=-1) % self.num_experts

        probs = torch.zeros(x.size(0), self.num_experts, device=x.device)
        probs.scatter_(1, hash_indices.unsqueeze(1), 1.0)
        probs = torch.clamp(probs, 1e-8, 1.0)
        logits = torch.log(probs)

        return logits, probs, {}


class RandomRouter(nn.Module):
    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, x):
        if x.numel() == 0:
            return (
                torch.zeros(x.size(0), self.num_experts, device=x.device),
                torch.zeros(x.size(0), self.num_experts, device=x.device),
                {},
            )

        logits = torch.rand(x.size(0), self.num_experts, device=x.device)
        logits = torch.clamp(logits, -1e4, 1e4)
        probs = F.softmax(logits - logits.max(dim=-1, keepdim=True).values, dim=-1)
        probs = torch.clamp(probs, 1e-8, 1.0)
        logits = torch.log(probs)

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
        min_capacity: int = 4,
        use_swiglu: bool = True,
        activation: str = "gelu",
        load_balance_weight: float = 1e-2,
        z_loss_weight: float = 1e-3,
        router_type: str = "noisy_topk",
        router_temperature: float = 1.0,
        router_init_std: float = 0.02,
        use_bias: bool = False,
        normalize_router_weights: bool = True,
        use_switch_gating: bool = True,
        use_expert_choice: bool = False,
        expert_choice_k: int = 4,
        use_gradient_checkpointing: bool = False,
        use_mixed_precision: bool = True,
        use_cosine_router: bool = False,
        expert_dropout: float = 0.0,
        router_noise_std: float = 0.0,
        expert_diversity_weight: float = 1e-4,
        use_capacity_dropping: bool = True,
        # Enhanced parameters
        switch_aux_loss_weight: float = 1e-2,
        use_expert_level_balancing: bool = False,
        expert_usage_ema_decay: float = 0.999,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.capacity_factor = capacity_factor
        self.min_capacity = min_capacity
        self.expert_dropout = expert_dropout
        self.load_balance_weight = load_balance_weight
        self.z_loss_weight = z_loss_weight
        self.expert_diversity_weight = expert_diversity_weight
        self.router_type = router_type
        self.router_temperature = router_temperature
        self.use_cosine_router = use_cosine_router
        self.router_noise_std = router_noise_std
        self.normalize_router_weights = normalize_router_weights
        self.use_switch_gating = use_switch_gating
        self.use_expert_choice = use_expert_choice
        self.expert_choice_k = expert_choice_k
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_mixed_precision = use_mixed_precision
        self.use_capacity_dropping = use_capacity_dropping

        # Enhanced parameters
        self.switch_aux_loss_weight = switch_aux_loss_weight
        self.use_expert_level_balancing = use_expert_level_balancing
        self.expert_usage_ema_decay = expert_usage_ema_decay

        self._needs_load_balance = load_balance_weight > 0
        self._needs_z_loss = z_loss_weight > 0
        self._has_router_temp = abs(router_temperature - 1.0) > 1e-6

        self.input_norm = nn.LayerNorm(d_model)

        # Initialize router
        if router_type == "noisy_topk":
            self.router = NoisyTopKRouter(
                d_model, num_experts, use_bias, use_switch_gating
            )
        elif router_type == "hash":
            self.router = HashRouter(d_model, num_experts)
        elif router_type == "random":
            self.router = RandomRouter(num_experts)
        elif router_type == "linear":
            self.router = nn.Linear(d_model, num_experts, bias=use_bias)
            if normalize_router_weights:
                nn.init.normal_(self.router.weight, mean=0.0, std=router_init_std)
            else:
                nn.init.kaiming_normal_(self.router.weight, mode="fan_in")
        else:
            raise ValueError(f"Unknown router_type: {router_type}")

        if self.use_cosine_router and not isinstance(
            self.router, (HashRouter, RandomRouter)
        ):
            self.router_scale = nn.Parameter(torch.tensor(math.log(num_experts)))

        self.experts = nn.ModuleList(
            [
                self._create_expert(d_model, d_ff, dropout, use_swiglu, activation)
                for _ in range(num_experts)
            ]
        )

        if TRITON_AVAILABLE:
            self.dispatcher = TritonMoEDispatcher(d_model, top_k)
        else:
            self.dispatcher = None

        # Statistics tracking
        self.register_buffer("_eps", torch.tensor(1e-8))
        self.register_buffer("expert_usage", torch.zeros(num_experts))
        self.register_buffer("expert_counts", torch.zeros(num_experts))
        self.register_buffer("momentum", torch.tensor(expert_usage_ema_decay))

        self.aux_loss = 0.0
        self.raw_logits = None

    def _create_expert(self, d_model, d_ff, dropout, use_swiglu, activation):
        if use_swiglu:
            return MoE_SwiGLUExpert(d_model, d_ff, dropout)
        return MoE_FFNExpert(d_model, d_ff, dropout, activation)

    def compute_capacity(self, num_tokens: int) -> int:
        return max(
            self.min_capacity,
            int(math.ceil(self.capacity_factor * num_tokens / self.num_experts)),
        )

    def _compute_router_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute router logits and store raw version for z-loss."""
        if isinstance(self.router, (NoisyTopKRouter, HashRouter, RandomRouter)):
            if isinstance(self.router, NoisyTopKRouter):
                raw_logits, logits, _, _ = self.router(x, return_raw_logits=True)
                self.raw_logits = raw_logits
            else:
                logits = self.router(x)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                self.raw_logits = logits
            return logits

        # Linear router
        if self.use_cosine_router and hasattr(self, "router_scale"):
            x_norm = F.normalize(x, dim=-1)
            w_norm = F.normalize(self.router.weight, dim=-1)
            logits = F.linear(x_norm, w_norm) * self.router_scale.exp()
        else:
            logits = self.router(x)

        # Store raw logits before transformations
        self.raw_logits = logits.clone()

        # Apply transformations
        if self._has_router_temp:
            logits = logits / self.router_temperature

        if self.training and self.router_noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.router_noise_std

        return logits

    def _update_expert_usage_ema(
        self, expert_probs: torch.Tensor, expert_indices: torch.Tensor
    ):
        """Update expert usage statistics with EMA."""
        if not self.training:
            return

        # Handle single expert case
        if expert_indices.dim() == 1:
            expert_indices = expert_indices.unsqueeze(-1)
            expert_probs = expert_probs.unsqueeze(-1)

        # Compute current usage
        current_usage = torch.zeros(self.num_experts, device=expert_probs.device)
        current_usage.scatter_add_(0, expert_indices.view(-1), expert_probs.view(-1))
        current_usage = current_usage / expert_indices.size(0)

        # Update with EMA
        self.expert_usage = (
            self.momentum * self.expert_usage + (1 - self.momentum) * current_usage
        )

    def _token_choice_routing(
        self, x: torch.Tensor, probs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """Token choice routing with capacity dropping."""
        original_shape = x.shape
        x_flat = x.view(-1, self.d_model)
        probs_flat = probs.view(-1, self.num_experts)

        # Get top-k
        if self.use_switch_gating and self.top_k == 1:
            top_probs, top_indices = torch.max(probs_flat, dim=-1, keepdim=True)
        else:
            top_probs, top_indices = torch.topk(probs_flat, self.top_k, dim=-1)

        # Expert dropout
        if self.training and self.expert_dropout > 0:
            top_probs = F.dropout(top_probs, p=self.expert_dropout, training=True)
            top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + self._eps)

        # Update usage stats
        if self.training:
            self._update_expert_usage_ema(top_probs, top_indices)

        # Capacity dropping
        num_tokens = x_flat.size(0)
        capacity = self.compute_capacity(num_tokens)

        if self.use_capacity_dropping:
            accept_mask = torch.zeros_like(top_probs, dtype=torch.bool)
            expert_counter = torch.zeros(
                self.num_experts, dtype=torch.int, device=x.device
            )

            for k in range(self.top_k):
                expert_ids = top_indices[:, k]
                for eid in range(self.num_experts):
                    mask = (expert_ids == eid).nonzero(as_tuple=False).squeeze(-1)
                    available = capacity - expert_counter[eid]
                    if available <= 0 or mask.numel() == 0:
                        continue
                    selected = mask[:available]
                    accept_mask[selected, k] = True
                    expert_counter[eid] += selected.numel()

            # Apply dropping and renormalize
            top_probs = top_probs * accept_mask.float()
            top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + self._eps)

        # Store for auxiliary loss
        self._last_routed_probs = top_probs.detach()
        self._last_routed_indices = top_indices.detach()

        # Route to experts
        routed = self._route_tokens_to_experts(x_flat, top_probs, top_indices)

        routing_stats = {
            "routing_type": "token_choice",
            "capacity": capacity,
            "expert_usage": self.expert_usage.clone(),
        }

        return routed.view(*original_shape[:-1], self.d_model), routing_stats

    def _route_tokens_to_experts(
        self, x: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Route tokens to experts using dispatcher or standard routing."""
        # Try to use Triton dispatcher first for better performance
        if self.dispatcher and self._dispatcher_is_compatible(indices):
            return self._dispatch_with_triton(x, probs, indices)

        # Fallback to standard routing
        return self._standard_routing(x, probs, indices)

    def _dispatcher_is_compatible(self, indices: torch.Tensor) -> bool:
        """Check if dispatcher can handle the current routing configuration."""
        return indices.dim() == 2 and indices.size(1) == self.top_k

    def _dispatch_with_triton(
        self, x: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Use Triton dispatcher for optimized routing."""
        expert_input_buf, expert_row_indices = self.dispatcher.dispatch(
            x, probs, indices
        )
        expert_output_buf = torch.empty_like(expert_input_buf)
        expert_counts = torch.bincount(indices.flatten(), minlength=self.num_experts)

        # Process through experts
        self._process_experts(expert_input_buf, expert_output_buf, expert_counts)

        # Combine outputs
        output = torch.zeros_like(x)
        if expert_output_buf.size(0) > 0:
            token_indices = (
                torch.arange(x.size(0), device=x.device)
                .unsqueeze(1)
                .expand(-1, self.top_k)
                .flatten()
            )
            output.index_add_(
                0, token_indices, expert_output_buf[expert_row_indices.view(-1)]
            )

        return output

    def _process_experts(
        self,
        expert_input_buf: torch.Tensor,
        expert_output_buf: torch.Tensor,
        expert_counts: torch.Tensor,
    ):
        """Process tokens through experts efficiently."""
        offset = 0
        for eid, count in enumerate(expert_counts):
            if count > 0:
                end = offset + count
                inp = expert_input_buf[offset:end]

                if self.use_gradient_checkpointing and self.training:
                    out = torch.utils.checkpoint.checkpoint(
                        self.experts[eid], inp, use_reentrant=False
                    )
                else:
                    out = self.experts[eid](inp)

                expert_output_buf[offset:end] = out
                offset = end

    def _standard_routing(
        self, x: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Standard token routing implementation."""
        B, D = x.shape
        K = indices.shape[1] if indices.dim() > 1 else 1

        if K == 1:
            x_expanded = x.unsqueeze(1)
            probs_expanded = probs.unsqueeze(-1) if probs.dim() == 1 else probs
        else:
            x_expanded = x.unsqueeze(1).expand(-1, K, -1)
            probs_expanded = probs

        x_flat = x_expanded.reshape(-1, D)
        probs_flat = probs_expanded.reshape(-1, 1)
        indices_flat = indices.reshape(-1)

        output = torch.zeros_like(x)
        for eid in range(self.num_experts):
            mask = indices_flat == eid
            if not mask.any():
                continue

            x_e = x_flat[mask]
            p_e = probs_flat[mask]

            if self.use_gradient_checkpointing and self.training:
                out_e = torch.utils.checkpoint.checkpoint(
                    self.experts[eid], x_e, use_reentrant=False
                )
            else:
                out_e = self.experts[eid](x_e)

            out_e = out_e * p_e
            token_ids = mask.nonzero(as_tuple=False).squeeze(1) // K
            output.index_add_(0, token_ids, out_e)

        return output

    def _compute_auxiliary_loss(
        self, probs: torch.Tensor, indices: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Compute all auxiliary losses."""
        if not self.training:
            return torch.tensor(0.0, device=probs.device)

        aux_loss = torch.tensor(0.0, device=probs.device)

        # Z-loss
        if self._needs_z_loss and self.raw_logits is not None:
            log_z = torch.logsumexp(self.raw_logits.view(-1, self.num_experts), dim=-1)
            z_loss = torch.mean(log_z**2)
            aux_loss += self.z_loss_weight * z_loss

        # Switch Transformer auxiliary loss
        if self.switch_aux_loss_weight > 0:
            indices_flat = indices.view(-1) if indices.dim() > 1 else indices
            expert_mask = F.one_hot(indices_flat, num_classes=self.num_experts).float()
            expert_usage_fraction = expert_mask.mean(dim=0)
            mean_router_probs = probs.mean(dim=0)
            switch_loss = (
                torch.sum(expert_usage_fraction * mean_router_probs) * self.num_experts
            )
            aux_loss += self.switch_aux_loss_weight * switch_loss

        # Load balance loss (legacy)
        if self._needs_load_balance:
            # Use routed probabilities if available
            if (
                hasattr(self, "_last_routed_probs")
                and self._last_routed_probs is not None
            ):
                routed_probs = self._last_routed_probs
                routed_indices = self._last_routed_indices
            else:
                routed_probs = torch.gather(
                    probs,
                    dim=-1,
                    index=indices.unsqueeze(-1) if indices.dim() == 1 else indices,
                )
                routed_indices = indices

            # Compute expert usage
            counts = torch.zeros(self.num_experts, device=probs.device)
            if routed_indices.dim() == 1:
                counts.scatter_add_(
                    0,
                    routed_indices,
                    routed_probs.squeeze(-1)
                    if routed_probs.dim() > 1
                    else routed_probs,
                )
            else:
                for k in range(routed_indices.size(1)):
                    valid_mask = routed_indices[:, k] >= 0
                    if valid_mask.any():
                        counts.scatter_add_(
                            0,
                            routed_indices[:, k][valid_mask],
                            routed_probs[:, k][valid_mask],
                        )

            usage = counts / (counts.sum() + self._eps)
            mean_probs = probs.mean(dim=0)
            load_loss = torch.sum(usage * mean_probs) * self.num_experts
            aux_loss += self.load_balance_weight * load_loss

        # Diversity loss
        if self.expert_diversity_weight > 0:
            entropy = -torch.sum(probs * torch.log(probs + self._eps), dim=-1).mean()
            aux_loss += self.expert_diversity_weight * (entropy)

        return aux_loss

    def _expert_choice_routing(
        self, x: torch.Tensor, probs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Expert Choice routing: experts choose their tokens.

        More efficient than Token Choice as each expert processes exactly
        expert_choice_k tokens, eliminating load imbalance and capacity issues.
        """
        original_shape = x.shape
        x_flat = x.view(-1, self.d_model)  # [B*L, d_model]
        probs_flat = probs.view(-1, self.num_experts)  # [B*L, num_experts]
        num_tokens = x_flat.size(0)

        # Each expert chooses top-k tokens based on routing probabilities
        k_per_expert = min(self.expert_choice_k, num_tokens)

        output = torch.zeros_like(x_flat)
        expert_selections = {}
        total_selected_tokens = 0

        for expert_id in range(self.num_experts):
            # Get this expert's probabilities for all tokens
            expert_probs = probs_flat[:, expert_id]  # [B*L]

            # Expert chooses top-k tokens with highest probabilities
            if k_per_expert < num_tokens:
                top_probs, top_indices = torch.topk(expert_probs, k_per_expert, dim=0)
            else:
                # If k >= num_tokens, take all tokens
                top_probs = expert_probs
                top_indices = torch.arange(num_tokens, device=x.device)

            if top_indices.numel() == 0:
                continue

            # Get selected tokens and process through expert
            selected_tokens = x_flat[top_indices]  # [k, d_model]

            # Process through expert
            if self.use_gradient_checkpointing and self.training:
                expert_output = torch.utils.checkpoint.checkpoint(
                    self.experts[expert_id], selected_tokens, use_reentrant=False
                )
            else:
                expert_output = self.experts[expert_id](selected_tokens)

            # Weight by routing probabilities
            weighted_output = expert_output * top_probs.unsqueeze(-1)

            # Add to output (multiple experts can contribute to same token)
            output.index_add_(0, top_indices, weighted_output)

            expert_selections[expert_id] = {
                "indices": top_indices,
                "probs": top_probs,
                "count": top_indices.numel(),
            }
            total_selected_tokens += top_indices.numel()

        # Normalize output by number of experts that contributed to each token
        token_expert_counts = torch.zeros(num_tokens, device=x.device)
        for selections in expert_selections.values():
            token_expert_counts.index_add_(
                0,
                selections["indices"],
                torch.ones_like(selections["indices"], dtype=torch.float),
            )

        # Avoid division by zero
        token_expert_counts = torch.clamp(token_expert_counts, min=1.0)
        output = output / token_expert_counts.unsqueeze(-1)

        # Update expert usage statistics for Expert Choice
        if self.training:
            self._update_expert_choice_usage_ema(expert_selections, num_tokens)

        routing_stats = {
            "routing_type": "expert_choice",
            "k_per_expert": k_per_expert,
            "total_selected_tokens": total_selected_tokens,
            "avg_tokens_per_expert": total_selected_tokens / self.num_experts,
            "expert_selections": {k: v["count"] for k, v in expert_selections.items()},
            "expert_usage": self.expert_usage.clone(),
        }

        return output.view(*original_shape[:-1], self.d_model), routing_stats

    def _update_expert_choice_usage_ema(self, expert_selections: Dict, num_tokens: int):
        """Update expert usage statistics for Expert Choice routing."""
        if not self.training:
            return

        # Compute current expert usage based on token selections
        current_usage = torch.zeros(self.num_experts, device=self.expert_usage.device)

        # Ensure all experts are accounted for (some might not have selections)
        for expert_id in range(self.num_experts):
            if expert_id in expert_selections:
                selection_info = expert_selections[expert_id]
                prob_sum = selection_info["probs"].sum().item()

                # Weight by probability sum normalized by total tokens
                current_usage[expert_id] = prob_sum / num_tokens
            # If expert not in selections, current_usage[expert_id] remains 0

        # Update with EMA - ensure tensors are on same device
        decay = (
            self.momentum.item()
            if isinstance(self.momentum, torch.Tensor)
            else self.momentum
        )
        self.expert_usage = decay * self.expert_usage + (1 - decay) * current_usage.to(
            self.expert_usage.device
        )

    def forward(self, x: torch.Tensor, return_aux_loss: bool = True):
        """Forward pass."""
        x_norm = self.input_norm(x)

        # Compute router probabilities
        logits = self._compute_router_logits(x_norm)
        probs = F.softmax(logits, dim=-1)

        # Route tokens
        if self.use_expert_choice:
            # Expert Choice: experts choose tokens
            output, routing_stats = self._expert_choice_routing(x_norm, probs)

            # For aux loss, we still need token->expert assignments
            # Use top-1 assignments for compatibility with aux loss computation
            _, top_indices = torch.max(probs.view(-1, self.num_experts), dim=-1)
        else:
            # Token Choice: tokens choose experts
            output, routing_stats = self._token_choice_routing(x_norm, probs)

            # Get indices for auxiliary loss computation
            if self.use_switch_gating and self.top_k == 1:
                _, top_indices = torch.max(probs.view(-1, self.num_experts), dim=-1)
            else:
                _, top_indices = torch.topk(
                    probs.view(-1, self.num_experts), self.top_k, dim=-1
                )

        # Compute auxiliary loss
        num_tokens = x.view(-1, self.d_model).size(0)
        if return_aux_loss and self.training:
            self.aux_loss = self._compute_auxiliary_loss(
                probs.view(-1, self.num_experts), top_indices, num_tokens
            )
        else:
            self.aux_loss = 0.0

        return (output, self.aux_loss) if return_aux_loss else output

    def get_expert_stats(self) -> Dict[str, torch.Tensor]:
        """Get expert usage statistics."""
        entropy = -torch.sum(
            self.expert_usage * torch.log(self.expert_usage + self._eps)
        )
        return {
            "expert_usage": self.expert_usage.clone(),
            "usage_entropy": entropy,
            "max_usage": self.expert_usage.max(),
            "min_usage": self.expert_usage.min(),
        }

    def reset_stats(self):
        """Reset statistics."""
        self.expert_usage.zero_()
        self.aux_loss = 0.0
        if hasattr(self.router, "reset_stats"):
            self.router.reset_stats()


class MoE_SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.dropout_p = dropout
        self._needs_dropout = dropout > 0.0
        self.d_model = d_model
        self.d_ff = d_ff

        self.gate_up_proj = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        nn.init.xavier_uniform_(self.gate_up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            TRITON_AVAILABLE
            and x.is_cuda
            and x.shape[0] >= 64
            and x.shape[1] >= 512
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
        hidden = F.silu(up) * gate  # ✅ correct SwiGLU

        if self.training and self._needs_dropout:
            hidden = F.dropout(hidden, p=self.dropout_p, training=self.training)

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

        self.activation = {
            "gelu": F.gelu,
            "relu": F.relu,
            "silu": F.silu,
            "swish": F.silu,
        }.get(activation.lower(), F.gelu)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)

        if self.training and self._needs_dropout:
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        return self.linear2(x)


############################################################
# End of MoE
############################################################


class _StandardFeedForwardBlock(nn.Module):
    def __init__(
        self, d_model, dim_ff, dropout=0.1, use_swiglu=True, activation="gelu"
    ):
        super().__init__()
        self.use_swiglu = use_swiglu
        self.dropout_p = dropout

        if use_swiglu:
            swiglu_dim = int(dim_ff * 4 / 3)
            self.w1 = nn.Linear(d_model, swiglu_dim, bias=False)
            self.w2 = nn.Linear(d_model, swiglu_dim, bias=False)
            self.w3 = nn.Linear(swiglu_dim, d_model)
        else:
            self.linear1 = nn.Linear(d_model, dim_ff)
            self.linear2 = nn.Linear(dim_ff, d_model)
            self.activation = {
                "relu": nn.ReLU(),
                "silu": nn.SiLU(),
                "swish": nn.SiLU(),
                "gelu": nn.GELU(),
            }.get(activation.lower(), nn.GELU())
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.use_swiglu:
            u = self.w1(x)
            v = self.w2(x)
            z = F.silu(v) * u  # Correct SwiGLU formulation
            z = F.dropout(z, p=self.dropout_p, training=self.training)
            return self.w3(z)
        else:
            x = self.linear1(x)
            x = self.activation(x)
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
