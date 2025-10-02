import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton availability
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

################################################################################
# Mixture of Experts (MoE) with Memory Optimization
################################################################################

if TRITON_AVAILABLE:
    @triton.jit
    def swiglu_fused_kernel(
        x_ptr, gate_up_w_ptr, down_w_ptr, out_ptr,
        N, D_MODEL, D_FF,
        stride_x_m, stride_x_k,
        stride_out_m, stride_out_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K_X: tl.constexpr,
        BLOCK_K_H: tl.constexpr,
    ):
        """Fixed fused SwiGLU kernel with proper accumulation."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_m = offs_m < N
        mask_n = offs_n < D_MODEL

        # Initialize accumulators properly
        G_acc = tl.zeros((BLOCK_M, BLOCK_K_X), dtype=tl.float32)
        U_acc = tl.zeros((BLOCK_M, BLOCK_K_X), dtype=tl.float32)

        # Phase 1: Accumulate G and U over input dimension
        for kx in range(0, D_MODEL, BLOCK_K_X):
            offs_kx = kx + tl.arange(0, BLOCK_K_X)
            mask_kx = offs_kx < D_MODEL

            # Load x sub-tile
            x_ptrs = x_ptr + (offs_m[:, None] * stride_x_m + offs_kx[None, :] * stride_x_k)
            x_sub = tl.load(x_ptrs, mask=mask_m[:, None] & mask_kx[None, :], other=0.0)

            # Load gate and up weights
            num_ff = min(BLOCK_K_X, D_FF)
            wg_ptrs = gate_up_w_ptr + (offs_kx[:, None] * (2 * D_FF) + tl.arange(0, num_ff)[None, :])
            wg_sub = tl.load(wg_ptrs, mask=mask_kx[:, None], other=0.0)

            wu_ptrs = gate_up_w_ptr + (offs_kx[:, None] * (2 * D_FF) + (D_FF + tl.arange(0, num_ff)[None, :]))
            wu_sub = tl.load(wu_ptrs, mask=mask_kx[:, None], other=0.0)

            # Accumulate
            G_part = tl.dot(x_sub, wg_sub)
            U_part = tl.dot(x_sub, wu_sub)
            
            if kx == 0:
                G_acc = G_part
                U_acc = U_part
            else:
                G_acc = G_acc + G_part
                U_acc = U_acc + U_part

        # Apply SwiGLU: SiLU(G) * U
        G_silu = G_acc * tl.sigmoid(G_acc)
        H_tile = G_silu * U_acc

        # Phase 2: H @ down_w
        out_tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for kh in range(0, D_FF, BLOCK_K_H):
            offs_kh = kh + tl.arange(0, BLOCK_K_H)
            mask_kh = offs_kh < D_FF

            # Slice H_tile (register tensor)
            H_sub = tl.zeros((BLOCK_M, BLOCK_K_H), dtype=tl.float32)
            for i in range(BLOCK_M):
                for j in range(BLOCK_K_H):
                    if offs_kh[j] < D_FF:
                        H_sub[i, j] = H_tile[i, offs_kh[j]]

            # Load down weight
            wd_ptrs = down_w_ptr + (offs_kh[:, None] * D_MODEL + offs_n[None, :])
            wd_sub = tl.load(wd_ptrs, mask=mask_kh[:, None] & mask_n[None, :], other=0.0)

            out_tile = out_tile + tl.dot(H_sub, wd_sub)

        # Store output
        out_ptrs = out_ptr + (offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
        tl.store(out_ptrs, out_tile, mask=mask_m[:, None] & mask_n[None, :])

    def triton_swiglu_forward(x, gate_up_weight, down_weight):
        """Optimized SwiGLU forward with proper kernel."""
        assert x.is_cuda and gate_up_weight.is_cuda and down_weight.is_cuda
        N, D_MODEL = x.shape
        D_FF = gate_up_weight.size(1) // 2

        out = torch.empty((N, D_MODEL), device=x.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K_X = min(64, triton.next_power_of_2(D_MODEL))
        BLOCK_K_H = min(64, triton.next_power_of_2(D_FF))

        grid = (triton.cdiv(N, BLOCK_M), triton.cdiv(D_MODEL, BLOCK_N))

        swiglu_fused_kernel[grid](
            x, gate_up_weight, down_weight, out,
            N, D_MODEL, D_FF,
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            BLOCK_K_X=BLOCK_K_X, BLOCK_K_H=BLOCK_K_H,
        )
        return out.to(dtype=x.dtype)

    @triton.jit
    def moe_dispatch_kernel(
        x_ptr, top_k_probs_ptr, top_k_indices_ptr,
        expert_row_indices_ptr, expert_input_ptr,
        N, D,
        stride_x_n, stride_x_d,
        stride_probs_n, stride_probs_k,
        stride_indices_n, stride_indices_k,
        stride_expert_indices_n, stride_expert_indices_k,
        stride_expert_input_n, stride_expert_input_d,
        K: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Memory-efficient dispatch kernel."""
        pid = tl.program_id(0)
        token_id = pid // K
        k = pid % K
        if token_id >= N:
            return

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        # Load and weight token
        x_ptrs = x_ptr + token_id * stride_x_n + offs_d * stride_x_d
        x_vec = tl.load(x_ptrs, mask=mask_d, other=0.0)

        prob_ptr = top_k_probs_ptr + token_id * stride_probs_n + k * stride_probs_k
        prob = tl.load(prob_ptr)
        prob = tl.maximum(prob, 1e-8)

        row_ptr = expert_row_indices_ptr + token_id * stride_expert_indices_n + k * stride_expert_indices_k
        row = tl.load(row_ptr)

        weighted = x_vec * prob

        expert_ptrs = expert_input_ptr + row * stride_expert_input_n + offs_d * stride_expert_input_d
        tl.store(expert_ptrs, weighted, mask=mask_d)


class TritonMoEDispatcher:
    """Memory-optimized MoE dispatcher with efficient buffer reuse."""
    
    def __init__(self, d_model: int, top_k: int, max_buffer_size: int = 32768):
        self.d_model = d_model
        self.top_k = top_k
        self.max_buffer_size = max_buffer_size
        
        # Single reusable buffer instead of dict of buffers
        self._buffer: Optional[torch.Tensor] = None
        self._buffer_device = None
        self._buffer_dtype = None
        
        # Reusable row indices buffer
        self._row_indices_buffer: Optional[torch.Tensor] = None

    @staticmethod
    def compute_expert_row_indices(
        top_k_indices: torch.Tensor, 
        expert_counts: torch.Tensor
    ) -> torch.Tensor:
        """Compute row indices with minimal allocations."""
        N, K = top_k_indices.shape
        device = top_k_indices.device
        
        # Use in-place operations where possible
        flat = top_k_indices.reshape(-1)
        sort_idx = torch.argsort(flat, stable=True)
        
        offsets = torch.zeros(expert_counts.size(0) + 1, device=device, dtype=torch.long)
        torch.cumsum(expert_counts, 0, out=offsets[1:])
        
        rows = torch.empty(N * K, device=device, dtype=torch.long)
        
        pos = 0
        for eid in range(expert_counts.size(0)):
            cnt = int(expert_counts[eid].item())
            if cnt > 0:
                start, end = int(offsets[eid].item()), int(offsets[eid + 1].item())
                sel = sort_idx[pos:pos + cnt]
                rows[sel] = torch.arange(start, end, device=device, dtype=torch.long)
                pos += cnt
        
        return rows.view(N, K)

    def _get_buffer(self, total_rows: int, D: int, device, dtype) -> torch.Tensor:
        """Get or resize single reusable buffer."""
        needed_size = total_rows * D
        
        if (self._buffer is None or 
            self._buffer_device != device or 
            self._buffer_dtype != dtype or
            self._buffer.numel() < needed_size):
            
            # Allocate with some headroom to reduce reallocations
            alloc_size = min(int(needed_size * 1.2), self.max_buffer_size * D)
            alloc_rows = alloc_size // D
            
            self._buffer = torch.empty((alloc_rows, D), device=device, dtype=dtype)
            self._buffer_device = device
            self._buffer_dtype = dtype
        
        return self._buffer[:total_rows]

    def dispatch(
        self, 
        x: torch.Tensor, 
        top_k_probs: torch.Tensor, 
        top_k_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Memory-efficient dispatch."""
        N, D = x.shape
        K = self.top_k
        
        # Count experts efficiently
        flat_indices = top_k_indices.view(-1)
        num_experts = top_k_indices.max().item() + 1 if top_k_indices.numel() > 0 else 0
        expert_counts = torch.bincount(flat_indices, minlength=num_experts)
        
        total_rows = int(expert_counts.sum().item())
        if total_rows == 0:
            return torch.empty((0, D), device=x.device, dtype=x.dtype), top_k_indices
        
        # Compute row indices
        row_indices = self.compute_expert_row_indices(top_k_indices, expert_counts)
        
        # Get reusable buffer
        expert_input = self._get_buffer(total_rows, D, x.device, x.dtype)
        
        if not TRITON_AVAILABLE or not x.is_cuda:
            # Fallback: optimized PyTorch path
            weighted = (x.unsqueeze(1) * top_k_probs.unsqueeze(-1)).reshape(-1, D)
            expert_input.zero_()
            expert_input.index_copy_(0, row_indices.view(-1), weighted)
            return expert_input[:total_rows].clone(), row_indices
        
        # Triton path
        BLOCK_D = min(128, triton.next_power_of_2(D))
        grid = (N * K,)
        
        moe_dispatch_kernel[grid](
            x_ptr=x, top_k_probs_ptr=top_k_probs,
            top_k_indices_ptr=top_k_indices,
            expert_row_indices_ptr=row_indices,
            expert_input_ptr=expert_input,
            N=N, D=D,
            stride_x_n=x.stride(0), stride_x_d=x.stride(1),
            stride_probs_n=top_k_probs.stride(0), stride_probs_k=top_k_probs.stride(1),
            stride_indices_n=top_k_indices.stride(0), stride_indices_k=top_k_indices.stride(1),
            stride_expert_indices_n=row_indices.stride(0), stride_expert_indices_k=row_indices.stride(1),
            stride_expert_input_n=expert_input.stride(0), stride_expert_input_d=expert_input.stride(1),
            K=K, BLOCK_D=BLOCK_D,
        )
        
        return expert_input[:total_rows], row_indices

    def cleanup_buffers(self):
        """Release buffers to free memory."""
        self._buffer = None
        self._row_indices_buffer = None
        self._buffer_device = None
        self._buffer_dtype = None


class NoisyTopKRouter(nn.Module):
    """Optimized router with reduced allocations."""
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        input_dropout: float = 0.0,
        jitter: float = 0.01,
        use_bias: bool = False,
        use_switch_gating: bool = True,
        clamp_range: Tuple[float, float] = (-1e4, 1e4),
        eps: float = 1e-8,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.jitter = jitter
        self.use_switch_gating = use_switch_gating
        self.eps = eps

        self.router = nn.Linear(d_model, num_experts, bias=use_bias)
        self._init_router()

        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else None
        self.clamp_min, self.clamp_max = clamp_range

        self.register_buffer("_eps_tensor", torch.tensor(eps))

    def _init_router(self):
        """Initialize router weights."""
        with torch.no_grad():
            std = 0.02
            bound = std * math.sqrt(3)
            self.router.weight.uniform_(-bound, bound)
            if self.router.bias is not None:
                self.router.bias.zero_()

    def forward(self, x: torch.Tensor, return_raw_logits: bool = False):
        """Forward with minimal allocations."""
        if x.numel() == 0:
            dummy = torch.zeros((*x.shape[:-1], self.num_experts), device=x.device, dtype=x.dtype)
            stats = {"router_entropy": torch.tensor(0.0, device=x.device)}
            return (dummy, dummy, stats) if not return_raw_logits else (dummy, dummy, dummy, stats)

        if self.training and self.input_dropout is not None:
            x = self.input_dropout(x)

        raw = self.router(x)
        
        # Apply jitter in-place when possible
        if self.training and self.jitter > 0:
            jittered = raw + torch.randn_like(raw) * self.jitter
        else:
            jittered = raw

        # Clamp in-place
        logits = jittered.clamp_(self.clamp_min, self.clamp_max)
        
        # Stable softmax
        logits_max = logits.max(dim=-1, keepdim=True).values
        exp_logits = torch.exp(logits - logits_max)
        probs = exp_logits / (exp_logits.sum(dim=-1, keepdim=True) + self.eps)
        probs = probs.clamp_(self.eps, 1.0)
        
        # Compute entropy efficiently
        log_probs = torch.log(probs)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        stats = {"router_entropy": entropy}
        
        if return_raw_logits:
            return raw, logits, probs, stats
        return logits, probs, stats


class MoE_SwiGLUExpert(nn.Module):
    """Memory-optimized SwiGLU expert."""
    
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
        """Forward with Triton acceleration when beneficial."""
        # Use Triton for large batches without dropout
        use_triton = (TRITON_AVAILABLE and x.is_cuda and 
                     x.shape[0] >= 64 and x.shape[1] >= 512 and
                     not (self.training and self._needs_dropout))
        
        if use_triton:
            try:
                return triton_swiglu_forward(x, self.gate_up_proj.weight, self.down_proj.weight)
            except Exception:
                pass

        # Standard path with in-place operations
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate).mul_(up)  # In-place multiplication
        
        if self.training and self._needs_dropout:
            hidden = F.dropout(hidden, p=self.dropout_p, training=True, inplace=False)
        
        return self.down_proj(hidden)


class MoE_FFNExpert(nn.Module):
    """Standard FFN expert."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.dropout_p = dropout
        self._needs_dropout = dropout > 0.0
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.activation = getattr(F, activation.lower(), F.gelu)
        
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.linear1(x))
        if self.training and self._needs_dropout:
            x = F.dropout(x, p=self.dropout_p, training=True, inplace=False)
        return self.linear2(x)


class MoEFeedForward(nn.Module):
    """Memory-optimized Mixture of Experts."""
    
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
        use_bias: bool = False,
        use_switch_gating: bool = True,
        use_gradient_checkpointing: bool = False,
        expert_dropout: float = 0.0,
        use_capacity_dropping: bool = True,
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
        self.use_switch_gating = use_switch_gating
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_capacity_dropping = use_capacity_dropping

        self._needs_load_balance = load_balance_weight > 0
        self._needs_z_loss = z_loss_weight > 0

        self.input_norm = nn.LayerNorm(d_model)

        # Router
        if router_type == "noisy_topk":
            self.router = NoisyTopKRouter(d_model, num_experts, use_bias=use_bias, 
                                         use_switch_gating=use_switch_gating)
        else:
            self.router = nn.Linear(d_model, num_experts, bias=use_bias)
            nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

        # Experts
        expert_class = MoE_SwiGLUExpert if use_swiglu else MoE_FFNExpert
        self.experts = nn.ModuleList([
            expert_class(d_model, d_ff, dropout) if use_swiglu
            else expert_class(d_model, d_ff, dropout, activation)
            for _ in range(num_experts)
        ])

        self.dispatcher = TritonMoEDispatcher(d_model, self.top_k) if TRITON_AVAILABLE else None

        # Stats
        self.register_buffer("_eps", torch.tensor(1e-8))
        self.register_buffer("expert_usage", torch.zeros(num_experts))
        self.register_buffer("momentum", torch.tensor(expert_usage_ema_decay))

        self.aux_loss = 0.0
        self.raw_logits = None
        
        # Reusable buffers
        self._capacity_mask_buffer = None

    def compute_capacity(self, num_tokens: int) -> int:
        """Compute expert capacity."""
        return max(self.min_capacity, int(math.ceil(self.capacity_factor * num_tokens / self.num_experts)))

    def _compute_router_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute router logits."""
        if isinstance(self.router, NoisyTopKRouter):
            raw, logits, _, _ = self.router(x, return_raw_logits=True)
            self.raw_logits = raw
            return logits
        
        logits = self.router(x)
        self.raw_logits = logits
        return logits

    def _update_expert_usage_ema(self, expert_probs: torch.Tensor, expert_indices: torch.Tensor):
        """Update expert usage statistics with EMA."""
        if not self.training:
            return
        
        if expert_indices.dim() == 1:
            expert_indices = expert_indices.unsqueeze(-1)
            expert_probs = expert_probs.unsqueeze(-1)

        current = torch.zeros(self.num_experts, device=expert_probs.device)
        current.scatter_add_(0, expert_indices.view(-1), expert_probs.view(-1))
        current = current / max(expert_indices.size(0), 1)

        decay = self.momentum.item() if isinstance(self.momentum, torch.Tensor) else self.momentum
        self.expert_usage.mul_(decay).add_(current, alpha=1 - decay)

    def _token_choice_routing_optimized(self, x: torch.Tensor, probs: torch.Tensor):
        """Memory-optimized token choice routing."""
        org_shape = x.shape
        x_flat = x.view(-1, self.d_model)
        p_flat = probs.view(-1, self.num_experts)

        # Top-k selection
        if self.use_switch_gating and self.top_k == 1:
            top_p, top_i = torch.max(p_flat, dim=-1, keepdim=True)
        else:
            top_p, top_i = torch.topk(p_flat, self.top_k, dim=-1)

        # Apply expert dropout
        if self.training and self.expert_dropout > 0:
            top_p = F.dropout(top_p, p=self.expert_dropout, training=True)
            top_p = top_p / (top_p.sum(dim=-1, keepdim=True) + self._eps)

        if self.training:
            self._update_expert_usage_ema(top_p, top_i)

        num_tokens = x_flat.size(0)
        capacity = self.compute_capacity(num_tokens)

        # Efficient capacity dropping using vectorized operations
        if self.use_capacity_dropping:
            # Count assignments per expert efficiently
            flat_indices = top_i.view(-1)
            counts_per_expert = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
            counts_per_expert.scatter_add_(0, flat_indices, torch.ones_like(flat_indices))
            
            # Create acceptance mask
            accept = torch.ones_like(top_p, dtype=torch.bool)
            for eid in range(self.num_experts):
                if counts_per_expert[eid] > capacity:
                    mask = (top_i == eid)
                    # Randomly drop excess assignments
                    drop_count = int(counts_per_expert[eid].item()) - capacity
                    indices = mask.nonzero(as_tuple=False)
                    if indices.numel() > drop_count:
                        drop_indices = indices[torch.randperm(indices.size(0), device=x.device)[:drop_count]]
                        accept[drop_indices[:, 0], drop_indices[:, 1]] = False
            
            top_p = top_p * accept.float()
            top_p = top_p / (top_p.sum(dim=-1, keepdim=True) + self._eps)

        self._last_routed_probs = top_p.detach()
        self._last_routed_indices = top_i.detach()

        # Route tokens
        routed = self._route_tokens_to_experts(x_flat, top_p, top_i)
        
        stats = {
            "routing_type": "token_choice",
            "capacity": capacity,
            "expert_usage": self.expert_usage.clone()
        }
        
        return routed.view(*org_shape[:-1], self.d_model), stats

    def _route_tokens_to_experts(
        self, 
        x: torch.Tensor, 
        probs: torch.Tensor, 
        indices: torch.Tensor
    ) -> torch.Tensor:
        """Route tokens with Triton dispatcher when available."""
        if self.dispatcher and indices.dim() == 2 and indices.size(1) == self.top_k:
            return self._dispatch_with_triton(x, probs, indices)
        return self._standard_routing_optimized(x, probs, indices)

    def _dispatch_with_triton(
        self, 
        x: torch.Tensor, 
        probs: torch.Tensor, 
        indices: torch.Tensor
    ) -> torch.Tensor:
        """Triton-accelerated dispatch."""
        expert_in, rows = self.dispatcher.dispatch(x, probs, indices)
        expert_out = torch.empty_like(expert_in)
        counts = torch.bincount(indices.flatten(), minlength=self.num_experts)

        # Process experts
        off = 0
        for eid, cnt in enumerate(counts):
            c = int(cnt.item())
            if c > 0:
                seg = expert_in[off:off + c]
                if self.use_gradient_checkpointing and self.training:
                    out = torch.utils.checkpoint.checkpoint(
                        self.experts[eid], seg, use_reentrant=False
                    )
                else:
                    out = self.experts[eid](seg)
                expert_out[off:off + c] = out
                off += c

        # Combine outputs
        out = torch.zeros_like(x)
        if expert_out.size(0) > 0:
            # Efficient scatter-add
            tok_idx = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand(-1, self.top_k)
            out.scatter_add_(0, tok_idx.reshape(-1, 1).expand(-1, self.d_model), 
                           expert_out[rows.view(-1)])
        
        return out

    def _standard_routing_optimized(
        self, 
        x: torch.Tensor, 
        probs: torch.Tensor, 
        indices: torch.Tensor
    ) -> torch.Tensor:
        """Optimized standard routing with reduced memory."""
        B, D = x.shape
        K = indices.shape[1] if indices.dim() > 1 else 1

        out = torch.zeros_like(x)
        
        # Process each expert efficiently
        for eid in range(self.num_experts):
            # Find tokens assigned to this expert
            if indices.dim() == 1:
                mask = (indices == eid)
                if not mask.any():
                    continue
                xe = x[mask]
                pe = probs[mask].unsqueeze(-1) if probs.dim() == 1 else probs[mask]
            else:
                mask = (indices == eid)
                if not mask.any():
                    continue
                
                # Get token and k positions
                token_positions, k_positions = mask.nonzero(as_tuple=True)
                
                if token_positions.numel() == 0:
                    continue
                
                xe = x[token_positions]
                pe = probs[token_positions, k_positions].unsqueeze(-1)
            
            # Run expert
            if self.use_gradient_checkpointing and self.training:
                ye = torch.utils.checkpoint.checkpoint(
                    self.experts[eid], xe, use_reentrant=False
                )
            else:
                ye = self.experts[eid](xe)
            
            # Weight and accumulate
            ye = ye * pe
            
            if indices.dim() == 1:
                out[mask] = out[mask] + ye
            else:
                out.index_add_(0, token_positions, ye)
        
        return out

    def _compute_auxiliary_loss(
        self, 
        probs: torch.Tensor, 
        indices: torch.Tensor, 
        num_tokens: int
    ) -> torch.Tensor:
        """Compute auxiliary losses efficiently."""
        if not self.training:
            return torch.tensor(0.0, device=probs.device)

        aux = torch.tensor(0.0, device=probs.device)

        # Z-loss (router confidence regularization)
        if self.z_loss_weight > 0 and self.raw_logits is not None:
            log_z = torch.logsumexp(self.raw_logits.view(-1, self.num_experts), dim=-1)
            aux = aux + self.z_loss_weight * (log_z ** 2).mean()

        # Load balancing loss
        if self.load_balance_weight > 0:
            if hasattr(self, "_last_routed_probs") and self._last_routed_probs is not None:
                rp = self._last_routed_probs
                ri = self._last_routed_indices
            else:
                rp = torch.gather(probs, -1, indices if indices.dim() > 1 else indices.unsqueeze(-1))
                ri = indices

            # Count tokens per expert
            counts = torch.zeros(self.num_experts, device=probs.device)
            if ri.dim() == 1:
                counts.scatter_add_(0, ri, rp.squeeze(-1) if rp.dim() > 1 else rp)
            else:
                for k in range(ri.size(1)):
                    valid = ri[:, k] >= 0
                    if valid.any():
                        counts.scatter_add_(0, ri[:, k][valid], rp[:, k][valid])
            
            usage = counts / (counts.sum() + self._eps)
            mean_p = probs.mean(dim=0)
            aux = aux + self.load_balance_weight * (usage * mean_p).sum() * self.num_experts

        return aux

    def forward(self, x: torch.Tensor, return_aux_loss: bool = True):
        """Forward pass with memory optimization."""
        x_norm = self.input_norm(x)
        logits = self._compute_router_logits(x_norm)
        probs = F.softmax(logits, dim=-1)

        out, stats = self._token_choice_routing_optimized(x_norm, probs)

        # Get indices for aux loss
        if self.use_switch_gating and self.top_k == 1:
            _, top_indices = torch.max(probs.view(-1, self.num_experts), dim=-1)
        else:
            _, top_indices = torch.topk(probs.view(-1, self.num_experts), self.top_k, dim=-1)

        num_tokens = x.view(-1, self.d_model).size(0)
        
        # Always compute aux loss if requested
        if return_aux_loss:
            if self.training:
                self.aux_loss = self._compute_auxiliary_loss(
                    probs.view(-1, self.num_experts), top_indices, num_tokens
                )
            else:
                self.aux_loss = torch.tensor(0.0, device=x.device)
            return out, self.aux_loss
        else:
            self.aux_loss = 0.0
            return out

    def get_expert_stats(self) -> Dict[str, torch.Tensor]:
        """Get expert usage statistics."""
        entropy = -(self.expert_usage * torch.log(self.expert_usage + self._eps)).sum()
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
        if hasattr(self, "_last_routed_probs"):
            self._last_routed_probs = None
            self._last_routed_indices = None


class _StandardFeedForwardBlock(nn.Module):
    """Standard FFN block."""
    
    def __init__(self, d_model, dim_ff, dropout=0.1, use_swiglu=True, activation="gelu"):
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
            self.activation = getattr(F, activation.lower(), F.gelu)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.use_swiglu:
            u = self.w1(x)
            v = self.w2(x)
            z = F.silu(u) * v
            if self.training and self.dropout_p > 0:
                z = F.dropout(z, p=self.dropout_p, training=True)
            return self.w3(z)
        else:
            x = self.linear1(x)
            x = self.activation(x)
            x = self.dropout(x)
            return self.linear2(x)


class FeedForwardBlock(nn.Module):
    """Unified feedforward block supporting both standard FFN and MoE."""
    
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

        if use_moe:
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
            self._supports_aux_loss = True
        else:
            self.block = _StandardFeedForwardBlock(
                d_model=d_model,
                dim_ff=dim_ff,
                dropout=dropout,
                use_swiglu=use_swiglu,
                activation=activation,
            )
            self._supports_aux_loss = False

    def forward(self, x, return_aux_loss=False):
        if self.use_moe:
            return self.block(x, return_aux_loss=return_aux_loss)
        out = self.block(x)
        return (out, 0.0) if return_aux_loss else out
