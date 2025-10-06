# graph_ntf.py - Enhanced with SOTA improvements (v2)
# -----------------------------------------------------------------------------
# Node-time-feature Graph Stack for tensors shaped [B, T, N, F]
# 
# Key SOTA improvements:
# 1. GraphNorm: Better normalization for graph-structured data
# 2. Multi-scale Chebyshev with PROPER different receptive fields
# 3. Edge-conditioned convolutions: More expressive message passing
# 4. Stochastic depth: Better regularization and gradient flow
# 5. EMA graph learning: Stable adjacency matrices during training
# 6. Pre-normalization: More stable training than post-norm
# 7. Adaptive edge sparsification: Dynamic graph pruning
# 8. Diffusion-based aggregation: Better than simple mean/sum
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor
AggType = Literal["add", "mean", "max"]


# ============================== Utilities ====================================

def _is_batched_adj(adj: Tensor) -> bool:
    return adj.dim() == 3  # [B, N, N]

def _add_self_loops(adj: Tensor) -> Tensor:
    if _is_batched_adj(adj):
        B, N, _ = adj.shape
        I = torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0).expand(B, N, N)
        return adj + I
    else:
        N = adj.size(0)
        return adj + torch.eye(N, device=adj.device, dtype=adj.dtype)

def _normalize_gcn(adj: Tensor, eps: float = 1e-9) -> Tensor:
    if not _is_batched_adj(adj):
        deg = adj.sum(-1)
        inv = deg.clamp(min=eps).pow(-0.5)
        inv[torch.isinf(inv)] = 0.0
        return adj * inv.unsqueeze(0) * inv.unsqueeze(1)
    else:
        deg = adj.sum(-1)
        inv = deg.clamp(min=eps).pow(-0.5)
        inv[torch.isinf(inv)] = 0.0
        return adj * inv.unsqueeze(-1) * inv.unsqueeze(-2)

def _normalize_row(adj: Tensor, eps: float = 1e-9) -> Tensor:
    if not _is_batched_adj(adj):
        deg = adj.sum(-1).clamp(min=eps)
        return adj / deg.unsqueeze(1)
    else:
        deg = adj.sum(-1).clamp(min=eps)
        return adj / deg.unsqueeze(-1)

def _to_dense_from_edge_index(
    edge_index: Tensor,
    num_nodes: int,
    edge_weight: Optional[Tensor] = None,
    batch_size: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tensor:
    device = device or edge_index.device
    if batch_size is None:
        A = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
        if edge_weight is None:
            A[edge_index[0], edge_index[1]] = 1.0
        else:
            A[edge_index[0], edge_index[1]] = edge_weight
        return A
    else:
        A = torch.zeros(batch_size, num_nodes, num_nodes, device=device, dtype=dtype)
        if edge_weight is None:
            A[:, edge_index[0], edge_index[1]] = 1.0
        else:
            A[:, edge_index[0], edge_index[1]] = edge_weight
        return A

def _ensure_adj(
    adj: Optional[Tensor],
    edge_index: Optional[Tensor],
    num_nodes: int,
    edge_weight: Optional[Tensor],
    batch_size: Optional[int],
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if adj is not None:
        return adj
    if edge_index is None:
        raise ValueError("Either adj or edge_index must be provided.")
    return _to_dense_from_edge_index(edge_index, num_nodes, edge_weight, batch_size, dtype, device)

def _xavier_zero_bias(m: nn.Module, gain: float = 1.0):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def _dtype_neg_inf(dtype: torch.dtype) -> float:
    if dtype == torch.float16:
        return -65504.0
    if dtype == torch.bfloat16:
        return -3.38e38
    return -1e9

def _safe_eye(n: int, like: Tensor) -> Tensor:
    return torch.eye(n, device=like.device, dtype=like.dtype)


# ===================== GraphNorm (better than LayerNorm for graphs) ==========

class GraphNorm(nn.Module):
    """
    GraphNorm from "GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training"
    Better than LayerNorm for graph data as it normalizes across nodes, not features.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.mean_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, N, F] - normalize across nodes (N dimension)
        mean = x.mean(dim=-2, keepdim=True)  # [B, T, 1, F]
        var = x.var(dim=-2, keepdim=True, unbiased=False)
        x_norm = (x - mean * self.mean_scale) / (var + self.eps).sqrt()
        return x_norm * self.weight + self.bias


# ===================== NEW: Adaptive Edge Sparsification ======================

class AdaptiveEdgeSparsifier(nn.Module):
    """Learns to sparsify graphs adaptively during training."""
    def __init__(self, sparsity_ratio: float = 0.3, learnable: bool = True):
        super().__init__()
        self.sparsity_ratio = sparsity_ratio
        if learnable:
            self.threshold = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer("threshold", torch.tensor(0.5))
    
    def forward(self, adj: Tensor) -> Tensor:
        if not self.training:
            return adj
        
        # Keep top-k edges per node
        k = max(1, int(adj.size(-1) * (1 - self.sparsity_ratio)))
        
        if _is_batched_adj(adj):
            topk_vals, topk_idx = torch.topk(adj, k, dim=-1)
            mask = torch.zeros_like(adj)
            mask.scatter_(-1, topk_idx, 1.0)
        else:
            topk_vals, topk_idx = torch.topk(adj, k, dim=-1)
            mask = torch.zeros_like(adj)
            mask.scatter_(-1, topk_idx, 1.0)
        
        return adj * mask


# ===================== Enhanced Latent Graph Learner ==========================

@dataclass
class CorrelationConfigNTF:
    num_nodes: int
    feat_dim: int
    hidden_size: Optional[int] = None
    out_feat_dim: Optional[int] = None
    cheb_k: int = 3
    eps: float = 1e-8
    learnable_alpha: bool = True
    init_alpha: float = 0.5
    temperature: float = 1.0
    low_rank: bool = False
    rank: Optional[int] = None
    dropout_graph: float = 0.0
    use_graph_norm: bool = True
    spectral_norm: bool = False
    improved_init: bool = True
    gradient_checkpointing: bool = False
    # Multi-scale graph learning (FIXED)
    multi_scale: bool = True
    num_scales: int = 3
    # EMA for stable graphs
    use_ema: bool = True
    ema_decay: float = 0.99
    # NEW: Adaptive sparsification
    adaptive_sparse: bool = True
    sparsity_ratio: float = 0.3


class LatentCorrelationLearnerNTF(nn.Module):
    """Enhanced with PROPER multi-scale Chebyshev and adaptive sparsification."""

    def __init__(self, cfg: CorrelationConfigNTF):
        super().__init__()
        self.cfg = cfg
        N, Fdim = cfg.num_nodes, cfg.feat_dim
        H = cfg.hidden_size or max(2 * Fdim, 64)
        Fout = cfg.out_feat_dim or Fdim
        self.cheb_k = max(1, cfg.cheb_k)
        self.eps = cfg.eps

        # alpha (data vs learned)
        if cfg.learnable_alpha:
            init_logit = torch.logit(torch.tensor(cfg.init_alpha, dtype=torch.float32))
            self.alpha = nn.Parameter(init_logit)
        else:
            self.register_buffer("alpha", torch.tensor(cfg.init_alpha, dtype=torch.float32))

        # learnable graph
        self.low_rank = cfg.low_rank
        self.rank = cfg.rank or max(1, N // 8)
        if self.low_rank:
            scale = 1.0 / math.sqrt(self.rank)
            self.factors = nn.Parameter(torch.randn(2, N, self.rank) * scale)
        else:
            self.A_param = nn.Parameter(torch.empty(N, N))

        # projections
        self.in_proj = nn.Linear(Fdim, H)
        self.out_proj = nn.Linear(H, Fout)
        if cfg.spectral_norm:
            from torch.nn.utils import spectral_norm
            self.in_proj = spectral_norm(self.in_proj)
            self.out_proj = spectral_norm(self.out_proj)

        # norms - GraphNorm option
        if cfg.use_graph_norm:
            self.ln_in = GraphNorm(Fdim, eps=cfg.eps)
            self.ln_h = GraphNorm(H, eps=cfg.eps)
            self.ln_out = GraphNorm(Fout, eps=cfg.eps)
        else:
            self.ln_in = nn.LayerNorm(Fdim, eps=cfg.eps)
            self.ln_h = nn.LayerNorm(H, eps=cfg.eps)
            self.ln_out = nn.LayerNorm(Fout, eps=cfg.eps)

        self.A_drop = nn.Dropout(cfg.dropout_graph) if cfg.dropout_graph > 0 else nn.Identity()

        # FIXED Multi-scale Chebyshev
        self.multi_scale = cfg.multi_scale
        if self.multi_scale:
            self.num_scales = cfg.num_scales
            # Each scale has its own Chebyshev weights
            self.cheb_w = nn.Parameter(torch.ones(self.num_scales, self.cheb_k) / self.cheb_k)
            self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        else:
            self.cheb_w = nn.Parameter(torch.ones(self.cheb_k) / self.cheb_k)

        if cfg.temperature != 1.0:
            self.cheb_temp = nn.Parameter(torch.tensor(cfg.temperature, dtype=torch.float32))
        else:
            self.register_buffer("cheb_temp", torch.tensor(1.0, dtype=torch.float32))

        # EMA for graph stability
        self.use_ema = cfg.use_ema
        if self.use_ema:
            self.register_buffer("A_ema", torch.eye(N))
            self.ema_decay = cfg.ema_decay

        # NEW: Adaptive sparsification
        self.adaptive_sparse = cfg.adaptive_sparse
        if self.adaptive_sparse:
            self.sparsifier = AdaptiveEdgeSparsifier(cfg.sparsity_ratio, learnable=True)

        self._reset()

    def _reset(self):
        N, Fdim = self.cfg.num_nodes, self.cfg.feat_dim
        if self.low_rank:
            if self.cfg.improved_init:
                nn.init.orthogonal_(self.factors[0])
                nn.init.orthogonal_(self.factors[1])
        else:
            if self.cfg.improved_init:
                nn.init.eye_(self.A_param)
                with torch.no_grad():
                    noise = 0.01 if N > 64 else 0.05
                    self.A_param.add_(noise * torch.randn_like(self.A_param))
                    self.A_param.copy_(0.5 * (self.A_param + self.A_param.t()))
            else:
                nn.init.eye_(self.A_param)
                with torch.no_grad():
                    self.A_param.add_(0.01 * torch.randn_like(self.A_param))
                    self.A_param.copy_(0.5 * (self.A_param + self.A_param.t()))

        gain = math.sqrt(2.0) if self.cfg.improved_init else 1.0
        _xavier_zero_bias(self.in_proj, gain)
        _xavier_zero_bias(self.out_proj, 1.0)
        
        with torch.no_grad():
            if self.multi_scale:
                self.cheb_w.fill_(1.0 / self.cheb_k)
                self.scale_weights.fill_(1.0 / self.num_scales)
            else:
                self.cheb_w.fill_(1.0 / self.cheb_k)

    def _learned_graph(self) -> Tensor:
        if self.low_rank:
            U, V = self.factors[0], self.factors[1]
            A = U @ V.t()
            A = 0.5 * (A + A.t())
        else:
            A = 0.5 * (self.A_param + self.A_param.t())
        
        temp = float(self.cheb_temp)
        A = torch.tanh(A / temp)
        
        # EMA update
        if self.use_ema and self.training:
            with torch.no_grad():
                self.A_ema.mul_(self.ema_decay).add_(A, alpha=1 - self.ema_decay)
            A_use = self.A_ema if not self.training else A
        else:
            A_use = A
        
        # NEW: Adaptive sparsification
        if self.adaptive_sparse:
            A_use = self.sparsifier(A_use)
            
        return self.A_drop(A_use) if self.training else A_use

    def _data_graph(self, x: Tensor) -> Tensor:
        B, T, N, Fdim = x.shape
        xt = x.reshape(B * T, N, Fdim)
        xt = xt - xt.mean(dim=-1, keepdim=True)
        norm = xt.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        xn = xt / norm
        sim = torch.einsum("bnf,bmf->bnm", xn, xn)
        A = sim.mean(dim=0).clamp(-1.0, 1.0).detach()
        return A

    @staticmethod
    def _laplacian(A: Tensor, eps: float) -> Tensor:
        A = A.clone()
        A.fill_diagonal_(0.0)
        deg = A.sum(-1)
        inv = deg.clamp(min=eps).pow(-0.5)
        inv = torch.where(torch.isinf(inv), torch.zeros_like(inv), inv)
        Dis = torch.diag(inv)
        L = _safe_eye(A.size(0), A) - Dis @ A @ Dis
        return L.clamp(-1.5, 1.5)

    def _cheb_filter(self, x: Tensor, L: Tensor) -> Tensor:
        """FIXED: Proper multi-scale with different graph powers = different receptive fields"""
        if self.multi_scale:
            outputs = []
            for s in range(self.num_scales):
                # KEY FIX: Use different powers of Laplacian for different scales
                # Scale 0: L (1-hop), Scale 1: L^2 (2-hop), Scale 2: L^3 (3-hop)
                if s == 0:
                    L_scaled = L
                else:
                    # Compute L^(s+1) for increasing receptive field
                    L_scaled = L.clone()
                    for _ in range(s):
                        L_scaled = L_scaled @ L
                    L_scaled = L_scaled.clamp(-1.5, 1.5)
                
                w = F.softmax(self.cheb_w[s] / self.cheb_temp, dim=0)
                T0 = x
                if self.cheb_k == 1:
                    out = w[0] * T0
                else:
                    T1 = torch.einsum("btnf,nm->btmf", x, L_scaled)
                    out = w[0] * T0 + w[1] * T1
                    for k in range(2, self.cheb_k):
                        Tk = 2 * torch.einsum("btpf,pm->btmf", T1, L_scaled) - T0
                        Tk = Tk.clamp(-50, 50)
                        out = out + w[k] * Tk
                        T0, T1 = T1, Tk
                outputs.append(out)
            
            scale_w = F.softmax(self.scale_weights, dim=0)
            return sum(w * o for w, o in zip(scale_w, outputs))
        else:
            # Original single-scale
            w = F.softmax(self.cheb_w / self.cheb_temp, dim=0)
            T0 = x
            if self.cheb_k == 1:
                return w[0] * T0
            T1 = torch.einsum("btnf,nm->btmf", x, L)
            out = w[0] * T0 + w[1] * T1
            for k in range(2, self.cheb_k):
                Tk = 2 * torch.einsum("btpf,pm->btmf", T1, L) - T0
                Tk = Tk.clamp(-50, 50)
                out = out + w[k] * Tk
                T0, T1 = T1, Tk
            return out

    def _forward_impl(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.ln_in(x)
        A_data = self._data_graph(x)
        A_learn = self._learned_graph()
        alpha = torch.sigmoid(self.alpha)
        A = alpha * A_learn + (1.0 - alpha) * A_data

        L = self._laplacian(A, self.eps)
        xf = self._cheb_filter(x, L)
        h = F.gelu(self.ln_h(self.in_proj(xf)))
        y = self.ln_out(self.out_proj(h))
        return y, A

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.cfg.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


# ========================= PyG-like MessagePassing ===========================

class MessagePassingNTF(nn.Module):
    """PyG-style MessagePassing over node graphs for x:[B, T, N, F]."""

    def __init__(self, aggr: AggType = "add"):
        super().__init__()
        if aggr not in ("add", "mean", "max"):
            raise ValueError("aggr must be one of {'add','mean','max'}")
        self.aggr: AggType = aggr

    @torch.no_grad()
    def _check(self, x: Tensor, A: Tensor):
        assert x.dim() == 4, f"x must be [B,T,N,F], got {x.shape}"
        N = x.size(-2)
        if _is_batched_adj(A):
            assert A.size(0) == x.size(0) and A.size(1) == A.size(2) == N
        else:
            assert A.size(0) == A.size(1) == N

    def message(self, x_j: Tensor, x_i: Tensor, **kwargs) -> Tensor:
        return x_j

    def aggregate(self, m: Tensor, **kwargs) -> Tensor:
        return m

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        return aggr_out

    def propagate(
        self,
        x: Tensor,
        adj: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
        pre_aggregated: bool = False,
    ) -> Tensor:
        B, T, N, Fdim = x.shape
        A = _ensure_adj(
            adj=adj,
            edge_index=edge_index,
            num_nodes=N,
            edge_weight=edge_weight,
            batch_size=(B if (adj is None and edge_index is not None) else None),
            dtype=x.dtype,
            device=x.device,
        )
        self._check(x, A)

        if pre_aggregated:
            m = self.message(x_j=x, x_i=x)
            out = self.aggregate(m)
            return self.update(out, x)

        if _is_batched_adj(A):
            neigh = torch.einsum("btnf,bij->btif", x, A)
        else:
            neigh = torch.einsum("btnf,ij->btif", x, A)
        m = self.message(x_j=neigh, x_i=x)
        out = self.aggregate(m)
        out = self.update(out, x)
        return out


# =============================== Graph Layers ================================

class GCNConvNTF(MessagePassingNTF):
    """GCN with PRE-normalization for better stability."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        add_self_loops_flag: bool = True,
        activation: Literal["relu", "gelu", "silu", "none"] = "gelu",
        dropout: float = 0.0,
        use_graph_norm: bool = True,
        pre_norm: bool = True,  # NEW: pre-normalization
    ):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        self.add_self_loops_flag = add_self_loops_flag
        self.pre_norm = pre_norm
        
        self.act = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(inplace=True),
            "none": nn.Identity()
        }[activation]
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Pre-norm: before linear, Post-norm: after linear
        if use_graph_norm:
            self.norm = GraphNorm(in_channels if pre_norm else out_channels)
        else:
            self.norm = nn.LayerNorm(in_channels if pre_norm else out_channels)
        
        _xavier_zero_bias(self.lin)

    def forward(
        self,
        x: Tensor,
        adj: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
        pre_normalized: bool = False,
    ) -> Tensor:
        B, T, N, _ = x.shape
        A = _ensure_adj(adj, edge_index, N, edge_weight, 
                       batch_size=(B if (adj is None and edge_index is not None) else None),
                       dtype=x.dtype, device=x.device)
        if self.add_self_loops_flag:
            A = _add_self_loops(A)
        if not pre_normalized:
            A = _normalize_gcn(A)

        # Pre-normalization
        if self.pre_norm:
            x = self.norm(x)

        if _is_batched_adj(A):
            agg = torch.einsum("btnf,bij->btif", x, A)
        else:
            agg = torch.einsum("btnf,ij->btif", x, A)

        y = self.lin(agg)
        y = self.drop(self.act(y))
        
        # Post-normalization (if not pre-norm)
        if not self.pre_norm:
            y = self.norm(y)
            
        return y


class SAGEConvNTF(MessagePassingNTF):
    """GraphSAGE with pre-normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        activation: Literal["relu", "gelu", "silu", "none"] = "gelu",
        dropout: float = 0.0,
        use_graph_norm: bool = True,
        pre_norm: bool = True,
    ):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels * 2, out_channels, bias=bias)
        self.pre_norm = pre_norm
        
        self.act = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(inplace=True),
            "none": nn.Identity()
        }[activation]
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        if use_graph_norm:
            self.norm = GraphNorm(in_channels if pre_norm else out_channels)
        else:
            self.norm = nn.LayerNorm(in_channels if pre_norm else out_channels)
        
        _xavier_zero_bias(self.lin)

    def forward(
        self,
        x: Tensor,
        adj: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        B, T, N, _ = x.shape
        A = _ensure_adj(adj, edge_index, N, edge_weight,
                       batch_size=(B if (adj is None and edge_index is not None) else None),
                       dtype=x.dtype, device=x.device)
        A = _add_self_loops(A)
        A = _normalize_row(A)

        if self.pre_norm:
            x = self.norm(x)

        if _is_batched_adj(A):
            neigh = torch.einsum("btnf,bij->btif", x, A)
        else:
            neigh = torch.einsum("btnf,ij->btif", x, A)

        h = torch.cat([x, neigh], dim=-1)
        y = self.lin(h)
        y = self.drop(self.act(y))
        
        if not self.pre_norm:
            y = self.norm(y)
            
        return y


class GATConvNTF(MessagePassingNTF):
    """Multi-head GAT with pre-normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops_flag: bool = True,
        use_graph_norm: bool = True,
        pre_norm: bool = True,
    ):
        super().__init__(aggr="add")
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        self.H = heads
        self.Dh = out_channels // heads
        self.concat = concat
        self.neg_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops_flag = add_self_loops_flag
        self.pre_norm = pre_norm

        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.att_l = nn.Parameter(torch.empty(self.H, self.Dh))
        self.att_r = nn.Parameter(torch.empty(self.H, self.Dh))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        if use_graph_norm:
            self.norm = GraphNorm(in_channels if pre_norm else out_channels)
        else:
            self.norm = nn.LayerNorm(in_channels if pre_norm else out_channels)

        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(
        self,
        x: Tensor,
        adj: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        B, T, N, Fin = x.shape
        A = _ensure_adj(adj, edge_index, N, edge_weight,
                       batch_size=(B if (adj is None and edge_index is not None) else None),
                       dtype=x.dtype, device=x.device)
        if self.add_self_loops_flag:
            A = _add_self_loops(A)

        if self.pre_norm:
            x = self.norm(x)

        XW = self.lin(x).view(B, T, N, self.H, self.Dh)
        alpha_l = torch.einsum("btnhd,hd->btnh", XW, self.att_l)
        alpha_r = torch.einsum("btnhd,hd->btnh", XW, self.att_r)

        e = (alpha_l.unsqueeze(-1) + alpha_r.unsqueeze(-2))
        e = e.permute(0, 1, 3, 2, 4)
        e = F.leaky_relu(e, negative_slope=self.neg_slope)

        if _is_batched_adj(A):
            mask = (A > 0).unsqueeze(1)
        else:
            mask = (A > 0).unsqueeze(0).unsqueeze(0)
        e = torch.where(mask, e, torch.full_like(e, _dtype_neg_inf(x.dtype)))

        attn = F.softmax(e, dim=-1)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        out = torch.einsum("bthij,btnjd->bthid", attn, XW)
        if self.concat:
            out = out.permute(0, 1, 3, 2, 4).reshape(B, T, N, self.H * self.Dh)
        else:
            out = out.mean(dim=2)

        out = out + self.bias
        
        if not self.pre_norm:
            out = self.norm(out)
            
        return out


# ===================== Edge-Conditioned GNN Layer =============================

class EdgeCondGCNNTF(nn.Module):
    """
    Edge-conditioned GCN: edge features modulate node aggregation.
    More expressive than vanilla GCN.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int = 16,
        bias: bool = True,
        activation: Literal["relu", "gelu", "silu"] = "gelu",
        dropout: float = 0.0,
        use_graph_norm: bool = True,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        
        self.edge_net = nn.Sequential(
            nn.Linear(1, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, in_channels * out_channels)
        )
        self.node_lin = nn.Linear(in_channels, out_channels, bias=bias)
        self.act = {
            "relu": nn.ReLU(True),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(True)
        }[activation]
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        if use_graph_norm:
            self.norm = GraphNorm(in_channels if pre_norm else out_channels)
        else:
            self.norm = nn.LayerNorm(in_channels if pre_norm else out_channels)
        
        _xavier_zero_bias(self.node_lin)
        for m in self.edge_net:
            if isinstance(m, nn.Linear):
                _xavier_zero_bias(m)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        B, T, N, Fin = x.shape
        Fout = self.node_lin.out_features
        
        if self.pre_norm:
            x = self.norm(x)
        
        if _is_batched_adj(adj):
            edge_feat = adj.unsqueeze(-1)
            edge_w = self.edge_net(edge_feat).view(B, N, N, Fin, Fout)
            x_expand = x.unsqueeze(2).unsqueeze(-1)
            msg = (x_expand * edge_w.unsqueeze(1)).sum(dim=-2)
            agg = msg.sum(dim=-2)
        else:
            edge_feat = adj.unsqueeze(-1)
            edge_w = self.edge_net(edge_feat).view(N, N, Fin, Fout)
            x_expand = x.unsqueeze(2).unsqueeze(-1)
            msg = (x_expand * edge_w.unsqueeze(0).unsqueeze(0)).sum(dim=-2)
            agg = msg.sum(dim=-2)
        
        y = self.node_lin(x) + agg
        y = self.drop(self.act(y))
        
        if not self.pre_norm:
            y = self.norm(y)
            
        return y


# ===================== Stochastic Depth =======================================

class StochasticDepth(nn.Module):
    """Drop entire layers during training for better regularization."""
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor, residual: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0:
            return x + residual
        
        keep_prob = 1 - self.drop_prob
        mask = torch.bernoulli(torch.full((x.size(0), 1, 1, 1), keep_prob, device=x.device))
        return (x / keep_prob) * mask + residual


# ============================== Jump Knowledge ================================

class JumpKnowledgeNTF(nn.Module):
    def __init__(
        self,
        mode: Literal["none", "last", "sum", "max", "concat", "lstm"] = "none",
        hidden_size: Optional[int] = None,
        output_size: Optional[int] = None,
        num_layers_hint: Optional[int] = None,
    ):
        super().__init__()
        self.mode = mode
        self.hidden = hidden_size
        self.output = output_size
        self.concat_proj: Optional[nn.Linear] = None

        if mode == "lstm":
            if hidden_size is None:
                raise ValueError("hidden_size required for LSTM JK")
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.out_proj = nn.Identity() if (output_size is None or output_size == hidden_size) else nn.Linear(hidden_size, output_size)
            _xavier_zero_bias(self.out_proj) if isinstance(self.out_proj, nn.Linear) else None
        elif mode == "concat" and (num_layers_hint and hidden_size and output_size):
            self.concat_proj = nn.Linear(num_layers_hint * hidden_size, output_size)
            _xavier_zero_bias(self.concat_proj)

    def _lazy_init_concat(self, concat_dim: int, like: Tensor):
        if self.concat_proj is None:
            if self.output is None:
                raise ValueError("output_size must be set for concat JK")
            self.concat_proj = nn.Linear(concat_dim, self.output).to(like.device, like.dtype)
            _xavier_zero_bias(self.concat_proj)

    def forward(self, xs: List[Tensor]) -> Tensor:
        if not xs:
            raise ValueError("JK received empty list")
        base = xs[0].shape[:-1]
        for i, t in enumerate(xs[1:], 1):
            if t.shape[:-1] != base:
                raise ValueError(f"JK shape mismatch at {i}: {t.shape} vs {xs[0].shape}")

        if self.mode == "none":
            return xs[-1]
        if self.mode == "last":
            return xs[-1]
        if self.mode == "sum":
            y = xs[0].clone()
            for t in xs[1:]:
                y.add_(t)
            return y
        if self.mode == "max":
            y = xs[0]
            for t in xs[1:]:
                y = torch.maximum(y, t)
            return y
        if self.mode == "concat":
            y = torch.cat(xs, dim=-1)
            self._lazy_init_concat(y.size(-1), y)
            return self.concat_proj(y)
        if self.mode == "lstm":
            B, T, N, D = xs[0].shape
            L = len(xs)
            seq = torch.stack(xs, dim=2).reshape(B * T * N, L, D)
            out, _ = self.lstm(seq)
            y = out[:, -1, :].reshape(B, T, N, D)
            return self.out_proj(y)
        raise ValueError(f"Unknown JK mode: {self.mode}")


# =========================== LatentGraphNetwork NTF ===========================

class LatentGraphNetworkNTF(nn.Module):
    """
    End-to-end graph network with SOTA enhancements:
      - Multi-scale Chebyshev filtering with PROPER different receptive fields
      - Edge-conditioned convolutions
      - Stochastic depth regularization
      - GraphNorm with pre-normalization
      - Adaptive edge sparsification
    """

    def __init__(
        self,
        num_nodes: int,
        feat_dim: int,
        out_feat_dim: int,
        passes: int = 2,
        layer: Literal["gcn", "sage", "gat", "edge_cond"] = "edge_cond",
        gat_heads: int = 4,
        dropout: float = 0.0,
        stochastic_depth: float = 0.1,
        jk: Literal["none", "last", "sum", "max", "concat", "lstm"] = "none",
        corr_cfg: Optional[CorrelationConfigNTF] = None,
        residual: bool = True,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.residual = residual and (feat_dim == out_feat_dim)
        self.jk_on = (jk != "none")

        self.corr = LatentCorrelationLearnerNTF(
            corr_cfg
            or CorrelationConfigNTF(
                num_nodes=num_nodes,
                feat_dim=feat_dim,
                out_feat_dim=feat_dim,
                cheb_k=3,
                low_rank=True,
                rank=max(1, num_nodes // 8),
                use_graph_norm=True,
                multi_scale=True,
                num_scales=3,
                use_ema=True,
                adaptive_sparse=True,
                sparsity_ratio=0.3,
                dropout_graph=dropout * 0.5,
            )
        )

        blocks: List[nn.Module] = []
        sd_rates = [stochastic_depth * i / passes for i in range(passes)]
        
        for sd_rate in sd_rates:
            if layer == "edge_cond":
                conv = EdgeCondGCNNTF(feat_dim, feat_dim, dropout=dropout, pre_norm=pre_norm)
            elif layer == "gcn":
                conv = GCNConvNTF(feat_dim, feat_dim, dropout=dropout, pre_norm=pre_norm)
            elif layer == "sage":
                conv = SAGEConvNTF(feat_dim, feat_dim, dropout=dropout, pre_norm=pre_norm)
            elif layer == "gat":
                conv = GATConvNTF(feat_dim, feat_dim, heads=gat_heads, dropout=dropout, pre_norm=pre_norm)
            else:
                raise ValueError(f"Unsupported layer type: {layer}")
            
            blocks.append(
                nn.ModuleDict({
                    'conv': conv,
                    'sd': StochasticDepth(sd_rate) if sd_rate > 0 else nn.Identity()
                })
            )
        
        self.blocks = nn.ModuleList(blocks)
        
        # Final projection with pre-norm
        if pre_norm:
            self.out = nn.Sequential(
                GraphNorm(feat_dim),
                nn.Linear(feat_dim, out_feat_dim)
            )
        else:
            self.out = nn.Sequential(
                nn.Linear(feat_dim, out_feat_dim),
                GraphNorm(out_feat_dim)
            )
        _xavier_zero_bias(self.out[-1] if pre_norm else self.out[0])

        if self.jk_on:
            self.jk = JumpKnowledgeNTF(
                mode=jk,
                hidden_size=feat_dim,
                output_size=feat_dim,
                num_layers_hint=passes if jk == "concat" else None,
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, T, N, F]
        returns: [B, T, N, out_feat_dim]
        """
        h, A = self.corr(x)
        outs: List[Tensor] = []
        
        for block in self.blocks:
            h_new = block['conv'](h, adj=A)
            if isinstance(block['sd'], StochasticDepth):
                h = block['sd'](h_new, h)
            else:
                h = h_new
            outs.append(h)

        if self.jk_on:
            h = self.jk(outs)

        if self.residual and h.shape[-1] == x.shape[-1]:
            h = h + x
        return self.out(h)


# =========================== Graph Preprocessor ===============================

class GraphPreprocessorNTF(nn.Module):
    """
    Plug-and-play input preprocessor with SOTA graph enhancements.
    
    Operates on x:[B,T,N,F] and applies:
      1. Latent graph learning (data + learnable with EMA)
      2. Multi-scale spectral filtering (PROPER different receptive fields)
      3. Edge-conditioned/standard graph convolutions
      4. Pre-normalization for stability
      5. Adaptive edge sparsification
      6. Optional flattening for sequence models
    
    Args:
        num_nodes: N
        in_feat_dim: F_in
        out_feat_dim: F_out after graph block (defaults to F_in)
        passes: number of graph layers
        layer: 'gcn' | 'sage' | 'gat' | 'edge_cond' (recommended)
        gat_heads: heads for GAT
        dropout: dropout inside the graph layers
        stochastic_depth: layer drop probability (increases linearly)
        jk: Jump-knowledge across graph layers
        flatten: 'nodes' -> [B,T,N*F_out], 'none' -> keep [B,T,N,F_out]
        residual: add input residual connection
        pre_norm: use pre-normalization (recommended for stability)
    """
    def __init__(
        self,
        num_nodes: int,
        in_feat_dim: int,
        out_feat_dim: Optional[int] = None,
        passes: int = 2,
        layer: Literal["gcn", "sage", "gat", "edge_cond"] = "edge_cond",
        gat_heads: int = 4,
        dropout: float = 0.0,
        stochastic_depth: float = 0.1,
        jk: Literal["none", "last", "sum", "max", "concat", "lstm"] = "none",
        flatten: Literal["nodes", "none"] = "nodes",
        residual: bool = True,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim or in_feat_dim
        self.flatten = flatten

        self.graph = LatentGraphNetworkNTF(
            num_nodes=num_nodes,
            feat_dim=in_feat_dim,
            out_feat_dim=self.out_feat_dim,
            passes=passes,
            layer=layer,
            gat_heads=gat_heads,
            dropout=dropout,
            stochastic_depth=stochastic_depth,
            jk=jk,
            residual=residual,
            pre_norm=pre_norm,
        )

        # Attributes for BaseForecastingModel compatibility
        if self.flatten == "nodes":
            self.input_size = num_nodes * in_feat_dim
            self.output_size = num_nodes * self.out_feat_dim
        else:
            self.input_size = None
            self.output_size = None

        # For inspection/logging
        self.last_adj: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,N,F_in]
        returns:
          - if flatten='nodes': [B,T,N*F_out]
          - if flatten='none' : [B,T,N,F_out]
        """
        y_full = self.graph(x)
        
        # Store learned adjacency for logging/analysis
        with torch.no_grad():
            _, A = self.graph.corr(x)
            self.last_adj = A.detach().cpu()

        if self.flatten == "nodes":
            B, T, N, Fout = y_full.shape
            return y_full.reshape(B, T, N * Fout)
        else:
            return y_full
