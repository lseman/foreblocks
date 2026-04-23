from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Tensor
from .common import safe_eye
from .common import xavier_zero_bias
from .norms import GraphNorm


class AdaptiveEdgeSparsifier(nn.Module):
    """Keep only the strongest edges during training."""

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

        k = max(1, int(adj.size(-1) * (1 - self.sparsity_ratio)))
        _, topk_idx = torch.topk(adj, k, dim=-1)
        mask = torch.zeros_like(adj)
        mask.scatter_(-1, topk_idx, 1.0)
        return adj * mask


@dataclass
class CorrelationConfig:
    num_nodes: int
    feat_dim: int
    hidden_size: int | None = None
    out_feat_dim: int | None = None
    cheb_k: int = 3
    eps: float = 1e-8
    learnable_alpha: bool = True
    init_alpha: float = 0.5
    temperature: float = 1.0
    low_rank: bool = False
    rank: int | None = None
    dropout_graph: float = 0.0
    use_graph_norm: bool = True
    spectral_norm: bool = False
    improved_init: bool = True
    gradient_checkpointing: bool = False
    multi_scale: bool = True
    num_scales: int = 3
    use_ema: bool = True
    ema_decay: float = 0.99
    adaptive_sparse: bool = True
    sparsity_ratio: float = 0.3


class LatentCorrelationLearner(nn.Module):
    """Learn a stable latent graph from data and parameters."""

    def __init__(self, cfg: CorrelationConfig):
        super().__init__()
        self.cfg = cfg
        num_nodes, feat_dim = cfg.num_nodes, cfg.feat_dim
        hidden = cfg.hidden_size or max(2 * feat_dim, 64)
        out_feat_dim = cfg.out_feat_dim or feat_dim
        self.cheb_k = max(1, cfg.cheb_k)
        self.eps = cfg.eps

        if cfg.learnable_alpha:
            init_logit = torch.logit(torch.tensor(cfg.init_alpha, dtype=torch.float32))
            self.alpha = nn.Parameter(init_logit)
        else:
            self.register_buffer(
                "alpha", torch.tensor(cfg.init_alpha, dtype=torch.float32)
            )

        self.low_rank = cfg.low_rank
        self.rank = cfg.rank or max(1, num_nodes // 8)
        if self.low_rank:
            scale = 1.0 / math.sqrt(self.rank)
            self.factors = nn.Parameter(torch.randn(2, num_nodes, self.rank) * scale)
        else:
            self.A_param = nn.Parameter(torch.empty(num_nodes, num_nodes))

        self.in_proj = nn.Linear(feat_dim, hidden)
        self.out_proj = nn.Linear(hidden, out_feat_dim)
        if cfg.spectral_norm:
            from torch.nn.utils import spectral_norm

            self.in_proj = spectral_norm(self.in_proj)
            self.out_proj = spectral_norm(self.out_proj)

        if cfg.use_graph_norm:
            self.ln_in = GraphNorm(feat_dim, eps=cfg.eps)
            self.ln_h = GraphNorm(hidden, eps=cfg.eps)
            self.ln_out = GraphNorm(out_feat_dim, eps=cfg.eps)
        else:
            self.ln_in = nn.LayerNorm(feat_dim, eps=cfg.eps)
            self.ln_h = nn.LayerNorm(hidden, eps=cfg.eps)
            self.ln_out = nn.LayerNorm(out_feat_dim, eps=cfg.eps)

        self.A_drop = (
            nn.Dropout(cfg.dropout_graph) if cfg.dropout_graph > 0 else nn.Identity()
        )

        self.multi_scale = cfg.multi_scale
        if self.multi_scale:
            self.num_scales = cfg.num_scales
            self.cheb_w = nn.Parameter(
                torch.ones(self.num_scales, self.cheb_k) / self.cheb_k
            )
            self.scale_weights = nn.Parameter(
                torch.ones(self.num_scales) / self.num_scales
            )
        else:
            self.cheb_w = nn.Parameter(torch.ones(self.cheb_k) / self.cheb_k)

        if cfg.temperature != 1.0:
            self.cheb_temp = nn.Parameter(
                torch.tensor(cfg.temperature, dtype=torch.float32)
            )
        else:
            self.register_buffer("cheb_temp", torch.tensor(1.0, dtype=torch.float32))

        self.use_ema = cfg.use_ema
        if self.use_ema:
            self.register_buffer("A_ema", torch.eye(num_nodes))
            self.ema_decay = cfg.ema_decay

        self.adaptive_sparse = cfg.adaptive_sparse
        if self.adaptive_sparse:
            self.sparsifier = AdaptiveEdgeSparsifier(cfg.sparsity_ratio, learnable=True)

        self._reset()

    def _reset(self) -> None:
        num_nodes = self.cfg.num_nodes
        if self.low_rank:
            if self.cfg.improved_init:
                nn.init.orthogonal_(self.factors[0])
                nn.init.orthogonal_(self.factors[1])
        else:
            nn.init.eye_(self.A_param)
            with torch.no_grad():
                noise = 0.01 if num_nodes > 64 else 0.05
                self.A_param.add_(noise * torch.randn_like(self.A_param))
                self.A_param.copy_(0.5 * (self.A_param + self.A_param.t()))

        gain = math.sqrt(2.0) if self.cfg.improved_init else 1.0
        xavier_zero_bias(self.in_proj, gain)
        xavier_zero_bias(self.out_proj, 1.0)

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

        if self.use_ema and self.training:
            with torch.no_grad():
                self.A_ema.mul_(self.ema_decay).add_(A, alpha=1 - self.ema_decay)
            A_use = A
        else:
            A_use = self.A_ema if self.use_ema else A

        if self.adaptive_sparse:
            A_use = self.sparsifier(A_use)

        return self.A_drop(A_use) if self.training else A_use

    def _data_graph(self, x: Tensor) -> Tensor:
        batch_size, steps, num_nodes, feat_dim = x.shape
        xt = x.reshape(batch_size * steps, num_nodes, feat_dim)
        xt = xt - xt.mean(dim=-1, keepdim=True)
        norm = xt.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        xn = xt / norm
        sim = torch.einsum("bnf,bmf->bnm", xn, xn)
        return sim.mean(dim=0).clamp(-1.0, 1.0).detach()

    @staticmethod
    def _laplacian(A: Tensor, eps: float) -> Tensor:
        A = A.clone()
        A.fill_diagonal_(0.0)
        deg = A.sum(-1)
        inv = deg.clamp(min=eps).pow(-0.5)
        inv = torch.where(torch.isinf(inv), torch.zeros_like(inv), inv)
        Dis = torch.diag(inv)
        L = safe_eye(A.size(0), A) - Dis @ A @ Dis
        return L.clamp(-1.5, 1.5)

    def _cheb_filter_single_scale(
        self, x: Tensor, L: Tensor, weights: Tensor
    ) -> Tensor:
        T0 = x
        if self.cheb_k == 1:
            return weights[0] * T0
        T1 = torch.einsum("btnf,nm->btmf", x, L)
        out = weights[0] * T0 + weights[1] * T1
        for _k in range(2, self.cheb_k):
            Tk = 2 * torch.einsum("btpf,pm->btmf", T1, L) - T0
            Tk = Tk.clamp(-50, 50)
            out = out + weights[_k] * Tk
            T0, T1 = T1, Tk
        return out

    def _cheb_filter(self, x: Tensor, L: Tensor) -> Tensor:
        if not self.multi_scale:
            weights = F.softmax(self.cheb_w / self.cheb_temp, dim=0)
            return self._cheb_filter_single_scale(x, L, weights)

        outputs = []
        for scale_idx in range(self.num_scales):
            L_scaled = L
            for _ in range(scale_idx):
                L_scaled = (L_scaled @ L).clamp(-1.5, 1.5)
            weights = F.softmax(self.cheb_w[scale_idx] / self.cheb_temp, dim=0)
            outputs.append(self._cheb_filter_single_scale(x, L_scaled, weights))

        scale_w = F.softmax(self.scale_weights, dim=0)
        return sum(weight * output for weight, output in zip(scale_w, outputs))

    def _forward_impl(self, x: Tensor) -> tuple[Tensor, Tensor]:
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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.cfg.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        return self._forward_impl(x)


__all__ = [
    "AdaptiveEdgeSparsifier",
    "CorrelationConfig",
    "LatentCorrelationLearner",
]
