from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

RouterOutput = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]

class Router(nn.Module, ABC):
    """Abstract router contract for MoE gating modules."""

    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = int(num_experts)
        # Exposed metrics (detached copies for hooks/logging)
        self.last_gate_logits: Optional[torch.Tensor] = None
        self.last_topk_idx: Optional[torch.Tensor] = None
        self.last_per_token_k: Optional[torch.Tensor] = None
        self.last_tokens_dropped: int = 0
        self.last_aux_loss: float = 0.0
        self.last_latency_ms: float = 0.0
        self.last_meta = None

    def _empty_out(
        self, x: torch.Tensor, return_raw_logits: bool = False
    ) -> RouterOutput | Tuple[torch.Tensor, torch.Tensor]:
        shp = (*x.shape[:-1], self.num_experts)
        z = x.new_zeros(shp)
        if return_raw_logits:
            return z, z, z, None, None, None, None, None
        return z, z

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        return_raw_logits: bool = False,
        tau: Optional[float] = None,
    ):
        raise NotImplementedError


class LinearRouter(Router):
    """Simple linear router baseline."""

    def __init__(self, d_model: int, num_experts: int, use_bias: bool = False):
        super().__init__(num_experts=num_experts)
        self.router = nn.Linear(d_model, num_experts, bias=use_bias)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        if self.router.bias is not None:
            nn.init.zeros_(self.router.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_raw_logits: bool = False,
        tau: Optional[float] = None,
    ):
        del tau
        if x.numel() == 0:
            return self._empty_out(x, return_raw_logits=return_raw_logits)
        logits = self.router(x)
        if return_raw_logits:
            return logits, logits, logits, None, None, None, None, None
        return logits, logits


class NoisyTopKRouter(Router):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        input_dropout: float = 0.0,
        jitter: float = 0.01,
        use_bias: bool = False,
        expert_bias_init: float = 0.0,
        clamp_range: Tuple[float, float] = (-1e4, 1e4),
    ):
        super().__init__(num_experts=num_experts)
        self.router = nn.Linear(d_model, num_experts, bias=use_bias)
        # Learnable per-expert logit bias (aux-loss-free load balancing control).
        self.expert_bias = nn.Parameter(
            torch.full((num_experts,), float(expert_bias_init))
        )
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else None
        self.jitter = float(jitter)
        self.clamp_min, self.clamp_max = clamp_range
        self._init_router()

    def _init_router(self):
        with torch.no_grad():
            std = 0.02
            bound = std * math.sqrt(3)
            self.router.weight.uniform_(-bound, bound)
            if self.router.bias is not None:
                self.router.bias.zero_()

    def _compute_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.input_dropout is not None:
            x = self.input_dropout(x)

        raw = self.router(x) + self.expert_bias
        if self.training and self.jitter > 0:
            logits = (raw + torch.randn_like(raw) * self.jitter).clamp_(
                self.clamp_min, self.clamp_max
            )
        else:
            logits = raw.clamp_(self.clamp_min, self.clamp_max)
        return raw, logits

    def forward(
        self,
        x: torch.Tensor,
        return_raw_logits: bool = False,
        tau: Optional[float] = None,
    ):
        del tau
        if x.numel() == 0:
            return self._empty_out(x, return_raw_logits=return_raw_logits)
        raw, logits = self._compute_logits(x)
        if return_raw_logits:
            return raw, logits, logits, None, None, None, None, None
        return logits, logits


class AdaptiveNoisyTopKRouter(NoisyTopKRouter):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        max_k: int = 4,
        k_head_dim: int = 32,
        k_tau: float = 1.0,
        input_dropout: float = 0.0,
        jitter: float = 0.01,
        use_bias: bool = False,
        expert_bias_init: float = 0.0,
        clamp_range: Tuple[float, float] = (-1e4, 1e4),
    ):
        super().__init__(
            d_model=d_model,
            num_experts=num_experts,
            input_dropout=input_dropout,
            jitter=jitter,
            use_bias=use_bias,
            expert_bias_init=expert_bias_init,
            clamp_range=clamp_range,
        )
        self.max_k = int(max_k)
        self.k_tau = float(k_tau)
        self.k_head = nn.Linear(d_model, k_head_dim, bias=True)
        self.k_logits = nn.Linear(k_head_dim, self.max_k, bias=True)
        nn.init.normal_(self.k_logits.weight, mean=0.0, std=0.02)

        self.last_k_logits: Optional[torch.Tensor] = None
        self.last_k: Optional[torch.Tensor] = None
        self.last_k_probs: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        return_raw_logits: bool = False,
        tau: Optional[float] = None,
    ):
        if x.numel() == 0:
            if return_raw_logits:
                shp = (*x.shape[:-1], self.num_experts)
                z = x.new_zeros(shp)
                k_shp = (*x.shape[:-1], self.max_k)
                kz = x.new_zeros(k_shp)
                k_idx = x.new_zeros((*x.shape[:-1],), dtype=torch.long)
                return z, z, z, kz, k_idx, kz, None, None
            return self._empty_out(x, return_raw_logits=False)

        raw, logits = self._compute_logits(x)

        k_feat = self.k_head(x)
        k_logits = self.k_logits(k_feat)
        use_tau = self.k_tau if tau is None else float(tau)
        if self.training:
            k_probs = F.softmax(k_logits / max(use_tau, 1e-3), dim=-1)
            k_sample = F.gumbel_softmax(k_logits, tau=max(use_tau, 1e-3), hard=True)
            per_token_k = (k_sample.argmax(-1) + 1).long()
        else:
            k_probs = F.softmax(k_logits, dim=-1)
            per_token_k = (k_logits.argmax(-1) + 1).long()

        self.last_k_logits = k_logits.detach()
        self.last_k = per_token_k.detach()
        self.last_k_probs = k_probs.detach()

        if return_raw_logits:
            return raw, logits, logits, k_logits, per_token_k, k_probs, None, None
        return logits, logits, per_token_k


class StraightThroughTopKRouter(NoisyTopKRouter):
    """Top-k sparse router with straight-through gradients."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        temperature: float = 1.0,
        input_dropout: float = 0.0,
        jitter: float = 0.01,
        use_bias: bool = False,
        expert_bias_init: float = 0.0,
        clamp_range: Tuple[float, float] = (-1e4, 1e4),
    ):
        super().__init__(
            d_model=d_model,
            num_experts=num_experts,
            input_dropout=input_dropout,
            jitter=jitter,
            use_bias=use_bias,
            expert_bias_init=expert_bias_init,
            clamp_range=clamp_range,
        )
        self.top_k = int(top_k)
        self.temperature = float(temperature)

    def forward(
        self,
        x: torch.Tensor,
        return_raw_logits: bool = False,
        tau: Optional[float] = None,
    ):
        if x.numel() == 0:
            return self._empty_out(x, return_raw_logits=return_raw_logits)

        raw, logits = self._compute_logits(x)
        temp = max(float(self.temperature if tau is None else tau), 1e-3)
        soft = F.softmax(logits / temp, dim=-1)
        k_eff = int(min(max(self.top_k, 1), logits.size(-1)))
        top_i = torch.topk(logits, k_eff, dim=-1, sorted=False).indices

        mask = torch.zeros_like(soft).scatter(1, top_i, 1.0)
        hard_sparse = mask * soft
        hard_sparse = hard_sparse / (hard_sparse.sum(dim=-1, keepdim=True) + 1e-12)
        st_sparse = hard_sparse + (soft - soft.detach())
        top_p = st_sparse.gather(1, top_i)

        if return_raw_logits:
            return raw, logits, st_sparse, None, None, None, top_p, top_i
        return logits, st_sparse


class ContinuousTopKRouter(NoisyTopKRouter):
    """Continuous/soft top-k approximation with optional perturb-and-pick."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        temperature: float = 1.0,
        perturb_std: float = 0.0,
        input_dropout: float = 0.0,
        jitter: float = 0.01,
        use_bias: bool = False,
        expert_bias_init: float = 0.0,
        clamp_range: Tuple[float, float] = (-1e4, 1e4),
    ):
        super().__init__(
            d_model=d_model,
            num_experts=num_experts,
            input_dropout=input_dropout,
            jitter=jitter,
            use_bias=use_bias,
            expert_bias_init=expert_bias_init,
            clamp_range=clamp_range,
        )
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.perturb_std = float(perturb_std)

    def forward(
        self,
        x: torch.Tensor,
        return_raw_logits: bool = False,
        tau: Optional[float] = None,
    ):
        if x.numel() == 0:
            return self._empty_out(x, return_raw_logits=return_raw_logits)

        raw, logits = self._compute_logits(x)
        temp = max(float(self.temperature if tau is None else tau), 1e-3)
        relaxed_logits = logits
        if self.training and self.perturb_std > 0:
            relaxed_logits = relaxed_logits + torch.randn_like(relaxed_logits) * self.perturb_std

        soft_probs = F.softmax(relaxed_logits / temp, dim=-1)
        k_eff = int(min(max(self.top_k, 1), soft_probs.size(-1)))
        top_p, top_i = torch.topk(soft_probs, k_eff, dim=-1, sorted=False)
        top_p = top_p / (top_p.sum(dim=-1, keepdim=True) + 1e-12)

        if return_raw_logits:
            return raw, logits, soft_probs, None, None, None, top_p, top_i
        return logits, soft_probs


class HashTopKRouter(NoisyTopKRouter):
    """Hash candidate routing with top-k selection inside hashed expert sets."""

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        num_hashes: int = 1,
        num_buckets: int = 64,
        bucket_size: int = 8,
        hash_seed: int = 17,
        input_dropout: float = 0.0,
        jitter: float = 0.01,
        use_bias: bool = False,
        expert_bias_init: float = 0.0,
        clamp_range: Tuple[float, float] = (-1e4, 1e4),
    ):
        super().__init__(
            d_model=d_model,
            num_experts=num_experts,
            input_dropout=input_dropout,
            jitter=jitter,
            use_bias=use_bias,
            expert_bias_init=expert_bias_init,
            clamp_range=clamp_range,
        )
        if num_hashes < 1:
            raise ValueError("num_hashes must be >= 1")
        if num_buckets < 2:
            raise ValueError("num_buckets must be >= 2")
        if bucket_size < 1:
            raise ValueError("bucket_size must be >= 1")

        self.top_k = int(top_k)
        self.num_hashes = int(num_hashes)
        self.num_buckets = int(num_buckets)
        self.bucket_size = int(bucket_size)
        self.hash_proj = nn.Linear(d_model, self.num_hashes, bias=False)
        nn.init.normal_(
            self.hash_proj.weight, mean=0.0, std=1.0 / math.sqrt(max(d_model, 1))
        )
        self.register_buffer(
            "hash_tables",
            self._build_hash_tables(seed=int(hash_seed)),
        )

    def _build_hash_tables(self, seed: int) -> torch.Tensor:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        tables = []
        for _ in range(self.num_hashes):
            rows = []
            for _ in range(self.num_buckets):
                if self.bucket_size <= self.num_experts:
                    idx = torch.randperm(self.num_experts, generator=g)[
                        : self.bucket_size
                    ]
                else:
                    idx = torch.randint(
                        low=0,
                        high=self.num_experts,
                        size=(self.bucket_size,),
                        generator=g,
                    )
                rows.append(idx)
            tables.append(torch.stack(rows, dim=0))
        return torch.stack(tables, dim=0)

    def _hash_candidates(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, D] -> candidates: [T, num_hashes * bucket_size]
        hash_logits = self.hash_proj(x)
        bucket_idx = torch.floor(torch.sigmoid(hash_logits) * self.num_buckets).long()
        bucket_idx = bucket_idx.clamp_(0, self.num_buckets - 1)

        chunks = []
        for h in range(self.num_hashes):
            chunks.append(self.hash_tables[h].index_select(0, bucket_idx[:, h]))
        return torch.cat(chunks, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        return_raw_logits: bool = False,
        tau: Optional[float] = None,
    ):
        del tau
        if x.numel() == 0:
            return self._empty_out(x, return_raw_logits=return_raw_logits)

        raw, logits = self._compute_logits(x)
        candidates = self._hash_candidates(x)
        cand_scores = logits.gather(1, candidates)

        k_eff = int(min(max(self.top_k, 1), cand_scores.size(-1)))
        top_v, top_pos = torch.topk(cand_scores, k_eff, dim=-1, sorted=False)
        top_i = candidates.gather(1, top_pos)
        m = top_v.max(dim=-1, keepdim=True).values
        expv = torch.exp(top_v - m)
        top_p = expv / (expv.sum(dim=-1, keepdim=True) + 1e-12)

        if return_raw_logits:
            return raw, logits, logits, None, None, None, top_p, top_i
        return logits, logits


__all__ = [
    "Router",
    "RouterOutput",
    "LinearRouter",
    "NoisyTopKRouter",
    "AdaptiveNoisyTopKRouter",
    "StraightThroughTopKRouter",
    "ContinuousTopKRouter",
    "HashTopKRouter",
]
