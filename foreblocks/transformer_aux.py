import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        num_experts=8,
        top_k=2,
        dropout=0.1,
        capacity_factor=1.25,
        expert_dropout=0.0,
        use_noisy_gating=False,
        min_capacity=4,
        use_swiglu=True,
        activation="gelu",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.capacity_factor = capacity_factor
        self.min_capacity = min_capacity
        self.use_noisy_gating = use_noisy_gating
        self.expert_dropout = expert_dropout
        self.noise_eps = 1e-2

        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.input_norm = nn.LayerNorm(d_model)

        self.experts = nn.ModuleList(
            [SwiGLUExpert(d_model, d_ff, dropout) for _ in range(num_experts)]
        )

        self.aux_loss = 0.0

    def forward(self, x, return_aux_loss=True):
        B, T, D = x.shape
        x_flat = self.input_norm(x.view(-1, D))  # [B*T, D]
        num_tokens = x_flat.size(0)

        with torch.amp.autocast("cuda", enabled=False):
            router_logits = self.router(x_flat.float())  # [B*T, num_experts]

        if self.training and self.expert_dropout > 0:
            mask = (
                (torch.rand(self.num_experts, device=x.device) > self.expert_dropout)
                .unsqueeze(0)
                .expand_as(router_logits)
            )
            router_logits = router_logits.masked_fill(~mask, float("-inf"))

        if self.use_noisy_gating and self.training:
            noise = torch.randn_like(router_logits) * self.noise_eps
            router_logits = router_logits + noise

        topk_scores, topk_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # [N, top_k]
        gate = F.softmax(topk_scores, dim=-1)  # [N, top_k]

        expanded_x = (
            x_flat.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, D)
        )  # [N*top_k, D]
        expert_indices = topk_indices.reshape(-1)  # [N*top_k]
        gates = gate.reshape(-1)  # [N*top_k]

        # Dispatch: create one list per expert
        expert_inputs = [
            expanded_x[expert_indices == i] for i in range(self.num_experts)
        ]

        # Process each expert in parallel using torch.cat/stack
        expert_outputs = [
            (
                self.experts[i](expert_inputs[i])
                if expert_inputs[i].numel() > 0
                else torch.zeros(0, D, device=x.device, dtype=expanded_x.dtype)
            )
            for i in range(self.num_experts)
        ]
        output_buffer = torch.empty_like(expanded_x)
        for i in range(self.num_experts):
            mask = expert_indices == i
            if mask.any():
                output_buffer[mask] = expert_outputs[i].to(output_buffer.dtype)

        weighted_output = output_buffer * gates.unsqueeze(-1)
        token_indices = (
            torch.arange(num_tokens, device=x.device)
            .unsqueeze(1)
            .repeat(1, self.top_k)
            .reshape(-1)
        )
        output = torch.zeros_like(x_flat)
        output.index_add_(0, token_indices, weighted_output)

        out = output.view(B, T, D)

        if self.training and return_aux_loss:
            probs = F.softmax(router_logits, dim=-1)
            expert_mean = torch.clamp(probs.mean(0), min=1e-6)
            self.aux_loss = -(expert_mean * expert_mean.log()).sum()
        else:
            self.aux_loss = 0.0

        return out, (self.aux_loss if return_aux_loss else out)


class SwiGLUExpert(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.norm = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w_out = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def safe_silu(self, x):
        x = torch.clamp(x, -20.0, 20.0)
        return x * torch.sigmoid(x)

    def forward(self, x):
        x = x.float()
        x = self.norm(x)
        x1 = self.w1(x)
        x2 = self.w2(x)
        gate = self.safe_silu(x2)
        hidden = x1 * gate
        hidden = torch.clamp(hidden, -1e4, 1e4)
        out = self.w_out(hidden)
        out = torch.clamp(out, -1e4, 1e4)
        return self.dropout(out)


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
