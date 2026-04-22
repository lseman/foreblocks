from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ProjectionRegistry(nn.Module):
    """
    Unified factory/registry for projection and fusion helper modules.

    Keys are structured by `(category, name, in_dim, out_dim, proj_type, kwargs...)`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.registry = nn.ModuleDict()

    @staticmethod
    def _safe_token(val: Any) -> str:
        return str(val).replace("|", "_").replace(":", "_").replace(" ", "_")

    def _make_key(
        self,
        *,
        category: str,
        name: str,
        in_dim: int,
        out_dim: int,
        proj_type: str,
        kwargs: dict[str, Any],
    ) -> str:
        parts = [
            self._safe_token(category),
            self._safe_token(name),
            str(int(in_dim)),
            str(int(out_dim)),
            self._safe_token(proj_type),
        ]
        if kwargs:
            extras = "|".join(
                f"{self._safe_token(k)}={self._safe_token(v)}"
                for k, v in sorted(kwargs.items(), key=lambda kv: kv[0])
            )
            parts.append(extras)
        return "::".join(parts)

    def register_custom(self, *, category: str, name: str, module: nn.Module) -> None:
        key = self._make_key(
            category=category,
            name=name,
            in_dim=-1,
            out_dim=-1,
            proj_type="registered",
            kwargs={},
        )
        self.registry[key] = module

    def get_registered(self, *, category: str, name: str) -> nn.Module | None:
        key = self._make_key(
            category=category,
            name=name,
            in_dim=-1,
            out_dim=-1,
            proj_type="registered",
            kwargs={},
        )
        if key in self.registry:
            return self.registry[key]
        return None

    def get_or_create(
        self,
        *,
        category: str,
        name: str,
        in_dim: int,
        out_dim: int,
        proj_type: str = "linear",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> nn.Module:
        key = self._make_key(
            category=category,
            name=name,
            in_dim=in_dim,
            out_dim=out_dim,
            proj_type=proj_type,
            kwargs=kwargs,
        )
        if key not in self.registry:
            self.registry[key] = self._build_module(
                in_dim=in_dim,
                out_dim=out_dim,
                proj_type=proj_type,
                **kwargs,
            )
        mod = self.registry[key]
        if device is not None or dtype is not None:
            mod = mod.to(device=device, dtype=dtype)
        return mod

    def _build_module(
        self,
        *,
        in_dim: int,
        out_dim: int,
        proj_type: str,
        **kwargs: Any,
    ) -> nn.Module:
        if proj_type == "identity":
            return nn.Identity()

        if proj_type == "linear":
            allow_identity = bool(kwargs.get("allow_identity", True))
            if allow_identity and int(in_dim) == int(out_dim):
                return nn.Identity()
            proj = nn.Linear(int(in_dim), int(out_dim))
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
            return proj

        if proj_type == "conv1d":
            kernel_size = int(kwargs.get("kernel_size", 1))
            padding = int(kwargs.get("padding", kernel_size // 2))
            bias = bool(kwargs.get("bias", True))
            conv = nn.Conv1d(
                in_channels=int(in_dim),
                out_channels=int(out_dim),
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            )
            nn.init.kaiming_uniform_(conv.weight, a=5**0.5)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
            return conv

        if proj_type == "hyper_mlp":
            hidden_dim = int(kwargs.get("hidden_dim", 64))
            out_features = int(kwargs.get("out_features", out_dim))
            dropout = float(kwargs.get("dropout", 0.0))
            drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            net = nn.Sequential(
                nn.Linear(int(in_dim), hidden_dim),
                nn.GELU(),
                drop,
                nn.Linear(hidden_dim, out_features),
            )
            if isinstance(net[-1], nn.Linear) and net[-1].bias is not None:
                nn.init.zeros_(net[-1].bias)
            return net

        if proj_type == "gate_mlp":
            hidden_dim = int(kwargs.get("hidden_dim", max(8, int(in_dim) // 2)))
            out_features = int(kwargs.get("out_features", 1))
            net = nn.Sequential(
                nn.Linear(int(in_dim), hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_features),
            )
            if isinstance(net[-1], nn.Linear) and net[-1].bias is not None:
                nn.init.zeros_(net[-1].bias)
            return net

        if proj_type == "lora_adapter":
            rank = int(kwargs.get("rank", max(1, int(in_dim) // 4)))
            net = nn.Sequential(
                nn.Linear(int(in_dim), rank, bias=False),
                nn.GELU(),
                nn.Linear(rank, int(out_dim), bias=False),
            )
            return net

        if proj_type == "multihead_attention":
            num_heads = int(kwargs["num_heads"])
            dropout = float(kwargs.get("dropout", 0.0))
            batch_first = bool(kwargs.get("batch_first", True))
            return nn.MultiheadAttention(
                embed_dim=int(in_dim),
                num_heads=num_heads,
                dropout=dropout,
                batch_first=batch_first,
            )

        raise ValueError(f"Unsupported proj_type: {proj_type}")
