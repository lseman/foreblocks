from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .paged import PagedKVCache


class KVProvider(ABC):
    @abstractmethod
    def get_kv(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        kv_latent: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return full/current K and V (possibly gathered)."""

    @abstractmethod
    def append(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_idx: int,
        kv_latent: Optional[torch.Tensor] = None,
    ) -> None: ...

    @abstractmethod
    def get_current_length(self, batch_idx: int) -> int: ...

    def get_start_positions(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.tensor(
            [self.get_current_length(b) for b in range(batch_size)],
            device=device,
            dtype=torch.long,
        )


class DenseKVProvider(KVProvider):
    def __init__(
        self,
        layer_state: Optional[Dict],
        cross_attention: bool,
        use_mla: bool = False,
        k_up_proj: Optional[nn.Module] = None,
        v_up_proj: Optional[nn.Module] = None,
    ):
        self.layer_state = layer_state
        self.cross_attention = cross_attention
        self.use_mla = bool(use_mla)
        self.k_up_proj = k_up_proj
        self.v_up_proj = v_up_proj

    def get_kv(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        kv_latent: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cross_attention:
            return new_k, new_v

        if self.layer_state is None:
            self.layer_state = {}

        if self.use_mla:
            if kv_latent is None:
                raise ValueError("MLA path expects kv_latent in DenseKVProvider.get_kv")
            if self.k_up_proj is None or self.v_up_proj is None:
                raise RuntimeError(
                    "MLA projections are not configured in DenseKVProvider"
                )

            if "kv_latent" in self.layer_state:
                latent_full = torch.cat(
                    [self.layer_state["kv_latent"], kv_latent], dim=1
                )
            else:
                latent_full = kv_latent
            self.layer_state["kv_latent"] = latent_full

            B, T, _ = latent_full.shape
            k_full = (
                self.k_up_proj(latent_full)
                .view(B, T, new_k.size(1), new_k.size(3))
                .transpose(1, 2)
            )
            v_full = (
                self.v_up_proj(latent_full)
                .view(B, T, new_v.size(1), new_v.size(3))
                .transpose(1, 2)
            )
            return k_full, v_full

        if "k" in self.layer_state:
            k_full = torch.cat([self.layer_state["k"], new_k], dim=2)
            v_full = torch.cat([self.layer_state["v"], new_v], dim=2)
        else:
            k_full, v_full = new_k, new_v

        self.layer_state["k"] = k_full
        self.layer_state["v"] = v_full
        return k_full, v_full

    def append(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_idx: int,
        kv_latent: Optional[torch.Tensor] = None,
    ) -> None:
        raise RuntimeError("DenseKVProvider.append is not used in batched mode.")

    def get_current_length(self, batch_idx: int) -> int:
        if self.cross_attention or self.layer_state is None:
            return 0
        if self.use_mla:
            latent_prev = self.layer_state.get("kv_latent")
            if isinstance(latent_prev, torch.Tensor):
                return int(latent_prev.size(1))
            return 0
        k_prev = self.layer_state.get("k")
        if not isinstance(k_prev, torch.Tensor):
            return 0
        return int(k_prev.size(2))


class PagedKVProvider(KVProvider):
    def __init__(
        self,
        cache: PagedKVCache,
        use_mla: bool = False,
        k_up_proj: Optional[nn.Module] = None,
        v_up_proj: Optional[nn.Module] = None,
    ):
        self.cache = cache
        self.use_mla = bool(use_mla)
        self.k_up_proj = k_up_proj
        self.v_up_proj = v_up_proj

    def get_kv(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        kv_latent: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_mla:
            if kv_latent is None:
                raise ValueError("MLA paged provider expects kv_latent.")
            if self.k_up_proj is None or self.v_up_proj is None:
                raise RuntimeError(
                    "MLA projections are not configured in PagedKVProvider"
                )

            if batch_idx is not None:
                self.append(new_k, new_v, batch_idx, kv_latent=kv_latent)
            else:
                B = kv_latent.size(0)
                for b in range(B):
                    self.append(new_k[b], new_v[b], b, kv_latent=kv_latent[b])

            latent = self.cache.gather_latent_batched()  # [B, T, L]
            B, T, _ = latent.shape
            Hkv = new_k.size(1)
            D = new_k.size(3)
            k = self.k_up_proj(latent).view(B, T, Hkv, D).transpose(1, 2)
            v = self.v_up_proj(latent).view(B, T, Hkv, D).transpose(1, 2)
            return k, v

        if batch_idx is not None:
            self.append(new_k, new_v, batch_idx)
        else:
            B = new_k.size(0)
            for b in range(B):
                self.append(new_k[b], new_v[b], b)
        return self.cache.gather_kv_batched()

    def append(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_idx: int,
        kv_latent: Optional[torch.Tensor] = None,
    ) -> None:
        if self.use_mla:
            if kv_latent is None:
                raise ValueError("MLA paged provider append expects kv_latent.")
            self.cache.append_step_latent(kv_latent, batch_idx)
            return
        self.cache.append_step(k, v, batch_idx)

    def get_current_length(self, batch_idx: int) -> int:
        return int(self.cache.seq_len[batch_idx].item())

    def get_start_positions(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        return self.cache.seq_len.to(device=device, dtype=torch.long).clone()
