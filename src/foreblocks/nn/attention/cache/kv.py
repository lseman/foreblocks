"""foreblocks.nn.attention.cache.kv.

Unified KV provider abstraction for dense and paged key-value caches.

Abstracts KV retrieval and appending behind a single interface, supporting both
contiguous dense storage (via dict-backed accumulation) and block-based paged
storage (wrapping PagedKVCache). Use as the KV accessor in your attention
implementation — the provider handles gathering, MLA latent decoding, and
batch-indexed appends.

Core API:
- KVProvider: abstract base for KV retrieval/appending
- DenseKVProvider: dense dict-backed KV accumulation
- PagedKVProvider: paged block-table KV wrapping PagedKVCache

"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from foreblocks.nn.attention.cache.paged import PagedKVCache


class KVProvider(ABC):
    @abstractmethod
    def get_kv(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        kv_latent: torch.Tensor | None = None,
        batch_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def append(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_idx: int,
        kv_latent: torch.Tensor | None = None,
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
        layer_state: dict | None,
        cross_attention: bool,
        use_mla: bool = False,
        k_up_proj: nn.Module | None = None,
        v_up_proj: nn.Module | None = None,
    ):
        self.layer_state = layer_state if layer_state is not None else {}
        self.cross_attention = cross_attention
        self.use_mla = bool(use_mla)
        self.k_up_proj = k_up_proj
        self.v_up_proj = v_up_proj
        # Per-batch state (populated lazily on first batch_idx call)
        self._batch_states: list[dict] | None = None

    def _get_batch_state(self, b: int) -> dict:
        """Get or create per-batch state dict."""
        if self._batch_states is None:
            # Initialize from existing layer_state keys (shared mode)
            shared = dict(self.layer_state)
            self._batch_states = [shared]
        while len(self._batch_states) <= b:
            # Copy shared state for new batch members
            prev = self._batch_states[-1] if self._batch_states else {}
            self._batch_states.append({k: v for k, v in prev.items()})
        return self._batch_states[b]

    def get_kv(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        kv_latent: torch.Tensor | None = None,
        batch_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cross_attention:
            return new_k, new_v

        # Batched (per-token) mode
        if batch_idx is not None:
            state = self._get_batch_state(batch_idx)
            if self.use_mla:
                if kv_latent is None:
                    raise ValueError("MLA path expects kv_latent in DenseKVProvider.get_kv")
                if self.k_up_proj is None or self.v_up_proj is None:
                    raise RuntimeError(
                        "MLA projections are not configured in DenseKVProvider"
                    )
                prev_latent = state.get("kv_latent")
                if isinstance(prev_latent, torch.Tensor):
                    latent_full = torch.cat([prev_latent, kv_latent], dim=0)
                else:
                    latent_full = kv_latent
                state["kv_latent"] = latent_full
                T = latent_full.size(0)
                Hkv = new_k.size(0)
                D = new_k.size(1)
                k_full = (
                    self.k_up_proj(latent_full)
                    .view(T, Hkv, D)
                    .transpose(0, 1)
                )
                v_full = (
                    self.v_up_proj(latent_full)
                    .view(T, Hkv, D)
                    .transpose(0, 1)
                )
                return k_full, v_full

            prev_k = state.get("k")
            prev_v = state.get("v")
            if isinstance(prev_k, torch.Tensor):
                k_full = torch.cat([prev_k, new_k], dim=1)
                v_full = torch.cat([prev_v, new_v], dim=1)
            else:
                k_full, v_full = new_k, new_v
            state["k"] = k_full
            state["v"] = v_full
            return k_full, v_full

        # Shared (prefill) mode — concatenate across batch dim
        if self.use_mla:
            if kv_latent is None:
                raise ValueError("MLA path expects kv_latent in DenseKVProvider.get_kv")
            if self.k_up_proj is None or self.v_up_proj is None:
                raise RuntimeError(
                    "MLA projections are not configured in DenseKVProvider"
                )

            prev_latent = self.layer_state.get("kv_latent")
            if isinstance(prev_latent, torch.Tensor):
                latent_full = torch.cat(
                    [prev_latent, kv_latent], dim=1
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

        prev_k = self.layer_state.get("k")
        prev_v = self.layer_state.get("v")
        if isinstance(prev_k, torch.Tensor):
            k_full = torch.cat([prev_k, new_k], dim=2)
            v_full = torch.cat([prev_v, new_v], dim=2)
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
        kv_latent: torch.Tensor | None = None,
    ) -> None:
        """Append new KV tokens for a single batch member.

        Used when the caller wants to accumulate state incrementally
        (e.g. during token-by-token decode) rather than calling get_kv
        which triggers concatenation + up-projection on every call.
        """
        if self.use_mla:
            if kv_latent is None:
                raise ValueError("MLA append expects kv_latent.")
            state = self._get_batch_state(batch_idx)
            prev = state.get("kv_latent")
            if isinstance(prev, torch.Tensor):
                state["kv_latent"] = torch.cat([prev, kv_latent], dim=0)
            else:
                state["kv_latent"] = kv_latent
            return

        state = self._get_batch_state(batch_idx)
        prev_k = state.get("k")
        prev_v = state.get("v")
        if isinstance(prev_k, torch.Tensor):
            state["k"] = torch.cat([prev_k, k], dim=1)
            state["v"] = torch.cat([prev_v, v], dim=1)
        else:
            state["k"] = k
            state["v"] = v

    def get_current_length(self, batch_idx: int) -> int:
        if self.cross_attention:
            return 0
        if self._batch_states is not None and 0 <= batch_idx < len(self._batch_states):
            state = self._batch_states[batch_idx]
        else:
            state = self.layer_state
        if self.use_mla:
            latent = state.get("kv_latent")
            if isinstance(latent, torch.Tensor):
                return int(latent.size(1))  # shared/per-batch: [B, T, L]
            return 0
        k_prev = state.get("k")
        if not isinstance(k_prev, torch.Tensor):
            return 0
        if self._batch_states is not None and 0 <= batch_idx < len(self._batch_states):
            return int(k_prev.size(1))  # per-batch: [H, T]
        return int(k_prev.size(2))  # shared mode: [B, H, T]


class StaticKVCache:
    is_compileable = True

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        max_cache_len: int,
        head_dim: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        shape = (batch_size, num_heads, max_cache_len, head_dim)
        self.keys = torch.zeros(shape, device=device, dtype=dtype)
        self.values = torch.zeros(shape, device=device, dtype=dtype)
        self.lengths = torch.zeros(batch_size, device=device, dtype=torch.long)
        self.max_cache_len = int(max_cache_len)

    @property
    def length(self) -> torch.Tensor:
        return self.lengths.max()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: torch.Tensor | None = None,
        update_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if value_states.shape != key_states.shape:
            raise ValueError("key_states and value_states must have identical shapes")

        batch_size, num_heads, token_count, head_dim = key_states.shape
        if (batch_size, num_heads, head_dim) != (
            self.keys.size(0),
            self.keys.size(1),
            self.keys.size(3),
        ):
            raise ValueError("KV update shape does not match the static cache")

        if cache_position is None:
            positions = self.lengths[:, None] + torch.arange(
                token_count, device=self.keys.device, dtype=torch.long
            )
        else:
            positions = cache_position.to(device=self.keys.device, dtype=torch.long)
            if positions.ndim == 1:
                positions = positions.unsqueeze(0).expand(batch_size, -1)
            if positions.shape != (batch_size, token_count):
                raise ValueError(
                    "cache_position must have shape [T] or [B, T], got "
                    f"{tuple(cache_position.shape)}"
                )
        compiler = getattr(torch, "compiler", None)
        is_compiling = bool(
            compiler is not None
            and hasattr(compiler, "is_compiling")
            and compiler.is_compiling()
        )
        if not is_compiling:
            if positions.numel() and bool((positions < 0).any()):
                raise ValueError("cache_position cannot contain negative positions")
            if positions.numel() and bool((positions >= self.max_cache_len).any()):
                raise ValueError("cache_position exceeds max_cache_len")

        indices = positions[:, None, :, None].expand_as(key_states)
        if update_mask is not None:
            active = update_mask.to(device=self.keys.device, dtype=torch.bool)
            if active.shape != (batch_size,):
                raise ValueError("update_mask must have shape [B]")
            active_values = active[:, None, None, None]
            old_keys = self.keys.gather(2, indices)
            old_values = self.values.gather(2, indices)
            key_states = torch.where(active_values, key_states, old_keys)
            value_states = torch.where(active_values, value_states, old_values)
        else:
            active = torch.ones(batch_size, device=self.keys.device, dtype=torch.bool)
        self.keys.scatter_(2, indices, key_states)
        self.values.scatter_(2, indices, value_states)
        if token_count:
            next_lengths = torch.maximum(self.lengths, positions.max(dim=1).values + 1)
            self.lengths.copy_(torch.where(active, next_lengths, self.lengths))
        # Return fixed-shape tensors; the attention path masks unused capacity.
        return self.keys, self.values

    def get_seq_length(self, batch_idx: int | None = None) -> int:
        length = self.length if batch_idx is None else self.lengths[batch_idx]
        return int(length.item())

    def get_seq_lengths(self) -> torch.Tensor:
        return self.lengths.clone()

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = max(0, self.get_seq_length() + max_length)
        self.lengths.clamp_(max=min(max_length, self.max_cache_len))

    def reset(self) -> None:
        self.keys.zero_()
        self.values.zero_()
        self.lengths.zero_()

    def state_dict(self) -> dict:
        return {
            "keys": self.keys.detach().cpu(),
            "values": self.values.detach().cpu(),
            "lengths": self.lengths.detach().cpu(),
            "max_cache_len": self.max_cache_len,
        }

    @classmethod
    def from_state_dict(cls, state: dict, *, device=None) -> "StaticKVCache":
        keys = state["keys"]
        cache = cls(
            keys.size(0),
            keys.size(1),
            state["max_cache_len"],
            keys.size(3),
            device=device or keys.device,
            dtype=keys.dtype,
        )
        cache.keys.copy_(keys.to(cache.keys.device))
        cache.values.copy_(state["values"].to(cache.values.device))
        cache.lengths.copy_(state["lengths"].to(cache.lengths.device))
        return cache

    def to(self, device) -> "StaticKVCache":
        return type(self).from_state_dict(self.state_dict(), device=device)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> "StaticKVCache":
        """Reorder cache sequences according to beam indices."""
        selected = beam_idx.to(self.keys.device)
        cache = type(self)(
            selected.numel(),
            self.keys.size(1),
            self.max_cache_len,
            self.keys.size(3),
            device=self.keys.device,
            dtype=self.keys.dtype,
        )
        cache.keys.copy_(self.keys.index_select(0, selected))
        cache.values.copy_(self.values.index_select(0, selected))
        cache.lengths.copy_(self.lengths.index_select(0, selected))
        return cache

    def batch_select(self, indices: torch.LongTensor) -> "StaticKVCache":
        return self.reorder_cache(indices)


class StaticKVProvider(KVProvider):
    def __init__(
        self,
        cache: StaticKVCache,
        cache_position: torch.Tensor | None = None,
        update_mask: torch.Tensor | None = None,
    ):
        self.cache = cache
        self.cache_position = cache_position
        self.update_mask = update_mask

    def get_kv(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        kv_latent: torch.Tensor | None = None,
        batch_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del kv_latent
        if batch_idx is not None:
            raise ValueError("StaticKVProvider expects batched updates")
        return self.cache.update(new_k, new_v, self.cache_position, self.update_mask)

    def append(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_idx: int,
        kv_latent: torch.Tensor | None = None,
    ) -> None:
        raise RuntimeError("StaticKVProvider.append is not used")

    def get_current_length(self, batch_idx: int) -> int:
        return self.cache.get_seq_length(batch_idx)

    def get_start_positions(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.cache_position is not None:
            positions = self.cache_position.to(device=device, dtype=torch.long)
            if positions.ndim == 1:
                return positions[0].expand(batch_size)
            return positions[:, 0]
        return self.cache.lengths.to(device=device)


class PagedKVProvider(KVProvider):
    def __init__(
        self,
        cache: PagedKVCache,
        use_mla: bool = False,
        k_up_proj: nn.Module | None = None,
        v_up_proj: nn.Module | None = None,
        update_mask: torch.Tensor | None = None,
    ):
        self.cache = cache
        self.use_mla = bool(use_mla)
        self.k_up_proj = k_up_proj
        self.v_up_proj = v_up_proj
        self.update_mask = update_mask

    def get_kv(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        kv_latent: torch.Tensor | None = None,
        batch_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
                    if self._is_active(b):
                        self.append(new_k[b], new_v[b], b, kv_latent=kv_latent[b])

            # Fast path: stream latent blocks through up-projection without
            # materializing the full [B, T, L] latent tensor.
            k, v = self.cache.upproject_latent_stream(
                self.k_up_proj, self.v_up_proj
            )
            return k, v

        if batch_idx is not None:
            self.append(new_k, new_v, batch_idx)
        else:
            B = new_k.size(0)
            for b in range(B):
                if self._is_active(b):
                    self.append(new_k[b], new_v[b], b)
        return self.cache.gather_kv_batched()

    def append(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_idx: int,
        kv_latent: torch.Tensor | None = None,
    ) -> None:
        if not self._is_active(batch_idx):
            return
        if self.use_mla:
            if kv_latent is None:
                raise ValueError("MLA paged provider append expects kv_latent.")
            self.cache.append_step_latent(kv_latent, batch_idx)
            return
        self.cache.append_step(k, v, batch_idx)

    def _is_active(self, batch_idx: int) -> bool:
        if self.update_mask is None:
            return True
        return bool(self.update_mask[batch_idx].item())

    def get_current_length(self, batch_idx: int) -> int:
        return int(self.cache.logical_seq_len[batch_idx].item())

    def get_start_positions(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        return self.cache.logical_seq_len.to(device=device, dtype=torch.long).clone()
