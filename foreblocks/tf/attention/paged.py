from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


class PagedKVCache:
    """
    Fixed-page KV cache for autoregressive self-attention (torch-only).
    Layout
    ------
    storage_k/v: [B, Hkv, max_blocks, block_size, D] (standard mode)
    storage_latent: [B, max_blocks, block_size, L] (MLA latent mode)
    Per sequence (batch index `b`) we keep:
      - block_table[b]: list[int]
            Global block ids (indices into dim=2 of storage).
      - write_pos[b]: tuple[int, int]
            (idx_in_table, offset_in_block).
      - seq_len[b]: total cached tokens for that sequence.
    This class is intentionally dumb and local:
      - No cross-layer sharing; you typically call `PagedKVCache.ensure()`
        per layer_state dict.
      - No cross-batch resizing; `B` is fixed at construction.
    """

    def __init__(
        self,
        batch_size: int,
        n_kv_heads: int,
        head_dim: int,
        latent_dim: Optional[int] = None,
        block_size: int = 128,
        max_blocks: int = 1024,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if n_kv_heads <= 0:
            raise ValueError(f"n_kv_heads must be > 0, got {n_kv_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        if max_blocks <= 0:
            raise ValueError(f"max_blocks must be > 0, got {max_blocks}")
        self.B = int(batch_size)
        self.Hkv = int(n_kv_heads)
        self.D = int(head_dim)
        self.block_size = int(block_size)
        self.max_blocks = int(max_blocks)
        self.device = device
        self.dtype = dtype or torch.get_default_dtype()
        self.latent_dim = int(latent_dim) if latent_dim is not None else None
        self.use_latent_cache = self.latent_dim is not None

        if self.use_latent_cache:
            self.storage_k = None
            self.storage_v = None
            self.storage_latent = torch.empty(
                (self.B, self.max_blocks, self.block_size, self.latent_dim),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            shape = (self.B, self.Hkv, self.max_blocks, self.block_size, self.D)
            self.storage_k = torch.empty(shape, device=self.device, dtype=self.dtype)
            self.storage_v = torch.empty(shape, device=self.device, dtype=self.dtype)
            self.storage_latent = None
        # Per-sequence metadata
        self.block_table: List[List[int]] = [[] for _ in range(self.B)]
        self.write_pos: List[Tuple[int, int]] = [
            (0, 0) for _ in range(self.B)
        ]  # (idx_in_table, offset_in_block)
        self.seq_len: Tensor = torch.zeros(self.B, device=self.device, dtype=torch.long)
        # Free block pool
        self._free_blocks: List[int] = list(range(self.max_blocks))

    # ---------------------------------------------------------------------
    # Construction / ensure
    # ---------------------------------------------------------------------

    @staticmethod
    def ensure(
        layer_state: Dict,
        batch_size: int,
        n_kv_heads: int,
        head_dim: int,
        latent_dim: Optional[int],
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
        max_blocks: int = 1024,
    ) -> "PagedKVCache":
        """
        Get or create a PagedKVCache stored inside a per-layer `layer_state` dict.
        This is the recommended entrypoint from your Transformer layer:
            cache = PagedKVCache.ensure(layer_state, B, Hkv, D, block_size, device, dtype)
        """
        cache = layer_state.get("paged_cache", None)
        if cache is None:
            cache = PagedKVCache(
                batch_size=batch_size,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                latent_dim=latent_dim,
                block_size=block_size,
                max_blocks=max_blocks,
                device=device,
                dtype=dtype,
            )
            layer_state["paged_cache"] = cache
        else:
            # Optional sanity check: if shapes change, hard-reset to avoid UB.
            if (
                cache.B != batch_size
                or cache.Hkv != n_kv_heads
                or cache.D != head_dim
                or cache.latent_dim
                != (int(latent_dim) if latent_dim is not None else None)
                or cache.block_size != block_size
            ):
                # You can choose to raise instead if you don't expect this case.
                cache = PagedKVCache(
                    batch_size=batch_size,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    latent_dim=latent_dim,
                    block_size=block_size,
                    max_blocks=max_blocks,
                    device=device,
                    dtype=dtype,
                )
                layer_state["paged_cache"] = cache
        return cache

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _check_batch_index(self, b: int) -> None:
        if not (0 <= b < self.B):
            raise IndexError(f"batch index out of range: {b} (B={self.B})")

    def _alloc_block_for_seq(self, b: int) -> None:
        self._check_batch_index(b)
        if not self._free_blocks:
            raise RuntimeError("PagedKVCache: out of blocks (increase max_blocks).")
        blk = self._free_blocks.pop()
        self.block_table[b].append(blk)
        self.write_pos[b] = (len(self.block_table[b]) - 1, 0)

    # ---------------------------------------------------------------------
    # Public API: writing
    # ---------------------------------------------------------------------

    def append(self, k_bhdt: Tensor, v_bhdt: Tensor) -> None:
        """
        Append new tokens for all sequences in the batch.
        Parameters
        ----------
        k_bhdt, v_bhdt : Tensor
            Shape [B, Hkv, T_new, D]. Must match the cache's (B, Hkv, D).
        Notes
        -----
        - Assumes all sequences append the same T_new tokens.
        - For uneven lengths, pad or call per-sequence.
        - This is batched for efficiency.
        """
        if self.use_latent_cache:
            raise RuntimeError(
                "PagedKVCache is in latent mode; use append_latent/append_step_latent."
            )
        if k_bhdt.ndim != 4:
            raise ValueError(
                f"k_bhdt must be [B, Hkv, T_new, D], got shape {tuple(k_bhdt.shape)}"
            )
        if v_bhdt.ndim != 4:
            raise ValueError(
                f"v_bhdt must be [B, Hkv, T_new, D], got shape {tuple(v_bhdt.shape)}"
            )
        if k_bhdt.shape != v_bhdt.shape:
            raise ValueError(
                f"k and v must have same shape, got {k_bhdt.shape} vs {v_bhdt.shape}"
            )
        if k_bhdt.shape[0] != self.B:
            raise ValueError(f"Expected batch_size={self.B}, got {k_bhdt.shape[0]}")
        if k_bhdt.shape[1] != self.Hkv:
            raise ValueError(f"Expected n_kv_heads={self.Hkv}, got {k_bhdt.shape[1]}")
        if k_bhdt.shape[3] != self.D:
            raise ValueError(f"Expected head_dim={self.D}, got {k_bhdt.shape[3]}")
        T_new = k_bhdt.size(2)
        if T_new == 0:
            return

        for b in range(self.B):
            # Lazily allocate first block for this sequence
            if len(self.block_table[b]) == 0:
                self._alloc_block_for_seq(b)
            remaining, src_start = T_new, 0
            while remaining > 0:
                blk_idx_in_table, offset = self.write_pos[b]
                blk_global = self.block_table[b][blk_idx_in_table]
                space = self.block_size - offset
                if space <= 0:
                    # No room in current block; allocate a new one.
                    self._alloc_block_for_seq(b)
                    blk_idx_in_table, offset = self.write_pos[b]
                    blk_global = self.block_table[b][blk_idx_in_table]
                    space = self.block_size - offset
                take = min(space, remaining)
                dst_slice = slice(offset, offset + take)
                src_slice = slice(src_start, src_start + take)
                # storage_*: [B, Hkv, max_blocks, block_size, D]
                self.storage_k[b, :, blk_global, dst_slice, :] = k_bhdt[
                    b, :, src_slice, :
                ]
                self.storage_v[b, :, blk_global, dst_slice, :] = v_bhdt[
                    b, :, src_slice, :
                ]
                offset += take
                src_start += take
                remaining -= take
                self.seq_len[b] += take
                self.write_pos[b] = (blk_idx_in_table, offset)

    def append_step(self, k_htd: Tensor, v_htd: Tensor, b: int) -> None:
        """Append one chunk for a single sequence b. k/v shapes: [Hkv,T,D] or [Hkv,D]."""
        if self.use_latent_cache:
            raise RuntimeError(
                "PagedKVCache is in latent mode; use append_step_latent instead."
            )
        self._check_batch_index(b)
        if k_htd.ndim == 2:
            k_htd = k_htd.unsqueeze(1)
        if v_htd.ndim == 2:
            v_htd = v_htd.unsqueeze(1)
        if k_htd.ndim != 3 or v_htd.ndim != 3:
            raise ValueError("k/v must be [Hkv,T,D] or [Hkv,D]")
        if k_htd.shape != v_htd.shape:
            raise ValueError(
                f"k and v must have same shape, got {k_htd.shape} vs {v_htd.shape}"
            )
        if k_htd.shape[0] != self.Hkv or k_htd.shape[2] != self.D:
            raise ValueError(
                f"Expected [Hkv,T,D]=[{self.Hkv},T,{self.D}], got {tuple(k_htd.shape)}"
            )

        if len(self.block_table[b]) == 0:
            self._alloc_block_for_seq(b)
        remaining, src_start = k_htd.size(1), 0
        while remaining > 0:
            blk_idx_in_table, offset = self.write_pos[b]
            blk_global = self.block_table[b][blk_idx_in_table]
            space = self.block_size - offset
            if space <= 0:
                self._alloc_block_for_seq(b)
                blk_idx_in_table, offset = self.write_pos[b]
                blk_global = self.block_table[b][blk_idx_in_table]
                space = self.block_size - offset

            take = min(space, remaining)
            dst_slice = slice(offset, offset + take)
            src_slice = slice(src_start, src_start + take)
            self.storage_k[b, :, blk_global, dst_slice, :] = k_htd[:, src_slice, :]
            self.storage_v[b, :, blk_global, dst_slice, :] = v_htd[:, src_slice, :]

            offset += take
            src_start += take
            remaining -= take
            self.seq_len[b] += take
            self.write_pos[b] = (blk_idx_in_table, offset)

    def append_latent(self, latent_btl: Tensor) -> None:
        """Append latent chunks for all sequences. Shape: [B,T,L]."""
        if not self.use_latent_cache:
            raise RuntimeError("PagedKVCache is not in latent mode.")
        if latent_btl.ndim != 3:
            raise ValueError(
                f"latent_btl must be [B,T,L], got {tuple(latent_btl.shape)}"
            )
        if latent_btl.shape[0] != self.B:
            raise ValueError(f"Expected batch_size={self.B}, got {latent_btl.shape[0]}")
        if latent_btl.shape[2] != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.latent_dim}, got {latent_btl.shape[2]}"
            )
        T_new = latent_btl.size(1)
        if T_new == 0:
            return
        for b in range(self.B):
            self.append_step_latent(latent_btl[b], b)

    def append_step_latent(self, latent_tl: Tensor, b: int) -> None:
        """Append one latent chunk for a single sequence b. Shapes: [T,L] or [L]."""
        if not self.use_latent_cache:
            raise RuntimeError("PagedKVCache is not in latent mode.")
        self._check_batch_index(b)
        if latent_tl.ndim == 1:
            latent_tl = latent_tl.unsqueeze(0)
        if latent_tl.ndim != 2:
            raise ValueError("latent must be [T,L] or [L]")
        if latent_tl.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim={self.latent_dim}, got {latent_tl.shape[1]}"
            )

        if len(self.block_table[b]) == 0:
            self._alloc_block_for_seq(b)
        remaining, src_start = latent_tl.size(0), 0
        while remaining > 0:
            blk_idx_in_table, offset = self.write_pos[b]
            blk_global = self.block_table[b][blk_idx_in_table]
            space = self.block_size - offset
            if space <= 0:
                self._alloc_block_for_seq(b)
                blk_idx_in_table, offset = self.write_pos[b]
                blk_global = self.block_table[b][blk_idx_in_table]
                space = self.block_size - offset

            take = min(space, remaining)
            dst_slice = slice(offset, offset + take)
            src_slice = slice(src_start, src_start + take)
            self.storage_latent[b, blk_global, dst_slice, :] = latent_tl[src_slice, :]

            offset += take
            src_start += take
            remaining -= take
            self.seq_len[b] += take
            self.write_pos[b] = (blk_idx_in_table, offset)

    # ---------------------------------------------------------------------
    # Public API: reading (contiguous gather)
    # ---------------------------------------------------------------------
    # These are primarily for debugging / non-paged attention paths.
    # Your fast decode path can operate directly on storage + block_table.

    def gather_kv_batched(self) -> Tuple[Tensor, Tensor]:
        """
        Gather all cached tokens into contiguous [B, Hkv, T_max, D] tensors.
        This is a convenience / debug method. For fast decode, you likely want
        to operate directly on `storage_k/v` using `block_table` and `seq_len`.
        """
        if self.use_latent_cache:
            raise RuntimeError("KV tensors are not stored in latent mode.")
        max_len = int(self.seq_len.max().item())
        if max_len == 0:
            empty_k = self.storage_k.new_zeros((self.B, self.Hkv, 0, self.D))
            empty_v = self.storage_v.new_zeros((self.B, self.Hkv, 0, self.D))
            return empty_k, empty_v
        k_out = torch.zeros(
            self.B,
            self.Hkv,
            max_len,
            self.D,
            device=self.storage_k.device,
            dtype=self.storage_k.dtype,
        )
        v_out = torch.zeros_like(k_out)
        for b in range(self.B):
            seq_len_b = int(self.seq_len[b].item())
            if seq_len_b == 0 or not self.block_table[b]:
                continue
            k_parts = []
            v_parts = []
            blocks = self.block_table[b]
            last_blk_idx = len(blocks) - 1
            for blk_idx, global_blk in enumerate(blocks):
                if blk_idx == last_blk_idx:
                    # Last block: only up to current write offset
                    offset = (
                        self.write_pos[b][1]
                        if blk_idx == self.write_pos[b][0]
                        else self.block_size
                    )
                    k_parts.append(self.storage_k[b, :, global_blk, :offset, :])
                    v_parts.append(self.storage_v[b, :, global_blk, :offset, :])
                else:
                    # Full block
                    k_parts.append(self.storage_k[b, :, global_blk, :, :])
                    v_parts.append(self.storage_v[b, :, global_blk, :, :])
            k_flat = torch.cat(k_parts, dim=1)  # [Hkv, seq_len_b, D]
            v_flat = torch.cat(v_parts, dim=1)
            k_out[b, :, :seq_len_b, :] = k_flat
            v_out[b, :, :seq_len_b, :] = v_flat
        return k_out, v_out

    def gather_kv_for_seq(self, b: int) -> Tuple[Tensor, Tensor]:
        """
        Gather KV for a single sequence b into contiguous [Hkv, T, D] tensors.
        Useful when you only decode one sequence at a time.
        """
        if self.use_latent_cache:
            raise RuntimeError("KV tensors are not stored in latent mode.")
        self._check_batch_index(b)
        seq_len = int(self.seq_len[b].item())
        if seq_len == 0 or not self.block_table[b]:
            empty_k = self.storage_k.new_zeros((self.Hkv, 0, self.D))
            empty_v = self.storage_v.new_zeros((self.Hkv, 0, self.D))
            return empty_k, empty_v
        k_parts = []
        v_parts = []
        blocks = self.block_table[b]
        last_blk_idx = len(blocks) - 1
        for blk_idx, global_blk in enumerate(blocks):
            if blk_idx == last_blk_idx:
                # Last block: only up to current write offset
                offset = (
                    self.write_pos[b][1]
                    if blk_idx == self.write_pos[b][0]
                    else self.block_size
                )
                k_parts.append(self.storage_k[b, :, global_blk, :offset, :])
                v_parts.append(self.storage_v[b, :, global_blk, :offset, :])
            else:
                # Full block
                k_parts.append(self.storage_k[b, :, global_blk, :, :])
                v_parts.append(self.storage_v[b, :, global_blk, :, :])
        k_flat = torch.cat(k_parts, dim=1)  # [Hkv, seq_len, D]
        v_flat = torch.cat(v_parts, dim=1)
        return k_flat, v_flat

    def gather_latent_batched(self) -> Tensor:
        """Gather latent cache into contiguous [B, T_max, L] tensor."""
        if not self.use_latent_cache:
            raise RuntimeError("Latent tensors are not stored in standard KV mode.")
        max_len = int(self.seq_len.max().item())
        if max_len == 0:
            return self.storage_latent.new_zeros((self.B, 0, self.latent_dim))

        out = torch.zeros(
            self.B,
            max_len,
            self.latent_dim,
            device=self.storage_latent.device,
            dtype=self.storage_latent.dtype,
        )
        for b in range(self.B):
            seq_len_b = int(self.seq_len[b].item())
            if seq_len_b == 0 or not self.block_table[b]:
                continue
            parts = []
            blocks = self.block_table[b]
            last_blk_idx = len(blocks) - 1
            for blk_idx, global_blk in enumerate(blocks):
                if blk_idx == last_blk_idx:
                    offset = (
                        self.write_pos[b][1]
                        if blk_idx == self.write_pos[b][0]
                        else self.block_size
                    )
                    parts.append(self.storage_latent[b, global_blk, :offset, :])
                else:
                    parts.append(self.storage_latent[b, global_blk, :, :])
            flat = torch.cat(parts, dim=0)
            out[b, :seq_len_b, :] = flat
        return out

    def gather_latent_for_seq(self, b: int) -> Tensor:
        """Gather latent cache for one sequence into contiguous [T, L]."""
        if not self.use_latent_cache:
            raise RuntimeError("Latent tensors are not stored in standard KV mode.")
        self._check_batch_index(b)
        seq_len = int(self.seq_len[b].item())
        if seq_len == 0 or not self.block_table[b]:
            return self.storage_latent.new_zeros((0, self.latent_dim))

        parts = []
        blocks = self.block_table[b]
        last_blk_idx = len(blocks) - 1
        for blk_idx, global_blk in enumerate(blocks):
            if blk_idx == last_blk_idx:
                offset = (
                    self.write_pos[b][1]
                    if blk_idx == self.write_pos[b][0]
                    else self.block_size
                )
                parts.append(self.storage_latent[b, global_blk, :offset, :])
            else:
                parts.append(self.storage_latent[b, global_blk, :, :])
        return torch.cat(parts, dim=0)

    # ---------------------------------------------------------------------
    # Reset / stats
    # ---------------------------------------------------------------------

    def reset_seq(self, b: int) -> None:
        """
        Reset a single sequence `b`, returning its blocks to the free pool.
        """
        self._check_batch_index(b)
        for blk in self.block_table[b]:
            self._free_blocks.append(blk)
        self.block_table[b].clear()
        self.write_pos[b] = (0, 0)
        self.seq_len[b] = 0

    def reset_all(self) -> None:
        """
        Reset all sequences and reclaim all blocks.
        """
        self._free_blocks = list(range(self.max_blocks))
        for b in range(self.B):
            self.block_table[b].clear()
            self.write_pos[b] = (0, 0)
            self.seq_len[b] = 0

    # Tiny helpers for debug / logging
    def get_seq_len(self, b: int) -> int:
        self._check_batch_index(b)
        return int(self.seq_len[b].item())

    def max_seq_len(self) -> int:
        return int(self.seq_len.max().item())

    def num_used_blocks(self) -> int:
        return self.max_blocks - len(self._free_blocks)
