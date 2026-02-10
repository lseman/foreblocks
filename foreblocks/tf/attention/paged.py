# ─────────────────────────────────────────────────────────────────────────────
# Paged KV Cache (torch-only, slightly enhanced)
# ─────────────────────────────────────────────────────────────────────────────
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


class PagedKVCache:
    """
    Fixed-page KV cache for autoregressive self-attention (torch-only).

    Layout
    ------
    storage_k/v: [B, Hkv, max_blocks, block_size, D]

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

        shape = (self.B, self.Hkv, self.max_blocks, self.block_size, self.D)
        self.storage_k = torch.empty(shape, device=self.device, dtype=self.dtype)
        self.storage_v = torch.empty(shape, device=self.device, dtype=self.dtype)

        # Per-sequence metadata
        self.block_table: List[List[int]] = [[] for _ in range(self.B)]
        self.write_pos: List[Tuple[int, int]] = [
            (0, 0) for _ in range(self.B)
        ]  # (idx_in_table, offset_in_block)
        self.seq_len: Tensor = torch.zeros(
            self.B, device=self.device, dtype=torch.long
        )

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
                or cache.block_size != block_size
            ):
                # You can choose to raise instead if you don't expect this case.
                cache = PagedKVCache(
                    batch_size=batch_size,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
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
            raise RuntimeError(
                "PagedKVCache: out of blocks (increase max_blocks)."
            )
        blk = self._free_blocks.pop()
        self.block_table[b].append(blk)
        self.write_pos[b] = (len(self.block_table[b]) - 1, 0)

    # ---------------------------------------------------------------------
    # Public API: writing
    # ---------------------------------------------------------------------
    def append_step(self, k_bhd1: Tensor, v_bhd1: Tensor, b: int) -> None:
        """
        Append new tokens for batch item `b`.

        Parameters
        ----------
        k_bhd1, v_bhd1 : Tensor
            Shape [Hkv, T_new, D]. Must match the cache's (Hkv, D).
        b : int
            Batch index in [0, B).

        Notes
        -----
        - This only handles one sequence index `b` at a time.
        - You can call this in a loop over batch dimension in your decoder.
        """
        self._check_batch_index(b)

        if k_bhd1.ndim != 3:
            raise ValueError(
                f"k_bhd1 must be [Hkv, T_new, D], got shape {tuple(k_bhd1.shape)}"
            )
        if v_bhd1.ndim != 3:
            raise ValueError(
                f"v_bhd1 must be [Hkv, T_new, D], got shape {tuple(v_bhd1.shape)}"
            )
        if k_bhd1.shape[0] != self.Hkv or v_bhd1.shape[0] != self.Hkv:
            raise ValueError(
                f"Expected Hkv={self.Hkv}, got k={k_bhd1.shape[0]}, v={v_bhd1.shape[0]}"
            )
        if k_bhd1.shape[2] != self.D or v_bhd1.shape[2] != self.D:
            raise ValueError(
                f"Expected head_dim={self.D}, got k={k_bhd1.shape[2]}, v={v_bhd1.shape[2]}"
            )
        if k_bhd1.shape != v_bhd1.shape:
            raise ValueError(
                f"k_bhd1 and v_bhd1 must have same shape, got {k_bhd1.shape} vs {v_bhd1.shape}"
            )

        T_new = k_bhd1.size(1)
        if T_new == 0:
            return

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
            self.storage_k[b, :, blk_global, dst_slice, :] = k_bhd1[:, src_slice, :]
            self.storage_v[b, :, blk_global, dst_slice, :] = v_bhd1[:, src_slice, :]

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
        max_len = int(self.seq_len.max().item())
        if max_len == 0:
            empty_k = self.storage_k.new_zeros(
                (self.B, self.Hkv, 0, self.D)
            )
            empty_v = self.storage_v.new_zeros(
                (self.B, self.Hkv, 0, self.D)
            )
            return empty_k, empty_v

        k_out = torch.zeros(
            self.B, self.Hkv, max_len, self.D,
            device=self.storage_k.device,
            dtype=self.storage_k.dtype,
        )
        v_out = torch.zeros_like(k_out)

        for b, blocks in enumerate(self.block_table):
            if not blocks:
                continue
            seq_len = int(self.seq_len[b].item())
            if seq_len == 0:
                continue

            blocks_tensor = torch.tensor(
                blocks, device=self.storage_k.device, dtype=torch.long
            )
            # [Hkv, n_blocks, BS, D] -> [Hkv, n_blocks * BS, D]
            gathered_k = self.storage_k[b, :, blocks_tensor]  # [Hkv, nb, BS, D]
            gathered_v = self.storage_v[b, :, blocks_tensor]

            k_flat = gathered_k.reshape(self.Hkv, -1, self.D)[:, :seq_len]
            v_flat = gathered_v.reshape(self.Hkv, -1, self.D)[:, :seq_len]

            k_out[b, :, :seq_len] = k_flat
            v_out[b, :, :seq_len] = v_flat

        return k_out, v_out

    def gather_kv_for_seq(self, b: int) -> Tuple[Tensor, Tensor]:
        """
        Gather KV for a single sequence b into contiguous [Hkv, T, D] tensors.
        Useful when you only decode one sequence at a time.
        """
        self._check_batch_index(b)
        seq_len = int(self.seq_len[b].item())
        if seq_len == 0 or not self.block_table[b]:
            empty_k = self.storage_k.new_zeros((self.Hkv, 0, self.D))
            empty_v = self.storage_v.new_zeros((self.Hkv, 0, self.D))
            return empty_k, empty_v

        blocks = self.block_table[b]
        blocks_tensor = torch.tensor(
            blocks, device=self.storage_k.device, dtype=torch.long
        )
        gathered_k = self.storage_k[b, :, blocks_tensor]  # [Hkv, nb, BS, D]
        gathered_v = self.storage_v[b, :, blocks_tensor]

        k_flat = gathered_k.reshape(self.Hkv, -1, self.D)[:, :seq_len]
        v_flat = gathered_v.reshape(self.Hkv, -1, self.D)[:, :seq_len]
        return k_flat, v_flat

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

