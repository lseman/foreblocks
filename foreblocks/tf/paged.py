# ─────────────────────────────────────────────────────────────────────────────
# Paged KV Cache (torch-only)
# ─────────────────────────────────────────────────────────────────────────────
from typing import Optional, Tuple

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Paged KV Cache (torch-only)
# ─────────────────────────────────────────────────────────────────────────────
class PagedKVCache:
    """
    Fixed-page KV cache for autoregressive self-attention.

    storage_k/v: [B, Hkv, max_blocks, block_size, D]
    Per sequence we keep:
      - block_table[b]: list of global block ids (indices into dim=2 of storage)
      - write_pos[b]: (idx_in_table, offset_in_block)
      - seq_len[b]: total cached tokens
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
        self.B = batch_size
        self.Hkv = n_kv_heads
        self.D = head_dim
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.device = device
        self.dtype = dtype

        shape = (self.B, self.Hkv, self.max_blocks, self.block_size, self.D)
        self.storage_k = torch.empty(shape, device=device, dtype=dtype)
        self.storage_v = torch.empty(shape, device=device, dtype=dtype)

        self.block_table = [[] for _ in range(self.B)]
        self.write_pos = [(0, 0) for _ in range(self.B)]  # (idx_in_table, offset_in_block)
        self.seq_len = torch.zeros(self.B, device=device, dtype=torch.long)

        self._free_blocks = list(range(self.max_blocks))
        self._scratch_k = None
        self._scratch_v = None
        self._scratch_size = 0

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
        return cache

    def _alloc_block_for_seq(self, b: int):
        if not self._free_blocks:
            raise RuntimeError("PagedKVCache: out of blocks (increase max_blocks).")
        blk = self._free_blocks.pop()
        self.block_table[b].append(blk)
        self.write_pos[b] = (len(self.block_table[b]) - 1, 0)

    def append_step(self, k_bhd1: torch.Tensor, v_bhd1: torch.Tensor, b: int):
        """Append T_new tokens for batch item b. k_bhd1/v_bhd1: [Hkv, T_new, D]."""
        T_new = k_bhd1.size(1)
        if T_new == 0:
            return
        if len(self.block_table[b]) == 0:
            self._alloc_block_for_seq(b)

        remaining, src_start = T_new, 0
        while remaining > 0:
            blk_idx_in_table, offset = self.write_pos[b]
            blk_global = self.block_table[b][blk_idx_in_table]
            space = self.block_size - offset
            take = min(space, remaining)

            dst_slice = slice(offset, offset + take)
            src_slice = slice(src_start, src_start + take)
            self.storage_k[b, :, blk_global, dst_slice, :] = k_bhd1[:, src_slice, :]
            self.storage_v[b, :, blk_global, dst_slice, :] = v_bhd1[:, src_slice, :]

            offset += take
            src_start += take
            remaining -= take
            self.seq_len[b] += take

            if offset == self.block_size and remaining > 0:
                self._alloc_block_for_seq(b)
                blk_idx_in_table, offset = self.write_pos[b]

            self.write_pos[b] = (blk_idx_in_table, offset)

    # (Old gather methods kept for compatibility; decode path below avoids them)
    def gather_kv_batched(self) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = int(self.seq_len.max().item())
        if max_len == 0:
            return (self.storage_k.new_zeros(self.B, self.Hkv, 0, self.D),
                    self.storage_v.new_zeros(self.B, self.Hkv, 0, self.D))
        k_out = torch.zeros(self.B, self.Hkv, max_len, self.D, device=self.device, dtype=self.dtype)
        v_out = torch.zeros_like(k_out)
        for b, blocks in enumerate(self.block_table):
            if not blocks:
                continue
            blocks_tensor = torch.tensor(blocks, device=self.device, dtype=torch.long)
            gathered_k = self.storage_k[b, :, blocks_tensor]  # [Hkv, n_blocks, BS, D]
            gathered_v = self.storage_v[b, :, blocks_tensor]
            seq_len = int(self.seq_len[b].item())
            k_out[b, :, :seq_len] = gathered_k.reshape(self.Hkv, -1, self.D)[:, :seq_len]
            v_out[b, :, :seq_len] = gathered_v.reshape(self.Hkv, -1, self.D)[:, :seq_len]
        return k_out, v_out

    def reset_seq(self, b: int):
        for blk in self.block_table[b]:
            self._free_blocks.append(blk)
        self.block_table[b].clear()
        self.write_pos[b] = (0, 0)
        self.seq_len[b] = 0



