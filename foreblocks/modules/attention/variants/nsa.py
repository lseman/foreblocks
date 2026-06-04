"""Native Sparse Attention (NSA) — hardware-aligned hierarchical sparse attention.

Implements the inference-time NSA mechanism from:

    Yuan, J., Gao, H., Dai, D., Luo, J., Zhao, L., Zhang, Z., … Liang, W.
    (2025).
    "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse
    Attention."
    arXiv:2502.11089 [[arXiv]](https://arxiv.org/abs/2502.11089)

NSA replaces dense attention with three parallel branches whose outputs are
combined per-token by independent learned gates (paper Eq. 5):

    o* = Σ_{c ∈ {cmp, slc, win}} g_c · Attn(q, K̃_c, Ṽ_c)

where each gate ``g_c ∈ [0, 1]`` comes from an MLP + sigmoid on the input
features (they are independent sigmoids, not a softmax over branches):

    * **Compression** (cmp): keys/values are pooled into block-level
      representations and attended over coarsely. The paper uses a learnable
      MLP φ with intra-block position encoding; this implementation uses
      (padding-aware) mean pooling per block as a lightweight substitute.
    * **Selection** (slc): the compression attention scores are reused as
      block-importance scores (paper Eq. 9, ``p_cmp = softmax(qᵀ K̃_cmp)``);
      the top-n blocks are selected and dense attention is computed over only
      the tokens in those blocks. Here the budget is parameterized as
      ``nsa_topk_ratio`` (fraction of blocks) rather than a fixed count.
    * **Sliding window** (win): local attention over a fixed recent window,
      delegated to :class:`SlidingWindowAttentionImpl`.

Notes
-----
This is a pure-PyTorch implementation intended for correctness and training,
not the fused Triton ``parallel_nsa`` kernel. It mean-pools for compression
rather than learning φ, and does not share block selection across GQA head
groups; otherwise the branch structure, score reuse, masking and gating follow
the paper.
"""

import math

import torch
import torch.nn.functional as F

from foreblocks.modules.attention.variants.sliding_window import SlidingWindowAttentionImpl


class NSAImpl:
    """Native Sparse Attention implementation (see module docstring).

    Wraps a parent multi-head attention module, reusing its projections,
    scale, masking helpers and output gating. ``parent.nsa_gate_proj`` is the
    ``Linear(head_dim, 3)`` producing the three branch gate logits; when it is
    ``None`` the layer falls back to the parent's dense attention.
    """

    def __init__(self, parent):
        self.parent = parent

    def forward(
        self,
        query,
        key,
        value,
        attn_mask,
        key_padding_mask,
        is_causal,
        need_weights,
        layer_state=None,
        **_,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        B, T_q, _ = query.shape
        q, k, v, _ = self.parent._prepare_qkv_attention(query, key, value, layer_state)
        out, weights = self._nsa_attention(
            q,
            k,
            v,
            attn_mask,
            key_padding_mask,
            is_causal,
            need_weights,
        )
        return self.parent._finalize_projected_output(out, B, T_q), weights, layer_state

    def _nsa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.parent.nsa_gate_proj is None:
            return self.parent._compute_attention(
                q,
                k,
                v,
                attn_mask,
                key_padding_mask,
                is_causal,
                need_weights,
            )

        compressed_out, block_scores, block_mask, block_size = (
            self._nsa_compressed_branch(
                q,
                k,
                v,
                attn_mask,
                key_padding_mask,
                is_causal,
            )
        )
        selected_out = self._nsa_selected_blocks_branch(
            q,
            k,
            v,
            attn_mask,
            key_padding_mask,
            is_causal,
            block_scores,
            block_mask,
            block_size,
        )
        sliding_out, _ = SlidingWindowAttentionImpl(self.parent).manual(
            q,
            k,
            v,
            attn_mask,
            key_padding_mask,
            is_causal,
            need_weights=False,
            apply_gate=False,
        )

        # Per-branch independent sigmoid gates g_c ∈ [0, 1] (paper Eq. 5).
        # Each branch contributes independently, so the gates do not sum to 1
        # (this is a sigmoid, not a softmax over branches).
        gate_logits = self.parent.nsa_gate_proj(q)
        gates = torch.sigmoid(gate_logits)
        out = (
            gates[..., 0:1] * compressed_out
            + gates[..., 1:2] * selected_out
            + gates[..., 2:3] * sliding_out
        )
        out = self.parent._apply_gated_attention(out)
        return out, None

    def _nsa_compressed_branch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        B, H, T_q, D = q.shape
        T_k = k.size(2)
        block_size = max(1, min(self.parent.nsa_block_size, T_k if T_k > 0 else 1))
        n_blocks = (T_k + block_size - 1) // block_size
        T_pad = n_blocks * block_size
        pad = T_pad - T_k

        if pad > 0:
            k_pad = F.pad(k, (0, 0, 0, pad))
            v_pad = F.pad(v, (0, 0, 0, pad))
        else:
            k_pad, v_pad = k, v

        k_blocks = k_pad.view(B, H, n_blocks, block_size, D)
        v_blocks = v_pad.view(B, H, n_blocks, block_size, D)

        token_valid = torch.ones(B, T_pad, device=q.device, dtype=torch.bool)
        if key_padding_mask is not None:
            kpm = key_padding_mask.bool()
            if pad > 0:
                kpm = F.pad(kpm, (0, pad), value=True)
            token_valid = ~kpm

        token_valid_block = token_valid.view(B, 1, n_blocks, block_size, 1)
        valid_count = token_valid_block.sum(dim=3).clamp_min(1).to(k.dtype)

        k_comp = (k_blocks * token_valid_block.to(k.dtype)).sum(dim=3) / valid_count
        v_comp = (v_blocks * token_valid_block.to(v.dtype)).sum(dim=3) / valid_count

        scores_blocks = torch.matmul(q, k_comp.transpose(-2, -1)) * self.parent.scale

        block_valid = token_valid.view(B, n_blocks, block_size).any(dim=-1)
        block_mask = (~block_valid).view(B, 1, 1, n_blocks).expand(B, H, T_q, n_blocks)

        if is_causal and not self.parent.cross_attention:
            q_pos = torch.arange(T_q, device=q.device).view(1, 1, T_q, 1)
            block_start = (torch.arange(n_blocks, device=q.device) * block_size).view(
                1, 1, 1, n_blocks
            )
            block_mask = block_mask | (block_start > q_pos)

        if attn_mask is not None:
            full = self.parent._normalize_attn_mask(attn_mask, B, H, T_q, T_k)
            if pad > 0:
                full = F.pad(full, (0, pad), value=True)
            block_all_masked = full.view(B, H, T_q, n_blocks, block_size).all(dim=-1)
            block_mask = block_mask | block_all_masked

        scores_blocks = scores_blocks.masked_fill(block_mask, float("-inf"))
        w_blocks = F.softmax(scores_blocks, dim=-1)
        w_blocks = torch.where(
            torch.isfinite(w_blocks), w_blocks, torch.zeros_like(w_blocks)
        )
        w_blocks = self.parent._dropout_weights(w_blocks)

        out = torch.matmul(w_blocks, v_comp)
        return out, scores_blocks, block_mask, block_size

    def _nsa_selected_blocks_branch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        block_scores: torch.Tensor,
        block_mask: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        B, H, T_q, D = q.shape
        T_k = k.size(2)
        n_blocks = block_scores.size(-1)
        topk_ratio = max(0.0, float(self.parent.nsa_topk_ratio))
        topk_blocks = max(1, min(n_blocks, int(math.ceil(topk_ratio * n_blocks))))

        safe_scores = block_scores.masked_fill(block_mask, -1e30)
        top_idx = torch.topk(safe_scores, k=topk_blocks, dim=-1).indices

        selected_blocks = torch.zeros(
            B, H, T_q, n_blocks, device=q.device, dtype=torch.bool
        )
        selected_blocks.scatter_(dim=-1, index=top_idx, value=True)

        token_to_block = (torch.arange(T_k, device=q.device) // block_size).clamp_max(
            n_blocks - 1
        )
        selected_tokens = selected_blocks.index_select(-1, token_to_block)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.parent.scale
        scores = scores.masked_fill(~selected_tokens, float("-inf"))

        if is_causal and not self.parent.cross_attention:
            causal_mask = torch.triu(
                torch.ones(T_q, T_k, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask.view(1, 1, T_q, T_k), float("-inf"))

        scores = self.parent._apply_masks(scores, attn_mask, key_padding_mask)

        weights = F.softmax(scores, dim=-1)
        weights = torch.where(
            torch.isfinite(weights), weights, torch.zeros_like(weights)
        )
        weights = self.parent._dropout_weights(weights)
        return torch.matmul(weights, v)
