"""Mixture of Block Attention (MoBA) — learned block routing for long context.

Implements MoBA from:

    Lu, E., Jiang, Z., Liu, J., Du, Y., Jiang, T., Hong, C., … Yang, Z. (2025).
    "MoBA: Mixture of Block Attention for Long-Context LLMs."
    arXiv:2502.13189 [[arXiv]](https://arxiv.org/abs/2502.13189)

MoBA applies a Mixture-of-Experts-style router over the attention itself: keys
are partitioned into blocks, each query selects its top-k most relevant blocks
(scored against per-block key means), and attention is computed only over the
tokens in the selected blocks. This keeps full-attention expressivity while
scaling sub-quadratically with context length.

This module provides both a fused flash-varlen backend (when available) and a
pure-PyTorch reference path, falling back to the latter if the flash ops cannot
be resolved.
"""

import math
import warnings

import torch
import torch.nn.functional as F

_FLASH_MOBA_OPS: tuple[object | None, object | None] | None = None
_FLASH_MOBA_WARNED = False


def _resolve_flash_varlen_ops() -> tuple[object | None, object | None]:
    global _FLASH_MOBA_OPS
    if _FLASH_MOBA_OPS is not None:
        return _FLASH_MOBA_OPS

    candidates = [
        (
            "flash_attn.flash_attn_interface",
            "_flash_attn_varlen_forward",
            "_flash_attn_varlen_backward",
        ),
        (
            "flash_attn_interface",
            "_flash_attn_varlen_forward",
            "_flash_attn_varlen_backward",
        ),
    ]
    for module_name, forward_name, backward_name in candidates:
        try:
            module = __import__(module_name, fromlist=[forward_name, backward_name])
        except ImportError:
            continue
        forward = getattr(module, forward_name, None)
        backward = getattr(module, backward_name, None)
        if callable(forward) and callable(backward):
            _FLASH_MOBA_OPS = (forward, backward)
            return _FLASH_MOBA_OPS

    _FLASH_MOBA_OPS = (None, None)
    return _FLASH_MOBA_OPS


def _warn_flash_backend_unavailable() -> None:
    global _FLASH_MOBA_WARNED
    if _FLASH_MOBA_WARNED:
        return
    warnings.warn(
        "[MultiAttention] MoBA flash backend unavailable, falling back to reference implementation."
    )
    _FLASH_MOBA_WARNED = True


def _exclusive_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    cu = torch.zeros(lengths.numel() + 1, dtype=torch.int32, device=lengths.device)
    if lengths.numel() > 0:
        cu[1:] = lengths.to(torch.int32).cumsum(dim=0)
    return cu


class MoBAAttentionImpl:
    """Mixture of Block Attention (MoBA) variant.

    Reference: "MoBA: Mixture of Block Attention for Long-Context LLMs"
    (arXiv:2502.13189).
    """

    def __init__(self, parent):
        self.parent = parent

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        need_weights: bool,
        layer_state=None,
        **_,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        B, T_q, _ = query.shape
        q, k, v, _ = self.parent._prepare_qkv_attention(query, key, value, layer_state)
        out, weights = self._moba_attention(
            q,
            k,
            v,
            attn_mask,
            key_padding_mask,
            is_causal,
            need_weights,
        )
        return self.parent._finalize_projected_output(out, B, T_q), weights, layer_state

    def _moba_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._can_use_flash_moba(
            q,
            k,
            v,
            attn_mask,
            key_padding_mask,
            is_causal,
            need_weights,
        ):
            try:
                out = self._flash_moba_attention(q, k, v)
                return out, None
            except Exception as exc:
                warnings.warn(
                    f"[MultiAttention] MoBA flash backend failed ({exc}), falling back to reference implementation."
                )

        return self._reference_moba_attention(
            q,
            k,
            v,
            attn_mask,
            key_padding_mask,
            is_causal,
            need_weights,
        )

    def _can_use_flash_moba(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        need_weights: bool,
    ) -> bool:
        return (
            q.is_cuda
            and k.is_cuda
            and v.is_cuda
            and (not self.parent.training)
            and is_causal
            and (not self.parent.cross_attention)
            and (not need_weights)
            and attn_mask is None
            and key_padding_mask is None
            and q.size(2) == k.size(2)
            and k.size(2) == v.size(2)
        )

    def _reference_moba_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, H, T_q, D = q.shape
        T_k = k.size(2)
        block_size = max(1, min(self.parent.moba_block_size, T_k if T_k > 0 else 1))
        if T_k == 0:
            out = torch.zeros_like(q)
            return out, (
                torch.zeros(B, H, T_q, 0, device=q.device, dtype=q.dtype)
                if need_weights
                else None
            )

        n_blocks = math.ceil(T_k / block_size)
        if n_blocks <= 1:
            return self.parent._compute_attention(
                q,
                k,
                v,
                attn_mask,
                key_padding_mask,
                is_causal,
                need_weights,
            )

        block_keys, block_valid, token_to_block = self._pool_block_keys(
            k,
            key_padding_mask,
            block_size,
        )
        selected_tokens = self._build_selected_token_mask(
            q=q,
            block_keys=block_keys,
            block_valid=block_valid,
            token_to_block=token_to_block,
            T_q=T_q,
            T_k=T_k,
            n_blocks=n_blocks,
            block_size=block_size,
            is_causal=is_causal,
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.parent.scale
        scores = scores.masked_fill(~selected_tokens, float("-inf"))

        if is_causal and not self.parent.cross_attention:
            causal_mask = torch.triu(
                torch.ones(T_q, T_k, device=q.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask.view(1, 1, T_q, T_k), float("-inf"))

        scores = self.parent._apply_masks(scores, attn_mask, key_padding_mask)

        weights = F.softmax(scores, dim=-1)
        weights = torch.where(
            torch.isfinite(weights), weights, torch.zeros_like(weights)
        )
        weights = self.parent._dropout_weights(weights)

        out = torch.matmul(weights, v)
        out = self.parent._apply_gated_attention(out)
        return out, (weights if need_weights else None)

    def _flash_moba_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        flash_forward, _ = _resolve_flash_varlen_ops()
        if flash_forward is None:
            _warn_flash_backend_unavailable()
            raise RuntimeError("flash-attn varlen interface is unavailable")

        B, H, T, D = q.shape
        block_size = max(1, min(self.parent.moba_block_size, T if T > 0 else 1))
        output = torch.empty_like(q)
        for batch_idx in range(B):
            output[batch_idx] = self._flash_moba_single_sample(
                flash_forward=flash_forward,
                q=q[batch_idx],
                k=k[batch_idx],
                v=v[batch_idx],
                block_size=block_size,
                num_heads=H,
                head_dim=D,
            )
        return self.parent._apply_gated_attention(output)

    def _flash_moba_single_sample(
        self,
        *,
        flash_forward,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_size: int,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        T = q.size(1)
        q_seq = q.transpose(0, 1).contiguous()
        k_seq = k.transpose(0, 1).contiguous()
        v_seq = v.transpose(0, 1).contiguous()

        local_cu = self._chunk_cu_seqlens(T, block_size, q.device)
        local_out, local_lse = self._call_flash_varlen(
            flash_forward,
            q=q_seq,
            k=k_seq,
            v=v_seq,
            cu_seqlens_q=local_cu,
            cu_seqlens_k=local_cu,
            max_seqlen_q=block_size,
            max_seqlen_k=block_size,
            causal=True,
        )
        local_lse_sh = self._normalize_lse(local_lse, total_q=T, n_heads=num_heads)

        remote_pack = self._build_flash_remote_pack(
            q=q_seq,
            k=k_seq,
            v=v_seq,
            block_size=block_size,
            num_heads=num_heads,
        )
        if remote_pack is None:
            return local_out.transpose(0, 1).contiguous()

        remote_out, remote_lse = self._call_flash_varlen(
            flash_forward,
            q=remote_pack["q"],
            k=remote_pack["k"],
            v=remote_pack["v"],
            cu_seqlens_q=remote_pack["cu_q"],
            cu_seqlens_k=remote_pack["cu_k"],
            max_seqlen_q=T,
            max_seqlen_k=block_size,
            causal=False,
        )
        mixed = self._mix_branch_outputs(
            local_out=local_out,
            local_lse_sh=local_lse_sh,
            remote_out=remote_out,
            remote_lse=remote_lse,
            remote_q_indices=remote_pack["q_indices"],
            total_q=T,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        return mixed.transpose(0, 1).contiguous()

    def _call_flash_varlen(
        self,
        flash_forward,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse, _, _ = flash_forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.parent.scale,
            causal=causal,
            dropout_p=0.0,
        )
        return out, lse

    def _chunk_cu_seqlens(
        self,
        length: int,
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        starts = torch.arange(0, length, block_size, dtype=torch.int32, device=device)
        ends = torch.clamp(starts + block_size, max=length)
        return torch.cat(
            [torch.zeros(1, dtype=torch.int32, device=device), ends],
            dim=0,
        )

    def _build_flash_remote_pack(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_size: int,
        num_heads: int,
    ) -> dict[str, torch.Tensor] | None:
        T = q.size(0)
        n_blocks = math.ceil(T / block_size)
        target_blocks = n_blocks - 1
        topk_remote = min(max(self.parent.moba_topk - 1, 0), max(target_blocks, 0))
        if target_blocks <= 0 or topk_remote <= 0:
            return None

        full_tokens = target_blocks * block_size
        k_targets = k[:full_tokens].view(target_blocks, block_size, num_heads, -1)
        v_targets = v[:full_tokens].view(target_blocks, block_size, num_heads, -1)
        gate_weight = k_targets.mean(dim=1).float()
        gate = torch.einsum("nhd,thd->nht", gate_weight, q.float())

        token_idx = torch.arange(T, device=q.device).view(1, 1, T)
        chunk_end = (
            (torch.arange(target_blocks, device=q.device, dtype=torch.int64) + 1)
            * block_size
        ).view(target_blocks, 1, 1)
        gate = gate.masked_fill(token_idx < chunk_end, -float("inf"))

        top_idx = torch.topk(
            gate, k=topk_remote, dim=0, largest=True, sorted=False
        ).indices
        gate_mask = torch.zeros_like(gate, dtype=torch.bool)
        gate_mask.scatter_(dim=0, index=top_idx, value=True)
        gate_mask = gate_mask & torch.isfinite(gate)

        q_lens = gate_mask.sum(dim=-1).reshape(-1)
        valid_experts = q_lens > 0
        if not bool(valid_experts.any()):
            return None

        q_indices = gate_mask.reshape(target_blocks, -1).nonzero(as_tuple=True)[-1]
        q_flat = q.transpose(0, 1).contiguous().view(num_heads * T, -1)
        q_selected = q_flat.index_select(0, q_indices).unsqueeze(1)
        q_sh_indices = (
            q_indices.remainder(T) * num_heads + q_indices.div(T, rounding_mode="floor")
        ).to(torch.long)

        k_experts = (
            k_targets
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(target_blocks * num_heads, block_size, -1)
        )
        v_experts = (
            v_targets
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(target_blocks * num_heads, block_size, -1)
        )
        k_selected = k_experts[valid_experts].reshape(-1, 1, k.size(-1))
        v_selected = v_experts[valid_experts].reshape(-1, 1, v.size(-1))
        cu_q = _exclusive_cu_seqlens(q_lens[valid_experts].to(torch.int32))
        num_valid = int(valid_experts.sum().item())
        cu_k = torch.arange(
            0,
            (num_valid + 1) * block_size,
            block_size,
            dtype=torch.int32,
            device=q.device,
        )

        return {
            "q": q_selected,
            "k": k_selected,
            "v": v_selected,
            "cu_q": cu_q,
            "cu_k": cu_k,
            "q_indices": q_sh_indices,
        }

    def _normalize_lse(
        self,
        lse: torch.Tensor,
        *,
        total_q: int,
        n_heads: int,
    ) -> torch.Tensor:
        if lse.shape == (n_heads, total_q):
            return lse.transpose(0, 1).contiguous()
        if lse.shape == (total_q, n_heads):
            return lse.contiguous()
        raise ValueError(
            f"Unexpected flash-attn lse shape {tuple(lse.shape)} for total_q={total_q}, n_heads={n_heads}"
        )

    def _mix_branch_outputs(
        self,
        *,
        local_out: torch.Tensor,
        local_lse_sh: torch.Tensor,
        remote_out: torch.Tensor,
        remote_lse: torch.Tensor,
        remote_q_indices: torch.Tensor,
        total_q: int,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        local_out_2d = local_out.float().view(-1, head_dim)
        local_lse_1d = local_lse_sh.float().reshape(-1)
        remote_lse_1d = self._normalize_lse(
            remote_lse, total_q=remote_out.size(0), n_heads=1
        )
        remote_lse_1d = remote_lse_1d.float().reshape(-1)

        max_lse = local_lse_1d.clone()
        max_lse.scatter_reduce_(
            0,
            remote_q_indices,
            remote_lse_1d,
            reduce="amax",
            include_self=True,
        )

        local_shift = local_lse_1d - max_lse
        remote_shift = remote_lse_1d - max_lse.index_select(0, remote_q_indices)

        mixed_se = local_shift.exp()
        mixed_se.index_add_(0, remote_q_indices, remote_shift.exp())
        mixed_lse = mixed_se.log()

        mixed_out = local_out_2d * (local_shift - mixed_lse).exp().unsqueeze(-1)
        remote_scaled = remote_out.float().view(-1, head_dim)
        remote_factor = (
            remote_shift - mixed_lse.index_select(0, remote_q_indices)
        ).exp()
        mixed_out.index_add_(
            0,
            remote_q_indices,
            remote_scaled * remote_factor.unsqueeze(-1),
        )

        return mixed_out.view(total_q, num_heads, head_dim).to(local_out.dtype)

    def _pool_block_keys(
        self,
        k: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, H, T_k, D = k.shape
        n_blocks = math.ceil(T_k / block_size)
        T_pad = n_blocks * block_size
        pad = T_pad - T_k

        if pad > 0:
            k_pad = F.pad(k, (0, 0, 0, pad))
        else:
            k_pad = k

        token_valid = torch.ones(B, T_pad, device=k.device, dtype=torch.bool)
        if key_padding_mask is not None:
            kpm = key_padding_mask.bool()
            if pad > 0:
                kpm = F.pad(kpm, (0, pad), value=True)
            token_valid = ~kpm

        k_blocks = k_pad.view(B, H, n_blocks, block_size, D)
        token_valid_block = token_valid.view(B, 1, n_blocks, block_size, 1)
        valid_count = token_valid_block.sum(dim=3).clamp_min(1).to(k.dtype)
        pooled = (k_blocks * token_valid_block.to(k.dtype)).sum(dim=3) / valid_count
        block_valid = token_valid.view(B, n_blocks, block_size).any(dim=-1)
        token_to_block = (torch.arange(T_k, device=k.device) // block_size).clamp_max(
            n_blocks - 1
        )
        return pooled, block_valid, token_to_block

    def _build_selected_token_mask(
        self,
        *,
        q: torch.Tensor,
        block_keys: torch.Tensor,
        block_valid: torch.Tensor,
        token_to_block: torch.Tensor,
        T_q: int,
        T_k: int,
        n_blocks: int,
        block_size: int,
        is_causal: bool,
    ) -> torch.Tensor:
        B, H, _, _ = q.shape
        block_scores = (
            torch.einsum("bhtd,bhnd->bhtn", q, block_keys) * self.parent.scale
        )

        block_mask = (~block_valid).view(B, 1, 1, n_blocks).expand(B, H, T_q, n_blocks)
        q_block_idx = (torch.arange(T_q, device=q.device) // block_size).clamp_max(
            n_blocks - 1
        )
        local_blocks = F.one_hot(q_block_idx, num_classes=n_blocks).to(torch.bool)
        local_blocks = local_blocks.view(1, 1, T_q, n_blocks).expand(
            B, H, T_q, n_blocks
        )

        if is_causal and not self.parent.cross_attention:
            block_end = (
                (torch.arange(n_blocks, device=q.device) + 1) * block_size - 1
            ).clamp_max(T_k - 1)
            q_pos = torch.arange(T_q, device=q.device).view(1, 1, T_q, 1)
            future_blocks = block_end.view(1, 1, 1, n_blocks) > q_pos
            block_mask = block_mask | future_blocks

        extra_k = min(max(self.parent.moba_topk - 1, 0), max(n_blocks - 1, 0))
        selected_blocks = local_blocks.clone()

        if extra_k > 0:
            remote_mask = block_mask | local_blocks
            safe_scores = block_scores.masked_fill(remote_mask, -1e30)
            top_idx = torch.topk(safe_scores, k=extra_k, dim=-1).indices
            remote_blocks = torch.zeros_like(selected_blocks)
            valid_remote = ~safe_scores.gather(-1, top_idx).eq(-1e30)
            remote_blocks.scatter_(dim=-1, index=top_idx, src=valid_remote)
            selected_blocks = selected_blocks | remote_blocks

        selected_tokens = selected_blocks.index_select(-1, token_to_block)
        return selected_tokens
