"""Position-encoding preparation for attention inputs.

Composable positional transform pipeline for Q/K tensors.

Registers named transform functions (RoPE, SyPE, or custom transforms) that are
applied sequentially to query and key tensors before attention computation.
Use to attach positional encoding transforms to a parent attention module.

Core API:
- PositionEncodingApplier: named-transform pipeline for Q/K positional encodings

"""

import torch

from foreblocks.ops.attention import triton_apply_rope, triton_apply_rope_pair


class PositionEncodingApplier:
    def __init__(self):
        self._transforms = []

    def add_transform(self, name: str, fn) -> None:
        self._transforms.append((str(name), fn))

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        **context,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for _, transform in self._transforms:
            q, k = transform(q, k, context)
        return q, k


class _AttentionPositions:
    """Positional transforms using the attention owner's registered modules."""

    def _build_position_encoding_applier(self) -> PositionEncodingApplier:
        applier = PositionEncodingApplier()
        applier.add_transform("rope", self._rope_position_transform)
        applier.add_transform("sype", self._sype_position_transform)
        # ALiBi is applied in _compute_attention, not here
        return applier

    def register_position_transform(self, name: str, fn) -> None:
        self.position_encoding_applier.add_transform(name, fn)

    def _rope_position_transform(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        context: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (self.use_rotary and self.rotary_emb is not None):
            return q, k
        seqlen_offset = context.get("seqlen_offset", 0)
        return self._apply_rope(q, k, seqlen_offset=seqlen_offset)

    def _sype_position_transform(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        context: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_sype:
            return q, k
        query = context.get("query")
        key = context.get("key")
        layer_state = context.get("layer_state")
        if query is None or key is None:
            return q, k
        if query.shape[1] != key.shape[1]:
            return q, k
        tau = self.sype_warp(query)
        if layer_state is not None:
            prev_tau = layer_state.get("sype_tau")
            if prev_tau is not None:
                prev_tau = prev_tau.to(device=tau.device, dtype=tau.dtype)
                if prev_tau.dim() == 0:
                    prev_tau = prev_tau.view(1)
                if prev_tau.dim() != 1 or prev_tau.shape[0] != tau.shape[0]:
                    raise ValueError(
                        f"sype_tau state must be [B], got {tuple(prev_tau.shape)} for batch {tau.shape[0]}"
                    )
                tau = tau + prev_tau.unsqueeze(1)
            layer_state["sype_tau"] = tau[:, -1].detach()
        return self.sype_rotator.rotate_qk(q, k, tau)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset=0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from foreblocks.nn.embeddings.rotary import (
            apply_rotary_emb,  # keep as-is for API compatibility
        )

        B, H, T, D = q.shape
        if isinstance(seqlen_offset, int):
            max_len = T + seqlen_offset
        else:
            max_len = T + int(seqlen_offset.max().item())

        # Update cached cos/sin up to max_len
        self.rotary_emb._update_cos_sin_cache(
            max_len,
            device=q.device,
            dtype=q.dtype,
        )

        # Switch to [B, T, H, D] for apply_rotary_emb
        q_bt_hd = q.transpose(1, 2).contiguous()
        k_bt_hd = k.transpose(1, 2).contiguous()

        cos_q, sin_q = self.rotary_emb._cos_cached, self.rotary_emb._sin_cached
        if getattr(self.rotary_emb, "scale", None) is None:
            cos_k, sin_k = cos_q, sin_q
        else:
            cos_k, sin_k = self.rotary_emb._cos_k_cached, self.rotary_emb._sin_k_cached

        use_triton_rope = (
            q.is_cuda
            and k.is_cuda
            and isinstance(seqlen_offset, int)
            and (not bool(self.rotary_emb.interleaved))
            and cos_q.dim() == 2
            and sin_q.dim() == 2
            and cos_k.dim() == 2
            and sin_k.dim() == 2
        )
        if use_triton_rope:
            try:
                if cos_k is cos_q and sin_k is sin_q:
                    # Common path (no rotary scale): one fused launch does q and k
                    q_tr, k_tr = triton_apply_rope(
                        q,
                        k,
                        cos_q,
                        sin_q,
                        seqlen_offset=seqlen_offset,
                    )
                else:
                    # Scaled RoPE: different cos/sin for k. Use the dedicated
                    # pair function so each tensor is processed with only its
                    # own cos/sin tables (no wasted work processing both).
                    q_tr, k_tr = triton_apply_rope_pair(
                        q,
                        k,
                        cos_q,
                        sin_q,
                        cos_k,
                        sin_k,
                        seqlen_offset=seqlen_offset,
                    )
                return q_tr.contiguous(), k_tr.contiguous()
            except Exception:
                pass

        q_bt_hd = apply_rotary_emb(
            q_bt_hd,
            cos_q,
            sin_q,
            interleaved=self.rotary_emb.interleaved,
            seqlen_offsets=seqlen_offset,
            inplace=False,
        )
        k_bt_hd = apply_rotary_emb(
            k_bt_hd,
            cos_k,
            sin_k,
            interleaved=self.rotary_emb.interleaved,
            seqlen_offsets=seqlen_offset,
            inplace=False,
        )

        # Back to [B, H, T, D]
        return (
            q_bt_hd.transpose(1, 2).contiguous(),
            k_bt_hd.transpose(1, 2).contiguous(),
        )
