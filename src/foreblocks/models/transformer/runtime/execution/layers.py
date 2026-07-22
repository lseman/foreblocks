"""Encoder and decoder layer invocation, including checkpoint dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch


class LayerInvokeOwner(Protocol):
    gate_lambda: float
    use_gateskip: bool
    use_mhc: bool
    mhc_n_streams: int
    mhc_sinkhorn_iters: int
    mhc_collapse: str

    def _run_with_checkpoint(
        self,
        fn: Callable[..., torch.Tensor],
        *inputs: torch.Tensor,
        use_checkpoint: bool,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class ModelLayerInvokeStrategy:
    owner: LayerInvokeOwner
    use_checkpoint: bool

    def run_encoder_layer(
        self,
        *,
        layer,
        x,
        src_mask,
        src_key_padding_mask,
        budget,
        streams,
        attention_residual_state,
        gateskip_active_mask,
    ):
        if self.use_checkpoint:

            def checkpointed(value):
                result, _ = layer(
                    value,
                    src_mask,
                    src_key_padding_mask,
                    gate_budget=budget,
                    gate_lambda=self.owner.gate_lambda,
                    use_gateskip=self.owner.use_gateskip,
                    streams=None,
                    use_mhc=self.owner.use_mhc,
                    mhc_n_streams=self.owner.mhc_n_streams,
                    mhc_sinkhorn_iters=self.owner.mhc_sinkhorn_iters,
                    mhc_collapse=self.owner.mhc_collapse,
                    gateskip_active_mask=gateskip_active_mask,
                )
                return result

            return self.owner._run_with_checkpoint(
                checkpointed, x, use_checkpoint=True
            ), streams
        return layer(
            x,
            src_mask,
            src_key_padding_mask,
            gate_budget=budget,
            gate_lambda=self.owner.gate_lambda,
            use_gateskip=self.owner.use_gateskip,
            streams=streams,
            use_mhc=self.owner.use_mhc,
            mhc_n_streams=self.owner.mhc_n_streams,
            mhc_sinkhorn_iters=self.owner.mhc_sinkhorn_iters,
            mhc_collapse=self.owner.mhc_collapse,
            attention_residual_state=attention_residual_state,
            gateskip_active_mask=gateskip_active_mask,
        )

    def run_decoder_layer(
        self,
        *,
        layer,
        x,
        memory,
        tgt_mask,
        memory_mask,
        tgt_key_padding_mask,
        memory_key_padding_mask,
        layer_state,
        prev_state,
        budget,
        streams,
        mtp_targets,
        attention_residual_state,
        gateskip_active_mask,
    ):
        args = (
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
        )
        common = dict(
            incremental_state=layer_state,
            prev_layer_state=prev_state,
            gate_budget=budget,
            gate_lambda=self.owner.gate_lambda,
            use_gateskip=self.owner.use_gateskip,
            use_mhc=self.owner.use_mhc,
            mhc_n_streams=self.owner.mhc_n_streams,
            mhc_sinkhorn_iters=self.owner.mhc_sinkhorn_iters,
            mhc_collapse=self.owner.mhc_collapse,
            mtp_targets=mtp_targets,
            gateskip_active_mask=gateskip_active_mask,
        )
        if self.use_checkpoint:

            def checkpointed(value):
                result, _, _ = layer(value, *args, streams=None, **common)
                return result

            result = self.owner._run_with_checkpoint(
                checkpointed, x, use_checkpoint=True
            )
            return result, layer_state, streams
        return layer(
            x,
            *args,
            streams=streams,
            attention_residual_state=attention_residual_state,
            **common,
        )


__all__ = ["LayerInvokeOwner", "ModelLayerInvokeStrategy"]
