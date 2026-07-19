"""foreblocks.models.transformer.core.base.

Base classes and execution plumbing for modular transformer encoder/decoder.

Implements ResidualBlockMixin and MHCBlockMixin that eliminate repetitive code
across encoder and decoder layers. Provides support for pre/post/sandwich
normalization, GateSkip residual gating, manifold-constrained hyper-connections
(mHC), Mixture-of-Depths routing, and attention residual tracking.
Attention-residual state lives in ``residual_state`` and tensor routing lives
in ``routing``; this module owns layer/model base classes.

Core API:
- ResidualBlockMixin: shared residual logic for non-mHC blocks
- MHCBlockMixin: shared stream-wise mHC block logic
- BaseTransformerLayer: base layer with FFN/MoE and aux_loss tracking
- BaseTransformer: abstract encoder/decoder base with embedding, layer building
- ResidualRunCfg: frozen config for residual computation

"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from foreblocks.layers.embeddings import LearnablePositionalEncoding, PositionalEncoding
from foreblocks.layers.norms import RMSNorm, create_norm_layer
from foreblocks.models.transformer.runtime.execution import (
    MHCBlockMixin,
    NormWrapper,
    ResidualBlockMixin,
    ResidualRunCfg,
    _LayerExecutionStrategy,
)
from foreblocks.models.transformer.features.mhc import (
    MHCHyperConnection,
    mhc_init_streams,
)
from foreblocks.models.transformer.features.patching import (
    PatchTokenizer,
)
from foreblocks.models.transformer.runtime.residual_state import (
    _attention_residual_values,
    _init_attention_residual_state,
)
from foreblocks.modules.attention.utils.residuals import (
    AttentionResidual,
    normalize_attention_residual_mode,
)
from foreblocks.modules.moe.ff import FeedForwardBlock
from foreblocks.modules.skip.gateskip import (
    BudgetScheduler,
)
from foreblocks.modules.skip.mod import (
    LayerDropoutSchedule,
    MoDBudgetScheduler,
    MoDRouter,
    mod_capacity,
    mod_routed_indices,
    mod_router_aux_loss,
    mod_topk_mask,
)

if TYPE_CHECKING:
    from foreblocks.models.transformer.core.decoder import TransformerDecoder
    from foreblocks.models.transformer.core.encoder import TransformerEncoder


# ──────────────────────────────────────────────────────────────────────────────
# Shared transformer layer base
# ──────────────────────────────────────────────────────────────────────────────
class BaseTransformerLayer(nn.Module):
    """
    Base layer with a FeedForwardBlock + MoE/GateSkip plumbing and aux_loss buffer.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_swiglu: bool = True,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        use_gateskip: bool = False,
        gate_budget: Optional[float] = None,
        gate_lambda: float = 0.1,
        # mHC
        use_mhc: bool = False,
        mhc_n_streams: int = 4,
        mhc_sinkhorn_iters: int = 20,
        mhc_collapse: str = "first",
        moe_use_latent: bool = False,
        moe_latent_dim: Optional[int] = None,
        moe_latent_d_ff: Optional[int] = None,
        use_attention_matching_compaction: bool = False,
        attention_matching_keep_ratio: float = 0.25,
        attention_matching_trigger_len: int = 512,
        attention_matching_min_keep: int = 64,
        attention_matching_query_budget: int = 64,
        attention_matching_force_single_step: bool = False,
    ):
        super().__init__()
        self.use_moe = use_moe
        self.use_gateskip = use_gateskip
        self.gate_budget = gate_budget
        self.gate_lambda = gate_lambda
        self.d_model = int(d_model)

        # mHC knobs (layer-level)
        self.use_mhc = bool(use_mhc)
        self.mhc_n_streams = int(mhc_n_streams)
        self.mhc_sinkhorn_iters = int(mhc_sinkhorn_iters)
        self.mhc_collapse = str(mhc_collapse)

        self.feed_forward = FeedForwardBlock(
            d_model=d_model,
            dim_ff=dim_feedforward,
            dropout=dropout,
            use_swiglu=use_swiglu,
            activation=activation,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            moe_use_latent=moe_use_latent,
            moe_latent_dim=moe_latent_dim,
            moe_latent_d_ff=moe_latent_d_ff,
        )

        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)

        # Float accumulator for MoE/aux loss, aggregated by the parent model.
        # Stored as float (not tensor) so _aggregate_aux_loss in BaseTransformer
        # can combine per-layer values into a single loss tensor.
        self._aux_loss: float = 0.0
        self._aux_loss_device: torch.device | None = None

    def _reset_aux_loss(self) -> None:
        self._aux_loss = 0.0

    def _update_aux_loss(self, new_loss: float | torch.Tensor) -> None:
        if torch.is_tensor(new_loss):
            val = float(new_loss.detach())
        else:
            val = float(new_loss)
        if val:
            self._aux_loss += val

    def _record_aux_loss_device(self, device: torch.device) -> None:
        """Call once per forward pass to pin the device for aggregation."""
        self._aux_loss_device = device

    def _make_exec_strategy(
        self,
        *,
        x: torch.Tensor,
        streams: Optional[torch.Tensor],
        attention_residual_state: Optional[dict] = None,
    ) -> _LayerExecutionStrategy:
        if attention_residual_state is not None:
            return _LayerExecutionStrategy(
                owner=self,
                use_mhc=False,
                x=attention_residual_state["current"],
                use_attention_residual=True,
                attention_residual_state=attention_residual_state,
            )

        if not self.use_mhc:
            return _LayerExecutionStrategy(owner=self, use_mhc=False, x=x)

        self._ensure_mhc_mixers()
        if streams is None:
            streams = mhc_init_streams(x, self.mhc_n_streams)
        else:
            if streams.dim() != 4:
                raise ValueError(f"mHC streams must be [B,N,T,D], got {streams.shape}")
            if streams.shape[1] != self.mhc_n_streams:
                raise ValueError(
                    f"mHC streams N={streams.shape[1]} != configured {self.mhc_n_streams}"
                )

        return _LayerExecutionStrategy(owner=self, use_mhc=True, streams=streams)

    def _new_mhc_connection(self) -> MHCHyperConnection:
        conn = MHCHyperConnection(
            d_model=self.d_model,
            n_streams=self.mhc_n_streams,
            sinkhorn_iters=self.mhc_sinkhorn_iters,
        )
        ref = next(self.parameters(), None)
        if ref is not None:
            conn = conn.to(device=ref.device, dtype=ref.dtype)
        return conn

    def _validate_runtime_mhc_overrides(
        self,
        *,
        use_mhc: Optional[bool],
        mhc_n_streams: Optional[int],
        mhc_sinkhorn_iters: Optional[int],
        mhc_collapse: Optional[str],
    ) -> None:
        """Reject explicit runtime overrides that conflict with configured values.

        Only checks parameters *passed to forward()*, not the model's own
        attributes.  The model setters (set_use_mhc / set_mhc_params) are
        deprecated — configure everything at construction time instead.
        """
        requested = {
            "use_mhc": use_mhc,
            "mhc_n_streams": mhc_n_streams,
            "mhc_sinkhorn_iters": mhc_sinkhorn_iters,
            "mhc_collapse": mhc_collapse,
        }
        configured = {
            "use_mhc": self.use_mhc,
            "mhc_n_streams": self.mhc_n_streams,
            "mhc_sinkhorn_iters": self.mhc_sinkhorn_iters,
            "mhc_collapse": self.mhc_collapse,
        }
        conflicts = [
            name for name, value in requested.items()
            if value is not None and value != configured[name]
        ]
        if conflicts:
            names = ", ".join(conflicts)
            raise ValueError(
                f"runtime mHC overrides ({names}) are no longer mutable; "
                "configure them when constructing the transformer"
            )

    def _build_residual_cfg(
        self,
        *,
        use_gateskip: Optional[bool],
        gate_budget: Optional[float],
        gate_lambda: Optional[float],
        training: bool,
    ) -> ResidualRunCfg:
        """Resolve runtime GateSkip knobs into a single config object."""
        _use_gk = self.use_gateskip if use_gateskip is None else bool(use_gateskip)
        _budget = self.gate_budget if gate_budget is None else gate_budget
        _lambda = self.gate_lambda if gate_lambda is None else float(gate_lambda)
        return ResidualRunCfg(
            use_gateskip=_use_gk,
            gate_budget=_budget,
            gate_lambda=_lambda,
            training=training,
        )

    def _finalize_gateskip_aux(
        self,
        cfg: ResidualRunCfg,
        aux_l2_terms: List[torch.Tensor],
    ) -> None:
        """Accumulate GateSkip regularization term once per layer forward."""
        if cfg.use_gateskip and cfg.gate_lambda > 0 and aux_l2_terms:
            self._update_aux_loss(cfg.gate_lambda * torch.stack(aux_l2_terms).mean())

    def _ff_forward_with_aux(
        self,
        x: torch.Tensor,
        mtp_targets: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Shared FFN / MoE forward with auxiliary loss accounting."""
        if self.use_moe:
            out, aux = self.feed_forward(
                x,
                return_aux_loss=True,
                mtp_targets=mtp_targets,
                padding_mask=padding_mask,
            )
            self._update_aux_loss(aux)
            return out
        return self.feed_forward(x)

    def _run_attnres_core(
        self,
        x: torch.Tensor,
        normw: NormWrapper,
        core_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[dict]]],
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        x_in = normw.norm(x) if normw.strategy in ("pre_norm", "sandwich_norm") else x
        out, updated = core_fn(x_in)
        out = normw.dropout(out)
        if normw.strategy in ("post_norm", "sandwich_norm"):
            out = normw.norm(out)
        return out, updated


# ──────────────────────────────────────────────────────────────────────────────
# Base transformer
# ──────────────────────────────────────────────────────────────────────────────
class BaseTransformer(nn.Module, ABC):
    """
    Base class shared by TransformerEncoder and TransformerDecoder.

    Handles:
      - input projection + positional / time embedding hook
      - layer construction with attention_mode routing
      - GateSkip runtime knobs and BudgetScheduler
      - gradient checkpointing (per-layer)
      - aux_loss aggregation (FIXED: uses executed layer indices)
      - Optional mHC knobs and stream propagation
      - Optional PatchTST-style patching knobs
      - Optional Mixture-of-Depths / Dynamic Layer Skipping
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        att_type: str = "standard",
        norm_strategy: str = "pre_norm",
        custom_norm: str = "rms",
        layer_norm_eps: float = 1e-5,
        max_seq_len: int = 5000,
        pos_encoding_scale: float = 1.0,
        pos_encoder: Optional[nn.Module] = None,
        pos_encoding_type: Literal["sinusoidal", "learnable", "rope", "alibi"] = "rope",
        rope_base: float = 10000.0,
        rope_scaling_type: Literal["none", "yarn", "ntk", "linear"] = "none",
        rope_scaling_factor: float = 1.0,
        use_gradient_checkpointing: bool = False,
        share_layers: bool = False,
        use_final_norm: bool = True,
        use_swiglu: bool = True,
        freq_modes: int = 32,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        # GateSkip toggles
        use_gateskip: bool = False,
        gate_budget: Optional[float] = None,
        gate_lambda: float = 0.1,
        # Attention mode
        attention_mode: Literal[
            "standard",
            "linear",
            "sype",
            "hybrid",
            "kimi",
            "hybrid_kimi",
            "kimi_3to1",
            "gated_delta",
            "hybrid_gdn",
            "gdn_3to1",
            "gla",
            "gla_hybrid",
            "gla_3to1",
            "deltanet",
            "deltanet_hybrid",
            "deltanet_3to1",
            "gated_deltanet",
            "gated_deltanet_hybrid",
            "gated_deltanet_3to1",
        ] = "standard",
        # mHC toggles
        use_mhc: bool = False,
        mhc_n_streams: int = 4,
        mhc_sinkhorn_iters: int = 20,
        mhc_collapse: str = "first",  # "first" or "mean"
        # PatchTST-style patching (best default: patch encoder only)
        patch_encoder: bool = True,
        patch_len: int = 16,
        patch_stride: int = 8,
        patch_pad_end: bool = True,
        # Mixture-of-Depths
        use_mod: bool = False,
        mod_mode: Literal["token", "seq"] = "token",
        mod_lambda: float = 0.05,
        mod_budget_scheduler: Optional[MoDBudgetScheduler] = None,
        # Global scaling for MoE aux loss (FIX)
        moe_aux_lambda: float = 1.0,
        use_attention_residual: bool = False,
        attn_residual_type: str = "full",
        attention_residual_block_size: int = 8,
        layer_dropout_schedule: Optional["LayerDropoutSchedule"] = None,
        initializer_range: float = 0.02,
        depth_scaled_init: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gateskip = use_gateskip
        self.gate_budget = gate_budget
        self.gate_lambda = gate_lambda
        self.pos_encoding_type = str(pos_encoding_type)
        self.rope_base = float(rope_base)
        self.rope_scaling_type = str(rope_scaling_type)
        self.rope_scaling_factor = float(rope_scaling_factor)
        # Backward-compatible behavior:
        # if caller sets att_type to a routed mode but leaves attention_mode default,
        # promote attention_mode so the intended path is used.
        if attention_mode == "standard" and att_type in {
            "linear",
            "sype",
            "kimi",
            "gated_delta",
        }:
            attention_mode = att_type
        self.attention_mode = attention_mode
        self.att_type = att_type
        self.budget_scheduler: Optional[BudgetScheduler] = None

        # MoE aux scaling (FIX)
        self.moe_aux_lambda = float(moe_aux_lambda)

        self.use_attention_residual = bool(use_attention_residual)
        self.attention_residual_mode = normalize_attention_residual_mode(
            attn_residual_type
        )
        self.attention_residual_block_size = int(attention_residual_block_size)
        if self.attention_residual_block_size <= 0:
            raise ValueError("attention_residual_block_size must be > 0")
        self.output_attention_residual = (
            AttentionResidual(self.d_model) if self.use_attention_residual else None
        )

        # mHC model-level
        self.use_mhc = bool(use_mhc)
        self.mhc_n_streams = int(mhc_n_streams)
        self.mhc_sinkhorn_iters = int(mhc_sinkhorn_iters)
        self.mhc_collapse = str(mhc_collapse)

        # PatchTST-style patching knobs (subclasses decide whether to apply)
        self.patch_encoder = bool(patch_encoder)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.patch_pad_end = bool(patch_pad_end)

        # Mixture-of-Depths knobs
        self.use_mod = bool(use_mod)
        self.mod_mode = str(mod_mode)
        self.mod_lambda = float(mod_lambda)
        self.mod_budget_scheduler = mod_budget_scheduler

        # Per-layer dropout schedule
        self.layer_dropout_schedule = layer_dropout_schedule
        self.initializer_range = float(initializer_range)
        self.depth_scaled_init = bool(depth_scaled_init)

        # Modules
        self.patcher = PatchTokenizer(
            self.d_model, self.patch_len, self.patch_stride, pad_end=self.patch_pad_end
        )

        self.input_adapter = nn.Linear(input_size, self.d_model)
        # Only instantiate input-level positional encoding when pos_encoding_type
        # demands it (sinusoidal / learnable). RoPE and ALiBi handle position
        # encoding internally inside the attention module.
        self.pos_encoder: nn.Module | None = None
        if pos_encoder is not None:
            self.pos_encoder = pos_encoder
        elif self.pos_encoding_type in ("sinusoidal", "learnable"):
            if self.attention_mode == "sype" or self.att_type == "sype":
                self.pos_encoder = LearnablePositionalEncoding(
                    self.d_model,
                    max_len=max_seq_len,
                    dropout=dropout,
                    scale_strategy="fixed",
                    scale_value=pos_encoding_scale,
                    use_layer_norm=False,
                )
            else:
                self.pos_encoder = PositionalEncoding(
                    self.d_model, max_len=max_seq_len, scale=pos_encoding_scale
                )

        self.register_buffer("_causal_mask", torch.empty(0, 0), persistent=False)

        # Build layers (per-layer attention type decided by attention_mode)
        if share_layers:
            layer_attn_type = self._get_layer_attention_type(0)
            layer_kwargs = self._build_layer_kwargs(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                att_type,
                norm_strategy,
                custom_norm,
                layer_norm_eps,
                use_swiglu,
                freq_modes,
                use_moe,
                num_experts,
                top_k,
                use_gateskip,
                gate_budget,
                gate_lambda,
                layer_attn_type,
                use_mhc=self.use_mhc,
                mhc_n_streams=self.mhc_n_streams,
                mhc_sinkhorn_iters=self.mhc_sinkhorn_iters,
                mhc_collapse=self.mhc_collapse,
                use_attention_residual=self.use_attention_residual,
                attn_residual_type=self.attention_residual_mode,
                attention_residual_block_size=self.attention_residual_block_size,
                pos_encoding_type=self.pos_encoding_type,
                rope_base=self.rope_base,
                rope_scaling_type=self.rope_scaling_type,
                rope_scaling_factor=self.rope_scaling_factor,
                **kwargs,
            )
            self.shared_layer = self._make_layer(**layer_kwargs)
            self.layers = None
        else:
            self.layers = nn.ModuleList()
            for i in range(num_layers):
                layer_attn_type = self._get_layer_attention_type(i)
                # Per-layer dropout: use the schedule if available, else flat dropout
                layer_dropout = (
                    self.layer_dropout_schedule.get_dropout(i)
                    if self.layer_dropout_schedule
                    else dropout
                )
                layer_kwargs = self._build_layer_kwargs(
                    d_model,
                    nhead,
                    dim_feedforward,
                    layer_dropout,
                    activation,
                    att_type,
                    norm_strategy,
                    custom_norm,
                    layer_norm_eps,
                    use_swiglu,
                    freq_modes,
                    use_moe,
                    num_experts,
                    top_k,
                    use_gateskip,
                    gate_budget,
                    gate_lambda,
                    layer_attn_type,
                    use_mhc=self.use_mhc,
                    mhc_n_streams=self.mhc_n_streams,
                    mhc_sinkhorn_iters=self.mhc_sinkhorn_iters,
                    mhc_collapse=self.mhc_collapse,
                    use_attention_residual=self.use_attention_residual,
                    attn_residual_type=self.attention_residual_mode,
                    attention_residual_block_size=self.attention_residual_block_size,
                    pos_encoding_type=self.pos_encoding_type,
                rope_base=self.rope_base,
                rope_scaling_type=self.rope_scaling_type,
                rope_scaling_factor=self.rope_scaling_factor,
                    **kwargs,
                )
                self.layers.append(self._make_layer(**layer_kwargs))
            self.shared_layer = None

        self.final_norm = (
            create_norm_layer(custom_norm, d_model, layer_norm_eps)
            if use_final_norm
            else nn.Identity()
        )

        # Per-layer gates (even if share_layers=True)
        self.mod_routers = nn.ModuleList(
            [
                MoDRouter(
                    d_model=self.d_model,
                    mode=self.mod_mode,
                    hidden=0,
                    init_bias=2.0,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Model-level aux_loss is a tensor buffer (set by _aggregate_aux_loss).
        # mod_aux_loss is a simple float accumulated during forward.
        self.aux_loss: torch.Tensor = torch.tensor(0.0)
        self.mod_aux_loss: float = 0.0
        self._materialize_configured_attention_backends()
        self.apply(self._init_weights)
        self._apply_depth_scaled_initialization()
        self._print_init_summary(
            att_type=att_type,
            custom_norm=custom_norm,
            norm_strategy=norm_strategy,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            share_layers=share_layers,
            use_final_norm=use_final_norm,
        )

    def _generate_causal_mask(
        self,
        size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        size = int(size)
        if size <= 0:
            return torch.empty(0, 0, device=device, dtype=dtype)

        mask = self._causal_mask
        needs_rebuild = (
            mask.numel() == 0
            or mask.device != device
            or mask.size(0) < size
            or mask.size(1) < size
        )
        if needs_rebuild:
            mask = torch.triu(
                torch.full(
                    (size, size), float("-inf"), device=device, dtype=torch.float32
                ),
                diagonal=1,
            )
            self._causal_mask = mask
        else:
            mask = mask[:size, :size]

        if mask.dtype != dtype:
            mask = mask.to(dtype=dtype)
        return mask

    # ---- Attention mode routing ------------------------------------------------
    def _get_layer_attention_type(self, layer_idx: int) -> str:
        mode = self.attention_mode
        if mode == "standard":
            return "standard"
        if mode == "linear":
            return "linear"
        if mode == "sype":
            return "sype"
        if mode == "kimi":
            return "kimi"
        if mode in ("hybrid", "hybrid_linear"):
            return "linear" if layer_idx < (self.num_layers - 1) else "standard"
        if mode in ("hybrid_kimi", "kimi_hybrid"):
            return "kimi" if layer_idx < (self.num_layers - 1) else "standard"
        if mode == "kimi_3to1":
            return "kimi" if (layer_idx % 4) < 3 else "standard"
        if mode == "gated_delta":
            return "gated_delta"
        if mode in ("hybrid_gdn", "gdn_hybrid"):
            return "gated_delta" if layer_idx < (self.num_layers - 1) else "standard"
        if mode == "gdn_3to1":
            return "gated_delta" if (layer_idx % 4) < 3 else "standard"

        # GLA (Gated Linear Attention)
        if mode == "gla":
            return "gla"
        if mode in ("gla_hybrid", "hybrid_gla"):
            return "gla" if layer_idx < (self.num_layers - 1) else "standard"
        if mode == "gla_3to1":
            return "gla" if (layer_idx % 4) < 3 else "standard"

        # DeltaNet
        if mode == "deltanet":
            return "deltanet"
        if mode in ("deltanet_hybrid", "hybrid_deltanet"):
            return "deltanet" if layer_idx < (self.num_layers - 1) else "standard"
        if mode == "deltanet_3to1":
            return "deltanet" if (layer_idx % 4) < 3 else "standard"

        # GatedDeltaNet (Mamba-2 style)
        if mode == "gated_deltanet":
            return "gated_deltanet"
        if mode in ("gated_deltanet_hybrid", "hybrid_gated_deltanet"):
            return "gated_deltanet" if layer_idx < (self.num_layers - 1) else "standard"
        if mode == "gated_deltanet_3to1":
            return "gated_deltanet" if (layer_idx % 4) < 3 else "standard"

        raise ValueError(f"Unknown attention_mode: {mode}")

    def _build_layer_kwargs(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        att_type,
        norm_strategy,
        custom_norm,
        layer_norm_eps,
        use_swiglu,
        freq_modes,
        use_moe,
        num_experts,
        top_k,
        use_gateskip,
        gate_budget,
        gate_lambda,
        layer_attention_type,
        # mHC
        use_mhc: bool,
        mhc_n_streams: int,
        mhc_sinkhorn_iters: int,
        mhc_collapse: str,
        use_attention_residual: bool,
        attn_residual_type: str,
        attention_residual_block_size: int,
        pos_encoding_type: str = "sinusoidal",
        rope_base: float = 10000.0,
        rope_scaling_type: str = "none",
        rope_scaling_factor: float = 1.0,
        **kwargs,
    ):
        return {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "activation": activation,
            "att_type": att_type,
            "norm_strategy": norm_strategy,
            "custom_norm": custom_norm,
            "layer_norm_eps": layer_norm_eps,
            "use_swiglu": use_swiglu,
            "freq_modes": freq_modes,
            "use_moe": use_moe,
            "num_experts": num_experts,
            "top_k": top_k,
            "use_gateskip": use_gateskip,
            "gate_budget": gate_budget,
            "gate_lambda": gate_lambda,
            "layer_attention_type": layer_attention_type,
            # mHC
            "use_mhc": use_mhc,
            "mhc_n_streams": mhc_n_streams,
            "mhc_sinkhorn_iters": mhc_sinkhorn_iters,
            "mhc_collapse": mhc_collapse,
            "use_attention_residual": use_attention_residual,
            "attn_residual_type": attn_residual_type,
            "attention_residual_block_size": attention_residual_block_size,
            # Positional encoding
            "pos_encoding_type": pos_encoding_type,
            "rope_base": rope_base,
            "rope_scaling_type": rope_scaling_type,
            "rope_scaling_factor": rope_scaling_factor,
            **kwargs,
        }

    # ---- GateSkip runtime setters ---------------------------------------------
    def _set_layer_gateskip_attrs(self, layer: nn.Module) -> None:
        if hasattr(layer, "use_gateskip"):
            layer.use_gateskip = bool(self.use_gateskip)
        if hasattr(layer, "gate_budget"):
            layer.gate_budget = self.gate_budget
        if hasattr(layer, "gate_lambda"):
            layer.gate_lambda = float(self.gate_lambda)

    def set_use_gateskip(self, flag: bool) -> None:
        self.use_gateskip = bool(flag)
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            self._set_layer_gateskip_attrs(layer)

    def set_gate_budget(self, budget: Optional[float]) -> None:
        self.gate_budget = budget
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            self._set_layer_gateskip_attrs(layer)

    def set_gate_lambda(self, lam: float) -> None:
        self.gate_lambda = float(lam)
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            self._set_layer_gateskip_attrs(layer)

    def set_budget_scheduler(self, scheduler: BudgetScheduler) -> None:
        self.budget_scheduler = scheduler

    # ---- mHC runtime setters (deprecated) --------------------------------------
    # These mutate attributes after construction which can leave already-
    # instantiated attention modules in an inconsistent state.  Configure
    # use_mhc / mhc_n_streams / etc. at construction time instead.

    def _set_layer_mhc_attrs(self, layer: nn.Module) -> None:
        if hasattr(layer, "use_mhc"):
            layer.use_mhc = bool(self.use_mhc)
        if hasattr(layer, "mhc_n_streams"):
            layer.mhc_n_streams = int(self.mhc_n_streams)
        if hasattr(layer, "mhc_sinkhorn_iters"):
            layer.mhc_sinkhorn_iters = int(self.mhc_sinkhorn_iters)
        if hasattr(layer, "mhc_collapse"):
            layer.mhc_collapse = str(self.mhc_collapse)
        if hasattr(layer, "_ensure_mhc_mixers"):
            layer._ensure_mhc_mixers()

    def set_use_mhc(self, flag: bool) -> None:
        warnings.warn(
            "set_use_mhc is deprecated; set use_mhc in TransformerConfig or "
            "the model constructor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.use_mhc = bool(flag)
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            self._set_layer_mhc_attrs(layer)

    def set_mhc_params(
        self,
        n_streams: Optional[int] = None,
        sinkhorn_iters: Optional[int] = None,
        collapse: Optional[str] = None,
    ) -> None:
        warnings.warn(
            "set_mhc_params is deprecated; configure use_mhc, mhc_n_streams, "
            "and mhc_collapse in the model constructor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if n_streams is not None:
            self.mhc_n_streams = int(n_streams)
        if sinkhorn_iters is not None:
            self.mhc_sinkhorn_iters = int(sinkhorn_iters)
        if collapse is not None:
            self.mhc_collapse = str(collapse)
        for i in range(self.num_layers):
            layer = self._get_layer(i)
            self._set_layer_mhc_attrs(layer)

    # ---- Mixture-of-Depths setters --------------------------------------------
    def set_use_mod(self, flag: bool) -> None:
        self.use_mod = bool(flag)

    def set_mod_budget_scheduler(self, scheduler: MoDBudgetScheduler) -> None:
        self.mod_budget_scheduler = scheduler

    # ---- Layer factory ---------------------------------------------------------
    @abstractmethod
    def _make_layer(self, **kwargs) -> nn.Module: ...

    # ---- Init & helpers --------------------------------------------------------
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.initializer_range)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.initializer_range)

    def _apply_depth_scaled_initialization(self) -> None:
        """Scale residual-output projections according to transformer depth."""
        if not self.depth_scaled_init or self.num_layers <= 0:
            return
        residual_std = self.initializer_range / math.sqrt(2.0 * self.num_layers)
        residual_suffixes = (
            "out_proj",
            "o_proj",
            "w3",
            "fc2",
        )
        with torch.no_grad():
            for name, module in self.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                leaf_name = name.rsplit(".", 1)[-1]
                if leaf_name in residual_suffixes or name.endswith("out_proj.2"):
                    nn.init.normal_(module.weight, mean=0.0, std=residual_std)

    def _get_layer(self, idx: int) -> nn.Module:
        return self.shared_layer if self.layers is None else self.layers[idx]

    def _should_print_init_summary(self) -> bool:
        dist = torch.distributed
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True

    def _print_init_summary(
        self,
        *,
        att_type: str,
        custom_norm: str,
        norm_strategy: str,
        use_moe: bool,
        num_experts: int,
        top_k: int,
        share_layers: bool,
        use_final_norm: bool,
    ) -> None:
        if not self._should_print_init_summary():
            return

        per_layer_attn = [
            self._get_layer_attention_type(i) for i in range(self.num_layers)
        ]
        attn_mix = ", ".join(sorted(set(per_layer_attn)))
        print(
            f"[{self.__class__.__name__}] "
            f"att_type={att_type} | "
            f"attention_mode={self.attention_mode} (layers={attn_mix}) | "
            f"norm={custom_norm}/{norm_strategy} final_norm={use_final_norm} | "
            f"moe={use_moe} experts={num_experts} top_k={top_k} moe_aux_lambda={self.moe_aux_lambda} | "
            f"gateskip={self.use_gateskip} budget={self.gate_budget} gate_lambda={self.gate_lambda} | "
            f"attnres={self.use_attention_residual}/{self.attention_residual_mode}"
            f"/block={self.attention_residual_block_size} | "
            f"mhc={self.use_mhc} streams={self.mhc_n_streams} collapse={self.mhc_collapse} | "
            f"patch(enc={self.patch_encoder},len={self.patch_len},stride={self.patch_stride}) | "
            f"mod={self.use_mod}/{self.mod_mode} | "
            f"shared_layers={share_layers} grad_ckpt={self.use_gradient_checkpointing}"
        )

    # FIX: aggregate aux loss over executed layer indices (supports skipping)
    def _aggregate_aux_loss(self, used_indices: List[int]) -> None:
        """Aggregate per-layer aux losses into a single tensor."""
        total_aux: float = 0.0
        device: torch.device | None = None
        for i in used_indices:
            layer = self._get_layer(i)
            if hasattr(layer, "_aux_loss"):
                layer._record_aux_loss_device(next(self.parameters()).device)
                if layer._aux_loss_device is not None:
                    device = layer._aux_loss_device
                total_aux += layer._aux_loss
        denom = max(len(used_indices), 1)
        mod_aux = float(self.mod_aux_loss) if hasattr(self, "mod_aux_loss") else 0.0
        if device is None:
            device = next(self.parameters()).device
        self.aux_loss = torch.tensor(
            ((total_aux / denom) * self.moe_aux_lambda + mod_aux),
            device=device,
        )

    @staticmethod
    def _run_with_checkpoint(fn, *inputs: torch.Tensor, use_checkpoint: bool):
        if not use_checkpoint:
            return fn(*inputs)
        return torch.utils.checkpoint.checkpoint(fn, *inputs, use_reentrant=False)

    def _get_runtime_budget(self) -> Optional[float]:
        if self.training and (self.budget_scheduler is not None):
            return self.budget_scheduler.get_budget()
        return self.gate_budget

    def _validate_attention_residual_runtime(self) -> None:
        if not self.use_attention_residual:
            return
        if self.use_gateskip:
            raise RuntimeError(
                "Paper-style Attention Residuals replace the residual path and are "
                "not compatible with GateSkip in this implementation."
            )
        if self.use_mhc:
            raise RuntimeError(
                "Paper-style Attention Residuals are not wired for mHC stream mixing."
            )
        if self.use_mod:
            raise RuntimeError(
                "Paper-style Attention Residuals are not wired for Mixture-of-Depths."
            )

    def _init_attention_residual_state(self, x: torch.Tensor) -> Optional[dict]:
        if not self.use_attention_residual:
            return None
        return _init_attention_residual_state(
            x, self.attention_residual_mode, self.attention_residual_block_size
        )

    def _finalize_attention_residual_output(
        self, state: Optional[dict], fallback: torch.Tensor
    ) -> torch.Tensor:
        if not self.use_attention_residual or state is None:
            return fallback
        if self.output_attention_residual is None:
            return fallback
        return self.output_attention_residual(_attention_residual_values(state))

    def _validate_mod_runtime(self) -> None:
        if not self.use_mod:
            return
        if self.mod_mode != "token":
            raise RuntimeError(
                "Mixture-of-Depths only supports token routing. Set mod_mode='token'."
            )
        if self.use_gateskip:
            raise RuntimeError(
                "Mixture-of-Depths is not wired together with GateSkip in this implementation."
            )
        if self.use_mhc:
            raise RuntimeError("Mixture-of-Depths is not wired for mHC stream mixing.")

    def _resolve_layer(self, layer_idx: int) -> nn.Module:
        layer = self._get_layer(layer_idx)
        if (self.shared_layer is not None) and hasattr(
            layer, "set_layer_attention_type"
        ):
            layer.set_layer_attention_type(self._get_layer_attention_type(layer_idx))
        return layer

    def _configured_attention_types(self) -> List[str]:
        return sorted(
            {self._get_layer_attention_type(i) for i in range(self.num_layers)}
        )

    def _materialize_configured_attention_backends(self) -> None:
        """
        Build the attention modules required by this model's routing schedule.

        Encoder/decoder layers instantiate their currently selected backend at
        construction time. Full models also call this hook so shared-layer
        schedules can materialize every backend they will route through. A
        parameter created during the first forward would otherwise be invisible
        to the optimizer and omitted from checkpoints created before warmup.
        """
        if self.shared_layer is not None:
            layer = self.shared_layer
            materialize = getattr(layer, "materialize_attention_type", None)
            if callable(materialize):
                for attn_type in self._configured_attention_types():
                    materialize(attn_type)
            return

        for i in range(self.num_layers):
            layer = self._get_layer(i)
            materialize = getattr(layer, "materialize_attention_type", None)
            if callable(materialize):
                materialize(self._get_layer_attention_type(i))

    def _get_layer_keep_rate(self, layer_idx: int) -> float:
        if self.mod_budget_scheduler is None:
            return 1.0
        return float(self.mod_budget_scheduler.get_keep_rate(layer_idx))

    def _prepare_layer_routing(
        self,
        layer_idx: int,
        x: torch.Tensor,
        active_mask: Optional[torch.Tensor],
    ) -> Tuple[
        nn.Module,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        layer = self._resolve_layer(layer_idx)
        if not self.use_mod:
            return layer, None, None, None, None

        router_logits = self.mod_routers[layer_idx](x).squeeze(-1)
        keep_mask = mod_topk_mask(
            router_logits,
            self._get_layer_keep_rate(layer_idx),
            active_mask=active_mask,
        )
        if self.training and self.mod_lambda > 0:
            self.mod_aux_loss += (
                self.mod_lambda
                * float(mod_router_aux_loss(
                    router_logits,
                    keep_mask,
                    active_mask=active_mask,
                ).detach())
            )

        capacity = mod_capacity(keep_mask)
        if capacity <= 0:
            return layer, router_logits, keep_mask, None, None
        indices, slot_mask = mod_routed_indices(keep_mask, capacity=capacity)
        return layer, router_logits, keep_mask, indices, slot_mask

    def _finalize_layer_stack(self, used_indices: List[int]) -> None:
        self._aggregate_aux_loss(used_indices)
        if self.training and self.budget_scheduler is not None:
            self.budget_scheduler.step()
        if self.training and self.mod_budget_scheduler is not None:
            self.mod_budget_scheduler.step()


__all__ = [
    "NormWrapper",
    "ResidualRunCfg",
    "ResidualBlockMixin",
    "MHCBlockMixin",
    "BaseTransformerLayer",
    "BaseTransformer",
]
