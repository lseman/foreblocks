from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from foreblocks.ui.node_spec import node

from .head_state import HeadStateManager
from .head_types import (
    ActiveHead,
    AlphaWeights,
    BaseRunState,
    ComposerMode,
    HeadSpec,
    ParallelAlignMode,
    ParallelCombine,
    ParallelNoneState,
    ParallelStructuredOutputs,
    RunStateList,
    SerialAddState,
    SerialInvertState,
    SerialNoneMerge,
    SerialNoneState,
)
from .projection_registry import ProjectionRegistry


@node(
    type_id="head_composer",
    name="HeadComposer",
    category="Misc",
    outputs=["head_composer"],
    color="bg-gradient-to-r from-purple-400 to-pink-500",
)
class HeadComposer(nn.Module):
    """
    Compose preprocessing heads with configurable serial/parallel policies and optional
    inverse mapping back to the original feature space.

    Head behavior is controlled at two levels:
      1) Per-head via `HeadSpec` (`combine`, `alpha_mode`, `alpha_mix_style`,
         add-carry projection).
      2) Per-stage via composer policies (parallel combine/align/project and serial
         merge behavior for `combine='none'` outputs).

    Alpha/combine rule:
      - `alpha_mode` is orthogonal to `combine` except one restriction:
        `alpha_mode='soft'` is invalid when a head resolves to `combine='invert'`.
      - Therefore:
          * `invert` supports `off` or `gate`
          * `add` supports `off`, `gate`, or `soft`
          * `none` supports `off`, `gate`, or `soft`

    Modes:
      - composer_mode="serial":
          x -> h1 -> h2 -> ... -> x_out
          Uses `HeadSpec.combine` for each head (`invert`/`add`/`none`).
      - composer_mode="parallel":
          y_i = head_i(x0) for each enabled head, then merge branches with
          `parallel_combine`.
          Parallel stage is forward-only (no inverse).
      - composer_mode="hybrid":
          x0 -> [parallel stage on x0] -> x_aug -> [serial stage on x_aug] -> x_out
          Inverse only undoes the serial stage.

    Parallel-stage policies:
      - parallel_combine:
          concat branches or fuse them with:
          concat | sum | mean | weighted_sum | hypernetwork_mix |
          attention_fusion | gated_fusion
      - parallel_align_mode:
          * "strict": reduction/fusion paths require exact branch shape match.
          * "project": auto-align branch time/features via lazy learned projections.
      - parallel_project / parallel_project_dim:
          optional output projection after parallel merge.
      - parallel_structured_outputs:
          controls tuple outputs in parallel stage:
          * "error": reject tuple outputs (legacy strict behavior)
          * "main": use first tensor (e.g., y from (y,state))
          * "main_add_second": when second is tensor, fuse main + second

    Serial-stage policy for heads that resolve to combine="none":
      - serial_none_merge:
          * "replace": use head output as next tensor (legacy behavior).
          * "add": residual add between current tensor and head output.
          * "concat": concatenate current tensor and head output on feature dim.
      - serial_none_project / serial_none_project_dim:
          optional projection for the selected merge policy.

    Inversion behavior:
      - `combine="invert"`: calls `head.invert(...)` in reverse order.
      - `combine="add"`: adds stored carry (optionally projected by `add_project`).
      - `combine="none"` and parallel stage are not directly inverted.

    Notes:
      - Projections are created lazily. If you want all params materialized before
        optimizer creation, call `warmup(example_x, example_y_like=...)`.
      - Lazy projection/fusion modules are managed by a unified `ProjectionRegistry`.
      - NAS alpha mixing applies per head (`alpha_mode`) and is compatible with
        the stage-level policies above.
    """

    _EPS: float = 1e-6

    def __init__(
        self,
        specs: list[HeadSpec] | None = None,
        *,
        # For hybrid, split specs:
        parallel_specs: list[HeadSpec] | None = None,
        serial_specs: list[HeadSpec] | None = None,
        output_dim: int | None = None,
        stop_gradient_on_carry: bool = False,
        alpha_temperature: float = 1.0,
        enable_nas: bool = False,
        composer_mode: ComposerMode = "serial",
        parallel_combine: ParallelCombine = "concat",
        # Parallel stage controls:
        parallel_align_mode: ParallelAlignMode = "strict",
        parallel_project: bool = False,
        parallel_project_dim: int | None = None,
        parallel_structured_outputs: ParallelStructuredOutputs = "error",
        parallel_hyper_hidden_dim: int = 64,
        parallel_attention_heads: int = 0,
        parallel_fusion_dropout: float = 0.0,
        moe_temperature: float = 1.0,
        gumbel_temperature: float = 0.5,
        anneal_alpha: bool = True,
        use_spectral_norm_invert: bool = True,
        # Serial stage controls for combine='none':
        serial_none_merge: SerialNoneMerge = "replace",
        serial_none_project: bool = False,
        serial_none_project_dim: int | None = None,
    ):
        super().__init__()

        # Normalize args
        specs: list[HeadSpec] = specs or []
        parallel_specs = parallel_specs or []
        serial_specs = serial_specs or []

        self.composer_mode: ComposerMode = composer_mode
        self.parallel_combine: ParallelCombine = parallel_combine
        self.parallel_align_mode: ParallelAlignMode = parallel_align_mode
        self.parallel_project = bool(parallel_project)
        self.parallel_project_dim = (
            int(parallel_project_dim) if parallel_project_dim is not None else None
        )
        self.parallel_structured_outputs: ParallelStructuredOutputs = (
            parallel_structured_outputs
        )
        self.parallel_hyper_hidden_dim = int(parallel_hyper_hidden_dim)
        self.parallel_attention_heads = int(parallel_attention_heads)
        self.parallel_fusion_dropout = float(parallel_fusion_dropout)
        self.moe_temperature = float(moe_temperature)
        self.gumbel_temperature = float(gumbel_temperature)
        self.anneal_alpha = bool(anneal_alpha)
        self.use_spectral_norm_invert = bool(use_spectral_norm_invert)
        self._alpha_step = 0
        self.serial_none_merge: SerialNoneMerge = serial_none_merge
        self.serial_none_project = bool(serial_none_project)
        self.serial_none_project_dim = (
            int(serial_none_project_dim)
            if serial_none_project_dim is not None
            else None
        )

        self.fixed_output_dim = output_dim
        self.stop_gradient_on_carry = stop_gradient_on_carry
        self.alpha_temperature = alpha_temperature
        self.enable_nas = enable_nas

        # Choose which specs are active given mode
        self.parallel_meta, self.serial_meta = self._resolve_mode_specs(
            composer_mode=composer_mode,
            specs=specs,
            parallel_specs=parallel_specs,
            serial_specs=serial_specs,
        )

        # Validate unique names across all heads
        self._validate_unique_names(self.parallel_meta, self.serial_meta)
        self._validate_spec_constraints()

        # Register heads as modules (keep ordering stable)
        if self.use_spectral_norm_invert:
            for spec in self.parallel_meta + self.serial_meta:
                if spec.spectral_norm and isinstance(spec.head, nn.Module):
                    try:
                        spec.head = spectral_norm(spec.head)
                    except Exception:
                        pass

        self.parallel_heads = nn.ModuleList([s.head for s in self.parallel_meta])
        self.serial_heads = nn.ModuleList([s.head for s in self.serial_meta])
        self.parallel_active_heads = self._build_active_heads(
            self.parallel_meta, is_parallel=True
        )
        self.serial_active_heads = self._build_active_heads(
            self.serial_meta, is_parallel=False
        )
        self.active_heads: list[ActiveHead] = (
            self.parallel_active_heads + self.serial_active_heads
        )

        for head in self.active_heads:
            head.fusion_scale_param = nn.Parameter(
                torch.tensor(float(head.spec.fusion_scale_init), dtype=torch.float32)
            )

        self.projection_registry = ProjectionRegistry()
        for spec in self.serial_meta:
            if spec.combine == "add" and spec.custom_add_proj is not None:
                self.projection_registry.register_custom(
                    category="add_carry_custom",
                    name=spec.name,
                    module=spec.custom_add_proj,
                )
        self.parallel_mix_logits = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, dtype=torch.float32))
                for _ in self.parallel_meta
            ]
        )
        for i, head in enumerate(self.parallel_active_heads):
            head.mix_logit = self.parallel_mix_logits[i]

        self.state_manager = HeadStateManager(
            self.active_heads,
            enable_nas=self.enable_nas,
            alpha_temperature=self.alpha_temperature,
            gumbel_temperature=self.gumbel_temperature,
        )

        # Used for deterministic projection sizing when output_dim is None
        self._last_base_dim: int | None = None

        # Validate mode constraints early
        self._validate_mode_constraints()

    # ──────────────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_mode_specs(
        *,
        composer_mode: str,
        specs: list[HeadSpec],
        parallel_specs: list[HeadSpec],
        serial_specs: list[HeadSpec],
    ) -> tuple[list[HeadSpec], list[HeadSpec]]:
        if composer_mode == "serial":
            if parallel_specs or serial_specs:
                raise ValueError(
                    "In composer_mode='serial', pass specs=... only (not parallel_specs/serial_specs)."
                )
            return [], list(specs)

        if composer_mode == "parallel":
            if specs and parallel_specs:
                raise ValueError(
                    "In composer_mode='parallel', pass exactly one of specs=... or parallel_specs=..., not both."
                )
            if serial_specs:
                raise ValueError(
                    "In composer_mode='parallel', pass parallel_specs=... or specs=..., not serial_specs."
                )
            return list(parallel_specs) if parallel_specs else list(specs), []

        if composer_mode == "hybrid":
            if specs:
                raise ValueError(
                    "In composer_mode='hybrid', pass parallel_specs=... and serial_specs=... (not specs)."
                )
            return list(parallel_specs), list(serial_specs)

        raise ValueError(f"Unknown composer_mode: {composer_mode}")

    @staticmethod
    def _build_active_heads(
        specs: list[HeadSpec], *, is_parallel: bool
    ) -> list[ActiveHead]:
        return [
            ActiveHead(spec=spec, index=index, is_parallel=is_parallel)
            for index, spec in enumerate(specs)
        ]

    @staticmethod
    def _validate_unique_names(
        parallel_specs: list[HeadSpec], serial_specs: list[HeadSpec]
    ) -> None:
        seen: set[str] = set()
        for s in parallel_specs + serial_specs:
            if s.name in seen:
                raise ValueError(
                    f"HeadSpec names must be unique across all heads; duplicated: '{s.name}'"
                )
            seen.add(s.name)

    def _validate_mode_constraints(self) -> None:
        valid_parallel_combine = (
            "concat",
            "sum",
            "mean",
            "weighted_sum",
            "hypernetwork_mix",
            "attention_fusion",
            "gated_fusion",
            "lora_mix",
            "moe_routing",
            "multi_scale_attn",
        )
        if self.parallel_combine not in valid_parallel_combine:
            raise ValueError(
                f"parallel_combine must be one of {valid_parallel_combine}, got {self.parallel_combine}"
            )

        if self.parallel_align_mode not in ("strict", "project"):
            raise ValueError(
                "parallel_align_mode must be one of ('strict','project'), "
                f"got {self.parallel_align_mode}"
            )
        if self.parallel_structured_outputs not in ("error", "main", "main_add_second"):
            raise ValueError(
                "parallel_structured_outputs must be one of ('error','main','main_add_second'), "
                f"got {self.parallel_structured_outputs}"
            )
        if self.serial_none_merge not in ("replace", "add", "concat", "lora_residual"):
            raise ValueError(
                "serial_none_merge must be one of ('replace','add','concat','lora_residual'), "
                f"got {self.serial_none_merge}"
            )
        if self.parallel_project_dim is not None and self.parallel_project_dim <= 0:
            raise ValueError(
                f"parallel_project_dim must be > 0, got {self.parallel_project_dim}"
            )
        if self.parallel_hyper_hidden_dim <= 0:
            raise ValueError(
                f"parallel_hyper_hidden_dim must be > 0, got {self.parallel_hyper_hidden_dim}"
            )
        if self.parallel_attention_heads < 0:
            raise ValueError(
                f"parallel_attention_heads must be >= 0, got {self.parallel_attention_heads}"
            )
        if self.parallel_fusion_dropout < 0.0 or self.parallel_fusion_dropout >= 1.0:
            raise ValueError(
                "parallel_fusion_dropout must be in [0,1), "
                f"got {self.parallel_fusion_dropout}"
            )
        if (
            self.serial_none_project_dim is not None
            and self.serial_none_project_dim <= 0
        ):
            raise ValueError(
                f"serial_none_project_dim must be > 0, got {self.serial_none_project_dim}"
            )

        # Parallel stage constraints: tuple outputs are only allowed when policy explicitly enables them.
        if self.composer_mode in ("parallel", "hybrid"):
            bad: list[str] = []
            if self.parallel_structured_outputs == "error":
                for s in self.parallel_meta:
                    if s.combine in ("invert", "add"):
                        bad.append(s.name)
            if bad:
                raise ValueError(
                    "Parallel stage rejects combine='invert'/'add' when parallel_structured_outputs='error'. "
                    "Set parallel_structured_outputs='main' or 'main_add_second' to allow tuple heads in parallel. "
                    f"Offending: {bad}"
                )

    def _validate_spec_constraints(self) -> None:
        for spec in self.parallel_meta + self.serial_meta:
            if spec.combine == "invert" and spec.alpha_mode == "soft":
                raise ValueError(
                    f"HeadSpec '{spec.name}': alpha_mode='soft' is invalid when combine='invert'. "
                    "Use alpha_mode='gate' or alpha_mode='off'."
                )

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _is_2tuple(obj: Any) -> bool:
        return isinstance(obj, (tuple, list)) and len(obj) == 2

    @staticmethod
    def _both_tensors(obj: Any) -> bool:
        return (
            isinstance(obj, (tuple, list))
            and len(obj) == 2
            and all(isinstance(t, torch.Tensor) for t in obj)
        )

    def _get_or_build_projection(
        self,
        *,
        category: str,
        name: str,
        in_dim: int,
        out_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        proj_type: str = "linear",
        registered_category: str | None = None,
        **kwargs: Any,
    ) -> nn.Module:
        if registered_category is not None:
            custom = self.projection_registry.get_registered(
                category=registered_category, name=name
            )
            if custom is not None:
                return custom.to(device=device, dtype=dtype)
        return self.projection_registry.get_or_create(
            category=category,
            name=name,
            in_dim=int(in_dim),
            out_dim=int(out_dim),
            proj_type=proj_type,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    @staticmethod
    def _align_time(x: torch.Tensor, target_T: int) -> torch.Tensor:
        B, T, F_ = x.shape
        if T == target_T:
            return x
        if T > target_T:
            return x[:, -target_T:, :]
        pad_len = target_T - T
        last = x[:, -1:, :].expand(B, pad_len, F_)
        return torch.cat([x, last], dim=1)

    @staticmethod
    def _ensure_same_shape_for_mixing(
        name: str,
        head_out: torch.Tensor,
        skip: torch.Tensor,
        context: str,
    ) -> None:
        if head_out.shape != skip.shape:
            raise RuntimeError(
                f"[{name}] {context} requires head output and skip tensor to have the same shape. "
                f"Got head={tuple(head_out.shape)} vs skip={tuple(skip.shape)}."
            )

    def _serial_none_target_dim(self, cur: torch.Tensor) -> int:
        if self.serial_none_project_dim is not None:
            return int(self.serial_none_project_dim)
        return int(cur.size(-1))

    def _merge_serial_none(
        self,
        *,
        cur: torch.Tensor,
        y: torch.Tensor,
        head_name: str,
        rec: SerialNoneState,
        spec: HeadSpec,
    ) -> torch.Tensor:
        mode = self.serial_none_merge
        if spec.combine == "lora_residual":
            mode = "lora_residual"
        rec.serial_none_merge = mode

        if mode == "replace":
            if self.serial_none_project or self.serial_none_project_dim is not None:
                target_dim = self._serial_none_target_dim(cur)
                proj = self._get_or_build_projection(
                    category="stage_proj",
                    name=f"serial_none_replace__{head_name}",
                    in_dim=int(y.size(-1)),
                    out_dim=target_dim,
                    device=cur.device,
                    dtype=cur.dtype,
                    allow_identity=True,
                )
                y = proj(y)
                rec.serial_none_project_dim = target_dim
            return y

        if y.size(0) != cur.size(0):
            raise RuntimeError(
                f"[{head_name}] serial_none_merge='{mode}' requires same batch size, "
                f"got y={tuple(y.shape)} vs cur={tuple(cur.shape)}"
            )

        if y.size(1) != cur.size(1):
            y = self._align_time(y, cur.size(1))
            rec.serial_none_time_aligned = True

        if mode == "add":
            if self.serial_none_project or self.serial_none_project_dim is not None:
                target_dim = self._serial_none_target_dim(cur)
                proj_y = self._get_or_build_projection(
                    category="stage_proj",
                    name=f"serial_none_add_head__{head_name}",
                    in_dim=int(y.size(-1)),
                    out_dim=target_dim,
                    device=cur.device,
                    dtype=cur.dtype,
                    allow_identity=True,
                )
                y = proj_y(y)
                if cur.size(-1) != target_dim:
                    proj_cur = self._get_or_build_projection(
                        category="stage_proj",
                        name=f"serial_none_add_skip__{head_name}",
                        in_dim=int(cur.size(-1)),
                        out_dim=target_dim,
                        device=cur.device,
                        dtype=cur.dtype,
                        allow_identity=True,
                    )
                    cur = proj_cur(cur)
                rec.serial_none_project_dim = target_dim
            else:
                self._ensure_same_shape_for_mixing(
                    head_name, y, cur, context="serial_none_merge='add'"
                )
            return cur + y

        if mode == "concat":
            merged = torch.cat([cur, y], dim=-1)
            if self.serial_none_project or self.serial_none_project_dim is not None:
                target_dim = self._serial_none_target_dim(cur)
                proj = self._get_or_build_projection(
                    category="stage_proj",
                    name=f"serial_none_concat__{head_name}",
                    in_dim=int(merged.size(-1)),
                    out_dim=target_dim,
                    device=cur.device,
                    dtype=cur.dtype,
                    allow_identity=True,
                )
                merged = proj(merged)
                rec.serial_none_project_dim = target_dim
            return merged

        if mode == "lora_residual":
            if spec.lora_rank is None:
                raise ValueError(
                    f"[{head_name}] serial_none_merge='lora_residual' requires spec.lora_rank"
                )
            delta = y - cur
            adapter = self._get_or_build_projection(
                category="lora_adapter",
                name=f"lora_residual_{head_name}",
                in_dim=int(delta.size(-1)),
                out_dim=int(delta.size(-1)),
                device=cur.device,
                dtype=cur.dtype,
                proj_type="lora_adapter",
                rank=int(spec.lora_rank),
            )
            return cur + adapter(delta)

        raise ValueError(f"Unknown serial_none_merge mode: {mode}")

    def _parallel_target_dim(self, x_ref: torch.Tensor) -> int:
        if self.parallel_project_dim is not None:
            return int(self.parallel_project_dim)
        return int(x_ref.size(-1))

    def _extract_parallel_tensor(self, spec: HeadSpec, out: Any) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            return out

        if not self._is_2tuple(out) or not isinstance(out[0], torch.Tensor):
            raise RuntimeError(
                f"[{spec.name}] parallel stage requires Tensor or 2-tuple with Tensor first element, "
                f"got {type(out).__name__}"
            )

        main = out[0]
        second = out[1]

        if self.parallel_structured_outputs == "error":
            raise RuntimeError(
                f"[{spec.name}] parallel stage received structured output but "
                "parallel_structured_outputs='error'. "
                "Set parallel_structured_outputs='main' or 'main_add_second'."
            )

        if self.parallel_structured_outputs == "main":
            return main

        if self.parallel_structured_outputs == "main_add_second":
            if not isinstance(second, torch.Tensor):
                return main

            extra = second
            if extra.size(0) != main.size(0):
                raise RuntimeError(
                    f"[{spec.name}] parallel_structured_outputs='main_add_second' requires same batch size. "
                    f"main={tuple(main.shape)} second={tuple(extra.shape)}"
                )
            if extra.size(1) != main.size(1):
                extra = self._align_time(extra, main.size(1))
            if extra.size(-1) != main.size(-1):
                proj = self._get_or_build_projection(
                    category="stage_proj",
                    name=f"parallel_structured_second__{spec.name}",
                    in_dim=int(extra.size(-1)),
                    out_dim=int(main.size(-1)),
                    device=main.device,
                    dtype=main.dtype,
                    allow_identity=True,
                )
                extra = proj(extra)
            return main + extra

        raise ValueError(
            f"Unknown parallel_structured_outputs mode: {self.parallel_structured_outputs}"
        )

    def _align_parallel_branches_for_fusion(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x_ref: torch.Tensor,
    ) -> list[tuple[ActiveHead, torch.Tensor]]:
        if not branches:
            return []

        if self.parallel_align_mode == "strict":
            base = branches[0][1]
            for i, (_, b) in enumerate(branches[1:], start=1):
                if b.shape != base.shape:
                    raise RuntimeError(
                        f"parallel_combine='{self.parallel_combine}' requires equal branch shapes in "
                        "parallel_align_mode='strict'. "
                        f"branch0={tuple(base.shape)}, branch{i}={tuple(b.shape)}. "
                        "Set parallel_align_mode='project' to auto-align."
                    )
            return branches

        target_dim = self._parallel_target_dim(x_ref)
        target_T = int(x_ref.size(1))
        aligned: list[tuple[ActiveHead, torch.Tensor]] = []
        for idx, (head, b) in enumerate(branches):
            if b.size(0) != x_ref.size(0):
                raise RuntimeError(
                    f"[{head.name}] parallel branch batch mismatch: "
                    f"branch={tuple(b.shape)} vs input={tuple(x_ref.shape)}"
                )
            if b.size(1) != target_T:
                b = self._align_time(b, target_T)
            proj = self._get_or_build_projection(
                category="stage_proj",
                name=f"parallel_align__{head.name}__{idx}",
                in_dim=int(b.size(-1)),
                out_dim=target_dim,
                device=x_ref.device,
                dtype=x_ref.dtype,
                allow_identity=True,
            )
            b = proj(b)
            aligned.append((head, b))
        return aligned

    def _get_or_build_parallel_hypernet(
        self,
        in_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        return self.projection_registry.get_or_create(
            category="parallel_hypernet",
            name=f"in{int(in_dim)}",
            in_dim=int(in_dim),
            out_dim=max(1, len(self.parallel_meta)),
            proj_type="hyper_mlp",
            hidden_dim=max(8, int(self.parallel_hyper_hidden_dim)),
            out_features=max(1, len(self.parallel_meta)),
            dropout=float(self.parallel_fusion_dropout),
            device=device,
            dtype=dtype,
        )

    def _get_or_build_parallel_gate_net(
        self,
        in_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        return self.projection_registry.get_or_create(
            category="parallel_gate",
            name=f"in{int(in_dim)}",
            in_dim=int(in_dim),
            out_dim=1,
            proj_type="gate_mlp",
            hidden_dim=max(8, int(in_dim) // 2),
            out_features=1,
            device=device,
            dtype=dtype,
        )

    def _get_or_build_moe_gate_net(
        self,
        in_dim: int,
        out_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        return self.projection_registry.get_or_create(
            category="parallel_moe_gate",
            name=f"in{int(in_dim)}_out{int(out_dim)}",
            in_dim=int(in_dim),
            out_dim=int(out_dim),
            proj_type="gate_mlp",
            hidden_dim=max(8, int(in_dim) // 2),
            out_features=int(out_dim),
            device=device,
            dtype=dtype,
        )

    def _resolve_parallel_attention_heads(self, embed_dim: int) -> int:
        if self.parallel_attention_heads > 0:
            if embed_dim % self.parallel_attention_heads != 0:
                raise ValueError(
                    f"parallel_attention_heads={self.parallel_attention_heads} does not divide "
                    f"embed_dim={embed_dim}"
                )
            return int(self.parallel_attention_heads)

        for h in (8, 4, 2, 1):
            if h <= embed_dim and embed_dim % h == 0:
                return h
        return 1

    def _get_or_build_parallel_attention(
        self,
        embed_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.MultiheadAttention:
        num_heads = self._resolve_parallel_attention_heads(embed_dim)
        mod = self.projection_registry.get_or_create(
            category="parallel_attention",
            name=f"embed{int(embed_dim)}",
            in_dim=int(embed_dim),
            out_dim=int(embed_dim),
            proj_type="multihead_attention",
            num_heads=int(num_heads),
            dropout=float(self.parallel_fusion_dropout),
            batch_first=True,
            device=device,
            dtype=dtype,
        )
        if not isinstance(mod, nn.MultiheadAttention):
            raise TypeError("ProjectionRegistry returned non-attention module")
        return mod

    # ──────────────────────────────────────────────────────────────────────────
    # α mixing primitives
    # ──────────────────────────────────────────────────────────────────────────
    def _alpha_weights_for_head(self, head: ActiveHead) -> AlphaWeights:
        return self.state_manager.alpha_weights_for_head(head)

    @staticmethod
    def _straight_through_hard_gate(g: torch.Tensor) -> torch.Tensor:
        return HeadStateManager.straight_through_hard_gate(g)

    def _compute_effective_weight(
        self,
        spec: HeadSpec,
        w_head: torch.Tensor | None,
        w_skip: torch.Tensor | None,
        hardened: bool | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.state_manager.compute_effective_weight(
            spec=spec,
            w_head=w_head,
            w_skip=w_skip,
            hardened=hardened,
            device=device,
            dtype=dtype,
        )

    def _mix_head_skip(
        self,
        *,
        spec: HeadSpec,
        head_out: torch.Tensor,
        skip: torch.Tensor,
        eff_head: torch.Tensor,
        eff_skip: torch.Tensor,
    ) -> torch.Tensor:
        if spec.alpha_mix_style == "blend":
            return head_out * eff_head + skip * eff_skip
        if spec.alpha_mix_style == "residual":
            return skip + eff_head * (head_out - skip)
        if spec.alpha_mix_style == "lora":
            if spec.lora_rank is None:
                raise ValueError(
                    f"[{spec.name}] alpha_mix_style='lora' requires spec.lora_rank to be set"
                )
            delta = head_out - skip
            adapter = self._get_or_build_projection(
                category="lora_adapter",
                name=f"lora_{spec.name}",
                in_dim=int(delta.size(-1)),
                out_dim=int(delta.size(-1)),
                device=delta.device,
                dtype=delta.dtype,
                proj_type="lora_adapter",
                rank=int(spec.lora_rank),
            )
            return skip + eff_head * adapter(delta)
        raise ValueError(
            f"[{spec.name}] unknown alpha_mix_style '{spec.alpha_mix_style}'"
        )

    def _mix_with_alpha(
        self,
        *,
        spec: HeadSpec,
        head_out: torch.Tensor,
        skip: torch.Tensor,
        w_head: torch.Tensor | None,
        w_skip: torch.Tensor | None,
        hardened: bool | None,
        device: torch.device,
        dtype: torch.dtype,
        context: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if w_head is None:
            return head_out, None

        self._ensure_same_shape_for_mixing(spec.name, head_out, skip, context=context)
        eff_head, eff_skip = self._compute_effective_weight(
            spec, w_head, w_skip, hardened, device, dtype
        )
        mixed = self._mix_head_skip(
            spec=spec,
            head_out=head_out,
            skip=skip,
            eff_head=eff_head,
            eff_skip=eff_skip,
        )
        return mixed, eff_head

    # ──────────────────────────────────────────────────────────────────────────
    # Ablation-friendly enable/disable API
    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def enable_head(self, name: str) -> None:
        self.state_manager.enable_head(name)

    @torch.no_grad()
    def disable_head(self, name: str) -> None:
        self.state_manager.disable_head(name)

    @torch.no_grad()
    def set_head_enabled(self, name: str, enabled: bool) -> None:
        self.state_manager.set_head_enabled(name, enabled)

    @torch.no_grad()
    def enable_all(self) -> None:
        self.state_manager.enable_all()

    @torch.no_grad()
    def disable_all(self) -> None:
        self.state_manager.disable_all()

    def enabled_report(self) -> dict[str, bool]:
        return self.state_manager.enabled_report()

    @contextmanager
    def temporary_disable(self, *names: str) -> Iterator[None]:
        with self.state_manager.temporary_disable(*names):
            yield

    @contextmanager
    def only_heads(self, *names: str) -> Iterator[None]:
        with self.state_manager.only_heads(*names):
            yield

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────
    def forward_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, RunStateList]:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B,T,F], got {tuple(x.shape)}")

        self._last_base_dim = int(x.size(-1))
        run_state: RunStateList = []
        cur = x

        if self.composer_mode in ("parallel", "hybrid"):
            cur, parallel_state = self._forward_parallel_stage_impl(cur)
            run_state.extend(parallel_state)

        if self.composer_mode in ("serial", "hybrid"):
            cur, serial_state = self._forward_serial_stage_impl(cur)
            run_state.extend(serial_state)

        return cur, run_state

    def forward(
        self,
        x: torch.Tensor,
        encoder: nn.Module | None = None,
    ) -> tuple[torch.Tensor, RunStateList]:
        out, state = self.forward_pre(x)
        if encoder is not None:
            out = encoder(out)
        return out, state

    # ──────────────────────────────────────────────────────────────────────────
    # Parallel stage (feature augmentation)
    # ──────────────────────────────────────────────────────────────────────────
    def _forward_parallel_stage_impl(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, RunStateList]:
        branches, run_state = self._collect_parallel_branches(x)

        if not branches:
            return x, run_state

        out = self._combine_parallel_branches(branches, x)

        if self.parallel_project or self.parallel_project_dim is not None:
            target_dim = self._parallel_target_dim(x)
            proj = self._get_or_build_projection(
                category="stage_proj",
                name="parallel_out",
                in_dim=int(out.size(-1)),
                out_dim=target_dim,
                device=x.device,
                dtype=x.dtype,
                allow_identity=True,
            )
            out = proj(out)

        return out, run_state

    def _collect_parallel_branches(
        self, x: torch.Tensor
    ) -> tuple[list[tuple[ActiveHead, torch.Tensor]], RunStateList]:
        run_state: RunStateList = []
        branches: list[tuple[ActiveHead, torch.Tensor]] = []

        for active_head in self.parallel_active_heads:
            spec = active_head.spec
            if not active_head.enabled:
                run_state.append(
                    BaseRunState(
                        kind="disabled",
                        name=active_head.name,
                        stage="parallel",
                    )
                )
                continue

            out = spec.head(x)
            y = self._extract_parallel_tensor(spec, out)
            w_head, w_skip = self._alpha_weights_for_head(active_head)

            rec = ParallelNoneState(
                name=active_head.name,
                stage="parallel",
                base_dim=self._last_base_dim,
            )
            y, eff_head = self._mix_with_alpha(
                spec=spec,
                head_out=y,
                skip=x,
                w_head=w_head,
                w_skip=w_skip,
                hardened=active_head.hardened,
                device=x.device,
                dtype=x.dtype,
                context="parallel alpha mixing",
            )
            if active_head.fusion_scale_param is not None:
                y = y * torch.sigmoid(active_head.fusion_scale_param)
            rec.mix_w_head = None if eff_head is None else float(eff_head.item())

            branches.append((active_head, y))
            run_state.append(rec)

        return branches, run_state

    def _combine_parallel_branches(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        handlers = {
            "concat": self._combine_parallel_concat,
            "sum": self._combine_parallel_sum,
            "mean": self._combine_parallel_mean,
            "weighted_sum": self._combine_parallel_weighted_sum,
            "hypernetwork_mix": self._combine_parallel_hypernetwork_mix,
            "attention_fusion": self._combine_parallel_attention_fusion,
            "gated_fusion": self._combine_parallel_gated_fusion,
            "lora_mix": self._combine_parallel_lora_mix,
            "moe_routing": self._combine_parallel_moe_routing,
            "multi_scale_attn": self._combine_parallel_multi_scale_attn,
        }
        try:
            return handlers[self.parallel_combine](branches, x)
        except KeyError as exc:
            raise ValueError(
                f"Unknown parallel_combine: {self.parallel_combine}"
            ) from exc

    @staticmethod
    def _combine_parallel_concat(
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        del x
        return torch.cat([branch for _, branch in branches], dim=-1)

    def _combine_parallel_sum(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        aligned = self._align_parallel_branches_for_fusion(branches, x)
        return torch.stack([branch for _, branch in aligned], dim=0).sum(dim=0)

    def _combine_parallel_mean(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        aligned = self._align_parallel_branches_for_fusion(branches, x)
        branch_tensors = [branch for _, branch in aligned]
        return torch.stack(branch_tensors, dim=0).mean(dim=0)

    def _combine_parallel_weighted_sum(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        aligned = self._align_parallel_branches_for_fusion(branches, x)
        if any(head.mix_logit is None for head, _ in aligned):
            raise RuntimeError("Missing parallel mix logits for weighted_sum.")

        logits = torch.stack(
            [head.mix_logit[0] for head, _ in aligned],
            dim=0,
        ).to(device=x.device)
        weights = F.softmax(logits.float(), dim=0).to(dtype=aligned[0][1].dtype)
        out = torch.zeros_like(aligned[0][1])
        for i, (_, branch) in enumerate(aligned):
            out = out + branch * weights[i]
        return out

    def _combine_parallel_hypernetwork_mix(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        aligned = self._align_parallel_branches_for_fusion(branches, x)
        branch_stack = torch.stack([branch for _, branch in aligned], dim=0)
        context = branch_stack.mean(dim=0).mean(dim=1)
        hyper = self._get_or_build_parallel_hypernet(
            in_dim=int(context.size(-1)),
            device=x.device,
            dtype=x.dtype,
        )
        logits_all = hyper(context)
        active_idx = torch.tensor(
            [head.index for head, _ in aligned],
            device=x.device,
            dtype=torch.long,
        )
        logits = logits_all.index_select(1, active_idx)
        weights = F.softmax(logits.float(), dim=1).to(dtype=aligned[0][1].dtype)
        out = torch.zeros_like(aligned[0][1])
        for i, (_, branch) in enumerate(aligned):
            out = out + branch * weights[:, i].view(-1, 1, 1)
        return out

    def _combine_parallel_attention(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        aligned = self._align_parallel_branches_for_fusion(branches, x)
        batch_size, time_steps, feature_dim = aligned[0][1].shape
        num_branches = len(aligned)
        attn = self._get_or_build_parallel_attention(
            embed_dim=feature_dim,
            device=x.device,
            dtype=x.dtype,
        )
        stacked = torch.stack([branch for _, branch in aligned], dim=2)
        tokens = stacked.reshape(batch_size * time_steps, num_branches, feature_dim)
        attn_out, _ = attn(tokens, tokens, tokens, need_weights=False)
        return attn_out.mean(dim=1).reshape(batch_size, time_steps, feature_dim)

    def _combine_parallel_attention_fusion(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self._combine_parallel_attention(branches, x)

    def _combine_parallel_multi_scale_attn(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self._combine_parallel_attention(branches, x)

    def _combine_parallel_lora_mix(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        aligned = self._align_parallel_branches_for_fusion(branches, x)
        out = torch.zeros_like(aligned[0][1])
        for head, branch in aligned:
            rank = head.spec.lora_rank or max(1, int(branch.size(-1) // 4))
            adapter = self._get_or_build_projection(
                category="lora_adapter",
                name=f"lora_mix_{head.name}",
                in_dim=int(branch.size(-1)),
                out_dim=int(branch.size(-1)),
                device=branch.device,
                dtype=branch.dtype,
                proj_type="lora_adapter",
                rank=rank,
            )
            out = out + adapter(branch)
        return out

    def _combine_parallel_moe_routing(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        aligned = self._align_parallel_branches_for_fusion(branches, x)
        context = x.mean(dim=1)
        moe_net = self._get_or_build_moe_gate_net(
            in_dim=int(context.size(-1)),
            out_dim=len(aligned),
            device=x.device,
            dtype=x.dtype,
        )
        logits = moe_net(context)
        weights = F.softmax(logits / self.moe_temperature, dim=-1)
        topk = min(max(1, min(head.spec.moe_k for head, _ in aligned)), len(aligned))
        _, topk_idx = torch.topk(weights, k=topk, dim=-1)
        mask = torch.zeros_like(weights).scatter_(1, topk_idx, 1.0)
        weights = weights * mask
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self._EPS)
        out = torch.zeros_like(aligned[0][1])
        for i, (_, branch) in enumerate(aligned):
            out = out + branch * weights[:, i].view(-1, 1, 1).to(dtype=branch.dtype)
        return out

    def _combine_parallel_gated_fusion(
        self,
        branches: list[tuple[ActiveHead, torch.Tensor]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        aligned = self._align_parallel_branches_for_fusion(branches, x)
        feature_dim = aligned[0][1].size(-1)
        gate_net = self._get_or_build_parallel_gate_net(
            in_dim=int(feature_dim),
            device=x.device,
            dtype=x.dtype,
        )
        gate_scores = torch.stack(
            [torch.sigmoid(gate_net(branch.mean(dim=1))) for _, branch in aligned],
            dim=1,
        )
        gate_norm = gate_scores / gate_scores.sum(dim=1, keepdim=True).clamp_min(
            self._EPS
        )
        out = torch.zeros_like(aligned[0][1])
        for i, (_, branch) in enumerate(aligned):
            out = out + branch * gate_norm[:, i, :].unsqueeze(1).to(dtype=branch.dtype)
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # Serial stage (reversible)
    # ──────────────────────────────────────────────────────────────────────────
    def _forward_serial_stage_impl(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, RunStateList]:
        run_state: RunStateList = []
        cur = x

        for active_head in self.serial_active_heads:
            cur, rec = self._apply_serial_head(cur, active_head)
            run_state.append(rec)

        return cur, run_state

    def _apply_serial_head(
        self, cur: torch.Tensor, active_head: ActiveHead
    ) -> tuple[torch.Tensor, BaseRunState]:
        spec = active_head.spec
        if not active_head.enabled:
            return (
                cur,
                BaseRunState(
                    kind="disabled",
                    name=active_head.name,
                    stage="serial",
                ),
            )

        out = spec.head(cur)
        w_head, w_skip = self._alpha_weights_for_head(active_head)
        handlers = {
            "invert": self._apply_serial_invert,
            "add": self._apply_serial_add,
            "none": self._apply_serial_none,
            "lora_residual": self._apply_serial_none,
        }
        try:
            handler = handlers[spec.combine]
        except KeyError as exc:
            raise ValueError(f"Unknown combine mode: '{spec.combine}'") from exc

        return handler(
            cur=cur,
            out=out,
            spec=spec,
            w_head=w_head,
            w_skip=w_skip,
            hardened=active_head.hardened,
            name=active_head.name,
        )

    def _apply_serial_invert(
        self,
        *,
        cur: torch.Tensor,
        out: Any,
        spec: HeadSpec,
        w_head: torch.Tensor | None,
        w_skip: torch.Tensor | None,
        hardened: bool | None,
        name: str,
    ) -> tuple[torch.Tensor, SerialInvertState]:
        del w_skip
        rec = SerialInvertState(
            name=name,
            stage="serial",
            base_dim=self._last_base_dim,
        )
        return self._apply_invert(cur, out, spec, w_head, hardened, rec)

    def _apply_serial_add(
        self,
        *,
        cur: torch.Tensor,
        out: Any,
        spec: HeadSpec,
        w_head: torch.Tensor | None,
        w_skip: torch.Tensor | None,
        hardened: bool | None,
        name: str,
    ) -> tuple[torch.Tensor, SerialAddState]:
        rec = SerialAddState(
            name=name,
            stage="serial",
            base_dim=self._last_base_dim,
        )
        return self._apply_add(cur, out, spec, w_head, w_skip, hardened, rec)

    def _apply_serial_none(
        self,
        *,
        cur: torch.Tensor,
        out: Any,
        spec: HeadSpec,
        w_head: torch.Tensor | None,
        w_skip: torch.Tensor | None,
        hardened: bool | None,
        name: str,
    ) -> tuple[torch.Tensor, SerialNoneState]:
        rec = SerialNoneState(
            name=name,
            stage="serial",
            base_dim=self._last_base_dim,
        )
        return self._apply_none(cur, out, spec, w_head, w_skip, hardened, rec)

    def _apply_invert(
        self,
        cur: torch.Tensor,
        out: Any,
        spec: HeadSpec,
        w_head: torch.Tensor | None,
        hardened: bool | None,
        rec: SerialInvertState,
    ) -> tuple[torch.Tensor, SerialInvertState]:
        if not self._is_2tuple(out) or not isinstance(out[0], torch.Tensor):
            raise RuntimeError(
                f"[{spec.name}] expected (y: Tensor, state: Any) for 'invert', got {type(out).__name__}"
            )
        y, state = out[0], out[1]

        if w_head is None:
            rec.state = state
            rec.head_ref = spec.head
            rec.gate_on = True
            return y, rec

        if spec.alpha_mode == "soft":
            raise ValueError(
                f"[{spec.name}] alpha_mode='soft' invalid for combine='invert' "
                "(use 'gate' or 'off')."
            )

        cur2, eff_head = self._mix_with_alpha(
            spec=spec,
            head_out=y,
            skip=cur,
            w_head=w_head,
            w_skip=None,
            hardened=hardened,
            device=cur.device,
            dtype=cur.dtype,
            context="invert gating",
        )
        if eff_head is None:
            raise RuntimeError(f"[{spec.name}] invert gate unexpectedly missing.")

        rec.state = state
        rec.head_ref = spec.head
        rec.gate_on = bool((eff_head > 0.5).item())
        rec.gate_value = float(w_head.item()) if torch.is_tensor(w_head) else None
        return cur2, rec

    def _apply_add(
        self,
        cur: torch.Tensor,
        out: Any,
        spec: HeadSpec,
        w_head: torch.Tensor | None,
        w_skip: torch.Tensor | None,
        hardened: bool | None,
        rec: SerialAddState,
    ) -> tuple[torch.Tensor, SerialAddState]:
        if not self._both_tensors(out):
            raise RuntimeError(
                f"[{spec.name}] expected (main: Tensor, carry: Tensor) for 'add', got {type(out).__name__}"
            )
        main, carry = out
        rec.add_project = bool(spec.add_project)

        if self.stop_gradient_on_carry:
            carry = carry.detach()

        rec.carry_shape = tuple(carry.shape)

        # Build projection NOW (best-effort), not inside inverse.
        if spec.add_project:
            target_dim = (
                int(self.fixed_output_dim)
                if self.fixed_output_dim is not None
                else int(rec.base_dim)
            )
            _ = self._get_or_build_projection(
                category="add_carry",
                name=spec.name,
                in_dim=int(carry.size(-1)),
                out_dim=target_dim,
                device=cur.device,
                dtype=cur.dtype,
                allow_identity=True,
                registered_category="add_carry_custom",
            )

        if w_head is None:
            rec.carry = carry
            rec.mix_w_head = None
            return main, rec

        cur2, eff_head = self._mix_with_alpha(
            spec=spec,
            head_out=main,
            skip=cur,
            w_head=w_head,
            w_skip=w_skip,
            hardened=hardened,
            device=cur.device,
            dtype=cur.dtype,
            context="add alpha mixing",
        )
        if eff_head is None:
            raise RuntimeError(f"[{spec.name}] add gate unexpectedly missing.")
        scale_carry = eff_head if spec.weight_carry else torch.ones_like(eff_head)

        rec.carry = carry * scale_carry
        rec.mix_w_head = float(eff_head.item())
        return cur2, rec

    def _apply_none(
        self,
        cur: torch.Tensor,
        out: Any,
        spec: HeadSpec,
        w_head: torch.Tensor | None,
        w_skip: torch.Tensor | None,
        hardened: bool | None,
        rec: SerialNoneState,
    ) -> tuple[torch.Tensor, SerialNoneState]:
        if not isinstance(out, torch.Tensor):
            raise RuntimeError(
                f"[{spec.name}] expected single Tensor for 'none', got {type(out).__name__}"
            )
        y = out

        if w_head is None:
            rec.mix_w_head = None
            y_eff = y
        else:
            y_eff, eff_head = self._mix_with_alpha(
                spec=spec,
                head_out=y,
                skip=cur,
                w_head=w_head,
                w_skip=w_skip,
                hardened=hardened,
                device=cur.device,
                dtype=cur.dtype,
                context="none alpha mixing",
            )
            if eff_head is None:
                raise RuntimeError(f"[{spec.name}] none gate unexpectedly missing.")
            rec.mix_w_head = float(eff_head.item())

        cur2 = self._merge_serial_none(
            cur=cur,
            y=y_eff,
            head_name=spec.name,
            rec=rec,
            spec=spec,
        )
        return cur2, rec

    # ──────────────────────────────────────────────────────────────────────────
    # Inverse (only serial stage)
    # ──────────────────────────────────────────────────────────────────────────
    def inverse_post(self, y: torch.Tensor, run_state: RunStateList) -> torch.Tensor:
        if self.composer_mode == "parallel":
            return y
        return self._inverse_serial_stage(y, run_state)

    def _inverse_serial_stage(
        self, y: torch.Tensor, run_state: RunStateList
    ) -> torch.Tensor:
        cur = y
        serial_states = [s for s in run_state if s.stage == "serial"]
        for state in reversed(serial_states):
            cur = self._inverse_serial_state(cur, state)
        return cur

    def _inverse_serial_state(
        self, cur: torch.Tensor, state: BaseRunState
    ) -> torch.Tensor:
        if state.kind in ("disabled", "parallel_none", "serial_none"):
            return cur
        if isinstance(state, SerialAddState):
            return self._inverse_add(cur, state)
        if isinstance(state, SerialInvertState):
            return self._inverse_invert(cur, state)
        raise ValueError(f"Unknown combine mode during inverse: '{state.kind}'")

    def _inverse_add(self, cur: torch.Tensor, state: SerialAddState) -> torch.Tensor:
        name = state.name
        if state.carry is None:
            raise RuntimeError(f"[{name}] invalid state for add inverse.")
        carry = state.carry
        base_dim = int(state.base_dim if state.base_dim is not None else cur.size(-1))
        add_project = bool(state.add_project)

        if carry.shape[1] != cur.shape[1]:
            carry = self._align_time(carry, cur.shape[1])

        if add_project:
            target_dim = (
                int(self.fixed_output_dim)
                if self.fixed_output_dim is not None
                else base_dim
            )
            proj = self._get_or_build_projection(
                category="add_carry",
                name=name,
                in_dim=int(carry.size(-1)),
                out_dim=target_dim,
                device=cur.device,
                dtype=cur.dtype,
                allow_identity=True,
                registered_category="add_carry_custom",
            )
            carry = proj(carry)

        if carry.size(-1) != cur.size(-1):
            raise RuntimeError(
                f"[{name}] carry feature dim {carry.size(-1)} != current {cur.size(-1)}. "
                "Set add_project=True or provide custom_add_proj or set output_dim."
            )

        return cur + carry

    def _inverse_invert(
        self, cur: torch.Tensor, state: SerialInvertState
    ) -> torch.Tensor:
        name = state.name
        head = state.head_ref
        st = state.state
        gate_on = state.gate_on

        if head is None:
            raise RuntimeError(
                f"[{name}] missing head_ref for invert (corrupted run_state)."
            )

        if not gate_on:
            return cur

        if not hasattr(head, "invert") or not callable(getattr(head, "invert")):
            raise RuntimeError(
                f"[{name}] combine='invert' but head {type(head).__name__} has no callable .invert()"
            )

        return head.invert(cur, st)

    # ──────────────────────────────────────────────────────────────────────────
    # NAS utilities
    # ──────────────────────────────────────────────────────────────────────────
    def arch_parameters(self) -> Iterator[nn.Parameter]:
        return self.state_manager.arch_parameters()

    def weight_parameters(self) -> Iterator[nn.Parameter]:
        if not self.enable_nas:
            return self.parameters()
        alpha_ids = {id(p) for p in self.state_manager.alphas.values()}
        return (p for p in self.parameters() if id(p) not in alpha_ids)

    def alpha_report(self) -> dict[str, dict[str, Any]]:
        return self.state_manager.alpha_report()

    @torch.no_grad()
    def discretize_(self, threshold: float = 0.5) -> dict[str, bool]:
        return self.state_manager.discretize_(threshold)

    @torch.no_grad()
    def clear_discretization_(self) -> None:
        self.state_manager.clear_discretization_()

    # ──────────────────────────────────────────────────────────────────────────
    # Warmup (forces projection registry materialization before optimizer)
    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def warmup(
        self,
        example_x: torch.Tensor,
        example_y_like: torch.Tensor | None = None,
    ) -> None:
        """
        Dry run forward_pre + inverse_post to force creation/registration of:
          - serial-stage add carry projections (if any)
          - any internal head buffers depending on shape

        Call once BEFORE creating the optimizer when using serial add_project=True.
        """
        was_training = self.training
        try:
            self.eval()
            _, st = self.forward_pre(example_x)

            if example_y_like is None:
                # best-effort y-like shape; you can pass your real model output tensor if you want
                Fdim = (
                    int(self.fixed_output_dim)
                    if self.fixed_output_dim is not None
                    else int(example_x.size(-1))
                )
                example_y_like = torch.zeros(
                    example_x.size(0),
                    example_x.size(1),
                    Fdim,
                    device=example_x.device,
                    dtype=example_x.dtype,
                )

            _ = self.inverse_post(example_y_like, st)
        finally:
            self.train(was_training)

    # ──────────────────────────────────────────────────────────────────────────
    # Introspection
    # ──────────────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        head_info = ", ".join(
            f"{h.name}({'on' if h.enabled else 'off'})" for h in self.active_heads
        )
        return (
            f"{self.__class__.__name__}("
            f"mode={self.composer_mode}, "
            f"parallel_combine={self.parallel_combine}, "
            f"parallel_align_mode={self.parallel_align_mode}, "
            f"parallel_structured_outputs={self.parallel_structured_outputs}, "
            f"parallel_project={self.parallel_project}, "
            f"serial_none_merge={self.serial_none_merge}, "
            f"serial_none_project={self.serial_none_project}, "
            f"heads=[{head_info}], "
            f"enable_nas={self.enable_nas}, "
            f"output_dim={self.fixed_output_dim})"
        )

    def summary(self) -> str:
        lines = [
            (
                f"HeadComposer mode={self.composer_mode} "
                f"(parallel_combine={self.parallel_combine}, "
                f"parallel_align_mode={self.parallel_align_mode}, "
                f"parallel_structured_outputs={self.parallel_structured_outputs}, "
                f"serial_none_merge={self.serial_none_merge})"
            ),
            (
                f"parallel_project={self.parallel_project}"
                f"{'' if self.parallel_project_dim is None else f'->{self.parallel_project_dim}'}; "
                f"parallel_hyper_hidden_dim={self.parallel_hyper_hidden_dim}; "
                f"parallel_attention_heads={self.parallel_attention_heads}; "
                f"parallel_fusion_dropout={self.parallel_fusion_dropout}; "
                f"serial_none_project={self.serial_none_project}"
                f"{'' if self.serial_none_project_dim is None else f'->{self.serial_none_project_dim}'}"
            ),
            f"parallel_heads={len(self.parallel_meta)}, serial_heads={len(self.serial_meta)}",
        ]
        for head in self.active_heads:
            spec = head.spec
            enabled = head.enabled
            status = "on" if enabled else "off"
            alpha_info = ""
            if self.enable_nas and spec.alpha_mode != "off":
                hardened = head.hardened
                if hardened is not None:
                    alpha_info = f" [hardened={'on' if hardened else 'off'}]"
                else:
                    w_head, _ = self.state_manager.deterministic_alpha_weights_for_head(
                        head
                    )
                    if w_head is not None:
                        alpha_info = f" [alpha={w_head.item():.3f}]"
            stage = head.stage
            lines.append(
                f"  {spec.name} ({stage}): {status}, combine={spec.combine}, "
                f"alpha={spec.alpha_mode}, mix={spec.alpha_mix_style}{alpha_info}"
            )
        return "\n".join(lines)

    @property
    def head_names(self) -> list[str]:
        return [s.name for s in (self.parallel_meta + self.serial_meta)]

    @property
    def enabled_heads(self) -> list[str]:
        return [h.name for h in self.active_heads if h.enabled]

    @property
    def disabled_heads(self) -> list[str]:
        return [h.name for h in self.active_heads if not h.enabled]
