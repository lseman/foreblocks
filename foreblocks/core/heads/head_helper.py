from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.ui.node_spec import node

BaseHeadLike = Union[nn.Module]

# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────
CombineMode = Literal["invert", "add", "none"]
AlphaMode = Literal["off", "gate", "soft"]
AlphaMixStyle = Literal["blend", "residual"]
AlphaWeights = Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]

ComposerMode = Literal["serial", "parallel", "hybrid"]
ParallelCombine = Literal[
    "concat",
    "sum",
    "mean",
    "weighted_sum",
    "hypernetwork_mix",
    "attention_fusion",
    "gated_fusion",
]
ParallelAlignMode = Literal["strict", "project"]
SerialNoneMerge = Literal["replace", "add", "concat"]
ParallelStructuredOutputs = Literal["error", "main", "main_add_second"]

@dataclass
class BaseRunState:
    kind: str
    name: str
    stage: Literal["parallel", "serial"]
    base_dim: Optional[int] = None


@dataclass
class ParallelNoneState(BaseRunState):
    kind: Literal["parallel_none"] = field(default="parallel_none", init=False)
    mix_w_head: Optional[float] = None


@dataclass
class SerialInvertState(BaseRunState):
    kind: Literal["serial_invert"] = field(default="serial_invert", init=False)
    gate_on: bool = True
    gate_value: Optional[float] = None
    state: Any = None
    head_ref: Optional[nn.Module] = None


@dataclass
class SerialAddState(BaseRunState):
    kind: Literal["serial_add"] = field(default="serial_add", init=False)
    carry: Optional[torch.Tensor] = None
    carry_shape: Tuple[int, ...] = ()
    mix_w_head: Optional[float] = None
    add_project: bool = True


@dataclass
class SerialNoneState(BaseRunState):
    kind: Literal["serial_none"] = field(default="serial_none", init=False)
    mix_w_head: Optional[float] = None
    serial_none_merge: Optional[SerialNoneMerge] = None
    serial_none_project_dim: Optional[int] = None
    serial_none_time_aligned: bool = False


RunStateEntry = Union[
    ParallelNoneState,
    SerialInvertState,
    SerialAddState,
    SerialNoneState,
    BaseRunState,
]
RunStateList = List[RunStateEntry]


@dataclass
class HeadSpec:
    """
    Describe how a head behaves and (optionally) how to invert it.

    combine:
      - "invert": force reversible; head must return (y, state) and implement invert(y, state)
      - "add" : force split-add; head must return (main, carry)
      - "none" : forward-only; head must return y

    add_project:
      - If True and combine == "add", project carry to match inverse target feature dim.

    α (architecture) controls:
      - alpha_mode:
          "off" : legacy behavior (no α)
          "gate" : head is gated on/off with straight-through sigmoid (invert-safe)
          "soft" : y = w_head * head(x) + w_skip * x (non-invert heads only)
      - alpha_mix_style:
          "blend" : y_eff = w_head * y + w_skip * x (legacy)
          "residual" : y_eff = x + w_head * (y - x)
          For gate mode this becomes residual gating with hard/soft gate weights.
      - alpha_init: initial scalar/logit for the gate/soft weights (default 0.0)
      - alpha_trainable: whether α is trainable
      - weight_carry: when combine="add", also scale the carry by the same head weight
    """

    head: BaseHeadLike
    name: str
    combine: CombineMode = "none"
    add_project: bool = True
    custom_add_proj: Optional[nn.Module] = None
    alpha_mode: AlphaMode = "off"
    alpha_init: float = 0.0
    alpha_trainable: bool = True
    weight_carry: bool = True
    alpha_mix_style: AlphaMixStyle = "blend"

    def __post_init__(self):
        valid_combine = ("invert", "add", "none")
        valid_alpha = ("off", "gate", "soft")
        valid_alpha_mix = ("blend", "residual")

        if self.combine not in valid_combine:
            raise ValueError(
                f"HeadSpec '{self.name}': combine must be one of {valid_combine}, got '{self.combine}'"
            )
        if self.alpha_mode not in valid_alpha:
            raise ValueError(
                f"HeadSpec '{self.name}': alpha_mode must be one of {valid_alpha}, got '{self.alpha_mode}'"
            )
        if self.alpha_mix_style not in valid_alpha_mix:
            raise ValueError(
                f"HeadSpec '{self.name}': alpha_mix_style must be one of {valid_alpha_mix}, "
                f"got '{self.alpha_mix_style}'"
            )
        if not self.name:
            raise ValueError("HeadSpec name cannot be empty")
        if self.custom_add_proj is not None and self.combine != "add":
            import warnings

            warnings.warn(
                f"HeadSpec '{self.name}': custom_add_proj is set but combine='{self.combine}' "
                "(only used when combine is 'add')"
            )

    def __repr__(self) -> str:
        return (
            f"HeadSpec(name={self.name!r}, combine={self.combine!r}, "
            f"alpha_mode={self.alpha_mode!r}, alpha_mix_style={self.alpha_mix_style!r}, "
            f"add_project={self.add_project})"
        )


@dataclass
class ActiveHead:
    spec: HeadSpec
    index: int
    is_parallel: bool
    enabled: bool = True
    hardened: Optional[bool] = None
    alpha_param: Optional[nn.Parameter] = None
    mix_logit: Optional[nn.Parameter] = None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def stage(self) -> Literal["parallel", "serial"]:
        return "parallel" if self.is_parallel else "serial"


class HeadStateManager(nn.Module):
    """
    Centralize mutable per-head state:
      - NAS alpha parameters
      - enable/disable masks
      - hardened architecture decisions
    """

    _EPS: float = 1e-6

    def __init__(
        self,
        active_heads: List[ActiveHead],
        *,
        enable_nas: bool,
        alpha_temperature: float,
    ) -> None:
        super().__init__()
        self._active_heads: List[ActiveHead] = list(active_heads)
        self._head_by_name: Dict[str, ActiveHead] = {
            h.name: h for h in self._active_heads
        }
        self.enable_nas = bool(enable_nas)
        self.alpha_temperature = float(alpha_temperature)

        self._alphas = nn.ParameterDict()
        if self.enable_nas:
            self._init_alpha_params(self._active_heads)

    @property
    def alphas(self) -> nn.ParameterDict:
        return self._alphas

    def _init_alpha_params(self, active_heads: List[ActiveHead]) -> None:
        for head in active_heads:
            spec = head.spec
            if spec.alpha_mode == "off":
                continue
            if spec.alpha_mode == "soft":
                p = nn.Parameter(
                    torch.tensor([spec.alpha_init, 0.0], dtype=torch.float32)
                )
            elif spec.alpha_mode == "gate":
                p = nn.Parameter(torch.tensor([spec.alpha_init], dtype=torch.float32))
            else:
                raise ValueError(
                    f"alpha_mode must be one of ('off','gate','soft'), got '{spec.alpha_mode}'"
                )
            p.requires_grad_(bool(spec.alpha_trainable))
            self._alphas[spec.name] = p
            head.alpha_param = p

    def alpha_weights_for_head(self, head: ActiveHead) -> AlphaWeights:
        spec = head.spec
        if not self.enable_nas or spec.alpha_mode == "off":
            return None, None

        a = head.alpha_param if head.alpha_param is not None else self._alphas[spec.name]
        tau = max(self._EPS, float(self.alpha_temperature))

        if spec.alpha_mode == "soft":
            w = F.softmax(a / tau, dim=0)
            return w[0], w[1]

        g = torch.sigmoid(a[0] / tau)
        return g, 1.0 - g

    @staticmethod
    def straight_through_hard_gate(g: torch.Tensor) -> torch.Tensor:
        g_hard = (g > 0.5).float()
        return g_hard.detach() - g.detach() + g

    def compute_effective_weight(
        self,
        spec: HeadSpec,
        w_head: Optional[torch.Tensor],
        w_skip: Optional[torch.Tensor],
        hardened: Optional[bool],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hardened is not None:
            h = float(hardened)
            return (
                torch.tensor(h, device=device, dtype=dtype),
                torch.tensor(1.0 - h, device=device, dtype=dtype),
            )

        if w_head is None:
            return (
                torch.tensor(1.0, device=device, dtype=dtype),
                torch.tensor(0.0, device=device, dtype=dtype),
            )

        if spec.alpha_mode == "soft":
            if w_skip is None:
                w_skip = 1.0 - w_head
            return w_head, w_skip

        g_used = self.straight_through_hard_gate(w_head)
        return g_used, 1.0 - g_used

    def check_head_exists(self, name: str) -> None:
        if name not in self._head_by_name:
            available = list(self._head_by_name.keys())
            raise KeyError(f"Unknown head name: '{name}'. Available: {available}")

    def get_head(self, name: str) -> ActiveHead:
        self.check_head_exists(name)
        return self._head_by_name[name]

    @torch.no_grad()
    def enable_head(self, name: str) -> None:
        head = self.get_head(name)
        head.enabled = True

    @torch.no_grad()
    def disable_head(self, name: str) -> None:
        head = self.get_head(name)
        head.enabled = False

    @torch.no_grad()
    def set_head_enabled(self, name: str, enabled: bool) -> None:
        head = self.get_head(name)
        head.enabled = bool(enabled)

    @torch.no_grad()
    def enable_all(self) -> None:
        for head in self._active_heads:
            head.enabled = True

    @torch.no_grad()
    def disable_all(self) -> None:
        for head in self._active_heads:
            head.enabled = False

    def enabled_report(self) -> Dict[str, bool]:
        return {h.name: h.enabled for h in self._active_heads}

    @contextmanager
    def temporary_disable(self, *names: str) -> Iterator[None]:
        previous_states: Dict[str, bool] = {}
        for name in names:
            head = self.get_head(name)
            previous_states[name] = head.enabled
            head.enabled = False
        try:
            yield
        finally:
            for name, prev in previous_states.items():
                self._head_by_name[name].enabled = prev

    @contextmanager
    def only_heads(self, *names: str) -> Iterator[None]:
        for name in names:
            self.check_head_exists(name)
        previous_states = {h.name: h.enabled for h in self._active_heads}
        self.disable_all()
        for name in names:
            self._head_by_name[name].enabled = True
        try:
            yield
        finally:
            for name, prev in previous_states.items():
                self._head_by_name[name].enabled = prev

    def arch_parameters(self) -> Iterator[nn.Parameter]:
        if not self.enable_nas:
            return iter([])
        return (p for p in self._alphas.values() if p.requires_grad)

    def alpha_report(self) -> Dict[str, Dict[str, Any]]:
        if not self.enable_nas:
            return {}

        rep: Dict[str, Dict[str, Any]] = {}
        for head in self._active_heads:
            spec = head.spec
            d: Dict[str, Any] = {
                "mode": spec.alpha_mode,
                "mix_style": spec.alpha_mix_style,
                "enabled": head.enabled,
                "hardened": head.hardened,
            }
            if spec.alpha_mode == "off" or spec.name not in self._alphas:
                rep[spec.name] = d
                continue

            w_head, w_skip = self.alpha_weights_for_head(head)
            if spec.alpha_mode == "soft":
                d["w_head"] = float(w_head.item())
                d["w_skip"] = float(w_skip.item())
            else:
                d["p_on"] = float(w_head.item())
            rep[spec.name] = d
        return rep

    @torch.no_grad()
    def discretize_(self, threshold: float = 0.5) -> Dict[str, bool]:
        if not self.enable_nas:
            return {}

        decisions: Dict[str, bool] = {}
        for head in self._active_heads:
            spec = head.spec
            if spec.alpha_mode == "off" or spec.name not in self._alphas:
                head.hardened = None
                continue
            w_head, _ = self.alpha_weights_for_head(head)
            decision = bool(w_head >= threshold)
            head.hardened = decision
            decisions[spec.name] = decision
        return decisions

    @torch.no_grad()
    def clear_discretization_(self) -> None:
        for head in self._active_heads:
            head.hardened = None


class ParallelComposer:
    """Parallel stage facade: branch collection + fusion."""

    def __init__(self, owner: "HeadComposer") -> None:
        self.owner = owner

    def forward_stage(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, RunStateList]:
        return self.owner._forward_parallel_stage_impl(x)


class SerialComposer:
    """Serial stage facade: reversible chain and inverse."""

    def __init__(self, owner: "HeadComposer") -> None:
        self.owner = owner

    def forward_stage(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, RunStateList]:
        return self.owner._forward_serial_stage_impl(x)

    def inverse_stage(
        self, y: torch.Tensor, run_state: RunStateList
    ) -> torch.Tensor:
        return self.owner._inverse_serial_stage(y, run_state)


class HeadOrchestrator:
    """Coordinate parallel and serial components according to composer mode."""

    def __init__(
        self,
        *,
        mode: ComposerMode,
        parallel: ParallelComposer,
        serial: SerialComposer,
    ) -> None:
        self.mode = mode
        self.parallel = parallel
        self.serial = serial

    def forward_pre(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, RunStateList]:
        if self.mode in ("parallel", "hybrid"):
            x_aug, st_parallel = self.parallel.forward_stage(x)
        else:
            x_aug, st_parallel = x, []

        if self.mode in ("serial", "hybrid"):
            x_out, st_serial = self.serial.forward_stage(x_aug)
        else:
            x_out, st_serial = x_aug, []

        return x_out, st_parallel + st_serial

    def inverse_post(
        self, y: torch.Tensor, run_state: RunStateList
    ) -> torch.Tensor:
        if self.mode == "parallel":
            return y
        return self.serial.inverse_stage(y, run_state)


class ProjectionRegistry(nn.Module):
    """
    Unified factory/registry for projection and fusion helper modules.

    Keys are structured by `(category, name, in_dim, out_dim, proj_type, kwargs...)`.
    """

    def __init__(self) -> None:
        super().__init__()
        # Do not touch nn.Module._modules (internal PyTorch container).
        self.registry = nn.ModuleDict()

    @staticmethod
    def _safe_token(val: Any) -> str:
        return str(val).replace("|", "_").replace(":", "_").replace(" ", "_")

    def _make_key(
        self,
        *,
        category: str,
        name: str,
        in_dim: int,
        out_dim: int,
        proj_type: str,
        kwargs: Dict[str, Any],
    ) -> str:
        parts = [
            self._safe_token(category),
            self._safe_token(name),
            str(int(in_dim)),
            str(int(out_dim)),
            self._safe_token(proj_type),
        ]
        if kwargs:
            extras = "|".join(
                f"{self._safe_token(k)}={self._safe_token(v)}"
                for k, v in sorted(kwargs.items(), key=lambda kv: kv[0])
            )
            parts.append(extras)
        return "::".join(parts)

    def register_custom(self, *, category: str, name: str, module: nn.Module) -> None:
        key = self._make_key(
            category=category,
            name=name,
            in_dim=-1,
            out_dim=-1,
            proj_type="registered",
            kwargs={},
        )
        self.registry[key] = module

    def get_registered(self, *, category: str, name: str) -> Optional[nn.Module]:
        key = self._make_key(
            category=category,
            name=name,
            in_dim=-1,
            out_dim=-1,
            proj_type="registered",
            kwargs={},
        )
        if key in self.registry:
            return self.registry[key]
        return None

    def get_or_create(
        self,
        *,
        category: str,
        name: str,
        in_dim: int,
        out_dim: int,
        proj_type: str = "linear",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> nn.Module:
        key = self._make_key(
            category=category,
            name=name,
            in_dim=in_dim,
            out_dim=out_dim,
            proj_type=proj_type,
            kwargs=kwargs,
        )
        if key not in self.registry:
            self.registry[key] = self._build_module(
                in_dim=in_dim,
                out_dim=out_dim,
                proj_type=proj_type,
                **kwargs,
            )
        mod = self.registry[key]
        if device is not None or dtype is not None:
            mod = mod.to(device=device, dtype=dtype)
        return mod

    def _build_module(
        self,
        *,
        in_dim: int,
        out_dim: int,
        proj_type: str,
        **kwargs: Any,
    ) -> nn.Module:
        if proj_type == "identity":
            return nn.Identity()

        if proj_type == "linear":
            allow_identity = bool(kwargs.get("allow_identity", True))
            if allow_identity and int(in_dim) == int(out_dim):
                return nn.Identity()
            proj = nn.Linear(int(in_dim), int(out_dim))
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
            return proj

        if proj_type == "conv1d":
            kernel_size = int(kwargs.get("kernel_size", 1))
            padding = int(kwargs.get("padding", kernel_size // 2))
            bias = bool(kwargs.get("bias", True))
            conv = nn.Conv1d(
                in_channels=int(in_dim),
                out_channels=int(out_dim),
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            )
            nn.init.kaiming_uniform_(conv.weight, a=5**0.5)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
            return conv

        if proj_type == "hyper_mlp":
            hidden_dim = int(kwargs.get("hidden_dim", 64))
            out_features = int(kwargs.get("out_features", out_dim))
            dropout = float(kwargs.get("dropout", 0.0))
            drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            net = nn.Sequential(
                nn.Linear(int(in_dim), hidden_dim),
                nn.GELU(),
                drop,
                nn.Linear(hidden_dim, out_features),
            )
            if isinstance(net[-1], nn.Linear) and net[-1].bias is not None:
                nn.init.zeros_(net[-1].bias)
            return net

        if proj_type == "gate_mlp":
            hidden_dim = int(kwargs.get("hidden_dim", max(8, int(in_dim) // 2)))
            out_features = int(kwargs.get("out_features", 1))
            net = nn.Sequential(
                nn.Linear(int(in_dim), hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_features),
            )
            if isinstance(net[-1], nn.Linear) and net[-1].bias is not None:
                nn.init.zeros_(net[-1].bias)
            return net

        if proj_type == "multihead_attention":
            num_heads = int(kwargs["num_heads"])
            dropout = float(kwargs.get("dropout", 0.0))
            batch_first = bool(kwargs.get("batch_first", True))
            return nn.MultiheadAttention(
                embed_dim=int(in_dim),
                num_heads=num_heads,
                dropout=dropout,
                batch_first=batch_first,
            )

        raise ValueError(f"Unsupported proj_type: {proj_type}")


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
        specs: Optional[List[HeadSpec]] = None,
        *,
        # For hybrid, split specs:
        parallel_specs: Optional[List[HeadSpec]] = None,
        serial_specs: Optional[List[HeadSpec]] = None,
        output_dim: Optional[int] = None,
        stop_gradient_on_carry: bool = False,
        alpha_temperature: float = 1.0,
        enable_nas: bool = False,
        composer_mode: ComposerMode = "serial",
        parallel_combine: ParallelCombine = "concat",
        # Parallel stage controls:
        parallel_align_mode: ParallelAlignMode = "strict",
        parallel_project: bool = False,
        parallel_project_dim: Optional[int] = None,
        parallel_structured_outputs: ParallelStructuredOutputs = "error",
        parallel_hyper_hidden_dim: int = 64,
        parallel_attention_heads: int = 0,
        parallel_fusion_dropout: float = 0.0,
        # Serial stage controls for combine='none':
        serial_none_merge: SerialNoneMerge = "replace",
        serial_none_project: bool = False,
        serial_none_project_dim: Optional[int] = None,
    ):
        super().__init__()

        # Normalize args
        specs: List[HeadSpec] = specs or []
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
        if composer_mode == "serial":
            if parallel_specs or serial_specs:
                raise ValueError(
                    "In composer_mode='serial', pass specs=... only (not parallel_specs/serial_specs)."
                )
            self.parallel_meta: List[HeadSpec] = []
            self.serial_meta: List[HeadSpec] = list(specs)
        elif composer_mode == "parallel":
            if specs and parallel_specs:
                raise ValueError(
                    "In composer_mode='parallel', pass exactly one of specs=... or parallel_specs=..., not both."
                )
            if serial_specs:
                raise ValueError(
                    "In composer_mode='parallel', pass parallel_specs=... or specs=..., not serial_specs."
                )
            self.parallel_meta = list(parallel_specs) if parallel_specs else list(specs)
            self.serial_meta = []
        elif composer_mode == "hybrid":
            if specs:
                raise ValueError(
                    "In composer_mode='hybrid', pass parallel_specs=... and serial_specs=... (not specs)."
                )
            self.parallel_meta = list(parallel_specs)
            self.serial_meta = list(serial_specs)
        else:
            raise ValueError(f"Unknown composer_mode: {composer_mode}")

        # Validate unique names across all heads
        self._validate_unique_names(self.parallel_meta, self.serial_meta)
        self._validate_spec_constraints()

        # Register heads as modules (keep ordering stable)
        self.parallel_heads = nn.ModuleList([s.head for s in self.parallel_meta])
        self.serial_heads = nn.ModuleList([s.head for s in self.serial_meta])
        self.parallel_active_heads: List[ActiveHead] = [
            ActiveHead(spec=s, index=i, is_parallel=True)
            for i, s in enumerate(self.parallel_meta)
        ]
        self.serial_active_heads: List[ActiveHead] = [
            ActiveHead(spec=s, index=i, is_parallel=False)
            for i, s in enumerate(self.serial_meta)
        ]
        self.active_heads: List[ActiveHead] = (
            self.parallel_active_heads + self.serial_active_heads
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
            [nn.Parameter(torch.zeros(1, dtype=torch.float32)) for _ in self.parallel_meta]
        )
        for i, head in enumerate(self.parallel_active_heads):
            head.mix_logit = self.parallel_mix_logits[i]

        self.state_manager = HeadStateManager(
            self.active_heads,
            enable_nas=self.enable_nas,
            alpha_temperature=self.alpha_temperature,
        )

        # Used for deterministic projection sizing when output_dim is None
        self._last_base_dim: Optional[int] = None

        # Validate mode constraints early
        self._validate_mode_constraints()

        # Stage components and orchestrator.
        self.parallel_composer = ParallelComposer(self)
        self.serial_composer = SerialComposer(self)
        self.orchestrator = HeadOrchestrator(
            mode=self.composer_mode,
            parallel=self.parallel_composer,
            serial=self.serial_composer,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _validate_unique_names(
        parallel_specs: List[HeadSpec], serial_specs: List[HeadSpec]
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
        if self.serial_none_merge not in ("replace", "add", "concat"):
            raise ValueError(
                "serial_none_merge must be one of ('replace','add','concat'), "
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
            bad: List[str] = []
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
        registered_category: Optional[str] = None,
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
    ) -> torch.Tensor:
        mode = self.serial_none_merge
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
        branches: List[Tuple[ActiveHead, torch.Tensor]],
        x_ref: torch.Tensor,
    ) -> List[Tuple[ActiveHead, torch.Tensor]]:
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
        aligned: List[Tuple[ActiveHead, torch.Tensor]] = []
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
        w_head: Optional[torch.Tensor],
        w_skip: Optional[torch.Tensor],
        hardened: Optional[bool],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        raise ValueError(
            f"[{spec.name}] unknown alpha_mix_style '{spec.alpha_mix_style}'"
        )

    def _mix_with_alpha(
        self,
        *,
        spec: HeadSpec,
        head_out: torch.Tensor,
        skip: torch.Tensor,
        w_head: Optional[torch.Tensor],
        w_skip: Optional[torch.Tensor],
        hardened: Optional[bool],
        device: torch.device,
        dtype: torch.dtype,
        context: str,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if w_head is None:
            return head_out, None

        self._ensure_same_shape_for_mixing(
            spec.name, head_out, skip, context=context
        )
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

    def enabled_report(self) -> Dict[str, bool]:
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
    def forward_pre(self, x: torch.Tensor) -> Tuple[torch.Tensor, RunStateList]:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B,T,F], got {tuple(x.shape)}")

        self._last_base_dim = int(x.size(-1))
        return self.orchestrator.forward_pre(x)

    def forward(
        self,
        x: torch.Tensor,
        encoder: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, RunStateList]:
        out, state = self.forward_pre(x)
        if encoder is not None:
            out = encoder(out)
        return out, state

    # ──────────────────────────────────────────────────────────────────────────
    # Parallel stage (feature augmentation)
    # ──────────────────────────────────────────────────────────────────────────
    def _forward_parallel_stage_impl(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, RunStateList]:
        run_state: RunStateList = []
        branches: List[Tuple[ActiveHead, torch.Tensor]] = []

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
            hardened = active_head.hardened

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
                hardened=hardened,
                device=x.device,
                dtype=x.dtype,
                context="parallel alpha mixing",
            )
            rec.mix_w_head = None if eff_head is None else float(eff_head.item())

            branches.append((active_head, y))
            run_state.append(rec)

        if not branches:
            return x, run_state

        if self.parallel_combine == "concat":
            out = torch.cat([b for _, b in branches], dim=-1)
        elif self.parallel_combine in ("sum", "mean"):
            aligned = self._align_parallel_branches_for_fusion(branches, x)
            branch_tensors = [b for _, b in aligned]
            out = torch.stack(branch_tensors, dim=0).sum(dim=0)
            if self.parallel_combine == "mean":
                out = out / float(len(branch_tensors))
        elif self.parallel_combine == "weighted_sum":
            aligned = self._align_parallel_branches_for_fusion(branches, x)
            if any(head.mix_logit is None for head, _ in aligned):
                raise RuntimeError("Missing parallel mix logits for weighted_sum.")
            logits = torch.stack(
                [head.mix_logit[0] for head, _ in aligned],
                dim=0,
            ).to(device=x.device)
            weights = F.softmax(logits.float(), dim=0).to(dtype=aligned[0][1].dtype)
            out = torch.zeros_like(aligned[0][1])
            for i, (head, b) in enumerate(aligned):
                out = out + b * weights[i]
        elif self.parallel_combine == "hypernetwork_mix":
            aligned = self._align_parallel_branches_for_fusion(branches, x)
            branch_stack = torch.stack([b for _, b in aligned], dim=0)  # [N,B,T,F]
            context = branch_stack.mean(dim=0).mean(dim=1)  # [B,F]
            hyper = self._get_or_build_parallel_hypernet(
                in_dim=int(context.size(-1)),
                device=x.device,
                dtype=x.dtype,
            )
            logits_all = hyper(context)  # [B, N_total_parallel]
            active_idx = torch.tensor(
                [head.index for head, _ in aligned],
                device=x.device,
                dtype=torch.long,
            )
            logits = logits_all.index_select(1, active_idx)
            weights = F.softmax(logits.float(), dim=1).to(dtype=aligned[0][1].dtype)
            out = torch.zeros_like(aligned[0][1])
            for i, (head, b) in enumerate(aligned):
                w_i = weights[:, i].view(-1, 1, 1)
                out = out + b * w_i
        elif self.parallel_combine == "attention_fusion":
            aligned = self._align_parallel_branches_for_fusion(branches, x)
            B, T, Fdim = aligned[0][1].shape
            num_branches = len(aligned)
            attn = self._get_or_build_parallel_attention(
                embed_dim=Fdim,
                device=x.device,
                dtype=x.dtype,
            )
            stacked = torch.stack([b for _, b in aligned], dim=2)  # [B,T,N,F]
            tokens = stacked.reshape(B * T, num_branches, Fdim)  # [B*T,N,F]
            attn_out, _ = attn(tokens, tokens, tokens, need_weights=False)
            out = attn_out.mean(dim=1).reshape(B, T, Fdim)
        elif self.parallel_combine == "gated_fusion":
            aligned = self._align_parallel_branches_for_fusion(branches, x)
            Fdim = aligned[0][1].size(-1)
            gate_net = self._get_or_build_parallel_gate_net(
                in_dim=int(Fdim),
                device=x.device,
                dtype=x.dtype,
            )
            gate_scores = torch.stack(
                [torch.sigmoid(gate_net(b.mean(dim=1))) for _, b in aligned], dim=1
            )  # [B,N,1]
            gate_norm = gate_scores / gate_scores.sum(dim=1, keepdim=True).clamp_min(
                self._EPS
            )
            out = torch.zeros_like(aligned[0][1])
            for i, (head, b) in enumerate(aligned):
                g_i = gate_norm[:, i, :].unsqueeze(1).to(dtype=b.dtype)  # [B,1,1]
                out = out + b * g_i
        else:
            raise ValueError(f"Unknown parallel_combine: {self.parallel_combine}")

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

    # ──────────────────────────────────────────────────────────────────────────
    # Serial stage (reversible)
    # ──────────────────────────────────────────────────────────────────────────
    def _forward_serial_stage_impl(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, RunStateList]:
        run_state: RunStateList = []
        cur = x

        for active_head in self.serial_active_heads:
            spec = active_head.spec
            if not active_head.enabled:
                run_state.append(
                    BaseRunState(
                        kind="disabled",
                        name=active_head.name,
                        stage="serial",
                    )
                )
                continue

            out = spec.head(cur)
            combine = spec.combine

            w_head, w_skip = self._alpha_weights_for_head(active_head)
            hardened = active_head.hardened

            if combine == "invert":
                rec = SerialInvertState(
                    name=active_head.name,
                    stage="serial",
                    base_dim=self._last_base_dim,
                )
                cur, rec = self._apply_invert(cur, out, spec, w_head, hardened, rec)
            elif combine == "add":
                rec = SerialAddState(
                    name=active_head.name,
                    stage="serial",
                    base_dim=self._last_base_dim,
                )
                cur, rec = self._apply_add(
                    cur, out, spec, w_head, w_skip, hardened, rec
                )
            elif combine == "none":
                rec = SerialNoneState(
                    name=active_head.name,
                    stage="serial",
                    base_dim=self._last_base_dim,
                )
                cur, rec = self._apply_none(
                    cur, out, spec, w_head, w_skip, hardened, rec
                )
            else:
                raise ValueError(f"Unknown combine mode: '{combine}'")

            run_state.append(rec)

        return cur, run_state

    def _apply_invert(
        self,
        cur: torch.Tensor,
        out: Any,
        spec: HeadSpec,
        w_head: Optional[torch.Tensor],
        hardened: Optional[bool],
        rec: SerialInvertState,
    ) -> Tuple[torch.Tensor, SerialInvertState]:
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
        w_head: Optional[torch.Tensor],
        w_skip: Optional[torch.Tensor],
        hardened: Optional[bool],
        rec: SerialAddState,
    ) -> Tuple[torch.Tensor, SerialAddState]:
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
        w_head: Optional[torch.Tensor],
        w_skip: Optional[torch.Tensor],
        hardened: Optional[bool],
        rec: SerialNoneState,
    ) -> Tuple[torch.Tensor, SerialNoneState]:
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
        )
        return cur2, rec

    # ──────────────────────────────────────────────────────────────────────────
    # Inverse (only serial stage)
    # ──────────────────────────────────────────────────────────────────────────
    def inverse_post(
        self, y: torch.Tensor, run_state: RunStateList
    ) -> torch.Tensor:
        return self.orchestrator.inverse_post(y, run_state)

    def _inverse_serial_stage(
        self, y: torch.Tensor, run_state: RunStateList
    ) -> torch.Tensor:
        cur = y
        serial_states = [s for s in run_state if s.stage == "serial"]
        for state in reversed(serial_states):
            if state.kind in (
                "disabled",
                "parallel_none",
                "serial_none",
            ):
                continue
            if isinstance(state, SerialAddState):
                cur = self._inverse_add(cur, state)
            elif isinstance(state, SerialInvertState):
                cur = self._inverse_invert(cur, state)
            else:
                raise ValueError(
                    f"Unknown combine mode during inverse: '{state.kind}'"
                )
        return cur

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

    def _inverse_invert(self, cur: torch.Tensor, state: SerialInvertState) -> torch.Tensor:
        name = state.name
        head = state.head_ref
        st = state.state
        gate_on = state.gate_on

        if head is None or st is None:
            raise RuntimeError(
                f"[{name}] missing head_ref/state for invert (corrupted run_state)."
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

    def alpha_report(self) -> Dict[str, Dict[str, Any]]:
        return self.state_manager.alpha_report()

    @torch.no_grad()
    def discretize_(self, threshold: float = 0.5) -> Dict[str, bool]:
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
        example_y_like: Optional[torch.Tensor] = None,
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
                    w_head, _ = self._alpha_weights_for_head(head)
                    if w_head is not None:
                        alpha_info = f" [alpha={w_head.item():.3f}]"
            stage = head.stage
            lines.append(
                f"  {spec.name} ({stage}): {status}, combine={spec.combine}, "
                f"alpha={spec.alpha_mode}, mix={spec.alpha_mix_style}{alpha_info}"
            )
        return "\n".join(lines)

    @property
    def head_names(self) -> List[str]:
        return [s.name for s in (self.parallel_meta + self.serial_meta)]

    @property
    def enabled_heads(self) -> List[str]:
        return [h.name for h in self.active_heads if h.enabled]

    @property
    def disabled_heads(self) -> List[str]:
        return [h.name for h in self.active_heads if not h.enabled]
