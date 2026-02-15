from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.ui.node_spec import node

BaseHeadLike = Union[nn.Module]

# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────
CombineMode = Literal["auto", "invert", "add", "none"]
AlphaMode = Literal["off", "gate", "soft"]
AlphaWeights = Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]

ComposerMode = Literal["serial", "parallel", "hybrid"]
ParallelCombine = Literal["concat", "sum", "mean"]


@dataclass
class HeadSpec:
    """
    Describe how a head behaves and (optionally) how to invert it.

    combine:
      - "auto" : infer behavior per output:
          * (y, state) and hasattr(head, 'invert') -> "invert"
          * (main, carry) and no invert -> "add"
          * y only -> "none"
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
      - alpha_init: initial scalar/logit for the gate/soft weights (default 0.0)
      - alpha_trainable: whether α is trainable
      - weight_carry: when combine="add", also scale the carry by the same head weight
    """

    head: BaseHeadLike
    name: str
    combine: CombineMode = "auto"
    add_project: bool = True
    custom_add_proj: Optional[nn.Module] = None
    alpha_mode: AlphaMode = "off"
    alpha_init: float = 0.0
    alpha_trainable: bool = True
    weight_carry: bool = True

    def __post_init__(self):
        valid_combine = ("auto", "invert", "add", "none")
        valid_alpha = ("off", "gate", "soft")

        if self.combine not in valid_combine:
            raise ValueError(
                f"HeadSpec '{self.name}': combine must be one of {valid_combine}, got '{self.combine}'"
            )
        if self.alpha_mode not in valid_alpha:
            raise ValueError(
                f"HeadSpec '{self.name}': alpha_mode must be one of {valid_alpha}, got '{self.alpha_mode}'"
            )
        if not self.name:
            raise ValueError("HeadSpec name cannot be empty")
        if self.custom_add_proj is not None and self.combine not in ("auto", "add"):
            import warnings

            warnings.warn(
                f"HeadSpec '{self.name}': custom_add_proj is set but combine='{self.combine}' "
                "(only used when combine is 'auto' or 'add')"
            )

    def __repr__(self) -> str:
        return (
            f"HeadSpec(name={self.name!r}, combine={self.combine!r}, "
            f"alpha_mode={self.alpha_mode!r}, add_project={self.add_project})"
        )


@node(
    type_id="head_composer",
    name="HeadComposer",
    category="Misc",
    outputs=["head_composer"],
    color="bg-gradient-to-r from-purple-400 to-pink-500",
)
class HeadComposer(nn.Module):
    """
    Preprocessing *inside* the network, with optional inversion back to the original space.

    Modes:
      - composer_mode="serial":
            x -> h1 -> h2 -> ... -> x_out
            Supports combine: invert/add/none (inverse_post supported for invert/add).
      - composer_mode="parallel":
            for each enabled head: y_i = head(x0) (optional per-head α mixing vs skip)
            x_out = combine(y_i) via concat/sum/mean
            Only supports forward-only heads (combine must be 'none' or 'auto' resolving to 'none').
            inverse_post is identity.
      - composer_mode="hybrid":
            parallel stage (feature augmentation) then serial stage (reversible heads):
              x0 -> [parallel heads on x0] -> x_aug -> [serial heads on x_aug] -> x_out
            Inverse only undoes the SERIAL stage (invert/add), because parallel augmentation is not invertible.

    Important: carry projections for 'add' (add_project=True) can be created lazily.
    To ensure all parameters exist BEFORE you create the optimizer, call:
        composer.warmup(example_x, example_y_like=...)
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
    ):
        super().__init__()

        # Normalize args
        specs: List[HeadSpec] = specs or []
        parallel_specs = parallel_specs or []
        serial_specs = serial_specs or []

        self.composer_mode: ComposerMode = composer_mode
        self.parallel_combine: ParallelCombine = parallel_combine

        self.fixed_output_dim = output_dim
        self.stop_gradient_on_carry = stop_gradient_on_carry
        self.alpha_temperature = alpha_temperature
        self.enable_nas = enable_nas

        # Choose which specs are active given mode
        if composer_mode == "serial":
            if parallel_specs or serial_specs:
                raise ValueError("In composer_mode='serial', pass specs=... only (not parallel_specs/serial_specs).")
            self.parallel_meta: List[HeadSpec] = []
            self.serial_meta: List[HeadSpec] = list(specs)
        elif composer_mode == "parallel":
            if serial_specs:
                raise ValueError("In composer_mode='parallel', pass parallel_specs=... or specs=..., not serial_specs.")
            self.parallel_meta = list(parallel_specs) if parallel_specs else list(specs)
            self.serial_meta = []
        elif composer_mode == "hybrid":
            if specs:
                raise ValueError("In composer_mode='hybrid', pass parallel_specs=... and serial_specs=... (not specs).")
            self.parallel_meta = list(parallel_specs)
            self.serial_meta = list(serial_specs)
        else:
            raise ValueError(f"Unknown composer_mode: {composer_mode}")

        # Validate unique names across all heads
        self._validate_unique_names(self.parallel_meta, self.serial_meta)

        # Register heads as modules (keep ordering stable)
        self.parallel_heads = nn.ModuleList([s.head for s in self.parallel_meta])
        self.serial_heads = nn.ModuleList([s.head for s in self.serial_meta])

        # Pre-register any custom projections for serial add heads
        self.add_projs = nn.ModuleDict()
        for spec in self.serial_meta:
            if spec.combine in ("auto", "add") and spec.custom_add_proj is not None:
                self.add_projs[spec.name] = spec.custom_add_proj

        # α parameters (can exist for BOTH parallel and serial heads)
        self._alphas = nn.ParameterDict()
        if self.enable_nas:
            self._init_alpha_params(self.parallel_meta + self.serial_meta)

        # Hardened choices after discretization
        self._hardened_choice: Dict[str, Optional[bool]] = {
            s.name: None for s in (self.parallel_meta + self.serial_meta)
        }

        # Ablation-friendly enable/disable mask
        self._enabled: Dict[str, bool] = {s.name: True for s in (self.parallel_meta + self.serial_meta)}

        # Cache for spec lookups
        self._spec_by_name: Dict[str, HeadSpec] = {s.name: s for s in (self.parallel_meta + self.serial_meta)}

        # Used for deterministic projection sizing when output_dim is None
        self._last_base_dim: Optional[int] = None

        # Validate mode constraints early
        self._validate_mode_constraints()

    # ──────────────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _validate_unique_names(parallel_specs: List[HeadSpec], serial_specs: List[HeadSpec]) -> None:
        seen: set[str] = set()
        for s in parallel_specs + serial_specs:
            if s.name in seen:
                raise ValueError(f"HeadSpec names must be unique across all heads; duplicated: '{s.name}'")
            seen.add(s.name)

    def _validate_mode_constraints(self) -> None:
        # Parallel stage constraints: only forward-only heads can be safely used there.
        if self.composer_mode in ("parallel", "hybrid"):
            bad: List[str] = []
            for s in self.parallel_meta:
                if s.combine in ("invert", "add"):
                    bad.append(s.name)
            if bad:
                raise ValueError(
                    "Parallel stage does not support combine='invert' or 'add' (no general inverse). "
                    f"Offending: {bad}"
                )
            if self.parallel_combine not in ("concat", "sum", "mean"):
                raise ValueError(f"parallel_combine must be one of ('concat','sum','mean'), got {self.parallel_combine}")

    def _init_alpha_params(self, specs: List[HeadSpec]) -> None:
        for spec in specs:
            if spec.alpha_mode == "off":
                continue
            if spec.alpha_mode == "soft":
                p = nn.Parameter(torch.tensor([spec.alpha_init, 0.0], dtype=torch.float32))
            elif spec.alpha_mode == "gate":
                p = nn.Parameter(torch.tensor([spec.alpha_init], dtype=torch.float32))
            else:
                raise ValueError(f"alpha_mode must be one of ('off','gate','soft'), got '{spec.alpha_mode}'")
            p.requires_grad_(bool(spec.alpha_trainable))
            self._alphas[spec.name] = p

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

    def _infer_combine(self, spec: HeadSpec, out: Any) -> CombineMode:
        """
        Safer 'auto' inference:
          - 'invert' only if:
              (y: Tensor, state: non-Tensor) AND head has callable .invert
          - 'add' if both outputs are tensors
          - else 'none'

        Recommendation:
          - For RevIN-like heads whose state is tensor/tuple-of-tensors, set combine='invert' explicitly.
        """
        if spec.combine != "auto":
            return spec.combine

        if self._is_2tuple(out):
            a, b = out[0], out[1]
            if isinstance(a, torch.Tensor):
                has_invert = hasattr(spec.head, "invert") and callable(getattr(spec.head, "invert"))
                if has_invert and not isinstance(b, torch.Tensor):
                    return "invert"
                if isinstance(b, torch.Tensor):
                    return "add"

        return "none"

    def _get_or_build_add_proj(
        self,
        name: str,
        in_dim: int,
        out_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        if name in self.add_projs:
            return self.add_projs[name].to(device=device, dtype=dtype)

        if in_dim == out_dim:
            proj = nn.Identity()
        else:
            proj = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

        self.add_projs[name] = proj
        return proj.to(device=device, dtype=dtype)

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

    # ──────────────────────────────────────────────────────────────────────────
    # α mixing primitives
    # ──────────────────────────────────────────────────────────────────────────
    def _alpha_weights(self, spec: HeadSpec) -> AlphaWeights:
        if not self.enable_nas or spec.alpha_mode == "off":
            return None, None

        a = self._alphas[spec.name]
        tau = max(self._EPS, float(self.alpha_temperature))

        if spec.alpha_mode == "soft":
            w = F.softmax(a / tau, dim=0)
            return w[0], w[1]

        g = torch.sigmoid(a[0] / tau)
        return g, 1.0 - g

    @staticmethod
    def _straight_through_hard_gate(g: torch.Tensor) -> torch.Tensor:
        g_hard = (g > 0.5).float()
        return g_hard.detach() - g.detach() + g

    def _compute_effective_weight(
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

        g_used = self._straight_through_hard_gate(w_head)
        return g_used, 1.0 - g_used

    # ──────────────────────────────────────────────────────────────────────────
    # Ablation-friendly enable/disable API
    # ──────────────────────────────────────────────────────────────────────────
    def _check_head_exists(self, name: str) -> None:
        if name not in self._enabled:
            available = list(self._enabled.keys())
            raise KeyError(f"Unknown head name: '{name}'. Available: {available}")

    @torch.no_grad()
    def enable_head(self, name: str) -> None:
        self._check_head_exists(name)
        self._enabled[name] = True

    @torch.no_grad()
    def disable_head(self, name: str) -> None:
        self._check_head_exists(name)
        self._enabled[name] = False

    @torch.no_grad()
    def set_head_enabled(self, name: str, enabled: bool) -> None:
        self._check_head_exists(name)
        self._enabled[name] = bool(enabled)

    @torch.no_grad()
    def enable_all(self) -> None:
        for k in self._enabled:
            self._enabled[k] = True

    @torch.no_grad()
    def disable_all(self) -> None:
        for k in self._enabled:
            self._enabled[k] = False

    def enabled_report(self) -> Dict[str, bool]:
        return dict(self._enabled)

    @contextmanager
    def temporary_disable(self, *names: str) -> Iterator[None]:
        previous_states: Dict[str, bool] = {}
        for name in names:
            self._check_head_exists(name)
            previous_states[name] = self._enabled[name]
            self._enabled[name] = False
        try:
            yield
        finally:
            for name, prev in previous_states.items():
                self._enabled[name] = prev

    @contextmanager
    def only_heads(self, *names: str) -> Iterator[None]:
        for name in names:
            self._check_head_exists(name)
        previous_states = dict(self._enabled)
        self.disable_all()
        for name in names:
            self._enabled[name] = True
        try:
            yield
        finally:
            self._enabled.update(previous_states)

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────
    def forward_pre(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [B,T,F], got {tuple(x.shape)}")

        self._last_base_dim = int(x.size(-1))

        # Stage A: parallel feature augmentation (optional)
        if self.composer_mode in ("parallel", "hybrid"):
            x_aug, st_parallel = self._forward_parallel_stage(x)
        else:
            x_aug, st_parallel = x, []

        # Stage B: serial reversible stage (optional)
        if self.composer_mode in ("serial", "hybrid"):
            x_out, st_serial = self._forward_serial_stage(x_aug)
        else:
            x_out, st_serial = x_aug, []

        # Inverse only needs serial state; still return full run_state for debugging
        run_state = st_parallel + st_serial
        return x_out, run_state

    def forward(
        self,
        x: torch.Tensor,
        encoder: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        out, state = self.forward_pre(x)
        if encoder is not None:
            out = encoder(out)
        return out, state

    # ──────────────────────────────────────────────────────────────────────────
    # Parallel stage (feature augmentation)
    # ──────────────────────────────────────────────────────────────────────────
    def _forward_parallel_stage(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        run_state: List[Dict[str, Any]] = []
        branches: List[torch.Tensor] = []

        for spec in self.parallel_meta:
            if not self._enabled.get(spec.name, True):
                run_state.append({"name": spec.name, "combine": "disabled", "stage": "parallel"})
                continue

            out = spec.head(x)
            if not isinstance(out, torch.Tensor):
                raise RuntimeError(
                    f"[{spec.name}] parallel stage requires head(x) -> Tensor, got {type(out).__name__}"
                )

            # In parallel stage, we treat everything as 'none' (forward-only).
            # You can still do α mixing, but only if shapes match x.
            y = out

            w_head, w_skip = self._alpha_weights(spec)
            hardened = self._hardened_choice.get(spec.name, None)

            rec: Dict[str, Any] = {
                "name": spec.name,
                "combine": "none",
                "stage": "parallel",
                "base_dim": self._last_base_dim,
            }

            if w_head is not None:
                if y.shape != x.shape:
                    raise RuntimeError(
                        f"[{spec.name}] parallel α mixing requires head(x) shape == x shape. "
                        f"Got head(x)={tuple(y.shape)} vs x={tuple(x.shape)}. "
                        "Either disable α for this head or make it output same feature dim."
                    )
                eff_head, eff_skip = self._compute_effective_weight(
                    spec, w_head, w_skip, hardened, x.device, x.dtype
                )
                y = y * eff_head + x * eff_skip
                rec["mix_w_head"] = float(eff_head.item())
            else:
                rec["mix_w_head"] = None

            branches.append(y)
            run_state.append(rec)

        if not branches:
            return x, run_state

        if self.parallel_combine == "concat":
            out = torch.cat(branches, dim=-1)
        elif self.parallel_combine in ("sum", "mean"):
            base = branches[0]
            for i, b in enumerate(branches[1:], start=1):
                if b.shape != base.shape:
                    raise RuntimeError(
                        f"parallel_combine='{self.parallel_combine}' requires equal shapes. "
                        f"branch0={tuple(base.shape)}, branch{i}={tuple(b.shape)}"
                    )
            out = torch.stack(branches, dim=0).sum(dim=0)
            if self.parallel_combine == "mean":
                out = out / float(len(branches))
        else:
            raise ValueError(f"Unknown parallel_combine: {self.parallel_combine}")

        return out, run_state

    # ──────────────────────────────────────────────────────────────────────────
    # Serial stage (reversible)
    # ──────────────────────────────────────────────────────────────────────────
    def _forward_serial_stage(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        run_state: List[Dict[str, Any]] = []
        cur = x

        for spec in self.serial_meta:
            if not self._enabled.get(spec.name, True):
                run_state.append({"name": spec.name, "combine": "disabled", "stage": "serial"})
                continue

            out = spec.head(cur)
            combine = self._infer_combine(spec, out)

            w_head, w_skip = self._alpha_weights(spec)
            hardened = self._hardened_choice.get(spec.name, None)

            rec: Dict[str, Any] = {
                "name": spec.name,
                "combine": combine,
                "stage": "serial",
                "base_dim": self._last_base_dim,
            }

            if combine == "invert":
                cur, rec = self._apply_invert(cur, out, spec, w_head, hardened, rec)
            elif combine == "add":
                cur, rec = self._apply_add(cur, out, spec, w_head, w_skip, hardened, rec)
            elif combine == "none":
                cur, rec = self._apply_none(cur, out, spec, w_head, w_skip, hardened, rec)
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
        rec: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not self._is_2tuple(out) or not isinstance(out[0], torch.Tensor):
            raise RuntimeError(
                f"[{spec.name}] expected (y: Tensor, state: Any) for 'invert', got {type(out).__name__}"
            )
        y, state = out[0], out[1]

        if w_head is None:
            rec.update({"state": state, "head_ref": spec.head, "gate_on": True})
            return y, rec

        if spec.alpha_mode == "soft":
            raise ValueError(f"[{spec.name}] alpha_mode='soft' invalid for combine='invert' (use 'gate').")

        if hardened is not None:
            gate_on = bool(hardened)
            g_used = torch.tensor(float(gate_on), device=cur.device, dtype=cur.dtype)
        else:
            g_used = self._straight_through_hard_gate(w_head)

        cur2 = y * g_used + cur * (1.0 - g_used)

        rec.update(
            {
                "state": state,
                "head_ref": spec.head,
                "gate_on": bool((g_used > 0.5).item()),
                "gate_value": float(w_head.item()) if torch.is_tensor(w_head) else None,
            }
        )
        return cur2, rec

    def _apply_add(
        self,
        cur: torch.Tensor,
        out: Any,
        spec: HeadSpec,
        w_head: Optional[torch.Tensor],
        w_skip: Optional[torch.Tensor],
        hardened: Optional[bool],
        rec: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not self._both_tensors(out):
            raise RuntimeError(
                f"[{spec.name}] expected (main: Tensor, carry: Tensor) for 'add', got {type(out).__name__}"
            )
        main, carry = out

        if self.stop_gradient_on_carry:
            carry = carry.detach()

        rec["carry_shape"] = tuple(carry.shape)

        # Build projection NOW (best-effort), not inside inverse.
        if spec.add_project:
            target_dim = (
                int(self.fixed_output_dim)
                if self.fixed_output_dim is not None
                else int(rec["base_dim"])
            )
            _ = self._get_or_build_add_proj(
                name=spec.name,
                in_dim=int(carry.size(-1)),
                out_dim=target_dim,
                device=cur.device,
                dtype=cur.dtype,
            )

        if w_head is None:
            rec.update({"carry": carry, "mix_w_head": None})
            return main, rec

        eff_head, eff_skip = self._compute_effective_weight(
            spec, w_head, w_skip, hardened, cur.device, cur.dtype
        )

        cur2 = main * eff_head + cur * eff_skip
        scale_carry = eff_head if spec.weight_carry else torch.ones_like(eff_head)

        rec.update({"carry": carry * scale_carry, "mix_w_head": float(eff_head.item())})
        return cur2, rec

    def _apply_none(
        self,
        cur: torch.Tensor,
        out: Any,
        spec: HeadSpec,
        w_head: Optional[torch.Tensor],
        w_skip: Optional[torch.Tensor],
        hardened: Optional[bool],
        rec: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not isinstance(out, torch.Tensor):
            raise RuntimeError(
                f"[{spec.name}] expected single Tensor for 'none', got {type(out).__name__}"
            )
        y = out

        if w_head is None:
            rec["mix_w_head"] = None
            return y, rec

        eff_head, eff_skip = self._compute_effective_weight(
            spec, w_head, w_skip, hardened, cur.device, cur.dtype
        )
        cur2 = y * eff_head + cur * eff_skip
        rec["mix_w_head"] = float(eff_head.item())
        return cur2, rec

    # ──────────────────────────────────────────────────────────────────────────
    # Inverse (only serial stage)
    # ──────────────────────────────────────────────────────────────────────────
    def inverse_post(self, y: torch.Tensor, run_state: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Undo SERIAL stage in reverse.
        Parallel stage is never inverted (feature augmentation).
        """
        if self.composer_mode == "parallel":
            return y

        cur = y
        # Only serial stage records are invertible; reverse them.
        serial_states = [s for s in run_state if s.get("stage") == "serial"]
        for state in reversed(serial_states):
            mode = state["combine"]
            name = state["name"]

            if mode in ("disabled", "none"):
                continue
            if mode == "add":
                cur = self._inverse_add(cur, state, name)
            elif mode == "invert":
                cur = self._inverse_invert(cur, state, name)
            else:
                raise ValueError(f"Unknown combine mode during inverse: '{mode}'")
        return cur

    def _inverse_add(self, cur: torch.Tensor, state: Dict[str, Any], name: str) -> torch.Tensor:
        carry = state["carry"]

        if carry.shape[1] != cur.shape[1]:
            carry = self._align_time(carry, cur.shape[1])

        spec = self._spec_by_name.get(name)
        if spec is not None and spec.add_project:
            target_dim = (
                int(self.fixed_output_dim)
                if self.fixed_output_dim is not None
                else int(state.get("base_dim", cur.size(-1)))
            )
            proj = self._get_or_build_add_proj(
                name=name,
                in_dim=int(carry.size(-1)),
                out_dim=target_dim,
                device=cur.device,
                dtype=cur.dtype,
            )
            carry = proj(carry)

        if carry.size(-1) != cur.size(-1):
            raise RuntimeError(
                f"[{name}] carry feature dim {carry.size(-1)} != current {cur.size(-1)}. "
                "Set add_project=True or provide custom_add_proj or set output_dim."
            )

        return cur + carry

    def _inverse_invert(self, cur: torch.Tensor, state: Dict[str, Any], name: str) -> torch.Tensor:
        head: Optional[nn.Module] = state.get("head_ref")
        st = state.get("state")
        if head is None or st is None:
            raise RuntimeError(f"[{name}] missing head_ref/state for invert (corrupted run_state).")

        gate_on = state.get("gate_on", True)
        if not gate_on:
            return cur

        if not hasattr(head, "invert") or not callable(getattr(head, "invert")):
            raise RuntimeError(f"[{name}] combine='invert' but head {type(head).__name__} has no callable .invert()")

        return head.invert(cur, st)

    # ──────────────────────────────────────────────────────────────────────────
    # NAS utilities
    # ──────────────────────────────────────────────────────────────────────────
    def arch_parameters(self) -> Iterator[nn.Parameter]:
        if not self.enable_nas:
            return iter([])
        return (p for p in self._alphas.values() if p.requires_grad)

    def weight_parameters(self) -> Iterator[nn.Parameter]:
        if not self.enable_nas:
            return self.parameters()
        alpha_ids = {id(p) for p in self._alphas.values()}
        return (p for p in self.parameters() if id(p) not in alpha_ids)

    def alpha_report(self) -> Dict[str, Dict[str, Any]]:
        if not self.enable_nas:
            return {}

        rep: Dict[str, Dict[str, Any]] = {}
        for spec in (self.parallel_meta + self.serial_meta):
            d: Dict[str, Any] = {
                "mode": spec.alpha_mode,
                "enabled": self._enabled.get(spec.name, True),
                "hardened": self._hardened_choice.get(spec.name),
            }
            if spec.alpha_mode == "off" or spec.name not in self._alphas:
                rep[spec.name] = d
                continue

            w_head, w_skip = self._alpha_weights(spec)
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
        for spec in (self.parallel_meta + self.serial_meta):
            if spec.alpha_mode == "off" or spec.name not in self._alphas:
                self._hardened_choice[spec.name] = None
                continue
            w_head, _ = self._alpha_weights(spec)
            decision = bool(w_head >= threshold)
            self._hardened_choice[spec.name] = decision
            decisions[spec.name] = decision
        return decisions

    @torch.no_grad()
    def clear_discretization_(self) -> None:
        for spec in (self.parallel_meta + self.serial_meta):
            self._hardened_choice[spec.name] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Warmup (forces add_projs creation before optimizer)
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
        self.eval()
        out, st = self.forward_pre(example_x)

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

    # ──────────────────────────────────────────────────────────────────────────
    # Introspection
    # ──────────────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        all_specs = self.parallel_meta + self.serial_meta
        head_info = ", ".join(f"{s.name}({'on' if self._enabled[s.name] else 'off'})" for s in all_specs)
        return (
            f"{self.__class__.__name__}("
            f"mode={self.composer_mode}, "
            f"parallel_combine={self.parallel_combine}, "
            f"heads=[{head_info}], "
            f"enable_nas={self.enable_nas}, "
            f"output_dim={self.fixed_output_dim})"
        )

    def summary(self) -> str:
        lines = [
            f"HeadComposer mode={self.composer_mode} (parallel_combine={self.parallel_combine})",
            f"parallel_heads={len(self.parallel_meta)}, serial_heads={len(self.serial_meta)}",
        ]
        for spec in (self.parallel_meta + self.serial_meta):
            enabled = self._enabled.get(spec.name, True)
            status = "on" if enabled else "off"
            alpha_info = ""
            if self.enable_nas and spec.alpha_mode != "off":
                hardened = self._hardened_choice.get(spec.name)
                if hardened is not None:
                    alpha_info = f" [hardened={'on' if hardened else 'off'}]"
                else:
                    w_head, _ = self._alpha_weights(spec)
                    if w_head is not None:
                        alpha_info = f" [alpha={w_head.item():.3f}]"
            stage = "parallel" if spec in self.parallel_meta else "serial"
            lines.append(f"  {spec.name} ({stage}): {status}, combine={spec.combine}, alpha={spec.alpha_mode}{alpha_info}")
        return "\n".join(lines)

    @property
    def head_names(self) -> List[str]:
        return [s.name for s in (self.parallel_meta + self.serial_meta)]

    @property
    def enabled_heads(self) -> List[str]:
        return [s.name for s in (self.parallel_meta + self.serial_meta) if self._enabled.get(s.name, True)]

    @property
    def disabled_heads(self) -> List[str]:
        return [s.name for s in (self.parallel_meta + self.serial_meta) if not self._enabled.get(s.name, True)]
