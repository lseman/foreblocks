# head_composer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.ui_aux.node_spec import node

BaseHeadLike = Union[nn.Module]  # or your foreblocks.core.BaseHead


@dataclass
class HeadSpec:
    """
    Describe how to combine/invert a head.

    combine:
      - "auto"  : infer behavior per output:
          * (y, state) and hasattr(head, 'invert') -> "invert"
          * (main, carry) and no invert           -> "add"
          * y only                                -> "none"
      - "invert": force reversible; head must return (y, state) and implement invert(y, state)
      - "add"   : force split-add; head must return (main, carry)
      - "none"  : force passthrough; head must return y
    add_project:
      - If True and combine == "add", project carry to match inverse target feature dim.
        (Target dim is detected at inverse time from `y` unless `output_dim` is given.)

    α (architecture) controls:
      - alpha_mode:
          "off"   : legacy behavior (no α)
          "gate"  : head is gated on/off with straight-through sigmoid (invert-safe)
          "soft"  : y = w_head * head(x) + w_skip * x  (non-invert heads only)
      - alpha_init: initial scalar/logit for the gate/soft weights (default 0.0)
      - alpha_trainable: whether α is trainable
      - weight_carry: when combine="add", also scale the carry by the same head weight
    """
    head: BaseHeadLike
    name: str
    combine: str = "auto"          # "auto" | "invert" | "add" | "none"
    add_project: bool = True
    custom_add_proj: Optional[nn.Module] = None

    # α controls (new)
    alpha_mode: str = "off"        # "off" | "gate" | "soft"
    alpha_init: float = 0.0
    alpha_trainable: bool = True
    weight_carry: bool = True      # only relevant when combine="add"


@node(
    type_id="head_composer",
    name="HeadComposer",
    category="Misc",
    outputs=["head_composer"],
    color="bg-gradient-to-r from-purple-400 to-pink-500",
)
class HeadComposer(nn.Module):
    """
    Chains multiple preprocessing heads before the encoder and provides an inverse
    that maps model outputs back to the original space (add trend back, invert RevIN, etc.).

    Now supports DARTS-style α parameters per head:
      - gate (ST-sigmoid): invert-safe binary routing of a head
      - soft mix: convex blend between skip(x) and head(x) for non-invert heads

    Typical:
        specs = [
            HeadSpec(RevINHead(F), name="revin",  combine="invert", alpha_mode="gate"),
            HeadSpec(DecompositionHead(...), name="decomp", combine="add", alpha_mode="soft"),
            HeadSpec(MultiScaleConvHead(F), name="msconv", combine="none", alpha_mode="soft"),
        ]
        composer = HeadComposer(specs, enable_nas=True, alpha_temperature=1.0)

        # --- Step A: optimize alphas only ---
        opt_alpha = torch.optim.Adam(composer.arch_parameters(), lr=1e-2)

        # --- Step B: freeze alphas; optimize model/heads weights ---
        opt_w = torch.optim.Adam(composer.weight_parameters(), lr=1e-3)
    """
    def __init__(
        self,
        specs: List[HeadSpec],
        output_dim: Optional[int] = None,     # optional fixed target feature dim for inverse add()
        stop_gradient_on_carry: bool = False, # detach carry before storing (can stabilize training)
        alpha_temperature: float = 1.0,       # temperature for gate/soft mixing
        enable_nas: bool = False,             # enable neural architecture search with α parameters
    ):
        super().__init__()
        # Ensure unique names
        seen = set()
        for s in specs:
            if s.name in seen:
                raise ValueError(f"HeadSpec names must be unique; duplicated: {s.name}")
            seen.add(s.name)

        self.specs = nn.ModuleList([s.head for s in specs])  # register modules
        self.meta: List[HeadSpec] = specs
        self.fixed_output_dim = output_dim
        self.stop_gradient_on_carry = stop_gradient_on_carry
        self.alpha_temperature = alpha_temperature
        self.enable_nas = enable_nas

        # Pre-register any custom projections
        self.add_projs = nn.ModuleDict()
        for spec in specs:
            if spec.combine in ("auto", "add") and spec.custom_add_proj is not None:
                self.add_projs[spec.name] = spec.custom_add_proj

        # --- α parameters per head (by name) ---
        # Only create alpha parameters if NAS is enabled
        self._alphas = nn.ParameterDict()
        if self.enable_nas:
            # soft: we use 2 logits [head, skip]; gate: single logit
            for spec in specs:
                if spec.alpha_mode == "off":
                    continue
                if spec.alpha_mode == "soft":
                    p = nn.Parameter(torch.tensor([spec.alpha_init, 0.0], dtype=torch.float32))
                elif spec.alpha_mode == "gate":
                    p = nn.Parameter(torch.tensor([spec.alpha_init], dtype=torch.float32))
                else:
                    raise ValueError(f"alpha_mode must be 'off'|'soft'|'gate', got {spec.alpha_mode}")
                p.requires_grad_(bool(spec.alpha_trainable))
                self._alphas[spec.name] = p

        # record hardened choices after discretization (optional)
        self._hardened_choice: Dict[str, Optional[bool]] = {s.name: None for s in specs}

    # ---------- helpers ----------

    @staticmethod
    def _is_2tuple(obj: Any) -> bool:
        return isinstance(obj, (tuple, list)) and len(obj) == 2

    @staticmethod
    def _both_tensors(obj: Any) -> bool:
        return isinstance(obj, (tuple, list)) and len(obj) == 2 and all(isinstance(t, torch.Tensor) for t in obj)

    def _infer_combine(self, spec: HeadSpec, out: Any) -> str:
        if spec.combine != "auto":
            return spec.combine
        if self._is_2tuple(out):
            if hasattr(spec.head, "invert") and callable(getattr(spec.head, "invert")) and isinstance(out[0], torch.Tensor):
                return "invert"
            if self._both_tensors(out):
                return "add"
            return "none"
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
            proj = self.add_projs[name]
            return proj.to(device=device, dtype=dtype)

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
        B, T, F = x.shape
        if T == target_T:
            return x
        if T > target_T:
            return x[:, -target_T:, :]
        pad_len = target_T - T
        last = x[:, -1:, :].expand(B, pad_len, F)
        return torch.cat([x, last], dim=1)

    # ---------- α mixing primitives ----------

    def _alpha_weights(self, spec: HeadSpec) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns (w_head, w_skip) depending on alpha_mode and enable_nas.
        - NAS disabled or alpha_mode "off": (None, None)
        - soft: softmax([α_head, α_skip]/τ) -> (w_head, w_skip)
        - gate: straight-through sigmoid(α/τ) -> (g, 1-g) but used as binary gate in forward logic
        """
        if not self.enable_nas or spec.alpha_mode == "off":
            return None, None

        a = self._alphas[spec.name]
        tau = self.alpha_temperature

        if spec.alpha_mode == "soft":
            w = F.softmax(a / max(1e-6, tau), dim=0)  # [2]
            return w[0], w[1]

        # gate
        g = torch.sigmoid(a[0] / max(1e-6, tau))  # scalar in (0,1)
        return g, 1.0 - g

    @staticmethod
    def _straight_through_hard_gate(g: torch.Tensor) -> torch.Tensor:
        """
        Convert g∈(0,1) to hard {0,1} with straight-through estimator:
          g_hard = (g>0.5).float() + (g - g.detach())
        """
        g_hard = (g > 0.5).float()
        return g_hard.detach() - g.detach() + g

    # ---------- public API ----------

    def forward_pre(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Apply all heads in order to produce the encoder input.

        Returns:
            x_out: transformed input for encoder [B,T,F']
            run_state: list of dicts (per-head) to be used by inverse_post
        """
        run_state: List[Dict[str, Any]] = []
        cur = x
        for spec in self.meta:
            head = spec.head
            out = head(cur)
            combine = self._infer_combine(spec, out)

            state_rec: Dict[str, Any] = {"name": spec.name, "combine": combine}

            # Resolve α (returns None if NAS is disabled)
            w_head, w_skip = self._alpha_weights(spec)

            # Hard overrides if previously discretized
            hardened = self._hardened_choice.get(spec.name, None)

            if combine == "invert":
                # Expected out = (y:Tensor, state)
                if not self._is_2tuple(out) or not isinstance(out[0], torch.Tensor):
                    raise RuntimeError(f"[{spec.name}] expected (y:Tensor, state:Any) for 'invert' combine.")
                y, state = out[0], out[1]

                # Invert heads must be EITHER on or off to preserve a valid inverse.
                if w_head is None:  # NAS disabled or alpha_mode="off"
                    cur = y
                    state_rec["state"] = state
                    state_rec["head_ref"] = head
                    state_rec["gate_on"] = True
                else:
                    if hardened is not None:
                        gate_on = bool(hardened)
                        g_used = torch.tensor(float(gate_on), device=cur.device, dtype=cur.dtype)
                    else:
                        g = w_head  # scalar
                        g_used = self._straight_through_hard_gate(g)  # ST binary

                    # y_or_skip = g* y + (1-g)*cur   but must be exactly choose for inverse consistency
                    # We use ST hard gate in the forward path:
                    cur = y * g_used + cur * (1.0 - g_used)

                    state_rec["state"] = state
                    state_rec["head_ref"] = head
                    state_rec["gate_on"] = bool((g_used > 0.5).item())
                    state_rec["gate_value"] = float(w_head.item())  # for reporting

            elif combine == "add":
                # Expected out = (main, carry) both tensors
                if not self._both_tensors(out):
                    raise RuntimeError(f"[{spec.name}] expected (main:Tensor, carry:Tensor) for 'add' combine.")
                main, carry = out
                if self.stop_gradient_on_carry:
                    carry = carry.detach()

                if w_head is None:  # NAS disabled or alpha_mode="off"
                    cur = main
                    state_rec["carry_shape"] = carry.shape
                    state_rec["carry"] = carry
                    state_rec["mix_w_head"] = None
                elif spec.alpha_mode == "soft":
                    # convex blend: y = w_head * main + w_skip * cur
                    if hardened is not None:
                        # Hardened as pick-head (True) or skip (False)
                        if hardened:
                            cur = main
                            mix_w_head = torch.tensor(1.0, device=cur.device, dtype=cur.dtype)
                        else:
                            cur = cur  # skip
                            mix_w_head = torch.tensor(0.0, device=cur.device, dtype=cur.dtype)
                    else:
                        cur = main * w_head + cur * w_skip
                        mix_w_head = w_head

                    state_rec["carry_shape"] = carry.shape
                    # Optionally scale carry by same head weight (keeps reconstruction consistent)
                    state_rec["carry"] = carry * (mix_w_head if spec.weight_carry else 1.0)
                    state_rec["mix_w_head"] = float(mix_w_head.item()) if isinstance(mix_w_head, torch.Tensor) else mix_w_head
                else:  # gate
                    if hardened is not None:
                        gate_on = bool(hardened)
                        g_used = torch.tensor(float(gate_on), device=cur.device, dtype=cur.dtype)
                    else:
                        g_used = self._straight_through_hard_gate(w_head)
                    cur = main * g_used + cur * (1.0 - g_used)

                    state_rec["carry_shape"] = carry.shape
                    state_rec["carry"] = carry * (g_used if spec.weight_carry else 1.0)
                    state_rec["mix_w_head"] = float(w_head.item())

            elif combine == "none":
                # Expected out = Tensor
                if not isinstance(out, torch.Tensor):
                    raise RuntimeError(f"[{spec.name}] expected single tensor for 'none' combine.")
                y = out

                if w_head is None:  # NAS disabled or alpha_mode="off"
                    cur = y
                    state_rec["mix_w_head"] = None
                elif spec.alpha_mode == "soft":
                    if hardened is not None:
                        cur = y if hardened else cur
                        mix_w_head = torch.tensor(1.0 if hardened else 0.0, device=cur.device, dtype=cur.dtype)
                    else:
                        cur = y * w_head + cur * w_skip
                        mix_w_head = w_head
                    state_rec["mix_w_head"] = float(mix_w_head.item()) if isinstance(mix_w_head, torch.Tensor) else mix_w_head
                else:  # gate
                    if hardened is not None:
                        gate_on = bool(hardened)
                        g_used = torch.tensor(float(gate_on), device=cur.device, dtype=cur.dtype)
                    else:
                        g_used = self._straight_through_hard_gate(w_head)
                    cur = y * g_used + cur * (1.0 - g_used)
                    state_rec["mix_w_head"] = float(w_head.item())

            else:
                raise ValueError(f"Unknown combine mode: {combine}")

            run_state.append(state_rec)

        return cur, run_state

    def inverse_post(self, y: torch.Tensor, run_state: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Undo heads in reverse: add carries, apply .invert(), or passthrough.

        Note: For 'invert' heads with α gating, we only invert if the forward
        gate was 'on' (recorded as gate_on). This preserves inverse consistency.
        """
        cur = y
        for state in reversed(run_state):
            mode = state["combine"]
            name = state["name"]

            if mode == "add":
                carry = state["carry"]

                # 1) time alignment
                if carry.shape[1] != cur.shape[1]:
                    carry = self._align_time(carry, cur.shape[1])

                # 2) feature projection (if requested)
                if any(s.name == name and s.add_project for s in self.meta):
                    target_dim = self.fixed_output_dim if self.fixed_output_dim is not None else cur.size(-1)
                    proj = self._get_or_build_add_proj(
                        name=name,
                        in_dim=carry.size(-1),
                        out_dim=target_dim,
                        device=cur.device,
                        dtype=cur.dtype,
                    )
                    carry = proj(carry)

                # 3) add back (final guard)
                if carry.size(-1) != cur.size(-1):
                    raise RuntimeError(
                        f"[{name}] carry feature dim {carry.size(-1)} != current {cur.size(-1)} "
                        "and add_project=False or projection not provided."
                    )
                cur = cur + carry

            elif mode == "invert":
                head: nn.Module = state.get("head_ref", None)
                st = state.get("state", None)
                if head is None or st is None:
                    raise RuntimeError(f"[{name}] missing head_ref/state for invert.")
                gate_on = state.get("gate_on", True)
                if gate_on:
                    if not hasattr(head, "invert") or not callable(getattr(head, "invert")):
                        raise RuntimeError(f"[{name}] combine='invert' but head has no callable .invert().")
                    cur = head.invert(cur, st)
                else:
                    # skipped at forward → nothing to invert
                    pass

            elif mode == "none":
                continue

            else:
                raise ValueError(f"Unknown combine mode during inverse: {mode}")

        return cur

    # ---------- α utilities for NAS loops ----------

    def arch_parameters(self):
        """
        Iterable over α parameters (for optimizer step A).
        Returns empty iterator if NAS is disabled.
        """
        if not self.enable_nas:
            return iter([])
        return (p for p in self._alphas.values() if p.requires_grad)

    def weight_parameters(self):
        """Iterable over non-α parameters (for optimizer step B)."""
        alpha_ids = {id(p) for p in self._alphas.values()}
        for m in self.modules():
            for p in m.parameters(recurse=False):
                if id(p) not in alpha_ids:
                    yield p
        # plus recursive children (heads)
        for name, module in self.named_modules():
            if module is self:
                continue
            for p in module.parameters(recurse=False):
                if id(p) not in alpha_ids:
                    yield p

    def alpha_report(self) -> Dict[str, Dict[str, float]]:
        """
        Returns {head_name: {'mode': ..., 'w_head': v, 'w_skip': v, 'hardened': bool_or_None}}
        Returns empty dict if NAS is disabled.
        """
        if not self.enable_nas:
            return {}
        
        rep: Dict[str, Dict[str, float]] = {}
        for spec in self.meta:
            d: Dict[str, Any] = {"mode": spec.alpha_mode, "hardened": self._hardened_choice.get(spec.name, None)}
            if spec.alpha_mode == "off":
                rep[spec.name] = d
                continue
            w_head, w_skip = self._alpha_weights(spec)
            if spec.alpha_mode == "soft":
                d["w_head"] = float(w_head.item())
                d["w_skip"] = float(w_skip.item())
            else:  # gate
                d["p_on"] = float(w_head.item())
            rep[spec.name] = d
        return rep

    @torch.no_grad()
    def discretize_(self, threshold: float = 0.5):
        """
        Harden α decisions in-place:
         - gate: on iff sigmoid(α/τ) >= threshold
         - soft: choose head iff w_head >= threshold, else skip
        Does nothing if NAS is disabled.
        """
        if not self.enable_nas:
            return
        
        for spec in self.meta:
            if spec.alpha_mode == "off":
                self._hardened_choice[spec.name] = None
                continue
            w_head, w_skip = self._alpha_weights(spec)
            if spec.alpha_mode == "soft":
                self._hardened_choice[spec.name] = bool(w_head >= threshold)
            else:
                self._hardened_choice[spec.name] = bool(w_head >= threshold)

    @torch.no_grad()
    def clear_discretization_(self):
        """Clear all hardened choices. Works regardless of NAS state."""
        for spec in self.meta:
            self._hardened_choice[spec.name] = None