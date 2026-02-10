# head_composer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

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
    """
    head: BaseHeadLike
    name: str
    combine: str = "auto"          # "auto" | "invert" | "add" | "none"
    add_project: bool = True       # only used for "add"
    custom_add_proj: Optional[nn.Module] = None  # optional override for "add" projection

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

    Typical:
        specs = [
            HeadSpec(RevINHead(F),              name="revin",  combine="invert"),
            HeadSpec(DecompositionHead(...),    name="decomp", combine="add"),
            HeadSpec(MultiScaleConvHead(F),     name="msconv", combine="none"),
        ]
        composer = HeadComposer(specs)

        x_enc, run_state = composer.forward_pre(x)  # to encoder
        y_hat = model(x_enc, ...)                   # decoder / head output
        y_rec = composer.inverse_post(y_hat, run_state)
    """
    def __init__(
        self,
        specs: List[HeadSpec],
        output_dim: Optional[int] = None,     # optional fixed target feature dim for inverse add()
        stop_gradient_on_carry: bool = False, # detach carry before storing (can stabilize training)
    ):
        super().__init__()
        # Ensure unique names for stable projection lookup
        seen = set()
        for s in specs:
            if s.name in seen:
                raise ValueError(f"HeadSpec names must be unique; duplicated: {s.name}")
            seen.add(s.name)

        self.specs = nn.ModuleList([s.head for s in specs])  # register modules
        self.meta: List[HeadSpec] = specs
        self.fixed_output_dim = output_dim
        self.stop_grad_on_carry = stop_gradient_on_carry

        # Pre-register any custom projections
        self.add_projs = nn.ModuleDict()
        for spec in specs:
            if spec.combine in ("auto", "add") and spec.custom_add_proj is not None:
                self.add_projs[spec.name] = spec.custom_add_proj

    # ---------- helpers ----------

    @staticmethod
    def _is_2tuple(obj: Any) -> bool:
        return isinstance(obj, (tuple, list)) and len(obj) == 2

    @staticmethod
    def _both_tensors(obj: Any) -> bool:
        return isinstance(obj, (tuple, list)) and len(obj) == 2 and all(isinstance(t, torch.Tensor) for t in obj)

    def _infer_combine(self, spec: HeadSpec, out: Any) -> str:
        """
        Auto rules:
          - If (y, state_like) and head has .invert → "invert"
          - Else if (main, carry) both tensors      → "add"
          - Else                                     → "none"
        """
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
        """
        Return a projection module to map carry[..., in_dim] -> [..., out_dim]
        Built lazily on the correct device/dtype and cached under `name`.
        """
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
        """
        Align x [B,T,F] to target_T:
        - if T >= target_T: keep the LAST target_T steps (tail align)
        - if T <  target_T: pad by repeating the last step
        """
        B, T, F = x.shape
        if T == target_T:
            return x
        if T > target_T:
            return x[:, -target_T:, :]
        pad_len = target_T - T
        last = x[:, -1:, :].expand(B, pad_len, F)
        return torch.cat([x, last], dim=1)

    # ---------- public API ----------

    def forward_pre(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Apply all heads in order to produce the encoder input.

        Returns:
            x_out: transformed input for encoder [B,T,F’]
            run_state: list of dicts (per-head) to be used by inverse_post
        """
        run_state: List[Dict[str, Any]] = []
        cur = x
        for spec in self.meta:
            head = spec.head
            out = head(cur)

            combine = self._infer_combine(spec, out)
            state_rec: Dict[str, Any] = {"name": spec.name, "combine": combine}

            if combine == "invert":
                if not self._is_2tuple(out) or not isinstance(out[0], torch.Tensor):
                    raise RuntimeError(f"[{spec.name}] expected (y:Tensor, state:Any) for 'invert' combine.")
                y, state = out[0], out[1]
                cur = y
                state_rec["state"] = state
                state_rec["head_ref"] = head

            elif combine == "add":
                if not self._both_tensors(out):
                    raise RuntimeError(f"[{spec.name}] expected (main:Tensor, carry:Tensor) for 'add' combine.")
                main, carry = out
                if self.stop_grad_on_carry:
                    carry = carry.detach()
                cur = main
                state_rec["carry_shape"] = carry.shape
                state_rec["carry"] = carry

            elif combine == "none":
                if not isinstance(out, torch.Tensor):
                    raise RuntimeError(f"[{spec.name}] expected single tensor for 'none' combine.")
                cur = out

            else:
                raise ValueError(f"Unknown combine mode: {combine}")

            run_state.append(state_rec)

        return cur, run_state

    def inverse_post(self, y: torch.Tensor, run_state: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Undo heads in reverse: add carries, apply .invert(), or passthrough.
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
                head: nn.Module = state["head_ref"]
                st = state["state"]
                if not hasattr(head, "invert") or not callable(getattr(head, "invert")):
                    raise RuntimeError(f"[{name}] combine='invert' but head has no callable .invert().")
                cur = head.invert(cur, st)

            elif mode == "none":
                continue

            else:
                raise ValueError(f"Unknown combine mode during inverse: {mode}")

        return cur