# head_composer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# If you have a BaseHead interface, we’ll accept either BaseHead or nn.Module.
BaseHeadLike = Union[nn.Module]  # or your foreblocks.core.BaseHead


@dataclass
class HeadSpec:
    """
    Describe how to combine/invert a head.

    combine:
      - "auto"  : infer behavior:
          * (y, state) and hasattr(head, 'invert') -> reversible
          * (main, carry) with no invert -> add back
          * y only -> none
      - "invert": force reversible; head must return (y, state) and implement invert(y, state)
      - "add"   : force split-add; head must return (main, carry)
      - "none"  : force passthrough; head must return y
    add_project:
      - If True and combine == "add", will project carry to match output_dim before adding back.
    """
    head: BaseHeadLike
    name: str
    combine: str = "auto"          # "auto" | "invert" | "add" | "none"
    add_project: bool = True       # only used for "add"
    # Optional per-head output projection override when combine == "add"
    custom_add_proj: Optional[nn.Module] = None


class HeadComposer(nn.Module):
    """
    Chains multiple preprocessing heads before the encoder and provides an inverse
    to map decoder outputs back to the original space.

    Usage:
        specs = [
            HeadSpec(RevINHead(F),          name="revin", combine="invert"),
            HeadSpec(DecompositionBlock(...), name="decomp", combine="add"),
            HeadSpec(MultiScaleConvHead(F), name="msconv", combine="none"),
        ]
        composer = HeadComposer(specs, output_dim=F_out)

        x_enc, run_state = composer.forward_pre(x)  # feed to encoder
        y_hat = model(x_enc, ...)                   # decoder output
        y_rec = composer.inverse_post(y_hat, run_state)  # add trend back, invert RevIN, etc.
    """
    def __init__(
        self,
        specs: List[HeadSpec],
        output_dim: Optional[int] = None,     # decoder output feature dim; needed if any "add" needs projection
        stop_gradient_on_carry: bool = False, # detach carry before storing (stabilizes some trainings)
    ):
        super().__init__()
        self.specs = nn.ModuleList([s.head for s in specs])  # register modules
        self.meta: List[HeadSpec] = specs
        self.output_dim = output_dim
        self.stop_grad_on_carry = stop_gradient_on_carry

        # Create per-head add-projections if needed
        self.add_projs = nn.ModuleDict()
        for spec in specs:
            if spec.combine in ("auto", "add"):
                # We'll instantiate projection lazily when we first see shapes,
                # unless user passed spec.custom_add_proj.
                if spec.custom_add_proj is not None:
                    self.add_projs[spec.name] = spec.custom_add_proj

    @staticmethod
    def _is_tuple_of_tensors(obj: Any) -> bool:
        return isinstance(obj, (tuple, list)) and len(obj) == 2 and all(isinstance(t, torch.Tensor) for t in obj)

    def _infer_combine(self, spec: HeadSpec, out: Any) -> str:
        if spec.combine != "auto":
            return spec.combine
        # auto rules
        if self._is_tuple_of_tensors(out):
            # If head has invert() -> likely reversible (returns (y, state) or (y, ctx))
            if hasattr(spec.head, "invert") and callable(getattr(spec.head, "invert")):
                return "invert"
            # Else treat as split-add
            return "add"
        # Single tensor
        return "none"

    @torch.no_grad()
    def _maybe_build_add_proj(self, name: str, in_dim: int, out_dim: int):
        # Create a projection if not present and needed
        if name in self.add_projs:
            return  # user provided or already built
        if in_dim != out_dim:
            proj = nn.Linear(in_dim, out_dim)
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
            self.add_projs[name] = proj
        else:
            # Identity as a placeholder (no-op)
            self.add_projs[name] = nn.Identity()

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
                # Expect (y, state)
                if not self._is_tuple_of_tensors(out) and not (isinstance(out, (tuple, list)) and len(out) == 2):
                    raise RuntimeError(f"[{spec.name}] expected (y, state) for 'invert' combine.")
                y, state = out
                cur = y
                state_rec["state"] = state
                state_rec["head_ref"] = head  # need to call invert later

            elif combine == "add":
                # Expect (main, carry)
                if not self._is_tuple_of_tensors(out):
                    raise RuntimeError(f"[{spec.name}] expected (main, carry) for 'add' combine.")
                main, carry = out
                if self.stop_grad_on_carry:
                    carry = carry.detach()
                cur = main
                state_rec["carry"] = carry
                # Prepare projection if necessary (lazy, using first batch)
                if self.output_dim is not None:
                    in_dim = carry.size(-1)
                    self._maybe_build_add_proj(spec.name, in_dim, self.output_dim)

            elif combine == "none":
                # Expect single tensor
                if not isinstance(out, torch.Tensor):
                    raise RuntimeError(f"[{spec.name}] expected single tensor for 'none' combine.")
                cur = out

            else:
                raise ValueError(f"Unknown combine mode: {combine}")

            run_state.append(state_rec)

        return cur, run_state

    def inverse_post(self, y: torch.Tensor, run_state: List[Dict[str, Any]]) -> torch.Tensor:
        cur = y
        for state in reversed(run_state):
            mode = state["combine"]
            name = state["name"]

            if mode == "add":
                carry = state["carry"]

                # 🔧 1) time-align carry to cur
                if carry.shape[1] != cur.shape[1]:
                    carry = self._align_time(carry, cur.shape[1])

                # 🔧 2) feature projection if needed
                if self.output_dim is not None:
                    proj = self.add_projs[name] if name in self.add_projs else None
                    if proj is None:
                        self._maybe_build_add_proj(name, carry.size(-1), self.output_dim)
                        proj = self.add_projs[name]
                    carry = proj(carry)

                # shapes now match
                cur = cur + carry

            elif mode == "invert":
                head: nn.Module = state["head_ref"]
                st = state["state"]
                cur = head.invert(cur, st)

            elif mode == "none":
                continue

            else:
                raise ValueError(f"Unknown combine mode during inverse: {mode}")

        return cur

    def _align_time(self, x: torch.Tensor, target_T: int) -> torch.Tensor:
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
        # T < target_T → pad with last value
        pad_len = target_T - T
        last = x[:, -1:, :].expand(B, pad_len, F)
        return torch.cat([x, last], dim=1)
