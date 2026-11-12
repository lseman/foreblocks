"""Private inverse execution for HeadComposer serial run states.

This stateless mixin replays serial state in reverse using the owning
composer's alignment and projection helpers. It registers no modules or state.
"""

from __future__ import annotations

import torch

from foreblocks.nn.heads.core.types import (
    BaseRunState,
    RunStateList,
    SerialAddState,
    SerialInvertState,
)


class _InverseExecution:
    """Inverse-stage methods using the owning composer's state and helpers."""

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

        if not hasattr(head, "invert") or not callable(head.invert):
            raise RuntimeError(
                f"[{name}] combine='invert' but head {type(head).__name__} has no callable .invert()"
            )

        return head.invert(cur, st)
