"""NCO environment protocol.

Any environment plugging into this framework must satisfy:

    obs = env.reset()          → dict (see NCO_OBS_KEYS)
    obs, reward, done = env.step(picks)   # picks: [B, N] float

Obs dict required keys
----------------------
  nodes    [B, N, f_node]   node/item features (combined static+dynamic)
  mask     [B, N]           bool, True = feasible action this step
  context  [B, f_ctx]       global state

Optional legacy keys (accepted by ActorCritic for backward compat)
  task_static   [B, N, f_s]
  task_dynamic  [B, N, f_d]
  glob          [B, f_g]
  candidate     [B, N]      alias for mask
  feas_start    [B, N]
  task_draw     [B, N]
  budget        [B]

Constraint functions
--------------------
A constraint_fn has signature:

    mask = constraint_fn(mask, picks_so_far, obs) -> [B, N] bool

It receives the current feasibility mask, the set of picks chosen in
prior autoregressive steps, and the full obs dict.  It returns a
refined mask.  Multiple constraint_fns can be composed with
`compose_constraints`.
"""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import torch


# Keys every compliant env must populate
NCO_OBS_KEYS = ("nodes", "mask", "context")

# Legacy keys accepted by models but not required (backward compat)
LEGACY_OBS_KEYS = (
    "task_static", "task_dynamic", "glob",
    "candidate", "feas_start", "task_draw", "budget",
)


@runtime_checkable
class NCOEnv(Protocol):
    """Structural protocol for NCO environments."""

    B: int   # batch size
    N: int   # number of nodes/items (may change per problem)

    def reset(self) -> dict: ...
    def step(self, picks: torch.Tensor) -> tuple[dict | None, torch.Tensor, bool]: ...


ConstraintFn = Callable[
    [torch.Tensor, torch.Tensor, dict],  # mask, chosen, obs
    torch.Tensor,                         # refined mask
]


def compose_constraints(*fns: ConstraintFn) -> ConstraintFn:
    """Apply multiple constraint functions in sequence (AND)."""
    def composed(mask: torch.Tensor, chosen: torch.Tensor, obs: dict) -> torch.Tensor:
        for fn in fns:
            mask = fn(mask, chosen, obs)
        return mask
    return composed


def device_constraint(device_col: int = -1) -> ConstraintFn:
    """Block tasks whose device was already used this step.

    Reads device assignment from obs['task_dynamic'][:, :, device_col].
    This was previously hardcoded inside ActorCritic.act().
    """
    def fn(mask: torch.Tensor, chosen: torch.Tensor, obs: dict) -> torch.Tensor:
        td = obs.get("task_dynamic")
        if td is None or td.shape[-1] == 0 or not chosen.any():
            return mask
        B = mask.shape[0]
        dev = mask.device
        task_devs = td[:, :, device_col]  # [B, N]
        # For each batch: find devices used by chosen tasks
        chosen_devs = (task_devs * chosen.float()).sum(1, keepdim=True)  # [B, 1]
        # chosen_devs is only meaningful where chosen is True
        # Build per-batch device id of the single pick already made
        n_chosen = chosen.sum(1)  # [B]
        has_pick = n_chosen > 0
        if not has_pick.any():
            return mask
        # Compute mean device id of chosen tasks (works for single-pick-per-step)
        safe_n = n_chosen.clamp(min=1).float().unsqueeze(1)
        picked_dev = (task_devs * chosen.float()).sum(1, keepdim=True) / safe_n  # [B,1]
        picked_dev = picked_dev.round().long()  # [B, 1]
        # Block any task whose device id matches picked_dev
        blocked = task_devs.long().eq(picked_dev) & has_pick.unsqueeze(1)  # [B, N]
        # Don't block already-chosen tasks (they're blocked by chosen mask already)
        blocked = blocked & (~chosen)
        return mask & (~blocked)
    return fn


def obs_nodes(obs: dict) -> torch.Tensor:
    """Return node features from obs, supporting both new and legacy formats."""
    if "nodes" in obs:
        return obs["nodes"]
    ts = obs.get("task_static")
    td = obs.get("task_dynamic")
    if ts is not None and td is not None:
        return torch.cat([ts, td], dim=-1)
    if ts is not None:
        return ts
    raise KeyError("obs must contain 'nodes' or 'task_static'")


def obs_mask(obs: dict) -> torch.Tensor:
    """Return feasibility mask from obs."""
    if "mask" in obs:
        return obs["mask"]
    if "candidate" in obs:
        return obs["candidate"]
    raise KeyError("obs must contain 'mask' or 'candidate'")


def obs_context(obs: dict) -> torch.Tensor:
    """Return global context from obs."""
    if "context" in obs:
        return obs["context"]
    if "glob" in obs:
        return obs["glob"]
    raise KeyError("obs must contain 'context' or 'glob'")
