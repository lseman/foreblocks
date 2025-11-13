from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as D
import torch.nn as nn


# ---------------------------------------------------------------------
# Helper: wrap outputs into a torch.distributions.Distribution
# Supported:
#   - return a Distribution directly, or
#   - return dict with {'logits': ...} -> Categorical
#   - return dict with {'mean': ..., 'log_std': ...} -> Normal (independent)
# You can extend this to Beta, TanhNormal, etc. as needed.
# ---------------------------------------------------------------------
def _as_distribution(out: Union[D.Distribution, Dict[str, torch.Tensor]]) -> D.Distribution:
    if isinstance(out, D.Distribution):
        return out
    if not isinstance(out, dict):
        raise TypeError("RL block must return a torch.distributions.Distribution or a dict of parameters.")
    if "logits" in out:
        return D.Categorical(logits=out["logits"])
    if "mean" in out and "log_std" in out:
        mean, log_std = out["mean"], out["log_std"]
        std = torch.exp(log_std)
        return D.Independent(D.Normal(mean, std), 1)
    raise ValueError("Unsupported distribution params. Provide {'logits'} or {'mean','log_std'} (or return a Distribution).")

# ---------------------------------------------------------------------
# Generic RL Block Receiver
#   - Calls a user-supplied rl_block(obs, state, context) -> dict
#   - Required keys from rl_block output:
#       {
#         'policy': Distribution OR {'logits': ...} OR {'mean','log_std'},
#         'value':  tensor (optional),
#         'state':  next recurrent state (optional),
#         'aux':    any extra dict (optional)
#       }
#   - This adapter:
#       * handles [B,...], [B,T,...], and [B,T,A,...] layouts
#       * samples actions, computes log_prob & entropy
#       * preserves layout in its outputs
# ---------------------------------------------------------------------
class GenericRLBlockReceiver(nn.Module):
    """
    Adapt arbitrary RL blocks to step/sequence/multi-agent rollouts.

    The wrapped block must implement:
        rl_block(obs, state=None, context=None) -> {
            'policy': Distribution or {'logits': ...} or {'mean','log_std'},
            'value':  Tensor [B, (A,) 1] or [B,...] (optional),
            'state':  any recurrent state for next step (optional),
            'aux':    dict of extras (optional),
        }

    Forward signatures:
        - forward(obs, state=None, context=None)
          where obs is [B,...] or [B,T,...] or [B,T,A,...].

    Returns a dict with keys:
        'actions', 'log_prob', 'entropy', 'value' (if provided), 'state', 'aux'
        Shapes match the leading dims of obs.
    """

    def __init__(self, rl_block: nn.Module):
        super().__init__()
        self.rl_block = rl_block

        # mark capability for upstream wrappers (mirrors your style)
        self._accepts_time_dim = True
        self._accepts_agent_dim = True

    # ----------------------------
    # Public API
    # ----------------------------
    def forward(
        self,
        obs: torch.Tensor,                # [B,...] or [B,T,...] or [B,T,A,...]
        state: Optional[Any] = None,      # recurrent state (per-batch or per-batch×agent)
        context: Optional[Dict] = None,   # optional extra info (masks, time feats, etc.)
        *,
        deterministic: bool = False,
        need_entropy: bool = True,
        need_logprob: bool = True,
    ) -> Dict[str, Any]:
        if obs.dim() == 2 or obs.dim() == 3:
            # Could still be [B, F] or [B, T] scalar; treat by leading dims
            # We'll use a generic rule by checking the first dim as batch.
            return self._forward_step(obs, state, context, deterministic, need_entropy, need_logprob)

        if obs.dim() >= 3:
            # Try to detect [B, T, ...] vs [B, T, A, ...]
            # Heuristic: if obs.dim() >= 4 and the third dim is "agents", we treat it as multi-agent
            # You can force multi-agent by passing context={'is_multi_agent': True}
            is_multi_agent = False
            if context is not None and "is_multi_agent" in context:
                is_multi_agent = bool(context["is_multi_agent"])
            else:
                is_multi_agent = (obs.dim() >= 4)

            if is_multi_agent:
                return self._forward_sequence_multiagent(obs, state, context, deterministic, need_entropy, need_logprob)
            else:
                return self._forward_sequence(obs, state, context, deterministic, need_entropy, need_logprob)

        raise ValueError(f"Unsupported observation rank: {obs.shape}")

    # ----------------------------
    # Internal: Single step [B,...]
    # ----------------------------
    def _forward_step(
        self,
        obs: torch.Tensor,
        state: Optional[Any],
        context: Optional[Dict],
        deterministic: bool,
        need_entropy: bool,
        need_logprob: bool,
    ) -> Dict[str, Any]:
        out = self.rl_block(obs, state=state, context=context)  # user block call

        # Parse policy distribution
        dist = _as_distribution(out["policy"]) if "policy" in out else None
        if dist is None:
            raise ValueError("rl_block must return a 'policy' distribution or params.")

        # Sample / mode
        if deterministic:
            if hasattr(dist, "probs"):  # categorical-like
                actions = dist.probs.argmax(dim=-1)
            elif hasattr(dist, "mean"):
                actions = dist.mean
            else:
                # Fallback: single sample but not stochastic (no perfect 'mode' defined)
                actions = dist.sample()
        else:
            actions = dist.sample()

        # Log prob & entropy
        logp = dist.log_prob(actions) if need_logprob else None
        ent  = dist.entropy() if need_entropy else None

        # Value & next state
        value = out.get("value", None)
        next_state = out.get("state", state)
        aux = out.get("aux", {})

        # Make sure shapes have a trailing singleton dim where needed for consistency
        result = {
            "actions": actions,
            "log_prob": logp,
            "entropy": ent,
            "value": value,
            "state": next_state,
            "aux": aux,
        }
        return result

    # ----------------------------
    # Internal: Sequence [B,T,...]
    # ----------------------------
    def _forward_sequence(
        self,
        obs: torch.Tensor,                # [B, T, ...]
        state: Optional[Any],
        context: Optional[Dict],
        deterministic: bool,
        need_entropy: bool,
        need_logprob: bool,
    ) -> Dict[str, Any]:
        B, T = obs.shape[0], obs.shape[1]

        actions  = []
        logps    = [] if need_logprob else None
        entropies= [] if need_entropy else None
        values   = []

        s = state
        for t in range(T):
            out_t = self._forward_step(
                obs[:, t, ...], s, context, deterministic, need_entropy, need_logprob
            )
            actions.append(out_t["actions"].unsqueeze(1))  # [B,1,*]
            if need_logprob:
                lp = out_t["log_prob"]
                # log_prob for Categorical returns [B] while Normal with Independent returns [B]
                logps.append(lp.unsqueeze(1))  # [B,1]
            if need_entropy:
                ent = out_t["entropy"]
                entropies.append(ent.unsqueeze(1))  # [B,1]
            if out_t["value"] is not None:
                v = out_t["value"]
                if v.dim() == 1:
                    v = v.unsqueeze(-1)  # [B,1]
                values.append(v.unsqueeze(1))  # [B,1,1] or [B,1,Dv]
            s = out_t["state"]

        actions = torch.cat(actions, dim=1)  # [B,T,*]
        logps = torch.cat(logps, dim=1) if need_logprob else None  # [B,T]
        entropies = torch.cat(entropies, dim=1) if need_entropy else None  # [B,T]
        values = torch.cat(values, dim=1) if len(values) > 0 else None     # [B,T,?]

        return {"actions": actions, "log_prob": logps, "entropy": entropies, "value": values, "state": s, "aux": {}}

    # ----------------------------
    # Internal: Sequence + Multi-agent [B,T,A,...]
    #   Flattens B·A for the inner call, iterates over T.
    #   State can be per-agent (e.g., dict of tensors shaped [B,A,...]).
    # ----------------------------
    def _forward_sequence_multiagent(
        self,
        obs: torch.Tensor,                # [B, T, A, ...]
        state: Optional[Any],
        context: Optional[Dict],
        deterministic: bool,
        need_entropy: bool,
        need_logprob: bool,
    ) -> Dict[str, Any]:
        B, T, A = obs.shape[0], obs.shape[1], obs.shape[2]

        # Utilities to reshape state per-agent if it's a tensor;
        # if it's a dict, we try to reshape each tensor field similarly.
        def _reshape_state_for_flat(s):
            if s is None:
                return None
            if torch.is_tensor(s):
                return s.reshape(B * A, *s.shape[2:])
            if isinstance(s, dict):
                r = {}
                for k, v in s.items():
                    if torch.is_tensor(v) and v.shape[:2] == (B, A):
                        r[k] = v.reshape(B * A, *v.shape[2:])
                    else:
                        r[k] = v
                return r
            # if custom structure, assume user handles shape
            return s

        def _reshape_state_from_flat(s_flat):
            if s_flat is None:
                return None
            if torch.is_tensor(s_flat):
                return s_flat.reshape(B, A, *s_flat.shape[1:])
            if isinstance(s_flat, dict):
                r = {}
                for k, v in s_flat.items():
                    if torch.is_tensor(v):
                        r[k] = v.reshape(B, A, *v.shape[1:])
                    else:
                        r[k] = v
                return r
            return s_flat

        actions  = []
        logps    = [] if need_logprob else None
        entropies= [] if need_entropy else None
        values   = []

        s = _reshape_state_for_flat(state)
        # Flatten agents: [B,T,A,...] -> per-step: [B·A, ...]
        for t in range(T):
            x_t = obs[:, t, ...]                   # [B, A, ...]
            x_t = x_t.reshape(B * A, *x_t.shape[2:])
            out_t = self._forward_step(
                x_t, s, context, deterministic, need_entropy, need_logprob
            )

            # actions back to [B, A, *]
            act_t = out_t["actions"].reshape(B, A, *out_t["actions"].shape[1:])
            actions.append(act_t.unsqueeze(1))  # [B,1,A,*]

            if need_logprob:
                lp = out_t["log_prob"].reshape(B, A, *(() if out_t["log_prob"].dim()==1 else out_t["log_prob"].shape[1:]))
                logps.append(lp.unsqueeze(1))   # [B,1,A,(...=1)]

            if need_entropy:
                ent = out_t["entropy"].reshape(B, A, *(() if out_t["entropy"].dim()==1 else out_t["entropy"].shape[1:]))
                entropies.append(ent.unsqueeze(1))  # [B,1,A,(...=1)]

            if out_t["value"] is not None:
                v = out_t["value"]
                if v.dim() == 1:
                    v = v.unsqueeze(-1)  # [B·A,1]
                v = v.reshape(B, A, *v.shape[1:])
                values.append(v.unsqueeze(1))  # [B,1,A,?]

            s = out_t["state"]  # still flat; updated recurrent state for next step

        actions = torch.cat(actions, dim=1)  # [B,T,A,*]
        logps = torch.cat(logps, dim=1) if need_logprob else None        # [B,T,A,(1)]
        entropies = torch.cat(entropies, dim=1) if need_entropy else None# [B,T,A,(1)]
        values = torch.cat(values, dim=1) if len(values) > 0 else None   # [B,T,A,?]

        next_state = _reshape_state_from_flat(s)
        return {"actions": actions, "log_prob": logps, "entropy": entropies, "value": values, "state": next_state, "aux": {}}
