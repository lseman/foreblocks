"""Unified NCO model — Transformer or BipartiteGNN encoder.

Architecture: pluggable encoder + autoregressive pointer decoder.

Encoder types (selected via encoder_type):
    * 'transformer' — TransformerEncoder (default, backward compat)
      Input: task_static, task_dynamic, glob from obs dict
    * 'bipartite' — BipartiteGNN (AM-style for MILP graphs)
      Input: nodes, edge_index, edge_features, n_vars from obs dict

Both encoders output compatible shapes:
    task_emb: [B, N, d_model] node embeddings
    context:  [B, d_model] pooled graph/context embedding

Shared decoder + value head work identically for both.

Key features:
    * Pluggable encoder (Transformer or BipartiteGNN)
    * Single-step pointer attention with STOP action
    * Tanh clipping for pointer attention (Bello et al.)
    * Log-probability masking (feasibility baked into softmax)
    * Value head for PPO baseline
    * Pluggable constraint_fn for problem-specific propagation
    * Self-evaluation (branch-and-score decoding)
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn


# ─── Unified Encoder Interface ───────────────────────────────────────────────


class UnifiedTransformerEncoder(nn.Module):
    """Wraps TransformerEncoder to match BipartiteGNN interface.

    Both encoder types must implement:
        encode(obs) -> (task_emb: [B,N,D], context: [B,D])
        n_vars: int  (set at runtime)
    """

    def __init__(
        self,
        f_static: int,
        f_dynamic: int,
        f_global: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
    ):
        super().__init__()
        self._inner = TransformerEncoder(
            f_static, f_dynamic, f_global, d_model, n_heads, n_layers
        )
        self.d_model = d_model
        self.n_vars = 0

    def encode(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode observation → [B,N,D] + [B,D]."""
        ts = obs["task_static"]
        td = obs["task_dynamic"]
        g = obs["glob"]
        return self._inner(ts, td, g)

    def forward(
        self,
        task_static: torch.Tensor,
        task_dynamic: torch.Tensor,
        glob: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Delegate to inner encoder."""
        return self._inner(task_static, task_dynamic, glob)

    def reset_n_vars(self, n: int):
        self.n_vars = n


class UnifiedBipartiteEncoder(nn.Module):
    """Wraps BipartiteGNNScheduler encoder to match TransformerEncoder interface.

    Extracts node embeddings and graph embedding from bipartite GNN,
    projecting graph embedding to d_model.
    """

    def __init__(self, scheduler):
        super().__init__()
        self._scheduler = scheduler
        self.d_model = scheduler.d_model
        self.n_vars = 0

    def encode(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode graph observation → [B,N,D] + [B,D]."""
        nodes = obs["nodes"]
        edge_index = obs.get("edge_index")
        edge_features = obs.get("edge_features")
        n_vars = obs.get("n_vars", self.n_vars) or self.n_vars
        self.n_vars = n_vars

        B, N, f = nodes.shape
        dev = nodes.device

        h = self._scheduler.node_proj(nodes)

        node_out = torch.zeros(B, N, self.d_model, device=dev)
        graph_emb = torch.zeros(B, self.d_model, device=dev)

        if edge_index is not None:
            if edge_index.ndim == 2:
                edge_index = edge_index.unsqueeze(0).expand(B, -1, -1)
            elif edge_index.shape[0] == 1 and B > 1:
                edge_index = edge_index.expand(B, -1, -1)

            if edge_features is not None:
                if edge_features.ndim == 2:
                    edge_features = edge_features.unsqueeze(0).expand(B, -1, -1)
                elif edge_features.shape[0] == 1 and B > 1:
                    edge_features = edge_features.expand(B, -1, -1)

            for b in range(B):
                ef = edge_features[b] if edge_features is not None else None
                n_h, n_g = self._scheduler.encoder(h[b], edge_index[b], ef, n_vars)
                node_out[b] = n_h
                graph_emb[b] = n_g
        else:
            for b in range(B):
                n_h, n_g = self._scheduler.encoder(
                    h[b], torch.empty((2, 0), device=dev), None, n_vars
                )
                node_out[b] = n_h
                graph_emb[b] = n_g

        graph_emb = self._scheduler.graph_proj(graph_emb)
        return node_out, graph_emb

    def reset_n_vars(self, n: int):
        self.n_vars = n


# ─── Decoder Adapter ─────────────────────────────────────────────────────────


class _DecoderAdapter(nn.Module):
    """Wraps PointerDecoder to match BipartiteDecoder.attention_logits() interface.

    PointerDecoder.logits(context, task_emb, stop) -> [B, N+1]
    BipartiteDecoder.attention_logits(context, node_emb) -> (attn [B,N], stop [B])

    This adapter normalizes to (attn, stop) format.
    """

    def __init__(self, d_model: int = 128, stop_param: Optional[torch.Tensor] = None):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        if stop_param is not None:
            self.register_parameter("stop", nn.Parameter(stop_param))
        else:
            self.stop = nn.Parameter(torch.randn(1, d_model))
        self.gru = nn.GRUCell(d_model, d_model)

    def attention_logits(
        self,
        context: torch.Tensor,
        node_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute pointer logits → (attn [B,N], stop [B])."""
        B = context.shape[0]
        stop_key = self.stop.expand(B, -1)
        logits = self._pointer_logits(context, node_emb, stop_key)
        return logits[:, :-1], logits[:, -1]

    def _pointer_logits(
        self,
        context: torch.Tensor,
        node_emb: torch.Tensor,
        stop_key: torch.Tensor,
    ) -> torch.Tensor:
        """Full pointer logits [B, N+1]."""
        B, N, _ = node_emb.shape
        q = self.W_q(context).unsqueeze(1)
        k = self.W_k(node_emb)
        v = self.W_v(node_emb)
        attn_raw = torch.tanh(q + k + v)
        attn = torch.sum(attn_raw * q, dim=-1).squeeze(1)
        stop_q = self.W_q(context).unsqueeze(1)
        stop_attn = torch.tanh(stop_q + self.W_k(stop_key).unsqueeze(1)).sum(dim=-1)
        return torch.cat([attn, stop_attn], dim=-1)

    def update(self, context: torch.Tensor, picked_tok: torch.Tensor) -> torch.Tensor:
        return self.gru(picked_tok, context)


# ─── Shared Decoder Loop ─────────────────────────────────────────────────────


def _shared_decode_loop(
    obs,
    task_emb: torch.Tensor,
    context_init: torch.Tensor,
    decoder_adapter,
    n_vars: int,
    greedy: bool = False,
    action: Optional[torch.Tensor] = None,
    action_order: Optional[torch.Tensor] = None,
    constraint_fn: Optional[Callable] = None,
    max_starts: Optional[int] = None,
    is_start: Optional[torch.Tensor] = None,
    use_legacy_device: bool = False,
    task_static: Optional[torch.Tensor] = None,
    td: Optional[torch.Tensor] = None,
    propagate_device: bool = False,
) -> dict:
    """Shared autoregressive decoding loop.

    Works with ANY encoder that produces [B,N,D] task_emb + [B,D] context.
    The decoder_adapter normalizes PointerDecoder → BipartiteDecoder interface.

    Context handling: uses simple [B,D] context (Transformer style).
    For BipartiteGNN context ([B,3D]), see BipartiteGNNScheduler.act().
    """
    if max_starts is None:
        max_starts = n_vars

    B, N, D = task_emb.shape
    dev = task_emb.device
    replay = action is not None

    mask = obs.get("mask", torch.ones(B, N, dtype=torch.bool, device=dev))
    var_mask = mask[:, :n_vars]

    feas = obs.get("candidate", mask)
    draw = obs.get("task_draw")
    budget = obs.get("budget")

    chosen = torch.zeros(B, n_vars, dtype=torch.bool, device=dev)
    logp = torch.zeros(B, device=dev)
    ent = torch.zeros(B, device=dev)
    alive = torch.ones(B, dtype=torch.bool, device=dev)
    order_out = torch.full((B, max_starts), n_vars, dtype=torch.long, device=dev)

    order: list[list[int]] = []
    ptr: list[int] = [0] * B
    if replay and action is not None:
        if action_order is not None:
            order_t = action_order.to(dev)
            order = [
                [j for j in order_t[b].tolist() if j < n_vars and action[b, j] > 0.5]
                for b in range(B)
            ]
        else:
            order = [torch.where(action[b] > 0.5)[0].tolist() for b in range(B)]

    context = context_init.detach().clone()
    remaining = (
        budget.clone().detach() if budget is not None else torch.zeros(B, device=dev)
    )

    for step in range(max_starts):
        remaining_mask = var_mask & (~chosen)
        has_cand = remaining_mask.any(1)
        alive = alive & has_cand
        if not alive.any():
            break

        attn_logits, stop_logits = decoder_adapter.attention_logits(
            context, task_emb[:, :n_vars]
        )
        logits = torch.cat([attn_logits, stop_logits.unsqueeze(1)], dim=-1)

        valid = torch.cat(
            [remaining_mask, torch.ones(B, 1, dtype=torch.bool, device=dev)], dim=-1
        )
        masked_logits = logits + valid.float().log()
        masked_logits = torch.where(
            torch.isfinite(masked_logits),
            masked_logits,
            torch.full_like(masked_logits, -1e9),
        )
        dist = torch.distributions.Categorical(logits=masked_logits)

        if not replay and constraint_fn is not None:
            sub_mask = torch.cat(
                [remaining_mask, torch.ones(B, 1, dtype=torch.bool, device=dev)], dim=-1
            )
            new_mask = constraint_fn(sub_mask, chosen, obs)
            var_mask = new_mask[:, :n_vars]

        if replay:
            pick = torch.full((B,), n_vars, dtype=torch.long, device=dev)
            for b in range(B):
                if alive[b] and ptr[b] < len(order[b]):
                    pick[b] = order[b][ptr[b]]
                    ptr[b] += 1
        elif greedy:
            pick = masked_logits.argmax(-1)
        else:
            pick = dist.sample()

        order_out[:, step] = pick
        step_logp = dist.log_prob(pick)
        step_ent = dist.entropy()
        logp = logp + torch.where(alive, step_logp, torch.zeros_like(step_logp))
        ent = ent + torch.where(alive, step_ent, torch.zeros_like(step_ent))

        is_stop = pick.eq(n_vars)
        took = alive & (~is_stop)
        idx = pick.clamp(max=n_vars - 1)

        if took.any():
            taken_b = torch.arange(B, device=dev)[took]
            chosen[taken_b, idx[taken_b]] = True

            if use_legacy_device and propagate_device and td is not None:
                device_col = -1
                if td.shape[-1] > 0:
                    device_used = td[taken_b, idx[taken_b], device_col].long()
                    # Note: device_used not tracked beyond this step for legacy compat
                    # Real device tracking requires persistent state (handled per-encoder)

            if is_start is not None:
                pass  # start tracking handled per-encoder for legacy compat

            if draw is not None and budget is not None:
                remaining = remaining.clone()
                remaining[taken_b] = remaining[taken_b] - draw[taken_b, idx[taken_b]]

            picked_tok = task_emb[taken_b, idx[taken_b]]
            new_ctx = decoder_adapter.update(context[taken_b], picked_tok)
            context = context.clone()
            context[taken_b] = new_ctx

        alive = alive & (~is_stop)

    return {
        "new_starts": chosen.float(),
        "logp": logp,
        "entropy": ent,
        "action_order": order_out,
    }


# ─── Transformer Encoder ─────────────────────────────────────────────────────


class TransformerEncoder(nn.Module):
    """Transformer encoder for task representation.

    Uses global context token + task tokens as input.
    """

    def __init__(
        self,
        f_static: int,
        f_dynamic: int,
        f_global: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(f_static + f_dynamic + f_global, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.glob_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(
        self,
        task_static: torch.Tensor,
        task_dynamic: torch.Tensor,
        glob: torch.Tensor,
    ):
        """Encode tasks and global context.

        Returns:
            task_tok: Task embeddings [B, J, d_model]
            glob_ctx: Global context embedding [B, d_model]
        """
        B, J, _ = task_static.shape
        feats = torch.cat(
            [task_static, task_dynamic, glob.unsqueeze(1).expand(B, J, -1)], dim=-1
        )
        tok = self.proj(feats)
        g = self.glob_token.expand(B, -1, -1)
        h = torch.cat([g, tok], dim=1)
        h_enc = self.enc(h)
        task_tok = h_enc[:, 1:]
        glob_ctx = h_enc[:, 0]
        return task_tok, glob_ctx


# ─── Pointer Decoder ─────────────────────────────────────────────────────────


class PointerDecoder(nn.Module):
    """Pointer decoder with STOP action.

    Single-step pointer attention with tanh clipping.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.stop = nn.Parameter(torch.randn(1, d_model))
        self.gru = nn.GRUCell(d_model, d_model)

    def logits(
        self,
        context: torch.Tensor,
        task_tok: torch.Tensor,
        stop_key: torch.Tensor,
    ):
        """Compute pointer attention logits.

        Uses tanh clipping for numerical stability (Bello et al.).

        Returns:
            logits: [B, J+1] (J task positions + 1 STOP)
        """
        B, J, _ = task_tok.shape
        q = self.W_q(context).unsqueeze(1)
        k = self.W_k(task_tok)
        v = self.W_v(task_tok)
        attn_raw = torch.tanh(q + k + v)
        attn = torch.sum(attn_raw * self.W_q(context).unsqueeze(1), dim=-1).squeeze(1)
        stop_q = self.W_q(context).unsqueeze(1)
        stop_attn = torch.tanh(stop_q + self.W_k(stop_key).unsqueeze(1)).sum(dim=-1)
        return torch.cat([attn, stop_attn], dim=-1)

    def update(
        self,
        context: torch.Tensor,
        picked_tok: torch.Tensor,
    ):
        """Update decoder context with picked token."""
        return self.gru(picked_tok, context)


# ─── Unified Actor-Critic ────────────────────────────────────────────────────


class ActorCritic(nn.Module):
    """Unified Actor-Critic policy with pluggable encoder.

    Accepts either:
        encoder_type='transformer' — TransformerEncoder + PointerDecoder
        encoder_type='bipartite'   — BipartiteGNN encoder + PointerDecoder

    The 'transformer' path is backward-compatible with existing code.
    The 'bipartite' path uses graph-structured input (nodes, edges).

    Args:
        encoder_type: 'transformer' or 'bipartite'
        f_static/f_dynamic/f_global: Transformer encoder feature dims
        f_node: Bipartite encoder node feature dim (for 'bipartite')
        d_model/n_heads/n_layers: shared hidden dimensions
        n_gnn_layers: number of GNN message-passing layers (for 'bipartite')
        propagate_device: legacy device constraint flag
        scheduler: pre-built BipartiteGNNScheduler (for 'bipartite')
    """

    def __init__(
        self,
        encoder_type: str = "transformer",
        # Transformer params
        f_static: int = 0,
        f_dynamic: int = 0,
        f_global: int = 0,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        # Bipartite params
        f_node: int = 0,
        n_gnn_layers: int = 2,
        edge_dim: int = 1,
        # Shared params
        propagate_device: bool = True,
        scheduler: Optional[object] = None,  # pre-built BipartiteGNNScheduler
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.propagate_device = propagate_device

        if encoder_type == "transformer":
            self.encoder = UnifiedTransformerEncoder(
                f_static, f_dynamic, f_global, d_model, n_heads, n_layers
            )
            self._decoder_adapter = _DecoderAdapter(d_model, None)
            self.decoder = PointerDecoder(d_model)
            self._n_vars = 0  # set at runtime
        elif encoder_type == "bipartite":
            if scheduler is not None:
                self.encoder = UnifiedBipartiteEncoder(scheduler)
            else:
                # Build scheduler from params
                from .bipartite_gnn import BipartiteGNNScheduler

                sch = BipartiteGNNScheduler(
                    f_node=f_node,
                    f_global=f_global,
                    d_model=d_model,
                    n_gnn_layers=n_gnn_layers,
                    edge_dim=edge_dim,
                )
                self.encoder = UnifiedBipartiteEncoder(sch)
                self._scheduler_ref = sch  # keep for n_vars
            # Bipartite context is [B, 3D] — project to [B, D] for decoder
            self._ctx_project = nn.Linear(3 * d_model, d_model)
            self._decoder_adapter = _DecoderAdapter(d_model, None)
            self.decoder = PointerDecoder(d_model)
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}. "
                f"Use 'transformer' or 'bipartite'."
            )

        # Value head (takes pooled task + glob = 2*d_model)
        self.value = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def value_of(self, task_emb, context):
        """Compute value estimate for a state."""
        pooled = task_emb.mean(1)
        return self.value(torch.cat([pooled, context], dim=-1)).squeeze(-1)

    # ── Encoder-specific context helpers ─────────────────────────────────

    def _context_transformer(
        self, obs, task_emb, context_init
    ) -> tuple[torch.Tensor, dict]:
        """Prepare context for Transformer encoder (simple [B,D])."""
        return context_init, {
            "is_start": obs.get("feas_start"),
            "use_legacy_device": self.propagate_device,
            "td": obs.get("task_dynamic"),
            "task_static": obs.get("task_static"),
        }

    def _context_bipartite(
        self, obs, task_emb, context_init
    ) -> tuple[torch.Tensor, dict]:
        """Prepare context for BipartiteGNN (concat [graph; first; last])."""
        B = context_init.shape[0]
        D = context_init.shape[1]
        dev = context_init.device

        # BipartiteGNN context: [graph; first_picked; last_picked]
        neg_one = -torch.ones(B, D, device=dev)
        context = torch.cat([context_init, neg_one, neg_one], dim=-1)

        return context, {
            "context_first": neg_one,
            "context_last": neg_one,
            "is_start": obs.get("feas_start"),
            "use_legacy_device": self.propagate_device,
            "td": obs.get("task_dynamic"),
            "task_static": obs.get("task_static"),
        }

    def _update_context_transformer(
        self, context, task_emb, idx, taken_b, decoder_context, n_vars
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """Update context for Transformer: context = gru(picked, context)."""
        picked_tok = task_emb[taken_b, idx]
        new_ctx = self.decoder.update(decoder_context[taken_b], picked_tok)
        context = context.clone()
        context[taken_b] = new_ctx
        return context, None, {}

    def _update_context_bipartite(
        self, context, task_emb, idx, taken_b, context_first, context_last, n_vars
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Update context for Bipartite: context = [graph; first; last]."""
        picked_tok = task_emb[taken_b, idx]
        context_first = context_first.clone()
        if context_first[taken_b].abs().sum() < 1e-3:
            context_first[taken_b] = picked_tok
        context_last = context_last.clone()
        context_last[taken_b] = picked_tok
        # Rebuild context
        context = torch.cat(
            [
                context[:, : context.shape[-1] // 3],  # graph (first third)
                context_first,
                context_last,
            ],
            dim=-1,
        )
        return context, context_first, context_last, {}

    # ── Main act() method ──────────────────────────────────────────────────

    def act(
        self,
        obs,
        max_starts=None,
        action=None,
        action_order=None,
        greedy=False,
        constraint_fn: Optional[Callable] = None,
    ):
        """Build the active set autoregressively.

        If `action` is given, recompute its log-prob/entropy (PPO update path).

        Returns dict with: new_starts[B, n_vars], logp[B], entropy[B], value[B].
        """
        # Encode
        task_emb, context_init = self.encoder.encode(obs)
        n_vars = self.encoder.n_vars or task_emb.shape[1]
        value = self.value_of(task_emb, context_init)

        dev = task_emb.device
        B = task_emb.shape[0]
        if max_starts is None:
            max_starts = n_vars

        # Prepare encoder-specific context
        if self.encoder_type == "bipartite":
            context, ctx_info = self._context_bipartite(obs, task_emb, context_init)
            context_first = ctx_info["context_first"]
            context_last = ctx_info["context_last"]
        else:
            context, ctx_info = self._context_transformer(obs, task_emb, context_init)
            context_first = None
            context_last = None

        replay = action is not None
        budget = obs.get("budget")
        draw = obs.get("task_draw")
        feas = obs.get("candidate", torch.ones(B, n_vars, dtype=torch.bool, device=dev))

        chosen = torch.zeros(B, n_vars, dtype=torch.bool, device=dev)
        start_used = torch.zeros(B, dtype=torch.bool, device=dev)
        logp = torch.zeros(B, device=dev)
        ent = torch.zeros(B, device=dev)
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        order_out = torch.full((B, max_starts), n_vars, dtype=torch.long, device=dev)

        # Replay order
        order: list[list[int]] = []
        ptr: list[int] = [0] * B
        if replay and action is not None:
            if action_order is not None:
                order_t = action_order.to(dev)
                order = [
                    [
                        j
                        for j in order_t[b].tolist()
                        if j < n_vars and action[b, j] > 0.5
                    ]
                    for b in range(B)
                ]
            else:
                order = [torch.where(action[b] > 0.5)[0].tolist() for b in range(B)]

        remaining = (
            budget.clone().detach()
            if budget is not None
            else torch.zeros(B, device=dev)
        )

        # Mask for pointer attention (over variable nodes only)
        mask = obs.get(
            "mask", torch.ones(B, task_emb.shape[1], dtype=torch.bool, device=dev)
        )
        var_mask = mask[:, :n_vars]

        for step in range(max_starts):
            remaining_mask = var_mask & (~chosen)
            has_cand = remaining_mask.any(1)
            alive = alive & has_cand
            if not alive.any():
                break

            # Compute logits
            if self.encoder_type == "bipartite":
                # Bipartite context is [graph; first; last] → 3D, project to D
                ctx_1d = self._ctx_project(context)
                attn, stop = self._decoder_adapter.attention_logits(
                    ctx_1d, task_emb[:, :n_vars]
                )
            else:
                # Transformer context is [B,D] → 1D
                stop_key = self.decoder.stop.expand(B, -1)
                logits_full = self.decoder.logits(
                    context, task_emb[:, :n_vars], stop_key
                )
                attn = logits_full[:, :-1]
                stop = logits_full[:, -1]

            logits = torch.cat([attn, stop.unsqueeze(1)], dim=-1)

            # Mask infeasible actions
            valid = torch.cat(
                [remaining_mask, torch.ones(B, 1, dtype=torch.bool, device=dev)], dim=-1
            )
            masked_logits = logits + valid.float().log()
            masked_logits = torch.where(
                torch.isfinite(masked_logits),
                masked_logits,
                torch.full_like(masked_logits, -1e9),
            )
            dist = torch.distributions.Categorical(logits=masked_logits)

            # Constraint propagation
            if not replay and constraint_fn is not None:
                sub_mask = torch.cat(
                    [remaining_mask, torch.ones(B, 1, dtype=torch.bool, device=dev)],
                    dim=-1,
                )
                new_mask = constraint_fn(sub_mask, chosen, obs)
                var_mask = new_mask[:, :n_vars]

            # Pick action
            if replay:
                pick = torch.full((B,), n_vars, dtype=torch.long, device=dev)
                for b in range(B):
                    if alive[b] and ptr[b] < len(order[b]):
                        pick[b] = order[b][ptr[b]]
                        ptr[b] += 1
            elif greedy:
                pick = masked_logits.argmax(-1)
            else:
                pick = dist.sample()

            order_out[:, step] = pick
            step_logp = dist.log_prob(pick)
            step_ent = dist.entropy()
            logp = logp + torch.where(alive, step_logp, torch.zeros_like(step_logp))
            ent = ent + torch.where(alive, step_ent, torch.zeros_like(step_ent))

            # STOP = n_vars
            is_stop = pick.eq(n_vars)
            took = alive & (~is_stop)
            idx = pick.clamp(max=n_vars - 1)

            if took.any():
                taken_b = torch.arange(B, device=dev)[took]
                chosen[taken_b, idx[taken_b]] = True

                if is_start := ctx_info.get("is_start") is not None:
                    pass  # start tracking simplified

                if draw is not None and budget is not None:
                    remaining = remaining.clone()
                    remaining[taken_b] = (
                        remaining[taken_b] - draw[taken_b, idx[taken_b]]
                    )

                # Update context
                picked_idx = idx[taken_b]  # only indices for tasks that took
                if self.encoder_type == "bipartite":
                    context, context_first, context_last, _ = (
                        self._update_context_bipartite(
                            context,
                            task_emb,
                            picked_idx,
                            taken_b,
                            ctx_info.get("context_first"),
                            ctx_info.get("context_last"),
                            n_vars,
                        )
                    )
                else:
                    context, _, _ = self._update_context_transformer(
                        context,
                        task_emb,
                        picked_idx,
                        taken_b,
                        context,
                        n_vars,
                    )

            alive = alive & (~is_stop)

        return {
            "new_starts": chosen.float(),
            "logp": logp,
            "entropy": ent,
            "value": value,
            "action_order": order_out,
        }

    # ── Self-Evaluation ─────────────────────────────────────────────────────

    def self_eval(
        self,
        obs,
        k: int = 4,
        temperature: float = 1.5,
        return_all: bool = False,
        constraint_fn: Optional[Callable] = None,
    ):
        """Self-evaluation decoding (branch-and-score).

        Generates K independent candidate schedules via stochastic decoding,
        scores each, picks the best.
        """
        ts, td, g = obs["task_static"], obs["task_dynamic"], obs["glob"]
        feas, draw, budget = obs["candidate"], obs["task_draw"], obs["budget"]
        is_start = obs["feas_start"]
        B, J, _ = ts.shape
        dev = ts.device
        max_starts = J

        # Encode once — shared across all K branches
        if self.encoder_type == "bipartite":
            task_tok, glob_ctx = self.encoder.encode(obs)
        else:
            task_tok, glob_ctx = self.encoder(ts, td, g)
        stop_key = self.decoder.stop.expand(B, -1)

        # Flatten K×B for vectorized decoding
        KB = k * B
        feas_flat = feas.unsqueeze(0).expand(k, B, J).reshape(KB, J)
        draw_flat = draw.unsqueeze(0).expand(k, B, J).reshape(KB, J)
        is_start_flat = is_start.unsqueeze(0).expand(k, B, J).reshape(KB, J)
        task_tok_flat = (
            task_tok.unsqueeze(0)
            .expand(k, B, J, task_tok.shape[-1])
            .reshape(KB, J, task_tok.shape[-1])
        )
        budget_flat = budget.unsqueeze(0).expand(k, B).reshape(KB)

        logp_all = torch.zeros(KB, device=dev)
        ent_all = torch.zeros(KB, device=dev)
        alive_all = torch.ones(KB, dtype=torch.bool, device=dev)
        chosen_all = torch.zeros(KB, J, dtype=torch.bool, device=dev)
        order_all = torch.full((KB, max_starts), J, dtype=torch.long, device=dev)
        start_used_all = torch.zeros(KB, dtype=torch.bool, device=dev)
        context_all = (
            glob_ctx.unsqueeze(0)
            .expand(k, B, glob_ctx.shape[-1])
            .reshape(KB, glob_ctx.shape[-1])
        )
        remaining_all = budget_flat

        for step in range(max_starts):
            block_starts = is_start_flat & start_used_all.unsqueeze(1)
            cand = (
                feas_flat
                & (~chosen_all)
                & (~block_starts)
                & (draw_flat <= remaining_all.unsqueeze(1) + 1e-6)
            )

            has_cand = cand.any(1)
            alive_all = alive_all & has_cand
            if not alive_all.any():
                break

            stop_key_flat = (
                stop_key.unsqueeze(0)
                .expand(k, B, stop_key.shape[-1])
                .reshape(KB, stop_key.shape[-1])
            )
            logits_batched = self.decoder.logits(
                context_all, task_tok_flat, stop_key_flat
            )

            valid_all = torch.cat(
                [cand, torch.ones(KB, 1, dtype=torch.bool, device=dev)], 1
            )
            logits_masked = logits_batched + valid_all.float().log()

            logits_scaled = logits_masked / max(temperature, 1e-6)
            logits_scaled = torch.where(
                torch.isfinite(logits_scaled),
                logits_scaled,
                torch.full_like(logits_scaled, -1e9),
            )

            dists = torch.distributions.Categorical(logits=logits_scaled)
            picks = dists.sample()
            order_all[:, step] = picks

            step_logp = dists.log_prob(picks)
            step_ent = dists.entropy()
            logp_all = logp_all + torch.where(
                alive_all, step_logp, torch.zeros_like(step_logp)
            )
            ent_all = ent_all + torch.where(
                alive_all, step_ent, torch.zeros_like(step_ent)
            )

            is_stop = picks.eq(J)
            took = alive_all & (~is_stop)
            idx = picks.clamp(max=J - 1)

            if took.any():
                taken_b = torch.arange(KB, device=dev)[took]
                chosen_all[taken_b, idx[taken_b]] = True

                picked_is_start = torch.zeros(KB, dtype=torch.bool, device=dev)
                picked_is_start[taken_b] = is_start_flat[taken_b, idx[taken_b]].bool()
                start_used_all = start_used_all | picked_is_start

                remaining_all = remaining_all.clone()
                remaining_all[taken_b] = (
                    remaining_all[taken_b] - draw_flat[taken_b, idx[taken_b]]
                )

                picked_tok = task_tok_flat[taken_b, idx[taken_b]]
                new_ctx = self.decoder.update(context_all[taken_b], picked_tok)
                context_all = context_all.clone()
                context_all[taken_b] = new_ctx

            alive_all = alive_all & (~is_stop)

        # Score: value with logp as tiebreaker
        tk = (
            task_tok.unsqueeze(0)
            .expand(k, B, J, task_tok.shape[-1])
            .reshape(KB, J, task_tok.shape[-1])
        )
        gc = (
            glob_ctx.unsqueeze(0)
            .expand(k, B, glob_ctx.shape[-1])
            .reshape(KB, glob_ctx.shape[-1])
        )
        val_flat = self.value_of(tk, gc)
        value_all = val_flat.reshape(k, B)
        logp_all = logp_all.reshape(k, B)
        ent_all = ent_all.reshape(k, B)

        val_spread = value_all.max(dim=0).values - value_all.min(dim=0).values
        val_min = value_all.min(dim=0).values.unsqueeze(0)
        val_max = value_all.max(dim=0).values.unsqueeze(0)
        val_range = (val_max - val_min).clamp(min=1e-6)
        val_norm = (value_all - val_min) / val_range

        logp_score = -logp_all
        # priority is feature 0 in ONTSEnv task_static.
        task_score_col = 0
        task_score = ts[:, :, task_score_col].clamp(min=0)
        proxy = (
            chosen_all.view(k, B, J).float() * task_score.unsqueeze(0).expand(k, B, J)
        ).sum(dim=-1)
        proxy_min = proxy.min(dim=0).values.unsqueeze(0)
        proxy_range = (proxy.max(dim=0).values.unsqueeze(0) - proxy_min).clamp(min=1e-6)
        proxy_norm = (proxy - proxy_min) / proxy_range

        confidence = (val_spread / (val_spread + 1.0)).clamp(max=1.0)
        alpha = confidence.unsqueeze(0)
        score_all = (
            0.7 * proxy_norm
            + 0.2 * alpha * val_norm
            + 0.1 * (1 - alpha) * 0.01 * logp_score
        )

        best_idx = score_all.argmax(dim=0)
        b_idx = torch.arange(B, device=dev)
        best_logp = logp_all[best_idx, b_idx]
        best_ent = ent_all[best_idx, b_idx]
        best_val = value_all[best_idx, b_idx]
        flat_idx = best_idx * B + b_idx
        best_starts = chosen_all.view(-1, J)[flat_idx].float()
        best_order = order_all.view(k, B, max_starts)[best_idx, b_idx]

        result = {
            "new_starts": best_starts,
            "logp": best_logp,
            "entropy": best_ent,
            "value": best_val,
            "action_order": best_order,
        }
        if return_all:
            result["all_starts"] = chosen_all.view(k, B, J).float()
            result["all_values"] = value_all
            result["all_logps"] = logp_all
            result["best_branch"] = best_idx
            result["all_orders"] = order_all.view(k, B, max_starts)
        return result
