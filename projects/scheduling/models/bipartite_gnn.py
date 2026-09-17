"""Bipartite GNN Encoder — AM-style architecture for MILP.

Based on Kool et al. (2019) Attention Model (Section 5.6 of the NCO tutorial):
    GNN encoder → message passing on bipartite graph → pooled graph context
    Pointer decoder → attention between context and node embeddings

Graph structure:
    Variable nodes (v_1 ... v_n) ←→ Constraint nodes (c_1 ... c_m)
    Edges: (v_j, c_i) where A[i,j] ≠ 0, weighted by coefficient value

Key advantages over Transformer encoder:
    * Permutation-invariant (no sequential ordering assumption)
    * Respects problem structure (constraints shape variable representations)
    * Can be re-run each step for dynamic/stochastic settings
    * Message passing propagates dual/shadow-price information
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

# ─── Bipartite GNN Encoder ─────────────────────────────────────────────────


class BipartiteGNN(nn.Module):
    """GNN encoder operating on bipartite MILP graph.

    Two types of message passing layers, applied alternately:
        1. Constraint-to-variable: constraint features → variable updates
           (propagates constraint tightness, slack, shadow prices)
        2. Variable-to-constraint: variable features → constraint updates
           (aggregates variable coefficients, costs, bounds)

    Each layer:
        msg_ji = A[i,j] * h_j          (edge-weighted message)
        h_i' = MLP(concat(h_i, sum_j(msg_ji)), h_old_i)

    After L layers, pool node embeddings → graph embedding (mean).

    Args:
        d_model: hidden dimension
        n_layers: number of message passing layers (pairs of directions)
        edge_dim: edge feature dimension
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 2,
        edge_dim: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.edge_dim = edge_dim
        self.n_layers = n_layers

        # ── Constraint-to-variable layer ──
        # Takes [E, d_model + edge_dim + d_model] → [E, d_model]
        self.c2v = nn.Sequential(
            nn.Linear(2 * d_model + edge_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # ── Variable-to-constraint layer ──
        self.v2c = nn.Sequential(
            nn.Linear(2 * d_model + edge_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Learnable positional encodings (amortized, shared across instances)
        self.max_nodes = 512
        self.pos_enc = nn.Embedding(2 * self.max_nodes, d_model)

        # Graph pooling: mean over all nodes
        # (graph embedding = mean of all node embeddings after message passing)

    def forward(
        self,
        node_features: torch.Tensor,  # [N, d_model] concatenated [v; c]
        edge_index: torch.Tensor,  # [2, E]
        edge_features: Optional[torch.Tensor] = None,  # [E, edge_dim]
        n_vars: int = 0,  # number of variable nodes (rest are constraints)
    ):
        """Run GNN encoder.

        Args:
            node_features: Concatenated [v_1...v_n, c_1...c_m] features [N, d]
            edge_index: [2, E] edge list, src and dst indices in node_features
            edge_features: Per-edge features [E, edge_dim]
            n_vars: number of variable nodes (indices 0..n_vars-1)

        Returns:
            node_out: Updated node embeddings [N, d_model]
            graph_emb: Pooled graph embedding [d_model]
        """
        h = node_features
        n = h.shape[0]
        dev = h.device

        # Positional encoding (amortized, per-node-type)
        node_idx = torch.arange(n, device=dev)
        pos = self.pos_enc(node_idx[: n + (self.max_nodes - n)])  # safe indexing
        h = h + pos

        for _ in range(self.n_layers):
            ef = edge_features
            dev = h.device
            # --- Direction 1: constraint → variable ---
            c2v_mask = edge_index[1] < n_vars  # [E]
            if c2v_mask.any():
                c2v_src = edge_index[0][c2v_mask]
                c2v_dst = edge_index[1][c2v_mask]
                edge_emb = self._edge_emb(
                    ef[c2v_mask] if ef is not None else None, c2v_src.numel(), dev
                )
                msg = self.c2v(torch.cat([h[c2v_src], edge_emb, h[c2v_dst]], dim=-1))
                h[c2v_dst] = h[c2v_dst] + msg

            # --- Direction 2: variable → constraint ---
            v2c_mask = ~c2v_mask
            if v2c_mask.any():
                v2c_src = edge_index[0][v2c_mask]
                v2c_dst = edge_index[1][v2c_mask]
                edge_emb = self._edge_emb(
                    ef[v2c_mask] if ef is not None else None, v2c_src.numel(), dev
                )
                msg = self.v2c(torch.cat([h[v2c_src], edge_emb, h[v2c_dst]], dim=-1))
                h[v2c_dst] = h[v2c_dst] + msg

        # Graph pooling: mean over all nodes
        graph_emb = h.mean(dim=0)
        return h, graph_emb

    def _edge_emb(
        self, edge_feat: Optional[torch.Tensor], n: int, dev: torch.device
    ) -> torch.Tensor:
        """Return edge embedding tensor of shape [n, edge_dim].

        When edge_feat is None or empty, creates zeros of correct shape.
        """
        if edge_feat is None or edge_feat.numel() == 0:
            return torch.zeros(n, self.edge_dim, device=dev)
        return edge_feat[:n] if edge_feat.shape[0] > n else edge_feat


# ─── Transformer Decoder (replaces RNN decoder) ────────────────────────────


class BipartiteDecoder(nn.Module):
    """Pointer decoder using transformer attention.

    Context vector = [graph_emb; first_picked; last_picked] (AM-style)
    Attention between context and node embeddings → pointer distribution.

    Uses log-prob masking for feasible actions (equivalent to
    softmax(mask * probs), numerically stable).
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_ctx = nn.Linear(3 * d_model, d_model)  # context projection
        # STOP pointer
        self.stop = nn.Parameter(torch.randn(1, d_model))

    def attention_logits(
        self,
        context: torch.Tensor,  # [B, 3*d_model]: [graph; first; last]
        node_emb: torch.Tensor,  # [N, d_model]
    ):
        """Compute pointer attention logits over nodes.

        Args:
            context: Context vector [B, 3*d_model]
            node_emb: Node embeddings [N, d_model]

        Returns:
            logits: [B, N] (one logit per node)
        """
        B = context.shape[0]
        # Project context
        ctx_proj = self.W_ctx(context)  # [B, d_model]
        q = ctx_proj.unsqueeze(1)  # [B, 1, d_model]
        k = self.W_k(node_emb)  # [N, d_model]

        # Pointer attention
        attn = torch.sum(torch.tanh(q + k), dim=-1).squeeze(1)  # [B, N]

        # STOP attention
        stop_k = self.W_k(self.stop.expand(1, -1))  # [d_model]
        stop_attn = torch.sum(torch.tanh(q + stop_k.unsqueeze(0)), dim=-1).squeeze(
            1
        )  # [B]

        return attn, stop_attn

    def step(
        self,
        context: torch.Tensor,  # [B, 3*d_model]
        node_emb: torch.Tensor,  # [N, d_model]
        mask: torch.Tensor,  # [B, N] bool, True = feasible
    ):
        """Single decoding step with masking.

        Args:
            context: Context vector [B, 3*d_model]
            node_emb: Node embeddings [N, d_model]
            mask: Feasibility mask [B, N], True = valid action

        Returns:
            dist: Categorical over [N+1] actions (N nodes + STOP)
        """
        attn, stop_attn = self.attention_logits(context, node_emb)
        logits = torch.cat([attn, stop_attn.unsqueeze(1)], dim=-1)  # [B, N+1]

        # Log-prob masking: valid=1 → +0, valid=0 → +(-inf)
        masked = logits + torch.cat(
            [mask.float().log(), torch.zeros(logits.size(0), 1, device=logits.device)],
            dim=-1,
        )

        # Replace non-finite
        masked = torch.where(
            torch.isfinite(masked),
            masked,
            torch.full_like(masked, -1e9),
        )

        return torch.distributions.Categorical(logits=masked)


# ─── Scheduler: BipartiteGNN + Decoder ─────────────────────────────────────


class BipartiteGNNScheduler(nn.Module):
    """End-to-end scheduler using BipartiteGNN encoder + pointer decoder.

    Architecture:
        1. Project input node features to d_model
        2. BipartiteGNN encoder: message passing on bipartite graph
        3. Pool graph embedding + form context vector
        4. Autoregressive decoding: context → attention → pick node → update context

    This is the AM (Attention Model) from Kool et al. 2019, adapted for
    MILP bipartite graph structure instead of complete graph.

    The encoder runs once per episode. The decoder autoregressively picks
    variable nodes (decision variables x_j ∈ {0, 1}).
    """

    def __init__(
        self,
        f_node: int,
        f_global: int = 32,
        d_model: int = 128,
        n_gnn_layers: int = 2,
        edge_dim: int = 1,
        # LEHD re-encoding
        reencode_every: int = 0,
    ):
        super().__init__()
        # Project raw node features to d_model
        self.node_proj = nn.Linear(f_node, d_model)
        self.f_node = f_node
        self.d_model = d_model

        # LEHD re-encoding
        self.reencode_every = reencode_every

        # Bipartite GNN encoder
        self.encoder = BipartiteGNN(
            d_model=d_model, n_layers=n_gnn_layers, edge_dim=edge_dim
        )

        # Pointer decoder
        self.decoder = BipartiteDecoder(d_model=d_model)

        # Graph embedding → context projection
        # Context = [graph_emb; first_picked; last_picked]
        self.graph_proj = nn.Linear(d_model, d_model)
        self.n_vars = 0  # set after graph construction

        # Value head
        self.value = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, node_features, edge_index, edge_features, n_vars):
        """Encode graph → return node embeddings + graph embedding.

        Args:
            node_features: [N, f_node] raw node features
            edge_index: [2, E] edge list
            edge_features: [E, edge_dim] per-edge features
            n_vars: number of variable nodes

        Returns:
            node_emb: [N, d_model] updated node embeddings
            graph_emb: [d_model] pooled graph embedding
        """
        h = self.node_proj(node_features)  # [N, d_model]
        return self.encoder(h, edge_index, edge_features, n_vars)

    def value_of(self, graph_emb, node_emb, context):
        """Value estimate for current state.

        Combines graph embedding, mean-pooled node embeddings, and context.
        """
        pooled = node_emb.mean(dim=0)  # [d_model]
        return self.value(torch.cat([graph_emb, pooled, context], dim=-1)).squeeze(-1)

    def act(self, obs, action=None, greedy=False):
        """Autoregressive decision using bipartite GNN encoder + pointer decoder.

        Args:
            obs: observation dict with:
                - nodes: [B, N, f_node] concatenated [variables; constraints]
                - edge_index: [2, E] or batched [B, 2, E]
                - edge_features: [E, edge_dim] or [B, E, edge_dim]
                - n_vars: int
                - mask: [B, N] bool feasibility mask
                - task_static: [B, n_vars, f_static] for picking feature extraction
            action: [B, n_vars] replay actions (0/1)
            greedy: use argmax instead of sampling

        Returns:
            dict with new_starts[B, n_vars], logp[B], entropy[B], value[B]
        """
        nodes = obs["nodes"]  # [B, N, f_node]
        mask = obs.get(
            "mask",
            torch.ones(
                nodes.shape[0], nodes.shape[1], dtype=torch.bool, device=nodes.device
            ),
        )
        n_vars = obs.get("n_vars", self.n_vars) or self.n_vars
        B, N, f = nodes.shape
        dev = nodes.device

        edge_index = obs.get("edge_index")
        edge_features = obs.get("edge_features")

        if edge_index is None:
            # No edges: fall back to mean-pool value
            h = self.node_proj(nodes)  # [B, N, d]
            graph_emb = h.mean(dim=1)  # [B, d]
            value = self.value(
                torch.cat([graph_emb, graph_emb, graph_emb], dim=-1)
            ).squeeze(-1)
            return {
                "new_starts": torch.zeros(B, n_vars, device=dev),
                "logp": torch.zeros(B, device=dev),
                "entropy": torch.zeros(B, device=dev),
                "value": value,
            }

        # Expand edge_index to [B, 2, E]
        if edge_index.ndim == 2:
            edge_index = edge_index.unsqueeze(0).expand(B, -1, -1)
        elif edge_index.shape[0] == 1 and B > 1:
            edge_index = edge_index.expand(B, -1, -1)

        # Expand edge_features to [B, E, edge_dim]
        if edge_features is not None and edge_features.ndim == 2:
            edge_features = edge_features.unsqueeze(0).expand(B, -1, -1)
        elif edge_features is not None and edge_features.shape[0] == 1 and B > 1:
            edge_features = edge_features.expand(B, -1, -1)

        # Encode graph once
        h_all = self.node_proj(nodes)  # [B, N, d]
        node_emb_all = torch.zeros(B, N, self.d_model, device=dev)
        graph_emb_all = torch.zeros(B, self.d_model, device=dev)

        for b in range(B):
            n_h, n_g = self.encoder(
                h_all[b],
                edge_index[b],
                edge_features[b] if edge_features is not None else None,
                n_vars,
            )
            node_emb_all[b] = n_h
            graph_emb_all[b] = n_g

        # Extract variable embeddings for pointer attention
        var_emb = node_emb_all[:, :n_vars]  # [B, n_vars, d]
        graph_emb = self.graph_proj(graph_emb_all)  # [B, d]

        # Value estimate
        pooled = var_emb.mean(dim=1)  # [B, d]
        context_vec = torch.cat([graph_emb, pooled, graph_emb], dim=-1)  # [B, 3d]
        value = self.value(context_vec).squeeze(-1)

        # Initialize context: [graph_emb; -1; -1] (no picks yet)
        neg_one = -torch.ones(B, self.d_model, device=dev)
        context = torch.cat([graph_emb, neg_one, neg_one], dim=-1)  # [B, 3d]
        context_first = neg_one  # first picked node embedding
        context_last = neg_one  # last picked node embedding

        # Autoregressive decoding
        chosen = torch.zeros(B, n_vars, dtype=torch.bool, device=dev)
        logp = torch.zeros(B, device=dev)
        ent = torch.zeros(B, device=dev)
        alive = torch.ones(B, dtype=torch.bool, device=dev)
        replay = action is not None

        order: list[list[int]] = []
        ptr: list[int] = [0] * B
        if replay and action is not None:
            order = [torch.where(action[b] > 0.5)[0].tolist() for b in range(B)]

        max_picks = n_vars
        for step in range(max_picks):
            remaining = mask & (~chosen)
            has_cand = remaining.any(1)
            alive = alive & has_cand
            if not alive.any():
                break

            # Context-aware pointer attention over remaining feasible variables
            dist = self.decoder.step(context, var_emb, remaining.float())

            if replay:
                pick = torch.full((B,), n_vars, dtype=torch.long, device=dev)
                for b in range(B):
                    if alive[b] and ptr[b] < len(order[b]):
                        pick[b] = order[b][ptr[b]]
                        ptr[b] += 1
            elif greedy:
                pick = dist.logits.argmax(-1)
            else:
                pick = dist.sample()

            step_logp = dist.log_prob(pick)
            step_ent = dist.entropy()
            logp = logp + torch.where(alive, step_logp, torch.zeros_like(step_logp))
            ent = ent + torch.where(alive, step_ent, torch.zeros_like(step_ent))

            # STOP action
            is_stop = pick.eq(n_vars)
            took = alive & (~is_stop)
            idx = pick.clamp(max=n_vars - 1)

            if took.any():
                taken_b = torch.arange(B, device=dev)[took]
                chosen[taken_b, idx[taken_b]] = True

                # Update context with picked node embedding
                picked_emb = var_emb[taken_b, idx[taken_b]]  # [B_t, d]

                if context_first[taken_b].abs().sum() < 1e-3:
                    context_first = context_first.clone()
                    context_first[taken_b] = picked_emb

                context_last = context_last.clone()
                context_last[taken_b] = picked_emb

                context = torch.cat([graph_emb, context_first, context_last], dim=-1)

            # LEHD re-encoding: re-run encoder at intervals
            if (
                self.reencode_every > 0
                and step > 0
                and (step + 1) % self.reencode_every == 0
            ):
                updated_obs = {
                    k: v.clone() if torch.is_tensor(v) else v for k, v in obs.items()
                }
                if "mask" in updated_obs:
                    updated_obs["mask"] = updated_obs["mask"] & (~chosen)
                h_new = self.node_proj(updated_obs["nodes"])  # [B, N, d]
                node_emb_new = torch.zeros(B, N, self.d_model, device=dev)
                if edge_index is not None:
                    if edge_index.ndim == 2:
                        ei_ = edge_index.unsqueeze(0).expand(B, -1, -1)
                    elif edge_index.shape[0] == 1 and B > 1:
                        ei_ = edge_index.expand(B, -1, -1)
                    else:
                        ei_ = edge_index
                    if edge_features is not None:
                        if edge_features.ndim == 2:
                            ef_ = edge_features.unsqueeze(0).expand(B, -1, -1)
                        elif edge_features.shape[0] == 1 and B > 1:
                            ef_ = edge_features.expand(B, -1, -1)
                        else:
                            ef_ = edge_features
                    else:
                        ef_ = None
                    for b_ in range(B):
                        ef_b = ef_[b_] if ef_ is not None else None
                        n_h, _ = self.encoder(
                            h_new[b_],
                            ei_[b_],
                            ef_b,
                            n_vars,
                        )
                        node_emb_new[b_] = n_h
                var_emb = node_emb_new[:, :n_vars]

            alive = alive & (~is_stop)

        return {
            "new_starts": chosen.float(),
            "logp": logp,
            "entropy": ent,
            "value": value,
        }


# ─── Factory ────────────────────────────────────────────────────────────────


def build_bipartite_gnn(
    f_node: int = 16,
    f_global: int = 32,
    d_model: int = 128,
    n_gnn_layers: int = 2,
    edge_dim: int = 1,
) -> BipartiteGNNScheduler:
    """Convenience factory for BipartiteGNN scheduler.

    Args:
        f_node: input node feature dimension
        f_global: global feature dimension (for backward compat)
        d_model: hidden dimension
        n_gnn_layers: number of bipartite message passing layers
        edge_dim: edge feature dimension

    Returns:
        BipartiteGNNScheduler model
    """
    return BipartiteGNNScheduler(
        f_node=f_node,
        f_global=f_global,
        d_model=d_model,
        n_gnn_layers=n_gnn_layers,
        edge_dim=edge_dim,
    )
