"""Adapters: Bridge existing encoder/decoder classes to NCOModel interface.

Wraps existing classes (TransformerEncoder, PointerDecoder, etc.) to
provide the unified interface expected by NCOModel.

These adapters enable backward compatibility — existing code works
without changes, while new code can use the registry-based factory.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class TransformerEncoderAdapter(nn.Module):
    """Adapter: TransformerEncoder -> NCOModel encoder interface.

    Wraps pointer_net.TransformerEncoder to provide:
        encode(obs) -> (task_emb, context)
        n_vars -> property
    """

    def __init__(self, encoder, n_vars: int = 0):
        super().__init__()
        self._inner = encoder
        self.d_model = encoder.d_model
        self.n_vars = n_vars if n_vars > 0 else 12

    def encode(self, obs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode via inner TransformerEncoder.

        Extracts task_static, task_dynamic, glob from obs dict
        and calls inner encoder.
        """
        task_static = obs["task_static"]
        task_dynamic = obs.get("task_dynamic", torch.zeros_like(task_static))
        glob = obs.get("glob", torch.zeros(task_static.shape[0], 4))

        # TransformerEncoder.__call__ delegates to forward(task_static, task_dynamic, glob)
        # Output: (task_emb, context)
        return self._inner(task_static, task_dynamic, glob)

    def forward(self, task_static, task_dynamic, glob):
        """Delegate to inner encoder."""
        return self._inner(task_static, task_dynamic, glob)


class PointerDecoderAdapter(nn.Module):
    """Adapter: PointerDecoder -> NCOModel decoder interface.

    Wraps pointer_net.PointerDecoder to provide:
        attention_logits(ctx, nodes) -> (attn_logits, stop_logits)
        create_context() -> context
        update_context(ctx, picked_tok) -> updated_ctx
    """

    def __init__(self, decoder, n_vars: int = 0):
        super().__init__()
        self._inner = decoder
        self.d_model = decoder.d_model
        self.n_vars = n_vars

    def attention_logits(
        self, ctx: torch.Tensor, nodes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute pointer attention + STOP logits directly.

        Computes the same logic as PointerDecoder.logits but with
        correct dimension handling.

        Args:
            ctx: [B, d_model] decoder context
            nodes: [B, N, d_model] node embeddings

        Returns:
            attn_logits: [B, N] node attention logits
            stop_logits: [B] STOP logit
        """
        B, N, d = nodes.shape
        # Pointer attention: tanh(q + k + v) pattern
        q = self._inner.W_q(ctx)  # [B, d]
        k = self._inner.W_k(nodes)  # [B, N, d]
        v = self._inner.W_v(nodes)  # [B, N, d]

        attn_raw = torch.tanh(q.unsqueeze(1) + k + v)  # [B, N, d]
        attn = torch.sum(attn_raw * q.unsqueeze(1), dim=-1)  # [B, N]

        # STOP: compare context with stop embedding
        stop_attn = torch.tanh(
            q.unsqueeze(1) + self._inner.W_k(self._inner.stop).unsqueeze(0)
        ).sum(dim=-1)
        # stop_attn is [B, 1], squeeze to [B]
        stop_logits = stop_attn.squeeze(-1)

        return attn, stop_logits

    def create_context(self) -> torch.Tensor:
        """Create initial decoder context (zero tensor)."""
        device = self._inner.W_q.weight.device
        return torch.zeros(1, self.d_model, device=device)

    def update_context(
        self, ctx: torch.Tensor, picked_tok: torch.Tensor
    ) -> torch.Tensor:
        """Update context with picked node embedding via GRUCell.

        Args:
            ctx: [B, d_model] current context
            picked_tok: [B] picked token indices or [B, d_model] embeddings

        Returns:
            Updated context [B, d_model]
        """
        return self._inner.update(ctx, picked_tok)


class BipartiteDecoderAdapter(nn.Module):
    """Adapter: BipartiteDecoder -> NCOModel decoder interface.

    Wraps bipartite_gnn.BipartiteDecoder to provide the unified interface.
    """

    def __init__(self, decoder):
        super().__init__()
        self._inner = decoder
        self.d_model = decoder.d_model

    def attention_logits(
        self, ctx: torch.Tensor, nodes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Delegate to inner BipartiteDecoder.attention_logits()."""
        return self._inner.attention_logits(ctx, nodes)

    def create_context(self) -> torch.Tensor:
        """Create initial context: [graph_emb; first_picked; last_picked].

        Initially first_picked and last_picked are zeros.
        """
        return torch.zeros(1, 3 * self.d_model, device=self._inner.W_q.weight.device)

    def update_context(
        self, ctx: torch.Tensor, picked_tok: torch.Tensor
    ) -> torch.Tensor:
        """Update context with picked node.

        Keeps first_picked (from step 0) and updates last_picked.
        """
        return ctx  # Placeholder


class BipartiteGNNAdapter(nn.Module):
    """Adapter: BipartiteGNN -> NCOModel encoder interface.

    Wraps bipartite_gnn.BipartiteGNN to provide:
        encode(obs) -> (task_emb, context)
        n_vars -> property
    """

    def __init__(self, encoder, n_vars: int = 0):
        super().__init__()
        self._inner = encoder
        self.d_model = encoder.d_model
        self.n_vars = n_vars if n_vars > 0 else 12
        # Project node features to d_model if needed
        self._node_proj = None

    def encode(self, obs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode via inner BipartiteGNN.

        Expects obs with:
            nodes: [B, N, f_node] node features
            edge_index: [2, E] edge list
            edge_features: [E, edge_dim] (optional)
        """
        nodes = obs["nodes"]  # [B, N, f_node]
        edge_index = obs["edge_index"]  # [2, E]
        edge_features = obs.get("edge_features", None)
        self.n_vars = int(obs.get("n_vars", self.n_vars))
        B = nodes.shape[0]

        # Project node features to d_model if f_node != d_model
        f_node = nodes.shape[-1]
        if f_node != self.d_model:
            if self._node_proj is None:
                self._node_proj = nn.Linear(f_node, self.d_model).to(nodes.device)
            nodes = self._node_proj(nodes)

        # BipartiteGNN.forward expects [N, d_model] (not batched)
        # Process batch one at a time
        task_embs = []
        contexts = []
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
            h, graph_emb = self._inner(
                node_features=nodes[b],
                edge_index=edge_index[b],
                edge_features=edge_features[b] if edge_features is not None else None,
                n_vars=self.n_vars,
            )
            task_embs.append(h.unsqueeze(0))  # [1, N, d]
            contexts.append(graph_emb.unsqueeze(0))  # [1, d]

        task_emb = torch.cat(task_embs, dim=0)  # [B, N, d]
        context = torch.cat(contexts, dim=0)  # [B, d]
        return task_emb, context

    @property
    def n_vars(self):
        return self._n_vars

    @n_vars.setter
    def n_vars(self, val):
        self._n_vars = val


def adapt_encoder(encoder, **kwargs):
    """Adapt any encoder to the NCOModel interface.

    Args:
        encoder: Encoder instance (TransformerEncoder, BipartiteGNN, etc.)
        **kwargs: Additional parameters (n_vars, etc.)

    Returns:
        Adapted encoder wrapper
    """
    encoder_type = type(encoder).__name__

    if "TransformerEncoder" in encoder_type:
        return TransformerEncoderAdapter(encoder, **kwargs)
    elif "BipartiteGNN" in encoder_type:
        return BipartiteGNNAdapter(encoder, **kwargs)
    else:
        # Generic pass-through (assumes encoder already has encode + n_vars)
        return encoder


def adapt_decoder(decoder, **kwargs):
    """Adapt any decoder to the NCOModel interface.

    Args:
        decoder: Decoder instance (PointerDecoder, BipartiteDecoder, etc.)
        **kwargs: Additional parameters

    Returns:
        Adapted decoder wrapper
    """
    decoder_type = type(decoder).__name__

    if "PointerDecoder" in decoder_type:
        return PointerDecoderAdapter(decoder, **kwargs)
    elif "BipartiteDecoder" in decoder_type:
        return BipartiteDecoderAdapter(decoder)
    else:
        # Generic pass-through
        return decoder
