"""Protocol ABCs for modular NCO framework.

Defines the interfaces that all encoder/decoder components must implement.
Existing classes can inherit from these for typing; the protocols are
optional — the NCOModel wrapper handles type checking internally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class EncoderProtocol(ABC):
    """Protocol for NCO encoders.

    All encoders must implement:
        encode(obs) -> (task_emb, context)
            task_emb: [B, N, d_model] node embeddings
            context:  [B, d_model] pooled graph/context embedding

    And expose:
        n_vars: int (number of decision variables)
    """

    @abstractmethod
    def encode(self, obs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation into node embeddings and context.

        Args:
            obs: Observation dict. Structure depends on encoder type:
                - Transformer: {"task_static": [...], "task_dynamic": [...], "glob": [...]}
                - Bipartite:   {"nodes": [...], "edge_index": [...], "edge_features": [...]}

        Returns:
            task_emb: [B, N, d_model] node embeddings
            context:  [B, d_model] pooled graph/context embedding
        """
        ...

    @property
    @abstractmethod
    def n_vars(self) -> int:
        """Number of decision variables. Set at runtime."""
        ...


class DecoderProtocol(ABC):
    """Protocol for NCO decoders.

    All decoders must implement:
        attention_logits(ctx, nodes) -> (attn_logits, stop_logits)
        create_context() -> initial_context
        update_context(ctx, picked_tok) -> updated_context
    """

    @abstractmethod
    def attention_logits(
        self, ctx: torch.Tensor, nodes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute pointer attention and STOP logits.

        Args:
            ctx:    [B, d_model] decoder context
            nodes:  [B, N, d_model] candidate node embeddings

        Returns:
            attn_logits: [B, N] attention logits for pointer selection
            stop_logits: [B] logit for STOP token
        """
        ...

    @abstractmethod
    def create_context(self) -> Any:
        """Create initial decoder context.

        Returns:
            Context object (shape depends on decoder type).
        """
        ...

    @abstractmethod
    def update_context(self, ctx: Any, picked_tok: torch.Tensor) -> Any:
        """Update decoder context after picking a node.

        Args:
            ctx: Current context (from create_context or previous update)
            picked_tok: [B] tensor of picked node indices/emb

        Returns:
            Updated context
        """
        ...


class TrainerProtocol(ABC):
    """Protocol for NCO trainers.

    All trainers must implement:
        update(env, iters, log_every) -> history
    """

    @abstractmethod
    def update(self, env, iters: int = 200, log_every: int = 20):
        """Run training for N update cycles.

        Args:
            env: NCO-compatible environment
            iters: Number of update iterations
            log_every: Log every N iterations

        Returns:
            history: list of mean return values per iteration
        """
        ...
