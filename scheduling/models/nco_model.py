"""NCOModel — Unified wrapper composing encoder + decoder + value head.

Composes any encoder (Transformer, BipartiteGNN, etc.) with any decoder
(Pointer, BipartiteAttention) via the shared autoregressive decode loop.

Key design:
    * Same act() / self_eval() / value_of() interface regardless of encoder type
    * Pluggable encoder via registry (register_encoder(name, cls))
    * Pluggable decoder via registry (register_decoder(name, cls))
    * LEHD-style re-encoding: re-encode at intervals during decoding
    * Constraint propagation: pluggable constraint_fn for feasibility masking

Usage:
    from models.nco_model import NCOModel, build_model

    # Using pre-registered encoders/decoders:
    model = build_model(
        encoder="transformer",
        decoder="pointer",
        d_model=64, n_heads=4, n_layers=2
    )

    # Using custom instances:
    from models.pointer_net import TransformerEncoder, PointerDecoder
    encoder = TransformerEncoder(f_static=9, f_dynamic=9, f_global=4)
    decoder = PointerDecoder(d_model=64, n_heads=4)
    model = NCOModel(encoder, decoder, d_value=64)

    # Training:
    out = model.act(obs)  # Returns {new_starts, logp, value, entropy}
    out = model.self_eval(obs, k=4, temperature=1.5)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


class EncoderRegistry:
    """Registry for pluggable encoder classes."""

    _registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, encoder_cls):
        """Register an encoder class by name.

        The encoder class must accept **kwargs including d_model, n_heads,
        n_layers, and optionally encoder-specific params.
        """
        cls._registry[name] = encoder_cls

    @classmethod
    def get(cls, name: str):
        """Get a registered encoder class."""
        if name not in cls._registry:
            raise ValueError(
                f"Unknown encoder '{name}'. Registered: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list(cls):
        return list(cls._registry.keys())


class DecoderRegistry:
    """Registry for pluggable decoder classes."""

    _registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, decoder_cls):
        """Register a decoder class by name."""
        cls._registry[name] = decoder_cls

    @classmethod
    def get(cls, name: str):
        """Get a registered decoder class."""
        if name not in cls._registry:
            raise ValueError(
                f"Unknown decoder '{name}'. Registered: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list(cls):
        return list(cls._registry.keys())


class NCOModel(nn.Module):
    """Unified NCO model wrapping encoder + decoder + value head.

    Supports:
        - Any encoder implementing: encode(obs) -> (task_emb, context) + n_vars
        - Any decoder implementing: attention_logits(ctx, nodes) -> (attn, stop)
        - LEHD re-encoding during decoding for large-instance generalization
        - Constraint propagation for feasibility masking
        - Self-evaluation with branch-and-score or multiple sampling

    Args:
        encoder: Encoder instance (from registry or direct instantiation)
        decoder: Decoder instance (from registry or direct instantiation)
        d_value: Dimension for value head (default: same as d_model)
        reencode_every: LEHD re-encoding frequency (0 = no re-encoding)
        constraint_fn: Optional constraint propagation function
    """

    def __init__(
        self,
        encoder: Any,
        decoder: Any,
        d_value: int = 0,
        reencode_every: int = 0,
        constraint_fn: Optional[Callable] = None,
    ):
        super().__init__()

        # Auto-adapt encoder if it doesn't have encode() method
        if not hasattr(encoder, "encode"):
            from models.adapters import adapt_encoder

            encoder_type = type(encoder).__name__
            n_vars = (
                getattr(encoder, "n", None) or getattr(encoder, "n_vars", None) or 12
            )
            self.encoder = adapt_encoder(encoder, n_vars=int(n_vars) if n_vars else 12)
        else:
            self.encoder = encoder

        # Auto-adapt decoder if it doesn't have create_context() method
        if not hasattr(decoder, "create_context"):
            from models.adapters import adapt_decoder

            n_vars = getattr(self.encoder, "n_vars", 12)
            self.decoder = adapt_decoder(decoder, n_vars=int(n_vars))
        else:
            self.decoder = decoder

        self.n_vars = getattr(self.encoder, "n_vars", 0)

        d_model = getattr(self.encoder, "d_model", 64)
        if d_value <= 0:
            d_value = d_model
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )
        self.reencode_every = reencode_every
        self.constraint_fn = constraint_fn

    def encode(self, obs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation. Delegates to encoder."""
        return self.encoder.encode(obs)

    def value_of(self, task_emb: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Value head: pooled state -> scalar value.

        Args:
            task_emb: [B, N, d_model] node embeddings
            context:  [B, d_model] graph context

        Returns:
            value: [B] scalar value per batch element
        """
        pooled = task_emb.mean(dim=1)  # [B, d_model]
        concat = torch.cat([pooled, context], dim=-1)  # [B, 2*d_model]
        return self.value_head(concat).squeeze(-1)

    def _make_mask(
        self,
        obs: Dict[str, Any],
        taken: torch.Tensor,
    ) -> torch.Tensor:
        """Build feasibility mask for current step.

        Combines base feasibility mask from obs with already-taken nodes,
        and optionally applies constraint propagation.

        Args:
            obs: Observation dict
            taken: [B, N] boolean tensor of picked nodes

        Returns:
            mask: [B, N] boolean mask (True = feasible)
        """
        mask = obs.get("mask", torch.ones_like(taken))
        mask = mask & (~taken)

        # Apply constraint propagation if provided
        if self.constraint_fn is not None:
            mask = self.constraint_fn(mask, None, obs)

        return mask

    def _apply_reencode(
        self,
        obs: Dict[str, Any],
        nodes: torch.Tensor,
        taken: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> torch.Tensor:
        """LEHD re-encoding: re-run encoder at intervals.

        In LEHD style, the encoder runs once initially, then is called
        again at step intervals during decoding to update node embeddings
        based on the evolving solution context.

        Args:
            obs: Original observation dict
            nodes: Current node embeddings (before re-encoding)
            taken: Boolean mask of already picked nodes [B, N]
            step: Current decoding step (0-indexed)

        Returns:
            Updated node embeddings after re-encoding
        """
        if self.reencode_every <= 0:
            return nodes

        # Re-encode at steps: reencode_every, 2*reencode_every, ...
        # step is 0-indexed, so re-encode when (step + 1) % reencode_every == 0 and step > 0
        if step > 0 and (step + 1) % self.reencode_every != 0:
            return nodes

        # Full re-encode with updated mask
        # Create a copy of obs with taken nodes masked out or marked as done
        updated_obs = {
            k: v.clone() if torch.is_tensor(v) else v for k, v in obs.items()
        }
        if "mask" in updated_obs and taken is not None:
            # Mask already taken nodes
            updated_obs["mask"] = updated_obs["mask"] & (~taken)

        # Re-encode
        task_emb_new, ctx_new = self.encoder.encode(updated_obs)
        return task_emb_new

    def act(
        self,
        obs: Dict[str, Any],
        action: Optional[torch.Tensor] = None,
        action_order: Any = None,
        temperature: float = 10.0,
        k: int = 1,
    ) -> Dict[str, Any]:
        """Single-step action selection (training or inference).

        Args:
            obs: Observation dict (format depends on encoder)
            action: Optional [B] actions for replay (training)
            action_order: Optional order tensor (training, usually None)
            temperature: Softmax temperature (default 1.0)
            k: Number of solutions to sample (default 1, >1 = self-eval)

        Returns:
            Dict with:
                new_starts: [B] selected action indices
                logp: [B] log-probabilities
                entropy: [B] entropy
                value: [B] value estimate
                stops: [B] STOP logits
        """
        B = obs["nodes"].shape[0] if "nodes" in obs else obs["task_static"].shape[0]
        dev = next(self.parameters()).device

        if k > 1:
            # Self-eval: sample k solutions, pick best
            return self._self_eval(obs, k=k, temperature=temperature)

        # Encode
        if action is not None:
            # Replay mode: encode once, use provided actions
            task_emb, ctx = self.encoder.encode(obs)
            return self._decode_replay(
                task_emb, ctx, obs, action, temperature, action_order
            )
        else:
            # Autoregressive mode
            task_emb, ctx = self.encoder.encode(obs)
            return self._decode_autoregressive(task_emb, ctx, obs, temperature)

    def _decode_replay(
        self,
        task_emb: torch.Tensor,
        ctx: torch.Tensor,
        obs: Dict[str, Any],
        action: torch.Tensor,
        temperature: float = 1.0,
        action_order: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Replay mode: compute log-probs for provided actions."""
        B = task_emb.shape[0]
        N = task_emb.shape[1]
        dev = task_emb.device

        # If action_order provided, use it directly (from autoregressive decoding)
        if action_order is not None and action_order.dim() > 1:
            max_steps = action_order.shape[1]
            action = action_order
            action_valid = action_order < N  # exclude STOP tokens (N) and padding
        # If action is a binary mask [B, N], convert to a padded index sequence
        elif (
            torch.is_floating_point(action)
            and action.dim() == 2
            and action.shape[1] == N
        ):
            selected = action > 0.5
            selected_counts = selected.sum(dim=1)
            try:
                max_steps = int(selected_counts.max().item())
            except (RuntimeError, ValueError):
                max_steps = 0
            if max_steps == 0:
                return {
                    "new_starts": torch.zeros(B, N, device=dev),
                    "logp": torch.zeros(B, device=dev),
                    "entropy": torch.zeros(B, device=dev),
                    "value": self.value_of(task_emb, ctx),
                    "stops": torch.zeros(B, device=dev),
                }

            idx = torch.arange(N, device=dev).expand(B, -1)
            sort_keys = torch.where(selected, idx, idx + N)
            action = torch.argsort(sort_keys, dim=1)[:, :max_steps]
            action_valid = torch.arange(max_steps, device=dev).unsqueeze(
                0
            ) < selected_counts.unsqueeze(1)
        else:
            max_steps = action.shape[1] if action.dim() > 1 else 1
            if action.dim() > 1:
                action_valid = action >= 0
            else:
                action_valid = torch.ones(B, 1, dtype=torch.bool, device=dev)

        taken = torch.zeros(B, N, dtype=torch.bool, device=dev)
        all_logp = torch.zeros(B, device=dev)
        all_entropy = torch.zeros(B, device=dev)
        stops = torch.zeros(B, device=dev)

        # Get initial context from encoder output (same as autoregressive)
        decoder_ctx = ctx.clone()

        for step in range(max_steps):
            # Replay action for this step
            if action.dim() > 1:
                a = action[:, step]
                step_valid = action_valid[:, step]
            else:
                a = action
                step_valid = action_valid[:, 0]

            # Ensure a is int64 for gather (full_logits has N task cols + 1 STOP col)
            a = a.to(torch.long)
            a = a.clamp(min=0, max=N)

            # Check if already stopped (all tasks taken or no valid actions)
            if not step_valid.any() or taken.all():
                break

            # Compute logits, normalize same as autoregressive path
            attn_logits, stop_logits = self.decoder.attention_logits(
                decoder_ctx, task_emb
            )
            all_logits = torch.cat([attn_logits, stop_logits.unsqueeze(-1)], dim=-1)
            m = all_logits.mean(dim=-1, keepdim=True)
            s = all_logits.std(dim=-1, keepdim=True) + 1e-8
            all_logits_n = (all_logits - m) / s

            # Mask infeasible and already-picked nodes (task logits only)
            mask = self._make_mask(obs, taken)
            masked_logits_n = all_logits_n[:, :N].masked_fill(~mask, -1e9)

            # Reconstruct full logits with masked tasks + unmasked stop (same as autoregressive)
            full_logits = torch.cat([masked_logits_n, all_logits_n[:, -1:]], dim=-1)

            # Log-prob of replayed action (over joint dist: tasks + stop, same as autoregressive)
            all_probs = torch.softmax(full_logits / temperature, dim=-1)
            action_logp = (
                torch.gather(all_probs, -1, a.unsqueeze(-1))
                .squeeze(-1)
                .log()
                .clamp(min=-10)
            )
            action_logp = action_logp.masked_fill(~step_valid, 0.0)

            # Entropy (for consistency with autoregressive path)
            step_ent = -torch.sum(all_probs * torch.log(all_probs + 1e-10), dim=-1)
            all_entropy += step_ent

            all_logp += action_logp
            stops = stop_logits

            # Update context and tracking
            valid = step_valid
            if valid.any():
                taken[valid] = taken[valid] | (
                    torch.arange(N, device=dev).unsqueeze(0) == a[valid].unsqueeze(1)
                )
                picked_emb = torch.gather(
                    task_emb,
                    1,
                    a[valid]
                    .unsqueeze(1)
                    .unsqueeze(-1)
                    .expand(-1, -1, task_emb.shape[-1]),
                ).squeeze(1)
                decoder_ctx = decoder_ctx.clone()
                decoder_ctx[valid] = self.decoder.update_context(
                    decoder_ctx[valid], picked_emb
                )

            # LEHD re-encoding
            task_emb = self._apply_reencode(obs, task_emb, taken, step)

        return {
            "new_starts": taken.float(),
            "logp": all_logp,
            "entropy": all_entropy,
            "value": self.value_of(task_emb, ctx),
            "stops": stops,
        }

    def _decode_autoregressive(
        self,
        task_emb: torch.Tensor,
        ctx: torch.Tensor,
        obs: Dict[str, Any],
        temperature: float,
    ) -> Dict[str, Any]:
        """Autoregressive mode: greedy or sampled action selection."""
        B = task_emb.shape[0]
        N = task_emb.shape[1]
        dev = task_emb.device

        taken = torch.zeros(B, N, dtype=torch.bool, device=dev)
        all_actions = torch.zeros(B, N, dtype=torch.long, device=dev)
        all_logp = torch.zeros(B, device=dev)
        all_entropy = torch.zeros(B, device=dev)
        stops = torch.zeros(B, device=dev)

        # Initial decoder context from encoder output
        decoder_ctx = ctx.clone()

        for step in range(N):
            # Compute logits
            attn_logits, stop_logits = self.decoder.attention_logits(
                decoder_ctx, task_emb
            )

            # Mask infeasible and already-picked
            mask = self._make_mask(obs, taken)

            # Combine task logits and stop logit, normalize before masking
            all_logits = torch.cat([attn_logits, stop_logits.unsqueeze(-1)], dim=-1)
            # Zero-mean, unit-var normalization (stable for large logits)
            m = all_logits.mean(dim=-1, keepdim=True)
            s = all_logits.std(dim=-1, keepdim=True) + 1e-8
            all_logits_n = (all_logits - m) / s

            # Apply mask to task logits only (STOP is always feasible)
            masked_logits_n = all_logits_n[:, :N].masked_fill(~mask, -1e9)
            # Reconstruct full logits with masked tasks + unmasked stop
            full_logits = torch.cat([masked_logits_n, all_logits_n[:, -1:]], dim=-1)
            all_probs = torch.softmax(full_logits / temperature, dim=-1)

            # Sample (temperature > 0) or greedy (temperature <= 0)
            if temperature <= 0:
                pick = full_logits.argmax(dim=-1)
                step_ent = torch.zeros(B, device=dev)
            else:
                dist = torch.distributions.Categorical(logits=full_logits / temperature)
                pick = dist.sample()
                step_ent = dist.entropy()
            action_logp = (
                torch.gather(all_probs, -1, pick.unsqueeze(-1))
                .squeeze(-1)
                .log()
                .clamp(min=-10)
            )
            do_stop = pick == N
            best_node = pick.clone()

            all_actions[:, step] = pick
            all_logp += action_logp
            all_entropy += step_ent
            stops = stop_logits

            # Check STOP
            if do_stop.all():
                break

            # Update tracking for non-stopping batch elements
            valid = ~do_stop
            if valid.any():
                taken[valid] = taken[valid] | (
                    torch.arange(N, device=dev).unsqueeze(0)
                    == best_node[valid].unsqueeze(1)
                )
                # Gather node embeddings for picked indices
                picked_emb = torch.gather(
                    task_emb,
                    1,
                    best_node[valid]
                    .unsqueeze(1)
                    .unsqueeze(-1)
                    .expand(-1, -1, task_emb.shape[-1]),
                ).squeeze(1)
                decoder_ctx = decoder_ctx.clone()
                decoder_ctx[valid] = self.decoder.update_context(
                    decoder_ctx[valid], picked_emb
                )

            # LEHD re-encoding
            task_emb = self._apply_reencode(obs, task_emb, taken, step)

        return {
            "new_starts": taken.float(),
            "logp": all_logp,
            "entropy": all_entropy,
            "value": self.value_of(task_emb, ctx),
            "stops": stops,
            "action_order": all_actions,  # sequence of picked task indices
        }

    def _self_eval(
        self,
        obs: Dict[str, Any],
        k: int = 4,
        temperature: float = 1.5,
    ) -> Dict[str, Any]:
        """Self-evaluation: sample k solutions, pick best.

        Uses sampling with temperature for exploration, then returns
        the action from the best solution.

        Args:
            obs: Observation dict
            k: Number of solutions to sample (default 4)
            temperature: Sampling temperature (default 1.5)

        Returns:
            Dict with new_starts from the best solution
        """
        B = obs["nodes"].shape[0] if "nodes" in obs else obs["task_static"].shape[0]
        N = obs["nodes"].shape[1] if "nodes" in obs else obs["task_static"].shape[1]
        dev = next(self.parameters()).device

        all_actions = []
        all_values = []

        for _ in range(k):
            # Encode
            task_emb, ctx = self.encoder.encode(obs)

            # Decode with sampling
            taken = torch.zeros(B, N, dtype=torch.bool, device=dev)
            actions = []
            decoder_ctx = self.decoder.create_context()

            for step in range(N):
                attn_logits, stop_logits = self.decoder.attention_logits(
                    decoder_ctx, task_emb
                )
                mask = self._make_mask(obs, taken)
                masked_logits = attn_logits.masked_fill(~mask, -1e9)

                stop_prob = torch.sigmoid(stop_logits)
                probs = torch.softmax(masked_logits / temperature, dim=-1)

                do_stop = torch.bernoulli(stop_prob) == 1
                if do_stop.any():
                    break

                node_probs = (1 - stop_prob.unsqueeze(-1)) * probs
                node_probs = node_probs / node_probs.sum(dim=-1, keepdim=True)
                picked = torch.multinomial(node_probs, 1).squeeze(-1)
                actions.append(picked)

                taken = taken | (
                    torch.arange(N, device=dev).unsqueeze(0) == picked.unsqueeze(1)
                )
                decoder_ctx = self.decoder.update_context(decoder_ctx, picked)

                # LEHD re-encoding
                task_emb = self._apply_reencode(obs, task_emb, taken, step)

            if actions:
                all_actions.append(torch.stack(actions, dim=1))
                all_values.append(self.value_of(task_emb, ctx))

        if all_actions:
            # Pick best solution (lowest value / highest return)
            vals = torch.stack(all_values, dim=1)  # [B, k]
            best_idx = vals.min(dim=-1).indices  # Pick lowest value
            best_actions = all_actions[0]  # All have same shape
            best_act = best_actions.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        else:
            best_act = torch.zeros(B, dtype=torch.long, device=dev)

        return {
            "new_starts": best_act,
            "logp": torch.zeros(B, device=dev),
            "entropy": torch.zeros(B, device=dev),
            "value": torch.zeros(B, device=dev),
            "stops": torch.zeros(B, device=dev),
        }


# ──────────────────────────────────────────────────────────
# Factory Functions
# ──────────────────────────────────────────────────────────


def build_model(
    encoder: str = "transformer",
    decoder: str = "pointer",
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_value: int = 0,
    reencode_every: int = 0,
    encoder_kwargs: Optional[Dict[str, Any]] = None,
    decoder_kwargs: Optional[Dict[str, Any]] = None,
) -> NCOModel:
    """Factory: build NCOModel from encoder/decoder names.

    Args:
        encoder: Encoder name (e.g., 'transformer', 'bipartite')
        decoder: Decoder name (e.g., 'pointer', 'bipartite')
        d_model: Hidden dimension (default 64)
        n_heads: Number of attention heads (default 4)
        n_layers: Number of layers (default 2)
        d_value: Value head dimension (0 = same as d_model)
        reencode_every: LEHD re-encoding frequency (0 = disabled)
        encoder_kwargs: Extra kwargs for encoder (e.g., f_static, edge_dim)
        decoder_kwargs: Extra kwargs for decoder (typically empty, only d_model)

    Returns:
        NCOModel instance

    Example:
        # Transformer + PointerDecoder
        model = build_model(
            encoder="transformer",
            decoder="pointer",
            d_model=64, n_heads=4, n_layers=2,
            encoder_kwargs={"f_static": 9, "f_dynamic": 9, "f_global": 4},
        )

        # BipartiteGNN + BipartiteDecoder
        model = build_model(
            encoder="bipartite",
            decoder="bipartite",
            d_model=128, n_layers=3,
            encoder_kwargs={"edge_dim": 1},
            reencode_every=5,
        )
    """
    encoder_cls = EncoderRegistry.get(encoder)
    decoder_cls = DecoderRegistry.get(decoder)

    ekw = dict(d_model=d_model, n_layers=n_layers)
    # Add n_heads only for Transformer-style encoders
    if encoder == "transformer":
        ekw["n_heads"] = n_heads
    # Add Transformer-specific params if present
    if encoder_kwargs:
        for k in ("f_static", "f_dynamic", "f_global"):
            if k in encoder_kwargs:
                ekw[k] = encoder_kwargs[k]
        # Add any remaining encoder-specific params
        for k, v in encoder_kwargs.items():
            if k not in ekw:
                ekw[k] = v
    encoder = encoder_cls(**ekw)

    dkw = dict(d_model=d_model)
    if decoder_kwargs:
        dkw.update(decoder_kwargs)
    decoder = decoder_cls(**dkw)

    return NCOModel(
        encoder=encoder,
        decoder=decoder,
        d_value=d_value,
        reencode_every=reencode_every,
    )


def register_builtin_models():
    """Register all built-in encoder/decoder classes.

    Call this once at startup to make encoders/decoders available
    by name to the factory and registry.
    """
    # Import here to avoid circular dependencies
    from models.pointer_net import TransformerEncoder, PointerDecoder
    from models.bipartite_gnn import BipartiteGNN, BipartiteDecoder

    # Register encoders
    EncoderRegistry.register("transformer", TransformerEncoder)
    EncoderRegistry.register("bipartite", BipartiteGNN)

    # Register decoders
    DecoderRegistry.register("pointer", PointerDecoder)
    DecoderRegistry.register("bipartite", BipartiteDecoder)
