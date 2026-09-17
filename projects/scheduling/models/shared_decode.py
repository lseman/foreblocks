"""Shared autoregressive decode loop for NCO models.

Pure function that handles the autoregressive generation loop:
    - Applies pointer attention + STOP
    - Masks already-picked and infeasible nodes
    - Supports greedy, sampling, and replay modes
    - Handles early STOP token detection

Works with any decoder implementing the DecoderProtocol interface:
    attention_logits(ctx, nodes) -> (attn_logits, stop_logits)
    create_context() -> initial_context
    update_context(ctx, picked_tok) -> updated_context

The loop returns (actions, log_probs, stops, values) that can be
consumed by any trainer (PPO, REINFORCE, POCO, SelfImprovement).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


def shared_decode_loop(
    attn_logits_fn: Callable[
        [torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
    ],
    create_context: Callable[[], Any],
    update_context: Callable[[Any, torch.Tensor], Any],
    nodes: torch.Tensor,
    mask: torch.Tensor,
    temp: float = 1.0,
    mode: str = "greedy",
    replay_actions: Optional[torch.Tensor] = None,
    stop_threshold: float = 0.5,
    max_steps: int = 512,
    ctx_init: Any = None,
) -> Dict[str, Any]:
    """Shared autoregressive decode loop.

    Handles the step-by-step generation of a solution:
        1. Initialize decoder context
        2. Loop: compute logits, mask, select, update context, check STOP
        3. Return actions, log-probs, STOP decisions, and value

    Args:
        attn_logits_fn: attention_logits(ctx, nodes) -> (attn_logits, stop_logits)
        create_context: create_context() -> initial_context
        update_context: update_context(ctx, picked_tok) -> updated_context
        nodes: [B, N, d_model] candidate node embeddings
        mask: [B, N] feasibility mask (True = can be picked)
        temp: softmax temperature (default 1.0)
        mode: "greedy", "sampling", or "replay"
        replay_actions: [B, max_steps] actions to replay (mode="replay" only)
        stop_threshold: STOP logit threshold (default 0.5)
        max_steps: maximum generation steps (default 512)
        ctx_init: optional pre-computed initial context (for reuse)

    Returns:
        Dict with:
            new_starts: [B] actions taken at each step
            logp: [B] cumulative log-probability
            entropy: [B] entropy at each step
            value: [B] value estimate (always 0.0 from loop, set by caller)
            stops: [B] STOP token logits
            n_picked: [B] number of nodes picked
    """
    B, N, _ = nodes.shape
    dev = nodes.device

    # Initialize tracking
    actions = torch.zeros(B, max_steps, dtype=torch.long, device=dev)
    log_probs = torch.zeros(B, device=dev)
    entropies = torch.zeros(B, device=dev)
    stopped = torch.zeros(B, dtype=torch.bool, device=dev)
    taken = torch.zeros(B, dtype=torch.bool, device=dev)
    n_picked = torch.zeros(B, dtype=torch.long, device=dev)
    last_stop_logits = torch.zeros(B, device=dev)

    # First picked node index (for context)
    first_picked_idx = torch.zeros(B, dtype=torch.long, device=dev)

    # Initialize context
    if ctx_init is not None:
        ctx = ctx_init
    else:
        ctx = create_context()

    # Track first and last picked for context updates
    first_picked = torch.zeros(B, dtype=torch.long, device=dev)

    for step in range(max_steps):
        # Check if all batches have stopped
        if stopped.all():
            break

        # Get logits for active batches
        active = ~stopped
        if not active.any():
            break

        attn_logits, stop_logits = attn_logits_fn(ctx, nodes)
        stop_logits = stop_logits.clamp(-20, 20)
        last_stop_logits = stop_logits  # Keep track for return

        # Clamp logits for numerical stability
        attn_logits = attn_logits.clamp(-20, 20)

        attn_logits, stop_logits = attn_logits_fn(ctx, nodes)

        # Clamp logits for numerical stability
        attn_logits = attn_logits.clamp(-20, 20)
        stop_logits = stop_logits.clamp(-20, 20)

        # Apply mask: set logits to -inf for infeasible nodes
        masked_logits = attn_logits.masked_fill(~mask, -1e9)

        # Compute pointer probabilities
        pointer_probs = torch.softmax(masked_logits / temp, dim=-1)

        # Compute STOP probability
        stop_prob = torch.sigmoid(stop_logits)

        # Compute combined distribution: pick between pointer and STOP
        # P(pick node j) = (1 - P(stop)) * P(pointer, j)
        # P(stop) = P(stop)
        n_active = active.sum().item()

        if mode == "greedy":
            # Greedy: pick argmax pointer or STOP
            if n_active > 0:
                pointer_idx = masked_logits[active].argmax(dim=-1)
                pointer_logit = masked_logits[active].max(dim=-1).values
                best_stop_logit = stop_logits[active]

                # Compare best pointer vs STOP
                better = pointer_logit > best_stop_logit

                new_starts = torch.where(
                    better, pointer_idx, torch.full_like(pointer_idx, -1)
                )
                logp = torch.where(
                    better,
                    torch.log(
                        pointer_probs[active]
                        .gather(-1, pointer_idx.unsqueeze(-1))
                        .squeeze(-1)
                        + 1e-10
                    ),
                    torch.log(stop_prob[active] + 1e-10),
                )
                entropy = torch.zeros_like(logp)

                stopped = stopped | better.logical_not() | (new_starts == -1)
                stopped = stopped | (new_starts == -1)
            else:
                new_starts = torch.full((B,), -1, dtype=torch.long, device=dev)
                logp = torch.zeros(B, device=dev)
                entropy = torch.zeros(B, device=dev)
                stopped = torch.ones(B, dtype=torch.bool, device=dev)

        elif mode == "sampling":
            # Sampling from pointer or STOP distribution
            if n_active > 0:
                # Bernoulli for STOP
                do_stop = torch.bernoulli(stop_prob[active]) == 1

                # Sample from pointer
                pointer_idx = torch.multinomial(pointer_probs[active], 1).squeeze(-1)

                new_starts = torch.where(
                    do_stop, torch.full_like(pointer_idx, -1), pointer_idx
                )

                if do_stop.any():
                    logp_stop = torch.log(stop_prob[active] + 1e-10)
                else:
                    logp_stop = torch.zeros_like(stop_prob[active]) - 1e9

                logp_ptr = torch.log(
                    pointer_probs[active]
                    .gather(-1, pointer_idx.unsqueeze(-1))
                    .squeeze(-1)
                    + 1e-10
                )
                logp = torch.where(do_stop, logp_stop, logp_ptr)

                entropy = -(
                    stop_prob[active] * torch.log(stop_prob[active] + 1e-10)
                    + (1 - stop_prob[active]) * pointer_probs[active].logsumexp(dim=-1)
                )
                entropy = torch.where(do_stop, entropy, entropy)

                stopped = stopped | do_stop
            else:
                new_starts = torch.full((B,), -1, dtype=torch.long, device=dev)
                logp = torch.zeros(B, device=dev)
                entropy = torch.zeros(B, device=dev)
                stopped = torch.ones(B, dtype=torch.bool, device=dev)

        else:
            # Replay mode: use provided actions
            assert replay_actions is not None, (
                "replay_actions required for mode='replay'"
            )
            new_starts = replay_actions[:, step]
            # Compute log-prob for the replayed action
            if n_active > 0:
                act_masked = new_starts[active]
                valid_mask = act_masked >= 0
                if valid_mask.any():
                    action_logp = (
                        pointer_probs[active]
                        .gather(-1, act_masked.clamp(min=0).unsqueeze(-1))
                        .squeeze(-1)
                    )
                    action_logp = action_logp.masked_fill(~valid_mask, 0.0)
                else:
                    action_logp = torch.zeros(len(active), device=dev)
                logp = action_logp
                entropy = torch.zeros_like(logp)
                stopped = torch.ones(B, dtype=torch.bool, device=dev)
            else:
                logp = torch.zeros(B, device=dev)
                entropy = torch.zeros(B, device=dev)
                stopped = torch.ones(B, dtype=torch.bool, device=dev)

        # Apply actions (only for active batches)
        if n_active > 0:
            actions[active, step] = new_starts[active]
            log_probs[active] += logp
            entropies[active] += entropy

        # Update tracking
        picked = (
            (new_starts >= 0)
            if n_active > 0
            else torch.zeros(B, dtype=torch.bool, device=dev)
        )

        if n_active > 0:
            taken[active] = taken[active] | picked
            # Record first picked
            newly_picked = picked & ~first_picked.ne(0)
            if newly_picked.any():
                new_indices = torch.arange(B, device=dev)
                for bi in range(B):
                    if newly_picked[bi]:
                        first_picked[bi] = new_starts[bi]
            n_picked += picked.long()

        # Update context with picked actions
        if mode == "replay":
            ctx = update_context(ctx, new_starts)
        else:
            # Only update context with valid picks
            if n_active > 0 and picked.any():
                ctx = update_context(ctx, new_starts)

        # Check early STOP (if STOP logit is high)
        if mode in ("greedy", "sampling"):
            stopped = stopped | (stop_prob > stop_threshold)

    return {
        "new_starts": actions[:, 0].clone(),  # First action for step 1
        "logp": log_probs,
        "entropy": entropies,
        "value": torch.zeros(B, device=dev),  # Placeholder, set by value head
        "stops": last_stop_logits,
        "n_picked": n_picked,
    }
