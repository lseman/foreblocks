"""Constraint-aware actor-critic policy for ONTS RL.

This module provides a policy that works with constraint-aware graph observations,
extending the standard ActorCritic to handle relaxability features while maintaining
compatibility with the existing candidate-based action space.

Architecture:
1. Node encoder: Linear → GELU → Linear → GELU (maps nodes to hidden dim)
2. Message passing: Optional (can be added later for better node representations)
3. Candidate pooling: Pool node representations using candidate masks
4. Actor: MLP from pooled candidates + global state to action logits (STOP + 4 candidates)
5. Critic: MLP from global state to value
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class ConstraintAwareActorCritic(nn.Module):
    """Actor-critic for constraint-aware graph observations with candidate-based actions.

    The policy sees:
    - Node features: [context; variable_0; ...; variable_{n-1}; constraint_0; ...; constraint_{m-1}]
    - Candidate masks: Boolean tensors indicating which nodes belong to each candidate
    - Global context: Summary statistics of the problem

    Action space: STOP (action 0) + 4 candidates (actions 1-4)
    """

    def __init__(self, node_feature_dim=22, hidden=32, n_candidates=4):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden = hidden
        self.n_candidates = n_candidates
        
        # Node encoder: maps node features to hidden dimension
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        
        # Global context encoder (for problem-level summary)
        self.global_input = nn.Sequential(
            nn.Linear(node_feature_dim, hidden),
            nn.GELU(),
        )
        
        # Actor: predicts action logits from pooled candidates + global state
        # Input: pooled_candidate_rep (hidden) + global_state (hidden) + candidate_features (variable)
        self.actor = nn.Sequential(
            nn.Linear(2 * hidden + 10, hidden),  # pooled + global + candidate features
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        
        # STOP action logit from global state
        self.stop = nn.Linear(hidden, 1)
        
        # Critic: predicts state value from global state
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        
        # Initialize weights
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.zeros_(self.stop.weight)
        nn.init.zeros_(self.stop.bias)
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

    def forward(self, nodes, candidate_masks, global_context, mask):
        """Forward pass through the policy.

        Args:
            nodes: [batch, n_nodes, node_feature_dim] — padded node features
            candidate_masks: [batch, n_candidates, n_nodes] — boolean mask for each candidate
            global_context: [batch, node_feature_dim] — global problem summary
            mask: [batch, n_candidates + 1] — action mask (STOP + candidates)

        Returns:
            distribution: Categorical over actions (STOP + candidates)
            value: [batch] — state value estimate
        """
        device = nodes.device
        
        # Encode nodes
        encoded = self.node_encoder(nodes)  # [batch, n_nodes, hidden]
        
        # Pool over candidate masks to get candidate representations
        # candidate_masks: [batch, n_candidates, n_nodes]
        # encoded: [batch, n_nodes, hidden]
        pooled_candidates = []
        for i in range(self.n_candidates):
            mask_i = candidate_masks[:, i, :]  # [batch, n_nodes]
            masked_encoded = encoded * mask_i.unsqueeze(-1)  # [batch, n_nodes, hidden]
            pooled = masked_encoded.sum(dim=1) / mask_i.sum(dim=1, keepdim=True).clamp(min=1)  # [batch, hidden]
            pooled_candidates.append(pooled)
        
        pooled_candidates = torch.stack(pooled_candidates, dim=1)  # [batch, n_candidates, hidden]
        
        # Global state
        global_state = self.global_input(global_context)  # [batch, hidden]
        
        # Candidate features (simplified: use pooled representation)
        # In practice, this would include candidate-specific features like operator type, etc.
        candidate_features = torch.zeros(nodes.shape[0], self.n_candidates, 10, device=device, dtype=encoded.dtype)
        
        # Actor: create logits for each candidate
        # Expand global_state and candidate_features for all candidates
        global_expanded = global_state.unsqueeze(1).repeat(1, self.n_candidates, 1)  # [batch, n_candidates, hidden]
        actor_inputs = torch.cat([
            pooled_candidates,  # [batch, n_candidates, hidden]
            global_expanded,    # [batch, n_candidates, hidden]
            candidate_features, # [batch, n_candidates, 10]
        ], dim=2)  # [batch, n_candidates, 2*hidden+10]
        
        action_logits = self.actor(actor_inputs).squeeze(-1)  # [batch, n_candidates]
        
        # Add STOP logit
        stop_logit = self.stop(global_state).squeeze(-1)  # [batch]
        all_logits = torch.cat([stop_logit.unsqueeze(1), action_logits], dim=1)  # [batch, n_candidates + 1]
        
        # Critic: state value
        value = self.critic(global_state).squeeze(-1)  # [batch]
        
        # Create distribution over actions
        distribution = Categorical(logits=all_logits.masked_fill(~mask, -torch.inf))
        
        return distribution, value
