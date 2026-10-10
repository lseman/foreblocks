"""Inference-only ordered-group search: no likelihood replay or stored trajectory."""
import torch

from .rl_bipartite import bipartite_observation


def sequence_inference(env, policy):
    """Sample the trained distribution once; retain validated search logs only.

    This preserves operator/cardinality/ordered-task sampling, and supplies no
    training likelihood or trajectory. Use sequence_rollout for RL collection.
    """
    while not env.done:
        started = env.clock()
        observation, mask = bipartite_observation(env)
        if mask[0]:
            env.step(0)
            env.log[-1]['selection_runtime'] = env.clock()-started
            break
        with torch.no_grad():
            distribution, _ = policy(observation, mask)
            action = distribution.sample().detach().cpu().tolist()
        selection_time = env.clock()-started
        env.step_sequence(action)
        env.log[-1]['selection_runtime'] = selection_time
    return env.incumbent
