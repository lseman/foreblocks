import itertools

import torch

from projects.scheduling.research.datasets import witness_instance
from projects.scheduling.rl_subset import NonemptySubset, SubsetActorCritic, SubsetEnv
from projects.scheduling.rl_subset_memory import best_untried_subset, memory_map_rollout


def test_best_untried_matches_exhaustive_joint_density():
    for logits in (torch.tensor([-.8,.1,1.]), torch.zeros(3), torch.tensor([-1.,-2.,-3.])):
        dist = NonemptySubset(logits)
        available = {bits for bits in itertools.product((0,1),repeat=3) if any(bits)}
        failed = set()
        while available:
            chosen = best_untried_subset(logits, failed)
            key = tuple(chosen.tolist())
            assert key in available
            best = max(float(dist.log_prob(torch.tensor(bits))) for bits in available)
            assert abs(float(dist.log_prob(chosen))-best) < 1e-6
            failed.add(key)
            available.remove(key)
        assert best_untried_subset(logits,failed) is None


def test_memory_map_never_retries_failed_group_without_schedule_change():
    torch.set_num_threads(1)
    instance,witness=witness_instance(seed=82103,regime='balanced')
    policy=SubsetActorCritic()
    env=SubsetEnv(instance,witness,budget=.3)
    memory_map_rollout(env,policy)
    failed=set()
    for step in env.log:
        if not step['executed']:
            continue
        key=tuple(sorted(step['jobs']))
        assert key not in failed
        if step['reward']>1e-9:
            failed.clear()
        else:
            failed.add(key)
    assert env.log and env.problem.config.validate(instance,env.incumbent).feasible
    assert env.objective>=env.initial_objective
