from itertools import product
from types import SimpleNamespace

import pytest
import torch

from projects.scheduling.rl_mistar import MistarActorCritic, MultiProposal


def observation():
    g = torch.Generator().manual_seed(17)
    current = torch.rand(3, 5, generator=g) > .5
    return SimpleNamespace(tasks=torch.randn(3, 15, generator=g),
        times=torch.randn(5, 9, generator=g), edges=torch.randn(3, 5, 6, generator=g),
        eligible=torch.ones(3, 5, dtype=torch.bool),
        groups=torch.tensor([[1, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=torch.bool),
        action_features=torch.randn(3, 12, generator=g), context=torch.randn(10, generator=g),
        current_schedule=current, history_schedules=torch.stack([current, ~current, current]),
        history_groups=torch.tensor([[1, 0, 0], [0, 1, 1], [1, 0, 0]], dtype=torch.bool),
        history_gains=torch.tensor([0., .1, .2]), history_objectives=torch.tensor([1., 2., 3.]),
        history_durations=torch.tensor([.1, .2, .3]))


def test_exact_joint_probability_entropy_and_duplicates():
    d = MultiProposal(torch.tensor([.3, -.2, -torch.inf]), 3)
    total = sum(d.log_prob(torch.tensor(a)).exp() for a in product(range(2), repeat=3))
    torch.testing.assert_close(total, torch.tensor(1.))
    torch.testing.assert_close(d.entropy(), 3*d.base.entropy())
    torch.testing.assert_close(d.log_prob(torch.tensor([0, 0, 0])), 3*d.base.log_prob(torch.tensor(0)))
    assert (d.sample() != 2).all()
    with pytest.raises(ValueError):
        d.log_prob(torch.tensor([0]))


def test_memory_retrieval_and_distinct_repetition():
    torch.manual_seed(3)
    model = MistarActorCritic(memory_topk=1)
    obs = observation()
    d, _ = model(obs, torch.ones(3, dtype=torch.bool))
    assert len(model.last_retrieved_indices) == 1
    idx = model.last_retrieved_indices.item()
    assert idx in (0, 2)
    model.memory = False
    without, _ = model(obs, torch.ones(3, dtype=torch.bool))
    assert not torch.allclose(d.logits, without.logits)
    model.memory = True
    task = model.task_input(obs.tasks)
    time = model.time_input(obs.times)
    _, penalty, _ = model._retrieve(obs, task, time)
    assert penalty[0] == 1 and penalty[1] == 0
    obs.history_schedules = ~obs.history_schedules
    _, penalty, _ = model._retrieve(obs, task, time)
    assert penalty[0] == 0  # matching group alone is not exact state repetition


def test_task_permutation_checkpoint_and_gradients(tmp_path):
    torch.manual_seed(4)
    model = MistarActorCritic(memory_topk=2)
    obs = observation()
    mask = torch.tensor([True, False, True])
    dist, value = model(obs, mask)
    perm = torch.tensor([2, 0, 1])
    other = SimpleNamespace(**vars(obs))
    for key in ('tasks', 'edges', 'eligible', 'current_schedule'):
        setattr(other, key, getattr(obs, key)[perm])
    other.groups = obs.groups[:, perm]
    other.history_groups = obs.history_groups[:, perm]
    other.history_schedules = obs.history_schedules[:, perm]
    pd, pv = model(other, mask)
    torch.testing.assert_close(dist.logits, pd.logits)
    torch.testing.assert_close(value, pv)
    assert torch.isneginf(dist.logits[1])
    loss = -dist.log_prob(torch.tensor([0, 2, 0, 2])) + value.square()
    loss.backward()
    assert model.memory_input[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    path = tmp_path/'policy.pt'
    model.save(path, metadata={'adaptation': True})
    loaded = MistarActorCritic.load(path)
    ld, lv = loaded(obs, mask)
    torch.testing.assert_close(ld.logits, dist.logits)
    torch.testing.assert_close(lv, value)
    assert loaded.training_metadata == {'adaptation': True}
    with pytest.raises(ValueError):
        loaded(obs, torch.zeros(3, dtype=torch.bool))


def test_empty_memory_and_no_eligible_rows():
    model = MistarActorCritic()
    obs = observation()
    obs.history_schedules = obs.history_schedules[:0]
    obs.eligible[0] = False
    dist, value = model(obs, torch.ones(3, dtype=torch.bool))
    assert torch.isfinite(dist.logits).all() and torch.isfinite(value)
