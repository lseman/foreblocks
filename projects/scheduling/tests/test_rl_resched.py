from types import SimpleNamespace

import pytest
import torch

from projects.scheduling.rl_resched import ReschedActorCritic


def observation():
    generator = torch.Generator().manual_seed(15)
    return SimpleNamespace(tasks=torch.randn(4, 15, generator=generator),
                           times=torch.randn(7, 9, generator=generator),
                           edges=torch.randn(4, 7, 6, generator=generator),
                           eligible=torch.rand(4, 7, generator=generator) > .4,
                           groups=torch.tensor([[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool),
                           action_features=torch.randn(3, 12, generator=generator),
                           context=torch.randn(10, generator=generator))


def test_resched_equivariant_tasks_and_candidates():
    torch.manual_seed(3)
    model = ReschedActorCritic(layers=2).eval()
    obs = observation()
    mask = torch.tensor([True, True, False])
    original, value = model(obs, mask)
    permutation = torch.tensor([2, 0, 3, 1])
    moved = SimpleNamespace(**vars(obs))
    moved.tasks, moved.edges, moved.eligible = obs.tasks[permutation], obs.edges[permutation], obs.eligible[permutation]
    moved.groups = obs.groups[:, permutation]
    equivalent, equivalent_value = model(moved, mask)
    torch.testing.assert_close(original.probs, equivalent.probs, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(value, equivalent_value, atol=1e-6, rtol=1e-6)
    order = torch.tensor([1, 2, 0])
    moved.groups, moved.action_features = moved.groups[order], moved.action_features[order]
    reordered, _ = model(moved, mask[order])
    torch.testing.assert_close(reordered.probs, original.probs[order])
    assert original.probs[2] == 0


def test_resched_isolated_nodes_gradients_and_edges():
    torch.manual_seed(8)
    model = ReschedActorCritic()
    obs = observation()
    obs.eligible[:] = False
    dist, value = model(obs, torch.ones(3, dtype=torch.bool))
    assert torch.isfinite(dist.probs).all() and torch.isfinite(value)
    obs.eligible[:] = True
    obs.edges.requires_grad_()
    dist, value = model(obs, torch.ones(3, dtype=torch.bool))
    (-dist.log_prob(torch.tensor(0)) + value.square()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert obs.edges.grad.abs().sum() > 0
    assert model.blocks[0].task_attention.in_proj_weight.grad.abs().sum() > 0
    assert model.blocks[0].edge.weight.grad.abs().sum() > 0


def test_resched_checkpoint_observation_and_invalid_mask(tmp_path):
    model = ReschedActorCritic(hidden=15, heads=3)
    obs = observation()
    expected = model(obs, torch.tensor([1, 1, 0], dtype=torch.bool))
    path = tmp_path / 'model.pt'
    model.save(path, {'adaptation': 'ONTScandidate'})
    loaded = ReschedActorCritic.load(path)
    actual = loaded(obs, torch.tensor([1, 1, 0], dtype=torch.bool))
    torch.testing.assert_close(expected[0].probs, actual[0].probs)
    torch.testing.assert_close(expected[1], actual[1])
    assert not loaded.training
    assert loaded.observation(SimpleNamespace(observe=lambda: obs)) is obs
    with pytest.raises(ValueError, match='mask'):
        loaded(obs, torch.zeros(3, dtype=torch.bool))
    with pytest.raises(ValueError):
        ReschedActorCritic(hidden=15, heads=2)
    torch.save({'format': 'wrong'}, path)
    with pytest.raises(ValueError, match='checkpoint'):
        ReschedActorCritic.load(path)
