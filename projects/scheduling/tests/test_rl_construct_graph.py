from dataclasses import replace

import torch

from projects.scheduling.rl_construct import ConstructionEnv, construct
from projects.scheduling.rl_construct_graph import GraphConstructPolicy, RelationAttention, graph_observation
from test_rl_construct import tiny


def test_graph_constructor_learns_and_supports_variable_dimensions():
    torch.manual_seed(7)
    model = GraphConstructPolicy()
    env = ConstructionEnv(tiny())
    choices = env.candidates()
    logits = model(graph_observation(env, choices))
    assert logits.shape == (len(choices),) and torch.isfinite(logits).all()
    row, logs, entropies = construct(tiny(), model, seed=2, training=True)
    assert row["success"]
    (-torch.stack(logs).sum() - .001 * torch.stack(entropies).sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_task_permutation_equivariance_without_index_embeddings():
    torch.manual_seed(7)
    original = tiny()
    order = torch.tensor([1, 0])
    permuted = replace(original, **{name: value[order] for name, value in vars(original).items() if name != "recurso_p"})
    model = GraphConstructPolicy().eval()
    a, b = ConstructionEnv(original), ConstructionEnv(permuted)
    ca, cb = a.candidates(), b.candidates()
    la, lb = model(graph_observation(a, ca)), model(graph_observation(b, cb))
    mapped = {(int(order[c.task]), c.row.tobytes()): float(logit.detach()) for c, logit in zip(cb, lb)}
    for c, logit in zip(ca, la):
        assert abs(float(logit.detach())-mapped[(c.task, c.row.tobytes())]) < 1e-5


def test_empty_edge_rows_are_finite():
    layer = RelationAttention(8)
    output = layer(torch.zeros(2, 8), torch.zeros(3, 8), torch.zeros(2, 3, 3), torch.zeros(2, 3, dtype=torch.bool))
    assert torch.isfinite(output).all()


def test_teacher_snapshot_is_preplacement_and_serializable():
    row, _, _ = construct(tiny(), None, mode="uniform", record_decisions=True, record_graph=True)
    first = row["decisions"][0]["graph"]
    assert all(first["remaining"]) and all(task[-1] == 0 for task in first["tasks"])
    assert torch.isfinite(GraphConstructPolicy()(first)).all()
