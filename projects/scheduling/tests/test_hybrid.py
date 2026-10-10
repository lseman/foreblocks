"""Behavioral verification against independent enumeration and complete validation."""

from itertools import product

import numpy as np
import pytest
import torch

from projects.scheduling.environments.onts_env import ONTSInstance
from projects.scheduling.hybrid import (
    rollout_proposal,
    select_neighborhood,
    solve_hybrid,
)
from projects.scheduling.learning import train_neighborhood_policy
from projects.scheduling.milp import ONTSConfig, ONTSMilp
from projects.scheduling.neighborhood import GraphBuilder, NeighborhoodPolicy
from projects.scheduling.validation import validate_schedule


def make_instance(**changes):
    data = dict(
        priority=[2],
        uso_p=[1],
        min_statup=[1],
        max_statup=[2],
        min_cpu_time=[2],
        max_cpu_time=[3],
        min_periodo_job=[3],
        max_periodo_job=[5],
        win_min=[0],
        win_max=[5],
        recurso_p=[1] * 5,
    )
    data.update(changes)
    return ONTSInstance(**{k: torch.tensor(v) for k, v in data.items()})


@pytest.mark.parametrize(
    "changes,config",
    [
        ({}, ONTSConfig()),
        ({"min_periodo_job": [6], "max_periodo_job": [8]}, ONTSConfig()),
        ({}, ONTSConfig(allow_truncated_final_run=False)),
        ({"recurso_p": [100] * 5}, ONTSConfig()),
        ({"max_periodo_job": [3]}, ONTSConfig()),
        ({"win_min": [2], "win_max": [4]}, ONTSConfig()),
        ({"uso_p": [18], "recurso_p": [0] * 5}, ONTSConfig(soc_inicial=0.04)),
        ({"priority": [-1]}, ONTSConfig()),
    ],
)
def test_milp_matches_exhaustive_feasible_set(changes, config):
    instance = make_instance(**changes)
    problem = ONTSMilp(instance, config)
    expected = []
    # Fix each candidate x; solver must agree with the independent validator.
    for bits in product([0, 1], repeat=instance.J * instance.T):
        x = torch.tensor(bits).reshape(instance.J, instance.T)
        report = config.validate(instance, x)
        lo, hi = problem.lower.copy(), problem.upper.copy()
        lo[: problem.n_x] = np.maximum(lo[: problem.n_x], x.ravel().numpy())
        hi[: problem.n_x] = np.minimum(hi[: problem.n_x], x.ravel().numpy())
        from scipy.optimize import Bounds, milp

        solved = milp(
            problem.c,
            integrality=problem.integrality,
            bounds=Bounds(lo, hi),
            constraints=problem.constraints,
        )
        assert (solved.status == 0) == report.feasible, (bits, report.violations)
        if report.feasible:
            expected.append(report.objective)
    result = problem.solve(time_limit=2)
    if expected:
        assert result.objective == max(expected)
        assert result.validation.feasible
        assert result.relative_gap == pytest.approx(0)
    else:
        assert result.status == "infeasible"
        assert result.schedule is None


def test_fixed_cells_and_best_incumbent_are_preserved():
    p = ONTSMilp(make_instance())
    incumbent = torch.tensor([[1, 1, 0, 0, 0]])
    free = torch.tensor([[False, False, False, True, True]])
    result = p.solve(incumbent=incumbent, free_mask=free)
    assert result.restricted
    assert torch.equal(result.schedule[~free], incumbent.bool()[~free])
    assert result.objective == 8
    held = p.solve(incumbent=result.schedule, free_mask=torch.zeros_like(free))
    assert held.objective == result.objective


def test_nearest_proposal_repair_produces_no_objective_bound():
    result = ONTSMilp(make_instance()).solve(proposal=torch.zeros(1, 5))
    assert result.validation.feasible
    assert result.upper_bound is None
    assert result.relative_gap is None


@pytest.mark.parametrize("selector", ["exact", "random", "heuristic"])
def test_end_to_end_validated_and_monotone(selector):
    result = solve_hybrid(
        make_instance(),
        selector=selector,
        time_limit=0.5,
        repair_time=0.1,
        initial_schedule=torch.tensor([[1, 1, 0, 0, 0]]),
    )
    assert validate_schedule(make_instance(), result.schedule).feasible
    assert result.objective == 8
    objectives = [
        row["objective"] for row in result.history if row["objective"] is not None
    ]
    assert objectives == sorted(objectives)
    assert result.upper_bound >= result.objective


def test_infeasible_is_distinct_from_unknown():
    instance = make_instance(win_min=[4], win_max=[4])
    result = solve_hybrid(instance, time_limit=1)
    assert result.schedule is None
    assert result.status == "infeasible"


def test_timeout_keeps_incumbent_and_restricted_bound_is_not_global(monkeypatch):
    from projects.scheduling.milp import SolveResult

    incumbent = torch.tensor([[1, 1, 0, 0, 0]])
    calls = []

    def fake_solve(self, **kwargs):
        restricted = kwargs.get("free_mask") is not None
        calls.append(restricted)
        return SolveResult(
            "limit",
            "timeout",
            incumbent.bool(),
            4,
            4 if restricted else 100,
            0,
            0.01,
            self.config.validate(self.instance, incumbent),
            restricted,
        )

    monkeypatch.setattr(ONTSMilp, "solve", fake_solve)
    result = solve_hybrid(
        make_instance(), initial_schedule=incumbent, time_limit=2, max_iterations=2
    )
    assert calls == [False, True, True]
    assert result.upper_bound == 100
    assert result.objective == 4
    assert result.status == "feasible"


def test_graph_policy_backward_and_checkpoint(tmp_path):
    p = ONTSMilp(make_instance())
    incumbent = torch.tensor([[1, 1, 0, 0, 0]])
    graph = GraphBuilder(p).build(incumbent)
    assert graph.constraints.shape[0] == p.A.shape[0]
    assert graph.coefficients.numel() == p.A.nnz
    policy = NeighborhoodPolicy(hidden=16, layers=1)
    logits = policy(graph)
    assert logits.shape == (p.n_x,)
    logits.sum().backward()
    assert policy.head[-1].weight.grad.abs().sum() > 0
    path = tmp_path / "policy.pt"
    policy.save(path, metadata={"trained": True})
    loaded = NeighborhoodPolicy.load(path)
    assert torch.allclose(policy(graph), loaded(graph))
    assert loaded.training_metadata["trained"]


def test_real_imitation_training_and_neural_search():
    instance = make_instance()
    policy, report = train_neighborhood_policy(
        [instance],
        rounds=2,
        epochs=2,
        init_time=1,
        repair_time=1,
        fraction=1,
        hidden=16,
        layers=1,
    )
    assert report["trained"]
    assert len(report["loss"]) == 2
    assert all(np.isfinite(report["loss"]))
    result = solve_hybrid(
        instance,
        selector="neural",
        policy=policy,
        initial_schedule=torch.tensor([[1, 1, 0, 0, 0]]),
        time_limit=1,
    )
    assert result.objective == 8


def test_neural_selector_is_used_when_global_optimum_is_unknown(monkeypatch):
    from projects.scheduling.milp import SolveResult

    incumbent = torch.tensor([[1, 1, 0, 0, 0]])

    def fake_solve(self, **kwargs):
        return SolveResult(
            "limit",
            "timeout",
            incumbent.bool(),
            4,
            100,
            0.96,
            0.01,
            self.config.validate(self.instance, incumbent),
            kwargs.get("free_mask") is not None,
        )

    monkeypatch.setattr(ONTSMilp, "solve", fake_solve)
    policy = NeighborhoodPolicy(hidden=16, layers=1)
    called = []
    hook = policy.register_forward_hook(lambda *args: called.append(True))
    try:
        result = solve_hybrid(
            make_instance(),
            selector="neural",
            policy=policy,
            initial_schedule=incumbent,
            time_limit=2,
            max_iterations=2,
            exploration=0,
        )
    finally:
        hook.remove()
    assert len(called) == 2
    assert result.iterations == 2


def test_neighborhood_budget_and_input_validation():
    mask = select_neighborhood(
        np.arange(15, dtype=float), J=3, T=5, fraction=0.4, block_size=3
    )
    assert mask.dtype == torch.bool
    assert mask.sum() == 6
    with pytest.raises(ValueError):
        solve_hybrid(make_instance(), selector="neural")
    with pytest.raises(ValueError):
        ONTSMilp(make_instance()).solve(free_mask=mask)


def test_existing_model_proposal_adapter_restores_training_mode():
    class Model(torch.nn.Module):
        def act(self, obs, temperature):
            return {"new_starts": torch.ones_like(obs["task_draw"])}

    model = Model()
    proposal = rollout_proposal(model)(make_instance())
    assert proposal.shape == (1, 5)
    assert model.training
    result = solve_hybrid(make_instance(), proposal_fn=lambda _: proposal, time_limit=1)
    assert result.schedule is not None
    assert validate_schedule(make_instance(), result.schedule).feasible


def test_windows_can_extend_beyond_finite_horizon():
    instance = make_instance(win_max=[20])
    result = ONTSMilp(instance).solve(time_limit=1)
    assert result.validation.feasible
    assert result.objective == 8


def test_bipartite_decoder_preserves_graph_first_and_last():
    from models.adapters import BipartiteDecoderAdapter
    from models.bipartite_gnn import BipartiteDecoder

    adapter = BipartiteDecoderAdapter(BipartiteDecoder(d_model=4))
    graph = torch.randn(3, 4)
    first, last = torch.randn(3, 4), torch.randn(3, 4)
    context = adapter.initialize_context(graph)
    context = adapter.update_context(context, first)
    context = adapter.update_context(context, last)
    assert torch.equal(context[:, :4], graph)
    assert torch.equal(context[:, 4:8], first)
    assert torch.equal(context[:, 8:], last)


def test_greedy_decode_is_finite_and_gathers_correct_batch_rows():
    from models.nco_model import NCOModel

    class Encoder(torch.nn.Module):
        d_model = 2

        def encode(self, obs):
            return obs["embeddings"], torch.zeros(3, 2)

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.picked = None

        def create_context(self):
            return torch.zeros(3, 2)

        def attention_logits(self, ctx, nodes):
            self.calls += 1
            if self.calls == 1:
                return torch.tensor(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 10.0], [0.0, 10.0, 0.0]]
                ), torch.tensor([10.0, -10.0, -10.0])
            return torch.zeros(3, 3), torch.full((3,), 10.0)

        def update_context(self, ctx, picked):
            self.picked = picked
            return ctx

    nodes = torch.arange(18, dtype=torch.float32).reshape(3, 3, 2)
    decoder = Decoder()
    model = NCOModel(Encoder(), decoder)
    result = model.act(
        {
            "embeddings": nodes,
            "task_static": nodes,
            "mask": torch.ones(3, 3, dtype=torch.bool),
        },
        temperature=0,
    )
    assert torch.isfinite(result["logp"]).all()
    assert torch.equal(decoder.picked, torch.stack([nodes[1, 2], nodes[2, 1]]))


def test_generator_uses_same_curtailment_model():
    from projects.scheduling.environments.onts_env import onts_instance_to_json_data
    from projects.scheduling.instances.generate_onts_instances import _feasibility_milp

    result = _feasibility_milp(
        onts_instance_to_json_data(make_instance(recurso_p=[100] * 5)),
        soc_inicial=0.7,
        limite_inferior=0,
        ef=0.9,
        v_bat=3.6,
        q=5,
        bat_usage=5,
        include_valid_inequalities=False,
        time_limit=1,
    )
    assert result.feasible


def test_coupled_two_job_energy_problem_matches_enumeration():
    instance = make_instance(
        priority=[3, 1],
        uso_p=[15, 10],
        min_statup=[0, 0],
        max_statup=[2, 2],
        min_cpu_time=[1, 1],
        max_cpu_time=[2, 2],
        min_periodo_job=[1, 1],
        max_periodo_job=[4, 4],
        win_min=[0, 0],
        win_max=[3, 3],
        recurso_p=[5, 0, 8],
    )
    config = ONTSConfig(soc_inicial=0.02)
    expected = []
    for bits in product([0, 1], repeat=6):
        x = torch.tensor(bits).reshape(2, 3)
        report = config.validate(instance, x)
        if report.feasible:
            expected.append(report.objective)
    result = ONTSMilp(instance, config).solve(time_limit=2)
    assert result.objective == max(expected)
    assert result.validation.feasible


def test_no_incumbent_timeout_is_unknown(monkeypatch):
    from projects.scheduling.milp import SolveResult

    def fake_solve(self, **kwargs):
        return SolveResult("limit", "timeout", None, None, 100, None, 0.01, None)

    monkeypatch.setattr(ONTSMilp, "solve", fake_solve)
    result = solve_hybrid(make_instance(), time_limit=1, max_iterations=1)
    assert result.status == "unknown"
    assert result.schedule is None
    assert result.upper_bound == 100
    assert result.relative_gap is None
