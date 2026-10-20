from dataclasses import replace

import pytest
import torch

from projects.scheduling.rl_architectures import ARCHITECTURES, ArchitecturePolicy, observation
from projects.scheduling.rl_construct import ConstructionEnv
from projects.scheduling.rl_recovery import SearchNode, recover_construct, recovery_features
from test_rl_construct import tiny


def state(instance):
    env = ConstructionEnv(instance)
    choices = env.candidates()
    node = SearchNode(env.schedule,env.remaining,[],[],initial_choices=len(choices))
    return env,choices,recovery_features(env,node,choices,1.,2.)


@pytest.mark.parametrize('architecture',ARCHITECTURES)
def test_architectures_variable_actions_and_recovery_gradients(architecture):
    torch.set_num_threads(1); torch.manual_seed(7)
    policy = ArchitecturePolicy(architecture)
    env,choices,features = state(tiny())
    logits = policy.score_choices(env,choices,features)
    assert logits.shape == (len(choices),) and torch.isfinite(logits).all()
    row,logs,entropies = recover_construct(tiny(),policy,budget=2.,training=True,max_decisions=32,
                                         record_decisions=True,record_observation=True)
    assert row['success'] and logs
    (-torch.stack(logs).sum()-.001*torch.stack(entropies).sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())
    for action,decision in zip(row['actions'],row['decisions']):
        o = decision['observation']; k = decision['index']
        assert o['task_indices'][k] == action['task'] and o['profiles'][k] == action['profile']
        assert o['features'] == decision['features']
        assert torch.isfinite(policy(o)).all()


@pytest.mark.parametrize('architecture',['mlp','bipartite','transformer'])
def test_unordered_task_architectures_are_permutation_equivariant(architecture):
    torch.set_num_threads(1); torch.manual_seed(7)
    original = tiny(); order = torch.tensor([1,0])
    permuted = replace(original,**{k:v[order] for k,v in vars(original).items() if k != 'recurso_p'})
    policy = ArchitecturePolicy(architecture).eval()
    a,ca,fa = state(original); b,cb,fb = state(permuted)
    la,lb = policy.score_choices(a,ca,fa),policy.score_choices(b,cb,fb)
    mapped = {(int(order[c.task]),c.row.tobytes()):float(logit.detach()) for c,logit in zip(cb,lb)}
    for c,logit in zip(ca,la):
        assert abs(float(logit.detach())-mapped[c.task,c.row.tobytes()]) < 1e-5


def test_milp_baseline_counts_setup_and_late_output_as_failure(monkeypatch):
    from types import SimpleNamespace
    from projects.scheduling.milp import ONTSConfig
    from projects.scheduling.research import rl_architecture_study as study
    clock = [0.]
    schedule = torch.tensor([[1,1,0,0,0,0],[0,0,1,1,0,0]],dtype=torch.bool)
    class Problem:
        def __init__(self,instance,config):
            clock[0] += .2
        def solve(self,*,time_limit,solver_threads):
            assert abs(time_limit-.75)<1e-6 and solver_threads==1
            clock[0] += .9
            return SimpleNamespace(schedule=schedule,status='optimal',upper_bound=6.)
    monkeypatch.setattr(study,'perf_counter',lambda:clock[0])
    monkeypatch.setattr(study,'ONTSMilp',Problem)
    row = study.milp_baseline(tiny(),ONTSConfig(),1.)
    assert row['feasible'] and row['status']=='optimal'
    assert not row['success'] and row['utility']==0 and abs(row['overrun']-.1)<1e-6
