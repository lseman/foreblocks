import itertools

import pytest
import torch

from projects.scheduling.research.datasets import witness_instance
from projects.scheduling.rl_bipartite import BipartiteObservation, bipartite_observation
from projects.scheduling.rl_sequence import (
    SequenceActorCritic, SequenceEnv, rloo_update, sequence_imitation,
    sequence_ppo_update, sequence_rollout,
)


def observation():
    instance,witness=witness_instance(seed=53291,regime='balanced')
    env=SequenceEnv(instance,witness,budget=3)
    return env,*bipartite_observation(env)


def test_ordered_composite_normalizes_over_all_actions_and_masks():
    torch.set_num_threads(1)
    _,obs,mask=observation()
    small=BipartiteObservation(obs.rows,obs.tasks[:3],obs.times,obs.released[:,:3],obs.edges[:3],obs.eligible[:3])
    model=SequenceActorCritic(hidden=8)
    for enabled in (torch.tensor([False,True,True,True]),torch.tensor([False,True,False,False])):
        dist,_=model(small,enabled)
        mass=0.
        for operator in torch.where(enabled[1:])[0].tolist():
            for count in range(1,4):
                for order in itertools.permutations(range(3),count):
                    action=[operator,count,*order,*([-1]*(3-count))]
                    mass+=float(dist.log_prob(action).exp().detach())
        assert mass==pytest.approx(1.,abs=1e-6)
        for _ in range(20):
            action=dist.sample()
            assert enabled[int(action[0])+1]
            assert len(set(action[2:2+int(action[1])].tolist()))==int(action[1])
        with pytest.raises(ValueError):
            dist.log_prob([0,2,1,1,-1])


def test_prefix_conditioning_and_task_permutation():
    torch.manual_seed(411)
    _,obs,mask=observation()
    permutation=torch.tensor([2,0,5,1,4,3])
    inverse=permutation.argsort()
    reordered=BipartiteObservation(obs.rows,obs.tasks[permutation],obs.times,
        obs.released[:,permutation],obs.edges[permutation],obs.eligible[permutation])
    model=SequenceActorCritic()
    dist,value=model(obs,mask)
    other,other_value=model(reordered,mask)
    action=dist.sample()
    transformed=action.clone()
    count=int(action[1]);transformed[2:2+count]=inverse[action[2:2+count]]
    torch.testing.assert_close(other.log_prob(transformed),dist.log_prob(action),atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(value,other_value)
    for correlated in (True,False):
        model.correlated=correlated
        dist,_=model(obs,mask)
        a=dist.pointer(0,3,[0]).logits[2:]
        b=dist.pointer(0,3,[1]).logits[2:]
        # Compare log-odds, since categorical normalizers reflect excluded tasks.
        delta=(a-a[0])-(b-b[0])
        if correlated:
            assert delta.abs().max()>1e-8
        else:
            torch.testing.assert_close(delta,torch.zeros_like(delta),atol=1e-6,rtol=1e-5)


def test_sequence_profile_and_milp_preserve_rows_and_likelihood():
    torch.set_num_threads(1)
    for operator in (0,1,2):
        env,obs,mask=observation()
        before=env.incumbent.clone()
        action=[operator,2,1,0,*([-1]*(env.problem.J-2))]
        env.step_sequence(action)
        assert env.problem.config.validate(env.problem.instance,env.incumbent).feasible
        torch.testing.assert_close(env.incumbent[2:],before[2:])
        assert env.objective>=env.initial_objective
        model=SequenceActorCritic(hidden=8)
        old=model(obs,mask)[0].log_prob(action).detach()
        assert float((model(obs,mask)[0].log_prob(action)-old).exp().detach())==pytest.approx(1.)
        assert env.log[-1]['ordered_action']==action


def test_training_and_safe_checkpoints(tmp_path):
    torch.set_num_threads(1)
    env,obs,mask=observation()
    model=SequenceActorCritic(hidden=8)
    jobs=[0,1]
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    sequence_imitation(model,optimizer,[(obs,mask,0,jobs,1.)],epochs=3,seed=1)
    env.deadline=env.clock()+.12
    trajectory=sequence_rollout(env,model)
    assert trajectory
    metrics=sequence_ppo_update(model,optimizer,[trajectory],epochs=1,minibatch_size=8,trace_lambda=1)
    assert metrics['updates']>0 and metrics['entropy'] is None
    with pytest.raises(ValueError):
        sequence_ppo_update(model,optimizer,[trajectory],entropy_coef=.01)
    path=tmp_path/'sequence.pt';model.save(path)
    loaded=SequenceActorCritic.load(path)
    torch.testing.assert_close(model(obs,mask)[0].log_prob(trajectory[0][2]),loaded(obs,mask)[0].log_prob(trajectory[0][2]))


def test_rloo_baseline_with_equal_returns_has_zero_actor_gradient():
    torch.set_num_threads(1)
    _,obs,mask=observation()
    model=SequenceActorCritic(hidden=8)
    action=[0,1,0,*([-1]*(len(obs.tasks)-1))]
    transition=(obs,mask,action,0.,0.,.1)
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    metrics=rloo_update(model,optimizer,[[[transition],[transition]]])
    assert metrics['advantage_std']==0
    for name,p in model.named_parameters():
        torch.testing.assert_close(before[name],p)
    with pytest.raises(ValueError):
        rloo_update(model,optimizer,[[[transition]]])
