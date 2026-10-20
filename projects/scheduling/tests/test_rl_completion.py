import copy

import torch

from projects.scheduling.rl_completion import CompletionPolicy, TransitionCollector, branch_labels, conservative_update, examples_from
from projects.scheduling.rl_recovery import recover_construct
from test_rl_construct import tiny


def test_critic_collects_all_decisions_and_labels_witnessed_paths():
    torch.set_num_threads(1);torch.manual_seed(4)
    policy=CompletionPolicy();collector=TransitionCollector(policy)
    row,_,_=recover_construct(tiny(),policy,budget=.5,max_decisions=16,decision_callback=collector)
    records=collector.finish(row)
    assert row['success'] and len(records)==row['sampled_decisions']
    assert any(r['completion_target']==1 for r in records)
    assert all(torch.isfinite(policy(r['observation'])).all() for r in records)
    assert 0<float(policy.state_value(records[0]['observation']['features']).detach())<2


def test_timeout_and_duplicate_are_not_negative_completion_labels():
    row=dict(events=[dict(kind='proposal',depth=0,task=0,profile=[1,0],probe=dict(status='unknown')),
                     dict(kind='proposal',depth=0,task=1,profile=[0,1],probe=dict(status='duplicate_state'))])
    assert branch_labels(row)=={}


def test_labels_include_profiles_and_proven_exhaustion():
    row=dict(events=[dict(kind='proposal',depth=0,task=0,profile=[1,0],probe=dict(status='not_ruled_out'),placed=True),
        dict(kind='backtrack',depth=1),
        dict(kind='proposal',depth=0,task=0,profile=[0,1],probe=dict(status='not_ruled_out'),placed=True),
        dict(kind='complete',credited=True,actions=[dict(task=0,profile=[0,1])])])
    assert branch_labels(row)=={0:0,2:1}


def test_conservative_update_rolls_back_model_and_optimizer():
    torch.set_num_threads(1);torch.manual_seed(4)
    model=CompletionPolicy();anchor=copy.deepcopy(model.actor).eval()
    collector=TransitionCollector(model)
    row,_,_=recover_construct(tiny(),model,budget=.5,max_decisions=8,record_decisions=True,record_observation=True,decision_callback=collector)
    records=collector.finish(row)
    for r in records:r['trajectory']=0
    optimizer=torch.optim.Adam(model.parameters(),lr=.1)
    before=copy.deepcopy(model.state_dict())
    result=conservative_update(model,anchor,optimizer,records,row['decisions'],examples_from(records),kl_limit=0.)
    assert not result['updated']
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items())
    assert optimizer.state_dict()['state']=={}


def test_accepted_actor_critic_update_has_finite_gradients_and_bound():
    torch.set_num_threads(1);torch.manual_seed(4)
    model=CompletionPolicy();anchor=copy.deepcopy(model.actor).eval()
    collector=TransitionCollector(model)
    row,_,_=recover_construct(tiny(),model,budget=.5,max_decisions=8,record_decisions=True,record_observation=True,decision_callback=collector)
    records=collector.finish(row)
    for r in records:r['trajectory']=0
    result=conservative_update(model,anchor,torch.optim.Adam(model.parameters(),lr=.0001),records,row['decisions'],examples_from(records))
    assert result['updated'] and result['actor_anchor_kl']<=.02
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_learned_pessimism_does_not_prune_feasible_actions():
    torch.set_num_threads(1)
    model=CompletionPolicy()
    with torch.no_grad():
        model.completion[-1].weight.zero_();model.completion[-1].bias.fill_(-100.)
    row,_,_=recover_construct(tiny(),model,budget=.5,max_decisions=32)
    assert row['success']


def test_placed_but_interrupted_branch_has_unknown_label():
    row=dict(events=[dict(kind='proposal',depth=0,task=0,profile=[1,0],probe=dict(status='not_ruled_out'),placed=True)])
    assert branch_labels(row)=={}
