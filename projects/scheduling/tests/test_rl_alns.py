from dataclasses import replace

import pytest
import torch

from projects.scheduling.research.datasets import witness_instance
from projects.scheduling.rl_alns import AdaptiveLNS, DRALNSActorCritic, LiteratureSearchEnv, literature_rollout
from projects.scheduling.rl_learning import ppo_batch_update
from projects.scheduling.rl_lns import RLLNSActorCritic
from projects.scheduling.rl_mistar import MistarActorCritic
from projects.scheduling.rl_resched import ReschedActorCritic


def environment(seed=7721, **kwargs):
    torch.set_num_threads(1)
    inst,x = witness_instance(seed=seed,regime='balanced')
    return LiteratureSearchEnv(inst,x,budget=3.,**kwargs)


def test_current_can_decrease_without_degrading_best_or_reward(monkeypatch):
    env = environment()
    initial = env.current.clone()
    improved,_,_ = __import__('projects.scheduling.job_profile',fromlist=['reprofile']).reprofile(
        env.problem,initial,range(env.problem.J),env.rng)
    assert env.problem.config.validate(env.problem.instance,improved).objective>env.initial_objective
    env.current = improved.clone();env.best = improved.clone()
    env.current_objective = env.best_objective = env.problem.config.validate(env.problem.instance,improved).objective
    a = env.actions[0];env.actions[0] = replace(a,temperature=1e6,jobs=tuple(range(env.problem.J)))
    monkeypatch.setattr(env,'_repair',lambda action,base:(initial,'test_perturb'))
    reward,_ = env.step_actions(0)
    assert reward==0 and env.log[-1]['accepted']
    assert torch.equal(env.current,initial.bool()) and torch.equal(env.best,improved)
    assert env.current_objective<env.best_objective
    assert env.log[-1]['acceptance_probability']<1


def test_batch_uses_same_base_and_expired_invalid_candidates_never_update(monkeypatch):
    env=environment();base=env.current.clone();seen=[]
    def repair(action,schedule):
        seen.append(schedule.clone());return schedule,'same'
    monkeypatch.setattr(env,'_repair',repair)
    env.step_actions([0,0,1])
    assert len(seen)==3 and all(torch.equal(x,base) for x in seen)
    assert env.log[-1]['action']==[0,0,1] and len(env.history)==3
    monkeypatch.setattr(env,'_repair',lambda action,schedule:(torch.zeros_like(schedule),'invalid'))
    env.step_actions(0)
    assert not env.log[-1]['accepted'] and torch.equal(env.current,base)
    # Return a feasible candidate only after the guard; it must never be eligible.
    def late(action,schedule):
        env.deadline=env.clock()-1;return schedule,'late'
    monkeypatch.setattr(env,'_repair',late)
    env.step_actions(0)
    assert not env.log[-1]['proposals'][0]['eligible'] and not env.log[-1]['accepted']
    assert torch.equal(env.best,base)


def test_real_perturbations_keep_task_and_energy_feasibility_and_memory_bound():
    env=environment(memory_size=2)
    for _ in range(5):
        idx=next(i for i,a in enumerate(env.actions) if a.operator=='perturb' and a.temperature>0)
        old=env.current.clone();group=env.actions[idx].jobs
        env.step_actions(idx)
        assert env.problem.config.validate(env.problem.instance,env.current).feasible
        outside=[j for j in range(env.problem.J) if j not in group]
        assert torch.equal(env.current[outside],old[outside])
        assert env.best_objective>=env.initial_objective and len(env.history)<=2
    obs,_=env.observe();env.current.zero_()
    assert obs.current_schedule.any() and obs.history_schedules.shape[0]==2


@pytest.mark.parametrize('cls,subset',[(ReschedActorCritic,False),(MistarActorCritic,False),(DRALNSActorCritic,False),(RLLNSActorCritic,True)])
def test_common_training_executes_correct_likelihood_and_safe_updates(cls,subset,tmp_path):
    env=environment();model=cls(hidden=8)
    trajectory=literature_rollout(env,model,subset=subset,max_steps=2)
    assert trajectory
    for obs,mask,action,old_logp,_,_ in trajectory:
        dist,_=model(obs,mask)
        assert float(dist.log_prob(torch.tensor(action)).detach())==pytest.approx(old_logp,abs=1e-5)
    metrics=ppo_batch_update(model,torch.optim.Adam(model.parameters(),lr=.001),[trajectory],epochs=1,minibatch_size=2,trace_lambda=1.)
    assert metrics['updates']>0
    assert env.problem.config.validate(env.problem.instance,env.best).feasible
    if isinstance(model,DRALNSActorCritic):
        path=tmp_path/'model.pt';model.save(path);loaded=model.load(path)
        obs,mask=model.observation(environment())
        torch.testing.assert_close(model(obs,mask)[0].logits,loaded(obs,mask)[0].logits)


def test_adaptive_control_updates_and_validation():
    env=environment();controller=AdaptiveLNS(segment=1)
    literature_rollout(env,adaptive=controller,max_steps=3)
    assert controller.weights and all(x>0 for x in controller.weights.values())
    assert env.objective>=env.initial_objective
    with pytest.raises(ValueError):LiteratureSearchEnv(env.problem.instance,env.best,temperatures=(-1,))
    with pytest.raises(ValueError):env.step_actions(torch.tensor([.5]))
    with pytest.raises(ValueError):env.step_subset(torch.zeros(env.problem.J))
