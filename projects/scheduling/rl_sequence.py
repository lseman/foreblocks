"""Ordered, context-dependent task groups with exact composite likelihoods."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .rl_bipartite import bipartite_observation
from .rl_learning import ppo_batch_update
from .rl_portfolio import PortfolioAction, PortfolioEnv
from .rl_subset import SubsetActorCritic, SubsetEnv


class SequenceEnv(SubsetEnv):
    def __init__(self, *args, mixed=True, **kwargs):
        self.mixed = mixed
        super().__init__(*args, **kwargs)
        if len(self.repair_budgets) != 2:
            raise ValueError('The sequence policy supports exactly two MILP allowances')

    def refresh(self):
        mask = torch.ones_like(self.incumbent)
        jobs = tuple(range(self.problem.J))
        self.actions = [PortfolioAction(4, mask, torch.zeros(10), jobs)]
        if self.mixed:
            self.actions += [PortfolioAction(0, mask, torch.zeros(10), jobs, allowance=a)
                             for a in self.repair_budgets]
        self.done = self.remaining() < self.profile_floor

    def step_sequence(self, action):
        action = torch.as_tensor(action).detach().cpu()
        j = self.problem.J
        if action.shape != (j+2,) or not torch.equal(action, action.long()):
            raise ValueError('Expected an integer ordered action')
        operator, count = int(action[0]), int(action[1])
        jobs = action[2:2+count].tolist()
        if not 0 <= operator < len(self.actions) or not 1 <= count <= j:
            raise ValueError('Invalid repair operator or cardinality')
        if len(set(jobs)) != count or any(not 0 <= job < j for job in jobs) or (action[2+count:] != -1).any():
            raise ValueError('Invalid task sequence or padding')
        mask = torch.zeros_like(self.incumbent)
        mask[jobs] = True
        prototype = self.actions[operator]
        self.actions[operator] = PortfolioAction(prototype.family, mask, torch.zeros(10),
                                                 tuple(sorted(jobs)), allowance=prototype.allowance)
        reward, done = PortfolioEnv.step(self, operator+1)
        self.log[-1]['ordered_action'] = action.long().tolist()
        return reward, done


class SequenceDistribution:
    """A distribution over operator, cardinality and an ordered task sequence.

    Repair consumes the set, but PPO retains its sampled latent order. Sequence
    probability is not confused with the sum over all orders of an unordered set.
    """
    def __init__(self, policy, tasks, context, mask):
        self.policy, self.tasks, self.context = policy, tasks, context
        self.width = len(tasks)
        available = torch.zeros(3, dtype=torch.bool, device=tasks.device)
        available[:len(mask)-1] = mask[1:].to(tasks.device)
        if not available.any():
            raise ValueError('No affordable repair action')
        self.operator = Categorical(logits=policy.operator(context).masked_fill(~available, -torch.inf))

    def cardinality(self, operator):
        j = self.width
        mode = nn.functional.one_hot(torch.as_tensor(operator, device=self.tasks.device),3).float()
        fractions = torch.arange(1,j+1,device=self.tasks.device).float()/j
        inputs = torch.cat([self.context.expand(j,-1),mode.expand(j,-1),fractions[:,None]],1)
        return Categorical(logits=self.policy.count(inputs).squeeze(-1))

    def pointer(self, operator, count, prefix):
        j, tasks = self.width, self.tasks
        chosen = torch.zeros(j,dtype=torch.bool,device=tasks.device)
        if prefix:
            chosen[prefix] = True
        selected = tasks[chosen].mean(0) if prefix and self.policy.correlated else torch.zeros_like(tasks[0])
        mode = nn.functional.one_hot(torch.as_tensor(operator,device=tasks.device),3).float()
        controls = torch.cat([mode,tasks.new_tensor([count/j,len(prefix)/j])])
        features = torch.cat([tasks,selected.expand(j,-1),self.context.expand(j,-1),controls.expand(j,-1)],1)
        return Categorical(logits=self.policy.pointer(features).squeeze(-1).masked_fill(chosen,-torch.inf))

    def generate(self, *, greedy=False):
        choose = lambda dist: int(dist.logits.argmax()) if greedy else int(dist.sample())
        operator = choose(self.operator)
        count = choose(self.cardinality(operator))+1
        prefix = []
        for _ in range(count):
            prefix.append(choose(self.pointer(operator,count,prefix)))
        return torch.tensor([operator,count,*prefix,*([-1]*(self.width-count))],device=self.tasks.device)

    def sample(self):
        return self.generate()

    def log_prob(self, action):
        action = torch.as_tensor(action,device=self.tasks.device)
        if action.shape != (self.width+2,) or not torch.equal(action,action.long()):
            raise ValueError('Invalid ordered action')
        operator,count = int(action[0]),int(action[1])
        if not 0 <= operator < 3 or not 1 <= count <= self.width:
            raise ValueError('Invalid operator or group size')
        order = action[2:2+count].tolist()
        if len(set(order)) != count or any(not 0 <= j < self.width for j in order) or (action[2+count:] != -1).any():
            raise ValueError('Invalid task order')
        result = self.operator.log_prob(action[0])+self.cardinality(operator).log_prob(action[1]-1)
        prefix = []
        for job in order:
            result = result+self.pointer(operator,count,prefix).log_prob(action.new_tensor(job))
            prefix.append(job)
        return result


class SequenceActorCritic(SubsetActorCritic):
    def __init__(self, hidden=24, *, correlated=True):
        super().__init__(hidden=hidden)
        del self.actor
        self.correlated = correlated
        self.operator = nn.Linear(2*hidden+8,3)
        self.count = nn.Sequential(nn.Linear(2*hidden+12,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.pointer = nn.Sequential(nn.Linear(4*hidden+13,hidden),nn.Tanh(),nn.Linear(hidden,1))
        for layer in (self.operator,self.count[-1],self.pointer[-1]):
            nn.init.orthogonal_(layer.weight,.01)
            nn.init.zeros_(layer.bias)

    def forward(self, observation, mask):
        device = next(self.parameters()).device
        task = self.task(observation.tasks.to(device))
        time = self.time(observation.times.to(device))
        eligible = observation.eligible.to(device).float()
        message = ((time[None]+self.edge(observation.edges.to(device)))*eligible[...,None]).sum(1)/eligible.sum(1).clamp(min=1)[:,None]
        task = task+message
        context = torch.cat([task.mean(0),time.mean(0),observation.rows[0,19:27].to(device)])
        return SequenceDistribution(self,task,context,mask),self.critic(context).squeeze(-1)

    def save(self, path, metadata=None):
        torch.save(dict(format='onts-sequence-v1',hidden=self.hidden,correlated=self.correlated,
                        state_dict=self.state_dict(),metadata=metadata or {}),path)

    @classmethod
    def load(cls,path):
        saved = torch.load(path,map_location='cpu',weights_only=True)
        if saved['format'] != 'onts-sequence-v1':
            raise ValueError('Unsupported sequence checkpoint')
        model = cls(saved['hidden'],correlated=saved['correlated'])
        model.load_state_dict(saved['state_dict'])
        return model.eval()


def sequence_rollout(env, policy=None, *, greedy=False):
    trajectory = []
    while not env.done:
        tick = env.clock()
        obs,mask = bipartite_observation(env)
        if mask[0]:
            env.step(0)
            env.log[-1]['selection_runtime'] = env.clock()-tick
            break
        if policy is None:
            operator = int(env.selection_rng.choice(torch.where(mask[1:])[0].tolist()))
            count = int(env.selection_rng.integers(1,env.problem.J+1))
            jobs = env.selection_rng.choice(env.problem.J,count,replace=False).tolist()
            action = [operator,count,*jobs,*([-1]*(env.problem.J-count))]
            logp,value = 0.,0.
        else:
            with torch.no_grad():
                dist,estimate = policy(obs,mask)
                tensor = dist.generate(greedy=greedy)
                action = tensor.cpu().tolist()
                logp,value = float(dist.log_prob(tensor)),float(estimate)
        selection_time = env.clock()-tick
        reward,_ = env.step_sequence(action)
        env.log[-1]['selection_runtime'] = selection_time
        trajectory.append((obs,mask,action,logp,value,reward))
    return trajectory


def sequence_imitation(policy, optimizer, examples, *, epochs=1, seed=0):
    rng,losses = np.random.default_rng(seed),[]
    for _ in range(epochs):
        for i in rng.permutation(len(examples)):
            obs,mask,operator,jobs,weight = examples[i]
            order = rng.permutation(jobs).tolist()
            action = [operator,len(jobs),*order,*([-1]*(len(obs.tasks)-len(jobs)))]
            dist,_ = policy(obs,mask)
            loss = -float(weight)*dist.log_prob(action)/(len(jobs)+2)
            optimizer.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(),.5,error_if_nonfinite=True)
            optimizer.step();losses.append(float(loss.detach()))
    return dict(updates=len(losses),loss=float(np.mean(losses)) if losses else 0.)


class _ZeroEntropy:
    """Compatibility wrapper: a zero regularizer, not a joint entropy estimate."""
    def __init__(self,dist):
        self.dist = dist
    def log_prob(self,action):
        return self.dist.log_prob(action)
    def entropy(self):
        return self.dist.context.sum()*0


class _NoEntropyPolicy(nn.Module):
    def __init__(self,policy):
        super().__init__();self.policy = policy
    def forward(self,obs,mask):
        dist,value = self.policy(obs,mask)
        return _ZeroEntropy(dist),value


def sequence_ppo_update(policy,optimizer,trajectories,**kwargs):
    if kwargs.pop('entropy_coef',0) != 0:
        raise ValueError('Sequence PPO disables entropy regularization')
    metrics = ppo_batch_update(_NoEntropyPolicy(policy),optimizer,trajectories,entropy_coef=0,**kwargs)
    metrics['entropy'] = None
    return metrics


def rloo_update(policy,optimizer,groups):
    """One on-policy REINFORCE update; other rollout returns supply the baseline."""
    losses,advantages = [],[]
    for trajectories in groups:
        if len(trajectories)<2:
            raise ValueError('Leave-one-out baselines need at least two rollouts')
        rewards = [sum(t[-1] for t in trajectory) for trajectory in trajectories]
        for i,trajectory in enumerate(trajectories):
            advantage = rewards[i]-(sum(rewards)-rewards[i])/(len(rewards)-1)
            if not trajectory:
                continue
            score = sum(policy(obs,mask)[0].log_prob(action) for obs,mask,action,*_ in trajectory)
            losses.append(-advantage*score)
            advantages.append(advantage)
    if not losses:
        return dict(updates=0)
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite leave-one-out loss')
    optimizer.zero_grad();loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(),.5,error_if_nonfinite=True)
    optimizer.step()
    return dict(updates=1,loss=float(loss.detach()),advantage_std=float(np.std(advantages)))
