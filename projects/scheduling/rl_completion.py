"""Completion supervision and conservative actor-critic for recovery search.

Learned completion scores only rank candidates. Unknown/censored branches are
never supervised as impossible, and learned scores never replace exact probes.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .rl_architectures import ArchitecturePolicy, observation


class CompletionPolicy(nn.Module):
    def __init__(self, width=32, guidance=.5):
        super().__init__()
        self.actor = ArchitecturePolicy('mlp', width)
        self.completion = nn.Sequential(nn.Linear(20,64),nn.Tanh(),nn.Linear(64,1))
        self.value = nn.Sequential(nn.Linear(40,64),nn.Tanh(),nn.Linear(64,1))
        self.guidance = guidance

    def completion_logits(self, features):
        return self.completion(torch.as_tensor(features,dtype=torch.float32)).squeeze(-1)

    def state_value(self, features):
        features = torch.as_tensor(features,dtype=torch.float32)
        return 2*torch.sigmoid(self.value(torch.cat([features.mean(0),features.max(0).values]))).squeeze()

    def forward(self, raw):
        # Classifier gradients come from evidence, not the policy objective.
        return self.actor(raw)+self.guidance*F.logsigmoid(self.completion_logits(raw['features'])).detach()

    def score_choices(self, env, choices, features):
        return self(observation(env,choices,features))


def branch_labels(row):
    """Map proposal event indices to witnessed completion / proven menu failure.

    A backtrack exhausts that node's available menu; this is a failure within
    the restricted search, not an ONTS infeasibility certificate. Duplicates and
    interrupted branches remain unknown unless a credited completion witnesses
    them. Each key uses the full task-profile prefix, not task IDs alone.
    """
    labels, paths, active = {}, {}, []
    def action_key(task,profile): return int(task),tuple(profile)
    for index,event in enumerate(row['events']):
        if event['kind']=='proposal':
            active=active[:event['depth']]
            prefix=tuple(active+[action_key(event['task'],event['profile'])])
            paths[index]=prefix
            if event['probe']['status']=='reject': labels[index]=0
            if event.get('placed'): active=list(prefix)
        elif event['kind']=='complete' and event['credited']:
            full=tuple(action_key(a['task'],a['profile']) for a in event['actions'])
            for proposal,prefix in paths.items():
                if full[:len(prefix)]==prefix: labels[proposal]=1
        elif event['kind']=='backtrack':
            exhausted=tuple(active[:event['depth']])
            for proposal,prefix in paths.items():
                if prefix==exhausted and labels.get(proposal)!=1:
                    labels[proposal]=0
            active=active[:max(0,event['depth']-1)]
    return labels


class TransitionCollector:
    """Temporary on-policy snapshots; raw tensors are not retained in run JSON."""
    def __init__(self, policy):
        self.policy=policy; self.records=[]

    def __call__(self, env, choices, features, index, logits):
        with torch.no_grad():
            raw=observation(env,choices,features)
            self.records.append(dict(observation=raw,index=index,
                old_log_prob=float(logits.log_softmax(0)[index]),
                old_value=float(self.policy.state_value(features)) if isinstance(self.policy,CompletionPolicy) else 0.))

    def finish(self,row):
        proposals=[i for i,event in enumerate(row['events']) if event['kind']=='proposal']
        if len(proposals)!=len(self.records): raise ValueError('Transition/proposal alignment mismatch')
        labels=branch_labels(row)
        for record,event_index in zip(self.records,proposals):
            record.update(event_index=event_index,target=row['reward'],completion_target=labels.get(event_index))
        return self.records


def completion_loss(model, examples):
    if not examples: return next(model.parameters()).sum()*0
    features=torch.tensor([e['features'] for e in examples],dtype=torch.float32)
    target=torch.tensor([e['label'] for e in examples],dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(model.completion_logits(features),target)


def examples_from(records):
    return [dict(features=r['observation']['features'][r['index']].tolist(),label=r['completion_target'],event_index=r['event_index'])
            for r in records if r['completion_target'] is not None]


def conservative_update(model, anchor, optimizer, records, labels, examples, *, policy_gradient=True,
                        clip=.2, kl_limit=.02, anchor_weight=.1):
    """One PPO/BC step, with exact sampled-state actor KL and atomic rollback.

    PPO uses a learned state baseline and undiscounted terminal returns.
    Per-search sums retain the variable-length search policy-gradient objective;
    advantages are not centered using the same sampled actions. The critic sees
    successful and failed budget-limited searches. It predicts search return,
    while the separate completion head predicts witnessed branch completion.
    """
    if not records: return dict(updated=False,reason='no_decisions')
    policy_terms=[]; value_terms=[]; anchor_terms=[]
    for record in records:
        raw=record['observation']; actor_logits=model.actor(raw)
        with torch.no_grad(): anchor_log=anchor(raw).log_softmax(0)
        actor_log=actor_logits.log_softmax(0)
        anchor_terms.append((anchor_log.exp()*(anchor_log-actor_log)).sum())
        if policy_gradient:
            logits=actor_logits+model.guidance*F.logsigmoid(model.completion_logits(raw['features'])).detach()
            ratio=torch.exp(logits.log_softmax(0)[record['index']]-record['old_log_prob'])
            advantage=record['target']-record['old_value']
            policy_terms.append(-torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage))
            value_terms.append(F.smooth_l1_loss(model.state_value(raw['features']),torch.tensor(record['target'])))
    bc=torch.stack([F.cross_entropy(model.actor(d['observation'])[None],torch.tensor([d['index']])) for d in labels]).mean()
    objective=(torch.stack(policy_terms).sum()/max(1,len({r['trajectory'] for r in records})) if policy_terms else bc*0)
    objective=objective+.1*bc+anchor_weight*torch.stack(anchor_terms).mean()+.5*completion_loss(model,examples)
    if not policy_gradient: objective=objective+.9*bc
    if value_terms: objective=objective+.5*torch.stack(value_terms).mean()
    before=copy.deepcopy(model.state_dict()); optimizer_before=copy.deepcopy(optimizer.state_dict())
    optimizer.zero_grad(); objective.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    with torch.no_grad():
        kl=torch.stack([(anchor(r['observation']).softmax(0)*(anchor(r['observation']).log_softmax(0)-model.actor(r['observation']).log_softmax(0))).sum() for r in records]).mean()
    accepted=bool(torch.isfinite(kl) and kl<=kl_limit and all(torch.isfinite(p).all() for p in model.parameters()))
    if not accepted:
        model.load_state_dict(before);optimizer.load_state_dict(optimizer_before)
    return dict(updated=accepted,loss=float(objective.detach()),actor_anchor_kl=float(kl),kl_limit=kl_limit,
                positive_labels=sum(e['label']==1 for e in examples),negative_labels=sum(e['label']==0 for e in examples))
