from unittest.mock import patch

import torch

from projects.scheduling.research.datasets import witness_instance
from projects.scheduling.rl_sequence import SequenceActorCritic, SequenceDistribution, SequenceEnv, sequence_rollout
from projects.scheduling.rl_sequence_inference import sequence_inference


class OneDecisionEnv(SequenceEnv):
    def step_sequence(self, action):
        self.done=True
        self.log.append(dict(ordered_action=list(action),reward=0.,executed=True))
        return 0.,True


def test_inference_preserves_sampled_action_without_likelihood_replay():
    torch.set_num_threads(1)
    instance,witness=witness_instance(seed=453921,regime='balanced')
    model=SequenceActorCritic(hidden=8)
    old=OneDecisionEnv(instance,witness,budget=2,clock=lambda:100.)
    fast=OneDecisionEnv(instance,witness,budget=2,clock=lambda:100.)
    torch.manual_seed(51)
    sequence_rollout(old,model)
    torch.manual_seed(51)
    with patch.object(SequenceDistribution,'log_prob',side_effect=AssertionError('Inference must not replay likelihood')):
        returned=sequence_inference(fast,model)
    assert old.log[0]['ordered_action']==fast.log[0]['ordered_action']
    torch.testing.assert_close(returned,witness.bool())


def test_inference_repair_feasibility_and_expired_stop():
    torch.set_num_threads(1)
    instance,witness=witness_instance(seed=936725,regime='energy')
    model=SequenceActorCritic(hidden=8)
    env=SequenceEnv(instance,witness,budget=.2)
    sequence_inference(env,model)
    checked=env.problem.config.validate(instance,env.incumbent)
    assert checked.feasible and checked.objective>=env.initial_objective and env.log
    expired=SequenceEnv(instance,witness,budget=1)
    expired.deadline=expired.clock()-1
    sequence_inference(expired,model)
    assert expired.done and expired.log[-1]['family']=='stop'
