"""Fresh controlled completion-critic / imitation-preserving PPO pilot."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..environments.onts_env import onts_instance_to_json_data
from ..milp import ONTSConfig, ONTSMilp
from ..rl_architectures import ArchitecturePolicy
from ..rl_completion import CompletionPolicy, TransitionCollector, completion_loss, conservative_update, examples_from
from ..rl_recovery import recover_construct
from .datasets import instance_hash, source_hashes
from .oracle import oracle_check
from .rl_architecture_study import milp_baseline, supervise, tight_instance, write

METHODS=('completion-uniform','imitation','replay','reinforce','critic-replay','critic-ppo')


def contrasts(rows):
    result=[]
    for first,second in [('critic-ppo','critic-replay'),('critic-ppo','imitation'),('critic-ppo','replay'),('critic-ppo','reinforce'),('critic-replay','replay')]:
        for split in ('all','test','ood'):
            ids=sorted({r['id'] for r in rows if split=='all' or r['split']==split})
            delta=np.array([np.mean([r['utility'] for r in rows if r['id']==i and r['method']==first])-np.mean([r['utility'] for r in rows if r['id']==i and r['method']==second]) for i in ids])
            samples=np.random.default_rng(1106).choice(delta,size=(4000,len(delta)),replace=True).mean(1)
            result.append(dict(first=first,second=second,split=split,instances=len(ids),difference=float(delta.mean()),interval=np.quantile(samples,[.025,.975]).tolist()))
    return result


def balanced(pool,rng,count=32):
    positive=[e for e in pool if e['label']==1];negative=[e for e in pool if e['label']==0]
    result=[]
    for part in (positive,negative):
        if part: result.extend(part[int(k)] for k in rng.integers(len(part),size=count//2))
    return result


def run(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    config=ONTSConfig(q=.2,bat_usage=.3,limite_inferior=.1)
    data={};manifest=[]
    for split,count,large,offset in [('train',args.train_per_regime,False,0),('test',args.test_per_regime,False,10000),('ood',args.ood_per_regime,True,20000)]:
        data[split]=[]
        for regime_index,regime in enumerate(('balanced','energy','periodic')):
            for n in range(count):
                instance=tight_instance(args.data_seed+offset+regime_index*1000+n,regime,large,config)
                identifier=f'{split}-{regime}-{n}';data[split].append((identifier,instance))
                write(output/f'{identifier}.json',onts_instance_to_json_data(instance))
                manifest.append(dict(id=identifier,hash=instance_hash(instance),split=split,regime=regime,jobs=instance.J,horizon=instance.T))
    prior=set()
    for path in [*Path(args.history_root).glob('**/manifest.json'),*Path(args.history_root).glob('**/protocol.json')]:
        previous=json.loads(path.read_text())
        if isinstance(previous,dict):prior.update(e['hash'] for e in previous.get('entries',previous.get('manifest',[])) if isinstance(e,dict) and 'hash' in e)
    hashes={e['hash'] for e in manifest}
    if len(hashes)!=len(manifest) or hashes&prior:raise ValueError('Instance overlap')
    sources=source_hashes()
    write(output/'protocol.json',dict(arguments=vars(args),config=asdict(config),manifest=manifest,source_hashes=sources,architectures=['mlp'],completion_critic=True,
        primary='critic-ppo minus critic-replay pooled on-time utility, instance-bootstrap exploratory 95% interval; secondary imitation/replay/REINFORCE contrasts',
        scope='Fresh tighter planted family; same pooled MLP actor and shared teachers. Learned completion score only ranks actions. No solver/witness at policy inference. Actor anchor KL is bounded only on collected training states, not globally. PPO/replay share critic examples, actor anchor and optimizer steps, not actor targets or training compute.',
        labels='Positive: credited feasible descendant with exact task-profile prefix. Negative: necessary rejection or exhausted restricted-menu subtree. Timeout/duplicate-only branches unlabeled. State value predicts observed final budget-limited search reward; timeout failure is an outcome, not an infeasibility claim.',
        prior_hashes_checked=len(prior)))
    root=Path(__file__).resolve().parents[1]
    for name,digest in sources.items():
        content=(root/name).read_bytes()
        if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('Source freeze mismatch')
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    print('Protocol and sources frozen; collecting successful and failed training branches',flush=True)
    teachers=[];labels=[];pool=[];teacher_evidence=[]
    for k,(identifier,instance) in enumerate(data['train']):
        rollouts=[]
        for attempt in range(args.teacher_rollouts):
            collector=TransitionCollector(None)
            row,_,_=recover_construct(instance,mode='priority' if attempt==0 else 'uniform',seed=args.data_seed+k*100+attempt,config=config,
                budget=args.teacher_budget,max_decisions=args.eval_decisions,record_decisions=True,record_observation=True,decision_callback=collector)
            records=collector.finish(row);examples=examples_from(records);pool.extend(examples)
            teacher_evidence.append(dict(id=identifier,attempt=attempt,examples=examples,proposals=len(records),unknown=sum(r['completion_target'] is None for r in records)))
            rollouts.append(row)
        successful=[r for r in rollouts if r['success']]
        if successful:labels.extend(max(successful,key=lambda r:r['objective'])['decisions'])
        teachers.append(dict(id=identifier,rollouts=rollouts))
    write(output/'teachers.json',teachers);write(output/'teacher_evidence.json',teacher_evidence)
    if not labels or not any(e['label']==1 for e in pool):raise ValueError('Missing successful teachers')
    models={};training=[];updates=[]
    for seed in args.policy_seeds:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        warm=ArchitecturePolicy('mlp',args.width);warm_optimizer=torch.optim.Adam(warm.parameters(),lr=.001)
        for epoch in range(args.teacher_epochs):
            for start in range(0,len(labels),8):supervise(warm,warm_optimizer,labels[start:start+8])
        models[('imitation',seed)]=copy.deepcopy(warm).eval()
        replay=copy.deepcopy(warm);reinforce=copy.deepcopy(warm)
        replay_optimizer=torch.optim.Adam(replay.parameters(),lr=.001);reinforce_optimizer=torch.optim.Adam(reinforce.parameters(),lr=.001)
        ppo=CompletionPolicy(args.width,args.guidance);ppo.actor.load_state_dict(warm.state_dict())
        critic_optimizer=torch.optim.Adam(ppo.completion.parameters(),lr=.001)
        for step in range(args.critic_pretrain):
            loss=completion_loss(ppo,balanced(pool,rng));critic_optimizer.zero_grad();loss.backward();critic_optimizer.step()
        control=copy.deepcopy(ppo);anchor=copy.deepcopy(warm).eval()
        for parameter in anchor.parameters():parameter.requires_grad_(False)
        ppo_optimizer=torch.optim.Adam(ppo.parameters(),lr=.0003);control_optimizer=torch.optim.Adam(control.parameters(),lr=.0003)
        for episode in range(args.episodes):
            identifier,instance=data['train'][int(rng.integers(len(data['train'])))];group=[];records=[]
            log_groups=[];entropy_groups=[];old_group=[]
            for trajectory in range(4):
                rollout_seed=seed*100000+episode*4+trajectory
                collector=TransitionCollector(ppo)
                row,_,_=recover_construct(instance,ppo,seed=rollout_seed,config=config,budget=args.train_budget,max_decisions=args.train_decisions,decision_callback=collector)
                observed=collector.finish(row)
                for record in observed:record['trajectory']=trajectory
                row['completion_evidence']=examples_from(observed);row['unknown_completion_labels']=sum(r['completion_target'] is None for r in observed)
                # Compact scalar trace permits value/ratio/evidence audit without archiving every raw observation.
                row['learning_trace']=[dict(event_index=r['event_index'],index=r['index'],old_log_prob=r['old_log_prob'],old_value=r['old_value'],completion_target=r['completion_target']) for r in observed]
                group.append(row);records.extend(observed)
                old,logs,entropies=recover_construct(instance,reinforce,seed=rollout_seed,config=config,budget=args.train_budget,max_decisions=args.train_decisions,training=True)
                old_group.append(old);log_groups.append(logs);entropy_groups.append(entropies)
            training.extend([dict(method='critic-ppo',seed=seed,episode=episode,id=identifier,rollouts=group),dict(method='reinforce',seed=seed,episode=episode,id=identifier,rollouts=old_group)])
            rewards=np.array([r['reward'] for r in old_group]);losses=[]
            for k,logs in enumerate(log_groups):
                if logs:losses.append(-(rewards[k]-(rewards.sum()-rewards[k])/3)*torch.stack(logs).sum()-.001*torch.stack(entropy_groups[k]).mean())
            if losses:
                reinforce_optimizer.zero_grad();torch.stack(losses).mean().backward();torch.nn.utils.clip_grad_norm_(reinforce.parameters(),1.);reinforce_optimizer.step()
            examples=examples_from(records)
            for epoch in range(args.ppo_epochs):
                batch=[labels[int(i)] for i in rng.integers(len(labels),size=8)]
                supervised=balanced(examples+pool,rng)
                ppo_update=conservative_update(ppo,anchor,ppo_optimizer,records,batch,supervised,kl_limit=args.kl_limit)
                control_update=conservative_update(control,anchor,control_optimizer,records,batch,supervised,policy_gradient=False,kl_limit=args.kl_limit)
                supervise(replay,replay_optimizer,batch)
                updates.append(dict(seed=seed,episode=episode,epoch=epoch,ppo=ppo_update,control=control_update))
            if (episode+1)%8==0:print(f'seed {seed}: {episode+1}/{args.episodes} updates',flush=True)
        for method,model in [('replay',replay),('reinforce',reinforce),('critic-replay',control),('critic-ppo',ppo)]:models[(method,seed)]=model.eval()
    write(output/'training.json',training);write(output/'updates.json',updates)
    checkpoint_hashes={}
    for (method,seed),model in models.items():
        path=output/f'{method}-{seed}.pt';torch.save(model.state_dict(),path);checkpoint_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    write(output/'checkpoints.json',checkpoint_hashes)
    print('Final checkpoints frozen; evaluating untouched instances',flush=True)
    rows=[];solver_rows=[];references=[]
    for split in ('test','ood'):
        for identifier,instance in data[split]:
            order=np.random.default_rng(args.data_seed+len(rows))
            for seed in args.policy_seeds:
                for method in order.permutation(METHODS).tolist():
                    row,_,_=recover_construct(instance,models.get((method,seed)),mode='uniform' if method=='completion-uniform' else 'sample',seed=seed,config=config,budget=args.eval_budget,max_decisions=args.eval_decisions)
                    valid,value,_=oracle_check(onts_instance_to_json_data(instance),row['schedule'],config)
                    if valid!=row['feasible'] or (valid and abs(value-row['objective'])>1e-4):raise ValueError('Oracle disagreement')
                    rows.append(dict(id=identifier,split=split,seed=seed,method=method,utility=row['reward'] if row['success'] else 0.,**row))
            solver_rows.append(dict(id=identifier,split=split,method='milp',**milp_baseline(instance,config,args.eval_budget)))
            result=ONTSMilp(instance,config).solve(time_limit=args.reference_budget,solver_threads=1)
            references.append(dict(id=identifier,status=result.status,objective=result.objective,upper_bound=result.upper_bound,runtime=result.runtime,schedule=result.schedule.int().tolist() if result.schedule is not None else None))
            print(f'evaluated {identifier}; reference {result.status}',flush=True)
    write(output/'evaluations.json',rows);write(output/'solver_evaluations.json',solver_rows);write(output/'references.json',references)
    paired=contrasts(rows);write(output/'contrasts.json',paired)
    lines=['# Completion-critic / conservative PPO pilot','','Same actor, raw observations and recovery backend; fresh instances. Feasibility includes failures and overruns. The primary comparison is critic-PPO versus critic-replay, not PPO versus an untrained network.','','| Split | Method | On-time feasible | Optimum hits | Mean utility | Runtime |','|---|---|---:|---:|---:|---:|']
    ref={r['id']:r for r in references}
    for split in ('all','test','ood'):
        for method in (*METHODS,'milp'):
            selected=[r for r in rows+solver_rows if r['method']==method and (split=='all' or r['split']==split)]
            optimal=sum(r['success'] and ref[r['id']]['status']=='optimal' and abs(r['objective']-ref[r['id']]['objective'])<1e-4 for r in selected)
            lines.append(f"| {split} | {method} | {sum(r['success'] for r in selected)}/{len(selected)} | {optimal}/{len(selected)} | {np.mean([r['utility'] for r in selected]):.6f} | {np.mean([r['runtime'] for r in selected]):.3f} |")
    lines+=['','Paired exploratory marginal utility intervals (seeds averaged within instance):','']
    for c in paired:lines.append(f"- {c['split']}: {c['first']} minus {c['second']}: {c['difference']:.6f} [{c['interval'][0]:.6f}, {c['interval'][1]:.6f}].")
    lines+=['',f"Actor updates accepted: PPO {sum(u['ppo']['updated'] for u in updates)}/{len(updates)}, critic-replay {sum(u['control']['updated'] for u in updates)}/{len(updates)}. Evaluation overruns: {sum(r['overrun']>0 for r in rows)}.",'','Completion labels describe witnessed success or necessary/restricted-menu failure. Censored branches are unknown. Balanced classifier training does not calibrate true feasibility probabilities. PPO/replay share classifier evidence, anchor constraints and attempted optimizer steps; value training is PPO-only. REINFORCE performs one optimizer step per group, PPO/replay perform the declared PPO epochs. Total training time/data volume is not matched. Nine held-out instances, synthetic family, short training and marginal intervals do not establish SOTA.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('output','history-root','backend-snapshot'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--data-seed',type=int,default=2026101600)
    parser.add_argument('--policy-seeds',type=int,nargs='+',default=[11,29,47])
    for key,default in [('episodes',32),('train-per-regime',3),('test-per-regime',2),('ood-per-regime',1),('teacher-rollouts',3),('teacher-epochs',10),('critic-pretrain',50),('ppo-epochs',2),('train-decisions',128),('eval-decisions',4096),('width',32)]:parser.add_argument('--'+key,type=int,default=default)
    for key,default in [('teacher-budget',.7),('train-budget',.35),('eval-budget',2.),('reference-budget',5.),('kl-limit',.02),('guidance',.5)]:parser.add_argument('--'+key,type=float,default=default)
    args=parser.parse_args()
    if args.width%4 or len(set(args.policy_seeds))!=len(args.policy_seeds):parser.error('Invalid width or repeated seeds')
    for key,value in vars(args).items():
        if isinstance(value,(int,float)) and key!='data_seed' and (not np.isfinite(value) or value<=0):parser.error(f'{key} must be positive and finite')
    run(args)


if __name__=='__main__':main()
