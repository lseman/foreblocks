"""Descriptive completion-critic tables and plots from frozen saved outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def render(output,paper):
    output=Path(output);paper=Path(paper)
    def read(name):return json.loads((output/name).read_text())
    rows=read('evaluations.json')+read('solver_evaluations.json');refs={r['id']:r for r in read('references.json')}
    names={'milp':'MILP','completion-uniform':'Uniform recovery','imitation':'Initial imitation','replay':'Teacher replay','reinforce':'REINFORCE','critic-replay':'Completion replay','critic-ppo':'Completion PPO'}
    methods=list(names);summary=[]
    for split in ('all','test','ood'):
        for method in methods:
            selected=[r for r in rows if r['method']==method and (split=='all' or r['split']==split)]
            gaps=[max(0.,refs[r['id']]['objective']-r['objective'])/max(1e-9,abs(refs[r['id']]['objective'])) for r in selected if r['success'] and refs[r['id']]['status']=='optimal']
            summary.append(dict(split=split,method=method,attempts=len(selected),successes=sum(r['success'] for r in selected),
                optimal=sum(r['success'] and refs[r['id']]['status']=='optimal' and abs(r['objective']-refs[r['id']]['objective'])<1e-4 for r in selected),
                utility=float(np.mean([r['utility'] for r in selected])),runtime=float(np.mean([r['runtime'] for r in selected])),
                conditional_gap=float(np.mean(gaps)) if gaps else None,overruns=sum(r['overrun']>0 for r in selected)))
    analysis=output/'analysis';analysis.mkdir(exist_ok=True);(analysis/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=[r'\begin{table}[htbp]',r'\centering',r'\small',r'\begin{tabular}{lrrrr}',r'\toprule',r'Method & Same-size feasible & Larger feasible & All optimal & Mean utility \\',r'\midrule']
    for method in methods:
        test,ood,all_result=[next(s for s in summary if s['split']==split and s['method']==method) for split in ('test','ood','all')]
        lines.append(f"{names[method]} & {test['successes']}/{test['attempts']} & {ood['successes']}/{ood['attempts']} & {all_result['optimal']}/{all_result['attempts']} & {all_result['utility']:.4f} "+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Fresh completion-critic pilot, two-second inference allowance. All failures and overruns remain in denominators. Learned arms use three seeds per instance; deterministic MILP runs once per instance. Completion replay and PPO share classifier evidence and actor protection.}',r'\label{tab:completion-critic-summary}',r'\end{table}']
    table='\n'.join(lines)+'\n';(analysis/'completion_critic_summary.tex').write_text(table);(paper/'tables'/'completion_critic_summary.tex').write_text(table)
    training=read('training.json');diagnostics=[]
    for method in ('critic-ppo','reinforce'):
        groups=[e for e in training if e['method']==method];trajectories=[r for e in groups for r in e['rollouts']]
        result=dict(method=method,trajectories=len(trajectories),successes=sum(r['success'] for r in trajectories),overruns=sum(r['overrun']>0 for r in trajectories),
            cap_hits=sum(r['sampled_decisions']>=read('protocol.json')['arguments']['train_decisions'] for r in trajectories))
        if method=='critic-ppo':
            for section,condition in [('first',lambda e:e['episode']<8),('last',lambda e:e['episode']>=24)]:
                values=[(r['reward']-record['old_value'])**2 for e in groups if condition(e) for r in e['rollouts'] for record in r['learning_trace']]
                result[section+'_value_mse']=float(np.mean(values)) if values else None
        diagnostics.append(result)
    (analysis/'learning_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for split,color,offset in [('test','#2b6f9c',-.18),('ood','#d47a25',.18)]:
        selected=[next(s for s in summary if s['method']==m and s['split']==split) for m in methods]
        axes[0].barh(np.arange(len(methods))+offset,[s['successes']/s['attempts'] for s in selected],height=.32,color=color,label='Same-size' if split=='test' else 'Larger')
    axes[0].set(yticks=np.arange(len(methods)),yticklabels=[names[m] for m in methods],xlim=(0,1),xlabel='On-time feasibility rate')
    axes[0].invert_yaxis();axes[0].set_title('Completion-guided policies',pad=28);axes[0].legend(frameon=False,loc='lower right',bbox_to_anchor=(1.,1.),ncol=2);axes[0].grid(axis='x',alpha=.2);axes[0].set_axisbelow(True)
    contrast=next(c for c in read('contrasts.json') if c['first']=='critic-ppo' and c['second']=='critic-replay' and c['split']=='all')
    axes[1].axvline(0,color='.4',linestyle='--',linewidth=1)
    selected=[c for c in read('contrasts.json') if c['first']=='critic-ppo' and c['split']=='all']
    for k,c in enumerate(selected):
        axes[1].plot(c['interval'],[k,k],color='#2b6f9c',linewidth=2);axes[1].plot(c['difference'],k,'o',color='#2b6f9c')
    axes[1].set(yticks=np.arange(len(selected)),yticklabels=[names[c['second']] for c in selected],xlabel='Completion PPO minus comparator: mean utility',title='Exploratory 95% paired intervals')
    axes[1].invert_yaxis();axes[1].grid(axis='x',alpha=.2)
    for suffix in ('pdf','png'):fig.savefig(analysis/f'completion_critic.{suffix}',dpi=200)
    plt.close(fig)
    (analysis/'render_completion_results.py').write_bytes(Path(__file__).read_bytes())
    hashes={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ('protocol.json','evaluations.json','solver_evaluations.json','references.json','contrasts.json','training.json','updates.json')}
    hashes['renderer']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(analysis/'input_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--paper',required=True)
    args=parser.parse_args();render(args.output,args.paper)
