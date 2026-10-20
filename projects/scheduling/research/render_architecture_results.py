"""Render descriptive architecture results; does not train/select checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def render(output, paper):
    output = Path(output); paper = Path(paper)
    def read(name):
        return json.loads((output/name).read_text())
    neural = read('evaluations.json'); solver = read('solver_evaluations.json')
    references = {r['id']:r for r in read('references.json')}
    protocol = read('protocol.json'); contrasts = read('contrasts.json')
    architectures = protocol['architectures']; all_rows = neural+solver
    methods = ['milp','completion-uniform']+[f'{a}-{arm}' for a in architectures for arm in ('imitation','replay','rl')]
    summary = []
    for split in ('all','test','ood'):
        for method in methods:
            rows = [r for r in all_rows if r['method']==method and (split=='all' or r['split']==split)]
            optimal = [r for r in rows if r['success'] and references[r['id']]['status']=='optimal' and abs(r['objective']-references[r['id']]['objective'])<1e-4]
            gaps = [max(0.,references[r['id']]['objective']-r['objective'])/max(1e-9,abs(references[r['id']]['objective'])) for r in rows if r['success'] and references[r['id']]['status']=='optimal']
            summary.append(dict(split=split,method=method,attempts=len(rows),successes=sum(r['success'] for r in rows),optimal=len(optimal),utility=float(np.mean([r['utility'] for r in rows])),
                runtime=float(np.mean([r['runtime'] for r in rows])),conditional_gap=float(np.mean(gaps)) if gaps else None,
                overruns=sum(r['overrun']>0 for r in rows),mean_decisions=float(np.mean([r.get('sampled_decisions',0) for r in rows]))))
    analysis = output/'analysis'; analysis.mkdir(exist_ok=True)
    (analysis/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    training=read('training.json'); diagnostics=[]
    for architecture in architectures:
        groups=[r for r in training if r['architecture']==architecture]
        trajectories=[t for r in groups for t in r['rollouts']]
        diagnostics.append(dict(architecture=architecture,groups=len(groups),trajectories=len(trajectories),
            successes=sum(t['success'] for t in trajectories),nonzero_reward_dispersion=sum(r['reward_std']>1e-9 for r in groups),
            mean_decisions=float(np.mean([t['sampled_decisions'] for t in trajectories])),overruns=sum(t['overrun']>0 for t in trajectories),
            cap_hits=sum(t['sampled_decisions']>=protocol['arguments']['train_decisions'] for t in trajectories)))
    (analysis/'training_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    names = {'milp':'MILP','completion-uniform':'Uniform recovery','mlp':'Pooled MLP','pointer':'Recurrent pointer','bipartite':'Bipartite GNN','transformer':'Transformer'}
    def label(method):
        if method in names: return names[method]
        a,arm=method.split('-'); return names[a]+' '+('IMIT.' if arm=='imitation' else arm.upper())
    lines = ['\\begin{table}[htbp]','\\centering','\\small','\\begin{tabular}{lrrrr}','\\toprule',
        r'Method & Same-size feasible & Larger feasible & All optimal & Mean utility \\',r'\midrule']
    for method in methods:
        test,ood,all_result = [next(s for s in summary if s['split']==split and s['method']==method) for split in ('test','ood','all')]
        lines.append(f"{label(method)} & {test['successes']}/{test['attempts']} & {ood['successes']}/{ood['attempts']} & {all_result['optimal']}/{all_result['attempts']} & {all_result['utility']:.4f} "+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\caption{Fresh architecture pilot under a common two-second allowance. On-time success and optimum hits include all attempts. MILP runs once per instance; learned and uniform policies run for three seeds. Architecture recipes share raw observations and recovery; parameter counts are not matched.}',r'\label{tab:architecture-summary}',r'\end{table}']
    table='\n'.join(lines)+'\n'; (analysis/'architecture_summary.tex').write_text(table)
    (paper/'tables'/'architecture_summary.tex').write_text(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    display=['milp','completion-uniform']+[f'{a}-rl' for a in architectures]
    for split,color,offset in [('test','#2b6f9c',-.18),('ood','#d47a25',.18)]:
        subset=[next(s for s in summary if s['split']==split and s['method']==m) for m in display]
        axes[0].barh(np.arange(len(display))+offset,[s['successes']/s['attempts'] for s in subset],height=.32,label='Same-size' if split=='test' else 'Larger',color=color)
    axes[0].set(yticks=np.arange(len(display)),yticklabels=[label(m) for m in display],xlim=(0,1),xlabel='On-time feasibility rate',title='RL policies versus baselines')
    axes[0].invert_yaxis();axes[0].set_title('RL policies versus baselines',pad=28);axes[0].legend(frameon=False,loc='lower right',bbox_to_anchor=(1.,1.),ncol=2);axes[0].grid(axis='x',alpha=.2);axes[0].set_axisbelow(True)
    for k,a in enumerate(architectures):
        c=next(c for c in contrasts if c['split']=='all' and c['first']==f'{a}-rl' and c['second']==f'{a}-replay')
        lower,upper=c['interval']; mean=c['difference']
        axes[1].plot([lower,upper],[k,k],color='#2b6f9c',linewidth=2);axes[1].plot(mean,k,'o',color='#2b6f9c')
    axes[1].axvline(0,color='.4',linestyle='--',linewidth=1)
    axes[1].set(yticks=np.arange(len(architectures)),yticklabels=[names[a] for a in architectures],xlabel='RL minus matched replay: mean utility',title='RL versus replay: exploratory 95% intervals')
    axes[1].invert_yaxis();axes[1].grid(axis='x',alpha=.2)
    for suffix in ('pdf','png'):
        fig.savefig(analysis/f'architecture_comparison.{suffix}',dpi=200)
    plt.close(fig)
    shutil.copyfile(analysis/"architecture_comparison.pdf",paper/"figures"/"architecture_comparison.pdf")
    (analysis/'render_architecture_results.py').write_bytes(Path(__file__).read_bytes())
    hashes={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ('protocol.json','evaluations.json','solver_evaluations.json','references.json','contrasts.json','training.json','training_costs.json')}
    hashes['renderer']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (analysis/'input_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--paper',required=True)
    args=parser.parse_args();render(args.output,args.paper)
