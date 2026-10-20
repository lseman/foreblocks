"""Training-only diagnostic against verified frozen critic-study sources."""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();study=Path(args.study).resolve();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((study/'protocol.json').read_text());hashes=json.loads((study/'checkpoints.json').read_text())
    for filename,digest in protocol['source_hashes'].items():
        if hashlib.sha256((study/'source'/filename).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source mismatch')
    methods=['imitation','replay','critic-ppo','critic-ppo-no-guidance','critic-ppo-cap128','critic-ppo-cap512']
    seed=protocol['arguments']['policy_seeds'][0]
    names=[f'{m}-{seed}.pt' for m in ['imitation','replay','critic-ppo']]
    for filename in names:
        if hashlib.sha256((study/filename).read_bytes()).hexdigest()!=hashes[filename]:raise ValueError('Checkpoint mismatch')
    diagnostic=dict(scope='Training instances only; one existing policy seed; descriptive diagnosis, no held-out tuning or superiority test. Shared .7-second allowance; default cap4096 except explicitly cap128/cap512.',seed=seed,methods=methods,budget=.7,input_checkpoint_hashes={n:hashes[n] for n in names},source_study=str(study),source_protocol_hash=hashlib.sha256((study/'protocol.json').read_bytes()).hexdigest())
    (output/'protocol.json').write_text(json.dumps(diagnostic,indent=2)+'\n');(output/'diagnose_completion_critic.py').write_bytes(Path(__file__).read_bytes())
    with TemporaryDirectory(prefix='onts-critic-diagnostic-') as directory:
        base=Path(directory);(base/'projects').mkdir();(base/'projects'/'__init__.py').write_text('');(base/'projects'/'scheduling').symlink_to(study/'source',target_is_directory=True);sys.path.insert(0,directory)
        import torch
        from projects.scheduling.rl_architectures import ArchitecturePolicy
        from projects.scheduling.rl_completion import CompletionPolicy
        from projects.scheduling.rl_recovery import recover_construct
        from projects.scheduling.milp import ONTSConfig
        from projects.scheduling.environments.onts_env import load_onts_instance,onts_instance_to_json_data
        from projects.scheduling.research.oracle import oracle_check
        from projects.scheduling.research.datasets import instance_hash
        torch.set_num_threads(1);config=ONTSConfig(**protocol['config']);models={}
        for method in ['imitation','replay','critic-ppo']:
            model=CompletionPolicy(protocol['arguments']['width'],protocol['arguments']['guidance']) if method=='critic-ppo' else ArchitecturePolicy('mlp',protocol['arguments']['width'])
            model.load_state_dict(torch.load(study/f'{method}-{seed}.pt',weights_only=True,map_location='cpu'));models[method]=model.eval()
        rows=[]
        for entry in protocol['manifest']:
            if entry['split']!='train':continue
            instance=load_onts_instance(study/f"{entry['id']}.json")
            if instance_hash(instance)!=entry['hash']:raise ValueError('Training input mismatch')
            for method in methods:
                model=models['critic-ppo'] if method.startswith('critic-ppo') else models[method]
                if isinstance(model,CompletionPolicy):model.guidance=0. if method.endswith('no-guidance') else protocol['arguments']['guidance']
                cap=128 if method.endswith('cap128') else 512 if method.endswith('cap512') else 4096
                row,_,_=recover_construct(instance,model,seed=seed,config=config,budget=.7,max_decisions=cap)
                valid,value,_=oracle_check(onts_instance_to_json_data(instance),row['schedule'],config)
                if valid!=row['feasible'] or (valid and abs(value-row['objective'])>1e-4):raise ValueError('Independent diagnostic disagreement')
                rows.append(dict(id=entry['id'],method=method,cap=cap,**row))
            print(f"diagnosed {entry['id']}",flush=True)
        (output/'evaluations.json').write_text(json.dumps(rows,indent=2)+'\n')
        summary=[dict(method=m,successes=sum(r['success'] for r in rows if r['method']==m),attempts=sum(r['method']==m for r in rows),mean_decisions=sum(r['sampled_decisions'] for r in rows if r['method']==m)/9) for m in methods]
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
