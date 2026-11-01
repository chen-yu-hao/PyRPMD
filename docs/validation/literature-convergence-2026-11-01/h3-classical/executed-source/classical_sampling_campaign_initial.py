"""Run fixed, resumable independent P1 ladders and retain every stage.

The maximum sample budget is fixed at creation. Intermediate checkpoint
stages retain exact prefix hashes, RNG and geometry, while the current
checkpoint retains all CV observations without duplicating large prefixes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.h3.mixing_sampling import run

SEEDS=(2026110101,2026110102,2026110103,2026110104)
STAGES=(20000,65536,262144,1048576)
WARMUP=2000


def array_hash(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def one_seed(job):
    base,library,schedule_file,seed,target=job
    base,library,schedule_file=map(Path,(base,library,schedule_file))
    schedule=json.loads(schedule_file.read_text())
    output=base/f'seed-{seed}'
    checkpoint=output/'checkpoint.npz'
    completed=0
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as data:
            completed=int(data['completed'])
    remaining=WARMUP+target-completed
    if remaining<0:
        raise ValueError('requested stage predates current checkpoint; use retained prefix evidence')
    if remaining:
        report=run(output,library,beads=1,seed=seed,warmup=WARMUP,production=STAGES[-1],
                   centers=schedule['centers'],stiffness=schedule['stiffness'],
                   checkpoint_interval=10000,diagnostic_interval=100,
                   stop_after=remaining,resume=checkpoint.exists())
    else:
        report=json.loads((output/'run-report.json').read_text())
    stage=output/'stages'/str(target)
    stage.mkdir(parents=True,exist_ok=True)
    with np.load(checkpoint,allow_pickle=False) as data:
        samples=data['samples']
        if samples.shape!=(target,len(schedule['centers'])):
            raise ValueError('unexpected checkpoint sample count')
        # Every earlier failed/partial stage remains exactly recoverable as
        # an ordered prefix of the full current CV array.
        for old in (output/'stages').glob('*/stage.json'):
            recorded=json.loads(old.read_text())
            n=recorded['production_sweeps']
            if array_hash(samples[:n])!=recorded['samples_prefix_sha256']:
                raise ValueError('continuation changed previously retained CV observations')
        state={name:data[name] for name in ['R','r','completed','rng','accepted','proposed','swaps','lineage']}
        np.savez_compressed(stage/'state.npz',**state)
        document={'schema':'pyrpmd.h3.classical-campaign-stage.v1','seed':seed,
                  'warmup_sweeps':WARMUP,'production_sweeps':target,
                  'maximum_fixed_production_budget':STAGES[-1],
                  'schedule_sha256':hashlib.sha256(schedule_file.read_bytes()).hexdigest(),
                  'samples_prefix_sha256':array_hash(samples),'samples_shape':list(samples.shape),
                  'samples_storage':'Ordered prefix of seed checkpoint.npz:samples; every prefix is checked before extending.',
                  'all_previous_prefixes_verified':True,
                  'checkpoint_sha256_at_stage':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  'state_snapshot_sha256':hashlib.sha256((stage/'state.npz').read_bytes()).hexdigest(),
                  'run_report':report,'physical_converged':False,
                  'campaign_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (stage/'stage.json').write_text(json.dumps(document,indent=2)+'\n')
    print(json.dumps({'event':'stage_completed','seed':seed,'production':target,
                      'checkpoint':str(checkpoint),'samples_sha256':document['samples_prefix_sha256']}),flush=True)
    return document


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--schedule',type=Path,required=True)
    p.add_argument('--production-stage',type=int,choices=STAGES,required=True)
    p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    a=p.parse_args()
    if not set(a.seeds)<=set(SEEDS):
        raise ValueError('campaign permits only the predeclared independent seeds')
    jobs=[(a.output,a.library,a.schedule,seed,a.production_stage) for seed in a.seeds]
    if a.workers==1:
        results=[one_seed(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            results=list(pool.map(one_seed,jobs))
    summary={'production_stage':a.production_stage,'seeds':list(a.seeds),
             'stage_files':[str(a.output/f'seed-{seed}'/'stages'/str(a.production_stage)/'stage.json') for seed in a.seeds],
             'all_stages_completed':len(results)==len(jobs),'physical_converged':False}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/f'campaign-stage-{a.production_stage}.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
