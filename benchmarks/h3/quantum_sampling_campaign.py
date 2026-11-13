"""Execute only the registered independent quantum umbrella ladders.

Stages preserve every earlier ordered CV prefix and its exact RNG/geometry
state. Refinement extends all requested seeds; it does not replace a seed
or select a favorable time interval.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.h3.mixing_sampling import run


def array_hash(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def one_seed(job):
    base,library,protocol_file,seed,stage=job
    base=Path(base);protocol_file=Path(protocol_file)
    protocol=json.loads(protocol_file.read_text())
    if stage not in protocol['fixed_refinement_prefixes'] or seed not in protocol['independent_ladder_seeds']:
        raise ValueError('seed/stage was not registered')
    if file_hash(ROOT/protocol['sampler_file'])!=protocol['sampler_sha256']:
        raise ValueError('sampler changed since protocol registration')
    schedule_path=ROOT/protocol['schedule_file']
    if file_hash(schedule_path)!=protocol['schedule_sha256']:
        raise ValueError('umbrella schedule changed since registration')
    schedule=json.loads(schedule_path.read_text())
    output=base/f'seed-{seed}'
    output.mkdir(parents=True,exist_ok=True)
    checkpoint=output/'checkpoint.npz'
    completed=0
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as old:
            completed=int(old['completed'])
            for path in (output/'stages').glob('*/stage.json'):
                receipt=json.loads(path.read_text());count=receipt['production_sweeps']
                if array_hash(old['samples'][:count])!=receipt['samples_prefix_sha256']:
                    raise ValueError('a previously retained sample prefix changed')
    remaining=protocol['warmup_sweeps']+stage-completed
    if remaining<0: raise ValueError('current checkpoint exceeds requested stage')
    started=time.perf_counter()
    if remaining:
        with (base/f'seed-{seed}-sampling.log').open('a') as log,redirect_stdout(log):
            run(output,library,beads=protocol['beads_primary'],seed=seed,
                warmup=protocol['warmup_sweeps'],production=protocol['fixed_refinement_prefixes'][-1],
                centers=schedule['centers'],stiffness=schedule['stiffness'],
                checkpoint_interval=1000,diagnostic_interval=10,
                resume=checkpoint.exists(),stop_after=remaining)
    destination=output/'stages'/str(stage);destination.mkdir(parents=True,exist_ok=True)
    with np.load(checkpoint,allow_pickle=False) as saved:
        if saved['samples'].shape!=(stage,len(schedule['centers'])):
            raise ValueError('checkpoint has unexpected production count')
        state={key:saved[key] for key in ('R','r','completed','rng','accepted','proposed','swaps','lineage')}
        np.savez_compressed(destination/'state.npz',**state)
        receipt=dict(schema='pyrpmd.h3.quantum-campaign-stage.v1',seed=seed,
            production_sweeps=stage,warmup_sweeps=protocol['warmup_sweeps'],
            samples_prefix_sha256=array_hash(saved['samples']),
            diagnostics_prefix_sha256=array_hash(saved['diagnostics']),
            samples_storage='all observations retained as ordered prefix of checkpoint.npz',
            previous_prefixes_verified=True,protocol_sha256=file_hash(protocol_file),
            checkpoint_sha256=file_hash(checkpoint),state_sha256=file_hash(destination/'state.npz'),
            driver_sha256=file_hash(__file__),elapsed_seconds_this_call=time.perf_counter()-started,
            literature_reproduced=False)
    (destination/'stage.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--production-stage',type=int,required=True)
    p.add_argument('--workers',type=int,choices=(1,2),default=2)
    p.add_argument('--seeds',type=int,nargs='+')
    a=p.parse_args();protocol=json.loads(a.protocol.read_text())
    seeds=a.seeds or protocol['independent_ladder_seeds']
    if len(set(seeds))!=len(seeds) or not set(seeds)<=set(protocol['independent_ladder_seeds']):
        raise ValueError('only distinct registered seeds are permitted')
    print(json.dumps({'event':'start','seeds':seeds,'stage':a.production_stage,
        'workers':a.workers,'protocol_sha256':file_hash(a.protocol)}),flush=True)
    jobs=[(a.output,a.library,a.protocol,seed,a.production_stage) for seed in seeds]
    receipts=[]
    if a.workers==1:
        for job in jobs:
            receipt=one_seed(job);receipts.append(receipt)
            print(json.dumps({'event':'stage_complete','seed':receipt['seed'],
                'production':receipt['production_sweeps']}),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(one_seed,job) for job in jobs]
            for future in as_completed(futures):
                receipt=future.result();receipts.append(receipt)
                print(json.dumps({'event':'stage_complete','seed':receipt['seed'],
                    'production':receipt['production_sweeps']}),flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/f'stage-{a.production_stage}.json').write_text(json.dumps({
        'protocol_sha256':file_hash(a.protocol),'requested_seeds':seeds,
        'completed_seeds':sorted(x['seed'] for x in receipts),'production':a.production_stage,
        },indent=2)+'\n')


if __name__=='__main__':main()
