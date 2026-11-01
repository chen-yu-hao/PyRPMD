"""Recover complete H3 checkpoints without repeating trajectories.

The unchanged mbar_rate_pmf implementation combines identical umbrella
Hamiltonians before allocating the bias matrix. All CV samples are retained.
It certifies solver convergence and overlap; the unchanged UI/parent/child
stages then run with 100 final UI bootstrap draws and zero redundant MBAR
bootstrap draws. This script is part of the numerical-data archive.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import h5py
import numpy as np
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.h3.sampling_control import source_snapshot
from pyrpmd.analysis.fes import mbar_rate_pmf, umbrella_integration
from pyrpmd.workflows.fes import write_json
from pyrpmd.parallel import run_tasks
from pyrpmd.system import System
from pyrpmd.workflows import fes,rate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    a=parser.parse_args();started=time.perf_counter()
    original=json.loads((a.run/'settings.json').read_text());config=original['config'];system=System.from_dict(original['system'])
    config['parallel']['trajectory_workers']=a.workers
    out=a.run/'run';out.mkdir(exist_ok=True)
    before=source_snapshot();config=rate._prepare(config,system,out)
    cv,specs,window_centers,k=fes.resolve_umbrellas(config,system)
    streams=[];hashes=[]
    for i,center in enumerate(window_centers):
        for r in range(config['sampling']['replicas']):
            path=out/'umbrella/checkpoints'/f'window_{i:05d}_replica_{r:04d}.npz'
            expected=fes._fingerprint(config,system,dict(window=i,replica=r,center=center,stiffness=k))
            with np.load(path,allow_pickle=False) as saved:
                if saved['fingerprint'].item()!=expected or saved['elapsed'].item()!=config['sampling']['equilibration_steps']+config['sampling']['production_steps']:
                    raise ValueError('incomplete or mismatched checkpoint '+str(path))
                streams.append(dict(window=i,replica=r,center=center,stiffness=k,
                                    samples=saved['values'].copy(),energies=saved['energies'].copy(),
                                    last_q=saved['q'].copy(),last_p=saved['p'].copy(),checkpoint=str(path)))
            hashes.append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    samples=[s['samples'][:,0] for s in streams]
    centers=np.array([s['center'][0] for s in streams]);stiffness=np.array([s['stiffness'][0,0] for s in streams])
    grid=np.linspace(0,centers.max(),401)
    curve,_=umbrella_integration(samples,centers,stiffness,config['beta'],grid)
    xi=float(grid[curve.argmax()]); options=rate.rate_pmf_options(config)
    print(json.dumps({'event':'mbar_start','samples':sum(len(x) for x in samples),'streams':len(streams),
                      'unique_states':len(window_centers),'xi_star':xi}),flush=True)
    with threadpool_limits(1):
        fit=mbar_rate_pmf(samples,centers,stiffness,config['beta'],reference=0.,xi_star=xi,
                          bins=options['bins'],ranges=options['ranges'],bootstrap=0,
                          min_count=options['min_count'],overlap_threshold=options['overlap_threshold'],
                          measure='rate_flux',compare_ui=True)
    write_json(a.run/'recovery-mbar.json',{key:value for key,value in fit.items() if key!='weights'})
    print(json.dumps({'event':'mbar_finished','status':fit['status'],'barrier':fit['free_energy_barrier_hartree'],
                      'elapsed_seconds':time.perf_counter()-started}),flush=True)
    # A transparent checkpoint-derived sample container: no trajectory is
    # relabelled as newly propagated and no unverified overlap is manufactured.
    with h5py.File(out/'umbrella/fes.h5','w') as saved:
        saved.attrs['schema']='pyrpmd.h3.audit-recovered-cv.v1'
        saved.attrs['measure']='rate_flux'
        saved.attrs['reconstruction']='mbar_rate_pmf; duplicate Hamiltonians combined; all original samples retained'
        saved.attrs['source_checkpoint_manifest']=json.dumps(hashes)
        saved.create_dataset('edges_0',data=fit['edges'])
        saved.create_dataset('mask',data=fit['mask'])
        saved.create_dataset('effective_samples_per_stream',data=fit['effective_samples_per_stream'])
        for i,s in enumerate(streams):
            g=saved.create_group('streams/'+str(i))
            for key in ('samples','energies','center','stiffness'):
                g.create_dataset(key,data=s[key])
            g.attrs.update(window=s['window'],replica=s['replica'])
    data={'connected':bool(fit['solver']['converged'] and fit['overlap_connected']),
          'statistical_inefficiency':fit['statistical_inefficiency'],
          'effective_samples_per_stream':fit['effective_samples_per_stream'],
          'weight_ess':fit['weight_ess'],'edges':[fit['edges']],'mask':fit['mask'],'raw_count':fit['raw_count']}
    config['analysis']['bootstrap']=100
    recrossing=rate._recrossing_options(config)
    pmf,parents=rate._pmf_and_parent_tasks({'data':data,'streams':streams,'summary':{'warnings':[]}},config,system,out,recrossing)
    pmf['intermediate_fes_bootstrap']=0
    pmf['intermediate_fes_purpose']='Checkpoint recovery: exact full-sample unique-state MBAR overlap; no intermediate bootstrap'
    write_json(out/'rate_pmf.json',pmf)
    def dispatch(function,tasks): return run_tasks(function,tasks,config['parallel'])
    if parents:
        sampled=dispatch(rate._parent_task,parents)
        children_tasks=rate._child_tasks(sampled,config,system,out,recrossing,pmf)
        children=rate._dispatch_children(children_tasks,config,dispatch)
    else:
        children=[]
    result=rate._aggregate(children,pmf,config,out,recrossing)
    write_json(out/'summary.json',result)
    (out/'input').mkdir(exist_ok=True);write_json(out/'input/resolved.json',config)
    after=source_snapshot()
    write_json(a.run/'recovery-report.json',{'schema':'pyrpmd.h3.checkpoint-recovery.v1',
        'elapsed_seconds':time.perf_counter()-started,'new_umbrella_steps':0,'all_original_cv_samples_used':True,
        'source_at_start':before,'source_at_end':after,'sources_unchanged':before==after,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_checkpoints':hashes,'intermediate_mbar_bootstrap':0,'final_ui_bootstrap':100,
        'physical_converged':False,'summary':{key:value for key,value in result.items() if key!='pmf'}})
    print(json.dumps({'event':'finished','elapsed_seconds':time.perf_counter()-started,
                      'qtst':pmf['k_qtst_cm3_s'],'kappa':result.get('kappa'),
                      'rate':result.get('rate_estimate_cm3_s')}),flush=True)

if __name__=='__main__':main()
