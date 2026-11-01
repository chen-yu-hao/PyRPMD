"""Small-memory, full-data UI diagnostic with no new trajectories.

This does not declare overlap or make a recrossing/rate estimate. The saved
PES is used only by the usual preparation step to reconstruct the exact
checkpoint fingerprint. All production coordinates remain in original order.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.h3.sampling_control import source_snapshot
from benchmarks.h3.classical_tst import MASS_AU
from pyrpmd.analysis.fes import umbrella_integration, _bootstrap_block_plan, _block_resample, cv_statistical_inefficiency
from pyrpmd.constants import AU_RATE_TO_CM3_S, AU_TIME_FS
from pyrpmd.io import write_json
from pyrpmd.system import System
from pyrpmd.workflows import rate, fes


def matched_reference(config, system):
    """Select only the audited 300 K, equal-ordinary-H Table 2 target."""
    if (not np.isclose(config.get('temperature_K', float('nan')), 300., rtol=0, atol=1e-8)
            or system.masses.shape != (3,)
            or not np.allclose(system.masses, MASS_AU, rtol=0, atol=1e-8)
            or list(system.atomic_numbers) != [1, 1, 1]
            or np.any(system.pbc)):
        raise ValueError('Table 2 reference requires nonperiodic H + H2 at 300 K with three ordinary-1H masses')
    return 1.56e-18 if config['beads'] == 1 else 2.51e-16


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    started=time.perf_counter(); args.output.mkdir(parents=True,exist_ok=True)
    settings=json.loads((args.run/'settings.json').read_text())
    config, system=settings['config'],System.from_dict(settings['system'])
    paper=matched_reference(config, system)
    before=source_snapshot()
    prepared=rate._prepare(config,system,args.output)
    cv,specs,centers,k=fes.resolve_umbrellas(prepared,system)
    streams=[]; provenance=[]; replicas=[]; windows=[]; all_centers=[]; stiffness=[]
    for i,center in enumerate(centers):
        for r in range(config['sampling']['replicas']):
            path=args.run/'run/umbrella/checkpoints'/f'window_{i:05d}_replica_{r:04d}.npz'
            expected=fes._fingerprint(prepared,system,dict(window=i,replica=r,center=center,stiffness=k))
            with np.load(path,allow_pickle=False) as saved:
                if saved['fingerprint'].item()!=expected:
                    raise ValueError('checkpoint physics mismatch '+str(path))
                if saved['elapsed'].item()!=config['sampling']['equilibration_steps']+config['sampling']['production_steps']:
                    raise ValueError('incomplete checkpoint '+str(path))
                streams.append(saved['values'][:,0].copy())
            provenance.append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            replicas.append(r);windows.append(i);all_centers.append(center[0]);stiffness.append(k[0,0])
    centers=np.asarray(all_centers); stiffness=np.asarray(stiffness);replicas=np.asarray(replicas)
    grid=np.linspace(0,centers.max(),401); beta=config['beta']
    gs=np.array([cv_statistical_inefficiency(x) for x in streams])
    counts=np.array([len(x) for x in streams]);blocks=_bootstrap_block_plan(counts,gs)
    curves={}
    for name,vals in [('all',streams),('first_half',[x[:len(x)//2] for x in streams]),
                      ('second_half',[x[len(x)//2:] for x in streams])]:
        y,_=umbrella_integration(vals,centers,stiffness,beta,grid); curves[name]=y-y[0]
    for r in np.unique(replicas):
        ix=np.flatnonzero(replicas==r)
        y,_=umbrella_integration([streams[i] for i in ix],centers[ix],stiffness[ix],beta,grid)
        curves[f'replica_{r}']=y-y[0]
    xi=float(grid[np.argmax(curves['all'])])
    rng=np.random.default_rng(np.random.SeedSequence([config['seed'],401])); draws=[]
    for _ in range(100):
        resampled=[_block_resample(x,int(block),rng) for x,block in zip(streams,blocks['block_lengths'])]
        y,_=umbrella_integration(resampled,centers,stiffness,beta,grid);draws.append(y-y[0])
    draws=np.array(draws);barriers=np.array([np.interp(xi,grid,y) for y in draws])
    ma,mb=[system.masses[list(f)].sum() for f in prepared['reaction']['fragments']]
    reduced=ma*mb/(ma+mb); rinf=prepared['reaction']['rinf']
    prefactor=4*np.pi*rinf*rinf/np.sqrt(2*np.pi*beta*reduced)
    def rates(y): return float(prefactor*np.exp(-beta*np.interp(xi,grid,y))*AU_RATE_TO_CM3_S)
    rows={name:{'barrier_at_fixed_xi_hartree':float(np.interp(xi,grid,y)),
                'maximum_xi':float(grid[np.argmax(y)]),'maximum_barrier_hartree':float(y.max()),
                'qtst_candidate_cm3_molecule_s':rates(y)} for name,y in curves.items()}
    qtst_draws=prefactor*np.exp(-beta*barriers)*AU_RATE_TO_CM3_S
    mean_drift=np.array([(x[len(x)//2:].mean()-x[:len(x)//2].mean())/x.std() for x in streams])
    arrays={'samples':np.concatenate(streams),'counts':counts,'centers':centers,'stiffness':stiffness,
            'replicas':replicas,'windows':np.array(windows),'effective_samples':counts/gs}
    metadata={'schema':'pyrpmd.h3.checkpoint-cv.v1','input_settings':str(args.run/'settings.json'),
              'input_settings_sha256':hashlib.sha256((args.run/'settings.json').read_bytes()).hexdigest(),
              'input_checkpoints':provenance,'beads':config['beads'],'beta':beta,'temperature_K':config['temperature_K'],
              'sample_interval_fs':config['sampling']['sample_interval']*config['sampling']['dt_au']*AU_TIME_FS,
              'arrays':{name:{'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
                        for name,a in arrays.items()}}
    np.savez_compressed(args.output/'production-cv.npz',metadata_json=np.array(json.dumps(metadata)),**arrays)
    names=list(curves)
    low,high=np.quantile(draws,[.025,.975],axis=0)
    np.savetxt(args.output/'ui-curves.csv',np.column_stack([grid,*[curves[n] for n in names],low,high]),delimiter=',',
               header=','.join(['xi',*[(n+'_hartree') for n in names],'bootstrap_p025_hartree','bootstrap_p975_hartree']),comments='')
    after=source_snapshot()
    report={'schema':'pyrpmd.h3.full-cv-ui-diagnostic.v1','run':str(args.run),'seed':config['seed'],'beads':config['beads'],
            'samples':int(counts.sum()),'streams':len(streams),'xi_star':xi,'temperature_K':config['temperature_K'],
            'prefactor_au':prefactor,'rows':rows,'paper_qtst_or_tst_cm3_molecule_s':paper,
            'ratio_to_paper':rows['all']['qtst_candidate_cm3_molecule_s']/paper,
            'bootstrap_draws':100,'barrier_draws_hartree':barriers.tolist(),
            'diagnostic_barrier_percentile_95_hartree':np.quantile(barriers,[.025,.975]),
            'diagnostic_qtst_percentile_95_cm3_molecule_s':np.quantile(qtst_draws,[.025,.975]),
            'block_plan':blocks,'effective_samples_quantiles':np.quantile(counts/gs,[0,.25,.5,.75,1]),
            'streams_below_20_effective_samples':int(np.count_nonzero(counts/gs<20)),
            'half_mean_drift_in_sigma_quantiles':np.quantile(np.abs(mean_drift),[0,.25,.5,.75,1]),
            'stationary_validated':False,'overlap_verified':False,'transmission_coefficient':None,'final_rate':None,
            'scope':'full-CV UI candidate and moving-block diagnostic, awaiting independent MBAR overlap/shape analysis; no new dynamics',
            'source_at_start':before,'source_at_end':after,'sources_unchanged':before==after,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'production_cv_sha256':hashlib.sha256((args.output/'production-cv.npz').read_bytes()).hexdigest(),
            'elapsed_seconds':time.perf_counter()-started}
    write_json(args.output/'ui-report.json',report)
    print(json.dumps({key:report[key] for key in ['beads','samples','ratio_to_paper','elapsed_seconds']}),flush=True)

if __name__=='__main__': main()
