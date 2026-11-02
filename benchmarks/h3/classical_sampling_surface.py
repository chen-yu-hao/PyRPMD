"""Independent classical TS flux ensemble and native real-time recrossing.

This samples the analytically reduced surface measure, not an umbrella or a
constrained-parent trajectory. Child dynamics are unmodified PyRPMD Dynamics.
The estimator uses antithetic Maxwell momenta and the same flux weights as
the production rate workflow. No published transmission value enters it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import ndtr,ndtri,roots_legendre
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.h3.classical_tst import MASS_AU,quadrature
from benchmarks.h3.classical_sampling_campaign import file_hash
from benchmarks.h3.pes import BKMP2Potential
from pyrpmd.constants import KB_AU,FS_TO_AU
from pyrpmd.cv import RPMDrateCoordinate
from pyrpmd.dynamics import Dynamics,State
from pyrpmd.io import write_json

BETA=1/(KB_AU*300.)
SEEDS=(2026110201,2026110202,2026110203,2026110204)
MU_R=1.7570000251
SIGMA_R=.1
SCALE_Y=.05


def surface_positions(rt):
    rt=np.asarray(rt,float);r,t=rt[...,0],rt[...,1]
    cosine=1-t*t/2
    q=np.zeros(rt.shape[:-1]+(3,3))
    q[...,0,0]=r
    q[...,2,0]=r*cosine
    q[...,2,1]=r*np.sqrt(np.maximum(0.,1-cosine*cosine))
    return q-q.mean(axis=-2,keepdims=True)


def surface_coordinate():
    coordinate=RPMDrateCoordinate([[0],[1,2]],np.full(3,MASS_AU),
        np.array([[-MU_R,0,0],[0,0,0],[MU_R,0,0]]),[[0,1]],[[1,2]],30.,
        equivalent_paths=[[0,2,1]])
    return coordinate.dividing_surface(1.)


def proposal(rng,count):
    lo,hi=ndtr((np.array([.5,8.])-MU_R)/SIGMA_R)
    r=MU_R+SIGMA_R*ndtri(lo+(hi-lo)*rng.random(count))
    y=-SCALE_Y*np.log1p(-rng.random(count)*(-np.expm1(-1/SCALE_Y)))
    return np.column_stack((r,2-y))


def proposal_log_density(rt):
    r,t=np.asarray(rt).T
    lo,hi=ndtr((np.array([.5,8.])-MU_R)/SIGMA_R)
    return (-.5*((r-MU_R)/SIGMA_R)**2-np.log(SIGMA_R*np.sqrt(2*np.pi))-np.log(hi-lo)
            -(2-t)/SCALE_Y-np.log(SCALE_Y)-np.log(-np.expm1(-1/SCALE_Y)))


def surface_log_density(potential,rt):
    rt=np.asarray(rt,float)
    result=potential.evaluate(surface_positions(rt))
    if np.any(result.status):raise ValueError('invalid surface geometry')
    r,t=rt.T
    # The constant 1/sqrt(2*pi*beta) cancels in the normalized source ensemble.
    return 4*np.log(r)+np.log(t)+.5*np.log((2+t*t)/MASS_AU)-BETA*result.energy,result.energy


def surface_moments(potential,order=256):
    nodes,weights=roots_legendre(order)
    r=.5+(nodes+1)*7.5/2;wr=weights*7.5/2
    t=1.5+nodes/2;wt=weights/2
    rr,tt=np.meshgrid(r,t,indexing='ij');rt=np.column_stack((rr.ravel(),tt.ravel()))
    logtarget,energy=surface_log_density(potential,rt)
    weight=(wr[:,None]*wt[None,:]).ravel()*np.exp(logtarget-logtarget.max())
    observations=np.column_stack((rt,rt*rt,energy));weight/=weight.sum()
    return weight@observations


def source_identity():
    paths=[Path(__file__),ROOT/'benchmarks/h3/pes.py',ROOT/'benchmarks/h3/classical_tst.py',
           ROOT/'src/pyrpmd/dynamics.py',ROOT/'src/pyrpmd/propagation.py',
           ROOT/'src/pyrpmd/cv.py',ROOT/'src/pyrpmd/constants.py']
    result={str(p.relative_to(ROOT)):file_hash(p) for p in paths}
    native=Path(importlib.import_module('pyrpmd._native').__file__)
    result['loaded_native:'+native.name]=file_hash(native)
    return result


def sample_surface(output,library,seed,*,chains=32,per_chain=64,warmup=256,interval=10):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    saved=output/'surface-parents.npz'
    if saved.exists():
        metadata=json.loads((output/'surface-source.json').read_text())
        if metadata['source_file_sha256']!=file_hash(saved):raise ValueError('source checkpoint mismatch')
        return metadata
    before=source_identity();potential=BKMP2Potential(library)
    rng=np.random.default_rng(np.random.SeedSequence([seed,101]))
    rt=proposal(rng,chains);target,energy=surface_log_density(potential,rt)
    density=proposal_log_density(rt)
    values=np.empty((chains,per_chain,2));energies=np.empty((chains,per_chain))
    accepted=np.zeros(chains,dtype=int)
    for step in range(warmup+per_chain*interval):
        candidate=proposal(rng,chains);new_target,new_energy=surface_log_density(potential,candidate)
        new_density=proposal_log_density(candidate)
        accept=np.log(rng.random(chains))<new_target-target+density-new_density
        rt[accept]=candidate[accept];target[accept]=new_target[accept]
        density[accept]=new_density[accept];energy[accept]=new_energy[accept]
        accepted+=accept
        elapsed=step+1-warmup
        if elapsed>0 and elapsed%interval==0:
            values[:,elapsed//interval-1]=rt;energies[:,elapsed//interval-1]=energy
    parent_rt=values.reshape(-1,2);q=surface_positions(parent_rt)
    channel=rng.integers(0,2,len(q))
    q[channel==1]=q[channel==1][:,[0,2,1]]
    observations=np.concatenate((values,values*values,energies[...,None]),axis=-1)
    chain_means=observations.mean(axis=1)
    mean=chain_means.mean(axis=0);se=chain_means.std(axis=0,ddof=1)/np.sqrt(chains)
    exact=surface_moments(potential)
    np.savez_compressed(saved,q=q,r_t=parent_rt,channel=channel,energy=energies.ravel(),
        chain_id=np.repeat(np.arange(chains),per_chain),rng=np.array(json.dumps(rng.bit_generator.state)))
    report={'schema':'pyrpmd.h3.classical-ts-source.v1','seed':seed,'parents':len(q),
            'chains':chains,'per_chain':per_chain,'warmup':warmup,'proposal_interval':interval,
            'acceptance':accepted/(warmup+per_chain*interval),
            'moment_names':['r','t','r_squared','t_squared','potential_energy'],
            'moments':mean,'moments_standard_error_across_chains':se,'quadrature_moments':exact,
            'moments_z':(mean-exact)/se,'chain_means':chain_means,
            'source_file_sha256':file_hash(saved),'source_at_start':before,
            'source_unchanged':before==source_identity(),'potential_metadata':potential.metadata,
            'scope':'Exact classical surface-flux measure; not a constrained-MD parent sampler.'}
    write_json(output/'surface-source.json',report);potential.close()
    return report


def paired_flux_values(velocity,flux,surface_values):
    surface_values=np.asarray(surface_values)
    return velocity/flux*((surface_values[0]>0).astype(float)-(surface_values[1]>0)),abs(velocity)/flux


def run_children(job):
    output,library,seed,timestep=job
    output=Path(output);source=output/f'seed-{seed}'
    sample_surface(source,library,seed)
    run=source/f'dt-{timestep:g}';run.mkdir(exist_ok=True)
    if (run/'children.npz').exists():return json.loads((run/'transmission.json').read_text())
    before=source_identity();started=time.perf_counter()
    with np.load(source/'surface-parents.npz',allow_pickle=False) as parents:
        positions=parents['q'].copy();chain_id=parents['chain_id'].copy()
    potential=BKMP2Potential(library)
    dynamics=Dynamics(potential,np.full(3,MASS_AU),BETA,1,timestep*FS_TO_AU,
                      native='required',loop_backend='native',block_steps=8192)
    surface=surface_coordinate()
    if not dynamics.can_fuse(observe=surface):raise ValueError('native child propagation unavailable')
    steps=round(400/timestep);stride=round(1/timestep);times=np.arange(1,401,dtype=float)
    count=len(positions)
    surfaces=np.empty((count,2,len(times)))
    numerators=np.empty((count,len(times)));denominators=np.empty(count)
    momenta=np.empty((count,3,3));final_q=np.empty((count,2,3,3));final_p=np.empty_like(final_q)
    energy_extrema=np.empty((count,2,3))
    for index,q in enumerate(positions):
        p=np.random.default_rng(np.random.SeedSequence([seed,307,index])).normal(size=(1,3,3))*np.sqrt(MASS_AU/BETA)
        momenta[index]=p[0]
        value,gradient=surface.centroid_value_gradient(q)
        if abs(value)>1e-10:raise ValueError('parent is not on the requested dividing surface')
        metric=np.sum(gradient*gradient)/MASS_AU
        flux=np.sqrt(metric/(2*np.pi*BETA));velocity=np.sum(gradient*p[0])/MASS_AU
        for side,sign in enumerate((1.,-1.)):
            state=State(q[None],sign*p)
            block=dynamics.propagate(state,steps,observe=surface,sample_interval=stride,record_centroids=False)
            if block.backend!='native':raise ValueError('unexpected child backend')
            surfaces[index,side]=block.values[:,0]
            energy_extrema[index,side]=[block.initial_energy,block.minimum_energy,block.maximum_energy]
            final_q[index,side]=state.q[0];final_p[index,side]=state.p[0]
        numerators[index],denominators[index]=paired_flux_values(velocity,flux,surfaces[index])
        if (index+1)%512==0:
            print(json.dumps({'seed':seed,'timestep':timestep,'parents_completed':index+1,
                              'elapsed_seconds':time.perf_counter()-started}),flush=True)
    np.savez_compressed(run/'children.npz',time_fs=times,surface_values=surfaces,
        numerator=numerators,denominator=denominators,initial_p=momenta,final_q=final_q,final_p=final_p,
        energy_extrema=energy_extrema,chain_id=chain_id)
    kappa=numerators.sum(axis=0)/denominators.sum()
    plateau={f'{lo}-{hi}':float(kappa[(times>=lo)&(times<=hi)].mean()) for lo,hi in [(80,100),(180,200),(380,400)]}
    initial=energy_extrema[:,:,0]
    energy_error=np.maximum(abs(energy_extrema[:,:,1]-initial),abs(energy_extrema[:,:,2]-initial))*BETA
    report={'schema':'pyrpmd.h3.classical-ts-transmission.v1','seed':seed,'timestep_fs':timestep,
            'parents':count,'antithetic_children':2*count,'horizon_fs':400.,'plateaus':plateau,
            'mean_denominator':float(denominators.mean()),'expected_mean_denominator':2.,
            'maximum_energy_error_kbt':float(energy_error.max()),
            'energy_error_kbt_quantiles':np.quantile(energy_error,[0,.5,.95,.99,1]),
            'propagation':dynamics.propagation_stats,'source_at_start':before,
            'source_unchanged':before==source_identity(),'source_parent_sha256':file_hash(source/'surface-parents.npz'),
            'children_sha256':file_hash(run/'children.npz'),'elapsed_seconds':time.perf_counter()-started,
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'Independent classical surface sampler plus unchanged native PyRPMD child dynamics.'}
    write_json(run/'transmission.json',report)
    np.savetxt(run/'kappa.csv',np.column_stack((times,kappa)),delimiter=',',header='time_fs,kappa',comments='')
    potential.close();return report


def combine(output,library):
    output=Path(output);rows=[];sources=[]
    for seed in SEEDS:
        sources.append(json.loads((output/f'seed-{seed}'/'surface-source.json').read_text()))
    chain_means=np.concatenate([source['chain_means'] for source in sources])
    mean=chain_means.mean(axis=0);se=chain_means.std(axis=0,ddof=1)/np.sqrt(len(chain_means))
    exact=np.asarray(sources[0]['quadrature_moments'])
    by_dt={}
    for dt in (.1,.05):
        reports=[json.loads((output/f'seed-{seed}'/f'dt-{dt:g}'/'transmission.json').read_text()) for seed in SEEDS]
        matrix=np.array([[r['plateaus'][name] for name in ('80-100','180-200','380-400')] for r in reports])
        by_dt[str(dt)]=matrix
        means=matrix.mean(axis=0);standard_errors=matrix.std(axis=0,ddof=1)/np.sqrt(len(SEEDS))
        halfwidth=student_t.ppf(.975,len(SEEDS)-1)*standard_errors
        rows.append({'timestep_fs':dt,'plateau_mean_across_seeds':means,'standard_error_across_seeds':standard_errors,
                     'student_t_95_halfwidth':halfwidth,'per_seed_plateaus':matrix,
                     'maximum_energy_error_kbt':max(r['maximum_energy_error_kbt'] for r in reports)})
    timestep_differences=by_dt['0.1']-by_dt['0.05']
    final=rows[-1];kappa=float(final['plateau_mean_across_seeds'][-1]);half=float(final['student_t_95_halfwidth'][-1])
    potential=BKMP2Potential(library);origin=potential.radial([1.4014719755,99.,99.])[0]
    tst=quadrature(potential,energy_origin=origin)['classical_tst_cm3_molecule_s'];potential.close()
    source_z=(mean-exact)/se
    gates={'surface_moments_within_3_standard_errors':bool(np.max(abs(source_z))<=3),
           'relative_kappa_95_halfwidth_at_most_1_percent':half/kappa<=.01,
           'maximum_time_plateau_difference_at_most_01':bool(max(np.ptp(row['plateau_mean_across_seeds']) for row in rows)<=.01),
           'maximum_timestep_difference_at_most_01':bool(np.max(abs(timestep_differences.mean(axis=0)))<=.01)}
    report={'schema':'pyrpmd.h3.classical-ts-transmission-comparison.v1',
            'independent_seeds':list(SEEDS),'parents_per_seed':2048,'results':rows,
            'surface_moments_pooled':mean,'surface_moments_quadrature':exact,'surface_moments_z':source_z,
            'paired_timestep_differences_mean':timestep_differences.mean(axis=0),
            'paired_timestep_differences_se':timestep_differences.std(axis=0,ddof=1)/np.sqrt(len(SEEDS)),
            'kappa_candidate':kappa,'kappa_student_t_95_interval':[kappa-half,kappa+half],
            'paper_classical_kappa':.994,'independent_classical_tst_cm3_molecule_s':tst,
            'independent_classical_rate_cm3_molecule_s':tst*kappa,
            'independent_classical_rate_95_interval':[tst*(kappa-half),tst*(kappa+half)],
            'paper_classical_rate_cm3_molecule_s':1.55e-18,'criteria':gates,
            'classical_reference_criteria_passed':all(gates.values()),
            'umbrella_sampling_validated_by_this_result':False,'quantum_rpmd_validated':False,
            'source_sha256':file_hash(Path(__file__))}
    write_json(output/'transmission-comparison.json',report)
    print(json.dumps({key:report[key] for key in ['kappa_candidate','kappa_student_t_95_interval',
        'independent_classical_rate_cm3_molecule_s','criteria']},indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--workers',type=int,choices=[1,2],default=2)
    a=p.parse_args()
    # Complete sources before concurrent dt runs to avoid source-file races.
    for seed in SEEDS:sample_surface(a.output/f'seed-{seed}',a.library,seed)
    jobs=[(a.output,a.library,seed,dt) for dt in (.1,.05) for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run_children,jobs))
    combine(a.output,a.library)


if __name__=='__main__':main()
