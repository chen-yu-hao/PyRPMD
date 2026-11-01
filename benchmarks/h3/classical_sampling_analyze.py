"""Validate WHAM against a deterministic oracle, then audit P1 ladders.

Whole exchange ladders are the independent uncertainty units. All time
splits keep the coupled windows together. No Gaussian-UI rate is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import roots_legendre
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.h3.classical_sampling import umbrella_moments
from benchmarks.h3.mixing_analysis import histogram_samples,solve_wham,rate_candidate,analyze_run
from benchmarks.h3.classical_sampling_campaign import SEEDS,array_hash,file_hash
from pyrpmd.constants import KB_AU
from pyrpmd.io import write_json


def oracle_check(reference,schedule,output):
    reference,schedule,output=map(Path,(reference,schedule,output))
    output.mkdir(parents=True,exist_ok=True)
    parameters=json.loads(schedule.read_text());beta=1/(KB_AU*300.)
    centers=np.array(parameters['centers']);stiffness=np.array(parameters['stiffness'])
    with np.load(reference,allow_pickle=False) as saved:
        grid=saved['xi'].copy();density=saved['density'].copy()
    logdensity=CubicSpline(grid,np.log(density))
    expected_logratio=float(logdensity(1.)-logdensity(0.))
    nodes,weights=roots_legendre(6)
    rows=[]
    for width in (.002,.001,.0005,.00025):
        edges=(np.arange(round(-.2/width),round(1.2/width)+2)-.5)*width
        mid=(edges[:-1]+edges[1:])/2
        integration_points=mid[:,None]+nodes[None,:]*width/2
        integrated=[]
        for center,k in zip(centers,stiffness):
            logs=logdensity(integration_points)-.5*beta*k*(integration_points-center)**2
            logs-=logs.max()
            probabilities=np.exp(logs)@weights*width/2
            integrated.append(probabilities/probabilities.sum()*1e6)
        fit=solve_wham(np.array(integrated),edges,centers,stiffness,beta)
        candidates={str(h):rate_candidate(fit,beta,halfwidth=h) for h in (.002,.003,.004,.006) if 2*h/width+1>=5}
        rows.append({'width':width,'solver_residual':fit['residual'],'connected':fit['overlap_connected'],
                     'candidate_relative_bias':{h:float(np.exp(v['log_density_ratio']-expected_logratio)-1) for h,v in candidates.items()}})
    # A midpoint-discretized probability oracle must be recovered exactly by
    # the midpoint-potential WHAM model; this is distinct from finite bin bias.
    width=.0005;edges=(np.arange(-400,2402)-.5)*width;mid=(edges[:-1]+edges[1:])/2
    logprob=logdensity(mid)[None,:]-.5*beta*stiffness[:,None]*(mid-centers[:,None])**2
    logprob-=logprob.max(axis=1)[:,None]
    H=np.exp(logprob);H/=H.sum(axis=1)[:,None];H*=1e6
    fit=solve_wham(H,edges,centers,stiffness,beta)
    difference=fit['log_density']-logdensity(mid)
    region=(mid>=0)&(mid<=1.02)
    report={'schema':'pyrpmd.h3.wham-independent-conditional-oracle.v1',
            'fractional_counts_are_quadrature_weights_not_observations':True,
            'exact_log_density_ratio_xi_1_over_0':expected_logratio,
            'discrete_oracle_log_density_error_after_constant_gauge':float(np.ptp(difference[region])),
            'physical_bin_and_local_fit_bias':rows,'reference_sha256':file_hash(reference),
            'schedule_sha256':file_hash(schedule),'source_sha256':file_hash(Path(__file__)),
            'analysis_source_sha256':file_hash(ROOT/'benchmarks/h3/mixing_analysis.py')}
    write_json(output/'wham-oracle.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report


def campaign_check(campaign,reference,schedule,output,*,stage):
    campaign,reference,schedule,output=map(Path,(campaign,reference,schedule,output))
    output.mkdir(parents=True,exist_ok=True)
    params=json.loads(schedule.read_text());centers=np.array(params['centers']);k=np.array(params['stiffness'])
    beta=1/(KB_AU*300.)
    with np.load(reference,allow_pickle=False) as data:
        exact_means,exact_variances=umbrella_moments(data['xi'],data['density'],centers,beta=beta,stiffness=k)
        exact_logratio=float(np.log(np.interp(1.,data['xi'],data['density'])/np.interp(0.,data['xi'],data['density'])))
    width=.0005;edges=(np.arange(round(-.3/width),round(1.3/width)+2)-.5)*width
    pooled=np.zeros((len(centers),len(edges)-1));pooled_parts=[np.zeros_like(pooled),np.zeros_like(pooled)]
    means=[];variances=[];half_means=[];half_variances=[];candidate_logs=[];half_logs=[];details=[]
    for seed in SEEDS:
        path=campaign/f'seed-{seed}'
        report,fit=analyze_run(path)
        if report['production_sweeps']!=stage:
            raise ValueError('analysis stage differs from completed full checkpoint')
        target=output/f'seed-{seed}';target.mkdir(exist_ok=True)
        write_json(target/'wham-report.json',report)
        np.savez_compressed(target/'wham-fit.npz',**fit)
        with np.load(path/'checkpoint.npz',allow_pickle=False) as data:
            samples=data['samples']
            expected=json.loads((path/'stages'/str(stage)/'stage.json').read_text())['samples_prefix_sha256']
            if array_hash(samples)!=expected:
                raise ValueError('stage CV checksum mismatch')
            pooled+=histogram_samples(samples,edges=edges)[0]
            for i,part in enumerate((samples[:stage//2],samples[stage//2:])):
                pooled_parts[i]+=histogram_samples(part,edges=edges)[0]
                half_means.append(part.mean(axis=0));half_variances.append(part.var(axis=0,ddof=1))
            mean=samples.mean(axis=0);variance=samples.var(axis=0)
            means.append(mean);variances.append(variance)
            se=np.sqrt(variance*np.asarray(report['cv_inefficiencies'])/stage)
            z=(mean-exact_means)/se
        candidate=report['estimates'][-1]
        candidate_logs.append(candidate['log_density_ratio'])
        half_logs.append([report['time_parts'][name]['log_density_ratio'] for name in ('first_half','second_half')])
        details.append({'seed':seed,'sample_mean':mean,'sample_variance':variance,'mean_error_diagnostic_z':z,
                        'variance_ratio_to_oracle':variance/exact_variances,
                        'relative_rate_error_to_oracle':float(np.exp(candidate['log_density_ratio']-exact_logratio)-1),
                        'qtst_cm3_molecule_s':candidate['qtst_cm3_molecule_s'],
                        'half_log_density_ratios':half_logs[-1],
                        'minimum_recorded_pair_bohr':report['minimum_sampled_bead_pair_bohr'],
                        'ess_quantiles':np.quantile(report['cv_ess'],[0,.25,.5,.75,1])})
    combined=solve_wham(pooled,edges,centers,k,beta);estimate=rate_candidate(combined,beta)
    split_estimates=[rate_candidate(solve_wham(h,edges,centers,k,beta),beta) for h in pooled_parts]
    # Independent full ladders, not their windows, determine this interval.
    logs=np.asarray(candidate_logs);se=float(logs.std(ddof=1)/np.sqrt(len(logs)))
    delta=float(student_t.ppf(.975,len(logs)-1)*se)
    interval=estimate['prefactor_cm3_molecule_s']*np.exp(logs.mean()+np.array([-delta,delta]))
    half_logs=np.asarray(half_logs);differences=half_logs[:,1]-half_logs[:,0]
    drift_se=float(differences.std(ddof=1)/np.sqrt(len(differences)))
    within=np.mean(half_variances,axis=0);between=(stage//2)*np.var(half_means,axis=0,ddof=1)
    rhat=np.sqrt(((stage//2-1)/(stage//2)*within+between/(stage//2))/within)
    comparison=float(np.exp(estimate['log_density_ratio']-exact_logratio)-1)
    gates={'relative_95_halfwidth_at_most_1_percent':bool(np.expm1(delta)<=.01),
           'relative_rate_error_at_most_2_percent':abs(comparison)<=.02,
           'oracle_inside_independent_seed_interval':bool(abs(logs.mean()-exact_logratio)<=delta),
           'pooled_temporal_half_drift_within_3_se':bool(abs(differences.mean())<=3*drift_se),
           'split_rhat_at_most_1_01_every_window':bool(rhat.max()<=1.01)}
    report={'schema':'pyrpmd.h3.classical-independent-ladder-comparison.v1','production_sweeps_per_ladder':stage,
            'independent_seeds':list(SEEDS),'windows':len(centers),'pooled_wham':estimate,
            'independent_seed_log_ratio_mean':float(logs.mean()),'independent_seed_log_ratio_standard_error':se,
            'independent_seed_student_t_95_interval_cm3_molecule_s':interval,
            'relative_95_interval_halfwidth_log_scale':delta,
            'relative_rate_error_to_oracle':comparison,'temporal_half_pooled_estimates':split_estimates,
            'paired_half_log_ratio_mean_difference':float(differences.mean()),'paired_half_difference_se':drift_se,
            'split_rhat_per_window':rhat,'equilibrium_means':exact_means,'equilibrium_variances':exact_variances,
            'per_seed':details,'criteria':gates,'equilibrium_sampling_criteria_passed':all(gates.values()),
            'transmission_coefficient':None,'full_thermal_rate_validated':False,
            'source_sha256':file_hash(Path(__file__)),'reference_sha256':file_hash(reference),
            'schedule_sha256':file_hash(schedule),'analysis_source_sha256':file_hash(ROOT/'benchmarks/h3/mixing_analysis.py')}
    write_json(output/'campaign-comparison.json',report)
    np.savez_compressed(output/'pooled-wham.npz',**combined)
    print(json.dumps({key:report[key] for key in ['production_sweeps_per_ladder','relative_rate_error_to_oracle',
        'relative_95_interval_halfwidth_log_scale','paired_half_log_ratio_mean_difference','criteria']},indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['oracle','campaign'])
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--schedule',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--campaign',type=Path)
    p.add_argument('--stage',type=int);a=p.parse_args()
    if a.mode=='oracle':oracle_check(a.reference,a.schedule,a.output)
    else:campaign_check(a.campaign,a.reference,a.schedule,a.output,stage=a.stage)


if __name__=='__main__':main()
