"""Audit every registered quantum ladder with cluster-level uncertainty.

The independent unit is an entire exchange ladder. The primary point is a
pooled WHAM fit; a leave-one-ladder-out jackknife and between-ladder log-rate
dispersion provide uncertainty. Time blocks always retain all windows.
Incomplete cohorts can be monitored explicitly but cannot pass the campaign.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import chi2, t as student_t

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from benchmarks.h3.mixing_analysis import histogram_samples, solve_wham, rate_candidate
from benchmarks.h3.quantum_sampling_campaign import array_hash, file_hash
from pyrpmd.analysis.fes import cv_statistical_inefficiency
from pyrpmd.io import write_json


def load_registered_prefix(path,stage):
    """Read one checkpoint inode and verify the retained production prefix."""
    path=Path(path)
    receipt=json.loads((path/'stages'/str(stage)/'stage.json').read_text())
    if receipt['production_sweeps']!=stage:
        raise ValueError('stage receipt count mismatch')
    with (path/'checkpoint.npz').open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest();stream.seek(0)
        with np.load(stream,allow_pickle=False) as data:
            all_samples=data['samples']
            if len(all_samples)<stage:raise ValueError('checkpoint is shorter than its registered prefix')
            samples=all_samples[:stage].copy()
            if array_hash(samples)!=receipt['samples_prefix_sha256']:
                raise ValueError('registered CV prefix checksum mismatch')
    return samples,dict(receipt=receipt,checkpoint_sha256=digest,
                       receipt_sha256=file_hash(path/'stages'/str(stage)/'stage.json'))


def cluster_statistics(pooled_log_rate,seed_logs,leave_one_out_logs,half_logs,within_variances):
    """Whole-ladder intervals and paired drift; no per-window independence."""
    logs=np.asarray(seed_logs,float);loo=np.asarray(leave_one_out_logs,float)
    halves=np.asarray(half_logs,float);variances=np.asarray(within_variances,float)
    n=len(logs)
    if (n<2 or loo.shape!=(n,) or halves.shape!=(n,2) or variances.shape!=(n,)
            or not np.isfinite(np.r_[logs,loo,halves.ravel(),variances,pooled_log_rate]).all()
            or np.any(variances<0)):
        raise ValueError('at least two complete finite ladders and valid variance estimates required')
    jackknife_se=float(np.sqrt((n-1)/n*np.sum((loo-loo.mean())**2)))
    independent_se=float(logs.std(ddof=1)/np.sqrt(n))
    # A pooled nonlinear fit and the mean log of separate fits need not have
    # identical variances at finite sample size. Retain the more conservative
    # of these two prespecified whole-ladder estimates.
    se=max(jackknife_se,independent_se)
    halfwidth=float(student_t.ppf(.975,n-1)*se)
    differences=halves[:,1]-halves[:,0]
    drift_se=float(differences.std(ddof=1)/np.sqrt(n))
    if np.any(variances==0):
        q=0. if np.ptp(logs)<1e-14 else None
        pvalue=1. if q==0. else 0.
    else:
        weights=1/variances;mean=float(np.average(logs,weights=weights))
        q=float(np.sum(weights*(logs-mean)**2));pvalue=float(chi2.sf(q,n-1))
    return dict(independent_ladders=n,jackknife_log_standard_error=jackknife_se,
        between_ladder_log_standard_error=independent_se,primary_log_standard_error=se,
        primary_student_t_95_log_halfwidth=halfwidth,
        primary_student_t_95_interval=np.exp(pooled_log_rate+np.array([-halfwidth,halfwidth])),
        independent_log_rate_mean=float(logs.mean()),
        paired_time_half_log_difference=float(differences.mean()),paired_time_half_standard_error=drift_se,
        heterogeneity_chi_square=q,heterogeneity_degrees_of_freedom=n-1,
        heterogeneity_p_approximate=pvalue,
        heterogeneity_note='Within-ladder variances use the larger 4- or 8-block estimate; chi-square calibration is approximate.',
        uncertainty_unit='independent entire exchange ladder')


def analyze(campaign,protocol_path,output,stage,*,available_only=False):
    campaign=Path(campaign);protocol_path=Path(protocol_path);output=Path(output)
    protocol=json.loads(protocol_path.read_text());started=time.perf_counter()
    source_before={str(Path(__file__).relative_to(ROOT)):file_hash(__file__),
                   'benchmarks/h3/mixing_analysis.py':file_hash(ROOT/'benchmarks/h3/mixing_analysis.py')}
    if stage not in protocol['fixed_refinement_prefixes'] or stage%8:
        raise ValueError('analysis needs a registered prefix divisible by eight')
    expected_seeds=protocol['independent_ladder_seeds']
    seeds=[seed for seed in expected_seeds if (campaign/f'seed-{seed}'/'stages'/str(stage)/'stage.json').exists()]
    if seeds!=expected_seeds and not available_only:
        raise ValueError('registered cohort incomplete; --available-only is diagnostic, never acceptance')
    if len(seeds)<2:raise ValueError('at least two completed ladders are needed')
    schedule=json.loads((ROOT/protocol['schedule_file']).read_text())
    if file_hash(ROOT/protocol['schedule_file'])!=protocol['schedule_sha256']:
        raise ValueError('registered schedule hash changed')
    centers=np.asarray(schedule['centers']);stiffness=np.asarray(schedule['stiffness'])
    widths=protocol['analysis']['bin_widths'];primary_width=widths[-1]
    edge_sets={w:(np.arange(round(-.5/w),round(1.5/w)+2)-.5)*w for w in widths}
    histograms={w:[] for w in widths};details=[];seed_logs=[];half_logs=[];variances=[]
    half_means=[];half_variances=[];beta=None;common_settings=None
    output.mkdir(parents=True,exist_ok=True)
    def fit_candidate(hist,edges,*,halfwidth=.004):
        fit=solve_wham(hist,edges,centers,stiffness,beta)
        candidate=rate_candidate(fit,beta,xi_star=protocol['analysis']['primary_xi'],halfwidth=halfwidth)
        return fit,candidate
    for seed in seeds:
        path=campaign/f'seed-{seed}';settings=json.loads((path/'settings.json').read_text())
        if (settings['seed']!=seed or settings['beads']!=protocol['beads_primary']
                or settings['warmup']!=protocol['warmup_sweeps']
                or settings['source_sha256'][protocol['sampler_file']]!=protocol['sampler_sha256']
                or not np.array_equal(settings['centers'],centers)
                or not np.array_equal(settings['stiffness'],stiffness)):
            raise ValueError('ladder settings do not match the protocol')
        scientific_settings={key:value for key,value in settings.items() if key!='seed'}
        if common_settings is not None and scientific_settings!=common_settings:
            raise ValueError('independent ladder scientific settings or source identities differ')
        common_settings=scientific_settings
        if not json.loads((path/'run-report.json').read_text())['source_unchanged']:
            raise ValueError('sampling source changed during execution')
        if beta is not None and beta!=settings['beta']:raise ValueError('ladder temperatures differ')
        beta=settings['beta'];samples,receipt=load_registered_prefix(path,stage)
        if receipt['receipt']['protocol_sha256']!=file_hash(protocol_path):
            raise ValueError('stage protocol checksum mismatch')
        if samples.shape!=(stage,len(centers)):raise ValueError('registered sample shape mismatch')
        per_width=[];primary_fit=None
        for w in widths:
            h,_=histogram_samples(samples,edges=edge_sets[w]);histograms[w].append(h)
            fit,candidate=fit_candidate(h,edge_sets[w]);per_width.append(dict(width=w,**candidate))
            if w==primary_width:primary_fit=fit
        current=[]
        for part in (samples[:stage//2],samples[stage//2:]):
            h,_=histogram_samples(part,edges=edge_sets[primary_width])
            _,candidate=fit_candidate(h,edge_sets[primary_width]);current.append(np.log(candidate['qtst_cm3_molecule_s']))
            half_means.append(part.mean(axis=0));half_variances.append(part.var(axis=0,ddof=1))
        block_logs={}
        for count in (4,8):
            estimates=[]
            for part in np.split(samples,count):
                h,_=histogram_samples(part,edges=edge_sets[primary_width])
                _,candidate=fit_candidate(h,edge_sets[primary_width]);estimates.append(np.log(candidate['qtst_cm3_molecule_s']))
            block_logs[str(count)]=estimates
        within=max(float(np.var(block_logs[str(count)],ddof=1)/count) for count in (4,8))
        inefficiencies=np.array([cv_statistical_inefficiency(samples[:,i]) for i in range(samples.shape[1])])
        seed_logs.append(np.log(per_width[-1]['qtst_cm3_molecule_s']));half_logs.append(current);variances.append(within)
        detail=dict(seed=seed,production_sweeps=stage,bin_refinement=per_width,time_half_log_rates=current,
            block_log_rates=block_logs,within_ladder_log_variance=within,
            cv_inefficiency_quantiles=np.quantile(inefficiencies,[0,.25,.5,.75,1]),
            cv_effective_sample_quantiles=np.quantile(stage/inefficiencies,[0,.25,.5,.75,1]),
            **receipt)
        details.append(detail);write_json(output/f'seed-{seed}.json',detail)
        np.savez_compressed(output/f'seed-{seed}-wham.npz',**primary_fit)
        print(json.dumps({'event':'seed_analyzed','seed':seed,'stage':stage}),flush=True)
        del samples
    pooled_candidates=[];primary=None;pooled_hist=None
    for w in widths:
        pooled=np.sum(histograms[w],axis=0)
        fit,candidate=fit_candidate(pooled,edge_sets[w])
        pooled_candidates.append(dict(width=w,solver_residual=fit['residual'],connected=fit['overlap_connected'],**candidate))
        if w==primary_width:primary=fit;pooled_hist=pooled
    primary_candidate=pooled_candidates[-1];lograte=np.log(primary_candidate['qtst_cm3_molecule_s'])
    loo=[]
    for h in histograms[primary_width]:
        _,candidate=fit_candidate(pooled_hist-h,edge_sets[primary_width]);loo.append(np.log(candidate['qtst_cm3_molecule_s']))
    statistics=cluster_statistics(lograte,seed_logs,loo,half_logs,variances)
    fit_widths={str(w):rate_candidate(primary,beta,xi_star=protocol['analysis']['primary_xi'],halfwidth=w)
                for w in [protocol['analysis']['local_log_density_fit_halfwidth'],*protocol['analysis']['fit_halfwidth_controls']]}
    bin_change=max(abs(np.log(r['qtst_cm3_molecule_s'])-lograte) for r in pooled_candidates)
    fit_change=max(abs(np.log(r['qtst_cm3_molecule_s'])-lograte) for r in fit_widths.values())
    paper=protocol['analysis']['primary_paper_qtst_cm3_molecule_s']
    comparison=float(np.exp(lograte)/paper-1);criteria=protocol['gates']
    gates=dict(complete_registered_cohort=seeds==expected_seeds,
        wham_residual_and_overlap=all(x['solver_residual']<criteria['wham_residual_max'] and x['connected'] for x in pooled_candidates),
        bin_refinement=bin_change<=criteria['bin_refinement_max_abs_log_rate_change'],
        local_fit_refinement=fit_change<=criteria['fit_width_max_abs_log_rate_change'],
        statistical_precision=statistics['primary_log_standard_error']<=criteria['target_qtst_relative_standard_error'],
        temporal_half_consistency=abs(statistics['paired_time_half_log_difference'])<=criteria['time_split_consistency_standard_errors']*statistics['paired_time_half_standard_error'],
        independent_seed_consistency=statistics['heterogeneity_p_approximate']>=criteria['independent_seed_heterogeneity_p_min'],
        paper_point_agreement=abs(comparison)<=criteria['paper_qtst_relative_difference_for_point_agreement'])
    within=np.mean(half_variances,axis=0);between=(stage//2)*np.var(half_means,axis=0,ddof=1)
    rhat=np.sqrt(((stage//2-1)/(stage//2)*within+between/(stage//2))/within)
    source_after={name:file_hash(ROOT/name) for name in source_before}
    if source_after!=source_before:raise ValueError('analysis source changed during execution')
    report=dict(schema='pyrpmd.h3.quantum-ladder-comparison.v1',
        created_utc=datetime.now(timezone.utc).isoformat(),production_sweeps_per_ladder=stage,
        expected_independent_seeds=expected_seeds,completed_independent_seeds=seeds,
        pooled_wham=primary_candidate,pooled_bin_refinement=pooled_candidates,
        maximum_bin_log_change=bin_change,maximum_local_fit_log_change=fit_change,
        local_fit_controls=fit_widths,statistics=statistics,split_rhat_per_window=rhat,
        paper_qtst_cm3_molecule_s=paper,relative_difference_from_paper=comparison,
        criteria=gates,registered_primary_qtst_gates_passed=all(gates.values()),
        bead_convergence_validated=False,physical_domain_validated=False,
        transmission_coefficient=None,full_rpmd_rate_validated=False,
        source_sha256=source_before,protocol_sha256=file_hash(protocol_path),
        elapsed_seconds=time.perf_counter()-started,
        note='Passing equilibrium gates alone does not validate bead/domain limits or a full dynamical rate.')
    write_json(output/'campaign-comparison.json',report);np.savez_compressed(output/'pooled-wham.npz',**primary)
    print(json.dumps({'qtst':primary_candidate['qtst_cm3_molecule_s'],
        'relative_difference_from_paper':comparison,'criteria':gates}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',type=int,required=True)
    p.add_argument('--available-only',action='store_true')
    a=p.parse_args();analyze(a.campaign,a.protocol,a.output,a.stage,available_only=a.available_only)


if __name__=='__main__':main()
