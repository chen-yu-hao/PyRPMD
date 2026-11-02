"""Memory-bounded histogram WHAM for independently seeded H3 ladders.

Unlike Gaussian umbrella integration, this estimator makes no Gaussian
assumption about the sampled CV distribution. It bins every observation,
and explicitly requires bin-width refinement. Correlated exchange ladders
are kept together for time-split diagnostics; independent seeds, not windows
or saved configurations, are the uncertainty units reported here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import logsumexp
from scipy.sparse.csgraph import connected_components
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from benchmarks.h3.classical_tst import MASS_AU
from pyrpmd.analysis.fes import cv_statistical_inefficiency
from pyrpmd.constants import AU_RATE_TO_CM3_S, KB_AU
from pyrpmd.io import write_json


def histogram_samples(samples, width=.0005, edges=None):
    """Histogram [sweeps,windows] without dropping any observation."""
    samples=np.asarray(samples,float)
    if samples.ndim!=2 or min(samples.shape)<1 or not np.isfinite(samples).all() or width<=0:
        raise ValueError('finite samples[sweeps,windows] and positive width required')
    if edges is None:
        lower=int(np.floor(samples.min()/width))-1
        upper=int(np.ceil(samples.max()/width))+1
        edges=(np.arange(lower,upper+2)-.5)*width
    edges=np.asarray(edges,float)
    hist=np.array([np.histogram(samples[:,i],edges)[0] for i in range(samples.shape[1])])
    if hist.sum()!=samples.size:
        raise ValueError('histogram range would discard samples')
    return hist,edges


def solve_wham(histogram, edges, centers, stiffness, beta, *, tolerance=1e-10,
               initial_free_energies=None, max_iterations=120):
    """Weighted-bin likelihood, with gauge-fixed Newton solve and residual check.

    Each bin represents its midpoint umbrella potential. Counts can be
    fractional for deterministic-oracle tests; this does not turn fractional
    counts into independent observations for an uncertainty calculation.
    """
    H=np.asarray(histogram,float);edges=np.asarray(edges,float)
    centers=np.asarray(centers,float);k=np.broadcast_to(np.asarray(stiffness,float),centers.shape)
    if (H.ndim!=2 or H.shape!=(len(centers),len(edges)-1) or
        not np.isfinite(H).all() or np.any(H<0) or np.any(H.sum(axis=1)<=0)
        or np.any(np.diff(edges)<=0) or beta<=0 or np.any(k<=0)):
        raise ValueError('invalid WHAM histogram/umbrella parameters')
    grid=(edges[:-1]+edges[1:])/2
    occupied=H.sum(axis=0)>0
    x=grid[occupied]; h=H[:,occupied]
    counts=h.sum(axis=1); n=counts/counts.sum(); logn=np.log(n)
    total=h.sum(axis=0)/counts.sum()
    u=.5*beta*k[:,None]*(x[None]-centers[:,None])**2
    f=np.zeros(len(n)) if initial_free_energies is None else np.asarray(initial_free_energies,float).copy()
    if f.shape!=n.shape or not np.isfinite(f).all(): raise ValueError('invalid initial free energies')
    f-=f[0]
    def equations(z):
        logd=logsumexp(logn[:,None]+z[:,None]-u,axis=0)
        logrho=np.log(total)-logd
        new=-logsumexp(logrho[None]-u,axis=1);new-=new[0]
        return logd,logrho,new,float(np.max(np.abs(new-z)))
    for _ in range(8): f=equations(f)[2]
    iterations=0
    for iterations in range(1,max_iterations+1):
        logd,_,_,residual=equations(f)
        if residual<tolerance: break
        prob=np.exp(logn[:,None]+f[:,None]-u-logd[None])
        sums=prob@total;grad=sums-n
        hessian=np.diag(sums)-(prob*total[None])@prob.T
        step=np.linalg.solve(hessian[1:,1:],grad[1:])
        descent=float(grad[1:]@step)
        if not np.isfinite(step).all() or descent<=0: break
        value=float(total@logd-n@f)
        alpha=min(1.,10/max(float(np.max(np.abs(step))),1e-30))
        for _ in range(30):
            candidate=f.copy();candidate[1:]-=alpha*step
            cd,_,_,cr=equations(candidate)
            cv=float(total@cd-n@candidate)
            if cv<=value-1e-4*alpha*descent or (cv<=value+1e-13 and cr<residual):
                f=candidate;break
            alpha*=.5
        else: break
    logd,logrho,_,residual=equations(f)
    # Close to floating-point limits, a fixed-point finish gives an independent
    # convergence condition rather than trusting optimizer success flags.
    fixed_iterations=0
    while residual>=tolerance and fixed_iterations<10000:
        f=equations(f)[2];fixed_iterations+=1
        logd,logrho,_,residual=equations(f)
    prob=np.exp(logn[:,None]+f[:,None]-u-logd[None])
    overlap=(prob*total[None])@prob.T/n[:,None]
    graph=(overlap+overlap.T)>1e-6
    nc,labels=connected_components(graph,directed=False)
    logrho-=logsumexp(logrho)
    density=np.full(len(grid),np.nan)
    density[occupied]=logrho-np.log(np.diff(edges)[occupied])
    return dict(grid=grid,log_density=density,occupied=occupied,
        histogram=H,counts=counts,free_energies=f,overlap=overlap,
        overlap_components=labels,overlap_connected=nc==1,
        residual=residual,converged=residual<tolerance,
        newton_iterations=iterations,fixed_point_iterations=fixed_iterations)


def log_density_at(fit, value, *, halfwidth=.004):
    x=np.asarray(fit['grid']);y=np.asarray(fit['log_density'])
    raw=np.asarray(fit['histogram']).sum(axis=0)
    select=(np.abs(x-value)<=halfwidth+1e-12)&np.isfinite(y)
    if select.sum()<5: raise ValueError('insufficient local density support')
    a=(x[select]-value)/halfwidth
    coeff=np.polynomial.polynomial.polyfit(a,y[select],2,w=np.sqrt(raw[select]))
    residual=y[select]-np.polynomial.polynomial.polyval(a,coeff)
    return dict(value=float(value),log_density=float(coeff[0]),halfwidth=halfwidth,
        polynomial_coefficients=coeff,bins=int(select.sum()),raw_count=float(raw[select].sum()),
        weighted_residual_rms=float(np.sqrt(np.average(residual**2,weights=raw[select]))))


def rate_candidate(fit, beta, *, xi_star=1., halfwidth=.004, rinf=30., mass=MASS_AU):
    if not fit['converged'] or not fit['overlap_connected']:
        raise ValueError('WHAM must converge and have connected sampled support')
    reactant=log_density_at(fit,0.,halfwidth=halfwidth)
    transition=log_density_at(fit,xi_star,halfwidth=halfwidth)
    logratio=transition['log_density']-reactant['log_density']
    prefactor=4*np.pi*rinf*rinf/np.sqrt(2*np.pi*beta*(2*mass/3))*AU_RATE_TO_CM3_S
    return dict(xi_star=xi_star,log_density_ratio=logratio,barrier_hartree=-logratio/beta,
        qtst_cm3_molecule_s=float(prefactor*np.exp(logratio)),prefactor_cm3_molecule_s=prefactor,
        reactant_fit=reactant,transition_fit=transition)


def analyze_run(path, *, widths=(.001,.0005,.00025), time_splits=True):
    path=Path(path);settings=json.loads((path/'settings.json').read_text())
    # Keep one open inode while hashing and loading: the sampler atomically
    # replaces its checkpoint, so hashing the path afterwards could describe
    # a newer prefix than the observations actually analyzed.
    with (path/'checkpoint.npz').open('rb') as source:
        digest=hashlib.sha256()
        for block in iter(lambda:source.read(1024*1024),b''):
            digest.update(block)
        checkpoint_sha256=digest.hexdigest()
        source.seek(0)
        with np.load(source,allow_pickle=False) as saved:
            samples=saved['samples'].copy();diagnostics=saved['diagnostics'].copy()
            accepted=saved['accepted'].copy();proposed=saved['proposed'].copy()
            completed=int(saved['completed'])
    if len(samples)<100: raise ValueError('not enough completed production samples')
    beta=settings['beta'];centers=np.array(settings['centers']);k=np.array(settings['stiffness'])
    estimates=[];reference=None
    for width in widths:
        hist,edges=histogram_samples(samples,width)
        fit=solve_wham(hist,edges,centers,k,beta)
        candidate=rate_candidate(fit,beta)
        estimates.append(dict(bin_width=width,solver_residual=fit['residual'],
            overlap_connected=fit['overlap_connected'],**candidate))
        reference=(fit,edges)
    parts={}
    if time_splits:
        for name,a,b in [('first_half',0,len(samples)//2),('second_half',len(samples)//2,len(samples))]:
            hist,edges=histogram_samples(samples[a:b],widths[-1])
            fit=solve_wham(hist,edges,centers,k,beta)
            parts[name]=rate_candidate(fit,beta)
    gs=np.array([cv_statistical_inefficiency(samples[:,i]) for i in range(samples.shape[1])])
    fit,edges=reference
    fit_widths={str(w):rate_candidate(fit,beta,halfwidth=w) for w in (.002,.003,.004,.006)}
    report=dict(schema='pyrpmd.h3.wham-audit.v1',path=str(path),seed=settings['seed'],beads=settings['beads'],
        completed_sweeps=completed,production_sweeps=len(samples),windows=samples.shape[1],
        checkpoint_sha256=checkpoint_sha256,
        settings_sha256=hashlib.sha256((path/'settings.json').read_bytes()).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        estimates=estimates,time_parts=parts,fit_width_sensitivity=fit_widths,
        cv_inefficiencies=gs,cv_ess=len(samples)/gs,
        minimum_sampled_bead_pair_bohr=float(diagnostics[:,:,3].min()),
        mean_energy_per_bead_hartree=float(diagnostics[:,:,4].mean()),
        mean_acceptance_per_move=(accepted/np.maximum(proposed,1)).mean(axis=1),
        physical_converged=False,literature_reproduced=False,
        uncertainty_unit='independent full exchange ladder; no per-window independence assumption')
    return report,fit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--quick',action='store_true')
    a=p.parse_args();start=time.perf_counter();a.output.mkdir(parents=True,exist_ok=True)
    report,fit=analyze_run(a.run,widths=(.0005,) if a.quick else (.001,.0005,.00025),time_splits=not a.quick)
    report['elapsed_seconds']=time.perf_counter()-start
    write_json(a.output/'report.json',report)
    np.savez_compressed(a.output/'wham.npz',**fit)
    print(json.dumps({'seed':report['seed'],'production':report['production_sweeps'],
        'qtst':report['estimates'][-1]['qtst_cm3_molecule_s'],'elapsed':report['elapsed_seconds']}),flush=True)


if __name__=='__main__': main()
