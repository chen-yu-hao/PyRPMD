"""Compare completed classical umbrella observations with a quadrature oracle.

This is read-only with respect to the previous audit. Endpoint geometries
are labelled endpoints; they do not stand in for unavailable trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pyrpmd.analysis.fes import umbrella_integration, cv_statistical_inefficiency
from benchmarks.h3.classical_sampling import gaussian_ui_from_moments


def diagnose(run, reference, output):
    run,reference,output=map(Path,(run,reference,output))
    output.mkdir(parents=True,exist_ok=True)
    settings=json.loads((run/'settings.json').read_text())
    config=settings['config']
    if config['beads']!=1:
        raise ValueError('only classical P=1 observations are supported')
    with np.load(reference,allow_pickle=False) as ref:
        exact_centers=ref['centers'].copy();exact_means=ref['means'].copy();exact_variances=ref['variances'].copy()
        oracle_xi=ref['analysis_xi'].copy();oracle_pmf=ref['exact_pmf_hartree'].copy()
    arrays=[];centers=[];records=[];hashes=[]
    for window,center in enumerate(exact_centers):
        for replica in range(config['sampling']['replicas']):
            p=run/'run/umbrella/checkpoints'/f'window_{window:05d}_replica_{replica:04d}.npz'
            with np.load(p,allow_pickle=False) as saved:
                x=saved['values'][:,0].copy();q=saved['q'][0].copy()
            arrays.append(x);centers.append(center)
            g=cv_statistical_inefficiency(x)
            error=np.sqrt(x.var()*g/len(x))
            r_vector=q[2]-q[1];R_vector=q[0]-(q[1]+q[2])/2
            r=np.linalg.norm(r_vector);R=np.linalg.norm(R_vector)
            u=abs(np.dot(r_vector,R_vector)/(r*R))
            records.append({'window':window,'replica':replica,'center':float(center),
                'sample_mean':float(x.mean()),'sample_variance':float(x.var()),
                'equilibrium_mean':float(exact_means[window]),'equilibrium_variance':float(exact_variances[window]),
                'statistical_inefficiency':float(g),'effective_samples':float(len(x)/g),
                'mean_error_diagnostic_se':float(error),'mean_error_in_diagnostic_se':float((x.mean()-exact_means[window])/error),
                'first_half_mean':float(x[:len(x)//2].mean()),'second_half_mean':float(x[len(x)//2:].mean()),
                'endpoint_R_bohr':float(R),'endpoint_bond_bohr':float(r),'endpoint_abs_cosine':float(u)})
            hashes.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    centers=np.array(centers);grid=np.linspace(0,1.02,1021);k=np.full(len(arrays),30.)
    equilibrium_ui,_=gaussian_ui_from_moments(exact_means,exact_variances,exact_centers,grid,beta=config['beta'])
    curves={}
    for name,values in [('all',arrays),('first_half',[x[:len(x)//2] for x in arrays]),('second_half',[x[len(x)//2:] for x in arrays])]:
        f,_=umbrella_integration(values,centers,k,config['beta'],grid)
        curves[name]=f-f[0]
    exact=np.interp(grid,oracle_xi,oracle_pmf)
    error=curves['all']-equilibrium_ui
    domains=[(0.,.25),(.25,.5),(.5,.75),(.75,.9),(.9,1.)]
    contributions=[{'xi_range':[low,high],'barrier_error_hartree':float(np.interp(high,grid,error)-np.interp(low,grid,error)),
                    'barrier_error_kbt':float(config['beta']*(np.interp(high,grid,error)-np.interp(low,grid,error)))} for low,high in domains]
    groups=[]
    for low,high in [(-.05,.05),(.05,.5),(.5,.9),(.9,1.05)]:
        selected=[r for r in records if low-1e-10<=r['center']<high+1e-10]
        groups.append({'centers':[low,high],'endpoint_count':len(selected),
                       'endpoint_mean_abs_cosine':float(np.mean([r['endpoint_abs_cosine'] for r in selected])),
                       'endpoint_mean_bond_bohr':float(np.mean([r['endpoint_bond_bohr'] for r in selected]))})
    report={'schema':'pyrpmd.h3.classical-window-diagnosis.v1','run':str(run),
            'reference':str(reference),'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'physical_sampling_status':'Previous trajectories were nonstationary; diagnostic standard errors may under-cover.',
            'samples':sum(len(x) for x in arrays),'window_records':records,
            'barrier_error_decomposition_relative_to_infinite_sample_ui':contributions,
            'endpoint_geometry_groups':groups,'endpoint_warning':'Only final checkpoint geometries are available; these are not production time-series distributions.',
            'mean_discrepancy_diagnostic_se_abs_quantiles':np.quantile(np.abs([r['mean_error_in_diagnostic_se'] for r in records]),[0,.25,.5,.75,.95,1]).tolist(),
            'effective_sample_quantiles':np.quantile([r['effective_samples'] for r in records],[0,.25,.5,.75,1]).tolist(),
            'checkpoints':hashes}
    (output/'window-diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savetxt(output/'window-diagnosis-pmf.csv',np.column_stack([grid,exact,equilibrium_ui,*curves.values()]),delimiter=',',
               header=','.join(['xi','exact_equilibrium','infinite_sampling_gaussian_ui',*curves]),comments='')
    print(json.dumps({key:report[key] for key in ['mean_discrepancy_diagnostic_se_abs_quantiles','effective_sample_quantiles','endpoint_geometry_groups','barrier_error_decomposition_relative_to_infinite_sample_ui']},indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();diagnose(a.run,a.reference,a.output)


if __name__=='__main__':main()
