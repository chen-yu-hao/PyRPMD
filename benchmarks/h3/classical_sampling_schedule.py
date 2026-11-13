"""Select a fixed umbrella ladder from equilibrium overlap and support only.

No experimental or published rate is an input. The initial mesh is refined
until every adjacent Bhattacharyya overlap is at least 0.60. Strong barriers
retain 30-hartree umbrellas; broad far-entrance windows use 2 hartree.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import trapezoid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.h3.classical_sampling import umbrella_moments,gaussian_ui_from_moments
from pyrpmd.constants import KB_AU


def build(reference, output, *, minimum_overlap=.60):
    reference,output=Path(reference),Path(output)
    with np.load(reference,allow_pickle=False) as data:
        x=data['xi'].copy();density=data['density'].copy()
        grid=data['analysis_xi'].copy();exact=data['exact_pmf_hartree'].copy()
    beta=1/(KB_AU*300.)
    centers=np.round(np.r_[np.arange(-.08,.801,.04),[.82,.84,.86,.88,.90,.92,.93],np.arange(.94,1.071,.01)],8)
    stiffness=np.r_[np.full(23,2.),[2.,5.,5.,10.,15.,20.,30.],np.full(14,30.)]
    insertions=[]
    def distributions():
        logweight=np.log(density)[:,None]-.5*beta*stiffness*(x[:,None]-centers)**2
        weight=np.exp(logweight-logweight.max(axis=0))
        return weight/trapezoid(weight,x,axis=0)
    for _ in range(100):
        weight=distributions()
        overlap=trapezoid(np.sqrt(weight[:,:-1]*weight[:,1:]),x,axis=0)
        i=int(overlap.argmin())
        if overlap[i]>=minimum_overlap:
            break
        center=float((centers[i]+centers[i+1])/2)
        spring=float(np.sqrt(stiffness[i]*stiffness[i+1]))
        insertions.append({'center':center,'stiffness':spring,'previous_overlap':float(overlap[i]),
                           'between':[float(centers[i]),float(centers[i+1])]})
        centers=np.insert(centers,i+1,center);stiffness=np.insert(stiffness,i+1,spring)
    else:
        raise RuntimeError('overlap schedule did not converge within100 insertions')
    means,variances=umbrella_moments(x,density,centers,beta=beta,stiffness=stiffness)
    ui,_=gaussian_ui_from_moments(means,variances,centers,grid,beta=beta,stiffness=stiffness)
    near=(x>=.998-1e-10)&(x<=1.002+1e-10)
    support=trapezoid(weight[near],x[near],axis=0)
    document={'schema':'pyrpmd.h3.oracle-designed-umbrella-schedule.v1',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'design_inputs':'Classical BKMP2 equilibrium conditional density only; no paper rate or candidate sampled rate.',
        'selection_rule':{'initial_far_entrance_spacing':.04,'far_entrance_stiffness_hartree':2.,
                          'barrier_stiffness_hartree':30.,'minimum_adjacent_bhattacharyya_overlap':minimum_overlap,
                          'refinement':'Bisect the lowest-overlap adjacent pair, using the geometric mean stiffness.'},
        'centers':centers.tolist(),'stiffness':stiffness.tolist(),'windows':len(centers),
        'insertions':insertions,'adjacent_bhattacharyya_overlap':overlap.tolist(),
        'minimum_adjacent_bhattacharyya_overlap':float(overlap.min()),
        'equilibrium_mean':means.tolist(),'equilibrium_variance':variances.tolist(),
        'probability_in_xi_1_plusminus_002_per_window':support.tolist(),
        'sum_probabilities_in_xi_1_plusminus_002':float(support.sum()),
        'expected_near_barrier_observations_per_20000_sweep_ladder':float(support.sum()*20000),
        'gaussian_ui_infinite_sample_rate_ratio_to_exact':float(np.exp(-beta*(ui.max()-exact.max()))),
        'analysis':'Full-data histogram WHAM with bin refinement; Gaussian UI is diagnostic only.',
        'initial_campaign':{'beads':1,'temperature_K':300.,'warmup_sweeps':2000,
                            'production_sweeps':20000,'independent_seeds':[2026110101,2026110102,2026110103,2026110104]},
        'reference':str(reference),'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(document,indent=2)+'\n')
    print(json.dumps({key:document[key] for key in ['windows','minimum_adjacent_bhattacharyya_overlap',
        'expected_near_barrier_observations_per_20000_sweep_ladder','gaussian_ui_infinite_sample_rate_ratio_to_exact','insertions']},indent=2),flush=True)
    return document


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.reference,a.output)


if __name__=='__main__':main()
