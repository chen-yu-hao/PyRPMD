"""Read-only replay and uncertainty diagnostics for the classical TS data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.h3.classical_sampling_surface import SEEDS,BETA
from benchmarks.h3.classical_tst import MASS_AU
from benchmarks.h3.classical_sampling_campaign import file_hash
from pyrpmd.io import write_json


def diagnose(data,output):
    data,output=Path(data),Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[];blocks=[];paired=[];native_steps=0
    for seed in SEEDS:
        source=data/f'seed-{seed}'
        with np.load(source/'surface-parents.npz',allow_pickle=False) as saved:
            q=saved['q'].copy();rt=saved['r_t'].copy();channel=saved['channel'].copy()
        previous=None
        for dt in (.1,.05):
            folder=source/f'dt-{dt:g}'
            report=json.loads((folder/'transmission.json').read_text())
            if file_hash(folder/'children.npz')!=report['children_sha256']:
                raise ValueError('child checksum mismatch')
            with np.load(folder/'children.npz',allow_pickle=False) as saved:
                p=saved['initial_p'];values=saved['surface_values'];times=saved['time_fs']
                numerator=saved['numerator'];denominator=saved['denominator'];chains=saved['chain_id']
                # Reconstruct the initial normal velocity directly from bond
                # velocities; do not call the production gradient function.
                partner=np.where(channel==0,1,2)
                qr=q[:,0]-q[np.arange(len(q)),partner]
                pr=p[:,0]-p[np.arange(len(q)),partner]
                breaking=q[:,2]-q[:,1];pbreaking=p[:,2]-p[:,1]
                velocity=(np.sum(breaking*pbreaking,axis=1)/np.linalg.norm(breaking,axis=1)
                          -np.sum(qr*pr,axis=1)/np.linalg.norm(qr,axis=1))/MASS_AU
                flux=np.sqrt((2+rt[:,1]**2)/(MASS_AU*2*np.pi*BETA))
                replay=(velocity/flux)[:,None]*((values[:,0]>0).astype(float)-(values[:,1]>0))
                numerator_error=float(np.max(abs(replay-numerator)))
                denominator_error=float(np.max(abs(abs(velocity)/flux-denominator)))
                if max(numerator_error,denominator_error)>1e-11:raise ValueError('independent flux replay failed')
                tail=(times>=380)&(times<=400)
                if dt==.05:
                    for chain in np.unique(chains):
                        select=chains==chain
                        blocks.append([float(numerator[select][:,tail].mean(axis=1).sum()),float(denominator[select].sum())])
                if previous is None:
                    previous={'initial_p':p.copy(),'outcomes':values>0,'values':values.copy()}
                else:
                    paired.append({'seed':seed,'initial_momenta_identical':bool(np.array_equal(previous['initial_p'],p)),
                        'changed_classifications_all_times':int(np.count_nonzero(previous['outcomes']!=(values>0))),
                        'changed_classifications_after_80fs':int(np.count_nonzero(previous['outcomes'][:,:,times>=80]!=(values[:,:,times>=80]>0))),
                        'maximum_surface_value_difference_bohr':float(np.max(abs(previous['values']-values)))})
            native_steps+=report['propagation']['native_steps']
            rows.append({'seed':seed,'timestep_fs':dt,'numerator_max_abs_replay_error':numerator_error,
                         'denominator_max_abs_replay_error':denominator_error,
                         'source_unchanged':report['source_unchanged'],'native_steps':report['propagation']['native_steps'],
                         'python_steps':report['propagation']['python_steps'],
                         'child_sha256':report['children_sha256']})
    blocks=np.asarray(blocks);kappa=float(blocks[:,0].sum()/blocks[:,1].sum())
    residual=blocks[:,0]-kappa*blocks[:,1]
    se=float(np.sqrt(len(blocks)/(len(blocks)-1)*np.sum(residual**2))/blocks[:,1].sum())
    half=float(student_t.ppf(.975,len(blocks)-1)*se)
    report={'schema':'pyrpmd.h3.classical-ts-independent-replay.v1','runs':rows,
        'total_native_physical_steps':native_steps,'paired_timestep_diagnostics':paired,
        'parent_chain_block_diagnostic':{'blocks':len(blocks),'parents_per_block':64,
            'pooled_ratio':kappa,'delta_method_standard_error':se,'student_t_95_interval':[kappa-half,kappa+half],
            'scope':'Supplementary chain-block diagnostic; the predeclared main uncertainty uses four independent full seeds.'},
        'zero_observed_timestep_difference_note':'No observed late-time classification changes is finite-sample evidence, not proof of exactly zero systematic error. Paired timesteps are not independent statistical replicas.',
        'source_sha256':file_hash(Path(__file__))}
    write_json(output/'transmission-replay.json',report)
    print(json.dumps({'native_steps':native_steps,'maximum_numerator_replay_error':max(r['numerator_max_abs_replay_error'] for r in rows),
        'chain_block_standard_error':se,'paired_timestep_diagnostics':paired},indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();diagnose(a.data,a.output)


if __name__=='__main__':main()
