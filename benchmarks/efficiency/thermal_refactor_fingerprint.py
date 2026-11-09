"""Collect byte-exact states in separately built checkouts.

PYTHONPATH=/baseline/src python thermal_refactor_fingerprint.py baseline.json
PYTHONPATH=/candidate/src python thermal_refactor_fingerprint.py candidate.json --reference baseline.json

Compare only matched builds: FMA and serial builds each need their own baseline.
Array hashes include signed zeros and NaN payloads; RNG/checkpoint text is exact.
"""
import argparse,hashlib,json
import numpy as np
from pyrpmd.native_ensemble import NativeEnsemble

def digest(value):
    if isinstance(value,np.ndarray):return hashlib.sha256(value.tobytes()).hexdigest()
    return value
p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--reference');a=p.parse_args()
results={}
for rng in ('mt19937_64','philox-v1'):
 for beads in (1,2,3,7,8,16,32,64,128):
  for atoms in (7,8,9,10,11,12,13):
   for fixed in (False,True):
    mode='pile-l' if atoms%2 else 'trpmd'
    init=np.random.default_rng(1749+beads+atoms)
    q=init.normal(scale=.1,size=(beads,atoms,3));mom=init.normal(size=q.shape)
    freeze=[0,atoms-1] if fixed else []
    if fixed:q[:,freeze]=q[0,freeze];mom[:,freeze]=0
    engine=NativeEnsemble(q,np.linspace(.7,3.1,atoms),2.3,.01,p=mom,
        potential={'kind':'harmonic','k':.5},fixed_atoms=freeze,
        thermostat={'mode':mode,'rng':rng,'gamma_centroid':0.0 if beads==16 or mode=='trpmd' else .7},
        integrator='cayley-bcb' if beads%3 else 'normal-mode-bab',seed=903)
    obs=[]
    for steps in (1,4,3):
     r=engine.run(steps,record_centroids=True)
     obs.extend([digest(r['samples']),digest(r['centroids'])])
    name=f'{rng}/P{beads}/N{atoms}/fixed={fixed}'
    results[name]={'state':{k:digest(v) for k,v in engine.state().items()},'observations':obs}
# Above-threshold OpenMP and partial-tile checks.
for rng in ('mt19937_64','philox-v1'):
 for threads in ((1,2,4) if __import__('pyrpmd._thermal',fromlist=['has_openmp']).has_openmp() else (1,)):
  for atoms in (383,384,385):
   q=np.random.default_rng(atoms).normal(scale=.1,size=(32,atoms,3))
   e=NativeEnsemble(q,np.linspace(.5,4.,atoms),2.,.01,seed=903,threads=threads,
       thermostat={'mode':'pile-l','rng':rng},potential={'kind':'harmonic','k':.5})
   r=e.run(7,record_centroids=True)
   results[f'{rng}/thread{threads}/N{atoms}']={'state':{k:digest(v) for k,v in e.state().items()},'samples':digest(r['samples']),'centroids':digest(r['centroids'])}
with open(a.output,'w') as f:json.dump(results,f,indent=2)
print('cases:',len(results))
if a.reference:
    with open(a.reference) as f:
        reference=json.load(f)
    failures=sorted(k for k in reference.keys() | results.keys() if reference.get(k)!=results.get(k))
    print('byte-exact mismatches:',len(failures))
    if failures:
        raise SystemExit('Mismatched cases: '+', '.join(failures))
