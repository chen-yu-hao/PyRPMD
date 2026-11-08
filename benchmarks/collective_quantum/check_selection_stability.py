from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from pyrpmd.analysis.goal_quantum_coordinates import screen_chart_subspace,choose_core_from_probes
from pyrpmd.analysis.curvilinear_quantum import triatomic_transfer_geometry
out=ROOT/'results/molecular_main';m=np.array([78.9183376,1.00782503223,34.968852682])*1822.888486209
records={};rng=np.random.default_rng(3119)
for T in [150,300,600]:
    a=np.load(out/f'nonlinear_pilot_T{T}.npz');u=a['u'];f=a['generalized_forces'];n=len(u)
    geom=lambda v:triatomic_transfer_geometry(v,m)
    fractions={}
    for frac in [.95,.99,.995]:
        r=screen_chart_subspace(u,f,geom,[1,0,0],np.arange(n),retained_fraction=frac)
        fractions[str(frac)]={'dimension':r['required_dimension'],'status':r['status']}
    counts={};failures=0
    for b in range(32):
        take=rng.integers(n,size=n)
        r=screen_chart_subspace(u[take],f[take],geom,[1,0,0],take)
        key=str(r['required_dimension']);counts[key]=counts.get(key,0)+1
        failures+=r['status']=='pilot_force_fit_unreliable'
    records[str(T)]={'retained_fraction_sensitivity':fractions,'group_bootstrap_dimension_counts':counts,'bootstrap_force_fit_failures':int(failures)}
    print(T,records[str(T)],flush=True)
records['approval_gate']=choose_core_from_probes([{'name':'nonlinear_transfer_1d','dimension':1,'target_id':'ClHBr-log-bond-transfer',
    'required_probes':['collective_stretch','bend','stretch_and_bend','finite_J_kinetic']}],{})
(out/'stability.json').write_text(json.dumps(records,indent=2))
