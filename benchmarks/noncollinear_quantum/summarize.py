"""Summarize measured artifacts; never promote missing gates to convergence."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys,platform
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from pyrpmd.analysis.bimolecular_active_space import select_molecular_subspace
from pyrpmd.analysis.bimolecular_scattering import RATE_AU_TO_CM3_S,KB_AU

def build(root=ROOT,bootstrap=64):
    root=Path(root);out=root/'results';data={'scope':'noncollinear RPMD baseline and gated quantum audit; NO corrected gas-phase quantum rate',
        'base_commit':'ffa14f247e453e9626b2e3fb3c288da0ca753d2e','reaction':'35Cl + H79Br -> H35Cl + 79Br',
        'pes':'historical BrHCl_LEPS1 parameterization, independent analytic implementation',
        'physical_converged':False,'rates':{},'selection':{},'quantum_audit':{}}
    rows=[]
    ref=json.loads((out/'reactant_reference.json').read_text())
    data['reactant_reference']=ref
    for file in sorted(out.glob('T*/summary.json')):
        name=file.parent.name;r=json.loads(file.read_text());cfg=json.loads((file.parent/'input.json').read_text())
        r['configuration']=cfg;r['integrator']=cfg.get('integrator','velocity-verlet');r['source_energy_relative_bias']=None
        key=str(int(r['temperature_K']))
        reference=ref.get(key,ref.get(key+'.0'))
        if reference is not None:r['source_energy_relative_bias']=r['initial_stats']['internal_energy_Eh']/reference['internal_energy_Eh']-1
        ann=r['outer_annulus'];r['sampled_outer_annulus_rate_fraction']=ann['rate_contribution_au']/r['rate_lower_au'] if r['rate_lower_au']>0 else None
        z=np.load(file.parent/'trajectories.npz');re=z['outcomes']==1
        if re.any():
            r['closest_approach_angle_reactive_degrees_quantiles']=np.quantile(np.rad2deg(z['angle_at_H_rad'][re]),[.1,.5,.9]).tolist()
        data['rates'][name]=r
        rows.append([name,r['temperature_K'],r['beads'],r['integrator'],r['sample_count'],cfg['bmax_bohr'],
            r['rate_lower_cm3_molecule_s'],r['rate_upper_cm3_molecule_s'],r['sampling_se_cm3_molecule_s'],r['reactive'],r['unfinished'],r['invalid'],False])
    for T in [150,300,600]:
        rec=json.loads((out/f'selection_T{T}.json').read_text());z=np.load(out/f'pilot_T{T}.npz')
        args=dict(reference=z['reference'],hessians=z['hessians'],beta=1/(KB_AU*T),observable_gradient=z['observable_gradient'])
        r2=select_molecular_subspace(z['positions'],z['forces'],z['masses'],max_dimension=2,**args)
        rng=np.random.default_rng(20261108+T);counts={};N=len(z['positions'])
        for k in range(bootstrap):
            ix=rng.integers(N,size=N);kw=dict(args,hessians=z['hessians'][ix])
            rr=select_molecular_subspace(z['positions'][ix],z['forces'][ix],z['masses'],max_dimension=3,**kw)
            label=f'{rr.selected_dimension}:{rr.status}';counts[label]=counts.get(label,0)+1
        rec['budget_two']=r2.as_dict();rec['pilot_resampling_counts']=counts;rec['pilot_resampling_replicates']=bootstrap
        rec['interpretation']='Conservative local heuristic; full rank does not prove a physical lower bound on quantum dimension.'
        data['selection'][str(T)]=rec
    for name in ['quantum_J0_pilot','quantum_J0_check']:
        path=out/name;r=json.loads((path/'summary.json').read_text());z=np.load(path/'quantum.npz');w=z['hbr_weight']
        r.pop('minimum_arrangement_purity',None)
        r['maximum_arrangement_impurity']=float(np.minimum(w,1-w).max())
        r['configuration']=json.loads((path/'input.json').read_text())
        r.update(reference_usable_for_rate_correction=False,physical_converged=False,
                 unestablished_gates=['arrangement channel localization','angular basis/window','radial discretization','asymptotic matching','total-J/parity sum'])
        data['quantum_audit'][name]=r
    coarse=np.load(out/'quantum_J0_pilot/quantum.npz');fine=np.load(out/'quantum_J0_check/quantum.npz')
    E=fine['total_energy'];mask=(E>=coarse['total_energy'].min())&(E<=coarse['total_energy'].max())
    a=np.interp(E[mask],coarse['total_energy'],coarse['CRP']);b=fine['CRP'][mask]
    data['quantum_refinement_comparison']={'common_TOTAL_energy_points':int(mask.sum()),
        'CRP_relative_L2_change':float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)),
        'edge_threshold_change_Eh':data['quantum_audit']['quantum_J0_check']['edge_reactant_threshold_Eh']-data['quantum_audit']['quantum_J0_pilot']['edge_reactant_threshold_Eh'],
        'meaning':'Several numerical parameters changed together. This is not an isolated error bound. No threshold alignment was used.'}
    paired={}
    for a,b in [('T300_P32','T300_cayley32'),('T300_cayley32','T300_cayley32_half')]:
        x=np.load(out/a/'trajectories.npz');y=np.load(out/b/'trajectories.npz')
        if not np.array_equal(x['weights'],y['weights']):continue
        r=data['rates'][a];pref=np.sqrt(8*KB_AU*r['temperature_K']/(np.pi*r['reduced_mass_au']))*np.pi*r['bmax_bohr']**2*RATE_AU_TO_CM3_S
        delta=pref*x['weights']*((y['outcomes']==1).astype(float)-(x['outcomes']==1))
        paired[a+'__'+b]={'rate_change':float(delta.mean()),'paired_sampling_SE':float(delta.std(ddof=1)/np.sqrt(len(delta))),
            'changed_outcomes':int(np.sum(x['outcomes']!=y['outcomes'])),'unit':'cm^3 molecule^-1 s^-1'}
    data['paired_checks']=paired
    data['limitations']=[
        'Rates are incoming-flux RPMD baselines on a historical PES, not selective-quantum corrected rates or experimental predictions.',
        'No full all-J quantum thermal rate. J=0 discrete-lead unitarity is insufficient and basis sensitivity is substantial.',
        'All three Cl+HBr pilot windows request all 3 internal directions under the conservative local selector; no low-dimensional claim follows.',
        'Generic arbitrary-atom Cartesian API is software-tested, not a real polyatomic quantum-rate benchmark.',
        'Sampling SE excludes preparation, bead, integrator, PES, impact cutoff, launch and quantum-transfer errors.',
        'Finite impact-tail bounds are unavailable. Any zero observed outer-annulus reactions does not prove a zero tail.',
        'Curvilinear reduced quantum kinetic operators and general residual-bath quantum feedback are NOT implemented.'
    ]
    with (out/'rates.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['run','T_K','P','integrator','trajectories','bmax_bohr','rate_lower_cm3_molecule_s','rate_upper_cm3_molecule_s','sampling_SE','reactive','unfinished','invalid','physical_converged']);wr.writerows(rows)
    (out/'validation_summary.json').write_text(json.dumps(data,indent=2,ensure_ascii=False))
    # Relative artifact hashes permit exact replay/audit without machine-specific paths.
    paths=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in str(p) and '.pytest_cache' not in str(p)
           and p.suffix in ('.py','.json','.npz','.csv') and p.name!='manifest.json']
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    manifest={'base_commit':data['base_commit'],'python':platform.python_version(),'numpy':np.__version__,
              'sha256':hashes,'note':'Raw arrays are in the downloadable archive, not committed to GitHub.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'rates':len(rows),'selection':{k:v['selection']['selected_dimension'] for k,v in data['selection'].items()},
                      'paired_checks':paired,'quantum_refinement':data['quantum_refinement_comparison']},indent=2))
    return data

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default=str(ROOT));parser.add_argument('--bootstrap',type=int,default=64)
    args=parser.parse_args();build(args.root,args.bootstrap)
