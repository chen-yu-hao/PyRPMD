"""Fresh noncollinear restrained-PIMD pilots; compare a nonlinear target chart.

Run in PR137 checkout. --upstream-dir supports the extracted audit helpers in
this downloadable archive. This calculates no reaction rates and makes no
claim of stationary-ensemble or active-space convergence.
"""
from pathlib import Path
import sys,argparse,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from pyrpmd.analysis.goal_quantum_coordinates import screen_chart_subspace
from pyrpmd.analysis.curvilinear_quantum import triatomic_transfer_geometry

def main(upstream_dir,out,n=256,cycles=800,temperatures=(150,300,600)):
    sys.path.insert(0,str(upstream_dir))
    import numba
    from pilot_selection import pilot
    from leps import MASSES,C_IN,KB,radial_one
    numba.set_num_threads(4);out.mkdir(exist_ok=True,parents=True);record_file=out/'molecular_selection.json';records=json.loads(record_file.read_text()) if record_file.exists() else {}
    for T in temperatures:
        start=time.perf_counter();seed=900001+T
        j,acc=pilot(n,32,1/(KB*T),cycles,seed)
        r=j[:,0];R=j[:,1];hcl=R-C_IN*r;brcl=R+(1-C_IN)*r
        r1=np.linalg.norm(r,axis=1);r2=np.linalg.norm(hcl,axis=1);r3=np.linalg.norm(brcl,axis=1)
        theta=np.arccos(np.clip(-np.sum(r*hcl,axis=1)/(r1*r2),-1,1))
        u=np.column_stack([np.log(r1/r2),np.log(r1+r2),theta]);fu=np.empty_like(u)
        for k in range(n):
            _,g=radial_one(np.array([r1[k],r2[k],r3[k]]));s=np.sin(theta[k]);c=np.cos(theta[k])
            h=np.array([g[0]+g[2]*(r1[k]-r2[k]*c)/r3[k],g[1]+g[2]*(r2[k]-r1[k]*c)/r3[k],g[2]*r1[k]*r2[k]*s/r3[k]])
            v=r1[k]*r2[k]/(r1[k]+r2[k]);J=np.array([[v,r1[k],0],[-v,r2[k],0],[0,0,1.]])
            fu[k]=-J.T@h
        geometry=lambda x:triatomic_transfer_geometry(x,MASSES)
        result=screen_chart_subspace(u,fu,geometry,[1,0,0],np.arange(n),max_dimension=3)
        result['run']={'T_K':T,'beads':32,'independent_chains':n,'hmc_cycles':cycles,'seed':seed,'acceptance_mean':float(acc.mean()),
            'angle_degrees_range':np.rad2deg([theta.min(),theta.max()]).tolist(),'elapsed_seconds':time.perf_counter()-start,
            'pilot_distribution':'restrained PIMD near TS; centroid geometries/physical PES forces','equilibrium_converged':False,
            'target':'log(r_BrH/r_HCl) response; not a full flux-side rate'}
        records[str(T)]=result
        np.savez_compressed(out/f'nonlinear_pilot_T{T}.npz',u=u,generalized_forces=fu,jacobi=j,acceptance=acc)
        print(T,result['required_dimension'],result['status'],result['diagnostics']['holdout_relative_force_error'],flush=True)
        (out/'molecular_selection.json').write_text(json.dumps(records,indent=2))
    return records
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--upstream-dir',type=Path,default=ROOT/'benchmarks/noncollinear_quantum');ap.add_argument('--output',type=Path,default=ROOT/'results');ap.add_argument('--n',type=int,default=256);ap.add_argument('--cycles',type=int,default=800);ap.add_argument('--temperatures',type=int,nargs='+',default=[150,300,600]);a=ap.parse_args();main(a.upstream_dir,a.output,a.n,a.cycles,a.temperatures)
