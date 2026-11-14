"""Independent restrained PIMD pilot and local molecular active-space analysis."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import sys,json,time
from pathlib import Path
import numpy as np,numba
from numba import njit,prange
from leps import MASSES,MB,MH,MC,MU_BOND,MU_IN,C_IN,KB,radial_one
from dynamics import jacobi_force,jacobi_energy,haar_rotate
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from pyrpmd.analysis.bimolecular_active_space import select_molecular_subspace

@njit(cache=True)
def bias(q,f):
    P=len(q);r=np.zeros(3);R=np.zeros(3)
    for j in range(P):r+=q[j,0]/P;R+=q[j,1]/P
    b=R+(1-C_IN)*r;c=R-C_IN*r
    d0=np.sqrt(np.sum(r*r));d1=np.sqrt(np.sum(c*c));d2=np.sqrt(np.sum(b*b))
    s=d0-d1;ds=s-(2.7204565-3.6420790);dd=d2-6.3625355
    U=.5*.05*dd*dd+.5*.02*ds*ds
    for j in range(P):
        f[j,0]-=.05*dd*(1-C_IN)*b/d2+.02*ds*(r/d0+C_IN*c/d1)
        f[j,1]-=.05*dd*b/d2-.02*ds*c/d1
    return P*U

@njit(cache=True,parallel=True)
def pilot(n,P,beta,cycles,seed):
    frames=np.empty((n,2,3));stats=np.zeros(n);wp2=(P/beta)**2;dt=4.
    for i in prange(n):
        np.random.seed(seed+997*i);q=np.zeros((P,2,3));q[:,0,2]=2.7204565;q[:,1,2]=3.642079+C_IN*2.7204565
        f=np.empty_like(q);accepted=0
        for cycle in range(cycles):
            v=np.empty_like(q)
            v[:,0]=np.random.normal(0.,np.sqrt(P*MU_BOND/beta),(P,3));v[:,1]=np.random.normal(0.,np.sqrt(P*MU_IN/beta),(P,3))
            x=q.copy();jacobi_force(x,f,wp2);e0=P*jacobi_energy(q,v,wp2)+bias(x,f);valid=True
            for st in range(20):
                v+=.5*dt*f;x[:,0]+=dt*v[:,0]/MU_BOND;x[:,1]+=dt*v[:,1]/MU_IN
                valid=jacobi_force(x,f,wp2)
                if not valid:break
                U=bias(x,f);v+=.5*dt*f
            if valid and np.log(np.random.random())<-beta/P*(P*jacobi_energy(x,v,wp2)+U-e0):q=x;accepted+=1
        for k in range(2):
            frames[i,k]=0
            for j in range(P):frames[i,k]+=q[j,k]/P
        stats[i]=accepted/cycles
    return frames,stats

def cartesian_force(x):
    x=np.asarray(x);pairs=[(0,1),(1,2),(0,2)]
    d=np.array([np.linalg.norm(x[a]-x[b]) for a,b in pairs]);E,g=radial_one(d);F=np.zeros_like(x)
    for i,(a,b) in enumerate(pairs):
        force=-g[i]*(x[a]-x[b])/d[i];F[a]+=force;F[b]-=force
    return E,F

def convert(j):
    out=np.zeros((len(j),3,3));out[:,0]=-MH/(MB+MH)*j[:,0];out[:,1]=MB/(MB+MH)*j[:,0];out[:,2]=j[:,1]
    out-=(out*MASSES[None,:,None]).sum(1,keepdims=True)/MASSES.sum();return out

def run(T,P=32,n=256,cycles=800,seed=705001):
    numba.set_num_threads(4);start=time.perf_counter();j,acc=pilot(n,P,1/(KB*T),cycles,seed);x=convert(j)
    f=np.empty_like(x);H=np.empty((n,9,9));h=2e-4
    for i,q in enumerate(x):
        _,f[i]=cartesian_force(q)
        for k in range(9):
            perturb=np.eye(9)[k].reshape(3,3)*h
            H[i,:,k]=-(cartesian_force(q+perturb)[1]-cartesian_force(q-perturb)[1]).ravel()/(2*h)
    antisym=float(np.max(abs(H-H.swapaxes(1,2))));H=.5*(H+H.swapaxes(1,2))
    # Medoid in pair distances, without using outcomes or a quantum answer.
    d=np.column_stack([np.linalg.norm(x[:,a]-x[:,b],axis=1) for a,b in [(0,1),(1,2),(0,2)]])
    index=np.argmin(np.sum((d-np.median(d,axis=0))**2,axis=1));ref=x[index]
    a=np.zeros((3,3));u=(ref[1]-ref[0])/np.linalg.norm(ref[1]-ref[0]);v=(ref[1]-ref[2])/np.linalg.norm(ref[1]-ref[2]);a[0]=-u;a[1]=u-v;a[2]=v
    result=select_molecular_subspace(x,f,MASSES,reference=ref,hessians=H,beta=1/(KB*T),observable_gradient=a,max_dimension=3)
    out={'temperature_K':T,'beads':P,'pilot_frames':n,'hmc_cycles':cycles,'seed':seed,'mean_acceptance':float(acc.mean()),
         'hessian_asymmetry_max':antisym,'reference_distances':d[index].tolist(),'elapsed_seconds':time.perf_counter()-start,
         'selection':result.as_dict(),'physical_converged':False}
    where=ROOT/'results';np.savez_compressed(where/f'pilot_T{T:g}.npz',positions=x,forces=f,hessians=H,masses=MASSES,reference=ref,observable_gradient=a)
    (where/f'selection_T{T:g}.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True);return out
if __name__=='__main__':
    for T in [150,300,600]:run(T,seed=705001+int(T))
