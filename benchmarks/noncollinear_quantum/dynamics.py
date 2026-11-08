"""Noncollinear Cartesian-Jacobi incoming-flux RPMD on BrHCl_LEPS1.

Two Jacobi vectors, physical masses, unrestricted bending/rotation. The source
is canonical bound HBr plus thermal incoming-flux samples at a finite plane.
Not asserted to be a converged equilibrium Bennett--Chandler rate calculation.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
import sys,json,time,argparse
from pathlib import Path
import numpy as np
import numba
from numba import njit,prange
from scipy.linalg import eigh
from cayley import setup as cayley_setup, free_step as cayley_free
from leps import D,RE,ALPHA,MU_BOND,MU_IN,MB,MH,MC,C_IN,KB,radial_one
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from pyrpmd.analysis.bimolecular_scattering import sample_collision_flux, thermal_impact_rate

@njit(cache=True)
def hbr_force(q,f,wp2):
    P=len(q)
    for j in range(P):
        r=np.sqrt(np.sum(q[j]**2))
        if r<.5 or r>7:return False
        x=np.exp(-ALPHA[0]*(r-RE[0]));dv=2*D[0]*ALPHA[0]*x*(1-x)
        for a in range(3):f[j,a]=-dv*q[j,a]/r-MU_BOND*wp2*(2*q[j,a]-q[(j-1)%P,a]-q[(j+1)%P,a])
    return True

@njit(cache=True)
def hbr_energy(q,p,wp2):
    E=0.
    for j in range(len(q)):
        r=np.sqrt(np.sum(q[j]**2));x=np.exp(-ALPHA[0]*(r-RE[0]));E+=D[0]*(1-x)**2
        for a in range(3):E+=p[j,a]**2/(2*MU_BOND)+.5*MU_BOND*wp2*(q[j,a]-q[(j-1)%len(q),a])**2
    return E

@njit(cache=True)
def haar_rotate(q):
    a=np.random.normal(0.,1.,4);a/=np.sqrt(np.sum(a*a));w,x,y,z=a
    R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    return q@R.T

@njit(cache=True,parallel=True)
def sample_hbr(n,P,beta,cycles,leapfrog_steps,dt,seed):
    qout=np.empty((n,P,3));pout=np.empty_like(qout);stats=np.empty((n,5));wp2=(P/beta)**2;bp=beta/P
    for i in prange(n):
        np.random.seed(seed+i*97);q=np.zeros((P,3));q[:,2]=RE[0];q=haar_rotate(q);f=np.empty_like(q);accepted=0
        for cycle in range(cycles):
            proposal=q.copy();p=np.random.normal(0.,np.sqrt(MU_BOND/bp),(P,3));e0=hbr_energy(q,p,wp2)
            good=hbr_force(proposal,f,wp2)
            for step in range(leapfrog_steps):
                p+=.5*dt*f;proposal+=dt*p/MU_BOND;good=hbr_force(proposal,f,wp2)
                if not good:break
                p+=.5*dt*f
            if good and np.log(np.random.random()) < -bp*(hbr_energy(proposal,p,wp2)-e0):q=proposal;accepted+=1
            q=haar_rotate(q)
        qout[i]=q;pout[i]=np.random.normal(0.,np.sqrt(MU_BOND/bp),(P,3));centroid=np.zeros(3)
        for j in range(P):centroid+=q[j]/P
        vr=0.;potential=0.;radius=0.;r2mean=0.
        for j in range(P):
            r=np.sqrt(np.sum(q[j]**2));x=np.exp(-ALPHA[0]*(r-RE[0]));dv=2*D[0]*ALPHA[0]*x*(1-x)
            potential+=D[0]*(1-x)**2/P;radius+=r/P;r2mean+=r*r/P
            for a in range(3):vr+=.5*(q[j,a]-centroid[a])*dv*q[j,a]/r/P
        stats[i,0]=accepted/cycles;stats[i,1]=radius;stats[i,2]=potential;stats[i,3]=potential+vr+3/(2*beta);stats[i,4]=r2mean
    return qout,pout,stats

def free_internal_paths(n,P,beta,mass,rng):
    freq=2*P/beta*np.sin(np.pi*np.arange(P//2+1)/P);freq[0]=1
    c=np.fft.rfft(rng.normal(size=(n,P,3)),axis=1,norm='ortho');c*=np.sqrt(P/(beta*mass))/freq[None,:,None];c[:,0,:]=0
    return np.fft.irfft(c,n=P,axis=1,norm='ortho')

@njit(cache=True)
def jacobi_force(q,f,wp2):
    P=q.shape[0]
    for j in range(P):
        r=q[j,0];R=q[j,1];hcl=R-C_IN*r;brcl=R+(1-C_IN)*r
        d=np.array([np.sqrt(np.sum(r*r)),np.sqrt(np.sum(hcl*hcl)),np.sqrt(np.sum(brcl*brcl))])
        if d.min()<.2:return False
        e,g=radial_one(d)
        if not np.isfinite(e) or not np.isfinite(g).all():return False
        for a in range(3):
            f[j,0,a]=-g[0]*r[a]/d[0]+C_IN*g[1]*hcl[a]/d[1]-(1-C_IN)*g[2]*brcl[a]/d[2]
            f[j,1,a]=-g[1]*hcl[a]/d[1]-g[2]*brcl[a]/d[2]
            f[j,0,a]-=MU_BOND*wp2*(2*q[j,0,a]-q[(j-1)%P,0,a]-q[(j+1)%P,0,a])
            f[j,1,a]-=MU_IN*wp2*(2*q[j,1,a]-q[(j-1)%P,1,a]-q[(j+1)%P,1,a])
    return True

@njit(cache=True)
def jacobi_energy(q,p,wp2):
    E=0.;P=len(q)
    for j in range(P):
        r=q[j,0];R=q[j,1];a=R-C_IN*r;b=R+(1-C_IN)*r
        d=np.array([np.sqrt(np.sum(r*r)),np.sqrt(np.sum(a*a)),np.sqrt(np.sum(b*b))]);e,_=radial_one(d);E+=e
        for k,m in enumerate((MU_BOND,MU_IN)):
            for v in range(3):E+=p[j,k,v]**2/(2*m)+.5*m*wp2*(q[j,k,v]-q[(j-1)%P,k,v])**2
    return E/P

@njit(cache=True)
def angular_momentum(q,p):
    v=np.zeros(3)
    for j in range(len(q)):
        for k in range(2):v+=np.cross(q[j,k],p[j,k])/len(q)
    return v

@njit(cache=True,parallel=True)
def propagate(q0,p0,beta,dt,tmax,Rexit,save_pilot=False,use_cayley=False):
    N,P,_,_=q0.shape;wp2=(P/beta)**2;plan=cayley_setup(P,dt,wp2);force_wp2=0. if use_cayley else wp2
    outcomes=np.full(N,-1.,dtype=np.float64);exitt=np.zeros(N);errors=np.zeros(N);angular_errors=np.zeros(N)
    minsep=np.full(N,1e6);bend=np.zeros(N);pilot=np.zeros((N,3,2,3));pilot_valid=np.zeros(N,dtype=np.int64)
    for i in prange(N):
        q=q0[i].copy();p=p0[i].copy();f=np.empty_like(q)
        if not jacobi_force(q,f,force_wp2):outcomes[i]=-2;continue
        e0=jacobi_energy(q,p,wp2);L0=angular_momentum(q,p)
        for step in range(round(tmax/dt)):
            if use_cayley:
                p+=.5*dt*f;cayley_free(q,p,np.array([MU_BOND,MU_IN]),dt,wp2,plan)
            else:
                for j in range(P):
                    for k,m in enumerate((MU_BOND,MU_IN)):
                        for a in range(3):p[j,k,a]+=.5*dt*f[j,k,a];q[j,k,a]+=dt*p[j,k,a]/m
            if not jacobi_force(q,f,force_wp2):outcomes[i]=-2;break
            for j in range(P):
                for k in range(2):
                    for a in range(3):p[j,k,a]+=.5*dt*f[j,k,a]
            if step%100==0:
                ee=jacobi_energy(q,p,wp2);errors[i]=max(errors[i],abs(ee-e0));angular_errors[i]=max(angular_errors[i],np.sqrt(np.sum((angular_momentum(q,p)-L0)**2)))
                r=np.zeros(3);R=np.zeros(3);vr=np.zeros(3);vR=np.zeros(3)
                for j in range(P):r+=q[j,0]/P;R+=q[j,1]/P;vr+=p[j,0]/(P*MU_BOND);vR+=p[j,1]/(P*MU_IN)
                hcl=R-C_IN*r;brcl=R+(1-C_IN)*r
                d0=np.sqrt(np.sum(r*r));d1=np.sqrt(np.sum(hcl*hcl));d2=np.sqrt(np.sum(brcl*brcl))
                if d2<minsep[i]:
                    minsep[i]=d2;bend[i]=np.arccos(min(1.,max(-1.,-np.dot(r,hcl)/(d0*d1))))
                    if save_pilot:
                        pilot[i,0,0]=r;pilot[i,0,1]=R
                        for k in range(2):
                            pilot[i,1,k]=0
                            for j in range(P):pilot[i,1,k]+=f[j,k]/P
                        pilot[i,2,0]=vr;pilot[i,2,1]=vR;pilot_valid[i]=1
                outward=np.dot(brcl,vR+(1-C_IN)*vr)>0
                if d2>Rexit and outward:
                    outcomes[i]=1. if d1<d0 and d1<5. else 0.;exitt[i]=(step+1)*dt;break
    return outcomes,exitt,errors,angular_errors,minsep,bend,pilot,pilot_valid

def reactant_reference(T,nrad=160,jmax=30,states=12):
    lo=.5;hi=7.;r=lo+(hi-lo)*np.arange(1,nrad+1)/(nrad+1);n=np.arange(1,nrad+1)
    S=np.sqrt(2/(nrad+1))*np.sin(np.pi*np.outer(n,n)/(nrad+1));K=(S*((np.pi*n/(hi-lo))**2/(2*MU_BOND)))@S.T
    V=D[0]*(1-np.exp(-ALPHA[0]*(r-RE[0])))**2;allE=[];allg=[];allr=[];allv=[]
    for j in range(jmax+1):
        E,U=eigh(K+np.diag(V+j*(j+1)/(2*MU_BOND*r*r)),subset_by_index=[0,states-1]);use=E<D[0]
        allE.extend(E[use]);allg.extend(np.full(use.sum(),2*j+1));allr.extend((U[:,use]**2).T@r);allv.extend((U[:,use]**2).T@V)
    E=np.array(allE);g=np.array(allg);beta=1/(KB*T);w=g*np.exp(-beta*(E-E.min()));w/=w.sum()
    return {'mean_bond_bohr':float(w@allr),'mean_potential_Eh':float(w@allv),'internal_energy_Eh':float(w@E),
            'ground_energy_Eh':float(E.min()),'states_count':len(E),'jmax':jmax,'nrad':nrad,
            'energy_levels_Eh':E.tolist(),'degeneracies':g.tolist()}

def run(config,output):
    config=dict(config)
    for key in ('trajectories','beads','hmc_cycles'):
        value=config.get(key,800 if key=='hmc_cycles' else 0)
        if isinstance(value,bool) or int(value)!=value or value<1:raise ValueError('Invalid integer '+key)
    if config.get('integrator','velocity-verlet') not in ('velocity-verlet','cayley-bcb'):raise ValueError('Unknown integrator')
    for key,default in [('temperature_K',0),('dt_au',4),('tmax_au',240000),('launch_R_bohr',14),('exit_radius_bohr',16),('bmax_bohr',6)]:
        if not np.isfinite(config.get(key,default)) or config.get(key,default)<=0:raise ValueError('Invalid '+key)
    out=Path(output);out.mkdir(parents=True,exist_ok=True);numba.set_num_threads(min(4,numba.get_num_threads()))
    n=int(config['trajectories']);P=int(config['beads']);T=float(config['temperature_K']);beta=1/(KB*T);seed=int(config['seed'])
    R0=float(config.get('launch_R_bohr',14.));bmax=float(config.get('bmax_bohr',8.));dt=float(config.get('dt_au',4.))
    if n<2 or P<1 or T<=0 or dt<=0 or R0<=bmax/2:raise ValueError('Invalid run configuration')
    receipt=out/'input.json';text=json.dumps(config,indent=2,sort_keys=True)
    if receipt.exists() and receipt.read_text()!=text:raise ValueError('Output contains different scientific input')
    receipt.write_text(text);tic=time.perf_counter();r,p_r,stats=sample_hbr(n,P,beta,int(config.get('hmc_cycles',800)),20,4.,seed)
    rng=np.random.default_rng(seed+100001);flux=sample_collision_flux(n,T,MU_IN,bmax,seed=seed+200001,proposal_temperature_ratio=config.get('proposal_temperature_ratio',1.))
    R=free_internal_paths(n,P,beta,MU_IN,rng);R[:,:,0]+=flux.impact_parameter[:,None];R[:,:,2]+=R0
    p_R=rng.normal(size=(n,P,3))*np.sqrt(P*MU_IN/beta);p_R-=p_R.mean(axis=1,keepdims=True);p_R[:,:,2]-=np.sqrt(2*MU_IN*flux.energy)[:,None]
    q=np.ascontiguousarray(np.stack([r,R],axis=2));p=np.ascontiguousarray(np.stack([p_r,p_R],axis=2));init_time=time.perf_counter()-tic
    print('prepared',T,P,n,'acceptance',stats[:,0].mean(),'seconds',init_time,flush=True)
    y,exit_t,err,Lerr,sep,bend,pilot,valid=propagate(q,p,beta,dt,float(config.get('tmax_au',240000.)),float(config.get('exit_radius_bohr',16.)),True,config.get('integrator','velocity-verlet')=='cayley-bcb')
    result=thermal_impact_rate(y,flux.importance_weight,temperature=T,reduced_mass=MU_IN,bmax=bmax,impact_parameters=flux.impact_parameter)
    result.update({'preparation_seconds':init_time,'elapsed_seconds':time.perf_counter()-tic,
        'integrator':config.get('integrator','velocity-verlet'),'method':'Cartesian-Jacobi incoming-flux RPMD' if P>1 else 'canonical classical noncollinear trajectories','beads':P,'seed':seed,
        'initial_stats':{'acceptance':float(stats[:,0].mean()),'mean_bond_bohr':float(stats[:,1].mean()),'mean_potential_Eh':float(stats[:,2].mean()),
                         'internal_energy_Eh':float(stats[:,3].mean()),'energy_sampling_se_Eh':float(stats[:,3].std(ddof=1)/np.sqrt(n))},
        'max_energy_drift_Eh':float(err.max()),'median_energy_drift_Eh':float(np.median(err)),
        'max_angular_momentum_drift_au':float(Lerr.max()),
        'reactive_bend_range_degrees':list(map(float,np.rad2deg([bend[y==1].min(),bend[y==1].max()]))) if np.any(y==1) else None,
        'electronic_convention':'one historical LEPS electronic surface; no halogen fine-structure weighting',
        'scope':'3D physical space; all six COM-free Cartesian-Jacobi components propagated; not collinear',
        'convergence':{'beads':False,'timestep':False,'launch_distance':False,'impact_cutoff':False,'sampling':False,'quantum_accuracy':False}})
    np.savez_compressed(out/'trajectories.npz',outcomes=y,energy=flux.energy,b=flux.impact_parameter,weights=flux.importance_weight,exit_time=exit_t,
       energy_drift=err,L_drift=Lerr,min_heavy_distance=sep,angle_at_H_rad=bend,pilot=pilot,pilot_valid=valid,source_stats=stats,source_paths=r[:min(n,512)])
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(json.loads(Path(a.input).read_text()),a.output)
