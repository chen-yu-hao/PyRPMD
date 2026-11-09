"""Canonical finite-P sampling. HMC is Metropolis corrected and independently seeded.
Fermi sampler is independent exact rejection; HMC output is labelled MCMC.
"""
import numpy as np
from numba import njit,prange
from models import force_one,energy_one

@njit(cache=True)
def project(p):
    P=p.shape[0];v=np.sum(p[:,0])/P
    for j in range(P):p[j,0]-=v

@njit(cache=True,parallel=True)
def hmc(q0,kind,coupling,beta,dt,steps,warmup,draws,thin,seed,constrain=False):
    N,P,D=q0.shape;out=np.empty((N,draws,P,D));accept=np.empty(N);wp2=(P/beta)**2
    for i in prange(N):
        np.random.seed(seed+i*7919);q=q0[i].copy();nacc=0;f=np.empty_like(q)
        for iteration in range(warmup+draws*thin):
            p=np.random.normal(0,np.sqrt(P/beta),(P,D))
            if constrain:project(p)
            E=energy_one(q,p,kind,coupling,wp2);new=q.copy();force_one(new,kind,coupling,wp2,f)
            for st in range(steps):
                if constrain:project(f)
                p+=.5*dt*f;new+=dt*p
                force_one(new,kind,coupling,wp2,f)
                if constrain:project(f)
                p+=.5*dt*f
            finite=np.isfinite(new).all()
            inside=True
            if kind==0:inside=np.min(new)>-3 and np.max(new)<10
            enew=energy_one(new,p,kind,coupling,wp2) if finite and inside else np.inf
            if np.log(np.random.random())<-beta/P*(enew-E):q=new;nacc+=1
            if iteration>=warmup and (iteration-warmup)%thin==0:out[i,(iteration-warmup)//thin]=q
        accept[i]=nacc/(warmup+draws*thin)
    return out,accept

def hmc_samples(model,P=32,nchains=256,warmup=600,draws=8,thin=10,seed=100,centroid=None):
    rng=np.random.default_rng(seed);q=np.zeros((nchains,P,model.dim))
    if centroid is not None:
        q[...,0]=centroid
        if model.dim>1:q[...,1]=.65*np.tanh(2*centroid)
        if model.dim>2:q[...,2]=.1*q[...,1]
    else:q+=rng.normal(scale=.05,size=q.shape)
    out,acc=hmc(q,model.code,model.coupling,model.beta,.02,25,warmup,draws,thin,seed,centroid is not None)
    return out,{'sampler':'Metropolis HMC','nchains':nchains,'draws_per_chain':draws,'warmup':warmup,'thin':thin,'acceptance':float(acc.mean()),'seed':seed,'independent_samples':False,'finite_bead_equilibrium_converged':False}

def fermi_samples(model,P,n,seed):
    rng=np.random.default_rng(seed);beta=model.beta;D=model.dim
    if D not in (2,3):raise ValueError('Fermi core requires x,y')
    wn=2*P/beta*np.sin(np.pi*np.arange(P//2+1)/P);bp=beta/P
    g=.35;c4=g*g/8+.005;w=1.3;c=model.coupling
    K=np.array([[4+w*w*c*c,-w*w*c],[-w*w*c,w*w]]) if D==3 else np.array([[4.]])
    inv=np.linalg.inv(wn[:,None,None]**2*np.eye(len(K))[None]+K);ev,R=np.linalg.eigh(K)
    scale=np.sqrt(1/bp/(wn[:,None]**2+ev[None]));parts=[];total=drawn=accepted=0
    while total<n:
        batch=max(512,min(8192,2*(n-total)));white=rng.normal(size=(batch,P))
        y=np.fft.irfft(np.fft.rfft(white,axis=1,norm='ortho')/np.sqrt(bp*(wn**2+1)),n=P,axis=1,norm='ortho')
        yy=y*y;src=np.fft.rfft(yy,axis=1,norm='ortho');mx=np.fft.irfft(-g*src*inv[None,:,0,0],n=P,axis=1,norm='ortho')
        action=bp*(c4*np.sum(yy*yy,axis=1)+.5*g*np.sum(yy*mx,axis=1))
        if action.min()<-1e-9:raise RuntimeError('Invalid rejection envelope')
        good=np.log(rng.random(batch))<-action;accepted+=int(good.sum());drawn+=batch
        take=min(int(good.sum()),n-total)
        if take:
            Y=y[good][:take];S=src[good][:take];noise=rng.normal(size=(take,P,len(K)))
            xb=np.fft.irfft(np.fft.rfft(noise,axis=1,norm='ortho')*scale[None],n=P,axis=1,norm='ortho')@R.T
            xb+=np.fft.irfft(-g*S[:,:,None]*inv[None,:,:,0],n=P,axis=1,norm='ortho')
            q=np.empty((take,P,D));q[...,0]=xb[...,0];q[...,1]=Y
            if D==3:q[...,2]=xb[...,1]
            parts.append(q);total+=take
    q=np.ascontiguousarray(np.concatenate(parts));p=rng.normal(size=q.shape)*np.sqrt(P/beta)
    return q,p,{'sampler':'independent conditional rejection','acceptance':accepted/drawn,'seed':seed,'independent_samples':True}
