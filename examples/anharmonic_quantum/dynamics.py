"""Physical-mass unthermostatted RPMD. Linear-response tangent estimator.
Metropolis HMC momenta are never reused for real time production dynamics.
"""
import numpy as np
from numba import njit,prange
from models import force_one

@njit(cache=True)
def tangent_force(q,dq,kind,coupling,wp2,df):
    P,D=q.shape;dep=(8.,6.,5.);aa=(.5,.55,.6)
    for j in range(P):
        x=q[j,0];dx=dq[j,0]
        if kind==0:
            for a in range(D):
                e=np.exp(-aa[a]*q[j,a]);df[j,a]=-2*dep[a]*aa[a]**2*e*(2*e-1)*dq[j,a]
        else:
            y=q[j,1];dy=dq[j,1];df[j,0]=-4*dx-.7*y*dy
            df[j,1]=-(1+.7*x+12*(.35**2/8+.005)*y*y)*dy-.7*y*dx
            if D>2:
                dis=dq[j,2]-coupling*dx;df[j,0]+=1.69*coupling*dis;df[j,2]=-1.69*dis
        for a in range(D):df[j,a]-=wp2*(2*dq[j,a]-dq[(j-1)%P,a]-dq[(j+1)%P,a])

@njit(cache=True,parallel=True)
def response_blocks(q0,p0,kind,coupling,beta,dt,steps,stride,nblocks=32):
    N,P,D=q0.shape;B=N//nblocks;nt=steps//stride+1;nobs=D if kind==0 else 1
    out=np.zeros((nblocks,nt,nobs));wp2=(P/beta)**2
    for ib in prange(nblocks):
        for i in range(ib*B,(ib+1)*B):
            for sign in range(2):
                q=q0[i].copy();p=p0[i].copy()
                if sign:p=-p
                dq=np.zeros_like(q);dp=np.zeros_like(q)
                for j in range(P):
                    if kind==0:
                        for a in range(D):dp[j,a]=1.
                    else:dp[j,0]=1.
                f=np.empty_like(q);df=np.empty_like(q)
                force_one(q,kind,coupling,wp2,f);tangent_force(q,dq,kind,coupling,wp2,df);it=1
                for st in range(1,steps+1):
                    for j in range(P):
                        for a in range(D):
                            p[j,a]+=.5*dt*f[j,a];q[j,a]+=dt*p[j,a]
                            dp[j,a]+=.5*dt*df[j,a];dq[j,a]+=dt*dp[j,a]
                    force_one(q,kind,coupling,wp2,f);tangent_force(q,dq,kind,coupling,wp2,df)
                    for j in range(P):
                        for a in range(D):p[j,a]+=.5*dt*f[j,a];dp[j,a]+=.5*dt*df[j,a]
                    if st%stride==0:
                        for a in range(nobs):out[ib,it,a]+=np.sum(dq[:,a])/(P*2*B)
                        it+=1
    return out

def transform(response,t,w,eta):
    r=np.asarray(response);wt=np.empty_like(t);wt[0]=(t[1]-t[0])/2;wt[-1]=(t[-1]-t[-2])/2;wt[1:-1]=(t[2:]-t[:-2])/2
    return np.einsum('bto,tw->bwo',r,np.exp(-np.outer(t,eta-1j*w))*wt[:,None],optimize=True)

@njit(cache=True,parallel=True)
def flux_side(q0,p0,beta,dt,steps,stride,nblocks=32):
    N,P,D=q0.shape;B=N//nblocks;out=np.zeros((nblocks,steps//stride+1));wp2=(P/beta)**2
    for ib in prange(nblocks):
        for i in range(ib*B,(ib+1)*B):
            for sign in range(2):
                q=q0[i].copy();p=p0[i].copy()
                if sign:p=-p
                f=np.empty_like(q);force_one(q,2,.15,wp2,f);it=1
                for st in range(1,steps+1):
                    for j in range(P):
                        for a in range(D):p[j,a]+=.5*dt*f[j,a];q[j,a]+=dt*p[j,a]
                    force_one(q,2,.15,wp2,f)
                    for j in range(P):
                        for a in range(D):p[j,a]+=.5*dt*f[j,a]
                    if st%stride==0:
                        h=1. if np.sum(q[:,0])>0 else 0.
                        out[ib,it]+=(1 if sign==0 else -1)*h/B;it+=1
    return out
