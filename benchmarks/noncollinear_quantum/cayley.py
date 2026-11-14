"""Physical-mass Cayley map for cyclic harmonic ring springs (no thermostat)."""
import numpy as np
from numba import njit

@njit(cache=True)
def setup(P, dt, wp2):
    a=.25*dt*dt*wp2; b=1+2*a
    inv=np.zeros(P);cp=np.zeros(P);z=np.zeros(P)
    if P>=3:
        diag=np.full(P,b);diag[0]=2*b;diag[-1]=b+a*a/b
        inv[0]=1/diag[0];cp[0]=-a*inv[0]
        for j in range(1,P):
            inv[j]=1/(diag[j]+a*cp[j-1]);cp[j]=-a*inv[j]
        u=np.zeros(P);u[0]=-b;u[-1]=-a
        z=tri_solve(u,a,inv,cp)
    return a,b,inv,cp,z

@njit(cache=True)
def tri_solve(rhs,a,inv,cp):
    n=len(rhs);x=np.empty(n);x[0]=rhs[0]*inv[0]
    for j in range(1,n):x[j]=(rhs[j]+a*x[j-1])*inv[j]
    for j in range(n-2,-1,-1):x[j]-=cp[j]*x[j+1]
    return x

@njit(cache=True)
def free_step(q,p,masses,dt,wp2,plan):
    P=q.shape[0];a,b,inv,cp,z=plan
    for k in range(len(masses)):
        mass=masses[k]
        for c in range(3):
            old=q[:,k,c].copy();rhs=np.empty(P)
            for j in range(P):rhs[j]=old[j]-a*(2*old[j]-old[(j-1)%P]-old[(j+1)%P])+dt*p[j,k,c]/mass
            if P==1:new=rhs
            elif P==2:
                new=np.empty(2);den=b*b-4*a*a
                new[0]=(b*rhs[0]+2*a*rhs[1])/den;new[1]=(2*a*rhs[0]+b*rhs[1])/den
            else:
                new=tri_solve(rhs,a,inv,cp);factor=(new[0]+a*new[-1]/b)/(1+z[0]+a*z[-1]/b)
                new-=factor*z
            summed=old+new
            for j in range(P):
                q[j,k,c]=new[j]
                p[j,k,c]-=.5*dt*mass*wp2*(2*summed[j]-summed[(j-1)%P]-summed[(j+1)%P])
