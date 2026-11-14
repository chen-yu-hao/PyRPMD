"""Energy/gradient reimplementation of the repository's BrHCl_LEPS1 oracle.

Historical semiempirical model; NOT a refitted or modern ab-initio surface.
Units and rounded sqrt(2) exactly follow benchmarks/brhcl/SOURCE.md.
Ordered distances: Br-H, H-Cl, Br-Cl. Masses: 79Br, 1H, 35Cl.
"""
import numpy as np
from numba import njit
D=np.array([90.24,106.40,52.09])/627.5095
RE=np.array([1.414,1.275,2.136])/0.529177106
ALPHA=np.array([1.851,1.867,1.923])*0.529177106
Z=.154
AMU=1822.888486209
MASSES=np.array([78.9183376,1.00782503223,34.968852682])*AMU
MB,MH,MC=MASSES
MU_R=MB*MC/(MB+MC)  # heavy pair
MU_Q=MH*(MB+MC)/(MB+MH+MC)
A=MC/(MB+MC); B=MB/(MB+MC)
SCALE=np.sqrt(MU_Q/MU_R)
MU_BOND=MB*MH/(MB+MH)
MU_IN=MC*(MB+MH)/(MB+MH+MC)
C_IN=MB/(MB+MH)
KB=3.166811563e-6

@njit(cache=True)
def radial_one(r):
    Q=np.empty(3);J=np.empty(3);dQ=np.empty(3);dJ=np.empty(3)
    for i in range(3):
        x=np.exp(-ALPHA[i]*(r[i]-RE[i]));c=D[i]/(4*(1+Z))
        Q[i]=c*((Z+3)*x*x-2*(1+3*Z)*x)
        J[i]=c*((1+3*Z)*x*x-2*(Z+3)*x)
        dQ[i]=-ALPHA[i]*c*(2*(Z+3)*x*x-2*(1+3*Z)*x)
        dJ[i]=-ALPHA[i]*c*(2*(1+3*Z)*x*x-2*(Z+3)*x)
    rad=np.sqrt((J[0]-J[1])**2+(J[1]-J[2])**2+(J[2]-J[0])**2)
    e=Q.sum()-rad/1.41421356+D[1]
    grad=np.empty(3)
    for i in range(3):
        grad[i]=dQ[i]-(3*J[i]-J.sum())*dJ[i]/(1.41421356*rad)
    return e,grad

def energy(distances):
    r=np.asarray(distances,float);x=np.exp(-ALPHA*(r-RE));c=D/(4*(1+Z))
    q=c*((Z+3)*x*x-2*(1+3*Z)*x);j=c*((1+3*Z)*x*x-2*(Z+3)*x)
    return q.sum(-1)-np.sqrt(np.sum((j-np.roll(j,1,axis=-1))**2,axis=-1))/1.41421356+D[1]

