"""Requested anharmonic benchmarks. Dimensionless hbar=m=kB=1.

Morse: three separable Morse stretches, with a declared finite box for sampling.
Fermi: stable xy^2 core and a linearly coupled harmonic spectator.
Eckart: open barrier, tanh-displaced strong bath, and weak x-coupled spectator.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit

@dataclass(frozen=True)
class Model:
    kind: str
    dim: int = 3
    beta: float = 6.0
    coupling: float = .15
    def __post_init__(self):
        if self.kind not in ('morse','fermi','eckart') or self.dim not in (1,2,3) or self.beta<=0:
            raise ValueError('Invalid model kind, dimension, or inverse temperature')
    @property
    def code(self): return {'morse':0,'fermi':1,'eckart':2}[self.kind]
    def force(self,z): return vector_force(np.asarray(z,float),self.code,self.coupling)
    def potential(self,z): return vector_potential(np.asarray(z,float),self.code,self.coupling)
    def hessian(self,z):
        z=np.asarray(z,float);H=np.empty(z.shape+(z.shape[-1],));eps=1e-5
        for j in range(z.shape[-1]):
            dx=np.zeros(z.shape[-1]);dx[j]=eps
            H[..., :,j]=-(self.force(z+dx)-self.force(z-dx))/(2*eps)
        return .5*(H+H.swapaxes(-1,-2))

def vector_potential(z,kind,coupling):
    D=z.shape[-1];q=z[...,0]
    if kind==0:
        dep=np.array([8.,6.,5.])[:D];a=np.array([.5,.55,.6])[:D]
        return np.sum(dep*np.expm1(-a*z)**2,axis=-1)
    if kind==1:
        v=2*q*q
        if D>1:
            y=z[...,1];v=v+.5*y*y+.35*q*y*y+(.35**2/8+.005)*y**4
        if D>2:v=v+.5*1.3**2*(z[...,2]-coupling*q)**2
        return v
    v=1/np.cosh(q)**2
    if D>1:
        x=z[...,1];v=v+.5*1.5**2*(x-.65*np.tanh(2*q))**2
    if D>2:v=v+.5*2.3**2*(z[...,2]-.10*z[...,1])**2
    return v

def vector_force(z,kind,coupling):
    D=z.shape[-1];q=z[...,0];f=np.zeros_like(z)
    if kind==0:
        dep=np.array([8.,6.,5.])[:D];a=np.array([.5,.55,.6])[:D];ex=np.exp(-a*z)
        return 2*dep*a*ex*(ex-1)
    if kind==1:
        f[...,0]=-4*q
        if D>1:
            y=z[...,1];f[...,0]-=.35*y*y;f[...,1]=-y-.7*q*y-4*(.35**2/8+.005)*y**3
        if D>2:
            dis=z[...,2]-coupling*q;f[...,0]+=1.3**2*coupling*dis;f[...,2]=-1.3**2*dis
        return f
    th=np.tanh(q);f[...,0]=2*th*(1-th*th)
    if D>1:
        t=np.tanh(2*q);dis=z[...,1]-.65*t
        f[...,0]+=1.5**2*dis*1.3*(1-t*t);f[...,1]=-1.5**2*dis
    if D>2:
        dis=z[...,2]-.10*z[...,1];f[...,1]+=2.3**2*.10*dis;f[...,2]=-2.3**2*dis
    return f

@njit(cache=True)
def force_one(q,kind,coupling,wp2,f):
    P,D=q.shape
    dep=(8.,6.,5.);aa=(.5,.55,.6)
    for j in range(P):
        x=q[j,0]
        if kind==0:
            for a in range(D):
                e=np.exp(-aa[a]*x) if a==0 else np.exp(-aa[a]*q[j,a])
                f[j,a]=2*dep[a]*aa[a]*e*(e-1)
        elif kind==1:
            f[j,0]=-4*x
            if D>1:
                y=q[j,1];f[j,0]-=.35*y*y;f[j,1]=-y-.7*x*y-4*(.35**2/8+.005)*y*y*y
            if D>2:
                d=q[j,2]-coupling*x;f[j,0]+=1.69*coupling*d;f[j,2]=-1.69*d
        else:
            t=np.tanh(x);f[j,0]=2*t*(1-t*t)
            if D>1:
                s=np.tanh(2*x);d=q[j,1]-.65*s
                f[j,0]+=2.25*d*1.3*(1-s*s);f[j,1]=-2.25*d
            if D>2:
                d=q[j,2]-.1*q[j,1];f[j,1]+=.529*d;f[j,2]=-5.29*d
        for a in range(D):f[j,a]-=wp2*(2*q[j,a]-q[(j-1)%P,a]-q[(j+1)%P,a])

@njit(cache=True)
def energy_one(q,p,kind,coupling,wp2):
    P,D=q.shape;e=0.;dep=(8.,6.,5.);aa=(.5,.55,.6)
    for j in range(P):
        x=q[j,0]
        if kind==0:
            for a in range(D):e+=dep[a]*np.expm1(-aa[a]*q[j,a])**2
        elif kind==1:
            e+=2*x*x
            if D>1:
                y=q[j,1];e+=.5*y*y+.35*x*y*y+(.35**2/8+.005)*y**4
            if D>2:e+=.845*(q[j,2]-coupling*x)**2
        else:
            t=np.tanh(x);e+=1-t*t
            if D>1:e+=1.125*(q[j,1]-.65*np.tanh(2*x))**2
            if D>2:e+=2.645*(q[j,2]-.1*q[j,1])**2
        for a in range(D):e+=.5*p[j,a]**2+.5*wp2*(q[j,a]-q[(j+1)%P,a])**2
    return e
