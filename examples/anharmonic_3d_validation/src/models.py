"""Three genuinely anharmonic, three-coordinate benchmark potentials.
Dimensionless hbar=m_x=m_y=m_z=kB=1.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Morse3D:
    D: tuple=(8.,7.,6.)
    a: tuple=(.65,.70,.75)
    gamma: tuple=(.12,.08,.06)
    beta:float=6.
    name:str='morse_zpe'
    def V(self,q):
        q=np.asarray(q);u=1-np.exp(-np.asarray(self.a)*q)
        return np.sum(np.asarray(self.D)*u*u,axis=-1)+self.gamma[0]*u[...,0]*u[...,1]+self.gamma[1]*u[...,0]*u[...,2]+self.gamma[2]*u[...,1]*u[...,2]
    def force(self,q):
        q=np.asarray(q); e=np.exp(-np.asarray(self.a)*q); u=1-e
        du=np.asarray(self.a)*e
        grad=2*np.asarray(self.D)*u*du
        grad=grad.copy()
        grad[...,0]+=du[...,0]*(self.gamma[0]*u[...,1]+self.gamma[1]*u[...,2])
        grad[...,1]+=du[...,1]*(self.gamma[0]*u[...,0]+self.gamma[2]*u[...,2])
        grad[...,2]+=du[...,2]*(self.gamma[1]*u[...,0]+self.gamma[2]*u[...,1])
        return -grad
    def harmonic(self):
        a=np.array(self.a);K=np.diag(2*np.array(self.D)*a*a)
        for i,j,g in [(0,1,self.gamma[0]),(0,2,self.gamma[1]),(1,2,self.gamma[2])]:K[i,j]=K[j,i]=g*a[i]*a[j]
        return K

@dataclass(frozen=True)
class Fermi3D:
    wx:float=2.;wy:float=1.;wb:float=1.3;g:float=.35;lam:float=.35;u:float=.005
    beta:float=6.
    name:str='fermi_resonance'
    @property
    def c4(self):return self.g*self.g/(2*self.wx*self.wx)+self.u
    def V(self,q):
        x,y,b=np.moveaxis(np.asarray(q),-1,0)
        return .5*self.wx**2*x*x+.5*self.wy**2*y*y+self.g*x*y*y+self.c4*y**4+.5*self.wb**2*(b-self.lam*x)**2
    def force(self,q):
        x,y,b=np.moveaxis(np.asarray(q),-1,0)
        return np.stack([-self.wx**2*x-self.g*y*y+self.wb**2*self.lam*(b-self.lam*x),-self.wy**2*y-2*self.g*x*y-4*self.c4*y**3,-self.wb**2*(b-self.lam*x)],axis=-1)
    def harmonic(self):
        return np.array([[self.wx**2+self.wb**2*self.lam**2,0,-self.wb**2*self.lam],[0,self.wy**2,0],[-self.wb**2*self.lam,0,self.wb**2]])

@dataclass(frozen=True)
class Eckart3D:
    V0:float=.45;a:float=1.0;wx:float=1.1;lam:float=.8;b:float=4.;wz:float=1.6;eps:float=.12;c4:float=.015
    beta:float=10.
    name:str='curved_eckart'
    def V(self,q):
        R,x,z=np.moveaxis(np.asarray(q),-1,0)
        sech2=1-np.tanh(self.a*R)**2
        y=x-self.lam*np.tanh(self.b*R)
        h=z-self.eps*x
        return self.V0*sech2+.5*self.wx**2*y*y+self.c4*y**4+.5*self.wz**2*h*h
    def force(self,q):
        R,x,z=np.moveaxis(np.asarray(q),-1,0)
        ta=np.tanh(self.a*R);tb=np.tanh(self.b*R)
        y=x-self.lam*tb;h=z-self.eps*x
        gy=self.wx**2*y+4*self.c4*y**3
        FR=2*self.V0*self.a*ta*(1-ta*ta)+gy*self.lam*self.b*(1-tb*tb)
        Fx=-gy+self.wz**2*self.eps*h
        Fz=-self.wz**2*h
        return np.stack([FR,Fx,Fz],axis=-1)
    def classical_mep(self, R):
        # Minimize transverse potential exactly at y=h=0; transition-path curvature is explicit.
        x=self.lam*np.tanh(self.b*np.asarray(R))
        return np.stack([R,x,self.eps*x],axis=-1)

MODELS={'morse':Morse3D(),'fermi':Fermi3D(),'eckart':Eckart3D()}