"""Independent three-coordinate quantum spectral reference solvers.
Units hbar=mass=1. Sinc DVR is used for Morse; sparse Fock for Fermi.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator,eigsh
from scipy.linalg import eigh
from models import Morse3D,Fermi3D

def sinc_kinetic(n,dx):
    j=np.arange(n);delta=j[:,None]-j[None,:]
    T=np.empty((n,n));mask=delta!=0
    T[~mask]=np.pi**2/(6*dx*dx)
    T[mask]=(-1.)**np.abs(delta[mask])/(dx*dx*delta[mask]**2)
    return T

def morse_spectrum(m:Morse3D,n=(19,19,19),bounds=((-2.5,4.0),(-2.5,4.0),(-2.5,4.0)),nstate=24):
    grids=[np.linspace(a,b,v) for (a,b),v in zip(bounds,n)]
    q=np.stack(np.meshgrid(*grids,indexing='ij'),axis=-1)
    V=np.asarray(m.V(q));Ts=[sinc_kinetic(v,grid[1]-grid[0]) for v,grid in zip(n,grids)]
    def matvec(vec):
        p=np.asarray(vec).reshape(n)
        out=V*p
        out=out+np.einsum('ia,ajk->ijk',Ts[0],p,optimize=True)
        out=out+np.einsum('jb,ibk->ijk',Ts[1],p,optimize=True)
        out=out+np.einsum('kc,ijc->ijk',Ts[2],p,optimize=True)
        return np.ascontiguousarray(out).ravel()
    op=LinearOperator((V.size,V.size),matvec=matvec,dtype=float)
    vals,vecs=eigsh(op,k=nstate,which='SA',tol=5e-10,v0=np.ones(V.size),maxiter=12000)
    perm=np.argsort(vals);vals=vals[perm];vecs=vecs[:,perm]
    # Grid vectors are l2-orthonormal; measure factor cancels in expectation.
    A=(vecs.T@(q[...,0].ravel()[:,None]*vecs))
    return vals,vecs,A,{'grid':list(n),'bounds':[list(b) for b in bounds], 'Vmin':float(np.min(V)), 'ground_energy':float(vals[0]),'first_excitations':(vals[1:8]-vals[0]).tolist()}

def fock_ops(n,frequency):
    a=np.zeros((n+5,n+5));v=np.sqrt(np.arange(1,n+5)/(2*frequency))
    a[np.arange(n+4),np.arange(1,n+5)]=v;a[np.arange(1,n+5),np.arange(n+4)]=v
    a2=a@a;a4=a2@a2
    return (sp.diags(frequency*(np.arange(n)+.5),format='csr'),sp.csr_matrix(a[:n,:n]),sp.csr_matrix(a2[:n,:n]),sp.csr_matrix(a4[:n,:n]))

def fermi_spectrum(m:Fermi3D,n=(12,18,10),states=44):
    op=[fock_ops(n[i],w) for i,w in enumerate([m.wx,m.wy,m.wb])]
    ids=[sp.eye(x,format='csr') for x in n]
    def embed(M,axis):
        A=sp.csr_matrix([[1.]])
        for i in range(3):A=sp.kron(A,M if i==axis else ids[i],format='csr')
        return A
    X=embed(op[0][1],0);X2=embed(op[0][2],0);Y2=embed(op[1][2],1);B=embed(op[2][1],2)
    H=embed(op[0][0],0)+embed(op[1][0]+m.c4*op[1][3],1)+embed(op[2][0],2)
    H=H+m.g*(X@Y2)+.5*m.wb**2*m.lam**2*X2-m.wb**2*m.lam*(X@B)
    E,U=eigsh(H,k=states,which='SA',tol=1e-11,v0=np.ones(H.shape[0]),maxiter=20000)
    ix=np.argsort(E);E=E[ix];U=U[:,ix]
    A=U.T@(X@U)
    return E,U,A,{'basis':list(n),'dimension':int(H.shape[0]),'states':states,'ground_energy':float(E[0]),'first_excitations':(E[1:8]-E[0]).tolist()}

def spectral_kubo(E,A,beta,t,theta=0.,maxtr=None):
    p=np.exp(-beta*(E-E[0]));p/=p.sum()
    ii,jj=np.triu_indices(len(E),1)
    d=E[jj]-E[ii];weight=2*(p[ii]-p[jj])*abs(A[ii,jj])**2/(beta*d)
    keep=weight>1e-14
    d=d[keep];weight=weight[keep]
    diag=np.sum(p*np.diag(A)**2)-(np.sum(p*np.diag(A)))**2
    corr=diag+np.cos(np.outer(t,d))@weight
    return corr,{'C0':float(corr[0]),'retained_transitions':int(keep.sum()),'max_discarded_weight':float(np.sum(weight[~np.ones_like(weight,dtype=bool)])),'ground_thermal_population':float(p[0]),'highest_retained_thermal_population':float(p[-1])},(d,weight)

if __name__=='__main__':
 from time import time
 for n in [(17,17,17),(21,21,21)]:
  ti=time();_,_,_,meta=morse_spectrum(Morse3D(),n=n,nstate=24);print('morse',n,round(time()-ti,2),meta,flush=True)
 for n in [(10,14,8),(12,18,10)]:
  ti=time();_,_,_,meta=fermi_spectrum(Fermi3D(),n=n,states=44);print('fermi',n,round(time()-ti,2),meta,flush=True)

def morse1_spectrum(m,ngrid=80,bounds=(-2.5,5.0),nstate=26):
    x=np.linspace(*bounds,ngrid)
    V=m.D[0]*(1-np.exp(-m.a[0]*x))**2
    H=sinc_kinetic(ngrid,x[1]-x[0])+np.diag(V)
    E,U=eigh(H,check_finite=False)
    E=E[:nstate];U=U[:,:nstate];A=U.T@(x[:,None]*U)
    return E,U,A,{'grid':ngrid,'bounds':list(bounds),'ground_energy':float(E[0])}

def fermi2_spectrum(m,n=(12,18),states=44):
    Xop=fock_ops(n[0],m.wx);Yop=fock_ops(n[1],m.wy)
    X=sp.kron(Xop[1],sp.eye(n[1]),format='csr')
    H=sp.kron(Xop[0],sp.eye(n[1]))+sp.kron(sp.eye(n[0]),Yop[0]+m.c4*Yop[3])
    H+=m.g*sp.kron(Xop[1],Yop[2])
    E,U=eigh(H.toarray(),check_finite=False)
    E=E[:states];U=U[:,:states];A=U.T@(X@U)
    return E,U,A,{'basis':list(n),'ground_energy':float(E[0])}