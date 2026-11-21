"""Independent scalar-formula oracle for isolated tests without a PyRPMD install.

Not used by the production workflow. The native test imports pyrpmd.cv instead.
Definition checked against GreenGroup/RPMDrate rpmdrate/_surface.f90 and
PyRPMD cv.py (ffa14f247e453e9626b2e3fb3c288da0ca753d2e).
No external RPMDrate/Fortran trajectory calculation is claimed by this oracle.
"""
from types import SimpleNamespace
import numpy as np


def distance(q,b):
    d=q[b[0]]-q[b[1]];r=np.linalg.norm(d)
    if r<1e-12:raise ValueError('Coincident atoms')
    g=np.zeros_like(q);g[b[0]]=d/r;g[b[1]]=-d/r
    return r,g


class FormulaOracle:
    estimator='centroid'
    def __init__(self,fragments,masses,reference,forming_bonds,breaking_bonds,rinf,equivalent_paths=None):
        self.masses=np.array(masses,float);self.reference=np.array(reference,float);self.rinf=float(rinf)
        self.com=SimpleNamespace(fragments=tuple(tuple(x) for x in fragments))
        self.forming=tuple(tuple(x) for x in forming_bonds);self.breaking=tuple(tuple(x) for x in breaking_bonds)
        self.reference_difference=[distance(self.reference,f)[0]-distance(self.reference,b)[0] for f,b in zip(self.forming,self.breaking)]
        perms=[tuple(range(len(masses)))]+[tuple(p['permutation'] if isinstance(p,dict) else p) for p in equivalent_paths or []]
        self.paths=[(tuple(tuple(p[i] for i in b) for b in self.forming),tuple(tuple(p[i] for i in b) for b in self.breaking)) for p in perms]
    def surfaces(self,q):
        q=np.asarray(q,float);w=np.zeros(len(q))
        for sign,frag in zip([1,-1],self.com.fragments):w[list(frag)]=sign*self.masses[list(frag)]/self.masses[list(frag)].sum()
        R=w@q;r=np.linalg.norm(R)
        if r<1e-12:raise ValueError('Zero fragment separation')
        s0=self.rinf-r;g0=-w[:,None]*R[None,:]/r
        terms=np.array([[ref-distance(q,f)[0]+distance(q,b)[0] for ref,f,b in zip(self.reference_difference,fo,br)] for fo,br in self.paths])
        j=np.argmin(terms.max(0));i=np.argmax(terms[:,j]);fo,br=self.paths[i]
        s1=terms[i,j];g1=-distance(q,fo[j])[1]+distance(q,br[j])[1]
        # This oracle is deliberately first-derivative-only; native adapter
        # consumes scalar/gradients, never these unused placeholder Hessians.
        return s0,g0,None,float(s1),g1,None
    def centroid_value_gradient(self,q):
        s0,g0,_,s1,g1,_=self.surfaces(q);d=s0-s1
        if abs(d)<1e-10:raise ValueError('xi is singular')
        return s0/d,g0/d-s0*(g0-g1)/d**2
    def dividing_surface(self,xi):return OracleSurface(self,xi)


class OracleSurface:
    def __init__(self,coordinate,xi):self.coordinate=coordinate;self.xi=float(xi)
    def centroid_value_gradient(self,q):
        s0,g0,_,s1,g1,_=self.coordinate.surfaces(q)
        return (1-self.xi)*s0+self.xi*s1,(1-self.xi)*g0+self.xi*g1
