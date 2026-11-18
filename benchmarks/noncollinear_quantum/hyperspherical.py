"""Finite-basis noncollinear triatomic scattering in heavy-pair Jacobi hyperspheres.

T_rho=-d_rho^2/(2mu), with transformed 6D measure. The hyperangular kinetic
operator is [-d_alpha^2 + j^2/sin^2(alpha) + l^2/cos^2(alpha)-1/4]/(2mu rho^2).
Both parity sectors and Coriolis terms are implemented for arbitrary integer J.
A finite rho/alpha/j/channel calculation is a truncated reference, not an
all-J-converged thermal rate. Exact unitarity of the discrete leads alone is
not a physical accuracy certificate.
"""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import numpy as np,json,time,argparse
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.special import lpmv,gammaln
from numpy.polynomial.legendre import leggauss
from leps import energy,MU_R,SCALE,A,B,D,KB,MU_IN
ROOT=Path(__file__).resolve().parents[2]

def helicity_kinetics(J,jmax,parity):
    if int(J)!=J or J<0 or int(jmax)!=jmax or jmax<0 or parity not in (-1,1):raise ValueError('Invalid angular quantum numbers')
    Ks=list(range(0 if parity==(-1)**J else 1,min(J,jmax)+1))
    pairs=[(K,j) for K in Ks for j in range(K,jmax+1)]
    labels={p:i for i,p in enumerate(pairs)};n=len(pairs);j2=np.zeros((n,n));l2=np.zeros((n,n))
    for idx,(K,j) in enumerate(pairs):
        j2[idx,idx]=j*(j+1);l2[idx,idx]=J*(J+1)+j*(j+1)-2*K*K
        if (K+1,j) in labels:
            val=-np.sqrt((J*(J+1)-K*(K+1))*(j*(j+1)-K*(K+1)))
            if K==0:val*=np.sqrt(2)
            i2=labels[(K+1,j)];l2[idx,i2]=l2[i2,idx]=val
    return pairs,j2,l2

def sine_fd_kinetic(n,length):
    """Symmetric eighth-order Dirichlet FD using odd boundary extension."""
    h=length/(n+1);c=np.array([-205/72,8/5,-1/5,8/315,-1/560]);H=np.zeros((n,n))
    for i in range(1,n+1):
        for offset in range(-4,5):
            j=i+offset;sign=1
            while j<0 or j>n+1:
                if j<0:j=-j;sign*=-1
                if j>n+1:j=2*(n+1)-j;sign*=-1
            if 1<=j<=n:H[i-1,j-1]+=-sign*c[abs(offset)]/(h*h)
    return sparse.csr_matrix(H)

def angular_setup(nalpha=96,alpha_max=.35,jmax=23,J=0,parity=1,nquad=None):
    pairs,j2,l2=helicity_kinetics(J,jmax,parity)
    if not pairs:raise ValueError('Empty angular symmetry sector')
    nq=nquad or max(jmax+1,24);u,w=leggauss(nq)
    a=alpha_max*np.arange(1,nalpha+1)/(nalpha+1);nfunc=len(pairs)
    transforms={};indices={}
    for K in sorted(set(k for k,j in pairs)):
        js=np.array([j for k,j in pairs if k==K]);ix=np.array([i for i,(k,j) in enumerate(pairs) if k==K])
        Y=np.column_stack([np.sqrt((2*j+1)/2)*np.exp(.5*(gammaln(j-K+1)-gammaln(j+K+1)))*lpmv(K,j,u) for j in js])
        transforms[K]=Y*np.sqrt(w[:,None]);indices[K]=ix
        if not np.allclose(transforms[K].T@transforms[K],np.eye(len(js)),atol=1e-10):raise ValueError('Angular quadrature is insufficient')
    ka=sine_fd_kinetic(nalpha,alpha_max)
    kinetic=sparse.kron(ka,sparse.eye(nfunc),format='csr')
    blocks=[j2/np.sin(x)**2+l2/np.cos(x)**2-.25*np.eye(nfunc) for x in a]
    kinetic+=sparse.block_diag(blocks,format='csr')
    return a,u,w,transforms,indices,kinetic,pairs

def distances(rho,alpha,u):
    R=rho*np.cos(alpha[:,None]);q=rho*np.sin(alpha[:,None])/SCALE
    r0=np.sqrt(np.maximum(q*q+A*A*R*R+2*A*R*q*u[None,:],1e-25))
    r1=np.sqrt(np.maximum(q*q+B*B*R*R-2*B*R*q*u[None,:],1e-25))
    return np.stack([r0,r1,np.broadcast_to(R,r0.shape)],axis=-1)

def make_channels(config):
    nalpha=int(config['nalpha']);jmax=int(config['jmax']);J=int(config.get('J',0));parity=int(config.get('parity',(-1)**J))
    a,u,w,Y,ix,T,pairs=angular_setup(nalpha,float(config.get('alpha_max',.35)),jmax,J,parity,config.get('nquad'))
    nfunc=len(pairs);nc=int(config['channels']);step=float(config['radial_step']);rmin=float(config.get('rho_min',3.5));rmax=float(config.get('rho_max',12))
    rho=np.linspace(rmin,rmax,round((rmax-rmin)/step)+1);vals=[];over=[];prev=None;start=time.perf_counter();boundary=[]
    for nr,R in enumerate(rho):
        dist=distances(R,a,u);V=energy(dist);blocks=[]
        for i in range(nalpha):
            bl=np.zeros((nfunc,nfunc))
            for K,U in Y.items():bl[np.ix_(ix[K],ix[K])]=U.T@(V[i,:,None]*U)
            blocks.append(bl)
        H=T/(2*MU_R*R*R)+sparse.block_diag(blocks,format='csr')
        E,U=eigsh(H,k=nc,sigma=0.,which='LM',tol=1e-9,v0=np.ones(H.shape[0]))
        order=np.argsort(E);E=E[order];U=U[:,order]
        if prev is not None:over.append(U.T@prev)
        vals.append(E);prev=U
        boundary.append(np.sum(U.reshape(nalpha,nfunc,nc)[-4:]**2,axis=(0,1)))
        if nr%20==0:print('angular',J,parity,nr,len(rho),'seconds',time.perf_counter()-start,flush=True)
    # Spatial arrangement weights, summing orthogonal helicity components.
    state=prev.reshape(nalpha,nfunc,nc);mask=dist[...,0]<dist[...,1];hbr=np.zeros(nc)
    for K,U in Y.items():
        amp=np.einsum('uj,ajc->auc',U,state[:,ix[K],:]);hbr+=np.sum(abs(amp)**2*mask[:,:,None],axis=(0,1))
    return {'rho':rho,'eps':np.array(vals),'overlap':np.array(over),'hbr_weight':hbr,'alpha':a,'cos_gamma':u,
            'upper_alpha_boundary_weight':np.array(boundary),'angular_pairs':np.array(pairs),'step':rho[1]-rho[0]}

def scattering(ch,energies):
    vals=ch['eps'];h=float(ch['step']);tau=1/(2*MU_R*h*h);n=vals.shape[1];W=-tau*ch['overlap'];I=np.eye(n,dtype=complex)
    hbr=ch['hbr_weight']>.5;prod=np.where(~hbr)[0];react=np.where(hbr)[0]
    if not len(prod) or not len(react):raise RuntimeError('Channel basis does not resolve both arrangements')
    out=[];unit=[];p0=[];incomplete=[]
    for E in energies:
        sigma=np.zeros((n,n),complex)
        for i in range(len(vals)-1):
            G=np.linalg.inv(np.diag(E-vals[i]-2*tau)-sigma)
            sigma=W[i]@G@W[i].T
        z=E-vals[-1]-2*tau;lead=np.empty(n,complex)
        for j,x in enumerate(z):
            lead[j]=.5*(x-1j*np.sqrt(4*tau*tau-x*x)) if abs(x)<2*tau else .5*(x-np.sign(x)*np.sqrt(x*x-4*tau*tau))
        gam=-2*lead.imag;G=np.linalg.inv(np.diag(z-lead)-sigma);S=I-1j*np.sqrt(gam)[:,None]*G*np.sqrt(gam)[None,:]
        active=gam>0;Sc=S[np.ix_(active,active)];unit.append(float(np.max(abs(Sc.conj().T@Sc-np.eye(active.sum())))) if active.any() else 0.)
        prob=gam[:,None]*gam[None,:]*abs(G)**2;out.append(float(prob[np.ix_(prod,react)].sum()));p0.append(float(prob[prod,react[0]].sum()))
        incomplete.append(bool(E>=vals[-1,-1]))
    return np.array(out),np.array(p0),np.array(unit),np.array(incomplete)

def run(config,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    receipt=out/'input.json';text=json.dumps(config,indent=2)
    if receipt.exists() and receipt.read_text()!=text:
        raise ValueError('Output contains different scientific input')
    receipt.write_text(text);ch=make_channels(config)
    react=ch['hbr_weight']>.5;E0=float(ch['eps'][-1,react].min());E=E0+np.linspace(2e-6,float(config.get('energy_max_above_edge',.008)),int(config.get('energy_points',61)))
    N,P,unit,inc=scattering(ch,E)
    np.savez_compressed(out/'quantum.npz',**ch,total_energy=E,CRP=N,P_v0=P,unitarity=unit,open_channels_truncated=inc)
    result={'J':int(config.get('J',0)),'parity':int(config.get('parity',1)),
      'scope':'noncollinear three-internal-coordinate finite-channel quantum calculation; NOT a full J-summed thermal rate',
      'edge_reactant_threshold_Eh':E0,'max_discrete_unitarity_defect':float(unit.max()),
      'energies_exceed_retained_channel_threshold':int(inc.sum()),
      'largest_alpha_boundary_probability':float(ch['upper_alpha_boundary_weight'].max()),
      'edge_arrangement_weights':ch['hbr_weight'].tolist(),
      'maximum_arrangement_impurity':float(np.minimum(ch['hbr_weight'],1-ch['hbr_weight']).max()),
      'reference_usable_for_rate_correction':False,
      'failed_or_unestablished_gates':['asymptotic channel localization','angular basis convergence','radial step convergence','total J sum'],
      'elapsed_seconds':time.perf_counter()-start,
      'physical_converged':False,'J_tail_complete':False}
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(json.loads(Path(a.input).read_text()),a.output)
