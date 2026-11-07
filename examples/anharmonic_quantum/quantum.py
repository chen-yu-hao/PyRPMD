"""Independent quantum references: sine DVR, Fock spectral sum, open leads.
Truncation/box/spacing checks are caller responsibilities and saved explicitly.
"""
import numpy as np
from scipy.linalg import eigh,solve
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

def spectral_lines(E,U,A,beta):
    p=np.exp(-beta*(E-E[0]));p/=p.sum();a=U.T.conj()@(A@U)
    i,j=np.triu_indices(len(E),1);de=E[j]-E[i];coef=2*(p[i]-p[j])*abs(a[i,j])**2
    keep=(de>1e-10)&(coef>1e-13);de=de[keep];coef=coef[keep]
    return de,coef,{'E0':float(E[0]),'E1':float(E[1]),'first_gap':float(E[1]-E[0]),'mean':float(np.dot(p,a.diagonal().real)),
        'thermal_energy':float(p@E),'top_10_population':float(p[-10:].sum()),'response_sum_rule':float(coef@de),
        'state_count':len(E),'C0':float(np.sum(coef/(beta*de))+np.dot(p,abs(a.diagonal())**2)-abs(np.dot(p,a.diagonal()))**2)}

def spectrum(de,coef,w,eta):
    s=eta-1j*np.asarray(w);return (de[:,None]/(de[:,None]**2+s[None,:]**2)).T@coef

def morse_quantum(beta=8.,n=300,bounds=(-3.,10.)):
    lo,hi=bounds;L=hi-lo;j=np.arange(1,n+1);x=lo+L*j/(n+1)
    S=np.sqrt(2/(n+1))*np.sin(np.pi*np.outer(j,j)/(n+1))
    T=(S*(.5*(np.pi*j/L)**2))@S.T
    lines=[];records=[]
    for D,a in zip((8.,6.,5.),(.5,.55,.6)):
        V=D*np.expm1(-a*x)**2;E,U=eigh(T+np.diag(V),check_finite=False)
        de,c,meta=spectral_lines(E,U,np.diag(x),beta)
        omega=a*np.sqrt(2*D);exact0=omega/2-a*a/8
        meta.update(analytic_E0=exact0,ground_energy_error=abs(float(E[0])-exact0),box=list(bounds),grid_points=n,
            dissociation_energy=D,population_above_dissociation=float(np.exp(-beta*(E[E>=D]-E[0])).sum()/np.exp(-beta*(E-E[0])).sum()))
        lines.append((de,c));records.append(meta)
    return lines,records

def oscillator(n,w):
    N=n+4;X=np.diag(np.sqrt(np.arange(1,N)/(2*w)),1);X=X+X.T;X2=X@X;X4=X2@X2
    return np.diag(w*(np.arange(n)+.5)),X[:n,:n],X2[:n,:n],X4[:n,:n]

def fermi_quantum(beta=6.,dim=3,coupling=.15,basis=(16,22,10),states=90):
    ns=basis[:dim];ops=[oscillator(ns[0],2),oscillator(ns[1],1)]
    if dim==3:ops.append(oscillator(ns[2],1.3))
    def emb(op,axis):
        out=sp.csr_matrix([[1.]])
        for i,n in enumerate(ns):out=sp.kron(out,sp.csr_matrix(op) if i==axis else sp.eye(n),format='csr')
        return out
    X=emb(ops[0][1],0);Y2=emb(ops[1][2],1)
    H=emb(ops[0][0],0)+emb(ops[1][0]+(.35**2/8+.005)*ops[1][3],1)+.35*(X@Y2)
    if dim==3:
        B=emb(ops[2][1],2);H+=emb(ops[2][0],2)+.845*coupling**2*emb(ops[0][2],0)-1.69*coupling*(X@B)
    if H.shape[0]<900:
        E,U=eigh(H.toarray(),check_finite=False);E=E[:states];U=U[:,:states]
    else:
        E,U=eigsh(H,k=states,which='SA',tol=2e-11,v0=np.random.default_rng(8821).normal(size=H.shape[0]));idx=np.argsort(E);E=E[idx];U=U[:,idx]
    de,c,meta=spectral_lines(E,U,X,beta);meta.update(basis=list(ns),dimension=dim,basis_dimension=H.shape[0])
    return de,c,meta

def eckart_scattering(energies,dim=3,nx=10,nz=4,dq=.10,L=7.,strength=.65,spectator=.10):
    """Multichannel discrete-q scattering, exact for the declared finite model.
    Retarded lead self-energies and recursive block Green functions. Energies
    are relative to the lowest asymptotic vibrational level. No CAP is used.
    Returns total CRP and ground-channel transmission, not a fitted resonance.
    """
    from scipy.linalg import sqrtm
    if dim==1:
        B=np.zeros((1,1));X=B;X2=B;Z=B;Z2=B
    else:
        hx,x,x2,_=oscillator(nx,1.5)
        if dim==2:B=hx;X=x;X2=x2;Z=np.zeros_like(X);Z2=Z
        else:
            hz,z,z2,_=oscillator(nz,2.3);Ix=np.eye(nx);Iz=np.eye(nz)
            X=np.kron(x,Iz);X2=np.kron(x2,Iz);Z=np.kron(Ix,z);Z2=np.kron(Ix,z2)
            B=np.kron(hx,Iz)+np.kron(Ix,hz)+2.645*spectator**2*X2-5.29*spectator*(X@Z)
    nc=B.shape[0];I=np.eye(nc);t=.5/dq**2
    q=np.arange(-L,L+dq/2,dq);f=strength*np.tanh(2*q)
    slices=np.array([B+((1/np.cosh(qi)**2)+1.125*fi**2 if dim>1 else 1/np.cosh(qi)**2)*I-(2.25*fi*X if dim>1 else 0) for qi,fi in zip(q,f)])
    HL=B+(1.125*strength**2*I+2.25*strength*X if dim>1 else 0)
    HR=B+(1.125*strength**2*I-2.25*strength*X if dim>1 else 0)
    eL,UL=eigh(HL);eR,UR=eigh(HR);origin=eL[0]
    def lead(E,ev,U):
        z=(E-ev-2*t).astype(complex);disc=np.lib.scimath.sqrt(z*z-4*t*t)
        sigma=(z-disc)/2
        # Outside the band choose the root of magnitude <= hopping.
        alt=(z+disc)/2;choose=(np.imag(sigma)>1e-12)|((abs(sigma)>t*(1+1e-12))&(abs(z)>=2*t))
        sigma[choose]=alt[choose];gam=np.maximum(-2*sigma.imag,0)
        return (U*sigma)@U.T,(U*np.sqrt(gam))@U.T,gam
    allN=[];allT=[];units=[];maxP=[]
    for energy in np.asarray(energies)+origin:
        SL,GL,gL=lead(energy,eL,UL);SR,GR,gR=lead(energy,eR,UR)
        prev=None;G=None
        for j,H in enumerate(slices):
            A=(energy-2*t+1e-12j)*I-H
            if j==0:A=A-SL
            else:A=A-t*t*prev
            if j==len(slices)-1:A=A-SR
            cur=solve(A,I,check_finite=False)
            G=cur if G is None else -t*cur@G
            prev=cur
        transmission=GR@G@GL
        allN.append(float(np.linalg.norm(transmission)**2))
        col=GR@G@(UL[:,0]*np.sqrt(gL[0]));allT.append(float(np.vdot(col,col).real))
        # Independent reflected Green block for v=0, retaining all open modes.
        back=None
        for j in range(len(slices)-1,-1,-1):
            A=(energy-2*t+1e-12j)*I-slices[j]
            A=A-(SR if j==len(slices)-1 else t*t*back)
            if j==0:A=A-SL
            back=solve(A,I,check_finite=False)
        refl=UL[:,0]-1j*GL@back@(UL[:,0]*np.sqrt(gL[0]))
        units.append(float(abs(np.vdot(refl,refl).real+allT[-1]-1)))
        maxP.append(float(np.linalg.svd(transmission,compute_uv=False).max()**2))
    return np.array(allN),np.array(allT),{'dimension':dim,'q_spacing':dq,'q_extent':L,'q_points':len(q),'bath_basis':[nx,nz][:max(0,dim-1)],
        'channel_count':nc,'lead_ground_energy':float(origin),'relative_lead_energies':(eL-origin).tolist(),
        'max_unitarity_defect':max(units),'max_transmission_eigenvalue':max(maxP),'q_discretization':'nearest-neighbor open-lead finite difference'}

def thermal_rate(energy,N,beta,lead_levels):
    Z=np.exp(-beta*np.asarray(lead_levels)).sum()
    return float(np.sqrt(beta/(2*np.pi))*np.trapezoid(np.exp(-beta*np.asarray(energy))*N,energy)/Z)
