"""Independent-block Gaussian-reference Metropolis PIMD and RPMD tangent flow.

All masses=1, hbar=1. A Metropolis sampler targets the exact finite-P
ring-polymer canonical density for the supplied potential. Weakly correlated
replica blocks, not individual PIMD samples, are the independent error units.
"""
import numpy as np
from scipy.linalg import eigh
from numba import njit,prange

@njit(cache=True)
def v_force(q,kind):
    d=q.shape[0];f=np.empty(d)
    if kind in (0,1):  # Morse 3D or Morse x only
        a=np.array([.65,.70,.75]);D=np.array([8.,7.,6.]);gam=np.array([.12,.08,.06])
        u=np.empty(d);du=np.empty(d);pot=0.
        for i in range(d):
            e=np.exp(-a[i]*q[i]);u[i]=1.-e;du[i]=a[i]*e
            pot+=D[i]*u[i]*u[i];f[i]=-2.*D[i]*u[i]*du[i]
        if d==3:
            pot+=gam[0]*u[0]*u[1]+gam[1]*u[0]*u[2]+gam[2]*u[1]*u[2]
            f[0]-=du[0]*(gam[0]*u[1]+gam[1]*u[2])
            f[1]-=du[1]*(gam[0]*u[0]+gam[2]*u[2])
            f[2]-=du[2]*(gam[1]*u[0]+gam[2]*u[1])
        return pot,f
    if kind in (2,3):  # Fermi 3D / Fermi 2D
        x=q[0];y=q[1];g=.35;c4=.35**2/(2*4.)+.005
        pot=2*x*x+.5*y*y+g*x*y*y+c4*y**4
        f[0]=-4*x-g*y*y;f[1]=-y-2*g*x*y-4*c4*y**3
        if kind==2:
            b=q[2]; h=b-.35*x
            pot+=.5*1.3**2*h*h;f[0]+=1.3**2*.35*h;f[2]=-1.3**2*h
        return pot,f
    raise ValueError('unknown potential kind')

@njit(cache=True)
def quartic_residual(q,K,beta):
    P,D=q.shape;total=0.
    for j in range(P):
        v,f=v_force(q[j],0 if D==3 else 1) # unused generic helper for Morse only
        h=0.
        for a in range(D):
            for b in range(D):h+=.5*q[j,a]*K[a,b]*q[j,b]
        total+=v-h
    return -beta/P*total

@njit(cache=True)
def log_residual_batch(z,K,beta,kind,shift):
    N,P,D=z.shape;out=np.empty(N)
    for i in range(N):
        val=0.
        for j in range(P):
            p,_=v_force(z[i,j],kind)
            h=0.
            for a in range(D):
                for b in range(D):h+=.5*(z[i,j,a]-shift[a])*K[a,b]*(z[i,j,b]-shift[b])
            val+=p-h
        out[i]=-beta/P*val
    return out

def gaussian_paths(K,beta,beads,n,rng,shift=None):
    eigen,U=eigh(K)
    if min(eigen)<=0:raise ValueError('Gaussian reference K must be positive definite')
    freq=2*beads/beta*np.sin(np.pi*np.arange(beads//2+1)/beads)
    scales=np.sqrt(beads/beta/(freq[:,None]**2+eigen[None,:]))
    white=rng.normal(size=(n,beads,len(eigen)))
    f=np.fft.rfft(white,axis=1,norm='ortho');f*=scales[None]
    q=np.fft.irfft(f,n=beads,axis=1,norm='ortho')@U.T
    return np.ascontiguousarray(q if shift is None else q+np.asarray(shift)[None,None,:])

def pimd_mh(model,K,kind,beta=6.,P=24,nblocks=16,nperblock=64,thin=8,burn=250,seed=1001,shift=None):
    rng=np.random.default_rng(seed);D=len(K)
    shift=np.zeros(D) if shift is None else np.asarray(shift,dtype=float)
    currents=gaussian_paths(K,beta,P,nblocks,rng,shift)
    lw=log_residual_batch(currents,K,beta,kind,shift)
    out=np.empty((nblocks*nperblock,P,D));accept=np.zeros(nblocks,int)
    proposed=0
    for step in range(burn+nperblock*thin):
        proposals=gaussian_paths(K,beta,P,nblocks,rng,shift)
        prop_lw=log_residual_batch(proposals,K,beta,kind,shift)
        yes=np.log(rng.random(nblocks))<prop_lw-lw
        currents[yes]=proposals[yes];lw[yes]=prop_lw[yes];accept+=yes
        proposed+=1
        if step>=burn and (step-burn+1)%thin==0:
            n=(step-burn+1)//thin-1
            for b in range(nblocks):out[b*nperblock+n]=currents[b]
    p=rng.normal(size=out.shape)*np.sqrt(P/beta)
    return np.ascontiguousarray(out),np.ascontiguousarray(p),{
        'acceptance_rate':float(accept.sum()/(nblocks*proposed)),
        'acceptance_by_block':list(map(float,accept/proposed)),
        'P':P,'nblocks':nblocks,'nperblock':nperblock,'thin':thin,'burn':burn,'seed':seed}

@njit(cache=True)
def forces(q,kind,beta,f,df=None,dq=None):
    P,D=q.shape;spring=(P/beta)**2
    for j in range(P):
        _,g=v_force(q[j],kind)
        for a in range(D):
            f[j,a]=g[a]-spring*(2*q[j,a]-q[(j-1)%P,a]-q[(j+1)%P,a])
        if df is not None:
            x=q[j,0]
            if kind in (0,1):
                a=.65;D0=8.;e=np.exp(-a*x)
                second=2*D0*a*a*(2*e*e-e)
                if kind==0:
                    uy=1-np.exp(-.70*q[j,1]);uz=1-np.exp(-.75*q[j,2])
                    second+=.65**2*e*(.12*uy+.08*uz)*(-1.)
                crossy=.12*.65*.70*e*np.exp(-.70*q[j,1]) if kind==0 else 0.
                crossz=.08*.65*.75*e*np.exp(-.75*q[j,2]) if kind==0 else 0.
            else:
                second=4.+(1.3**2*.35**2 if kind==2 else 0.)
                crossy=2*.35*q[j,1];crossz=-1.3**2*.35 if kind==2 else 0.
            for a in range(D):
                lap=0.
                if a==0:lap-=second*dq[j,0]+crossy*dq[j,1]
                if D==3 and a==0:lap-=crossz*dq[j,2]
                if a==1:lap-=crossy*dq[j,0]
                if D==3 and a==2:lap-=crossz*dq[j,0]
                if a==1:
                    if kind==0:lap-= (2*7.*.70**2*(2*np.exp(-2*.70*q[j,1])-np.exp(-.70*q[j,1])))*dq[j,1]
                    else:lap-= (1.+2*.35*x+12*(.35**2/8.+.005)*q[j,1]**2)*dq[j,1]
                if a==2:
                    if kind==0:lap-=(2*6.*.75**2*(2*np.exp(-2*.75*q[j,2])-np.exp(-.75*q[j,2])))*dq[j,2]
                    else:lap-=1.3**2*dq[j,2]
                # Morse off diagonal y-z and x-y,x-z already in x force;
                # add Hessian cross and coupling curvature in y,z forces.
                if kind==0 and a==1:
                    lap-=.06*.70*.75*np.exp(-.70*q[j,1]-.75*q[j,2])*dq[j,2]
                    lap-= -.70**2*np.exp(-.70*q[j,1])*(.12*(1-np.exp(-.65*x))+.06*(1-np.exp(-.75*q[j,2])))*dq[j,1]
                if kind==0 and a==2:
                    lap-=.06*.70*.75*np.exp(-.70*q[j,1]-.75*q[j,2])*dq[j,1]
                    lap-= -.75**2*np.exp(-.75*q[j,2])*(.08*(1-np.exp(-.65*x))+.06*(1-np.exp(-.70*q[j,1])))*dq[j,2]
                df[j,a]=lap-spring*(2*dq[j,a]-dq[(j-1)%P,a]-dq[(j+1)%P,a])

@njit(cache=True,parallel=True)
def propagate_kubo_tangent(q0,p0,kind,beta,dt,steps,stride,nblocks):
    N,P,D=q0.shape;nper=N//nblocks;nt=steps//stride+1;out=np.zeros((nblocks,nt));
    for b in prange(nblocks):
        for sample in range(b*nper,(b+1)*nper):
            for sign in range(2):
                q=q0[sample].copy();p=p0[sample].copy();
                if sign==1:p=-p
                dq=np.zeros_like(q);dp=np.zeros_like(p)
                for j in range(P):dp[j,0]=1.
                f=np.empty_like(q);df=np.empty_like(q)
                forces(q,kind,beta,f,df,dq)
                idx=1
                for step in range(1,steps+1):
                    for j in range(P):
                        for a in range(D):
                            p[j,a]+=.5*dt*f[j,a];dp[j,a]+=.5*dt*df[j,a]
                            q[j,a]+=dt*p[j,a];dq[j,a]+=dt*dp[j,a]
                    forces(q,kind,beta,f,df,dq)
                    for j in range(P):
                        for a in range(D):p[j,a]+=.5*dt*f[j,a];dp[j,a]+=.5*dt*df[j,a]
                    if step%stride==0:
                        val=0.
                        for j in range(P):val+=dq[j,0]/P
                        out[b,idx]+=val/(2*nper)
                        idx+=1
    return out

def centroid_virial(q,model,kind,beta):
    cen=q.mean(axis=1,keepdims=True)
    V=model.V(q) if hasattr(model,'V') else None
    # explicit model force works with shape (...,D)
    F=model.force(q)
    return .5*q.shape[-1]/beta+np.mean(V,axis=1)-.5*np.mean(np.sum((q-cen)*F,axis=2),axis=1)

@njit(cache=True,parallel=True)
def propagate_centroid_correlation(q0,p0,kind,beta,dt,steps,stride,nblocks,center):
    """Bounded Kubo RPMD centroid x autocorrelation. Reverse-momentum pairing."""
    N,P,D=q0.shape;nper=N//nblocks;nt=steps//stride+1
    out=np.zeros((nblocks,nt));spring=(P/beta)**2
    for b in prange(nblocks):
        for sample in range(b*nper,(b+1)*nper):
            initial=0.
            for j in range(P):initial+=q0[sample,j,0]/P
            initial-=center
            for sign in range(2):
                q=q0[sample].copy();p=p0[sample].copy()
                if sign==1:p=-p
                f=np.empty_like(q);forces(q,kind,beta,f)
                out[b,0]+=initial*initial/(2*nper)
                it=1
                for step in range(1,steps+1):
                    for j in range(P):
                        for a in range(D):
                            p[j,a]+=.5*dt*f[j,a]
                            q[j,a]+=dt*p[j,a]
                    forces(q,kind,beta,f)
                    for j in range(P):
                        for a in range(D):p[j,a]+=.5*dt*f[j,a]
                    if step%stride==0:
                        now=0.
                        for j in range(P):now+=q[j,0]/P
                        now-=center
                        out[b,it]+=initial*now/(2*nper)
                        it+=1
    return out