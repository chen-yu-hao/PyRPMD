"""PES-agnostic molecular collision assembly and finite-time RPMD integration.

Arbitrary numbers of atoms and internal coordinates are allowed. Callers must
supply canonical, COM-free fragment path ensembles (including their required
rotational/vibrational state convention). No guessed quantum state population,
thermostat, hidden dimensional reduction or model PES is inserted.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .bimolecular_scattering import sample_collision_flux


def assemble_collision(fragment_a, fragment_b, masses_a, masses_b, *,
                       temperature, bmax, launch_R, seed=0,
                       proposal_temperature_ratio=1.0):
    """Combine two prepared internal path ensembles, giving random orientations.

    Each fragment is (q,p) with shape [trajectories,beads,atoms,3]. Its COM and
    total momentum must vanish beadwise. Input arrays are never modified.
    Global COM paths/momenta vanish beadwise in the returned ensemble.
    """
    aq,ap=(np.asarray(v,float) for v in fragment_a);bq,bp=(np.asarray(v,float) for v in fragment_b)
    ma=np.asarray(masses_a,float);mb=np.asarray(masses_b,float)
    for q,p,m in [(aq,ap,ma),(bq,bp,mb)]:
        if (q.ndim!=4 or q.shape!=p.shape or q.shape[-1]!=3 or min(q.shape[:3])<1
                or m.shape!=(q.shape[2],) or np.any(m<=0)):
            raise ValueError('Bad fragment path or mass dimensions')
        if not all(np.isfinite(v).all() for v in (q,p,m)):raise ValueError('Nonfinite fragment data')
        if np.max(abs((q*m[None,None,:,None]).sum(2))/m.sum())>1e-9 or np.max(abs(p.sum(2)))>1e-8:
            raise ValueError('Fragment COM and momentum must vanish beadwise')
    if aq.shape[:2]!=bq.shape[:2] or not np.isfinite(launch_R) or launch_R<=0:
        raise ValueError('Matched ensemble sizes and positive launch distance required')
    n,P=aq.shape[:2];MA=ma.sum();MB=mb.sum();mu=MA*MB/(MA+MB)
    flux=sample_collision_flux(n,temperature,mu,bmax,seed=seed,proposal_temperature_ratio=proposal_temperature_ratio)
    rng=np.random.default_rng(seed+1)
    def rotate(q,p):
        R=Rotation.random(n,random_state=rng).as_matrix()
        return np.einsum('npac,ndc->npad',q,R),np.einsum('npac,ndc->npad',p,R)
    aq,ap=rotate(aq,ap);bq,bp=rotate(bq,bp)
    freq=2*P/flux.beta*np.sin(np.pi*np.arange(P//2+1)/P);freq[0]=1.
    normal=np.fft.rfft(rng.normal(size=(n,P,3)),axis=1,norm='ortho')
    normal*=np.sqrt(P/(flux.beta*mu))/freq[None,:,None];normal[:,0]=0
    R=np.fft.irfft(normal,n=P,axis=1,norm='ortho');R[:,:,0]+=flux.impact_parameter[:,None];R[:,:,2]+=launch_R
    pr=rng.normal(size=(n,P,3))*np.sqrt(P*mu/flux.beta);pr-=pr.mean(1,keepdims=True)
    pr[:,:,2]-=np.sqrt(2*mu*flux.energy)[:,None]
    aq-=MB/(MA+MB)*R[:,:,None,:];bq+=MA/(MA+MB)*R[:,:,None,:]
    ap-=ma[None,None,:,None]/MA*pr[:,:,None,:];bp+=mb[None,None,:,None]/MB*pr[:,:,None,:]
    return np.concatenate([aq,bq],axis=2),np.concatenate([ap,bp],axis=2),flux


def propagate_cartesian(q0,p0,masses,*,beta,dt,steps,potential,classifier,
                         check_stride=20):
    """Reference Verlet RPMD for arbitrary molecular PES callbacks.

    potential(flat_bead_frames) returns (energy[B],forces[B,atoms,3]), or an
    object with .energy and .forces. classifier(centroids, velocities) returns
    -1 (continue), 0 (non-target exit), 1 (target exit). Callback exceptions
    abort rather than silently reclassifying samples. The optimized triatomic
    benchmark is independently compared with this map in regression tests.
    """
    q=np.array(q0,dtype=float,copy=True);p=np.array(p0,dtype=float,copy=True);m=np.asarray(masses,float)
    if (q.ndim!=4 or q.shape!=p.shape or q.shape[-1]!=3 or min(q.shape[:3])<1
                or m.shape!=(q.shape[2],) or np.any(m<=0)):
        raise ValueError('Bad Cartesian ensemble')
    if not all(np.isfinite(v).all() for v in (q,p,m)) or beta<=0 or dt<=0 or not np.isfinite([beta,dt]).all():
        raise ValueError('Invalid numerical state')
    for value in (steps,check_stride):
        if isinstance(value,(bool,np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or int(value)!=value or value<1:raise ValueError('Positive integer steps/stride required')
    N,P,A,_=q.shape;wp2=(P/beta)**2
    def evaluate(x):
        result=potential(x.reshape(-1,A,3))
        en,force=(result.energy,result.forces) if hasattr(result,'forces') else result
        en=np.asarray(en,float);force=np.asarray(force,float)
        if en.shape!=(len(x)*P,) or force.shape!=(len(x)*P,A,3) or not np.isfinite(en).all() or not np.isfinite(force).all():
            raise ValueError('PES violated the physical energy/negative-gradient contract')
        force=force.reshape(x.shape)-m[None,None,:,None]*wp2*(2*x-np.roll(x,1,axis=1)-np.roll(x,-1,axis=1))
        return en.reshape(len(x),P),force
    def hamiltonian(x,v,en):
        return (en.sum(1)+(v*v/(2*m[None,None,:,None])).sum((1,2,3))+
                .5*wp2*((x-np.roll(x,1,axis=1))**2*m[None,None,:,None]).sum((1,2,3)))/P
    en,f=evaluate(q);initial=hamiltonian(q,p,en);drift=np.zeros(N);outcomes=np.full(N,-1);exit_step=np.zeros(N,int)
    for step in range(1,int(steps)+1):
        active=np.flatnonzero(outcomes<0)
        if not len(active):break
        v=p[active]+.5*dt*f[active];x=q[active]+dt*v/m[None,None,:,None]
        ee,ff=evaluate(x);v+=.5*dt*ff;q[active]=x;p[active]=v;f[active]=ff
        if step%check_stride==0 or step==steps:
            now=hamiltonian(x,v,ee);drift[active]=np.maximum(drift[active],abs(now-initial[active]))
            code=np.asarray(classifier(x.mean(1),(v/m[None,None,:,None]).mean(1)))
            if code.shape!=(len(active),) or not np.isin(code,[-1,0,1]).all():raise ValueError('Invalid classifier result')
            done=active[code>=0];outcomes[done]=code[code>=0];exit_step[done]=step
    return {'positions':q,'momenta':p,'outcomes':outcomes,'exit_step':exit_step,
            'max_energy_drift_per_trajectory':drift,'physical_converged':False}


def run_bimolecular_scattering(fragment_a,fragment_b,masses_a,masses_b,*,
        temperature,bmax,launch_R,potential,classifier,dt,steps,seed=0,
        proposal_temperature_ratio=1.0,check_stride=20,electronic_weight=1.0,
        symmetry_factor=1.0,provenance=None):
    """Run a PES-independent noncollinear incoming-flux RPMD ensemble.

    This is the executable arbitrary-atom baseline, not an exact selective
    quantum propagator. Independent, canonical fragment ensembles are required
    inputs. The rate and raw outcomes retain unfinished trajectories. Supply a
    provenance mapping identifying the PES and internal-state preparation.
    """
    from .bimolecular_scattering import thermal_impact_rate
    from collections.abc import Mapping
    if not isinstance(provenance,Mapping) or not provenance.get('pes_identity') or not provenance.get('reactant_preparation'):
        raise ValueError('Explicit PES and reactant preparation provenance required')
    q,p,flux=assemble_collision(fragment_a,fragment_b,masses_a,masses_b,
        temperature=temperature,bmax=bmax,launch_R=launch_R,seed=seed,
        proposal_temperature_ratio=proposal_temperature_ratio)
    m=np.r_[masses_a,masses_b];MA=np.sum(masses_a);MB=np.sum(masses_b)
    trajectories=propagate_cartesian(q,p,m,beta=flux.beta,dt=dt,steps=steps,
        potential=potential,classifier=classifier,check_stride=check_stride)
    rate=thermal_impact_rate(trajectories['outcomes'],flux.importance_weight,
        temperature=temperature,reduced_mass=MA*MB/(MA+MB),bmax=bmax,
        electronic_weight=electronic_weight,symmetry_factor=symmetry_factor,
        impact_parameters=flux.impact_parameter)
    rate.update({'atoms':len(m),'beads':q.shape[1],'method':'incoming-flux Cartesian RPMD',
                 'provenance':dict(provenance),'physical_converged':False})
    return {'rate':rate,'trajectories':trajectories,'flux':flux}
