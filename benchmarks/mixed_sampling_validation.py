"""Independent linear-operator references for mixed sampling and BCOCB-RESPA.

Only general matrix exponentials, Cayley solves, square roots and direct
trigonometric bases are used here. No production integration coefficients are
read. MTS invariant covariance is NOT assumed equal to the exact Gibbs law.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import block_diag, expm, sqrtm, solve_discrete_lyapunov


def real_modes(beads):
    j = np.arange(beads)
    cols = [np.ones(beads) / math.sqrt(beads)]
    labels = [('constant', 0)]
    for k in range(1, (beads + 1) // 2):
        cols += [np.sqrt(2 / beads) * np.cos(2 * np.pi * k * j / beads),
                 np.sqrt(2 / beads) * np.sin(2 * np.pi * k * j / beads)]
        labels += [('cos', k), ('sin', k)]
    if beads % 2 == 0:
        cols += [(-1.) ** j / math.sqrt(beads)]
        labels += [('cos', beads // 2)]
    return np.column_stack(cols), labels


def contraction(beads, contracted):
    if beads == contracted:
        return np.eye(beads)
    u, labels = real_modes(beads)
    small, retain = real_modes(contracted)
    return math.sqrt(contracted / beads) * small @ u[:, [labels.index(x) for x in retain]].T


def real_sqrt(matrix):
    answer = sqrtm(matrix)
    if np.max(np.abs(np.imag(answer))) > 1e-10:
        raise ValueError('reference square root is not real')
    return np.asarray(answer.real, dtype=float)


def free_root(beads, beta, masses, dt, method='bcocb'):
    """Principal square root of the FULL h Cayley map, not cay(h*A/2)."""
    mass = np.tile(np.repeat(masses, 3), beads)
    atomdim = len(masses) * 3
    I = np.eye(beads)
    laplacian = 2 * I - np.roll(I, 1, 1) - np.roll(I, -1, 1)
    K = (beads / beta)**2 * np.kron(laplacian, np.diag(np.repeat(masses, 3)))
    zero = np.zeros_like(K)
    A = np.block([[zero, np.diag(1 / mass)], [-K, zero]])
    I = np.eye(len(A))
    return (real_sqrt(np.linalg.solve(I - dt * A / 2, I + dt * A / 2))
            if method == 'bcocb' else expm(dt * A / 2))


def projected_pile_step(q, p, masses, beta, dt, stiffness, fragments, noise,
                        method='bcocb', gamma_centroid=.7):
    """Independent Cartesian mass-metric projection at every substep.

    Anchors are zero COM. noise[P,N,3] is in the real orthonormal mode basis,
    like the legacy scalar NumPy PILE stream (not the Torch bead-noise stream).
    """
    P, N, _ = q.shape
    d = q.size
    mass = np.tile(np.repeat(masses, 3), P)
    M = np.diag(mass)
    inv = np.diag(1 / mass)
    rows = []
    for frag in fragments:
        for axis in range(3):
            row = np.zeros((P, N, 3))
            row[:, frag, axis] = np.asarray(masses)[frag]
            rows.append(row.ravel())
    C = np.asarray(rows)
    qp = np.eye(d) - inv @ C.T @ np.linalg.solve(C @ inv @ C.T, C)
    pp = np.eye(d) - C.T @ np.linalg.solve(C @ inv @ C.T, C @ inv)
    projector = block_diag(qp, pp)
    I = np.eye(d)
    B = np.block([[I, 0 * I], [-dt * stiffness * I / 2, I]])
    S = free_root(P, beta, masses, dt, method)
    U, labels = real_modes(P)
    omega = 2 * P / beta * np.sin(np.pi * np.array([x[1] for x in labels]) / P)
    gamma = 2 * omega
    gamma[0] = gamma_centroid
    decay = np.exp(-gamma * dt)
    D = np.kron((U * decay) @ U.T, np.eye(3 * N))
    kick_noise = np.kron(U * np.sqrt(-np.expm1(-2 * gamma * dt)), np.eye(3 * N))
    kick_noise = kick_noise @ (np.sqrt(mass * P / beta) * noise.ravel())
    phase = projector @ np.r_[q.ravel(), p.ravel()]
    phase = projector @ B @ phase
    phase = projector @ S @ phase
    phase = projector @ (block_diag(I, D) @ phase + np.r_[np.zeros(d), kick_noise])
    phase = projector @ S @ phase
    phase = projector @ B @ phase
    return phase[:d].reshape(q.shape), phase[d:].reshape(p.shape), projector


def mts_gaussian_map(beads, beta, mass, dt, inner, fast=.9, slow=.1, contracted=None,
                     method='bcocb', thermostat='pile-l', gamma_centroid=1.):
    """Return L,Q,H for a full outer step: x'=L x + H z, Cov(z)=I.

    A single active Cartesian coordinate suffices. H has M*P columns: a fresh
    independent noise field is used in each inner step. Q sums all propagated
    noise contributions; retaining just the final inner noise would be wrong.
    """
    P = beads
    h = dt / inner
    I = np.eye(P)
    laplacian = 2 * I - np.roll(I, 1, 1) - np.roll(I, -1, 1)
    spring = mass * (P / beta)**2 * laplacian
    A = np.block([[0 * I, I / mass], [-spring, 0 * I]])
    J = np.eye(2 * P)
    S = (real_sqrt(np.linalg.solve(J - h * A / 2, J + h * A / 2))
         if method == 'bcocb' else expm(h * A / 2))
    Bf = np.block([[I, 0 * I], [-h * fast * I / 2, I]])
    T = contraction(P, P if contracted is None else contracted)
    Ks = slow * (P / len(T)) * T.T @ T
    Bs = np.block([[I, 0 * I], [-dt * Ks / 2, I]])
    U, labels = real_modes(P)
    omega = 2 * P / beta * np.sin(np.pi * np.array([v[1] for v in labels]) / P)
    gamma = 2 * omega
    gamma[0] = gamma_centroid if thermostat == 'pile-l' else 0
    if thermostat is None:
        gamma[:] = 0
    D = expm(-h * (U * gamma) @ U.T)
    O = block_diag(I, D)
    # Spectral square root is evaluated with its exact nonnegative eigenvalues
    # to avoid an imaginary 1e-8 roundoff root for the undamped centroid.
    amplitude = (U * np.sqrt(-np.expm1(-2 * h * gamma))) @ U.T
    H0 = np.vstack([0 * I, np.sqrt(mass * P / beta) * amplitude])
    F = Bf @ S @ O @ S @ Bf
    G = Bf @ S @ H0
    L, H = Bs.copy(), np.zeros((2 * P, 0))
    for _ in range(inner):
        L = F @ L
        H = np.column_stack([F @ H, G])
    L, H = Bs @ L, Bs @ H
    return L, H @ H.T, H


def invariant_summary(**parameters):
    L, Q, _ = mts_gaussian_map(**parameters)
    radius = float(max(abs(np.linalg.eigvals(L))))
    if radius >= 1:
        return {'spectral_radius': radius, 'stationary_covariance': None}
    C = solve_discrete_lyapunov(L, Q)
    return {'spectral_radius': radius, 'stationary_covariance': C.tolist(),
            'lyapunov_residual': float(np.max(abs(C - L @ C @ L.T - Q)))}


def estimate(values):
    """Pointwise Student-t interval over independent replica observations."""
    from scipy.stats import t
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    half = float(t.ppf(.975, len(values)-1) * se)
    return {'mean': mean, 'se': se, 'ci95': [mean-half, mean+half],
            'independent_replicas': len(values), 'replica_values': values.tolist()}


def ideal_gas_npt(replicas=64, burn_time=80., production_time=400.):
    """Check the isotropic BZP volume law, including the N+1 Jacobian term.

    Replicas, not correlated saved times, are the uncertainty units. Internal
    ring modes do not contribute additional powers of V: only centroids scale.
    No COM constraint is applied. The exact volume density is V**N exp(-beta*p*V).
    """
    from pyrpmd.native_ensemble import NativeEnsemble
    beta, pressure, atoms = 2., .1, 2
    expected_mean = (atoms+1)/(beta*pressure)
    expected_second = (atoms+1)*(atoms+2)/(beta*pressure)**2
    rows = []
    for method in ('bcocb', 'baoab'):
        for P in (1,4):
            for dt in (.04,.02):
                moments = []
                for replica in range(replicas):
                    e = NativeEnsemble(
                        np.zeros((P,atoms,3)), [1.,2.], beta, dt,
                        cell=np.eye(3)*expected_mean**(1/3), potential={'kind':'free'},
                        thermostat={'mode':'pile-l','gamma_centroid':1.},
                        barostat={'mode':'bzp','pressure':pressure,'tau':.5,'gamma':1.},
                        integrator='normal-mode-bab' if method=='bcocb' else 'cayley-bcb',
                        sampling_integrator=method, seed=20261102+replica)
                    e.run(round(burn_time/dt))
                    samples=e.run(round(production_time/dt),stride=round(.2/dt))['samples']
                    volume=samples[:,6]
                    moments.append([float(volume.mean()),float(np.mean(volume**2))])
                values=np.array(moments)
                row={'method':method,'beads':P,'dt':dt,'beta':beta,'pressure':pressure,
                     'atoms':atoms,'burn_time':burn_time,'production_time':production_time,
                     'seeds':list(range(20261102,20261102+replicas)),
                     'volume':estimate(values[:,0]),'volume_second_moment':estimate(values[:,1]),
                     'exact_volume':expected_mean,'exact_second_moment':expected_second}
                rows.append(row)
                print('npt',method,P,dt,row['volume']['mean'],row['volume']['ci95'],flush=True)
    return {'reference':'Gamma(shape=N+1, scale=1/(beta*pressure)); no removed COM DOF','rows':rows}



def periodic_npt(replicas=64, burn_time=80., production_time=400.):
    """Nonzero periodic forces, P=1, exact fractional-coordinate partition.

    V(q,H)=sum h_j*(1-cos(2*pi*s_j))/2, s=q@H^-1.
    The partition separates V^N and bounded fractional integrals. Hence volume
    is still Gamma(N+1,1/(beta*p)), and <U> is independently given by Bessel
    functions. Do NOT extend that volume statement to P>1 with this potential.
    """
    from scipy.special import i0e,i1e
    from pyrpmd.native_ensemble import NativeEnsemble
    beta,pressure,atoms=2.,.1,2
    heights=np.array([.2,.3,.4])
    z=beta*heights/2
    exact_u=float(atoms*np.sum(heights/2*(1-i1e(z)/i0e(z))))
    exact_v=(atoms+1)/(beta*pressure)
    rows=[]
    for method in ('bcocb','baoab'):
        for dt in (.04,.02):
            observations=[]
            for replica in range(replicas):
                e=NativeEnsemble(np.zeros((1,atoms,3)),[1.,2.],beta,dt,
                    cell=np.eye(3)*exact_v**(1/3),
                    potential={'kind':'periodic_cosine','heights':heights},
                    thermostat={'mode':'pile-l','gamma_centroid':1.},
                    barostat={'mode':'bzp','pressure':pressure,'tau':.5,'gamma':1.},
                    integrator='normal-mode-bab' if method=='bcocb' else 'cayley-bcb',
                    sampling_integrator=method,seed=307+replica)
                e.run(round(burn_time/dt))
                samples=e.run(round(production_time/dt),stride=round(.2/dt))['samples']
                observations.append([float(samples[:,6].mean()),float(samples[:,2].mean())])
            v=np.array(observations)
            rows.append({'method':method,'beads':1,'dt':dt,'beta':beta,'pressure':pressure,
                         'atoms':atoms,'heights':heights.tolist(),'seeds':list(range(307,307+replicas)),
                         'burn_time':burn_time,'production_time':production_time,
                         'volume':estimate(v[:,0]),'potential':estimate(v[:,1]),
                         'exact_volume':exact_v,'exact_potential':exact_u})
            print('periodic-npt',method,dt,rows[-1]['volume']['mean'],rows[-1]['potential']['mean'],flush=True)
    return {'reference':'Gamma volume and analytic modified-Bessel fractional cosine partition, P=1 only','rows':rows}


def harmonic_mts_sampling(n=4096, duration=40.):
    import torch
    from pyrpmd.torch_backend import TorchDynamics,TorchPES,TorchRPCTerm,TorchSplitPotential,TorchState
    class Harmonic(torch.nn.Module):
        metadata={'batch_independent':True}
        def __init__(self,k):super().__init__();self.k=k
        def forward(self,q):return self.k*q.square().sum((1,2))/2,-self.k*q
    P,beta,mass=8,4.,1.7
    rows=[]
    for C in (1,3,8):
        for dt,M in ((.16,4),(.08,4),(.04,4)):
            potential=TorchSplitPotential([TorchRPCTerm(TorchPES(Harmonic(.9)),P)],
                                          [TorchRPCTerm(TorchPES(Harmonic(.1)),P,C)])
            d=TorchDynamics(potential,[mass],beta,P,dt,inner_steps=M,integrator='cayley-bcocb',
                            thermostat='pile-l',seed=302)
            zero=torch.zeros((n,P,1,3),dtype=torch.float64);state=TorchState(zero,zero)
            steps=round(duration/dt);d.propagate(state,steps)
            L,Q,_=mts_gaussian_map(P,beta,mass,dt,M,contracted=C)
            Cinf=solve_discrete_lyapunov(L,Q);Ln=np.linalg.matrix_power(L,steps)
            Ct=Cinf-Ln@Cinf@Ln.T
            I=np.eye(P);T=contraction(P,C)
            K=mass*(P/beta)**2*(2*I-np.roll(I,1,1)-np.roll(I,-1,1))+.9*I+.1*P/C*T.T@T
            exact=float(np.trace(np.linalg.inv(K))/beta)
            q2=estimate(state.q.square().mean((1,2,3)).numpy())
            predicted=float(np.trace(Ct[:P,:P])/P)
            rows.append({'beads':P,'contracted':C,'beta':beta,'mass':mass,'dt':dt,'inner':M,
                         'steps':steps,'seed':302,'measured_q2':q2,'finite_time_q2':predicted,
                         'stationary_q2':float(np.trace(Cinf[:P,:P])/P),
                         'contracted_Gibbs_q2':exact,'z_finite_time':(q2['mean']-predicted)/q2['se'],
                         'spectral_radius':float(max(abs(np.linalg.eigvals(L)))),
                         'force_evaluations':d.force_evaluations})
            print('mts',C,dt,'z',rows[-1]['z_finite_time'],flush=True)
    return {'rows':rows,'uncertainty':'independent trajectories; Cartesian observations averaged within each trajectory'}


def fragment_sampling(n=128, duration=24., dt=.04):
    from pyrpmd.dynamics import Dynamics,State
    from pyrpmd.potentials import HarmonicPotential
    P,beta,masses,k,fragments=4,3.,np.array([1.,2.,5.]),.6,[[0,1],[2]]
    shape=(P,3,3);dim=np.prod(shape);zeros=np.zeros(shape)
    rows=[]
    for method in ('bcocb','baoab'):
        columns=[]
        for column in np.eye(2*dim):
            q,p,_=projected_pile_step(column[:dim].reshape(shape),column[dim:].reshape(shape),
                                      masses,beta,dt,k,fragments,zeros,method)
            columns.append(np.r_[q.ravel(),p.ravel()])
        L=np.array(columns).T;columns=[]
        for column in np.eye(dim):
            q,p,_=projected_pile_step(zeros,zeros,masses,beta,dt,k,fragments,column.reshape(shape),method)
            columns.append(np.r_[q.ravel(),p.ravel()])
        H=np.array(columns).T;Q=H@H.T;cov=solve_discrete_lyapunov(L,Q)
        steps=round(duration/dt);Ln=np.linalg.matrix_power(L,steps);finite=cov-Ln@cov@Ln.T
        samples=[];q_residual=p_residual=0.
        for replica in range(n):
            d=Dynamics(HarmonicPotential(k),masses,beta,P,dt,seed=8100+replica,native=False,
                       integrator='normal-mode-bab',loop_backend='python')
            s=State(zeros,zeros)
            d.thermalize(s,steps,thermostat='pile-l',method=method,gamma_centroid=.7,
                         fragments=fragments,fragment_reference=np.zeros((3,3)),resample=False)
            samples.append([np.mean(s.q**2),np.mean(s.p**2)])
            for frag in fragments:
                q_residual=max(q_residual,float(np.max(abs(np.sum(s.q[:,frag]*masses[None,frag,None],axis=(0,1))))))
                p_residual=max(p_residual,float(np.max(abs(s.p[:,frag].sum((0,1))))))
        samples=np.array(samples)
        rows.append({'method':method,'n':n,'steps':steps,'dt':dt,'beta':beta,'masses':masses.tolist(),
                     'fragments':fragments,'q2':estimate(samples[:,0]),'p2':estimate(samples[:,1]),
                     'finite_time_q2':float(np.trace(finite[:dim,:dim])/dim),
                     'finite_time_p2':float(np.trace(finite[dim:,dim:])/dim),
                     'max_mass_weighted_q_constraint':q_residual,'max_p_constraint':p_residual,
                     'seed_start':8100,'covariance_residual':float(np.max(abs(cov-L@cov@L.T-Q)))})
        print('fragments',method,rows[-1]['q2']['mean'],rows[-1]['finite_time_q2'],flush=True)
    return {'rows':rows,'reference':'independent projected Cartesian Gaussian map, cold-start finite-time covariance'}


def quartic_mts(n=256):
    import torch
    from torch_observable_validation import Oscillator,transfer_quartic,observables,summary
    from pyrpmd.torch_backend import TorchDynamics,TorchPES,TorchRPCTerm,TorchSplitPotential,TorchState
    P,beta=16,1.
    references=[transfer_quartic(P,beta,points=g) for g in (256,512)]
    rows=[]
    for dt,M in ((.02,1),(.08,4),(.04,4)):
        fast=TorchPES(Oscillator(stiffness=1.,quartic=0.))
        slow=TorchPES(Oscillator(stiffness=-1.,quartic=1.))
        d=TorchDynamics(TorchSplitPotential([TorchRPCTerm(fast,P)],[TorchRPCTerm(slow,P)]),
                        [1.],beta,P,dt,inner_steps=M,integrator='cayley-bcocb',thermostat='pile-l',seed=2201)
        zero=torch.zeros((n,P,1,3),dtype=torch.float64);s=TorchState(zero,zero)
        d.propagate(s,round(16/dt));accum={}
        for sample in range(400):
            d.propagate(s,round(.16/dt))
            for key,value in observables(s.q[:,:,0,0].numpy(),beta).items():
                if key not in accum:accum[key]=np.zeros(n)
                accum[key]+=value/400
        rows.append({'dt':dt,'inner':M,'beads':P,'beta':beta,'n':n,'seed':2201,
                     'burn_time':16.,'production_time':64.,'sample_interval':.16,
                     'observables':{k:summary(v) for k,v in accum.items()},'force_evaluations':d.force_evaluations})
        print('quartic',dt,M,rows[-1]['observables']['kinetic_virial'],flush=True)
    return {'reference':references,'rows':rows,'split':'Vfast=x^2/2; Vslow=x^4/4-x^2/2; Pc=P',
            'uncertainty':'Student-t over independent trajectory time means, not individual time points'}


def main():
    import argparse,hashlib,inspect,json,platform,time
    from pathlib import Path
    import torch
    from pyrpmd.torch_backend import TorchDynamics
    from pyrpmd.dynamics import Dynamics
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--section',choices=['all','npt','periodic-npt','mts','fragments','quartic'],default='all')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(1)
    start=time.perf_counter()
    result={'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                           'cuda_available':torch.cuda.is_available(),'torch_threads':torch.get_num_threads()},
            'source_hashes':{str(cls):hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                             for cls in (Dynamics,TorchDynamics)},
            'scope':'CPU numerical validation; no chemistry or GPU performance claim'}
    for name,fn in [('npt',ideal_gas_npt),('periodic-npt',periodic_npt),('mts',harmonic_mts_sampling),('fragments',fragment_sampling),('quartic',quartic_mts)]:
        if args.section in ('all',name):
            result[name]=fn();result['elapsed_seconds']=time.perf_counter()-start
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
