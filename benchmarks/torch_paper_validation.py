"""Reproducible paper/analytic validation; see docs/torch_validation_report.md.
PYTHONPATH=src python benchmarks/torch_paper_validation.py --output results.json
No trained-potential, rate-workflow, or GPU speedup claim is made.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import math
import platform
from pathlib import Path
import time
import numpy as np
from scipy.linalg import expm, solve_discrete_lyapunov
import torch
from pyrpmd.torch_backend import TorchPES, TorchRPCTerm, TorchSplitPotential, TorchState, TorchDynamics

SCHEMES = ('normal-mode-bab', 'cayley-bcb')

class Polynomial(torch.nn.Module):
    """One active Cartesian coordinate, exactly the 1D paper model."""
    metadata = {'batch_independent': True}
    def __init__(self, quadratic=1., cubic=0., quartic=0.):
        super().__init__()
        self.a, self.b, self.c = quadratic, cubic, quartic
        self.calls = self.frames = 0
    def forward(self, q):
        self.calls += 1
        self.frames += len(q)
        x = q[:, 0, 0]
        energy = .5*self.a*x.square() + self.b*x**3 + .25*self.c*x**4
        force = torch.zeros_like(q)
        force[:, 0, 0] = -(self.a*x + 3*self.b*x*x + self.c*x**3)
        return energy, force

class IsotropicHarmonic(torch.nn.Module):
    metadata = {'batch_independent': True}
    def __init__(self, stiffness=1.):
        super().__init__()
        self.stiffness = stiffness
    def forward(self, q):
        return .5*self.stiffness*q.square().sum((1, 2)), -self.stiffness*q


def frequencies(beads, beta):
    return 2*beads/beta*np.sin(np.pi*np.arange(beads//2+1)/beads)


def harmonic_sample(n, beads, beta, rng, stiffness=1.):
    omega = frequencies(beads, beta)
    x = np.fft.irfft(np.fft.rfft(rng.normal(size=(n, beads)), norm='ortho') *
                     np.sqrt(beads/beta/(stiffness+omega**2)), n=beads, norm='ortho')
    p = rng.normal(size=x.shape)*np.sqrt(beads/beta)
    return x, p


def quartic_sample(n, beads, beta, rng):
    """IID exact finite-P Gibbs positions by Gaussian-proposal rejection.
    Vquartic - Vharmonic + 1/4 = (x^2-1)^2/4 >= 0; spring action cancels.
    """
    batches = []
    accepted = proposed = 0
    while accepted < n:
        if proposed > 1000*n:
            raise RuntimeError('quartic rejection sampler exhausted its proposal budget')
        x, _ = harmonic_sample(max(2*(n-accepted), 128), beads, beta, rng)
        log_accept = -.25*beta*np.mean((x*x-1.)**2, axis=1)
        keep = np.log(rng.uniform(size=len(x))) < log_accept
        batches.append(x[keep]); accepted += int(keep.sum()); proposed += len(x)
    x = np.concatenate(batches)[:n]
    p = rng.normal(size=x.shape)*np.sqrt(beads/beta)
    return x, p, accepted/proposed


def state_1d(x, p, device='cpu'):
    q3 = np.zeros((*x.shape, 1, 3)); p3 = np.zeros_like(q3)
    q3[:, :, 0, 0] = x; p3[:, :, 0, 0] = p
    return TorchState.from_arrays(q3, p3, device=device)


def wilson(successes, n):
    z = 1.959963984540054; f = successes/n; den = 1+z*z/n
    center = (f+z*z/(2*n))/den
    half = z*math.sqrt(f*(1-f)/n+z*z/(4*n*n))/den
    return [max(0., center-half), min(1., center+half)]


def stability_case(kind, beads, dt, seed, n=1000, duration=100., device='cpu'):
    rng = np.random.default_rng(seed); beta = 1.
    if kind == 'harmonic':
        x, p = harmonic_sample(n, beads, beta, rng); acceptance = None
    else:
        x, p, acceptance = quartic_sample(n, beads, beta, rng)
    results = []
    for scheme in SCHEMES:
        model = Polynomial(quadratic=float(kind == 'harmonic'), quartic=float(kind == 'quartic'))
        dyn = TorchDynamics(TorchPES(model, device=device), [1.], beta, beads, dt, integrator=scheme)
        state = state_1d(x, p, device); initial = dyn.energy(state)
        maximum = torch.zeros(n, dtype=torch.float64, device=device)
        steps = int(math.floor(duration/dt+1e-9)); active = torch.arange(n, device=device)
        for _ in range(steps):
            dyn.step(state)
            error = ((dyn.energy(state)-initial)/initial).abs()
            maximum[active] = torch.maximum(maximum[active], error)
            # Censor at the paper's first failure, before overflow. Models are
            # independent; dropping failed rows cannot affect surviving rows.
            keep = error <= .1
            if not keep.all().item():
                active = active[keep]
                if len(active) == 0: break
                initial = initial[keep]
                state = TorchState(state.q[keep], state.p[keep], state.step, state.time)
        failed = n-len(active)
        results.append(dict(potential=kind, beads=beads, beta=beta, mass=1., dt=dt,
                            scheme=scheme, seed=seed, trajectories=n, steps=steps,
                            simulated_time=steps*dt, failed=failed, failure_fraction=failed/n,
                            failure_wilson95=wilson(failed,n),
                            quartic_proposal_acceptance=acceptance,
                            max_relative_energy_at_censor_or_end=float(maximum.max().item())))
    return results


def free_matrix(omega, h, scheme):
    """Independent 2x2 q,p update (m=1), not production coefficients."""
    A = np.array([[0.,1.],[-omega*omega,0.]])
    if scheme == 'cayley-bcb':
        return np.linalg.solve(np.eye(2)-.5*h*A, np.eye(2)+.5*h*A)
    return expm(h*A)


def harmonic_matrix(omega, h, scheme, stiffness=1.):
    B = np.array([[1.,0.],[-.5*h*stiffness,1.]])
    return B @ free_matrix(omega,h,scheme) @ B


def harmonic_independent_check(seed=1001, n=1000, device="cpu"):
    """Cross-check paper-condition trajectories using independent matrices."""
    P, beta, dt, steps = 16, 1., .1, 1000
    x, p = harmonic_sample(n,P,beta,np.random.default_rng(seed)); omega = frequencies(P,beta)
    weights = np.full(len(omega),2.); weights[[0,-1]]=1.; out=[]
    for scheme in SCHEMES:
        matrices = np.array([harmonic_matrix(w,dt,scheme) for w in omega])
        xp = np.stack([np.fft.rfft(x,norm='ortho'),np.fft.rfft(p,norm='ortho')],axis=-1)
        def energy(v):
            return .5*np.sum(weights*((1+omega**2)*abs(v[:,:,0])**2+abs(v[:,:,1])**2),axis=1)
        initial=energy(xp); maximum=np.zeros(n)
        for _ in range(steps):
            xp=np.einsum('kij,bkj->bki',matrices,xp)
            maximum=np.maximum(maximum,abs(energy(xp)/initial-1))
        dyn=TorchDynamics(TorchPES(Polynomial(),device=device),[1.],beta,P,dt,integrator=scheme)
        s=state_1d(x[:32],p[:32],device);dyn.propagate(s,steps)
        xr=np.fft.irfft(xp[:32,:,0],n=P,norm='ortho'); pr=np.fft.irfft(xp[:32,:,1],n=P,norm='ortho')
        error=max(np.max(abs(s.q[:,:,0,0].cpu().numpy()-xr)),np.max(abs(s.p[:,:,0,0].cpu().numpy()-pr)))
        out.append(dict(scheme=scheme,seed=seed,trajectories=n,failed=int((maximum>.1).sum()),
                        max_state_error=float(error),max_mode_spectral_radius=float(max(
                        max(abs(np.linalg.eigvals(M))) for M in matrices))))
    return out


def stationary_mode(omega, beta, P, dt, scheme, gamma_centroid=1.):
    M=harmonic_matrix(omega,dt,scheme)
    gamma=gamma_centroid if omega == 0 else 2*omega
    c=math.exp(-.5*dt*gamma); O=np.diag([1.,c]); R=np.diag([0.,P/beta*(1-c*c)])
    L=O@M@O; Q=O@M@R@M.T@O.T+R
    return solve_discrete_lyapunov(L,Q), L, Q


def sampling_case(P, dt, scheme, seed=7001, n=1000, beta=8., time_total=32., device='cpu'):
    dyn=TorchDynamics(TorchPES(IsotropicHarmonic(),device=device),[1.],beta,P,dt,
                      integrator=scheme,thermostat='pile-l',seed=seed)
    zeros=torch.zeros((n,P,1,3),dtype=torch.float64,device=device)
    state=TorchState(zeros,zeros)  # Cold start, not a pre-loaded Gibbs ensemble.
    steps=round(time_total/dt); dyn.propagate(state,steps)
    q=state.q.cpu().numpy();p=state.p.cpu().numpy()
    # One observation per independent trajectory; Cartesian coordinates are
    # averaged within each trajectory before estimating its standard error.
    per_traj=(q*q).mean((1,2,3)); centroid=(q.mean(1)**2).mean((1,2))
    omega=frequencies(P,beta); weights=np.full(len(omega),2.); weights[0]=1.
    if P%2==0: weights[-1]=1.
    predicted=[]; finite_time=[]
    for w in omega:
        C,L,_=stationary_mode(w,beta,P,dt,scheme); Ln=np.linalg.matrix_power(L,steps)
        predicted.append(C[0,0]); finite_time.append((C-Ln@C@Ln.T)[0,0])
    true_q=float(np.dot(weights,P/beta/(1+omega**2))/P)
    numeric_q=float(np.dot(weights,predicted)/P); finite_q=float(np.dot(weights,finite_time)/P)
    sample=float(per_traj.mean());se=float(per_traj.std(ddof=1)/math.sqrt(n))
    return dict(beads=P,beta=beta,dt=dt,scheme=scheme,seed=seed,trajectories=n,
                initialization='q=p=0',steps=steps,simulated_time=steps*dt,
                q2=sample,q2_se=se,q2_ci95=[sample-1.96*se,sample+1.96*se],
                finite_P_q2=true_q,quantum_limit_q2=.5/math.tanh(beta/2),
                discrete_stationary_q2=numeric_q,discrete_finite_time_q2=finite_q,
                bias_percent=100*(numeric_q/true_q-1),z_vs_finite_time=(sample-finite_q)/se,
                centroid_q2=float(centroid.mean()),centroid_reference=1/beta,
                momentum_q2=float((p*p).mean()),momentum_reference=P/beta)


def real_basis(P):
    """Independent real basis: centroid, cos(1), sin(1), ..., Nyquist."""
    j=np.arange(P);cols=[np.ones(P)/math.sqrt(P)];labels=[('const',0)]
    for k in range(1,(P+1)//2):
        cols.extend([np.sqrt(2/P)*np.cos(2*np.pi*k*j/P),np.sqrt(2/P)*np.sin(2*np.pi*k*j/P)])
        labels.extend([('cos',k),('sin',k)])
    if P%2==0: cols.append((-1.)**j/math.sqrt(P));labels.append(('cos',P//2))
    return np.column_stack(cols),labels


def reference_contraction(P,C):
    if P==C:return np.eye(P)
    up,labels=real_basis(P);uc,small=real_basis(C)
    return math.sqrt(C/P)*uc@up[:,[labels.index(k) for k in small]].T


def rpc_sweep(device="cpu"):
    model=TorchPES(IsotropicHarmonic(),device=device);errors=[];count=0
    for P in (1,2,3,4,5,8,16,31,32):
        for C in range(1,P+1):
            actual=TorchRPCTerm(model,P,C).matrix.cpu().numpy(); ref=reference_contraction(P,C)
            errors.append(np.max(abs(actual-ref)));count+=1
    P,beta,kfast,kslow=32,8.,.9,.1;eye=np.eye(P)
    L=2*eye-np.roll(eye,1,axis=1)-np.roll(eye,-1,axis=1)
    full=np.linalg.inv((P/beta)**2*L+(kfast+kslow)*eye)*(P/beta)
    true=float(np.trace(full)/P);rows=[]
    for C in (1,2,3,4,8,16,32):
        T=reference_contraction(P,C);K=(P/beta)**2*L+kfast*eye+kslow*(P/C)*T.T@T
        q2=float(np.trace(np.linalg.inv(K))*(P/beta)/P)
        rows.append(dict(beads=P,contracted_beads=C,finite_P_q2=true,contracted_q2=q2,
                         rpc_bias_percent=100*(q2/true-1),slow_frames_per_ring=C))
    return dict(matrix_cases=count,max_projection_error=float(max(errors)),harmonic_bias=rows,
                note='Analytic contracted-Hamiltonian check, not a liquid-water reproduction')


def mts_sweep(device="cpu"):
    P,beta,kfast,kslow,C,Tfinal=8,4.,.9,.1,3,2.
    q,p=harmonic_sample(4,P,beta,np.random.default_rng(91)); transform=reference_contraction(P,C);I=np.eye(P)
    K=(P/beta)**2*(2*I-np.roll(I,1,1)-np.roll(I,-1,1))+kfast*I+kslow*P/C*transform.T@transform
    A=np.block([[np.zeros_like(I),I],[-K,np.zeros_like(I)]])
    expected=np.concatenate([q,p],axis=1)@expm(Tfinal*A).T;out=[]
    for scheme in SCHEMES:
        for M in (1,2,4):
            rows=[]
            for dt in (.08,.04,.02):
                fast,slow=Polynomial(kfast),Polynomial(kslow)
                split=TorchSplitPotential([TorchRPCTerm(TorchPES(fast,device=device),P)], [TorchRPCTerm(TorchPES(slow,device=device),P,C)])
                d=TorchDynamics(split,[1.],beta,P,dt,inner_steps=M,integrator=scheme)
                s=state_1d(q,p,device);e0=d.energy(s);d.propagate(s,round(Tfinal/dt));e1=d.energy(s)
                actual=np.concatenate([s.q[:,:,0,0].cpu().numpy(),s.p[:,:,0,0].cpu().numpy()],axis=1)
                rows.append(dict(dt=dt,steps=round(Tfinal/dt),inner_steps=M,
                                 state_error=float(np.max(abs(actual-expected))),
                                 max_relative_energy_error=float(torch.max(abs(e1/e0-1))),
                                 force_evaluations=dict(d.force_evaluations)))
            out.append(dict(scheme=scheme,inner_steps=M,rows=rows,
                            error_ratios=[rows[i]['state_error']/rows[i+1]['state_error'] for i in (0,1)]))
    cost=[]
    for label,dt,M,c in [('small-step-full',.01,1,P),('mts-only',.04,4,P),('mts-rpc',.04,4,2)]:
        f,s=Polynomial(kfast),Polynomial(kslow)
        split=TorchSplitPotential([TorchRPCTerm(TorchPES(f,device=device),P)], [TorchRPCTerm(TorchPES(s,device=device),P,c)])
        d=TorchDynamics(split,[1.],beta,P,dt,inner_steps=M);d.propagate(state_1d(q,p,device),round(4/dt))
        cost.append(dict(case=label,trajectories=4,physical_time=4.,inner_dt=.01,
                         slow_calls=s.calls,slow_frames=s.frames,fast_calls=f.calls))
    return dict(convergence=out,equal_inner_step_cost=cost)


def resonance_case(device="cpu", n=4096, seed=1729):
    """2019 nonergodicity mechanism, not its gamma=omega Figure 6/7 protocol.
    The public backend uses gamma=2*omega internally. One cold-start snapshot
    per independent trajectory, with Cartesian squares averaged within it.
    """
    P, beta, dt = 8, 8., math.pi/2
    rows = []
    for scheme in SCHEMES:
        d=TorchDynamics(TorchPES(Polynomial(quadratic=0),device=device),[1.],beta,P,dt,
                        integrator=scheme,thermostat="trpmd",seed=seed)
        zero=torch.zeros((n,P,1,3),dtype=torch.float64,device=device)
        state=TorchState(zero,zero);d.propagate(state,100)
        mode=torch.fft.rfft(state.q,dim=1,norm="ortho")[:,-1,0,:].real
        values=mode.square().mean(1).cpu().numpy()
        mean=float(values.mean());se=float(values.std(ddof=1)/math.sqrt(n))
        rows.append(dict(scheme=scheme,beads=P,beta=beta,dt=dt,steps=100,seed=seed,
                         trajectories=n,internal_gamma_over_omega=2.,nyquist_omega=2.,
                         q2=mean,q2_se=se,q2_ci95=[mean-1.96*se,mean+1.96*se],
                         equilibrium_q2=.25,initialization="q=p=0"))
    return rows


def rpc_sampling_case(C=3, scheme="normal-mode-bab", device="cpu", n=1000,
                      P=8, beta=8., dt=.1, M=4, duration=32., seed=417):
    """Test MTS+RPC+PILE together against their independent Gaussian map.
    All three Cartesian components are confined. The exact finite-step
    covariance is distinct from the exact contracted Gibbs covariance.
    """
    fast=TorchRPCTerm(TorchPES(IsotropicHarmonic(.9),device=device),P)
    slow=TorchRPCTerm(TorchPES(IsotropicHarmonic(.1),device=device),P,C)
    dyn=TorchDynamics(TorchSplitPotential([fast],[slow]),[1.],beta,P,dt,
                      inner_steps=M,integrator=scheme,thermostat="pile-l",seed=seed)
    zero=torch.zeros((n,P,1,3),dtype=torch.float64,device=device)
    state=TorchState(zero,zero);steps=round(duration/dt);dyn.propagate(state,steps)
    values=state.q.square().mean((1,2,3)).cpu().numpy()
    U,_=real_basis(P);T=reference_contraction(P,C);I=np.eye(P)
    spring=(P/beta)**2*(2*I-np.roll(I,1,1)-np.roll(I,-1,1))
    omega2=np.maximum(np.diag(U.T@spring@U),0.)
    correction=np.diag(.1*(P/C)*U.T@T.T@T@U)
    continuous=[];stationary=[];finite=[]
    for k,(w2,ks) in enumerate(zip(omega2,correction)):
        w=math.sqrt(w2); kick=np.array([[1.,0.],[-.5*dt*ks,1.]])
        A=kick@np.linalg.matrix_power(harmonic_matrix(w,dt/M,scheme,.9),M)@kick
        gamma=1. if k==0 else 2*w
        c=math.exp(-gamma*dt/2);O=np.diag([1.,c]);R=np.diag([0.,P/beta*(1-c*c)])
        L=O@A@O;Q=O@A@R@A.T@O.T+R
        cov=solve_discrete_lyapunov(L,Q);Ln=np.linalg.matrix_power(L,steps)
        continuous.append(P/beta/(w2+.9+ks));stationary.append(cov[0,0])
        finite.append((cov-Ln@cov@Ln.T)[0,0])
    sample=float(values.mean());se=float(values.std(ddof=1)/math.sqrt(n))
    return dict(beads=P,contracted_beads=C,inner_steps=M,beta=beta,dt=dt,scheme=scheme,
                trajectories=n,seed=seed,steps=steps,initialization="q=p=0",q2=sample,
                q2_se=se,q2_ci95=[sample-1.96*se,sample+1.96*se],
                exact_contracted_q2=float(np.mean(continuous)),
                discrete_stationary_q2=float(np.mean(stationary)),
                discrete_finite_time_q2=float(np.mean(finite)),
                z_vs_finite_time=(sample-float(np.mean(finite)))/se)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path('torch-paper-results.json'))
    ap.add_argument('--section',choices=['all','stability','sampling','rpc-mts','resonance','rpc-sampling'],default='all')
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args();torch.set_num_threads(1)
    src=Path(inspect.getfile(TorchDynamics)).read_bytes()
    result={'tested_backend_sha256':hashlib.sha256(src).hexdigest(),
            'environment':{'python':platform.python_version(),'torch':torch.__version__,
                           'numpy':np.__version__,'cuda_available':torch.cuda.is_available(),
                           'device':args.device,'threads':torch.get_num_threads()},
            'references':{'Cayley':'10.1063/1.5120282','PILE':'10.1063/1.3489925',
                          'RPC':'10.1063/1.2953308','RESPA':'10.1063/1.463137'}}
    start=time.perf_counter()
    def save():
        result['elapsed_seconds']=time.perf_counter()-start
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if args.section in ('all','stability'):
        result['harmonic_matrix_reference']=harmonic_independent_check(device=args.device);result['stability']=[]
        cases=[('harmonic',16,.05,1001),('harmonic',16,.1,1001),('harmonic',16,.1,1002),
               ('harmonic',16,.1,1003),('quartic',54,.01,2001),('quartic',54,.029,2001),
               ('quartic',54,.05,2001),('quartic',54,.10,2001)]
        for kind,P,dt,seed in cases:
            rows=stability_case(kind,P,dt,seed,device=args.device)
            result['stability'].extend(rows);save()
            print('stability',kind,P,dt,seed,[(r['scheme'],r['failed']) for r in rows],flush=True)
    if args.section in ('all','sampling'):
        result['sampling']=[]
        for P,dt in [(1,.02),(8,.02),(16,.02),(32,.02),(16,.1),(16,.2)]:
            for scheme in SCHEMES:
                row=sampling_case(P,dt,scheme,device=args.device);result['sampling'].append(row);save()
                print('sampling',P,dt,scheme,'z=',row['z_vs_finite_time'],'bias%=',row['bias_percent'],flush=True)
    if args.section in ('all','rpc-mts'):
        result['rpc']=rpc_sweep(args.device);result['mts']=mts_sweep(args.device);save();print('rpc/mts complete',flush=True)
    if args.section in ('all','resonance'):
        result['resonance']=resonance_case(args.device);save()
    if args.section in ('all','rpc-sampling'):
        result['rpc_sampling']=[rpc_sampling_case(C,scheme,args.device)
                                for C in (1,3,8) for scheme in SCHEMES];save()
    save()

if __name__ == '__main__':main()
