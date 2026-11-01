"""Independent observable checks for the experimental Torch backend.

References: Korol et al., JCP 151, 124103 (2019), Eqs. 12/27/29 and Fig. 4c;
Korol et al., JCP 152, 104102 (2020), Eqs. 20-23/28/45.
This is NOT a digitization of their figures or a water/chemistry benchmark.
Run: PYTHONPATH=src python benchmarks/torch_observable_validation.py --section all --output result.json
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
from scipy.linalg import eigh, eigh_tridiagonal, expm, solve_discrete_lyapunov
from scipy.stats import t as student_t
import torch
from pyrpmd.torch_backend import TorchPES, TorchRPCTerm, TorchSplitPotential, TorchState, TorchDynamics

SCHEMES = ("normal-mode-bab", "cayley-bcb")


class Oscillator(torch.nn.Module):
    """One active coordinate; all other Cartesian forces are exactly zero."""
    metadata = {"batch_independent": True}

    def __init__(self, stiffness=0., quartic=1.):
        super().__init__()
        self.stiffness, self.quartic = stiffness, quartic

    def forward(self, q):
        x = q[:, 0, 0]
        force = torch.zeros_like(q)
        force[:, 0, 0] = -self.stiffness*x - self.quartic*x**3
        return .5*self.stiffness*x*x + .25*self.quartic*x**4, force


def frequencies(P, beta):
    """Full real-mode frequencies; degeneracies explicitly represented."""
    return 2*P/beta*np.sin(np.pi*np.arange(P)/P)


def state_1d(q, p, device="cpu"):
    Q = np.zeros((*q.shape, 1, 3)); P = np.zeros_like(Q)
    Q[:, :, 0, 0], P[:, :, 0, 0] = q, p
    return TorchState.from_arrays(Q, P, device=device)


def summary(values, confidence=.95):
    """One independent value per trajectory; never count beads/times as IID."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1)/math.sqrt(n)
    half = student_t.ppf((1+confidence)/2, n-1)*se
    return {"mean": mean.tolist(), "se": se.tolist(),
            "ci": np.stack((mean-half, mean+half), axis=-1).tolist(), "n": n}


def harmonic_iid(n, P, beta, rng):
    omega = frequencies(P, beta)[:P//2+1]
    q = np.fft.irfft(np.fft.rfft(rng.normal(size=(n, P)), norm="ortho") *
                      np.sqrt(P/beta/(1+omega*omega)), n=P, norm="ortho")
    return q, rng.normal(size=q.shape)*math.sqrt(P/beta)


def quartic_iid(n, P, beta, rng):
    """Exact finite-P Gibbs rejection from the harmonic ring, not MD equilibration."""
    chunks = []; accepted = proposed = 0
    while accepted < n:
        if proposed > 1000*n:
            raise RuntimeError("rejection sampler exhausted its proposal budget")
        q, _ = harmonic_iid(max(128, 2*(n-accepted)), P, beta, rng)
        keep = np.log(rng.random(len(q))) < -.25*beta*np.mean((q*q-1)**2, axis=1)
        chunks.append(q[keep]); accepted += int(keep.sum()); proposed += len(q)
    q = np.concatenate(chunks)[:n]
    return q, rng.normal(size=q.shape)*math.sqrt(P/beta), accepted/proposed


def quantum_quartic(beta=1., points=2048, extent=8., levels=80):
    """Independent 1D Schrodinger finite-difference reference (hbar=m=1)."""
    dx = 2*extent/(points+1)
    x = np.linspace(-extent+dx, extent-dx, points)
    V = .25*x**4
    E, U = eigh_tridiagonal(1/dx**2+V, np.full(points-1, -.5/dx**2),
                            select="i", select_range=(0, levels-1))
    weight = np.exp(-beta*(E-E[0])); weight /= weight.sum()
    potential = float(weight @ ((U*U).T @ V))
    return {"points": points, "extent": extent, "levels": levels,
            "q2": float(weight @ ((U*U).T @ x**2)),
            "potential": potential, "kinetic": float(weight@E-potential),
            "total_energy": float(weight@E), "ground_energy": float(E[0]),
            "highest_level_weight": float(weight[-1])}


def transfer_quartic(P, beta=1., points=512, extent=6.):
    """Finite-P transfer operator quadrature, independent of dynamics and RPC.

    Tr K**P gives the primitive finite-P path integral. Coordinate correlations
    use Tr[A K**r B K**(P-r)], with exact matrix powers in its eigenbasis.
    Grid/truncation convergence must be checked separately from statistical CI.
    """
    dx = 2*extent/points
    x = -extent + (np.arange(points)+.5)*dx
    V, g = .25*x**4, x**3
    K = np.exp(-P/(2*beta)*(x[:, None]-x[None, :])**2
               -beta/(2*P)*(V[:, None]+V[None, :])) * dx*math.sqrt(P/(2*np.pi*beta))
    eig, U = eigh(K)
    largest = eig[-1]
    if eig[0] < -1e-12*largest:
        raise ValueError("transfer quadrature is not positive semidefinite")
    keep = eig > 1e-14*largest
    eig, U = eig[keep]/largest, U[:, keep]
    weights = eig**P; Z = weights.sum(); weights /= Z
    q2 = float(weights @ ((U*U).T @ (x*x)))
    potential = float(weights @ ((U*U).T @ V))
    A, B = U.T @ (g[:, None]*U), U.T @ (x[:, None]*U)
    AB = A*B.T
    correlations = []
    for r in range(P):
        correlations.append(float(np.sum(eig[:, None]**(P-r)*eig[None, :]**r*AB)/Z))
    qg = float(weights @ ((U*U).T @ (x*g)))
    kinetic_virial = .5/beta + .5*(qg-np.mean(correlations))
    qq1 = float(np.sum(eig[:, None]**(P-1)*eig[None, :]*(B*B.T))/Z)
    kinetic_primitive = P/(2*beta)-P*P/(beta*beta)*(q2-qq1)
    return {"beads": P, "beta": beta, "points": points, "extent": extent,
            "retained_eigenvalues": len(eig), "q2": q2, "potential": potential,
            "kinetic_virial": float(kinetic_virial),
            "kinetic_primitive": float(kinetic_primitive)}


def observables(q, beta):
    P = q.shape[1]
    centroid = q.mean(axis=1, keepdims=True)
    return {"q2": np.mean(q*q, axis=1), "potential": .25*np.mean(q**4, axis=1),
            "kinetic_virial": .5/beta+.5*np.mean((q-centroid)*q**3, axis=1),
            "kinetic_primitive": P/(2*beta)-.5*(P/beta)**2*np.mean((q-np.roll(q, 1, axis=1))**2, axis=1)}


def reference_section(n=16384):
    out = {"quantum_grid_convergence": [quantum_quartic(points=p) for p in (1024, 2048, 4096)],
           "finite_P": []}
    for P in (1, 8, 16, 54):
        grids = [transfer_quartic(P, points=p) for p in (256, 512)]
        q, _, acceptance = quartic_iid(n, P, 1., np.random.default_rng(11000+P))
        samples = {k: summary(v) for k, v in observables(q, 1.).items()}
        out["finite_P"].append({"beads": P, "seed": 11000+P, "quadrature": grids,
                                "iid_rejection": samples, "acceptance": acceptance})
    return out


def correlation_section(n=2048, duration=10., device="cpu", seed=20261101):
    P, beta, interval = 54, 1., .1
    q, p, acceptance = quartic_iid(n, P, beta, np.random.default_rng(seed))
    q0 = q.mean(1)
    configs = [("normal-mode-bab", .0025), ("normal-mode-bab", .005), ("normal-mode-bab", .01),
               ("cayley-bcb", .01), ("cayley-bcb", .05), ("cayley-bcb", .1)]
    products = []; maximum_drifts = []
    for scheme, dt in configs:
        d = TorchDynamics(TorchPES(Oscillator(), device=device), [1.], beta, P, dt, integrator=scheme)
        s = state_1d(q, p, device); E0 = d.energy(s); drift = torch.zeros(n, dtype=torch.float64, device=device)
        values = [q0*q0]
        for _ in range(round(duration/interval)):
            d.propagate(s, round(interval/dt))
            values.append(s.q[:, :, 0, 0].mean(1).cpu().numpy()*q0)
            drift = torch.maximum(drift, (d.energy(s)/E0-1).abs())
        products.append(np.array(values).T)
        maximum_drifts.append(float(drift.max().item()))
    scale = float(np.mean(q0*q0))
    rows = []
    for i, (scheme, dt) in enumerate(configs):
        delta = products[i]-products[0]
        stat = summary(delta)
        # Bonferroni simultaneous intervals over all compared times AND all
        # non-reference methods; retain the paired ensemble in each difference.
        family = max(1, (len(configs)-1)*len(delta[0]))
        critical = student_t.ppf(1-.05/(2*family), n-1)
        upper = np.abs(np.asarray(stat["mean"])) + critical*np.asarray(stat["se"])
        rows.append({"scheme": scheme, "dt": dt, "correlation": summary(products[i]),
                     "paired_difference": stat, "rms_mean_difference_over_C0": float(np.sqrt(np.mean(delta.mean(0)**2))/scale),
                     "max_mean_difference_over_C0": float(np.max(abs(delta.mean(0)))/scale),
                     "simultaneous_95_upper_absolute_difference_over_C0": float(np.max(upper)/scale),
                     "max_relative_energy_drift_at_observations": maximum_drifts[i]})
    return {"seed": seed, "trajectories": n, "beads": P, "beta": beta, "mass": 1.,
            "potential": "x**4/4", "thermostat": None, "time": (np.arange(round(duration/interval)+1)*interval).tolist(),
            "interval": interval, "acceptance": acceptance, "C0": scale, "rows": rows,
            "reference": "paired normal-mode BAB dt=0.0025; not exact quantum dynamics or digitized paper data"}


def mode_map(omega, dt, mass, stiffness, scheme):
    A = np.array([[0., 1/mass], [-mass*omega**2, 0.]])
    free = (expm(dt*A) if scheme == "normal-mode-bab" else
            np.linalg.solve(np.eye(2)-dt*A/2, np.eye(2)+dt*A/2))
    kick = np.array([[1., 0.], [-dt*stiffness/2, 1.]])
    return kick @ free @ kick


def mode_covariance(omega, dt, mass, stiffness, P, beta, scheme):
    M = mode_map(omega, dt, mass, stiffness, scheme)
    gamma = 1. if abs(omega) < 1e-14 else 2*omega
    decay = math.exp(-gamma*dt/2)
    O = np.diag([1., decay]); R = np.diag([0., mass*P/beta*(-math.expm1(-gamma*dt))])
    L = O@M@O; Q = O@M@R@M.T@O + R
    if max(abs(np.linalg.eigvals(L))) >= 1:
        raise ValueError("mode has no attracting stationary covariance")
    return solve_discrete_lyapunov(L, Q), L, Q


def paper_mode_covariance(omega, dt, mass, stiffness, P, beta, scheme):
    physical = stiffness/mass
    if scheme == "cayley-bcb":
        q2 = P/(beta*mass*(physical+omega**2))/(1-dt*dt*physical/4)
    else:
        theta = dt*omega
        theta_cot = 1. if abs(theta) < 1e-14 else theta/math.tan(theta)
        q2 = P/(beta*mass)/(omega**2+physical*theta_cot-(dt*physical/2)**2)
    return np.diag([q2, mass*P/beta])


def dimension_section(n=1024, device="cpu"):
    """OBCBO is stable but NOT dimension-free; compare published analytic bias."""
    beta, dt, steps = 8., .5, 1600
    rows = []
    for P in (8, 32, 128, 256):
        omega = frequencies(P, beta)
        exact = P/beta/(1+omega*omega)
        discrete = exact/(1-dt*dt/4)
        finite = []
        for w in omega:
            C, L, _ = mode_covariance(w, dt, 1., 1., P, beta, "cayley-bcb")
            Ln = np.linalg.matrix_power(L, steps)
            finite.append((C-Ln@C@Ln.T)[0, 0])
        finite = np.array(finite)
        def expected(C):
            return {"q2": float(C.sum()/P),
                    "kinetic_primitive": float(P/(2*beta)-np.sum(omega*omega*C)/(2*P)),
                    "kinetic_virial": float(1/(2*beta)+(C[1:].sum())/(2*P))}
        d = TorchDynamics(TorchPES(Oscillator(stiffness=1., quartic=0.), device=device),
                          [1.], beta, P, dt, integrator="cayley-bcb", thermostat="pile-l", seed=5000+P)
        q = np.zeros((n, P)); s = state_1d(q, q, device); d.propagate(s, steps)
        x = s.q[:, :, 0, 0].cpu().numpy()
        measured = {"q2": np.mean(x*x, axis=1),
                    "kinetic_primitive": P/(2*beta)-.5*(P/beta)**2*np.mean((x-np.roll(x, 1, axis=1))**2, axis=1),
                    "kinetic_virial": .5/beta+.5*np.mean((x-x.mean(1, keepdims=True))*x, axis=1)}
        rows.append({"beads": P, "seed": 5000+P, "steps": steps,
                     "finite_P_exact": expected(exact), "paper_stationary": expected(discrete),
                     "finite_time_independent": expected(finite),
                     "sampled": {k: summary(v) for k, v in measured.items()}})
    return {"beta": beta, "dt": dt, "trajectories": n, "scheme": "OBCBO",
            "potential": "x**2/2", "cold_start": True, "rows": rows}


def sampling_section(n=256, device="cpu"):
    P, beta, equil_time, production_time, interval = 16, 1., 20., 100., .2
    reference = transfer_quartic(P)
    rows = []
    for scheme in SCHEMES:
        for dt in (.02, .1):
            d = TorchDynamics(TorchPES(Oscillator(), device=device), [1.], beta, P, dt,
                              integrator=scheme, thermostat="pile-l", seed=717)
            zero = np.zeros((n, P)); s = state_1d(zero, zero, device)
            d.propagate(s, round(equil_time/dt))
            sums = {k: np.zeros(n) for k in observables(zero, beta)}
            for _ in range(round(production_time/interval)):
                d.propagate(s, round(interval/dt))
                obs = observables(s.q[:, :, 0, 0].cpu().numpy(), beta)
                for k, v in obs.items(): sums[k] += v
            stats = {k: summary(v/round(production_time/interval)) for k, v in sums.items()}
            rows.append({"scheme": scheme, "dt": dt, "seed": 717,
                         "measured": stats,
                         "z_vs_finite_P": {k: (s["mean"]-reference[k])/s["se"] for k, s in stats.items()}})
    return {"beads": P, "beta": beta, "trajectories": n, "equil_time": equil_time,
            "production_time": production_time, "observation_interval": interval,
            "reference": reference, "rows": rows,
            "uncertainty": "standard error across independent trajectory means, not across correlated time samples"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("references", "correlation", "dimension", "sampling", "all"), default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", type=int)
    args = parser.parse_args()
    if args.trajectories is not None and args.trajectories < 2:
        parser.error("--trajectories must be at least 2")
    torch.set_num_threads(1)
    import pyrpmd.torch_backend as backend
    result = {"environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "torch": torch.__version__, "device": args.device, "threads": 1,
                               "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "backend_sha256": hashlib.sha256(Path(inspect.getfile(backend)).read_bytes()).hexdigest()}}
    functions = {"references": (reference_section, 16384), "correlation": (correlation_section, 2048),
                 "dimension": (dimension_section, 1024), "sampling": (sampling_section, 256)}
    start = time.perf_counter()
    for name, (fn, n) in functions.items():
        if args.section in (name, "all"):
            kwargs = {} if name == "references" else {"device": args.device}
            result[name] = fn(n=n if args.trajectories is None else args.trajectories, **kwargs)
            result["seconds"] = time.perf_counter()-start
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
            print(f"{name}: complete; {args.output}", flush=True)


if __name__ == "__main__":
    main()
