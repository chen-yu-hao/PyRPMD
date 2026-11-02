"""BCOCB implementation checks and repeated observable protocols.

Korol et al., JCP 152, 104102 (2020), DOI 10.1063/1.5134810,
Eqs. 17, 22, 23. Independent SciPy square roots / Lyapunov equations,
not production coefficients. Quartic references use transfer quadrature.
Run from a checkout: PYTHONPATH=src python benchmarks/torch_bcocb_validation.py \
    --section all --output bcocb.json
These CPU observations are not GPU performance or chemistry validation.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from scipy.linalg import sqrtm, solve_discrete_lyapunov
from scipy.stats import t as student_t
import torch
from pyrpmd.torch_backend import TorchDynamics, TorchPES

_spec = importlib.util.spec_from_file_location(
    "observable_reference", Path(__file__).with_name("torch_observable_validation.py"))
obs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(obs)
BCOCB = "cayley-bcocb"


def independent_step(omega, dt, mass=1., stiffness=1., P=8, beta=8., gamma=1.):
    """Return conditional mean/noise maps using a general matrix square root.

    State is (q,p); noise is ONE standard real normal variate at the middle O.
    The central OU momentum variance is m*P/beta, not a fitted endpoint value.
    """
    I = np.eye(2)
    A = np.array([[0., 1/mass], [-mass*omega**2, 0.]])
    C = np.linalg.solve(I-dt*A/2, I+dt*A/2)
    S = np.real_if_close(sqrtm(C), tol=1000)
    if np.iscomplexobj(S):
        raise ValueError("unexpected complex principal Cayley square root")
    B = np.array([[1., 0.], [-dt*stiffness/2, 1.]])
    O = np.diag([1., math.exp(-gamma*dt)])
    noise = np.array([0., math.sqrt(mass*P/beta*(-math.expm1(-2*gamma*dt)))])
    L = B@S@O@S@B
    G = B@S@noise
    return L, np.outer(G, G), G, S


def paper_covariance(omega, dt, mass=1., stiffness=1., P=8, beta=8.):
    """Eqs. 22-23, a=0, converted from paper velocities to real-mass momenta."""
    if not 0 < dt*math.sqrt(stiffness/mass) < 2:
        raise ValueError("harmonic physical-force stability requires dt*sqrt(k/m)<2")
    return np.diag([P/(beta*(stiffness+mass*omega**2)),
                    mass*P/beta*(1-dt*dt*stiffness/(4*mass))])


def algebra_section():
    rows = []
    for mass, stiffness in ((1., .5), (1., 2.), (7., 2.)):
        for omega in (0., .7, 3., 17., 100.):
            for dt in (.025, .17, .5):
                for gamma in (.3, 2.7):
                    L, Q, _, S = independent_step(omega, dt, mass, stiffness, gamma=gamma)
                    C = solve_discrete_lyapunov(L, Q)
                    exact = paper_covariance(omega, dt, mass, stiffness)
                    scale = np.sqrt(np.outer(np.diag(exact), np.diag(exact)))
                    rows.append({"mass": mass, "stiffness": stiffness, "omega": omega,
                                 "dt": dt, "gamma": gamma,
                                 "scaled_covariance_error": float(np.max(abs(C-exact)/scale)),
                                 "scaled_stationarity_residual": float(np.max(abs(exact-L@exact@L.T-Q)/scale)),
                                 "spectral_radius": float(max(abs(np.linalg.eigvals(L))))})
    return {"cases": rows, "max_scaled_covariance_error": max(r["scaled_covariance_error"] for r in rows)}


def dimension_section(n=1024, device="cpu", steps=1600):
    """Repeat the prior cold-start high-P test, with BOTH splittings."""
    beta, dt = 8., .5
    rows = []
    for P in (8, 32, 128, 256):
        omega = obs.frequencies(P, beta)
        exact = P/beta/(1+omega**2)
        for scheme in ("cayley-bcb", BCOCB):
            stationary, finite, p_finite = [], [], []
            for w in omega:
                if scheme == BCOCB:
                    L, Q, _, _ = independent_step(w, dt, P=P, beta=beta,
                                                   gamma=1. if w == 0 else 2*w)
                    C = solve_discrete_lyapunov(L, Q)
                else:
                    C, L, Q = obs.mode_covariance(w, dt, 1., 1., P, beta, scheme)
                Ln = np.linalg.matrix_power(L, steps)
                Ct = C-Ln@C@Ln.T
                stationary.append(C[0, 0]); finite.append(Ct[0, 0]); p_finite.append(Ct[1, 1])
            def expected(variances):
                a = np.asarray(variances)
                return {"q2": float(np.mean(a)),
                        "kinetic_primitive": float(P/(2*beta)-np.dot(omega**2, a)/(2*P)),
                        "kinetic_virial": float(.5/beta+np.sum(a[1:])/(2*P))}
            dyn = TorchDynamics(TorchPES(obs.Oscillator(1., 0.), device=device), [1.], beta, P, dt,
                                integrator=scheme, thermostat="pile-l", seed=5000+P)
            zero = np.zeros((n, P)); state = obs.state_1d(zero, zero, device)
            dyn.propagate(state, steps)
            q, p = state.q[:, :, 0, 0].cpu().numpy(), state.p[:, :, 0, 0].cpu().numpy()
            values = {"q2": np.mean(q*q, axis=1),
                      "kinetic_primitive": P/(2*beta)-.5*(P/beta)**2*np.mean((q-np.roll(q, 1, axis=1))**2, axis=1),
                      "kinetic_virial": .5/beta+.5*np.mean((q-q.mean(1, keepdims=True))*q, axis=1),
                      "p2": np.mean(p*p, axis=1)}
            finite_expectation = expected(finite); finite_expectation["p2"] = float(np.mean(p_finite))
            stats = {k: obs.summary(v) for k, v in values.items()}
            rows.append({"scheme": scheme, "beads": P, "seed": 5000+P, "steps": steps,
                         "finite_P_exact": expected(exact), "discrete_stationary": expected(stationary),
                         "independent_finite_time": finite_expectation, "sampled": stats,
                         "z_vs_finite_time": {k: (v["mean"]-finite_expectation[k])/v["se"] for k,v in stats.items()}})
    return {"beta": beta, "dt": dt, "trajectories": n, "cold_start": True,
            "potential": "q**2/2", "rows": rows}


def sampling_section(n=256, device="cpu"):
    """Identical n, times, seed and estimators to the previous quartic protocol."""
    P, beta, equil, production, interval = 16, 1., 20., 100., .2
    refs = [obs.transfer_quartic(P, points=points) for points in (256, 512)]
    reference = refs[-1]; rows = []
    for scheme in (*obs.SCHEMES, BCOCB):
        for dt in (.02, .1):
            dyn = TorchDynamics(TorchPES(obs.Oscillator(), device=device), [1.], beta, P, dt,
                                integrator=scheme, thermostat="pile-l", seed=717)
            zero = np.zeros((n, P)); state = obs.state_1d(zero, zero, device)
            dyn.propagate(state, round(equil/dt))
            sums = {k: np.zeros(n) for k in obs.observables(zero, beta)}
            blocks = round(production/interval)
            for _ in range(blocks):
                dyn.propagate(state, round(interval/dt))
                for k, v in obs.observables(state.q[:, :, 0, 0].cpu().numpy(), beta).items():
                    sums[k] += v
            stats = {k: obs.summary(v/blocks) for k,v in sums.items()}
            rows.append({"scheme": scheme, "dt": dt, "seed": 717, "measured": stats,
                         "z_vs_finite_P": {k: (v["mean"]-reference[k])/v["se"] for k,v in stats.items()}})
    return {"beads": P, "beta": beta, "trajectories": n, "equil_time": equil,
            "production_time": production, "interval": interval, "reference_grids": refs,
            "rows": rows, "uncertainty": "Student-t intervals across independent trajectory means"}


def correlation_section(n=2048, device="cpu", duration=10.):
    """Repeat thermostat-free RPMD: BCOCB must collapse to the existing BCB.

    This is NOT a BCOCB-thermostatted quantum-dynamics validation. No PILE
    thermostat is inserted into a physical RPMD production trajectory.
    """
    P, beta, seed, interval = 54, 1., 20261101, .1
    q, p, acceptance = obs.quartic_iid(n, P, beta, np.random.default_rng(seed))
    q0 = q.mean(1)
    configs = [("normal-mode-bab", .0025), ("normal-mode-bab", .005)]
    configs += [(scheme, dt) for scheme in ("cayley-bcb", BCOCB) for dt in (.01, .05, .1)]
    products, states = [], []
    for scheme, dt in configs:
        dyn = TorchDynamics(TorchPES(obs.Oscillator(), device=device), [1.], beta, P, dt, integrator=scheme)
        state = obs.state_1d(q, p, device); values = [q0*q0]
        for _ in range(round(duration/interval)):
            dyn.propagate(state, round(interval/dt))
            values.append(state.q[:, :, 0, 0].mean(1).cpu().numpy()*q0)
        products.append(np.asarray(values).T)
        states.append(np.concatenate([state.q.cpu().numpy().ravel(), state.p.cpu().numpy().ravel()]))
    C0 = float(np.mean(q0*q0)); rows = []
    # Family includes all non-reference methods and all recorded times.
    critical = student_t.ppf(1-.05/(2*(len(configs)-1)*products[0].shape[1]), n-1)
    for i, (scheme, dt) in enumerate(configs):
        delta = obs.summary(products[i]-products[0])
        upper = np.abs(delta["mean"])+critical*np.asarray(delta["se"])
        row = {"scheme": scheme, "dt": dt, "correlation": obs.summary(products[i]),
               "paired_difference": delta, "max_mean_difference_over_C0": float(max(abs(np.asarray(delta["mean"])))/C0),
               "simultaneous95_upper_over_C0": float(max(upper)/C0)}
        if scheme == BCOCB:
            index = configs.index(("cayley-bcb", dt))
            row["max_state_difference_vs_bcb"] = float(max(abs(states[i]-states[index])))
            row["max_correlation_difference_vs_bcb"] = float(np.max(abs(products[i]-products[index])))
        rows.append(row)
    return {"trajectories": n, "beads": P, "beta": beta, "seed": seed, "thermostat": None,
            "acceptance": acceptance, "C0": C0, "time": (np.arange(products[0].shape[1])*interval).tolist(),
            "reference": "small-step BAB dt=.0025 RPMD, not exact quantum dynamics", "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", choices=("algebra", "dimension", "sampling", "correlation", "all"), default="all")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--trajectories", type=int)
    args = ap.parse_args()
    if args.trajectories is not None and args.trajectories < 2:
        ap.error("--trajectories must be at least 2")
    torch.set_num_threads(1)
    result = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "numpy": np.__version__, "device": args.device, "cuda_available": torch.cuda.is_available(),
                                "threads": torch.get_num_threads()},
              "backend_sha256": hashlib.sha256(Path(inspect.getfile(TorchDynamics)).read_bytes()).hexdigest(),
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "observable_reference_sha256": hashlib.sha256(Path(obs.__file__).read_bytes()).hexdigest(),
              "reference_doi": "10.1063/1.5134810"}
    functions = {"algebra": (algebra_section, None), "dimension": (dimension_section, 1024),
                 "sampling": (sampling_section, 256), "correlation": (correlation_section, 2048)}
    start = time.perf_counter()
    for name, (fn, n) in functions.items():
        if args.section not in (name, "all"):
            continue
        result[name] = fn() if n is None else fn(n=n if args.trajectories is None else args.trajectories, device=args.device)
        result["seconds"] = time.perf_counter()-start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print(f"{name}: complete", flush=True)


if __name__ == "__main__":
    main()
