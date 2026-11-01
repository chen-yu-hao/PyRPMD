#!/usr/bin/env python3
"""Tully DAC paper benchmark: distinguish numerical, sampling, and method bias.

Liu et al., J. Chem. Phys. 161, 024102 (2024), doi:10.1063/5.0208708.
Uses the *published author-provided* CSV anchors copied into this repository.
This is one nuclear bead, Gaussian Wigner x/p, and exact fixed-q spin flow.
The reduced 2-state spin identity is regression-tested against the unmodified
PyRPMD NRPMDDynamics/SMNRPMDDynamics scalar implementations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RADIUS_W = np.sqrt(3.)
MASS = 2000.


def reference(momentum):
    if momentum not in (5, 15, 30):
        raise ValueError("only pinned DAC p0=5, p0=15 and p0=30 reference curves are accepted")
    path = ROOT / f"dac_p{momentum}_liu2024_reference.csv"
    with path.open(encoding="utf-8") as stream:
        lines = [line for line in stream if line.strip() and not line.startswith("#")]
    if lines[0].strip() != "time,sigma_0_0_real,sigma_1_1_real":
        raise ValueError("unexpected published CSV schema")
    data = np.loadtxt(lines[1:], delimiter=",", ndmin=2)
    end = {5: 6000, 15: 2800, 30: 1200}[momentum]
    expected_times = np.arange(0, end+1, 40., dtype=float)
    if (data.shape != (len(expected_times), 3) or
            not np.array_equal(data[:, 0], expected_times) or
            not np.all(np.isfinite(data)) or
            np.max(abs(data[:, 1]+data[:, 2]-1)) > 2e-7 or
            np.min(data[:, 1:]) < -2e-7):
        raise ValueError("published data has missing times, zero tail, or invalid populations")
    return data, path


def fields(x):
    """Published Tully II A=0.10 B=0.28 C=0.015 D=0.06 E0=0.05."""
    exponential = np.exp(-0.28*x*x)
    v11 = -0.10*exponential+0.05
    dv11 = 0.056*x*exponential
    hx = 0.015*np.exp(-0.06*x*x)
    dhx = -0.12*x*hx
    return 0.5*v11, hx, -0.5*v11, 0.5*dv11, dhx, -0.5*dv11


def exact_fixed_q_spin_flow(q, p, spin, duration):
    """Integrate h0+h·s exactly with x frozen, including the momentum impulse.

    s is an unnormalized Bloch spin: |s|=2 (MMST-focused), sqrt(3)
    (W-spin), or 1 (Ehrenfest pure electronic state).
    This produces the same 2-state flow as the PyRPMD mapping kernels.
    """
    h0, hx, hz, dh0, dhx, dhz = fields(q)
    magnitude = np.hypot(hx, hz)
    nx, nz = hx/magnitude, hz/magnitude
    proj = spin[:, 0]*nx + spin[:, 2]*nz
    parallel_x, parallel_z = proj*nx, proj*nz
    perp_x, perp_y, perp_z = (spin[:, 0]-parallel_x, spin[:, 1],
                             spin[:, 2]-parallel_z)
    cross_x = -nz*spin[:, 1]
    cross_y = nz*spin[:, 0]-nx*spin[:, 2]
    cross_z = nx*spin[:, 1]
    angle = 2*duration*magnitude
    sine = np.sin(angle)
    cosine = np.cos(angle)
    integrated_x = duration*(parallel_x+np.sinc(angle/np.pi)*perp_x+
               (0.5*angle*np.sinc(angle/(2*np.pi))**2)*cross_x)
    integrated_z = duration*(parallel_z+np.sinc(angle/np.pi)*perp_z+
               (0.5*angle*np.sinc(angle/(2*np.pi))**2)*cross_z)
    p -= duration*dh0 + dhx*integrated_x + dhz*integrated_z
    spin[:, 0] = parallel_x+cosine*perp_x+sine*cross_x
    spin[:, 1] = cosine*perp_y+sine*cross_y
    spin[:, 2] = parallel_z+cosine*perp_z+sine*cross_z


def one_step(q, p, spin, dt):
    exact_fixed_q_spin_flow(q, p, spin, dt/2)
    q += dt*p/MASS
    exact_fixed_q_spin_flow(q, p, spin, dt/2)


def energy(q, p, spin):
    h0, hx, hz, *_ = fields(q)
    return p*p/(2*MASS)+h0+hx*spin[:, 0]+hz*spin[:, 2]


def prepare(momentum, n, seed, replica):
    streams = np.random.SeedSequence([20261101, seed, momentum, replica]).spawn(2)
    nuclear, electronic = (np.random.default_rng(s) for s in streams)
    q = -9+nuclear.standard_normal(n)/np.sqrt(2)
    p = momentum+nuclear.standard_normal(n)/np.sqrt(2)
    phi = electronic.uniform(0, 2*np.pi, n)
    sz = electronic.uniform(-1, 1, n)
    return q, p, phi, sz


def starting_spin(method, phi, uz):
    n = len(phi)
    spin = np.zeros((n, 3))
    if method == "ehrenfest":
        spin[:, 2] = 1.
    elif method == "mmst-focused":
        spin[:, 0] = np.sqrt(3.)*np.cos(phi)
        spin[:, 1] = np.sqrt(3.)*np.sin(phi)
        spin[:, 2] = 1.
    elif method == "w-focused":
        spin[:, 0] = np.sqrt(2.)*np.cos(phi)
        spin[:, 1] = np.sqrt(2.)*np.sin(phi)
        spin[:, 2] = 1.
    elif method == "spm-w-fullsphere":
        spin[:, 0] = RADIUS_W*np.sqrt(1-uz**2)*np.cos(phi)
        spin[:, 1] = RADIUS_W*np.sqrt(1-uz**2)*np.sin(phi)
        spin[:, 2] = RADIUS_W*uz
    else:
        raise ValueError("unknown method")
    return spin


def simulate(momentum, n, dt, seed, replica, methods, *, retain=False):
    source, path = reference(momentum)
    time = source[:, 0]
    if (n < 2 or dt <= 0 or
            not np.isfinite(dt) or abs(40/dt-round(40/dt)) > 1e-10):
        raise ValueError("n >= 2 and finite positive dt evenly dividing 40 a.u. are required")
    q0, p0, phi, uz = prepare(momentum, n, seed, replica)
    summary = {}
    raw_data = {}
    for method in methods:
        q, p = q0.copy(), p0.copy()
        spin = starting_spin(method, phi, uz)
        initial_spin = spin.copy()
        init_energy = energy(q, p, spin)
        radius_squared = float(np.mean(np.sum(spin**2, axis=1)))
        values = np.empty((n, len(time)))
        sample_error = np.empty(len(time))
        obs_i = 0
        for step in range(round(time[-1]/dt)+1):
            if obs_i < len(time) and abs(step*dt-time[obs_i]) < 1e-8:
                if method == "spm-w-fullsphere":
                    # Liu et al., Eq. 19–20: full sphere, W radius, Q=1,
                    # factor F=2 for normalized uniform spin-sphere sampling.
                    # The initial projector is SIGNED; do not self-normalize it.
                    init_projector = (1.+initial_spin[:, 2])/2.
                    values[:, obs_i] = 2*init_projector*(1.-spin[:, 2])/2.
                else:
                    values[:, obs_i] = (1.-spin[:, 2])/2.
                sample_error[obs_i] = max(abs(energy(q,p,spin)-init_energy))
                obs_i += 1
            if step*dt < time[-1]-dt/2:
                one_step(q, p, spin, dt)
        if obs_i != len(time):
            raise RuntimeError("missing observations")
        mean = values.mean(0)
        sem = values.std(0, ddof=1)/np.sqrt(n)
        diff = mean-source[:, 2]
        result = {
          "mean": mean.tolist(), "sem": sem.tolist(),
          "rmse_vs_published": float(np.sqrt(np.mean(diff**2))),
          "maximum_population_error": float(max(abs(diff))),
          "population_end": float(mean[-1]), "sem_end": float(sem[-1]),
          "reference_end": float(source[-1,2]),
          "end_bias_in_SEM": float(diff[-1]/sem[-1]),
          "max_energy_drift": float(sample_error.max()),
          "spin_length_squared": radius_squared,
          "max_spin_norm_error": float(max(abs(np.sum(spin**2,axis=1)-radius_squared))),
          "initial_population_mean": float(mean[0]),
          "observable": ("2*P0_W(initial)*P1_W(final), unnormalized signed Eq20"
                         if method=="spm-w-fullsphere" else
                         "diabatic W/MMST signed final-time population"
                         if method in ("mmst-focused","w-focused") else
                         "Ehrenfest diabatic wavefunction population")
        }
        if method == "spm-w-fullsphere":
            control = values-values[:, [0]]
            result["known_zero_initial_control_variate"] = {
               "mean": control.mean(axis=0).tolist(),
               "sem": (control.std(axis=0,ddof=1)/np.sqrt(n)).tolist(),
               "note": "variance reduction using P1(0)=0; original Eq20 above is unchanged"}
        summary[method] = result
        if retain:
            raw_data[method.replace("-", "_")] = values
    payload = {
      "schema":"PyRPMD.paper.Tully_DAC.statistical_comparison.v2",
      "paper":"Liu et al. J. Chem. Phys. 161, 024102 (2024); doi:10.1063/5.0208708",
      "reference":{"path":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                   "times":time.tolist(),"p1":source[:,2].tolist()},
      "physics":{"A":0.10,"B":0.28,"C":0.015,"D":0.06,"E0":0.05,
                 "mass":MASS,"initial_position":-9.,"initial_momentum":momentum,
                 "wigner_x_variance":0.5,"wigner_p_variance":0.5,
                 "initial_diabatic_state":0,"nuclear_beads":1},
      "execution":{"n":n,"dt":dt,"seed":seed,"replica":replica,
                   "numpy":np.__version__,"python":platform.python_version()},
      "methods":summary,
      "claims":{"exact_quantum_propagator":False,"physical_converged":False,
                "paper_level_method_agreement_certified":False,
                "warning":"One nuclear bead; focused W differs from paper full-sphere SPM-W; no physical method error tolerance asserted"}}
    return payload, raw_data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--momentum",type=int,choices=(5,15,30),required=True)
    parser.add_argument("--n",type=int,default=2048)
    parser.add_argument("--dt",type=float,default=4.)
    parser.add_argument("--seed",type=int,default=731)
    parser.add_argument("--replica",type=int,default=0)
    parser.add_argument("--methods",nargs="+",choices=(
        "mmst-focused","w-focused","spm-w-fullsphere","ehrenfest"),
        default=["mmst-focused","w-focused","spm-w-fullsphere","ehrenfest"])
    parser.add_argument("--raw",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    payload,raw = simulate(args.momentum,args.n,args.dt,args.seed,args.replica,
                           args.methods,retain=args.raw)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    if args.raw:
        np.savez_compressed(args.output.with_suffix(".npz"),
                            **raw,**{"time":np.asarray(payload["reference"]["times"])})
    for name,entry in payload["methods"].items():
        print(name,"p0=",args.momentum,"n=",args.n,"dt=",args.dt,
              "p1_end=",entry["population_end"],"+/-",entry["sem_end"],
              "quantum=",entry["reference_end"],
              "rmse=",entry["rmse_vs_published"],flush=True)


if __name__ == "__main__":
    main()
