#!/usr/bin/env python3
"""Reproducible Tully-I scattering with public PyRPMD engines and a quantum reference.

Model: Tully, JCP 93, 1061 (1990), Eq. 21, DOI 10.1063/1.459170.
Reference values are independently computed, NOT digitized paper points.
Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python this.py
--output docs/validation/tier2-literature-2026-11-01/nonadiabatic --workers 4.
Every seeded job is checkpointed after each trajectory and can be resumed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.stats import norm

from pyrpmd.adiabatic import AdiabaticPythonPotential, AdiabaticResult, AdiabaticRPSHDynamics
from pyrpmd.dynamics import State, native_capabilities
from pyrpmd.mash import MASHDynamics
from pyrpmd.nonadiabatic import DiabaticResult, NRPMDDynamics, NRPMDState
from pyrpmd.adiabatic import QDNRPMDDynamics


ROOT = Path(__file__).resolve().parents[2]
CHANNELS = ("transmitted_lower", "transmitted_upper", "reflected_lower", "reflected_upper", "unresolved")
MAIN_SEEDS = (731921, 182331, 904577, 411953)


@dataclass(frozen=True)
class TullySimpleAvoidedCrossing:
    """Eq. 21, atomic units; differentiable real adiabatic provider with overlaps."""

    a: float = .01
    b: float = 1.6
    c: float = .005
    d: float = 1.
    mass: float = 2000.
    sign_flips: bool = False

    def elements(self, x):
        x = np.asarray(x, dtype=float)
        diagonal = self.a*np.sign(x)*(-np.expm1(-self.b*np.abs(x)))
        coupling = self.c*np.exp(-self.d*x*x)
        diagonal_gradient = self.a*self.b*np.exp(-self.b*np.abs(x))
        coupling_gradient = -2*self.d*x*coupling
        return diagonal, coupling, diagonal_gradient, coupling_gradient

    def diabatic(self, q):
        v, coupling, dv, dc = self.elements(q[:, 0, 0])
        matrix = np.zeros((len(q), 2, 2))
        matrix[:, 0, 0], matrix[:, 1, 1] = v, -v
        matrix[:, 0, 1] = matrix[:, 1, 0] = coupling
        gradient = np.zeros((len(q), 2, 2, 1, 3))
        gradient[:, 0, 0, 0, 0], gradient[:, 1, 1, 0, 0] = dv, -dv
        gradient[:, 0, 1, 0, 0] = gradient[:, 1, 0, 0, 0] = dc
        return DiabaticResult(matrix, gradient)

    def evaluate(self, q, *, previous_q=None, previous_context=None):
        v, coupling, dv, dc = self.elements(q[:, 0, 0])
        radius = np.hypot(v, coupling)
        gradient_radius = (v*dv+coupling*dc)/radius
        angle = .5*np.arctan2(coupling, v)
        angle_gradient = .5*(v*dc-coupling*dv)/(radius*radius)
        sine, cosine = np.sin(angle), np.cos(angle)
        basis = np.stack([np.stack([-sine, cosine], -1),
                          np.stack([cosine, sine], -1)], -2)
        gradient = np.zeros((len(q), 2, 1, 3))
        gradient[:, 0, 0, 0], gradient[:, 1, 0, 0] = -gradient_radius, gradient_radius
        nac = np.zeros((len(q), 2, 2, 1, 3))
        nac[:, 0, 1, 0, 0], nac[:, 1, 0, 0, 0] = angle_gradient, -angle_gradient
        if self.sign_flips:
            signs = np.ones((len(q), 2))
            signs[:, 0] = np.where(np.floor(q[:, 0, 0]*3).astype(int) % 2, -1., 1.)
            basis *= signs[:, None, :]
            nac *= signs[:, :, None, None, None]*signs[:, None, :, None, None]
        overlap = None if previous_context is None else previous_context.swapaxes(1, 2)@basis
        return AdiabaticResult(np.stack([-radius, radius], -1), gradient, nac,
                               overlap, context=basis)


def quantum_scattering(momentum, *, dx=.0025, boundary=12., model=None):
    """Stationary two-channel Schrödinger scattering using open discrete leads.

    Central second differences approximate -1/(2m)*d²/dx². The exterior of
    each endpoint is an exactly eliminated constant diagonal lead. Its complex
    Bloch factor imposes an outgoing (or decaying, closed-channel) boundary.
    The flux ratio is sin(k_out*dx)/sin(k_in*dx), the exact discrete current.
    Mesh refinement removes the finite-difference dispersion error.
    """
    model = TullySimpleAvoidedCrossing() if model is None else model
    count = int(np.ceil(2*boundary/dx))+1
    x = np.linspace(-boundary, boundary, count)
    spacing = float(x[1]-x[0])
    v, coupling, _, _ = model.elements(x)
    hopping = 1/(2*model.mass*spacing*spacing)
    energy = momentum*momentum/(2*model.mass)-model.a
    local = np.column_stack([v, -v])
    diagonal = (2*hopping+local-energy).astype(complex).reshape(-1)

    def outgoing_angle(endpoint):
        angle = np.arccos(np.asarray(1-(energy-endpoint)/(2*hopping), complex))
        return np.where(angle.imag < 0, angle.conjugate(), angle)

    left, right = outgoing_angle(local[0]), outgoing_angle(local[-1])
    diagonal[:2] -= hopping*np.exp(1j*left)
    diagonal[-2:] -= hopping*np.exp(1j*right)
    adjacent = np.zeros(2*count-1)
    adjacent[::2] = coupling
    nearest = np.full(2*count-2, -hopping)
    operator = diags([nearest, adjacent, diagonal, adjacent, nearest],
                     [-2, -1, 0, 1, 2], shape=(2*count, 2*count), format="csc")
    rhs = np.zeros(2*count, complex)
    rhs[0] = -2j*hopping*np.sin(left[0])
    wave = spsolve(operator, rhs).reshape(count, 2)
    incident_current = np.sin(left[0]).real
    if incident_current <= 0:
        raise ValueError("incident channel is closed or outside the finite-difference band")
    reflected = abs(wave[0]-np.array([1., 0.]))**2*np.sin(left).real/incident_current
    transmitted = abs(wave[-1])**2*np.sin(right).real/incident_current
    probabilities = [transmitted[1], transmitted[0], reflected[0], reflected[1]]
    return {"momentum": float(momentum), "dx": spacing, "boundary": boundary,
            "grid_points": count, "energy": energy,
            **dict(zip(CHANNELS[:4], map(float, probabilities))),
            "flux_error": abs(float(sum(probabilities))-1.)}


def wilson(successes, total):
    """Two-sided 95% Wilson binomial interval, including zero-count channels."""
    z = float(norm.ppf(.975))
    proportion = successes/total
    denominator = 1+z*z/total
    center = (proportion+z*z/(2*total))/denominator
    half = z*np.sqrt(proportion*(1-proportion)/total+z*z/(4*total*total))/denominator
    return float(max(0., center-half)), float(min(1., center+half))


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temporary.replace(path)


def run_trajectory(method, momentum, dt, seed, *, boundary=8., max_flight_times=12,
                   electronic_substeps=4, sign_flips=False):
    model = TullySimpleAvoidedCrossing(sign_flips=sign_flips)
    provider = AdiabaticPythonPotential(model)
    cls = AdiabaticRPSHDynamics if method == "rpsh" else MASHDynamics
    dynamics = cls(provider, [model.mass], 1., 1, dt, 2, native=True,
                   electronic_substeps=electronic_substeps, seed=seed)
    q = np.zeros((1, 1, 3)); q[0, 0, 0] = -boundary
    p = np.zeros_like(q)
    # Match the specified asymptotic incoming energy at the finite boundary.
    energy = momentum*momentum/(2*model.mass)-model.a
    potential = float(model.evaluate(q).energies[0, 0])
    p[0, 0, 0] = np.sqrt(2*model.mass*(energy-potential))
    state = dynamics.initialize(State(q, p), active_surface=0)
    max_steps = int(np.ceil(max_flight_times*2*boundary*model.mass/momentum/dt))
    initial_coefficients = state.coefficients.copy()
    max_energy_error = 0.
    max_norm_error = 0.
    accepted = frustrated = 0
    channel = "unresolved"
    started = time.perf_counter()
    for _ in range(max_steps):
        dynamics.step(state)
        current_energy = (float(state.p[0, 0, 0])**2/(2*model.mass)
                          +float(state.bead_frame.result.energies[0, state.active_surface]))
        max_energy_error = max(max_energy_error, abs(current_energy-energy))
        max_norm_error = max(max_norm_error, abs(float(np.vdot(state.coefficients, state.coefficients).real)-1))
        if dynamics.last_hop is not None:
            accepted += int(dynamics.last_hop.accepted)
            frustrated += int(dynamics.last_hop.frustrated)
        position = state.q[0, 0, 0]
        if position >= boundary and state.p[0, 0, 0] > 0:
            channel = CHANNELS[state.active_surface]
            break
        if position <= -boundary and state.p[0, 0, 0] < 0:
            channel = CHANNELS[2+state.active_surface]
            break
    return {"channel": channel, "steps": int(state.step), "time_au": state.time,
            "final_x": float(state.q[0, 0, 0]), "final_p": float(state.p[0, 0, 0]),
            "active_surface": int(state.active_surface), "accepted_hops": accepted,
            "frustrated_hops": frustrated, "max_energy_error": max_energy_error,
            "final_energy_error": current_energy-energy, "max_norm_error": max_norm_error,
            "initial_coefficients_real": initial_coefficients.real.tolist(),
            "initial_coefficients_imag": initial_coefficients.imag.tolist(),
            "seconds": time.perf_counter()-started}


def _run_job(spec):
    path = Path(spec["path"])
    identity = {k: v for k, v in spec.items() if k != "path"}
    if path.exists():
        result = json.loads(path.read_text())
        if result["identity"] != identity:
            raise ValueError("checkpoint identity differs: "+str(path))
    else:
        result = {"identity": identity, "complete": False, "trajectories": []}
    for index in range(len(result["trajectories"]), spec["count"]):
        entropy = [spec["seed"], int(spec["momentum"]*1000), index]
        trajectory = run_trajectory(spec["method"], spec["momentum"], spec["dt"],
                                    np.random.SeedSequence(entropy), boundary=spec["boundary"],
                                    electronic_substeps=spec["electronic_substeps"])
        trajectory.update(trajectory=index, seed_entropy=entropy)
        result["trajectories"].append(trajectory)
        result["complete"] = len(result["trajectories"]) == spec["count"]
        _atomic_json(path, result)
    return str(path)


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def reference_grid(momenta):
    rows = []
    for momentum in momenta:
        for dx in [.005, .0025, .00125]:
            rows.append(quantum_scattering(momentum, dx=dx))
        rows.append(quantum_scattering(momentum, dx=.0025, boundary=16.))
    # Eq.20's sigma_x=20/k gives standard deviation sigma_k=k/20.
    # These Gaussian averages provide context for the paper's wavepacket
    # reference, separate from the monochromatic trajectory comparison.
    nodes, weights = np.polynomial.hermite.hermgauss(17)
    packet = []
    for momentum in momenta:
        samples = [quantum_scattering(momentum+np.sqrt(2)*(momentum/20)*node, dx=.0025)
                   for node in nodes]
        packet.append({"momentum": momentum, "sigma_momentum": momentum/20,
                       "quadrature_nodes": len(nodes), "dx": .0025,
                       **{c: float(np.dot(weights, [row[c] for row in samples])/np.sqrt(np.pi))
                          for c in CHANNELS[:4]}})
    return rows, packet


def transport_validation(momentum=15., dt=4., steps=160):
    """Published-model direct-QC/diabatic equivalence and arbitrary phase checks."""
    q = np.zeros((1, 1, 3)); q[0, 0, 0] = -2.
    p = np.zeros_like(q); p[0, 0, 0] = momentum
    model = TullySimpleAvoidedCrossing(sign_flips=True)
    class Diabatic:
        def evaluate(self, positions):
            return model.diabatic(positions)
    direct = QDNRPMDDynamics(AdiabaticPythonPotential(model), [model.mass], 1., 1, dt, 2)
    local = direct.initialize(State(q, p), seed=31)
    u = local.electronic_frame.result.context
    z = np.einsum("bij,bj->bi", u, local.mapping_q+1j*local.mapping_p)
    fixed = NRPMDState(q, p, mapping_q=z.real, mapping_p=z.imag)
    diabatic = NRPMDDynamics(Diabatic(), [model.mass], 1., 1, dt, 2)
    q_error = p_error = z_error = energy_error = 0.
    for _ in range(steps):
        direct.step(local); diabatic.step(fixed)
        u = local.electronic_frame.result.context
        z = np.einsum("bij,bj->bi", u, local.mapping_q+1j*local.mapping_p)
        q_error = max(q_error, float(np.max(abs(local.q-fixed.q))))
        p_error = max(p_error, float(np.max(abs(local.p-fixed.p))))
        z_error = max(z_error, float(np.max(abs(z-fixed.mapping_q-1j*fixed.mapping_p))))
        energy_error = max(energy_error, abs(direct.energy(local)-diabatic.energy(fixed)))
    gauge = []
    for method in ["rpsh", "mash"]:
        a = run_trajectory(method, momentum, dt, 2719)
        b = run_trajectory(method, momentum, dt, 2719, sign_flips=True)
        gauge.append({"method": method, "same_channel": a["channel"] == b["channel"],
                      "same_step_count": a["steps"] == b["steps"],
                      "final_x_difference": abs(a["final_x"]-b["final_x"]),
                      "final_p_difference": abs(a["final_p"]-b["final_p"])})
    return {"model": "Tully-I", "qd_steps": steps, "dt": dt,
            "qd_vs_diabatic_max_q_error": q_error, "max_p_error": p_error,
            "max_fixed_basis_mapping_error": z_error, "max_energy_difference": energy_error,
            "sign_flip_scattering": gauge,
            "scope": "numerical representation/gauge validation; not a QD-NRPMD scattering accuracy claim"}


def summarize(output, jobs, references, packet):
    trajectories, seed_rows, rows = [], [], []
    for job in jobs:
        result = json.loads(Path(job["path"]).read_text())
        if not result["complete"]:
            raise ValueError("incomplete scattering job")
        for trajectory in result["trajectories"]:
            trajectories.append({"method": job["method"], "momentum": job["momentum"], "dt": job["dt"],
                                 "block_seed": job["seed"], "trajectory": trajectory["trajectory"],
                                 **{k: trajectory[k] for k in ("channel", "steps", "time_au", "final_x", "final_p",
                                                              "accepted_hops", "frustrated_hops", "max_energy_error",
                                                              "final_energy_error", "max_norm_error", "seconds")}})
        for channel in CHANNELS:
            count = sum(r["channel"] == channel for r in result["trajectories"])
            low, high = wilson(count, job["count"])
            seed_rows.append({"method": job["method"], "momentum": job["momentum"], "dt": job["dt"],
                              "seed": job["seed"], "channel": channel, "count": count, "n": job["count"],
                              "probability": count/job["count"], "ci95_low": low, "ci95_high": high})
    groups = sorted({(r["method"], r["momentum"], r["dt"]) for r in trajectories})
    for method, momentum, dt in groups:
        selected = [r for r in trajectories if (r["method"], r["momentum"], r["dt"]) == (method, momentum, dt)]
        reference = min((r for r in references if r["momentum"] == momentum), key=lambda r: r["dx"])
        for channel in CHANNELS:
            count, total = sum(r["channel"] == channel for r in selected), len(selected)
            low, high = wilson(count, total)
            exact = reference.get(channel, 0.)
            rows.append({"method": method, "momentum": momentum, "dt": dt, "channel": channel,
                         "count": count, "n": total, "probability": count/total,
                         "ci95_low": low, "ci95_high": high, "quantum_monochromatic": exact,
                         "difference_from_quantum": count/total-exact,
                         "quantum_inside_ci95": low <= exact <= high,
                         "mean_max_energy_error": float(np.mean([r["max_energy_error"] for r in selected])),
                         "largest_energy_error": max(r["max_energy_error"] for r in selected),
                         "largest_norm_error": max(r["max_norm_error"] for r in selected)})
    paired = []
    for method in ("rpsh", "mash"):
        for momentum in sorted({r["momentum"] for r in trajectories}):
            for coarse, fine in [(8., 4.), (4., 2.)]:
                left = {r["trajectory"]: r for r in trajectories if r["method"] == method and r["momentum"] == momentum
                        and r["dt"] == coarse and r["block_seed"] == MAIN_SEEDS[0]}
                right = {r["trajectory"]: r for r in trajectories if r["method"] == method and r["momentum"] == momentum
                         and r["dt"] == fine and r["block_seed"] == MAIN_SEEDS[0]}
                keys = sorted(set(left)&set(right))
                if not keys:
                    continue
                for channel in CHANNELS:
                    differences = np.array([int(right[k]["channel"] == channel)-int(left[k]["channel"] == channel) for k in keys])
                    standard_error = float(differences.std(ddof=1)/np.sqrt(len(keys))) if len(keys) > 1 else 0.
                    paired.append({"method": method, "momentum": momentum, "coarse_dt": coarse, "fine_dt": fine,
                                   "channel": channel, "paired_n": len(keys), "difference_fine_minus_coarse": float(differences.mean()),
                                   "paired_standard_error": standard_error,
                                   "changed_outcomes": int(sum(left[k]["channel"] != right[k]["channel"] for k in keys))})
    _write_csv(output/"trajectories.csv", trajectories)
    _write_csv(output/"seed_blocks.csv", seed_rows)
    _write_csv(output/"channel_probabilities.csv", rows)
    if paired:
        _write_csv(output/"paired_timestep_comparison.csv", paired)
    _write_csv(output/"quantum_reference_convergence.csv", references)
    _write_csv(output/"quantum_paper_width_average.csv", packet)
    summary = {"total_trajectories": len(trajectories), "complete_jobs": len(jobs),
               "unresolved": sum(r["channel"] == "unresolved" for r in trajectories),
               "method_scope": "P=1 direct-data FSSH and fixed-step endpoint MASH; no decoherence correction",
               "reference_kind": "independent stationary quantum calculation of published model, not digitized paper values",
               "max_quantum_flux_error": max(r["flux_error"] for r in references),
               "rows": rows, "paired_timestep_comparisons": paired}
    _atomic_json(output/"summary.json", summary)
    plot_results(output, rows, references)
    return summary


def plot_results(output, rows, references):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    momenta = sorted({r["momentum"] for r in rows})
    for ax, channel, title in zip(axes, CHANNELS[:3], ["Transmission: lower", "Transmission: upper", "Reflection: lower"]):
        exact = [min((r for r in references if r["momentum"] == p), key=lambda r: r["dx"])[channel] for p in momenta]
        ax.plot(momenta, exact, "kD--", label="Independent quantum")
        for method, offset, color in [("rpsh", -.3, "#1670a5"), ("mash", .3, "#bd5b23")]:
            selected = sorted((r for r in rows if r["method"] == method and r["dt"] == 4 and r["channel"] == channel),
                              key=lambda r: r["momentum"])
            if selected:
                y = np.array([r["probability"] for r in selected])
                error = np.array([[r["probability"]-r["ci95_low"] for r in selected],
                                  [r["ci95_high"]-r["probability"] for r in selected]])
                ax.errorbar(np.array(momenta)+offset, y, yerr=error, fmt="o", capsize=3, color=color,
                            label="FSSH" if method == "rpsh" else "Endpoint MASH")
        ax.set(title=title, xlabel="Incident momentum (a.u.)", ylim=(-.03, 1.03))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Scattering probability (95% Wilson interval)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Tully model I • actual public-engine trajectories • Δt = 4 a.u.")
    fig.savefig(output/"scattering_channels.png", dpi=180)
    fig.savefig(output/"scattering_channels.svg")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--momenta", type=float, nargs="+", default=[8.8, 15., 25.])
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), ROOT/"src/pyrpmd/adiabatic.py", ROOT/"src/pyrpmd/mash.py"]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    references, packet = reference_grid(args.momenta)
    _write_csv(output/"quantum_reference_convergence.csv", references)
    _write_csv(output/"quantum_paper_width_average.csv", packet)
    _atomic_json(output/"transport_validation.json", transport_validation())
    if args.reference_only:
        return
    code_identity = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    jobs = []
    for method in ["rpsh", "mash"]:
        for momentum in args.momenta:
            for dt in [8., 4., 2.]:
                for seed in (MAIN_SEEDS if dt == 4 else MAIN_SEEDS[:1]):
                    name = f"{method}_p{momentum:g}_dt{dt:g}_seed{seed}.json"
                    jobs.append({"method": method, "momentum": momentum, "dt": dt, "seed": seed,
                                 "count": args.block_size, "boundary": 8., "electronic_substeps": 4,
                                 "code_identity": code_identity, "path": str(output/"jobs"/name)})
    _atomic_json(output/"manifest.json", {"model": "Tully 1990 Eq.21 (model I)", "mass": 2000.,
                 "parameters": {"A": .01, "B": 1.6, "C": .005, "D": 1.}, "sources": source_hashes,
                 "python": platform.python_version(), "numpy": np.__version__, "native": native_capabilities(),
                 "workers": args.workers, "jobs": jobs,
                 "sampling": "four independent seed blocks at dt4; first block paired across dt8/4/2"})
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_job, job): job for job in jobs}
        for future in as_completed(futures):
            print("completed", Path(future.result()).name, flush=True)
    result = summarize(output, jobs, references, packet)
    print(json.dumps({"trajectories": result["total_trajectories"], "unresolved": result["unresolved"],
                      "wall_seconds": time.perf_counter()-started}), flush=True)


if __name__ == "__main__":
    main()
