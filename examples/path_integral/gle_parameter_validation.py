"""Reproduce harmonic checks of generated PIGLET baths without matrix fixtures.

Run from an installed checkout, for example::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
        examples/path_integral/gle_parameter_validation.py --output validation.json

The references below solve the complete coordinate/momentum/auxiliary SDE
independently of the fitting code.  Trajectories start at q=0 with the ordinary
NativeEnsemble free-bath initialization; they are not initialized from the
desired harmonic answer.  Independent atoms provide independent replicates.
Different frequencies share a native harmonic calculation by using k=1 and
m=1/omega**2.  All quantities are in atomic units.

This is a harmonic implementation and sampling check, not an anharmonic or
physical-convergence claim.  The generated parameter records are included in
the output; no third-party numerical parameter data are used.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov, solve_discrete_lyapunov
from threadpoolctl import threadpool_info, threadpool_limits

from pyrpmd.gle_parameters import generate_piglet
from pyrpmd.native_ensemble import NativeEnsemble


def _mode_labels(beads):
    # Real orthonormal basis: centroid, cos/sin pairs, optional Nyquist mode.
    return ([0] + [k for k in range(1, (beads + 1) // 2) for _ in range(2)]
            + ([beads // 2] if beads % 2 == 0 else []))


def _energies(widths, omega, beta):
    potential = 0.5 * omega**2 * np.mean(widths)
    kinetic = 0.5 / beta + 0.5 * omega**2 * np.sum(widths[1:]) / len(widths)
    return np.array([potential, kinetic])


def harmonic_reference(parameters, frequencies, *, dt, burn_time):
    """Independent continuous and discrete harmonic covariance calculations.

    q and p are mass-scaled.  D S + S D.T = Q determines the continuous
    covariance.  The discrete check independently composes B/2-C/2-O-C/2-B/2
    linear maps and solves S=M S M.T+Q_step, without the production propagator.
    It separates timestep bias from the fitted continuous target.
    """
    beta, beads = parameters["beta"], parameters["beads"]
    aa, cc = np.asarray(parameters["A"]), np.asarray(parameters["C"])
    springs = 2 * beads / beta * np.sin(np.pi * np.array(_mode_labels(beads)) / beads)
    theta = beads / beta
    cases = []
    for omega in frequencies:
        widths, discrete_widths, transient_widths = [], [], []
        residuals, decays, autocorrelation_times = [], [], []
        for spring, a, c in zip(springs, aa, cc):
            n = len(a) + 1
            drift = np.zeros((n, n))
            drift[0, 1], drift[1, 0] = -1, omega**2 + spring**2
            drift[1:, 1:] = a
            diffusion = np.zeros_like(drift)
            diffusion[1:, 1:] = theta * (a @ c + c @ a.T)
            covariance = solve_continuous_lyapunov(drift, diffusion)
            covariance = 0.5 * (covariance + covariance.T)
            if np.linalg.eigvalsh(covariance).min() <= 0:
                raise ValueError("the independent harmonic covariance is not positive definite")
            widths.append(covariance[0, 0])
            residuals.append(float(np.linalg.norm(drift @ covariance + covariance @ drift.T - diffusion)
                                   / np.linalg.norm(diffusion)))
            decays.append(float(np.linalg.eigvals(drift).real.min()))
            integral = solve_continuous_lyapunov(drift, np.outer(covariance[:, 0], covariance[:, 0]))
            autocorrelation_times.append(float(integral[0, 0] / covariance[0, 0]**2))

            initial = np.zeros_like(drift)
            initial[1:, 1:] = theta * c
            decay = expm(-burn_time * drift)
            transient = covariance + decay @ (initial - covariance) @ decay.T
            transient_widths.append(transient[0, 0])

            kick, free, bath = np.eye(n), np.eye(n), np.eye(n)
            kick[1, 0] = -0.5 * dt * omega**2
            factor = 1 / math.sqrt(1 + (0.5 * dt * spring)**2)
            free[:2, :2] = factor * np.array([[1., 0.5 * dt], [-0.5 * dt * spring**2, 1.]])
            bath[1:, 1:] = expm(-dt * a)
            bath_noise = np.zeros_like(drift)
            bath_noise[1:, 1:] = theta * (c - bath[1:, 1:] @ c @ bath[1:, 1:].T)
            following = kick @ free
            transition = following @ bath @ free @ kick
            step_noise = following @ bath_noise @ following.T
            discrete = solve_discrete_lyapunov(transition, step_noise)
            discrete_widths.append(discrete[0, 0])

        exact = omega / (4 * math.tanh(beta * omega / 2))
        continuous = _energies(widths, omega, beta)
        discrete = _energies(discrete_widths, omega, beta)
        transient = _energies(transient_widths, omega, beta)
        canonical = _energies(theta / (omega**2 + springs**2), omega, beta)
        cases.append({
            "omega_au": float(omega), "beta_omega": float(beta * omega),
            "exact_quantum_potential_and_kinetic_hartree": exact,
            "continuous_piglet_hartree": continuous.tolist(),
            "continuous_relative_error": (continuous / exact - 1).tolist(),
            "canonical_finite_bead_hartree": canonical.tolist(),
            "canonical_finite_bead_relative_error": (canonical / exact - 1).tolist(),
            "discrete_piglet_hartree": discrete.tolist(),
            "discrete_relative_error": (discrete / exact - 1).tolist(),
            "timestep_bias_relative_to_continuous": (discrete / continuous - 1).tolist(),
            "initialization_relative_bias_at_burn": ((transient - continuous) / exact).tolist(),
            "minimum_full_drift_decay_rate_au": min(decays),
            "mode_potential_autocorrelation_times_au": autocorrelation_times,
            "maximum_normalized_lyapunov_residual": max(residuals),
        })
    return cases


def _steps(duration, dt):
    count = round(duration / dt)
    if count <= 0 or not math.isclose(count * dt, duration, rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError("durations must be positive integer multiples of dt")
    return count


def _estimate(values, exact):
    """Use both independent-atom and time-block uncertainty estimates."""
    # values has shape (saved times, independent atoms, V/Kcv).
    replicas = values.mean(axis=0)
    replica_se = replicas.std(axis=0, ddof=1) / np.sqrt(len(replicas))
    blocks = np.array([part.mean(axis=(0, 1)) for part in np.array_split(values, 20)])
    block_se = blocks.std(axis=0, ddof=1) / np.sqrt(len(blocks))
    error = np.maximum(replica_se, block_se)
    means = values.mean(axis=(0, 1))
    return {label: {
        "mean_hartree": float(means[i]),
        "relative_error_to_exact": float(means[i] / exact - 1),
        "sampling_standard_error_hartree": float(error[i]),
        "relative_sampling_standard_error": float(error[i] / exact),
        "independent_atom_standard_error_hartree": float(replica_se[i]),
        "time_block_standard_error_hartree": float(block_se[i]),
    } for i, label in enumerate(("potential", "kinetic_centroid_virial"))}


def sample_harmonics(thermostat, *, beta, beads, frequencies, dt, natoms=16,
                     burn_time=100., production_time=300., sample_interval=.2, seed=7109):
    """Run independent harmonic groups through the unchanged native executor."""
    frequencies = np.asarray(frequencies, dtype=float)
    if (frequencies.ndim != 1 or not len(frequencies) or not np.isfinite(frequencies).all()
            or np.any(frequencies <= 0) or isinstance(natoms, bool)
            or not isinstance(natoms, (int, np.integer)) or natoms < 2):
        raise ValueError("positive frequencies and at least two independent atoms are required")
    burn_steps, production_steps, stride = (_steps(value, dt) for value in
                                            (burn_time, production_time, sample_interval))
    if burn_steps % stride or production_steps % stride or production_steps // stride < 40:
        raise ValueError("burn and production durations must contain complete sampling intervals and at least 40 frames")
    total_atoms = natoms * len(frequencies)
    start = time.perf_counter()
    engine = NativeEnsemble(
        np.zeros((beads, total_atoms, 3)), np.repeat(1 / frequencies**2, natoms), beta, dt,
        potential={"kind": "harmonic", "k": 1.0}, thermostat=thermostat, seed=seed,
    )
    setup_seconds = time.perf_counter() - start
    start = time.perf_counter()
    engine.run(burn_steps, stride=burn_steps)
    result = engine.run(production_steps, stride=stride, record_centroids=True, record_moments=True)
    propagation_seconds = time.perf_counter() - start
    centroid_square = np.sum(result["centroids"]**2, axis=-1)
    radius_square = result["radii_gyration_squared"]
    potential = 0.5 * (centroid_square + radius_square) / 3
    kinetic = 0.5 / beta + 0.5 * radius_square / 3
    # Independent per-atom estimators also check the production aggregate rows.
    np.testing.assert_allclose(potential.mean(axis=1), result["samples"][:, 2] / (3 * total_atoms), rtol=1e-12)
    np.testing.assert_allclose(kinetic.mean(axis=1), result["samples"][:, 4] / (3 * total_atoms), rtol=1e-12)
    observations = np.stack((potential, kinetic), axis=-1)
    cases = []
    for index, omega in enumerate(frequencies):
        exact = float(omega / (4 * np.tanh(beta * omega / 2)))
        cases.append({"omega_au": float(omega), "beta_omega": float(beta * omega),
                      "exact_quantum_potential_and_kinetic_hartree": exact,
                      **_estimate(observations[:, index * natoms:(index + 1) * natoms], exact)})
    return {
        "thermostat": result["thermostat"], "beads": beads, "beta": beta, "dt_au": dt,
        "seed": seed, "initialization": "q=0; native joint free-bath momentum/auxiliary draw",
        "independent_atoms_per_frequency": natoms, "cartesian_oscillators_per_atom": 3,
        "total_atoms": total_atoms, "burn_time_au": burn_time, "production_time_au": production_time,
        "sample_interval_au": sample_interval, "saved_frames": len(observations), "time_blocks": 20,
        "uncertainty": "maximum of independent-atom standard error and time-block standard error",
        "timing": {"construction_seconds": setup_seconds, "propagation_seconds": propagation_seconds,
                   "steps_including_burn": burn_steps + production_steps,
                   "microseconds_per_step": 1e6 * propagation_seconds / (burn_steps + production_steps)},
        "fingerprint": engine.fingerprint, "final_state_digest": engine.state()["digest"],
        "rng_abi": getattr(engine._native, "rng_abi", "unavailable"),
        "python_step_callbacks": result["python_step_callbacks"], "cases": cases,
        "physical_converged": False,
    }


def _hardware(cpu_description):
    return {
        "cpu_description": cpu_description or platform.processor() or "unavailable",
        "cpu_description_source": "explicit CLI description" if cpu_description else "platform.processor",
        "platform": platform.platform(), "machine": platform.machine(),
        "python_version": platform.python_version(),
        "os_reported_logical_cpu_count": os.cpu_count(),
        "allocated_vcpus": "not probed; os.cpu_count is not an allocation guarantee",
        "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "threadpools": [{key: pool.get(key) for key in ("internal_api", "num_threads", "version")}
                        for pool in threadpool_info()],
    }


def run_validation(*, natoms=16, burn_time=100., production_time=300., dt=.01,
                   cpu_description=None, threads=1):
    start = time.perf_counter()
    beta, frequencies = 8., (.5, 1., 1.5)
    band, experiments, half_step = (.5 / beta, 12 / beta), [], None
    with threadpool_limits(limits=threads):
        hardware = _hardware(cpu_description)
        for beads in (2, 4, 8):
            fit_start = time.perf_counter()
            parameters = generate_piglet(band, beta=beta, beads=beads, naux=2)
            generation_seconds = time.perf_counter() - fit_start
            common = dict(beta=beta, beads=beads, frequencies=frequencies, dt=dt, natoms=natoms,
                          burn_time=burn_time, production_time=production_time)
            piglet = sample_harmonics({"mode": "piglet", "gle": parameters}, seed=7109 + beads, **common)
            canonical = sample_harmonics({"mode": "pile-l", "gamma_centroid": math.sqrt(band[0] * band[1])},
                                         seed=8109 + beads, **common)
            experiments.append({"beads": beads, "parameters": parameters, "generation_seconds": generation_seconds,
                                "harmonic_reference": harmonic_reference(parameters, frequencies, dt=dt, burn_time=burn_time),
                                "piglet": piglet, "canonical_pimd": canonical})
            if beads == 4:
                half_step = {
                    "harmonic_reference": harmonic_reference(parameters, [frequencies[-1]], dt=dt / 2, burn_time=burn_time),
                    "piglet": sample_harmonics({"mode": "piglet", "gle": parameters}, beta=beta, beads=beads,
                                                frequencies=[frequencies[-1]], dt=dt / 2, natoms=natoms,
                                                burn_time=burn_time, production_time=production_time, seed=9109),
                }
    return {
        "schema": "pyrpmd.generated-gle-validation.v1", "physical_converged": False,
        "scope": "generated harmonic matrices, independent Lyapunov references and finite native trajectories only",
        "reference_energy_order": ["potential", "kinetic_centroid_virial"],
        "units": "atomic: bohr, hartree, electron mass, atomic time; beta in inverse hartree",
        "hardware": hardware, "requested_threads": threads, "experiments": experiments,
        "half_timestep_check": half_step, "elapsed_seconds": time.perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--atoms-per-frequency", type=int, default=16)
    parser.add_argument("--burn-time", type=float, default=100.)
    parser.add_argument("--production-time", type=float, default=300.)
    parser.add_argument("--dt", type=float, default=.01)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpu-description")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    report = run_validation(natoms=args.atoms_per_frequency, burn_time=args.burn_time,
                            production_time=args.production_time, dt=args.dt,
                            cpu_description=args.cpu_description, threads=args.threads)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
