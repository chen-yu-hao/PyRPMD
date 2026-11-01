#!/usr/bin/env python3
"""Reproducible PA-CMD/f-CMD checks for Cao--Voth's published quartic model.

Source: Cao and Voth, JCP 99, 10070 (1993), Eq. (18), p. 10072:
https://doi.org/10.1063/1.465512
https://web.mit.edu/jianshucaogroup/pdfdir/JCP99-10070-1993.pdf

V(q)=q^2/2 + (2.4/24)q^4, beta=5, m=hbar=1.  Our exact *Kubo*
reference is independently calculated; the paper's Fig. 1 plots the real
quantum correlation, so none of the generated curves are digitized paper data.
The rounded paper statement that E0 increases by 12% is checked separately.

One engine contains independent copies of the separable 1-D model in its atom
and Cartesian axes. No interactions couple them. Independent master seeds,
not time origins or FFT bins, define reported uncertainty. Exact rejection
sampling from a harmonic-ring Gaussian supplies finite-P canonical starting
coordinates; no equilibration/ergodicity claim is needed for those starts.
Actual PACMDDynamics.step and the public IR analysis propagate/analyze them.

Run from the checkout:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python \
    benchmarks/tier2_literature/spectroscopy.py --output OUTPUT
Use --quick only to smoke-check the pipeline, not for the committed evidence.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy.linalg import eigh, solve_discrete_lyapunov
from scipy.optimize import minimize_scalar
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits

from pyrpmd.analysis.spectroscopy import relative_ir_spectrum
from pyrpmd.cmd import PACMDDynamics, centroid_force_training_data, fit_quadratic_centroid_pmf
from pyrpmd.dynamics import State
from pyrpmd.potentials.base import PESResult


BETA = 5.0
QUARTIC = 0.1
ROOT = Path(__file__).resolve().parents[2]


class QuarticPotential:
    """A sum of independent paper Hamiltonians, each with m=hbar=1."""

    def __init__(self, quartic=QUARTIC):
        self.quartic = float(quartic)

    def evaluate(self, q):
        squared = q * q
        return PESResult((0.5 * squared + self.quartic * squared * squared).sum(axis=(1, 2)),
                         -q * (1.0 + 4.0 * self.quartic * squared))


def normal_modes(beads, beta=BETA):
    """Independent real Fourier construction for the initial-state sampler."""
    j = np.arange(beads)
    columns = [np.ones(beads) / np.sqrt(beads)]
    frequencies = [0.0]
    for k in range(1, (beads + 1) // 2):
        columns += [np.sqrt(2 / beads) * np.cos(2 * np.pi * k * j / beads),
                    np.sqrt(2 / beads) * np.sin(2 * np.pi * k * j / beads)]
        frequencies += [2 * beads / beta * np.sin(np.pi * k / beads)] * 2
    if beads % 2 == 0:
        columns += [(-1.) ** j / np.sqrt(beads)]
        frequencies += [2 * beads / beta]
    return np.stack(columns, axis=1), np.asarray(frequencies)


def canonical_rings(beads, copies, rng, *, beta=BETA, quartic=QUARTIC):
    """Independent exact finite-P draws; harmonic proposal, positive quartic weight.

    Returns [P,copies]. The acceptance probability is
    exp[-beta/P * quartic * sum_b q_b^4] <= 1, independently for every copy.
    The normal-mode Gaussian already includes BOTH harmonic PES and springs.
    """
    if beads < 1 or copies < 1 or beta <= 0 or quartic < 0:
        raise ValueError("positive beads/copies/beta and nonnegative quartic required")
    transform, frequencies = normal_modes(beads, beta)
    sigma = np.sqrt(beads / beta / (1 + frequencies ** 2))
    result = np.empty((beads, copies))
    pending = np.arange(copies)
    proposals = 0
    while len(pending):
        proposed = transform @ (sigma[:, None] * rng.normal(size=(beads, len(pending))))
        log_probability = -beta / beads * quartic * np.sum(proposed ** 4, axis=0)
        accepted = np.log(rng.random(len(pending))) < log_probability
        result[:, pending[accepted]] = proposed[:, accepted]
        proposals += len(pending)
        pending = pending[~accepted]
    return result, copies / proposals


def exact_reference(basis=100, *, beta=BETA, quartic=QUARTIC):
    """HO-basis diagonalization with padded operators, independent of MD forces.

    Padding before projecting q^4 avoids the false boundary matrix elements
    produced by raising an already truncated q matrix to its fourth power.
    """
    ladder = np.diag(np.sqrt(np.arange(1, basis + 4)), 1)
    x = (ladder + ladder.T) / np.sqrt(2.)
    x2, x4 = x @ x, x @ x @ x @ x
    hamiltonian = np.diag(np.arange(basis) + .5) + quartic * x4[:basis, :basis]
    energies, vectors = eigh(hamiltonian)
    position = vectors.T @ x[:basis, :basis] @ vectors
    populations = np.exp(-beta * (energies - energies[0]))
    populations /= populations.sum()
    low, high = np.triu_indices(basis, 1)
    gaps = energies[high] - energies[low]
    weights = 2 * (populations[low] - populations[high]) / (beta * gaps) * position[low, high] ** 2
    keep = weights > 1e-15
    gaps, weights, low, high = gaps[keep], weights[keep], low[keep], high[keep]
    diagonal = float(np.sum(populations * np.diag(position) ** 2))
    q2 = float(np.sum(populations * np.diag(vectors.T @ x2[:basis, :basis] @ vectors)))
    return {"basis": basis, "energies": energies, "gaps": gaps, "weights": weights,
            "low": low, "high": high, "diagonal": diagonal, "q2": q2,
            "c0": float(weights.sum() + diagonal),
            "curvature": float(-np.dot(weights, gaps * gaps)),
            "ground_energy": float(energies[0]), "fundamental": float(energies[1] - energies[0])}


def exact_correlation(reference, times):
    return reference["diagonal"] + reference["weights"] @ np.cos(reference["gaps"][:, None] * times[None])


def expected_windowed_ir(reference, frames, sample_dt, nfft, *, beta=BETA):
    """Expected finite-record periodogram of the exact Kubo correlation.

    Every line A cos(Delta*t) is represented by independent sine/cosine
    quadratures, each with variance A. Demeaning, Hann taper, rFFT doubling,
    and beta*omega^2 use the SAME conventions as relative_ir_spectrum.
    This gives an expectation, not a claim that zero padding resolves lines.
    """
    time_axis = np.arange(frames) * sample_dt
    window = np.hanning(frames)
    power = np.zeros(nfft // 2 + 1)
    for gap, weight in zip(reference["gaps"], reference["weights"]):
        quadratures = np.stack((np.cos(gap * time_axis), np.sin(gap * time_axis)))
        quadratures -= quadratures.mean(axis=1, keepdims=True)
        transformed = np.fft.rfft(quadratures * window, n=nfft, axis=1)
        power += weight * np.sum(np.abs(transformed) ** 2, axis=0)
    power *= sample_dt / np.sum(window * window)
    power[1:-1 if nfft % 2 == 0 else None] *= 2
    omega = 2 * np.pi * np.fft.rfftfreq(nfft, sample_dt)
    return omega, beta * omega ** 2 * power


def peak_frequency(omega, intensity, lower=.7, upper=1.7):
    """Interpolate a peak within an explicit fundamental-frequency window."""
    candidates = np.flatnonzero((omega >= lower) & (omega <= upper))
    index = candidates[np.argmax(intensity[candidates])]
    if not 0 < index < len(omega) - 1:
        return float(omega[index])
    values = np.log(np.maximum(intensity[index - 1:index + 2], np.finfo(float).tiny))
    denominator = values[0] - 2 * values[1] + values[2]
    fraction = 0.0 if denominator == 0 else .5 * (values[0] - values[2]) / denominator
    return float(omega[index] + np.clip(fraction, -.5, .5) * (omega[1] - omega[0]))


def cosine_frequency(times, correlation, *, maximum_time=15.):
    """Descriptive least-squares phase frequency, not a quantum line estimator."""
    keep = times <= maximum_time
    t, y = times[keep], correlation[keep]
    def residual(omega):
        design = np.stack((np.cos(omega * t), np.sin(omega * t)), axis=1)
        fit = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        return float(np.mean((fit - y) ** 2))
    return float(minimize_scalar(residual, bounds=(.8, 1.5), method="bounded",
                                 options={"xatol": 1e-12}).x)


def mean_error(values):
    values = np.asarray(values)
    mean = np.mean(values, axis=0)
    error = np.std(values, axis=0, ddof=1) / np.sqrt(len(values))
    return mean, error


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def code_identity():
    paths = [Path(__file__), ROOT / "src/pyrpmd/cmd.py", ROOT / "src/pyrpmd/dynamics.py",
             ROOT / "src/pyrpmd/analysis/spectroscopy.py"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


@dataclass(frozen=True)
class Case:
    label: str
    beads: int = 32
    omega: float = 8.
    dt: float = .01
    friction_scale: float = .5


def all_cases():
    return [Case("classical", beads=1), Case("baseline"),
            *(Case(f"omega_{frequency:g}", omega=frequency) for frequency in (1., 2., 4., 16.)),
            Case("dt_0.02", dt=.02), Case("dt_0.005", dt=.005),
            Case("beads_16", beads=16), Case("beads_64", beads=64)]


def run_case(case, *, seeds, atoms, duration, sample_dt, output, reference):
    parameters = {**asdict(case), "seeds": seeds, "atoms": atoms, "duration": duration,
                  "sample_dt": sample_dt, "code": code_identity()}
    identity = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
    cache = output / f"case-{case.label}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["fingerprint"]) == identity:
                return {key: data[key].copy() for key in data.files if key != "fingerprint"}
    sample_steps = int(round(sample_dt / case.dt))
    total = int(round(duration / case.dt))
    if not np.isclose(sample_steps * case.dt, sample_dt) or total % sample_steps:
        raise ValueError("sample_dt and duration must be exact multiples of each timestep")
    frames = total // sample_steps + 1
    nfft = 4 * (1 << (frames - 1).bit_length())
    times = np.arange(frames) * sample_dt
    omega, exact_ir = expected_windowed_ir(reference, frames, sample_dt, nfft)
    correlations, spectra, coordinate_second_moment, kinetic_second_moment, acceptances = [], [], [], [], []
    started = time.perf_counter()
    for seed in seeds:
        initial_rng = np.random.default_rng(np.random.SeedSequence([seed, 1729, case.beads]))
        q, acceptance = canonical_rings(case.beads, 3 * atoms, initial_rng)
        dynamics = PACMDDynamics(QuarticPotential(), np.ones(atoms), BETA, case.beads, case.dt,
            target_frequency=case.omega, friction_scale=case.friction_scale,
            seed=np.random.default_rng(np.random.SeedSequence([seed, 2718])))
        state = State(q.reshape(case.beads, atoms, 3), dynamics.sample_momenta())
        centroids = np.empty((frames, atoms, 3))
        velocities = np.empty_like(centroids)
        centroids[0], velocities[0] = state.q.mean(axis=0), state.p.mean(axis=0)
        for step in range(1, total + 1):
            dynamics.step(state)
            if step % sample_steps == 0:
                index = step // sample_steps
                centroids[index], velocities[index] = state.q.mean(axis=0), state.p.mean(axis=0)
        correlations.append(np.mean(centroids * centroids[0], axis=(1, 2)))
        coordinate_second_moment.append(np.mean(centroids * centroids, axis=(1, 2)))
        kinetic_second_moment.append(np.mean(velocities * velocities, axis=(1, 2)))
        # Each atom is an independent vector of three independent scalar copies.
        # Divide its dot-product spectrum by three for the one-dimensional model.
        analysis = relative_ir_spectrum(centroids.transpose(1, 0, 2)[:, :, None, :], sample_dt,
            beta=BETA, normalize=False, window="hann", nfft=nfft)
        spectra.append(analysis["unnormalized_relative_ir"] / 3)
        acceptances.append(acceptance)
    result = {"time": times, "omega": omega, "correlation": np.asarray(correlations),
              "ir": np.asarray(spectra), "exact_ir": exact_ir,
              "q2": np.asarray(coordinate_second_moment), "v2": np.asarray(kinetic_second_moment),
              "acceptance": np.asarray(acceptances), "runtime_seconds": np.asarray(time.perf_counter() - started)}
    np.savez_compressed(cache, fingerprint=identity, **result)
    print(f"{case.label}: {float(result['runtime_seconds']):.2f}s", flush=True)
    return result


def harmonic_limit():
    """Measured centroid frequency, and independent linear invariant covariance."""
    rows = []
    beads = 32
    transform, ring_frequencies = normal_modes(beads)
    for target in (2., 8., 16.):
        for dt in (.1, .05, .025, .0125):
            dynamics = PACMDDynamics(QuarticPotential(0), [1.], BETA, beads, dt,
                                     target_frequency=target, friction_scale=.5, seed=42)
            state = State(np.ones((beads, 1, 3)), np.zeros((beads, 1, 3)))
            trace = [1.]
            for _ in range(int(round(30 / dt))):
                dynamics.step(state)
                trace.append(float(state.q[:, 0, 0].mean()))
            times = np.arange(len(trace)) * dt
            measured = cosine_frequency(times, np.asarray(trace), maximum_time=30)
            expected = 2 / dt * np.arcsin(dt / 2)
            covariance_errors, map_errors = [], []
            for mode, ring in enumerate(ring_frequencies):
                mass = 1. if mode == 0 else (ring / target) ** 2
                frequency = 0. if mode == 0 else target
                gamma = .7 if mode == 0 else target
                phase = frequency * dt / 2
                drift = dt / 2 * np.sinc(phase / np.pi)
                free = np.array([[np.cos(phase), drift / mass],
                                 [-mass * frequency ** 2 * drift, np.cos(phase)]])
                kick = np.array([[1., 0.], [-dt / 2, 1.]])
                thermostat = np.diag([1., np.exp(-gamma * dt)])
                propagation = kick @ free @ thermostat @ free @ kick
                injection = kick @ free @ np.array([0., np.sqrt(mass * beads / BETA * -np.expm1(-2 * gamma * dt))])
                covariance = solve_discrete_lyapunov(propagation, np.outer(injection, injection))
                canonical_qvar = beads / BETA / (ring * ring + 1.)
                covariance_errors.append(abs(covariance[0, 0] / canonical_qvar - 1.))
                outputs = []
                for q0, p0 in ((0., 0.), (1., 0.), (0., 1.)):
                    q, p = np.zeros((beads, 1, 3)), np.zeros((beads, 1, 3))
                    q[:, 0, 0], p[:, 0, 0] = transform[:, mode] * q0, transform[:, mode] * p0
                    sample = State(q, p)
                    dynamics.rng = np.random.default_rng(131)
                    dynamics.sample_step(sample, gamma_centroid=.7)
                    outputs.append(np.array([transform[:, mode] @ sample.q[:, 0, 0],
                                             transform[:, mode] @ sample.p[:, 0, 0]]))
                measured_map = np.stack([outputs[1] - outputs[0], outputs[2] - outputs[0]], axis=1)
                map_errors.append(np.max(np.abs(measured_map - propagation)))
            rows.append({"dt": dt, "omega_internal": target, "physical_frequency": 1.,
                         "measured_frequency": measured, "verlet_frequency": expected,
                         "frequency_error": measured - 1., "map_frequency_error": measured - expected,
                         "trajectory_max_error": float(np.max(np.abs(trace - np.cos(times)))),
                         "max_mode_configurational_covariance_relative_error": max(covariance_errors),
                         "linear_map_max_abs_error": max(map_errors)})
    return rows


def fcmd_force_match(*, seeds, training_frames, output, times, reference):
    """Train the shipped conservative quadratic fitter on independent PI samples.

    The exact dynamics of each fitted quadratic is evaluated analytically,
    isolating PMF basis/sampling error from a second MD error. This does not
    call a quadratic fit a general anharmonic f-CMD potential.
    """
    rows, correlations, hessians, holdout_rows = [], [], [], []
    for seed in seeds:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 31415]))
        values, acceptance = canonical_rings(32, training_frames * 3, rng)
        rings = values.reshape(32, training_frames, 1, 3).transpose(1, 0, 2, 3)
        forces = -rings * (1 + .4 * rings * rings)
        fitted = fit_quadratic_centroid_pmf(rings, forces, beta=BETA)
        potential = fitted.potential
        hessian = potential.hessian
        eigenvalues = np.linalg.eigvalsh(hessian)
        if np.any(eigenvalues <= 0):
            raise RuntimeError("fitted quadratic is not a stable confining PMF")
        heldout, _ = canonical_rings(32, training_frames * 3, rng)
        heldout = heldout.reshape(32, training_frames, 1, 3).transpose(1, 0, 2, 3)
        target = (-heldout * (1 + .4 * heldout * heldout)).mean(axis=1)
        centroids = heldout.mean(axis=1)
        predictions = potential.evaluate(centroids).forces
        error = predictions - target
        scalar_x, scalar_f = centroids.reshape(-1), target.reshape(-1)
        edges = np.linspace(-1.2, 1.2, 25)
        for left, right in zip(edges[:-1], edges[1:]):
            selected = (scalar_x >= left) & (scalar_x < right)
            count = int(selected.sum())
            if count >= 20:
                holdout_rows.append({"seed": seed, "x": (left + right) / 2, "count": count,
                    "mean_force": float(np.mean(scalar_f[selected])),
                    "force_standard_error": float(np.std(scalar_f[selected], ddof=1) / np.sqrt(count))})
        correlation = np.mean(np.cos(np.sqrt(eigenvalues)[:, None] * times) / (BETA * eigenvalues[:, None]), axis=0)
        correlations.append(correlation)
        hessians.append(hessian)
        keep = times <= 15
        rows.append({"seed": seed, "training_frames": training_frames, "acceptance": acceptance,
                     "training_force_rmse": fitted.force_rmse,
                     "heldout_force_rmse": float(np.sqrt(np.mean(error * error))),
                     "hessian_mean_eigenvalue": float(np.mean(eigenvalues)),
                     "frequency_mean": float(np.mean(np.sqrt(eigenvalues))),
                     "frequency_min": float(np.min(np.sqrt(eigenvalues))),
                     "frequency_max": float(np.max(np.sqrt(eigenvalues))),
                     "c0": float(correlation[0]),
                     "normalized_correlation_rmse_t15": float(np.sqrt(np.mean((correlation[keep] - exact_correlation(reference, times[keep])) ** 2)) / reference["c0"])})
    write_csv(output / "fcmd-per-seed.csv", rows)
    write_csv(output / "fcmd-heldout-force.csv", holdout_rows)
    np.savez_compressed(output / "fcmd-fit.npz", time=times, correlation=correlations, hessian=hessians)
    return rows, np.asarray(correlations), holdout_rows


def summarize_case(case, data, reference, seeds):
    times, omega = data["time"], data["omega"]
    exact = exact_correlation(reference, times)
    early = times <= 15
    band = (omega >= .5) & (omega <= 2.)
    expected = data["exact_ir"]
    expected_area = np.trapezoid(expected[band], omega[band])
    expected_peak = peak_frequency(omega, expected)
    rows = []
    for replica, seed in enumerate(seeds):
        correlation, intensity = data["correlation"][replica], data["ir"][replica]
        area = np.trapezoid(intensity[band], omega[band])
        spectrum_difference = intensity[band] / area - expected[band] / expected_area
        spectrum_l2 = np.sqrt(np.trapezoid(spectrum_difference ** 2, omega[band]) /
                              np.trapezoid((expected[band] / expected_area) ** 2, omega[band]))
        rows.append({"case": case.label, "seed": seed, "beads": case.beads, "dt": case.dt,
                     "omega_internal": case.omega, "friction_scale": case.friction_scale,
                     "c0": float(correlation[0]), "c0_error": float(correlation[0] - reference["c0"]),
                     "correlation_rmse_t15": float(np.sqrt(np.mean((correlation[early] - exact[early]) ** 2))),
                     "normalized_correlation_rmse_t15": float(np.sqrt(np.mean((correlation[early] - exact[early]) ** 2)) / reference["c0"]),
                     "phase_fit_frequency_t15": cosine_frequency(times, correlation),
                     "ir_peak_frequency": peak_frequency(omega, intensity),
                     "ir_peak_frequency_error": peak_frequency(omega, intensity) - expected_peak,
                     "normalized_ir_shape_l2": float(spectrum_l2),
                     "mean_centroid_variance": float(data["q2"][replica].mean()),
                     "mean_centroid_velocity_variance": float(data["v2"][replica].mean()),
                     "canonical_initial_acceptance": float(data["acceptance"][replica])})
    summary = {"case": case.label, "beads": case.beads, "dt": case.dt,
               "omega_internal": case.omega, "seeds": len(seeds),
               "runtime_seconds": float(data["runtime_seconds"])}
    for key in ("c0", "correlation_rmse_t15", "normalized_correlation_rmse_t15", "phase_fit_frequency_t15",
                "ir_peak_frequency", "ir_peak_frequency_error", "normalized_ir_shape_l2",
                "mean_centroid_variance", "mean_centroid_velocity_variance"):
        mean, error = mean_error([row[key] for row in rows])
        summary[key], summary[key + "_standard_error"] = float(mean), float(error)
    mean_corr = data["correlation"].mean(axis=0)
    summary["ensemble_correlation_normalized_rmse_t15"] = float(np.sqrt(np.mean((mean_corr[early] - exact[early]) ** 2)) / reference["c0"])
    return summary, rows


def make_figures(output, cases, data, reference, harmonic, fits, fcmd_correlations, holdout):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for frequency in (2., 8., 16.):
        chosen = sorted((row for row in harmonic if row["omega_internal"] == frequency), key=lambda row: row["dt"])
        axes[0].loglog([row["dt"] for row in chosen], [abs(row["frequency_error"]) for row in chosen], "o-", label=f"Ω={frequency:g}")
        axes[1].loglog([row["dt"] for row in chosen], [max(1e-16, row["max_mode_configurational_covariance_relative_error"]) for row in chosen], "o-", label=f"Ω={frequency:g}")
    axes[0].set(xlabel="Timestep (a.u.)", ylabel="Centroid frequency absolute error", title="Harmonic limit: ω=1")
    axes[1].set(xlabel="Timestep (a.u.)", ylabel="Maximum relative position-variance error", title="Independent discrete Lyapunov covariance")
    for ax in axes:
        ax.legend()
        ticks = [.0125, .025, .05, .1]
        ax.set_xticks(ticks, labels=[f"{value:g}" for value in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
    fig.savefig(output / "harmonic-limit.png", dpi=180)
    plt.close(fig)
    baseline = data["baseline"]
    times = baseline["time"]
    confidence = float(student_t.ppf(.975, len(baseline["correlation"]) - 1))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    exact = exact_correlation(reference, times)
    axes[0].plot(times, exact, color="black", label="Exact quantum Kubo (calculated)")
    for label, caption in (("classical", "Classical"), ("baseline", "PA-CMD P=32, Ω=8")):
        mean, error = mean_error(data[label]["correlation"])
        line, = axes[0].plot(times, mean, label=caption)
        axes[0].fill_between(times, mean - confidence * error, mean + confidence * error, color=line.get_color(), alpha=.17)
        axes[1].plot(times, (mean - exact) / reference["c0"], color=line.get_color(), label=caption)
    mean, error = mean_error(fcmd_correlations)
    axes[0].plot(times, mean, "--", label="Quadratic force-matched PMF")
    axes[1].plot(times, (mean - exact) / reference["c0"], "--", label="Quadratic PMF")
    axes[0].set(ylabel="Kubo position correlation", title="Cao–Voth quartic, β=5; bands: seed-group 95% t intervals")
    axes[1].set(xlabel="Time (a.u.)", ylabel="Error / exact C(0)", xlim=(0, 15))
    axes[0].legend(fontsize=9)
    axes[1].axhline(0, color="black", lw=.6)
    fig.savefig(output / "quartic-correlation.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    omega = baseline["omega"]
    normalization = baseline["exact_ir"].max()
    ax.plot(omega, baseline["exact_ir"] / normalization, color="black", label="Exact quantum, same record/taper")
    for label, caption in (("classical", "Classical"), ("baseline", "PA-CMD Ω=8"), ("omega_1", "PA-CMD Ω=1")):
        mean, error = mean_error(data[label]["ir"])
        line, = ax.plot(omega, mean / normalization, label=caption)
        ax.fill_between(omega, (mean - confidence * error) / normalization,
                        (mean + confidence * error) / normalization, color=line.get_color(), alpha=.17)
    ax.axvline(reference["fundamental"], ls=":", color="gray", label="Exact E₁−E₀")
    ax.set(xlim=(.7, 1.7), ylim=(0, None), xlabel="Angular frequency (a.u.)",
           ylabel="Relative IR / common exact peak", title="Linear dipole q; finite-record Hann broadening retained")
    ax.legend(fontsize=9)
    fig.savefig(output / "quartic-spectrum.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    selections = [([case for case in cases if case.label in ("baseline", "omega_1", "omega_2", "omega_4", "omega_16")], "omega", "Internal Ω (a.u.)"),
                  ([case for case in cases if case.label in ("baseline", "dt_0.02", "dt_0.005")], "dt", "Timestep (a.u.)"),
                  ([case for case in cases if case.label in ("baseline", "beads_16", "beads_64")], "beads", "Beads")]
    for ax, (chosen, field, label) in zip(axes, selections):
        chosen = sorted(chosen, key=lambda case: getattr(case, field))
        for case in chosen:
            values = [peak_frequency(data[case.label]["omega"], row) for row in data[case.label]["ir"]]
            mean, error = mean_error(values)
            ax.errorbar(getattr(case, field), mean, yerr=confidence * error, fmt="o", color="tab:blue", capsize=3)
        ax.axhline(peak_frequency(omega, baseline["exact_ir"]), color="black", ls="--", label="Windowed exact peak")
        ax.set(xlabel=label, ylabel="IR peak frequency (a.u.)")
        ticks = [getattr(case, field) for case in chosen]
        ax.set_xticks(ticks, labels=[f"{value:g}" for value in ticks])
    axes[0].legend(fontsize=8)
    fig.suptitle("Sensitivity and independent-seed 95% t intervals; not physical convergence certification")
    fig.savefig(output / "sensitivity.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for x in sorted({row["x"] for row in holdout}):
        values = [row["mean_force"] for row in holdout if row["x"] == x]
        if len(values) > 1:
            mean, error = mean_error(values)
            ax.errorbar(x, mean, yerr=confidence * error, fmt=".", color="tab:blue", capsize=2)
    x = np.linspace(-1.2, 1.2, 200)
    ax.plot(x, -x - .4 * x ** 3, color="gray", ls=":", label="Physical force")
    curvature = np.mean([row["hessian_mean_eigenvalue"] for row in fits])
    ax.plot(x, -curvature * x, label="Fitted quadratic centroid force")
    ax.plot([], [], ".", color="tab:blue", label="Held-out conditional mean, binned")
    ax.set(xlabel="Centroid q (a.u.)", ylabel="Force (a.u.)", title="Finite basis limitation: the fitted PMF is only quadratic")
    ax.legend(fontsize=9)
    fig.savefig(output / "fcmd-force-fit.png", dpi=180)
    plt.close(fig)


def run(output, *, quick=False):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = [1943 + 104729 * index for index in range(2 if quick else 8)]
    atoms, duration, training_frames = (4, 8., 1024) if quick else (32, 32., 16384)
    sample_dt = .02
    cases = all_cases()
    references = [exact_reference(size) for size in (40, 60, 100, 160)]
    reference = references[-1]
    parameters = {"model": "Cao-Voth-1993-Eq18", "potential": "q^2/2 + (2.4/24)*q^4",
        "beta": BETA, "mass": 1., "hbar": 1., "units": "atomic; frequencies angular",
        "quick_smoke_only": quick, "master_seeds": seeds, "atoms_per_seed": atoms,
        "independent_scalar_copies_per_seed": 3 * atoms, "duration": duration, "sample_dt": sample_dt,
        "training_frames_per_seed": training_frames, "cases": [asdict(case) for case in cases],
        "uncertainty": "standard error across master-seed groups; displayed 95% Student-t intervals",
        "initial_ensemble": "independent exact finite-P rejection samples, harmonic-ring proposal",
        "code_sha256": code_identity(), "python": platform.python_version(), "numpy": np.__version__}
    try:
        parameters["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        parameters["git_head"] = None
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    basis_rows = [{"basis": item["basis"], "ground_energy": item["ground_energy"],
        "fundamental": item["fundamental"], "c0_kubo": item["c0"], "q2_ordinary": item["q2"],
        "minus_correlation_curvature": -item["curvature"], "curvature_sum_rule": 1 / BETA,
        "ground_energy_shift_percent": 100 * (item["ground_energy"] / .5 - 1)} for item in references]
    write_csv(output / "exact-basis-convergence.csv", basis_rows)
    write_csv(output / "exact-transition-lines.csv", ({"lower": int(low), "upper": int(high),
        "angular_frequency": float(gap), "kubo_cosine_weight": float(weight),
        "relative_ir_line_weight": float(BETA * gap * gap * weight)}
        for low, high, gap, weight in zip(reference["low"], reference["high"], reference["gaps"], reference["weights"])))
    harmonic = harmonic_limit()
    write_csv(output / "harmonic-limit.csv", harmonic)
    data = {case.label: run_case(case, seeds=seeds, atoms=atoms, duration=duration,
            sample_dt=sample_dt, output=output, reference=reference) for case in cases}
    summaries, per_seed = [], []
    for case in cases:
        summary, rows = summarize_case(case, data[case.label], reference, seeds)
        summaries.append(summary)
        per_seed += rows
    write_csv(output / "cases.csv", summaries)
    write_csv(output / "per-seed.csv", per_seed)
    write_csv(output / "time-correlations.csv", ({"case": case.label, "time": float(t),
        "correlation": float(mean), "standard_error": float(error),
        "exact_kubo": float(exact)} for case in cases
        for t, mean, error, exact in zip(data[case.label]["time"], *mean_error(data[case.label]["correlation"]),
            exact_correlation(reference, data[case.label]["time"]))))
    write_csv(output / "spectra.csv", ({"case": case.label, "angular_frequency": float(omega),
        "relative_ir": float(mean), "standard_error": float(error), "exact_windowed_ir": float(exact)}
        for case in cases for omega, mean, error, exact in zip(data[case.label]["omega"],
            *mean_error(data[case.label]["ir"]), data[case.label]["exact_ir"]) if omega <= 6))
    paired = []
    baseline = [row for row in per_seed if row["case"] == "baseline"]
    for case in cases:
        selected = [row for row in per_seed if row["case"] == case.label]
        for metric in ("ir_peak_frequency", "phase_fit_frequency_t15", "mean_centroid_variance"):
            differences = [row[metric] - base[metric] for row, base in zip(selected, baseline)]
            mean, error = mean_error(differences)
            paired.append({"case": case.label, "reference_case": "baseline", "metric": metric,
                           "paired_difference": float(mean), "paired_standard_error": float(error)})
    write_csv(output / "paired-sensitivity.csv", paired)
    fits, fcmd_correlations, holdout = fcmd_force_match(seeds=seeds, training_frames=training_frames,
        output=output, times=data["baseline"]["time"], reference=reference)
    make_figures(output, cases, data, reference, harmonic, fits, fcmd_correlations, holdout)
    summary = {"status": "complete", "quick_smoke_only": quick,
        "reference_type": "independently calculated exact Kubo reference for a published model",
        "paper_figure_digitized": False, "paper_model_equation": 18, "paper_model_page": 10072,
        "paper_ground_shift_rounded_percent": 12., "calculated_ground_shift_percent": basis_rows[-1]["ground_energy_shift_percent"],
        "exact_fundamental": reference["fundamental"], "exact_kubo_c0": reference["c0"],
        "basis_100_to_160_fundamental_difference": abs(references[-2]["fundamental"] - reference["fundamental"]),
        "basis_100_to_160_c0_difference": abs(references[-2]["c0"] - reference["c0"]),
        "harmonic_linear_map_max_abs_error": max(row["linear_map_max_abs_error"] for row in harmonic),
        "harmonic_verlet_frequency_max_abs_error": max(abs(row["map_frequency_error"]) for row in harmonic),
        "cases": summaries, "fcmd": {key: {"mean": float(mean_error([row[key] for row in fits])[0]),
            "standard_error": float(mean_error([row[key] for row in fits])[1])}
            for key in ("frequency_mean", "heldout_force_rmse", "c0", "normalized_correlation_rmse_t15")},
        "wall_seconds": time.perf_counter() - started,
        "physical_converged": False,
        "limitations": ["1-D linear-dipole model, not a molecular water/H2/D2 or Raman/SFG validation",
            "PA-CMD approaches approximate centroid dynamics, not exact anharmonic quantum dynamics",
            "finite-P and finite-step errors, finite trajectory/seed uncertainty are reported separately",
            "quadratic f-CMD basis cannot represent a general anharmonic centroid PMF",
            "Hone 2005/2006 primary abstracts read; original full text inaccessible in this environment",
            "paper Fig. 1 uses the real quantum correlation; our comparison uses exact Kubo correlations"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output": str(output), "wall_seconds": summary["wall_seconds"]}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, quick=args.quick)
