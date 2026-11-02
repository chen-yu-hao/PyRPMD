"""Reproducible f-QCMD research example for an analytic gas-phase water model.

The potential is a water-inspired quartic expansion of two Morse stretches,
plus a harmonic bend.  Its parameters below are deliberately specified in
atomic units: this is NOT q-TIP4P/F or the Partridge--Schwenke water surface.
Independent, noninteracting molecules share a vectorized force evaluation;
their molecular spectra are averaged as replicas, without inter-molecule
dipole cross terms.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
      examples/spectroscopy/fqcmd_water.py --output runs/fqcmd-water

The default run is a pilot, not a convergence claim. Repeat with more beads,
longer sampling, independent seeds, smaller timesteps, and larger PA-CMD
target frequencies before interpreting its frequency difference. In
particular, 32 beads at 150 K have not been established as converged.

Training follows Fletcher et al., JCP 155, 231101 (2021),
https://doi.org/10.1063/5.0076704: standard unconstrained PIMD distributions,
iterative Boltzmann inversion of separable corrections, then classical MD.
The primary comparator is the existing PA-CMD implementation. The optional
``centroid-ibi`` comparator applies the same separable IBI fit to Cartesian
centroid distributions; it is a different approximation from PA-CMD and from
PyRPMD's existing quadratic force-matched f-CMD. No liquid-water IR claim or
exact quantum reference is implied by this illustrative potential.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from pyrpmd.analysis.spectroscopy import relative_ir_spectrum
from pyrpmd.checkpoint import atomic_savez
from pyrpmd.cmd import PACMDDynamics
from pyrpmd.constants import AMU_TO_AU, FS_TO_AU, KB_AU, WAVENUMBER_TO_AU
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.io import atomic_write_text
from pyrpmd.potentials.base import PESResult, Potential, PotentialError, coordinates
from pyrpmd.system import System


class AnalyticWaterPotential(Potential):
    """Independent bent O,H,H molecules; E in hartree and F in hartree/bohr.

    V = D sum_i [(a dr_i)^2 - (a dr_i)^3 + 7 (a dr_i)^4 / 12]
        + k_theta (theta-theta0)^2 / 2.

    This globally confining quartic has an OH harmonic frequency near the
    water stretching region. All parameters are illustrative, not fitted
    spectroscopic constants. No inter-molecule interactions are evaluated.
    """

    parameters = {
        "D_hartree": 0.18,
        "a_inverse_bohr": 1.2,
        "r0_bohr": 1.8,
        "theta0_radian": float(np.deg2rad(104.5)),
        "k_theta_hartree": 0.15,
        "hydrogen_charge_e": 0.4,
        "masses_amu": [15.99491461957, 1.00782503223, 1.00782503223],
    }
    metadata = {
        "native_batch": True,
        "execution_device": "cpu",
        "thread_safety": "thread_safe",
        "model": "illustrative_noninteracting_quartic_water_v1",
    }

    def evaluate(self, x):
        q = coordinates(x)
        if q.shape[1] % 3:
            raise PotentialError("analytic water requires contiguous O,H,H triples")
        local = q.reshape(len(q), -1, 3, 3)
        a = local[:, :, 1] - local[:, :, 0]
        b = local[:, :, 2] - local[:, :, 0]
        r1, r2 = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
        if np.any(r1 < 1e-10) or np.any(r2 < 1e-10):
            raise PotentialError("analytic water contains a zero OH bond")
        u, v = a / r1[..., None], b / r2[..., None]
        cosine = np.clip(np.sum(u * v, axis=-1), -1.0, 1.0)
        sine = np.linalg.norm(np.cross(u, v), axis=-1)
        if np.any(sine < 1e-10):
            raise PotentialError("analytic water requires non-collinear bonds")
        angle = np.arctan2(sine, cosine)
        p = self.parameters
        dr = np.stack((r1, r2), axis=-1) - p["r0_bohr"]
        scaled = p["a_inverse_bohr"] * dr
        stretch = p["D_hartree"] * (scaled**2 - scaled**3 + 7 * scaled**4 / 12)
        derivative = p["D_hartree"] * p["a_inverse_bohr"] * (
            2 * scaled - 3 * scaled**2 + 7 * scaled**3 / 3
        )
        displacement = angle - p["theta0_radian"]
        bend = 0.5 * p["k_theta_hartree"] * displacement**2
        angular_derivative = p["k_theta_hartree"] * displacement
        dtheta_da = (cosine[..., None] * u - v) / (r1 * sine)[..., None]
        dtheta_db = (cosine[..., None] * v - u) / (r2 * sine)[..., None]
        force = np.empty_like(local)
        force[:, :, 1] = -(derivative[..., 0, None] * u
                              + angular_derivative[..., None] * dtheta_da)
        force[:, :, 2] = -(derivative[..., 1, None] * v
                              + angular_derivative[..., None] * dtheta_db)
        force[:, :, 0] = -force[:, :, 1] - force[:, :, 2]
        return PESResult(np.sum(stretch.sum(axis=-1) + bend, axis=1),
                         force.reshape(q.shape))

    def molecular_dipoles(self, x):
        """Neutral linear charge dipole, one vector per molecule, in e*bohr."""
        q = coordinates(x)
        if q.shape[1] % 3:
            raise PotentialError("analytic water requires contiguous O,H,H triples")
        local = q.reshape(len(q), -1, 3, 3)
        return self.parameters["hydrogen_charge_e"] * (
            local[:, :, 1] + local[:, :, 2] - 2 * local[:, :, 0]
        )

    @property
    def identity(self):
        payload = json.dumps(self.parameters, sort_keys=True, separators=(",", ":"))
        return "analytic-quartic-water:v1:" + hashlib.sha256(payload.encode()).hexdigest()

    def equilibrium(self, molecules=1):
        """Separated copies in O,H,H order; no interactions depend on spacing."""
        if isinstance(molecules, bool) or not isinstance(molecules, int) or molecules < 1:
            raise ValueError("molecules must be a positive integer")
        p = self.parameters
        q = np.zeros((molecules, 3, 3))
        q[:, 1, 0] = p["r0_bohr"] * np.cos(p["theta0_radian"] / 2)
        q[:, 2, 0] = q[:, 1, 0]
        q[:, 1, 1] = p["r0_bohr"] * np.sin(p["theta0_radian"] / 2)
        q[:, 2, 1] = -q[:, 1, 1]
        q[:, :, 2] = 8 * np.arange(molecules)[:, None]
        return q.reshape(-1, 3)

    def masses(self, molecules=1):
        return np.tile(np.asarray(self.parameters["masses_amu"]) * AMU_TO_AU,
                       molecules)


def build_potential():
    """Importable factory for a spectroscopy input's ``potential.factory``."""
    return AnalyticWaterPotential()


def build_properties():
    """Total-system dipole surface for the standard spectroscopy workflow."""
    potential = AnalyticWaterPotential()

    def evaluate(q):
        return {"dipole": potential.molecular_dipoles(q).sum(axis=1)}

    return evaluate


def force_reference(potential, *, displacement=1e-5):
    """Central energy differences, including bent and distorted structures."""
    q = potential.equilibrium(2)[None]
    q += np.random.default_rng(314159).normal(scale=0.08, size=q.shape)
    analytical = potential.evaluate(q).forces.copy()
    numerical = np.empty_like(analytical)
    for atom in range(q.shape[1]):
        for axis in range(3):
            plus, minus = q.copy(), q.copy()
            plus[0, atom, axis] += displacement
            minus[0, atom, axis] -= displacement
            numerical[0, atom, axis] = -(potential.evaluate(plus).energy[0]
                                         - potential.evaluate(minus).energy[0]) / (2 * displacement)
    error = float(np.max(np.abs(analytical - numerical)))
    if error > 5e-8:
        raise RuntimeError(f"analytic water finite-difference force check failed: {error}")
    return {"max_absolute_force_error_hartree_per_bohr": error,
            "finite_difference_displacement_bohr": displacement}


def harmonic_frequencies(potential):
    """Isolated equilibrium mass-weighted Hessian reference, in cm^-1.

    The Hessian is differentiated from the analytic Cartesian forces; six
    translations/rotations are removed by retaining the three positive modes.
    These are classical small-amplitude frequencies, not quantum fundamentals.
    """
    q = potential.equilibrium()[None]
    hessian = np.empty((9, 9))
    displacement = 1e-5
    for column in range(9):
        plus, minus = q.copy(), q.copy()
        plus.reshape(-1)[column] += displacement
        minus.reshape(-1)[column] -= displacement
        hessian[:, column] = -(potential.evaluate(plus).forces.ravel()
                                - potential.evaluate(minus).forces.ravel()) / (2 * displacement)
    masses = np.repeat(potential.masses(), 3)
    eigenvalues = np.linalg.eigvalsh((hessian + hessian.T) / 2
                                    / np.sqrt(masses[:, None] * masses[None, :]))
    return (np.sqrt(eigenvalues[-3:]) / WAVENUMBER_TO_AU).tolist()


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _positive_float(text):
    value = float(text)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _nonnegative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def _nonnegative_float(text):
    value = float(text)
    if not np.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=Path("runs/fqcmd-water"))
    p.add_argument("--temperature", type=_positive_float, default=150.0)
    p.add_argument("--beads", type=_positive_int, default=32)
    p.add_argument("--molecules", type=_positive_int, default=32,
                   help="independent gas molecules vectorized as one noninteracting system")
    p.add_argument("--seed", type=_nonnegative_int, default=1729)
    p.add_argument("--dt-fs", type=_positive_float, default=0.1)
    p.add_argument("--pimd-warmup", type=_positive_int, default=2000)
    p.add_argument("--pimd-steps", type=_positive_int, default=8000)
    p.add_argument("--sample-every", type=_positive_int, default=10)
    p.add_argument("--ibi-warmup", type=_positive_int, default=1000)
    p.add_argument("--ibi-steps", type=_positive_int, default=8000)
    p.add_argument("--ibi-iterations", type=_positive_int, default=4)
    p.add_argument("--ibi-damping", type=_positive_float, default=0.5)
    p.add_argument("--ibi-regularization", type=_nonnegative_float, default=0.01)
    p.add_argument("--ibi-degree", type=_nonnegative_int, default=4)
    p.add_argument("--bond-bins", type=_positive_int, default=100)
    p.add_argument("--angle-bins", type=_positive_int, default=100)
    p.add_argument("--production-warmup", type=_positive_int, default=2000)
    p.add_argument("--production-steps", type=_positive_int, default=20000)
    p.add_argument("--spectrum-every", type=_positive_int, default=2)
    p.add_argument("--gamma-per-fs", type=_positive_float, default=0.05)
    p.add_argument("--pacmd-dt-fs", type=_positive_float, default=0.025)
    p.add_argument("--pacmd-target-cm1", type=_positive_float, default=14000.0)
    p.add_argument("--pacmd-friction-scale", type=_positive_float, default=1.0)
    p.add_argument("--comparison", choices=("pa-cmd", "centroid-ibi", "both", "none"),
                   default="pa-cmd")
    return p


def _new_dynamics(potential, masses, beta, beads, dt, seed):
    # This is an explicitly selected NumPy reference implementation. It does
    # not require, promise, or silently fall back from a fused native PES loop.
    return Dynamics(potential, masses, beta, beads, dt, seed=seed,
                    native=False, loop_backend="python", integrator="normal-mode-bab")


def _sample(dynamics, state, steps, gamma, *, save_every=None, observation=None):
    started = time.perf_counter()
    for step in range(1, steps + 1):
        dynamics.sample_step(state, gamma_centroid=gamma)
        if save_every is not None and step % save_every == 0:
            observation(state.q)
    return time.perf_counter() - started


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _distribution_error(target, sampled):
    # An in-sample histogram diagnostic; correlated saved frames do not give
    # an independent effective sample count or a physical convergence flag.
    return [float(0.5 * np.sum(np.abs(a / a.sum() - b / b.sum())))
            for a, b in zip(target, sampled)]


def _fit_model(initial, target, base, options, *, label, seed_offset):
    from pyrpmd.qcmd import FQCMDPotential, coordinate_histograms, update_fqcmd_model

    model = initial
    q = base.equilibrium(options.molecules)[None]
    records = []
    collected = {}
    for iteration in range(options.ibi_iterations + 1):
        seed = options.seed + seed_offset + iteration
        dynamics = _new_dynamics(FQCMDPotential(base, model), model.geometry.masses,
                                 model.beta, 1, options.dt_fs * FS_TO_AU, seed)
        state = State(q, dynamics.sample_momenta())
        gamma = options.gamma_per_fs / FS_TO_AU
        warm_wall = _sample(dynamics, state, options.ibi_warmup, gamma)
        counts = [np.zeros(len(edge) - 1, dtype=np.int64) for edge in model.edges]

        def observe(positions):
            values = coordinate_histograms(positions, model.geometry, model.groups,
                                           model.edges, bead_average=False)
            for total, incoming in zip(counts, values):
                total += incoming

        wall = _sample(dynamics, state, options.ibi_steps, gamma,
                       save_every=options.sample_every, observation=observe)
        q = state.q.copy()
        record = {"iteration": iteration, "seed": seed,
                  "histogram_total_variation": _distribution_error(target, counts),
                  "saved_frames": options.ibi_steps // options.sample_every,
                  "sampling_wall_seconds": wall, "warmup_wall_seconds": warm_wall,
                  "microseconds_per_batch_sampling_step": wall / options.ibi_steps * 1e6}
        collected[f"{label}_iteration_{iteration}_bond_counts"] = counts[0]
        collected[f"{label}_iteration_{iteration}_angle_counts"] = counts[1]
        print(f"{label} IBI {iteration}: TV={record['histogram_total_variation']} "
              f"sampling={wall:.2f}s", flush=True)
        if iteration < options.ibi_iterations:
            model, update = update_fqcmd_model(
                model, target, tuple(counts), damping=options.ibi_damping,
                regularization=options.ibi_regularization,
                polynomial_degree=options.ibi_degree,
            )
            record["update_diagnostics"] = _json_safe(update)
        records.append(record)
    model.save(options.output / f"{label}-model.npz")
    return model, records, collected


def _stretch_summary(spectrum):
    wave, intensity = spectrum["wavenumber_cm1"], spectrum["relative_ir"]
    chosen = (wave >= 2500) & (wave <= 4500)
    if not np.any(chosen) or np.sum(intensity[chosen]) <= 0:
        raise RuntimeError("production record has no resolved OH stretch-band signal")
    x, y = wave[chosen], intensity[chosen]
    return {"analysis_band_cm1": [2500.0, 4500.0],
            "peak_grid_cm1": float(x[np.argmax(y)]),
            "band_first_moment_cm1": float(np.sum(x * y) / np.sum(y)),
            "frequency_grid_spacing_cm1": float(wave[1] - wave[0]),
            "record_resolution_cm1": float(spectrum["resolution_au"] / WAVENUMBER_TO_AU * 2 * np.pi),
            "window": spectrum["window"], "peak_interpolation": False,
            "physical_converged": False}


def _production(base, potential, options, *, label, beta, seed, pimd_q=None):
    masses = base.masses(options.molecules)
    gamma = options.gamma_per_fs / FS_TO_AU
    if label == "pa-cmd":
        dt_fs = options.pacmd_dt_fs
        dynamics = PACMDDynamics(
            base, masses, beta, options.beads, dt_fs * FS_TO_AU,
            target_frequency=options.pacmd_target_cm1 * WAVENUMBER_TO_AU,
            friction_scale=options.pacmd_friction_scale, seed=seed,
        )
        q = pimd_q.copy()
        production_steps = int(round(options.production_steps * options.dt_fs / dt_fs))
        warmup_steps = int(round(options.production_warmup * options.dt_fs / dt_fs))
        save_every = int(round(options.spectrum_every * options.dt_fs / dt_fs))
        if save_every < 1:
            raise ValueError("PA-CMD timestep exceeds requested spectrum sampling cadence")
    else:
        dt_fs = options.dt_fs
        dynamics = _new_dynamics(potential, masses, beta, 1, dt_fs * FS_TO_AU, seed)
        q = base.equilibrium(options.molecules)[None]
        production_steps, warmup_steps = options.production_steps, options.production_warmup
        save_every = options.spectrum_every
    state = State(q, dynamics.sample_momenta())
    warm_wall = _sample(dynamics, state, warmup_steps, gamma)
    energy_initial = dynamics.energy(state)
    dipoles, energies = [], []
    started = time.perf_counter()
    for step in range(1, production_steps + 1):
        dynamics.step(state)
        if step % save_every == 0:
            dipoles.append(base.molecular_dipoles(state.q).mean(axis=0))
            if label != "pa-cmd":
                energies.append(dynamics.energy(state))
    wall = time.perf_counter() - started
    # [independent molecule, time, one effective bead, xyz]. Linear charges
    # make mean-bead dipole identical to the centroid dipole for PA-CMD.
    dipoles = np.asarray(dipoles).transpose(1, 0, 2)[:, :, None, :]
    spectrum = relative_ir_spectrum(dipoles, dt_fs * save_every * FS_TO_AU,
                                    beta=beta, window="hann", normalize=False)
    summary = _stretch_summary(spectrum)
    summary.update({"method": label, "seed": seed, "production_steps": production_steps,
                    "dt_fs": dt_fs, "warmup_steps": warmup_steps,
                    "duration_fs": production_steps * dt_fs,
                    "saved_frames": dipoles.shape[1], "sample_interval_fs": dt_fs * save_every,
                    "independent_noninteracting_molecules": options.molecules,
                    "production_wall_seconds": wall, "warmup_wall_seconds": warm_wall,
                    "microseconds_per_batch_production_step": wall / production_steps * 1e6,
                    "microseconds_per_molecule_production_step": wall / production_steps
                                                                        / options.molecules * 1e6,
                    "centroid_thermostatted_during_production": False})
    if label == "pa-cmd":
        summary.update({"internal_modes_thermostatted": True,
                        "target_wavenumber_cm1": options.pacmd_target_cm1,
                        "internal_friction_scale": options.pacmd_friction_scale,
                        "energy_conservation_test": "not applicable: internal modes are thermostatted"})
    else:
        summary.update({"internal_modes_thermostatted": False,
                        "max_nve_energy_drift_hartree_per_molecule":
                        float(np.max(np.abs(np.asarray(energies) - energy_initial)) / options.molecules)})
    atomic_savez(options.output / f"{label}-spectrum.npz", kind="PyRPMD.fqcmd-water-spectrum.v1",
                 dipoles_au=dipoles, wavenumber_cm1=spectrum["wavenumber_cm1"],
                 relative_ir=spectrum["relative_ir"],
                 relative_ir_standard_error=spectrum["relative_ir_standard_error"]
                 if spectrum["relative_ir_standard_error"] is not None else np.array([]),
                 sample_interval_au=np.array(spectrum["sample_interval_au"]),
                 nve_sampled_energy_hartree=np.asarray(energies))
    print(f"{label} OH band maximum: {summary['peak_grid_cm1']:.1f} cm^-1; "
          f"production={wall:.2f}s", flush=True)
    return summary


def run(options):
    from pyrpmd.qcmd import (FQCMDModel, FQCMDPotential, TriatomicQuasiCentroid,
                            coordinate_histograms, fqcmd_base_identity)

    if options.ibi_damping > 1:
        raise ValueError("IBI damping must lie in (0,1]")
    if options.ibi_degree > 8:
        raise ValueError("IBI polynomial degree must not exceed eight")
    if options.bond_bins < 4 or options.angle_bins < 4:
        raise ValueError("each interpolation grid requires at least four bins")
    if options.pimd_steps < options.sample_every or options.ibi_steps < options.sample_every:
        raise ValueError("PIMD/IBI runs must each contain at least one saved frame")
    if options.production_steps // options.spectrum_every < 8:
        raise ValueError("production requires at least eight saved spectrum frames")
    if options.comparison in ("pa-cmd", "both"):
        # Keep comparison duration/cadence exact instead of rounding the user's
        # scientific settings to an alternative observation grid.
        for count, name in ((options.production_steps, "production duration"),
                            (options.production_warmup, "production warmup"),
                            (options.spectrum_every, "spectrum cadence")):
            ratio = count * options.dt_fs / options.pacmd_dt_fs
            if ratio < 1 or not np.isclose(ratio, round(ratio), rtol=1e-12, atol=1e-12):
                raise ValueError(f"PA-CMD timestep must divide the {name} exactly")
    options.output.mkdir(parents=True, exist_ok=True)
    base = AnalyticWaterPotential()
    masses = base.masses(options.molecules)
    beta = 1 / (KB_AU * options.temperature)
    geometry = TriatomicQuasiCentroid(np.arange(3 * options.molecules).reshape(-1, 3), masses)
    system = System(tuple(["O", "H", "H"] * options.molecules), masses,
                    base.equilibrium(options.molecules))
    potential_spec = {"provider": "python", "factory": str(Path(__file__).resolve()) + ":build_potential",
                      "params": {}}
    base_identity = fqcmd_base_identity(potential_spec, system)
    groups = np.tile([0, 0, 1], (options.molecules, 1))
    edges = (np.linspace(0.5, 3.5, options.bond_bins + 1),
             np.linspace(0.15, np.pi - 0.15, options.angle_bins + 1))
    scientific_options = {key: value for key, value in vars(options).items() if key != "output"}
    identity = {"example": "analytic-quartic-water-fqcmd-v1",
                "base_potential_identity": base_identity, "potential_spec": potential_spec,
                "parameters": base.parameters, "settings": scientific_options}
    run_digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    started = time.perf_counter()
    print(f"Analytic gas water pilot: T={options.temperature:g} K, P={options.beads}, "
          f"independent molecules={options.molecules}; physical_converged=False", flush=True)
    force_check = force_reference(base)
    dynamics = _new_dynamics(base, masses, beta, options.beads,
                             options.dt_fs * FS_TO_AU, options.seed)
    q = np.broadcast_to(base.equilibrium(options.molecules),
                        (options.beads, 3 * options.molecules, 3)).copy()
    state = State(q, dynamics.sample_momenta())
    gamma = options.gamma_per_fs / FS_TO_AU
    warm_wall = _sample(dynamics, state, options.pimd_warmup, gamma)
    target = [np.zeros(len(edge) - 1, dtype=np.int64) for edge in edges]
    centroid = [np.zeros(len(edge) - 1, dtype=np.int64) for edge in edges]
    internal_frames = []
    centroid_frames = []

    def observe(positions):
        counts = coordinate_histograms(positions[None], geometry, groups, edges, bead_average=True)
        classical_counts = coordinate_histograms(positions.mean(axis=0)[None], geometry, groups,
                                                 edges, bead_average=False)
        for total, incoming in zip(target, counts):
            total += incoming
        for total, incoming in zip(centroid, classical_counts):
            total += incoming
        internal_frames.append(geometry.mean_coordinates(positions))
        centroid_frames.append(geometry.internal_coordinates(positions.mean(axis=0)))

    sampling_wall = _sample(dynamics, state, options.pimd_steps, gamma,
                            save_every=options.sample_every, observation=observe)
    print(f"PIMD: {len(internal_frames)} saved frames, {sampling_wall:.2f}s", flush=True)
    provenance = {"training": "unconstrained PIMD followed by separable IBI",
                  "run_digest": run_digest, "temperature_kelvin": options.temperature,
                  "pimd_beads": options.beads, "scientific_options": scientific_options,
                  "physical_converged": False}
    initial = FQCMDModel(geometry, groups, edges,
                        tuple(np.zeros(len(edge) - 1) for edge in edges),
                        beta=beta, base_identity=base_identity, provenance=provenance)
    model, fit_records, training = _fit_model(initial, tuple(target), base, options,
                                             label="f-qcmd", seed_offset=10000)
    production = {"f-qcmd": _production(base, FQCMDPotential(base, model), options,
                                        label="f-qcmd", beta=beta, seed=options.seed + 20000)}
    all_fit_records = {"f-qcmd": fit_records}
    if options.comparison in ("centroid-ibi", "both"):
        centroid_provenance = {**provenance,
                              "training": "Cartesian-centroid-distribution separable IBI comparator",
                              "is_fqcmd": False}
        centroid_initial = FQCMDModel(geometry, groups, edges,
                                     tuple(np.zeros(len(edge) - 1) for edge in edges),
                                     beta=beta, base_identity=base_identity,
                                     provenance=centroid_provenance)
        comparator, records, histograms = _fit_model(
            centroid_initial, tuple(centroid), base, options,
            label="centroid-ibi", seed_offset=40000)
        training.update(histograms)
        all_fit_records["centroid-ibi"] = records
        production["centroid-ibi"] = _production(
            base, FQCMDPotential(base, comparator), options, label="centroid-ibi",
            beta=beta, seed=options.seed + 50000)
    if options.comparison in ("pa-cmd", "both"):
        production["pa-cmd"] = _production(base, base, options, label="pa-cmd", beta=beta,
                                           seed=options.seed + 30000, pimd_q=state.q)
    differences = {
        name: {"f_qcmd_minus_comparator_peak_cm1":
               production["f-qcmd"]["peak_grid_cm1"] - value["peak_grid_cm1"],
               "f_qcmd_minus_comparator_band_first_moment_cm1":
               production["f-qcmd"]["band_first_moment_cm1"] - value["band_first_moment_cm1"]}
        for name, value in production.items() if name != "f-qcmd"
    }
    atomic_savez(options.output / "training-distributions.npz", kind="PyRPMD.fqcmd-water-training.v1",
                 bond_edges_bohr=edges[0], angle_edges_radian=edges[1],
                 target_bond_counts=target[0], target_angle_counts=target[1],
                 centroid_target_bond_counts=centroid[0], centroid_target_angle_counts=centroid[1],
                 quasi_centroid_internal_coordinates=np.asarray(internal_frames),
                 cartesian_centroid_internal_coordinates=np.asarray(centroid_frames), **training)
    hardware = {"platform": platform.platform(), "machine": platform.machine(),
                "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
                "implementation": "explicit NumPy reference dynamics; vectorized independent molecules"}
    report = {"schema": "PyRPMD.fqcmd-water-reference.v1", "run_digest": run_digest,
              "identity": identity, "hardware": hardware, "force_reference": force_check,
              "classical_harmonic_frequencies_cm1": harmonic_frequencies(base),
              "pimd": {"seed": options.seed, "saved_frames": len(internal_frames),
                        "warmup_wall_seconds": warm_wall, "sampling_wall_seconds": sampling_wall,
                        "microseconds_per_batch_sampling_step": sampling_wall / options.pimd_steps * 1e6,
                        "saved_frame_interval_fs": options.sample_every * options.dt_fs,
                        "frames_are_not_claimed_independent": True},
              "ibi": all_fit_records, "production": production, "frequency_differences": differences,
              "total_wall_seconds": time.perf_counter() - started,
              "physical_converged": False,
              "unconverged_parameters": ["PIMD bead count and timestep", "equilibration and statistical sampling",
                                         "IBI histogram grid, polynomial smoothing and regularization",
                                         "separable correction approximation and IBI iteration count",
                                         "production timestep, duration, window and independent seed repetitions",
                                         "PA-CMD adiabatic separation and internal friction"],
              "literature_comparison": {
                  "expected_direction": "f-QCMD removes the artificial CMD stretch red shift",
                  "same_potential_as_literature": False,
                  "reference": "https://doi.org/10.1063/5.0076704",
                  "caveat": "This analytic model is neither PS water nor q-TIP4P/F. No exact quantum or liquid-water replication."}}
    atomic_write_text(options.output / "report.json", json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps(_json_safe({"frequency_differences": differences,
                                "physical_converged": False,
                                "total_wall_seconds": report["total_wall_seconds"]}), indent=2), flush=True)
    return report


if __name__ == "__main__":
    run(parser().parse_args())
