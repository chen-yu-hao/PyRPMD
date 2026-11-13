"""Finite-bath, two-electron RP-IESH / Fourier quantum comparison.

The quantum calculation lifts a one-electron Newns--Anderson Hamiltonian to a
fixed-particle-number Fock basis using explicit fermion creation/annihilation
signs. It then propagates the nuclei with the independent periodic Fourier
solver. The reported observable is the one-particle density, whose trace is
the electron count, not a vector of Fock-state probabilities.

Both calculations start from a harmonic nuclear canonical distribution and
the same fixed-diabatic electronic determinant. RP-IESH approximates that
nuclear distribution with a finite ring; its active adiabatic determinant is
sampled from the exact Slater-minor probabilities at the centroid. This
electronic preparation is a Python-API example, not an extra TOML option.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \\
        examples/nonadiabatic/rp_iesh_reference.py --trajectories 128

All quantities use atomic units. Grid, box and time refinements quantify
reference sensitivity; trajectory SEM does not measure approximation error.
This small isolated electronic bath is not a continuum metallic reservoir.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from math import comb
from pathlib import Path
import platform
import time

import numpy as np

from pyrpmd.analysis.quantum_reference import SplitOperator1D, harmonic_canonical_ensemble
from pyrpmd.iesh import NewnsAndersonModel
from pyrpmd.nonadiabatic_sampling import sample_harmonic_ring


def _integer(value, name, minimum=1):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(value, name):
    raw = np.asarray(value)
    if (raw.shape != () or raw.dtype.kind not in "fiu"
            or not np.isfinite(raw) or raw <= 0):
        raise ValueError(f"{name} must be positive and finite")
    return float(raw)


def _steps(duration, dt):
    duration, dt = _positive(duration, "duration"), _positive(dt, "dt")
    steps = round(duration/dt)
    if steps < 1 or not np.isclose(steps*dt, duration, atol=1e-12, rtol=0.):
        raise ValueError("duration must be a positive integral number of steps")
    return steps


class FockSpace:
    """Small exact reference basis; no trajectory or fast-determinant algebra.

    Basis vectors are ascending tuples of occupied orbitals, with creation
    operators in ascending order. ``operators[i,j]`` is a_i^dagger a_j.
    The explicit dimension cap prevents accidental many-body allocations.
    """

    def __init__(self, norbitals, nelectrons):
        self.norbitals = _integer(norbitals, "norbitals")
        self.nelectrons = _integer(nelectrons, "nelectrons")
        if self.norbitals > 16 or self.nelectrons > self.norbitals:
            raise ValueError("reference requires 1 <= nelectrons <= norbitals <= 16")
        self.dimension = comb(self.norbitals, self.nelectrons)
        if self.dimension > 256:
            raise ValueError("reference Fock dimension must not exceed 256")
        self.basis = tuple(combinations(range(self.norbitals), self.nelectrons))
        masks = [sum(1 << index for index in occupied) for occupied in self.basis]
        lookup = {mask: index for index, mask in enumerate(masks)}
        operators = np.zeros((self.norbitals, self.norbitals,
                              self.dimension, self.dimension))
        for column, mask in enumerate(masks):
            for annihilate in range(self.norbitals):
                if not mask & (1 << annihilate):
                    continue
                sign = (-1)**(mask & ((1 << annihilate)-1)).bit_count()
                removed = mask ^ (1 << annihilate)
                for create in range(self.norbitals):
                    if removed & (1 << create):
                        continue
                    parity = (removed & ((1 << create)-1)).bit_count()
                    row = lookup[removed | (1 << create)]
                    operators[create, annihilate, row, column] = sign*(-1)**parity
        self.operators = operators

    def lift(self, one_electron):
        """Lift h[... ,M,M] to sum_ij h_ij a_i^dagger a_j."""
        matrix = np.asarray(one_electron)
        if (matrix.ndim < 2 or matrix.shape[-2:] != (self.norbitals, self.norbitals)
                or matrix.dtype.kind not in "fciu" or not np.isfinite(matrix).all()
                or not np.allclose(matrix, matrix.conj().swapaxes(-1, -2), atol=1e-12, rtol=0.)):
            raise ValueError("one-electron matrix must be finite Hermitian [...,M,M]")
        return np.einsum("...ij,ijab->...ab", matrix, self.operators, optimize=True)

    def slater_coefficients(self, orbitals):
        """Expand orthonormal fixed-basis orbitals into signed Fock minors."""
        coefficients = np.asarray(orbitals)
        if (coefficients.shape != (self.norbitals, self.nelectrons)
                or coefficients.dtype.kind not in "fciu" or not np.isfinite(coefficients).all()
                or not np.allclose(coefficients.conj().T @ coefficients,
                                   np.eye(self.nelectrons), atol=1e-10, rtol=0.)):
            raise ValueError("orbitals must be finite orthonormal [norbitals,nelectrons]")
        return np.asarray([np.linalg.det(coefficients[list(occupied)]) for occupied in self.basis])

    def one_particle_density(self, fock_density):
        """Return rho_ij=<a_j^dagger a_i>, with the C C^dagger convention."""
        density = np.asarray(fock_density)
        if (density.ndim < 2 or density.shape[-2:] != (self.dimension, self.dimension)
                or density.dtype.kind not in "fciu" or not np.isfinite(density).all()):
            raise ValueError("Fock density must be finite [...,dimension,dimension]")
        return np.einsum("...ab,ijba->...ji", density, self.operators, optimize=True)


def quantum_reference(model, space, *, points=128, dt=.05, duration=20.,
                      mass=1000., beta=80., omega=.02, half_box=2., sample_stride=None):
    """Independent finite-grid canonical-nucleus / fixed-diabatic calculation.

    The first ``nelectrons`` fixed diabatic orbitals are occupied initially.
    Nuclear U0 is added once to the Fock Hamiltonian. The returned tuple is the
    Fourier result, its one-particle density series, and the truncated thermal
    mixture weight (which does not include grid or box error).
    """
    points = _integer(points, "points", 4)
    half_box = _positive(half_box, "half_box")
    steps = _steps(duration, dt)
    # Retain every reference step by default so edge/norm maxima include the
    # intervening propagation, not only the initial and final observations.
    stride = 1 if sample_stride is None else _integer(sample_stride, "sample_stride")
    grid = np.linspace(-half_box, half_box, points, endpoint=False)
    q = np.zeros((points, 1, 3))
    q[:, 0, 0] = grid
    matrix = space.lift(model.evaluate(q).matrix)
    matrix += model.bare_potential(q).energy[:, None, None]*np.eye(space.dimension)
    thermal = harmonic_canonical_ensemble(grid, mass, omega, beta, tail_tolerance=1e-12)
    initial = np.zeros((len(thermal.weights), points, space.dimension), dtype=complex)
    initial[:, :, 0] = thermal.wavefunctions
    reference = SplitOperator1D(grid, matrix, mass, dt).propagate(
        initial, steps, weights=thermal.weights, sample_stride=stride)
    return reference, space.one_particle_density(reference.density), thermal.omitted_weight


def initialize_diabatic_determinant(driver, nuclear, space, electronic_rng):
    """Sample the active determinant consistently with the fixed diabatic C0."""
    centroid = nuclear.q.mean(axis=0, keepdims=True)
    _, vectors = np.linalg.eigh(driver.potential.evaluate(centroid).matrix[0])
    initial = np.eye(space.norbitals, dtype=complex)[:, :space.nelectrons]
    adiabatic = vectors.conj().T @ initial
    probabilities = np.abs(space.slater_coefficients(adiabatic))**2
    if not np.isclose(probabilities.sum(), 1., atol=1e-12, rtol=0.):
        raise ValueError("initial determinant probabilities do not sum to one")
    selected = electronic_rng.choice(space.dimension, p=probabilities)
    return driver.initialize(nuclear, occupied=space.basis[selected],
                             adiabatic_coefficients=adiabatic)


def compare(*, trajectories=128, beads=8, seed=20261103, dt=.2, duration=20.,
            points=128, reference_dt=.05, electronic_substeps=2):
    """Report finite-bath agreement and numerical diagnostics without a pass flag."""
    from pyrpmd.rp_iesh import RPIESHDynamics

    trajectories = _integer(trajectories, "trajectories", 2)
    beads = _integer(beads, "beads")
    seed = _integer(seed, "seed", 0)
    steps = _steps(duration, dt)
    mass, beta, omega, half_box = 1000., 80., .02, 2.
    model = NewnsAndersonModel(
        nband=2, band_min=-.1, band_max=.1, level_slope=.08,
        hybridization=.04, bare_force_constant=mass*omega**2)
    space = FockSpace(model.nstates, 2)
    reference_options = dict(duration=duration, mass=mass, beta=beta, omega=omega)
    base, base_density, _ = quantum_reference(
        model, space, points=points, dt=reference_dt, half_box=half_box, **reference_options)
    temporal, temporal_density, _ = quantum_reference(
        model, space, points=points, dt=reference_dt/2, half_box=half_box, **reference_options)
    spatial, spatial_density, _ = quantum_reference(
        model, space, points=2*points, dt=reference_dt/2, half_box=half_box, **reference_options)
    reference, density, omitted = quantum_reference(
        model, space, points=3*points, dt=reference_dt/2, half_box=1.5*half_box, **reference_options)

    children = np.random.SeedSequence(seed).spawn(trajectories)
    populations, initial_populations, final_densities = [], [], []
    energy_changes, hop_energy_changes, hop_energy_changes_abs = [], [], []
    accepted, frustrated = 0, 0
    max_probability, max_centroid_balance, max_ring_jump = 0., 0., 0.
    started = time.perf_counter()
    for child in children:
        nuclear_seed, electronic_seed, dynamics_seed = child.spawn(3)
        nuclear = sample_harmonic_ring(
            [mass], beta, beads, [omega], coordinates=[0], count=1,
            seed=np.random.default_rng(nuclear_seed))[0]
        driver = RPIESHDynamics(
            model, [mass], beta, beads, dt, nelectrons=space.nelectrons,
            seed=np.random.default_rng(dynamics_seed), native=False,
            electronic_substeps=electronic_substeps)
        state = initialize_diabatic_determinant(
            driver, nuclear, space, np.random.default_rng(electronic_seed))
        initial_populations.append(np.diag(driver.density_matrix(state)).real)
        initial_energy = driver.energy(state)
        jump, absolute_jump = 0., 0.
        for _ in range(steps):
            driver.step(state)
            max_probability = max(max_probability, float(driver.last_probabilities.sum()))
            if driver.last_hop is not None:
                hop = driver.last_hop
                accepted += hop.accepted
                frustrated += hop.frustrated
                max_centroid_balance = max(max_centroid_balance, abs(hop.centroid_energy_error))
                max_ring_jump = max(max_ring_jump, abs(hop.ring_energy_change))
                jump += hop.ring_energy_change
                absolute_jump += abs(hop.ring_energy_change)
        final_density = driver.density_matrix(state)
        final_densities.append(final_density)
        populations.append(final_density.diagonal().real)
        energy_changes.append(driver.energy(state)-initial_energy)
        hop_energy_changes.append(jump)
        hop_energy_changes_abs.append(absolute_jump)
    elapsed = time.perf_counter()-started
    samples = np.asarray(populations)
    mean, sem = samples.mean(axis=0), samples.std(axis=0, ddof=1)/np.sqrt(trajectories)
    exact = density[-1].diagonal().real
    delta = mean-exact
    energy_changes, hop_energy_changes = np.asarray(energy_changes), np.asarray(hop_energy_changes)
    mean_density = np.mean(final_densities, axis=0)
    return {
        "method": "rp-iesh", "method_variant": "gardner-2023-centroid",
        "model_identity": model.to_dict(), "nelectrons": space.nelectrons,
        "fock_basis": [list(occupied) for occupied in space.basis],
        "trajectories": trajectories, "beads": beads, "seed": seed,
        "dt_au": dt, "duration_au": duration, "electronic_substeps": electronic_substeps,
        "mass_electron": mass, "beta_hartree_inverse": beta, "harmonic_omega_au": omega,
        "initial_electronic_basis": "fixed_diabatic",
        "initial_occupied": list(space.basis[0]),
        "initial_active_sampling": "squared_adiabatic_Slater_minors_at_centroid",
        "population_basis": "fixed_diabatic", "population_normalization": space.nelectrons,
        "rp_iesh_initial_mean": np.mean(initial_populations, axis=0).tolist(),
        "rp_iesh_final_mean": mean.tolist(), "rp_iesh_final_sem": sem.tolist(),
        "fourier_final_populations": exact.tolist(),
        "population_difference": delta.tolist(), "absolute_population_error": abs(delta).tolist(),
        "difference_in_sampling_sem": [float(d/s) if s > 0 else None for d, s in zip(delta, sem)],
        "rp_iesh_final_density_real": mean_density.real.tolist(),
        "rp_iesh_final_density_imag": mean_density.imag.tolist(),
        "fourier_final_density_real": density[-1].real.tolist(),
        "fourier_final_density_imag": density[-1].imag.tolist(),
        "fourier_points": 3*points, "fourier_half_box_bohr": 1.5*half_box,
        "fourier_dt_au": reference_dt/2,
        "fourier_time_refinement_max_density_diff": float(np.max(abs(base_density[-1]-temporal_density[-1]))),
        "fourier_space_refinement_max_density_diff": float(np.max(abs(temporal_density[-1]-spatial_density[-1]))),
        "fourier_box_refinement_max_density_diff": float(np.max(abs(spatial_density[-1]-density[-1]))),
        "fourier_max_edge_population": float(np.max(reference.edge_population)),
        "fourier_max_norm_error": float(np.max(abs(reference.norm-1.))),
        "fourier_omitted_thermal_weight": float(omitted),
        "rp_iesh_accepted_hops": int(accepted), "rp_iesh_frustrated_hops": int(frustrated),
        "rp_iesh_max_step_hop_probability": max_probability,
        "rp_iesh_max_abs_centroid_hop_balance_error_hartree": max_centroid_balance,
        "rp_iesh_max_abs_ring_hop_energy_change_hartree": max_ring_jump,
        "rp_iesh_max_abs_total_energy_change_hartree": float(np.max(abs(energy_changes))),
        "rp_iesh_max_abs_accumulated_ring_hop_energy_change_hartree": float(np.max(abs(hop_energy_changes))),
        "rp_iesh_max_accumulated_absolute_ring_hop_energy_change_hartree": float(np.max(hop_energy_changes_abs)),
        "rp_iesh_max_abs_energy_change_after_subtracting_hop_changes_hartree":
            float(np.max(abs(energy_changes-hop_energy_changes))),
        "rp_iesh_elapsed_seconds": elapsed,
        "rp_iesh_microseconds_per_step": elapsed*1e6/(trajectories*steps),
        "platform": platform.platform(), "processor": platform.processor(),
        
        "limitations": [
            "finite isolated two-electron, three-orbital bath; no metallic continuum limit",
            "finite-bead harmonic nuclear preparation; initially factorized diabatic determinant",
            "centroid RPIESH is approximate and need not agree with exact coupled quantum dynamics",
            "full ring Hamiltonian changes at hops; centroid balance and ring jumps are distinct diagnostics",
            "pointwise SEM measures independent-trajectory sampling only, not method bias",
            "RP-IESH timestep, bead and ensemble convergence are not established by this comparison",
            "Fourier refinements probe only this finite grid, box, time and thermal truncation",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=int, default=128)
    parser.add_argument("--beads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--dt", type=float, default=.2)
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--points", type=int, default=128)
    parser.add_argument("--reference-dt", type=float, default=.05)
    parser.add_argument("--electronic-substeps", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(trajectories=args.trajectories, beads=args.beads, seed=args.seed,
                     dt=args.dt, duration=args.duration, points=args.points,
                     reference_dt=args.reference_dt, electronic_substeps=args.electronic_substeps)
    payload = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output:
        from pyrpmd.io import atomic_write_text
        atomic_write_text(args.output, payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
