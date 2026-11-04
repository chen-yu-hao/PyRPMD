"""Compare BA-RP-MASH adiabatic populations with a 1D Fourier wavepacket.

The model has constant adiabatic energies and rotating eigenvectors, so its
nonzero NAC is independently known. BA electronic evolution depends on the
centroid velocity; free internal ring modes are spectators in this model.
This isolates population preparation, gap/NAC bead averaging and full-ring
hops. It is not a benchmark of general nuclear quantum effects or a claim
that BA-RP-MASH reproduces arbitrary coupled quantum dynamics.

The finite Gaussian packet starts on the local lower adiabatic surface.
Trajectory centroids sample its positive scalar nuclear Wigner function;
spins use the MASH population importance distribution. For P>1 the explicit
zero-centroid internal displacement is a diagnostic ring, not a coupled
thermal ensemble. The high kinetic energy makes the distinction between
the scalar Wigner preparation and the exact position-dependent electronic
wavepacket small, but it is retained as a limitation of the comparison.

All physical inputs use atomic units. Example from the repository root::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
        examples/nonadiabatic/rp_mash_ba_reference.py \
        --trajectories 128 --output ba-rp-mash-reference.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from pyrpmd.analysis.quantum_reference import SplitOperator1D
from pyrpmd.dynamics import State
from pyrpmd.nonadiabatic import DiabaticResult
from pyrpmd.rp_mash_ba import RPMASHBADynamics


class RotatingSurfaces:
    """V(x)=g[cos(kx) sigma_z + sin(kx) sigma_x], d_+-=-k/2."""

    def __init__(self, half_gap=.02, wave_number=1.):
        self.half_gap = float(half_gap)
        self.wave_number = float(wave_number)

    def evaluate(self, q):
        angle = self.wave_number*q[:, 0, 0]
        cosine, sine = np.cos(angle), np.sin(angle)
        matrix = np.empty((len(q), 2, 2))
        matrix[:, 0, 0], matrix[:, 1, 1] = self.half_gap*cosine, -self.half_gap*cosine
        matrix[:, 0, 1] = matrix[:, 1, 0] = self.half_gap*sine
        gradients = np.zeros((len(q), 2, 2, 1, 3))
        gradients[:, 0, 0, 0, 0] = -self.half_gap*self.wave_number*sine
        gradients[:, 1, 1, 0, 0] = self.half_gap*self.wave_number*sine
        gradients[:, 0, 1, 0, 0] = gradients[:, 1, 0, 0, 0] = self.half_gap*self.wave_number*cosine
        return DiabaticResult(matrix, gradients)


def quantum_packet(model, *, points, dt, duration, mass=1000., momentum=100.,
                   width=.2, center=-1., half_box=6.):
    grid = np.linspace(-half_box, half_box, points, endpoint=False)
    dx = grid[1]-grid[0]
    q = np.zeros((points, 1, 3))
    q[:, 0, 0] = grid
    angle = model.wave_number*grid/2
    upper = np.column_stack((np.cos(angle), np.sin(angle)))
    lower = np.column_stack((-np.sin(angle), np.cos(angle)))
    psi = (np.exp(-(grid-center)**2/(4*width**2)+1j*momentum*(grid-center))[:, None]
           * lower)
    psi /= np.sqrt(dx*np.sum(abs(psi)**2))
    steps = max(1, int(np.ceil(duration/dt)))
    actual_dt = duration/steps
    reference = SplitOperator1D(grid, model.evaluate(q).matrix, mass, actual_dt)
    result = reference.propagate(psi, steps, sample_stride=steps)
    final = result.final_wavefunctions[0]
    upper_population = float(dx*np.sum(abs(np.einsum("xi,xi->x", upper, final))**2))
    return {
        "populations": [1.-upper_population, upper_population],
        "points": points, "dt_au": actual_dt, "steps": steps,
        "max_edge_population": float(np.max(result.edge_population)),
        "max_norm_error": float(np.max(abs(result.norm-1.))),
    }


def exact_rotating_surface_population(model, duration, *, mass=1000., momentum=100.,
                                      width=.2, quadrature_order=64):
    """Infinite-line solution after the exact position-dependent unitary.

    U(x) transforms H to (p-k*sigma_y/2)^2/(2m)+g*sigma_z. Each
    momentum therefore has an independent, known two-level propagator.
    The Gaussian momentum integral is evaluated by Gauss-Hermite quadrature.
    """
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
    momenta = momentum + np.sqrt(2)*nodes/(2*width)
    transverse = model.wave_number*momenta/(2*mass)
    magnitude = np.hypot(model.half_gap, transverse)
    probability = (transverse/magnitude)**2*np.sin(magnitude*duration)**2
    return float(np.dot(weights, probability)/np.sqrt(np.pi))


def comparison(*, trajectories=128, beads=4, seed=20261103, half_gap=.02,
               points=1024, reference_dt=.1, duration=None, output_steps=4,
               rtol=1e-9, atol=1e-11):
    if trajectories < 2 or beads < 1 or points < 64 or output_steps < 1:
        raise ValueError("require trajectories>=2, beads>=1, points>=64 and output_steps>=1")
    if not np.isfinite(half_gap) or half_gap <= 5e-11:
        raise ValueError("half_gap must exceed the driver's default degeneracy threshold / 2")
    if not np.isfinite(reference_dt) or reference_dt <= 0:
        raise ValueError("reference_dt must be positive and finite")
    mass, momentum, width, center, beta = 1000., 100., .2, -1., 100.
    duration = np.pi/(2*momentum/mass) if duration is None else float(duration)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite")
    dt = duration/output_steps
    model = RotatingSurfaces(half_gap)
    quantum_options = dict(duration=duration, mass=mass, momentum=momentum,
                           width=width, center=center)
    started = time.perf_counter()
    reference = quantum_packet(model, points=points, dt=reference_dt, **quantum_options)
    temporal = quantum_packet(model, points=points, dt=reference_dt/2, **quantum_options)
    spatial = quantum_packet(model, points=2*points, dt=reference_dt/2, **quantum_options)
    analytic = exact_rotating_surface_population(model, duration, mass=mass,
                                                 momentum=momentum, width=width)
    analytic_refined = exact_rotating_surface_population(
        model, duration, mass=mass, momentum=momentum, width=width, quadrature_order=96)
    quantum_seconds = time.perf_counter()-started

    # A zero-sum internal ring deformation distinguishes P>1 from an array
    # of coincident beads. It does not affect this constant-gap/NAC model's
    # exact mean velocity, and is not presented as thermal path sampling.
    displacement = .1*np.cos(2*np.pi*np.arange(beads)/beads)
    displacement -= displacement.mean()
    populations, energy_changes, norm_errors = [], [], []
    accepted, frustrated = 0, 0
    children = np.random.SeedSequence(seed).spawn(trajectories)
    started = time.perf_counter()
    for child in children:
        nuclear_seed, electronic_seed = child.spawn(2)
        generator = np.random.default_rng(nuclear_seed)
        q, p = np.zeros((beads, 1, 3)), np.zeros((beads, 1, 3))
        q[:, 0, 0] = generator.normal(center, width) + displacement
        p[:, 0, 0] = generator.normal(momentum, 1/(2*width))
        driver = RPMASHBADynamics(model, [mass], beta, beads, dt,
                                 rtol=rtol, atol=atol, max_step=dt/4)
        state = driver.initialize(State(q, p), active_surface=0,
                                  rng=np.random.default_rng(electronic_seed))
        initial_energy = driver.energy(state)
        for _ in range(output_steps):
            driver.step(state)
            accepted += sum(event.accepted for event in driver.last_hops)
            frustrated += sum(event.frustrated for event in driver.last_hops)
        populations.append(driver.populations(state))
        energy_changes.append(driver.energy(state)-initial_energy)
        norm_errors.append(abs(driver.electronic_norm(state)-1.))
    elapsed = time.perf_counter()-started
    samples = np.asarray(populations)
    estimate = samples.mean(axis=0)
    sem = samples.std(axis=0, ddof=1)/np.sqrt(trajectories)
    exact = np.array(spatial["populations"])
    return {
        "method": "rp-mash-ba", "model": "constant_gap_rotating_eigenvectors",
        "model_formula": "V(x)=g*[cos(x)*sigma_z+sin(x)*sigma_x]; d_+-=-1/2",
        "half_gap_hartree": half_gap, "beads": beads, "beta_inverse_hartree": beta,
        "trajectories": trajectories, "seed": seed,
        "mass_electron": mass, "initial_center_bohr": center,
        "initial_position_std_bohr": width, "initial_mean_momentum_au": momentum,
        "initial_momentum_std_au": 1/(2*width),
        "internal_displacement_bohr": displacement.tolist(),
        "duration_au": duration, "output_dt_au": dt, "output_steps": output_steps,
        "max_internal_step_au": dt/4, "rtol": rtol, "atol": atol,
        "population_basis": "common_bead_adiabatic",
        "population_order": ["lower", "upper"],
        "population_estimator": "active_surface_indicator",
        "mash_initial_populations": [1., 0.],
        "mash_final_populations": estimate.tolist(), "mash_population_sem": sem.tolist(),
        "fourier": spatial,
        "analytic_infinite_line_populations": [1.-analytic_refined, analytic_refined],
        "analytic_momentum_quadrature_refinement_diff": abs(analytic-analytic_refined),
        "fourier_analytic_population_max_diff": abs(spatial["populations"][1]-analytic_refined),
        "absolute_population_error": abs(estimate-exact).tolist(),
        "fourier_time_refinement_max_diff": float(np.max(abs(
            np.array(reference["populations"])-np.array(temporal["populations"])))),
        "fourier_space_refinement_max_diff": float(np.max(abs(
            np.array(temporal["populations"])-exact))),
        "mash_max_abs_energy_change_hartree": float(np.max(abs(np.array(energy_changes)))),
        "mash_max_spin_norm_error": float(np.max(norm_errors)),
        "accepted_events": int(accepted), "frustrated_events": int(frustrated),
        "mash_elapsed_seconds": elapsed,
        "mash_microseconds_per_trajectory_output_step": elapsed*1e6/(trajectories*output_steps),
        "fourier_elapsed_seconds": quantum_seconds,
        "execution": {"platform": platform.platform(), "machine": platform.machine(),
                      "python": platform.python_version(), "numpy": np.__version__,
                      "scipy": scipy.__version__,
                      "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                      "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS")},
        "physical_converged": False,
        "limitations": [
            "finite packet, finite trajectory ensemble and approximate trajectory method",
            "SEM includes independent-trajectory sampling only, not method or discretization bias",
            "constant gap/NAC isolates ring averaging; it does not establish general nuclear quantum effects",
            "scalar centroid Wigner sampling omits position-dependent electronic Wigner corrections",
            "specified internal ring deformation is not a coupled thermal equilibrium distribution",
            "reference time/grid refinements are diagnostics, not proof of complete physical convergence",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=int, default=128)
    parser.add_argument("--beads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--half-gap", type=float, default=.02)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument("--reference-dt", type=float, default=.1)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--output-steps", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = comparison(trajectories=args.trajectories, beads=args.beads, seed=args.seed,
                        half_gap=args.half_gap, points=args.points,
                        reference_dt=args.reference_dt, duration=args.duration,
                        output_steps=args.output_steps)
    payload = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output:
        from pyrpmd.io import atomic_write_text
        atomic_write_text(args.output, payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
