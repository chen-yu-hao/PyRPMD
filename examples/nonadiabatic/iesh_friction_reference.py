"""Cold-bath, slow-nuclear IESH / P=1 EF-RPMD comparison in atomic units.

The reference uses the finite-band electronic ground-state PES associated with
the IESH Hamiltonian, with analytic spinless, constant-Gamma, zero-temperature wideband friction
of Gardner et al., JCP 158, 064101 (2023), Eqs. 33--34. The OU bath has a large
finite beta, so its very small thermal noise remains explicitly present.
IESH follows its current occupied determinant after a hop; the common ground
surface defines the mechanical-energy baseline, not every IESH force.

This demonstrates qualitative energy loss, not continuum or physical
convergence. In particular, Gamma -> 0 at fixed velocity is not a valid
electronic-friction limit. A finite isolated electronic band eventually recurs.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
        examples/nonadiabatic/iesh_friction_reference.py --trajectories 128
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from pyrpmd.dynamics import State
from pyrpmd.electronic_friction import ElectronicFrictionDynamics, FrictionResult
from pyrpmd.iesh import IESHDynamics, NewnsAndersonModel
from pyrpmd.potentials.base import PESResult, Potential


class GroundStatePES(Potential):
    """Independent scalar PES for the filled finite-band ground determinant."""

    def __init__(self, model, nelectrons):
        self.model, self.nelectrons = model, nelectrons

    def evaluate(self, q):
        result = self.model.evaluate(q)
        energies, vectors = np.linalg.eigh(result.matrix)
        filled = vectors[:, :, :self.nelectrons]
        gradient = np.einsum("bsi,bstac,bti->bac", filled, result.gradients, filled,
                             optimize=True)
        bare = self.model.bare_potential(q)
        return PESResult(bare.energy+energies[:, :self.nelectrons].sum(axis=1),
                         bare.forces-gradient)


class WideBandFriction:
    """Physical eta, not an inverse-time friction rate; one spin channel."""

    def __init__(self, model):
        if model.hybridization <= 0. or model.hybridization_slope != 0.:
            raise ValueError("this reference requires positive, coordinate-independent Gamma")
        self.model = model

    def evaluate(self, q):
        model = self.model
        x = q.reshape(len(q), -1)[:, model.coordinate]-model.center
        level = model.level_energy+model.level_slope*x
        half_gamma = model.hybridization/2
        spectral = half_gamma/(np.pi*((model.chemical_potential-level)**2+half_gamma**2))
        eta = np.pi*model.level_slope**2*spectral**2
        tensor = np.zeros((len(q), 3*model.natoms, 3*model.natoms))
        tensor[:, model.coordinate, model.coordinate] = eta
        return FrictionResult(tensor)


def compare(*, trajectories=128, nband=63, dt=1., duration=200., seed=20261102,
            electronic_substeps=2, hybridization=.05):
    if trajectories < 2:
        raise ValueError("at least two independent trajectories are required for SEM")
    steps = round(duration/dt)
    if steps < 1 or not np.isclose(steps*dt, duration, atol=1e-12, rtol=0.):
        raise ValueError("duration must be a positive integral number of steps")
    model = NewnsAndersonModel(
        nband=nband, band_min=-.25, band_max=.25, chemical_potential=0.,
        discretization="midpoint", level_energy=0., level_slope=.02,
        hybridization=hybridization, bare_force_constant=0.)
    nelectrons = model.nstates//2
    ground, friction = GroundStatePES(model, nelectrons), WideBandFriction(model)
    mass, beta, initial_position, initial_momentum = 2000., 1e8, -1., 20.
    q, p = np.zeros((1, 1, 3)), np.zeros((1, 1, 3))
    q[0, 0, 0], p[0, 0, 0] = initial_position, initial_momentum
    initial_mechanical = initial_momentum**2/(2*mass)+ground.evaluate(q).energy[0]
    iesh_loss, friction_loss, total_energy_drift, final_excitation = [], [], [], []
    max_probability, accepted, frustrated = 0., 0, 0
    children = np.random.SeedSequence(seed).spawn(2*trajectories)
    iesh_seconds, friction_seconds = 0., 0.
    for index in range(trajectories):
        driver = IESHDynamics(model, [mass], beta, 1, dt, nelectrons=nelectrons,
                              seed=np.random.default_rng(children[2*index]), native=False,
                              electronic_substeps=electronic_substeps,
                              max_hop_probability=.2)
        state = driver.initialize(State(q, p))
        initial_total = driver.energy(state)
        started = time.perf_counter()
        for _ in range(steps):
            driver.step(state)
            max_probability = max(max_probability, float(driver.last_probabilities.sum()))
            if driver.last_hop is not None:
                accepted += driver.last_hop.accepted
                frustrated += driver.last_hop.frustrated
        iesh_seconds += time.perf_counter()-started
        final_total = driver.energy(state)
        mechanical = float(np.sum(state.p**2/(2*mass))+ground.evaluate(state.q).energy[0])
        iesh_loss.append(initial_mechanical-mechanical)
        total_energy_drift.append(final_total-initial_total)
        final_excitation.append(final_total-mechanical)
        reference = ElectronicFrictionDynamics(
            ground, [mass], beta, 1, dt, friction,
            seed=np.random.default_rng(children[2*index+1]), native=False)
        reference_state = State(q, p)
        started = time.perf_counter()
        for _ in range(steps):
            reference.step(reference_state)
        friction_seconds += time.perf_counter()-started
        mechanical = float(np.sum(reference_state.p**2/(2*mass))
                           + ground.evaluate(reference_state.q).energy[0])
        friction_loss.append(initial_mechanical-mechanical)

    def statistics(values):
        return {"mean_hartree": float(np.mean(values)),
                "sem_hartree": float(np.std(values, ddof=1)/np.sqrt(trajectories))}

    return {
        "model_identity": model.to_dict(), "nelectrons": nelectrons,
        "trajectories": trajectories, "seed": seed, "dt_au": dt,
        "duration_au": duration, "electronic_substeps": electronic_substeps,
        "mass_electron": mass, "beta_hartree_inverse": beta,
        "initial_position_bohr": initial_position, "initial_momentum_au": initial_momentum,
        "iesh_ground_surface_mechanical_energy_loss": statistics(iesh_loss),
        "ef_rpmd_p1_ground_surface_mechanical_energy_loss": statistics(friction_loss),
        "iesh_final_excitation_energy": statistics(final_excitation),
        "iesh_max_abs_total_energy_drift_hartree": float(np.max(np.abs(total_energy_drift))),
        "iesh_max_step_hop_probability": max_probability,
        "accepted_hops": int(accepted), "frustrated_hops": int(frustrated),
        "initial_velocity_times_level_slope_over_Gamma_squared":
            abs(initial_momentum/mass*model.level_slope/model.hybridization**2),
        "bandwidth_over_Gamma": (model.band_max-model.band_min)/model.hybridization,
        "uniform_band_recurrence_time_estimate_au":
            2*np.pi*nband/(model.band_max-model.band_min),
        "iesh_microseconds_per_step": iesh_seconds*1e6/(trajectories*steps),
        "ef_rpmd_p1_microseconds_per_step": friction_seconds*1e6/(trajectories*steps),
        
        "limitations": ["finite band, fixed electron number and finite trajectory ensemble",
                        "cold initial electronic determinant; no electronic thermostat",
                        "zero-temperature continuum friction with a small finite OU bath temperature",
                        "qualitative comparison; neither method is an exact quantum reference",
                        "P=1 only; step, band, electron count and ensemble convergence remain necessary"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=int, default=128)
    parser.add_argument("--nband", type=int, default=63)
    parser.add_argument("--dt", type=float, default=1.)
    parser.add_argument("--duration", type=float, default=200.)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--electronic-substeps", type=int, default=2)
    parser.add_argument("--hybridization", type=float, default=.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(trajectories=args.trajectories, nband=args.nband, dt=args.dt,
                     duration=args.duration, seed=args.seed,
                     electronic_substeps=args.electronic_substeps, hybridization=args.hybridization)
    payload = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output:
        from pyrpmd.io import atomic_write_text
        atomic_write_text(args.output, payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
