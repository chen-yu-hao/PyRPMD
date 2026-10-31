"""Full-dimensional curved-parent/NVE diagnostic at the published SP1 geometry.

Run after fetch_model.py. This short check tests numerical invariants, not a
transmission plateau. No reference rate or guessed kappa enters propagation.
"""
import json
import numpy as np
from common import ROOT, metadynamics
from pyrpmd.constants import ANGSTROM_TO_AU, EV_TO_AU, FS_TO_AU
from pyrpmd.cv import PeriodicDistance
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.input import JobSpec
from pyrpmd.potentials import make_potential
from pyrpmd.workflows.surface import DistanceSurfaceCoordinate
from pyrpmd.workflows.rate import project_positions, project_momenta, constrained_step


def main():
    config, system = JobSpec.from_mapping(metadynamics()).resolve()
    provider = make_potential(config["potential"], system)
    records = []
    try:
        for beads, timestep_fs in ((1, .2), (4, .2), (4, .1)):
            steps = round(20 / timestep_fs)
            dyn = Dynamics(provider, system.masses, config["beta"], beads, timestep_fs * FS_TO_AU,
                           seed=7331, integrator="velocity-verlet", fixed_atoms=system.fixed_atoms,
                           fixed_positions=system.geometry)
            coordinate = DistanceSurfaceCoordinate(PeriodicDistance([0, 2], system.cell, system.pbc), system, "lower")
            surface = coordinate.dividing_surface(1.98 * ANGSTROM_TO_AU)
            initial = np.array(json.loads((ROOT / "transition_state.json").read_text())) * ANGSTROM_TO_AU
            initial[list(system.fixed_atoms)] = system.geometry[list(system.fixed_atoms)]
            q = project_positions(np.repeat(initial[None], beads, axis=0), surface, system.masses)
            state = State(q, dyn.sample_momenta())
            project_momenta(state.p, state.q, surface, system.masses)
            residual, tangent, frozen = 0., 0., 0.
            for _ in range(steps):
                constrained_step(dyn, state, surface)
                value, g = surface.centroid_value_gradient(state.q.mean(axis=0))
                residual = max(residual, abs(value))
                tangent = max(tangent, abs(np.sum(g * state.p.mean(axis=0) / system.masses[:, None])))
                frozen = max(frozen, float(np.max(np.abs(state.q[:, system.fixed_atoms] - system.geometry[list(system.fixed_atoms)]))))
            p = dyn.sample_momenta()
            drifts = []
            for sign in (-1., 1.):
                child = State(state.q, sign * p)
                e0, error = dyn.energy(child), 0.
                for _ in range(steps):
                    dyn.step(child)
                    error = max(error, abs(dyn.energy(child) - e0) / beads / EV_TO_AU)
                drifts.append(error)
                np.testing.assert_array_equal(child.p[:, system.fixed_atoms], 0.)
                np.testing.assert_array_equal(child.q[:, system.fixed_atoms], np.broadcast_to(system.geometry[list(system.fixed_atoms)], child.q[:, system.fixed_atoms].shape))
            assert residual < 1e-9 and tangent < 1e-12 and frozen == 0.
            records.append({"beads": beads, "natoms": system.natoms, "mobile_cartesian_dof": 63,
                            "ring_dof": 63 * beads, "timestep_fs": timestep_fs,
                            "parent_steps": steps, "child_steps_per_sign": steps,
                            "constraint_max_bohr": residual, "tangent_velocity_max_au": tangent,
                            "fixed_position_error_bohr": frozen, "child_energy_drift_eV_per_bead": drifts,
                            "max_drift_kBT": max(drifts) * EV_TO_AU * config["beta"]})
    finally:
        provider.close()
    print(json.dumps({"checks": records, "is_rate_validation": False}, indent=2))


if __name__ == "__main__":
    main()
