"""Exercise the shipped periodic PES and a real ASE slab calculator."""

from pathlib import Path

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics, State
from pyrpmd.input import JobSpec
from pyrpmd.potentials import make_potential, check_potential


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "surface"


@pytest.mark.parametrize("symbols, expected", [("Cu", ["Cu"]), ("Cu2", ["Cu", "Cu"]), (["D", "T"], ["H", "H"])])
def test_ase_constructor_preserves_formula_string_api(symbols, expected):
    pytest.importorskip("ase")
    from ase.calculators.emt import EMT
    from pyrpmd.potentials.optional import ASEPotential

    potential = ASEPotential(symbols, EMT())
    assert potential.atoms.get_chemical_symbols() == expected
    potential.close()


def test_teaching_periodic_surface_forces_and_lattice_invariance():
    config, system = JobSpec.from_file(EXAMPLES / "diffusion.inp").resolve()
    potential = make_potential(config["potential"], system)
    q = np.repeat(system.geometry[None], 2, axis=0)
    q[:, 0] += [[1.1, 0.4, 0.13], [-0.8, -0.3, -0.2]]
    try:
        assert check_potential(potential, q)["max_absolute_force_error"] < 1e-10
        result = potential.evaluate(q)
        shifted = q.copy()
        shifted[:, 0] += 2 * system.cell[0] - 3 * system.cell[1]
        other = potential.evaluate(shifted)
        np.testing.assert_allclose(other.energy, result.energy, atol=1e-15)
        np.testing.assert_allclose(other.forces, result.forces, atol=1e-15)
    finally:
        potential.close()


def test_real_ase_emt_slab_runs_with_mobile_upper_layers_and_fixed_bottom():
    pytest.importorskip("ase")
    config, system = JobSpec.from_file(EXAMPLES / "ase_emt.py").resolve()
    potential = make_potential(config["potential"], system)
    try:
        assert system.pbc == (True, True, False)
        assert system.cell[1, 0] != 0  # Actual skew Cu(111) cell.
        assert len(system.fixed_atoms) == 4 < system.natoms
        q = np.repeat(system.geometry[None], 4, axis=0)
        dyn = Dynamics(potential, system.masses, config["beta"], 4, config["dt_au"],
                       seed=31, fixed_atoms=system.fixed_atoms, fixed_positions=system.geometry)
        state = State(q, dyn.sample_momenta())
        initial = dyn.energy(state)
        for _ in range(4):
            dyn.step(state)
        np.testing.assert_array_equal(state.q[:, system.fixed_atoms], q[:, system.fixed_atoms])
        np.testing.assert_array_equal(state.p[:, system.fixed_atoms], 0)
        assert np.linalg.norm(state.q[:, system.mobile_mask] - q[:, system.mobile_mask]) > 0
        assert abs(dyn.energy(state) - initial) / 4 * config["beta"] < 0.01
    finally:
        potential.close()
