"""Published water equations checked independently of OpenMM's expression tree."""

import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.water.qtip4pf import (
    PARAMETERS, D_MASS_DA, cluster_energy, dipoles, initial_grid, make_system,
)
from pyrpmd.constants import ANGSTROM_TO_AU


def monomer():
    r = PARAMETERS["r_eq_angstrom"] * ANGSTROM_TO_AU
    theta = math.radians(PARAMETERS["theta_eq_degrees"])
    return np.array([[0., 0, 0], [r, 0, 0], [r * math.cos(theta), r * math.sin(theta), 0]])


def dimer():
    q = np.concatenate([monomer(), monomer() @ np.diag([-1., -1., 1.]) + [5.5, .7, .4]])
    q[1, 2] += .04
    q[5, 1] -= .03
    return q[None]


def test_monomer_energy_and_neutral_dipole():
    q = monomer()[None]
    assert cluster_energy(q)[0] == pytest.approx(0., abs=1e-28)
    expected = .5 * PARAMETERS["q_M_e"] * PARAMETERS["gamma"] * (q[0, 1] + q[0, 2])
    np.testing.assert_allclose(dipoles(q)[0], expected, rtol=1e-14)
    np.testing.assert_allclose(dipoles(q + [50, -30, 70]), dipoles(q), atol=1e-13)


def test_full_cluster_energy_is_translation_rotation_and_molecule_permutation_invariant():
    q = dimer()
    rotation = np.array([[0., 1, 0], [-1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(cluster_energy(q @ rotation + 8), cluster_energy(q), atol=2e-16)
    np.testing.assert_allclose(cluster_energy(q[:, [3, 4, 5, 0, 1, 2]]), cluster_energy(q), atol=2e-16)
    np.testing.assert_allclose(dipoles(q @ rotation), dipoles(q) @ rotation, atol=1e-14)


def test_initial_grid_is_deterministic_whole_water_with_finite_energy():
    q, box = initial_grid()
    q2, box2 = initial_grid()
    np.testing.assert_array_equal(q, q2)
    assert box == box2 and box > 6
    assert q.shape == (24, 3) and np.isfinite(cluster_energy(q[None])).all()


@pytest.mark.parametrize("kwargs", [
    {"molecules": 0}, {"molecules": True}, {"molecules": 1, "box_angstrom": 10},
    {"molecules": 1, "ewald_tolerance": 2},
    {"molecules": 1, "hydrogen_mass_da": -1},
    {"molecules": 1, "dispersion_correction": "false"},
])
def test_bad_water_configuration_rejected_before_model_construction(kwargs):
    with pytest.raises(ValueError):
        make_system(**kwargs)


def test_openmm_energy_and_virtual_site_forces_match_independent_cluster_equations():
    pytest.importorskip("openmm")
    from pyrpmd.potentials.openmm import OpenMMPotential
    q = dimer()
    with OpenMMPotential(make_system(molecules=2), ["O", "H", "H"] * 2,
                         platform="Reference") as potential:
        result = potential.evaluate(q)
        # OpenMM and SciPy use slightly different Coulomb physical constants.
        np.testing.assert_allclose(result.energy, cluster_energy(q), atol=2e-10, rtol=2e-7)
        numeric = np.empty_like(q)
        epsilon = 2e-5
        for atom in range(6):
            for axis in range(3):
                plus, minus = q.copy(), q.copy()
                plus[:, atom, axis] += epsilon
                minus[:, atom, axis] -= epsilon
                numeric[:, atom, axis] = -(cluster_energy(plus) - cluster_energy(minus)) / (2 * epsilon)
        np.testing.assert_allclose(result.forces, numeric, atol=2e-9, rtol=2e-6)
        np.testing.assert_allclose(result.forces.sum(axis=1), 0, atol=2e-15)
        assert potential.system.getNumConstraints() == 0
        assert potential.particle_indices.tolist() == [0, 1, 2, 4, 5, 6]


def test_isotope_substitution_changes_masses_only():
    pytest.importorskip("openmm")
    from openmm import unit
    from pyrpmd.potentials.openmm import OpenMMPotential
    heavy = make_system(molecules=2, hydrogen_mass_da=D_MASS_DA)
    assert heavy.getParticleMass(1).value_in_unit(unit.dalton) == D_MASS_DA
    with OpenMMPotential(heavy, ["O", "D", "D"] * 2, platform="Reference") as potential:
        np.testing.assert_allclose(potential.evaluate(dimer()).energy, cluster_energy(dimer()), atol=2e-10, rtol=2e-7)


def test_periodic_pme_invariant_to_whole_molecule_images():
    pytest.importorskip("openmm")
    from pyrpmd.potentials.openmm import OpenMMPotential
    q, box = initial_grid()
    cell = np.eye(3) * box * ANGSTROM_TO_AU
    with OpenMMPotential(make_system(molecules=8, box_angstrom=box, cutoff_angstrom=2.5),
                         ["O", "H", "H"] * 8, cell=cell, pbc=[True] * 3,
                         platform="Reference") as potential:
        first = potential.evaluate(q[None])
        image = q.copy()
        image[:3] += cell[0] - cell[1]
        second = potential.evaluate(image[None])
        np.testing.assert_allclose(second.energy, first.energy, atol=2e-12)
        np.testing.assert_allclose(second.forces, first.forces, atol=2e-12)
        np.testing.assert_allclose(dipoles(image[None]), dipoles(q[None]), atol=1e-14)


def test_recorded_water_pilot_replays_without_openmm():
    from benchmarks.water.replay import replay
    path = Path(__file__).resolve().parents[1] / "docs/validation/ecosystem-2026-11-01/water-pilot"
    result = replay(path)
    assert result["artifact_integrity_passed"] and result["estimator_replay_passed"]
    assert result["diffusion_status"] == "ballistic_or_drifting"
    assert not result["physical_converged"] and not result["literature_reproduced"]


def test_recorded_water_pilot_tampering_is_detected(tmp_path):
    from benchmarks.water.replay import replay
    path = Path(__file__).resolve().parents[1] / "docs/validation/ecosystem-2026-11-01/water-pilot"
    target = tmp_path / "pilot"
    shutil.copytree(path, target)
    (target / "model.xml").write_text("tampered model")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        replay(target)
