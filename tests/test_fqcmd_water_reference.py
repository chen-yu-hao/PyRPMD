"""Independent checks of the illustrative water model used by the f-QCMD demo."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.constants import WAVENUMBER_TO_AU
from pyrpmd.potentials.base import PotentialError


@pytest.fixture(scope="module")
def example():
    path = Path(__file__).parents[1] / "examples" / "spectroscopy" / "fqcmd_water.py"
    spec = importlib.util.spec_from_file_location("fqcmd_water_reference_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analytic_water_cartesian_force_matches_energy_finite_difference(example):
    result = example.force_reference(example.AnalyticWaterPotential())
    assert result["max_absolute_force_error_hartree_per_bohr"] < 2e-9


def test_analytic_water_is_rigid_motion_covariant_and_molecule_additive(example):
    potential = example.AnalyticWaterPotential()
    rng = np.random.default_rng(702)
    q = np.tile(potential.equilibrium(3)[None], (5, 1, 1))
    q += rng.normal(scale=0.07, size=q.shape)
    before = potential.evaluate(q)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation[:, -1] *= np.linalg.det(rotation)
    moved = q @ rotation + np.array([10.3, -17.1, 4.5])
    after = potential.evaluate(moved)
    np.testing.assert_allclose(after.energy, before.energy, atol=5e-15)
    np.testing.assert_allclose(after.forces, before.forces @ rotation, atol=5e-15)
    np.testing.assert_allclose(potential.molecular_dipoles(moved),
                               potential.molecular_dipoles(q) @ rotation, atol=5e-15)
    separate = [potential.evaluate(q[:, start:start + 3]) for start in range(0, 9, 3)]
    np.testing.assert_allclose(before.energy, sum(result.energy for result in separate), atol=5e-17)
    np.testing.assert_allclose(before.forces.sum(axis=1), 0, atol=5e-17)
    torque = np.sum(np.cross(q, before.forces), axis=1)
    np.testing.assert_allclose(torque, 0, atol=5e-15)


def test_equilibrium_hessian_matches_internal_coordinate_harmonic_reference(example):
    potential = example.AnalyticWaterPotential()
    q = potential.equilibrium()
    np.testing.assert_allclose(potential.evaluate(q[None]).forces, 0, atol=2e-15)
    assert potential.evaluate(q[None]).energy[0] < 1e-28
    # Independent local quadratic energy: 1/2 sum k_s (J_s dq)^2.
    vectors = q[1:] - q[0]
    lengths = np.linalg.norm(vectors, axis=1)
    u, v = vectors / lengths[:, None]
    cosine, sine = np.dot(u, v), np.linalg.norm(np.cross(u, v))
    jacobian = np.zeros((3, 3, 3))
    jacobian[0, 1], jacobian[0, 0] = u, -u
    jacobian[1, 2], jacobian[1, 0] = v, -v
    jacobian[2, 1] = (cosine * u - v) / (lengths[0] * sine)
    jacobian[2, 2] = (cosine * v - u) / (lengths[1] * sine)
    jacobian[2, 0] = -jacobian[2, 1] - jacobian[2, 2]
    p = potential.parameters
    stiffness = np.diag([2 * p["D_hartree"] * p["a_inverse_bohr"]**2] * 2
                        + [p["k_theta_hartree"]])
    jacobian = jacobian.reshape(3, 9)
    mass = np.repeat(potential.masses(), 3)
    hessian = jacobian.T @ stiffness @ jacobian / np.sqrt(mass[:, None] * mass[None, :])
    expected = np.sqrt(np.linalg.eigvalsh(hessian)[-3:]) / WAVENUMBER_TO_AU
    np.testing.assert_allclose(example.harmonic_frequencies(potential), expected, rtol=2e-9)
    assert 1200 < expected[0] < 2000
    assert np.all((expected[1:] > 3500) & (expected[1:] < 4500))


@pytest.mark.parametrize("kind", ["wrong_atoms", "zero_bond", "linear"])
def test_analytic_water_rejects_unsupported_geometry(example, kind):
    potential = example.AnalyticWaterPotential()
    q = potential.equilibrium()[None]
    if kind == "wrong_atoms":
        q = q[:, :2]
    elif kind == "zero_bond":
        q[:, 1] = q[:, 0]
    else:
        q[:, 2] = 2 * q[:, 0] - q[:, 1]
    with pytest.raises(PotentialError):
        potential.evaluate(q)
