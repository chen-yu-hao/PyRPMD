"""Independent small-system references for the q-TIP4P/F f-QCMD example."""

import numpy as np
import pytest

from examples.qcmd.water_monomer import (
    MONOMER_MASSES, PARAMETERS, QTIP4PFMonomers, geometry, molecular_dipoles,
)
from pyrpmd.constants import ANGSTROM_TO_AU, BOHR_ANGSTROM, KCAL_MOL_TO_AU
from pyrpmd.potentials.base import check_potential


def _scalar_energy(q):
    """Scalar Eq. (3)/(4) oracle; no production force/geometry helpers."""
    total = 0.
    for a in range(0, len(q), 3):
        v = q[a + 1] - q[a]
        w = q[a + 2] - q[a]
        for vector in (v, w):
            stretch = 2.287 * (np.sqrt(np.dot(vector, vector)) * BOHR_ANGSTROM - .9419)
            total += 116.09 * (stretch * stretch - stretch**3 + 7 * stretch**4 / 12)
        theta = np.arctan2(np.linalg.norm(np.cross(v, w)), np.dot(v, w))
        total += 87.85 / 2 * (theta - 107.4 * np.pi / 180)**2
    return total * KCAL_MOL_TO_AU


def test_monomer_force_matches_independent_scalar_energy_differences():
    q = geometry(2)[None] + np.random.default_rng(97).normal(scale=.05, size=(4, 6, 3))
    actual = QTIP4PFMonomers().evaluate(q)
    np.testing.assert_allclose(actual.energy, [_scalar_energy(frame) for frame in q], atol=2e-16)
    eps = 2e-5
    reference = np.empty_like(q)
    for b in range(len(q)):
        for a in range(q.shape[1]):
            for c in range(3):
                plus, minus = q[b].copy(), q[b].copy()
                plus[a, c] += eps
                minus[a, c] -= eps
                reference[b, a, c] = -(_scalar_energy(plus) - _scalar_energy(minus)) / (2 * eps)
    np.testing.assert_allclose(actual.forces, reference, atol=2e-10)
    assert check_potential(QTIP4PFMonomers(), q)["max_absolute_force_error"] < 1e-9


def test_monomer_equilibrium_symmetries_and_virtual_site_dipole():
    q = geometry(2)[None]
    result = QTIP4PFMonomers().evaluate(q)
    np.testing.assert_allclose(result.energy, 0., atol=1e-26)
    np.testing.assert_allclose(result.forces, 0., atol=1e-14)
    mol = q.reshape(1, 2, 3, 3)
    msite = .73612 * mol[:, :, 0] + (1 - .73612) * (mol[:, :, 1] + mol[:, :, 2]) / 2
    reference = 1.1128 * ((mol[:, :, 1] + mol[:, :, 2]) / 2 - msite)
    np.testing.assert_allclose(molecular_dipoles(q), reference, atol=2e-15)
    rotation, _ = np.linalg.qr(np.random.default_rng(13).normal(size=(3, 3)))
    transformed = q @ rotation + np.array([17., -23., 11.])
    np.testing.assert_allclose(molecular_dipoles(transformed), reference @ rotation, atol=7e-15)
    np.testing.assert_allclose(QTIP4PFMonomers().evaluate(transformed).energy, result.energy, atol=1e-26)


def test_harmonic_cartesian_hessian_matches_wilson_internal_coordinate_reference():
    q = geometry()[None]
    hessian = np.empty((9, 9))
    eps = 1e-5
    potential = QTIP4PFMonomers()
    for i in range(9):
        plus, minus = q.copy(), q.copy()
        plus.flat[i] += eps
        minus.flat[i] -= eps
        hessian[:, i] = -(potential.evaluate(plus).forces - potential.evaluate(minus).forces).ravel() / (2 * eps)
    masses = np.repeat(MONOMER_MASSES, 3)
    eigenvalues = np.linalg.eigvalsh(hessian / np.sqrt(masses[:, None] * masses[None, :]))

    # Wilson G follows directly from scalar r1,r2,theta differential geometry;
    # at equilibrium the potential Hessian is B^T diag(k_r,k_r,k_theta) B.
    r = .9419 * ANGSTROM_TO_AU
    theta = np.deg2rad(107.4)
    m_o, m_h, _ = MONOMER_MASSES
    cosine, sine = np.cos(theta), np.sin(theta)
    metric = np.array([
        [1 / m_h + 1 / m_o, cosine / m_o, -sine / (m_o * r)],
        [cosine / m_o, 1 / m_h + 1 / m_o, -sine / (m_o * r)],
        [-sine / (m_o * r), -sine / (m_o * r),
         2 * (1 / m_h + (1 - cosine) / m_o) / r**2],
    ])
    k_r = 2 * 116.09 * KCAL_MOL_TO_AU * (2.287 * BOHR_ANGSTROM)**2
    root_k = np.sqrt([k_r, k_r, 87.85 * KCAL_MOL_TO_AU])
    reference = np.linalg.eigvalsh(root_k[:, None] * metric * root_k[None, :])
    np.testing.assert_allclose(eigenvalues[-3:], reference, rtol=2e-10, atol=1e-14)
    assert np.max(abs(eigenvalues[:6])) < 1e-12


@pytest.mark.parametrize("q", [np.zeros((2, 3, 3)), np.zeros((2, 4, 3)),
                               np.array([[[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.]]])])
def test_monomer_rejects_invalid_or_singular_geometry(q):
    with pytest.raises(ValueError):
        QTIP4PFMonomers().evaluate(q)
