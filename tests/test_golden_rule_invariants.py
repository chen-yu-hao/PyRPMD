"""Coordinate covariance and energy-origin invariants of the public GR API.

These tests use two unequal atomic masses, noncontiguous active Cartesian
coordinates and two noncommuting diabatic Hessians.  No production action,
Hessian, Schur-complement or partition implementation is used as an oracle.
"""
import math

import numpy as np
import pytest

from pyrpmd.constants import KB_AU
from pyrpmd.golden_rule import golden_rule_rate, optimize_golden_rule_instanton
from pyrpmd.potentials.base import PESResult


_ACTIVE = np.array([0, 5])  # atom0.x and atom1.z
_MASK = np.array([[True, False, False], [False, False, True]])
_MASSES = np.array([3.2, 7.5])
_K0 = np.array([[0.4, 0.12], [0.12, 0.8]])
_K1 = np.array([[0.7, -0.08], [-0.08, 0.3]])
_CENTER0 = np.array([-0.5, 0.2])
_CENTER1 = np.array([0.8, -0.3])
_BETA = 4.7


class CoupledQuadraticSurface:
    """A genuinely coupled two-dimensional PES embedded in two atoms."""

    def __init__(self, hessian, center, offset=0.0):
        self.matrix = np.asarray(hessian)
        self.center = np.asarray(center)
        self.offset = offset

    def evaluate(self, q):
        displacement = q.reshape(len(q), 6)[:, _ACTIVE] - self.center
        gradient = displacement @ self.matrix
        forces = np.zeros((len(q), 6))
        forces[:, _ACTIVE] = -gradient
        return PESResult(
            0.5 * np.sum(displacement * gradient, axis=1) + self.offset,
            forces.reshape(q.shape),
        )

    def hessian(self, q):
        full = np.zeros((len(q), 6, 6))
        for i, row in enumerate(_ACTIVE):
            for j, column in enumerate(_ACTIVE):
                full[:, row, column] = self.matrix[i, j]
        return full


def _solve(scale, reference=0.0):
    # x=C*y implies M_y=C^T*M_x*C, K_y=C^T*K_x*C and y_min=C^-1*x_min.
    # Both actions and the fully normalized rate must be coordinate invariant.
    scale = np.asarray(scale)
    transform = np.diag(scale)
    masses = _MASSES * scale**2
    reactant_hessian = transform @ _K0 @ transform
    potentials = (
        CoupledQuadraticSurface(reactant_hessian, _CENTER0 / scale),
        CoupledQuadraticSurface(transform @ _K1 @ transform, _CENTER1 / scale, -0.07),
    )
    initial0 = np.zeros((4, 2, 3))  # three intervals on state 0
    initial1 = np.zeros((6, 2, 3))  # five intervals on state 1
    orbit = optimize_golden_rule_instanton(
        *potentials, masses, 1 / (KB_AU * _BETA), initial0, initial1,
        active_mask=_MASK, energy_reference_hartree=reference,
        gradient_tolerance=1e-11, energy_tolerance_hartree=1e-12,
        tau_fraction_tolerance=1e-13,
    )
    # Independent continuum partition of this bound quadratic reactant.
    # The electronic minimum remains zero when only the reference changes.
    frequencies = np.sqrt(np.linalg.eigvalsh(
        reactant_hessian / np.sqrt(masses[:, None] * masses[None, :])
    ))
    log_partition = (
        -float(np.log(2 * np.sinh(_BETA * frequencies / 2)).sum())
        + _BETA * reference
    )
    rate = golden_rule_rate(
        orbit, coupling_hartree=1e-3, log_reactant_partition=log_partition,
    )
    return orbit, rate


@pytest.fixture(scope="module")
def base_result():
    return _solve(np.ones(2))


def test_different_masses_and_noncontiguous_coordinates_obey_coordinate_covariance(base_result):
    base, base_rate = base_result
    scale = np.array([2.0, 0.3])
    transformed, rate = _solve(scale)

    assert transformed.action_hbar == pytest.approx(base.action_hbar, abs=1e-10)
    assert transformed.tau1_au == pytest.approx(base.tau1_au, abs=1e-9)
    assert transformed.log_coordinate_prefactor == pytest.approx(
        base.log_coordinate_prefactor, abs=1e-10,
    )
    assert transformed.time_schur_curvature_hartree2 == pytest.approx(
        base.time_schur_curvature_hartree2, rel=1e-9,
    )
    np.testing.assert_allclose(
        transformed.path0_bohr[:, _MASK] * scale, base.path0_bohr[:, _MASK],
        rtol=0, atol=2e-9,
    )
    np.testing.assert_allclose(
        transformed.path1_bohr[:, _MASK] * scale, base.path1_bohr[:, _MASK],
        rtol=0, atol=2e-9,
    )
    assert rate["log_reactant_partition"] == pytest.approx(
        base_rate["log_reactant_partition"], abs=1e-13,
    )
    assert rate["log_rate_au"] == pytest.approx(base_rate["log_rate_au"], abs=1e-9)
    assert rate["rate"] == pytest.approx(base_rate["rate"], rel=1e-9)


def test_energy_reference_changes_action_and_partition_but_not_rate_for_fixed_pes(base_result):
    base, base_rate = base_result
    reference = 0.37
    shifted, rate = _solve(np.ones(2), reference)

    # The PES offsets have not changed.  This specifically checks a reference
    # change, rather than shifting the PES and reference together.
    assert shifted.action_hbar - base.action_hbar == pytest.approx(
        -_BETA * reference, abs=1e-10,
    )
    assert shifted.energy0_hartree - base.energy0_hartree == pytest.approx(-reference, abs=1e-11)
    assert shifted.energy1_hartree - base.energy1_hartree == pytest.approx(-reference, abs=1e-11)
    assert shifted.energy_reference_hartree == reference
    assert shifted.tau1_au == pytest.approx(base.tau1_au, abs=1e-9)
    assert rate["log_reactant_partition"] - base_rate["log_reactant_partition"] == pytest.approx(
        _BETA * reference, abs=1e-14,
    )
    assert rate["log_rate_au"] == pytest.approx(base_rate["log_rate_au"], abs=1e-9)
    assert math.isfinite(rate["rate"])
    assert rate["rate"] == pytest.approx(base_rate["rate"], rel=1e-9)
