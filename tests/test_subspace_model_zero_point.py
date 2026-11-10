"""Regression tests for reaction-path environment zero-point projections.

The three path projections are deliberately exercised on a separable harmonic
normal mode first.  This keeps the tests independent of the exact-rate and
RPMD samplers while checking the contract used by both halves of ``C_S``.
"""

import numpy as np
import pytest

from pyrpmd.analysis.subspace_model import project_path_subspace


MASS = 1060.0
OMEGA = 0.010


def _harmonic_potential(points):
    points = np.asarray(points, dtype=float)
    return 0.5 * MASS * OMEGA**2 * points[..., 1] ** 2


def _harmonic_gradient(points):
    points = np.asarray(points, dtype=float)
    out = np.zeros_like(points)
    out[..., 1] = MASS * OMEGA**2 * points[..., 1]
    return out


def _harmonic_hessian(_point):
    return np.array([[0.0, 0.0], [0.0, MASS * OMEGA**2]])


def _straight_path_model(*, zero_point):
    path_s = np.linspace(-4.0, 4.0, 25)
    path = np.column_stack([path_s, np.zeros_like(path_s)])
    axis = np.linspace(0.0, 8.0 * np.sqrt(MASS), 41)
    return project_path_subspace(
        _harmonic_potential,
        _harmonic_gradient,
        path,
        [MASS, MASS],
        axis,
        environment="adiabatic",
        zero_point=zero_point,
        hessian=_harmonic_hessian,
        dividing_surface=4.0 * np.sqrt(MASS),
        reactant_reference=1.0 * np.sqrt(MASS),
    )


@pytest.mark.parametrize("zero_point", ["harmonic", "rph", "dvr"])
def test_straight_path_zero_point_options_are_consistent(zero_point):
    """A straight separable valley has no curvature correction to make."""
    model = _straight_path_model(zero_point=zero_point)
    expected = 0.5 * OMEGA
    assert np.allclose(model.values, expected, atol=1e-6)


def test_harmonic_normal_cut_dvr_matches_harmonic_zpe_to_one_part_per_million():
    harmonic = _straight_path_model(zero_point="harmonic")
    dvr = _straight_path_model(zero_point="dvr")
    assert np.max(np.abs(dvr.values - harmonic.values)) < 1e-6
    assert dvr.provenance["zero_point"] == "dvr"


def test_dvr_free_energy_matches_harmonic_normal_cut_free_energy():
    beta = 100.0
    model = project_path_subspace(
        _harmonic_potential,
        _harmonic_gradient,
        np.column_stack([np.linspace(-4.0, 4.0, 25), np.zeros(25)]),
        [MASS, MASS],
        np.linspace(0.0, 8.0 * np.sqrt(MASS), 21),
        environment="free_energy",
        beta=beta,
        zero_point="dvr",
        hessian=_harmonic_hessian,
        dividing_surface=4.0 * np.sqrt(MASS),
        reactant_reference=1.0 * np.sqrt(MASS),
    )
    expected = np.log(2.0 * np.sinh(0.5 * beta * OMEGA)) / beta
    assert np.allclose(model.values, expected, atol=1e-10)


def test_unknown_path_zero_point_option_is_rejected():
    with pytest.raises(ValueError, match="zero_point"):
        _straight_path_model(zero_point="not-a-projection")


RADIUS = 5.0
RADIAL_FORCE = 0.10
PHYSICAL_OMEGA = 0.20
# The radial Hessian is deliberately chosen so that the force-curvature
# connection is a visible part of the raw normal frequency.
RAW_NORMAL_CURVATURE = PHYSICAL_OMEGA**2 - RADIAL_FORCE / RADIUS


def _curved_circle_potential(points):
    points = np.asarray(points, dtype=float)
    radius = np.linalg.norm(points, axis=-1)
    q = radius - RADIUS
    return RADIAL_FORCE * q + 0.5 * RAW_NORMAL_CURVATURE * q**2


def _curved_circle_gradient(points):
    points = np.asarray(points, dtype=float)
    radius = np.linalg.norm(points, axis=-1)
    q = radius - RADIUS
    return ((RADIAL_FORCE + RAW_NORMAL_CURVATURE * q) / radius)[..., None] * points


def _curved_circle_hessian(point):
    point = np.asarray(point, dtype=float)
    radius = np.linalg.norm(point)
    normal = point / radius
    tangent_projector = np.eye(2) - np.outer(normal, normal)
    radial = RADIAL_FORCE + RAW_NORMAL_CURVATURE * (radius - RADIUS)
    return RAW_NORMAL_CURVATURE * np.outer(normal, normal) + (radial / radius) * tangent_projector


def test_rph_removes_the_curvature_induced_normal_frequency():
    """A circular analytic path exposes the MHA force-curvature term."""
    theta = np.linspace(-0.8, 0.8, 81)
    path = RADIUS * np.column_stack([np.cos(theta), np.sin(theta)])
    arc = RADIUS * (theta - theta[0])
    axis = np.linspace(arc[0], arc[-1], 61)
    harmonic = project_path_subspace(
        _curved_circle_potential,
        _curved_circle_gradient,
        path,
        [1.0, 1.0],
        axis,
        environment="adiabatic",
        zero_point="harmonic",
        hessian=_curved_circle_hessian,
        dividing_surface=float(axis[len(axis) // 2]),
        reactant_reference=float(axis[10]),
    )
    rph = project_path_subspace(
        _curved_circle_potential,
        _curved_circle_gradient,
        path,
        [1.0, 1.0],
        axis,
        environment="adiabatic",
        zero_point="rph",
        hessian=_curved_circle_hessian,
        dividing_surface=float(axis[len(axis) // 2]),
        reactant_reference=float(axis[10]),
    )
    # The raw normal cut sees sqrt(omega^2 - F/R), whereas the MHA
    # force-curvature subtraction restores the physical omega mode.
    assert np.allclose(harmonic.provenance["zero_point_range"], [0.5 * np.sqrt(RAW_NORMAL_CURVATURE)] * 2, atol=2e-6)
    assert np.allclose(rph.provenance["zero_point_range"], [0.5 * PHYSICAL_OMEGA] * 2, atol=5e-6)
    assert rph.provenance["rph_max_correction"] == pytest.approx(RADIAL_FORCE / RADIUS, abs=2e-5)
