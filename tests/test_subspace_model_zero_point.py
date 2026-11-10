"""Regression tests for reaction-path environment zero-point projections.

The three path projections are deliberately exercised on a separable harmonic
normal mode first.  This keeps the tests independent of the exact-rate and
RPMD samplers while checking the contract used by both halves of ``C_S``.
"""

import numpy as np
import pytest

from pyrpmd.analysis.subspace_model import path_curvature, project_path_subspace


MASS = 1060.0
OMEGA = 0.010
CURVATURE_H = 0.9
CURVATURE_A = 0.66
CURVATURE_BARRIER = 0.425 / 27.211386


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


def _curvature_potential(points):
    points = np.asarray(points, dtype=float)
    s, y = points[..., 0], points[..., 1]
    sech2 = 1.0 / np.cosh(s / CURVATURE_A) ** 2
    centre = CURVATURE_H * sech2
    return CURVATURE_BARRIER * sech2 + 0.5 * MASS * OMEGA**2 * (y - centre) ** 2


def _curvature_gradient(points):
    points = np.asarray(points, dtype=float)
    s, y = points[..., 0], points[..., 1]
    tanh = np.tanh(s / CURVATURE_A)
    sech2 = 1.0 / np.cosh(s / CURVATURE_A) ** 2
    centre = CURVATURE_H * sech2
    centre_gradient = -2.0 * CURVATURE_H / CURVATURE_A * tanh * sech2
    out = np.empty_like(points)
    out[..., 0] = (-2.0 * CURVATURE_BARRIER / CURVATURE_A * tanh * sech2
                   - MASS * OMEGA**2 * (y - centre) * centre_gradient)
    out[..., 1] = MASS * OMEGA**2 * (y - centre)
    return out


def _straight_path_model(*, zero_point, **dvr_options):
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
        **dvr_options,
    )


@pytest.mark.parametrize("zero_point", ["harmonic", "rph", "dvr"])
def test_straight_path_zero_point_options_are_consistent(zero_point):
    """A straight separable valley has no curvature correction to make."""
    model = _straight_path_model(zero_point=zero_point)
    expected = 0.5 * OMEGA
    assert np.allclose(model.values, expected, atol=1e-6)


@pytest.mark.parametrize("omega", [1e-4, 1e-6, 1e-10])
def test_default_dvr_box_resolves_soft_positive_mode(monkeypatch, omega):
    monkeypatch.setitem(globals(), "OMEGA", omega)
    model = _straight_path_model(zero_point="dvr")
    np.testing.assert_allclose(model.values, .5 * omega, rtol=1e-7, atol=1e-13)


@pytest.mark.parametrize("zero_point", ["harmonic", "rph", "dvr"])
def test_zero_point_and_sct_selectors_compose_on_straight_path(zero_point):
    reference = _straight_path_model(zero_point=zero_point)
    actual = _straight_path_model(zero_point=zero_point, curvature="small")
    np.testing.assert_allclose(actual.values, reference.values, atol=1e-12)
    assert actual.provenance["effective_mass_ratio_min"] == 1.


def test_harmonic_normal_cut_dvr_matches_harmonic_zpe_to_one_part_per_million():
    harmonic = _straight_path_model(zero_point="harmonic")
    dvr = _straight_path_model(zero_point="dvr")
    assert np.max(np.abs(dvr.values - harmonic.values)) < 1e-6
    assert dvr.provenance["zero_point"] == "dvr"


def test_dvr_cut_grid_refinement_is_stable():
    """Changing both DVR resolution controls leaves a converged cut unchanged."""
    coarse = _straight_path_model(zero_point="dvr", dvr_points=41, dvr_half_width=80.0)
    refined_points = _straight_path_model(zero_point="dvr", dvr_points=81, dvr_half_width=80.0)
    widened_box = _straight_path_model(zero_point="dvr", dvr_points=81, dvr_half_width=100.0)

    # Compare the complete projected path, rather than one representative cut.
    assert np.max(np.abs(coarse.values - refined_points.values)) < 1e-8
    assert np.max(np.abs(refined_points.values - widened_box.values)) < 1e-8
    assert coarse.provenance["dvr_points"] == 41
    assert refined_points.provenance["dvr_points"] == 81
    assert refined_points.provenance["dvr_half_width"] == 80.0
    assert widened_box.provenance["dvr_half_width"] == 100.0


def test_curvature_point_dvr_cut_grid_refinement_is_stable():
    """The grid check also holds at the h=0.9 curvature-family barrier point."""
    path_coordinate = np.linspace(-16.0, 16.0, 801)
    path = np.column_stack([path_coordinate,
                             CURVATURE_H / np.cosh(path_coordinate / CURVATURE_A) ** 2])
    arc, _ = path_curvature(path, [1060.0, 1060.0], samples=801)
    top = path_coordinate[np.argmax(CURVATURE_BARRIER / np.cosh(path_coordinate / CURVATURE_A) ** 2)]
    centre = float(np.interp(top, path_coordinate, arc))
    axis = centre + np.arange(-8.0, 10.0, 2.0)

    def project(points, half_width):
        return project_path_subspace(
            lambda x: float(_curvature_potential(np.asarray(x)[None])[0]),
            lambda x: _curvature_gradient(np.asarray(x)[None])[0],
            path,
            [1060.0, 1060.0],
            axis,
            environment="adiabatic",
            zero_point="dvr",
            dvr_points=points,
            dvr_half_width=half_width,
            dividing_surface=centre,
            reactant_reference=float(axis[0]),
        )

    coarse = project(41, 80.0)
    refined_points = project(81, 80.0)
    widened_box = project(81, 100.0)

    assert np.max(np.abs(coarse.values - refined_points.values)) < 1e-8
    assert np.max(np.abs(refined_points.values - widened_box.values)) < 1e-8
    assert coarse.provenance["dvr_points"] == 41
    assert refined_points.provenance["dvr_points"] == 81
    assert refined_points.provenance["dvr_half_width"] == 80.0
    assert widened_box.provenance["dvr_half_width"] == 100.0


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
