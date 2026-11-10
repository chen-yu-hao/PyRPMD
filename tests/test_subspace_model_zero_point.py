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
RAW_NORMAL_CURVATURE = 0.02


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


def test_rph_preserves_the_actual_linear_normal_cut_frequency():
    """Differentiate the stated potential instead of inferring a target frequency."""
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
    # At theta=0 a normal displacement changes only the radius. Its second
    # derivative is k, independently of radial force F or path radius R.
    point = np.array([RADIUS, 0.0])
    normal = np.array([1.0, 0.0])
    step = 1e-3
    actual = (_curved_circle_potential(point + step * normal)
              - 2 * _curved_circle_potential(point)
              + _curved_circle_potential(point - step * normal)) / step**2
    assert actual == pytest.approx(RAW_NORMAL_CURVATURE, rel=1e-8)
    expected = 0.5 * np.sqrt(actual)
    assert np.allclose(harmonic.provenance["zero_point_range"], [expected] * 2, atol=2e-6)
    assert np.allclose(rph.values, harmonic.values, atol=1e-13)
    assert rph.provenance["rph_max_correction"] == 0.0


def test_rph_accepts_an_analytic_hessian_without_a_gradient():
    path = np.column_stack([np.linspace(-4.0, 4.0, 25), np.zeros(25)])
    model = project_path_subspace(
        _harmonic_potential, None, path, [MASS, MASS],
        np.linspace(0, 8*np.sqrt(MASS), 9), zero_point="rph",
        hessian=_harmonic_hessian, dividing_surface=4*np.sqrt(MASS))
    assert np.allclose(model.values, OMEGA / 2, atol=1e-13)


@pytest.mark.parametrize("omega", [1e-3, 1e-5])
def test_dvr_default_box_resolves_soft_positive_modes(omega):
    path = np.column_stack([np.linspace(-2, 2, 9), np.zeros(9)])
    potential = lambda x: 0.5 * omega**2 * x[1]**2
    hessian = lambda x: np.diag([0.0, omega**2])
    model = project_path_subspace(potential, None, path, [1, 1], np.linspace(0, 4, 9),
                                  zero_point="dvr", hessian=hessian, dividing_surface=2)
    assert np.allclose(model.values, omega / 2, rtol=1e-9, atol=1e-14)
    assert model.provenance["minimum_complement_curvature"] == pytest.approx(omega**2)


def test_dvr_unstable_diagnostic_counts_points_and_keeps_curvature_units():
    path = np.column_stack([np.linspace(-2, 2, 9), np.zeros((9, 2))])
    potential = lambda x: np.sum(-0.02*x[1:]**2 + 0.0001*x[1:]**4)
    hessian = lambda x: np.diag([0.0, -0.04, -0.04])
    model = project_path_subspace(potential, None, path, [1, 1, 1], np.linspace(0, 4, 9),
                                  zero_point="dvr", hessian=hessian, dividing_surface=2,
                                  dvr_half_width=15, dvr_points=81)
    assert model.provenance["unstable_complement_points"] == 9
    assert model.provenance["minimum_complement_curvature"] == pytest.approx(-0.04)


@pytest.mark.parametrize("selector", ["rph", "dvr"])
def test_old_zero_point_definition_is_not_reused(selector):
    from types import SimpleNamespace
    from pyrpmd.analysis.subspace_pipeline import _saved_model_matches_options

    construction = {"zero_point": selector, "environment": "adiabatic", "dvr_points": 81}
    saved = SimpleNamespace(dimension=1, model={"construction": construction})
    options = {"zero_point": selector}
    assert not _saved_model_matches_options(saved, 1, options)
    construction["zero_point_definition"] = "linear-normal-cuts-v2"
    assert _saved_model_matches_options(saved, 1, options)


def test_pipeline_rejects_fractional_dvr_points_without_truncating():
    from types import SimpleNamespace
    from pyrpmd.analysis.subspace_pipeline import DEFAULTS, path_model

    path = np.column_stack([np.linspace(-2, 2, 9), np.zeros((9, 2))])
    run = SimpleNamespace(cartesian_masses=np.ones(3), beta=1.0, xi_star=0.0,
                          energy=lambda x: float(x[0]**2), gradient=lambda x: 2*x,
                          xi=lambda x: float(x[0]))
    with pytest.raises(ValueError, match="dvr_points must be an integer"):
        path_model(run, path, {**DEFAULTS["quantum"], "zero_point": "dvr",
                               "dvr_points": 51.5})
