"""Check published analytic landscapes and independent stationary solutions."""

import numpy as np
import pytest

from benchmarks.transition_state_search.analytic import (
    Adams,
    CerjanMiller,
    MullerBrown,
    SURFACES,
    independent_references,
    summarize_geometry,
)
from pyrpmd.workflows.neb import climbing_image_neb
from pyrpmd.workflows.saddle import refine_transition_state


@pytest.fixture(scope="module")
def references():
    return {(item["surface"], item["id"]): item for item in independent_references()}


@pytest.mark.parametrize("key", [
    ("cerjan_miller", "minimum"),
    ("cerjan_miller", "left_saddle"),
    ("cerjan_miller", "right_saddle"),
    *(("muller_brown", f"SP{i}") for i in range(1, 6)),
])
def test_independent_stationary_points_match_published_precision(key, references):
    reference = references[key]
    assert reference["paper"]["matches_printed_precision"]
    assert reference["gradient_max_unscaled"] < 1e-9
    assert reference["index"] == reference["paper"]["index"]
    if key[0] == "cerjan_miller" and reference["index"] == 1:
        # This energy follows analytically from Eq.21; it is not a quoted
        # paper energy or an independent chemistry benchmark.
        assert abs(reference["energy_unscaled"] - np.exp(-1)) < 1e-12


@pytest.mark.parametrize("factory", [CerjanMiller, MullerBrown, Adams])
@pytest.mark.parametrize("position", [[0.27, 0.35], [-0.6, 1.2]])
def test_analytic_reference_derivatives_match_energy_differences(factory, position):
    surface = factory()
    position = np.asarray(position, dtype=float)
    _, gradient, hessian = surface.raw(position)
    numerical_gradient = np.empty(2)
    numerical_hessian = np.empty((2, 2))
    step = 1e-5
    for i in range(2):
        delta = np.eye(2)[i] * step
        plus = surface.raw(position + delta)
        minus = surface.raw(position - delta)
        numerical_gradient[i] = (plus[0] - minus[0]) / (2 * step)
        numerical_hessian[:, i] = (plus[1] - minus[1]) / (2 * step)
    np.testing.assert_allclose(gradient, numerical_gradient, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(hessian, numerical_hessian, atol=1e-5, rtol=1e-6)


@pytest.mark.parametrize("key", [
    ("cerjan_miller", "left_saddle"),
    ("cerjan_miller", "right_saddle"),
    ("muller_brown", "SP4"),
    ("muller_brown", "SP5"),
    ("adams", "right_saddle"),
    ("adams", "lower_saddle"),
])
@pytest.mark.parametrize("offset", [[0.12, -0.08, 0.05], [-0.10, 0.09, -0.03]])
def test_prfo_recovers_independent_saddles_with_an_exact_index_check(key, offset, references):
    surface = SURFACES[key[0]]()
    reference = references[key]
    guess = (np.r_[reference["xy"], 0.0] + offset)[None]
    result = refine_transition_state(surface, guess, gradient_tolerance=1e-8)
    assert result["converged"], result["diagnostics"]
    measured = summarize_geometry(surface, result["geometry"], reference)
    assert measured["exact_max_force"] <= 1e-8 * (1 + 1e-6)
    assert measured["exact_index"] == 1
    assert measured["reference_coordinate_error"] < 1e-6
    assert measured["reference_energy_error_unscaled"] < 1e-9


@pytest.mark.parametrize("path", [("SP1", "SP3", "SP5"), ("SP3", "SP2", "SP4")])
@pytest.mark.parametrize("images", [7, 13])
def test_cineb_and_refinement_recover_both_muller_brown_barriers(path, images, references):
    surface = MullerBrown()
    start, end, reference = [references[("muller_brown", item)] for item in path]
    left, right = np.array([[*start["xy"], 0.0]]), np.array([[*end["xy"], 0.0]])
    band = climbing_image_neb(surface, left, right, images=images, gradient_tolerance=1e-5)
    assert band["converged"], band["diagnostics"]
    np.testing.assert_array_equal(band["images"][0], left)
    np.testing.assert_array_equal(band["images"][-1], right)
    assert band["max_force"] <= 1e-5
    refined = refine_transition_state(
        surface, band["geometry"], initial_mode=band["tangent"], gradient_tolerance=1e-9
    )
    assert refined["converged"], refined["diagnostics"]
    measured = summarize_geometry(surface, refined["geometry"], reference)
    assert measured["reference_coordinate_error"] < 1e-7
    assert measured["reference_energy_error_unscaled"] < 1e-10
    assert measured["exact_max_force"] <= 1e-9 * (1 + 1e-6)
    assert measured["exact_index"] == 1


@pytest.mark.parametrize("key", [
    ("cerjan_miller", "minimum"),
    ("adams", "minimum"),
    ("adams", "maximum"),
    *(("muller_brown", f"SP{i}") for i in (1, 2, 3)),
])
def test_literature_minima_and_maximum_are_rejected_as_transition_states(key, references):
    reference = references[key]
    result = refine_transition_state(SURFACES[key[0]](), [[*reference["xy"], 0.0]])
    assert not result["converged"]
    assert result["iterations"] == 0


@pytest.mark.parametrize("xy", [
    [-1.169, 0.741], [-1.559, 1.543], [-0.097, 1.076], [-0.995, -0.053]
])
def test_nonstationary_gradient_squared_minima_are_not_false_saddles(xy):
    # Doye-Wales2002 Table I labels these NSP1-NSP4. Their small curvature
    # does not imply a stationary point: the physical force must be checked.
    surface = MullerBrown()
    geometry = np.array([[*xy, 0.0]])
    assert np.max(np.abs(surface.exact(geometry)[1])) > 1e-3
    result = refine_transition_state(
        surface, geometry, max_steps=1, max_step=1e-5, gradient_tolerance=1e-8
    )
    assert not result["converged"]
    assert result["max_force"] > 1e-3
