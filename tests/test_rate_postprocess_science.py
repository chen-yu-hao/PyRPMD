"""Independent physical invariants for two-dimensional rate postprocessing.

These tests use uniform ideal-gas geometry, a Gaussian transverse partition
function, changes of collective-variable coordinates, and admissible signed
paired fluxes.  They do not use trajectories or fitted numbers from the new
implementation as their reference answers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erf

from pyrpmd.analysis.rate_postprocess import (
    channel_rates,
    flux_slice_rate_2d,
    surface_integral_2d,
    surface_rate_2d,
)
from pyrpmd.analysis.recrossing import summarize_recrossing
from pyrpmd.constants import AU_RATE_TO_CM3_S


def _flat_grid():
    # The first x cell has width 1.5; the full y domain has width 8.
    # Consequently the chosen reactant region has area 12, regardless of
    # the very unequal bin widths or the number of bins.
    edges = (np.array([-2.0, -0.5, 0.2, 1.0, 4.0]),
             np.array([-3.0, -1.0, 0.0, 0.7, 3.0, 5.0]))
    free_energy = np.zeros((4, 5))
    reactants = np.zeros_like(free_energy, dtype=bool)
    reactants[0, :] = True
    curve = np.array([[0.6, -2.5], [0.6, 4.5]])
    return free_energy, edges, reactants, curve


def _flat_rate(*, offset=0.0, x_scale=1.0, y_scale=1.0):
    energy, edges, reactants, curve = _flat_grid()
    beta = 2.3
    jacobian = x_scale * y_scale
    # A PMF is a density: under x'=a*x,y'=b*y it gains log(a*b)/beta.
    energy = energy + offset + math.log(jacobian) / beta
    transformed_edges = (edges[0] * x_scale, edges[1] * y_scale)
    transformed_curve = curve * np.array([x_scale, y_scale])
    return surface_rate_2d(
        energy, transformed_edges, beta=beta, curve=transformed_curve,
        normal_speed=3.0 * x_scale,
        reactant_mask=reactants, reactant_volume_au=24.0,
    )


def test_uniform_gas_flux_uses_actual_area_on_nonuniform_grid():
    result = _flat_rate()
    # Collision frequency: V * (surface length * positive velocity) / area.
    expected = 24.0 * 7.0 * 3.0 / 12.0
    assert result["k_tst_au"] == pytest.approx(expected, rel=2e-12)
    assert result["k_tst_cm3_s"] == pytest.approx(
        expected * AU_RATE_TO_CM3_S, rel=2e-12,
    )
    assert result["converged"] is False


@pytest.mark.parametrize("offset", [-1.0e4, 1.0e4])
def test_rate_is_invariant_to_extreme_free_energy_zero(offset):
    # Neither exp(+23000) nor exp(-23000) is representable in float64.
    # The observable remains ordinary-sized because the zero must cancel.
    assert _flat_rate(offset=offset)["k_tst_au"] == pytest.approx(
        _flat_rate()["k_tst_au"], rel=2e-10,
    )


def test_rate_is_invariant_to_collective_variable_unit_changes():
    # This is a coordinate change, not a physical change in the gas volume.
    # Density, normal velocity and line element must transform together.
    baseline = _flat_rate()["k_tst_au"]
    assert _flat_rate(x_scale=5.0, y_scale=0.25)["k_tst_au"] == pytest.approx(
        baseline, rel=2e-12,
    )


def test_curve_parameterization_and_orientation_do_not_change_positive_flux():
    energy, edges, _, curve = _flat_grid()
    kwargs = dict(beta=0.8, normal_speed=1.7)
    whole = surface_integral_2d(energy, edges, curve=curve, **kwargs)
    subdivided = np.linspace(curve[0], curve[1], 41)
    split = surface_integral_2d(energy, edges, curve=subdivided, **kwargs)
    reversed_curve = surface_integral_2d(
        energy, edges, curve=subdivided[::-1], **kwargs,
    )
    assert math.exp(whole["log_integral"]) == pytest.approx(7.0 * 1.7)
    assert split["log_integral"] == pytest.approx(whole["log_integral"], abs=1e-12)
    assert reversed_curve["log_integral"] == pytest.approx(
        whole["log_integral"], abs=1e-12,
    )


def test_gaussian_transverse_partition_function_has_correct_width():
    beta, curvature, speed, limit = 1.4, 0.8, 0.7, 6.0
    edges = (np.array([-1.0, 0.0, 1.0]), np.linspace(-limit, limit, 601))
    y = 0.5 * (edges[1][1:] + edges[1][:-1])
    energy = np.broadcast_to(0.5 * curvature * y**2, (2, len(y))).copy()
    result = surface_integral_2d(
        energy, edges, beta=beta, curve=[[0.0, -limit], [0.0, limit]],
        normal_speed=speed,
    )
    expected = speed * math.sqrt(2 * math.pi / (beta * curvature)) * erf(
        limit * math.sqrt(beta * curvature / 2),
    )
    # The PMF is sampled at cell centers, so there is a small, controlled
    # interpolation error relative to the continuous harmonic function.
    assert math.exp(result["log_integral"]) == pytest.approx(expected, rel=1e-4)


def test_anisotropic_mass_metric_on_oblique_surface():
    beta = 1.6
    edges = (np.linspace(-1.0, 3.0, 17), np.linspace(-1.0, 4.0, 21))
    energy = np.zeros((16, 20))
    inverse_mass = np.diag([0.25, 4.0])

    def positive_speed(points, normals):
        assert points.shape == normals.shape
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
        return np.sqrt(np.einsum("ni,ij,nj->n", normals, inverse_mass, normals)
                       / (2 * math.pi * beta))

    result = surface_integral_2d(
        energy, edges, beta=beta, curve=[[0.0, 0.0], [2.0, 3.0]],
        normal_speed=positive_speed,
    )
    # Tangent=(2,3), so |ds|^2 n^T G n = 9 G_xx + 4 G_yy.
    expected = math.sqrt((9 * 0.25 + 4 * 4.0) / (2 * math.pi * beta))
    assert math.exp(result["log_integral"]) == pytest.approx(expected, rel=1e-12)


def test_inverse_mass_uses_unconditional_positive_velocity_average():
    energy, edges, _, curve = _flat_grid()
    beta, inv_mass = 1.7, 0.4
    result = surface_integral_2d(
        energy, edges, beta=beta, curve=curve, inverse_mass=inv_mass,
    )
    # Conditional mean v|v>0 would be twice as large and would fail this.
    expected = 7.0 * math.sqrt(inv_mass / (2 * math.pi * beta))
    assert math.exp(result["log_integral"]) == pytest.approx(expected, rel=1e-12)


def test_reference_surface_normalization_preserves_known_entrance_flux():
    energy, edges, _, curve = _flat_grid()
    # Both surfaces are in the same uniform equilibrium density. Their
    # integrated fluxes differ only by length (7 versus 4) and speed (3/2).
    result = surface_rate_2d(
        energy, edges, beta=1.3, curve=curve, normal_speed=3.0,
        reference_curve=[[-1.0, -2.0], [-1.0, 2.0]],
        reference_normal_speed=2.0, reference_rate_au=11.0,
    )
    assert result["k_tst_au"] == pytest.approx(11.0 * 21.0 / 8.0, rel=1e-12)


def test_a_masked_strip_cannot_be_skipped_by_sparse_curve_vertices():
    energy, edges, _, curve = _flat_grid()
    mask = np.zeros_like(energy, dtype=bool)
    mask[:, 2] = True
    with pytest.raises(ValueError, match="mask|support|unsupported"):
        surface_integral_2d(
            energy, edges, beta=1.0, curve=curve, normal_speed=1.0, mask=mask,
        )


def test_missing_bins_away_from_curve_do_not_destroy_a_local_integral():
    energy, edges, _, curve = _flat_grid()
    mask = np.zeros_like(energy, dtype=bool)
    mask[-1, :] = True
    energy[mask] = np.nan
    result = surface_integral_2d(
        energy, edges, beta=1.0, curve=curve, normal_speed=1.0, mask=mask,
    )
    assert math.exp(result["log_integral"]) == pytest.approx(7.0)


def test_reactant_partition_function_cannot_silently_drop_missing_bins():
    energy, edges, reactants, curve = _flat_grid()
    mask = np.zeros_like(energy, dtype=bool)
    mask[0, 0] = True
    with pytest.raises(ValueError, match="mask|support|reactant"):
        surface_rate_2d(
            energy, edges, beta=1.0, curve=curve, normal_speed=1.0,
            reactant_mask=reactants, reactant_volume_au=1.0, mask=mask,
        )


@pytest.mark.parametrize("speed", [0.0, -1.0, np.nan, np.inf])
def test_nonphysical_velocity_scales_are_rejected(speed):
    energy, edges, _, curve = _flat_grid()
    with pytest.raises(ValueError):
        surface_integral_2d(energy, edges, beta=1.0, curve=curve, normal_speed=speed)


def test_mass_and_speed_cannot_both_be_supplied():
    energy, edges, _, curve = _flat_grid()
    with pytest.raises(ValueError):
        surface_integral_2d(
            energy, edges, beta=1.0, curve=curve,
            normal_speed=1.0, inverse_mass=1.0,
        )


def _separable_flux_fes():
    edges = (np.array([-0.5, 0.5, 1.5, 2.5]),
             np.array([-3.0, -1.0, 0.0, 0.4, 2.0, 5.0]))
    beta = 0.7
    energy = np.array([0.0, 1.1, 0.3])[:, None] + np.array(
        [2.1, 0.2, 0.0, 0.7, 3.8],
    )[None, :]
    return energy, edges, beta


def test_flux_weighted_two_dimensional_fes_marginalizes_to_one_dimensional_rate():
    energy, edges, beta = _separable_flux_fes()
    result = flux_slice_rate_2d(
        energy, edges, beta=beta, surface=1.0, reference=0.0,
        reference_rate_au=17.0,
    )
    # The same transverse partition function cancels. No additional
    # Maxwell velocity or arbitrary transverse-bin count may be inserted.
    assert result["k_tst_au"] == pytest.approx(17.0 * math.exp(-beta * 1.1), rel=2e-12)


@pytest.mark.parametrize("offset", [-1.0e5, 1.0e5])
def test_flux_reference_ratio_survives_unrepresentable_absolute_weights(offset):
    energy, edges, beta = _separable_flux_fes()
    result = flux_slice_rate_2d(
        energy + offset, edges, beta=beta, surface=1.0, reference=0.0,
        reference_rate_au=17.0,
    )
    assert result["k_tst_au"] == pytest.approx(17.0 * math.exp(-beta * 1.1), rel=5e-10)


@pytest.mark.parametrize("options", [{"measure": "cv_density"}, {"flux_axis": 1}])
def test_flux_measure_does_not_allow_reinterpretation_or_rotation(options):
    energy, edges, beta = _separable_flux_fes()
    with pytest.raises(ValueError):
        flux_slice_rate_2d(
            energy, edges, beta=beta, surface=1.0, reference=0.0,
            reference_rate_au=17.0, **options,
        )


def _signed_parents():
    # One negative term is real: forward P2, backward P1 gives (-1,+1).
    # Its total is zero, so it must not be counted as a new net reaction.
    outcomes = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0], [1.0, 0.0]])
    outcomes = np.tile(outcomes, (2, 1))
    num = np.repeat(outcomes[:, None, :], 80, axis=1)
    num[:, :12, :] = 0.0
    den = np.ones(len(outcomes))
    ids = np.array([(i // 4, i % 4) for i in range(len(outcomes))])
    return np.arange(80, dtype=float) * 0.5, num, den, ids


def test_signed_backward_product_contribution_changes_both_rate_and_branching():
    time, num, den, ids = _signed_parents()
    result = channel_rates(8.0, time, num, den, ids, ["P1", "P2"], bootstrap=80, seed=14)
    np.testing.assert_allclose(result["kappa"], [0.25, 0.5], rtol=1e-14)
    assert result["total_kappa"] == pytest.approx(0.75)
    np.testing.assert_allclose(result["channel_rate_estimates_au"], [2.0, 4.0])
    np.testing.assert_allclose(result["branching_fraction_estimates"], [1 / 3, 2 / 3])
    assert result["converged"] is False


def test_joint_bootstrap_preserves_exact_channel_conservation_and_covariance():
    n_parent, n_time = 40, 80
    num = np.zeros((n_parent, n_time, 2))
    selected_p1 = np.r_[np.arange(15), np.arange(20, 25)]
    num[:, :, 1] = 1.0
    num[selected_p1, :, 0] = 1.0
    num[selected_p1, :, 1] = 0.0
    num[:, :10, :] = 0.0
    den = np.ones(n_parent)
    ids = np.array([(i // 20, i % 20) for i in range(n_parent)])
    result = channel_rates(
        2.0, np.arange(n_time, dtype=float), num, den, ids, ["P1", "P2"],
        bootstrap=160, seed=891,
    )
    draws = np.asarray(result["bootstrap_kappa"])
    assert draws.shape == (160, 2)
    np.testing.assert_allclose(draws.sum(axis=1), 1.0, atol=2e-14)
    covariance = np.cov(draws, rowvar=False)
    assert covariance[0, 0] > 0
    assert covariance[0, 1] == pytest.approx(-covariance[0, 0], abs=1e-14)
    assert covariance[1, 1] == pytest.approx(covariance[0, 0], abs=1e-14)


def test_single_channel_agrees_with_existing_signed_flux_estimator():
    time, num, den, ids = _signed_parents()
    total = num.sum(axis=2)
    children = [
        {"chain": int(chain), "parent": int(parent), "pair": 0,
         "numerator": row, "denominator": float(weight)}
        for (chain, parent), row, weight in zip(ids, total, den)
    ]
    existing = summarize_recrossing(
        children, child_steps=len(time), plateau_fraction=0.25,
        plateau_tolerance=0.1, bootstrap=80, rng=np.random.default_rng(14),
    )
    result = channel_rates(
        8.0, time, total[:, :, None], den, ids, ["exchange"], bootstrap=80, seed=14,
    )
    np.testing.assert_allclose(np.asarray(result["kappa_curve"])[:, 0], existing["curve"])
    assert result["kappa"][0] == pytest.approx(existing["kappa"])
    np.testing.assert_allclose(result["branching_fraction_estimates"], [1.0])


def test_flat_total_does_not_hide_unconverged_channel_branching():
    time = np.arange(80, dtype=float)
    first = np.linspace(0.1, 0.9, len(time))
    num = np.tile(np.column_stack((first, 1 - first))[None, :, :], (4, 1, 1))
    result = channel_rates(
        1.0, time, num, np.ones(4), [[0, 0], [0, 1], [1, 0], [1, 1]],
        ["P1", "P2"], plateau_fraction=0.5, bootstrap=40, seed=87,
    )
    assert result["total_kappa"] == pytest.approx(1.0)
    assert result["channel_rates_au"] is None
    assert result["branching_fractions"] is None


def test_negative_net_channel_is_retained_as_diagnostic_not_clipped():
    time = np.arange(80, dtype=float)
    # Two parent pairs can give (-1,+1)+(0,+1), with denominator 2.
    num = np.tile(np.array([-1.0, 2.0]), (4, len(time), 1))
    result = channel_rates(
        3.0, time, num, np.full(4, 2.0), [[0, 0], [0, 1], [1, 0], [1, 1]],
        ["P1", "P2"], bootstrap=40, seed=99,
    )
    np.testing.assert_allclose(result["kappa"], [-0.5, 1.0])
    np.testing.assert_allclose(result["channel_rate_estimates_au"], [-1.5, 3.0])
    np.testing.assert_allclose(result["branching_fraction_estimates"], [-1.0, 2.0])
    assert result["channel_rates_au"] is None
    assert result["branching_fractions"] is None


def test_zero_net_reaction_has_no_defined_branching_fraction():
    time = np.arange(80, dtype=float)
    result = channel_rates(
        3.0, time, np.zeros((4, len(time), 2)), np.ones(4),
        [[0, 0], [0, 1], [1, 0], [1, 1]], ["P1", "P2"], bootstrap=40,
    )
    assert result["total_kappa"] == 0
    assert result["branching_fractions"] is None
    assert result["channel_rates_au"] is None


def test_duplicate_parent_identity_is_not_extra_sampling():
    time, num, den, ids = _signed_parents()
    ids[1] = ids[0]
    with pytest.raises(ValueError, match="duplicate|identit|unique"):
        channel_rates(1.0, time, num, den, ids, ["P1", "P2"])


def test_nonmonotone_time_axis_is_rejected():
    time, num, den, ids = _signed_parents()
    time[20] = time[19]
    with pytest.raises(ValueError, match="time|increas"):
        channel_rates(1.0, time, num, den, ids, ["P1", "P2"])


def test_surface_and_flux_source_identities_cannot_be_mixed():
    time, num, den, ids = _signed_parents()
    with pytest.raises(ValueError, match="identit|source|surface"):
        channel_rates(
            1.0, time, num, den, ids, ["P1", "P2"],
            source_identity="PES-A:300K:surface-1", flux_identity="PES-B:300K:surface-1",
        )


def test_bootstrap_copies_of_one_parent_do_not_establish_sampling_uncertainty():
    time = np.arange(80, dtype=float)
    numerator = np.tile(np.array([0.3, 0.2]), (1, len(time), 1))
    result = channel_rates(
        7.0, time, numerator, [1.0], [[0, 0]], ["P1", "P2"],
        bootstrap=250, seed=18,
    )
    assert result["rate_uncertainty"]["bootstrap_valid"] == 250
    assert result["rate_uncertainty"]["available"] is False
    assert result["kappa_standard_error"] is None
    assert result["channel_rate_standard_error_au"] is None
    assert result["branching_fraction_standard_error"] is None
    assert result["converged"] is False


def test_zero_flux_parent_chains_do_not_supply_independent_information():
    time = np.arange(80, dtype=float)
    numerator = np.zeros((8, len(time), 2))
    numerator[0, :, :] = [0.3, 0.2]
    denominator = np.r_[1.0, np.zeros(7)]
    ids = [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3]]
    result = channel_rates(
        7.0, time, numerator, denominator, ids, ["P1", "P2"],
        bootstrap=100, seed=18,
    )
    assert result["independent_parent_chains"] == 1
    assert result["rate_uncertainty"]["available"] is False
    assert result["channel_rate_standard_error_au"] is None


def test_duplicated_product_membership_cannot_make_total_kappa_exceed_one():
    time = np.arange(80, dtype=float)
    # Assigning the same forward child to two overlapping product labels
    # produces (1,1) with weight 1; it is not a valid product partition.
    with pytest.raises(ValueError, match="exclusive|denominator|exceed"):
        channel_rates(
            1.0, time, np.ones((2, len(time), 2)), [1.0, 1.0],
            [[0, 0], [1, 0]], ["P1", "P2"],
        )


def test_static_pmf_uncertainty_is_not_claimed_from_child_resampling():
    time, num, den, ids = _signed_parents()
    result = channel_rates(8.0, time, num, den, ids, ["P1", "P2"], bootstrap=80, seed=14)
    assert result["rate_uncertainty"]["static_uncertainty_included"] is False
    assert "held fixed" in result["rate_uncertainty"]["conditional_quantity"]
