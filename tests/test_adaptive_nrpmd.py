"""Independent Gaussian reweighting and curved-valley path integration checks."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import ndtr, ndtri

from pyrpmd.adaptive_nrpmd import (
    AdaptivePathOptions, reweight_thermal_umbrellas, run_adaptive_nrpmd_path,
)
from pyrpmd.bias import UmbrellaBias
from pyrpmd.cv import Cartesian, CVStack
from pyrpmd.nonadiabatic import DiabaticResult, NRPMDState
from pyrpmd.thermal_nrpmd import (
    ThermalEnsemble, ThermalNRPMDDynamics, _static_umbrella_identity,
    thermal_flux_side,
)


class GaussianMatrix:
    """A confining matrix model with an analytically known scalar marginal."""

    def __init__(self, spring=0.8, mean=0.35, states=1):
        self.spring, self.mean, self.states = spring, mean, states

    def evaluate(self, q):
        delta = q.copy()
        delta[:, 0, 0] -= self.mean
        energy = 0.5 * self.spring * np.sum(delta**2, axis=(1, 2))
        identity = np.eye(self.states)
        return DiabaticResult(
            energy[:, None, None] * identity,
            self.spring * delta[:, None, None] * identity[None, :, :, None, None],
        )


def gaussian_windows(*, centers=(-1.0, 0.25, 1.5), stiffness=1.3, beta=2.0,
                     count=1000, seed=371, signed=False, paired_signs=False):
    """Draw known restrained Gaussians independently of ThermalNRPMDSampler.

    Shuffled Gaussian quantiles reduce quadrature error without introducing a
    sorted trajectory. For one state the mapping marginals x² exp(-x²) and
    Maxwell momenta are also sampled analytically. The optional two-state
    signed fixtures deliberately define a synthetic signed measure to test
    reweighting algebra; they are not claimed to sample a particular SEO model.
    """
    states = 2 if signed or paired_signs else 1
    potential = GaussianMatrix(states=states)
    cv = Cartesian(0)
    rng = np.random.default_rng(seed)
    centers = np.asarray(centers, dtype=float)[:, None]
    windows = []
    for center in centers:
        bias = UmbrellaBias(cv, center, stiffness)
        mean = (potential.spring * potential.mean + stiffness * center[0]) / (potential.spring + stiffness)
        sigma = np.sqrt(1 / (beta * (potential.spring + stiffness)))
        quantiles = ndtri((np.arange(count) + 0.5) / count)
        rng.shuffle(quantiles)
        x_values = mean + sigma * quantiles
        if paired_signs:
            x_values = np.repeat(x_values, 2)
        parents, signs, populations = [], [], []
        for i, value in enumerate(x_values):
            q = rng.normal(scale=np.sqrt(1 / (beta * potential.spring)), size=(1, 1, 3))
            q[0, 0, 0] = value
            p = rng.normal(scale=np.sqrt(1.7 / beta), size=q.shape)
            x = np.sqrt(rng.gamma(1.5, size=(1, states))) * rng.choice([-1, 1], (1, states))
            y = np.sqrt(rng.gamma(1.5, size=(1, states))) * rng.choice([-1, 1], (1, states))
            parents.append(NRPMDState(q, p, mapping_q=x, mapping_p=y))
            signs.append((1 if i % 2 else -1) if paired_signs else (-1 if signed and value < -0.6 else 1))
            populations.append([1.2, -0.2] if states == 2 else [1.0])
        windows.append(ThermalEnsemble(
            tuple(parents), signs, populations, [1.7], beta, potential,
            np.arange(len(parents), dtype=int),
            {"distribution": "absolute_SEO_projected_trace_with_static_umbrella",
             "static_bias": True, "static_bias_identity": _static_umbrella_identity(bias),
             "source": "independent_analytic_gaussian_fixture"},
        ))
    return tuple(windows), cv, centers, stiffness


@pytest.fixture(scope="module")
def gaussian_data():
    return gaussian_windows()


def test_mbar_recovers_analytic_gaussian_moments_and_window_free_energies(gaussian_data):
    windows, cv, centers, stiffness = gaussian_data
    result = reweight_thermal_umbrellas(*gaussian_data)
    assert result.unavailable_reasons == []
    mean = windows[0].potential.mean
    spring = windows[0].potential.spring
    beta = windows[0].beta
    values = result.cv_values[:, 0]
    assert result.expectation(values) == pytest.approx(mean, abs=0.007)
    assert result.expectation((values - mean)**2) == pytest.approx(1 / (beta * spring), rel=0.025)
    # Analytic reduced partition-function ratios of restrained Gaussians.
    expected_f = beta * spring * stiffness * (centers[:, 0] - mean)**2 / (2 * (spring + stiffness))
    expected_f -= expected_f[0]
    np.testing.assert_allclose(
        result.diagnostics["absolute_measure_window_free_energies"], expected_f, atol=0.002,
    )
    assert result.average_sign == pytest.approx(1.0)
    assert result.summary()["physical_converged"] is False
    assert result.diagnostics["sampling_time_is_physical_time"] is False
    assert result.weights.sum() == pytest.approx(1.0)
    assert np.all(result.weights >= 0)


def test_signed_reweighting_matches_analytic_signed_gaussian_and_retains_estimators():
    data = gaussian_windows(count=1400, signed=True)
    result = reweight_thermal_umbrellas(*data)
    assert result.unavailable_reasons == []
    mean = data[0][0].potential.mean
    sigma = np.sqrt(1 / (data[0][0].beta * data[0][0].potential.spring))
    z = (-0.6 - mean) / sigma
    probability = ndtr(z)
    expected_sign = 1 - 2 * probability
    expected_mean = mean + 2 * sigma * np.exp(-0.5 * z*z) / np.sqrt(2*np.pi) / expected_sign
    assert result.average_sign == pytest.approx(expected_sign, abs=0.006)
    assert result.expectation(result.cv_values[:, 0]) == pytest.approx(expected_mean, abs=0.009)
    np.testing.assert_allclose(result.expectation(result.ensemble.electronic_populations), [1.2, -0.2], atol=1e-12)
    assert np.any(result.ensemble.signs < 0)
    assert np.any(result.ensemble.electronic_populations < 0)
    # Positive importance weights and signed estimators are separate arrays.
    assert np.all(result.weights >= 0)
    parents = result.resample_parents(200, seed=7)
    indices = parents.sampling_diagnostics["source_flat_indices"]
    assert np.any(parents.signs < 0)
    np.testing.assert_array_equal(parents.signs, result.ensemble.signs[indices])
    np.testing.assert_array_equal(parents.electronic_populations,
                                  result.ensemble.electronic_populations[indices])


@pytest.mark.parametrize("mismatch", ["beta", "provider", "masses", "bias", "cv", "stiffness", "unbiased", "missing_identity"])
def test_reweighting_rejects_incompatible_distributions_and_frozen_bias_identity(gaussian_data, mismatch):
    windows, cv, centers, stiffness = gaussian_data
    windows = list(windows)
    centers = centers.copy()
    if mismatch == "beta":
        windows[1] = replace(windows[1], beta=windows[1].beta + 0.1)
    elif mismatch == "provider":
        windows[1] = replace(windows[1], potential=GaussianMatrix())
    elif mismatch == "masses":
        windows[1] = replace(windows[1], masses=[1.8])
    elif mismatch == "bias":
        centers[1, 0] += 0.1
    elif mismatch == "cv":
        cv = Cartesian(0, "y")
    elif mismatch == "stiffness":
        stiffness += 0.1
    else:
        diagnostics = dict(windows[1].sampling_diagnostics)
        if mismatch == "unbiased":
            diagnostics["static_bias"] = False
        else:
            del diagnostics["static_bias_identity"]
        windows[1] = replace(windows[1], sampling_diagnostics=diagnostics)
    with pytest.raises(ValueError, match="share potential|frozen umbrella/CV identity"):
        reweight_thermal_umbrellas(windows, cv, centers, stiffness)


def test_disconnected_windows_are_unavailable_instead_of_an_absolute_free_energy():
    data = gaussian_windows(centers=(-8.0, 8.0), stiffness=100.0, count=120)
    result = reweight_thermal_umbrellas(*data, min_effective_samples=1)
    assert "disconnected_umbrella_overlap" in result.unavailable_reasons
    assert result.summary()["available"] is False
    with pytest.raises(ValueError, match="disconnected_umbrella_overlap"):
        result.expectation(result.cv_values[:, 0])
    with pytest.raises(ValueError, match="disconnected_umbrella_overlap"):
        result.resample_parents(2)


def test_unresolved_average_sign_blocks_observables_and_parent_resampling():
    data = gaussian_windows(count=120, paired_signs=True)
    result = reweight_thermal_umbrellas(*data)
    assert abs(result.average_sign) < 1e-14
    assert "average_sign_below_threshold" in result.unavailable_reasons
    assert "average_sign_not_resolved_by_diagnostic" in result.unavailable_reasons
    with pytest.raises(ValueError, match="average_sign"):
        result.expectation(result.cv_values[:, 0])
    with pytest.raises(ValueError, match="average_sign"):
        result.resample_parents(2)


def test_resampling_keeps_complete_phase_space_signs_estimators_and_ancestry(gaussian_data):
    result = reweight_thermal_umbrellas(*gaussian_data)
    parents = result.resample_parents(64, seed=91)
    again = result.resample_parents(64, seed=91)
    diagnostics = parents.sampling_diagnostics
    assert diagnostics["static_bias"] is False
    assert diagnostics["source_static_bias"] is True
    assert diagnostics["parent_resampling"] == "mbar_multinomial"
    assert diagnostics["uncertainty_requires_source_chain_bootstrap"] is True
    assert diagnostics["source_flat_indices"] == again.sampling_diagnostics["source_flat_indices"]
    assert diagnostics["unique_source_parents"] == len(set(diagnostics["source_flat_indices"]))
    for index, source_index in enumerate(diagnostics["source_flat_indices"]):
        parent, source = parents.states[index], result.ensemble.states[source_index]
        for key in ("q", "p", "mapping_q", "mapping_p"):
            np.testing.assert_array_equal(getattr(parent, key), getattr(source, key))
            assert not np.shares_memory(getattr(parent, key), getattr(source, key))
        assert (parent.step, parent.time) == (source.step, source.time)
        assert parents.signs[index] == result.ensemble.signs[source_index]
        assert parents.sweeps[index] == result.ensemble.sweeps[source_index]
        np.testing.assert_array_equal(parents.electronic_populations[index], result.ensemble.electronic_populations[source_index])
        assert diagnostics["source_window_indices"][index] == result.window_indices[source_index]
        assert diagnostics["source_sample_indices"][index] == result.sample_indices[source_index]
    parents.states[0].q[:] = 999
    assert np.max(np.abs(result.ensemble.states[diagnostics["source_flat_indices"][0]].q)) < 10


def test_only_reweighted_parents_enter_unbiased_correlations_and_keep_uncertainty_guard(gaussian_data):
    result = reweight_thermal_umbrellas(*gaussian_data)
    dynamics = ThermalNRPMDDynamics(
        result.ensemble.potential, result.ensemble.masses, result.ensemble.beta,
        1, 0.01, 1, native=False,
    )

    def surface(centroid):
        gradient = np.zeros_like(centroid)
        gradient[0, 0] = 1
        return centroid[0, 0], gradient

    with pytest.raises(ValueError, match="must be reweighted"):
        thermal_flux_side(result.ensemble, dynamics, surface, steps=3)
    parents = result.resample_parents(32, seed=17)
    correlation = thermal_flux_side(parents, dynamics, surface, steps=3)
    summary = correlation.summarize(start=0.0, stop=0.03, block_size=4)
    assert summary["rate_curve_available"] is True
    assert summary["standard_error"] is None
    assert summary["plateau_detected"] is False
    assert summary["uncertainty_reason"] == "source_chain_bootstrap_required_after_reweighting"


class CurvedValley:
    """U=.15 x²+2.5[y-.8(1-x²)]²+.5 z² has an upward curved valley."""

    def evaluate(self, q):
        x, y, z = q[:, 0, 0], q[:, 0, 1], q[:, 0, 2]
        displacement = y - 0.8 * (1 - x*x)
        energy = 0.15*x*x + 2.5*displacement**2 + 0.5*z*z
        gradient = np.zeros_like(q)
        gradient[:, 0, 0] = 0.3*x + 8.0*x*displacement
        gradient[:, 0, 1] = 5.0*displacement
        gradient[:, 0, 2] = z
        return DiabaticResult(energy[:, None, None], gradient[:, None, None])


def test_adaptive_path_moves_into_curved_valley_freezes_endpoints_and_uses_fresh_production():
    images = np.zeros((3, 2, 1, 3))
    images[:, :, 0, 0] = np.array([-1.0, 0.0, 1.0])[:, None]
    original = images.copy()
    cv = CVStack([Cartesian(0, "x"), Cartesian(0, "y")])
    options = AdaptivePathOptions(
        max_iterations=2, pilot_samples=140, pilot_burn_in=60,
        production_samples=100, production_burn_in=60, production_stride=1,
        relaxation=0.8, max_step=0.3, path_tolerance=1e-8,
        min_effective_samples=1.0, nuclear_step=0.35, centroid_step=0.4,
    )
    potential = CurvedValley()
    first = run_adaptive_nrpmd_path(potential, [1.7], 2.0, images, 1, cv, 4.0, options=options, seed=192)
    second = run_adaptive_nrpmd_path(potential, [1.7], 2.0, images, 1, cv, 4.0, options=options, seed=192)
    np.testing.assert_array_equal(images, original)
    np.testing.assert_array_equal(first.centers[[0, -1]], [[-1.0, 0.0], [1.0, 0.0]])
    assert first.centers[1, 1] > 0.05
    np.testing.assert_array_equal(first.centers, second.centers)
    np.testing.assert_array_equal(first.reweighted.weights, second.reweighted.weights)
    assert first.history == second.history
    assert len(first.history) == options.max_iterations
    assert len(first.reweighted.ensemble.states) == len(images) * options.production_samples
    assert first.reweighted.diagnostics["samples_per_window"] == [options.production_samples] * len(images)
    assert np.min(first.reweighted.ensemble.sweeps) == options.production_burn_in + options.production_stride
    assert np.max(first.reweighted.ensemble.sweeps) == options.production_burn_in + options.production_samples * options.production_stride
    assert all(parent.time == 0 and parent.step == 0 for parent in first.reweighted.ensemble.states)
    summary = first.summary()
    assert summary["production_path_frozen"] is True
    assert summary["pilot_samples_used_in_MBAR"] is False
    assert summary["sampling_time_is_physical_time"] is False
    assert summary["physical_converged"] is False
    assert not first.centers.flags.writeable


@pytest.mark.parametrize("field,value", [("max_iterations", 0), ("pilot_samples", True),
                                         ("production_stride", 0), ("relaxation", 1.1),
                                         ("overlap_threshold", 1.0), ("min_average_sign", 0.0)])
def test_adaptive_budgets_and_guards_are_validated(field, value):
    with pytest.raises(ValueError):
        AdaptivePathOptions(**{field: value})
