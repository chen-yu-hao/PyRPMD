"""Extra time observables alter uncertainty blocks, never MBAR point estimates."""
from __future__ import annotations

import numpy as np
import pytest
from pyrpmd.analysis import fes


@pytest.fixture(scope="module")
def stationary_problem():
    rng = np.random.default_rng(51918)
    centers = np.array([[-.9], [0.], [.9]])
    counts = [4096, 3584, 4608]
    stiffness, beta, physical_curvature = np.array([[2.]]), 1.3, .7
    samples, hidden = [], []
    for count, center in zip(counts, centers[:,0]):
        mean = 2*center/(2+physical_curvature)
        sigma = np.sqrt(1/(beta*(2+physical_curvature)))
        samples.append(rng.normal(mean, sigma, size=(count,1)))
        # Stationary AR(1) from the first frame, independent of the biased CV.
        # Its marginal is N(0,1), but its timescale is deliberately much slower.
        rho = .98
        z = rng.normal(size=count)
        for i in range(1,count):
            z[i] = rho*z[i-1] + np.sqrt(1-rho*rho)*z[i]
        hidden.append(np.column_stack([z, rng.normal(size=count)]))
    kwargs = dict(samples=samples, centers=centers, stiffness=stiffness, beta=beta,
                  bins=24, ranges=[[-2.,2.]], bootstrap=0, min_count=10)
    return kwargs, hidden


def test_slow_stationary_hidden_modes_enlarge_blocks_but_not_point_weights(stationary_problem):
    kwargs, hidden = stationary_problem
    original = fes.reconstruct_fes(**kwargs, return_reweighting=True)
    result = fes.reconstruct_fes(**kwargs, correlation_observables=hidden, return_reweighting=True)
    for key in ("weights", "free_energy", "probability_mass", "density", "raw_count", "mask",
                "window_free_energies", "overlap", "components"):
        np.testing.assert_array_equal(result[key], original[key])
    assert result["_reweighting"].signature == original["_reweighting"].signature
    assert result["connected"] == original["connected"] is True
    expected_hidden = np.array([fes.statistical_inefficiency(a) for a in hidden])
    np.testing.assert_array_equal(result["cv_statistical_inefficiency"], original["statistical_inefficiency"])
    np.testing.assert_array_equal(result["additional_statistical_inefficiency"], expected_hidden)
    np.testing.assert_array_equal(result["statistical_inefficiency"], np.maximum(expected_hidden, original["statistical_inefficiency"]))
    assert np.all(result["statistical_inefficiency"] > 10*original["statistical_inefficiency"])
    assert np.all(result["bootstrap_block_plan"]["block_lengths"] > original["bootstrap_block_plan"]["block_lengths"])
    np.testing.assert_array_equal(result["effective_samples_per_stream"],
                                  np.array([len(a) for a in kwargs["samples"]])/result["statistical_inefficiency"])


def test_explicit_none_preserves_original_output_contract(stationary_problem):
    kwargs, _ = stationary_problem
    original = fes.reconstruct_fes(**kwargs)
    explicit = fes.reconstruct_fes(**kwargs, correlation_observables=None)
    assert explicit.keys() == original.keys()
    assert "additional_statistical_inefficiency" not in original
    for key in ("weights", "free_energy", "standard_error", "statistical_inefficiency"):
        np.testing.assert_array_equal(explicit[key], original[key])
    np.testing.assert_array_equal(explicit["bootstrap_block_plan"]["block_lengths"],
                                  original["bootstrap_block_plan"]["block_lengths"])


def test_extra_observations_cannot_shrink_cv_correlation_blocks(stationary_problem):
    kwargs, _ = stationary_problem
    original = fes.reconstruct_fes(**kwargs)
    # Supplying a copy of the original CV gives exactly the same g, including
    # any finite-sample fluctuation in this IID stationary test trajectory.
    result = fes.reconstruct_fes(**kwargs, correlation_observables=kwargs["samples"])
    np.testing.assert_array_equal(result["statistical_inefficiency"], original["statistical_inefficiency"])
    np.testing.assert_array_equal(result["bootstrap_block_plan"]["block_lengths"],
                                  original["bootstrap_block_plan"]["block_lengths"])


def test_actual_bootstrap_resamples_using_enlarged_hidden_blocks(stationary_problem, monkeypatch):
    kwargs, hidden = stationary_problem
    recorded = []
    original_resampler = fes._block_resample
    def recording_resampler(x, block, rng):
        recorded.append(block)
        return original_resampler(x, block, rng)
    monkeypatch.setattr(fes, "_block_resample", recording_resampler)
    config = {**kwargs, "bootstrap": 3, "seed": 19}
    result = fes.reconstruct_fes(**config, correlation_observables=hidden)
    expected = result["bootstrap_block_plan"]["block_lengths"].tolist()*3
    assert recorded == expected
    assert result["bootstrap_attempted"] == 3
    assert np.isfinite(result["standard_error"][~result["mask"]]).any()


def test_constant_hidden_coordinate_marks_unresolved_uncertainty(stationary_problem):
    kwargs, _ = stationary_problem
    constant = [np.ones((len(a),1)) for a in kwargs["samples"]]
    result = fes.reconstruct_fes(**{**kwargs, "bootstrap": 2}, correlation_observables=constant)
    original = fes.reconstruct_fes(**kwargs)
    counts = np.array([len(a) for a in kwargs["samples"]])
    np.testing.assert_array_equal(result["additional_statistical_inefficiency"], counts)
    np.testing.assert_array_equal(result["weights"], original["weights"])
    np.testing.assert_array_equal(result["free_energy"], original["free_energy"])
    assert result["bootstrap_block_plan"]["status"] == "unavailable"
    assert result["bootstrap_block_plan"]["block_length_capped"].all()
    assert np.isnan(result["standard_error"]).all()


@pytest.mark.parametrize("bad", [
    3., [], [np.ones((8,1))],
    [np.ones((7,1)),np.ones((8,1))],
    [np.ones(8),np.ones(8)],
    [np.ones((8,0)),np.ones((8,0))],
    [np.ones((8,1,1)),np.ones((8,1,1))],
    [np.ones((8,1)),np.ones((8,2))],
    [np.full((8,1),np.nan),np.ones((8,1))],
    [np.ones((8,1)),np.full((8,1),np.inf)],
    [np.ones((8,1),complex),np.ones((8,1))],
    [np.ones((8,1),bool),np.ones((8,1))],
    [np.full((8,1),"1"),np.ones((8,1))],
])
def test_invalid_extra_streams_fail_before_mbar(bad, monkeypatch):
    def no_mbar(*args, **kwargs):
        raise AssertionError("invalid extra observables must be rejected before MBAR")
    monkeypatch.setattr(fes, "solve_mbar", no_mbar)
    samples = [np.linspace(-1,1,8)[:,None],np.linspace(-.5,1.5,8)[:,None]]
    with pytest.raises(ValueError, match="correlation_observables"):
        fes.reconstruct_fes(samples, [[0.],[.5]], [[1.]], 1., correlation_observables=bad)
