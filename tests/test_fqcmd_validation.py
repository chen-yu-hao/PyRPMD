"""Independent statistical references for the gas-water validation harness."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def example():
    path = Path(__file__).parents[1] / "examples" / "spectroscopy" / "fqcmd_statistics.py"
    spec = importlib.util.spec_from_file_location("fqcmd_water_statistics_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_matches_exact_two_molecule_resampling_distribution(example):
    # Enumerating the four equiprobable bootstrap samples gives first moments
    # [3200, 3600, 3600, 4000] and peaks [3200, 3200, 3200, 4000].
    wave = np.array([2400., 3200., 4000., 4800.])
    ir = np.array([[0., 1., 0., 0.], [0., 0., 1., 0.]])
    result = example.stretch_statistics(wave, ir, samples=20000, seed=17)
    assert result["estimates"] == {"peak_grid_cm1": 3200., "band_first_moment_cm1": 3600.}
    exact_moment_sd = np.std([3200., 3600., 3600., 4000.])
    exact_peak_sd = np.std([3200., 3200., 3200., 4000.])
    assert result["standard_error_cm1"]["band_first_moment_cm1"] == pytest.approx(exact_moment_sd, rel=.03)
    assert result["standard_error_cm1"]["peak_grid_cm1"] == pytest.approx(exact_peak_sd, rel=.03)
    assert result["percentile_95_interval_cm1"]["band_first_moment_cm1"] == [3200., 4000.]
    assert result["conditional_on_fixed_potential_and_dynamics"] is True
    assert result["includes_training_uncertainty"] is False
    assert result["physical_converged"] is False


def test_statistics_use_mean_spectrum_instead_of_mean_of_molecular_moments(example):
    wave = np.array([2000., 3000., 4000., 5000.])
    ir = np.array([[0., 1., 0., 0.], [0., 0., 3., 0.]])
    result = example.stretch_statistics(wave, ir, seed=5)
    assert result["estimates"]["band_first_moment_cm1"] == 3750.
    assert result["estimates"]["peak_grid_cm1"] == 4000.
    assert result["per_molecule_band_first_moment_cm1"] == [3000., 4000.]
    assert result == example.stretch_statistics(wave, ir, seed=5)
    np.testing.assert_array_equal(ir, [[0., 1., 0., 0.], [0., 0., 3., 0.]])


def test_one_molecule_does_not_report_a_zero_error_bar(example):
    result = example.stretch_statistics([2500., 3500., 4500.], [[0., 1., 0.]])
    assert result["standard_error_cm1"] == {"peak_grid_cm1": None, "band_first_moment_cm1": None}
    assert result["percentile_95_interval_cm1"] == {"peak_grid_cm1": None, "band_first_moment_cm1": None}
    assert "only one" in result["uncertainty_status"]


@pytest.mark.parametrize("wave,ir", [
    ([2500., 2500., 4500.], [[0., 1., 0.]]),
    ([2500., 3500., np.nan], [[0., 1., 0.]]),
    ([2500., 3500., 4500.], [[0., -1., 0.]]),
    ([2500., 3500., 4500.], [[0., 0., 0.]]),
    ([2500., 3500., 4500.], [0., 1., 0.]),
    ([100., 200., 300.], [[0., 1., 0.]]),
])
def test_statistics_reject_undefined_data(example, wave, ir):
    with pytest.raises(ValueError):
        example.stretch_statistics(wave, ir)


@pytest.mark.parametrize("options", [{"samples": True}, {"samples": 1}, {"seed": -1}, {"seed": True}])
def test_statistics_validate_resampling_settings(example, options):
    with pytest.raises(ValueError):
        example.stretch_statistics([2500., 3500., 4500.], [[0., 1., 0.]], **options)
