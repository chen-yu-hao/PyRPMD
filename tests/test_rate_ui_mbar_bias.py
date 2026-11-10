"""Regression for issue #170: a well-sampled Gaussian UI barrier can be biased.

The comparison reuses the UI workflow's *existing* histogram MBAR point fit.
No MBAR bootstrap, second solver, or physical rate convergence is assumed.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pyrpmd.constants import KB_AU
from pyrpmd.runner import prepare
from pyrpmd.workflows.rate_pmf import (
    _ui_mbar_point_comparison,
    _ui_rate_pmf,
)


BETA = 1.0 / (1500.0 * KB_AU)
UI_BARRIER = 0.028559
MBAR_BARRIER = 0.027788


def _histogram(barrier=MBAR_BARRIER, bins=240):
    edges = np.linspace(-0.1, 1.1, bins + 1)
    grid = 0.5 * (edges[1:] + edges[:-1])
    return {
        "edges": [edges],
        "free_energy": barrier * grid,
        "mask": np.zeros(bins, dtype=bool),
        "raw_count": np.full(bins, 100),
        "connected": True,
        "statistical_inefficiency": [1., 1.],
        "effective_samples_per_stream": [50., 50.],
        "weight_ess": 100.,
    }


def _compare(histogram=None, *, ui=UI_BARRIER, stderr=0.000348, surface=1.0):
    return _ui_mbar_point_comparison(
        _histogram() if histogram is None else histogram,
        reference=0.0, surface=surface, ui_barrier=ui,
        beta=BETA, standard_error=stderr,
        stiffness=np.array([0.24, 0.48]),
    )


def test_soft_window_exchange_regression_detects_ui_rate_underestimate():
    # D+H2 1500 K figures from #170: the UI bootstrap error misses the bias.
    check = _compare()
    assert check["ready"]
    assert check["method"] == "existing_histogram_mbar_point_estimate"
    assert check["bootstrap_draws"] == 0
    assert check["mbar_minus_ui_hartree"] == pytest.approx(-0.000771)
    assert check["status"] == "discrepant"
    assert check["threshold_hartree"] == pytest.approx(
        max(0.000348, 0.1 / BETA)
    )
    assert check["ui_over_mbar_qtst_ratio"] == pytest.approx(
        np.exp(-BETA * 0.000771)
    )


def test_agreement_does_not_emit_a_spurious_model_bias_flag():
    check = _compare(ui=MBAR_BARRIER + 0.00001)
    assert check["status"] == "consistent"
    assert check["ready"]


def test_masked_path_and_outside_endpoint_cannot_be_extrapolated():
    histogram = _histogram()
    histogram["mask"][120] = True
    check = _compare(histogram)
    assert check["status"] == "unavailable"
    assert check["reason"] == "unsupported_histogram_path"
    assert check["mbar_barrier_hartree"] is None
    check = _compare(_histogram(), surface=1.2)
    assert check["reason"] == "endpoint_outside_histogram_centers"


def test_coarse_histogram_is_not_treated_as_independent_evidence():
    check = _compare(_histogram(bins=10))
    assert check["status"] == "unavailable"
    assert check["reason"] == "histogram_bin_wider_than_half_thermal_width"


def test_ui_rate_analysis_exposes_actionable_discrepancy_and_window_shapes():
    samples = [
        np.linspace(-0.08, 0.2, 50),
        np.linspace(0.85, 1.12, 50),
    ]
    centers = np.array([0.0, 1.0])
    stiffness = np.array([0.24, 0.48])
    options = {
        "points": 401, "xi_star": 1.0, "search_range": None,
        "bootstrap": 0, "analysis_seed": 5,
    }

    def fitted_ui(_samples, _centers, _stiffness, _beta, grid):
        return UI_BARRIER * grid, np.full_like(grid, UI_BARRIER)

    with patch("pyrpmd.workflows.rate_pmf.umbrella_integration", fitted_ui):
        estimate = _ui_rate_pmf(
            {"data": _histogram()}, {"beta": BETA},
            centers, samples, stiffness, options,
        )
    assert estimate["ready"]  # Warning rather than a silent rate-withholding policy.
    assert estimate["ui_mbar_comparison"]["status"] == "discrepant"
    assert any("Gaussian UI/MBAR point-estimate barrier disagreement" in w
               and "0.028559" in w and "0.027788" in w
               for w in estimate["warnings"])
    assert len(estimate["window_moments"]) == 2
    assert estimate["window_moments"][0]["thermal_sigma_xi"] == pytest.approx(
        np.sqrt(1 / (BETA * stiffness[0]))
    )
    assert estimate["window_moments"][0]["skewness"] == pytest.approx(0, abs=1e-12)


def test_prepare_exposes_width_and_nearest_center_spacing():
    root = Path(__file__).resolve().parents[1]
    report = prepare(root / "examples/rate/research.inp", numerical=False)
    window_plan = report["umbrella_preparation"]["window_plan"]
    entrance = next(w for w in window_plan if w["source"] == "entrance")
    barrier = next(w for w in window_plan if w["source"] == "barrier")
    assert entrance["thermal_sigma_xi"] == pytest.approx(
        np.sqrt(KB_AU / 0.0016)
    )
    assert barrier["thermal_sigma_xi"] == pytest.approx(
        np.sqrt(KB_AU / 0.0032)
    )
    assert entrance["nearest_center_spacing_xi"] == pytest.approx(0.02)
    assert barrier["nearest_center_spacing_xi"] == pytest.approx(0.01)


def test_implicit_rate_windows_use_narrow_gaussian_width(tmp_path):
    from pyrpmd.input import JobSpec
    from pyrpmd.workflows.rate_pmf import _prepare

    source = Path(__file__).resolve().parents[1] / 'examples/rate/minimal.inp'
    config, system = JobSpec.from_file(source).resolve()
    assert config['umbrella']['kappa'] * config['beta'] == pytest.approx(500.)
    stage = _prepare(config, system, tmp_path, _write=False)
    assert stage['beta'] * stage['umbrella']['kappa'] == pytest.approx(500.)


def test_historical_resolved_input_keeps_its_implicit_spring(tmp_path):
    from pyrpmd.input import JobSpec
    from pyrpmd.workflows.rate_pmf import _prepare

    source = Path(__file__).resolve().parents[1] / 'examples/rate/minimal.inp'
    config, system = JobSpec.from_file(source).resolve()
    config.pop('umbrella')  # historical resolved inputs did not persist this default
    stage = _prepare(config, system, tmp_path, _write=False)
    assert stage['beta'] * stage['umbrella']['kappa'] == pytest.approx(50.)
