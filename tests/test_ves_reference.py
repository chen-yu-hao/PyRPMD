"""Stochastic VES learning against independent integrals and umbrella/MBAR.

The sampler draws exact continuous IID configurations for each *current* VES
bias using rejection; the optimizer receives sampled observations, never exact
moments or the true coefficients.  This deliberately isolates VES optimization
from MD mixing.  The companion example records actual MD workflow evidence.
"""

import json

import numpy as np
import pytest

from examples.ves_validation import _ValidationPotential, reference_case
from pyrpmd.io import write_report_json


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("target", ["uniform", "well_tempered"])
def test_stochastic_ves_matches_quadrature_and_independent_umbrella_mbar(dimension, target):
    result = reference_case(dimension, target)
    assert result["quadrature_order_8_vs_5_max_hartree"] < 1e-9
    assert result["connected"]
    assert result["supported_bins"] >= (19 if dimension == 1 else 90)
    assert result["samples_per_replica"] == 19200
    assert all(state["observations"] == 19200 for state in result["states"])
    assert all(state["updates"] == 300 for state in result["states"])
    # The declared error goal is 0.1 kBT and at least a fivefold decrease from
    # the zero-bias estimate.  It prevents a broad confidence band from hiding
    # an optimizer that has learned no useful surface.
    assert result["ves_quadrature_rmse_hartree"] < .1/result["beta"]
    assert result["learning_rmse_ratio"] < .2
    # A 99% Bonferroni Student-t band accounts for simultaneous comparisons
    # and only six independently optimized histories (five degrees of freedom).
    # This regression diagnostic does not establish physical convergence.
    band = result["bonferroni_student_t_multiplier"]
    assert result["ves_quadrature_max_sem"] < band
    assert result["ves_mbar_max_combined_sem"] < band
    assert result["mbar_quadrature_max_sem"] < 5


@pytest.mark.parametrize("dimension", [1, 2])
def test_validation_md_potential_force_matches_independent_finite_differences(dimension):
    provider = _ValidationPotential(dimension)
    q = np.array([[[-.71, .33, .48]], [[.39, -.64, -.17]]])
    analytic = provider.evaluate(q)
    step = 1e-6
    for atom in range(q.shape[1]):
        for axis in range(3):
            plus, minus = q.copy(), q.copy()
            plus[:, atom, axis] += step
            minus[:, atom, axis] -= step
            force = -(provider.evaluate(plus).energy - provider.evaluate(minus).energy)/(2*step)
            np.testing.assert_allclose(force, analytic.forces[:, atom, axis], atol=2e-10)


@pytest.mark.parametrize("shape", [(2, 2, 3), (1, 0, 3), (1, 1, 2)])
def test_validation_potential_rejects_non_single_atom_systems(shape):
    with pytest.raises(ValueError, match="one atom"):
        _ValidationPotential(2).evaluate(np.zeros(shape))


def test_validation_report_contains_portable_grid_and_gauge_metadata(tmp_path):
    # A short, deliberately unconverged run checks report portability, not
    # physical accuracy.  The sampling acceptance test above has its own budget.
    result = reference_case(1, "uniform", iterations=2, replicas=2,
                            samples_per_window=1500, bootstrap=4)
    path = tmp_path / "report.json"
    write_report_json(path, result)
    saved = json.loads(path.read_text())
    edge, = saved["edges"]
    assert len(edge) == len(saved["ves_mean"]) + 1
    assert edge[0] == -1.5 and edge[-1] == 1.5
    assert np.all(np.diff(edge) > 0)
    reference, = saved["reference_bin"]
    assert saved["reference_free_energy"][reference] == 0.0
    assert saved["ves_mean"][reference] == 0.0
    assert saved["ves_sem"][reference] == 0.0
