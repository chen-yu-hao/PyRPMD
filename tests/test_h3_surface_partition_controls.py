"""Independent normalization and independent-group uncertainty for controls."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import t as student_t, ttest_ind

from benchmarks.h3.surface_partition import reference_quadrature
from benchmarks.h3.surface_partition_controls import (
    independent_difference, matching_reference, qtst_offset,
)


class FlatPotential:
    def evaluate(self, q):
        return SimpleNamespace(energy=np.zeros(len(q)), status=np.zeros(len(q), dtype=int))


def test_domain_qtst_comparison_keeps_independent_classical_shape_normalization():
    # P=1: quantum/classical corrections are exactly one, yet changing the
    # radial domain changes the TS and reactant volumes by different powers.
    offsets = []
    maximum = 2.
    for minimum in (.5, .8):
        ts = reference_quadrature(FlatPotential(), "ts", beta=2., minimum=minimum,
                                  maximum=maximum, radial_order=32, angular_order=32)
        reactant = reference_quadrature(FlatPotential(), "reactant", beta=2., minimum=minimum,
                                        maximum=maximum, radial_order=32, angular_order=16)
        offsets.append(qtst_offset(ts.log_classical, reactant.log_classical, 0.))
    exact_ratio = ((maximum**5-.8**5)/(maximum**3-.8**3)
                   / ((maximum**5-.5**5)/(maximum**3-.5**3)))
    assert np.exp(offsets[1]-offsets[0]) == pytest.approx(exact_ratio, rel=3e-14)
    assert exact_ratio != pytest.approx(1., abs=.01)


def test_independent_control_uncertainty_matches_scipy_raw_data_welch_test():
    first = np.array([.1, .2, .4, .3])
    second = np.array([-.1, 0., .1, .05, .08, .12, -.02, .04])
    result = independent_difference(first, second)
    independent = ttest_ind(first, second, equal_var=False)
    statistic = result["mean_log_difference"]/result["standard_error"]
    assert statistic == pytest.approx(independent.statistic, rel=1e-14)
    p = 2*student_t.sf(abs(statistic), result["Welch_degrees_of_freedom"])
    assert p == pytest.approx(independent.pvalue, rel=2e-14)
    lower, upper = result["CI95_log_difference"]
    assert (not lower <= 0 <= upper) == bool(independent.pvalue < .05)
    assert not result["CI95_wholly_within_margin"]


def test_equivalence_requires_entire_interval_inside_predeclared_margin():
    baseline = np.array([-.003, -.002, -.001, 0., .001, .002, .003, .004])
    control = np.array([.001, .002, .003, .004])
    assert independent_difference(control, baseline)["CI95_wholly_within_margin"]
    # A point inside 5% is insufficient if its independent-seed interval is wide.
    wide = np.array([-.06, -.02, .04, .08])
    result = independent_difference(wide, baseline)
    assert abs(result["mean_log_difference"]) < np.log(1.05)
    assert not result["CI95_wholly_within_margin"]


def test_reactant_oracle_selection_binds_beads_and_domain():
    selected = {"beads": 128, "domain_bohr": [.6, 8.], "log_quantum_classical_ratio": -7.}
    data = {"bead_refinement": [{"beads": 64, "domain_bohr": [.5, 8.]}],
            "wall_stress_checks": [selected]}
    assert matching_reference(data, 128, .6) is selected
    with pytest.raises(ValueError, match="exactly one"):
        matching_reference(data, 128, .5)
    data["wall_stress_checks"].append(dict(selected))
    with pytest.raises(ValueError, match="exactly one"):
        matching_reference(data, 128, .6)
