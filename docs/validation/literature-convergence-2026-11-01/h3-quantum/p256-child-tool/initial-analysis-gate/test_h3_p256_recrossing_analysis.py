"""Independent statistical and finite-P identity checks; no trajectories run."""
from copy import deepcopy
import json

import numpy as np
import pytest
from scipy import stats

from benchmarks.h3 import p256_recrossing_analyze as analysis
from benchmarks.h3 import p256_surface_recrossing as producer


def test_independent_welch_interval_matches_scipy_unequal_variance_oracle():
    first = np.array([.02, .05, -.04, .01, .09, -.07, .03, .04])
    second = np.array([.005, -.003, .012, -.006, .009, -.010, .013, -.008])
    expected = stats.ttest_ind(first, second, equal_var=False).confidence_interval(.95)
    observed = analysis.independent_difference(first, second, np.log(1.05))
    np.testing.assert_allclose(observed['CI95'], [expected.low, expected.high], rtol=1e-13, atol=1e-15)
    assert observed['groups_paired_by_index'] is False
    assert 7 < observed['Welch_df'] < 14
    assert observed['CI95_wholly_within_margin'] is False


def test_group_reordering_does_not_artificially_pair_independent_seeds():
    a = np.array([1., 3., 7., 2., 8., 3., -2., 6.])/100
    b = np.array([3., -1., 2., -2., 4., 1., 5., 1.])/100
    first = analysis.independent_difference(a, b)
    second = analysis.independent_difference(a, b[::-1])
    np.testing.assert_allclose(first['CI95'], second['CI95'], atol=1e-16)


@pytest.mark.parametrize('shift,passes', [(0., True), (.04, True), (.06, False)])
def test_full_interval_equivalence_and_degenerate_observation_are_explicit(shift, passes):
    result = analysis.independent_difference(np.full(8, shift), np.zeros(8), np.log(1.05))
    assert result['degenerate_observed_variance'] is True
    assert result['CI95_wholly_within_margin'] is passes
    assert result['CI95'] == [shift, shift]


@pytest.mark.parametrize('a,b', [(np.ones(7), np.ones(8)), (np.ones(8), np.ones(9)),
                                (np.r_[np.ones(7), np.nan], np.ones(8)),
                                (np.ones((8,1)), np.ones(8))])
def test_incomplete_or_nonfinite_groups_cannot_form_primary_interval(a, b):
    with pytest.raises(ValueError): analysis.independent_difference(a, b)


def baseline(log_qtst, kappa):
    return {'statistics': {'per_seed_log_qtst': np.asarray(log_qtst).tolist(),
                          'per_seed_log_rate': (np.asarray(log_qtst)+np.log(kappa)).tolist()}}


def test_joint_chain_rate_retains_partition_transmission_covariance():
    delta = np.arange(8)*.005
    log_ts = -2.1+delta
    kappa = .77*np.exp(-delta)
    old = baseline(log_ts, kappa)
    result = analysis.full_rate_statistics(log_ts, kappa, 0., 1., old)
    assert result['positive_finite_kappa_for_every_chain'] is True
    statistics = result['statistics']
    assert statistics['log_rate_statistics']['standard_error'] < 1e-14
    assert statistics['same_chain_log_partition_log_kappa_covariance_of_mean'][0][1] < 0
    assert statistics['incorrect_independence_log_rate_variance_for_comparison_only'] > 1e-5
    assert result['bead_comparison']['CI95_wholly_within_margin'] is True


@pytest.mark.parametrize('invalid', [0., -.01, np.nan, np.inf])
def test_nonpositive_or_nonfinite_chain_retained_without_lograte_inference(invalid):
    kappa = np.full(8, .77)
    kappa[3] = invalid
    result = analysis.full_rate_statistics(np.full(8, -2.), kappa, -7., 1., baseline(np.full(8, -2.), np.full(8, .77)))
    assert result['positive_finite_kappa_for_every_chain'] is False
    assert result['statistics'] is None and result['bead_comparison'] is None
    assert len(result['per_chain_kappa']) == 8
    assert result['per_chain_kappa'][3] == (float(invalid) if np.isfinite(invalid) else None)


def reactant_inputs():
    design = json.loads(producer.DESIGN.read_text())
    reference = json.loads((producer.ROOT/design['statistics_and_acceptance']['reactant_reference']['path']).read_text())
    child = design['child_experiment_definition']
    protocol = {'temperature_K':300., 'mass_au':child['mass_au'], 'beta':child['beta'],
                'model':design['parent_generation']['cohorts'][0]['expected_settings']['model']}
    return reference, protocol, design


def test_P256_uses_its_own_finite_P_normalization():
    reference, protocol, design = reactant_inputs()
    row = analysis.finite_P_reactant(reference, protocol, design)
    assert row['beads'] == 256
    assert row['log_quantum_classical_ratio'] == -7.282252233219134
    assert row['log_quantum_classical_ratio'] != reference['primary_log_quantum_classical_ratio']


@pytest.mark.parametrize('change', ['mass','temperature','kernel','domain','PES','duplicate','unconverged'])
def test_mismatched_reactant_reference_rejected(change):
    reference, protocol, design = reactant_inputs()
    row = next(row for row in reference['bead_refinement'] if row['beads'] == 256)
    if change == 'mass': protocol['mass_au'] *= 2
    elif change == 'temperature': protocol['temperature_K'] = 301.
    elif change == 'kernel': row['free_kernel_width_bohr'] *= 1.01
    elif change == 'domain': row['domain_bohr'][0] = .4
    elif change == 'PES': protocol['model']['source_sha256'] = '0'*64
    elif change == 'duplicate': reference['bead_refinement'].append(deepcopy(row))
    else: reference['numerical_converged'] = False
    with pytest.raises(ValueError): analysis.finite_P_reactant(reference, protocol, design)
