"""Scientific regression cases for precision-controller uncertainty inputs."""

import json
import math

import numpy as np
import pytest
from scipy.stats import t

from pyrpmd.precision_statistics import (
    combine_log_rate_uncertainty,
    log_rate_interval,
    numerical_equivalence,
    paired_log_rate_uncertainty,
    paired_rate_replica_statistics,
    pmf_barrier_log_error,
)


def test_rate_error_retains_signed_covariance():
    correlated = combine_log_rate_uncertainty(0.2, 0.2, covariance=0.039)
    anticorrelated = combine_log_rate_uncertainty(0.2, 0.2, covariance=-0.039)
    independent = combine_log_rate_uncertainty(0.2, 0.2, independent=True)
    assert correlated["log_rate_variance"] == pytest.approx(0.158)
    assert anticorrelated["log_rate_variance"] == pytest.approx(0.002)
    assert independent["log_rate_variance"] == pytest.approx(0.08)
    assert independent["assumptions"] == ["independent_pmf_recrossing_estimates"]
    assert correlated["log_rate_standard_error"] > independent["log_rate_standard_error"]


@pytest.mark.parametrize("pmf,recrossing,covariance", [
    (None, 0.1, 0.0), (0.1, None, 0.0), (0.1, 0.1, None),
])
def test_missing_error_or_dependence_cannot_become_zero(pmf, recrossing, covariance):
    result = combine_log_rate_uncertainty(pmf, recrossing, covariance=covariance)
    assert not result["estimable"]
    assert result["log_rate_variance"] is None
    assert result["log_rate_standard_error"] is None
    assert result["reasons"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), True])
def test_invalid_marginal_errors_are_rejected(bad):
    with pytest.raises(ValueError):
        combine_log_rate_uncertainty(bad, 0.1, independent=True)


def test_impossible_covariance_or_contradictory_independence_is_rejected():
    with pytest.raises(ValueError, match="positive semidefinite"):
        combine_log_rate_uncertainty(0.1, 0.2, covariance=0.03)
    with pytest.raises(ValueError, match="positive semidefinite"):
        combine_log_rate_uncertainty(0.1, 0.2, covariance=-0.03)
    with pytest.raises(ValueError, match="zero variance"):
        combine_log_rate_uncertainty(0.0, 0.2, covariance=0.001)
    with pytest.raises(ValueError, match="zero covariance"):
        combine_log_rate_uncertainty(0.1, 0.2, covariance=0.001, independent=True)
    with pytest.raises(ValueError, match="finite"):
        combine_log_rate_uncertainty(0.1, 0.2, covariance=float("nan"))


def test_pmf_endpoint_difference_retains_covariance_and_beta_units():
    result = pmf_barrier_log_error(
        2.0, barrier_variance=0.04, reference_variance=0.04, covariance=0.039,
    )
    assert result["estimable"]
    assert result["barrier_difference_variance"] == pytest.approx(0.002)
    assert result["pmf_variance"] == pytest.approx(0.008)
    assert result["pmf_log_standard_error"] == pytest.approx(math.sqrt(0.008))
    missing = pmf_barrier_log_error(
        2.0, barrier_variance=0.04, reference_variance=0.04, covariance=None,
    )
    assert not missing["estimable"]
    assert missing["pmf_log_standard_error"] is None


def test_pmf_endpoint_covariance_is_validated():
    with pytest.raises(ValueError, match="positive semidefinite"):
        pmf_barrier_log_error(1, barrier_variance=0.04, reference_variance=0.04,
                             covariance=0.1)
    with pytest.raises(ValueError, match="positive"):
        pmf_barrier_log_error(0, barrier_variance=0.04, reference_variance=0.04,
                             covariance=0)


def test_physical_pairing_preserves_covariance_of_means():
    first = np.array([1.0, 2.0, 4.0, 8.0])
    second = np.array([0.5, 1.0, 2.0, 4.0])
    result = paired_log_rate_uncertainty(first, second)
    assert result["estimable"]
    assert result["log_rate_variance"] == pytest.approx(np.var(first + second, ddof=1) / 4)
    assert result["covariance"] == pytest.approx(np.cov(first, second)[0, 1] / 4)
    assert result["degrees_of_freedom"] == 3
    assert result["independent_replicates"] == 4
    with pytest.raises(ValueError, match="cannot exceed"):
        paired_log_rate_uncertainty(first, second, degrees_of_freedom=100)


def test_joint_bootstrap_count_is_not_physical_sample_size():
    first, second = [0.0, 0.2, 0.4, 0.6], [0.2, 0.1, 0.3, 0.2]
    bootstrap = paired_log_rate_uncertainty(first, second, sample_kind="joint_bootstrap")
    physical = paired_log_rate_uncertainty(first, second)
    assert bootstrap["log_rate_variance"] == pytest.approx(physical["log_rate_variance"] * 4)
    assert bootstrap["degrees_of_freedom"] is None
    assert bootstrap["bootstrap_resamples"] == 4
    assert not log_rate_interval(
        1.0, bootstrap["log_rate_standard_error"], alpha=0.05,
        degrees_of_freedom=bootstrap["degrees_of_freedom"],
    )["estimable"]
    supplied = paired_log_rate_uncertainty(first, second, sample_kind="joint_bootstrap",
                                           degrees_of_freedom=12)
    assert supplied["degrees_of_freedom"] == 12


def test_anticorrelated_paired_factors_can_cancel_without_negative_variance():
    result = paired_log_rate_uncertainty([1, 2, 3, 4], [-1, -2, -3, -4])
    assert result["estimable"]
    assert result["log_rate_variance"] == pytest.approx(0, abs=1e-14)


@pytest.mark.parametrize("first,second", [([1], [2]), ([], []), (None, [1, 2])])
def test_insufficient_paired_samples_remain_unestimable(first, second):
    result = paired_log_rate_uncertainty(first, second)
    assert not result["estimable"]
    assert result["log_rate_standard_error"] is None


@pytest.mark.parametrize("first,second", [([1, 2], [1]), ([[1, 2]], [[3, 4]]),
                                        ([1, float("nan")], [2, 3])])
def test_invalid_pairing_is_rejected(first, second):
    with pytest.raises(ValueError):
        paired_log_rate_uncertainty(first, second)


def test_rate_interval_uses_physical_t_degrees_and_relative_worst_side():
    result = log_rate_interval(2.0, 0.1, alpha=0.05, degrees_of_freedom=3)
    expected_half_log = t.isf(0.025, 3) * 0.1
    assert result["interval"] == pytest.approx([2 * math.exp(-expected_half_log),
                                                2 * math.exp(expected_half_log)])
    low, high = result["interval"]
    assert result["relative_half_width"] == pytest.approx(max(2 - low, high - 2) / 2)
    assert "approximate" in result["method"]
    stricter = log_rate_interval(2.0, 0.1, alpha=0.005, degrees_of_freedom=3)
    assert stricter["relative_half_width"] > result["relative_half_width"]


@pytest.mark.parametrize("rate,error,df", [(1, None, 10), (None, 0.1, 10), (1, 0.1, None)])
def test_missing_interval_inputs_do_not_select_normal_fallback(rate, error, df):
    result = log_rate_interval(rate, error, alpha=0.05, degrees_of_freedom=df)
    assert not result["estimable"]
    assert result["interval"] is None
    assert result["relative_half_width"] is None


@pytest.mark.parametrize("df", [0, 0.5, float("inf"), float("nan")])
def test_invalid_degrees_of_freedom_are_rejected(df):
    with pytest.raises(ValueError):
        log_rate_interval(1, 0.1, alpha=0.05, degrees_of_freedom=df)


@pytest.mark.parametrize("alpha", [0, 1, -0.1, float("nan")])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError):
        log_rate_interval(1, 0.1, alpha=alpha, degrees_of_freedom=10)


def test_huge_errors_never_leak_infinity_into_reports():
    rate = log_rate_interval(1, 1e308, alpha=0.05, degrees_of_freedom=2)
    comparison = numerical_equivalence(0, 1e308, log_margin=0.05,
                                       alpha=0.05, degrees_of_freedom=2)
    assert not rate["estimable"]
    assert not comparison["estimable"] and not comparison["established"]
    json.dumps([rate, comparison], allow_nan=False)
    with pytest.raises(ValueError, match="representable"):
        combine_log_rate_uncertainty(1e308, 0.1, independent=True)


def test_noisy_nonsignificant_difference_does_not_establish_equivalence():
    noisy = numerical_equivalence(0, 0.1, log_margin=0.05,
                                  alpha=0.05, degrees_of_freedom=20)
    precise = numerical_equivalence(0.005, 0.005, log_margin=0.05,
                                    alpha=0.05, degrees_of_freedom=20)
    assert noisy["estimable"] and not noisy["established"]
    assert precise["estimable"] and precise["established"]
    assert noisy["interval"][0] < 0 < noisy["interval"][1]
    missing = numerical_equivalence(0, None, log_margin=0.05,
                                    alpha=0.05, degrees_of_freedom=20)
    assert not missing["estimable"] and not missing["established"]


def test_equivalence_boundary_is_not_accepted():
    result = numerical_equivalence(0.05, 0, log_margin=0.05,
                                   alpha=0.05, degrees_of_freedom=20)
    assert result["estimable"] and not result["established"]
    assert result["interval"] == [0.05, 0.05]


def test_all_successful_reports_are_json_serializable():
    reports = [
        combine_log_rate_uncertainty(0.1, 0.2, independent=True, degrees_of_freedom=12),
        paired_log_rate_uncertainty([1, 2, 3], [2, 1, 4]),
        pmf_barrier_log_error(2, barrier_variance=0.02, reference_variance=0.03, covariance=0),
        log_rate_interval(0.2, 0.1, alpha=0.05, degrees_of_freedom=12),
        numerical_equivalence(0.01, 0.01, log_margin=0.05, alpha=0.05,
                              degrees_of_freedom=12),
    ]
    json.dumps(reports, allow_nan=False)


def test_arithmetic_replica_rate_does_not_certify_the_geometric_mean():
    result = paired_rate_replica_statistics([0.0, 0.0], np.log([0.5, 1.5]))
    assert result["estimable"]
    assert result["rate"] == pytest.approx(1.0)
    assert result["rate"] != pytest.approx(math.sqrt(0.75))
    assert result["rate_standard_error"] == pytest.approx(0.5)
    assert result["log_rate_variance"] == pytest.approx(0.25)
    assert result["degrees_of_freedom"] == 1


@pytest.mark.parametrize("first,second,sign", [
    ([1., 2., 3., 4.], [1., 2., 3., 4.], 1),
    ([1., 2., 3., 4.], [4., 3., 2., 1.], -1),
])
def test_arithmetic_replica_covariance_reconstructs_direct_sample_error(first, second, sign):
    rates = np.asarray(first) * np.asarray(second)
    result = paired_rate_replica_statistics(np.log(first), np.log(second))
    direct_error = float(np.std(rates, ddof=1) / math.sqrt(len(rates)))
    assert result["rate"] == pytest.approx(np.mean(rates))
    assert result["rate_standard_error"] == pytest.approx(direct_error)
    assert result["log_rate_standard_error"] == pytest.approx(direct_error / np.mean(rates))
    assert result["pmf_variance"] + result["recrossing_variance"] + 2 * result["covariance"] == pytest.approx(
        direct_error ** 2 / np.mean(rates) ** 2,
    )
    assert result["covariance"] * sign > 0
    assert result["degrees_of_freedom"] == len(rates) - 1
    assert result["independent_replicates"] == len(rates)
    assert "not a unique causal" in result["attribution"]
    json.dumps(result, allow_nan=False)


def test_arithmetic_replica_mean_handles_large_overall_log_offsets():
    factors = np.array([0.9, 1.0, 1.1, 1.2])
    result = paired_rate_replica_statistics(600 + np.log(factors), [-600.] * 4)
    assert result["rate"] == pytest.approx(np.mean(factors))
    assert result["rate_standard_error"] == pytest.approx(np.std(factors, ddof=1) / 2)


def test_arithmetic_small_resolved_noise_survives_centering_roundoff():
    rates = np.array([1.0, 1.002, 0.999])
    result = paired_rate_replica_statistics([math.log(2)] * 3, np.log(rates / 2))
    assert result["rate_standard_error"] == pytest.approx(np.std(rates, ddof=1) / math.sqrt(3))


def test_arithmetic_cancellation_cannot_silently_erase_resolved_rate_error():
    q = np.array([1., 2., 3., 4.])
    rates = 1 + np.array([-1., 1., -1., 1.]) * 1e-12
    with pytest.raises(ValueError, match="cancellation"):
        paired_rate_replica_statistics(np.log(q), np.log(rates / q))


@pytest.mark.parametrize("first,second", [(None, [0, 1]), ([0], [0]), ([], [])])
def test_arithmetic_replica_missing_data_are_not_zero_error(first, second):
    result = paired_rate_replica_statistics(first, second)
    assert not result["estimable"]
    assert result["rate_standard_error"] is None
    assert result["log_rate_standard_error"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("first,second", [
    ([0, float("nan")], [0, 0]),
    ([1000, 1000], [0, 0]),
    ([-1000, -1000], [0, 0]),
    ([1000, 0], [-1000, 0]),
])
def test_arithmetic_replica_unrepresentable_values_are_rejected(first, second):
    with pytest.raises(ValueError):
        paired_rate_replica_statistics(first, second)
