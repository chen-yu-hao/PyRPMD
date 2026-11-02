"""Conservative uncertainty primitives for thermal-rate precision control.

These functions consume *estimated* sampling errors; they do not establish
mixing, independent parent chains, a recrossing plateau, or numerical
convergence.  In particular, Student-t/delta intervals below are approximate
for nonlinear RPMD estimators.  A caller must supply physical degrees of
freedom and the per-look alpha from its frozen inference plan.  Bootstrap
resample counts are never interpreted as physical sample counts.

The propagation formulas retain covariance (NIST TN 1297, Appendix A).
Student-t quantiles use scipy.stats.t; there is no normal fallback when the
degrees of freedom are unknown.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from scipy.stats import t


def _number(value, name, *, nonnegative=False, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _optional_number(value, name, **kwargs):
    return None if value is None else _number(value, name, **kwargs)


def _df(value):
    value = _optional_number(value, "degrees_of_freedom", positive=True)
    if value is not None and value < 1:
        raise ValueError("degrees_of_freedom must be at least one")
    return value


def _alpha(value):
    value = _number(value, "alpha", positive=True)
    if value >= 1:
        raise ValueError("alpha must be strictly between zero and one")
    return value


def _square(value, name):
    result = value * value
    if not math.isfinite(result) or (value != 0 and result == 0):
        raise ValueError(f"{name} is outside the representable variance range")
    return result


def _psd_covariance(first_variance, second_variance, covariance):
    """Validate a 2x2 covariance matrix, tolerating only last-bit rounding."""
    limit = math.sqrt(first_variance) * math.sqrt(second_variance)
    if limit == 0:
        if covariance != 0:
            raise ValueError("covariance is incompatible with a zero variance")
        return 0.0
    if abs(covariance) > limit:
        if abs(covariance) - limit > 8 * math.ulp(limit):
            raise ValueError("covariance matrix must be positive semidefinite")
        covariance = math.copysign(limit, covariance)
    return covariance


def _combined_variance(first_variance, second_variance, covariance):
    # Scale before summation so legitimate cancellation does not overflow.
    scale = max(first_variance, second_variance, abs(covariance))
    if scale == 0:
        return 0.0
    scaled = math.fsum((first_variance / scale, second_variance / scale,
                        2 * (covariance / scale)))
    if scaled < 0:
        if scaled < -16 * np.finfo(float).eps:
            raise ValueError("combined variance must be nonnegative")
        scaled = 0.0
    variance = scale * scaled
    if not math.isfinite(variance):
        raise ValueError("combined variance is outside the finite float range")
    return variance


def combine_log_rate_uncertainty(
    pmf_log_standard_error,
    recrossing_log_standard_error,
    *,
    covariance=None,
    independent=False,
    degrees_of_freedom=None,
):
    """Combine errors of log(k_QTST) and log(kappa), including covariance.

    ``covariance`` is the covariance of these two *estimated log factors*,
    not a correlation coefficient or a covariance of individual observations.
    Independence must be explicitly justified by the calling workflow.  If
    neither covariance nor independence is supplied, the total error remains
    unavailable even when both marginal errors are known.
    """
    pmf = _optional_number(pmf_log_standard_error, "pmf_log_standard_error",
                           nonnegative=True)
    recrossing = _optional_number(recrossing_log_standard_error,
                                  "recrossing_log_standard_error",
                                  nonnegative=True)
    covariance = _optional_number(covariance, "covariance")
    df = _df(degrees_of_freedom)
    if not isinstance(independent, (bool, np.bool_)):
        raise ValueError("independent must be boolean")
    if independent and covariance not in (None, 0):
        raise ValueError("independent=True requires zero covariance")
    if independent:
        covariance = 0.0
    pmf_variance = None if pmf is None else _square(pmf, "PMF standard error")
    recrossing_variance = (None if recrossing is None else
                           _square(recrossing, "recrossing standard error"))
    reasons = []
    if pmf is None:
        reasons.append("missing_pmf_log_standard_error")
    if recrossing is None:
        reasons.append("missing_recrossing_log_standard_error")
    if covariance is None:
        reasons.append("unknown_pmf_recrossing_covariance")
    result = {
        "estimable": not reasons,
        "pmf_variance": pmf_variance,
        "recrossing_variance": recrossing_variance,
        "covariance": covariance,
        "log_rate_variance": None,
        "log_rate_standard_error": None,
        "degrees_of_freedom": df,
        "method": "covariance_propagation_log_rate",
        "assumptions": ["independent_pmf_recrossing_estimates"] if independent else [],
        "reasons": reasons,
    }
    if reasons:
        return result
    covariance = _psd_covariance(pmf_variance, recrossing_variance, covariance)
    variance = _combined_variance(pmf_variance, recrossing_variance, covariance)
    result.update(covariance=covariance, log_rate_variance=variance,
                  log_rate_standard_error=math.sqrt(variance))
    return result


def pmf_barrier_log_error(beta, *, barrier_variance, reference_variance, covariance):
    """Propagate endpoint covariance through -beta*(F_surface-F_reference).

    Endpoint variances and covariance are in squared energy units, while beta
    must use the reciprocal energy unit.  The surface and reference must be
    fixed by the inference design; this does not account for surface selection
    performed on the same production data.
    """
    beta = _number(beta, "beta", positive=True)
    surface = _optional_number(barrier_variance, "barrier_variance", nonnegative=True)
    reference = _optional_number(reference_variance, "reference_variance",
                                  nonnegative=True)
    covariance = _optional_number(covariance, "covariance")
    reasons = []
    if surface is None:
        reasons.append("missing_barrier_variance")
    if reference is None:
        reasons.append("missing_reference_variance")
    if covariance is None:
        reasons.append("unknown_pmf_endpoint_covariance")
    result = {
        "estimable": not reasons,
        "barrier_variance": surface,
        "reference_variance": reference,
        "covariance": covariance,
        "barrier_difference_variance": None,
        "pmf_variance": None,
        "pmf_log_standard_error": None,
        "method": "pmf_endpoint_covariance",
        "reasons": reasons,
    }
    if reasons:
        return result
    covariance = _psd_covariance(surface, reference, covariance)
    difference = _combined_variance(surface, reference, -covariance)
    log_error = beta * math.sqrt(difference)
    log_variance = _square(log_error, "PMF log standard error")
    result.update(covariance=covariance, barrier_difference_variance=difference,
                  pmf_variance=log_variance, pmf_log_standard_error=log_error)
    return result


def paired_log_rate_uncertainty(
    log_qtst,
    log_kappa,
    *,
    sample_kind="independent_replicates",
    degrees_of_freedom=None,
):
    """Retain pairing in physical replicates or joint bootstrap estimates.

    Independent physical replicates estimate the covariance of their means
    (sample covariance divided by n) with n-1 degrees of freedom.  Joint
    bootstrap draws already describe the distribution of the estimator:
    their covariance is *not* divided by the bootstrap resample count and
    physical degrees of freedom must be supplied separately.  Shared array
    indices must correspond to genuine joint resampling, not arbitrary pairing
    of independently generated bootstrap distributions.
    """
    if sample_kind not in ("independent_replicates", "joint_bootstrap"):
        raise ValueError("sample_kind must be independent_replicates or joint_bootstrap")
    df = _df(degrees_of_freedom)
    if log_qtst is None or log_kappa is None:
        result = combine_log_rate_uncertainty(None, None, degrees_of_freedom=df)
        result.update(method="paired_" + sample_kind, sample_kind=sample_kind)
        result["reasons"] = ["missing_paired_log_factor_samples"]
        return result
    try:
        first, second = np.asarray(log_qtst, dtype=float), np.asarray(log_kappa, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("paired log factors must be finite numeric vectors") from exc
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("paired log factors must be one-dimensional and equally sized")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("paired log factors must be finite")
    n = len(first)
    if n < 2:
        result = combine_log_rate_uncertainty(None, None, degrees_of_freedom=df)
        result.update(method="paired_" + sample_kind, sample_kind=sample_kind)
        result["reasons"] = ["fewer_than_two_paired_samples"]
        return result
    if sample_kind == "independent_replicates":
        if df is not None and df > n - 1:
            raise ValueError("degrees_of_freedom cannot exceed independent replicate count minus one")
        if df is None:
            df = float(n - 1)
    with np.errstate(over="ignore", invalid="ignore"):
        matrix = np.cov(np.vstack((first, second)), ddof=1)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("sample covariance is outside the finite float range")
    if sample_kind == "independent_replicates":
        matrix /= n
    result = combine_log_rate_uncertainty(
        math.sqrt(float(matrix[0, 0])), math.sqrt(float(matrix[1, 1])),
        covariance=float(matrix[0, 1]), degrees_of_freedom=df,
    )
    result.update(method="paired_" + sample_kind, sample_kind=sample_kind)
    if sample_kind == "independent_replicates":
        result["independent_replicates"] = n
        result["assumptions"] = ["independent_physical_replicates"]
    else:
        result["bootstrap_resamples"] = n
        result["assumptions"] = ["joint_resampling_preserves_sampling_dependence"]
    return result


def paired_rate_replica_statistics(log_qtst, log_kappa):
    """Estimate the arithmetic mean rate from independent complete replicas.

    Pairing of QTST and recrossing factors is retained within each replica.
    A symmetric product-fluctuation decomposition assigns half of the product
    interaction to each factor.  Its covariance of means reconstructs the
    relative variance of the arithmetic rate mean.  This is a descriptive
    decomposition, not a unique causal allocation of sampling error.

    Independent replicas provide n-1 physical degrees of freedom.  Subsequent
    log-delta intervals remain approximate, and this operation does not remove
    any finite-sample bias already present in a complete replica estimator.
    """
    missing = log_qtst is None or log_kappa is None
    if not missing:
        try:
            first = np.asarray(log_qtst, dtype=float)
            second = np.asarray(log_kappa, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("paired log factors must be finite numeric vectors") from exc
        if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
            raise ValueError("paired log factors must be one-dimensional and equally sized")
        if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
            raise ValueError("paired log factors must be finite")
    if missing or len(first) < 2:
        result = combine_log_rate_uncertainty(None, None)
        result.update(rate=None, standard_error=None, rate_standard_error=None,
                      method="symmetrized_product_fluctuation")
        result["reasons"] = ["missing_paired_log_factor_samples" if missing else
                              "fewer_than_two_independent_replicates"]
        return result

    n = len(first)
    max_q, max_c = float(np.max(first)), float(np.max(second))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        q = np.exp(first - max_q)
        c = np.exp(second - max_c)
        products = q * c
        qbar, cbar, rbar = float(np.mean(q)), float(np.mean(c)), float(np.mean(products))
    if rbar <= 0 or not math.isfinite(rbar):
        raise ValueError("scaled replica products are outside the representable range")
    try:
        log_rate = math.fsum((max_q, max_c, math.log(rbar)))
        rate = math.exp(log_rate)
    except (OverflowError, ValueError) as exc:
        raise ValueError("arithmetic replica rate is outside the representable range") from exc
    if rate <= 0 or not math.isfinite(rate):
        raise ValueError("arithmetic replica rate is outside the representable range")

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        pmf_fluctuation = 0.5 * (q - qbar) * (c + cbar) / rbar
        recrossing_fluctuation = 0.5 * (c - cbar) * (q + qbar) / rbar
        matrix = np.cov(np.vstack((pmf_fluctuation, recrossing_fluctuation)), ddof=1) / n
        direct_variance = float(np.var(products / rbar, ddof=1) / n)
    if not np.all(np.isfinite(matrix)) or not math.isfinite(direct_variance):
        raise ValueError("replica product covariance is outside the representable range")
    result = combine_log_rate_uncertainty(
        math.sqrt(float(matrix[0, 0])), math.sqrt(float(matrix[1, 1])),
        covariance=float(matrix[0, 1]), degrees_of_freedom=n - 1,
    )
    # Comparing the reconstruction to the direct sample variance also catches
    # loss of the rate fluctuation in strongly cancelling factor covariances.
    scale = max(result["pmf_variance"], result["recrossing_variance"], direct_variance)
    # Centering a normalized product near one incurs O(eps) absolute error in
    # its fluctuations, hence an O(eps*SD) error in its variance.  A tolerance
    # based only on eps*variance is too strict for small but resolved noise.
    eps = np.finfo(float).eps
    tolerance = 64 * eps * (scale + math.sqrt(direct_variance) + eps)
    if abs(result["log_rate_variance"] - direct_variance) > tolerance:
        raise ValueError("factor covariance does not resolve the arithmetic rate variance")
    if direct_variance > 0 and result["log_rate_variance"] < direct_variance * (1 - 1e-6):
        raise ValueError("arithmetic rate variance is lost in factor covariance cancellation")
    standard_error = rate * math.sqrt(direct_variance)
    if not math.isfinite(standard_error):
        raise ValueError("arithmetic rate standard error is outside the finite float range")
    result.update(
        rate=rate, standard_error=standard_error, rate_standard_error=standard_error,
        log_rate_variance=direct_variance, log_rate_standard_error=math.sqrt(direct_variance),
        independent_replicates=n, method="symmetrized_product_fluctuation",
        estimand="arithmetic_mean_of_complete_replica_rates",
        attribution="descriptive symmetric product split; not a unique causal variance allocation",
        assumptions=["independent_complete_rate_replicates", "same_physical_rate_estimand",
                     "finite_replica_estimator_bias_not_removed", "subsequent_log_delta_interval_is_approximate"],
    )
    return result


def _critical_value(alpha, degrees_of_freedom):
    if alpha / 2 == 0:
        return None
    # isf is stable for small alpha; ppf(1-alpha/2) can round its argument to 1.
    critical = float(t.isf(alpha / 2, degrees_of_freedom))
    return critical if math.isfinite(critical) else None


def log_rate_interval(rate, log_standard_error, *, alpha, degrees_of_freedom):
    """Approximate Student-t/delta interval on a positive thermal rate.

    The interval is ``rate*exp(+/- t*SE_log)``.  Marginal standard errors and
    covariance may be bootstrap/delta estimates, so a t multiplier alone does
    not make this an exact finite-sample interval.  Missing physical degrees
    of freedom or an unrepresentable interval leave precision unestablished.
    """
    rate = _optional_number(rate, "rate", positive=True)
    error = _optional_number(log_standard_error, "log_standard_error", nonnegative=True)
    alpha, df = _alpha(alpha), _df(degrees_of_freedom)
    reasons = []
    if rate is None:
        reasons.append("missing_rate")
    if error is None:
        reasons.append("missing_log_standard_error")
    if df is None:
        reasons.append("missing_physical_degrees_of_freedom")
    result = {
        "estimable": False,
        "interval": None,
        "relative_half_width": None,
        "confidence_level": 1 - alpha,
        "alpha": alpha,
        "degrees_of_freedom": df,
        "critical_value": None,
        "method": "approximate_student_t_delta_log_rate",
        "reasons": reasons,
    }
    if reasons:
        return result
    critical = _critical_value(alpha, df)
    result["critical_value"] = critical
    if critical is None:
        reasons.append("student_t_quantile_unrepresentable")
        return result
    half_width = critical * error
    try:
        relative_half_width = math.expm1(half_width)
        low, high = rate * math.exp(-half_width), rate * math.exp(half_width)
    except OverflowError:
        reasons.append("rate_interval_unrepresentable")
        return result
    if not all(math.isfinite(x) for x in (relative_half_width, low, high)):
        reasons.append("rate_interval_unrepresentable")
        return result
    result.update(estimable=True, interval=[low, high],
                  relative_half_width=relative_half_width)
    return result


def numerical_equivalence(
    log_ratio,
    standard_error,
    *,
    log_margin,
    alpha,
    degrees_of_freedom,
):
    """Establish equivalence only by strict interval containment.

    The supplied error describes the *log rate contrast* and must include
    covariance when the controlled comparisons are paired.  The two-sided
    1-alpha interval is deliberately at least as conservative as a TOST test
    at alpha (which ordinarily uses a central 1-2*alpha interval).  Failure
    to establish equivalence is not evidence of a physical difference.
    """
    ratio = _optional_number(log_ratio, "log_ratio")
    error = _optional_number(standard_error, "standard_error", nonnegative=True)
    margin = _number(log_margin, "log_margin", positive=True)
    alpha, df = _alpha(alpha), _df(degrees_of_freedom)
    reasons = []
    if ratio is None:
        reasons.append("missing_log_ratio")
    if error is None:
        reasons.append("missing_log_ratio_standard_error")
    if df is None:
        reasons.append("missing_physical_degrees_of_freedom")
    result = {
        "established": False,
        "estimable": False,
        "interval": None,
        "log_margin": margin,
        "confidence_level": 1 - alpha,
        "alpha": alpha,
        "degrees_of_freedom": df,
        "method": "approximate_student_t_log_ratio_containment",
        "reasons": reasons,
    }
    if reasons:
        return result
    critical = _critical_value(alpha, df)
    if critical is None:
        reasons.append("student_t_quantile_unrepresentable")
        return result
    half_width = critical * error
    low, high = ratio - half_width, ratio + half_width
    if not math.isfinite(low) or not math.isfinite(high):
        reasons.append("log_ratio_interval_unrepresentable")
        return result
    established = -margin < low and high < margin
    if not established:
        reasons.append("interval_not_inside_equivalence_margin")
    result.update(estimable=True, established=established, interval=[low, high])
    return result
