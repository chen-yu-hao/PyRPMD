"""Physical-time rescaling for audited infrequent MetaD / OPES-flooding runs.

The boost uses bias relative to the transition-region bias, not an arbitrary
energy zero. Eligibility is an explicit input: timeseries algebra cannot
prove that a dividing surface was unperturbed or that a CV resolves it.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import kstest

from .constants import ATOMIC_TIME_S


def rescale_biased_time(times, bias, *, beta, transition_bias, transition_unbiased,
                       deposited_in_transition=False):
    """Integrate exp(beta*(V-V_TS)) over completed, piecewise-constant steps.

    ``times`` contains N+1 step boundaries in atomic time, ``bias`` contains
    the N physical bias energies held during each integration interval.
    ``transition_bias`` is scalar or N values in the same energy gauge.
    OPES-flooding usually needs its negative asymptotic bias as this reference.
    Passing sparse trajectory samples silently would misestimate the clock;
    the caller must record the bias on every integration interval.
    """
    if transition_unbiased is not True:
        raise ValueError("kinetic rescaling requires an explicit unperturbed-transition-region audit")
    if not isinstance(deposited_in_transition, (bool, np.bool_)) or deposited_in_transition:
        raise ValueError("bias deposition in the transition region invalidates infrequent kinetics")
    times, bias, reference = [np.asarray(x) for x in (times, bias, transition_bias)]
    if (times.dtype.kind not in "iuf" or bias.dtype.kind not in "iuf" or reference.dtype.kind not in "iuf"
            or times.ndim != 1 or len(times) < 2 or bias.shape != (len(times) - 1,)
            or reference.shape not in ((), bias.shape)
            or not all(np.isfinite(x).all() for x in (times, bias, reference))):
        raise ValueError("provide increasing finite N+1 times, N biases, and scalar or N transition biases")
    # Subtraction in unsigned/integer input dtypes can wrap, accepting a
    # backwards clock or giving a negative exposure. Work in the floating
    # representation used by the exponential integral and reject lost steps.
    with np.errstate(over="ignore", invalid="ignore"):
        times, bias, reference = [np.asarray(x, dtype=float) for x in (times, bias, reference)]
        intervals = np.diff(times)
        elapsed = float(times[-1] - times[0])
    if (not all(np.isfinite(x).all() for x in (times, bias, reference, intervals))
            or np.any(intervals <= 0) or not np.isfinite(elapsed) or elapsed <= 0):
        raise ValueError("increasing finite clock intervals must be representable in float64")
    if (isinstance(beta, (bool, np.bool_)) or not np.isscalar(beta) or np.iscomplexobj(beta)
            or not np.isfinite(beta) or beta <= 0):
        raise ValueError("beta must be positive and finite")
    with np.errstate(over="ignore", invalid="ignore"):
        delta = bias - reference
        logs = np.log(intervals) + float(beta) * delta
    if not np.isfinite(logs).all():
        raise ValueError("biased clock exponent is not representable")
    log_time = float(logsumexp(logs))
    if log_time > np.log(np.finfo(float).max):
        raise ValueError("rescaled time overflows; retain a log-time analysis")
    physical = float(np.exp(log_time))
    acceleration = physical / elapsed
    if not np.isfinite(acceleration) or acceleration <= 0 or physical <= 0:
        raise ValueError("rescaled time or acceleration factor is not representable")
    return {"elapsed_biased_time_au": elapsed, "rescaled_time_au": physical,
            "log_rescaled_time_au": log_time, "acceleration_factor": acceleration,
            "beta_au": float(beta), "intervals": len(bias), "transition_unbiased": True,
            "converged": False}


def estimate_biased_rate(rescaled_times, *, events=None, seed=0, bootstrap=1000):
    """Exponential first-passage MLE, with explicit right-censored exposure.

    Independent replicas are required. Censored runs contribute observation
    time without being counted as events. For complete samples, a parametric
    bootstrap refits the exponential scale on each draw before comparing KS
    statistics; an ordinary fitted-parameter KS p-value is not valid.
    No physical convergence or transition-state audit is inferred here.
    """
    times = np.asarray(rescaled_times)
    if (times.dtype.kind not in "iuf" or times.ndim != 1 or not len(times)
            or not np.isfinite(times).all() or np.any(times <= 0)):
        raise ValueError("rescaled first-passage/exposure times must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        times = np.asarray(times, dtype=float)
        exposure = float(np.sum(times))
    if not np.isfinite(times).all() or np.any(times <= 0) or not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("total rescaled exposure must be positive and representable in float64")
    events = np.ones(len(times), bool) if events is None else np.asarray(events)
    if events.shape != times.shape or events.dtype.kind != "b":
        raise ValueError("events must contain one explicit boolean per exposure")
    if isinstance(bootstrap, (bool, np.bool_)) or not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    count = int(np.sum(events))
    rate = count / exposure
    if not np.isfinite(rate / ATOMIC_TIME_S):
        raise ValueError("estimated rate is not representable")
    statistic = pvalue = None
    if count == len(times) and len(times) >= 3:
        mean = float(np.mean(times))
        statistic = float(kstest(times / mean, "expon").statistic)
        if bootstrap:
            rng = np.random.default_rng(seed)
            larger = 0
            for _ in range(bootstrap):
                sample = rng.exponential(size=len(times))
                larger += kstest(sample / sample.mean(), "expon").statistic >= statistic
            pvalue = float((larger + 1) / (bootstrap + 1))
    return {"method": "rescaled_exponential_first_passage", "replicas": len(times),
            "events": count, "censored": int(len(times) - count),
            "exposure_au": exposure, "rate_au": rate, "rate_s_inv": rate / ATOMIC_TIME_S,
            "mean_first_passage_au": 1 / rate if rate else None,
            "ks_statistic": statistic, "ks_refitted_bootstrap_pvalue": pvalue,
            "converged": False}
