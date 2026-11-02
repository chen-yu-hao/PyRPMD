"""Harmonic reactant partitions on the two golden-rule imaginary-time grids.

Each interval uses the same primitive kinetic kernel and trapezoidal endpoint
potential weights as GoldenRuleAction. The two constant-step sections need not
have equal steps or equal interval counts. A two-by-two transfer-matrix product
evaluates each normal mode in constant work, without a bead Hessian or a dense
determinant. This is the discrete Gaussian partition, not a continuum-Q shortcut.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from .validation import finite_positive, positive_integer


_LOG2 = math.log(2.0)
_LOG4 = math.log(4.0)


def _logsinh_of_logx(logx):
    """log(sinh(exp(logx))) without small-argument cancellation or overflow."""
    if logx < -20:
        # sinh(x)/x differs from one by less than binary64 precision here.
        return logx
    if logx > math.log(sys.float_info.max):
        raise ValueError("harmonic partition exceeds the finite logarithmic range")
    x = math.exp(logx)
    if x < 20:
        return math.log(math.sinh(x))
    return x + math.log1p(-math.exp(-2*x)) - _LOG2


def _log_segment_exponent(log_u, intervals):
    """log(2*P*asinh(u/2)), retaining arbitrarily small positive u in logs."""
    x = log_u - _LOG2
    if x < -20:
        log_asinh = x
    else:
        # asinh(exp(x)) = log(exp(x) + sqrt(exp(2*x)+1)).
        value = float(np.logaddexp(x, 0.5 * np.logaddexp(2*x, 0.0)))
        log_asinh = math.log(value)
    return _LOG2 + math.log(intervals) + log_asinh


def harmonic_segment_log_partition(frequencies_au, nintervals0, nintervals1,
                                    tau0_au, tau1_au, *, electronic_degeneracy=1):
    """Return log Q for positive harmonic frequencies on a two-section ring.

    ``frequencies_au`` contains angular frequencies (not curvatures), in inverse
    atomic time. ``tau0_au + tau1_au`` is beta; each section has Pi intervals of
    duration tau_i/P_i. Electronic degeneracy is included; electronic minimum
    energies are deliberately not included. No free translation or rotation
    is represented. All frequencies and durations must be positive and finite.

    For each mode, define u_i=omega*tau_i/P_i, A_i=1+u_i**2/4 and
    a_i=2*P_i*asinh(u_i/2). If c=(A0+A1)/(2*sqrt(A0*A1)), then
    Q**(-2)=4*sinh((a0+a1)/2)**2 + 2*(c-1)*sinh(a0)*sinh(a1).
    This follows from the trace of the two oscillator transfer matrices.
    All sums/products in that expression are evaluated logarithmically.
    Equal steps recover 1/[2*sinh(P*asinh(beta*omega/(2*P)))].
    """
    raw = np.asarray(frequencies_au)
    if (raw.dtype.kind not in "fiu" or raw.ndim != 1
            or not np.isfinite(raw).all() or np.any(raw <= 0)):
        raise ValueError("frequencies_au must be a one-dimensional array of positive finite angular frequencies")
    frequencies = np.asarray(raw, dtype=float)
    counts = (positive_integer(nintervals0, "nintervals0"),
              positive_integer(nintervals1, "nintervals1"))
    times = (finite_positive(tau0_au, "tau0_au"), finite_positive(tau1_au, "tau1_au"))
    electronic = positive_integer(electronic_degeneracy, "electronic_degeneracy")
    log_steps = [math.log(tau)-math.log(p) for p, tau in zip(counts, times)]
    log_q = math.log(electronic)
    for omega in frequencies:
        log_u = [math.log(float(omega))+step for step in log_steps]
        log_a = [_log_segment_exponent(u, p) for u, p in zip(log_u, counts)]
        log_a_sum_half = float(np.logaddexp(*log_a)) - _LOG2
        first = _LOG4 + 2*_logsinh_of_logx(log_a_sum_half)
        log_A = [float(np.logaddexp(0.0, 2*u-_LOG4)) for u in log_u]
        difference = 0.25*abs(log_A[0]-log_A[1])
        if difference == 0:
            second = -math.inf
        else:
            # c-1=2*sinh((log A0-log A1)/4)^2 avoids subtracting 1
            # from nearly identical transfer-matrix impedances.
            log_c_minus_one = _LOG2 + 2*_logsinh_of_logx(math.log(difference))
            second = (_LOG2 + log_c_minus_one
                      + _logsinh_of_logx(log_a[0]) + _logsinh_of_logx(log_a[1]))
        log_q -= 0.5*float(np.logaddexp(first, second))
    if not math.isfinite(log_q):
        raise ValueError("harmonic partition exceeds the finite logarithmic range")
    return float(log_q)
