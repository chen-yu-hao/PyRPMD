"""Periodic Gaussian links with bounded image-sum truncation.

For a row lattice B and alpha > 0 this module evaluates
``sum_n exp(-alpha * |d + n B|**2)`` and the first two displacement
moments of that sum.  These are different from derivatives of a quadratic
minimum-image spring.  Orthogonal periodic directions are separable; broad
Gaussians use the Poisson-dual sum, including the small-volume limit.

The reported bounds concern omitted terms, not floating-point roundoff.
Their scope is one link: relative kernel error, absolute first-moment error
in units alpha**(-1/2), and Frobenius second-moment error divided by
``alpha**(-1) + ||second_moment||``.  A uniform bound over the fractional
cell determines a fixed cutoff, so differentiating the retained sum does
not differentiate a position-dependent cutoff.  General skew cells may
exhaust the explicit summation budget and then fail rather than silently
switching to a minimum-image approximation.

See Higer, Feldman and Hirshberg (2025), arXiv:2501.17618, for periodic
bosonic link factorization.  Poisson duality is applied to the same links.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np
from scipy.special import log_ndtr


DEFAULT_WINDING_TOLERANCE = 1e-12
DEFAULT_WINDING_MAX_TERMS = 100000
_MAX_CACHED_TERMS = 1000000
_PAIR_CHUNK = 262144
_LOG_TWO = math.log(2.0)
_LOG_PI = math.log(math.pi)
_NEG_INF = float("-inf")


def validate_winding_options(tolerance, max_terms, periodic_mode=None):
    """Validate numerical controls, including controls supplied to old modes."""
    if (isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance)
            or not isinstance(tolerance, (int, float, np.integer, np.floating))):
        raise ValueError("winding_tolerance must be a finite real number in (0,1)")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("winding_tolerance must be a finite real number in (0,1)")
    if (isinstance(max_terms, (bool, np.bool_))
            or not isinstance(max_terms, (int, np.integer))
            or not 1 <= max_terms <= _MAX_CACHED_TERMS):
        raise ValueError(f"winding_max_terms must be an integer in [1,{_MAX_CACHED_TERMS}]")
    max_terms = int(max_terms)
    if periodic_mode != "full_winding" and (
            tolerance != DEFAULT_WINDING_TOLERANCE or max_terms != DEFAULT_WINDING_MAX_TERMS):
        raise ValueError("winding controls require periodic_mode='full_winding'")
    return tolerance, max_terms


def _logadd(*values):
    out = _NEG_INF
    for value in values:
        out = float(np.logaddexp(out, value))
    return out


def _log_zero_tail(c, cutoff, shift):
    """Two-sided Gaussian tail <= twice (first term + tail integral)."""
    x = cutoff + 1 - shift
    z = math.sqrt(c) * x
    exponent = -(z * z)
    log_erfc = _LOG_TWO + float(log_ndtr(-math.sqrt(2.0) * z))
    integral = 0.5 * _LOG_PI - _LOG_TWO - 0.5 * math.log(c) + log_erfc
    return _LOG_TWO + _logadd(exponent, integral)


def _log_square_tail(c, cutoff):
    """Two-sided n**2 exp(-c n**2) tail, for c >= pi and cutoff >= 0."""
    x = cutoff + 1
    z = math.sqrt(c) * x
    exponent = -(z * z)
    log_erfc = _LOG_TWO + float(log_ndtr(-math.sqrt(2.0) * z))
    return _LOG_TWO + _logadd(2 * math.log(x) + exponent,
        math.log(x) + exponent - _LOG_TWO - math.log(c),
        0.5 * _LOG_PI - 2 * _LOG_TWO - 1.5 * math.log(c) + log_erfc)


def _cube_tails(k, log_s0, log_s2, log_t0, log_t2):
    log_r0 = math.log(k) + log_t0 + (k - 1) * log_s0
    first = log_t2 + (k - 1) * log_s0
    second = (math.log(k - 1) + log_t0 + log_s2 + (k - 2) * log_s0
              if k > 1 else _NEG_INF)
    return log_r0, math.log(k) + _logadd(first, second)


def _bound_float(value):
    # Positive subnormal output remains conservative when exp underflows.
    return max(float(np.exp(value)), float(np.nextafter(0.0, 1.0)))


@dataclass(frozen=True)
class WindingMoments:
    log_kernel: np.ndarray
    mean_displacement: np.ndarray
    second_moment: np.ndarray


class _LatticePlan:
    """One k-dimensional periodic subspace, in dimensionless coordinates."""

    def __init__(self, basis, alpha, tolerance, max_terms):
        self.basis = np.array(basis, dtype=float, copy=True)
        self.alpha = alpha
        self.sqrt_alpha = math.sqrt(alpha)
        self.k = len(basis)
        u, singular, vt = np.linalg.svd(self.basis, full_matrices=False)
        self.u, self.singular, self.vt = u, singular, vt
        self.projector = vt.T @ vt
        log_c = math.log(alpha) + 2 * math.log(singular[-1])
        log_C = math.log(alpha) + 2 * math.log(singular[0])
        log_dual_c = 2 * _LOG_PI - log_C
        self.representation = "reciprocal" if log_dual_c >= _LOG_PI else "direct"
        if self.representation == "reciprocal":
            self.log_c, self.log_C = log_dual_c, 2 * _LOG_PI - log_c
            # A smaller c gives a conservative tail bound even when the true
            # dimensionless reciprocal spacing overflows a floating scalar.
            self.c = math.exp(min(self.log_c, math.log(1e6)))
            self.log_prefactor = 0.5 * self.k * (_LOG_PI - math.log(alpha)) - float(np.log(singular).sum())
            self.log_s0 = _logadd(0.0, _log_zero_tail(self.c, 0, 0.0))
            self.log_s2 = _log_square_tail(self.c, 0)
            off_diagonal = math.expm1(self.k * self.log_s0)
            if off_diagonal >= 1:
                raise ValueError("reciprocal winding denominator has no positive certified bound")
            self.log_lower = math.log1p(-off_diagonal)
        else:
            if not math.isfinite(log_c) or log_c < math.log(np.finfo(float).tiny):
                raise ValueError("periodic Gaussian is too anisotropic for a reliable direct winding sum")
            if log_C > math.log(np.finfo(float).max) - 8:
                raise ValueError("periodic Gaussian spring scale overflows float64")
            self.c = math.exp(log_c)
            self.log_c, self.log_C = log_c, log_C
            self.log_s0 = _logadd(0.0, _log_zero_tail(self.c, 0, 0.5))
            log_coefficient = _logadd(math.log(4.0) - 1 - log_c, _LOG_TWO)
            self._log_square_coefficient = log_coefficient
            self.log_s2 = log_coefficient + _logadd(0.0, _log_zero_tail(self.c / 2, 0, 0.5))
            corners = np.array(list(itertools.product((-0.5, 0.5), repeat=self.k)))
            scaled = self.sqrt_alpha * (corners @ self.basis)
            self.log_lower = -float(np.max(np.sum(scaled * scaled, axis=1)))
        cutoff = 0
        while (2 * cutoff + 1) ** self.k <= max_terms:
            logs = self._error_logs(cutoff)
            if max(logs) <= math.log(tolerance):
                break
            cutoff += 1
        else:
            raise ValueError("full_winding sum exceeds winding_max_terms before its truncation bounds reach winding_tolerance")
        self.cutoff = cutoff
        self.terms = (2 * cutoff + 1) ** self.k
        self.error_bounds = tuple(_bound_float(value) for value in logs)
        self.indices = np.asarray(list(itertools.product(range(-cutoff, cutoff + 1), repeat=self.k)), dtype=float)
        if self.representation == "direct":
            self.vectors = self.indices @ (self.sqrt_alpha * self.basis)
        elif cutoff:
            # Dimensionless reciprocal vectors g/(2 sqrt(alpha)).
            self.vectors = (self.indices @ u / singular) @ vt * (math.pi / self.sqrt_alpha)
            exponent = np.sum(self.vectors * self.vectors, axis=1)
            self.weights = np.exp(-exponent)
        else:
            self.vectors = np.zeros((1, 3))
            self.weights = np.ones(1)

    def _error_logs(self, cutoff):
        if self.representation == "direct":
            log_t0 = _log_zero_tail(self.c, cutoff, 0.5)
            log_t2 = self._log_square_coefficient + _log_zero_tail(self.c / 2, cutoff, 0.5)
            r0, rsq = _cube_tails(self.k, self.log_s0, self.log_s2, log_t0, log_t2)
            r0 -= self.log_lower
            r2 = self.log_C + rsq - self.log_lower
            log_inside_radius2 = self.log_C + math.log(self.k) + 2 * math.log(cutoff + 0.5)
            e1 = _logadd(0.5 * (r0 + r2), 0.5 * log_inside_radius2 + r0)
            e2 = _logadd(r2, log_inside_radius2 + r0)
        else:
            log_t0 = _log_zero_tail(self.c, cutoff, 0.0)
            log_t2 = _log_square_tail(self.c, cutoff)
            r0, rsq = _cube_tails(self.k, self.log_s0, self.log_s2, log_t0, log_t2)
            total_sq = math.log(self.k) + self.log_s2 + (self.k - 1) * self.log_s0
            tail_first = 0.5 * (self.log_C + r0 + rsq)
            total_first = 0.5 * (self.log_C + self.k * self.log_s0 + total_sq)
            total_second = self.log_C + total_sq
            e1 = _logadd(tail_first, total_first - self.log_lower + r0) - self.log_lower
            e2 = _logadd(self.log_C + rsq, total_second - self.log_lower + r0) - self.log_lower
            r0 -= self.log_lower
        # Adding a deterministic perpendicular vector introduces cross terms.
        # 2|perp|/(1+|perp|^2) <= 1 controls their normalized matrix error.
        return r0, e1, _logadd(e2, e1)

    def evaluate(self, flat):
        projected = flat @ self.vt.T
        fractional = (projected / self.singular) @ self.u.T
        if not np.isfinite(fractional).all() or np.any(np.abs(fractional) > 2 ** 48):
            raise ValueError("displacement is too large for reliable winding fractional coordinates")
        fractional -= np.rint(fractional)
        perpendicular = np.zeros_like(flat) if self.k == 3 else flat - projected @ self.vt
        perp_scaled = self.sqrt_alpha * perpendicular
        count = len(flat)
        log_kernel = np.empty(count)
        mean = np.empty((count, 3))
        second = np.empty((count, 3, 3))
        chunk = max(1, _PAIR_CHUNK // self.terms)
        for first in range(0, count, chunk):
            stop = min(count, first + chunk)
            t = fractional[first:stop]
            perp = perp_scaled[first:stop]
            if self.representation == "direct":
                vectors = t[:, None] @ (self.sqrt_alpha * self.basis) + self.vectors[None]
                exponent = np.sum(vectors * vectors, axis=2)
                offset = np.min(exponent, axis=1)
                weights = np.exp(-(exponent - offset[:, None]))
                total = weights.sum(axis=1)
                weights /= total[:, None]
                parallel_mean = np.einsum("bt,btc->bc", weights, vectors)
                parallel_second = np.einsum("bt,bti,btj->bij", weights, vectors, vectors)
                logarithm = -offset + np.log(total)
            elif self.cutoff == 0:
                parallel_mean = np.zeros_like(perp)
                parallel_second = np.broadcast_to(0.5 * self.projector, (stop - first, 3, 3))
                logarithm = np.full(stop - first, self.log_prefactor)
            else:
                phase = 2 * math.pi * (t @ self.indices.T)
                cos_weight = np.cos(phase) * self.weights
                denominator = np.sum(cos_weight, axis=1)
                if not np.isfinite(denominator).all() or np.any(denominator <= 0):
                    raise ValueError("reciprocal winding kernel lost its positive denominator")
                sine_weight = np.sin(phase) * self.weights
                parallel_mean = (sine_weight @ self.vectors) / denominator[:, None]
                parallel_second = 0.5 * self.projector - np.einsum(
                    "bt,ti,tj->bij", cos_weight, self.vectors, self.vectors) / denominator[:, None, None]
                logarithm = self.log_prefactor + np.log(denominator)
            log_kernel[first:stop] = logarithm - np.sum(perp * perp, axis=1)
            mean[first:stop] = (parallel_mean + perp) / self.sqrt_alpha
            second[first:stop] = (parallel_second + np.einsum("bi,bj->bij", perp, perp)
                + np.einsum("bi,bj->bij", perp, parallel_mean)
                + np.einsum("bi,bj->bij", parallel_mean, perp)) / self.alpha
        return WindingMoments(log_kernel, mean, second)


class GaussianWindingKernel:
    """Cached full-winding link evaluator with a fixed per-link term budget.

    In an orthogonal cell the budget counts the sum of evaluated 1D terms;
    for a skew cell it counts the k-dimensional vector terms. Link batches
    target at most 262144 link/term pairs; a single link with more terms is
    evaluated on its own. The actual upper bound is reported in metadata.
    At most one million vector terms may be cached, independently of the
    requested numerical tolerance, which also bounds the largest pair batch.
    """

    def __init__(self, geometry, alpha, *, tolerance=DEFAULT_WINDING_TOLERANCE,
                 max_terms=DEFAULT_WINDING_MAX_TERMS):
        self.tolerance, self.max_terms = validate_winding_options(tolerance, max_terms, "full_winding")
        if (isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha)
                or not isinstance(alpha, (int, float, np.integer, np.floating))
                or not np.isfinite(alpha) or alpha <= 0):
            raise ValueError("periodic Gaussian alpha must be finite and positive")
        self.alpha = float(alpha)
        if not geometry.pbc.any():
            raise ValueError("full_winding requires at least one periodic direction")
        self.geometry = geometry
        # The MIC geometry deliberately tolerates a small angular error in its
        # orthogonal fast path. A full Gaussian must not discard a real shear:
        # alpha can amplify that error beyond the declared truncation bound.
        # Rotated floating-point cells with nonzero cross products use the
        # general lattice sum; exactly orthogonal rows retain separability.
        gram = geometry.basis @ geometry.basis.T
        self.separable = np.array_equal(gram, np.diag(np.diag(gram)))
        k = len(geometry.basis)
        if self.separable:
            self.plans = tuple(_LatticePlan(row[None], self.alpha,
                self.tolerance / (16 * k), self.max_terms) for row in geometry.basis)
        else:
            self.plans = (_LatticePlan(geometry.basis, self.alpha, self.tolerance, self.max_terms),)
        self.terms = sum(plan.terms for plan in self.plans)
        if self.terms > self.max_terms:
            raise ValueError("separable full_winding sum exceeds winding_max_terms")
        if self.separable:
            errors = np.asarray([plan.error_bounds for plan in self.plans])
            relative = math.expm1(float(np.log1p(errors[:, 0]).sum()))
            first = math.hypot(*errors[:, 1])
            second = float(errors[:, 2].sum()) + 2 * first + 2 * first * first
            self.error_bounds = (relative, first, second)
            self.unit = geometry.basis / np.sqrt(geometry.squared_lengths)[:, None]
        else:
            self.error_bounds = self.plans[0].error_bounds
        if max(self.error_bounds) > self.tolerance:
            raise ValueError("combined winding truncation error exceeds winding_tolerance")
        self.metadata = {
            "target": "infinite_periodic_Gaussian_image_sum",
            "method": "orthogonal_separable" if self.separable else "reduced_skew_lattice",
            "representations": [plan.representation for plan in self.plans],
            "cutoffs": [plan.cutoff for plan in self.plans],
            "terms_per_link": self.terms,
            "term_budget_per_link": self.max_terms,
            "working_pair_budget": max(_PAIR_CHUNK, max(plan.terms for plan in self.plans)),
            "link_batch_pair_target": _PAIR_CHUNK,
            "tolerance": self.tolerance,
            "relative_kernel_truncation_bound": self.error_bounds[0],
            "dimensionless_first_moment_truncation_bound": self.error_bounds[1],
            "scaled_second_moment_truncation_bound": self.error_bounds[2],
            "bound_scope": "one_link_uniform_fractional_cell_excluding_float64_roundoff",
            "second_moment_scale": "alpha_inverse_plus_frobenius_norm_of_computed_second_moment",
        }

    def evaluate(self, displacements):
        value = np.asarray(displacements)
        if value.ndim < 1 or value.shape[-1] != 3 or value.dtype.kind not in "iuf":
            raise ValueError("winding displacements must be real arrays ending in 3")
        flat = np.asarray(value, dtype=float).reshape(-1, 3)
        if not np.isfinite(flat).all():
            raise ValueError("winding displacements must be finite")
        shape = value.shape[:-1]
        if not self.separable:
            result = self.plans[0].evaluate(flat)
        else:
            components = flat @ self.unit.T
            perpendicular = np.zeros_like(flat) if len(self.unit) == 3 else flat - components @ self.unit
            mean = perpendicular.copy()
            second = np.einsum("bi,bj->bij", perpendicular, perpendicular)
            logarithm = -self.alpha * np.sum(perpendicular * perpendicular, axis=1)
            for index, plan in enumerate(self.plans):
                part = plan.evaluate(components[:, index, None] * self.unit[index])
                logarithm += part.log_kernel
                second += part.second_moment + np.einsum("bi,bj->bij", mean, part.mean_displacement)
                second += np.einsum("bi,bj->bij", part.mean_displacement, mean)
                mean += part.mean_displacement
            result = WindingMoments(logarithm, mean, second)
        if (not np.isfinite(result.log_kernel).all()
                or not np.isfinite(result.mean_displacement).all()
                or not np.isfinite(result.second_moment).all()):
            raise ValueError("periodic Gaussian kernel or moments overflow float64")
        return WindingMoments(result.log_kernel.reshape(shape),
            result.mean_displacement.reshape(shape + (3,)),
            result.second_moment.reshape(shape + (3, 3)))


__all__ = ["GaussianWindingKernel", "WindingMoments"]
