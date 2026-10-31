"""Internal matrix-path algebra for the model nonadiabatic APIs.

Every derivative is analytic.  Positive scalar rescalings are numerical
bookkeeping only: derivatives of the original matrices, divided by those same
scales, are propagated.  Prefix/suffix products keep the bead cost linear.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import exprel

from .nonadiabatic import DiabaticResult, validate_diabatic_result
from .potentials.base import PotentialError, coordinates


def integrator_signature(integrator):
    """Configuration affecting an already-created native or NumPy ring map."""
    return (integrator.beta, integrator.nbeads, integrator.dt, integrator.scheme,
            integrator.threads, integrator.native, tuple(integrator.masses))


def real_diabatic(potential, q, nstates=None):
    """Evaluate a fixed, real diabatic basis, without dropping imaginary terms."""
    # Providers borrow coordinates; own this buffer so a faulty provider cannot
    # mutate an integrator state during an energy/force query.
    q = np.array(coordinates(q), copy=True)
    result = validate_diabatic_result(potential.evaluate(q), q.shape, nstates)
    if (np.any(np.imag(result.matrix) != 0)
            or np.any(np.imag(result.gradients) != 0)):
        raise PotentialError("this method requires a real symmetric diabatic matrix")
    return DiabaticResult(np.array(result.matrix.real, copy=True),
                         np.array(result.gradients.real, copy=True))


@dataclass(frozen=True)
class ExponentialPath:
    matrices: np.ndarray
    gradients: np.ndarray
    log_scales: np.ndarray
    common: np.ndarray
    common_gradients: np.ndarray


def exponential_path(result, tau):
    """Scaled exp(-tau*(V-Tr(V)/S)), including its Frechet derivative.

The common scalar potential is returned separately to avoid cancellation of
large energy offsets.  No eigenvector differentiation or gap division occurs.
    """
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    matrix, gradients = result.matrix, result.gradients
    nstates = matrix.shape[1]
    common = np.trace(matrix, axis1=1, axis2=2) / nstates
    common_gradients = np.einsum("biiac->bac", gradients) / nstates
    eye = np.eye(nstates)
    centered = matrix - common[:, None, None] * eye
    derivative = gradients - np.einsum("ij,bac->bijac", eye, common_gradients)
    energies, vectors = np.linalg.eigh(centered)
    shifted = energies - energies[:, :1]
    values = np.exp(-tau * shifted)
    # Divided differences of exp(-tau E).  Use the lower of the two energies
    # so that exprel only receives nonpositive arguments, including degeneracy.
    gap = np.abs(energies[:, :, None] - energies[:, None, :])
    lower = np.minimum(shifted[:, :, None], shifted[:, None, :])
    divided = -tau * np.exp(-tau * lower) * exprel(-tau * gap)
    matrices = np.einsum("bsi,bi,bti->bst", vectors, values, vectors,
                         optimize=True)
    transformed = np.einsum("bsi,bstac,btj->bijac", vectors, derivative,
                            vectors, optimize=True)
    dmatrices = np.einsum("bsi,bijac,btj->bstac", vectors,
                          divided[..., None, None] * transformed,
                          vectors, optimize=True)
    log_scales = -tau * energies[:, 0]
    if not (np.isfinite(matrices).all() and np.isfinite(dmatrices).all()
            and np.isfinite(log_scales).all()):
        raise PotentialError("matrix exponential or derivative overflow")
    return ExponentialPath(matrices, dmatrices, log_scales,
                           common, common_gradients)


class ScaledProduct:
    """Ordered matrix product and linear-cost cyclic derivative contractions."""

    def __init__(self, factors):
        factors = np.asarray(factors)
        if (factors.ndim != 3 or not len(factors)
                or factors.shape[1] != factors.shape[2]
                or not np.isfinite(factors).all()):
            raise PotentialError("invalid matrix path factors")
        self.factors = factors
        count, size, _ = factors.shape
        self.prefix = np.empty((count + 1, size, size), dtype=factors.dtype)
        self.suffix = np.empty_like(self.prefix)
        self.prefix_logs = np.zeros(count + 1)
        self.suffix_logs = np.zeros(count + 1)
        self.prefix[0] = np.eye(size)
        self.suffix[count] = np.eye(size)
        for index in range(count):
            value = self.prefix[index] @ factors[index]
            scale = float(np.max(np.abs(value)))
            if not np.isfinite(scale) or scale <= 0:
                raise PotentialError("matrix path has zero or unresolvable weight")
            self.prefix[index + 1] = value / scale
            self.prefix_logs[index + 1] = self.prefix_logs[index] + np.log(scale)
        for index in range(count - 1, -1, -1):
            value = factors[index] @ self.suffix[index + 1]
            scale = float(np.max(np.abs(value)))
            if not np.isfinite(scale) or scale <= 0:
                raise PotentialError("matrix path has zero or unresolvable weight")
            self.suffix[index] = value / scale
            self.suffix_logs[index] = self.suffix_logs[index + 1] + np.log(scale)
        self.matrix = self.prefix[-1]
        self.log_scale = float(self.prefix_logs[-1])

    def holes(self):
        """Cyclic products scaled by the scale of the full ordered product."""
        logs = self.prefix_logs[:-1] + self.suffix_logs[1:] - self.log_scale
        with np.errstate(over="ignore", invalid="ignore"):
            holes = (self.suffix[1:] @ self.prefix[:-1]) * np.exp(logs)[:, None, None]
        if not np.isfinite(holes).all():
            raise PotentialError("ill-conditioned matrix-path derivative")
        return holes

    def insertion(self, index, derivative):
        """Full-product-scaled insertion, retaining any trailing derivative axes."""
        log_scale = (self.prefix_logs[index] + self.suffix_logs[index + 1]
                     - self.log_scale)
        with np.errstate(over="ignore", invalid="ignore"):
            value = np.einsum("ij,jk...,kl->il...", self.prefix[index],
                              derivative, self.suffix[index + 1], optimize=True)
            value *= np.exp(log_scale)
        if not np.isfinite(value).all():
            raise PotentialError("ill-conditioned matrix-path derivative")
        return value


def positive_trace_path(path):
    """Return centered log trace and its nuclear log derivative."""
    product = ScaledProduct(path.matrices)
    trace = float(np.trace(product.matrix))
    if not np.isfinite(trace) or trace <= 0:
        raise PotentialError("electronic matrix path must have positive trace")
    gradient = np.einsum("bijac,bji->bac", path.gradients,
                         product.holes(), optimize=True) / trace
    log_trace = np.log(trace) + product.log_scale + float(path.log_scales.sum())
    if not np.isfinite(gradient).all() or not np.isfinite(log_trace):
        raise PotentialError("electronic matrix-path trace is unresolvable")
    return float(log_trace), gradient

