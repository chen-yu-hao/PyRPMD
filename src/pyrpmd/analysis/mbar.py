"""MBAR equations, solver diagnostics, and reusable umbrella weights.

Equations: Shirts & Chodera, JCP 129,124105 (2008),
doi:10.1063/1.2978177. Implemented directly with SciPy, not copied from
PyMBAR. A shared physical reduced potential cancels only for equal T,
bead count, PES, and Hamiltonian across all windows.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.csgraph import connected_components
from scipy.special import logsumexp


@dataclass
class MBARResult:
    free_energies: np.ndarray
    weights: np.ndarray
    state_weights: np.ndarray | None
    overlap: np.ndarray
    components: np.ndarray
    converged: bool
    residual: float
    diagnostics: dict | None = None


def _reweighting_signature(reduced_potentials, counts, overlap_threshold):
    """Bind reusable target weights to the exact ordered bias problem."""
    digest = hashlib.sha256()
    for value, dtype in ((reduced_potentials, np.float64), (counts, np.int64)):
        array = np.ascontiguousarray(value, dtype=dtype)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    digest.update(float(overlap_threshold).hex().encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class UmbrellaReweighting:
    """Compact MBAR result reusable across histogram/binning choices.

    This holds no per-state/per-sample matrix. The fingerprint verifies the
    full ordered reduced potentials and stream counts before target weights
    are reused; bins, support masks and basin estimates are always recomputed.
    """

    signature: str
    free_energies: np.ndarray
    weights: np.ndarray
    overlap: np.ndarray
    components: np.ndarray
    converged: bool
    residual: float
    diagnostics: dict | None

    def __post_init__(self):
        for name in ("free_energies", "weights", "overlap", "components"):
            value = np.ascontiguousarray(getattr(self, name))
            # Immutable backing survives attempts to re-enable writes. The
            # constructor also runs on process/MPI deserialization below.
            snapshot = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
            object.__setattr__(self, name, snapshot)
        object.__setattr__(self, "diagnostics", copy.deepcopy(self.diagnostics))

    def __reduce__(self):
        return (type(self), (self.signature, self.free_energies, self.weights,
                            self.overlap, self.components, self.converged,
                            self.residual, self.diagnostics))

    @classmethod
    def from_result(cls, result, reduced_potentials, counts, overlap_threshold):
        return cls(_reweighting_signature(reduced_potentials, counts, overlap_threshold),
                   result.free_energies, result.weights, result.overlap, result.components,
                   result.converged, result.residual, result.diagnostics)

    def validate_problem(self, reduced_potentials, counts, overlap_threshold):
        if self.signature != _reweighting_signature(reduced_potentials, counts, overlap_threshold):
            raise ValueError("MBAR reweighting belongs to different samples, umbrellas or overlap settings")


def _identical_rows(rows):
    """Exact row groups in first-occurrence order; verify hash collisions."""
    representatives, inverse, buckets = [], [], {}
    for i, row in enumerate(rows):
        key = hash(np.ascontiguousarray(row).tobytes())
        group = next(
            (
                g
                for g in buckets.get(key, ())
                if np.array_equal(row, rows[representatives[g]])
            ),
            None,
        )
        if group is None:
            group = len(representatives)
            representatives.append(i)
            buckets.setdefault(key, []).append(group)
        inverse.append(group)
    return np.asarray(representatives), np.asarray(inverse)


def solve_mbar(
    u_kn,
    counts,
    target_u=None,
    tolerance=1e-9,
    max_iterations=10000,
    overlap_threshold=1e-6,
    *,
    initial_free_energies=None,
    combine_identical_states=True,
    method="auto",
    return_state_weights=True,
):
    """Return MBAR target weights, state overlap and identifiability diagnostics.

    u_kn[K,N] are dimensionless potentials, counts[K] sum to N; only sampled
    states are passed. target_u[N] defaults to zero after cancelling the same
    physical Hamiltonian from every sampling/target reduced potential. Exact
    duplicate rows are combined internally, then all state-shaped outputs are
    expanded to the original order. Time streams are never combined here.
    ``return_state_weights=False`` omits the expanded [N,K] state-weight array
    and returns ``state_weights=None``. Target weights, overlap, free energies
    and diagnostics are unchanged; unique-state weights are still computed
    internally for overlap. The default preserves the full public result.

    ``auto`` uses a gauge-constrained damped Newton solve, with the original
    L-BFGS/fixed-point procedure as fallback. ``legacy`` is retained for
    independent numerical comparisons. Both require the same final fixed-point
    residual, regardless of optimizer termination. A warm start changes only
    convergence speed, not the reduced potentials or target distribution.
    """
    u = np.asarray(u_kn, float)
    raw_counts = np.asarray(counts)
    if raw_counts.dtype.kind not in "iuf" or not np.isfinite(raw_counts).all():
        raise ValueError("MBAR counts must be finite positive integers")
    counts = raw_counts.astype(int)
    if (
        u.ndim != 2
        or min(u.shape) == 0
        or counts.shape != (len(u),)
        or np.any(counts <= 0)
        or counts.sum() != u.shape[1]
        or not np.array_equal(counts, raw_counts)
    ):
        raise ValueError("MBAR requires u[K,N], positive counts[K] summing to N")
    if not np.isfinite(u).all():
        raise ValueError("MBAR energies must be finite")
    target = np.zeros(u.shape[1]) if target_u is None else np.asarray(target_u, float)
    if target.shape != (u.shape[1],) or not np.isfinite(target).all():
        raise ValueError("target reduced potential must be finite [N]")
    if (
        not np.isfinite(tolerance)
        or tolerance <= 0
        or isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations < 1
        or not np.isfinite(overlap_threshold)
        or overlap_threshold < 0
        or method not in ("auto", "legacy")
    ):
        raise ValueError("invalid MBAR solver options")
    original_counts = counts
    initial = (
        np.zeros(len(counts))
        if initial_free_energies is None
        else np.asarray(initial_free_energies, float)
    )
    if initial.shape != counts.shape or not np.isfinite(initial).all():
        raise ValueError("initial_free_energies must be finite [K]")
    if combine_identical_states:
        representatives, state_map = _identical_rows(u)
    else:
        representatives = state_map = np.arange(len(counts))
    counts = np.bincount(state_map, weights=counts).astype(int)
    u = u[representatives]
    f = initial[representatives].copy()
    f -= f[0]
    logn = np.log(counts)
    # A per-frame common offset improves conditioning without changing weights.
    common = np.min(u, axis=0)
    u = u - common[None]
    target = target - common
    k = len(counts)

    def equations(z):
        logd = logsumexp(logn[:, None] + z[:, None] - u, axis=0)
        new = -logsumexp(-u - logd[None], axis=1)
        new -= new[0]
        return new, logd, float(np.max(np.abs(new - z)))

    # A short initialization is followed by Newton; long fixed-point mixing
    # across narrow umbrellas was the principal cost in the previous solver.
    warm_steps = min(200 if method == "legacy" else 8, max_iterations)
    for _ in range(warm_steps):
        new, _, residual = equations(f)
        if residual < tolerance:
            break
        f = new
    newton_iterations = 0
    fallback = method == "legacy"
    if method == "auto" and k > 1:
        for newton_iterations in range(1, min(80, max_iterations) + 1):
            _, logd, residual = equations(f)
            if residual < tolerance:
                break
            probabilities = np.exp(logn[:, None] + f[:, None] - u - logd)
            totals = probabilities.sum(axis=1)
            gradient = totals - counts
            hessian = np.diag(totals) - probabilities @ probabilities.T
            try:
                step = np.linalg.solve(hessian[1:, 1:], gradient[1:])
            except np.linalg.LinAlgError:
                fallback = True
                break
            descent = float(gradient[1:] @ step)
            if not np.isfinite(step).all() or not np.isfinite(descent) or descent <= 0:
                fallback = True
                break
            value = float(logd.sum() - counts @ f)
            alpha = min(1.0, 20.0 / max(float(np.max(np.abs(step))), 1e-30))
            accepted = False
            rounding = 16 * np.finfo(float).eps * max(1.0, abs(value))
            for _ in range(24):
                candidate = f.copy()
                candidate[1:] -= alpha * step
                _, candidate_logd, candidate_residual = equations(candidate)
                candidate_value = float(candidate_logd.sum() - counts @ candidate)
                armijo = candidate_value <= value - 1e-4 * alpha * descent
                near_roundoff = (
                    candidate_value <= value + rounding
                    and candidate_residual < residual
                )
                if np.isfinite(candidate_value) and (armijo or near_roundoff):
                    f = candidate
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                fallback = True
                break
        if equations(f)[2] >= tolerance:
            fallback = True
    lbfgs_iterations = 0
    if k > 1 and fallback and equations(f)[2] >= tolerance:

        def objective(x):
            z = np.concatenate(([0.0], x))
            logd = logsumexp(logn[:, None] + z[:, None] - u, axis=0)
            memberships = np.exp(logn[:, None] + z[:, None] - u - logd)
            value = float(logd.sum() - counts @ z)
            grad = memberships.sum(axis=1) - counts
            return value, grad[1:]

        fit = minimize(
            objective,
            f[1:],
            method="L-BFGS-B",
            jac=True,
            options={
                "ftol": 1e-14,
                "gtol": tolerance,
                "maxiter": max_iterations,
                "maxls": 40,
            },
        )
        f = np.concatenate(([0.0], fit.x))
        lbfgs_iterations = int(fit.nit)
    # Independently certify the exact MBAR equations, including fallback cases.
    polish_iterations = 0
    for polish_iterations in range(max_iterations):
        new, _, residual = equations(f)
        if residual < tolerance:
            break
        f = new
    _, logd, residual = equations(f)
    unique_weights = np.exp(f[:, None] - u - logd[None]).T
    logw = -target - logd
    weights = np.exp(logw - logsumexp(logw))
    gram = unique_weights.T @ unique_weights
    overlap = gram[state_map[:, None], state_map[None, :]] * original_counts[None, :]
    graph = np.maximum(overlap, overlap.T) > overlap_threshold
    _, components = connected_components(graph, directed=False)
    return MBARResult(
        f[state_map],
        weights,
        unique_weights[:, state_map] if return_state_weights else None,
        overlap,
        components,
        residual < tolerance,
        residual,
        {
            "method": (
                "single_state"
                if k == 1
                else ("lbfgs_fixed_point" if fallback else "damped_newton")
            ),
            "unique_states": k,
            "original_states": len(original_counts),
            "state_to_unique": state_map,
            "newton_iterations": newton_iterations,
            "lbfgs_iterations": lbfgs_iterations,
            "polish_iterations": polish_iterations,
            "warm_start": initial_free_energies is not None,
        },
    )
