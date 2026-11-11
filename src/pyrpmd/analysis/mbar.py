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


def _finite_logsumexp(values, axis=None):
    """Stable real reduction for the solver's checked finite arrays.

    MBAR needs neither signed weights nor complex/array-API dispatch. Avoid
    the extra exponentiation and full-size maximum masks in that general
    path. Exceptional optimizer iterates retain SciPy's nonfinite semantics.
    """
    maximum = np.max(values, axis=axis, keepdims=True)
    if not np.isfinite(maximum).all():
        return logsumexp(values, axis=axis)
    total = np.exp(values - maximum).sum(axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(total), axis=axis)


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
    # Normalized natural-log target weights. ``weights`` underflows to zero
    # for frames far below the dominant weight; histogram estimators that
    # must not silently lose such mass accumulate in this log domain.
    log_weights: np.ndarray | None = None


def _reweighting_signature(reduced_potentials, counts, overlap_threshold):
    """Bind reusable target weights to the exact ordered bias problem."""
    digest = hashlib.sha256()
    for value, dtype in ((reduced_potentials, np.float64), (counts, np.int64)):
        from .mbar_problem import ReducedPotentials
        if isinstance(value, ReducedPotentials):
            digest.update(str(value.shape).encode("ascii"))
            for row in range(value.shape[0]):
                selected = value.select(rows=[row])
                for start in range(0, value.shape[1], 4096):
                    array = np.ascontiguousarray(selected.block(start, min(start+4096, value.shape[1])))
                    digest.update(memoryview(array).cast("B"))
            continue
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
    backend="auto",
    threads=1,
):
    """Return MBAR target weights, state overlap and identifiability diagnostics.

    u_kn[K,N] are dimensionless potentials, counts[K] sum to N; only sampled
    states are passed. target_u[N] defaults to zero after cancelling the same
    physical Hamiltonian from every sampling/target reduced potential. Exact
    duplicate rows are combined internally, then all state-shaped outputs are
    expanded to the original order. Time streams are never combined here.
    ``return_state_weights=False`` streams the overlap accumulation and omits
    every [N,K] state-weight array. ``backend='auto'`` uses the optional native
    C++ kernel, releasing the GIL, or bounded NumPy blocks when unavailable.
    ``threads`` controls one native solve and is capped by CPU affinity.
    The default preserves the full public state-weight result when requested.

    ``method='auto'`` uses a gauge-constrained damped Newton solve, with the original
    L-BFGS/fixed-point procedure as fallback. ``method='legacy'`` is retained for
    independent numerical comparisons. Both require the same final fixed-point
    residual, regardless of optimizer termination. A warm start changes only
    convergence speed, not the reduced potentials or target distribution.
    """
    from .mbar_problem import ReducedPotentials
    u = u_kn if isinstance(u_kn, ReducedPotentials) else np.asarray(u_kn, float)
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
    if isinstance(u, ReducedPotentials):
        finite = all(np.isfinite(a).all() for a in (u.values,u.centers,u.stiffness,u.periods))
    else:
        finite = all(np.isfinite(row).all() for row in u)
    if not finite:
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
        or backend not in ("auto", "native", "numpy")
        or isinstance(threads, (bool, np.bool_))
        or not isinstance(threads, (int, np.integer)) or threads < 1
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
    if method != "legacy":
        from .mbar_streaming import solve
        result = solve(u, counts, np.ascontiguousarray(target), initial, tolerance=tolerance,
                       max_iterations=max_iterations, overlap_threshold=overlap_threshold,
                       combine_identical_states=combine_identical_states,
                       return_state_weights=return_state_weights, backend=backend, threads=int(threads),
                       warm_start=initial_free_energies is not None)
        result.diagnostics["warm_start"] = initial_free_energies is not None
        return result
    u = np.asarray(u)
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
    reduce_log = logsumexp if method == "legacy" else _finite_logsumexp

    def equations(z):
        logd = reduce_log(logn[:, None] + z[:, None] - u, axis=0)
        new = -reduce_log(-u - logd[None], axis=1)
        new -= new[0]
        return new, logd, float(np.max(np.abs(new - z)))

    # A short initialization is followed by Newton; long fixed-point mixing
    # across narrow umbrellas was the principal cost in the previous solver.
    warm_default = 200 if method == "legacy" else (8 if initial_free_energies is None else 2)
    warm_steps = min(warm_default, max_iterations)
    for _ in range(warm_steps):
        new, _, residual = equations(f)
        if residual < tolerance:
            break
        f = new
    lbfgs_iterations = 0
    if k > 1 and equations(f)[2] >= tolerance:

        def objective(x):
            z = np.concatenate(([0.0], x))
            logd = reduce_log(logn[:, None] + z[:, None] - u, axis=0)
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
    logw = logw - reduce_log(logw)
    weights = np.exp(logw)
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
            "method": "single_state" if k == 1 else "lbfgs_fixed_point",
            "unique_states": k,
            "original_states": len(original_counts),
            "state_to_unique": state_map,
            "newton_iterations": 0,
            "lbfgs_iterations": lbfgs_iterations,
            "polish_iterations": polish_iterations,
            "warm_start": initial_free_energies is not None,
        },
        logw,
    )
