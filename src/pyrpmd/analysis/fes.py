"""MBAR and multidimensional histogram FES, with correlated block bootstrap.

MBAR equations: Shirts & Chodera, JCP 129,124105 (2008),
doi:10.1063/1.2978177. The solver here is implemented directly with SciPy;
it is not copied from PyMBAR. A shared physical reduced potential cancels
only for equal T, bead count, PES, and Hamiltonian across all windows.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.sparse.csgraph import connected_components
from ..bias import displacement


def statistical_inefficiency(series):
    """Initial-positive-sequence estimate; a diagnostic, not proof of mixing."""
    x = np.asarray(series, float)
    if x.ndim == 2:
        return max(statistical_inefficiency(x[:, i]) for i in range(x.shape[1]))
    n = len(x)
    if n < 4:
        return 1.0
    x = x - x.mean()
    var = float(x @ x / n)
    if var <= np.finfo(float).tiny:
        return float(n)
    nfft = 1 << (2 * n - 1).bit_length()
    spec = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(spec * spec.conj(), nfft)[:n]
    ac = ac / (np.arange(n, 0, -1) * var)
    total = 0.0
    for lag in range(1, n - 1, 2):
        pair = ac[lag] + ac[lag + 1]
        if not np.isfinite(pair) or pair <= 0:
            break
        total += pair
    return float(np.clip(1 + 2 * total, 1, n))


@dataclass
class MBARResult:
    free_energies: np.ndarray
    weights: np.ndarray
    state_weights: np.ndarray
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
):
    """Return MBAR target weights, state overlap and identifiability diagnostics.

    u_kn[K,N] are dimensionless potentials, counts[K] sum to N; only sampled
    states are passed. target_u[N] defaults to zero after cancelling the same
    physical Hamiltonian from every sampling/target reduced potential. Exact
    duplicate rows are combined internally, then all state-shaped outputs are
    expanded to the original order. Time streams are never combined here.

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
        unique_weights[:, state_map],
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


def umbrella_reduced_potentials(samples, centers, stiffness, beta, periods=None):
    x = np.asarray(samples, float)
    centers = np.asarray(centers, float)
    stiffness = np.asarray(stiffness, float)
    if stiffness.ndim == 2:
        stiffness = np.broadcast_to(stiffness, (len(centers),) + stiffness.shape)
    values = []
    for center, k in zip(centers, stiffness):
        delta = displacement(x, center, periods)
        values.append(beta * 0.5 * np.einsum("ni,ij,nj->n", delta, k, delta))
    return np.asarray(values)


def _histogram(x, weights, edges, beta, min_count=1):
    mass, _ = np.histogramdd(x, bins=edges, weights=weights)
    count, _ = np.histogramdd(x, bins=edges)
    volumes = np.ones(tuple(len(e) - 1 for e in edges))
    for d, e in enumerate(edges):
        shape = [1] * len(edges)
        shape[d] = len(e) - 1
        volumes *= np.diff(e).reshape(shape)
    density = mass / volumes
    mask = (count < min_count) | (mass <= 0)
    f = np.full_like(mass, np.nan)
    f[~mask] = -np.log(density[~mask]) / beta
    if np.any(~mask):
        f[~mask] -= np.nanmin(f)
    return mass, density, count, f, mask


def _block_resample(x, block, rng):
    n = len(x)
    block = max(1, min(int(block), n))
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    ix = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
    return x[ix]


def reconstruct_fes(
    samples,
    centers,
    stiffness,
    beta,
    bins=40,
    ranges=None,
    periods=None,
    bootstrap=20,
    seed=0,
    min_count=1,
    overlap_threshold=1e-6,
    return_reweighting=False,
):
    """Reconstruct 1D/2D density w.r.t. ds1...dsD, not bin probability mass.

    Bootstrap resamples contiguous circular time blocks independently inside
    each window/replica. No unsupported region is filled by interpolation.
    Disconnected overlap components are reported; a single global FES is then
    masked because its relative offset is not identifiable.
    """
    arrays = [np.asarray(a, float) for a in samples]
    if not arrays or any(a.ndim != 2 or not len(a) for a in arrays):
        raise ValueError("samples must contain nonempty [Ns,D] arrays")
    d = arrays[0].shape[1]
    if d not in (1, 2) or any(
        a.shape[1] != d or not np.isfinite(a).all() for a in arrays
    ):
        raise ValueError("FES supports finite one- or two-dimensional samples")
    centers = np.asarray(centers, float)
    if centers.shape != (len(arrays), d):
        raise ValueError("one center per sampling stream required")
    k = np.asarray(stiffness, float)
    if k.ndim == 2:
        k = np.broadcast_to(k, (len(arrays), d, d))
    if k.shape != (len(arrays), d, d):
        raise ValueError("stiffness shape mismatch")
    if any(not np.allclose(a, a.T) or np.linalg.eigvalsh(a).min() <= 0 for a in k):
        raise ValueError("all umbrella matrices must be symmetric positive definite")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive")
    if (
        isinstance(min_count, bool)
        or not isinstance(min_count, (int, np.integer))
        or min_count < 1
    ):
        raise ValueError("min_count must be a positive integer")
    x = np.concatenate(arrays)
    counts = np.array([len(a) for a in arrays])
    if np.isscalar(bins):
        bins = [bins] * d
    if len(bins) != d:
        raise ValueError("bins must have one entry per CV")
    if ranges is None:
        ranges = [(float(x[:, j].min()), float(x[:, j].max())) for j in range(d)]
    edges = []
    for j, b in enumerate(bins):
        if np.isscalar(b):
            lo, hi = ranges[j]
            if (
                isinstance(b, bool)
                or not isinstance(b, (int, np.integer))
                or b < 2
                or not hi > lo
            ):
                raise ValueError("histogram needs integer >=2 bins and nonzero range")
            edges.append(np.linspace(lo, hi, int(b) + 1))
        else:
            edge = np.asarray(b, float)
            if edge.ndim != 1 or len(edge) < 3 or np.any(np.diff(edge) <= 0):
                raise ValueError("invalid bin edges")
            edges.append(edge)
    u = umbrella_reduced_potentials(x, centers, k, beta, periods)
    mbar = solve_mbar(u, counts, overlap_threshold=overlap_threshold)
    if not mbar.converged:
        raise RuntimeError(
            f"MBAR did not converge; equation residual={mbar.residual:g}"
        )
    mass, density, count, f, mask = _histogram(x, mbar.weights, edges, beta, min_count)
    gs = np.array([statistical_inefficiency(a) for a in arrays])
    se = np.full_like(f, np.nan)
    connected = len(set(mbar.components)) == 1
    boots = []
    if (
        isinstance(bootstrap, bool)
        or not isinstance(bootstrap, (int, np.integer))
        or bootstrap < 0
        or bootstrap == 1
    ):
        raise ValueError("bootstrap must be integer zero or >=2")
    if connected and bootstrap and np.any(~mask):
        rng = np.random.default_rng(seed)
        ref = np.unravel_index(np.nanargmin(f), f.shape)
        for _ in range(int(bootstrap)):
            draws = [
                _block_resample(a, int(np.ceil(g)), rng) for a, g in zip(arrays, gs)
            ]
            xb = np.concatenate(draws)
            ub = umbrella_reduced_potentials(xb, centers, k, beta, periods)
            mb = solve_mbar(
                ub,
                counts,
                overlap_threshold=overlap_threshold,
                initial_free_energies=mbar.free_energies,
            )
            if not mb.converged or len(set(mb.components)) > 1:
                continue
            _, _, _, fb, _ = _histogram(xb, mb.weights, edges, beta, min_count)
            if np.isfinite(fb[ref]):
                boots.append(fb - fb[ref])
        if len(boots) >= 2:
            b = np.asarray(boots)
            valid = np.isfinite(b).sum(axis=0)
            means = np.divide(
                np.nansum(b, axis=0), valid, out=np.zeros_like(f), where=valid > 0
            )
            square = np.nansum((b - means) ** 2, axis=0)
            np.sqrt(
                np.divide(square, valid - 1, out=se, where=valid > 1),
                out=se,
                where=valid > 1,
            )
    if not connected:
        # Conditional component surfaces may be reconstructed separately, but
        # displaying arbitrary offsets as one FES would be scientifically wrong.
        f[:] = np.nan
        mask[:] = True
        se[:] = np.nan
    ess = 1 / float(np.sum(mbar.weights**2))
    result = {
        "free_energy": f,
        "probability_mass": mass,
        "density": density,
        "standard_error": se,
        "mask": mask,
        "raw_count": count,
        "edges": edges,
        "overlap": mbar.overlap,
        "components": mbar.components,
        "connected": connected,
        "window_free_energies": mbar.free_energies,
        "statistical_inefficiency": gs,
        "effective_samples_per_stream": counts / gs,
        "weight_ess": ess,
        "bootstrap_replicates": len(boots),
        "weights": mbar.weights,
        "measure": "cv_density",
        "energy_unit": "hartree",
    }
    if return_reweighting:
        result["_reweighting"] = UmbrellaReweighting.from_result(mbar, u, counts, overlap_threshold)
    return result


def _rate_bracket(grid, mask, point, name):
    exact = np.flatnonzero(np.isclose(grid, point, rtol=0, atol=1e-14))
    if len(exact):
        i = int(exact[0])
        return ((i, i, 0.0), None) if not mask[i] else (None, name + "_masked")
    right = int(np.searchsorted(grid, point))
    if right == 0 or right == len(grid):
        return None, name + "_outside_histogram_centers"
    left = right - 1
    if mask[left] or mask[right]:
        return None, name + "_masked"
    return (left, right, float((point - grid[left]) / (grid[right] - grid[left]))), None


def _rate_value(values, bracket):
    left, right, fraction = bracket
    return float((1 - fraction) * values[left] + fraction * values[right])


def _rate_support(grid, mask, reference, surface):
    ref, reason = _rate_bracket(grid, mask, reference, "reference")
    if reason:
        return ref, None, reason
    target, reason = _rate_bracket(grid, mask, surface, "dividing_surface")
    if reason:
        return ref, target, reason
    lo, hi = min(ref[0], target[0]), max(ref[1], target[1])
    if mask[lo : hi + 1].any():
        return ref, target, "masked_interval_between_endpoints"
    return ref, target, None


def _rate_draw_statistics(values):
    values = np.asarray(values, float)
    return {
        "valid": len(values),
        "standard_error_hartree": (
            float(values.std(ddof=1)) if len(values) >= 2 else None
        ),
        "percentile_95_ci_hartree": (
            np.quantile(values, [0.025, 0.975]).tolist() if len(values) >= 2 else None
        ),
    }


def mbar_rate_pmf(
    samples,
    centers,
    stiffness,
    beta,
    *,
    xi_star=None,
    reference=0.0,
    bins=100,
    ranges=None,
    search_range=None,
    bootstrap=40,
    seed=0,
    min_count=5,
    overlap_threshold=1e-6,
    measure="rate_flux",
    compare_ui=False,
    ui_points=401,
    solver_options=None,
):
    """Histogram-MBAR rate PMF in the *already sampled* rate-flux measure.

    Samples are independent time streams, each [N] or [N,1], in the common
    ensemble exp(-beta*H_P/P)*f_xi*exp(-beta*W_i). Only beta*W_i is removed:
    no additional P, metric, radial Jacobian, or physical-energy factor occurs.
    Identical umbrella Hamiltonians are merged exactly for the MBAR equations;
    circular block resampling remains independent within every original stream.

    Endpoints are linearly interpolated in log histogram density between
    neighboring supported bin centers. A masked gap anywhere between reference
    and surface is rejected. With xi_star=None, select the unsmoothed maximum
    bin center in search_range (default [reference,max(centers)]), requiring the
    entire search interval to be supported. Each bootstrap keeps that surface
    fixed; its uncertainty is conditional on this selection, not peak-location
    uncertainty. Unsupported point estimates return ready=False; invalid input
    raises ValueError. Missing uncertainties remain None/NaN, never zero.

    compare_ui adds a cheap Gaussian-UI calculation on the *same* bootstrap
    draws, exposing paired estimator differences without a second MBAR solve.
    Numerical convergence is not physical or sampling convergence.
    """
    if measure != "rate_flux":
        raise ValueError("mbar_rate_pmf requires samples in the rate_flux measure")
    arrays = []
    for sample in samples:
        a = np.asarray(sample, float)
        if a.ndim == 2 and a.shape[1] == 1:
            a = a[:, 0]
        if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
            raise ValueError(
                "rate PMF samples must be finite scalar streams with >=2 points"
            )
        arrays.append(a)
    if not arrays:
        raise ValueError("at least one sampling stream is required")
    centers = np.asarray(centers, float)
    if centers.shape not in ((len(arrays),), (len(arrays), 1)):
        raise ValueError("one scalar center per time stream is required")
    centers = centers.reshape(-1)
    stiffness = np.asarray(stiffness, float)
    if stiffness.shape not in (
        (),
        (1,),
        (1, 1),
        (len(arrays),),
        (len(arrays), 1),
        (len(arrays), 1, 1),
    ):
        raise ValueError("stiffness must be scalar or one scalar matrix per stream")
    if stiffness.size == 1:
        stiffness = np.full(len(arrays), float(stiffness.reshape(-1)[0]))
    else:
        stiffness = stiffness.reshape(-1)
    if (
        centers.shape != (len(arrays),)
        or stiffness.shape != centers.shape
        or not np.isfinite(centers).all()
        or not np.isfinite(stiffness).all()
        or np.any(stiffness <= 0)
        or not np.isfinite(beta)
        or beta <= 0
        or not np.isfinite(reference)
    ):
        raise ValueError(
            "invalid scalar umbrella centers, stiffness, beta, or reference"
        )
    for value, minimum, name in [
        (bootstrap, 0, "bootstrap"),
        (min_count, 1, "min_count"),
        (ui_points, 2, "ui_points"),
    ]:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(name + " must be an integer >= " + str(minimum))
    if bootstrap == 1:
        raise ValueError("bootstrap must be zero or >=2")
    if xi_star is not None and not np.isfinite(xi_star):
        raise ValueError("xi_star must be finite")
    options = dict(solver_options or {})
    if set(options) - {"tolerance", "max_iterations", "method"}:
        raise ValueError(
            "solver_options supports tolerance, max_iterations, and method"
        )
    counts = np.asarray([len(a) for a in arrays])
    offsets = np.concatenate(([0], np.cumsum(counts)))
    x = np.concatenate(arrays)
    raw_bins = np.asarray(bins)
    if raw_bins.ndim == 2 and raw_bins.shape[0] == 1:
        bins = raw_bins[0]
    elif raw_bins.ndim == 1 and raw_bins.size == 1:
        bins = raw_bins[0]
    if np.isscalar(bins):
        if (
            isinstance(bins, (bool, np.bool_))
            or not isinstance(bins, (int, np.integer))
            or bins < 3
        ):
            raise ValueError("bins must be an integer >=3 or explicit edges")
        limits = np.asarray(
            [x.min(), x.max()] if ranges is None else ranges, float
        ).reshape(-1)
        if (
            limits.shape != (2,)
            or not np.isfinite(limits).all()
            or limits[1] <= limits[0]
        ):
            raise ValueError("ranges must contain one finite increasing interval")
        edges = np.linspace(*limits, int(bins) + 1)
    else:
        edges = np.asarray(bins, float)
        if (
            edges.ndim != 1
            or len(edges) < 4
            or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)
        ):
            raise ValueError("explicit bins must contain >=4 increasing finite edges")
    grid = 0.5 * (edges[1:] + edges[:-1])
    representatives, state_map = _identical_rows(np.column_stack((centers, stiffness)))
    fit_counts = np.bincount(state_map, weights=counts).astype(int)
    u = umbrella_reduced_potentials(
        x[:, None],
        centers[representatives, None],
        stiffness[representatives, None, None],
        beta,
    )
    fit = solve_mbar(u, fit_counts, overlap_threshold=overlap_threshold, **options)
    connected = len(np.unique(fit.components)) == 1
    mass, density, count, free, mask = _histogram(
        x[:, None], fit.weights, [edges], beta, min_count
    )
    absolute = np.full_like(free, np.nan)
    absolute[~mask] = -np.log(density[~mask]) / beta
    warnings = []
    reason = None
    if not fit.converged:
        reason = "mbar_not_converged"
    elif not connected:
        reason = "disconnected_overlap"
    selection = "fixed" if xi_star is not None else "maximum_supported_bin_center"
    auto_range = None
    if xi_star is None:
        auto_range = np.asarray(
            [reference, centers.max()] if search_range is None else search_range, float
        )
        if (
            auto_range.shape != (2,)
            or not np.isfinite(auto_range).all()
            or auto_range[1] <= auto_range[0]
        ):
            raise ValueError("search_range must be one finite increasing interval")
        low, low_reason = _rate_bracket(grid, mask, auto_range[0], "search_lower")
        high, high_reason = _rate_bracket(grid, mask, auto_range[1], "search_upper")
        search_reason = low_reason or high_reason
        if search_reason is None and mask[low[0] : high[1] + 1].any():
            search_reason = "masked_search_interval"
        eligible = np.flatnonzero(
            (grid >= auto_range[0]) & (grid <= auto_range[1]) & ~mask
        )
        if not len(eligible):
            search_reason = search_reason or "no_supported_search_centers"
        reason = reason or search_reason
        if reason is None:
            selected = int(eligible[np.argmax(absolute[eligible])])
            xi_star = float(grid[selected])
            if selected in (eligible[0], eligible[-1]):
                warnings.append(
                    "The unsmoothed maximum is at an edge of the search centers; inspect or extend the search range."
                )
    reference_bracket = surface_bracket = None
    if xi_star is not None:
        reference_bracket, surface_bracket, endpoint_reason = _rate_support(
            grid, mask, reference, xi_star
        )
        reason = reason or endpoint_reason
    ready = reason is None and xi_star is not None
    barrier = None
    if reference_bracket is not None:
        free = absolute - _rate_value(absolute, reference_bracket)
    if ready:
        barrier = _rate_value(absolute, surface_bracket) - _rate_value(
            absolute, reference_bracket
        )
    if not fit.converged or not connected:
        free[:] = np.nan
        mask[:] = True
    gs = np.asarray([statistical_inefficiency(a) for a in arrays])
    blocks = np.minimum(counts, np.ceil(gs).astype(int))
    moments = []
    for i, a in enumerate(arrays):
        mean, std = float(a.mean()), float(a.std())
        z = (a - mean) / std if std else None
        moments.append(
            {
                "stream": i,
                "center": float(centers[i]),
                "samples": len(a),
                "mean": mean,
                "standard_deviation": std,
                "skewness": float(np.mean(z**3)) if z is not None else None,
                "excess_kurtosis": float(np.mean(z**4) - 3) if z is not None else None,
                "statistical_inefficiency": float(gs[i]),
                "block_length": int(blocks[i]),
                "estimated_effective_samples": float(counts[i] / gs[i]),
            }
        )

    def endpoint_details(bracket):
        if bracket is None:
            return None
        items = []
        for index in sorted(set(bracket[:2])):
            selected = (x >= edges[index]) & (x < edges[index + 1])
            w = fit.weights[selected]
            items.append(
                {
                    "bin": index,
                    "center": float(grid[index]),
                    "raw_count": int(count[index]),
                    "density": float(density[index]),
                    "weighted_effective_samples_without_time_correction": (
                        float(w.sum() ** 2 / (w @ w)) if len(w) else 0.0
                    ),
                    "contributing_streams": sum(
                        bool(selected[start:stop].any())
                        for start, stop in zip(offsets[:-1], offsets[1:])
                    ),
                }
            )
        return {"bracket": list(bracket), "bins": items}

    ui = {"requested": bool(compare_ui), "ready": False, "reason": None}
    ui_grid = None
    if compare_ui and ready:
        lower, upper = min(reference, xi_star), max(reference, xi_star, centers.max())
        if lower == upper:
            lower, upper = grid[0], grid[-1]
        ui_grid = np.linspace(lower, upper, ui_points)
        try:
            ui_curve, _ = umbrella_integration(
                arrays, centers, stiffness, beta, ui_grid
            )
            ui_barrier = float(
                np.interp(xi_star, ui_grid, ui_curve)
                - np.interp(reference, ui_grid, ui_curve)
            )
            ui.update(
                ready=True,
                xi_star=xi_star,
                reference=reference,
                free_energy_barrier_hartree=ui_barrier,
                mbar_minus_ui_hartree=barrier - ui_barrier,
                xi=ui_grid,
                free_energy=ui_curve - np.interp(reference, ui_grid, ui_curve),
            )
        except ValueError as exc:
            ui["reason"] = "ui_fit_failed: " + str(exc)
    draws, endpoints, curves, records, failures = [], [], [], [], []
    ui_draws, paired_mbar, differences, ui_failures = [], [], [], []
    rng = np.random.default_rng(seed)
    if ready:
        for draw in range(int(bootstrap)):
            indices = np.concatenate(
                [
                    _block_resample(np.arange(n), block, rng) + start
                    for n, block, start in zip(counts, blocks, offsets[:-1])
                ]
            )
            xb = x[indices]
            record = {"draw": draw, "valid": False, "reason": None}
            try:
                mb = solve_mbar(
                    u[:, indices],
                    fit_counts,
                    overlap_threshold=overlap_threshold,
                    initial_free_energies=fit.free_energies,
                    **options,
                )
                record["solver_residual"] = float(mb.residual)
                if not mb.converged:
                    record["reason"] = "mbar_not_converged"
                elif len(np.unique(mb.components)) != 1:
                    record["reason"] = "disconnected_overlap"
                else:
                    _, db, _, _, maskb = _histogram(
                        xb[:, None], mb.weights, [edges], beta, min_count
                    )
                    rb, sb, record["reason"] = _rate_support(
                        grid, maskb, reference, xi_star
                    )
                    if record["reason"] is None:
                        ab = np.full_like(db, np.nan)
                        ab[~maskb] = -np.log(db[~maskb]) / beta
                        rv, sv = _rate_value(ab, rb), _rate_value(ab, sb)
                        b = sv - rv
                        record.update(
                            valid=True,
                            reference_free_energy_hartree=rv,
                            surface_free_energy_hartree=sv,
                            barrier_hartree=b,
                        )
                        endpoints.append([rv, sv])
                        draws.append(b)
                        curves.append(ab - rv)
                        if ui["ready"]:
                            try:
                                ub, _ = umbrella_integration(
                                    [
                                        xb[start:stop]
                                        for start, stop in zip(
                                            offsets[:-1], offsets[1:]
                                        )
                                    ],
                                    centers,
                                    stiffness,
                                    beta,
                                    ui_grid,
                                )
                                uv = float(
                                    np.interp(xi_star, ui_grid, ub)
                                    - np.interp(reference, ui_grid, ub)
                                )
                                ui_draws.append(uv)
                                paired_mbar.append(b)
                                differences.append(b - uv)
                                record["ui_barrier_hartree"] = uv
                            except ValueError as exc:
                                ui_failures.append(
                                    {
                                        "draw": draw,
                                        "reason": "ui_fit_failed",
                                        "message": str(exc),
                                    }
                                )
            except (FloatingPointError, np.linalg.LinAlgError, RuntimeError) as exc:
                record["reason"] = "mbar_solver_error"
                record["message"] = str(exc)
            records.append(record)
            if not record["valid"]:
                failures.append(
                    {key: value for key, value in record.items() if key != "valid"}
                )
    curve_se = np.full_like(free, np.nan)
    curve_valid = np.zeros(len(grid), int)
    if len(curves) >= 2:
        bcurves = np.asarray(curves)
        curve_valid = np.isfinite(bcurves).sum(axis=0)
        mean = np.divide(
            np.nansum(bcurves, axis=0),
            curve_valid,
            out=np.zeros_like(free),
            where=curve_valid > 0,
        )
        square = np.nansum((bcurves - mean) ** 2, axis=0)
        variance = np.divide(
            square,
            curve_valid - 1,
            out=np.full_like(free, np.nan),
            where=curve_valid > 1,
        )
        curve_se = np.sqrt(variance)
    stats = _rate_draw_statistics(draws)
    failure_counts = {}
    for failure in failures:
        failure_counts[failure["reason"]] = failure_counts.get(failure["reason"], 0) + 1
    if reason:
        warnings.append("Rate PMF unavailable: " + reason)
    if bootstrap and len(draws) < 2:
        warnings.append(
            "Fewer than two valid bootstrap draws; the barrier uncertainty is unavailable."
        )
    if failures:
        warnings.append(
            "Some bootstrap draws failed; inspect each recorded reason and the valid fraction."
        )
    if np.any(counts / gs < 20):
        warnings.append(
            "Some streams have fewer than 20 estimated independent samples; overlap alone does not establish mixing."
        )
    warnings.append(
        "Histogram PMF retains bin-width bias; uncertainty holds the selected surface fixed and is not proof of physical convergence."
    )
    ui.update(
        paired_ui_bootstrap_barriers=ui_draws,
        paired_mbar_bootstrap_barriers=paired_mbar,
        paired_difference_bootstrap_hartree=differences,
        paired_difference_statistics=_rate_draw_statistics(differences),
        ui_barrier_statistics=_rate_draw_statistics(ui_draws),
        failures=ui_failures,
    )
    return {
        "ready": ready,
        "status": "ready" if ready else reason,
        "xi_star": xi_star,
        "reference": float(reference),
        "free_energy_barrier_hartree": barrier,
        "barrier_standard_error_hartree": stats["standard_error_hartree"],
        "barrier_percentile_95_ci_hartree": stats["percentile_95_ci_hartree"],
        "pmf_bootstrap_barriers": draws,
        "xi": grid,
        "free_energy": free,
        "standard_error": curve_se,
        "mask": mask,
        "edges": edges,
        "density": density,
        "raw_count": count,
        "overlap_connected": connected,
        "overlap": fit.overlap,
        "components": fit.components,
        "window_free_energies": fit.free_energies[state_map],
        "weights": fit.weights,
        "statistical_inefficiency": gs,
        "effective_samples_per_stream": counts / gs,
        "weight_ess": float(1 / (fit.weights @ fit.weights)),
        "window_moments": moments,
        "support": {
            "reference": endpoint_details(reference_bracket),
            "dividing_surface": endpoint_details(surface_bracket),
            "search_range": auto_range.tolist() if auto_range is not None else None,
            "continuous_interval": ready,
            "selection": selection,
            "smoothing": "none",
            "failure_reason": reason,
        },
        "solver": {
            "converged": fit.converged,
            "residual": float(fit.residual),
            "unique_umbrella_hamiltonians": len(representatives),
            "stream_to_state": state_map,
            "overlap_indexing": "unique umbrella Hamiltonians",
            "details": fit.diagnostics,
        },
        "bootstrap": {
            "requested": int(bootstrap),
            "attempted": len(records),
            "valid": len(draws),
            "valid_fraction": len(draws) / bootstrap if bootstrap else None,
            "failures": failures,
            "failure_counts": failure_counts,
            "draws": records,
            "skipped_reason": reason if not ready else None,
            "endpoint_values_hartree": np.asarray(endpoints).reshape(-1, 2),
            "endpoint_covariance_hartree_squared": (
                np.cov(np.asarray(endpoints).T, ddof=1) if len(endpoints) >= 2 else None
            ),
            "curve_valid_draws": curve_valid,
            "surface_held_fixed": True,
            "block_lengths": blocks,
        },
        "ui_comparison": ui,
        "warnings": warnings,
        "measure": "rate_flux",
        "energy_unit": "hartree",
        "estimator": "histogram_mbar",
        "converged": False,
    }


def umbrella_integration(samples, centers, kappa, beta, grid):
    """RPMDrate 1D Gaussian umbrella integration, using sampled mean/variance.

    Mathematical port of RPMDrate main.py computePotentialOfMeanForce,
    Copyright (c) 2012 Allen, Green, Suleimanov, MIT. Gaussian-window
    approximation is explicit; MBAR overlap must be inspected separately.
    """
    arrays = [np.asarray(a, float).reshape(-1) for a in samples]
    means = np.array([x.mean() for x in arrays])
    variances = np.array([x.var() for x in arrays])
    if np.any(variances <= 1e-20):
        raise ValueError("zero-variance umbrella stream")
    centers = np.asarray(centers, float).reshape(-1)
    kappa = np.asarray(kappa, float).reshape(-1)
    grid = np.asarray(grid, float)
    if (
        len(centers) != len(arrays)
        or len(kappa) != len(arrays)
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("invalid umbrella integration shapes or grid")
    logs = (
        np.log([len(x) for x in arrays])[None]
        - 0.5 * np.log(2 * np.pi * variances)[None]
    )
    logs = logs - 0.5 * (grid[:, None] - means[None]) ** 2 / variances[None]
    weights = np.exp(logs - logsumexp(logs, axis=1)[:, None])
    gradients = (grid[:, None] - means[None]) / (beta * variances[None]) - kappa[
        None
    ] * (grid[:, None] - centers[None])
    derivative = np.sum(weights * gradients, axis=1)
    f = np.zeros_like(grid)
    f[1:] = np.cumsum(0.5 * np.diff(grid) * (derivative[:-1] + derivative[1:]))
    f -= f.min()
    return f, derivative
