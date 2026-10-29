"""MBAR and multidimensional histogram FES, with correlated block bootstrap.

MBAR equations: Shirts & Chodera, JCP 129,124105 (2008),
doi:10.1063/1.2978177. The solver here is implemented directly with SciPy;
it is not copied from PyMBAR. A shared physical reduced potential cancels
only for equal T, bead count, PES, and Hamiltonian across all windows.
"""

from __future__ import annotations
from dataclasses import dataclass
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


def solve_mbar(
    u_kn,
    counts,
    target_u=None,
    tolerance=1e-9,
    max_iterations=10000,
    overlap_threshold=1e-6,
):
    """Return MBAR target weights, state overlap and identifiability diagnostics.

    u_kn[K,N] are dimensionless potentials, counts[K] sum to N; only sampled
    states are passed. target_u[N] defaults to zero after cancelling the same
    physical Hamiltonian from every sampling/target reduced potential.
    """
    u = np.asarray(u_kn, float)
    counts = np.asarray(counts, dtype=int)
    if (
        u.ndim != 2
        or counts.shape != (len(u),)
        or np.any(counts <= 0)
        or counts.sum() != u.shape[1]
    ):
        raise ValueError("MBAR requires u[K,N], positive counts[K] summing to N")
    if not np.isfinite(u).all():
        raise ValueError("MBAR energies must be finite")
    target = np.zeros(u.shape[1]) if target_u is None else np.asarray(target_u, float)
    if target.shape != (u.shape[1],) or not np.isfinite(target).all():
        raise ValueError("target reduced potential must be finite [N]")
    logn = np.log(counts)
    # A per-frame common offset improves conditioning without changing weights.
    common = np.min(u, axis=0)
    u = u - common[None]
    target = target - common
    k = len(counts)
    f = np.zeros(k)
    # Fixed point initialization is useful for chains of overlapping windows.
    for _ in range(min(200, max_iterations)):
        logd = logsumexp(logn[:, None] + f[:, None] - u, axis=0)
        new = -logsumexp(-u - logd[None], axis=1)
        new -= new[0]
        if np.max(np.abs(new - f)) < tolerance:
            f = new
            break
        f = new
    if k > 1:

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
    # Polish and independently check the equations, not just optimizer status.
    for _ in range(max_iterations):
        logd = logsumexp(logn[:, None] + f[:, None] - u, axis=0)
        new = -logsumexp(-u - logd[None], axis=1)
        new -= new[0]
        residual = float(np.max(np.abs(new - f)))
        f = new
        if residual < tolerance:
            break
    logd = logsumexp(logn[:, None] + f[:, None] - u, axis=0)
    state_weights = np.exp(f[:, None] - u - logd[None]).T
    logw = -target - logd
    weights = np.exp(logw - logsumexp(logw))
    overlap = (state_weights.T @ state_weights) * counts[None, :]
    graph = np.maximum(overlap, overlap.T) > overlap_threshold
    _, components = connected_components(graph, directed=False)
    return MBARResult(
        f, weights, state_weights, overlap, components, residual < tolerance, residual
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
            mb = solve_mbar(ub, counts, overlap_threshold=overlap_threshold)
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
    return {
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
