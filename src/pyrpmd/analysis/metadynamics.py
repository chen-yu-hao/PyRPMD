"""Finite-time, bias-derived metadynamics free-energy estimates.

Standard: F(s) approximately -V(s,t)+C (Laio & Parrinello, PNAS 99,
12562 (2002), doi:10.1073/pnas.202427399). Well tempered: F(s) approaches
-gamma/(gamma-1)*V(s,t)+C (Barducci et al., PRL 100, 020603 (2008),
doi:10.1103/PhysRevLett.100.020603).

Neither relation licenses treating the evolving-bias samples as fixed-umbrella
MBAR data. No kinetic estimator or time-dependent trajectory reweighting is
implemented here. Unsampled bins are masked even though V can be evaluated.
"""

from __future__ import annotations
import numpy as np


def _cv_signature(cv):
    """Semantic identity for built-in CVs; plugin parameters remain user-owned."""
    from ..cv import (CVStack, Distance, BondDifference, Cartesian, Angle,
                      COMDistance, RPMDrateCoordinate, RateSurface,
                      PeriodicDistance, SurfaceHeight, SurfaceLateral)
    from ..cv_periodic import (PeriodicBondDifference, PeriodicAngle,
                               LinearProjection, FractionalCoordinate)

    kind = type(cv)
    if kind is CVStack:
        return ("stack", tuple(_cv_signature(c) for c in cv.cvs))
    common = (kind.__module__, kind.__qualname__, getattr(cv, "estimator", None),
              tuple(cv.units), tuple(cv.periods))
    if kind in (Distance, Angle):
        parameters = cv.atoms
    elif kind is BondDifference:
        parameters = cv.bonds
    elif kind is Cartesian:
        parameters = (cv.atom, cv.axis)
    elif kind is COMDistance:
        parameters = tuple(cv.weights)
    elif kind is RPMDrateCoordinate:
        parameters = (_cv_signature(cv.com), cv.rinf,
                      tuple(cv.reference_difference), tuple(cv.paths))
    elif kind is RateSurface:
        parameters = (_cv_signature(cv.coordinate), cv.xi)
    elif kind in (PeriodicDistance, PeriodicAngle, PeriodicBondDifference):
        atoms = cv.bonds if kind is PeriodicBondDifference else cv.atoms
        parameters = (tuple(atoms), tuple(cv.cell.ravel()), tuple(cv.pbc))
    elif kind in (SurfaceHeight, SurfaceLateral, LinearProjection, FractionalCoordinate):
        parameters = (tuple(cv._affine_gradient().ravel()), cv.origin)
    else:
        # A plugin may have arbitrary mutable parameters and external state.
        # Introspection cannot establish equality of its mathematical mapping.
        parameters = ()
    return common + (parameters,)


def _grid_signature(bias):
    if not hasattr(bias, "grid_algorithm"):
        return None
    return (bias.grid_algorithm, tuple(bias.grid_ranges.ravel()), tuple(bias.grid_bins))


def bias_fes(biases, samples, *, bins=40, ranges=None, min_count=1):
    """Estimate F from independently deposited biases on a common grid.

    Replica bias histories are evaluated separately, never merged. Estimates
    share one gauge only when at least one bin is sampled by every replica.
    The reported replica spread is descriptive; it is not a convergence test
    or a statistically calibrated error bar for the unknown equilibrium FES.
    All replicas must represent the same temperature and CV mapping. Built-in
    CV parameters are checked; callers using custom CV plugins must ensure
    identical plugin parameters and mathematical semantics themselves.
    """
    if not biases or len(biases) != len(samples):
        raise ValueError("one bias history and sample array per replica required")
    dimension = biases[0].cv.dimension
    if dimension not in (1, 2):
        raise ValueError("metadynamics FES supports one or two dimensions")
    if any(b.cv.dimension != dimension or b.mode != biases[0].mode or
           b.bias_factor != biases[0].bias_factor or b.beta != biases[0].beta or
           tuple(b.cv.units) != tuple(biases[0].cv.units) or
           tuple(b.cv.periods) != tuple(biases[0].cv.periods) or
           _grid_signature(b) != _grid_signature(biases[0]) or
           _cv_signature(b.cv) != _cv_signature(biases[0].cv)
           for b in biases):
        raise ValueError("metadynamics replicas must share their temperature, method and CVs")
    arrays = [np.asarray(x, float).copy() for x in samples]
    if any(x.ndim != 2 or x.shape[1] != dimension or not len(x) or
           not np.isfinite(x).all() for x in arrays):
        raise ValueError("metadynamics analysis requires finite nonempty [N,D] samples")
    if isinstance(min_count, (bool, np.bool_)) or not isinstance(min_count, (int, np.integer)) or min_count < 1:
        raise ValueError("analysis.min_count must be a positive integer")
    if np.isscalar(bins):
        bins = [bins] * dimension
    if len(bins) != dimension or any(isinstance(b, (bool, np.bool_)) or
            not isinstance(b, (int, np.integer)) or b < 2 for b in bins):
        raise ValueError("metadynamics bins must contain an integer >=2 per CV")
    if np.prod(bins, dtype=float) > 1_000_000:
        raise ValueError("metadynamics analysis grid exceeds one million points")
    if ranges is None:
        if hasattr(biases[0], "grid_ranges"):
            ranges = biases[0].grid_ranges.tolist()
    if ranges is None:
        all_samples = np.concatenate(arrays)
        ranges = []
        for j, period in enumerate(biases[0].cv.periods):
            if period is not None:
                ranges.append([-period / 2, period / 2])
            else:
                lo, hi = float(all_samples[:, j].min()), float(all_samples[:, j].max())
                width = max(hi - lo, float(biases[0].sigma[j]))
                ranges.append([lo - 0.05 * width, hi + 0.05 * width])
    bounds = np.asarray(ranges, float)
    if bounds.shape != (dimension, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("analysis.ranges needs a finite increasing [lower,upper] per CV")
    for j, period in enumerate(biases[0].cv.periods):
        if period is not None:
            if not np.isclose(bounds[j, 1] - bounds[j, 0], period, rtol=1e-10, atol=1e-12):
                raise ValueError("a periodic metadynamics analysis range must span exactly one period")
            for x in arrays:
                x[:, j] = (x[:, j] - bounds[j, 0]) % period + bounds[j, 0]
    edges = [np.linspace(lo, hi, int(n) + 1) for (lo, hi), n in zip(bounds, bins)]
    centers = [0.5 * (a[1:] + a[:-1]) for a in edges]
    grid = np.stack(np.meshgrid(*centers, indexing="ij"), axis=-1)
    flat = grid.reshape(-1, dimension)
    values = []
    # Bound temporary [grid,hill] allocations even for long HILLS histories.
    for bias in biases:
        chunks = []
        chunk_size = (1024 if hasattr(bias, "grid_algorithm") else
                      max(1, min(1024, 1_000_000 // max(1, len(bias.hills)))))
        for start in range(0, len(flat), chunk_size):
            chunks.append(bias.physical_energy(flat[start:start + chunk_size]))
        values.append(np.concatenate(chunks).reshape(grid.shape[:-1]))
    values = np.asarray(values)
    count = np.asarray([np.histogramdd(x, bins=edges)[0].astype(int) for x in arrays])
    masks = count < min_count
    # A history with zero hills cannot be an estimated free-energy landscape.
    for i, bias in enumerate(biases):
        if not bias.hills:
            masks[i] = True
    common = ~np.any(masks, axis=0)
    raw = np.asarray([-b.fes_factor * value for b, value in zip(biases, values)])
    free = np.full(raw.shape[1:], np.nan)
    spread = np.full_like(free, np.nan)
    reference = None
    if np.any(common):
        average = raw.mean(axis=0)
        reference = np.unravel_index(np.argmin(np.where(common, average, np.inf)), common.shape)
        shifted = raw - raw[(slice(None),) + reference].reshape((-1,) + (1,) * dimension)
        free[common] = shifted.mean(axis=0)[common]
        if len(biases) > 1:
            spread[common] = shifted.std(axis=0, ddof=1)[common]
    else:
        shifted = raw.copy()
        for i in range(len(biases)):
            if np.any(~masks[i]):
                shifted[i] -= np.min(raw[i][~masks[i]])
    shifted[masks] = np.nan
    return {
        "free_energy": free,
        "bias_per_replica": values,
        "free_energy_per_replica": shifted,
        "replica_spread": spread,
        "standard_error": np.full_like(free, np.nan),
        "mask": ~common,
        "mask_per_replica": masks,
        "raw_count": count.sum(axis=0),
        "raw_count_per_replica": count,
        "edges": edges,
        "reference_bin": None if reference is None else list(reference),
        "common_support": bool(np.any(common)),
        "method": "bias_estimate",
        "mode": biases[0].mode,
        "fes_factor": biases[0].fes_factor,
        "measure": "cv_density",
        "energy_unit": "hartree",
        "converged": False,
    }
