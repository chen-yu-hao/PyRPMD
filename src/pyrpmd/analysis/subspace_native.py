"""Optional fused C++/OpenMP kernels for static, cubic-grid subspace models.

The NumPy reference remains available. Arbitrary Python potentials are never
silently replaced by a fitted grid, nor called while the native GIL is released.
RNG draws and HMC acceptance/adaptation remain on their established Python clock.
"""
from __future__ import annotations

import numpy as np


def positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def make_backend(ring, backend, threads):
    """Return a native adapter, or a documented NumPy fallback."""
    from .subspace_model import GridPotential, SubspaceModel

    if backend not in ("auto", "numpy", "native"):
        raise ValueError("subspace backend must be auto, numpy or native")
    threads = positive_integer(threads, "threads")
    if backend == "numpy":
        if threads != 1:
            raise ValueError("threads>1 requires the native subspace backend")
        return None, "explicit NumPy reference"
    reason = None
    model = ring.model
    if type(model) is not SubspaceModel or model.function is not None or model.gradient is not None:
        reason = "native subspace propagation requires an unmodified static grid model"
    else:
        grid = model.interpolant
        if type(grid) is not GridPotential or grid._spline is None or not hasattr(grid._spline, "c"):
            reason = "native subspace propagation requires the NdBSpline representation"
    try:
        from pyrpmd import _subspace_native as native
    except ImportError:
        native = None
        reason = reason or "the optional _subspace_native extension is not installed"
    if native is not None and getattr(native, "API_VERSION", None) != 1:
        reason = "incompatible _subspace_native API version"
    if reason:
        if backend == "native" or threads != 1:
            raise ValueError(reason)
        return None, reason
    if threads > 1 and not native.OPENMP_ENABLED:
        raise ValueError("threads>1 requires an OpenMP-enabled subspace build")
    return NativeRing(ring, native, threads), None


class NativeRing:
    def __init__(self, ring, native, threads):
        self.model = ring.model
        self.dimension = ring.d
        self.mode_matrix = ring.c
        self.grid = ring.model.interpolant
        self.native = native
        spline = self.grid._spline
        self.context = native.make_context(
            tuple(np.ascontiguousarray(t, dtype=np.float64) for t in spline.t),
            np.ascontiguousarray(spline.c, dtype=np.float64),
            np.ascontiguousarray(ring.mass), np.ascontiguousarray(ring.c),
            np.ascontiguousarray(ring.omega), threads,
        )

    def _count(self, counts):
        self.grid.evaluations += counts[0]
        self.grid.outside += counts[1]

    def potential(self, modes):
        beads = np.ascontiguousarray(np.einsum("jk,bkd->bjd", self.mode_matrix, modes))
        energy = np.empty(beads.shape[:-1], dtype=np.float64)
        gradient = np.empty_like(beads)
        try:
            counts = self.native.evaluate(self.context, beads.reshape(-1, self.dimension),
                                          energy.ravel(), gradient.reshape(-1, self.dimension))
        except FloatingPointError:
            # Preserve reference counting/error semantics for a rejected proposal.
            value, gradient = self.model.energy_and_gradient(beads)
            if not np.isfinite(value).all() or not np.isfinite(gradient).all():
                raise FloatingPointError("subspace potential returned non-finite values")
            return value.sum(axis=1), gradient
        self._count(counts)
        return energy.sum(axis=1), gradient

    def advance(self, state, dt, steps, constrained, *, side=None, surface=0.0):
        # Proposal buffers are private: a failure cannot mutate accepted states.
        q, p, f = [np.array(v, dtype=np.float64, order="C", copy=True) for v in state[:3]]
        energy = np.empty(q.shape[0], dtype=np.float64)
        gradient = np.empty_like(q)
        counts = self.native.advance(self.context, q, p, f, energy, gradient, float(dt),
                                     int(steps), int(constrained), side, float(surface))
        self._count(counts)
        return q, p, f, energy, gradient
