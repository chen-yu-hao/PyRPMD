"""Native trajectory adapter; the Python solver retains its RNG/statistics."""

from __future__ import annotations

import numpy as np

from .subspace_rpmd import _RingPolymer


class NativeRingPolymer(_RingPolymer):
    def __init__(self, model, beta, nbeads, threads):
        from pyrpmd import _subspace

        super().__init__(model, beta, nbeads)
        self.native = _subspace
        knots, coefficients = model.interpolant._native_data
        self.plan = _subspace.create(knots, coefficients, self.p)
        self.threads = threads
        self.backend = "native"

    def _record(self, info):
        self.threads_used = max(self.threads_used, info["threads"])
        self.model.interpolant.evaluations += info["evaluations"]
        self.model.interpolant.outside += info["outside"]

    def potential(self, modes):
        points = np.ascontiguousarray(self.beads(modes).reshape(-1, self.d))
        energy = np.empty(len(points))
        gradient = np.empty_like(points)
        self._record(self.native.evaluate(self.plan, points, energy, gradient, self.threads))
        return energy.reshape(-1, self.p).sum(axis=1), gradient.reshape(modes.shape)

    def propagate(self, modes, momenta, force, dt, propagator, constrained, steps, *, surface=None):
        # Private outputs preserve the Python solver's transactional HMC proposal:
        # a failed trial never modifies the accepted state or its cached force.
        q, p, f = [np.array(a, dtype=float, order="C", copy=True) for a in (modes, momenta, force)]
        energy = np.empty(len(q))
        gradient = np.empty_like(q)
        side = None if surface is None else np.empty((steps + 1, len(q)), dtype=bool)
        info = self.native.advance(self.plan, q, p, f, energy, gradient,
                                   np.ascontiguousarray(propagator), float(dt), int(steps),
                                   int(constrained), self.threads, side,
                                   0.0 if surface is None else float(surface))
        self._record(info)
        self.native_steps += steps * len(q)
        return (q, p, f, energy, gradient), side


def make_ring(model, beta, nbeads, backend, threads):
    if backend not in ("auto", "python", "native"):
        raise ValueError("backend must be 'auto', 'python' or 'native'")
    if isinstance(threads, bool) or int(threads) != threads or threads < 1:
        raise ValueError("threads must be a positive integer")
    reason = None
    if backend != "python":
        if model.function is not None or model.gradient is not None:
            reason = "Python potential callbacks require backend='python'; use an explicit gridded model for native splines"
        else:
            try:
                return NativeRingPolymer(model, beta, nbeads, int(threads))
            except ImportError:
                reason = "pyrpmd._subspace extension is unavailable"
        if backend == "native":
            raise RuntimeError(reason)
    ring = _RingPolymer(model, beta, nbeads)
    ring.fallback_reason = reason
    return ring
