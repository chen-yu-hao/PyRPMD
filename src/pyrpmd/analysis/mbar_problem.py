"""Reduced potentials evaluated in bounded blocks, including bootstrap views."""
from __future__ import annotations

import numpy as np


class ReducedPotentials:
    """A dense input view or harmonic umbrellas without allocating K x N.

    State and sample selections retain the original arrays. In particular,
    resampling columns does not copy a reduced-potential matrix.
    """

    def __init__(self, values, *, centers=None, stiffness=None, beta=1., periods=None,
                 rows=None, indices=None):
        self.values = np.ascontiguousarray(values, dtype=np.float64)
        self.beta = float(beta)
        self.centers = np.empty((0, 0)) if centers is None else np.ascontiguousarray(centers, dtype=float)
        self.stiffness = np.empty((0, 0, 0)) if stiffness is None else np.ascontiguousarray(stiffness, dtype=float)
        self.periods = np.empty(0) if centers is None else np.array(
            [0. if p is None else p for p in ((None,) * self.values.shape[1] if periods is None else periods)], dtype=float)
        states = self.values.shape[0] if centers is None else len(self.centers)
        samples = self.values.shape[1] if centers is None else len(self.values)
        self.rows = np.arange(states, dtype=np.int64) if rows is None else np.ascontiguousarray(rows, dtype=np.int64)
        self.indices = np.empty(0, dtype=np.int64) if indices is None else np.ascontiguousarray(indices, dtype=np.int64)
        self.shape = (len(self.rows), len(self.indices) if indices is not None else samples)
        self.ndim = 2

    def __len__(self):
        return self.shape[0]

    def block(self, start, stop):
        ix = self.indices[start:stop] if len(self.indices) else slice(start, stop)
        if not len(self.centers):
            columns = self.indices[start:stop] if len(self.indices) else np.arange(start, stop)
            return self.values[np.ix_(self.rows, columns)]
        x = self.values[ix]
        delta = x[None] - self.centers[self.rows, None]
        for axis, period in enumerate(self.periods):
            if period:
                delta[:, :, axis] = (delta[:, :, axis] + period / 2) % period - period / 2
        return .5 * self.beta * np.einsum("kni,kij,knj->kn", delta, self.stiffness[self.rows], delta)

    def select(self, rows=None, indices=None):
        state_rows = self.rows if rows is None else self.rows[rows]
        sample_indices = (self.indices if len(self.indices) else None)
        if indices is not None:
            sample_indices = indices if sample_indices is None else sample_indices[indices]
        return ReducedPotentials(self.values, centers=self.centers if len(self.centers) else None,
                                 stiffness=self.stiffness, beta=self.beta,
                                 periods=tuple(None if p == 0 else p for p in self.periods),
                                 rows=state_rows, indices=sample_indices)

    def __getitem__(self, key):
        rows, columns = key
        if isinstance(rows, slice) and rows == slice(None):
            return self.select(indices=columns)
        return np.asarray(self)[key]

    def __array__(self, dtype=None, copy=None):
        result = np.empty(self.shape, dtype=float)
        for start in range(0, self.shape[1], 4096):
            result[:, start:start+4096] = self.block(start, min(start+4096, self.shape[1]))
        return result.astype(dtype, copy=False) if dtype is not None else result

    def representatives(self):
        from .mbar import _identical_rows
        if len(self.centers):
            return _identical_rows(np.column_stack((self.centers[self.rows],
                                                   self.stiffness[self.rows].reshape(len(self.rows), -1))))
        return _identical_rows([self.values[row] for row in self.rows])

    @property
    def nbytes(self):
        # Per-draw vectors and bounded solver workspace, not a virtual K*N matrix.
        k, n = self.shape
        return 8 * (8 * n + 4 * k * k + 2 * k * min(n, 1024))


def umbrella_problem(samples, centers, stiffness, beta, periods=None):
    """Build a lazy umbrella problem after the caller validates the CV data."""
    centers = np.asarray(centers, float)
    k = np.asarray(stiffness, float)
    if k.ndim == 2:
        k = np.broadcast_to(k, (len(centers), *k.shape))
    return ReducedPotentials(samples, centers=centers, stiffness=k, beta=beta, periods=periods)
