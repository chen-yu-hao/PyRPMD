"""Blocked, scaled triangular Sylvester solve using LAPACK diagonal tiles.

Solves T**H X - X T = scale*C. Off-diagonal updates use level-3 BLAS
through NumPy matmul; diagonal blocks retain LAPACK's scaling safeguards.
No replacement of Schur decomposition, CAP, threshold or normalization is made.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.linalg.lapack import ztrsyl

from .subspace_native import positive_integer


def triangular_sylvester(triangular, rhs, *, solver="auto", block_size=64):
    """Return ``(X, scale, info, diagnostics)`` in LAPACK's scaling convention.

    A perturbed/singular diagonal tile or underflowed cumulative scale causes
    a full reference LAPACK solve. The fallback does not conceal ``info``.
    ``auto`` uses blocked updates only above 256 rows. Neither input is mutated.
    """
    block_size = positive_integer(block_size, "sylvester_block_size")
    if solver not in ("auto", "lapack", "blocked"):
        raise ValueError("sylvester_solver must be auto, lapack or blocked")
    t = np.asarray(triangular, dtype=np.complex128)
    c = np.asarray(rhs, dtype=np.complex128)
    if t.ndim != 2 or t.shape[0] != t.shape[1] or c.shape != t.shape or not len(t):
        raise ValueError("triangular and rhs must be nonempty square matrices of the same shape")
    if not np.isfinite(t).all() or not np.isfinite(c).all():
        raise ValueError("Sylvester inputs must be finite")
    if np.any(np.tril(t, -1) != 0):
        raise ValueError("T must be upper triangular")
    selected = ("blocked" if len(t) > 256 else "lapack") if solver == "auto" else solver
    start = perf_counter()
    fallback = None

    def reference():
        return ztrsyl(t, t, c, trana="C", tranb="N", isgn=-1)

    if selected == "lapack" or len(t) <= block_size:
        x, scale, info = reference()
        selected = "lapack"
    else:
        x, scale, info, fallback = _blocked(t, c, block_size)
        if fallback:
            x, scale, info = reference()
            selected = "lapack"
    return x, scale, info, {"solver": selected, "block_size": block_size,
                            "fallback_reason": fallback, "seconds": perf_counter() - start}


def _blocked(t, c, block_size):
    n = len(t)
    blocks = [slice(i, min(n, i + block_size)) for i in range(0, n, block_size)]
    diagonal = [np.array(t[s, s], order="F", copy=True) for s in blocks]
    x = np.zeros_like(c, order="F")
    scaled_rhs = np.array(c, order="F", copy=True)
    scale = 1.0
    for jb, col in enumerate(blocks):
        for ib, row in enumerate(blocks):
            residual = scaled_rhs[row, col].copy(order="F")
            if row.start:
                residual -= t[:row.start, row].conj().T @ x[:row.start, col]
            if col.start:
                residual += x[row, :col.start] @ t[:col.start, col]
            if not np.isfinite(residual).all():
                return None, None, None, "non-finite off-diagonal update"
            tile, local_scale, info = ztrsyl(diagonal[ib], diagonal[jb], residual,
                                             trana="C", tranb="N", isgn=-1, overwrite_c=1)
            if info != 0:
                return None, None, None, f"diagonal tile returned LAPACK info={info}"
            if not np.isfinite(tile).all() or not 0 < local_scale <= 1:
                return None, None, None, "non-finite or invalidly scaled diagonal tile"
            if local_scale != 1.0:
                scale *= local_scale
                if scale == 0.0:
                    return None, None, None, "cumulative scaling underflow"
                x *= local_scale
                scaled_rhs *= local_scale
            x[row, col] = tile
    return x, scale, 0, None
