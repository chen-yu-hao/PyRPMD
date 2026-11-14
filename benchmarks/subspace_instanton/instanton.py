"""Ring-polymer instantons of model potentials (unit mass, hbar = 1) and their unstable mode.

The ring-polymer potential of ``N`` beads at inverse temperature ``beta`` is

    U_N(x) = sum_i V(x_i) + (omega_N^2 / 2) sum_i |x_i - x_{i+1}|^2,   omega_N = N / beta.

Above the crossover temperature its only saddle is the collapsed polymer at the
barrier top, whose unstable frequency is the barrier frequency.  Below it the
saddle is the instanton, with one unstable mode of frequency ``eta_1`` and one
zero mode (cyclic permutation of the beads).
"""

from __future__ import annotations

import numpy as np


def ring_hessian(hessian, x, omega_n):
    """Hessian of ``U_N`` at beads ``x[N, D]``; ``hessian(x) -> [N, D, D]``."""
    n, d = x.shape
    out = np.zeros((n * d, n * d))
    local = hessian(x)
    eye = np.eye(d)
    for i in range(n):
        a = slice(i * d, (i + 1) * d)
        out[a, a] += local[i] + 2.0 * omega_n**2 * eye
        for j in ((i + 1) % n, (i - 1) % n):
            out[a, slice(j * d, (j + 1) * d)] -= omega_n**2 * eye
    return out


def find_instanton(gradient, hessian, saddle, mode, barrier_frequency, beta, *, nbeads=128, amplitude=None,
                   start=None, max_iterations=400, tolerance=1e-10, max_step=0.3):
    """Newton search for the instanton; ``None`` above the crossover temperature.

    ``gradient(x[N, D]) -> [N, D]`` and ``hessian(x[N, D]) -> [N, D, D]`` are
    vectorized over beads.  The start is the collapsed polymer stretched along
    the unstable ``mode`` of the ``saddle``.  The zero mode is removed from
    each Newton step by a pseudo-inverse.  Returns ``eta1`` (unstable-mode
    frequency), the action ``S = beta_N U_N``, the beads and diagnostics.
    """
    saddle = np.atleast_1d(np.asarray(saddle, dtype=float))
    mode = np.atleast_1d(np.asarray(mode, dtype=float))
    mode = mode / np.linalg.norm(mode)
    if beta * barrier_frequency <= 2.0 * np.pi * 1.0005:
        return None
    omega_n = nbeads / beta
    if start is None:
        if amplitude is None:
            amplitude = 2.4 * np.sqrt(max(1.0 - (2.0 * np.pi / (beta * barrier_frequency)) ** 2, 0.02))
        x = saddle[None, :] + amplitude * np.cos(2.0 * np.pi * np.arange(nbeads) / nbeads)[:, None] * mode[None, :]
    else:
        x = np.array(start, dtype=float)
    converged = False
    for _ in range(max_iterations):
        g = gradient(x) + omega_n**2 * (2.0 * x - np.roll(x, 1, 0) - np.roll(x, -1, 0))
        values, vectors = np.linalg.eigh(ring_hessian(hessian, x, omega_n))
        keep = np.abs(values) > 1e-7 * np.abs(values).max()
        projected = vectors.T @ g.ravel()
        step = -(vectors[:, keep] @ (projected[keep] / values[keep])).reshape(x.shape)
        size = np.abs(step).max()
        if size > max_step:
            step *= max_step / size
        x = x + step
        if np.abs(g).max() < tolerance and size < 1e-9:
            converged = True
            break
    spread = float(np.ptp((x - saddle) @ mode))
    if spread < 1e-3:
        return None
    values = np.linalg.eigvalsh(ring_hessian(hessian, x, omega_n))
    return {"x": x, "eta1": float(np.sqrt(-values[0])), "negative_modes": int(np.sum(values < -1e-8)),
            "zero_mode": float(values[1]), "spread": spread, "converged": converged,
            "gradient_max": float(np.abs(g).max()), "nbeads": int(nbeads), "beta": float(beta)}


def steepest_descent_factor(beta, barrier_frequency, instanton):
    """``ln eps_SD``: ``ln[2 pi / (beta eta_1)]`` below the crossover, 0 above it."""
    if instanton is None:
        return 0.0
    return float(np.log(2.0 * np.pi / (beta * instanton["eta1"])))
