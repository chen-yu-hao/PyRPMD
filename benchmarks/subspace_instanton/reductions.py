"""One-dimensional reductions of a 2D valley model along its minimum-energy path.

All four models are symmetric (``V(-s, +-Q) = V(s, Q)``), so the path is
followed from the saddle towards products and mirrored.  The reductions are
potentials on a uniform grid that tend to zero at both ends:

``mep``        ``V`` on the path, arc length as coordinate;
``fe``         plus the harmonic quantum free energy of the transverse mode,
               ``beta^-1 ln[sinh(beta w_perp / 2) / sinh(beta w_inf / 2)]``
               (the first-order exact projection of ``docs/subspace-transfer-theory.md``);
``fe+sct``     the same on the small-curvature coordinate ``d tau = sqrt(mu_eff) d sigma``;
``bare``       ``V0 sech^2(s/a)`` with the bare mass along ``s``: the bath relaxed
               statically and given no inertia.

Every reduction has a *skeleton*: the same path and mass with the bare
potential, i.e. without the transverse free energy.  ``mep`` is the skeleton
of ``fe``, ``mep+sct`` that of ``fe+sct``, and ``bare`` is its own.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline


def minimum_energy_path(model, *, s_max=13.0, spacing=0.01, start=1e-3):
    """Steepest descent from the saddle towards ``+s``: arc length, points, V, w_perp, curvature."""
    saddle, unstable, _, mode = model.saddle()

    def direction(_, x):
        g = model.gradient(x)
        return -g / np.linalg.norm(g)

    reached = lambda _, x: x[0] - s_max
    reached.terminal = True
    solution = solve_ivp(direction, [0.0, 80.0], saddle + start * mode, method="LSODA", rtol=1e-10, atol=1e-12,
                         events=reached, dense_output=True, max_step=0.05)
    # Uniform arc-length grid that contains the saddle (the integration starts ``start``
    # along the unstable mode): an uneven first interval makes the spline curvature ring.
    sigma = np.arange(spacing, solution.t[-1] + start, spacing)
    points = np.vstack([saddle, solution.sol(sigma - start).T])
    sigma = np.concatenate([[0.0], sigma])
    # Continue through the saddle by the symmetry of the model, so that derivatives at
    # sigma = 0 are two-sided: V(-s, Q) = V(s, Q) for an even valley floor (a mirror),
    # V(-s, -Q) = V(s, Q) for the odd "tanh" floor (an inversion through the saddle).
    if model.bend == "tanh":
        mirrored = 2.0 * saddle - points[:0:-1]
    else:
        mirrored = points[:0:-1] * np.array([-1.0, 1.0])
    spline = CubicSpline(np.concatenate([-sigma[:0:-1], sigma]), np.vstack([mirrored, points]))
    tangent = spline(sigma, 1)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    curvature = np.linalg.norm(spline(sigma, 2), axis=1)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    hessian = model.hessian(points)
    perpendicular = np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", normal, hessian, normal), 1e-12))
    return {"sigma": sigma, "points": points, "energy": model.energy(points), "w_perp": perpendicular,
            "curvature": curvature, "unstable_frequency": unstable, "end_s": float(points[-1, 0])}


def _symmetric_grid(half, values, *, length=18.0, dx=0.01):
    """Even 1D potential on a uniform grid from half-line data; zero beyond the data."""
    x = np.arange(-length, length + 0.5 * dx, dx)
    spline = CubicSpline(half, values)
    inside = np.abs(x) <= half[-1]
    return x, np.where(inside, spline(np.minimum(np.abs(x), half[-1])), 0.0)


def reductions(model, beta, path=None):
    """``{name: (x, U)}`` for the reductions listed in the module docstring, and the SCT mass."""
    path = minimum_energy_path(model) if path is None else path
    sigma, energy, w, kappa = path["sigma"], path["energy"], path["w_perp"], path["curvature"]
    out = {"mep": _symmetric_grid(sigma, energy - energy[-1])}
    free = energy + np.log(np.sinh(0.5 * beta * w) / np.sinh(0.5 * beta * model.omega_inf)) / beta
    free = free - free[-1]
    out["fe"] = _symmetric_grid(sigma, free)
    width = 1.0 / np.sqrt(w)
    a_bar = kappa * width
    mass = np.minimum(1.0, np.exp(-2.0 * a_bar - a_bar**2 + np.gradient(width, sigma) ** 2))
    tau = np.concatenate([[0.0], np.cumsum(0.5 * (np.sqrt(mass[1:]) + np.sqrt(mass[:-1])) * np.diff(sigma))])
    out["fe+sct"] = _symmetric_grid(tau, free)
    # Skeleton of "fe+sct": the same coordinate with the bare potential (see run_skeleton.py).
    out["mep+sct"] = _symmetric_grid(tau, energy - energy[-1])
    x = np.arange(-18.0, 18.005, 0.01)
    out["bare"] = (x, model.barrier(x))
    return out, mass


def barrier_frequency(x, potential):
    """Barrier frequency of a tabulated symmetric barrier at its top (x = 0)."""
    i = len(x) // 2
    return float(np.sqrt(-(potential[i + 1] - 2.0 * potential[i] + potential[i - 1]) / (x[1] - x[0]) ** 2))
