"""Cl + HBr -> HCl + Br on the BrHCl_LEPS1 surface (atom order Br, H, Cl).

Helpers shared by the subspace-correction benchmark: flat-vector potential
and gradient callables, isotope masses and the collinear saddle point.
Build the surface first with ``python benchmarks/brhcl/build_pes.py``.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from pyrpmd.constants import AMU_TO_AU

MASSES_AMU = np.array([78.9183376, 1.00782503207, 34.96885268])  # 79Br, 1H, 35Cl
MASSES = MASSES_AMU * AMU_TO_AU
CARTESIAN_MASSES = np.repeat(MASSES, 3)


class Surface:
    def __init__(self):
        from benchmarks.brhcl.pes import BrHClPotential

        self.potential = BrHClPotential()
        self.calls = 0

    def evaluate(self, x):
        x = np.asarray(x, dtype=float).reshape(-1, 3, 3)
        self.calls += len(x)
        result = self.potential.evaluate(x)
        return np.asarray(result.energy, float), -np.asarray(result.forces, float).reshape(len(x), 9)

    def energy(self, x):
        return float(self.evaluate(x)[0][0])

    def gradient(self, x):
        return self.evaluate(x)[1][0]

    def hessian(self, x, step=1e-4):
        x = np.asarray(x, dtype=float).ravel()
        h = np.empty((9, 9))
        for i in range(9):
            e = np.zeros(9)
            e[i] = step
            h[:, i] = (self.gradient(x + e) - self.gradient(x - e)) / (2 * step)
        return 0.5 * (h + h.T)


def collinear(r_brh, r_hcl):
    return np.array([[-r_brh, 0.0, 0.0], [0.0, 0.0, 0.0], [r_hcl, 0.0, 0.0]])


def saddle(surface):
    def residual(r):
        g = surface.gradient(collinear(*r)).reshape(3, 3)
        return np.array([-g[0, 0], g[2, 0]])

    best = None
    for a in np.linspace(2.6, 3.0, 5):
        for b in np.linspace(3.2, 4.0, 5):
            solution = least_squares(residual, [a, b], bounds=([2.0, 2.0], [6.0, 6.0]),
                                     xtol=1e-14, ftol=1e-14, gtol=1e-14)
            if np.linalg.norm(residual(solution.x)) < 1e-9:
                energy = surface.energy(collinear(*solution.x))
                if best is None or energy > best[0]:
                    best = (energy, solution.x)
    return collinear(*best[1])
