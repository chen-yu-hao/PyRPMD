"""Collinear H + H2 on BKMP2 in mass-weighted symmetric coordinates.

All three masses are equal, so with r1 = r(Ha-Hb), r2 = r(Hb-Hc)

    s = (r2 - r1) sqrt(m/6)    (reaction coordinate, dividing surface s = 0)
    t = (r1 + r2) sqrt(m/2)    (symmetric stretch)

are orthonormal mass-weighted coordinates of the centre-of-mass-free collinear
system (unit mass).  Build the PES first: ``python benchmarks/h3/build_pes.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.h3.pes import BKMP2Potential  # noqa: E402
from pyrpmd.constants import AMU_TO_AU  # noqa: E402

MASS = 1.00782503223 * AMU_TO_AU
CS = np.sqrt(6.0 / MASS)
CT = np.sqrt(2.0 / MASS)
ASYMPTOTE = -0.174524771  # Eh, BKMP2 H + H2 at r(H2) = 1.401, r -> infinity (classical)
WALL = 0.5  # Eh above the asymptote: cap on the repulsive wall (thermally irrelevant at <= 600 K)


def bonds(points):
    p = np.asarray(points, float)
    d = p[..., 0] * CS
    u = p[..., 1] * CT
    return 0.5 * (u - d), 0.5 * (u + d)


def to_st(r1, r2):
    return (np.asarray(r2) - r1) / CS, (np.asarray(r1) + r2) / CT


class Collinear:
    """V(s, t) relative to the classical H + H2 asymptote, with its exact gradient."""

    def __init__(self, library=None):
        self.pes = BKMP2Potential(library)

    def _evaluate(self, points):
        p = np.asarray(points, float)
        shape = p.shape[:-1]
        r1, r2 = (a.ravel() for a in bonds(p))
        energy = np.full(r1.size, WALL)
        gradient = np.zeros((r1.size, 2))
        ok = (r1 > 0.3) & (r2 > 0.3)
        if ok.any():
            x = np.zeros((int(ok.sum()), 3, 3))
            x[:, 1, 2] = r1[ok]
            x[:, 2, 2] = r1[ok] + r2[ok]
            res = self.pes.evaluate(x)
            e = res.energy - ASYMPTOTE
            good = (res.status == 0) & (e < WALL)
            dr2 = -res.forces[:, 2, 2]
            dr1 = -res.forces[:, 1, 2] - res.forces[:, 2, 2]
            g = np.stack([0.5 * CS * (dr2 - dr1), 0.5 * CT * (dr1 + dr2)], axis=-1)
            idx = np.flatnonzero(ok)
            energy[idx[good]] = e[good]
            gradient[idx[good]] = g[good]
        return energy.reshape(shape), gradient.reshape(shape + (2,))

    def __call__(self, points):
        return self._evaluate(points)[0]

    def gradient(self, points):
        return self._evaluate(points)[1]
