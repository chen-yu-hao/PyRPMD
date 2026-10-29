"""Analytic teaching PES for A + BC <-> AB + C, with two free asymptotes.

This is deliberately NOT a fitted H3 PES or a physical rate benchmark.
Coordinates are bohr; energies/forces are hartree/hartree per bohr.
V = D*(e1^2 - 2e1 + e2^2 - 2e2) + A*e1*e2
    + B*e1*e2*(1 + u1 dot u2), e_i=exp[-alpha*(r_i-r0)].
The saddle is linear, r1=r2=r0+log((2D+A)/(2D))/alpha.
At either separated asymptote only one Morse diatomic remains.
"""

import numpy as np
from pyrpmd.potentials import PESResult


class ToyExchange:
    metadata = {
        "thread_safety": "reentrant",
        "execution_device": "cpu",
        "model": "analytic teaching exchange; not physical H3",
    }

    def __init__(
        self, depth=0.08, alpha=1.2, equilibrium=1.2, coupling=0.32, bending=0.1
    ):
        self.depth, self.alpha, self.equilibrium = depth, alpha, equilibrium
        self.coupling, self.bending = coupling, bending
        if min(depth, alpha, equilibrium, bending) <= 0 or coupling <= 2 * depth:
            raise ValueError(
                "teaching model requires positive parameters and coupling > 2*depth"
            )

    def evaluate(self, positions):
        x = np.asarray(positions, float)
        if x.ndim != 3 or x.shape[1:] != (3, 3):
            raise ValueError("toy exchange expects exactly three atoms")
        a = x[:, 0] - x[:, 1]
        b = x[:, 2] - x[:, 1]
        ra = np.linalg.norm(a, axis=1)
        rb = np.linalg.norm(b, axis=1)
        if np.any(ra < 1e-12) or np.any(rb < 1e-12):
            raise ValueError("coincident teaching-model atoms")
        ua = a / ra[:, None]
        ub = b / rb[:, None]
        cos = np.sum(ua * ub, axis=1)
        ea = np.exp(-self.alpha * (ra - self.equilibrium))
        eb = np.exp(-self.alpha * (rb - self.equilibrium))
        pair = ea * eb
        coupling = self.coupling + self.bending * (1 + cos)
        energy = self.depth * (ea * ea - 2 * ea + eb * eb - 2 * eb) + pair * coupling
        dra = 2 * self.depth * self.alpha * ea * (1 - ea) - self.alpha * pair * coupling
        drb = 2 * self.depth * self.alpha * eb * (1 - eb) - self.alpha * pair * coupling
        ga = (
            dra[:, None] * ua
            + self.bending * pair[:, None] * (ub - cos[:, None] * ua) / ra[:, None]
        )
        gb = (
            drb[:, None] * ub
            + self.bending * pair[:, None] * (ua - cos[:, None] * ub) / rb[:, None]
        )
        force = np.empty_like(x)
        force[:, 0] = -ga
        force[:, 2] = -gb
        force[:, 1] = ga + gb
        return PESResult(energy, force)


def build(**params):
    return ToyExchange(**params)
