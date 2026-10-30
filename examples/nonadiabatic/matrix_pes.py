"""Artificial two-state A + BC <-> AB + C teaching PES, in atomic units.

Not a fitted H3 PES or a validated chemical model. The base exchange form is
from PyRPMD's examples/rate/toy_exchange.py; a localized real diabatic coupling
and an arbitrary constant state gap are added here.
"""

import numpy as np
from pyrpmd.nonadiabatic import DiabaticResult


class TwoStateExchange:
    metadata = {
        "execution_device": "cpu", "thread_safety": "reentrant",
        "model": "artificial two-state exchange; not physical H3",
    }

    def __init__(self, coupling=0.003, gap=0.002):
        self.coupling, self.gap = float(coupling), float(gap)
        if not np.isfinite([self.coupling, self.gap]).all():
            raise ValueError("coupling and gap must be finite")

    def evaluate(self, q):
        q = np.asarray(q, dtype=float)
        if q.ndim != 3 or q.shape[1:] != (3, 3):
            raise ValueError("TwoStateExchange requires exactly three atoms")
        a, b = q[:, 0] - q[:, 1], q[:, 2] - q[:, 1]
        ra, rb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        if np.any(ra < 1e-12) or np.any(rb < 1e-12):
            raise ValueError("coincident atoms")
        ua, ub = a / ra[:, None], b / rb[:, None]
        cosine = np.sum(ua * ub, axis=1)
        depth, alpha, equilibrium, repulsion, bending = 0.08, 1.2, 1.2, 0.32, 0.1
        ea, eb = np.exp(-alpha * (ra - equilibrium)), np.exp(-alpha * (rb - equilibrium))
        pair = ea * eb
        barrier = repulsion + bending * (1 + cosine)
        energy = depth * (ea * ea - 2 * ea + eb * eb - 2 * eb) + pair * barrier
        dra = 2 * depth * alpha * ea * (1 - ea) - alpha * pair * barrier
        drb = 2 * depth * alpha * eb * (1 - eb) - alpha * pair * barrier
        ga = dra[:, None] * ua + bending * pair[:, None] * (ub - cosine[:, None] * ua) / ra[:, None]
        gb = drb[:, None] * ub + bending * pair[:, None] * (ua - cosine[:, None] * ub) / rb[:, None]
        gradient = np.stack((ga, -ga - gb, gb), axis=1)
        # Localized at the exchange region; decays to zero in both asymptotes.
        center, width = 2.0, 1.0
        off = self.coupling * np.exp(-((ra - center)**2 + (rb - center)**2) / (2 * width**2))
        dca = -off[:, None] * (ra - center)[:, None] * ua / width**2
        dcb = -off[:, None] * (rb - center)[:, None] * ub / width**2
        coupling_gradient = np.stack((dca, -dca - dcb, dcb), axis=1)
        matrix = np.empty((len(q), 2, 2))
        gradients = np.empty((len(q), 2, 2, 3, 3))
        matrix[:, 0, 0], matrix[:, 1, 1] = energy, energy + self.gap
        matrix[:, 0, 1] = matrix[:, 1, 0] = off
        gradients[:, 0, 0] = gradients[:, 1, 1] = gradient
        gradients[:, 0, 1] = gradients[:, 1, 0] = coupling_gradient
        return DiabaticResult(matrix, gradients)


def make(**params):
    return TwoStateExchange(**params)
