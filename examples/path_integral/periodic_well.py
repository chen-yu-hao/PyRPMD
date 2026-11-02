"""Analytic periodic wells with explicit cell energy for SC-NPT mechanics.

This is a reproducible integration example, not a fitted materials model.
Cell vectors are rows. The force-squared SC correction has a nonzero strain
derivative even though the fractional-coordinate cosine has zero virial.
"""

import numpy as np

from pyrpmd.potentials import PESResult


class PeriodicWell:
    metadata = {"periodic": True, "model_scope": "analytic_validation_only"}

    def evaluate(self, q, *, cell, pbc):
        inverse = np.linalg.inv(cell)
        volume = abs(np.linalg.det(cell))
        phase = 2 * np.pi * q @ inverse
        energy = .001 * np.sum(1 - np.cos(phase), axis=(1, 2)) + .01 / volume
        force = -.002 * np.pi * np.sin(phase) @ inverse.T
        virial = np.repeat((np.eye(3) * .01 / volume)[None], len(q), axis=0)
        return PESResult(energy, force, virial=virial)


def build():
    return PeriodicWell()
