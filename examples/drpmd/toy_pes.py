"""Smooth two-atom Gaussian repulsion for adapter testing; no reactive chemistry."""

import numpy as np
from pyrpmd.potentials import PESResult


class GaussianRepulsion:
    metadata = {"execution_device": "cpu", "thread_safety": "reentrant"}

    def __init__(self, height_hartree=0.002, width_bohr=1.5):
        self.height = float(height_hartree)
        self.width = float(width_bohr)
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Gaussian height and width must be positive")

    def evaluate(self, q):
        if q.shape[1] != 2:
            raise ValueError("This toy PES accepts exactly two atoms")
        displacement = q[:, 0] - q[:, 1]
        energy = self.height * np.exp(
            -np.sum(displacement**2, axis=1) / (2 * self.width**2)
        )
        force = energy[:, None] * displacement / self.width**2
        return PESResult(energy, np.stack((force, -force), axis=1))


def build_pes(**params):
    return GaussianRepulsion(**params)
