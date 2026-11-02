"""Confined real two-state model for the thermal-rate workflow smoke example."""
import numpy as np

from pyrpmd.nonadiabatic import DiabaticResult


class CoupledHarmonic:
    def evaluate(self, q):
        scalar = .5*np.sum(q*q, axis=(1, 2))
        matrix = np.zeros((len(q), 2, 2))
        matrix[:, 0, 0] = scalar+.2*q[:, 0, 0]
        matrix[:, 1, 1] = scalar-.2*q[:, 0, 0]+.1
        matrix[:, 0, 1] = matrix[:, 1, 0] = .05
        gradients = np.zeros((len(q), 2, 2, q.shape[1], 3))
        gradients[:, 0, 0] = gradients[:, 1, 1] = q
        gradients[:, 0, 0, 0, 0] += .2
        gradients[:, 1, 1, 0, 0] -= .2
        return DiabaticResult(matrix, gradients)


def build():
    return CoupledHarmonic()
