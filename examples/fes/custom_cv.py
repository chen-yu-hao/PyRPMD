"""An importable full-ring CV factory with an explicit [D,P,N,3] Jacobian."""

import numpy as np
from pyrpmd.cv import CVResult


class RotatedCentroid:
    metadata = {"dimension": 2, "units": ["bohr", "bohr"], "periods": [None, None]}

    def __init__(self, atom=0):
        self.atom = atom

    def value_and_jacobian(self, state):
        q = state.q
        p = len(q)
        x, y = q[:, self.atom, :2].mean(axis=0)
        value = np.array([(x + y) / np.sqrt(2), (x - y) / np.sqrt(2)])
        jacobian = np.zeros((2,) + q.shape)
        jacobian[0, :, self.atom, :2] = 1 / (np.sqrt(2) * p)
        jacobian[1, :, self.atom, 0] = 1 / (np.sqrt(2) * p)
        jacobian[1, :, self.atom, 1] = -1 / (np.sqrt(2) * p)
        return CVResult(value, jacobian)


def build(atom=0):
    return RotatedCentroid(atom)
