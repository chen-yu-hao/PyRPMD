"""Synthetic property surface for checking IR/Raman/SFG plumbing only."""

import numpy as np


def build():
    def evaluate(q):
        total = q.sum(axis=1)
        # The shared linear coordinate produces a nonzero cross response.
        # These are deliberately toy tensors, not a property model of H.
        alpha = np.broadcast_to(3 * np.eye(3), (len(q), 3, 3)).copy()
        alpha[:, 0, 0] += .2 * total[:, 2]
        alpha[:, 1, 1] -= .1 * total[:, 2]
        return {"dipole": total, "polarizability": alpha}
    return evaluate
