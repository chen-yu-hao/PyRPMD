"""Analytic reference models. These are tests/examples, not fitted reaction PESs."""

import numpy as np
from .base import Potential, PESResult, coordinates


class FreePotential(Potential):
    metadata = {
        "provider": "free",
        "thread_safety": "reentrant",
        "execution_device": "cpu",
    }

    def evaluate(self, x):
        x = coordinates(x)
        return PESResult(np.zeros(len(x)), np.zeros_like(x))


class HarmonicPotential(Potential):
    metadata = {
        "provider": "harmonic",
        "thread_safety": "reentrant",
        "execution_device": "cpu",
    }

    def __init__(self, k=1.0, center=None):
        self.k = np.asarray(k, dtype=float)
        self.center = np.asarray(0.0 if center is None else center, dtype=float)
        if (
            not np.isfinite(self.k).all()
            or np.any(self.k < 0)
            or not np.isfinite(self.center).all()
        ):
            raise ValueError("invalid harmonic parameters")

    def evaluate(self, x):
        x = coordinates(x)
        d = x - self.center
        f = -self.k * d
        return PESResult(
            0.5 * np.sum(self.k * d * d, axis=(1, 2)),
            np.broadcast_to(f, x.shape).copy(),
        )


class MorsePairPotential(Potential):
    metadata = {
        "provider": "morse_pair",
        "thread_safety": "reentrant",
        "execution_device": "cpu",
    }

    def __init__(self, atoms=(0, 1), depth=0.1, alpha=1.0, equilibrium=1.4):
        self.atoms = tuple(atoms)
        self.depth = float(depth)
        self.alpha = float(alpha)
        self.equilibrium = float(equilibrium)
        if (
            len(self.atoms) != 2
            or len(set(self.atoms)) != 2
            or min(self.atoms) < 0
            or min(self.depth, self.alpha, self.equilibrium) <= 0
        ):
            raise ValueError("invalid Morse parameters")

    def evaluate(self, x):
        x = coordinates(x)
        a, b = self.atoms
        d = x[:, a] - x[:, b]
        r = np.linalg.norm(d, axis=1)
        if np.any(r <= 1e-14):
            raise ValueError("coincident Morse-pair atoms")
        e = np.exp(-self.alpha * (r - self.equilibrium))
        v = self.depth * (1 - e) ** 2
        dv = 2 * self.depth * self.alpha * e * (1 - e)
        f = np.zeros_like(x)
        f[:, a] = -dv[:, None] * d / r[:, None]
        f[:, b] = -f[:, a]
        return PESResult(v, f)


class DoubleWellPotential(Potential):
    """Separable Cartesian quartic model for validating barriers and FES only."""

    metadata = {
        "provider": "double_well",
        "thread_safety": "reentrant",
        "execution_device": "cpu",
    }

    def __init__(self, height=0.01, width=1.0, atom=0, axis=0, confinement=0.01):
        self.height = float(height)
        self.width = float(width)
        self.atom = int(atom)
        self.axis = int(axis)
        self.confinement = float(confinement)
        if (
            self.height <= 0
            or self.width <= 0
            or self.atom < 0
            or self.axis not in (0, 1, 2)
            or self.confinement < 0
        ):
            raise ValueError("invalid double-well parameters")

    def evaluate(self, x):
        x = coordinates(x)
        y = x[:, self.atom, self.axis] / self.width
        v = self.height * (y * y - 1) ** 2
        f = -self.confinement * x
        vv = (
            0.5 * self.confinement * np.sum(x * x, axis=(1, 2))
            - 0.5 * self.confinement * x[:, self.atom, self.axis] ** 2
        )
        f[:, self.atom, self.axis] = -4 * self.height * y * (y * y - 1) / self.width
        return PESResult(v + vv, f)
