"""Analytic reference models. These are tests/examples, not fitted reaction PESs."""

import numpy as np
from .base import Potential, PESResult, coordinates
from ..geometry import validate_cell_pbc


class FreePotential(Potential):
    metadata = {
        "provider": "free",
        "thread_safety": "reentrant",
        "execution_device": "cpu",
        "periodic": True,
    }

    def evaluate(self, x, *, cell=None, pbc=None):
        x = coordinates(x)
        return PESResult(np.zeros(len(x)), np.zeros_like(x))


class PeriodicCosinePotential(Potential):
    """Analytic fixed-lattice reference; not a fitted metal/hydrogen potential.

    Each active atom feels height[a]/2*(1-cos(2*pi*repeats[a]*s[a]))
    along periodic fractional coordinates. The optional confinement is a
    Cartesian harmonic potential in the subspace perpendicular to all periodic
    lattice vectors. A separable cosine model alone does not imply diffusion.
    Parameters use bohr/hartree, including the supplied fixed cell.
    """

    metadata = {
        "provider": "periodic_cosine", "thread_safety": "reentrant",
        "execution_device": "cpu", "periodic": True,
        "model_scope": "analytic_validation_only",
    }

    def __init__(self, cell, pbc=None, height=0.01, repeats=(1, 1, 1), atoms=None, confinement=0.0):
        self.cell, self.pbc = validate_cell_pbc(cell, pbc)
        if self.cell is None or not self.pbc.any():
            raise ValueError("periodic_cosine requires a periodic cell")
        height_value = np.asarray(height)
        if height_value.dtype.kind not in "iuf" or height_value.shape not in ((), (3,)):
            raise ValueError("periodic_cosine height must be a scalar or three energies")
        self.height = np.broadcast_to(height_value.astype(float), (3,)).copy()
        r = np.asarray(repeats)
        if r.shape != (3,) or r.dtype.kind not in "iu" or np.any(r < 1):
            raise ValueError("periodic_cosine repeats must be three positive integers")
        self.repeats = r.copy()
        self.confinement = float(confinement)
        if not np.isfinite(self.height).all() or np.any(self.height < 0) or not np.isfinite(self.confinement) or self.confinement < 0:
            raise ValueError("periodic_cosine energies and confinement must be finite and nonnegative")
        if atoms is None:
            self.atoms = None
        else:
            ids = np.asarray(atoms)
            if ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in "iu" or np.any(ids < 0) or len(np.unique(ids)) != len(ids):
                raise ValueError("periodic_cosine atoms must be distinct nonnegative integers")
            self.atoms = ids.copy()
        self.inverse = np.linalg.inv(self.cell)
        basis = self.cell[self.pbc]
        self.perpendicular = np.eye(3) - np.linalg.pinv(basis) @ basis
        self.height[~self.pbc] = 0.0

    def evaluate(self, x):
        x = coordinates(x)
        if self.atoms is not None and np.max(self.atoms) >= x.shape[1]:
            raise ValueError("periodic_cosine atom index out of range")
        selected = x if self.atoms is None else x[:, self.atoms]
        phase = 2 * np.pi * (selected @ self.inverse) * self.repeats
        e = 0.5 * np.sum(self.height * (1 - np.cos(phase)), axis=(1, 2))
        gradient_fractional = np.pi * self.height * self.repeats * np.sin(phase)
        force = -gradient_fractional @ self.inverse.T
        perpendicular = selected @ self.perpendicular
        e += 0.5 * self.confinement * np.sum(perpendicular * perpendicular, axis=(1, 2))
        force -= self.confinement * perpendicular
        forces = np.zeros_like(x)
        if self.atoms is None:
            forces[:] = force
        else:
            forces[:, self.atoms] = force
        return PESResult(e, forces)


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
