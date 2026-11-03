"""Analytic reference models. These are tests/examples, not fitted reaction PESs."""

import numpy as np
from .base import Potential, PESResult, coordinates
from ..geometry import FixedCellGeometry, validate_cell_pbc


class _BuiltinPotential(Potential):
    def native_binding(self, natoms=None):
        """Lease a compiled, immutable parameter snapshot for ``natoms``.

        Dynamics supplies its atom count before the first force evaluation.
        Direct callers can provide it explicitly, or omit it after evaluate().
        Only the exact, unmodified built-in evaluator is eligible.
        """
        from ._builtin_native import builtin_binding
        return builtin_binding(self, natoms)


_BUILTIN_NATIVE_BINDING = _BuiltinPotential.native_binding


class FreePotential(_BuiltinPotential):
    metadata = {
        "provider": "free",
        "thread_safety": "reentrant",
        "execution_device": "cpu",
        "periodic": True,
    }

    def evaluate(self, x, *, cell=None, pbc=None):
        x = coordinates(x)
        self._native_natoms = x.shape[1]
        return PESResult(np.zeros(len(x)), np.zeros_like(x), virial=np.zeros((len(x), 3, 3)))

    def set_cell(self, cell):
        """The free-particle energy is cell independent; retain a validated cell."""
        self.cell, self.pbc = validate_cell_pbc(cell, [True, True, True])


class PeriodicCosinePotential(_BuiltinPotential):
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
        self._native_natoms = x.shape[1]
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


class LennardJonesPotential(Potential):
    """Periodic shifted Lennard-Jones reference with an explicit backend.

    All values are canonical atomic units: ``epsilon`` in hartree and
    ``sigma``, ``cutoff`` and ``skin`` in bohr. Lattice vectors are rows. The
    cutoff must not exceed half the shortest cell face height, so one exact
    minimum image represents every interacting pair. ``backend='native'`` uses
    the C ABI v2 evaluator and deterministic cached Verlet lists;
    ``backend='numpy'`` is the independent all-pairs reference implementation.
    """

    metadata = {
        "provider": "lj",
        "thread_safety": "context_serial",
        "execution_device": "cpu",
        "periodic": True,
        "capabilities": {"virial": True, "native_batch": True, "stateful": False},
        "virial_convention": "minus_dE_dstrain",
        "model_scope": "analytic_validation_only",
    }

    def __init__(
        self, cell, *, epsilon=1.0, sigma=1.0, cutoff=None, skin=0.3,
        pbc=None, backend="native",
    ):
        values = (epsilon, sigma, skin) if cutoff is None else (epsilon, sigma, cutoff, skin)
        if any(isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError("Lennard-Jones parameters must be real physical scalars, not booleans")
        try:
            self.epsilon = float(epsilon)
            self.sigma = float(sigma)
            self.cutoff = 2.5 * self.sigma if cutoff is None else float(cutoff)
            self.skin = float(skin)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Lennard-Jones parameters must be real physical scalars") from exc
        if backend not in ("native", "numpy"):
            raise ValueError("Lennard-Jones backend must be 'native' or 'numpy'")
        self.backend = backend
        if (
            not np.isfinite([self.epsilon, self.sigma, self.cutoff, self.skin]).all()
            or self.epsilon <= 0
            or self.sigma <= 0
            or self.cutoff <= 0
            or self.skin < 0
            or not np.isfinite(self.cutoff + self.skin)
        ):
            raise ValueError("Lennard-Jones parameters must be finite with positive epsilon, sigma and cutoff and nonnegative skin")
        with np.errstate(over="ignore", invalid="ignore"):
            cutoff64 = np.float64(self.cutoff)
            list_cutoff64 = np.float64(self.cutoff + self.skin)
            self._cutoff2 = float(cutoff64 * cutoff64)
            self._list_cutoff2 = float(list_cutoff64 * list_cutoff64)
            sr = np.float64(self.sigma / self.cutoff)
            sr6 = sr ** 6
            self._energy_shift = float(4.0 * self.epsilon * (sr6 * sr6 - sr6))
        if not np.isfinite([self._cutoff2, self._list_cutoff2, self._energy_shift]).all():
            raise ValueError("Lennard-Jones parameters overflow in atomic units")
        self.set_cell(cell, pbc=pbc)

    def set_cell(self, cell, pbc=None):
        cell, flags = validate_cell_pbc(cell, pbc)
        if cell is None or not flags.all():
            raise ValueError("Lennard-Jones requires a full-rank cell with three periodic axes")
        inverse = np.linalg.inv(cell)
        face_heights = 1.0 / np.linalg.norm(inverse, axis=0)
        if self.cutoff > 0.5 * float(np.min(face_heights)) * (1.0 + 64.0 * np.finfo(float).eps):
            raise ValueError(
                "Lennard-Jones cutoff exceeds half the shortest cell face height"
            )
        self.cell = cell
        self.pbc = flags
        self._geometry = FixedCellGeometry(cell, flags)

    def native_binding(self, natoms=None):
        from ..propagation import NativeUnavailableError
        raise NativeUnavailableError(
            "periodic Lennard-Jones uses the cell-aware C ABI v2 evaluator and cannot be leased as a C ABI v1 fused binding"
        )

    def _evaluate_numpy(self, x):
        x = coordinates(x)
        batches, natoms, _ = x.shape
        energy = np.zeros(batches)
        forces = np.zeros_like(x)
        virial = np.zeros((batches, 3, 3))
        for batch in range(batches):
            for i in range(natoms - 1):
                displacement = self._geometry.minimum_image(x[batch, i + 1:] - x[batch, i])
                r2 = np.einsum("ij,ij->i", displacement, displacement)
                if np.any(r2 <= np.finfo(float).tiny):
                    raise ValueError("coincident Lennard-Jones atoms")
                active = np.nonzero(r2 < self._cutoff2)[0]
                for offset in active:
                    j = i + 1 + int(offset)
                    d = displacement[offset]
                    inverse_r2 = 1.0 / r2[offset]
                    sr2 = self.sigma * self.sigma * inverse_r2
                    sr6 = sr2 * sr2 * sr2
                    sr12 = sr6 * sr6
                    energy[batch] += (
                        4.0 * self.epsilon * (sr12 - sr6) - self._energy_shift
                    )
                    pair_force = (
                        24.0 * self.epsilon * (2.0 * sr12 - sr6) * inverse_r2 * d
                    )
                    # displacement is q_j - q_i, hence the force on i is -pair_force.
                    forces[batch, i] -= pair_force
                    forces[batch, j] += pair_force
                    virial[batch] += np.outer(d, pair_force)
        self._native_natoms = natoms
        return PESResult(energy, forces, virial=virial)

    def evaluate(self, x):
        if self.backend == "numpy":
            return self._evaluate_numpy(x)
        from ._builtin_native import evaluate_lj
        result = evaluate_lj(self, x)
        self._native_natoms = result.forces.shape[1]
        return result


class HarmonicPotential(_BuiltinPotential):
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
        self._native_natoms = x.shape[1]
        d = x - self.center
        f = -self.k * d
        return PESResult(
            0.5 * np.sum(self.k * d * d, axis=(1, 2)),
            np.broadcast_to(f, x.shape).copy(),
        )


class MorsePairPotential(_BuiltinPotential):
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
        self._native_natoms = x.shape[1]
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


class DoubleWellPotential(_BuiltinPotential):
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
        self._native_natoms = x.shape[1]
        y = x[:, self.atom, self.axis] / self.width
        v = self.height * (y * y - 1) ** 2
        f = -self.confinement * x
        vv = (
            0.5 * self.confinement * np.sum(x * x, axis=(1, 2))
            - 0.5 * self.confinement * x[:, self.atom, self.axis] ** 2
        )
        f[:, self.atom, self.axis] = -4 * self.height * y * (y * y - 1) / self.width
        return PESResult(v + vv, f)


# Capture at provider-module import, before any optional batch dispatcher is
# loaded. Class-level scientific overrides must not acquire an implicit promise
# that frames are independent merely because the class has a built-in name.
_BUILTIN_EVALUATE = {kind: kind.evaluate for kind in (
    FreePotential, HarmonicPotential, MorsePairPotential,
    DoubleWellPotential, PeriodicCosinePotential,
)}
