"""Model potentials with exact full-dimensional references for the RPH 2D subspace.

All coordinates are mass-weighted with unit mass (hbar = 1, hartree).  The
reaction coordinate ``x`` crosses a symmetric Eckart barrier; the transverse
valleys bend, so the minimum-energy path is curved and a one-dimensional
model cannot cut the corner:

``curved_2d``
    ``V = V0 sech^2(x/a) + w_y^2 (y - g(x))^2 / 2`` with ``g = b exp(-x^2/c^2)``:
    a planar bend.  The RPH development of a planar path is the plane
    itself, so the RPH 2D model must reproduce the full system.
``curved_3d``
    adds ``w_z(x)^2 (z - h(x))^2 / 2`` with ``h = b_z x exp(-x^2/c_z^2)`` and a
    frequency that softens at the barrier: the path also bends in ``z`` and is
    not planar; the RPH normal can only carry part of the curvature.
``embedded``
    a model plus uncoupled harmonic modes, rotated by a random orthogonal
    matrix.  Uncoupled constant-frequency modes cancel from both the quantum
    and the RPMD rate, so every rate ratio must equal that of the model.

The barrier frequency (1500 cm^-1) and transverse frequencies are those of a
light-atom transfer; the crossover temperature is about 345 K.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from pyrpmd.constants import KB_AU

CM = 1.0 / 219474.6313632


@dataclass(frozen=True)
class Curved:
    """Curved-valley Eckart model in 2 or 3 mass-weighted coordinates."""

    dimension: int = 2
    V0: float = 0.016
    omega_barrier: float = 1500.0 * CM
    omega_y: float = 2000.0 * CM
    b: float = 10.0
    c: float = 20.0
    omega_z: float = 3000.0 * CM
    soften_z: float = 0.3
    b_z: float = 0.25
    c_z: float = 25.0

    @property
    def a(self) -> float:
        return float(np.sqrt(2.0 * self.V0) / self.omega_barrier)

    def _parts(self, q):
        q = np.asarray(q, dtype=float)
        x, y = q[..., 0], q[..., 1]
        sech2 = 1.0 / np.cosh(x / self.a) ** 2
        g = self.b * np.exp(-x**2 / self.c**2)
        dg = -2.0 * x / self.c**2 * g
        value = self.V0 * sech2 + 0.5 * self.omega_y**2 * (y - g) ** 2
        dx = -2.0 * self.V0 / self.a * sech2 * np.tanh(x / self.a) - self.omega_y**2 * (y - g) * dg
        grads = [dx, self.omega_y**2 * (y - g)]
        if self.dimension == 3:
            z = q[..., 2]
            h = self.b_z * x * np.exp(-x**2 / self.c_z**2)
            dh = self.b_z * np.exp(-x**2 / self.c_z**2) * (1.0 - 2.0 * x**2 / self.c_z**2)
            w = self.omega_z * (1.0 - self.soften_z * sech2)
            dw = self.omega_z * self.soften_z * 2.0 / self.a * sech2 * np.tanh(x / self.a)
            value = value + 0.5 * w**2 * (z - h) ** 2
            grads[0] = grads[0] + w * dw * (z - h) ** 2 - w**2 * (z - h) * dh
            grads.append(w**2 * (z - h))
        return value, np.stack(grads, axis=-1)

    def energy(self, q):
        return self._parts(q)[0]

    def gradient(self, q):
        return self._parts(q)[1]


@dataclass(frozen=True)
class Embedded:
    """``base`` plus uncoupled harmonic modes, in coordinates rotated by ``rotation``."""

    base: Curved
    frequencies: tuple
    seed: int = 7

    @property
    def dimension(self) -> int:
        return self.base.dimension + len(self.frequencies)

    @property
    def rotation(self) -> np.ndarray:
        q, r = np.linalg.qr(np.random.default_rng(self.seed).normal(size=(self.dimension,) * 2))
        return q * np.sign(np.diag(r))

    def _split(self, x):
        u = np.asarray(x, dtype=float) @ self.rotation      # model coordinates = R^T x
        return u[..., : self.base.dimension], u[..., self.base.dimension:]

    def energy(self, x):
        core, bath = self._split(x)
        omega = np.asarray(self.frequencies)
        return self.base.energy(core) + 0.5 * np.sum(omega**2 * bath**2, axis=-1)

    def gradient(self, x):
        core, bath = self._split(x)
        omega = np.asarray(self.frequencies)
        model = np.concatenate([self.base.gradient(core), omega**2 * bath], axis=-1)
        return model @ self.rotation.T

    def embed(self, u):
        """Model coordinates (bath at zero) -> rotated Cartesian coordinates."""
        u = np.asarray(u, dtype=float)
        full = np.concatenate([u, np.zeros(u.shape[:-1] + (len(self.frequencies),))], axis=-1)
        return full @ self.rotation.T


def model_run(potential, temperature: float, nbeads: int, *, reference_distance: float = 120.0,
              transition_state=None):
    """A rate-run stand-in for the subspace pipeline on abstract coordinates.

    ``xi`` is linear along the reaction direction at the saddle, 1 there and 0
    ``reference_distance`` towards the reactants (x < 0 in model coordinates).
    """
    dimension = potential.dimension
    ts = np.zeros(dimension) if transition_state is None else np.asarray(transition_state, dtype=float)
    direction = np.zeros(dimension)
    direction[0] = 1.0
    if isinstance(potential, Embedded):
        ts = potential.embed(np.zeros(potential.base.dimension)) if transition_state is None else ts
        direction = potential.embed(np.eye(potential.base.dimension)[0])

    def xi_value_gradient(x):
        value = 1.0 + float((np.asarray(x, dtype=float).ravel() - ts) @ direction) / reference_distance
        return value, direction / reference_distance

    # The rate run only knows a transition-state guess, not the exact saddle.
    guess = ts + 0.3 * np.cos(np.arange(dimension))
    run = SimpleNamespace(
        root=None, beta=1.0 / (KB_AU * temperature), nbeads=int(nbeads), rigid_body=False, xi_star=1.0,
        cartesian_masses=np.ones(dimension), system=SimpleNamespace(geometry=guess, masses=np.ones(dimension)),
        energy=lambda x: float(potential.energy(np.asarray(x, dtype=float))),
        gradient=lambda x: np.asarray(potential.gradient(np.asarray(x, dtype=float)), dtype=float),
        xi_value_gradient=xi_value_gradient,
        xi=lambda x: xi_value_gradient(x)[0],
        xi_gradient=lambda x: xi_value_gradient(x)[1],
    )
    return run
