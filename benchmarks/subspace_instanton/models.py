"""Two-dimensional valley models that isolate the regimes of a subspace correction.

    V(s, Q) = V0 sech^2(s / a) + omega(s)^2 (Q - g(s))^2 / 2,     hbar = m = 1,

with ``a = sqrt(2 V0)`` so that the bare barrier frequency is 1.  The transverse
frequency ``omega(s) = omega_inf - d_omega sech^2(s / a)`` changes the zero-point
energy along the path, and the valley floor ``Q = g(s)`` bends it:

``M1``  zero-point change only, straight path (``g = 0``);
``M2``  curved path only, ``g = -g0 sech^2(s / a)`` (corner cutting);
``M3``  both;
``M4``  a slow bath (``omega = 0.5``, half the barrier frequency) displaced
        between reactants and products, ``g = g0 tanh(s / a_g)``: bilinear
        system--bath coupling at the saddle, the sudden (non-adiabatic) regime.

The parameter values are those of the model family in the package
``subspace_rpmd_code`` supplied for comparison (M1--M4 there); the code here is
an independent implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Valley:
    name: str
    V0: float = 2.1
    omega_inf: float = 1.5
    d_omega: float = 0.0
    bend: str = "none"            # "none", "sech2" or "tanh"
    g0: float = 0.0
    a_g: float | None = None

    @property
    def a(self) -> float:
        return float(np.sqrt(2.0 * self.V0))

    @property
    def width_g(self) -> float:
        return self.a if self.a_g is None else float(self.a_g)

    # --- ingredients, vectorized over s --------------------------------------
    def barrier(self, s):
        return self.V0 / np.cosh(s / self.a) ** 2

    def barrier_slope(self, s):
        return -2.0 * self.V0 / self.a * np.tanh(s / self.a) / np.cosh(s / self.a) ** 2

    def omega(self, s):
        return self.omega_inf - self.d_omega / np.cosh(s / self.a) ** 2

    def omega_slope(self, s):
        return 2.0 * self.d_omega / self.a * np.tanh(s / self.a) / np.cosh(s / self.a) ** 2

    def floor(self, s):
        if self.bend == "none":
            return np.zeros_like(np.asarray(s, dtype=float))
        if self.bend == "sech2":
            return -self.g0 / np.cosh(s / self.width_g) ** 2
        return self.g0 * np.tanh(s / self.width_g)

    def floor_slope(self, s):
        if self.bend == "none":
            return np.zeros_like(np.asarray(s, dtype=float))
        if self.bend == "sech2":
            return 2.0 * self.g0 / self.width_g * np.tanh(s / self.width_g) / np.cosh(s / self.width_g) ** 2
        return self.g0 / self.width_g / np.cosh(s / self.width_g) ** 2

    # --- potential on points [..., 2] ------------------------------------------
    def energy(self, x):
        x = np.asarray(x, dtype=float)
        s, q = x[..., 0], x[..., 1]
        return self.barrier(s) + 0.5 * self.omega(s) ** 2 * (q - self.floor(s)) ** 2

    def gradient(self, x):
        x = np.asarray(x, dtype=float)
        s, q = x[..., 0], x[..., 1]
        w, d = self.omega(s), q - self.floor(s)
        ds = self.barrier_slope(s) + w * self.omega_slope(s) * d * d - w * w * d * self.floor_slope(s)
        return np.stack([ds, w * w * d], axis=-1)

    def hessian(self, x, h=1e-5):
        """Central differences of the gradient, ``[..., 2, 2]``."""
        x = np.asarray(x, dtype=float)
        out = np.empty(x.shape + (2,))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            out[..., j] = (self.gradient(x + e) - self.gradient(x - e)) / (2.0 * h)
        return 0.5 * (out + np.swapaxes(out, -1, -2))

    def saddle(self):
        """Saddle point, unstable frequency, stable frequency and unstable mode (pointing to +s)."""
        x = np.array([0.0, float(self.floor(0.0))])
        values, vectors = np.linalg.eigh(self.hessian(x))
        mode = vectors[:, 0] * (1.0 if vectors[0, 0] >= 0 else -1.0)
        return x, float(np.sqrt(-values[0])), float(np.sqrt(values[1])), mode


def family() -> dict[str, Valley]:
    return {
        "M1": Valley("M1", omega_inf=1.8, d_omega=0.7),
        "M2": Valley("M2", omega_inf=1.5, bend="sech2", g0=1.0),
        "M3": Valley("M3", omega_inf=1.8, d_omega=0.7, bend="sech2", g0=1.0),
        "M4": Valley("M4", omega_inf=0.5, bend="tanh", g0=2.0 * np.sqrt(2.0), a_g=2.0),
    }


def eckart(v: float, omega_b: float = 1.0):
    """Symmetric Eckart barrier ``v sech^2(x/a)`` with barrier frequency ``omega_b`` (energy in units of hbar omega_b = 1)."""
    a = np.sqrt(2.0 * v) / omega_b
    potential = lambda x: v / np.cosh(np.asarray(x) / a) ** 2
    slope = lambda x: -2.0 * v / a * np.tanh(np.asarray(x) / a) / np.cosh(np.asarray(x) / a) ** 2
    curvature = lambda x: 2.0 * v / a**2 * (2.0 * np.tanh(np.asarray(x) / a) ** 2 - 1.0 / np.cosh(np.asarray(x) / a) ** 2) \
        / np.cosh(np.asarray(x) / a) ** 2
    return potential, slope, curvature, a
