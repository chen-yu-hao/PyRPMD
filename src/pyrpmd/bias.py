"""Biases return the energy added to H_P and its full physical force.

User umbrella energies W are physical energies: H_P receives P*W. The flux
metric is a direct mathematical port of RPMDrate/Caracal's MIT-licensed
add_bias_potential/umbrella routines; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations
import numpy as np


def displacement(values, centers, periods=None):
    delta = np.asarray(values, float) - np.asarray(centers, float)
    if periods is not None:
        for i, period in enumerate(periods):
            if period is not None:
                delta[..., i] = (delta[..., i] + period / 2) % period - period / 2
    return delta


class UmbrellaBias:
    def __init__(self, cv, center, stiffness):
        self.cv = cv
        self.center = np.asarray(center, float)
        k = np.asarray(stiffness, float)
        if k.ndim == 0:
            k = np.eye(cv.dimension) * k
        if k.ndim == 1:
            k = np.diag(k)
        if self.center.shape != (cv.dimension,) or k.shape != (
            cv.dimension,
            cv.dimension,
        ):
            raise ValueError("umbrella center/stiffness has the wrong CV dimension")
        if not np.isfinite(k).all() or not np.isfinite(self.center).all():
            raise ValueError("umbrella parameters must be finite")
        if not np.allclose(k, k.T) or np.linalg.eigvalsh(k).min() <= 0:
            raise ValueError("umbrella stiffness must be symmetric positive definite")
        self.stiffness = k

    def physical_energy(self, values):
        d = displacement(values, self.center, self.cv.periods)
        return 0.5 * np.einsum("...i,ij,...j->...", d, self.stiffness, d)

    def energy_force(self, q):
        result = self.cv.value_and_jacobian(q)
        d = displacement(result.value, self.center, self.cv.periods)
        p = len(q)
        energy = p * 0.5 * float(d @ self.stiffness @ d)
        forces = -p * np.einsum("d,dpac->pac", self.stiffness @ d, result.jacobian)
        return energy, forces


class FluxMetricBias:
    """-P*log(fs)/beta with fs=sqrt(g M^-1 g/(2*pi*beta)).

    Only centroid CVs with an analytic centroid Hessian are accepted. This
    belongs to the RPMDrate rate-PMF measure, not generic FES.
    """

    def __init__(self, cv, masses, beta):
        if getattr(cv, "estimator", None) != "centroid":
            raise ValueError("rate flux metric requires a centroid coordinate")
        if not hasattr(cv, "centroid_value_gradient_hessian"):
            raise ValueError("rate flux metric requires the CV Hessian")
        self.cv = cv
        self.masses = np.asarray(masses, float)
        self.beta = float(beta)

    def energy_force(self, q):
        _, g, h = self.cv.centroid_value_gradient_hessian(np.mean(q, axis=0))
        z = float(np.sum(g * g / self.masses[:, None]))
        if not np.isfinite(z) or z <= 1e-30:
            raise ValueError("rate coordinate has a zero/invalid mass metric")
        fs = np.sqrt(z / (2 * np.pi * self.beta))
        f = np.einsum("acbd,bd->ac", h, g / self.masses[:, None]) / (self.beta * z)
        return -len(q) * np.log(fs) / self.beta, np.broadcast_to(f, q.shape).copy()


class CompositeBias:
    def __init__(self, *biases):
        self.biases = tuple(b for b in biases if b is not None)

    def energy_force(self, q):
        energy = 0.0
        force = np.zeros_like(q)
        for bias in self.biases:
            e, f = bias.energy_force(q)
            energy += e
            force += f
        return energy, force
