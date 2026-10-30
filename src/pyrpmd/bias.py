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


_BUILTIN_BIAS_METHODS = {
    cls: cls.energy_force for cls in (UmbrellaBias, FluxMetricBias, CompositeBias)
}


class MetadynamicsBias:
    """History-dependent standard or well-tempered metadynamics in CV space.

    Heights are physical energies: ``energy_force`` adds P*V to H_P and
    differentiates the CV's *full* ring Jacobian.  The tempering exponent is
    -beta*V/(gamma-1), never -beta*P*V/(gamma-1).  See Barducci, Bussi and
    Parrinello, PRL 100, 020603 (2008), doi:10.1103/PhysRevLett.100.020603.

    A periodic dimension uses a wrapped Gaussian (a sum over images), not a
    minimum-image Gaussian with a discontinuous force at half a period.
    Depositions are explicit events; evaluating forces never changes history.
    This class deliberately has no frozen native-bias descriptor.
    """

    history_dependent = True

    def __init__(self, cv, sigma, height, beta, *, mode="well_tempered",
                 bias_factor=None, hills=None):
        self.cv = cv
        self.sigma = np.atleast_1d(np.asarray(sigma, float)).copy()
        if self.sigma.shape != (cv.dimension,) or not np.isfinite(self.sigma).all() or np.any(self.sigma <= 0):
            raise ValueError("metadynamics sigma must contain one positive finite width per CV")
        self.sigma.flags.writeable = False
        self.height, self.beta = float(height), float(beta)
        if not np.isfinite(self.height) or self.height <= 0 or not np.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("metadynamics height and beta must be positive and finite")
        if mode not in ("standard", "well_tempered"):
            raise ValueError("metadynamics mode must be standard or well_tempered")
        self.mode = mode
        if mode == "standard":
            if bias_factor is not None:
                raise ValueError("standard metadynamics does not use bias_factor")
            self.bias_factor = None
        else:
            self.bias_factor = float(10.0 if bias_factor is None else bias_factor)
            if not np.isfinite(self.bias_factor) or self.bias_factor <= 1:
                raise ValueError("well-tempered bias_factor must be finite and > 1")
        if len(cv.periods) != cv.dimension or any(
            p is not None and (not np.isfinite(p) or p <= 0) for p in cv.periods
        ):
            raise ValueError("CV periods must be positive finite numbers or None")
        self.hills = []
        for hill in ([] if hills is None else hills):
            self._append(hill)

    @staticmethod
    def _kernel(delta, sigma, period):
        """Unit-height Gaussian and its derivative, including smooth images."""
        if period is None:
            value = np.exp(-0.5 * (delta / sigma) ** 2)
            return value, -delta / sigma**2 * value
        d = (delta + 0.5 * period) % period - 0.5 * period
        ratio = sigma / period
        if ratio >= 0.2:
            # Poisson summation converges quickly for broad Gaussians. The
            # normalization makes the deposited height exactly the input w.
            count = max(1, int(np.ceil(np.sqrt(2 * 36.0) / (2 * np.pi * ratio))))
            modes = np.arange(1, count + 1, dtype=float)
            coeff = np.exp(-0.5 * (2 * np.pi * ratio * modes) ** 2)
            phase = d[..., None] * (2 * np.pi / period) * modes
            norm = 1.0 + 2 * coeff.sum()
            value = (1.0 + 2 * np.sum(coeff * np.cos(phase), axis=-1)) / norm
            derivative = -2 * np.sum(coeff * modes * np.sin(phase), axis=-1) * (2 * np.pi / period) / norm
            return value, derivative
        count = int(np.ceil(8.5 * ratio)) + 1
        images = np.arange(-count, count + 1, dtype=float) * period
        shifts = d[..., None] + images
        terms = np.exp(-0.5 * (shifts / sigma) ** 2)
        norm = np.exp(-0.5 * (images / sigma) ** 2).sum()
        return terms.sum(axis=-1) / norm, np.sum(-shifts / sigma**2 * terms, axis=-1) / norm

    def physical_energy_gradient(self, values):
        values = np.asarray(values, float)
        if values.ndim < 1 or values.shape[-1] != self.cv.dimension or not np.isfinite(values).all():
            raise ValueError("metadynamics values must be finite [...,D]")
        if not self.hills:
            return np.zeros(values.shape[:-1]), np.zeros_like(values)
        centers = np.asarray([h["center"] for h in self.hills])
        heights = np.asarray([h["height"] for h in self.hills])
        delta = values[..., None, :] - centers
        factors, derivatives = [], []
        for j, (sigma, period) in enumerate(zip(self.sigma, self.cv.periods)):
            f, g = self._kernel(delta[..., j], sigma, period)
            factors.append(f)
            derivatives.append(g)
        energy = np.sum(heights * np.prod(factors, axis=0), axis=-1)
        gradient = np.empty_like(values)
        for j in range(self.cv.dimension):
            term = heights * derivatives[j]
            for k in range(self.cv.dimension):
                if k != j:
                    term = term * factors[k]
            gradient[..., j] = np.sum(term, axis=-1)
        return energy, gradient

    def physical_energy(self, values):
        return self.physical_energy_gradient(values)[0]

    def energy_force(self, q):
        result = self.cv.value_and_jacobian(q)
        energy, gradient = self.physical_energy_gradient(result.value)
        p = len(q)
        return p * float(energy), -p * np.einsum("d,dpac->pac", gradient, result.jacobian)

    def _append(self, hill):
        center = np.asarray(hill["center"], float)
        height, time = float(hill["height"]), float(hill["time"])
        step = hill["step"]
        before = float(hill.get("bias_before", 0.0))
        if center.shape != (self.cv.dimension,) or not np.isfinite(center).all():
            raise ValueError("invalid metadynamics hill center")
        if not np.isfinite([height, time, before]).all() or height < 0 or time < 0 or before < -1e-12:
            raise ValueError("invalid metadynamics hill height/time/bias")
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)) or step < 1:
            raise ValueError("metadynamics hill step must be a positive integer")
        if self.hills and (step <= self.hills[-1]["step"] or time <= self.hills[-1]["time"]):
            raise ValueError("metadynamics hill clocks must strictly increase")
        self.hills.append({"center": center.tolist(), "height": height,
                           "step": int(step), "time": time, "bias_before": before})

    def deposit(self, values, *, step, time):
        """Add one hill after a successful integration step, without RNG use."""
        center = np.asarray(values, float)
        if center.shape != (self.cv.dimension,):
            raise ValueError("one metadynamics deposition needs a single [D] center")
        before = float(self.physical_energy(center))
        height = self.height
        if self.mode == "well_tempered":
            height *= np.exp(-self.beta * before / (self.bias_factor - 1.0))
        hill = {"center": center, "height": height, "step": step,
                "time": time, "bias_before": before}
        self._append(hill)
        return dict(self.hills[-1])

    @property
    def fes_factor(self):
        """Asymptotic F=-factor*V+C; finite-time bias is only an estimate."""
        return 1.0 if self.mode == "standard" else self.bias_factor / (self.bias_factor - 1.0)
