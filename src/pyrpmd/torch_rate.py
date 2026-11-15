"""Device-resident rate CVs and fixed umbrella forces.

Only algebraic built-in rate coordinates are accepted.  The objects in this
module operate on ``[B,P,N,3]`` tensors and return the full-ring derivative,
so umbrella and flux-metric forces can be added to :class:`TorchDynamics`
without converting coordinates to NumPy.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import Tensor

from .torch_backend import TensorPESResult


def _distance(q, a, b):
    delta = q[:, a] - q[:, b]
    radius = torch.linalg.vector_norm(delta, dim=-1)
    if torch.any(radius < 1e-12).item():
        raise ValueError("rate CV derivative is undefined for coincident atoms")
    gradient = torch.zeros((*q.shape[:2], 3), dtype=q.dtype, device=q.device)
    unit = delta / radius[:, None]
    gradient[:, a], gradient[:, b] = unit, -unit
    return radius, gradient


class TorchCV:
    dimension = 1

    def evaluate_centroid(self, centroid):
        raise NotImplementedError

    def evaluate_tensor(self, q):
        if q.ndim != 4 or q.shape[-1] != 3:
            raise ValueError("Torch rate CV expects [B,P,N,3]")
        value, gradient = self.evaluate_centroid(q.mean(dim=1))
        jacobian = gradient[:, None, None].expand(-1, 1, q.shape[1], -1, -1) / q.shape[1]
        return value[:, None], jacobian


class TorchDistanceCV(TorchCV):
    def __init__(self, atoms):
        if len(atoms) != 2 or atoms[0] == atoms[1]:
            raise ValueError("distance CV needs two distinct atoms")
        self.atoms = tuple(int(x) for x in atoms)

    def evaluate_centroid(self, centroid):
        return _distance(centroid, *self.atoms)


class TorchCartesianCV(TorchCV):
    def __init__(self, atom, axis=0):
        self.atom, self.axis = int(atom), int(axis)
        if self.axis not in (0, 1, 2) or self.atom < 0:
            raise ValueError("invalid Cartesian CV")

    def evaluate_centroid(self, centroid):
        value = centroid[:, self.atom, self.axis]
        gradient = torch.zeros_like(centroid)
        gradient[:, self.atom, self.axis] = 1.0
        return value, gradient


class TorchRPMDrateCV(TorchCV):
    """Tensor implementation of ``RPMDrateCoordinate``'s regular branch."""
    def __init__(self, coordinate):
        self.rinf = float(coordinate.rinf)
        self.weights = None
        self.paths = tuple(coordinate.paths)
        self.reference_difference = tuple(float(x) for x in coordinate.reference_difference)
        self.regular_branch_only = bool(getattr(coordinate, "regular_branch_only", False))
        self._weights_np = np.asarray(coordinate.com.weights, dtype=np.float64)

    def _weights(self, q):
        if self.weights is None or self.weights.device != q.device:
            self.weights = torch.as_tensor(self._weights_np, dtype=q.dtype, device=q.device)
        return self.weights

    def evaluate_surfaces_centroid(self, centroid):
        batch, atoms = centroid.shape[:2]
        weights = self._weights(centroid)
        com = (centroid * weights[None, :, None]).sum(dim=1)
        radius = torch.linalg.vector_norm(com, dim=-1)
        if torch.any(radius < 1e-12).item():
            raise ValueError("rate CV COM distance derivative is undefined")
        com_unit = com / radius[:, None]
        gcom = weights[None, :, None] * com_unit[:, None, :]
        s0 = self.rinf - radius
        gs0 = -gcom

        candidate_values = []
        candidate_gradients = []
        for path, differences in zip(self.paths, (self.reference_difference for _ in self.paths)):
            values, gradients = [], []
            forming, breaking = path
            for ref, f, b in zip(differences, forming, breaking):
                rf, gf = _distance(centroid, *f)
                rb, gb = _distance(centroid, *b)
                values.append(float(ref) - rf + rb)
                gradients.append(-gf + gb)
            candidate_values.append(torch.stack(values, dim=1))
            candidate_gradients.append(torch.stack(gradients, dim=1))
        values = torch.stack(candidate_values, dim=1)       # [B,S,M]
        gradients = torch.stack(candidate_gradients, dim=1) # [B,S,M,N,3]
        maxima = values.amax(dim=1)
        bond = maxima.argmin(dim=1)
        rows = torch.arange(batch, device=centroid.device)
        s1 = maxima[rows, bond]
        path = values[rows, :, bond].argmax(dim=1)
        g1 = gradients[rows, path, bond]
        b = s0 - s1
        if self.regular_branch_only and torch.any(b <= 1e-10).item():
            raise ValueError("RPMDrate xi left the regular branch")
        if torch.any(torch.abs(b) < 1e-10).item():
            raise ValueError("RPMDrate xi is singular where s0=s1")
        return s0, gs0, s1, g1

    def evaluate_centroid(self, centroid):
        s0, gs0, s1, g1 = self.evaluate_surfaces_centroid(centroid)
        b = s0 - s1
        value = s0 / b
        gradient = gs0 / b[:, None, None] - s0[:, None, None] * g1 / b.square()[:, None, None]
        return value, gradient


class TorchRateSurface(TorchCV):
    def __init__(self, surface):
        self.xi = float(surface.xi)
        self.coordinate = TorchRPMDrateCV(surface.coordinate)

    def evaluate_centroid(self, centroid):
        s0, g0, s1, g1 = self.coordinate.evaluate_surfaces_centroid(centroid)
        return ((1 - self.xi) * s0 + self.xi * s1,
                (1 - self.xi) * g0 + self.xi * g1)


class TorchCVStack(TorchCV):
    def __init__(self, cvs):
        self.cvs = tuple(cvs)
        if not self.cvs:
            raise ValueError("Torch CV stack cannot be empty")
        self.dimension = len(self.cvs)

    def evaluate_centroid(self, centroid):
        values, gradients = zip(*(cv.evaluate_centroid(centroid) for cv in self.cvs))
        return torch.stack(values, dim=1), torch.stack(gradients, dim=1)

    def evaluate_tensor(self, q):
        values, gradient = self.evaluate_centroid(q.mean(dim=1))
        jacobian = gradient[:, :, None].expand(-1, -1, q.shape[1], -1, -1) / q.shape[1]
        return values, jacobian


def torch_cv_from_cpu(cv):
    """Convert a checked built-in PyRPMD CV into its tensor counterpart."""
    from .cv import Cartesian, CVStack, Distance, RPMDrateCoordinate
    if isinstance(cv, CVStack):
        return TorchCVStack([torch_cv_from_cpu(component) for component in cv.cvs])
    from .cv import RateSurface
    if isinstance(cv, RateSurface):
        return TorchRateSurface(cv)
    if isinstance(cv, RPMDrateCoordinate):
        return TorchRPMDrateCV(cv)
    if isinstance(cv, Distance):
        return TorchDistanceCV(cv.atoms)
    if isinstance(cv, Cartesian):
        return TorchCartesianCV(cv.atom, cv.axis)
    raise ValueError(f"GPU rate path does not support CV type {type(cv).__name__}")


def _displacement(value, center, periods):
    delta = value - center
    for i, period in enumerate(periods):
        if period is not None:
            delta[:, i] = torch.remainder(delta[:, i] + period / 2, period) - period / 2
    return delta


class TorchUmbrellaBias:
    def __init__(self, cv, center, stiffness, periods=()):
        self.cv = cv
        self.center_np = np.asarray(center, dtype=np.float64)
        matrix = np.asarray(stiffness, dtype=np.float64)
        if matrix.ndim == 0:
            matrix = np.eye(len(self.center_np)) * matrix
        if matrix.ndim == 1:
            matrix = np.diag(matrix)
        if matrix.shape != (len(self.center_np), len(self.center_np)):
            raise ValueError("GPU umbrella stiffness has the wrong dimension")
        if not np.allclose(matrix, matrix.T) or np.linalg.eigvalsh(matrix).min() <= 0:
            raise ValueError("GPU umbrella stiffness must be positive definite")
        self.stiffness_np = matrix
        self.periods = tuple(periods)
        self._center = self._matrix = None

    def evaluate_tensor(self, q):
        values, jacobian = self.cv.evaluate_tensor(q)
        if self._center is None or self._center.device != q.device:
            self._center = torch.as_tensor(self.center_np, dtype=q.dtype, device=q.device)
            self._matrix = torch.as_tensor(self.stiffness_np, dtype=q.dtype, device=q.device)
        delta = _displacement(values, self._center, self.periods)
        kd = delta @ self._matrix.T
        physical = 0.5 * (kd * delta).sum(dim=1)
        beads = q.shape[1]
        force = -beads * torch.einsum("bd,bdpna->bpna", kd, jacobian)
        return TensorPESResult(physical[:, None].expand(-1, beads), force)


class TorchFluxMetricBias:
    def __init__(self, cv, masses, beta):
        self.cv = cv
        self.masses_np = np.asarray(masses, dtype=np.float64)
        self.beta = float(beta)

    def evaluate_tensor(self, q):
        # The CV gradient is differentiable with respect to the centroid. The
        # autograd scope is local to this observable; the physical JaxPIP
        # module still returns analytic forces and is never differentiated.
        with torch.enable_grad():
            centroid = q.mean(dim=1).detach().requires_grad_(True)
            value, gradient = self.cv.evaluate_centroid(centroid)
            masses = torch.as_tensor(self.masses_np, dtype=q.dtype, device=q.device)
            metric = (gradient.square() / masses[None, :, None]).sum(dim=(1, 2))
            if torch.any(metric <= 1e-30).item():
                raise ValueError("rate coordinate has an invalid GPU mass metric")
            hvp = torch.autograd.grad(metric.sum(), centroid, create_graph=False)[0] / 2
            fs = torch.sqrt(metric / (2 * math.pi * self.beta))
            physical = -torch.log(fs) / self.beta
            force = hvp / (self.beta * metric)[:, None, None]
            beads = q.shape[1]
            return TensorPESResult(physical[:, None].expand(-1, beads),
                                   force[:, None].expand(-1, beads, -1, -1))


class TorchBiasStack:
    def __init__(self, biases):
        self.biases = tuple(biases)

    def evaluate_tensor(self, q):
        energy = torch.zeros(q.shape[:2], dtype=q.dtype, device=q.device)
        force = torch.zeros_like(q)
        for bias in self.biases:
            result = bias.evaluate_tensor(q)
            energy.add_(result.energy)
            force.add_(result.forces)
        return TensorPESResult(energy, force)


__all__ = ["TorchCVStack", "TorchRPMDrateCV", "TorchRateSurface", "TorchUmbrellaBias",
           "TorchFluxMetricBias", "TorchBiasStack", "torch_cv_from_cpu"]
