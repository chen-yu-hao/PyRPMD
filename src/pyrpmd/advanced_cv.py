"""Differentiable torsion, coordination, alignment, path and learned CVs.

All Cartesian coordinates are continuous/unwrapped and in bohr. Periodic
bond vectors are only used when a cell is explicitly supplied. Path distances
are weighted, optimally aligned mean-square displacements divided by scale²;
both path outputs are dimensionless. See ``docs/tier2-sampling.md``.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

from .cv import CVResult, CollectiveVariable, _coordinates, _indices


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a positive finite number")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _ring(q):
    raw = np.asarray(getattr(q, "q", q))
    if raw.dtype.kind not in "iuf":
        raise ValueError("CV coordinates must be real numbers")
    q = _coordinates(raw)
    if min(q.shape[:2]) < 1:
        raise ValueError("CV coordinates need at least one bead and atom")
    return q


def _atoms(atoms, n=None, *, count=None):
    indices = _indices(atoms)
    if (not indices or len(set(indices)) != len(indices) or min(indices) < 0
            or (count is not None and len(indices) != count)
            or (n is not None and max(indices) >= n)):
        raise ValueError("CV atoms must be distinct nonnegative in-range indices")
    return indices


class Dihedral(CollectiveVariable):
    """Signed torsion in [-pi,pi], with analytic derivatives.

    The sign is atan2((u × v)·w, v·w), where u is the central 1→2
    bond and v,w are the terminal bonds projected perpendicular to u.
    A bead average is a circular mean, and fails for an undefined resultant.
    Collinear bonds have no defined torsion and are rejected.
    """

    units = ("radian",)
    periods = (2 * np.pi,)

    def __init__(self, atoms, *, cell=None, pbc=None, **kwargs):
        super().__init__(**kwargs)
        self.atoms = _atoms(atoms, count=4)
        from .geometry import FixedCellGeometry

        self._geometry = FixedCellGeometry(cell, pbc)

    def centroid_value_gradient(self, q):
        q = np.asarray(q, float)
        a, b, c, d = _atoms(self.atoms, len(q), count=4)
        b0, b1, b2 = self._geometry.minimum_image(
            np.asarray([q[a] - q[b], q[c] - q[b], q[d] - q[c]])
        )
        length = np.linalg.norm(b1)
        if length < 1e-12:
            raise ValueError("dihedral central bond has zero length")
        u = b1 / length
        av, aw = float(b0 @ u), float(b2 @ u)
        v, w = b0 - av * u, b2 - aw * u
        if min(np.linalg.norm(v), np.linalg.norm(w)) < 1e-12:
            raise ValueError("dihedral is undefined for collinear bonds")
        x, y = float(v @ w), float(np.cross(u, v) @ w)
        norm = x * x + y * y
        dx, dy = -y / norm, x / norm
        gv = dx * w + dy * np.cross(w, u)
        gw = dx * v + dy * np.cross(u, v)
        gu = dy * np.cross(v, w)
        gu -= av * gv + b0 * float(gv @ u)
        gu -= aw * gw + b2 * float(gw @ u)
        gb0 = gv - u * float(gv @ u)
        gb2 = gw - u * float(gw @ u)
        gb1 = (gu - u * float(gu @ u)) / length
        grad = np.zeros_like(q)
        grad[a], grad[b], grad[c], grad[d] = gb0, -gb0 - gb1, gb1 - gb2, gb2
        return float(np.arctan2(y, x)), grad

    def value_and_jacobian(self, q):
        q = _ring(q)
        if self.estimator == "centroid":
            return super().value_and_jacobian(q)
        results = [self.centroid_value_gradient(bead) for bead in q]
        angles = np.array([v for v, _ in results])
        sine, cosine = np.sin(angles).mean(), np.cos(angles).mean()
        resultant = sine * sine + cosine * cosine
        if resultant < 1e-24:
            raise ValueError("bead-average dihedral has an undefined circular mean")
        factor = (cosine * np.cos(angles) + sine * np.sin(angles)) / (len(q) * resultant)
        jacobian = np.asarray([g for _, g in results]) * factor[:, None, None]
        return CVResult(np.array([np.arctan2(sine, cosine)]), jacobian[None])


class CoordinationNumber(CollectiveVariable):
    """Sum 1/(1+(r/r0)**n), the rational n:2n coordination switch.

    With one group, each unordered pair occurs once. Two groups must be
    disjoint. Even n>=2 makes the switch differentiable at r=0. The switch
    and derivative are evaluated without powers overflowing at long range.
    """

    def __init__(self, group_a, group_b=None, *, r0, exponent=6, cell=None, pbc=None, **kwargs):
        super().__init__(**kwargs)
        self.group_a = _atoms(group_a)
        self.group_b = None if group_b is None else _atoms(group_b)
        if self.group_b is not None and set(self.group_a).intersection(self.group_b):
            raise ValueError("coordination groups must be disjoint")
        self.r0 = _positive(r0, "coordination r0")
        if (isinstance(exponent, (bool, np.bool_)) or not isinstance(exponent, (int, np.integer))
                or exponent < 2 or exponent % 2):
            raise ValueError("coordination exponent must be an even integer >=2")
        self.exponent = int(exponent)
        self.pairs = (tuple(combinations(self.group_a, 2)) if self.group_b is None
                      else tuple((a, b) for a in self.group_a for b in self.group_b))
        if not self.pairs:
            raise ValueError("coordination needs at least one atom pair")
        from .geometry import FixedCellGeometry

        self._geometry = FixedCellGeometry(cell, pbc)

    def centroid_value_gradient(self, q):
        q = np.asarray(q, float)
        _atoms(self.group_a, len(q))
        if self.group_b is not None:
            _atoms(self.group_b, len(q))
        pairs = np.asarray(self.pairs, int)
        vector = self._geometry.minimum_image(q[pairs[:, 0]] - q[pairs[:, 1]])
        radius = np.linalg.norm(vector, axis=1)
        with np.errstate(divide="ignore"):
            log_power = self.exponent * (np.log(radius) - np.log(self.r0))
        switch = expit(-log_power)
        gradient = np.zeros_like(vector)
        nonzero = radius > 0
        coefficient = -self.exponent * switch[nonzero] * expit(log_power[nonzero]) / radius[nonzero]
        gradient[nonzero] = coefficient[:, None] * (vector[nonzero] / radius[nonzero, None])
        grad = np.zeros_like(q)
        np.add.at(grad, pairs[:, 0], gradient)
        np.add.at(grad, pairs[:, 1], -gradient)
        return float(switch.sum()), grad


def _reference(value):
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix.lower() != ".npy":
            raise ValueError("CV reference files must be .npy arrays in bohr (allow_pickle=False)")
        value = np.load(path, allow_pickle=False)
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError("CV references must be finite real coordinates")
    return np.array(raw, dtype=float, copy=True)


class RMSD(CollectiveVariable):
    """Weighted RMSD after a proper Kabsch rotation and/or translation.

    The same normalized weights are used for alignment and displacement, so
    the optimal-rotation envelope theorem gives the exact Cartesian gradient.
    RMSD has no unique gradient at an exact match; use squared_value_gradient
    (as PathCV does) there. Reflections are never allowed.
    """

    units = ("bohr",)

    def __init__(self, reference, *, atoms=None, weights=None, align="optimal", **kwargs):
        super().__init__(**kwargs)
        self.reference = _reference(reference)
        if self.reference.ndim != 2 or self.reference.shape[-1] != 3 or not len(self.reference):
            raise ValueError("RMSD reference must have shape [N,3]")
        self.atoms = _atoms(range(len(self.reference)) if atoms is None else atoms, len(self.reference))
        self.align = align
        if align not in ("optimal", "center", "none"):
            raise ValueError("RMSD align must be optimal, center or none")
        raw = np.ones(len(self.atoms)) if weights is None else np.asarray(weights)
        if (raw.dtype.kind not in "iuf" or raw.shape != (len(self.atoms),)
                or not np.isfinite(raw).all() or np.any(raw <= 0)):
            raise ValueError("RMSD weights must be positive finite values for selected atoms")
        scaled = np.asarray(raw, float) / np.max(raw)
        self.weights = scaled / np.sum(scaled)
        self.reference.flags.writeable = False
        self.weights.flags.writeable = False

    def squared_value_gradient(self, q):
        q = np.asarray(q, float)
        if q.shape != self.reference.shape or not np.isfinite(q).all():
            raise ValueError("RMSD geometry must match its finite [N,3] reference")
        ids = list(self.atoms)
        x, y = q[ids].copy(), self.reference[ids].copy()
        w = self.weights
        if self.align != "none":
            x -= np.einsum("n,nc->c", w, x)
            y -= np.einsum("n,nc->c", w, y)
        rotation = np.eye(3)
        if self.align == "optimal":
            u, singular, vt = np.linalg.svd(x.T @ (w[:, None] * y))
            determinant = np.linalg.det(u @ vt)
            # At a nonunique optimum the distance can have a cusp. A straight
            # or single-point reference remains valid (rotational symmetry),
            # provided the unresolved rotations leave the reference unchanged.
            # A collinear/collapsed current geometry against a reference with
            # higher rank has different forces on different optimal branches.
            rank = int(np.count_nonzero(singular > 1e-12 * singular[0]))
            if rank < 2:
                reference_singular = np.linalg.svd(np.sqrt(w)[:, None] * y, compute_uv=False)
                reference_rank = int(np.count_nonzero(reference_singular > 1e-12 * reference_singular[0]))
                if reference_rank > rank:
                    raise ValueError("RMSD optimal rotation has a nonunique gradient at this rank-deficient geometry")
            if determinant < 0 and singular[1] > 1e-14 and abs(singular[1] - singular[2]) < 1e-12 * singular[0]:
                raise ValueError("RMSD optimal proper rotation is nonunique")
            rotation = u @ np.diag([1.0, 1.0, np.sign(determinant)]) @ vt
        residual = x @ rotation - y
        value = float(np.sum(w[:, None] * residual**2))
        local = 2 * w[:, None] * (residual @ rotation.T)
        if self.align != "none":
            local -= w[:, None] * local.sum(axis=0)
        grad = np.zeros_like(q)
        grad[ids] = local
        return value, grad

    def centroid_value_gradient(self, q):
        squared, grad = self.squared_value_gradient(q)
        value = np.sqrt(squared)
        if value < 1e-12:
            raise ValueError("RMSD gradient is undefined at an exact reference match; use a squared-distance path CV")
        return float(value), grad / (2 * value)


class PathCV:
    """Smooth path progress s and distance z from reference configurations.

    d_k = MSD(q,ref_k)/scale²; w_k = exp(-lambda*d_k)/sum(exp).
    s = sum(k*w_k), with one-based node indices, and
    z = -log(sum(exp(-lambda*d_k)))/lambda. The conventional unnormalized
    sum is intentional: z may be negative close to several reference nodes.
    """

    def __init__(self, references, *, path_lambda, scale=1.0, component="both", atoms=None,
                 weights=None, align="optimal", estimator="centroid", name=None):
        references = _reference(references)
        if references.ndim != 3 or references.shape[0] < 2 or references.shape[-1] != 3:
            raise ValueError("path references must have shape [M,N,3], M>=2")
        self.path_lambda = _positive(path_lambda, "path_lambda")
        self.scale = _positive(scale, "path scale")
        if not np.isfinite(self.scale**2) or self.scale**2 == 0:
            raise ValueError("path scale squared must be representable")
        if component not in ("s", "z", "both") or estimator not in ("centroid", "bead_average"):
            raise ValueError("path needs component s/z/both and estimator centroid/bead_average")
        self.component, self.estimator = component, estimator
        self.name = name or "PathCV"
        self.dimension = 2 if component == "both" else 1
        self.units = ("dimensionless",) * self.dimension
        self.periods = (None,) * self.dimension
        self.references = tuple(RMSD(r, atoms=atoms, weights=weights, align=align) for r in references)

    def _evaluate(self, q):
        results = [r.squared_value_gradient(q) for r in self.references]
        distances = np.asarray([v for v, _ in results]) / self.scale**2
        gradients = np.asarray([g for _, g in results]) / self.scale**2
        logs = -self.path_lambda * distances
        normalizer = logsumexp(logs)
        weights = np.exp(logs - normalizer)
        nodes = np.arange(1, len(weights) + 1)
        progress = float(nodes @ weights)
        values = np.array([progress, -normalizer / self.path_lambda])
        jacobian = np.array([
            -self.path_lambda * np.einsum("k,kac->ac", weights * (nodes - progress), gradients),
            np.einsum("k,kac->ac", weights, gradients),
        ])
        selection = slice(None) if self.component == "both" else [0 if self.component == "s" else 1]
        return values[selection], jacobian[selection]

    def value_and_jacobian(self, q):
        q = _ring(q)
        if self.estimator == "centroid":
            values, grad = self._evaluate(q.mean(axis=0))
            jacobian = np.broadcast_to(grad[:, None] / len(q), (self.dimension,) + q.shape).copy()
        else:
            results = [self._evaluate(bead) for bead in q]
            values = np.mean([v for v, _ in results], axis=0)
            jacobian = np.stack([g for _, g in results], axis=1) / len(q)
        if not np.isfinite(values).all() or not np.isfinite(jacobian).all():
            raise ValueError("path values or derivatives are not representable")
        return CVResult(values, jacobian)

    def value(self, q):
        return self.value_and_jacobian(q).value


class TorchScriptCV:
    """Autograd CV loaded from a process-local TorchScript model.

    Models receive float64 positions in bohr and return one finite tensor.
    Centroid models map [N,3]→[D], bead-average models use the same map on each
    bead, and full-ring models map [P,N,3]→[D]. Explicit metadata prevents
    guessing units or output dimensions from a model. No finite differences,
    training, or implicit coordinate/unit conversion is performed.
    """

    def __init__(self, model, *, dimension=1, units=None, periods=None, estimator="centroid",
                 device="cpu", name=None):
        if (isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, (int, np.integer))
                or dimension < 1):
            raise ValueError("TorchScript CV dimension must be a positive integer")
        self.dimension = int(dimension)
        self.units = tuple(("dimensionless",) * dimension if units is None else units)
        self.periods = tuple((None,) * dimension if periods is None else periods)
        if (len(self.units) != dimension or len(self.periods) != dimension
                or any(u not in ("bohr", "hartree", "radian", "dimensionless") for u in self.units)
                or any(p is not None and (not np.isfinite(p) or p <= 0) for p in self.periods)):
            raise ValueError("TorchScript CV metadata must declare canonical units and valid periods per output")
        if estimator not in ("centroid", "bead_average", "full_ring"):
            raise ValueError("unknown TorchScript CV estimator")
        if estimator == "bead_average" and any(p is not None for p in self.periods):
            raise ValueError("periodic TorchScript bead averages need an explicit circular model")
        self.model_path = str(Path(model).resolve())
        self.device, self.estimator = str(device), estimator
        self.name = name or Path(model).stem
        try:
            import torch
        except ImportError as exc:
            raise ImportError("TorchScript CVs require PyTorch; install PyRPMD[gpu]") from exc
        self._torch = torch
        from .io import file_sha256

        self.model_sha256 = file_sha256(self.model_path)
        self.model = torch.jit.load(self.model_path, map_location=self.device).eval().to(dtype=torch.float64)
        if file_sha256(self.model_path) != self.model_sha256:
            raise ValueError("TorchScript CV model changed while loading")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def value_and_jacobian(self, q):
        q = _ring(q)
        torch = self._torch
        with torch.enable_grad():
            tensor = torch.tensor(q.copy(), dtype=torch.float64, device=self.device, requires_grad=True)
            if self.estimator == "centroid":
                output = self.model(tensor.mean(dim=0))
            elif self.estimator == "bead_average":
                output = torch.stack([self.model(bead) for bead in tensor]).mean(dim=0)
            else:
                output = self.model(tensor)
            if (not isinstance(output, torch.Tensor) or tuple(output.shape) != (self.dimension,)
                    or not output.is_floating_point() or output.is_complex() or not bool(torch.isfinite(output).all())):
                raise ValueError("TorchScript CV must return one finite real [D] tensor")
            if not output.requires_grad:
                raise ValueError("TorchScript CV output is detached from positions; autograd is required")
            jacobian = []
            for index in range(self.dimension):
                if output[index].requires_grad:
                    gradient = torch.autograd.grad(output[index], tensor, retain_graph=index + 1 < self.dimension,
                                                   allow_unused=True)[0]
                else:
                    gradient = None
                jacobian.append(torch.zeros_like(tensor) if gradient is None else gradient)
            jacobian = torch.stack(jacobian)
            if not bool(torch.isfinite(jacobian).all()):
                raise ValueError("TorchScript CV returned nonfinite derivatives")
            return CVResult(output.detach().cpu().numpy().copy(), jacobian.detach().cpu().numpy().copy())

    def value(self, q):
        return self.value_and_jacobian(q).value
