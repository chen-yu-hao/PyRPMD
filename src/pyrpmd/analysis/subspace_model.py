"""One- to three-dimensional subspace Hamiltonians for RPMD defect correction.

A subspace model ``H_S = T_S + V_S`` is the object shared by the exact
quantum solver (:mod:`pyrpmd.analysis.subspace_quantum`) and the ring-polymer
solver (:mod:`pyrpmd.analysis.subspace_rpmd`).  The correction factor

    C_S = k_QM(H_S) / k_RPMD(H_S)

measures only the RPMD *method* error inside ``S`` because numerator and
denominator use the same Hamiltonian.  The full-dimensional RPMD rate keeps
the environment (zero-point energy, adiabatic following, recrossing) and is
multiplied by ``C_S``.

Coordinates are mass weighted (``s = sqrt(m) x``, atomic units, hbar = 1), so
every subspace axis of a projected model has unit mass.  Axis 0 is the
reaction coordinate; the dividing surface is ``s_0 = dividing_surface`` and
the product side is ``s_0 > dividing_surface``.

Three constructions are provided:

``model_from_function``
    The Hamiltonian already is one-, two- or three-dimensional.
``project_linear_subspace``
    A linear subspace of a larger system.  The complement is frozen,
    statically relaxed, or treated as a vibrationally adiabatic harmonic bath
    (relaxed energy plus the complement zero-point energy).
``project_path_subspace``
    A curved one-dimensional path (for example a ring-polymer mean path) with
    a normal-cut vibrationally adiabatic bath and arc-length coordinate.

The environment treatment fixes ``H_S``; it is not a claim about the full
system.  The research note's curvature-times-width diagnostic,
:func:`curvature_width_criterion`, indicates when a one-dimensional path
cannot carry the transverse quantum width and a further coordinate belongs in
``S``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.linalg import null_space
from scipy.special import logsumexp

MAX_DIMENSION = 3
_REACTANT_KINDS = ("asymptotic", "well")


def _axis(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size < 8 or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be a finite one-dimensional grid with at least eight points")
    axis = np.array(raw, dtype=float, copy=True)
    step = np.diff(axis)
    if step[0] <= 0 or not np.allclose(step, step[0], rtol=1e-9, atol=0.0):
        raise ValueError(f"{name} must be strictly increasing and uniformly spaced")
    axis.flags.writeable = False
    return axis


def _positive_vector(value: Any, size: int, name: str) -> np.ndarray:
    raw = np.asarray(value, dtype=float)
    if raw.ndim == 0:
        raw = np.full(size, float(raw))
    if raw.shape != (size,) or not np.isfinite(raw).all() or np.any(raw <= 0):
        raise ValueError(f"{name} must be positive with one value per coordinate")
    out = raw.copy()
    out.flags.writeable = False
    return out


def _snake_indices(shape: Sequence[int]):
    """Boustrophedon order over a tensor grid; neighbours are adjacent points."""
    if len(shape) == 1:
        yield from ((i,) for i in range(shape[0]))
        return
    for i in range(shape[0]):
        inner = list(_snake_indices(shape[1:]))
        if i % 2:
            inner.reverse()
        for rest in inner:
            yield (i, *rest)


class GridPotential:
    """Tensor-product cubic spline of a gridded potential with gradients.

    Outside the grid the potential is continued as a constant along the
    exceeded axes (zero force there): flat in an asymptotic channel, and no
    polynomial blow-up for trajectories that run out of the box.  Such points
    are counted in :attr:`outside_fraction`; a ring polymer that leaves the
    grid is sampling a potential the exact quantum calculation never saw.
    """

    def __init__(self, axes: Sequence[np.ndarray], values: np.ndarray):
        from scipy.interpolate import make_interp_spline

        self.axes = tuple(axes)
        self.dimension = len(self.axes)
        self.lower = np.array([a[0] for a in self.axes])
        self.upper = np.array([a[-1] for a in self.axes])
        self.evaluations = 0
        self.outside = 0
        coefficients = np.asarray(values, dtype=float)
        knots = []
        for index, axis in enumerate(self.axes):
            spline = make_interp_spline(axis, coefficients, k=3, axis=index)
            knots.append(spline.t)
            coefficients = np.moveaxis(spline.c, 0, index)
        self._fallback = None
        try:
            from scipy.interpolate import NdBSpline
        except ImportError:  # pragma: no cover - SciPy < 1.12
            NdBSpline = None
        if NdBSpline is not None:
            self._spline = NdBSpline(tuple(knots), coefficients, 3)
        elif self.dimension == 1:  # pragma: no cover - SciPy < 1.12
            from scipy.interpolate import BSpline

            self._spline = None
            base = BSpline(knots[0], coefficients, 3)
            self._fallback = (base, base.derivative())
        else:  # pragma: no cover - SciPy < 1.12
            from scipy.interpolate import RegularGridInterpolator

            self._spline = None
            self._fallback = RegularGridInterpolator(
                self.axes, np.asarray(values, dtype=float), method="cubic",
                bounds_error=False, fill_value=None,
            )

    @property
    def outside_fraction(self) -> float:
        return float(self.outside / self.evaluations) if self.evaluations else 0.0

    def _count(self, flat: np.ndarray) -> None:
        self.evaluations += len(flat)
        self.outside += int(np.count_nonzero(np.any((flat < self.lower) | (flat > self.upper), axis=1)))

    def energy(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        flat = np.clip(points.reshape(-1, self.dimension), self.lower, self.upper)
        if self._spline is not None:
            value = self._spline(flat)
        elif isinstance(self._fallback, tuple):  # pragma: no cover
            value = self._fallback[0](flat[:, 0])
        else:  # pragma: no cover
            value = self._fallback(flat)
        return np.asarray(value, dtype=float).reshape(points.shape[:-1])

    def energy_and_gradient(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float)
        raw = points.reshape(-1, self.dimension)
        self._count(raw)
        flat = np.clip(raw, self.lower, self.upper)
        clamped = flat != raw
        gradient = np.empty_like(flat)
        if self._spline is not None:
            value = self._spline(flat)
            for axis in range(self.dimension):
                order = [0] * self.dimension
                order[axis] = 1
                gradient[:, axis] = self._spline(flat, nu=tuple(order))
        elif isinstance(self._fallback, tuple):  # pragma: no cover
            value = self._fallback[0](flat[:, 0])
            gradient[:, 0] = self._fallback[1](flat[:, 0])
        else:  # pragma: no cover
            value = self._fallback(flat)
            for axis in range(self.dimension):
                step = 1e-4 * (self.axes[axis][1] - self.axes[axis][0])
                shift = np.zeros(self.dimension)
                shift[axis] = step
                gradient[:, axis] = (self._fallback(flat + shift) - self._fallback(flat - shift)) / (2 * step)
        gradient[clamped] = 0.0
        return (np.asarray(value, dtype=float).reshape(points.shape[:-1]),
                gradient.reshape(points.shape))


@dataclass(frozen=True)
class SubspaceModel:
    """Gridded 1-3D Hamiltonian ``-sum_a d^2/(2 m_a ds_a^2) + V_S(s)``.

    ``values`` is ``V_S`` on the tensor grid ``axes`` (``indexing='ij'``).
    ``function``/``gradient``, when given, are vectorized callables on
    ``[..., dimension]`` arrays used by the ring-polymer solver; otherwise a
    cubic spline of ``values`` is used.  ``reactant='asymptotic'`` normalizes
    rates per unit length of axis 0 at ``reactant_reference`` (a scattering
    reactant); ``reactant='well'`` normalizes by the population on the
    reactant side of the grid.
    """

    axes: tuple
    values: np.ndarray
    masses: np.ndarray
    dividing_surface: float
    reactant: str = "asymptotic"
    reactant_reference: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    function: Callable[[np.ndarray], np.ndarray] | None = None
    gradient: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self):
        axes = tuple(_axis(a, f"axes[{i}]") for i, a in enumerate(self.axes))
        if not 1 <= len(axes) <= MAX_DIMENSION:
            raise ValueError("a subspace model must have one to three coordinates")
        shape = tuple(len(a) for a in axes)
        values = np.array(self.values, dtype=float, copy=True)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"values must be finite with grid shape {shape}")
        values.flags.writeable = False
        masses = _positive_vector(self.masses, len(axes), "masses")
        surface = float(self.dividing_surface)
        axis0 = axes[0]
        if not axis0[1] < surface < axis0[-2]:
            raise ValueError("dividing_surface must lie inside the reaction-coordinate grid")
        if self.reactant not in _REACTANT_KINDS:
            raise ValueError(f"reactant must be one of {_REACTANT_KINDS}")
        reference = self.reactant_reference
        if reference is None:
            reference = float(axis0[0] + 0.25 * (axis0[-1] - axis0[0]))
        reference = float(reference)
        if not axis0[0] <= reference < surface:
            raise ValueError("reactant_reference must lie on the reactant side of the dividing surface")
        for name in ("function", "gradient"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable or None")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "dividing_surface", surface)
        object.__setattr__(self, "reactant_reference", reference)
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "_interpolant", None)

    @property
    def dimension(self) -> int:
        return len(self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(a) for a in self.axes)

    @property
    def spacing(self) -> np.ndarray:
        return np.array([a[1] - a[0] for a in self.axes])

    def grid_points(self) -> np.ndarray:
        return np.stack(np.meshgrid(*self.axes, indexing="ij"), axis=-1)

    @property
    def interpolant(self) -> GridPotential:
        if self._interpolant is None:
            object.__setattr__(self, "_interpolant", GridPotential(self.axes, self.values))
        return self._interpolant

    def energy(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if self.function is not None:
            return np.asarray(self.function(points), dtype=float)
        return self.interpolant.energy(points)

    def energy_and_gradient(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float)
        if points.shape[-1] != self.dimension:
            raise ValueError("points must end with the subspace dimension")
        if self.function is None:
            return self.interpolant.energy_and_gradient(points)
        value = np.asarray(self.function(points), dtype=float)
        if self.gradient is not None:
            gradient = np.asarray(self.gradient(points), dtype=float)
        else:
            gradient = np.empty_like(points)
            for axis in range(self.dimension):
                step = 1e-4 * self.spacing[axis]
                shift = np.zeros(self.dimension)
                shift[axis] = step
                gradient[..., axis] = (self.function(points + shift) - self.function(points - shift)) / (2 * step)
        return value, gradient

    def describe(self) -> dict:
        """JSON-safe summary of the Hamiltonian, without the grid values."""
        return {
            "dimension": self.dimension,
            "shape": list(self.shape),
            "bounds": [[float(a[0]), float(a[-1])] for a in self.axes],
            "masses": self.masses.tolist(),
            "dividing_surface": self.dividing_surface,
            "reactant": self.reactant,
            "reactant_reference": self.reactant_reference,
            "potential_minimum": float(self.values.min()),
            "potential_source": "callable" if self.function is not None else "cubic spline of the grid",
            "construction": dict(self.provenance),
        }


def model_from_function(
    potential: Callable[[np.ndarray], np.ndarray],
    axes: Sequence[Any],
    *,
    gradient: Callable[[np.ndarray], np.ndarray] | None = None,
    masses: Any = 1.0,
    dividing_surface: float = 0.0,
    reactant: str = "asymptotic",
    reactant_reference: float | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> SubspaceModel:
    """Wrap a vectorized 1-3D potential ``V(points[..., d])`` as a model."""
    if not callable(potential):
        raise TypeError("potential must be callable")
    grids = tuple(_axis(a, f"axes[{i}]") for i, a in enumerate(axes))
    points = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)
    values = np.asarray(potential(points), dtype=float)
    info = {"kind": "function"}
    info.update(dict(provenance or {}))
    return SubspaceModel(grids, values, masses, dividing_surface, reactant, reactant_reference,
                         info, potential, gradient)


def _orthonormal_basis(basis: Any, size: int) -> np.ndarray:
    raw = np.asarray(basis, dtype=float)
    if raw.ndim == 1:
        raw = raw[:, None]
    if raw.ndim != 2 or raw.shape[0] != size or not 1 <= raw.shape[1] <= MAX_DIMENSION:
        raise ValueError("basis must have shape [coordinates, 1..3] in mass-weighted space")
    if not np.isfinite(raw).all():
        raise ValueError("basis must be finite")
    q, r = np.linalg.qr(raw)
    if np.min(np.abs(np.diag(r))) <= 1e-10 * max(1.0, np.max(np.abs(r))):
        raise ValueError("basis columns must be linearly independent")
    # Keep the user's orientation for each column.
    return q * np.sign(np.diag(r))[None, :]


def _complement_hessian(x, complement, sqrt_m, gradient, hessian, step):
    if hessian is not None:
        h = np.asarray(hessian(x), dtype=float)
        if h.shape != (len(x), len(x)) or not np.isfinite(h).all():
            raise ValueError("hessian must return a finite [n, n] matrix")
        mass_weighted = h / sqrt_m[:, None] / sqrt_m[None, :]
        k = complement.T @ mass_weighted @ complement
    else:
        k = np.empty((complement.shape[1], complement.shape[1]))
        for column in range(complement.shape[1]):
            dx = step * complement[:, column] / sqrt_m
            difference = (np.asarray(gradient(x + dx), float) - np.asarray(gradient(x - dx), float)) / (2 * step)
            k[:, column] = complement.T @ (difference / sqrt_m)
    return 0.5 * (k + k.T)


def _vibrational_complement(x, directions, sqrt_m, rigid_body):
    """Mass-weighted complement of ``directions``, optionally without rigid-body modes."""
    columns = [np.asarray(directions, dtype=float).reshape(len(sqrt_m), -1)]
    if rigid_body:
        from .centroid_path import rigid_body_modes

        columns.append(rigid_body_modes(np.asarray(x).reshape(-1, 3), (sqrt_m**2).reshape(-1, 3)[:, 0]))
    block = np.column_stack(columns)
    q, singular, _ = np.linalg.svd(block, full_matrices=False)
    span = q[:, singular > 1e-10 * singular.max()]
    return null_space(span.T) if span.shape[1] < len(sqrt_m) else np.zeros((len(sqrt_m), 0))


def _bath_zero_point(k: np.ndarray) -> tuple[float, float]:
    eigenvalues = np.linalg.eigvalsh(k) if k.size else np.zeros(0)
    return 0.5 * float(np.sum(np.sqrt(np.maximum(eigenvalues, 0.0)))), float(eigenvalues.min()) if eigenvalues.size else np.inf


def _bath_free_energy(k: np.ndarray, beta: float) -> tuple[float, float]:
    """Harmonic quantum free energy ``sum_k ln(2 sinh(beta w_k / 2)) / beta`` of the complement.

    It reduces to the zero-point energy when ``beta w_k >> 1``.  Its first-order
    response to a change of the complement Hessian is the thermal average of
    that change, which is what makes the projected ``C_S`` transfer to first
    order in the environment coupling (docs/subspace-transfer-theory.md).
    Non-positive curvatures are dropped, as for the zero-point energy.
    """
    eigenvalues = np.linalg.eigvalsh(k) if k.size else np.zeros(0)
    omega = np.sqrt(eigenvalues[eigenvalues > 0])
    x = beta * omega
    # ln(2 sinh(x/2)) = x/2 + ln(1 - exp(-x)), stable for large x.
    value = float(np.sum(0.5 * omega + np.log(-np.expm1(-x)) / beta))
    return value, float(eigenvalues.min()) if eigenvalues.size else np.inf


def _environment_term(k, environment, beta):
    if environment == "free_energy":
        return _bath_free_energy(k, beta)
    return _bath_zero_point(k)


def _check_environment(environment, beta, allowed):
    if environment not in allowed:
        raise ValueError(f"environment must be one of {allowed}")
    if environment == "free_energy":
        if beta is None or not np.isfinite(float(beta)) or float(beta) <= 0:
            raise ValueError("environment='free_energy' needs a positive beta")
        return float(beta)
    return None


def project_linear_subspace(
    potential: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    reference: Any,
    masses: Any,
    basis: Any,
    axes: Sequence[Any],
    *,
    environment: str = "adiabatic",
    hessian: Callable[[np.ndarray], np.ndarray] | None = None,
    dividing_surface: float = 0.0,
    reactant: str = "asymptotic",
    reactant_reference: float | None = None,
    relax_tolerance: float = 1e-7,
    max_iterations: int = 200,
    finite_difference_step: float = 1e-3,
    rigid_body: bool = False,
    skip_above: float | None = None,
    wall_energy: float | None = None,
    beta: float | None = None,
) -> SubspaceModel:
    """Project a full Cartesian Hamiltonian onto a 1-3D mass-weighted subspace.

    ``x(s, y) = reference + (U s + W y) / sqrt(m)`` with orthonormal
    mass-weighted ``U`` (``basis``) and its complement ``W``.  ``environment``:

    ``"frozen"``
        ``V_S(s) = V(x(s, 0))``.
    ``"relaxed"``
        ``V_S(s) = min_y V(x(s, y))``.
    ``"adiabatic"`` (default)
        relaxed energy plus ``sum_k omega_k(s) / 2`` from the complement
        Hessian at the relaxed point, i.e. a vibrationally adiabatic model.
    ``"free_energy"``
        relaxed energy plus the harmonic quantum free energy
        ``sum_k ln(2 sinh(beta omega_k(s) / 2)) / beta`` of the complement at
        inverse temperature ``beta`` (required).  It equals ``"adiabatic"``
        when every ``beta omega_k >> 1`` and, unlike it, gets the thermal
        average of the coupling right at first order, so ``C_S`` transfers to
        first order also when complement modes are thermally excited.  The
        model then belongs to that temperature only.

    ``potential(x)`` and ``gradient(x)`` act on one Cartesian vector.  The
    complement Hessian is taken from ``hessian(x)`` when supplied, otherwise
    by central differences of the gradient.  Negative complement curvature is
    recorded (its zero-point term is dropped), never hidden.  ``rigid_body``
    (molecules in 3D Cartesian coordinates) removes the overall translations
    and rotations from the complement, as in a reaction-path Hamiltonian;
    without it, off-equilibrium rotations acquire spurious curvature.
    ``skip_above`` keeps the frozen energy (no relaxation, no zero-point
    term) at points whose frozen energy exceeds it -- regions a grid needs
    only as walls.  ``wall_energy`` replaces failed or non-finite potential
    evaluations (e.g. overlapping atoms outside the PES domain); the count is
    recorded.
    """
    beta = _check_environment(environment, beta, ("frozen", "relaxed", "adiabatic", "free_energy"))
    if not callable(potential) or not callable(gradient):
        raise TypeError("potential and gradient must be callable")
    from scipy.optimize import minimize

    x0 = np.asarray(reference, dtype=float)
    if x0.ndim != 1 or x0.size < 1 or not np.isfinite(x0).all():
        raise ValueError("reference must be a finite Cartesian vector")
    n = x0.size
    mass = _positive_vector(masses, n, "masses")
    sqrt_m = np.sqrt(mass)
    u = _orthonormal_basis(basis, n)
    d = u.shape[1]
    grids = tuple(_axis(a, f"axes[{i}]") for i, a in enumerate(axes))
    if len(grids) != d:
        raise ValueError("one grid axis is required per basis column")
    if rigid_body and n % 3:
        raise ValueError("rigid_body requires Cartesian coordinates of atoms in 3D")
    w = (_vibrational_complement(x0, u, sqrt_m, rigid_body) if n > d else np.zeros((n, 0)))
    shape = tuple(len(a) for a in grids)
    values = np.empty(shape)
    zpe = np.zeros(shape)
    y = np.zeros(w.shape[1])
    unstable, failures, min_curvature, max_gradient = 0, 0, np.inf, 0.0

    def cartesian(s, yy):
        return x0 + (u @ s + w @ yy) / sqrt_m

    def objective(yy, s):
        x = cartesian(s, yy)
        return float(potential(x)), w.T @ (np.asarray(gradient(x), float) / sqrt_m)

    skipped, walls = 0, 0

    def safe_frozen(s):
        try:
            value = float(potential(cartesian(s, np.zeros(w.shape[1]))))
        except Exception:
            if wall_energy is None:
                raise
            return np.nan
        return value

    for index in _snake_indices(shape):
        s = np.array([grids[a][index[a]] for a in range(d)])
        if skip_above is not None or wall_energy is not None:
            frozen = safe_frozen(s)
            if not np.isfinite(frozen):
                if wall_energy is None:
                    raise FloatingPointError("potential is not finite on the subspace grid")
                values[index] = float(wall_energy)
                walls += 1
                continue
            if skip_above is not None and frozen > skip_above:
                values[index] = frozen
                skipped += 1
                continue
        if environment != "frozen" and w.shape[1]:
            result = minimize(objective, y, args=(s,), jac=True, method="BFGS",
                              options={"gtol": relax_tolerance, "maxiter": int(max_iterations)})
            failures += int(not result.success)
            y = result.x
            max_gradient = max(max_gradient, float(np.linalg.norm(result.jac)))
        x = cartesian(s, y if environment != "frozen" else np.zeros_like(y))
        values[index] = float(potential(x))
        if environment in ("adiabatic", "free_energy") and w.shape[1]:
            local = _vibrational_complement(x, u, sqrt_m, True) if rigid_body else w
            k = _complement_hessian(x, local, sqrt_m, gradient, hessian, finite_difference_step)
            zpe[index], lowest = _environment_term(k, environment, beta)
            min_curvature = min(min_curvature, lowest)
            unstable += int(lowest < 0)
    if not np.isfinite(values).all():
        raise FloatingPointError("projected potential is not finite")
    info = {
        "kind": "linear_projection",
        "environment": environment,
        "full_dimension": n,
        "complement_dimension": int(w.shape[1]),
        "rigid_body_removed": bool(rigid_body),
        "points_skipped_above": int(skipped),
        "wall_points": int(walls),
        "basis_mass_weighted": u.tolist(),
        "relaxation_failures": int(failures),
        "max_relaxed_gradient": max_gradient,
        "unstable_complement_points": int(unstable),
        "minimum_complement_curvature": None if not np.isfinite(min_curvature) else min_curvature,
        "zero_point_range": [float(zpe.min()), float(zpe.max())],
        "beta": beta,
    }
    return SubspaceModel(grids, values + zpe, np.ones(d), dividing_surface, reactant,
                         reactant_reference, info)


def _arc_length_spline(path_mass_weighted: np.ndarray, samples: int = 4001):
    from scipy.interpolate import make_interp_spline

    chord = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path_mass_weighted, axis=0), axis=1))]
    if np.any(np.diff(chord) <= 0):
        raise ValueError("path points must be distinct and ordered")
    k = 3 if len(chord) >= 4 else 1
    spline = make_interp_spline(chord, path_mass_weighted, k=k, axis=0)
    fine = np.linspace(0.0, chord[-1], samples)
    speed = np.linalg.norm(spline.derivative()(fine), axis=1)
    arc = np.r_[0.0, np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(fine))]
    return spline, fine, arc


def path_curvature(path: Any, masses: Any = 1.0, *, samples: int = 401) -> tuple[np.ndarray, np.ndarray]:
    """Arc length and curvature of a path in mass-weighted coordinates."""
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[0] < 4 or not np.isfinite(points).all():
        raise ValueError("path must be a finite [points, coordinates] array with at least four points")
    sqrt_m = np.sqrt(_positive_vector(masses, points.shape[1], "masses"))
    spline, fine, arc = _arc_length_spline(points * sqrt_m[None, :])
    lam = np.interp(np.linspace(0.0, arc[-1], samples), arc, fine)
    first = spline.derivative()(lam)
    second = spline.derivative(2)(lam)
    speed = np.linalg.norm(first, axis=1)
    cross = np.sum(second * second, axis=1) * speed**2 - np.sum(first * second, axis=1) ** 2
    curvature = np.sqrt(np.maximum(cross, 0.0)) / np.maximum(speed, 1e-300) ** 3
    return np.linspace(0.0, arc[-1], samples), curvature



def _sinc_dvr_kinetic_1d(axis: np.ndarray) -> np.ndarray:
    """Return the unit-mass Colbert--Miller sinc-DVR kinetic matrix.

    The path coordinate and all normal coordinates are mass weighted, hence
    every normal cut has unit mass.  Keeping this small helper here avoids an
    import cycle with :mod:`subspace_quantum`, which uses the same matrix for
    the full subspace Hamiltonian.
    """
    axis = np.asarray(axis, dtype=float)
    n = axis.size
    spacing = float(axis[1] - axis[0])
    difference = np.arange(n)[:, None] - np.arange(n)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = 2.0 * np.where(difference % 2 == 0, 1.0, -1.0) / difference.astype(float) ** 2
    matrix[np.diag_indices(n)] = np.pi**2 / 3.0
    return matrix / (2.0 * spacing * spacing)


def _normal_dvr_term(
    potential: Callable[[np.ndarray], float],
    x: np.ndarray,
    direction: np.ndarray,
    sqrt_m: np.ndarray,
    base_energy: float,
    frequency: float,
    environment: str,
    beta: float | None,
    *,
    points: int,
    half_width: float | None,
    spacing: float | None,
) -> tuple[float, float]:
    """Solve one mass-weighted normal cut with a finite sinc DVR.

    ``direction`` is a unit vector in mass-weighted Cartesian coordinates and
    ``q`` in the DVR is therefore converted back to Cartesian displacement as
    ``q * direction / sqrt(m)``.  The returned term is relative to the path
    energy, so adding all independent normal cuts to ``base_energy`` preserves
    the same energy origin as the harmonic normal-cut model.  For an
    adiabatic environment this is the lowest eigenvalue; for ``free_energy``
    it is ``-beta**-1 log Tr exp(-beta H_cut)`` relative to the path energy.
    """
    frequency = float(frequency)
    if half_width is None:
        # Eight oscillator lengths place the hard walls far beyond the ground
        # state (the harmonic-limit error is below 1e-8 with the default grid).
        half_width = 8.0 / np.sqrt(frequency)
    half_width = float(half_width)
    if not np.isfinite(half_width) or half_width <= 0:
        raise ValueError("dvr_half_width must be finite and positive")
    if spacing is not None:
        spacing = float(spacing)
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("dvr_spacing must be finite and positive")
        count = int(np.ceil(2.0 * half_width / spacing)) + 1
    else:
        count = int(points)
    if count < 8:
        raise ValueError("dvr_points must be at least eight")
    # An odd number keeps q=0 on the grid and gives a symmetric hard wall.
    if count % 2 == 0:
        count += 1
    axis = np.linspace(-half_width, half_width, count)
    cuts = x[None, :] + axis[:, None] * direction[None, :] / sqrt_m[None, :]
    energies = np.asarray([float(potential(point)) for point in cuts], dtype=float)
    if not np.isfinite(energies).all():
        raise FloatingPointError("normal-cut DVR encountered a non-finite potential")
    hamiltonian = _sinc_dvr_kinetic_1d(axis) + np.diag(energies)
    spectrum = np.linalg.eigvalsh(hamiltonian)
    relative = spectrum - float(base_energy)
    if environment == "free_energy":
        # All DVR states contribute to the one-dimensional quantum partition
        # function; logsumexp avoids overflow at the low temperatures used by
        # subspace-transfer calculations.
        value = -float(logsumexp(-float(beta) * relative)) / float(beta)
    else:
        value = float(relative[0])
    return value, float(spectrum[0])


def _path_dvr_environment(
    potential: Callable[[np.ndarray], float],
    x: np.ndarray,
    complement: np.ndarray,
    sqrt_m: np.ndarray,
    k: np.ndarray,
    base_energy: float,
    environment: str,
    beta: float | None,
    *,
    points: int,
    half_width: float | None,
    spacing: float | None,
) -> tuple[float, float, int]:
    """Sum independent normal-mode DVR free energies at one path point."""
    if complement.shape[1] == 0:
        return 0.0, np.inf, 0
    eigenvalues, eigenvectors = np.linalg.eigh(k)
    # Use Hessian normal modes rather than arbitrary null-space directions.
    # Degenerate modes still need a coupled-mode convergence check when the
    # anharmonic potential is not separable.
    total = 0.0
    lowest = float(np.min(eigenvalues))
    unstable = int(np.count_nonzero(eigenvalues < 0.0))
    for value, vector in zip(eigenvalues, eigenvectors.T):
        direction = complement @ vector
        term, _ = _normal_dvr_term(
            potential, x, direction, sqrt_m, base_energy,
            # Harmonic ZPE drops non-positive modes.  A DVR cut still needs a
            # finite box for such a mode; use a conservative 1e-4 Eh
            # curvature scale for the box-size heuristic while retaining the
            # actual anharmonic cut potential in the Hamiltonian.
            np.sqrt(float(value) if value > 0.0 else 1.0e-4), environment, beta,
            points=points, half_width=half_width, spacing=spacing,
        )
        total += term
    return float(total), lowest, unstable


def project_path_subspace(
    potential: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray] | None,
    path: Any,
    masses: Any,
    axis: Any,
    *,
    environment: str = "adiabatic",
    hessian: Callable[[np.ndarray], np.ndarray] | None = None,
    dividing_surface: float | None = None,
    reactant: str = "asymptotic",
    reactant_reference: float | None = None,
    finite_difference_step: float = 1e-3,
    rigid_body: bool = False,
    beta: float | None = None,
    zero_point: str = "harmonic",
    dvr_points: int = 81,
    dvr_half_width: float | None = None,
    dvr_spacing: float | None = None,
) -> SubspaceModel:
    """One-dimensional vibrationally adiabatic model along a curved path.

    ``path`` is an ordered ``[points, n]`` Cartesian polyline (for example a
    ring-polymer mean path).  It is splined in mass-weighted space and
    re-parametrized by arc length ``xi``, so the model mass is one.
    ``V_S(xi) = V(x(xi)) + sum_k omega_k(xi) / 2`` with frequencies from the
    Hessian projected onto the normal plane (``environment='adiabatic'``),
    the harmonic quantum free energy of those modes at ``beta``
    (``'free_energy'``), or ``V(x(xi))`` alone (``'frozen'``).  With
    ``zero_point='rph'`` the quadratic reaction-path potential uses the same
    normal-plane Hessian ``N.T @ H @ N`` as ``harmonic``. Since the normal
    displacement at fixed arc length is linear, no gradient-curvature term
    is subtracted. Curvature enters kinetic and mode couplings, which this
    one-dimensional projection does not restore. Check
    :func:`curvature_width_criterion` before trusting a one-dimensional ``S``.
    ``axis`` is the arc-length grid and must lie within the path.
    ``rigid_body`` removes overall translations and rotations from the
    normal plane (molecules in 3D Cartesian coordinates).
    ``zero_point`` selects the transverse quantum term for an adiabatic or
    free-energy environment: ``"harmonic"`` is the historical normal-cut
    Hessian term, ``"rph"`` is the reaction-path projected Hessian term, and
    ``"dvr"`` solves independent one-dimensional normal cuts with a sinc DVR.
    The latter replaces ``hbar omega/2`` by the cut ground state (or by its
    finite-temperature quantum free energy when ``environment="free_energy"``).
    ``dvr_points``, ``dvr_half_width`` and ``dvr_spacing`` control the normal
    cut grids; the half-width defaults to eight oscillator lengths.
    """
    beta = _check_environment(environment, beta, ("frozen", "adiabatic", "free_energy"))
    if zero_point not in ("harmonic", "rph", "dvr"):
        raise ValueError("zero_point must be one of ('harmonic', 'rph', 'dvr')")
    try:
        dvr_points_int = int(dvr_points)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("dvr_points must be an integer >= 8") from error
    if dvr_points_int != dvr_points or dvr_points_int < 8:
        raise ValueError("dvr_points must be an integer >= 8")
    dvr_points = dvr_points_int
    if dvr_half_width is not None and (not np.isfinite(float(dvr_half_width)) or float(dvr_half_width) <= 0):
        raise ValueError("dvr_half_width must be finite and positive")
    if dvr_spacing is not None and (not np.isfinite(float(dvr_spacing)) or float(dvr_spacing) <= 0):
        raise ValueError("dvr_spacing must be finite and positive")
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[0] < 4 or not np.isfinite(points).all():
        raise ValueError("path must be a finite [points, coordinates] array with at least four points")
    n = points.shape[1]
    mass = _positive_vector(masses, n, "masses")
    sqrt_m = np.sqrt(mass)
    spline, fine, arc = _arc_length_spline(points * sqrt_m[None, :])
    grid = _axis(axis, "axis")
    if grid[0] < -1e-9 * arc[-1] or grid[-1] > arc[-1] * (1 + 1e-9):
        raise ValueError(f"axis must lie within the path arc length [0, {arc[-1]:.6g}]")
    if environment != "frozen" and gradient is None and hessian is None:
        raise ValueError("an adiabatic path model needs a gradient or a Hessian")
    lam = np.interp(grid, arc, fine)
    positions = spline(lam)
    tangents = spline.derivative()(lam)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    values = np.empty(len(grid))
    zpe = np.zeros(len(grid))
    unstable, min_curvature = 0, np.inf
    for i, (u, tangent) in enumerate(zip(positions, tangents)):
        x = u / sqrt_m
        values[i] = float(potential(x))
        if environment != "frozen" and n > 1:
            complement = _vibrational_complement(x, tangent, sqrt_m, rigid_body)
            k = _complement_hessian(x, complement, sqrt_m, gradient, hessian, finite_difference_step)
            if zero_point == "dvr":
                zpe[i], lowest, dvr_unstable = _path_dvr_environment(
                    potential, x, complement, sqrt_m, k, values[i], environment, beta,
                    points=dvr_points, half_width=dvr_half_width, spacing=dvr_spacing,
                )
                unstable += int(dvr_unstable > 0)
            else:
                # Both quadratic selectors use the Hessian of the same
                # linear normal cut. A force-curvature subtraction would
                # change that potential's actual transverse frequencies.
                zpe[i], lowest = _environment_term(k, environment, beta)
                unstable += int(lowest < 0)
            min_curvature = min(min_curvature, lowest)
    xi, curvature = path_curvature(points, mass)
    if dividing_surface is None:
        dividing_surface = float(grid[np.argmax(values + zpe)])
    info = {
        "kind": "path_projection",
        "environment": environment,
        "zero_point": zero_point,
        "zero_point_definition": "linear-normal-cuts-v2",
        "dvr_points": dvr_points if zero_point == "dvr" else None,
        "dvr_half_width": dvr_half_width if zero_point == "dvr" else None,
        "dvr_spacing": dvr_spacing if zero_point == "dvr" else None,
        "full_dimension": n,
        "path_length": float(arc[-1]),
        "max_path_curvature": float(np.max(curvature)),
        "unstable_complement_points": int(unstable),
        "minimum_complement_curvature": None if not np.isfinite(min_curvature) else min_curvature,
        "zero_point_range": [float(zpe.min()), float(zpe.max())],
        "rph_max_correction": 0.0 if zero_point == "rph" else None,
        "cut": "normal plane in mass-weighted coordinates",
        "rigid_body_removed": bool(rigid_body),
        "beta": beta,
    }
    return SubspaceModel((grid,), values + zpe, np.ones(1), dividing_surface, reactant,
                         reactant_reference, info)


def relaxed_path(
    potential: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    start: Any,
    direction: Any,
    masses: Any,
    *,
    step: float,
    max_steps: int = 400,
    max_angle_degrees: float = 20.0,
    min_step: float | None = None,
    asymptote_tolerance: float = 1e-6,
    rigid_body: bool = True,
) -> np.ndarray:
    """Follow a valley from ``start`` along ``direction`` (mass-weighted).

    Downhill, each new point minimizes ``V`` on the sphere of radius ``step``
    around the current point, restricted to the forward hemisphere
    (Gonzalez--Schlegel steepest-descent construction; directions
    ``(t + W y)/|t + W y|`` with ``W`` the complement of the tangent ``t`` and,
    for molecules, of the rigid-body modes).  Where the forward valley still
    rises (a core path ending before the barrier top) the point is instead
    the minimum on the hyperplane ``step`` ahead.  The step is halved when
    the tangent turns by more than ``max_angle_degrees`` and grows back
    afterwards.  The follower stops when the energy changed by less than
    ``asymptote_tolerance`` over the last ten points or after ``max_steps``.
    Returns the Cartesian points ``[k, n]`` including ``start``.
    """
    from scipy.optimize import minimize

    x = np.asarray(start, dtype=float).ravel().copy()
    n = x.size
    sqrt_m = np.sqrt(_positive_vector(masses, n, "masses"))
    t = np.asarray(direction, dtype=float).ravel()
    if t.shape != (n,) or not np.isfinite(t).all() or np.linalg.norm(t) == 0:
        raise ValueError("direction must be a finite nonzero mass-weighted vector")
    if rigid_body and n % 3:
        raise ValueError("rigid_body requires Cartesian coordinates of atoms in 3D")
    t = t / np.linalg.norm(t)
    smallest = step / 64.0 if min_step is None else float(min_step)
    energy = float(potential(x))
    points, energies = [x.copy()], [energy]
    current = float(step)
    cos_limit = np.cos(np.radians(max_angle_degrees))

    def sphere(y, w, h):
        v = t + w @ y
        norm = np.linalg.norm(v)
        d = v / norm
        trial = x + h * d / sqrt_m
        g = np.asarray(gradient(trial), float) / sqrt_m
        # d(d)/dy = (W - d d^T W)/|v|
        return float(potential(trial)), h * (w.T @ g - (d @ g) * (w.T @ d)) / norm

    def plane(y, w, h):
        trial = x + (h * t + w @ y) / sqrt_m
        return float(potential(trial)), w.T @ (np.asarray(gradient(trial), float) / sqrt_m)

    while len(points) <= max_steps:
        w = _vibrational_complement(x, t, sqrt_m, rigid_body)
        result = minimize(sphere, np.zeros(w.shape[1]), args=(w, current), jac=True, method="BFGS",
                          options={"gtol": 1e-10, "maxiter": 400})
        v = t + w @ result.x
        trial = x + current * (v / np.linalg.norm(v)) / sqrt_m
        trial_energy = float(result.fun)
        if trial_energy > energy:
            result = minimize(plane, np.zeros(w.shape[1]), args=(w, current), jac=True, method="BFGS",
                              options={"gtol": 1e-10, "maxiter": 400})
            trial = x + (current * t + w @ result.x) / sqrt_m
            trial_energy = float(result.fun)
        move = (trial - x) * sqrt_m
        new_t = move / np.linalg.norm(move)
        if new_t @ t < cos_limit and current > smallest:
            current *= 0.5
            continue
        x, t, energy = trial, new_t, trial_energy
        points.append(x.copy())
        energies.append(energy)
        current = min(step, 1.5 * current)
        if len(energies) > 10 and np.ptp(energies[-10:]) < asymptote_tolerance:
            break
    return np.asarray(points)


def assemble_reaction_path(
    core: Any,
    potential: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    masses: Any,
    *,
    step: float,
    max_steps: int = 400,
    rigid_body: bool = True,
    max_angle_degrees: float = 20.0,
    asymptote_tolerance: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """Extend a sampled core path (e.g. the ring-polymer mean path) to both asymptotes.

    ``core`` is ordered from reactants to products, ``[points, n]`` Cartesian.
    The ends are continued with :func:`relaxed_path` along the end tangents.
    The returned ``info`` records which points are sampled and which are the
    relaxed (classical valley) continuation.
    """
    points = np.asarray(core, dtype=float).reshape(len(core), -1)
    if len(points) < 2:
        raise ValueError("a core path needs at least two points")
    sqrt_m = np.sqrt(_positive_vector(masses, points.shape[1], "masses"))
    mw = points * sqrt_m
    options = dict(step=step, max_steps=max_steps, rigid_body=rigid_body,
                   max_angle_degrees=max_angle_degrees, asymptote_tolerance=asymptote_tolerance)
    back = relaxed_path(potential, gradient, points[0], mw[0] - mw[1], masses, **options)
    ahead = relaxed_path(potential, gradient, points[-1], mw[-1] - mw[-2], masses, **options)
    path = np.concatenate([back[:0:-1], points, ahead[1:]])
    info = {
        "core_points": int(len(points)),
        "reactant_continuation_points": int(len(back) - 1),
        "product_continuation_points": int(len(ahead) - 1),
        "core_slice": [int(len(back) - 1), int(len(back) - 1 + len(points))],
        "continuation": "steepest-descent sphere (uphill: hyperplane) path following in mass-weighted coordinates",
        "step": float(step),
    }
    return path, info


def ring_polymer_mean_path(reaction_values: Any, centroid_means: Any, *, bandwidth: float) -> tuple[np.ndarray, np.ndarray]:
    """Smooth constrained-centroid bath means ``xbar(q)`` into a path ``f(q)``.

    ``reaction_values`` are the constrained reaction-coordinate centroids and
    ``centroid_means`` the sampled bath-centroid means ``[points, bath]``.
    Local quadratic regression with a Gaussian kernel of width ``bandwidth``
    returns the smoothed path and its local slope ``f'(q)``; the slope enters
    the adiabatic zero-point energy, so it is taken from the local fit rather
    than by differencing noisy means.
    """
    q = np.asarray(reaction_values, dtype=float)
    means = np.asarray(centroid_means, dtype=float)
    if means.ndim == 1:
        means = means[:, None]
    if q.ndim != 1 or len(q) < 4 or means.shape[0] != len(q) or not np.isfinite(q).all() or not np.isfinite(means).all():
        raise ValueError("reaction_values [points] and centroid_means [points, bath] must be finite")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    smooth = np.empty_like(means)
    slope = np.empty_like(means)
    for i, center in enumerate(q):
        delta = q - center
        weight = np.exp(-0.5 * (delta / bandwidth) ** 2)
        design = np.column_stack([np.ones_like(delta), delta, delta * delta]) * np.sqrt(weight)[:, None]
        coefficients = np.linalg.lstsq(design, means * np.sqrt(weight)[:, None], rcond=None)[0]
        smooth[i], slope[i] = coefficients[0], coefficients[1]
    return smooth, slope


def curvature_width_criterion(curvature: Any, width: Any, *, threshold: float = 0.5) -> dict:
    """Curvature times transverse quantum width, the one-dimensional-path test.

    ``curvature`` is the mass-weighted path curvature and ``width`` the
    transverse quantum width (for example ``sqrt(2)`` times the bead RMS
    spread) at the same points.  ``kappa*rho << 1`` lets a one-dimensional path
    carry the coupling; ``kappa*rho ~ 1`` means the vibration straddles the
    bend and the transverse coordinate belongs in ``S``.  The default 0.5 was
    fixed from two benchmark systems and is provisional.
    """
    kappa = np.asarray(curvature, dtype=float)
    rho = np.asarray(width, dtype=float)
    if kappa.shape != rho.shape or not np.isfinite(kappa).all() or not np.isfinite(rho).all() or np.any(rho < 0):
        raise ValueError("curvature and width must be finite arrays of the same shape")
    product_value = np.abs(kappa) * rho
    peak = float(np.max(product_value)) if product_value.size else 0.0
    return {
        "curvature_width": product_value.tolist(),
        "maximum": peak,
        "threshold": float(threshold),
        "include_transverse_mode": bool(peak > threshold),
        "provisional_threshold": True,
    }


__all__ = [
    "MAX_DIMENSION", "GridPotential", "SubspaceModel", "model_from_function",
    "project_linear_subspace", "project_path_subspace", "path_curvature", "relaxed_path",
    "assemble_reaction_path",
    "ring_polymer_mean_path", "curvature_width_criterion",
]
