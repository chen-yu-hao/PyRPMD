"""Two-dimensional reaction-path Hamiltonian subspace on the minimum-energy path.

The 1D model follows the minimum-energy path (MEP) ``s`` and treats every
transverse mode adiabatically, so it cannot cut corners: tunnelling on the
inner side of a curved path is missing.  The reaction-path Hamiltonian adds
the transverse coordinate ``q`` along which the path curves.  With the
mass-weighted path ``w(s)``, unit tangent ``t(s)`` and a normal ``n(s)``,

    x(s, q) = w(s) + q n(s),    T = p_s^2 / (2 (1 - kappa(s) q)^2) + p_q^2 / 2,

where ``kappa = t' . n``.  The normal is transported along the path without
rotating about it (a Bishop frame, ``n' = -kappa t``), so the ribbon
``x(s, q)`` has exactly this metric -- no torsion term is neglected.  The
curvature of the path *across* ``n`` couples ``s`` to the remaining modes,
which stay adiabatic as in the 1D model; ``n`` is chosen to carry as much of
the curvature near the barrier as possible (``captured_curvature``).

A ribbon with metric ``diag((1 - kappa q)^2, 1)`` is isometric to a strip of
the plane: unroll it along a plane curve ``c(s)`` with the same signed
curvature, ``X(s, q) = c(s) + q N(s)``.  In ``X`` the Hamiltonian is the flat
``-(d^2/dX^2 + d^2/dY^2)/2 + V_S(X)``, which the existing tensor-grid quantum
and ring-polymer solvers take unchanged -- including the region inside the
bend that a corner-cutting tunnelling path crosses.  For a two-dimensional
system the development is congruent to the system itself, so the model is
exact there.  Each grid point is mapped back by its nearest point on
``c(s)``; beyond the centre of curvature that choice picks the nearer branch.

``V_S(X) = V(x(s, q)) + E_bath(s)`` with ``E_bath`` the zero-point energy
(``adiabatic``) or harmonic quantum free energy (``free_energy``) of the
Hessian projected onto the complement of ``t``, ``n`` and the rigid-body
modes on the path, or nothing (``frozen``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .subspace_paths import mass_weighted_hessian, rigid_modes

_HARTREE_TO_WAVENUMBER = 219474.6313632


@dataclass
class PathDevelopment:
    """A path in mass-weighted space with its Bishop normal, unrolled into a plane."""

    s: np.ndarray            # arc length of the samples
    w: np.ndarray            # mass-weighted points [k, n]
    t: np.ndarray            # unit tangents [k, n]
    n: np.ndarray            # transported unit normals [k, n]
    kappa: np.ndarray        # signed curvature along n
    curvature: np.ndarray    # magnitude of the full curvature vector
    plane: np.ndarray        # developed curve c(s) [k, 2]
    tangent2: np.ndarray     # developed unit tangent [k, 2]
    normal2: np.ndarray      # developed unit normal [k, 2]
    sqrt_m: np.ndarray
    captured_curvature: float
    record: dict = field(default_factory=dict)
    _tree: Any = None

    def orient(self, origin_arc: float, axis: Any) -> None:
        """Move ``c(origin_arc)`` to the origin and turn ``axis`` (a developed direction) onto +X."""
        axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        rotation = np.array([[axis[0], axis[1]], [-axis[1], axis[0]]])
        origin = np.array([np.interp(origin_arc, self.s, self.plane[:, i]) for i in range(2)])
        self.plane = (self.plane - origin) @ rotation.T
        self.tangent2 = self.tangent2 @ rotation.T
        self.normal2 = self.normal2 @ rotation.T
        self._tree = None

    def point(self, s: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Developed point, tangent and normal at arc lengths ``s``."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        c = np.column_stack([np.interp(s, self.s, self.plane[:, i]) for i in range(2)])
        theta = np.interp(s, self.s, np.unwrap(np.arctan2(self.tangent2[:, 1], self.tangent2[:, 0])))
        tangent = np.column_stack([np.cos(theta), np.sin(theta)])
        return c, tangent, np.column_stack([-tangent[:, 1], tangent[:, 0]])

    def locate(self, points: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(s, q)`` of developed points: nearest point on ``c(s)`` and the signed offset along ``N``."""
        from scipy.spatial import cKDTree

        points = np.atleast_2d(np.asarray(points, dtype=float))
        if self._tree is None:
            self._tree = cKDTree(self.plane)
        _, index = self._tree.query(points)
        s = self.s[index].copy()
        for _ in range(3):
            c, tangent, _ = self.point(s)
            s = np.clip(s + np.sum((points - c) * tangent, axis=1), self.s[0], self.s[-1])
        c, _, normal = self.point(s)
        return s, np.sum((points - c) * normal, axis=1)

    def configuration(self, s: Any, q: Any) -> np.ndarray:
        """Cartesian configurations ``x(s, q) = (w(s) + q n(s)) / sqrt(m)``, ``[m, n]``."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        q = np.broadcast_to(np.asarray(q, dtype=float), s.shape)
        w = np.column_stack([np.interp(s, self.s, self.w[:, i]) for i in range(self.w.shape[1])])
        normal = np.column_stack([np.interp(s, self.s, self.n[:, i]) for i in range(self.n.shape[1])])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        return (w + q[:, None] * normal) / self.sqrt_m[None, :]

    def frame(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        """Mass-weighted tangent and normal at ``s``."""
        t = np.array([np.interp(s, self.s, self.t[:, i]) for i in range(self.t.shape[1])])
        n = np.array([np.interp(s, self.s, self.n[:, i]) for i in range(self.n.shape[1])])
        t /= np.linalg.norm(t)
        n -= (n @ t) * t
        return t, n / np.linalg.norm(n)


def develop_path(path: Any, masses: Any, *, rigid_body: bool = True, weights: Any = None,
                 spacing: float | None = None, extend: float = 0.0, fallback: Any = None,
                 seeds: int = 5) -> PathDevelopment:
    """Spline a Cartesian path, transport a normal along it and unroll the ribbon.

    ``masses`` are per coordinate.  ``weights`` (one per path point) weight
    the curvature that the normal should capture -- the pipeline uses
    ``exp(-beta (V_top - V))`` so that the region near the barrier decides.
    The normal is the Bishop transport of a seed; the seeds tried are the
    curvature directions at the ``seeds`` points of largest weighted
    curvature, and the one that captures the largest weighted ``kappa^2`` is
    kept.  A path without curvature takes ``fallback`` (or any normal).
    ``extend`` adds straight continuations of that length at both ends.
    """
    from scipy.interpolate import CubicSpline

    path = np.asarray(path, dtype=float).reshape(len(path), -1)
    masses = np.asarray(masses, dtype=float).ravel()
    sqrt_m = np.sqrt(masses)
    mw = path * sqrt_m
    steps = np.linalg.norm(np.diff(mw, axis=0), axis=1)
    keep = np.r_[True, steps > 1e-12 * max(steps.max(), 1.0)]
    mw, path = mw[keep], path[keep]
    chord = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(mw, axis=0), axis=1))]
    if len(chord) < 4:
        raise ValueError("a path development needs at least four distinct points")
    ds = float(spacing) if spacing else 0.25 * float(np.median(np.diff(chord)))
    spline = CubicSpline(chord, mw, axis=0)
    s = np.linspace(0.0, chord[-1], max(int(np.ceil(chord[-1] / ds)) + 1, 8))
    w = spline(s)
    t = spline(s, 1)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    weight = np.ones(len(s)) if weights is None else np.interp(s, chord, np.asarray(weights, float)[keep])
    if extend > 0:
        count = int(np.ceil(extend / (s[1] - s[0])))
        h = s[1] - s[0]
        before = w[0] - np.outer(np.arange(count, 0, -1) * h, t[0])
        after = w[-1] + np.outer(np.arange(1, count + 1) * h, t[-1])
        w = np.vstack([before, w, after])
        t = np.vstack([np.repeat(t[:1], count, 0), t, np.repeat(t[-1:], count, 0)])
        s = np.r_[s[0] - h * np.arange(count, 0, -1), s, s[-1] + h * np.arange(1, count + 1)]
        weight = np.r_[np.zeros(count), weight, np.zeros(count)]
    # Normal projector at each sample: without the tangent and the rigid-body modes.
    removed = []
    for k in range(len(s)):
        rigid = rigid_modes(w[k] / sqrt_m, masses, rigid_body)
        block = np.column_stack([t[k], rigid]) if rigid.shape[1] else t[k][:, None]
        q, _ = np.linalg.qr(block)
        removed.append(q)

    def normal_part(k, v):
        return v - removed[k] @ (removed[k].T @ v)

    derivative = np.gradient(t, s, axis=0)
    curvature_vectors = np.array([normal_part(k, derivative[k]) for k in range(len(s))])
    curvature = np.linalg.norm(curvature_vectors, axis=1)
    total = float(np.sum(weight * curvature**2))

    def transport(seed_index, seed):
        n = np.zeros_like(w)
        n[seed_index] = seed / np.linalg.norm(seed)
        for order in (range(seed_index + 1, len(s)), range(seed_index - 1, -1, -1)):
            previous = seed_index
            for k in order:
                v = normal_part(k, n[previous])
                norm = np.linalg.norm(v)
                if norm < 1e-8:
                    raise ValueError("the path turns by 90 degrees within one development step; refine it")
                n[k] = v / norm
                previous = k
        return n

    ranked = [int(k) for k in np.argsort(-weight * curvature**2)[:max(1, int(seeds))]
              if curvature[k] > 1e-12]
    candidates = [(k, curvature_vectors[k]) for k in ranked]
    if not candidates:
        origin = int(np.argmax(weight)) if weight.any() else len(s) // 2
        seed = None if fallback is None else normal_part(origin, np.asarray(fallback, dtype=float).ravel())
        if seed is None or np.linalg.norm(seed) < 1e-12:
            basis = np.linalg.svd(np.eye(len(t[origin])) - removed[origin] @ removed[origin].T)[0]
            seed = basis[:, 0]
        candidates = [(origin, seed)]
    best = None
    for k, seed in candidates:
        n = transport(k, seed)
        kappa = np.einsum("ij,ij->i", curvature_vectors, n)
        captured = float(np.sum(weight * kappa**2)) / total if total > 0 else 1.0
        if best is None or captured > best[0]:
            best = (captured, n, kappa, k)
    captured, n, kappa, seed_index = best
    theta = np.r_[0.0, np.cumsum(0.5 * (kappa[1:] + kappa[:-1]) * np.diff(s))]
    tangent2 = np.column_stack([np.cos(theta), np.sin(theta)])
    plane = np.vstack([np.zeros(2), np.cumsum(0.5 * (tangent2[1:] + tangent2[:-1]) * np.diff(s)[:, None], axis=0)])
    normal2 = np.column_stack([-tangent2[:, 1], tangent2[:, 0]])
    record = {"samples": int(len(s)), "spacing": float(s[1] - s[0]), "extension": float(extend),
              "seed_arc": float(s[seed_index]), "captured_curvature": captured,
              "max_curvature": float(curvature.max()), "max_signed_curvature": float(np.max(np.abs(kappa))),
              "turn_angle_degrees": float(np.degrees(theta[-1] - theta[0]))}
    return PathDevelopment(s, w, t, n, kappa, curvature, plane, tangent2, normal2, sqrt_m, captured, record)


def rph_bath(gradient: Callable[[np.ndarray], np.ndarray], development: PathDevelopment, arcs: Sequence[float],
             masses: Any, *, rigid_body: bool = True, environment: str = "adiabatic", beta: float | None = None,
             displacement: float = 5e-3, soft_mode_frequency: float = 2.3e-4) -> dict:
    """Bath term on the path: modes orthogonal to ``t``, ``n`` and the rigid-body modes.

    Returns ``arc``, ``energy`` (zero-point or harmonic quantum free energy),
    ``normal_curvature`` (``n.H.n``) and the stability record the validity
    gate reads: curvatures below ``-soft_mode_frequency**2`` are unstable,
    those of smaller magnitude soft (no contribution).
    """
    from scipy.linalg import null_space

    if environment not in ("frozen", "adiabatic", "free_energy"):
        raise ValueError("environment must be frozen, adiabatic or free_energy")
    if environment == "free_energy" and beta is None:
        raise ValueError("a free_energy bath needs beta")
    masses = np.asarray(masses, dtype=float)
    energy, normal_curvature, unstable = [], [], []
    minimum, soft = np.inf, 0
    for arc in arcs:
        x = development.configuration(arc, 0.0)[0]
        hessian = mass_weighted_hessian(gradient, x, masses, displacement=displacement)
        t, n = development.frame(arc)
        normal_curvature.append(float(n @ hessian @ n))
        if environment == "frozen":
            energy.append(0.0)
            continue
        rigid = rigid_modes(x, masses, rigid_body)
        complement = null_space(np.column_stack([t, n, rigid]).T)
        values = np.linalg.eigvalsh(complement.T @ hessian @ complement) if complement.shape[1] else np.zeros(0)
        if values.size:
            minimum = min(minimum, float(values[0]))
        threshold = soft_mode_frequency ** 2
        bad = values[values < -threshold]
        soft += int(np.sum(np.abs(values) <= threshold))
        for value in bad:
            unstable.append({"arc_length": float(arc), "curvature": float(value),
                             "imaginary_frequency_cm": float(np.sqrt(-value) * _HARTREE_TO_WAVENUMBER)})
        omega = np.sqrt(values[values > threshold])
        if environment == "adiabatic":
            energy.append(float(0.5 * np.sum(omega)))
        else:
            energy.append(float(np.sum(0.5 * omega + np.log1p(-np.exp(-beta * omega)) / beta)))
    return {"arc": np.asarray(arcs, dtype=float), "energy": np.asarray(energy),
            "normal_curvature": np.asarray(normal_curvature),
            "bath_stable": not unstable, "significant_unstable": unstable,
            "unstable_characters": ["unclassified"] if unstable else None,
            "minimum_complement_curvature": None if not np.isfinite(minimum) else minimum,
            "soft_complement_points": soft}


def project_rph_values(potential: Callable[[np.ndarray], float], development: PathDevelopment, axes: Sequence[Any],
                       bath: Mapping[str, Any], *, arc_range: tuple[float, float], half_width: float,
                       wall_energy: float) -> np.ndarray:
    """``V_S`` on the developed tensor grid ``axes`` (``indexing='ij'``).

    Points farther than ``half_width`` from the path are walls
    (``wall_energy``); so is any value above it, and a failed or non-finite
    potential.  Along the path the channel is never closed: beyond
    ``arc_range`` (the box) the potential continues on the straight extension
    of the development.  A wall there would send ring polymers that left
    through a channel end back across the dividing surface, which the
    quantum rate, absorbing them first, never sees.  ``arc_range`` only
    limits where the bath term is interpolated (it is held constant beyond).
    """
    X, Y = np.meshgrid(*[np.asarray(a, dtype=float) for a in axes], indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel()])
    s, q = development.locate(points)
    values = np.full(len(points), float(wall_energy))
    inside = np.abs(q) <= half_width
    configurations = development.configuration(s[inside], q[inside])
    extra = np.interp(s[inside], bath["arc"], bath["energy"])
    computed = np.empty(len(configurations))
    for i, x in enumerate(configurations):
        try:
            computed[i] = float(potential(x))
        except (ArithmeticError, ValueError, RuntimeError):
            computed[i] = np.inf
    computed = np.where(np.isfinite(computed), computed + extra, np.inf)
    values[inside] = np.minimum(computed, float(wall_energy))
    return values.reshape(X.shape)
