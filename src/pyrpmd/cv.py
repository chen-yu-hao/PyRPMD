"""Differentiable collective variables with an explicit ring estimator.

The RPMDrate coordinate and its flux metric follow the MIT-licensed
RPMDrate _surface.f90/_main.f90 (Allen, Green, Suleimanov, 2012) and Caracal
umbrella.f90 (Steffen, 2023). See THIRD_PARTY_NOTICES.md for attribution.
Every low-level length and derivative is in atomic units.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import SimpleNamespace
import numpy as np


@dataclass(frozen=True)
class CVResult:
    value: np.ndarray
    jacobian: np.ndarray


def _coordinates(q):
    q = np.asarray(getattr(q, "q", q), dtype=np.float64)
    if q.ndim != 3 or q.shape[-1] != 3 or not np.isfinite(q).all():
        raise ValueError("CV coordinates must be finite [P,N,3] arrays")
    return q


def _indices(atoms):
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in atoms):
        raise ValueError("atom indices must be integers")
    return tuple(int(i) for i in atoms)


def _distance(q, a, b, hessian=False):
    if a == b or min(a, b) < 0 or max(a, b) >= len(q):
        raise ValueError("distance requires two distinct in-range atom indices")
    v = q[a] - q[b]
    r = float(np.linalg.norm(v))
    if r < 1e-12:
        raise ValueError("CV derivative is undefined for coincident atoms")
    u = v / r
    g = np.zeros_like(q)
    g[a], g[b] = u, -u
    if not hessian:
        return r, g
    h = np.zeros((len(q), 3, len(q), 3))
    block = (np.eye(3) - np.outer(u, u)) / r
    for i, si in ((a, 1), (b, -1)):
        for j, sj in ((a, 1), (b, -1)):
            h[i, :, j, :] = si * sj * block
    return r, g, h


class CollectiveVariable:
    dimension = 1
    units = ("dimensionless",)
    periods = (None,)

    def __init__(self, estimator="centroid", name=None):
        if estimator not in ("centroid", "bead_average"):
            raise ValueError("geometric CV estimator must be centroid or bead_average")
        self.estimator = estimator
        self.name = name or type(self).__name__

    def value_and_jacobian(self, q):
        q = _coordinates(q)
        p = len(q)
        if self.estimator == "centroid":
            value, grad = self.centroid_value_gradient(q.mean(axis=0))
            jac = np.broadcast_to(grad[None, None] / p, (1,) + q.shape).copy()
        else:
            values, grads = zip(*(self.centroid_value_gradient(x) for x in q))
            value, jac = float(np.mean(values)), np.asarray(grads)[None] / p
        return CVResult(np.atleast_1d(value), jac)

    def value(self, q):
        return self.value_and_jacobian(q).value


class Distance(CollectiveVariable):
    units = ("bohr",)

    def __init__(self, atoms, **kwargs):
        super().__init__(**kwargs)
        if len(atoms) != 2:
            raise ValueError("distance atoms must have length 2")
        self.atoms = _indices(atoms)

    def centroid_value_gradient(self, q):
        return _distance(q, *self.atoms)

    def centroid_value_gradient_hessian(self, q):
        return _distance(q, *self.atoms, hessian=True)


class BondDifference(CollectiveVariable):
    units = ("bohr",)

    def __init__(self, atoms=None, bonds=None, **kwargs):
        super().__init__(**kwargs)
        if bonds is None:
            if atoms is None or len(atoms) != 3:
                raise ValueError("bond_difference needs three atoms or two bonds")
            bonds = [atoms[:2], atoms[1:]]
        if len(bonds) != 2 or any(len(b) != 2 for b in bonds):
            raise ValueError("bond_difference bonds must be [[i,j],[k,l]]")
        self.bonds = tuple(_indices(b) for b in bonds)

    def centroid_value_gradient(self, q):
        a, ga = _distance(q, *self.bonds[0])
        b, gb = _distance(q, *self.bonds[1])
        return a - b, ga - gb


class Cartesian(CollectiveVariable):
    """A Cartesian component, useful for model systems and restrained sampling."""

    units = ("bohr",)
    constant_gradient = True

    def __init__(self, atom, axis=0, **kwargs):
        super().__init__(**kwargs)
        self.atom = _indices([atom])[0]
        self.axis = {"x": 0, "y": 1, "z": 2}.get(axis, axis)
        if (
            isinstance(self.axis, bool)
            or not isinstance(self.axis, (int, np.integer))
            or self.axis not in (0, 1, 2)
        ):
            raise ValueError("Cartesian axis must be x, y, z, 0, 1 or 2")

    def centroid_value_gradient(self, q):
        if self.atom < 0 or self.atom >= len(q):
            raise ValueError("Cartesian atom index out of range")
        g = np.zeros_like(q)
        g[self.atom, self.axis] = 1.0
        return float(q[self.atom, self.axis]), g

    def centroid_value_gradient_hessian(self, q):
        value, gradient = self.centroid_value_gradient(q)
        return value, gradient, np.zeros((len(q), 3, len(q), 3))


class Angle(CollectiveVariable):
    units = ("radian",)

    def __init__(self, atoms, **kwargs):
        super().__init__(**kwargs)
        if len(atoms) != 3 or len(set(atoms)) != 3:
            raise ValueError("angle needs three distinct atoms")
        self.atoms = _indices(atoms)

    def centroid_value_gradient(self, q):
        a, b, c = self.atoms
        if min(a, b, c) < 0 or max(a, b, c) >= len(q):
            raise ValueError("angle atom index out of range")
        x, y = q[a] - q[b], q[c] - q[b]
        rx, ry = np.linalg.norm(x), np.linalg.norm(y)
        if min(rx, ry) < 1e-12:
            raise ValueError("angle derivative is undefined for coincident atoms")
        u, v = x / rx, y / ry
        cosine = np.clip(np.dot(u, v), -1.0, 1.0)
        sine = np.sqrt(max(0.0, 1 - cosine * cosine))
        if sine < 1e-10:
            raise ValueError("angle derivative is undefined at a linear geometry")
        g = np.zeros_like(q)
        g[a] = (cosine * u - v) / (rx * sine)
        g[c] = (cosine * v - u) / (ry * sine)
        g[b] = -g[a] - g[c]
        return float(np.arccos(cosine)), g


class COMDistance(CollectiveVariable):
    units = ("bohr",)

    def __init__(self, fragments, masses, **kwargs):
        super().__init__(**kwargs)
        if len(fragments) != 2 or any(not len(f) for f in fragments):
            raise ValueError("COM distance requires two nonempty fragments")
        self.fragments = tuple(_indices(f) for f in fragments)
        flat = [a for f in self.fragments for a in f]
        if len(set(flat)) != len(flat) or min(flat) < 0 or max(flat) >= len(masses):
            raise ValueError("COM fragments must contain distinct in-range atoms")
        self.masses = np.asarray(masses, dtype=float)
        if (
            self.masses.ndim != 1
            or not np.isfinite(self.masses).all()
            or np.any(self.masses <= 0)
        ):
            raise ValueError("COM masses must be finite and positive")
        self.weights = np.zeros(len(masses))
        for sign, f in zip((1, -1), self.fragments):
            f = np.asarray(f)
            self.weights[f] = sign * self.masses[f] / self.masses[f].sum()

    def centroid_value_gradient_hessian(self, q):
        v = np.einsum("a,ac->c", self.weights, q)
        r = float(np.linalg.norm(v))
        if r < 1e-12:
            raise ValueError("COM distance derivative undefined at zero separation")
        u = v / r
        g = self.weights[:, None] * u
        h = np.einsum(
            "a,b,cd->acbd", self.weights, self.weights, (np.eye(3) - np.outer(u, u)) / r
        )
        return r, g, h

    def centroid_value_gradient(self, q):
        r, g, _ = self.centroid_value_gradient_hessian(q)
        return r, g


class RPMDrateCoordinate(CollectiveVariable):
    """xi=s0/(s0-s1), including atom-permuted equivalent transition states.

    Match RPMDrate _surface.f90 exactly: s1=min_m max_n values[n,m], where n
    labels equivalent TSs and m labels paired bond changes. Gradient/Hessian
    use the selected branch. Branch ties are piecewise nondifferentiable;
    finite-difference derivative tests must avoid these seams.
    """

    units = ("dimensionless",)

    def __init__(
        self,
        fragments,
        masses,
        reference,
        forming_bonds,
        breaking_bonds,
        rinf,
        equivalent_paths=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.estimator != "centroid":
            raise ValueError("RPMDrate rate coordinate requires centroid estimator")
        self.com = COMDistance(fragments, masses)
        self.masses = np.asarray(masses, float)
        self.reference = np.asarray(reference, float)
        self.forming = tuple(_indices(b) for b in forming_bonds)
        self.breaking = tuple(_indices(b) for b in breaking_bonds)
        if not self.forming or len(self.forming) != len(self.breaking):
            raise ValueError(
                "RPMDrate coordinate needs equally many nonempty forming/breaking bonds"
            )
        self.rinf = float(rinf)
        if not np.isfinite(self.rinf) or self.rinf <= 0:
            raise ValueError("rinf must be positive")
        self.reference_difference = [
            _distance(self.reference, *f)[0] - _distance(self.reference, *b)[0]
            for f, b in zip(self.forming, self.breaking)
        ]
        identity = tuple(range(len(self.masses)))
        self.permutations = [identity]
        partition = {frozenset(f) for f in self.com.fragments}
        for entry in equivalent_paths or ():
            if isinstance(entry, dict):
                if set(entry) != {"permutation"}:
                    raise ValueError(
                        "equivalent path tables accept only an atom permutation"
                    )
                entry = entry["permutation"]
            permutation = _indices(entry)
            if sorted(permutation) != list(identity):
                raise ValueError(
                    "equivalent path must be a complete 0-based atom permutation"
                )
            if permutation in self.permutations:
                raise ValueError(
                    "duplicate equivalent atom permutation (identity is already included)"
                )
            if not np.allclose(
                self.masses, self.masses[list(permutation)], rtol=1e-10, atol=1e-8
            ):
                raise ValueError(
                    "equivalent atom permutation must preserve isotope masses"
                )
            moved = {frozenset(permutation[i] for i in f) for f in self.com.fragments}
            if moved != partition:
                raise ValueError(
                    "equivalent atom permutation must preserve the reactant partition"
                )
            self.permutations.append(permutation)
        self.paths = [
            (
                tuple(tuple(p[i] for i in b) for b in self.forming),
                tuple(tuple(p[i] for i in b) for b in self.breaking),
            )
            for p in self.permutations
        ]
        # The entire reference geometry is permuted together with its bond
        # labels; every mapped reference bond length therefore stays the same.
        # Evaluating mapped bonds at the unpermuted TS would be incorrect.

    def surfaces(self, q):
        r, gr, hr = self.com.centroid_value_gradient_hessian(q)
        s0, g0, h0 = self.rinf - r, -gr, -hr
        candidates = []
        for forming, breaking in self.paths:
            row = []
            for ref, f, b in zip(self.reference_difference, forming, breaking):
                rf, gf, hf = _distance(q, *f, hessian=True)
                rb, gb, hb = _distance(q, *b, hessian=True)
                row.append((ref - rf + rb, -gf + gb, -hf + hb))
            candidates.append(row)
        values = np.array([[c[0] for c in row] for row in candidates])
        bond = int(np.argmin(np.max(values, axis=0)))
        transition_state = int(np.argmax(values[:, bond]))
        s1, g1, h1 = candidates[transition_state][bond]
        return s0, g0, h0, s1, g1, h1

    def centroid_value_gradient_hessian(self, q):
        a, ga, ha, s1, g1, h1 = self.surfaces(q)
        b, gb, hb = a - s1, ga - g1, ha - h1
        if abs(b) < 1e-10:
            raise ValueError("RPMDrate xi is singular where s0=s1")
        h = (
            ha / b
            - (
                np.einsum("ac,bd->acbd", ga, gb)
                + np.einsum("ac,bd->acbd", gb, ga)
                + a * hb
            )
            / b**2
        )
        h += 2 * a * np.einsum("ac,bd->acbd", gb, gb) / b**3
        return a / b, ga / b - a * gb / b**2, h

    def centroid_value_gradient(self, q):
        s, g, _ = self.centroid_value_gradient_hessian(q)
        return s, g

    def dividing_surface(self, xi):
        return RateSurface(self, float(xi))


class RateSurface(CollectiveVariable):
    units = ("bohr",)

    def __init__(self, coordinate, xi):
        super().__init__(estimator="centroid", name="rate_dividing_surface")
        self.coordinate = coordinate
        self.xi = float(xi)

    def centroid_value_gradient_hessian(self, q):
        s0, g0, h0, s1, g1, h1 = self.coordinate.surfaces(q)
        a = self.xi
        return (1 - a) * s0 + a * s1, (1 - a) * g0 + a * g1, (1 - a) * h0 + a * h1

    def centroid_value_gradient(self, q):
        s, g, _ = self.centroid_value_gradient_hessian(q)
        return s, g


class CVStack:
    def __init__(self, cvs):
        if not cvs:
            raise ValueError("at least one CV is required")
        self.cvs = tuple(cvs)
        self.dimension = sum(c.dimension for c in cvs)
        self.units = tuple(u for c in cvs for u in c.units)
        self.periods = tuple(p for c in cvs for p in c.periods)

    def value_and_jacobian(self, q):
        rs = [c.value_and_jacobian(q) for c in self.cvs]
        return CVResult(
            np.concatenate([r.value for r in rs]),
            np.concatenate([r.jacobian for r in rs], axis=0),
        )

    def value(self, q):
        return self.value_and_jacobian(q).value


class PythonCV:
    """Full-ring plugin protocol; factory returns a CV with explicit metadata.

    The plugin receives a read-only StateView-like object and returns a CVResult
    or (value,jacobian). Its metadata units must be canonical atomic units.
    """

    def __init__(self, factory, params, system, estimator="full_ring", name=None):
        from .potentials.python import import_object

        self.plugin = import_object(factory)(**params)
        attr = factory.rsplit(":", 1)[-1]
        self.system = system
        self.name = name or attr
        self.estimator = estimator
        meta = getattr(self.plugin, "metadata", None)

        def field(n, default):
            if meta is None:
                return getattr(self.plugin, n, default)
            return (
                meta.get(n, default)
                if isinstance(meta, dict)
                else getattr(meta, n, default)
            )

        dimension = field("dimension", 1)
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, (int, np.integer))
            or dimension < 1
        ):
            raise ValueError("Python CV dimension must be a positive integer")
        self.dimension = int(dimension)
        self.units = tuple(field("units", ("dimensionless",) * self.dimension))
        self.periods = tuple(field("periods", (None,) * self.dimension))
        if len(self.units) != self.dimension or len(self.periods) != self.dimension:
            raise ValueError("Python CV metadata dimensions disagree")
        if not all(
            u in ("bohr", "hartree", "radian", "dimensionless") for u in self.units
        ):
            raise ValueError(
                "Python CV must declare canonical AU/radian/dimensionless output units"
            )
        if any(p is not None and (not np.isfinite(p) or p <= 0) for p in self.periods):
            raise ValueError("CV periods must be positive finite numbers or None")
        if estimator not in ("centroid", "bead_average", "full_ring"):
            raise ValueError("unknown Python CV estimator")

    def value_and_jacobian(self, q):
        q = _coordinates(q)
        view = q.view()
        view.flags.writeable = False
        masses = self.system.masses.view()
        masses.flags.writeable = False
        state = SimpleNamespace(
            q=view,
            masses=masses,
            atom_ids=self.system.atom_ids,
            step_index=0,
            time_au=0.0,
            cell=None if getattr(self.system, "cell", None) is None else self.system.cell.copy(),
            pbc=np.asarray(getattr(self.system, "pbc", (False, False, False))).copy(),
        )
        result = self.plugin.value_and_jacobian(state)
        if hasattr(result, "value"):
            value, jac = result.value, result.jacobian
        else:
            value, jac = result
        value, jac = np.asarray(value, float), np.asarray(jac, float)
        if value.shape != (self.dimension,) or jac.shape != (self.dimension,) + q.shape:
            raise ValueError("custom CV must return [D] and [D,P,N,3]")
        if not np.isfinite(value).all() or not np.isfinite(jac).all():
            raise ValueError("custom CV returned non-finite values or derivatives")
        return CVResult(value, jac)

    def value(self, q):
        return self.value_and_jacobian(q).value


def make_cv(spec, system):
    if hasattr(spec, "value_and_jacobian"):
        return spec
    from .units import parse_quantity

    kind = spec.get("kind", "distance")
    kwargs = {"estimator": spec.get("estimator", "centroid"), "name": spec.get("name")}
    periodic = bool(np.any(getattr(system, "pbc", False)))
    if periodic:
        from .cv_periodic import PeriodicDistance, PeriodicBondDifference, PeriodicAngle

        if kind == "distance":
            return PeriodicDistance(spec["atoms"], system.cell, system.pbc, **kwargs)
        if kind in ("difference", "bond_difference"):
            return PeriodicBondDifference(cell=system.cell, pbc=system.pbc,
                atoms=spec.get("atoms"), bonds=spec.get("bonds"), **kwargs)
        if kind == "angle":
            return PeriodicAngle(spec["atoms"], system.cell, system.pbc, **kwargs)
        if kind in ("com_distance", "rpmdrate", "rpmdrate_xi"):
            raise ValueError("Periodic COM/reaction CV needs an explicit molecular image convention; use a periodic distance or continuous projection")
    if kind in ("projection", "fractional"):
        from .cv_periodic import LinearProjection, FractionalCoordinate

        options = dict(atom=spec["atom"], masses=system.masses,
                       reference_atoms=spec.get("reference_atoms"), **kwargs)
        options["origin"] = parse_quantity(spec.get("origin", 0.0),
                                           "dimensionless" if kind == "fractional" else "length")
        if kind == "fractional":
            return FractionalCoordinate(axis=spec.get("axis", 0), cell=system.cell,
                                        pbc=system.pbc, **options)
        return LinearProjection(direction=spec["direction"], **options)
    if kind == "distance":
        return Distance(spec["atoms"], **kwargs)
    if kind in ("difference", "bond_difference"):
        return BondDifference(
            atoms=spec.get("atoms"), bonds=spec.get("bonds"), **kwargs
        )
    if kind == "angle":
        return Angle(spec["atoms"], **kwargs)
    if kind == "cartesian":
        return Cartesian(spec["atom"], spec.get("axis", 0), **kwargs)
    if kind == "com_distance":
        return COMDistance(
            spec.get("fragments", system.fragments), system.masses, **kwargs
        )
    if kind in ("rpmdrate", "rpmdrate_xi"):
        return RPMDrateCoordinate(
            spec.get("fragments", system.fragments),
            system.masses,
            spec.get("reference", system.geometry),
            spec["forming_bonds"],
            spec["breaking_bonds"],
            parse_quantity(spec["rinf"], "length"),
            equivalent_paths=spec.get("equivalent_paths"),
            **kwargs,
        )
    if kind == "python":
        return PythonCV(spec["factory"], spec.get("params", {}), system, **kwargs)
    raise ValueError(f"unknown CV kind: {kind}")


def make_cvs(specs, system, names=None):
    if names is not None:
        by_name = {s["name"]: s for s in specs}
        if len(by_name) != len(specs):
            raise ValueError("CV names must be unique")
        specs = [by_name[n] for n in names]
    return CVStack([make_cv(s, system) for s in specs])
