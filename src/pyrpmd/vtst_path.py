"""Mass-scaled minimum-energy paths and endpoint validation for VTST."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import AMU_TO_AU
from .vtst_core import (
    VTSTError, _finite_array, _jsonable, _positive, _positive_integer,
    _provider_energy_gradient, _saddle_mode, _scaled_gradient, _scaled_hessian,
    cartesian_coordinates, evaluate_energy_gradient_hessian,
    mass_scaled_coordinates,
)


@dataclass(frozen=True)
class MEPOptions:
    """Controls for bidirectional mass-scaled steepest-descent integration."""

    step_bohr: float = 0.05
    initial_displacement_bohr: float = 0.02
    max_steps_per_side: int = 80
    gradient_tolerance_hartree_per_bohr: float = 1.0e-5
    hessian_step_bohr: float = 1.0e-4
    negative_eigenvalue_tolerance: float = 1.0e-9
    scaling_mass: float = AMU_TO_AU
    minimum_steps_per_side: int = 2

    def validated(self) -> "MEPOptions":
        return MEPOptions(
            step_bohr=_positive(self.step_bohr, "step_bohr"),
            initial_displacement_bohr=_positive(
                self.initial_displacement_bohr, "initial_displacement_bohr"
            ),
            max_steps_per_side=_positive_integer(
                self.max_steps_per_side, "max_steps_per_side"
            ),
            gradient_tolerance_hartree_per_bohr=_positive(
                self.gradient_tolerance_hartree_per_bohr,
                "gradient_tolerance_hartree_per_bohr",
            ),
            hessian_step_bohr=_positive(self.hessian_step_bohr, "hessian_step_bohr"),
            negative_eigenvalue_tolerance=_positive(
                self.negative_eigenvalue_tolerance,
                "negative_eigenvalue_tolerance",
                allow_zero=True,
            ),
            scaling_mass=_positive(self.scaling_mass, "scaling_mass"),
            minimum_steps_per_side=_positive_integer(
                self.minimum_steps_per_side, "minimum_steps_per_side", minimum=0
            ),
        )


@dataclass
class MEPPath:
    """A signed mass-scaled path ordered from reactants to products."""

    s_bohr: np.ndarray
    geometries_bohr: np.ndarray
    energies_hartree: np.ndarray
    gradients_hartree_per_bohr: np.ndarray
    hessians_hartree_per_bohr2: np.ndarray
    tangent_scaled: np.ndarray
    side_converged: tuple[bool, bool]
    saddle_eigenvalues_omega2: np.ndarray
    scaling_mass: float
    diagnostics: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        count = len(self.s_bohr)
        if count < 3:
            raise VTSTError("MEPPath requires at least three points")
        if not np.all(np.diff(self.s_bohr) > 0):
            raise VTSTError("MEPPath.s_bohr must be strictly increasing")
        if self.geometries_bohr.shape[0] != count:
            raise VTSTError("MEPPath geometry count does not match s")
        if self.energies_hartree.shape != (count,):
            raise VTSTError("MEPPath energy count does not match s")
        if self.gradients_hartree_per_bohr.shape != self.geometries_bohr.shape:
            raise VTSTError("MEPPath gradient shape does not match geometry")
        dimension = int(np.prod(self.geometries_bohr.shape[1:]))
        if self.hessians_hartree_per_bohr2.shape != (count, dimension, dimension):
            raise VTSTError("MEPPath Hessian shape does not match geometry")
        if self.tangent_scaled.shape != (count, dimension):
            raise VTSTError("MEPPath tangent shape does not match geometry")

    @property
    def natoms(self) -> int:
        return self.geometries_bohr.shape[1]

    @property
    def saddle_index(self) -> int:
        return int(np.argmin(np.abs(self.s_bohr)))

    def reversed(self) -> "MEPPath":
        return MEPPath(
            s_bohr=-self.s_bohr[::-1],
            geometries_bohr=self.geometries_bohr[::-1].copy(),
            energies_hartree=self.energies_hartree[::-1].copy(),
            gradients_hartree_per_bohr=self.gradients_hartree_per_bohr[::-1].copy(),
            hessians_hartree_per_bohr2=self.hessians_hartree_per_bohr2[::-1].copy(),
            tangent_scaled=-self.tangent_scaled[::-1].copy(),
            side_converged=(self.side_converged[1], self.side_converged[0]),
            saddle_eigenvalues_omega2=self.saddle_eigenvalues_omega2.copy(),
            scaling_mass=self.scaling_mass,
            diagnostics=list(self.diagnostics),
        )

    def to_dict(self, *, include_hessians: bool = True) -> dict[str, Any]:
        result = {
            "s_bohr": self.s_bohr,
            "geometries_bohr": self.geometries_bohr,
            "energies_hartree": self.energies_hartree,
            "gradients_hartree_per_bohr": self.gradients_hartree_per_bohr,
            "tangent_scaled": self.tangent_scaled,
            "side_converged": self.side_converged,
            "saddle_eigenvalues_omega2": self.saddle_eigenvalues_omega2,
            "scaling_mass_electron_masses": self.scaling_mass,
            "diagnostics": self.diagnostics,
        }
        if include_hessians:
            result["hessians_hartree_per_bohr2"] = self.hessians_hartree_per_bohr2
        return _jsonable(result)


def _path_tangents(
    geometries: np.ndarray,
    s: np.ndarray,
    masses: np.ndarray,
    scaling_mass: float,
) -> np.ndarray:
    x = np.array(
        [mass_scaled_coordinates(q, masses, scaling_mass=scaling_mass) for q in geometries]
    )
    edge_order = 2 if len(s) >= 3 else 1
    derivative = np.gradient(x, s, axis=0, edge_order=edge_order)
    norm = np.linalg.norm(derivative, axis=1)
    if np.any(norm <= 1.0e-13):
        raise VTSTError("MEP contains coincident points or an undefined tangent")
    tangent = derivative / norm[:, None]
    # Remove arbitrary finite-difference phase flips.
    for index in range(1, len(tangent)):
        if np.dot(tangent[index - 1], tangent[index]) < 0:
            tangent[index] *= -1
    return tangent


def trace_mep(
    potential: Any,
    saddle_geometry_bohr: np.ndarray,
    masses: Sequence[float],
    *,
    options: MEPOptions | None = None,
) -> MEPPath:
    """Trace a mass-scaled MEP in both directions from a first-order saddle.

    A second-order Heun step follows normalized steepest descent in isoinertial
    coordinates.  A short monotonic-energy line search rejects unstable steps.
    Energy, gradient, and Hessian are retained at every accepted path point.
    """
    opts = (options or MEPOptions()).validated()
    saddle = _finite_array(saddle_geometry_bohr, None, "saddle_geometry_bohr")
    if saddle.ndim != 2 or saddle.shape[1] != 3:
        raise VTSTError("saddle_geometry_bohr must have shape (N,3)")
    mass = _finite_array(masses, (len(saddle),), "masses")
    if np.any(mass <= 0):
        raise VTSTError("masses must be positive")
    saddle_energy, saddle_gradient, saddle_hessian = evaluate_energy_gradient_hessian(
        potential, saddle, finite_difference_step=opts.hessian_step_bohr
    )
    max_force = float(np.max(np.abs(saddle_gradient)))
    if max_force > max(10.0 * opts.gradient_tolerance_hartree_per_bohr, 1.0e-4):
        raise VTSTError(
            f"MEP start is not a stationary saddle: max gradient {max_force:g} hartree/bohr"
        )
    unstable, eigenvalues = _saddle_mode(
        saddle,
        mass,
        saddle_hessian,
        scaling_mass=opts.scaling_mass,
        negative_tolerance=opts.negative_eigenvalue_tolerance,
    )
    saddle_x = mass_scaled_coordinates(saddle, mass, scaling_mass=opts.scaling_mass)

    def direction(x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        q = cartesian_coordinates(x, mass, scaling_mass=opts.scaling_mass)
        energy, gradient = _provider_energy_gradient(potential, q)
        gx = _scaled_gradient(gradient, mass, opts.scaling_mass)
        norm = float(np.linalg.norm(gx))
        if norm <= 1.0e-20:
            return energy, gradient, np.zeros_like(gx)
        return energy, gradient, -gx / norm

    def branch(sign: float) -> tuple[list[tuple[np.ndarray, float, np.ndarray, np.ndarray]], bool]:
        x = saddle_x + sign * opts.initial_displacement_bohr * unstable
        records: list[tuple[np.ndarray, float, np.ndarray, np.ndarray]] = []
        converged = False
        previous_energy = saddle_energy
        for step_index in range(opts.max_steps_per_side):
            q = cartesian_coordinates(x, mass, scaling_mass=opts.scaling_mass)
            energy, gradient, hessian = evaluate_energy_gradient_hessian(
                potential, q, finite_difference_step=opts.hessian_step_bohr
            )
            records.append((x.copy(), energy, gradient, hessian))
            max_gradient = float(np.max(np.abs(gradient)))
            if (
                step_index >= opts.minimum_steps_per_side
                and max_gradient <= opts.gradient_tolerance_hartree_per_bohr
            ):
                converged = True
                break
            gx = _scaled_gradient(gradient, mass, opts.scaling_mass)
            gradient_norm = float(np.linalg.norm(gx))
            if gradient_norm <= 1.0e-20:
                converged = True
                break
            first = -gx / gradient_norm
            # Normalized IRC integration needs a vanishing step near a channel
            # minimum; otherwise a weak asymptotic force can hide an oscillation
            # in a stiff bound mode.  A directional Newton length supplies that
            # local scale without changing the path direction.
            hx = _scaled_hessian(hessian, mass, opts.scaling_mass)
            directional_curvature = abs(float(first @ hx @ first))
            newton_length = (
                gradient_norm / directional_curvature
                if directional_curvature > 1.0e-16
                else opts.step_bohr
            )
            accepted = False
            step_size = min(opts.step_bohr, max(1.0e-8, 0.8 * newton_length))
            for _ in range(12):
                predictor = x + step_size * first
                _, _, second = direction(predictor)
                # A normalized-gradient predictor can cross a nearby minimum.
                # In that case the two unit directions nearly cancel, and
                # normalizing their tiny residual would amplify an unrelated
                # asymptotic force.  Refine the local step instead.
                if np.any(second) and float(np.dot(first, second)) < -0.25:
                    step_size *= 0.5
                    continue
                trial_direction = first if not np.any(second) else first + second
                trial_norm = float(np.linalg.norm(trial_direction))
                if trial_norm <= 1.0e-12:
                    step_size *= 0.5
                    continue
                trial = x + step_size * trial_direction / trial_norm
                trial_energy, _ = _provider_energy_gradient(
                    potential,
                    cartesian_coordinates(trial, mass, scaling_mass=opts.scaling_mass),
                )
                tolerance = 1.0e-10 * max(1.0, abs(previous_energy))
                if trial_energy <= energy + tolerance:
                    x = trial
                    previous_energy = trial_energy
                    accepted = True
                    break
                step_size *= 0.5
            if not accepted:
                break
        return records, converged

    minus, minus_converged = branch(-1.0)
    plus, plus_converged = branch(1.0)
    if not minus or not plus:
        raise VTSTError("MEP integration failed before producing both path branches")

    # Reverse the minus branch so the path is ordered through the saddle.
    combined = list(reversed(minus))
    combined.append((saddle_x, saddle_energy, saddle_gradient, saddle_hessian))
    combined.extend(plus)
    x_values = np.array([record[0] for record in combined])
    geometries = np.array(
        [cartesian_coordinates(x, mass, scaling_mass=opts.scaling_mass) for x in x_values]
    )
    energies = np.array([record[1] for record in combined])
    gradients = np.array([record[2] for record in combined])
    hessians = np.array([record[3] for record in combined])
    saddle_index = len(minus)
    segment = np.linalg.norm(np.diff(x_values, axis=0), axis=1)
    s = np.empty(len(combined), dtype=float)
    s[saddle_index] = 0.0
    for index in range(saddle_index - 1, -1, -1):
        s[index] = s[index + 1] - segment[index]
    for index in range(saddle_index + 1, len(combined)):
        s[index] = s[index - 1] + segment[index - 1]
    tangent = _path_tangents(geometries, s, mass, opts.scaling_mass)
    diagnostics = [
        "MEP integrated with normalized mass-scaled Heun steepest descent",
        f"negative saddle modes: {int(np.count_nonzero(eigenvalues < -opts.negative_eigenvalue_tolerance))}",
        f"reactant-side endpoint max gradient: {np.max(np.abs(gradients[0])):.6g} hartree/bohr",
        f"product-side endpoint max gradient: {np.max(np.abs(gradients[-1])):.6g} hartree/bohr",
    ]
    if not (minus_converged and plus_converged):
        diagnostics.append(
            "One or both MEP sides reached the requested integration limit before the force tolerance."
        )
    return MEPPath(
        s_bohr=s,
        geometries_bohr=geometries,
        energies_hartree=energies,
        gradients_hartree_per_bohr=gradients,
        hessians_hartree_per_bohr2=hessians,
        tangent_scaled=tangent,
        side_converged=(minus_converged, plus_converged),
        saddle_eigenvalues_omega2=eigenvalues,
        scaling_mass=opts.scaling_mass,
        diagnostics=diagnostics,
    )


def validate_reaction_path_endpoints(
    path: MEPPath,
    symbols: Sequence[str],
    reaction: Mapping[str, Any],
    *,
    bond_scale: float = 1.25,
    covalent_radii: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Check that the two MEP termini realize the declared bond changes.

    The check is deliberately topological.  It does not infer a reaction from
    composition and it does not silently choose an orientation when both
    directions satisfy an underspecified declaration.
    """
    if not isinstance(reaction, Mapping):
        raise VTSTError("reaction must be a mapping")
    if len(symbols) != path.natoms:
        raise VTSTError("symbols must contain one entry per atom")
    from .workflows.prepare import bonding_graph, graph_fragments

    scale = _positive(bond_scale, "bond_scale")
    left = bonding_graph(
        path.geometries_bohr[0], symbols, scale, radii=covalent_radii
    )
    right = bonding_graph(
        path.geometries_bohr[-1], symbols, scale, radii=covalent_radii
    )

    def pairs(name: str) -> set[tuple[int, int]]:
        raw = reaction.get(name, ())
        result: set[tuple[int, int]] = set()
        for item in raw:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or any(
                    isinstance(atom, (bool, np.bool_))
                    or not isinstance(atom, (int, np.integer))
                    for atom in item
                )
            ):
                raise VTSTError(f"reaction.{name} must contain atom-index pairs")
            i, j = sorted((int(item[0]), int(item[1])))
            if i == j or i < 0 or j >= path.natoms:
                raise VTSTError(f"reaction.{name} contains an invalid atom-index pair")
            result.add((i, j))
        return result

    forming = pairs("forming_bonds")
    breaking = pairs("breaking_bonds")
    if not forming or not breaking or len(forming) != len(breaking):
        raise VTSTError(
            "endpoint validation requires equal nonempty reaction.forming_bonds and reaction.breaking_bonds"
        )

    declared_reactants = reaction.get("fragments")
    declared_products = reaction.get("product_fragments")

    def canonical(groups: Sequence[Sequence[int]] | None) -> set[frozenset[int]] | None:
        if groups is None:
            return None
        return {frozenset(int(atom) for atom in group) for group in groups}

    expected_reactants = canonical(declared_reactants)
    expected_products = canonical(declared_products)

    def check(start: set[tuple[int, int]], stop: set[tuple[int, int]]) -> bool:
        if not breaking <= start or forming & start:
            return False
        if breaking & stop or not forming <= stop:
            return False
        if (start - breaking) | forming != stop:
            return False
        start_fragments = canonical(graph_fragments(path.natoms, start))
        stop_fragments = canonical(graph_fragments(path.natoms, stop))
        if expected_reactants is not None and start_fragments != expected_reactants:
            return False
        if expected_products is not None and stop_fragments != expected_products:
            return False
        return True

    forward = check(left, right)
    reverse = check(right, left)
    if forward and reverse:
        orientation = "ambiguous_symmetric"
        connected = True
    elif forward:
        orientation = "left_to_right"
        connected = True
    elif reverse:
        orientation = "right_to_left"
        connected = True
    else:
        orientation = "inconsistent"
        connected = False
    return {
        "checked": True,
        "connected": connected,
        "orientation": orientation,
        "left_bonds": [list(pair) for pair in sorted(left)],
        "right_bonds": [list(pair) for pair in sorted(right)],
        "left_fragments": graph_fragments(path.natoms, left),
        "right_fragments": graph_fragments(path.natoms, right),
        "forming_bonds": [list(pair) for pair in sorted(forming)],
        "breaking_bonds": [list(pair) for pair in sorted(breaking)],
        "bond_scale": scale,
    }
