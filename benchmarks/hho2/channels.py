"""Geometric diagnostics for the H + HO2 benchmark in H0,H1,O0,O1 order.

Channel labels describe centroid connectivity only. They are not validated
scattering outcomes or branching fractions. Likewise, a path on a PMF grid
establishes connectivity of the sampled two-dimensional projection only; it
cannot resolve hidden coordinates or demonstrate a dynamical reaction path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import heapq

import numpy as np

from pyrpmd.constants import BOHR_ANGSTROM


ATOM_LABELS = ("H0", "H1", "O0", "O1")
BOND_PAIRS = (
    (0, 1, "hh"),
    (0, 2, "oh"),
    (0, 3, "oh"),
    (1, 2, "oh"),
    (1, 3, "oh"),
    (2, 3, "oo"),
)
CHANNELS = (
    "reactant_H0+H1O2",
    "exchanged_H1+H0O2",
    "H2+O2",
    "OH+OH",
    "H2O+O",
    "intermediate/ambiguous",
)


@dataclass(frozen=True)
class BondThresholds:
    """Illustrative, configurable centroid bond thresholds in angstrom.

    A distance <= ``formed`` is a bond, a distance >= ``broken`` is absent,
    and every distance strictly between them is ambiguous. These bounds
    require sensitivity checks on the chosen PES and temperature before
    chemical interpretation. There is no persistent trajectory history.
    """

    hh_formed: float = 1.00
    hh_broken: float = 1.40
    oh_formed: float = 1.25
    oh_broken: float = 1.75
    oo_formed: float = 1.65
    oo_broken: float = 2.10

    def __post_init__(self):
        for kind in ("hh", "oh", "oo"):
            formed = getattr(self, kind + "_formed")
            broken = getattr(self, kind + "_broken")
            if (
                isinstance(formed, (bool, np.bool_))
                or isinstance(broken, (bool, np.bool_))
                or not np.isfinite([formed, broken]).all()
                or not 0 < formed < broken
            ):
                raise ValueError(
                    f"{kind} thresholds must be finite and satisfy 0 < formed < broken"
                )


def _centroid_angstrom(positions, units):
    q = np.asarray(positions, dtype=float)
    if q.shape == (4, 3):
        centroid = q
    elif q.ndim == 3 and q.shape[0] > 0 and q.shape[1:] == (4, 3):
        centroid = q.mean(axis=0)
    else:
        raise ValueError("positions must have shape (4, 3) or (P, 4, 3), P >= 1")
    if not np.isfinite(q).all():
        raise ValueError("positions must be finite")
    if units not in ("bohr", "angstrom"):
        raise ValueError("units must be 'bohr' or 'angstrom'")
    return centroid * (BOHR_ANGSTROM if units == "bohr" else 1.0)


def centroid_distances(positions, *, units="bohr"):
    """Return all six interatomic centroid distances, in angstrom.

    ``positions`` is a centroid geometry ``(4, 3)`` or a ring polymer
    ``(P, 4, 3)`` with atom order H0,H1,O0,O1. This computes distances of
    centroid positions, not the average of bead distances. The benchmark
    assumes nonperiodic gas-phase coordinates; no minimum image is applied.
    """
    q = _centroid_angstrom(positions, units)
    return {
        f"{ATOM_LABELS[i]}-{ATOM_LABELS[j]}": float(np.linalg.norm(q[i] - q[j]))
        for i, j, _ in BOND_PAIRS
    }


def _components(bonds):
    remaining = set(range(4))
    components = []
    while remaining:
        seen = {min(remaining)}
        stack = list(seen)
        while stack:
            atom = stack.pop()
            for i, j in bonds:
                other = j if atom == i else i if atom == j else None
                if other is not None and other not in seen:
                    seen.add(other)
                    stack.append(other)
        remaining.difference_update(seen)
        components.append(sorted(seen))
    return components


def classify_centroid(positions, *, units="bohr", thresholds=None):
    """Return a JSON-compatible channel label and complete bond diagnostics.

    ``thresholds`` is a :class:`BondThresholds` or a mapping with its keyword
    arguments (all in angstrom). Any gray-zone bond yields
    ``intermediate/ambiguous``, even if the definite bonds resemble a product.
    Channel matches require the complete expected bond graph, preventing
    transient complexes or overcoordinated structures from being products.
    H0/H1 labels retain incoming-atom identity for exchange diagnostics.
    """
    if thresholds is None:
        thresholds = BondThresholds()
    elif isinstance(thresholds, Mapping):
        thresholds = BondThresholds(**thresholds)
    elif not isinstance(thresholds, BondThresholds):
        raise TypeError("thresholds must be BondThresholds or a mapping")

    distances = centroid_distances(positions, units=units)
    bonds = set()
    states = {}
    ambiguous = []
    for i, j, kind in BOND_PAIRS:
        name = f"{ATOM_LABELS[i]}-{ATOM_LABELS[j]}"
        distance = distances[name]
        if distance <= getattr(thresholds, kind + "_formed"):
            states[name] = "formed"
            bonds.add((i, j))
        elif distance >= getattr(thresholds, kind + "_broken"):
            states[name] = "broken"
        else:
            states[name] = "ambiguous"
            ambiguous.append(name)

    patterns = (
        ("reactant_H0+H1O2", ({(1, 2), (2, 3)}, {(1, 3), (2, 3)})),
        ("exchanged_H1+H0O2", ({(0, 2), (2, 3)}, {(0, 3), (2, 3)})),
        ("H2+O2", ({(0, 1), (2, 3)},)),
        ("OH+OH", ({(0, 2), (1, 3)}, {(0, 3), (1, 2)})),
        ("H2O+O", ({(0, 2), (1, 2)}, {(0, 3), (1, 3)})),
    )
    channel = "intermediate/ambiguous"
    reason = "bond_threshold_gray_zone" if ambiguous else "unrecognized_connectivity"
    if not ambiguous:
        for label, alternatives in patterns:
            if any(bonds == graph for graph in alternatives):
                channel = label
                reason = "complete_centroid_bond_graph_match"
                break
    return {
        "channel": channel,
        "reason": reason,
        "atom_order": list(ATOM_LABELS),
        "distances_angstrom": distances,
        "bond_states": states,
        "ambiguous_bonds": ambiguous,
        "components": _components(bonds),
        "thresholds_angstrom": asdict(thresholds),
        "interpretation": "centroid geometry only; not a validated dynamical outcome",
    }


def _grid_index(value, shape, name):
    try:
        index = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must contain two integer grid indices") from error
    if len(index) != 2 or any(
        isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
        for i in index
    ):
        raise ValueError(f"{name} must contain two integer grid indices")
    if any(i < 0 or i >= size for i, size in zip(index, shape)):
        raise ValueError(f"{name} lies outside the PMF grid")
    return tuple(int(i) for i in index)


def minimax_path(free_energy, start, goal, *, mask=None):
    """Minimize the maximum grid free energy along a four-neighbor path.

    ``start`` and ``goal`` are two integer index pairs. ``mask=True`` means
    unsupported. A numpy masked array is also accepted. NaN, +/-inf and
    masked cells are impassable, including endpoints. A disconnected result
    has ``path=None`` and ``bottleneck=None``. Diagonal steps and interpolation
    across unsupported bins are never used. Energies retain the input units;
    barriers are relative to each endpoint, with the endpoints on the path.

    This graph calculation diagnoses only the supported 2D projection. It
    does not establish equilibration, a high-dimensional connection, a
    transition state, or any kinetic rate. Choose the support mask using the
    sampling/uncertainty diagnostics before calling this function.
    """
    array = np.ma.asarray(free_energy, dtype=float)
    free = np.asarray(array.data, dtype=float)
    if free.ndim != 2 or not free.size:
        raise ValueError("free_energy must be a nonempty two-dimensional grid")
    blocked = np.ma.getmaskarray(array) | ~np.isfinite(free)
    if mask is not None:
        extra_mask = np.asarray(mask, dtype=bool)
        if extra_mask.shape != free.shape:
            raise ValueError("mask must have the same shape as free_energy")
        blocked |= extra_mask
    start = _grid_index(start, free.shape, "start")
    goal = _grid_index(goal, free.shape, "goal")
    result = {
        "connected": False,
        "path": None,
        "bottleneck": None,
        "barrier_from_start": None,
        "barrier_from_goal": None,
        "reason": "disconnected_supported_grid",
        "interpretation": "connectivity of supported 2D projection only",
    }
    if blocked[start] or blocked[goal]:
        result["reason"] = "unsupported_endpoint"
        return result

    costs = np.full(free.shape, np.inf)
    costs[start] = free[start]
    queue = [(float(free[start]), start)]
    parent = {}
    while queue:
        cost, current = heapq.heappop(queue)
        if cost != costs[current]:
            continue
        if current == goal:
            path = [current]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            result.update(
                connected=True,
                path=[list(index) for index in path],
                bottleneck=float(cost),
                barrier_from_start=float(cost - free[start]),
                barrier_from_goal=float(cost - free[goal]),
                reason="supported_four_neighbor_path",
            )
            return result
        row, col = current
        for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            i, j = neighbor
            if not (0 <= i < free.shape[0] and 0 <= j < free.shape[1]):
                continue
            if blocked[neighbor]:
                continue
            candidate = max(cost, float(free[neighbor]))
            if candidate < costs[neighbor]:
                costs[neighbor] = candidate
                parent[neighbor] = current
                heapq.heappush(queue, (candidate, neighbor))
    return result
