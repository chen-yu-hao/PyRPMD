"""Centroid fragment diagnostics in fixed Br,H,Cl atom order.

The three chemical arrangements are distinct: H+BrCl, HCl+Br, HBr+Cl.
Labels describe geometry under configurable distance cutoffs, not committed
scattering products. All three distances, including the unbiased Br-Cl
distance, are required. The grid path routine is shared with the H+HO2
benchmark and only describes a supported two-dimensional projection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from pyrpmd.constants import BOHR_ANGSTROM
from benchmarks.hho2.channels import minimax_path


ATOM_LABELS = ("Br", "H", "Cl")
BOND_PAIRS = ((0, 1, "hbr"), (1, 2, "hcl"), (0, 2, "brcl"))
CHANNELS = ("H+BrCl", "HCl+Br", "HBr+Cl", "intermediate/ambiguous")


@dataclass(frozen=True)
class BondThresholds:
    """Illustrative centroid distance thresholds, in angstrom.

    BrHCl_LEPS1 uses Re(HBr)=1.414, Re(HCl)=1.275 and Re(BrCl)=2.136 A.
    The default formed limits are about 25--30% above those lengths, with
    a deliberately separate broken limit. These margins are diagnostic
    choices requiring sensitivity checks, not parameters fitted to kinetics.
    Distances <= formed are bonds; distances >= broken are absent; values
    strictly between the limits are ambiguous. No trajectory history or
    residence-time commitment is inferred. ``minimum_distance`` rejects
    obvious atomic overlap without excluding legitimate collinear geometries.
    """

    hbr_formed: float = 1.80
    hbr_broken: float = 2.35
    hcl_formed: float = 1.65
    hcl_broken: float = 2.15
    brcl_formed: float = 2.65
    brcl_broken: float = 3.35
    minimum_distance: float = 0.40

    def __post_init__(self):
        values = asdict(self)
        if any(isinstance(value, (bool, np.bool_)) for value in values.values()):
            raise ValueError("bond thresholds must be finite positive distances")
        try:
            numeric = np.asarray(list(values.values()), dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("bond thresholds must be finite positive distances") from error
        if not np.isfinite(numeric).all() or np.any(numeric <= 0):
            raise ValueError("bond thresholds must be finite positive distances")
        # Normalize numeric scalar types to keep the diagnostics strict JSON.
        for key, value in zip(values, numeric):
            object.__setattr__(self, key, float(value))
        for kind in ("hbr", "hcl", "brcl"):
            if not self.minimum_distance < getattr(self, kind + "_formed") < getattr(self, kind + "_broken"):
                raise ValueError("each bond requires minimum_distance < formed < broken")


def _centroid_angstrom(positions, units):
    q = np.asarray(positions, dtype=float)
    if units not in ("bohr", "angstrom"):
        raise ValueError("units must be 'bohr' or 'angstrom'")
    if not np.isfinite(q).all():
        raise ValueError("positions must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        if q.shape == (3, 3):
            centroid = q
        elif q.ndim == 3 and q.shape[0] > 0 and q.shape[1:] == (3, 3):
            centroid = q.mean(axis=0)
        else:
            raise ValueError("positions must have shape (3,3) or (P,3,3), P >= 1")
        result = centroid * (BOHR_ANGSTROM if units == "bohr" else 1.0)
    if not np.isfinite(result).all():
        raise ValueError("centroid positions overflowed")
    return result


def centroid_distances(positions, *, units="bohr"):
    """Return Br-H, H-Cl and Br-Cl centroid distances in angstrom.

    Input is one centroid geometry ``(3,3)`` or bead positions ``(P,3,3)``
    in fixed Br,H,Cl order, using nonperiodic gas-phase coordinates. The
    distance of centroids is computed, not the average of bead distances.
    """
    q = _centroid_angstrom(positions, units)
    with np.errstate(over="ignore", invalid="ignore"):
        distances = {
            f"{ATOM_LABELS[i]}-{ATOM_LABELS[j]}": float(np.linalg.norm(q[i] - q[j]))
            for i, j, _ in BOND_PAIRS
        }
    if not np.isfinite(list(distances.values())).all():
        raise ValueError("centroid distances overflowed")
    return distances


def _components(bonds):
    remaining, components = set(range(3)), []
    while remaining:
        component, queue = {min(remaining)}, [min(remaining)]
        while queue:
            atom = queue.pop()
            for i, j in bonds:
                other = j if atom == i else i if atom == j else None
                if other is not None and other not in component:
                    component.add(other)
                    queue.append(other)
        components.append(sorted(component))
        remaining.difference_update(component)
    return components


def _thresholds(value):
    if value is None:
        return BondThresholds()
    if isinstance(value, Mapping):
        return BondThresholds(**value)
    if isinstance(value, BondThresholds):
        return value
    raise TypeError("thresholds must be BondThresholds or a mapping")


def classify_centroid(positions, *, units="bohr", thresholds=None):
    """Classify only complete separated-fragment bond graphs.

    A channel requires exactly one formed pair and the other two pairs
    broken. Any gray bond, extra bond, fully separated geometry or atomic
    overlap is intermediate/ambiguous. HBr and HCl are never interchanged
    by sorting distances or atom positions. ``thresholds`` is a
    :class:`BondThresholds` instance or a mapping of its keyword arguments.
    """
    thresholds = _thresholds(thresholds)
    distances = centroid_distances(positions, units=units)
    states, bonds, ambiguous, overlapping = {}, set(), [], []
    for i, j, kind in BOND_PAIRS:
        name = f"{ATOM_LABELS[i]}-{ATOM_LABELS[j]}"
        distance = distances[name]
        if distance < thresholds.minimum_distance:
            overlapping.append(name)
        if distance <= getattr(thresholds, kind + "_formed"):
            states[name] = "formed"
            bonds.add((i, j))
        elif distance >= getattr(thresholds, kind + "_broken"):
            states[name] = "broken"
        else:
            states[name] = "ambiguous"
            ambiguous.append(name)
    channel = "intermediate/ambiguous"
    reason = (
        "overlapping_atoms" if overlapping else
        "bond_threshold_gray_zone" if ambiguous else
        "unrecognized_connectivity"
    )
    if not overlapping and not ambiguous:
        for label, graph in (
            ("H+BrCl", {(0, 2)}),
            ("HCl+Br", {(1, 2)}),
            ("HBr+Cl", {(0, 1)}),
        ):
            if bonds == graph:
                channel, reason = label, "complete_centroid_bond_graph_match"
                break
    return {
        "channel": channel,
        "reason": reason,
        "atom_order": list(ATOM_LABELS),
        "distances_angstrom": distances,
        "bond_states": states,
        "ambiguous_bonds": ambiguous,
        "overlapping_pairs": overlapping,
        "components": _components(bonds),
        "thresholds_angstrom": asdict(thresholds),
        "interpretation": "centroid fragment geometry only; not a committed scattering outcome",
    }


def classify_centroids(q_batch, *, units="bohr", thresholds=None):
    """Vectorized labels for already computed centroids of shape ``[N,3,3]``.

    Atom order, units, gray zones and overlap rules are identical to
    :func:`classify_centroid`. This API treats the leading dimension as
    independent frames, never as beads. For a bead trajectory ``[N,P,3,3]``,
    first compute its centroid along the bead axis. Empty batches are valid.
    """
    thresholds = _thresholds(thresholds)
    q = np.asarray(q_batch, dtype=float)
    if q.ndim != 3 or q.shape[1:] != (3, 3) or not np.isfinite(q).all():
        raise ValueError("q_batch must contain finite centroid positions of shape [N,3,3]")
    if units not in ("bohr", "angstrom"):
        raise ValueError("units must be 'bohr' or 'angstrom'")
    with np.errstate(over="ignore", invalid="ignore"):
        q = q * (BOHR_ANGSTROM if units == "bohr" else 1.0)
        distance = np.linalg.norm(q[:, [0, 1, 0]] - q[:, [1, 2, 2]], axis=2)
    if not np.isfinite(distance).all():
        raise ValueError("centroid distances overflowed")
    kinds = [kind for _, _, kind in BOND_PAIRS]
    formed = distance <= np.array([getattr(thresholds, kind + "_formed") for kind in kinds])
    broken = distance >= np.array([getattr(thresholds, kind + "_broken") for kind in kinds])
    valid = (
        (distance >= thresholds.minimum_distance).all(axis=1)
        & (formed | broken).all(axis=1)
        & (formed.sum(axis=1) == 1)
    )
    labels = np.full(len(q), "intermediate/ambiguous", dtype=f"<U{max(map(len, CHANNELS))}")
    for index, channel in enumerate(("HBr+Cl", "HCl+Br", "H+BrCl")):
        labels[valid & formed[:, index]] = channel
    return labels


def select_channel_basins(channel_names, fractions, free_energy, supported, *, minimum_fraction=0.8):
    """Choose each distinct chemical channel's supported minimum-F bin.

    ``fractions`` contains reweighted conditional channel fractions on the
    PMF grid, including ambiguous configurations in their denominator.
    Channels are not merged or renormalized. Missing dominant bins do not
    imply absent chemical channels. This function supplies seeds to
    :func:`minimax_path`, which never establishes a kinetic connection.
    """
    names = list(channel_names)
    free = np.asarray(free_energy, dtype=float)
    support = np.asarray(supported, dtype=bool)
    fraction = np.asarray(fractions, dtype=float)
    if free.ndim != 2 or not free.size or support.shape != free.shape:
        raise ValueError("free_energy and supported must have the same nonempty 2D shape")
    if fraction.shape != (len(names),) + free.shape:
        raise ValueError("fractions need one 2D grid per channel")
    if len(set(names)) != len(names) or any(name not in CHANNELS for name in names):
        raise ValueError("channel names must be unique BrHCl channel labels")
    if isinstance(minimum_fraction, (bool, np.bool_)) or not np.isfinite(minimum_fraction) or not 0 < minimum_fraction <= 1:
        raise ValueError("minimum_fraction must be finite in (0,1]")
    usable = support & np.isfinite(free)
    local = fraction[:, usable]
    if (not np.isfinite(local).all() or np.any(local < 0) or np.any(local > 1)
            or np.any(local.sum(axis=0) > 1 + 1e-10)):
        raise ValueError("supported conditional channel fractions must be finite probabilities")
    basins = {}
    for name, values in zip(names, fraction):
        if name == "intermediate/ambiguous":
            continue
        good = usable & (values >= minimum_fraction)
        if np.any(good):
            index = np.nanargmin(np.where(good, free, np.nan))
            basins[name] = tuple(int(i) for i in np.unravel_index(index, free.shape))
    return basins


__all__ = [
    "ATOM_LABELS", "BOND_PAIRS", "CHANNELS", "BondThresholds",
    "centroid_distances", "classify_centroid", "classify_centroids",
    "select_channel_basins", "minimax_path",
]
