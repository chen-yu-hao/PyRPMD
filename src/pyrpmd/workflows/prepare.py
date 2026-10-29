"""Transparent transition-state preparation from a validated PES.

Finite-difference Hessian and mass-weighted downhill traces are numerical
preparation, not a claim of a converged quantum-chemistry IRC. Directional
ambiguities are returned explicitly rather than assigned an arbitrary rate.
"""

from __future__ import annotations
import copy
from collections import Counter
import numpy as np
from ..units import parse_quantity

_COVALENT_ANG = {
    "H": 0.31,
    "D": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03,
    "Ca": 1.76,
    "Br": 1.20,
    "I": 1.39,
}


def bonding_graph(geometry, symbols, scale=1.25):
    from ..constants import BOHR_ANGSTROM

    q = np.asarray(geometry, float) * BOHR_ANGSTROM
    missing = set(symbols) - _COVALENT_ANG.keys()
    if missing:
        raise ValueError(
            f"automatic connectivity has no radii for {sorted(missing)}; supply reaction explicitly"
        )
    bonds = set()
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            if np.linalg.norm(q[i] - q[j]) < scale * (
                _COVALENT_ANG[symbols[i]] + _COVALENT_ANG[symbols[j]]
            ):
                bonds.add((i, j))
    return bonds


def graph_fragments(natoms, bonds):
    neighbors = [set() for _ in range(natoms)]
    for i, j in bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)
    unseen = set(range(natoms))
    groups = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        group = []
        while stack:
            i = stack.pop()
            group.append(i)
            for j in sorted(neighbors[i] & unseen):
                unseen.remove(j)
                stack.append(j)
        groups.append(sorted(group))
    return groups


def _potential_values(provider, q):
    r = provider.evaluate(np.ascontiguousarray(q[None]))
    if (
        np.any(r.status)
        or not np.isfinite(r.energy).all()
        or not np.isfinite(r.forces).all()
    ):
        raise ValueError("PES failed during transition-state preparation")
    return float(r.energy[0]), np.asarray(r.forces[0], float)


def numerical_hessian(provider, geometry, step=1e-4):
    q = np.asarray(geometry, float)
    flat = q.ravel()
    h = np.empty((flat.size, flat.size))
    if step <= 0:
        raise ValueError("Hessian displacement must be positive")
    for i in range(flat.size):
        plus = flat.copy()
        minus = flat.copy()
        plus[i] += step
        minus[i] -= step
        _, fp = _potential_values(provider, plus.reshape(q.shape))
        _, fm = _potential_values(provider, minus.reshape(q.shape))
        h[:, i] = -(fp - fm).ravel() / (2 * step)
    return 0.5 * (h + h.T)


def downhill_trace(
    provider,
    geometry,
    masses,
    direction,
    displacement=0.1,
    step=0.06,
    max_steps=400,
    gradient_tolerance=1e-5,
):
    q = np.asarray(geometry, float).copy() + displacement * np.asarray(direction, float)
    mass = np.asarray(masses, float)[:, None]
    path = [q.copy()]
    energies = []
    for _ in range(max_steps):
        e, f = _potential_values(provider, q)
        energies.append(e)
        if np.max(np.abs(f)) < gradient_tolerance:
            break
        v = f / mass
        norm = np.max(np.linalg.norm(v, axis=1))
        if norm < 1e-20:
            break
        delta = v * (step / norm)
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            trial = q + scale * delta
            en, _ = _potential_values(provider, trial)
            if en <= e:
                q = trial
                path.append(q.copy())
                accepted = True
                break
        if not accepted:
            break
    return {"geometry": q, "path": np.asarray(path), "energies": np.asarray(energies)}


def _species_key(fragments, symbols, bonds):
    # Composition plus within-fragment bond element pairs distinguishes common
    # valence isomers while retaining atom-permutation-equivalent directions.
    keys = []
    for frag in fragments:
        s = set(frag)
        composition = tuple(sorted(Counter(symbols[i] for i in frag).items()))
        bond_types = tuple(
            sorted(
                tuple(sorted((symbols[i], symbols[j])))
                for i, j in bonds
                if i in s and j in s
            )
        )
        keys.append((composition, bond_types))
    return tuple(sorted(keys))


def _same_channel(fragments_a, fragments_b, symbols, bonds_a, bonds_b, masses=None):
    """Labeled connectivity isomorphism, not composition-only equivalence."""
    if _species_key(fragments_a, symbols, bonds_a) != _species_key(
        fragments_b, symbols, bonds_b
    ):
        return False

    def same_graph(a, b):
        if len(a) != len(b):
            return False
        na = {i: {j for j in a if tuple(sorted((i, j))) in bonds_a} for i in a}
        nb = {i: {j for j in b if tuple(sorted((i, j))) in bonds_b} for i in b}
        candidates = {
            i: [
                j
                for j in b
                if symbols[i] == symbols[j]
                and len(na[i]) == len(nb[j])
                and (
                    masses is None
                    or np.isclose(masses[i], masses[j], rtol=1e-10, atol=1e-8)
                )
            ]
            for i in a
        }
        order = sorted(a, key=lambda i: (len(candidates[i]), -len(na[i])))
        mapping = {}
        used = set()

        def match(depth):
            if depth == len(order):
                return True
            i = order[depth]
            for j in candidates[i]:
                if j in used:
                    continue
                if any((k in na[i]) != (v in nb[j]) for k, v in mapping.items()):
                    continue
                mapping[i] = j
                used.add(j)
                if match(depth + 1):
                    return True
                used.remove(j)
                del mapping[i]
            return False

        return match(0)

    return any(
        same_graph(fragments_a[0], fragments_b[p[0]])
        and same_graph(fragments_a[1], fragments_b[p[1]])
        for p in ((0, 1), (1, 0))
    )


def prepare_rate(config, system, potential=None, numerical=False):
    reaction = copy.deepcopy(config.get("reaction", {}))
    reaction["rinf"] = reaction.get(
        "rinf_au", reaction.get("rinf", reaction.get("Rinf", "12 angstrom"))
    )
    if "fragments" not in reaction and getattr(system, "fragments", None):
        reaction["fragments"] = [list(f) for f in system.fragments]
    required = ("fragments", "forming_bonds", "breaking_bonds")
    explicit = all(reaction.get(k) for k in required)
    if explicit:
        _validate_reaction(reaction, system.natoms)
        reaction["rinf"] = parse_quantity(reaction.get("rinf", "12 angstrom"), "length")
        _validate_equivalent_paths(reaction, system)
        return {
            "ready": True,
            "reaction": reaction,
            "diagnostics": ["explicit reaction definition"],
            "candidates": [],
        }
    if not numerical:
        return {
            "ready": False,
            "reaction": reaction,
            "requires_numerical_preparation": True,
            "diagnostics": [
                "A Hessian and two downhill traces are needed; use prepare --pilot or run."
            ],
            "candidates": [],
        }
    owned = potential is None
    if owned:
        from ..potentials import make_potential

        potential = make_potential(config["potential"], system)
    try:
        options = reaction.get("preparation", {})
        e, f = _potential_values(potential, system.geometry)
        maximum = float(np.max(np.abs(f)))
        threshold = float(options.get("gradient_tolerance", 1e-3))
        if maximum > threshold:
            raise ValueError(
                f"supplied TS is not stationary: max force {maximum:g} > {threshold:g} hartree/bohr"
            )
        h = numerical_hessian(
            potential,
            system.geometry,
            parse_quantity(options.get("hessian_step", 1e-4), "length"),
        )
        repeated = np.repeat(system.masses, 3)
        weighted = h / np.sqrt(repeated[:, None] * repeated[None, :])
        vals, vecs = np.linalg.eigh(weighted)
        negative = np.flatnonzero(
            vals < -float(options.get("negative_eigenvalue_tolerance", 1e-8))
        )
        diagnostics = [
            f"max TS force = {maximum:.6g} hartree/bohr",
            f"negative mass-weighted Hessian eigenvalues = {len(negative)}",
        ]
        if len(negative) != 1:
            return {
                "ready": False,
                "reaction": reaction,
                "diagnostics": diagnostics
                + [
                    "Automatic preparation requires exactly one resolved unstable mode; provide reaction explicitly or refine TS."
                ],
                "hessian_eigenvalues": vals.tolist(),
                "candidates": [],
            }
        direction = (vecs[:, negative[0]] / np.sqrt(repeated)).reshape(
            system.geometry.shape
        )
        direction /= np.max(np.linalg.norm(direction, axis=1))
        traces = [
            downhill_trace(
                potential,
                system.geometry,
                system.masses,
                sign * direction,
                displacement=parse_quantity(options.get("displacement", 0.1), "length"),
                step=parse_quantity(options.get("trace_step", 0.06), "length"),
                max_steps=int(options.get("trace_steps", 400)),
            )
            for sign in (-1, 1)
        ]
        bonds = [
            bonding_graph(
                t["geometry"], system.symbols, float(options.get("bond_scale", 1.25))
            )
            for t in traces
        ]
        fragments = [graph_fragments(system.natoms, b) for b in bonds]
        candidates = []
        for side in (0, 1):
            if len(fragments[side]) != 2:
                continue
            forming = sorted(bonds[1 - side] - bonds[side])
            breaking = sorted(bonds[side] - bonds[1 - side])
            if not forming or len(forming) != len(breaking):
                continue
            candidate = {
                "fragments": fragments[side],
                "forming_bonds": [list(b) for b in forming],
                "breaking_bonds": [list(b) for b in breaking],
                "entry_side": side,
                "rinf": parse_quantity(reaction.get("rinf", "12 angstrom"), "length"),
            }
            candidates.append(candidate)
        if "entry_side" in reaction:
            candidates = [
                c for c in candidates if c["entry_side"] == reaction["entry_side"]
            ]
        if reaction.get("fragments"):
            requested = {frozenset(f) for f in reaction["fragments"]}
            candidates = [
                c
                for c in candidates
                if {frozenset(f) for f in c["fragments"]} == requested
            ]
        if len(candidates) == 2 and _same_channel(
            fragments[0],
            fragments[1],
            system.symbols,
            bonds[0],
            bonds[1],
            system.masses,
        ):
            candidates = candidates[:1]
            diagnostics.append(
                "Permutation-equivalent forward/reverse reactant identities: canonical direction selected."
            )
        if len(candidates) == 1:
            reaction.update(candidates[0])
            _validate_reaction(reaction, system.natoms)
            _validate_equivalent_paths(reaction, system)
        else:
            diagnostics.append(
                "Entrance/channel is ambiguous; select reaction.entry_side or provide fragments and bond changes."
            )
        return {
            "ready": len(candidates) == 1,
            "reaction": reaction,
            "diagnostics": diagnostics,
            "candidates": candidates,
            "hessian_eigenvalues": vals.tolist(),
            "trace_endpoints": [t["geometry"].tolist() for t in traces],
        }
    finally:
        if owned and hasattr(potential, "close"):
            potential.close()


def _validate_reaction(reaction, natoms):
    fs = reaction["fragments"]
    if len(fs) != 2 or any(not f for f in fs):
        raise ValueError("bimolecular rate requires two nonempty fragments")
    flat = [i for f in fs for i in f]
    if sorted(flat) != list(range(natoms)):
        raise ValueError("reaction fragments must partition all atoms")
    forming = reaction["forming_bonds"]
    breaking = reaction["breaking_bonds"]
    if not forming or len(forming) != len(breaking):
        raise ValueError(
            "paired forming/breaking bond lists must have equal nonzero length"
        )
    for bond in forming + breaking:
        if len(bond) != 2 or bond[0] == bond[1] or min(bond) < 0 or max(bond) >= natoms:
            raise ValueError("invalid forming/breaking bond atom index")


def _validate_equivalent_paths(reaction, system):
    if not reaction.get("equivalent_paths"):
        return
    from ..cv import RPMDrateCoordinate

    cv = RPMDrateCoordinate(
        reaction["fragments"],
        system.masses,
        system.geometry,
        reaction["forming_bonds"],
        reaction["breaking_bonds"],
        parse_quantity(reaction["rinf"], "length"),
        equivalent_paths=reaction["equivalent_paths"],
    )
    for permutation in cv.permutations[1:]:
        if any(
            system.symbols[i] != system.symbols[j] for i, j in enumerate(permutation)
        ):
            raise ValueError(
                "equivalent atom permutation must preserve element identities"
            )
