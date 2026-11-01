"""Transparent transition-state preparation from a validated PES.

Finite-difference Hessian and mass-weighted downhill traces are numerical
preparation, not a claim of a converged quantum-chemistry IRC. Directional
ambiguities are returned explicitly rather than assigned an arbitrary rate.
"""

from __future__ import annotations
import copy
from collections import Counter
from collections.abc import Mapping
import numpy as np
from ..covalent import COVALENT_RADII_ANGSTROM, covalent_radii
from ..units import parse_quantity

# Kept for the dRPMD pilot classifier, which shares this connectivity table.
_COVALENT_ANG = {**COVALENT_RADII_ANGSTROM, "D": 0.31, "T": 0.31}


def bonding_graph(geometry, symbols, scale=1.25, radii=None):
    """Heuristic connectivity using sourced or explicitly overridden radii."""
    from ..constants import BOHR_ANGSTROM

    q = np.asarray(geometry)
    symbols = tuple(symbols)
    if q.dtype.kind not in "fiu" or q.shape != (len(symbols), 3) or not np.isfinite(q).all():
        raise ValueError("Connectivity requires finite real geometry[N,3] and N symbols")
    if isinstance(scale, (bool, np.bool_)) or not np.isscalar(scale):
        raise ValueError("bond_scale must be positive and finite")
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("bond_scale must be positive and finite")
    q = np.asarray(q, float) * BOHR_ANGSTROM
    radii = covalent_radii(symbols, radii)
    bonds = set()
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            if np.linalg.norm(q[i] - q[j]) < scale * (
                radii[i] + radii[j]
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
    from ..potentials.base import PotentialError, coordinates, validate_result

    try:
        batch = coordinates(np.asarray(q)[None])
        r = validate_result(provider.evaluate(batch), batch.shape)
    except PotentialError as error:
        raise ValueError(f"PES failed during transition-state preparation: {error}") from error
    # Retain this evaluation across later PES calls. Compiled and Python
    # providers may overwrite their force buffers when evaluating the next
    # finite-difference displacement.
    return float(r.energy[0]), np.array(r.forces[0], dtype=float, copy=True)


def numerical_hessian(provider, geometry, step=1e-4):
    from ..potentials.base import coordinates

    q = coordinates(np.asarray(geometry)[None])[0]
    flat = q.ravel()
    h = np.empty((flat.size, flat.size))
    if isinstance(step, (bool, np.bool_)) or not np.isscalar(step) or not np.isfinite(step) or step <= 0:
        raise ValueError("Hessian displacement must be positive and finite")
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


def preparation_options(reaction):
    """Validate numerical search controls without constructing a PES."""
    raw = reaction.get("preparation", {})
    if not isinstance(raw, Mapping):
        raise ValueError("reaction.preparation must be a table")
    allowed = {
        "refine", "max_steps", "max_step", "gradient_tolerance", "hessian_step",
        "negative_eigenvalue_tolerance", "displacement", "trace_step", "trace_steps",
        "bond_scale", "covalent_radii", "neb",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown reaction.preparation fields: {sorted(unknown)}")
    result = dict(raw)
    if "refine" in raw and not isinstance(raw["refine"], (bool, np.bool_)):
        raise ValueError("reaction.preparation.refine must be boolean")

    def positive(value, name, dimension=None):
        if isinstance(value, np.bool_):
            raise ValueError(f"{name} must be positive and finite")
        value = parse_quantity(value, dimension)
        if value <= 0:
            raise ValueError(f"{name} must be positive and finite")
        return value

    def count(value, name, minimum=1):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return int(value)

    quantities = {
        "max_step": (0.2, "length"),
        "gradient_tolerance": (1e-5, "force"),
        "hessian_step": (1e-4, "length"),
        "negative_eigenvalue_tolerance": (1e-8, (-2, 1, 0, -1, 0, 0)),
        "displacement": (0.1, "length"),
        "trace_step": (0.06, "length"),
        "bond_scale": (1.25, "dimensionless"),
    }
    for key, (default, dimension) in quantities.items():
        result[key] = positive(raw.get(key, default), f"preparation.{key}", dimension)
    for key, default in (("max_steps", 100), ("trace_steps", 400)):
        result[key] = count(raw.get(key, default), f"preparation.{key}")
    # Check overrides even if no connectivity calculation is needed yet.
    covalent_radii((), raw.get("covalent_radii"))
    neb = raw.get("neb", {})
    if not isinstance(neb, Mapping):
        raise ValueError("reaction.preparation.neb must be a table")
    unknown = set(neb) - {
        "images", "spring_constant", "max_steps", "gradient_tolerance",
        "max_step", "climb_after", "initial_path",
    }
    if unknown:
        raise ValueError(f"Unknown reaction.preparation.neb fields: {sorted(unknown)}")
    resolved_neb = dict(neb)
    for key, default, minimum in (("images", 7, 3), ("max_steps", 1000, 1), ("climb_after", 20, 0)):
        resolved_neb[key] = count(neb.get(key, default), f"preparation.neb.{key}", minimum)
    for key, default, dimension in (
        ("spring_constant", 0.1, "spring"),
        ("gradient_tolerance", 1e-4, "force"),
        ("max_step", 0.1, "length"),
    ):
        resolved_neb[key] = positive(neb.get(key, default), f"preparation.neb.{key}", dimension)
    if "initial_path" in neb:
        path = np.asarray(neb["initial_path"])
        if (path.dtype.kind not in "fiu" or path.ndim != 3 or path.shape[-1] != 3
                or path.shape[0] != resolved_neb["images"] or path.shape[1] < 1
                or not np.isfinite(path).all()):
            raise ValueError("preparation.neb.initial_path must be finite real [images,N,3] coordinates in bohr")
    result["neb"] = resolved_neb
    return result


def prepared_system(system, prepared):
    """Return the optimized molecular identity without mutating the input."""
    if "transition_state" not in prepared:
        return system
    from ..system import System

    return System.from_dict({**system.to_dict(), "geometry": prepared["transition_state"]})


def prepare_rate(config, system, potential=None, numerical=False):
    """Prepare a bimolecular rate from a TS guess or atom-mapped endpoints.

    Explicit reactions retain their inexpensive legacy path unless refinement
    is requested. Numerical preparation always validates a searched saddle
    before inferring connectivity; unsuccessful searches are inspectable but
    never marked ready. All returned coordinates are in bohr.
    """
    from ..io import jsonable
    from ..system import reaction_endpoints

    reaction = copy.deepcopy(config.get("reaction", {}))
    options = preparation_options(reaction)
    reaction["rinf"] = reaction.get(
        "rinf_au", reaction.get("rinf", reaction.get("Rinf", "12 angstrom"))
    )
    if "fragments" not in reaction and getattr(system, "fragments", None):
        reaction["fragments"] = [list(f) for f in system.fragments]
    required = ("fragments", "forming_bonds", "breaking_bonds")
    explicit = all(reaction.get(k) for k in required)
    spec = config.get("system", {})
    endpoints = None
    if isinstance(spec, Mapping) and ("reactant" in spec or "product" in spec):
        endpoints = reaction_endpoints(spec)
        if endpoints[0].symbols != system.symbols or not np.array_equal(endpoints[0].masses, system.masses):
            raise ValueError("Reactant/product atom identities and masses must match the supplied System")
    if "neb" in reaction.get("preparation", {}) and endpoints is None:
        raise ValueError("reaction.preparation.neb requires system.reactant and system.product")
    if explicit:
        _validate_reaction(reaction, system.natoms)
        reaction["rinf"] = parse_quantity(reaction["rinf"], "length")
        if endpoints is None and not options.get("refine", False):
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
                "Numerical TS preparation (CI-NEB for endpoints, P-RFO refinement, "
                "Hessian and downhill traces) is needed; use prepare --pilot or run."
            ],
            "candidates": [],
        }
    if any(system.pbc) or system.fixed_atoms:
        raise ValueError("Automatic rate preparation requires a nonperiodic system without fixed atoms")
    owned = potential is None
    if owned:
        from ..potentials import make_potential

        potential = make_potential(config["potential"], system)
    try:
        q = system.geometry.copy()
        diagnostics = []
        result = {
            "ready": False, "reaction": reaction, "diagnostics": diagnostics,
            "candidates": [], "transition_state": q.tolist(),
        }
        initial_mode = None
        if endpoints is not None:
            from .neb import climbing_image_neb

            band = climbing_image_neb(
                potential, endpoints[0].geometry, endpoints[1].geometry, **options["neb"]
            )
            result["neb"] = jsonable(band)
            q = band["geometry"].copy()
            result["transition_state"] = q.tolist()
            diagnostics.extend(band["diagnostics"])
            if not band["converged"]:
                return result
            initial_mode = band["tangent"]
        _, f = _potential_values(potential, q)
        maximum = float(np.max(np.abs(f)))
        threshold = options["gradient_tolerance"]
        if options.get("refine", True) and (
            maximum > threshold or endpoints is not None or options.get("refine", False)
        ):
            from .saddle import refine_transition_state

            refined = refine_transition_state(
                potential, q, masses=system.masses, initial_mode=initial_mode,
                max_steps=options["max_steps"], max_step=options["max_step"],
                gradient_tolerance=threshold, hessian_step=options["hessian_step"],
                negative_eigenvalue_tolerance=options["negative_eigenvalue_tolerance"],
            )
            result["refinement"] = jsonable(refined)
            q = refined["geometry"].copy()
            result["transition_state"] = q.tolist()
            diagnostics.extend(refined["diagnostics"])
            result["hessian_eigenvalues"] = list(refined["hessian_eigenvalues"])
            if not refined["converged"]:
                return result
            maximum = refined["max_force"]
        if maximum > threshold:
            raise ValueError(
                f"supplied TS is not stationary: max force {maximum:g} > {threshold:g} hartree/bohr; "
                "enable reaction.preparation.refine"
            )
        h = numerical_hessian(potential, q, options["hessian_step"])
        repeated = np.repeat(system.masses, 3)
        weighted = h / np.sqrt(repeated[:, None] * repeated[None, :])
        vals, vecs = np.linalg.eigh(weighted)
        negative = np.flatnonzero(vals < -options["negative_eigenvalue_tolerance"])
        diagnostics.extend([
            f"max TS force = {maximum:.6g} hartree/bohr",
            f"negative mass-weighted Hessian eigenvalues = {len(negative)}",
        ])
        result["hessian_eigenvalues"] = vals.tolist()
        if len(negative) != 1:
            diagnostics.append(
                "Automatic preparation requires exactly one resolved unstable mode; "
                "provide a better TS guess or a reactant/product path."
            )
            return result
        optimized_system = prepared_system(system, result)
        if explicit and endpoints is None:
            _validate_equivalent_paths(reaction, optimized_system)
            result["ready"] = True
            diagnostics.append("explicit reaction definition with validated refined TS")
            return result
        direction = (vecs[:, negative[0]] / np.sqrt(repeated)).reshape(q.shape)
        direction /= np.max(np.linalg.norm(direction, axis=1))
        traces = [
            downhill_trace(
                potential, q, system.masses, sign * direction,
                displacement=options["displacement"], step=options["trace_step"],
                max_steps=options["trace_steps"],
            )
            for sign in (-1, 1)
        ]
        bonds = [
            bonding_graph(t["geometry"], system.symbols, options["bond_scale"],
                          radii=options.get("covalent_radii"))
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
            candidates.append({
                "fragments": fragments[side],
                "forming_bonds": [list(b) for b in forming],
                "breaking_bonds": [list(b) for b in breaking],
                "entry_side": side,
                "rinf": parse_quantity(reaction["rinf"], "length"),
            })
        if endpoints is not None:
            endpoint_bonds = [
                bonding_graph(endpoint.geometry, system.symbols, options["bond_scale"],
                              radii=options.get("covalent_radii"))
                for endpoint in endpoints
            ]
            candidates = [
                c for c in candidates
                if bonds[c["entry_side"]] == endpoint_bonds[0]
                and bonds[1 - c["entry_side"]] == endpoint_bonds[1]
            ]
            diagnostics.append(
                "Entrance direction checked against labeled reactant/product endpoint connectivity."
            )
            if not candidates:
                diagnostics.append("TS downhill traces do not connect the supplied endpoint bond graphs.")
        if "entry_side" in reaction:
            candidates = [c for c in candidates if c["entry_side"] == reaction["entry_side"]]
        if reaction.get("fragments"):
            requested = {frozenset(f) for f in reaction["fragments"]}
            candidates = [c for c in candidates if {frozenset(f) for f in c["fragments"]} == requested]
        if explicit:
            for key in ("forming_bonds", "breaking_bonds"):
                requested = {tuple(sorted(b)) for b in reaction[key]}
                candidates = [c for c in candidates if {tuple(sorted(b)) for b in c[key]} == requested]
        if len(candidates) == 2 and _same_channel(
            fragments[0], fragments[1], system.symbols, bonds[0], bonds[1], system.masses,
        ):
            candidates = candidates[:1]
            diagnostics.append(
                "Permutation-equivalent forward/reverse reactant identities: canonical direction selected."
            )
        if len(candidates) == 1:
            reaction.update(candidates[0])
            _validate_reaction(reaction, system.natoms)
            _validate_equivalent_paths(reaction, optimized_system)
        else:
            diagnostics.append(
                "Entrance/channel is ambiguous; select reaction.entry_side or provide fragments and bond changes."
            )
        result.update(
            ready=len(candidates) == 1, candidates=candidates,
            trace_endpoints=[t["geometry"].tolist() for t in traces],
        )
        return result
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
