"""Explicit, separately normalized entrance/transition-state rate problems.

A transition state is a dividing-surface reference, not a unique reactive
event. Two surfaces may measure the same asymptotic flux. This module therefore
constructs independent Bennett--Chandler problems and never sums their rates.
The network currently shares one ordered atomic/isotopic identity; entrances
may have different two-fragment partitions, PESs and statistical weights.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .io import jsonable
from .propagation import scientific_config
from .system import System, read_xyz, system_from_config
from .units import parse_quantity


_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_OVERRIDES = {"dynamics", "sampling", "umbrella", "analysis", "rate", "recrossing"}
_ENTRANCE_FIELDS = {
    "id", "description", "fragments", "rinf", "electronic_weight", "potential",
    "symbols", "masses",
} | _OVERRIDES
_TS_FIELDS = {
    "id", "description", "entrance", "channel", "geometry", "transition_state",
    "geometry_unit", "forming_bonds", "breaking_bonds", "equivalent_paths",
    "symmetry_factor", "electronic_weight", "potential", "preparation",
    "symbols", "masses",
} | _OVERRIDES
_CHANNEL_FIELDS = {"id", "description", "fragments"}
_EXECUTION_FIELDS = {
    "parallel", "output", "resume", "_resume", "_input_path", "_base_directory",
    "_warnings",
}


def _table(value, name, allowed):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a table")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} fields: {', '.join(sorted(unknown))}")
    return copy.deepcopy(dict(value))


def _identifier(value, name):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{name} must be 1--64 letters, digits, underscores or hyphens")
    return value


def _records(value, name, allowed):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty array of tables")
    result = {}
    for raw in value:
        item = _table(raw, name, allowed)
        ident = _identifier(item.get("id"), f"{name}.id")
        if ident in result:
            raise ValueError(f"Duplicate {name} id {ident!r}")
        result[ident] = item
    return result


def _fragments(value, natoms, name, *, bimolecular=True):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty atom partition")
    if bimolecular and len(value) != 2:
        raise ValueError(f"{name} requires two nonempty entrance fragments")
    groups = []
    for group in value:
        if not isinstance(group, (list, tuple)) or not group:
            raise ValueError(f"{name} contains an empty/invalid fragment")
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in group):
            raise ValueError(f"{name} atom indices must be integers")
        groups.append([int(i) for i in group])
    if sorted(i for group in groups for i in group) != list(range(natoms)):
        raise ValueError(f"{name} must partition every atom exactly once")
    return groups


def _weight(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite statistical weight")
    number = float(value)
    if not np.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return number


def _check_identity(item, system, name):
    if "symbols" in item and tuple(item["symbols"]) != system.symbols:
        raise ValueError(f"{name} symbols/order differ from the global atomic identity")
    if "masses" in item:
        try:
            masses = np.array([parse_quantity(m, "mass") for m in item["masses"]])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} masses must match the global isotope masses") from exc
        if masses.shape != system.masses.shape or not np.allclose(masses, system.masses, rtol=1e-12, atol=0):
            raise ValueError(f"{name} masses differ from the global isotope masses; use a separate network")


def _path(base, value):
    p = Path(value).expanduser()
    return str((base / p).resolve()) if not p.is_absolute() else str(p.resolve())


def _system(raw, first_ts, base):
    """Read the shared identity; the first TS may provide a missing geometry."""
    from .input import _SECTIONS

    if isinstance(raw, System):
        return System.from_dict(raw.to_dict())
    spec = _table(raw, "system", _SECTIONS["system"])
    if not any(k in spec for k in ("geometry", "transition_state", "reactants")):
        source = first_ts.get("geometry", first_ts.get("transition_state"))
        if source is None:
            raise ValueError("Specify system.geometry or a geometry for each transition state")
        spec["geometry"] = copy.deepcopy(source)
        spec["geometry_unit"] = first_ts.get("geometry_unit", "angstrom" if isinstance(source, (str, Path)) else "bohr")
    for key in ("geometry", "transition_state"):
        if isinstance(spec.get(key), (str, Path)):
            spec[key] = _path(base, spec[key])
    if "reactants" in spec:
        spec["reactants"] = [_path(base, p) for p in spec["reactants"]]
    if "separation" in spec:
        spec["separation_au"] = parse_quantity(spec["separation"], "length")
    system = system_from_config(spec)
    if "symbols" in spec and tuple(spec["symbols"]) != system.symbols:
        raise ValueError("system.symbols/order disagree with the reference XYZ identity")
    return system


def _geometry(item, system, base):
    if "geometry" in item and "transition_state" in item:
        raise ValueError(f"TS {item['id']}: use geometry or transition_state, not both")
    source = item.get("geometry", item.get("transition_state"))
    if source is None:
        raise ValueError(f"TS {item['id']}: an explicit geometry is required")
    if isinstance(source, (str, Path)):
        path = _path(base, source)
        read = read_xyz(path, unit=item.get("geometry_unit", "angstrom"), masses=system.masses)
        if read.symbols != system.symbols:
            raise ValueError(f"TS {item['id']}: XYZ symbols/order differ from the global identity")
        return read.geometry, path
    if np.iscomplexobj(source):
        raise ValueError(f"TS {item['id']}: geometry must be real")
    geometry = np.asarray(source, float) * parse_quantity(f"1 {item.get('geometry_unit', 'bohr')}", "length")
    if geometry.shape != system.geometry.shape or not np.isfinite(geometry).all():
        raise ValueError(f"TS {item['id']}: geometry must be a finite [N,3] array")
    return geometry, None


def _potential(global_potential, entrance, transition_state):
    """A narrower PES specification replaces the entire wider specification."""
    result = global_potential
    scope = "global"
    for name, item in (("entrance", entrance), ("transition_state", transition_state)):
        if "potential" in item:
            result, scope = item["potential"], name
    if not isinstance(result, Mapping) or not result:
        raise ValueError(f"TS {transition_state['id']}: specify a global, entrance or TS potential")
    if "provider" not in result and "type" not in result:
        raise ValueError(f"TS {transition_state['id']}: a potential override must be complete and specify provider")
    return copy.deepcopy(dict(result)), scope


def _scientific_job(config):
    return {
        k: v for k, v in scientific_config(config).items()
        if k not in _EXECUTION_FIELDS
    }


def path_identity(config, system, *, entrance, transition_state, channel):
    value = {
        "format": "PyRPMD.network.path.v1", "entrance": entrance,
        "transition_state": transition_state, "channel": channel,
        "config": _scientific_job(config), "system": system.to_dict(),
    }
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _surface_key(config, system):
    """Catch duplicate flux surfaces, including translated/rotated TS copies.

    Only reference bond-length differences enter this coordinate. A second
    reference with the same differences is not a second channel contribution.
    This exact-definition check does not prove distinct surfaces disjoint.
    """
    from .workflows.rate import rate_coordinate

    coordinate = rate_coordinate(config, system)
    # The coordinate is min_bond(max_equivalent_TS(term)). Preserve columns:
    # sorting each TS row independently would confuse different min/max maps.
    columns = []
    for index, difference in enumerate(coordinate.reference_difference):
        reference = 0.0 if abs(float(difference)) < 5e-11 else round(float(difference), 10)
        columns.append(sorted(
            (tuple(sorted(forming[index])), tuple(sorted(breaking[index])), reference)
            for forming, breaking in coordinate.paths
        ))
    value = {
        "fragments": sorted(sorted(f) for f in config["reaction"]["fragments"]),
        "columns": sorted(columns),
        "potential": config["potential"],
        "symbols": system.symbols, "masses": system.masses,
    }
    return json.dumps(jsonable(value), sort_keys=True, allow_nan=False)


def resolve_network_job(mapping, *, base_directory=None, input_path=None):
    """Resolve a user network through the ordinary single-rate JobSpec rules.

    Called by JobSpec.resolve() for task='network'. Every child changes its task
    to 'rate' and removes 'network' before recursive validation, so it cannot
    recurse into this function. Returned jobs are JSON-serializable and retain
    explicit source references for the public dependency-hash manifest.
    """
    from .input import JobSpec, _ROOT
    from .workflows.prepare import prepare_rate

    raw = _table(mapping, "network job", set(_ROOT) | {"network"})
    if raw.get("task") != "network":
        raise ValueError("resolve_network_job requires task='network'")
    base = Path(base_directory or ".").resolve()
    specification = _table(raw.get("network"), "network", {"aggregation", "entrances", "channels", "transition_states"})
    if specification.get("aggregation", "separate") != "separate":
        raise ValueError("network.aggregation only supports 'separate': TS surfaces may overlap or measure the same flux; a verified physical partition is required before summation")
    entrances = _records(specification.get("entrances"), "network.entrances", _ENTRANCE_FIELDS)
    channels = _records(specification.get("channels"), "network.channels", _CHANNEL_FIELDS)
    transitions = _records(specification.get("transition_states"), "network.transition_states", _TS_FIELDS)
    system = _system(raw.get("system", {}), transitions[min(transitions)], base)
    if isinstance(raw.get("seed", 0), (bool, np.bool_)) or not isinstance(raw.get("seed", 0), (int, np.integer)) or raw.get("seed", 0) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(raw.get("seed", 0))
    for ident, entrance in entrances.items():
        _check_identity(entrance, system, f"entrance {ident}")
        entrance["fragments"] = _fragments(entrance.get("fragments"), system.natoms, f"entrance {ident}.fragments")
        if "rinf" not in entrance:
            raise ValueError(f"entrance {ident}: explicit rinf is required for independent normalization")
        entrance["rinf"] = parse_quantity(entrance["rinf"], "length")
        if not np.isfinite(entrance["rinf"]) or entrance["rinf"] <= 0:
            raise ValueError(f"entrance {ident}.rinf must be positive")
        entrance["electronic_weight"] = _weight(entrance.get("electronic_weight", 1), f"entrance {ident}.electronic_weight")
    for ident, channel in channels.items():
        if "fragments" in channel:
            channel["fragments"] = _fragments(channel["fragments"], system.natoms, f"channel {ident}.fragments", bimolecular=False)
    if raw.get("reaction"):
        raise ValueError("network jobs define reactions in entrances/transition_states; omit the global reaction table")
    jobs, used_surfaces, used_entrances, used_channels = [], {}, set(), set()
    first_resolved = None
    for ident in sorted(transitions):
        transition = transitions[ident]
        entry_id, channel_id = transition.get("entrance"), transition.get("channel")
        if entry_id not in entrances:
            raise ValueError(f"TS {ident}: unknown entrance {entry_id!r}")
        if channel_id not in channels:
            raise ValueError(f"TS {ident}: unknown channel {channel_id!r}")
        entrance = entrances[entry_id]
        used_entrances.add(entry_id)
        used_channels.add(channel_id)
        _check_identity(transition, system, f"TS {ident}")
        geometry, source = _geometry(transition, system, base)
        path_system = System(system.symbols, system.masses, geometry, tuple(tuple(f) for f in entrance["fragments"]), system.charge, system.multiplicity)
        potential, potential_scope = _potential(raw.get("potential"), entrance, transition)
        child = copy.deepcopy({k: v for k, v in raw.items() if k not in ("network", "system", "reaction", "potential")})
        child["task"] = "rate"
        child["system"] = path_system
        child["potential"] = potential
        # A network has one explicit outer execution layout. In its absence,
        # use serial task transport; a per-path Torch PES can still use its
        # configured device without inferring a different worker layout.
        child.setdefault("parallel", {}).setdefault("mode", "serial")
        for scope in (entrance, transition):
            for section in _OVERRIDES:
                if section in scope:
                    if not isinstance(scope[section], Mapping):
                        raise ValueError(f"TS {ident}: {section} override must be a table")
                    if section == "dynamics" and {"loop_backend", "block_steps"} & set(scope[section]):
                        raise ValueError("Network execution block/backend settings belong in the shared dynamics table")
                    child.setdefault(section, {}).update(copy.deepcopy(scope[section]))
        # Logical IDs make RNG identities invariant under input-list reordering.
        child["seed"] = int.from_bytes(hashlib.sha256(f"{seed}/{entry_id}/{ident}".encode()).digest()[:8], "little")
        reaction = {
            "fragments": entrance["fragments"], "rinf": entrance["rinf"],
            "electronic_weight": _weight(transition.get("electronic_weight", entrance["electronic_weight"]), f"TS {ident}.electronic_weight"),
            "symmetry_factor": _weight(transition.get("symmetry_factor", 1), f"TS {ident}.symmetry_factor", positive=True),
        }
        for name in ("forming_bonds", "breaking_bonds"):
            if name not in transition:
                raise ValueError(f"TS {ident}: explicit {name} are required")
            reaction[name] = copy.deepcopy(transition[name])
        for name in ("equivalent_paths", "preparation"):
            if name in transition:
                reaction[name] = copy.deepcopy(transition[name])
        if reaction.get("equivalent_paths") and reaction["symmetry_factor"] != 1:
            raise ValueError(f"TS {ident}: equivalent_paths already count exchange paths; symmetry_factor must be 1 to prevent duplicate path weights")
        child["reaction"] = reaction
        spec = JobSpec.from_mapping(child, base_directory=base)
        resolved, resolved_system = spec.resolve()
        prepared = prepare_rate(resolved, resolved_system, numerical=False)
        if not prepared["ready"]:
            raise ValueError(f"TS {ident}: explicit reaction preparation is not ready")
        # Constructing the coordinate also checks exact integer atom indices.
        signature = _surface_key(resolved, resolved_system)
        if signature in used_surfaces:
            other = used_surfaces[signature]
            raise ValueError(f"Duplicate transition-state dividing surface: {ident!r} and {other!r}; use replicas rather than counting it as another channel")
        used_surfaces[signature] = ident
        if first_resolved is None:
            first_resolved = resolved
        clean = _scientific_job(resolved)
        jobs.append({
            "id": ident, "entrance": entry_id, "channel": channel_id,
            "description": transition.get("description", ""),
            "reference_source": source, "potential_scope": potential_scope,
            "config": jsonable(clean), "system": resolved_system.to_dict(),
            "fingerprint": path_identity(clean, resolved_system, entrance=entry_id, transition_state=ident, channel=channel_id),
        })
    if set(entrances) != used_entrances:
        raise ValueError(f"Entrances without transition states: {sorted(set(entrances) - used_entrances)}")
    if set(channels) != used_channels:
        raise ValueError(f"Channels without transition states: {sorted(set(channels) - used_channels)}")
    jobs.sort(key=lambda job: (job["entrance"], job["channel"], job["id"]))
    normalized_entrances = []
    for ident in sorted(entrances):
        entrance = entrances[ident]
        fragment_masses = [float(system.masses[f].sum()) for f in entrance["fragments"]]
        normalized_entrances.append({
            "id": ident, "description": entrance.get("description", ""),
            "fragments": entrance["fragments"], "rinf_bohr": entrance["rinf"],
            "fragment_masses_au": fragment_masses,
            "reduced_mass_au": float(np.prod(fragment_masses) / sum(fragment_masses)),
            "electronic_weight": entrance["electronic_weight"],
        })
    result = copy.deepcopy(first_resolved)
    result.update(task="network", system=system.to_dict(), reaction={}, seed=seed)
    result["dynamics"] = copy.deepcopy(raw.get("dynamics", {}))
    # All effective PESs are recorded per path; a first-path PES is not a global PES.
    result["potential"] = {}
    result["network"] = {
        "schema_version": 1, "aggregation": "separate",
        "entrances": normalized_entrances,
        "channels": [channels[k] for k in sorted(channels)], "jobs": jobs,
    }
    result["_input_path"] = str(Path(input_path).resolve()) if input_path else None
    result["_base_directory"] = str(base)
    result["_warnings"] = list(dict.fromkeys(result.get("_warnings", []) + [
        "Network results are separately normalized dividing-surface estimates; TS labels do not prove disjoint physical channels, and no rates are added across surfaces or entrances."
    ]))
    validate_network(result, system)
    return result, system


def validate_network(config, system):
    """Validate a resolved network, including identities saved for restart."""
    network = config.get("network", {})
    if config.get("task") != "network" or network.get("schema_version") != 1 or network.get("aggregation") != "separate":
        raise ValueError("Expected a resolved schema-v1 separate reaction network")
    if not isinstance(network.get("jobs"), list) or not network["jobs"]:
        raise ValueError("Resolved network contains no transition-state jobs")
    entrances = {e["id"]: e for e in network["entrances"]}
    channels = {c["id"]: c for c in network["channels"]}
    if len(entrances) != len(network["entrances"]) or len(channels) != len(network["channels"]):
        raise ValueError("Duplicate entrance/channel ids in resolved network")
    for ident in entrances:
        _identifier(ident, "entrance.id")
    for ident in channels:
        _identifier(ident, "channel.id")
    ids, surfaces = set(), set()
    for job in network["jobs"]:
        ident = _identifier(job["id"], "transition_state.id")
        if ident in ids or job["entrance"] not in entrances or job["channel"] not in channels:
            raise ValueError("Invalid or duplicate resolved transition-state references")
        ids.add(ident)
        path_system = System.from_dict(job["system"])
        _check_identity(path_system.to_dict(), system, f"TS {ident}")
        _fragments(job["config"]["reaction"]["fragments"], system.natoms, f"TS {ident}.fragments")
        entrance = entrances[job["entrance"]]
        if ([list(f) for f in path_system.fragments] != entrance["fragments"]
                or job["config"]["reaction"]["fragments"] != entrance["fragments"]):
            raise ValueError(f"TS {ident} changed its entrance partition")
        expected = path_identity(job["config"], path_system, entrance=job["entrance"], transition_state=ident, channel=job["channel"])
        if job.get("fingerprint") != expected:
            raise ValueError(f"TS {ident} scientific fingerprint changed")
        reaction = job["config"]["reaction"]
        if float(reaction.get("rinf_au", reaction["rinf"])) != entrance["rinf_bohr"]:
            raise ValueError(f"TS {ident} entrance normalization changed")
        signature = _surface_key(job["config"], path_system)
        if signature in surfaces:
            raise ValueError("Duplicate transition-state dividing surface in resolved network")
        surfaces.add(signature)
    return {
        "entrances": len(entrances), "channels": len(channels),
        "transition_states": len(ids), "aggregation": "separate",
        "rate_unit": "cm^3 molecule^-1 s^-1", "total_rate_cm3_s": None,
    }


def network_jobs(config, system, output=None):
    """Yield path-local configurations with the current execution settings."""
    validate_network(config, system)
    for job in config["network"]["jobs"]:
        local = copy.deepcopy(job["config"])
        local["parallel"] = copy.deepcopy(config.get("parallel", {}))
        local["output"] = copy.deepcopy(config.get("output", {}))
        if output is not None:
            local["output"]["directory"] = str(Path(output) / "paths" / job["entrance"] / job["id"])
        local["resume"] = bool(config.get("resume", False))
        local["_base_directory"] = config.get("_base_directory")
        local["_warnings"] = []
        for name in ("loop_backend", "block_steps"):
            if name in config.get("dynamics", {}):
                local.setdefault("dynamics", {})[name] = config["dynamics"][name]
        yield job, local, System.from_dict(job["system"])
