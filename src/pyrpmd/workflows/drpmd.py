"""Thermal direct-collision RPMD, including persistent parallel parent chains.

Collision assembly, centroid connectivity and flux-estimator logic are adapted
from chen-yu-hao/RPMDrate, commit 9a44d6b378d8bf835d67e1b3b7f78b27bdc84a78,
rpmdrate/drpmd.py (MIT). Original RPMDrate copyright (c) 2012 Joshua W. Allen,
William H. Green and Yury V. Suleimanov; see THIRD_PARTY_NOTICES.md and LICENSE.
The array layout here is [bead,atom,xyz], physical forces use -grad(V), and
parallel scheduling/checkpoints are new PyRPMD code.
"""

from __future__ import annotations

from collections import Counter
from pyrpmd.propagation import loop_options, scientific_config

import copy
from dataclasses import asdict, is_dataclass
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time
import tomllib

import numpy as np

from pyrpmd.dynamics import Dynamics, State
from pyrpmd.parallel import execution_plan, make_task_potential, run_tasks, root_call

KB = 3.1668115634556e-6
RATE_AU_TO_CM3_S = (5.29177210903e-9) ** 3 / 2.4188843265857e-17
_STATUSES = ("product", "nonreactive", "unresolved", "failed")


def _field(system, name):
    return system[name] if isinstance(system, dict) else getattr(system, name)


def _positive(value, name, allow_zero=False):
    value = float(value)
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}"
        )
    return value


def _integer(value, name, minimum=1):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def validate_fragments(fragments, natoms):
    if len(fragments) != 2 or any(len(f) == 0 for f in fragments):
        raise ValueError("dRPMD requires exactly two nonempty reactant fragments")
    flat = [i for f in fragments for i in f]
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in flat):
        raise ValueError("Fragment indices must be 0-based integers")
    if sorted(flat) != list(range(natoms)):
        raise ValueError("Fragments must partition all atoms exactly once")
    return tuple(tuple(int(i) for i in f) for f in fragments)


def fragment_com(q, masses, fragment):
    ids = list(fragment)
    return (
        np.sum(np.asarray(q).mean(axis=0)[ids] * masses[ids, None], axis=0)
        / masses[ids].sum()
    )


def fragment_velocity(p, masses, fragment):
    ids = list(fragment)
    return np.asarray(p).mean(axis=0)[ids].sum(axis=0) / masses[ids].sum()


def _rotation(rng):
    quaternion = rng.normal(size=4)
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def sample_collision(
    beta, reduced_mass, bmax, separation, rng, sampling="thermal_flux", energy=None
):
    beta = _positive(beta, "beta")
    mu = _positive(reduced_mass, "reduced_mass")
    bmax = _positive(bmax, "bmax")
    separation = _positive(separation, "separation")
    if bmax >= separation:
        raise ValueError("Initial radial separation R0 must be greater than bmax")
    if sampling == "thermal_flux":
        energy = float(rng.gamma(2.0, 1.0 / beta))
    elif sampling == "maxwell":
        energy = float(rng.gamma(1.5, 1.0 / beta))
    elif sampling == "fixed_energy":
        energy = _positive(energy, "collision.energy_au")
    else:
        raise ValueError(
            "collision.sampling must be thermal_flux, maxwell or fixed_energy"
        )
    impact = float(bmax * math.sqrt(rng.random()))
    return {
        "sampling": sampling,
        "collision_energy_au": energy,
        "speed_au": math.sqrt(2.0 * energy / mu),
        "impact_parameter_au": impact,
        "azimuth_rad": float(2 * math.pi * rng.random()),
        "area_au": math.pi * bmax * bmax,
    }


def launch_collision(
    parent,
    masses,
    fragments,
    beta,
    bmax,
    separation,
    rng,
    sampling="thermal_flux",
    energy=None,
    orientation="isotropic",
):
    """Preserve thermal internal modes and replace only centroid COM motion."""
    masses = np.asarray(masses, dtype=float)
    fragments = validate_fragments(fragments, len(masses))
    if orientation not in ("isotropic", "fixed"):
        raise ValueError("collision.orientation must be isotropic or fixed")
    fm = np.array([masses[list(f)].sum() for f in fragments])
    total = fm.sum()
    mu = fm.prod() / total
    event = sample_collision(beta, mu, bmax, separation, rng, sampling, energy)
    q, p = parent.q.copy(), parent.p.copy()
    b, phi, v = event["impact_parameter_au"], event["azimuth_rad"], event["speed_au"]
    separation_vector = np.array(
        [
            b * math.cos(phi),
            b * math.sin(phi),
            math.sqrt(separation * separation - b * b),
        ]
    )
    centers = (-fm[1] / total * separation_vector, fm[0] / total * separation_vector)
    velocities = (
        np.array([0.0, 0.0, fm[1] / total * v]),
        np.array([0.0, 0.0, -fm[0] / total * v]),
    )
    for fragment, center, velocity in zip(fragments, centers, velocities):
        ids = list(fragment)
        local_q = q[:, ids] - fragment_com(q, masses, fragment)[None, None, :]
        local_p = (
            p[:, ids]
            - masses[None, ids, None]
            * fragment_velocity(p, masses, fragment)[None, None, :]
        )
        if orientation == "isotropic":
            rotation = _rotation(rng)
            local_q = local_q @ rotation.T
            local_p = local_p @ rotation.T
        q[:, ids] = local_q + center[None, None, :]
        p[:, ids] = local_p + masses[None, ids, None] * velocity[None, None, :]
    return State(q, p), event


def formula(symbols):
    counts = Counter(symbols)
    return "".join(
        s + (str(counts[s]) if counts[s] != 1 else "") for s in sorted(counts)
    )


def _formula_counts(text):
    if not isinstance(text, str) or not text:
        raise ValueError("Product formulas must be nonempty strings")
    counts, end = Counter(), 0
    for match in re.finditer(r"([A-Z][a-z]?)([0-9]*)", text):
        if match.start() != end:
            raise ValueError(f"Invalid product formula {text!r}")
        count = int(match.group(2) or "1")
        if count < 1:
            raise ValueError("Formula counts must be positive")
        counts[match.group(1)] += count
        end = match.end()
    if end != len(text):
        raise ValueError(f"Invalid product formula {text!r}")
    return counts


def _canonical_formula(text):
    counts = _formula_counts(text)
    return formula([s for s, n in counts.items() for _ in range(n)])


class ConnectivityClassifier:
    """Explicit centroid bond graph plus separation and outward motion.

    Formula-only channels cannot distinguish atom exchange or isomers. Such
    problems must supply an identity-aware Python classifier.
    """

    def __init__(self, symbols, masses, fragments, cutoffs, channels, exit_distance):
        self.symbols = tuple(symbols)
        self.masses = np.asarray(masses, dtype=float)
        self.fragments = validate_fragments(fragments, len(self.symbols))
        self.cutoffs = {}
        for pair, value in cutoffs.items():
            parts = pair.split("-")
            if len(parts) != 2:
                raise ValueError("Cutoff keys must be element pairs such as H-O")
            self.cutoffs[tuple(sorted(parts))] = _positive(value, f"bond cutoff {pair}")
        for i, s in enumerate(self.symbols):
            for other in self.symbols[i + 1 :]:
                if tuple(sorted((s, other))) not in self.cutoffs:
                    raise ValueError(f"Missing explicit bond cutoff for {s}-{other}")
        self.exit_distance = _positive(exit_distance, "exit_distance")
        if self.cutoffs and self.exit_distance <= max(self.cutoffs.values()):
            raise ValueError("Exit distance must exceed all bond cutoffs")
        self.channels = {}
        for channel, species in channels.items():
            if not isinstance(channel, str) or not channel or channel in _STATUSES:
                raise ValueError("Invalid or reserved product channel label")
            if not isinstance(species, (list, tuple)) or len(species) < 2:
                raise ValueError(
                    "Each product channel requires at least two separated species"
                )
            if sum((_formula_counts(x) for x in species), Counter()) != Counter(
                self.symbols
            ):
                raise ValueError(
                    f"Product channel {channel!r} does not conserve atom counts"
                )
            signature = tuple(sorted(_canonical_formula(x) for x in species))
            if signature in self.channels:
                raise ValueError(
                    "Duplicate channel stoichiometry requires a custom classifier"
                )
            self.channels[signature] = channel
        self.reactants = tuple(
            sorted(formula(self.symbols[i] for i in f) for f in self.fragments)
        )
        if self.reactants in self.channels:
            raise ValueError(
                "Exchange/isomer channels require an identity-aware Python classifier"
            )

    def bond_graph(self, q):
        centroid = np.asarray(q).mean(axis=0)
        bonds = set()
        for i, symbol in enumerate(self.symbols):
            for j in range(i + 1, len(self.symbols)):
                cutoff = self.cutoffs[tuple(sorted((symbol, self.symbols[j])))]
                if np.linalg.norm(centroid[i] - centroid[j]) < cutoff:
                    bonds.add((i, j))
        return bonds

    def components(self, q):
        adjacency = [set() for _ in self.symbols]
        for i, j in self.bond_graph(q):
            adjacency[i].add(j)
            adjacency[j].add(i)
        remaining, components = set(range(len(self.symbols))), []
        while remaining:
            queue, component = [min(remaining)], set()
            while queue:
                atom = queue.pop()
                if atom not in component:
                    component.add(atom)
                    queue.extend(adjacency[atom] - component)
            remaining -= component
            components.append(sorted(component))
        return components

    def validate_reactants(self, q):
        if {frozenset(f) for f in self.components(q)} != {
            frozenset(f) for f in self.fragments
        }:
            raise ValueError(
                "Thermal sample no longer has the specified reactant connectivity"
            )

    def __call__(self, state):
        components = self.components(state.q)
        if len(components) < 2:
            return None
        centers = [fragment_com(state.q, self.masses, f) for f in components]
        velocities = [fragment_velocity(state.p, self.masses, f) for f in components]
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                r, v = centers[j] - centers[i], velocities[j] - velocities[i]
                if np.linalg.norm(r) < self.exit_distance or np.dot(r, v) <= 0:
                    return None
        return self.classify_components(state, components)

    def classify_components(self, state, components):
        signature = tuple(
            sorted(formula(self.symbols[i] for i in f) for f in components)
        )
        if signature == self.reactants:
            return {"status": "nonreactive", "channel": None}
        channel = self.channels.get(signature)
        if channel is None:
            return None
        return {"status": "product", "channel": channel}


class PilotConnectivityClassifier(ConnectivityClassifier):
    """Provisional geometric labels, with no inferred chemical identities.

    New molecular compositions get candidate labels. Equal compositions with
    exchanged atom memberships or changed internal bond graphs stay unresolved.
    """

    def __init__(
        self, symbols, masses, fragments, cutoffs, channels, exit_distance, geometry
    ):
        super().__init__(symbols, masses, fragments, cutoffs, channels, exit_distance)
        q = np.asarray(geometry)[None]
        try:
            self.validate_reactants(q)
        except ValueError as exc:
            raise ValueError(
                "Inferred initial connectivity does not match the supplied reactant fragments; provide explicit outcomes.bond_cutoffs or outcomes.factory"
            ) from exc
        self.reference_bonds = self.bond_graph(q)
        self.reference_components = {frozenset(f) for f in self.fragments}

    def classify_components(self, state, components):
        signature = tuple(
            sorted(formula(self.symbols[i] for i in f) for f in components)
        )
        if signature == self.reactants:
            unchanged_atoms = {
                frozenset(f) for f in components
            } == self.reference_components
            if unchanged_atoms and self.bond_graph(state.q) == self.reference_bonds:
                return {"status": "nonreactive", "channel": None}
            return None
        channel = self.channels.get(signature, "candidate:" + "+".join(signature))
        return {"status": "product", "channel": channel}


def resolve_pilot_outcomes(system, outcomes=None, *, separation_au, preset="pilot"):
    """Resolve the two-structure pilot's missing geometric classifier explicitly.

    Input frontends should call this after resolving units, before writing the
    manifest. The returned mapping is JSON-compatible and idempotent. Explicit
    factories/cutoffs retain their existing semantics; production does not infer
    a classifier. Unsupported radii or inconsistent initial fragments fail.
    """
    c = copy.deepcopy(outcomes or {})
    if c.get("factory"):
        return c
    cutoffs = c.get("bond_cutoffs_au", c.get("bond_cutoffs_bohr"))
    if cutoffs and c.get("classifier_mode") != "pilot_connectivity":
        return c
    if preset != "pilot":
        raise ValueError(
            "Production dRPMD requires explicit outcomes.bond_cutoffs or an identity-aware outcomes.factory; automatic pilot classification is unvalidated"
        )
    if not cutoffs:
        from pyrpmd.constants import BOHR_ANGSTROM
        from pyrpmd.workflows.prepare import _COVALENT_ANG

        symbols = tuple(_field(system, "symbols"))
        missing = sorted(set(symbols) - _COVALENT_ANG.keys())
        if missing:
            raise ValueError(
                f"Pilot outcome inference has no covalent radii for {missing}; supply explicit outcomes.bond_cutoffs or outcomes.factory"
            )
        scale = 1.25
        cutoffs = {}
        for i, symbol in enumerate(symbols):
            for other in symbols[i + 1 :]:
                key = "-".join(sorted((symbol, other)))
                cutoffs[key] = (
                    scale
                    * (_COVALENT_ANG[symbol] + _COVALENT_ANG[other])
                    / BOHR_ANGSTROM
                )
        c.update(
            {
                "bond_cutoffs_au": cutoffs,
                "inferred_radii_angstrom": {
                    s: _COVALENT_ANG[s] for s in sorted(set(symbols))
                },
                "inference_scale": scale,
                "classification_assumptions": [
                    "Centroid bonds use 1.25 times the sum of tabulated covalent radii; no bond orders, charge or electronic-state identity are inferred.",
                    "Only separated, outward-moving components with persistent labels can resolve; one connected complex remains unresolved.",
                    "Changed molecular compositions receive provisional candidate labels, not validated product identities.",
                    "An unchanged composition resolves as nonreactive only if original atom memberships and the initial centroid bond graph are preserved; exchange and same-composition graph changes remain unresolved.",
                    "Chemical rate, cross section and chemical-outcome bounds are withheld until a validated explicit classifier is supplied.",
                ],
            }
        )
    c["classifier_mode"] = "pilot_connectivity"
    c["classifier_unvalidated"] = True
    c["auto_channels"] = True
    c.setdefault("exit_distance_au", _positive(separation_au, "system.separation_au"))
    classifier = PilotConnectivityClassifier(
        _field(system, "symbols"),
        _field(system, "masses"),
        _field(system, "fragments"),
        cutoffs,
        c.get("channels", {}),
        c["exit_distance_au"],
        _field(system, "geometry"),
    )
    c["reference_bonds"] = [list(pair) for pair in sorted(classifier.reference_bonds)]
    c["reactant_formulas"] = list(classifier.reactants)
    return c


def _make_classifier(config, system):
    c = config.get("outcomes", {})
    if c.get("factory"):
        from pyrpmd.potentials.python import import_object

        return import_object(c["factory"])(system=system, **c.get("params", {}))
    cutoffs = c.get("bond_cutoffs_au", c.get("bond_cutoffs_bohr"))
    if not cutoffs:
        return None
    if c.get("classifier_mode") == "pilot_connectivity":
        return PilotConnectivityClassifier(
            _field(system, "symbols"),
            _field(system, "masses"),
            _field(system, "fragments"),
            cutoffs,
            c.get("channels", {}),
            c.get("exit_distance_au", config["system"]["separation_au"]),
            _field(system, "geometry"),
        )
    return ConnectivityClassifier(
        _field(system, "symbols"),
        _field(system, "masses"),
        _field(system, "fragments"),
        cutoffs,
        c.get("channels", {}),
        c.get("exit_distance_au", config["system"]["separation_au"]),
    )


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_npz(path, metadata, **arrays):
    from ..checkpoint import atomic_savez

    atomic_savez(
        path,
        kind=metadata.get("format", "drpmd-arrays"),
        metadata=np.array(json.dumps(_jsonable(metadata), allow_nan=False)),
        **arrays,
    )


def _load_npz(path, *, expected_kind=None):
    from ..checkpoint import CheckpointError, load_checkpoint

    with load_checkpoint(path, expected_kind=expected_kind) as data:
        meta = json.loads(str(data["metadata"].item()))
        if expected_kind is not None and meta.get("format") != expected_kind:
            raise CheckpointError(
                f"Expected checkpoint payload format {expected_kind!r}, got {meta.get('format')!r}"
            )
        arrays = {key: data[key].copy() for key in data.files if key != "metadata"}
    return meta, arrays


def _dependency_fingerprints(config):
    """Hash declared local PES/model/classifier inputs, without loading models.

    Python transitive imports, system shared libraries and remote i-PI engines
    are outside this check and must be pinned by the run environment.
    """
    files = {}

    def add_file(value, base=Path(".")):
        try:
            path = Path(value)
            path = path.resolve() if path.is_absolute() else (base / path).resolve()
            exists = path.is_file()
        except (OSError, ValueError):
            return False
        if exists and str(path) not in files:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            files[str(path)] = digest.hexdigest()
        return exists

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("factory", "function", "module") and isinstance(item, str):
                    module = item.split(":", 1)[0]
                    if not add_file(module):
                        try:
                            spec = importlib.util.find_spec(module)
                        except (ImportError, ValueError, AttributeError):
                            spec = None
                        if spec is not None and spec.origin not in (
                            None,
                            "built-in",
                            "frozen",
                        ):
                            add_file(spec.origin)
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, (str, Path)):
            add_file(value)

    walk(config.get("potential", {}))
    walk(config.get("outcomes", {}))
    manifest = config.get("potential", {}).get("manifest")
    if manifest:
        path = Path(manifest).resolve()
        data = (
            json.loads(path.read_text())
            if path.suffix.lower() == ".json"
            else tomllib.loads(path.read_text())
        )
        for key in ("objects", "wrapper_sources"):
            for name in data.get(key, []):
                add_file(name, path.parent)
    return files


def _physics_hash(config, system, parents):
    config = scientific_config(config)
    scientific = {
        k: v
        for k, v in config.items()
        if k not in ("parallel", "output", "resume", "_resume")
        and not k.startswith("_")
    }
    payload = {
        "config": scientific,
        "system": _jsonable(system),
        "parents": parents,
        "local_dependencies_sha256": _dependency_fingerprints(config),
        "format": "PyRPMD.drpmd.physics.v1",
    }
    return hashlib.sha256(
        json.dumps(_jsonable(payload), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _normalize_label(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = {
            "status": (
                "nonreactive" if value in ("reactants", "nonreactive") else "product"
            ),
            "channel": None if value in ("reactants", "nonreactive") else value,
        }
    if isinstance(value, (tuple, list)) and len(value) == 2:
        value = {"status": value[0], "channel": value[1]}
    if not isinstance(value, dict) or value.get("status") not in (
        "product",
        "nonreactive",
    ):
        raise ValueError(
            "Classifier must return None or product/nonreactive plus channel"
        )
    if value["status"] == "product" and not value.get("channel"):
        raise ValueError("Product classification requires a channel")
    return (
        value["status"],
        value.get("channel") if value["status"] == "product" else None,
    )


def _restore_rng(dynamics, state):
    if state is not None:
        dynamics.rng.bit_generator.state = state


def _rng_state(dynamics):
    return _jsonable(dynamics.rng.bit_generator.state)


def _mapping_arrays(state, prefix=""):
    if not hasattr(state, "mapping_q"):
        return {}
    return {prefix + "mapping_q": state.mapping_q, prefix + "mapping_p": state.mapping_p}


def _nrpmd_trajectory(dynamics, state, classifier, config, out, trajectory_id,
                      parent_id, launch, physics_hash, resume):
    """Signed mapping-state scattering with complete electronic restart state.

    The Python step boundary is intentional: scalar-PES fused loops cannot
    propagate mapping variables. Failed physical/classifier steps roll back
    together, while checkpoint I/O failures abort the job without relabeling it.
    """
    from pyrpmd.nonadiabatic import NRPMDState

    c, outcomes = config["collision"], config.get("outcomes", {})
    na = config["nonadiabatic"]
    max_steps = _integer(c["max_steps"], "collision.max_steps")
    check = _integer(outcomes.get("check_interval", 10), "outcomes.check_interval")
    persistence = _integer(outcomes.get("persistence_steps", check * 3),
                           "outcomes.persistence_steps", 0)
    output = config.get("output", {})
    stride = _integer(output.get("trajectory_stride", max(1, max_steps // 1000)), "trajectory_stride")
    checkpoint_interval = _integer(output.get("checkpoint_steps", max(1, max_steps // 10)), "checkpoint_steps")
    tolerance = float(na["asymptotic_coupling_tolerance"])
    base = Path(out) / "trajectories" / f"trajectory-{trajectory_id:08d}"
    result_path, checkpoint = base.with_suffix(".json"), base.with_suffix(".checkpoint.npz")
    if result_path.exists():
        if not resume:
            raise FileExistsError(f"Trajectory already exists: {result_path}; use resume explicitly")
        result = json.loads(result_path.read_text())
        if result.get("physics_hash") != physics_hash or result.get("trajectory_id") != trajectory_id:
            raise ValueError("Completed trajectory does not match this restart")
        population = np.asarray(result.get("electronic_populations"), dtype=float)
        if population.shape != (na["nstates"],) or not np.isfinite(population).all():
            raise ValueError("Completed NRPMD trajectory lacks valid electronic populations")
        return result
    initial = final = drift = None
    done, candidate, candidate_since = 0, None, None
    status, channel, reason = "unresolved", None, "maximum_propagation_time"
    trace_steps, trace_q, trace_energy, trace_population = [], [], [], []
    if resume and checkpoint.exists():
        meta, arrays = _load_npz(checkpoint, expected_kind="PyRPMD.drpmd.trajectory-checkpoint.v1")
        if meta.get("physics_hash") != physics_hash or meta.get("trajectory_id") != trajectory_id:
            raise ValueError("Trajectory checkpoint has incompatible physical parameters")
        if "mapping_q" not in arrays or "mapping_p" not in arrays:
            raise ValueError("NRPMD checkpoint lacks electronic mapping state")
        state = NRPMDState(arrays["q"], arrays["p"], meta["step"], meta["time"],
                           arrays["mapping_q"], arrays["mapping_p"])
        if state.mapping_q.shape != (config["beads"], na["nstates"]):
            raise ValueError("NRPMD checkpoint mapping shape does not match configuration")
        done = meta["done"]
        initial, final, drift = meta["initial"], meta["final"], meta["drift"]
        candidate = tuple(meta["candidate"]) if meta["candidate"] is not None else None
        candidate_since = meta["candidate_since"]
        status, channel, reason = meta["status"], meta["channel"], meta["reason"]
        trace_steps = arrays["trace_steps"].tolist()
        trace_q = list(arrays["trace_centroid"])
        trace_energy = arrays["trace_energy"].tolist()
        trace_population = list(arrays["trace_populations"])
        _integer(done, "checkpoint.done", 0)
        if done > max_steps or state.step != done or not math.isclose(
                state.time, done * config["dt_au"], rel_tol=1e-11, abs_tol=1e-12):
            raise ValueError("NRPMD checkpoint has inconsistent trajectory clock")
        length = len(trace_steps)
        if (arrays["trace_centroid"].shape != (length, *state.q.shape[1:])
                or arrays["trace_energy"].shape != (length,)
                or arrays["trace_populations"].shape != (length, na["nstates"])
                or arrays["trace_steps"].dtype.kind not in "iu"
                or (length and (trace_steps[0] != 0 or trace_steps[-1] > done))
                or np.any(np.diff(trace_steps) <= 0)
                or any(not np.isfinite(arrays[key]).all() for key in
                       ("trace_centroid", "trace_energy", "trace_populations"))):
            raise ValueError("NRPMD checkpoint has invalid trajectory traces")
    else:
        try:
            if dynamics.asymptotic_coupling(state) > tolerance:
                raise ValueError("Incoming electronic channels are not asymptotically decoupled")
            initial = float(dynamics.energy(state)) / config["beads"]
            if not math.isfinite(initial):
                raise FloatingPointError("Nonfinite initial NRPMD energy")
            final, drift = initial, 0.0
            trace_steps, trace_q, trace_energy, trace_population = (
                [0], [state.q.mean(axis=0).copy()], [initial], [state.populations().copy()])
        except Exception as exc:
            status, reason = "failed", f"initial_energy: {type(exc).__name__}: {exc}"
    started = time.perf_counter()

    def save_checkpoint():
        _atomic_npz(checkpoint, {
            "format": "PyRPMD.drpmd.trajectory-checkpoint.v1", "physics_hash": physics_hash,
            "trajectory_id": trajectory_id, "parent_id": parent_id,
            "step": state.step, "time": state.time, "done": done,
            "initial": initial, "final": final, "drift": drift,
            "candidate": candidate, "candidate_since": candidate_since,
            "status": status, "channel": channel, "reason": reason, "launch": launch,
            "propagation": {"native_steps": 0, "python_steps": done},
        }, q=state.q, p=state.p, **_mapping_arrays(state),
            trace_steps=np.asarray(trace_steps, dtype=np.int64),
            trace_centroid=np.asarray(trace_q).reshape((-1, *state.q.shape[1:])),
            trace_energy=np.asarray(trace_energy),
            trace_populations=np.asarray(trace_population).reshape((-1, na["nstates"])))

    while done < max_steps and status == "unresolved":
        before = state.copy()
        previous_candidate, previous_since = candidate, candidate_since
        try:
            dynamics.step(state)
            energy = float(dynamics.energy(state)) / config["beads"]
            populations = state.populations()
            if not math.isfinite(energy) or not np.isfinite(populations).all():
                raise FloatingPointError("Nonfinite NRPMD diagnostics")
            number = done + 1
            if classifier is not None and number % check == 0:
                label = _normalize_label(classifier(state))
                if label is not None and dynamics.asymptotic_coupling(state) > tolerance:
                    label = None
                if label != candidate:
                    candidate, candidate_since = label, number if label else None
                if label is not None and number - candidate_since >= persistence:
                    status, channel = label
                    reason = "stable_separated_outgoing_decoupled_channels"
        except Exception as exc:
            state = before
            candidate, candidate_since = previous_candidate, previous_since
            status, channel, reason = "failed", None, f"{type(exc).__name__}: {exc}"
            break
        done += 1
        final, drift = energy, max(drift, abs(energy - initial))
        if done % stride == 0 or done == max_steps or status != "unresolved":
            trace_steps.append(done)
            trace_q.append(state.q.mean(axis=0).copy())
            trace_energy.append(energy)
            trace_population.append(populations.copy())
        if done % checkpoint_interval == 0 or status != "unresolved":
            save_checkpoint()
    if classifier is None and status == "unresolved":
        reason = "no_validated_product_classifier"
    save_checkpoint()
    _atomic_npz(base.with_suffix(".npz"), {
        "format": "PyRPMD.drpmd.trajectory.v1", "physics_hash": physics_hash,
        "trajectory_id": trajectory_id, "parent_id": parent_id, "launch": launch,
        "hamiltonian_convention": "sum_beads_beta_over_P",
        "units": {"q": "bohr", "p": "atomic", "time": "atomic", "energy": "hartree"},
        "electronic_estimator": "bead_mean_MMST_signed",
    }, final_q=state.q, final_p=state.p, **_mapping_arrays(state, "final_"),
        steps=np.asarray(trace_steps, dtype=np.int64),
        time_au=np.asarray(trace_steps) * config["dt_au"],
        centroid=np.asarray(trace_q).reshape((-1, *state.q.shape[1:])),
        energy_per_bead=np.asarray(trace_energy),
        electronic_populations=np.asarray(trace_population).reshape((-1, na["nstates"])))
    record = {
        "trajectory_id": trajectory_id, "parent_id": parent_id, "physics_hash": physics_hash,
        "status": status, "channel": channel, "reason": reason, "steps": done,
        "classification_unvalidated": bool(outcomes.get("classifier_unvalidated", False)),
        "time_au": done * config["dt_au"], "initial_energy_per_bead": initial,
        "final_energy_per_bead": final, "max_energy_drift_per_bead": drift,
        "launch": launch, "wall_seconds": time.perf_counter() - started,
        "propagation": {"native_steps": 0, "python_steps": done},
        "electronic_populations": state.populations().tolist(),
    }
    _atomic_json(result_path, record)
    return record


def _trajectory(
    dynamics,
    state,
    classifier,
    config,
    out,
    trajectory_id,
    parent_id,
    launch,
    physics_hash,
    resume,
):
    if config.get("nonadiabatic"):
        return _nrpmd_trajectory(dynamics, state, classifier, config, out,
                                 trajectory_id, parent_id, launch, physics_hash, resume)
    c, outcomes = config["collision"], config.get("outcomes", {})
    max_steps = _integer(c["max_steps"], "collision.max_steps")
    check = _integer(outcomes.get("check_interval", 10), "outcomes.check_interval")
    persistence = _integer(
        outcomes.get("persistence_steps", check * 3), "outcomes.persistence_steps", 0
    )
    output_config = config.get("output", {})
    stride = _integer(
        output_config.get("trajectory_stride", max(1, max_steps // 1000)),
        "trajectory_stride",
    )
    checkpoint_interval = _integer(
        output_config.get("checkpoint_steps", max(1, max_steps // 10)),
        "checkpoint_steps",
    )
    base = out / "trajectories" / f"trajectory-{trajectory_id:08d}"
    result_path, checkpoint = base.with_suffix(".json"), base.with_suffix(
        ".checkpoint.npz"
    )
    if resume and result_path.exists():
        result = json.loads(result_path.read_text())
        if (
            result.get("physics_hash") != physics_hash
            or result.get("trajectory_id") != trajectory_id
        ):
            raise ValueError("Completed trajectory does not match this restart")
        return result
    if result_path.exists() and not resume:
        raise FileExistsError(
            f"Trajectory already exists: {result_path}; use resume explicitly"
        )
    initial = final = drift = None
    candidate, candidate_since, done = None, None, 0
    status, channel, reason = "unresolved", None, "maximum_propagation_time"
    trace_steps, trace_q, trace_energy = [], [], []
    previous_propagation = {"native_steps": 0, "python_steps": 0}
    if resume and checkpoint.exists():
        meta, arrays = _load_npz(
            checkpoint, expected_kind="PyRPMD.drpmd.trajectory-checkpoint.v1"
        )
        if (
            meta.get("physics_hash") != physics_hash
            or meta.get("trajectory_id") != trajectory_id
        ):
            raise ValueError(
                "Trajectory checkpoint has incompatible physical parameters"
            )
        state = State(arrays["q"], arrays["p"], meta["step"], meta["time"])
        done = meta["done"]
        initial, final, drift = meta["initial"], meta["final"], meta["drift"]
        candidate = tuple(meta["candidate"]) if meta["candidate"] is not None else None
        candidate_since = meta["candidate_since"]
        status = meta.get("status", "unresolved")
        channel, reason = meta.get("channel"), meta.get(
            "reason", "maximum_propagation_time"
        )
        trace_steps = list(arrays["trace_steps"])
        trace_q = list(arrays["trace_centroid"])
        trace_energy = list(arrays["trace_energy"])
        if "propagation" in meta:
            previous_propagation.update(meta["propagation"])
        elif done:
            # Older checkpoints did not record which backend ran these steps.
            # Preserve them without inventing a Python/native attribution.
            previous_propagation["untracked_steps"] = done
    else:
        try:
            initial = float(dynamics.energy(state)) / config["beads"]
            if not math.isfinite(initial):
                raise FloatingPointError("Nonfinite initial total ring energy")
            final, drift = initial, 0.0
            trace_steps, trace_q, trace_energy = (
                [0],
                [state.q.mean(axis=0).copy()],
                [initial],
            )
        except Exception as exc:
            # A valid parent may launch outside a PES domain after rotation or
            # rearrangement. Preserve its q/p and count this collision as failed.
            initial = final = drift = None
            status, channel = "failed", None
            reason = f"initial_energy: {type(exc).__name__}: {exc}"
    started = time.perf_counter()
    stats_before = copy.deepcopy(dynamics.propagation_stats)

    def trajectory_propagation():
        counts = dict(previous_propagation)
        for key in ("native_steps", "python_steps"):
            counts[key] += dynamics.propagation_stats[key] - stats_before[key]
        return counts

    def save_checkpoint():
        _atomic_npz(
            checkpoint,
            {
                "format": "PyRPMD.drpmd.trajectory-checkpoint.v1",
                "physics_hash": physics_hash,
                "trajectory_id": trajectory_id,
                "parent_id": parent_id,
                "step": state.step,
                "time": state.time,
                "done": done,
                "initial": initial,
                "final": final,
                "drift": drift,
                "candidate": candidate,
                "candidate_since": candidate_since,
                "status": status,
                "channel": channel,
                "reason": reason,
                "launch": launch,
                "propagation": trajectory_propagation(),
            },
            q=state.q,
            p=state.p,
            trace_steps=np.asarray(trace_steps, dtype=np.int64),
            trace_centroid=np.asarray(trace_q).reshape((-1, *state.q.shape[1:])),
            trace_energy=np.asarray(trace_energy),
        )

    while done < max_steps and status == "unresolved":
        count = min(dynamics.block_steps, 4096, max_steps - done,
                    check - done % check, checkpoint_interval - done % checkpoint_interval)
        before = state.copy()
        start_done = done
        stats_before_block = copy.deepcopy(dynamics.propagation_stats)
        block = dynamics.propagate_block(state, count, record_states=True, raise_on_error=False)
        valid = block.completed
        failure = block.error
        previous_candidate, previous_since = candidate, candidate_since
        energy = block.energies / config["beads"]
        # The next classifier boundary is always the final step of a block.
        # An arbitrary Python classifier still sees the exact full q/p state.
        if not failure and classifier is not None and (done + valid) % check == 0:
            try:
                label = _normalize_label(classifier(state))
                block.centroids[valid - 1] = state.q.mean(axis=0)
                number = done + valid
                if label != candidate:
                    candidate, candidate_since = label, number if label else None
                if label is not None and number - candidate_since >= persistence:
                    status, channel = label
                    reason = ("stable_separated_outgoing_products" if status == "product"
                              else "stable_separated_outgoing_reactants")
            except Exception as exc:
                # A classifier failure rolls back only its step, just like the
                # original per-step workflow, not the whole completed prefix.
                valid -= 1
                if valid:
                    state = State(block.q[valid-1], block.p[valid-1],
                                  int(block.steps[valid-1]), float(block.times[valid-1]))
                else:
                    state = before
                candidate, candidate_since = previous_candidate, previous_since
                failure = f"{type(exc).__name__}: {exc}"
        # Only accepted physical steps contribute to trajectory accounting.
        # A classifier may reject the final step after the block committed it.
        if valid != block.completed:
            dynamics.propagation_stats = stats_before_block
            dynamics.propagation_stats[block.backend + "_steps"] += valid
            if block.fallback_reason:
                reasons = dynamics.propagation_stats["fallback_reasons"]
                reasons[block.fallback_reason] = reasons.get(block.fallback_reason, 0) + valid
        if valid:
            final = float(energy[valid-1])
            drift = max(drift, float(np.max(np.abs(energy[:valid] - initial))))
            numbers = start_done + np.arange(1, valid + 1)
            keep = (numbers % stride == 0) | (numbers == max_steps)
            if status != "unresolved" and not failure:
                keep[-1] = True
            selected = np.flatnonzero(keep)
            trace_steps.extend(numbers[selected].tolist())
            trace_q.extend(block.centroids[selected].copy())
            trace_energy.extend(energy[selected].tolist())
            done += valid
        if failure:
            status, channel, reason = "failed", None, failure
            break
        # I/O exceptions remain job failures, never chemical trajectory failures.
        if done % checkpoint_interval == 0 or status != "unresolved":
            save_checkpoint()
    if classifier is None and status == "unresolved":
        reason = "no_validated_product_classifier"
    save_checkpoint()
    _atomic_npz(
        base.with_suffix(".npz"),
        {
            "format": "PyRPMD.drpmd.trajectory.v1",
            "physics_hash": physics_hash,
            "trajectory_id": trajectory_id,
            "parent_id": parent_id,
            "units": {
                "q": "bohr",
                "p": "atomic",
                "time": "atomic",
                "energy": "hartree",
            },
            "hamiltonian_convention": "sum_beads_beta_over_P",
            "launch": launch,
        },
        final_q=state.q,
        final_p=state.p,
        steps=np.asarray(trace_steps, dtype=np.int64),
        time_au=np.asarray(trace_steps) * config["dt_au"],
        centroid=np.asarray(trace_q).reshape((-1, *state.q.shape[1:])),
        energy_per_bead=np.asarray(trace_energy),
    )
    record = {
        "trajectory_id": trajectory_id,
        "parent_id": parent_id,
        "physics_hash": physics_hash,
        "status": status,
        "channel": channel,
        "reason": reason,
        "steps": done,
        "classification_unvalidated": bool(
            outcomes.get("classifier_unvalidated", False)
        ),
        "time_au": done * config["dt_au"],
        "initial_energy_per_bead": initial,
        "final_energy_per_bead": final,
        "max_energy_drift_per_bead": drift,
        "launch": launch,
        "wall_seconds": time.perf_counter() - started,
        "propagation": trajectory_propagation(),
    }
    _atomic_json(result_path, record)
    return record


def _parent_task(task):
    config, system = task["config"], task["system"]
    parent_id, ids, physics_hash = (
        task["parent_id"],
        task["trajectory_ids"],
        task["physics_hash"],
    )
    out = Path(task["output"])
    masses = np.asarray(_field(system, "masses"), dtype=float)
    fragments = validate_fragments(_field(system, "fragments"), len(masses))
    beads, beta, dt = config["beads"], config["beta"], config["dt_au"]
    s = config.get("sampling", {})
    thermal_dt = float(s.get("dt_au", dt))
    equil = _integer(
        s.get("equilibration_steps", 1000), "sampling.equilibration_steps", 0
    )
    interval = _integer(
        s.get("interval_steps", s.get("sample_interval", 100)),
        "sampling.interval_steps",
    )
    andersen = _integer(s.get("andersen_interval", 10), "sampling.andersen_interval")
    options = loop_options(config)
    method = options["integrator"]
    if s.get("thermostat", "andersen") != "andersen" or s.get(
        "integrator", method + "-andersen"
    ) not in (method + "-andersen", method):
        raise ValueError(
            f"dRPMD parent sampling requires {method}-andersen with Andersen thermostat; "
            "sampling and production must use the same selected integrator"
        )
    potential = make_task_potential(config["potential"], system)
    plan = execution_plan(config.get("parallel"))
    seed = int(config.get("seed", 0))
    native = config.get("native", config.get("dynamics", {}).get("native", True))
    parent_seed = np.random.SeedSequence([seed, 1, parent_id])
    na = config.get("nonadiabatic")
    if na:
        from pyrpmd.nonadiabatic import DiagonalPotential, NRPMDDynamics, NRPMDState
        thermal_potential = DiagonalPotential(potential, na["initial_state"], na["nstates"])
    else:
        thermal_potential = potential
    dynamics = Dynamics(
        thermal_potential,
        masses,
        beta,
        beads,
        thermal_dt,
        seed=parent_seed,
        native=native,
        threads=plan.threads,
        **options,
    )
    if na:
        production = NRPMDDynamics(potential, masses, beta, beads, dt, na["nstates"],
                                   native=native, threads=plan.threads, integrator=method)
    else:
        production = Dynamics(
            potential,
            masses,
            beta,
            beads,
            dt,
            seed=np.random.SeedSequence([seed, 3, parent_id]),
            native=native,
            threads=plan.threads,
            **options,
        )
    classifier = _make_classifier(config, system)
    parent_path = out / "checkpoints" / f"parent-{parent_id:06d}.npz"
    resume = bool(config.get("resume", config.get("_resume", False)))
    geometry = np.asarray(_field(system, "geometry"), dtype=float)
    parent = State(np.repeat(geometry[None], beads, axis=0), dynamics.sample_momenta())
    burn_done, next_index, pending, records = 0, 0, None, []
    if parent_path.exists():
        if not resume:
            raise FileExistsError(
                f"Parent checkpoint exists: {parent_path}; use resume explicitly"
            )
        meta, arrays = _load_npz(
            parent_path, expected_kind="PyRPMD.drpmd.parent-checkpoint.v1"
        )
        if (
            meta.get("physics_hash") != physics_hash
            or meta.get("trajectory_ids") != ids
        ):
            raise ValueError(
                "Parent checkpoint does not match this job or parent allocation"
            )
        parent = State(arrays["q"], arrays["p"], meta["step"], meta["time"])
        _restore_rng(dynamics, meta["rng"])
        burn_done, next_index, pending, records = (
            meta["burn_done"],
            meta["next_index"],
            meta["pending"],
            meta["records"],
        )
    checkpoint_steps = _integer(
        config.get("output", {}).get("checkpoint_steps", 1000), "checkpoint_steps"
    )
    thermal_seconds = 0.0

    def save_parent():
        _atomic_npz(
            parent_path,
            {
                "format": "PyRPMD.drpmd.parent-checkpoint.v1",
                "physics_hash": physics_hash,
                "parent_id": parent_id,
                "trajectory_ids": ids,
                "step": parent.step,
                "time": parent.time,
                "rng": _rng_state(dynamics),
                "burn_done": burn_done,
                "next_index": next_index,
                "pending": pending,
                "records": records,
                "hamiltonian_convention": "sum_beads_beta_over_P",
            },
            q=parent.q,
            p=parent.p,
            masses=masses,
        )

    try:
        while burn_done < equil:
            count = min(checkpoint_steps, equil - burn_done)
            start = time.perf_counter()
            dynamics.thermalize(
                parent, count, interval=andersen, fragments=fragments, resample=False
            )
            thermal_seconds += time.perf_counter() - start
            burn_done += count
            save_parent()
        for index in range(next_index, len(ids)):
            trajectory_id = ids[index]
            if pending != trajectory_id:
                start = time.perf_counter()
                dynamics.thermalize(
                    parent,
                    interval,
                    interval=andersen,
                    fragments=fragments,
                    resample=False,
                )
                thermal_seconds += time.perf_counter() - start
                if classifier is not None and hasattr(classifier, "validate_reactants"):
                    classifier.validate_reactants(parent.q)
                pending = trajectory_id
                save_parent()
            launch_rng = np.random.default_rng(
                np.random.SeedSequence([seed, 2, trajectory_id])
            )
            child, launch = launch_collision(
                parent,
                masses,
                fragments,
                beta,
                config["collision"]["bmax_au"],
                config["system"]["separation_au"],
                launch_rng,
                sampling=config["collision"].get("sampling", "thermal_flux"),
                energy=config["collision"].get("energy_au"),
                orientation=config["collision"].get("orientation", "isotropic"),
            )
            if na:
                mapping_rng = np.random.default_rng(np.random.SeedSequence([seed, 4, trajectory_id]))
                child = NRPMDState.from_nuclear(child, na["nstates"], na["initial_state"], mapping_rng)
            _atomic_npz(
                out / "states" / f"initial-{trajectory_id:08d}.npz",
                {
                    "format": "PyRPMD.drpmd.launch.v1",
                    "physics_hash": physics_hash,
                    "trajectory_id": trajectory_id,
                    "parent_id": parent_id,
                    "parent_step": parent.step,
                    "parent_time_au": parent.time,
                    "launch": launch,
                    "beta": beta,
                    "beads": beads,
                    "fragments": fragments,
                    "symbols": _field(system, "symbols"),
                },
                thermal_q=parent.q,
                thermal_p=parent.p,
                q=child.q,
                p=child.p,
                **_mapping_arrays(child),
                masses=masses,
            )
            record = _trajectory(
                production,
                child,
                classifier,
                config,
                out,
                trajectory_id,
                parent_id,
                launch,
                physics_hash,
                resume,
            )
            record["parent_sample_step"] = parent.step
            records.append(record)
            next_index, pending = index + 1, None
            save_parent()
        return {
            "parent_id": parent_id,
            "records": records,
            "thermal_steps": parent.step,
            "thermal_seconds": thermal_seconds,
        }
    finally:
        if hasattr(potential, "close"):
            potential.close()


def _wilson(successes, n, z=1.959963984540054):
    p = successes / n
    denominator = 1 + z * z / n
    middle = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, middle - half), min(1.0, middle + half)]


def _electronic_summary(records, config, weights, channels, allowed):
    """Joint signed estimates; one parent draw is shared by every output column."""
    na = config["nonadiabatic"]
    if set(channels) & {"total_reaction", "nonreactive"}:
        raise ValueError("NRPMD product channel names total_reaction and nonreactive are reserved")
    labels = na.get("state_labels", [str(i) for i in range(na["nstates"])])
    population = np.asarray([r["electronic_populations"] for r in records], dtype=float)
    if population.shape != (len(records), na["nstates"]) or not np.isfinite(population).all():
        raise ValueError("NRPMD results have missing or invalid electronic populations")
    norm_error = float(np.max(np.abs(population.sum(axis=1) - 1.0)))
    if norm_error > 1e-8:
        raise ValueError("NRPMD electronic populations do not sum to one")
    indicators = {
        "total_reaction": np.array([r["status"] == "product" for r in records]),
        "nonreactive": np.array([r["status"] == "nonreactive" for r in records]),
    }
    indicators.update({name: np.array([r["status"] == "product" and r["channel"] == name
                                      for r in records]) for name in channels})
    columns, values, entries = [], [], {}
    fixed = config["collision"].get("sampling", "thermal_flux") == "fixed_energy"
    for channel, indicator in indicators.items():
        joint = weights[:, None] * indicator[:, None] * population
        chemical = float(np.mean(weights * indicator))
        means = joint.mean(axis=0)
        entries[channel] = {
            "signed_observed_contributions": means.tolist(),
            "rate_cm3_molecule_s": means.tolist() if allowed and not fixed else None,
            "cross_section_bohr2": means.tolist() if allowed and fixed else None,
            "parent_cluster_bootstrap95": None,
            "chemical_observed_contribution": chemical,
            "electronic_sum_error": float(means.sum() - chemical),
        }
        columns.extend([[channel, label] for label in labels])
        values.append(joint)
    matrix = np.concatenate(values, axis=1)
    covariance = None
    parents = sorted({r["parent_id"] for r in records})
    if allowed and len(parents) >= 2:
        groups = [np.array([i for i, r in enumerate(records) if r["parent_id"] == parent])
                  for parent in parents]
        rng = np.random.default_rng(np.random.SeedSequence([int(config.get("seed", 0)), 7]))
        draws = np.empty((400, matrix.shape[1]))
        for i in range(len(draws)):
            idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            draws[i] = matrix[idx].mean(axis=0)
        covariance = np.atleast_2d(np.cov(draws, rowvar=False, ddof=1)).tolist()
        bounds = np.quantile(draws, [0.025, 0.975], axis=0).T
        for i, entry in enumerate(entries.values()):
            entry["parent_cluster_bootstrap95"] = bounds[i * len(labels):(i+1) * len(labels)].tolist()
    return {
        "method": "nrpmd", "experimental": True, "physically_validated": False,
        "initial_state": na["initial_state"], "state_labels": labels,
        "estimator": "bead_mean_MMST_signed", "population_sum_max_error": norm_error,
        "final_estimates_available": bool(allowed),
        "channels": entries, "covariance_columns": columns,
        "parent_cluster_bootstrap_covariance": covariance,
        "units": "bohr^2" if fixed else "cm^3 molecule^-1 s^-1",
        "note": "Signed mapping populations are not probabilities and are never clipped. Observed contributions use all launched trajectories in the denominator and are diagnostics only when unresolved/failed trajectories or unvalidated cutoffs/classification withhold final estimates. No binomial interval or probability bound applies. This is specified-incident-state scattering, not an electronic thermal-equilibrium rate.",
    }


def summarize(records, config, system):
    n = len(records)
    if not n:
        raise ValueError("Cannot summarize zero trajectories")
    counts = Counter(r["status"] for r in records)
    if set(counts) - set(_STATUSES):
        raise ValueError("Unrecognized trajectory status")
    masses = np.asarray(_field(system, "masses"), dtype=float)
    fragments = validate_fragments(_field(system, "fragments"), len(masses))
    fm = np.array([masses[list(f)].sum() for f in fragments])
    mu, beta = fm.prod() / fm.sum(), config["beta"]
    area = math.pi * config["collision"]["bmax_au"] ** 2
    sampling = config["collision"].get("sampling", "thermal_flux")
    electronic_weight = float(config.get("reaction", {}).get("electronic_weight", 1.0))
    if electronic_weight < 0 or not math.isfinite(electronic_weight):
        raise ValueError("Electronic weight must be finite and nonnegative")
    if sampling == "thermal_flux":
        weights = np.full(
            n,
            area
            * math.sqrt(8 / (math.pi * beta * mu))
            * RATE_AU_TO_CM3_S
            * electronic_weight,
        )
    elif sampling == "maxwell":
        weights = np.array(
            [
                area * r["launch"]["speed_au"] * RATE_AU_TO_CM3_S * electronic_weight
                for r in records
            ]
        )
    else:
        weights = np.full(n, area)
    unknown = np.array([r["status"] in ("unresolved", "failed") for r in records])
    outcomes = config.get("outcomes", {})
    classifier_unvalidated = bool(outcomes.get("classifier_unvalidated", False))
    cutoff_unvalidated = bool(
        config.get("collision", {}).get("bmax_unvalidated", False)
    )
    rate_allowed = (
        sampling != "fixed_energy"
        and not cutoff_unvalidated
        and not classifier_unvalidated
    )
    channels = sorted(
        set(config.get("outcomes", {}).get("channels", {}))
        | {r["channel"] for r in records if r["status"] == "product"}
    )
    parent_ids = sorted({r["parent_id"] for r in records})

    def estimate(indicator):
        indicator = np.asarray(indicator, dtype=bool)
        values = weights * indicator
        lower, upper = float(values.mean()), float((values + weights * unknown).mean())
        point = lower if not unknown.any() and rate_allowed else None
        entry = {
            "count": int(indicator.sum()),
            "rate_cm3_molecule_s": point,
            "event_fraction": float(indicator.mean()),
            "candidate_only": classifier_unvalidated,
            "observed_lower_cm3_molecule_s": (
                lower
                if sampling != "fixed_energy" and not classifier_unvalidated
                else None
            ),
            "possible_upper_cm3_molecule_s": (
                upper
                if sampling != "fixed_energy" and not classifier_unvalidated
                else None
            ),
            "cross_section_bohr2": (
                lower
                if sampling == "fixed_energy"
                and not unknown.any()
                and not classifier_unvalidated
                and not cutoff_unvalidated
                else None
            ),
            "iid_wilson95_cm3_molecule_s": None,
            "parent_cluster_bootstrap95": None,
        }
        if sampling == "thermal_flux" and not unknown.any() and rate_allowed:
            entry["iid_wilson95_cm3_molecule_s"] = [
                float(weights[0] * p) for p in _wilson(int(indicator.sum()), n)
            ]
        # Parent-level resampling preserves dependence caused by reusing a chain.
        if len(parent_ids) >= 2 and not unknown.any() and rate_allowed:
            rng = np.random.default_rng(
                np.random.SeedSequence([int(config.get("seed", 0)), 7])
            )
            groups = [
                np.array([i for i, r in enumerate(records) if r["parent_id"] == parent])
                for parent in parent_ids
            ]
            draws = []
            for _ in range(400):
                idx = np.concatenate(
                    [groups[i] for i in rng.integers(0, len(groups), len(groups))]
                )
                draws.append(float(values[idx].mean()))
            entry["parent_cluster_bootstrap95"] = [
                float(x) for x in np.quantile(draws, [0.025, 0.975])
            ]
        return entry

    summary = {
        "format": "PyRPMD.drpmd.summary.v1",
        "task": "drpmd",
        "trajectories": n,
        "counts": {s: counts[s] for s in _STATUSES},
        "parents": len(parent_ids),
        "all_trajectories_resolved": not bool(unknown.any()),
        "temperature_K": config["temperature_K"],
        "beads": config["beads"],
        "sampling": sampling,
        "electronic_weight": electronic_weight,
        "classification_unvalidated": classifier_unvalidated,
        "classification": {
            "mode": outcomes.get(
                "classifier_mode",
                (
                    "python_factory"
                    if outcomes.get("factory")
                    else "explicit_connectivity"
                ),
            ),
            "unvalidated": classifier_unvalidated,
            "bond_cutoffs_bohr": outcomes.get(
                "bond_cutoffs_au", outcomes.get("bond_cutoffs_bohr", {})
            ),
            "inferred_radii_angstrom": outcomes.get("inferred_radii_angstrom", {}),
            "inference_scale": outcomes.get("inference_scale"),
            "reference_bonds": outcomes.get("reference_bonds"),
            "reactant_formulas": outcomes.get("reactant_formulas"),
            "assumptions": outcomes.get("classification_assumptions", []),
        },
        "total_reaction": estimate([r["status"] == "product" for r in records]),
        "channels": {
            channel: estimate(
                [r["status"] == "product" and r["channel"] == channel for r in records]
            )
            for channel in channels
        },
        "uncertainty_note": "Outcome bounds are not confidence limits. IID Wilson intervals ignore parent-chain dependence; parent-cluster bootstrap requires enough independent chains and does not establish numerical or cutoff convergence.",
        "convergence_note": "Check timestep, bead count, thermalization, sample autocorrelation, R0, bmax, maximum time and classification. Fixed-energy data are not a thermal rate.",
    }

    if config.get("nonadiabatic"):
        summary["nonadiabatic"] = _electronic_summary(
            records, config, weights, channels,
            not unknown.any() and not cutoff_unvalidated and not classifier_unvalidated)
    return summary


def _prepare_output(output):
    out = Path(output)
    for directory in ("checkpoints", "states", "trajectories", "results"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    return str(out.resolve())


def _write_summary(results, config, system, output, expected):
    records = [record for parent in results for record in parent["records"]]
    records.sort(key=lambda record: record["trajectory_id"])
    if [r["trajectory_id"] for r in records] != list(range(expected)):
        raise RuntimeError("Trajectory ledger has missing or duplicate global IDs")
    summary = summarize(records, config, system)
    summary["parent_thermal_steps"] = {
        str(p["parent_id"]): p["thermal_steps"] for p in results
    }
    summary["records"] = records
    out = Path(output)
    _atomic_json(out / "results" / "summary.json", summary)
    with (out / "results" / "channels.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "channel",
                "count",
                "rate_cm3_molecule_s",
                "observed_lower_cm3_molecule_s",
                "possible_upper_cm3_molecule_s",
            ]
        )
        for channel, entry in {
            "total_reaction": summary["total_reaction"],
            **summary["channels"],
        }.items():
            writer.writerow(
                [
                    channel,
                    entry["count"],
                    entry["rate_cm3_molecule_s"],
                    entry["observed_lower_cm3_molecule_s"],
                    entry["possible_upper_cm3_molecule_s"],
                ]
            )
    if "nonadiabatic" in summary:
        with (out / "results" / "electronic_channels.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["chemical_channel", "electronic_state", "signed_observed_contribution",
                             "rate_cm3_molecule_s", "cross_section_bohr2"])
            na = summary["nonadiabatic"]
            for channel, entry in na["channels"].items():
                for i, label in enumerate(na["state_labels"]):
                    writer.writerow([channel, label, entry["signed_observed_contributions"][i],
                                     entry["rate_cm3_molecule_s"][i] if entry["rate_cm3_molecule_s"] is not None else None,
                                     entry["cross_section_bohr2"][i] if entry["cross_section_bohr2"] is not None else None])
    return summary


def run_drpmd(config, system, output, executor=None):
    """Run N total collisions from persistent independent thermal parent chains.

    Each parent gets N//n collisions plus one for the first N%n parents. A parent
    equilibrates once, advances by interval_steps before every child, and is
    never overwritten by the departing collision trajectory. All ranks call this
    function in MPI mode; its final summary is broadcast to all ranks.
    """
    boundary = system if isinstance(system, dict) else system.to_dict()
    from ..geometry import validate_cell_pbc
    _, pbc = validate_cell_pbc(boundary.get("cell"), getattr(system, "pbc", boundary.get("pbc")))
    if pbc.any() or boundary.get("fixed_atoms"):
        raise ValueError("Gas-phase dRPMD does not support periodic/fixed-atom systems; use a surface workflow")
    config = copy.deepcopy(config)
    if ("nonadiabatic" in config
            or config.get("potential", {}).get("provider", config.get("potential", {}).get("type")) == "diabatic_python"):
        from pyrpmd.input import validate_nonadiabatic
        config.setdefault("task", "drpmd")
        validate_nonadiabatic(config, system=system)
        if set(config.get("outcomes", {}).get("channels", {})) & {"total_reaction", "nonreactive"}:
            raise ValueError("NRPMD product channel names total_reaction and nonreactive are reserved")
    config["outcomes"] = resolve_pilot_outcomes(
        system,
        config.get("outcomes"),
        separation_au=config["system"]["separation_au"],
        preset=config.get("preset", "pilot"),
    )
    n = _integer(config["collision"]["trajectories"], "collision.trajectories")
    _positive(config["collision"]["bmax_au"], "collision.bmax_au")
    if config["collision"].get("impact_sampling", "uniform_area") != "uniform_area":
        raise ValueError("Only uniform-area impact sampling is implemented")
    plan = execution_plan(config.get("parallel"))
    default_parents = (
        plan.trajectory_groups if plan.backend == "mpi" else plan.trajectory_workers
    )
    parents = min(
        n,
        _integer(
            config.get("sampling", {}).get("parents", default_parents),
            "sampling.parents",
        ),
    )
    physics_hash = root_call(_physics_hash, config, system, parents)
    directory = root_call(_prepare_output, output)
    tasks, start = [], 0
    for parent_id in range(parents):
        count = n // parents + (parent_id < n % parents)
        tasks.append(
            {
                "config": config,
                "system": system,
                "output": directory,
                "parent_id": parent_id,
                "trajectory_ids": list(range(start, start + count)),
                "physics_hash": physics_hash,
            }
        )
        start += count
    results = (
        executor.map(_parent_task, tasks)
        if executor is not None
        else run_tasks(_parent_task, tasks, config.get("parallel"))
    )
    return root_call(_write_summary, results, config, system, directory, n)
