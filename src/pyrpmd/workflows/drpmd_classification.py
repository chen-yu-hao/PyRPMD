"""Centroid connectivity classifiers and persistent outcome labels.

Collision assembly, centroid connectivity and flux estimators are adapted from
chen-yu-hao/RPMDrate, commit 9a44d6b378d8bf835d67e1b3b7f78b27bdc84a78,
rpmdrate/drpmd.py (MIT). Original RPMDrate copyright (c) 2012 Joshua W. Allen,
William H. Green and Yury V. Suleimanov; see THIRD_PARTY_NOTICES.md and LICENSE.
"""

from __future__ import annotations

from collections import Counter
import copy
import re

import numpy as np

from .drpmd_sampling import (
    _field, _positive, fragment_com, fragment_velocity, validate_fragments,
)

_STATUSES = ("product", "nonreactive", "unresolved", "failed")


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


def _original_classifier_methods(instance, cls, methods):
    """Only optimize the exact built-in implementation, including its hooks."""
    return type(instance) is cls and all(
        getattr(getattr(instance, name, None), "__func__", None) is function
        for name, function in methods.items()
    )


def _connectivity_pair_data(classifier):
    """Cache element-pair lookups, refreshing after public parameter changes."""
    symbols = tuple(classifier.symbols)
    cached = getattr(classifier, "_pair_data", None)
    if cached is None or cached[0] != symbols or cached[1] != classifier.cutoffs:
        pairs = [(i, j) for i in range(len(symbols)) for j in range(i + 1, len(symbols))]
        indices = np.asarray(pairs, dtype=np.intp).reshape((-1, 2))
        cutoffs = np.array([
            classifier.cutoffs[tuple(sorted((symbols[i], symbols[j])))]
            for i, j in pairs
        ], dtype=float)
        cached = (symbols, dict(classifier.cutoffs), indices, cutoffs)
        classifier._pair_data = cached
    return cached[2], cached[3]


def _centroid_bonds(classifier, centroid):
    pairs, cutoffs = _connectivity_pair_data(classifier)
    delta = centroid[pairs[:, 0]] - centroid[pairs[:, 1]]
    distances = np.linalg.norm(delta, axis=1)
    # Batched reductions can round differently from the original scalar norm.
    # Re-evaluate threshold neighbours using that exact operation so a strict
    # bond cutoff keeps its original behaviour at floating-point boundaries.
    near = np.abs(distances - cutoffs) <= (
        8 * np.finfo(float).eps * np.maximum(1.0, np.maximum(distances, np.abs(cutoffs)))
    )
    for index in np.flatnonzero(near):
        distances[index] = np.linalg.norm(delta[index])
    return {tuple(pair) for pair in pairs[distances < cutoffs].tolist()}


def _bond_components(bonds, natoms):
    adjacency = [set() for _ in range(natoms)]
    for i, j in bonds:
        adjacency[i].add(j)
        adjacency[j].add(i)
    remaining, components = set(range(natoms)), []
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
        return _centroid_bonds(self, centroid)

    def components(self, q):
        return _bond_components(self.bond_graph(q), len(self.symbols))

    def validate_reactants(self, q):
        if {frozenset(f) for f in self.components(q)} != {
            frozenset(f) for f in self.fragments
        }:
            raise ValueError(
                "Thermal sample no longer has the specified reactant connectivity"
            )

    def __call__(self, state):
        builtin = _original_classifier_methods(
            self, ConnectivityClassifier, _CONNECTIVITY_CLASSIFIER_METHODS
        ) or _original_classifier_methods(
            self, PilotConnectivityClassifier, _PILOT_CLASSIFIER_METHODS
        )
        if builtin:
            centroid = np.asarray(state.q).mean(axis=0)
            bonds = _centroid_bonds(self, centroid)
            components = _bond_components(bonds, len(self.symbols))
        else:
            # Preserve all extension hooks for subclasses and patched instances.
            components = self.components(state.q)
        if len(components) < 2:
            return None
        if builtin:
            momentum = np.asarray(state.p).mean(axis=0)
            masses = [self.masses[f].sum() for f in components]
            centers = [np.sum(centroid[f] * self.masses[f, None], axis=0) / mass
                       for f, mass in zip(components, masses)]
            velocities = [momentum[f].sum(axis=0) / mass
                          for f, mass in zip(components, masses)]
        else:
            centers = [fragment_com(state.q, self.masses, f) for f in components]
            velocities = [fragment_velocity(state.p, self.masses, f) for f in components]
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                r, v = centers[j] - centers[i], velocities[j] - velocities[i]
                if np.linalg.norm(r) < self.exit_distance or np.dot(r, v) <= 0:
                    return None
        if builtin and type(self) is PilotConnectivityClassifier:
            return self._classify_with_bonds(state, components, bonds)
        return self.classify_components(state, components)

    def native_outcome_descriptor(self):
        """Opt in to native event checks only for the unchanged explicit classifier."""
        if not _original_classifier_methods(
            self, ConnectivityClassifier, _CONNECTIVITY_CLASSIFIER_METHODS
        ):
            return None
        try:
            symbols = tuple(self.symbols)
            masses = np.asarray(self.masses, dtype=float)
            _, cutoffs = _connectivity_pair_data(self)
            signatures = [self.reactants, *self.channels]
            if (
                masses.shape != (len(symbols),)
                or not np.isfinite(masses).all() or (masses <= 0).any()
                or not np.isfinite(cutoffs).all() or (cutoffs <= 0).any()
                or not np.isfinite(self.exit_distance) or self.exit_distance <= 0
                or (cutoffs >= self.exit_distance).any()
                or len(self.reactants) != 2
                or any(len(signature) < 2 or signature == self.reactants
                       for signature in self.channels)
                or any(not isinstance(s, str) or not re.fullmatch(r"[A-Z][a-z]?", s)
                       for s in symbols)
                or any(tuple(s) != tuple(sorted(_canonical_formula(x) for x in s))
                       for s in signatures)
                or any(not isinstance(label, str) or not label for label in self.channels.values())
                or len(set(self.channels.values())) != len(self.channels)
            ):
                return None
            return {
                "kind": "connectivity", "masses": masses.tolist(),
                "symbols": list(symbols), "pair_cutoffs": cutoffs.tolist(),
                "reactants": list(self.reactants),
                "channels": [list(signature) for signature in self.channels],
                "channel_labels": list(self.channels.values()),
                "exit_distance": float(self.exit_distance),
            }
        except (AttributeError, KeyError, TypeError, ValueError):
            # Mutated/custom public parameters retain the Python behaviour.
            return None

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
        return self._classify_with_bonds(state, components)

    def _classify_with_bonds(self, state, components, bonds=None):
        signature = tuple(
            sorted(formula(self.symbols[i] for i in f) for f in components)
        )
        if signature == self.reactants:
            unchanged_atoms = {
                frozenset(f) for f in components
            } == self.reference_components
            if unchanged_atoms and (
                self.bond_graph(state.q) if bonds is None else bonds
            ) == self.reference_bonds:
                return {"status": "nonreactive", "channel": None}
            return None
        channel = self.channels.get(signature, "candidate:" + "+".join(signature))
        return {"status": "product", "channel": channel}


_CONNECTIVITY_CLASSIFIER_METHODS = {
    name: getattr(ConnectivityClassifier, name)
    for name in ("__call__", "bond_graph", "components", "classify_components", "validate_reactants")
}
_PILOT_CLASSIFIER_METHODS = {
    name: getattr(PilotConnectivityClassifier, name)
    for name in (*_CONNECTIVITY_CLASSIFIER_METHODS, "_classify_with_bonds")
}


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
