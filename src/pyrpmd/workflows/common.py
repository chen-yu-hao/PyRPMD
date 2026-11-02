"""Shared workflow identities, atomic output and collective-variable validation.

Checkpoint schema 2 identifies the selected FES method independently of the
input spelling. The schema-1 reader remains exact so old checkpoints can be
continued without weakening any scientific compatibility check.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np

from ..checkpoint import atomic_savez
from ..cv import make_cvs
from ..io import fingerprint as _stream_fingerprint
from ..io import report_jsonable as jsonable, write_report_json as write_json
from ..validation import checked_integer
from ..units import DIMS, unit_factor
def _legacy_fingerprint(config, system, extras):
    return _stream_fingerprint(config, system, extras)


def _atomic_npz(path, **arrays):
    atomic_savez(path, kind="arrays", **arrays)


def _dimension(unit):
    return {
        "bohr": DIMS["length"],
        "radian": DIMS["angle"],
        "hartree": DIMS["energy"],
        "dimensionless": DIMS["dimensionless"],
    }[unit]
FES_DIMENSIONS = {"umbrella": (1, 2, 3), "metadynamics": (1, 2, 3), "opes": (1, 2),
                  "abf": (1, 2), "eabf": (1, 2), "steered": (1, 2)}


def fes_method(config):
    """Return the scientific method, including the two historical task aliases."""
    task = config.get("task", "fes")
    table = config.get("fes", {})
    if not isinstance(table, Mapping):
        raise ValueError("fes must be a table")
    default = task if task in ("metadynamics", "opes") else "umbrella"
    if "task" not in config and "method" not in table:
        # Direct Python workflows historically did not require a task key.
        # Their selected bias table still identifies the physical algorithm.
        selected = [name for name in FES_DIMENSIONS if name in config]
        if len(selected) > 1:
            raise ValueError("FES configuration must select exactly one method")
        if selected:
            default = selected[0]
    method = table.get("method", default)
    if not isinstance(method, str) or method not in FES_DIMENSIONS:
        raise ValueError("fes.method must be one of " + ", ".join(FES_DIMENSIONS))
    if task in ("metadynamics", "opes") and method != task:
        raise ValueError(f"task='{task}' conflicts with fes.method='{method}'")
    return method


def _selected_metadynamics_pile_map(config, method):
    """Identify mixed MetaD inputs whose historical loop ignored the map."""
    sampling = config.get("sampling", {})
    selected = sampling.get("integrator")
    if (method != "metadynamics" or sampling.get("thermostat") != "pile-l"
            or selected not in ("bcocb", "baoab")):
        return False
    from .sampling import legacy_metadynamics_sampling, middle_sampling_method

    scheme = config.get("dynamics", {}).get("integrator", "cayley-bcb")
    return (legacy_metadynamics_sampling(sampling)
            and selected != middle_sampling_method(scheme))


def _fingerprint(config, system, extras):
    """Hash physical settings, normalizing FES aliases and recording the method."""
    if config.get("task", "fes") not in ("fes", "metadynamics", "opes"):
        return _legacy_fingerprint(config, system, extras)
    method = fes_method(config)
    if set(config.get("fes", {})) - {"method"}:
        raise ValueError("FES workflows require resolved configuration; use JobSpec or "
                         "pyrpmd.run to normalize nested fes parameters first")
    canonical = dict(config)
    canonical.pop("fes", None)
    canonical["task"] = "fes"
    # A separate method tag cannot collide with any former scientific setting.
    identity = {"schema": 2, "method": method,
                "physics_fingerprint": _legacy_fingerprint(canonical, system, extras)}
    if _selected_metadynamics_pile_map(config, method):
        # Old legacy MetaD loops silently used the dynamics kernel instead.
        # Their mixed-input histories cannot seed the corrected sampling map.
        identity["metadynamics_pile_map"] = 1
    return hashlib.sha256(json.dumps(identity, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def fingerprint_matches(saved, config, system, extras):
    """Accept current identities and the exact pre-unification FES identity.

    Only input aliases/namespace metadata are migrated. Sampling, CVs, bias,
    masses, seed, integration, potential energy convention and stream identity
    stay in the hash.
    """
    saved = str(saved)
    if saved == _fingerprint(config, system, extras):
        return True
    if config.get("task", "fes") not in ("fes", "metadynamics", "opes"):
        return False
    legacy = dict(config)
    legacy.pop("fes", None)
    method = fes_method(config)
    if _selected_metadynamics_pile_map(config, method):
        return False
    legacy["task"] = "fes" if method == "umbrella" else method
    if saved == _legacy_fingerprint(legacy, system, extras):
        return True
    # Direct Python workflows historically also accepted configs without task.
    if "task" not in config:
        legacy.pop("task", None)
        return saved == _legacy_fingerprint(legacy, system, extras)
    return False


def validate_cv_dimension(cv, method):
    """Apply the same dimension check and diagnostic to every FES method."""
    if not isinstance(method, str) or method not in FES_DIMENSIONS:
        raise ValueError("fes.method must be one of " + ", ".join(FES_DIMENSIONS))
    supported = FES_DIMENSIONS[method]
    if cv.dimension not in supported:
        raise ValueError(f"{method} FES supports {supported[0]}–{supported[-1]} CV dimensions; "
                         f"got {cv.dimension}")
    return cv


def resolve_cvs(config, system, method, *, specs=None, names=None):
    """Resolve selected CVs, supported dimensions and numerical input-unit scales."""
    if specs is None:
        specs = config.get("cvs", [])
    if isinstance(specs, Mapping):
        specs = [dict(value, name=name) for name, value in specs.items()]
    if names is None:
        names = config.get(method, {}).get("cvs")
    if isinstance(names, str):
        names = [names]
    cv = validate_cv_dimension(make_cvs(specs, system, names), method)
    selected = specs if names is None else [
        next(spec for spec in specs if isinstance(spec, Mapping) and spec.get("name") == name)
        for name in names
    ]
    scales = []
    for spec, component in zip(selected, cv.cvs):
        units = spec.get("unit", spec.get("units", component.units)) if isinstance(spec, Mapping) else component.units
        if isinstance(units, str):
            units = [units] * component.dimension
        if len(units) != component.dimension:
            raise ValueError("CV input units dimension mismatch")
        scales.extend(1.0 if unit == "dimensionless" else unit_factor(unit, _dimension(canonical))
                      for unit, canonical in zip(units, component.units))
    return cv, selected, scales
