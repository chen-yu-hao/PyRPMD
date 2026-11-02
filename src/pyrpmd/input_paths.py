"""Resolve explicit files and local plugin references relative to an input."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

from .input_schema import InputError, _integer, _unknown_fields
from .potentials import _potential_spec_fields


def _path(base, value):
    return (
        str((base / Path(value)).resolve())
        if not Path(value).is_absolute()
        else str(Path(value).resolve())
    )


def _plugin_paths(value, base):
    """Resolve file-based plugins without changing normal module:object imports."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("factory", "module", "function", "initializer") and isinstance(
                item, str
            ):
                module, separator, attribute = item.partition(":")
                if module.endswith((".py", ".so")):
                    value[key] = _path(base, module) + (
                        separator + attribute if separator else ""
                    )
            else:
                _plugin_paths(item, base)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _plugin_paths(item, base)


def _initial_ensemble_spec(section, base):
    """Read references relative to the input and retain their canonical data."""
    from .initial_states import normalize_initial_ensemble

    section = deepcopy(section)
    fragments = section.get("fragments", ())
    if isinstance(fragments, (list, tuple)):
        for fragment in fragments:
            if isinstance(fragment, Mapping):
                for key in ("hessian_file", "modes_file"):
                    if key in fragment:
                        if not isinstance(fragment[key], (str, Path)) or not str(fragment[key]):
                            raise InputError(f"initial_ensemble {key} must be a file path")
                        fragment[key] = _path(base, fragment[key])
    try:
        return normalize_initial_ensemble(section)
    except (ValueError, TypeError, KeyError, OSError) as error:
        raise InputError(str(error)) from error


def _nested_potential_paths(spec, base, nbeads):
    """Resolve model files at every level of a committee/delta specification."""
    if not isinstance(spec, dict):
        raise InputError("nested potential specifications must be tables")
    kind = spec.get("provider", spec.get("type", "builtin"))
    allowed = _potential_spec_fields(kind)
    if set(spec) - allowed:
        raise _unknown_fields(set(spec) - allowed, allowed, "nested potential")
    if kind == "committee":
        members = spec.get("members")
        if not isinstance(members, list) or len(members) < 2:
            raise InputError("committee requires at least two potential.members")
        for member in members:
            _nested_potential_paths(member, base, nbeads)
    elif kind == "delta":
        for key in ("reference", "correction"):
            _nested_potential_paths(spec.get(key), base, nbeads)
        if "contracted_beads" in spec:
            pc = _integer(spec["contracted_beads"], "contracted_beads")
            if pc > nbeads:
                raise InputError("contracted_beads cannot exceed beads")
            if "nbeads" in spec and spec["nbeads"] != nbeads:
                raise InputError("delta potential nbeads must match the input bead count")
            spec["nbeads"] = nbeads
    else:
        for key in ("library", "manifest", "weights"):
            if key in spec:
                spec[key] = _path(base, spec[key])
        if "model" in spec and kind not in ("builtin", "fairchem"):
            spec["model"] = _path(base, spec["model"])
        elif "model" in spec and kind == "fairchem":
            model = spec["model"]
            if isinstance(model, str) and (model.startswith((".", "~", "/"))
                    or Path(model).suffix.lower() in (".pt", ".pth", ".ckpt", ".bin")
                    or (base / model).is_file()):
                spec["model"] = _path(base, model)
        for key in ("model", "weights", "reference"):
            if isinstance(spec.get("params", {}).get(key), str):
                spec["params"][key] = _path(base, spec["params"][key])


def _cv_model_paths(config, base):
    specs = config.get("cvs", [])
    if isinstance(specs, dict):
        specs = specs.values()
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        for key in ("model", "reference", "references"):
            if isinstance(spec.get(key), str):
                spec[key] = _path(base, spec[key])
