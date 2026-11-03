"""Strict spectroscopy frontend reusing the standard molecular input grammar."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path


def _parameter_paths(value, base):
    """Resolve explicitly named model/file parameters, never arbitrary labels."""
    from .input import _path, _plugin_paths

    _plugin_paths(value, base)
    if isinstance(value, dict):
        for key, item in value.items():
            if (key in ("file", "path", "model", "weights", "checkpoint")
                    or key.endswith(("_file", "_path"))):
                if isinstance(item, str):
                    value[key] = _path(base, item)
            elif key in ("files", "paths") or key.endswith(("_files", "_paths")):
                if isinstance(item, (list, tuple)):
                    value[key] = [_path(base, p) if isinstance(p, str) else p for p in item]
            else:
                _parameter_paths(item, base)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _parameter_paths(item, base)


def resolve_spectroscopy_job(data, *, base_directory=None, input_path=None):
    """Resolve task=spectroscopy into canonical atomic units and model paths.

    Only spectroscopy/property fields are removed for temporary PIMD parsing;
    unknown standard fields remain subject to JobSpec's strict validator. No
    unsupported task-specific section is silently dropped or reinterpreted.
    """
    from .input import InputError, JobSpec, _path, _plugin_paths, _unknown_fields
    from .dynamics import normalize_integrator
    from .workflows.spectroscopy import (PROPERTY_FIELDS, SPECTROSCOPY_FIELDS,
                                         resolve_spectroscopy)

    if not isinstance(data, Mapping) or data.get("task") != "spectroscopy":
        raise InputError("resolve_spectroscopy_job requires task='spectroscopy'")
    raw = deepcopy(dict(data))
    forbidden = {
        "reaction", "collision", "outcomes", "cvs", "umbrella", "analysis", "rate",
        "recrossing", "network", "metadynamics", "opes", "adaptive_umbrella", "pcet",
        "diffusion", "site_rate", "surface", "nonadiabatic", "replica_exchange", "barostat",
        "electronic_friction", "initial_ensemble", "path_integral", "active_learning",
    }
    incompatible = forbidden.intersection(raw)
    if incompatible:
        raise InputError("spectroscopy does not support sections: " + ", ".join(sorted(incompatible)))
    for section, fields in (("spectroscopy", SPECTROSCOPY_FIELDS), ("properties", PROPERTY_FIELDS)):
        value = raw.get(section, {})
        if not isinstance(value, Mapping):
            raise InputError(f"{section} must be a table")
        if set(value) - fields:
            raise _unknown_fields(set(value) - fields, fields, section)
    spec, properties = raw.pop("spectroscopy", {}), raw.pop("properties", {})
    dynamics = raw.setdefault("dynamics", {})
    sampling = raw.setdefault("sampling", {})
    output = raw.setdefault("output", {})
    for name, table in (("dynamics", dynamics), ("sampling", sampling), ("output", output)):
        if not isinstance(table, Mapping):
            raise InputError(f"{name} must be a table")
    try:
        dynamics["integrator"] = normalize_integrator(dynamics.get("integrator", "normal-mode-bab"))
    except ValueError as error:
        raise InputError(str(error)) from error
    if dynamics["integrator"] != "normal-mode-bab":
        raise InputError("spectroscopy requires dynamics.integrator='normal-mode-bab'")
    if {"parents", "andersen_interval", "thermostat_params"}.intersection(sampling):
        raise InputError("spectroscopy sampling does not accept parents or Andersen/NHC parameters")
    sampling.setdefault("thermostat", "pile-l")
    sampling.setdefault("integrator", "baoab")
    unused_output = {"trajectory_stride", "trajectory_storage", "chunk_frames", "diagnostics", "store_configurations"}
    if unused_output.intersection(output):
        raise InputError("spectroscopy saves averaged property traces; unsupported output controls: "
                         + ", ".join(sorted(unused_output.intersection(output))))
    output.setdefault("directory", "runs/spectroscopy")
    raw["task"] = "pimd"
    base = Path(base_directory or ".").resolve()
    resolved, system = JobSpec(raw, base, input_path).resolve()
    resolved["task"] = "spectroscopy"
    resolved["spectroscopy"] = deepcopy(dict(spec))
    resolved["properties"] = deepcopy(dict(properties))
    for key in ("fqcmd_model_file", "hcmd_model_file"):
        if key in resolved["spectroscopy"]:
            model = resolved["spectroscopy"][key]
            if not isinstance(model, str) or not model.strip():
                raise InputError(f"spectroscopy.{key} must be a file path")
            resolved["spectroscopy"][key] = _path(base, model)
    _plugin_paths(resolved["properties"], base)
    if "model" in resolved["properties"]:
        if not isinstance(resolved["properties"]["model"], str) or not resolved["properties"]["model"]:
            raise InputError("properties.model must be a file path")
        resolved["properties"]["model"] = _path(base, resolved["properties"]["model"])
    _parameter_paths(resolved["properties"].get("params", {}), base)
    try:
        resolve_spectroscopy(resolved, system)
    except (ValueError, TypeError, KeyError) as error:
        raise InputError(str(error)) from error
    resolved["spectroscopy"].setdefault("method", "pa-cmd")
    return resolved, system
