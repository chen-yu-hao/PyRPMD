"""Canonical FES method selection and checked nested input namespaces."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import warnings

from .input_schema import InputError, _SECTIONS, _unknown_fields

_FES_METHODS = ("umbrella", "metadynamics", "opes")
_FES_COMMON = ("cvs", "sampling", "analysis")


def fes_method(config):
    """Select a method with the same policy as direct workflow calls."""
    from .workflows.common import fes_method as resolve_method

    try:
        return resolve_method(config)
    except (ValueError, TypeError) as error:
        raise InputError(str(error)) from error



def normalize_fes_task(config, *, warn=True):
    """Normalize resolved aliases and reject options that would be ignored."""
    if config.get("task") not in ("fes", "metadynamics", "opes"):
        if "fes" in config:
            raise InputError("[fes] requires task='fes'")
        return config
    method, messages = _normalize_fes_input(config, warn=False)
    for section in (method, "sampling", "analysis"):
        if section not in config:
            continue
        table = config[section]
        if not isinstance(table, Mapping):
            raise InputError(f"{section} must be a table")
        allowed = _SECTIONS[section] | ({"dt_au"} if section == "sampling" else set())
        unknown = set(table) - allowed
        if unknown:
            raise _unknown_fields(unknown, allowed, section)
    if method != "umbrella":
        for section in ("adaptive_umbrella", "replica_exchange"):
            if section in config:
                raise InputError(f"{section} requires fes.method='umbrella'")
    notes = config.setdefault("_warnings", [])
    for message in messages:
        if message not in notes:
            notes.append(message)
            if warn:
                warnings.warn(message, FutureWarning, stacklevel=2)
    return config



def _normalize_fes_input(data, *, warn=True):
    """Flatten checked public subtables into the workflow configuration."""
    task = data.get("task")
    if task not in ("fes", "metadynamics", "opes"):
        if "fes" in data:
            raise InputError("[fes] requires task='fes'")
        return None, []
    method = fes_method(data)
    table = data.get("fes", {})
    allowed = {"method", *_FES_COMMON, *_FES_METHODS}
    unknown = set(table) - allowed
    if unknown:
        raise _unknown_fields(unknown, allowed, "fes")
    for other in _FES_METHODS:
        if other != method and (other in table or other in data):
            raise InputError(f"fes.method='{method}' does not accept {other} parameters")
    for common in _FES_COMMON:
        if common in table:
            if common in data:
                raise InputError(f"Use either [fes].{common} or top-level {common}, not both")
            data[common] = deepcopy(table[common])
    if method in table:
        if not isinstance(table[method], Mapping):
            raise InputError(f"fes.{method} must be a table")
        if method in data:
            raise InputError(f"Use either [fes.{method}] or [{method}], not both")
        data[method] = deepcopy(table[method])
    # Keep only the method in the resolved namespace. Existing consumers share
    # the same canonical CV, sampling and analysis sections.
    notes = []
    if task != "fes":
        notes.append(f"task='{task}' is deprecated; use task='fes' with [fes] method='{method}'.")
        if warn:
            warnings.warn(notes[-1], FutureWarning, stacklevel=3)
    data["task"] = "fes"
    data["fes"] = {"method": method}
    return method, notes
