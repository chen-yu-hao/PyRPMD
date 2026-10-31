"""Resolve explicit files and local plugin references relative to an input."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

from .input_schema import InputError


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
