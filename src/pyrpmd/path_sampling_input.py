"""Strict molecular frontend for resumable transition-interface sampling.

Path sampling owns its Monte Carlo/FFS budgets.  The ordinary molecular,
potential, time and resource parsers are shared, while unrelated equilibrium
sampling controls are rejected instead of silently becoming unused options.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import sys

from .input_paths import _cv_model_paths, _plugin_paths
from .input_runtime import resolve_dynamics, resolve_output, resolve_parallel
from .input_schema import InputError, _SECTIONS, _integer, _unknown_fields
from .input_system import resolve_potential, resolve_system
from .input_tasks import validate_input_preflight
from .system import System


_ROOT = set(
    "schema_version task preset temperature beads seed system potential dynamics "
    "cvs parallel output native description path_sampling".split()
)
_OUTPUT = {"directory", "checkpoint_steps"}


def resolve_path_sampling_job(data, *, base_directory=None, input_path=None):
    """Normalize a path-sampling input without allocating a run or trajectory.

    ``resolve_path_sampling`` is also called by the resolved runner, keeping
    scientific validation consistent for TOML, Python and resume entrypoints.
    Other tasks never enter this resolver, preserving their existing identities.
    """
    from .workflows.path_sampling import resolve_path_sampling

    if not isinstance(data, Mapping) or data.get("task") != "path_sampling":
        raise InputError("resolve_path_sampling_job requires task='path_sampling'")
    raw = deepcopy(dict(data))
    unknown = set(raw) - _ROOT
    if unknown:
        raise _unknown_fields(unknown, _ROOT, "path_sampling top-level")
    schema_version = _integer(raw.get("schema_version", 1), "schema_version")
    if schema_version not in (1, 2):
        raise InputError("Supported schema_version values are 1 and 2")
    for section in ("system", "potential", "dynamics", "parallel", "output"):
        if section not in raw:
            continue
        value = raw[section]
        if section == "system" and isinstance(value, System):
            continue
        if not isinstance(value, Mapping):
            raise InputError(f"{section} must be a table")
        allowed = _OUTPUT if section == "output" else _SECTIONS[section]
        unknown = set(value) - allowed
        if unknown:
            raise _unknown_fields(unknown, allowed, section)
    if not isinstance(raw.get("path_sampling"), Mapping):
        raise InputError("path_sampling must be an explicit table")
    if not isinstance(raw.get("cvs", []), (Mapping, list, tuple)):
        raise InputError("cvs must be a table or a list of CV tables")
    validate_input_preflight(raw)
    base = Path(base_directory or ".").resolve()
    _plugin_paths(raw, base)
    _cv_model_paths(raw, base)
    # The common production preset requires an explicit sampling section.
    # Here the explicit path_sampling table supplies that role; no PIMD
    # thermostat, time budget or default is copied into the resolved job.
    runtime_input = dict(raw, sampling={})
    try:
        runtime = resolve_dynamics(runtime_input, selected=False, initial_ensemble=None)
        system, sys_spec = resolve_system(
            raw, base, runtime, initial_ensemble=None, source_pmf=False,
        )
        potential = resolve_potential(raw, base, system, beads=runtime.beads)
        if (raw["path_sampling"].get("method") == "retis"
                and raw["path_sampling"].get("exchange") == "infinite"):
            raw.setdefault("parallel", {}).setdefault("scheduling", "dynamic")
        parallel = resolve_parallel(raw, potential)
        raw.setdefault("output", {}).setdefault("checkpoint_steps", 100)
        output = resolve_output(raw, base, runtime.dt, selected=False)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        if isinstance(error, InputError):
            raise
        raise InputError(str(error)) from error
    resolved = {
        **raw,
        "schema_version": schema_version,
        "preset": runtime.preset,
        "temperature_K": runtime.temperature,
        "beta": runtime.beta,
        "beads": runtime.beads,
        "seed": runtime.seed,
        "dt_au": runtime.dt,
        "system": sys_spec,
        "potential": potential,
        "parallel": parallel,
        "output": output,
        "_input_path": input_path,
        "_base_directory": str(base),
        "_warnings": runtime.notes,
        "hamiltonian_convention": "sum_beads_beta_over_P",
    }
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    try:
        resolved["path_sampling"] = resolve_path_sampling(resolved, system)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        if isinstance(error, InputError):
            raise
        raise InputError(str(error)) from error
    return resolved, system
