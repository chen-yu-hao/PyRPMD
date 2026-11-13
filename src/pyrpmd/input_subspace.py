"""Checked opt-in subspace postprocessing for the native rate input."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from .input_schema import InputError, _integer, _unknown_fields
from .validation import finite_real

_SECTION = "rate.subspace_correction"
_FIELDS = {"enabled", "mode", "dimension", "environment", "threshold", "reuse", "path", "quantum", "rpmd",
           "curvature", "zero_point", "dvr_points", "dvr_half_width", "dvr_spacing", "hessian_displacement",
           "validity", "execution"}
_EXECUTION = {"cpus", "workers", "blas_threads", "memory_limit_mb"}
_VALIDITY = {"unconverged_path", "unstable_bath", "outside_grid", "max_outside_fraction",
             "soft_mode_frequency", "on_invalid"}
_PATH = {"source", "step", "max_steps", "spread_factor", "asymptote_tolerance", "asymptote_patience", "smoothing",
         "least_action_points", "instanton_beads", "instanton_max_iterations", "crossover_temperature"}
_QUANTUM = {"spacing_1d", "spacing_2d", "refinement", "absorber_fraction", "threshold_margin",
            "truncation_kt", "truncation_min", "padding", "dense_limit", "reference_kt", "product_drop_kt"}
_RPMD = {"windows", "walkers", "burn_in", "sweeps", "leapfrog_steps", "hmc_dt", "trajectories",
         "md_dt", "md_time", "plateau_fraction", "bootstrap", "seed"}


def _table(value, fields, name):
    if not isinstance(value, Mapping):
        raise InputError(f"{name} must be a table")
    if set(value) - fields:
        raise _unknown_fields(set(value) - fields, fields, name)
    return deepcopy(dict(value))


def _real(value, name, *, zero=False):
    return finite_real(value, name, positive=not zero, nonnegative=zero,
                       allow_strings=False, error_type=InputError)


def rate_subspace_options(config):
    """Return a validated request; absence remains byte-compatible with old inputs.

    Temperature, masses, PES, bead count and output directory are inherited
    from the rate run, never separately configurable in this section. All
    dimensional subspace numerical controls use atomic units.
    """
    rate = config.get("rate", {})
    if not isinstance(rate, Mapping) or "subspace_correction" not in rate:
        return None
    if config.get("task") != "rate":
        raise InputError(f"[{_SECTION}] requires task='rate'")
    options = _table(rate["subspace_correction"], _FIELDS, _SECTION)
    for key, default in (("enabled", False), ("reuse", True)):
        value = options.setdefault(key, default)
        if not isinstance(value, bool):
            raise InputError(f"{_SECTION}.{key} must be boolean")
    for key, default, choices in (
        ("mode", "apply", ("apply", "factor_only")),
        ("environment", "adiabatic", ("frozen", "adiabatic", "free_energy")),
        ("curvature", "none", ("none", "small")),
        ("zero_point", "harmonic", ("harmonic", "rph", "dvr")),
    ):
        value = options.setdefault(key, default)
        if not isinstance(value, str) or value not in choices:
            raise InputError(f"{_SECTION}.{key} must be {' or '.join(choices)}")
    dimension = options.setdefault("dimension", "auto")
    if not (isinstance(dimension, str) and dimension == "auto"):
        dims = dimension if isinstance(dimension, (list, tuple)) else [dimension]
        if not dims:
            raise InputError(f"{_SECTION}.dimension must not be empty")
        dims = [_integer(d, f"{_SECTION}.dimension", 1, 2) for d in dims]
        if len(dims) != len(set(dims)):
            raise InputError(f"{_SECTION}.dimension contains duplicate dimensions")
        options["dimension"] = sorted(dims) if isinstance(dimension, (list, tuple)) else dims[0]
    options["threshold"] = _real(options.get("threshold", 0.5), f"{_SECTION}.threshold")
    options["dvr_points"] = _integer(options.get("dvr_points", 81), f"{_SECTION}.dvr_points", 8)
    for key in ("dvr_half_width", "dvr_spacing", "hessian_displacement"):
        if key in options:
            options[key] = _real(options[key], f"{_SECTION}.{key}")
    # Validity gates: "reject" is the default; "allow" is a recorded override.
    validity = _table(options.get("validity", {}), _VALIDITY, f"{_SECTION}.validity")
    for key, value in validity.items():
        name = f"{_SECTION}.validity.{key}"
        if key == "on_invalid":
            if value not in ("stop", "diagnose"):
                raise InputError(f"{name} must be stop or diagnose")
        elif key in ("max_outside_fraction", "soft_mode_frequency"):
            validity[key] = _real(value, name, zero=True)
            if key == "max_outside_fraction" and validity[key] > 1:
                raise InputError(f"{name} must lie in [0, 1]")
        elif value not in ("reject", "allow"):
            raise InputError(f"{name} must be reject or allow")
    options["validity"] = validity
    # Execution resources only: never part of the physics or factor-cache identity.
    execution = _table(options.get("execution", {}), _EXECUTION, f"{_SECTION}.execution")
    for key, value in execution.items():
        name = f"{_SECTION}.execution.{key}"
        if key == "cpus" and value == "auto":
            continue
        if key == "memory_limit_mb":
            execution[key] = _real(value, name)
        else:
            execution[key] = _integer(value, name, 1)
    options["execution"] = execution
    # path.max_steps is optional: without it the continuations have no step budget.
    minima = {"max_steps": 1, "asymptote_patience": 1, "least_action_points": 2, "instanton_beads": 1,
              "instanton_max_iterations": 1, "refinement": 1, "dense_limit": 1, "windows": 4,
              "walkers": 2, "burn_in": 0, "sweeps": 2, "leapfrog_steps": 1,
              "trajectories": 2, "bootstrap": 2, "seed": 0}
    for group, fields in (("path", _PATH), ("quantum", _QUANTUM), ("rpmd", _RPMD)):
        table = _table(options.get(group, {}), fields, f"{_SECTION}.{group}")
        for key, value in table.items():
            name = f"{_SECTION}.{group}.{key}"
            if key == "source":
                if value not in ("centroid", "bead_weighted", "instanton", "least_action"):
                    raise InputError(f"{name} must be centroid, bead_weighted, instanton or least_action")
            elif key in minima:
                table[key] = _integer(value, name, minima[key])
            elif key == "spacing_2d" and isinstance(value, (list, tuple)):
                if len(value) != 2:
                    raise InputError(f"{name} must be a scalar or two spacings")
                table[key] = [_real(x, name) for x in value]
            else:
                table[key] = _real(value, name, zero=key in {"truncation_min", "smoothing", "padding"})
            if key == "absorber_fraction" and not 0 < table[key] < 0.5:
                raise InputError(f"{name} must lie in (0, 0.5)")
            if key == "plateau_fraction" and not 0 < table[key] <= 1:
                raise InputError(f"{name} must lie in (0, 1]")
        options[group] = table
    if options["enabled"]:
        pmf = rate.get("pmf", {})
        method = pmf.get("method", "umbrella") if isinstance(pmf, Mapping) else None
        if method != "umbrella" or rate.get("pmf_source") not in (None, "umbrella"):
            raise InputError(f"{_SECTION} requires locally sampled umbrella windows; "
                             "external PMFs, metadynamics and OPES have no centroid_path moments")
    return options


def normalize_rate_subspace(config):
    """Enable additive centroid statistics before any sampling is planned."""
    options = rate_subspace_options(config)
    if options is not None:
        config["rate"]["subspace_correction"] = options
        if options["enabled"]:
            config.setdefault("output", {})["centroid_path"] = True
    return options
