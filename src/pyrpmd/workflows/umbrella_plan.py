"""Deterministic umbrella plans with one physical stiffness per window.

Legacy ``stiffness``/``kappa`` numbers remain atomic physical stiffnesses.
The new ``force_constant``/``force_matrix`` numbers are RPMDrate-style
temperature-normalized coefficients: k_hartree = coefficient * temperature_K.
Explicit physical unit strings retain their physical meaning.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
import copy
import hashlib
import json
import numbers
import re

import numpy as np

from ..bias import UmbrellaBias
from ..constants import KB_AU
from ..units import DIMS, parse_quantity
from .common import _dimension, checked_integer, resolve_cvs


_FORCES = ("force_constant", "force_matrix", "stiffness_au", "stiffness", "kappa")
_SELECTORS = ("centers_au", "centers", "grid", "regions", "windows")
_NEW_FIELDS = {"force_constant", "force_matrix", "regions", "windows", "coordinate"}
_REGION_FIELDS = {"id", "start", "stop", "spacing", "include_endpoints", "sampling", *_FORCES}
_WINDOW_FIELDS = {"id", "center", "sampling", *_FORCES}
_NORMALIZED_UNIT = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?:1/[tTkK]|/[tTkK]|[tTkK]\^-1)\s*$"
)


@dataclass(frozen=True)
class WindowSpec:
    """One bias state; stable ids are separate from scientific fingerprints."""

    center: np.ndarray
    stiffness: np.ndarray
    identifier: str | None = None
    source: str | None = None
    sampling: dict | None = None


def _new_interface(config, umbrella):
    return config.get("schema_version", 1) >= 2 or bool(_NEW_FIELDS.intersection(umbrella))


def _temperature(config):
    if "temperature_K" in config:
        value = parse_quantity(config["temperature_K"], "temperature")
    elif "temperature" in config:
        value = parse_quantity(config["temperature"], "temperature")
    elif "beta" in config:
        value = 1.0 / (parse_quantity(config["beta"]) * KB_AU)
    else:
        raise ValueError("temperature is required for RPMDrate-normalized force constants")
    if not np.isfinite(value) or value <= 0:
        raise ValueError("temperature must be positive for normalized force constants")
    return value


def _force_value(value, dimension, config, normalized):
    if not normalized:
        return parse_quantity(value, dimension)
    if isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)):
        return parse_quantity(value, dimension) * _temperature(config)
    if isinstance(value, str):
        match = _NORMALIZED_UNIT.fullmatch(value)
        if match:
            if dimension != DIMS["energy"]:
                raise ValueError("'1/T' shorthand requires a dimensionless CV; use explicit stiffness/K units")
            return parse_quantity(float(match[1])) * _temperature(config)
    # A dimensional string is explicit: first physical K, then physical K/Kelvin.
    try:
        return parse_quantity(value, dimension)
    except ValueError as physical_error:
        reduced_dimension = tuple(a - b for a, b in zip(dimension, DIMS["temperature"]))
        try:
            return parse_quantity(value, reduced_dimension) * _temperature(config)
        except ValueError:
            raise physical_error


def _force_choice(table, *, strict, path):
    keys = [key for key in _FORCES if key in table]
    if strict and len(keys) > 1:
        raise ValueError(f"{path} declares conflicting force-constant fields: {', '.join(keys)}")
    if not keys:
        return None
    # Historical aliases have always preferred atomic, then stiffness, then kappa.
    if not strict:
        keys = [key for key in ("stiffness_au", "stiffness", "kappa") if key in table] or keys
    return keys[0], table[keys[0]]


def _stiffness(choice, cv, config):
    if choice is None:
        raise ValueError("umbrella.force_constant or stiffness is required")
    key, raw = choice
    normalized = key in ("force_constant", "force_matrix")
    if key == "force_constant" and not isinstance(raw, (str, numbers.Real)):
        raise ValueError("force_constant is scalar; use force_matrix for CV axes or windows for per-window values")
    if key == "force_matrix" and np.asarray(raw, dtype=object).ndim != 2:
        raise ValueError("force_matrix must be a square matrix with one row per CV")
    if isinstance(raw, (str, numbers.Real)):
        raw = [[raw if i == j else 0.0 for j in range(cv.dimension)] for i in range(cv.dimension)]
    elif np.asarray(raw, dtype=object).ndim == 1:
        if len(raw) != cv.dimension:
            raise ValueError("diagonal stiffness dimension mismatch")
        raw = [[raw[i] if i == j else 0.0 for j in range(cv.dimension)] for i in range(cv.dimension)]
    if np.shape(raw) != (cv.dimension, cv.dimension):
        raise ValueError("stiffness must be scalar, diagonal, or square matrix")
    dims = [_dimension(unit) for unit in cv.units]
    result = np.empty((cv.dimension, cv.dimension))
    for i in range(cv.dimension):
        for j in range(cv.dimension):
            dimension = tuple(e - a - b for e, a, b in zip(DIMS["energy"], dims[i], dims[j]))
            result[i, j] = _force_value(raw[i][j], dimension, config, normalized)
    return result


def _ordinary_centers(table, cv, center):
    if "centers_au" in table:
        return [center(value, atomic=True) for value in table["centers_au"]]
    if "centers" in table:
        return [center(value) for value in table["centers"]]
    if "grid" in table:
        grid = table["grid"]
        grid = [grid] if isinstance(grid, Mapping) else grid
        if not isinstance(grid, (list, tuple, np.ndarray)) or len(grid) != cv.dimension:
            raise ValueError("umbrella.grid needs one axis per CV")
        axes = []
        for axis_number, axis in enumerate(grid):
            def value(raw):
                record = [0.0] * cv.dimension
                record[axis_number] = raw
                return center(record)[axis_number]
            if isinstance(axis, Mapping):
                axes.append(np.linspace(value(axis["start"]), value(axis["stop"]),
                                        checked_integer(axis["count"], "grid count")))
            else:
                axes.append([value(item) for item in axis])
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, cv.dimension)
    if all(key in table for key in ("start", "stop", "count")) and cv.dimension == 1:
        return np.linspace(center(table["start"])[0], center(table["stop"])[0],
                           checked_integer(table["count"], "umbrella.count"))[:, None]
    raise ValueError("specify umbrella.centers, grid, regions, or windows")


def _region_centers(region, center, *, path):
    if set(region) - _REGION_FIELDS:
        raise ValueError(f"Unknown {path} fields: {', '.join(sorted(set(region) - _REGION_FIELDS))}")
    if not all(key in region for key in ("start", "stop", "spacing")):
        raise ValueError(f"{path} requires start, stop, and spacing")
    lo, hi, spacing = (center(region[key])[0] for key in ("start", "stop", "spacing"))
    if not np.isfinite([lo, hi, spacing]).all() or hi <= lo or spacing <= 0:
        raise ValueError(f"{path} requires finite start < stop and positive spacing")
    ratio = (hi - lo) / spacing
    count = round(ratio)
    if count < 1 or not np.isclose(ratio, count, rtol=1e-10, atol=1e-10):
        raise ValueError(f"{path} width must be an integer multiple of spacing")
    if count > 1_000_000:
        raise ValueError(f"{path} generates more than one million windows")
    inclusion = region.get("include_endpoints", "both")
    if inclusion not in ("left", "right", "both", "neither"):
        raise ValueError(f"{path}.include_endpoints must be left, right, both, or neither")
    first = 0 if inclusion in ("left", "both") else 1
    end = count + 1 if inclusion in ("right", "both") else count
    decimal_lo, decimal_step = Decimal(str(lo)), Decimal(str(spacing))
    values = [[lo if index == 0 else hi if index == count
               else float(decimal_lo + index * decimal_step)] for index in range(first, end)]
    if not values:
        raise ValueError(f"{path} generates no window centers")
    return values, (lo, hi)


def resolve_plan(config, system):
    """Resolve centers and K together, preserving untouched legacy behavior."""
    umbrella = dict(config.get("umbrella", {}))
    if "coordinate" in umbrella:
        name = umbrella["coordinate"]
        if not isinstance(name, str) or not name:
            raise ValueError("umbrella.coordinate must name one CV")
        if "cvs" in umbrella and umbrella["cvs"] != [name]:
            raise ValueError("umbrella.coordinate conflicts with umbrella.cvs")
        umbrella["cvs"] = [name]
    cv, selected, scales = resolve_cvs({**config, "umbrella": umbrella}, system, "umbrella")
    dims = [_dimension(unit) for unit in cv.units]
    strict = _new_interface(config, umbrella)

    def center(value, atomic=False):
        if cv.dimension == 1 and isinstance(value, (str, numbers.Real)):
            value = [value]
        if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != cv.dimension:
            raise ValueError("umbrella center dimension mismatch")
        return [parse_quantity(item, dim) * (scale if not atomic and isinstance(item, numbers.Real) else 1)
                for item, dim, scale in zip(value, dims, scales)]

    sources = [key for key in _SELECTORS if key in umbrella]
    if any(key in umbrella for key in ("start", "stop", "count")):
        sources.append("start/stop/count")
    if strict and len(sources) != 1:
        raise ValueError("specify exactly one umbrella window source: centers, grid, regions, windows, or start/stop/count")
    default_force = _force_choice(umbrella, strict=strict, path="umbrella")
    records = []
    if "regions" in umbrella:
        if cv.dimension != 1:
            raise ValueError("umbrella.regions currently requires a one-dimensional CV; use grid or windows")
        regions = umbrella["regions"]
        if not isinstance(regions, list) or not regions:
            raise ValueError("umbrella.regions must be a nonempty array of tables")
        ranges = []
        ids = set()
        for index, region in enumerate(regions):
            if not isinstance(region, Mapping):
                raise ValueError("umbrella.regions entries must be tables")
            path = f"umbrella.regions[{index}]"
            source = region.get("id", f"region-{index}")
            if not isinstance(source, str) or not source or source in ids:
                raise ValueError("umbrella region ids must be unique nonempty strings")
            ids.add(source)
            values, bounds = _region_centers(region, center, path=path)
            if any(max(bounds[0], other[0]) < min(bounds[1], other[1]) for other in ranges):
                raise ValueError("umbrella region definitions overlap; make their interval boundaries explicit")
            ranges.append(bounds)
            for value in values:
                records.append((value, region, source, None))
    elif "windows" in umbrella:
        windows = umbrella["windows"]
        if not isinstance(windows, list) or not windows:
            raise ValueError("umbrella.windows must be a nonempty array of tables")
        for index, window in enumerate(windows):
            if not isinstance(window, Mapping) or "center" not in window:
                raise ValueError("umbrella.windows entries need a center")
            if set(window) - _WINDOW_FIELDS:
                raise ValueError("Unknown umbrella.windows fields: " + ", ".join(sorted(set(window) - _WINDOW_FIELDS)))
            records.append((center(window["center"]), window, "windows", window.get("id")))
    else:
        records = [(value, {}, sources[0] if sources else None, None)
                   for value in _ordinary_centers(umbrella, cv, center)]

    windows = []
    ids, previous_centers = set(), set()
    for index, (value, record, source, identifier) in enumerate(records):
        value = np.asarray(value, float)
        if value.shape != (cv.dimension,) or not np.isfinite(value).all():
            raise ValueError("invalid umbrella centers")
        # Equivalent endpoint expressions can differ by a rounding bit after
        # unit conversion; those are one physical center, not separate states.
        key = tuple(0.0 if item == 0 else float(format(item, ".14g")) for item in value)
        if strict and key in previous_centers:
            raise ValueError("duplicate umbrella center; fix endpoint inclusion or use chains_per_window")
        previous_centers.add(key)
        force = _force_choice(record, strict=True, path=f"umbrella window {index}") or default_force
        stiffness = _stiffness(force, cv, config)
        UmbrellaBias(cv, value, stiffness)
        sampling = None
        if "sampling" in record:
            from ..input_stages import normalize_sampling_stage
            sampling_dt = config.get("sampling", {}).get("dt_au", config["dt_au"])
            sampling = normalize_sampling_stage(record["sampling"], sampling_dt,
                                                path=f"umbrella window {index}.sampling")
        if strict:
            if identifier is None:
                encoded = json.dumps(key, separators=(",", ":"), allow_nan=False).encode()
                identifier = "center-" + hashlib.sha256(encoded).hexdigest()[:24]
            if not isinstance(identifier, str) or not identifier or identifier in ids:
                raise ValueError("umbrella window ids must be unique nonempty strings")
            ids.add(identifier)
        windows.append(WindowSpec(value, stiffness, identifier, source, sampling))
    if not windows:
        raise ValueError("invalid umbrella centers")

    heterogeneous = any(not np.array_equal(w.stiffness, windows[0].stiffness) for w in windows[1:])
    overrides = any(w.sampling for w in windows)
    for backend in ("adaptive_umbrella", "replica_exchange"):
        options = config.get(backend, {})
        if options and options.get("enabled", True) and (heterogeneous or overrides):
            raise ValueError(f"{backend} does not yet support heterogeneous window force constants or per-window sampling")
    return cv, selected, windows


def window_sampling(config, window):
    """Merge a partial window stage without retaining stale timing aliases."""
    sampling = copy.deepcopy(config["sampling"])
    overrides = window.sampling or {}
    if "thermostat" in overrides:
        for key in ("thermostat", "thermostat_params", "gamma_centroid", "integrator", "andersen_schedule", "andersen_interval"):
            sampling.pop(key, None)
    for key in ("equilibration", "production", "interval", "save_interval"):
        canonical = {"equilibration": "equilibration_steps", "production": "production_steps",
                     "interval": "interval_steps", "save_interval": "interval_steps"}[key]
        if canonical in overrides:
            sampling.pop(key, None)
    sampling.update(overrides)
    return sampling


def plan_preview(config, system):
    """Serializable resolved windows for preparation/run inspection."""
    cv, _, windows = resolve_plan(config, system)
    result = []
    for index, window in enumerate(windows):
        sampling = window_sampling(config, window)
        stride = sampling.get("sample_interval", sampling.get("interval_steps", 1))
        result.append({
            "index": index, "id": window.identifier, "source": window.source,
            "center_au": window.center.tolist(), "stiffness_au": window.stiffness.tolist(),
            "chains": sampling.get("replicas", 1),
            "equilibration_steps": sampling["equilibration_steps"],
            "production_steps": sampling["production_steps"], "sample_stride_steps": stride,
            "raw_samples_per_chain": sampling["production_steps"] // stride,
        })
    return {"dimension": cv.dimension, "cv_units": list(cv.units), "windows": result}
