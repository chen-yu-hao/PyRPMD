"""One saved free-energy surface contract for every sampling method.

The array archive is authoritative and contains its own metadata.  JSON and CSV
are regenerable views.  Legacy umbrella, MetaD and OPES archives are adapted at
this boundary, so consumers do not need to infer each producer's file format.
Replica spread remains explicitly descriptive; it is never relabelled as a
standard error or divided by the square root of the number of histories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from ..checkpoint import atomic_savez, load_checkpoint
from ..io import write_json


SCHEMA = "pyrpmd.fes-result"
SCHEMA_VERSION = 1
METHODS = ("umbrella", "metadynamics", "opes")
UNCERTAINTY_KINDS = ("block_bootstrap", "replica_spread")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _scalar(value, name):
    value = np.asarray(value)
    if value.shape != ():
        raise ValueError(f"FES {name} must be scalar")
    return value.item()


def _declared_method(metadata):
    """Return a declared method without guessing from the generic FES task."""
    table = metadata.get("fes", {})
    if not isinstance(table, Mapping):
        raise ValueError("FES method metadata must be a table")
    if "method" in table and table["method"] not in METHODS:
        raise ValueError("FES method must be umbrella, metadynamics or opes")
    candidates = [value for value in (table.get("method"), metadata.get("method"),
                                      metadata.get("task")) if value in METHODS]
    if len(set(candidates)) > 1:
        raise ValueError("FES method contradicts the declared task or method metadata")
    return candidates[0] if candidates else None


def fes_method(metadata):
    """Resolve canonical/legacy method identity, rejecting conflicting aliases."""
    return _declared_method(metadata) or "umbrella"


def _measure(metadata):
    candidates = [metadata[key] for key in ("measure", "sampling_measure")
                  if metadata.get(key) is not None]
    if candidates and any(value != candidates[0] for value in candidates[1:]):
        raise ValueError("FES measure contradicts the declared sampling_measure")
    return candidates[0] if candidates else None


def estimate_fes_uncertainty(estimates, *, kind, mask, independent=True):
    """Return a grid of bootstrap errors or descriptive replica spread.

    Estimates must already share a physical measure, grid and reference gauge.
    Bootstrap inputs are complete block-bootstrap reconstruction draws; replica
    inputs are independently reconstructed bias histories.  A single history,
    correlated shared-bias walkers, and bins with fewer than two valid draws
    have unavailable (NaN) uncertainty.
    """
    if kind not in UNCERTAINTY_KINDS:
        raise ValueError("FES uncertainty kind must be block_bootstrap or replica_spread")
    mask = np.asarray(mask, bool)
    values = np.asarray(estimates, float)
    if values.ndim != mask.ndim + 1 or values.shape[1:] != mask.shape:
        raise ValueError("FES uncertainty estimates must have shape [replicate, *grid]")
    error = np.full(mask.shape, np.nan)
    if independent and len(values) >= 2:
        if kind == "replica_spread":
            # Every contributing replica must support a displayed grid cell.
            error = values.std(axis=0, ddof=1)
        else:
            valid = np.isfinite(values).sum(axis=0)
            means = np.divide(np.nansum(values, axis=0), valid,
                              out=np.zeros(mask.shape), where=valid > 0)
            square = np.nansum((values - means) ** 2, axis=0)
            np.sqrt(np.divide(square, valid - 1, out=error, where=valid > 1),
                    out=error, where=valid > 1)
    error[mask] = np.nan
    return error


@dataclass
class FESResult:
    """A validated grid, free energy, uncertainty and method provenance.

    ``edges`` describes the histogram cells; ``grid`` gives their centers.
    ``uncertainty`` is the common error field and ``uncertainty_kind`` identifies
    its statistical meaning.  Method-specific diagnostics live in ``extras``
    and ``metadata`` rather than changing the core surface layout.
    """

    edges: tuple[np.ndarray, ...]
    free_energy: np.ndarray
    uncertainty: np.ndarray
    mask: np.ndarray
    cv_units: tuple[str, ...]
    method: str
    uncertainty_kind: str
    energy_unit: str = "hartree"
    metadata: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)
    uncertainty_issue: str | None = None

    def __post_init__(self):
        self.free_energy = np.asarray(self.free_energy, float)
        if self.free_energy.ndim not in (1, 2, 3) or not self.free_energy.size:
            raise ValueError("FES requires a nonempty 1D, 2D or 3D free-energy grid")
        self.edges = tuple(np.asarray(edge, float) for edge in self.edges)
        if len(self.edges) != self.dimension:
            raise ValueError("FES needs one edge array per CV dimension")
        for axis, edge in enumerate(self.edges):
            if (edge.shape != (self.free_energy.shape[axis] + 1,)
                    or not np.isfinite(edge).all() or np.any(np.diff(edge) <= 0)):
                raise ValueError("FES bin edges must be finite, increasing and match the grid")
        self.mask = np.asarray(self.mask, bool)
        if self.mask.shape != self.free_energy.shape:
            raise ValueError("FES mask shape differs from the free-energy grid")
        self.mask = self.mask | ~np.isfinite(self.free_energy)
        self.uncertainty = np.asarray(self.uncertainty, float)
        if self.uncertainty.shape != self.free_energy.shape:
            raise ValueError("FES uncertainty and its mask must match the grid")
        self.uncertainty = np.where(self.mask | ~np.isfinite(self.uncertainty)
                                    | (self.uncertainty < 0), np.nan, self.uncertainty)
        self.cv_units = tuple(self.cv_units)
        if len(self.cv_units) != self.dimension or any(not isinstance(unit, str) or not unit for unit in self.cv_units):
            raise ValueError("CV unit count does not match the saved FES dimension")
        if self.method not in METHODS:
            raise ValueError("FES method must be umbrella, metadynamics or opes")
        if self.uncertainty_kind not in UNCERTAINTY_KINDS:
            raise ValueError("FES uncertainty kind must be block_bootstrap or replica_spread")
        expected_kind = "block_bootstrap" if self.method == "umbrella" else "replica_spread"
        if self.uncertainty_kind != expected_kind:
            raise ValueError("FES uncertainty kind contradicts its sampling method")
        if self.energy_unit != "hartree":
            raise ValueError("Native FES arrays must declare hartree energy units")
        self.metadata = dict(self.metadata)
        self.extras = dict(self.extras)
        declared = _declared_method(self.metadata)
        if declared is not None and declared != self.method:
            raise ValueError("FES method contradicts the declared task or method metadata")
        measure = _measure(self.metadata)
        if measure is not None:
            self.metadata["measure"] = measure

    @property
    def dimension(self):
        return self.free_energy.ndim

    @property
    def grid(self):
        centers = [(edge[:-1] + edge[1:]) * 0.5 for edge in self.edges]
        return np.stack(np.meshgrid(*centers, indexing="ij"), axis=-1)

    @property
    def uncertainty_mask(self):
        return self.mask | ~np.isfinite(self.uncertainty)

    @property
    def measure(self):
        return self.metadata.get("measure")

    @property
    def error(self):
        """Alias for the common uncertainty field, without changing its meaning."""
        return self.uncertainty

    @classmethod
    def from_data(cls, data, *, method, cv_units, metadata=None):
        """Adapt an in-memory reconstruction without changing its numerics."""
        arrays = {key: value for key, value in data.items() if isinstance(value, np.ndarray)}
        arrays.update({f"edges_{axis}": edge for axis, edge in enumerate(data["edges"])})
        arrays.update({key: np.asarray(data[key]) for key in ("measure", "energy_unit", "beta")
                       if key in data})
        provenance = {key: data[key] for key in ("measure", "energy_unit", "beta") if key in data}
        provenance.update(metadata or {})
        declared = _declared_method(provenance)
        if declared is not None and declared != method:
            raise ValueError("FES method contradicts the declared task or method metadata")
        if provenance.get("method") is not None and provenance["method"] not in METHODS:
            provenance.setdefault("algorithm", provenance["method"])
        provenance.update(method=method, units=list(cv_units))
        arrays["method"] = np.asarray(method)
        arrays["cv_units"] = np.asarray(cv_units)
        return cls.from_arrays(arrays, metadata=provenance)

    @classmethod
    def from_arrays(cls, arrays, *, metadata=None, strict_uncertainty=True):
        """Read the common schema or centrally adapt a historical NPZ payload."""
        arrays = dict(arrays)
        supplied = dict(metadata or {})
        saved = {}
        if "fes_metadata" in arrays:
            saved = json.loads(str(_scalar(arrays["fes_metadata"], "metadata")))
            if not isinstance(saved, dict):
                raise ValueError("FES metadata must be a JSON object")
        for source in (arrays, saved, supplied):
            if "fes_schema" in source:
                version = _scalar(source.get("fes_schema_version", -1), "schema version")
                if (_scalar(source["fes_schema"], "schema") != SCHEMA
                        or isinstance(version, bool) or not isinstance(version, int)
                        or version != SCHEMA_VERSION):
                    raise ValueError("Unsupported FESResult schema")
        declared = [_declared_method(source) for source in (supplied, saved)]
        if "method" in arrays:
            declared.append(_scalar(arrays["method"], "method"))
        methods = {method for method in declared if method is not None}
        if len(methods) > 1:
            raise ValueError("FES method contradicts the declared task or method metadata")
        if methods and not methods.issubset(METHODS):
            raise ValueError("FES method must be umbrella, metadynamics or opes")
        provenance = {**supplied, **saved}
        if provenance.get("method") is not None and provenance["method"] not in METHODS:
            provenance.setdefault("algorithm", provenance["method"])
        if methods:
            provenance["method"] = next(iter(methods))
        measures = [_measure(source) for source in (supplied, saved)]
        for key in ("measure", "sampling_measure"):
            if key in arrays:
                measures.append(_scalar(arrays[key], key))
        measures = [value for value in measures if value is not None]
        if measures and any(value != measures[0] for value in measures[1:]):
            raise ValueError("Summary measure contradicts the saved FES array measure")
        if measures:
            provenance["measure"] = measures[0]
        for key in ("method", "measure", "energy_unit", "beta", "uncertainty_kind", "flux_axis"):
            if key in arrays or saved.get(key) is not None:
                value = _scalar(arrays[key] if key in arrays else saved[key], key)
                if key == "method" and value not in METHODS:
                    raise ValueError("FES method must be umbrella, metadynamics or opes")
                for source in (supplied, saved):
                    previous = source.get(key)
                    if previous is None or (key == "method" and previous not in METHODS):
                        continue
                    if previous != value:
                        raise ValueError(f"Summary {key} contradicts the saved FES array {key}")
                provenance[key] = value
        free = np.asarray(arrays["free_energy"], float)
        units = arrays.get("cv_units", provenance.get("units", ["unknown"] * free.ndim))
        units = np.asarray(units).tolist()
        if isinstance(units, str):
            units = [units]
        if "cv_units" in arrays:
            for source in (supplied, saved):
                previous = source.get("units")
                if isinstance(previous, str):
                    previous = [previous]
                if previous is not None and list(previous) != units:
                    raise ValueError("Summary CV units contradict the saved FES array units")
        provenance["units"] = units
        method = fes_method(provenance)
        kind = provenance.get("uncertainty_kind", "block_bootstrap" if method == "umbrella" else "replica_spread")
        error_key = "uncertainty" if "uncertainty" in arrays else (
            "standard_error" if kind == "block_bootstrap" else "replica_spread")
        error = np.asarray(arrays.get(error_key, np.full(free.shape, np.nan)), float)
        error_mask = np.asarray(arrays.get("uncertainty_mask", arrays.get(
            "standard_error_mask" if kind == "block_bootstrap" else "replica_spread_mask",
            np.zeros(free.shape, bool))), bool)
        issue = None
        if error.shape != free.shape or error_mask.shape != free.shape:
            issue = "FES uncertainty and its mask must match the grid"
            if strict_uncertainty:
                raise ValueError(issue)
            error = np.full(free.shape, np.nan)
        else:
            error = np.where(error_mask, np.nan, error)
        core = {"free_energy", "uncertainty", "uncertainty_mask", "mask", "cv_units", "method",
                "energy_unit", "measure", "beta", "uncertainty_kind", "fes_schema", "fes_schema_version",
                "fes_metadata", "grid"}
        extras = {key: value for key, value in arrays.items() if key not in core and not key.startswith("edges_")}
        return cls(edges=tuple(arrays[f"edges_{axis}"] for axis in range(free.ndim)),
                   free_energy=free, uncertainty=error, mask=arrays.get("mask", ~np.isfinite(free)),
                   cv_units=tuple(units), method=method, uncertainty_kind=kind,
                   energy_unit=provenance.get("energy_unit", "hartree"), metadata=provenance,
                   extras=extras, uncertainty_issue=issue)

    @classmethod
    def read(cls, path, *, summary=None, strict_uncertainty=True):
        path = Path(path)
        if path.is_dir():
            path = path / "fes.npz"
        if summary is None:
            for candidate in (path.parent / "summary.json", path.parent / "results" / "summary.json"):
                if candidate.is_file():
                    summary = json.loads(candidate.read_text(encoding="utf-8"))
                    break
        with load_checkpoint(path) as archive:
            return cls.from_arrays(dict(archive.items()), metadata=summary,
                                   strict_uncertainty=strict_uncertainty)

    def summary(self):
        declared = _declared_method(self.metadata)
        if declared is not None and declared != self.method:
            raise ValueError("FES method contradicts the declared task or method metadata")
        _measure(self.metadata)
        result = dict(self.metadata)
        old_method = result.get("method")
        if old_method is not None and old_method not in METHODS:
            result.setdefault("algorithm", old_method)
        result.update(fes_schema=SCHEMA, fes_schema_version=SCHEMA_VERSION,
                      task="fes", method=self.method, dimension=self.dimension,
                      units=list(self.cv_units), energy_unit=self.energy_unit,
                      uncertainty_kind=self.uncertainty_kind)
        result.setdefault("uncertainty_note", (
            "Block-bootstrap standard error (1 SE); unavailable bins are NaN."
            if self.uncertainty_kind == "block_bootstrap" else
            "Spread of independent replica estimates (1 SD); descriptive, not a calibrated standard error."))
        return _json_value(result)

    def arrays(self):
        """Return the authoritative portable payload, retaining old diagnostics."""
        arrays = dict(self.extras)
        arrays.update({f"edges_{axis}": edge for axis, edge in enumerate(self.edges)})
        arrays.update(free_energy=self.free_energy, uncertainty=self.uncertainty,
                      uncertainty_mask=self.uncertainty_mask, mask=self.mask,
                      grid=self.grid, cv_units=np.asarray(self.cv_units),
                      energy_unit=np.asarray(self.energy_unit), method=np.asarray(self.method),
                      uncertainty_kind=np.asarray(self.uncertainty_kind),
                      fes_schema=np.asarray(SCHEMA), fes_schema_version=np.asarray(SCHEMA_VERSION),
                      fes_metadata=np.asarray(json.dumps(self.summary(), sort_keys=True, allow_nan=False)))
        for key in ("measure", "beta", "flux_axis"):
            if self.metadata.get(key) is not None:
                arrays[key] = np.asarray(self.metadata[key])
        # These compatibility fields keep older consumers readable without
        # confusing replica spread with a statistically calibrated error.
        arrays.setdefault("standard_error", self.uncertainty if self.uncertainty_kind == "block_bootstrap"
                          else np.full(self.free_energy.shape, np.nan))
        arrays.setdefault("replica_spread", self.uncertainty if self.uncertainty_kind == "replica_spread"
                          else np.full(self.free_energy.shape, np.nan))
        return arrays


def write_fes_result(output, result, summary=None):
    """Write the same NPZ/CSV/JSON layout for every FES sampling method.

    Method-specific raw histories and sampling checkpoints remain separate
    diagnostic artifacts.  The returned summary should be used by the workflow
    and outer runner so their method/uncertainty declarations agree.
    """
    if not isinstance(result, FESResult):
        raise TypeError("write_fes_result requires a FESResult")
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    if summary is not None:
        result.metadata = {**summary, **result.metadata}
    files = dict(result.metadata.get("files", {}))
    files.update(data=str(path / "fes.npz"), table=str(path / "fes.csv"), metadata=str(path / "fes.json"))
    result.metadata["files"] = files
    canonical = result.summary()
    columns = [result.grid[..., axis].ravel() for axis in range(result.dimension)]
    columns.extend([result.free_energy.ravel(), result.uncertainty.ravel(), result.mask.ravel(),
                    result.uncertainty_mask.ravel()])
    count = np.asarray(result.extras.get("raw_count", np.full(result.free_energy.shape, np.nan)))
    if count.shape != result.free_energy.shape:
        raise ValueError("Raw counts must match the FES grid")
    columns.append(count.ravel())
    atomic_savez(path / "fes.npz", kind="fes_result", **result.arrays())
    temporary = path / "fes.csv.tmp"
    np.savetxt(temporary, np.column_stack(columns), delimiter=",", comments="",
               header=",".join([f"cv{axis}_{unit}" for axis, unit in enumerate(result.cv_units)] +
                               ["free_energy_hartree", "uncertainty_hartree", "masked",
                                "uncertainty_masked", "raw_count"]))
    temporary.replace(path / "fes.csv")
    write_json(path / "fes.json", canonical)
    write_json(path / "summary.json", canonical)
    return canonical
