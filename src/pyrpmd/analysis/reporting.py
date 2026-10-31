"""Saved-result analysis and reproducible scientific figures.

This module never imports a run's input, CV factory or potential.  The numerical
workflows remain the authority for rate acceptance and statistical uncertainty.
Matplotlib is imported only when figures are requested.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .. import __version__
from ..checkpoint import load_checkpoint
from ..constants import AU_TIME_FS
from ..io import jsonable
from ..units import unit_factor
from .fes_result import FESResult, fes_method

SCHEMA = "pyrpmd.analysis-report"
REFERENCES = [
    {"title": "Gui et al. (2022): Cayley RPMD, OH + CH4; PMF, recrossing, rates and KIE",
     "url": "https://doi.org/10.1021/acs.jctc.2c00522"},
    {"title": "Yang et al.: channel-resolved H + MgH RPMD",
     "url": "https://arxiv.org/abs/2007.07059"},
    {"title": "Li et al. (2023): surface RPMD and sampling convergence",
     "url": "https://doi.org/10.1039/D2SC06559B"},
    {"title": "Post-RPMDrate: plotting-workflow inspiration (independent implementation)",
     "url": "https://github.com/chen-yu-hao/Post-RPMDrate"},
    {"title": "JCTC author guidelines: graphics size, resolution and accessibility",
     "url": "https://researcher-resources.acs.org/publish/author_guidelines?coden=jctcce"},
    {"title": "Nature research figure guide: sizing, editable text and accessible graphics",
     "url": "https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/"},
]


def _number(value):
    if isinstance(value, (bool, str, list, dict)) or value is None:
        return None
    try:
        answer = float(value)
    except (TypeError, ValueError):
        return None
    return answer if np.isfinite(answer) else None


def _source(run, path, **extra):
    path = Path(path).resolve()
    # A second digest at publication checks that a concurrent run did not change
    # the source between collection and rendering.
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    previous = run["_sources"].get(str(path))
    if previous is not None and previous["sha256"] != digest:
        raise RuntimeError(f"Source changed during analysis: {path}")
    run["_sources"][str(path)] = {
        "path": str(path), "sha256": digest, "size": path.stat().st_size, **extra,
    }


def _json(run, path):
    _source(run, path)
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _npz(run, path):
    _source(run, path)
    with load_checkpoint(path) as data:
        run["_sources"][str(Path(path).resolve())]["integrity"] = data.envelope["integrity"]
        return dict(data.items())


def _finding(run, code, message, level="warning"):
    run["findings"].append({"level": level, "code": code, "message": message})


def _skip(run, figure, reason):
    run["skipped"].append({"figure": figure, "reason": reason})


def _guard(run, figure, action):
    """Record optional-artifact failures and return whether collection succeeded."""
    try:
        action()
        return True
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        _skip(run, figure, str(error))
        _finding(run, "artifact_unavailable", f"{figure}: {error}")
        return False


def _line(run, suffix, title, xlabel, ylabel, series, caption, **extra):
    plot = {"id": f"{run['id']}-{suffix}", "kind": "line", "title": title,
            "xlabel": xlabel, "ylabel": ylabel, "series": series,
            "caption": caption, **extra}
    run["_plots"].append(plot)
    return plot


def _selected_cv_names(config, task, method=None):
    """Resolve the saved sampling selection in the producer's axis order."""
    section = method if method in ("umbrella", "metadynamics", "opes") else fes_method({**config, "task": task})
    selected = config.get("fes", {}).get(section, config.get(section, {})).get("cvs")
    if selected is None:
        selected = config.get("fes", {}).get("cvs")
    if selected is not None:
        return [selected] if isinstance(selected, str) else list(selected)
    specifications = config.get("cvs", [])
    if isinstance(specifications, dict):
        return list(specifications)
    return [spec.get("name", f"CV {index + 1}") for index, spec in enumerate(specifications)]


def _cv_axes(run, summary, dimension, length_unit, data=None):
    config = run["_config"]
    selected = _selected_cv_names(config, summary.get("task", run["task"]), summary.get("method"))
    specifications = config.get("cvs", [])
    if isinstance(specifications, dict):
        specifications = [dict(value, name=name) for name, value in specifications.items()]
    definitions = {c.get("name"): c for c in specifications}
    units = summary.get("units")
    if isinstance(units, str):
        units = [units]
    if data is not None and "cv_units" in data:
        saved_units = np.asarray(data["cv_units"]).tolist()
        if isinstance(saved_units, str):
            saved_units = [saved_units]
        if units is not None and units != saved_units:
            run["_comparison_identity_available"] = False
            raise ValueError("Summary CV units contradict the saved FES array units")
        units = saved_units
    if not units:
        units = ["dimensionless"] if run["task"] == "rate" and dimension == 1 else ["unknown"] * dimension
    if len(units) != dimension:
        raise ValueError("CV unit count does not match the saved FES dimension")
    axes = []
    for j, unit in enumerate(units):
        name = selected[j] if j < len(selected) else f"CV {j + 1}"
        definition = definitions.get(name, {})
        if definition.get("kind") in ("distance", "bond"):
            atoms = definition.get("atoms", [])
            symbols = run["_system"].get("symbols", [])
            if len(atoms) == 2 and all(isinstance(a, int) and 0 <= a < len(symbols) for a in atoms):
                name = name + " [" + "-".join(f"{symbols[a]}{a+1}" for a in atoms) + "]"
        factor = 1.0
        shown_unit = str(unit)
        if unit in ("bohr", "angstrom"):
            factor = unit_factor(unit, "length") / unit_factor(length_unit, "length")
            shown_unit = length_unit
        if unit == "unknown":
            _finding(run, "cv_unit_missing", f"Unit for {name} is unavailable; coordinates were not converted.")
        axes.append((f"{name} ({shown_unit})", factor))
    return axes


def _curve_arrays(data, xkey, ykey, errkey=None):
    x, y = np.asarray(data[xkey], float), np.asarray(data[ykey], float)
    if x.ndim != 1 or y.shape != x.shape or not len(x):
        raise ValueError("Curve coordinates and values must be nonempty 1D arrays of equal size")
    if not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("Curve coordinates must be finite and strictly increasing")
    mask = np.asarray(data.get("mask", np.zeros(x.shape)), bool)
    if mask.shape != x.shape:
        raise ValueError("Curve mask shape differs from values")
    y = np.where(mask | ~np.isfinite(y), np.nan, y)
    error = None
    if errkey and errkey in data:
        error = np.asarray(data[errkey], float)
        if error.shape != y.shape:
            raise ValueError("Curve uncertainty shape differs from values")
        error_mask = np.asarray(data.get("standard_error_mask", np.zeros(x.shape)), bool)
        if error_mask.shape != x.shape:
            raise ValueError("Uncertainty mask shape differs from values")
        error = np.where(error_mask | ~np.isfinite(y) | ~np.isfinite(error) | (error < 0), np.nan, error)
    return x, y, error


def _table(run, path):
    _source(run, path)
    values = np.genfromtxt(path, names=True, delimiter=",", encoding="utf-8", ndmin=1)
    if not values.dtype.names:
        raise ValueError("Expected a named CSV table")
    return {key: values[key] for key in values.dtype.names}


def _rate_artifacts(run, energy_unit, length_unit):
    path = Path(run["path"])
    summary = run["_summary"]
    surface = run["task"] == "surface_rate"
    name = "surface_pmf" if surface else "rate_pmf"
    def pmf():
        npz = path / (name + ".npz")
        if npz.exists():
            data = _npz(run, npz)
        else:
            data = _table(run, path / (name + ".csv"))
            if surface:
                data["xi"] = data["coordinate_bohr"]
                data["mask"] = data.get("masked", np.zeros_like(data["xi"]))
            data["free_energy"] = data["free_energy_hartree"]
            if "standard_error_hartree" in data:
                data["standard_error"] = data["standard_error_hartree"]
        x, y, error = _curve_arrays(data, "xi", "free_energy", "standard_error")
        factor = 1 / unit_factor(energy_unit, "energy")
        xf = 1 / unit_factor(length_unit, "length") if surface else 1.0
        xlabel = f"Surface coordinate ({length_unit})" if surface else "Reaction coordinate $\\xi$ (dimensionless)"
        plot = _line(run, "pmf", f"{run['label']}: PMF", xlabel,
                    f"Free energy ({energy_unit})", [{"label": run["label"], "x": x * xf,
                    "y": y * factor, "yerr": error * factor if error is not None else None}],
                    "Saved PMF gauge and unsupported gaps retained. Bands, where available, are the saved standard error (1 SE), not a 95% confidence interval.")
        xi_star = _number(summary.get("pmf", {}).get("xi_star"))
        if xi_star is not None:
            plot["vlines"] = [{"x": xi_star * xf, "label": "Dividing surface"}]
        plot["_compare"] = "pmf"
        run["_pmf_axis"] = xlabel
        run["_measure"] = summary.get("pmf", {}).get("measure", "rate_flux" if not surface else "surface_rate")
        run["metrics"]["pmf_supported_points"] = int(np.isfinite(y).sum())
        if np.isfinite(y).any():
            peak = int(np.nanargmax(y))
            if peak in (0, len(y) - 1):
                _finding(run, "pmf_edge_maximum", "The largest displayed PMF value is at a grid edge; inspect domain support before interpreting a barrier.")
    if run["task"] != "site_rate":
        _guard(run, "rate PMF", pmf)
    def recrossing():
        data = _npz(run, path / "recrossing.npz") if (path / "recrossing.npz").exists() else _table(run, path / "recrossing.csv")
        key = "time_au" if "time_au" in data else "time_fs"
        x, y, error = _curve_arrays(data, key, "kappa", "standard_error")
        x = x * AU_TIME_FS if key == "time_au" else x
        plot = _line(run, "recrossing", f"{run['label']}: transmission", "Time (fs)", "Transmission coefficient $\\kappa$",
                    [{"label": run["label"], "x": x, "y": y, "yerr": error}],
                    "Signed transmission coefficients are not clipped. Bands are saved parent-aware standard errors when available. A selected tail is a diagnostic interval, not proof of numerical convergence.")
        start = _number(summary.get("plateau_start_au"))
        if start is None and _number(summary.get("plateau_start_fs")) is not None:
            start = summary["plateau_start_fs"] / AU_TIME_FS
        if start is not None and x[0] <= start * AU_TIME_FS <= x[-1]:
            plot["spans"] = [{"start": start * AU_TIME_FS, "stop": float(x[-1]),
                              "label": "Selected tail" if not summary.get("plateau_detected") else "Detected plateau interval"}]
        plot["_compare"] = "recrossing"
        if summary.get("plateau_detected") is False:
            _finding(run, "recrossing_no_plateau", "The producer did not detect a supported positive recrossing plateau; extend children/sampling before using a thermal rate.")
    _guard(run, "recrossing", recrossing)


def _fes_artifacts(run, folder, energy_unit, length_unit):
    path = Path(folder)
    summary_path = path / "summary.json"
    summary = _json(run, summary_path) if summary_path.exists() else run["_summary"]
    try:
        provenance = {"task": run["task"], **summary}
        config = run["_config"]
        if path == Path(run["path"]) and config.get("task") in ("fes", "metadynamics", "opes"):
            # The saved selection determines axis order. Refuse a surface whose
            # method disagrees with that selection instead of guessing labels.
            provenance["fes"] = {"method": fes_method(config)}
        result = FESResult.from_arrays(_npz(run, path / "fes.npz"),
                                       metadata=provenance,
                                       strict_uncertainty=False)
    except ValueError:
        run["_comparison_identity_available"] = False
        raise
    summary = result.summary()
    data = {**result.extras, "cv_units": np.asarray(result.cv_units)}
    measure = result.measure
    if measure not in ("cv_density", "rate_flux"):
        run["_comparison_identity_available"] = False
        _finding(run, "fes_measure_unavailable", "The FES measure is unavailable or unsupported; curves are not grouped or overlaid.")
    if path == Path(run["path"]):
        run["_measure"] = run["metadata"]["measure"] = measure
        run["_fes_result"] = result
        run["metadata"]["fes_method"] = result.method
        run["metadata"]["uncertainty_kind"] = result.uncertainty_kind
    f = result.free_energy
    dimension = result.dimension
    edges = result.edges
    axes = _cv_axes(run, summary, dimension, length_unit, data)
    mask = result.mask
    factor = 1 / unit_factor(energy_unit, "energy")
    prefix = "umbrella" if path != Path(run["path"]) else "fes"
    if prefix == "umbrella" and summary.get("samples") is not None:
        run["metadata"]["samples"] = summary["samples"]
        run["metadata"]["sample_scope"] = "PMF umbrella production samples (not recrossing children)"
    caption = "Free energy in the saved CV measure and gauge. Unsupported bins remain masked; no interpolation or extrapolation is used. "
    caption += summary["uncertainty_note"]
    error_label = {"block_bootstrap": "Standard error", "replica_spread": "Replica spread"}[result.uncertainty_kind]
    error_band = {"block_bootstrap": "1 SE", "replica_spread": "1 SD (descriptive)"}[result.uncertainty_kind]
    run["metrics"][f"{prefix}_supported_bins"] = int((~mask).sum())
    run["metrics"][f"{prefix}_total_bins"] = int(mask.size)
    if summary.get("overlap_connected") is False:
        _finding(run, "disconnected_overlap", "Disconnected umbrella overlap: a single global FES cannot be identified.")
    if np.any(mask):
        _finding(run, "masked_fes_bins", f"{int(mask.sum())}/{mask.size} FES bins are unsupported and remain unfilled.", "info")
    error = np.full(f.shape, np.nan)
    def uncertainty():
        nonlocal error
        if result.uncertainty_issue:
            raise ValueError(result.uncertainty_issue)
        error = result.uncertainty
    _guard(run, "FES uncertainty", uncertainty)
    if dimension == 1:
        x = (edges[0][:-1] + edges[0][1:]) * 0.5 * axes[0][1]
        plot = _line(run, prefix, f"{run['label']}: FES", axes[0][0],
                    f"Free energy ({energy_unit})", [{"label": run["label"], "x": x,
                    "y": np.where(mask, np.nan, f) * factor,
                    "yerr": error * factor}], caption + f" Bands show saved {error_band}.")
        if prefix == "fes":
            plot["_compare"] = "pmf"
            run["_pmf_axis"] = axes[0][0]
            run["_measure"] = measure
    elif dimension == 2:
        run["_plots"].append({"id": f"{run['id']}-{prefix}", "kind": "heatmap", "title": f"{run['label']}: 2D FES",
                               "xlabel": axes[0][0], "ylabel": axes[1][0], "xedges": edges[0] * axes[0][1],
                               "yedges": edges[1] * axes[1][1], "z": f * factor, "mask": mask,
                               "colorbar": f"Free energy ({energy_unit})", "caption": caption})
        if np.isfinite(error).any():
            run["_plots"].append({**run["_plots"][-1], "id": f"{run['id']}-{prefix}-error",
                                   "title": f"{run['label']}: FES uncertainty", "z": error * factor,
                                   "mask": mask | ~np.isfinite(error), "colorbar": f"{error_label} ({energy_unit})",
                                   "caption": f"Saved uncertainty ({error_band}). Missing uncertainty is masked, not zero."})
    else:
        # Show actual orthogonal slices, never an implicit projection or a
        # minimum-energy marginal that would change the observable's meaning.
        reference = np.unravel_index(np.argmin(np.where(mask, np.inf, f)), f.shape)
        if np.all(mask):
            reference = tuple(size // 2 for size in f.shape)
        for fixed in range(3):
            shown = [axis for axis in range(3) if axis != fixed]
            selector = [slice(None)] * 3
            selector[fixed] = reference[fixed]
            selector = tuple(selector)
            value = result.grid[reference][fixed] * axes[fixed][1]
            run["_plots"].append({"id": f"{run['id']}-{prefix}-slice-{fixed}", "kind": "heatmap",
                "title": f"{run['label']}: 3D FES slice ({axes[fixed][0]} = {value:.6g})",
                "xlabel": axes[shown[0]][0], "ylabel": axes[shown[1]][0],
                "xedges": edges[shown[0]] * axes[shown[0]][1],
                "yedges": edges[shown[1]] * axes[shown[1]][1],
                "z": f[selector] * factor, "mask": mask[selector],
                "colorbar": f"Free energy ({energy_unit})",
                "caption": caption + " Orthogonal slice through the lowest supported bin; not a marginal free energy."})
    def counts():
        count = np.asarray(data["raw_count"], float)
        if count.shape != f.shape or not np.isfinite(count).all() or np.any(count < 0):
            raise ValueError("Raw counts must match the FES and be finite and nonnegative")
        if dimension == 1:
            _line(run, prefix + "-counts", f"{run['label']}: sampling coverage", axes[0][0], "Saved sample count",
                  [{"label": "Raw count", "x": x, "y": count}], "Raw saved frames are correlated and are not independent sample counts.")
        elif dimension == 2:
            run["_plots"].append({"id": f"{run['id']}-{prefix}-counts", "kind": "heatmap", "title": f"{run['label']}: sampling coverage",
                                   "xlabel": axes[0][0], "ylabel": axes[1][0], "xedges": edges[0] * axes[0][1],
                                   "yedges": edges[1] * axes[1][1], "z": count, "colorbar": "Saved sample count",
                                   "caption": "Actual raw counts, including zero-count bins. Counts do not measure statistical independence."})
    if "raw_count" in data:
        _guard(run, "FES sampling coverage", counts)
    def overlap_matrix():
        overlap = np.asarray(data["overlap"], float)
        if (overlap.ndim != 2 or not overlap.size or overlap.shape[0] != overlap.shape[1]
                or not np.isfinite(overlap).all() or np.any(overlap < 0)
                or np.any(overlap > 1 + 1e-8)):
            raise ValueError("Overlap must be a nonempty square stream matrix with finite values between zero and one")
        stream_edges = np.arange(overlap.shape[0] + 1) - 0.5
        run["_plots"].append({"id": f"{run['id']}-{prefix}-overlap", "kind": "heatmap",
                               "title": f"{run['label']}: MBAR overlap", "xlabel": "Sampling stream j", "ylabel": "Sampling stream i",
                               "xedges": stream_edges, "yedges": stream_edges, "z": overlap.T,
                               "colorbar": "Overlap O(i,j)", "caption": "Saved MBAR overlap matrix. Stream indices can include independent replicas of the same window."})
    if "overlap" in data:
        _guard(run, "MBAR overlap", overlap_matrix)
    def effective_samples():
        ess = np.asarray(data["effective_samples_per_stream"], float)
        if ess.ndim != 1 or not len(ess) or not np.isfinite(ess).all() or np.any(ess < 0):
            raise ValueError("Effective sample counts must be a nonempty finite nonnegative 1D stream array")
        _line(run, prefix + "-ess", f"{run['label']}: effective samples", "Sampling stream", "Estimated effective samples",
              [{"label": "ESS", "x": np.arange(len(ess)), "y": ess, "style": "points"}],
              "Correlation-adjusted effective sample size, estimated by the existing analysis. This is a sampling diagnostic, not proof of equilibration.")
        run["metrics"][f"{prefix}_minimum_stream_ess"] = float(np.min(ess))
    if "effective_samples_per_stream" in data:
        _guard(run, "Effective sample counts", effective_samples)
    from .report_sampling import collect_sampling
    # Sampling diagnostics must use the same checked units and measure as the
    # FES axes, including when those declarations exist only in the NPZ.
    sampling_summary = {**summary, "measure": measure}
    if "cv_units" in data:
        sampling_summary["units"] = np.asarray(data["cv_units"]).tolist()
    _guard(run, "Window sampling diagnostics", lambda: collect_sampling(run, path, sampling_summary, axes, energy_unit))


def _rate_metrics(run):
    summary = run["_summary"]
    task = run["task"]
    suffix = {"rate": "cm3_s", "surface_rate": "s_inv", "site_rate": "s"}.get(task)
    if suffix is None:
        return
    pmf = summary.get("pmf", {})
    rate = _number(summary.get("rate_" + suffix))
    estimate = _number(summary.get("rate_estimate_" + suffix))
    error = _number(summary.get("rate_standard_error_" + suffix))
    unit = "cm3 molecule^-1 s^-1" if task == "rate" else "s^-1"
    run["_rate"] = {"value": rate, "estimate": estimate, "standard_error": error if error is not None and error >= 0 else None,
                    "unit": unit, "qtst": _number(pmf.get("k_qtst_" + suffix, summary.get("k_qtst_" + suffix)))}
    for key in ("kappa", "kappa_standard_error", "plateau_detected", "parent_configurations", "child_trajectories", "maximum_endpoint_energy_error_kbt"):
        if key in summary:
            run["metrics"][key] = summary[key]
    run["metrics"].update({"rate": rate, "rate_unit": unit, "diagnostic_rate_estimate": estimate,
                           "rate_standard_error": run["_rate"]["standard_error"], "k_qtst": run["_rate"]["qtst"],
                           "rate_percentile_95_ci": summary.get("rate_percentile_95_ci_" + suffix),
                           "rate_uncertainty": summary.get("rate_uncertainty")})
    for key in ("free_energy_barrier_hartree", "barrier_standard_error_hartree",
                "barrier_percentile_95_ci_hartree", "xi_star", "measure"):
        if key in pmf:
            run["metrics"][key] = pmf[key]
    if rate is None:
        _finding(run, "rate_withheld", "The saved workflow withholds the thermal rate. Any diagnostic estimate remains a candidate and is excluded from default rate comparisons.")
    if run["_rate"]["standard_error"] is None:
        _finding(run, "rate_uncertainty_unavailable", "Complete saved rate uncertainty is unavailable; conditional recrossing error is not substituted.")
    if _number(summary.get("maximum_endpoint_energy_error_kbt")) is not None and summary["maximum_endpoint_energy_error_kbt"] > 0.1:
        _finding(run, "child_energy_error", "Saved child endpoint energy error exceeds 0.1 kBT per bead; inspect the timestep.")


def _comparison_sources(run):
    """Use saved preparation/model provenance without executing producer code."""
    config, path = run["_config"], Path(run["path"])
    run["_comparison_identity_available"] = True
    if run["task"] == "rate":
        reaction = config.get("reaction", {})
        preparation = path / "preparation.json"
        if preparation.is_file():
            reaction = None
            try:
                saved = _json(run, preparation)
                if saved.get("ready") is not True:
                    raise ValueError("Saved reaction preparation is not ready")
                reaction = saved.get("reaction")
            except (OSError, ValueError, KeyError, TypeError) as error:
                _finding(run, "reaction_identity_unavailable", str(error))
        if (not isinstance(reaction, dict) or any(
                not isinstance(reaction.get(key), list) or not reaction[key]
                for key in ("fragments", "forming_bonds", "breaking_bonds"))):
            run["_comparison_identity_available"] = False
            _finding(run, "reaction_identity_unavailable",
                     "The actual fragments and forming/breaking bonds are unavailable; this rate is not grouped, overlaid, fitted or used for KIE comparisons.")
        else:
            run["_prepared_reaction"] = reaction

    manifest = path / "manifest.json"
    if manifest.is_file():
        try:
            saved = _json(run, manifest)
            files = saved.get("files")
            if (saved.get("schema", "pyrpmd.run") != "pyrpmd.run"
                    or not isinstance(files, dict)
                    or any(not isinstance(name, str) or not isinstance(digest, str)
                           or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                           for name, digest in files.items())):
                raise ValueError("Saved run manifest lacks valid model-file digests")
            # Inputs may differ by temperature and initial coordinates. Their
            # decoded contents, including the rate's TS reference, are compared
            # separately; keep model/plugin dependencies in the identity.
            source_input = config.get("_input_path")
            excluded = ({source_input} if isinstance(source_input, str)
                        and Path(source_input).suffix.lower() in (".inp", ".toml", ".json") else set())

            def exclude_paths(value):
                if isinstance(value, dict):
                    for item in value.values():
                        exclude_paths(item)
                elif isinstance(value, list):
                    for item in value:
                        exclude_paths(item)
                elif isinstance(value, str):
                    excluded.add(value)

            exclude_paths(config.get("system", {}))
            run["_model_files"] = {name: digest for name, digest in files.items() if name not in excluded}
        except (OSError, ValueError, KeyError, TypeError) as error:
            run["_comparison_identity_available"] = False
            _finding(run, "model_identity_unavailable", str(error))
    elif any(key in config.get("potential", {}) for key in
             ("library", "manifest", "model", "weights", "factory", "module", "function", "initializer")):
        run["_comparison_identity_available"] = False
        _finding(run, "model_identity_unavailable",
                 "Saved model/plugin file digests are unavailable; external-model runs are not automatically compared.")


def collect_run(directory, *, label=None, prefix="run-001", energy_unit="kcal/mol", length_unit="angstrom"):
    """Collect saved data and plot specifications without importing Matplotlib."""
    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Run directory does not exist: {path}")
    unit_factor(energy_unit, "energy")
    unit_factor(length_unit, "length")
    default_label = path.parent.name if path.name == "run" else path.name
    run = {"id": prefix, "path": str(path), "label": str(label or default_label), "metrics": {},
           "metadata": {}, "warnings": [], "findings": [], "skipped": [], "_sources": {}, "_plots": []}
    summary_path = path / "results" / "summary.json"
    if not summary_path.is_file():
        summary_path = path / "summary.json"
    config_path, system_path = path / "input" / "resolved.json", path / "input" / "system.json"
    config = run["_config"] = _json(run, config_path) if config_path.is_file() else {}
    run["_system"] = _json(run, system_path) if system_path.is_file() else config.get("system", {})
    if not summary_path.is_file() and any((path / part / "adaptive-plan.json").is_file()
                                         for part in ("", "umbrella", "pmf")):
        summary = {"task": config.get("task", "unknown"), "status": "summary_unavailable"}
        _finding(run, "summary_unavailable", "No saved result summary is available. Adaptive plan diagnostics describe the recorded pilot state only; final results and convergence are unavailable.")
    else:
        summary = _json(run, summary_path)
    run["_summary"] = summary
    run["task"] = summary.get("task", config.get("task", "unknown"))
    run["status"] = summary.get("status", "not_recorded")
    run["converged"] = summary.get("converged")
    _comparison_sources(run)
    dt = _number(config.get("dt_au"))
    reaction, collision = run.get("_prepared_reaction", config.get("reaction", {})), config.get("collision", {})
    run["metadata"] = {
        "temperature_K": summary.get("temperature_K", config.get("temperature_K")),
        "beads": summary.get("beads", config.get("beads")), "dt_fs": None if dt is None else dt * AU_TIME_FS,
        "samples": summary.get("samples", summary.get("trajectories")),
        "integrator": config.get("dynamics", {}).get("integrator"), "producer_version": summary.get("version"),
        "measure": summary.get("measure", summary.get("pmf", {}).get("measure")),
        "rinf_bohr": _number(reaction.get("rinf_au", reaction.get("rinf"))),
        "bmax_bohr": _number(collision.get("bmax_au", collision.get("bmax"))),
    }
    run["warnings"] = list(dict.fromkeys(str(w) for w in summary.get("warnings", []) + summary.get("input_warnings", [])))
    if not config:
        _finding(run, "resolved_input_missing", "Saved resolved input is unavailable; missing temperatures, CV names and physics identity are not guessed.")
    _finding(run, "convergence_scope", summary.get("convergence_note", "This report preserves producer diagnostics; numerical and physical convergence require controlled comparisons."), "info")
    _rate_metrics(run)
    def adaptive_progress(folder):
        from .report_sampling import collect_adaptive_progress
        if run["task"] == "unknown":
            # In a partial archive the plan location can survive without either
            # the summary or resolved input. Retain each plan's location rather
            # than guessing a task or assigning duplicate figure identifiers.
            for candidate in (path, path / "umbrella", path / "pmf"):
                collect_adaptive_progress(run, candidate,
                                          prefix="sampling" if candidate == path else candidate.name)
        else:
            collect_adaptive_progress(run, folder)
    if run["task"] in ("rate", "surface_rate", "site_rate"):
        _rate_artifacts(run, energy_unit, length_unit)
        folder = path / ("pmf" if run["task"] == "site_rate" else "umbrella")
        collected = False
        if (folder / "fes.npz").exists():
            collected = _guard(run, "umbrella sampling", lambda: _fes_artifacts(run, folder, energy_unit, length_unit))
        else:
            _skip(run, "umbrella sampling", "Saved umbrella/fes.npz or pmf/fes.npz is unavailable")
        if not collected:
            adaptive_progress(folder)
    elif (path / "fes.npz").exists():
        if not _guard(run, "free-energy surface", lambda: _fes_artifacts(run, path, energy_unit, length_unit)):
            adaptive_progress(path)
    else:
        adaptive_progress(path)
        if run["task"] in ("fes", "metadynamics", "opes"):
            _skip(run, "free-energy surface", "Saved fes.npz is unavailable; available pilot diagnostics are reported separately")
    from .report_observables import collect_observables
    supplemental = collect_observables(path, summary, config, prefix=prefix, label=run["label"],
                                       energy_unit=energy_unit, length_unit=length_unit)
    run["_plots"].extend(supplemental.get("plots", []))
    run["metrics"].update(supplemental.get("metrics", {}))
    run["findings"].extend(supplemental.get("findings", []))
    run["skipped"].extend(supplemental.get("skipped", []))
    for record in supplemental.get("source_records", []):
        previous = run["_sources"].get(record["path"])
        if previous is not None and previous["sha256"] != record["sha256"]:
            raise RuntimeError(f"Source changed during analysis: {record['path']}")
        run["_sources"][record["path"]] = record
    for source in supplemental.get("sources", []):
        if str(Path(source).resolve()) not in run["_sources"]:
            _source(run, source)
    if not run["_plots"]:
        _skip(run, "figures", "No supported numerical plot artifacts were found; the saved summary is still included")
    context = []
    for key, template in (("temperature_K", "T = {:g} K"), ("beads", "P = {:g}"), ("dt_fs", "dt = {:g} fs")):
        value = _number(run["metadata"].get(key))
        if value is not None:
            context.append(template.format(value))
    if run.get("_rate", {}).get("value", 0) is None:
        context.append("Rate withheld by source")
    if context:
        for plot in run["_plots"]:
            plot["note"] = "; ".join(context) + (". " + plot["note"] if plot.get("note") else "")
    return run


def _validate_destination(output, runs, overwrite):
    if output.is_symlink():
        raise ValueError("Report destination must not be a symbolic link")
    resolved = output.resolve()
    for run in runs:
        source = Path(run["path"])
        if source == resolved or resolved in source.parents:
            raise ValueError("Report destination must not replace or contain an input run directory")
    if not output.exists():
        return
    if not output.is_dir():
        raise ValueError("Report output must be a directory")
    if not any(output.iterdir()):
        return
    marker = output / "report.json"
    if not overwrite or not marker.is_file():
        raise ValueError("Report destination is not empty; use another directory or --overwrite for an existing PyRPMD report")
    old = json.loads(marker.read_text(encoding="utf-8"))
    if old.get("schema") != SCHEMA or not isinstance(old.get("generated_files"), list):
        raise ValueError("Refusing to overwrite a directory without a PyRPMD report inventory")
    inventory = set(old["generated_files"])
    if any(p.is_symlink() for p in output.rglob("*")):
        raise ValueError("Refusing to overwrite a report containing symbolic links")
    unexpected = [str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() and str(p.relative_to(output)) not in inventory]
    if unexpected:
        raise ValueError("Report destination contains files outside its generated inventory: " + ", ".join(unexpected[:5]))


def generate_report(directories, output=None, *, formats=("png", "pdf"), dpi=300,
                    energy_unit="kcal/mol", length_unit="angstrom", labels=None,
                    series=None, reference=None, include_estimates=False,
                    arrhenius_fit=False, kie=None, convergence=None, overwrite=False,
                    style="jctc"):
    """Generate figures, reusable CSV data and an HTML/JSON/Markdown report.

    ``directories`` is a path or sequence of run paths.  Multiple runs require an
    explicit output.  ``series`` supplies one grouping label per run; automatic
    grouping is deliberately conservative.  Fits are descriptive and opt-in.
    An existing report can be replaced only with ``overwrite=True`` and an intact
    generated-file inventory. Source results are never modified.
    """
    if isinstance(directories, (str, os.PathLike)):
        directories = [directories]
    directories = list(directories)
    if not directories:
        raise ValueError("At least one run directory is required")
    if labels is not None and len(labels) != len(directories):
        raise ValueError("labels must contain one label per run")
    if series is not None and len(series) != len(directories):
        raise ValueError("series must contain one series name per run")
    if isinstance(formats, str):
        formats = (formats,)
    formats = tuple(dict.fromkeys(formats))
    if not formats or set(formats) - {"png", "pdf", "svg"}:
        raise ValueError("Figure formats must be png, pdf or svg")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or not 72 <= dpi <= 2400:
        raise ValueError("dpi must be an integer between 72 and 2400")
    if style not in ("jctc", "nature"):
        raise ValueError("Figure style must be jctc or nature")
    if output is None:
        if len(directories) != 1:
            raise ValueError("An explicit report output is required for multiple runs")
        output = Path(directories[0]) / "analysis-report"
    output = Path(output).expanduser().absolute()
    runs = [collect_run(d, label=labels[i] if labels else None, prefix=f"run-{i+1:03d}",
                        energy_unit=energy_unit, length_unit=length_unit) for i, d in enumerate(directories)]
    _validate_destination(output, runs, overwrite)
    if reference:
        _source(runs[0], reference)
    from .report_compare import collect_comparisons
    compare = collect_comparisons(runs, series=series, reference=reference,
                                  include_estimates=include_estimates, arrhenius_fit=arrhenius_fit,
                                  kie=kie, convergence=convergence)
    plots = [p for run in runs for p in run["_plots"]] + compare["plots"]
    source_map = {}
    for run in runs:
        for source in run["_sources"].values():
            old = source_map.get(source["path"])
            if old is not None and old["sha256"] != source["sha256"]:
                raise RuntimeError(f"Source changed between collected runs: {source['path']}")
            source_map[source["path"]] = source
    report = {
        "schema": SCHEMA, "schema_version": 1, "title": "PyRPMD analysis report",
        "generated_at": datetime.now(timezone.utc).isoformat(), "software_version": __version__,
        "output": str(output), "settings": {"formats": list(formats), "dpi": dpi, "style": style,
            "energy_unit": energy_unit, "length_unit": length_unit, "include_estimates": include_estimates,
            "arrhenius_fit": arrhenius_fit, "kie": kie, "convergence": convergence, "series": series},
        "runs": [{k: v for k, v in run.items() if not k.startswith("_")} for run in runs],
        "findings": [{**f, "run": run["label"]} for run in runs for f in run["findings"]] + compare["findings"],
        "skipped": [{**s, "run": run["label"]} for run in runs for s in run["skipped"]] + compare["skipped"],
        "comparisons": compare["metrics"], "references": REFERENCES,
        "sources": sorted(source_map.values(), key=lambda s: s["path"]),
        "interpretation": "Source acceptance fields are preserved, not revalidated. SE, confidence intervals, replica spread and unresolved-outcome bounds are distinct. No new dynamics, PES or input code is executed.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".pyrpmd-report-", dir=output.parent))
    try:
        from .report_render import render_figures, write_report
        report["figures"] = render_figures(plots, stage, formats=formats, dpi=dpi, style=style)
        report["dependencies"] = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")}
        with (stage / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["run", "task", "status", "section", "quantity", "value"])
            for run in report["runs"]:
                for section in ("metadata", "metrics"):
                    for key, value in run[section].items():
                        writer.writerow([run["label"], run["task"], run["status"], section, key,
                                         "" if value is None else json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False)])
        write_report(report, stage)
        report["generated_files"] = sorted([str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file()] + ["report.json"])
        (stage / "report.json").write_text(json.dumps(jsonable(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for source in report["sources"]:
            with Path(source["path"]).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != source["sha256"]:
                    raise RuntimeError(f"Source changed during reporting; rerun after the simulation finishes: {source['path']}")
        # Recheck just before replacement to protect files added while rendering.
        _validate_destination(output, runs, overwrite)
        backup = None
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=".pyrpmd-report-previous-", dir=output.parent))
            backup.rmdir()
            os.replace(output, backup)
        try:
            os.replace(stage, output)
        except OSError:
            if backup is not None:
                os.replace(backup, output)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return jsonable(report)
