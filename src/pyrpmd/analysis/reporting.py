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
    """A damaged optional artifact must be visible without hiding other results."""
    try:
        action()
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        _skip(run, figure, str(error))
        _finding(run, "artifact_unavailable", f"{figure}: {error}")


def _line(run, suffix, title, xlabel, ylabel, series, caption, **extra):
    plot = {"id": f"{run['id']}-{suffix}", "kind": "line", "title": title,
            "xlabel": xlabel, "ylabel": ylabel, "series": series,
            "caption": caption, **extra}
    run["_plots"].append(plot)
    return plot


def _cv_axes(run, summary, dimension, length_unit, data=None):
    config = run["_config"]
    table = config.get("metadynamics" if summary.get("task") == "metadynamics" else "umbrella", {})
    selected = table.get("cvs", [])
    if isinstance(selected, str):
        selected = [selected]
    specifications = config.get("cvs", [])
    if isinstance(specifications, dict):
        specifications = [dict(value, name=name) for name, value in specifications.items()]
    definitions = {c.get("name"): c for c in specifications}
    units = summary.get("units")
    if units is None and data is not None and "cv_units" in data:
        units = np.asarray(data["cv_units"]).tolist()
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
    data = _npz(run, path / "fes.npz")
    declared_energy = str(np.asarray(data.get("energy_unit", summary.get("energy_unit", "hartree"))).item())
    if declared_energy != "hartree":
        raise ValueError("Native FES arrays must declare hartree energy units")
    f = np.asarray(data["free_energy"], float)
    if f.ndim not in (1, 2) or not f.size:
        raise ValueError("Automatic FES figures support nonempty 1D or 2D grids")
    dimension = f.ndim
    edges = [np.asarray(data[f"edges_{j}"], float) for j in range(dimension)]
    for j, edge in enumerate(edges):
        if edge.shape != (f.shape[j] + 1,) or not np.isfinite(edge).all() or np.any(np.diff(edge) <= 0):
            raise ValueError("FES bin edges must be finite, increasing and match the grid")
    axes = _cv_axes(run, summary, dimension, length_unit, data)
    mask = np.asarray(data.get("mask", ~np.isfinite(f)), bool)
    if mask.shape != f.shape:
        raise ValueError("FES mask shape differs from the free-energy grid")
    mask = mask | ~np.isfinite(f)
    factor = 1 / unit_factor(energy_unit, "energy")
    prefix = "umbrella" if path != Path(run["path"]) else "fes"
    if prefix == "umbrella" and summary.get("samples") is not None:
        run["metadata"]["samples"] = summary["samples"]
        run["metadata"]["sample_scope"] = "PMF umbrella production samples (not recrossing children)"
    is_meta = summary.get("task") == "metadynamics"
    caption = ("Finite-time metadynamics bias estimate; biased time is not a reaction-rate clock. " if is_meta else "Free energy in the saved CV measure and gauge. ")
    caption += "Unsupported bins remain masked; no interpolation or extrapolation is used."
    if is_meta:
        caption += " Replica spread is descriptive and is not a standard error."
    run["metrics"][f"{prefix}_supported_bins"] = int((~mask).sum())
    run["metrics"][f"{prefix}_total_bins"] = int(mask.size)
    if summary.get("overlap_connected") is False:
        _finding(run, "disconnected_overlap", "Disconnected umbrella overlap: a single global FES cannot be identified.")
    if np.any(mask):
        _finding(run, "masked_fes_bins", f"{int(mask.sum())}/{mask.size} FES bins are unsupported and remain unfilled.", "info")
    error = np.asarray(data.get("standard_error", np.full(f.shape, np.nan)), float)
    if error.shape != f.shape:
        raise ValueError("FES uncertainty shape differs from the grid")
    error = np.where(mask | (error < 0), np.nan, error)
    if dimension == 1:
        x = (edges[0][:-1] + edges[0][1:]) * 0.5 * axes[0][1]
        plot = _line(run, prefix, f"{run['label']}: {'bias FES' if is_meta else 'FES'}", axes[0][0],
                    f"Free energy ({energy_unit})", [{"label": run["label"], "x": x,
                    "y": np.where(mask, np.nan, f) * factor,
                    "yerr": None if is_meta else error * factor}], caption + (" Bands are saved 1 SE." if not is_meta else ""))
        if prefix == "fes":
            plot["_compare"] = "pmf"
            run["_pmf_axis"] = axes[0][0]
            run["_measure"] = summary.get("measure", "unknown")
    else:
        run["_plots"].append({"id": f"{run['id']}-{prefix}", "kind": "heatmap", "title": f"{run['label']}: 2D {'bias FES' if is_meta else 'FES'}",
                               "xlabel": axes[0][0], "ylabel": axes[1][0], "xedges": edges[0] * axes[0][1],
                               "yedges": edges[1] * axes[1][1], "z": f * factor, "mask": mask,
                               "colorbar": f"Free energy ({energy_unit})", "caption": caption})
        if np.isfinite(error).any() and not is_meta:
            run["_plots"].append({**run["_plots"][-1], "id": f"{run['id']}-{prefix}-error",
                                   "title": f"{run['label']}: FES uncertainty", "z": error * factor,
                                   "mask": mask | ~np.isfinite(error), "colorbar": f"Standard error ({energy_unit})",
                                   "caption": "Saved bootstrap standard error (1 SE). Missing uncertainty is masked, not zero."})
    if "raw_count" in data:
        count = np.asarray(data["raw_count"], float)
        if count.shape != f.shape or np.any(count[np.isfinite(count)] < 0):
            raise ValueError("Raw counts must match the FES and be nonnegative")
        if dimension == 1:
            _line(run, prefix + "-counts", f"{run['label']}: sampling coverage", axes[0][0], "Saved sample count",
                  [{"label": "Raw count", "x": x, "y": count}], "Raw saved frames are correlated and are not independent sample counts.")
        else:
            run["_plots"].append({"id": f"{run['id']}-{prefix}-counts", "kind": "heatmap", "title": f"{run['label']}: sampling coverage",
                                   "xlabel": axes[0][0], "ylabel": axes[1][0], "xedges": edges[0] * axes[0][1],
                                   "yedges": edges[1] * axes[1][1], "z": count, "colorbar": "Saved sample count",
                                   "caption": "Actual raw counts, including zero-count bins. Counts do not measure statistical independence."})
    if "overlap" in data:
        overlap = np.asarray(data["overlap"], float)
        if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
            raise ValueError("Overlap must be a square stream matrix")
        stream_edges = np.arange(overlap.shape[0] + 1) - 0.5
        run["_plots"].append({"id": f"{run['id']}-{prefix}-overlap", "kind": "heatmap",
                               "title": f"{run['label']}: MBAR overlap", "xlabel": "Sampling stream j", "ylabel": "Sampling stream i",
                               "xedges": stream_edges, "yedges": stream_edges, "z": overlap.T,
                               "colorbar": "Overlap O(i,j)", "caption": "Saved MBAR overlap matrix. Stream indices can include independent replicas of the same window."})
    if "effective_samples_per_stream" in data:
        ess = np.asarray(data["effective_samples_per_stream"], float)
        if ess.ndim != 1:
            raise ValueError("Effective sample counts must be a 1D stream array")
        _line(run, prefix + "-ess", f"{run['label']}: effective samples", "Sampling stream", "Estimated effective samples",
              [{"label": "ESS", "x": np.arange(len(ess)), "y": ess, "style": "points"}],
              "Correlation-adjusted effective sample size, estimated by the existing analysis. This is a sampling diagnostic, not proof of equilibration.")
        run["metrics"][f"{prefix}_minimum_stream_ess"] = float(np.nanmin(ess)) if len(ess) else None
    from .report_sampling import collect_sampling
    collect_sampling(run, path, summary, axes, energy_unit)


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
                           "rate_standard_error": run["_rate"]["standard_error"], "k_qtst": run["_rate"]["qtst"]})
    for key in ("free_energy_barrier_hartree", "barrier_standard_error_hartree", "xi_star", "measure"):
        if key in pmf:
            run["metrics"][key] = pmf[key]
    if rate is None:
        _finding(run, "rate_withheld", "The saved workflow withholds the thermal rate. Any diagnostic estimate remains a candidate and is excluded from default rate comparisons.")
    if error is None:
        _finding(run, "rate_uncertainty_unavailable", "Complete saved rate uncertainty is unavailable; conditional recrossing error is not substituted.")
    if _number(summary.get("maximum_endpoint_energy_error_kbt")) is not None and summary["maximum_endpoint_energy_error_kbt"] > 0.1:
        _finding(run, "child_energy_error", "Saved child endpoint energy error exceeds 0.1 kBT per bead; inspect the timestep.")


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
    summary = run["_summary"] = _json(run, summary_path)
    config_path, system_path = path / "input" / "resolved.json", path / "input" / "system.json"
    config = run["_config"] = _json(run, config_path) if config_path.is_file() else {}
    run["_system"] = _json(run, system_path) if system_path.is_file() else config.get("system", {})
    run["task"] = summary.get("task", config.get("task", "unknown"))
    run["status"] = summary.get("status", "not_recorded")
    run["converged"] = summary.get("converged")
    dt = _number(config.get("dt_au"))
    reaction, collision = config.get("reaction", {}), config.get("collision", {})
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
    if run["task"] in ("rate", "surface_rate", "site_rate"):
        _rate_artifacts(run, energy_unit, length_unit)
        folder = path / ("pmf" if run["task"] == "site_rate" else "umbrella")
        if (folder / "fes.npz").exists():
            _guard(run, "umbrella sampling", lambda: _fes_artifacts(run, folder, energy_unit, length_unit))
        else:
            _skip(run, "umbrella sampling", "Saved umbrella/fes.npz or pmf/fes.npz is unavailable")
    elif (path / "fes.npz").exists():
        _guard(run, "free-energy surface", lambda: _fes_artifacts(run, path, energy_unit, length_unit))
    from .report_observables import collect_observables
    supplemental = collect_observables(path, summary, config, prefix=prefix, label=run["label"], energy_unit=energy_unit)
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
                    arrhenius_fit=False, kie=None, convergence=None, overwrite=False):
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
        "output": str(output), "settings": {"formats": list(formats), "dpi": dpi,
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
        report["figures"] = render_figures(plots, stage, formats=formats, dpi=dpi)
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
