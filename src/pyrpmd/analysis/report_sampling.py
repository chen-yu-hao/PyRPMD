"""Saved umbrella, metadynamics and adaptive-sampling figure adapters.

All statistics are conditional diagnostics of the saved coordinate chart.
Custom CV factories and input files are never imported or executed here.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from ..constants import AU_TIME_FS
from ..units import DIMS, parse_quantity, unit_factor
from .diagnostics import correlated_mean_diagnostics
from .reporting import _finding, _guard, _json, _line, _npz, _skip, _source, _table


def _sample_array(value, dimension):
    data = np.asarray(value, dtype=float)
    if data.ndim == 1 and dimension == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[1] != dimension or not len(data) or not np.isfinite(data).all():
        raise ValueError("Raw sampling streams must be nonempty finite (frames, CVs) arrays")
    return data


def _selected(items, limit):
    indices = np.unique(np.linspace(0, len(items) - 1, min(limit, len(items)), dtype=int))
    return [items[i] for i in indices]


def _read_adaptive_plan(run, path, *, require_frozen=False):
    """Read one immutable snapshot; frozen production metadata needs integrity."""
    cache = run.setdefault("_sampling_plan_cache", {})
    key = str(Path(path).resolve())
    if key not in cache:
        cache[key] = _json(run, path)
    data = cache[key]
    if "checksum" in data:
        import hashlib
        canonical = json.dumps({name: value for name, value in data.items() if name != "checksum"},
                               sort_keys=True, separators=(",", ":"), allow_nan=False)
        if hashlib.sha256(canonical.encode()).hexdigest() != data["checksum"]:
            raise ValueError("Adaptive plan checksum mismatch")
    elif require_frozen:
        raise ValueError("Adaptive plan lacks the checksum required for production centers")
    if require_frozen and (data.get("schema_version") != 1 or data.get("status") != "frozen"):
        raise ValueError("Adaptive production centers require a supported frozen plan")
    return data


def _checkpoint_metadata(run, path, summary, dimension):
    """Adaptive input grids describe pilots; only the frozen plan owns production."""
    plan_path = path / "adaptive-plan.json"
    adaptive = run["_config"].get("adaptive_umbrella", {})
    enabled = adaptive.get("enabled", bool(adaptive)) if isinstance(adaptive, dict) else bool(adaptive)
    expected_plan = enabled or bool(summary.get("adaptive_umbrella")) or (path / "adaptation").exists()
    if plan_path.exists():
        plan = _read_adaptive_plan(run, plan_path, require_frozen=True)
        centers = np.asarray(plan["centers"], dtype=float)
        stiffness = np.asarray(plan["stiffness"], dtype=float)
        if (centers.ndim != 2 or centers.shape[1] != dimension or not len(centers)
                or stiffness.shape != (dimension, dimension)
                or not np.isfinite(centers).all() or not np.isfinite(stiffness).all()):
            raise ValueError("Frozen adaptive center/stiffness dimensions are invalid")
        _finding(run, "frozen_umbrella_metadata",
                 "Production checkpoint centers and stiffness come from the checksummed frozen adaptive plan; pilot histories are excluded.", "info")
        return centers, stiffness
    if expected_plan:
        raise ValueError("Adaptive production requires adaptive-plan.json; the original input grid cannot identify production window centers")
    return _canonical_metadata(run["_config"], summary, dimension)


def _canonical_metadata(config, summary, dimension):
    """Decode numeric/unit declarations, without invoking make_cvs or a PES."""
    umbrella = config.get("umbrella", {})
    units = summary.get("units", ["dimensionless"] * dimension if config.get("task") == "rate" else [])
    dimensions = {"bohr": "length", "angstrom": "length", "radian": "angle", "hartree": "energy", "dimensionless": "dimensionless"}
    if len(units) != dimension or any(unit not in dimensions for unit in units):
        raise ValueError("Canonical CV units are required to decode checkpoint umbrella centers")
    dims = [DIMS[dimensions[unit]] for unit in units]
    specs = config.get("cvs", [])
    if isinstance(specs, dict):
        specs = [dict(spec, name=name) for name, spec in specs.items()]
    selected = umbrella.get("cvs")
    if isinstance(selected, str):
        selected = [selected]
    by_name = {spec.get("name"): spec for spec in specs}
    specs = [by_name.get(name, {}) for name in selected] if selected else specs
    declared = []
    for i, spec in enumerate(specs):
        unit = spec.get("unit", spec.get("units", units[min(i, dimension - 1)]))
        declared.extend(unit if isinstance(unit, (list, tuple)) else [unit])
    if len(declared) != dimension:
        declared = units
    scales = [1.0 if unit == "dimensionless" else unit_factor(unit, dim) for unit, dim in zip(declared, dims)]

    def component(value, j, atomic=False):
        factor = 1.0 if atomic or isinstance(value, str) else scales[j]
        return parse_quantity(value, dims[j]) * factor

    def center(values, atomic=False):
        if dimension == 1 and isinstance(values, (int, float, str)):
            values = [values]
        if len(values) != dimension:
            raise ValueError("Saved umbrella center dimensions do not match the FES")
        return [component(value, j, atomic) for j, value in enumerate(values)]

    if "centers_au" in umbrella:
        centers = [center(value, True) for value in umbrella["centers_au"]]
    elif "centers" in umbrella:
        centers = [center(value) for value in umbrella["centers"]]
    elif "grid" in umbrella:
        grid = umbrella["grid"]
        grid = [grid] if isinstance(grid, dict) else grid
        if len(grid) != dimension:
            raise ValueError("Saved umbrella grid dimensions do not match the FES")
        axes = []
        for j, axis in enumerate(grid):
            if isinstance(axis, dict):
                count = axis["count"]
                if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                    raise ValueError("Saved umbrella grid count must be a positive integer")
                axes.append(np.linspace(component(axis["start"], j), component(axis["stop"], j), count))
            else:
                axes.append([component(value, j) for value in axis])
        centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dimension)
    elif dimension == 1 and all(key in umbrella for key in ("start", "stop", "count")):
        count = umbrella["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Saved umbrella count must be a positive integer")
        centers = np.linspace(center(umbrella["start"])[0], center(umbrella["stop"])[0], count)[:, None]
    else:
        raise ValueError("Saved checkpoint centers are unavailable")
    centers = np.asarray(centers, float)
    if centers.ndim != 2 or centers.shape[1] != dimension or not np.isfinite(centers).all():
        raise ValueError("Saved umbrella centers are invalid")
    raw = umbrella.get("stiffness_au", umbrella.get("stiffness", umbrella.get("kappa")))
    if raw is None:
        raise ValueError("Saved checkpoint umbrella stiffness is unavailable")
    matrix = np.asarray(raw, dtype=object)
    if matrix.ndim == 0:
        matrix = np.zeros((dimension, dimension), dtype=object)
        np.fill_diagonal(matrix, raw)
    elif matrix.ndim == 1 and matrix.size == dimension:
        matrix = np.diag(matrix)
    if matrix.shape != (dimension, dimension):
        raise ValueError("Saved umbrella stiffness dimensions are invalid")
    stiffness = np.empty(matrix.shape)
    for i in range(dimension):
        for j in range(dimension):
            dim = tuple(e - a - b for e, a, b in zip(DIMS["energy"], dims[i], dims[j]))
            stiffness[i, j] = parse_quantity(matrix[i, j], dim)
    return centers, stiffness


def _stream_metadata(stream, dimension):
    stream["samples"] = _sample_array(stream["samples"], dimension)
    center = np.asarray(stream.get("center"), dtype=float)
    stiffness = np.asarray(stream.get("stiffness"), dtype=float)
    if center.shape == (dimension,) and stiffness.shape == (dimension, dimension):
        if not np.isfinite(center).all() or not np.isfinite(stiffness).all():
            raise ValueError("Saved umbrella center/stiffness is nonfinite")
        stream["center"], stream["stiffness"] = center, stiffness
    else:
        if "center" in stream or "stiffness" in stream:
            raise ValueError("Saved umbrella center/stiffness has inconsistent dimensions")
        stream["center"] = stream["stiffness"] = None
    return stream


def _hdf5_streams(run, path, dimension):
    import h5py

    _source(run, path)
    result = []
    with h5py.File(path, "r") as saved:
        if saved.attrs.get("energy_unit", "hartree") != "hartree":
            raise ValueError("Sampling HDF5 stiffness must use hartree")
        if "streams" not in saved:
            raise ValueError("Sampling HDF5 has no raw streams")
        if not isinstance(saved.get("streams", getlink=True), h5py.HardLink):
            raise ValueError("External HDF5 stream links are not supported")
        groups = saved["streams"]
        for name in sorted(groups, key=lambda key: int(key) if key.isdigit() else key):
            if not isinstance(groups.get(name, getlink=True), h5py.HardLink):
                raise ValueError("External HDF5 stream links are not supported")
            group = groups[name]
            data = {}
            for key in ("samples", "center", "stiffness"):
                if not isinstance(group.get(key, getlink=True), h5py.HardLink):
                    raise ValueError("External HDF5 data links are not supported")
                dataset = group[key]
                if dataset.is_virtual or dataset.external:
                    raise ValueError("External/virtual HDF5 sampling storage is not supported")
                data[key] = dataset[...]
            data.update(window=int(group.attrs["window"]), replica=int(group.attrs["replica"]))
            result.append(_stream_metadata(data, dimension))
    return result


def _umbrella_streams(run, path, summary, dimension):
    streams = []
    hdf5 = path / "fes.h5"
    if hdf5.exists():
        _guard(run, "raw HDF5 sampling", lambda: streams.extend(_hdf5_streams(run, hdf5, dimension)))
    if streams:
        return streams
    checkpoints = sorted((path / "checkpoints").glob("window_*_replica_*.npz"))
    if not checkpoints:
        return []
    centers = stiffness = None
    try:
        centers, stiffness = _checkpoint_metadata(run, path, summary, dimension)
    except (KeyError, TypeError, ValueError) as error:
        _finding(run, "umbrella_metadata_missing", f"Checkpoint traces remain available, but center/K plots are unavailable: {error}")
    for checkpoint in checkpoints:
        def read(checkpoint=checkpoint):
            match = re.fullmatch(r"window_(\d+)_replica_(\d+)\.npz", checkpoint.name)
            if match is None:
                raise ValueError("Unrecognized umbrella checkpoint filename")
            window, replica = map(int, match.groups())
            saved = _npz(run, checkpoint)
            stream = {"window": window, "replica": replica, "samples": saved["values"]}
            if centers is not None and window < len(centers):
                stream.update(center=centers[window], stiffness=stiffness)
            streams.append(_stream_metadata(stream, dimension))
        _guard(run, checkpoint.name, read)
    return streams


def _time(stream, config):
    samples = stream["samples"]
    clocks = stream.get("sample_steps")
    dt = config.get("sampling", {}).get("dt_au", config.get("dt_au"))
    if clocks is not None and isinstance(dt, (int, float)) and np.isfinite(dt) and dt > 0:
        clocks = np.asarray(clocks, float)
        if clocks.shape != (len(samples),) or not np.isfinite(clocks).all() or np.any(np.diff(clocks) <= 0):
            raise ValueError("Saved sampling steps are invalid")
        return clocks * dt * AU_TIME_FS, "Biased sampling time (fs)"
    return np.arange(1, len(samples) + 1), "Saved sample index"


def _traces(run, streams, axes, prefix, *, biased=False):
    chosen = _selected(streams, 6)
    selection = ", ".join(f"w{s.get('window', 0)}/r{s['replica']}" for s in chosen)
    caption = (f"Shows {len(chosen)} of {len(streams)} streams, evenly selected in sorted window/replica order: {selection}. "
               "All saved frames of the selected streams are retained. ")
    caption += ("The changing bias makes these nonstationary diagnostics; sampling time is not reaction time. " if biased else
                "Thermostatted sampling trajectories do not represent reaction kinetics. ")
    caption += "Linear-coordinate running means do not prove equilibration and are not circular averages."
    for j, (label, factor) in enumerate(axes):
        raw, running = [], []
        clocks = [_time(stream, run["_config"]) for stream in chosen]
        # Mixed clock availability must not combine fs and indices on one axis.
        common_clock = len({clock[1] for clock in clocks}) == 1
        xlabel = clocks[0][1] if common_clock else "Saved sample index"
        for stream, clock in zip(chosen, clocks):
            values = stream["samples"][:, j] * factor
            x = clock[0] if common_clock else np.arange(1, len(values) + 1)
            name = f"r{stream['replica']}" if biased else f"w{stream['window']}/r{stream['replica']}"
            raw.append({"label": name, "x": x, "y": values})
            running.append({"label": name, "x": x, "y": np.cumsum(values) / np.arange(1, len(values) + 1)})
        _line(run, f"{prefix}-trace-{j+1}", f"{run['label']}: CV {j+1} traces", xlabel, label, raw, caption)
        _line(run, f"{prefix}-running-mean-{j+1}", f"{run['label']}: CV {j+1} running mean", xlabel, label, running, caption)


def _umbrella_diagnostics(run, streams, axes, energy_unit, prefix):
    streams.sort(key=lambda stream: (stream["window"], stream["replica"]))
    identities = [(stream["window"], stream["replica"]) for stream in streams]
    if len(identities) != len(set(identities)):
        raise ValueError("Saved umbrella streams repeat a window/replica identity")
    groups = {}
    stream_records = []
    for stream in streams:
        groups.setdefault(stream["window"], []).append(stream)
        diagnostics = []
        for j, (_, factor) in enumerate(axes):
            values = stream["samples"][:, j] * factor
            item = correlated_mean_diagnostics(values)
            item["mean"] = float(values.mean())
            item["sample_variance"] = float(values.var(ddof=1)) if len(values) > 1 else None
            diagnostics.append(item)
        stream_records.append({"window": stream["window"], "replica": stream["replica"], "coordinates": diagnostics})
    windows = sorted(groups)
    window_records = []
    for window in windows:
        group = groups[window]
        samples = np.concatenate([stream["samples"] for stream in group])
        known = all(stream["center"] is not None for stream in group)
        if known and any(not np.array_equal(stream["center"], group[0]["center"]) or not np.array_equal(stream["stiffness"], group[0]["stiffness"]) for stream in group[1:]):
            raise ValueError("Replicas assigned to one window have inconsistent center/stiffness")
        window_records.append({"window": window, "samples": len(samples), "replicas": len(group),
            "center_atomic": group[0]["center"].tolist() if known else None,
            "stiffness_atomic": group[0]["stiffness"].tolist() if known else None,
            "mean": (samples.mean(axis=0) * [axis[1] for axis in axes]).tolist(),
            "sample_variance": (samples.var(axis=0, ddof=1) * np.square([axis[1] for axis in axes])).tolist() if len(samples) > 1 else [None] * len(axes)})
    run.setdefault("sampling_diagnostics", {})[prefix] = {
        "coordinate_labels": [axis[0] for axis in axes], "streams": stream_records, "windows": window_records,
        "uncertainty_scope": "Stream errors are conditional on estimated block length; replicas are pooled only for descriptive window distributions, never for a naive mean SE.",
    }
    run["metrics"][prefix + "_raw_streams"] = len(streams)
    run["metrics"][prefix + "_sampled_windows"] = len(windows)
    _traces(run, streams, axes, prefix)
    chosen = _selected(windows, 8)
    choice = f"Marginal histograms for {len(chosen)} of {len(windows)} windows, evenly selected: {chosen}. All window diagnostics are retained in report.json and the offset/variance plots."
    energy_factor = 1 / unit_factor(energy_unit, "energy")
    for j, (label, factor) in enumerate(axes):
        unit = label.rsplit("(", 1)[-1].rstrip(")") if "(" in label else "displayed coordinate unit"
        variance = [row["sample_variance"][j] for row in window_records]
        _line(run, f"{prefix}-variance-{j+1}", f"{run['label']}: CV {j+1} window variance", "Window index",
              f"Sample variance ({unit}^2)", [{"label": "Biased distribution variance", "x": windows, "y": variance, "style": "points"}],
              "All sampled windows. Replicas are pooled for descriptive sample variance; this is not uncertainty of the mean or the PMF. Circular branch-cut effects are not removed.")
        supported = [row for row in window_records if row["center_atomic"] is not None]
        if supported:
            centers = np.asarray([row["center_atomic"][j] * factor for row in supported])
            means = np.asarray([row["mean"][j] for row in supported])
            sd = np.sqrt(np.asarray([row["sample_variance"][j] for row in supported], float))
            _line(run, f"{prefix}-offset-{j+1}", f"{run['label']}: CV {j+1} window displacement", label,
                  f"Mean minus center ({unit})", [{"label": "Mean offset ± sample SD", "x": centers, "y": means - centers, "yerr": sd, "style": "points"}],
                  "All windows with saved centers. Bars are ±1 sample standard deviation of the biased distribution, not a standard error or confidence interval.",
                  hlines=[{"y": 0.0, "label": "Centered"}])
            stiffness = [row["stiffness_atomic"][j][j] * energy_factor / factor**2 for row in supported]
            _line(run, f"{prefix}-stiffness-{j+1}", f"{run['label']}: CV {j+1} umbrella stiffness", label,
                  f"K[{j+1},{j+1}] ({energy_unit}/{unit}^2)", [{"label": "Diagonal K", "x": centers, "y": stiffness, "style": "points"}],
                  "Diagonal physical umbrella stiffness from V=1/2 ΔsᵀKΔs. Cross terms are omitted from this plot and retained in report.json; no bead factor is applied.")
        else:
            _skip(run, f"CV {j+1} umbrella center/stiffness", "Raw samples contain no safely decodable center and stiffness metadata")
        distributions = [np.concatenate([stream["samples"][:, j] for stream in groups[window]]) * factor for window in chosen]
        lower = min(values.min() for values in distributions)
        upper = max(values.max() for values in distributions)
        if lower == upper:
            lower, upper = lower - max(abs(lower) * .01, 1e-6), upper + max(abs(upper) * .01, 1e-6)
        edges = np.linspace(lower, upper, 49)
        histograms = [{"label": f"Window {window}", "x": (edges[:-1] + edges[1:]) * .5,
                       "y": np.histogram(values, bins=edges, density=True)[0]} for window, values in zip(chosen, distributions)]
        _line(run, f"{prefix}-distributions-{j+1}", f"{run['label']}: CV {j+1} umbrella distributions", label,
              f"Probability density ({unit}^-1)", histograms,
              choice + " Marginal overlap alone does not establish joint overlap; use the saved MBAR overlap matrix. Histogram bins are for display only and do not recompute the PMF.")


def _metadynamics(run, path, axes, energy_unit, prefix):
    heights, biases = [], []
    factor = 1 / unit_factor(energy_unit, "energy")
    for file in sorted(path.glob("replicas/*/HILLS.csv")):
        def read(file=file):
            data = _table(run, file)
            time = np.asarray(data["time_au"], float) * AU_TIME_FS
            if time.ndim != 1 or not len(time) or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
                raise ValueError("HILLS time must be finite and increasing")
            for key, target in (("height_hartree", heights), ("bias_before_hartree", biases)):
                values = np.asarray(data[key], float)
                if values.shape != time.shape or not np.isfinite(values).all():
                    raise ValueError("HILLS energies must be finite and match the time array")
                target.append({"label": f"Replica {file.parent.name}", "x": time, "y": values * factor})
        _guard(run, str(file.relative_to(path)), read)
    caption = "Saved deposited hill history. This is biased sampling time, not reaction time; decreasing hill heights do not establish a converged FES or a kinetic rate."
    for name, title, values in (("hill-height", "deposited hill height", heights), ("hill-bias", "bias before deposition", biases)):
        if values:
            _line(run, prefix + "-" + name, f"{run['label']}: {title}", "Biased sampling time (fs)", f"Energy ({energy_unit})", values, caption)
        else:
            _skip(run, title, "Saved replicas/*/HILLS.csv is unavailable or invalid")
    streams = []
    for checkpoint in sorted((path / "checkpoints").glob("metadynamics_*.npz")):
        def read(checkpoint=checkpoint):
            data = _npz(run, checkpoint)
            match = re.fullmatch(r"metadynamics_(\d+)\.npz", checkpoint.name)
            if not match:
                raise ValueError("Unrecognized metadynamics checkpoint name")
            streams.append({"replica": int(match[1]), "samples": _sample_array(data["samples"], len(axes)),
                            "sample_steps": data.get("sample_steps")})
        _guard(run, checkpoint.name, read)
    if streams:
        _traces(run, streams, axes, prefix, biased=True)
    else:
        _skip(run, "metadynamics CV traces", "Saved metadynamics checkpoint samples are unavailable")


def _adaptive(run, path, prefix):
    plan = path / "adaptive-plan.json"
    if not plan.exists():
        return
    data = _read_adaptive_plan(run, plan)
    rounds = data.get("rounds", [])
    run["metrics"][prefix + "_adaptive_status"] = data.get("status")
    if not rounds:
        _skip(run, "adaptive progress", "Adaptive plan contains no completed diagnostic rounds")
        return
    x = np.asarray([row["round"] for row in rounds], float)
    if not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("Adaptive diagnostic round indices must increase")
    caption = "Saved adaptation pilots only; these samples are discarded before production. Pilot overlap/ESS targets do not certify physical or production convergence."
    metrics = (("overlap", "Spanning-tree overlap", [row.get("spanning_overlap") for row in rounds]),
               ("ess", "Minimum effective samples per window", [min(row["effective_samples_per_window"], default=np.nan) for row in rounds]),
               ("windows", "Umbrella windows", [len(row["centers"]) for row in rounds]))
    for name, label, values in metrics:
        _line(run, f"{prefix}-adaptive-{name}", f"{run['label']}: adaptive {name}", "Pilot round", label,
              [{"label": label, "x": x, "y": values}], caption)


def collect_sampling(run, path, summary, axes, energy_unit):
    """Append optional sampling plots, descriptive statistics and provenance."""
    path = Path(path)
    prefix = "umbrella" if path != Path(run["path"]) else "sampling"
    if summary.get("task") == "metadynamics":
        _guard(run, "metadynamics sampling history", lambda: _metadynamics(run, path, axes, energy_unit, prefix))
    else:
        def umbrella():
            streams = _umbrella_streams(run, path, summary, len(axes))
            if streams:
                _umbrella_diagnostics(run, streams, axes, energy_unit, prefix)
            else:
                _skip(run, "raw umbrella sampling", "Neither fes.h5 streams nor saved window checkpoints are available")
        _guard(run, "raw umbrella diagnostics", umbrella)
    _guard(run, "adaptive sampling progress", lambda: _adaptive(run, path, prefix))
