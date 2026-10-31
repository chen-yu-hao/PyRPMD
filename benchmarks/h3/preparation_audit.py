"""Compact isolated-diatom audits for the fixed preparation sensitivity study.

The original study initializer and ``run(audit=True)`` perform all sampling
and native dynamics. Only the parent's submitted audit callable is temporarily
rebound; spawned workers import this module and call the original audit worker.
The binding is always restored. No production or core source is modified.

The primary full-hold means come from native, every-step trapezoidal integrals.
A 0.2 fs trace independently checks those means and supplies a final-50-fs
window. Uncertainty is calculated across independent trajectory means, never
across correlated time points. Only a 5 fs trace and exact important endpoints
are serialized, in a compact column/record format; no coordinates are dumped.

Run the registered twelve-case design after the native implementation is frozen::

    PYTHONPATH=src:. python -m benchmarks.h3.preparation_audit \
        --output runs/state_preparation_study/isolated_audit --workers 8

``--dry-run`` prints the complete plan without initializing a PES or writing
files. Small smoke runs must use a separate output directory and an explicitly
smaller ``--samples``; they must not be pooled with the planned 512-draw cases.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import threading

import numpy as np

from benchmarks.h3 import state_preparation_study as study


_ORIGINAL_AUDIT_ONE = study._audit_one
DENSE_STRIDE_FS = 0.2
SPARSE_STRIDE_FS = 5.0
FINAL_WINDOW_FS = 50.0
HOLD_TIME_FS = 100.0
POINTS = {"v0:0.55": (0, 0.55), "v1:0.90": (1, 0.90), "v1:1.10": (1, 1.10)}
VARIANTS = {
    "direct_paper": ("direct", "paper", 0.0),
    "direct_none": ("direct", "none", 0.0),
    "switch50_none": ("switch", "none", 50.0),
    "switch200_none": ("switch", "none", 200.0),
}
TRACE_UNITS = {
    "time_fs": "fs", "lambda": "dimensionless",
    "kinetic_per_bead_hartree": "hartree",
    "harmonic_potential_per_bead_hartree": "hartree",
    "full_potential_per_bead_hartree": "hartree",
    "spring_per_bead_hartree": "hartree",
    "ring_hamiltonian_per_bead_hartree": "hartree",
    "switching_work_per_bead_hartree": "hartree",
    "energy_balance_residual_per_bead_hartree": "hartree",
    "primitive_internal_per_bead_hartree": "hartree",
    "centroid_bond_bohr": "bohr",
    "centroid_radial_momentum_au": "electron_mass * bohr / atomic_time",
    "bond_spread_bohr": "bohr", "minimum_bond_bohr": "bohr",
    "maximum_bond_bohr": "bohr", "centroid_harmonic_energy_hartree": "hartree",
    "lambda_rate_per_atomic_time": "atomic_time^-1",
    "hold_energy_drift_per_bead_hartree": "hartree",
}
STATISTICAL_SCOPE = (
    "Each independent trajectory ID contributes one equally weighted time mean. "
    "SEM uses the sample standard deviation (ddof=1) divided by sqrt(n), and is "
    "null for n<2. Means are conditional on successful preparation/reduction; "
    "all failures remain in attempt counts. Time points are not independent samples."
)


def _times(trace):
    if not trace:
        raise ValueError("A nonempty dense trace is required")
    times = np.asarray([row["time_fs"] for row in trace], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Trace times must be finite and strictly increasing")
    return times


def _time_tolerance(times):
    return 64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(times))))


def time_window_average(trace, start_fs, end_fs):
    """Trapezoidal mean of a piecewise-linear trace on an explicit interval.

    Boundary interpolation is allowed only inside the observed time span, and
    is reported. Missing/nonfinite values inside the interval fail explicitly;
    they are never dropped to manufacture an apparently complete mean.
    """
    times = _times(trace)
    start, end = float(start_fs), float(end_fs)
    tolerance = _time_tolerance(times)
    if not math.isfinite(start+end) or not end > start:
        raise ValueError("A finite positive averaging interval is required")
    if start < times[0]-tolerance or end > times[-1]+tolerance:
        raise ValueError("Averaging interval extends beyond the dense trace")
    # Native times can differ from their nominal fs endpoints by a few ulps.
    start = max(start, float(times[0]))
    end = min(end, float(times[-1]))
    for which in ("start", "end"):
        boundary = start if which == "start" else end
        nearest = int(np.argmin(np.abs(times-boundary)))
        if abs(times[nearest]-boundary) <= tolerance:
            if which == "start":
                start = float(times[nearest])
            else:
                end = float(times[nearest])
    inside = (times > start+tolerance) & (times < end-tolerance)
    nodes = np.r_[start, times[inside], end]
    # Include bracketing rows when a boundary lies between observed nodes.
    left = max(0, int(np.searchsorted(times, start, side="right"))-1)
    right = min(len(times)-1, int(np.searchsorted(times, end, side="left")))
    values = {}
    columns = tuple(key for key in trace[0] if key != "time_fs")
    if any(set(row) != set(trace[0]) for row in trace):
        raise ValueError("Dense trace columns changed within a trajectory")
    for key in columns:
        raw = [row[key] for row in trace[left:right+1]]
        if any(value is None for value in raw):
            raise ValueError(f"Missing {key} inside the averaging interval")
        observed = np.asarray(raw, dtype=float)
        if not np.isfinite(observed).all():
            raise ValueError(f"Nonfinite {key} inside the averaging interval")
        sampled = np.interp(nodes, times[left:right+1], observed)
        integral = np.sum(np.diff(nodes) * (sampled[:-1]+sampled[1:]) / 2)
        values[key] = float(integral / (end-start))
    exact = lambda x: bool(np.any(np.abs(times-x) <= tolerance))
    return {"source": "dense_trace_trapezoid", "start_fs": start, "end_fs": end,
            "duration_fs": end-start,
            "dense_points": int(np.count_nonzero((times >= start-tolerance) & (times <= end+tolerance))),
            "integration_nodes": len(nodes),
            "interpolated_boundaries": {"start": not exact(start), "end": not exact(end)},
            "values": values}


def sparse_trace(trace, switch_time_fs, *, stride_fs=SPARSE_STRIDE_FS):
    """Keep native records near the sparse grid, plus exact required endpoints."""
    times = _times(trace)
    stride = float(stride_fs)
    if not math.isfinite(stride) or stride <= 0:
        raise ValueError("Sparse trace stride must be positive and finite")
    tolerance = _time_tolerance(times)
    selected = set()
    for endpoint in (0.0, float(switch_time_fs), float(times[-1])):
        index = int(np.argmin(np.abs(times-endpoint)))
        if abs(times[index]-endpoint) > tolerance:
            raise ValueError(f"The native trace lacks required endpoint {endpoint} fs")
        selected.add(index)
    offsets = []
    for target in np.arange(0.0, times[-1]+tolerance, stride):
        index = int(np.argmin(np.abs(times-target)))
        offsets.append(float(abs(times[index]-target)))
        selected.add(index)
    columns = list(trace[0])
    unknown = set(columns)-set(TRACE_UNITS)
    if unknown or any(set(row) != set(columns) for row in trace):
        raise ValueError(f"Unsupported or inconsistent trace columns: {sorted(unknown)}")
    return {"columns": columns, "units": {key: TRACE_UNITS[key] for key in columns},
            "records": [[trace[index][key] for key in columns] for index in sorted(selected)]}, {
                "dense_records": len(trace), "sparse_records": len(selected),
                "sparse_stride_fs": stride,
                "maximum_sparse_grid_offset_fs": max(offsets, default=0.0),
                "mandatory_endpoints_fs": sorted({0.0, float(switch_time_fs), float(times[-1])}),
                "selection": "Nearest recorded point to each sparse-grid time; exact native 0/ramp-end/final records retained; no coordinate arrays",
            }


def reduce_audit_row(row):
    """Reduce one successful worker result before it crosses the process pipe."""
    if row["status"] != "prepared":
        return row
    preparation = row["preparation_study"]
    dense = preparation.pop("trace")
    native = preparation["switch"]
    if native["status"] != "completed":
        raise ValueError("A successful audit must have completed native preparation")
    start = float(native["switch_time_fs"])
    duration = float(native["hold_time_fs"])
    end = start+duration
    if duration <= 0 or not math.isclose(float(native["accepted_hold_duration_fs"]), duration,
                                        rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("The requested hold interval was not completed")
    native_values = dict(native["hold_time_average"])
    if not native_values or any(value is None or not math.isfinite(value) for value in native_values.values()):
        raise ValueError("Native hold means must be finite and complete")
    full = time_window_average(dense, start, end)
    final = time_window_average(dense, max(start, end-FINAL_WINDOW_FS), end)
    final["requested_window_fs"] = FINAL_WINDOW_FS
    preparation["time_averages"] = {
        "hold": {"source": "native_every_step_trapezoid", "start_fs": start, "end_fs": end,
                 "duration_fs": duration, "native_dt_fs": native["dt_fs"],
                 "native_steps": round(duration/native["dt_fs"]), "values": native_values},
        "hold_trace_check": full, "final_50fs": final,
    }
    preparation["trace"], sampling = sparse_trace(dense, start)
    sampling["dense_stride_fs"] = native["trace_stride_fs"]
    sampling["averaging_before_thinning"] = True
    preparation["trace_sampling"] = sampling
    return row


def audit_one(trajectory_id):
    """Top-level spawn-picklable replacement for the parent's submitted task."""
    row = _ORIGINAL_AUDIT_ONE(trajectory_id)
    try:
        return reduce_audit_row(row)
    except Exception as exc:
        # A reducer error is explicit and retained; never serialize its huge
        # dense trace or silently discard that trajectory from the denominator.
        row.get("preparation_study", {}).pop("trace", None)
        row.update(status="failed", preparation_status="prepared", failure_stage="audit_reduction",
                   reason=f"{type(exc).__name__}: {exc}")
        return row


@contextmanager
def audit_callable_binding():
    """Mutate only the parent's scheduling reference, and restore on any exit."""
    if multiprocessing.current_process().name != "MainProcess" or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Audit scheduling must run in the parent process/main thread")
    previous = study._audit_one
    if previous is not _ORIGINAL_AUDIT_ONE:
        raise RuntimeError("The study audit callable was already replaced")
    study._audit_one = audit_one
    try:
        yield
    finally:
        study._audit_one = previous


def _statistics(values):
    values = np.asarray(values, dtype=float)
    return {"n": len(values), "mean": float(values.mean()) if len(values) else None,
            "sem": float(values.std(ddof=1)/math.sqrt(len(values))) if len(values) > 1 else None}


def summarize_time_averages(rows):
    """Aggregate trajectory means with equal weight, independently of trace length."""
    attempts = prepared = 0
    windows, samples, differences = {}, {}, {}
    for row in rows:
        attempts += 1
        if row["status"] != "prepared":
            continue
        prepared += 1
        averages = row["preparation_study"]["time_averages"]
        for name, window in averages.items():
            metadata = {key: window[key] for key in ("source", "start_fs", "end_fs", "duration_fs")}
            if name in windows:
                expected = windows[name]
                if expected["source"] != metadata["source"] or any(
                        not math.isclose(expected[key], metadata[key], rel_tol=1e-12, abs_tol=1e-10)
                        for key in ("start_fs", "end_fs", "duration_fs")):
                    raise ValueError("Cannot pool trajectory means with different time windows")
            else:
                windows[name] = metadata
                samples[name] = {}
            for column, value in window["values"].items():
                if not math.isfinite(value):
                    raise ValueError("Nonfinite trajectory mean in audit aggregation")
                samples[name].setdefault(column, []).append(value)
        native, traced = averages["hold"]["values"], averages["hold_trace_check"]["values"]
        for column in native.keys() & traced.keys():
            differences.setdefault(column, []).append(native[column]-traced[column])
    for name, metadata in windows.items():
        metadata["values"] = {column: {**_statistics(values), "unit": TRACE_UNITS[column]}
                              for column, values in samples[name].items()}
    return {"schema": "pyrpmd.preparation-audit.time-averages.v1",
            "attempts": attempts, "prepared": prepared, "failed": attempts-prepared,
            "windows": windows,
            "native_minus_trace_hold": {column: {**_statistics(values), "unit": TRACE_UNITS[column]}
                                        for column, values in differences.items()},
            "statistics_scope": STATISTICAL_SCOPE}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_wrapper(snapshot):
    if _sha(snapshot["path"]) != snapshot["sha256"]:
        raise RuntimeError("Audit wrapper changed during the run; existing ledgers are retained")


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def planned_cases(*, samples=512, id_offset=910000, points=None, variants=None):
    """Return the fixed physical design, optionally selecting cases for a smoke run."""
    if (isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0
            or isinstance(id_offset, bool) or not isinstance(id_offset, int) or id_offset < 0):
        raise ValueError("samples must be a positive integer and id_offset a nonnegative integer")
    points = list(POINTS if points is None else points)
    variants = list(VARIANTS if variants is None else variants)
    if (not points or not variants or len(set(points)) != len(points) or len(set(variants)) != len(variants)
            or not set(points) <= POINTS.keys() or not set(variants) <= VARIANTS.keys()):
        raise ValueError("Select nonempty, unique registered point and variant names")
    cases = []
    for point in points:
        v, energy = POINTS[point]
        for variant in variants:
            kind, conditioning, switch = VARIANTS[variant]
            config = study.LaunchConfig(v=v, total_energy_ev=energy, method="ne_rpmd", beads=32,
                samples=samples, id_offset=id_offset, bmax=3.5 if v == 0 else 4.0)
            spec = study.PreparationSpec(kind=kind, conditioning=conditioning, switch_time_fs=switch,
                hold_time_fs=HOLD_TIME_FS, preparation_dt_fs=0.02, trace_stride_fs=DENSE_STRIDE_FS)
            cases.append({"name": f"v{v}_E{energy:.2f}_{variant}".replace(".", "p"),
                          "point": point, "variant": variant, "config": asdict(config),
                          "preparation": asdict(spec)})
    return cases


def run_cases(output, *, workers=8, samples=512, id_offset=910000, points=None, variants=None):
    """Run selected isolated cases, retaining wrapper and per-case provenance."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    cases = planned_cases(samples=samples, id_offset=id_offset, points=points, variants=variants)
    snapshot = {"path": str(Path(__file__).resolve()), "sha256": _sha(__file__)}
    _verify_wrapper(snapshot)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "pyrpmd.preparation-audit.manifest.v1", "status": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(), "wrapper_source": snapshot,
        "workers": workers, "cases": cases, "completed_cases": [],
        "trace_policy": {"dense_stride_fs": DENSE_STRIDE_FS, "sparse_stride_fs": SPARSE_STRIDE_FS,
                         "required_endpoints": "0, ramp endpoint, final", "column_units": TRACE_UNITS},
        "energy_definitions": {
            "native_potential_and_hamiltonian": "Native harmonic/full potentials and H_lambda/P include the common energy_zero_hartree in each row's switch summary",
            "primitive_internal_per_bead_hartree": "K + Vfull - energy_zero - S per bead, including during the ramp; not the driven Hamiltonian",
            "centroid_harmonic_energy_hartree": "pbar^2/(2*mu) + mu*omega^2*(rbar-req)^2/2; zero at the harmonic minimum",
            "work_balance": "H_lambda/P - initial_harmonic_H/P - switching_work/P",
        }, "statistics_scope": STATISTICAL_SCOPE,
        "source_verification": "This wrapper is hashed before/after each case; study.run separately freezes and verifies its sources and native library",
        "scope": "Isolated collinear BC only; incoming atom is unchanged and no collision is propagated; experimental smooth switching does not establish a pure vibrational state"}
    manifest_path = output/"audit_manifest.json"
    _write_json(manifest_path, manifest)
    try:
        with audit_callable_binding():
            for case in cases:
                _verify_wrapper(snapshot)
                case_output = output/case["name"]
                study.run(study.LaunchConfig(**case["config"]),
                          study.PreparationSpec(**case["preparation"]),
                          case_output, workers=workers, audit=True)
                _verify_wrapper(snapshot)
                with (case_output/"trajectories.jsonl").open() as stream:
                    means = summarize_time_averages(json.loads(line) for line in stream if line.strip())
                means["wrapper_source"] = snapshot
                _write_json(case_output/"audit_time_averages.json", means)
                manifest["completed_cases"].append(case["name"])
                _write_json(manifest_path, manifest)
        _verify_wrapper(snapshot)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["wrapper_unchanged_after_run"] = _sha(snapshot["path"]) == snapshot["sha256"]
        changed_after_completion = not manifest["wrapper_unchanged_after_run"] and manifest["status"] == "completed"
        if changed_after_completion:
            manifest.update(status="failed", error="Audit wrapper changed after final case verification")
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, manifest)
        if changed_after_completion:
            raise RuntimeError(manifest["error"])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--id-offset", type=int, default=910000)
    parser.add_argument("--point", choices=POINTS, action="append")
    parser.add_argument("--variant", choices=VARIANTS, action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    options = dict(samples=args.samples, id_offset=args.id_offset, points=args.point, variants=args.variant)
    if args.dry_run:
        print(json.dumps(planned_cases(**options), indent=2, allow_nan=False))
    elif args.output is None:
        parser.error("--output is required unless --dry-run is used")
    else:
        run_cases(args.output, workers=args.workers, **options)


if __name__ == "__main__":
    # Submit an importable module-level callable, including when started -m.
    from benchmarks.h3.preparation_audit import main as module_main
    module_main()
