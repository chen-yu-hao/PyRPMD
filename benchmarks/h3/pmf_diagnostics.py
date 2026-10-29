"""Fixed-surface MBAR diagnosis of a saved H3 rate calculation.

This reuses the existing rate-flux ensemble and recrossing coefficient. It is
an estimator comparison on the same data, not an independent validation or a
converged physical rate. No input, checkpoint, or production result is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from pyrpmd.analysis.fes import (
    _block_resample,
    _histogram,
    solve_mbar,
    statistical_inefficiency,
    umbrella_integration,
    umbrella_reduced_potentials,
)

HARTREE_KCAL_MOL = 627.5094740631


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bracket(grid, mask, point):
    """Require supported neighboring bin centers; never clamp or extrapolate."""
    exact = np.flatnonzero(np.isclose(grid, point, rtol=0, atol=1e-14))
    if len(exact):
        i = int(exact[0])
        return (i, i, 0.0) if not mask[i] else None
    right = int(np.searchsorted(grid, point))
    if right == 0 or right == len(grid) or mask[right - 1] or mask[right]:
        return None
    left = right - 1
    return left, right, float((point - grid[left]) / (grid[right] - grid[left]))


def _interpolate(values, bracket):
    left, right, fraction = bracket
    return float((1 - fraction) * values[left] + fraction * values[right])


def _histogram_barrier(x, weights, edges, beta, xi, min_count):
    mass, density, count, free, mask = _histogram(
        x[:, None], weights, [edges], beta, min_count=min_count
    )
    grid = 0.5 * (edges[1:] + edges[:-1])
    reference = _bracket(grid, mask, 0.0)
    surface = _bracket(grid, mask, xi)
    supported = reference is not None and surface is not None
    barrier = (
        _interpolate(free, surface) - _interpolate(free, reference)
        if supported
        else None
    )
    if reference is not None:
        free = free - _interpolate(free, reference)
    return {
        "barrier": barrier,
        "free": free,
        "grid": grid,
        "mask": mask,
        "mass": mass,
        "density": density,
        "count": count,
        "reference": reference,
        "surface": surface,
    }


def _statistics(values):
    a = np.asarray(values, float)
    if not len(a):
        return {"replicates": 0, "standard_error": None, "percentile_95_ci": None}
    return {
        "replicates": len(a),
        "standard_error": float(a.std(ddof=1)) if len(a) > 1 else None,
        "percentile_95_ci": np.quantile(a, [0.025, 0.975]).tolist(),
    }


def _unique_bias_states(centers, stiffness, counts):
    """Combine identical Hamiltonians, without combining trajectory streams."""
    indices, mapping, totals = [], [], []
    keys = {}
    for index, (center, matrix, count) in enumerate(zip(centers, stiffness, counts)):
        key = (center.tobytes(), matrix.tobytes())
        if key not in keys:
            keys[key] = len(indices)
            indices.append(index)
            totals.append(0)
        group = keys[key]
        mapping.append(group)
        totals[group] += int(count)
    return np.asarray(indices), np.asarray(totals), mapping


def _finite(value):
    """Strict portable JSON: unsupported numbers are null, never NaN."""
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_finite(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def analyze_run(run, *, bins=(50, 100, 150), bootstrap=40, seed=73019):
    """Return a JSON-compatible diagnosis, reading a completed rate run only.

    ``run`` may be the run directory or its enclosing benchmark directory.
    The saved HDF5 stream order is retained and checked against every original
    checkpoint (which may be an earlier exact prefix). Each bootstrap draw uses contiguous blocks within each stream,
    refits MBAR once, and retains paired endpoint and UI/MBAR differences.
    """
    import h5py

    started = time.perf_counter()
    run = Path(run).resolve()
    if not (run / "umbrella").is_dir() and (run / "run/umbrella").is_dir():
        run = run / "run"
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    bins = tuple(bins)
    if not bins or any(
        isinstance(b, bool) or not isinstance(b, int) or b < 3 for b in bins
    ):
        raise ValueError("bins must contain integers >= 3")
    summary = json.loads((run / "summary.json").read_text())
    config = json.loads((run / "input/resolved.json").read_text())
    pmf = summary["pmf"]
    xi = float(pmf["xi_star"])
    beta = float(config["beta"])
    if not np.isfinite([xi, beta]).all() or beta <= 0:
        raise ValueError("saved xi_star and beta must be finite; beta must be positive")
    min_count = int(config.get("analysis", {}).get("min_count", 1))
    h5_path = run / "umbrella/fes.h5"
    arrays, centers, stiffness, moments, fingerprints = [], [], [], [], []
    with h5py.File(h5_path, "r") as h:
        if h.attrs["measure"] != "rate_flux":
            raise ValueError("diagnostic requires the saved rate_flux sampling measure")
        limits = h["edges_0"][[0, -1]]
        groups = h["streams"]
        for stream_id, key in enumerate(sorted(groups, key=int)):
            if int(key) != stream_id:
                raise ValueError("HDF5 stream numbering is not contiguous")
            group = groups[key]
            window, replica = int(group.attrs["window"]), int(group.attrs["replica"])
            checkpoint = (
                run
                / "umbrella/checkpoints"
                / (f"window_{window:05d}_replica_{replica:04d}.npz")
            )
            values = np.asarray(group["samples"])
            if values.ndim != 2 or values.shape[1] != 1 or len(values) < 2:
                raise ValueError("saved samples must be nonempty scalar-CV streams")
            if not np.isfinite(values).all():
                raise ValueError("saved samples contain nonfinite values")
            with np.load(checkpoint, allow_pickle=False) as saved:
                checkpoint_samples = len(saved["values"])
                if not np.array_equal(values[:checkpoint_samples], saved["values"]):
                    raise ValueError(f"HDF5/checkpoint samples differ: {checkpoint}")
                fingerprints.append(str(saved["fingerprint"]))
            center = np.asarray(group["center"])
            matrix = np.asarray(group["stiffness"])
            if center.shape != (1,) or matrix.shape != (1, 1) or matrix[0, 0] <= 0:
                raise ValueError(
                    "saved umbrella parameters have invalid shapes or stiffness"
                )
            x = values[:, 0]
            mean, sigma = float(x.mean()), float(x.std())
            z = (x - mean) / sigma if sigma > 0 else np.full_like(x, np.nan)
            g = statistical_inefficiency(x)
            moments.append(
                {
                    "stream": stream_id,
                    "window": window,
                    "replica": replica,
                    "center": float(center[0]),
                    "stiffness_hartree": float(matrix[0, 0]),
                    "samples": len(x),
                    "mean": mean,
                    "standard_deviation": sigma,
                    "checkpoint_samples": checkpoint_samples,
                    "skewness": float(np.mean(z**3)),
                    "excess_kurtosis": float(np.mean(z**4) - 3),
                    "statistical_inefficiency": g,
                    "block_length": int(np.ceil(g)),
                    "estimated_effective_samples": len(x) / g,
                }
            )
            arrays.append(x)
            centers.append(center)
            stiffness.append(matrix)
    centers, stiffness = np.asarray(centers), np.asarray(stiffness)
    counts = np.asarray([len(a) for a in arrays])
    offsets = np.concatenate(([0], np.cumsum(counts)))
    x = np.concatenate(arrays)
    u = umbrella_reduced_potentials(x[:, None], centers, stiffness, beta)
    unique, fit_counts, state_map = _unique_bias_states(centers, stiffness, counts)
    fit = solve_mbar(u[unique], fit_counts)
    if not fit.converged or len(np.unique(fit.components)) != 1:
        raise ValueError(
            "saved ensemble has unresolved or disconnected MBAR normalization"
        )
    weight_difference = 0.0
    if len(unique) != len(counts):
        original = solve_mbar(u, counts)
        weight_difference = float(np.max(np.abs(original.weights - fit.weights)))
        if (
            not original.converged
            or len(np.unique(original.components)) != 1
            or weight_difference > 1e-9
        ):
            raise ValueError(
                "aggregation of identical umbrella Hamiltonians failed equivalence check"
            )
    u = u[unique]
    with np.load(run / "rate_pmf.npz", allow_pickle=False) as saved:
        ui_grid = np.asarray(saved["xi"])
    if not ui_grid[0] <= 0 <= xi <= ui_grid[-1]:
        raise ValueError(
            "saved UI grid does not bracket the same reference and surface"
        )
    ui_curve, _ = umbrella_integration(
        arrays, centers[:, 0], stiffness[:, 0, 0], beta, ui_grid
    )
    ui_barrier = float(
        np.interp(xi, ui_grid, ui_curve) - np.interp(0, ui_grid, ui_curve)
    )
    if not np.isclose(
        ui_barrier, pmf["free_energy_barrier_hartree"], rtol=1e-10, atol=1e-12
    ):
        raise ValueError("recomputed UI barrier differs from the saved result")
    edges = [np.linspace(*limits, b + 1) for b in bins]
    base = [_histogram_barrier(x, fit.weights, e, beta, xi, min_count) for e in edges]
    draws = [[] for _ in bins]
    paired_differences = [[] for _ in bins]
    ui_draws = []
    failures = {"mbar": 0, "ui": 0}
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap):
        indices = np.concatenate(
            [
                _block_resample(np.arange(count), moment["block_length"], rng) + offset
                for count, offset, moment in zip(counts, offsets[:-1], moments)
            ]
        )
        xb = x[indices]
        mb = solve_mbar(u[:, indices], fit_counts)
        if not mb.converged or len(np.unique(mb.components)) != 1:
            failures["mbar"] += 1
            continue
        ui_value = None
        try:
            ub, _ = umbrella_integration(
                [xb[start:stop] for start, stop in zip(offsets[:-1], offsets[1:])],
                centers[:, 0],
                stiffness[:, 0, 0],
                beta,
                ui_grid,
            )
            ui_value = float(np.interp(xi, ui_grid, ub) - np.interp(0, ui_grid, ub))
            ui_draws.append(ui_value)
        except ValueError:
            failures["ui"] += 1
        for i, edge in enumerate(edges):
            barrier = _histogram_barrier(xb, mb.weights, edge, beta, xi, min_count)[
                "barrier"
            ]
            if barrier is not None:
                draws[i].append(barrier)
                if ui_value is not None:
                    paired_differences[i].append(barrier - ui_value)
    recorded_rate = summary.get("rate_cm3_s")
    estimate_rate = summary.get("rate_estimate_cm3_s")
    per_bins = []
    for nbin, edge, result, boot, difference in zip(
        bins, edges, base, draws, paired_differences
    ):
        barrier = result["barrier"]
        factor = (
            float(np.exp(-beta * (barrier - ui_barrier)))
            if barrier is not None
            else None
        )
        endpoints = {}
        for name in ("reference", "surface"):
            bracket = result[name]
            support = []
            if bracket is not None:
                for index in sorted(set(bracket[:2])):
                    select = (x >= edge[index]) & (x < edge[index + 1])
                    local = fit.weights[select]
                    contributing = sum(
                        bool(select[start:stop].any())
                        for start, stop in zip(offsets[:-1], offsets[1:])
                    )
                    support.append(
                        {
                            "bin": index,
                            "center": result["grid"][index],
                            "raw_count": int(result["count"][index]),
                            "density": result["density"][index],
                            "weighted_effective_samples_without_time_correction": (
                                float(local.sum() ** 2 / (local @ local))
                                if len(local)
                                else 0.0
                            ),
                            "contributing_streams": contributing,
                        }
                    )
            endpoints[name] = support
        stats = _statistics(boot)
        corrected_ci = None
        if estimate_rate is not None and len(boot):
            corrected_ci = np.quantile(
                float(estimate_rate) * np.exp(-beta * (np.asarray(boot) - ui_barrier)),
                [0.025, 0.975],
            ).tolist()
        per_bins.append(
            {
                "bins": nbin,
                "bin_width": float(edge[1] - edge[0]),
                "endpoints_supported": barrier is not None,
                "barrier_hartree": barrier,
                "barrier_kcal_mol": (
                    barrier * HARTREE_KCAL_MOL if barrier is not None else None
                ),
                "mbar_minus_ui_hartree": (
                    barrier - ui_barrier if barrier is not None else None
                ),
                "barrier_bootstrap_hartree": stats,
                "bootstrap_barriers_hartree": boot,
                "paired_mbar_minus_ui_bootstrap_hartree": _statistics(difference),
                "paired_mbar_minus_ui_draws_hartree": difference,
                "bootstrap_valid_fraction": (
                    len(boot) / bootstrap if bootstrap else None
                ),
                "same_kappa_rate_ratio_to_ui": factor,
                "same_kappa_qtst_cm3_s": (
                    float(pmf["k_qtst_cm3_s"]) * factor if factor is not None else None
                ),
                "same_kappa_rate_estimate_cm3_s": (
                    float(estimate_rate) * factor
                    if estimate_rate is not None and factor is not None
                    else None
                ),
                "same_kappa_rate_cm3_s": (
                    float(recorded_rate) * factor
                    if recorded_rate is not None and factor is not None
                    else None
                ),
                "conditional_rate_percentile_95_ci_cm3_s": corrected_ci,
                "endpoint_support": endpoints,
                "curve": {
                    "xi": result["grid"],
                    "free_energy_hartree": result["free"],
                    "mask": result["mask"],
                },
            }
        )
    ranked = sorted(
        moments,
        key=lambda m: max(abs(m["skewness"]), abs(m["excess_kurtosis"])),
        reverse=True,
    )
    return _finite(
        {
            "diagnostic": "Fixed-xi histogram MBAR versus Gaussian umbrella integration",
            "run": str(run),
            "fixed_xi_star": xi,
            "reference_xi": 0.0,
            "beta_hartree_inverse": beta,
            "temperature_K": config["temperature_K"],
            "measure": "rate_flux; identical stored sampling measure for both estimators",
            "common_histogram_range": limits,
            "min_count": min_count,
            "endpoint_estimator": "Linear interpolation of log density between supported bracketing bin centers; no extrapolation",
            "ui": {
                "barrier_hartree": ui_barrier,
                "barrier_kcal_mol": ui_barrier * HARTREE_KCAL_MOL,
                "kappa": summary.get("kappa"),
                "rate_estimate_cm3_s": estimate_rate,
                "rate_cm3_s": recorded_rate,
                "paired_bootstrap_hartree": _statistics(ui_draws),
                "bootstrap_barriers_hartree": ui_draws,
                "curve": {
                    "xi": ui_grid,
                    "free_energy_hartree": ui_curve - np.interp(0, ui_grid, ui_curve),
                },
            },
            "per_bins": per_bins,
            "window_moments": moments,
            "extreme_windows": ranked[:5],
            "maximum_absolute_skewness": max(abs(m["skewness"]) for m in moments),
            "maximum_absolute_excess_kurtosis": max(
                abs(m["excess_kurtosis"]) for m in moments
            ),
            "mbar": {
                "connected": True,
                "residual": fit.residual,
                "streams": len(arrays),
                "samples": len(x),
                "unique_umbrella_hamiltonians": len(unique),
                "stream_to_hamiltonian": state_map,
                "aggregation_target_weight_maximum_absolute_difference": weight_difference,
            },
            "bootstrap": {
                "requested": bootstrap,
                "seed": seed,
                "failures": failures,
                "method": "Independent circular blocks within each original stream; refit MBAR once per draw; paired endpoint covariance retained",
            },
            "sources": {
                "hdf5_sha256": _sha256(h5_path),
                "summary_sha256": _sha256(run / "summary.json"),
                "checkpoint_fingerprints_in_hdf5_order": fingerprints,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "independent_validation": False,
            "converged": False,
            "caveats": [
                "The estimators reuse the same finite correlated samples; their comparison is not independent physical validation.",
                "Gaussian UI may be biased for non-Gaussian windows; more samples alone do not remove that approximation.",
                "Histogram MBAR retains bin-width bias and finite-sampling uncertainty; connected overlap is not proof of mixing.",
                "The dividing surface and original recrossing coefficient are held fixed; moving xi_star would require new constrained parent and child trajectories.",
                "Rate confidence intervals condition on the recorded kappa and do not include its uncertainty or bead, timestep, Rinf, or equilibration errors.",
                "Unsupported endpoints yield null estimates; this report does not alter the original workflow publication gates.",
                "Complete HDF5 samples are used; checkpoint prefixes are checked exactly and any shorter checkpoint length is reported per stream.",
            ],
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bins", nargs="+", type=int, default=[50, 100, 150])
    parser.add_argument("--bootstrap", type=int, default=40)
    parser.add_argument("--seed", type=int, default=73019)
    args = parser.parse_args()
    result = analyze_run(
        args.run, bins=tuple(args.bins), bootstrap=args.bootstrap, seed=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "xi_star": result["fixed_xi_star"],
                "ui_barrier_kcal_mol": result["ui"]["barrier_kcal_mol"],
                "mbar_barriers_kcal_mol": [
                    entry["barrier_kcal_mol"] for entry in result["per_bins"]
                ],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
