"""Read-only MBAR/WHAM/UI comparison of saved rate-flux umbrella samples.

Run from a checkout with ``python -m benchmarks.h3.rate_pmf_comparison RUN``.
Every curve is diagnostic. No rate, PMF readiness, or production file is updated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from benchmarks.h3.mixing_analysis import solve_wham
from pyrpmd.analysis.fes_statistics import cv_statistical_inefficiency
from pyrpmd.analysis.mbar import solve_mbar
from pyrpmd.analysis.rate_pmf import _rate_support, _rate_value
from pyrpmd.analysis.rate_support import (
    classify_point, fit_log_weights, histogram_report, log_histogram,
    weight_diagnostics, window_anomalies,
)
from pyrpmd.analysis.umbrella import umbrella_integration, umbrella_reduced_potentials
from pyrpmd.io import jsonable


def _curve(grid, free, mask, reference, surface):
    ref, target, reason = _rate_support(grid, mask, reference, surface)
    barrier = None if reason else _rate_value(free, target) - _rate_value(free, ref)
    shifted = free.copy()
    if ref is not None:
        shifted -= _rate_value(free, ref)
    return {
        "grid": grid, "free_energy_hartree": shifted, "mask": mask,
        "supported": reason is None, "support_failure": reason,
        "diagnostic_barrier_hartree": barrier,
        "support": {name: classify_point(grid, mask, point) for name, point in
                    (("reference", reference), ("dividing_surface", surface))},
    }


def compare_samples(samples, centers, stiffness, beta, *, ranges, xi_star,
                    reference=0.0, bins=(240, 480, 960), ui_points=(401, 801, 1601),
                    min_count=1):
    """Compare the same complete streams at a fixed surface; never select a rate.

    WHAM extends its histogram by whole bins to retain all sampled tails. The
    displayed grid stays fixed to ``ranges``, so endpoint support is unchanged.
    UI support uses the coarsest requested empirical histogram, independently
    of its integration grid. Kish ESS is not defined for Gaussian UI; its
    original streams instead retain time-correlation ESS diagnostics.
    """
    arrays = [np.asarray(a, float) for a in samples]
    arrays = [a[:, 0] if a.ndim == 2 and a.shape[1] == 1 else a for a in arrays]
    centers = np.asarray(centers, float).reshape(-1)
    stiffness = np.broadcast_to(np.asarray(stiffness, float).reshape(-1), centers.shape)
    bounds = np.asarray(ranges, float)
    if (not arrays or any(a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() for a in arrays)
            or len(arrays) != len(centers) or not np.isfinite(centers).all()
            or not np.isfinite(stiffness).all() or np.any(stiffness <= 0)
            or not np.isfinite(beta) or beta <= 0 or bounds.shape != (2,)
            or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]
            or not np.isfinite([reference, xi_star]).all()):
        raise ValueError("finite scalar streams, umbrellas, beta, bounds and fixed endpoints required")
    bins, ui_points = tuple(bins), tuple(ui_points)
    for values, name in ((bins, "bins"), (ui_points, "ui_points"), ((min_count,), "min_count")):
        if not values or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                             or n < (1 if name == "min_count" else 2) for n in values):
            raise ValueError(name + " must contain positive integer resolutions")
    x = np.concatenate(arrays)
    counts = np.array([len(a) for a in arrays])
    offsets = np.r_[0, np.cumsum(counts)]
    u = umbrella_reduced_potentials(x[:, None], centers[:, None],
                                   stiffness[:, None, None], beta)
    fit = solve_mbar(u, counts, return_state_weights=False)
    log_weights = fit_log_weights(fit)
    inefficiency = np.array([cv_statistical_inefficiency(a[:, None]) for a in arrays])
    streams = {"counts": counts, "statistical_inefficiency": inefficiency,
               "effective_samples": counts / inefficiency}
    result = {
        "schema": "pyrpmd.rate-pmf-comparison.v1", "role": "diagnostic_only",
        "measure": "rate_flux", "beta": beta, "reference": reference, "xi_star": xi_star,
        "rate": None, "same_saved_samples": True, "streams": streams,
        "window_anomalies": window_anomalies(arrays, centers, stiffness, beta),
        "mbar": [], "wham": [], "ui": [],
        "limitations": ["No sampling, bead-number, or timestep convergence is inferred.",
                        "Kish ESS does not include time-correlation corrections.",
                        "Gaussian UI assumes Gaussian window densities; no global target-weight ESS is defined.",
                        "Binning and integration-grid sensitivity are not uncertainty intervals."],
    }
    for n in bins:
        edges = np.linspace(*bounds, n + 1)
        grid = (edges[:-1] + edges[1:]) / 2
        _, density, raw, mask = log_histogram(x, log_weights, edges, min_count)
        record = _curve(grid, -density / beta, mask, reference, xi_star)
        record.update(histogram=histogram_report(edges), raw_count=raw,
                      solver={"converged": fit.converged, "residual": fit.residual,
                              "overlap_connected": len(np.unique(fit.components)) == 1})
        brackets = {name: support["bracket"] for name, support in record["support"].items()}
        record["ess"] = weight_diagnostics(log_weights, x, offsets, edges, brackets)
        result["mbar"].append(record)

        # Preserve this grid's bin width/alignment, including every outlying frame.
        width = (bounds[1] - bounds[0]) / n
        left = min(0, int(np.floor((x.min() - bounds[0]) / width)))
        right = max(n, int(np.ceil((x.max() - bounds[0]) / width)))
        full_edges = bounds[0] + np.arange(left, right + 1) * width
        full_edges[-left:n - left + 1] = edges
        full_edges[0] = min(full_edges[0], x.min())
        full_edges[-1] = max(full_edges[-1], x.max())
        hist = np.array([np.histogram(a, full_edges)[0] for a in arrays])
        try:
            wham = solve_wham(hist, full_edges, centers, stiffness, beta)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            result["wham"].append({"histogram": histogram_report(edges),
                "fit_samples": int(hist.sum()), "fit_range": full_edges[[0, -1]],
                "supported": False, "support_failure": "wham_solver_failed",
                "solver": {"converged": False, "error": str(exc)}, "ess": None,
                "diagnostic_barrier_hartree": None, "wham_minus_mbar_hartree": None})
            continue
        selection = slice(-left, n - left)
        log_density = wham["log_density"][selection]
        occupied = wham["occupied"][selection]
        raw = hist.sum(axis=0)[selection]
        record = _curve(grid, -log_density / beta, ~occupied | (raw < min_count), reference, xi_star)
        index = np.clip(np.searchsorted(full_edges, x, side="right") - 1, 0, hist.shape[1] - 1)
        # Every frame in a WHAM bin carries that bin's fitted mass / raw count.
        wham_log_weights = (wham["log_density"][index] + np.log(np.diff(full_edges)[index])
                            - np.log(hist.sum(axis=0)[index]))
        brackets = {name: support["bracket"] for name, support in record["support"].items()}
        ess = weight_diagnostics(wham_log_weights, x, offsets, edges, brackets)
        ess["definition"] = "Kish ESS of binned WHAM target weights; no time-correlation correction."
        record.update(histogram=histogram_report(edges), raw_count=raw, ess=ess,
                      fit_range=full_edges[[0, -1]], fit_samples=int(hist.sum()),
                      solver={"converged": wham["converged"], "residual": wham["residual"],
                              "overlap_connected": wham["overlap_connected"]})
        mb = result["mbar"][-1]["diagnostic_barrier_hartree"]
        wb = record["diagnostic_barrier_hartree"]
        record["wham_minus_mbar_hartree"] = None if mb is None or wb is None else wb - mb
        result["wham"].append(record)

    support_edges = np.linspace(*bounds, min(bins) + 1)
    support_grid = (support_edges[:-1] + support_edges[1:]) / 2
    raw = np.histogram(x, support_edges)[0]
    _, _, reason = _rate_support(support_grid, raw < min_count, reference, xi_star)
    for n in ui_points:
        grid = np.linspace(*bounds, n)
        try:
            free, _ = umbrella_integration(arrays, centers, stiffness, beta, grid)
            record = _curve(grid, free, np.zeros(n, bool), reference, xi_star)
            if reason:
                record.update(supported=False, support_failure=reason, diagnostic_barrier_hartree=None)
        except (ValueError, FloatingPointError) as exc:
            record = {"grid": grid, "supported": False, "support_failure": str(exc),
                      "diagnostic_barrier_hartree": None}
        record.update(points=n, empirical_support_histogram=histogram_report(support_edges),
                      empirical_raw_count=raw,
                      empirical_support={name: classify_point(support_grid, raw < min_count, point)
                                         for name, point in (("reference", reference), ("dividing_surface", xi_star))},
                      ess={"global_kish_ess": None, "definition": "not defined for Gaussian UI",
                           "per_stream": streams})
        mb = result["mbar"][0]["diagnostic_barrier_hartree"]
        ub = record["diagnostic_barrier_hartree"]
        record["ui_minus_mbar_hartree"] = None if mb is None or ub is None else ub - mb
        record["mbar_reference_bins"] = bins[0]
        result["ui"].append(record)
    return jsonable(result, nonfinite="null")


def compare_run(run, *, output=None, xi_star=None, bins=(240, 480, 960),
                ui_points=(401, 801, 1601), min_count=1):
    """Read a standard rate run, publishing only a new comparison artifact."""
    run = Path(run).resolve()
    output = Path(output) if output is not None else run / "rate_pmf_comparison.json"
    # Exclusive creation below also protects against a concurrent publication.
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"comparison output must be a new file: {output}")
    pmf_path, samples_path = run / "rate_pmf.json", run / "umbrella/fes.h5"
    source_pmf = json.loads(pmf_path.read_text())
    surface = source_pmf.get("xi_star") if xi_star is None else xi_star
    if surface is None:
        raise ValueError("saved PMF has no dividing surface; supply an explicit diagnostic --xi-star")
    with h5py.File(samples_path, "r") as saved:
        if saved.attrs.get("measure") != "rate_flux":
            raise ValueError("comparison requires rate_flux umbrella samples")
        arrays, centers, stiffness, identities = [], [], [], []
        for key in sorted(saved["streams"], key=int):
            group = saved["streams"][key]
            sample, center, matrix = (np.asarray(group[name]) for name in ("samples", "center", "stiffness"))
            if sample.ndim != 2 or sample.shape[1] != 1 or center.shape != (1,) or matrix.shape != (1, 1):
                raise ValueError("comparison requires one scalar coordinate per umbrella stream")
            arrays.append(sample[:, 0]); centers.append(center[0]); stiffness.append(matrix[0, 0])
            identities.append({"stream": key, "window": int(group.attrs["window"]),
                               "replica": int(group.attrs["replica"])})
        result = compare_samples(arrays, centers, stiffness, float(saved.attrs["beta"]),
                                 ranges=np.asarray(saved["edges_0"])[[0, -1]], xi_star=surface,
                                 reference=source_pmf.get("reference", 0.0), bins=bins,
                                 ui_points=ui_points, min_count=min_count)
    result.update(source={"samples": str(samples_path), "pmf": str(pmf_path),
                          "stream_identities": identities}, source_pmf=source_pmf,
                  source_ready_unchanged=source_pmf.get("ready"))
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as destination:
        destination.write(rendered)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--xi-star", type=float)
    parser.add_argument("--bins", type=int, nargs="+", default=[240, 480, 960])
    parser.add_argument("--ui-points", type=int, nargs="+", default=[401, 801, 1601])
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args()
    result = compare_run(**vars(args))
    print(json.dumps({"role": result["role"], "source_ready_unchanged": result["source_ready_unchanged"],
                      "output": str(args.output or args.run / "rate_pmf_comparison.json")}))


if __name__ == "__main__":
    main()
