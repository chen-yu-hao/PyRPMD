"""Gas-phase rate preparation, PMF estimates and parent initial configurations."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from ..analysis.fes import umbrella_integration, _block_resample, _bootstrap_block_plan
from ..constants import AU_RATE_TO_CM3_S
from ..input import rate_pmf_options
from .prepare import prepare_rate
from ..checkpoint import atomic_savez as _atomic_npz
from ..io import write_report_json as write_json, report_jsonable as jsonable
from .fes import resolve_umbrellas
from .rate_constraints import rate_coordinate

FLUX_MEASURE = "delta(xi-x) * sqrt(grad(xi) M^-1 grad(xi)/(2*pi*beta))"


def _prepare(config, system, output):
    pmf_options = rate_pmf_options(config)
    prepared = prepare_rate(config, system, numerical=True)
    write_json(Path(output) / "preparation.json", prepared)
    if not prepared["ready"]:
        raise ValueError(
            "TS-only reaction preparation is ambiguous: "
            + "; ".join(prepared["diagnostics"])
            + ". See preparation.json and specify reaction.fragments/forming_bonds/breaking_bonds."
        )
    result = copy.deepcopy(config)
    result["reaction"] = prepared["reaction"]
    reaction = result["reaction"]
    result["cvs"] = [
        {
            "name": "xi",
            "kind": "rpmdrate_xi",
            "estimator": "centroid",
            "fragments": reaction["fragments"],
            "forming_bonds": reaction["forming_bonds"],
            "breaking_bonds": reaction["breaking_bonds"],
            "rinf": reaction["rinf"],
        }
    ]
    if reaction.get("equivalent_paths"):
        result["cvs"][0]["equivalent_paths"] = reaction["equivalent_paths"]
    umbrella = result.setdefault("umbrella", {})
    if not any(k in umbrella for k in ("centers", "centers_au", "grid", "start")):
        umbrella.update(start=-0.05, stop=1.05, count=23)
    if not any(k in umbrella for k in ("stiffness", "stiffness_au", "kappa")):
        umbrella["kappa"] = 50.0 / result["beta"]
    umbrella["cvs"] = ["xi"]
    if result.get("analysis", {}).get("measure", "rate_flux") != "rate_flux":
        raise ValueError(
            "standard RPMD rate sampling requires analysis.measure='rate_flux'"
        )
    result.setdefault("analysis", {}).update(measure="rate_flux")
    result["analysis"].setdefault("bins", 100)
    result["analysis"].setdefault(
        "bootstrap",
        result["analysis"].get(
            "bootstrap_replicates", result["analysis"].get("bootstrap_samples", 20)
        ),
    )
    _, _, centers, _ = resolve_umbrellas(result, system)
    _check_pmf_domain(centers[:, 0], pmf_options)
    return result


def _check_pmf_domain(centers, options):
    centers = np.asarray(centers, float)
    if centers.min() > 0 or centers.max() < 1:
        raise ValueError("rate umbrellas must cover xi=0 (reactants) through xi=1 (TS)")
    if options["xi_star"] is not None and not 0 <= options["xi_star"] <= centers.max():
        raise ValueError("dividing surface outside sampled PMF range")
    if (
        options["search_range"] is not None
        and options["search_range"][1] > centers.max()
    ):
        raise ValueError(
            "rate.pmf_search_range extends beyond the umbrella-center range"
        )


def _histogram_endpoint_support(data, value):
    """Observed histogram support, used to qualify the Gaussian UI model."""
    edges = np.asarray(data["edges"][0])
    axis = 0.5 * (edges[:-1] + edges[1:])
    mask = np.asarray(data["mask"], bool).reshape(-1)
    index = int(np.searchsorted(axis, value))
    if index < len(axis) and np.isclose(axis[index], value, rtol=0, atol=1e-14):
        bracket = [index]
    elif 0 < index < len(axis):
        bracket = [index - 1, index]
    else:
        return {
            "coordinate": value,
            "supported": False,
            "reason": "outside_bin_centers",
        }
    supported = not mask[bracket].any()
    return {
        "coordinate": value,
        "supported": bool(supported),
        "reason": None if supported else "masked_endpoint_bracket",
        "bracket_centers": axis[bracket].tolist(),
        "raw_count": np.asarray(data["raw_count"]).reshape(-1)[bracket].tolist(),
    }


def _ui_rate_pmf(sampled, config, centers, samples, stiffness, options):
    """Retain the existing Gaussian UI estimator with explicit diagnostics."""
    grid = np.linspace(0.0, centers.max(), options["points"])
    curve, derivative = umbrella_integration(
        samples, centers, stiffness, config["beta"], grid
    )
    if not np.isfinite(curve).all() or not np.isfinite(derivative).all():
        raise FloatingPointError(
            "nonfinite Gaussian UI curve; inspect the umbrella streams"
        )
    interval = options["search_range"] or [0.0, float(centers.max())]
    eligible = np.flatnonzero((grid >= interval[0]) & (grid <= interval[1]))
    if not len(eligible):
        raise ValueError("rate.pmf_search_range contains no UI grid points")
    xi = options["xi_star"]
    if xi is None:
        xi = float(grid[eligible[np.argmax(curve[eligible])]])
    barrier = float(np.interp(xi, grid, curve) - curve[0])
    connected = bool(sampled["data"]["connected"])
    gs = np.asarray(sampled["data"]["statistical_inefficiency"])
    block_plan = _bootstrap_block_plan(np.asarray([len(s) for s in samples]), gs)
    rng = np.random.default_rng(np.random.SeedSequence([options["analysis_seed"], 401]))
    barriers, curves, records, endpoints, failures = [], [], [], [], []
    if connected:
        for draw in range(options["bootstrap"]):
            record = {"draw": draw, "valid": False, "reason": None}
            draws = [
                _block_resample(s, block, rng)
                for s, block in zip(samples, block_plan["block_lengths"])
            ]
            try:
                fb, _ = umbrella_integration(
                    draws, centers, stiffness, config["beta"], grid
                )
                rv, sv = float(fb[0]), float(np.interp(xi, grid, fb))
                if not np.isfinite(fb).all() or not np.isfinite(sv - rv):
                    raise FloatingPointError("nonfinite UI bootstrap curve")
                barriers.append(sv - rv)
                curves.append(fb - rv)
                endpoints.append([rv, sv])
                record.update(
                    valid=True,
                    reference_free_energy_hartree=rv,
                    surface_free_energy_hartree=sv,
                    barrier_hartree=sv - rv,
                )
            except (ValueError, FloatingPointError) as error:
                record.update(reason="ui_fit_failed", message=str(error))
                failures.append(
                    {key: value for key, value in record.items() if key != "valid"}
                )
            records.append(record)
    barrier_error = float(np.std(barriers, ddof=1)) if len(barriers) >= 2 else None
    standard_error = (
        np.std(curves, axis=0, ddof=1)
        if len(curves) >= 2
        else np.full_like(curve, np.nan)
    )
    if block_plan["status"] == "unavailable":
        barrier_error = None
        standard_error[:] = np.nan
    empirical = {
        "reference": _histogram_endpoint_support(sampled["data"], 0.0),
        "dividing_surface": _histogram_endpoint_support(sampled["data"], xi),
    }
    warnings = [
        "Gaussian UI models each biased window by its mean and variance; inspect non-Gaussian window distributions or compare with MBAR."
    ]
    warnings.extend(block_plan["warnings"])
    if not all(point["supported"] for point in empirical.values()):
        warnings.append(
            "A Gaussian UI endpoint lacks observed histogram support; the model value is not an interpolation of measured supported bins."
        )
    if options["bootstrap"] and len(barriers) < 2:
        warnings.append(
            "Fewer than two valid UI bootstrap draws; barrier uncertainty is unavailable."
        )
    if failures:
        warnings.append(
            "Some UI bootstrap draws failed; inspect the recorded reasons and valid fraction."
        )
    if not connected:
        warnings.append(
            "Disconnected overlap does not identify the PMF barrier; no recrossing trajectories are launched."
        )
    return {
        "ready": connected,
        "status": "ready" if connected else "disconnected_overlap",
        "xi_star": xi,
        "reference": 0.0,
        "free_energy_barrier_hartree": barrier if connected else None,
        "barrier_standard_error_hartree": barrier_error,
        "barrier_percentile_95_ci_hartree": (
            np.quantile(barriers, [0.025, 0.975]).tolist()
            if len(barriers) >= 2 and block_plan["status"] != "unavailable"
            else None
        ),
        "pmf_bootstrap_barriers": barriers,
        "xi": grid,
        "free_energy": curve,
        "standard_error": standard_error,
        "derivative": derivative,
        "mask": np.full(len(grid), not connected),
        "overlap_connected": connected,
        "support": {
            "model": "Gaussian-window integration within the umbrella-center range",
            "reference": {"coordinate": 0.0, "model_supported": connected},
            "dividing_surface": {"coordinate": xi, "model_supported": connected},
            "observed_histogram": empirical,
            "search_range": interval,
            "selection": (
                "fixed"
                if options["xi_star"] is not None
                else "maximum_gaussian_ui_grid_point"
            ),
            "failure_reason": None if connected else "disconnected_overlap",
        },
        "bootstrap": {
            "requested": options["bootstrap"],
            "attempted": len(records),
            "valid": len(barriers),
            "valid_fraction": (
                len(barriers) / options["bootstrap"] if options["bootstrap"] else None
            ),
            "failures": failures,
            "failure_counts": {"ui_fit_failed": len(failures)} if failures else {},
            "draws": records,
            "surface_held_fixed": True,
            "skipped_reason": None if connected else "disconnected_overlap",
            "block_lengths": block_plan["block_lengths"],
            "block_plan": block_plan,
            "endpoint_values_hartree": np.asarray(endpoints).reshape(-1, 2),
            "endpoint_covariance_hartree_squared": (
                np.cov(np.asarray(endpoints).T, ddof=1) if len(endpoints) >= 2 else None
            ),
            "curve_valid_draws": np.full(len(grid), len(curves)),
        },
        "effective_samples_per_stream": sampled["data"]["effective_samples_per_stream"],
        "weight_ess": sampled["data"]["weight_ess"],
        "warnings": warnings,
    }


def _save_pmf_arrays(output, estimate, method):
    grid = np.asarray(estimate["xi"])
    mask = np.asarray(estimate["mask"], bool)
    curve = np.where(mask, np.nan, estimate["free_energy"])
    error = np.where(mask, np.nan, estimate["standard_error"])
    arrays = {
        "xi": grid,
        "free_energy": curve,
        "standard_error": error,
        "mask": mask,
        "standard_error_mask": mask | ~np.isfinite(error),
        "bootstrap_barriers": np.asarray(estimate["pmf_bootstrap_barriers"]),
        "method": method,
        "sampling_measure": "rate_flux",
        "xi_star": np.asarray(
            estimate["xi_star"] if estimate["xi_star"] is not None else np.nan
        ),
        "reference": 0.0,
        "ready": bool(estimate["ready"]),
    }
    for key in (
        "derivative",
        "density",
        "raw_count",
        "edges",
        "overlap",
        "components",
        "window_free_energies",
    ):
        if key in estimate:
            arrays[key] = estimate[key]
    _atomic_npz(Path(output) / "rate_pmf.npz", **arrays)
    columns = [grid, curve]
    names = ["xi", "free_energy_hartree"]
    if "derivative" in estimate:
        columns.append(np.where(mask, np.nan, estimate["derivative"]))
        names.append("derivative_hartree")
    columns.extend([error, mask.astype(int), (mask | ~np.isfinite(error)).astype(int)])
    names.extend(["standard_error_hartree", "mask", "standard_error_mask"])
    np.savetxt(
        Path(output) / "rate_pmf.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(names),
        comments="",
    )


def _pmf_and_parent_tasks(sampled, config, system, output, options):
    streams = sampled["streams"]
    centers = np.array([s["center"][0] for s in streams])
    samples = [s["samples"][:, 0] for s in streams]
    stiffness = np.array([s["stiffness"][0, 0] for s in streams])
    pmf_options = rate_pmf_options(config)
    _check_pmf_domain(centers, pmf_options)
    method = pmf_options["method"]
    if method == "mbar":
        from ..analysis.fes import mbar_rate_pmf

        estimate = mbar_rate_pmf(
            samples,
            centers,
            stiffness,
            config["beta"],
            xi_star=pmf_options["xi_star"],
            reference=0.0,
            bins=pmf_options["bins"],
            ranges=pmf_options["ranges"],
            search_range=pmf_options["search_range"],
            bootstrap=pmf_options["bootstrap"],
            seed=np.random.SeedSequence([pmf_options["analysis_seed"], 401]),
            min_count=pmf_options["min_count"],
            overlap_threshold=pmf_options["overlap_threshold"],
            compare_ui=pmf_options["compare_ui"],
            ui_points=pmf_options["points"],
            measure="rate_flux",
        )
    else:
        estimate = _ui_rate_pmf(
            sampled, config, centers, samples, stiffness, pmf_options
        )
    _save_pmf_arrays(output, estimate, method)
    xi = estimate["xi_star"]
    barrier = estimate["free_energy_barrier_hartree"]
    fragments = config["reaction"]["fragments"]
    ma, mb = [system.masses[list(f)].sum() for f in fragments]
    reduced = ma * mb / (ma + mb)
    rinf = config["reaction"]["rinf"]
    prefactor = 4 * np.pi * rinf * rinf / np.sqrt(2 * np.pi * config["beta"] * reduced)
    electronic = float(config["reaction"].get("electronic_weight", 1.0))
    symmetry = float(config["reaction"].get("symmetry_factor", 1.0))
    if not np.isfinite(symmetry) or symmetry <= 0:
        raise ValueError("reaction.symmetry_factor must be positive")
    qtst = (
        float(prefactor * np.exp(-config["beta"] * barrier) * electronic * symmetry)
        if estimate["ready"]
        else None
    )
    if qtst is not None and not np.isfinite(qtst):
        raise FloatingPointError("nonfinite QTST estimate; inspect PMF sampling")
    source_status = estimate["status"]
    status = (
        "ready"
        if estimate["ready"]
        else (
            "insufficient_overlap"
            if source_status == "disconnected_overlap"
            else (
                "pmf_solver_failed"
                if source_status == "mbar_not_converged"
                else "unsupported_pmf"
            )
        )
    )
    diagnostics = {
        "schema_version": 2,
        "method": method,
        "ready": bool(estimate["ready"]),
        "status": status,
        "failure_reason": None if estimate["ready"] else source_status,
        "xi_star": xi,
        "reference": 0.0,
        "free_energy_barrier_hartree": barrier,
        "barrier_standard_error_hartree": estimate["barrier_standard_error_hartree"],
        "barrier_percentile_95_ci_hartree": estimate[
            "barrier_percentile_95_ci_hartree"
        ],
        "k_qtst_au": qtst,
        "k_qtst_cm3_s": qtst * AU_RATE_TO_CM3_S if qtst is not None else None,
        "reactant_prefactor_au": prefactor,
        "electronic_weight": electronic,
        "symmetry_factor": symmetry,
        "rinf_bohr": rinf,
        "reduced_mass_au": float(reduced),
        "pmf_method": (
            "Gaussian umbrella integration; MBAR overlap checked"
            if method == "ui"
            else "Histogram MBAR with paired endpoint bootstrap"
        ),
        "measure": FLUX_MEASURE,
        "sampling_measure": "rate_flux",
        "overlap_connected": bool(estimate["overlap_connected"]),
        "pmf_bootstrap_barriers": estimate["pmf_bootstrap_barriers"],
        "warnings": list(
            dict.fromkeys(sampled["summary"]["warnings"] + estimate["warnings"])
        ),
        "intermediate_fes_bootstrap": (
            0 if method == "mbar" else pmf_options["bootstrap"]
        ),
        "intermediate_fes_purpose": "Saved umbrella samples, overlap and a preliminary histogram; the selected rate PMF backend owns barrier bootstrap.",
    }
    for key in (
        "support",
        "solver",
        "bootstrap",
        "window_moments",
        "ui_comparison",
        "effective_samples_per_stream",
        "weight_ess",
    ):
        if key in estimate:
            diagnostics[key] = estimate[key]
    diagnostics["uncertainty"] = {
        "available": estimate["barrier_standard_error_hartree"] is not None,
        "block_status": estimate.get("bootstrap", {}).get("block_plan", {}).get("status"),
        "reason": (
            None
            if estimate["barrier_standard_error_hartree"] is not None
            else (
                "bootstrap_disabled"
                if pmf_options["bootstrap"] == 0
                else (
                    "insufficient_correlated_blocks"
                    if estimate.get("bootstrap", {}).get("block_plan", {}).get("status") == "unavailable"
                    else "fewer_than_two_valid_bootstrap_draws"
                )
            )
        ),
        "surface_held_fixed": True,
        "selection_uncertainty_included": False,
    }
    diagnostics = jsonable(diagnostics)
    write_json(Path(output) / "rate_pmf.json", diagnostics)
    if not estimate["ready"]:
        return diagnostics, []
    nearest = np.argsort(abs(centers - xi))
    permutations = rate_coordinate(config, system).permutations
    tasks = []
    for chain in range(options["parent_replicas"]):
        source = streams[nearest[(chain // len(permutations)) % len(nearest)]]["last_q"]
        initial_q = np.empty_like(source)
        initial_q[:, list(permutations[chain % len(permutations)]), :] = source
        tasks.append(
            {
                "config": config,
                "system": system,
                "options": options,
                "chain": chain,
                "xi_star": xi,
                "initial_q": initial_q,
                "output": str(output),
            }
        )
    return diagnostics, tasks
