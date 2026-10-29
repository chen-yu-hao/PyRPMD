"""Prespecified BKMP2 H3 PMF study with independent seeds and new recrossing.

The four formal cases are fixed before inspecting their results. MBAR is the
primary rate PMF; Gaussian UI uses the same samples and paired bootstrap draws.
Passing the implementation checks never sets physical convergence to true.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from pyrpmd.analysis.fes import mbar_rate_pmf, statistical_inefficiency
from pyrpmd.constants import AU_TIME_FS, KB_AU
from pyrpmd.dynamics import native_capabilities
from pyrpmd.input import JobSpec
from pyrpmd.io import file_sha256, fingerprint, referenced_files
from pyrpmd.potentials.python import import_object
from pyrpmd.runner import run_resolved
from pyrpmd.workflows.fes import jsonable, write_json
from pyrpmd.workflows.rate import rate_coordinate

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "convergence_inputs/base.inp"
HARTREE_KCAL_MOL = 627.5094740631
SCHEMA = "pyrpmd.h3.pmf-convergence.v1"
CASES = {
    "k400-a": {"seed": 821071, "beta_k": 400.0, "start": -0.05, "spacing": 0.025},
    "k400-b": {"seed": 821072, "beta_k": 400.0, "start": -0.05, "spacing": 0.025},
    "k800-a": {"seed": 831071, "beta_k": 800.0, "start": -0.06, "spacing": 0.02},
    "k800-b": {"seed": 831072, "beta_k": 800.0, "start": -0.06, "spacing": 0.02},
}


def _digest(value):
    raw = json.dumps(
        jsonable(value), sort_keys=True, allow_nan=False, separators=(",", ":")
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def default_workers():
    return min(4, os.cpu_count() or 1)


def case_settings(case, smoke=False):
    """Return a fresh immutable-by-convention scientific recipe."""
    if not isinstance(smoke, bool):
        raise ValueError("smoke must be an explicit boolean")
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}; choose one of {', '.join(CASES)}")
    result = {
        **CASES[case],
        "case": case,
        "smoke": bool(smoke),
        "temperature_K": 1500.0,
        "beads": 16,
        "timestep_fs": 0.1,
        "stop": 1.10,
        "replicas": 2,
        "equilibration_fs": 300.0,
        "production_fs": 1000.0,
        "sample_interval_fs": 0.5,
        "andersen_interval_steps": 50,
        "xi_star": 1.00925,
        "reference_xi": 0.0,
        "rinf_bohr": 10.0,
        "histogram_range": [-0.30, 1.40],
        "bins": 170,
        "sensitivity_bins": [85, 340],
        "bootstrap": 40,
        "min_count": 5,
        "parent_replicas": 2,
        "parent_samples_per_chain": 32,
        "parent_equilibration_fs": 200.0,
        "parent_interval_fs": 10.0,
        "child_pairs_per_parent": 2,
        "child_time_fs": 80.0,
        "recrossing_bootstrap": 100,
    }
    intervals = (result["stop"] - result["start"]) / result["spacing"]
    if abs(intervals - round(intervals)) > 1e-10:
        raise ValueError("window interval must be exactly divisible by its spacing")
    result["windows"] = int(round(intervals)) + 1
    if smoke:
        # Keep every umbrella and both initial arrangement replicas. Coarser
        # analysis and short trajectories exercise the API, not convergence.
        result.update(
            equilibration_fs=10.0,
            production_fs=20.0,
            bins=34,
            sensitivity_bins=[17, 68],
            bootstrap=2,
            parent_samples_per_chain=2,
            parent_equilibration_fs=5.0,
            parent_interval_fs=2.0,
            child_pairs_per_parent=1,
            child_time_fs=5.0,
            recrossing_bootstrap=10,
        )
    return result


def compute_budget(settings):
    dt = settings["timestep_fs"]
    streams = settings["windows"] * settings["replicas"]
    per_stream = round((settings["equilibration_fs"] + settings["production_fs"]) / dt)
    samples = streams * round(
        settings["production_fs"] / settings["sample_interval_fs"]
    )
    parents = settings["parent_replicas"] * settings["parent_samples_per_chain"]
    children = 2 * parents * settings["child_pairs_per_parent"]
    return {
        "windows": settings["windows"],
        "streams": streams,
        "umbrella_steps": streams * per_stream,
        "saved_samples": samples,
        "parent_configurations": parents,
        "child_trajectories": children,
        "child_steps": children * round(settings["child_time_fs"] / dt),
        "parent_steps": settings["parent_replicas"]
        * round(
            (
                settings["parent_equilibration_fs"]
                + settings["parent_samples_per_chain"] * settings["parent_interval_fs"]
            )
            / dt
        ),
        "unique_state_reduced_potential_bytes": settings["windows"] * samples * 8,
        "primary_bootstrap_draws": settings["bootstrap"],
        "planned_mbar_fits": settings["bootstrap"] + 4,
        "fit_count_note": "One intermediate FES, one primary PMF plus bootstrap, two bootstrap-free bin checks.",
        "memory_note": "The matrix byte count is one array, not a peak-memory guarantee.",
    }


def study_plan():
    return {
        "schema": SCHEMA,
        "action": "plan",
        "converged": False,
        "cases": [
            {"settings": (s := case_settings(case)), "budget": compute_budget(s)}
            for case in CASES
        ],
        "reference_timing": {
            "github_run_id": 40911869436,
            "old_windows": 23,
            "old_streams": 46,
            "old_samples": 92000,
            "whole_seconds": 1021.9715933589999,
            "forty_draw_diagnostic_seconds": 553.0377933210001,
            "note": "Measured with the previous solver. The updated solver must be timed separately; these are not new-run predictions.",
        },
        "design": [
            "Different global seeds are independent runs; bootstrap draws are not independent runs.",
            "Every formal case recomputes constrained parents and children at xi=1.00925.",
            "The primary MBAR and Gaussian-UI comparison share the same bootstrap draw.",
            "85/340-bin sensitivities use zero additional bootstrap draws.",
            "Two seeds per stiffness do not establish bead, timestep, Rinf, or full statistical convergence.",
        ],
    }


def configuration(case, output, workers=None, library=None, smoke=False):
    settings = case_settings(case, smoke)
    workers = default_workers() if workers is None else workers
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    spec = JobSpec.from_file(TEMPLATE)
    spec.data["seed"] = settings["seed"]
    spec.data["potential"]["params"] = {
        "library": str(Path(library or HERE / "build/libbkmp2.so").resolve())
    }
    spec.data["parallel"] = {
        "backend": "process" if workers > 1 else "serial",
        "trajectory_workers": workers,
        "bead_workers": 1,
        "threads_per_rank": 1,
    }
    spec.data["sampling"].update(
        equilibration=f"{settings['equilibration_fs']} fs",
        production=f"{settings['production_fs']} fs",
        interval=f"{settings['sample_interval_fs']} fs",
        replicas=settings["replicas"],
    )
    spec.data["umbrella"].update(
        start=settings["start"],
        stop=settings["stop"],
        count=settings["windows"],
        kappa=settings["beta_k"] * KB_AU * settings["temperature_K"],
    )
    spec.data["analysis"].update(
        bins=settings["bins"],
        ranges=[settings["histogram_range"]],
        bootstrap=settings["bootstrap"],
        seed=settings["seed"] + 73019,
        min_count=settings["min_count"],
    )
    spec.data["rate"].update(pmf_method="mbar", dividing_surface=settings["xi_star"])
    spec.data["recrossing"].update(
        parent_replicas=settings["parent_replicas"],
        parent_samples=settings["parent_samples_per_chain"],
        equilibration=f"{settings['parent_equilibration_fs']} fs",
        parent_interval=f"{settings['parent_interval_fs']} fs",
        child_pairs=settings["child_pairs_per_parent"],
        child_time=f"{settings['child_time_fs']} fs",
        bootstrap=settings["recrossing_bootstrap"],
    )
    spec.data["output"]["directory"] = str(Path(output).resolve() / "run")
    config, system = spec.resolve()
    return config, system, {**settings, "workers": workers, "cpu_count": os.cpu_count()}


def source_identity(config):
    import pyrpmd

    package = Path(pyrpmd.__file__).resolve().parent
    files = {
        str(path.relative_to(package)): file_sha256(path)
        for path in sorted(package.rglob("*.py"))
    }
    files["benchmark/convergence.py"] = file_sha256(Path(__file__))
    files["benchmark/convergence_inputs/base.inp"] = file_sha256(TEMPLATE)
    for path in sorted(package.glob("_native*.so")):
        files["native_binary/" + path.name] = file_sha256(path)
    native_source = HERE.parents[1] / "native/cayley.cpp"
    if native_source.is_file():
        files["native/cayley.cpp"] = file_sha256(native_source)
    return {
        "package_and_runner_sources": files,
        "referenced_files": referenced_files(config),
    }


def load_samples(run, settings):
    """Require final sample counts, energy arrays and recovery clocks to agree."""
    import h5py

    run = Path(run)
    arrays, centers, stiffness, fingerprints, mismatches, radii = [], [], [], [], [], []
    identities = []
    expected_samples = round(settings["production_fs"] / settings["sample_interval_fs"])
    expected_steps = round(
        (settings["equilibration_fs"] + settings["production_fs"])
        / settings["timestep_fs"]
    )
    with h5py.File(run / "umbrella/fes.h5", "r") as h:
        if h.attrs["measure"] != "rate_flux":
            raise ValueError("the convergence study requires the rate_flux measure")
        for stream, key in enumerate(sorted(h["streams"], key=int)):
            if int(key) != stream:
                raise ValueError("non-contiguous HDF5 stream numbering")
            group = h["streams"][key]
            window, replica = int(group.attrs["window"]), int(group.attrs["replica"])
            identities.append((window, replica))
            values, energies = np.asarray(group["samples"]), np.asarray(
                group["energies"]
            )
            path = (
                run
                / "umbrella/checkpoints"
                / f"window_{window:05d}_replica_{replica:04d}.npz"
            )
            with np.load(path, allow_pickle=False) as saved:
                checks = {
                    "samples_exact": np.array_equal(values, saved["values"]),
                    "energies_exact": np.array_equal(energies, saved["energies"]),
                    "sample_count": len(values) == expected_samples,
                    "elapsed_complete": int(saved["elapsed"]) == expected_steps,
                }
                if not all(checks.values()):
                    mismatches.append({"window": window, "replica": replica, **checks})
                fingerprints.append(str(saved["fingerprint"]))
                q = np.asarray(saved["q"])
                radii.append(
                    float(np.sqrt(np.mean(np.sum((q - q.mean(0)) ** 2, axis=-1))))
                )
            if (
                values.ndim != 2
                or values.shape[1] != 1
                or not np.isfinite(values).all()
            ):
                raise ValueError("window samples must be finite scalar coordinates")
            arrays.append(values[:, 0])
            centers.append(float(np.asarray(group["center"])[0]))
            stiffness.append(float(np.asarray(group["stiffness"])[0, 0]))
    expected_streams = settings["windows"] * settings["replicas"]
    expected_identities = {
        (window, replica)
        for window in range(settings["windows"])
        for replica in range(settings["replicas"])
    }
    identities_complete = (
        len(set(identities)) == len(identities)
        and set(identities) == expected_identities
    )
    integrity = {
        "passed": not mismatches
        and len(arrays) == expected_streams
        and identities_complete,
        "streams_checked": len(arrays),
        "expected_streams": expected_streams,
        "stream_identities_complete": identities_complete,
        "expected_samples_per_stream": expected_samples,
        "expected_elapsed_steps": expected_steps,
        "exact_sample_count_values_and_energies": not mismatches,
        "mismatches": mismatches,
    }
    return (
        arrays,
        np.asarray(centers),
        np.asarray(stiffness),
        {
            "checkpoint_integrity": integrity,
            "checkpoint_fingerprints": fingerprints,
            "samples": sum(map(len, arrays)),
            "streams": len(arrays),
            "ring_radius_of_gyration_minimum_bohr": min(radii) if radii else None,
        },
    )


def recrossing_diagnostics(run, config, system, result):
    prepared = json.loads((Path(run) / "preparation.json").read_text())
    prepared_config = copy.deepcopy(config)
    prepared_config["reaction"] = prepared["reaction"]
    xi = result.get("pmf", {}).get("xi_star")
    residuals, branches = [], [0, 0]
    if xi is not None:
        surface = rate_coordinate(prepared_config, system).dividing_surface(xi)
        for path in sorted((Path(run) / "checkpoints").glob("parent_*.npz")):
            with np.load(path, allow_pickle=False) as saved:
                for q in saved["snapshots"]:
                    centroid = q.mean(0)
                    residuals.append(abs(surface.centroid_value_gradient(centroid)[0]))
                    branch = int(
                        np.linalg.norm(centroid[0] - centroid[2])
                        < np.linalg.norm(centroid[0] - centroid[1])
                    )
                    branches[branch] += 1
    curve = {}
    if (Path(run) / "recrossing.npz").is_file():
        with np.load(Path(run) / "recrossing.npz", allow_pickle=False) as saved:
            curve = {key: saved[key] for key in saved.files}
    transmission = curve.get("kappa", np.array([]))
    den = curve.get("parent_denominator", np.array([]))
    return {
        "maximum_parent_constraint_residual_bohr": (
            max(residuals) if residuals else None
        ),
        "parent_surface_branch_counts": branches,
        "positive_initial_flux": bool(len(den)) and bool(np.all(den > 0)),
        "kappa_first": float(transmission[0]) if len(transmission) else None,
        "kappa_range": float(np.ptp(transmission)) if len(transmission) else None,
        "maximum_endpoint_energy_error_hartree_per_bead": result.get(
            "maximum_endpoint_energy_error_hartree_per_bead"
        ),
        "maximum_endpoint_energy_error_kbt": result.get(
            "maximum_endpoint_energy_error_kbt"
        ),
    }


def quality_gates(settings, result, diagnostics):
    """The original scientific thresholds apply unchanged, including in smoke."""
    pmf = result.get("pmf", {})
    xi, barrier = pmf.get("xi_star"), pmf.get("free_energy_barrier_hartree")
    kappa = result.get("kappa")
    error = diagnostics.get("maximum_endpoint_energy_error_kbt")
    constraint = diagnostics.get("maximum_parent_constraint_residual_bohr")
    span = diagnostics.get("kappa_range")
    boot = pmf.get("bootstrap", {})
    records = boot.get("draws", [])
    ui = pmf.get("ui_comparison", {})
    return {
        "mbar_is_primary_method": pmf.get("method") == "mbar",
        "prespecified_surface_used": xi is not None
        and abs(xi - settings["xi_star"]) < 1e-12,
        "overlap_connected": bool(pmf.get("overlap_connected", False)),
        "minimum_stream_effective_samples_ge_5": diagnostics.get(
            "effective_samples_minimum", 0
        )
        >= 5,
        "positive_pmf_barrier": barrier is not None and 0 < barrier < 0.2,
        "reactant_reference_sampled": diagnostics.get(
            "reactant_reference_sample_count", 0
        )
        >= 30,
        "dividing_surface_sampled": diagnostics.get("dividing_surface_sample_count", 0)
        >= 30,
        "parent_constraint_satisfied": constraint is not None and constraint < 1e-8,
        "both_equivalent_ts_components_initialized": min(
            diagnostics.get("parent_surface_branch_counts", [0, 0])
        )
        > 0,
        "positive_initial_flux": bool(diagnostics.get("positive_initial_flux", False)),
        "actual_child_ensemble": result.get("child_trajectories", 0) >= 128,
        "nontrivial_positive_recrossing": kappa is not None
        and 0 < kappa < 0.995
        and span is not None
        and span > 0.002,
        "tail_platform_observed": bool(result.get("plateau_detected", False)),
        "offset_independent_endpoint_energy_error_below_0p1_kbt": error is not None
        and error < 0.1,
        "final_samples_match_checkpoints": diagnostics["checkpoint_integrity"][
            "passed"
        ],
        "native_kernel_used": bool(diagnostics.get("native_kernel_used", False)),
        "bootstrap_attempts_all_recorded": boot.get("requested")
        == settings["bootstrap"]
        and boot.get("attempted") == settings["bootstrap"]
        and len(records) == settings["bootstrap"],
        "all_requested_bootstrap_draws_valid": boot.get("valid")
        == settings["bootstrap"],
        "all_requested_paired_ui_draws_valid": bool(ui.get("ready"))
        and len(ui.get("paired_difference_bootstrap_hartree", []))
        == settings["bootstrap"]
        and not ui.get("failures", []),
    }


def _compact_pmf(pmf):
    # Curves and dense overlap matrices live in array files. Every bootstrap
    # attempt and its reason remains in JSON for audit and independent reports.
    omit = {
        "xi",
        "free_energy",
        "standard_error",
        "mask",
        "edges",
        "density",
        "raw_count",
        "weights",
        "overlap",
        "components",
        "window_free_energies",
        "effective_samples_per_stream",
        "statistical_inefficiency",
    }
    result = {key: value for key, value in pmf.items() if key not in omit}
    if "ui_comparison" in result:
        result["ui_comparison"] = {
            k: v
            for k, v in result["ui_comparison"].items()
            if k not in ("xi", "free_energy")
        }
    return result


def analyze_case(run, config, system, settings, result):
    run = Path(run)
    samples, centers, stiffness, diagnostics = load_samples(run, settings)
    pmf = dict(result.get("pmf", {}))
    if (run / "rate_pmf.json").is_file():
        pmf.update(json.loads((run / "rate_pmf.json").read_text()))
    result = {**result, "pmf": pmf}
    gs = np.asarray([statistical_inefficiency(a) for a in samples])
    ess = np.asarray([len(a) for a in samples]) / gs
    combined = np.concatenate(samples)
    diagnostics.update(
        effective_samples_minimum=float(ess.min()),
        effective_samples_median=float(np.median(ess)),
        effective_samples_sum=float(ess.sum()),
        statistical_inefficiency_maximum=float(gs.max()),
        reactant_reference_sample_count=int(np.count_nonzero(abs(combined) < 0.05)),
        dividing_surface_sample_count=int(
            np.count_nonzero(abs(combined - settings["xi_star"]) < 0.025)
        ),
        window_moments=pmf.get("window_moments", []),
        support=pmf.get("support"),
        solver=pmf.get("solver"),
        native_kernel_used=bool(native_capabilities()["available"]),
        **recrossing_diagnostics(run, config, system, result),
    )
    curves = {}
    with np.load(run / "rate_pmf.npz", allow_pickle=False) as saved:
        for key in ("xi", "free_energy", "mask", "standard_error"):
            if key in saved:
                curves[f"{key}_{settings['bins']}"] = saved[key]
    if f"mask_{settings['bins']}" not in curves:
        raise ValueError("the MBAR backend must preserve its primary PMF mask")
    ui = pmf.get("ui_comparison", {})
    if ui.get("ready"):
        curves["ui_xi"] = np.asarray(ui["xi"])
        curves["ui_free_energy"] = np.asarray(ui["free_energy"])
    sensitivities = []
    for bins in settings["sensitivity_bins"]:
        other = mbar_rate_pmf(
            samples,
            centers,
            stiffness,
            config["beta"],
            xi_star=settings["xi_star"],
            reference=0.0,
            bins=bins,
            ranges=settings["histogram_range"],
            bootstrap=0,
            min_count=settings["min_count"],
            compare_ui=False,
        )
        for key in ("xi", "free_energy", "mask", "standard_error"):
            curves[f"{key}_{bins}"] = other[key]
        barrier = other["free_energy_barrier_hartree"]
        main_barrier = pmf.get("free_energy_barrier_hartree")
        ratio = (
            float(np.exp(-config["beta"] * (barrier - main_barrier)))
            if barrier is not None and main_barrier is not None
            else None
        )
        rate = result.get("rate_estimate_cm3_s")
        sensitivities.append(
            {
                "bins": bins,
                "ready": other["ready"],
                "status": other["status"],
                "free_energy_barrier_hartree": barrier,
                "support": other["support"],
                "solver": other["solver"],
                "bootstrap": other["bootstrap"],
                "same_surface_rate_ratio_to_primary": ratio,
                "same_surface_rate_estimate_cm3_molecule_s": (
                    rate * ratio if rate is not None and ratio is not None else None
                ),
                "uncertainty_available": False,
            }
        )
    np.savez_compressed(run.parent / "comparison-curves.npz", **curves)
    gates = quality_gates(settings, result, diagnostics)
    report = {
        "pmf": _compact_pmf(pmf),
        "diagnostics": diagnostics,
        "bin_sensitivity": sensitivities,
        "production_gates": gates,
        "gates": gates.copy(),
        "recrossing": {
            k: result.get(k)
            for k in (
                "kappa",
                "kappa_standard_error",
                "plateau_detected",
                "plateau_start_au",
                "plateau_drift",
                "plateau_drift_standard_error",
                "parent_configurations",
                "child_pairs",
                "child_trajectories",
                "child_time_fs",
            )
        },
        "rate_estimate_cm3_molecule_s": result.get("rate_estimate_cm3_s"),
        "rate_standard_error_cm3_molecule_s": result.get("rate_standard_error_cm3_s"),
        "rate_percentile_95_ci_cm3_molecule_s": result.get(
            "rate_percentile_95_ci_cm3_s"
        ),
        "conditional_rate_standard_error_cm3_molecule_s": result.get(
            "conditional_rate_standard_error_cm3_s"
        ),
        "rate_uncertainty": result.get("rate_uncertainty"),
        "workflow_status": result.get("status"),
        "warnings": result.get("warnings", []),
        "files": {
            "pmf": "run/rate_pmf.npz",
            "samples": "run/umbrella/fes.h5",
            "recrossing": "run/recrossing.npz",
            "curves": "comparison-curves.npz",
            "figure_png": "figures/pmf-convergence.png",
            "figure_pdf": "figures/pmf-convergence.pdf",
        },
    }
    if settings["smoke"]:
        required = (
            "mbar_is_primary_method",
            "prespecified_surface_used",
            "overlap_connected",
            "offset_independent_endpoint_energy_error_below_0p1_kbt",
            "final_samples_match_checkpoints",
            "native_kernel_used",
        )
        report["gates"] = {key: gates[key] for key in required}
        report["scope"] = (
            "Software smoke only; unchanged production gates are recorded separately and this run is ineligible for convergence comparison."
        )
    return report


def plot_case(output):
    """Render only recorded data; masked MBAR intervals stay visibly absent."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    report = json.loads((output / "benchmark.json").read_text())
    settings = report["settings"]
    with np.load(output / "comparison-curves.npz", allow_pickle=False) as saved:
        curves = {key: saved[key] for key in saved.files}
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 11.2), constrained_layout=True)
    bins = settings["bins"]
    x, f, mask = (curves[f"{key}_{bins}"] for key in ("xi", "free_energy", "mask"))
    axes[0].plot(
        x,
        np.ma.masked_where(mask, f) * HARTREE_KCAL_MOL,
        color="#1565c0",
        lw=2,
        label=f"Primary MBAR ({bins} bins; masks retained)",
    )
    if "ui_xi" in curves:
        axes[0].plot(
            curves["ui_xi"],
            curves["ui_free_energy"] * HARTREE_KCAL_MOL,
            color="#c45a00",
            ls="--",
            label="Gaussian UI (same samples)",
        )
    axes[0].axvline(
        settings["xi_star"],
        color="0.35",
        ls=":",
        label=f"Fixed surface: {settings['xi_star']}",
    )
    axes[0].axvline(0, color="0.7", ls=":")
    axes[0].set(
        xlabel="Reaction coordinate xi",
        ylabel="Free energy relative to xi=0 (kcal/mol)",
    )
    axes[0].legend(fontsize=8)
    moments = report["diagnostics"].get("window_moments", [])
    for key, label, color in (
        ("skewness", "Skewness", "#6a359c"),
        ("excess_kurtosis", "Excess kurtosis", "#008577"),
    ):
        points = [(m["center"], m[key]) for m in moments if m.get(key) is not None]
        if points:
            a = np.asarray(points)
            axes[1].scatter(
                a[:, 0], a[:, 1], s=17, alpha=0.75, label=label, color=color
            )
    axes[1].axhline(0, color="0.6", lw=0.8)
    axes[1].set(xlabel="Umbrella center", ylabel="Standardized window moment")
    axes[1].legend(fontsize=8)
    path = output / "run/recrossing.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            t = saved["time_au"] * AU_TIME_FS
            kappa = saved["kappa"]
            axes[2].plot(
                t, kappa, color="#1565c0", label="New recrossing at the fixed surface"
            )
            if "standard_error" in saved:
                error = saved["standard_error"]
                axes[2].fill_between(
                    t,
                    kappa - error,
                    kappa + error,
                    color="#1565c0",
                    alpha=0.2,
                    label="Bootstrap standard error",
                )
    axes[2].set(xlabel="Child time (fs)", ylabel="Transmission coefficient kappa")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.15)
    scope = (
        "SOFTWARE SMOKE"
        if report["smoke"]
        else "Convergence study; not established convergence"
    )
    fig.suptitle(
        f"H + H2 / BKMP2: {report['case']}, seed {report['seed']}, beta*K={settings['beta_k']:g}\n"
        f"1500 K, 16 beads, dt=0.1 fs | {scope}",
        fontsize=12,
    )
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = directory / f"pmf-convergence.{suffix}"
        fig.savefig(path, dpi=180)
        paths.append(str(path))
    plt.close(fig)
    return paths


def run_case(case, output, *, workers=None, library=None, resume=False, smoke=False):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(
            f"output directory is not empty: {output}; choose a new directory or --resume"
        )
    config, system, settings = configuration(case, output, workers, library, smoke)
    identity = source_identity(config)
    identity_path = output / "provenance/source-identity.json"
    if resume:
        if not identity_path.is_file():
            raise ValueError("resume requires the original convergence source identity")
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError(
                "runner, package, input, or PES files changed; cannot resume this recorded study"
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "provenance").mkdir(exist_ok=True)
    if not resume:
        write_json(identity_path, identity)
        shutil.copy2(Path(__file__), output / "provenance/convergence-source.py")
        shutil.copy2(TEMPLATE, output / "provenance/input-template.inp")
    controls = {
        k: v
        for k, v in settings.items()
        if k not in ("case", "seed", "workers", "cpu_count")
    }
    started = time.perf_counter()
    report = {
        "schema": SCHEMA,
        "case": case,
        "seed": settings["seed"],
        "task": "rate",
        "smoke": bool(smoke),
        "passed": False,
        "converged": False,
        "settings": settings,
        "budget": compute_budget(settings),
        "scope": "Prespecified denser/stronger umbrellas with independent seeds; no declaration of physical convergence.",
        "fingerprints": {
            "workflow_physics": fingerprint(config, system),
            "comparison_controls": _digest(controls),
            "source_identity": _digest(identity),
            "input_files": identity["referenced_files"],
        },
        "limitations": [
            "Independent seeds, bootstrap uncertainty, and bin checks address different questions.",
            "Gaussian UI and MBAR use the same recorded samples and are not independent simulations.",
            "The reported rate uses a new constrained parent/child ensemble at the prespecified dividing surface.",
            "This campaign holds temperature, beads, timestep, and Rinf fixed; their convergence is not tested.",
        ],
    }
    try:
        provider = import_object(str(HERE / "pes.py") + ":build")(
            **config["potential"]["params"]
        )
        report["pes"] = dict(provider.metadata)
        provider.close()
        print(
            json.dumps(
                {
                    "event": "start",
                    "case": case,
                    "seed": settings["seed"],
                    "smoke": smoke,
                    "settings": settings,
                    "budget": report["budget"],
                }
            ),
            flush=True,
        )
        result = run_resolved(config, system, output=output / "run", resume=resume)
        report.update(analyze_case(output / "run", config, system, settings, result))
        report["gates"]["sources_unchanged_during_run"] = (
            source_identity(config) == identity
        )
        report["passed"] = all(report["gates"].values())
        manifest = json.loads((output / "run/manifest.json").read_text())
        report["fingerprints"]["workflow_manifest"] = manifest["physics_fingerprint"]
        report["fingerprints"]["final_hdf5_sha256"] = file_sha256(
            output / "run/umbrella/fes.h5"
        )
        write_json(output / "benchmark.json", report)
        plot_case(output)
    except Exception as error:
        report["passed"] = False
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report.setdefault("gates", {})["workflow_completed"] = False
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
    report["wall_seconds"] = time.perf_counter() - started
    write_json(output / "benchmark.json", report)
    print(
        json.dumps(
            jsonable(
                {
                    "case": case,
                    "seed": report["seed"],
                    "smoke": report["smoke"],
                    "passed": report["passed"],
                    "converged": False,
                    "gates": report.get("gates"),
                    "wall_seconds": report["wall_seconds"],
                    "report": str(output / "benchmark.json"),
                    "pmf": {
                        k: report.get("pmf", {}).get(k)
                        for k in (
                            "method",
                            "xi_star",
                            "free_energy_barrier_hartree",
                            "barrier_standard_error_hartree",
                        )
                    },
                    "recrossing": report.get("recrossing"),
                    "rate_estimate_cm3_molecule_s": report.get(
                        "rate_estimate_cm3_molecule_s"
                    ),
                    "error": report.get("error"),
                }
            ),
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan", help="Print exact work counts without dynamics")
    plan.add_argument("--output", type=Path)
    run = sub.add_parser("run", help="Run one prespecified independent case")
    run.add_argument("--case", choices=CASES, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--workers", type=int)
    run.add_argument("--library", type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--smoke",
        action="store_true",
        help="Short software exercise; excluded from formal comparisons",
    )
    plot = sub.add_parser("plot", help="Render an existing result without dynamics")
    plot.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "plan":
        value = study_plan()
        if args.output:
            write_json(args.output, value)
        print(json.dumps(value, indent=2, allow_nan=False))
        return 0
    if args.action == "plot":
        print("\n".join(plot_case(args.output)))
        return 0
    result = run_case(
        args.case,
        args.output,
        workers=args.workers,
        library=args.library,
        resume=args.resume,
        smoke=args.smoke,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
