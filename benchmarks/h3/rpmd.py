"""Physical BKMP2 H3 FES / PMF / recrossing regression, not a converged rate.

Build the real PES first using build_pes.py. The two stages keep all raw
samples and masks; acceptance checks do not compare a short noisy rate to a
literature number. Energies are reported independently of the PES energy zero.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
import traceback
import numpy as np

from pyrpmd.input import JobSpec
from pyrpmd.runner import run_resolved
from pyrpmd.dynamics import native_capabilities
from pyrpmd.constants import AU_TIME_FS
from pyrpmd.potentials.python import import_object
from pyrpmd.workflows.fes import write_json
from pyrpmd.workflows.rate import rate_coordinate

HERE = Path(__file__).resolve().parent


def configuration(stage, output, workers=2, profile="ci", library=None):
    if stage not in ("fes2d", "rate"):
        raise ValueError("stage must be fes2d or rate")
    if profile not in ("ci", "extended"):
        raise ValueError("profile must be ci or extended")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    spec = JobSpec.from_file(HERE / ("fes2d.inp" if stage == "fes2d" else "rate.inp"))
    spec.data["potential"]["params"] = {
        "library": str(Path(library or HERE / "build/libbkmp2.so").resolve())
    }
    spec.data["parallel"] = {
        "backend": "process" if workers > 1 else "serial",
        "trajectory_workers": workers,
        "bead_workers": 1,
        "threads_per_rank": 1,
    }
    spec.data["output"]["directory"] = str(Path(output).resolve() / "run")
    if profile == "extended":
        spec.data["sampling"].update(
            equilibration="250 fs",
            production="2000 fs" if stage == "rate" else "1000 fs",
        )
        spec.data["analysis"]["bootstrap"] = 40
        if stage == "rate":
            spec.data["recrossing"].update(
                parent_samples=128,
                child_pairs=4,
                equilibration="500 fs",
                parent_interval="20 fs",
                child_time="150 fs",
                bootstrap=200,
            )
    return spec.resolve()


def _streams(directory):
    import h5py

    values = []
    radii = []
    mismatches = []
    with h5py.File(Path(directory) / "fes.h5", "r") as h:
        for key in sorted(h["streams"], key=int):
            group = h["streams"][key]
            window, replica = int(group.attrs["window"]), int(group.attrs["replica"])
            samples = np.asarray(group["samples"])
            values.append(samples)
            checkpoint = (
                Path(directory)
                / "checkpoints"
                / (f"window_{window:05d}_replica_{replica:04d}.npz")
            )
            with np.load(checkpoint, allow_pickle=False) as data:
                saved = np.asarray(data["values"])
                exact = np.array_equal(samples, saved)
                prefix = len(saved) <= len(samples) and np.array_equal(
                    samples[: len(saved)], saved
                )
                if not exact:
                    mismatches.append(
                        {
                            "window": window,
                            "replica": replica,
                            "hdf5_samples": len(samples),
                            "checkpoint_samples": len(saved),
                            "checkpoint_elapsed_steps": int(data["elapsed"]),
                            "exact_prefix": bool(prefix),
                        }
                    )
                if not prefix:
                    raise ValueError(
                        f"Final HDF5/checkpoint sample values disagree: {checkpoint}"
                    )
                q = np.asarray(data["q"])
                radii.append(
                    float(np.sqrt(np.mean(np.sum((q - q.mean(0)) ** 2, axis=-1))))
                )
    return (
        values,
        np.array(radii),
        {
            "passed": not mismatches,
            "streams_checked": len(values),
            "exact_sample_count_and_values": not mismatches,
            "mismatches": mismatches,
            "authoritative_samples": "final fes.h5 streams; checkpoint prefixes are never substituted",
        },
    )


def _sampling_diagnostics(directory):
    summary = json.loads((Path(directory) / "summary.json").read_text())
    with np.load(Path(directory) / "fes.npz", allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}
    streams, radii, integrity = _streams(directory)
    effective = np.asarray(data["effective_samples_per_stream"])
    return (
        {
            "connected": bool(summary["overlap_connected"]),
            "streams": len(streams),
            "samples": int(summary["samples"]),
            "effective_samples_minimum": float(effective.min()),
            "effective_samples_median": float(np.median(effective)),
            "effective_samples_sum": float(effective.sum()),
            "statistical_inefficiency_maximum": float(
                np.max(data["statistical_inefficiency"])
            ),
            "weight_effective_samples": float(summary["weight_effective_samples"]),
            "supported_bins": int(np.count_nonzero(~data["mask"])),
            "masked_fraction": float(np.mean(data["mask"])),
            "ring_radius_of_gyration_minimum_bohr": float(radii.min()),
            "ring_radius_of_gyration_median_bohr": float(np.median(radii)),
            "bootstrap_replicates": int(summary["bootstrap_replicates"]),
            "checkpoint_integrity": integrity,
        },
        data,
        streams,
    )


def _fes_report(config, output, result):
    diagnostics, data, streams = _sampling_diagnostics(output / "run")
    f = data["free_energy"]
    se = data["standard_error"]
    mask = data["mask"]
    common = (~mask) & (~mask.T) & np.isfinite(f) & np.isfinite(f.T)
    difference = f - f.T
    rms = float(np.sqrt(np.mean(difference[common] ** 2))) if common.any() else None
    denominator = np.sqrt(se * se + se.T * se.T)
    available = common & np.isfinite(denominator) & (denominator > 1e-14)
    z = abs(difference[available]) / denominator[available]
    diagnostics["permutation_mirror"] = {
        "identity": "F(r01,r12)=F(r12,r01), atom 0/2 exchange",
        "common_supported_bins": int(common.sum()),
        "rms_hartree": rms,
        "rms_kbt": rms * config["beta"] if rms is not None else None,
        "median_absolute_diagonal_error_z": float(np.median(z)) if len(z) else None,
        "z_note": "Descriptive standardized differences; bin-to-bin bootstrap covariance is not retained. No symmetry averaging was applied.",
    }
    values = np.concatenate(streams)
    ts_count = int(np.count_nonzero(np.all(abs(values - 1.7570000251) < 0.15, axis=1)))
    diagnostics["sample_count_near_ts"] = ts_count
    gates = {
        "real_ring_polymer": config["beads"] > 1
        and diagnostics["ring_radius_of_gyration_minimum_bohr"] > 1e-6,
        "overlap_connected": diagnostics["connected"],
        "minimum_stream_effective_samples_ge_5": diagnostics[
            "effective_samples_minimum"
        ]
        >= 5,
        "two_dimensional_supported_surface": f.ndim == 2
        and diagnostics["supported_bins"] >= 60,
        "supported_bins_finite": bool(np.isfinite(f[~mask]).all()),
        "mirror_domain_observed": int(common.sum()) >= 30,
        "ts_region_sampled": ts_count >= 30,
        "bootstrap_available": diagnostics["bootstrap_replicates"] >= 5,
        "final_samples_match_checkpoints": diagnostics["checkpoint_integrity"][
            "passed"
        ],
    }
    return {
        "diagnostics": diagnostics,
        "gates": gates,
        "cv_names": ["centroid_r01", "centroid_r12"],
        "cv_units": ["bohr", "bohr"],
        "window_ranges_bohr": [[1.35, 2.35], [1.35, 2.35]],
        "histogram_ranges_bohr": [[1.1, 2.8], [1.1, 2.8]],
        "measure": "centroid distance probability density ds1 ds2; no radial-Jacobian subtraction",
        "files": {
            "surface": "run/fes.npz",
            "samples": "run/fes.h5",
            "table": "run/fes.csv",
        },
    }


def _rate_report(config, system, output, result):
    diagnostics, data, streams = _sampling_diagnostics(output / "run/umbrella")
    prepared = json.loads((output / "run/preparation.json").read_text())
    prepared_config = copy.deepcopy(config)
    prepared_config["reaction"] = prepared["reaction"]
    surface = rate_coordinate(prepared_config, system).dividing_surface(
        result["pmf"]["xi_star"]
    )
    constraints = []
    branches = [0, 0]
    for path in sorted((output / "run/checkpoints").glob("parent_*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            for q in saved["snapshots"]:
                centroid = q.mean(0)
                constraints.append(abs(surface.centroid_value_gradient(centroid)[0]))
                branch = int(
                    np.linalg.norm(centroid[0] - centroid[2])
                    < np.linalg.norm(centroid[0] - centroid[1])
                )
                branches[branch] += 1
    curve = {}
    recrossing_path = output / "run/recrossing.npz"
    if recrossing_path.exists():
        with np.load(recrossing_path, allow_pickle=False) as saved:
            curve = {key: saved[key] for key in saved.files}
    kappa = curve.get("kappa", np.array([]))
    den = curve.get("parent_denominator", np.array([]))
    samples = np.concatenate(streams)[:, 0]
    xi = result["pmf"]["xi_star"]
    diagnostics.update(
        {
            "maximum_parent_constraint_residual_bohr": (
                max(constraints) if constraints else None
            ),
            "parent_surface_branch_counts": branches,
            "reactant_reference_sample_count": int(
                np.count_nonzero(abs(samples) < 0.05)
            ),
            "dividing_surface_sample_count": int(
                np.count_nonzero(abs(samples - xi) < 0.025)
            ),
            "kappa_first": float(kappa[0]) if len(kappa) else None,
            "kappa_last": float(kappa[-1]) if len(kappa) else None,
            "kappa_range": float(np.ptp(kappa)) if len(kappa) else None,
            "maximum_endpoint_energy_error_hartree_per_bead": result.get(
                "maximum_endpoint_energy_error_hartree_per_bead"
            ),
            "maximum_endpoint_energy_error_kbt": result.get(
                "maximum_endpoint_energy_error_kbt"
            ),
        }
    )
    error = result.get("maximum_endpoint_energy_error_kbt")
    measured = result.get("kappa")
    gates = {
        "overlap_connected": diagnostics["connected"],
        "minimum_stream_effective_samples_ge_5": diagnostics[
            "effective_samples_minimum"
        ]
        >= 5,
        "positive_pmf_barrier": 0 < result["pmf"]["free_energy_barrier_hartree"] < 0.2,
        "reactant_reference_sampled": diagnostics["reactant_reference_sample_count"]
        >= 30,
        "dividing_surface_sampled": diagnostics["dividing_surface_sample_count"] >= 30,
        "parent_constraint_satisfied": bool(constraints) and max(constraints) < 1e-8,
        "both_equivalent_ts_components_initialized": min(branches) > 0,
        "positive_initial_flux": bool(len(den)) and bool(np.all(den > 0)),
        "actual_child_ensemble": result.get("child_trajectories", 0) >= 128,
        "nontrivial_positive_recrossing": measured is not None
        and 0 < measured < 0.995
        and diagnostics["kappa_range"] > 0.002,
        "tail_platform_observed": bool(result.get("plateau_detected", False)),
        "offset_independent_endpoint_energy_error_below_0p1_kbt": error is not None
        and error < 0.1,
        "final_samples_match_checkpoints": diagnostics["checkpoint_integrity"][
            "passed"
        ],
    }
    analyze = import_object(str(HERE / "pmf_diagnostics.py") + ":analyze_run")
    comparison = analyze(output / "run", bins=(50, 100, 150), bootstrap=40, seed=73019)
    write_json(output / "pmf-diagnostics.json", comparison)
    compact = {
        key: comparison[key]
        for key in (
            "fixed_xi_star",
            "reference_xi",
            "measure",
            "common_histogram_range",
            "maximum_absolute_skewness",
            "maximum_absolute_excess_kurtosis",
            "extreme_windows",
            "mbar",
            "bootstrap",
            "independent_validation",
            "converged",
            "caveats",
        )
    }
    compact["ui"] = {
        k: v
        for k, v in comparison["ui"].items()
        if k not in ("curve", "bootstrap_barriers_hartree")
    }
    compact["per_bins"] = [
        {
            k: v
            for k, v in entry.items()
            if k
            not in (
                "curve",
                "bootstrap_barriers_hartree",
                "paired_mbar_minus_ui_draws_hartree",
            )
        }
        for entry in comparison["per_bins"]
    ]
    primary = next(entry for entry in comparison["per_bins"] if entry["bins"] == 100)
    return {
        "diagnostics": diagnostics,
        "gates": gates,
        "pmf": result.get("pmf"),
        "pmf_estimator_comparison": compact,
        "secondary_fixed_surface_mbar_rate_estimate_cm3_molecule_s": primary[
            "same_kappa_rate_estimate_cm3_s"
        ],
        "secondary_mbar_conditional_rate_95_ci_cm3_molecule_s": primary[
            "conditional_rate_percentile_95_ci_cm3_s"
        ],
        "analysis_warnings": [
            "The default rate PMF uses Gaussian umbrella integration; inspect the same-surface MBAR comparison and non-Gaussian window moments before interpreting its numerical rate.",
            "The secondary histogram-MBAR estimate reuses these same trajectories and the unchanged dividing surface/kappa; it is not independent validation or a converged rate. Its reported interval conditions on kappa.",
        ],
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
        "finite_time_rate_estimate_cm3_molecule_s": result.get("rate_estimate_cm3_s"),
        "rate_standard_error_cm3_molecule_s": result.get("rate_standard_error_cm3_s"),
        "surface_definition": "s1 = max(r12-r01, r12-r02); both equivalent exits included, no extra factor two",
        "symmetry_factor": 1.0,
        "electronic_weight": 1.0,
        "files": {
            "pmf": "run/rate_pmf.npz",
            "recrossing": "run/recrossing.npz",
            "umbrella": "run/umbrella/fes.npz",
            "samples": "run/umbrella/fes.h5",
            "pmf_estimator_diagnostics": "pmf-diagnostics.json",
        },
    }


def run_benchmark(stage, output, workers=2, profile="ci", library=None, resume=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {
        "schema": "pyrpmd.h3.rpmd-regression.v1",
        "task": stage,
        "profile": profile,
        "passed": False,
        "converged": False,
        "scope": "Short real BKMP2 physical-system regression; not bead/timestep/Rinf/statistical convergence or a literature rate validation",
    }
    try:
        config, system = configuration(stage, output, workers, profile, library)
        provider = import_object(str(HERE / "pes.py") + ":build")(
            **config["potential"]["params"]
        )
        report["pes"] = provider.metadata
        provider.close()
        report["settings"] = {
            "temperature_K": config["temperature_K"],
            "beads": config["beads"],
            "timestep_au": config["dt_au"],
            "timestep_fs": config["dt_au"] * AU_TIME_FS,
            "workers": workers,
            "sampling": config["sampling"],
            "reaction": config.get("reaction", {}),
            "native": native_capabilities(),
            "seed": config["seed"],
        }
        report.update(
            {
                key: report["settings"][key]
                for key in ("temperature_K", "beads", "timestep_fs")
            }
        )
        print(
            json.dumps(
                {
                    "event": "start",
                    "stage": stage,
                    "profile": profile,
                    "output": str(output),
                    "beads": config["beads"],
                    "temperature_K": config["temperature_K"],
                }
            ),
            flush=True,
        )
        result = run_resolved(config, system, output=output / "run", resume=resume)
        details = (
            _fes_report(config, output, result)
            if stage == "fes2d"
            else _rate_report(config, system, output, result)
        )
        report.update(details)
        report["workflow_status"] = result["status"]
        report["warnings"] = result.get("warnings", []) + report.pop(
            "analysis_warnings", []
        )
        report["gates"]["native_kernel_used"] = native_capabilities()["available"]
        report["passed"] = all(report["gates"].values())
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report.setdefault("gates", {})["workflow_completed"] = False
        traceback.print_exc()
    report["wall_seconds"] = time.perf_counter() - started
    write_json(output / "benchmark.json", report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fes2d", "rate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--profile", choices=("ci", "extended"), default="ci")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run_benchmark(
        args.stage, args.output, args.workers, args.profile, args.library, args.resume
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
