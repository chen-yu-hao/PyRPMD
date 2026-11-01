#!/usr/bin/env python3
"""Analyze water checkpoints without treating origins or segments as replicas.

Produces compact, replayable scalar/correlation evidence. Raw trajectories are
still required to recompute correlations from coordinates. Adjacent NVE segments
share a configuration chain even when their momenta were freshly sampled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.water.campaign import atomic_json, atomic_npz, sha256, verify_manifest
from pyrpmd.analysis.diffusion import analyze_diffusion
from pyrpmd.constants import BOHR_ANGSTROM, FS_TO_AU, KCAL_MOL_TO_AU


def load_observations(directory, manifest):
    pieces = {}
    for record in manifest["chunks"]:
        with np.load(directory / record["file"], allow_pickle=False) as data:
            for key in data.files:
                if key not in ("q", "p", "state_step", "state_time", "box_angstrom", "rng_json"):
                    pieces.setdefault(key, []).append(data[key])
    return {key: np.concatenate(values, axis=0) for key, values in pieces.items()}


def rdf_oxygen(positions_bohr, box_angstrom, *, bin_width=.025):
    positions = np.asarray(positions_bohr) * BOHR_ANGSTROM
    if positions.ndim != 3 or positions.shape[-1] != 3 or len(positions) < 1:
        raise ValueError("RDF needs nonempty [frame,oxygen,3]")
    n = positions.shape[1]
    edges = np.arange(0, box_angstrom / 2 + bin_width / 2, bin_width)
    histogram = np.zeros(len(edges) - 1)
    i, j = np.triu_indices(n, k=1)
    for frame in positions:
        dr = frame[i] - frame[j]
        dr -= box_angstrom * np.rint(dr / box_angstrom)
        histogram += np.histogram(np.linalg.norm(dr, axis=-1), edges)[0]
    shells = 4 * np.pi / 3 * (edges[1:]**3 - edges[:-1]**3)
    ideal_pairs = len(positions) * (n * (n - 1) / 2) * shells / box_angstrom**3
    return (edges[:-1] + edges[1:]) / 2, histogram / ideal_pairs


def summarize_series(values, mask):
    selected = np.asarray(values)[mask]
    if not len(selected):
        return None
    return {"mean": float(selected.mean()), "standard_deviation": float(selected.std()),
            "minimum": float(selected.min()), "maximum": float(selected.max()), "frames": len(selected)}


def literature_diffusion_reference(beads, isotope):
    """Return only a table entry actually reported for this dynamics/isotope."""
    if beads == 1:
        value, error, table = (.199, .002, "IV") if isotope == "H2O" else (None, None, None)
    else:
        value, error, table = (.221 if isotope == "H2O" else .172), .001, "III"
    return {"diffusion_angstrom2_ps": value, "standard_error_angstrom2_ps": error,
            "source": "https://arxiv.org/pdf/1011.1047", "table": table,
            "comparison_eligible": False, "agreement_test": None,
            "unavailable_reason": "The paper does not report a classical D2O diffusion target." if value is None else None}


def analyze(directory, output, *, make_plot=True):
    directory, output = Path(directory), Path(output)
    manifest = json.loads((directory / "manifest.json").read_text())
    verify_manifest(directory, manifest)
    data = load_observations(directory, manifest)
    config = manifest["config"]
    output.mkdir(parents=True, exist_ok=True)
    dt_ps = config["dt_fs"] * config["record_stride"] / 1000
    n = config["cells_per_axis"] ** 3
    time_ps = data["step"] * config["dt_fs"] / 1000
    equilibrium_ps = config["equilibration_steps"] * config["dt_fs"] / 1000
    sampling = data["segment"] == -1
    scalar_keys = [key for key, value in data.items() if value.ndim == 1]
    atomic_npz(output / "observables.npz", **{key: data[key] for key in scalar_keys})
    report = {"schema": "pyrpmd.water-campaign-analysis.v1", "conditions": config,
              "box_angstrom": manifest["box_angstrom"], "molecules": n,
              "committed_steps": manifest["committed_steps"], "complete": manifest["complete"],
              "elapsed_seconds": manifest["elapsed_seconds"],
              "physical_converged": False, "literature_reproduced": False,
              "validated_diffusion_angstrom2_ps": None, "validated_standard_error_angstrom2_ps": None,
              "equilibration": {}, "segments": [], "fit_window_comparison": [],
              "eligibility": {"published_equilibration_ps": 100,
                              "this_equilibration_ps": equilibrium_ps,
                              "published_NVE_segments": 2000, "published_segment_ps": 4,
                              "published_independent_batches": 20, "this_independent_batches": 0,
                              "source_dynamics": "RPMD P32 (classical comparison P1)",
                              "this_dynamics": "real-mass RPMD" if config["beads"] > 1 else "classical MD",
                              "source_electrostatic_contraction_angstrom": 5,
                              "this_contraction": None,
                              "source_MTS_inter_fs": .5, "source_MTS_intra_fs": .1,
                              "this_MTS": None}}
    report["analysis_source_sha256"] = {name: sha256(ROOT / name) for name in (
        "benchmarks/water/analyze_campaign.py", "src/pyrpmd/analysis/diffusion.py",
        "src/pyrpmd/constants.py")}
    window_ps = min(5.0, equilibrium_ps / 2)
    report["equilibration_window_duration_ps"] = window_ps
    for label, lower, upper in (("first_window", 0, window_ps),
                                ("last_window", max(0, equilibrium_ps - window_ps), equilibrium_ps)):
        mask = sampling & (time_ps > lower) & (time_ps <= upper)
        report["equilibration"][label] = {key: summarize_series(data[key], mask) for key in scalar_keys if key not in ("step", "segment", "segment_step")}
    curves = {}
    early_end = min(1.0, equilibrium_ps / 2)
    early = np.flatnonzero(sampling & (time_ps <= early_end))
    late = np.flatnonzero(sampling & (time_ps > max(0, equilibrium_ps - window_ps)))
    if len(early) and len(late):
        stride = max(1, round(.1 / dt_ps))
        r, g_early = rdf_oxygen(data["oxygen_centroid_bohr"][early[::stride]], manifest["box_angstrom"])
        _, g_late = rdf_oxygen(data["oxygen_centroid_bohr"][late[::stride]], manifest["box_angstrom"])
        curves.update(rdf_r_angstrom=r, rdf_early=g_early, rdf_late=g_late)
        region = (r >= 2.4) & (r <= 3.3)
        report["oxygen_rdf"] = {"coordinate": "oxygen centroid; equals physical coordinate only for P=1",
                                 "normalization": "unordered N(N-1)/2 pairs / cubic volume; ideal finite sample g=1",
                                 "bin_width_angstrom": .025, "frame_spacing_ps": stride * dt_ps,
                                 "early_interval_ps": [0, early_end],
                                 "late_interval_ps": [equilibrium_ps - window_ps, equilibrium_ps],
                                 "late_peak_position_angstrom": float(r[region][g_late[region].argmax()]),
                                 "late_peak_height": float(g_late[region].max()),
                                 "literature_curve_digitized": False}
    all_fits = {}
    for segment in sorted(set(data["segment"]) - {-1}):
        select = data["segment"] == segment
        segment_steps = data["segment_step"][select]
        if not len(segment_steps) or segment_steps[0] != 0:
            raise ValueError("NVE segment lacks its initial observation")
        if len(segment_steps) < 16:
            continue
        if np.any(np.diff(segment_steps) != config["record_stride"]):
            raise ValueError("nonuniform or duplicate NVE observations")
        positions = data["molecular_com_bohr"][select]
        velocities = data["molecular_velocity_au"][select]
        times = segment_steps * config["dt_fs"] * FS_TO_AU
        maximum_ps = min(2.0, (len(times) - 1) * dt_ps / 2)
        max_lag = int(round(maximum_ps / dt_ps))
        result = analyze_diffusion(positions, times, velocities=velocities,
                                   max_lag=max_lag, fit_range=(maximum_ps / 2 * 1000 * FS_TO_AU,
                                                            maximum_ps * 1000 * FS_TO_AU), bootstrap=0)
        hamiltonian = data["hamiltonian_per_bead_hartree"][select]
        delta = (hamiltonian - hamiltonian[0]) / (n * KCAL_MOL_TO_AU)
        row = {"segment": int(segment), "duration_ps": float(segment_steps[-1] * config["dt_fs"] / 1000),
               "frames": len(times), "status": result["status"],
               "max_energy_change_per_water_kcal_mol": float(np.max(abs(delta))),
               "energy_linear_drift_per_water_kcal_mol_ps": float(np.polyfit(segment_steps * config["dt_fs"] / 1000, delta, 1)[0]),
               "temperature_centroid_K": summarize_series(data["temperature_centroid_K"], select),
               "raw_D_MSD_angstrom2_ps": result["raw_fit"]["diffusion_m2_s"] * 1e8,
               "raw_D_VACF_at_maxlag_angstrom2_ps": result["vacf_integral_m2_s"][-1] * 1e8,
               "log_msd_slope": result["raw_fit"]["log_msd_slope"],
               "half_window_relative_difference": result["raw_fit"]["relative_half_window_difference"],
               "independent_replica": False}
        report["segments"].append(row)
        curves[f"segment_{segment}_time_ps"] = np.asarray(result["lag_time_fs"]) / 1000
        curves[f"segment_{segment}_msd_angstrom2"] = np.asarray(result["msd_au"]) * BOHR_ANGSTROM**2
        curves[f"segment_{segment}_vacf_au"] = result["vacf_au"]
        curves[f"segment_{segment}_D_VACF_angstrom2_ps"] = np.asarray(result["vacf_integral_m2_s"]) * 1e8
        for lower, upper in ((.1, .5), (.5, 1), (1, 2)):
            if upper > maximum_ps:
                continue
            fit = analyze_diffusion(positions, times, max_lag=max_lag,
                                    fit_range=(lower * 1000 * FS_TO_AU, upper * 1000 * FS_TO_AU), bootstrap=0)
            all_fits.setdefault((lower, upper), []).append({"segment": int(segment),
                "raw_D_angstrom2_ps": fit["raw_fit"]["diffusion_m2_s"] * 1e8,
                "log_msd_slope": fit["raw_fit"]["log_msd_slope"], "status": fit["status"]})
    for window, fits in all_fits.items():
        values = np.asarray([fit["raw_D_angstrom2_ps"] for fit in fits])
        report["fit_window_comparison"].append({"window_ps": list(window), "segments": fits,
            "mean_raw_D_angstrom2_ps": float(values.mean()),
            "segment_standard_deviation_angstrom2_ps": float(values.std(ddof=1)) if len(values) > 1 else None,
            "conditional_segment_SEM_angstrom2_ps": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
            "uncertainty_warning": "Adjacent segments are correlated; this conditional spread is not a validated standard error."})
    report["literature_target"] = literature_diffusion_reference(config["beads"], config["isotope"])
    report["limitations"] = [
        "The equilibration segment begins on a random oriented cubic grid; thermal stationarity does not establish structural equilibration.",
        "Production segments share successive configurations. Fresh Maxwell momenta do not prove independence of slow liquid structure.",
        "Overlapping time origins and different molecules are not independent uncertainty units.",
        "No accepted diffusion value, uncertainty, infinite-size extrapolation, or literature reproduction is inferred.",
        "The printed Table I parameters, specified isotope masses, and full per-bead PME are explicit; numerical identity to the original source is not claimed.",
        "No PA-CMD infrared spectrum is computed. Ordinary RPMD spectra have a different dynamics method."]
    atomic_npz(output / "curves.npz", **curves)
    shutil.copyfile(directory / "model.xml", output / "model.xml")
    last = directory / manifest["chunks"][-1]["file"]
    with np.load(last, allow_pickle=False) as checkpoint:
        atomic_npz(output / "endpoint.npz", **{key: checkpoint[key] for key in ("q", "p", "state_step", "state_time", "box_angstrom", "rng_json")})
    atomic_json(output / "raw-manifest.json", manifest)
    report["artifacts"] = {name: {"sha256": sha256(output / name), "bytes": (output / name).stat().st_size}
                           for name in ("observables.npz", "curves.npz", "model.xml", "endpoint.npz", "raw-manifest.json")}
    atomic_json(output / "report.json", report)
    if make_plot:
        plot(output, report)
    return report


def replay_compact(output):
    """Verify retained bytes and independently reconstruct recorded estimators.

    This checks analysis of saved correlations, not the raw coordinate-to-
    correlation transform. The separate raw-data archive supports that check.
    """
    output = Path(output)
    report = json.loads((output / "report.json").read_text())
    for name, identity in report["artifacts"].items():
        path = output / name
        if path.name != name or sha256(path) != identity["sha256"] or path.stat().st_size != identity["bytes"]:
            raise ValueError(f"compact evidence checksum mismatch: {name}")
    from scipy.integrate import cumulative_trapezoid
    with np.load(output / "curves.npz", allow_pickle=False) as curves:
        for row in report["segments"]:
            prefix = f"segment_{row['segment']}_"
            t = curves[prefix + "time_ps"]
            velocity_factor = BOHR_ANGSTROM * 1000 * FS_TO_AU
            integral = cumulative_trapezoid(curves[prefix + "vacf_au"], t, initial=0) * velocity_factor**2 / 3
            np.testing.assert_allclose(integral, curves[prefix + "D_VACF_angstrom2_ps"], atol=2e-12, rtol=2e-12)
        for window in report["fit_window_comparison"]:
            low, high = window["window_ps"]
            for row in window["segments"]:
                prefix = f"segment_{row['segment']}_"
                t = curves[prefix + "time_ps"]
                mask = (t >= low) & (t <= high * (1 + 1e-12)) & (t > 0)
                slope = np.polyfit(t[mask], curves[prefix + "msd_angstrom2"][mask], 1)[0] / 6
                np.testing.assert_allclose(slope, row["raw_D_angstrom2_ps"], atol=2e-12, rtol=2e-12)
    return {"artifact_integrity_passed": True, "correlation_estimator_replay_passed": True,
            "raw_coordinate_replay_requires_archive": True, "physical_converged": False,
            "literature_reproduced": False}


def plot(output, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    values = np.load(output / "observables.npz", allow_pickle=False)
    curves = np.load(output / "curves.npz", allow_pickle=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3), constrained_layout=True)
    t = values["step"] * report["conditions"]["dt_fs"] / 1000
    axes[0, 0].plot(t, values["temperature_centroid_K"], lw=.6, alpha=.7, color="#2563eb")
    axes[0, 0].axhline(298, color="#1f2937", ls="--", lw=1)
    axes[0, 0].axvline(report["conditions"]["equilibration_steps"] * report["conditions"]["dt_fs"] / 1000, color="#b45309", lw=1)
    axes[0, 0].set(xlabel="Simulation time (ps)", ylabel="Centroid kinetic temperature (K)", title="Thermalization and NVE segments")
    if "rdf_r_angstrom" in curves.files:
        early_range = report["oxygen_rdf"]["early_interval_ps"]
        late_range = report["oxygen_rdf"]["late_interval_ps"]
        axes[0, 1].plot(curves["rdf_r_angstrom"], curves["rdf_early"], label=f"{early_range[0]:g}–{early_range[1]:g} ps", color="#94a3b8")
        axes[0, 1].plot(curves["rdf_r_angstrom"], curves["rdf_late"], label=f"{late_range[0]:g}–{late_range[1]:g} ps", color="#2563eb")
        axes[0, 1].set(xlim=(2, 7), xlabel="O–O distance (Å)", ylabel="gOO(r)", title="Liquid structure: finite pilot")
        axes[0, 1].legend(frameon=False)
    for row in report["segments"]:
        key = f"segment_{row['segment']}_"
        lag = curves[key + "time_ps"]
        axes[1, 0].plot(lag, curves[key + "msd_angstrom2"], label=f"Segment {row['segment'] + 1}")
        axes[1, 1].plot(lag, curves[key + "D_VACF_angstrom2_ps"], label=f"Segment {row['segment'] + 1}")
    axes[1, 0].set(xlabel="Lag time (ps)", ylabel="Molecular COM MSD (Å²)", title="Many origins; correlated observations")
    if report["literature_target"]["diffusion_angstrom2_ps"] is not None:
        axes[1, 1].axhline(report["literature_target"]["diffusion_angstrom2_ps"], ls="--", lw=1, color="#1f2937", label="Published target")
    axes[1, 1].set(xlabel="Integration limit (ps)", ylabel="VACF integral / 3 (Å²/ps)", title="Finite-time estimates; no convergence claim")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].legend(frameon=False)
    fig.suptitle(f"q-TIP4P/F · {report['molecules']} molecules · P={report['conditions']['beads']} · CPU pilot", fontsize=14)
    fig.savefig(output / "diagnostics.png", dpi=150)
    fig.savefig(output / "diagnostics.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--replay-compact", action="store_true")
    args = parser.parse_args()
    if args.replay_compact:
        print(json.dumps(replay_compact(args.run), indent=2))
        return
    if args.output is None:
        parser.error("--output is required when analyzing raw checkpoints")
    report = analyze(args.run, args.output, make_plot=not args.no_plot)
    print(json.dumps({key: report[key] for key in ("committed_steps", "complete", "physical_converged", "segments")}, indent=2))


if __name__ == "__main__":
    main()
