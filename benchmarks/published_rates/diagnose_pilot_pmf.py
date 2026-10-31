"""Replay pilot PMF support, UI/MBAR, ESS and stationarity without any dynamics.

Pack original saved runs once, or replay the compact committed sample payloads.
Input files are read only. No potential, momentum sampling or propagation runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np
from threadpoolctl import threadpool_limits

from pyrpmd.analysis.fes import mbar_rate_pmf, umbrella_integration
from pyrpmd.constants import AU_RATE_TO_CM3_S
from pyrpmd.workflows.fes import jsonable

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "pyrpmd.pilot-pmf-replay.v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack(label, run, destination):
    if re.fullmatch(r"[a-z0-9_-]+", label) is None:
        raise ValueError("pilot label must contain lowercase letters, digits, underscores or hyphens")
    run = Path(run).resolve()
    files = {name: run/name for name in (
        "input/resolved.json", "rate_pmf.json", "summary.json", "umbrella/fes.h5")}
    config = json.loads(files["input/resolved.json"].read_text())
    pmf = json.loads(files["rate_pmf.json"].read_text())
    summary = json.loads(files["summary.json"].read_text())
    registry = ROOT/"benchmarks/published_rates/references.json"
    reference = next(case for case in json.loads(registry.read_text())["cases"]
                     if case["id"] == "h_h2_bkmp2_rpmd_300k")
    with h5py.File(files["umbrella/fes.h5"], "r") as saved:
        groups = [saved["streams"][str(i)] for i in range(len(saved["streams"]))]
        samples = [group["samples"][:, 0] for group in groups]
        payload = {
            "samples": np.concatenate(samples),
            "counts": np.array([len(sample) for sample in samples], dtype=np.int64),
            "centers": np.array([group["center"][0] for group in groups]),
            "stiffness": np.array([group["stiffness"][0, 0] for group in groups]),
            "replicas": np.array([group.attrs["replica"] for group in groups], dtype=np.int64),
            "windows": np.array([group.attrs["window"] for group in groups], dtype=np.int64),
            "edges": saved["edges_0"][:],
            "saved_effective_samples": saved["effective_samples_per_stream"][:],
            "saved_histogram_mask": saved["mask"][:],
        }
        measure = str(saved.attrs["measure"])
    metadata = {
        "schema": SCHEMA, "label": label, "original_run": str(run),
        "original_files": {name: {"path": str(path), "sha256": sha256(path)}
                           for name, path in files.items()},
        "payload_arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                                  "sha256_c_order_bytes": hashlib.sha256(value.tobytes()).hexdigest()}
                           for name, value in payload.items()},
        "beta_hartree_inverse": config["beta"], "temperature_K": config["temperature_K"],
        "beads": config["beads"], "measure": measure, "xi_star": pmf["xi_star"],
        "reference_xi": pmf["reference"], "saved_ui_barrier_hartree": pmf["free_energy_barrier_hartree"],
        "saved_kappa": summary["kappa"], "saved_rate_estimate_cm3_molecule_s": summary["rate_estimate_cm3_s"],
        "reactant_prefactor_au": pmf["reactant_prefactor_au"],
        "reference": {"id": reference["id"], "doi": reference["source"]["doi"],
                      "qtst_rate_cm3_molecule_s": reference["qtst_rate_cm3_molecule_s"],
                      "registry_sha256": sha256(registry)},
        "description": "Exact saved CV samples and umbrella definitions only; no PES or trajectory states needed.",
    }
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination/f"{label}-pmf-samples.npz"
    np.savez_compressed(path, metadata_json=np.array(json.dumps(metadata, sort_keys=True)), **payload)
    return path


def diagnose(path):
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"]))
        if metadata.get("schema") != SCHEMA or metadata.get("measure") != "rate_flux":
            raise ValueError("unexpected replay schema or PMF measure")
        arrays = {key: saved[key].copy() for key in metadata["payload_arrays"]}
    for name, value in arrays.items():
        expected = metadata["payload_arrays"][name]
        if (list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]
                or hashlib.sha256(value.tobytes()).hexdigest() != expected["sha256_c_order_bytes"]):
            raise ValueError(f"replay payload integrity mismatch: {name}")
    counts = arrays["counts"]
    if counts.ndim != 1 or np.any(counts < 4) or counts.sum() != len(arrays["samples"]):
        raise ValueError("replay sample counts are invalid")
    splits = np.cumsum(counts)[:-1]
    samples = list(np.split(arrays["samples"], splits))
    centers, stiffness, replicas = (arrays[name] for name in ("centers", "stiffness", "replicas"))
    edges = arrays["edges"]
    beta, xi = metadata["beta_hartree_inverse"], metadata["xi_star"]
    grid = np.linspace(metadata["reference_xi"], centers.max(), 401)
    subsets = {"all": samples,
               "first_half": [sample[:len(sample)//2] for sample in samples],
               "second_half": [sample[len(sample)//2:] for sample in samples]}
    ui = {}
    for label, subset in subsets.items():
        curve, _ = umbrella_integration(subset, centers, stiffness, beta, grid)
        curve -= curve[0]
        ui[label] = {"barrier_at_original_xi_hartree": float(np.interp(xi, grid, curve)),
                     "maximum_barrier_hartree": float(curve.max()),
                     "maximum_xi": float(grid[np.argmax(curve)]),
                     "free_energy_at_xi_0_25_0_50_0_75_0_90_hartree":
                         [float(np.interp(point, grid, curve)) for point in (.25, .5, .75, .9)]}
    for replica in np.unique(replicas):
        selected = np.flatnonzero(replicas == replica)
        curve, _ = umbrella_integration([samples[i] for i in selected], centers[selected], stiffness[selected], beta, grid)
        curve -= curve[0]
        ui[f"replica_{replica}"] = {"barrier_at_original_xi_hartree": float(np.interp(xi, grid, curve)),
                                   "maximum_barrier_hartree": float(curve.max()),
                                   "maximum_xi": float(grid[np.argmax(curve)])}
    with threadpool_limits(1):
        mbar = mbar_rate_pmf(
            samples, centers, stiffness, beta, reference=metadata["reference_xi"], xi_star=xi,
            bins=len(edges)-1, ranges=[float(edges[0]), float(edges[-1])],
            bootstrap=0, min_count=5, overlap_threshold=1e-6, measure="rate_flux",
            compare_ui=True, ui_points=401,
        )
    means = np.array([sample.mean() for sample in samples])
    sigma = np.array([sample.std() for sample in samples])
    half_drift = np.array([(sample[len(sample)//2:].mean()-sample[:len(sample)//2].mean())
                           /sample.std() for sample in samples])
    effective = arrays["saved_effective_samples"]
    histogram_centers = .5*(edges[1:]+edges[:-1])
    surface_windows = np.flatnonzero(np.isclose(centers, 1.))
    barrier = mbar["free_energy_barrier_hartree"]
    ui_barrier = ui["all"]["barrier_at_original_xi_hartree"]
    published_barrier = -np.log(metadata["reference"]["qtst_rate_cm3_molecule_s"]
                               /(metadata["reactant_prefactor_au"]*AU_RATE_TO_CM3_S))/beta
    return {
        "pilot": metadata["label"], "physical_validation": False,
        "replay_file": path.name, "replay_file_sha256": sha256(path),
        "original_input_files": metadata["original_files"],
        "reference": metadata["reference"],
        "temperature_K": metadata["temperature_K"], "beads": metadata["beads"],
        "beta_hartree_inverse": beta, "umbrella_stiffness_hartree": np.unique(stiffness).tolist(),
        "streams": len(samples), "samples_per_stream": np.unique(counts).tolist(),
        "original_xi_star": xi, "saved_kappa": metadata["saved_kappa"],
        "saved_candidate_rate_cm3_molecule_s": metadata["saved_rate_estimate_cm3_molecule_s"],
        "published_qtst_implied_barrier_with_same_prefactor_hartree": float(published_barrier),
        "ui": ui, "ui_replay_absolute_error_hartree": abs(ui_barrier-metadata["saved_ui_barrier_hartree"]),
        "mbar": {"ready": bool(mbar["ready"]), "status": mbar["status"],
                 "barrier_hartree": barrier, "minus_ui_barrier_hartree": None if barrier is None else barrier-ui_barrier,
                 "qtst_ratio_to_ui": None if barrier is None else float(np.exp(-beta*(barrier-ui_barrier))),
                 "overlap_connected": bool(mbar["overlap_connected"]),
                 "support": mbar["support"], "solver": mbar["solver"],
                 "bootstrap_requested": 0},
        "histogram": {"masked_bin_centers": histogram_centers[arrays["saved_histogram_mask"]].tolist(),
                      "bin_count": len(edges)-1},
        "effective_samples": {"quantiles_0_25_50_75_100": np.quantile(effective, [0, .25, .5, .75, 1]).tolist(),
                              "streams_below_20": int(np.count_nonzero(effective < 20))},
        "stationarity": {"median_absolute_half_mean_drift_over_sample_sigma": float(np.median(abs(half_drift))),
                         "maximum_absolute_half_mean_drift_over_sample_sigma": float(max(abs(half_drift))),
                         "median_far_entrance_sigma_xi": float(np.median(sigma[(centers > .05) & (centers < .8)])),
                         "flat_free_energy_umbrella_sigma_xi": float(np.sqrt(1/(beta*stiffness[0]))),
                         "center_1_streams": [{"window": int(arrays["windows"][i]), "replica": int(replicas[i]),
                                               "mean": float(means[i]), "sigma": float(sigma[i]),
                                               "min": float(samples[i].min()), "max": float(samples[i].max())}
                                              for i in surface_windows]},
        "interpretation": [
            "No new dynamics were run; temporal halves and replicas are diagnostic subsamples, not a convergence proof.",
            "A connected overlap graph does not establish observed dividing-surface support or ergodicity.",
            "A supported MBAR point estimate does not establish sufficiently long equilibrium sampling.",
            "The target-implied barrier is derived from a published QTST value and this calculation's prefactor; it is not a separately published barrier.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "DIRECTORY"), default=[])
    parser.add_argument("--replay", action="append", type=Path, default=[])
    parser.add_argument("--pack-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.run and args.pack_directory is None:
        parser.error("--run requires --pack-directory")
    paths = list(args.replay)
    paths.extend(pack(label, run, args.pack_directory) for label, run in args.run)
    if not paths:
        parser.error("provide --run or --replay")
    if args.output.resolve() in [path.resolve() for path in paths]:
        parser.error("output must not replace a replay input")
    sources = ["benchmarks/published_rates/diagnose_pilot_pmf.py", "src/pyrpmd/analysis/fes.py",
               "src/pyrpmd/analysis/diagnostics.py", "src/pyrpmd/constants.py"]
    report = {"schema": "pyrpmd.pilot-estimator-diagnostic.v1", "physical_validation": False,
              "new_dynamics_runs": 0, "source_sha256": {name: sha256(ROOT/name) for name in sources},
              "records": [diagnose(path) for path in paths]}
    original_inputs = {Path(item["path"]).resolve() for record in report["records"]
                       for item in record["original_input_files"].values()}
    if args.output.resolve() in original_inputs:
        parser.error("output must not replace an original saved input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(report), indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output), "pilots": [
        {"name": record["pilot"], "ui_barrier": record["ui"]["all"]["barrier_at_original_xi_hartree"],
         "mbar_status": record["mbar"]["status"], "mbar_barrier": record["mbar"]["barrier_hartree"]}
        for record in report["records"]]}, indent=2))


if __name__ == "__main__":
    main()
