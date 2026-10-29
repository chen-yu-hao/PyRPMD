#!/usr/bin/env python3
"""Read-only, independent archive audit for the four formal H3 PMF cases.

No pyrpmd imports, fitting, resampling or trajectory propagation are performed.
Requires NumPy and h5py. JSON is written even when an individual file is corrupt;
exit 0 means all required checks passed, exit 1 means an audit failure.
Unavailable original compiled binaries are explicitly listed, not certified.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np


CASES = {
    "k400-a": (821071, 400.0, -0.05, 0.025, 47),
    "k400-b": (821072, 400.0, -0.05, 0.025, 47),
    "k800-a": (831071, 800.0, -0.06, 0.020, 59),
    "k800-b": (831072, 800.0, -0.06, 0.020, 59),
}
KCAL_PER_HARTREE = 627.5094740631


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_digest(value, compact=True):
    options = dict(sort_keys=True, allow_nan=False)
    if compact:
        options["separators"] = (",", ":")
    return hashlib.sha256(json.dumps(value, **options).encode()).hexdigest()


def clean(value):
    """Keep unavailable numerical results null, never NaN/Infinity or zero."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def subset_mismatches(expected, actual, prefix=""):
    result = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [prefix]
        for key, value in expected.items():
            name = f"{prefix}.{key}" if prefix else key
            if key not in actual:
                result.append(name + " (missing)")
            else:
                result.extend(subset_mismatches(value, actual[key], name))
    elif expected != actual:
        result.append(prefix)
    return result


def inefficiency_direct(values):
    """Same initial-positive-pair convention, independently computed by correlate."""
    x = np.asarray(values, float)
    x = x - np.mean(x)
    n = len(x)
    variance = np.dot(x, x) / n
    if variance <= np.finfo(float).tiny:
        return float(n)
    ac = np.correlate(x, x, mode="full")[n - 1 :] / np.arange(n, 0, -1) / variance
    total = 0.0
    for lag in range(1, n - 1, 2):
        pair = ac[lag] + ac[lag + 1]
        if not np.isfinite(pair) or pair <= 0:
            break
        total += pair
    return float(np.clip(1 + 2 * total, 1, n))


def audit(study, source_root=None):
    study = Path(study).resolve()
    report = {
        "schema": "pyrpmd.h3.independent-artifact-audit.v1",
        "study_directory": str(study),
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_script_sha256": sha256(__file__),
        "scope": "Read-only archive/array/hash/algebra verification; no MD, fitting or bootstrap.",
        "passed": False,
        "checks": [],
        "errors": [],
        "hashes": {},
        "streams": [],
        "npz_archives": [],
        "source_verification": {"verified": [], "unavailable": [], "mismatches": []},
        "limitations": [
            "Array equality confirms consistency of saved products, not independent MD correctness.",
            "A matching saved hash identifies bytes; it does not establish physical convergence.",
            "Original remote compiled binaries absent from this artifact cannot be independently hashed.",
            "Visual figure review is separate; this script does not certify figure rendering.",
        ],
    }

    def check(name, condition, detail=None):
        item = {"name": name, "passed": bool(condition)}
        if detail is not None:
            item["detail"] = clean(detail)
        report["checks"].append(item)
        if not item["passed"]:
            report["errors"].append(item)

    def load(name):
        path = study / name
        try:
            report["hashes"][name] = sha256(path)
            return json.loads(
                path.read_text(),
                parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + x)),
            )
        except Exception as exc:
            check("read:" + name, False, f"{type(exc).__name__}: {exc}")
            return None

    files = [
        "benchmark.json", "run/rate_pmf.json", "solver-check.json",
        "run/input/resolved.json", "run/input/system.json", "run/preparation.json",
        "run/manifest.json", "provenance/source-identity.json",
    ]
    loaded = [load(name) for name in files]
    if any(item is None for item in loaded):
        return report
    benchmark, pmf, solver, config, system, preparation, manifest, identity = loaded
    case = benchmark.get("case")
    report["case"] = case
    check("known_formal_case", case in CASES, case)
    if case not in CASES:
        return report
    seed, beta_k, start, spacing, windows = CASES[case]
    expected = dict(
        seed=seed, beta_k=beta_k, start=start, spacing=spacing, windows=windows,
        smoke=False, beads=16, replicas=2, temperature_K=1500.0, timestep_fs=0.1,
        equilibration_fs=300.0, production_fs=1000.0, sample_interval_fs=0.5,
        xi_star=1.00925, reference_xi=0.0, bins=170, bootstrap=40,
        histogram_range=[-0.3, 1.4], sensitivity_bins=[85, 340],
        parent_replicas=2, parent_samples_per_chain=32, child_pairs_per_parent=2,
        child_time_fs=80.0, recrossing_bootstrap=100,
    )
    mismatch = subset_mismatches(expected, benchmark.get("settings", {}))
    check("formal_prespecified_settings", not mismatch, mismatch)
    check("formal_not_smoke", benchmark.get("smoke") is False)
    check("formal_seed", benchmark.get("seed") == seed and config.get("seed") == seed)
    check("benchmark_schema", benchmark.get("schema") == "pyrpmd.h3.pmf-convergence.v1")
    expected_streams, samples_per_stream, expected_steps = windows * 2, 2000, 13000
    report["expected"] = {"streams": expected_streams, "samples_per_stream": 2000, "steps_per_stream": 13000}

    excluded = {"output", "parallel", "resume", "_resume", "_warnings", "_input_path", "_base_directory"}
    workflow_fp = json_digest({"config": {k: v for k, v in config.items() if k not in excluded}, "system": system}, compact=False)
    check("workflow_physics_fingerprint", workflow_fp == benchmark["fingerprints"]["workflow_physics"] == manifest["physics_fingerprint"], workflow_fp)
    check("source_identity_digest", json_digest(identity) == benchmark["fingerprints"]["source_identity"])
    check("solver_config_hash", report["hashes"]["run/input/resolved.json"] == solver["source_config_sha256"])

    # Reconstruct only the serialized preparation changes; do not evaluate the PES.
    sampled = copy.deepcopy(config)
    sampled["reaction"] = preparation["reaction"]
    reaction = sampled["reaction"]
    sampled["cvs"] = [{
        "name": "xi", "kind": "rpmdrate_xi", "estimator": "centroid",
        **{k: reaction[k] for k in ("fragments", "forming_bonds", "breaking_bonds", "rinf")},
    }]
    if reaction.get("equivalent_paths"):
        sampled["cvs"][0]["equivalent_paths"] = reaction["equivalent_paths"]
    sampled["umbrella"]["cvs"] = ["xi"]
    sampled["analysis"].update(measure="rate_flux", bootstrap=0)
    stream_excluded = {"parallel", "output", "analysis", "resume", "_resume", "_input_path", "_base_directory", "_warnings"}
    sampled_physics = {k: v for k, v in sampled.items() if k not in stream_excluded}
    identities, fingerprints, ess, moments = [], [], [], []
    h5path = study / "run/umbrella/fes.h5"
    try:
        actual_hash = sha256(h5path)
        report["hashes"]["run/umbrella/fes.h5"] = actual_hash
        check("hdf5_hash", actual_hash == benchmark["fingerprints"]["final_hdf5_sha256"] == solver["source_hdf5_sha256"])
        with h5py.File(h5path, "r") as h5:
            check("hdf5_rate_flux_measure", h5.attrs.get("measure") == "rate_flux")
            keys = sorted(h5["streams"], key=int)
            check("hdf5_stream_numbering", keys == [str(i) for i in range(expected_streams)])
            for key in keys:
                item = {"stream": int(key), "passed": False}
                report["streams"].append(item)
                try:
                    group = h5["streams"][key]
                    window, replica = int(group.attrs["window"]), int(group.attrs["replica"])
                    identities.append((window, replica))
                    values, energies = group["samples"][:], group["energies"][:]
                    center, stiffness = group["center"][:], group["stiffness"][:]
                    path = study / "run/umbrella/checkpoints" / f"window_{window:05d}_replica_{replica:04d}.npz"
                    item.update(window=window, replica=replica, sample_shape=list(values.shape), energy_shape=list(energies.shape), checkpoint=str(path.relative_to(study)), checkpoint_sha256=sha256(path))
                    checks = {
                        "sample_shape": values.shape == (samples_per_stream, 1),
                        "energy_shape": energies.shape == (samples_per_stream,),
                        "samples_finite": bool(np.isfinite(values).all()),
                        "energies_finite": bool(np.isfinite(energies).all()),
                        "center": center.shape == (1,) and abs(float(center[0]) - (start + spacing * window)) < 1e-12,
                        "stiffness": stiffness.shape == (1, 1) and abs(float(stiffness[0, 0]) * config["beta"] - beta_k) < 1e-9,
                    }
                    expected_fp = json_digest({
                        "schema": 1, "physics": sampled_physics, "sampling_measure": "rate_flux", "system": system,
                        "stream": {"window": window, "replica": replica, "center": center.tolist(), "stiffness": stiffness.tolist()},
                    })
                    with np.load(path, allow_pickle=False) as saved:
                        actual_fp = str(saved["fingerprint"])
                        fingerprints.append(actual_fp)
                        time_error = abs(float(saved["time"]) - expected_steps * config["dt_au"])
                        item.update(step=int(saved["step"]), elapsed=int(saved["elapsed"]), time_au=float(saved["time"]), time_error_au=time_error, actual_physics_fingerprint=actual_fp, expected_physics_fingerprint=expected_fp)
                        checks.update(
                            samples_exact=np.array_equal(values, saved["values"]),
                            energies_exact=np.array_equal(energies, saved["energies"]),
                            steps=int(saved["step"]) == expected_steps,
                            elapsed=int(saved["elapsed"]) == expected_steps,
                            clock=bool(np.isfinite(saved["time"])) and time_error <= 1e-6,
                            physics_fingerprint=actual_fp == expected_fp,
                        )
                        for name in ("q", "p"):
                            array = saved[name]
                            checks[name + "_shape"] = array.shape == (16, len(system["symbols"]), 3)
                            checks[name + "_finite"] = bool(np.isfinite(array).all())
                            item[name + "_shape"] = list(array.shape)
                    item["checks"] = {k: bool(v) for k, v in checks.items()}
                    item["passed"] = all(checks.values())
                    check(f"stream:{key}", item["passed"], [k for k, v in checks.items() if not v])
                    if values.shape == (samples_per_stream, 1) and np.isfinite(values).all():
                        x = values[:, 0]
                        g = inefficiency_direct(x)
                        sd = np.std(x)
                        z = (x - np.mean(x)) / sd if sd > 0 else np.full_like(x, np.nan)
                        ess.append(len(x) / g)
                        moments.append({"stream": int(key), "mean": float(np.mean(x)), "standard_deviation": float(sd), "skewness": float(np.mean(z**3)), "excess_kurtosis": float(np.mean(z**4) - 3), "statistical_inefficiency": g, "estimated_effective_samples": len(x) / g})
                except Exception as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
                    check(f"stream:{key}", False, item["error"])
        expected_ids = {(w, r) for w in range(windows) for r in range(2)}
        check("complete_unique_stream_identities", len(identities) == expected_streams and len(set(identities)) == len(identities) and set(identities) == expected_ids)
        check("checkpoint_fingerprint_list", fingerprints == benchmark["diagnostics"]["checkpoint_fingerprints"])
        report["sample_diagnostics"] = {"streams_read": len(report["streams"]), "samples_read": sum(v["sample_shape"][0] for v in report["streams"] if v.get("sample_shape")), "effective_samples_minimum": min(ess) if ess else None, "effective_samples_median": float(np.median(ess)) if ess else None, "effective_samples_sum": sum(ess), "moments": moments}
        if ess:
            for key in ("effective_samples_minimum", "effective_samples_median", "effective_samples_sum"):
                check("recomputed:" + key, np.isclose(report["sample_diagnostics"][key], benchmark["diagnostics"][key], rtol=1e-10, atol=1e-10))
    except Exception as exc:
        check("hdf5_or_stream_audit", False, f"{type(exc).__name__}: {exc}")

    # Decode every member of every NPZ, including child/parent checkpoint archives.
    # This catches truncated ZIP members independently of advertised saved counts.
    for path in sorted((study / "run").rglob("*.npz")):
        item = {"path": str(path.relative_to(study)), "readable": False}
        try:
            item.update(size_bytes=path.stat().st_size, sha256=sha256(path))
            with np.load(path, allow_pickle=False) as arrays:
                item["members"] = {k: {"shape": list(arrays[k].shape), "dtype": str(arrays[k].dtype)} for k in arrays.files}
            item["readable"] = True
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        report["npz_archives"].append(item)
        if not item["readable"]:
            check("npz_readability:" + item["path"], False, item.get("error"))
    check("all_npz_members_decoded", bool(report["npz_archives"]) and all(v["readable"] for v in report["npz_archives"]), {"archives": len(report["npz_archives"])})

    try:
        mismatches = subset_mismatches(benchmark["pmf"], pmf, "pmf")
        check("benchmark_pmf_matches_raw_json", not mismatches, mismatches)
        check("fixed_surface_and_reference", pmf["xi_star"] == 1.00925 and pmf["reference"] == 0.0)
        check("pmf_method_and_measure", pmf["method"] == "mbar" and pmf["sampling_measure"] == "rate_flux" and pmf["ready"] is True)
        ui = pmf["ui_comparison"]
        mb = np.asarray(ui["paired_mbar_bootstrap_barriers"], dtype=float)
        uv = np.asarray(ui["paired_ui_bootstrap_barriers"], dtype=float)
        delta = np.asarray(ui["paired_difference_bootstrap_hartree"], dtype=float)
        check("paired_draw_shapes_and_finiteness", mb.shape == uv.shape == delta.shape == (40,) and np.isfinite(mb).all() and np.isfinite(uv).all() and np.isfinite(delta).all())
        check("paired_draw_difference_exact", np.array_equal(delta, mb - uv))
        check("main_and_paired_mbar_draws_exact", np.array_equal(mb, np.asarray(pmf["pmf_bootstrap_barriers"])))
        boot = pmf["bootstrap"]
        draws = boot["draws"]
        check("bootstrap_40_valid_no_failures", boot["requested"] == boot["attempted"] == boot["valid"] == 40 and not boot["failures"] and not ui["failures"] and [d["draw"] for d in draws] == list(range(40)) and all(d["valid"] is True for d in draws))
        check("bootstrap_ledger_matches_paired_arrays", np.array_equal(mb, np.asarray([d["barrier_hartree"] for d in draws])) and np.array_equal(uv, np.asarray([d["ui_barrier_hartree"] for d in draws])))
        endpoints = np.array([[d["reference_free_energy_hartree"], d["surface_free_energy_hartree"]] for d in draws])
        check("bootstrap_endpoints_difference", np.allclose(endpoints[:, 1] - endpoints[:, 0], mb, rtol=0, atol=1e-15))
        stats = [("mbar", mb, pmf["barrier_standard_error_hartree"], pmf["barrier_percentile_95_ci_hartree"]), ("ui", uv, ui["ui_barrier_statistics"]["standard_error_hartree"], ui["ui_barrier_statistics"]["percentile_95_ci_hartree"]), ("paired_difference", delta, ui["paired_difference_statistics"]["standard_error_hartree"], ui["paired_difference_statistics"]["percentile_95_ci_hartree"])]
        for label, values, se, ci in stats:
            check(label + "_standard_error", se is not None and np.isclose(values.std(ddof=1), se, rtol=1e-12, atol=1e-15))
            check(label + "_percentile_interval", ci is not None and np.allclose(np.quantile(values, [.025, .975]), ci, rtol=1e-12, atol=1e-15))
        report["pmf_recomputed"] = {
            "xi_star": pmf["xi_star"], "reference": pmf["reference"],
            "mbar_barrier_kcal_mol": pmf["free_energy_barrier_hartree"] * KCAL_PER_HARTREE,
            "ui_barrier_kcal_mol": ui["free_energy_barrier_hartree"] * KCAL_PER_HARTREE,
            "paired_difference_kcal_mol": (pmf["free_energy_barrier_hartree"] - ui["free_energy_barrier_hartree"]) * KCAL_PER_HARTREE,
            "paired_difference_se_kcal_mol": delta.std(ddof=1) * KCAL_PER_HARTREE,
            "paired_difference_95_ci_kcal_mol": np.quantile(delta, [.025, .975]) * KCAL_PER_HARTREE,
            "mbar_ui_bootstrap_correlation": np.corrcoef(mb, uv)[0, 1],
        }
        check("reported_paired_point_difference", np.isclose(ui["mbar_minus_ui_hartree"], pmf["free_energy_barrier_hartree"] - ui["free_energy_barrier_hartree"], rtol=0, atol=1e-15))
        bin_checks = []
        for row in benchmark["bin_sensitivity"]:
            ratio = math.exp(-config["beta"] * (row["free_energy_barrier_hartree"] - pmf["free_energy_barrier_hartree"]))
            ok = row["ready"] is True and row["support"]["continuous_interval"] is True and np.isclose(ratio, row["same_surface_rate_ratio_to_primary"], rtol=1e-12, atol=0)
            bin_checks.append({"bins": row["bins"], "barrier_kcal_mol": row["free_energy_barrier_hartree"] * KCAL_PER_HARTREE, "recomputed_rate_ratio": ratio, "passed": bool(ok)})
        report["bin_sensitivity_recomputed"] = bin_checks
        check("bin_sensitivity_algebra_and_support", sorted(v["bins"] for v in bin_checks) == [85, 340] and all(v["passed"] for v in bin_checks))
    except Exception as exc:
        check("pmf_algebra", False, f"{type(exc).__name__}: {exc}")

    try:
        rec = benchmark["recrossing"]
        with np.load(study / "run/recrossing.npz", allow_pickle=False) as data:
            time, curve, num, den, boots = (data[k] for k in ["time_au", "kappa", "parent_numerator", "parent_denominator", "bootstrap_kappa"])
            check("recrossing_array_shapes_and_finite", time.shape == curve.shape == (800,) and num.shape == (64, 800) and den.shape == (64,) and boots.shape == (100,) and all(np.isfinite(v).all() for v in (time, curve, num, den, boots)))
            check("recrossing_clock", np.allclose(time, np.arange(1, 801) * config["dt_au"], rtol=1e-12, atol=1e-12))
            check("recrossing_ratio_from_parent_arrays", np.all(den > 0) and np.allclose(curve, num.sum(axis=0) / den.sum(), rtol=0, atol=1e-14))
            tail_start = int(np.searchsorted(time, rec["plateau_start_au"]))
            tail = curve[tail_start:]
            check("kappa_tail_estimate", tail.size > 0 and np.isclose(np.mean(tail), rec["kappa"], rtol=0, atol=1e-14))
            check("kappa_bootstrap_se", np.isclose(np.std(boots, ddof=1), rec["kappa_standard_error"], rtol=0, atol=1e-14))
            report["recrossing_recomputed"] = {"kappa": float(np.mean(tail)), "kappa_standard_error": float(np.std(boots, ddof=1)), "tail_start_index": tail_start, "tail_points": len(tail), "tail_minimum": float(np.min(tail)), "tail_maximum": float(np.max(tail))}
        rate = pmf["k_qtst_cm3_s"] * rec["kappa"]
        check("rate_qtst_times_kappa", np.isclose(rate, benchmark["rate_estimate_cm3_molecule_s"], rtol=1e-13, atol=0))
        check("qtst_boltzmann_factor", np.isclose(pmf["reactant_prefactor_au"] * math.exp(-config["beta"] * pmf["free_energy_barrier_hartree"]), pmf["k_qtst_au"], rtol=1e-12, atol=0))
    except Exception as exc:
        check("recrossing_algebra", False, f"{type(exc).__name__}: {exc}")

    check("independent_solver_same_data", solver.get("passed") is True and all(solver.get("gates", {}).values()) and solver.get("streams") == expected_streams and solver.get("samples") == expected_streams * 200 and solver.get("stride") == 10)
    report["recorded_independent_solver"] = solver.get("independent_pymbar")
    if source_root is None:
        candidates = list(study.parents) + [Path.cwd()]
        source_root = next((p for p in candidates if (p / "src/pyrpmd").is_dir()), None)
    elif source_root is not None:
        source_root = Path(source_root).resolve()
    report["source_verification"]["source_root"] = str(source_root) if source_root else None
    for name, expected_hash in identity["package_and_runner_sources"].items():
        path = None
        if name == "benchmark/convergence.py":
            path = study / "provenance/convergence-source.py"
        elif name == "benchmark/convergence_inputs/base.inp":
            path = study / "provenance/input-template.inp"
        elif source_root is not None and not name.startswith("native_binary/"):
            path = source_root / name if name.startswith("native/") else source_root / "src/pyrpmd" / name
        record = {"name": name, "expected_sha256": expected_hash}
        if path is None or not path.is_file():
            record["reason"] = "original_binary_not_archived" if name.startswith("native_binary/") else "source_file_unavailable"
            report["source_verification"]["unavailable"].append(record)
            continue
        record.update(path=str(path), actual_sha256=sha256(path))
        category = "verified" if record["actual_sha256"] == expected_hash else "mismatches"
        report["source_verification"][category].append(record)
    for original, expected_hash in identity["referenced_files"].items():
        record = {"name": original, "expected_sha256": expected_hash}
        if original.endswith((".so", ".o")):
            record["reason"] = "original_remote_binary_not_archived; local rebuild is not the same artifact"
            report["source_verification"]["unavailable"].append(record)
            continue
        marker = "/benchmarks/"
        path = source_root / ("benchmarks/" + original.split(marker, 1)[1]) if source_root is not None and marker in original else None
        if path is None or not path.is_file():
            record["reason"] = "referenced_source_unavailable"
            report["source_verification"]["unavailable"].append(record)
            continue
        record.update(path=str(path), actual_sha256=sha256(path))
        category = "verified" if record["actual_sha256"] == expected_hash else "mismatches"
        report["source_verification"][category].append(record)
    check("available_source_hashes", not report["source_verification"]["mismatches"], {"verified": len(report["source_verification"]["verified"]), "unavailable": len(report["source_verification"]["unavailable"]), "mismatches": report["source_verification"]["mismatches"]})
    report["source_verification"]["complete"] = not report["source_verification"]["unavailable"]
    report["passed"] = not report["errors"]
    return clean(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="case directory containing benchmark.json and run/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, help="optional matching repository checkout; automatically discovered otherwise")
    args = parser.parse_args()
    try:
        result = audit(args.study, args.source_root)
    except Exception as exc:
        result = {"schema": "pyrpmd.h3.independent-artifact-audit.v1", "passed": False, "study_directory": str(args.study.resolve()), "errors": [{"name": "audit_exception", "passed": False, "detail": f"{type(exc).__name__}: {exc}"}]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(clean(result), indent=2, allow_nan=False) + "\n")
    # stdout is also strict JSON, so both output forms can be archived automatically.
    print(json.dumps({"passed": result["passed"], "case": result.get("case"), "output": str(args.output.resolve()), "checks": len(result.get("checks", [])), "failures": len(result.get("errors", [])), "streams": len(result.get("streams", [])), "npz_archives": len(result.get("npz_archives", []))}, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
