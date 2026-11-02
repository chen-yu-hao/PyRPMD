"""Read-only all-eight comparison of existing original/control coarse histories.

No interpolation, alignment transformation, permissive tolerance, PES, Context,
RNG draw or new propagation. Stored float64 checkpoints are compared directly;
float64 control observations are also cast to the original stored dtype.
"""
from __future__ import annotations
import time
ENTRY = time.monotonic()
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[5]
WORKSPACE = ROOT.parent
OLD = WORKSPACE / "water-raw-archive-replay-2026-11-01/extracted/pyrpmd-water-classical-eight-chain-2026-11-01/raw"
NEW = WORKSPACE / "water-paired-timestep-archive-replay-2026-11-01-attempt2/extracted/control"
SEEDS = tuple(range(1101, 1109))
FIELDS = ("step", "segment_step", "molecular_com_bohr", "molecular_velocity_au", "oxygen_centroid_bohr", "hamiltonian_per_bead_hartree")
REGISTRATION_SHA = "0d73545ac4751b84bdf64a99d693dee23a7f683bb3cd38e2cff5b7d8055e8365"
CONTROL_REPORT_SHA = "6d657d7dc11d98dd78ac8a2659a9907c8b94eedba0aa013dcd2d3d5db54aa370"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    os.sched_setaffinity(0, {4})
    os.nice(10)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS"):
        os.environ[key] = "1"
    sys.dont_write_bytecode = True
    def expired(_signum, _frame):
        raise TimeoutError("saved-history comparison exceeded60 wall seconds")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 60-(time.monotonic()-ENTRY))
    require(not (HERE / "report.json").exists(), "never replace correspondence evidence")
    import numpy as np
    require(sha(NEW / "registration.json") == REGISTRATION_SHA and sha(NEW / "analysis/report.json") == CONTROL_REPORT_SHA,
            "fixed control inputs changed")
    reg, control = read(NEW / "registration.json"), read(NEW / "analysis/report.json")
    require(reg["fixed_seeds"] == list(SEEDS) and [r["seed"] for r in control["chains"]] == list(SEEDS)
            and control["all_eight_pairs_complete"], "whole fixed cohort required")
    guards = {ROOT/name: digest for name, digest in reg["source_sha256"].items()}
    guards[Path(__file__).resolve()] = sha(__file__)
    inputs, curve_rows, results = {}, [], []
    def checked(path, digest=None):
        path = Path(path)
        actual = sha(path)
        require(digest is None or actual == digest, "immutable input checksum differs")
        inputs[str(path)] = actual
        return path
    checked(NEW / "registration.json", REGISTRATION_SHA)
    checked(NEW / "analysis/report.json", CONTROL_REPORT_SHA)
    for seed in SEEDS:
        selected = reg["selection"][str(seed)]
        original = OLD / "runs" / f"classical-{seed}"
        fresh = NEW / "runs" / f"seed-{seed}"
        om = read(checked(original / "manifest.json", selected["manifest_sha256"]))
        cm = read(checked(fresh / "dt025/manifest.json", control["chains"][seed-1101]["branches"]["dt025"]["manifest_sha256"]))
        im = read(checked(fresh / "initial/initial.json"))
        require(om["complete"] and cm["complete"] and om["config"]["seed"] == seed
                and om["config"]["dt_fs"] == cm["native_config"]["dt_fs"] == .25
                and om["config"]["record_stride"] == cm["native_config"]["record_stride"] == 20
                and om["config"]["equilibration_record_stride"] == 400
                and om["artifacts"]["model.xml"] == cm["artifacts"]["model.xml"]
                and om["artifacts"]["integrator.xml"] == cm["artifacts"]["integrator.xml"],
                "same frozen coarse configuration/model required")
        with np.load(checked(original / selected["checkpoint"]["file"], selected["checkpoint"]["sha256"]), allow_pickle=False) as data:
            before = {k: data[k].copy() for k in ("q", "rng_json", "state_step", "active_segment", "engine_forces_kj_mol_nm")}
        require(int(before["state_step"]) == 400000 and int(before["active_segment"]) == -1, "wrong source phase")
        with np.load(checked(fresh / "initial/initial.npz", im["initial_sha256"]), allow_pickle=False) as data:
            initial = {k: data[k].copy() for k in ("q", "p", "rng_before_json", "rng_after_json")}
        all_data, endpoints, saved_rng = {}, {}, []
        for kind, directory, manifest, start, stop in (("original", original, om, 100, 104), ("control", fresh / "dt025", cm, 0, 5)):
            parts = {key: [] for key in FIELDS}
            endpoints[kind] = {}
            for number in range(start, stop):
                row = manifest["chunks"][number]
                require(row["file"] == f"chunk-{number:06d}.npz", "checkpoint identity/order differs")
                with np.load(checked(directory / row["file"], row["sha256"]), allow_pickle=False) as data:
                    mask = data["segment"] == 0
                    require(mask.all(), "first4ps comparison must contain only fixed segment0")
                    for key in FIELDS:
                        parts[key].append(data[key][mask].copy())
                    relative = int(data["state_step"])-(400000 if kind == "original" else 0)
                    endpoints[kind][relative] = {key: data[key].copy() for key in ("q", "p", "engine_forces_kj_mol_nm")}
                    if kind == "original":
                        saved_rng.append(json.loads(str(data["rng_json"])))
            all_data[kind] = {key: np.concatenate(value) for key, value in parts.items()}
        a, b = all_data["original"], all_data["control"]
        expected = np.arange(0, 16001, 20)
        require(np.array_equal(a["step"]-400000, expected) and np.array_equal(b["step"], expected)
                and np.array_equal(a["segment_step"], expected) and np.array_equal(b["segment_step"], expected),
                "exact unmodified common5fs grid required")
        fields = {}
        differences = {}
        for key in FIELDS[2:]:
            require(a[key].shape == b[key].shape and np.isfinite(a[key]).all() and np.isfinite(b[key]).all(), "finite whole observation arrays required")
            cast = b[key].astype(a[key].dtype)
            axes = tuple(range(1, a[key].ndim))
            equal = np.all(a[key] == cast, axis=axes) if axes else a[key] == cast
            delta = abs(a[key].astype(float)-b[key])
            per_frame = np.max(delta, axis=axes) if axes else delta
            differences[key] = per_frame
            bad = np.flatnonzero(~equal)
            fields[key] = {"original_dtype": str(a[key].dtype), "control_dtype": str(b[key].dtype),
                           "exact_after_cast_to_original_dtype": bool(equal.all()),
                           "exact_frames_after_cast": int(equal.sum()), "frames": len(equal),
                           "first_mismatching_time_fs": None if not len(bad) else float(expected[bad[0]]*.25),
                           "initial_frame_equal_after_cast": bool(equal[0]),
                           "maximum_abs_difference_native_units": float(delta.max()),
                           "final_frame_max_abs_difference_native_units": float(per_frame[-1])}
        checkpoints = []
        for step in (4000, 8000, 12000, 16000):
            x, y = endpoints["original"][step], endpoints["control"][step]
            checkpoints.append({"time_fs": step*.25,
                                "q_exact": bool(np.array_equal(x["q"], y["q"])),
                                "p_exact": bool(np.array_equal(x["p"], y["p"])),
                                "q_max_abs_difference_bohr": float(np.max(abs(x["q"]-y["q"]))),
                                "p_max_abs_difference_au": float(np.max(abs(x["p"]-y["p"])))})
        force_delta = endpoints["control"][0]["engine_forces_kj_mol_nm"]-before["engine_forces_kj_mol_nm"]
        initial_q_exact = np.array_equal(before["q"], initial["q"])
        rng_before_exact = json.loads(str(before["rng_json"])) == json.loads(str(initial["rng_before_json"]))
        rng_after_exact = all(state == json.loads(str(initial["rng_after_json"])) for state in saved_rng)
        require(initial_q_exact and rng_before_exact and rng_after_exact, "starting state/draw provenance differs")
        row = {"seed": seed, "frames": 801, "original_absolute_step_origin": 400000,
               "record_interval_fs_original": 5, "record_interval_fs_control": 5,
               "original_preparation_record_interval_fs": 100,
               "same_saved_start_q_bitwise": bool(initial_q_exact), "same_before_draw_RNG_state": rng_before_exact,
               "same_after_draw_RNG_state_all_original_first_segment_checkpoints": rng_after_exact,
               "new_RNG_draws_performed": 0, "full_original_post_draw_p0_separately_saved": False,
               "initial_rebuilt_force_max_abs_difference_kj_mol_nm": float(np.max(abs(force_delta))),
               "initial_rebuilt_force_RMS_difference_kj_mol_nm": float(np.sqrt(np.mean(force_delta**2))),
               "fields": fields, "full_precision_checkpoints": checkpoints,
               "all_saved_coarse_observations_exact_after_dtype_cast": all(x["exact_after_cast_to_original_dtype"] for x in fields.values()),
               "all_four_full_precision_q_p_checkpoints_exact": all(x["q_exact"] and x["p_exact"] for x in checkpoints)}
        results.append(row)
        for i, step in enumerate(expected):
            curve_rows.append([seed, int(step), float(step*.25), *[float(differences[key][i]) for key in FIELDS[2:]]])
    for path, digest in inputs.items():
        require(sha(path) == digest, "saved input changed during read-only comparison")
    for path, digest in guards.items():
        require(sha(path) == digest, "frozen source changed during comparison")
    with (HERE / "all-time-differences.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "relative_step", "time_fs", *["maximum_abs_delta_"+key for key in FIELDS[2:]]])
        writer.writerows(curve_rows)
    eligible = all(row["all_saved_coarse_observations_exact_after_dtype_cast"] and row["all_four_full_precision_q_p_checkpoints_exact"] for row in results)
    report = {"schema": "pyrpmd.water-all-eight-saved-coarse-history-correspondence.v1",
              "created_utc": datetime.now(timezone.utc).isoformat(), "all8_complete_first_segments_checked": True,
              "fixed_seeds": list(SEEDS), "all_frames_per_seed": 801, "interpolation_or_alignment_transform": False,
              "numeric_equality_rule": "Exact array equality after only float64-to-original-dtype casting, plus exact float64 q/p at each1ps checkpoint. No numerical tolerance is used for correspondence.",
              "coarse_reuse_eligible_under_requested_exact_correspondence_condition": eligible,
              "new_all25_segment_reuse_study_proposed_or_adopted": False,
              "chains": results, "inputs": inputs, "source_hashes": {str(p): v for p,v in guards.items()},
              "input_files_unchanged": True, "source_files_unchanged": True,
              "new_PES_calls": 0, "new_RNG_draws": 0, "new_dynamics_steps": 0, "transport_estimates_recomputed": False,
              "original64ps_control_failures_reclassified": False, "CPU_affinity": [4], "threads": 1,
              "CPU_seconds_before_write": time.process_time(), "wall_seconds_before_write": time.monotonic()-ENTRY,
              "maximum_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "interpretation": "The comparison tests saved-history identity, not statistical timestep equivalence. Shared source q/RNG and initial rounded observables cannot imply bitwise future CPU trajectories. No approximate or favorable-subset reuse is authorized."}
    with (HERE / "report.json").open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"all8_checked": True, "coarse_reuse_eligible": eligible,
                      "source_unchanged": True, "CPU_seconds": report["CPU_seconds_before_write"]}))


if __name__ == "__main__":
    main()
