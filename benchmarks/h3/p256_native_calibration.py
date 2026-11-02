"""One excluded, registered P=256 map/restart check; never a rate experiment.

The executable installs process limits before importing scientific libraries.
The output directory and RUN_ONCE marker are permanent, including on failure.
"""
from __future__ import annotations

import time
ENTRY_WALL = time.monotonic()

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import weakref

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/validation/literature-convergence-2026-11-01/h3-quantum/p256-native-calibration"
PROPOSAL = EVIDENCE / "proposal.json"
PROPOSAL_SHA = "d33f221b2ce8e869b9f7c3577021726bba0ac25b23f0deac3c08e30c5d47c196"
OUTPUT = ROOT.parent / "literature-convergence-quantum/p256-native-calibration-v1"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def cpu_seconds():
    value = resource.getrusage(resource.RUSAGE_SELF)
    return value.ru_utime + value.ru_stime


def admission():
    if cpu_seconds() >= 55 or time.monotonic() - ENTRY_WALL >= 115:
        raise TimeoutError("registered admission budget exhausted")


def install_limits():
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    os.sched_setaffinity(0, {6})
    for name in THREADS:
        os.environ[name] = "1"

    def expired(_signum, _frame):
        raise TimeoutError("registered wall-time budget exhausted")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, 120 - (time.monotonic() - ENTRY_WALL)))
    require(resource.getrlimit(resource.RLIMIT_CPU) == (60, 60), "CPU hard limit differs")
    require(os.sched_getaffinity(0) == {6}, "CPU affinity differs")


def authorization(review_path):
    require(sha(PROPOSAL) == PROPOSAL_SHA, "registered proposal changed")
    proposal = read(PROPOSAL)
    adoption = read(EVIDENCE / "root-adoption.json")
    review = read(review_path)
    require(adoption["adopted"] is True and adoption["proposal_sha256"] == PROPOSAL_SHA,
            "root adoption does not match")
    limits = adoption["resources"]
    require(limits["CPU_affinity"] == [6] and limits["workers"] == 1 and limits["threads"] == 1
            and limits["whole_process_hard_CPU_seconds"] == 60
            and limits["admission_CPU_seconds"] == 55 and limits["whole_process_wall_seconds"] == 120,
            "adopted resources differ")
    require(review["approved_for_once_execution"] is True
            and review["proposal_sha256"] == PROPOSAL_SHA
            and review["source_sha256"] == sha(__file__), "independent source review does not match")
    return proposal


def identity_guards(proposal, review_path):
    rows = proposal["source_bindings"] + proposal["prior_gate_sources"]
    rows += list(proposal["parent"]["files"].values())
    rows += [proposal["parent"][name] for name in
             ("global_complete_manifest", "registration", "root_exit_receipt", "complete_audit")]
    rows += [proposal["physical_parameters"][name] for name in ("library", "build")]
    guards = {str(Path(row["path"]).resolve()): row["sha256"] for row in rows}
    for path in (PROPOSAL, EVIDENCE / "root-adoption.json", Path(review_path), Path(__file__),
                 ROOT / "benchmarks/h3/source.json"):
        guards[str(path.resolve())] = sha(path)
    check_guards(guards)
    return guards


def check_guards(guards):
    for path, expected in guards.items():
        require(sha(path) == expected, f"registered identity changed: {path}")


def claim(output, registration):
    Path(output).mkdir(exist_ok=False)
    write_new(Path(output) / "RUN_ONCE.json", registration)


def checkpoint_write(path, arrays, fingerprint, frame):
    from pyrpmd.checkpoint import atomic_savez
    require(not Path(path).exists(), "checkpoint output already exists")
    atomic_savez(path, kind="h3.p256-excluded-native-calibration", fingerprint=fingerprint,
                 completed_frame=frame, **arrays)


def checkpoint_read(path, fingerprint, expected_frame):
    from pyrpmd.checkpoint import load_checkpoint
    with load_checkpoint(path, expected_kind="h3.p256-excluded-native-calibration",
                         require_integrity=True) as saved:
        require(str(saved["fingerprint"].item()) == fingerprint, "checkpoint fingerprint changed")
        require(int(saved["completed_frame"].item()) == expected_frame, "checkpoint frame changed")
        return {name: saved[name].copy() for name in saved if name not in
                ("fingerprint", "completed_frame")}


def import_science(proposal):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import numpy as np
    from benchmarks.h3 import quantum_surface_recrossing as original
    from pyrpmd.dynamics import Dynamics, State
    from pyrpmd.checkpoint import load_checkpoint
    from pyrpmd.constants import FS_TO_AU
    imported = {}
    for row in proposal["source_bindings"]:
        path = Path(row["path"]).resolve()
        relative = path.relative_to(ROOT)
        if relative.parts[0] == "src":
            name = ".".join(relative.with_suffix("").parts[1:]) if path.suffix == ".py" else "pyrpmd._native"
        else:
            name = ".".join(relative.with_suffix("").parts)
        actual = Path(importlib.import_module(name).__file__).resolve()
        require(actual == path and sha(actual) == row["sha256"], f"import route differs: {name}")
        imported[name] = {"path": str(actual), "sha256": sha(actual)}
    return np, original, Dynamics, State, load_checkpoint, FS_TO_AU, imported


def run_path(science, proposal, q0, p0, dt_fs, name, fingerprint, output):
    np, original, Dynamics, State, _load, fs_to_au, _imports = science
    block_steps = round(1.0 / dt_fs)
    native = name != "python"
    arrays = None
    dispatch = {"native_steps": 0, "manual_python_steps": 0, "python_fallback_steps": 0}
    recreated = False
    segments = ((0, 2), (2, 4)) if name == "restart" else ((0, 4),)
    midpoint = output / f"{dt_fs:g}-{name}-midpoint.npz"
    state = None
    for start, stop in segments:
        admission()
        potential = original.BKMP2Potential(proposal["physical_parameters"]["library"]["path"])
        dyn = Dynamics(potential, original.MASS, original.BETA, 256, dt_fs * fs_to_au,
                       native="required", loop_backend="native" if native else "python")
        surface = original.surface_coordinate()
        segment_complete = False
        try:
            if start:
                arrays = checkpoint_read(midpoint, fingerprint, 2)
                state = State(arrays["q"][-1], arrays["p"][-1],
                              step=int(arrays["step"][-1]), time=float(arrays["time"][-1]))
                recreated = True
            else:
                state = State(q0, p0)
                arrays = {"q": state.q[None].copy(), "p": state.p[None].copy(),
                          "surface": np.asarray([surface.value(state.q)[0]]),
                          "energy": np.asarray([dyn.energy(state)]),
                          "step": np.asarray([state.step], dtype=np.int64),
                          "time": np.asarray([state.time])}
            for frame in range(start + 1, stop + 1):
                admission()
                if native:
                    block = dyn.propagate(state, block_steps, observe=surface,
                                          sample_interval=block_steps, record_centroids=False)
                    require(block.backend == "native", "unexpected Python fallback")
                    value, energy = block.values[-1, 0], block.energies[-1]
                else:
                    for _ in range(block_steps):
                        dyn.step(state)
                        dispatch["manual_python_steps"] += 1
                    value, energy = surface.value(state.q)[0], dyn.energy(state)
                row = {"q": state.q, "p": state.p, "surface": value, "energy": energy,
                       "step": state.step, "time": state.time}
                for key in arrays:
                    require(np.isfinite(row[key]).all(), f"nonfinite {key}")
                    arrays[key] = np.concatenate((arrays[key], np.asarray(row[key])[None]))
            dispatch["native_steps"] += dyn.propagation_stats["native_steps"]
            dispatch["python_fallback_steps"] += dyn.propagation_stats["python_steps"]
            if stop == 2:
                checkpoint_write(midpoint, arrays, fingerprint, 2)
            segment_complete = True
        except BaseException:
            if state is not None:
                checkpoint_write(output / f"{dt_fs:g}-{name}-failed-state.npz",
                                 {"q": state.q, "p": state.p, "step": state.step, "time": state.time},
                                 fingerprint, int(state.step // block_steps))
            raise
        finally:
            potential.close()
            old_dyn, old_pes = weakref.ref(dyn), weakref.ref(potential)
            del dyn, potential
            gc.collect()
            if segment_complete:
                require(old_dyn() is None and old_pes() is None, "old Dynamics/PES Python context retained")
    expected = 4 * block_steps
    require(dispatch["native_steps"] == (expected if native else 0)
            and dispatch["manual_python_steps"] == (0 if native else expected)
            and dispatch["python_fallback_steps"] == 0, "dispatch step accounting differs")
    require(int(arrays["step"][-1]) == expected, "final step differs")
    checkpoint_write(output / f"{dt_fs:g}-{name}-trace.npz", arrays, fingerprint, 4)
    return arrays, {"dispatch": dispatch, "recreated_context": recreated,
                    "energy_range_hartree": float(np.ptp(arrays["energy"])),
                    "trace_sha256": sha(output / f"{dt_fs:g}-{name}-trace.npz")}


def comparisons(np, native, python, restarted, proposal):
    for label, trace in (("native", native), ("python", python), ("restart", restarted)):
        for key, values in trace.items():
            require(np.isfinite(values).all(), f"nonfinite {label} {key} trace (including initial frame)")
    tolerance = proposal["gates"]["map_comparison"]
    atol = {"q": tolerance["absolute_tolerance_q_bohr"], "p": tolerance["absolute_tolerance_p_au"],
            "surface": tolerance["absolute_tolerance_surface_bohr"],
            "energy": tolerance["absolute_tolerance_energy_hartree"], "time": 2e-10}
    checks = {}
    for key in native:
        exact = (native[key].dtype == restarted[key].dtype and native[key].shape == restarted[key].shape
                 and native[key].tobytes() == restarted[key].tobytes())
        require(exact, f"native restart {key} is not bitwise identical")
        if key == "step":
            require(np.array_equal(native[key], python[key]), "Python/native step count differs")
        else:
            np.testing.assert_allclose(native[key], python[key], atol=atol[key],
                                       rtol=tolerance["relative_tolerance"], equal_nan=False)
        checks[key] = {"restart_bitwise_equal": exact,
                       "maximum_python_native_absolute_difference": float(np.max(np.abs(native[key] - python[key])))}
    return checks


def execute(proposal, guards, review_path):
    science = import_science(proposal)
    np, original, _dyn, _state, load_checkpoint, _fs, imports = science
    check_guards(guards)
    parent = proposal["parent"]
    with load_checkpoint(parent["files"]["checkpoint.npz"]["path"],
                         expected_kind="h3.surface-partition", require_integrity=True) as saved:
        require(str(saved["fingerprint"].item()) == parent["settings_fingerprint"], "parent fingerprint differs")
        require(int(saved["endpoint_completed_sweeps"][0]) == 2100, "parent sweep differs")
        q0 = saved["endpoint_q"][0].copy()
    require(q0.shape == (256, 3, 3) and q0.dtype.str == parent["dtype"]
            and hashlib.sha256(q0.tobytes()).hexdigest() == parent["contiguous_array_sha256"], "selected parent differs")
    rng = np.random.default_rng(np.random.SeedSequence([20261101, 256, 902731]))
    p0 = rng.standard_normal((256, 3, 3)) * np.sqrt(original.MASS[None, :, None] * 256 / original.BETA)
    qhash, phash = hashlib.sha256(q0.tobytes()).hexdigest(), hashlib.sha256(p0.tobytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps({"proposal": PROPOSAL_SHA, "source": sha(__file__),
                                            "q": qhash, "p": phash}, sort_keys=True).encode()).hexdigest()
    checkpoint_write(OUTPUT / "initial-state.npz", {"q": q0, "p": p0, "step": 0, "time": 0.0}, fingerprint, 0)
    write_new(OUTPUT / "initial-ledger.json", {"proposal_sha256": PROPOSAL_SHA,
        "fingerprint": fingerprint, "parent_q_sha256": qhash, "calibration_p_sha256": phash,
        "initial_state_file_sha256": sha(OUTPUT / "initial-state.npz"),
        "rng_entropy": [20261101, 256, 902731], "source_guards": guards,
        "imported_modules": imports, "PES_evaluations_before_ledger": 0,
        "CPU_seconds_before_PES": cpu_seconds()})
    rows = []
    for dt in (0.1, 0.05):
        values, details = {}, {}
        for name in ("native", "python", "restart"):
            values[name], details[name] = run_path(science, proposal, q0, p0, dt, name,
                                                 fingerprint, OUTPUT)
        checks = comparisons(np, values["native"], values["python"], values["restart"], proposal)
        row = {"timestep_fs": dt, "horizon_fs": 4, "checks": checks, "paths": details}
        rows.append(row)
        write_new(OUTPUT / f"checks-{dt:g}.json", row)
    require(hashlib.sha256(q0.tobytes()).hexdigest() == qhash
            and hashlib.sha256(p0.tobytes()).hexdigest() == phash, "initial state was mutated")
    steps = sum(sum(path["dispatch"].values()) for row in rows for path in row["paths"].values())
    require(steps == 360, "total physical map steps differs")
    check_guards(guards)
    return {"all_checks_passed": True, "actual_physical_map_steps": steps,
            "native_steps": 240, "manual_python_steps": 120, "python_fallback_steps": 0,
            "timesteps": rows, "source_guards_before": guards, "source_guards_after": guards,
            "all_source_and_input_guards_unchanged": True, "imported_modules": imports,
            "initial_state_sha256": sha(OUTPUT / "initial-state.npz"),
            "fingerprint": fingerprint, "rate_or_transmission_computed": False}


def main(argv=None):
    install_limits()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args(argv)
    proposal = authorization(args.review)
    guards = identity_guards(proposal, args.review)
    registration = {"proposal_sha256": PROPOSAL_SHA, "source_sha256": sha(__file__),
                    "review_sha256": sha(args.review), "root_adoption_sha256": sha(EVIDENCE / "root-adoption.json"),
                    "PID": os.getpid(), "CPU_affinity": sorted(os.sched_getaffinity(0)),
                    "CPU_limit": list(resource.getrlimit(resource.RLIMIT_CPU)),
                    "threads": {name: os.environ[name] for name in THREADS},
                    "created_utc": datetime.now(timezone.utc).isoformat()}
    claim(OUTPUT, registration)
    result = {"all_checks_passed": False, "actual_physical_map_steps": None}
    error = None
    try:
        admission()
        result = execute(proposal, guards, args.review)
    except BaseException as caught:
        error = caught
        result["error"] = {"type": type(caught).__name__, "message": str(caught)}
        result["source_guards_before"] = guards
        result["source_guards_after"] = {path: sha(path) if Path(path).is_file() else None for path in guards}
        result["all_source_and_input_guards_unchanged"] = result["source_guards_after"] == guards
    result.update({"schema": "pyrpmd.h3.p256-excluded-native-calibration.v1", "registration": registration,
        "source_sha256": sha(__file__), "proposal_sha256": PROPOSAL_SHA,
        "created_utc": datetime.now(timezone.utc).isoformat(), "CPU_seconds": cpu_seconds(),
        "wall_seconds": time.monotonic() - ENTRY_WALL,
        "peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "outputs": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
                    for p in sorted(OUTPUT.iterdir()) if p.is_file()]})
    write_new(OUTPUT / "report.json", result)
    print(json.dumps({"all_checks_passed": result["all_checks_passed"],
                      "actual_physical_map_steps": result["actual_physical_map_steps"],
                      "report_sha256": sha(OUTPUT / "report.json")}), flush=True)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
