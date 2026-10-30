"""Compare two checkouts on the same real BKMP2 dRPMD workload.

Example (build the pinned PES with build_pes.py first)::

    python benchmarks/h3/drpmd_optimization.py \
        --baseline /path/to/pristine-checkout --candidate . \
        --library /path/to/libbkmp2.so --output runs/drpmd-optimization --build

Both checkouts run in fresh processes, on the same CPU and PES binary, with the
same classification cadence, persistence, output and checkpoint settings. The
core comparison explicitly enables the candidate's opt-in within-block force
reuse when supported. Workflow comparisons use their normal production paths.
This measures implementation efficiency, not physical rate convergence.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from unittest.mock import patch


THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
SOURCE_FILES = (
    "setup.py", "native/cayley.cpp", "native/fused.hpp", "native/propagation.h",
    "src/pyrpmd/dynamics.py", "src/pyrpmd/propagation.py",
    "src/pyrpmd/block_propagation.py", "src/pyrpmd/workflows/drpmd.py",
    "benchmarks/h3/drpmd.py", "benchmarks/h3/pes.py",
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _git(root, *arguments):
    result = subprocess.run(["git", *arguments], cwd=root, capture_output=True,
                            text=True, check=True)
    return result.stdout.strip()


def _manifest(root, library):
    import numpy as np
    import pyrpmd
    from pyrpmd import _native

    if not Path(pyrpmd.__file__).resolve().is_relative_to(root):
        raise AssertionError("Imported Python package is outside the selected checkout")
    if not Path(_native.__file__).resolve().is_relative_to(root):
        raise AssertionError("Imported native extension is outside the selected checkout")
    provenance = json.loads(library.with_suffix(".build.json").read_text())
    library_hash = _sha256(library)
    if provenance["library_sha256"] != library_hash:
        raise AssertionError("PES binary differs from its build provenance")
    return {
        "checkout": str(root), "commit": _git(root, "rev-parse", "HEAD"),
        "git_status": _git(root, "status", "--short"),
        "sources_sha256": {p: _sha256(root / p) for p in SOURCE_FILES},
        "native_extension": str(_native.__file__),
        "native_extension_sha256": _sha256(_native.__file__),
        "native_outcomes_api": getattr(_native, "DRPMD_OUTCOMES_API", 0),
        "python": sys.version, "numpy": np.__version__,
        "platform": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": {k: os.environ.get(k) for k in THREAD_ENV},
        "pes_library": str(library), "pes_library_sha256": library_hash,
        "pes_source_sha256": provenance["source"]["sha256"],
        "pes_compiler": provenance["compiler"], "pes_flags": provenance["flags"],
    }


def _digest(arrays, signature):
    import numpy as np
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode())
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _core(args, beads, block_steps):
    import numpy as np
    from benchmarks.h3.pes import BKMP2Potential
    from benchmarks.loop_native.benchmark import initial, MASS_H, BETA, DT_AU
    from pyrpmd.block_propagation import propagate_block
    from pyrpmd.dynamics import Dynamics, State

    q, p = initial(beads)
    dynamics = Dynamics(BKMP2Potential(args.library), MASS_H, BETA, beads,
                        DT_AU, native="required", threads=1, loop_backend="native")
    state = State(q, p)
    reuse = args.variant == "candidate" and "reuse_forces" in inspect.signature(propagate_block).parameters
    options = {"reuse_forces": True} if reuse else {}
    # Warm the actual loaded PES/extension in every timing process, then start
    # from the identical untouched state. No cache survives this warmup block.
    warmup_steps = min(block_steps, 8)
    dynamics.propagate_block(state.copy(), warmup_steps, record_energy=True,
                             record_states=False, backend="native", **options)
    native_before = dynamics.propagation_stats["native_steps"]
    rows = []
    started = time.perf_counter()
    remaining = args.core_steps
    while remaining:
        count = min(block_steps, remaining)
        block = dynamics.propagate_block(state, count, record_energy=True,
                                         record_states=False, backend="native", **options)
        if not block.success:
            raise AssertionError(block.error)
        rows.append(block)
        remaining -= count
    elapsed = time.perf_counter() - started
    if dynamics.propagation_stats["native_steps"] - native_before != args.core_steps:
        raise AssertionError("Core case did not execute every step natively")
    arrays = {"final_q": state.q, "final_p": state.p}
    for key in ("steps", "times", "energies", "centroids", "centroid_momenta"):
        arrays[key] = np.concatenate([getattr(row, key) for row in rows])
    signature = {"steps": state.step, "time": state.time}
    return {
        "seconds": elapsed, "signature": signature,
        "propagation": {"native_steps": args.core_steps, "python_steps": 0},
        "settings": {"beads": beads, "steps": args.core_steps,
                     "block_steps": block_steps, "dt_au": DT_AU,
                     "beta": BETA, "masses": MASS_H.tolist(),
                     "per_step_energy_and_centroid": True,
                     "record_states": False, "warmup_steps": warmup_steps,
                     "reuse_forces_requested": reuse},
        "timed_scope": "native propagate_block plus gathering identical observations; no initialization or file I/O",
    }, arrays


def _configuration(args, check_interval):
    from benchmarks.h3.drpmd import job_mapping
    from pyrpmd.input import JobSpec

    mapping = job_mapping(trajectories=args.trajectories, workers=1, parents=2,
                          beads=16, equilibration_fs=5., sample_interval_fs=2.,
                          max_time_fs=args.max_time_fs)
    mapping["potential"]["params"]["library"] = str(args.library)
    mapping["dynamics"].update(loop_backend="native", block_steps=256)
    mapping["outcomes"]["check_interval"] = check_interval
    mapping["output"].pop("checkpoint_interval", None)
    mapping["output"].update(checkpoint_steps=args.checkpoint_steps,
                             trajectory_stride=args.trajectory_stride)
    return JobSpec.from_mapping(mapping).resolve()


def _workflow_arrays(directory, count):
    import numpy as np
    arrays = {}
    for i in range(count):
        with np.load(directory / "trajectories" / f"trajectory-{i:08d}.npz") as source:
            for key in ("final_q", "final_p", "steps", "time_au", "centroid", "energy_per_bead"):
                arrays[f"trajectory_{i}_{key}"] = source[key].copy()
        with np.load(directory / "states" / f"initial-{i:08d}.npz") as source:
            for key in ("q", "p", "thermal_q", "thermal_p"):
                arrays[f"launch_{i}_{key}"] = source[key].copy()
    return arrays


def _workflow_signature(result):
    return {
        "counts": result["counts"], "parent_thermal_steps": result["parent_thermal_steps"],
        "records": [{key: record[key] for key in
                     ("trajectory_id", "parent_id", "status", "channel", "reason", "steps", "time_au")}
                    for record in result["records"]],
    }


def _restart_checks(config, system, directory, result, arrays):
    import numpy as np
    from pyrpmd.workflows import drpmd

    completed = copy.deepcopy(config)
    completed["resume"] = True
    completed["dynamics"]["block_steps"] = 17
    restored = drpmd.run_drpmd(completed, system, directory)
    if restored["records"] != result["records"]:
        raise AssertionError("Completed restart changed trajectory records")

    # A durable mid-trajectory checkpoint, with nonaligned output/check intervals.
    # The uninterrupted comparison uses exactly the same configuration.
    control = copy.deepcopy(config)
    control["output"].update(checkpoint_steps=37, trajectory_stride=7)
    control_directory = directory.parent / "restart-control"
    control_result = drpmd.run_drpmd(control, system, control_directory)
    control_arrays = _workflow_arrays(control_directory, len(control_result["records"]))
    interrupted_directory = directory.parent / "restart-interrupted"
    original_save = drpmd._atomic_npz
    interrupted = False

    def save_then_interrupt(path, metadata, **data):
        nonlocal interrupted
        original_save(path, metadata, **data)
        if (not interrupted and metadata.get("format") == "PyRPMD.drpmd.trajectory-checkpoint.v1"
                and 0 < metadata.get("done", 0) < control["collision"]["max_steps"]):
            interrupted = True
            raise KeyboardInterrupt("benchmark interruption after durable checkpoint")

    try:
        with patch.object(drpmd, "_atomic_npz", side_effect=save_then_interrupt):
            drpmd.run_drpmd(control, system, interrupted_directory)
    except KeyboardInterrupt:
        if not interrupted:
            raise
    if not interrupted:
        raise AssertionError("No intermediate trajectory checkpoint was reached")
    resume = copy.deepcopy(control)
    resume["resume"] = True
    resume["dynamics"]["block_steps"] = 7
    resumed = drpmd.run_drpmd(resume, system, interrupted_directory)
    if _workflow_signature(resumed) != _workflow_signature(control_result):
        raise AssertionError("Interrupted restart changed outcomes or stop steps")
    resumed_arrays = _workflow_arrays(interrupted_directory, len(resumed["records"]))
    for key in control_arrays:
        np.testing.assert_array_equal(control_arrays[key], resumed_arrays[key], err_msg=key)
    # Confirm reading the already-completed output did not change any samples.
    for key, value in _workflow_arrays(directory, len(result["records"])).items():
        np.testing.assert_array_equal(arrays[key], value, err_msg=key)
    return {"completed_records_exact": True, "interrupted_arrays_exact": True,
            "interrupted_status_and_stop_exact": True, "resumed_block_steps": 7,
            "checkpoint_steps": 37, "trajectory_stride": 7}


def _workflow(args, check_interval, verify):
    from pyrpmd.workflows import drpmd

    config, system = _configuration(args, check_interval)
    directory = args.data_dir / "workflow"
    started = time.perf_counter()
    result = drpmd.run_drpmd(config, system, directory)
    elapsed = time.perf_counter() - started
    native = sum(r["propagation"]["native_steps"] for r in result["records"])
    steps = sum(r["steps"] for r in result["records"])
    if native != steps or not steps or result["counts"]["failed"]:
        raise AssertionError("Workflow failed or did not use native propagation for all accepted steps")
    arrays = _workflow_arrays(directory, len(result["records"]))
    report = {
        "seconds": elapsed, "signature": _workflow_signature(result),
        "propagation": {"native_steps": native, "python_steps": 0},
        "diagnostics": [{key: r[key] for key in
                        ("initial_energy_per_bead", "final_energy_per_bead", "max_energy_drift_per_bead")}
                       for r in result["records"]],
        "settings": {"beads": config["beads"], "dt_au": config["dt_au"],
                     "beta": config["beta"], "seed": config.get("seed"),
                     "trajectories": args.trajectories, "parents": config["sampling"]["parents"],
                     "max_steps": config["collision"]["max_steps"],
                     "check_interval": config["outcomes"]["check_interval"],
                     "persistence_steps": config["outcomes"]["persistence_steps"],
                     "checkpoint_steps": config["output"]["checkpoint_steps"],
                     "trajectory_stride": config["output"]["trajectory_stride"],
                     "block_steps": config["dynamics"]["block_steps"]},
        "timed_scope": "full run_drpmd including potential initialization, parent sampling, classification, output, checkpoints and summary",
    }
    if verify:
        report["restart_checks"] = _restart_checks(config, system, directory, result, arrays)
    return report, arrays


def _worker(args):
    for name in THREAD_ENV:
        os.environ[name] = "1"
    if args.cpu is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {args.cpu})
    root = args.checkout.resolve()
    sys.path[:0] = [str(root / "src"), str(root)]
    import numpy as np

    args.data_dir.mkdir(parents=True, exist_ok=False)
    manifest = _manifest(root, args.library)
    if args.case.startswith("core-"):
        _, beads, block = args.case.split("-")
        report, arrays = _core(args, int(beads), int(block))
    else:
        kind, cadence = args.case.split("-")
        report, arrays = _workflow(args, int(cadence), kind == "verify")
    report.update(case=args.case, variant=args.variant, manifest=manifest,
                  data_digest=_digest(arrays, report["signature"]))
    np.savez(args.data_dir / "arrays.npz", **arrays)
    _json(args.data_dir / "result.json", report)


def _compare(first, second):
    import numpy as np
    left = json.loads((first / "result.json").read_text())
    right = json.loads((second / "result.json").read_text())
    if left["signature"] != right["signature"]:
        raise AssertionError(f"Outcome/stop/time mismatch: {first} versus {second}")
    settings_left = dict(left["settings"])
    settings_right = dict(right["settings"])
    settings_left.pop("reuse_forces_requested", None)
    settings_right.pop("reuse_forces_requested", None)
    if settings_left != settings_right:
        raise AssertionError("Physical or observation settings differ between checkouts")
    maximum = {"q": 0., "p": 0., "energy": 0., "time": 0.}
    exact = True
    with np.load(first / "arrays.npz") as a, np.load(second / "arrays.npz") as b:
        if set(a.files) != set(b.files):
            raise AssertionError("Saved observation keys differ")
        for key in a.files:
            aa, bb = a[key], b[key]
            if aa.shape != bb.shape:
                raise AssertionError(f"Observation shape mismatch for {key}")
            exact &= np.array_equal(aa, bb)
            if "steps" in key:
                np.testing.assert_array_equal(aa, bb, err_msg=key)
                continue
            category = "energy" if ("energ" in key) else "time" if ("time" in key) else "p" if ("momenta" in key or key.endswith("_p")) else "q"
            tolerance = {"q": 2e-8, "p": 2e-7, "energy": 2e-10, "time": 1e-12}[category]
            np.testing.assert_allclose(aa, bb, atol=tolerance, rtol=1e-10, err_msg=key)
            maximum[category] = max(maximum[category], float(np.max(np.abs(aa - bb))) if aa.size else 0.)
    if "diagnostics" in left:
        for a, b in zip(left["diagnostics"], right["diagnostics"], strict=True):
            for key in a:
                np.testing.assert_allclose(a[key], b[key], atol=2e-10, rtol=1e-10, err_msg=key)
    return {"signature_exact": True, "observation_settings_identical": True,
            "all_arrays_bitwise_equal": bool(exact), "maximum_absolute_errors": maximum}


def _controller(args):
    if args.repeats < 3:
        raise ValueError("At least three repeats are required")
    if args.trajectories < 2 or args.core_steps < 1 or args.max_time_fs <= 0:
        raise ValueError("Positive work and at least two trajectories are required")
    for name in THREAD_ENV:
        os.environ[name] = "1"
    if args.cpu is None and hasattr(os, "sched_getaffinity"):
        args.cpu = min(os.sched_getaffinity(0))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    roots = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    environment = os.environ.copy()
    script = Path(__file__).resolve()
    builds = {}
    for name, root in roots.items():
        if args.build:
            command = [sys.executable, "setup.py", "build_ext", "--inplace", "--force"]
            with (output / f"{name}-build.log").open("w") as log:
                subprocess.run(command, cwd=root, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            builds[name] = {"command": command, "log": f"{name}-build.log"}
    report = {"schema": "pyrpmd.drpmd-optimization.v1", "passed": False,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "command": sys.argv,
              "physical_convergence": False, "harness_sha256": _sha256(script),
              "repeats": args.repeats, "warmups_per_case_per_checkout": 1,
              "timing_order_seed": args.seed, "builds": builds, "cases": [],
              "scope": "same-machine same-PES implementation comparison; no chemical rate convergence claim"}
    rng = random.Random(args.seed)
    source_identities = {}

    def execute(case, name, label):
        directory = output / case / name / label
        command = [sys.executable, str(script), "--worker", "--checkout", str(roots[name]),
                   "--variant", name, "--case", case, "--data-dir", str(directory),
                   "--library", str(args.library), "--core-steps", str(args.core_steps),
                   "--trajectories", str(args.trajectories), "--max-time-fs", str(args.max_time_fs),
                   "--checkpoint-steps", str(args.checkpoint_steps),
                   "--trajectory-stride", str(args.trajectory_stride)]
        if args.cpu is not None:
            command.extend(["--cpu", str(args.cpu)])
        child_environment = environment.copy()
        child_environment["PYTHONPATH"] = os.pathsep.join([str(roots[name] / "src"), str(roots[name])])
        result = subprocess.run(command, cwd=roots[name], env=child_environment,
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"{case}/{name}/{label} failed:\n{result.stdout}\n{result.stderr}")
        value = json.loads((directory / "result.json").read_text())
        identity = {key: value["manifest"][key] for key in
                    ("commit", "sources_sha256", "native_extension_sha256", "pes_library_sha256")}
        if name in source_identities and source_identities[name] != identity:
            raise AssertionError(f"Sources, extension or PES changed during timing: {name}")
        source_identities[name] = identity
        return value, directory

    try:
        cases = [f"core-{beads}-{block}" for beads in (16, 32) for block in (5, 256)]
        cases += ["workflow-1", "workflow-5"]
        for case in cases:
            references = {}
            seconds = {name: [] for name in roots}
            order = []
            warmup_order = list(roots)
            rng.shuffle(warmup_order)
            for name in warmup_order:
                references[name] = execute(case, name, "warmup")
            verification = _compare(references["baseline"][1], references["candidate"][1])
            for repeat in range(args.repeats):
                names = list(roots)
                rng.shuffle(names)
                order.append(names)
                for name in names:
                    value, _ = execute(case, name, f"repeat-{repeat}")
                    if value["data_digest"] != references[name][0]["data_digest"]:
                        raise AssertionError(f"Nondeterministic repeated output for {case}/{name}")
                    seconds[name].append(value["seconds"])
            medians = {name: statistics.median(times) for name, times in seconds.items()}
            row = {"case": case, "seconds": seconds, "median_seconds": medians,
                   "speedup": medians["baseline"] / medians["candidate"],
                   "verification": verification, "repeat_order": order,
                   "warmup_order": warmup_order,
                   "settings": {name: references[name][0]["settings"] for name in roots},
                   "data_digests": {name: references[name][0]["data_digest"] for name in roots},
                   "signature": references["candidate"][0]["signature"],
                   "timed_scope": references["candidate"][0]["timed_scope"]}
            report["cases"].append(row)
            report["manifests"] = {name: references[name][0]["manifest"] for name in roots}
            print(json.dumps({"case": case, "speedup": row["speedup"],
                              "median_seconds": medians, "verified": True}), flush=True)
            _json(output / "report.json", report)
        report["restart_checks"] = {}
        for name in roots:
            value, directory = execute("verify-5", name, "restart")
            report["restart_checks"][name] = value["restart_checks"]
        report["passed"] = True
    finally:
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _json(output / "report.json", report)
    print(json.dumps({"passed": True, "report": str(output / "report.json")}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build", action="store_true", help="force equal setup.py native builds before timing")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=847291)
    parser.add_argument("--core-steps", type=int, default=2000)
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--max-time-fs", type=float, default=200.)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--trajectory-stride", type=int, default=10)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkout", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=("baseline", "candidate"), help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.library = args.library.resolve()
    if args.worker:
        if not all((args.checkout, args.variant, args.case, args.data_dir)):
            parser.error("Worker arguments are incomplete")
        _worker(args)
    else:
        if args.baseline is None or args.output is None:
            parser.error("--baseline and --output are required")
        _controller(args)


if __name__ == "__main__":
    main()
