"""Real, source-pinned Ni-H acceptance for independent materials trajectories.

The short matrix checks implementation equivalence, not converged diffusion.
Every mode runs in a fresh interpreter; its real ASE provider calls are counted
without replacing the potential. See batch_acceptance.md for the longer study.
"""

from __future__ import annotations

import argparse
import atexit
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.materials.ni_h_eam import PROVENANCE, job, make_slab, obtain_model, verify_adapter

TRACE_ENV = "PYRPMD_MATERIALS_ACCEPTANCE_TRACE"
STOP_ENV = "PYRPMD_MATERIALS_ACCEPTANCE_STOP_STEP"
STOP_MESSAGE = "materials acceptance interruption after an actual integration step"
ARRAY_FIELDS = ("positions", "velocities", "observables", "radii_gyration_squared", "sample_steps", "times")
MODES = {
    "serial": {"backend": "serial", "trajectory_workers": 1, "trajectory_batch_size": 1},
    "process": {"backend": "process", "trajectory_workers": 2, "trajectory_batch_size": 1},
    "batch": {"backend": "serial", "trajectory_workers": 1, "trajectory_batch_size": 2},
    "process_batch": {"backend": "process", "trajectory_workers": 2, "trajectory_batch_size": 2},
    "mpi": {"backend": "mpi", "trajectory_groups": 2, "trajectory_batch_size": 1},
    "mpi_batch": {"backend": "mpi", "trajectory_groups": 2, "trajectory_batch_size": 2},
    "mpi_dynamic": {"backend": "mpi", "trajectory_groups": 2, "trajectory_batch_size": 1,
                    "scheduling": "dynamic", "coordinator_ranks": 1},
    "mpi_dynamic_batch": {"backend": "mpi", "trajectory_groups": 2, "trajectory_batch_size": 2,
                          "scheduling": "dynamic", "coordinator_ranks": 1},
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def install_worker_probe():
    """Install only in this benchmark's isolated workers, including spawn.

    A bounded shape histogram avoids timing every call or retaining coordinates.
    Multiprocessing finalizers also flush workers that bypass ordinary atexit.
    """
    directory = os.environ.get(TRACE_ENV)
    if directory is None:
        return lambda: None
    from multiprocessing.util import Finalize
    from pyrpmd.dynamics import Dynamics
    from pyrpmd.potentials.optional import ASEPotential

    shapes = Counter()
    original = ASEPotential.evaluate

    def measured(self, q):
        result = original(self, q)
        shapes[tuple(np.shape(q))] += 1
        return result

    ASEPotential.evaluate = measured

    def flush():
        write_json(Path(directory) / f"calls-{os.getpid()}.json", {
            "pid": os.getpid(),
            "shapes": [{"shape": list(shape), "calls": count} for shape, count in sorted(shapes.items())],
        })

    atexit.register(flush)
    Finalize(None, flush, exitpriority=10)
    if STOP_ENV in os.environ:
        target = int(os.environ[STOP_ENV])
        for method in ("step", "sample_step"):
            real_step = getattr(Dynamics, method)

            def interrupted(self, state, *args, _step=real_step, **kwargs):
                result = _step(self, state, *args, **kwargs)
                if state.step == target:
                    raise RuntimeError(STOP_MESSAGE)
                return result

            setattr(Dynamics, method, interrupted)
    return flush


# Spawn imports __mp_main__ before executing submitted tasks. No hooks are
# installed in normal imports, pytest, or the outer benchmark coordinator.
_flush_probe = install_worker_probe()


def configuration(model, task="pimd", *, replicas=5, beads=4, seed=7261,
                  equilibration_steps=4, production_steps=16, dt_fs=0.1,
                  lateral_repeat=1, mode="serial", sample_interval=2):
    config = job(model, beads=beads, quick=True, seed=seed)
    atoms = make_slab().repeat((lateral_repeat, lateral_repeat, 1))
    symbols = atoms.get_chemical_symbols()
    config["task"] = task
    config["system"].update(symbols=symbols, geometry=atoms.positions.tolist(),
                            cell=atoms.cell[:].tolist(), pbc=atoms.pbc.tolist())
    # This exact EAM implementation has independent, history-free frames. Do not
    # generalize its explicit opt-in to arbitrary ASE calculators or ML models.
    config["potential"]["trajectory_batch_safe"] = True
    config["dynamics"].update(timestep=f"{dt_fs:.17g} fs", block_steps=3, loop_backend="python")
    config["sampling"].update(replicas=replicas, equilibration_steps=equilibration_steps,
                              production_steps=production_steps, sample_interval=sample_interval)
    if task == "diffusion":
        config["diffusion"].update(atom_indices=[i for i, s in enumerate(symbols) if s == "H"],
                                   reference_atoms=[i for i, s in enumerate(symbols) if s == "Ni"])
    else:
        config.pop("diffusion", None)
    config["parallel"] = {"mode": "serial", "bead_workers": 1, "threads_per_rank": 1, **MODES[mode]}
    config["output"].update(checkpoint_steps=3, trajectory_storage="stream", chunk_frames=3)
    return config


def source_identity():
    import importlib.util

    paths = [Path(__file__), ROOT / "benchmarks/materials/ni_h_eam.py"]
    paths += list((ROOT / "src/pyrpmd").rglob("*.py"))
    paths += [p for p in (ROOT / "native").rglob("*") if p.suffix in (".cpp", ".h", ".hpp")]
    campaign = ROOT / "benchmarks/materials/batch_convergence.py"
    if campaign.is_file():
        paths.append(campaign)
    result = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    native = importlib.util.find_spec("pyrpmd._native")
    result["loaded_native_extension"] = "unavailable" if native is None else {
        "path": native.origin, "sha256": hashlib.sha256(Path(native.origin).read_bytes()).hexdigest()}
    return result


def physical_oracle(model):
    """Direct ASE, Cartesian/strain derivatives, PBC and reordered frame checks."""
    from ase.calculators.eam import EAM
    from pyrpmd.constants import ANGSTROM_TO_AU, BOHR_ANGSTROM, EV_TO_AU
    from pyrpmd.input import JobSpec
    from pyrpmd.potentials import make_potential

    basic = verify_adapter(model)
    config, system = JobSpec.from_mapping(configuration(model)).resolve()
    provider = make_potential(config["potential"], system)
    atoms = make_slab()
    atoms.positions[-1] += [.13, -.07, .04]
    atoms.calc = EAM(potential=str(model))
    coordinates = atoms.positions.copy()
    frames = np.repeat(coordinates[None], 3, axis=0)
    frames[1, -1] += [.03, -.01, .02]
    frames[2, 0] += [-.02, .02, .01]
    q = frames * ANGSTROM_TO_AU
    before = q.copy()
    try:
        result = provider.evaluate(q)
        reverse = provider.evaluate(q[::-1].copy())
        energy_error = force_error = virial_error = order_error = 0.0
        for index, frame in enumerate(frames):
            atoms.positions = frame
            energy_error = max(energy_error, abs(result.energy[index] - atoms.get_potential_energy() * EV_TO_AU))
            force_error = max(force_error, float(np.max(abs(result.forces[index] - atoms.get_forces() * EV_TO_AU * BOHR_ANGSTROM))))
            virial = -atoms.get_stress(voigt=False) * atoms.get_volume() * EV_TO_AU
            virial_error = max(virial_error, float(np.max(abs(result.virial[index] - virial))))
            for field in ("energy", "forces", "virial"):
                order_error = max(order_error, float(np.max(abs(getattr(result, field)[index] - getattr(reverse, field)[2-index]))))
        strain_errors = []
        for row in range(3):
            for column in range(3):
                energies = []
                epsilon = 2e-5
                for sign in (1, -1):
                    transform = np.eye(3)
                    transform[row, column] += sign * epsilon
                    strained = make_slab()
                    strained.cell = atoms.cell[:] @ transform.T
                    strained.positions = coordinates @ transform.T
                    strained.calc = EAM(potential=str(model))
                    energies.append(strained.get_potential_energy())
                derivative = -(energies[0] - energies[1]) / (2 * epsilon)
                strain_errors.append(abs(derivative - result.virial[0, row, column] / EV_TO_AU))
        unchanged = bool(np.array_equal(q, before))
    finally:
        provider.close()
    checks = {
        "energy_hartree_max_error": float(energy_error),
        "force_hartree_per_bohr_max_error": force_error,
        "virial_hartree_max_error": virial_error,
        "reordered_frames_max_error_atomic_units": order_error,
        "all_nine_strain_components_max_error_eV": float(max(strain_errors)),
        "unwrapped_input_unchanged": unchanged,
        "direct_ase_and_cartesian_fd": basic,
    }
    checks["passed"] = bool(basic["passed"] and unchanged and energy_error < 1e-10
                            and force_error < 1e-10 and virial_error < 1e-10
                            and order_error < 1e-10 and max(strain_errors) < 1e-4)
    return checks


def _worker(request_path):
    from pyrpmd import resume_run, run

    request = json.loads(Path(request_path).read_text())
    start = time.perf_counter()
    success = False
    try:
        if request.get("resume"):
            resume_run(request["output"], parallel=request["parallel"], execution={"block_steps": 7})
        else:
            run(request["configuration"], output=request["output"])
        success = True
    finally:
        _flush_probe()
        own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        scale = 1 if sys.platform == "darwin" else 1024
        write_json(Path(os.environ[TRACE_ENV]) / f"resources-{os.getpid()}.json", {
            "success": success, "wall_seconds": time.perf_counter() - start,
            "self_cpu_seconds": own.ru_utime + own.ru_stime,
            "children_cpu_seconds": children.ru_utime + children.ru_stime,
            "self_peak_rss_bytes": own.ru_maxrss * scale,
            "largest_child_peak_rss_bytes": children.ru_maxrss * scale,
        })


def execute(config, output, evidence, *, mpi_command=None, interrupt_step=None, resume=False):
    """One isolated mode; retain input, stdout/stderr and bounded telemetry."""
    output, evidence = Path(output).resolve(), Path(evidence).resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    request = {"configuration": config, "output": str(output), "resume": resume,
               "parallel": config["parallel"]}
    request_path = evidence / "request.json"
    write_json(request_path, request)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(request_path)]
    if config["parallel"]["backend"] == "mpi":
        if not mpi_command:
            raise ValueError("MPI modes require an explicit launcher, for example --mpi-command 'mpiexec -n 2'")
        command = shlex.split(mpi_command) + command
    environment = dict(os.environ)
    environment.update({TRACE_ENV: str(evidence), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"})
    environment.pop(STOP_ENV, None)
    if interrupt_step is not None:
        environment[STOP_ENV] = str(interrupt_step)
    before_steps = sum(s["elapsed"] for s in read_streams(output, require_all=False)[2]) if resume else 0
    start = time.perf_counter()
    with (evidence / "stdout.txt").open("w") as stdout, (evidence / "stderr.txt").open("w") as stderr:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
    wall = time.perf_counter() - start
    if interrupt_step is None and completed.returncode:
        raise RuntimeError(f"Acceptance worker exited {completed.returncode}; inspect {evidence / 'stderr.txt'}")
    if interrupt_step is not None and (completed.returncode == 0 or STOP_MESSAGE not in (evidence / "stderr.txt").read_text()):
        raise AssertionError("The intended integration interruption was not observed")
    histogram = Counter()
    for path in evidence.glob("calls-*.json"):
        for row in json.loads(path.read_text())["shapes"]:
            histogram[tuple(row["shape"])] += row["calls"]
    resources = [json.loads(p.read_text()) for p in evidence.glob("resources-*.json")]
    if not resources:
        raise AssertionError("Worker resource receipts are missing")
    calls = sum(histogram.values())
    frames = sum(shape[0] * count for shape, count in histogram.items())
    after_steps = sum(s["elapsed"] for s in read_streams(output, require_all=False)[2])
    advanced = after_steps - before_steps
    return {
        "returncode": completed.returncode, "wall_seconds_including_startup_and_probe": wall,
        "cpu_seconds_including_spawn_children": sum(x["self_cpu_seconds"] + x["children_cpu_seconds"] for x in resources),
        "rank_resource_receipts": resources,
        "rss_scope": "per-rank self peak and largest terminated child peak; not simultaneous process-tree RSS",
        "pes_calls": calls, "pes_frames": frames,
        "pes_frames_per_wall_second": frames / wall,
        "shapes": [{"shape": list(shape), "calls": count} for shape, count in sorted(histogram.items())],
        "committed_replica_steps_advanced": advanced,
        "committed_replica_steps_per_wall_second": advanced / wall,
        "evidence": str(evidence),
    }


def read_streams(directory, *, require_all=True):
    from pyrpmd.system import System
    from pyrpmd.workflows.materials import _read_stream

    directory = Path(directory)
    config = json.loads((directory / "input/resolved.json").read_text())
    system = System.from_dict(json.loads((directory / "input/system.json").read_text()))
    count = config["sampling"]["replicas"]
    present = [replica for replica in range(count)
               if (directory / f"checkpoints/materials_{replica:04d}.npz").is_file()]
    if require_all and present != list(range(count)):
        raise AssertionError("Missing one or more registered replica checkpoints")
    streams = [_read_stream(config, system, directory, replica) for replica in present]
    return config, system, streams


def compare_runs(reference, candidate):
    """Strict same-machine identity, including RNG and all saved physical data."""
    _, _, left = read_streams(reference)
    _, _, right = read_streams(candidate)
    if len(left) != len(right) or not left:
        raise AssertionError("Replica counts differ or no saved replicas exist")
    for a, b in zip(left, right):
        if a["replica"] != b["replica"] or not a["complete"] or not b["complete"]:
            raise AssertionError("Replica identity/completion differs")
        for field in ARRAY_FIELDS:
            np.testing.assert_array_equal(a[field], b[field], err_msg=f"replica {a['replica']}: {field}")
        for field in ("q", "p", "step", "time"):
            np.testing.assert_array_equal(getattr(a["state"], field), getattr(b["state"], field))
        if a["rng"] != b["rng"] or a["propagation"] != b["propagation"]:
            raise AssertionError("RNG or accepted propagation counters changed")
    summaries = [json.loads((Path(p) / "results/summary.json").read_text()) for p in (reference, candidate)]
    fields = ("estimators", "sampling_diagnostics", "radius_of_gyration_squared_bohr2", "diffusion",
              "analysis_status", "nve_energy_diagnostics", "samples_per_replica")
    for field in fields:
        if summaries[0].get(field) != summaries[1].get(field):
            raise AssertionError(f"Summary statistic differs: {field}")
    return {"passed": True, "replicas": len(left), "q_p_rng_streams_statistics": "exact_same_machine",
            "frames_per_replica": [len(s["times"]) for s in left]}


def _chunk_hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(directory).glob("samples/materials_*/*.npz") if p.is_file()}


def recovery_check(config, reference, directory, evidence):
    interrupted = copy.deepcopy(config)
    interrupted["parallel"].update(MODES["batch"])
    failure_step = config["sampling"]["equilibration_steps"] + 5
    failure = execute(interrupted, directory, evidence / "interrupted", interrupt_step=failure_step)
    _, _, partial = read_streams(directory, require_all=False)
    if not partial or not any(0 < s["elapsed"] < failure_step for s in partial):
        raise AssertionError("No genuine partially integrated checkpoint was retained")
    prefixes = _chunk_hashes(directory)
    resumed = copy.deepcopy(config)
    resumed["parallel"].update(MODES["process_batch"], trajectory_batch_size=3, trajectory_workers=2)
    continuation = execute(resumed, directory, evidence / "continued", resume=True)
    for path, digest in prefixes.items():
        if hashlib.sha256((Path(directory) / path).read_bytes()).hexdigest() != digest:
            raise AssertionError("A committed pre-interruption chunk changed")
    equivalent = compare_runs(reference, directory)
    checkpoint_bytes = {p.name: p.read_bytes() for p in (Path(directory) / "checkpoints").glob("materials_*.npz")}
    terminal = execute(resumed, directory, evidence / "terminal", resume=True)
    if terminal["pes_calls"] != 0:
        raise AssertionError("Completed restart evaluated the PES")
    if any((Path(directory) / "checkpoints" / name).read_bytes() != value for name, value in checkpoint_bytes.items()):
        raise AssertionError("Completed restart rewrote a checkpoint")
    return {"passed": True, "injected_after_actual_step": failure_step,
            "partial_elapsed_steps": [s["elapsed"] for s in partial],
            "committed_chunks_preserved": len(prefixes), "packing": "2 lanes -> 3 lanes, block 3 -> 7",
            "equivalence": equivalent, "interruption": failure, "continuation": continuation,
            "terminal_restart": terminal}


def run_acceptance(model, output, *, modes=("serial", "process", "batch", "process_batch"),
                   mpi_command=None, mpi_dynamic_command=None, tasks=("pimd", "diffusion"), recovery=True):
    model, output = obtain_model(model), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new empty acceptance directory")
    if "serial" not in modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError("Modes must be distinct supported names and include serial")
    if not tasks or any(task not in ("pimd", "diffusion") for task in tasks):
        raise ValueError("Tasks must be pimd and/or diffusion")
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema": "pyrpmd.materials-batch-acceptance.v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "potential": PROVENANCE, "source_at_start": source_identity(),
              "versions": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "ase")},
               "implementation_checks_passed": False,
              "scope": "fixed-cell real Ni-H force/virial contracts and finite-step serial/process/task-batch/MPI equivalence",
              "performance_scope": "single measurement per mode, startup and count probe included; no speedup assertion",
              "cases": [], "recovery": {}}
    write_json(output / "report.json", report)
    try:
        report["oracle"] = physical_oracle(model)
        if not report["oracle"]["passed"]:
            raise AssertionError("The real force/unit/PBC/virial oracle failed")
        for task in tasks:
            ordered_modes = ("serial",) + tuple(m for m in modes if m != "serial")
            baseline = None
            for mode in ordered_modes:
                config = configuration(model, task, mode=mode)
                directory = output / f"{task}-{mode}"
                launcher = mpi_dynamic_command if mode.startswith("mpi_dynamic") else mpi_command
                measured = execute(config, directory, output / "evidence" / f"{task}-{mode}", mpi_command=launcher)
                if baseline is None:
                    baseline = measured
                else:
                    if measured["pes_frames"] != baseline["pes_frames"]:
                        raise AssertionError("Packing changed the number of evaluated physical frames")
                if config["parallel"]["trajectory_batch_size"] > 1:
                    if not any(row["shape"][0] > config["beads"] for row in measured["shapes"]):
                        raise AssertionError("Requested batching never reached the real ASE provider")
                    if measured["pes_calls"] >= baseline["pes_calls"]:
                        raise AssertionError("Task batching did not reduce actual physical provider calls")
                equivalence = compare_runs(output / f"{task}-serial", directory)
                report["cases"].append({"task": task, "mode": mode, "configuration": config,
                                        "measurement": measured, "equivalence": equivalence})
                write_json(output / "report.json", report)
            if recovery:
                report["recovery"][task] = recovery_check(configuration(model, task), output / f"{task}-serial",
                                                         output / f"{task}-recovered", output / "evidence" / f"{task}-recovery")
        report["source_at_end"] = source_identity()
        if report["source_at_end"] != report["source_at_start"]:
            raise AssertionError("Reviewed source changed during acceptance")
        report["implementation_checks_passed"] = True
    except BaseException as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--modes", default="serial,process,batch,process_batch")
    parser.add_argument("--mpi-command", help="Explicit two-rank launch command, e.g. 'mpiexec -n 2'")
    parser.add_argument("--mpi-dynamic-command", help="Explicit three-rank C1/G2/B1 command, e.g. 'mpiexec -n 3'")
    parser.add_argument("--quick", action="store_true", help="Explicitly select the finite-step acceptance matrix (default)")
    args = parser.parse_args()
    if args.worker is not None:
        _worker(args.worker)
        return
    if args.model is None or args.output is None:
        parser.error("--model and --output are required")
    model = obtain_model(args.model, download=args.download)
    result = run_acceptance(model, args.output, modes=tuple(args.modes.split(",")),
                            mpi_command=args.mpi_command, mpi_dynamic_command=args.mpi_dynamic_command)
    print(json.dumps({"implementation_checks_passed": result["implementation_checks_passed"],
                       "report": str(args.output / "report.json")}))


if __name__ == "__main__":
    main()
