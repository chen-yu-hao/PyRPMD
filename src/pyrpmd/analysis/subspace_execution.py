"""CPU/memory bounded QM and RPMD tasks, with optional MPI and task restart.

All MPI ranks call ``subspace_corrections``. Rank zero schedules, checkpoints
and also computes. Completed tasks survive a launcher/job restart; an interrupted
task is rerun with its original seed. There are no collectives in trajectory loops.
"""

from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import socket
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .subspace_quantum import QuantumRateResult, _json_safe, exact_quantum_rate, quantum_solver_plan
from .subspace_rpmd import RingPolymerRateResult, _thermal_frequency, ring_polymer_rate


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _affinity_cpus():
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def _options(execution):
    options = dict(execution or {})
    unknown = set(options) - {"backend", "cpus", "workers", "blas_threads", "memory_limit_mb", "checkpoint_dir", "resume"}
    if unknown:
        raise ValueError(f"unknown subspace execution options: {sorted(unknown)}")
    options.setdefault("backend", "process")
    if options["backend"] not in ("serial", "process", "mpi"):
        raise ValueError("execution.backend must be serial, process or mpi")
    raw_cpus = options.get("cpus", 1)
    if isinstance(raw_cpus, str) and raw_cpus == "auto":
        options["cpus"] = _affinity_cpus()
    else:
        options["cpus"] = _positive_int(raw_cpus, "execution.cpus")
    options["blas_threads"] = _positive_int(options.get("blas_threads", 8), "execution.blas_threads")
    if "workers" in options:
        options["workers"] = _positive_int(options["workers"], "execution.workers")
    options["memory_limit_mb"] = float(options.get("memory_limit_mb", 4096.))
    if not np.isfinite(options["memory_limit_mb"]) or options["memory_limit_mb"] <= 0:
        raise ValueError("execution.memory_limit_mb must be finite and positive")
    if not isinstance(options.get("resume", True), bool):
        raise ValueError("execution.resume must be boolean")
    return options


def _memory_mb(kind, model, beta, beads, options):
    size = int(np.prod(model.shape))
    if kind == "quantum":
        # The solver's own plan: the scheduler budgets the route the solve will take.
        return quantum_solver_plan(model, **options)["estimated_memory_mb"]
    windows = options.get("windows", 48)
    nw = int(windows) if np.ndim(windows) == 0 else len(windows)
    walkers, trajectories = options.get("walkers", 32), options.get("trajectories", 512)
    dt = options.get("md_dt") or 0.1 / _thermal_frequency(model, beta)
    duration = options.get("md_time") or 0.3 * (model.axes[0][-1] - model.axes[0][0]) * np.sqrt(beta * model.masses[0])
    steps = max(10, int(np.ceil(duration / dt)))
    # State/proposal copies plus side, numerator and bootstrap indexing arrays.
    return 64 + (192 * max(nw * walkers, 2 * trajectories) * beads * model.dimension
                 + 48 * trajectories * (steps + 1) + 64 * size) / 2**20


def _tasks(cases):
    from .subspace_correction import _validate_rpmd_model

    tasks = []
    for key, case in cases.items():
        model, beta, beads = case["model"], case["beta"], case["nbeads"]
        rpmd_model = case.get("rpmd_model")
        _validate_rpmd_model(model, rpmd_model)
        for kind, potential in (("quantum", model), ("rpmd", rpmd_model if rpmd_model is not None else model)):
            options = dict(case.get(kind + "_options") or {})
            # Pipeline absorbers can be closures; evaluate only at the original
            # DVR grid where the quantum solver would evaluate them anyway.
            if kind == "quantum":
                for name in ("absorber", "truncation_energy"):
                    if callable(options.get(name)):
                        options[name] = np.asarray(options[name](model.grid_points().reshape(-1, model.dimension)))
            tasks.append({"key": (key, kind), "kind": kind, "model": potential, "beta": beta,
                          "nbeads": beads, "options": options,
                          "memory_mb": _memory_mb(kind, potential, beta, beads, options)})
    return tasks


@lru_cache(maxsize=1)
def _implementation_digest():
    import scipy

    digest = hashlib.sha256((np.__version__ + scipy.__version__).encode())
    for name in ("subspace_rpmd.py", "subspace_model.py", "subspace_quantum.py", "subspace_native.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    try:
        from pyrpmd import _subspace
    except ImportError:
        pass
    else:
        digest.update(Path(_subspace.__file__).read_bytes())
    return digest.digest()


def _fingerprint(task):
    model = task["model"]
    if model.function is not None or model.gradient is not None:
        raise ValueError("task checkpoints require explicit gridded models, without Python potential callbacks")
    digest = hashlib.sha256(b"pyrpmd.subspace.tasks.v1")
    digest.update(_implementation_digest())

    def add(value):
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str((array.dtype.str, array.shape)).encode())
            digest.update(array.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                add(key)
                add(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                add(item)
        else:
            # Input identity must not turn NaN/Inf into JSON null: otherwise
            # an invalid option could reuse a checkpoint for a valid None default.
            scalar = value.item() if isinstance(value, np.generic) else value
            digest.update(json.dumps(scalar, sort_keys=True, allow_nan=False).encode())
        digest.update(b"\0")

    add((task["kind"], task["beta"], task["nbeads"], model.axes, model.values, model.masses,
         model.dividing_surface, model.reactant, model.reactant_reference, model.provenance,
         {k: v for k, v in task["options"].items() if k not in ("threads", "memory_limit_mb")}))
    return digest.hexdigest()


class _Checkpoints:
    def __init__(self, tasks, options):
        self.directory = Path(options["checkpoint_dir"]) if options.get("checkpoint_dir") else None
        self.resume = options.get("resume", True)
        self.paths = {}
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            for task in tasks:
                fingerprint = _fingerprint(task)
                self.paths[task["key"]] = (self.directory / (fingerprint + ".json"), fingerprint)

    def load(self, task):
        if self.directory is None or not self.resume:
            return None
        path, fingerprint = self.paths[task["key"]]
        if not path.exists():
            return None
        record = json.loads(path.read_text())
        if record["fingerprint"] != fingerprint:
            raise ValueError(f"checkpoint identity mismatch: {path}")
        data = record["result"]
        cls = QuantumRateResult if task["kind"] == "quantum" else RingPolymerRateResult
        # JSON null is the existing representation of an undefined error bar.
        for key in ("rate_standard_error", "qtst_log_standard_error", "transmission_standard_error"):
            if key in data and data[key] is None:
                data[key] = float("nan")
        result = cls(**data)
        result.diagnostics["checkpoint_reused"] = True
        return result

    def save(self, task, result):
        if self.directory is None:
            return
        path, fingerprint = self.paths[task["key"]]
        text = json.dumps({"fingerprint": fingerprint, "result": _json_safe(asdict(result))}, allow_nan=False)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(text)
        temporary.replace(path)


def _run_task(task, threads, memory_mb):
    options = dict(task["options"])
    started, cpu = time.perf_counter(), time.process_time()
    start_time = time.time_ns()
    blas_threads = min(threads, task.get("blas_threads", 8))
    with threadpool_limits(limits=blas_threads, user_api="blas"):
        if task["kind"] == "quantum":
            options["memory_limit_mb"] = min(memory_mb, options.get("memory_limit_mb") or memory_mb)
            result = exact_quantum_rate(task["model"], task["beta"], **options)
        else:
            options["threads"] = threads
            result = ring_polymer_rate(task["model"], task["beta"], task["nbeads"], **options)
        pools = [{k: p.get(k) for k in ("internal_api", "prefix", "num_threads")}
                 for p in threadpool_info() if p["user_api"] == "blas"]
    result.diagnostics["execution"] = {
        "pid": os.getpid(), "host": socket.gethostname(), "allocated_cpus": threads,
        "threads": blas_threads if task["kind"] == "quantum" else threads,
        "blas_threads": blas_threads, "blas_pools": pools,
        "estimated_peak_memory_mb": task["memory_mb"], "memory_budget_mb": memory_mb,
        "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu,
        "start_time_ns": start_time, "end_time_ns": time.time_ns(),
    }
    result.diagnostics["checkpoint_reused"] = False
    return result


@contextmanager
def _passive_openmp():
    """Spawned workers inherit OMP_WAIT_POLICY=PASSIVE unless the user set it.

    Idle OpenMP dynamics threads then sleep during Python sampling/analysis.
    Workers start lazily, so the variable is set for the pool's lifetime and
    the parent's environment is restored afterwards.
    """
    if "OMP_WAIT_POLICY" in os.environ:
        yield
        return
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    try:
        yield
    finally:
        os.environ.pop("OMP_WAIT_POLICY", None)


def _task_profile(task, cpus, blas_threads):
    """Estimate task work (s at 1 CPU), serial fraction alpha, and scaling cap."""
    model, beta, beads, options = task["model"], task["beta"], task["nbeads"], task["options"]
    if task["kind"] == "quantum":
        plan = quantum_solver_plan(model, **options)
        if plan["route"] == "sylvester":
            n = max(1, int(plan["retained_points"]))
            work = 1e-3 + 3.2e-9 * (n ** 3)
            if n < 256:
                alpha, cap = 1.0, 1
            elif n < 1500:
                alpha, cap = 0.35, max(1, min(4, blas_threads, cpus))
            else:
                alpha, cap = 0.40, max(1, min(blas_threads, cpus))
        else:
            grid_n = max(1, int(np.prod(model.shape)))
            states = max(1, min(int(options.get("max_flux_states", 256)), 64))
            work = 1e-2 + 2.0e-6 * grid_n * states * model.dimension
            alpha, cap = 0.20, max(1, min(blas_threads, cpus))
    else:
        windows = options.get("windows", 48)
        nw = int(windows) if np.ndim(windows) == 0 else len(windows)
        walkers, trajectories = int(options.get("walkers", 32)), int(options.get("trajectories", 512))
        burn_in, sweeps = int(options.get("burn_in", 200)), int(options.get("sweeps", 400))
        leapfrog = int(options.get("leapfrog_steps", 8))
        dt = float(options.get("md_dt") or 0.1 / _thermal_frequency(model, beta))
        duration = float(options.get("md_time") or 0.3 * (model.axes[0][-1] - model.axes[0][0]) * np.sqrt(beta * model.masses[0]))
        steps = max(10, int(np.ceil(duration / dt)))
        surf_sweeps = int(np.ceil(trajectories / max(walkers, 1)))
        hmc_steps = (nw * walkers * (burn_in + sweeps) + surf_sweeps * walkers) * leapfrog
        md_total = 2 * trajectories * steps
        polymer_steps = hmc_steps + md_total
        bead_evals = polymer_steps * int(beads)
        max_batch = max(nw * walkers, 2 * trajectories)
        try:
            from pyrpmd import _subspace
            native_omp = bool(getattr(_subspace, "openmp", 0))
        except ImportError:
            native_omp = False
        uses_native = (
            native_omp
            and options.get("backend", "auto") != "python"
            and model.function is None
            and model.gradient is None
        )
        if uses_native:
            work = 5e-3 + bead_evals * 1.3e-7 * (3.5 ** (model.dimension - 1))
            alpha = (0.28 * hmc_steps + 0.05 * md_total) / max(polymer_steps, 1)
            cap = max(1, min(cpus, max_batch))
        else:
            work = 5e-3 + bead_evals * 1.2e-6 * (2.2 ** (model.dimension - 1))
            alpha, cap = 1.0, 1
    return {"work": float(work), "alpha": float(alpha), "cap": int(cap)}


def _est_wall(profile, n_threads, overhead=0.02):
    eff = max(1, min(int(n_threads), profile["cap"]))
    return overhead + profile["work"] * (profile["alpha"] + (1.0 - profile["alpha"]) / eff)


def _best_overlap_split(p_heavy, p_partner, others_profiles, total_cpus):
    best = None
    for c_heavy in range(1, min(p_heavy["cap"], total_cpus - 1) + 1):
        c_other = total_cpus - c_heavy
        c_partner = min(p_partner["cap"], c_other)
        c_heavy_eff = min(p_heavy["cap"], total_cpus - c_partner)
        c_lane = total_cpus - c_heavy_eff
        lane_heavy = _est_wall(p_heavy, c_heavy_eff)
        lane_other = sum(_est_wall(p, min(c_lane, p["cap"])) for p in others_profiles)
        cost = max(lane_heavy, lane_other)
        key = (cost, -(c_heavy_eff + c_partner), abs(c_heavy_eff - c_partner))
        if best is None or key < best[0]:
            best = (key, cost, c_heavy_eff, c_partner)
    return best[1], best[2], best[3]


def _local(tasks, options, checkpoints, results):
    cpus, memory = options["cpus"], options["memory_limit_mb"]
    available = _affinity_cpus()
    if cpus > available:
        raise ValueError(f"execution.cpus={cpus} exceeds CPU affinity ({available})")
    for task in tasks:
        if task["memory_mb"] > memory:
            raise MemoryError(f"{task['key']} needs about {task['memory_mb']:.1f} MiB; budget is {memory:.1f} MiB")
    if not tasks:
        return
    max_workers = 1 if options["backend"] == "serial" else min(options.get("workers", cpus), cpus, len(tasks))
    if max_workers == 1:
        for task in tasks:
            value = _run_task(task, cpus, memory)
            checkpoints.save(task, value)
            results[task["key"]] = value
        return
    # Validate serialization before starting workers, including user callbacks.
    try:
        pickle.dumps(tasks)
    except (TypeError, pickle.PicklingError, AttributeError) as error:
        raise ValueError("process tasks need pickleable models; use gridded models or serial execution") from error
    profiles = {task["key"]: _task_profile(task, cpus, options["blas_threads"]) for task in tasks}
    pending = list(tasks)
    active = {}

    with _passive_openmp(), ProcessPoolExecutor(max_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        def submit(task, t_cpus):
            pending[:] = [item for item in pending if item is not task]
            future = pool.submit(_run_task, task, t_cpus, task["memory_mb"])
            active[future] = {
                "task": task,
                "cpus": t_cpus,
                "started": time.perf_counter(),
                "est": _est_wall(profiles[task["key"]], t_cpus),
            }

        while pending or active:
            while len(active) < max_workers and pending:
                free_cpus = cpus - sum(item["cpus"] for item in active.values())
                free_mem = memory - sum(item["task"]["memory_mb"] for item in active.values())
                if free_cpus < 1:
                    break
                admissible = [t for t in pending if t["memory_mb"] <= free_mem]
                if not admissible:
                    break
                if not active:
                    heaviest = max(admissible, key=lambda t: (_est_wall(profiles[t["key"]], free_cpus), -admissible.index(t)))
                    p_heavy = profiles[heaviest["key"]]
                    others_fit = [t for t in admissible if t is not heaviest and heaviest["memory_mb"] + t["memory_mb"] <= free_mem]
                    if others_fit:
                        if (
                            p_heavy["cap"] < free_cpus
                            and _est_wall(p_heavy, p_heavy["cap"])
                            >= 0.5 * sum(_est_wall(profiles[t["key"]], free_cpus - p_heavy["cap"]) for t in others_fit)
                        ):
                            submit(heaviest, p_heavy["cap"])
                            continue
                        if max_workers - len(active) >= 2 and free_cpus >= 2:
                            partner = max(others_fit, key=lambda t: (_est_wall(profiles[t["key"]], free_cpus), -admissible.index(t)))
                            others_profiles = [profiles[t["key"]] for t in others_fit]
                            par_cost, c_heavy, c_partner = _best_overlap_split(
                                p_heavy, profiles[partner["key"]], others_profiles, free_cpus
                            )
                            seq_cost = (
                                _est_wall(p_heavy, min(free_cpus, p_heavy["cap"]))
                                + sum(_est_wall(p, min(free_cpus, p["cap"])) for p in others_profiles)
                            )
                            if par_cost < seq_cost:
                                submit(heaviest, c_heavy)
                                submit(partner, c_partner)
                                continue
                            t0 = min(admissible, key=lambda t: (_est_wall(profiles[t["key"]], free_cpus), admissible.index(t)))
                        else:
                            t0 = admissible[0]
                    else:
                        t0 = admissible[0]
                    t0_cpus = (
                        free_cpus
                        if (len(pending) == 1 and t0["kind"] == "rpmd")
                        else min(free_cpus, profiles[t0["key"]]["cap"])
                    )
                    submit(t0, t0_cpus)
                else:
                    now = time.perf_counter()
                    soonest = min(active.values(), key=lambda item: max(0.0, item["est"] - (now - item["started"])))
                    rem_wait = max(0.0, soonest["est"] - (now - soonest["started"]))
                    combined_cpus = min(cpus, free_cpus + soonest["cpus"])
                    submitted = False
                    for cand in sorted(admissible, key=lambda t: (-_est_wall(profiles[t["key"]], free_cpus), admissible.index(t))):
                        pc = profiles[cand["key"]]
                        if pc["cap"] <= free_cpus or _est_wall(pc, free_cpus) <= rem_wait + _est_wall(pc, combined_cpus):
                            cand_cpus = (
                                free_cpus
                                if (len(admissible) == 1 and cand["kind"] == "rpmd" and not active)
                                else min(free_cpus, pc["cap"])
                            )
                            submit(cand, cand_cpus)
                            submitted = True
                            break
                    if not submitted:
                        break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)["task"]
                value = future.result()
                checkpoints.save(task, value)
                results[task["key"]] = value


def _packet(task, threads, memory):
    try:
        return _run_task(task, threads, memory), None
    except Exception as error:
        return None, f"{task['key']}: {type(error).__name__}: {error}"


def _mpi(tasks, options, checkpoints, results, comm, *, auto_cpus=False):
    from mpi4py import MPI

    local = comm.Split_type(MPI.COMM_TYPE_SHARED)
    affinity = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 1))
    union = set().union(*local.allgather(affinity))
    node_cpus = len(union) if auto_cpus else options["cpus"]
    threads = min(len(affinity), node_cpus // local.size)
    capacity = (threads, options["memory_limit_mb"] / local.size)
    error = None
    if threads < 1 or node_cpus > len(union):
        error = "MPI ranks per node and CPU affinity are incompatible with execution.cpus (a per-node budget)"
    capacities = comm.allgather(capacity)
    errors = comm.allgather(error)
    local.Free()
    if any(errors):
        raise ValueError(next(e for e in errors if e))
    error = None
    if comm.rank == 0:
        if any(not any(t["memory_mb"] <= cap[1] for cap in capacities) for t in tasks):
            error = "a subspace task exceeds every MPI rank's memory share; use fewer ranks per node or a larger budget"
    error = comm.bcast(error, root=0)
    if error:
        raise MemoryError(error)
    if comm.rank != 0:
        while True:
            task = comm.recv(source=0, tag=41)
            if task is None:
                break
            comm.send(_packet(task, *capacity), dest=0, tag=42)
    else:
        pending, active, idle = list(tasks), {}, set(range(comm.size))
        future = None
        with ThreadPoolExecutor(1) as own_worker:
            while pending or active:
                for rank in sorted(idle):
                    index = next((i for i, t in enumerate(pending) if t["memory_mb"] <= capacities[rank][1]), None)
                    if index is None or error:
                        continue
                    task = pending.pop(index)
                    active[rank] = task
                    idle.remove(rank)
                    if rank == 0:
                        future = own_worker.submit(_packet, task, *capacity)
                    else:
                        comm.send(task, dest=rank, tag=41)
                completed = []
                if future is not None and future.done():
                    completed.append((0, future.result()))
                    future = None
                status = MPI.Status()
                while comm.Iprobe(source=MPI.ANY_SOURCE, tag=42, status=status):
                    rank = status.Get_source()
                    completed.append((rank, comm.recv(source=rank, tag=42)))
                for rank, (value, failure) in completed:
                    task = active.pop(rank)
                    idle.add(rank)
                    if failure:
                        error = error or failure
                    else:
                        try:
                            checkpoints.save(task, value)
                            results[task["key"]] = value
                        except Exception as exc:
                            error = error or f"checkpoint write failed: {exc}"
                if error:
                    pending.clear()
                if active and not completed:
                    time.sleep(0.005)
        for rank in range(1, comm.size):
            comm.send(None, dest=rank, tag=41)
    error = comm.bcast(error, root=0)
    if error:
        raise RuntimeError(error)


def subspace_corrections(cases, *, execution=None):
    """Compute named corrections concurrently, preserving each task's estimator.

    ``cases`` maps names to ``model``, ``beta``, ``nbeads`` and optional
    ``quantum_options``, ``rpmd_options``, ``rpmd_model``. In MPI mode only rank
    zero supplies cases; all ranks call, and all receive the ordered result.
    Budgets are per process group on a local node, or per node under MPI.
    """
    from .subspace_correction import _assemble_correction

    raw_cpus = (execution or {}).get("cpus")
    auto_cpus = isinstance(raw_cpus, str) and raw_cpus == "auto"
    options = _options(execution)
    comm = None
    if options["backend"] == "mpi":
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    root = comm is None or comm.rank == 0
    tasks, checkpoints, results, failure = [], None, {}, None
    if root:
        try:
            tasks = _tasks(cases)
            for task in tasks:
                task["blas_threads"] = options["blas_threads"]
            checkpoints = _Checkpoints(tasks, options)
            pending = []
            for task in tasks:
                saved = checkpoints.load(task)
                if saved is None:
                    pending.append(task)
                else:
                    results[task["key"]] = saved
            tasks = pending
            if comm is not None:
                pickle.dumps(tasks)
        except Exception as error:
            if comm is None:
                raise
            failure = f"{type(error).__name__}: {error}"
    if comm is not None:
        failure = comm.bcast(failure, root=0)
        if failure:
            raise RuntimeError(failure)
        _mpi(tasks, options, checkpoints, results, comm, auto_cpus=auto_cpus)
    else:
        _local(tasks, options, checkpoints, results)
    corrections, failure = None, None
    if root:
        try:
            corrections = {key: _assemble_correction(case["model"], case["beta"], case["nbeads"],
                                                     results[(key, "quantum")], results[(key, "rpmd")])
                           for key, case in cases.items()}
        except Exception as error:
            if comm is None:
                raise
            failure = f"{type(error).__name__}: {error}"
    if comm is not None:
        corrections, failure = comm.bcast((corrections, failure), root=0)
        if failure:
            raise RuntimeError(failure)
    return corrections
