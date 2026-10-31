"""Independent tasks, persistent bead workers and hierarchical MPI execution.

The MPI path uses one trajectory/window group per B ranks. Every bead result
belongs to its current collective call; no stale/asynchronous forces are used.
No world barrier is inserted into an integration step.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
import copy
import multiprocessing as mp
from multiprocessing.util import Finalize
import os
import pickle
import sys
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits


def _int(value, name, minimum=1):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class ExecutionPlan:
    backend: str
    mode: str
    trajectory_workers: int
    bead_workers: int
    threads: int
    devices: tuple
    trajectories_per_gpu: int
    trajectory_groups: int
    scheduling: str = "static"
    coordinator_ranks: int = 0


def execution_plan(config=None):
    c = dict(config or {})
    backend = c.get("backend", "serial")
    mode = c.get("mode", c.get("pes_mode", "serial"))
    # Older user inputs used the backend names for PES execution modes.
    if backend in ("bead", "gpu"):
        mode, backend = backend, "process"
    if backend not in ("serial", "process", "mpi") or mode not in (
        "serial",
        "bead",
        "gpu",
    ):
        raise ValueError(
            "parallel backend must be serial/process/mpi and mode serial/bead/gpu"
        )
    workers = _int(
        c.get("trajectory_workers", c.get("trajectory_groups", 1)), "trajectory_workers"
    )
    beads = _int(c.get("bead_workers", 1), "bead_workers")
    threads = _int(c.get("threads_per_rank", c.get("threads", 1)), "threads_per_rank")
    groups = _int(c.get("trajectory_groups", workers), "trajectory_groups")
    if _int(c.get("pes_ranks", 1), "pes_ranks") != 1:
        raise ValueError(
            "This executor supports pes_ranks=1; internal MPI PES needs a dedicated adapter"
        )
    scheduling = c.get("scheduling", "static")
    if scheduling not in ("static", "dynamic"):
        raise ValueError("parallel.scheduling must be static or dynamic")
    coordinators = _int(c.get("coordinator_ranks", int(scheduling == "dynamic")),
                        "coordinator_ranks", 0)
    if scheduling == "dynamic":
        if backend != "mpi" or coordinators != 1:
            raise ValueError("Dynamic scheduling requires backend='mpi' and coordinator_ranks=1")
    elif coordinators != 0:
        raise ValueError("Static scheduling requires coordinator_ranks=0")
    raw_devices = c.get("devices", c.get("gpus", [0] if mode == "gpu" else []))
    if isinstance(raw_devices, (str, int)):
        raw_devices = [raw_devices]
    devices = tuple(
        str(x) if str(x).startswith("cuda:") else f"cuda:{x}" for x in raw_devices
    )
    if any(not device[5:].isdigit() for device in devices):
        raise ValueError("GPU devices must be nonnegative integer IDs or cuda:<ID>")
    if len(set(devices)) != len(devices):
        raise ValueError("GPU device list contains duplicates")
    per_gpu = _int(c.get("trajectories_per_gpu", 1), "trajectories_per_gpu")
    if mode == "gpu":
        if not devices:
            raise ValueError("GPU mode requires at least one device")
        if beads != 1:
            raise ValueError(
                "GPU batching uses bead_workers=1; frames are batched within each GPU provider"
            )
        expected = len(devices) * per_gpu
        if "trajectory_workers" in c and workers != expected and backend != "mpi":
            raise ValueError(
                "trajectory_workers must equal len(devices)*trajectories_per_gpu in GPU mode"
            )
        if backend != "mpi":
            workers = expected
    if backend != "mpi" and workers > 1:
        backend = "process"
    if beads > 1:
        mode = "bead" if mode != "gpu" else mode
    return ExecutionPlan(
        backend, mode, workers, beads, threads, devices, per_gpu, groups,
        scheduling, coordinators,
    )


def _optional_world():
    # Merely importing mpi4py.MPI may call MPI_Init and abort in environments
    # without transport/socket access. A serial workflow must not initialize MPI
    # just because an optional wheel happens to be installed.
    mpi = sys.modules.get("mpi4py.MPI")
    if mpi is not None and mpi.Is_initialized() and not mpi.Is_finalized():
        return mpi.COMM_WORLD
    launched = any(
        key in os.environ
        for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "MPI_LOCALRANKID")
    )
    if not launched:
        return None
    try:
        from mpi4py import MPI
    except (ImportError, RuntimeError, OSError) as exc:
        raise RuntimeError(
            "An MPI launcher was detected but mpi4py/MPI is unavailable"
        ) from exc
    return MPI.COMM_WORLD


def is_root():
    """Do not initialize MPI for an ordinary serial/process invocation."""
    world = _optional_world()
    return world is None or world.Get_rank() == 0


def broadcast(value, root=0):
    """World broadcast for multi-stage workflows; identity without MPI."""
    world = _optional_world()
    return value if world is None else world.bcast(value, root=root)


def root_call(function, *args, **kwargs):
    """Run analysis/I/O on root and broadcast its value or its exception."""
    payload = None
    if is_root():
        try:
            payload = (function(*args, **kwargs), None)
        except Exception as exc:
            payload = (None, _result_error(exc))
    value, error = broadcast(payload)
    if error is not None:
        raise RuntimeError("Root stage failed: " + error)
    return value


def _result_error(exc):
    return f"{type(exc).__name__}: {exc}"


def _combine(results):
    from pyrpmd.potentials import PESResult, PotentialError

    virials = [r.virial for r in results]
    present = [v is not None for v in virials]
    if any(present) and not all(present):
        raise PotentialError("Bead PES chunks disagree on optional virial availability")

    return PESResult(
        np.concatenate([np.asarray(r.energy) for r in results]),
        np.concatenate([np.asarray(r.forces) for r in results]),
        np.concatenate([np.asarray(r.status, dtype=np.int32) for r in results]),
        tuple(m for r in results for m in r.messages),
        None if not any(present) else np.concatenate(virials),
    )


_BEAD_PROVIDER = None


@contextmanager
def _blas_thread_limit():
    """Limit loaded BLAS pools and newly imported libraries for one task scope.

    These controls are process-wide. They deliberately leave OpenMP dynamics
    threads alone, and restore both the environment and loaded pool settings
    when an in-process batch returns or raises.
    """
    variables = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    previous = {name: os.environ.get(name) for name in variables}
    try:
        for name in variables:
            os.environ[name] = "1"
        with threadpool_limits(limits=1, user_api="blas"):
            yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _bead_init(spec, system):
    global _BEAD_PROVIDER
    from pyrpmd.potentials import make_potential

    # Keep one limiter for the worker's entire lifetime, rather than scanning
    # loaded libraries on every PES request. Finalize also runs for the normal
    # multiprocessing exit path, where ordinary atexit hooks are not assured.
    limits = ExitStack()
    try:
        limits.enter_context(_blas_thread_limit())
        _BEAD_PROVIDER = make_potential(spec, system)
        # A provider factory may have loaded another BLAS implementation.
        limits.enter_context(threadpool_limits(limits=1, user_api="blas"))
    except BaseException:
        limits.close()
        raise
    Finalize(None, limits.close, exitpriority=10)


def _bead_evaluate(x):
    return _BEAD_PROVIDER.evaluate(x)


class ProcessBeadPotential:
    """Persistent spawn workers isolate non-reentrant legacy Fortran libraries."""

    def __init__(self, spec, system, workers):
        self.workers = _int(workers, "bead_workers")
        self.metadata = {"provider": "process-bead", "workers": self.workers}
        self.pool = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=mp.get_context("spawn"),
            initializer=_bead_init,
            initargs=(spec, system),
        )
        self.closed = False

    def evaluate(self, positions):
        if self.closed:
            raise RuntimeError("Potential is closed")
        x = np.asarray(positions, dtype=np.float64)
        if x.ndim != 3 or len(x) == 0 or x.shape[2] != 3:
            raise ValueError("PES positions must have shape [B,N,3] with B>0")
        parts = [
            np.ascontiguousarray(a)
            for a in np.array_split(x, min(self.workers, len(x)))
        ]
        return _combine(list(self.pool.map(_bead_evaluate, parts)))

    def close(self):
        if not self.closed:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.closed = True


class _MPIBeadPotential:
    def __init__(self, spec, system, context):
        from pyrpmd.potentials import make_potential

        self.comm = context.comm
        self._owner_context = context
        self.identifier = context.next_id
        context.next_id += 1
        self.metadata = {"provider": "mpi-bead", "workers": self.comm.Get_size()}
        self.comm.bcast(("create", self.identifier, spec, system), root=0)
        local_error = None
        self.local = None
        try:
            self.local = make_potential(spec, system)
        except Exception as exc:
            local_error = _result_error(exc)
        errors = self.comm.gather(local_error, root=0)
        self.closed = False
        if any(errors):
            # Every rank has returned from creation. Release any contexts that
            # succeeded before reporting another rank's initialization error.
            try:
                self.close()
            except Exception as exc:
                errors.append(_result_error(exc))
            raise RuntimeError(
                "PES initialization failed: " + "; ".join(e for e in errors if e)
            )
        context.providers.append(self)

    def evaluate(self, positions):
        if self.closed:
            raise RuntimeError("Potential is closed")
        x = np.ascontiguousarray(positions, dtype=np.float64)
        if x.ndim != 3 or x.shape[-1] != 3 or len(x) < 1:
            raise ValueError("PES positions must have shape [B,N,3]")
        parts = [
            np.ascontiguousarray(y) for y in np.array_split(x, self.comm.Get_size())
        ]
        self.comm.bcast(("evaluate", self.identifier), root=0)
        local = self.comm.scatter(parts, root=0)
        response = _evaluate_response(self.local, local)
        responses = self.comm.gather(response, root=0)
        errors = [r[1] for r in responses if r[1] is not None]
        if errors:
            raise RuntimeError("MPI bead PES failure: " + "; ".join(errors))
        return _combine([r[0] for r in responses if r[0] is not None])

    def close(self):
        if not self.closed:
            self.comm.bcast(("close", self.identifier), root=0)
            error = None
            try:
                if hasattr(self.local, "close"):
                    self.local.close()
            except Exception as exc:
                error = _result_error(exc)
            errors = self.comm.gather(error, root=0)
            self.closed = True
            if self in self._owner_context.providers:
                self._owner_context.providers.remove(self)
            self._owner_context = None
            if any(errors):
                raise RuntimeError(
                    "PES close failed: " + "; ".join(e for e in errors if e)
                )


def _evaluate_response(provider, x):
    if len(x) == 0:
        return None, None
    try:
        return provider.evaluate(x), None
    except Exception as exc:
        return None, _result_error(exc)


def _mpi_service(comm):
    from pyrpmd.potentials import make_potential

    providers = {}
    while True:
        command = comm.bcast(None, root=0)
        op = command[0]
        if op == "finish":
            break
        if op == "create":
            _, key, spec, system = command
            error = None
            try:
                providers[key] = make_potential(spec, system)
            except Exception as exc:
                error = _result_error(exc)
            comm.gather(error, root=0)
        elif op == "evaluate":
            x = comm.scatter(None, root=0)
            comm.gather(_evaluate_response(providers.get(command[1]), x), root=0)
        elif op == "close":
            provider = providers.pop(command[1], None)
            error = None
            try:
                if provider is not None and hasattr(provider, "close"):
                    provider.close()
            except Exception as exc:
                error = _result_error(exc)
            comm.gather(error, root=0)
        else:
            raise RuntimeError(f"Invalid internal MPI command {op!r}")
    for provider in providers.values():
        if hasattr(provider, "close"):
            provider.close()


@dataclass
class _Context:
    plan: ExecutionPlan
    slot: int
    comm: Any = None
    next_id: int = 0
    providers: list = field(default_factory=list)


_CONTEXT = None


@contextmanager
def _context(plan, slot, comm=None):
    global _CONTEXT
    previous = _CONTEXT
    _CONTEXT = _Context(plan, slot, comm)
    try:
        yield _CONTEXT
    finally:
        errors = []
        try:
            # A task can fail before its own finally block or omit close().
            # Finish these collectives before the group service receives finish.
            for provider in tuple(_CONTEXT.providers):
                if not provider.closed:
                    try:
                        provider.close()
                    except Exception as exc:
                        errors.append(_result_error(exc))
        finally:
            _CONTEXT = previous
        if errors:
            raise RuntimeError("MPI PES cleanup failed: " + "; ".join(errors))


def make_task_potential(spec, system):
    """Construct the appropriate PES inside a task dispatched by run_tasks."""
    from pyrpmd.potentials import make_potential

    context = _CONTEXT
    if context is None:
        return make_potential(spec, system)
    plan = context.plan
    spec = copy.deepcopy(spec)
    if plan.mode == "gpu":
        device = plan.devices[
            (context.slot // plan.trajectories_per_gpu) % len(plan.devices)
        ]
        spec["device"] = device
        # Device choice is process-local. It does not modify visible device IDs.
        if spec.get("provider", spec.get("type")) not in (
            "torch",
            "python",
            "ase",
            "deepmd",
            "mace",
            "nequip",
            "fairchem",
        ):
            raise ValueError(
                "GPU execution requires a provider with an actual GPU implementation"
            )
    if context.comm is not None and context.comm.Get_size() > 1:
        return _MPIBeadPotential(spec, system, context)
    if plan.mode == "bead" and plan.bead_workers > 1:
        return ProcessBeadPotential(spec, system, plan.bead_workers)
    provider = make_potential(spec, system)
    if plan.mode == "gpu":
        metadata = getattr(provider, "metadata", {})
        declared = (
            metadata.get("execution_device", metadata.get("device", ""))
            if isinstance(metadata, dict)
            else ""
        )
        requested = spec["device"]
        if str(declared) != requested and not (
            declared == "cuda" and requested == "cuda:0"
        ):
            if hasattr(provider, "close"):
                provider.close()
            raise ValueError(
                f"GPU task requested {requested}, but provider declares {declared!r}"
            )
    return provider


def _execute_batch(payload):
    function, indexed_tasks, plan, slot = payload
    # NumPy is imported before spawn invokes this function, so environment
    # variables alone cannot limit its already initialized BLAS thread pool.
    with _blas_thread_limit(), _context(plan, slot):
        return [(index, function(task)) for index, task in indexed_tasks]


_TASK_TAG = 4101
_RESULT_TAG = 4102


def _dynamic_dispatch(control, leaders, count, mpi):
    """Refill the first available group; drain all in-flight work on failure.

    Only the coordinator calls this helper. A separate communicator keeps these
    point-to-point messages distinct from workflow and bead collectives.
    """
    next_index, active, results, errors = 0, {}, {}, []
    for leader in leaders:
        index = next_index if next_index < count else None
        control.send(index, dest=leader, tag=_TASK_TAG)
        if index is not None:
            active[leader] = index
            next_index += 1
    while active:
        status = mpi.Status()
        packet = control.recv(source=mpi.ANY_SOURCE, tag=_RESULT_TAG, status=status)
        leader = status.Get_source()
        expected = active.pop(leader)
        try:
            index, value, error = pickle.loads(packet)
            if index != expected:
                raise RuntimeError(f"group {leader} returned task {index}, expected {expected}")
        except Exception as exc:
            index, value, error = expected, None, _result_error(exc)
        if error is not None:
            errors.append(f"task {index}: {error}")
        else:
            results[index] = value
        # A failed task stops new work. Already-running groups still finish and
        # receive STOP, so no peer remains blocked in recv or in a bead service.
        index = next_index if not errors and next_index < count else None
        control.send(index, dest=leader, tag=_TASK_TAG)
        if index is not None:
            active[leader] = index
            next_index += 1
    return [results[i] for i in sorted(results)], errors


def _dynamic_worker(function, tasks, plan, slot, group, control):
    while True:
        index = control.recv(source=0, tag=_TASK_TAG)
        if index is None:
            return
        try:
            with _context(plan, slot, group):
                value = function(tasks[index])
        except Exception as exc:
            value, error = None, _result_error(exc)
        else:
            error = None
        try:
            packet = pickle.dumps((index, value, error), protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            # Serialize before send: an unpicklable task result must become a
            # collective error, rather than strand the coordinator in recv.
            packet = pickle.dumps((index, None, "Task result serialization failed: " + _result_error(exc)),
                                  protocol=pickle.HIGHEST_PROTOCOL)
        control.send(packet, dest=0, tag=_RESULT_TAG)


def _mpi_run(function, tasks, plan):
    try:
        from mpi4py import MPI
    except (ImportError, RuntimeError, OSError) as exc:
        raise RuntimeError(
            "MPI backend requires the optional mpi4py package and an MPI runtime"
        ) from exc
    world = MPI.COMM_WORLD
    expected = plan.coordinator_ranks + plan.trajectory_groups * plan.bead_workers
    if world.Get_size() != expected:
        raise ValueError(
            f"MPI world has {world.Get_size()} ranks; expected "
            f"coordinator_ranks+G*B={expected}"
        )
    rank = world.Get_rank()
    worker_rank = rank - plan.coordinator_ranks
    group_id = worker_rank // plan.bead_workers if worker_rank >= 0 else MPI.UNDEFINED
    device_slot = group_id
    if plan.mode == "gpu":
        # CUDA device IDs are local to each node / CUDA_VISIBLE_DEVICES. Global
        # rank arithmetic is wrong when the launcher scatters ranks over nodes.
        local = world.Split_type(MPI.COMM_TYPE_SHARED, key=rank)
        workers = local.allgather(worker_rank >= 0)
        device_slot = sum(workers[:local.Get_rank()])
        expected_local = len(plan.devices) * plan.trajectories_per_gpu
        local_error = None
        local_workers = sum(workers)
        if local_workers and local_workers != expected_local:
            local_error = (
                f"MPI GPU node {MPI.Get_processor_name()} has {local_workers} worker ranks; "
                f"expected len(devices)*trajectories_per_gpu={expected_local}"
            )
        local.Free()
        placement_errors = world.allgather(local_error)
        if any(placement_errors):
            raise ValueError("; ".join(sorted(set(e for e in placement_errors if e))))
    task_counts = world.allgather(len(tasks))
    if len(set(task_counts)) != 1:
        raise ValueError("Every MPI rank must supply the same task list")
    group = world.Split(group_id, rank)
    control = world.Dup() if plan.scheduling == "dynamic" else None
    results, error = [], None
    if worker_rank < 0:
        leaders = [1 + i * plan.bead_workers for i in range(plan.trajectory_groups)]
        results, errors = _dynamic_dispatch(control, leaders, len(tasks), MPI)
        if errors:
            error = "; ".join(errors)
    elif group.Get_rank() == 0:
        try:
            if control is not None:
                _dynamic_worker(function, tasks, plan, device_slot, group, control)
            else:
                with _context(plan, device_slot, group):
                    for index in range(group_id, len(tasks), plan.trajectory_groups):
                        results.append((index, function(tasks[index])))
        except Exception as exc:
            error = _result_error(exc)
        finally:
            group.bcast(("finish",), root=0)
    else:
        try:
            _mpi_service(group)
        except Exception as exc:
            error = _result_error(exc)
    all_errors = world.allgather(error)
    if control is not None:
        control.Free()
        gathered = None
    else:
        gathered = world.gather(results if group.Get_rank() == 0 else [], root=0)
    if worker_rank >= 0:
        group.Free()
    if any(all_errors):
        raise RuntimeError("MPI task failed: " + "; ".join(e for e in all_errors if e))
    if rank != 0:
        return None
    if plan.scheduling == "dynamic":
        return results
    flat = [item for sub in gathered for item in sub]
    return [value for _, value in sorted(flat, key=lambda pair: pair[0])]


def run_tasks(function, tasks, parallel_config=None):
    """Run importable ``function(task)``; MPI returns data on rank zero only.

    Static process slots own task batches, keeping GPU placement fixed. MPI
    dynamic scheduling uses one coordinator and refills complete trajectory
    groups; each group's bead ranks remain attached to that group.
    """
    tasks = list(tasks)
    plan = execution_plan(parallel_config)
    if plan.backend == "mpi":
        return _mpi_run(function, tasks, plan)
    world = _optional_world()
    if world is not None and world.Get_size() > 1:
        raise ValueError(
            "A multi-rank MPI launch requires parallel.backend='mpi'; local backends would duplicate tasks"
        )
    if not tasks:
        return []
    count = min(plan.trajectory_workers, len(tasks))
    batches = [
        [(i, tasks[i]) for i in range(slot, len(tasks), count)] for slot in range(count)
    ]
    if count == 1:
        values = _execute_batch((function, batches[0], plan, 0))
    else:
        payloads = [(function, batch, plan, slot) for slot, batch in enumerate(batches)]
        with ProcessPoolExecutor(
            max_workers=count, mp_context=mp.get_context("spawn")
        ) as pool:
            values = [
                item for sub in pool.map(_execute_batch, payloads) for item in sub
            ]
    return [value for _, value in sorted(values, key=lambda pair: pair[0])]


class TaskExecutor:
    def __init__(self, config=None):
        self.config = dict(config or {})

    def map(self, function, tasks):
        return run_tasks(function, tasks, self.config)
