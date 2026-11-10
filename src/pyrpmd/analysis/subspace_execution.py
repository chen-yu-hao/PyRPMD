"""Ordered subspace QM/RPMD tasks, process/MPI transport and checked stage reuse.

Models are immutable numerical snapshots. Python callbacks are accepted only by
serial, uncached execution; they are never silently approximated by grid values.
Caches contain completed solver results, not interrupted HMC/quantum trajectories.
"""
from __future__ import annotations

from dataclasses import asdict, fields, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .subspace_model import SubspaceModel
from .subspace_native import positive_integer


def execution_options(options=None):
    given = dict(options or {})
    defaults = {"backend": "serial", "workers": 1, "scheduling": "static",
                "rpmd_threads": 1, "quantum_threads": 1, "memory_budget_mb": None,
                "cache_directory": None, "reuse": False}
    unknown = set(given) - set(defaults)
    if unknown:
        raise ValueError(f"unknown subspace execution options: {sorted(unknown)}")
    result = {**defaults, **given}
    if result["backend"] not in ("serial", "process", "mpi"):
        raise ValueError("execution.backend must be serial, process or mpi")
    if result["scheduling"] not in ("static", "dynamic"):
        raise ValueError("execution.scheduling must be static or dynamic")
    for name in ("workers", "rpmd_threads", "quantum_threads"):
        result[name] = positive_integer(result[name], name)
    if result["backend"] == "serial" and result["workers"] != 1:
        raise ValueError("serial execution requires workers=1")
    if not isinstance(result["reuse"], bool):
        raise ValueError("execution.reuse must be boolean")
    budget = result["memory_budget_mb"]
    if budget is not None and (isinstance(budget, bool) or not np.isfinite(budget) or budget <= 0):
        raise ValueError("memory_budget_mb must be finite and positive")
    return result


def snapshot_model(model):
    if type(model) is not SubspaceModel or model.function is not None or model.gradient is not None:
        raise ValueError("parallel/cached subspace tasks require an unmodified static-grid SubspaceModel")
    return SubspaceModel(model.axes, model.values, model.masses, model.dividing_surface,
                         model.reactant, model.reactant_reference, model.provenance)


def freeze_quantum_options(model, options):
    options = dict(options or {})
    for name in ("absorber", "truncation_energy"):
        if callable(options.get(name)):
            values = np.asarray(options[name](model.grid_points().reshape(-1, model.dimension)), dtype=float)
            options[name] = np.broadcast_to(values.reshape(model.shape) if values.size == np.prod(model.shape)
                                            else values, model.shape).copy()
    return options


def _canonical(value):
    if isinstance(value, np.ndarray):
        a = np.ascontiguousarray(value)
        if a.dtype.hasobject:
            raise TypeError("object arrays cannot identify a subspace task")
        return {"shape": a.shape, "dtype": a.dtype.str, "sha256": sha256(a.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"cannot fingerprint {type(value).__name__}")


def task_identity(task):
    import scipy
    from . import subspace_model, subspace_quantum, subspace_rpmd, subspace_native, subspace_linalg
    sources = {}
    for module in (subspace_model, subspace_quantum, subspace_rpmd, subspace_native, subspace_linalg):
        sources[module.__name__] = sha256(Path(module.__file__).read_bytes()).hexdigest()
    sources[__name__] = sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        from pyrpmd import _subspace_native
        sources["native_binary"] = sha256(Path(_subspace_native.__file__).read_bytes()).hexdigest()
    except ImportError:
        sources["native_binary"] = None
    model = task["model"]
    identity = {"schema": 1, "kind": task["kind"], "beta": task["beta"],
                "nbeads": task["nbeads"] if task["kind"] == "rpmd" else None,
                "axes": model.axes, "values": model.values, "masses": model.masses,
                "surface": model.dividing_surface, "reactant": model.reactant,
                "reference": model.reactant_reference, "provenance": model.provenance,
                "options": task["options"], "sources": sources,
                "numpy": np.__version__, "scipy": scipy.__version__}
    return sha256(json.dumps(_canonical(identity), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _read_cache(directory, key, kind):
    from .subspace_quantum import QuantumRateResult
    from .subspace_rpmd import RingPolymerRateResult
    cls = QuantumRateResult if kind == "quantum" else RingPolymerRateResult
    path = Path(directory) / f"{kind}-{key}.npz"
    try:
        manifest = json.loads(path.with_suffix(".json").read_text())
        if manifest.get("key") != key or manifest.get("sha256") != sha256(path.read_bytes()).hexdigest():
            return None
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {f.name for f in fields(cls)}:
                return None
            values = {name: (json.loads(str(z[name])) if name == "diagnostics" else
                             (z[name].item() if z[name].ndim == 0 else z[name].copy())) for name in z.files}
        return cls(**values)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(directory, key, kind, result):
    from ..io import atomic_write_text
    from .subspace_quantum import _json_safe
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind}-{key}.npz"
    values = asdict(result)
    values["diagnostics"] = json.dumps(_json_safe(values["diagnostics"]), sort_keys=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{kind}-", suffix=".npz", dir=directory)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **values); stream.flush(); os.fsync(stream.fileno())
        digest = sha256(Path(temporary).read_bytes()).hexdigest()
        os.replace(temporary, path)
        atomic_write_text(path.with_suffix(".json"), json.dumps({"key": key, "sha256": digest}))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _solve_task(task):
    from .subspace_quantum import exact_quantum_rate
    from .subspace_rpmd import ring_polymer_rate
    cache = task.get("cache_directory")
    key = task_identity(task) if cache else None
    if cache and task["reuse"]:
        saved = _read_cache(cache, key, task["kind"])
        if saved is not None:
            return replace(saved, diagnostics={**saved.diagnostics, "stage_cache_hit": True})
    start = perf_counter()
    with threadpool_limits(limits=task["blas_threads"], user_api="blas"):
        if task["kind"] == "quantum":
            result = exact_quantum_rate(task["model"], task["beta"], **task["options"])
        else:
            result = ring_polymer_rate(task["model"], task["beta"], task["nbeads"], **task["options"])
    result = replace(result, diagnostics={**result.diagnostics, "stage_cache_hit": False,
                                         "solver_wall_seconds": perf_counter() - start})
    if cache:
        _write_cache(cache, key, task["kind"], result)
    return result


def execute_rate_tasks(requests, options=None):
    """Compute ordered (QM, RPMD) pairs for one or more model requests.

    Every MPI rank must call this function; it returns results on all ranks.
    MPI distributes whole solver tasks (not a distributed dense eigensolver).
    """
    from ..parallel import run_tasks
    from ..parallel.execution import _optional_world, broadcast, root_call
    settings = execution_options(options)
    world = _optional_world()
    if world is not None and world.Get_size() > 1 and settings["backend"] != "mpi":
        raise ValueError("multi-rank subspace execution requires execution.backend='mpi'")
    if settings["backend"] == "mpi" and world is None:
        raise ValueError("execution.backend='mpi' requires an MPI launch")

    def prepare():
        tasks = []
        parallel = settings["backend"] != "serial"
        from .subspace_correction import _matched_rpmd_model
        for request in requests:
            positive_integer(request["nbeads"], "nbeads")
            if not np.isfinite(request["beta"]) or request["beta"] <= 0:
                raise ValueError("beta must be finite and positive")
            _matched_rpmd_model(request["model"], request.get("rpmd_model"))
            for kind in ("quantum", "rpmd"):
                model = request.get("rpmd_model") if kind == "rpmd" else None
                model = request["model"] if model is None else model
                opts = dict(request.get(kind + "_options") or {})
                if kind == "quantum":
                    opts = freeze_quantum_options(model, opts)
                else:
                    opts.setdefault("threads", settings["rpmd_threads"])
                if parallel or settings["cache_directory"]:
                    model = snapshot_model(model)
                tasks.append({"model": model, "kind": kind, "beta": float(request["beta"]),
                              "nbeads": int(request["nbeads"]), "options": opts,
                              "cache_directory": settings["cache_directory"], "reuse": settings["reuse"],
                              "blas_threads": settings["quantum_threads"] if kind == "quantum" else 1})
        return tasks

    tasks = root_call(prepare) if settings["backend"] == "mpi" else prepare()
    transport = {"backend": settings["backend"], "trajectory_workers": settings["workers"],
                 "scheduling": settings["scheduling"], "bead_workers": 1, "mode": "serial"}
    if settings["backend"] == "mpi":
        coordinators = int(settings["scheduling"] == "dynamic")
        groups = world.Get_size() - coordinators
        if groups < 1:
            raise ValueError("dynamic MPI requires a coordinator plus at least one worker")
        transport.update(trajectory_groups=groups, coordinator_ranks=coordinators)
    # A total local memory budget conservatively serializes solver tasks. Each
    # dense solve is checked before allocation. MPI budgets apply per rank.
    budget = settings["memory_budget_mb"]
    if budget is not None:
        for task in tasks:
            if task["kind"] == "quantum":
                configured = task["options"].get("max_memory_mb")
                task["options"]["max_memory_mb"] = budget if configured is None else min(budget, configured)
        if settings["backend"] == "process":
            transport["trajectory_workers"] = 1
    if settings["backend"] == "serial":
        results = [_solve_task(task) for task in tasks]
    else:
        results = run_tasks(_solve_task, tasks, transport)
        if settings["backend"] == "mpi":
            results = broadcast(results)
    return [(results[i], results[i + 1]) for i in range(0, len(results), 2)]
