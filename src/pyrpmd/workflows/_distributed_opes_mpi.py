"""Distributed OPES workflow helpers."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
from ..warm_start import initial_ring

from . import _opes_serial as _serial
for _name, _value in vars(_serial).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

from ..analysis.fes_result import FESResult, write_fes_result
from ..checkpoint import load_checkpoint
from ..opes import SharedOPESBias
from ..walker_exchange import (
    AsyncWalkerStore, allgather_centers, ensure_manifest, make_record, mpi_world,
    read_committed_records, validate_record)

MPI_KIND = "PyRPMD.shared-opes.mpi.v1"
ASYNC_KIND = "PyRPMD.shared-opes.async-walker.v1"

from ._distributed_opes_common import (
    _analyze_shared, _new_shared_bias, _shared_fingerprint,
    _validate_shared_options)

def _root_collective(comm, operation, label):
    """Run a root-only operation without stranding peers on an exception."""
    payload = failure = None
    if comm.Get_rank() == 0:
        try:
            payload = operation()
        except BaseException as error:
            failure = f"{type(error).__name__}: {error}"
    failure = comm.bcast(failure, root=0)
    if failure is not None:
        raise RuntimeError(f"{label} failed on MPI root: {failure}")
    return comm.bcast(payload, root=0)


def _mpi_path(output):
    return Path(output) / "checkpoints" / "opes_shared_mpi.npz"


def _read_mpi_root(config, system, output, cv, options):
    path = _mpi_path(output)
    if not path.exists():
        return None
    with load_checkpoint(path, expected_kind=MPI_KIND, require_integrity=True) as data:
        if not fingerprint_matches(
                data["fingerprint"], checkpoint_cv_config(config), system,
                {"opes_shared": 1, "walkers_mode": "mpi-sync"}):
            raise ValueError("shared MPI OPES checkpoint physics differs")
        walkers = config["sampling"]["replicas"]
        elapsed_array = np.asarray(data["elapsed"])
        if (elapsed_array.shape != () or elapsed_array.dtype.kind not in "iu"
                or not 0 <= int(elapsed_array) <= options["neq"] + options["nprod"]):
            raise ValueError("shared MPI OPES checkpoint has an invalid clock")
        elapsed = int(elapsed_array)
        if int(np.asarray(data["step"])) != elapsed:
            raise ValueError("shared MPI OPES state step disagrees with clock")
        dt = config["sampling"].get("dt_au", config["dt_au"])
        if not np.isclose(float(data["time"]), elapsed * dt, rtol=1e-10, atol=1e-12):
            raise ValueError("shared MPI OPES state time disagrees with clock")
        shape = (walkers, config["beads"], len(system.masses), 3)
        if (data["q"].shape != shape or data["p"].shape != shape
                or not np.isfinite(data["q"]).all() or not np.isfinite(data["p"]).all()):
            raise ValueError("shared MPI OPES state shape or values mismatch")
        expected = max(0, elapsed - options["neq"]) // options["interval"]
        samples = data["samples"].copy()
        steps = data["sample_steps"].copy()
        sampled_bias = data["sample_bias"].copy()
        energies = data["sample_energies"].copy()
        expected_steps = (options["neq"] + np.arange(1, expected + 1)
                          * options["interval"])
        if (samples.shape != (expected, walkers, cv.dimension)
                or steps.shape != (expected,) or not np.array_equal(steps, expected_steps)
                or sampled_bias.shape != (expected, walkers)
                or energies.shape != (expected, walkers)
                or not np.isfinite(samples).all()
                or not np.isfinite(sampled_bias).all()
                or not np.isfinite(energies).all()):
            raise ValueError("shared MPI OPES samples disagree with clock")
        rng = json.loads(str(data["rng"]))
        propagation = json.loads(str(data["propagation"]))
        operators = json.loads(str(data["operator_states"]))
        if not all(isinstance(value, list) and len(value) == walkers
                   for value in (rng, propagation, operators)):
            raise ValueError("shared MPI OPES per-walker state inventory is invalid")
        for state in rng:
            np.random.default_rng().bit_generator.state = state
        bias_state = json.loads(str(data["bias_state"]))
        bias = _new_shared_bias(cv, options, config["beta"], walkers, bias_state)
        updates = max(0, elapsed - options["neq"]) // options["pace"]
        if len(bias) != updates * walkers:
            raise ValueError("shared MPI OPES kernel count disagrees with clock")
        return {"elapsed": elapsed, "q": data["q"].copy(),
                "p": data["p"].copy(), "rng": rng,
                "bias_state": bias_state, "samples": list(samples),
                "sample_steps": list(steps), "sample_bias": list(sampled_bias),
                "sample_energies": list(energies), "propagation": propagation,
                "operator_states": operators}


def _read_mpi(config, system, output, cv, options, comm):
    return _root_collective(
        comm, lambda: _read_mpi_root(config, system, output, cv, options),
        "shared MPI OPES checkpoint read")


def _run_mpi(config, system, output, *, return_data=False):
    comm = mpi_world()
    cv, _, options = resolve_opes(config, system)
    walkers = _validate_shared_options(config, options)
    if comm.Get_size() != walkers:
        raise ValueError(f"shared OPES needs {walkers} MPI ranks, got {comm.Get_size()}")
    rank = comm.Get_rank()
    sampler = fes_sampler(config)
    restored = _read_mpi(config, system, output, cv, options, comm)
    provider = make_task_potential(config["potential"], system)
    try:
        dynamics = Dynamics(
            provider, system.masses, config["beta"], config["beads"],
            config["sampling"].get("dt_au", config["dt_au"]),
            seed=np.random.SeedSequence([int(config.get("seed", 0)), 811, rank]),
            native=config.get("native", config.get("dynamics", {}).get("native", True)),
            threads=config.get("parallel", {}).get("threads_per_rank", 1),
            **loop_options(config))
        if restored is None:
            state = State(initial_ring(config, system),
                          dynamics.sample_momenta())
            bias = _new_shared_bias(cv, options, config["beta"], walkers)
            elapsed = 0
            saved = {"samples": [], "sample_steps": [], "sample_bias": [],
                     "sample_energies": []}
        else:
            elapsed = restored["elapsed"]
            state = State(restored["q"][rank], restored["p"][rank], elapsed,
                          elapsed * config["sampling"].get("dt_au", config["dt_au"]))
            dynamics.rng.bit_generator.state = restored["rng"][rank]
            dynamics.propagation_stats = restored["propagation"][rank]
            if sampler.operator_checkpoint and restored["operator_states"][rank] is not None:
                dynamics.load_operator_state_dict(restored["operator_states"][rank])
            bias = _new_shared_bias(
                cv, options, config["beta"], walkers, restored["bias_state"])
            saved = {key: restored[key] for key in (
                "samples", "sample_steps", "sample_bias", "sample_energies")}
        checkpoint = _mpi_path(output)

        def save():
            local = {"q": state.q.copy(), "p": state.p.copy(),
                     "rng": copy.deepcopy(dynamics.rng.bit_generator.state),
                     "propagation": copy.deepcopy(dynamics.propagation_stats),
                     "operator": (dynamics.operator_state_dict()
                                  if sampler.operator_checkpoint else None)}
            gathered = comm.gather(local, root=0)

            def commit():
                atomic_savez(
                    checkpoint, kind=MPI_KIND,
                    fingerprint=_shared_fingerprint(config, system, "mpi-sync"),
                    elapsed=elapsed, step=elapsed, time=state.time,
                    q=np.stack([item["q"] for item in gathered]),
                    p=np.stack([item["p"] for item in gathered]),
                    rng=json.dumps([item["rng"] for item in gathered]),
                    bias_state=json.dumps(bias.state_dict(), allow_nan=False),
                    samples=np.asarray(saved["samples"]).reshape(
                        -1, walkers, cv.dimension),
                    sample_steps=np.asarray(saved["sample_steps"], dtype=np.int64),
                    sample_bias=np.asarray(saved["sample_bias"]).reshape(-1, walkers),
                    sample_energies=np.asarray(saved["sample_energies"]).reshape(-1, walkers),
                    propagation=json.dumps([item["propagation"] for item in gathered]),
                    operator_states=json.dumps([item["operator"] for item in gathered]))
                return True

            _root_collective(comm, commit, "shared MPI OPES checkpoint write")

        save()
        while elapsed < options["neq"] + options["nprod"]:
            previous = state.copy()
            old_rng = copy.deepcopy(dynamics.rng.bit_generator.state)
            old_stats = copy.deepcopy(dynamics.propagation_stats)
            old_operator = (dynamics.operator_state_dict()
                            if sampler.operator_checkpoint else None)
            local_error = None
            value = observed = energy = None
            production = elapsed + 1 - options["neq"]
            record = production > 0 and production % options["interval"] == 0
            due = production > 0 and production % options["pace"] == 0
            try:
                sampler.step(dynamics, state, bias)
                if record or due:
                    value = np.asarray(cv.value(state.q), dtype=np.float64)
                    if value.shape != (cv.dimension,) or not np.isfinite(value).all():
                        raise ValueError("CV returned invalid OPES observation")
                if record:
                    observed = float(bias.physical_energy(value))
                    energy = float(dynamics.energy(state, bias))
                    if not np.isfinite([observed, energy]).all():
                        raise ValueError("shared OPES observables must be finite")
            except BaseException as error:
                local_error = f"{type(error).__name__}: {error}"
            errors = comm.allgather(local_error)
            if any(error is not None for error in errors):
                state = previous
                dynamics.rng.bit_generator.state = old_rng
                dynamics.propagation_stats = old_stats
                if old_operator is not None:
                    dynamics.load_operator_state_dict(old_operator)
                save()
                raise RuntimeError("MPI OPES walker failed: " + "; ".join(
                    error for error in errors if error is not None))
            if record:
                values = np.asarray(comm.allgather(value), dtype=np.float64)
                observations = np.asarray(comm.allgather(observed), dtype=np.float64)
                energies = np.asarray(comm.allgather(energy), dtype=np.float64)
            if due:
                centers = allgather_centers(value, walkers=walkers, comm=comm)
                bias.update_walkers(centers, step=state.step, time=state.time)
            elapsed += 1
            dynamics._record_python_steps(1)
            if record and rank == 0:
                saved["samples"].append(values.copy())
                saved["sample_steps"].append(elapsed)
                saved["sample_bias"].append(observations.copy())
                saved["sample_energies"].append(energies.copy())
            if elapsed % config.get("output", {}).get("checkpoint_steps", 1000) == 0:
                save()
        save()
        return _root_collective(
            comm, lambda: _analyze_shared(
                bias, saved["samples"], config, system, output, options,
                mode="mpi-sync", checkpoint_files=[checkpoint], complete=True,
                return_data=return_data),
            "shared MPI OPES analysis")
    finally:
        if hasattr(provider, "close"):
            provider.close()


def reanalyze_mpi(config, system, output, cv, options, *, return_data=False):
    restored = _read_mpi_root(config, system, output, cv, options)
    if restored is None:
        raise ValueError("shared MPI OPES checkpoint does not exist")
    bias = _new_shared_bias(
        cv, options, config["beta"], config["sampling"]["replicas"],
        restored["bias_state"])
    return _analyze_shared(
        bias, restored["samples"], config, system, output, options,
        mode="mpi-sync", checkpoint_files=[_mpi_path(output)],
        complete=restored["elapsed"] == options["neq"] + options["nprod"],
        return_data=return_data)
