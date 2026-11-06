"""Distributed shared-metadynamics workflow helpers."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np
from ..bias import CompositeBias
from ..checkpoint import atomic_savez, load_checkpoint
from ..dynamics import Dynamics, State
from ..potentials import make_potential
from ..propagation import loop_options
from ..units import parse_quantity
from ..walker_exchange import allgather_centers, mpi_world
from .common import _fingerprint, fingerprint_matches, write_json
from .metadynamics import resolve_metadynamics, resolve_walls
from .shared_metadynamics import MPI_KIND, _analyze, _bias, validate_shared

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


def _read_mpi_checkpoint_root(config, system, output, cv, options):
    path = Path(output) / "checkpoints" / "shared_metadynamics_mpi.npz"
    if not path.exists():
        return None
    with load_checkpoint(path, expected_kind=MPI_KIND, require_integrity=True) as data:
        if not fingerprint_matches(
                data["fingerprint"], config, system,
                {"shared_metadynamics": 2, "walkers_mode": "mpi-sync"}):
            raise ValueError("MPI shared metadynamics checkpoint physics differs")
        elapsed_array = np.asarray(data["elapsed"])
        if (elapsed_array.shape != () or elapsed_array.dtype.kind not in "iu"
                or not 0 <= int(elapsed_array) <= options["neq"] + options["nprod"]):
            raise ValueError("MPI shared metadynamics checkpoint has an invalid clock")
        elapsed = int(elapsed_array)
        walkers = config["sampling"]["replicas"]
        shape = (walkers, config["beads"], system.natoms, 3)
        if data["q"].shape != shape or data["p"].shape != shape:
            raise ValueError("MPI shared metadynamics state shape mismatch")
        dt = config["sampling"].get("dt_au", config["dt_au"])
        if not np.isclose(float(data["time"]), elapsed * dt, rtol=1e-10, atol=1e-12):
            raise ValueError("MPI shared metadynamics time disagrees with clock")
        expected = max(0, elapsed - options["neq"]) // options["interval"]
        steps = data["sample_steps"].copy()
        expected_steps = (options["neq"] + np.arange(1, expected + 1)
                          * options["interval"])
        if not np.array_equal(steps, expected_steps):
            raise ValueError("MPI shared metadynamics sample clock is inconsistent")
        samples = data["samples"].copy()
        sampled_bias = data["sample_bias"].copy()
        if (samples.shape != (expected, walkers, cv.dimension)
                or sampled_bias.shape != (expected, walkers)
                or not np.isfinite(samples).all()
                or not np.isfinite(sampled_bias).all()):
            raise ValueError("MPI shared metadynamics samples are invalid")
        configurations = data["configurations"].copy()
        expected_config = ((expected, *shape) if options["store_configurations"]
                           else (0, *shape))
        if configurations.shape != expected_config or not np.isfinite(configurations).all():
            raise ValueError("MPI shared metadynamics configurations are invalid")
        rng = json.loads(str(data["rng"]))
        if not isinstance(rng, list) or len(rng) != walkers:
            raise ValueError("MPI shared metadynamics RNG inventory is invalid")
        for state in rng:
            np.random.default_rng().bit_generator.state = state
        hills = json.loads(str(data["hills"]))
        bias = _bias(cv, config, options, hills)
        production = max(0, elapsed - options["neq"])
        if bias.hill_count != walkers * (production // options["pace"]):
            raise ValueError("MPI shared metadynamics history disagrees with clock")
        for event, hill in enumerate(hills):
            expected_step = (options["neq"]
                             + (event // walkers + 1) * options["pace"])
            if (hill["step"] != expected_step or not np.isclose(
                    hill["time"], expected_step * dt,
                    rtol=1e-10, atol=1e-12)):
                raise ValueError(
                    "MPI shared metadynamics hill clock disagrees with schedule")
        return {"elapsed": elapsed, "q": data["q"].copy(),
                "p": data["p"].copy(), "rng": rng, "hills": hills,
                "sample_steps": list(steps), "samples": list(samples),
                "sample_bias": list(sampled_bias),
                "configurations": list(configurations)}


def _mpi_checkpoint(config, system, output, cv, options, comm):
    return _root_collective(
        comm, lambda: _read_mpi_checkpoint_root(
            config, system, output, cv, options),
        "MPI shared metadynamics checkpoint read")


def _run_mpi(config, system, output, *, return_data=False):
    comm = mpi_world()
    cv, _, options = resolve_metadynamics(config, system)
    checked = copy.deepcopy(config)
    checked["parallel"] = {}
    walkers = validate_shared(checked, options)
    if comm.Get_size() != walkers:
        raise ValueError(
            f"MPI shared metadynamics needs {walkers} ranks, got {comm.Get_size()}")
    rank = comm.Get_rank()
    checkpoint = Path(output) / "checkpoints" / "shared_metadynamics_mpi.npz"
    restored = _mpi_checkpoint(config, system, output, cv, options, comm)
    provider = make_potential(config["potential"], system)
    try:
        dynamics = Dynamics(
            provider, system.masses, config["beta"], config["beads"],
            config["sampling"].get("dt_au", config["dt_au"]),
            seed=np.random.SeedSequence([config.get("seed", 0), 2267, rank]),
            native=config.get("native", config.get("dynamics", {}).get("native", True)),
            threads=config.get("parallel", {}).get("threads_per_rank", 1),
            fixed_atoms=system.fixed_atoms, fixed_positions=system.geometry,
            **loop_options(config))
        if restored is None:
            state = State(np.repeat(system.geometry[None], config["beads"], axis=0),
                          dynamics.sample_momenta())
            bias, elapsed = _bias(cv, config, options), 0
            saved = {"samples": [], "sample_steps": [], "sample_bias": [],
                     "configurations": []}
        else:
            elapsed = restored["elapsed"]
            state = State(restored["q"][rank], restored["p"][rank], elapsed,
                          elapsed * config["sampling"].get("dt_au", config["dt_au"]))
            dynamics.integrator._check(state.q, state.p)
            dynamics._check_fixed(state.q, state.p)
            dynamics.rng.bit_generator.state = restored["rng"][rank]
            bias = _bias(cv, config, options, restored["hills"])
            saved = {key: restored[key] for key in (
                "samples", "sample_steps", "sample_bias", "configurations")}
        walls = resolve_walls(config, system)
        total_bias = CompositeBias(bias, walls)
        shape = (walkers, config["beads"], system.natoms, 3)

        def save():
            local = {"q": state.q.copy(), "p": state.p.copy(),
                     "rng": copy.deepcopy(dynamics.rng.bit_generator.state)}
            gathered = comm.gather(local, root=0)

            def commit():
                atomic_savez(
                    checkpoint, kind=MPI_KIND,
                    fingerprint=_fingerprint(
                        config, system,
                        {"shared_metadynamics": 2, "walkers_mode": "mpi-sync"}),
                    elapsed=elapsed, time=state.time,
                    q=np.stack([item["q"] for item in gathered]),
                    p=np.stack([item["p"] for item in gathered]),
                    rng=json.dumps([item["rng"] for item in gathered]),
                    hills=json.dumps(bias.hills, allow_nan=False),
                    sample_steps=np.asarray(saved["sample_steps"], dtype=np.int64),
                    samples=np.asarray(saved["samples"]).reshape(
                        -1, walkers, cv.dimension),
                    sample_bias=np.asarray(saved["sample_bias"]).reshape(-1, walkers),
                    configurations=np.asarray(saved["configurations"]).reshape(-1, *shape))
                return True

            _root_collective(comm, commit, "MPI shared metadynamics checkpoint write")

        save()
        sampling = config["sampling"]
        while elapsed < options["neq"] + options["nprod"]:
            previous = state.copy()
            rng = copy.deepcopy(dynamics.rng.bit_generator.state)
            count = bias.hill_count
            local_error = None
            local_value = None
            try:
                active = (total_bias if elapsed >= options["neq"] or
                          config["metadynamics"].get("walls_during_equilibration", True)
                          else bias)
                if sampling.get("thermostat", "andersen") == "pile-l":
                    dynamics.sample_step(
                        state, active,
                        gamma_centroid=parse_quantity(
                            sampling.get("gamma_centroid", "0.01 fs^-1"), "frequency"),
                        method=sampling.get("integrator"))
                else:
                    if state.step % sampling.get("andersen_interval", 10) == 0:
                        state.p[:] = dynamics.sample_momenta()
                    dynamics.step(state, active)
                production = elapsed + 1 - options["neq"]
                record = production > 0 and production % options["interval"] == 0
                deposit = production > 0 and production % options["pace"] == 0
                if record or deposit:
                    local_value = np.asarray(cv.value(state.q), dtype=np.float64)
            except BaseException as error:  # coordinated rollback, no rank hangs
                local_error = f"{type(error).__name__}: {error}"
            errors = comm.allgather(local_error)
            if any(error is not None for error in errors):
                state = previous
                dynamics.rng.bit_generator.state = rng
                bias.truncate_history(count)
                save()
                raise RuntimeError("MPI walker step failed: " + "; ".join(
                    error for error in errors if error is not None))
            values = None
            if record or deposit:
                values = allgather_centers(
                    local_value, walkers=walkers, comm=comm)
            if record:
                # Match the serial implementation's vectorized evaluation,
                # including its floating-point summation order.
                before = np.asarray(bias.physical_energy(values), dtype=np.float64)
            if deposit:
                bias.deposit_walkers(values, step=state.step, time=state.time)
            elapsed += 1
            if record and rank == 0:
                saved["samples"].append(values.copy())
                saved["sample_steps"].append(elapsed)
                saved["sample_bias"].append(before.copy())
                if options["store_configurations"]:
                    saved["configurations"].append(np.stack(comm.gather(
                        state.q.copy(), root=0)))
            elif record and options["store_configurations"]:
                comm.gather(state.q.copy(), root=0)
            if elapsed % config.get("output", {}).get("checkpoint_steps", 1000) == 0:
                save()
        save()
        def analyze():
            ensemble = dict(
                states=[], rng=[], bias=bias, elapsed=elapsed,
                samples=saved["samples"], sample_steps=saved["sample_steps"],
                sample_bias=saved["sample_bias"],
                configurations=saved["configurations"])
            result = _analyze(
                config, system, output, cv, options, ensemble, return_data)
            summary = result["summary"] if return_data else result
            summary["walkers_mode"] = "mpi-sync"
            summary["physical_converged"] = False
            summary["files"]["checkpoint"] = str(checkpoint)
            summary["warnings"].append(
                "MPI ranks commit every deposition from one rank-ordered allgather; this is the serial shared-batch algorithm.")
            write_json(Path(output) / "summary.json", summary)
            return result

        return _root_collective(
            comm, analyze, "MPI shared metadynamics analysis")
    finally:
        if hasattr(provider, "close"):
            provider.close()


def reanalyze_mpi(config, system, output, cv, options, *, return_data=False):
    restored = _read_mpi_checkpoint_root(config, system, output, cv, options)
    if restored is None:
        raise ValueError("MPI shared metadynamics checkpoint does not exist")
    bias = _bias(cv, config, options, restored["hills"])
    ensemble = dict(
        states=[], rng=[], bias=bias, elapsed=restored["elapsed"],
        samples=restored["samples"], sample_steps=restored["sample_steps"],
        sample_bias=restored["sample_bias"],
        configurations=restored["configurations"])
    result = _analyze(
        config, system, output, cv, options, ensemble, return_data)
    summary = result["summary"] if return_data else result
    summary["walkers_mode"] = "mpi-sync"
    summary["physical_converged"] = False
    summary["files"]["checkpoint"] = str(
        Path(output) / "checkpoints" / "shared_metadynamics_mpi.npz")
    write_json(Path(output) / "summary.json", summary)
    return result
