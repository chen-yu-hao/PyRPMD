"""Restartable, unweighted population dynamics on explicit diabatic models.

One integrity-checked file owns each trajectory's electronic/nuclear state,
random generator and observations. A failed step never removes a trajectory
from the statistical denominator. This workflow does not estimate thermal
reaction rates or accept signed thermal SEO parents.
"""
from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path

import numpy as np

from ..checkpoint import atomic_savez, load_checkpoint, CheckpointError
from ..dynamics import State
from ..io import atomic_write_text, fingerprint, jsonable, write_json
from ..nonadiabatic import NRPMDState, NRPMDDynamics
from ..nonadiabatic_ensemble import density_matrix, NonadiabaticEnsembleResult
from ..parallel import execution_plan, run_tasks
from ..nonadiabatic_registry import potential_from_identity as model_from_dict, make_dynamics
from ..propagation import scientific_config
from ..surface_hopping import RPSHState, RPSHDynamics
from ..system import System


_KIND = "PyRPMD.nonadiabatic_trajectory.v1"
_HISTORY = ("density", "energy", "norm", "centroid_q", "centroid_p")


def _driver(config, system, potential, rng):
    return make_dynamics(config, system, rng, potential=potential)


def _streams(config, index):
    # Stable logical IDs, independent of worker placement and restart chunks.
    return tuple(np.random.default_rng(np.random.SeedSequence(
        config["seed"], spawn_key=(int(index), role))) for role in range(3))


def _initialize_state(config, system, dynamics, nuclear_rng, electronic_rng):
    initial = config["initial"]
    if initial["kind"] == "point":
        q = np.broadcast_to(system.geometry, (config["beads"], system.natoms, 3))
        p = np.broadcast_to(initial["momenta"], q.shape)
        nuclear = State(q, p)
    else:
        from ..nonadiabatic_sampling import sample_harmonic_ring
        nuclear = sample_harmonic_ring(
            system.masses, config["beta"], config["beads"], initial["frequencies"],
            coordinates=initial["coordinates"], centers=initial["centers"],
            centroid_momenta=initial["centroid_momenta"], count=1, seed=nuclear_rng)[0]
    if config["method"] == "nrpmd":
        return NRPMDState.from_nuclear(nuclear, dynamics.nstates,
                                       initial["initial_state"], electronic_rng)
    return dynamics.initialize(nuclear, active_surface=initial["active_surface"])


def _observe(dynamics, state):
    record = {
        "density": density_matrix(state), "energy": dynamics.energy(state),
        "norm": (np.sum(state.mapping_q**2 + state.mapping_p**2, axis=1)
                 if isinstance(state, NRPMDState)
                 else np.array([np.vdot(state.coefficients, state.coefficients).real])),
        "centroid_q": state.q.mean(axis=0), "centroid_p": state.p.mean(axis=0),
    }
    if isinstance(state, RPSHState):
        record.update(amplitude=dynamics.populations(state), active=state.active_surface)
    if not all(np.isfinite(value).all() for value in record.values()):
        raise FloatingPointError("nonfinite nonadiabatic observation")
    return record


def _append(history, dynamics, state):
    # Evaluate all observables before committing any of this frame.
    record = _observe(dynamics, state)
    history["steps"].append(int(state.step))
    for name, value in record.items():
        history[name].append(np.array(value, copy=True))


def _save(path, config, identity, index, state, rng, history):
    metadata = {
        "format": _KIND, "physics_fingerprint": identity,
        "trajectory": int(index), "method": config["method"],
        "rng": copy.deepcopy(rng.bit_generator.state),
    }
    arrays = {
        "metadata": np.asarray(json.dumps(jsonable(metadata), sort_keys=True, allow_nan=False)),
        "q": state.q, "p": state.p, "step": np.asarray(state.step),
        "time": np.asarray(state.time),
        **{"history_" + key: np.asarray(value) for key, value in history.items()},
    }
    if isinstance(state, NRPMDState):
        arrays.update(mapping_q=state.mapping_q, mapping_p=state.mapping_p)
    else:
        arrays.update(coefficients=state.coefficients,
                      active_surface=np.asarray(state.active_surface))
    atomic_savez(path, kind=_KIND, **arrays)


def _load(path, config, identity, index, dynamics, rng):
    with load_checkpoint(path, expected_kind=_KIND, require_integrity=True) as saved:
        data = {key: saved[key] for key in saved.files}
    history_names = {"steps", *_HISTORY}
    state_names = {"mapping_q", "mapping_p"}
    if config["method"] == "rpsh":
        history_names.update(("amplitude", "active"))
        state_names = {"coefficients", "active_surface"}
    inventory = ({"metadata", "q", "p", "step", "time"} | state_names
                 | {"history_" + key for key in history_names})
    if set(data) != inventory:
        raise CheckpointError("nonadiabatic checkpoint payload inventory mismatch")
    raw_metadata = data.pop("metadata")
    if raw_metadata.shape != () or raw_metadata.dtype.kind != "U":
        raise CheckpointError("invalid nonadiabatic checkpoint metadata dtype")
    metadata = json.loads(str(raw_metadata.item()))
    if (set(metadata) != {"format", "physics_fingerprint", "trajectory", "method", "rng"}
            or metadata["format"] != _KIND or metadata["physics_fingerprint"] != identity
            or type(metadata["trajectory"]) is not int or metadata["trajectory"] != index
            or metadata["method"] != config["method"]):
        raise CheckpointError("nonadiabatic checkpoint identity mismatch")
    step = data.pop("step")
    time = data.pop("time")
    target, stride = config["ensemble"]["steps"], config["ensemble"]["sample_stride"]
    if (step.shape != () or step.dtype.kind not in "iu" or not 0 <= int(step) <= target
            or time.shape != () or time.dtype.kind != "f" or time.dtype.itemsize != 8
            or not np.isfinite(time)
            or not np.isclose(float(time), int(step)*config["dt_au"], rtol=2e-11, atol=1e-12)):
        raise CheckpointError("invalid nonadiabatic checkpoint clock")
    real_state = ("q", "p", "mapping_q", "mapping_p") if config["method"] == "nrpmd" else ("q", "p")
    if any(data[key].dtype.kind != "f" or data[key].dtype.itemsize != 8 for key in real_state):
        raise CheckpointError("nonadiabatic checkpoint nuclear/mapping arrays must be float64")
    if config["method"] == "rpsh" and (
            data["coefficients"].dtype.kind != "c" or data["coefficients"].dtype.itemsize != 16):
        raise CheckpointError("nonadiabatic checkpoint coefficients must be complex128")
    common = (data.pop("q"), data.pop("p"), int(step), float(time))
    try:
        if config["method"] == "nrpmd":
            state = NRPMDState(*common, data.pop("mapping_q"), data.pop("mapping_p"))
        else:
            active = data.pop("active_surface")
            if active.shape != () or active.dtype.kind not in "iu":
                raise CheckpointError("invalid active_surface checkpoint")
            state = RPSHState(*common, data.pop("coefficients"), int(active))
        dynamics._check(state)
    except (ValueError, TypeError, OverflowError) as error:
        raise CheckpointError("invalid nonadiabatic checkpoint state") from error
    # This task has focused NRPMD initialization, unlike the separate thermal
    # API with fluctuating mapping norms. Preserve its defining norm invariant.
    if config["method"] == "nrpmd" and not np.allclose(
            np.sum(state.mapping_q**2+state.mapping_p**2, axis=1),
            dynamics.nstates+2., rtol=0., atol=1e-8):
        raise CheckpointError("checkpoint focused NRPMD mapping norm is invalid")
    expected_steps = np.arange(0, int(step)+1, stride, dtype=int)
    if int(step) == target and target % stride:
        expected_steps = np.append(expected_steps, target)
    nframes, nstates = len(expected_steps), dynamics.nstates
    expected_shapes = {
        "steps": (nframes,), "density": (nframes, nstates, nstates),
        "energy": (nframes,),
        "norm": (nframes, config["beads"] if config["method"] == "nrpmd" else 1),
        "centroid_q": (nframes, len(dynamics.masses), 3),
        "centroid_p": (nframes, len(dynamics.masses), 3),
    }
    if config["method"] == "rpsh":
        expected_shapes.update(amplitude=(nframes, nstates), active=(nframes,))
    if set(data) != {"history_" + key for key in expected_shapes}:
        raise CheckpointError("nonadiabatic checkpoint history inventory mismatch")
    history = {}
    for key, shape in expected_shapes.items():
        value = data["history_" + key]
        if key in ("steps", "active"):
            valid_dtype = value.dtype.kind in "iu"
        elif key == "density":
            valid_dtype = value.dtype.kind == "c" and value.dtype.itemsize == 16
        else:
            valid_dtype = value.dtype.kind == "f" and value.dtype.itemsize == 8
        if not valid_dtype:
            raise CheckpointError(f"invalid checkpoint history {key} dtype")
        if value.shape != shape or not np.isfinite(value).all():
            raise CheckpointError(f"invalid checkpoint history {key}")
        history[key] = [row.copy() for row in value]
    if not np.array_equal(data["history_steps"], expected_steps):
        raise CheckpointError("checkpoint observation schedule mismatch")
    rho = data["history_density"]
    if not np.allclose(rho, rho.swapaxes(-1, -2).conj(), rtol=0, atol=1e-8):
        raise CheckpointError("checkpoint density estimator is not Hermitian")
    if not np.allclose(np.trace(rho, axis1=-2, axis2=-1), 1., rtol=0., atol=1e-8):
        raise CheckpointError("checkpoint density estimator must have unit trace")
    expected_norm = nstates+2. if config["method"] == "nrpmd" else 1.
    if not np.allclose(data["history_norm"], expected_norm, rtol=0., atol=1e-8):
        raise CheckpointError("checkpoint electronic norm history is invalid")
    if config["method"] == "rpsh":
        if np.any(data["history_active"] < 0) or np.any(data["history_active"] >= nstates):
            raise CheckpointError("checkpoint active surface is out of range")
        amplitude = data["history_amplitude"]
        if np.any(amplitude < 0) or not np.allclose(amplitude.sum(axis=-1), 1., rtol=0., atol=1e-8):
            raise CheckpointError("checkpoint adiabatic amplitude populations are invalid")
        if not np.allclose(rho@rho, rho, rtol=0., atol=1e-8):
            raise CheckpointError("checkpoint RPSH density must be a pure amplitude density")
    # At observation checkpoints (including every completed trajectory), the
    # state and the final saved frame must describe the same physical instant.
    # Between observation times, comparing centroids would be incorrect.
    if int(expected_steps[-1]) == state.step:
        try:
            current = _observe(dynamics, state)
        except Exception as error:
            raise CheckpointError("checkpoint final observation cannot be re-evaluated") from error
        for key, value in current.items():
            recorded = data["history_" + key][-1]
            agrees = (np.array_equal(value, recorded) if key == "active" else
                      np.allclose(value, recorded, rtol=2e-11, atol=1e-8))
            if not agrees:
                raise CheckpointError(f"checkpoint state/final observation mismatch for {key}")
    # Validate the saved generator on a detached object before changing ours.
    restored_rng = np.random.default_rng()
    try:
        restored_rng.bit_generator.state = metadata["rng"]
    except (ValueError, TypeError, KeyError) as error:
        raise CheckpointError("invalid nonadiabatic random generator checkpoint") from error
    rng.bit_generator.state = copy.deepcopy(restored_rng.bit_generator.state)
    return state, history


def _trajectory(task):
    config, system_data, directory, index = task
    system = System.from_dict(system_data)
    potential = model_from_dict(config["model"])
    nuclear_rng, electronic_rng, rng = _streams(config, index)
    dynamics = _driver(config, system, potential, rng)
    identity = fingerprint(scientific_config(config), system)
    path = Path(directory)/"checkpoints"/f"nonadiabatic_{index:06d}.npz"
    failed = Path(directory)/"logs"/f"nonadiabatic_{index:06d}_failure.json"
    state = None
    try:
        if path.exists():
            if not config.get("resume", False):
                raise ValueError("existing trajectory requires resume")
            state, history = _load(path, config, identity, index, dynamics, rng)
        else:
            state = _initialize_state(config, system, dynamics, nuclear_rng, electronic_rng)
            history = {key: [] for key in ("steps", *_HISTORY)}
            if config["method"] == "rpsh":
                history.update(amplitude=[], active=[])
            _append(history, dynamics, state)
            _save(path, config, identity, index, state, rng, history)
        target = config["ensemble"]["steps"]
        stride = config["ensemble"]["sample_stride"]
        block = config["dynamics"]["block_steps"]
        while state.step < target:
            dynamics.step(state)
            if state.step % stride == 0 or state.step == target:
                _append(history, dynamics, state)
            if state.step % block == 0 or state.step == target:
                _save(path, config, identity, index, state, rng, history)
        if failed.exists():
            failed.unlink()
        return str(path)
    except BaseException as error:
        # Preserve the previous complete checkpoint. Never publish a reduced
        # success ensemble or advance RNG without its corresponding state.
        write_json(failed, {"trajectory": index,
                           "last_in_memory_step": None if state is None else int(state.step),
                           "error_type": type(error).__name__, "error": str(error),
                           "checkpoint": str(path), "resume_from_last_complete_checkpoint": True})
        raise
    finally:
        potential.close()


def prepare_nonadiabatic(config, system, *, numerical=False):
    if config["method"] not in {"nrpmd", "rpsh"}:
        from .nonadiabatic_extensions import prepare_extended_nonadiabatic
        return prepare_extended_nonadiabatic(config, system, numerical=numerical)
    potential = model_from_dict(config["model"])
    streams = _streams(config, 0)
    try:
        dynamics = _driver(config, system, potential, streams[2])
        state = _initialize_state(config, system, dynamics, *streams[:2])
        checked = _observe(dynamics, state)
        result = {"method": config["method"], "nstates": potential.nstates,
                  "model": config["model"], "initialization": config["initial"]["kind"],
                  "state_shape": list(state.q.shape),
                  "initial_energy_hartree": checked["energy"],
                  "physical_converged": False}
        if numerical:
            from ..potentials.diabatic import check_diabatic_potential
            result["matrix_gradient_check"] = check_diabatic_potential(potential, state.q, potential.nstates)
        return result
    finally:
        potential.close()


def run_nonadiabatic(config, system, directory):
    from . import validate_bond_constraint_workflow
    validate_bond_constraint_workflow(system, "nonadiabatic")
    if config["method"] not in {"nrpmd", "rpsh"}:
        from .nonadiabatic_extensions import run_extended_nonadiabatic
        return run_extended_nonadiabatic(config, system, directory)
    plan = execution_plan(config["parallel"])
    if plan.backend not in ("serial", "process") or plan.mode != "serial" or plan.bead_workers != 1:
        raise ValueError("model nonadiabatic ensembles support serial or trajectory-process execution")
    if config["dynamics"].get("loop_backend", "auto") not in ("auto", "python"):
        raise ValueError("matrix dynamics requires loop_backend='auto' or 'python'")
    tasks = [(config, system.to_dict(), str(directory), i)
             for i in range(config["ensemble"]["trajectories"])]
    run_tasks(_trajectory, tasks, config["parallel"])
    return analyze_nonadiabatic(config, system, directory)


def analyze_nonadiabatic(config, system, directory):
    """Verify and reduce all completed trajectory checkpoints, without propagation."""
    from . import validate_bond_constraint_workflow
    validate_bond_constraint_workflow(system, "nonadiabatic")
    if config["method"] not in {"nrpmd", "rpsh"}:
        from .nonadiabatic_extensions import analyze_extended_nonadiabatic
        return analyze_extended_nonadiabatic(config, system, directory)
    directory = Path(directory)
    identity = fingerprint(scientific_config(config), system)
    histories, states, rng_states = [], [], []
    potential = model_from_dict(config["model"])
    try:
        for index in range(config["ensemble"]["trajectories"]):
            rng = _streams(config, index)[2]
            dynamics = _driver(config, system, potential, rng)
            state, history = _load(directory/"checkpoints"/f"nonadiabatic_{index:06d}.npz",
                                   config, identity, index, dynamics, rng)
            if state.step != config["ensemble"]["steps"]:
                raise CheckpointError(f"trajectory {index} is incomplete; resume before ensemble reduction")
            histories.append(history)
            states.append(state)
            rng_states.append({"dynamics": copy.deepcopy(rng.bit_generator.state)})
    finally:
        potential.close()
    steps = np.asarray(histories[0]["steps"], dtype=int)
    stack = lambda name: np.asarray([history[name] for history in histories])
    result = NonadiabaticEnsembleResult(
        method=config["method"], time=steps*config["dt_au"], sample_steps=steps,
        start_times=np.zeros(len(states)), start_steps=np.zeros(len(states), dtype=int),
        diabatic_density=stack("density"), energy=stack("energy"), electronic_norm=stack("norm"),
        final_states=tuple(states), rng_states=tuple(rng_states), independent=True,
        adiabatic_amplitude_populations=stack("amplitude") if config["method"] == "rpsh" else None,
        active_surfaces=stack("active").astype(int) if config["method"] == "rpsh" else None,
        metadata={"initialization": config["initial"]["kind"]})
    _write_results(directory, result, config, identity, stack("centroid_q"), stack("centroid_p"))
    return {
        "task": "nonadiabatic", "method": result.method,
        "model": config["model"]["model"], "trajectories": len(states),
        "steps": config["ensemble"]["steps"], "frames": len(steps),
        "population_basis": "fixed_diabatic", "estimator": (
            "signed_mmst" if result.method == "nrpmd" else "amplitude_density"),
        "final_populations": result.mean_populations[-1].tolist(),
        "final_population_sem": None if result.population_sem is None else result.population_sem[-1].tolist(),
        "sem_unavailable_reason": result.sem_unavailable_reason,
        "uncertainty": "pointwise SEM across independent trajectories; excludes systematic method and discretization errors",
        "max_abs_energy_drift_at_saved_frames_hartree": float(np.max(np.abs(result.energy_drift))),
        "max_electronic_norm_drift_at_saved_frames": float(np.max(np.abs(result.norm_drift))),
        "final_active_surface_fractions": (None if result.method == "nrpmd"
                                           else result.active_surface_fractions[-1].tolist()),
        "physical_converged": False, "completed": True,
        "initialization": config["initial"], "physics_fingerprint": identity,
        "arrays": str(directory/"results"/"nonadiabatic.npz"),
        "time_series": str(directory/"results"/"populations.csv"),
    }


def _write_results(directory, result, config, identity, centroid_q, centroid_p):
    metadata = {"format": "PyRPMD.nonadiabatic_ensemble.v1", "method": result.method,
                "physics_fingerprint": identity, "independent_trajectories": result.independent,
                "physical_converged": False, "population_basis": "fixed_diabatic",
                "density_basis": "fixed_diabatic", "statistics_axis": "trajectory",
                "population_estimator": ("signed_mmst" if result.method == "nrpmd"
                                         else "amplitude_density"),
                "time_unit": "atomic", "energy_unit": "hartree",
                "sem_unavailable_reason": result.sem_unavailable_reason}
    if result.method == "rpsh":
        metadata.update(amplitude_population_basis="centroid_adiabatic",
                        active_surface_basis="centroid_adiabatic")
    arrays = {"metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
              "time": result.time, "steps": result.sample_steps,
              "diabatic_density": result.diabatic_density,
              "population_mean": result.mean_populations,
              "density_mean": result.mean_density,
              "energy": result.energy, "electronic_norm": result.electronic_norm,
              "centroid_q": centroid_q, "centroid_p": centroid_p}
    for key, value in (("population_sem", result.population_sem),
                       ("density_sem_real", result.density_sem_real),
                       ("density_sem_imag", result.density_sem_imag),
                       ("adiabatic_amplitude_populations", result.adiabatic_amplitude_populations),
                       ("adiabatic_amplitude_population_mean", result.mean_adiabatic_amplitude_populations),
                       ("adiabatic_amplitude_population_sem", result.adiabatic_amplitude_population_sem),
                       ("active_surfaces", result.active_surfaces),
                       ("active_surface_counts", result.active_surface_counts),
                       ("active_surface_fractions", result.active_surface_fractions),
                       ("active_surface_sem", result.active_surface_sem)):
        if value is not None:
            arrays[key] = value
    atomic_savez(directory/"results"/"nonadiabatic.npz", kind=metadata["format"], **arrays)
    text = io.StringIO()
    writer = csv.writer(text)
    nstates = result.diabatic_density.shape[-1]
    header = ["time_au"] + [f"population_{i}" for i in range(nstates)]
    if result.population_sem is not None:
        header += [f"population_sem_{i}" for i in range(nstates)]
    if result.method == "rpsh":
        header += [f"active_surface_fraction_{i}" for i in range(nstates)]
        header += [f"adiabatic_amplitude_population_{i}" for i in range(nstates)]
    writer.writerow(header)
    for frame, time in enumerate(result.time):
        row = [time, *result.mean_populations[frame]]
        if result.population_sem is not None:
            row.extend(result.population_sem[frame])
        if result.method == "rpsh":
            row.extend(result.active_surface_fractions[frame])
            row.extend(result.mean_adiabatic_amplitude_populations[frame])
        writer.writerow(row)
    atomic_write_text(directory/"results"/"populations.csv", text.getvalue())
