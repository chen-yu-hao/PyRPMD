"""Restartable IESH, centroid RP-IESH and Runeson--Manolopoulos MASH ensembles.

The IESH one-particle density has trace equal to the number of electrons.
Multistate MASH uses its signed observable weights, with unit trace. Neither
quantity is silently replaced by an active-surface histogram. Fixed-diabatic
coefficients, all random streams and committed observations share one atomic
checkpoint; no adiabatic gauge needs to be reconstructed from a previous run.
"""
from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path

import numpy as np

from ..checkpoint import CheckpointError, atomic_savez, load_checkpoint
from ..io import atomic_write_text, fingerprint, jsonable, write_json
from ..nonadiabatic_registry import initialize_state, make_dynamics
from ..parallel import execution_plan, run_tasks
from ..propagation import scientific_config
from ..system import System
from .nonadiabatic import _streams


_KIND = "PyRPMD.multistate_nonadiabatic_trajectory.v1"
_RESULT_KIND = "PyRPMD.multistate_nonadiabatic_ensemble.v1"
_IESH_METHODS = frozenset(("iesh", "rp-iesh"))
_METHODS = _IESH_METHODS | {"multistate-mash"}
_ROLES = ("nuclear", "electronic", "dynamics")
_HISTORY = ("steps", "density", "population", "adiabatic_population", "active",
            "energy", "norm", "centroid_q", "centroid_p")
_RP_DIAGNOSTICS = ("hop_energy_change", "hop_energy_change_abs")


def _history_names(method):
    return _HISTORY + _RP_DIAGNOSTICS if method == "rp-iesh" else _HISTORY


def _driver(config, system, rng):
    if (config["method"] not in _METHODS
            or config["method"] != "rp-iesh" and config["beads"] != 1):
        raise ValueError("this workflow requires P=1 IESH/multistate-mash or centroid RP-IESH")
    driver = make_dynamics(config, system, rng)
    if driver.rng is not rng:
        driver.potential.close()
        raise ValueError("dynamics factory must preserve the supplied trajectory RNG")
    return driver


def _state_names(method):
    if method == "rp-iesh":
        return ("coefficients", "occupied") + _RP_DIAGNOSTICS
    return ("coefficients", "occupied") if method in _IESH_METHODS else (
        "coefficients", "active_surface", "sampling")


def _electron_count(config):
    return config["dynamics"]["nelectrons"] if config["method"] in _IESH_METHODS else 1


def _estimator_metadata(config):
    iesh = config["method"] in _IESH_METHODS
    result = {
        "population_basis": "fixed_diabatic", "density_basis": "fixed_diabatic",
        "population_estimator": "slater_one_particle_density" if iesh else "runeson_manolopoulos_2023",
        "population_normalization": _electron_count(config),
        "adiabatic_population_basis": "adiabatic",
        "adiabatic_population_estimator": "slater_one_particle_density" if iesh else "runeson_manolopoulos_2023",
        "active_basis": "adiabatic_orbitals" if iesh else "adiabatic_surfaces",
        "coherence_available": True,
        "electronic_initial_sampling": "specified_occupied_determinant" if iesh else "largest_population_uniform_complex_sphere",
        "electronic_backend": "numpy", "electronic_device": "cpu",
    }
    if config["method"] == "rp-iesh":
        result.update(
            method_variant="gardner-2023-centroid", method_status="research",
            electronic_geometry="centroid", adiabatic_population_geometry="centroid",
            energy_estimator="ring_polymer_hamiltonian_sum",
            energy_conservation="centroid_hop_balance_only",
            hop_rescaling="centroid_gap_uniform_bead_momentum_shift",
            frustrated_hops="keep",
        )
    return result


def _observe(dynamics, state, method):
    dynamics._check(state)
    density = np.asarray(dynamics.density_matrix(state, basis="diabatic"), dtype=np.complex128)
    if method in _IESH_METHODS:
        active = np.asarray(state.occupied, dtype=np.int64)
        adiabatic = dynamics.populations(state)
        norm = np.sum(abs(state.coefficients)**2, axis=0)
    else:
        active = np.asarray(state.active_surface, dtype=np.int64)
        adiabatic = dynamics.populations(state, basis="adiabatic")
        norm = np.atleast_1d(np.vdot(state.coefficients, state.coefficients).real)
    record = {
        "density": density, "population": density.diagonal().real.copy(),
        "adiabatic_population": np.asarray(adiabatic, dtype=np.float64), "active": active,
        "energy": np.asarray(dynamics.energy(state), dtype=np.float64),
        "norm": np.asarray(norm, dtype=np.float64),
        "centroid_q": state.q.mean(axis=0), "centroid_p": state.p.mean(axis=0),
    }
    if method == "rp-iesh":
        record.update({key: np.asarray(getattr(state, key), dtype=np.float64)
                       for key in _RP_DIAGNOSTICS})
    if not all(np.isfinite(value).all() for value in record.values()):
        raise FloatingPointError("nonfinite multistate nonadiabatic observation")
    return record


def _append(history, dynamics, state, method):
    record = _observe(dynamics, state, method)
    history["steps"].append(np.int64(state.step))
    for key, value in record.items():
        history[key].append(np.array(value, copy=True))


def _save(path, config, identity, index, state, streams, history):
    metadata = {
        "format": _KIND, "physics_fingerprint": identity, "trajectory": int(index),
        "method": config["method"], "state_class": type(state).__name__,
        "rngs": {role: copy.deepcopy(rng.bit_generator.state) for role, rng in zip(_ROLES, streams)},
    }
    arrays = {
        "metadata": np.asarray(json.dumps(jsonable(metadata), sort_keys=True, allow_nan=False)),
        "q": state.q, "p": state.p, "step": np.asarray(state.step, dtype=np.int64),
        "time": np.asarray(state.time, dtype=np.float64),
        **{"history_"+key: np.asarray(value) for key, value in history.items()},
    }
    for key in _state_names(config["method"]):
        dtype = (np.int64 if key in ("occupied", "active_surface")
                 else np.float64 if key in _RP_DIAGNOSTICS else None)
        arrays[key] = np.asarray(getattr(state, key), dtype=dtype)
    atomic_savez(path, kind=_KIND, **arrays)


def _typed(value, shape, dtype, label):
    if value.shape != shape or value.dtype != np.dtype(dtype) or not np.isfinite(value).all():
        raise CheckpointError(f"invalid checkpoint {label} shape, dtype or finite values")


def _validate_density_history(data, method, states, electrons):
    rho, population = data["history_density"], data["history_population"]
    if (not np.allclose(rho, rho.conj().swapaxes(-1, -2), rtol=0., atol=1e-8)
            or not np.allclose(np.trace(rho, axis1=-2, axis2=-1), electrons, rtol=0., atol=1e-8)
            or not np.array_equal(population, rho.diagonal(axis1=-2, axis2=-1).real)):
        raise CheckpointError("checkpoint density/population must be Hermitian with the method's trace")
    adiabatic = data["history_adiabatic_population"]
    if not np.allclose(adiabatic.sum(axis=-1), electrons, rtol=0., atol=1e-8):
        raise CheckpointError("checkpoint adiabatic population has incorrect normalization")
    if method in _IESH_METHODS:
        if (not np.allclose(rho@rho, rho, rtol=0., atol=1e-8)
                or np.any(adiabatic < -1e-8) or np.any(adiabatic > 1.+1e-8)):
            raise CheckpointError("checkpoint IESH density must be a Slater projector")
    else:
        alpha = (states-1)/(sum(1./k for k in range(1, states+1))-1.)
        lower = (1.-alpha)/states
        shifted = rho-lower*np.eye(states)
        if (not np.allclose(shifted@shifted, alpha*shifted, rtol=0., atol=1e-8)
                or np.any(adiabatic < lower-1e-8) or np.any(adiabatic > lower+alpha+1e-8)):
            raise CheckpointError("checkpoint MASH density violates its signed estimator weights")


def _load(path, config, identity, index, dynamics, streams):
    method = config["method"]
    with load_checkpoint(path, expected_kind=_KIND, require_integrity=True) as saved:
        data = {key: saved[key] for key in saved.files}
    inventory = ({"metadata", "q", "p", "step", "time"} | set(_state_names(method))
                 | {"history_"+key for key in _history_names(method)})
    if set(data) != inventory:
        raise CheckpointError("multistate nonadiabatic checkpoint payload inventory mismatch")
    raw = data["metadata"]
    if raw.shape != () or raw.dtype.kind != "U":
        raise CheckpointError("invalid checkpoint metadata dtype")
    try:
        metadata = json.loads(str(raw.item()))
    except (TypeError, ValueError) as error:
        raise CheckpointError("invalid checkpoint metadata JSON") from error
    class_name = {"iesh": "IESHState", "rp-iesh": "RPIESHState",
                  "multistate-mash": "MultistateMASHState"}[method]
    if (not isinstance(metadata, dict)
            or set(metadata) != {"format", "physics_fingerprint", "trajectory", "method", "state_class", "rngs"}
            or metadata["format"] != _KIND or metadata["physics_fingerprint"] != identity
            or type(metadata["trajectory"]) is not int or metadata["trajectory"] != index
            or metadata["method"] != method or metadata["state_class"] != class_name):
        raise CheckpointError("multistate nonadiabatic checkpoint identity mismatch")
    _typed(data["step"], (), "int64", "step")
    _typed(data["time"], (), "float64", "time")
    step, time = int(data["step"]), float(data["time"])
    target, stride = config["ensemble"]["steps"], config["ensemble"]["sample_stride"]
    if not 0 <= step <= target or not np.isclose(time, step*config["dt_au"], rtol=2e-11, atol=1e-12):
        raise CheckpointError("invalid checkpoint clock")
    atoms, states, electrons = len(dynamics.masses), dynamics.nstates, _electron_count(config)
    for key in ("q", "p"):
        _typed(data[key], (config["beads"], atoms, 3), "float64", key)
    _typed(data["coefficients"], (states, electrons) if method in _IESH_METHODS else (states,),
           "complex128", "coefficients")
    if method in _IESH_METHODS:
        _typed(data["occupied"], (electrons,), "int64", "occupied")
    else:
        _typed(data["active_surface"], (), "int64", "active_surface")
        if (data["sampling"].shape != () or data["sampling"].dtype.kind != "U"
                or str(data["sampling"].item()) != "population"):
            raise CheckpointError("checkpoint multistate MASH sampling must be population")
    try:
        common = (data["q"], data["p"], step, time, data["coefficients"])
        if method == "rp-iesh":
            from ..rp_iesh import RPIESHState
            for key in _RP_DIAGNOSTICS:
                _typed(data[key], (), "float64", key)
            state = RPIESHState(*common, data["occupied"],
                                **{key: float(data[key]) for key in _RP_DIAGNOSTICS})
        elif method == "iesh":
            from ..iesh import IESHState
            state = IESHState(*common, data["occupied"])
        else:
            from ..mash import MultistateMASHState
            state = MultistateMASHState(*common, int(data["active_surface"]), str(data["sampling"].item()))
        dynamics._check(state)
    except Exception as error:
        raise CheckpointError("invalid multistate nonadiabatic checkpoint state") from error
    expected_steps = np.arange(0, step+1, stride, dtype=np.int64)
    if step == target and target % stride:
        expected_steps = np.append(expected_steps, np.int64(target))
    frames = len(expected_steps)
    shapes = {
        "steps": (frames,), "density": (frames, states, states), "population": (frames, states),
        "adiabatic_population": (frames, states),
        "active": (frames, electrons) if method in _IESH_METHODS else (frames,),
        "energy": (frames,), "norm": (frames, electrons),
        "centroid_q": (frames, atoms, 3), "centroid_p": (frames, atoms, 3),
    }
    if method == "rp-iesh":
        shapes.update({key: (frames,) for key in _RP_DIAGNOSTICS})
    history = {}
    for key, shape in shapes.items():
        value = data["history_"+key]
        dtype = "int64" if key in ("steps", "active") else "complex128" if key == "density" else "float64"
        _typed(value, shape, dtype, "history_"+key)
        history[key] = [row.copy() for row in value]
    if not np.array_equal(data["history_steps"], expected_steps):
        raise CheckpointError("checkpoint observation schedule mismatch")
    if not np.allclose(data["history_norm"], 1., rtol=0., atol=1e-8):
        raise CheckpointError("checkpoint electronic norm history is invalid")
    _validate_density_history(data, method, states, electrons)
    active = data["history_active"]
    if np.any(active < 0) or np.any(active >= states):
        raise CheckpointError("checkpoint active electronic state is out of range")
    if method in _IESH_METHODS and any(len(np.unique(row)) != electrons for row in active):
        raise CheckpointError("checkpoint IESH active occupations violate Pauli exclusion")
    if method == "rp-iesh":
        # Include the unsampled terminal state in the cumulative-history check.
        signed = np.append(data["history_hop_energy_change"], state.hop_energy_change)
        absolute = np.append(data["history_hop_energy_change_abs"], state.hop_energy_change_abs)
        tolerance = 32*np.finfo(float).eps*np.maximum(1., absolute)
        if (np.any(absolute < 0.) or np.any(np.diff(absolute) < 0.)
                or np.any(np.abs(signed) > absolute+tolerance)
                or np.any(np.abs(np.diff(signed)) > np.diff(absolute)+tolerance[1:])):
            raise CheckpointError("checkpoint cumulative RP-IESH hop energy history is inconsistent")
    if method == "multistate-mash":
        adiabatic = data["history_adiabatic_population"]
        selected = np.take_along_axis(adiabatic, active[:, None], axis=1)[:, 0]
        if np.any(selected < adiabatic.max(axis=1)-1e-8):
            raise CheckpointError("checkpoint MASH active surface and observable populations disagree")
    # Re-create the declared initial ensemble on detached streams.  A resealed
    # checkpoint cannot silently substitute a different preparation while
    # retaining an otherwise internally consistent state/history payload.
    initial_streams = _streams(config, index)
    try:
        initial_state = initialize_state(config, System.from_dict(config["system"]),
                                         dynamics, *initial_streams[:2])
        initial_record = _observe(dynamics, initial_state, method)
    except Exception as error:
        raise CheckpointError("checkpoint initial preparation cannot be re-evaluated") from error
    for key, value in initial_record.items():
        previous = data["history_"+key][0]
        equal = (np.array_equal(value, previous) if key == "active" or key in _RP_DIAGNOSTICS
                 else np.allclose(value, previous, rtol=2e-11, atol=1e-8))
        if not equal:
            raise CheckpointError(f"checkpoint initial preparation mismatch for {key}")
    if int(expected_steps[-1]) == state.step:
        try:
            observed = _observe(dynamics, state, method)
        except Exception as error:
            raise CheckpointError("checkpoint final observation cannot be re-evaluated") from error
        for key, value in observed.items():
            previous = data["history_"+key][-1]
            equal = (np.array_equal(value, previous) if key == "active" or key in _RP_DIAGNOSTICS
                     else np.allclose(value, previous, rtol=2e-11, atol=1e-8))
            if not equal:
                raise CheckpointError(f"checkpoint state/final observation mismatch for {key}")
    # Restore only after validating the entire checkpoint, including every RNG.
    if not isinstance(metadata["rngs"], dict) or set(metadata["rngs"]) != set(_ROLES):
        raise CheckpointError("checkpoint random stream inventory mismatch")
    restored = []
    try:
        for role in _ROLES:
            temporary = np.random.default_rng()
            temporary.bit_generator.state = metadata["rngs"][role]
            restored.append(copy.deepcopy(temporary.bit_generator.state))
    except (TypeError, ValueError, KeyError) as error:
        raise CheckpointError("invalid checkpoint random generator state") from error
    for role, initial_rng, value in zip(_ROLES, initial_streams, restored):
        # Nuclear/electronic RNGs are used only by initialization; MASH's
        # propagation is deterministic and never consumes the dynamics stream.
        if role != "dynamics" or method == "multistate-mash" or step == 0:
            if value != initial_rng.bit_generator.state:
                raise CheckpointError(f"checkpoint {role} random stream differs from its declared preparation")
    for rng, value in zip(streams, restored):
        rng.bit_generator.state = value
    return state, history


def _trajectory(task):
    config, system_data, directory, index = task
    system = System.from_dict(system_data)
    streams = _streams(config, index)
    dynamics = state = None
    path = Path(directory)/"checkpoints"/f"nonadiabatic_{index:06d}.npz"
    failure = Path(directory)/"logs"/f"nonadiabatic_{index:06d}_failure.json"
    try:
        dynamics = _driver(config, system, streams[2])
        identity = fingerprint(scientific_config(config), system)
        if path.exists():
            if not config.get("resume", False):
                raise ValueError("existing trajectory requires resume")
            state, history = _load(path, config, identity, index, dynamics, streams)
        else:
            state = initialize_state(config, system, dynamics, *streams[:2])
            history = {key: [] for key in _history_names(config["method"])}
            _append(history, dynamics, state, config["method"])
            _save(path, config, identity, index, state, streams, history)
        target = config["ensemble"]["steps"]
        stride, block = config["ensemble"]["sample_stride"], config["dynamics"]["block_steps"]
        while state.step < target:
            dynamics.step(state)
            if state.step % stride == 0 or state.step == target:
                _append(history, dynamics, state, config["method"])
            if state.step % block == 0 or state.step == target:
                _save(path, config, identity, index, state, streams, history)
        if failure.exists():
            failure.unlink()
        return str(path)
    except BaseException as error:
        write_json(failure, {"trajectory": int(index), "last_in_memory_step": None if state is None else int(state.step),
                             "error_type": type(error).__name__, "error": str(error), "checkpoint": str(path),
                             "resume_from_last_complete_checkpoint": True})
        raise
    finally:
        if dynamics is not None:
            dynamics.potential.close()


def _sem(values):
    return None if len(values) < 2 else np.std(values, axis=0, ddof=1)/np.sqrt(len(values))


def _write_results(directory, config, identity, histories):
    stack = lambda key: np.asarray([value[key] for value in histories])
    steps = np.asarray(histories[0]["steps"], dtype=np.int64)
    density, population, energy, norm = stack("density"), stack("population"), stack("energy"), stack("norm")
    active = stack("active")
    indicators = np.eye(population.shape[-1])[active]
    if config["method"] in _IESH_METHODS:
        indicators = indicators.sum(axis=-2)
    reason = "fewer_than_two_trajectories" if len(histories) < 2 else None
    metadata = {
        "format": _RESULT_KIND, "method": config["method"], "physics_fingerprint": identity,
        "independent_trajectories": True, "statistics_axis": "trajectory", "time_unit": "atomic",
        "energy_unit": "hartree", "physical_converged": False, "sem_unavailable_reason": reason,
        **_estimator_metadata(config),
    }
    arrays = {
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
        "time": steps*config["dt_au"], "steps": steps,
        "populations": population, "population_mean": population.mean(axis=0),
        "diabatic_density": density, "density_mean": density.mean(axis=0),
        "adiabatic_populations": stack("adiabatic_population"),
        "adiabatic_population_mean": stack("adiabatic_population").mean(axis=0),
        "active_electronic_states": active, "active_population_mean": indicators.mean(axis=0),
        "active_population_counts": indicators.sum(axis=0).astype(np.int64),
        "energy": energy, "energy_drift": energy-energy[:, :1],
        "electronic_norm": norm, "norm_drift": norm-norm[:, :1],
        "centroid_q": stack("centroid_q"), "centroid_p": stack("centroid_p"),
    }
    if config["method"] == "rp-iesh":
        arrays.update({key: stack(key) for key in _RP_DIAGNOSTICS})
        cumulative = arrays["hop_energy_change"]
        arrays["energy_drift_minus_hop_changes"] = arrays["energy_drift"]-(cumulative-cumulative[:, :1])
    if len(histories) > 1:
        arrays.update(population_sem=_sem(population), density_sem_real=_sem(density.real),
                      density_sem_imag=_sem(density.imag), active_population_sem=_sem(indicators),
                      adiabatic_population_sem=_sem(arrays["adiabatic_populations"]))
    atomic_savez(directory/"results"/"nonadiabatic.npz", kind=_RESULT_KIND, **arrays)
    text = io.StringIO()
    writer = csv.writer(text)
    fields = [("population", arrays["population_mean"]),
              ("adiabatic_population", arrays["adiabatic_population_mean"]),
              ("active_population", arrays["active_population_mean"])]
    if "population_sem" in arrays:
        fields.append(("population_sem", arrays["population_sem"]))
    writer.writerow(["time_au"]+[f"{name}_{i}" for name, _ in fields for i in range(population.shape[-1])])
    for frame, time in enumerate(arrays["time"]):
        writer.writerow([time]+[float(values[frame, i]) for _, values in fields for i in range(population.shape[-1])])
    atomic_write_text(directory/"results"/"populations.csv", text.getvalue())
    return arrays, metadata


def prepare_multistate_nonadiabatic(config, system, *, numerical=False):
    streams = _streams(config, 0)
    dynamics = _driver(config, system, streams[2])
    try:
        state = initialize_state(config, system, dynamics, *streams[:2])
        record = _observe(dynamics, state, config["method"])
        result = {
            "method": config["method"], "nstates": dynamics.nstates, "model": config["model"],
            "initialization": config["initial"]["kind"], "state_shape": list(state.q.shape),
            "electronic_state_class": type(state).__name__, "integrator": config["dynamics"]["integrator"],
            "initial_energy_hartree": float(record["energy"]), "physical_converged": False,
            **_estimator_metadata(config),
        }
        if numerical:
            from ..potentials.diabatic import check_diabatic_potential
            result["matrix_gradient_check"] = check_diabatic_potential(dynamics.potential, state.q, dynamics.nstates)
        return result
    finally:
        dynamics.potential.close()


def run_multistate_nonadiabatic(config, system, directory):
    from . import validate_bond_constraint_workflow
    validate_bond_constraint_workflow(system, "nonadiabatic")
    plan = execution_plan(config["parallel"])
    if plan.backend not in ("serial", "process") or plan.mode != "serial" or plan.bead_workers != 1:
        raise ValueError("multistate ensembles support serial or trajectory-process execution")
    if config["dynamics"].get("loop_backend", "auto") not in ("auto", "python"):
        raise ValueError("multistate methods require loop_backend='auto' or 'python'")
    tasks = [(config, system.to_dict(), str(directory), index) for index in range(config["ensemble"]["trajectories"])]
    run_tasks(_trajectory, tasks, config["parallel"])
    return analyze_multistate_nonadiabatic(config, system, directory)


def analyze_multistate_nonadiabatic(config, system, directory):
    from . import validate_bond_constraint_workflow
    validate_bond_constraint_workflow(system, "nonadiabatic")
    directory = Path(directory)
    identity = fingerprint(scientific_config(config), system)
    histories = []
    for index in range(config["ensemble"]["trajectories"]):
        streams = _streams(config, index)
        dynamics = _driver(config, system, streams[2])
        try:
            state, history = _load(directory/"checkpoints"/f"nonadiabatic_{index:06d}.npz", config, identity,
                                   index, dynamics, streams)
            if state.step != config["ensemble"]["steps"]:
                raise CheckpointError(f"trajectory {index} is incomplete; resume before ensemble reduction")
            histories.append(history)
        finally:
            dynamics.potential.close()
    arrays, metadata = _write_results(directory, config, identity, histories)
    result = {
        "task": "nonadiabatic", "method": config["method"], "model": config["model"]["model"],
        "trajectories": len(histories), "steps": config["ensemble"]["steps"], "frames": len(arrays["steps"]),
        **_estimator_metadata(config), "estimator": metadata["population_estimator"],
        "final_populations": arrays["population_mean"][-1].tolist(),
        "final_adiabatic_populations": arrays["adiabatic_population_mean"][-1].tolist(),
        "final_active_populations": arrays["active_population_mean"][-1].tolist(),
        "final_population_sem": None if "population_sem" not in arrays else arrays["population_sem"][-1].tolist(),
        "sem_unavailable_reason": metadata["sem_unavailable_reason"],
        "uncertainty": "pointwise SEM across independent trajectories; excludes systematic method and discretization errors",
        "max_abs_energy_drift_at_saved_frames_hartree": float(np.max(np.abs(arrays["energy_drift"]))),
        "max_electronic_norm_drift_at_saved_frames": float(np.max(np.abs(arrays["norm_drift"]))),
        "physical_converged": False, "completed": True, "initialization": config["initial"],
        "physics_fingerprint": identity, "arrays": str(directory/"results"/"nonadiabatic.npz"),
        "time_series": str(directory/"results"/"populations.csv"),
    }
    if config["method"] == "rp-iesh":
        result.update(
            max_abs_cumulative_hop_energy_change_at_saved_frames_hartree=float(np.max(np.abs(arrays["hop_energy_change"]))),
            max_cumulative_abs_hop_energy_change_at_saved_frames_hartree=float(np.max(arrays["hop_energy_change_abs"])),
            max_abs_energy_drift_minus_hop_changes_at_saved_frames_hartree=float(
                np.max(np.abs(arrays["energy_drift_minus_hop_changes"]))),
        )
    return result
